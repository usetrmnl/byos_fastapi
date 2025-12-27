from typing import Dict, List

from fastapi.testclient import TestClient

from trmnl_server import main, models
from trmnl_server.services import state
from trmnl_server import utils
from io import BytesIO
from urllib.parse import urlparse


def _prime_rotation_master() -> Dict:
    master = state.global_state['rotation_master']
    master.clear()
    master.update({
        'bmp_entries': [b'A', b'B'],
        'png_entries': [b'A', b'B'],
        'hashes': ['hashA', 'hashB'],
        'meta': [
            {
                'id': 'hashA',
                'hash': 'hashA',
                'plugin': 'TestPlugin',
                'label': 'Entry A',
                'url_png': '/web/a.png',
                'url_bmp': '/web/a.bmp'
            },
            {
                'id': 'hashB',
                'hash': 'hashB',
                'plugin': 'TestPlugin',
                'label': 'Entry B',
                'url_png': '/web/b.png',
                'url_bmp': '/web/b.bmp'
            }
        ],
        'selected_ids': [],
        'version': 0
    })
    return master


def _prime_rotation_with_dummy_frames() -> Dict:
    """Populate rotation master with a real BMP and PNG so preview conversion succeeds."""
    master = state.global_state['rotation_master']
    master.clear()
    dummy_path = utils.asset_path('img', 'dummy.bmp')
    with open(dummy_path, 'rb') as handle:
        bmp_bytes = handle.read()
    png_bytes = utils.convert_bmp_bytes_to_png(BytesIO(bmp_bytes)).getvalue()
    master.update({
        'bmp_entries': [bmp_bytes],
        'png_entries': [png_bytes],
        'hashes': ['hashA'],
        'meta': [
            {
                'id': 'hashA',
                'hash': 'hashA',
                'plugin': 'TestPlugin',
                'label': 'Entry A',
                'url_png': '/web/a.png',
                'url_bmp': '/web/a.bmp'
            }
        ],
        'selected_ids': ['hashA'],
        'version': 0
    })
    return master


def _prime_rotation_with_two_frames() -> Dict:
    """Populate rotation with two distinct frames so we can verify preview updates."""
    master = state.global_state['rotation_master']
    master.clear()
    dummy_path = utils.asset_path('img', 'dummy.bmp')
    with open(dummy_path, 'rb') as handle:
        bmp_bytes1 = handle.read()
    # Create a second BMP by toggling one pixel via PIL to ensure valid encoding
    img = utils.load_image(str(dummy_path))
    img = utils.ensure_image_mode(img, '1')
    img_copy = img.copy()
    img_copy.putpixel((0, 0), 0 if img_copy.getpixel((0, 0)) == 255 else 255)
    bmp_buffer = BytesIO()
    img_copy.save(bmp_buffer, format='BMP')
    bmp_bytes2 = bmp_buffer.getvalue()
    png_bytes1 = utils.convert_bmp_bytes_to_png(BytesIO(bmp_bytes1)).getvalue()
    png_bytes2 = utils.convert_bmp_bytes_to_png(BytesIO(bmp_bytes2)).getvalue()
    master.update({
        'bmp_entries': [bytes(bmp_bytes1), bytes(bmp_bytes2)],
        'png_entries': [png_bytes1, png_bytes2],
        'hashes': ['hashA', 'hashB'],
        'meta': [
            {
                'id': 'hashA',
                'hash': 'hashA',
                'plugin': 'TestPlugin',
                'label': 'Entry A',
                'url_png': '/web/a.png',
                'url_bmp': '/web/a.bmp'
            },
            {
                'id': 'hashB',
                'hash': 'hashB',
                'plugin': 'TestPlugin',
                'label': 'Entry B',
                'url_png': '/web/b.png',
                'url_bmp': '/web/b.bmp'
            }
        ],
        'selected_ids': ['hashA', 'hashB'],
        'version': 0
    })
    return master


def _reset_device_context(*device_ids: str) -> None:
    for device_id in device_ids:
        for key in ('devices', 'device_playlists', 'client_metrics', 'device_profiles'):
            state.global_state.setdefault(key, {}).pop(device_id, None)
        models.delete_device_state(device_id)
        models.delete_rotation_playlist(device_id)
        models.delete_device_playlist_binding(device_id)


def test_save_playlist_updates_rotation_and_order():
    client = TestClient(main.app)
    master = _prime_rotation_master()
    device_id = 'test-device'
    _reset_device_context(device_id)

    try:
        # Initial GET
        resp = client.get('/rotation')
        assert resp.status_code == 200
        data = resp.json()
        assert data['entries'][0]['id'] == 'hashA'
        assert data['entries'][1]['id'] == 'hashB'

        # Post new playlist selecting only hashB
        resp = client.post('/rotation', json={'playlist': ['hashB']})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload['playlists']['default'] == ['hashB']
        assert master['selected_ids'] == ['hashB']
        # Version should bump
        assert master['version'] == 1

        # Trigger display request so device records the new playlist state
        resp_display = client.get('/api/display', headers={'X-Device-Id': device_id})
        assert resp_display.status_code == 200
        device_state = state.get_device_state(device_id)
        with state.STATE_LOCK:
            assert device_state['playlist_indexes'] == [1]
            assert device_state['playlist_ids'] == ['hashB']
            assert device_state['request_count'] == 1
            assert device_state['last_entry_hash'] == 'hashB'
    finally:
        _reset_device_context(device_id)


def test_device_specific_playlist_update():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'device-specific'
    _reset_device_context(device_id)

    try:
        resp = client.post('/rotation', json={'playlist': ['hashA'], 'device_id': device_id})
        assert resp.status_code == 200
        payload = resp.json()
        device_playlists: Dict[str, List[str]] = payload['playlists'].get('devices') or {}
        assert device_playlists.get(device_id) == ['hashA']
        assert state.get_playlist_selection(device_id) == ['hashA']
    finally:
        _reset_device_context(device_id)


def test_devices_endpoint_lists_known_devices():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'metrics-device'
    _reset_device_context(device_id)
    try:
        state.get_device_state(device_id)
        state.update_client_metrics(device_id, refresh_rate=90, battery_voltage=3.8, rssi=-55)

        resp = client.get('/devices', params={'include_default': 'false'})
        assert resp.status_code == 200
        payload = resp.json()
        devices = payload['devices']
        assert all(entry['device_id'] != state.DEFAULT_DEVICE_ID for entry in devices)
        assert any(entry['device_id'] == device_id for entry in devices)
    finally:
        _reset_device_context(device_id)


def test_device_profile_update_endpoint():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'kitchen-panel'
    _reset_device_context(device_id)
    try:
        state.get_device_state(device_id)

        resp = client.patch(
            f'/devices/{device_id}',
            json={
                'friendly_name': 'Kitchen Display',
                'refresh_interval': 180,
                'playlist': ['hashB']
            }
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload['device_id'] == device_id
        assert payload['friendly_name'] == 'Kitchen Display'
        assert payload['refresh_interval'] == 180
        assert payload['playlist'] == ['hashB']

        profile = state.ensure_device_profile(device_id)
        assert profile['friendly_name'] == 'Kitchen Display'

    finally:
        _reset_device_context(device_id)

def test_image_token_resolves_device_without_headers():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'token-device'
    _reset_device_context(device_id)

    try:
        # Bind a named playlist so we can verify the resolved frame index.
        state.set_named_playlist('Charts', ['hashB'])
        state.set_device_playlist_binding(device_id, 'Charts')

        resp_display = client.get('/api/display', headers={'X-Device-Id': device_id, 'fw-version': '1.6.9'})
        assert resp_display.status_code == 200
        image_url = resp_display.json().get('image_url')
        assert isinstance(image_url, str) and 'token=' in image_url
        token = image_url.split('token=')[-1]
        assert token

        device_state = state.get_device_state(device_id)
        with state.STATE_LOCK:
            pending_idx = device_state.get('pending_entry_index')
        assert pending_idx == 1

        resp_image = client.get(f'/image/grayscale.png?token={token}')
        assert resp_image.status_code == 200
        assert resp_image.content == state.get_rotation_png_bytes(pending_idx)
    finally:
        _reset_device_context(device_id)


def test_playlist_entry_forces_bmp_even_when_grayscale_supported():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'force-bmp-device'
    _reset_device_context(device_id)

    try:
        resp = client.post('/rotation', json={'playlist': ['hashB@bmp'], 'device_id': device_id})
        assert resp.status_code == 200

        resp_display = client.get('/api/display', headers={'X-Device-Id': device_id, 'fw-version': '1.6.9'})
        assert resp_display.status_code == 200
        image_url = resp_display.json().get('image_url')
        assert isinstance(image_url, str)
        assert '/image/screen' in image_url

        parsed = urlparse(image_url)
        resp_image = client.get(f"{parsed.path}?{parsed.query}")
        assert resp_image.status_code == 200
        assert resp_image.content == state.get_rotation_bmp_bytes(1)
    finally:
        _reset_device_context(device_id)


def test_playlist_entry_forces_png_when_grayscale_supported():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'force-png-device'
    _reset_device_context(device_id)

    try:
        resp = client.post('/rotation', json={'playlist': ['hashB@png'], 'device_id': device_id})
        assert resp.status_code == 200

        resp_display = client.get('/api/display', headers={'X-Device-Id': device_id, 'fw-version': '1.6.9'})
        assert resp_display.status_code == 200
        image_url = resp_display.json().get('image_url')
        assert isinstance(image_url, str)
        assert '/image/grayscale' in image_url

        parsed = urlparse(image_url)
        resp_image = client.get(f"{parsed.path}?{parsed.query}")
        assert resp_image.status_code == 200
        assert resp_image.content == state.get_rotation_png_bytes(1)
    finally:
        _reset_device_context(device_id)


def test_preview_endpoint_persists_last_frame():
    client = TestClient(main.app)
    _prime_rotation_with_dummy_frames()
    device_id = 'preview-device'
    _reset_device_context(device_id)

    try:
        # Trigger a display fetch which should capture the preview
        resp = client.get('/api/display', headers={'X-Device-Id': device_id})
        assert resp.status_code == 200

        device_state = state.get_device_state(device_id)
        with state.STATE_LOCK:
            preview_index = device_state.get('current_preview_entry_index')
            preview_url = device_state.get('current_preview_url')
            preview_token = device_state.get('current_preview_token')

        assert preview_index is not None, 'preview index should be cached after display fetch'
        assert preview_url and preview_url.startswith(f'/preview/{device_id}')
        assert preview_token, 'preview token should be set'

        # Preview endpoint should return the cached PNG
        resp_preview = client.get(f'/preview/{device_id}')
        assert resp_preview.status_code == 200
        assert resp_preview.headers.get('content-type') == 'image/png'
        expected_bytes = state.get_rotation_png_bytes(preview_index)
        assert resp_preview.content == expected_bytes

        # A different device with no preview should 404
        resp_missing = client.get('/preview/unknown-device')
        assert resp_missing.status_code == 404
    finally:
        _reset_device_context(device_id)


def test_named_playlist_binding_affects_rotation_and_deletion_unbinds():
    client = TestClient(main.app)
    _prime_rotation_master()
    device_id = 'bound-device'
    _reset_device_context(device_id)

    try:
        resp = client.post('/playlists', json={'name': 'morning', 'playlist': ['hashA']})
        assert resp.status_code == 200

        resp = client.patch(f'/devices/{device_id}', json={'playlist_name': 'morning'})
        assert resp.status_code == 200

        resp_display = client.get('/api/display', headers={'X-Device-Id': device_id})
        assert resp_display.status_code == 200
        device_state = state.get_device_state(device_id)
        with state.STATE_LOCK:
            assert device_state['playlist_ids'] == ['hashA']

        resp_delete = client.delete('/playlists/morning')
        assert resp_delete.status_code == 200

        device_state = state.get_device_state(device_id)
        with state.STATE_LOCK:
            assert device_state.get('request_count') == 0

        resp_display = client.get('/api/display', headers={'X-Device-Id': device_id})
        assert resp_display.status_code == 200
        device_state = state.get_device_state(device_id)
        with state.STATE_LOCK:
            assert device_state['playlist_ids'] == ['hashA', 'hashB']
    finally:
        _reset_device_context(device_id)


def test_named_playlist_requires_at_least_one_entry():
    client = TestClient(main.app)
    _prime_rotation_master()

    resp = client.post('/playlists', json={'name': 'empty', 'playlist': []})
    assert resp.status_code == 400


def test_preview_updates_on_subsequent_frames():
    client = TestClient(main.app)
    _prime_rotation_with_two_frames()
    device_id = 'preview-advancing'
    _reset_device_context(device_id)

    try:
        # First frame
        resp1 = client.get('/api/display', headers={'X-Device-Id': device_id})
        assert resp1.status_code == 200
        state_after_first = state.get_device_state(device_id)
        with state.STATE_LOCK:
            first_token = state_after_first.get('current_preview_token')
            first_index = state_after_first.get('current_preview_entry_index')
        assert first_index is not None
        assert first_token
        resp_preview_first = client.get(f'/preview/{device_id}')
        assert resp_preview_first.status_code == 200
        first_bytes = resp_preview_first.content

        # Second frame (rotation advances)
        resp2 = client.get('/api/display', headers={'X-Device-Id': device_id})
        assert resp2.status_code == 200
        state_after_second = state.get_device_state(device_id)
        with state.STATE_LOCK:
            second_token = state_after_second.get('current_preview_token')
            second_index = state_after_second.get('current_preview_entry_index')
        assert second_index is not None
        assert second_token
        resp_preview_second = client.get(f'/preview/{device_id}')
        assert resp_preview_second.status_code == 200
        second_bytes = resp_preview_second.content
        assert second_bytes
        assert second_bytes != first_bytes, 'Preview bytes should update to the next rotation frame'
        assert second_token != first_token, 'Preview token should change when preview bytes change'
    finally:
        _reset_device_context(device_id)


def test_preview_sequence_stores_each_frame():
    client = TestClient(main.app)
    master = _prime_rotation_with_two_frames()
    device_id = 'preview-sequence'
    _reset_device_context(device_id)

    try:
        expected_sequence = master['png_entries']
        for idx in range(3):
            resp = client.get('/api/display', headers={'X-Device-Id': device_id})
            assert resp.status_code == 200
            device_state = state.get_device_state(device_id)
            with state.STATE_LOCK:
                preview_token = device_state.get('current_preview_token')
                preview_index = device_state.get('current_preview_entry_index')
            assert preview_index is not None
            assert preview_token
            expected_bytes = expected_sequence[idx % len(expected_sequence)]

            resp_preview = client.get(f'/preview/{device_id}')
            assert resp_preview.status_code == 200
            assert resp_preview.content == expected_bytes
    finally:
        _reset_device_context(device_id)


def test_playlist_persistence_across_plugin_refresh():
    """Test that user-defined playlists are preserved during plugin refresh."""
    # Setup initial state with a user-defined playlist
    master = _prime_rotation_master()
    device_id = 'test-persistence'
    _reset_device_context(device_id)
    
    # Set a user-defined playlist (only hashB, not all entries)  
    user_playlist = ['hashB']
    
    # Save to database to simulate existing persistent state
    state.persist_default_playlist(user_playlist)
    
    # Reset rotation master to initial state to simulate fresh server start
    _prime_rotation_master()
    # Clear selected_ids and has_persistent_playlist to simulate fresh start  
    state.rotation_master()['selected_ids'] = []
    state.rotation_master()['has_persistent_playlist'] = False
    
    # Initialize playlists from storage (this should set has_persistent_playlist flag)
    state.initialize_rotation_playlists_from_storage()
    
    # Verify the playlist was loaded and flag was set
    assert master['selected_ids'] == user_playlist
    assert master.get('has_persistent_playlist') == True
    
    # Simulate what happens when set_primary_rotation_assets is called during plugin refresh
    from trmnl_server.plugins.base import PluginOutput
    from trmnl_server.services import state as state_module
    from trmnl_server import utils
    from io import BytesIO
    
    # Create mock plugin assets
    dummy_path = utils.asset_path('img', 'dummy.bmp')
    with open(dummy_path, 'rb') as handle:
        bmp_bytes = handle.read()
    png_bytes = utils.convert_bmp_bytes_to_png(BytesIO(bmp_bytes)).getvalue()
    
    # Write temp files for the assets
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        bmp_path = os.path.join(tmp_dir, 'test.bmp')
        png_path = os.path.join(tmp_dir, 'test.png')
        
        with open(bmp_path, 'wb') as f:
            f.write(bmp_bytes)
        with open(png_path, 'wb') as f:
            f.write(png_bytes)
        
        assets = PluginOutput(
            monochrome_path=bmp_path,
            grayscale_path=png_path
        )
        
        try:
            # This should NOT overwrite the user playlist since it already exists and has_persistent_playlist flag is set
            state_module.set_primary_rotation_assets('TestPlugin', assets)
            
            # Verify the user playlist is preserved
            assert master['selected_ids'] == user_playlist, f"Expected {user_playlist}, got {master['selected_ids']}"
            assert len(master['meta']) > 0, "Meta entries should be populated"
            
            # Verify the playlist wasn't auto-filled with all available entries
            available_ids = [entry.get('id') for entry in master['meta'] if entry.get('id')]
            assert len(available_ids) > len(user_playlist), "Should have multiple available entries"
            assert master['selected_ids'] != available_ids, "Should not auto-fill with all available entries"
            
        finally:
            _reset_device_context(device_id)
