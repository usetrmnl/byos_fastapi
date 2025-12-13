(function () {
  const { h, render, Fragment } = preact;
  const { useState, useEffect, useMemo, useCallback, useRef } = preactHooks;
  const html = htm.bind(h);

  const TABS = [
    { id: 'home', label: 'Home', icon: 'fa-home' },
    { id: 'devices', label: 'Devices', icon: 'fa-tablet-screen-button' },
    { id: 'logs', label: 'Server Logs', icon: 'fa-history' },
    { id: 'rotation', label: 'Playlists', icon: 'fa-images' }
  ];

  const LOG_PAGE_SIZE = 50;
  const LOG_BUFFER_LIMIT = 200;

  function parsePlaylistToken(value) {
    if (typeof value !== 'string') {
      return { id: value, mode: null, raw: value };
    }
    const trimmed = value.trim();
    if (!trimmed) {
      return { id: '', mode: null, raw: value };
    }
    const atIndex = trimmed.indexOf('@');
    const baseId = atIndex >= 0 ? trimmed.slice(0, atIndex) : trimmed;
    const suffix = atIndex >= 0 ? trimmed.slice(atIndex + 1).trim().toLowerCase() : '';
    if (!suffix || suffix === 'auto') {
      return { id: baseId, mode: null, raw: baseId };
    }
    if (suffix === 'bmp' || suffix === 'mono') {
      return { id: baseId, mode: 'bmp', raw: `${baseId}@bmp` };
    }
    if (suffix === 'png' || suffix === 'gray' || suffix === 'grayscale') {
      return { id: baseId, mode: 'png', raw: `${baseId}@png` };
    }
    return { id: baseId, mode: null, raw: baseId };
  }

  function buildPlaylistTokensFromOrder(orderIds, tokens) {
    const tokenByBase = new Map();
    (tokens || []).forEach((token) => {
      const parsed = parsePlaylistToken(token);
      if (parsed && parsed.id) {
        tokenByBase.set(parsed.id, parsed.raw);
      }
    });
    return Array.from(orderIds || []).filter((id) => tokenByBase.has(id)).map((id) => tokenByBase.get(id));
  }

  function uniqueOrdered(list) {
    const seen = new Set();
    const out = [];
    (list || []).forEach((value) => {
      if (!seen.has(value)) {
        seen.add(value);
        out.push(value);
      }
    });
    return out;
  }

  const DEFAULT_STATUS = {
    server: { cpu_load: 0, current_time: '', uptime: '' },
    client: {
      device_id: '',
      friendly_name: '',
      battery_voltage: 0,
      battery_voltage_max: 5,
      battery_voltage_min: 2.5,
      battery_state: 0,
      wifi_signal: 0,
      wifi_signal_strength: 0,
      refresh_time: 0,
      last_contact: '',
      current_entry_hash: '',
      current_plugin_id: '',
      current_preview_url: '',
      current_preview_token: '',
      profile: {
        refresh_interval: null,
        time_zone: null,
        last_seen: null
      }
    },
    client_data_db: [],
    devices: [],
    playlists: []
  };

  function withCacheBuster(url, seed, fallbackToNow = true) {
    if (!url) {
      return url;
    }
    if (seed == null && !fallbackToNow) {
      return url;
    }
    const raw = seed != null ? seed : Date.now();
    const marker = encodeURIComponent(String(raw));
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}_=${marker}`;
  }

  function useInterval(callback, delay, enabled = true) {
    useEffect(() => {
      if (!enabled || delay == null) return undefined;
      const id = setInterval(callback, delay);
      return () => clearInterval(id);
    }, [callback, delay, enabled]);
  }

  function computeRotationOrder(entries, playlistIds, prevOrder = []) {
    const entryIds = (entries || []).map((e) => e.id).filter(Boolean);
    const seen = new Set();
    const ordered = [];

    const playlistBaseIds = uniqueOrdered((playlistIds || []).map((token) => parsePlaylistToken(token).id).filter(Boolean));

    // 1) Playlist (active) order from server/client
    (playlistBaseIds || []).forEach((id) => {
      if (entryIds.includes(id) && !seen.has(id)) {
        ordered.push(id);
        seen.add(id);
      }
    });

    // 2) Preserve previous ordering for remaining entries
    prevOrder.forEach((id) => {
      if (entryIds.includes(id) && !seen.has(id)) {
        ordered.push(id);
        seen.add(id);
      }
    });

    // 3) Append any new entries not seen before
    entryIds.forEach((id) => {
      if (!seen.has(id)) {
        ordered.push(id);
        seen.add(id);
      }
    });

    return ordered;
  }

  function parseUptime(uptimeStr) {
    if (!uptimeStr || typeof uptimeStr !== 'string') {
      return null;
    }
    const trimmed = uptimeStr.trim();
    if (!trimmed) {
      return null;
    }
    let dayCount = 0;
    let timePortion = trimmed;
    const dayMatch = trimmed.match(/(\d+)\s+day/);
    if (dayMatch) {
      dayCount = Number(dayMatch[1]) || 0;
      const commaIndex = trimmed.indexOf(',');
      timePortion = commaIndex >= 0 ? trimmed.slice(commaIndex + 1).trim() : trimmed;
    }
    const parts = timePortion.split(':').map((value) => Number(value));
    if (parts.some((value) => Number.isNaN(value))) {
      return null;
    }
    let hours = 0;
    let minutes = 0;
    let seconds = 0;
    if (parts.length === 3) {
      [hours, minutes, seconds] = parts;
    } else if (parts.length === 2) {
      [minutes, seconds] = parts;
    } else if (parts.length === 1) {
      [seconds] = parts;
    } else {
      return null;
    }
    return (dayCount * 86400) + (hours * 3600) + (minutes * 60) + seconds;
  }

  function formatUptime(totalSeconds) {
    if (!Number.isFinite(totalSeconds)) {
      return '';
    }
    const rounded = Math.max(0, Math.floor(totalSeconds));
    const days = Math.floor(rounded / 86400);
    let remainder = rounded - (days * 86400);
    const hours = Math.floor(remainder / 3600);
    remainder -= hours * 3600;
    const minutes = Math.floor(remainder / 60);
    const seconds = remainder - (minutes * 60);
    const timePart = `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    if (days > 0) {
      return `${days} day${days === 1 ? '' : 's'}, ${timePart}`;
    }
    return timePart;
  }

  function formatDisplayTimestamp(value) {
    if (value == null || value === '') {
      return 'N/A';
    }
    let candidate = '';
    if (typeof value === 'number') {
      if (value <= 0) {
        return 'N/A';
      }
      candidate = new Date(value * 1000).toISOString();
    } else if (typeof value === 'string') {
      candidate = value.trim();
      if (!candidate) {
        return 'N/A';
      }
    } else {
      return 'N/A';
    }
    const match = candidate.match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})/);
    if (match) {
      return `${match[1]} ${match[2]}`;
    }
    if (candidate.includes('T')) {
      return candidate.replace('T', ' ');
    }
    return candidate;
  }

  function arraysEqual(lhs = [], rhs = []) {
    if (lhs === rhs) {
      return true;
    }
    if (!lhs || !rhs || lhs.length !== rhs.length) {
      return false;
    }
    for (let idx = 0; idx < lhs.length; idx += 1) {
      if (lhs[idx] !== rhs[idx]) {
        return false;
      }
    }
    return true;
  }

  function normalizeDevicesList(rawDevices = []) {
    if (!Array.isArray(rawDevices)) {
      return [];
    }
    const seen = new Set();
    return rawDevices.filter((device) => {
      if (!device || !device.device_id || device.device_id === 'default') {
        return false;
      }
      if (seen.has(device.device_id)) {
        return false;
      }
      seen.add(device.device_id);
      return true;
    });
  }

  function App() {
    const [activeTab, setActiveTab] = useState('home');
    const [status, setStatus] = useState(DEFAULT_STATUS);
    const [devices, setDevices] = useState([]);
    const [selectedDeviceId, setSelectedDeviceId] = useState('');
    const [rotation, setRotation] = useState({ entries: [], playlists: { default: [] } });
    const [rotationOrder, setRotationOrder] = useState([]);
    const [activeRotationIds, setActiveRotationIds] = useState([]);
    const [activePlaylistTokens, setActivePlaylistTokens] = useState([]);
    const [rotationTarget, setRotationTarget] = useState('default');
    const [draftPlaylists, setDraftPlaylists] = useState({});
    const [logsData, setLogsData] = useState([]);
    const [logsAutoRefresh, setLogsAutoRefresh] = useState(true);
    const [logsLoading, setLogsLoading] = useState(false);
    const [logsError, setLogsError] = useState(null);
    const [rotationFeedback, setRotationFeedback] = useState('');
    const [theme, setTheme] = useState(() => {
      if (typeof window === 'undefined') {
        return 'light';
      }
      const stored = window.localStorage.getItem('trmnl-theme');
      if (stored === 'light' || stored === 'dark') {
        return stored;
      }
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    });
    const [pageVisible, setPageVisible] = useState(() => {
      if (typeof document === 'undefined') {
        return true;
      }
      return document.visibilityState !== 'hidden';
    });
    const lastLogIdRef = useRef(null);
    const pendingLogResetRef = useRef(0);
    const mergeDevices = useCallback((incoming) => {
      if (!Array.isArray(incoming) || incoming.length === 0) {
        return;
      }
      setDevices((prev) => {
        const map = new Map();
        (Array.isArray(prev) ? prev : []).forEach((device) => {
          if (device && device.device_id && device.device_id !== 'default') {
            map.set(device.device_id, device);
          }
        });
        incoming.forEach((device) => {
          if (device && device.device_id && device.device_id !== 'default') {
            map.set(device.device_id, device);
          }
        });
        return Array.from(map.values());
      });
    }, []);
    const statusInterval = useMemo(() => {
      const refreshSeconds = Number(status?.client?.refresh_time) || 60;
      return Math.max(5000, (refreshSeconds * 1000) / 2);
    }, [status?.client?.refresh_time]);
    const deviceOptions = useMemo(() => {
      const source = devices.length ? devices : (status.devices || []);
      return normalizeDevicesList(source);
    }, [devices, status.devices]);
    const handleDeviceSelect = useCallback((nextId) => {
      if (!nextId) {
        return;
      }
      setSelectedDeviceId(nextId);
    }, []);
    const toggleTheme = useCallback(() => {
      setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
    }, []);
    const handleRotationTargetChange = useCallback((value) => {
      const nextTarget = value || 'default';
      if (rotationTarget && rotationTarget !== 'default') {
        setDraftPlaylists((prev) => {
          if (!Object.prototype.hasOwnProperty.call(prev, rotationTarget)) {
            return prev;
          }
          const activeOrdered = rotationOrder.filter((id) => activeRotationIds.includes(id));
          const current = Array.isArray(prev[rotationTarget]) ? prev[rotationTarget] : [];
          if (arraysEqual(current, activeOrdered)) {
            return prev;
          }
          return { ...prev, [rotationTarget]: activeOrdered };
        });
      }
      setRotationTarget(nextTarget);
    }, [activeRotationIds, rotationOrder, rotationTarget]);
    const resolvePlaylistForTarget = useCallback((snapshot, target, drafts) => {
      if (!snapshot || !snapshot.playlists) {
        return [];
      }
      if (target && target !== 'default') {
        if (drafts && Object.prototype.hasOwnProperty.call(drafts, target)) {
          return Array.from(drafts[target] || []);
        }
        const named = snapshot.playlists?.named || {};
        if (named[target]) {
          return Array.from(named[target]);
        }
      }
      return Array.from(snapshot.playlists?.default || []);
    }, []);
    const fetchStatus = useCallback(async (deviceOverride) => {
      try {
        const targetDevice = deviceOverride || selectedDeviceId;
        const params = new URLSearchParams();
        if (targetDevice) {
          params.set('device_id', targetDevice);
        }
        const query = params.toString();
        const res = await fetch(withCacheBuster(query ? `/status?${query}` : '/status'));
        if (!res.ok) return;
        const data = await res.json();
        setStatus(data);
        if (Array.isArray(data?.devices)) {
          mergeDevices(data.devices);
        }
        const incomingDeviceId = data?.client?.device_id;
        if (incomingDeviceId) {
          if (!selectedDeviceId) {
            setSelectedDeviceId(incomingDeviceId);
          } else if (deviceOverride && incomingDeviceId !== selectedDeviceId) {
            setSelectedDeviceId(incomingDeviceId);
          }
        }
      } catch (e) {
        console.warn('status fetch failed', e);
      }
    }, [mergeDevices, selectedDeviceId]);

    const fetchDevices = useCallback(async () => {
      try {
        const res = await fetch(withCacheBuster('/devices?include_default=false'));
        if (!res.ok) return;
        const data = await res.json();
        if (Array.isArray(data?.devices)) {
          mergeDevices(data.devices);
        }
      } catch (e) {
        console.warn('devices fetch failed', e);
      }
    }, [mergeDevices]);

    useEffect(() => {
      if (typeof document !== 'undefined') {
        document.documentElement.dataset.theme = theme;
      }
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem('trmnl-theme', theme);
      }
    }, [theme]);

    useEffect(() => {
      if (typeof document === 'undefined') {
        return undefined;
      }
      const handleVisibilityChange = () => setPageVisible(document.visibilityState !== 'hidden');
      document.addEventListener('visibilitychange', handleVisibilityChange);
      return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
    }, []);

    const fetchRotation = useCallback(async () => {
      try {
        const res = await fetch(withCacheBuster('/rotation'));
        if (!res.ok) return;
        const data = await res.json();
        setRotation(data);
      } catch (e) {
        console.warn('rotation fetch failed', e);
      }
    }, []);

    const handleDeviceUpdate = useCallback(
      async (deviceId, payload) => {
        if (!deviceId) {
          return false;
        }
        try {
          const res = await fetch(withCacheBuster(`/devices/${deviceId}`), {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          if (!res.ok) {
            throw new Error(`Status ${res.status}`);
          }
          const nextStatusDevice = deviceId === selectedDeviceId ? deviceId : (selectedDeviceId || deviceId);
          await fetchStatus(nextStatusDevice);
          await fetchRotation();
          return true;
        } catch (err) {
          console.warn('device update failed', err);
          return false;
        }
      },
      [fetchRotation, fetchStatus, selectedDeviceId]
    );

    const fetchLogs = useCallback(
      async (options = {}) => {
        const { reset = false } = options;
        if (reset) {
          lastLogIdRef.current = null;
          pendingLogResetRef.current += 1;
          setLogsLoading(true);
        }
        setLogsError(null);
        try {
          const params = new URLSearchParams({
            format: 'json',
            limit: String(LOG_PAGE_SIZE)
          });
          if (!reset && lastLogIdRef.current != null) {
            params.set('after', String(lastLogIdRef.current));
          }
          const res = await fetch(withCacheBuster(`/server/log?${params.toString()}`), {
            cache: 'no-store',
            headers: { Accept: 'application/json' }
          });
          if (!res.ok) throw new Error(`status ${res.status}`);
          const data = await res.json();
          if (reset) {
            setLogsData(data.slice(-LOG_BUFFER_LIMIT));
          } else if (data.length) {
            setLogsData((prev) => {
              const existingIds = new Set(prev.map((entry) => entry.id));
              const merged = [...prev];
              data.forEach((entry) => {
                if (!existingIds.has(entry.id)) {
                  merged.push(entry);
                }
              });
              if (merged.length > LOG_BUFFER_LIMIT) {
                return merged.slice(-LOG_BUFFER_LIMIT);
              }
              return merged;
            });
          }
          if (data.length) {
            lastLogIdRef.current = data[data.length - 1].id;
          }
        } catch (e) {
          console.warn('logs fetch failed', e);
          setLogsError(e.message || String(e));
        } finally {
          if (reset) {
            pendingLogResetRef.current = Math.max(0, pendingLogResetRef.current - 1);
            if (pendingLogResetRef.current === 0) {
              setLogsLoading(false);
            }
          }
        }
      },
      []
    );

    const refreshLogs = useCallback(() => {
      fetchLogs({ reset: true });
    }, [fetchLogs]);

    const appendLogs = useCallback(() => {
      fetchLogs({ reset: false });
    }, [fetchLogs]);

    useEffect(() => {
      fetchStatus();
      fetchRotation();
      fetchDevices();
    }, [fetchStatus, fetchRotation, fetchDevices]);

    useEffect(() => {
      if (activeTab === 'rotation') {
        fetchRotation();
        fetchDevices();
      }
    }, [activeTab, fetchRotation, fetchDevices]);

    useEffect(() => {
      if (activeTab === 'home') {
        fetchStatus();
      }
    }, [activeTab, fetchStatus]);

    useEffect(() => {
      if (activeTab !== 'rotation' && selectedDeviceId) {
        fetchStatus(selectedDeviceId);
      }
    }, [activeTab, selectedDeviceId, fetchStatus]);

    useEffect(() => {
      if (!rotation || !rotation.entries) {
        return;
      }
      const playlistTokens = resolvePlaylistForTarget(rotation, rotationTarget, draftPlaylists);
      const playlistBaseIds = uniqueOrdered((playlistTokens || []).map((token) => parsePlaylistToken(token).id).filter(Boolean));
      setActivePlaylistTokens((prev) => (arraysEqual(prev, playlistTokens) ? prev : playlistTokens));
      setActiveRotationIds((prev) => (arraysEqual(prev, playlistBaseIds) ? prev : playlistBaseIds));
      setRotationOrder((prev) => {
        const next = computeRotationOrder(rotation.entries, playlistTokens, prev);
        return arraysEqual(prev, next) ? prev : next;
      });
    }, [rotation, rotationTarget, resolvePlaylistForTarget, draftPlaylists]);

    useInterval(fetchStatus, statusInterval, pageVisible && activeTab !== 'rotation');
    useInterval(appendLogs, 5000, activeTab === 'logs' && logsAutoRefresh);

    useEffect(() => {
      if (activeTab === 'logs') {
        refreshLogs();
      }
    }, [activeTab, refreshLogs]);

    const toggleSelection = (id) => {
      setActivePlaylistTokens((prevTokens) => {
        const existing = (prevTokens || []).find((token) => parsePlaylistToken(token).id === id);
        const nextTokens = existing
          ? (prevTokens || []).filter((token) => parsePlaylistToken(token).id !== id)
          : [...(prevTokens || []), id];

        const nextActiveIds = uniqueOrdered(nextTokens.map((token) => parsePlaylistToken(token).id).filter(Boolean));
        setActiveRotationIds((prevIds) => (arraysEqual(prevIds, nextActiveIds) ? prevIds : nextActiveIds));

        if (rotationTarget && rotationTarget !== 'default') {
          setDraftPlaylists((draftPrev) => {
            if (!Object.prototype.hasOwnProperty.call(draftPrev, rotationTarget)) {
              return draftPrev;
            }
            const orderedTokens = buildPlaylistTokensFromOrder(rotationOrder, nextTokens);
            const current = Array.isArray(draftPrev[rotationTarget]) ? draftPrev[rotationTarget] : [];
            if (arraysEqual(current, orderedTokens)) {
              return draftPrev;
            }
            return { ...draftPrev, [rotationTarget]: orderedTokens };
          });
        }
        return nextTokens;
      });
    };

    const toggleForceOneBit = useCallback((id, enabled) => {
      setActivePlaylistTokens((prevTokens) => {
        const tokens = Array.isArray(prevTokens) ? prevTokens : [];
        const idx = tokens.findIndex((token) => parsePlaylistToken(token).id === id);
        if (idx < 0) {
          return tokens;
        }
        const current = parsePlaylistToken(tokens[idx]);
        const nextRaw = enabled ? `${current.id}@bmp` : current.id;
        const nextTokens = tokens.slice();
        nextTokens[idx] = nextRaw;

        if (rotationTarget && rotationTarget !== 'default') {
          setDraftPlaylists((draftPrev) => {
            if (!Object.prototype.hasOwnProperty.call(draftPrev, rotationTarget)) {
              return draftPrev;
            }
            const orderedTokens = buildPlaylistTokensFromOrder(rotationOrder, nextTokens);
            const currentDraft = Array.isArray(draftPrev[rotationTarget]) ? draftPrev[rotationTarget] : [];
            if (arraysEqual(currentDraft, orderedTokens)) {
              return draftPrev;
            }
            return { ...draftPrev, [rotationTarget]: orderedTokens };
          });
        }
        return nextTokens;
      });
    }, [rotationOrder, rotationTarget]);

    const handleRotationReorder = useCallback((nextOrder) => {
      setRotationOrder(nextOrder);
      if (rotationTarget && rotationTarget !== 'default') {
        setDraftPlaylists((prev) => {
          if (!Object.prototype.hasOwnProperty.call(prev, rotationTarget)) {
            return prev;
          }
          const ordered = buildPlaylistTokensFromOrder(nextOrder || [], activePlaylistTokens);
          const current = Array.isArray(prev[rotationTarget]) ? prev[rotationTarget] : [];
          if (arraysEqual(current, ordered)) {
            return prev;
          }
          return { ...prev, [rotationTarget]: ordered };
        });
      }
    }, [activePlaylistTokens, rotationTarget]);

    const buildActivePlaylist = useCallback(() => {
      return buildPlaylistTokensFromOrder(rotationOrder, activePlaylistTokens);
    }, [rotationOrder, activePlaylistTokens]);

    const savePlaylistForTarget = useCallback(
      async (targetId) => {
        const playlist = buildActivePlaylist();
        if (!playlist.length) {
          return { success: false, count: 0, message: 'Select at least one plugin' };
        }
        try {
          const isDefault = !targetId || targetId === 'default';
          const endpoint = isDefault ? '/rotation' : '/playlists';
          const payload = isDefault ? { playlist } : { name: targetId, playlist };
          const res = await fetch(withCacheBuster(endpoint), {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
          if (!res.ok) {
            throw new Error(`Status ${res.status}`);
          }
          const data = await res.json();
          setRotation(data);
          if (!isDefault) {
            setDraftPlaylists((prev) => {
              if (!Object.prototype.hasOwnProperty.call(prev, targetId)) {
                return prev;
              }
              const next = { ...prev };
              delete next[targetId];
              return next;
            });
          }
          return { success: true, count: playlist.length };
        } catch (err) {
          console.error('playlist save failed', err);
          return { success: false, count: playlist.length, message: 'Failed to save playlist' };
        }
      },
      [buildActivePlaylist, setRotation]
    );

    const deletePlaylist = useCallback(async (name) => {
      const target = (name || '').trim();
      if (!target || target === 'default') {
        return false;
      }
      if (Object.prototype.hasOwnProperty.call(draftPlaylists, target)) {
        setDraftPlaylists((prev) => {
          const next = { ...prev };
          delete next[target];
          return next;
        });
        if (rotationTarget === target) {
          setRotationTarget('default');
        }
        return true;
      }
      try {
        const res = await fetch(withCacheBuster(`/playlists/${encodeURIComponent(target)}`), { method: 'DELETE' });
        if (!res.ok) {
          throw new Error(`Status ${res.status}`);
        }
        const data = await res.json();
        setRotation(data);
        if (rotationTarget === target) {
          setRotationTarget('default');
        }
        return true;
      } catch (err) {
        console.warn('playlist delete failed', err);
        setRotationFeedback('Unable to delete playlist');
        return false;
      }
    }, [draftPlaylists, rotationTarget, setRotation, setRotationFeedback]);

    const createPlaylist = useCallback((rawName) => {
      const target = (rawName || '').trim();
      if (!target || target === 'default') {
        setRotationFeedback('Invalid playlist name');
        return false;
      }
      const existing = rotation?.playlists?.named || {};
      if (existing[target] || Object.prototype.hasOwnProperty.call(draftPlaylists, target)) {
        setRotationFeedback('Playlist already exists');
        return false;
      }
      setDraftPlaylists((prev) => ({ ...prev, [target]: [] }));
      setRotationTarget(target);
      setRotationFeedback('New playlist created. Toggle at least one plugin, then Save.');
      return true;
    }, [draftPlaylists, rotation, setRotationFeedback]);

    const persistRotation = async () => {
      setRotationFeedback('');
      const scopeLabel = rotationTarget === 'default' ? 'default playlist' : `playlist ${rotationTarget}`;
      const result = await savePlaylistForTarget(rotationTarget);
      if (result.success) {
        setRotationFeedback(`Saved ${result.count} image(s) for ${scopeLabel}`);
      } else {
        setRotationFeedback(result.message || 'Failed to save playlist');
      }
    };

    const handleDeviceAssignmentToggle = useCallback(
      async (deviceId, enabled) => {
        if (!deviceId || deviceId === 'default' || !rotationTarget || rotationTarget === 'default') {
          return;
        }
        const named = rotation?.playlists?.named || {};
        const isDraft = Object.prototype.hasOwnProperty.call(draftPlaylists, rotationTarget) && !named[rotationTarget];
        if (isDraft) {
          setRotationFeedback('Save the playlist before assigning devices');
          return;
        }
        const payload = enabled ? { playlist_name: rotationTarget } : { playlist_name: null };
        const ok = await handleDeviceUpdate(deviceId, payload);
        if (ok) {
          setRotationFeedback(enabled ? `Assigned ${deviceId} to ${rotationTarget}` : `Unassigned ${deviceId}`);
        } else {
          setRotationFeedback('Failed to update device assignment');
        }
      },
      [draftPlaylists, handleDeviceUpdate, rotation, rotationTarget, setRotationFeedback]
    );

    const logsList = useMemo(() => {
      if (!logsData || logsData.length === 0) {
        return [];
      }
      return [...logsData].reverse();
    }, [logsData]);

    return html`
      <div class="app-shell">
        <${Menu} activeTab=${activeTab} setActiveTab=${setActiveTab} />
        <${Topbar}
          status=${status}
          theme=${theme}
          onToggleTheme=${toggleTheme}
          uptimeRaw=${status?.server?.uptime}
          devices=${deviceOptions}
          selectedDeviceId=${selectedDeviceId}
          onSelectDevice=${handleDeviceSelect}
        />
        <div class="container-wrapper">
          ${activeTab === 'home'
            ? html`<${Container} id="home" activeTab=${activeTab}>
                <${HomeTab}
                  status=${status}
                  devices=${deviceOptions}
                  onSelectDevice=${handleDeviceSelect}
                  selectedDeviceId=${selectedDeviceId}
                />
              </${Container}>`
            : null}
          ${activeTab === 'devices'
            ? html`<${Container} id="devices" activeTab=${activeTab}>
                <${DevicesTab}
                  devices=${deviceOptions}
                  selectedDeviceId=${selectedDeviceId}
                  onSelectDevice=${handleDeviceSelect}
                  onUpdateDevice=${handleDeviceUpdate}
                  deviceHistory=${status.client_data_db}
                  clientStats=${status.client}
                />
              </${Container}>`
            : null}
          ${activeTab === 'logs'
            ? html`<${Container} id="logs" activeTab=${activeTab}>
                <${LogsTab}
                  logsList=${logsList}
                  onRefresh=${refreshLogs}
                  autoRefresh=${logsAutoRefresh}
                  setAutoRefresh=${setLogsAutoRefresh}
                  loading=${logsLoading}
                  error=${logsError}
                />
              </${Container}>`
            : null}
          ${activeTab === 'rotation'
            ? html`<${Container} id="rotation" activeTab=${activeTab}>
                <${RotationTab}
                  rotation=${rotation}
                  rotationOrder=${rotationOrder}
                  activeRotationIds=${activeRotationIds}
                  activePlaylistTokens=${activePlaylistTokens}
                  toggleSelection=${toggleSelection}
                  toggleForceOneBit=${toggleForceOneBit}
                  persistRotation=${persistRotation}
                  rotationFeedback=${rotationFeedback}
                  onReorder=${handleRotationReorder}
                  rotationTarget=${rotationTarget}
                  onTargetChange=${handleRotationTargetChange}
                  devices=${deviceOptions}
                  onAssignmentToggle=${handleDeviceAssignmentToggle}
                  onCreatePlaylist=${createPlaylist}
                  onDeletePlaylist=${deletePlaylist}
                  draftNames=${Object.keys(draftPlaylists || {})}
                />
              </${Container}>`
            : null}
        </div>
      </div>
    `;
  }

  function Menu({ activeTab, setActiveTab }) {
    return html`
      <div class="menu">
        ${TABS.map(
          (tab) => html`
            <a class=${activeTab === tab.id ? 'active' : ''} onClick=${() => setActiveTab(tab.id)}>
              <span class="icon"><i class=${`fas ${tab.icon}`}></i></span>${tab.label}
            </a>`
        )}
      </div>
    `;
  }

  function Topbar({ status, theme, onToggleTheme, uptimeRaw, devices, selectedDeviceId, onSelectDevice }) {
    const deviceList = normalizeDevicesList((Array.isArray(devices) && devices.length ? devices : status.devices) || []);
    const selectorValue = selectedDeviceId || (deviceList[0]?.device_id || '');
    const uptimeDisplay = uptimeRaw || status.server.uptime || '--';
    return html`
      <div class="topbar">
        <div class="topbar-left">
          <div id="menu-button" class="topbar-item"><i class="fas fa-bars"></i></div>
          <div id="topbar-title" class="topbar-item topbar-title"><h3>TRMNL Server</h3></div>
        </div>
        <div class="topbar-metrics">
          <div class="status-item">
            <i class="fas fa-clock"></i>
            <span id="top_uptime">${uptimeDisplay}</span>
          </div>
          <div class="status-item">
        <button type="button" class="theme-toggle" onClick=${onToggleTheme}>
          <i class=${`fas ${theme === 'dark' ? 'fa-moon' : 'fa-sun'}`}></i>
          <span>${theme === 'dark' ? 'Dark' : 'Light'}</span>
        </button>
        </div>
        </div>
      </div>
    `;
  }

  function Container({ id, activeTab, children }) {
    return html`<div class=${`container ${activeTab === id ? 'active' : ''}`} id=${`container_${id}`}>${children}</div>`;
  }

  function HomeTab({ status, devices, onSelectDevice, selectedDeviceId }) {
    const serverTimeDisplay = formatDisplayTimestamp(status.server.current_time);
    const deviceList = normalizeDevicesList((devices && devices.length ? devices : status.devices) || []);
    const effectiveSelectedId = selectedDeviceId || status.client.device_id || 'default';
    const deviceSubtitle = deviceList.length ? `${deviceList.length} active` : 'Waiting for devices';
    const serverSubtitle = serverTimeDisplay !== 'N/A' ? `Updated ${serverTimeDisplay}` : 'Awaiting metrics';
    return html`
      <${Fragment}>
        <div class="home-row">
          <div class="section home-card home-card-devices">
            <div class="section-heading">
              <h2>Connected Devices</h2>
              <span class="section-subtitle">${deviceSubtitle}</span>
            </div>
            ${deviceList.length
              ? html`
                  <${DeviceGrid}
                    devices=${deviceList}
                    selectedDeviceId=${effectiveSelectedId}
                    onSelectDevice=${onSelectDevice}
                  />
                `
              : html`<div class="device-grid-empty">No devices have checked in yet.</div>`}
          </div>
          <div class="section home-card home-card-status">
            <div class="section-heading">
              <h2>Server Status</h2>
              <span class="section-subtitle">${serverSubtitle}</span>
            </div>
            <${StatusItem}
              name="CPU Load"
              value=${`${status.server.cpu_load} %`}
              barId="cpu-load-bar"
              barValue=${status.server.cpu_load}
              barClass="progress-bar-blue"
            />
            <div class="status-item"><p class="status-name">Current Time:</p><p>${serverTimeDisplay}</p></div>
            <div class="status-item"><p class="status-name">Uptime:</p><p>${status.server.uptime}</p></div>
            <div class="status-item"><p class="status-name">Devices Online:</p><p>${deviceList.length}</p></div>
          </div>
        </div>
      </${Fragment}>
    `;
  }

  function StatusItem({ name, value, barId, barValue, barClass }) {
    return html`
      <div class="status-item">
        <p class="status-name">${name}:</p>
        <p class="status-value">${value}</p>
        <div class="progress-bar">
          <div id=${barId} class=${`progress-bar-inner ${barClass}`} style=${{ width: `${barValue || 0}%` }}></div>
        </div>
      </div>
    `;
  }

  function DeviceGrid({ devices, selectedDeviceId, onSelectDevice }) {
    const normalizedDevices = normalizeDevicesList(devices || []);
    if (!normalizedDevices.length) {
      return html`<div class="device-grid-empty">No devices recorded yet</div>`;
    }
    return html`
      <div class="device-grid">
        ${normalizedDevices.map((device) => {
          const friendly = device.friendly_name || device.device_id;
          const metrics = device.metrics || {};
          const profile = device.profile || {};
          const voltageValue = Number(metrics.battery_voltage);
          const wifiValue = metrics.rssi;
          const voltage = Number.isFinite(voltageValue) ? `${voltageValue.toFixed(2)} V` : 'N/A';
          const wifi = Number.isFinite(wifiValue) ? `${wifiValue} dBm` : 'N/A';
          const rawRefresh = metrics.refresh_rate ?? profile.refresh_interval;
          const refreshNumber = Number(rawRefresh);
          const refreshText = Number.isFinite(refreshNumber) && refreshNumber > 0 ? `${refreshNumber}s` : 'N/A';
          const lastContact = formatDisplayTimestamp(metrics.last_contact || profile.last_seen);
          const previewUrl = device.state?.current_preview_url;
          const previewSeed = device.state?.current_preview_token || device.state?.current_entry_hash;
          const previewSrc = withCacheBuster(previewUrl, previewSeed, false);
          const pluginLabel = device.state?.current_plugin_id || 'Awaiting rotation';
          const playlistName = (device.playlist_name && String(device.playlist_name).trim()) ? String(device.playlist_name).trim() : 'default';
          const playlistLabel = playlistName === 'default' ? 'Default playlist' : playlistName;
          const isActive = device.device_id === selectedDeviceId;
          return html`
            <button
              type="button"
              key=${device.device_id}
              class=${`device-card${isActive ? ' active' : ''}`}
              onClick=${() => onSelectDevice && onSelectDevice(device.device_id)}
            >
              <div class="device-card-preview">
                ${previewSrc
                  ? html`<img
                      class="playlist-card-thumb device-card-thumb"
                      src=${previewSrc}
                      alt=${`Current frame for ${friendly}`}
                    />`
                  : html`<div class="device-card-thumb-placeholder">No preview yet</div>`}
              </div>
              <div class="device-card-header">
                <span class="device-card-name">${friendly}</span>
                <span class="device-card-id">${device.device_id}</span>
              </div>
              <div class="device-card-body">
                <div class="device-card-metric"><i class="fas fa-battery-half"></i>${voltage}</div>
                <div class="device-card-metric"><i class="fas fa-wifi"></i>${wifi}</div>
                <div class="device-card-metric"><i class="fas fa-sync-alt"></i>${refreshText}</div>
              </div>
              <div class="device-card-footer">
                <span class="device-card-playlist">Playlist: ${playlistLabel}</span>
                <span class="device-card-plugin">Now showing: ${pluginLabel}</span>
                <span>Last contact: ${lastContact}</span>
              </div>
            </button>
          `;
        })}
      </div>
    `;
  }

  function DevicesTab({
    devices,
    selectedDeviceId,
    onSelectDevice,
    onUpdateDevice,
    deviceHistory,
    clientStats
  }) {
    const deviceList = normalizeDevicesList(devices || []);
    const activeDevice = deviceList.find((device) => device.device_id === selectedDeviceId) || deviceList[0];
    const hasDevices = deviceList.length > 0;
    const [formValues, setFormValues] = useState({ friendly_name: '', refresh_interval: '', time_zone: '' });
    const [saving, setSaving] = useState(false);
    const [feedback, setFeedback] = useState('');
    const history = Array.isArray(deviceHistory) ? deviceHistory : [];
    const historyMatchesSelection = activeDevice && clientStats?.device_id === activeDevice.device_id;
    let filteredHistory = historyMatchesSelection ? history : [];
    if (filteredHistory && filteredHistory.length > 10) {
      filteredHistory = filteredHistory.slice(-10);
    }

    useEffect(() => {
      if (!activeDevice) {
        return;
      }
      setFormValues({
        friendly_name: activeDevice.friendly_name || '',
        refresh_interval: activeDevice.refresh_interval || activeDevice.metrics?.refresh_rate || '',
        time_zone: (activeDevice.profile && activeDevice.profile.time_zone) || ''
      });
      setFeedback('');
    }, [activeDevice]);

    useEffect(() => {
      if (!deviceList.length || !onSelectDevice) {
        return;
      }
      const exists = deviceList.some((device) => device.device_id === selectedDeviceId);
      if (!exists && deviceList[0]) {
        onSelectDevice(deviceList[0].device_id);
      }
    }, [deviceList, onSelectDevice, selectedDeviceId]);

    const handleInput = (key, value) => {
      setFormValues((prev) => ({ ...prev, [key]: value }));
    };

    const handleSubmit = async (event) => {
      event.preventDefault();
      if (!activeDevice || !onUpdateDevice) {
        return;
      }
      setSaving(true);
      const payload = {};
      if (formValues.friendly_name !== (activeDevice.friendly_name || '')) {
        payload.friendly_name = formValues.friendly_name;
      }
      const refreshValue = Number(formValues.refresh_interval);
      if (Number.isFinite(refreshValue) && refreshValue > 0) {
        payload.refresh_interval = refreshValue;
      }
      if (formValues.time_zone !== ((activeDevice.profile && activeDevice.profile.time_zone) || '')) {
        payload.time_zone = formValues.time_zone;
      }
      if (!Object.keys(payload).length) {
        setFeedback('Nothing to update');
        setSaving(false);
        return;
      }
      const success = await onUpdateDevice(activeDevice.device_id, payload);
      setFeedback(success ? 'Saved' : 'Unable to save');
      setSaving(false);
    };

    const metrics = activeDevice?.metrics || {};
    const profile = activeDevice?.profile || {};
    const batteryValue = Number(metrics.battery_voltage);
    const batteryText = Number.isFinite(batteryValue) ? `${batteryValue.toFixed(2)} V` : 'N/A';
    const wifiValue = Number(metrics.rssi);
    const wifiText = Number.isFinite(wifiValue) ? `${wifiValue} dBm` : 'N/A';
    const lastSeenText = formatDisplayTimestamp(metrics.last_contact || profile.last_seen);

    if (!hasDevices) {
      return html`
        <div class="devices-page">
          <div class="device-grid-empty devices-empty">
            No devices found in the database yet. A TRMNL will appear here once it checks in.
          </div>
        </div>
      `;
    }
    return html`
      <div class="devices-page">
        <div class="devices-layout">
          <div class="devices-column devices-column-left">
            <div class="devices-card devices-list-card">
              <div class="section-heading">
                <h2>Devices</h2>
                <span class="section-subtitle">${deviceList.length} total</span>
              </div>
              <${DeviceGrid}
                devices=${deviceList}
                selectedDeviceId=${selectedDeviceId}
                onSelectDevice=${onSelectDevice}
              />
            </div>
            ${activeDevice
              ? html`
                  <div class="devices-card device-settings-card">
                    <div class="section-heading">
                      <h2>Device settings</h2>
                      <span class="section-subtitle">ID: ${activeDevice.device_id}</span>
                    </div>
                    <div class="device-stats">
                      <span><i class="fas fa-bolt"></i>${batteryText}</span>
                      <span><i class="fas fa-wifi"></i>${wifiText}</span>
                      <span><i class="fas fa-clock"></i>${lastSeenText}</span>
                    </div>
                    <form class="device-form" onSubmit=${handleSubmit}>
                      <label>
                        <span>Friendly name</span>
                        <input
                          type="text"
                          value=${formValues.friendly_name}
                          onInput=${(e) => handleInput('friendly_name', e.target.value)}
                        />
                      </label>
                      <label>
                        <span>Refresh interval (seconds)</span>
                        <input
                          type="number"
                          min="30"
                          step="30"
                          value=${formValues.refresh_interval}
                          onInput=${(e) => handleInput('refresh_interval', e.target.value)}
                        />
                      </label>
                      <label>
                        <span>Time zone</span>
                        <input
                          type="text"
                          placeholder="e.g. Europe/Berlin"
                          value=${formValues.time_zone}
                          onInput=${(e) => handleInput('time_zone', e.target.value)}
                        />
                      </label>
                      <div class="device-form-actions">
                        <button type="submit" disabled=${saving}>${saving ? 'Saving…' : 'Save changes'}</button>
                        <span class="device-form-feedback">${feedback}</span>
                      </div>
                    </form>
                  </div>`
              : html`<div class="devices-card"><p class="device-history-empty">Select a device to edit settings.</p></div>`}
          </div>
          <div class="devices-column devices-column-right">
            ${activeDevice
              ? html`
                  <div class="metrics-card">
                    <div class="section-heading">
                      <h2>Battery & Signal</h2>
                      <span class="section-subtitle">
                        ${filteredHistory.length ? `${filteredHistory.length} samples (most recent)` : 'No telemetry yet'}
                      </span>
                    </div>
                    ${filteredHistory.length
                      ? html`
                          <${BatteryCharts}
                            data=${filteredHistory}
                            batteryMin=${clientStats?.battery_voltage_min}
                            batteryMax=${clientStats?.battery_voltage_max}
                          />
                        `
                      : html`<p class="device-history-empty">No telemetry stored for this device yet.</p>`}
                  </div>
                  <div class="metrics-card">
                    <div class="section-heading">
                      <h3>Telemetry samples</h3>
                      <span class="section-subtitle">Latest readings</span>
                    </div>
                    ${filteredHistory.length
                      ? html`<${StatsTable} data=${filteredHistory} />`
                      : html`<p class="device-history-empty">No telemetry stored for this device yet.</p>`}
                  </div>
                `
              : html`<div class="metrics-card"><p class="device-history-empty">Select a device to view metrics.</p></div>`}
          </div>
        </div>
      </div>
    `;
  }

  function BatteryCharts({ data, batteryMin = 2.5, batteryMax = 5.0 }) {
    const voltageCanvasRef = useRef(null);
    const rssiCanvasRef = useRef(null);
    const voltageChartRef = useRef(null);
    const rssiChartRef = useRef(null);

    useEffect(() => {
      if (!data || data.length === 0 || typeof Chart === 'undefined') {
        return undefined;
      }

      const timestamps = data.map((entry) => entry.timestamp);
      const voltages = data.map((entry) => entry.battery_voltage);
      const socs = data.map((entry) => {
        const v = entry.battery_voltage;
        const soc = ((v - batteryMin) / (batteryMax - batteryMin)) * 100;
        return Math.max(0, Math.min(soc, 100));
      });

      if (voltageChartRef.current) {
        voltageChartRef.current.destroy();
      }

      voltageChartRef.current = new Chart(voltageCanvasRef.current.getContext('2d'), {
        type: 'line',
        data: {
          labels: timestamps,
          datasets: [
            {
              label: 'Battery Voltage',
              data: voltages,
              borderColor: 'rgba(75, 192, 192, 1)',
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              fill: true,
              yAxisID: 'y-voltage'
            },
            {
              label: 'Battery SOC (%)',
              data: socs,
              borderColor: 'rgba(255, 99, 132, 1)',
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              fill: true,
              yAxisID: 'y-soc'
            }
          ]
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: 'time',
              time: { unit: 'hour', displayFormats: { hour: 'dd.MM. HH:mm' } }
            },
            'y-voltage': {
              beginAtZero: false,
              title: { display: true, text: 'Battery Voltage (V)' },
              position: 'left'
            },
            'y-soc': {
              beginAtZero: true,
              title: { display: true, text: 'Battery SOC (%)' },
              position: 'right',
              grid: { drawOnChartArea: false }
            }
          }
        }
      });

      const rssiValues = data.map((entry) => entry.rssi);
      if (rssiChartRef.current) {
        rssiChartRef.current.destroy();
      }
      rssiChartRef.current = new Chart(rssiCanvasRef.current.getContext('2d'), {
        type: 'line',
        data: {
          labels: timestamps,
          datasets: [
            {
              label: 'WiFi Signal Strength (dBm)',
              data: rssiValues,
              borderColor: 'rgba(75, 192, 192, 1)',
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              fill: true,
              yAxisID: 'y-rssi'
            },
            {
              label: 'WiFi Signal Strength (%)',
              data: rssiValues.map(getWifiStrength),
              borderColor: 'rgba(255, 99, 132, 1)',
              backgroundColor: 'rgba(255, 99, 132, 0.2)',
              fill: true,
              yAxisID: 'y-strength'
            }
          ]
        },
        options: {
          animation: false,
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              type: 'time',
              time: { unit: 'hour', displayFormats: { hour: 'dd.MM. HH:mm' } }
            },
            'y-rssi': {
              beginAtZero: false,
              title: { display: true, text: 'RSSI (dBm)' }
            },
            'y-strength': {
              beginAtZero: false,
              title: { display: true, text: 'Wifi Strength (%)' },
              position: 'right'
            }
          }
        }
      });

      return () => {
        if (voltageChartRef.current) {
          voltageChartRef.current.destroy();
          voltageChartRef.current = null;
        }
        if (rssiChartRef.current) {
          rssiChartRef.current.destroy();
          rssiChartRef.current = null;
        }
      };
    }, [data, batteryMin, batteryMax]);

    if (!data || data.length === 0) {
      return html`<div>No data</div>`;
    }

    return html`
      <div class="battery-charts">
        <div class="battery-chart"><canvas ref=${voltageCanvasRef}></canvas></div>
        <div class="battery-chart"><canvas ref=${rssiCanvasRef}></canvas></div>
      </div>
    `;
  }

  function StatsTable({ data }) {
    if (!data || data.length === 0) {
      return html`<div>No data</div>`;
    }
    const rows = [...data].slice(-100).reverse();
    return html`
      <div class="stats-table-wrapper">
        <table class="stats-table">
          <thead>
            <tr><th>Timestamp</th><th>Voltage (V)</th><th>RSSI (dBm)</th><th>WiFi (%)</th></tr>
          </thead>
          <tbody>
            ${rows.map(
              (entry, idx) => html`
                <tr key=${idx}>
                  <td>${formatDisplayTimestamp(entry.timestamp)}</td>
                  <td>${entry.battery_voltage != null ? Number(entry.battery_voltage).toFixed(2) : ''}</td>
                  <td>${entry.rssi != null ? entry.rssi : ''}</td>
                  <td>${entry.rssi != null ? Math.round(getWifiStrength(entry.rssi)) : ''}</td>
                </tr>
              `
            )}
          </tbody>
        </table>
        ${data.length > 100
          ? html`<div class="stats-table-note">Showing latest 100 of ${data.length} entries</div>`
          : null}
      </div>
    `;
  }

  function LogsTab({ logsList, onRefresh, autoRefresh, setAutoRefresh, loading, error }) {
    return html`
      <div class="section">
    <div class="section-heading">
      <h2>Server Logs</h2>
      <div class="log-controls">
      <label class="log-autorefresh">
        <input type="checkbox" checked=${autoRefresh} onChange=${(e) => setAutoRefresh(e.target.checked)} />
        <span>Auto-refresh</span>
      </label>
      <button type="button" onClick=${onRefresh}>Refresh</button>
      </div>
    </div>
        ${loading
          ? html`<div class="log-container" id="log-container">Loading...</div>`
          : error
          ? html`<div class="log-container" id="log-container">Error: ${error}</div>`
          : html`<div class="log-container" id="log-container">
              ${logsList.length === 0
                ? html`<div class="log-entry empty">No logs available.</div>`
                : logsList.map(
                    (log) => html`
                      <div class="log-entry" key=${log.id}>
                        <div class="log-entry-header">
                          <span class="timestamp">${formatDisplayTimestamp(log.timestamp)}</span>
                          <span class="context" title=${log.context}>${
                            log.context ? `[${log.context}]` : '(no context)'
                          }</span>
                        </div>
                        <pre class="info" title=${log.info}>${log.info || ''}</pre>
                      </div>
                    `
                  )}
            </div>`}
      </div>
    `;
  }

  function RotationTab({
    rotation,
    rotationOrder,
    activeRotationIds,
    activePlaylistTokens,
    toggleSelection,
    toggleForceOneBit,
    persistRotation,
    rotationFeedback,
    onReorder,
    rotationTarget,
    onTargetChange,
    devices,
    onAssignmentToggle,
    onCreatePlaylist,
    onDeletePlaylist,
    draftNames
  }) {
    const entries = rotation.entries || [];
    const [newPlaylistName, setNewPlaylistName] = useState('');
    const entriesById = useMemo(() => {
      const map = {};
      entries.forEach((e) => {
        map[e.id] = e;
      });
      return map;
    }, [entries]);

    const orderedIds = useMemo(() => {
      return rotationOrder && rotationOrder.length ? rotationOrder : entries.map((e) => e.id);
    }, [rotationOrder, entries]);

    const forceOneBitIds = useMemo(() => {
      const forced = new Set();
      (activePlaylistTokens || []).forEach((token) => {
        const parsed = parsePlaylistToken(token);
        if (parsed && parsed.id && parsed.mode === 'bmp') {
          forced.add(parsed.id);
        }
      });
      return forced;
    }, [activePlaylistTokens]);
    const normalizedDevices = useMemo(() => normalizeDevicesList(devices || []), [devices]);
    const namedPlaylists = rotation.playlists?.named || {};
    const bindings = rotation.playlists?.bindings || {};
    const playlistOptions = useMemo(() => {
      const seen = new Set(['default']);
      const base = [{ id: 'default', label: 'Default playlist' }];
      Object.keys(namedPlaylists || {}).sort().forEach((name) => {
        if (!seen.has(name)) {
          base.push({ id: name, label: name });
          seen.add(name);
        }
      });
      (Array.isArray(draftNames) ? draftNames : []).sort().forEach((name) => {
        if (name && !seen.has(name)) {
          base.push({ id: name, label: name });
          seen.add(name);
        }
      });
      return base;
    }, [namedPlaylists, draftNames]);
    const playlistLabel = rotationTarget === 'default' ? 'Default playlist' : rotationTarget;
    const isDraft = rotationTarget && rotationTarget !== 'default'
      && Array.isArray(draftNames)
      && draftNames.includes(rotationTarget)
      && !namedPlaylists[rotationTarget];
    const canDelete = rotationTarget && rotationTarget !== 'default' && (isDraft || namedPlaylists[rotationTarget]);

    return html`
      <div class="section">
        <div class="section-heading rotation-heading">
          <div>
            <h3>Playlist</h3>
            <p class="section-subtitle">Editing: ${playlistLabel}${isDraft ? ' (unsaved)' : ''}</p>
          </div>
          <div class="rotation-target-control">
            <div class="rotation-target-row">
              <label>
                <span>Target</span>
                <select value=${rotationTarget} onChange=${(e) => onTargetChange && onTargetChange(e.target.value)}>
                  ${playlistOptions.map((option) => html`<option value=${option.id}>${option.label}</option>`)}
                </select>
              </label>
              <button
                type="button"
                class="theme-toggle"
                onClick=${() => {
                  if (onCreatePlaylist && onCreatePlaylist(newPlaylistName)) {
                    setNewPlaylistName('');
                  }
                }}
                title="Add playlist"
              >
                <i class="fas fa-plus"></i>
                <span>Add</span>
              </button>
              <button
                type="button"
                class="theme-toggle"
                disabled=${!canDelete}
                onClick=${() => onDeletePlaylist && onDeletePlaylist(rotationTarget)}
                title="Delete playlist"
              >
                <i class="fas fa-trash"></i>
                <span>Delete</span>
              </button>
              <label>
                <span>New</span>
                <input
                  type="text"
                  value=${newPlaylistName}
                  placeholder="Playlist name"
                  onInput=${(e) => setNewPlaylistName(e.target.value)}
                />
              </label>
            </div>
          </div>
        </div>
        <p>Drag thumbnails to reorder. Toggle to enable/disable; disabled plugins stay in the grid but are skipped by the server.</p>
        <${PlaylistGrid}
          entriesById=${entriesById}
          orderedIds=${orderedIds}
          activeIds=${activeRotationIds}
          forceOneBitIds=${forceOneBitIds}
          onReorder=${onReorder}
          onToggleActive=${toggleSelection}
          onToggleForceOneBit=${toggleForceOneBit}
        />
        <div class="rotation-actions">
          <button type="button" disabled=${!activeRotationIds.length} onClick=${persistRotation}>Save Playlist</button>
          <span id="rotation-feedback" class="rotation-feedback">${rotationFeedback}</span>
        </div>
        <div class="rotation-assignment rotation-playlist-manager">
          <div class="section-heading">
            <h4>Device assignments</h4>
            <span class="section-subtitle">Select devices that should use this playlist</span>
          </div>
          ${rotationTarget === 'default'
            ? html`<p class="rotation-assignment-empty">Default playlist applies to unassigned devices.</p>`
            : isDraft
            ? html`<p class="rotation-assignment-empty">Save this playlist before assigning devices.</p>`
            : normalizedDevices.length
            ? html`
                <div class="rotation-assignment-list">
                  ${normalizedDevices.map((device) => {
                    const deviceId = device.device_id;
                    const assigned = bindings[deviceId] === rotationTarget;
                    const detail = assigned ? 'Assigned' : 'Follows default';
                    return html`
                      <label class="rotation-assignment-row" key=${deviceId}>
                        <input
                          type="checkbox"
                          checked=${assigned}
                          onChange=${(e) => onAssignmentToggle && onAssignmentToggle(deviceId, e.target.checked)}
                        />
                        <div class="rotation-assignment-info">
                          <strong>${device.friendly_name || deviceId}</strong>
                          <span class="section-subtitle">${deviceId}</span>
                        </div>
                        <span class="rotation-assignment-detail">${detail}</span>
                      </label>`;
                  })}
                </div>`
            : html`<p class="rotation-assignment-empty">No devices available for assignment yet.</p>`}
        </div>
      </div>
    `;
  }

  function PlaylistGrid({ entriesById, orderedIds, activeIds, forceOneBitIds, onReorder, onToggleActive, onToggleForceOneBit }) {
    const [draggingId, setDraggingId] = useState(null);
    const [dropIndex, setDropIndex] = useState(null);
    const uniqueOrderedIds = useMemo(() => Array.from(new Set(orderedIds)), [orderedIds]);

    const handleDropAt = (index) => {
      if (!draggingId) return;
      const next = uniqueOrderedIds.filter((id) => id !== draggingId);
      next.splice(index, 0, draggingId);
      onReorder(next);
      setDraggingId(null);
      setDropIndex(null);
    };

    const renderDropZone = (index) => html`
      <div
        key=${`dz-${index}`}
        class=${`playlist-dropzone${dropIndex === index ? ' active' : ''}`}
        onDragEnter=${(e) => {
          e.preventDefault();
          setDropIndex(index);
        }}
        onDragOver=${(e) => {
          e.preventDefault();
          setDropIndex(index);
        }}
        onDragLeave=${(e) => {
          e.preventDefault();
          setDropIndex((cur) => (cur === index ? null : cur));
        }}
        onDrop=${(e) => {
          e.preventDefault();
          handleDropAt(index);
        }}
      ></div>
    `;

    const items = [];
    items.push(renderDropZone(0));
    uniqueOrderedIds.forEach((id, idx) => {
      const entry = entriesById[id] || { label: id, id };
      const active = activeIds.includes(id);
      const forceOneBit = forceOneBitIds && typeof forceOneBitIds.has === 'function' ? forceOneBitIds.has(id) : false;
      items.push(html`
        <div
          key=${id}
          class=${`playlist-card${active ? '' : ' disabled'}`}
          draggable=${true}
          onDragStart=${(e) => {
            setDraggingId(id);
            if (e.dataTransfer) {
              e.dataTransfer.effectAllowed = 'move';
              e.dataTransfer.setData('text/plain', id);
            }
          }}
          onDragEnd=${() => {
            setDraggingId(null);
            setDropIndex(null);
          }}
        >
          <img class="playlist-card-thumb" src=${entry.url_png || entry.url_bmp} alt=${entry.label} loading="lazy" decoding="async" />
          <div class="playlist-card-label">${entry.label || entry.id}</div>
          <label class="playlist-card-toggle">
            <input type="checkbox" checked=${active} onChange=${() => onToggleActive(id)} />
            <span>${active ? 'On' : 'Off'}</span>
          </label>
          <label class="playlist-card-toggle">
            <input
              type="checkbox"
              disabled=${!active}
              checked=${forceOneBit}
              onChange=${(e) => onToggleForceOneBit && onToggleForceOneBit(id, e.target.checked)}
            />
            <span>Force 1-bit</span>
          </label>
        </div>
      `);
      items.push(renderDropZone(idx + 1));
    });

    return html`<div class="playlist-grid">${items}</div>`;
  }

  function getWifiStrength(rssi) {
    if (rssi <= -100) {
      return 0;
    }
    if (rssi >= -50) {
      return 100;
    }
    return 2 * (rssi + 100);
  }

  render(html`<${App} />`, document.getElementById('app-root'));
})();
