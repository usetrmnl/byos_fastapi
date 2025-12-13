import datetime
import json
from typing import Dict, List, Optional, Tuple

from sqlalchemy import DateTime, Float, Integer, String, Text, UniqueConstraint, create_engine, inspect, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from . import config

logger = config.logger


def _utcnow() -> datetime.datetime:
    """Return a timezone-aware UTC timestamp for SQLAlchemy defaults."""
    return datetime.datetime.now(datetime.timezone.utc)


def _database_url() -> str:
    return f"sqlite:///{config.DATABASE_PATH}"


engine = create_engine(_database_url(), connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False)
SessionLocal.configure(bind=engine)


class Base(DeclarativeBase):
    pass


class BatteryStatus(Base):
    __tablename__ = "battery_status"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow)
    voltage: Mapped[float] = mapped_column(Float)
    rssi: Mapped[int] = mapped_column(Integer)

    def __repr__(self) -> str:
        return f"<BatteryStatus(timestamp={self.timestamp}, voltage={self.voltage}, rssi={self.rssi})>"


class LogEntry(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow)
    context: Mapped[str] = mapped_column(String)
    info: Mapped[str] = mapped_column(String)

    def __repr__(self) -> str:
        return f"<LogEntry(timestamp={self.timestamp}, context={self.context}, info={self.info})>"


class RotationPlaylist(Base):
    __tablename__ = "rotation_playlists"
    __table_args__ = (
        UniqueConstraint("name", "device_id", name="uq_rotation_playlists_name_device"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, default="default")
    device_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    selected_ids: Mapped[str] = mapped_column(Text, default="[]")
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return f"<RotationPlaylist(name={self.name}, device_id={self.device_id})>"


class DevicePlaylistBinding(Base):
    __tablename__ = "device_playlist_bindings"

    device_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    playlist_name: Mapped[str] = mapped_column(String, default="default")
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return f"<DevicePlaylistBinding(device_id={self.device_id}, playlist_name={self.playlist_name})>"


class DeviceState(Base):
    __tablename__ = "device_states"

    device_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    rotation_version: Mapped[int] = mapped_column(Integer, default=0)
    rotation_index: Mapped[int] = mapped_column(Integer, default=-1)
    last_entry_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rotation_hash_order: Mapped[str] = mapped_column(Text, default="[]")
    current_plugin_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<DeviceState(device_id={self.device_id}, rotation_version={self.rotation_version}, "
            f"rotation_index={self.rotation_index})>"
        )


class ConfigEntry(Base):
    __tablename__ = "config_entries"

    key: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    value: Mapped[str] = mapped_column(String)

    def __repr__(self) -> str:
        return f"<ConfigEntry(key={self.key}, value={self.value})>"


class DeviceProfile(Base):
    __tablename__ = "device_profiles"

    device_id: Mapped[str] = mapped_column(String, primary_key=True, index=True)
    friendly_name: Mapped[str] = mapped_column(String, default="")
    refresh_interval: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    time_zone: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)
    last_seen: Mapped[datetime.datetime] = mapped_column(DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return f"<DeviceProfile(device_id={self.device_id}, friendly_name={self.friendly_name})>"


def reconfigure_engine() -> None:
    """Recreate the SQLite engine to follow the current config path."""
    global engine
    new_engine = create_engine(_database_url(), connect_args={"check_same_thread": False})
    if engine:
        engine.dispose()
    engine = new_engine
    SessionLocal.configure(bind=engine)


def init_db() -> None:
    """Initialize the database by creating all tables."""
    reconfigure_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_device_state_schema()


def _ensure_device_state_schema() -> None:
    try:
        inspector = inspect(engine)
        columns = {column['name'] for column in inspector.get_columns('device_states')}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to inspect device_states schema: %s", exc)
        return
    if 'current_plugin_id' in columns:
        return
    try:
        with engine.begin() as connection:
            connection.execute(text('ALTER TABLE device_states ADD COLUMN current_plugin_id VARCHAR'))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to add current_plugin_id column: %s", exc)


def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_battery_status(voltage: float, rssi: int) -> BatteryStatus:
    """Add a new battery status entry."""
    with SessionLocal() as db:
        status = BatteryStatus(voltage=voltage, rssi=rssi)
        db.add(status)
        db.commit()
        db.refresh(status)
        return status


def get_battery_history(
    limit: int = 30,
    from_date: Optional[datetime.datetime] = None,
    to_date: Optional[datetime.datetime] = None
) -> List[BatteryStatus]:
    """Get battery history with optional filtering."""
    with SessionLocal() as db:
        query = select(BatteryStatus).order_by(BatteryStatus.timestamp.desc())

        if from_date and to_date:
            query = query.where(BatteryStatus.timestamp >= from_date, BatteryStatus.timestamp <= to_date)

        if limit:
            query = query.limit(limit)

        result = db.execute(query)
        return list(result.scalars().all())


def add_log_entry(context: str, info: str) -> LogEntry:
    """Add a new log entry."""
    with SessionLocal() as db:
        log = LogEntry(context=context, info=info)
        db.add(log)
        db.commit()
        db.refresh(log)
        return log


def get_logs(limit: int = 20) -> List[LogEntry]:
    """Get the latest log entries ordered oldest-to-newest."""
    with SessionLocal() as db:
        query = select(LogEntry).order_by(LogEntry.timestamp.desc()).limit(limit)
        result = db.execute(query)
        logs = list(result.scalars().all())
        return sorted(logs, key=lambda entry: entry.timestamp)


def get_logs_after(last_id: int, limit: int = 50) -> List[LogEntry]:
    """Get log entries with an ID greater than the provided cursor."""
    with SessionLocal() as db:
        query = (
            select(LogEntry)
            .where(LogEntry.id > last_id)
            .order_by(LogEntry.timestamp.asc())
        )
        result = db.execute(query)
        return list(result.scalars().all())


def _serialize_list(values: Optional[List[str]]) -> str:
    return json.dumps(values or [])


def _deserialize_list(payload: Optional[str]) -> List[str]:
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def get_rotation_playlist(device_id: Optional[str] = None, name: str = "default") -> Optional[List[str]]:
    with SessionLocal() as db:
        stmt = (
            select(RotationPlaylist)
            .where(RotationPlaylist.name == name)
            .where(RotationPlaylist.device_id == device_id)
        )
        result = db.execute(stmt).scalars().first()
        if result is None:
            return None
        return _deserialize_list(result.selected_ids)


def save_rotation_playlist(
    selected_ids: List[str],
    device_id: Optional[str] = None,
    name: str = "default"
) -> None:
    payload = _serialize_list(selected_ids)
    timestamp = _utcnow()
    with SessionLocal() as db:
        stmt = (
            select(RotationPlaylist)
            .where(RotationPlaylist.name == name)
            .where(RotationPlaylist.device_id == device_id)
        )
        row = db.execute(stmt).scalars().first()
        if row is None:
            row = RotationPlaylist(name=name, device_id=device_id, selected_ids=payload, updated_at=timestamp)
            db.add(row)
        else:
            row.selected_ids = payload
            row.updated_at = timestamp
        db.commit()


def list_device_playlists(name: str = "default") -> List[Tuple[str, List[str]]]:
    with SessionLocal() as db:
        stmt = (
            select(RotationPlaylist)
            .where(RotationPlaylist.name == name)
            .where(RotationPlaylist.device_id.isnot(None))
        )
        rows = db.execute(stmt).scalars().all()
        return [
            (row.device_id, _deserialize_list(row.selected_ids))
            for row in rows
        ]


def delete_rotation_playlist(device_id: str, name: str = "default") -> None:
    with SessionLocal() as db:
        stmt = (
            select(RotationPlaylist)
            .where(RotationPlaylist.name == name)
            .where(RotationPlaylist.device_id == device_id)
        )
        row = db.execute(stmt).scalars().first()
        if row is None:
            return
        db.delete(row)
        db.commit()


def list_named_rotation_playlists() -> List[Tuple[str, List[str]]]:
    with SessionLocal() as db:
        stmt = (
            select(RotationPlaylist)
            .where(RotationPlaylist.device_id.is_(None))
            .where(RotationPlaylist.name != 'default')
        )
        rows = db.execute(stmt).scalars().all()
        return [(row.name, _deserialize_list(row.selected_ids)) for row in rows]


def delete_named_rotation_playlist(name: str) -> None:
    with SessionLocal() as db:
        stmt = (
            select(RotationPlaylist)
            .where(RotationPlaylist.name == name)
            .where(RotationPlaylist.device_id.is_(None))
        )
        row = db.execute(stmt).scalars().first()
        if row is None:
            return
        db.delete(row)
        db.commit()


def list_device_playlist_bindings() -> List[Tuple[str, str]]:
    with SessionLocal() as db:
        stmt = select(DevicePlaylistBinding)
        rows = db.execute(stmt).scalars().all()
        return [(row.device_id, row.playlist_name) for row in rows]


def get_device_playlist_binding(device_id: str) -> Optional[str]:
    with SessionLocal() as db:
        row = db.get(DevicePlaylistBinding, device_id)
        if row is None:
            return None
        return row.playlist_name


def set_device_playlist_binding(device_id: str, playlist_name: str) -> None:
    timestamp = _utcnow()
    with SessionLocal() as db:
        row = db.get(DevicePlaylistBinding, device_id)
        if row is None:
            row = DevicePlaylistBinding(device_id=device_id, playlist_name=playlist_name, updated_at=timestamp)
            db.add(row)
        else:
            row.playlist_name = playlist_name
            row.updated_at = timestamp
        db.commit()


def delete_device_playlist_binding(device_id: str) -> None:
    with SessionLocal() as db:
        row = db.get(DevicePlaylistBinding, device_id)
        if row is None:
            return
        db.delete(row)
        db.commit()


def get_device_state(device_id: str) -> Optional[Dict[str, object]]:
    with SessionLocal() as db:
        row = db.get(DeviceState, device_id)
        if row is None:
            return None
        return {
            'rotation_version': row.rotation_version,
            'rotation_index': row.rotation_index,
            'last_entry_hash': row.last_entry_hash,
            'rotation_hash_order': _deserialize_list(row.rotation_hash_order),
            'current_plugin_id': row.current_plugin_id
        }


def save_device_state(
    device_id: str,
    rotation_version: int,
    rotation_index: int,
    rotation_hash_order: List[str],
    last_entry_hash: Optional[str],
    current_plugin_id: Optional[str]
) -> None:
    timestamp = _utcnow()
    payload = _serialize_list(rotation_hash_order)
    with SessionLocal() as db:
        row = db.get(DeviceState, device_id)
        if row is None:
            row = DeviceState(
                device_id=device_id,
                rotation_version=rotation_version,
                rotation_index=rotation_index,
                last_entry_hash=last_entry_hash,
                rotation_hash_order=payload,
                current_plugin_id=current_plugin_id,
                updated_at=timestamp
            )
            db.add(row)
        else:
            row.rotation_version = rotation_version
            row.rotation_index = rotation_index
            row.last_entry_hash = last_entry_hash
            row.rotation_hash_order = payload
            row.current_plugin_id = current_plugin_id
            row.updated_at = timestamp
        db.commit()


def delete_device_state(device_id: str) -> None:
    with SessionLocal() as db:
        row = db.get(DeviceState, device_id)
        if row is None:
            return
        db.delete(row)
        db.commit()


def _profile_to_dict(profile: DeviceProfile) -> Dict[str, Optional[str]]:
    return {
        'device_id': profile.device_id,
        'friendly_name': profile.friendly_name,
        'refresh_interval': profile.refresh_interval,
        'time_zone': profile.time_zone,
        'last_seen': profile.last_seen
    }


def get_device_profile(device_id: str) -> Optional[Dict[str, Optional[str]]]:
    with SessionLocal() as db:
        profile = db.get(DeviceProfile, device_id)
        if profile is None:
            return None
        return _profile_to_dict(profile)


def ensure_device_profile(device_id: str) -> Dict[str, Optional[str]]:
    timestamp = _utcnow()
    with SessionLocal() as db:
        profile = db.get(DeviceProfile, device_id)
        if profile is None:
            profile = DeviceProfile(
                device_id=device_id,
                created_at=timestamp,
                updated_at=timestamp,
                last_seen=timestamp
            )
            db.add(profile)
            db.commit()
            db.refresh(profile)
        return _profile_to_dict(profile)


def update_device_profile(
    device_id: str,
    *,
    friendly_name: Optional[str] = None,
    refresh_interval: Optional[int] = None,
    time_zone: Optional[str] = None
) -> Dict[str, Optional[str]]:
    timestamp = _utcnow()
    with SessionLocal() as db:
        profile = db.get(DeviceProfile, device_id)
        if profile is None:
            profile = DeviceProfile(device_id=device_id, created_at=timestamp)
            db.add(profile)
        if friendly_name is not None:
            profile.friendly_name = friendly_name
        if refresh_interval is not None:
            profile.refresh_interval = refresh_interval
        if time_zone is not None:
            profile.time_zone = time_zone
        profile.updated_at = timestamp
        db.commit()
        db.refresh(profile)
        return _profile_to_dict(profile)


def touch_device_last_seen(device_id: str) -> None:
    timestamp = _utcnow()
    with SessionLocal() as db:
        profile = db.get(DeviceProfile, device_id)
        if profile is None:
            profile = DeviceProfile(device_id=device_id, created_at=timestamp)
            db.add(profile)
        profile.last_seen = timestamp
        db.commit()


def list_device_profiles() -> List[Dict[str, Optional[str]]]:
    with SessionLocal() as db:
        rows = db.execute(select(DeviceProfile).order_by(DeviceProfile.device_id)).scalars().all()
        return [_profile_to_dict(row) for row in rows]


def save_config_entry(key: str, value: str) -> None:
    """Persist a configuration key/value pair for future restarts."""
    with SessionLocal() as db:
        entry = db.get(ConfigEntry, key)
        if entry is None:
            entry = ConfigEntry(key=key, value=str(value))
            db.add(entry)
        else:
            entry.value = str(value)
        db.commit()


def load_config_entries() -> Dict[str, str]:
    """Return all persisted configuration entries as a key/value mapping."""
    with SessionLocal() as db:
        stmt = select(ConfigEntry)
        rows = db.execute(stmt).scalars().all()
        return {row.key: row.value for row in rows}
