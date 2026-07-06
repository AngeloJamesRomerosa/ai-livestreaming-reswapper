from dataclasses import dataclass, field
from typing import Optional, Set

@dataclass
class Session:
    sid: str
    face_path: Optional[str] = None
    active: bool = False
    latest_frame: Optional[bytes] = None
    latest_frame_ts: float = 0.0
    viewers: Set = field(default_factory=set)

_sessions: dict = {}

def create(sid: str) -> Session:
    s = Session(sid=sid)
    _sessions[sid] = s
    return s

def get(sid: str) -> Optional[Session]:
    return _sessions.get(sid)

def remove(sid: str):
    _sessions.pop(sid, None)
