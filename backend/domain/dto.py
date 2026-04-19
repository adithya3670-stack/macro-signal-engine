from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class SnapshotCreateRequest:
    tag: str = "Snapshot"
    name: str = "Snapshot"
    description: str = ""


@dataclass(slots=True)
class SnapshotRestoreRequest:
    snapshot_id: str


@dataclass(slots=True)
class SnapshotRecord:
    id: str
    timestamp: str
    display_time: str
    tag: str
    description: str = ""
    files_count: int = 0
    model_roots: List[Dict[str, Any]] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "display_time": self.display_time,
            "tag": self.tag,
            "description": self.description,
            "files_count": self.files_count,
            "model_roots": self.model_roots,
            "data": self.data or {},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SnapshotRecord":
        return cls(
            id=str(payload.get("id", "")),
            timestamp=str(payload.get("timestamp", "")),
            display_time=str(payload.get("display_time", payload.get("timestamp", ""))),
            tag=str(payload.get("tag", payload.get("name", "Snapshot"))),
            description=str(payload.get("description", "")),
            files_count=int(payload.get("files_count", 0)),
            model_roots=list(payload.get("model_roots", [])),
            data=payload.get("data") if isinstance(payload.get("data"), dict) else {},
        )


@dataclass(slots=True)
class ModelSourceSelection:
    source: str
    category: str
    model_type: str
    year: Optional[str] = None
    model_dir: Optional[str] = None
    label: str = "Live Auto-Pilot"


@dataclass(slots=True)
class HoldoutSplitRequest:
    cutoff_year: int = 2023


@dataclass(slots=True)
class HoldoutRunRequest:
    cutoff_year: int = 2023
    model_type: str = "ml"
    dl_mode: str = "balanced"
    use_existing: bool = False


@dataclass(slots=True)
class PortfolioProfileSaveRequest:
    name: str
    config: Dict[str, Any]


@dataclass(slots=True)
class BacktestModelDeleteRequest:
    category: str
    year: str
