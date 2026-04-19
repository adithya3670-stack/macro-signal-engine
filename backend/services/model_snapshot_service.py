from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.domain.dto import SnapshotCreateRequest, SnapshotRecord
from backend.infrastructure.model_artifacts import ModelArtifactResolver
from backend.shared.http import ServiceError


class ModelSnapshotService:
    """
    Snapshot metadata service that preserves existing artifact layout.

    Snapshots are stored as JSON manifests under /snapshots for strict backward
    compatibility with prior route behavior while exposing the new frontend alias
    contract fields (`tag`, `display_time`, `files_count`).
    """

    def __init__(
        self,
        artifact_resolver: Optional[ModelArtifactResolver] = None,
        cache_ref: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.artifacts = artifact_resolver or ModelArtifactResolver()
        self.cache_ref = cache_ref if isinstance(cache_ref, dict) else {}

    def _snapshot_file(self, snapshot_id: str) -> Path:
        return self.artifacts.ensure_snapshots_dir() / f"{snapshot_id}.json"

    @staticmethod
    def _json_default(value):
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return str(value)
        return str(value)

    def _load_snapshot_file(self, path: Path) -> SnapshotRecord:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "tag" not in payload and "name" in payload:
            payload["tag"] = payload["name"]
        if "display_time" not in payload:
            payload["display_time"] = payload.get("timestamp", "")
        if "files_count" not in payload:
            payload["files_count"] = 0
        return SnapshotRecord.from_dict(payload)

    def create_snapshot(self, request: SnapshotCreateRequest) -> Dict[str, Any]:
        timestamp = datetime.now()
        snapshot_id = timestamp.strftime("%Y%m%d_%H%M%S")
        model_roots = self.artifacts.collect_model_root_stats()
        files_count = sum(int(item.get("files_count", 0)) for item in model_roots)

        record = SnapshotRecord(
            id=snapshot_id,
            timestamp=timestamp.isoformat(),
            display_time=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            tag=request.tag or request.name or "Snapshot",
            description=request.description or "",
            files_count=files_count,
            model_roots=model_roots,
            data=self.cache_ref.get("drivers", {}) if self.cache_ref else {},
        )

        # Normalize to plain JSON-safe primitives before writing.
        record.data = json.loads(json.dumps(record.data or {}, default=self._json_default))

        path = self._snapshot_file(snapshot_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2)

        return record.to_dict()

    def list_snapshots(self) -> List[Dict[str, Any]]:
        snapshot_dir = self.artifacts.ensure_snapshots_dir()
        records: List[SnapshotRecord] = []
        for path in snapshot_dir.glob("*.json"):
            try:
                records.append(self._load_snapshot_file(path))
            except Exception:
                continue
        records.sort(key=lambda rec: rec.timestamp, reverse=True)
        return [rec.to_dict() for rec in records]

    def restore_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        path = self._snapshot_file(snapshot_id)
        if not path.exists():
            raise ServiceError(
                message=f"Snapshot '{snapshot_id}' not found.",
                status_code=404,
                code="snapshot_not_found",
            )
        record = self._load_snapshot_file(path)
        if self.cache_ref is not None and isinstance(record.data, dict):
            self.cache_ref["drivers"] = record.data
        return record.to_dict()
