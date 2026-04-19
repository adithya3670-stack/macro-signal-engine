from __future__ import annotations

from typing import Any, Dict, Optional

from backend.domain.dto import SnapshotCreateRequest
from backend.services.model_snapshot_service import ModelSnapshotService


class DeepLearningSnapshotLifecycle:
    """Snapshot lifecycle facade for training/inference consumers."""

    def __init__(self, snapshot_service: Optional[ModelSnapshotService] = None) -> None:
        self.snapshot_service = snapshot_service or ModelSnapshotService()

    def create(self, tag: str, description: str = "") -> Dict[str, Any]:
        req = SnapshotCreateRequest(tag=tag, name=tag, description=description)
        return self.snapshot_service.create_snapshot(req)

    def list(self):
        return self.snapshot_service.list_snapshots()

    def restore(self, snapshot_id: str):
        return self.snapshot_service.restore_snapshot(snapshot_id)
