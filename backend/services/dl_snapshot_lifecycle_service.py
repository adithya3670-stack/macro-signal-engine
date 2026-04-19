from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from analysis.dl.snapshot_store import create_model_snapshot, list_model_snapshots, restore_model_snapshot


class DLSnapshotLifecycleService:
    """Snapshot + config lifecycle boundary for DL artifacts."""

    CONFIG_FILENAME = "dl_config.json"

    def load_config(self, version_dir: str) -> Dict[str, Any]:
        config_path = os.path.join(version_dir, self.CONFIG_FILENAME)
        if not os.path.exists(config_path):
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    def save_config(self, version_dir: str, config: Dict[str, Any]) -> str:
        os.makedirs(version_dir, exist_ok=True)
        config_path = os.path.join(version_dir, self.CONFIG_FILENAME)
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=4)
        return config_path

    def create_snapshot(self, model_dir: str, tag: Optional[str] = None) -> Dict[str, Any]:
        return create_model_snapshot(model_dir=model_dir, tag=tag)

    def list_snapshots(self, model_dir: str) -> List[Dict[str, Any]]:
        return list_model_snapshots(model_dir=model_dir)

    def restore_snapshot(self, model_dir: str, snapshot_id: str) -> bool:
        return bool(restore_model_snapshot(model_dir=model_dir, snapshot_id=snapshot_id))
