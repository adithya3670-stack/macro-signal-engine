from __future__ import annotations

import datetime
import glob
import json
import os
import shutil
from typing import Any, Dict, List, Optional


def create_model_snapshot(model_dir: str, tag: Optional[str] = None) -> Dict[str, Any]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_id = f"snap_{timestamp}"
    if tag:
        snapshot_id += f"_{tag}"

    snapshot_dir = os.path.join(model_dir, "snapshots", snapshot_id)
    os.makedirs(snapshot_dir, exist_ok=True)

    files_to_copy = glob.glob(os.path.join(model_dir, "*.*"))
    count = 0
    for file_path in files_to_copy:
        if os.path.isfile(file_path):
            shutil.copy2(file_path, snapshot_dir)
            count += 1

    metadata = {
        "id": snapshot_id,
        "timestamp": timestamp,
        "display_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "files_count": count,
        "tag": tag or "Manual Save",
    }
    with open(os.path.join(snapshot_dir, "snapshot_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=4)
    return metadata


def list_model_snapshots(model_dir: str) -> List[Dict[str, Any]]:
    snapshots_root = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snapshots_root):
        return []

    snapshots: List[Dict[str, Any]] = []
    for folder_name in os.listdir(snapshots_root):
        full_path = os.path.join(snapshots_root, folder_name)
        if not os.path.isdir(full_path):
            continue
        meta_path = os.path.join(full_path, "snapshot_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                snapshots.append(json.load(handle))
        else:
            snapshots.append({"id": folder_name, "timestamp": "Unknown", "tag": "Legacy", "display_time": folder_name})

    snapshots.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return snapshots


def restore_model_snapshot(model_dir: str, snapshot_id: str) -> bool:
    snapshot_dir = os.path.join(model_dir, "snapshots", snapshot_id)
    if not os.path.exists(snapshot_dir):
        raise FileNotFoundError(f"Snapshot {snapshot_id} not found.")

    files = glob.glob(os.path.join(snapshot_dir, "*.*"))
    for file_path in files:
        file_name = os.path.basename(file_path)
        if file_name == "snapshot_meta.json":
            continue
        destination = os.path.join(model_dir, file_name)
        shutil.copy2(file_path, destination)
    return True
