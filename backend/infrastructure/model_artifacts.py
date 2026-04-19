from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from backend.domain.dto import ModelSourceSelection


class ModelArtifactResolver:
    """Centralized artifact/path resolver without changing folder layout."""

    MODEL_ROOTS = (
        "models_dl",
        "models_price_3d",
        "models_price_1w",
        "models_price_1m",
        "models_regime",
        "MasterDl",
        os.path.join("models", "holdout_dl"),
        os.path.join("models", "holdout_price_3d"),
        os.path.join("models", "holdout_price_1w"),
        os.path.join("models", "holdout_price_1m"),
    )

    def __init__(self, base_dir: str | None = None) -> None:
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            self.base_dir = Path(__file__).resolve().parents[2]
        self.snapshots_dir = self.base_dir / "snapshots"

    def list_holdout_dl_folders(self) -> List[str]:
        holdout_dir = self.base_dir / "models" / "holdout_dl"
        if not holdout_dir.exists():
            return []
        return sorted(
            [entry.name for entry in holdout_dir.iterdir() if entry.is_dir()],
        )

    def resolve_model_source(
        self,
        model_category: str | None,
        model_year: str | None,
        dl_folder: str | None,
        model_type: str,
    ) -> ModelSourceSelection:
        category = (model_category or "dl").lower().strip()
        year = str(model_year).strip() if model_year not in {None, ""} else None
        folder = str(dl_folder).strip() if dl_folder not in {None, ""} else "default"

        if folder not in {"default", "rolling_master"}:
            year = folder.split("_", 1)[1] if folder.startswith("dl_") else folder
            category = "dl"

        if model_type.startswith("dl_"):
            year = model_type.split("_", 1)[1]
            category = "dl"

        if folder == "rolling_master":
            return ModelSourceSelection(
                source="rolling_master",
                category=category,
                model_type=model_type,
                year=None,
                model_dir=str(self.base_dir / "MasterDl"),
                label="Rolling Master DL",
            )

        if year and str(year).lower() != "latest":
            model_dir = self.base_dir / "models" / "holdout_dl" / str(year)
            return ModelSourceSelection(
                source="holdout",
                category=category,
                model_type=model_type,
                year=str(year),
                model_dir=str(model_dir),
                label=f"{category.upper()} {year}",
            )

        return ModelSourceSelection(
            source="live",
            category=category,
            model_type=model_type,
            year=None,
            model_dir=str(self.base_dir / "models_dl"),
            label="Live Auto-Pilot",
        )

    def collect_model_root_stats(self) -> List[Dict[str, object]]:
        stats: List[Dict[str, object]] = []
        for rel in self.MODEL_ROOTS:
            root = self.base_dir / rel
            if not root.exists():
                continue
            files_count = 0
            for _path, _dirs, files in os.walk(root):
                files_count += len(files)
            stats.append(
                {
                    "name": str(rel),
                    "path": str(root),
                    "files_count": files_count,
                }
            )
        return stats

    def ensure_snapshots_dir(self) -> Path:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        return self.snapshots_dir

    def to_dict(self, selection: ModelSourceSelection) -> Dict[str, object]:
        return asdict(selection)
