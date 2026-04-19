from __future__ import annotations

import shutil
from pathlib import Path


class BacktestModelAdminService:
    """Service for backtest artifact administration operations."""

    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parents[2]

    def _resolve_target_dir(self, category: str, year: str) -> Path:
        category_norm = str(category).strip().lower()
        year_norm = str(year).strip()
        if category_norm == "dl":
            return self.base_dir / "models" / "holdout_dl" / year_norm
        return self.base_dir / "models" / "holdout" / year_norm

    def delete_model(self, category: str, year: str) -> bool:
        target_dir = self._resolve_target_dir(category=category, year=year)
        if not target_dir.exists() or not target_dir.is_dir():
            return False
        shutil.rmtree(target_dir)
        return True
