from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class PortfolioProfileStorePaths:
    data_dir: Path
    file_path: Path


class PortfolioProfileStore:
    """Filesystem adapter for portfolio profile persistence."""

    DEFAULT_FILENAME = "portfolio_profiles.json"
    DEFAULT_PAYLOAD: Dict[str, Dict[str, Any]] = {
        "Default Strategy": {
            "universe": ["SP500", "Gold"],
            "strategy_config": {},
            "last_modified": "2024-01-01",
        }
    }

    def __init__(self, data_dir: str | Path | None = None, file_path: str | Path | None = None) -> None:
        self.paths = self._resolve_paths(data_dir=data_dir, file_path=file_path)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_file_exists()

    def _resolve_paths(
        self,
        data_dir: str | Path | None,
        file_path: str | Path | None,
    ) -> PortfolioProfileStorePaths:
        base_dir = Path(__file__).resolve().parents[2]
        if file_path is not None:
            resolved_file = Path(file_path).resolve()
            resolved_data = resolved_file.parent
        else:
            resolved_data = Path(data_dir).resolve() if data_dir is not None else (base_dir / "data")
            resolved_file = resolved_data / self.DEFAULT_FILENAME
        return PortfolioProfileStorePaths(data_dir=resolved_data, file_path=resolved_file)

    @property
    def data_dir(self) -> str:
        return str(self.paths.data_dir)

    @property
    def file_path(self) -> str:
        return str(self.paths.file_path)

    def ensure_file_exists(self) -> None:
        if self.paths.file_path.exists():
            return
        with self.paths.file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.DEFAULT_PAYLOAD, handle, indent=4)

    def read_all(self) -> Dict[str, Dict[str, Any]]:
        try:
            with self.paths.file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                normalized: Dict[str, Dict[str, Any]] = {}
                for key, value in payload.items():
                    if isinstance(value, dict):
                        normalized[str(key)] = value
                return normalized
            return {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def write_all(self, payload: Dict[str, Dict[str, Any]]) -> None:
        with self.paths.file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
