from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

from backend.infrastructure.portfolio_profile_store import PortfolioProfileStore


class PortfolioProfileService:
    """Service wrapper for portfolio profile persistence operations."""

    def __init__(
        self,
        store: PortfolioProfileStore | None = None,
        data_dir: str | None = None,
    ) -> None:
        self.store = store or PortfolioProfileStore(data_dir=data_dir)

    def get_all_profiles(self) -> List[str]:
        data = self.store.read_all()
        return list(data.keys())

    def get_profile(self, name: str) -> Dict[str, Any] | None:
        data = self.store.read_all()
        profile = data.get(name)
        if isinstance(profile, dict):
            return profile
        return None

    def save_profile(self, name: str, config: Dict[str, Any]) -> bool:
        data = self.store.read_all()
        payload = deepcopy(config)
        payload["last_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data[name] = payload
        self.store.write_all(data)
        return True

    def delete_profile(self, name: str) -> bool:
        data = self.store.read_all()
        if name not in data:
            return False
        del data[name]
        self.store.write_all(data)
        return True
