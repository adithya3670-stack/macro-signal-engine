from __future__ import annotations

from typing import Any, Dict

from backend.infrastructure.portfolio_profile_store import PortfolioProfileStore
from backend.services.portfolio_profile_service import PortfolioProfileService


class PortfolioManager:
    """
    Backward-compatible adapter for legacy analysis imports.
    Delegates profile persistence to backend infrastructure + service layers.
    """

    def __init__(self, data_dir: str | None = None):
        self.store = PortfolioProfileStore(data_dir=data_dir)
        self.service = PortfolioProfileService(store=self.store)
        self.data_dir = self.store.data_dir
        self.file_path = self.store.file_path

    def _ensure_file_exists(self) -> None:
        self.store.ensure_file_exists()

    def _load_data(self) -> Dict[str, Dict[str, Any]]:
        return self.store.read_all()

    def _save_data(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.store.write_all(data)

    def get_all_profiles(self):
        return self.service.get_all_profiles()

    def get_profile(self, name):
        return self.service.get_profile(name)

    def save_profile(self, name, config):
        return self.service.save_profile(name, config)

    def delete_profile(self, name):
        return self.service.delete_profile(name)
