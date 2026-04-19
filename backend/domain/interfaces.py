from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class DataDomainService(Protocol):
    def load(self) -> Dict[str, Any]:
        ...

    def refresh(self, start_date: Optional[str], end_date: Optional[str]) -> Dict[str, Any]:
        ...


class BacktestDomainService(Protocol):
    def run_backtest_v2(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class PortfolioDomainService(Protocol):
    def run_portfolio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class PortfolioProfileDomainService(Protocol):
    def get_all_profiles(self) -> List[str]:
        ...

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        ...

    def save_profile(self, name: str, config: Dict[str, Any]) -> bool:
        ...

    def delete_profile(self, name: str) -> bool:
        ...


class ModelArtifactsDomainService(Protocol):
    def list_snapshots(self) -> List[Dict[str, Any]]:
        ...

    def create_snapshot(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def restore_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        ...


class RegimeDomainService(Protocol):
    def rebuild(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def predict_latest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class AutomationDomainService(Protocol):
    def get_public_config(self) -> Dict[str, Any]:
        ...

    def save_config(self, payload: Dict[str, Any]) -> None:
        ...


class BacktestModelAdminDomainService(Protocol):
    def delete_model(self, category: str, year: str) -> bool:
        ...
