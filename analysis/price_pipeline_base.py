from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class PriceHorizonPipeline(ABC):
    """Common contract for 3d/1w/1m price-regression managers."""

    horizon_name: str

    @abstractmethod
    def refresh_feature_cache(self) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def train_holdout_pipeline(self, holdout_year: int, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict_latest(self, year: Optional[int] = None):
        raise NotImplementedError

    @abstractmethod
    def load_metrics(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        raise NotImplementedError

    @abstractmethod
    def load_feature_registry(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_model_metadata(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_model_scaler(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        raise NotImplementedError
