from __future__ import annotations

from typing import Dict, Optional

from analysis.price_1m_regression import Price1MRegressionManager
from analysis.price_1w_regression import Price1WRegressionManager
from analysis.price_3d_regression import Price3DRegressionManager
from analysis.price_pipeline_base import PriceHorizonPipeline


class Price3DPipelineAdapter(PriceHorizonPipeline):
    horizon_name = "3d"

    def __init__(self) -> None:
        self.manager = Price3DRegressionManager()

    def refresh_feature_cache(self) -> Dict[str, object]:
        return self.manager.refresh_feature_cache()

    def train_holdout_pipeline(self, holdout_year: int, **kwargs):
        return self.manager.train_holdout_pipeline(holdout_year=holdout_year, **kwargs)

    def predict_latest(self, year: Optional[int] = None):
        return self.manager.predict_latest(year=year) if year else self.manager.predict_latest()

    def load_metrics(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        return self.manager.load_metrics(holdout_year=holdout_year, model_dir=model_dir)

    def load_feature_registry(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        return self.manager.load_feature_registry(holdout_year=holdout_year, model_dir=model_dir)

    def load_model_metadata(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        return self.manager.load_model_metadata(
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def load_model_scaler(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        return self.manager.load_model_scaler(
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )


class Price1WPipelineAdapter(PriceHorizonPipeline):
    horizon_name = "1w"

    def __init__(self) -> None:
        self.manager = Price1WRegressionManager()

    def refresh_feature_cache(self) -> Dict[str, object]:
        return self.manager.refresh_feature_cache()

    def train_holdout_pipeline(self, holdout_year: int, **kwargs):
        return self.manager.train_holdout_pipeline(holdout_year=holdout_year, **kwargs)

    def predict_latest(self, year: Optional[int] = None):
        return self.manager.predict_latest(year=year) if year else self.manager.predict_latest()

    def load_metrics(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        return self.manager.load_metrics(holdout_year=holdout_year, model_dir=model_dir)

    def load_feature_registry(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        return self.manager.load_feature_registry(holdout_year=holdout_year, model_dir=model_dir)

    def load_model_metadata(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        return self.manager.load_model_metadata(
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def load_model_scaler(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        return self.manager.load_model_scaler(
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )


class Price1MPipelineAdapter(PriceHorizonPipeline):
    horizon_name = "1m"

    def __init__(self) -> None:
        self.manager = Price1MRegressionManager()

    def refresh_feature_cache(self) -> Dict[str, object]:
        return self.manager.refresh_feature_cache()

    def train_holdout_pipeline(self, holdout_year: int, **kwargs):
        return self.manager.train_holdout_pipeline(holdout_year=holdout_year, **kwargs)

    def predict_latest(self, year: Optional[int] = None):
        return self.manager.predict_latest(year=year) if year else self.manager.predict_latest()

    def load_metrics(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        return self.manager.load_metrics(holdout_year=holdout_year, model_dir=model_dir)

    def load_feature_registry(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None):
        return self.manager.load_feature_registry(holdout_year=holdout_year, model_dir=model_dir)

    def load_model_metadata(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        return self.manager.load_model_metadata(
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def load_model_scaler(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        return self.manager.load_model_scaler(
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )
