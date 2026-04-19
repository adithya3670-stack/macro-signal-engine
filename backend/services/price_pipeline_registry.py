from __future__ import annotations

from typing import Dict

from analysis.price_pipeline_adapters import (
    Price1MPipelineAdapter,
    Price1WPipelineAdapter,
    Price3DPipelineAdapter,
)
from analysis.price_pipeline_base import PriceHorizonPipeline


class PricePipelineRegistry:
    """Unified access point for 3d/1w/1m pipeline adapters."""

    def __init__(self) -> None:
        self._pipelines: Dict[str, PriceHorizonPipeline] = {
            "3d": Price3DPipelineAdapter(),
            "1w": Price1WPipelineAdapter(),
            "1m": Price1MPipelineAdapter(),
        }

    def get(self, horizon: str) -> PriceHorizonPipeline:
        key = str(horizon).strip().lower()
        if key not in self._pipelines:
            raise KeyError(f"Unsupported horizon: {horizon}")
        return self._pipelines[key]

    def horizons(self):
        return sorted(self._pipelines.keys())
