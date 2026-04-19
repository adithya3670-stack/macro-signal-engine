from __future__ import annotations

from typing import Any, Dict, Optional

from analysis.deep_learning_model import DLMacroModel


class DeepLearningInferenceService:
    """Service boundary for inference/read-only model operations."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = model_dir

    def predict_latest(self, model_type: str = "lstm") -> Dict[str, Any]:
        model = DLMacroModel(model_dir=self.model_dir) if self.model_dir else DLMacroModel()
        return model.predict_latest(model_type=model_type)
