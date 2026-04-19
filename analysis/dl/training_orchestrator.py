from __future__ import annotations

from typing import Any, Dict, Optional

from analysis.deep_learning_model import DLMacroModel


class DeepLearningTrainingOrchestrator:
    """Service boundary for training orchestration."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = model_dir

    def _model(self) -> DLMacroModel:
        return DLMacroModel(model_dir=self.model_dir) if self.model_dir else DLMacroModel()

    def train_all(
        self,
        model_type: str,
        epochs: int,
        **kwargs: Any,
    ) -> Any:
        model = self._model()
        return model.train_all_models(model_type=model_type, epochs=epochs, **kwargs)

    def train_master_bundle(self, epochs: int = 20) -> Dict[str, Any]:
        model = self._model()
        outputs = {}
        for model_type in ["lstm", "transformer", "nbeats"]:
            outputs[model_type] = model.train_all_models(
                model_type=model_type,
                epochs=epochs,
                train_cutoff_date=None,
            )
        return outputs
