from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, MutableMapping

from backend.services.dl_hyperparameter_search_service import DLHyperparameterSearchService


class DLTrainingOrchestrationService:
    """Coordinates holdout DL training/search modes with stable semantics."""

    ARCHITECTURES = ("lstm", "transformer", "nbeats")

    def __init__(
        self,
        hyperparameter_service: DLHyperparameterSearchService | None = None,
    ) -> None:
        self.hyperparameter_service = hyperparameter_service or DLHyperparameterSearchService()

    def train_holdout_mode(
        self,
        *,
        builder: Any,
        cutoff_date: str,
        dl_mode: str,
        dl_config: MutableMapping[str, Any],
        send_update: Callable[..., str],
    ) -> Iterator[str]:
        mode = str(dl_mode or "balanced").strip().lower()

        if mode == "quick":
            yield send_update(10, "Training Quick LSTM...", "Starting...")
            builder.train_all_models(
                model_type="lstm",
                train_cutoff_date=cutoff_date,
                epochs=5,
                config_dict=dict(dl_config),
            )
            return

        if mode == "balanced":
            yield send_update(10, "Training Balanced LSTM...", "In progress...")
            builder.train_all_models(
                model_type="lstm",
                train_cutoff_date=cutoff_date,
                epochs=30,
                config_dict=dict(dl_config),
            )
            return

        if mode in {"lite", "deep"}:
            search_iterations = 5 if mode == "lite" else 50
            train_epochs = 30 if mode == "lite" else 60
            optimize_label = "Running Quick Search (5 iters)..." if mode == "lite" else "Running Hyperparameter Search (50 iters)..."
            train_label = "Training (30 epochs)..." if mode == "lite" else "Deep Training (60 epochs)..."

            total_steps = len(self.ARCHITECTURES) * 2
            step_count = 0
            for arch in self.ARCHITECTURES:
                step_count += 1
                pct = 10 + int((step_count / total_steps) * 80)
                mode_suffix = " (Lite)" if mode == "lite" else ""
                yield send_update(
                    pct,
                    f"Optimizing {arch.upper()}{mode_suffix}",
                    optimize_label,
                )
                self.hyperparameter_service.optimize_architecture(
                    builder=builder,
                    model_type=arch,
                    iterations=search_iterations,
                    base_config=dl_config,
                )

                step_count += 1
                pct = 10 + int((step_count / total_steps) * 80)
                yield send_update(
                    pct,
                    f"Training {arch.upper()}{mode_suffix}",
                    train_label,
                )
                builder.train_all_models(
                    model_type=arch,
                    train_cutoff_date=cutoff_date,
                    epochs=train_epochs,
                    config_dict=dict(dl_config),
                )
            return

        # Backward-compatible fallback.
        yield send_update(10, "Training Balanced LSTM...", "Mode not recognized; using balanced.")
        builder.train_all_models(
            model_type="lstm",
            train_cutoff_date=cutoff_date,
            epochs=30,
            config_dict=dict(dl_config),
        )
