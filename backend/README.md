# Backend Layering (Refactor Baseline)

This package introduces explicit backend layers while preserving existing Flask routes and API contracts.

- `backend/domain`: typed DTOs + domain protocols
- `backend/services`: business/domain services
- `backend/infrastructure`: file-system artifact resolution
- `backend/shared`: cross-cutting helpers (error envelopes, normalization)

Current route adapters:

- `analysis/backtest_api.py` is a thin adapter delegating:
  `PortfolioSimulationService`, `PortfolioProfileService`, and `BacktestModelAdminService`
- `routes/backtest_holdout.py` delegates holdout split/run/list flows to `HoldoutBacktestService`
- `routes/analysis.py` delegates snapshot lifecycle to `ModelSnapshotService`
- `routes/training.py` exposes compatibility alias `/api/train/forecast_stream`
- `analysis/automation_manager.py` uses env-first typed automation config store

Infrastructure note:

- `PortfolioProfileStore` is the canonical adapter for `portfolio_profiles.json`.
- `analysis/portfolio_manager.py` is now a backward-compatible adapter facade.

Portfolio simulation decomposition:

- `PortfolioSimulationService` now orchestrates specialized services only.
- `ModelSourceSelectionService` handles live/holdout/rolling source selection.
- `SignalResolutionService` handles signal universe, blending, and alignment.
- `SimulationExecutionService` handles strategy + backtest/portfolio engine execution.
- `PortfolioRuntimeSupportService` handles date-window filtering, auto-correction, and metric recomputation.

Deep-learning holdout decomposition:

- `HoldoutBacktestService` keeps the route contract but delegates orchestration.
- `DLTrainingOrchestrationService` coordinates quick/balanced/lite/deep training modes.
- `DLHyperparameterSearchService` handles architecture-level optimization calls.
- `DLInferenceService` handles range inference, metrics loading, and weighted signal blending.
- `DLSnapshotLifecycleService` handles `dl_config.json` + snapshot lifecycle boundaries.
