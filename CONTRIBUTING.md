# Contributing

Thanks for contributing.

## Development Baseline

- Python 3.11 recommended (matches CI)
- Install runtime + test locks before running checks
- Keep commits focused and reproducible

## Before Opening A PR

Run these locally:

```powershell
python -m compileall -q analysis backtesting routes services data config backend app.py
python -c "from app import app; assert app is not None; print('import-ok')"
python scripts/generate_api_contract_manifest.py --check contracts/api_contract_manifest.json
pytest -q tests/test_api_contract_manifest.py tests/test_model_snapshot_service.py tests/test_automation_config_store.py tests/test_backtest_orchestration_service.py tests/test_startup_import_health.py tests/test_dl_decomposition.py
```

If your change touches portfolio/backtest behavior, also run:

```powershell
pytest -q tests/test_portfolio_backtest_regression_parity.py
```

## Artifact And Data Policy

- Do not commit model weights, checkpoints, or large generated datasets.
- Keep example fixtures small and deterministic.
- Use environment variables for secrets; never commit plaintext credentials.

## PR Quality Bar

- Preserve existing API contracts unless explicitly versioned.
- Add or update tests for behavior changes.
- Keep route handlers thin; place business logic in backend services.
