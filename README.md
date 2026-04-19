# MacroEconomic

MacroEconomic is a Flask-based research and backtesting platform for macro-aware
portfolio simulation, holdout/model evaluation, and price-horizon forecasting.

This public repository follows a **source-only policy**:
- code, tests, contracts, and lightweight fixtures are versioned
- large model artifacts and full runtime datasets are intentionally excluded

## Why This Repo Is Structured This Way

The project emphasizes reproducible engineering:
- explicit backend service boundaries (`backend/services`)
- API contract snapshot checks (`contracts/api_contract_manifest.json`)
- focused CI gates for compile/import health and regression tests
- deterministic test fixtures under `tests/fixtures`

## Architecture At A Glance

- `app.py`: Flask app entrypoint and blueprint registration
- `routes/`: thin HTTP adapters
- `backend/`: typed DTOs, services, infrastructure adapters, shared utilities
- `analysis/`: model/training/inference orchestration modules
- `backtesting/`: simulation engines and strategy logic
- `contracts/`: API contract manifests
- `tests/`: unit/integration/regression test coverage

## Quickstart (Local)

### 1. Create environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies

Runtime + test stack:

```powershell
pip install -r requirements/runtime.lock.txt -r requirements/test.lock.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

Optional expanded stacks:
- `requirements/train.lock.txt`
- `requirements/research.lock.txt`

### 3. Configure automation secrets (optional)

Copy the template and set environment variables:

```powershell
Copy-Item data/automation_config.example.json data/automation_config.json
$env:MACRO_AUTO_EMAIL_PASSWORD="your-password"
```

### 4. Run app

```powershell
python app.py
```

## Validation Commands

Compile/import health:

```powershell
python -m compileall -q analysis backtesting routes services data config backend app.py
python -c "from app import app; assert app is not None; print('import-ok')"
```

API contract check:

```powershell
python scripts/generate_api_contract_manifest.py --check contracts/api_contract_manifest.json
```

Focused CI-equivalent tests:

```powershell
pytest -q tests/test_api_contract_manifest.py tests/test_model_snapshot_service.py tests/test_automation_config_store.py tests/test_backtest_orchestration_service.py tests/test_startup_import_health.py tests/test_dl_decomposition.py
```

## Public Artifact Policy

The following are excluded from Git by design:
- trained model weights and checkpoints (`*.pth`, `*.pt`, `*.pkl`, etc.)
- large runtime outputs (`models*`, `MasterDl`, `SequentialDLModels`, etc.)
- full historical datasets and generated logs

Use runtime generation flows or external artifact storage for those assets.

## Creating A Publishable Snapshot

A whitelist-based exporter is provided to prepare a clean public tree:

```powershell
python scripts/create_public_snapshot.py --output public_release
```

The output folder contains only publishable source assets and performs basic
size/secret checks.

## License

MIT (see [LICENSE](LICENSE)).
