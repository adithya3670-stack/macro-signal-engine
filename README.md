# Macro Signal Engine

Macro Signal Engine is a Flask-based research and backtesting platform for macro-aware
portfolio simulation, holdout/model evaluation, and price-horizon forecasting.

Repository: [adithya3670-stack/macro-signal-engine](https://github.com/adithya3670-stack/macro-signal-engine)

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

## Product Walkthrough (From Current UI Screens)

### 1. Data Management + Historical Trends

The Data Collection screen is designed for two ingestion modes:
- `Update to Latest (Smart)`: append only missing dates to your local history
- `Reset / Full Download (2008–Present)`: full rebuild mode for clean backfills

Custom extraction supports explicit `From` and `To` dates, then writes normalized
outputs into the expected local CSV contracts under `data/`.

The lower trend panel overlays key context series (for example `SP500`, `VIX`,
and policy-rate proxies) so users can visually validate alignment before training.
This helps catch bad joins, date gaps, and regime-shift discontinuities early.

### 2. Fear & Greed, Rates, Inflation, and Labor Panels

The macro dashboard presents grouped mini-panels to quickly sanity-check data health:
- Fear & Greed proxies: momentum, strength, breadth, options, junk demand, volatility, safe-haven demand
- Interest rates and curve state: Fed funds, 10Y yield, 10Y-3M spread
- Inflation and prices: CPI and PPI trajectories
- Labor and growth context: GDP growth, unemployment, nonfarm payrolls

These grouped views are intended as pre-model diagnostics. If one group looks stale
or structurally inconsistent, users should refresh or repair their upstream data
connector before running feature generation or model training.

### 3. Feature Engineering Workspace

The Feature Engineering screen turns raw macro-market inputs into model-ready signals:
- `Signal Factory` starts the engineered-feature pipeline
- left-side controls let users select benchmark asset and feature overlays
- the main chart compares market behavior vs engineered signals (example:
  `SP500` vs `Liquidity_Impulse`) on synchronized timelines

The intended workflow is:
1. Ingest/refresh raw datasets.
2. Run feature generation.
3. Inspect signal-vs-price behavior visually.
4. Proceed to Deep Learning Studio / Backtest Lab only after signal quality checks pass.

## Quickstart (Local)

### 1. Clone repository

```powershell
git clone https://github.com/adithya3670-stack/macro-signal-engine.git
cd macro-signal-engine
```

### 2. Create environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install dependencies

Runtime + test stack:

```powershell
pip install -r requirements/runtime.lock.txt -r requirements/test.lock.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

Optional expanded stacks:
- `requirements/train.lock.txt`
- `requirements/research.lock.txt`

### 4. Provide local datasets (required)

This repository is **source-only** and does not ship built-in vendor connectors.
You can connect your own ingestion pipeline from any source you choose (for example
Yahoo Finance, FRED, paid market-data APIs, internal databases, or exported files),
then write normalized CSV inputs under `data/` before running refresh/training flows.

Required files:
- `data/market_data.csv` with `Date, SP500, Nasdaq, DJIA, Russell2000`
- `data/macro_data.csv` with `Date` plus `FEDFUNDS, CPIAUCSL, PPIACO, UNRATE, PAYEMS, M2SL, T10Y3M, UMCSENT, WALCL, DGS10, A191RL1Q225SBEA`
- `data/sentiment_data.csv` with `Date, VIX`
- `data/indicators_data.csv` with `Date, Momentum, Strength_RSI, Breadth_Vol, Options_VIX, Junk_Bond_Demand, Volatility_Spread, Safe_Haven_Demand`
- `data/commodities_data.csv` with `Date, Gold, Silver, Oil, Copper` (`Bitcoin` optional)

Notes:
- Date parsing expects a `Date` column (or first column) and daily timestamps.
- Older raw ticker names are auto-mapped when possible (for example `^GSPC -> SP500`, `^VIX -> VIX`).
- Keep credentials and API keys out of Git (use local env vars/secrets managers).
- If you use third-party providers, make sure your ingestion usage/storage follows their
  terms, licensing, and attribution requirements.

### 5. Configure automation secrets (optional)

Copy the template and set environment variables:

```powershell
Copy-Item data/automation_config.example.json data/automation_config.json
$env:MACRO_AUTO_EMAIL_PASSWORD="your-password"
```

### 6. Run app

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
