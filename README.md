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

## End-To-End Operator Guide (Start To Finish)

This section is a practical runbook using the three UI screens you shared:
- Screen A: Data Management + Historical Trends
- Screen B: Macro mini-panels (Fear/Greed, Rates, Inflation, Labor)
- Screen C: Feature Engineering + Signal Factory

### Stage 0. Local setup (one time)

1. Clone and enter the repo.
2. Create and activate a virtual environment.
3. Install runtime + test dependencies.
4. Set optional device control with `MACRO_TORCH_DEVICE` (`auto`, `cpu`, `cuda`, `cuda:0`, `mps`).

Use the exact commands in the Quickstart section below.

### Stage 1. Prepare data contracts (required)

1. Build your own ingestion connector from a provider you are allowed to use
   (for example Yahoo Finance, FRED, paid APIs, internal DBs, or exported files).
2. Normalize data to the required CSV schema under `data/`.
3. Confirm daily date coverage and no future-dated rows.
4. Keep all provider credentials/API tokens in environment variables only.

If your column names differ, map them to this project contract before running the app.

### Stage 2. Launch the app

1. Start Flask with `python app.py`.
2. Open the UI and begin in `Data Collection`.
3. Confirm existing files are detected before triggering refresh operations.

### Stage 3. Use Screen A (Data Collection)

1. Click `Update to Latest (Smart)` for normal daily use.
2. Use `Reset / Full Download (2008-Present)` only for full historical rebuilds.
3. For audits, set a `From` and `To` date and click `Extract Custom Range`.
4. Validate the lower `Historical Trends` chart:

- `SP500` should not be flat-lined or full of gaps.
- `VIX` should show event spikes.
- policy-rate series should move stepwise over regimes.

If one series is stale or shifted, fix your upstream ingestion mapping before feature generation.

### Stage 4. Use Screen B (Macro Diagnostics Panels)

Treat this as your data quality checkpoint before modeling.

1. Fear/Greed proxies: confirm momentum/strength/breadth/options/volatility/safe-haven panels are updating.
2. Rates block: confirm Fed funds, 10Y yield, and 10Y-3M curve are coherent.
3. Inflation block: verify CPI/PPI continuity and update cadence.
4. Labor/Growth block: verify GDP, unemployment, payroll series are populated and aligned.

Pass criteria:

- no all-zero panels
- no obviously shifted date windows
- no abrupt truncation in a subset of panels

### Stage 5. Use Screen C (Feature Engineering)

1. Open `Feature Engineering`.
2. Click `Run Feature Engine` in the `Signal Factory` card.
3. In controls, choose benchmark asset (for example `SP500`).
4. Choose engineered feature (for example `Liquidity_Impulse`).
5. Inspect the overlay chart for relationship quality:

- feature should react around known stress/liquidity regimes
- long flat segments usually mean stale or failed upstream inputs

Only proceed to training once feature overlays look credible.

### Stage 6. Deep Learning Studio

1. Move to `DEEP LEARNING STUDIO`.
2. Select target horizon/model setup used by your workflow.
3. Start training or inference run.
4. For GPU users, set `MACRO_TORCH_DEVICE` before launch:

- shared default: `auto`
- specific GPU: `cuda:0` (or another valid index)
- fallback-safe: unavailable CUDA automatically drops to CPU

This keeps training portable across different user hardware without code changes.

### Stage 7. Backtest Lab (Rotation)

1. Open `BACKTEST LAB (ROTATION)`.
2. Run backtests with your chosen signal/model source.
3. Compare metrics against prior trusted baseline windows.
4. Investigate changes that exceed your tolerance thresholds before promoting.

### Stage 8. Portfolio Management + Trailing Live Portfolio

1. In `PORTFOLIO MGMT`, review allocations and risk posture from latest signals.
2. In `TRAILING LIVE PORTFOLIO`, monitor rolling behavior and drift.
3. Re-run Stage 3-5 checks when live behavior diverges from expected regimes.

### Stage 9. Model Ensembling

1. Open `MODEL ENSEMBLING`.
2. Combine eligible model outputs under your policy rules.
3. Re-validate combined outputs in backtest before operational use.

### Stage 10. Public-source readiness checks (before sharing)

1. Compile/import health:
   `python -m compileall -q analysis backtesting routes services data config backend app.py`
2. Contract snapshot check:
   `python scripts/generate_api_contract_manifest.py --check contracts/api_contract_manifest.json`
3. Test gate:
   `pytest -q`
4. Secret/PII sanity check:
   ensure no passwords/tokens/personal runtime values are tracked.
5. Artifact policy check:
   confirm no large model/data/log artifacts are committed.

If all stages pass, your repo state is ready for a public source-only release flow.

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

### 4. Configure training device (optional)

Device selection is portable across users via `MACRO_TORCH_DEVICE`:
- `auto` (default): use CUDA if available, otherwise fall back to CPU
- `cpu`: force CPU mode
- `cuda`: use default CUDA device
- `cuda:0`, `cuda:1`, ...: choose a specific GPU index
- `mps`: Apple Silicon Metal backend (if available)

Examples:

```powershell
$env:MACRO_TORCH_DEVICE="auto"
# or
$env:MACRO_TORCH_DEVICE="cuda:0"
```

If a requested device is unavailable, the runtime safely falls back to CPU.

### 5. Provide local datasets (required)

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

### 6. Configure automation secrets (optional)

Copy the template and set environment variables:

```powershell
Copy-Item data/automation_config.example.json data/automation_config.json
$env:MACRO_AUTO_EMAIL_PASSWORD="your-password"
```

### 7. Run app

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
