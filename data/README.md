# Data Directory Policy

This public repository keeps this directory source-first by design.

- Tracked:
  - loader/source modules (`*.py`)
  - lightweight configuration template (`automation_config.example.json`)
  - this policy file
- Not tracked:
  - full runtime datasets (`*.csv`)
  - logs and generated research outputs

## Data Inputs Are User-Managed

Built-in Yahoo Finance and FRED fetch integrations are intentionally removed.
Users must gather their own data and place CSV files in this folder, including:

- `market_data.csv`
- `macro_data.csv`
- `sentiment_data.csv`
- `indicators_data.csv`
- `commodities_data.csv`

## Automation Config

Template file:
- `automation_config.example.json`

Local runtime file (ignored by Git):
- `automation_config.json`

Secrets must come from environment variables (for example
`MACRO_AUTO_EMAIL_PASSWORD`).
