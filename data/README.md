# Data Directory Policy

This public repository keeps this directory minimal by design.

- Tracked: lightweight configuration templates only
- Not tracked: full runtime datasets, logs, and generated research outputs

Use runtime endpoints/pipelines to generate local data when needed.

## Automation Config

Template file:
- `automation_config.example.json`

Local runtime file (ignored by Git):
- `automation_config.json`

Secrets must come from environment variables (for example
`MACRO_AUTO_EMAIL_PASSWORD`).
