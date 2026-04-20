# Local Security Scan Reports

These reports are committed as temporary evidence while GitHub-hosted Actions
are blocked by an account billing lock.

## Included Files

- `gitleaks-report.json`: secret scan output from `gitleaks git .`
- `gitleaks-summary.txt`: quick status summary (`findings=0` for latest run)
- `pip-audit-report.json`: dependency vulnerability output
- `pip-audit-summary.txt`: quick status summary (`packages_with_vulns=4`, `total_vulns=8` for latest run)

## Re-run Commands

```powershell
gitleaks git . --report-format json --report-path reports/security/gitleaks-report.json --redact

python -m pip_audit --no-deps --disable-pip `
  -r requirements/runtime.lock.txt `
  -r requirements/train.lock.txt `
  -r requirements/research.lock.txt `
  -r requirements/test.lock.txt `
  --format json `
  --output reports/security/pip-audit-report.json
```

## Notes

- `pip-audit` exits nonzero when vulnerabilities are found. This is expected.
- Current `pip-audit` mode uses `--no-deps --disable-pip` for compatibility with
  this local environment; once GitHub Actions billing is restored, CI should be
  the source of truth for scheduled security visibility.
