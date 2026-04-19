# Dependency Sets

Pinned dependency sets are split by concern:

- `runtime.lock.txt`: production/runtime dependencies
- `train.lock.txt`: runtime + model training stack
- `research.lock.txt`: runtime + research/plotting helpers
- `test.lock.txt`: runtime + test tooling

The existing root `requirements.txt` is intentionally retained for backward compatibility.
