from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

DIRECTORY_ALLOWLIST = (
    ".github",
    "analysis",
    "backend",
    "backtesting",
    "config",
    "contracts",
    "requirements",
    "routes",
    "scripts",
    "services",
    "static",
    "templates",
    "tests",
    "data",
)

ROOT_FILE_ALLOWLIST = (
    ".gitignore",
    "app.py",
    "CONTRIBUTING.md",
    "LICENSE",
    "pyproject.toml",
    "README.md",
    "requirements.txt",
)

DATA_FILE_ALLOWLIST = {
    "README.md",
    "automation_config.example.json",
}
DATA_FILE_SUFFIX_ALLOWLIST = {".py"}

IGNORED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "public_release",
}

IGNORED_EXTENSIONS = {
    ".pth",
    ".pt",
    ".pkl",
    ".joblib",
    ".parquet",
    ".onnx",
    ".h5",
    ".ckpt",
    ".npy",
    ".npz",
    ".log",
}

MAX_FILE_SIZE = 100 * 1024 * 1024

SUSPECT_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)

EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
SAFE_EMAIL_SUFFIXES = ("@example.com", "@example.org", "@example.net", "@test.local", "@localhost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a source-only public snapshot.")
    parser.add_argument(
        "--output",
        default="public_release",
        help="Output directory relative to repository root (default: public_release)",
    )
    return parser.parse_args()


def is_data_file_allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.parts[0] != "data":
        return True
    if rel.name in DATA_FILE_ALLOWLIST:
        return True
    return rel.suffix.lower() in DATA_FILE_SUFFIX_ALLOWLIST


def should_copy(path: Path) -> bool:
    if any(part in IGNORED_DIR_NAMES for part in path.parts):
        return False
    if path.suffix.lower() in IGNORED_EXTENSIONS:
        return False
    if not is_data_file_allowed(path):
        return False
    return True


def copy_tree(src_dir: Path, dest_dir: Path) -> None:
    for item in src_dir.rglob("*"):
        if item.is_dir():
            continue
        if not should_copy(item):
            continue
        relative = item.relative_to(ROOT)
        target = dest_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def copy_root_files(dest_dir: Path, filenames: Iterable[str]) -> None:
    for name in filenames:
        src = ROOT / name
        if not src.exists() or not src.is_file():
            continue
        target = dest_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


def audit_snapshot(dest_dir: Path) -> None:
    too_large = []
    suspect_hits = []
    all_files = [p for p in dest_dir.rglob("*") if p.is_file()]

    for file_path in all_files:
        size = file_path.stat().st_size
        if size > MAX_FILE_SIZE:
            too_large.append((file_path, size))

        if file_path.suffix.lower() not in {".py", ".json", ".toml", ".md", ".txt", ".yml", ".yaml", ".csv"}:
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for pattern in SUSPECT_PATTERNS:
            if pattern.search(text):
                suspect_hits.append((file_path, pattern.pattern))

        for email in EMAIL_PATTERN.findall(text):
            lowered = email.lower()
            if any(lowered.endswith(suffix) for suffix in SAFE_EMAIL_SUFFIXES):
                continue
            suspect_hits.append((file_path, f"email:{email}"))

    total_size = sum(path.stat().st_size for path in all_files)
    print(f"Snapshot files: {len(all_files)}")
    print(f"Snapshot size: {total_size / (1024 * 1024):.2f} MB")

    if too_large:
        print("ERROR: Found files larger than 100MB:")
        for file_path, size in too_large:
            print(f"  - {file_path} ({size / (1024 * 1024):.2f} MB)")
        raise SystemExit(2)

    if suspect_hits:
        print("WARNING: Potential sensitive literals found:")
        for file_path, marker in suspect_hits:
            print(f"  - {file_path}: {marker}")
        raise SystemExit(3)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / args.output

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_root_files(output_dir, ROOT_FILE_ALLOWLIST)
    for directory in DIRECTORY_ALLOWLIST:
        src = ROOT / directory
        if src.exists() and src.is_dir():
            copy_tree(src, output_dir)

    audit_snapshot(output_dir)
    print(f"Public snapshot created at: {output_dir}")


if __name__ == "__main__":
    main()
