from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def collect_routes() -> List[Dict[str, object]]:
    try:
        from app import app

        routes = []
        for rule in app.url_map.iter_rules():
            path = str(rule.rule)
            if not path.startswith("/api/"):
                continue
            methods = sorted([m for m in rule.methods if m not in {"HEAD", "OPTIONS"}])
            routes.append({"path": path, "methods": methods})

        routes.sort(key=lambda item: (item["path"], ",".join(item["methods"])))
        return routes
    except Exception:
        return collect_routes_fallback()


def _parse_blueprint_prefixes() -> Dict[str, str]:
    app_file = ROOT / "app.py"
    if not app_file.exists():
        return {}
    content = app_file.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"app\.register_blueprint\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*url_prefix\s*=\s*['\"]([^'\"]+)['\"]\s*\)"
    )
    return {name: prefix for name, prefix in pattern.findall(content)}


def _parse_methods(raw: str | None) -> List[str]:
    if not raw:
        return ["GET"]
    found = re.findall(r"['\"]([A-Za-z]+)['\"]", raw)
    methods = [method.upper() for method in found] if found else ["GET"]
    return sorted(set(methods))


def collect_routes_fallback() -> List[Dict[str, object]]:
    decorator_pattern = re.compile(
        r"@([A-Za-z_][A-Za-z0-9_]*)\.route\(\s*['\"]([^'\"]+)['\"](?:\s*,\s*methods\s*=\s*\[([^\]]*)\])?"
    )
    prefixes = _parse_blueprint_prefixes()
    routes: Dict[str, List[str]] = {}

    for file_path in list((ROOT / "routes").glob("*.py")) + list((ROOT / "analysis").glob("*.py")):
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        for blueprint_name, route_path, methods_raw in decorator_pattern.findall(content):
            full_path = route_path
            if not full_path.startswith("/"):
                full_path = f"/{full_path}"

            prefix = prefixes.get(blueprint_name)
            if prefix and not full_path.startswith("/api/"):
                full_path = f"{prefix.rstrip('/')}{full_path}"

            if not full_path.startswith("/api/"):
                continue

            methods = _parse_methods(methods_raw)
            routes[full_path] = sorted(set((routes.get(full_path) or []) + methods))

    output = [{"path": path, "methods": methods} for path, methods in sorted(routes.items())]
    return output


def normalize_routes(routes: List[Dict[str, object]]) -> Dict[str, Tuple[str, ...]]:
    merged: Dict[str, set[str]] = {}
    for item in routes:
        path = str(item["path"])
        methods = {str(method) for method in item.get("methods", [])}
        merged.setdefault(path, set()).update(methods)
    return {path: tuple(sorted(methods)) for path, methods in merged.items()}


def build_manifest() -> Dict[str, object]:
    return {
        "generated_at": datetime.now().isoformat(),
        "required_aliases": [
            "/api/models/snapshots",
            "/api/models/snapshot",
            "/api/models/restore",
            "/api/train/forecast_stream",
        ],
        "routes": collect_routes(),
    }


def check_manifest(path: Path) -> int:
    if not path.exists():
        print(f"contract-missing: {path}")
        return 1

    with path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)

    current = build_manifest()
    expected_routes = normalize_routes(expected.get("routes", []))
    current_routes = normalize_routes(current["routes"])

    missing_routes = []
    missing_methods = []
    for route, methods in expected_routes.items():
        if route not in current_routes:
            missing_routes.append(route)
            continue
        current_methods = set(current_routes[route])
        expected_methods = set(methods)
        if not expected_methods.issubset(current_methods):
            missing_methods.append(
                {
                    "path": route,
                    "expected": sorted(expected_methods),
                    "current": sorted(current_methods),
                }
            )

    alias_errors = []
    current_paths = set(current_routes.keys())
    for alias in expected.get("required_aliases", []):
        if alias not in current_paths:
            alias_errors.append(alias)

    if missing_routes or missing_methods or alias_errors:
        print("contract-check-failed")
        if missing_routes:
            print("missing-routes:", json.dumps(missing_routes, indent=2))
        if missing_methods:
            print("method-mismatches:", json.dumps(missing_methods, indent=2))
        if alias_errors:
            print("missing-aliases:", json.dumps(alias_errors, indent=2))
        return 1

    print("contract-check-ok")
    return 0


def write_manifest(path: Path) -> int:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"contract-written: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate/check API contract manifest.")
    parser.add_argument(
        "--output",
        default="contracts/api_contract_manifest.json",
        help="Path to write manifest (default: contracts/api_contract_manifest.json)",
    )
    parser.add_argument(
        "--check",
        default=None,
        help="Path to expected manifest to validate against.",
    )
    args = parser.parse_args()

    if args.check:
        return check_manifest(Path(args.check))
    return write_manifest(Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
