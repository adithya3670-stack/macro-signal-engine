import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_api_contract_manifest import build_manifest, normalize_routes  # noqa: E402


def test_contract_manifest_file_exists():
    manifest_path = ROOT / "contracts" / "api_contract_manifest.json"
    assert manifest_path.exists(), "contracts/api_contract_manifest.json is missing"


def test_required_aliases_present_in_runtime_snapshot():
    manifest = build_manifest()
    current_paths = {entry["path"] for entry in manifest["routes"]}
    for alias in manifest["required_aliases"]:
        assert alias in current_paths, f"Missing required compatibility alias route: {alias}"


def test_manifest_routes_are_still_supported():
    manifest_path = ROOT / "contracts" / "api_contract_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)

    runtime = build_manifest()
    expected_routes = normalize_routes(expected.get("routes", []))
    runtime_routes = normalize_routes(runtime["routes"])

    missing = [route for route in expected_routes.keys() if route not in runtime_routes]
    assert not missing, f"Missing API routes from manifest: {missing}"

    method_violations = []
    for route, methods in expected_routes.items():
        runtime_methods = set(runtime_routes.get(route, ()))
        if not set(methods).issubset(runtime_methods):
            method_violations.append(
                {"path": route, "expected": list(methods), "runtime": sorted(runtime_methods)}
            )
    assert not method_violations, f"HTTP method contract mismatch: {method_violations}"
