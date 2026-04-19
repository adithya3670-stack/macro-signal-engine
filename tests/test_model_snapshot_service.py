import json
from pathlib import Path

from backend.domain.dto import SnapshotCreateRequest
from backend.infrastructure.model_artifacts import ModelArtifactResolver
from backend.services.model_snapshot_service import ModelSnapshotService


def test_snapshot_create_list_restore(tmp_path):
    base = Path(tmp_path)
    (base / "models_dl").mkdir(parents=True, exist_ok=True)
    (base / "models_dl" / "sample_model.bin").write_text("x", encoding="utf-8")
    (base / "models_price_3d").mkdir(parents=True, exist_ok=True)
    (base / "models_price_3d" / "metrics.json").write_text("{}", encoding="utf-8")

    cache = {"drivers": {"Short Term (30d)": {"driver": "SP500", "correlation": 0.5}}}
    resolver = ModelArtifactResolver(base_dir=str(base))
    service = ModelSnapshotService(artifact_resolver=resolver, cache_ref=cache)

    created = service.create_snapshot(SnapshotCreateRequest(tag="baseline", description="unit test"))
    assert created["id"]
    assert created["tag"] == "baseline"
    assert created["files_count"] >= 2

    snapshots = service.list_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["id"] == created["id"]
    assert snapshots[0]["display_time"]

    cache["drivers"] = {"Short Term (30d)": {"driver": "No Data", "correlation": 0}}
    restored = service.restore_snapshot(created["id"])
    assert restored["id"] == created["id"]
    assert cache["drivers"]["Short Term (30d)"]["driver"] == "SP500"

    manifest_path = base / "snapshots" / f"{created['id']}.json"
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "model_roots" in persisted
    assert "files_count" in persisted
