from backend.services.backtest_model_admin_service import BacktestModelAdminService


def test_delete_model_resolves_dl_and_ml_paths(tmp_path):
    service = BacktestModelAdminService(base_dir=str(tmp_path))

    dl_dir = tmp_path / "models" / "holdout_dl" / "2024"
    dl_dir.mkdir(parents=True)
    (dl_dir / "marker.txt").write_text("ok", encoding="utf-8")
    assert service.delete_model("dl", "2024") is True
    assert not dl_dir.exists()

    ml_dir = tmp_path / "models" / "holdout" / "2022"
    ml_dir.mkdir(parents=True)
    assert service.delete_model("ml", "2022") is True
    assert not ml_dir.exists()


def test_delete_model_returns_false_when_missing(tmp_path):
    service = BacktestModelAdminService(base_dir=str(tmp_path))
    assert service.delete_model("dl", "2099") is False
