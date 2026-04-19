import pytest


def test_app_import_and_blueprint_registration_health():
    pytest.importorskip("pandas_datareader")

    from app import app

    assert app is not None
    rules = {rule.rule for rule in app.url_map.iter_rules()}
    assert "/api/dashboard" in rules
    assert "/api/train/forecast_stream" in rules
    assert "/api/models/snapshots" in rules
