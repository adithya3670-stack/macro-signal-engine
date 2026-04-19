from flask import Flask

import analysis.backtest_api as backtest_routes
from analysis.backtest_api import backtest_bp


def _client():
    app = Flask(__name__)
    app.register_blueprint(backtest_bp)
    return app.test_client()


def test_holdout_folders_route_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.simulation_service, "list_holdout_folders", lambda: {"folders": ["default"]})
    response = _client().get("/api/holdout_folders")
    assert response.status_code == 200
    assert response.get_json() == {"folders": ["default"]}


def test_profiles_get_route_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.profile_service, "get_all_profiles", lambda: ["A", "B"])
    response = _client().get("/api/profiles")
    assert response.status_code == 200
    assert response.get_json() == ["A", "B"]


def test_profile_detail_not_found_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.profile_service, "get_profile", lambda _name: None)
    response = _client().get("/api/profiles/missing")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Not found"}


def test_profile_save_validation_contract():
    response = _client().post("/api/profiles", json={"name": "demo"})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Name and config required"}


def test_profile_save_success_contract(monkeypatch):
    called = {}

    def fake_save(name, config):
        called["name"] = name
        called["config"] = config
        return True

    monkeypatch.setattr(backtest_routes.profile_service, "save_profile", fake_save)
    response = _client().post("/api/profiles", json={"name": "demo", "config": {"top_n": 2}})

    assert response.status_code == 200
    assert response.get_json() == {"success": True}
    assert called["name"] == "demo"
    assert called["config"] == {"top_n": 2}


def test_profile_delete_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.profile_service, "delete_profile", lambda _name: True)
    response = _client().delete("/api/profiles/demo")
    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_profile_delete_not_found_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.profile_service, "delete_profile", lambda _name: False)
    response = _client().delete("/api/profiles/missing")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Not found"}


def test_delete_model_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.model_admin_service, "delete_model", lambda _c, _y: True)
    response = _client().delete("/api/backtest/models/dl/2024")
    assert response.status_code == 200
    assert response.get_json() == {"success": True}


def test_delete_model_not_found_contract(monkeypatch):
    monkeypatch.setattr(backtest_routes.model_admin_service, "delete_model", lambda _c, _y: False)
    response = _client().delete("/api/backtest/models/dl/4040")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Model not found"}


def test_delete_model_exception_contract(monkeypatch):
    def _raise(_c, _y):
        raise RuntimeError("boom")

    monkeypatch.setattr(backtest_routes.model_admin_service, "delete_model", _raise)
    response = _client().delete("/api/backtest/models/dl/2024")
    assert response.status_code == 500
    assert response.get_json() == {"error": "boom"}


def test_run_backtest_v2_delegates(monkeypatch):
    monkeypatch.setattr(backtest_routes.simulation_service, "run_backtest_v2", lambda payload: {"ok": payload.get("x")})
    response = _client().post("/api/backtest/v2/run", json={"x": 1})
    assert response.status_code == 200
    assert response.get_json() == {"ok": 1}


def test_run_portfolio_delegates(monkeypatch):
    monkeypatch.setattr(backtest_routes.simulation_service, "run_portfolio", lambda payload: {"ok": payload.get("x")})
    response = _client().post("/api/portfolio/run", json={"x": 2})
    assert response.status_code == 200
    assert response.get_json() == {"ok": 2}
