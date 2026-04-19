from flask import Flask

import routes.backtest_holdout as holdout_routes
from routes.backtest_holdout import holdout_bp


def _client():
    app = Flask(__name__)
    app.register_blueprint(holdout_bp)
    return app.test_client()


def test_prepare_split_route_contract(monkeypatch):
    expected = {
        "train_rows": 10,
        "train_start": "2019-01-01",
        "train_end": "2023-12-31",
        "test_rows": 3,
        "test_start": "2024-01-01",
        "test_end": "2024-01-31",
        "cutoff_year": 2023,
    }
    monkeypatch.setattr(holdout_routes.holdout_service, "prepare_split", lambda _req: expected)

    response = _client().post("/api/backtest/prepare_split", json={"cutoff_year": 2023})
    data = response.get_json()

    assert response.status_code == 200
    assert data["cutoff_year"] == 2023
    assert {"train_rows", "test_rows", "train_start", "train_end", "test_start", "test_end"} <= set(data.keys())


def test_run_holdout_route_contract(monkeypatch):
    fake_stream = iter(
        [
            'data: {"progress": 100, "message": "Complete!"}\n\n',
            "data: DONE\n\n",
        ]
    )
    monkeypatch.setattr(holdout_routes.holdout_service, "stream_holdout", lambda _req: fake_stream)

    response = _client().get("/api/backtest/run_holdout?cutoff_year=2024&model_type=dl&use_existing=true")

    assert response.status_code == 200
    assert response.mimetype == "text/event-stream"
    body = response.get_data(as_text=True)
    assert "data: DONE" in body


def test_list_models_route_contract(monkeypatch):
    expected = [
        {
            "id": "dl_2024",
            "cutoff_year": "2024",
            "category": "dl",
            "model_type": "dl_auto",
            "dl_mode": "balanced",
            "created_at": "2026-01-01T00:00:00",
            "cutoff_date": "2024-12-31",
        }
    ]
    monkeypatch.setattr(holdout_routes.holdout_service, "list_models", lambda: expected)

    response = _client().get("/api/backtest/models")

    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert data[0]["id"] == "dl_2024"


def test_list_models_route_returns_empty_list_on_failure(monkeypatch):
    def _raise():
        raise RuntimeError("boom")

    monkeypatch.setattr(holdout_routes.holdout_service, "list_models", _raise)
    response = _client().get("/api/backtest/models")

    assert response.status_code == 200
    assert response.get_json() == []
