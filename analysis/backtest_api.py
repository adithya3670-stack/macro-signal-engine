from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.domain.dto import BacktestModelDeleteRequest, PortfolioProfileSaveRequest
from backend.services.backtest_model_admin_service import BacktestModelAdminService
from backend.services.portfolio_profile_service import PortfolioProfileService
from backend.services.portfolio_simulation_service import PortfolioSimulationService
from backend.shared.http import error_payload, error_status

backtest_bp = Blueprint("backtest_v2", __name__)

simulation_service = PortfolioSimulationService()
profile_service = PortfolioProfileService()
model_admin_service = BacktestModelAdminService()


@backtest_bp.route("/api/holdout_folders")
def list_holdout_folders():
    """Returns available holdout_dl year folders for model selection."""
    return jsonify(simulation_service.list_holdout_folders())


@backtest_bp.route("/api/profiles", methods=["GET"])
def get_profiles():
    return jsonify(profile_service.get_all_profiles())


@backtest_bp.route("/api/profiles/<name>", methods=["GET"])
def get_profile_detail(name):
    profile = profile_service.get_profile(name)
    if profile:
        return jsonify(profile)
    return jsonify({"error": "Not found"}), 404


@backtest_bp.route("/api/profiles", methods=["POST"])
def save_profile():
    data = request.get_json(silent=True) or {}
    profile_name = data.get("name")
    profile_config = data.get("config")
    if not profile_name or not profile_config:
        return jsonify({"error": "Name and config required"}), 400

    save_request = PortfolioProfileSaveRequest(
        name=str(profile_name),
        config=profile_config,
    )
    profile_service.save_profile(save_request.name, save_request.config)
    return jsonify({"success": True})


@backtest_bp.route("/api/profiles/<name>", methods=["DELETE"])
def delete_profile(name):
    if profile_service.delete_profile(name):
        return jsonify({"success": True})
    return jsonify({"error": "Not found"}), 404


@backtest_bp.route("/api/backtest/models/<category>/<year>", methods=["DELETE"])
def delete_model(category, year):
    try:
        delete_request = BacktestModelDeleteRequest(
            category=str(category),
            year=str(year),
        )
        if model_admin_service.delete_model(delete_request.category, delete_request.year):
            return jsonify({"success": True})
        return jsonify({"error": "Model not found"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@backtest_bp.route("/api/backtest/v2/run", methods=["POST"])
def run_backtest_v2():
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(simulation_service.run_backtest_v2(data))
    except Exception as exc:
        return jsonify(error_payload(exc)), error_status(exc)


@backtest_bp.route("/api/portfolio/run", methods=["POST"])
def run_portfolio():
    """Runs the portfolio simulation through the domain service layer."""
    try:
        data = request.get_json(silent=True) or {}
        return jsonify(simulation_service.run_portfolio(data))
    except Exception as exc:
        return jsonify(error_payload(exc)), error_status(exc)
