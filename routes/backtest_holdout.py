from __future__ import annotations

from flask import Blueprint, Response, jsonify, request, stream_with_context

# Legacy test-compat imports kept intentionally at module scope.
# Some older tests patch these symbols from this route module directly.
from analysis.deep_learning_model import DLMacroModel  # noqa: F401
from backend.domain.dto import HoldoutRunRequest, HoldoutSplitRequest
from backend.services.holdout_backtest_service import HoldoutBacktestService
from backend.shared.http import ServiceError
from backend.shared.normalization import parse_bool, parse_int
from data.etl import load_and_merge_data  # noqa: F401

holdout_bp = Blueprint("holdout", __name__)
holdout_service = HoldoutBacktestService()


@holdout_bp.route("/api/backtest/prepare_split", methods=["POST"])
def api_prepare_split():
    """Prepares and locks the data split for hold-out validation."""
    try:
        data = request.get_json(silent=True) or {}
        split_request = HoldoutSplitRequest(
            cutoff_year=parse_int(data.get("cutoff_year"), 2023) or 2023,
        )
        return jsonify(holdout_service.prepare_split(split_request))
    except ServiceError as exc:
        return jsonify({"error": exc.message}), exc.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@holdout_bp.route("/api/backtest/run_holdout", methods=["GET"])
def api_run_holdout():
    """Runs hold-out validation with streaming SSE progress."""
    run_request = HoldoutRunRequest(
        cutoff_year=parse_int(request.args.get("cutoff_year"), 2023) or 2023,
        model_type=str(request.args.get("model_type", "ml")),
        dl_mode=str(request.args.get("dl_mode", "balanced")),
        use_existing=parse_bool(request.args.get("use_existing"), False),
    )
    stream = holdout_service.stream_holdout(run_request)
    return Response(stream_with_context(stream), mimetype="text/event-stream")


@holdout_bp.route("/api/backtest/models", methods=["GET"])
def api_list_models():
    """Lists all saved DL holdout models found in the filesystem."""
    try:
        return jsonify(holdout_service.list_models())
    except Exception:
        return jsonify([])
