from backend.services.backtest_orchestration import BacktestOrchestrationService


def test_signal_universe_builder_single_and_hybrid():
    svc = BacktestOrchestrationService()

    single = svc.build_signal_universe(["SP500", "Gold"], horizon="1m", use_hybrid=False)
    assert single == ["SP500_1m", "Gold_1m"]

    hybrid = svc.build_signal_universe(["SP500"], horizon="1m", use_hybrid=True)
    assert hybrid == ["SP500_1w", "SP500_1m", "SP500_3m"]


def test_hybrid_weight_resolution():
    svc = BacktestOrchestrationService()
    default_weights = svc.get_hybrid_weights(None)
    assert round(default_weights["1w"], 4) == round(1 / 3, 4)

    profile = svc.get_hybrid_weights("long_max")
    assert profile == {"1w": 0.10, "1m": 0.30, "3m": 0.60}


def test_model_source_resolution_prefers_dl_folder_and_model_type():
    svc = BacktestOrchestrationService()
    selection = svc.resolve_portfolio_selection(
        model_category=None,
        model_year=None,
        strat_config={"dl_folder": "dl_2024", "model_type": "winner_ensemble_accuracy"},
    )
    assert selection.source == "holdout"
    assert selection.year == "2024"

    rolling = svc.resolve_portfolio_selection(
        model_category="dl",
        model_year="latest",
        strat_config={"dl_folder": "rolling_master", "model_type": "ensemble"},
    )
    assert rolling.source == "rolling_master"
