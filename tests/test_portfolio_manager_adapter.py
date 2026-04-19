from analysis.portfolio_manager import PortfolioManager


def test_portfolio_manager_adapter_preserves_legacy_shape(tmp_path):
    manager = PortfolioManager(data_dir=str(tmp_path))

    assert manager.data_dir == str(tmp_path)
    assert manager.file_path.endswith("portfolio_profiles.json")
    assert "Default Strategy" in manager.get_all_profiles()

    config = {"universe": ["SP500"], "strategy_config": {"top_n": 3}}
    assert manager.save_profile("Adapter Profile", config) is True
    loaded = manager.get_profile("Adapter Profile")
    assert loaded is not None
    assert loaded["universe"] == ["SP500"]
    assert "last_modified" in loaded

    raw = manager._load_data()
    assert "Adapter Profile" in raw

    assert manager.delete_profile("Adapter Profile") is True
