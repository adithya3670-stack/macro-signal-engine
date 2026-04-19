from backend.infrastructure.portfolio_profile_store import PortfolioProfileStore


def test_store_creates_default_profile_file(tmp_path):
    store = PortfolioProfileStore(data_dir=str(tmp_path))
    payload = store.read_all()

    assert "Default Strategy" in payload
    assert store.file_path.endswith("portfolio_profiles.json")


def test_store_read_write_roundtrip(tmp_path):
    store = PortfolioProfileStore(data_dir=str(tmp_path))
    data = store.read_all()
    data["Custom"] = {"universe": ["SP500"], "strategy_config": {"top_n": 2}}
    store.write_all(data)

    loaded = store.read_all()
    assert "Custom" in loaded
    assert loaded["Custom"]["universe"] == ["SP500"]
