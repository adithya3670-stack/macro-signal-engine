from backend.infrastructure.portfolio_profile_store import PortfolioProfileStore
from backend.services.portfolio_profile_service import PortfolioProfileService


def test_profile_service_crud_roundtrip(tmp_path):
    store = PortfolioProfileStore(data_dir=str(tmp_path))
    service = PortfolioProfileService(store=store)

    assert "Default Strategy" in service.get_all_profiles()

    input_config = {"universe": ["SP500"], "strategy_config": {"top_n": 2}}
    assert service.save_profile("My Profile", input_config) is True
    assert "My Profile" in service.get_all_profiles()
    assert "last_modified" not in input_config

    profile = service.get_profile("My Profile")
    assert profile is not None
    assert profile["universe"] == ["SP500"]
    assert "last_modified" in profile

    assert service.delete_profile("My Profile") is True
    assert service.get_profile("My Profile") is None
