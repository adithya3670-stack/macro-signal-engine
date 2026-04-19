import json
from pathlib import Path

from backend.services.automation_config_store import AutomationConfigStore


def test_legacy_plaintext_password_migrates_out_of_storage(tmp_path):
    config_file = Path(tmp_path) / "automation_config.json"
    config_file.write_text(
        json.dumps(
            {
                "enabled": True,
                "time": "12:30",
                "email_sender": "sender@example.com",
                "email_password": "legacy-secret",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    store = AutomationConfigStore(str(config_file), env={})
    model = store.load()
    assert model.resolved_email_password == "legacy-secret"
    assert model.to_public_dict()["email_password"] == "*****"

    persisted = json.loads(config_file.read_text(encoding="utf-8"))
    assert "email_password" not in persisted
    assert persisted["email_password_env"] == "MACRO_AUTO_EMAIL_PASSWORD"


def test_env_secret_has_precedence_and_updates_are_masked(tmp_path):
    config_file = Path(tmp_path) / "automation_config.json"
    store = AutomationConfigStore(
        str(config_file),
        env={"MY_AUTO_PASSWORD": "env-secret"},
    )

    model = store.load()
    model = store.apply_update(
        model,
        {
            "email_password_env": "MY_AUTO_PASSWORD",
            "email_sender": "sender@example.com",
            "email_password": "*****",
        },
    )
    assert model.resolved_email_password == "env-secret"
    assert model.to_public_dict()["email_password"] == "*****"

    model = store.apply_update(model, {"email_password": "runtime-secret"})
    assert model.resolved_email_password == "runtime-secret"

    persisted = json.loads(config_file.read_text(encoding="utf-8"))
    assert "email_password" not in persisted
