from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from backend.shared.normalization import parse_bool


DEFAULT_SECRET_ENV = "MACRO_AUTO_EMAIL_PASSWORD"


@dataclass
class AutomationConfigModel:
    enabled: bool = False
    time: str = "20:30"
    profile_name: str = ""
    last_success: Optional[str] = None
    lock_enabled: bool = False
    lock_profile: str = ""
    lock_last_update: Optional[str] = None
    email_enabled: bool = False
    email_recipient: str = ""
    email_sender: str = ""
    email_password_env: str = DEFAULT_SECRET_ENV
    resolved_email_password: str = ""

    @classmethod
    def from_dict(
        cls,
        payload: Optional[Dict[str, Any]],
        env: Mapping[str, str],
    ) -> "AutomationConfigModel":
        raw = dict(payload or {})
        env_key = str(raw.get("email_password_env") or DEFAULT_SECRET_ENV)
        env_secret = env.get(env_key, "")
        legacy_secret = str(raw.get("email_password", "") or "")
        return cls(
            enabled=parse_bool(raw.get("enabled"), default=False),
            time=str(raw.get("time", "20:30")),
            profile_name=str(raw.get("profile_name", "")),
            last_success=raw.get("last_success"),
            lock_enabled=parse_bool(raw.get("lock_enabled"), default=False),
            lock_profile=str(raw.get("lock_profile", "")),
            lock_last_update=raw.get("lock_last_update"),
            email_enabled=parse_bool(raw.get("email_enabled"), default=False),
            email_recipient=str(raw.get("email_recipient", "")),
            email_sender=str(raw.get("email_sender", "")),
            email_password_env=env_key,
            resolved_email_password=env_secret or legacy_secret,
        )

    def merge_update(self, update: Optional[Dict[str, Any]], env: Mapping[str, str]) -> "AutomationConfigModel":
        patch = dict(update or {})
        next_cfg = AutomationConfigModel(**self.__dict__)

        if "enabled" in patch:
            next_cfg.enabled = parse_bool(patch.get("enabled"), default=next_cfg.enabled)
        if "time" in patch and patch.get("time") is not None:
            next_cfg.time = str(patch.get("time"))
        if "profile_name" in patch and patch.get("profile_name") is not None:
            next_cfg.profile_name = str(patch.get("profile_name"))
        if "last_success" in patch:
            next_cfg.last_success = patch.get("last_success")
        if "lock_enabled" in patch:
            next_cfg.lock_enabled = parse_bool(patch.get("lock_enabled"), default=next_cfg.lock_enabled)
        if "lock_profile" in patch and patch.get("lock_profile") is not None:
            next_cfg.lock_profile = str(patch.get("lock_profile"))
        if "lock_last_update" in patch:
            next_cfg.lock_last_update = patch.get("lock_last_update")
        if "email_enabled" in patch:
            next_cfg.email_enabled = parse_bool(patch.get("email_enabled"), default=next_cfg.email_enabled)
        if "email_recipient" in patch and patch.get("email_recipient") is not None:
            next_cfg.email_recipient = str(patch.get("email_recipient"))
        if "email_sender" in patch and patch.get("email_sender") is not None:
            next_cfg.email_sender = str(patch.get("email_sender"))
        if "email_password_env" in patch and patch.get("email_password_env"):
            next_cfg.email_password_env = str(patch.get("email_password_env"))

        incoming_password = str(patch.get("email_password", "") or "")
        if incoming_password and incoming_password != "*****":
            next_cfg.resolved_email_password = incoming_password
        else:
            env_secret = env.get(next_cfg.email_password_env, "")
            if env_secret:
                next_cfg.resolved_email_password = env_secret

        return next_cfg

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "time": self.time,
            "profile_name": self.profile_name,
            "last_success": self.last_success,
            "lock_enabled": self.lock_enabled,
            "lock_profile": self.lock_profile,
            "lock_last_update": self.lock_last_update,
            "email_enabled": self.email_enabled,
            "email_recipient": self.email_recipient,
            "email_sender": self.email_sender,
            "email_password_env": self.email_password_env,
        }

    def to_public_dict(self) -> Dict[str, Any]:
        payload = self.to_storage_dict()
        payload["email_password"] = "*****" if self.resolved_email_password else ""
        return payload


class AutomationConfigStore:
    def __init__(self, config_file: str, env: Optional[Mapping[str, str]] = None) -> None:
        self.config_file = Path(config_file)
        self.env = env or os.environ

    def load(self) -> AutomationConfigModel:
        payload: Dict[str, Any] = {}
        if self.config_file.exists():
            try:
                with self.config_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                payload = {}

        model = AutomationConfigModel.from_dict(payload, self.env)

        # One-way migration: strip persisted plaintext password if it existed.
        if "email_password" in payload:
            self.persist(model)

        return model

    def persist(self, model: AutomationConfigModel) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with self.config_file.open("w", encoding="utf-8") as handle:
            json.dump(model.to_storage_dict(), handle, indent=4)

    def apply_update(
        self,
        current: AutomationConfigModel,
        update: Optional[Dict[str, Any]],
    ) -> AutomationConfigModel:
        next_cfg = current.merge_update(update, self.env)
        self.persist(next_cfg)
        return next_cfg
