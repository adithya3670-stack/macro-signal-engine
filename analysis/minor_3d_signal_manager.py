import datetime as dt
import json
import os
import subprocess
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.settings import (
    BASE_DIR,
    MINOR_3D_RESEARCH_SCRIPT,
    PRICE_3D_FEATURES_FILE,
    RESEARCH_MINOR_EVENT_DIR,
    SP500_DRAWDOWN_EVENTS_FILE,
)


class Minor3DSignalManager:
    """Manager for precision-first 3D minor drawdown signals."""

    WEEKLY_REEVAL_DAYS = 7
    DEFAULT_PRECISION_FLOOR = 0.20
    DEFAULT_COOLDOWN = 15
    DEFAULT_SEED = 42
    DEFAULT_WARMUP = 120
    DEFAULT_RULE_SEARCH_SAMPLES = 110000

    POLICY_FILE = "minor_3d_policy.json"
    META_FILE = "minor_3d_policy_meta.json"
    LATEST_FILE = "minor_3d_latest_signals.json"
    COMBO_RESULTS_FILE = "minor_3d_rule_combo_results.csv"

    def __init__(
        self,
        research_dir: str = RESEARCH_MINOR_EVENT_DIR,
        features_file: str = PRICE_3D_FEATURES_FILE,
        events_file: str = SP500_DRAWDOWN_EVENTS_FILE,
        script_path: str = MINOR_3D_RESEARCH_SCRIPT,
    ):
        self.research_dir = research_dir
        self.features_file = features_file
        self.events_file = events_file
        self.script_path = script_path
        os.makedirs(self.research_dir, exist_ok=True)

    def rebuild(
        self,
        precision_floor: Optional[float] = None,
        cooldown: Optional[int] = None,
        seed: Optional[int] = None,
        warmup: Optional[int] = None,
        rule_search_samples: Optional[int] = None,
    ) -> Dict[str, object]:
        precision_floor = self._resolve_precision_floor(precision_floor)
        cooldown = int(cooldown if cooldown is not None else self.DEFAULT_COOLDOWN)
        seed = int(seed if seed is not None else self.DEFAULT_SEED)
        warmup = int(warmup if warmup is not None else self.DEFAULT_WARMUP)
        rule_search_samples = int(rule_search_samples if rule_search_samples is not None else self.DEFAULT_RULE_SEARCH_SAMPLES)

        run_result = self._run_research(
            cooldown=cooldown,
            seed=seed,
            warmup=warmup,
            rule_search_samples=rule_search_samples,
        )
        policy = self._compute_policy(precision_floor=precision_floor)
        now = self._now_utc()
        meta = {
            "last_eval_utc": self._to_iso(now),
            "next_eval_due_utc": self._to_iso(now + dt.timedelta(days=self.WEEKLY_REEVAL_DAYS)),
            "precision_floor": float(precision_floor),
            "cooldown": cooldown,
            "seed": seed,
            "warmup": warmup,
            "rule_search_samples": rule_search_samples,
            "script_path": self.script_path,
            "features_file": self.features_file,
            "events_file": self.events_file,
        }

        self._write_json(self._path(self.POLICY_FILE), policy)
        self._write_json(self._path(self.META_FILE), meta)

        return {
            "status": "success",
            "research_dir": self.research_dir,
            "policy": policy,
            "meta": meta,
            "run_stdout_tail": run_result["stdout_tail"],
            "run_stderr_tail": run_result["stderr_tail"],
        }

    def load_policy(self, auto_refresh: bool = True, precision_floor: Optional[float] = None) -> Dict[str, object]:
        refresh = self._maybe_auto_refresh(auto_refresh=auto_refresh, precision_floor=precision_floor)
        policy = self._load_or_compute_policy(precision_floor=precision_floor)
        meta = self._read_json(self._path(self.META_FILE), default={})
        return {
            "status": "success",
            "policy": policy,
            "meta": meta,
            "auto_refresh": refresh,
        }

    def load_latest_status(self, auto_refresh: bool = True, precision_floor: Optional[float] = None) -> Dict[str, object]:
        refresh = self._maybe_auto_refresh(auto_refresh=auto_refresh, precision_floor=precision_floor)
        policy = self._load_or_compute_policy(precision_floor=precision_floor)
        latest = self._read_json(self._path(self.LATEST_FILE), default={})
        meta = self._read_json(self._path(self.META_FILE), default={})

        signals = latest.get("signals", {}) if isinstance(latest, dict) else {}
        combo_latest = signals.get("combo_latest", {}) if isinstance(signals, dict) else {}
        actionable_combo = policy["actionable"]["combo"]
        watchlist_combo = policy["watchlist"]["combo"]

        actionable_raw = bool(combo_latest.get(actionable_combo, {}).get("latest_alert", False))
        watchlist_raw = bool(combo_latest.get(watchlist_combo, {}).get("latest_alert", False))
        actionable_enabled = bool(policy["actionable"]["enabled"])
        actionable = actionable_enabled and actionable_raw

        if actionable:
            state = "ACTIONABLE"
            reason_code = "ACTIONABLE_SIGNAL"
        elif watchlist_raw:
            state = "WATCHLIST"
            reason_code = "WATCHLIST_SIGNAL"
        else:
            state = "CLEAR"
            reason_code = "NO_ACTIVE_SIGNAL"

        return {
            "status": "success",
            "as_of_date": latest.get("as_of_date"),
            "state": state,
            "reason_code": reason_code,
            "actionable": {
                "combo": actionable_combo,
                "enabled": actionable_enabled,
                "raw_signal": actionable_raw,
                "effective_signal": actionable,
                "policy_reason_code": policy["actionable"]["reason_code"],
                "precision_floor": policy["config"]["precision_floor"],
                "combo_metrics": policy["combo_metrics"].get(actionable_combo, {}),
            },
            "watchlist": {
                "combo": watchlist_combo,
                "raw_signal": watchlist_raw,
                "combo_metrics": policy["combo_metrics"].get(watchlist_combo, {}),
            },
            "top_rules": signals.get("top_rules", []),
            "policy": policy,
            "meta": meta,
            "auto_refresh": refresh,
        }

    def _maybe_auto_refresh(self, auto_refresh: bool, precision_floor: Optional[float]) -> Dict[str, object]:
        if not auto_refresh:
            return {"attempted": False, "refreshed": False}

        meta = self._read_json(self._path(self.META_FILE), default={})
        now = self._now_utc()
        needs_refresh = self._is_refresh_due(meta=meta, now=now)

        if not needs_refresh:
            return {"attempted": True, "refreshed": False}

        floor = self._resolve_precision_floor(precision_floor, meta=meta)
        cooldown = int(meta.get("cooldown", self.DEFAULT_COOLDOWN))
        seed = int(meta.get("seed", self.DEFAULT_SEED))
        warmup = int(meta.get("warmup", self.DEFAULT_WARMUP))
        rule_search_samples = int(meta.get("rule_search_samples", self.DEFAULT_RULE_SEARCH_SAMPLES))
        rebuilt = self.rebuild(
            precision_floor=floor,
            cooldown=cooldown,
            seed=seed,
            warmup=warmup,
            rule_search_samples=rule_search_samples,
        )
        return {
            "attempted": True,
            "refreshed": True,
            "refresh_at_utc": self._to_iso(now),
            "rebuild_status": rebuilt.get("status", "unknown"),
        }

    def _is_refresh_due(self, meta: Dict[str, object], now: dt.datetime) -> bool:
        if not meta:
            return True

        next_due = self._parse_iso(meta.get("next_eval_due_utc"))
        if next_due is not None:
            return now >= next_due

        last_eval = self._parse_iso(meta.get("last_eval_utc"))
        if last_eval is None:
            return True
        return now >= (last_eval + dt.timedelta(days=self.WEEKLY_REEVAL_DAYS))

    def _load_or_compute_policy(self, precision_floor: Optional[float]) -> Dict[str, object]:
        precision_floor = self._resolve_precision_floor(precision_floor)
        policy_path = self._path(self.POLICY_FILE)
        policy = self._read_json(policy_path, default=None)
        if not isinstance(policy, dict):
            policy = self._compute_policy(precision_floor=precision_floor)
            self._write_json(policy_path, policy)
            return policy

        existing_floor = float(policy.get("config", {}).get("precision_floor", self.DEFAULT_PRECISION_FLOOR))
        if abs(existing_floor - precision_floor) > 1e-12:
            policy = self._compute_policy(precision_floor=precision_floor)
            self._write_json(policy_path, policy)
        return policy

    def _compute_policy(self, precision_floor: float) -> Dict[str, object]:
        combo_path = self._path(self.COMBO_RESULTS_FILE)
        if not os.path.exists(combo_path):
            raise FileNotFoundError(
                f"Missing combo results at {combo_path}. Run rebuild once to generate research artifacts."
            )

        combo_df = pd.read_csv(combo_path)
        if combo_df.empty or "combo" not in combo_df.columns:
            raise ValueError(f"Combo results file at {combo_path} is empty or malformed.")

        combo_metrics = {}
        for combo, subset in combo_df.groupby("combo"):
            tp = int(pd.to_numeric(subset.get("tp"), errors="coerce").fillna(0).sum())
            fp = int(pd.to_numeric(subset.get("fp"), errors="coerce").fillna(0).sum())
            alerts = int(pd.to_numeric(subset.get("alerts"), errors="coerce").fillna(0).sum())
            covered = int(pd.to_numeric(subset.get("covered_events"), errors="coerce").fillna(0).sum())
            total = int(pd.to_numeric(subset.get("total_events"), errors="coerce").fillna(0).sum())
            precision = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
            recall = (covered / total) if total > 0 else np.nan
            combo_metrics[combo] = {
                "tp": tp,
                "fp": fp,
                "alerts": alerts,
                "precision": None if pd.isna(precision) else float(precision),
                "covered_events": covered,
                "total_events": total,
                "event_recall": None if pd.isna(recall) else float(recall),
            }

        actionable_combo = "vote_2of3" if "vote_2of3" in combo_metrics else ("or_any" if "or_any" in combo_metrics else None)
        watchlist_combo = "or_any" if "or_any" in combo_metrics else actionable_combo
        if actionable_combo is None:
            raise ValueError("No valid combo rows found in combo results.")

        actionable_precision = combo_metrics[actionable_combo].get("precision")
        enabled = actionable_precision is not None and actionable_precision >= precision_floor
        reason = "USE_ACTIONABLE" if enabled else "LOW_PRECISION"

        return {
            "generated_at_utc": self._to_iso(self._now_utc()),
            "config": {
                "precision_floor": float(precision_floor),
                "weekly_reeval_days": self.WEEKLY_REEVAL_DAYS,
            },
            "actionable": {
                "combo": actionable_combo,
                "enabled": bool(enabled),
                "reason_code": reason,
            },
            "watchlist": {
                "combo": watchlist_combo,
            },
            "combo_metrics": combo_metrics,
        }

    def _run_research(self, cooldown: int, seed: int, warmup: int, rule_search_samples: int) -> Dict[str, object]:
        if not os.path.exists(self.script_path):
            raise FileNotFoundError(f"Research script not found at {self.script_path}")
        cmd = [
            sys.executable,
            self.script_path,
            "--features-file",
            self.features_file,
            "--events-file",
            self.events_file,
            "--outdir",
            self.research_dir,
            "--cooldown",
            str(cooldown),
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--rule-search-samples",
            str(rule_search_samples),
        ]
        completed = subprocess.run(
            cmd,
            cwd=BASE_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "stdout_tail": completed.stdout[-2000:] if completed.stdout else "",
            "stderr_tail": completed.stderr[-2000:] if completed.stderr else "",
        }

    def _resolve_precision_floor(self, precision_floor: Optional[float], meta: Optional[Dict[str, object]] = None) -> float:
        if precision_floor is not None:
            return float(precision_floor)
        if meta and meta.get("precision_floor") is not None:
            return float(meta.get("precision_floor"))
        return float(self.DEFAULT_PRECISION_FLOOR)

    def _path(self, filename: str) -> str:
        return os.path.join(self.research_dir, filename)

    def _read_json(self, path: str, default):
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: str, payload: Dict[str, object]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def _now_utc() -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    @staticmethod
    def _to_iso(value: dt.datetime) -> str:
        return value.astimezone(dt.timezone.utc).isoformat()

    @staticmethod
    def _parse_iso(value) -> Optional[dt.datetime]:
        if not value or not isinstance(value, str):
            return None
        try:
            parsed = dt.datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
