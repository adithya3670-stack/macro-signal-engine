import datetime as dt
import inspect
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from analysis.regime_engine import RegimeStateEngine


class RegimeForecastEngine:
    """Forecasts next regime state using blended Markov + multinomial logistic models."""

    HORIZON_STEPS = {"3d": 3, "1w": 5, "1m": 20}
    MARKOV_ALPHA = 1.0
    DEFAULT_MARKOV_WEIGHT = 0.7
    LOGIT_MIN_ROWS = 80
    LOGIT_MIN_CLASS_ROWS = 12

    MACRO_FEATURES = ["VIX", "Liquidity_Impulse", "Real_Yield", "CPI_YoY", "Curve_Steepening"]
    GROUP_SHORT_FEATURE_SUFFIXES = ["_ret_1", "_ret_3", "_ret_5", "_vol_5", "_vol_10", "_trend_20", "_trend_60"]

    def build_forecasts(
        self,
        state_df: pd.DataFrame,
        feature_tables: Dict[str, pd.DataFrame],
        holdout_start: str = "2025-01-01",
        holdout_end: str = "2025-12-31",
        shadow_start: str = "2026-01-01",
    ) -> Dict[str, object]:
        holdout_start_ts = pd.Timestamp(holdout_start)
        holdout_end_ts = pd.Timestamp(holdout_end)
        shadow_start_ts = pd.Timestamp(shadow_start)

        horizons: Dict[str, Dict[str, object]] = {}
        latest_payload: Dict[str, Dict[str, object]] = {}

        for horizon, steps in self.HORIZON_STEPS.items():
            horizon_feature_df = feature_tables.get(horizon, pd.DataFrame(index=state_df.index))
            groups_payload: Dict[str, Dict[str, object]] = {}
            latest_payload[horizon] = {}

            for group, assets in RegimeStateEngine.ASSET_GROUPS.items():
                group_result = self._build_group_forecast(
                    horizon=horizon,
                    group=group,
                    steps=steps,
                    state_df=state_df,
                    feature_df=horizon_feature_df,
                    assets=assets,
                    holdout_start_ts=holdout_start_ts,
                    holdout_end_ts=holdout_end_ts,
                    shadow_start_ts=shadow_start_ts,
                )
                groups_payload[group] = group_result
                latest_payload[horizon][group] = group_result.get("latest", {})

            horizons[horizon] = {
                "horizon": horizon,
                "steps_ahead": steps,
                "groups": groups_payload,
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "horizons": horizons,
            "latest": latest_payload,
        }

    def _build_group_forecast(
        self,
        horizon: str,
        group: str,
        steps: int,
        state_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        assets: List[str],
        holdout_start_ts: pd.Timestamp,
        holdout_end_ts: pd.Timestamp,
        shadow_start_ts: pd.Timestamp,
    ) -> Dict[str, object]:
        index = pd.DatetimeIndex(state_df.index).sort_values()
        composite_col = f"{horizon}_{group}_composite_id"
        if composite_col not in state_df.columns or len(index) == 0:
            return self._empty_group_payload(horizon=horizon, group=group, steps=steps)

        states = pd.Series(state_df[composite_col], index=index).astype(str)
        future_dates = pd.Series(index, index=index).shift(-steps)
        target_states = states.shift(-steps)

        feature_frame = self._build_feature_frame(
            horizon=horizon,
            group=group,
            state_df=state_df.reindex(index),
            feature_df=feature_df.reindex(index),
            assets=assets,
        )

        base = pd.DataFrame(index=index)
        base["state"] = states
        base["target_state"] = target_states
        base["future_date"] = future_dates

        valid_target_mask = base["target_state"].notna() & base["future_date"].notna()
        train_mask = (base.index < holdout_start_ts) & valid_target_mask & (base["future_date"] < holdout_start_ts)
        holdout_mask = (
            (base.index >= holdout_start_ts)
            & (base.index <= holdout_end_ts)
            & valid_target_mask
            & (base["future_date"] <= holdout_end_ts)
        )
        shadow_mask = (base.index >= shadow_start_ts) & valid_target_mask

        train_rows = base.loc[train_mask]
        state_space = self._state_space(train_rows["state"], train_rows["target_state"])
        if not state_space:
            return self._empty_group_payload(horizon=horizon, group=group, steps=steps)

        markov_model = self._fit_markov(
            state_space=state_space,
            from_states=train_rows["state"].astype(str).tolist(),
            to_states=train_rows["target_state"].astype(str).tolist(),
        )

        logit_model = self._fit_logit(
            X=feature_frame.loc[train_mask],
            y=train_rows["target_state"].astype(str),
            state_space=state_space,
        )
        logit_model["blend_weight"] = self._calibrate_blend_weight(
            base_df=base,
            markov_model=markov_model,
            logit_model=logit_model,
            holdout_start_ts=holdout_start_ts,
        )

        all_masks = (train_mask | holdout_mask | shadow_mask) | (base.index == base.index[-1])
        pred_rows = []
        for ts in base.index[all_masks]:
            current_state = str(base.at[ts, "state"])
            markov_dist, transition_support = self._markov_distribution(
                markov_model=markov_model,
                current_state=current_state,
            )
            logit_dist = self._logit_distribution(logit_model=logit_model, x_row=feature_frame.loc[ts], state_space=state_space)
            logit_conf = max(logit_dist.values()) if logit_dist else None
            blend_weight = self._adaptive_blend_weight(
                logit_model=logit_model,
                logit_confidence=logit_conf,
                transition_support=transition_support,
            )
            blended = self._blend_distributions(markov_dist=markov_dist, logit_dist=logit_dist, markov_weight=blend_weight)
            top3 = self._topk(blended, k=3)
            pred_rows.append(
                {
                    "date": ts,
                    "future_date": base.at[ts, "future_date"],
                    "actual_state": base.at[ts, "target_state"] if pd.notna(base.at[ts, "target_state"]) else None,
                    "pred_state": top3[0]["key"] if top3 else None,
                    "top3": top3,
                    "confidence": float(top3[0]["prob"]) if top3 else 0.0,
                    "transition_support": int(transition_support),
                    "blend_markov_weight": float(blend_weight),
                    "probs": blended,
                }
            )

        holdout_diag = self._score_predictions(pred_rows=pred_rows, mask_dates=set(base.index[holdout_mask]), state_space=state_space)
        shadow_diag = self._score_predictions(pred_rows=pred_rows, mask_dates=set(base.index[shadow_mask]), state_space=state_space)

        latest_row = pred_rows[-1] if pred_rows else None
        latest = self._latest_payload(horizon=horizon, group=group, steps=steps, row=latest_row)

        return {
            "horizon": horizon,
            "group": group,
            "steps_ahead": steps,
            "train_rows": int(train_mask.sum()),
            "holdout_rows": int(holdout_mask.sum()),
            "shadow_rows": int(shadow_mask.sum()),
            "class_count": int(len(state_space)),
            "markov_alpha": float(self.MARKOV_ALPHA),
            "markov_state_count": int(len(markov_model["state_space"])),
            "logit": {
                "available": bool(logit_model["available"]),
                "reason": logit_model["reason"],
                "train_rows": int(logit_model["train_rows"]),
                "class_count": int(logit_model["class_count"]),
                "markov_weight_used": float(self._blend_weight(logit_model=logit_model)),
            },
            "diagnostics": {
                "holdout": holdout_diag,
                "shadow": shadow_diag,
            },
            "latest": latest,
        }

    def _build_feature_frame(
        self,
        horizon: str,
        group: str,
        state_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        assets: List[str],
    ) -> pd.DataFrame:
        def to_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
            if col in frame.columns:
                return pd.to_numeric(frame[col], errors="coerce")
            return pd.Series(default, index=frame.index, dtype=float)

        out = pd.DataFrame(index=state_df.index)

        rule_cols = ["rule_inflation_high", "rule_liquidity_expanding", "rule_risk_off", "rule_rates_positive"]
        for col in rule_cols:
            out[col] = to_series(state_df, col, default=0.0).fillna(0.0)

        latent_state_col = f"{horizon}_{group}_latent_state"
        latent_conf_col = f"{horizon}_{group}_latent_conf"
        out["latent_state"] = to_series(state_df, latent_state_col, default=-1.0).fillna(-1.0)
        out["latent_conf"] = to_series(state_df, latent_conf_col, default=0.0).fillna(0.0)

        for col in self.MACRO_FEATURES:
            out[col] = to_series(feature_df, col, default=0.0).replace([np.inf, -np.inf], np.nan)

        for asset in assets:
            for suffix in self.GROUP_SHORT_FEATURE_SUFFIXES:
                col = f"{asset}{suffix}"
                if col in feature_df.columns:
                    out[col] = pd.to_numeric(feature_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
        return out.astype(float)

    def _fit_markov(self, state_space: List[str], from_states: List[str], to_states: List[str]) -> Dict[str, object]:
        size = len(state_space)
        state_to_idx = {state: idx for idx, state in enumerate(state_space)}
        counts = np.zeros((size, size), dtype=float)

        for src, dst in zip(from_states, to_states):
            i = state_to_idx.get(str(src))
            j = state_to_idx.get(str(dst))
            if i is None or j is None:
                continue
            counts[i, j] += 1.0

        smoothed = counts + float(self.MARKOV_ALPHA)
        row_sums = smoothed.sum(axis=1, keepdims=True)
        probs = np.divide(smoothed, row_sums, out=np.zeros_like(smoothed), where=row_sums > 0)

        prior_counts = counts.sum(axis=0) + float(self.MARKOV_ALPHA)
        prior = prior_counts / np.clip(prior_counts.sum(), 1e-8, None)

        return {
            "state_space": state_space,
            "state_to_idx": state_to_idx,
            "probs": probs,
            "prior": prior,
            "row_support": counts.sum(axis=1),
        }

    def _fit_logit(self, X: pd.DataFrame, y: pd.Series, state_space: List[str]) -> Dict[str, object]:
        clean_y = y.astype(str)
        train_rows = int(len(clean_y))
        class_count = int(clean_y.nunique())
        min_rows = max(self.LOGIT_MIN_ROWS, class_count * self.LOGIT_MIN_CLASS_ROWS)

        if train_rows < min_rows:
            return {
                "available": False,
                "reason": f"insufficient_rows:{train_rows}<{min_rows}",
                "train_rows": train_rows,
                "class_count": class_count,
            }
        if class_count < 2:
            return {
                "available": False,
                "reason": "insufficient_classes",
                "train_rows": train_rows,
                "class_count": class_count,
            }

        scaler = RobustScaler()
        X_values = X.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0).values
        X_scaled = scaler.fit_transform(X_values)

        try:
            logistic_kwargs = {
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced",
            }
            # sklearn>=1.8 removed `multi_class`; keep compatibility across versions.
            if "multi_class" in inspect.signature(LogisticRegression).parameters:
                logistic_kwargs["multi_class"] = "multinomial"
            model = LogisticRegression(**logistic_kwargs)
            model.fit(X_scaled, clean_y.values)
            return {
                "available": True,
                "reason": "ok",
                "train_rows": train_rows,
                "class_count": class_count,
                "model": model,
                "scaler": scaler,
                "X_ref": X,
                "state_space": state_space,
            }
        except Exception as exc:
            return {
                "available": False,
                "reason": f"logit_fit_failed:{type(exc).__name__}:{str(exc)[:180]}",
                "train_rows": train_rows,
                "class_count": class_count,
            }

    def _markov_distribution(self, markov_model: Dict[str, object], current_state: str):
        state_to_idx = markov_model["state_to_idx"]
        probs = markov_model["probs"]
        row_support = markov_model["row_support"]
        if current_state in state_to_idx:
            idx = state_to_idx[current_state]
            row = probs[idx]
            support = int(row_support[idx])
        else:
            row = markov_model["prior"]
            support = 0
        return {state: float(row[i]) for i, state in enumerate(markov_model["state_space"])}, support

    def _logit_distribution(self, logit_model: Dict[str, object], x_row: pd.Series, state_space: List[str]) -> Optional[Dict[str, float]]:
        if not logit_model.get("available"):
            return None
        try:
            scaler = logit_model["scaler"]
            model = logit_model["model"]
            x_values = pd.to_numeric(x_row, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).values.reshape(1, -1)
            x_scaled = scaler.transform(x_values)
            probs = model.predict_proba(x_scaled)[0]
            out = {state: 0.0 for state in state_space}
            for cls, prob in zip(model.classes_, probs):
                out[str(cls)] = float(prob)
            return out
        except Exception:
            return None

    def _blend_weight(self, logit_model: Dict[str, object]) -> float:
        if not logit_model.get("available"):
            return 1.0
        return float(logit_model.get("blend_weight", self.DEFAULT_MARKOV_WEIGHT))

    def _adaptive_blend_weight(
        self,
        logit_model: Dict[str, object],
        logit_confidence: Optional[float],
        transition_support: Optional[int],
    ) -> float:
        base = self._blend_weight(logit_model=logit_model)
        if base >= 1.0:
            return 1.0

        conf_term = 0.0
        if logit_confidence is not None:
            conf_term = 0.08 * float(np.clip(0.5 - logit_confidence, -0.5, 0.5))

        trans_term = 0.0
        if transition_support is not None:
            if transition_support < 8:
                trans_term = 0.06
            elif transition_support < 20:
                trans_term = 0.03

        return float(np.clip(base + conf_term + trans_term, 0.62, 0.90))

    def _blend_distributions(
        self,
        markov_dist: Dict[str, float],
        logit_dist: Optional[Dict[str, float]],
        markov_weight: float,
    ) -> Dict[str, float]:
        if not logit_dist:
            total = np.clip(sum(markov_dist.values()), 1e-8, None)
            return {k: float(v / total) for k, v in markov_dist.items()}

        out = {}
        keys = sorted(set(markov_dist.keys()) | set(logit_dist.keys()))
        for key in keys:
            pm = float(markov_dist.get(key, 0.0))
            pl = float(logit_dist.get(key, 0.0))
            out[key] = float(markov_weight * pm + (1.0 - markov_weight) * pl)

        total = np.clip(sum(out.values()), 1e-8, None)
        return {k: float(v / total) for k, v in out.items()}

    def _score_predictions(self, pred_rows: List[Dict[str, object]], mask_dates: set, state_space: List[str]) -> Dict[str, object]:
        eval_rows = [
            row
            for row in pred_rows
            if row["date"] in mask_dates and row.get("actual_state") is not None and row.get("pred_state") is not None
        ]
        if not eval_rows:
            return {
                "rows": 0,
                "top1_accuracy": None,
                "top3_recall": None,
                "brier_score": None,
                "confidence_mean": None,
            }

        correct = 0
        top3_hits = 0
        brier_vals = []
        conf_vals = []
        state_to_idx = {state: idx for idx, state in enumerate(state_space)}

        for row in eval_rows:
            actual = str(row["actual_state"])
            pred = str(row["pred_state"])
            top3_keys = [str(item["key"]) for item in row.get("top3", [])]
            probs = row.get("probs", {})

            correct += int(pred == actual)
            top3_hits += int(actual in top3_keys)
            conf_vals.append(float(row.get("confidence", 0.0)))

            y_true = np.zeros(len(state_space), dtype=float)
            if actual in state_to_idx:
                y_true[state_to_idx[actual]] = 1.0
            y_prob = np.array([float(probs.get(state, 0.0)) for state in state_space], dtype=float)
            brier_vals.append(float(np.sum((y_prob - y_true) ** 2)))

        rows = len(eval_rows)
        return {
            "rows": int(rows),
            "top1_accuracy": float(correct / rows),
            "top3_recall": float(top3_hits / rows),
            "brier_score": float(np.mean(brier_vals)),
            "confidence_mean": float(np.mean(conf_vals)),
        }

    def _calibrate_blend_weight(
        self,
        base_df: pd.DataFrame,
        markov_model: Dict[str, object],
        logit_model: Dict[str, object],
        holdout_start_ts: pd.Timestamp,
    ) -> float:
        if not logit_model.get("available"):
            return 1.0

        cal_start = holdout_start_ts - pd.Timedelta(days=540)
        cal = base_df[
            (base_df.index >= cal_start)
            & (base_df.index < holdout_start_ts)
            & base_df["target_state"].notna()
        ]
        if len(cal) < 120:
            return float(self.DEFAULT_MARKOV_WEIGHT)

        state_space = markov_model["state_space"]
        state_to_idx = {state: i for i, state in enumerate(state_space)}
        markov_losses = []
        logit_losses = []

        for ts, row in cal.iterrows():
            actual = str(row["target_state"])
            if actual not in state_to_idx:
                continue
            if ts not in logit_model["X_ref"].index:
                continue
            markov_dist, _ = self._markov_distribution(markov_model=markov_model, current_state=str(row["state"]))
            logit_dist = self._logit_distribution(logit_model=logit_model, x_row=logit_model["X_ref"].loc[ts], state_space=state_space)
            if logit_dist is None:
                continue

            y_true = np.zeros(len(state_space), dtype=float)
            y_true[state_to_idx[actual]] = 1.0
            pm = np.array([markov_dist.get(s, 0.0) for s in state_space], dtype=float)
            pl = np.array([logit_dist.get(s, 0.0) for s in state_space], dtype=float)
            markov_losses.append(float(np.sum((pm - y_true) ** 2)))
            logit_losses.append(float(np.sum((pl - y_true) ** 2)))

        if len(markov_losses) < 80 or len(logit_losses) < 80:
            return float(self.DEFAULT_MARKOV_WEIGHT)

        markov_brier = float(np.mean(markov_losses))
        logit_brier = float(np.mean(logit_losses))
        if logit_brier + 0.01 < markov_brier:
            return 0.60
        if markov_brier + 0.01 < logit_brier:
            return 0.85
        return float(self.DEFAULT_MARKOV_WEIGHT)

    def _latest_payload(self, horizon: str, group: str, steps: int, row: Optional[Dict[str, object]]) -> Dict[str, object]:
        if not row:
            return {
                "horizon": horizon,
                "group": group,
                "as_of_date": None,
                "target_date": None,
                "selected_key": None,
                "confidence": 0.0,
                "transition_support": 0,
                "top3": [],
            }
        as_of_date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        future_date = row.get("future_date")
        if pd.isna(future_date):
            target_date = pd.bdate_range(pd.Timestamp(row["date"]), periods=steps + 1)[-1].strftime("%Y-%m-%d")
        else:
            target_date = pd.Timestamp(future_date).strftime("%Y-%m-%d")
        return {
            "horizon": horizon,
            "group": group,
            "as_of_date": as_of_date,
            "target_date": target_date,
            "selected_key": row.get("pred_state"),
            "confidence": float(row.get("confidence", 0.0)),
            "transition_support": int(row.get("transition_support", 0)),
            "top3": row.get("top3", []),
        }

    def _empty_group_payload(self, horizon: str, group: str, steps: int) -> Dict[str, object]:
        return {
            "horizon": horizon,
            "group": group,
            "steps_ahead": steps,
            "train_rows": 0,
            "holdout_rows": 0,
            "shadow_rows": 0,
            "class_count": 0,
            "markov_alpha": float(self.MARKOV_ALPHA),
            "markov_state_count": 0,
            "logit": {"available": False, "reason": "no_data", "train_rows": 0, "class_count": 0, "markov_weight_used": 1.0},
            "diagnostics": {
                "holdout": {"rows": 0, "top1_accuracy": None, "top3_recall": None, "brier_score": None, "confidence_mean": None},
                "shadow": {"rows": 0, "top1_accuracy": None, "top3_recall": None, "brier_score": None, "confidence_mean": None},
            },
            "latest": {
                "horizon": horizon,
                "group": group,
                "as_of_date": None,
                "target_date": None,
                "selected_key": None,
                "confidence": 0.0,
                "transition_support": 0,
                "top3": [],
            },
        }

    def _state_space(self, current_states: pd.Series, next_states: pd.Series) -> List[str]:
        universe = pd.Index(current_states.astype(str).tolist() + next_states.astype(str).tolist()).dropna().unique().tolist()
        universe = [x for x in universe if x and str(x).lower() != "nan"]
        universe.sort()
        return universe

    @staticmethod
    def _topk(probs: Dict[str, float], k: int = 3) -> List[Dict[str, object]]:
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [{"key": key, "prob": float(round(prob, 6))} for key, prob in ranked]
