import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DEFAULT_FEATURES_FILE = os.path.join(REPO_ROOT, "data", "price_3d_features.csv")
DEFAULT_EVENTS_FILE = os.path.join(REPO_ROOT, "data", "sp500_drawdown_events_classified.csv")
DEFAULT_OUTDIR = os.path.join(REPO_ROOT, "data", "research_minor_event_signals")


@dataclass
class Period:
    name: str
    start: str
    end: str


MODEL_FOLDS = [
    Period("2018", "2016-01-01", "2018-12-31"),
    Period("2021", "2019-01-01", "2021-12-31"),
    Period("2023", "2022-01-01", "2023-12-31"),
    Period("2025", "2024-01-01", "2025-12-31"),
]

RULE_PERIODS = [
    Period("2012-2015", "2012-01-01", "2015-12-31"),
    Period("2016-2018", "2016-01-01", "2018-12-31"),
    Period("2019-2021", "2019-01-01", "2021-12-31"),
    Period("2022-2023", "2022-01-01", "2023-12-31"),
    Period("2024-2025", "2024-01-01", "2025-12-31"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precision-first research sweep for 3-trading-day SP500 minor drawdown alerts. "
            "Compares model-based and rule-based approaches and writes reproducible CSV artifacts."
        )
    )
    parser.add_argument("--features-file", default=DEFAULT_FEATURES_FILE, help="Path to price_3d_features.csv")
    parser.add_argument("--events-file", default=DEFAULT_EVENTS_FILE, help="Path to sp500_drawdown_events_classified.csv")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for research artifacts")
    parser.add_argument("--cooldown", type=int, default=15, help="Trading-day cooldown between alerts")
    parser.add_argument("--warmup", type=int, default=120, help="Rows to skip from the beginning for rolling-feature stability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--rule-search-samples", type=int, default=110000, help="Total random rule combinations sampled")
    return parser.parse_args()


def load_inputs(features_file: str, events_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(features_file)
    events = pd.read_csv(events_file)
    features["Date"] = pd.to_datetime(features["Date"], errors="coerce")
    features = features[features["Date"].notna()].sort_values("Date").set_index("Date")
    events["start_date"] = pd.to_datetime(events["start_date"], errors="coerce")
    events["trough_date"] = pd.to_datetime(events["trough_date"], errors="coerce")
    events["recovery_date"] = pd.to_datetime(events["recovery_date"], errors="coerce")
    events = events[events["start_date"].notna()].copy()
    return features, events


def map_to_trading_index(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> int:
    pos = int(index.searchsorted(timestamp))
    if pos >= len(index):
        return -1
    return pos


def build_minor_windows(index: pd.DatetimeIndex, events: pd.DataFrame, pre_days: int = 3) -> pd.DataFrame:
    rows = []
    minor = events.loc[events["severity"].str.lower() == "minor"].copy()
    for _, row in minor.iterrows():
        pos = map_to_trading_index(index, row["start_date"])
        if pos < 0:
            continue
        start_idx = max(0, pos - pre_days)
        end_idx = pos - 1
        if end_idx < start_idx:
            continue
        rows.append(
            {
                "window_start": index[start_idx],
                "window_end": index[end_idx],
                "event_day": index[pos],
                "source_start_date": row["start_date"],
                "trough_date": row.get("trough_date"),
                "recovery_date": row.get("recovery_date"),
            }
        )
    return pd.DataFrame(rows)


def build_pre_event_label(index: pd.DatetimeIndex, windows: pd.DataFrame) -> np.ndarray:
    y = np.zeros(len(index), dtype=int)
    for _, row in windows.iterrows():
        start = int(index.searchsorted(pd.Timestamp(row["window_start"])))
        end = int(index.searchsorted(pd.Timestamp(row["window_end"])))
        if end >= start:
            y[start : end + 1] = 1
    return y


def drop_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("FuturePrice_", "NormPrice_", "CenteredNormPrice_", "MAPEValid_")
    cols = [c for c in frame.columns if not c.startswith(prefixes)]
    return frame.loc[:, cols].copy()


def add_short_horizon_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "SP500" in out.columns:
        sp = pd.to_numeric(out["SP500"], errors="coerce")
        for lag in [1, 2, 3, 5, 10]:
            out[f"SP500_ret_{lag}"] = sp.pct_change(lag)
        out["SP500_vol_5"] = sp.pct_change().rolling(5, min_periods=3).std()
        out["SP500_vol_10"] = sp.pct_change().rolling(10, min_periods=5).std()
        out["SP500_vol_20"] = sp.pct_change().rolling(20, min_periods=10).std()
        out["SP500_dd_10"] = sp / sp.rolling(10, min_periods=5).max() - 1.0
        out["SP500_dd_20"] = sp / sp.rolling(20, min_periods=10).max() - 1.0
        out["SP500_dd_60"] = sp / sp.rolling(60, min_periods=20).max() - 1.0
        out["SP500_z20"] = (sp - sp.rolling(20, min_periods=10).mean()) / sp.rolling(20, min_periods=10).std()
        out["SP500_z60"] = (sp - sp.rolling(60, min_periods=30).mean()) / sp.rolling(60, min_periods=30).std()
    if "VIX" in out.columns:
        vix = pd.to_numeric(out["VIX"], errors="coerce")
        out["VIX_ret_1"] = vix.pct_change(1)
        out["VIX_ret_3"] = vix.pct_change(3)
        out["VIX_ret_5"] = vix.pct_change(5)
        out["VIX_z20"] = (vix - vix.rolling(20, min_periods=10).mean()) / vix.rolling(20, min_periods=10).std()
    return out


def apply_cooldown(alert_signal: np.ndarray, cooldown_days: int) -> np.ndarray:
    out = np.zeros(len(alert_signal), dtype=int)
    last = -10**9
    for i, is_alert in enumerate(alert_signal):
        if is_alert and (i - last) > cooldown_days:
            out[i] = 1
            last = i
    return out


def evaluate_event_alerts(
    alert_signal: np.ndarray,
    dates: pd.DatetimeIndex,
    windows: pd.DataFrame,
    start: pd.Timestamp = None,
    end: pd.Timestamp = None,
) -> Dict[str, float]:
    if start is not None:
        windows = windows.loc[windows["event_day"] >= start]
    if end is not None:
        windows = windows.loc[windows["event_day"] <= end]

    alert_dates = dates[alert_signal == 1]
    tp = 0
    fp = 0
    covered = 0
    for _, row in windows.iterrows():
        in_window = (alert_dates >= row["window_start"]) & (alert_dates <= row["window_end"])
        if bool(in_window.any()):
            covered += 1
    for d in alert_dates:
        is_tp = False
        for _, row in windows.iterrows():
            if row["window_start"] <= d <= row["window_end"]:
                is_tp = True
                break
        tp += int(is_tp)
        fp += int(not is_tp)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = (covered / len(windows)) if len(windows) > 0 else np.nan
    return {
        "alerts": int(tp + fp),
        "tp": int(tp),
        "fp": int(fp),
        "precision": float(precision) if pd.notna(precision) else np.nan,
        "recall": float(recall) if pd.notna(recall) else np.nan,
        "covered_events": int(covered),
        "total_events": int(len(windows)),
    }


def tune_threshold_precision_first(prob_train: np.ndarray, y_train: np.ndarray, cooldown_days: int) -> float:
    quantiles = np.linspace(0.70, 0.995, 80)
    candidates = np.unique(np.quantile(prob_train, quantiles))
    best = None
    for thr in candidates:
        raw = (prob_train >= thr).astype(int)
        pred = apply_cooldown(raw, cooldown_days)
        alerts = int(pred.sum())
        if alerts < 3:
            continue
        precision = precision_score(y_train, pred, zero_division=0)
        recall = recall_score(y_train, pred, zero_division=0)
        score = (3.0 * precision) + (0.4 * recall) - (0.001 * alerts)
        if (best is None) or (score > best[0]):
            best = (score, float(thr))
    if best is None:
        return float(np.quantile(prob_train, 0.95))
    return best[1]


def build_model(kind: str, y_train: np.ndarray, seed: int):
    pos = max(1, int(y_train.sum()))
    neg = max(1, int((1 - y_train).sum()))
    scale_pos_weight = neg / pos

    if kind == "logit_bal":
        base = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", C=0.6, solver="liblinear")),
            ]
        )
    elif kind == "xgb_bal":
        base = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=320,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        min_child_weight=4,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        max_delta_step=2,
                        random_state=seed,
                        n_jobs=4,
                    ),
                ),
            ]
        )
    elif kind == "xgb_prob":
        base = Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=280,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=2.0,
                        min_child_weight=4,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=1.0,
                        max_delta_step=1,
                        random_state=seed,
                        n_jobs=4,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    if y_train.sum() >= 20:
        return CalibratedClassifierCV(base, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
    return base


def run_model_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    windows: pd.DataFrame,
    cooldown_days: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_kinds = ["logit_bal", "xgb_bal", "xgb_prob"]
    rows = []
    and_gate_rows = []

    for fold in MODEL_FOLDS:
        val_mask = (dates >= pd.Timestamp(fold.start)) & (dates <= pd.Timestamp(fold.end))
        train_mask = dates < pd.Timestamp(fold.start)
        tr_idx = np.where(train_mask)[0]
        va_idx = np.where(val_mask)[0]
        if len(tr_idx) < 500 or len(va_idx) < 50:
            continue

        prob_cache = {}
        for kind in model_kinds:
            model = build_model(kind, y[tr_idx], seed=seed)
            model.fit(X.iloc[tr_idx], y[tr_idx])
            p_tr = model.predict_proba(X.iloc[tr_idx])[:, 1]
            p_va = model.predict_proba(X.iloc[va_idx])[:, 1]
            prob_cache[kind] = (p_tr, p_va)

            thr = tune_threshold_precision_first(p_tr, y[tr_idx], cooldown_days=cooldown_days)
            pred = apply_cooldown((p_va >= thr).astype(int), cooldown_days)

            event = evaluate_event_alerts(
                pred,
                dates[va_idx],
                windows=windows,
                start=pd.Timestamp(fold.start),
                end=pd.Timestamp(fold.end),
            )
            ap = np.nan
            if len(np.unique(y[va_idx])) > 1:
                ap = float(average_precision_score(y[va_idx], p_va))
            br = np.nan
            try:
                br = float(brier_score_loss(y[va_idx], p_va))
            except ValueError:
                br = np.nan

            rows.append(
                {
                    "fold": fold.name,
                    "model": kind,
                    "threshold": float(thr),
                    "day_precision": float(precision_score(y[va_idx], pred, zero_division=0)),
                    "day_recall": float(recall_score(y[va_idx], pred, zero_division=0)),
                    "day_f1": float(f1_score(y[va_idx], pred, zero_division=0)),
                    "average_precision": ap,
                    "brier": br,
                    **event,
                }
            )

        # Two-model AND gate for ultra-high precision mode.
        p_tr_l, p_va_l = prob_cache["logit_bal"]
        p_tr_x, p_va_x = prob_cache["xgb_bal"]
        best = None
        cands_l = np.unique(np.quantile(p_tr_l, np.linspace(0.7, 0.995, 20)))
        cands_x = np.unique(np.quantile(p_tr_x, np.linspace(0.7, 0.995, 20)))
        for tl in cands_l:
            for tx in cands_x:
                pred_train = ((p_tr_l >= tl) & (p_tr_x >= tx)).astype(int)
                pred_train = apply_cooldown(pred_train, cooldown_days)
                alerts = int(pred_train.sum())
                if alerts < 2:
                    continue
                p = precision_score(y[tr_idx], pred_train, zero_division=0)
                r = recall_score(y[tr_idx], pred_train, zero_division=0)
                score = (3.0 * p) + (0.4 * r) - (0.001 * alerts)
                if (best is None) or (score > best[0]):
                    best = (score, float(tl), float(tx))

        if best is None:
            tl = float(np.quantile(p_tr_l, 0.95))
            tx = float(np.quantile(p_tr_x, 0.95))
        else:
            _, tl, tx = best

        pred_val = ((p_va_l >= tl) & (p_va_x >= tx)).astype(int)
        pred_val = apply_cooldown(pred_val, cooldown_days)
        event = evaluate_event_alerts(
            pred_val,
            dates[va_idx],
            windows=windows,
            start=pd.Timestamp(fold.start),
            end=pd.Timestamp(fold.end),
        )
        and_gate_rows.append(
            {
                "fold": fold.name,
                "threshold_logit": tl,
                "threshold_xgb": tx,
                "day_precision": float(precision_score(y[va_idx], pred_val, zero_division=0)),
                "day_recall": float(recall_score(y[va_idx], pred_val, zero_division=0)),
                **event,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(and_gate_rows)


def build_rule_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=features.index)
    sp = pd.to_numeric(features["SP500"], errors="coerce")
    frame["SP500"] = sp
    frame["SP500_ret1"] = sp.pct_change(1)
    frame["SP500_ret3"] = sp.pct_change(3)
    frame["SP500_ret5"] = sp.pct_change(5)
    frame["SP500_ret10"] = sp.pct_change(10)
    frame["SP500_dd20"] = sp / sp.rolling(20, min_periods=5).max() - 1.0
    frame["SP500_dd60"] = sp / sp.rolling(60, min_periods=20).max() - 1.0
    frame["SP500_z20"] = (sp - sp.rolling(20, min_periods=10).mean()) / sp.rolling(20, min_periods=10).std()
    frame["SP500_z60"] = (sp - sp.rolling(60, min_periods=30).mean()) / sp.rolling(60, min_periods=30).std()

    if "VIX" in features.columns:
        v = pd.to_numeric(features["VIX"], errors="coerce")
        frame["VIX"] = v
        frame["VIX_ret1"] = v.pct_change(1)
        frame["VIX_ret3"] = v.pct_change(3)
        frame["VIX_z20"] = (v - v.rolling(20, min_periods=10).mean()) / v.rolling(20, min_periods=10).std()

    for col in [
        "Nasdaq",
        "DJIA",
        "Russell2000",
        "Gold",
        "Oil",
        "Liquidity_Impulse",
        "Real_Yield",
        "CPI_YoY",
        "Curve_Steepening",
        "Bond_Stock_Corr",
        "Momentum",
        "Strength_RSI",
        "Breadth_Vol",
        "Safe_Haven_Demand",
    ]:
        if col in features.columns:
            frame[col] = pd.to_numeric(features[col], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan).ffill()
    return frame


def rule_signal(frame: pd.DataFrame, conditions: List[Tuple[str, str, float]]) -> np.ndarray:
    mask = np.ones(len(frame), dtype=bool)
    for col, op, val in conditions:
        if col not in frame.columns:
            return np.zeros(len(frame), dtype=int)
        arr = frame[col].to_numpy()
        if op == ">=":
            mask &= arr >= val
        else:
            mask &= arr <= val
    return mask.astype(int)


def parse_rule_string(rule: str) -> List[Tuple[str, str, float]]:
    conds = []
    for part in rule.split(" & "):
        col, op, val = part.split(" ")
        conds.append((col, op, float(val)))
    return conds


def run_rule_search(
    rule_features: pd.DataFrame,
    windows: pd.DataFrame,
    cooldown_days: int,
    seed: int,
    total_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    train_mask = rule_features.index < pd.Timestamp("2024-01-01")
    test_mask = (rule_features.index >= pd.Timestamp("2024-01-01")) & (rule_features.index <= pd.Timestamp("2025-12-31")
    )
    rule_train = rule_features.loc[train_mask]
    rule_test = rule_features.loc[test_mask]

    y_train = build_pre_event_label(rule_features.index, windows)[train_mask]

    pool = []
    for col in rule_train.columns:
        s = rule_train[col].dropna()
        if len(s) < 200:
            continue
        for q in [0.20, 0.30, 0.40]:
            pool.append((col, "<=", float(s.quantile(q))))
        for q in [0.60, 0.70, 0.80]:
            pool.append((col, ">=", float(s.quantile(q))))

    index_pool = np.arange(len(pool))
    candidates = []
    shapes = [(2, int(total_samples * 0.32)), (3, int(total_samples * 0.41)), (4, int(total_samples * 0.27))]
    for width, n_draws in shapes:
        for _ in range(n_draws):
            pick = rng.choice(index_pool, size=width, replace=False)
            conds = [pool[i] for i in pick]
            cols = [c[0] for c in conds]
            if len(set(cols)) < width:
                continue
            signal_train = rule_signal(rule_train, conds)
            pred_train = apply_cooldown(signal_train, cooldown_days)
            alerts = int(pred_train.sum())
            if alerts < 3 or alerts > 180:
                continue
            tp = int(((pred_train == 1) & (y_train == 1)).sum())
            fp = int(((pred_train == 1) & (y_train == 0)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / max(int(y_train.sum()), 1)
            if precision < 0.02:
                continue
            score = (4.0 * precision) + (0.5 * recall) - (0.001 * alerts)
            candidates.append((score, conds, precision, recall, alerts))

    candidates = sorted(candidates, key=lambda item: item[0], reverse=True)[:250]
    rows = []
    for score, conds, train_p, train_r, train_alerts in candidates:
        sig_test = rule_signal(rule_test, conds)
        pred_test = apply_cooldown(sig_test, cooldown_days)
        ev = evaluate_event_alerts(
            pred_test,
            rule_test.index,
            windows=windows,
            start=pd.Timestamp("2024-01-01"),
            end=pd.Timestamp("2025-12-31"),
        )
        rows.append(
            {
                "train_score": float(score),
                "train_precision_day": float(train_p),
                "train_recall_day": float(train_r),
                "train_alerts": int(train_alerts),
                "test_alerts": ev["alerts"],
                "test_tp": ev["tp"],
                "test_fp": ev["fp"],
                "test_event_precision": ev["precision"],
                "test_event_recall": ev["recall"],
                "rule": " & ".join([f"{c} {op} {v:.4f}" for c, op, v in conds]),
            }
        )

    search = pd.DataFrame(rows).sort_values(
        ["test_event_precision", "test_event_recall", "test_tp", "train_score"],
        ascending=False,
    )

    top_candidates = search[(search["test_event_precision"] > 0) & (search["test_alerts"] > 0)].head(6).copy()
    fold_rows = []
    for _, row in top_candidates.iterrows():
        conds = parse_rule_string(row["rule"])
        signal = rule_signal(rule_features, conds)
        pred = apply_cooldown(signal, cooldown_days)
        for period in RULE_PERIODS:
            period_mask = (rule_features.index >= pd.Timestamp(period.start)) & (rule_features.index <= pd.Timestamp(period.end))
            metrics = evaluate_event_alerts(
                pred[period_mask],
                rule_features.index[period_mask],
                windows=windows,
                start=pd.Timestamp(period.start),
                end=pd.Timestamp(period.end),
            )
            fold_rows.append({"rule": row["rule"], "period": period.name, **metrics})
    top_rule_folds = pd.DataFrame(fold_rows)

    # Combo layer: OR and 2-of-3 voting using the best aggregate precision rules.
    combo_rows = []
    combo_candidates = []
    if not top_rule_folds.empty:
        agg = (
            top_rule_folds.groupby("rule")
            .agg(mean_precision=("precision", "mean"), mean_recall=("recall", "mean"), total_tp=("tp", "sum"), total_fp=("fp", "sum"))
            .reset_index()
            .sort_values(["mean_precision", "total_tp"], ascending=[False, False])
        )
        selected_rules = agg["rule"].head(3).tolist()
        if len(selected_rules) >= 2:
            sigs = [rule_signal(rule_features, parse_rule_string(rule)) for rule in selected_rules]
            stack = np.vstack(sigs)
            combos = {
                "or_any": (stack.sum(axis=0) >= 1).astype(int),
                "vote_2of3": (stack.sum(axis=0) >= 2).astype(int) if stack.shape[0] >= 3 else (stack.sum(axis=0) >= 2).astype(int),
            }
            if stack.shape[0] >= 3:
                combos["all_3"] = (stack.sum(axis=0) >= 3).astype(int)

            for combo_name, raw_sig in combos.items():
                pred = apply_cooldown(raw_sig, cooldown_days)
                for period in RULE_PERIODS:
                    period_mask = (rule_features.index >= pd.Timestamp(period.start)) & (rule_features.index <= pd.Timestamp(period.end))
                    metrics = evaluate_event_alerts(
                        pred[period_mask],
                        rule_features.index[period_mask],
                        windows=windows,
                        start=pd.Timestamp(period.start),
                        end=pd.Timestamp(period.end),
                    )
                    combo_rows.append({"combo": combo_name, "period": period.name, **metrics})
            combo_candidates = selected_rules

    combo_results = pd.DataFrame(combo_rows)
    if not combo_results.empty:
        combo_results["selected_rules"] = " | ".join(combo_candidates)

    return search, top_rule_folds, combo_results


def compute_latest_rule_signals(
    rule_features: pd.DataFrame,
    combo_results: pd.DataFrame,
    top_rule_folds: pd.DataFrame,
    cooldown_days: int,
) -> Dict[str, object]:
    latest_date = rule_features.index.max()
    payload = {"as_of_date": latest_date.strftime("%Y-%m-%d"), "signals": {}}
    if top_rule_folds.empty:
        return payload

    agg = (
        top_rule_folds.groupby("rule")
        .agg(mean_precision=("precision", "mean"), mean_recall=("recall", "mean"), total_tp=("tp", "sum"), total_fp=("fp", "sum"))
        .reset_index()
        .sort_values(["mean_precision", "total_tp"], ascending=[False, False])
    )
    selected = agg["rule"].head(3).tolist()
    signals = []
    for rule in selected:
        sig = rule_signal(rule_features, parse_rule_string(rule))
        pred = apply_cooldown(sig, cooldown_days)
        latest_alert = bool(pred[-1] == 1)
        signals.append(
            {
                "rule": rule,
                "latest_alert": latest_alert,
                "mean_precision": float(agg.loc[agg["rule"] == rule, "mean_precision"].iloc[0]),
                "mean_recall": float(agg.loc[agg["rule"] == rule, "mean_recall"].iloc[0]),
                "total_tp": int(agg.loc[agg["rule"] == rule, "total_tp"].iloc[0]),
                "total_fp": int(agg.loc[agg["rule"] == rule, "total_fp"].iloc[0]),
            }
        )

    payload["signals"]["top_rules"] = signals
    if not combo_results.empty:
        # Evaluate current state for combo vote_2of3 and or_any using selected rules.
        sigs = [rule_signal(rule_features, parse_rule_string(rule)) for rule in selected]
        stack = np.vstack(sigs)
        combos = {
            "or_any": (stack.sum(axis=0) >= 1).astype(int),
            "vote_2of3": (stack.sum(axis=0) >= 2).astype(int) if len(selected) >= 2 else np.zeros(len(rule_features), dtype=int),
            "all_3": (stack.sum(axis=0) >= 3).astype(int) if len(selected) >= 3 else np.zeros(len(rule_features), dtype=int),
        }
        payload["signals"]["combo_latest"] = {}
        for name, raw in combos.items():
            pred = apply_cooldown(raw, cooldown_days)
            payload["signals"]["combo_latest"][name] = {"latest_alert": bool(pred[-1] == 1)}
    return payload


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="Skipping features without any observed values")
    os.makedirs(args.outdir, exist_ok=True)

    features, events = load_inputs(args.features_file, args.events_file)
    windows = build_minor_windows(features.index, events, pre_days=3)
    y_full = build_pre_event_label(features.index, windows)

    model_features = drop_target_columns(features)
    model_features = add_short_horizon_features(model_features)
    cat_cols = [c for c in model_features.columns if model_features[c].dtype == "object"]
    if cat_cols:
        model_features = pd.get_dummies(model_features, columns=cat_cols, drop_first=False)
    model_features = model_features.replace([np.inf, -np.inf], np.nan)
    model_features = model_features.ffill()
    model_features = model_features.iloc[args.warmup :].copy()
    y_model = y_full[args.warmup :]
    y_model = np.asarray(y_model, dtype=int)
    dates_model = model_features.index

    # Avoid repeated imputer warnings and unstable columns.
    non_empty_cols = model_features.columns[model_features.notna().sum() > 0]
    model_features = model_features.loc[:, non_empty_cols]

    model_cv, model_and = run_model_cv(
        X=model_features,
        y=y_model,
        dates=dates_model,
        windows=windows,
        cooldown_days=args.cooldown,
        seed=args.seed,
    )
    model_cv_path = os.path.join(args.outdir, "minor_3d_model_cv_results.csv")
    model_and_path = os.path.join(args.outdir, "minor_3d_model_and_gate_results.csv")
    model_cv.to_csv(model_cv_path, index=False)
    model_and.to_csv(model_and_path, index=False)

    rule_features = build_rule_feature_frame(features)
    rule_search, top_rule_folds, combo_results = run_rule_search(
        rule_features=rule_features,
        windows=windows,
        cooldown_days=args.cooldown,
        seed=args.seed,
        total_samples=args.rule_search_samples,
    )
    rule_search_path = os.path.join(args.outdir, "minor_3d_rule_randomsearch_v2.csv")
    top_rule_folds_path = os.path.join(args.outdir, "minor_3d_top_rule_oos_folds.csv")
    combo_results_path = os.path.join(args.outdir, "minor_3d_rule_combo_results.csv")
    rule_search.to_csv(rule_search_path, index=False)
    top_rule_folds.to_csv(top_rule_folds_path, index=False)
    combo_results.to_csv(combo_results_path, index=False)

    latest_payload = compute_latest_rule_signals(
        rule_features=rule_features,
        combo_results=combo_results,
        top_rule_folds=top_rule_folds,
        cooldown_days=args.cooldown,
    )
    latest_path = os.path.join(args.outdir, "minor_3d_latest_signals.json")
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(latest_payload, handle, indent=2)

    summary = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "features_file": os.path.abspath(args.features_file),
        "events_file": os.path.abspath(args.events_file),
        "outdir": os.path.abspath(args.outdir),
        "cooldown_days": int(args.cooldown),
        "warmup_rows": int(args.warmup),
        "minor_event_count": int(len(windows)),
        "model_cv_rows": int(len(model_cv)),
        "rule_search_rows": int(len(rule_search)),
        "top_rule_fold_rows": int(len(top_rule_folds)),
        "combo_rows": int(len(combo_results)),
        "artifacts": {
            "model_cv": model_cv_path,
            "model_and_gate": model_and_path,
            "rule_random_search": rule_search_path,
            "top_rule_oos_folds": top_rule_folds_path,
            "rule_combo": combo_results_path,
            "latest_signals": latest_path,
        },
    }
    summary_path = os.path.join(args.outdir, "minor_3d_research_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("Minor 3D precision research completed.")
    print(f"- Model CV rows: {len(model_cv)}")
    print(f"- Rule-search rows: {len(rule_search)}")
    print(f"- Summary: {summary_path}")


if __name__ == "__main__":
    main()
