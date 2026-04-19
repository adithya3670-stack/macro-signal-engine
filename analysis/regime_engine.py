import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler


class RegimeStateEngine:
    """Builds rule-based and latent regime states for each horizon/asset-group."""

    ASSET_GROUPS = {
        "equities": ["SP500", "Nasdaq", "DJIA", "Russell2000"],
        "precious": ["Gold", "Silver"],
        "commodities": ["Copper", "Oil"],
    }

    HORIZON_DWELL = {"3d": 3, "1w": 5, "1m": 10}
    CONFIDENCE_FLOOR = 0.55

    BASE_MACRO_FEATURES = [
        "VIX",
        "Liquidity_Impulse",
        "Real_Yield",
        "CPI_YoY",
        "Curve_Steepening",
        "Bond_Stock_Corr",
        "PPI_YoY",
        "UNRATE",
        "PAYEMS",
        "DGS10",
    ]

    ASSET_FEATURE_SUFFIXES = [
        "_ret_1",
        "_ret_3",
        "_ret_5",
        "_ret_10",
        "_vol_5",
        "_vol_10",
        "_vol_20",
        "_drawdown_20",
        "_drawdown_60",
        "_trend_20",
        "_trend_60",
        "_zscore_20",
        "_zscore_60",
    ]

    def build_state_history(self, master_df: pd.DataFrame, feature_tables: dict, holdout_start: str = "2025-01-01") -> pd.DataFrame:
        master = self._ensure_datetime_index(master_df)
        states = self._build_rule_states(master)
        states = states.reindex(master.index).ffill().bfill()

        holdout_ts = pd.Timestamp(holdout_start)

        for horizon, table in feature_tables.items():
            features = self._ensure_datetime_index(table)
            features = features.reindex(states.index).ffill().bfill()

            states[f"{horizon}_rule_id"] = states["rule_code"].map(lambda code: f"H={horizon}|R={code}")

            for group, assets in self.ASSET_GROUPS.items():
                latent_frame = self._build_latent_frame(features, assets)
                labels, conf = self._fit_predict_latent(latent_frame, holdout_ts)
                labels = self._apply_stability(
                    labels=labels,
                    confidence=conf,
                    dwell_days=self.HORIZON_DWELL.get(horizon, 3),
                    conf_floor=self.CONFIDENCE_FLOOR,
                )

                state_col = f"{horizon}_{group}_latent_state"
                conf_col = f"{horizon}_{group}_latent_conf"
                latent_id_col = f"{horizon}_{group}_latent_id"
                composite_col = f"{horizon}_{group}_composite_id"

                states[state_col] = labels.astype(int)
                states[conf_col] = conf.astype(float)
                states[latent_id_col] = states[state_col].map(lambda val: f"H={horizon}|G={group}|L={int(val)}")
                states[composite_col] = states.apply(
                    lambda row: f"H={horizon}|G={group}|R={row['rule_code']}|L={int(row[state_col])}",
                    axis=1,
                )

        return states

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            if "Date" in out.columns:
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                out = out.set_index("Date")
            elif "Unnamed: 0" in out.columns:
                out = out.rename(columns={"Unnamed: 0": "Date"})
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                out = out.set_index("Date")
            else:
                raise ValueError("Expected DatetimeIndex or Date column for regime state build.")
        out = out[out.index.notna()].sort_index()
        return out

    def _build_rule_states(self, master: pd.DataFrame) -> pd.DataFrame:
        idx = master.index

        inflation_high = self._binary_state(
            label_series=master.get("Regime_Inflation"),
            label_positive_token="High",
            numeric_series=master.get("CPI_YoY"),
            numeric_threshold=3.0,
            index=idx,
        )
        liquidity_expanding = self._binary_state(
            label_series=master.get("Regime_Liquidity"),
            label_positive_token="Expanding",
            numeric_series=master.get("Liquidity_Impulse"),
            numeric_threshold=0.0,
            index=idx,
        )
        risk_off = self._binary_state(
            label_series=master.get("Regime_Risk"),
            label_positive_token="Risk Off",
            numeric_series=master.get("VIX"),
            numeric_threshold=20.0,
            index=idx,
        )
        rates_positive = self._binary_state(
            label_series=master.get("Regime_Rates"),
            label_positive_token="Positive",
            numeric_series=master.get("Real_Yield"),
            numeric_threshold=0.0,
            index=idx,
        )

        out = pd.DataFrame(index=idx)
        out["rule_inflation_high"] = inflation_high
        out["rule_liquidity_expanding"] = liquidity_expanding
        out["rule_risk_off"] = risk_off
        out["rule_rates_positive"] = rates_positive
        out["rule_code"] = (
            "I"
            + inflation_high.astype(int).astype(str)
            + "L"
            + liquidity_expanding.astype(int).astype(str)
            + "R"
            + risk_off.astype(int).astype(str)
            + "T"
            + rates_positive.astype(int).astype(str)
        )
        return out

    def _binary_state(self, label_series, label_positive_token: str, numeric_series, numeric_threshold: float, index) -> pd.Series:
        if label_series is not None:
            labels = pd.Series(label_series, index=index).astype(str)
            mask = labels.str.contains(label_positive_token, case=False, na=False)
            return mask.astype(int)
        if numeric_series is not None:
            values = pd.to_numeric(pd.Series(numeric_series, index=index), errors="coerce")
            return (values > numeric_threshold).fillna(False).astype(int)
        return pd.Series(0, index=index, dtype=int)

    def _build_latent_frame(self, features: pd.DataFrame, assets: list) -> pd.DataFrame:
        cols = []
        for asset in assets:
            if asset in features.columns:
                cols.append(asset)
            for suffix in self.ASSET_FEATURE_SUFFIXES:
                name = f"{asset}{suffix}"
                if name in features.columns:
                    cols.append(name)
        for base_col in self.BASE_MACRO_FEATURES:
            if base_col in features.columns:
                cols.append(base_col)

        cols = list(dict.fromkeys(cols))
        if not cols:
            return pd.DataFrame(index=features.index)

        frame = features.loc[:, cols].apply(pd.to_numeric, errors="coerce")
        frame = frame.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        return frame

    def _fit_predict_latent(self, frame: pd.DataFrame, holdout_start: pd.Timestamp):
        if frame.empty:
            return np.full(len(frame.index), -1, dtype=int), np.zeros(len(frame.index), dtype=float)

        train_mask = frame.index < holdout_start
        train = frame.loc[train_mask].dropna()

        if len(train) < 40 or frame.shape[1] == 0:
            return np.full(len(frame.index), -1, dtype=int), np.zeros(len(frame.index), dtype=float)

        full = frame.ffill().bfill().fillna(0.0)
        scaler = RobustScaler()
        scaled_train = scaler.fit_transform(train)
        scaled_full = scaler.transform(full)

        n_components = min(3, max(1, len(train) // 20))
        if n_components < 2:
            return np.zeros(len(frame.index), dtype=int), np.ones(len(frame.index), dtype=float)

        try:
            gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type="full")
            gmm.fit(scaled_train)
            probs = gmm.predict_proba(scaled_full)
            labels = np.argmax(probs, axis=1).astype(int)
            conf = np.max(probs, axis=1).astype(float)
            return labels, conf
        except Exception:
            return np.full(len(frame.index), -1, dtype=int), np.zeros(len(frame.index), dtype=float)

    @staticmethod
    def _apply_stability(labels: np.ndarray, confidence: np.ndarray, dwell_days: int, conf_floor: float) -> np.ndarray:
        labels = np.asarray(labels, dtype=int)
        confidence = np.asarray(confidence, dtype=float)
        if len(labels) == 0:
            return labels
        if dwell_days <= 1:
            out = labels.copy()
            for i in range(1, len(out)):
                if confidence[i] < conf_floor:
                    out[i] = out[i - 1]
            return out

        stable = labels.copy()
        active = stable[0]
        pending = None
        pending_count = 0

        for i in range(1, len(labels)):
            cand = labels[i]
            if confidence[i] < conf_floor:
                stable[i] = active
                pending = None
                pending_count = 0
                continue

            if cand == active:
                stable[i] = active
                pending = None
                pending_count = 0
                continue

            if pending == cand:
                pending_count += 1
            else:
                pending = cand
                pending_count = 1

            if pending_count >= dwell_days:
                active = cand
                pending = None
                pending_count = 0
            stable[i] = active

        return stable
