import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analysis.feature_engine import FeatureEngineer
from config.settings import ASSETS, MASTER_DATA_FILE, PRICE_1M_FEATURES_FILE


class Price1MFeatureBuilder:
    """Builds and caches a dedicated feature table for 1-month price forecasting."""

    def __init__(
        self,
        master_data_path: str = MASTER_DATA_FILE,
        output_path: str = PRICE_1M_FEATURES_FILE,
        assets: Optional[Iterable[str]] = None,
    ):
        self.master_data_path = master_data_path
        self.output_path = output_path
        self.assets = list(assets or ASSETS)

    def ensure_feature_file(self, force: bool = False) -> str:
        if force or not os.path.exists(self.output_path):
            self.build_and_save()
        return self.output_path

    def build_and_save(self) -> pd.DataFrame:
        df = self.build_features()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path)
        return df

    def build_features(self) -> pd.DataFrame:
        master_df = self.load_master_data()
        engineered = FeatureEngineer(master_df.copy()).generate_features()
        engineered = engineered.loc[:, [c for c in engineered.columns if not c.startswith("Target_")]]
        engineered = self._encode_categoricals(engineered)
        engineered = self._add_short_horizon_price_features(engineered)
        engineered = self._add_targets(engineered)
        engineered = engineered.replace([np.inf, -np.inf], np.nan)
        engineered = engineered.ffill()
        return engineered

    def load_master_data(self) -> pd.DataFrame:
        if not os.path.exists(self.master_data_path):
            raise FileNotFoundError(f"Master dataset not found at {self.master_data_path}")

        df = pd.read_csv(self.master_data_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
        elif "Unnamed: 0" in df.columns:
            df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
        else:
            raise ValueError("Master dataset does not contain a Date column.")

        df = df[df.index.notna()].sort_index()
        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return df

    def build_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = df.copy()
        engineered = self._encode_categoricals(engineered)
        engineered = self._add_short_horizon_price_features(engineered)
        engineered = self._add_targets(engineered)
        return engineered

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        obj_cols = [c for c in df.columns if df[c].dtype == "object"]
        for col in obj_cols:
            df[col] = pd.Categorical(df[col]).codes.astype(float)
        return df

    def _add_short_horizon_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Price1MFeatureBuilder requires a DatetimeIndex.")

        lags = [1, 2, 3, 5, 10, 20]
        vols = [5, 10, 20, 60]
        shaped = [10, 20, 60, 120]
        new_cols = {}

        for asset in self.assets:
            if asset not in df.columns:
                continue

            price = pd.to_numeric(df[asset], errors="coerce")
            returns = price.pct_change()

            for lag in lags:
                new_cols[f"{asset}_lag_{lag}"] = price.shift(lag)
                new_cols[f"{asset}_ret_{lag}"] = price.pct_change(lag) * 100.0

            for window in vols:
                new_cols[f"{asset}_vol_{window}"] = returns.rolling(window).std() * 100.0

            for window in shaped:
                roll_mean = price.rolling(window).mean()
                roll_std = price.rolling(window).std()
                roll_max = price.rolling(window).max()
                new_cols[f"{asset}_drawdown_{window}"] = price / roll_max - 1.0
                new_cols[f"{asset}_trend_{window}"] = price / roll_mean - 1.0
                new_cols[f"{asset}_zscore_{window}"] = (price - roll_mean) / roll_std.replace(0, np.nan)

        weekdays = df.index.dayofweek
        for day in range(5):
            new_cols[f"dow_{day}"] = (weekdays == day).astype(int)
        new_cols["is_month_end"] = df.index.is_month_end.astype(int)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon = 20
        new_cols = {}
        for asset in self.assets:
            if asset not in df.columns:
                continue

            current_price = pd.to_numeric(df[asset], errors="coerce")
            future_price = current_price.shift(-horizon)
            norm_price = future_price / current_price.replace(0, np.nan)
            mape_valid = ((current_price > 1.0) & (future_price > 1.0)).astype(int)

            new_cols[f"FuturePrice_{asset}_1m"] = future_price
            new_cols[f"NormPrice_{asset}_1m"] = norm_price
            new_cols[f"CenteredNormPrice_{asset}_1m"] = norm_price - 1.0
            new_cols[f"MAPEValid_{asset}_1m"] = mape_valid

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df
