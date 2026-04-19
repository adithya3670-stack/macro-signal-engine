from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_and_preprocess_core(
    *,
    dl_model,
    engineered_features_file: str,
    master_data_file: str,
) -> pd.DataFrame:
    """
    Load engineered features with robust fallbacks and ensure required target columns exist.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate_paths = [
        engineered_features_file,
        master_data_file,
        dl_model.data_path,
        os.path.join(base_dir, "data", "final_data.csv"),
        "data/final_data.csv",
    ]
    path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("No data source found for deep learning preprocessing.")

    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
    elif "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
    elif len(df.columns) > 0:
        first_col = df.columns[0]
        if pd.api.types.is_object_dtype(df[first_col]) or pd.api.types.is_string_dtype(df[first_col]):
            parsed = pd.to_datetime(df[first_col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[first_col] = parsed
                df.set_index(first_col, inplace=True)

    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index.notna()].sort_index()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().fillna(0)

    # Ensure binary target columns exist (needed by training/optimization paths).
    horizons_map = {"1d": 1, "1w": 5, "1m": 21, "3m": 63}
    for asset in dl_model.assets:
        if asset not in df.columns:
            continue
        for horizon in dl_model.horizons:
            days = horizons_map.get(horizon)
            if days is None:
                continue
            target_col = f"Target_{asset}_{horizon}"
            if target_col not in df.columns:
                future_ret = df[asset].shift(-days) / df[asset] - 1
                df[target_col] = (future_ret > 0).astype(int)

    return df


def create_sequences_core(
    *,
    X_data,
    window_size: int,
    y_data: Optional[np.ndarray] = None,
    target_alignment: str = "next",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create sliding windows.
    target_alignment:
        - 'next' (default): y[i + window_size] for next-step prediction
        - 'last': y[i + window_size - 1] for same-window alignment
    """
    X_seq, y_seq = [], []
    seq_count = len(X_data) - window_size
    if seq_count <= 0:
        return np.array(X_seq), np.array(y_seq) if y_data is not None else None

    if y_data is None:
        for i in range(seq_count):
            X_seq.append(X_data[i : i + window_size])
        return np.array(X_seq), None

    offset = window_size if target_alignment == "next" else (window_size - 1)
    for i in range(seq_count):
        target_idx = i + offset
        if target_idx >= len(y_data):
            break
        X_seq.append(X_data[i : i + window_size])
        y_seq.append(y_data[target_idx])

    return np.array(X_seq), np.array(y_seq)
