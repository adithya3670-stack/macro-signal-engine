from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_user_supplied_timeseries(
    *,
    file_path: str,
    dataset_name: str,
    start_date=None,
    end_date=None,
    required_columns: Optional[Iterable[str]] = None,
    column_aliases: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load a user-managed CSV timeseries file and normalize it to a Date index.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found at '{path}'. "
            "Provide the dataset locally before running this pipeline."
        )

    df = pd.read_csv(path)
    if df.empty:
        columns = list(required_columns) if required_columns else []
        return pd.DataFrame(columns=columns)

    if column_aliases:
        df = df.rename(columns=column_aliases)

    date_col = "Date"
    if date_col not in df.columns:
        if "Unnamed: 0" in df.columns:
            date_col = "Unnamed: 0"
        else:
            date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")]

    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]

    if required_columns:
        missing = [column for column in required_columns if column not in df.columns]
        if missing:
            raise ValueError(
                f"{dataset_name} is missing required columns: {', '.join(missing)}"
            )
        df = df.loc[:, list(required_columns)]

    return df
