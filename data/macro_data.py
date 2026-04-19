from __future__ import annotations

from config.settings import MACRO_DATA_FILE
from data.local_data_loader import load_user_supplied_timeseries

_REQUIRED_MACRO_COLUMNS = [
    "FEDFUNDS",
    "CPIAUCSL",
    "PPIACO",
    "UNRATE",
    "PAYEMS",
    "M2SL",
    "T10Y3M",
    "UMCSENT",
    "WALCL",
    "DGS10",
    "A191RL1Q225SBEA",
]


def get_macro_data(start_date, end_date):
    """
    Load user-supplied macroeconomic data from local CSV.
    """
    return load_user_supplied_timeseries(
        file_path=MACRO_DATA_FILE,
        dataset_name="macro_data",
        start_date=start_date,
        end_date=end_date,
        required_columns=_REQUIRED_MACRO_COLUMNS,
    )
