from __future__ import annotations

from config.settings import INDICATORS_DATA_FILE
from data.local_data_loader import load_user_supplied_timeseries


def get_indicators_data(start_date, end_date):
    """
    Load user-supplied indicator data from local CSV.
    """
    return load_user_supplied_timeseries(
        file_path=INDICATORS_DATA_FILE,
        dataset_name="indicators_data",
        start_date=start_date,
        end_date=end_date,
        required_columns=[
            "Momentum",
            "Strength_RSI",
            "Breadth_Vol",
            "Options_VIX",
            "Junk_Bond_Demand",
            "Volatility_Spread",
            "Safe_Haven_Demand",
        ],
    )
