from __future__ import annotations

from config.settings import MARKET_DATA_FILE
from data.local_data_loader import load_user_supplied_timeseries


def get_market_data(start_date, end_date):
    """
    Load user-supplied market data from local CSV.
    """
    return load_user_supplied_timeseries(
        file_path=MARKET_DATA_FILE,
        dataset_name="market_data",
        start_date=start_date,
        end_date=end_date,
        required_columns=["SP500", "Nasdaq", "DJIA", "Russell2000"],
        column_aliases={
            "^GSPC": "SP500",
            "^IXIC": "Nasdaq",
            "^DJI": "DJIA",
            "^RUT": "Russell2000",
        },
    )
