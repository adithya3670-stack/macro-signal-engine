from __future__ import annotations

from config.settings import COMMODITIES_DATA_FILE
from data.local_data_loader import load_user_supplied_timeseries


def get_commodities_data(start_date, end_date):
    """
    Load user-supplied commodities/crypto data from local CSV.
    """
    return load_user_supplied_timeseries(
        file_path=COMMODITIES_DATA_FILE,
        dataset_name="commodities_data",
        start_date=start_date,
        end_date=end_date,
        required_columns=["Gold", "Silver", "Oil", "Copper"],
        column_aliases={
            "GC=F": "Gold",
            "SI=F": "Silver",
            "CL=F": "Oil",
            "HG=F": "Copper",
            "BTC-USD": "Bitcoin",
        },
    )
