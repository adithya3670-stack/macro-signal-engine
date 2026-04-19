from __future__ import annotations

from config.settings import SENTIMENT_DATA_FILE
from data.local_data_loader import load_user_supplied_timeseries


def get_sentiment_data(start_date, end_date):
    """
    Load user-supplied sentiment data from local CSV.
    """
    return load_user_supplied_timeseries(
        file_path=SENTIMENT_DATA_FILE,
        dataset_name="sentiment_data",
        start_date=start_date,
        end_date=end_date,
        required_columns=["VIX"],
        column_aliases={"^VIX": "VIX"},
    )
