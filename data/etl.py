import pandas as pd
from data.market_data import get_market_data
from data.macro_data import get_macro_data
from data.sentiment_data import get_sentiment_data
from data.indicators import get_indicators_data
from data.commodities_data import get_commodities_data
from datetime import datetime, timedelta
from config.settings import (
    MARKET_DATA_FILE, MACRO_DATA_FILE, SENTIMENT_DATA_FILE, 
    INDICATORS_DATA_FILE, COMMODITIES_DATA_FILE
)


def _safe_outer_join(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join two frames on index without failing on overlapping column names.
    Existing left-side columns are preserved; duplicate right-side columns are dropped.
    """
    overlap = set(left_df.columns).intersection(set(right_df.columns))
    if overlap:
        right_df = right_df.drop(columns=sorted(overlap), errors="ignore")
    return left_df.join(right_df, how="outer")


def load_and_merge_data(start_date, end_date, save_to_disk=True):
    """
    Fetches Market, Macro, Sentiment, and Commodities data and merges them.
    """
    
    print("Fetching Market Data...")
    market_df = get_market_data(start_date, end_date)
    if save_to_disk: market_df.to_csv(MARKET_DATA_FILE)
    
    print("Fetching Macro Data...")
    macro_df = get_macro_data(start_date, end_date)
    if save_to_disk: macro_df.to_csv(MACRO_DATA_FILE)
    
    print("Fetching Sentiment Data...")
    sentiment_df = get_sentiment_data(start_date, end_date)
    if save_to_disk: sentiment_df.to_csv(SENTIMENT_DATA_FILE)

    print("Fetching Indicators Data...")
    indicators_df = get_indicators_data(start_date, end_date)
    if save_to_disk: indicators_df.to_csv(INDICATORS_DATA_FILE)

    print("Fetching Commodities & Crypto Data...")
    commodities_df = get_commodities_data(start_date, end_date)
    if save_to_disk: commodities_df.to_csv(COMMODITIES_DATA_FILE)
    
    # Merge all dataframes on the index (Date)
    print("Merging Data...")
    
    # First merge daily datasets
    merged_df = _safe_outer_join(market_df, sentiment_df)
    merged_df = _safe_outer_join(merged_df, indicators_df)
    merged_df = _safe_outer_join(merged_df, commodities_df)
    
    # Merge macro data. Since it's monthly/weekly, we join and then forward fill ALL columns.
    merged_df = _safe_outer_join(merged_df, macro_df)
    
    # Forward fill macro data to propagate monthly values to daily positions
    # We do this before dropping market NAs to ensuring we carry forward the last known macro value
    merged_df = merged_df.ffill()
    
    # Drop rows where we don't have Market data (weekends/holidays from macro entries)
    merged_df = merged_df.dropna(subset=['SP500', 'Nasdaq'])
    
    # Fill remaining NaNs if any (e.g. VIX might miss a day)
    merged_df = merged_df.ffill()
    
    print(f"Data Loaded. Shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=365*2)
    df = load_and_merge_data(start, end)
    print(df.tail(20))
    print(df.describe())
