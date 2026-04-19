import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from config.settings import (
    BASE_DIR, DATA_DIR, EXTRACTED_DATA_FILE, ENGINEERED_FEATURES_FILE,
    MASTER_DATA_FILE, MARKET_DATA_FILE, MACRO_DATA_FILE, SENTIMENT_DATA_FILE,
    INDICATORS_DATA_FILE, COMMODITIES_DATA_FILE
)
from data.etl import load_and_merge_data
from analysis.correlations import calculate_correlations, get_latest_drivers

# Global Cache
CACHE = {
    'data': None,
    'correlations': None,
    'drivers': None
}

def clean_for_json(obj):
    """Recursively clean NaNs, Infs, and NaTs from dictionary/list for JSON serialization."""
    if obj is None:
        return None
        
    if obj is pd.NaT:
        return None
    
    if not isinstance(obj, (dict, list, str, int, float, bool)):
        if pd.isna(obj):
            return None
            
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0
        return obj
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
        
    return obj

def load_from_cache_or_csv():
    """Attempts to load data from CSV if cache is empty."""
    if CACHE['data'] is not None:
        return True
        
    # Priority: Master -> Extracted
    target_path = None
    if os.path.exists(MASTER_DATA_FILE):
        target_path = MASTER_DATA_FILE
        print(f"Loading Master Data from {target_path}...")
    elif os.path.exists(EXTRACTED_DATA_FILE):
        target_path = EXTRACTED_DATA_FILE
        print(f"Loading Partial Data from {target_path}...")
        
    if target_path:
        try:
            # Load CSV
            df = pd.read_csv(target_path)
            
            # Check for Date column
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.set_index('Date', inplace=True)
            elif 'Unnamed: 0' in df.columns:
                df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df.set_index('Date', inplace=True)
            
            # Drop invalid dates
            if df.index.hasnans:
                print(f"Dropping {df.index.isna().sum()} rows with invalid dates.")
                df = df[df.index.notnull()]
                
            # Sanitize numeric data
            df = df.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Recalculate analysis
            correlations = calculate_correlations(df)
            drivers = get_latest_drivers(df, correlations)
            
            CACHE['data'] = df
            CACHE['correlations'] = correlations
            CACHE['drivers'] = drivers
            print("Loaded data from disk.")
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
    return False

def refresh_data(start_date=None, end_date=None):
    print(f"Refreshing Data... Start: {start_date}, End: {end_date}")
    
    # Defaults
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=365*2)
        
    df = load_and_merge_data(start_date, end_date)
    correlations = calculate_correlations(df)
    drivers = get_latest_drivers(df, correlations)
    
    # Handle NaN/Inf
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Clean correlations
    for term, corr_df in correlations.items():
        correlations[term] = corr_df.replace([np.inf, -np.inf], 0).fillna(0)
        
    CACHE['data'] = df
    CACHE['correlations'] = correlations
    CACHE['drivers'] = drivers
    
    # Save Persistence
    try:
        df.to_csv(EXTRACTED_DATA_FILE)
        print(f"Data saved to {EXTRACTED_DATA_FILE}")
    except Exception as e:
        print(f"Warning: Could not save to CSV: {e}")
        
    print("Data Refreshed.")
