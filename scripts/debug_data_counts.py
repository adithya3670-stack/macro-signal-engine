
import pandas as pd
import numpy as np

def check_data():
    df = pd.read_csv('data/engineered_features.csv')
    
    # Handle Date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
    elif 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.set_index('Date', inplace=True)
        
    df = df.replace([np.inf, -np.inf], np.nan)
    
    exclude_cols = [c for c in df.columns if 'Target_' in c or 'Regime_' in c or 'Date' in c]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Total Features: {len(feature_cols)}")
    print(f"Features: {feature_cols[:10]}...")
    
    assets = ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Bitcoin', 'Oil']
    horizons = ['1w']
    
    for asset in assets:
        target_col = f'Target_{asset}_1w'
        if target_col not in df.columns:
            print(f"MISSING TARGET: {target_col}")
            continue
            
        subset = df[feature_cols + [target_col]]
        valid = subset.dropna()
        print(f"{asset}: Total Rows={len(df)}, Target Non-NaN={df[target_col].count()}, Valid Rows (with features)={len(valid)}")
        
        if len(valid) == 0:
            # Find culprit features
            print(f"  --- Debugging {asset} NaNs ---")
            for col in feature_cols:
                if df[col].isnull().all():
                    print(f"  Feature {col} is ALL NaN")
                elif df[col].count() < 100:
                    print(f"  Feature {col} has only {df[col].count()} rows")

if __name__ == "__main__":
    check_data()
