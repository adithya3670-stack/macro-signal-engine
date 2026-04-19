import pandas as pd
import os

def repair_master_data():
    path = os.path.join('master_data', 'master_dataset.csv')
    if not os.path.exists(path):
        print("Master dataset not found.")
        return

    print("Loading master dataset...")
    df = pd.read_csv(path)
    
    # Check what column holds the date
    date_col = 'Date'
    if 'Date' not in df.columns:
        if 'Unnamed: 0' in df.columns:
            date_col = 'Unnamed: 0'
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        else:
            print("No Date column found!")
            return

    print("Parsing dates (Enforcing YYYY-MM-DD)...")
    # We suspect the current order implies a YDM parse, so the STRINGS are likely YYYY-MM-DD but were sorted wrong.
    # We just need to parse them strictly as Y-M-D and sort.
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Date'])
    
    # Set Index
    df.set_index('Date', inplace=True)
    
    # Sort
    print("Sorting index...")
    df.sort_index(inplace=True)
    
    # Remove duplicates
    print("Removing duplicates...")
    df = df[~df.index.duplicated(keep='last')]
    
    print(f"Saving fixed dataset ({len(df)} rows) to {path}...")
    df.to_csv(path)
    print("Done.")

if __name__ == "__main__":
    repair_master_data()
