
import pandas as pd
import os
import glob

# Config
DATA_FILE = 'data/engineered_features.csv'
MODELS_DIR = 'models_dl'
ASSETS = {
    'SP500': 'SP500',
    'Nasdaq': 'Nasdaq',
    'DJIA': 'DJIA',
    'Russell': 'Russell2000',
    'Gold': 'Gold',
    'Silver': 'Silver',
    'Copper': 'Copper',
    'Oil': 'Oil',
    'Bitcoin': 'Bitcoin'
}

print("="*60)
print(" ASSET READINESS AUDIT")
print("="*60)

# 1. Check Data Availability
print(f"\n[1] Checking Data ({DATA_FILE})...")
if not os.path.exists(DATA_FILE):
    print("CRITICAL: Data file not found!")
    exit()

df = pd.read_csv(DATA_FILE, nrows=5)
cols = set(df.columns)

data_status = {}
for ui_name, col_name in ASSETS.items():
    has_price = col_name in cols
    # Check for at least one target (e.g., 1m)
    has_target = f'Target_{col_name}_1m' in cols
    data_status[ui_name] = (has_price, has_target)

print(f"{'Asset':<10} | {'Price Data':<12} | {'Targets (Trainable)':<20}")
print("-" * 46)
for asset, (price, target) in data_status.items():
    p_mark = "OK" if price else "MISSING"
    t_mark = "OK" if target else "MISSING"
    print(f"{asset:<10} | {p_mark:<12} | {t_mark:<20}")

# 2. Check Model Availability
print(f"\n[2] Checking Trained Models ({MODELS_DIR})...")
if not os.path.exists(MODELS_DIR):
    print("CRITICAL: Models directory not found!")
    exit()

model_files = os.listdir(MODELS_DIR)
model_status = {}

for ui_name, col_name in ASSETS.items():
    # Check for any .pt file starting with the asset name
    # We typically look for {Asset}_1m_lstm.pt etc.
    patterns = [f"{col_name}_1m_lstm.pt", f"{col_name}_1m_transformer.pt"]
    
    found = []
    for p in patterns:
        if p in model_files:
            found.append(p)
            
    model_status[ui_name] = found

print(f"{'Asset':<10} | {'Trained Deep Learning Models'}")
print("-" * 60)
for asset, found in model_status.items():
    status = "READY" if len(found) > 0 else "NEEDS TRAINING"
    details = f"({len(found)} models found)" if found else "(0 found)"
    print(f"{asset:<10} | {status:<15} {details}")

print("\n" + "="*60)
