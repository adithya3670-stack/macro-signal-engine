import pandas as pd
import os

master_path = 'master_data/master_dataset.csv'
df = pd.read_csv(master_path, index_col=0, parse_dates=True)

print(f"Original End Date: {df.index.max()}")
print(f"Original Length: {len(df)}")

# Sort index to be sure
df.sort_index(inplace=True)

# Drop last 5 rows
truncated_df = df.iloc[:-5]

print(f"New End Date: {truncated_df.index.max()}")
print(f"New Length: {len(truncated_df)}")

truncated_df.to_csv(master_path)
print("Truncated master dataset saved.")
