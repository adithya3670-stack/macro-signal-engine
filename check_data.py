import pandas as pd

df = pd.read_csv('data/engineered_features.csv')

# Check columns
print('Columns (first 10):', df.columns.tolist()[:10])

# Find date column
date_col = None
for col in df.columns:
    if 'date' in col.lower() or 'unnamed' in col.lower():
        date_col = col
        break

if date_col:
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max()
    min_date = df[date_col].min()
    print(f'\nDate range: {min_date} to {max_date}')
    print(f'Total rows: {len(df)}')
    print('\nLast 5 dates:')
    last_dates = df[date_col].drop_duplicates().sort_values().tail(5)
    for date in last_dates:
        print(date.strftime('%Y-%m-%d'))
else:
    print('\nNo date column found')
