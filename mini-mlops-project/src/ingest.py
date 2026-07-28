import pandas as pd
import datetime
import os

def ingest_batch_data(historical_path, batch_path):
    print(f"[{datetime.datetime.now()}] Starting micro-batch ingestion...")
    df_hist = pd.read_csv(historical_path)
    df_batch = pd.read_csv(batch_path)
    
    # Append new daily data
    df_merged = pd.concat([df_hist, df_batch], ignore_index=True)
    
    # Overwrite the master training table (simulating a data lake update)
    df_merged.to_csv(historical_path, index=False)
    
    print(f"Ingested {len(df_batch)} new rows. Master dataset updated to {len(df_merged)} rows.")
    
    # Clean up the processed batch file
    os.remove(batch_path)

if __name__ == '__main__':
    ingest_batch_data('data/telco_churn_historical.csv', 'data/telco_churn_daily_batch.csv')