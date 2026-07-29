import pandas as pd
import datetime
import os

def ingest_data(url: str, output_path: str):
    print(f"[{datetime.datetime.now()}] Ingesting data from {url}...")
    df = pd.read_csv(url)
    
    # Simulate appending to a training table
    mode = 'a' if os.path.exists(output_path) else 'w'
    header = not os.path.exists(output_path)
    df.to_csv(output_path, mode=mode, header=header, index=False)
    
    print(f"[{datetime.datetime.now()}] Ingested {len(df)} rows. Saved to {output_path}")

if __name__ == "__main__":
    train_url = "https://raw.githubusercontent.com/somj57/2025EM1100101_Somya_Jaiswal_Machine_Learning_Model_Engineering/refs/heads/master/house_pricing_data/train.csv"
    os.makedirs("data/raw", exist_ok=True)
    ingest_data(train_url, "data/raw/training_table.csv")