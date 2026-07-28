import pandas as pd
import numpy as np
import os

def generate_sample_data(filename, n_rows=1000):
    np.random.seed(42)
    os.makedirs('data', exist_ok=True)
    
    data = {
        'customerID': [f'CUST_{i}_{np.random.randint(100,999)}' for i in range(n_rows)],
        'tenure': np.random.randint(1, 72, n_rows),
        'MonthlyCharges': np.random.uniform(20.0, 120.0, n_rows),
        'SeniorCitizen': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        'InternetService': np.random.choice(['Yes', 'No'], n_rows),
        'PhoneService': np.random.choice(['Yes', 'No'], n_rows),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_rows),
        'Churn': np.random.choice([0, 1], n_rows, p=[0.73, 0.27])
    }
    df = pd.DataFrame(data)
    
    # Induce a missing value in the batch to trigger the data quality monitor
    if 'batch' in filename:
        df.loc[0, 'MonthlyCharges'] = np.nan
        
    df.to_csv(filename, index=False)
    print(f"Generated {n_rows} rows in {filename}")

if __name__ == '__main__':
    generate_sample_data('data/telco_churn_historical.csv', 1000)
    generate_sample_data('data/telco_churn_daily_batch.csv', 100)