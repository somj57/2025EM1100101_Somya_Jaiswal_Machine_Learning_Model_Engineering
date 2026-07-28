import pandas as pd

def check_data_quality_and_drift(train_path, batch_path):
    print("--- Data Quality & Drift Monitor ---")
    df_train = pd.read_csv(train_path)
    df_batch = pd.read_csv(batch_path)
    
    # 1. Quality Check: Missing values
    null_counts = df_batch.isnull().sum().sum()
    if null_counts > 0:
        print(f"[QUALITY ALERT] Found {null_counts} missing values in the new batch.")
        
    # 2. Drift Check: Shifting distributions
    train_mean = df_train['MonthlyCharges'].mean()
    batch_mean = df_batch['MonthlyCharges'].mean()
    drift_diff = abs(train_mean - batch_mean)
    
    # Alert if the mean monthly charge shifts by more than $15
    threshold = 15.0 
    
    print(f"Training Mean Charge: ${train_mean:.2f}")
    print(f"Batch Mean Charge: ${batch_mean:.2f}")
    
    if drift_diff > threshold:
        print(f"[DRIFT ALERT] MonthlyCharges distribution shifted by ${drift_diff:.2f}")
    else:
        print("[STATUS] Data distribution is stable.")
        
    # 3. Retraining Trigger Logic
    print("\n--- Retraining Evaluation ---")
    days_since_last_train = 8
    
    if days_since_last_train >= 14 or drift_diff > threshold:
        print(">>> ACTION: Triggering automated retraining pipeline. <<<")
    else:
        print(">>> ACTION: No retraining required today. <<<")
        
if __name__ == "__main__":
    check_data_quality_and_drift('data/telco_churn_historical.csv', 'data/telco_churn_daily_batch.csv')