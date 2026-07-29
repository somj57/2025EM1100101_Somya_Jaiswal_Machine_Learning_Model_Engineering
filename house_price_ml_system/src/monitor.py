import pandas as pd
import numpy as np

def check_data_drift(train_path: str, new_batch_path: str):
    """Simple drift check comparing the mean of TotalBsmtSF"""
    df_train = pd.read_csv(train_path)
    df_new = pd.read_csv(new_batch_path)
    
    train_mean_sf = df_train['TotalBsmtSF'].mean()
    new_mean_sf = df_new['TotalBsmtSF'].mean()
    
    drift_ratio = abs(train_mean_sf - new_mean_sf) / train_mean_sf
    
    print(f"Training Mean BsmtSF: {train_mean_sf:.2f} | New Batch Mean: {new_mean_sf:.2f}")
    if drift_ratio > 0.15: # 15% threshold
        print("WARNING: Significant Data Drift Detected in TotalBsmtSF! Alerting MLOps Team.")
        return True
    return False

def evaluate_retraining_trigger(drift_detected: bool, days_since_last_train: int, recent_rmse: float, rmse_threshold: float):
    """Pseudo-code logic for triggering a retrain"""
    if days_since_last_train >= 30:
        return "TRIGGER: Scheduled monthly retraining."
    elif drift_detected:
        return "TRIGGER: Data drift exceeded threshold."
    elif recent_rmse > rmse_threshold:
        return "TRIGGER: Model performance degraded (RMSE too high)."
    
    return "SKIP: No retraining needed."