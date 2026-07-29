import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from features import HouseFeatureEngineer

def train_and_evaluate():
    df = pd.read_csv("data/raw/training_table.csv")
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Baseline Model Pipeline
    baseline = Pipeline([
        ('features', HouseFeatureEngineer()),
        ('model', DummyRegressor(strategy="mean"))
    ])
    
    # Candidate Model Pipeline
    candidate = Pipeline([
        ('features', HouseFeatureEngineer()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train
    baseline.fit(X_train, y_train)
    candidate.fit(X_train, y_train)
    
    # Evaluate
    base_preds = baseline.predict(X_test)
    cand_preds = candidate.predict(X_test)
    
    base_rmse = root_mean_squared_error(y_test, base_preds)
    cand_rmse = root_mean_squared_error(y_test, cand_preds)
    
    print(f"Baseline RMSE: ${base_rmse:,.2f}")
    print(f"Candidate RMSE: ${cand_rmse:,.2f}")
    
    # Guardrail logic
    os.makedirs("models/v1", exist_ok=True)
    metrics = {"baseline_rmse": base_rmse, "candidate_rmse": cand_rmse}
    
    if cand_rmse < base_rmse:
        print("Candidate model is better. Promoting to production!")
        joblib.dump(candidate, "models/v1/model.joblib")
        with open("models/v1/metrics.json", "w") as f:
            json.dump(metrics, f)
    else:
        print("Candidate failed to beat baseline.")

if __name__ == "__main__":
    train_and_evaluate()