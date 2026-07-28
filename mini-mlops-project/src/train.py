import pandas as pd
import os
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from features import engineer_features

def train_and_evaluate():
    # 1. Load Data
    df = pd.read_csv('data/telco_churn_historical.csv')
    
    # Apply shared feature engineering
    df_processed = engineer_features(df)
    
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Baseline
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    base_preds = baseline.predict(X_val)
    base_probs = baseline.predict_proba(X_val)[:, 1]
    
    base_acc = accuracy_score(y_val, base_preds)
    base_auc = roc_auc_score(y_val, base_probs)
    
    # 3. Train Candidate Model
    candidate = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
    candidate.fit(X_train, y_train)
    cand_preds = candidate.predict(X_val)
    cand_probs = candidate.predict_proba(X_val)[:, 1]
    
    cand_acc = accuracy_score(y_val, cand_preds)
    cand_auc = roc_auc_score(y_val, cand_probs)
    
    # 4. Evaluation Harness & Guardrail Rule
    print(f"Baseline AUC: {base_auc:.4f} | Candidate AUC: {cand_auc:.4f}")
    
    promoted_model = baseline
    model_version = "v1.0-baseline"
    
    # Promotion threshold: AUC >= 0.70 and doesn't underperform baseline significantly
    if cand_auc >= 0.70 and cand_auc >= (base_auc - 0.01):
        print("Guardrail passed: Candidate model promoted to registry.")
        promoted_model = candidate
        model_version = "v2.0-candidate"
    else:
        print("Guardrail failed: Candidate rejected. Baseline retained.")
        
    # 5. Save Artifacts
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts/eval', exist_ok=True)
    
    joblib.dump(promoted_model, 'models/promoted_model.pkl')
    
    report = {
        "model_version": model_version,
        "baseline_metrics": {"accuracy": base_acc, "roc_auc": base_auc},
        "candidate_metrics": {"accuracy": cand_acc, "roc_auc": cand_auc},
        "promotion_decision": model_version
    }
    with open('artifacts/eval/report.json', 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == '__main__':
    train_and_evaluate()