import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shared preprocessing function for offline training and online serving.
    Executing this function universally guarantees no training-serving skew.
    """
    df = df.copy()
    
    # 1. Fill missing values
    default_charge = 50.0
    if not df['MonthlyCharges'].isna().all():
        default_charge = df['MonthlyCharges'].median()
    df['MonthlyCharges'] = df['MonthlyCharges'].fillna(default_charge)
    
    # 2. Ratio feature: Value over time
    df['charges_to_tenure_ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)
    
    # 3. Aggregation: Total connected services
    df['total_services'] = df.apply(lambda row: (1 if row.get('InternetService') == 'Yes' else 0) + 
                                                (1 if row.get('PhoneService') == 'Yes' else 0), axis=1)
    
    # 4. Interaction Boolean: High-value senior customers
    df['senior_high_spender'] = ((df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > 80)).astype(int)
    
    # 5. Ordinal Encoding
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['contract_encoded'] = df['Contract'].map(contract_map).fillna(0)
    
    # Remove original component columns as the new derived features encapsulate their utility entirely
    df = df.drop(columns=['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PhoneService', 'customerID'], errors='ignore')
    
    # Ensure standard column order
    expected_cols = ['SeniorCitizen', 'charges_to_tenure_ratio', 'total_services', 'senior_high_spender', 'contract_encoded']
    
    if 'Churn' in df.columns:
        expected_cols.append('Churn')
        
    return df[expected_cols]