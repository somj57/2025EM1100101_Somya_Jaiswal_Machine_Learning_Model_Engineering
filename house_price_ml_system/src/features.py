import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        # 1. House Age
        X_copy['House_Age'] = X_copy['YrSold'] - X_copy['YearBuilt']
        # 2. Remodel Age
        X_copy['Remodel_Age'] = X_copy['YrSold'] - X_copy['YearRemodAdd']
        # 3. Total Square Footage
        X_copy['Total_Square_Footage'] = X_copy['TotalBsmtSF'].fillna(0) + X_copy['1stFlrSF'].fillna(0) + X_copy['2ndFlrSF'].fillna(0)
        # 4. Total Bathrooms
        X_copy['Total_Bathrooms'] = (X_copy['FullBath'].fillna(0) + 
                                     0.5 * X_copy['HalfBath'].fillna(0) + 
                                     X_copy['BsmtFullBath'].fillna(0) + 
                                     0.5 * X_copy['BsmtHalfBath'].fillna(0))
        # 5. Has Pool
        X_copy['Has_Pool'] = (X_copy['PoolArea'] > 0).astype(int)
        
        # Keep only features we want to model to simplify the API
        features = ['House_Age', 'Remodel_Age', 'Total_Square_Footage', 'Total_Bathrooms', 'Has_Pool', 'OverallQual']
        return X_copy[features].fillna(0)