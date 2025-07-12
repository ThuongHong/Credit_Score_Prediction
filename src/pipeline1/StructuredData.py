import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.placeholders = ["_", "____", "n/a", "N/A", "NA", "?", "#F%$D@*&8", "!@9#%8", "", " "]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in X_copy.columns:
            X_copy[col] = X_copy[col].replace(self.placeholders, np.nan)
        return X_copy

class NumericCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_columns):
        self.numeric_columns = numeric_columns
    
    def fit(self, X, y=None):
        return self
    
    def clean_numeric(self, val):
        if pd.isna(val):
            return np.nan
        val = str(val).lower().strip().replace(",", "")
        digits = re.findall(r"\d+", val)
        if not digits:
            return np.nan
        try:
            return float("".join(digits))
        except:
            return np.nan
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.numeric_columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].apply(self.clean_numeric)
        return X_copy

class CreditAgeParser(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def parse_credit_age(self, val):
        if pd.isna(val):
            return np.nan
        val = str(val).lower()
        match = re.search(r"(\d+)\s+years.*?(\d+)\s+months", val)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return years + months / 12
        return np.nan
    
    def transform(self, X):
        X_copy = X.copy()
        if 'Credit_History_Age' in X_copy.columns:
            X_copy['Credit_History_Age'] = X_copy['Credit_History_Age'].apply(self.parse_credit_age)
        return X_copy

class LoanTypeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()
        self.fitted = False
    
    def fit(self, X, y=None):
        if 'Type_of_Loan' in X.columns:
            # Clean and prepare loan lists
            loan_lists = self._prepare_loan_lists(X['Type_of_Loan'])
            self.mlb.fit(loan_lists)
            self.fitted = True
        return self
    
    def _prepare_loan_lists(self, series):
        cleaned = (
            series.fillna("")
            .str.replace(" and ", ", ", regex=False)
            .str.replace(r"\s*,\s*", ",", regex=True)
            .str.strip()
            .str.rstrip(",")
        )
        return cleaned.apply(lambda x: x.split(",") if x else [])
    
    def transform(self, X):
        X_copy = X.copy()
        if 'Type_of_Loan' in X_copy.columns and self.fitted:
            loan_lists = self._prepare_loan_lists(X_copy['Type_of_Loan'])
            loan_dummies = pd.DataFrame(
                self.mlb.transform(loan_lists), 
                columns=self.mlb.classes_, 
                index=X_copy.index
            )
            X_copy = pd.concat([X_copy.drop(columns=["Type_of_Loan"]), loan_dummies], axis=1)
        return X_copy

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.categorical_columns = categorical_columns
        self.encoders = {}
        
    def fit(self, X, y=None):
        for col in self.categorical_columns:
            if col in X.columns:
                # Get unique values after cleaning
                cleaned_values = X[col].astype(str).str.lower().str.strip().unique()
                self.encoders[col] = cleaned_values
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.categorical_columns:
            if col in X_copy.columns:
                # Clean and encode
                X_copy[col] = X_copy[col].astype(str).str.lower().str.strip()
                dummies = pd.get_dummies(X_copy[col], prefix=col, drop_first=True)
                X_copy = pd.concat([X_copy.drop(columns=[col]), dummies], axis=1)
        return X_copy

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le = LabelEncoder()
        self.fitted = False
        
    def fit(self, X, y=None):
        if 'Credit_Score' in X.columns:
            cleaned_target = X['Credit_Score'].astype(str).str.lower().str.strip()
            self.le.fit(cleaned_target)
            self.fitted = True
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if 'Credit_Score' in X_copy.columns and self.fitted:
            cleaned_target = X_copy['Credit_Score'].astype(str).str.lower().str.strip()
            X_copy['Credit_Score'] = self.le.transform(cleaned_target)
        return X_copy
    