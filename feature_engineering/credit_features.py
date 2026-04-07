"""
Domain-specific feature engineering for credit risk.

Teammates: expand this file with more features.
Each feature group can be toggled on/off via config.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Build domain-specific credit features from raw columns.

    Currently builds a baseline set. Add more feature groups here
    without touching other pipeline components.
    """

    # Columns this transformer needs as input
    REQUIRED_COLS = [
        "NumberOfLatePayments30Days",
        "NumberOfLatePayments60Days",
        "NumberOfLatePayments90Days",
        "NumberOfChargeOffs",
        "NumberOfCollections",
        "NumberOfBankruptcies",
        "DebtToIncomeRatio",
        "RevolvingUtilizationRate",
        "AnnualIncome",
        "RequestedLoanAmount",
        "SavingsBalance",
        "TotalCreditLimit",
        "NumberOfOpenAccounts",
        "CreditHistoryLengthMonths",
        "Age",
        "EmploymentLengthYears",
    ]

    def __init__(self):
        self.feature_names_ = []

    def fit(self, X, y=None):
        # Stateless — no fitting needed
        return self

    def transform(self, X):
        df = self._to_df(X)
        features = {}

        # --- Delinquency features ---
        features["TotalLatePayments"] = (
            df.get("NumberOfLatePayments30Days", 0).fillna(0)
            + df.get("NumberOfLatePayments60Days", 0).fillna(0)
            + df.get("NumberOfLatePayments90Days", 0).fillna(0)
        )
        features["DelinquencySeverity"] = (
            df.get("NumberOfLatePayments30Days", 0).fillna(0) * 1
            + df.get("NumberOfLatePayments60Days", 0).fillna(0) * 2
            + df.get("NumberOfLatePayments90Days", 0).fillna(0) * 3
            + df.get("NumberOfChargeOffs", 0).fillna(0) * 4
            + df.get("NumberOfCollections", 0).fillna(0) * 4
            + df.get("NumberOfBankruptcies", 0).fillna(0) * 5
        )

        # --- Financial ratios ---
        income = df.get("AnnualIncome", pd.Series(1)).fillna(1).replace(0, 1)
        features["LoanToIncomeRatio"] = (
            df.get("RequestedLoanAmount", 0).fillna(0) / income
        )
        features["SavingsToLoanRatio"] = (
            df.get("SavingsBalance", 0).fillna(0)
            / df.get("RequestedLoanAmount", pd.Series(1)).fillna(1).replace(0, 1)
        )

        # --- Credit utilization ---
        credit_limit = (
            df.get("TotalCreditLimit", pd.Series(1)).fillna(1).replace(0, 1)
        )
        features["AvailableCreditRatio"] = (
            1 - df.get("RevolvingUtilizationRate", 0).fillna(0)
        ).clip(lower=0)

        # --- Stability ---
        age = df.get("Age", pd.Series(1)).fillna(1).replace(0, 1)
        features["EmploymentStability"] = (
            df.get("EmploymentLengthYears", 0).fillna(0) / age
        )
        features["CreditHistoryPerAge"] = (
            df.get("CreditHistoryLengthMonths", 0).fillna(0) / (age * 12)
        )

        # --- Account density ---
        credit_history = (
            df.get("CreditHistoryLengthMonths", pd.Series(1)).fillna(1).replace(0, 1)
        )
        features["AccountDensity"] = (
            df.get("NumberOfOpenAccounts", 0).fillna(0) / credit_history * 12
        )

        self.feature_names_ = list(features.keys())
        result = pd.DataFrame(features, index=df.index)
        return result.values.astype(np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)

    @staticmethod
    def _to_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
