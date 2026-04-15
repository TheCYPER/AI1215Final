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

    FE-v3 redesign (2026-04-15):
    - Dropped duplicates of raw columns (LoanToIncomeRatio exists in CSV).
    - Dropped trivial transforms a tree learns for free (AvailableCreditRatio
      is just `1 - RevolvingUtilizationRate`).
    - Collapsed six per-column has_* indicators into a single `missing_count`
      so the model sees row-level data completeness as one dimension.
    - Added denser financial-posture signals (LiquidityRatio, CreditUsage,
      DelinquencyRate, InquiriesPerAccount).
    """

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
        "CheckingBalance",
        "TotalCreditLimit",
        "NumberOfOpenAccounts",
        "NumberOfCreditCards",
        "CreditHistoryLengthMonths",
        "Age",
        "EmploymentLengthYears",
        "TotalAssets",
        "MortgageOutstandingBalance",
        "AutoLoanOutstandingBalance",
        "StudentLoanOutstandingBalance",
        "NumberOfHardInquiries12Mo",
        "NumberOfHardInquiries24Mo",
        "SecondaryMonthlyIncome",
        "InvestmentPortfolioValue",
        "PropertyValue",
        "VehicleValue",
        "CollateralValue",
        "NumberOfDependents",
        "NumberOfDependentsUnder18",
        "TotalMonthlyIncome",
        "MonthlyGrossIncome",
        "MonthlyPaymentEstimate",
    ]

    def __init__(self):
        self.feature_names_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = self._to_df(X)
        features = {}

        # --- Delinquency ---
        late_30 = df.get("NumberOfLatePayments30Days", 0).fillna(0)
        late_60 = df.get("NumberOfLatePayments60Days", 0).fillna(0)
        late_90 = df.get("NumberOfLatePayments90Days", 0).fillna(0)
        features["TotalLatePayments"] = late_30 + late_60 + late_90
        features["DelinquencySeverity"] = (
            late_30 * 1 + late_60 * 2 + late_90 * 3
            + df.get("NumberOfChargeOffs", 0).fillna(0) * 4
            + df.get("NumberOfCollections", 0).fillna(0) * 4
            + df.get("NumberOfBankruptcies", 0).fillna(0) * 5
        )

        # --- Financial ratios (LoanToIncomeRatio dropped — duplicate of raw CSV column) ---
        features["SavingsToLoanRatio"] = (
            df.get("SavingsBalance", 0).fillna(0)
            / df.get("RequestedLoanAmount", pd.Series(1)).fillna(1).replace(0, 1)
        )

        # --- Stability (AvailableCreditRatio dropped — trivial 1−Utilization, tree learns for free) ---
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
        open_acc_raw = df.get("NumberOfOpenAccounts", 0).fillna(0)
        open_acc_safe = open_acc_raw.replace(0, 1) if hasattr(open_acc_raw, "replace") else pd.Series(1, index=df.index)
        features["AccountDensity"] = open_acc_raw / credit_history * 12

        # --- FE-v3: single row-level missingness signal ---
        # Replaces six per-column has_* indicators: same underlying info,
        # one dimension instead of six, lets the tree split on "how complete
        # is this application" rather than six tiny binary splits.
        features["missing_count"] = df.isna().sum(axis=1).astype(np.int32)

        # --- FE-v3: net worth & derived financial posture ---
        total_assets = df.get("TotalAssets", 0).fillna(0)
        mortgage = df.get("MortgageOutstandingBalance", 0).fillna(0)
        auto = df.get("AutoLoanOutstandingBalance", 0).fillna(0)
        student = df.get("StudentLoanOutstandingBalance", 0).fillna(0)
        total_liab = mortgage + auto + student
        features["NetWorth"] = total_assets - total_liab

        loan_amt = df.get("RequestedLoanAmount", 0).fillna(0)
        features["LoanToAssets"] = loan_amt / total_assets.replace(0, 1)
        features["DebtCoverage"] = total_assets / total_liab.replace(0, 1)

        # --- FE-v3: liquidity (can you cover monthly payments?) ---
        savings = df.get("SavingsBalance", 0).fillna(0)
        checking = df.get("CheckingBalance", 0).fillna(0)
        monthly_pay = (
            df.get("MonthlyPaymentEstimate", pd.Series(1)).fillna(1).replace(0, 1)
        )
        features["LiquidityRatio"] = (savings + checking) / monthly_pay

        # --- FE-v3: actual dollar credit utilization ---
        credit_limit = df.get("TotalCreditLimit", 0).fillna(0)
        util_rate = df.get("RevolvingUtilizationRate", 0).fillna(0)
        features["CreditUsage"] = credit_limit * util_rate

        # --- FE-v3: per-account delinquency and inquiry rates ---
        denom = open_acc_raw.replace(0, 1) if hasattr(open_acc_raw, "replace") else pd.Series(1, index=df.index)
        features["DelinquencyRate"] = features["TotalLatePayments"] / denom
        inq_24 = df.get("NumberOfHardInquiries24Mo", 0).fillna(0)
        features["InquiriesPerAccount"] = inq_24 / denom

        # --- FE-v3: recent inquiries ratio (kept from v2) ---
        inq_12 = df.get("NumberOfHardInquiries12Mo", 0).fillna(0)
        inq_24_safe = inq_24.replace(0, np.nan)
        features["RecentInquiriesRatio"] = (inq_12 / inq_24_safe).fillna(0.0)

        # --- FE-v3: household structure ---
        deps = df.get("NumberOfDependents", 0).fillna(0)
        deps_u18 = df.get("NumberOfDependentsUnder18", 0).fillna(0)
        features["DependentsOver18"] = (deps - deps_u18).clip(lower=0)
        monthly_income = df.get("TotalMonthlyIncome", 0).fillna(0)
        features["IncomePerMember"] = monthly_income / (deps + 1)

        # --- FE-v3: income composition ---
        gross = df.get("MonthlyGrossIncome", 0).fillna(0)
        features["PrimaryIncomeRatio"] = gross / monthly_income.replace(0, 1)

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
