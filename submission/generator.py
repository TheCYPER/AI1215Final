"""
Generate Kaggle submission by combining classification and regression predictions.

Output format: Id, RiskTier (int 0-4), InterestRate (float 4.99-35.99)
"""

import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from configs.config import Config

logger = logging.getLogger("submission")


class SubmissionGenerator:
    """Merge two trained pipelines into a single Kaggle submission."""

    def __init__(self, config: Config):
        self.config = config

    def generate(
        self,
        clf_pipeline_path: Optional[str] = None,
        reg_pipeline_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load both pipelines, predict on test data, output submission CSV.

        Args:
            clf_pipeline_path: Path to classification pipeline joblib.
            reg_pipeline_path: Path to regression pipeline joblib.
            output_path: Where to save the CSV.

        Returns:
            Submission DataFrame.
        """
        if clf_pipeline_path is None:
            clf_pipeline_path = f"{self.config.paths.models_dir}/classification_pipeline.joblib"
        if reg_pipeline_path is None:
            reg_pipeline_path = f"{self.config.paths.models_dir}/regression_pipeline.joblib"
        if output_path is None:
            output_path = f"{self.config.paths.predictions_dir}/submission.csv"

        # Load test data
        test_df = pd.read_csv(self.config.paths.test_csv)
        logger.info(f"Loaded {len(test_df)} test samples")

        # Classification predictions
        clf_pipeline = joblib.load(clf_pipeline_path)
        risk_tier = clf_pipeline.predict(test_df)
        risk_tier = np.clip(risk_tier, 0, 4).astype(int)
        logger.info(f"RiskTier distribution: {np.bincount(risk_tier, minlength=5)}")

        # Regression predictions
        reg_pipeline = joblib.load(reg_pipeline_path)
        interest_rate = reg_pipeline.predict(test_df)
        lo, hi = self.config.training.interest_rate_range
        interest_rate = np.clip(interest_rate, lo, hi).round(2)
        logger.info(
            f"InterestRate range: [{interest_rate.min():.2f}, {interest_rate.max():.2f}]"
        )

        # Build submission
        submission = pd.DataFrame({
            "Id": range(len(test_df)),
            "RiskTier": risk_tier,
            "InterestRate": interest_rate,
        })
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")

        return submission
