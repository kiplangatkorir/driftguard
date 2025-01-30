import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

class DriftDetector:
    def __init__(self, reference_data):
        """
        Initializes the DriftDetector with reference data.
        :param reference_data: The reference (training) data used for drift comparison.
        """
        self.reference_data = reference_data
        self.min_samples = 2  # Minimum samples needed for drift detection

    def detect_drift(self, new_data):
        """
        Detects drift between the reference and new data.
        :param new_data: The new data (from production or new batch).
        :return: Drift report (a dictionary containing drift scores and p-values).
        """
        drift_report = {}

        # Handle empty reference data
        if self.reference_data.empty:
            return drift_report

        # Handle empty new data
        if new_data.empty:
            return drift_report
        
        for column in self.reference_data.columns:
            if column not in new_data.columns:
                continue

            # Check if we have enough samples in both datasets
            if len(self.reference_data[column]) < self.min_samples or len(new_data[column]) < self.min_samples:
                continue

            # Check if the feature contains valid numeric data
            if not self._is_valid_numeric_feature(self.reference_data[column]) or \
               not self._is_valid_numeric_feature(new_data[column]):
                continue

            drift_result = self._detect_feature_drift(self.reference_data[column], new_data[column])
            
            # Only include valid results in the report
            if drift_result is not None:
                drift_report[column] = drift_result
        
        return drift_report

    def _detect_feature_drift(self, ref_feature, new_feature):
        """
        Detects drift for a single feature using the Kolmogorov-Smirnov test (for numerical data).
        :param ref_feature: The reference feature (from training data).
        :param new_feature: The new feature (from production data).
        :return: A dictionary containing the p-value and drift score, or None if test fails.
        """
        try:
            # Remove NaN values before performing the test
            ref_feature = ref_feature.dropna()
            new_feature = new_feature.dropna()

            # Check if we still have enough samples after removing NaN values
            if len(ref_feature) < self.min_samples or len(new_feature) < self.min_samples:
                return None

            ks_stat, p_value = ks_2samp(ref_feature, new_feature)
            drift_score = 1 - p_value

            return {
                "p_value": p_value,
                "drift_score": drift_score
            }
        except Exception as e:
            # Log the error if needed
            # print(f"Error in drift detection: {str(e)}")
            return None

    def _is_valid_numeric_feature(self, feature):
        """
        Checks if a feature contains valid numeric data for drift detection.
        :param feature: The feature to check.
        :return: Boolean indicating if the feature is valid for drift detection.
        """
        try:
            # Check if feature is numeric
            if not pd.api.types.is_numeric_dtype(feature):
                return False

            # Check if feature has non-NaN values
            valid_values = feature.dropna()
            if len(valid_values) < self.min_samples:
                return False

            # Check if feature has some variation (not all same value)
            if valid_values.nunique() < 2:
                return False

            return True
        except:
            return False