import numpy as np
from scipy.stats import ks_2samp, chi2_contingency

class DriftDetector:
    def __init__(self, reference_data):
        """
        Initializes the DriftDetector with reference data.
        :param reference_data: The reference (training) data used for drift comparison.
        """
        self.reference_data = reference_data

    def detect_drift(self, new_data):
        """
        Detects drift between the reference and new data.
        :param new_data: The new data (from production or new batch).
        :return: Drift report (a dictionary containing drift scores and p-values).
        """
        drift_report = {}

        for column in new_data.columns:
            drift_report[column] = self._detect_feature_drift(self.reference_data[column], new_data[column])
        
        return drift_report

    def _detect_feature_drift(self, ref_feature, new_feature):
        """
        Detects drift for a single feature using the Kolmogorov-Smirnov test (for numerical data).
        :param ref_feature: The reference feature (from training data).
        :param new_feature: The new feature (from production data).
        :return: A dictionary containing the p-value and drift score.
        """
        ks_stat, p_value = ks_2samp(ref_feature, new_feature)
        
        drift_score = 1 - p_value  
        
        return {"p_value": p_value, "drift_score": drift_score}
