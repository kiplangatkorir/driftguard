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
        self.min_samples = 2  

    def detect_drift(self, new_data):
        """
        Detects drift between the reference and new data.
        :param new_data: The new data (from production or new batch).
        :return: Drift report (a dictionary containing drift scores and p-values).
        """
        drift_report = {}

        if new_data.empty:
            return drift_report
        
        if self.reference_data.empty and not new_data.empty:
            for column in new_data.columns:
                if len(new_data[column]) >= self.min_samples:
                    drift_report[column] = {
                        "p_value": 0.0,  
                        "drift_score": 1.0
                    }
            return drift_report

        for column in self.reference_data.columns:
            if column not in new_data.columns:
                continue

            ref_feature = self.reference_data[column]
            new_feature = new_data[column]

            if len(ref_feature) >= self.min_samples and len(new_feature) >= self.min_samples:
                drift_result = self._detect_feature_drift(ref_feature, new_feature)
                if drift_result is not None:
                    drift_report[column] = drift_result
        
        return drift_report

    def _detect_feature_drift(self, ref_feature, new_feature):
        """
        Detects drift for a single feature using the Kolmogorov-Smirnov test.
        :param ref_feature: The reference feature data
        :param new_feature: The new feature data
        :return: A dictionary containing the p-value and drift score
        """
        try:
            if isinstance(ref_feature, (pd.Series, pd.DataFrame)):
                ref_feature = ref_feature.values
            if isinstance(new_feature, (pd.Series, pd.DataFrame)):
                new_feature = new_feature.values
            
            ref_feature = np.asarray(ref_feature).ravel()
            new_feature = np.asarray(new_feature).ravel()

            ref_feature = ref_feature[~np.isnan(ref_feature)]
            new_feature = new_feature[~np.isnan(new_feature)]

            ks_stat, p_value = ks_2samp(ref_feature, new_feature)
            drift_score = 1 - p_value

            return {
                "p_value": float(p_value),  
                "drift_score": float(drift_score)
            }
            
        except Exception as e:
            return {
                "p_value": 0.0,  
                "drift_score": 1.0
            }