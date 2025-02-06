import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DriftDetector')

class DriftDetector:
    def __init__(self, reference_data):
        """
        Initializes the DriftDetector with reference data.
        :param reference_data: The reference (training) data used for drift comparison.
        """
        self.reference_data = reference_data
        self.min_samples = max(10, len(self.reference_data) * 0.01)  
    
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
                    drift_report[column] = {"p_value": 0.0, "drift_score": 1.0}
            return drift_report

        def process_column(column):
            if column not in new_data.columns:
                return None
            
            ref_feature = self.reference_data[column]
            new_feature = new_data[column]
            
            if len(ref_feature) >= self.min_samples and len(new_feature) >= self.min_samples:
                if ref_feature.dtype == 'O' or ref_feature.nunique() < 10:  
                    return column, self._detect_categorical_drift(ref_feature, new_feature)
                else:  
                    return column, self._detect_feature_drift(ref_feature, new_feature)
            return None
        
        results = Parallel(n_jobs=-1)(delayed(process_column)(col) for col in self.reference_data.columns)
        drift_report = {col: result for col, result in results if result is not None}
        return drift_report

    def _detect_feature_drift(self, ref_feature, new_feature):
        """
        Detects drift for a single numerical feature using the Kolmogorov-Smirnov test.
        """
        try:
            ref_feature, new_feature = map(lambda x: np.asarray(x.dropna()).ravel(), [ref_feature, new_feature])
            ks_stat, p_value = ks_2samp(ref_feature, new_feature)
            return {"p_value": float(p_value), "drift_score": 1 - float(p_value)}
        except Exception as e:
            logger.error(f"Drift detection failed for numerical feature: {str(e)}")
            return {"p_value": 0.0, "drift_score": 1.0}

    def _detect_categorical_drift(self, ref_feature, new_feature):
        """
        Uses Chi-Square test for categorical feature drift detection.
        """
        try:
            ref_counts = pd.Series(ref_feature).value_counts(normalize=True)
            new_counts = pd.Series(new_feature).value_counts(normalize=True)
            combined = pd.DataFrame({'ref': ref_counts, 'new': new_counts}).fillna(0)
            chi2_stat, p_value, _, _ = chi2_contingency([combined['ref'], combined['new']])
            return {"p_value": float(p_value), "drift_score": 1 - float(p_value)}
        except Exception as e:
            logger.error(f"Drift detection failed for categorical feature: {str(e)}")
            return {"p_value": 0.0, "drift_score": 1.0}
