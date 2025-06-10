import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
from driftguard.core.drift import DriftDetector
from driftguard.core.monitor import ModelMonitor
from driftguard.alert_manager import AlertManager
from driftguard.version_manager import VersionManager
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AdvancedModelMonitoring')

def setup_monitoring_example():
    """Set up the monitoring example with the Wine dataset."""
    # Load environment variables
    load_dotenv()
    
    # Load and prepare data
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Split data into training, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_initial_model(X_train, y_train):
    """Train initial model version."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_improved_model(X_train, y_train):
    """Train improved model version with different parameters."""
    model = LogisticRegression(max_iter=2000, C=0.1)
    model.fit(X_train, y_train)
    return model

def simulate_data_drift(X, magnitude=0.5):
    """Simulate data drift by adding noise to features."""
    X_drifted = X.copy()
    noise = np.random.normal(0, magnitude, X_drifted.shape)
    X_drifted = X_drifted + noise
    return X_drifted

def main():
    """Main function to run the advanced monitoring example."""
    logger.info("Starting advanced model monitoring example...")
    
    # Setup
    X_train, X_val, X_test, y_train, y_val, y_test = setup_monitoring_example()
    
    # Initialize monitoring components
    version_manager = VersionManager()
    alert_manager = AlertManager(threshold=0.6)
    
    # Configure alert recipient
    try:
        alert_manager.set_recipient_email(
            "korirg543@gmail.com",
            "Machine Learning Engineer"
        )
    except ValueError as e:
        logger.error(f"Failed to set recipient email: {e}")
        return
    
    # Phase 1: Initial Model Deployment
    logger.info("\nPhase 1: Initial Model Deployment")
    logger.info("Training initial model...")
    
    initial_model = train_initial_model(X_train, y_train)
    initial_version = version_manager.create_new_version(initial_model)
    logger.info(f"Initial model version: {initial_version}")
    
    # Set up monitoring for initial model
    monitor = ModelMonitor(initial_model)
    drift_detector = DriftDetector(reference_data=X_train)
    
    # Baseline performance
    baseline_performance = monitor.track_performance(X_val, y_val)
    logger.info(f"Baseline performance: {baseline_performance}")
    
    # Phase 2: Production Monitoring
    logger.info("\nPhase 2: Production Monitoring")
    
    # Simulate different scenarios
    scenarios = [
        ("Normal", X_test),
        ("Mild Drift", simulate_data_drift(X_test, 0.3)),
        ("Severe Drift", simulate_data_drift(X_test, 0.8))
    ]
    
    for scenario_name, scenario_data in scenarios:
        logger.info(f"\nRunning scenario: {scenario_name}")
        
        # Performance monitoring
        current_performance = monitor.track_performance(scenario_data, y_test)
        performance_drop = baseline_performance['accuracy'] - current_performance['accuracy']
        
        # Drift detection
        drift_report = drift_detector.detect_drift(scenario_data)
        
        # Calculate maximum drift
        max_drift_score = max(
            report["drift_score"] 
            for report in drift_report.values()
        )
        
        # Prepare detailed report
        message = f"""
        Scenario Analysis: {scenario_name}
        
        Performance Metrics:
        - Current Accuracy: {current_performance['accuracy']:.3f}
        - Baseline Accuracy: {baseline_performance['accuracy']:.3f}
        - Performance Drop: {performance_drop:.3f}
        
        Model Version Info:
        - Current Version: {initial_version}
        - Model Type: Logistic Regression
        
        Drift Analysis:
        - Maximum Drift Score: {max_drift_score:.3f}
        - Alert Threshold: {alert_manager.threshold}
        
        Feature-level Drift Details:
        """
        
        for feature, report in drift_report.items():
            message += f"\n- {feature}:"
            message += f"\n  Drift Score: {report['drift_score']:.3f}"
            message += f"\n  P-value: {report['p_value']:.3f}"
        
        # Alert logic
        if max_drift_score > alert_manager.threshold:
            logger.warning(f"High drift detected in scenario: {scenario_name}")
            alert_manager.check_and_alert(
                drift_score=max_drift_score,
                message=message
            )
            
            # Individual feature alerts for severe drift
            for feature, report in drift_report.items():
                if report["drift_score"] > alert_manager.threshold:
                    feature_message = (
                        f"Severe drift detected in feature '{feature}'\n"
                        f"Drift Score: {report['drift_score']:.3f}\n"
                        f"P-value: {report['p_value']:.3f}\n"
                        f"Current Model Version: {initial_version}"
                    )
                    alert_manager.check_and_alert(
                        drift_score=report["drift_score"],
                        message=feature_message
                    )
        
        # Print current scenario results
        print(f"\nScenario: {scenario_name}")
        print(f"Performance: {current_performance}")
        print(f"Max Drift Score: {max_drift_score:.3f}")
        
        # If severe performance drop, trigger model retraining
        if performance_drop > 0.1:
            logger.warning("Significant performance drop detected. Initiating model retraining...")
            
            # Phase 3: Model Update
            logger.info("\nPhase 3: Model Update")
            
            # Train improved model
            improved_model = train_improved_model(X_train, y_train)
            new_version = version_manager.create_new_version(improved_model)
            
            # Evaluate new model
            monitor = ModelMonitor(improved_model)
            new_performance = monitor.track_performance(scenario_data, y_test)
            
            # Compare performances
            performance_improvement = new_performance['accuracy'] - current_performance['accuracy']
            
            update_message = f"""
            Model Update Report:
            - Previous Version: {initial_version}
            - New Version: {new_version}
            - Previous Accuracy: {current_performance['accuracy']:.3f}
            - New Accuracy: {new_performance['accuracy']:.3f}
            - Improvement: {performance_improvement:.3f}
            """
            
            logger.info(update_message)
            
            # Alert about model update
            if performance_improvement > 0:
                alert_manager.send_alert(
                    f"Model updated successfully. Performance improved by {performance_improvement:.3f}"
                )
            else:
                alert_manager.send_alert(
                    "Model update did not improve performance. Further investigation needed."
                )
    
    # Print final monitoring statistics
    stats = alert_manager.get_alert_statistics()
    print("\nFinal Monitoring Statistics:")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"Successful Alerts: {stats['successful_alerts']}")
    print(f"Failed Alerts: {stats['failed_alerts']}")
    
    versions = version_manager.list_versions()
    print("\nModel Version History:")
    for version in versions:
        print(f"- Version: {version}")

if __name__ == "__main__":
    main()