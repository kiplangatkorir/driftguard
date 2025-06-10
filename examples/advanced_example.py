import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
from driftguard.core.drift import DriftDetector
from driftguard.core.monitor import ModelMonitor
from driftguard.core.alert_manager import AlertManager
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import logging
from dotenv import load_dotenv
from driftguard.core.config import DriftConfig

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
    alert_manager = AlertManager(threshold=0.85)  # More conservative threshold
    
    # Configure alert recipient
    try:
        alert_manager.set_recipient_email("korirg543@gmail.com")
        logger.info("Recipient configuration updated: korirg543@gmail.com")
    except ValueError as e:
        logger.error(f"Invalid email configuration: {str(e)}")
        sys.exit(1)
    
    # Phase 1: Initial Model Deployment
    logger.info("\nPhase 1: Initial Model Deployment")
    logger.info("Training initial model...")

    # Train model and get predictions
    initial_model = train_initial_model(X_train, y_train)
    val_predictions = initial_model.predict(X_val)
    test_predictions = initial_model.predict(X_test)

    # Initialize ModelMonitor
    monitor = ModelMonitor()
    monitor.initialize(
        reference_predictions=val_predictions,
        reference_labels=y_val
    )

    # Get baseline performance
    baseline_performance = monitor.track(predictions=val_predictions, labels=y_val)
    logger.info(f"Initial model trained")
    logger.info(f"Baseline performance: {baseline_performance}")

    # Initialize drift detector
    drift_config = DriftConfig()
    drift_detector = DriftDetector(config=drift_config)
    drift_detector.initialize(reference_data=X_train)
    
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
        current_performance = monitor.track(predictions=initial_model.predict(scenario_data), labels=y_test)
        
        # Get numeric values before comparison
        baseline_acc = baseline_performance['accuracy']['value']
        current_acc = current_performance['accuracy']['value']
        performance_drop = baseline_acc - current_acc
        
        # Drift detection
        drift_report = drift_detector.detect(scenario_data)
        
        # Prepare detailed report
        message = f"""
        Scenario Analysis: {scenario_name}
        
        Performance Metrics:
        - Current Accuracy: {current_performance['accuracy']['value']:.3f}
        - Baseline Accuracy: {baseline_performance['accuracy']['value']:.3f}
        - Performance Drop: {performance_drop:.3f}
        
        Model Type: Logistic Regression
        
        Drift Analysis:
        - Alert Threshold: {alert_manager.threshold}
        
        Drift Details:
        """
        
        # Process drift reports with correct attributes
        for report in drift_report:
            # Get report details
            method = report.method
            score = report.score
            features = ", ".join(report.features)  # features is a list
            
            # Build messages
            message += f"\nMethod: {method}\n  Features: {features}\n  Score: {score:.3f}\n"
            
            # Check for alerts
            if score > alert_manager.threshold:
                alert_message = f"Drift detected in features {features} (method: {method}, score: {score:.3f})"
                alert_manager.check_and_alert(drift_score=score, message=alert_message)

        max_drift_score = max(report.score for report in drift_report)
        
        # Alert logic
        alert_sent = alert_manager.check_and_alert(
            drift_score=max_drift_score,
            message=f"Drift detected in scenario {scenario_name} (score: {max_drift_score:.3f})"
        )
        if alert_sent:
            logger.info(f"New alert triggered for {scenario_name}")
        
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
            
            # Evaluate new model
            monitor = ModelMonitor()
            monitor.initialize(
                reference_predictions=improved_model.predict(X_val),
                reference_labels=y_val
            )
            new_performance = monitor.track(predictions=improved_model.predict(scenario_data), labels=y_test)
            
            # Get numeric values before comparison
            new_acc = new_performance['accuracy']['value']
            current_acc = current_performance['accuracy']['value']
            performance_improvement = new_acc - current_acc
            
            update_message = f"""
            Model Update Report:
            - Previous Accuracy: {current_performance['accuracy']['value']:.3f}
            - New Accuracy: {new_performance['accuracy']['value']:.3f}
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

if __name__ == "__main__":
    main()