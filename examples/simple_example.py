import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
from driftmonitor.drift_detector import DriftDetector
from driftmonitor.model_monitor import ModelMonitor
from driftmonitor.alert_manager import AlertManager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelMonitoring')

def setup_monitoring_example():
    """Set up the monitoring example with the Iris dataset."""
    # Load environment variables
    load_dotenv()
    
    # Load and prepare data
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create a validation set to simulate production data
    X_val, X_prod, y_val, y_prod = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_prod, y_train, y_val, y_prod

def train_model(X_train, y_train):
    """Train the model and return it."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def simulate_drift(X_prod, drift_magnitude=0.5):
    """Simulate drift in production data."""
    X_drifted = X_prod.copy()
    # Add random noise to simulate drift
    noise = np.random.normal(0, drift_magnitude, X_drifted.shape)
    X_drifted = X_drifted + noise
    return X_drifted

def main():
    """Main function to run the monitoring example."""
    logger.info("Starting model monitoring example...")
    
    # Setup
    X_train, X_val, X_prod, y_train, y_val, y_prod = setup_monitoring_example()
    
    # Train model
    logger.info("Training model...")
    model = train_model(X_train, y_train)
    
    # Initialize monitoring components
    monitor = ModelMonitor(
        model,
        metric_functions={
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro')
        }
    )
    
    drift_detector = DriftDetector(
        reference_data=X_train,
        drift_threshold=0.5
    )
    
    # Set up alerts
    alert_manager = AlertManager(threshold=0.5)
    
    # Configure recipient email (in a real application, this would be done by the user)
    try:
        alert_manager.set_recipient_email(
            "kiplangatgilbert00@gmail.com",
            "Data Scientist"
        )
    except ValueError as e:
        logger.error(f"Failed to set recipient email: {e}")
        return
    
    # Baseline performance
    logger.info("Calculating baseline performance...")
    baseline_performance = monitor.track_performance(X_val, y_val)
    logger.info(f"Baseline performance: {baseline_performance}")
    
    # Simulate production monitoring
    logger.info("Starting production monitoring simulation...")
    
    # Simulate different scenarios
    scenarios = [
        ("Normal", X_prod),
        ("Minor Drift", simulate_drift(X_prod, 0.3)),
        ("Major Drift", simulate_drift(X_prod, 0.8))
    ]
    
    for scenario_name, scenario_data in scenarios:
        logger.info(f"\nRunning scenario: {scenario_name}")
        
        # Performance monitoring
        current_performance = monitor.track_performance(scenario_data, y_prod)
        
        # Drift detection
        drift_report = drift_detector.detect_drift(scenario_data)
        
        # Process drift results and send alerts if needed
        max_drift_score = max(
            report["drift_score"] 
            for report in drift_report.values()
        )
        
        # Prepare detailed message
        message = f"""
        Scenario: {scenario_name}
        
        Performance Metrics:
        - Current Accuracy: {current_performance['accuracy']:.3f}
        - Baseline Accuracy: {baseline_performance['accuracy']:.3f}
        - Performance Drop: {(baseline_performance['accuracy'] - current_performance['accuracy']):.3f}
        
        Drift Analysis:
        - Maximum Drift Score: {max_drift_score:.3f}
        - Threshold: {alert_manager.threshold}
        
        Feature-level Drift:
        """
        
        for feature, report in drift_report.items():
            message += f"\n- {feature}: {report['drift_score']:.3f}"
            
            # Send individual feature alerts if needed
            if report["drift_score"] > alert_manager.threshold:
                alert_manager.check_and_alert(
                    drift_score=report["drift_score"],
                    message=f"High drift detected in feature '{feature}'!"
                )
        
        # Send overall alert if max drift is high
        if max_drift_score > alert_manager.threshold:
            logger.warning(f"High drift detected in scenario: {scenario_name}")
            alert_manager.check_and_alert(
                drift_score=max_drift_score,
                message=message
            )
        else:
            logger.info(f"No significant drift detected in scenario: {scenario_name}")
        
        # Print monitoring statistics
        print(f"\nScenario: {scenario_name}")
        print(f"Performance: {current_performance}")
        print(f"Max Drift Score: {max_drift_score:.3f}")
        
    # Print final alert statistics
    stats = alert_manager.get_alert_statistics()
    print("\nAlert Statistics:")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"Successful Alerts: {stats['successful_alerts']}")
    print(f"Failed Alerts: {stats['failed_alerts']}")

if __name__ == "__main__":
    main()