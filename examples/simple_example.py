import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
from driftguard.core.monitor import ModelMonitor
from driftguard.alert_manager import AlertManager
from driftguard.core.drift import DriftDetector
from sklearn.ensemble import RandomForestClassifier
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
    
    # Initialize drift detector
    drift_detector = DriftDetector()
    drift_detector.initialize(reference_data=X_train)
    
    # Initialize monitoring components
    monitor = ModelMonitor()
    monitor.initialize(
        reference_predictions=model.predict(X_train),
        reference_labels=y_train
    )
    alert_manager = AlertManager(threshold=0.5)
    
    # Configure recipient email (in a real application, this would be done by the user)
    try:
        alert_manager.set_recipient_email(
            "kiplangatgilbert00@gmail.com",
            "Machine Learning Engineer"
        )
    except ValueError as e:
        logger.error(f"Failed to set recipient email: {e}")
        return
    
    # Baseline performance
    logger.info("Calculating baseline performance...")
    baseline_metrics = monitor.track(
        predictions=model.predict(X_val),
        labels=y_val
    )
    logger.info(f"Baseline performance: {baseline_metrics}")
    
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
        current_metrics = monitor.track(
            predictions=model.predict(scenario_data),
            labels=y_prod
        )
        
        # Detect drift
        drift_reports = drift_detector.detect(scenario_data)
        
        # Process drift results and send alerts if needed
        max_drift_score = max(report.score for report in drift_reports)
        
        # Prepare detailed message
        message = f"""
        Drift Monitoring Alert
        
        Max Drift Score: {max_drift_score:.3f}
        
        Performance Metrics:
        """
        
        for metric, value in current_metrics.items():
            message += f"- {metric}: {value['value']:.3f}"
            if value['degraded']:
                message += " (Degraded)"
            message += "\n"
        
        message += """
        
        Feature-level Drift:
        """
        
        for report in drift_reports:
            message += f"\n- {report.features[0]}:"
            message += f"\n  Method: {report.method}"
            message += f"\n  Score: {report.score:.3f}"
            message += f"\n  Threshold: {report.threshold:.3f}"
            message += f"\n  Has Drift: {'Yes' if report.has_drift else 'No'}"
        
        # Print results to console
        print(f"\nPerformance Metrics:")
        for metric, value in current_metrics.items():
            print(f"- {metric}: {value['value']:.3f}", end="")
            if value['degraded']:
                print(" (Degraded)")
            else:
                print()
        
        print("\nFeature-level drift details:")
        for report in drift_reports:
            print(f"{report.features[0]}:")
            print(f"  - Method: {report.method}")
            print(f"  - Score: {report.score:.3f}")
            print(f"  - Threshold: {report.threshold:.3f}")
            print(f"  - Has Drift: {'Yes' if report.has_drift else 'No'}")
        
        # Send overall alert if max drift is high
        if max_drift_score > alert_manager.threshold:
            logger.warning(f"High drift detected in scenario: {scenario_name}")
            alert_manager.check_and_alert(
                drift_score=max_drift_score,
                message=message
            )
        else:
            logger.info(f"No significant drift detected in scenario: {scenario_name}")
    
    # Print final alert statistics
    stats = alert_manager.get_alert_statistics()
    print("\nAlert Statistics:")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"Successful Alerts: {stats['successful_alerts']}")
    print(f"Failed Alerts: {stats['failed_alerts']}")

if __name__ == "__main__":
    main()