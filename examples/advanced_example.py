import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
import numpy as np
from datetime import datetime
import shap
from driftguard.core.drift import DriftDetector
from driftguard.core.monitor import ModelMonitor
from driftguard.core.alert_manager import AlertManager
from driftguard.core.config import ModelMonitorConfig, DriftConfig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import logging
from dotenv import load_dotenv
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model.predict_proba, X_train)
    baseline_shap = explainer(X_train)
    
    logger.info("Initial model trained")
    return model, explainer, baseline_shap

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

def plot_drift_summary(features, scenario_name):
    """Create PDF drift summary report"""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drift_report_{scenario_name}_{timestamp}.pdf"
    
    with PdfPages(filename) as pdf:
        # Create title page
        plt.figure(figsize=(8, 11))
        plt.axis('off')
        plt.text(0.5, 0.8, f"DriftGuard Report\n{scenario_name}", 
                 ha='center', va='center', size=20)
        plt.text(0.5, 0.6, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                 ha='center', va='center', size=12)
        pdf.savefig()
        plt.close()
        
        # Create drift scores plot
        plt.figure(figsize=(10, 6))
        features = sorted(features, key=lambda x: x['drift_score'], reverse=True)
        plt.bar([f['feature'] for f in features], 
                [f['drift_score'] for f in features])
        plt.title('Feature Drift Scores')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Drift Score')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Create importance change plot
        plt.figure(figsize=(10, 6))
        plt.bar([f['feature'] for f in features], 
                [f['importance_change'] for f in features])
        plt.title('Feature Importance Changes')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Importance Change')
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        # Create summary table
        plt.figure(figsize=(10, 6))
        plt.axis('off')
        table_data = [[f['feature'], 
                      f"{f['drift_score']:.3f}", 
                      f"{f['importance_change']:.3f}"] 
                     for f in features[:10]]
        plt.table(cellText=table_data,
                 colLabels=['Feature', 'Drift Score', 'Importance Δ'],
                 loc='center')
        plt.title('Top 10 Drifted Features')
        pdf.savefig()
        plt.close()
        
    return filename

def main():
    """Main function to run the advanced monitoring example."""
    logger.info("Starting advanced model monitoring example...")
    
    # Setup
    X_train, X_val, X_test, y_train, y_val, y_test = setup_monitoring_example()
    
    # Initialize alert manager
    alert_manager = AlertManager(threshold=0.85)

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
    initial_model, explainer, baseline_shap = train_initial_model(X_train, y_train)
    val_predictions = initial_model.predict(X_val)
    test_predictions = initial_model.predict(X_test)

    # Initialize model monitor
    monitor_config = ModelMonitorConfig(
        metrics=['accuracy', 'precision', 'recall', 'f1'],
        retrain_threshold=0.15,
        max_retrains=3
    )
    monitor = ModelMonitor(config=monitor_config)
    monitor.initialize(
        reference_predictions=initial_model.predict(X_train),
        reference_labels=y_train
    )
    monitor.attach_model(initial_model)  # Separate model attachment

    # Get baseline performance
    baseline_performance = monitor.track(predictions=val_predictions, labels=y_val)
    logger.info(f"Baseline performance: {baseline_performance}")

    # Initialize drift detector
    drift_config = DriftConfig()
    drift_detector = DriftDetector(config=drift_config)
    drift_detector.attach_model(initial_model)  # Attach model before initialization
    drift_detector.initialize(reference_data=X_train)
    
    # Phase 2: Production Monitoring
    logger.info("\nPhase 2: Production Monitoring")
    
    # Simulate different scenarios
    scenarios = [
        ("Normal", X_test),
        ("Mild Drift", simulate_data_drift(X_test, 0.3)),
        ("Severe Drift", simulate_data_drift(X_test, 0.8))
    ]
    
    for scenario_name, X_scenario in scenarios:
        logger.info(f"\nRunning scenario: {scenario_name}")
        
        # Time performance monitoring
        perf_start = time.time()
        current_performance = monitor.track(predictions=initial_model.predict(X_scenario), labels=y_test)
        perf_time = time.time() - perf_start
        
        # Time drift detection
        drift_start = time.time()
        drift_reports = drift_detector.detect(X_scenario, parallel=True)
        drift_time = time.time() - drift_start
        
        logger.info(f"Performance monitoring time: {perf_time:.2f}s")
        logger.info(f"Parallel drift detection time: {drift_time:.2f}s")
        
        # Get numeric values before comparison
        baseline_acc = baseline_performance['accuracy']['value']
        current_acc = current_performance['accuracy']['value']
        performance_drop = baseline_acc - current_acc
        
        # Detect drift with parallel processing
        # drift_reports = drift_detector.detect(X_scenario, parallel=True)
        
        # Calculate SHAP values for drifted features
        current_shap = explainer(X_scenario)
        
        # Compare with baseline
        important_features = []
        for report in drift_reports:
            if report.score > alert_manager.threshold:
                feature_idx = X_train.columns.get_loc(report.features[0])
                importance = np.abs(current_shap.values[:,feature_idx]).mean() - \
                             np.abs(baseline_shap.values[:,feature_idx]).mean()
                important_features.append({
                    'feature': report.features[0], 
                    'drift_score': report.score,
                    'importance_change': importance
                })
                
                # Enhance alert message with SHAP info
                alert_msg = f"""Drift detected in features {report.features} 
                Method: {report.method}
                Score: {report.score:.3f}
                Importance Change: {importance:.3f}
                """
                alert_manager.check_and_alert(drift_score=report.score, message=alert_msg)
        
        logger.info(f"Important drifted features: {important_features}")
        
        # Process drift reports with correct attributes
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
        
        for report in drift_reports:
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

        max_drift_score = max(report.score for report in drift_reports)
        
        # Alert logic
        alert_sent = alert_manager.check_and_alert(
            drift_score=max_drift_score,
            message=f"Drift detected in scenario {scenario_name} (score: {max_drift_score:.3f})"
        )
        if alert_sent:
            logger.info(f"New alert triggered for {scenario_name}")
        
        # Print current scenario results
        print(f"\n{'='*50}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*50}")
        
        # Performance metrics table
        print("\nPERFORMANCE METRICS:")
        print(f"{'Metric':<15}{'Current':>10}{'Reference':>12}{'Change':>10}{'Status':>10}")
        for metric, values in current_performance.items():
            change = values['value'] - values['reference']
            status = "" if values['degraded'] else ""
            print(f"{metric:<15}{values['value']:>10.3f}{values['reference']:>12.3f}{change:>+10.3f}{status:>10}")
        
        # Drift statistics
        print("\nDRIFT STATISTICS:")
        print(f"- Features analyzed: {len(drift_reports)}")
        print(f"- Drifted features: {len(important_features)}")
        print(f"- Max drift score: {max_drift_score:.3f}")
        print(f"- Avg drift score: {sum(r.score for r in drift_reports)/len(drift_reports):.3f}")
        
        # Generate and show visualization
        if important_features:
            report_file = plot_drift_summary(important_features, scenario_name)
            print(f"\nPDF report saved to: {report_file}")
            
            # Show top features in console
            top_drifted = sorted(important_features, key=lambda x: x['drift_score'], reverse=True)[:5]
            print("\nTOP DRIFTED FEATURES:")
            print(f"{'Feature':<25}{'Drift Score':>15}{'Importance Change':>18}{'Method':>15}")
            for feature in top_drifted:
                print(f"{feature['feature']:<25}{feature['drift_score']:>15.3f}{feature['importance_change']:>18.3f}{'':>15}")
        
        print(f"\n{'='*50}")
        
        # If significant performance drop, trigger model retraining
        if performance_drop > 0.1:
            logger.warning("Significant performance drop detected")
            
            # Check if automatic retraining should trigger
            if monitor.should_retrain(current_performance):
                logger.info("Initiating automated model retraining...")
                improved_model = monitor.retrain_model(
                    X_new=pd.concat([X_train, X_scenario]),
                    y_new=pd.concat([y_train, y_test])
                )
                
                # Update all components with new model
                drift_detector.attach_model(improved_model)
                drift_detector.initialize(reference_data=X_train)
                explainer = shap.Explainer(improved_model.predict_proba, X_train)
                
                logger.info("Retraining complete. Monitoring with updated model.")
    
    # Print final monitoring statistics
    stats = alert_manager.get_alert_statistics()
    print("\nFinal Monitoring Statistics:")
    print(f"Total Alerts: {stats['total_alerts']}")
    print(f"Successful Alerts: {stats['successful_alerts']}")
    print(f"Failed Alerts: {stats['failed_alerts']}")

if __name__ == "__main__":
    main()