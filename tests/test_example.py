"""
Example script demonstrating DriftGuard functionality.
"""
import asyncio
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from driftguard.core.guardian import DriftGuard
from driftguard.core.config import ConfigManager

async def main():
    # Generate sample data
    print("Generating sample data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Split into reference and monitoring data
    X_ref, X_monitor, y_ref, y_monitor = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    # Create a sample model
    print("Training sample model...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_ref, y_ref)
    
    # Convert to pandas DataFrames
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    reference_df = pd.DataFrame(X_ref, columns=feature_names)
    monitoring_df = pd.DataFrame(X_monitor, columns=feature_names)
    
    # Initialize DriftGuard
    print("Initializing DriftGuard...")
    monitor = DriftGuard(
        model=model,
        reference_data=reference_df,
        model_type="classification",
        project_name="example_project"
    )
    
    # Monitor first batch (no drift)
    print("\nMonitoring first batch (no drift)...")
    first_batch = monitoring_df.iloc[:200]
    first_labels = y_monitor[:200]
    
    results = await monitor.monitor_batch(
        first_batch,
        first_labels,
        metadata={"batch_id": "batch_1"}
    )
    
    print("First batch results:")
    print(f"Drift detected: {results['drift_detected']}")
    print("Performance metrics:", results['performance_metrics'])
    
    # Introduce artificial drift
    print("\nMonitoring second batch (with artificial drift)...")
    second_batch = monitoring_df.iloc[200:400].copy()
    second_labels = y_monitor[200:400]
    
    # Add drift by shifting feature values
    second_batch['feature_0'] += 2.0
    second_batch['feature_1'] *= 1.5
    
    results = await monitor.monitor_batch(
        second_batch,
        second_labels,
        metadata={"batch_id": "batch_2"}
    )
    
    print("Second batch results:")
    print(f"Drift detected: {results['drift_detected']}")
    if results['drift_detected']:
        print("\nDrift reports:")
        for report in results['drift_reports']:
            if report['drift_score'] > monitor.config_manager.config.drift.threshold:
                print(f"- {report['feature_name']}: score={report['drift_score']:.3f}")
    
    # Get monitoring summary
    print("\nGetting monitoring summary...")
    summary = monitor.get_monitoring_summary()
    print("Project status:", summary['status'])
    print("Performance history:", summary.get('performance', {}))

if __name__ == "__main__":
    asyncio.run(main())
