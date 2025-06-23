"""Performance test for optimized drift detection"""
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from driftguard.core.drift import DriftDetector
from driftguard.core.config import DriftConfig

# Generate test data
print("Generating test data...")
np.random.seed(42)
ref_data = pd.DataFrame({
    f'feature_{i}': np.random.normal(0, 1, 10000) 
    for i in range(50)
})

test_data = pd.DataFrame({
    f'feature_{i}': np.random.normal(0.1, 1.1, 50000) 
    for i in range(50)
})

# Initialize detector
print("Initializing detector...")
config = DriftConfig(methods=['ks', 'psi', 'jsd'])
detector = DriftDetector(config)
detector.initialize(ref_data)

# Test performance
print("\nRunning optimized detection...")
print(f"Dataset: {len(test_data)} samples x {len(test_data.columns)} features")
print(f"Methods: {config.methods}")
print(f"Batch size: 1000")

start = time.time()
results = detector.detect(test_data, batch_size=1000)
duration = time.time() - start

# Show results
print("\n=== Results ===")
print(f"Total time: {duration:.2f} seconds")
print(f"Features with drift: {len([r for r in results if r.score > r.threshold])}/{len(results)}")
print(f"Top drifted features:")
for r in sorted(results, key=lambda x: x.score, reverse=True)[:5]:
    print(f"- {r.feature}: {r.method}={r.score:.3f} (threshold={r.threshold:.3f})")
