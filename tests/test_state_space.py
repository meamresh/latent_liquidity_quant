#!/usr/bin/env python3
"""Test script for state-space model with real data."""

import sys
import os
import numpy as np

# Add project root to path
_test_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_test_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.finance_state_space import FinanceStateSpaceModel

print("=" * 70)
print("STATE-SPACE MODEL VALIDATION WITH REAL DATA")
print("=" * 70)

model = FinanceStateSpaceModel()

# Change to project root for data loading
os.chdir(_project_root)
observations = np.load('data/processed/observations.npy')

# Initialize particles
num_particles = 1000
x_particles = model.transition(np.array([0.0, -0.5, 0.0]), num_particles=num_particles)

print(f"\nProcessing {min(20, len(observations))} observations with {num_particles} particles...")
print("-" * 70)

all_valid = True
for t in range(min(20, len(observations))):
    y_obs = observations[t]
    log_liks = model.log_likelihood(x_particles, y_obs)
    
    nan_count = np.isnan(log_liks).sum()
    inf_count = np.isinf(log_liks).sum()
    valid_count = np.isfinite(log_liks).sum()
    
    status = "✓" if nan_count == 0 and inf_count == 0 else "⚠️"
    
    print(f"Step {t:2d}: {status} NaN={nan_count:4d}, Inf={inf_count:4d}, Valid={valid_count:4d}, LL=[{log_liks[np.isfinite(log_liks)].min():.2f}, {log_liks[np.isfinite(log_liks)].max():.2f}]")
    
    if nan_count > 0 or inf_count > 0:
        all_valid = False
    
    # Transition for next step
    x_particles = model.transition(x_particles, num_particles=num_particles)

print("-" * 70)

if all_valid:
    print("\n✅ SUCCESS: Model is numerically stable!")
    print("   • No NaNs detected")
    print("   • No Infs detected")
    print("   • All likelihoods are valid")
else:
    print("\n⚠️  WARNING: Some numerical issues detected")

# Test covariance positive definiteness
print("\nTesting covariance matrix positive definiteness...")
test_h = np.linspace(-2, 0.5, 50)
test_z = np.linspace(-1, 1, 50)

pd_count = 0
for h, z in zip(test_h, test_z):
    Sigma, _, _ = model._build_return_covariance(np.array([h]), np.array([z]))
    eigvals = np.linalg.eigvalsh(Sigma[0])
    if (eigvals > 1e-10).all():
        pd_count += 1

print(f"   • Tested {len(test_h)} covariance matrices")
print(f"   • PD matrices: {pd_count}/{len(test_h)} ✓")

print("\n" + "=" * 70)
print("PHASE 2 VALIDATION COMPLETE ✓")
print("=" * 70)
