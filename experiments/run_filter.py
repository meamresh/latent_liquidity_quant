#!/usr/bin/env python3
"""
Test script for running various filters on a simple state-space model.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ekf():
    """Test Extended Kalman Filter on a simple nonlinear system."""
    try:
        from filters.ekf import ExtendedKalmanFilter
        print("Testing EKF...")

        # Simple nonlinear system: x = [position, velocity], measurement = range
        def motion_model(x, dt=0.1):
            return np.array([x[0] + x[1]*dt, x[1]])  # constant velocity

        def measurement_model(x):
            return np.array([np.sqrt(x[0]**2 + x[1]**2)])  # range

        def jacobian_motion(x, dt=0.1):
            return np.array([[1, dt], [0, 1]])

        def jacobian_measurement(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            if r == 0:
                return np.array([[0, 0]])
            return np.array([[x[0]/r, x[1]/r]])

        # Initialize filter
        ekf = ExtendedKalmanFilter(
            initial_state=np.array([0.0, 1.0]),
            initial_covariance=np.eye(2) * 0.1,
            process_noise=np.eye(2) * 0.01,
            measurement_noise=np.array([[0.1]])
        )

        # Simulate data
        true_states = []
        measurements = []
        x_true = np.array([0.0, 1.0])
        dt = 0.1

        for t in range(100):
            # True motion
            x_true = motion_model(x_true, dt)
            true_states.append(x_true.copy())

            # Measurement with noise
            z = measurement_model(x_true) + np.random.normal(0, 0.1)
            measurements.append(z)

            # Filter prediction and update
            ekf.predict(jacobian_motion(x_true, dt))
            ekf.update(z, jacobian_measurement(x_true))

        print("EKF test completed successfully.")
        return True

    except ImportError as e:
        print(f"Failed to import EKF: {e}")
        return False
    except Exception as e:
        print(f"EKF test failed: {e}")
        return False

def test_particle_filter():
    """Test Bootstrap Particle Filter."""
    try:
        from filters.particle_filter import ParticleFilter
        print("Testing Particle Filter...")

        # Simple linear system for testing
        def motion_model(x):
            return x + np.random.normal(0, 0.1, x.shape)

        def measurement_model(x):
            return x + np.random.normal(0, 0.1, x.shape)

        # Initialize filter
        pf = ParticleFilter(
            num_particles=100,
            initial_particles=np.random.normal(0, 1, (100, 1)),
            motion_model=motion_model,
            measurement_model=measurement_model
        )

        # Run a few steps
        for _ in range(10):
            pf.predict()
            z = np.array([0.0])  # dummy measurement
            pf.update(z)

        print("Particle Filter test completed successfully.")
        return True

    except ImportError as e:
        print(f"Failed to import Particle Filter: {e}")
        return False
    except Exception as e:
        print(f"Particle Filter test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running filter tests...")

    results = []
    results.append(("EKF", test_ekf()))
    results.append(("Particle Filter", test_particle_filter()))

    print("\nTest Results:")
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{name}: {status}")

    passed = sum(1 for _, s in results if s)
    print(f"\nPassed: {passed}/{len(results)} tests")