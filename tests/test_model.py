import os
import sys
import pytest
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.finance_state_space import FinanceStateSpaceModel

@pytest.fixture
def model():
    return FinanceStateSpaceModel()

def test_model_initialization(model):
    assert model.dim_state == 3
    assert model.dim_obs == 6
    assert hasattr(model, 'transition')
    assert hasattr(model, 'log_likelihood')

def test_transition_shape(model):
    x_prev = np.array([0.0, -0.5, 0.0])
    num_particles = 100
    x_next = model.transition(x_prev, num_particles=num_particles)
    assert x_next.shape == (num_particles, 3)

def test_transition_stability(model):
    # Test for NaNs over multiple steps
    x = np.array([[0.0, -0.5, 0.0]])
    for _ in range(50):
        x = model.transition(x)
        assert not np.isnan(x).any()
        assert not np.isinf(x).any()

def test_log_likelihood_shape(model):
    num_particles = 50
    x_particles = np.random.normal(0, 1, (num_particles, 3))
    y_obs = np.array([0.01, -0.01, 0.0, 20.0, 4.0, 0.0])
    log_liks = model.log_likelihood(x_particles, y_obs)
    assert log_liks.shape == (num_particles,)

def test_log_likelihood_values(model):
    # Basic check that likelihood is finite for reasonable particles
    x_particles = np.array([[0.0, -0.5, 0.0]])
    y_obs = np.array([0.0, 0.0, 0.0, 15.0, 4.0, 0.0]) # Perfect match for mean
    log_liks = model.log_likelihood(x_particles, y_obs)
    assert np.isfinite(log_liks[0])

def test_covariance_pos_def(model):
    # Test that _ensure_positive_definite actually works
    Sigma = np.array([[[1.0, 0.999, 0.999], [0.999, 1.0, 0.999], [0.999, 0.999, 1.0]]])
    Sigma_pd, adjusted = model._ensure_positive_definite(Sigma)
    eigvals = np.linalg.eigvalsh(Sigma_pd[0])
    assert (eigvals > 0).all()
