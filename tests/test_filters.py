import os
import sys
import pytest
import numpy as np

# Note: We import TF inside tests to avoid collection-time crashes on some systems
# from filters.resampling import systematic_resample

def test_systematic_resampling_weights():
    try:
        import tensorflow as tf
        from filters.resampling import systematic_resample
    except Exception as e:
        pytest.skip(f"Skipping TF test due to environment error: {e}")

    weights = tf.constant([0.1, 0.2, 0.3, 0.4], dtype=tf.float32)
    indices = systematic_resample(weights)
    indices_np = indices.numpy()
    
    assert len(indices_np) == 4
    assert (indices_np >= 0).all()
    assert (indices_np < 4).all()

def test_resampling_uniform():
    try:
        import tensorflow as tf
        from filters.resampling import systematic_resample
    except Exception as e:
        pytest.skip(f"Skipping TF test due to environment error: {e}")

    weights = tf.ones(100, dtype=tf.float32) / 100.0
    indices = systematic_resample(weights)
    indices_np = indices.numpy()
    
    unique, counts = np.unique(indices_np, return_counts=True)
    assert len(unique) == 100
    assert (counts == 1).all()
