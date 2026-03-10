#!/usr/bin/env python3
"""
Run Phase 3: Integrate Finance Model into Particle Filter.
Loads real data, runs the filter with 500/1000 particles, counts ESS, and plots results.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.finance_state_space import FinanceStateSpaceModel
from filters.diff_particle_filter import BootstrapModel, StandardParticleFilter

def run_experiment(num_particles=1000):
    print(f"\n{'='*60}")
    print(f"RUNNING FINANCE FILTER EXPERIMENT WITH {num_particles} PARTICLES")
    print(f"{'='*60}")

    # 1. Load Data
    print("Loading observation data...")
    try:
        observations = np.load(os.path.join(_project_root, 'data/processed/observations.npy'))
        dates = np.load(os.path.join(_project_root, 'data/processed/dates.npy'))
        print(f"Loaded {len(observations)} days of data.")
    except FileNotFoundError:
        print("Error: Could not find data files. Please ensure data_pipeline has been run.")
        return

    # 2. Initialize Model
    ssm = FinanceStateSpaceModel()

    # Wrap ssm into BootstrapModel for the filter
    def sample_initial(N, y1):
        # Initial guess: L=0, h=-0.5, z=0
        initial_particles = ssm.transition(np.array([0.0, -0.5, 0.0]), num_particles=N)
        # Initial log-likelihood
        log_w1 = ssm.log_likelihood(initial_particles, y1)
        return tf.convert_to_tensor(initial_particles, dtype=tf.float32), tf.convert_to_tensor(log_w1, dtype=tf.float32)

    def transition(t, x_prev, y_t):
        # Propagate
        particles_next = ssm.transition(x_prev.numpy())
        # Compute log-likelihood
        log_likelihood_t = ssm.log_likelihood(particles_next, y_t.numpy())
        return tf.convert_to_tensor(particles_next, dtype=tf.float32), tf.convert_to_tensor(log_likelihood_t, dtype=tf.float32)

    model = BootstrapModel(sample_initial=sample_initial, transition=transition)

    # 3. Initialize Filter
    pf = StandardParticleFilter(model=model, num_particles=num_particles, resample_threshold=0.5)

    # 4. Filter Loop
    # We use a manual loop instead of pf.call to easily track posterior means and ESS
    N = num_particles
    log_N = np.log(N)
    
    posterior_means = []
    ess_history = []
    
    # Step 0: Initial
    y_obs = tf.convert_to_tensor(observations, dtype=tf.float32)
    x_0, log_w_0 = model.sample_initial(N, y_obs[0])
    
    # Compute mean for state
    w_0 = tf.nn.softmax(log_w_0)
    posterior_means.append(tf.reduce_sum(w_0[:, None] * x_0, axis=0).numpy())
    ess_history.append((1.0 / tf.reduce_sum(w_0**2)).numpy())
    
    curr_x = x_0
    curr_log_w = tf.nn.log_softmax(log_w_0)

    print("Running filtering steps...")
    for t in tqdm(range(1, len(observations))):
        # 1. Resample if ESS is low
        w = tf.nn.softmax(curr_log_w)
        ess = 1.0 / tf.reduce_sum(w**2)
        
        if ess < 0.5 * N:
            # Multinomial resample
            idx = tf.random.categorical(curr_log_w[None, :], N)[0]
            curr_x = tf.gather(curr_x, idx)
            curr_log_w = tf.fill([N], -tf.cast(log_N, tf.float32))
        
        # 2. Propagate and weight
        curr_x, log_lik_t = model.transition(t, curr_x, y_obs[t])
        curr_log_w = curr_log_w + tf.squeeze(log_lik_t)
        
        # 3. Store estimate before normalization of weights for next step
        # (Though Standard PF usually normalizes every step)
        w_norm = tf.nn.softmax(curr_log_w)
        posterior_means.append(tf.reduce_sum(w_norm[:, None] * curr_x, axis=0).numpy())
        ess_history.append((1.0 / tf.reduce_sum(w_norm**2)).numpy())
        
        # Normalize weights for next step to avoid numerical issues
        curr_log_w = curr_log_w - tf.reduce_logsumexp(curr_log_w)

    posterior_means = np.array(posterior_means)
    ess_history = np.array(ess_history)

    # 5. Plotting and Sanity Checks
    print("\nGenerating plots...")
    os.makedirs(os.path.join(_project_root, 'results'), exist_ok=True)
    
    plt.figure(figsize=(15, 12))
    
    # L_t (Liquidity)
    plt.subplot(4, 1, 1)
    plt.plot(dates, posterior_means[:, 0], label='Liquidity (L_t)', color='blue')
    plt.title(f'Posterior Means (Particles: {num_particles})')
    plt.ylabel('L_t')
    plt.legend()
    plt.grid(True)
    
    # h_t (Log-volatility)
    plt.subplot(4, 1, 2)
    plt.plot(dates, posterior_means[:, 1], label='Log-Volatility (h_t)', color='red')
    plt.ylabel('h_t')
    plt.legend()
    plt.grid(True)
    
    # z_t (Correlation correlation)
    plt.subplot(4, 1, 3)
    plt.plot(dates, posterior_means[:, 2], label='Correlation State (z_t)', color='green')
    plt.ylabel('z_t')
    plt.legend()
    plt.grid(True)
    
    # ESS
    plt.subplot(4, 1, 4)
    plt.plot(dates, ess_history, label='ESS', color='purple')
    plt.axhline(y=0.5*num_particles, color='black', linestyle='--', label='Threshold')
    plt.ylabel('ESS')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(_project_root, f'results/finance_filter_{num_particles}.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")

    # Sanity Checks
    # 2008 check
    idx_2008 = [i for i, d in enumerate(dates) if str(d).startswith('2008')]
    if idx_2008:
        max_L_2008 = np.max(posterior_means[idx_2008, 0])
        print(f"Sanity Check (2008): Max Liquidity (L) = {max_L_2008:.4f}")
    
    # 2020 check
    idx_2020 = [i for i, d in enumerate(dates) if str(d).startswith('2020')]
    if idx_2020:
        max_h_2020 = np.max(posterior_means[idx_2020, 1])
        print(f"Sanity Check (2020): Max Log-Volatility (h) = {max_h_2020:.4f}")

    return posterior_means, ess_history

if __name__ == "__main__":
    # Ensure matplotlib works without display
    import matplotlib
    matplotlib.use('Agg')
    
    # Run with 1000 particles (user's target)
    run_experiment(500)
