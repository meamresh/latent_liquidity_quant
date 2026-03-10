#!/usr/bin/env python3
"""
Phase 4: Crisis Predictor via Forward Simulation.
Estimates real-time systemic risk by simulating forward paths from the filtered state.
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
from filters.diff_particle_filter import BootstrapModel

def run_crisis_prediction(num_particles=1000, forecast_horizon=60, sim_stride=5):
    print(f"\n{'='*60}")
    print(f"RUNNING CRISIS PREDICTION EXPERIMENT")
    print(f"Particles: {num_particles}, Horizon: {forecast_horizon} days")
    print(f"{'='*60}")

    # 1. Load Data
    try:
        observations = np.load(os.path.join(_project_root, 'data/processed/observations.npy'))
        dates = np.load(os.path.join(_project_root, 'data/processed/dates.npy'))
        print(f"Loaded {len(observations)} days of data.")
    except FileNotFoundError:
        print("Error: Could not find data files.")
        return

    # 2. Initialize Model
    ssm = FinanceStateSpaceModel()

    # Define filtering model
    def sample_initial(N, y1):
        initial_particles = ssm.transition(np.array([0.0, -0.5, 0.0]), num_particles=N)
        log_w1 = ssm.log_likelihood(initial_particles, y1)
        return tf.convert_to_tensor(initial_particles, dtype=tf.float32), tf.convert_to_tensor(log_w1, dtype=tf.float32)

    def transition(t, x_prev, y_t):
        particles_next = ssm.transition(x_prev.numpy())
        log_likelihood_t = ssm.log_likelihood(particles_next, y_t.numpy())
        return tf.convert_to_tensor(particles_next, dtype=tf.float32), tf.convert_to_tensor(log_likelihood_t, dtype=tf.float32)

    model = BootstrapModel(sample_initial=sample_initial, transition=transition)

    # 3. Filter and Predict Loop
    results = {
        'dates': dates,
        'L_mean': [], 'L_q05': [], 'L_q95': [],
        'h_mean': [], 'h_q05': [], 'h_q95': [],
        'z_mean': [], 'z_q05': [], 'z_q95': [],
        'P_L_gt_2': [], # Tail probability L > 2.0
        'crisis_prob': [] # Prob(Crisis within 60 days)
    }

    # Internal state
    N = num_particles
    y_obs = tf.convert_to_tensor(observations, dtype=tf.float32)
    x_curr, log_w_curr = model.sample_initial(N, y_obs[0])
    
    print("Processing filtering and forward simulation...")
    for t in tqdm(range(len(observations))):
        # --- Update Step (Smoothing weights for stats) ---
        if t > 0:
            # Predict step
            x_curr, log_lik_t = model.transition(t, x_curr, y_obs[t])
            log_w_curr = log_w_curr + tf.squeeze(log_lik_t)
            
            # Resample if ESS < 0.5N
            w = tf.nn.softmax(log_w_curr)
            ess = 1.0 / tf.reduce_sum(w**2)
            if ess < 0.5 * N:
                idx = tf.random.categorical(log_w_curr[None, :], N)[0]
                x_curr = tf.gather(x_curr, idx)
                log_w_curr = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))
        
        # --- Collect Filtered Distribution Stats ---
        # Note: If we just resampled, weights are uniform. 
        # If not, we use current weights.
        w_norm = tf.nn.softmax(log_w_curr).numpy()
        x_parts = x_curr.numpy()
        
        # W-Mean
        l_mean = np.sum(w_norm * x_parts[:, 0])
        h_mean = np.sum(w_norm * x_parts[:, 1])
        z_mean = np.sum(w_norm * x_parts[:, 2])
        
        # Simple Quantiles (using mean estimate as weights are often near-uniform after resample)
        # For precision, weighted quantiles would be better, but np.percentile on particles is good proxy
        results['L_mean'].append(l_mean)
        results['L_q05'].append(np.percentile(x_parts[:, 0], 5))
        results['L_q95'].append(np.percentile(x_parts[:, 0], 95))
        
        results['h_mean'].append(h_mean)
        results['h_q05'].append(np.percentile(x_parts[:, 1], 5))
        results['h_q95'].append(np.percentile(x_parts[:, 1], 95))
        
        results['z_mean'].append(z_mean)
        results['z_q05'].append(np.percentile(x_parts[:, 2], 5))
        results['z_q95'].append(np.percentile(x_parts[:, 2], 95))
        
        # Systemic signals
        results['P_L_gt_2'].append(np.sum(w_norm * (x_parts[:, 0] > 2.0)))

        # --- Forward simulation (every sim_stride days or near crisis) ---
        # To avoid extreme slowdown, we simulate every stride, but filtering is daily.
        if t % sim_stride == 0 or results['P_L_gt_2'][-1] > 0.1:
            # FIX: Resample according to weights before simulation
            # This ensures we start from the correct probability distribution
            if np.allclose(w_norm, w_norm[0]):
                sim_parts = x_parts.copy()
            else:
                idx = np.random.choice(N, size=N, p=w_norm)
                sim_parts = x_parts[idx]
                
            has_crisis = np.zeros(N, dtype=bool)
            
            # Stress Score Parameters (Weights for L, sigma, rho)
            # Risk = 1.0*L + 2.0*sigma + 1.0*rho
            # Threshold = 5.0 (Experimental)
            a, b, c = 1.0, 2.0, 1.0
            threshold = 5.0
            
            for _ in range(forecast_horizon):
                sim_parts = ssm.transition(sim_parts) # stochastic
                L_sim = sim_parts[:, 0]
                sigma_sim = np.exp(sim_parts[:, 1] / 2.0)
                rho_sim = np.tanh(sim_parts[:, 2])
                
                # Refined Crisis Logic: Continuous Stress Score
                stress_score = a*L_sim + b*sigma_sim + c*rho_sim
                crisis_mask = stress_score > threshold
                
                has_crisis |= crisis_mask
                if np.all(has_crisis):
                    break
            
            results['crisis_prob'].append(np.mean(has_crisis))
        else:
            # Interpolate or fill with previous if not simulating
            results['crisis_prob'].append(results['crisis_prob'][-1] if len(results['crisis_prob']) > 0 else 0.0)

    # 4. Convert to arrays
    for key in results:
        if key != 'dates':
            results[key] = np.array(results[key])

    # 5. Visualization
    print("\nSaving prediction results...")
    os.makedirs(os.path.join(_project_root, 'results'), exist_ok=True)
    
    plt.figure(figsize=(15, 14))
    
    # Financial Stress (L_t)
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(dates, results['L_mean'], label='Liquidity Stress (L_t)', color='blue')
    ax1.fill_between(dates, results['L_q05'], results['L_q95'], alpha=0.2, color='blue', label='90% CI')
    ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Stress Threshold')
    ax1.set_ylabel('L_t')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Inferred Market States with Crisis Risk Forecasting')

    # Latent Volatility (h_t)
    ax2 = plt.subplot(4, 1, 2)
    ax2.plot(dates, results['h_mean'], label='Log-Volatility (h_t)', color='orange')
    ax2.fill_between(dates, results['h_q05'], results['h_q95'], alpha=0.2, color='orange')
    ax2.set_ylabel('h_t')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Systemic Risk Signal: Probability(L > 2)
    ax3 = plt.subplot(4, 1, 3)
    ax3.fill_between(dates, results['P_L_gt_2'], color='red', alpha=0.5, label='Tail Risk P(L > 2.0)')
    ax3.set_ylabel('Tail Probability')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Crisis Forecast: Prob(Crisis in 60d)
    ax4 = plt.subplot(4, 1, 4)
    ax4.plot(dates, results['crisis_prob'], color='black', linewidth=2, label='Crisis Probability (60-day Horizon)')
    ax4.fill_between(dates, results['crisis_prob'], color='black', alpha=0.1)
    ax4.set_ylabel('Probability')
    ax4.set_xlabel('Date')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(_project_root, 'results/crisis_forecast.png'))
    print("Plot saved to results/crisis_forecast.png")

    # --- SAVE DATA FOR PHASE 5 ---
    data_path = os.path.join(_project_root, 'results/crisis_res.npz')
    # Save results as compressed npz
    # We filter out 'dates' from the loop or just save it explicitly
    save_dict = {key: results[key] for key in results if key != 'dates'}
    save_dict['dates'] = dates
    np.savez_compressed(data_path, **save_dict)
    print(f"Data saved to {data_path}")

    # Predictive Analysis - Check if risk rises BEFORE events
    # 2008 check
    idx_2007 = [i for i, d in enumerate(dates) if str(d).startswith('2007')]
    if idx_2007:
        avg_risk_2007 = np.mean(results['crisis_prob'][idx_2007])
        print(f"Analysis: Average Crisis Risk in 2007 (Pre-Crisis): {avg_risk_2007:.2%}")

    return results

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    run_crisis_prediction(num_particles=1000)
