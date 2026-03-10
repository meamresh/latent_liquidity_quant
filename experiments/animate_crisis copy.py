#!/usr/bin/env python3
"""
Phase 5: Dynamic Crisis Animation.
Visualizes the filtering process and forward simulations spawning day-to-day.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.dates as mdates

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.finance_state_space import FinanceStateSpaceModel

def create_animation(num_frames=250, start_date='2007-06-01', num_fan_paths=15):
    print("Loading data and initializing model for predictive animation...")
    res_data = np.load(os.path.join(_project_root, 'results/crisis_res.npz'), allow_pickle=True)
    obs_data = np.load(os.path.join(_project_root, 'data/processed/observations.npy'))
    dates = res_data['dates']
    
    ssm = FinanceStateSpaceModel()
    spy_price = np.cumprod(1 + obs_data[:, 0])

    # Find starting index
    start_idx = np.where(dates >= np.datetime64(start_date))[0][0]
    total_len = len(dates)
    end_idx = min(start_idx + num_frames, total_len)
    
    anim_dates = dates[start_idx:end_idx]
    
    # We need to run the filter from the beginning to have particles at start_idx
    # Standard Model for filtering
    N = 500 # Use 500 for animation speed
    x_curr = ssm.transition(np.array([0.0, -0.5, 0.0]), num_particles=N)
    log_w_curr = np.zeros(N) - np.log(N)

    # Pre-run filter up to start_idx
    print(f"Pre-running filter to {start_date}...")
    for t in range(start_idx):
        # Transition
        x_curr = ssm.transition(x_curr)
        log_lik = ssm.log_likelihood(x_curr, obs_data[t])
        log_w_curr += log_lik
        
        # Simple Resample
        w = np.exp(log_w_curr - np.max(log_w_curr))
        w /= np.sum(w)
        ess = 1.0 / np.sum(w**2)
        if ess < 0.5 * N:
            idx = np.random.choice(N, size=N, p=w)
            x_curr = x_curr[idx]
            log_w_curr = np.zeros(N) - np.log(N)

    # PRE-CALCULATE ALL FILTERED STATS FOR ANIMATION
    print(f"Filtering {len(anim_dates)} frames...")
    all_l_mean = []
    all_spy_norm = []
    all_particles = [] # Need these for fan
    all_weights = []

    for t_idx in range(len(anim_dates)):
        t = start_idx + t_idx
        x_curr = ssm.transition(x_curr)
        log_lik = ssm.log_likelihood(x_curr, obs_data[t])
        log_w_curr += log_lik
        
        w = np.exp(log_w_curr - np.max(log_w_curr))
        w /= np.sum(w)
        
        all_l_mean.append(np.sum(w * x_curr[:, 0]))
        all_spy_norm.append(spy_price[t] / spy_price[start_idx])
        
        # Save a subset of info for the fan
        all_particles.append(x_curr.copy())
        all_weights.append(w.copy())
        
        # Resample for next step
        ess = 1.0 / np.sum(w**2)
        if ess < 0.5 * N:
            idx = np.random.choice(N, size=N, p=w)
            x_curr = x_curr[idx]
            log_w_curr = np.zeros(N) - np.log(N)

    # Convert to arrays
    all_l_mean = np.array(all_l_mean)
    all_spy_norm = np.array(all_spy_norm)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('Systemic Risk: Predictive Forward Simulation', fontsize=16, fontweight='bold')

    # Panel 1: SPY + Forecast Fan
    ax1.set_xlim(mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1] + np.timedelta64(20, 'D')))
    ax1.set_ylim(0.4, 1.3)
    line_spy, = ax1.plot([], [], color='blue', lw=2, label='SPY Index (Observed)')
    fan_lines = [ax1.plot([], [], color='gray', alpha=0.3, lw=0.8)[0] for _ in range(num_fan_paths)]
    ax1.set_ylabel('Normalized Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3) # Increased grid alpha

    # Panel 2: Latent Stress (L_t)
    ax2.set_xlim(mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1]))
    ax2.set_ylim(-0.5, 6.0) # Tightened scale for better sensitivity
    line_l, = ax2.plot([], [], color='green', lw=1.5, label='Latent Stress (L_t)')
    ax2.axhline(2.0, color='black', ls='--', alpha=0.3)
    ax2.set_ylabel('L_t')
    ax2.grid(True, alpha=0.3) # Increased grid alpha

    # Panel 3: Crisis Probability + Events
    ax3.set_xlim(mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1]))
    ax3.set_ylim(0, 1.1)
    line_prob, = ax3.plot([], [], color='red', lw=2, label='P(Crisis next 20d)')
    ax3.set_ylabel('Probability')
    ax3.grid(True, alpha=0.3) # Increased grid alpha
    
    # Event Markers
    events = {'2008-03-14': 'Bear Stearns', '2008-09-15': 'Lehman'}
    for d_str, lbl in events.items():
        dt = mdates.date2num(np.datetime64(d_str))
        ax3.axvline(dt, color='black', ls='--', alpha=0.6)
        ax3.text(dt, 0.9, lbl, rotation=90, ha='right', fontsize=9)

    date_text = ax1.text(0.02, 0.92, '', transform=ax1.transAxes, fontweight='bold')

    def update(frame):
        # Slice data
        curr_d_num = mdates.date2num(anim_dates[:frame+1])
        line_spy.set_data(curr_d_num, all_spy_norm[:frame+1])
        line_l.set_data(curr_d_num, all_l_mean[:frame+1])
        line_prob.set_data(curr_d_num, res_data['crisis_prob'][start_idx:start_idx+frame+1])
        
        # Forecast Fan
        fan_particles = all_particles[frame]
        w = all_weights[frame]
        resample_idx = np.random.choice(N, size=num_fan_paths, p=w)
        sampled_fan = fan_particles[resample_idx]
        
        curr_price = all_spy_norm[frame]
        for i in range(num_fan_paths):
            path_returns = []
            curr_p = sampled_fan[i]
            for _ in range(20):
                curr_p = ssm.transition(curr_p[None, :])[0]
                sigma = np.exp(curr_p[1] / 2.0)
                r = np.random.normal(0, sigma * 0.01)
                path_returns.append(r)
            
            sim_prices = curr_price * np.cumprod(1 + np.array(path_returns))
            forecast_dates = [anim_dates[frame] + np.timedelta64(k, 'D') for k in range(1, 21)]
            fan_lines[i].set_data(mdates.date2num(forecast_dates), sim_prices)
        
        date_text.set_text(f'Date: {str(anim_dates[frame])[:10]} | L_t: {all_l_mean[frame]:.2f}')
        return [line_spy, line_l, line_prob, date_text] + fan_lines

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    ani = FuncAnimation(fig, update, frames=len(anim_dates), blit=False)
    fps = 10
    
    os.makedirs(os.path.join(_project_root, 'results'), exist_ok=True)
    output_path = os.path.join(_project_root, 'results/crisis_evolution_refined.gif')
    print(f"Saving animation to {output_path} (FPS={fps})...")
    ani.save(output_path, writer='pillow', fps=fps)
    print("Animation saved.")

    output_path = os.path.join(_project_root, 'results/crisis_evolution.mp4')
    
    # Try using FFMpegWriter
    try:
        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(output_path, writer=writer)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"FFMpeg failed: {e}. Trying as GIF...")
        output_path = os.path.join(_project_root, 'results/crisis_evolution.gif')
        ani.save(output_path, writer='pillow', fps=15)
        print(f"Animation saved to {output_path}")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    # EXTENDED: Reach the 2008 Lehman bankruptcy (Sep 2008)
    # Start: 2007-06-01, Lehman: 2008-09-15 (~330 trading days)
    create_animation(num_frames=350, start_date='2007-06-01')
