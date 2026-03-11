#!/usr/bin/env python3
"""
Vanguard Phase: Systemic Risk Forecast Evolution.
Features:
- Time-Slice Forecast Cones (Fading "storm tracks" of previous expectations)
- Crisis Heatmap (Dynamic background tinting tied to risk levels)
- Vectorized 20-day predictive forecasts with quantile bands
- High-fidelity causal narrative layout
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from models.finance_state_space import FinanceStateSpaceModel

def create_vanguard_animation(num_frames=350, start_date='2007-06-01', num_sims=120):
    print("Initializing Vanguard-grade systemic risk animation...")
    res_path = os.path.join(_project_root, 'results/crisis_res.npz')
    obs_path = os.path.join(_project_root, 'data/processed/observations.npy')
    
    res_data = np.load(res_path, allow_pickle=True)
    obs_data = np.load(obs_path)
    dates = res_data['dates']
    
    ssm = FinanceStateSpaceModel()
    spy_price = np.cumprod(1 + obs_data[:, 0])

    # Find indices
    start_idx = np.where(dates >= np.datetime64(start_date))[0][0]
    total_len = len(dates)
    end_idx = min(start_idx + num_frames, total_len)
    anim_dates = dates[start_idx:end_idx]

    # DATA PRE-CALCULATION (Filtering)
    N = 500
    x_curr = ssm.transition(np.array([0.0, -0.5, 0.0]), num_particles=N)
    log_w_curr = np.zeros(N) - np.log(N)

    print(f"Filtering {len(anim_dates)} frames for predictive stability...")
    all_l_mean = []
    all_spy_norm = []
    all_particles = [] 
    all_weights = []

    # Pre-run filter up to start_idx
    for t in range(start_idx):
        x_curr = ssm.transition(x_curr)
        log_lik = ssm.log_likelihood(x_curr, obs_data[t])
        log_w_curr += log_lik
        w = np.exp(log_w_curr - np.max(log_w_curr))
        w /= np.sum(w)
        if 1.0 / np.sum(w**2) < 0.5 * N:
            idx = np.random.choice(N, size=N, p=w)
            x_curr = x_curr[idx]
            log_w_curr = np.zeros(N) - np.log(N)

    # Filtering loop for animation frames
    for t_idx in range(len(anim_dates)):
        t = start_idx + t_idx
        x_curr = ssm.transition(x_curr)
        log_lik = ssm.log_likelihood(x_curr, obs_data[t])
        log_w_curr += log_lik
        w = np.exp(log_w_curr - np.max(log_w_curr))
        w /= np.sum(w)
        
        all_l_mean.append(np.sum(w * x_curr[:, 0]))
        all_spy_norm.append(spy_price[t] / spy_price[start_idx])
        all_particles.append(x_curr.copy())
        all_weights.append(w.copy())
        
        if 1.0 / np.sum(w**2) < 0.5 * N:
            idx = np.random.choice(N, size=N, p=w)
            x_curr = x_curr[idx]
            log_w_curr = np.zeros(N) - np.log(N)

    all_l_mean = np.array(all_l_mean)
    all_spy_norm = np.array(all_spy_norm)
    all_probs = res_data['crisis_prob'][start_idx:end_idx]

    # PLOT SETUP
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [2.2, 1, 1]})
    fig.subplots_adjust(hspace=0.35, top=0.9)
    fig.suptitle('Systemic Risk Forecast Evolution - Particle Filter', 
                 fontsize=22, fontweight='bold', color='#1a1a1a')

    # Panel 1: Price Panel with Atmosphere
    ax1.set_title('Market Price & Forward Uncertainty Cones', loc='left', fontsize=14, fontweight='bold')
    ax1.set_xlim(mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1] + np.timedelta64(25, 'D')))
    ax1.set_ylim(0.4, 1.3)
    line_spy, = ax1.plot([], [], color='blue', lw=2.5, label='SPY Index (Observed)', zorder=10)
    
    # Current Fan Objects
    horizon = 20
    fan_fill_95 = ax1.fill_between([], [], [], color='blue', alpha=0.15, label='90% Confidence (Current)', zorder=8)
    fan_fill_50 = ax1.fill_between([], [], [], color='blue', alpha=0.25, label='50% Confidence (Current)', zorder=9)
    fan_median, = ax1.plot([], [], color='blue', ls='--', lw=1.2, alpha=0.7, label='Median Forecast', zorder=9)
    
    # State tracking for Time-Slice Cones
    history_cones = [] # List of PolyCollections
    cone_history_step = 8 # Only leave a "track" every N frames
    
    # Heatmap Atmosphere Row (Professional Macro-Fund Style)
    # We use a narrow horizontal strip at the bottom to avoid visual artifacts
    risk_strip = np.zeros((1, len(anim_dates)))
    extent = [mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1]), 0.40, 0.45]
    heat_map = ax1.imshow(risk_strip, extent=extent, origin='lower', aspect='auto',
                          cmap='Reds', vmin=0, vmax=1.0, alpha=0.5, zorder=0)

    ax1.set_ylabel('Normalized Price', fontsize=13)
    ax1.legend(loc='upper left', fontsize=10, ncol=2, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, ls=':')

    # Panel 2: Inferred Dynamics
    ax2.set_title('Latent Systemic Pressure (Mechanism)', loc='left', fontsize=12, fontweight='bold')
    ax2.set_xlim(mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1]))
    ax2.set_ylim(-.5, 7)
    line_l, = ax2.plot([], [], color='#2e7d32', lw=1.8, label='Latent Stress (L_t)')
    ax2.axhline(2.0, color='black', ls='--', alpha=0.3, label='Stress Threshold')
    ax2.set_ylabel('Liquidity Stress, $L_t$', fontsize=13)
    ax2.grid(True, alpha=0.3, ls=':')

    # Panel 3: Risk Signal
    ax3.set_title('Tail Event Probability (Predictive Signal)', loc='left', fontsize=12, fontweight='bold')
    ax3.set_xlim(mdates.date2num(anim_dates[0]), mdates.date2num(anim_dates[-1]))
    ax3.set_ylim(0, 1.1)
    line_prob, = ax3.plot([], [], color='#c62828', lw=2.5, label='P(Crisis next 20d)')
    ax3.axhline(0.5, color='darkred', ls=':', alpha=0.4, label='Risk Threshold (0.5)')
    ax3.set_ylabel('Pr(Crisis)', fontsize=13)
    ax3.grid(True, alpha=0.3, ls=':')
    
    # Research Annotations
    events = {'2008-03-14': 'Bear Stearns Failure', '2008-09-15': 'Lehman Bankruptcy'}
    for d_str, lbl in events.items():
        dt = mdates.date2num(np.datetime64(d_str))
        ax1.axvline(dt, color='black', ls=':', alpha=0.6, lw=1)
        ax3.axvline(dt, color='black', ls='--', alpha=0.5, lw=1)
        ax3.text(dt, 1.05, lbl, rotation=0, ha='center', fontsize=10, fontweight='bold', transform=ax3.get_xaxis_transform())

    date_text = ax1.text(0.5, 1.1, '', transform=ax1.transAxes, ha='center', fontsize=14, fontweight='bold',
                     bbox=dict(facecolor='white', edgecolor='#333', boxstyle='round,pad=0.5', alpha=0.95))

    def update(frame):
        # 1. Update Heatmap Atmosphere (Smooth imshow reveal)
        curr_risk = np.zeros_like(risk_strip)
        curr_risk[0, :frame+1] = all_probs[:frame+1]
        heat_map.set_data(curr_risk)

        # 2. Update Historical Lines
        curr_d_num = mdates.date2num(anim_dates[:frame+1])
        line_spy.set_data(curr_d_num, all_spy_norm[:frame+1])
        line_l.set_data(curr_d_num, all_l_mean[:frame+1])
        line_prob.set_data(curr_d_num, all_probs[:frame+1])
        
        # 3. Time-Slice Forecast Cones (History)
        # Decay alpha of existing patches
        for p in history_cones:
            curr_a = p.get_alpha()
            if curr_a is not None:
                p.set_alpha(curr_a * 0.999) # Much slower fade for persistent "tracks"
        
        # Remove dead patches
        history_cones[:] = [p for p in history_cones if (p.get_alpha() or 0) > 0.005]

        # 4. New Forecast Generation
        particles = all_particles[frame]
        weights = all_weights[frame]
        resample_idx = np.random.choice(N, size=num_sims, p=weights)
        curr_p = particles[resample_idx]
        
        all_sim_prices = np.zeros((num_sims, horizon + 1))
        all_sim_prices[:, 0] = all_spy_norm[frame]
        
        for h in range(horizon):
            curr_p = ssm.transition(curr_p)
            sigma = np.exp(curr_p[:, 1] / 2.0)
            returns = np.random.normal(0, sigma * 0.01, size=num_sims)
            all_sim_prices[:, h+1] = all_sim_prices[:, h] * (1 + returns)
        
        f_dates = [anim_dates[frame] + np.timedelta64(k, 'D') for k in range(horizon + 1)]
        f_d_num = mdates.date2num(f_dates)
        p5, p25, p50, p75, p95 = np.percentile(all_sim_prices, [5, 25, 50, 75, 95], axis=0)
        
        # Update current fan
        nonlocal fan_fill_95, fan_fill_50
        fan_fill_95.remove()
        fan_fill_50.remove()
        fan_fill_95 = ax1.fill_between(f_d_num, p5, p95, color='blue', alpha=0.18, zorder=8)
        fan_fill_50 = ax1.fill_between(f_d_num, p25, p75, color='blue', alpha=0.28, zorder=9)
        fan_median.set_data(f_d_num, p50)
        
        # Leave a fading trace every N frames
        if frame % cone_history_step == 0:
            trace_p = ax1.fill_between(f_d_num, p5, p95, color='gray', alpha=0.08, zorder=5)
            history_cones.append(trace_p)

        date_text.set_text(f'MARKET DATE: {str(anim_dates[frame])[:10]} | STRESS L_t: {all_l_mean[frame]:.2f}')
        return [line_spy, line_l, line_prob, date_text, fan_median, fan_fill_95, fan_fill_50]

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.tick_params(labelsize=11)

    ani = FuncAnimation(fig, update, frames=len(anim_dates), blit=False)
    
    os.makedirs(os.path.join(_project_root, 'results'), exist_ok=True)
    out_gif = os.path.join(_project_root, 'results/vanguard_risk_evolution.gif')
    print(f"Generating Vanguard-grade animation to {out_gif}...")
    ani.save(out_gif, writer='pillow', fps=12)
    print("Vanguard asset completed successfully.")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    create_vanguard_animation()
