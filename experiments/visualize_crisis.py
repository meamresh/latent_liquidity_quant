#!/usr/bin/env python3
"""
Phase 5: Advanced Crisis Visualization.
Generates high-impact dual-axis plots of market price vs. crisis risk.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter

# Add project root to path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def create_advanced_plots():
    print("Loading simulation data...")
    res_path = os.path.join(_project_root, 'results/crisis_res.npz')
    obs_path = os.path.join(_project_root, 'data/processed/observations.npy')
    
    if not os.path.exists(res_path) or not os.path.exists(obs_path):
        print("Error: Missing data files. Run crisis_predictor.py first.")
        return

    res = np.load(res_path, allow_pickle=True)
    obs = np.load(obs_path)
    
    dates = res['dates']
    crisis_prob = res['crisis_prob']
    l_mean = res['L_mean']
    l_q05 = res['L_q05']
    l_q95 = res['L_q95']
    p_l_gt_2 = res['P_L_gt_2']
    
    # 1. Calculate Cumulative SPY Returns (Price Proxy)
    # Assuming first column of observations is SPY returns
    spy_returns = obs[:, 0]
    spy_price = np.cumprod(1 + spy_returns)
    
    print("Generating dual-axis crisis plot...")
    plt.figure(figsize=(16, 10))
    
    # --- TOP PANEL: SPY vs Crisis Probability ---
    ax1 = plt.subplot(2, 1, 1)
    
    # Left Axis: SPY Price
    color_spy = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('SPY Price (Normalized)', color=color_spy, fontsize=12)
    ax1.plot(dates, spy_price, color=color_spy, linewidth=2, label='SPY Index', zorder=5)
    ax1.tick_params(axis='y', labelcolor=color_spy)
    ax1.set_ylim(0.4, 1.3)
    ax1.set_title('Market Performance vs. Systemic Crisis Probability', fontsize=14, fontweight='bold')
    
    # Crash Region Highlight (Sept 2008 - Mar 2009)
    ax1.axvspan(np.datetime64('2008-09-01'), np.datetime64('2009-03-01'), 
                color='red', alpha=0.08, label='2008 Market Crash', zorder=0)

    # Event Annotations
    events = {
        '2008-03-14': 'Bear Stearns',
        '2008-09-15': 'Lehman Brothers',
        '2020-02-20': 'COVID Peak'
    }
    for date_str, label in events.items():
        dt = np.datetime64(date_str)
        if dt in dates:
            ax1.axvline(dt, color='black', linestyle='--', alpha=0.6)
            ax1.text(dt, 1.25, label, rotation=0, ha='center', fontsize=10, fontweight='bold', transform=ax1.get_xaxis_transform())

    # Right Axis: Crisis Probability
    ax2 = ax1.twinx()
    color_risk = 'tab:red'
    ax2.set_ylabel('P(Crisis within Horizon)', color=color_risk, fontsize=12)
    ax2.fill_between(dates, crisis_prob, color=color_risk, alpha=0.2, label='Crisis Probability')
    ax2.plot(dates, crisis_prob, color=color_risk, linewidth=1.5, alpha=0.8)
    ax2.axhline(0.5, color='darkred', linestyle=':', alpha=0.5, label='Risk Threshold (0.5)')
    ax2.tick_params(axis='y', labelcolor=color_risk)
    ax2.set_ylim(0, 1.1)
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # --- BOTTOM PANEL: Liquidity Tail Risk & Latent State ---
    ax3 = plt.subplot(2, 1, 2)
    
    # Plot L_t with CI
    ax3.plot(dates, l_mean, color='green', label='Liquidity Stress (L_t) Mean', alpha=0.8)
    ax3.fill_between(dates, l_q05, l_q95, color='green', alpha=0.15, label='90% Confidence Interval')
    ax3.axhline(y=2.0, color='black', linestyle='--', alpha=0.5, label='Stress Threshold (L=2.0)')
    
    # Use reasonable limits for Lt
    l_min, l_max = np.percentile(l_mean, [0, 99.5])
    ax3.set_ylim(min(-0.1, l_min), l_max * 1.5)

    # Overlay Tail Probability P(L > 2)
    ax4 = ax3.twinx()
    ax4.bar(dates, p_l_gt_2, color='darkred', alpha=0.3, width=1.0, label='Tail Risk P(L > 2.0)')
    ax4.set_ylabel('Tail Probability', color='darkred')
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(axis='y', labelcolor='darkred')
    
    ax3.set_ylabel('Liquidity State (L_t)')
    ax3.set_title('Inferred Liquidity Stress & Distributional Tail Risk', fontsize=12)
    
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.join(_project_root, 'results'), exist_ok=True)
    output_path = os.path.join(_project_root, 'results/crisis_analysis_advanced.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

    # --- CRITICAL EVENTS DETAIL PLOTS ---
    # Zoom in on 2008
    plt.figure(figsize=(14, 7))
    mask_2008 = (dates >= np.datetime64('2007-06-01')) & (dates <= np.datetime64('2009-03-31'))
    
    ax_zoom = plt.gca()
    ax_zoom.set_title('2008 Financial Crisis: Leading Crisis Signals', fontsize=14, fontweight='bold')
    ax_zoom.plot(dates[mask_2008], spy_price[mask_2008] / spy_price[mask_2008][0], 
                color='blue', label='SPY (Normalized)', linewidth=2)
    ax_zoom.axvspan(np.datetime64('2008-09-01'), np.datetime64('2009-03-01'), color='red', alpha=0.08, label='Crash Window')
    ax_zoom.set_ylabel('Market Price')

    ax_zoom_r = ax_zoom.twinx()
    ax_zoom_r.fill_between(dates[mask_2008], crisis_prob[mask_2008], color='red', alpha=0.25, label='P(Crisis)')
    ax_zoom_r.plot(dates[mask_2008], crisis_prob[mask_2008], color='red', alpha=0.8, linewidth=1.5)
    ax_zoom_r.set_ylabel('Crisis Probability')
    ax_zoom_r.set_ylim(0, 1.1)
    
    # Annotation
    lehman_dt = np.datetime64('2008-09-15')
    ax_zoom.axvline(lehman_dt, color='black', ls='--')
    ax_zoom.text(lehman_dt, 0.9, 'Lehman Bankruptcy', rotation=0, ha='center', fontweight='bold', transform=ax_zoom.get_xaxis_transform())

    lines_z1, labels_z1 = ax_zoom.get_legend_handles_labels()
    lines_z2, labels_z2 = ax_zoom_r.get_legend_handles_labels()
    ax_zoom.legend(lines_z1 + lines_z2, labels_z1 + labels_z2, loc='lower left')
    ax_zoom.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(_project_root, 'results/crisis_zoom_2008.png'), dpi=300)
    print("Zoomed plot for 2008 crisis saved.")

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    create_advanced_plots()
