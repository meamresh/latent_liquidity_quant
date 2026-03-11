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

from models.finance_state_space import FinanceStateSpaceModel

# Comprehensive Global Events Dictionary (Unified with animate_crisis)
GLOBAL_EVENTS = {
    '2008-03-14': 'Bear Stearns Failure',
    '2008-09-15': 'Lehman Bankruptcy',
    '2012-07-26': 'Eurozone ("Whatever it takes")',
    '2016-06-24': 'Brexit Vote',
    '2020-03-16': 'COVID Peak Stress',
    '2023-03-10': 'SVB Collapse'
}

def create_vanguard_plots(start_date=None, end_date=None, output_name='crisis_analysis_advanced.png', is_notebook=False):
    """
    Generates high-impact dual-axis plots for a specific interval.
    """
    import matplotlib
    if not is_notebook:
        matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    print(f"Generating Vanguard-grade static analysis: {start_date or 'Start'} to {end_date or 'End'}...")
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

    # Filter by date
    mask = np.ones(len(dates), dtype=bool)
    if start_date:
        mask &= (dates >= np.datetime64(start_date))
    if end_date:
        mask &= (dates <= np.datetime64(end_date))
    
    if not np.any(mask):
        print(f"No data found for interval {start_date} to {end_date}")
        return

    d_plt = dates[mask]
    spy_plt = spy_price[mask] / spy_price[mask][0] # Normalize to start of interval
    prob_plt = crisis_prob[mask]
    l_plt = l_mean[mask]
    l05_plt = l_q05[mask]
    l95_plt = l_q95[mask]
    tail_plt = p_l_gt_2[mask]

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1.2, 1]})
    fig.subplots_adjust(hspace=0.3)
    
    # --- TOP PANEL: SPY vs Crisis Probability ---
    color_spy = 'tab:blue'
    ax1.set_ylabel('SPY Price (Normalized)', color=color_spy, fontsize=12)
    ax1.plot(d_plt, spy_plt, color=color_spy, linewidth=2.5, label='SPY Index', zorder=5)
    ax1.tick_params(axis='y', labelcolor=color_spy)
    ax1.set_ylim(min(spy_plt)*0.9, max(spy_plt)*1.1)
    ax1.set_title('Market Performance vs. Systemic Crisis Probability', fontsize=16, fontweight='bold')
    
    # Right Axis: Crisis Probability
    ax2 = ax1.twinx()
    color_risk = 'tab:red'
    ax2.set_ylabel('P(Crisis next 60d)', color=color_risk, fontsize=12)
    ax2.fill_between(d_plt, prob_plt, color=color_risk, alpha=0.15, label='Crisis Probability')
    ax2.plot(d_plt, prob_plt, color=color_risk, linewidth=1.5, alpha=0.6)
    ax2.axhline(0.5, color='darkred', linestyle=':', alpha=0.4, label='Risk Threshold (0.5)')
    ax2.tick_params(axis='y', labelcolor=color_risk)
    ax2.set_ylim(0, 1.1)
    
    # Global Events
    relevant_events = {k: v for k, v in GLOBAL_EVENTS.items() if (np.datetime64(k) >= d_plt[0]) and (np.datetime64(k) <= d_plt[-1])}
    for d_str, lbl in relevant_events.items():
        dt = np.datetime64(d_str)
        ax1.axvline(dt, color='black', linestyle='--', alpha=0.6, lw=1)
        ax1.text(dt, 1.05, lbl, rotation=0, ha='center', fontsize=10, fontweight='bold', transform=ax1.get_xaxis_transform())

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, ncol=2)
    ax1.grid(True, alpha=0.3, ls=':')

    # --- BOTTOM PANEL: Liquidity Tail Risk & Latent State ---
    ax3.plot(d_plt, l_plt, color='green', label='Liquidity Stress ($L_t$)', alpha=0.8, lw=2)
    ax3.fill_between(d_plt, l05_plt, l95_plt, color='green', alpha=0.1, label='90% CI')
    ax3.axhline(y=2.0, color='black', linestyle='--', alpha=0.3, label='Stress Threshold (L=2.0)')
    ax3.set_ylim(-0.5, 7.0)
    ax3.set_ylabel('Liquidity Stress, $L_t$', fontsize=12)
    ax3.set_title('Inferred Liquidity Stress & Tail Risk P(L > 2.0)', fontsize=14)
    
    # Overlay Tail Probability
    ax4 = ax3.twinx()
    ax4.bar(d_plt, tail_plt, color='darkred', alpha=0.2, width=1.0, label='Tail Risk $P(L_t > 2.0)$')
    ax4.set_ylabel('Tail Probability', color='darkred')
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(axis='y', labelcolor='darkred')
    
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, ls=':')

    plt.tight_layout()
    
    if is_notebook:
        plt.show()
        return fig

    os.makedirs(os.path.join(_project_root, 'results'), exist_ok=True)
    out_path = os.path.join(_project_root, 'results', output_name)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Vanguard analysis saved to {out_path}")
    return out_path

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    create_vanguard_plots(start_date='2007-06-01', end_date='2009-03-01', output_name='vanguard_static_2008.png')
    create_vanguard_plots(start_date='2019-12-01', end_date='2020-06-01', output_name='vanguard_static_covid.png')
