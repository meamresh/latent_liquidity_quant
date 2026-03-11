#!/usr/bin/env python3
"""
Latent Liquidity & Systemic Risk Quantification - Unified CLI.
Consolidates inference, static analysis, and Vanguard animations.
"""

import argparse
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiments.crisis_predictor import run_crisis_prediction
from experiments.visualize_crisis import create_vanguard_plots
from experiments.animate_crisis import create_vanguard_animation

def main():
    parser = argparse.ArgumentParser(description='Latent Liquidity & Systemic Risk CLI')
    subparsers = parser.add_subparsers(dest='command', help='Sub-command to run')

    # Predict subcommand
    predict_parser = subparsers.add_parser('predict', help='Run systemic risk inference engine')
    predict_parser.add_argument('--num-particles', type=int, default=500, help='Number of particles for filtering')

    # Plot subcommand
    plot_parser = subparsers.add_parser('plot', help='Generate static research analysis plots')
    plot_parser.add_argument('--start', type=str, default='2007-06-01', help='Start date (YYYY-MM-DD)')
    plot_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    plot_parser.add_argument('--output', type=str, default='vanguard_static_analysis.png', help='Output filename')

    # Animate subcommand
    animate_parser = subparsers.add_parser('animate', help='Generate Vanguard-grade risk evolution GIFs')
    animate_parser.add_argument('--start', type=str, default='2007-06-01', help='Start date (YYYY-MM-DD)')
    animate_parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    animate_parser.add_argument('--output', type=str, default='vanguard_evolution.gif', help='Output filename')
    animate_parser.add_argument('--sims', type=int, default=120, help='Number of simulations per frame')

    args = parser.parse_args()

    if args.command == 'predict':
        print(f"--- Running Prediction (N={args.num_particles}) ---")
        run_crisis_prediction() # Note: We could update crisis_predictor to take arguments if needed
    
    elif args.command == 'plot':
        print(f"--- Generating Static Plot: {args.start} to {args.end or 'End'} ---")
        create_vanguard_plots(start_date=args.start, end_date=args.end, output_name=args.output)
    
    elif args.command == 'animate':
        print(f"--- Generating Vanguard Animation: {args.start} to {args.end or 'End'} ---")
        create_vanguard_animation(start_date=args.start, end_date=args.end, output_name=args.output, num_sims=args.sims)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
