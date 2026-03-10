#!/usr/bin/env python3
"""
Compute features for latent liquidity modeling.
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_log_returns(data):
    """
    Compute daily log returns for SPY, TLT, HYG.

    Args:
        data: DataFrame with price data

    Returns:
        DataFrame with log returns
    """
    symbols = ['SPY', 'TLT', 'HYG']
    returns = {}

    for symbol in symbols:
        if symbol in data.columns:
            returns[f'r_{symbol}'] = np.log(data[symbol] / data[symbol].shift(1))

    return pd.DataFrame(returns, index=data.index)

def compute_realized_volatility(data, window=30):
    """
    Compute 30-day rolling realized volatility of SPY.

    Args:
        data: DataFrame with price data
        window: Rolling window size

    Returns:
        Series with volatility
    """
    if 'SPY' not in data.columns:
        return pd.Series(dtype=float, name='vol_SPY')

    spy_returns = np.log(data['SPY'] / data['SPY'].shift(1))
    vol = spy_returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    vol.name = 'vol_SPY'

    return vol

def compute_rolling_correlation(data, window=30):
    """
    Compute 30-day rolling average correlation between SPY, TLT, HYG.

    Args:
        data: DataFrame with price data
        window: Rolling window size

    Returns:
        Series with average correlation
    """
    symbols = ['SPY', 'TLT', 'HYG']
    available_symbols = [s for s in symbols if s in data.columns]

    if len(available_symbols) < 2:
        return pd.Series(dtype=float, name='avg_corr')

    returns = compute_log_returns(data)[[f'r_{s}' for s in available_symbols]]

    # Compute pairwise correlations and average them
    avg_corrs = []
    for i in range(len(returns)):
        if i >= window - 1:
            window_data = returns.iloc[i-window+1:i+1]
            if len(window_data) >= window:
                corr_matrix = window_data.corr()
                # Get average of all pairwise correlations (excluding self-correlations)
                correlations = []
                for j in range(len(available_symbols)):
                    for k in range(j+1, len(available_symbols)):
                        correlations.append(corr_matrix.iloc[j, k])
                avg_corr = np.mean(correlations) if correlations else np.nan
            else:
                avg_corr = np.nan
        else:
            avg_corr = np.nan
        avg_corrs.append(avg_corr)

    return pd.Series(avg_corrs, index=returns.index, name='avg_corr')

def build_observation_vector(data):
    """
    Build the final observation vector.

    y_t = [r_SPY_t, r_TLT_t, r_HYG_t, VIX_t, Spread_t, Corr_t]

    Args:
        data: DataFrame with all required data

    Returns:
        Tuple of (observations, returns, dates)
    """
    # Compute returns
    returns_df = compute_log_returns(data)
    returns = returns_df[['r_SPY', 'r_TLT', 'r_HYG']].values

    # Get VIX
    vix = data['VIX'].values if 'VIX' in data.columns else np.full(len(data), np.nan)

    # Get spread
    spread = data['IG_Spread'].values if 'IG_Spread' in data.columns else np.full(len(data), np.nan)

    # Compute volatility
    vol = compute_realized_volatility(data).values

    # Compute correlation
    corr = compute_rolling_correlation(data).values

    # Combine into observation vector
    observations = np.column_stack([
        returns_df['r_SPY'].values,
        returns_df['r_TLT'].values,
        returns_df['r_HYG'].values,
        vix,
        spread,
        corr
    ])

    # Get valid dates (non-NaN observations)
    valid_mask = ~np.isnan(observations).any(axis=1)
    observations = observations[valid_mask]
    returns = returns[valid_mask]
    dates = data.index[valid_mask]

    logger.info(f"Built {len(observations)} valid observation vectors")

    return observations, returns, dates

if __name__ == "__main__":
    # Load raw data
    data_file = 'data/raw/financial_data.csv'
    if os.path.exists(data_file):
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        # Rename ^VIX to VIX for consistency
        data = data.rename(columns={'^VIX': 'VIX'})
        logger.info(f"Loaded data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")

        # Build observation vectors
        observations, returns, dates = build_observation_vector(data)

        # Save as numpy arrays
        os.makedirs('data/processed', exist_ok=True)
        np.save('data/processed/observations.npy', observations)
        np.save('data/processed/returns.npy', returns)
        np.save('data/processed/dates.npy', dates)

        print("Features computed and saved:")
        print(f"Observations shape: {observations.shape}")
        print(f"Returns shape: {returns.shape}")
        print(f"Dates shape: {dates.shape}")
        if len(observations) > 0:
            print(f"Sample observation: {observations[0]}")
        else:
            print("No valid observations found!")
    else:
        print("Raw data not found. Run download.py first.")