#!/usr/bin/env python3
"""
Download financial data for latent liquidity modeling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_daily_data(start_date='2005-01-01', end_date=None):
    """
    Download daily financial data.

    Returns:
        DataFrame with adjusted closes and VIX
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Download equity ETFs
    symbols = ['SPY', 'TLT', 'HYG']
    logger.info(f"Downloading data for {symbols}")

    equity_data = yf.download(symbols, start=start_date, end=end_date)['Close']

    # Download VIX
    logger.info("Downloading VIX data")
    vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']
    vix_data.name = 'VIX'

    # Combine data
    data = pd.concat([equity_data, vix_data], axis=1)

    # Forward fill missing values (for weekends/holidays)
    data = data.fillna(method='ffill')

    logger.info(f"Downloaded data shape: {data.shape}")
    return data

def download_credit_spread(start_date='2005-01-01', end_date=None):
    """
    Download ICE BofA IG credit spread from FRED.

    Returns:
        Series with credit spread
    """
    import pandas_datareader.data as web
    from pandas_datareader import fred

    if end_date is None:
        end_date = datetime.now()

    logger.info("Downloading ICE BofA IG credit spread")
    spread = web.DataReader('BAMLH0A0HYM2', 'fred', start_date, end_date)
    spread = spread.squeeze()
    spread.name = 'IG_Spread'

    return spread

if __name__ == "__main__":
    # Download data
    data = download_daily_data()
    spread = download_credit_spread()

    # Combine all data
    all_data = pd.concat([data, spread], axis=1)

    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    all_data.to_csv('data/raw/financial_data.csv')

    print("Data downloaded and saved to data/raw/financial_data.csv")
    print(f"Data shape: {all_data.shape}")
    print(all_data.head())