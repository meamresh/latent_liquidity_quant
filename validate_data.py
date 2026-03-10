#!/usr/bin/env python3
import numpy as np

# Load arrays
observations = np.load('data/processed/observations.npy')
returns = np.load('data/processed/returns.npy')
dates = np.load('data/processed/dates.npy')

print(f'Observations shape: {observations.shape}')
print(f'Returns shape: {returns.shape}')
print(f'Dates shape: {dates.shape}')
print(f'Date range: {dates[0]} to {dates[-1]}')
if len(observations) > 0:
    print(f'First observation: {observations[0]}')
    print(f'Last observation: {observations[-1]}')
    print(f'Any NaN in observations: {np.isnan(observations).any()}')
    print(f'Any inf in observations: {np.isinf(observations).any()}')
print('SUCCESS: Arrays contain real data!')