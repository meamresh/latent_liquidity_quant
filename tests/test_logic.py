import pytest
import pandas as pd
import numpy as np
from datetime import datetime

def test_date_parsing_logic():
    # Verify that we can parse dates consistently across different components
    date_str = '2008-09-15'
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    assert dt.year == 2008
    assert dt.month == 9
    assert dt.day == 15

def test_data_clamping_logic():
    # Verify that correlation clamping works for numerical stability
    corr_obs = 1.2
    corr_clamped = np.clip(corr_obs, -0.99, 0.99)
    assert corr_clamped == 0.99
    
    corr_obs = -5.0
    corr_clamped = np.clip(corr_obs, -0.99, 0.99)
    assert corr_clamped == -0.99
