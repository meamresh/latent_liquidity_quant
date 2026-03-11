import os
import sys
import pytest
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def model():
    from models.finance_state_space import FinanceStateSpaceModel
    return FinanceStateSpaceModel()
