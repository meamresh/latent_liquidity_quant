"""Tests for latent liquidity quantitative modeling system."""

import sys
import os

# Add parent directory to path so tests can import from project root
_test_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_test_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
