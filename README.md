# Latent Liquidity & Systemic Risk Quantification

🌐 [Interactive Portfolio](https://meamresh.github.io/latent-liquidity.html) • 📊 [Research Visualizations](results/) • 📄 [Notebook Analysis](notebooks/)

<p align="center">
  <img src="results/vanguard_risk_evolution.gif" width="400">
</p>

This repository implements a non-linear State-Space Model (SSM) and an advanced Bayesian filtering framework to infer latent liquidity stress ($L_t$) and quantify systemic risk transitions in global financial markets.

## Key Features

- **Bayesian Inference Engine**: Non-linear filtering using a custom `FinanceStateSpaceModel` to extract hidden stress signals from multi-asset return/volatility data (SPY, TLT, HYG).
- **Predictive Horizon Mapping**: 60-day forward-looking Monte Carlo simulations to calculate the distribution of future market outcomes.
- **Systemic Risk Signaling**: Automated calculation of tail-risk probabilities ($P(L_t > 2.0)$) and predictive crisis transition metrics.
- **Vanguard Visualization**: Publication-grade animations featuring "Storm Tracks" (fading forecast cones) and "Atmospheric Heatmaps" to communicate the evolution of market expectations.
- **Era Exploration**: Built-in support for analyzing historical regimes, including the 2008 GFC, Eurozone Crisis, Brexit, COVID-19, and the 2023 SVB collapse.
- **Notebook Integrated**: Optimized for stable Jupyter Notebook usage with robust path resolution and headless plotting support.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

The project features a unified CLI (`main.py`) to streamline research workflows.

### 1. Run inference & Prediction
Generate latent state estimates and crisis probabilities:
```bash
python3 main.py predict
```

### 2. Static Research Analysis
Generate high-fidelity dual-axis plots for any historical era:
```bash
python3 main.py plot --start 2008-01-01 --end 2008-12-31 --output gfc_analysis.png
```

### 3. Vanguard Animation Suite
Generate professional "Storm Track" animations of risk evolution:
```bash
python3 main.py animate --start 2020-01-01 --end 2020-05-01 --output covid_era.gif
```

## Testing & Quality

The project includes a robust `pytest` suite to ensure mathematical stability and filter reliability.

```bash
# Run all tests
pytest tests/
```

## Project Structure

- `models/`: Non-linear finance state-space models and transition logic.
- `filters/`: Particle filtering implementations (Standard, LEDH).
- `experiments/`: Core research scripts for prediction and publication-grade visualization.
- `notebooks/`: Interactive exploratory analysis and era-specific deep dives.
- `results/`: Output artifacts (GIFs, PNGs, and pre-calculated `.npz` simulation data).

## Research Visualization

The project specializes in **Vanguard-grade storytelling**, layering market observables over inferred latent mechanisms:
- **Phase 9 (Visual Persistence)**: Fading Forecast Cones (Storm Tracks) visualize the "collapse" of future expectations during regime shifts.
- **Phase 10 (Multi-Era)**: Automatic detection and annotation of global liquidity events (2008-2023).

---
*Developed for quantitative macro-finance and systemic risk research.*
