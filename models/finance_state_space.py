#!/usr/bin/env python3
"""
Financial State-Space Model for Latent Liquidity Modeling.

State vector: x_t = [L_t, h_t, z_t]
- L_t: Latent liquidity factor
- h_t: Log volatility 
- z_t: Correlation state (pre-tanh)

Transition dynamics and observation likelihoods for particle filtering.
"""

import numpy as np
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinanceStateSpaceModel:
    """
    Financial state-space model with vectorized transition and likelihood functions.
    
    Designed for particle filtering with multiple particles processed simultaneously.
    """
    
    def __init__(self, dim_state=3, dim_obs=6):
        """
        Initialize the state-space model.
        
        Args:
            dim_state: Dimension of state (always 3: L, h, z)
            dim_obs: Dimension of observations (6: r_SPY, r_TLT, r_HYG, VIX, Spread, Corr)
        """
        self.dim_state = dim_state  # [L_t, h_t, z_t]
        self.dim_obs = dim_obs      # [r_SPY, r_TLT, r_HYG, VIX, Spread, Corr]
        
        # Transition parameters (hyperparameters for tuning)
        self.phi_L = 0.95           # AR coefficient for liquidity
        self.beta_L = 0.3           # Volatility effect on liquidity
        self.mu_h = -0.5            # Mean log-volatility
        self.phi_h = 0.98           # AR coefficient for log-vol
        self.phi_z = 0.95           # AR coefficient for correlation state
        self.gamma = 0.2            # Liquidity effect on correlation
        
        # Process noise standard deviations
        self.sigma_L = 0.05         # Liquidity noise
        self.sigma_h = 0.1          # Log-vol noise
        self.sigma_z = 0.05         # Correlation state noise
        
        # Observation model parameters
        # VIX = a_vix + b_vix * exp(h/2) + noise
        self.a_vix = 15.0
        self.b_vix = 8.0
        self.sigma_vix = 2.0
        
        # Spread = a_spread + b_spread * (L - L_mean) + noise
        self.a_spread = 4.0
        self.b_spread = 2.0
        self.sigma_spread = 0.5
        
        # Correlation = a_corr + b_corr * tanh(z) + noise
        self.a_corr = 0.0
        self.b_corr = 0.5
        self.sigma_corr = 0.1
        
        # Return covariance scaling
        self.sigma_spy = 0.01
        self.sigma_tlt = 0.008
        self.sigma_hyg = 0.012
        
        logger.info(f"Initialized FinanceStateSpaceModel with state_dim={dim_state}, obs_dim={dim_obs}")
    
    def transition(self, x_prev, num_particles=None):
        """
        Vectorized state transition: x_t ~ p(x_t | x_{t-1})
        
        State transitions:
        - L_next = phi_L * L + beta_L * exp(h) + noise_L
        - h_next = mu_h + phi_h * (h - mu_h) + noise_h
        - z_next = phi_z * z + gamma * L + noise_z
        
        Args:
            x_prev: Previous state, shape [num_particles, 3] or [3]
            num_particles: Number of particles (if x_prev is 1D)
            
        Returns:
            x_next: Next state, shape [num_particles, 3]
        """
        x_prev = np.asarray(x_prev, dtype=np.float64)
        
        # Handle 1D input
        if x_prev.ndim == 1:
            if num_particles is None:
                num_particles = 1
            x_prev = np.tile(x_prev, (num_particles, 1))
        
        num_particles = x_prev.shape[0]
        
        # Extract state components
        L = x_prev[:, 0]      # Liquidity
        h = x_prev[:, 1]      # Log-volatility
        z = x_prev[:, 2]      # Correlation state
        
        # Sample process noise
        epsilon_L = np.random.normal(0, self.sigma_L, num_particles)
        epsilon_h = np.random.normal(0, self.sigma_h, num_particles)
        epsilon_z = np.random.normal(0, self.sigma_z, num_particles)
        
        # Transition equations
        L_next = self.phi_L * L + self.beta_L * np.exp(h) + epsilon_L
        h_next = self.mu_h + self.phi_h * (h - self.mu_h) + epsilon_h
        z_next = self.phi_z * z + self.gamma * L + epsilon_z
        
        # Stack into state matrix
        x_next = np.column_stack([L_next, h_next, z_next])
        
        # Check for NaNs or infs
        if np.isnan(x_next).any() or np.isinf(x_next).any():
            logger.warning("NaNs or Infs detected in transition!")
        
        return x_next
    
    def _build_return_covariance(self, h, z):
        """
        Build 3x3 return covariance matrix.
        
        Sigma = sigma^2 * R(rho) where sigma^2 = exp(h)
        
        Args:
            h: Log-volatility, shape [num_particles]
            z: Correlation state, shape [num_particles]
            
        Returns:
            Sigma: Covariance matrix, shape [num_particles, 3, 3]
            sigma: Volatility exp(h/2)
            rho: Correlation tanh(z)
        """
        num_particles = len(h)
        
        # Volatilities
        sigma_sq = np.exp(h)      # Shape: [num_particles]
        sigma = np.sqrt(sigma_sq)  # For VIX likelihood
        
        # Correlation from z
        rho = np.tanh(z)           # Shape: [num_particles]
        
        # Build correlation matrix R(rho)
        R = np.zeros((num_particles, 3, 3))
        for i in range(3):
            R[:, i, i] = 1.0
        R[:, 0, 1] = R[:, 1, 0] = rho
        R[:, 0, 2] = R[:, 2, 0] = rho
        R[:, 1, 2] = R[:, 2, 1] = rho
        
        # Sigma = sigma^2 * R
        Sigma = sigma_sq[:, np.newaxis, np.newaxis] * R
        
        return Sigma, sigma, rho
    
    def _ensure_positive_definite(self, Sigma, min_eig_threshold=1e-6):
        """
        Ensure covariance matrix is positive definite.
        
        Adds small multiple of identity if needed.
        
        Args:
            Sigma: Covariance matrix, shape [num_particles, 3, 3] or [3, 3]
            min_eig_threshold: Minimum eigenvalue threshold
            
        Returns:
            Sigma_pd: PD covariance matrix, adjusted if needed
            adjusted: Boolean array indicating which were adjusted
        """
        if Sigma.ndim == 2:
            # Single matrix case
            Sigma = Sigma[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        num_particles = Sigma.shape[0]
        Sigma_pd = Sigma.copy()
        adjusted = np.zeros(num_particles, dtype=bool)
        
        for p in range(num_particles):
            eigenvalues = np.linalg.eigvalsh(Sigma[p])
            min_eig = eigenvalues.min()
            
            if min_eig < min_eig_threshold:
                adjustment = min_eig_threshold - min_eig + 1e-8
                Sigma_pd[p] += adjustment * np.eye(3)
                adjusted[p] = True
        
        if squeeze_output:
            Sigma_pd = Sigma_pd[0]
        
        return Sigma_pd, adjusted
    
    def log_likelihood(self, x_particles, y_obs):
        """
        Compute observation log-likelihood for each particle.
        
        Observations:
        - Returns: [r_SPY, r_TLT, r_HYG] ~ N(0, Sigma(h, z))
        - VIX ~ N(a_vix + b_vix * sigma, sigma_vix²)
        - Spread ~ N(a_spread + b_spread * L, sigma_spread²)
        - Corr ~ N(a_corr + b_corr * rho, sigma_corr²)
        
        Args:
            x_particles: Particle states, shape [num_particles, 3]
            y_obs: Observation vector, shape [6]
                  [r_SPY, r_TLT, r_HYG, VIX, Spread, rolling_correlation]
        
        Returns:
            log_likelihoods: Log-likelihood for each particle, shape [num_particles]
        """
        x_particles = np.asarray(x_particles, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)
        
        if x_particles.ndim == 1:
            x_particles = x_particles[np.newaxis, :]
        
        num_particles = x_particles.shape[0]
        
        # Extract state components
        L = x_particles[:, 0]      # Liquidity
        h = x_particles[:, 1]      # Log-volatility
        z = x_particles[:, 2]      # Correlation state
        
        # Extract observation components
        returns = y_obs[:3]         # [r_SPY, r_TLT, r_HYG]
        vix_obs = y_obs[3]
        spread_obs = y_obs[4]
        corr_obs = y_obs[5]
        
        # Build return covariance matrix
        Sigma, sigma, rho = self._build_return_covariance(h, z)
        
        # Ensure positive definiteness
        Sigma, adjusted = self._ensure_positive_definite(Sigma)
        
        if adjusted.any():
            logger.debug(f"Adjusted {adjusted.sum()} covariance matrices for PD")
        
        # Compute log-likelihood of returns
        log_lik_returns = np.zeros(num_particles)
        log_const_3d = -1.5 * np.log(2 * np.pi)
        
        for p in range(num_particles):
            try:
                # Use Cholesky decomposition for better stability
                # Sigma = LL^T
                # log|Sigma| = 2 * sum(log(L_ii))
                # y^T Sigma^-1 y = (L^-1 y)^T (L^-1 y)
                L_chol = np.linalg.cholesky(Sigma[p])
                logdet = 2.0 * np.sum(np.log(np.diag(L_chol)))
                
                # Solve L * u = returns
                u = np.linalg.solve(L_chol, returns)
                mahal = np.sum(u**2)
                
                log_lik_returns[p] = -0.5 * logdet - 0.5 * mahal + log_const_3d
                
            except (np.linalg.LinAlgError, ValueError):
                log_lik_returns[p] = -np.inf
        
        # Scalar likelihood constants
        log_const_1d = -0.5 * np.log(2 * np.pi)
        
        # VIX likelihood: VIX ~ N(a_vix + b_vix * sigma, sigma_vix²)
        vix_pred = self.a_vix + self.b_vix * sigma
        log_lik_vix = -0.5 * ((vix_obs - vix_pred) / self.sigma_vix) ** 2 - np.log(self.sigma_vix) + log_const_1d
        
        # Spread likelihood: Spread ~ N(a_spread + b_spread * L, sigma_spread²)
        spread_pred = self.a_spread + self.b_spread * L
        log_lik_spread = -0.5 * ((spread_obs - spread_pred) / self.sigma_spread) ** 2 - np.log(self.sigma_spread) + log_const_1d
        
        # Correlation likelihood: Corr ~ N(a_corr + b_corr * rho, sigma_corr²)
        corr_pred = self.a_corr + self.b_corr * rho
        
        # Clamp observed correlation to valid range for numerical stability
        corr_obs_clamped = np.clip(corr_obs, -0.99, 0.99)
        
        log_lik_corr = -0.5 * ((corr_obs_clamped - corr_pred) / self.sigma_corr) ** 2 - np.log(self.sigma_corr) + log_const_1d
        
        # Total log-likelihood
        log_likelihoods = log_lik_returns + log_lik_vix + log_lik_spread + log_lik_corr
        
        # Check for NaNs or infinities
        invalid_mask = ~np.isfinite(log_likelihoods)
        if invalid_mask.any():
            logger.debug(f"Found {invalid_mask.sum()} invalid log-likelihoods")
            # Replace with floor
            log_likelihoods[invalid_mask] = -1000.0
        
        return log_likelihoods
    
    def get_parameters(self):
        """Get all model parameters as a dictionary."""
        return {
            # Transition parameters
            'phi_L': self.phi_L,
            'beta_L': self.beta_L,
            'mu_h': self.mu_h,
            'phi_h': self.phi_h,
            'phi_z': self.phi_z,
            'gamma': self.gamma,
            # Process noise
            'sigma_L': self.sigma_L,
            'sigma_h': self.sigma_h,
            'sigma_z': self.sigma_z,
            # Observation model - VIX
            'a_vix': self.a_vix,
            'b_vix': self.b_vix,
            'sigma_vix': self.sigma_vix,
            # Observation model - Spread
            'a_spread': self.a_spread,
            'b_spread': self.b_spread,
            'sigma_spread': self.sigma_spread,
            # Observation model - Correlation
            'a_corr': self.a_corr,
            'b_corr': self.b_corr,
            'sigma_corr': self.sigma_corr,
            # Return volatilities
            'sigma_spy': self.sigma_spy,
            'sigma_tlt': self.sigma_tlt,
            'sigma_hyg': self.sigma_hyg,
        }
    
    def set_parameters(self, params):
        """Update model parameters from dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter: {key}")


def test_state_space_model():
    """Test the state-space model with synthetic data."""
    logger.info("Testing FinanceStateSpaceModel...")
    
    # Initialize model
    model = FinanceStateSpaceModel()
    
    # Initial state
    x0 = np.array([0.0, -0.5, 0.0])  # [L, h, z]
    
    # Test transition
    logger.info("Testing transition function...")
    x_particles = model.transition(x0, num_particles=1000)
    assert x_particles.shape == (1000, 3), f"Expected shape (1000, 3), got {x_particles.shape}"
    assert not np.isnan(x_particles).any(), "NaNs in transition output"
    logger.info(f"✓ Transition OK: mean state = {x_particles.mean(axis=0)}")
    
    # Test likelihood with synthetic observation
    logger.info("Testing log-likelihood function...")
    y_obs = np.array([0.01, -0.005, 0.008, 20.0, 3.5, 0.1])  # Synthetic observation
    log_liks = model.log_likelihood(x_particles, y_obs)
    
    assert log_liks.shape == (1000,), f"Expected shape (1000,), got {log_liks.shape}"
    assert not np.isnan(log_liks).any(), "NaNs in log-likelihood output"
    assert np.isfinite(log_liks).all(), "Non-finite values in log-likelihood"
    
    logger.info(f"✓ Log-likelihood OK: mean LL = {log_liks.mean():.2f}, std = {log_liks.std():.2f}")
    logger.info(f"   Min LL = {log_liks.min():.2f}, Max LL = {log_liks.max():.2f}")
    
    # Test multiple steps
    logger.info("Testing multiple transition steps...")
    x = x0[np.newaxis, :]  # Shape: [1, 3]
    for t in range(10):
        x = model.transition(x, num_particles=1)
        assert not np.isnan(x).any(), f"NaNs at step {t}"
    logger.info(f"✓ Multi-step OK: final state = {x[0]}")
    
    logger.info("All tests passed! ✓")
    
    return model


if __name__ == "__main__":
    # Run tests
    model = test_state_space_model()
    
    print("\n" + "="*60)
    print("MODEL PARAMETERS")
    print("="*60)
    params = model.get_parameters()
    for key, value in params.items():
        print(f"{key:20s}: {value:.6f}")
