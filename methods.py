# brtacfr_estimator.py

import os
import warnings

# Configure PyTensor BEFORE any imports that use it
os.environ.setdefault('PYTENSOR_FLAGS', 'optimizer=fast_compile,exception_verbosity=low')

# Suppress PyTensor BLAS warning
warnings.filterwarnings('ignore', message='.*PyTensor could not link to a BLAS.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')

import numpy as np
import pandas as pd
from scipy.stats import gamma

# =============================================================================
# Custom Log-Probability Function for the Prior
# =============================================================================

def logp(value, mu, lam, sigma, T):
    """
    Custom log-probability for the mixture prior on beta.

    This prior combines a Fused LASSO penalty with a Normal distribution,
    encouraging both smoothness (piecewise constant) and regularization
    towards a prior mean (mu).

    Args:
        value (pyt.Tensor): The parameter vector (beta).
        mu (pyt.Tensor): The prior mean for the Normal component.
        lam (pyt.Tensor): The smoothing parameter for the Fused LASSO penalty.
        sigma (float): The standard deviation for the Normal component.
        T (int): The length of the time series.

    Returns:
        pyt.Tensor: The log-probability of the prior.
    """
    # Lazy import so plotting-only runs don't require PyMC/PyTensor installed/working.
    import pytensor.tensor as pt

    # Fused LASSO component: Penalizes large differences between adjacent time points
    fused_lasso_logp = -lam*pt.sum(pt.abs(pt.diff(value)))+(T-1)*pt.log(lam/2)

    # Normal component: Pulls the estimate towards the prior mean (logit-cCFR)
    normal_logp = -0.5*pt.sum((value-mu)**2)/sigma**2-0.5*T*pt.log(2*np.pi*sigma**2)
    
    return fused_lasso_logp + normal_logp

# =============================================================================
# Helper Function for mCFR Calculation
# =============================================================================

def lambda_summary_stats(draws):
    """
    Compute median and 95% CrI (2.5%, 97.5%) from a 1D array of lambda draws.
    
    Args:
        draws (np.ndarray): 1D array of posterior/variational draws for lambda.
    
    Returns:
        tuple: (median, q025, q975). Returns (None, None, None) if draws is empty or None.
    """
    if draws is None or len(draws) == 0:
        return (None, None, None)
    draws = np.asarray(draws).flatten()
    return (
        float(np.median(draws)),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def mCFR_EST(c_t, d_t, f_k):
    """
    Estimates the modified Case Fatality Rate (mCFR).

    This estimator adjusts the denominator of the crude CFR by convolving the
    case counts with the delay distribution.

    Args:
        c_t (np.ndarray): Daily case counts.
        d_t (np.ndarray): Daily death counts.
        f_k (np.ndarray): Probability mass function of the delay distribution.

    Returns:
        np.ndarray: The time series of mCFR estimates.
    """
    T = c_t.shape[0]
    # Adjust the denominator by the delay distribution
    c_t_d = np.array([np.sum(np.flip(f_k[:i])*c_t[:i]) for i in np.arange(1,T+1)])
    # Compute the cumulative fatality rate with the adjusted denominator
    return np.array([np.sum(d_t[:i])/(np.sum(c_t_d[:i])+1e-10) for i in np.arange(1,T+1)])

# =============================================================================
# Main BrtaCFR Estimator Function
# =============================================================================

def BrtaCFR_estimator(c_t, d_t, F_paras, lambda_scale=1.0, n_draws=1000):
    """
    Estimates the Bayesian real-time adjusted Case Fatality Rate (BrtaCFR).

    This function implements the Bayesian model described in the manuscript to
    estimate the time-varying case fatality rate, p(t).

    Args:
        c_t (np.array): Time series of daily confirmed cases.
        d_t (np.array): Time series of daily deaths.
        F_paras (tuple): Parameters (mean, shape) for the Gamma delay distribution.
        lambda_scale (float): Scale (beta) for the half-Cauchy prior on lambda; default 1.0.
        n_draws (int): Number of posterior draws to sample (default 1000).

    Returns:
        dict: A dictionary containing the posterior mean, 95% credible intervals,
              and optionally lambda_draws (1D array) when inference succeeds.
    """
    # Lazy import so scripts that only need plotting/checkpoint reading can run
    # even if PyMC/PyTensor are unavailable or mismatched.
    import pymc as pm
    
    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay
        
    # Calculate delay distribution PMF from Gamma distribution
    F_k = gamma.cdf(np.arange(T+1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    
    # f(K) matrix
    f_mat = np.zeros((T,T))
    for i in range(T):
        f_mat += np.diag(np.ones(T-i)*f_k[i],-i)
    # c(t) matrix
    c_mat = np.diag(c_t)
    fc_mat = np.dot(f_mat, c_mat)
    
    # --- 1. Calculate cCFR (crude Case Fatality Rate) ---
    cCFR_est = np.cumsum(d_t) / (np.cumsum(c_t) + 1e-10)
    
    # Clip cCFR to avoid extreme values
    cCFR_est = np.clip(cCFR_est, 1e-8, 1 - 1e-8)
    
    # --- 2. Bayesian Model for BrtaCFR ---
    with pm.Model() as model:
        # Priors
        # Prior mean for beta_t based on logit-transformed cCFR
        beta_tilde = np.log((cCFR_est+1e-10)/(1 - (cCFR_est+1e-10)))
        
        # Check for NaN or Inf in initialization
        if not np.all(np.isfinite(beta_tilde)):
            beta_tilde = np.nan_to_num(beta_tilde, nan=0.0, posinf=5.0, neginf=-5.0)
        
        # Hyperprior for the smoothing parameter lambda: λ ~ HalfCauchy(0, lambda_scale)
        lambda_param = pm.HalfCauchy('lambda', beta=lambda_scale)
        
        # Priors for beta_t
        beta = pm.CustomDist('beta',
                              beta_tilde, lambda_param, 5, T,
                              logp=logp,
                              size=T)
    
        # Transformation to get fatality rate p_t
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
    
        # Calculate expected deaths mu_t (convolution)
        mu_t = pm.Deterministic('mu_t', pm.math.dot(fc_mat, p_t))
        
        # Likelihood
        pm.Poisson('deaths', mu=mu_t, observed=d_t)

    # --- 3. Inference ---
    try:
        with model:
            # Use ADVI for fast approximation with default optimizer
            # Fixed at 100,000 iterations for all cases
            n_iter = 100000
            approx = pm.fit(n_iter, method=pm.ADVI(random_seed=2025), 
                           progressbar=True)
            
        # Draw samples from the approximated posterior distribution
        idata = approx.sample(draws=n_draws, random_seed=2025)
        
        # --- 4. Extract and Return Results ---
        BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
        CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
        lambda_draws = idata.posterior['lambda'].values.flatten()
        p_samples = idata.posterior['p_t'].values.reshape(-1, T)
        # ADVI approx.sample() may omit Deterministics; compute mu_t from p_t if missing
        if 'mu_t' in idata.posterior:
            mu_samples = idata.posterior['mu_t'].values.reshape(-1, T)
        else:
            mu_samples = (fc_mat @ p_samples.T).T  # (n_draws, T), same as model: mu_t = fc_mat @ p_t
        
        # Check for NaN in results
        if not np.all(np.isfinite(BrtaCFR_est)):
            raise ValueError("NaN in posterior estimates")
        
        # R4-8: CrI at 50%, 80%, 95% for curve-level coverage
        pt_cri = {
            0.50: (np.quantile(p_samples, 0.25, axis=0), np.quantile(p_samples, 0.75, axis=0)),
            0.80: (np.quantile(p_samples, 0.10, axis=0), np.quantile(p_samples, 0.90, axis=0)),
            0.95: (np.quantile(p_samples, 0.025, axis=0), np.quantile(p_samples, 0.975, axis=0)),
        }
        mut_cri = {
            0.50: (np.quantile(mu_samples, 0.25, axis=0), np.quantile(mu_samples, 0.75, axis=0)),
            0.80: (np.quantile(mu_samples, 0.10, axis=0), np.quantile(mu_samples, 0.90, axis=0)),
            0.95: (np.quantile(mu_samples, 0.025, axis=0), np.quantile(mu_samples, 0.975, axis=0)),
        }
        
        results = {
            'mean': BrtaCFR_est,
            'lower': CrI[0, :],
            'upper': CrI[1, :],
            'lambda_draws': lambda_draws,
            'pt_cri': pt_cri,
            'mut_cri': mut_cri,
        }
        
        return results
    
    except Exception as e:
        # Fallback to mCFR if optimization fails
        warnings.warn(f"ADVI optimization failed: {str(e)}. Using mCFR fallback.")
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        return {
            'mean': mCFR_result,
            'lower': mCFR_result * 0.5,
            'upper': mCFR_result * 1.5,
            'lambda_draws': None,
            'pt_cri': None,
            'mut_cri': None,
        }