# brtacfr_estimator.py

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
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
    # Fused LASSO component: Penalizes large differences between adjacent time points
    fused_lasso_logp = -lam*pt.sum(pt.abs(pt.diff(value)))+(T-1)*pt.log(lam/2)

    # Normal component: Pulls the estimate towards the prior mean (logit-cCFR)
    normal_logp = -0.5*pt.sum((value-mu)**2)/sigma**2-0.5*T*pt.log(2*np.pi*sigma**2)
    
    return fused_lasso_logp + normal_logp

# =============================================================================
# Helper Function for mCFR Calculation
# =============================================================================

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

def BrtaCFR_estimator(c_t, d_t, F_paras):
    """
    Estimates the Bayesian real-time adjusted Case Fatality Rate (BrtaCFR).

    This function implements the Bayesian model described in the manuscript to
    estimate the time-varying case fatality rate, p(t).

    Args:
        c_t (np.array): Time series of daily confirmed cases.
        d_t (np.array): Time series of daily deaths.
        F_paras (tuple): Parameters (mean, shape) for the Gamma delay distribution.

    Returns:
        dict: A dictionary containing the posterior mean, 95% credible intervals,
              and the full posterior samples for the fatality rate p_t.
    """
    
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
    
    # --- 2. Bayesian Model for BrtaCFR ---
    with pm.Model() as model:
        # Priors
        # Prior mean for beta_t based on logit-transformed cCFR
        beta_tilde = np.log((cCFR_est+1e-10)/(1 - (cCFR_est+1e-10)))
        
        # Hyperprior for the smoothing parameter lambda
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        
        # Priors for beta_t
        beta = pm.CustomDist('beta',
                              beta_tilde, lambda_param, 5, T,
                              logp=logp,
                              size=T)
    
        # Transformation to get fatality rate p_t
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
    
        # Calculate expected deaths mu_t (convolution)
        mu_t = pm.math.dot(fc_mat, p_t)
        
        # Likelihood
        pm.Poisson('deaths', mu=mu_t, observed=d_t)

    # --- 3. Inference ---
    with model:
        # Use ADVI for fast approximation
        approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)
        
    # Draw samples from the approximated posterior distribution
    idata = approx.sample(draws=1000, random_seed=2025)
    
    # --- 4. Extract and Return Results ---
    BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
    CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
    
    results = {
        'mean': BrtaCFR_est,
        'lower': CrI[0, :],
        'upper': CrI[1, :]
    }
    
    return results