# brtacfr_estimator.py

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import gamma

def logp(value, mu, lam, sigma, T):
    """
    Estimate the log-likelihood for the beta mixture prior (Normal+Fused LASSO)
    """
    fused_lasso_logp = -lam*pt.sum(pt.abs(pt.diff(value)))+(T-1)*pt.log(lam/2)
    normal_logp = -0.5*pt.sum((value-mu)**2)/sigma**2-0.5*T*pt.log(2*np.pi*sigma**2)
    return fused_lasso_logp + normal_logp

def mCFR_EST(c_t, d_t, f_k):
    """
    Estimate the modified case fatality rate
    """
    T = c_t.shape[0]
    c_t_d = np.array([np.sum(np.flip(f_k[:i])*c_t[:i]) for i in np.arange(1,T+1)])
    return np.array([np.sum(d_t[:i])/(np.sum(c_t_d[:i])+1e-10) for i in np.arange(1,T+1)])

def BrtaCFR_estimator(c_t, d_t, F_paras):
    """
    Estimates the Bayesian real-time adjusted Case Fatality Rate (BrtaCFR).

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

    # Inference
    with model:
        # Use ADVI for fast approximation
        approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)
        
    # Draw samples from the posterior
    idata = approx.sample(draws=1000, random_seed=2025)
    
    # Extract results
    BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
    CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
    
    results = {
        'mean': BrtaCFR_est,
        'lower': CrI[0, :],
        'upper': CrI[1, :]
    }
    
    return results