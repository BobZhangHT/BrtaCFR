#!/usr/bin/env python
# run_all_simulations.py

"""
Unified simulation framework for BrtaCFR manuscript.

This script integrates all simulation analyses with:
- Shared data generation (no redundant replications)
- Checkpoint support (resume from interruptions)
- Parallel computation (multi-core processing)
- Fast demo mode (quick verification)

Analyses included:
0. Main Analysis (Original manuscript simulation)
1. Simulation Table (Runtime, convergence, MAE, PPC)
2. Sensitivity Analysis (Gamma, sigma, distributions)
3. MCMC vs ADVI Comparison

Usage:
    python run_all_simulations.py                 # Full analysis
    python run_all_simulations.py --demo          # Quick demo (2 reps)
    python run_all_simulations.py --resume        # Resume from checkpoint
    python run_all_simulations.py --only main     # Run specific analysis

Author: BrtaCFR Team
Date: October 2025
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import gamma, weibull_min, lognorm
from scipy.special import gamma as gamma_func
from scipy.optimize import fsolve
import pymc as pm
import pytensor.tensor as pt
import arviz as az

# Import core methods
from methods import BrtaCFR_estimator, mCFR_EST, logp

# =============================================================================
# Configuration
# =============================================================================

# Default configuration for full analysis
DEFAULT_CONFIG = {
    'main_reps': 1000,           # Main analysis replications
    'sensitivity_reps': 100,      # Sensitivity analysis replications
    'mcmc_reps': 50,              # MCMC comparison replications
    'n_jobs': -1,                 # Parallel jobs (-1 = all cores)
    'checkpoint_dir': './checkpoints',
    'output_dir': './outputs',
}

# Demo mode configuration (fast check)
DEMO_CONFIG = {
    'main_reps': 2,
    'sensitivity_reps': 10,
    'mcmc_reps': 5,
    'n_jobs': -1,
    'checkpoint_dir': './checkpoints_demo',
    'output_dir': './outputs_demo',
}

# Scenario definitions
T_PERIOD = 200
DAYS = np.arange(1, T_PERIOD + 1)
CT = 3000 - 5 * np.abs(100 - DAYS)

SCENARIOS = {
    'A': {'name': 'Constant', 'pt': np.full(T_PERIOD, 0.034)},
    'B': {'name': 'Exponential Growth', 'pt': 0.01 * np.exp(0.012 * DAYS)},
    'C': {'name': 'Delayed Growth', 'pt': 0.04 * np.exp(0.016 * np.where(DAYS > 60, np.minimum(40, DAYS - 60), 0))},
    'D': {'name': 'Decay', 'pt': 0.1 * np.exp(-0.009 * np.where(DAYS > 70, DAYS - 70, 0))},
    'E': {'name': 'Peak', 'pt': 0.1 * np.exp(-0.015 * np.abs(DAYS - 80))},
    'F': {'name': 'Valley', 'pt': 0.015 * np.exp(0.018 * np.abs(DAYS - 120))}
}

# Delay distribution parameters
TRUE_GAMMA_MEAN = 15.43
TRUE_GAMMA_SHAPE = 2.03

# Sensitivity cases
GAMMA_SENSITIVITY = {
    'True': {'mean': 15.43, 'shape': 2.03},
    'Mean+20%': {'mean': 15.43 * 1.2, 'shape': 2.03},
    'Mean-20%': {'mean': 15.43 * 0.8, 'shape': 2.03},
    'Shape+50%': {'mean': 15.43, 'shape': 2.03 * 1.5},
    'Shape-50%': {'mean': 15.43, 'shape': 2.03 * 0.5},
}

SIGMA_SENSITIVITY = {
    'Sigma_5': 5,
    'Sigma_10': 10,
    'Sigma_20': 20,
    'Sigma_50': 50,
}

# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resumable computation."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, name, data):
        """Save checkpoint data."""
        filepath = self.checkpoint_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"  💾 Checkpoint saved: {name}")
    
    def load(self, name):
        """Load checkpoint data."""
        filepath = self.checkpoint_dir / f"{name}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None
    
    def exists(self, name):
        """Check if checkpoint exists."""
        filepath = self.checkpoint_dir / f"{name}.pkl"
        return filepath.exists()
    
    def clear(self, name=None):
        """Clear specific checkpoint or all checkpoints."""
        if name:
            filepath = self.checkpoint_dir / f"{name}.pkl"
            if filepath.exists():
                filepath.unlink()
        else:
            for f in self.checkpoint_dir.glob("*.pkl"):
                f.unlink()

# =============================================================================
# Data Generation (Shared across all analyses)
# =============================================================================

def generate_simulation_data(scenario_key, rep_idx, seed_offset=0):
    """
    Generate simulation data for a single replication.
    This data is shared across multiple analyses.
    
    Returns:
        dict with CT, dt, pt_true, and metadata
    """
    np.random.seed(rep_idx + seed_offset)
    
    pt_true = SCENARIOS[scenario_key]['pt']
    
    # Generate deaths with true Gamma distribution
    mean_delay, shape_delay = TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE
    scale_delay = mean_delay / shape_delay
    F_k = gamma.cdf(np.arange(T_PERIOD + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    
    true_deaths_no_delay = np.random.binomial(CT.astype(int), pt_true)
    dt = np.array([np.sum(np.flip(f_k[:i]) * true_deaths_no_delay[:i]) 
                   for i in np.arange(1, T_PERIOD + 1)])
    
    return {
        'CT': CT,
        'dt': dt,
        'pt_true': pt_true,
        'scenario': scenario_key,
        'rep_idx': rep_idx,
        'seed': rep_idx + seed_offset
    }

# =============================================================================
# Analysis 0: Main Analysis (Original Manuscript)
# =============================================================================

def run_main_analysis_single(data, include_diagnostics=True):
    """
    Run main analysis for a single replication.
    If include_diagnostics=True, collect data for simulation table.
    """
    CT, dt, pt_true = data['CT'], data['dt'], data['pt_true']
    
    # Calculate delay distribution
    mean_delay, shape_delay = TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE
    scale_delay = mean_delay / shape_delay
    F_k = gamma.cdf(np.arange(T_PERIOD + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    
    # cCFR
    cCFR = np.cumsum(dt) / (np.cumsum(CT) + 1e-10)
    
    # mCFR
    mCFR = mCFR_EST(CT, dt, f_k)
    
    # BrtaCFR with diagnostics if requested
    if include_diagnostics:
        start_time = time.time()
        brta_results = run_brtacfr_with_diagnostics(CT, dt, (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE))
        runtime = time.time() - start_time
        
        # Compute metrics
        mae = np.mean(np.abs(brta_results['mean'] - pt_true))
        coverage = np.mean((pt_true >= brta_results['lower']) & (pt_true <= brta_results['upper']))
        
        # Posterior predictive check
        ppc = posterior_predictive_check(dt, brta_results['mu_samples'])
        
        return {
            'cCFR': cCFR,
            'mCFR': mCFR,
            'BrtaCFR_mean': brta_results['mean'],
            'BrtaCFR_lower': brta_results['lower'],
            'BrtaCFR_upper': brta_results['upper'],
            # Diagnostic data
            'runtime': runtime,
            'ess': brta_results['ess'],
            'mcse': brta_results['mcse'],
            'mae': mae,
            'coverage': coverage,
            'ppp_total': ppc['ppp_total'],
            'ppp_chi2': ppc['ppp_chi2'],
        }
    else:
        brta_results = BrtaCFR_estimator(CT, dt, (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE))
        return {
            'cCFR': cCFR,
            'mCFR': mCFR,
            'BrtaCFR_mean': brta_results['mean'],
            'BrtaCFR_lower': brta_results['lower'],
            'BrtaCFR_upper': brta_results['upper'],
        }

def run_brtacfr_with_diagnostics(c_t, d_t, F_paras):
    """BrtaCFR with full diagnostics for simulation table."""
    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay
    
    F_k = gamma.cdf(np.arange(T+1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    
    f_mat = np.zeros((T, T))
    for i in range(T):
        f_mat += np.diag(np.ones(T-i) * f_k[i], -i)
    c_mat = np.diag(c_t)
    fc_mat = np.dot(f_mat, c_mat)
    
    cCFR_est = np.cumsum(d_t) / (np.cumsum(c_t) + 1e-10)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.Deterministic('mu_t', pm.math.dot(fc_mat, p_t))
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    with model:
        approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)
    
    idata = approx.sample(draws=1000, random_seed=2025)
    
    BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
    CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
    mu_samples = idata.posterior['mu_t'].values.reshape(-1, T)
    
    try:
        ess = az.ess(idata, var_names=['p_t'])['p_t'].mean().values
        mcse = az.mcse(idata, var_names=['p_t'])['p_t'].mean().values
    except:
        ess, mcse = None, None
    
    return {
        'mean': BrtaCFR_est,
        'lower': CrI[0, :],
        'upper': CrI[1, :],
        'mu_samples': mu_samples,
        'ess': ess,
        'mcse': mcse,
    }

def posterior_predictive_check(observed_deaths, mu_samples):
    """Perform posterior predictive check."""
    pred_deaths = np.random.poisson(mu_samples)
    obs_stat = np.sum(observed_deaths)
    pred_stats = np.sum(pred_deaths, axis=1)
    
    ppp = np.mean(pred_stats >= obs_stat)
    if ppp > 0.5:
        ppp = 1 - ppp
    ppp = 2 * ppp
    
    pred_mean = np.mean(pred_deaths, axis=0)
    pred_std = np.std(pred_deaths, axis=0) + 1e-10
    obs_chi2 = np.sum(((observed_deaths - pred_mean) / pred_std) ** 2)
    
    pred_chi2 = np.array([np.sum(((pred_deaths[i] - pred_mean) / pred_std) ** 2) 
                          for i in range(len(pred_deaths))])
    
    ppp_chi2 = np.mean(pred_chi2 >= obs_chi2)
    if ppp_chi2 > 0.5:
        ppp_chi2 = 1 - ppp_chi2
    ppp_chi2 = 2 * ppp_chi2
    
    return {'ppp_total': ppp, 'ppp_chi2': ppp_chi2}

def run_main_analysis(config, checkpoint_mgr, resume=False):
    """Run main analysis for all scenarios."""
    print("\n" + "="*80)
    print("ANALYSIS 0: MAIN ANALYSIS (Original Manuscript)")
    print("="*80)
    
    checkpoint_name = 'main_analysis'
    
    if resume and checkpoint_mgr.exists(checkpoint_name):
        print("  📂 Loading from checkpoint...")
        return checkpoint_mgr.load(checkpoint_name)
    
    n_reps = config['main_reps']
    print(f"  Replications: {n_reps}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Collecting diagnostic data: Yes")
    
    all_results = {}
    
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Processing Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        
        # Generate all data first (can be checkpointed separately)
        data_checkpoint = f'data_main_{scenario_key}'
        if resume and checkpoint_mgr.exists(data_checkpoint):
            print(f"    📂 Loading data from checkpoint...")
            all_data = checkpoint_mgr.load(data_checkpoint)
        else:
            print(f"    Generating {n_reps} replications...")
            all_data = Parallel(n_jobs=config['n_jobs'])(
                delayed(generate_simulation_data)(scenario_key, i, seed_offset=0)
                for i in tqdm(range(n_reps), desc=f"    Data gen")
            )
            checkpoint_mgr.save(data_checkpoint, all_data)
        
        # Run analysis on all data
        print(f"    Running analysis...")
        results = Parallel(n_jobs=config['n_jobs'])(
            delayed(run_main_analysis_single)(data, include_diagnostics=True)
            for data in tqdm(all_data, desc=f"    Analysis")
        )
        
        # Aggregate
        all_results[scenario_key] = {
            'cCFR_avg': np.mean([r['cCFR'] for r in results], axis=0),
            'mCFR_avg': np.mean([r['mCFR'] for r in results], axis=0),
            'BrtaCFR_avg': np.mean([r['BrtaCFR_mean'] for r in results], axis=0),
            'BrtaCFR_lower_avg': np.mean([r['BrtaCFR_lower'] for r in results], axis=0),
            'BrtaCFR_upper_avg': np.mean([r['BrtaCFR_upper'] for r in results], axis=0),
            # Diagnostic data for simulation table
            'runtime_mean': np.mean([r['runtime'] for r in results]),
            'runtime_std': np.std([r['runtime'] for r in results]),
            'ess_values': [r['ess'] for r in results if r['ess'] is not None],
            'mcse_values': [r['mcse'] for r in results if r['mcse'] is not None],
            'mae_values': [r['mae'] for r in results],
            'coverage_values': [r['coverage'] for r in results],
            'ppp_total_values': [r['ppp_total'] for r in results],
            'ppp_chi2_values': [r['ppp_chi2'] for r in results],
        }
    
    checkpoint_mgr.save(checkpoint_name, all_results)
    print("\n  ✅ Main analysis complete!")
    
    return all_results

# =============================================================================
# Analysis 1: Simulation Table (Uses data from main analysis)
# =============================================================================

def generate_simulation_table(main_results, output_dir):
    """Generate simulation table from main analysis results."""
    print("\n" + "="*80)
    print("ANALYSIS 1: SIMULATION TABLE")
    print("="*80)
    print("  Using diagnostic data from main analysis...")
    
    table_data = []
    
    for scenario_key, results in main_results.items():
        row = {
            'Scenario': f"{scenario_key}: {SCENARIOS[scenario_key]['name']}",
            'Runtime_Mean': results['runtime_mean'],
            'Runtime_SD': results['runtime_std'],
            'ESS_Mean': np.mean(results['ess_values']) if results['ess_values'] else None,
            'ESS_SD': np.std(results['ess_values']) if results['ess_values'] else None,
            'MCSE_Mean': np.mean(results['mcse_values']) if results['mcse_values'] else None,
            'MCSE_SD': np.std(results['mcse_values']) if results['mcse_values'] else None,
            'MAE_Mean': np.mean(results['mae_values']),
            'MAE_SD': np.std(results['mae_values']),
            'Coverage_Mean': np.mean(results['coverage_values']),
            'Coverage_SD': np.std(results['coverage_values']),
            'PPP_Total_Mean': np.mean(results['ppp_total_values']),
            'PPP_Chi2_Mean': np.mean(results['ppp_chi2_values']),
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = Path(output_dir) / 'simulation_table_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved to: {csv_path}")
    
    # Save LaTeX
    latex_path = Path(output_dir) / 'simulation_table_latex.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"  ✅ Saved to: {latex_path}")
    
    return df

# =============================================================================
# Analysis 2: Sensitivity Analysis
# =============================================================================

def run_sensitivity_gamma(config, checkpoint_mgr, resume=False):
    """Sensitivity to Gamma parameters."""
    print("\n" + "="*80)
    print("ANALYSIS 2a: SENSITIVITY - Gamma Parameters")
    print("="*80)
    
    checkpoint_name = 'sensitivity_gamma'
    if resume and checkpoint_mgr.exists(checkpoint_name):
        print("  📂 Loading from checkpoint...")
        return checkpoint_mgr.load(checkpoint_name)
    
    n_reps = config['sensitivity_reps']
    scenario_key = 'B'  # Use scenario B for sensitivity
    results = {}
    
    for case_name, case_params in GAMMA_SENSITIVITY.items():
        print(f"\n  Case: {case_name}")
        
        # Generate data with true distribution
        data_list = [generate_simulation_data(scenario_key, i, seed_offset=10000) 
                     for i in range(n_reps)]
        
        # Estimate with potentially misspecified parameters
        F_paras = (case_params['mean'], case_params['shape'])
        estimates = Parallel(n_jobs=config['n_jobs'])(
            delayed(BrtaCFR_estimator)(d['CT'], d['dt'], F_paras)
            for d in tqdm(data_list, desc=f"    {case_name}")
        )
        
        pt_true = SCENARIOS[scenario_key]['pt']
        maes = [np.mean(np.abs(est['mean'] - pt_true)) for est in estimates]
        
        results[case_name] = {
            'mean_estimate': np.mean([est['mean'] for est in estimates], axis=0),
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes),
        }
    
    checkpoint_mgr.save(checkpoint_name, results)
    print("  ✅ Gamma sensitivity complete!")
    return results

def BrtaCFR_estimator_custom_sigma(c_t, d_t, F_paras, sigma_prior):
    """BrtaCFR estimator with custom prior variance."""
    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay
    
    F_k = gamma.cdf(np.arange(T+1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    
    f_mat = np.zeros((T, T))
    for i in range(T):
        f_mat += np.diag(np.ones(T-i) * f_k[i], -i)
    c_mat = np.diag(c_t)
    fc_mat = np.dot(f_mat, c_mat)
    
    cCFR_est = np.cumsum(d_t) / (np.cumsum(c_t) + 1e-10)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, sigma_prior, T, 
                            logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    with model:
        approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)
    
    idata = approx.sample(draws=1000, random_seed=2025)
    BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
    CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
    
    return {'mean': BrtaCFR_est, 'lower': CrI[0, :], 'upper': CrI[1, :]}

def run_sensitivity_sigma(config, checkpoint_mgr, resume=False):
    """Sensitivity to prior variance sigma."""
    print("\n" + "="*80)
    print("ANALYSIS 2b: SENSITIVITY - Prior Variance σ²")
    print("="*80)
    
    checkpoint_name = 'sensitivity_sigma'
    if resume and checkpoint_mgr.exists(checkpoint_name):
        print("  📂 Loading from checkpoint...")
        return checkpoint_mgr.load(checkpoint_name)
    
    n_reps = config['sensitivity_reps']
    scenario_key = 'B'
    results = {}
    
    for case_name, sigma_val in SIGMA_SENSITIVITY.items():
        print(f"\n  Case: {case_name} (σ={sigma_val})")
        
        data_list = [generate_simulation_data(scenario_key, i, seed_offset=20000) 
                     for i in range(n_reps)]
        
        F_paras = (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
        estimates = Parallel(n_jobs=config['n_jobs'])(
            delayed(BrtaCFR_estimator_custom_sigma)(d['CT'], d['dt'], F_paras, sigma_val)
            for d in tqdm(data_list, desc=f"    {case_name}")
        )
        
        pt_true = SCENARIOS[scenario_key]['pt']
        maes = [np.mean(np.abs(est['mean'] - pt_true)) for est in estimates]
        
        results[case_name] = {
            'mean_estimate': np.mean([est['mean'] for est in estimates], axis=0),
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes),
        }
    
    checkpoint_mgr.save(checkpoint_name, results)
    print("  ✅ Sigma sensitivity complete!")
    return results

def get_weibull_pmf(mean, variance, T):
    """Compute Weibull PMF with moment matching."""
    from scipy.optimize import fsolve
    
    def equations(params):
        k, lam = params
        eq1 = lam * gamma_func(1 + 1/k) - mean
        eq2 = lam**2 * (gamma_func(1 + 2/k) - (gamma_func(1 + 1/k))**2) - variance
        return [eq1, eq2]
    
    try:
        k, lam = fsolve(equations, [2.0, mean])
        if k <= 0 or lam <= 0:
            raise ValueError
        cdf_values = weibull_min.cdf(np.arange(T+1), c=k, scale=lam)
        return np.diff(cdf_values)
    except:
        k = (mean / np.sqrt(variance))**2
        lam = mean / gamma_func(1 + 1/k)
        cdf_values = weibull_min.cdf(np.arange(T+1), c=k, scale=lam)
        return np.diff(cdf_values)

def get_lognormal_pmf(mean, variance, T):
    """Compute Lognormal PMF with moment matching."""
    mu = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    cdf_values = lognorm.cdf(np.arange(T+1), s=sigma, scale=np.exp(mu))
    return np.diff(cdf_values)

def estimate_with_custom_pmf(c_t, d_t, f_k):
    """BrtaCFR with custom delay distribution PMF."""
    T = len(c_t)
    
    f_mat = np.zeros((T, T))
    for i in range(T):
        if i < len(f_k):
            f_mat += np.diag(np.ones(T-i) * f_k[i], -i)
    c_mat = np.diag(c_t)
    fc_mat = np.dot(f_mat, c_mat)
    
    cCFR_est = np.cumsum(d_t) / (np.cumsum(c_t) + 1e-10)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    with model:
        approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)
    
    idata = approx.sample(draws=1000, random_seed=2025)
    BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
    CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
    
    return {'mean': BrtaCFR_est, 'lower': CrI[0, :], 'upper': CrI[1, :]}

def run_sensitivity_distribution(config, checkpoint_mgr, resume=False):
    """Sensitivity to different delay distributions."""
    print("\n" + "="*80)
    print("ANALYSIS 2c: SENSITIVITY - Delay Distributions")
    print("="*80)
    
    checkpoint_name = 'sensitivity_dist'
    if resume and checkpoint_mgr.exists(checkpoint_name):
        print("  📂 Loading from checkpoint...")
        return checkpoint_mgr.load(checkpoint_name)
    
    n_reps = config['sensitivity_reps']
    scenario_key = 'B'
    
    # Calculate true mean and variance
    true_scale = TRUE_GAMMA_MEAN / TRUE_GAMMA_SHAPE
    true_variance = TRUE_GAMMA_SHAPE * (true_scale ** 2)
    
    dist_cases = {
        'Gamma': {'type': 'gamma'},
        'Weibull': {'type': 'weibull'},
        'Lognormal': {'type': 'lognormal'},
    }
    
    results = {}
    
    for case_name, case_info in dist_cases.items():
        print(f"\n  Case: {case_name}")
        
        data_list = [generate_simulation_data(scenario_key, i, seed_offset=30000) 
                     for i in range(n_reps)]
        
        if case_info['type'] == 'gamma':
            F_paras = (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
            estimates = Parallel(n_jobs=config['n_jobs'])(
                delayed(BrtaCFR_estimator)(d['CT'], d['dt'], F_paras)
                for d in tqdm(data_list, desc=f"    {case_name}")
            )
        else:
            # Generate PMF
            if case_info['type'] == 'weibull':
                pmf = get_weibull_pmf(TRUE_GAMMA_MEAN, true_variance, T_PERIOD)
            else:  # lognormal
                pmf = get_lognormal_pmf(TRUE_GAMMA_MEAN, true_variance, T_PERIOD)
            
            estimates = Parallel(n_jobs=config['n_jobs'])(
                delayed(estimate_with_custom_pmf)(d['CT'], d['dt'], pmf)
                for d in tqdm(data_list, desc=f"    {case_name}")
            )
        
        pt_true = SCENARIOS[scenario_key]['pt']
        maes = [np.mean(np.abs(est['mean'] - pt_true)) for est in estimates]
        
        results[case_name] = {
            'mean_estimate': np.mean([est['mean'] for est in estimates], axis=0),
            'mae_mean': np.mean(maes),
            'mae_std': np.std(maes),
        }
    
    checkpoint_mgr.save(checkpoint_name, results)
    print("  ✅ Distribution sensitivity complete!")
    return results

# =============================================================================
# Analysis 3: MCMC vs ADVI Comparison
# =============================================================================

def run_brtacfr_mcmc(c_t, d_t, F_paras, n_samples=500, n_chains=2, tune=500):
    """Run BrtaCFR with MCMC (NUTS sampler)."""
    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay
    
    F_k = gamma.cdf(np.arange(T+1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    
    f_mat = np.zeros((T, T))
    for i in range(T):
        f_mat += np.diag(np.ones(T-i) * f_k[i], -i)
    c_mat = np.diag(c_t)
    fc_mat = np.dot(f_mat, c_mat)
    
    cCFR_est = np.cumsum(d_t) / (np.cumsum(c_t) + 1e-10)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    with model:
        idata = pm.sample(draws=n_samples, tune=tune, chains=n_chains,
                         random_seed=2025, progressbar=False, return_inferencedata=True)
    
    BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
    CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
    
    try:
        rhat = az.rhat(idata, var_names=['p_t'])['p_t'].mean().values
        ess_bulk = az.ess(idata, var_names=['p_t'], method='bulk')['p_t'].mean().values
        n_divergences = idata.sample_stats['diverging'].sum().values
    except:
        rhat, ess_bulk, n_divergences = None, None, None
    
    return {
        'mean': BrtaCFR_est,
        'lower': CrI[0, :],
        'upper': CrI[1, :],
        'rhat': rhat,
        'ess_bulk': ess_bulk,
        'n_divergences': n_divergences,
    }

def run_mcmc_comparison(config, checkpoint_mgr, resume=False):
    """MCMC vs ADVI comparison."""
    print("\n" + "="*80)
    print("ANALYSIS 3: MCMC vs ADVI Comparison")
    print("="*80)
    
    checkpoint_name = 'mcmc_comparison'
    if resume and checkpoint_mgr.exists(checkpoint_name):
        print("  📂 Loading from checkpoint...")
        return checkpoint_mgr.load(checkpoint_name)
    
    n_reps = config['mcmc_reps']
    # Use subset of scenarios for speed
    test_scenarios = ['A', 'B', 'E']
    
    comparison_results = {}
    
    for scenario_key in test_scenarios:
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        
        # Generate data
        data_list = [generate_simulation_data(scenario_key, i, seed_offset=40000) 
                     for i in range(n_reps)]
        pt_true = SCENARIOS[scenario_key]['pt']
        
        # Run MCMC
        print(f"    Running MCMC...")
        mcmc_start = time.time()
        mcmc_results = []
        for data in tqdm(data_list, desc="    MCMC"):
            try:
                result = run_brtacfr_mcmc(data['CT'], data['dt'], 
                                         (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE))
                mcmc_results.append({
                    'success': True,
                    'runtime': time.time() - mcmc_start,
                    'mae': np.mean(np.abs(result['mean'] - pt_true)),
                    'coverage': np.mean((pt_true >= result['lower']) & (pt_true <= result['upper'])),
                })
                mcmc_start = time.time()
            except Exception as e:
                mcmc_results.append({'success': False})
        
        # Run ADVI
        print(f"    Running ADVI...")
        advi_start = time.time()
        advi_results = []
        for data in tqdm(data_list, desc="    ADVI"):
            try:
                result = BrtaCFR_estimator(data['CT'], data['dt'], 
                                          (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE))
                advi_results.append({
                    'success': True,
                    'runtime': time.time() - advi_start,
                    'mae': np.mean(np.abs(result['mean'] - pt_true)),
                    'coverage': np.mean((pt_true >= result['lower']) & (pt_true <= result['upper'])),
                })
                advi_start = time.time()
            except Exception as e:
                advi_results.append({'success': False})
        
        # Aggregate
        mcmc_success = [r for r in mcmc_results if r['success']]
        advi_success = [r for r in advi_results if r['success']]
        
        comparison_results[scenario_key] = {
            'mcmc_runtime_mean': np.mean([r['runtime'] for r in mcmc_success]) if mcmc_success else None,
            'mcmc_runtime_std': np.std([r['runtime'] for r in mcmc_success]) if mcmc_success else None,
            'advi_runtime_mean': np.mean([r['runtime'] for r in advi_success]) if advi_success else None,
            'advi_runtime_std': np.std([r['runtime'] for r in advi_success]) if advi_success else None,
            'speedup': np.mean([r['runtime'] for r in mcmc_success]) / np.mean([r['runtime'] for r in advi_success]) if (mcmc_success and advi_success) else None,
            'mcmc_mae_mean': np.mean([r['mae'] for r in mcmc_success]) if mcmc_success else None,
            'advi_mae_mean': np.mean([r['mae'] for r in advi_success]) if advi_success else None,
            'mcmc_coverage_mean': np.mean([r['coverage'] for r in mcmc_success]) if mcmc_success else None,
            'advi_coverage_mean': np.mean([r['coverage'] for r in advi_success]) if advi_success else None,
        }
    
    checkpoint_mgr.save(checkpoint_name, comparison_results)
    print("  ✅ MCMC comparison complete!")
    return comparison_results

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_main_analysis(main_results, output_dir):
    """Generate main analysis plots."""
    print("\n  Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    
    for i, (scenario_key, results) in enumerate(main_results.items()):
        ax = axes[i]
        pt_true = SCENARIOS[scenario_key]['pt']
        
        ax.plot(DAYS, pt_true, 'k-', linewidth=2.5, label='True')
        ax.plot(DAYS, results['cCFR_avg'], 'g-', label='cCFR')
        ax.plot(DAYS, results['mCFR_avg'], color='orange', label='mCFR')
        ax.plot(DAYS, results['BrtaCFR_avg'], 'r--', linewidth=2, label='BrtaCFR')
        ax.fill_between(DAYS, results['BrtaCFR_lower_avg'], results['BrtaCFR_upper_avg'], 
                        color='blue', alpha=0.2, label='95% CrI')
        
        ax.set_title(f"({scenario_key}) {SCENARIOS[scenario_key]['name']}", fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend()
    
    fig.text(0.5, 0.04, 'Days', ha='center', fontsize=14)
    fig.text(0.08, 0.5, 'Fatality Rate', ha='center', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'simulation.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()

def plot_sensitivity_results(gamma_results, sigma_results, dist_results, output_dir):
    """Plot all sensitivity analysis results."""
    print("\n  Generating sensitivity plots...")
    
    # Plot 1: Gamma sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(DAYS, SCENARIOS['B']['pt'], 'k-', linewidth=2.5, label='True')
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for (case_name, results), color in zip(gamma_results.items(), colors):
        ax1.plot(DAYS, results['mean_estimate'], label=case_name.replace('_', ' '), 
                color=color, linestyle='--')
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Fatality Rate', fontsize=12)
    ax1.set_title('(A) Gamma Parameter Misspecification', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    case_names = [n.replace('_', ' ') for n in gamma_results.keys()]
    mae_means = [r['mae_mean'] for r in gamma_results.values()]
    mae_stds = [r['mae_std'] for r in gamma_results.values()]
    ax2.bar(case_names, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('(B) MAE Comparison', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'sensitivity_gamma.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Plot 2: Sigma sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(DAYS, SCENARIOS['B']['pt'], 'k-', linewidth=2.5, label='True')
    colors = ['red', 'blue', 'green', 'orange']
    for (case_name, results), color in zip(sigma_results.items(), colors):
        sigma_val = SIGMA_SENSITIVITY[case_name]
        ax1.plot(DAYS, results['mean_estimate'], label=f"σ={sigma_val}", 
                color=color, linestyle='--')
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Fatality Rate', fontsize=12)
    ax1.set_title('(A) Prior σ² Sensitivity', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    sigma_labels = [f"σ={SIGMA_SENSITIVITY[n]}" for n in sigma_results.keys()]
    mae_means = [r['mae_mean'] for r in sigma_results.values()]
    mae_stds = [r['mae_std'] for r in sigma_results.values()]
    ax2.bar(sigma_labels, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('(B) MAE Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'sensitivity_sigma.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Plot 3: Distribution sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(DAYS, SCENARIOS['B']['pt'], 'k-', linewidth=2.5, label='True')
    colors = ['red', 'blue', 'green']
    for (case_name, results), color in zip(dist_results.items(), colors):
        ax1.plot(DAYS, results['mean_estimate'], label=case_name, 
                color=color, linestyle='--')
    ax1.set_xlabel('Days', fontsize=12)
    ax1.set_ylabel('Fatality Rate', fontsize=12)
    ax1.set_title('(A) Distribution Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    dist_names = list(dist_results.keys())
    mae_means = [r['mae_mean'] for r in dist_results.values()]
    mae_stds = [r['mae_std'] for r in dist_results.values()]
    ax2.bar(dist_names, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
    ax2.set_ylabel('Mean Absolute Error', fontsize=12)
    ax2.set_title('(B) MAE Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'sensitivity_distributions.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Save summary CSV
    summary_data = []
    for case_name, results in gamma_results.items():
        summary_data.append({
            'Analysis': 'Gamma',
            'Case': case_name,
            'MAE_Mean': results['mae_mean'],
            'MAE_SD': results['mae_std']
        })
    for case_name, results in sigma_results.items():
        summary_data.append({
            'Analysis': 'Sigma',
            'Case': case_name,
            'MAE_Mean': results['mae_mean'],
            'MAE_SD': results['mae_std']
        })
    for case_name, results in dist_results.items():
        summary_data.append({
            'Analysis': 'Distribution',
            'Case': case_name,
            'MAE_Mean': results['mae_mean'],
            'MAE_SD': results['mae_std']
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = Path(output_dir) / 'sensitivity_analysis_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved to: {csv_path}")

def plot_mcmc_comparison(mcmc_results, output_dir):
    """Plot MCMC vs ADVI comparison."""
    print("\n  Generating MCMC comparison plots...")
    
    scenarios = list(mcmc_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Runtime
    ax1 = axes[0, 0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    mcmc_times = [mcmc_results[s]['mcmc_runtime_mean'] for s in scenarios]
    advi_times = [mcmc_results[s]['advi_runtime_mean'] for s in scenarios]
    mcmc_errs = [mcmc_results[s]['mcmc_runtime_std'] for s in scenarios]
    advi_errs = [mcmc_results[s]['advi_runtime_std'] for s in scenarios]
    
    ax1.bar(x - width/2, mcmc_times, width, yerr=mcmc_errs, label='MCMC', alpha=0.8, capsize=5)
    ax1.bar(x + width/2, advi_times, width, yerr=advi_errs, label='ADVI', alpha=0.8, capsize=5)
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('(A) Runtime Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup
    ax2 = axes[0, 1]
    speedups = [mcmc_results[s]['speedup'] for s in scenarios]
    ax2.bar(scenarios, speedups, alpha=0.8, color='green')
    ax2.axhline(y=1, color='r', linestyle='--', label='No speedup')
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('(B) ADVI Speedup', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(speedups):
        ax2.text(i, v + 0.5, f'{v:.1f}x', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: MAE comparison
    ax3 = axes[1, 0]
    mcmc_maes = [mcmc_results[s]['mcmc_mae_mean'] for s in scenarios]
    advi_maes = [mcmc_results[s]['advi_mae_mean'] for s in scenarios]
    ax3.bar(x - width/2, mcmc_maes, width, label='MCMC', alpha=0.8)
    ax3.bar(x + width/2, advi_maes, width, label='ADVI', alpha=0.8)
    ax3.set_xlabel('Scenario', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('(C) Accuracy Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Coverage comparison
    ax4 = axes[1, 1]
    mcmc_cov = [mcmc_results[s]['mcmc_coverage_mean'] for s in scenarios]
    advi_cov = [mcmc_results[s]['advi_coverage_mean'] for s in scenarios]
    ax4.bar(x - width/2, mcmc_cov, width, label='MCMC', alpha=0.8)
    ax4.bar(x + width/2, advi_cov, width, label='ADVI', alpha=0.8)
    ax4.axhline(y=0.95, color='r', linestyle='--', label='Nominal 95%')
    ax4.set_xlabel('Scenario', fontsize=12)
    ax4.set_ylabel('Coverage Rate', fontsize=12)
    ax4.set_title('(D) Coverage Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios)
    ax4.set_ylim([0.85, 1.0])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'mcmc_vs_advi_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Save CSV
    comparison_data = []
    for scenario in scenarios:
        comparison_data.append({
            'Scenario': f"{scenario}: {SCENARIOS[scenario]['name']}",
            'MCMC_Runtime_Mean': mcmc_results[scenario]['mcmc_runtime_mean'],
            'ADVI_Runtime_Mean': mcmc_results[scenario]['advi_runtime_mean'],
            'Speedup': mcmc_results[scenario]['speedup'],
            'MCMC_MAE': mcmc_results[scenario]['mcmc_mae_mean'],
            'ADVI_MAE': mcmc_results[scenario]['advi_mae_mean'],
            'MCMC_Coverage': mcmc_results[scenario]['mcmc_coverage_mean'],
            'ADVI_Coverage': mcmc_results[scenario]['advi_coverage_mean'],
        })
    
    df = pd.DataFrame(comparison_data)
    csv_path = Path(output_dir) / 'mcmc_vs_advi_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved to: {csv_path}")

# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified BrtaCFR simulation framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode (fast, 2 reps)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoints')
    parser.add_argument('--clear-checkpoints', action='store_true',
                       help='Clear all checkpoints before running')
    parser.add_argument('--only', choices=['main', 'sensitivity', 'mcmc', 'all'],
                       default='all', help='Run specific analysis')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Select configuration
    config = DEMO_CONFIG.copy() if args.demo else DEFAULT_CONFIG.copy()
    if args.n_jobs != -1:
        config['n_jobs'] = args.n_jobs
    
    # Setup directories
    checkpoint_dir = config['checkpoint_dir']
    output_dir = config['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    if args.clear_checkpoints:
        print("🗑️  Clearing all checkpoints...")
        checkpoint_mgr.clear()
    
    print("="*80)
    print("UNIFIED BRTACFR SIMULATION FRAMEWORK")
    print("="*80)
    print(f"Mode: {'DEMO' if args.demo else 'FULL'}")
    print(f"Replications: main={config['main_reps']}, sensitivity={config['sensitivity_reps']}, mcmc={config['mcmc_reps']}")
    print(f"Parallel jobs: {config['n_jobs']}")
    print(f"Resume: {args.resume}")
    print(f"Output: {output_dir}")
    print("="*80)
    
    start_time = time.time()
    
    # Run analyses
    if args.only in ['main', 'all']:
        print("\n" + "🔬"*40)
        print("PHASE 1: MAIN ANALYSIS & SIMULATION TABLE")
        print("🔬"*40)
        main_results = run_main_analysis(config, checkpoint_mgr, args.resume)
        plot_main_analysis(main_results, output_dir)
        simulation_table = generate_simulation_table(main_results, output_dir)
    
    if args.only in ['sensitivity', 'all']:
        print("\n" + "🔬"*40)
        print("PHASE 2: SENSITIVITY ANALYSIS")
        print("🔬"*40)
        gamma_results = run_sensitivity_gamma(config, checkpoint_mgr, args.resume)
        sigma_results = run_sensitivity_sigma(config, checkpoint_mgr, args.resume)
        dist_results = run_sensitivity_distribution(config, checkpoint_mgr, args.resume)
        plot_sensitivity_results(gamma_results, sigma_results, dist_results, output_dir)
    
    if args.only in ['mcmc', 'all']:
        print("\n" + "🔬"*40)
        print("PHASE 3: MCMC VS ADVI COMPARISON")
        print("🔬"*40)
        mcmc_results = run_mcmc_comparison(config, checkpoint_mgr, args.resume)
        plot_mcmc_comparison(mcmc_results, output_dir)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("🎉 ALL ANALYSES COMPLETE! 🎉")
    print("="*80)
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Outputs saved to: {output_dir}")
    
    # List output files
    print("\nGenerated files:")
    output_path = Path(output_dir)
    for f in sorted(output_path.glob("*")):
        size = f.stat().st_size / 1024  # KB
        print(f"  ✅ {f.name} ({size:.1f} KB)")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("  1. Review all generated figures and tables")
    print("  2. Check REVIEWER_RESPONSE_SUMMARY.md for manuscript text")
    print("  3. Integrate results into manuscript")
    print("  4. Submit revision!")
    print("="*80)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        print("Partial results saved in checkpoints. Use --resume to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

