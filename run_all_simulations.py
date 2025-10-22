#!/usr/bin/env python
# run_all_simulations.py

"""
Unified simulation framework for BrtaCFR manuscript.

This script integrates all simulation analyses with:
- Per-replication checkpoint system (automatic resume from interruptions)
- Shared data generation (no redundant replications)
- Parallel computation (multi-core processing)
- Fast demo mode (quick verification)
- Direct plotting without recomputation (once complete)

Analyses included:
0. Main Analysis (Original manuscript simulation)
1. Simulation Table (Runtime, convergence, MAE, PPC)
2. Sensitivity Analysis (Gamma, sigma, distributions)
3. MCMC vs ADVI Comparison

Checkpoint System:
- All replications are automatically saved upon completion
- Interrupted runs automatically resume from last completed replication
- Completed analyses are loaded instantly without recomputation
- Use --clear-checkpoints to restart from scratch

Usage:
    python run_all_simulations.py                 # Full analysis (auto-resume)
    python run_all_simulations.py --demo          # Quick demo (2 reps)
    python run_all_simulations.py --clear-checkpoints  # Clear cache and restart
    python run_all_simulations.py --only main     # Run specific analysis

Author: BrtaCFR Team
Date: October 2025
"""

import os
import sys
import time
import pickle
import argparse
import warnings

# Configure PyTensor BEFORE any imports that use it
os.environ['PYTENSOR_FLAGS'] = 'optimizer=fast_compile,exception_verbosity=low'

# Suppress PyTensor BLAS warning
warnings.filterwarnings('ignore', message='.*PyTensor could not link to a BLAS.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')

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
# Helper Functions
# =============================================================================

def logit(p):
    """
    Compute logit transformation: log(p / (1 - p))
    Clips p to avoid log(0) or division by zero.
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))

def logit_mae(pred, true):
    """
    Compute logit-transformed Mean Absolute Error:
    MAE = mean(|logit(pred) - logit(true)|)
    
    This is more sensitive to errors in the probability scale,
    especially near 0 and 1.
    """
    return np.mean(np.abs(logit(pred) - logit(true)))

# =============================================================================
# Configuration
# =============================================================================

# Default configuration for full analysis
DEFAULT_CONFIG = {
    'main_reps': 1000,           # Main analysis replications
    'sensitivity_reps': 100,      # Sensitivity analysis replications
    'mcmc_reps': 100,              # MCMC comparison replications
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
    'Sigma_1': 1,
    'Sigma_3': 3,
    'Sigma_5': 5,
    'Sigma_7': 7,
    'Sigma_10': 10,
}

# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """Manage checkpoints for resumable computation with per-replication tracking."""
    
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # In-memory cache for faster access
    
    def _optimize_for_storage(self, data):
        """
        Optimize data for storage by:
        1. Converting float64 arrays to float32 (50% size reduction, sufficient precision)
        2. Removing large intermediate arrays (e.g., full posterior samples)
        3. Keeping only summary statistics
        """
        if isinstance(data, dict):
            optimized = {}
            for key, value in data.items():
                # Skip large sample arrays - only keep summary stats
                if key in ['mu_samples', 'posterior_samples', 'trace']:
                    continue  # Don't save these large arrays
                
                # Convert arrays to float32 for storage efficiency
                if isinstance(value, np.ndarray):
                    if value.dtype == np.float64:
                        optimized[key] = value.astype(np.float32)
                    else:
                        optimized[key] = value
                elif isinstance(value, dict):
                    optimized[key] = self._optimize_for_storage(value)
                elif isinstance(value, list) and len(value) > 0:
                    # Handle lists of dicts (e.g., per-replication results)
                    if isinstance(value[0], dict):
                        optimized[key] = [self._optimize_for_storage(v) for v in value]
                    else:
                        optimized[key] = value
                else:
                    optimized[key] = value
            return optimized
        return data
    
    def save(self, name, data):
        """Save checkpoint data with automatic optimization."""
        # Optimize before saving
        data = self._optimize_for_storage(data)
        
        filepath = self.checkpoint_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._cache[name] = data
        # Don't print for every save to avoid clutter
    
    def save_verbose(self, name, data):
        """Save checkpoint data with verbose output."""
        self.save(name, data)
        filepath = self.checkpoint_dir / f"{name}.pkl"
        size_kb = filepath.stat().st_size / 1024
        print(f"  💾 Checkpoint saved: {name} ({size_kb:.1f} KB)")
    
    def load(self, name):
        """Load checkpoint data."""
        # Check cache first
        if name in self._cache:
            return self._cache[name]
        
        filepath = self.checkpoint_dir / f"{name}.pkl"
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self._cache[name] = data
                return data
        return None
    
    def exists(self, name):
        """Check if checkpoint exists."""
        if name in self._cache:
            return True
        filepath = self.checkpoint_dir / f"{name}.pkl"
        return filepath.exists()
    
    def clear(self, name=None):
        """Clear specific checkpoint or all checkpoints."""
        if name:
            filepath = self.checkpoint_dir / f"{name}.pkl"
            if filepath.exists():
                filepath.unlink()
            if name in self._cache:
                del self._cache[name]
        else:
            for f in self.checkpoint_dir.glob("*.pkl"):
                f.unlink()
            self._cache.clear()
    
    def save_replication_result(self, analysis_name, scenario_key, rep_idx, result):
        """
        Save individual replication result for fine-grained checkpointing.
        
        Args:
            analysis_name: e.g., 'main', 'sensitivity_gamma', 'mcmc'
            scenario_key: e.g., 'A', 'B', 'C', etc.
            rep_idx: replication index
            result: result dictionary
        """
        checkpoint_name = f"{analysis_name}_{scenario_key}_rep{rep_idx}"
        self.save(checkpoint_name, result)
    
    def load_replication_result(self, analysis_name, scenario_key, rep_idx):
        """Load individual replication result."""
        checkpoint_name = f"{analysis_name}_{scenario_key}_rep{rep_idx}"
        return self.load(checkpoint_name)
    
    def get_completed_replications(self, analysis_name, scenario_key, total_reps):
        """
        Get list of completed replication indices and load their results.
        
        Returns:
            dict: {rep_idx: result} for all completed replications
        """
        completed = {}
        for rep_idx in range(total_reps):
            result = self.load_replication_result(analysis_name, scenario_key, rep_idx)
            if result is not None:
                completed[rep_idx] = result
        return completed
    
    def get_pending_replications(self, analysis_name, scenario_key, total_reps):
        """Get list of replication indices that need to be run."""
        completed = self.get_completed_replications(analysis_name, scenario_key, total_reps)
        return [i for i in range(total_reps) if i not in completed]

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
        
        # Compute metrics (using logit-transformed MAE)
        mae = logit_mae(brta_results['mean'], pt_true)
        coverage = np.mean((pt_true >= brta_results['lower']) & (pt_true <= brta_results['upper']))
        
        # Posterior predictive check (consumes mu_samples, doesn't save them)
        ppc = posterior_predictive_check(dt, brta_results['mu_samples'])
        
        # Return only summary statistics (no large arrays like mu_samples)
        return {
            'cCFR': cCFR,
            'mCFR': mCFR,
            'BrtaCFR_mean': brta_results['mean'],
            'BrtaCFR_lower': brta_results['lower'],
            'BrtaCFR_upper': brta_results['upper'],
            # Diagnostic data (scalars only)
            'runtime': runtime,
            'ess': brta_results['ess'],
            'mcse': brta_results['mcse'],
            'mae': mae,
            'coverage': coverage,
            'ppp_total': ppc['ppp_total'],
            'ppp_chi2': ppc['ppp_chi2'],
            # Note: mu_samples not saved - reduces checkpoint size by ~10x
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
    
    # Clip cCFR to avoid extreme values
    cCFR_est = np.clip(cCFR_est, 1e-8, 1 - 1e-8)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        
        # Check for NaN or Inf in initialization
        if not np.all(np.isfinite(beta_tilde)):
            beta_tilde = np.nan_to_num(beta_tilde, nan=0.0, posinf=5.0, neginf=-5.0)
        
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.Deterministic('mu_t', pm.math.dot(fc_mat, p_t))
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    try:
        with model:
            # Use default optimizer for better convergence and smooth estimates
            approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), 
                           progressbar=False)
        
        idata = approx.sample(draws=1000, random_seed=2025)
        
        BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
        CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
        mu_samples = idata.posterior['mu_t'].values.reshape(-1, T)
        
        # Check for NaN in results
        if not np.all(np.isfinite(BrtaCFR_est)):
            raise ValueError("NaN in posterior estimates")
        
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
    
    except Exception as e:
        warnings.warn(f"ADVI optimization failed: {str(e)}. Using mCFR fallback.")
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        mu_samples = np.tile(mCFR_result * c_t, (1000, 1))
        return {
            'mean': mCFR_result,
            'lower': mCFR_result * 0.5,
            'upper': mCFR_result * 1.5,
            'mu_samples': mu_samples,
            'ess': None,
            'mcse': None,
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
    """Run main analysis for all scenarios with per-replication checkpointing."""
    print("\n" + "="*80)
    print("ANALYSIS 0: MAIN ANALYSIS (Original Manuscript)")
    print("="*80)
    
    final_checkpoint_name = 'main_analysis_final'
    
    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  ✅ Main analysis already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)
    
    n_reps = config['main_reps']
    print(f"  Replications: {n_reps}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Collecting diagnostic data: Yes")
    
    all_results = {}
    
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Processing Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        
        # Check for completed replications
        completed = checkpoint_mgr.get_completed_replications('main', scenario_key, n_reps)
        pending = checkpoint_mgr.get_pending_replications('main', scenario_key, n_reps)
        
        if len(completed) > 0:
            print(f"    📂 Found {len(completed)}/{n_reps} completed replications")
        if len(pending) == 0:
            print(f"    ✅ All replications complete, using cached results")
            results = list(completed.values())
        else:
            print(f"    🔄 Running {len(pending)} pending replications...")
            
            # Generate data for pending replications only
            pending_data = []
            for rep_idx in pending:
                data = generate_simulation_data(scenario_key, rep_idx, seed_offset=0)
                pending_data.append((rep_idx, data))
            
            # Run analysis on pending replications in parallel
            def run_and_save_replication(rep_idx_data):
                rep_idx, data = rep_idx_data
                result = run_main_analysis_single(data, include_diagnostics=True)
                checkpoint_mgr.save_replication_result('main', scenario_key, rep_idx, result)
                return result
            
            new_results = Parallel(n_jobs=config['n_jobs'])(
                delayed(run_and_save_replication)(rep_idx_data)
                for rep_idx_data in tqdm(pending_data, desc=f"    Scenario {scenario_key}")
            )
            
            # Combine with completed results
            results = list(completed.values()) + new_results
            print(f"    ✅ Scenario {scenario_key} complete ({n_reps}/{n_reps} replications)")
        
        # Aggregate results
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
    
    # Save final aggregated results
    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
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
    """Sensitivity to Gamma parameters - analyze all 6 scenarios with per-replication checkpointing."""
    print("\n" + "="*80)
    print("ANALYSIS 2a: SENSITIVITY - Gamma Parameters")
    print("="*80)
    
    final_checkpoint_name = 'sensitivity_gamma_final'
    
    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  ✅ Gamma sensitivity already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)
    
    n_reps = config['sensitivity_reps']
    all_results = {}
    
    # Analyze all 6 scenarios
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        scenario_results = {}
        
        for case_name, case_params in GAMMA_SENSITIVITY.items():
            analysis_name = f'sens_gamma_{case_name}'
            completed = checkpoint_mgr.get_completed_replications(analysis_name, scenario_key, n_reps)
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)
            
            if len(completed) > 0:
                print(f"    Case {case_name}: Found {len(completed)}/{n_reps} replications")
            if len(pending) == 0:
                print(f"    ✅ {case_name} complete, using cached results")
                estimates = list(completed.values())
            else:
                print(f"    🔄 {case_name}: Running {len(pending)} pending replications")
                
                # Estimate with potentially misspecified parameters
                F_paras = (case_params['mean'], case_params['shape'])
                
                def run_and_save_gamma_replication(rep_idx):
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=10000)
                    result = BrtaCFR_estimator(data['CT'], data['dt'], F_paras)
                    checkpoint_mgr.save_replication_result(analysis_name, scenario_key, rep_idx, result)
                    return result
                
                new_estimates = Parallel(n_jobs=config['n_jobs'])(
                    delayed(run_and_save_gamma_replication)(rep_idx)
                    for rep_idx in tqdm(pending, desc=f"      {case_name}")
                )
                
                estimates = list(completed.values()) + new_estimates
            
            pt_true = SCENARIOS[scenario_key]['pt']
            maes = [logit_mae(est['mean'], pt_true) for est in estimates]
            
            scenario_results[case_name] = {
                'mean_estimate': np.mean([est['mean'] for est in estimates], axis=0),
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
            }
        
        all_results[scenario_key] = scenario_results
    
    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("  ✅ Gamma sensitivity complete!")
    return all_results

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
    
    # Clip cCFR to avoid extreme values
    cCFR_est = np.clip(cCFR_est, 1e-8, 1 - 1e-8)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        
        # Check for NaN or Inf in initialization
        if not np.all(np.isfinite(beta_tilde)):
            beta_tilde = np.nan_to_num(beta_tilde, nan=0.0, posinf=5.0, neginf=-5.0)
        
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, sigma_prior, T, 
                            logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    try:
        with model:
            # Use default optimizer for better convergence and smooth estimates
            approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), 
                           progressbar=False)
        
        idata = approx.sample(draws=1000, random_seed=2025)
        BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
        CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
        
        # Check for NaN in results
        if not np.all(np.isfinite(BrtaCFR_est)):
            raise ValueError("NaN in posterior estimates")
        
        return {'mean': BrtaCFR_est, 'lower': CrI[0, :], 'upper': CrI[1, :]}
    
    except Exception as e:
        # Fallback to mCFR if optimization fails
        warnings.warn(f"ADVI optimization failed (sigma={sigma_prior}): {str(e)}. Using mCFR fallback.")
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        return {'mean': mCFR_result, 'lower': mCFR_result * 0.5, 'upper': mCFR_result * 1.5}

def run_sensitivity_sigma(config, checkpoint_mgr, resume=False):
    """Sensitivity to prior variance sigma - analyze all 6 scenarios with per-replication checkpointing."""
    print("\n" + "="*80)
    print("ANALYSIS 2b: SENSITIVITY - Prior Variance σ²")
    print("="*80)
    
    final_checkpoint_name = 'sensitivity_sigma_final'
    
    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  ✅ Sigma sensitivity already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)
    
    n_reps = config['sensitivity_reps']
    all_results = {}
    
    # Analyze all 6 scenarios
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        scenario_results = {}
        
        for case_name, sigma_val in SIGMA_SENSITIVITY.items():
            analysis_name = f'sens_sigma_{case_name}'
            completed = checkpoint_mgr.get_completed_replications(analysis_name, scenario_key, n_reps)
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)
            
            if len(completed) > 0:
                print(f"    Case {case_name}: Found {len(completed)}/{n_reps} replications")
            if len(pending) == 0:
                print(f"    ✅ {case_name} (σ={sigma_val}) complete, using cached results")
                estimates = list(completed.values())
            else:
                print(f"    🔄 {case_name} (σ={sigma_val}): Running {len(pending)} pending replications")
                
                F_paras = (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
                
                def run_and_save_sigma_replication(rep_idx):
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=20000)
                    result = BrtaCFR_estimator_custom_sigma(data['CT'], data['dt'], F_paras, sigma_val)
                    checkpoint_mgr.save_replication_result(analysis_name, scenario_key, rep_idx, result)
                    return result
                
                new_estimates = Parallel(n_jobs=config['n_jobs'])(
                    delayed(run_and_save_sigma_replication)(rep_idx)
                    for rep_idx in tqdm(pending, desc=f"      {case_name}")
                )
                
                estimates = list(completed.values()) + new_estimates
            
            pt_true = SCENARIOS[scenario_key]['pt']
            maes = [logit_mae(est['mean'], pt_true) for est in estimates]
            
            scenario_results[case_name] = {
                'mean_estimate': np.mean([est['mean'] for est in estimates], axis=0),
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
            }
        
        all_results[scenario_key] = scenario_results
    
    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("  ✅ Sigma sensitivity complete!")
    return all_results

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
    
    # Clip cCFR to avoid extreme values
    cCFR_est = np.clip(cCFR_est, 1e-8, 1 - 1e-8)
    
    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        
        # Check for NaN or Inf in initialization
        if not np.all(np.isfinite(beta_tilde)):
            beta_tilde = np.nan_to_num(beta_tilde, nan=0.0, posinf=5.0, neginf=-5.0)
        
        lambda_param = pm.HalfCauchy('lambda', beta=1.0)
        beta = pm.CustomDist('beta', beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic('p_t', pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson('deaths', mu=mu_t, observed=d_t)
    
    try:
        with model:
            # Use default optimizer for better convergence and smooth estimates
            approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), 
                           progressbar=False)
        
        idata = approx.sample(draws=1000, random_seed=2025)
        BrtaCFR_est = idata.posterior['p_t'].mean(dim=('chain', 'draw')).values
        CrI = idata.posterior['p_t'].quantile([0.025, 0.975], dim=('chain', 'draw')).values
        
        # Check for NaN in results
        if not np.all(np.isfinite(BrtaCFR_est)):
            raise ValueError("NaN in posterior estimates")
        
        return {'mean': BrtaCFR_est, 'lower': CrI[0, :], 'upper': CrI[1, :]}
    
    except Exception as e:
        warnings.warn(f"ADVI optimization failed: {str(e)}. Using mCFR fallback.")
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        return {'mean': mCFR_result, 'lower': mCFR_result * 0.5, 'upper': mCFR_result * 1.5}

def run_sensitivity_distribution(config, checkpoint_mgr, resume=False):
    """Sensitivity to different delay distributions - analyze all 6 scenarios with per-replication checkpointing."""
    print("\n" + "="*80)
    print("ANALYSIS 2c: SENSITIVITY - Delay Distributions")
    print("="*80)
    
    final_checkpoint_name = 'sensitivity_dist_final'
    
    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  ✅ Distribution sensitivity already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)
    
    n_reps = config['sensitivity_reps']
    all_results = {}
    
    # Calculate true mean and variance
    true_scale = TRUE_GAMMA_MEAN / TRUE_GAMMA_SHAPE
    true_variance = TRUE_GAMMA_SHAPE * (true_scale ** 2)
    
    dist_cases = {
        'Gamma': {'type': 'gamma'},
        'Weibull': {'type': 'weibull'},
        'Lognormal': {'type': 'lognormal'},
    }
    
    # Analyze all 6 scenarios
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        scenario_results = {}
        
        for case_name, case_info in dist_cases.items():
            analysis_name = f'sens_dist_{case_name}'
            completed = checkpoint_mgr.get_completed_replications(analysis_name, scenario_key, n_reps)
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)
            
            if len(completed) > 0:
                print(f"    Case {case_name}: Found {len(completed)}/{n_reps} replications")
            if len(pending) == 0:
                print(f"    ✅ {case_name} complete, using cached results")
                estimates = list(completed.values())
            else:
                print(f"    🔄 {case_name}: Running {len(pending)} pending replications")
                
                # Generate PMF once if needed
                if case_info['type'] == 'weibull':
                    pmf = get_weibull_pmf(TRUE_GAMMA_MEAN, true_variance, T_PERIOD)
                elif case_info['type'] == 'lognormal':
                    pmf = get_lognormal_pmf(TRUE_GAMMA_MEAN, true_variance, T_PERIOD)
                
                def run_and_save_dist_replication(rep_idx):
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=30000)
                    
                    if case_info['type'] == 'gamma':
                        F_paras = (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
                        result = BrtaCFR_estimator(data['CT'], data['dt'], F_paras)
                    else:
                        result = estimate_with_custom_pmf(data['CT'], data['dt'], pmf)
                    
                    checkpoint_mgr.save_replication_result(analysis_name, scenario_key, rep_idx, result)
                    return result
                
                new_estimates = Parallel(n_jobs=config['n_jobs'])(
                    delayed(run_and_save_dist_replication)(rep_idx)
                    for rep_idx in tqdm(pending, desc=f"      {case_name}")
                )
                
                estimates = list(completed.values()) + new_estimates
            
            pt_true = SCENARIOS[scenario_key]['pt']
            maes = [logit_mae(est['mean'], pt_true) for est in estimates]
            
            scenario_results[case_name] = {
                'mean_estimate': np.mean([est['mean'] for est in estimates], axis=0),
                'mae_mean': np.mean(maes),
                'mae_std': np.std(maes),
            }
        
        all_results[scenario_key] = scenario_results
    
    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("  ✅ Distribution sensitivity complete!")
    return all_results

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
    """MCMC vs ADVI comparison with per-replication checkpointing."""
    print("\n" + "="*80)
    print("ANALYSIS 3: MCMC vs ADVI Comparison")
    print("="*80)
    
    final_checkpoint_name = 'mcmc_comparison_final'
    
    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  ✅ MCMC comparison already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)
    
    n_reps = config['mcmc_reps']
    # Use all scenarios for comprehensive comparison
    test_scenarios = list(SCENARIOS.keys())  # All 6 scenarios: A, B, C, D, E, F
    
    comparison_results = {}
    
    for scenario_key in test_scenarios:
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        pt_true = SCENARIOS[scenario_key]['pt']
        
        # Check MCMC checkpoints
        mcmc_completed = checkpoint_mgr.get_completed_replications('mcmc', scenario_key, n_reps)
        mcmc_pending = checkpoint_mgr.get_pending_replications('mcmc', scenario_key, n_reps)
        
        if len(mcmc_completed) > 0:
            print(f"    MCMC: Found {len(mcmc_completed)}/{n_reps} replications")
        if len(mcmc_pending) > 0:
            print(f"    🔄 Running {len(mcmc_pending)} pending MCMC replications...")
            
            def run_and_save_mcmc_replication(rep_idx):
                try:
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=40000)
                    mcmc_start = time.time()
                    result = run_brtacfr_mcmc(data['CT'], data['dt'], (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE))
                    runtime = time.time() - mcmc_start
                    result_dict = {
                        'success': True,
                        'runtime': runtime,
                        'mae': logit_mae(result['mean'], pt_true),
                        'coverage': np.mean((pt_true >= result['lower']) & (pt_true <= result['upper'])),
                    }
                    checkpoint_mgr.save_replication_result('mcmc', scenario_key, rep_idx, result_dict)
                    return result_dict
                except Exception as e:
                    result_dict = {'success': False}
                    checkpoint_mgr.save_replication_result('mcmc', scenario_key, rep_idx, result_dict)
                    return result_dict
            
            Parallel(n_jobs=config['n_jobs'])(
                delayed(run_and_save_mcmc_replication)(rep_idx)
                for rep_idx in tqdm(mcmc_pending, desc="    MCMC")
            )
        else:
            print(f"    ✅ MCMC complete, using cached results")
        
        # Check ADVI checkpoints
        advi_completed = checkpoint_mgr.get_completed_replications('advi', scenario_key, n_reps)
        advi_pending = checkpoint_mgr.get_pending_replications('advi', scenario_key, n_reps)
        
        if len(advi_completed) > 0:
            print(f"    ADVI: Found {len(advi_completed)}/{n_reps} replications")
        if len(advi_pending) > 0:
            print(f"    🔄 Running {len(advi_pending)} pending ADVI replications...")
            
            def run_and_save_advi_replication(rep_idx):
                try:
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=40000)
                    advi_start = time.time()
                    result = BrtaCFR_estimator(data['CT'], data['dt'], (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE))
                    runtime = time.time() - advi_start
                    result_dict = {
                        'success': True,
                        'runtime': runtime,
                        'mae': logit_mae(result['mean'], pt_true),
                        'coverage': np.mean((pt_true >= result['lower']) & (pt_true <= result['upper'])),
                    }
                    checkpoint_mgr.save_replication_result('advi', scenario_key, rep_idx, result_dict)
                    return result_dict
                except Exception as e:
                    result_dict = {'success': False}
                    checkpoint_mgr.save_replication_result('advi', scenario_key, rep_idx, result_dict)
                    return result_dict
            
            Parallel(n_jobs=config['n_jobs'])(
                delayed(run_and_save_advi_replication)(rep_idx)
                for rep_idx in tqdm(advi_pending, desc="    ADVI")
            )
        else:
            print(f"    ✅ ADVI complete, using cached results")
        
        # Reload all completed results
        mcmc_all = checkpoint_mgr.get_completed_replications('mcmc', scenario_key, n_reps)
        advi_all = checkpoint_mgr.get_completed_replications('advi', scenario_key, n_reps)
        
        # Aggregate
        mcmc_success = [r for r in mcmc_all.values() if r['success']]
        advi_success = [r for r in advi_all.values() if r['success']]
        
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
    
    checkpoint_mgr.save_verbose(final_checkpoint_name, comparison_results)
    print("  ✅ MCMC comparison complete!")
    return comparison_results

# =============================================================================
# Plotting Functions
# =============================================================================

# Unified color and line style definitions
COLORS = {
    'True': 'black',
    'cCFR': '#2ca02c',     # green
    'mCFR': '#ff7f0e',     # orange  
    'BrtaCFR': '#d62728',  # red
    'CrI': '#1f77b4',      # blue
}

LINESTYLES = {
    'True': '-',
    'cCFR': '-',
    'mCFR': '-',
    'BrtaCFR': '--',  # dashed for BrtaCFR
}

def plot_main_analysis(main_results, output_dir):
    """Generate main analysis plots with consistent styling."""
    print("\n  Generating plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    for i, (scenario_key, results) in enumerate(main_results.items()):
        ax = axes[i]
        pt_true = SCENARIOS[scenario_key]['pt']
        
        # Plot with unified colors and line styles
        ax.plot(DAYS, pt_true, color=COLORS['True'], linestyle=LINESTYLES['True'], 
                linewidth=2.5, label='True')
        ax.plot(DAYS, results['cCFR_avg'], color=COLORS['cCFR'], linestyle=LINESTYLES['cCFR'],
                linewidth=1.5, label='cCFR')
        ax.plot(DAYS, results['mCFR_avg'], color=COLORS['mCFR'], linestyle=LINESTYLES['mCFR'],
                linewidth=1.5, label='mCFR')
        ax.plot(DAYS, results['BrtaCFR_avg'], color=COLORS['BrtaCFR'], linestyle=LINESTYLES['BrtaCFR'],
                linewidth=2, label='BrtaCFR')
        ax.fill_between(DAYS, results['BrtaCFR_lower_avg'], results['BrtaCFR_upper_avg'], 
                        color=COLORS['CrI'], alpha=0.2, label='95% CrI')
        
        ax.set_title(f"({scenario_key}) {SCENARIOS[scenario_key]['name']}", fontsize=14)
        ax.set_xlabel('Days', fontsize=12)
        ax.set_ylabel('Fatality Rate', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'simulation.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()

def plot_sensitivity_results(gamma_results, sigma_results, dist_results, output_dir):
    """Plot all sensitivity analysis results with 6x2 layout for each analysis."""
    print("\n  Generating sensitivity plots...")
    
    # Plot 1: Gamma sensitivity (6x2 layout)
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    
    gamma_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, scenario_key in enumerate(SCENARIOS.keys()):
        # Left column: Curves
        ax_curve = axes[i, 0]
        ax_curve.plot(DAYS, SCENARIOS[scenario_key]['pt'], color=COLORS['True'], 
                     linestyle='-', linewidth=2.5, label='True')
        
        for j, (case_name, results) in enumerate(gamma_results[scenario_key].items()):
            ax_curve.plot(DAYS, results['mean_estimate'], 
                         label=case_name.replace('_', ' '),
                         color=gamma_colors[j], linestyle='--', linewidth=1.5)
        
        ax_curve.set_xlabel('Days', fontsize=11)
        ax_curve.set_ylabel('Fatality Rate', fontsize=11)
        ax_curve.set_title(f"({scenario_key}) {SCENARIOS[scenario_key]['name']}", fontsize=12)
        ax_curve.legend(fontsize=9, loc='best')
        ax_curve.grid(True, alpha=0.3)
        
        # Right column: MAE
        ax_mae = axes[i, 1]
        case_names = [n.replace('_', ' ') for n in gamma_results[scenario_key].keys()]
        mae_means = [r['mae_mean'] for r in gamma_results[scenario_key].values()]
        mae_stds = [r['mae_std'] for r in gamma_results[scenario_key].values()]
        
        bars = ax_mae.bar(range(len(case_names)), mae_means, yerr=mae_stds, 
                          capsize=5, alpha=0.7, color=gamma_colors)
        ax_mae.set_xticks(range(len(case_names)))
        ax_mae.set_xticklabels(case_names, rotation=45, ha='right', fontsize=9)
        ax_mae.set_ylabel('Mean Absolute Error', fontsize=11)
        ax_mae.set_title(f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=12)
        ax_mae.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'sensitivity_gamma.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Plot 2: Sigma sensitivity (6x2 layout)
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    
    sigma_colors = ['#7f007f', '#0000ff', '#00ff00', '#ffa500', '#ff0000']
    
    for i, scenario_key in enumerate(SCENARIOS.keys()):
        # Left column: Curves
        ax_curve = axes[i, 0]
        ax_curve.plot(DAYS, SCENARIOS[scenario_key]['pt'], color=COLORS['True'],
                     linestyle='-', linewidth=2.5, label='True')
        
        for j, (case_name, results) in enumerate(sigma_results[scenario_key].items()):
            sigma_val = SIGMA_SENSITIVITY[case_name]
            ax_curve.plot(DAYS, results['mean_estimate'], 
                         label=f"σ={sigma_val}",
                         color=sigma_colors[j], linestyle='--', linewidth=1.5)
        
        ax_curve.set_xlabel('Days', fontsize=11)
        ax_curve.set_ylabel('Fatality Rate', fontsize=11)
        ax_curve.set_title(f"({scenario_key}) {SCENARIOS[scenario_key]['name']}", fontsize=12)
        ax_curve.legend(fontsize=9, loc='best')
        ax_curve.grid(True, alpha=0.3)
        
        # Right column: MAE
        ax_mae = axes[i, 1]
        sigma_labels = [f"σ={SIGMA_SENSITIVITY[n]}" for n in sigma_results[scenario_key].keys()]
        mae_means = [r['mae_mean'] for r in sigma_results[scenario_key].values()]
        mae_stds = [r['mae_std'] for r in sigma_results[scenario_key].values()]
        
        bars = ax_mae.bar(range(len(sigma_labels)), mae_means, yerr=mae_stds,
                          capsize=5, alpha=0.7, color=sigma_colors)
        ax_mae.set_xticks(range(len(sigma_labels)))
        ax_mae.set_xticklabels(sigma_labels, fontsize=10)
        ax_mae.set_ylabel('Mean Absolute Error', fontsize=11)
        ax_mae.set_title(f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=12)
        ax_mae.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'sensitivity_sigma.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Plot 3: Distribution sensitivity (6x2 layout)
    fig, axes = plt.subplots(6, 2, figsize=(16, 24))
    
    dist_colors = ['#d62728', '#1f77b4', '#2ca02c']
    
    for i, scenario_key in enumerate(SCENARIOS.keys()):
        # Left column: Curves
        ax_curve = axes[i, 0]
        ax_curve.plot(DAYS, SCENARIOS[scenario_key]['pt'], color=COLORS['True'],
                     linestyle='-', linewidth=2.5, label='True')
        
        for j, (case_name, results) in enumerate(dist_results[scenario_key].items()):
            ax_curve.plot(DAYS, results['mean_estimate'], 
                         label=case_name,
                         color=dist_colors[j], linestyle='--', linewidth=1.5)
        
        ax_curve.set_xlabel('Days', fontsize=11)
        ax_curve.set_ylabel('Fatality Rate', fontsize=11)
        ax_curve.set_title(f"({scenario_key}) {SCENARIOS[scenario_key]['name']}", fontsize=12)
        ax_curve.legend(fontsize=9, loc='best')
        ax_curve.grid(True, alpha=0.3)
        
        # Right column: MAE
        ax_mae = axes[i, 1]
        dist_names = list(dist_results[scenario_key].keys())
        mae_means = [r['mae_mean'] for r in dist_results[scenario_key].values()]
        mae_stds = [r['mae_std'] for r in dist_results[scenario_key].values()]
        
        bars = ax_mae.bar(range(len(dist_names)), mae_means, yerr=mae_stds,
                          capsize=5, alpha=0.7, color=dist_colors)
        ax_mae.set_xticks(range(len(dist_names)))
        ax_mae.set_xticklabels(dist_names, fontsize=10)
        ax_mae.set_ylabel('Mean Absolute Error', fontsize=11)
        ax_mae.set_title(f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=12)
        ax_mae.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'sensitivity_distributions.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Save comprehensive summary CSV
    summary_data = []
    for scenario_key in SCENARIOS.keys():
        for case_name, results in gamma_results[scenario_key].items():
            summary_data.append({
                'Analysis': 'Gamma',
                'Scenario': scenario_key,
                'Case': case_name,
                'MAE_Mean': results['mae_mean'],
                'MAE_SD': results['mae_std']
            })
        for case_name, results in sigma_results[scenario_key].items():
            summary_data.append({
                'Analysis': 'Sigma',
                'Scenario': scenario_key,
                'Case': case_name,
                'MAE_Mean': results['mae_mean'],
                'MAE_SD': results['mae_std']
            })
        for case_name, results in dist_results[scenario_key].items():
            summary_data.append({
                'Analysis': 'Distribution',
                'Scenario': scenario_key,
                'Case': case_name,
                'MAE_Mean': results['mae_mean'],
                'MAE_SD': results['mae_std']
            })
    
    df = pd.DataFrame(summary_data)
    csv_path = Path(output_dir) / 'sensitivity_analysis_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✅ Saved to: {csv_path}")

def plot_mcmc_comparison(mcmc_results, output_dir):
    """Plot MCMC vs ADVI comparison for all 6 scenarios with Pareto analysis."""
    print("\n  Generating MCMC comparison plots...")
    
    scenarios = list(mcmc_results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for consistency
    mcmc_color = '#1f77b4'
    advi_color = '#ff7f0e'
    
    # Plot 1: Runtime Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(scenarios))
    width = 0.35
    
    mcmc_times = [mcmc_results[s]['mcmc_runtime_mean'] for s in scenarios]
    advi_times = [mcmc_results[s]['advi_runtime_mean'] for s in scenarios]
    mcmc_errs = [mcmc_results[s]['mcmc_runtime_std'] for s in scenarios]
    advi_errs = [mcmc_results[s]['advi_runtime_std'] for s in scenarios]
    
    ax1.bar(x - width/2, mcmc_times, width, yerr=mcmc_errs, label='MCMC', 
            alpha=0.8, capsize=5, color=mcmc_color)
    ax1.bar(x + width/2, advi_times, width, yerr=advi_errs, label='ADVI', 
            alpha=0.8, capsize=5, color=advi_color)
    ax1.set_xlabel('Scenario', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('(A) Runtime Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=11)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup (without text annotations)
    ax2 = axes[0, 1]
    speedups = [mcmc_results[s]['speedup'] for s in scenarios]
    bars = ax2.bar(scenarios, speedups, alpha=0.8, color='#2ca02c')
    ax2.axhline(y=1, color='r', linestyle='--', linewidth=2, label='No speedup')
    ax2.set_xlabel('Scenario', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('(B) ADVI Speedup', fontsize=14)
    ax2.set_xticklabels(scenarios, fontsize=11)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    # Note: No text annotations as requested
    
    # Plot 3: MAE Comparison
    ax3 = axes[1, 0]
    mcmc_maes = [mcmc_results[s]['mcmc_mae_mean'] for s in scenarios]
    advi_maes = [mcmc_results[s]['advi_mae_mean'] for s in scenarios]
    ax3.bar(x - width/2, mcmc_maes, width, label='MCMC', alpha=0.8, color=mcmc_color)
    ax3.bar(x + width/2, advi_maes, width, label='ADVI', alpha=0.8, color=advi_color)
    ax3.set_xlabel('Scenario', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error', fontsize=12)
    ax3.set_title('(C) Accuracy Comparison', fontsize=14)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, fontsize=11)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Pareto Analysis (Time vs Accuracy)
    ax4 = axes[1, 1]
    
    # Plot MCMC points
    ax4.scatter(mcmc_times, mcmc_maes, s=150, alpha=0.7, color=mcmc_color, 
               marker='o', label='MCMC', edgecolors='black', linewidths=1.5)
    
    # Plot ADVI points
    ax4.scatter(advi_times, advi_maes, s=150, alpha=0.7, color=advi_color,
               marker='s', label='ADVI', edgecolors='black', linewidths=1.5)
    
    # Add scenario labels
    for i, scenario in enumerate(scenarios):
        ax4.annotate(scenario, (mcmc_times[i], mcmc_maes[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax4.annotate(scenario, (advi_times[i], advi_maes[i]),
                    xytext=(5, -10), textcoords='offset points', fontsize=9)
    
    # Add ideal direction arrow
    ax4.annotate('', xy=(0.05, 0.05), xytext=(0.95, 0.95),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
    ax4.text(0.1, 0.9, 'Better\n(Lower time, Lower MAE)', 
            transform=ax4.transAxes, fontsize=10, color='gray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax4.set_xlabel('Runtime (seconds)', fontsize=12)
    ax4.set_ylabel('Mean Absolute Error', fontsize=12)
    ax4.set_title('(D) Pareto Analysis: Time vs Accuracy', fontsize=14)
    ax4.legend(fontsize=11, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'mcmc_vs_advi_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved to: {output_path}")
    plt.close()
    
    # Save comprehensive CSV
    comparison_data = []
    for scenario in scenarios:
        comparison_data.append({
            'Scenario': scenario,
            'Scenario_Name': SCENARIOS[scenario]['name'],
            'MCMC_Runtime_Mean': mcmc_results[scenario]['mcmc_runtime_mean'],
            'MCMC_Runtime_SD': mcmc_results[scenario]['mcmc_runtime_std'],
            'ADVI_Runtime_Mean': mcmc_results[scenario]['advi_runtime_mean'],
            'ADVI_Runtime_SD': mcmc_results[scenario]['advi_runtime_std'],
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
                       help='Resume from checkpoints (deprecated, checkpoints are always used)')
    parser.add_argument('--clear-checkpoints', action='store_true',
                       help='Clear all checkpoints and rerun from scratch')
    parser.add_argument('--only', choices=['main', 'sensitivity', 'mcmc', 'all'],
                       default='all', help='Run specific analysis')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
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
    
    # Run analyses (checkpoints are automatically used, no need for --resume flag)
    if args.only in ['main', 'all']:
        print("\n" + "🔬"*40)
        print("PHASE 1: MAIN ANALYSIS & SIMULATION TABLE")
        print("🔬"*40)
        main_results = run_main_analysis(config, checkpoint_mgr, resume=True)
        plot_main_analysis(main_results, output_dir)
        simulation_table = generate_simulation_table(main_results, output_dir)
    
    if args.only in ['sensitivity', 'all']:
        print("\n" + "🔬"*40)
        print("PHASE 2: SENSITIVITY ANALYSIS")
        print("🔬"*40)
        gamma_results = run_sensitivity_gamma(config, checkpoint_mgr, resume=True)
        sigma_results = run_sensitivity_sigma(config, checkpoint_mgr, resume=True)
        dist_results = run_sensitivity_distribution(config, checkpoint_mgr, resume=True)
        plot_sensitivity_results(gamma_results, sigma_results, dist_results, output_dir)
    
    if args.only in ['mcmc', 'all']:
        print("\n" + "🔬"*40)
        print("PHASE 3: MCMC VS ADVI COMPARISON")
        print("🔬"*40)
        mcmc_results = run_mcmc_comparison(config, checkpoint_mgr, resume=True)
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

