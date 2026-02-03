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
    python run_all_simulations.py --demo          # Quick demo (2 reps); includes full lambda run (R4-1)
    python run_all_simulations.py --clear-checkpoints  # Clear cache and restart
    python run_all_simulations.py --only main     # Run specific analysis
    python run_all_simulations.py --only lambda --do_prior_predictive --do_lambda_sensitivity  # R4-1 only

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
os.environ["PYTENSOR_FLAGS"] = "optimizer=fast_compile,exception_verbosity=low"

# Suppress PyTensor BLAS warning
warnings.filterwarnings("ignore", message=".*PyTensor could not link to a BLAS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")

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
from scipy import stats

# Import core methods
from methods import BrtaCFR_estimator, mCFR_EST, logp, lambda_summary_stats

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


def curve_level_coverage_from_cri(truth_1d, cri_by_level):
    """
    Curve-level coverage for one replication from precomputed CrI bounds.
    C_r = mean_t I(truth_t in [CI_lower(t), CI_upper(t)]).
    cri_by_level: dict level -> (lower_1d, upper_1d), e.g. {0.5: (q25, q75), ...}.
    Returns dict level -> C_r (float in [0,1]). Handles None/missing cri_by_level.
    """
    out = {}
    if cri_by_level is None or not isinstance(cri_by_level, dict):
        return out
    truth_flat = np.asarray(truth_1d).flatten()
    for level, bounds in cri_by_level.items():
        if bounds is None or not isinstance(bounds, (tuple, list)) or len(bounds) < 2:
            continue
        lo, hi = bounds[0], bounds[1]
        if lo is None or hi is None:
            continue
        lo, hi = np.asarray(lo).flatten(), np.asarray(hi).flatten()
        t = min(len(truth_flat), len(lo), len(hi))
        if t == 0:
            continue
        truth = truth_flat[:t]
        inside = (truth >= lo[:t]) & (truth <= hi[:t])
        out[level] = float(np.mean(inside))
    return out


def average_curve_level_coverage(
    results_list, truth_pt, truth_mut, pt_cri_key="pt_cri", mut_cri_key="mut_cri"
):
    """
    Average curve-level coverage C̄ = mean_r C_r over replications.
    results_list: list of dicts, each may have pt_cri, mut_cri (dict level -> (lower, upper)).
    truth_pt, truth_mut: 1d arrays (same length as CrI).
    Returns (cov_pt_by_level, cov_mut_by_level) each dict level -> C̄.
    """
    cov_pt = {L: [] for L in COVERAGE_LEVELS}
    cov_mut = {L: [] for L in COVERAGE_LEVELS}
    for r in results_list:
        c_pt = curve_level_coverage_from_cri(truth_pt, r.get(pt_cri_key))
        c_mut = curve_level_coverage_from_cri(truth_mut, r.get(mut_cri_key))
        for L in COVERAGE_LEVELS:
            if L in c_pt:
                cov_pt[L].append(c_pt[L])
            if L in c_mut:
                cov_mut[L].append(c_mut[L])
    out_pt = {L: float(np.mean(cov_pt[L])) if cov_pt[L] else None for L in COVERAGE_LEVELS}
    out_mut = {L: float(np.mean(cov_mut[L])) if cov_mut[L] else None for L in COVERAGE_LEVELS}
    return out_pt, out_mut


# =============================================================================
# Configuration
# =============================================================================

# Default configuration for full analysis
DEFAULT_CONFIG = {
    "main_reps": 1000,  # Main analysis replications
    "sensitivity_reps": 100,  # Sensitivity analysis replications
    "mcmc_reps": 10,  # MCMC comparison replications
    "n_jobs": 16,  # Limit parallel jobs to prevent memory issues
    "checkpoint_dir": "./checkpoints",
    "output_dir": "./outputs",
}

# Demo mode configuration (fast check)
DEMO_CONFIG = {
    "main_reps": 2,
    "sensitivity_reps": 2,
    "mcmc_reps": 2,
    "n_jobs": 4,  # Limit parallel jobs to prevent memory issues
    "checkpoint_dir": "./checkpoints_demo",
    "output_dir": "./outputs_demo",
}

# Scenario definitions
T_PERIOD = 200
DAYS = np.arange(1, T_PERIOD + 1)
CT = 3000 - 5 * np.abs(100 - DAYS)

SCENARIOS = {
    "A": {"name": "Constant", "pt": np.full(T_PERIOD, 0.034)},
    "B": {"name": "Exponential Growth", "pt": 0.01 * np.exp(0.012 * DAYS)},
    "C": {
        "name": "Delayed Growth",
        "pt": 0.04 * np.exp(0.016 * np.where(DAYS > 60, np.minimum(40, DAYS - 60), 0)),
    },
    "D": {"name": "Decay", "pt": 0.1 * np.exp(-0.009 * np.where(DAYS > 70, DAYS - 70, 0))},
    "E": {"name": "Peak", "pt": 0.1 * np.exp(-0.015 * np.abs(DAYS - 80))},
    "F": {"name": "Valley", "pt": 0.015 * np.exp(0.018 * np.abs(DAYS - 120))},
}

# Delay distribution parameters
TRUE_GAMMA_MEAN = 15.43
TRUE_GAMMA_SHAPE = 2.03

# =============================================================================
# Calculate True μ_t for each scenario (for PPC visualization)
# =============================================================================


def calculate_true_mu_t(scenario_key):
    """
    Calculate true μ_t (expected deaths) using true CFR and true delay distribution.
    μ_t = Σ_{k=0}^{t-1} f_k * c_{t-k} * p_{t-k}
    where:
        - c_t: cumulative cases at time t
        - p_t: true CFR at time t
        - f_k: delay distribution PMF
    """
    pt_true = SCENARIOS[scenario_key]["pt"]
    mean_delay, shape_delay = TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE
    scale_delay = mean_delay / shape_delay

    # Calculate delay distribution PMF
    F_k = gamma.cdf(np.arange(T_PERIOD + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)

    # Calculate true deaths without delay
    true_deaths_no_delay = CT * pt_true

    # Apply delay distribution to get true μ_t
    mu_t_true = np.array(
        [np.sum(np.flip(f_k[:i]) * true_deaths_no_delay[:i]) for i in np.arange(1, T_PERIOD + 1)]
    )

    return mu_t_true


# Pre-calculate true μ_t for all scenarios
TRUE_MU_T = {scenario_key: calculate_true_mu_t(scenario_key) for scenario_key in SCENARIOS.keys()}

# Sensitivity cases
GAMMA_SENSITIVITY = {
    "True": {"mean": 15.43, "shape": 2.03},
    "Mean+20%": {"mean": 15.43 * 1.2, "shape": 2.03},
    "Mean-20%": {"mean": 15.43 * 0.8, "shape": 2.03},
    "Shape+50%": {"mean": 15.43, "shape": 2.03 * 1.5},
    "Shape-50%": {"mean": 15.43, "shape": 2.03 * 0.5},
}

SIGMA_SENSITIVITY = {
    "Sigma_1": 1,
    "Sigma_3": 3,
    "Sigma_5": 5,
    "Sigma_7": 7,
    "Sigma_10": 10,
}

# Default half-Cauchy scale grid for lambda prior (R4-1)
DEFAULT_LAMBDA_SCALES = [0.1, 0.5, 1.0, 5.0, 10.0]
N_PRIOR_DRAWS = 500  # for prior predictive smoothness check

# R4-8: Curve-level coverage levels (CrI 50%, 80%, 95%)
COVERAGE_LEVELS = [0.50, 0.80, 0.95]
# Default method for simulation curve-level coverage figure (main manuscript method)
DEFAULT_SIMULATION_METHOD = "ADVI"

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
                if key in ["mu_samples", "posterior_samples", "trace"]:
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
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._cache[name] = data
        # Don't print for every save to avoid clutter

    def save_verbose(self, name, data):
        """Save checkpoint data with verbose output."""
        self.save(name, data)
        filepath = self.checkpoint_dir / f"{name}.pkl"
        size_kb = filepath.stat().st_size / 1024
        print(f"  [SAVED] Checkpoint saved: {name} ({size_kb:.1f} KB)")

    def load(self, name):
        """Load checkpoint data."""
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        filepath = self.checkpoint_dir / f"{name}.pkl"
        if filepath.exists():
            with open(filepath, "rb") as f:
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

    pt_true = SCENARIOS[scenario_key]["pt"]

    # Generate deaths with true Gamma distribution
    mean_delay, shape_delay = TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE
    scale_delay = mean_delay / shape_delay
    F_k = gamma.cdf(np.arange(T_PERIOD + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)

    true_deaths_no_delay = np.random.binomial(CT.astype(int), pt_true)
    dt = np.array(
        [np.sum(np.flip(f_k[:i]) * true_deaths_no_delay[:i]) for i in np.arange(1, T_PERIOD + 1)]
    )

    return {
        "CT": CT,
        "dt": dt,
        "pt_true": pt_true,
        "scenario": scenario_key,
        "rep_idx": rep_idx,
        "seed": rep_idx + seed_offset,
    }


# =============================================================================
# Analysis 0: Main Analysis (Original Manuscript)
# =============================================================================


def run_main_analysis_single(data, include_diagnostics=True):
    """
    Run main analysis for a single replication.
    If include_diagnostics=True, collect data for simulation table.
    """
    CT, dt, pt_true = data["CT"], data["dt"], data["pt_true"]

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
        brta_results = run_brtacfr_with_diagnostics(
            CT, dt, (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE), lambda_scale=1.0
        )
        runtime = time.time() - start_time

        # Compute metrics (using logit-transformed MAE) for all methods
        brta_mae = logit_mae(brta_results["mean"], pt_true)
        cCFR_mae = logit_mae(cCFR, pt_true)
        mCFR_mae = logit_mae(mCFR, pt_true)
        coverage = np.mean((pt_true >= brta_results["lower"]) & (pt_true <= brta_results["upper"]))

        # Return only summary statistics (mu_t_quantiles for PPC visualization)
        return {
            "cCFR": cCFR,
            "mCFR": mCFR,
            "BrtaCFR_mean": brta_results["mean"],
            "BrtaCFR_lower": brta_results["lower"],
            "BrtaCFR_upper": brta_results["upper"],
            # Diagnostic data
            "runtime": runtime,
            "mae": brta_mae,
            "cCFR_mae": cCFR_mae,
            "mCFR_mae": mCFR_mae,
            "coverage": coverage,
            "elbo_trace": brta_results["elbo_trace"],
            "mu_t_quantiles": brta_results["mu_t_quantiles"],
            "lambda_draws": brta_results.get("lambda_draws"),
            "pt_cri": brta_results.get("pt_cri"),
            "mut_cri": brta_results.get("mut_cri"),
            "observed_deaths": brta_results.get("observed_deaths"),
            "death_ppc_quantiles": brta_results.get("death_ppc_quantiles"),
        }
    else:
        brta_results = BrtaCFR_estimator(
            CT, dt, (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE), lambda_scale=1.0
        )
        return {
            "cCFR": cCFR,
            "mCFR": mCFR,
            "BrtaCFR_mean": brta_results["mean"],
            "BrtaCFR_lower": brta_results["lower"],
            "BrtaCFR_upper": brta_results["upper"],
        }


def run_brtacfr_with_diagnostics(c_t, d_t, F_paras, lambda_scale=1.0):
    """BrtaCFR with diagnostics: calculate μ_t quantiles for PPC functional ribbon visualization."""
    import time

    # Lazy import so plotting-only runs don't require PyMC/PyTensor installed/working.
    import pymc as pm

    start_time = time.time()

    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay

    F_k = gamma.cdf(np.arange(T + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)

    f_mat = np.zeros((T, T))
    for i in range(T):
        f_mat += np.diag(np.ones(T - i) * f_k[i], -i)
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

        lambda_param = pm.HalfCauchy("lambda", beta=lambda_scale)
        beta = pm.CustomDist("beta", beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic("p_t", pm.math.sigmoid(beta))
        mu_t = pm.Deterministic("mu_t", pm.math.dot(fc_mat, p_t))
        pm.Poisson("deaths", mu=mu_t, observed=d_t)

    try:
        with model:
            # Use default optimizer for better convergence and smooth estimates
            approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)

        # Extract ELBO trace from approximation history
        # Note: approx.hist contains negative log-likelihood values, we need to convert to ELBO
        if hasattr(approx, "hist") and approx.hist is not None:
            # approx.hist contains negative log-likelihood, convert to ELBO (which should be negative)
            elbo_trace = -approx.hist  # Convert to actual ELBO (negative values)
        else:
            elbo_trace = None

        idata = approx.sample(draws=1000, random_seed=2025)

        BrtaCFR_est = idata.posterior["p_t"].mean(dim=("chain", "draw")).values
        CrI = idata.posterior["p_t"].quantile([0.025, 0.975], dim=("chain", "draw")).values
        p_samples = idata.posterior["p_t"].values.reshape(-1, T)
        # ADVI approx.sample() may omit Deterministics; compute mu_t from p_t if missing
        if "mu_t" in idata.posterior:
            mu_samples = idata.posterior["mu_t"].values.reshape(-1, T)
        else:
            mu_samples = (
                fc_mat @ p_samples.T
            ).T  # (n_draws, T), same as model: mu_t = fc_mat @ p_t
        lambda_draws = idata.posterior["lambda"].values.flatten()

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

        # Return μ_t posterior quantiles for PPC functional ribbon visualization
        mu_t_quantiles = {
            "q10": np.quantile(mu_samples, 0.10, axis=0),
            "q90": np.quantile(mu_samples, 0.90, axis=0),
            "q05": np.quantile(mu_samples, 0.05, axis=0),
            "q95": np.quantile(mu_samples, 0.95, axis=0),
            "median": np.median(mu_samples, axis=0),
        }
        # Posterior predictive check for observed death counts: d_rep ~ Poisson(μ_t)
        rng = np.random.default_rng(2026)
        death_ppc_samples = rng.poisson(mu_samples)  # (n_draws, T)
        death_ppc_quantiles = {
            "q025": np.quantile(death_ppc_samples, 0.025, axis=0),
            "q25": np.quantile(death_ppc_samples, 0.25, axis=0),
            "q50": np.quantile(death_ppc_samples, 0.50, axis=0),
            "q75": np.quantile(death_ppc_samples, 0.75, axis=0),
            "q975": np.quantile(death_ppc_samples, 0.975, axis=0),
        }
        runtime = time.time() - start_time

        return {
            "mean": BrtaCFR_est,
            "lower": CrI[0, :],
            "upper": CrI[1, :],
            "elbo_trace": elbo_trace,
            "mu_t_quantiles": mu_t_quantiles,
            "runtime": runtime,
            "lambda_draws": lambda_draws,
            "pt_cri": pt_cri,
            "mut_cri": mut_cri,
            "observed_deaths": np.asarray(d_t).flatten(),
            "death_ppc_quantiles": death_ppc_quantiles,
        }

    except Exception as e:
        warnings.warn(f"ADVI optimization failed: {str(e)}. Using mCFR fallback.")
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        runtime = time.time() - start_time
        return {
            "mean": mCFR_result,
            "lower": mCFR_result * 0.5,
            "upper": mCFR_result * 1.5,
            "elbo_trace": None,
            "mu_t_quantiles": None,
            "runtime": runtime,
            "lambda_draws": None,
            "pt_cri": None,
            "mut_cri": None,
            "observed_deaths": np.asarray(d_t).flatten(),
            "death_ppc_quantiles": None,
        }


def run_main_analysis(config, checkpoint_mgr, resume=False):
    """Run main analysis for all scenarios with per-replication checkpointing."""
    print("\n" + "=" * 80)
    print("ANALYSIS 0: MAIN ANALYSIS (Original Manuscript)")
    print("=" * 80)

    final_checkpoint_name = "main_analysis_final"

    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  [OK] Main analysis already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)

    n_reps = config["main_reps"]
    print(f"  Replications: {n_reps}")
    print(f"  Scenarios: {len(SCENARIOS)}")
    print(f"  Collecting diagnostic data: Yes")

    all_results = {}

    for scenario_key in SCENARIOS.keys():
        print(f"\n  Processing Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")

        # Check for completed replications
        completed = checkpoint_mgr.get_completed_replications("main", scenario_key, n_reps)
        pending = checkpoint_mgr.get_pending_replications("main", scenario_key, n_reps)

        if len(completed) > 0:
            print(f"    [FOUND] Found {len(completed)}/{n_reps} completed replications")
        if len(pending) == 0:
            print(f"    [OK] All replications complete, using cached results")
            results = list(completed.values())
        else:
            print(f"    [RUNNING] Running {len(pending)} pending replications...")

            # Pre-generate all data to avoid repeated generation in parallel
            print(f"    [INFO] Pre-generating data for {len(pending)} replications...")
            pending_data = []
            for rep_idx in pending:
                data = generate_simulation_data(scenario_key, rep_idx, seed_offset=0)
                pending_data.append((rep_idx, data))

            # Run analysis on pending replications in parallel with optimized settings
            def run_and_save_replication(rep_idx_data):
                rep_idx, data = rep_idx_data
                result = run_main_analysis_single(data, include_diagnostics=True)
                checkpoint_mgr.save_replication_result("main", scenario_key, rep_idx, result)
                return result

            new_results = Parallel(n_jobs=config["n_jobs"])(
                delayed(run_and_save_replication)(rep_idx_data)
                for rep_idx_data in tqdm(pending_data, desc=f"    Scenario {scenario_key}")
            )

            # Combine with completed results
            results = list(completed.values()) + new_results
            print(f"    [OK] Scenario {scenario_key} complete ({n_reps}/{n_reps} replications)")

        # Pool lambda draws across replications for posterior lambda summary (R4-1)
        lambda_draws_list = [
            r["lambda_draws"] for r in results if r.get("lambda_draws") is not None
        ]
        if lambda_draws_list:
            lambda_pooled = np.concatenate([np.asarray(d).flatten() for d in lambda_draws_list])
            lam_med, lam_q025, lam_q975 = lambda_summary_stats(lambda_pooled)
        else:
            lam_med, lam_q025, lam_q975 = None, None, None

        # R4-8: Average curve-level coverage (C̄) for p_t and mu_t at 50/80/95%
        pt_true = SCENARIOS[scenario_key]["pt"]
        mu_t_true = TRUE_MU_T[scenario_key]
        cov_pt, cov_mut = average_curve_level_coverage(results, pt_true, mu_t_true)

        # Aggregate results
        all_results[scenario_key] = {
            "cCFR_avg": np.mean([r["cCFR"] for r in results], axis=0),
            "mCFR_avg": np.mean([r["mCFR"] for r in results], axis=0),
            "BrtaCFR_avg": np.mean([r["BrtaCFR_mean"] for r in results], axis=0),
            "BrtaCFR_lower_avg": np.mean([r["BrtaCFR_lower"] for r in results], axis=0),
            "BrtaCFR_upper_avg": np.mean([r["BrtaCFR_upper"] for r in results], axis=0),
            # Diagnostic data for simulation table
            "runtime_mean": np.mean([r["runtime"] for r in results]),
            "runtime_std": np.std([r["runtime"] for r in results]),
            "mae_values": [r["mae"] for r in results],
            "cCFR_mae_values": [r["cCFR_mae"] for r in results],
            "mCFR_mae_values": [r["mCFR_mae"] for r in results],
            "coverage_values": [r["coverage"] for r in results],
            "elbo_traces": [r["elbo_trace"] for r in results if r["elbo_trace"] is not None],
            "mu_t_quantiles_list": [
                r["mu_t_quantiles"] for r in results if r["mu_t_quantiles"] is not None
            ],
            "observed_deaths_list": [
                r["observed_deaths"] for r in results if r.get("observed_deaths") is not None
            ],
            "death_ppc_quantiles_list": [
                r["death_ppc_quantiles"]
                for r in results
                if r.get("death_ppc_quantiles") is not None
            ],
            "lambda_median": lam_med,
            "lambda_q025": lam_q025,
            "lambda_q975": lam_q975,
            "coverage_pt_50": cov_pt.get(0.50),
            "coverage_pt_80": cov_pt.get(0.80),
            "coverage_pt_95": cov_pt.get(0.95),
            "coverage_mut_50": cov_mut.get(0.50),
            "coverage_mut_80": cov_mut.get(0.80),
            "coverage_mut_95": cov_mut.get(0.95),
        }

    # Save final aggregated results
    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("\n  [OK] Main analysis complete!")

    return all_results


def write_lambda_summary_sim(main_results, output_dir):
    """Write posterior lambda summaries (median, 95% CrI) from main analysis to output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario_key in SCENARIOS.keys():
        r = main_results.get(scenario_key, {})
        lam_med, lam_q025, lam_q975 = (
            r.get("lambda_median"),
            r.get("lambda_q025"),
            r.get("lambda_q975"),
        )
        if lam_med is None:
            continue
        rows.append(
            {
                "Scenario": scenario_key,
                "Scenario_Name": SCENARIOS[scenario_key]["name"],
                "lambda_median": lam_med,
                "lambda_q025": lam_q025,
                "lambda_q975": lam_q975,
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "lambda_summary_sim.csv"
    tex_path = Path(output_dir) / "lambda_summary_sim.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"  [OK] Saved to: {csv_path}, {tex_path}")


def plot_sim_curvelevel_coverage(main_results, output_dir):
    """R4-8: Simulation curve-level coverage figure (p_t and mu_t, 50/80/95%) and summary CSV."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    scenarios = list(SCENARIOS.keys())
    rows = []
    for sk in scenarios:
        r = main_results.get(sk, {})
        for theta, key_prefix in [("pt", "coverage_pt"), ("mut", "coverage_mut")]:
            for level in COVERAGE_LEVELS:
                key = f"{key_prefix}_{int(level*100)}"
                val = r.get(key)
                rows.append(
                    {
                        "scenario": sk,
                        "method": DEFAULT_SIMULATION_METHOD,
                        "theta": theta,
                        "level": level,
                        "avg_curve_coverage": val if val is not None else np.nan,
                    }
                )

    has_any = any(pd.DataFrame(rows)["avg_curve_coverage"].notna())
    if not has_any:
        print(
            "  [SKIP] No curve-level coverage data in main_results; skip fig_sim_curvelevel_coverage."
        )
        return

    df = pd.DataFrame(rows)
    csv_path = Path(output_dir) / "curvelevel_coverage_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved to: {csv_path}")

    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(scenarios))
    width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax_idx, (theta, title) in enumerate([("pt", r"$p_t$"), ("mut", r"$\mu_t$")]):
        ax = axes[ax_idx]
        for i, level in enumerate(COVERAGE_LEVELS):
            vals = [
                main_results.get(sk, {}).get(f"coverage_{theta}_{int(level*100)}", np.nan)
                for sk in scenarios
            ]
            vals = [np.clip(v, 0, 1) if v is not None and not np.isnan(v) else np.nan for v in vals]
            off = (i - 1) * width
            bars = ax.bar(
                x + off,
                vals,
                width,
                label=f"{int(level*100)}% CrI",
                color=colors[i],
                edgecolor="black",
            )
        for level in COVERAGE_LEVELS:
            ax.axhline(y=level, color="r", linestyle="--", linewidth=1.5, zorder=10)
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, fontweight="bold")
        ax.set_ylabel("Average curve-level coverage", fontweight="bold", labelpad=8)
        ax.set_xlabel("Scenario", fontweight="bold")
        ax.set_title(f"Curve-level coverage: {title}", fontweight="bold")
        ax.set_ylim(0, 1)
        if ax_idx == 0:
            ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.5, axis="y")

    plt.tight_layout(pad=1.2)
    fig_path = Path(output_dir) / "fig_sim_curvelevel_coverage_pt_mut.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {fig_path}")
    plt.rcParams.update(plt.rcParamsDefault)


# =============================================================================
# Analysis 1: Simulation Table (Uses data from main analysis)
# =============================================================================


def generate_simulation_table(main_results, output_dir):
    """Generate simulation table from main analysis results."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: SIMULATION TABLE")
    print("=" * 80)
    print("  Using diagnostic data from main analysis...")

    table_data = []

    for scenario_key, results in main_results.items():
        row = {
            "Scenario": f"{scenario_key}: {SCENARIOS[scenario_key]['name']}",
            "runtime_mean": results["runtime_mean"],
            "runtime_std": results["runtime_std"],
            "mae_mean": np.mean(results["mae_values"]),
            "mae_std": np.std(results["mae_values"]),
            "cCFR_mae_mean": np.mean(results["cCFR_mae_values"]),
            "cCFR_mae_std": np.std(results["cCFR_mae_values"]),
            "mCFR_mae_mean": np.mean(results["mCFR_mae_values"]),
            "mCFR_mae_std": np.std(results["mCFR_mae_values"]),
            "coverage_mean": np.mean(results["coverage_values"]),
            "coverage_std": np.std(results["coverage_values"]),
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Save CSV
    csv_path = Path(output_dir) / "simulation_table_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved to: {csv_path}")

    # Save LaTeX
    latex_path = Path(output_dir) / "simulation_table_latex.tex"
    with open(latex_path, "w") as f:
        f.write(df.to_latex(index=False, float_format="%.4f"))
    print(f"  [OK] Saved to: {latex_path}")

    return df


# =============================================================================
# Analysis 2: Sensitivity Analysis
# =============================================================================


def run_sensitivity_gamma(config, checkpoint_mgr, resume=False):
    """Sensitivity to Gamma parameters - analyze all 6 scenarios with per-replication checkpointing."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2a: SENSITIVITY - Gamma Parameters")
    print("=" * 80)

    final_checkpoint_name = "sensitivity_gamma_final"

    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  [OK] Gamma sensitivity already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)

    n_reps = config["sensitivity_reps"]
    all_results = {}

    # Analyze all 6 scenarios
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        scenario_results = {}

        for case_name, case_params in GAMMA_SENSITIVITY.items():
            analysis_name = f"sens_gamma_{case_name}"
            completed = checkpoint_mgr.get_completed_replications(
                analysis_name, scenario_key, n_reps
            )
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)

            if len(completed) > 0:
                print(f"    Case {case_name}: Found {len(completed)}/{n_reps} replications")
            if len(pending) == 0:
                print(f"    [OK] {case_name} complete, using cached results")
                estimates = list(completed.values())
            else:
                print(f"    [RUNNING] {case_name}: Running {len(pending)} pending replications")

                # Estimate with potentially misspecified parameters
                F_paras = (case_params["mean"], case_params["shape"])

                def run_and_save_gamma_replication(rep_idx):
                    # Reuse data generation to avoid duplicate computation
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=10000)
                    # Use faster estimator without diagnostics for sensitivity analysis
                    result = BrtaCFR_estimator(data["CT"], data["dt"], F_paras)
                    checkpoint_mgr.save_replication_result(
                        analysis_name, scenario_key, rep_idx, result
                    )
                    return result

                new_estimates = Parallel(n_jobs=config["n_jobs"])(
                    delayed(run_and_save_gamma_replication)(rep_idx)
                    for rep_idx in tqdm(pending, desc=f"      {case_name}")
                )

                estimates = list(completed.values()) + new_estimates

            pt_true = SCENARIOS[scenario_key]["pt"]
            maes = [logit_mae(est["mean"], pt_true) for est in estimates]

            scenario_results[case_name] = {
                "mean_estimate": np.mean([est["mean"] for est in estimates], axis=0),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
            }

        all_results[scenario_key] = scenario_results

    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("  [OK] Gamma sensitivity complete!")
    return all_results


def BrtaCFR_estimator_custom_sigma(c_t, d_t, F_paras, sigma_prior, lambda_scale=1.0):
    """BrtaCFR estimator with custom prior variance."""
    import pymc as pm

    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay

    F_k = gamma.cdf(np.arange(T + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)

    f_mat = np.zeros((T, T))
    for i in range(T):
        f_mat += np.diag(np.ones(T - i) * f_k[i], -i)
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

        lambda_param = pm.HalfCauchy("lambda", beta=lambda_scale)
        beta = pm.CustomDist("beta", beta_tilde, lambda_param, sigma_prior, T, logp=logp, size=T)
        p_t = pm.Deterministic("p_t", pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson("deaths", mu=mu_t, observed=d_t)

    try:
        with model:
            # Use default optimizer for better convergence and smooth estimates
            approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)

        idata = approx.sample(draws=1000, random_seed=2025)
        BrtaCFR_est = idata.posterior["p_t"].mean(dim=("chain", "draw")).values
        CrI = idata.posterior["p_t"].quantile([0.025, 0.975], dim=("chain", "draw")).values

        # Check for NaN in results
        if not np.all(np.isfinite(BrtaCFR_est)):
            raise ValueError("NaN in posterior estimates")

        return {"mean": BrtaCFR_est, "lower": CrI[0, :], "upper": CrI[1, :]}

    except Exception as e:
        # Fallback to mCFR if optimization fails
        warnings.warn(
            f"ADVI optimization failed (sigma={sigma_prior}): {str(e)}. Using mCFR fallback."
        )
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        return {"mean": mCFR_result, "lower": mCFR_result * 0.5, "upper": mCFR_result * 1.5}


def run_sensitivity_sigma(config, checkpoint_mgr, resume=False):
    """Sensitivity to prior variance sigma - analyze all 6 scenarios with per-replication checkpointing."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2b: SENSITIVITY - Prior Variance σ²")
    print("=" * 80)

    final_checkpoint_name = "sensitivity_sigma_final"

    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  [OK] Sigma sensitivity already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)

    n_reps = config["sensitivity_reps"]
    all_results = {}

    # Analyze all 6 scenarios
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        scenario_results = {}

        for case_name, sigma_val in SIGMA_SENSITIVITY.items():
            analysis_name = f"sens_sigma_{case_name}"
            completed = checkpoint_mgr.get_completed_replications(
                analysis_name, scenario_key, n_reps
            )
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)

            if len(completed) > 0:
                print(f"    Case {case_name}: Found {len(completed)}/{n_reps} replications")
            if len(pending) == 0:
                print(f"    [OK] {case_name} (σ={sigma_val}) complete, using cached results")
                estimates = list(completed.values())
            else:
                print(
                    f"    [RUNNING] {case_name} (σ={sigma_val}): Running {len(pending)} pending replications"
                )

                F_paras = (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)

                def run_and_save_sigma_replication(rep_idx):
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=20000)
                    result = BrtaCFR_estimator_custom_sigma(
                        data["CT"], data["dt"], F_paras, sigma_val
                    )
                    checkpoint_mgr.save_replication_result(
                        analysis_name, scenario_key, rep_idx, result
                    )
                    return result

                new_estimates = Parallel(n_jobs=config["n_jobs"])(
                    delayed(run_and_save_sigma_replication)(rep_idx)
                    for rep_idx in tqdm(pending, desc=f"      {case_name}")
                )

                estimates = list(completed.values()) + new_estimates

            pt_true = SCENARIOS[scenario_key]["pt"]
            maes = [logit_mae(est["mean"], pt_true) for est in estimates]

            scenario_results[case_name] = {
                "mean_estimate": np.mean([est["mean"] for est in estimates], axis=0),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
            }

        all_results[scenario_key] = scenario_results

    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("  [OK] Sigma sensitivity complete!")
    return all_results


def get_weibull_pmf(mean, variance, T):
    """Compute Weibull PMF with moment matching."""
    from scipy.optimize import fsolve

    def equations(params):
        k, lam = params
        eq1 = lam * gamma_func(1 + 1 / k) - mean
        eq2 = lam**2 * (gamma_func(1 + 2 / k) - (gamma_func(1 + 1 / k)) ** 2) - variance
        return [eq1, eq2]

    try:
        k, lam = fsolve(equations, [2.0, mean])
        if k <= 0 or lam <= 0:
            raise ValueError
        cdf_values = weibull_min.cdf(np.arange(T + 1), c=k, scale=lam)
        return np.diff(cdf_values)
    except:
        k = (mean / np.sqrt(variance)) ** 2
        lam = mean / gamma_func(1 + 1 / k)
        cdf_values = weibull_min.cdf(np.arange(T + 1), c=k, scale=lam)
        return np.diff(cdf_values)


def get_lognormal_pmf(mean, variance, T):
    """Compute Lognormal PMF with moment matching."""
    mu = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma = np.sqrt(np.log(1 + variance / mean**2))
    cdf_values = lognorm.cdf(np.arange(T + 1), s=sigma, scale=np.exp(mu))
    return np.diff(cdf_values)


def estimate_with_custom_pmf(c_t, d_t, f_k, lambda_scale=1.0):
    """BrtaCFR with custom delay distribution PMF."""
    import pymc as pm

    T = len(c_t)

    f_mat = np.zeros((T, T))
    for i in range(T):
        if i < len(f_k):
            f_mat += np.diag(np.ones(T - i) * f_k[i], -i)
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

        lambda_param = pm.HalfCauchy("lambda", beta=lambda_scale)
        beta = pm.CustomDist("beta", beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic("p_t", pm.math.sigmoid(beta))
        mu_t = pm.math.dot(fc_mat, p_t)
        pm.Poisson("deaths", mu=mu_t, observed=d_t)

    try:
        with model:
            # Use default optimizer for better convergence and smooth estimates
            approx = pm.fit(100000, method=pm.ADVI(random_seed=2025), progressbar=False)

        idata = approx.sample(draws=1000, random_seed=2025)
        BrtaCFR_est = idata.posterior["p_t"].mean(dim=("chain", "draw")).values
        CrI = idata.posterior["p_t"].quantile([0.025, 0.975], dim=("chain", "draw")).values

        # Check for NaN in results
        if not np.all(np.isfinite(BrtaCFR_est)):
            raise ValueError("NaN in posterior estimates")

        return {"mean": BrtaCFR_est, "lower": CrI[0, :], "upper": CrI[1, :]}

    except Exception as e:
        warnings.warn(f"ADVI optimization failed: {str(e)}. Using mCFR fallback.")
        mCFR_result = mCFR_EST(c_t, d_t, f_k)
        return {"mean": mCFR_result, "lower": mCFR_result * 0.5, "upper": mCFR_result * 1.5}


def run_sensitivity_distribution(config, checkpoint_mgr, resume=False):
    """Sensitivity to different delay distributions - analyze all 6 scenarios with per-replication checkpointing."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2c: SENSITIVITY - Delay Distributions")
    print("=" * 80)

    final_checkpoint_name = "sensitivity_dist_final"

    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  [OK] Distribution sensitivity already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)

    n_reps = config["sensitivity_reps"]
    all_results = {}

    # Calculate true mean and variance
    true_scale = TRUE_GAMMA_MEAN / TRUE_GAMMA_SHAPE
    true_variance = TRUE_GAMMA_SHAPE * (true_scale**2)

    dist_cases = {
        "Gamma": {"type": "gamma"},
        "Weibull": {"type": "weibull"},
        "Lognormal": {"type": "lognormal"},
    }

    # Analyze all 6 scenarios
    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        scenario_results = {}

        for case_name, case_info in dist_cases.items():
            analysis_name = f"sens_dist_{case_name}"
            completed = checkpoint_mgr.get_completed_replications(
                analysis_name, scenario_key, n_reps
            )
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)

            if len(completed) > 0:
                print(f"    Case {case_name}: Found {len(completed)}/{n_reps} replications")
            if len(pending) == 0:
                print(f"    [OK] {case_name} complete, using cached results")
                estimates = list(completed.values())
            else:
                print(f"    [RUNNING] {case_name}: Running {len(pending)} pending replications")

                # Generate PMF once if needed
                if case_info["type"] == "weibull":
                    pmf = get_weibull_pmf(TRUE_GAMMA_MEAN, true_variance, T_PERIOD)
                elif case_info["type"] == "lognormal":
                    pmf = get_lognormal_pmf(TRUE_GAMMA_MEAN, true_variance, T_PERIOD)

                def run_and_save_dist_replication(rep_idx):
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=30000)

                    if case_info["type"] == "gamma":
                        F_paras = (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
                        result = BrtaCFR_estimator(data["CT"], data["dt"], F_paras)
                    else:
                        result = estimate_with_custom_pmf(data["CT"], data["dt"], pmf)

                    checkpoint_mgr.save_replication_result(
                        analysis_name, scenario_key, rep_idx, result
                    )
                    return result

                new_estimates = Parallel(n_jobs=config["n_jobs"])(
                    delayed(run_and_save_dist_replication)(rep_idx)
                    for rep_idx in tqdm(pending, desc=f"      {case_name}")
                )

                estimates = list(completed.values()) + new_estimates

            pt_true = SCENARIOS[scenario_key]["pt"]
            maes = [logit_mae(est["mean"], pt_true) for est in estimates]

            scenario_results[case_name] = {
                "mean_estimate": np.mean([est["mean"] for est in estimates], axis=0),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
            }

        all_results[scenario_key] = scenario_results

    checkpoint_mgr.save_verbose(final_checkpoint_name, all_results)
    print("  [OK] Distribution sensitivity complete!")
    return all_results


# =============================================================================
# R4-1: Lambda (half-Cauchy scale) sensitivity and prior predictive
# =============================================================================


def run_lambda_sensitivity(config, checkpoint_mgr, lambda_scales, output_dir, resume=False):
    """Sensitivity to half-Cauchy scale s for lambda prior. One row per scenario, two panels: trajectory + MAE bars."""
    print("\n" + "=" * 80)
    print("R4-1: LAMBDA SCALE SENSITIVITY (Simulations)")
    print("=" * 80)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    final_name = "lambda_sensitivity_final"

    if not resume and checkpoint_mgr.exists(final_name):
        print("  [OK] Lambda sensitivity already complete! Loading...")
        all_results = checkpoint_mgr.load(final_name)
        _plot_lambda_sensitivity(all_results, lambda_scales, output_dir)
        _save_lambda_mae_csv(all_results, lambda_scales, output_dir)
        return all_results

    n_reps = config["sensitivity_reps"]
    all_results = {}

    for scenario_key in SCENARIOS.keys():
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        pt_true = SCENARIOS[scenario_key]["pt"]
        scenario_results = {}

        for s in lambda_scales:
            case_name = f"s_{s:.2f}".replace(".", "_")
            analysis_name = f"lambda_sens_{case_name}"
            completed = checkpoint_mgr.get_completed_replications(
                analysis_name, scenario_key, n_reps
            )
            pending = checkpoint_mgr.get_pending_replications(analysis_name, scenario_key, n_reps)

            if len(pending) == 0:
                estimates = list(completed.values())
            else:

                def run_and_save(rep_idx):
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=50000)
                    result = BrtaCFR_estimator(
                        data["CT"], data["dt"], (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE), lambda_scale=s
                    )
                    checkpoint_mgr.save_replication_result(
                        analysis_name, scenario_key, rep_idx, result
                    )
                    return result

                new_estimates = Parallel(n_jobs=config["n_jobs"])(
                    delayed(run_and_save)(rep_idx) for rep_idx in tqdm(pending, desc=f"    s={s}")
                )
                estimates = list(completed.values()) + new_estimates

            maes = [logit_mae(est["mean"], pt_true) for est in estimates]
            scenario_results[s] = {
                "mean_estimate": np.mean([est["mean"] for est in estimates], axis=0),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "mae_se": np.std(maes) / np.sqrt(len(maes)) if len(maes) > 0 else 0.0,
            }
        all_results[scenario_key] = scenario_results

    checkpoint_mgr.save_verbose(final_name, all_results)
    _plot_lambda_sensitivity(all_results, lambda_scales, output_dir)
    _save_lambda_mae_csv(all_results, lambda_scales, output_dir)
    print("  [OK] Lambda sensitivity complete!")
    return all_results


def _save_lambda_mae_csv(all_results, lambda_scales, output_dir):
    rows = []
    for scenario_key, scenario_results in all_results.items():
        for s in lambda_scales:
            r = scenario_results.get(s, {})
            rows.append(
                {
                    "scenario": scenario_key,
                    "s": s,
                    "mae_mean": r.get("mae_mean"),
                    "mae_se": r.get("mae_se"),
                }
            )
    pd.DataFrame(rows).to_csv(Path(output_dir) / "mae_by_lambda_scale.csv", index=False)


def _plot_lambda_sensitivity(all_results, lambda_scales, output_dir):
    """One figure: 3x4 layout (same as sensitivity_sigma), trajectory | MAE per scenario; legend only top-left."""
    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.titlesize": 24,
            "axes.labelsize": 22,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 16,
            "axes.linewidth": 2,
            "grid.linewidth": 1.5,
            "xtick.major.width": 2.5,
            "ytick.major.width": 2.5,
            "xtick.major.size": 10,
            "ytick.major.size": 10,
        }
    )
    fig, axes = plt.subplots(3, 4, figsize=(28, 21))
    lambda_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Format s for x-tick labels: s=0.1, s=0.5, s=1, s=5, s=10
    def s_label(s):
        return f"s={int(s)}" if s == int(s) else f"s={s:.1f}"

    for i, scenario_key in enumerate(SCENARIOS.keys()):
        row = i // 2
        col_curve = (i % 2) * 2
        col_mae = (i % 2) * 2 + 1
        ax_curve = axes[row, col_curve]
        ax_mae = axes[row, col_mae]
        pt_true = SCENARIOS[scenario_key]["pt"]
        ax_curve.plot(
            DAYS, pt_true, color="black", linestyle="-", linewidth=5, label="True", alpha=0.9
        )

        for j, s in enumerate(lambda_scales):
            r = all_results[scenario_key].get(s, {})
            if "mean_estimate" in r:
                ax_curve.plot(
                    DAYS,
                    r["mean_estimate"],
                    label=s_label(s),
                    color=lambda_colors[j % len(lambda_colors)],
                    linestyle="--",
                    linewidth=4,
                    alpha=0.9,
                )

        ax_curve.set_xlabel("Days", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_ylabel("Fatality Rate", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_title(
            f"({scenario_key}) {SCENARIOS[scenario_key]['name']}",
            fontsize=24,
            fontweight="bold",
            pad=14,
        )
        if i == 0:
            ax_curve.legend(fontsize=16, loc="best", framealpha=0.9, edgecolor="black")
        ax_curve.grid(True, alpha=0.5, linewidth=1.5)
        ax_curve.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

        mae_means = [
            all_results[scenario_key].get(s, {}).get("mae_mean", np.nan) for s in lambda_scales
        ]
        mae_ses = [all_results[scenario_key].get(s, {}).get("mae_se", 0) for s in lambda_scales]
        x_pos = np.arange(len(lambda_scales))
        ax_mae.bar(
            x_pos,
            mae_means,
            yerr=mae_ses,
            capsize=8,
            alpha=0.8,
            color=lambda_colors[: len(lambda_scales)],
            edgecolor="black",
            linewidth=2,
        )
        ax_mae.set_xticks(x_pos)
        ax_mae.set_xticklabels([s_label(s) for s in lambda_scales], fontsize=18, fontweight="bold")
        ax_mae.set_ylabel("Mean Absolute Error", fontsize=22, fontweight="bold", labelpad=12)
        ax_mae.set_title(
            f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=24, fontweight="bold", pad=14
        )
        ax_mae.grid(True, alpha=0.5, axis="y", linewidth=1.5)
        ax_mae.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

    plt.tight_layout(pad=1.5)
    fig_path = Path(output_dir) / "fig_lambda_scale_sensitivity.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {fig_path}")
    plt.rcParams.update(plt.rcParamsDefault)


def run_prior_predictive(lambda_scales, output_dir, n_draws=N_PRIOR_DRAWS, seed=2040):
    """Prior predictive check: implied smoothness of p_t (TV_p) per scenario, no inference."""
    print("\n" + "=" * 80)
    print("R4-1: PRIOR PREDICTIVE SMOOTHNESS (Simulations)")
    print("=" * 80)

    from scipy.stats import halfcauchy, laplace

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def expit(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    np.random.seed(seed)
    records = []

    for scenario_key in SCENARIOS.keys():
        pt_true = SCENARIOS[scenario_key]["pt"]
        T_scn = len(pt_true)
        beta0 = np.log(
            np.clip(np.mean(pt_true), 1e-10, 1 - 1e-10)
            / (1 - np.clip(np.mean(pt_true), 1e-10, 1 - 1e-10))
        )

        for s in lambda_scales:
            for draw_id in range(n_draws):
                lam = halfcauchy.rvs(scale=s)
                b = 1.0 / (lam + 1e-10)
                delta = laplace.rvs(scale=b, size=T_scn - 1)
                beta_t = np.empty(T_scn)
                beta_t[0] = beta0
                for t in range(1, T_scn):
                    beta_t[t] = beta_t[t - 1] + delta[t - 1]
                p_t = expit(beta_t)
                TV_p = np.sum(np.abs(np.diff(p_t)))
                records.append({"scenario": scenario_key, "s": s, "draw_id": draw_id, "TV_p": TV_p})

    df = pd.DataFrame(records)
    csv_path = Path(output_dir) / "prior_pred_smoothness.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved to: {csv_path}")

    # One figure: one subplot per scenario, x=s, y=distribution of TV_p (boxplot)
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.ravel()
    for i, scenario_key in enumerate(SCENARIOS.keys()):
        ax = axes[i]
        sub = df[df["scenario"] == scenario_key]
        positions = [np.where(np.array(lambda_scales) == s)[0][0] for s in lambda_scales]
        data = [sub[sub["s"] == s]["TV_p"].values for s in lambda_scales]
        bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax.set_xticks(range(len(lambda_scales)))
        ax.set_xticklabels([f"s={int(s)}" if s == int(s) else f"s={s}" for s in lambda_scales])
        ax.set_ylabel("TV(p)")
        ax.set_title(f"({scenario_key}) {SCENARIOS[scenario_key]['name']}", fontweight="bold")
        ax.grid(True, alpha=0.5, axis="y")
    plt.tight_layout()
    fig_path = Path(output_dir) / "fig_prior_predictive_smoothness.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {fig_path}")


# =============================================================================
# Analysis 3: MCMC vs ADVI Comparison
# =============================================================================


def run_brtacfr_mcmc(c_t, d_t, F_paras, n_samples=500, n_chains=2, tune=500, lambda_scale=1.0):
    """
    Run BrtaCFR with MCMC using NUTS sampler.

    Conservative settings for stability:
    - 500 samples (sufficient for comparison)
    - 500 burn-in (tune) iterations
    - 2 chains for faster computation
    - NUTS sampler (default, more robust than MH)
    """
    # Lazy imports so plotting-only runs can succeed without PyMC stack.
    import pymc as pm
    import arviz as az

    T = len(c_t)
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay

    F_k = gamma.cdf(np.arange(T + 1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)

    f_mat = np.zeros((T, T))
    for i in range(T):
        f_mat += np.diag(np.ones(T - i) * f_k[i], -i)
    c_mat = np.diag(c_t)
    fc_mat = np.dot(f_mat, c_mat)

    cCFR_est = np.cumsum(d_t) / (np.cumsum(c_t) + 1e-10)

    with pm.Model() as model:
        beta_tilde = np.log((cCFR_est + 1e-10) / (1 - (cCFR_est + 1e-10)))
        lambda_param = pm.HalfCauchy("lambda", beta=lambda_scale)
        beta = pm.CustomDist("beta", beta_tilde, lambda_param, 5, T, logp=logp, size=T)
        p_t = pm.Deterministic("p_t", pm.math.sigmoid(beta))
        mu_t = pm.Deterministic("mu_t", pm.math.dot(fc_mat, p_t))
        pm.Poisson("deaths", mu=mu_t, observed=d_t)

    with model:
        idata = pm.sample(
            draws=n_samples,
            tune=tune,
            chains=n_chains,
            random_seed=2025,
            progressbar=False,
            return_inferencedata=True,
        )

    BrtaCFR_est = idata.posterior["p_t"].mean(dim=("chain", "draw")).values
    CrI = idata.posterior["p_t"].quantile([0.025, 0.975], dim=("chain", "draw")).values
    p_samples = idata.posterior["p_t"].values.reshape(-1, T)
    mu_samples = idata.posterior["mu_t"].values.reshape(-1, T)
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

    try:
        rhat = az.rhat(idata, var_names=["p_t"])["p_t"].mean().values
        ess_bulk = az.ess(idata, var_names=["p_t"], method="bulk")["p_t"].mean().values
        n_divergences = idata.sample_stats["diverging"].sum().values
    except:
        rhat, ess_bulk, n_divergences = None, None, None

    return {
        "mean": BrtaCFR_est,
        "lower": CrI[0, :],
        "upper": CrI[1, :],
        "rhat": rhat,
        "ess_bulk": ess_bulk,
        "n_divergences": n_divergences,
        "pt_cri": pt_cri,
        "mut_cri": mut_cri,
    }


def run_mcmc_comparison(config, checkpoint_mgr, resume=False):
    """MCMC vs ADVI comparison with per-replication checkpointing."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: MCMC vs ADVI Comparison")
    print("=" * 80)

    final_checkpoint_name = "mcmc_comparison_final"

    # Check if analysis is fully complete
    if not resume and checkpoint_mgr.exists(final_checkpoint_name):
        print("  [OK] MCMC comparison already complete! Loading existing results...")
        return checkpoint_mgr.load(final_checkpoint_name)

    n_reps = config["mcmc_reps"]
    # Use all scenarios for comprehensive comparison
    test_scenarios = list(SCENARIOS.keys())  # All 6 scenarios: A, B, C, D, E, F

    comparison_results = {}

    for scenario_key in test_scenarios:
        print(f"\n  Scenario {scenario_key}: {SCENARIOS[scenario_key]['name']}")
        pt_true = SCENARIOS[scenario_key]["pt"]

        # Check MCMC checkpoints
        mcmc_completed = checkpoint_mgr.get_completed_replications("mcmc", scenario_key, n_reps)
        mcmc_pending = checkpoint_mgr.get_pending_replications("mcmc", scenario_key, n_reps)

        if len(mcmc_completed) > 0:
            print(f"    MCMC: Found {len(mcmc_completed)}/{n_reps} replications")
        if len(mcmc_pending) > 0:
            print(f"    [RUNNING] Running {len(mcmc_pending)} pending MCMC replications...")
            print(f"    Note: MCMC runs sequentially (parallel MCMC can cause memory issues)")

            # Run MCMC sequentially (parallel MCMC causes memory/threading issues)
            for rep_idx in tqdm(mcmc_pending, desc="    MCMC"):
                try:
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=40000)
                    mcmc_start = time.time()
                    result = run_brtacfr_mcmc(
                        data["CT"], data["dt"], (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
                    )
                    runtime = time.time() - mcmc_start
                    result_dict = {
                        "success": True,
                        "runtime": runtime,
                        "mae": logit_mae(result["mean"], pt_true),
                        "coverage": np.mean(
                            (pt_true >= result["lower"]) & (pt_true <= result["upper"])
                        ),
                        "pt_cri": result.get("pt_cri"),
                        "mut_cri": result.get("mut_cri"),
                    }
                    checkpoint_mgr.save_replication_result(
                        "mcmc", scenario_key, rep_idx, result_dict
                    )
                except Exception as e:
                    print(f"      [ERROR] MCMC failed for {scenario_key} rep {rep_idx}: {str(e)}")
                    result_dict = {"success": False, "error": str(e)}
                    checkpoint_mgr.save_replication_result(
                        "mcmc", scenario_key, rep_idx, result_dict
                    )
        else:
            print(f"    [OK] MCMC complete, using cached results")

        # Check ADVI checkpoints
        advi_completed = checkpoint_mgr.get_completed_replications("advi", scenario_key, n_reps)
        advi_pending = checkpoint_mgr.get_pending_replications("advi", scenario_key, n_reps)

        if len(advi_completed) > 0:
            print(f"    ADVI: Found {len(advi_completed)}/{n_reps} replications")
        if len(advi_pending) > 0:
            print(f"    [RUNNING] Running {len(advi_pending)} pending ADVI replications...")
            print(f"    Note: ADVI runs sequentially (like old version)")

            # Run ADVI sequentially (like old version)
            for rep_idx in tqdm(advi_pending, desc="    ADVI"):
                try:
                    data = generate_simulation_data(scenario_key, rep_idx, seed_offset=40000)
                    advi_start = time.time()
                    result = BrtaCFR_estimator(
                        data["CT"], data["dt"], (TRUE_GAMMA_MEAN, TRUE_GAMMA_SHAPE)
                    )
                    runtime = time.time() - advi_start
                    result_dict = {
                        "success": True,
                        "runtime": runtime,
                        "mae": logit_mae(result["mean"], pt_true),
                        "coverage": np.mean(
                            (pt_true >= result["lower"]) & (pt_true <= result["upper"])
                        ),
                        "pt_cri": result.get("pt_cri"),
                        "mut_cri": result.get("mut_cri"),
                    }
                    checkpoint_mgr.save_replication_result(
                        "advi", scenario_key, rep_idx, result_dict
                    )
                except Exception as e:
                    print(f"      [ERROR] ADVI failed for {scenario_key} rep {rep_idx}: {str(e)}")
                    result_dict = {"success": False, "error": str(e)}
                    checkpoint_mgr.save_replication_result(
                        "advi", scenario_key, rep_idx, result_dict
                    )
        else:
            print(f"    [OK] ADVI complete, using cached results")

        # Reload all completed results
        mcmc_all = checkpoint_mgr.get_completed_replications("mcmc", scenario_key, n_reps)
        advi_all = checkpoint_mgr.get_completed_replications("advi", scenario_key, n_reps)

        # Aggregate and report success rates
        mcmc_success = [r for r in mcmc_all.values() if r["success"]]
        advi_success = [r for r in advi_all.values() if r["success"]]

        print(f"    MCMC: {len(mcmc_success)}/{len(mcmc_all)} successful")
        print(f"    ADVI: {len(advi_success)}/{len(advi_all)} successful")

        if len(mcmc_success) == 0:
            print(f"    [WARNING] No successful MCMC runs for scenario {scenario_key}")
        if len(advi_success) == 0:
            print(f"    [WARNING] No successful ADVI runs for scenario {scenario_key}")

        mu_t_true = TRUE_MU_T[scenario_key]
        mcmc_cov_pt, mcmc_cov_mut = (
            average_curve_level_coverage(mcmc_success, pt_true, mu_t_true)
            if mcmc_success
            else ({L: None for L in COVERAGE_LEVELS}, {L: None for L in COVERAGE_LEVELS})
        )
        advi_cov_pt, advi_cov_mut = (
            average_curve_level_coverage(advi_success, pt_true, mu_t_true)
            if advi_success
            else ({L: None for L in COVERAGE_LEVELS}, {L: None for L in COVERAGE_LEVELS})
        )

        comparison_results[scenario_key] = {
            "mcmc_runtime_mean": (
                np.mean([r["runtime"] for r in mcmc_success]) if mcmc_success else None
            ),
            "mcmc_runtime_std": (
                np.std([r["runtime"] for r in mcmc_success]) if mcmc_success else None
            ),
            "advi_runtime_mean": (
                np.mean([r["runtime"] for r in advi_success]) if advi_success else None
            ),
            "advi_runtime_std": (
                np.std([r["runtime"] for r in advi_success]) if advi_success else None
            ),
            "speedup": (
                np.mean([r["runtime"] for r in mcmc_success])
                / np.mean([r["runtime"] for r in advi_success])
                if (mcmc_success and advi_success)
                else None
            ),
            "mcmc_mae_mean": np.mean([r["mae"] for r in mcmc_success]) if mcmc_success else None,
            "advi_mae_mean": np.mean([r["mae"] for r in advi_success]) if advi_success else None,
            "mcmc_coverage_mean": (
                np.mean([r["coverage"] for r in mcmc_success]) if mcmc_success else None
            ),
            "advi_coverage_mean": (
                np.mean([r["coverage"] for r in advi_success]) if advi_success else None
            ),
            "mcmc_coverage_pt_50": mcmc_cov_pt.get(0.50),
            "mcmc_coverage_pt_80": mcmc_cov_pt.get(0.80),
            "mcmc_coverage_pt_95": mcmc_cov_pt.get(0.95),
            "mcmc_coverage_mut_50": mcmc_cov_mut.get(0.50),
            "mcmc_coverage_mut_80": mcmc_cov_mut.get(0.80),
            "mcmc_coverage_mut_95": mcmc_cov_mut.get(0.95),
            "advi_coverage_pt_50": advi_cov_pt.get(0.50),
            "advi_coverage_pt_80": advi_cov_pt.get(0.80),
            "advi_coverage_pt_95": advi_cov_pt.get(0.95),
            "advi_coverage_mut_50": advi_cov_mut.get(0.50),
            "advi_coverage_mut_80": advi_cov_mut.get(0.80),
            "advi_coverage_mut_95": advi_cov_mut.get(0.95),
        }

    checkpoint_mgr.save_verbose(final_checkpoint_name, comparison_results)
    print("  [OK] MCMC comparison complete!")
    return comparison_results


# =============================================================================
# Plotting Functions
# =============================================================================

# Unified color and line style definitions
COLORS = {
    "True": "black",
    "cCFR": "#2ca02c",  # green
    "mCFR": "#ff7f0e",  # orange
    "BrtaCFR": "#d62728",  # red
    "CrI": "#1f77b4",  # blue
}

LINESTYLES = {
    "True": "-",
    "cCFR": "-",
    "mCFR": "-",
    "BrtaCFR": "--",  # dashed for BrtaCFR
}


def plot_main_analysis(main_results, output_dir):
    """Generate main analysis plots optimized for A4 print quality."""
    print("\n  Generating plots...")

    # Optimized for A4 print quality
    plt.rcParams.update(
        {
            "font.size": 18,  # Base font size (increased)
            "axes.titlesize": 22,  # Title font size
            "axes.labelsize": 20,  # Axis label font size
            "xtick.labelsize": 16,  # X-axis tick font size (increased)
            "ytick.labelsize": 16,  # Y-axis tick font size (increased)
            "legend.fontsize": 16,  # Legend font size (increased)
            "figure.titlesize": 24,  # Figure title font size
            "lines.linewidth": 4,  # Thicker lines for print
            "axes.linewidth": 1.5,  # Thicker axis borders
            "grid.linewidth": 1.2,  # Thicker grid lines
            "xtick.major.width": 2,  # Thicker tick marks
            "ytick.major.width": 2,
            "xtick.major.size": 8,  # Longer tick marks
            "ytick.major.size": 8,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(32, 16))  # Larger figure
    axes = axes.ravel()

    for i, (scenario_key, results) in enumerate(main_results.items()):
        ax = axes[i]
        pt_true = SCENARIOS[scenario_key]["pt"]

        # Plot with thicker lines for better print visibility
        ax.plot(
            DAYS,
            pt_true,
            color=COLORS["True"],
            linestyle=LINESTYLES["True"],
            linewidth=5,
            label="True",
            alpha=0.95,
        )
        ax.plot(
            DAYS,
            results["cCFR_avg"],
            color=COLORS["cCFR"],
            linestyle=LINESTYLES["cCFR"],
            linewidth=4,
            label="cCFR",
            alpha=0.95,
        )
        ax.plot(
            DAYS,
            results["mCFR_avg"],
            color=COLORS["mCFR"],
            linestyle=LINESTYLES["mCFR"],
            linewidth=4,
            label="mCFR",
            alpha=0.95,
        )
        ax.plot(
            DAYS,
            results["BrtaCFR_avg"],
            color=COLORS["BrtaCFR"],
            linestyle=LINESTYLES["BrtaCFR"],
            linewidth=5,
            label="BrtaCFR",
            alpha=0.95,
        )
        ax.fill_between(
            DAYS,
            results["BrtaCFR_lower_avg"],
            results["BrtaCFR_upper_avg"],
            color=COLORS["CrI"],
            alpha=0.25,
            label="95% CrI",
        )

        ax.set_title(
            f"({scenario_key}) {SCENARIOS[scenario_key]['name']}",
            fontsize=22,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Days", fontsize=20, fontweight="bold", labelpad=10)
        ax.set_ylabel("Fatality Rate", fontsize=20, fontweight="bold", labelpad=10)
        ax.grid(True, linestyle=":", alpha=0.6, linewidth=1.2)

        if i == 0:
            ax.legend(
                loc="upper right",
                fontsize=14,
                framealpha=0.95,
                edgecolor="black",
                frameon=True,
                bbox_to_anchor=(1.0, 1.0),
            )

    plt.tight_layout(pad=1.0)
    output_path = Path(output_dir) / "simulation.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {output_path}")
    plt.close()

    # Reset rcParams to default
    plt.rcParams.update(plt.rcParamsDefault)


def plot_sensitivity_results(gamma_results, sigma_results, dist_results, output_dir):
    """Plot all sensitivity analysis results with 3x4 layout for each analysis, optimized for A4 horizontal."""
    print("\n  Generating sensitivity plots...")

    # Set global font sizes for better publication quality - optimized for A4 horizontal
    plt.rcParams.update(
        {
            "font.size": 20,  # Base font size (increased for horizontal layout)
            "axes.titlesize": 24,  # Title font size
            "axes.labelsize": 22,  # Axis label font size
            "xtick.labelsize": 18,  # X-axis tick font size
            "ytick.labelsize": 18,  # Y-axis tick font size
            "legend.fontsize": 16,  # Legend font size
            "figure.titlesize": 26,  # Figure title font size
            "lines.linewidth": 4,  # Thicker lines
            "axes.linewidth": 2,  # Thicker axis borders
            "grid.linewidth": 1.5,  # Thicker grid lines
            "xtick.major.width": 2.5,  # Thicker tick marks
            "ytick.major.width": 2.5,
            "xtick.major.size": 10,  # Longer tick marks
            "ytick.major.size": 10,
        }
    )

    # Plot 1: Gamma sensitivity (3x4 layout)
    # Row 1: A_Curve, A_MAE, B_Curve, B_MAE
    # Row 2: C_Curve, C_MAE, D_Curve, D_MAE
    # Row 3: E_Curve, E_MAE, F_Curve, F_MAE
    fig, axes = plt.subplots(3, 4, figsize=(28, 21))  # Optimized for A4 horizontal

    gamma_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, scenario_key in enumerate(SCENARIOS.keys()):
        # Map to 3x4 layout: row = i // 2, col_curve = (i % 2) * 2, col_mae = (i % 2) * 2 + 1
        row = i // 2
        col_curve = (i % 2) * 2
        col_mae = (i % 2) * 2 + 1

        # Curves plot
        ax_curve = axes[row, col_curve]
        ax_curve.plot(
            DAYS,
            SCENARIOS[scenario_key]["pt"],
            color=COLORS["True"],
            linestyle="-",
            linewidth=5,
            label="True",
            alpha=0.9,
        )

        for j, (case_name, results) in enumerate(gamma_results[scenario_key].items()):
            ax_curve.plot(
                DAYS,
                results["mean_estimate"],
                label=case_name.replace("_", " "),
                color=gamma_colors[j],
                linestyle="--",
                linewidth=4,
                alpha=0.9,
            )

        ax_curve.set_xlabel("Days", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_ylabel("Fatality Rate", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_title(
            f"({scenario_key}) {SCENARIOS[scenario_key]['name']}",
            fontsize=24,
            fontweight="bold",
            pad=14,
        )
        if i == 0:  # Only show legend on first subplot
            ax_curve.legend(fontsize=16, loc="best", framealpha=0.9, edgecolor="black")
        ax_curve.grid(True, alpha=0.5, linewidth=1.5)
        ax_curve.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

        # MAE plot
        ax_mae = axes[row, col_mae]
        case_names = [n.replace("_", " ") for n in gamma_results[scenario_key].keys()]
        mae_means = [r["mae_mean"] for r in gamma_results[scenario_key].values()]
        mae_stds = [r["mae_std"] for r in gamma_results[scenario_key].values()]

        bars = ax_mae.bar(
            range(len(case_names)),
            mae_means,
            yerr=mae_stds,
            capsize=10,
            alpha=0.8,
            color=gamma_colors,
            edgecolor="black",
            linewidth=2,
        )
        ax_mae.set_xticks(range(len(case_names)))
        ax_mae.set_xticklabels(case_names, rotation=45, ha="right", fontsize=18, fontweight="bold")
        ax_mae.set_ylabel("Mean Absolute Error", fontsize=22, fontweight="bold", labelpad=12)
        ax_mae.set_title(
            f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=24, fontweight="bold", pad=14
        )
        ax_mae.grid(True, alpha=0.5, axis="y", linewidth=1.5)
        ax_mae.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

    plt.tight_layout(pad=1.5)
    output_path = Path(output_dir) / "sensitivity_gamma.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {output_path}")
    plt.close()

    # Plot 2: Sigma sensitivity (3x4 layout)
    # Row 1: A_Curve, A_MAE, B_Curve, B_MAE
    # Row 2: C_Curve, C_MAE, D_Curve, D_MAE
    # Row 3: E_Curve, E_MAE, F_Curve, F_MAE
    fig, axes = plt.subplots(3, 4, figsize=(28, 21))  # Optimized for A4 horizontal

    sigma_colors = ["#7f007f", "#0000ff", "#00ff00", "#ffa500", "#ff0000"]

    for i, scenario_key in enumerate(SCENARIOS.keys()):
        # Map to 3x4 layout: row = i // 2, col_curve = (i % 2) * 2, col_mae = (i % 2) * 2 + 1
        row = i // 2
        col_curve = (i % 2) * 2
        col_mae = (i % 2) * 2 + 1

        # Curves plot
        ax_curve = axes[row, col_curve]
        ax_curve.plot(
            DAYS,
            SCENARIOS[scenario_key]["pt"],
            color=COLORS["True"],
            linestyle="-",
            linewidth=5,
            label="True",
            alpha=0.9,
        )

        for j, (case_name, results) in enumerate(sigma_results[scenario_key].items()):
            sigma_val = SIGMA_SENSITIVITY[case_name]
            ax_curve.plot(
                DAYS,
                results["mean_estimate"],
                label=f"σ={sigma_val}",
                color=sigma_colors[j],
                linestyle="--",
                linewidth=4,
                alpha=0.9,
            )

        ax_curve.set_xlabel("Days", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_ylabel("Fatality Rate", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_title(
            f"({scenario_key}) {SCENARIOS[scenario_key]['name']}",
            fontsize=24,
            fontweight="bold",
            pad=14,
        )
        if i == 0:  # Only show legend on first subplot
            ax_curve.legend(fontsize=16, loc="best", framealpha=0.9, edgecolor="black")
        ax_curve.grid(True, alpha=0.5, linewidth=1.5)
        ax_curve.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

        # MAE plot
        ax_mae = axes[row, col_mae]
        sigma_labels = [f"σ={SIGMA_SENSITIVITY[n]}" for n in sigma_results[scenario_key].keys()]
        mae_means = [r["mae_mean"] for r in sigma_results[scenario_key].values()]
        mae_stds = [r["mae_std"] for r in sigma_results[scenario_key].values()]

        bars = ax_mae.bar(
            range(len(sigma_labels)),
            mae_means,
            yerr=mae_stds,
            capsize=10,
            alpha=0.8,
            color=sigma_colors,
            edgecolor="black",
            linewidth=2,
        )
        ax_mae.set_xticks(range(len(sigma_labels)))
        ax_mae.set_xticklabels(sigma_labels, fontsize=18, fontweight="bold")
        ax_mae.set_ylabel("Mean Absolute Error", fontsize=22, fontweight="bold", labelpad=12)
        ax_mae.set_title(
            f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=24, fontweight="bold", pad=14
        )
        ax_mae.grid(True, alpha=0.5, axis="y", linewidth=1.5)
        ax_mae.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

    plt.tight_layout(pad=1.5)
    output_path = Path(output_dir) / "sensitivity_sigma.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {output_path}")
    plt.close()

    # Plot 3: Distribution sensitivity (3x4 layout)
    # Row 1: A_Curve, A_MAE, B_Curve, B_MAE
    # Row 2: C_Curve, C_MAE, D_Curve, D_MAE
    # Row 3: E_Curve, E_MAE, F_Curve, F_MAE
    fig, axes = plt.subplots(3, 4, figsize=(28, 21))  # Optimized for A4 horizontal

    dist_colors = ["#d62728", "#1f77b4", "#2ca02c"]

    for i, scenario_key in enumerate(SCENARIOS.keys()):
        # Map to 3x4 layout: row = i // 2, col_curve = (i % 2) * 2, col_mae = (i % 2) * 2 + 1
        row = i // 2
        col_curve = (i % 2) * 2
        col_mae = (i % 2) * 2 + 1

        # Curves plot
        ax_curve = axes[row, col_curve]
        ax_curve.plot(
            DAYS,
            SCENARIOS[scenario_key]["pt"],
            color=COLORS["True"],
            linestyle="-",
            linewidth=5,
            label="True",
            alpha=0.9,
        )

        for j, (case_name, results) in enumerate(dist_results[scenario_key].items()):
            ax_curve.plot(
                DAYS,
                results["mean_estimate"],
                label=case_name,
                color=dist_colors[j],
                linestyle="--",
                linewidth=4,
                alpha=0.9,
            )

        ax_curve.set_xlabel("Days", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_ylabel("Fatality Rate", fontsize=22, fontweight="bold", labelpad=12)
        ax_curve.set_title(
            f"({scenario_key}) {SCENARIOS[scenario_key]['name']}",
            fontsize=24,
            fontweight="bold",
            pad=14,
        )
        if i == 0:  # Only show legend on first subplot
            ax_curve.legend(fontsize=16, loc="best", framealpha=0.9, edgecolor="black")
        ax_curve.grid(True, alpha=0.5, linewidth=1.5)
        ax_curve.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

        # MAE plot
        ax_mae = axes[row, col_mae]
        dist_names = list(dist_results[scenario_key].keys())
        mae_means = [r["mae_mean"] for r in dist_results[scenario_key].values()]
        mae_stds = [r["mae_std"] for r in dist_results[scenario_key].values()]

        bars = ax_mae.bar(
            range(len(dist_names)),
            mae_means,
            yerr=mae_stds,
            capsize=10,
            alpha=0.8,
            color=dist_colors,
            edgecolor="black",
            linewidth=2,
        )
        ax_mae.set_xticks(range(len(dist_names)))
        ax_mae.set_xticklabels(dist_names, fontsize=18, fontweight="bold")
        ax_mae.set_ylabel("Mean Absolute Error", fontsize=22, fontweight="bold", labelpad=12)
        ax_mae.set_title(
            f'MAE - {SCENARIOS[scenario_key]["name"]}', fontsize=24, fontweight="bold", pad=14
        )
        ax_mae.grid(True, alpha=0.5, axis="y", linewidth=1.5)
        ax_mae.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)

    plt.tight_layout(pad=1.5)
    output_path = Path(output_dir) / "sensitivity_distributions.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {output_path}")
    plt.close()

    # Reset rcParams to default
    plt.rcParams.update(plt.rcParamsDefault)

    # Save comprehensive summary CSV
    summary_data = []
    for scenario_key in SCENARIOS.keys():
        for case_name, results in gamma_results[scenario_key].items():
            summary_data.append(
                {
                    "Analysis": "Gamma",
                    "Scenario": scenario_key,
                    "Case": case_name,
                    "MAE_Mean": results["mae_mean"],
                    "MAE_SD": results["mae_std"],
                }
            )
        for case_name, results in sigma_results[scenario_key].items():
            summary_data.append(
                {
                    "Analysis": "Sigma",
                    "Scenario": scenario_key,
                    "Case": case_name,
                    "MAE_Mean": results["mae_mean"],
                    "MAE_SD": results["mae_std"],
                }
            )
        for case_name, results in dist_results[scenario_key].items():
            summary_data.append(
                {
                    "Analysis": "Distribution",
                    "Scenario": scenario_key,
                    "Case": case_name,
                    "MAE_Mean": results["mae_mean"],
                    "MAE_SD": results["mae_std"],
                }
            )

    df = pd.DataFrame(summary_data)
    csv_path = Path(output_dir) / "sensitivity_analysis_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved to: {csv_path}")


def plot_mcmc_comparison(mcmc_results, output_dir):
    """Plot MCMC vs ADVI comparison for all scenarios with Pareto analysis."""
    print("\n  Generating MCMC comparison plots...")

    # Set global font sizes for better publication quality
    plt.rcParams.update(
        {
            "font.size": 16,  # Base font size
            "axes.titlesize": 18,  # Title font size
            "axes.labelsize": 16,  # Axis label font size
            "xtick.labelsize": 14,  # X-axis tick font size
            "ytick.labelsize": 14,  # Y-axis tick font size
            "legend.fontsize": 14,  # Legend font size
            "figure.titlesize": 20,  # Figure title font size
            "lines.linewidth": 3,  # Default line width
        }
    )

    scenarios = list(mcmc_results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # Colors for consistency
    mcmc_color = "#1f77b4"
    advi_color = "#ff7f0e"

    # Plot 1: Runtime Comparison (A)
    ax1 = axes[0, 0]
    x = np.arange(len(scenarios))
    width = 0.35

    mcmc_times = [mcmc_results[s]["mcmc_runtime_mean"] for s in scenarios]
    advi_times = [mcmc_results[s]["advi_runtime_mean"] for s in scenarios]
    mcmc_errs = [mcmc_results[s]["mcmc_runtime_std"] for s in scenarios]
    advi_errs = [mcmc_results[s]["advi_runtime_std"] for s in scenarios]

    # If inference failed (e.g., PyMC stack unavailable/mismatched) these may be None.
    # Fallback: re-plot from the previously exported CSV if present.
    if any(v is None for v in (mcmc_times + advi_times + mcmc_errs + advi_errs)):
        csv_path = Path(output_dir) / "mcmc_vs_advi_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            scenarios = df["Scenario"].tolist()
            x = np.arange(len(scenarios))
            mcmc_times = df["MCMC_Runtime_Mean"].tolist()
            advi_times = df["ADVI_Runtime_Mean"].tolist()
            mcmc_errs = df["MCMC_Runtime_SD"].tolist()
            advi_errs = df["ADVI_Runtime_SD"].tolist()
            # MAE values also used below
            mcmc_maes_fallback = df["MCMC_MAE"].tolist()
            advi_maes_fallback = df["ADVI_MAE"].tolist()
        else:
            raise RuntimeError(
                "MCMC/ADVI results are missing (None) and fallback CSV does not exist. "
                "Cannot plot mcmc_vs_advi_comparison."
            )

    ax1.bar(
        x - width / 2,
        mcmc_times,
        width,
        yerr=mcmc_errs,
        label="MCMC",
        alpha=0.8,
        capsize=8,
        color=mcmc_color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.bar(
        x + width / 2,
        advi_times,
        width,
        yerr=advi_errs,
        label="ADVI",
        alpha=0.8,
        capsize=8,
        color=advi_color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax1.set_xlabel("Scenario", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Runtime (seconds)", fontsize=16, fontweight="bold")
    ax1.set_title("(A) Runtime Comparison", fontsize=18, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=14, framealpha=0.9, edgecolor="black")
    ax1.grid(True, alpha=0.5, axis="y", linewidth=1.5)
    ax1.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)

    # Plot 2: Speedup (without text annotations)
    ax2 = axes[0, 1]
    if "mcmc_maes_fallback" in locals():
        speedups = df["Speedup"].tolist()
    else:
        speedups = [mcmc_results[s]["speedup"] for s in scenarios]
    bars = ax2.bar(
        scenarios, speedups, alpha=0.8, color="#2ca02c", edgecolor="black", linewidth=1.5
    )
    ax2.axhline(y=1, color="r", linestyle="--", linewidth=3, label="No speedup")
    ax2.set_xlabel("Scenario", fontsize=16, fontweight="bold")
    ax2.set_ylabel("Speedup Factor", fontsize=16, fontweight="bold")
    ax2.set_title("(B) ADVI Speedup", fontsize=18, fontweight="bold")
    ax2.set_xticklabels(scenarios, fontsize=14, fontweight="bold")
    ax2.legend(fontsize=14, framealpha=0.9, edgecolor="black")
    ax2.grid(True, alpha=0.5, axis="y", linewidth=1.5)
    ax2.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
    # Note: No text annotations as requested

    # Plot 3: MAE Comparison (no MCMC/ADVI legend)
    ax3 = axes[1, 0]
    if "mcmc_maes_fallback" in locals():
        mcmc_maes = mcmc_maes_fallback
        advi_maes = advi_maes_fallback
    else:
        mcmc_maes = [mcmc_results[s]["mcmc_mae_mean"] for s in scenarios]
        advi_maes = [mcmc_results[s]["advi_mae_mean"] for s in scenarios]
    ax3.bar(
        x - width / 2,
        mcmc_maes,
        width,
        alpha=0.8,
        color=mcmc_color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.bar(
        x + width / 2,
        advi_maes,
        width,
        alpha=0.8,
        color=advi_color,
        edgecolor="black",
        linewidth=1.5,
    )
    ax3.set_xlabel("Scenario", fontsize=16, fontweight="bold")
    ax3.set_ylabel("Mean Absolute Error", fontsize=16, fontweight="bold")
    ax3.set_title("(C) Accuracy Comparison", fontsize=18, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios, fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.5, axis="y", linewidth=1.5)
    ax3.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)

    # Plot 4: Pareto Analysis (Time vs Accuracy)
    ax4 = axes[1, 1]

    # Plot MCMC points
    ax4.scatter(
        mcmc_times,
        mcmc_maes,
        s=200,
        alpha=0.8,
        color=mcmc_color,
        marker="o",
        label="MCMC",
        edgecolors="black",
        linewidths=2,
    )

    # Plot ADVI points
    ax4.scatter(
        advi_times,
        advi_maes,
        s=200,
        alpha=0.8,
        color=advi_color,
        marker="s",
        label="ADVI",
        edgecolors="black",
        linewidths=2,
    )

    # Add scenario labels
    for i, scenario in enumerate(scenarios):
        ax4.annotate(
            scenario,
            (mcmc_times[i], mcmc_maes[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )
        ax4.annotate(
            scenario,
            (advi_times[i], advi_maes[i]),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )

    # Add ideal direction arrow
    ax4.annotate(
        "",
        xy=(0.05, 0.05),
        xytext=(0.95, 0.95),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=3, color="gray", alpha=0.7),
    )
    ax4.text(
        0.1,
        0.9,
        "Better\n(Lower time, Lower MAE)",
        transform=ax4.transAxes,
        fontsize=12,
        color="gray",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="black", linewidth=1),
    )

    ax4.set_xlabel("Runtime (seconds)", fontsize=16, fontweight="bold")
    ax4.set_ylabel("Mean Absolute Error", fontsize=16, fontweight="bold")
    ax4.set_title("(D) Pareto Analysis: Time vs Accuracy", fontsize=18, fontweight="bold")
    ax4.legend(fontsize=14, loc="upper right", framealpha=0.9, edgecolor="black")
    ax4.grid(True, alpha=0.5, linewidth=1.5)
    ax4.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)

    # Helper: draw nominal coverage lines (red dashed, same as panel B); no legend text for dashed
    def _add_nominal_coverage_lines(ax, add_text=False):
        for level in COVERAGE_LEVELS:
            ax.axhline(y=level, color="r", linestyle="--", linewidth=1.5, zorder=10)
        if add_text:
            ax.text(
                0.02,
                0.98,
                "Dashed: nominal (0.5/0.8/0.95)",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                style="italic",
            )
        ax.set_ylim(0, 1)

    # Colors per level (50/80/95) so groups are distinguishable; MCMC=solid, ADVI=hatched
    level_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green for 50%, 80%, 95%

    # Plot 5 (E): Coverage comparison for p_t
    ax5 = axes[0, 2]
    has_coverage = all(mcmc_results[s].get("mcmc_coverage_pt_50") is not None for s in scenarios)
    if has_coverage:
        n_scen = len(scenarios)
        bar_w = 0.11
        x_centers = np.arange(n_scen)
        level_offsets = [-2 * bar_w, 0, 2 * bar_w]
        for i, level in enumerate(COVERAGE_LEVELS):
            base = level_offsets[i]
            mcmc_vals = [
                np.clip(mcmc_results[s].get(f"mcmc_coverage_pt_{int(level*100)}"), 0, 1) or np.nan
                for s in scenarios
            ]
            advi_vals = [
                np.clip(mcmc_results[s].get(f"advi_coverage_pt_{int(level*100)}"), 0, 1) or np.nan
                for s in scenarios
            ]
            ax5.bar(
                x_centers + base - bar_w / 2,
                mcmc_vals,
                bar_w,
                label=f"{int(level*100)}% CrI" if i == 0 else None,
                color=level_colors[i],
                edgecolor="black",
                linewidth=1,
            )
            ax5.bar(
                x_centers + base + bar_w / 2,
                advi_vals,
                bar_w,
                color=level_colors[i],
                edgecolor="black",
                linewidth=1,
                hatch="///",
            )
        ax5.set_xticks(x_centers)
        ax5.set_xticklabels(scenarios, fontsize=14, fontweight="bold")
        ax5.set_ylabel("Average curve-level coverage", fontsize=16, fontweight="bold")
        ax5.set_xlabel("Scenario", fontsize=16, fontweight="bold")
        ax5.set_title("(E) Coverage: " + r"$p_t$", fontsize=18, fontweight="bold")
        # Legend: levels (50/80/95% CrI) + MCMC/ADVI; lower right so it doesn't block top; only E shows MCMC/ADVI
        from matplotlib.patches import Patch

        handles_level = [Patch(facecolor=level_colors[i], edgecolor="black") for i in range(3)]
        labels_level = [f"{int(COVERAGE_LEVELS[i]*100)}% CrI" for i in range(3)]
        handles_method = [
            Patch(facecolor="gray", edgecolor="black"),
            Patch(facecolor="gray", edgecolor="black", hatch="///"),
        ]
        labels_method = ["MCMC", "ADVI"]
        ax5.legend(
            handles_level + handles_method,
            labels_level + labels_method,
            fontsize=11,
            framealpha=0.9,
            edgecolor="black",
            loc="lower right",
        )
        _add_nominal_coverage_lines(ax5, add_text=False)
        ax5.grid(True, alpha=0.5, axis="y")
        ax5.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
    else:
        ax5.set_title("(E) Coverage: " + r"$p_t$", fontsize=18, fontweight="bold")
        ax5.text(
            0.5,
            0.5,
            "No per-level coverage data",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=14,
        )
        ax5.set_ylim(0, 1)

    # Plot 6 (F): Coverage comparison for mu_t
    ax6 = axes[1, 2]
    if has_coverage:
        for i, level in enumerate(COVERAGE_LEVELS):
            base = level_offsets[i]
            mcmc_vals = [
                np.clip(mcmc_results[s].get(f"mcmc_coverage_mut_{int(level*100)}"), 0, 1) or np.nan
                for s in scenarios
            ]
            advi_vals = [
                np.clip(mcmc_results[s].get(f"advi_coverage_mut_{int(level*100)}"), 0, 1) or np.nan
                for s in scenarios
            ]
            ax6.bar(
                x_centers + base - bar_w / 2,
                mcmc_vals,
                bar_w,
                color=level_colors[i],
                edgecolor="black",
                linewidth=1,
            )
            ax6.bar(
                x_centers + base + bar_w / 2,
                advi_vals,
                bar_w,
                color=level_colors[i],
                edgecolor="black",
                linewidth=1,
                hatch="///",
            )
        ax6.set_xticks(x_centers)
        ax6.set_xticklabels(scenarios, fontsize=14, fontweight="bold")
        ax6.set_ylabel("Average curve-level coverage", fontsize=16, fontweight="bold")
        ax6.set_xlabel("Scenario", fontsize=16, fontweight="bold")
        ax6.set_title("(F) Coverage: " + r"$\mu_t$", fontsize=18, fontweight="bold")
        # F: no legend (levels + MCMC/ADVI shown in E only)
        _add_nominal_coverage_lines(ax6, add_text=False)
        ax6.grid(True, alpha=0.5, axis="y")
        ax6.tick_params(axis="both", which="major", labelsize=14, width=2, length=6)
    else:
        ax6.set_title("(F) Coverage: " + r"$\mu_t$", fontsize=18, fontweight="bold")
        ax6.text(
            0.5,
            0.5,
            "No per-level coverage data",
            ha="center",
            va="center",
            transform=ax6.transAxes,
            fontsize=14,
        )
        ax6.set_ylim(0, 1)

    plt.tight_layout()
    output_path = Path(output_dir) / "mcmc_vs_advi_comparison.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {output_path}")
    plt.close()

    # Reset rcParams to default
    plt.rcParams.update(plt.rcParamsDefault)

    # Save comprehensive CSV
    comparison_data = []
    for scenario in scenarios:
        comparison_data.append(
            {
                "Scenario": scenario,
                "Scenario_Name": SCENARIOS[scenario]["name"],
                "MCMC_Runtime_Mean": mcmc_results[scenario]["mcmc_runtime_mean"],
                "MCMC_Runtime_SD": mcmc_results[scenario]["mcmc_runtime_std"],
                "ADVI_Runtime_Mean": mcmc_results[scenario]["advi_runtime_mean"],
                "ADVI_Runtime_SD": mcmc_results[scenario]["advi_runtime_std"],
                "Speedup": mcmc_results[scenario]["speedup"],
                "MCMC_MAE": mcmc_results[scenario]["mcmc_mae_mean"],
                "ADVI_MAE": mcmc_results[scenario]["advi_mae_mean"],
                "MCMC_Coverage": mcmc_results[scenario]["mcmc_coverage_mean"],
                "ADVI_Coverage": mcmc_results[scenario]["advi_coverage_mean"],
            }
        )

    df = pd.DataFrame(comparison_data)
    csv_path = Path(output_dir) / "mcmc_vs_advi_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved to: {csv_path}")

    # Append MCMC/ADVI curve-level coverage to curvelevel_coverage_summary.csv (R4-8)
    if has_coverage:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(output_dir) / "curvelevel_coverage_summary.csv"
        new_rows = []
        for scenario in scenarios:
            for method in ("MCMC", "ADVI"):
                prefix = method.lower()
                for theta, key_pt in [
                    ("pt", f"{prefix}_coverage_pt"),
                    ("mut", f"{prefix}_coverage_mut"),
                ]:
                    for level in COVERAGE_LEVELS:
                        key = f"{key_pt}_{int(level*100)}"
                        val = mcmc_results[scenario].get(key)
                        new_rows.append(
                            {
                                "scenario": scenario,
                                "method": method,
                                "theta": theta,
                                "level": level,
                                "avg_curve_coverage": val if val is not None else np.nan,
                            }
                        )
        new_df = pd.DataFrame(new_rows)
        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(summary_path, index=False)
        print(f"  [OK] Appended MCMC/ADVI coverage to: {summary_path}")


def plot_elbo_traces(main_results, output_dir):
    """Generate 2x3 ELBO trace plots optimized for A4 print quality."""
    print("\n  Generating ELBO trace plots...")

    # Optimized for A4 print quality
    plt.rcParams.update(
        {
            "font.size": 18,  # Base font size (increased)
            "axes.titlesize": 22,  # Title font size
            "axes.labelsize": 20,  # Axis label font size
            "xtick.labelsize": 16,  # X-axis tick font size (increased)
            "ytick.labelsize": 16,  # Y-axis tick font size (increased)
            "legend.fontsize": 16,  # Legend font size (increased)
            "figure.titlesize": 24,  # Figure title font size
            "lines.linewidth": 4,  # Thicker lines
            "axes.linewidth": 1.5,  # Thicker axis borders
            "grid.linewidth": 1.2,  # Thicker grid lines
            "xtick.major.width": 2,  # Thicker tick marks
            "ytick.major.width": 2,
            "xtick.major.size": 8,  # Longer tick marks
            "ytick.major.size": 8,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))  # Larger figure
    scenarios = list(main_results.keys())

    for i, scenario in enumerate(scenarios):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Get scenario data
        scenario_data = main_results[scenario]
        runtime_mean = scenario_data["runtime_mean"]
        elbo_traces = scenario_data.get("elbo_traces", [])

        # Use the first available ELBO trace (from first replication)
        if elbo_traces and len(elbo_traces) > 0:
            elbo_trace = np.array(elbo_traces[0])

            # Convert to Neg ELBO (ELBO opposite number) and scale by 10^5
            elbo_trace = -elbo_trace  # Convert to Neg ELBO
            elbo_trace = elbo_trace / 1e5  # Scale to 10^5 scientific notation

            # Subsample for efficiency (every 10th point)
            step = 10
            elbo_trace = elbo_trace[::step]
            iterations = np.arange(1, len(elbo_trace) * step + 1, step)

            # Rolling mean and std (window = 200 iterations)
            window = min(200, len(elbo_trace) // 5)
            rolling_mean = pd.Series(elbo_trace).rolling(window=window, center=True).mean()
            rolling_std = pd.Series(elbo_trace).rolling(window=window, center=True).std()

            # Plot ELBO trace with thicker lines
            ax.plot(iterations, elbo_trace, alpha=0.25, color="lightblue", linewidth=1.5)

            # Plot rolling mean and bands
            ax.plot(iterations, rolling_mean, color="blue", linewidth=4, label="Rolling Mean")
            ax.fill_between(
                iterations,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.25,
                color="blue",
                label="±1 SD",
            )

            # Add horizontal line for convergence threshold (only if final_elbo is not NaN)
            final_elbo = rolling_mean.iloc[-1]
            if not np.isnan(final_elbo):
                ax.axhline(
                    y=final_elbo,
                    color="red",
                    linestyle="--",
                    linewidth=3,
                    label=f"Final ELBO: {final_elbo:.1f}",
                )
        else:
            # Fallback: No ELBO trace available
            ax.text(
                0.5,
                0.5,
                "ELBO trace not available",
                transform=ax.transAxes,
                fontsize=18,
                ha="center",
                va="center",
            )

        # Add runtime annotation with larger font
        # ax.text(0.02, 0.98, f'Runtime: {runtime_mean:.1f}s',
        #        transform=ax.transAxes, fontsize=16, fontweight='bold',
        #        verticalalignment='top',
        #        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
        #                 edgecolor='black', linewidth=1.5))

        # Set y-axis label for Neg ELBO with fixed scientific notation
        ax.set_ylabel("Neg ELBO (×10$^{{5}}$)", fontsize=20, fontweight="bold", labelpad=10)

        # Formatting
        ax.set_xlabel("ADVI Iterations", fontsize=20, fontweight="bold", labelpad=10)
        ax.set_title(
            f'Scenario {scenario}: {SCENARIOS[scenario]["name"]}',
            fontsize=22,
            fontweight="bold",
            pad=12,
        )
        ax.grid(True, alpha=0.5, linewidth=1.2)

        if i == 0:  # Only show legend on first subplot
            ax.legend(
                loc="upper right",
                fontsize=14,
                framealpha=0.95,
                edgecolor="black",
                frameon=True,
                bbox_to_anchor=(1.0, 1.0),
            )

    plt.tight_layout(pad=1.0)
    output_path = Path(output_dir) / "elbo_traces.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {output_path}")
    plt.close()

    # Reset rcParams to default
    plt.rcParams.update(plt.rcParamsDefault)


def plot_mae_and_ppc(main_results, output_dir):
    """Generate two 3x4 figures: (1) μ_t MAE+PPC (mae_and_ppc.pdf), (2) observed death MAE+PPC (mae_and_ppc_death_counts.pdf). First PPC legend at lower right."""
    print("\n  Generating MAE and PPC plots (μ_t + observed death counts)...")

    # Optimized for A4 horizontal print quality
    plt.rcParams.update(
        {
            "font.size": 22,  # Base font size (increased for horizontal layout)
            "axes.titlesize": 26,  # Title font size
            "axes.labelsize": 24,  # Axis label font size
            "xtick.labelsize": 20,  # X-axis tick font size (increased)
            "ytick.labelsize": 20,  # Y-axis tick font size (increased)
            "legend.fontsize": 18,  # Legend font size (increased)
            "figure.titlesize": 28,  # Figure title font size
            "lines.linewidth": 5,  # Thicker lines for better visibility
            "axes.linewidth": 2,  # Thicker axis borders
            "grid.linewidth": 1.5,  # Thicker grid lines
            "xtick.major.width": 2.5,  # Thicker tick marks
            "ytick.major.width": 2.5,
            "xtick.major.size": 10,  # Longer tick marks
            "ytick.major.size": 10,
        }
    )

    # 3x4 layout: 3 rows x 4 columns for A4 horizontal
    # Row 1: A_MAE, A_PPC, B_MAE, B_PPC
    # Row 2: C_MAE, C_PPC, D_MAE, D_PPC
    # Row 3: E_MAE, E_PPC, F_MAE, F_PPC
    fig, axes = plt.subplots(3, 4, figsize=(28, 21))  # Optimized for A4 horizontal
    scenarios = list(main_results.keys())

    # Define colors consistent with simulation.pdf
    colors = {"BrtaCFR": "#1f77b4", "cCFR": "#ff7f0e", "mCFR": "#2ca02c"}  # Blue  # Orange  # Green

    for i, scenario in enumerate(scenarios):
        scenario_data = main_results[scenario]

        # Map to 3x4 layout: row = i // 2, col_mae = (i % 2) * 2, col_ppc = (i % 2) * 2 + 1
        row = i // 2
        col_mae = (i % 2) * 2
        col_ppc = (i % 2) * 2 + 1

        # MAE comparison as boxplots
        ax_mae = axes[row, col_mae]

        # Prepare data for boxplot (all replications)
        mae_data = [
            scenario_data["mae_values"],  # BrtaCFR
            scenario_data["cCFR_mae_values"],  # cCFR
            scenario_data["mCFR_mae_values"],  # mCFR
        ]

        # Create boxplot with thicker lines for print
        bp = ax_mae.boxplot(
            mae_data,
            labels=["BrtaCFR", "cCFR", "mCFR"],
            patch_artist=True,
            widths=0.6,
            boxprops=dict(linewidth=3),
            medianprops=dict(linewidth=5, color="red"),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            flierprops=dict(markersize=10),
        )  # Larger outlier markers

        # Color the boxes
        for patch, method in zip(bp["boxes"], ["BrtaCFR", "cCFR", "mCFR"]):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.7)

        ax_mae.set_xlabel("Method", fontsize=24, fontweight="bold", labelpad=12)
        ax_mae.set_ylabel("MAE", fontsize=24, fontweight="bold", labelpad=12)
        ax_mae.set_title(
            f'Scenario {scenario}: {SCENARIOS[scenario]["name"]}',
            fontsize=26,
            fontweight="bold",
            pad=14,
        )
        ax_mae.grid(True, alpha=0.5, axis="y", linewidth=1.5)

        # PPC Functional Ribbon Visualization
        ax_ppc = axes[row, col_ppc]

        # Get μ_t quantiles from all replications
        mu_t_quantiles_list = scenario_data.get("mu_t_quantiles_list", [])

        if mu_t_quantiles_list:
            # Inner band (80%): median of each replication's 80% prediction band
            inner_lower = np.median([q["q10"] for q in mu_t_quantiles_list], axis=0)
            inner_upper = np.median([q["q90"] for q in mu_t_quantiles_list], axis=0)
            # Outer band (90%): 5-95% quantile range of each replication's 90% prediction band
            outer_lower = np.quantile([q["q05"] for q in mu_t_quantiles_list], 0.05, axis=0)
            outer_upper = np.quantile([q["q95"] for q in mu_t_quantiles_list], 0.95, axis=0)
            # Median curve: functional median of posterior predictive means across replications
            median_curve = np.median([q["median"] for q in mu_t_quantiles_list], axis=0)
            # True mu_t (expected deaths under true CFR and case counts)
            true_mu_t = TRUE_MU_T[scenario]
            # Plot functional ribbons with improved color contrast
            # Outer band (5-95%): Represents the 5-95% prediction interval across replications
            ax_ppc.fill_between(
                DAYS, outer_lower, outer_upper, color="lightskyblue", alpha=0.6, label="5-95% band"
            )

            # Inner band (80%): Represents the median 80% prediction interval from individual replications
            ax_ppc.fill_between(
                DAYS, inner_lower, inner_upper, color="steelblue", alpha=0.7, label="80% band"
            )

            # Median prediction: Functional median of posterior predictive means across replications
            ax_ppc.plot(
                DAYS, median_curve, color="red", linewidth=5, label="Median prediction", alpha=0.9
            )

            # True μ_t: The actual expected deaths based on true CFR and cases
            ax_ppc.plot(
                DAYS,
                true_mu_t,
                color="black",
                linewidth=5,
                linestyle="-",
                label=r"True $\mu_t$",
                alpha=0.9,
            )

            if i == 0:  # Only show legend on first subplot (Scenario A)
                ax_ppc.legend(
                    loc="lower right",
                    fontsize=18,
                    framealpha=0.95,
                    edgecolor="black",
                    frameon=True,
                    bbox_to_anchor=(1.0, 0.0),
                )
        else:
            ax_ppc.text(
                0.5,
                0.5,
                "PPC data not available",
                transform=ax_ppc.transAxes,
                fontsize=22,
                ha="center",
                va="center",
            )

        ax_ppc.set_xlabel("Days", fontsize=24, fontweight="bold", labelpad=12)
        ax_ppc.set_ylabel(r"Expected Deaths ($\mu_t$)", fontsize=24, fontweight="bold", labelpad=12)
        ax_ppc.set_title(
            f'Scenario {scenario}: {SCENARIOS[scenario]["name"]}',
            fontsize=26,
            fontweight="bold",
            pad=14,
        )
        ax_ppc.grid(True, alpha=0.5, linewidth=1.5)

    plt.tight_layout(pad=1.5)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out1 = Path(output_dir) / "mae_and_ppc.pdf"
    plt.savefig(out1, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"  [OK] Saved to: {out1}")
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Death-count PPC — same 3x4 layout: left col = MAE comparison, right col = death-count PPC
    # -------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(3, 4, figsize=(28, 21))
    for i, scenario in enumerate(scenarios):
        scenario_data = main_results[scenario]
        row = i // 2
        col_mae = (i % 2) * 2
        col_ppc = (i % 2) * 2 + 1
        obs_list = scenario_data.get("observed_deaths_list", [])
        ppc_list = scenario_data.get("death_ppc_quantiles_list", [])

        # Left column: MAE comparison (same as Figure 1)
        ax_mae = axes2[row, col_mae]
        mae_data = [
            scenario_data["mae_values"],
            scenario_data["cCFR_mae_values"],
            scenario_data["mCFR_mae_values"],
        ]
        bp = ax_mae.boxplot(
            mae_data,
            labels=["BrtaCFR", "cCFR", "mCFR"],
            patch_artist=True,
            widths=0.6,
            boxprops=dict(linewidth=3),
            medianprops=dict(linewidth=5, color="red"),
            whiskerprops=dict(linewidth=3),
            capprops=dict(linewidth=3),
            flierprops=dict(markersize=10),
        )
        for patch, method in zip(bp["boxes"], ["BrtaCFR", "cCFR", "mCFR"]):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.7)
        ax_mae.set_xlabel("Method", fontsize=24, fontweight="bold", labelpad=12)
        ax_mae.set_ylabel("MAE", fontsize=24, fontweight="bold", labelpad=12)
        ax_mae.set_title(
            f'Scenario {scenario}: {SCENARIOS[scenario]["name"]}',
            fontsize=26,
            fontweight="bold",
            pad=14,
        )
        ax_mae.grid(True, alpha=0.5, axis="y", linewidth=1.5)

        ax_ppc = axes2[row, col_ppc]
        if obs_list and ppc_list:
            obs = np.asarray(obs_list[0]).flatten()
            q = ppc_list[0]
            days = np.arange(1, len(obs) + 1)
            ax_ppc.fill_between(
                days, q["q025"], q["q975"], color="lightskyblue", alpha=0.6, label="95% PPC"
            )
            ax_ppc.fill_between(
                days, q["q25"], q["q75"], color="steelblue", alpha=0.7, label="50% PPC"
            )
            ax_ppc.plot(days, q["q50"], color="red", linewidth=5, label="Median PPC", alpha=0.9)
            ax_ppc.plot(
                days, obs, color="black", linewidth=3, linestyle="-", label="Observed", alpha=0.9
            )
            if i == 0:
                ax_ppc.legend(
                    loc="lower right",
                    fontsize=18,
                    framealpha=0.95,
                    edgecolor="black",
                    frameon=True,
                    bbox_to_anchor=(1.0, 0.0),
                )
        else:
            ax_ppc.text(
                0.5,
                0.5,
                "PPC data not available",
                transform=ax_ppc.transAxes,
                fontsize=22,
                ha="center",
                va="center",
            )
        ax_ppc.set_xlabel("Days", fontsize=24, fontweight="bold", labelpad=12)
        ax_ppc.set_ylabel("Death count", fontsize=24, fontweight="bold", labelpad=12)
        ax_ppc.set_title(
            f'Scenario {scenario}: {SCENARIOS[scenario]["name"]}',
            fontsize=26,
            fontweight="bold",
            pad=14,
        )
        ax_ppc.grid(True, alpha=0.5, linewidth=1.5)

    plt.tight_layout(pad=1.5)
    out2 = Path(output_dir) / "mae_and_ppc_death_counts.pdf"
    plt.savefig(out2, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [OK] Saved to: {out2}")

    plt.rcParams.update(plt.rcParamsDefault)


# =============================================================================
# Main Execution
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified BrtaCFR simulation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--demo", action="store_true", help="Demo mode (fast, 2 reps)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints (deprecated, checkpoints are always used)",
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear all checkpoints and rerun from scratch",
    )
    parser.add_argument(
        "--only",
        choices=["main", "sensitivity", "mcmc", "lambda", "all"],
        default="all",
        help="Run specific analysis (lambda = R4-1 prior pred + lambda sensitivity only)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores)"
    )
    # R4-1: Lambda (half-Cauchy scale) sensitivity
    parser.add_argument(
        "--lambda_scales",
        type=str,
        default="0.1,0.5,1,5,10",
        help="Comma-separated half-Cauchy scale grid (default: 0.1, 0.5, 1, 5, 10)",
    )
    parser.add_argument(
        "--do_lambda_sensitivity",
        action="store_true",
        help="Run lambda scale sensitivity analysis (simulations only)",
    )
    parser.add_argument(
        "--do_prior_predictive",
        action="store_true",
        help="Run prior predictive smoothness check (simulations only)",
    )

    args = parser.parse_args()

    # Select configuration
    config = DEMO_CONFIG.copy() if args.demo else DEFAULT_CONFIG.copy()
    if args.n_jobs != -1:
        config["n_jobs"] = args.n_jobs

    # Setup directories
    checkpoint_dir = config["checkpoint_dir"]
    output_dir = config["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    if args.clear_checkpoints:
        print("[INFO] Clearing all checkpoints...")
        checkpoint_mgr.clear()

    print("=" * 80)
    print("UNIFIED BRTACFR SIMULATION FRAMEWORK")
    print("=" * 80)
    print(f"Mode: {'DEMO' if args.demo else 'FULL'}")
    print(
        f"Replications: main={config['main_reps']}, sensitivity={config['sensitivity_reps']}, mcmc={config['mcmc_reps']}"
    )
    print(f"Parallel jobs: {config['n_jobs']}")
    print(f"Resume: {args.resume}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    start_time = time.time()

    # Run analyses (checkpoints are automatically used, no need for --resume flag)
    # Parse lambda scale grid (R4-1)
    try:
        lambda_scales = [float(x.strip()) for x in args.lambda_scales.split(",") if x.strip()]
    except ValueError:
        lambda_scales = DEFAULT_LAMBDA_SCALES
    if not lambda_scales:
        lambda_scales = DEFAULT_LAMBDA_SCALES

    if args.only == "lambda":
        if args.do_prior_predictive:
            run_prior_predictive(lambda_scales, output_dir)
        if args.do_lambda_sensitivity:
            run_lambda_sensitivity(config, checkpoint_mgr, lambda_scales, output_dir, resume=True)
        if not (args.do_prior_predictive or args.do_lambda_sensitivity):
            print(
                "  [INFO] --only lambda: set --do_prior_predictive and/or --do_lambda_sensitivity to run."
            )
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("R4-1 LAMBDA ANALYSES COMPLETE!")
        print("=" * 80)
        print(f"Outputs: {output_dir}/")
        return

    if args.only in ["main", "all"]:
        print("\n" + "=" * 40)
        print("PHASE 1: MAIN ANALYSIS & SIMULATION TABLE")
        print("=" * 40)
        main_results = run_main_analysis(config, checkpoint_mgr, resume=True)
        write_lambda_summary_sim(main_results, output_dir)
        plot_sim_curvelevel_coverage(main_results, output_dir)
        plot_main_analysis(main_results, output_dir)
        plot_elbo_traces(main_results, output_dir)
        plot_mae_and_ppc(main_results, output_dir)
        simulation_table = generate_simulation_table(main_results, output_dir)

    if args.only in ["sensitivity", "all"]:
        print("\n" + "=" * 40)
        print("PHASE 2: SENSITIVITY ANALYSIS")
        print("=" * 40)
        gamma_results = run_sensitivity_gamma(config, checkpoint_mgr, resume=True)
        sigma_results = run_sensitivity_sigma(config, checkpoint_mgr, resume=True)
        dist_results = run_sensitivity_distribution(config, checkpoint_mgr, resume=True)
        plot_sensitivity_results(gamma_results, sigma_results, dist_results, output_dir)
        # R4-1: Full run for lambda (prior predictive + lambda scale sensitivity) in both --demo and --full
        print("\n" + "=" * 40)
        print("PHASE 2b: R4-1 LAMBDA (Prior predictive + Lambda scale sensitivity)")
        print("=" * 40)
        run_prior_predictive(lambda_scales, output_dir)
        run_lambda_sensitivity(config, checkpoint_mgr, lambda_scales, output_dir, resume=True)

    if args.only in ["mcmc", "all"]:
        print("\n" + "=" * 40)
        print("PHASE 3: MCMC VS ADVI COMPARISON")
        print("=" * 40)
        mcmc_results = run_mcmc_comparison(config, checkpoint_mgr, resume=True)
        plot_mcmc_comparison(mcmc_results, output_dir)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE!")
    print("=" * 80)
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"Outputs saved to: {output_dir}")

    # List output files
    print("\nGenerated files:")
    output_path = Path(output_dir)
    for f in sorted(output_path.glob("*")):
        size = f.stat().st_size / 1024  # KB
        print(f"  [OK] {f.name} ({size:.1f} KB)")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Review all generated figures and tables")
    print("  2. Check REVIEWER_RESPONSE_SUMMARY.md for manuscript text")
    print("  3. Integrate results into manuscript")
    print("  4. Submit revision!")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Analysis interrupted by user.")
        print("Partial results saved in checkpoints. Use --resume to continue.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
