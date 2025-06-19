# run_simulation.py

import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import gamma

# Import the core estimator function and the mCFR helper
from brtacfr_estimator import BrtaCFR_estimator, mCFR_EST

# =============================================================================
# Global Simulation Parameters
# =============================================================================

N_REPLICATIONS = 2 # Use a small number for testing; set to 1000 for full analysis
T_PERIOD = 200
DAYS = np.arange(1, T_PERIOD + 1)
ROOT_DIR = "./results" # Root directory for saving replication data

# =============================================================================
# Simulation Scenarios and Analysis Cases
# =============================================================================

# Define daily case counts (constant across all scenarios)
CT = 3000 - 5 * np.abs(100 - DAYS)

# Define the six scenarios for the true time-varying fatality rate p(t)
SCENARIOS = {
    'A': {'name': '(A)', 'pt': np.full(T_PERIOD, 0.034)},
    'B': {'name': '(B)', 'pt': 0.01 * np.exp(0.012 * DAYS)},
    'C': {'name': '(C)', 'pt': 0.04 * np.exp(0.016 * np.where(DAYS > 60, np.minimum(40, DAYS - 60), 0))},
    'D': {'name': '(D)', 'pt': 0.1 * np.exp(-0.009 * np.where(DAYS > 70, DAYS - 70, 0))},
    'E': {'name': '(E)', 'pt': 0.1 * np.exp(-0.015 * np.abs(DAYS - 80))},
    'F': {'name': '(F)', 'pt': 0.015 * np.exp(0.018 * np.abs(DAYS - 120))}
}

# Define delay distribution parameters for the default and sensitivity analysis
F_PARAS_CASES = {
    'Default': (15.43, 2.03),
    'Sensitivity_P': (18.3, 2.03), # P for 'Plus' (BrtaCFR+)
    'Sensitivity_M': (10.1, 0.53)  # M for 'Minus' (BrtaCFR-)
}

# =============================================================================
# Core Simulation and Plotting Functions
# =============================================================================

def get_delay_dist_pmf(F_paras, T):
    """Computes the normalized Probability Mass Function for the delay distribution."""
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay
    F_k = gamma.cdf(np.arange(T+1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    return f_k

def run_single_replication(rep_idx, scenario_key, analysis_case, output_dir):
"""
    Runs a single simulation replication for a given scenario and analysis case.
    This function is designed to be executed in parallel by joblib.
    It saves results to a file and skips the run if the file already exists.
    """

    # Save/Resume Logic: Skip if this replication has already been completed.
    result_file = os.path.join(output_dir, f"rep_{rep_idx}.npz")
    if os.path.exists(result_file):
        return result_file

    # --- 1. Generate Simulated Data ---
    np.random.seed(rep_idx)
    F_paras_model = F_PARAS_CASES[analysis_case]
    delay_dist_pmf_true = get_delay_dist_pmf(F_PARAS_CASES['Default'], T_PERIOD)
    pt_true = SCENARIOS[scenario_key]['pt']

    # Generate true number of deaths for each day's cases
    true_deaths_no_delay = np.random.binomial(CT.astype(int), pt_true)
    # Distribute these deaths over time according to the delay distribution
    dt = np.array([np.sum(np.flip(delay_dist_pmf_true[:i])*true_deaths_no_delay[:i]) for i in np.arange(1,T_PERIOD+1)])

    # --- 2. Calculate Estimators ---
    cCFR = np.cumsum(dt) /  (np.cumsum(CT) + 1e-10)
    
    delay_dist_pmf_model = get_delay_dist_pmf(F_paras_model, T_PERIOD)
    mCFR = mCFR_EST(CT, dt, delay_dist_pmf_model)

    brta_results = BrtaCFR_estimator(CT, dt, F_paras_model)

    # --- 3. Save Results to File ---
    np.savez_compressed(
        result_file,
        cCFR=cCFR, mCFR=mCFR,
        brtaCFR_mean=brta_results['mean'],
        brtaCFR_lower=brta_results['lower'],
        brtaCFR_upper=brta_results['upper']
    )
    return result_file


def plot_main_analysis(results, scenarios):
    """Generates the main analysis plot (simulation.pdf)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    
    for i, (scenario_key, scenario_data) in enumerate(results.items()):
        ax = axes[i]
        pt_true = scenarios[scenario_key]['pt']
        
        ax.plot(DAYS, pt_true, color='black', linewidth=2.5, label='True')
        ax.plot(DAYS, scenario_data['cCFR_avg'], color='green', label='cCFR')
        ax.plot(DAYS, scenario_data['mCFR_avg'], color='orange', label='mCFR')
        ax.plot(DAYS, scenario_data['BrtaCFR_avg'], color='red', linestyle='--', linewidth=2, label='BrtaCFR')
        ax.fill_between(DAYS, scenario_data['BrtaCFR_lower_avg'], scenario_data['BrtaCFR_upper_avg'], color='blue', alpha=0.2, label='95% CrI')
        
        ax.set_title(scenarios[scenario_key]['name'], fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.text(0.5, 0.04, 'Days', ha='center', va='center', fontsize=14)
    fig.text(0.08, 0.5, 'Fatality Rate', ha='center', va='center', rotation='vertical', fontsize=14)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    plt.tight_layout(rect=[0.1, 0.05, 0.95, 0.95])
    plt.savefig("simulation.pdf", dpi=300, bbox_inches='tight')
    print("\nSaved main analysis plot to simulation.pdf")
    plt.close()


def plot_sensitivity_analysis(all_results, scenarios):
    """Generates the sensitivity analysis plot (simulation_sensitivity.pdf)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    # Extract results for each analysis case
    brta_default = all_results['Default']
    brta_plus = all_results['Sensitivity_P']
    brta_minus = all_results['Sensitivity_M']

    for i, scenario_key in enumerate(scenarios.keys()):
        ax = axes[i]
        pt_true = scenarios[scenario_key]['pt']

        ax.plot(DAYS, pt_true, color='black', linewidth=2.5, label='True')
        ax.plot(DAYS, brta_default[scenario_key]['BrtaCFR_avg'], color='red', linestyle='--', linewidth=2, label='BrtaCFR')
        ax.plot(DAYS, brta_plus[scenario_key]['BrtaCFR_avg'], color='blue', label='BrtaCFR+')
        ax.plot(DAYS, brta_minus[scenario_key]['BrtaCFR_avg'], color='purple', label='BrtaCFR-')
        
        ax.set_title(scenarios[scenario_key]['name'], fontsize=16)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.text(0.5, 0.04, 'Days', ha='center', va='center', fontsize=14)
    fig.text(0.08, 0.5, 'Fatality Rate', ha='center', va='center', rotation='vertical', fontsize=14)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=12)
    
    plt.tight_layout(rect=[0.1, 0.05, 0.95, 0.95])
    plt.savefig("simulation_sensitivity.pdf", dpi=300, bbox_inches='tight')
    print("Saved sensitivity analysis plot to simulation_sensitivity.pdf")
    plt.close()

# =============================================================================
# Main Execution Block
# =============================================================================

def main():
    """
    Main function to run all simulations for all analysis cases,
    aggregate the results, and generate the final plots.
    """
    
    all_aggregated_results = {}

    # --- 1. Run All Simulations First ---
    # This loop ensures all data is generated before moving to plotting
    for analysis_case in F_PARAS_CASES.keys():
        print(f"\n{'='*60}")
        print(f"PROCESSING ANALYSIS CASE: {analysis_case}")
        print(f"{'='*60}")
        
        all_aggregated_results[analysis_case] = {}
        for scenario_key in SCENARIOS.keys():
            output_dir = os.path.join(ROOT_DIR, analysis_case, scenario_key)
            os.makedirs(output_dir, exist_ok=True)
            
            with tqdm(total=N_REPLICATIONS, desc=f"Scenario {scenario_key} ({analysis_case})") as pbar:
                results_files = Parallel(n_jobs=-1)(
                    delayed(run_single_replication)(i, scenario_key, analysis_case, output_dir) for i in range(N_REPLICATIONS)
                )
                pbar.update(N_REPLICATIONS)
            
            # --- 2. Aggregate Results ---
            replication_results = [np.load(f) for f in results_files]
            
            # Average the estimates across all replications
            all_aggregated_results[analysis_case][scenario_key] = {
                'cCFR_avg': np.mean([res['cCFR'] for res in replication_results], axis=0),
                'mCFR_avg': np.mean([res['mCFR'] for res in replication_results], axis=0),
                'BrtaCFR_avg': np.mean([res['brtaCFR_mean'] for res in replication_results], axis=0),
                'BrtaCFR_lower_avg': np.mean([res['brtaCFR_lower'] for res in replication_results], axis=0),
                'BrtaCFR_upper_avg': np.mean([res['brtaCFR_upper'] for res in replication_results], axis=0)
            }

    # --- 3. Generate the Two Final Plots ---
    print("\nAll simulations complete. Generating final plots...")
    plot_main_analysis(all_aggregated_results['Default'], SCENARIOS)
    plot_sensitivity_analysis(all_aggregated_results, SCENARIOS)
    print("\nProcess finished.")

if __name__ == '__main__':
    main()