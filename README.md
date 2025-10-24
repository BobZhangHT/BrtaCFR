# BrtaCFR: Bayesian Real-time Case Fatality Rate Estimation

## Overview

BrtaCFR is a comprehensive Bayesian framework for real-time estimation of case fatality rates (CFR) from epidemic surveillance data. The method employs Automatic Differentiation Variational Inference (ADVI) to deliver rapid, uncertainty-quantified estimates of time-varying fatality rates with robust credible intervals, enabling timely public health decision-making during epidemic outbreaks.

## Key Features

- **Real-time Estimation**: Provides CFR estimates as surveillance data becomes available
- **Uncertainty Quantification**: Bayesian credible intervals for all estimates with comprehensive diagnostic validation
- **Multiple Epidemic Patterns**: Handles diverse epidemic scenarios including constant, exponential growth, delayed growth, decay, peak, and valley patterns
- **Comparative Benchmarking**: Systematic evaluation against conventional CFR (cCFR) and modified CFR (mCFR) methods
- **Comprehensive Diagnostics**: ELBO trace monitoring, posterior predictive checks, and convergence diagnostics
- **Real-world Applications**: Validated on COVID-19 data from multiple countries with practical implementation examples

## Installation

### Prerequisites

- Python 3.8 or higher
- Required packages: numpy, scipy, matplotlib, pandas, pymc, arviz, tqdm, joblib

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/BrtaCFR.git
cd BrtaCFR

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Demo Mode (Recommended for first-time users)

```bash
# Run demo analysis (2 replications for quick testing)
python run_all_simulations.py --demo

# Run full analysis
python run_all_simulations.py
```

### Real-world Data Applications

```bash
# Analyze COVID-19 data from Germany
python run_application_covid_GER.py

# Analyze COVID-19 data from Japan
python run_application_covid_JP.py
```

## Usage

### Basic Analysis

```python
from run_all_simulations import run_brtacfr_with_diagnostics

# Example epidemic data
c_t = [10, 15, 20, 25, 30, 35, 40, 45, 50]  # cumulative cases
d_t = [1, 2, 3, 4, 5, 6, 7, 8, 9]           # cumulative deaths
F_paras = (7.0, 2.0)  # delay distribution parameters (mean, shape)

# Run BrtaCFR analysis
results = run_brtacfr_with_diagnostics(c_t, d_t, F_paras)

# Access results
print(f"CFR estimate: {results['mean']}")
print(f"95% Credible Interval: [{results['lower']}, {results['upper']}]")
print(f"Runtime: {results['runtime']:.2f} seconds")
```

### Complete Analysis Pipeline

```python
from run_all_simulations import run_main_analysis, plot_main_analysis, CheckpointManager

# Run complete simulation study
config = {
    'main_reps': 100,
    'sensitivity_reps': 50,
    'mcmc_reps': 20,
    'n_jobs': 8,
    'checkpoint_dir': './checkpoints'
}

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager(config['checkpoint_dir'])

# Execute analysis
main_results = run_main_analysis(config, checkpoint_mgr, resume=False)

# Generate publication-ready plots
plot_main_analysis(main_results, './outputs')
```

### Real-world Data Analysis

```python
# Example: Analyzing COVID-19 data from Germany
from run_application_covid_GER import analyze_germany_data

# Load and analyze real COVID-19 data
results = analyze_germany_data()

# Access time-varying CFR estimates
print(f"Latest CFR estimate: {results['latest_cfr']:.4f}")
print(f"95% Credible Interval: {results['credible_interval']}")
```

### Custom Scenario Analysis

```python
from run_all_simulations import SCENARIOS, run_brtacfr_with_diagnostics
import numpy as np

# Define custom scenario
custom_scenario = {
    'name': 'Custom Pattern',
    'pt': lambda t: 0.05 + 0.02 * np.sin(t/10),  # time-varying CFR
    'T': 200
}

# Run analysis for specific time points
T = 200
t_range = np.arange(1, T+1)
pt_true = custom_scenario['pt'](t_range)

# Generate synthetic data
np.random.seed(42)
c_t = np.random.poisson(10, T)
d_t = np.random.poisson(pt_true * 10, T)

# Run BrtaCFR
F_paras = (7.0, 2.0)  # delay distribution parameters
results = run_brtacfr_with_diagnostics(c_t, d_t, F_paras)
```

### Configuration

Modify `DEMO_CONFIG` in `run_all_simulations.py`:

```python
DEMO_CONFIG = {
    'main_reps': 100,        # Number of replications
    'sensitivity_reps': 50,  # Sensitivity analysis replications
    'mcmc_reps': 20,         # MCMC comparison replications
    'n_jobs': 8,             # Parallel processing (recommended: 8 cores)
    'checkpoint_dir': './checkpoints',  # Checkpoint directory
}
```

## Output Files

The analysis generates comprehensive output files organized by analysis type:

### Simulation Results
- `simulation.pdf`: Main results showing CFR estimates across all scenarios
- `elbo_traces.pdf`: ADVI convergence diagnostics with ELBO trace plots
- `mae_and_ppc.pdf`: Mean Absolute Error boxplots and posterior predictive checks
- `simulation_table_results.csv`: Numerical results table with summary statistics
- `simulation_table_latex.tex`: LaTeX table for publication

### Sensitivity Analysis
- `sensitivity_distributions.pdf`: Parameter sensitivity analysis results
- `sensitivity_gamma.pdf`: Gamma parameter sensitivity plots
- `sensitivity_sigma.pdf`: Sigma parameter sensitivity plots

### Real-world Applications
- `covid_germany_cfr_curves.pdf`: COVID-19 CFR analysis for Germany
- `covid_japan_cfr_curves.pdf`: COVID-19 CFR analysis for Japan
- `covid_germany_derived_timeseries.csv`: Derived time series data for Germany
- `covid_japan_derived_timeseries.csv`: Derived time series data for Japan

## Method Description

BrtaCFR employs a sophisticated Bayesian hierarchical framework for modeling time-varying case fatality rates:

1. **Observation Model**: Deaths follow a Poisson distribution with rate parameter accounting for reporting delays
2. **Process Model**: CFR evolves according to specified temporal patterns with appropriate prior distributions
3. **Inference**: ADVI provides fast, approximate posterior sampling with convergence monitoring
4. **Diagnostics**: Comprehensive ELBO monitoring, posterior predictive checks, and convergence diagnostics ensure reliable results
5. **Validation**: Extensive sensitivity analysis and real-world data validation demonstrate method robustness

## Scenarios

The framework handles six comprehensive epidemic scenarios representing diverse outbreak patterns:

- **Constant**: Stable CFR over time with minimal variation
- **Exponential Growth**: Monotonic increasing CFR following exponential pattern
- **Delayed Growth**: CFR increase after initial stabilization period
- **Decay**: Monotonic decreasing CFR over time
- **Peak**: CFR peaks at intermediate time point then decreases
- **Valley**: CFR decreases initially then increases (U-shaped pattern)

## Citation

If you use BrtaCFR in your research, please cite:

```bibtex
@article{brtacfr2024,
  title={BrtaCFR: Bayesian Real-time Case Fatality Rate Estimation},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, support, or collaboration opportunities, please contact:

**Dr. Hengtao Zhang**  
Email: [zhanght@gdou.edu.cn](mailto:zhanght@gdou.edu.cn)  
Institution: Guangdong Ocean University
