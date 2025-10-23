# BrtaCFR: Bayesian Real-time Case Fatality Rate Estimation

## Overview

BrtaCFR is a Bayesian framework for real-time estimation of case fatality rates (CFR) from epidemic data. The method uses Automatic Differentiation Variational Inference (ADVI) to provide rapid, uncertainty-quantified estimates of time-varying fatality rates with credible intervals.

## Key Features

- **Real-time Estimation**: Provides CFR estimates as data becomes available
- **Uncertainty Quantification**: Bayesian credible intervals for all estimates
- **Multiple Scenarios**: Handles various epidemic patterns (constant, exponential growth, delayed growth, decay, peak, valley)
- **Comparative Analysis**: Benchmarks against conventional CFR (cCFR) and modified CFR (mCFR) methods
- **Comprehensive Diagnostics**: ELBO trace monitoring, posterior predictive checks, and convergence diagnostics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/BrtaCFR.git
cd BrtaCFR

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo analysis (2 replications for quick testing)
python run_all_simulations.py

# Run full analysis (default: 100 replications)
python run_all_simulations.py --config full
```

## Usage

### Basic Analysis

```python
from run_all_simulations import run_brtacfr_with_diagnostics

# Example data
c_t = [10, 15, 20, 25, 30]  # cumulative cases
d_t = [1, 2, 3, 4, 5]       # cumulative deaths
F_paras = {'gamma': 0.1, 'sigma': 0.2}  # parameters

# Run BrtaCFR analysis
results = run_brtacfr_with_diagnostics(c_t, d_t, F_paras)
```

### Configuration

Modify `DEMO_CONFIG` in `run_all_simulations.py`:

```python
DEMO_CONFIG = {
    'main_reps': 100,        # Number of replications
    'sensitivity_reps': 50,  # Sensitivity analysis replications
    'mcmc_reps': 20,         # MCMC comparison replications
    'n_jobs': -1,            # Parallel processing (-1 = all cores)
    'checkpoint_dir': './checkpoints',  # Checkpoint directory
}
```

## Output Files

The analysis generates several output files:

- `simulation.pdf`: Main results showing CFR estimates across scenarios
- `elbo_traces.pdf`: ADVI convergence diagnostics
- `mae_and_ppc.pdf`: Mean Absolute Error boxplots and posterior predictive checks
- `sensitivity_analysis.pdf`: Parameter sensitivity analysis
- `mcmc_comparison.pdf`: ADVI vs MCMC comparison
- `simulation_table_results.csv`: Numerical results table
- `simulation_table_latex.tex`: LaTeX table for publication

## Method Description

BrtaCFR models the case fatality rate as a time-varying parameter using a Bayesian hierarchical framework:

1. **Observation Model**: Deaths follow a Poisson distribution with rate parameter
2. **Process Model**: CFR evolves according to specified temporal patterns
3. **Inference**: ADVI provides fast, approximate posterior sampling
4. **Diagnostics**: ELBO monitoring ensures convergence

## Scenarios

The framework handles six epidemic scenarios:

- **Constant**: Stable CFR over time
- **Exponential Growth**: Increasing CFR
- **Delayed Growth**: CFR increase after initial period
- **Decay**: Decreasing CFR over time
- **Peak**: CFR peaks then decreases
- **Valley**: CFR decreases then increases

## Performance

- **Speed**: ADVI provides ~20x faster inference than MCMC
- **Accuracy**: Comparable accuracy to MCMC with proper convergence
- **Scalability**: Handles real-time data streams efficiently

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

For questions or support, please contact [zhanght@gdou.edu.cn](mailto:zhanght@gdou.edu.cn).

## Acknowledgments

- Built with PyMC for Bayesian inference
- Visualization with Matplotlib and Seaborn
- Parallel processing with Joblib