# BrtaCFR: Bayesian Real-time Case Fatality Rate Estimation

## Overview

BrtaCFR is a Bayesian framework for real-time estimation of time-varying case fatality rates (CFR) from epidemic surveillance data. It uses Automatic Differentiation Variational Inference (ADVI) to provide fast, uncertainty-quantified CFR estimates with credible intervals, supporting timely public health decisions during outbreaks.

## Key Features

- **Real-time estimation**: CFR estimates and 95% credible intervals as data arrive
- **Uncertainty quantification**: Curve-level coverage (50%, 80%, 95% CrI) and posterior predictive checks
- **Six epidemic scenarios**: Constant, exponential growth, delayed growth, decay, peak, valley
- **Benchmarks**: Comparison with cCFR and mCFR; optional MCMC vs ADVI comparison
- **Diagnostics**: ELBO traces, MAE, and PPC for expected and observed deaths
- **Sensitivity**: Gamma delay, prior variance (σ), delay distribution, and lambda prior
- **Real-data applications**: COVID-19 examples for Germany (JHU CSSE) and Japan (WHO)

## Installation

### Prerequisites

- Python 3.8+
- Dependencies are listed in `requirements.txt`.

### Setup

```bash
git clone https://github.com/yourusername/BrtaCFR.git
cd BrtaCFR
pip install -r requirements.txt
```

**Note:** NumPy 2.0.x is incompatible with PyTensor (missing `numpy.lib.array_utils`). The requirements pin `numpy>=1.23.0,<2.0.0`. If you prefer NumPy 2, use `numpy>=2.1` with recent PyMC/PyTensor.

## Quick Start

### Simulation (demo vs full)

```bash
# Demo: 2 replications per analysis, outputs in outputs_demo/
python run_all_simulations.py --demo

# Full: 1000 main, 100 sensitivity, 10 MCMC replications; outputs in outputs/
python run_all_simulations.py
```

### Real-data applications

```bash
# Germany (JHU CSSE data); outputs in output_application/
python run_application_covid_GER.py

# Japan (WHO data)
python run_application_covid_JP.py

# Optional: save posterior lambda summary (median, q025, q975) and use 5000 draws
python run_application_covid_GER.py --save_lambda_summary --lambda_scale 1.0
python run_application_covid_JP.py --save_lambda_summary --lambda_scale 1.0
```

## Usage

### Simulation framework

```bash
# Run only main analysis (or sensitivity / mcmc / lambda)
python run_all_simulations.py --only main
python run_all_simulations.py --only sensitivity
python run_all_simulations.py --only mcmc
python run_all_simulations.py --only lambda --do_prior_predictive --do_lambda_sensitivity

# Limit parallel jobs and clear checkpoints to restart
python run_all_simulations.py --n-jobs 8
python run_all_simulations.py --clear-checkpoints
```

### Using the estimator in code

```python
from methods import BrtaCFR_estimator, mCFR_EST

# Daily cases and deaths (numpy arrays)
c_t = ...   # daily new cases, length T
d_t = ...   # daily new deaths, length T
F_paras = (15.43, 2.03)  # delay distribution (mean, shape) for Gamma

results = BrtaCFR_estimator(c_t, d_t, F_paras, lambda_scale=1.0, n_draws=1000)

# results: mean, lower, upper (95% CrI); optional lambda_draws, pt_cri, mut_cri
print(results["mean"])
print(results["lower"], results["upper"])
```

### Configuration (run_all_simulations.py)

Replication counts and paths are set in `DEFAULT_CONFIG` and `DEMO_CONFIG`:

| Config key         | Full  | Demo |
|--------------------|-------|------|
| `main_reps`        | 1000  | 2    |
| `sensitivity_reps`| 100   | 2    |
| `mcmc_reps`        | 10    | 2    |
| `output_dir`       | `./outputs` | `./outputs_demo` |
| `checkpoint_dir`   | `./checkpoints` | `./checkpoints_demo` |

Prior predictive smoothness uses a fixed **500 draws** per (scenario, λ scale) (`N_PRIOR_DRAWS`), independent of demo/full.

## Output Files

### Simulation (outputs/ or outputs_demo/)

- **Main:** `simulation.pdf`, `fig_sim_curvelevel_coverage_pt_mut.pdf`, `curvelevel_coverage_summary.csv`, `lambda_summary_sim.csv` (.tex), `elbo_traces.pdf`, `mae_and_ppc.pdf`, `mae_and_ppc_death_counts.pdf`, `simulation_table_results.csv`, `simulation_table_latex.tex`
- **Sensitivity:** `sensitivity_gamma.pdf`, `sensitivity_sigma.pdf`, `sensitivity_distributions.pdf`, `sensitivity_analysis_summary.csv`
- **Lambda:** `fig_prior_predictive_smoothness.pdf`, `prior_pred_smoothness.csv`, `fig_lambda_scale_sensitivity.pdf`, `mae_by_lambda_scale.csv`
- **MCMC vs ADVI:** `mcmc_vs_advi_comparison.pdf`, `mcmc_vs_advi_comparison.csv` (and curve-level coverage appended to `curvelevel_coverage_summary.csv`)

### Real-data applications (output_application/)

- **Germany:** `covid_germany_lambda_summary.csv`, `covid_germany_cfr_curves_smooth.pdf`, `covid_germany_cfr_curves_raw.pdf`, `covid_germany_cases_deaths_daily.pdf`, `covid_germany_cases_deaths_ma7.pdf`, `covid_germany_derived_timeseries.csv`, `covid_germany_delay_pmf.csv`
- **Japan:** `covid_japan_lambda_summary.csv`, `covid_japan_cfr_curves_smooth.pdf`, `covid_japan_cfr_curves_raw.pdf`, `covid_japan_cases_deaths_daily.pdf`, `covid_japan_cases_deaths_ma7.pdf`, `covid_japan_derived_timeseries.csv`, `covid_japan_delay_pmf.csv`

With `--save_lambda_summary`, Germany/Japan also write or append to `lambda_summary_real.csv` (and `.tex` in the same directory for GER; Japan uses `outputs/lambda_summary/` for the shared real summary).

## Method Summary

- **Observation model:** Deaths Poisson with rate that accounts for reporting delay.
- **Process model:** Time-varying logit(CFR) with fused LASSO–type smoothness; lambda (smoothing) has a half-Cauchy prior.
- **Inference:** ADVI (default); optional NUTS MCMC for comparison.
- **Diagnostics:** ELBO convergence, MAE (logit scale), posterior predictive checks for μ_t and observed deaths, curve-level coverage.

## Scenarios (simulations)

- **A – Constant;** **B – Exponential growth;** **C – Delayed growth;** **D – Decay;** **E – Peak;** **F – Valley**

## Project Structure

- `run_all_simulations.py` – Simulation pipeline, checkpointing, plotting
- `methods.py` – `BrtaCFR_estimator`, `mCFR_EST`, `lambda_summary_stats`, custom prior logp
- `run_application_covid_GER.py` – Germany application (JHU CSSE)
- `run_application_covid_JP.py` – Japan application (WHO)
- `requirements.txt` – Python dependencies

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

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

**Dr. Hengtao Zhang**  
Email: [zhanght@gdou.edu.cn](mailto:zhanght@gdou.edu.cn)  
Institution: Guangdong Ocean University
