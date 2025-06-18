# A Bayesian Framework for Real-Time Estimation of Time-Varying Case Fatality Rates


This repository contains the official Python implementation for the manuscript: "**A Bayesian Framework for Real-Time Estimation of Time-Varying Case Fatality Rates with Application to COVID-19 Policy Periods in Japan**".

Our work introduces the Bayesian real-time adjusted Case Fatality Rate (BrtaCFR) estimator, a method designed to provide robust, delay-adjusted, and time-varying estimates of disease severity. It uses only standard epidemiological surveillance data (daily case and death counts) and offers a complete solution for uncertainty quantification and data-driven smoothing, overcoming key limitations of previous frequentist approaches.

## Key Features

* **Real-Time Estimation**: Tracks daily changes in disease severity.

* **Delay Adjustment**: Explicitly models the time lag between case confirmation and death to reduce bias.

* **Bayesian Framework**: Provides full posterior distributions for robust uncertainty quantification via 95% Credible Intervals.

* **Automatic Smoothing**: Uses a fused LASSO prior with a data-driven hyperprior, eliminating the need for manual tuning of smoothing parameters.

* **Efficient Inference**: Implemented using PyMC and Automatic Differentiation Variational Inference (ADVI) for rapid computation suitable for real-time monitoring.

## Installation

To set up the necessary environment to run the code, please follow these steps:

1. **Clone the repository:**

   ```
   git clone https://github.com/your-username/BrtaCFR.git
   cd BrtaCFR
   ```

2. **Create and activate a virtual environment (recommended):**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```
   pip install -r requirements.txt
   ```

## Usage

The analyses from the manuscript can be reproduced using the provided scripts.

### Simulation Study & Sensitivity Analysis

To run the six simulation scenarios (with 1000 replications each) and the full sensitivity analysis, execute the following command in your terminal:

```
python run_simulation.py
```

This single command will perform the entire simulation pipeline:

* It runs replications in parallel using all available CPU cores.

* Progress for each scenario and analysis case is displayed with a tqdm progress bar.

* **Save & Resume**: Results for each individual replication are saved in the results/ directory. If the script is stopped and restarted, it will automatically skip the already completed replications.

* **Outputs**: Upon completion, the script generates two publication-quality PDF figures in the root directory:

  1. `simulation.pdf`: Compares cCFR, mCFR, and BrtaCFR against the true fatality rate.

  2. `simulation_sensitivity.pdf`: Compares the BrtaCFR estimates under the three different delay distribution assumptions (BrtaCFR, BrtaCFR+, BrtaCFR-).

### Real-Data Application (Japan)

To reproduce the analysis on the COVID-19 data from Japan, which compares cCFR, mCFR, and the smoothed BrtaCFR, run:

```
python run_application.py
```

This script will first check for the WHO dataset and download it if it's not present. It will then run the estimators and save the resulting plot as `japan_application_results.pdf`.



## Repository Structure

```
.
├── brtacfr_estimator.py       # Core module with the BrtaCFR estimation logic
├── run_simulation.py         # Script to run the simulation studies from the paper
├── run_application.py        # Script to run the real-data application for Japan
├── requirements.txt          # Required Python packages for reproducibility
├── LICENSE                # MIT License file
└── README.md               # This documentation
```

## Contact

For any questions, comments, or suggestions, please feel free to contact the first author or corresponding author:
* Hengtao Zhang: zhanght@gdou.edu.cn
* Yuanke Qu: quxiaoke@gdou.edu.cn

