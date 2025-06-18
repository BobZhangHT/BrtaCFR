# A Bayesian Framework for Real-Time Estimation of Time-Varying Case Fatality Rates



This repository contains the official Python implementation for the manuscript: "**A Bayesian Framework for Real-Time Estimation of Time-Varying Case Fatality Rates with Application to COVID-19 Policy Periods in Japan**".&#x20;



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
   git clone https://github.com/your-username/BrtaCFR.
   git cd BrtaCFR
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

&#x20;

The analyses from the manuscript can be reproduced using the provided scripts.

### 1. Simulation Study

To run the six simulation scenarios and generate the corresponding plot (as seen in Figure 1 of the manuscript), execute the following command:

Bash

```
python run_simulation.py
```

This script will run the estimation for all six scenarios (A-F), print the progress to the console, and save the final plot as simulation_results.png. The plot demonstrates the BrtaCFR estimator's superior ability to capture diverse fatality rate patterns compared to cCFR and mCFR.



### 2. Real-Data Application (Japan)

To reproduce the analysis on the COVID-19 data from Japan (as seen in Figure 4 of the manuscript), run:

Bash

```
python run_application.py
```

This script will first check for the WHO dataset (WHO-COVID-19-global-daily-data.csv) and download it if it is not present. It will then process the data for Japan, run the BrtaCFR estimator, and save the resulting plot as japan_application_results.png. The plot shows how the estimator captures the dynamic changes in the fatality rate in Japan in relation to major pandemic waves and public health policies.



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

&#x20;

For any questions, comments, or suggestions, please feel free to contact the first author or corresponding author:

* Hengtao Zhang: zhanght@gdou.edu.cn

* Yuanke Qu: quxiaoke@gdou.edu.cn

