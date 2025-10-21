# run_application.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import os
from scipy.stats import gamma
from statsmodels.nonparametric.kernel_regression import KernelReg

# Import the core estimator function and the mCFR helper
from methods import BrtaCFR_estimator, mCFR_EST

# =============================================================================
# Global Parameters
# =============================================================================

DATA_URL = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
DATA_FILE = "WHO-COVID-19-global-daily-data.csv"
COUNTRY = "Japan"

# =============================================================================
# Helper Functions
# =============================================================================

def download_data(url, filename):
    """
    Downloads data from the specified URL if the file does not already exist.

    Args:
        url (str): The URL of the data source.
        filename (str): The local filename to save the data to.
    """
    if not os.path.exists(filename):
        print(f"Downloading data from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Data saved to {filename}")

def get_delay_dist_pmf(F_paras, T):
    """
    Computes the normalized Probability Mass Function (PMF) for the delay distribution.

    Args:
        F_paras (tuple): Parameters (mean, shape) for the Gamma delay distribution.
        T (int): The length of the time series for which to compute the PMF.

    Returns:
        np.ndarray: The normalized PMF of the delay distribution.
    """
    mean_delay, shape_delay = F_paras
    scale_delay = mean_delay / shape_delay
    F_k = gamma.cdf(np.arange(T+1), a=shape_delay, scale=scale_delay)
    f_k = np.diff(F_k)
    return f_k

# =============================================================================
# Main Execution Block
# =============================================================================

def main():
    """
    Main function to run the entire real-data application for Japan:
    1. Downloads and prepares the data.
    2. Runs all CFR estimators.
    3. Smooths the BrtaCFR results for visualization.
    4. Generates and saves the final plot.
    """
    # --- 1. Data Loading and Preparation ---
    download_data(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE, parse_dates=['Date_reported'])
    
    # Filter for Japan and the specific time period of interest
    df_country = df[df['Country'] == COUNTRY].copy()
    df_country = df_country[
        (df_country['Date_reported'] >= '2020-01-16') & 
        (df_country['Date_reported'] <= '2022-10-31')
    ]
    df_country = df_country.sort_values('Date_reported').reset_index(drop=True)
    
    # Extract cases, deaths, and dates, ensuring non-negativity
    ct = df_country['New_cases'].values
    dt = df_country['New_deaths'].values
    ct[ct < 0] = 0
    dt[dt < 0] = 0
    dates = df_country['Date_reported']

    # --- 2. Run All Estimators ---
    print("\n--- Running Estimators ---")
    print(f"Running BrtaCFR estimator for {COUNTRY}...")
    F_paras_default = (15.43, 2.03)
    # Use the default delay distribution parameters from the manuscript
    results = BrtaCFR_estimator(ct, dt, F_paras_default)
    print("BrtaCFR estimation complete.")

    # Calculate cCFR and mCFR for comparison
    delay_dist_pmf = get_delay_dist_pmf(F_paras_default, len(ct))
    cCFR = np.cumsum(dt) / (np.cumsum(ct)+1e-10)
    mCFR = mCFR_EST(ct, dt, delay_dist_pmf)
    print("cCFR and mCFR calculation complete.")

    # --- 3. Smooth the BrtaCFR for Visualization ---
    # Use Nadaraya-Watson kernel regression for smoothing. The bandwidth (bw) of 21
    # corresponds to a 3-week window, which helps visualize the underlying trend.
    T = ct.shape[0]
    kreg_brtacfr = KernelReg(endog=results['mean'], 
                             exog=np.arange(1,T+1), 
                             var_type='c', 
                             reg_type='ll', 
                             bw=[21],
                             ckertype='gaussian')
    kreg_brtaCFR_CrI_Low = KernelReg(endog=results['lower'], 
                                     exog=np.arange(1,T+1), 
                                     var_type='c', 
                                     reg_type='ll', 
                                     bw=[21], 
                                     ckertype='gaussian')
    kreg_brtaCFR_CrI_Up = KernelReg(endog=results['upper'], 
                                    exog=np.arange(1,T+1), 
                                    var_type='c', 
                                    reg_type='ll', 
                                    bw=[21], 
                                    ckertype='gaussian')
    
    smoothed_brtacfr = kreg_brtacfr.fit(np.arange(1,T+1))[0]
    smooth_brtacfr_CrIL = kreg_brtaCFR_CrI_Low.fit(np.arange(1,T+1))[0]
    smooth_brtacfr_CrIU = kreg_brtaCFR_CrI_Up.fit(np.arange(1,T+1))[0]

    # --- 4. Generate and Save Final Plot ---
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the different estimators
    ax.plot(dates, cCFR, color='green', label='cCFR')
    ax.plot(dates, mCFR, color='orange', label='mCFR')
    ax.plot(dates, smoothed_brtacfr, color='red', linestyle='--', linewidth=2, label='BrtaCFR (Smoothed)')
    ax.fill_between(dates, smooth_brtacfr_CrIL, smooth_brtacfr_CrIU, color='blue', alpha=0.2, label='95% Credible Interval')

    # Final plot styling
    ax.set_title(f'Case Fatality Rate Estimators for {COUNTRY}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Fatality Rate', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 0.15)

    # Format the x-axis to show dates clearly
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Save the figure and show the plot
    plt.tight_layout()
    plt.savefig('japan_application_results.pdf', dpi=300)
    print("Saved application plot to japan_application_results.pdf")
    plt.show()
    
if __name__ == '__main__':
    main()