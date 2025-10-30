# run_application_covid_GER.py

"""
COVID-19 CFR Analysis for Germany using Johns Hopkins CSSE Data
Data source: https://github.com/CSSEGISandData/COVID-19

Reference:
  Dong E, Du H, Gardner L. An interactive web-based dashboard to track COVID-19 
  in real time. Lancet Inf Dis. 20(5):533-534. doi: 10.1016/S1473-3099(20)30120-1
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.stats import gamma
from statsmodels.nonparametric.kernel_regression import KernelReg
from datetime import datetime, date

# Suppress PyTensor BLAS warning
warnings.filterwarnings('ignore', message='.*PyTensor could not link to a BLAS.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')

# Import the core estimator function and the mCFR helper
from methods import BrtaCFR_estimator, mCFR_EST

# =============================================================================
# Global Parameters
# =============================================================================

COUNTRY = "Germany"
START_DATE = "2020-01-27"  
END_DATE = "2022-09-30"     # Original end date for analysis
FIT_EXTEND_DAYS = 30        # Extend fitting by 40 days to reduce tail effects

# Johns Hopkins CSSE GitHub raw URLs
CONFIRMED_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
DEATHS_URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

# =============================================================================
# Helper Functions
# =============================================================================

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


def download_csv_to_local(url, local_path):
    """
    Downloads a CSV file from URL and saves it locally.
    
    Args:
        url (str): The raw GitHub URL for the CSV file.
        local_path (str): Local file path to save the CSV.
    
    Returns:
        bool: True if download successful, False otherwise.
    """
    try:
        import urllib.request
        print(f"  Downloading from: {url}")
        urllib.request.urlretrieve(url, local_path)
        print(f"  ✓ Saved to: {local_path}")
        return True
    except Exception as e:
        print(f"  ✗ Error downloading from {url}: {e}")
        return False


def load_jhu_data(local_path, url=None):
    """
    Loads data from local CSV file. If file doesn't exist, downloads it first.
    
    Args:
        local_path (str): Local file path of the CSV.
        url (str, optional): URL to download from if file doesn't exist.
    
    Returns:
        pd.DataFrame: The loaded data, or None if failed.
    """
    # Check if file exists locally
    if os.path.exists(local_path):
        print(f"  Found local file: {local_path}")
        try:
            df = pd.read_csv(local_path)
            print(f"  ✓ Loaded from local file")
            return df
        except Exception as e:
            print(f"  ✗ Error reading local file: {e}")
            if url is None:
                return None
            print(f"  Attempting to re-download...")
    
    # File doesn't exist or failed to read, download it
    if url is not None:
        if download_csv_to_local(url, local_path):
            try:
                df = pd.read_csv(local_path)
                return df
            except Exception as e:
                print(f"  ✗ Error reading downloaded file: {e}")
                return None
    else:
        print(f"  ✗ File not found and no URL provided")
        return None
    
    return None


def extract_country_timeseries(df, country_name, start_date, end_date):
    """
    Extracts time series data for a specific country from JHU CSSE format.
    
    Args:
        df (pd.DataFrame): The JHU CSSE data (wide format).
        country_name (str): Name of the country to extract.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
        tuple: (dates, cumulative_values) as pandas Series.
    """
    # Filter for the country
    country_row = df[df['Country/Region'] == country_name]
    
    if country_row.empty:
        raise ValueError(f"Country '{country_name}' not found in the data")
    
    # If there are multiple rows (provinces), sum them
    if len(country_row) > 1:
        # Sum all provinces (skip first 4 columns: Province/State, Country/Region, Lat, Long)
        country_data = country_row.iloc[:, 4:].sum(axis=0)
    else:
        # Single row, extract data (skip first 4 columns)
        country_data = country_row.iloc[0, 4:]
    
    # Parse dates from column names (format: M/D/YY)
    dates = pd.to_datetime(country_data.index, format='%m/%d/%y')
    values = country_data.values
    
    # Create a DataFrame
    ts_df = pd.DataFrame({
        'date': dates,
        'cumulative': values
    })
    
    # Filter for date range
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    ts_df = ts_df[(ts_df['date'] >= start_dt) & (ts_df['date'] <= end_dt)]
    
    return ts_df['date'], ts_df['cumulative']


def compute_daily_new(cumulative_data):
    """
    Computes daily new cases/deaths from cumulative data.
    
    Args:
        cumulative_data (pd.Series): Cumulative counts.
    
    Returns:
        np.ndarray: Daily new counts.
    """
    daily = cumulative_data.diff().fillna(cumulative_data.iloc[0] if len(cumulative_data) > 0 else 0)
    
    # Convert to numpy array and handle any NaN values
    daily_array = daily.values
    daily_array = np.nan_to_num(daily_array, nan=0.0)
    
    # Ensure non-negativity (sometimes corrections cause negative values)
    daily_array[daily_array < 0] = 0
    
    return daily_array


# =============================================================================
# Main Execution Block
# =============================================================================

def main():
    """
    Main function to run the entire real-data application for Germany:
    1. Downloads data from Johns Hopkins CSSE GitHub.
    2. Extracts Germany data for the specified date range.
    3. Runs all CFR estimators.
    4. Smooths the BrtaCFR results for visualization.
    5. Generates and saves the final plots.
    6. Saves all curves to CSV for further analysis.
    """
    # --- 0. Create output directory and define local file paths ---
    output_dir = "output_application"
    os.makedirs(output_dir, exist_ok=True)
    
    # Local file paths for downloaded CSV files
    confirmed_local = "time_series_covid19_confirmed_global.csv"
    deaths_local = "time_series_covid19_deaths_global.csv"
    
    # --- 1. Data Loading from Johns Hopkins CSSE ---
    print(f"\n{'='*80}")
    print(f"COVID-19 CFR Analysis for {COUNTRY}")
    print(f"{'='*80}")
    # Calculate extended end date for fitting
    from datetime import datetime, timedelta
    end_date_obj = datetime.strptime(END_DATE, "%Y-%m-%d")
    extended_end_date = (end_date_obj + timedelta(days=FIT_EXTEND_DAYS)).strftime("%Y-%m-%d")
    
    print(f"Data source: Johns Hopkins CSSE COVID-19 Data Repository")
    print(f"  GitHub: https://github.com/CSSEGISandData/COVID-19")
    print(f"Country: {COUNTRY}")
    print(f"Analysis date range: {START_DATE} to {END_DATE}")
    print(f"Fitting date range: {START_DATE} to {extended_end_date} (extended by {FIT_EXTEND_DAYS} days)")
    
    print(f"\nLoading confirmed cases data...")
    confirmed_df = load_jhu_data(confirmed_local, CONFIRMED_URL)
    if confirmed_df is None:
        print("ERROR: Failed to load confirmed cases data.")
        return
    
    print(f"✓ Confirmed cases data loaded ({len(confirmed_df)} countries/regions)")
    
    print(f"\nLoading deaths data...")
    deaths_df = load_jhu_data(deaths_local, DEATHS_URL)
    if deaths_df is None:
        print("ERROR: Failed to load deaths data.")
        return
    
    print(f"✓ Deaths data loaded ({len(deaths_df)} countries/regions)")
    
    # --- 2. Extract Germany Data ---
    print(f"\nExtracting {COUNTRY} data...")
    
    # Extract data with extended end date for fitting
    dates_conf, cumulative_confirmed = extract_country_timeseries(
        confirmed_df, COUNTRY, START_DATE, extended_end_date
    )
    
    dates_deaths, cumulative_deaths = extract_country_timeseries(
        deaths_df, COUNTRY, START_DATE, extended_end_date
    )
    
    # Ensure dates align
    if not dates_conf.equals(dates_deaths):
        print("WARNING: Dates do not align between confirmed and deaths data")
        # Use intersection of dates
        common_dates = dates_conf[dates_conf.isin(dates_deaths)]
        cumulative_confirmed = cumulative_confirmed[dates_conf.isin(common_dates)]
        cumulative_deaths = cumulative_deaths[dates_deaths.isin(common_dates)]
        dates = common_dates.reset_index(drop=True)
    else:
        dates = dates_conf.reset_index(drop=True)
    
    print(f"✓ Extracted data for {len(dates)} days")
    
    # --- 3. Compute Daily New Cases and Deaths ---
    print(f"\nComputing daily new cases and deaths...")
    
    ct = compute_daily_new(cumulative_confirmed)
    dt = compute_daily_new(cumulative_deaths)
    
    # Impute missing values with 0 (as requested)
    ct = np.nan_to_num(ct, nan=0.0)
    dt = np.nan_to_num(dt, nan=0.0)
    
    print(f"✓ Daily data computed")
    
    # --- 4. Data Summary ---
    print(f"\nData Summary:")
    print(f"  Total days: {len(ct)}")
    print(f"  Total cumulative cases: {cumulative_confirmed.iloc[-1]:,.0f}")
    print(f"  Total cumulative deaths: {cumulative_deaths.iloc[-1]:,.0f}")
    print(f"  Overall crude CFR: {(cumulative_deaths.iloc[-1]/cumulative_confirmed.iloc[-1]*100):.2f}%")
    print(f"  Days with zero cases: {(ct == 0).sum()}")
    print(f"  Days with zero deaths: {(dt == 0).sum()}")

    # --- 5. Run All Estimators ---
    print(f"\n{'='*80}")
    print("Running CFR Estimators")
    print(f"{'='*80}")
    
    T = len(ct)  # Total number of days
    print(f"Running BrtaCFR estimator for {COUNTRY}...")
    print(f"  Note: This may take several minutes for {T} days of data...")
    print(f"  Using ADVI with 100,000 iterations")
    print(f"  Progress bar will appear below:")
    
    F_paras_default = (15.43, 2.03)
    # Use the default delay distribution parameters from the manuscript
    results = BrtaCFR_estimator(ct, dt, F_paras_default)
    print("\n✓ BrtaCFR estimation complete.")

    # Calculate cCFR and mCFR for comparison
    delay_dist_pmf = get_delay_dist_pmf(F_paras_default, len(ct))
    cCFR = np.cumsum(dt) / (np.cumsum(ct)+1e-10)
    mCFR = mCFR_EST(ct, dt, delay_dist_pmf)
    print("✓ cCFR and mCFR calculation complete.")
    
    # --- 5b. Truncate data to original analysis period ---
    # Find the index corresponding to the original END_DATE
    original_end_idx = None
    for i, date in enumerate(dates):
        if date.strftime("%Y-%m-%d") == END_DATE:
            original_end_idx = i + 1  # +1 because we want to include this day
            break
    
    if original_end_idx is not None:
        print(f"Truncating data to original analysis period (up to {END_DATE})...")
        # Truncate all arrays to original analysis period
        ct = ct[:original_end_idx]
        dt = dt[:original_end_idx]
        dates = dates[:original_end_idx]
        cumulative_confirmed = cumulative_confirmed[:original_end_idx]
        cumulative_deaths = cumulative_deaths[:original_end_idx]
        results['mean'] = results['mean'][:original_end_idx]
        results['lower'] = results['lower'][:original_end_idx]
        results['upper'] = results['upper'][:original_end_idx]
        cCFR = cCFR[:original_end_idx]
        mCFR = mCFR[:original_end_idx]
        delay_dist_pmf = delay_dist_pmf[:original_end_idx]
        T = len(ct)  # Update T to reflect truncated length
        print(f"✓ Data truncated to {T} days for analysis")
    else:
        print(f"WARNING: Could not find {END_DATE} in dates, using full extended dataset")

    # --- 6. Smooth the BrtaCFR for Visualization ---
    # Use Nadaraya-Watson kernel regression for smoothing. The bandwidth (bw) of 21
    # corresponds to a 3-week window, which helps visualize the underlying trend.
    print("✓ Smoothing BrtaCFR estimates...")
    # T is already defined above
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
    
    # Apply smoothing with non-negative constraint
    smoothed_brtacfr = np.maximum(kreg_brtacfr.fit(np.arange(1,T+1))[0], 0)
    smooth_brtacfr_CrIL = np.maximum(kreg_brtaCFR_CrI_Low.fit(np.arange(1,T+1))[0], 0)
    smooth_brtacfr_CrIU = np.maximum(kreg_brtaCFR_CrI_Up.fit(np.arange(1,T+1))[0], 0)

    # --- 6b. Calculate 7-day moving averages ---
    cases_ma7 = pd.Series(ct).rolling(window=7, min_periods=1, center=False).mean().values
    deaths_ma7 = pd.Series(dt).rolling(window=7, min_periods=1, center=False).mean().values

    # --- 7. Generate and Save Final Plot (CFR Curves) ---
    print(f"\n{'='*80}")
    print("Generating Plots")
    print(f"{'='*80}")
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the different estimators
    ax.plot(dates, cCFR, color='green', label='cCFR')
    ax.plot(dates, mCFR, color='orange', label='mCFR')
    ax.plot(dates, smoothed_brtacfr, color='red', linestyle='--', linewidth=2, label='BrtaCFR (Smoothed)')
    ax.fill_between(dates, smooth_brtacfr_CrIL, smooth_brtacfr_CrIU, color='blue', alpha=0.2, label='95% Credible Interval')

    # Final plot styling
    ax.set_title(f'Case Fatality Rate Estimators for {COUNTRY}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fatality Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 0.15)

    # Format the x-axis to show dates clearly
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'covid_germany_cfr_curves.pdf'), dpi=300)
    print(f"✓ Saved CFR plot to {output_dir}/covid_germany_cfr_curves.pdf")
    plt.close()
    
    # --- 8. Generate Daily Cases and Deaths Plot ---
    fig2, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(dates, ct, color="black", linewidth=1.1, label="Cases (daily)")
    ax1.set_ylabel("Cases (daily)", color="black", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx()
    ax2.plot(dates, dt, color="red", linewidth=1.1, label="Deaths (daily)")
    ax2.set_ylabel("Deaths (daily)", color="red", fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title(f"COVID-19 ({COUNTRY}) — Daily Cases and Deaths", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Bold tick labels
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    fig2.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'covid_germany_cases_deaths_daily.pdf'), dpi=300)
    print(f"✓ Saved daily cases/deaths plot to {output_dir}/covid_germany_cases_deaths_daily.pdf")
    plt.close()
    
    # --- 9. Generate 7-day Moving Averages Plot ---
    fig3, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(dates, cases_ma7, color="black", linewidth=1.8, label="Cases: 7-day Moving Average")
    ax1.set_ylabel("Cases: 7-day Moving Average", color="black", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')
    
    ax2 = ax1.twinx()
    ax2.plot(dates, deaths_ma7, color="red", linewidth=2.0,
             label="Deaths: 7-day Moving Average")
    ax2.set_ylabel("Deaths: 7-day Moving Average", color="red", fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title(f"COVID-19 ({COUNTRY}) — 7-day Moving Averages", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Bold tick labels
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    # Unified legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)
    
    fig3.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'covid_germany_cases_deaths_ma7.pdf'), dpi=300)
    print(f"✓ Saved 7-day MA plot to {output_dir}/covid_germany_cases_deaths_ma7.pdf")
    plt.close()
    
    # --- 10. Save all curves to CSV for further analysis ---
    print(f"\n{'='*80}")
    print("Exporting Data")
    print(f"{'='*80}")
    
    curves_df = pd.DataFrame({
        'date': dates.dt.date.astype(str),
        'cases_daily': ct,
        'deaths_daily': dt,
        'cumulative_cases': cumulative_confirmed.values,
        'cumulative_deaths': cumulative_deaths.values,
        'cases_ma7': cases_ma7,
        'deaths_ma7': deaths_ma7,
        'cCFR': cCFR,
        'mCFR': mCFR,
        'BrtaCFR_mean_raw': results['mean'],
        'BrtaCFR_lower_raw': results['lower'],
        'BrtaCFR_upper_raw': results['upper'],
        'BrtaCFR_mean_smooth': smoothed_brtacfr,
        'BrtaCFR_lower_smooth': smooth_brtacfr_CrIL,
        'BrtaCFR_upper_smooth': smooth_brtacfr_CrIU
    })
    curves_df.to_csv(os.path.join(output_dir, 'covid_germany_derived_timeseries.csv'), index=False)
    print(f"✓ Saved time series to {output_dir}/covid_germany_derived_timeseries.csv")
    
    # --- 11. Save delay distribution PMF to CSV ---
    pmf_df = pd.DataFrame({
        'day': np.arange(1, len(delay_dist_pmf) + 1),
        'f_k': delay_dist_pmf
    })
    pmf_df.to_csv(os.path.join(output_dir, 'covid_germany_delay_pmf.csv'), index=False)
    print(f"✓ Saved delay PMF to {output_dir}/covid_germany_delay_pmf.csv")
    
    # --- Final Summary ---
    print(f"\n{'='*80}")
    print("✅ Analysis Complete!")
    print(f"{'='*80}")
    print(f"\nData Citation:")
    print(f"  Dong E, Du H, Gardner L. An interactive web-based dashboard to track")
    print(f"  COVID-19 in real time. Lancet Inf Dis. 20(5):533-534.")
    print(f"  doi: 10.1016/S1473-3099(20)30120-1")
    print(f"  URL: https://github.com/CSSEGISandData/COVID-19")
    print(f"\nAll outputs saved to '{output_dir}/' directory:")
    print("  - covid_germany_cfr_curves.pdf")
    print("  - covid_germany_cases_deaths_daily.pdf")
    print("  - covid_germany_cases_deaths_ma7.pdf")
    print("  - covid_germany_derived_timeseries.csv (all curves with date index)")
    print("  - covid_germany_delay_pmf.csv (delay distribution)")
    print(f"{'='*80}\n")
    
    plt.show()
    
if __name__ == '__main__':
    main()
