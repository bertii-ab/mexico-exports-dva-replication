"""
Stage 3: Non-linear ARDL (NARDL) Estimation
Estimates short- and long-run asymmetric transmission of real effective exchange
rate (REER) movements to domestic value added (DVA) in Mexican exports.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load the dataset
file_path = './Interpolated_TiVA_DVA_Final.csv'
df = pd.read_csv(file_path)

# Display the first few rows
print(df.head())

# Load the second dataset
file_path_eer = './mexico_monthly_eer_bis_official.csv'
df_eer = pd.read_csv(file_path_eer)

# Display the first few rows
print(df_eer.head())

# Convert Month to datetime
df_eer['Month'] = pd.to_datetime(df_eer['Month'])

# Resample to quarterly frequency (QE = Quarter End), taking the mean of the values
df_eer_quarterly = df_eer.resample('QE', on='Month').mean().reset_index()

# Create a 'Time' column in YYYY-QX format to match the first dataframe
df_eer_quarterly['Time'] = df_eer_quarterly['Month'].dt.year.astype(str) + '-Q' + df_eer_quarterly['Month'].dt.quarter.astype(str)

# Display the quarterly dataframe
print(df_eer_quarterly.head())

# Merge the dataframes on the 'Time' column
df_merged = pd.merge(df, df_eer_quarterly[['Time', 'NEER (Nominal - Broad)', 'REER (Real - Broad)']], on='Time', how='inner')

# Display the merged dataframe
print(df_merged.head())
print(f"Merged dataframe shape: {df_merged.shape}")

# 2. Extract the target series and define the sample size
y = df['TiVA_DVA_ANN'].values
T = len(y)

# 3. Define the independent variables for the regression
# We test for a shift in the mean, so we just use an intercept (constant)
X = np.ones((T, 1))
k = X.shape[1]  # Number of parameters (1 for the intercept)

# 4. Fit the restricted model (Full sample, assuming NO structural break)
model_restricted = sm.OLS(y, X).fit()
ssr_restricted = model_restricted.ssr

# 5. Setup the Andrews Test parameters
# Standard practice is to trim the first and last 15% of the data
# to ensure sufficient sample sizes for the subsample regressions.
trimming_fraction = 0.15
start_idx = int(T * trimming_fraction)
end_idx = int(T * (1 - trimming_fraction))

wald_stats = []
candidate_breaks = range(start_idx, end_idx)

# 6. Loop over all candidate breakpoints to calculate the Chow F-statistic
for tau in candidate_breaks:
    # Subsample 1: Before the candidate break
    y1, X1 = y[:tau], X[:tau]
    model1 = sm.OLS(y1, X1).fit()
    ssr1 = model1.ssr

    # Subsample 2: After the candidate break
    y2, X2 = y[tau:], X[tau:]
    model2 = sm.OLS(y2, X2).fit()
    ssr2 = model2.ssr

    # Calculate the unrestricted Sum of Squared Residuals (SSR)
    ssr_unrestricted = ssr1 + ssr2

    # Compute the F-Statistic (Wald variant) for this candidate break
    # Formula: [(SSR_restricted - SSR_unrestricted) / k] / [SSR_unrestricted / (T - 2k)]
    f_stat = ((ssr_restricted - ssr_unrestricted) / k) / (ssr_unrestricted / (T - 2 * k))
    wald_stats.append(f_stat)

# 7. Identify the Sup-Wald statistic (The Andrews Test Statistic)
sup_wald_stat = np.max(wald_stats)
break_idx = candidate_breaks[np.argmax(wald_stats)]
break_time = df['Time'].iloc[break_idx]

print(f"Andrews Sup-Wald Statistic: {sup_wald_stat:.4f}")
print(f"Estimated Structural Breakpoint: Index {break_idx} ({break_time})")

# 8. Visualization of the test statistics - Save as PNG
plt.figure(figsize=(10, 5))
plt.plot(df['Time'].iloc[candidate_breaks], wald_stats, color='blue', label='Wald F-Statistic Path')
plt.axvline(x=break_time, color='red', linestyle='--', label=f'Supremum Break ({break_time})')
plt.title('Andrews Sup-Wald Test for Structural Change in TiVA_DVA_ANN')
plt.xlabel('Time (Quarterly)')
plt.ylabel('F-Statistic')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig('plot_structural_break.png', dpi=150)
plt.close()
print("Saved plot_structural_break.png")

# Export structural break plot data to CSV
break_plot_df = pd.DataFrame({
    'Time': df['Time'].iloc[candidate_breaks].values,
    'Wald_F_Statistic': wald_stats
})
break_plot_df.to_csv('plot_structural_break.csv', index=False)
print("Saved plot_structural_break.csv")

# Create a subset for data after 2010 Q3 (starting from 2010-Q4)
df_subset = df_merged[df_merged['Time'] > '2010-Q4'].copy()

# Verify the range of the new subset
print(f"Subset created with {len(df_subset)} rows.")
print(f"Range: {df_subset['Time'].min()} to {df_subset['Time'].max()}")
print(df_subset.head())

cols_of_interest = ['TiVA_DVA_ANN', 'REER (Real - Broad)']
stats = df_subset[cols_of_interest].describe().T

# Adding skewness and kurtosis for a more complete profile
stats['skew'] = df_subset[cols_of_interest].skew()
stats['kurtosis'] = df_subset[cols_of_interest].kurtosis()

print("--- Descriptive Statistics: Mexico TiVA and REER ---")
print(stats.round(4))

# Export descriptive stats
stats.round(4).to_csv('nardl_descriptive_stats.txt', sep='\t')
print("Saved nardl_descriptive_stats.txt")

from statsmodels.tsa.stattools import adfuller

def run_adf_test(series, name, adf_results_list):
    series = series.dropna()
    result = adfuller(series)
    print(f"--- ADF Test: {name} ---")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")
    status = "Stationary" if result[1] < 0.05 else "Non-Stationary"
    print(f"Conclusion: {status}\n")
    adf_results_list.append({
        'Variable': name,
        'ADF_Statistic': round(result[0], 4),
        'p_value': round(result[1], 4),
        'Critical_1pct': round(result[4]['1%'], 4),
        'Critical_5pct': round(result[4]['5%'], 4),
        'Critical_10pct': round(result[4]['10%'], 4),
        'Conclusion': status
    })

# Prepare Log-Differences
df_subset['ln_TiVA'] = np.log(df_subset['TiVA_DVA_ANN'])
df_subset['ln_REER'] = np.log(df_subset['REER (Real - Broad)'])
df_subset['d_ln_TiVA'] = df_subset['ln_TiVA'].diff()
df_subset['d_ln_REER'] = df_subset['ln_REER'].diff()

# Execution
adf_results = []
print("UNIT ROOT TESTS (LEVELS)")
run_adf_test(df_subset['TiVA_DVA_ANN'], "TiVA (Levels)", adf_results)
run_adf_test(df_subset['REER (Real - Broad)'], "REER (Levels)", adf_results)

print("UNIT ROOT TESTS (FIRST LOG DIFFERENCES)")
run_adf_test(df_subset['d_ln_TiVA'], "ln_TiVA (1st Diff)", adf_results)
run_adf_test(df_subset['d_ln_REER'], "ln_REER (1st Diff)", adf_results)

# Export ADF results
adf_df = pd.DataFrame(adf_results)
adf_df.to_csv('nardl_adf_tests.txt', sep='\t', index=False)
print("Saved nardl_adf_tests.txt")

# Create a subset from 2011 onwards
df_subset_2011 = df_subset.copy()

# Verify the range of the new subset
print(f"Subset created with {len(df_subset)} rows.")
print(f"Range: {df_subset['Time'].min()} to {df_subset['Time'].max()}")

# Convert 'Time' (e.g., '1995-Q1') to proper datetime format using PeriodIndex
df_subset_2011['Time'] = pd.PeriodIndex(df_subset_2011['Time'], freq='Q').to_timestamp()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot TiVA_DVA_ANN on the first axis
color1 = 'tab:blue'
ax1.set_xlabel('Año')
ax1.set_ylabel('TiVA_DVA_ANN', color=color1)
ax1.plot(df_subset_2011['Time'], df_subset_2011['TiVA_DVA_ANN'], color=color1, label='TiVA_DVA_ANN', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

# Create a second axis for REER
ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('REER (Real - Broad)', color=color2)
ax2.plot(df_subset_2011['Time'], df_subset_2011['REER (Real - Broad)'], color=color2, label='REER', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Evolución de TiVA_DVA_ANN y REER (Trimestral)')
fig.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig('plot_dva_reer_timeseries.png', dpi=150)
plt.close()
print("Saved plot_dva_reer_timeseries.png")

# Export DVA-REER time series plot data to CSV
ts_plot_df = df_subset_2011[['Time', 'TiVA_DVA_ANN', 'REER (Real - Broad)']].copy()
ts_plot_df.to_csv('plot_dva_reer_timeseries.csv', index=False)
print("Saved plot_dva_reer_timeseries.csv")

"""### NARDL: Asymmetrical Effects Analysis
We decompose the `REER` variable into its positive and negative partial sums to capture asymmetric responses.
"""

# Visualize the partial sums to see the asymmetry

# 1. Calculate the log and log-difference of REER
df_subset_2011['ln_REER'] = np.log(df_subset_2011['REER (Real - Broad)'])
df_subset_2011['d_ln_reer'] = df_subset_2011['ln_REER'].diff()

# 2. Decompose log differences into positive and negative shocks
df_subset_2011['reer_p'] = df_subset_2011['d_ln_reer'].clip(lower=0)
df_subset_2011['reer_n'] = df_subset_2011['d_ln_reer'].clip(upper=0)

# 3. Calculate partial sums (accumulated asymmetric effects)
df_subset_2011['REER_pos'] = df_subset_2011['reer_p'].cumsum()
df_subset_2011['REER_neg'] = df_subset_2011['reer_n'].cumsum()

# 4. Visualization
plt.figure(figsize=(10, 5))
plt.plot(df_subset_2011['Time'], df_subset_2011['REER_pos'], label='REER+ (Appreciations)', color='green')
plt.plot(df_subset_2011['Time'], df_subset_2011['REER_neg'], label='REER- (Depreciations)', color='red')
plt.title('Partial Sums of REER (Asymmetric Components)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plot_partial_sums.png', dpi=150)
plt.close()
print("Saved plot_partial_sums.png")

# Export partial sums plot data to CSV
ps_plot_df = df_subset_2011[['Time', 'REER_pos', 'REER_neg']].copy()
ps_plot_df.to_csv('plot_partial_sums.csv', index=False)
print("Saved plot_partial_sums.csv")

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Reshape the 'TiVA_DVA_ANN' column to a 2D array for the scaler
tiva_data = df_merged['TiVA_DVA_ANN'].values.reshape(-1, 1)

# Fit and transform the data
df_merged['TiVA_DVA_ANN_normalized'] = scaler.fit_transform(tiva_data)

# Display the first few rows with the new normalized column
print("DataFrame with normalized TiVA_DVA_ANN:")
print(df_merged[['Time', 'TiVA_DVA_ANN', 'TiVA_DVA_ANN_normalized']].head())

import statsmodels.formula.api as smf
import itertools
import ast

# --- 0. Data Preparation for NARDL with Normalized TiVA ---
# FIX: Ensure 'Time' is a datetime object before filtering
if not pd.api.types.is_datetime64_any_dtype(df_merged['Time']):
    df_merged['Time_dt'] = pd.to_datetime(df_merged['Time'].str.replace(r'-(Q\d)', lambda x: x.group(0), regex=True))
    # Alternative for Quarterly strings:
    df_merged['Time_dt'] = pd.PeriodIndex(df_merged['Time'], freq='Q').to_timestamp()

# Create a subset from df_merged from 2011 onwards
df_nardl_data = df_merged[df_merged['Time'] > '2010-Q3'].copy()

# Create natural log of normalized TiVA
df_nardl_data['ln_TiVA_normalized'] = np.log(df_nardl_data['TiVA_DVA_ANN_normalized'])

# Calculate the difference of the logged normalized dependent variable
df_nardl_data['d_ln_TiVA_normalized'] = df_nardl_data['ln_TiVA_normalized'].diff()

# Lagged level of the LOGGED normalized dependent variable
df_nardl_data['L1_ln_TiVA_normalized'] = df_nardl_data['ln_TiVA_normalized'].shift(1)

# Ensure REER is logged and decomposed
df_nardl_data['ln_REER'] = np.log(df_nardl_data['REER (Real - Broad)'])
df_nardl_data['d_ln_reer'] = df_nardl_data['ln_REER'].diff()
df_nardl_data['reer_p'] = df_nardl_data['d_ln_reer'].clip(lower=0)
df_nardl_data['reer_n'] = df_nardl_data['d_ln_reer'].clip(upper=0)
df_nardl_data['REER_pos'] = df_nardl_data['reer_p'].cumsum()
df_nardl_data['REER_neg'] = df_nardl_data['reer_n'].cumsum()

# Create Lags 1 to 4
for i in range(1, 5):
    df_nardl_data[f'd_ln_TiVA_normalized_L{i}'] = df_nardl_data['d_ln_TiVA_normalized'].shift(i)
    df_nardl_data[f'd_ln_REER_L{i}'] = df_nardl_data['d_ln_reer'].shift(i)

df_clean_nardl = df_nardl_data.dropna()

# --- 1. THE MULTI-VARIABLE GRID SEARCH ---
base_terms_normalized = ['L1_ln_TiVA_normalized', 'REER_pos', 'REER_neg']
lag_options = [1, 2, 3, 4]
all_combos = []
for r in range(len(lag_options) + 1):
    all_combos.extend(itertools.combinations(lag_options, r))

grid_results_normalized = []
for ar_combo in all_combos:
    for reer_combo in all_combos:
        ar_terms = [f"d_ln_TiVA_normalized_L{i}" for i in ar_combo]
        reer_terms = [f"d_ln_REER_L{i}" for i in reer_combo]
        formula = 'd_ln_TiVA_normalized ~ ' + ' + '.join(base_terms_normalized + ar_terms + reer_terms)
        try:
            model = smf.ols(formula=formula, data=df_clean_nardl).fit()
            coef_l1 = model.params.get('L1_ln_TiVA_normalized', np.nan)
            coef_pos = model.params.get('REER_pos', np.nan)
            coef_neg = model.params.get('REER_neg', np.nan)
            theta_pos = -coef_pos / coef_l1 if coef_l1 != 0 else np.nan
            theta_neg = -coef_neg / coef_l1 if coef_l1 != 0 else np.nan
            
            grid_results_normalized.append({
                'AR_Lags': str(ar_combo) if ar_combo else "None",
                'REER_Lags': str(reer_combo) if reer_combo else "None",
                'AIC': model.aic,
                'coef_L1_TiVA_normalized': coef_l1,
                'pval_L1_TiVA_normalized': model.pvalues.get('L1_ln_TiVA_normalized', np.nan),
                'pval_REER_pos': model.pvalues.get('REER_pos', np.nan),
                'pval_REER_neg': model.pvalues.get('REER_neg', np.nan),
                'theta_pos': theta_pos,
                'theta_neg': theta_neg
            })
        except: pass

grid_df_normalized = pd.DataFrame(grid_results_normalized).sort_values(by='AIC').reset_index(drop=True)
valid_thetas = grid_df_normalized[grid_df_normalized['coef_L1_TiVA_normalized'] < 0]
print("\n=== DISTRIBUTION OF PRINCIPAL PARAMETERS (GRID SEARCH) ===")
print("Theta+ (Appreciation Multiplier) Stats:")
print(valid_thetas['theta_pos'].describe())
print("\nTheta- (Depreciation Multiplier) Stats:")
print(valid_thetas['theta_neg'].describe())
print("==========================================================\n")
print("--- Top 10 NARDL Models (AIC) ---")
print(grid_df_normalized.head(10))

# Filtering for Cointegration
final_grid_df = grid_df_normalized[
    (grid_df_normalized['coef_L1_TiVA_normalized'] < 0) &
    (grid_df_normalized['pval_L1_TiVA_normalized'] < 0.10) &
    ((grid_df_normalized['pval_REER_pos'] < 0.10) | (grid_df_normalized['pval_REER_neg'] < 0.10))
].copy()

if not final_grid_df.empty:
    best = final_grid_df.iloc[0]
    print(f"\nBest Model: AR {best['AR_Lags']}, REER {best['REER_Lags']}")
    # Re-fit and show summary
    ar_l = ast.literal_eval(best['AR_Lags']) if best['AR_Lags'] != 'None' else []
    reer_l = ast.literal_eval(best['REER_Lags']) if best['REER_Lags'] != 'None' else []
    best_formula = 'd_ln_TiVA_normalized ~ ' + ' + '.join(base_terms_normalized + [f'd_ln_TiVA_normalized_L{i}' for i in ar_l] + [f'd_ln_REER_L{i}' for i in reer_l])
    best_model = smf.ols(formula=best_formula, data=df_clean_nardl).fit()
    print(best_model.summary())

    # Export NARDL coefficients
    coeff_df = pd.DataFrame({
        'Variable': best_model.params.index,
        'Coefficient': best_model.params.values,
        'Std_Error': best_model.bse.values,
        't_Statistic': best_model.tvalues.values,
        'p_Value': best_model.pvalues.values,
        'CI_Lower': best_model.conf_int()[0].values,
        'CI_Upper': best_model.conf_int()[1].values
    })
    coeff_df.to_csv('nardl_coefficients.txt', sep='\t', index=False)
    print("Saved nardl_coefficients.txt")

    # Export model diagnostics
    with open('nardl_diagnostics.txt', 'w') as f:
        f.write("NARDL Model Diagnostics\n")
        f.write("=" * 60 + "\n")
        f.write(f"R-squared: {best_model.rsquared:.6f}\n")
        f.write(f"Adjusted R-squared: {best_model.rsquared_adj:.6f}\n")
        f.write(f"F-statistic: {best_model.fvalue:.4f}\n")
        f.write(f"F-statistic p-value: {best_model.f_pvalue:.6f}\n")
        f.write(f"AIC: {best_model.aic:.4f}\n")
        f.write(f"BIC: {best_model.bic:.4f}\n")
        f.write(f"Log-Likelihood: {best_model.llf:.4f}\n")
        f.write(f"Number of Observations: {best_model.nobs:.0f}\n")
        f.write(f"Degrees of Freedom (Model): {best_model.df_model:.0f}\n")
        f.write(f"Degrees of Freedom (Residuals): {best_model.df_resid:.0f}\n")
        f.write(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(best_model.resid):.4f}\n")
        f.write(f"\nFormula: {best_formula}\n")
        f.write(f"Best AR Lags: {best['AR_Lags']}\n")
        f.write(f"Best REER Lags: {best['REER_Lags']}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("\nFull Summary:\n")
        f.write(str(best_model.summary()))
    print("Saved nardl_diagnostics.txt")

else:
    print("\nNo models met the criteria.")

"""### Post-Estimation: Bounds Test and Long-Run Asymmetry
We check for cointegration using the F-statistic (Bounds Test) and evaluate if the long-run multipliers for positive and negative REER shocks are statistically different.
"""

if 'best_model' in locals():
    # 1. Bounds Test (Wald F-statistic for Cointegration)
    # Null Hypothesis (H0): L1_ln_TiVA_normalized = REER_pos = REER_neg = 0
    # Note: Compare the result to Pesaran et al. (2001) critical values for k=2
    bounds_test = best_model.wald_test("L1_ln_TiVA_normalized = 0, REER_pos = 0, REER_neg = 0")

    # 2. Long-Run Multiplier Calculation
    # Formula: L = -(Beta_shock / Lambda_ecm)
    lambda_ecm = best_model.params['L1_ln_TiVA_normalized']
    beta_pos = best_model.params['REER_pos']
    beta_neg = best_model.params['REER_neg']

    lr_pos = -(beta_pos / lambda_ecm)
    lr_neg = -(beta_neg / lambda_ecm)

    # 3. Wald Test for Long-Run Asymmetry
    # Null Hypothesis (H0): lr_pos = lr_neg (Impact of appreciation = Impact of depreciation)
    # This is equivalent to testing REER_pos / L1_ln_TiVA = REER_neg / L1_ln_TiVA
    asymmetry_test = best_model.wald_test("REER_pos = REER_neg")

    print("--- ARDL Bounds Test (F-statistic) ---")
    print(bounds_test)
    print(f"\n--- Long-Run Multipliers ---")
    print(f"Long-run impact of appreciation (REER+): {lr_pos:.4f}")
    print(f"Long-run impact of depreciation (REER-): {lr_neg:.4f}")
    print(f"\n--- Wald Test for Long-Run Asymmetry ---")
    print(asymmetry_test)

    if asymmetry_test.pvalue < 0.05:
        print("\nConclusion: Reject H0 of symmetry. The long-run relationship is statistically ASYMMETRIC.")
    else:
        print("\nConclusion: Fail to reject H0. The relationship is statistically SYMMETRIC.")

    # Export Wald test results
    with open('nardl_wald_test.txt', 'w') as f:
        f.write("NARDL Wald Test Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("--- ARDL Bounds Test (F-statistic) ---\n")
        f.write(f"H0: L1_ln_TiVA_normalized = REER_pos = REER_neg = 0\n")
        f.write(str(bounds_test) + "\n\n")
        f.write("--- Long-Run Multipliers ---\n")
        f.write(f"Lambda (ECM coefficient): {lambda_ecm:.6f}\n")
        f.write(f"Beta_pos (REER+): {beta_pos:.6f}\n")
        f.write(f"Beta_neg (REER-): {beta_neg:.6f}\n")
        f.write(f"Long-run impact of appreciation (REER+): {lr_pos:.6f}\n")
        f.write(f"Long-run impact of depreciation (REER-): {lr_neg:.6f}\n\n")
        f.write("--- Wald Test for Long-Run Asymmetry ---\n")
        f.write(f"H0: REER_pos = REER_neg (Symmetric)\n")
        f.write(str(asymmetry_test) + "\n\n")
        if asymmetry_test.pvalue < 0.05:
            f.write("Conclusion: Reject H0 of symmetry. The long-run relationship is statistically ASYMMETRIC.\n")
        else:
            f.write("Conclusion: Fail to reject H0. The relationship is statistically SYMMETRIC.\n")
    print("Saved nardl_wald_test.txt")

else:
    print("Best model not found. Please run the NARDL grid search cell first.")

import statsmodels.tsa.vector_ar.vecm as vecm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# Prepare the data using df_clean_nardl from previous steps
# We'll use the logged normalized TiVA and logged REER for cointegration tests
data_for_johansen = df_clean_nardl[['ln_TiVA_normalized', 'ln_REER']]

print("--- Augmented Dickey-Fuller Test (ADF) for Stationarity ---")
for col in data_for_johansen.columns:
    result = adfuller(data_for_johansen[col])
    print(f"ADF Test for {col}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Critical Values (1%, 5%, 10%): {result[4]}")
    if result[1] > 0.05: # Typically, we check for non-stationarity at levels
        print(f"  Conclusion: {col} is likely non-stationary (Fail to reject H0).")
    else:
        print(f"  Conclusion: {col} is likely stationary (Reject H0).")
    print("\n")

# --- Determine Optimal Lag Order for VAR Model ---
# The Johansen test requires the lag order of the VAR model in levels (p) or in differences (k_ar_diff = p-1)
# We'll use VAR().select_order() to find the optimal lag for the levels data.
max_lags = 4 # Based on quarterly data, 4 lags represent one year
var_model = VAR(data_for_johansen)
lag_selection_results = var_model.select_order(maxlags=max_lags)
optimal_lag_order = lag_selection_results.aic # Use AIC for optimal lag order (p)
print(f"--- Optimal VAR Lag Order (based on AIC): {optimal_lag_order} ---")
k_ar_diff = optimal_lag_order - 1 # k_ar_diff for Johansen is p-1

# --- Perform Johansen Cointegration Test ---
# det_order=-1: no deterministic trend, no constant (use if data is already demeaned or differenced to remove constant)
# det_order=0: constant, no trend (suitable for I(1) series with a constant mean)
# det_order=1: constant and linear trend

# Given we have log-transformed data, a constant term is usually appropriate.
# If series were trend-stationary, a trend term might be needed.
# For I(1) series, typically det_order=0 (constant in cointegrating relations) or det_order=1 (trend in cointegrating relations)
# Let's start with det_order=0 (constant in the cointegration relation)

result = vecm.coint_johansen(data_for_johansen, det_order=0, k_ar_diff=k_ar_diff)

print("\n--- Johansen Cointegration Test Results ---")
print("Eigenvalues:", result.eig.round(4))
print("\nTrace Statistic and Critical Values:")
print("  Test Stat (Trace): ", result.lr1.round(4))
print("  Critical Values (90%, 95%, 99%):", result.cvt.round(4))
print("\nMax-Eigen Statistic and Critical Values:")
print("  Test Stat (Max-Eigen):", result.lr2.round(4))
print("  Critical Values (90%, 95%, 99%):", result.cvm.round(4))

print("\nInterpretation:")
print("  The Trace and Max-Eigen statistics are compared to their critical values.")
print("  If the test statistic is greater than the critical value, we reject the null hypothesis of r cointegrating relations.")
print("  The null hypothesis (H0) for the Trace test is that there are at most r cointegrating vectors.")
print("  The null hypothesis (H0) for the Max-Eigen test is that there are at most r cointegrating vectors against r+1.")
print("  r=0: No cointegration.")
print("  r=1: One cointegrating vector.")

# Explicitly checking for cointegration (r=0 vs r>=1)
# Using 5% critical values for trace test
if result.lr1[0] > result.cvt[0, 1]: # Compare trace stat for r=0 with 95% critical value
    print("\nBased on Trace Statistic: Reject H0 (r=0). There is evidence of at least one cointegrating relationship.")
else:
    print("\nBased on Trace Statistic: Fail to reject H0 (r=0). No evidence of cointegration.")

# Using 5% critical values for max-eigen test
if result.lr2[0] > result.cvm[0, 1]: # Compare max-eigen stat for r=0 with 95% critical value
    print("Based on Max-Eigen Statistic: Reject H0 (r=0). There is evidence of at least one cointegrating relationship.")
else:
    print("Based on Max-Eigen Statistic: Fail to reject H0 (r=0). No evidence of cointegration.")

# Export Granger causality / Johansen results
with open('nardl_granger.txt', 'w') as f:
    f.write("Johansen Cointegration Test Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Optimal VAR Lag Order (AIC): {optimal_lag_order}\n")
    f.write(f"k_ar_diff: {k_ar_diff}\n\n")
    f.write(f"Eigenvalues: {result.eig.round(4)}\n\n")
    f.write("Trace Statistic and Critical Values:\n")
    f.write(f"  Test Stat (Trace):  {result.lr1.round(4)}\n")
    f.write(f"  Critical Values (90%, 95%, 99%): {result.cvt.round(4)}\n\n")
    f.write("Max-Eigen Statistic and Critical Values:\n")
    f.write(f"  Test Stat (Max-Eigen): {result.lr2.round(4)}\n")
    f.write(f"  Critical Values (90%, 95%, 99%): {result.cvm.round(4)}\n\n")
    if result.lr1[0] > result.cvt[0, 1]:
        f.write("Trace Test: Reject H0 (r=0). Evidence of cointegration.\n")
    else:
        f.write("Trace Test: Fail to reject H0 (r=0). No evidence of cointegration.\n")
    if result.lr2[0] > result.cvm[0, 1]:
        f.write("Max-Eigen Test: Reject H0 (r=0). Evidence of cointegration.\n")
    else:
        f.write("Max-Eigen Test: Fail to reject H0 (r=0). No evidence of cointegration.\n")
print("Saved nardl_granger.txt")

print("\n\n=== ALL NARDL SCRIPTS COMPLETED SUCCESSFULLY ===")