# -*- coding: utf-8 -*-
"""
Bootstrap Cumulative Dynamic Multipliers (IRFs) with 95% Confidence Intervals
following Shin, Yu & Greenwood-Nimmo (2014).
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

# Set seed for reproducibility
np.random.seed(42)

# Load and prepare data
df = pd.read_csv('./Interpolated_TiVA_DVA_Final.csv')
df_eer = pd.read_csv('./mexico_monthly_eer_bis_official.csv')
df_eer['Month'] = pd.to_datetime(df_eer['Month'])
df_eer_quarterly = df_eer.resample('QE', on='Month').mean().reset_index()
df_eer_quarterly['Time'] = df_eer_quarterly['Month'].dt.year.astype(str) + '-Q' + df_eer_quarterly['Month'].dt.quarter.astype(str)
df_merged = pd.merge(df, df_eer_quarterly[['Time', 'NEER (Nominal - Broad)', 'REER (Real - Broad)']], on='Time', how='inner')

scaler = MinMaxScaler()
tiva_data = df_merged['TiVA_DVA_ANN'].values.reshape(-1, 1)
df_merged['TiVA_DVA_ANN_normalized'] = scaler.fit_transform(tiva_data)

# Period index for Time
df_merged['Time_dt'] = pd.PeriodIndex(df_merged['Time'], freq='Q').to_timestamp()

# Subset from 2011 onwards (Time > 2010-Q3)
df_nardl_data = df_merged[df_merged['Time'] > '2010-Q3'].copy()

# Create log of normalized TiVA
df_nardl_data['ln_TiVA_normalized'] = np.log(df_nardl_data['TiVA_DVA_ANN_normalized'])
df_nardl_data['d_ln_TiVA_normalized'] = df_nardl_data['ln_TiVA_normalized'].diff()
df_nardl_data['L1_ln_TiVA_normalized'] = df_nardl_data['ln_TiVA_normalized'].shift(1)

# Ensure REER is logged and decomposed
df_nardl_data['ln_REER'] = np.log(df_nardl_data['REER (Real - Broad)'])
df_nardl_data['d_ln_reer'] = df_nardl_data['ln_REER'].diff()
df_nardl_data['reer_p'] = df_nardl_data['d_ln_reer'].clip(lower=0)
df_nardl_data['reer_n'] = df_nardl_data['d_ln_reer'].clip(upper=0)
df_nardl_data['REER_pos'] = df_nardl_data['reer_p'].cumsum()
df_nardl_data['REER_neg'] = df_nardl_data['reer_n'].cumsum()

# Create Lags 1 to 2
for i in range(1, 3):
    df_nardl_data[f'd_ln_TiVA_normalized_L{i}'] = df_nardl_data['d_ln_TiVA_normalized'].shift(i)

df_clean = df_nardl_data.dropna().copy()

# Fit the original model
formula = 'd_ln_TiVA_normalized ~ L1_ln_TiVA_normalized + REER_pos + REER_neg + d_ln_TiVA_normalized_L1 + d_ln_TiVA_normalized_L2'
model_orig = smf.ols(formula=formula, data=df_clean).fit()

# Get residuals
residuals = model_orig.resid
# Center residuals
residuals_centered = residuals - residuals.mean()

print("Original Model Parameters:")
print(model_orig.params)

# Function to simulate dynamic multipliers
def simulate_response(lam, phi, beta1, beta2, H, shock_direction=1.0):
    y = np.zeros(H + 3)
    dy = np.zeros(H + 3)
    x = np.zeros(H + 3)
    for t in range(2, H + 3):
        x[t] = shock_direction
        dy[t] = lam * y[t-1] + phi * x[t-1] + beta1 * dy[t-1] + beta2 * dy[t-2]
        y[t] = y[t-1] + dy[t]
    return y[2:H+3]

H = 20
# Original simulation
m_pos_orig = simulate_response(
    model_orig.params['L1_ln_TiVA_normalized'],
    model_orig.params['REER_pos'],
    model_orig.params['d_ln_TiVA_normalized_L1'],
    model_orig.params['d_ln_TiVA_normalized_L2'],
    H, shock_direction=1.0
)
m_neg_orig = simulate_response(
    model_orig.params['L1_ln_TiVA_normalized'],
    model_orig.params['REER_neg'],
    model_orig.params['d_ln_TiVA_normalized_L1'],
    model_orig.params['d_ln_TiVA_normalized_L2'],
    H, shock_direction=-1.0
)

# Bootstrap settings
B = 1000
bootstrap_m_pos = []
bootstrap_m_neg = []

# To perform residual bootstrap, we need to generate y* recursively
T_est = len(df_clean)
y_orig = df_clean['ln_TiVA_normalized'].values
dy_orig = df_clean['d_ln_TiVA_normalized'].values
x_pos = df_clean['REER_pos'].values
x_neg = df_clean['REER_neg'].values

alpha0 = model_orig.params['Intercept']
lam = model_orig.params['L1_ln_TiVA_normalized']
phi_pos = model_orig.params['REER_pos']
phi_neg = model_orig.params['REER_neg']
beta1 = model_orig.params['d_ln_TiVA_normalized_L1']
beta2 = model_orig.params['d_ln_TiVA_normalized_L2']

print("\nStarting bootstrap...")
for b in range(B):
    # Resample residuals
    e_boot = np.random.choice(residuals_centered, size=T_est, replace=True)
    
    # Reconstruct dy* and y*
    dy_boot = np.zeros(T_est)
    y_boot = np.zeros(T_est)
    
    # Initialize with original values for the first 2 observations
    dy_boot[0] = dy_orig[0]
    dy_boot[1] = dy_orig[1]
    y_boot[0] = y_orig[0]
    y_boot[1] = y_orig[1]
    
    for t in range(2, T_est):
        dy_boot[t] = (alpha0 + 
                      lam * y_boot[t-1] + 
                      phi_pos * x_pos[t] + 
                      phi_neg * x_neg[t] + 
                      beta1 * dy_boot[t-1] + 
                      beta2 * dy_boot[t-2] + 
                      e_boot[t])
        y_boot[t] = y_boot[t-1] + dy_boot[t]
        
    # Create temporary dataframe for OLS fit
    df_boot = pd.DataFrame({
        'd_ln_TiVA_normalized': dy_boot,
        'L1_ln_TiVA_normalized': np.roll(y_boot, 1),
        'REER_pos': x_pos,
        'REER_neg': x_neg,
        'd_ln_TiVA_normalized_L1': np.roll(dy_boot, 1),
        'd_ln_TiVA_normalized_L2': np.roll(dy_boot, 2)
    })
    # Drop first two rows due to shift/roll indexing
    df_boot = df_boot.iloc[2:]
    
    try:
        model_boot = smf.ols(formula=formula, data=df_boot).fit()
        
        # Check stability
        l_ecm = model_boot.params['L1_ln_TiVA_normalized']
        if l_ecm >= 0 or l_ecm < -2.0:
            continue
            
        m_pos_b = simulate_response(
            l_ecm,
            model_boot.params['REER_pos'],
            model_boot.params['d_ln_TiVA_normalized_L1'],
            model_boot.params['d_ln_TiVA_normalized_L2'],
            H, shock_direction=1.0
        )
        m_neg_b = simulate_response(
            l_ecm,
            model_boot.params['REER_neg'],
            model_boot.params['d_ln_TiVA_normalized_L1'],
            model_boot.params['d_ln_TiVA_normalized_L2'],
            H, shock_direction=-1.0
        )
        
        bootstrap_m_pos.append(m_pos_b)
        bootstrap_m_neg.append(m_neg_b)
    except:
        pass

bootstrap_m_pos = np.array(bootstrap_m_pos)
bootstrap_m_neg = np.array(bootstrap_m_neg)

print(f"Completed bootstrap. Valid runs: {len(bootstrap_m_pos)}")

# Calculate percentiles for 95% CI
lower_pos = np.percentile(bootstrap_m_pos, 2.5, axis=0)
upper_pos = np.percentile(bootstrap_m_pos, 97.5, axis=0)
lower_neg = np.percentile(bootstrap_m_neg, 2.5, axis=0)
upper_neg = np.percentile(bootstrap_m_neg, 97.5, axis=0)

# Create Output DataFrame
horizons = list(range(H + 1))
irf_df = pd.DataFrame({
    'Horizon': horizons,
    'IRF_Appreciation': m_pos_orig,
    'Appreciation_Lower': lower_pos,
    'Appreciation_Upper': upper_pos,
    'IRF_Depreciation': m_neg_orig,
    'Depreciation_Lower': lower_neg,
    'Depreciation_Upper': upper_neg,
    'LR_Appreciation': -phi_pos / lam,
    'LR_Depreciation': phi_neg / lam
})

# Save to CSV
irf_df.to_csv('plot_irf_dynamic_multipliers.csv', index=False)
print("Saved plot_irf_dynamic_multipliers.csv with confidence intervals")
print(irf_df.head(10))
