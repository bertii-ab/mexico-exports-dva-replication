# -*- coding: utf-8 -*-
"""
Toda-Yamamoto Granger Causality Test
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import MinMaxScaler

# Load data
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
df_nardl_data['ln_TiVA_normalized'] = np.log(df_nardl_data['TiVA_DVA_ANN_normalized'])
df_nardl_data['ln_REER'] = np.log(df_nardl_data['REER (Real - Broad)'])

df_clean = df_nardl_data[['ln_TiVA_normalized', 'ln_REER']].dropna().copy()

# Toda-Yamamoto settings:
# optimal lag length p = 1 (determined by AIC)
# max integration order d_max = 1 (since both series are I(1))
# Estimate VAR(p + d_max) = VAR(2)

# Create lagged variables
df_clean['y'] = df_clean['ln_TiVA_normalized']
df_clean['x'] = df_clean['ln_REER']

df_clean['y_L1'] = df_clean['y'].shift(1)
df_clean['y_L2'] = df_clean['y'].shift(2)
df_clean['x_L1'] = df_clean['x'].shift(1)
df_clean['x_L2'] = df_clean['x'].shift(2)

df_var = df_clean.dropna().copy()

# Equation 1: y_t = c_1 + phi_1*y_{t-1} + phi_2*y_{t-2} + beta_1*x_{t-1} + beta_2*x_{t-2}
model_y = smf.ols(formula='y ~ y_L1 + y_L2 + x_L1 + x_L2', data=df_var).fit()

# Equation 2: x_t = c_2 + theta_1*y_{t-1} + theta_2*y_{t-2} + gamma_1*x_{t-1} + gamma_2*x_{t-2}
model_x = smf.ols(formula='x ~ y_L1 + y_L2 + x_L1 + x_L2', data=df_var).fit()

# Wald test for Granger causality:
# H0: x does not Granger cause y (beta_1 = 0)
wald_x_to_y = model_y.wald_test('x_L1 = 0')

# H0: y does not Granger cause x (theta_1 = 0)
wald_y_to_x = model_x.wald_test('y_L1 = 0')

print("=== Toda-Yamamoto Granger Causality Test ===")
print("\nREER -> DVA Granger Causality:")
print(f"Wald statistic: {wald_x_to_y.statistic[0][0]:.4f}")
print(f"p-value: {wald_x_to_y.pvalue:.4f}")

print("\nDVA -> REER Granger Causality:")
print(f"Wald statistic: {wald_y_to_x.statistic[0][0]:.4f}")
print(f"p-value: {wald_y_to_x.pvalue:.4f}")

# Write to text file
with open('toda_yamamoto_results.txt', 'w') as f:
    f.write("Toda-Yamamoto Granger Causality Test Results\n")
    f.write("=" * 60 + "\n\n")
    f.write("VAR(2) estimated: y_t = c_1 + phi_1*y_{t-1} + phi_2*y_{t-2} + beta_1*x_{t-1} + beta_2*x_{t-2}\n\n")
    f.write("REER -> DVA causality (H0: beta_1 = 0):\n")
    f.write(f"  Wald test statistic: {wald_x_to_y.statistic[0][0]:.4f}\n")
    f.write(f"  p-value: {wald_x_to_y.pvalue:.6f}\n\n")
    f.write("DVA -> REER causality (H0: theta_1 = 0):\n")
    f.write(f"  Wald test statistic: {wald_y_to_x.statistic[0][0]:.4f}\n")
    f.write(f"  p-value: {wald_y_to_x.pvalue:.6f}\n")
print("\nSaved toda_yamamoto_results.txt")
