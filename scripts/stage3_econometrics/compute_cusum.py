# -*- coding: utf-8 -*-
"""
Compute CUSUM and CUSUMSQ stability tests for the NARDL model
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import recursive_olsresiduals
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

df_subset = df_merged[df_merged['Time'] > '2010-Q4'].copy()
df_subset['ln_TiVA_normalized'] = np.log(df_subset['TiVA_DVA_ANN_normalized'])
df_subset['ln_REER'] = np.log(df_subset['REER (Real - Broad)'])
df_subset['d_ln_TiVA_normalized'] = df_subset['ln_TiVA_normalized'].diff()
df_subset['d_ln_reer'] = df_subset['ln_REER'].diff()

df_subset['reer_p'] = df_subset['d_ln_reer'].clip(lower=0)
df_subset['reer_n'] = df_subset['d_ln_reer'].clip(upper=0)
df_subset['REER_pos'] = df_subset['reer_p'].cumsum()
df_subset['REER_neg'] = df_subset['reer_n'].cumsum()

df_subset['L1_ln_TiVA_normalized'] = df_subset['ln_TiVA_normalized'].shift(1)
df_subset['d_ln_TiVA_normalized_L1'] = df_subset['d_ln_TiVA_normalized'].shift(1)
df_subset['d_ln_TiVA_normalized_L2'] = df_subset['d_ln_TiVA_normalized'].shift(2)

df_clean = df_subset[['d_ln_TiVA_normalized', 'L1_ln_TiVA_normalized', 'REER_pos', 'REER_neg', 
                       'd_ln_TiVA_normalized_L1', 'd_ln_TiVA_normalized_L2']].dropna()

# We can run OLS using statsmodels.api OLS to use diagnostic tools
Y = df_clean['d_ln_TiVA_normalized']
X = df_clean[['L1_ln_TiVA_normalized', 'REER_pos', 'REER_neg', 'd_ln_TiVA_normalized_L1', 'd_ln_TiVA_normalized_L2']]
X = sm.add_constant(X)

# Recursive residuals
res = recursive_olsresiduals(sm.OLS(Y, X).fit())

# statsmodels recursive_olsresiduals returns (resids, status, ...)
# res[0]: recursive residuals
# res[5]: cusum
# res[6]: cusum2 (CUSUMSQ)
cusum = res[5]
cusum2 = res[6]

# Plot data or check if they cross the 5% significance bounds
# For CUSUM, the 5% critical values are approximately constant bounds or lines.
# recursive_olsresiduals returns a tuple: (recursive_residuals, status, ...)
# Let's inspect the statsmodels function signature:
# recursive_olsresiduals(res_ols) -> returns:
# - raw_resids
# - recursive_resids
# - mat_W (coefficients)
# - cusum
# - cusum_ci
# - cusum2 (CUSUMSQ)
# - cusum2_ci
# Let's verify by printing them.
print("CUSUM array length:", len(res[3]))
print("CUSUM:", res[3])
print("CUSUM CI:", res[4])

# Let's check if the CUSUM values exceed the CI bounds
exceeds_cusum = np.any(np.abs(res[3]) > res[4][1])
print("Does CUSUM exceed 5% significance bounds?", exceeds_cusum)

# Let's check CUSUMSQ (which is res[5])
# The CUSUMSQ critical bounds:
# Let's see:
print("CUSUMSQ array length:", len(res[5]))
print("CUSUMSQ:", res[5])
