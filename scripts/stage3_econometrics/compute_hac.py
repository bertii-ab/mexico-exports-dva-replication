# -*- coding: utf-8 -*-
"""
Compute HAC (Newey-West) Standard Errors for the NARDL model
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

# Standard OLS model
formula = 'd_ln_TiVA_normalized ~ L1_ln_TiVA_normalized + REER_pos + REER_neg + d_ln_TiVA_normalized_L1 + d_ln_TiVA_normalized_L2'
model_standard = smf.ols(formula=formula, data=df_clean).fit()

# HAC (Newey-West) standard errors
# maxlags=3 (standard quarterly choice)
model_hac = smf.ols(formula=formula, data=df_clean).fit(cov_type='HAC', cov_kwds={'maxlags': 3})

print("==================== Standard OLS ====================")
print(model_standard.summary().tables[1])

print("\n==================== HAC (Newey-West) ====================")
print(model_hac.summary().tables[1])
