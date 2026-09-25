# -*- coding: utf-8 -*-
"""
Toda-Yamamoto Granger Causality Test with different lag lengths
"""
import numpy as np
import pandas as pd
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
df_clean['y'] = df_clean['ln_TiVA_normalized']
df_clean['x'] = df_clean['ln_REER']

# Function to run Toda-Yamamoto Granger causality
def test_ty(p, d_max=1):
    df_temp = df_clean.copy()
    # Create lags up to p + d_max
    y_lags = []
    x_lags = []
    for i in range(1, p + d_max + 1):
        df_temp[f'y_L{i}'] = df_temp['y'].shift(i)
        df_temp[f'x_L{i}'] = df_temp['x'].shift(i)
        y_lags.append(f'y_L{i}')
        x_lags.append(f'x_L{i}')
    
    df_var = df_temp.dropna().copy()
    
    formula_y = 'y ~ ' + ' + '.join(y_lags + x_lags)
    formula_x = 'x ~ ' + ' + '.join(y_lags + x_lags)
    
    model_y = smf.ols(formula=formula_y, data=df_var).fit()
    model_x = smf.ols(formula=formula_x, data=df_var).fit()
    
    # Wald test for Granger causality:
    # H0: x does not cause y (first p lags of x are 0)
    hyp_x_to_y = ', '.join([f'x_L{i} = 0' for i in range(1, p + 1)])
    wald_x_to_y = model_y.wald_test(hyp_x_to_y, scalar=True)
    
    # H0: y does not cause x (first p lags of y are 0)
    hyp_y_to_x = ', '.join([f'y_L{i} = 0' for i in range(1, p + 1)])
    wald_y_to_x = model_x.wald_test(hyp_y_to_x, scalar=True)
    
    print(f"Lag p = {p}:")
    print(f"  REER -> DVA: statistic={wald_x_to_y.statistic:.4f}, p-value={wald_x_to_y.pvalue:.4f}")
    print(f"  DVA -> REER: statistic={wald_y_to_x.statistic:.4f}, p-value={wald_y_to_x.pvalue:.4f}")

for p in [1, 2, 3, 4, 5, 6, 7, 8]:
    test_ty(p)
