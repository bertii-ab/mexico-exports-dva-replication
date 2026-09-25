# -*- coding: utf-8 -*-
"""
Toda-Yamamoto Granger Causality Test on the full original series (from start to end)
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('./Interpolated_TiVA_DVA_Final.csv')
df_eer = pd.read_csv('./mexico_monthly_eer_bis_official.csv')
df_eer['Month'] = pd.to_datetime(df_eer['Month'])
df_eer_quarterly = df_eer.resample('QE', on='Month').mean().reset_index()
df_eer_quarterly['Time'] = df_eer_quarterly['Month'].dt.year.astype(str) + '-Q' + df_eer_quarterly['Month'].dt.quarter.astype(str)
df_merged = pd.merge(df, df_eer_quarterly[['Time', 'NEER (Nominal - Broad)', 'REER (Real - Broad)']], on='Time', how='inner')

print(f"Total merged data rows: {len(df_merged)}")
print(f"Start time: {df_merged['Time'].iloc[0]}, End time: {df_merged['Time'].iloc[-1]}")

# Use log variables directly without scaling to [0, 1] (which creates a zero value)
df_full = df_merged.copy()
df_full['ln_TiVA'] = np.log(df_full['TiVA_DVA_ANN'])
df_full['ln_REER'] = np.log(df_full['REER (Real - Broad)'])

df_clean = df_full[['Time', 'ln_TiVA', 'ln_REER']].dropna().copy()
df_clean['y'] = df_clean['ln_TiVA']
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
    
    print(f"Lag p = {p}: (N = {len(df_var)})")
    print(f"  REER -> DVA: statistic={wald_x_to_y.statistic:.4f}, p-value={wald_x_to_y.pvalue:.4f}")
    print(f"  DVA -> REER: statistic={wald_y_to_x.statistic:.4f}, p-value={wald_y_to_x.pvalue:.4f}")

for p in [1, 2, 3, 4, 5, 6, 7, 8]:
    test_ty(p)
