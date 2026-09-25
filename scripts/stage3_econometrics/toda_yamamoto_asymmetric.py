# -*- coding: utf-8 -*-
"""
Asymmetric Toda-Yamamoto Granger Causality Test
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tsa.api import VAR
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

df_nardl_data['d_ln_reer'] = df_nardl_data['ln_REER'].diff()
df_nardl_data['reer_p'] = df_nardl_data['d_ln_reer'].clip(lower=0)
df_nardl_data['reer_n'] = df_nardl_data['d_ln_reer'].clip(upper=0)
df_nardl_data['REER_pos'] = df_nardl_data['reer_p'].cumsum()
df_nardl_data['REER_neg'] = df_nardl_data['reer_n'].cumsum()

df_clean = df_nardl_data[['ln_TiVA_normalized', 'REER_pos', 'REER_neg']].dropna().copy()
df_clean.columns = ['y', 'x_pos', 'x_neg']

# Determine optimal lag order for the 3-variable VAR using statsmodels VAR
var_data = df_clean.copy()
var_model = VAR(var_data)
lag_selection = var_model.select_order(maxlags=4)
print("VAR Lag Selection (AIC):", lag_selection.aic)
print("VAR Lag Selection (BIC):", lag_selection.bic)
p_optimal = lag_selection.aic

# Let's run Toda-Yamamoto with different lag lengths p
def test_asymmetric_ty(p, d_max=1):
    df_temp = df_clean.copy()
    y_lags = []
    pos_lags = []
    neg_lags = []
    
    # Create lags up to p + d_max
    for i in range(1, p + d_max + 1):
        df_temp[f'y_L{i}'] = df_temp['y'].shift(i)
        df_temp[f'x_pos_L{i}'] = df_temp['x_pos'].shift(i)
        df_temp[f'x_neg_L{i}'] = df_temp['x_neg'].shift(i)
        y_lags.append(f'y_L{i}')
        pos_lags.append(f'x_pos_L{i}')
        neg_lags.append(f'x_neg_L{i}')
        
    df_var = df_temp.dropna().copy()
    
    # Formula for y
    formula_y = 'y ~ ' + ' + '.join(y_lags + pos_lags + neg_lags)
    model_y = smf.ols(formula=formula_y, data=df_var).fit()
    
    # Wald test for REER_pos -> y (first p lags of x_pos are 0)
    hyp_pos = ', '.join([f'x_pos_L{i} = 0' for i in range(1, p + 1)])
    wald_pos = model_y.wald_test(hyp_pos, scalar=True)
    
    # Wald test for REER_neg -> y (first p lags of x_neg are 0)
    hyp_neg = ', '.join([f'x_neg_L{i} = 0' for i in range(1, p + 1)])
    wald_neg = model_y.wald_test(hyp_neg, scalar=True)
    
    print(f"Lag p = {p}:")
    print(f"  REER+ -> DVA (Appreciation): statistic={wald_pos.statistic:.4f}, p-value={wald_pos.pvalue:.4f}")
    print(f"  REER- -> DVA (Depreciation): statistic={wald_neg.statistic:.4f}, p-value={wald_neg.pvalue:.4f}")

for p in [1, 2, 3, 4]:
    test_asymmetric_ty(p)
