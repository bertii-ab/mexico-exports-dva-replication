# -*- coding: utf-8 -*-
"""
Generate Toda-Yamamoto Granger Causality LaTeX tables for both full and post-break samples
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

# Setup variables
df_merged['ln_TiVA'] = np.log(df_merged['TiVA_DVA_ANN'])
df_merged['ln_REER'] = np.log(df_merged['REER (Real - Broad)'])

# Subsamples
samples = {
    'Full Sample (1995-Q1 to 2022-Q4)': df_merged.copy(),
    'Post-Break Sample (2011-Q1 to 2022-Q4)': df_merged[df_merged['Time'] > '2010-Q3'].copy()
}

for name, df_sample in samples.items():
    print(f"\n==========================================")
    print(f"Results for: {name} (N = {len(df_sample)})")
    print(f"==========================================")
    
    df_clean = df_sample[['Time', 'ln_TiVA', 'ln_REER']].dropna().copy()
    df_clean['y'] = df_clean['ln_TiVA']
    df_clean['x'] = df_clean['ln_REER']
    
    print(f"{'Lag p':<8} | {'REER -> DVA':<25} | {'DVA -> REER':<25}")
    print(f"{'-'*8}-+-{'-'*25}-+-{'-'*25}")
    
    # We will compute results for p from 1 to 4
    for p in range(1, 5):
        d_max = 1
        df_temp = df_clean.copy()
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
        
        # Wald test
        hyp_x_to_y = ', '.join([f'x_L{i} = 0' for i in range(1, p + 1)])
        wald_x_to_y = model_y.wald_test(hyp_x_to_y, scalar=True)
        
        hyp_y_to_x = ', '.join([f'y_L{i} = 0' for i in range(1, p + 1)])
        wald_y_to_x = model_x.wald_test(hyp_y_to_x, scalar=True)
        
        print(f"p = {p:<5} | Stat={wald_x_to_y.statistic:6.4f} (p={wald_x_to_y.pvalue:6.4f}) | Stat={wald_y_to_x.statistic:6.4f} (p={wald_y_to_x.pvalue:6.4f})")
