"""
Stage 2: ANN-MIDAS Temporal Disaggregation
Disaggregates annual OECD TiVA series to quarterly frequency using
Artificial Neural Network Mixed-Data Sampling (ANN-MIDAS).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_annual = pd.read_csv('./master_annual.csv')
df_monthly = pd.read_csv('./master_monthly.csv')
df_quarterly = pd.read_csv('./master_quarterly.csv')

print("DataFrames reloaded:")
print("df_annual head:")
print(df_annual.head())
print("df_monthly head:")
print(df_monthly.head())
print("df_quarterly head:")
print(df_quarterly.head())



import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import warnings
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.interp.denton import dentonm

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")

# ==========================================
# 1. The ANN-MIDAS-AR Architecture
# ==========================================
class MultiFeature_ANN_MIDAS_AR(nn.Module):
    def __init__(self, monthly_lags, quarterly_lags, num_dummies, hidden_nodes=5):
        super(MultiFeature_ANN_MIDAS_AR, self).__init__()

        self.num_m = len(monthly_lags)
        self.num_q = len(quarterly_lags)
        self.m_lags = monthly_lags
        self.q_lags = quarterly_lags
        self.num_dummies = num_dummies
        self.num_ar = 1

        self.delta_m = nn.Parameter(torch.zeros(self.num_m, 2))
        self.delta_q = nn.Parameter(torch.zeros(self.num_q, 2))

        self.hidden = nn.Linear(self.num_m + self.num_q + self.num_dummies + self.num_ar, hidden_nodes)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_nodes, 1)

    def exponential_almon_weights(self, delta, lags):
        l = torch.arange(lags, dtype=torch.float32)
        weight_unscaled = torch.exp(delta[0] * l + delta[1] * (l ** 2))
        return weight_unscaled / torch.sum(weight_unscaled)

    def forward(self, x_quarterly, x_monthly, x_dummies, x_ar):
        batch_size = x_quarterly.shape[0]
        agg_m = torch.zeros(batch_size, self.num_m)
        agg_q = torch.zeros(batch_size, self.num_q)

        for f in range(self.num_m):
            lag = self.m_lags[f]
            w = self.exponential_almon_weights(self.delta_m[f], lag)
            agg_m[:, f] = torch.sum(x_monthly[:, f, -lag:] * w, dim=1)

        for f in range(self.num_q):
            lag = self.q_lags[f]
            w = self.exponential_almon_weights(self.delta_q[f], lag)
            agg_q[:, f] = torch.sum(x_quarterly[:, f, -lag:] * w, dim=1)

        self.x_aligned = torch.cat((agg_q, agg_m, x_dummies, x_ar), dim=1)
        g = self.activation(self.hidden(self.x_aligned))
        y_pred = self.output(g)
        return y_pred

# ==========================================
# 2. Dynamic Shock Detection & Tensor Prep
# ==========================================
def detect_dynamic_shocks(df, columns, window_size, threshold=2.25):
    df_shocks = pd.DataFrame(index=df.index)
    for col in columns:
        rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
        rolling_std = rolling_std.fillna(method='bfill').replace(0, 0.001)
        z_scores = np.abs((df[col] - rolling_mean) / rolling_std)
        df_shocks[f'{col}_shock'] = (z_scores > threshold).astype(float)
    return df_shocks

def prepare_tensors_with_ar_and_shocks(df_annual, df_quarterly, df_monthly, max_m_lag=12, max_q_lag=4):
    print("Formatting Data, identifying anomaly shocks, and structuring AR(1)...")
    df_m = df_monthly.copy().sort_values('Month').reset_index(drop=True)
    df_q = df_quarterly.copy()
    m_cols = ['Inter_goods_NO', 'TotalExports', 'US_IndustrialProduction', 'GSCPI']
    q_cols = ['ValueAdded']

    for col in m_cols:
        df_m[col] = df_m[col].interpolate(method='linear').bfill()

    df_m['Year'] = pd.to_datetime(df_m['Month']).dt.year
    df_q['Year'] = df_q['Quarter'].astype(str).str[:4].astype(int)

    m_shocks = detect_dynamic_shocks(df_m, m_cols, window_size=18, threshold=2.25)
    df_m = pd.concat([df_m, m_shocks], axis=1)
    q_shocks = detect_dynamic_shocks(df_q, q_cols, window_size=6, threshold=2.25)
    df_q = pd.concat([df_q, q_shocks], axis=1)

    df_m[m_cols] = (df_m[m_cols] - df_m[m_cols].mean()) / df_m[m_cols].std()
    df_q[q_cols] = (df_q[q_cols] - df_q[q_cols].mean()) / df_q[q_cols].std()

    df_a = df_annual.sort_values('Year').reset_index(drop=True)
    df_a['TiVA_DVA_Lag1'] = df_a['TiVA_DVA'].shift(1).bfill()
    df_a['TiVA_DVA_Lag1_Scaled'] = (df_a['TiVA_DVA_Lag1'] - df_a['TiVA_DVA_Lag1'].mean()) / df_a['TiVA_DVA_Lag1'].std()

    years = sorted(df_a['Year'].unique())
    X_q_l, X_m_l, Y_l, X_d_l, X_ar_l = [], [], [], [], []
    valid_y = []

    for y in years:
        q_data = df_q[df_q['Year'] == y]['ValueAdded'].values
        m_data = df_m[df_m['Year'] == y][m_cols].values
        if len(q_data) == 4 and m_data.shape == (12, len(m_cols)):
            X_q_l.append(q_data.reshape(1, max_q_lag))
            X_m_l.append(m_data.T)
            m_s = df_m[df_m['Year'] == y][[f'{c}_shock' for c in m_cols]].sum().values
            q_s = df_q[df_q['Year'] == y][[f'{c}_shock' for c in q_cols]].sum().values
            X_d_l.append(np.concatenate([q_s, m_s]))
            X_ar_l.append([df_a[df_a['Year'] == y]['TiVA_DVA_Lag1_Scaled'].values[0]])
            Y_l.append([df_a[df_a['Year'] == y]['TiVA_DVA'].values[0]])
            valid_y.append(y)

    return torch.tensor(np.array(X_q_l), dtype=torch.float32), torch.tensor(np.array(X_m_l), dtype=torch.float32), torch.tensor(np.array(X_d_l), dtype=torch.float32), torch.tensor(np.array(X_ar_l), dtype=torch.float32), torch.tensor(np.array(Y_l), dtype=torch.float32), valid_y

# ==========================================
# 3. Classical Linear Baseline (FIXED)
# ==========================================
def compute_linear_baseline(df_annual, df_quarterly, df_monthly):
    print("Computing classical OLS structural baseline...")
    df_m = df_monthly.copy()
    df_q = df_quarterly.copy()

    # FIXED: Ensure 'Year' column is generated in the copies
    if 'Year' not in df_m.columns: df_m['Year'] = pd.to_datetime(df_m['Month']).dt.year
    if 'Year' not in df_q.columns: df_q['Year'] = df_q['Quarter'].astype(str).str[:4].astype(int)

    years = sorted(df_annual['Year'].unique())
    X_a, y_a = [], []

    for y in years:
        q_val = df_q[df_q['Year'] == y]['ValueAdded'].sum()
        m_exp = df_m[df_m['Year'] == y]['TotalExports'].sum()
        target = df_annual[df_annual['Year'] == y]['TiVA_DVA'].values[0]
        X_a.append([q_val, m_exp])
        y_a.append(target)

    ols = LinearRegression().fit(X_a, y_a)
    proxy = []
    for y in years:
        q_v = df_q[df_q['Year'] == y]['ValueAdded'].values
        m_v = df_m[df_m['Year'] == y]['TotalExports'].values
        if len(m_v) == 12 and len(q_v) == 4:
            q_exp = [np.sum(m_v[0:3]), np.sum(m_v[3:6]), np.sum(m_v[6:9]), np.sum(m_v[9:12])]
            for i in range(4):
                proxy.append(ols.intercept_ + (ols.coef_[0] * q_v[i]) + (ols.coef_[1] * q_exp[i]))

    proxy = np.array(proxy)
    if np.min(proxy) <= 0: proxy = proxy + abs(np.min(proxy)) + 1.0
    return dentonm(proxy, y_a, freq="aq")

# ==========================================
# 4. Grid Search & Master Pipeline
# ==========================================
def run_structurally_constrained_grid_search(X_q, X_m, X_dummies, X_ar, Y_true, param_grid, linear_baseline, raw_q, tolerance_pct=0.15, train_ratio=0.8):
    split_idx = int(len(Y_true) * train_ratio)
    X_q_t, X_q_v = X_q[:split_idx], X_q[split_idx:]
    X_m_t, X_m_v = X_m[:split_idx], X_m[split_idx:]
    X_d_t, X_d_v = X_dummies[:split_idx], X_dummies[split_idx:]
    X_ar_t, X_ar_v = X_ar[:split_idx], X_ar[split_idx:]
    Y_t, Y_v = Y_true[:split_idx], Y_true[split_idx:]

    best_loss, best_params, best_state = float('inf'), None, None
    grid_results = []
    keys, values = zip(*param_grid.items())
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        model = MultiFeature_ANN_MIDAS_AR(params['m_lags'], params['q_lags'], X_dummies.shape[1], params['hidden_nodes'])
        optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(params['epochs']):
            model.train(); optimizer.zero_grad()
            loss = criterion(model(X_q_t, X_m_t, X_d_t, X_ar_t), Y_t)
            loss.backward(); optimizer.step()

        for epoch in range(params['epochs']):
            model.train()
            optimizer.zero_grad()
            pred_train = model(X_q_t, X_m_t, X_d_t, X_ar_t)

            # THE FIX: Apply a "Burn-in". Ignore the first year (index 0)
            # where the AR(1) and lags are synthetic backfilled data.
            if len(pred_train) > 1:
                loss = criterion(pred_train[1:], Y_t[1:])
            else:
                loss = criterion(pred_train, Y_t)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_q_v, X_m_v, X_d_v, X_ar_v), Y_v).item()
            _ = model(X_q, X_m, X_dummies, X_ar)
            ind = model.hidden(model.x_aligned).sum(dim=1).numpy()
            if np.min(ind) <= 0: ind += abs(np.min(ind)) + 1.0
            ann_q = dentonm(raw_q * np.repeat(ind / np.mean(ind), 4), Y_true.numpy().flatten(), freq="aq")
            dev = np.max(np.abs((ann_q - linear_baseline) / (linear_baseline + 1e-6)))
            score = v_loss + (dev - tolerance_pct) * 1e6 if dev > tolerance_pct else v_loss

            if score < best_loss: best_loss, best_params, best_state = score, params, model.state_dict()
            # Inside the evaluation block, right before Denton-Cholette...
        raw_quarterly_pos = raw_q+ abs(np.min(raw_q)) + 0.1
        expanded_indicator = raw_quarterly_pos * np.repeat(ind/ np.mean(ind), 4)

        # THE FIX: Boundary Clamping
        # Force the first 4 quarters and last 8 quarters of the ANN proxy
        # to tightly hug the raw quarterly distribution, stripping out extreme network variance.
        expanded_indicator[:4] = raw_quarterly_pos[:4]  # Clamp Start
        expanded_indicator[-8:] = raw_quarterly_pos[-8:] # Clamp End (COVID Recovery)

        # Now apply Denton
        ann_quarterly_dva = dentonm(expanded_indicator, Y_true.numpy().flatten(), freq="aq")
        grid_results.append({'score': score, 'params': params})
        
    print(f"\n--- ANN-MIDAS Grid Search Diagnostics ---")
    scores = [r['score'] for r in grid_results]
    print(f"Total Models Evaluated: {len(scores)}")
    print(f"Score Distribution: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}, Min={np.min(scores):.4f}, Max={np.max(scores):.4f}")
    print(f"Best Score: {best_loss:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"-----------------------------------------\n")
    
    m = MultiFeature_ANN_MIDAS_AR(best_params['m_lags'], best_params['q_lags'], X_dummies.shape[1], best_params['hidden_nodes'])
    m.load_state_dict(best_state); return m

def execute_fully_constrained_disaggregation(df_annual, df_quarterly, df_monthly):
    torch.manual_seed(42); np.random.seed(42)
    linear_baseline = compute_linear_baseline(df_annual, df_quarterly, df_monthly)
    X_q, X_m, X_d, X_ar, Y_t, valid_y = prepare_tensors_with_ar_and_shocks(df_annual, df_quarterly, df_monthly)
    raw_q = X_q.numpy().flatten() + abs(np.min(X_q.numpy())) + 0.1

    param_grid = {'hidden_nodes': [3, 5, 8], 'lr': [0.01, 0.005], 'epochs': [500,800], 'q_lags': [[4]], 'm_lags': [[12,12,12,12], [6,6,3,12]]}
    best_model = run_structurally_constrained_grid_search(X_q, X_m, X_d, X_ar, Y_t, param_grid, linear_baseline, raw_q)

    best_model.eval()
    with torch.no_grad():
        _ = best_model(X_q, X_m, X_d, X_ar)
        ind = best_model.hidden(best_model.x_aligned).sum(dim=1).numpy()
        if np.min(ind) <= 0: ind += abs(np.min(ind)) + 1.0
        final_ann = dentonm(raw_q * np.repeat(ind / np.mean(ind), 4), Y_t.numpy().flatten(), freq="aq")

    final_df = pd.DataFrame({'Year': np.repeat(valid_y, 4), 'Quarter': np.tile(['Q1','Q2','Q3','Q4'], len(valid_y)), 'TiVA_DVA_ANN': final_ann, 'TiVA_DVA_Linear': linear_baseline})
    final_df['Time'] = final_df['Year'].astype(str) + "-" + final_df['Quarter']
    print("Disaggregation sequence successfully executed.")
    return final_df

# ==========================================
# Run the Code
# ==========================================
final_data = execute_fully_constrained_disaggregation(df_annual, df_quarterly, df_monthly)
print(final_data.head())
# final_data.to_csv("Interpolated_TiVA_DVA_Final.csv", index=False)

# ==========================================
# Plot 1: ANN vs Linear Baseline
# ==========================================
plt.figure(figsize=(16, 8))
plt.plot(final_data['Time'], final_data['TiVA_DVA_ANN'], label='TiVA_DVA_ANN (Neural Network)', marker='.', linestyle='-')
plt.plot(final_data['Time'], final_data['TiVA_DVA_Linear'], label='TiVA_DVA_Linear (Classical Baseline)', marker='x', linestyle='--')

plt.title('Comparison of Neural Network Disaggregation and Classical Linear Baseline')
plt.xlabel('Time (Year-Quarter)')
plt.ylabel('TiVA DVA Value')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot_ann_vs_linear.png', dpi=150)
plt.close()
print("Saved plot_ann_vs_linear.png")

# Export plot data to CSV
plot1_df = final_data[['Time', 'TiVA_DVA_ANN', 'TiVA_DVA_Linear']].copy()
plot1_df.to_csv('plot_ann_vs_linear.csv', index=False)
print("Saved plot_ann_vs_linear.csv")

# ==========================================
# Plot 2: ANN vs Annual Benchmark
# ==========================================
# 1. Prepare Annual Data for plotting (divided by 4)
df_annual_plot = df_annual.copy()
df_annual_plot['TiVA_DVA_Yearly_Avg'] = df_annual_plot['TiVA_DVA'] / 4

# To make the x-axis coincide, we create a 'Time' string for the annual data
# We place the annual marker at 'Q2.5' or similar to center it,
# but to align with the categorical 'Time' axis of final_data, we'll map to the mid-year quarter.
df_annual_plot['Time_Mid'] = df_annual_plot['Year'].astype(str) + "-Q2"

# 2. Plotting
plt.figure(figsize=(16, 8))

# Plot the Quarterly ANN results
plt.plot(final_data['Time'], final_data['TiVA_DVA_ANN'],
         label='TiVA_DVA_ANN (Quarterly)', marker='.', linestyle='-', alpha=0.7, color='tab:blue')

# Plot the Annual benchmark (divided by 4)
# We filter df_annual to only include years present in final_data
valid_years = final_data['Year'].unique()
df_annual_filtered = df_annual_plot[df_annual_plot['Year'].isin(valid_years)]

plt.step(df_annual_filtered['Time_Mid'], df_annual_filtered['TiVA_DVA_Yearly_Avg'],
         where='mid', label='TiVA_DVA (Annual / 4)', color='tab:red', linestyle='--', linewidth=2, marker='s')

plt.title('Comparison: Quarterly ANN Disaggregation vs. Annual Benchmark (Normalized)')
plt.xlabel('Time (Year-Quarter)')
plt.ylabel('Value')
plt.xticks(final_data['Time'][::4], rotation=45) # Show labels every year for clarity
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('plot_ann_vs_annual.png', dpi=150)
plt.close()
print("Saved plot_ann_vs_annual.png")

# Export plot data to CSV
plot2_ann = final_data[['Time', 'TiVA_DVA_ANN']].copy()
plot2_annual = df_annual_filtered[['Time_Mid', 'TiVA_DVA_Yearly_Avg']].copy()
plot2_annual.columns = ['Time', 'TiVA_DVA_Yearly_Avg']
plot2_merged = pd.merge(plot2_ann, plot2_annual, on='Time', how='outer').sort_values('Time')
plot2_merged.to_csv('plot_ann_vs_annual.csv', index=False)
print("Saved plot_ann_vs_annual.csv")

# ==========================================
# Plot 3: Validation - Yearly Sum
# ==========================================
# 1. Calculate the yearly sum of the quarterly ANN results
ann_yearly_sum = final_data.groupby('Year')['TiVA_DVA_ANN'].sum().reset_index()

# 2. Filter original annual data to match the years present in the ANN results
valid_years = ann_yearly_sum['Year'].unique()
df_annual_subset = df_annual[df_annual['Year'].isin(valid_years)].sort_values('Year')

# 3. Plotting
plt.figure(figsize=(12, 6))

plt.plot(ann_yearly_sum['Year'], ann_yearly_sum['TiVA_DVA_ANN'],
         label='Sum of TiVA_DVA_ANN (Quarterly)', marker='o', linestyle='-', linewidth=2)

plt.plot(df_annual_subset['Year'], df_annual_subset['TiVA_DVA'],
         label='Original TiVA_DVA (Yearly)', marker='x', linestyle='--', color='red', alpha=0.6)

plt.title('Validation: Yearly Sum of ANN Quarters vs. Original Annual Benchmark')
plt.xlabel('Year')
plt.ylabel('Value')
plt.xticks(ann_yearly_sum['Year'], rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_validation_yearly.png', dpi=150)
plt.close()
print("Saved plot_validation_yearly.png")

# Export plot data to CSV
plot3_df = pd.merge(ann_yearly_sum, df_annual_subset[['Year', 'TiVA_DVA']], on='Year', how='outer', suffixes=('_ANN_Sum', '_Original'))
plot3_df.to_csv('plot_validation_yearly.csv', index=False)
print("Saved plot_validation_yearly.csv")

final_data.to_csv("Interpolated_TiVA_DVA_Final.csv", index=False)
print("Saved Interpolated_TiVA_DVA_Final.csv")