# -*- coding: utf-8 -*-
"""
Compute Asymmetric Cumulative Dynamic Multipliers (IRFs)
following Shin, Yu & Greenwood-Nimmo (2014).

The estimated NARDL-ECM is:
  Δy_t = α₀ + λ·y_{t-1} + θ⁺·x⁺_{t-1} + θ⁻·x⁻_{t-1}
       + β₁·Δy_{t-1} + β₂·Δy_{t-2} + ε_t

Where y = ln(DVA_normalized), x⁺/x⁻ = REER partial sums.
Best REER short-run lags = None (no contemporaneous or lagged Δx terms).

The cumulative dynamic multiplier m_h^{+} traces the cumulative response
of y to a unit positive shock (+1) in x⁺ over h periods.
The cumulative dynamic multiplier m_h^{-} traces the cumulative response
of y to a unit negative shock (-1) in x⁻ over h periods.
"""

import numpy as np
import pandas as pd

# === Estimated NARDL-ECM Coefficients (from nardl_coefficients.txt) ===
alpha0 = -0.6920129509478707     # Intercept
lam    = -1.1198934763973765     # λ  (ECM speed of adjustment)
phi_pos =  1.395223186294142     # coefficient on REER_pos (levels)
phi_neg =  0.48981859633056585   # coefficient on REER_neg (levels)
beta1  =  0.27840092170205905    # AR(1) coefficient
beta2  =  0.33051300100841820    # AR(2) coefficient

# Long-run multipliers (for reference / asymptotic targets)
theta_pos = -phi_pos / lam  # ≈ 1.2459 (appreciation is a positive shock, target is +theta^+)
theta_neg = -phi_neg / lam  # ≈ 0.4374 (depreciation is a negative shock, target is -theta^-)

print(f"Long-run appreciation multiplier (θ⁺): {theta_pos:.4f}")
print(f"Long-run depreciation multiplier (θ⁻): {theta_neg:.4f}")
print(f"Depreciation target for negative shock (-θ⁻): {-theta_neg:.4f}")

# === Simulation Horizon ===
H = 20  # quarters

# === Compute Dynamic Multipliers via Simulation ===
def simulate_response(phi, H, shock_direction=1.0):
    """Simulate the response of y to a permanent unit step in x with direction."""
    y = np.zeros(H + 3)   # y[-2], y[-1], y[0], ..., y[H]  (index shifted by 2)
    dy = np.zeros(H + 3)
    x = np.zeros(H + 3)

    # Pre-shock steady state: y = x = 0 for t < 0
    # Shock: x jumps to shock_direction at t=0 and stays
    for t in range(2, H + 3):  # t=2 corresponds to period 0 in our indexing
        x[t] = shock_direction  # permanent unit step

        # NARDL-ECM equation:
        # Δy_t = α₀ + λ·y_{t-1} + φ·x_{t-1} + β₁·Δy_{t-1} + β₂·Δy_{t-2}
        # Note: we set α₀ = 0 for the multiplier simulation (deviations from SS)
        dy[t] = lam * y[t-1] + phi * x[t-1] + beta1 * dy[t-1] + beta2 * dy[t-2]
        y[t] = y[t-1] + dy[t]

    # Extract from period 0 onwards
    return y[2:H+3]

# Simulate for positive (appreciation, +1) and negative (depreciation, -1) shocks
m_pos = simulate_response(phi_pos, H, shock_direction=1.0)
m_neg = simulate_response(phi_neg, H, shock_direction=-1.0)

# === Create Output DataFrame ===
horizons = list(range(H + 1))
irf_df = pd.DataFrame({
    'Horizon': horizons,
    'IRF_Appreciation': m_pos,
    'IRF_Depreciation': m_neg,
    'LR_Appreciation': theta_pos,
    'LR_Depreciation': -theta_neg
})

print("\n=== Asymmetric Cumulative Dynamic Multipliers ===")
print(irf_df.to_string(index=False))

# Save to CSV for pgfplots
irf_df.to_csv('plot_irf_dynamic_multipliers.csv', index=False)
print("\nSaved plot_irf_dynamic_multipliers.csv")

