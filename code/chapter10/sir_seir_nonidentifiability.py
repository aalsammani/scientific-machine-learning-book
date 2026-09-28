"""Structural non-identifiability of SIR and SEIR models from incidence data (Chapter 10, section 10.3,
Theorem 10.1; Chapter 15, section 15.1, Theorem 15.1; Figures 10.1 and 15.1). The observation is
the incidence y(t) = beta(t) S(t) I(t) / N with N = 1, and
beta(t) = 0.5 (1 - 0.4 / (1 + exp(-(t - 40)/5))).

For each model:
  1. solve the true system (solve_ivp, DOP853, rtol 1e-12, atol 1e-14) on [0, 120]
     (24001 output points);
  2. build an alternative parameter set by the explicit construction in the proof (integrating
     factors evaluated by cumulative Simpson quadrature) and check the algebraic identity;
  3. independent check: interpolate the constructed beta'(t) (cubic spline), solve the model
     forward from the alternative initial state with the alternative rates, and compare the
     resulting incidence with y(t); this does not use the constructed compartments;
  4. check admissibility: positivity and R'(0) = N - S0 - E0' - I0' >= 0;
  5. observations at a single time: for SIR, members of the remaining family are computed on a
     grid of gamma values and recorded with an admissibility flag; for SEIR, admissible members
     matching E(t1) and I(t1) are searched on a 9 x 9 grid of (sigma, gamma);
     observations at two distinct times (t1 = 30, t2 = 60) give strictly monotone scalar equations
     for the rates, which are solved numerically (brentq).
SIR: S0 = 0.99, I0 = 0.01, gamma = 1/7; alternative gamma' = 1/10, I0' = 0.005.
SEIR: S0 = 0.98, E0 = I0 = 0.01, sigma = 1/5, gamma = 1/7; alternative sigma' = 1/8,
gamma' = 1/10, E0' = 0.006, I0' = 0.008.

Usage:
    python code/chapter10/sir_seir_nonidentifiability.py
Outputs: results/chapter10/sir_nonidentifiability_verification.json, sir_trajectories.npz;
results/chapter15/seir_nonidentifiability_verification.json (SIR and SEIR), seir_trajectories.npz.
Runtime: a few seconds. Deterministic.
"""
import json, os
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp, cumulative_simpson
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

RESULTS = Path(__file__).resolve().parents[2] / "results"
os.makedirs(RESULTS / "chapter10", exist_ok=True)
os.makedirs(RESULTS / "chapter15", exist_ok=True)

T, NT = 120.0, 24001
t = np.linspace(0, T, NT)
def beta(t):  # transmission rate with a smooth intervention around t = 40
    return 0.5 * (1 - 0.4 / (1 + np.exp(-(t - 40) / 5)))
opt = dict(method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t)

def cumint(f):
    return cumulative_simpson(f, x=t, initial=0.0)

def lin_state(drive, rate, x0):
    """x' = drive - rate x, x(0) = x0, via the integrating factor."""
    return np.exp(-rate * t) * (x0 + cumint(np.exp(rate * t) * drive))

out = {"beta": "0.5*(1 - 0.4/(1+exp(-(t-40)/5)))", "T": T}

# ---------------- SIR ----------------
S0, I0, g = 0.99, 0.01, 1/7
sol = solve_ivp(lambda s, x: [-beta(s)*x[0]*x[1], beta(s)*x[0]*x[1] - g*x[1]], (0, T), [S0, I0], **opt)
S, I = sol.y; y = beta(t) * S * I
g2, I02 = 1/10, 0.005
S_c = S0 - cumint(y)
I2 = lin_state(y, g2, I02)
b2 = y / (S_c * I2)
alg = float(np.max(np.abs(b2 * S_c * I2 - y)))
b2s = CubicSpline(t, b2)
fw = solve_ivp(lambda s, x: [-b2s(s)*x[0]*x[1], b2s(s)*x[0]*x[1] - g2*x[1]], (0, T), [S0, I02], **opt)
y_fw = b2s(t) * fw.y[0] * fw.y[1]
sir = dict(true=dict(S0=S0, I0=I0, gamma=g), alt=dict(gamma=g2, I0=I02),
           algebraic_identity_max=alg,
           forward_check_max_abs_incidence_diff=float(np.max(np.abs(y_fw - y))),
           forward_check_rel=float(np.max(np.abs(y_fw - y)) / np.max(y)),
           R0_alt=1 - S0 - I02, min_I_alt=float(I2.min()), min_beta_alt=float(b2.min()),
           max_rel_diff_I=float(np.max(np.abs(I2 - I)) / np.max(I)),
           max_S_diff=float(np.max(np.abs(S_c - S))))
# one observation I(t1): a one-parameter family remains
t1, t2 = 30.0, 60.0; k1, k2 = np.searchsorted(t, t1), np.searchsorted(t, t2)
fam = []
for gg in np.linspace(0.13, 0.16, 7):
    I0g = (I[k1] - np.exp(-gg*t[k1]) * cumint(np.exp(gg*t)*y)[k1]) * np.exp(gg*t[k1])
    Ig = lin_state(y, gg, I0g)
    fam.append(dict(gamma=float(gg), I0=float(I0g), admissible=bool(0 < I0g <= 1 - S0 and Ig.min() > 0),
                    I_at_t1=float(Ig[k1]), I_at_t2=float(Ig[k2])))
sir["one_observation_family"] = dict(t1=t1, true_I_t1=float(I[k1]), members=fam)
# two observations I(t1), I(t2): h(gamma) = I(t2) predicted from I(t1) is strictly decreasing
def h(gg):
    prop = np.exp(-gg*(t[k2]-t[k1])) * I[k1] + np.trapezoid(np.exp(-gg*(t[k2]-t[k1:k2+1]))*y[k1:k2+1], t[k1:k2+1])
    return prop - I[k2]
grid = np.linspace(0.02, 1.0, 200); hv = np.array([h(gg) for gg in grid])
g_rec = brentq(h, 0.02, 1.0)
sir["two_observations"] = dict(t1=t1, t2=t2, h_strictly_decreasing_on_grid=bool(np.all(np.diff(hv) < 0)),
                               gamma_recovered=float(g_rec), gamma_true=g)
out["SIR"] = sir
np.savez(RESULTS / "chapter10" / "sir_trajectories.npz", t=t, y=y, I_true=I, I_alt=I2,
         beta_true=beta(t), beta_alt=b2, S=S)
print("SIR", {k: v for k, v in sir.items() if k not in ("one_observation_family",)})
print("SIR one-observation family:", sir["one_observation_family"])

# ---------------- SEIR ----------------
S0, E0, I0, sg, g = 0.98, 0.01, 0.01, 1/5, 1/7
sol = solve_ivp(lambda s, x: [-beta(s)*x[0]*x[2], beta(s)*x[0]*x[2] - sg*x[1], sg*x[1] - g*x[2]],
                (0, T), [S0, E0, I0], **opt)
S, E, I = sol.y; y = beta(t) * S * I
sg2, g2, E02, I02 = 1/8, 1/10, 0.006, 0.008
S_c = S0 - cumint(y)
E2 = lin_state(y, sg2, E02)
I2 = lin_state(sg2 * E2, g2, I02)
b2 = y / (S_c * I2)
alg = float(np.max(np.abs(b2 * S_c * I2 - y)))
b2s = CubicSpline(t, b2)
fw = solve_ivp(lambda s, x: [-b2s(s)*x[0]*x[2], b2s(s)*x[0]*x[2] - sg2*x[1], sg2*x[1] - g2*x[2]],
               (0, T), [S0, E02, I02], **opt)
y_fw = b2s(t) * fw.y[0] * fw.y[2]
seir = dict(true=dict(S0=S0, E0=E0, I0=I0, sigma=sg, gamma=g), alt=dict(sigma=sg2, gamma=g2, E0=E02, I0=I02),
            algebraic_identity_max=alg,
            forward_check_max_abs_incidence_diff=float(np.max(np.abs(y_fw - y))),
            forward_check_rel=float(np.max(np.abs(y_fw - y)) / np.max(y)),
            R0_alt=1 - S0 - E02 - I02, min_E_alt=float(E2.min()), min_I_alt=float(I2.min()),
            min_beta_alt=float(b2.min()), max_rel_diff_I=float(np.max(np.abs(I2 - I)) / np.max(I)),
            max_rel_diff_E=float(np.max(np.abs(E2 - E)) / np.max(E)))
# E(t1) and I(t1) observed once: exhibit two distinct members matching both values
mem = []
for sgg in np.linspace(0.19, 0.21, 9):
    for gg in np.linspace(0.135, 0.15, 9):
        E0g = (E[k1] - np.exp(-sgg*t[k1]) * cumint(np.exp(sgg*t)*y)[k1]) * np.exp(sgg*t[k1])
        Eg = lin_state(y, sgg, E0g)
        I0g = (I[k1] - np.exp(-gg*t[k1]) * cumint(np.exp(gg*t)*sgg*Eg)[k1]) * np.exp(gg*t[k1])
        Ig = lin_state(sgg*Eg, gg, I0g)
        ok = E0g > 0 and I0g > 0 and E0g + I0g <= 1 - S0 and Eg.min() > 0 and Ig.min() > 0
        if ok and abs(sgg - sg) + abs(gg - g) > 1e-3:
            mem.append(dict(sigma=float(sgg), gamma=float(gg), E0=float(E0g), I0=float(I0g), E_t1=float(Eg[k1]),
                            I_t1=float(Ig[k1]), I_t2=float(Ig[k2]), R0=float(1 - S0 - E0g - I0g)))
seir_admissible_count = len(mem)
mem = mem[:1] + mem[-1:] if mem else mem
seir["single_time_E_and_I_family"] = dict(t1=t1, true_E_t1=float(E[k1]), true_I_t1=float(I[k1]),
                                          true_I_t2=float(I[k2]), members=mem,
                                          admissible_members_found_on_9x9_grid=seir_admissible_count)
# E and I at two times: sigma from E, then gamma from I (each a strictly monotone scalar equation)
def hE(ss):
    return np.exp(-ss*(t[k2]-t[k1]))*E[k1] + np.trapezoid(np.exp(-ss*(t[k2]-t[k1:k2+1]))*y[k1:k2+1], t[k1:k2+1]) - E[k2]
s_rec = brentq(hE, 0.02, 2.0)
def hI(gg):
    drv = s_rec * E
    return np.exp(-gg*(t[k2]-t[k1]))*I[k1] + np.trapezoid(np.exp(-gg*(t[k2]-t[k1:k2+1]))*drv[k1:k2+1], t[k1:k2+1]) - I[k2]
g_rec = brentq(hI, 0.02, 2.0)
seir["two_times_each"] = dict(sigma_recovered=float(s_rec), gamma_recovered=float(g_rec), sigma_true=sg, gamma_true=g)
out["SEIR"] = seir
print("SEIR", {k: v for k, v in seir.items() if k != "single_time_E_and_I_family"})
print("SEIR single-time family:", seir["single_time_E_and_I_family"])

np.savez(RESULTS / "chapter15" / "seir_trajectories.npz", t=t, y=y, I_true=I, I_alt=I2, E_true=E, E_alt=E2)
json.dump(out, open(RESULTS / "chapter15" / "seir_nonidentifiability_verification.json", "w"), indent=1)
json.dump(out["SIR"], open(RESULTS / "chapter10" / "sir_nonidentifiability_verification.json", "w"), indent=1)
