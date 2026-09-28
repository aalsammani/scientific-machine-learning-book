"""Figure 13.1: Ornstein-Uhlenbeck drift and diffusion estimates against the sampling interval (left;
Chapter 13, section 13.4) and the Kalman filter for the damped oscillator (right; section 13.5),
from results/chapter13/ou_drift_diffusion.json and results/chapter13/kalman_filter_study.json.

Usage:
    python code/figures/make_fig13_1.py
Output: figures/chapter13/F13.1_sde_and_kalman.png and .pdf (reads stored results only; runtime a few seconds).
"""
import json, os, shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results" / "chapter13"
FIG_DIR = REPO_ROOT / "figures" / "chapter13"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "legend.frameon": False, "savefig.dpi": 200})
BLUE, VERM, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#555555"
R = str(RESULTS)
ou = json.load(open(f"{R}/ou_drift_diffusion.json"))
kf = json.load(open(f"{R}/kalman_filter_study.json"))

fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
rows = ou["results"]; dt = np.array([r["dt"] for r in rows])
th, ths = np.array([r["theta_mean"] for r in rows]) / ou["theta_true"], np.array([r["theta_std"] for r in rows]) / ou["theta_true"]
sg, sgs = np.array([r["sigma_mean"] for r in rows]) / ou["sigma_true"], np.array([r["sigma_std"] for r in rows]) / ou["sigma_true"]
fine = np.logspace(-3, np.log10(4), 200)
th_lim = (1 - np.exp(-ou["theta_true"] * fine)) / fine / ou["theta_true"]
sg_lim = np.sqrt((1 - np.exp(-ou["theta_true"] * fine)) / (ou["theta_true"] * fine))
a.plot(fine, th_lim, color=BLUE, lw=1.0, ls="--")
a.plot(fine, sg_lim, color=VERM, lw=1.0, ls="--")
a.errorbar(dt, th, yerr=ths, color=BLUE, marker="o", ls="none", capsize=3, label=r"drift $\hat\theta/\theta$")
a.errorbar(dt, sg, yerr=sgs, color=VERM, marker="s", ls="none", capsize=3, label=r"diffusion $\hat\sigma/\sigma$")
a.axhline(1.0, color=GRAY, lw=0.8)
a.set_xscale("log"); a.set_ylim(0.35, 1.15)
a.set_xlabel(r"sampling interval $\Delta t$ (record length $T = 2000$)"); a.set_ylabel("estimate / true value")
a.set_title("Ornstein–Uhlenbeck: mean $\\pm$ 1 s.d., 20 records")
a.legend(fontsize=9.5, loc="lower left")

tr = kf["seed0_trajectory"]; t = np.array(tr["t"]); n = int(np.searchsorted(t, 10.0)) + 1
xt = np.array(tr["x_true"]); xe = np.array(tr["x_kalman"]); y = np.array(tr["y_obs"])
b.plot(t[:n], xt[:n, 0], color="black", lw=1.6, label="true position")
b.plot(t[:n], y[:n], ".", color=GRAY, ms=4, label="noisy observations")
b.plot(t[:n], xe[:n, 0], color=VERM, lw=1.3, ls="--", label="Kalman estimate")
b.set_xlabel("time"); b.set_ylabel("position"); b.set_title("Kalman filter, one realization")
b.set_ylim(-1.2, 1.2); b.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3,
                                  handlelength=1.6, columnspacing=1.0)
fig.tight_layout()
name = "F13.1_sde_and_kalman.png"
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / name, bbox_inches="tight")
fig.savefig(FIG_DIR / f"{name[:-4]}.pdf", bbox_inches="tight")
print("wrote", name)
