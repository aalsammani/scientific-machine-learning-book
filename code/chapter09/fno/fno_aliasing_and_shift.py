"""Failure modes of the Fourier neural operator (Chapter 9, sections 9.3 and 9.5; Figure 9.1, right
panel).

  (a) Aliasing: a 12-mode model (seed 0, retrained here with the protocol of
      fno_resolution_transfer.py) evaluated on grids of 8, 12, 16, 20, 32 and 64 points. Inputs
      contain modes up to 8, so 16 points is the Nyquist limit.
  (b) Distribution shift: inputs built from modes 9-16 only (never seen in training), coefficients
      from numpy.random.default_rng(999) with the envelope 1/(m + 1) for mode m, evaluated at
      64 points.
  (c) Mode truncation: a model keeping only 4 modes (seed 1), evaluated in distribution at
      64 points.
The seed-0 model trained here is a fresh training run in the same process as (c); its in-
distribution error (0.27%) differs slightly from the seed-0 entry of Table 9.1 (0.23%), which comes
from fno_resolution_transfer.py (see the initialization caveat in fno1d.py).

Usage:
    python code/chapter09/fno/fno_aliasing_and_shift.py
Output: results/chapter09/fno/failure_ood.json
Runtime: about 2 minutes on one core.
"""
import torch, numpy as np, json, time
from pathlib import Path
from fno1d import FNO1d, sample_u0_coeffs, eval_on_grid, exact_propagate, MAX_MODE_TRAIN, NU, T
from fno_resolution_transfer import train_seed, eval_resolution, N_TRAIN, N_TEST
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)

RESULTS = Path(__file__).resolve().parents[3] / "results" / "chapter09" / "fno"

# Retrain the seed-0, 12-mode configuration once; it is used for both the aliasing and the shift test.
model, coeffs_all, train_s = train_seed(0, modes=12, width=16, adam_iters=2000)
coeffs_test_indist = coeffs_all[N_TRAIN:]

# (a) Aliasing at coarse resolution: inputs contain modes up to 8, so N = 16 points is the Nyquist
# limit (the mode-8 cosine is still sampled, its sine part is lost); N = 64 is a well-resolved control.
aliasing_results = {}
for n_grid in [8, 12, 16, 20, 32, 64]:
    aliasing_results[n_grid] = eval_resolution(model, coeffs_test_indist, n_grid)
print("aliasing sweep:", {k: round(v['median_rel_l2'],4) for k,v in aliasing_results.items()})

# (b) OOD distribution shift: input drawn from HIGHER frequency modes (9-16) never seen in training
rng_ood = np.random.default_rng(999)
def sample_ood_coeffs(n_samples, rng):
    modes_lo, modes_hi = 9, 16
    n_modes = modes_hi - modes_lo + 1
    re = rng.normal(0, 1.0, (n_samples, n_modes))
    im = rng.normal(0, 1.0, (n_samples, n_modes))
    c = re + 1j*im
    decay = np.array([1.0/(k+1) for k in range(modes_lo, modes_hi+1)])
    c = c*decay[None,:]
    return c, list(range(modes_lo, modes_hi+1))

def eval_ood(model, n_samples, n_grid, rng):
    coeffs, modes_list = sample_ood_coeffs(n_samples, rng)
    x = np.linspace(0, 2*np.pi, n_grid, endpoint=False)
    u0 = np.zeros((n_samples, n_grid)); u_true = np.zeros((n_samples, n_grid))
    for i,k in enumerate(modes_list):
        ck = coeffs[:,i]
        u0 += ck.real[:,None]*np.cos(k*x)[None,:] - ck.imag[:,None]*np.sin(k*x)[None,:]
        ck_decayed = ck*np.exp(-NU*k**2*T)
        u_true += ck_decayed.real[:,None]*np.cos(k*x)[None,:] - ck_decayed.imag[:,None]*np.sin(k*x)[None,:]
    with torch.no_grad():
        pred = model(torch.tensor(u0)).numpy()
    relerr = np.linalg.norm(pred-u_true,axis=1)/np.linalg.norm(u_true,axis=1)
    return float(np.median(relerr))

ood_relerr_64 = eval_ood(model, 100, 64, rng_ood)
print("OOD (modes 9-16) @ res 64:", ood_relerr_64, " vs in-distribution:", aliasing_results[64]['median_rel_l2'])

# (c) Mode-truncation failure: train a SECOND model with too few retained Fourier modes (modes=4)
# on the SAME training distribution (modes 1-8), and test both in-distribution and on OOD high-freq input
model_lowmodes, coeffs_all2, _ = train_seed(1, modes=4, width=16, adam_iters=2000)
coeffs_test2 = coeffs_all2[N_TRAIN:]
lowmodes_indist = eval_resolution(model_lowmodes, coeffs_test2, 64)
print("low-mode model (modes=4) in-distribution @ res64:", lowmodes_indist['median_rel_l2'])

json.dump(dict(aliasing_sweep=aliasing_results, ood_relerr_modes9to16_res64=ood_relerr_64,
               indist_relerr_res64=aliasing_results[64]['median_rel_l2'],
               lowmodes_model_modes=4, lowmodes_indist_relerr_res64=lowmodes_indist['median_rel_l2'],
               fullmodes_model_modes=12),
          open(RESULTS / "failure_ood.json", "w"), indent=1)
