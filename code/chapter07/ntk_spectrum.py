"""Residual neural-tangent-kernel spectrum at initialization for Running Example A (Chapter 7,
section 7.4; Figure 7.1, left panel).

Computes K_ij = <grad_theta N[u_theta](x_i), grad_theta N[u_theta](x_j)> with N[u] = u_t - nu u_xx
(nu = 0.5) for the 3 x 32 tanh network of code/chapter01/exA_three_ways_pytorch.py (LeCun-normal
weights, zero biases, float64) on 80 points x = linspace(0.02, 0.98, 80) at a fixed time slice.

Provenance: the stored spectrum results/chapter07/ntk_spectrum.json (plotted in Figure 7.1) has no
preserved producing script and does not record the seed, time slice or Fourier convention. This
script
  1. identifies the Fourier convention from the stored eigenvectors;
  2. searches initialization, seed and time slice for the configuration reproducing the stored
     spectrum (seed 0, t = 0.1 reproduces the first eigenvalues to about 1e-10 relative);
  3. reports the full spectrum for that configuration and for ten seeds, from eigh(K) and from the
     SVD of the Jacobian J (K = J J^T), and marks which eigenpairs are above the rounding floor.
Eigenvalues below the floor (including the negative ones) carry no information about frequency.

Usage:
    python code/chapter07/ntk_spectrum.py
Output: results/chapter07/ntk_spectrum_rerun.json (the stored file is only read).
Runtime: about a minute on one core. Deterministic.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter01"))
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import NU, T  # noqa: E402
from exA_three_ways_pytorch import MLP, LAYERS  # noqa: E402

RES = str(REPO_ROOT / "results" / "chapter07")
STORED = os.path.join(RES, "ntk_spectrum.json")
OUT = os.path.join(RES, "ntk_spectrum_rerun.json")

N_PTS = 80
X = np.linspace(0.02, 0.98, N_PTS)
EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------- model / kernel
def make_model(seed, init):
    if init == "lecun_normal_zero_bias":          # reference architecture (Ch. 1 / Ch. 6)
        return MLP(LAYERS, seed)
    if init == "torch_default":                    # nn.Linear default init under manual_seed
        torch.manual_seed(seed)
        m = MLP(LAYERS, seed)
        torch.manual_seed(seed)
        for lin in m.linears:
            lin.reset_parameters()
        return m
    raise ValueError(init)


def residual_jacobian(model, x, t_val):
    """Rows: grad_theta of r_i = u_t - NU u_xx at (x_i, t_val). Final-layer bias excluded
    (r does not depend on it, so its column is identically zero)."""
    params = [p for n, p in model.named_parameters() if n != f"linears.{len(LAYERS) - 2}.bias"]
    xt = torch.tensor(x).clone().requires_grad_(True)
    tt = torch.full((len(x),), float(t_val)).requires_grad_(True)
    u = model(xt, tt)
    u_t, u_x = torch.autograd.grad(u, (tt, xt), torch.ones_like(u), create_graph=True)
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0]
    r = u_t - NU * u_xx
    rows = []
    for i in range(len(x)):
        g = torch.autograd.grad(r[i], params, retain_graph=True)
        rows.append(torch.cat([gi.reshape(-1) for gi in g]))
    return torch.stack(rows).detach().numpy()


def spectrum(J):
    K = J @ J.T
    w, V = np.linalg.eigh(K)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    U, s, _ = np.linalg.svd(J, full_matrices=False)   # s descending
    return K, w, V, s ** 2, U


# ---------------------------------------------------------------- Fourier conventions
def dom_freq(v, conv):
    v = np.asarray(v, float)
    if conv == "sine_k1_80":
        ks = np.arange(1, N_PTS + 1)
        c = np.sin(np.pi * ks[:, None] * X[None, :]) @ v
        return int(ks[np.argmax(np.abs(c))])
    if conv == "rfft":
        return int(np.argmax(np.abs(np.fft.rfft(v))))
    if conv == "rfft_nodc":
        return int(np.argmax(np.abs(np.fft.rfft(v))[1:]) + 1)
    if conv == "rfft_demean":
        return int(np.argmax(np.abs(np.fft.rfft(v - v.mean()))))
    if conv == "fft_nodc":
        F = np.abs(np.fft.fft(v))[1:N_PTS // 2 + 1]
        return int(np.argmax(F) + 1)
    raise ValueError(conv)


CONVS = ["sine_k1_80", "rfft", "rfft_nodc", "rfft_demean", "fft_nodc"]


def freqs(V, cols, conv):
    return [dom_freq(V[:, j], conv) for j in cols]


# ---------------------------------------------------------------- summaries
def summarize(w, V, s2, U, conv):
    lam1 = w[0]
    floor_eigh = N_PTS * EPS * lam1              # rounding floor of eigh(K)
    floor_svd = (N_PTS * EPS) ** 2 * lam1        # sigma_i^2 below (n eps sigma_1)^2 not resolved
    n_res_eigh = int(np.sum(w > floor_eigh))
    n_res_svd = int(np.sum(s2 > floor_svd))
    return dict(
        eig_first12=w[:12].tolist(),
        svd_sq_first20=s2[:20].tolist(),
        orders_first8=float(np.log10(w[0] / w[7])),
        orders_first12=float(np.log10(w[0] / w[11])) if w[11] > 0 else None,
        orders_first12_svd=float(np.log10(s2[0] / s2[11])),
        n_negative_eigs=int(np.sum(w < 0)),
        eigh_noise_floor=floor_eigh,
        n_resolved_eigh=n_res_eigh,
        svd_noise_floor=floor_svd,
        n_resolved_svd=n_res_svd,
        top5_freqs=freqs(V, range(5), conv),
        top5_freqs_rfft_with_dc=freqs(V, range(5), "rfft"),
        top5_freqs_sine_basis=freqs(V, range(5), "sine_k1_80"),
        bottom5_freqs_eigh_last5_descending_order=freqs(V, range(N_PTS - 5, N_PTS), conv),
        bottom5_freqs_eigh_smallest_first=freqs(V, range(N_PTS - 1, N_PTS - 6, -1), conv),
        resolved_eigh_freqs=freqs(V, range(n_res_eigh), conv),
        resolved_svd_freqs=freqs(U, range(n_res_svd), conv),
        smallest5_resolved_svd_freqs=freqs(U, range(max(0, n_res_svd - 5), n_res_svd), conv),
    )


def main():
    t0 = time.perf_counter()
    stored = json.load(open(STORED))
    sw = np.array(stored["eigvals"])
    assert stored["n"] == N_PTS and np.allclose(stored["x"], X), "stored x grid differs"
    out = dict(script="code/chapter07/ntk_spectrum.py", nu=NU, T=T, layers=LAYERS, n_points=N_PTS,
               x="linspace(0.02, 0.98, 80) (identical to stored x)", dtype="float64",
               torch=torch.__version__)

    # 1. Fourier convention, identified on the stored eigenvectors
    top = np.array(stored["eigvecs_top5"]); bot = np.array(stored["eigvecs_bottom5"])
    conv_check = {}
    for c in CONVS:
        conv_check[c] = dict(top=[dom_freq(top[:, j], c) for j in range(5)],
                             bottom=[dom_freq(bot[:, j], c) for j in range(5)])
    # stored eigvecs_bottom5 columns are in the reverse order of stored bottom_eigvec_freqs
    matches = [c for c in CONVS if conv_check[c]["top"] == stored["top_eigvec_freqs"]
               and conv_check[c]["bottom"] in (stored["bottom_eigvec_freqs"], stored["bottom_eigvec_freqs"][::-1])]
    conv = "rfft_nodc" if "rfft_nodc" in matches else (matches[0] if matches else "rfft_nodc")
    out["fourier_convention_check_on_stored_vectors"] = conv_check
    out["fourier_convention_used"] = conv
    out["fourier_convention_matches_stored"] = matches
    print("convention check:", conv_check, "-> using", conv, flush=True)

    # stored-file diagnostics
    lam1 = sw[0]
    out["stored_diagnostics"] = dict(
        n_negative=int(np.sum(sw < 0)),
        n_above_eigh_floor=int(np.sum(sw > N_PTS * EPS * lam1)),
        eigh_floor=N_PTS * EPS * lam1,
        smallest5=sorted(sw.tolist())[:5],
        orders_first8=float(np.log10(sw[0] / sw[7])),
        orders_first12=float(np.log10(sw[0] / sw[11])))

    # 2. search over init, seed, time slice
    seeds = list(range(10))
    tvals = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
    inits = ["lecun_normal_zero_bias", "torch_default"]
    search = []
    for init in inits:
        for seed in seeds:
            model = make_model(seed, init)
            for tv in tvals:
                J = residual_jacobian(model, X, tv)
                w = np.sort(np.linalg.eigvalsh(J @ J.T))[::-1]
                err = float(np.max(np.abs(w[:8] - sw[:8]) / sw[:8]))
                search.append(dict(init=init, seed=seed, t=tv, first4=w[:4].tolist(), max_rel_err_first8=err))
    search.sort(key=lambda d: d["max_rel_err_first8"])
    out["search_best10"] = search[:10]
    best = search[0]
    print("best matches:", flush=True)
    for d in search[:10]:
        print("  ", d, flush=True)

    # 3. full analysis at the best configuration
    model = make_model(best["seed"], best["init"])
    J = residual_jacobian(model, X, best["t"])
    K, w, V, s2, U = spectrum(J)
    out["best_config"] = dict(init=best["init"], seed=best["seed"], t=best["t"])
    bs = summarize(w, V, s2, U, conv)
    bs["eigvals_all"] = w.tolist()
    bs["rel_diff_vs_stored_first12"] = ((w[:12] - sw[:12]) / sw[:12]).tolist()
    bs["top5_eigvec_abs_cosine_vs_stored"] = [float(abs(V[:, j] @ top[:, j])) for j in range(5)]
    out["best_config_result"] = bs
    print("best-config summary:", json.dumps({k: v for k, v in bs.items() if k != "eigvals_all"}, indent=1), flush=True)

    # 4. seed sweep with the reference init at the best time slice and at t = 0.5
    sweep = {}
    for tv in sorted({best["t"], 0.5}):
        rows = []
        for seed in range(10):
            model = make_model(seed, "lecun_normal_zero_bias")
            J = residual_jacobian(model, X, tv)
            K, w, V, s2, U = spectrum(J)
            s = summarize(w, V, s2, U, conv)
            s["seed"] = seed
            rows.append(s)
            print(f"t={tv} seed={seed} eig1-8={np.round(w[:8], 6).tolist()} orders8={s['orders_first8']:.2f} "
                  f"orders12(svd)={s['orders_first12_svd']:.2f} neg={s['n_negative_eigs']} "
                  f"nres_eigh={s['n_resolved_eigh']} nres_svd={s['n_resolved_svd']} top5={s['top5_freqs']} "
                  f"bot5={s['bottom5_freqs_eigh_last5_descending_order']} res_eigh_freqs={s['resolved_eigh_freqs']} "
                  f"svd_small5={s['smallest5_resolved_svd_freqs']}", flush=True)
        sweep[f"t={tv}"] = rows
    out["seed_sweep_lecun"] = sweep
    out["wall_s"] = time.perf_counter() - t0
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, "wall", round(out["wall_s"], 1), "s")


if __name__ == "__main__":
    main()
