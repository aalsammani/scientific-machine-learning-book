"""Running Example A three ways (JAX implementation): verified finite differences, supervised
regression, and a physics-informed network, with sine-mode tracking during training.

Book: Chapter 1, section 1.8 (Tables 1.3-1.5, section 1.8.5) and section 1.9 (Figure 1.5 uses the
mode histories written here). code/chapter01/spectral_bias_control.py and
code/chapter07/residual_bound_check.py import the network definitions from this module.

Problem (code/reference_problems/exA_heat.py): u_t = 0.5 u_xx on (0,1) x (0,1], zero Dirichlet data,
u(x,0) = sin(pi x) + 0.5 sin(5 pi x); exact solution known in closed form.
  * Finite differences (result prefix ch01-e01): FTCS on 81 points with r = 0.4.
  * Regression (ch01-e02): 2000 points drawn without replacement from the FTCS solution at t > 0.
  * Physics-informed network (ch01-e03): residual, boundary and initial-condition mean-squared losses
    with unit weights; n_r = 2000, n_b = 100 per boundary, n_0 = 100 uniformly random points
    drawn from jax.random.PRNGKey(seed).
  * Network [2, 32, 32, 32, 1], tanh, LeCun-normal initialization, 2241 parameters, float64.
  * Adam (5000 iterations, learning rate 1e-3), then SciPy L-BFGS-B (at most 10000 iterations,
    maxcor 50, ftol 1e-15, gtol 1e-12).
  * Evaluation on a 201 x 101 grid against the exact solution; extrapolation on t in (1, 1.5].
  * Time-blind baselines (ch01-f02): the initial condition used as the prediction at all times.
The environment variables ADAM_ITERS and LBFGS_ITERS override the iteration counts for quick smoke
tests only; the book's results use the defaults.

Usage:
    python code/chapter01/exA_three_ways.py                # FD, baselines, both learned methods, seeds 0-4
    python code/chapter01/exA_three_ways.py pinn 0 1 2     # physics-informed network only, chosen seeds
    python code/chapter01/exA_three_ways.py regression 0

Outputs (results/chapter01/): ch01-e01_fd.json, ch01-f02_timeblind.json, environment.json,
ch01-e02_regression_seed{s}.json, ch01-e03_pinn_seed{s}.json and the matching *_history.json files
(loss and sine-mode coefficients every 100 iterations). Every call also rewrites ch01-e01_fd.json,
ch01-f02_timeblind.json and environment.json (the FD and baseline parts are cheap and deterministic;
environment.json then describes the machine of the rerun).
Runtime (recorded environment, one CPU core): about 130 s per physics-informed run and 30 s per
regression run. Stochastic but seeded; repeatable on a fixed platform and software stack.
Requires jax and optax (requirements-jax.txt).
"""
import json, time, sys, os, platform
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp, optax
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "reference_problems"))
from exA_heat import exact, ftcs, NU, T, MODES

OUT = str(REPO_ROOT / "results" / "chapter01")
os.makedirs(OUT, exist_ok=True)
LAYERS = [2, 32, 32, 32, 1]
ADAM_ITERS, ADAM_LR, LBFGS_ITERS = int(os.environ.get("ADAM_ITERS",5000)), 1e-3, int(os.environ.get("LBFGS_ITERS",10000))
N_R, N_B, N_0 = 2000, 100, 100
TRACK_EVERY = 100
EVAL_X, EVAL_T = np.linspace(0, 1, 201), np.linspace(0, T, 101)
EXTRAP_T = np.linspace(T, 1.5 * T, 51)[1:]


def init_params(key, layers):
    params = []
    for din, dout in zip(layers[:-1], layers[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (din, dout)) * jnp.sqrt(1.0 / din)  # LeCun normal (tanh)
        params.append((W, jnp.zeros(dout)))
    return params


def mlp(params, xt):
    h = xt
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[..., 0]


def n_params(params): return int(sum(W.size + b.size for W, b in params))


def u_fn(params, x, t): return mlp(params, jnp.stack([x, t], -1))


def residual(params, x, t):
    u = lambda x, t: u_fn(params, x, t)
    u_t = jax.vmap(jax.grad(u, 1))(x, t)
    u_xx = jax.vmap(jax.grad(jax.grad(u, 0), 0))(x, t)
    return u_t - NU * u_xx


def flatten(params):
    leaves, tree = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    return jnp.concatenate([l.ravel() for l in leaves]), (tree, shapes)


def unflatten(vec, meta):
    tree, shapes = meta; out = []; i = 0
    for s in shapes:
        n = int(np.prod(s)); out.append(vec[i:i + n].reshape(s)); i += n
    return jax.tree_util.tree_unflatten(tree, out)


def mode_coeffs(params, t_val, K=8):
    """Sine-series coefficients of u_theta(., t_val) on (0,1) by trapezoid with 401 points."""
    xg = jnp.linspace(0, 1, 401)
    u = u_fn(params, xg, jnp.full_like(xg, t_val))
    ks = jnp.arange(1, K + 1)
    S = jnp.sin(jnp.pi * ks[:, None] * xg[None, :])
    w = jnp.full(401, 1 / 400); w = w.at[0].set(0.5 / 400); w = w.at[-1].set(0.5 / 400)
    return 2 * (S * (u * w)[None, :]).sum(1)


def exact_coeffs(t_val, K=8):
    c = np.zeros(K)
    for k, a in MODES: c[k - 1] = a * np.exp(-NU * (k * np.pi) ** 2 * t_val)
    return c


TRACK_T = (0.0, 0.02, 0.1)


def track(params):
    return {f"t{t}": np.asarray(mode_coeffs(params, t)).tolist() for t in TRACK_T}


def evaluate(params, extra=None):
    X, Tg = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
    Uex = exact(X, Tg)
    t0 = time.perf_counter()
    U = np.asarray(u_fn(params, jnp.asarray(X.ravel()), jnp.asarray(Tg.ravel()))).reshape(X.shape)
    inf_t = time.perf_counter() - t0
    e = U - Uex
    Xi, Ti = X[1:-1, 1:], Tg[1:-1, 1:]
    R = np.asarray(residual(params, jnp.asarray(Xi.ravel()), jnp.asarray(Ti.ravel())))
    bc = np.abs(U[[0, -1], :]).max()
    ic = np.linalg.norm(U[:, 0] - Uex[:, 0]) / np.linalg.norm(Uex[:, 0])
    Xe, Te = np.meshgrid(EVAL_X, EXTRAP_T, indexing="ij")
    Ue = np.asarray(u_fn(params, jnp.asarray(Xe.ravel()), jnp.asarray(Te.ravel()))).reshape(Xe.shape)
    ex = exact(Xe, Te)
    return dict(rel_l2=float(np.linalg.norm(e) / np.linalg.norm(Uex)),
                rel_linf=float(np.abs(e).max() / np.abs(Uex).max()),
                rel_residual_l2=float(np.linalg.norm(R) / np.linalg.norm(Uex[1:-1, 1:])),
                residual_linf=float(np.abs(R).max()),
                bc_linf=float(bc), ic_rel_l2=float(ic),
                extrap_rel_l2_vs_extrap_norm=float(np.linalg.norm(Ue - ex) / np.linalg.norm(ex)),
                extrap_l2_rel_to_domain_norm=float(np.linalg.norm(Ue - ex) / np.linalg.norm(Uex) * np.sqrt(Uex.size / ex.size)),
                extrap_abs_linf=float(np.abs(Ue - ex).max()),
                inference_s_per_20301_points=inf_t, n_params=n_params(params))


def train(loss_fn, params, tag, seed):
    """Adam then L-BFGS; returns params, history, wall-clocks."""
    hist = []
    opt = optax.adam(ADAM_LR); state = opt.init(params)
    vg = jax.jit(jax.value_and_grad(loss_fn))

    @jax.jit
    def step(p, s):
        l, g = vg(p); upd, s = opt.update(g, s); return optax.apply_updates(p, upd), s, l
    t0 = time.perf_counter()
    for it in range(ADAM_ITERS):
        params, state, l = step(params, state)
        if it % TRACK_EVERY == 0:
            hist.append(dict(iter=it, phase="adam", loss=float(l), modes=track(params)))
    adam_wall = time.perf_counter() - t0
    vec, meta = flatten(params)
    f = jax.jit(lambda v: jax.value_and_grad(lambda v: loss_fn(unflatten(v, meta)))(v))
    it_counter = [ADAM_ITERS]
    def fun(v):
        l, g = f(jnp.asarray(v)); return float(l), np.asarray(g, dtype=np.float64)
    def cb(v):
        it_counter[0] += 1
        if it_counter[0] % TRACK_EVERY == 0:
            p = unflatten(jnp.asarray(v), meta)
            hist.append(dict(iter=it_counter[0], phase="lbfgs", loss=float(loss_fn(p)), modes=track(p)))
    t0 = time.perf_counter()
    res = minimize(fun, np.asarray(vec), jac=True, method="L-BFGS-B", callback=cb,
                   options=dict(maxiter=LBFGS_ITERS, maxfun=LBFGS_ITERS * 2, ftol=1e-15, gtol=1e-12, maxcor=50))
    lbfgs_wall = time.perf_counter() - t0
    params = unflatten(jnp.asarray(res.x), meta)
    hist.append(dict(iter=it_counter[0], phase="final", loss=float(res.fun), modes=track(params)))
    return params, hist, dict(adam_s=adam_wall, lbfgs_s=lbfgs_wall, lbfgs_iters=int(res.nit), lbfgs_msg=str(res.message))


def run_pinn(seed):
    key = jax.random.PRNGKey(seed)
    kd, kp = jax.random.split(key)
    k1, k2, k3, k4 = jax.random.split(kd, 4)
    xr = jax.random.uniform(k1, (N_R,)); tr = jax.random.uniform(k2, (N_R,)) * T
    tb = jax.random.uniform(k3, (N_B,)) * T
    x0 = jax.random.uniform(k4, (N_0,)); u0 = jnp.asarray(exact(np.asarray(x0), 0.0))
    def loss(p):
        lr = jnp.mean(residual(p, xr, tr) ** 2)
        lb = jnp.mean(u_fn(p, jnp.zeros(N_B), tb) ** 2) + jnp.mean(u_fn(p, jnp.ones(N_B), tb) ** 2)
        l0 = jnp.mean((u_fn(p, x0, jnp.zeros(N_0)) - u0) ** 2)
        return lr + lb + l0
    params = init_params(kp, LAYERS)
    params, hist, walls = train(loss, params, "pinn", seed)
    global _last_params; _last_params = params
    m = evaluate(params); m.update(walls); m["seed"] = seed
    return m, hist


def run_regression(seed, fd_data):
    x_d, t_d, u_d = fd_data
    key = jax.random.PRNGKey(seed)
    kd, kp = jax.random.split(key)
    idx = jax.random.choice(kd, x_d.shape[0], (N_R,), replace=False)
    xs, ts, us = jnp.asarray(x_d[idx]), jnp.asarray(t_d[idx]), jnp.asarray(u_d[idx])
    def loss(p): return jnp.mean((u_fn(p, xs, ts) - us) ** 2)
    params = init_params(kp, LAYERS)
    params, hist, walls = train(loss, params, "regression", seed)
    m = evaluate(params); m.update(walls); m["seed"] = seed
    return m, hist


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[2:]] or [0, 1, 2, 3, 4]
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which not in ("both", "pinn", "regression"):
        raise SystemExit("usage: python code/chapter01/exA_three_ways.py [both|pinn|regression] [SEED ...]")
    # ch01-e01: verified FTCS solution (r = 0.4) on 81 points; also the regression data source
    nx = 81; dx = 1 / (nx - 1); nt = int(np.ceil(T / (0.4 * dx ** 2 / NU)))
    t0 = time.perf_counter(); xg, tg, U = ftcs(nx, nt); fd_wall = time.perf_counter() - t0
    Xg, Tg = np.meshgrid(xg, tg, indexing="ij"); Uex = exact(Xg, Tg)
    fd = dict(nx=nx, nt=nt, rel_l2=float(np.linalg.norm(U.T - Uex) / np.linalg.norm(Uex)),
              rel_linf=float(np.abs(U.T - Uex).max() / np.abs(Uex).max()), wall_s=fd_wall, n_dof=nx - 2)
    json.dump(fd, open(f"{OUT}/ch01-e01_fd.json", "w"), indent=1)
    mask = (Tg > 0)  # exclude t=0 to mimic interior data
    fd_data = (Xg[mask], Tg[mask], U.T[mask])
    # ch01-f02: time-blind baselines
    tb = {}
    for nu_case in (NU, 0.01):
        X, Tt = np.meshgrid(EVAL_X, EVAL_T, indexing="ij")
        ue = exact(X, Tt, nu=nu_case); ic = exact(X, 0.0 * Tt, nu=nu_case)
        tb[f"nu_{nu_case}_IC_as_prediction"] = float(np.linalg.norm(ic - ue) / np.linalg.norm(ue))
        ue1 = exact(X, Tt, nu=nu_case, modes=((1, 1.0),)); ic1 = exact(X, 0 * Tt, nu=nu_case, modes=((1, 1.0),))
        tb[f"nu_{nu_case}_single_mode_IC_as_prediction"] = float(np.linalg.norm(ic1 - ue1) / np.linalg.norm(ue1))
    json.dump(tb, open(f"{OUT}/ch01-f02_timeblind.json", "w"), indent=1)
    print("FD:", fd); print("time-blind:", tb)
    env = dict(platform=platform.platform(), jax=jax.__version__, optax=optax.__version__, device=str(jax.devices()[0]),
               float64=True, layers=LAYERS, adam=dict(iters=ADAM_ITERS, lr=ADAM_LR), lbfgs_iters=LBFGS_ITERS,
               n_r=N_R, n_b=N_B, n_0=N_0)
    json.dump(env, open(f"{OUT}/environment.json", "w"), indent=1)
    for seed in seeds:
        for tag, fn in (("ch01-e03_pinn", run_pinn), ("ch01-e02_regression", lambda s: run_regression(s, fd_data))):
            if which != "both" and which not in tag: continue
            m, hist = fn(seed)
            json.dump(m, open(f"{OUT}/{tag}_seed{seed}.json", "w"), indent=1)
            json.dump(hist, open(f"{OUT}/{tag}_seed{seed}_history.json", "w"))
            print(tag, seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k in ("rel_l2", "rel_linf", "rel_residual_l2", "extrap_abs_linf", "adam_s", "lbfgs_s")}, flush=True)
