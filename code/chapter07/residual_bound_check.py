"""Numerical check of the residual-to-error bounds of Chapter 7, section 7.2 (Theorem 7.1), in its
hard-constraint and soft-constraint forms.

The bounds hold for any sufficiently smooth trial function, so any training state is a valid test.
The script trains, for two seeds each, (i) a soft-constraint network (residual, boundary and initial
losses) and (ii) a network with the boundary conditions imposed exactly, u = x(1-x) N(x, t), and the
initial condition as a penalty; both use the JAX network of code/chapter01/exA_three_ways.py and
1500 Adam iterations (learning rate 1e-3), an early stage of training in which the initial residual
dominates. The error, the residual norms and both bounds are then evaluated with trapezoidal
quadrature on a 401 x 401 grid, with C = (2 nu pi^2)^(-1/2).

Usage:
    python code/chapter07/residual_bound_check.py
Output: results/chapter07/ch08-t81_check.json
Runtime: a few minutes on one core. Seeded; requires jax and optax (requirements-jax.txt).
"""
import os, sys, json, numpy as np, jax, jax.numpy as jnp, optax
from pathlib import Path
jax.config.update("jax_enable_x64", True)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "chapter01"))
from exA_three_ways import init_params, mlp, u_fn, residual, LAYERS, exact, NU, T, N_R, N_B, N_0
OUT = str(REPO_ROOT / "results" / "chapter07")
xg = np.linspace(0, 1, 401); tg = np.linspace(0, T, 401); X, Tt = np.meshgrid(xg, tg, indexing="ij")
wx = np.full(401, 1/400); wx[[0, -1]] = 0.5/400; wt = wx.copy()
def l2x(F): return np.sqrt((F**2 * wx[:, None]).sum(0))            # ||.||_{L2(0,1)} for each t
def l2Q(F, tmax_idx=None): return np.sqrt(((F**2 * wx[:, None]) * wt[None, :]).sum())
res = {}
for mode in ("soft", "hard"):
    for seed in (0, 1):
        key = jax.random.PRNGKey(seed); kd, kp = jax.random.split(key)
        k1, k2, k3, k4 = jax.random.split(kd, 4)
        xr = jax.random.uniform(k1, (N_R,)); tr = jax.random.uniform(k2, (N_R,)) * T
        tb = jax.random.uniform(k3, (N_B,)) * T; x0 = jax.random.uniform(k4, (N_0,)); u0 = jnp.asarray(exact(np.asarray(x0), 0.0))
        p = init_params(kp, LAYERS)
        if mode == "hard":
            U = lambda p, x, t: x * (1 - x) * mlp(p, jnp.stack([x, t], -1))
        else:
            U = lambda p, x, t: mlp(p, jnp.stack([x, t], -1))
        def R(p, x, t):
            u = lambda x, t: U(p, x, t)
            return jax.vmap(jax.grad(u, 1))(x, t) - NU * jax.vmap(jax.grad(jax.grad(u, 0), 0))(x, t)
        def loss(p):
            l = jnp.mean(R(p, xr, tr)**2) + jnp.mean((U(p, x0, jnp.zeros(N_0)) - u0)**2)
            if mode == "soft": l += jnp.mean(U(p, jnp.zeros(N_B), tb)**2) + jnp.mean(U(p, jnp.ones(N_B), tb)**2)
            return l
        opt = optax.adam(1e-3); st = opt.init(p); step = jax.jit(lambda p, s: (lambda g: (optax.apply_updates(p, opt.update(g, s)[0]), opt.update(g, s)[1]))(jax.grad(loss)(p)))
        for it in range(1500): p, st = step(p, st)
        # components on the fine grid
        Ug = np.asarray(U(p, jnp.asarray(X.ravel()), jnp.asarray(Tt.ravel()))).reshape(X.shape)
        E = Ug - exact(X, Tt)
        Ri = np.asarray(R(p, jnp.asarray(X.ravel()), jnp.asarray(Tt.ravel()))).reshape(X.shape)
        r0 = Ug[:, 0] - exact(xg, 0.0)
        rb0 = Ug[0, :]; rb1 = Ug[-1, :]
        drb0 = np.asarray(jax.vmap(jax.grad(lambda t: U(p, 0.0, t)))(jnp.asarray(tg))); drb1 = np.asarray(jax.vmap(jax.grad(lambda t: U(p, 1.0, t)))(jnp.asarray(tg)))
        C = (2 * NU * np.pi**2) ** -0.5
        lhs = l2x(E).max()
        # hard form: ||e(t)|| <= ||r0|| + C ||r_i||_{L2(Q)}
        rhs_hard = np.sqrt((r0**2 * wx).sum()) + C * l2Q(Ri)
        # soft form via lifting l = (1-x) rb0 + x rb1
        L = (1 - X) * rb0[None, :] + X * rb1[None, :]; Lt = (1 - X) * drb0[None, :] + X * drb1[None, :]
        w0 = r0 - L[:, 0]
        rhs_soft = np.sqrt((w0**2 * wx).sum()) + C * l2Q(Ri - Lt) + (np.abs(rb0) + np.abs(rb1)).max() / np.sqrt(3)
        res[f"{mode}_seed{seed}"] = dict(max_t_L2_error=float(lhs), bound_hard_form=float(rhs_hard), bound_soft_form=float(rhs_soft),
                                          r0_L2=float(np.sqrt((r0**2*wx).sum())), ri_L2Q=float(l2Q(Ri)), rb_Linf=float(max(np.abs(rb0).max(), np.abs(rb1).max())),
                                          drb_L2=float(np.sqrt(((drb0**2 + drb1**2) * wt).sum())), C=float(C),
                                          hard_bound_valid=bool(lhs <= rhs_hard + 1e-12) if mode == "hard" else None,
                                          soft_bound_valid=bool(lhs <= rhs_soft + 1e-12))
        print(mode, seed, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in res[f"{mode}_seed{seed}"].items()})
json.dump(res, open(f"{OUT}/ch08-t81_check.json", "w"), indent=1)
