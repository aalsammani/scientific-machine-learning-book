"""Adam-only fit of the Example A initial condition with three amplitude configurations (control for
Chapter 1, section 1.9): does the higher mode k = 5 converge later independently of its amplitude
and of the data support?

Configurations (a1, a5): (1, 0.5) as in the book, (0.5, 1) swapped, (1, 1) equal. Dense data on 201
points; MLP [1, 32, 32, 32, 1], tanh, LeCun normal; Adam (learning rate 1e-3, 6001 iterations);
seeds 0-4. Sine coefficients are computed by the trapezoidal rule on 401 points every 50 iterations.

Usage:
    python code/chapter01/exA_spectral_control.py
Output: results/chapter01/ch01-f01_spectral_control.json
Runtime: a few minutes on one CPU core. Seeded; requires jax and optax (requirements-jax.txt).
"""
import json, os, sys, numpy as np, jax, jax.numpy as jnp, optax
from pathlib import Path
jax.config.update("jax_enable_x64", True)
OUT = str(Path(__file__).resolve().parents[2] / "results" / "chapter01")
def init(key, layers=(1, 32, 32, 32, 1)):
    p = []
    for a, b in zip(layers[:-1], layers[1:]):
        key, k = jax.random.split(key); p.append((jax.random.normal(k, (a, b)) * jnp.sqrt(1 / a), jnp.zeros(b)))
    return p
def mlp(p, x):
    h = x[:, None]
    for W, b in p[:-1]: h = jnp.tanh(h @ W + b)
    return (h @ p[-1][0] + p[-1][1])[:, 0]
xg = jnp.linspace(0, 1, 401); w = jnp.full(401, 1 / 400).at[0].set(0.5 / 400).at[-1].set(0.5 / 400)
S = jnp.sin(jnp.pi * jnp.arange(1, 9)[:, None] * xg[None, :])
def coeffs(p): return 2 * (S * (mlp(p, xg) * w)[None, :]).sum(1)
res = {}
for name, (a1, a5) in {"book_IC_1_0.5": (1.0, 0.5), "swapped_0.5_1": (0.5, 1.0), "equal_1_1": (1.0, 1.0)}.items():
    xd = jnp.linspace(0, 1, 201); yd = a1 * jnp.sin(jnp.pi * xd) + a5 * jnp.sin(5 * jnp.pi * xd)
    loss = lambda p: jnp.mean((mlp(p, xd) - yd) ** 2)
    opt = optax.adam(1e-3)
    @jax.jit
    def step(p, s):
        l, g = jax.value_and_grad(loss)(p); u, s = opt.update(g, s); return optax.apply_updates(p, u), s, l
    runs = []
    for seed in range(5):
        p = init(jax.random.PRNGKey(seed)); s = opt.init(p); hist = []
        for it in range(6001):
            p, s, l = step(p, s)
            if it % 50 == 0:
                c = np.asarray(coeffs(p)); hist.append(dict(iter=it, loss=float(l), e1=float(abs(c[0] - a1) / a1), e5=float(abs(c[4] - a5) / a5)))
        f = lambda key, th: next((h["iter"] for h in hist if h[key] < th), None)
        runs.append(dict(seed=seed, first_k1_lt10=f("e1", .1), first_k5_lt10=f("e5", .1), first_k1_lt1=f("e1", .01), first_k5_lt1=f("e5", .01), final_e1=hist[-1]["e1"], final_e5=hist[-1]["e5"], hist=hist))
        print(name, {k: v for k, v in runs[-1].items() if k != "hist"}, flush=True)
    res[name] = runs
json.dump(res, open(f"{OUT}/ch01-f01_spectral_control.json", "w"))
