"""Frequency-ordered learning on the initial condition alone (controls for Chapter 1, section 1.9).

  (a) MLP [1, 32, 32, 32, 1] (tanh, LeCun normal) fitted to a1 sin(pi x) + a5 sin(5 pi x) on 201
      points by full-batch gradient descent (learning rate 0.05, 20001 iterations), with the book's
      amplitudes (1, 0.5) and with swapped amplitudes (0.5, 1); seeds 0-4.
  (b) Linear sine-basis model u = sum_k c_k sin(k pi x) fitted by gradient descent from c = 0
      (learning rate 0.5, 2001 iterations); one deterministic run (null control).
Relative errors of the first and fifth sine coefficients are recorded every 200 (MLP) or 20
(linear model) iterations.

Usage:
    python code/chapter01/spectral_bias_control.py
Output: results/chapter01/ch01-f01_control.json
Runtime: a few minutes on one CPU core. Seeded; requires jax (requirements-jax.txt).
"""
import json, os, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from exA_three_ways import init_params, mlp, LAYERS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "chapter01")
x = jnp.linspace(0, 1, 201); K = 8; ks = jnp.arange(1, K + 1)
S = jnp.sin(jnp.pi * ks[:, None] * x[None, :])
w = jnp.full(201, 1 / 200).at[0].set(0.5 / 200).at[-1].set(0.5 / 200)
def coeffs(u): return 2 * (S * (u * w)[None, :]).sum(1)
def target(a1, a5): return a1 * jnp.sin(jnp.pi * x) + a5 * jnp.sin(5 * jnp.pi * x)
res = {}
for name, (a1, a5) in (("mlp_gd_original", (1.0, 0.5)), ("mlp_gd_swapped", (0.5, 1.0))):
    y = target(a1, a5); ex = np.array(coeffs(y))
    for seed in range(5):
        p = init_params(jax.random.PRNGKey(seed), [1] + LAYERS[1:])
        loss = lambda p: jnp.mean((mlp(p, x[:, None]) - y) ** 2)
        g = jax.jit(jax.grad(loss)); lr = 0.05
        step = jax.jit(lambda p: jax.tree_util.tree_map(lambda a, b: a - lr * b, p, g(p)))
        hist = []
        for it in range(20001):
            if it % 200 == 0:
                c = np.array(coeffs(mlp(p, x[:, None]))); hist.append((it, float(abs(c[0]-ex[0])/abs(ex[0])), float(abs(c[4]-ex[4])/abs(ex[4]))))
            p = step(p)
        res[f"{name}_seed{seed}"] = hist
        print(name, seed, "iters to rel err<0.1: k1", next((i for i, e1, e5 in hist if e1 < 0.1), None), "k5", next((i for i, e1, e5 in hist if e5 < 0.1), None))
# linear null control: u = sum c_k sin(k pi x), GD on c from zero
y = target(1.0, 0.5); ex = np.array(coeffs(y)); c = jnp.zeros(K)
lossl = lambda c: jnp.mean((c @ S - y) ** 2); gl = jax.jit(jax.grad(lossl)); lr = 0.5
hist = []
for it in range(2001):
    if it % 20 == 0: hist.append((it, float(abs(c[0]-ex[0])/abs(ex[0])), float(abs(c[4]-ex[4])/abs(ex[4]))))
    c = c - lr * gl(c)
res["linear_sine_basis_gd"] = hist
print("linear: iters to <0.1: k1", next((i for i, e1, e5 in hist if e1 < 0.1), None), "k5", next((i for i, e1, e5 in hist if e5 < 0.1), None))
json.dump(res, open(f"{OUT}/ch01-f01_control.json", "w"))
