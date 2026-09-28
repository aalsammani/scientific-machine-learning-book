"""Derivative correctness (adjoint and automatic differentiation against finite differences),
network shapes and parameter counts quoted in the book."""
import numpy as np
import pytest
import torch

from conftest import load_module


def test_continuous_adjoint_matches_exact_gradient():
    adj = load_module("chapter05/adjoint_vs_fd_verification.py")
    g_adj, g_fd = adj.grads(2.2, 3001)
    g_ex = adj.exact_grad(2.2)
    # on the refined grid the trapezoidal-quadrature error is small (section 5.7)
    assert abs(g_adj / g_ex - 1) < 1e-6
    assert abs(g_fd / g_ex - 1) < 1e-6


def test_pinn_residual_derivatives_match_finite_differences():
    ex = load_module("chapter01/exA_three_ways_pytorch.py")
    model = ex.MLP(ex.LAYERS, 0)
    x0, t0, h = 0.37, 0.41, 1e-4
    x = torch.tensor([x0], dtype=torch.float64); t = torch.tensor([t0], dtype=torch.float64)
    r_ad = ex.residual(model, x, t).item()
    with torch.no_grad():
        u = lambda a, b: model(torch.tensor([a]), torch.tensor([b])).item()
        u_t = (u(x0, t0 + h) - u(x0, t0 - h)) / (2 * h)
        u_xx = (u(x0 + h, t0) - 2 * u(x0, t0) + u(x0 - h, t0)) / h ** 2
    assert abs(r_ad - (u_t - ex.NU * u_xx)) < 1e-5 * max(1.0, abs(r_ad))


def test_example_A_network_parameter_count():
    ex = load_module("chapter01/exA_three_ways_pytorch.py")
    assert ex.n_params(ex.MLP(ex.LAYERS, 0)) == 2241


def test_burgers_network_has_three_hidden_layers_of_48():
    mod = load_module("chapter06/burgers_pinn.py")
    assert mod.LAYERS == [2, 48, 48, 48, 1]
    assert mod.n_params(mod.MLP(mod.LAYERS, 0)) == 4897


def test_deeponet_shapes_and_parameter_count():
    don = load_module("chapter09/darcy/deeponet.py", "deeponet")
    net = don.DeepONet(5, 2, 32, 0)
    assert sum(p.numel() for p in net.parameters()) == 13057
    out = net(torch.zeros(7, 5, dtype=torch.float64), torch.zeros(441, 2, dtype=torch.float64))
    assert tuple(out.shape) == (7, 441)


def test_fno_parameter_count_and_resolution_agnostic_output():
    fno = load_module("chapter09/fno/fno1d.py", "fno1d")
    model = fno.FNO1d(modes=12, width=16, seed=0)
    assert sum(p.numel() for p in model.parameters()) == 13153
    for n in (32, 64, 256):
        assert tuple(model(torch.zeros(3, n, dtype=torch.float64)).shape) == (3, n)


def test_fno_reference_is_exact_heat_semigroup():
    fno = load_module("chapter09/fno/fno1d.py", "fno1d")
    coeffs = np.zeros((1, 8), dtype=complex); coeffs[0, 2] = 1.0     # cos(3x)
    u = fno.exact_propagate(coeffs, 64, fno.T)
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    np.testing.assert_allclose(u[0], np.exp(-fno.NU * 9 * fno.T) * np.cos(3 * x), atol=1e-14)


def test_kalman_update_is_deterministic_for_a_seed():
    kf = load_module("chapter13/kalman_filter_demo.py")
    r1, _ = kf.run(3)
    r2, _ = kf.run(3)
    assert r1 == r2
    assert abs(np.linalg.det(kf.A) - (1 - 2 * kf.ZETA * kf.OMEGA0 * kf.DT + kf.OMEGA0 ** 2 * kf.DT ** 2)) < 1e-15


def test_residual_bound_constant():
    # C = (2 nu pi^2)^(-1/2) in Theorem 7.1's hard-constraint bound, nu = 0.5
    assert abs((2 * 0.5 * np.pi ** 2) ** -0.5 - 1 / np.pi) < 1e-15


def test_jax_network_matches_specification():
    pytest.importorskip("jax")
    ex = load_module("chapter01/exA_three_ways.py")
    import jax
    params = ex.init_params(jax.random.PRNGKey(0), ex.LAYERS)
    assert ex.n_params(params) == 2241
