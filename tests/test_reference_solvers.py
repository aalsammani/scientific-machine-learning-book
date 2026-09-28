"""Verification of the classical reference solvers: exact solutions, manufactured solutions and
observed orders of convergence."""
import numpy as np
import pytest

from conftest import load_module

exA = load_module("reference_problems/exA_heat.py")


def test_exact_solution_satisfies_heat_equation():
    x = np.linspace(0, 1, 401)[1:-1]
    t, h = 0.3, 1e-4
    ut = (exA.exact(x, t + h) - exA.exact(x, t - h)) / (2 * h)
    uxx = (exA.exact(x + h, t) - 2 * exA.exact(x, t) + exA.exact(x - h, t)) / h ** 2
    # central-difference truncation error O(h^2) times derivatives of size up to ~(5 pi)^4
    assert np.max(np.abs(ut - exA.NU * uxx)) < 1e-5


def test_exact_solution_boundary_and_initial_values():
    t = np.linspace(0, 1, 11)
    assert np.max(np.abs(exA.exact(0.0, t))) < 1e-15
    assert np.max(np.abs(exA.exact(1.0, t))) < 1e-14
    x = np.linspace(0, 1, 101)
    np.testing.assert_allclose(exA.exact(x, 0.0), np.sin(np.pi * x) + 0.5 * np.sin(5 * np.pi * x), atol=1e-15)


def test_ftcs_rejects_unstable_step():
    with pytest.raises(AssertionError):
        exA.ftcs(21, 10)          # r = nu dt / dx^2 = 0.5 * 0.1 / 0.0025 = 20 > 1/2


def test_ftcs_second_order_in_space():
    errs, hs = [], []
    for nx in (21, 41, 81):
        dx = 1 / (nx - 1)
        nt = int(np.ceil(exA.T / (0.4 * dx ** 2 / exA.NU)))
        x, t, U = exA.ftcs(nx, nt)
        errs.append(exA.rel_errors(U, exA.exact(x[None, :], t[:, None]))[0]); hs.append(dx)
    order = exA.observed_order(hs, errs)
    assert np.all(np.abs(order - 2.0) < 0.1), order


def test_crank_nicolson_manufactured_solution_order_two():
    NU = exA.NU
    um = lambda x, t: np.sin(np.pi * x) * np.cos(3 * t) + x * (1 - x) * t ** 2
    src = lambda x, t: (-3 * np.sin(np.pi * x) * np.sin(3 * t) + 2 * x * (1 - x) * t
                        - NU * (-(np.pi ** 2) * np.sin(np.pi * x) * np.cos(3 * t) - 2 * t ** 2))
    errs, hs = [], []
    for nx in (21, 41, 81):
        dx = 1 / (nx - 1); nt = int(round(exA.T / dx))
        x, t, U = exA.crank_nicolson(nx, nt, u0=lambda x: um(x, 0.0), source=src)
        errs.append(exA.rel_errors(U, um(x[None, :], t[:, None]))[0]); hs.append(dx)
    order = exA.observed_order(hs, errs)
    assert np.all(np.abs(order - 2.0) < 0.1), order


def test_p1_finite_elements_converge_at_second_order():
    fem = load_module("chapter06/fem_1d.py")
    errs, hs = [], []
    for nx in (81, 161, 321):
        h = 1 / (nx - 1); nt = int(round(exA.T / h))
        x, t, U = fem.fem_cn(nx, nt)
        X, Tg = np.meshgrid(x, t, indexing="ij")
        Uex = exA.exact(X, Tg)
        errs.append(np.linalg.norm(U.T - Uex) / np.linalg.norm(Uex)); hs.append(h)
    order = exA.observed_order(hs, errs)
    # the coarse-grid orders are lower (1.23, 1.60 in section 3.8); at these resolutions the rate is near 2
    assert np.all(order > 1.9), order


def test_cole_hopf_matches_method_of_lines():
    refs = load_module("reference_problems/refs_B_to_G.py")
    nu = 0.01 / np.pi
    t_eval = np.array([0.25])
    x, U = refs.burgers_mol(1024, nu, 0.25, t_eval)
    ref = refs.cole_hopf(x, 0.25, nu)
    assert np.linalg.norm(U[0] - ref) / np.linalg.norm(ref) < 1e-3


def test_poisson_five_point_second_order():
    mod = load_module("chapter08/poisson2d_fd_convergence.py")
    (h1, e1), (h2, e2) = mod.solve(15), mod.solve(31)
    assert abs(np.log(e1 / e2) / np.log(h1 / h2) - 2.0) < 0.05
