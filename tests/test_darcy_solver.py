"""Darcy solver (Running Example D): manufactured-solution order, symmetry, the legacy boundary
defect, and bitwise agreement with the stored dataset."""
import numpy as np

from conftest import RESULTS, load_module

darcy = load_module("chapter09/darcy/darcy_data.py", "darcy_data")


def _mms_errors(boundary):
    # u = sin(pi x) sin(2 pi y), a = 1: f = 5 pi^2 u
    errs, hs = [], []
    for n in (11, 23, 47):
        h, X, Y = darcy.grid(n)
        u = darcy.solve_field(np.ones_like(X), 5 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(2 * np.pi * Y), h, boundary)
        errs.append(np.abs(u - np.sin(np.pi * X) * np.sin(2 * np.pi * Y)).max()); hs.append(h)
    return np.log(np.array(errs[:-1]) / errs[1:]) / np.log(np.array(hs[:-1]) / hs[1:])


def test_corrected_boundary_is_second_order():
    assert np.all(np.abs(_mms_errors("corrected") - 2.0) < 0.1)


def test_legacy_boundary_is_first_order():
    # Chapter 14, section 14.4, Instance 1: the legacy boundary conductance reduces the order to one
    assert np.all(np.abs(_mms_errors("legacy") - 1.0) < 0.2)


def test_matrix_is_symmetric_positive_definite():
    h, X, Y = darcy.grid(9)
    a = darcy.make_a(np.array([0.5, -0.3, 0.8, 0.1, -0.9]), X, Y)
    A = darcy.assemble(a, h).toarray()
    np.testing.assert_allclose(A, A.T, atol=1e-10 * np.abs(A).max())
    assert np.linalg.eigvalsh(A).min() > 0


def test_coefficient_family_positive():
    rng = np.random.default_rng(1)
    h, X, Y = darcy.grid(21)
    for c in rng.uniform(-1, 1, (50, 5)):
        assert darcy.make_a(c, X, Y).min() >= 1 - 0.4 - 1e-12


def test_stored_dataset_reproduced():
    d = np.load(RESULTS / "chapter09" / "darcy_dataset.npz")
    assert str(d["boundary"]) == "corrected"
    coeffs = np.random.default_rng(0).uniform(-1, 1, (500, 5))
    np.testing.assert_array_equal(coeffs, d["coeffs"])
    for i in (0, 1, 250, 499):
        _, _, _, u = darcy.darcy_solve(coeffs[i])
        np.testing.assert_allclose(u, d["U"][i], rtol=1e-12, atol=1e-15)
