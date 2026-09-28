"""The stored results contain the numbers printed in the book, at the printed precision.

Each check reads a committed result file (no computation) and compares it with the value quoted in
the book; the tolerance is half a unit in the last printed digit unless stated otherwise. These tests
guard against accidental modification of the stored results and document where each quoted number
comes from.
"""
import numpy as np
import pytest

from conftest import load_json


def close(value, printed, digits):
    """True if value rounds to `printed` with `digits` decimals."""
    return abs(value - printed) <= 0.5 * 10 ** (-digits) + 1e-12


# ---------------------------------------------------------------- Chapter 1
def test_ch01_table_1_3_finite_differences():
    fd = load_json("results/chapter01/ch01-e01_fd.json")
    assert close(fd["rel_l2"] * 1e4, 3.4, 1) and close(fd["rel_linf"] * 1e4, 5.6, 1)


def test_ch01_table_1_5_time_blind():
    tb = load_json("results/chapter01/ch01-f02_timeblind.json")
    assert close(100 * tb["nu_0.5_IC_as_prediction"], 295, 0)
    assert close(100 * tb["nu_0.01_single_mode_IC_as_prediction"], 5.8, 1)


def test_ch01_pytorch_pinn_median_and_range():
    e = [load_json(f"results/chapter01/ch01-e03_pinn_seed{s}_pytorch.json")["rel_l2"] for s in range(5)]
    assert close(100 * np.median(e), 0.65, 2) and close(100 * min(e), 0.22, 2) and close(100 * max(e), 0.69, 2)


# ---------------------------------------------------------------- Chapters 2-5
def test_ch02_lorenz_rate():
    assert close(load_json("results/chapter02/lorenz_sensitivity.json")["fitted_rate_14_to_29"], 0.89, 2)


def test_ch03_rk4_orders():
    orders = load_json("results/chapter03/rk4_order_verification.json")["observed_order"]
    for o, p in zip(orders, (3.90, 3.99, 4.00, 4.00)):
        assert close(o, p, 2)


def test_ch03_monte_carlo_n_var():
    rows = load_json("results/chapter03/monte_carlo_rate_verification.json")["rows"]
    for r, p in zip(rows, (0.130, 0.123, 0.138, 0.126, 0.122)):
        assert close(r["n"] * r["empirical_var"], p, 3)


def test_ch05_tikhonov():
    t = load_json("results/chapter05/tikhonov_discrepancy_verification.json")
    assert close(t["naive_err"], 2027, 0) and close(t["oracle_best_err"], 0.501, 3) and close(t["discrepancy_err"], 0.503, 3)


# ---------------------------------------------------------------- Chapters 6-8
def test_ch06_burgers_errors():
    e = sorted(100 * load_json(f"results/chapter06/ch07-e03_burgers_seed{s}.json")["rel_l2"] for s in range(5))
    for v, p in zip(e, (3.1, 3.2, 4.4, 10.7, 11.6)):
        assert close(v, p, 1)


def test_ch06_hard_constraint_errors():
    # the book lists the four seeds' errors in increasing order (seed 1 is the 3.6% run)
    e = sorted(100 * load_json(f"results/chapter06/ch07-e04_hard_seed{s}.json")["rel_l2"] for s in range(4))
    for v, p in zip(e, (3.6, 16.7, 17.0, 64.1)):
        assert close(v, p, 1)


def test_ch06_small_viscosity_burgers():
    e = sorted(100 * load_json(f"results/chapter06/ch07-f03_burgers_seed{s}.json")["rel_l2"] for s in range(3))
    for v, p in zip(e, (17, 23, 42)):
        assert close(v, p, 0)


def test_ch08_deep_ritz_resampled():
    e = [100 * load_json(f"results/chapter08/deepritz_resample_seed{s}.json")["rel_l2"] for s in range(3)]
    for v, p, d in zip(e, (1.4, 1.7, 0.47), (1, 1, 2)):
        assert close(v, p, d)


def test_ch08_poisson_fd():
    rows = load_json("data/reference/exA/poisson2d_fd_verification.json")["rows"]
    assert close(rows[1]["rel_l2"] * 1e4, 8.0, 1)


# ---------------------------------------------------------------- Chapter 9
def test_ch09_table_9_1_fno():
    res = {r["seed"]: r for r in load_json("results/chapter09/fno/resolution_transfer.json")["results"]}
    for seed, p in zip(range(5), (0.23, 0.24, 0.25, 0.34, 0.27)):
        for n in ("32", "64", "128", "256"):
            assert close(100 * res[seed]["per_resolution"][n]["median_rel_l2"], p, 2)


def test_ch09_darcy_verification_orders():
    cases = load_json("results/chapter09/darcy_verification/manufactured_solution_convergence.json")["cases"]
    finest = {k: cases[k]["corrected"][-1]["order_max"] for k in ("M1", "M2", "M3")}
    assert close(finest["M1"], 2.00, 2) and close(finest["M3"], 2.00, 2) and close(finest["M2"], 1.95, 2)


def test_ch09_deeponet_sweep_medians():
    sw = load_json("results/chapter09/deeponet_sweep_summary.json")
    ns = ("25", "50", "100", "200", "400")
    norm = [100 * np.median(sw["normalized"][n]["medians_by_seed"]) for n in ns]
    raw = [100 * np.median(sw["firstedition_protocol"][n]["medians_by_seed"]) for n in ns]
    for v, p in zip(norm, (0.34, 0.27, 0.20, 0.20, 0.18)):
        assert close(v, p, 2)
    assert close(raw[0], 39, 0)
    for v, p in zip(raw[1:], (7.3, 1.9, 1.8, 1.8)):
        assert close(v, p, 1)


def test_ch09_break_even():
    b = load_json("results/chapter09/break_even_timing.json")
    assert close(b["classical_per_solve_s"] * 1e3, 0.94, 2)
    assert close(b["break_even_queries_deeponet_batched_eval"] / 1e4, 6.7, 1)
    assert b["break_even_queries_deeponet_single_query_eval"] is None


# ---------------------------------------------------------------- Chapters 10-15
def test_ch10_sir_identity_and_forward_check():
    s = load_json("results/chapter10/sir_nonidentifiability_verification.json")
    assert s["algebraic_identity_max"] < 1e-16
    assert close(s["forward_check_rel"] * 1e12, 3, 0)
    assert abs(s["two_observations"]["gamma_recovered"] - 1 / 7) < 1e-8


def test_ch11_sindy_counts():
    levels = load_json("results/chapter11/sindy_noise_sweep.json")["levels"]
    assert [lv["spurious_median"] for lv in levels] == [1, 1, 3, 8, 14, 17]
    assert [round(100 * lv["frac_any_missing"]) for lv in levels] == [0, 0, 2, 4, 12, 26]


def test_ch12_fisher_information_and_contamination():
    d = load_json("results/chapter12/darcy_inversion_dense.json")
    ev = d["fim_sparse15"]["eigenvalues"]
    for v, p in zip(ev, (0.064, 0.089, 0.497, 1.529, 5.175)):
        assert close(v, p, 3)
    assert close(d["fim_sparse15"]["condition_number"], 80, 0)
    assert close(d["classical"]["rmse"], 0.088, 3)
    rm = sorted(v["rmse"] for v in d["surrogates"].values())
    assert close(rm[0], 0.093, 3) and close(rm[1], 0.242, 3)


def test_ch13_kalman_summary():
    k = load_json("results/chapter13/kalman_filter_study.json")
    s = k["summary"]
    assert close(100 * s["err_kalman_position"]["median"], 2.1, 1)
    assert close(100 * s["err_raw_obs"]["median"], 7.1, 1)
    assert close(100 * s["err_model_only"]["median"], 5.3, 1)
    assert round(50 * k["fraction_kalman_better_than_both"]) == 48
    assert round(50 * k["fraction_kalman_velocity_better_than_model_only"]) == 26


def test_ch13_ou_biases():
    rows = {r["dt"]: r for r in load_json("results/chapter13/ou_drift_diffusion.json")["results"]}
    assert close(100 * rows[4.0]["theta_bias_rel"], -57, 0) and close(100 * rows[4.0]["sigma_bias_rel"], -35, 0)
    assert close(100 * rows[0.1]["theta_bias_rel"], -2.7, 1) and close(100 * rows[0.1]["sigma_bias_rel"], -1.2, 1)


def test_ch14_calibration_coverage():
    cov = load_json("results/chapter14/calibration_check_rerun.json")["eval_seed_2999"]["coverage"]
    for q, p in zip(("0.5", "0.68", "0.8", "0.9", "0.95"), (61, 69, 73, 78, 83)):
        assert close(100 * cov[q], p, 0)


def test_ch14_distribution_shift():
    d = load_json("results/chapter14/deeponet_distribution_shift.json")
    ins = [v["in_family"]["median"] for v in d["deeponet_normalized_n200"].values()]
    outs = [v["out_of_family"]["median"] for v in d["deeponet_normalized_n200"].values()]
    assert close(100 * min(ins), 0.15, 2) and close(100 * max(ins), 0.23, 2)
    assert close(100 * min(outs), 1.32, 2) and close(100 * max(outs), 1.35, 2)
    assert close(100 * d["exact_in_family_surrogate_on_out_of_family"]["median"], 1.37, 2)
    assert close(100 * d["mean_predictor"]["out_of_family"]["median"], 0.42, 2)


def test_ch15_seir_identity():
    s = load_json("results/chapter15/seir_nonidentifiability_verification.json")["SEIR"]
    assert s["algebraic_identity_max"] < 1e-16
    assert close(s["forward_check_rel"] * 1e11, 5, 0)
