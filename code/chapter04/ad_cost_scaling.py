"""Cost of reverse-mode automatic differentiation relative to a forward evaluation (Chapter 4,
section 4.6).

  * PyTorch, float64, CPU, torch.manual_seed(0).
  * Network [d, 256, 1] with a tanh hidden layer, d in {10, 1e2, 1e3, 1e4, 1e5}; one input sample
    x ~ N(0, I_d); scalar output.
  * t_fwd: mean wall time of one forward evaluation under torch.no_grad().
  * t_grad: mean wall time of one forward evaluation plus one reverse pass; gradient with respect to
    all network parameters ("params") and, separately, with respect to the input ("input").
  * 10 warm-up and 100 timed repetitions; ratio = t_grad / t_fwd. Each sweep is repeated three
    times, with the default thread count and with one thread ("params_1thread").
The stored original (ad_cost_scaling.json) holds the first run quoted in section 4.6; the repeated
single-thread sweeps in the rerun file give the ranges quoted there.

Usage:
    python code/chapter04/ad_cost_scaling.py
Output: results/chapter04/ad_cost_scaling_rerun.json
Runtime: 1-3 minutes. Timings are hardware dependent and vary between runs; they are not portable.
"""
import json
import platform
import time
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[2] / "results" / "chapter04" / "ad_cost_scaling_rerun.json"
DS = [10, 100, 1000, 10000, 100000]
H = 256
WARM, REPS, N_REPEAT = 10, 100, 3


def time_it(fn):
    for _ in range(WARM):
        fn()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    return (time.perf_counter() - t0) / REPS


def one_sweep(mode):
    rows = []
    for d in DS:
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(d, H), torch.nn.Tanh(), torch.nn.Linear(H, 1)).double()
        x = torch.randn(1, d, dtype=torch.float64)

        def fwd():
            with torch.no_grad():
                net(x)

        if mode == "params":
            def grad():
                net.zero_grad(set_to_none=True)
                net(x).sum().backward()
        else:
            xg = x.clone().requires_grad_(True)
            for p in net.parameters():
                p.requires_grad_(False)

            def grad():
                torch.autograd.grad(net(xg).sum(), xg)

        tf = time_it(fwd)
        tg = time_it(grad)
        rows.append({"d": d, "t_fwd": tf, "t_grad": tg, "ratio": tg / tf})
    return rows


def main():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    res = {"torch": torch.__version__, "machine": platform.processor() or platform.machine(),
           "threads": torch.get_num_threads(), "warmup": WARM, "reps": REPS, "sweeps": {}}
    for mode in ["params", "input"]:
        res["sweeps"][mode] = [one_sweep(mode) for _ in range(N_REPEAT)]
        for r, rows in enumerate(res["sweeps"][mode]):
            print(mode, "repeat", r, " ".join("d=%d:%.2f" % (row["d"], row["ratio"]) for row in rows))
    # single-thread sweep (the stored original run used a single-CPU machine)
    torch.set_num_threads(1)
    res["sweeps"]["params_1thread"] = [one_sweep("params") for _ in range(N_REPEAT)]
    for r, rows in enumerate(res["sweeps"]["params_1thread"]):
        print("params_1thread repeat", r, " ".join("d=%d:%.2f" % (row["d"], row["ratio"]) for row in rows))
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
