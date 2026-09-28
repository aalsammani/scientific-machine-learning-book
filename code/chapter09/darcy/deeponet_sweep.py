"""DeepONet data-cost sweep on the Darcy dataset (Chapter 9, section 9.6; Figure 9.3).

Trains deeponet.py for training-set sizes n in {25, 50, 100, 200, 400} and seeds 0-4 under two
protocols (50 trainings): raw outputs (directory sweep_firstedition_protocol, the original
protocol) and standardized outputs (sweep_normalized). Model states are saved for n = 200, because
Chapters 12 and 14 reuse them. One further job retrains the n = 200, seed 0, raw-output model on
the archived legacy dataset, as a check that the training protocol reproduces the originally
reported error. Jobs whose output file already exists are skipped; each job is a single-threaded
process, and jobs run in parallel (accuracy only; timings for the break-even analysis come from
break_even_timing.py). Afterwards the medians are aggregated.

Usage:
    python code/chapter09/darcy/deeponet_sweep.py
Outputs: results/chapter09/sweep_{firstedition_protocol,normalized}/deeponet_n{n}_seed{s}.json,
state_n200_seed{s}.pt, results/chapter09/legacy_first_order_solver/reproduction_n200_seed0.json and
results/chapter09/deeponet_sweep_summary.json.
Runtime: about 1 hour of single-core compute (about 10 minutes with 14 parallel workers).
To recompute the stored sweep, delete (or move) the stored per-run JSON files first; otherwise only
the aggregation is repeated.
"""
import subprocess, sys, os, json, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
R = str(HERE.parents[1] / "results" / "chapter09")
PY = sys.executable
jobs = []
for proto, flag in (("firstedition_protocol", []), ("normalized", ["--normalize"])):
    d = f"{R}/sweep_{proto}"; os.makedirs(d, exist_ok=True)
    for n in (25, 50, 100, 200, 400):
        for s in range(5):
            out = f"{d}/deeponet_n{n}_seed{s}.json"
            state = ["--state", f"{d}/state_n{n}_seed{s}.pt"] if n == 200 else []
            if not os.path.exists(out):
                jobs.append([PY, str(HERE / "deeponet.py"), str(n), str(s), "--out", out] + flag + state)
leg = f"{R}/legacy_first_order_solver/reproduction_n200_seed0.json"
if not os.path.exists(leg):
    jobs.append([PY, str(HERE / "deeponet.py"), "200", "0", "--data", "results/chapter09/legacy_first_order_solver/darcy_dataset.npz", "--out", leg])

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return cmd[-1] if r.returncode == 0 else f"FAILED {cmd}: {r.stderr[-500:]}"

with ThreadPoolExecutor(max_workers=min(14, os.cpu_count() or 1)) as ex:
    for msg in ex.map(run, jobs):
        print(msg, flush=True)

summary = {}
for proto in ("firstedition_protocol", "normalized"):
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/sweep_{proto}/deeponet_n*_seed*.json"))]
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_train_used"], []).append(r)
    summary[proto] = {n: dict(medians_by_seed=[r["rel_l2_median"] for r in sorted(v, key=lambda r: r["seed"])],
                             train_mse_by_seed=[r["train_mse"] for r in sorted(v, key=lambda r: r["seed"])])
                      for n, v in sorted(by_n.items())}
if os.path.exists(leg):
    summary["legacy_reproduction_n200_seed0"] = json.load(open(leg))["rel_l2_median"]
json.dump(summary, open(f"{R}/deeponet_sweep_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
