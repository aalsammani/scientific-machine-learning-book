"""Aliasing demonstration (Chapter 3, section 3.5).

  * Sample sin(12 x) at N = 16 equally spaced points x_j = 2 pi j / N on [0, 2 pi).
  * Compute the discrete Fourier transform and report the dominant wavenumber |k| (rfft, 0..N/2)
    and the fraction of energy there.
  * Predicted alias of k = 12 on 16 points: |12 - 16| = 4 (fold across the Nyquist wavenumber 8).
  * Check the identity sin(12 x_j) = -sin(4 x_j) on the samples.

Usage:
    python code/chapter03/aliasing_demo.py
Output: results/chapter03/aliasing_demo_rerun.json (aliasing_demo.json is the stored original).
Runtime: under a second. Deterministic.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "chapter03" / "aliasing_demo_rerun.json"


def main():
    N, k_true = 16, 12
    x = 2 * np.pi * np.arange(N) / N
    u = np.sin(k_true * x)
    U = np.fft.rfft(u)
    energy = np.abs(U) ** 2
    dom = int(np.argmax(energy))
    Ufull = np.fft.fft(u)
    full_nonzero_bins = [int(i) for i in np.nonzero(np.abs(Ufull) > 1e-9 * np.max(np.abs(Ufull)))[0]]
    kfold = k_true % N
    pred = min(kfold, N - kfold)
    out = {"N": N, "k_true": k_true, "nyquist": N // 2,
           "dominant_detected_mode": dom,
           "energy_fraction_at_dominant": float(energy[dom] / energy.sum()),
           "predicted_alias": pred,
           "full_fft_nonzero_bins": full_nonzero_bins,
           "full_fft_signed_wavenumbers": [int(b) if b <= N // 2 else int(b - N) for b in full_nonzero_bins],
           "max_abs_sin12_plus_sin4_on_grid": float(np.max(np.abs(u + np.sin(4 * x))))}
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
