#!/usr/bin/env python3
"""obs_plateau_mix_v5.py (builder analyst; the mixing term of observables 13.1's gate, version 5 of obs_plateau_mix_v4.py): the junction's
branch-to-long-wave amplitude |t| measured on the slice itself, at the reading's own geometry, frequency and mass.

region_mixing(fn, R_a, R_b, w_a, w_b, omega, m): the region chain of version 4 (the inner junction at R_a with PL-ext's log blend of width w_a,
the outer junction inside the surface at R_b with width w_b) built on fn (the slice's dense functions, rho -> (N, v)), its scattering matrix
at (omega, m) (obs_plateau_v4.scatter), the inward branch channels classified (obs_plateau_mix_v2.classify), and the largest amplitudes from a
branch channel into the long-wave channel (|t|) and into the other branch, the pass-through minimum, the full unitarity. The gate's
contamination is formed by obs_plateau_v7 from |t| and the reading row's own beta, p and c (13.1 (1)): |delta c|/|c| <= |t|/((kappa' a)^2 |c|
pi r |beta| (1 - p)), acceptance 2 percent. The command line runs it on a slice for a list of geometries (testing). Nothing else."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import math
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obs_plateau_mix_v2 as M2  # noqa: E402
import obs_plateau_mix_v4 as M4  # noqa: E402
import obs_plateau_v4 as P  # noqa: E402


def region_mixing(fn, R_a, R_b, wa, wb, omega, m):
    R, v, N = M4.region_chain2(fn, R_a, R_b, wa, wb)
    sc = P.scatter(v, N, m, 0.0, omega)
    S = sc["S"]
    full = None
    if S.shape[0] == S.shape[1]:
        I = np.eye(S.shape[0])
        full = (float(np.max(np.abs(S.conj().T @ S - I))), float(np.max(np.abs(S @ S.conj().T - I))))
    lab_in = [(side, M2.classify(md, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1]), md["k"]) for side, md in sc["chans"]]
    lab_out = [(side, M2.classify(mo, N[0] if side == "L" else N[-1], v[0] if side == "L" else v[-1]), mo["k"]) for side, i, mo in sc["outs"]]
    cross = lw = 0.0
    thru = []
    for c, (si, li, ki) in enumerate(lab_in):
        if si != "L" or not li.startswith("branch"):
            continue
        for o, (so, lo, ko) in enumerate(lab_out):
            amp = abs(S[o, c])
            if so == "R" and lo == li:
                thru.append(amp)
            elif lo.startswith("branch") and lo != li:
                cross = max(cross, amp)
            elif lo == "longwave":
                lw = max(lw, amp)
    sel = (R >= R_a) & (R <= R_b)
    return {"branch_to_longwave_max": lw, "cross_branch_max": cross, "pass_through_min": (min(thru) if thru else None), "unitarity_full": full,
            "unitarity_columns": sc["unitarity"], "N_range": [float(N[sel].min()), float(N[sel].max())] if sel.any() else None,
            "v_a": float(v[0]), "dk_a": math.acos(min(1.0, float(N[0]) / abs(float(v[0])))) if v[0] != 0 else None}


def contamination(t, kappa_prime, c_abs, r, beta2):
    """13.1 (1): |delta c|/|c| <= |t|/((kappa' a)^2 |c| pi r |beta| (1 - p)), p = |beta|^2 (the Fermi Jacobian)."""
    if beta2 is None or beta2 <= 0 or beta2 >= 1 or c_abs <= 0:
        return None
    return t / (kappa_prime ** 2 * c_abs * math.pi * r * math.sqrt(beta2) * (1.0 - beta2))
