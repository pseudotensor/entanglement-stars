#!/usr/bin/env python
"""ec_common.py (busi lane, cold check of EVOLVEDCUT_route_compare_v1 and EVOLVEDCUT_error_statement_v1): the paths on the box,
the readers and the entropy function shared by ec_arith.py, ec_spectra.py and ec_recompute.py. Own code throughout; nothing is
imported from either route's modules or from the comparison's scripts. Every input is opened read-only."""
import hashlib
import json
import math
import os

import numpy as np

OUT = "<internal>/evolvedcut_v1"
R1_DIR = "<internal>/evolvedcut"
R2_DIR = "<internal>/EVOLVEDCUT_secondroute_v1"
R2_RESULTS = R2_DIR + "/results.json"
R2_TABLES = R2_DIR + "/tables.json"
R2_DIAG = R2_DIR + "/error_statement/matched_purity_diagnostics.json"
SLICES = "<internal>/c19_rs64_A_R2_tau8_ell4_avg_r11"
SLICE_FMT = SLICES + "/orbitals_c19_rs64_A_R2_tau8_ell4_avg_r11_T%07.2f.npz"
CLOCKS = [12, 15, 16, 20, 24, 25, 28, 30, 35, 40]
S8_CLOCKS = [12, 13, 14, 16, 20, 25, 30, 40]
KAPPAS = list(range(1, 11))
TOL_ID = 2.1e-10          # the comparison's per-channel allowance (route 1's stated purity gap / identity defect bound)
W_SUM = 880.0             # sum of the weights 16 kappa, kappa 1..10
TOL_W = W_SUM * TOL_ID    # 1.848e-7, printed 1.85e-7 by the comparison


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def jload(path):
    with open(path) as f:
        return json.load(f)


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, allow_nan=True)


def r1_path(T, hist="prod"):
    return R1_DIR + "/evolvedcut_%s_T%07.2f.json" % (hist, T)


def h2(x):
    """binary entropy in nats of one occupation, clipped to [0, 1] (the convention both routes state)."""
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -(x * math.log(x) + (1.0 - x) * math.log1p(-x))


def entropy(ev):
    """sum of h2 over an occupation spectrum, compensated summation."""
    return math.fsum(h2(x) for x in np.asarray(ev, dtype=float).ravel())


def entropy_split(ev, near=1e-9):
    """the entropy split into the part from occupations within `near` of 0 or 1 (a numerical cluster if the ideal values are
    exactly 0 or 1) and the rest; returns (S_total, S_near, S_rest, n_near, max |deviation| in the near set)."""
    ev = np.asarray(ev, dtype=float).ravel()
    dev = np.minimum(np.abs(ev), np.abs(1.0 - ev))
    near_m = dev < near
    s_near = math.fsum(h2(x) for x in ev[near_m])
    s_rest = math.fsum(h2(x) for x in ev[~near_m])
    return s_near + s_rest, s_near, s_rest, int(near_m.sum()), (float(dev[near_m].max()) if near_m.any() else 0.0)


def r2_rows_index(results):
    """route 2 rows keyed by (clock, kappa)."""
    return {(int(round(r["clock_rs"])), int(r["kappa"])): r for r in results["rows"]}


def r2_cut(row, n_in):
    """the saved partition of a route 2 row with n_in interior sites (None if not saved)."""
    for c in row["cuts"]:
        if int(c["interior_sites"]) == int(n_in):
            return c
    return None
