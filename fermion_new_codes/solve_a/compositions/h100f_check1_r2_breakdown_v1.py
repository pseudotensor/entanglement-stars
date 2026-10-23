#!/usr/bin/env python
"""h100f_check1_r2_breakdown_v1.py (solve 8a7b5274): Check 1's h100-f v3 records against R2's stored records at 60.0625 and 60.125, leaf by leaf,
split into bookkeeping leaves (counters, timings, caches, iteration histories) and physics numbers; named quantities and fields rows relative."""
import json, numpy as np
W = "<internal>/comp1"; TAG = "C32_s4_ell4_tau2_twofield_v3_off"
SKIP = ("seconds", "timing", "slice_saved", "light_slice_saved", "engine_v34", "lag_s", "jvp_v35")
BOOK = ("kernel_cache", "iteration_history", "nfev", "_s", "iterations", "eigendecompositions", "solver.", "solve_v32", "cholesky_calls", "hits", "misses", "label")
def leaves(o, p=(), out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            if any(s in str(k) for s in SKIP): continue
            leaves(v, p + (str(k),), out)
    elif isinstance(o, list):
        for i, v in enumerate(o): leaves(v, p + (str(i),), out)
    else: out[".".join(p)] = o
    return out
J3 = json.load(open(W + "/out/check1_v3/collapse_%s.json" % TAG)); JR = json.load(open(W + "/in/R2run/collapse_%s.json" % TAG))
for t in (60.0625, 60.125):
    a = leaves(next(r for r in J3["records"] if abs(r["T_over_rs"] - t) < 1e-9)); b = leaves(next(r for r in JR["records"] if abs(r["T_over_rs"] - t) < 1e-9))
    phys, book = [], []
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            if x != y: book.append((k, None, x, y))
            continue
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300) if x != y else 0.0
        (book if any(s in k for s in BOOK) else phys).append((k, rel, x, y))
    r = np.array([q[1] for q in phys]); nz = r[r > 0]
    print("T/rs %g: physics numbers %d, bitwise equal %d; relative difference: median %.2e, 90th %.2e, 99th %.2e, max %.2e" % (t, len(r), int((r == 0).sum()), np.median(r), np.percentile(r, 90), np.percentile(r, 99), r.max()))
    print("  physics leaves with rel > 1e-8:")
    for k, rel, x, y in sorted(phys, key=lambda q: -q[1])[:25]:
        if rel > 1e-8: print("    %-70s %.3e  (%r vs %r)" % (k[:70], rel, x, y))
    print("  bookkeeping leaves differing: %d, e.g. %s" % (sum(1 for q in book if q[1] is None or q[1] > 0), [q[0] for q in book if q[1] is None or q[1] > 0][:12]))
    for n in ("E", "E_over_E0", "H_in", "s11.ball_beyond_16.energy_density_route", "s11.ball_beyond_24.energy_density_route", "light_v36.H_in_light", "light_v36.E_inv_heavy", "M9.collar.I_E_cut"):
        if n in a and n in b and isinstance(a[n], float): print("    %-45s %.17g vs %.17g  rel %.2e" % (n, a[n], b[n], abs(a[n] - b[n]) / max(abs(b[n]), 1e-300)))
F3 = np.load(W + "/out/check1_v3/collapse_%s_fields.npz" % TAG, allow_pickle=True); FR = np.load(W + "/in/R2run/collapse_%s_fields.npz" % TAG, allow_pickle=True)
T3, TR = np.asarray(F3["T"], float), np.asarray(FR["T"], float)
for t in (60.0625, 60.125):
    i3 = int(np.argmin(np.abs(T3 - 32 * t))); iR = int(np.argmin(np.abs(TR - 32 * t)))
    s = ["%s %.1e" % (k, float(np.max(np.abs(F3[k][i3] - FR[k][iR])) / max(np.max(np.abs(FR[k][iR])), 1e-300))) for k in F3.files if F3[k].dtype != object and F3[k].ndim == 2 and F3[k].shape[0] == len(T3)]
    print("fields rows at %g, max|d|/max|x|: %s" % (t, "; ".join(s)))
