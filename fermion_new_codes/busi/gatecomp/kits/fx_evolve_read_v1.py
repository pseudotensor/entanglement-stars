#!/usr/bin/env python
"""fx_evolve_read_v1.py (busi, GATECOMP, ReviewA_resview_v1 tests (a), (b), #405): reads fx_evolve_v1.sh's runs (box CPU, small).
(a) model consistency: Ct (view tau=1e-6), Cm (+ mix=plain), Cs (+ site_fields=dense): each one record at T_slice + 1e-6 a against P's
    record at the slice clock, on the record floor (tolcmp_v3 C_LINES; site fields N, v, N_applied, v_applied); Ct and Cm must agree, Cs must
    DIFFER by the sigma-averaging's size (printed beside: P's applied averaged site fields against its point-sampled re-solve at the slice).
(b) bookkeeping: V2 (tau 1) and V4 (tau 0.5) from P's T 1 slice (fixture r_s/a 16, 512 sites): every own clock equal to T_slice + n tau_new exactly, fields rows on the
    clocks, the T 2 slice written at 2, the segment end at 2, the first restarted record's LZ rate s_T, tau_LZ, dE/dT and kept residual
    against the next records (no jump); Richardson at 2: |P - V2| / |V2 - V4| (2^p, p = 1 for the step as the source reads it); |V2 - H| at
    2 against the pre-slice step error |P - H| at 1.
  fx_evolve_read_v1.py FX_DIR TOLCMP FM_DIR
"""
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_common_v1 as GC  # noqa: E402

O, TCP, FMD = sys.argv[1:4]
sp = importlib.util.spec_from_file_location("tolcmp_v3", TCP); TC = importlib.util.module_from_spec(sp); sp.loader.exec_module(TC)
rs = 16.0
L = lambda n: GC.load_run(os.path.join(O, n), "fx")  # noqa: E731
JP, FP = L("P")
RP = GC.recs_by_T(JP, rs); IP = GC.rows_by_T(FP, rs)
SITE = ("N", "v", "N_applied", "v_applied")


def rec_diff(Jx, Fx, tx, Jy, Fy, ty):
    rx = GC.recs_by_T(Jx, rs)[tx]; ry = GC.recs_by_T(Jy, rs)[ty]
    acc = TC.Acc(); TC.cmp_record(rx, ry, acc, tx)
    p = {c: max([v[0] for v in Lq] or [0.0]) for c, Lq in acc.cls.items()}
    ix = GC.rows_by_T(Fx, rs)[tx]; iy = GC.rows_by_T(Fy, rs)[ty]
    f = {k: float(np.max(np.abs(np.asarray(Fx[k][ix], float) - np.asarray(Fy[k][iy], float)))) for k in SITE}
    return {"E_over_E0": p.get("E_over_E0", 0.0), "roots": p.get("roots", 0.0), "depth": p.get("depth", 0.0), "fields": max(f.values()), "fields_by": f,
            "decisions": len(acc.decisions), "conv": (rx.get("converged"), ry.get("converged"))}


def floor_ok(d):
    LN = TC.C_LINES
    return all(d[c] <= LN[c] for c in ("E_over_E0", "roots", "depth", "fields"))


print("== fx_evolve_read_v1: %s; record floor %s" % (O, json.dumps(TC.C_LINES)))
ts = 1.0
print("\n(a) model consistency: one record at T_slice + 1e-6 a against P's record at T/rs %.4f" % ts)
six = GC.load_sixterm(FMD)
pf = GC.point_free(JP, FP, rs, six, ts, ts)
for n, must in (("Ct", "agree"), ("Cm", "agree"), ("Cs", "differ")):
    try:
        Jc, Fc = L(n)
    except Exception as e:
        print("   %s: not readable (%s)" % (n, e)); continue
    own = sorted(GC.recs_by_T(GC.own(Jc), rs))
    if not own:
        print("   %s: no own record" % n); continue
    d = rec_diff(Jc, Fc, own[0], JP, FP, ts)
    ok = floor_ok(d)
    verdict = ("PASS (agrees at the floor)" if ok else "FAIL (does not agree)") if must == "agree" else ("PASS (differs)" if not ok else "VOID (identical at the floor: the option does nothing on this route)")
    print("   %s at T/rs %.9f: E/E0 %.2e roots %.2e depth %.2e site fields %.2e (%s) decisions %d conv %s: %s" % (
        n, own[0], d["E_over_E0"], d["roots"], d["depth"], d["fields"], ", ".join("%s %.1e" % kv for kv in d["fields_by"].items()), d["decisions"], d["conv"], verdict))
if pf:
    q = pf[0]
    print("   the averaging's size at the slice (P, free re-solve): max |N_point - N_applied| %.2e, max |v_point - v_applied| %.2e; min(N-|v|) %.4e averaged vs %.4e point" % (q[7], q[8], q[1], q[3]))
print("\n(b) bookkeeping and Richardson across the restart")
runs = {}
for n, tau in (("V2", 1.0), ("V4", 0.5), ("H", 1.0)):
    try:
        runs[n] = L(n) + (tau,)
    except Exception as e:
        print("   %s: not readable (%s)" % (n, e))
for n in ("V2", "V4"):
    if n not in runs:
        continue
    J, F, tau = runs[n]
    own = [r for r in GC.own(J)["records"] if r.get("E") is not None]
    T = [r["T"] for r in own]; exp = [ts * rs + (i + 1) * tau for i in range(len(T))]
    clocks_ok = T == exp
    rows = GC.rows_by_T(F, rs); rows_ok = all(round(t / rs, 6) in rows for t in T)
    sl = os.path.join(O, n, "states", "orbitals_fx_T0002.00.npz")
    slT = float(np.load(sl)["T"]) / rs if os.path.exists(sl) else None
    endT = T[-1] / rs if T else None
    print("   %s (tau %g): %d own records; clocks exactly T_slice + n tau: %s (first %.6f, last %.6f); fields rows on every clock: %s; slice 2.0 written at %s; segment end at %s" % (
        n, tau, len(T), clocks_ok, T[0] / rs if T else -1, endT or -1, rows_ok, slT, endT))
    cols = []
    for r in own[:6]:
        c = r.get("continuation") or {}
        sT = max([q.get("s_T") or 0.0 for q in c.values()] or [0.0]); tlz = min([q.get("tau_LZ") or np.inf for q in c.values()] or [np.inf])
        cols.append((r["T"] / rs, sT, tlz, r.get("dE_dT"), (r.get("solve_v32") or {}).get("kept_residual")))
    for q in cols:
        print("      T/rs %.5f  s_T max %.3e  tau_LZ min %.3e  dE/dT %s  kept residual %s" % (q[0], q[1], q[2], GC.fmt(q[3], "%.3e"), GC.fmt(q[4], "%.2e")))
    if len(cols) >= 3:
        rest = [q[1] for q in cols[1:]]; med = float(np.median(rest)) if rest else 0.0
        print("      first restarted record's s_T against the next records' median: %.3f x" % ((cols[0][1] / med) if med else float("nan")))
t5 = 2.0
if all(n in runs for n in ("V2", "V4")):
    dPV2 = rec_diff(JP, FP, t5, runs["V2"][0], runs["V2"][1], t5); dV2V4 = rec_diff(runs["V2"][0], runs["V2"][1], t5, runs["V4"][0], runs["V4"][1], t5)
    print("   Richardson at T/rs 2.0: |P - V2| E/E0 %.2e depth %.2e fields %.2e; |V2 - V4| E/E0 %.2e depth %.2e fields %.2e; ratios %s %s %s (2^p = 2 for p = 1)" % (
        dPV2["E_over_E0"], dPV2["depth"], dPV2["fields"], dV2V4["E_over_E0"], dV2V4["depth"], dV2V4["fields"],
        *[GC.fmt(dPV2[c] / dV2V4[c] if dV2V4[c] else None, "%.2f") for c in ("E_over_E0", "depth", "fields")]))
if "H" in runs and "V2" in runs:
    dVH = rec_diff(runs["V2"][0], runs["V2"][1], t5, runs["H"][0], runs["H"][1], t5); dPH = rec_diff(JP, FP, ts, runs["H"][0], runs["H"][1], ts)
    print("   |V2 - H| at 2.0: E/E0 %.2e depth %.2e fields %.2e; the pre-slice step error |P - H| at 1.0: E/E0 %.2e depth %.2e fields %.2e: %s" % (
        dVH["E_over_E0"], dVH["depth"], dVH["fields"], dPH["E_over_E0"], dPH["depth"], dPH["fields"],
        "within (no O(1) restart discrepancy)" if all(dVH[c] <= max(2.0 * dPH[c], TC.C_LINES[c] * 1e-3) for c in ("E_over_E0", "depth", "fields")) else "LARGER than the pre-slice error (read it)"))
