#!/usr/bin/env python
"""cmp_rd2_replay_v1.py (busi, analyst; PERF RD-2/RD-5 self-test (c)): the replay's records against the budget of busi/notes/PERF_RD2_build_v1.md
sec. 1 (c), written before the test. C1: each option arm (C = --fields collocation, W = + --fields-warm-start on) against the default arm I, same
box CPU, same threads, same slice. C2: each arm against the production records (R1's H100 run: the mirror's json, fields and spectra npz).
Per record: every float leaf compared; the gated keys against their thresholds; the set decisions named in the budget required equal
(per channel continuation mode, nsub, verified, inversions, n_intermediate, n_pocket, mask_equals_negative_set; n_pocket and n_refill_R0 per
channel; the trapping-zero and tangency counts; converged; late_surface flag; the srf step refinement's discrete entries; the spectra masks);
every other discrete difference reported (iteration, eigendecomposition and solve counts, nfev, kernel cache counters and the like); the fields
npz rows N, v max |d|. Timing per arm and record: the record wall, the solves (count, seconds, seconds per solve), the eighs.
Box only; reads, never writes the inputs.
usage: cmp_rd2_replay_v1.py --arm I=DIR:TAG --arm C=DIR:TAG [--arm W=DIR:TAG] --rs RS [--prod-json J --prod-dir D --prod-tag T] [--t-min X] --out JSON"""
import argparse
import json
import math
import os
import re

import numpy as np

SKIP_KEYS = {"seconds", "timing", "slice_saved", "label", "record_label", "compute_s", "start", "fields_rd2", "snapshot_seconds", "snapshot_bytes"}
TH = {"E_over_E0": 1e-9, "root": 3.4e-7, "min_N_minus_absv": 4.6e-9, "fields": 1e-7}
FLOOR = {"E_over_E0": 3.0e-7, "root": 3.4e-6, "min_N_minus_absv": 4.6e-8, "fields": 1e-5}
ROOTS = {"R_t", "r_in", "R_c"}   # the notes' list (trapping zeros, R_t, r_in, R_c, threshold radii); r_min_N_minus_absv (a minimum's location) is reported
CONT_SET = {"mode", "nsub", "verified", "inversions", "n_intermediate", "n_pocket", "mask_equals_negative_set"}


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in SKIP_KEYS or k.endswith("_s"):
                continue
            yield from leaves(v, p + "/" + str(k))
    elif isinstance(x, list):
        yield p + "#len", len(x)
        for i, v in enumerate(x):
            yield from leaves(v, p + "[%d]" % i)
    else:
        yield p, x


def is_set_decision(k):
    m = re.match(r"^/continuation/[^/]+/([^/\[#]+)$", k)
    if m:
        return m.group(1) in CONT_SET
    if re.match(r"^/(n_pocket|n_refill_R0)/[^/]+$", k):
        return True
    if k in ("/trapping_zeros#len", "/tangencies#len", "/converged", "/late_surface/flag"):
        return True
    if k.startswith("/srf/"):
        return True
    return False


def cmp_record(a, b):
    La, Lb = dict(leaves(a)), dict(leaves(b))
    fl = {}; disc = []
    for k in sorted(set(La) | set(Lb)):
        x, y = La.get(k, "<absent>"), Lb.get(k, "<absent>")
        if isinstance(x, bool) or isinstance(y, bool) or isinstance(x, str) or isinstance(y, str) or x is None or y is None:
            if x != y:
                disc.append((k, x, y))
        elif isinstance(x, int) and isinstance(y, int):
            if x != y:
                disc.append((k, x, y))
        else:
            x, y = float(x), float(y)
            if math.isnan(x) and math.isnan(y):
                continue
            d = abs(x - y)
            fl[k] = (d, d / max(abs(x), abs(y), 1e-300))
    return fl, disc


def gate(fl, disc, th, srf_numbers_set=True):
    out = []; ok = True
    e = fl.get("/E_over_E0", (0.0, 0.0))[0]
    ok &= e <= th["E_over_E0"]; out.append("E/E0 |d| %.2e (<= %.1e)" % (e, th["E_over_E0"]))
    rd = max([fl[k][0] for k in fl if k.lstrip("/") in ROOTS or k.startswith("/trapping_zeros[") or k.startswith("/threshold_radius/")] or [0.0])
    ok &= rd <= th["root"]; out.append("roots |d| max %.2e (<= %.1e)" % (rd, th["root"]))
    m = fl.get("/min_N_minus_absv", (0.0, 0.0))[0]
    ok &= m <= th["min_N_minus_absv"]; out.append("min(N-|v|) |d| %.2e (<= %.1e)" % (m, th["min_N_minus_absv"]))
    rsh = fl.get("/R_sh", (0.0, 0.0))[0]
    ok &= rsh == 0.0; out.append("R_sh |d| %.1e (== 0)" % rsh)
    out.append("r_min_N_minus_absv |d| %.2e (the minimum's location; reported, not gated)" % fl.get("/r_min_N_minus_absv", (0.0, 0.0))[0])
    sets = [(k, x, y) for k, x, y in disc if is_set_decision(k)]
    other = [(k, x, y) for k, x, y in disc if not is_set_decision(k)]
    # srf floats that are decisions (tau_in/tau_out, grid clock) must be equal as well
    srf_f = [(k, v) for k, v in fl.items() if k.startswith("/srf/") and v[0] != 0.0 and any(s in k for s in ("tau_in", "tau_out", "grid_T_over_rs", "T_over_rs"))]
    ok &= not sets and not srf_f
    out.append("set decisions: %d differences%s%s" % (len(sets) + len(srf_f), (" " + "; ".join("%s: %r vs %r" % s for s in sets[:6])) if sets else "",
                                                     (" srf clocks " + "; ".join("%s %.1e" % (k, v[0]) for k, v in srf_f[:4])) if srf_f else ""))
    out.append("other discrete differences (reported, not gated): %d%s" % (len(other), (" " + "; ".join("%s: %r vs %r" % (k, x, y) for k, x, y in other[:8])) if other else ""))
    big = sorted(((v[1], k, v[0]) for k, v in fl.items()), reverse=True)[:3]
    out.append("largest relative float differences (reported): " + "; ".join("%s %.1e (abs %.1e)" % (k, r, d) for r, k, d in big))
    return ok, out


def rows_by_T(Z, rs):
    return {round(float(t) / rs, 6): i for i, t in enumerate(np.asarray(Z["T"], float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR:TAG (the first is the reference arm)")
    ap.add_argument("--rs", type=float, required=True)
    ap.add_argument("--prod-json", default=None); ap.add_argument("--prod-dir", default=None); ap.add_argument("--prod-tag", default=None)
    ap.add_argument("--t-min", type=float, default=-1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arms = []
    for s in a.arm:
        n, rest = s.split("=", 1); d, t = rest.rsplit(":", 1); arms.append((n, d, t))
    ref = arms[0][0]
    J = {}; Z = {}; T = {}
    for n, d, t in arms:
        jj = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
        J[n] = {round(r["T_over_rs"], 6): r for r in jj["records"] if r["T_over_rs"] > a.t_min + 1e-9}
        Z[n] = {k: np.load(os.path.join(d, "collapse_%s_%s.npz" % (t, k)), allow_pickle=True) for k in ("fields", "spectra")}
    if a.prod_json:
        P = json.load(open(a.prod_json))["records"]
        J["prod"] = {round(r["T_over_rs"], 6): r for r in P}
        Z["prod"] = {k: np.load(os.path.join(a.prod_dir, "collapse_%s_%s.npz" % (a.prod_tag, k)), allow_pickle=True) for k in ("fields", "spectra")}
    ts = sorted(set.intersection(*[set(J[n]) for n, _, _ in arms]))
    rep = {"records": {}, "timing": {}, "verdict": {}}
    print("records in every arm: %s; record sets equal: %s" % (ts, all(sorted(J[n]) == sorted(J[ref]) for n, _, _ in arms)))
    verdict = {n: (True if n != ref else None) for n, _, _ in arms}
    verdict_floor = {n: True for n, _, _ in arms}
    if not all(sorted(J[n]) == sorted(J[ref]) for n, _, _ in arms):
        for n, _, _ in arms[1:]:
            if sorted(J[n]) != sorted(J[ref]):
                verdict[n] = False
    for t in ts:
        print("== T/rs %.6f" % t)
        ent = {}
        pairs = [("C1 %s vs %s" % (n, ref), n, ref, TH) for n, _, _ in arms[1:]]
        if a.prod_json:
            pairs += [("C2 %s vs prod" % n, n, "prod", FLOOR) for n, _, _ in arms]
        for name, x, y, th in pairs:
            if t not in J[y]:
                print("  %s: no record" % name); continue
            fl, disc = cmp_record(J[x][t], J[y][t])
            ok, lines = gate(fl, disc, th)
            ia, ib = rows_by_T(Z[x]["fields"], a.rs), rows_by_T(Z[y]["fields"], a.rs)
            if t in ia and t in ib:
                fN = float(np.max(np.abs(Z[x]["fields"]["N"][ia[t]].astype(float) - Z[y]["fields"]["N"][ib[t]].astype(float))))
                fv = float(np.max(np.abs(Z[x]["fields"]["v"][ia[t]].astype(float) - Z[y]["fields"]["v"][ib[t]].astype(float))))
                okf = max(fN, fv) <= th["fields"]
                sa, sb = rows_by_T(Z[x]["spectra"], a.rs), rows_by_T(Z[y]["spectra"], a.rs)
                ks = [k for k in Z[x]["spectra"].files if k.startswith("mask_")]
                mask_ok = all(np.array_equal(Z[x]["spectra"][k][sa[t]], Z[y]["spectra"][k][sb[t]]) for k in ks)
                emax = max(float(np.max(np.abs(Z[x]["spectra"][k.replace("mask", "e")][sa[t]] - Z[y]["spectra"][k.replace("mask", "e")][sb[t]]))) for k in ks)
                lines.append("fields max |dN| %.2e |dv| %.2e (<= %.0e); spectra masks %s, e max |d| %.1e" % (fN, fv, th["fields"], "equal" if mask_ok else "DIFFER", emax))
                ok &= okf and mask_ok
            else:
                lines.append("fields/spectra row missing"); ok = False
            print("  %s: %s" % (name, "PASS" if ok else "FAIL"))
            for l_ in lines:
                print("     " + l_)
            ent[name] = {"ok": bool(ok), "lines": lines}
            if name.startswith("C1"):
                verdict[x] = verdict[x] and bool(ok)
            else:
                verdict_floor[x] = verdict_floor[x] and bool(ok)
        rep["records"]["%.6f" % t] = ent
    for n, _, _ in arms:
        rows = []
        for t in ts:
            r = J[n][t]; tm = r.get("timing") or {}
            sn = tm.get("solve_n") or 0; ss = tm.get("solve_s") or 0.0
            rows.append({"T_over_rs": t, "seconds": r.get("seconds"), "solve_n": sn, "solve_s": ss, "per_solve_s": (ss / sn) if sn else None,
                         "eigh_n": tm.get("eigh_n"), "eigh_s": tm.get("eigh_s"), "iterations": r.get("iterations"), "fields_rd2": r.get("fields_rd2")})
        rep["timing"][n] = rows
        tot = sum(x["seconds"] or 0 for x in rows); sn = sum(x["solve_n"] for x in rows); ss = sum(x["solve_s"] for x in rows)
        print("timing %s: records %s s (total %.0f s); solves %d in %.1f s (%.3f s per solve); eighs %s in %s s; iterations %s" % (
            n, ", ".join("%.0f" % (x["seconds"] or 0) for x in rows), tot, sn, ss, ss / sn if sn else float("nan"),
            [x["eigh_n"] for x in rows], ["%.0f" % (x["eigh_s"] or 0) for x in rows], [x["iterations"] for x in rows]))
    rep["verdict"] = {"C1": {n: ("PASS" if v else "FAIL") for n, v in verdict.items() if v is not None}, "C2": {n: ("PASS" if v else "FAIL") for n, v in verdict_floor.items()} if a.prod_json else None}
    print("VERDICT over %d records: C1 %s; C2 %s" % (len(ts), rep["verdict"]["C1"], rep["verdict"]["C2"]))
    json.dump(rep, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
