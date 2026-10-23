#!/usr/bin/env python
"""selftest_read_v1.py (busi, analyst; CVCONFIRM): the field injection self-test's checks (b) to (d) on the box fixture of
selftest_cvconfirm_v1.sh ((a), A1 against A1P, is cmp_prefix_v1.py's). Reads only.
  (b) A2's record at T_inject: N_applied, v_applied equal the source's stored row at the mapped sites bitwise; its field_inject entry;
      A2's records before T_inject equal A1's (every fields-npz row; json records by cmp_rd13m_v1.diff's rule);
  (c) A2s (self-injection of A1P's own row): every fields, spectra and handoff npz row after the slice equals A1's, and the json records
      equal A1's by cmp_rd13m_v1.diff's rule with the keys field_inject and eigendecompositions (the injected evaluation's 10 extra
      eigendecompositions) left out and the counters of its one extra field solve (fields_rd2 since_previous_record, v36 kernel_cache)
      listed apart;
  (d) the refusals: each R_* log ends in its field inject refusal and wrote no run json.
    python selftest_read_v1.py SELFTEST_DIR"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmp_rd13m_v1 import diff

O = sys.argv[1]; RS = 16.0; TI = 0.1875; TS = 0.0625


def load(n):
    J = json.load(open(os.path.join(O, n, "collapse_fx.json")))
    Z = {s: np.load(os.path.join(O, n, "collapse_fx%s.npz" % s), allow_pickle=True) for s in ("_fields", "_spectra", "_handoff")}
    return J, Z


def rows_equal(Za, Zb, t_lo, t_hi, skip=()):   # the clocks t_lo < t <= t_hi
    out = []
    for s in Za:
        a, b = Za[s], Zb[s]
        Ta = np.round(np.asarray(a["T"], float) / RS, 6); Tb = np.round(np.asarray(b["T"], float) / RS, 6)
        for t in sorted(set(Ta) & set(Tb)):
            if not (t_lo + 1e-9 < t <= t_hi + 1e-9):
                continue
            ia, ib = int(np.where(Ta == t)[0][0]), int(np.where(Tb == t)[0][0])
            for k in a.files:
                if k in skip or k not in b.files or a[k].ndim == 0 or a[k].shape[0] != len(Ta):
                    continue
                x, y = a[k][ia], b[k][ib]
                if x.dtype == object:
                    same = np.array_equal(np.asarray(x, float), np.asarray(y, float))
                else:
                    same = np.array_equal(x, y, equal_nan=(x.dtype.kind in "fc"))
                if not same:
                    out.append("%s[%s] T%.4f" % (s, k, t))
    return out


def recs_equal(Ja, Jb, t_lo, t_hi, drop=()):
    ra = {round(r["T"] / RS, 6): r for r in Ja["records"]}; rb = {round(r["T"] / RS, 6): r for r in Jb["records"]}
    out = []
    for t in sorted(set(ra) & set(rb)):
        if t_lo + 1e-9 < t <= t_hi + 1e-9:
            x = {k: v for k, v in ra[t].items() if k not in drop}; y = {k: v for k, v in rb[t].items() if k not in drop}
            diff(x, y, "T%.4f" % t, out)
    return out


res = {}
J1, Z1 = load("A1"); J2, Z2 = load("A2"); Js, Zs = load("A2s"); JS, ZS = load("Su2")
n = len(Z1["_fields"]["R"])
T2 = np.round(np.asarray(Z2["_fields"]["T"], float) / RS, 6); i2 = int(np.where(T2 == TI)[0][0])
TS2 = np.round(np.asarray(ZS["_fields"]["T"], float) / RS, 6); iS = int(np.where(TS2 == TI)[0][0])
eq = {k: bool(np.array_equal(Z2["_fields"][k][i2], np.asarray(ZS["_fields"][k][iS])[:n])) for k in ("N_applied", "v_applied")}
r2 = [r for r in J2["records"] if abs(r["T"] / RS - TI) < 1e-9][0]
ent = r2.get("field_inject")
pre_rows = rows_equal(Z2, Z1, TS, TI - 1e-3); pre_recs = recs_equal(J2, J1, TS, TI - 1e-3)   # the records strictly before T_inject
own_N = float(np.max(np.abs(np.asarray(Z1["_fields"]["N_applied"][i2]) - np.asarray(ZS["_fields"]["N_applied"][iS])[:n])))
res["b"] = {"injected_equals_source_at_mapped_sites": eq, "field_inject": ent, "pre_injection_rows_differing": pre_rows, "pre_injection_record_diffs": pre_recs[:10],
            "A1_minus_source_N_applied_at_T_inject": own_N, "head_field_inject": J2["head"].get("field_inject"),
            "PASS": all(eq.values()) and ent is not None and not pre_rows and not pre_recs and ent["sites_mapped"] == n and ent["source_sites"] == 2 * n}
c_rows = rows_equal(Zs, Z1, TS, 1e9); c_all = recs_equal(Js, J1, TS, 1e9, drop=("field_inject", "eigendecompositions"))
CTR = ("/fields_rd2/since_previous_record/", "/v36/kernel_cache/")   # the field module's and the kernel cache's counters: the injected evaluation's one extra solve
c_recs = [q for q in c_all if not any(s in q for s in CTR)]; c_ctr = [q for q in c_all if any(s in q for s in CTR)]
rs_ = [r for r in Js["records"] if abs(r["T"] / RS - TI) < 1e-9][0].get("field_inject") or {}
res["c"] = {"rows_differing": c_rows[:20], "n_rows_differing": len(c_rows), "record_diffs": c_recs[:20], "n_record_diffs": len(c_recs), "counter_diffs_allowed": c_ctr,
            "self_injection_entry": {k: rs_.get(k) for k in ("injected_minus_own_N_applied", "injected_minus_own_v_applied", "injected_residual_N", "own_residual_N", "solve_minus_source_N")},
            "PASS": not c_rows and not c_recs and rs_.get("injected_minus_own_N_applied") == 0.0}
d = {}
for nm, want in (("R_offclock", "holds 0 rows at T/rs 0.190000"), ("R_atslice", "is not after the slice's clock"), ("R_shifted", "does not carry this chain's 256 radii"),
                 ("R_half", "--inject-fields and --inject-T-over-rs go together")):
    log = open(os.path.join(O, nm, "log.txt")).read()
    d[nm] = {"refused_as_expected": want in log, "no_run_json": not os.path.exists(os.path.join(O, nm, "collapse_fx.json")),
             "last_line": [l for l in log.splitlines() if l.strip()][-1][:200]}
res["d"] = dict(d, PASS=all(q["refused_as_expected"] and q["no_run_json"] for q in d.values()))
print(json.dumps(res, indent=1, default=float))
print("SELFTEST (b) injected record = source at the mapped sites, pre-injection records = A1: %s" % ("PASS" if res["b"]["PASS"] else "FAIL"))
print("SELFTEST (c) self-injection reproduces A1 (npz rows bitwise; json records but field_inject and eigendecompositions): %s" % ("PASS" if res["c"]["PASS"] else "FAIL"))
print("SELFTEST (d) refusals at start: %s" % ("PASS" if res["d"]["PASS"] else "FAIL"))
with open(os.path.join(O, "selftest_read_v1.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=float)
sys.exit(0 if all(res[k]["PASS"] for k in "bcd") else 1)
