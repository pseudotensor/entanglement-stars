#!/usr/bin/env python
"""compare_s11_segment_v2.py (the lane's analyst builder; engine spec 40.1 (1), (2)): compare_s11_segment_v1.py with (1) the resume floor
printed on every segment line (E/E0 1.5e-07, the roots 1.7e-06 lattice units, min(N - |v|) 2.3e-08, G2's measured resolution) and a difference
within it read "no difference"; the verdict 39.4 (2)'s criterion on the roots with E/E0, min(N - |v|) and the masks beside; (2) --mode routes:
two certified runs against each other (--run the eigenvalue route, --ref the full route; or the reverse), the 40.1 (2) verdict: if they agree
within the floor on every compared record (E/E0, the outer and inner roots, min(N - |v|)), the masks equal on every channel, and every channel's
Weyl certificate passes on every sub-step in both runs (a failing channel marked), the eigenvalue route is the certified construction of any
further certified run and the full route its oracle; if not, the full route alone is certified and the eigenvalue route a record. The label line
names the routes. Everything else is v1's. v1's docstring follows.
compare_s11_segment_v1.py (builder analyst, the HORIZON version 2 builder; engine spec 39.4 (2)): a v34 run restarted from a stored
slice of the S11 run, against that run's own records at the same clocks.
--mode gate: the resume gate (the restart with every v34 continuation option off must reproduce the stored records): per record, every json
    number compared (the wall-clock keys, the labels and strings, the record's M9 diagnostic (it restarts with the run), slice_saved,
    engine_v34, the iteration counters' timing excluded), and every fields npz row at the run's clocks (N, v, eps_tot, J_tot, eps_pairs,
    J_pairs, N_applied, v_applied) and every spectra row (each channel's mask bits and eigenvalues); prints the counts, the largest absolute
    difference, the differing keys and PASS or FAIL (PASS: every compared number and array element bitwise equal).
--mode segment: the certified segment against the LZ run, per record: E/E0, the trapping zeros (the roots), min(N - |v|), the masks (each
    channel's stored mask bits equal, and mask = negative set), the continuation's verification per channel (the certified run's certificate:
    the lead's count, pieces, refinements, failing pieces, margin) and the oracle's comparison; the lead's criterion printed: whether the roots
    agree within the LZ run's own motion between records (the outer root's change from the record before, at each compared record with a
    root in both); writes an md and a json.
Every number is printed by the script. Usage: compare_s11_segment_v2.py --mode gate|segment|routes --run DIR/TAG --ref DIR/TAG [--md --out-json]."""
import argparse
import json
import os
import sys

import numpy as np

FLOOR = {"E_over_E0": 1.5e-07, "roots": 1.7e-06, "min_N_minus_absv": 2.3e-08}
SKIP = {"seconds", "timing", "label", "record_label", "slice_saved", "engine_v34", "M9", "continuation_label", "seconds_preparation"}
FIELD_KEYS = ("N", "v", "eps_tot", "J_tot", "eps_pairs", "J_pairs", "N_applied", "v_applied")


def load(p):
    d, t = os.path.split(p)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
    F = np.load(os.path.join(d, "collapse_%s_fields.npz" % t), allow_pickle=True)
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % t))
    return J, F, S


def walk(x, y, path, diffs, st):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in SKIP:
                continue
            if k not in x or k not in y:
                diffs.append("%s/%s: present in one only" % (path, k)); continue
            walk(x[k], y[k], "%s/%s" % (path, k), diffs, st)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diffs.append("%s: lengths %d and %d" % (path, len(x), len(y))); return
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, "%s[%d]" % (path, i), diffs, st)
    elif isinstance(x, str) or isinstance(y, str):
        return
    else:
        num = isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool)
        if num:
            st["numbers"] += 1
            if x == x and y == y:
                st["max"] = max(st["max"], abs(float(x) - float(y)))
        same = (x == y) or (isinstance(x, float) and isinstance(y, float) and x != x and y != y)
        if not same:
            diffs.append("%s: %r against %r" % (path, x, y))


def fl(d, key, f):
    if d is None:
        return "-"
    return ("no difference (%s)" % (f % d)) if abs(d) <= FLOOR[key] else (f % d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["gate", "segment", "routes"]); ap.add_argument("--run", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--md", default=None); ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    J, F, S = load(a.run); Jr, Fr, Sr = load(a.ref)
    rs = float(J["args"]["rs"])
    T0 = J["head"]["resume_slice_v34"]["T_over_rs"] * rs
    new = [r for r in J["records"] if r["T"] > T0 + 1e-9]
    byT = {round(r["T"], 6): r for r in Jr["records"]}
    kap = [int(k) for k in range(1, int(J["args"]["kappa_sea"]) + 1)]
    out = {"script": "compare_s11_segment_v2.py", "floor": FLOOR, "mode": a.mode, "run": a.run, "ref": a.ref, "restart_T_over_rs": T0 / rs, "records": []}
    if a.mode == "gate":
        diffs = []; st = {"numbers": 0, "max": 0.0}; nel = 0; fmax = 0.0; fd = []
        for r in new:
            q = byT.get(round(r["T"], 6))
            if q is None:
                diffs.append("T/rs %.4f: no stored record" % (r["T"] / rs)); continue
            walk(r, q, "T%.4f" % (r["T"] / rs), diffs, st)
            i = int(np.argmin(np.abs(np.asarray(F["T"], float) - r["T"]))); j = int(np.argmin(np.abs(np.asarray(Fr["T"], float) - r["T"])))
            for k_ in FIELD_KEYS:
                x = np.asarray(F[k_][i]); y = np.asarray(Fr[k_][j])
                nel += x.size
                if x.shape != y.shape or not np.array_equal(x, y):
                    fd.append("T%.4f/%s" % (r["T"] / rs, k_))
                if x.shape == y.shape and x.size:
                    fmax = max(fmax, float(np.max(np.abs(x - y))))
            i = int(np.argmin(np.abs(np.asarray(S["T"], float) - r["T"]))); j = int(np.argmin(np.abs(np.asarray(Sr["T"], float) - r["T"])))
            for k in kap:
                for nm in ("mask_%d" % k, "e_%d" % k):
                    x = np.asarray(S[nm][i]); y = np.asarray(Sr[nm][j])
                    nel += x.size
                    if not np.array_equal(x, y):
                        fd.append("T%.4f/%s" % (r["T"] / rs, nm))
                    if x.shape == y.shape and x.size and x.dtype.kind == "f":
                        fmax = max(fmax, float(np.max(np.abs(x - y))))
        out.update({"records_compared": len(new), "json_numbers": st["numbers"], "json_differences": len(diffs), "json_max_abs": st["max"],
                    "array_elements": nel, "array_differences": len(fd), "array_max_abs": fmax, "pass": (not diffs and not fd),
                    "json_diff_list": diffs[:60], "array_diff_list": fd[:60]})
        print("resume gate: %d records (T/rs %s); json numbers compared %d, differences %d, the largest absolute difference %.3e; array elements compared %d "
              "(fields rows %s, spectra masks and eigenvalues), differing arrays %d, the largest absolute difference %.3e; %s" % (
                  len(new), ", ".join("%.4f" % (r["T"] / rs) for r in new), st["numbers"], len(diffs), st["max"], nel, ", ".join(FIELD_KEYS), len(fd), fmax,
                  "GATE PASS" if out["pass"] else "GATE FAIL"))
        for d in diffs[:30]:
            print("  json: " + d)
        for d in fd[:30]:
            print("  arrays: " + d)
    else:
        routes = a.mode == "routes"
        rl = lambda J_: (J_["args"].get("weyl_certified") or "LZ")
        L = ["# The certified S11 segment%s (engine spec 39.4 (2), 40.1 (1), (2))" % (" on the %s route against the %s route" % (rl(J), rl(Jr)) if routes else " against the LZ run"), "",
             "Composed by compare_s11_segment_v2.py from %s (engine_collapse_v34, restarted at T/r_s %.4f; the %s route) and %s (%s); every number printed by the script." % (
                 a.run, T0 / rs, rl(J), a.ref, ("engine_collapse_v34, the %s route" % rl(Jr)) if routes else "the stored LZ run, engine_collapse_v33"), "",
             "The floor (40.1 (1); G2's resume of the LZ run to the solver's tolerance): E/E0 %.1e, the roots %.1e lattice units, min(N - |v|) %.1e; a difference within it reads \"no difference\"." % (
                 FLOOR["E_over_E0"], FLOOR["roots"], FLOOR["min_N_minus_absv"]), "",
             "| T/r_s | E/E0 run | E/E0 reference | difference | roots run | roots reference | outer root difference | the reference's outer root change from the record before | min(N - abs(v)) run | reference | difference | masks equal (channels) | mask = negative set (run; reference) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        prev_Rt = None
        prevq = byT.get(round(T0, 6))
        prev_Rt = (prevq.get("trapping_zeros") or [None])[-1] if prevq else None
        crit = []
        for r in new:
            q = byT.get(round(r["T"], 6))
            if q is None:
                continue
            i = int(np.argmin(np.abs(np.asarray(S["T"], float) - r["T"]))); j = int(np.argmin(np.abs(np.asarray(Sr["T"], float) - r["T"])))
            meq = [k for k in kap if np.array_equal(np.asarray(S["mask_%d" % k][i]), np.asarray(Sr["mask_%d" % k][j]))]
            zc = r.get("trapping_zeros") or []; zl = q.get("trapping_zeros") or []
            Rc = zc[-1] if zc else None; Rl = zl[-1] if zl else None
            dR = (Rc - Rl) if (Rc is not None and Rl is not None) else None
            mot = (Rl - prev_Rt) if (Rl is not None and prev_Rt is not None) else None
            if dR is not None and mot is not None:
                crit.append({"T_over_rs": r["T"] / rs, "outer_root_difference": dR, "LZ_motion": mot, "within": abs(dR) <= abs(mot)})
            cert = {k: (r.get("continuation") or {}).get(k, {}).get("certified") for k in (r.get("continuation") or {})}
            row = {"T_over_rs": r["T"] / rs, "E_over_E0": r["E_over_E0"], "E_over_E0_LZ": q["E_over_E0"], "roots": zc, "roots_LZ": zl, "outer_root_difference": dR,
                   "LZ_outer_root_motion": mot, "min_N_minus_absv": r["min_N_minus_absv"], "min_N_minus_absv_LZ": q["min_N_minus_absv"],
                   "masks_equal_channels": meq, "mask_equals_negative": r.get("mask_equals_negative_set"), "mask_equals_negative_LZ": q.get("mask_equals_negative_set"),
                   "continuation": {k: {kk: v.get(kk) for kk in ("mode", "nsub", "verified", "margin_min", "norm_dH_G", "gap_between_blocks", "inversions", "n_pocket")} for k, v in (r.get("continuation") or {}).items()},
                   "certificate": cert, "oracle": r.get("certificate_oracle"), "eigendecompositions": r.get("eigendecompositions"), "seconds": r.get("seconds")}
            out["records"].append(row)
            dI = (zc[0] - zl[0]) if (zc and zl and len(zc) == len(zl)) else None
            dE = r["E_over_E0"] - q["E_over_E0"]; dm = r["min_N_minus_absv"] - q["min_N_minus_absv"]
            row["inner_root_difference"] = dI; row["E_difference"] = dE; row["min_difference"] = dm
            row["roots_count_equal"] = len(zc) == len(zl)
            row["within_floor"] = bool(abs(dE) <= FLOOR["E_over_E0"] and abs(dm) <= FLOOR["min_N_minus_absv"] and len(zc) == len(zl)
                                       and (dR is None or abs(dR) <= FLOOR["roots"]) and (dI is None or abs(dI) <= FLOOR["roots"]) and len(meq) == len(kap))
            L.append("| %.4f | %.7f | %.7f | %s | %s | %s | %s | %s | %+.4f | %+.4f | %s | %d of %d | %s; %s |" % (
                r["T"] / rs, r["E_over_E0"], q["E_over_E0"], fl(dE, "E_over_E0", "%+.2e"), ", ".join("%.6f" % z for z in zc) or "none", ", ".join("%.6f" % z for z in zl) or "none",
                fl(dR, "roots", "%+.3e"), ("%+.4f" % mot) if mot is not None else "-", r["min_N_minus_absv"], q["min_N_minus_absv"], fl(dm, "min_N_minus_absv", "%+.2e"), len(meq), len(kap),
                r.get("mask_equals_negative_set"), q.get("mask_equals_negative_set")))
            prev_Rt = Rl
        L += ["", "Per record and channel, the certified continuation at the accepted fields (the lead's count ceil(2 ||dh||/gap); the pieces after halving; the failing pieces; the smallest margin gap - 2 ||dh_piece||) and the oracle (the full route through the same pieces: its mask against the run's, the subspace residual; version 3's one-step continuation: its mask against the run's):", "",
              "| T/r_s | channel | ||dh||_G | gap at the start | the lead's count | pieces | refinements | failing pieces | margin | verified | eigvals calls; s | oracle: full mask = run; residual; full tests pass; tests agree | oracle: v3 one-step mask = run; its verification |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for row in out["records"]:
            for k in sorted(row["certificate"] or {}, key=int):
                c = row["certificate"][k] or {}
                o = (row["oracle"] or {}).get(k) or {}
                of = o.get("full") or {}; o3 = o.get("v3_one_step") or {}
                L.append("| %.4f | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s; %s | %s; %s; %s; %s | %s; %s |" % (
                    row["T_over_rs"], k, ("%.4e" % c["norm_dH_G"]) if c.get("norm_dH_G") is not None else "-", ("%.4e" % c["gap_start"]) if c.get("gap_start") is not None else "-",
                    c.get("n_lead", "-"), c.get("pieces", "-"), c.get("refinements", "-"), c.get("failed_pieces", "-"), ("%.3e" % c["margin_min"]) if c.get("margin_min") is not None else "-",
                    c.get("verified", c.get("status")), c.get("eigvals_calls", "-"), ("%.1f" % c["eigvals_seconds"]) if c.get("eigvals_seconds") is not None else "-",
                    of.get("mask_equals_run", "-"), ("%.2e" % of["residual_run_on_full"]) if of.get("residual_run_on_full") is not None else "-", of.get("full_tests_pass", "-"),
                    of.get("tests_agree", "-"), o3.get("mask_equals_run", "-"), o3.get("verified", "-")))
        within = [c["within"] for c in crit]
        verdict = ("the roots agree within the LZ run's own motion between records at every compared record with a root in both (%d of %d): the pathfinder stands with its label" % (sum(within), len(within))
                   if within and all(within) else
                   ("the roots do not agree within the LZ run's own motion at %d of %d compared records: the formation is labelled \"on an uncertified transport\" and the owner's r_s/a 32 decision waits on a certified run" % (
                       len(within) - sum(within), len(within)) if within else "no record with a root in both runs: the criterion is not read"))
        if routes:
            def cert_ok(J_):
                bad = []
                for r_ in [x for x in J_["records"] if x["T"] > T0 + 1e-9]:
                    for k_, c_ in ((r_.get("continuation") or {}).items()):
                        cc = c_.get("certified") or {}
                        if not (cc.get("verified") is True and not cc.get("failed_pieces")):
                            bad.append("T/r_s %.4f channel %s (%s; failing pieces %s)" % (r_["T"] / rs, k_, cc.get("verified", cc.get("status")), cc.get("failed_pieces")))
                return bad
            b1, b2 = cert_ok(J), cert_ok(Jr)
            agree = bool(out["records"]) and all(x["within_floor"] for x in out["records"])
            verdict = (("the two routes agree within the floor on all %d records, the masks equal on every channel, and every channel's certificate passes on every sub-step in both: "
                        "the eigenvalue route is the certified construction of any further certified run and the full route its oracle (40.1 (2))" % len(out["records"]))
                       if (agree and not b1 and not b2) else
                       ("the routes do not meet 40.1 (2)'s condition (records within the floor: %d of %d; certificate failures: run %s; reference %s): the full route alone is certified and the eigenvalue route a record" % (
                           sum(1 for x in out["records"] if x["within_floor"]), len(out["records"]), "; ".join(b1) or "none", "; ".join(b2) or "none")))
            out["routes"] = {"agree_within_floor": agree, "certificate_failures_run": b1, "certificate_failures_ref": b2}
        out["criterion"] = {"rows": crit, "verdict": verdict}
        L += ["", ("- The routes (40.1 (2)): %s." if routes else "- The lead's criterion (39.4 (2)), read at the floor of 40.1 (1): %s.") % verdict, "- Rows: %s." % "; ".join("T/r_s %.4f: outer root difference %+.3e against the LZ run's motion %+.4f" % (c["T_over_rs"], c["outer_root_difference"], c["LZ_motion"]) for c in crit)]
        if a.md:
            open(a.md, "w").write("\n".join(L) + "\n")
        print("\n".join(L))
    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
