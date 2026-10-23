#!/usr/bin/env python
"""read_newton_spectra_v1.py (the lane's analyst builder; engine spec 45.1 (3)): a no-run reading of two stored collapse runs of the sigma/a 2
member (the stopped v36 run and the pod 2 v32 member) at T/r_s 0.5 to 1.0: per record each solve's name, status, evaluations, kept residual,
confirming residual and |dE/E|, whether it met the stop rule (the v32 continuation's stages and the direct Newton's last residual before it;
the v36 run's kept and confirming residuals), the v36 run's jvp_v35 entry (exact products, dropped near-degenerate pairs, extra evaluations),
and from the stored spectra the smallest and second smallest |eigenvalue| per channel per record in both runs, with the occupied mask's
boundary (the largest |e| among unoccupied-negative or smallest among occupied-positive is not inferred: only |e| and the mask's value at the
smallest |e| are printed). Every number printed by the script. Usage: --run DIR/TAG --ref DIR/TAG --label-run --label-ref --tmin --tmax --md --out-json."""
import argparse, json, os, time
import numpy as np


def load(p):
    d, t = os.path.split(p)
    J = json.load(open(os.path.join(d, "collapse_%s.json" % t)))
    S = np.load(os.path.join(d, "collapse_%s_spectra.npz" % t))
    return J, S


def spec(S, T, kap):
    Ts = np.asarray(S["T"], float); i = int(np.argmin(np.abs(Ts - T)))
    if abs(Ts[i] - T) > 1e-9:
        return None
    out = {}
    for k in kap:
        e = np.asarray(S["e_%d" % k][i], float); o = np.argsort(np.abs(e))
        out[k] = {"min_abs": float(abs(e[o[0]])), "second_abs": float(abs(e[o[1]])), "e_at_min": float(e[o[0]])}
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("--run", "--ref", "--md", "--out-json"): ap.add_argument(x, required=True)
    ap.add_argument("--label-run", default="run"); ap.add_argument("--label-ref", default="ref")
    ap.add_argument("--tmin", type=float, default=0.5); ap.add_argument("--tmax", type=float, default=1.0)
    a = ap.parse_args()
    runs = {a.label_run: load(a.run), a.label_ref: load(a.ref)}
    out = {"script": "read_newton_spectra_v1.py", "run": a.run, "ref": a.ref, "records": {}}
    L = ["# read_newton_spectra_v1: %s and %s at T/r_s %.3f to %.3f (engine spec 45.1 (3); no run)" % (a.label_run, a.label_ref, a.tmin, a.tmax), "",
         "Composed %s by code/engine/read_newton_spectra_v1.py from the stored json and spectra npz of %s and %s; every number printed by the script." % (
             time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), a.run, a.ref), "", "## Newton histories", "",
         "| T/r_s | run | solve | status | evaluations | kept residual | confirming residual | confirming dE/E | meets stop rule |", "|---|---|---|---|---|---|---|---|---|"]
    SP = []
    for lab, (J, S) in runs.items():
        rs = float(J["args"]["rs"]); kap = list(range(1, int(J["args"]["kappa_sea"]) + 1))
        for r in J["records"]:
            if r.get("E") is None: continue
            t = r["T"] / rs
            if t < a.tmin - 1e-9 or t > a.tmax + 1e-9: continue
            s = r.get("solve_v32") or {}
            key = "%.4f" % t; out["records"].setdefault(key, {})[lab] = {"solves": [], "jvp_v35": r.get("jvp_v35"), "spectrum": None,
                                                                          "kept_residual": s.get("kept_residual"), "continuation": s.get("continuation")}
            for so in s.get("solves", []):
                e = {k: so.get(k) for k in ("name", "status", "evaluations", "kept_residual", "confirming_residual", "confirming_dE", "meets_stop_rule")}
                out["records"][key][lab]["solves"].append(e)
                L.append("| %s | %s | %s | %s | %s | %.2e | %.2e | %.2e | %s |" % (key, lab, e["name"], (e["status"] or "-")[:70], e["evaluations"],
                         e["kept_residual"] or 0, e["confirming_residual"] or 0, e["confirming_dE"] or 0, e["meets_stop_rule"]))
            jv = r.get("jvp_v35")
            if jv: L.append("| %s | %s | jvp_v35 | products %s, dropped pairs %s, extra evaluations %s, ref mismatch %s | | | | | |" % (
                key, lab, jv.get("products"), jv.get("dropped_pairs"), jv.get("extra_evaluations"), jv.get("ref_mismatch")))
            sp = spec(S, r["T"], kap); out["records"][key][lab]["spectrum"] = sp
            if sp: SP.append((key, lab, sp))
    L += ["", "## Smallest |eigenvalue| per channel (second smallest after the slash; lattice units)", "",
          "| T/r_s | run | " + " | ".join("ch %d" % k for k in range(1, 11)) + " |", "|---|---|" + "---|" * 10]
    for key, lab, sp in sorted(SP):
        L.append("| %s | %s | %s |" % (key, lab, " | ".join("%.3e / %.3e" % (sp[k]["min_abs"], sp[k]["second_abs"]) for k in sorted(sp))))
    allmin = [(sp[k]["min_abs"], key, lab, k) for key, lab, sp in SP for k in sp]
    if allmin:
        m = min(allmin); L += ["", "- the smallest |eigenvalue| over these records, runs and channels: %.3e (T/r_s %s, %s, channel %d)" % m]
        out["smallest"] = {"value": m[0], "T_over_rs": m[1], "run": m[2], "channel": m[3]}
    open(a.md, "w").write("\n".join(L) + "\n"); json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("records %d; done" % len(out["records"]))


if __name__ == "__main__":
    main()
