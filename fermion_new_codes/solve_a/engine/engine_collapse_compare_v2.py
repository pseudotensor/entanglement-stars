#!/usr/bin/env python
"""engine_collapse_compare_v2.py (builder analyst; the validation of engine_collapse_v17 of SPEC_final4b_engine.md 18.5 against an earlier
engine's run with the same physical arguments, the reference and the runs in their own directories): per record matched by T, the
differences of the accepted state: E (relative), the shell pair energy (relative), R2 - R0 (relative), the stored fields N and v (the
solve from the accepted sources: max over the sites, absolute), the applied fields N_applied, v_applied (the fields of the propagating
h_m), and beside them each run's own last self-consistency change (max |dN|, |dv| of its iteration history, where recorded): the
reference's under-convergence bounds the agreement that can be read. Also the iterations, the eigendecompositions and the seconds per
record. Args: --ref-dir --ref --dir --others (tags) --records (the first K records) --md."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json

import numpy as np


def load(d, tag):
    J = json.load(open(os.path.join(d, "collapse_%s.json" % tag)))
    Z = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
    return J, Z


def rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def last_change(rec):
    h = rec.get("iteration_history")
    if not h:
        return None
    return max(h[-1][1], h[-1][2])


def f1(x):
    return ("%.1e" % x) if x is not None else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--records", type=int, default=5)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    J0, Z0 = load(a.ref_dir, a.ref)
    T0 = np.asarray(Z0["T"], float)
    recs0 = [r for r in J0["records"] if r.get("E") is not None]
    L = ["## %s against %s (engine_collapse_compare_v2.py; reference %s/%s, %s)" % (", ".join(a.others), a.ref, os.path.basename(os.path.normpath(a.ref_dir)), a.ref, J0.get("script")), ""]
    keys = ("m", "g", "rs", "R0_over_rs", "reading", "nsites", "ell", "tau", "kappa_sea", "reference", "shell")
    L.append("Reference arguments: %s" % json.dumps({k: J0["args"].get(k) for k in keys}))
    for tag in a.others:
        J1, Z1 = load(a.dir, tag)
        T1 = np.asarray(Z1["T"], float)
        recs1 = [r for r in J1["records"] if r.get("E") is not None]
        diff_args = {k: (J0["args"].get(k), J1["args"].get(k)) for k in keys if J0["args"].get(k) != J1["args"].get(k)}
        L += ["", "### %s (%s; site fields %s, stop |dE/E| %s, fields %s, %s)" % (tag, J1.get("script"), J1["args"].get("site_fields", "dense"), J1["args"].get("stop_E", "1e-9 (E only)"),
                                                                              J1["args"].get("stop_fields", "-"), J1["args"].get("mix", "Anderson beta 0.5")),
              "", "Arguments differing from the reference: %s" % (json.dumps(diff_args) if diff_args else "none"), "",
              "| T/rs | E rel | pairs rel | R2-R0 rel | max dN (stored) | max dv (stored) | max dN (applied) | max dv (applied) | ref's last change | this run's last change | iterations ref; this | eigh this | seconds ref; this |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        worst = {"E": 0.0, "N": 0.0, "Napp": 0.0, "ref_last": 0.0}
        for i1, q in enumerate(recs1[:a.records]):
            k0 = int(np.argmin(np.abs(T0 - q["T"])))
            k1 = int(np.argmin(np.abs(T1 - q["T"])))
            if abs(T0[k0] - q["T"]) > 1e-9 or k0 >= len(recs0):
                L.append("| %.3f | no reference record |" % (q["T"] / J1["args"]["rs"]))
                continue
            p = recs0[k0]
            dN = float(np.max(np.abs(np.asarray(Z1["N"][k1]) - np.asarray(Z0["N"][k0]))))
            dv = float(np.max(np.abs(np.asarray(Z1["v"][k1]) - np.asarray(Z0["v"][k0]))))
            dNa = dva = None
            if "N_applied" in Z0.files and "N_applied" in Z1.files and len(Z0["N_applied"]) > k0:
                dNa = float(np.max(np.abs(np.asarray(Z1["N_applied"][k1]) - np.asarray(Z0["N_applied"][k0]))))
                dva = float(np.max(np.abs(np.asarray(Z1["v_applied"][k1]) - np.asarray(Z0["v_applied"][k0]))))
            eR = rel(p["E"], q["E"])
            lc0, lc1 = last_change(p), last_change(q)
            worst["E"] = max(worst["E"], eR); worst["N"] = max(worst["N"], dN)
            if dNa is not None:
                worst["Napp"] = max(worst["Napp"], dNa)
            if lc0 is not None:
                worst["ref_last"] = max(worst["ref_last"], lc0)
            L.append("| %.3f | %.1e | %s | %s | %.1e | %.1e | %s | %s | %s | %s | %s; %s | %s | %.0f; %.0f |" % (
                q["T"] / J1["args"]["rs"], eR, f1(rel(p.get("E_pairs_shells"), q.get("E_pairs_shells"))), f1(rel(p.get("R2_minus_R0_F"), q.get("R2_minus_R0_F"))),
                dN, dv, f1(dNa), f1(dva), f1(lc0), f1(lc1), p.get("iterations"), q.get("iterations"), q.get("eigendecompositions", "-"),
                p.get("seconds", float("nan")), q.get("seconds", float("nan"))))
        L += ["", "- largest over these records: E rel %.1e, N (stored) %.1e, N (applied) %.1e; the reference's own last self-consistency change up to %.1e." % (
            worst["E"], worst["N"], worst["Napp"], worst["ref_last"])]
    L += ["", "- every number above is printed by the script from the runs' json and fields npz."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
