#!/usr/bin/env python
"""engine_collapse_compare_v1.py (builder analyst; the equivalence test of engine_collapse_v12 against engine_collapse_v11): per record
of a reference collapse run and each other run with the same arguments, the differences of the accepted state: E (relative), the pair
energy in the shells, R_t, min(N - |v|), R2 - R0, M1's parts X1, X2, Y1, Y2 and I_E, the certificate's B, q_c and eps_map; from the
fields npz the largest differences of N, v (absolute) and of eps_tot, J_tot, eps_pairs (relative to their largest values); the iteration
counts and the seconds per record (v12's timing where present). Args: --ref (tag) --others (tags) --dir --md."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--others", nargs="+", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    J0, Z0 = load(a.dir, a.ref)
    r0 = [r for r in J0["records"] if r.get("E") is not None]
    L = ["## engine_collapse_v12 against v11 on the same arguments (engine_collapse_compare_v1.py; reference %s, %s)" % (a.ref, J0["script"]), "",
         "Arguments: %s" % json.dumps({k: J0["args"][k] for k in ("rs", "nsites", "ell", "tau", "kappa_sea", "reference", "T_over_rs")}), ""]
    for tag in a.others:
        J1, Z1 = load(a.dir, tag)
        r1 = [r for r in J1["records"] if r.get("E") is not None]
        L += ["### %s (%s; reuse_first %s, verify_at_end %s)" % (tag, J1["script"], J1["args"].get("reuse_first"), J1["args"].get("verify_at_end")), "",
              "| T/rs | E rel | pairs rel | R_t | min(N-abs v) diff | R2-R0 rel | X1, X2, Y1, Y2, I_E rel (max) | B ref; this | eps_map ref; this | max dN | max dv | eps_tot rel | J_tot rel | eps_pairs rel | iterations ref; this | seconds ref; this |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        n = min(len(r0), len(r1), len(Z0["T"]), len(Z1["T"]))
        worst = {"E": 0.0, "N": 0.0, "eps": 0.0}
        for i in range(n):
            p, q = r0[i], r1[i]
            m1 = max([rel(p["M1_parts"][k], q["M1_parts"][k]) or 0.0 for k in ("X1", "X2", "Y1", "Y2", "I_E_start")])
            dN = float(np.max(np.abs(np.asarray(Z0["N"][i], float) - np.asarray(Z1["N"][i], float))))
            dv = float(np.max(np.abs(np.asarray(Z0["v"][i], float) - np.asarray(Z1["v"][i], float))))
            fr = lambda key: float(np.max(np.abs(np.asarray(Z0[key][i], float) - np.asarray(Z1[key][i], float))) / max(float(np.max(np.abs(np.asarray(Z0[key][i], float)))), 1e-300))
            eE = rel(p["E"], q["E"])
            worst["E"] = max(worst["E"], eE); worst["N"] = max(worst["N"], dN); worst["eps"] = max(worst["eps"], fr("eps_tot"))
            c0, c1 = p.get("certificate") or {}, q.get("certificate") or {}
            L.append("| %.3f | %.1e | %s | %s | %.1e | %s | %.1e | %s; %s | %s; %s | %.1e | %.1e | %.1e | %.1e | %.1e | %s; %s | %.0f; %.0f |" % (
                p["T_over_rs"], eE, ("%.1e" % rel(p["E_pairs_shells"], q["E_pairs_shells"])) if p["E_pairs_shells"] else "-",
                ("%s; %s" % (p["R_t"], q["R_t"])) if (p["R_t"] or q["R_t"]) else "-", abs(p["min_N_minus_absv"] - q["min_N_minus_absv"]),
                ("%.1e" % rel(p["R2_minus_R0_F"], q["R2_minus_R0_F"])) if p["R2_minus_R0_F"] else "-", m1,
                ("%.1e" % c0["B"]) if c0.get("B") is not None else "unc.", ("%.1e" % c1["B"]) if c1.get("B") is not None else "unc.",
                ("%.1e" % c0["eps_map"]) if c0.get("eps_map") is not None else "-", ("%.1e" % c1["eps_map"]) if c1.get("eps_map") is not None else "-",
                dN, dv, fr("eps_tot"), fr("J_tot"), fr("eps_pairs"), p["iterations"], q["iterations"], p["seconds"], q["seconds"]))
        L += ["", "- worst over %d records: E rel %.1e, N %.1e, eps_tot rel %.1e" % (n, worst["E"], worst["N"], worst["eps"])]
        tms = [q.get("timing") for q in r1 if q.get("timing")]
        if tms:
            L += ["- %s's timing per record: %s" % (tag, "; ".join("eigh %d (%.1f s), solves %d (%.1f s), certificate %.1f s, M1 %.1f s, continuation %.1f s, verification %.1f s, reused %d fast %d blocks %d midpoint %d fallbacks %d" % (
                t["eigh_n"], t["eigh_s"], t["solve_n"], t["solve_s"], t["certificate_s"], t["M1_s"], t["continuation_s"], t["verify_s"], t["reused"], t["fast"], t["block"],
                t["midpoint"], t["fallbacks"]) for t in tms))]
        L.append("")
    L += ["- every number above is printed by the script from the records and the fields npz files."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
