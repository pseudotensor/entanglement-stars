#!/usr/bin/env python
"""Strand 5 production P1 revision 3's record reads (JUDGMENT_strand5_production.md sec. 5.2's BUILDER items; builder analyst; no solve):
(1) the (0.01, 0.1) local twin's stage-2 lock split at z_c = 0.823379 from each record given (--twin-records: P1run1's version 10 record and
J4s2run9b's version 9 record of the local_ka0 (0, 0) family): the family's measured (A - B)/A, the stage-2 matter and vacuum parts, the
closure with mu dN_B, beside the lead's stage-1 attribution (the whole defect, P14's strain, their difference) read from the family record;
(2) the minimum of V_eff(q) = (2 phi - q phi')/(2 phi^2) over 0 < q <= pi (a uniform grid of 20000 points plus the endpoint) and its q, per
kernel family and per local twin stencil, with V_eff(0) beside it (strand5_P3_p06_v4.V_eff, the same function revision 6 used).
Writes --md and --out-json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_P3_p06_v4 as P4  # noqa: E402
import strand5_lattice_star_v3 as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True, help="the P3eff launcher's family list (label|family|stage2|karg|table|twin; ...)")
    ap.add_argument("--twin-records", required=True, help="comma list of stage-2 records of the (0.01, 0.1) local_ka0 (0, 0) family")
    ap.add_argument("--twin-family", required=True, help="that family's record (J4run4/lstar_m0.01_aH0.1_local_ka0_g0_0_pk.json)")
    ap.add_argument("--zc", type=float, default=0.823379)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    out = {"twin": [], "veff_min": []}
    F = json.load(open(a.twin_family))
    fm = [m for m in F["members"] if abs(m["z_c"] - a.zc) < 1e-6][0]
    for p in a.twin_records.split(","):
        R = json.load(open(p))
        m = [x for x in R["members"] if abs(x["z_c"] - a.zc) < 1e-6]
        if not m:
            out["twin"].append({"record": p, "missing": True})
            continue
        lk = m[0]["lock"]
        out["twin"].append({"record": p, "version": R["meta"].get("version"), "z_c": m[0]["z_c"],
                            "family_measured_AmB_over_A": fm.get("measured_AmB_over_A"), "family_P14_TF_prediction_AmB_over_A": fm.get("P14_TF_prediction_AmB_over_A"),
                            "matter_part_of_AmB_over_A": lk.get("matter_part_of_AmB_over_A"), "vac_part_of_AmB_over_A": lk.get("vac_part_of_AmB_over_A"),
                            "closure_total_with_mu_dNB": lk.get("closure_total_with_mu_dNB"), "closure_total": lk.get("closure_total")})
    q = np.concatenate([np.linspace(np.pi / 20000, np.pi, 20000)])
    for item in a.families.split(";"):
        lab, fam_p, s2_p, karg, tab_p, twin_p = item.split("|")
        M = json.load(open(fam_p))["meta"]
        kk, kv = karg.split(" ", 1)
        for op in (M["operator"], "local_ka0"):
            args = types.SimpleNamespace(kernel=kv if kk == "--kernel" else None, packets=kv if kk == "--packets" else None, mlight=M["mlight"],
                                         aH=M["aH"], operator=op)
            stencil, kop, info = S.setup_kernel(args)
            v = P4.V_eff(stencil, q)
            i = int(np.argmin(v))
            v0 = float(P4.mp_Veff(np.array(sorted(stencil), float), np.array([stencil[int(n)] for n in sorted(stencil)]), 1e-6))
            out["veff_min"].append({"family": lab, "operator": op, "min_V_eff": float(v[i]), "q_at_min": float(q[i]), "V_eff_near_0": v0,
                                    "V_eff_at_pi": float(v[-1]), "negative_count": int(np.sum(v < 0)), "R": int(max(stencil))})
    L = ["# Strand 5 P1 revision 3's record reads (strand5_P1rev3_reads_v1.py)", "", "## (1) The (0.01, 0.1) local twin at z_c = %g" % a.zc, ""]
    for t in out["twin"]:
        L.append("- " + ", ".join("%s %s" % (k, v) for k, v in t.items()))
    L += ["", "## (2) V_eff over 0 < q <= pi (20000 points): its minimum and q, V_eff(1e-6), V_eff(pi), the count of negative values", "",
          "| family | operator | min V_eff | q at min | V_eff(1e-6) | V_eff(pi) | negative count | R |", "|---|---|---|---|---|---|---|---|"]
    for v in out["veff_min"]:
        L.append("| %s | %s | %.6e | %.6f | %.6e | %.6e | %d | %d |" % (v["family"], v["operator"], v["min_V_eff"], v["q_at_min"], v["V_eff_near_0"],
                                                                     v["V_eff_at_pi"], v["negative_count"], v["R"]))
    open(a.md, "w").write("\n".join(L) + "\n")
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
