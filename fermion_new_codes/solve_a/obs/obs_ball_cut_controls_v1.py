#!/usr/bin/env python3
"""obs_ball_cut_controls_v1.py (builder analyst; S6 = J4): the dense-control table from obs_ball_cut_v1.py's validate records and the lane's
strand-3 dense code's records (balls.py measure --method dense).  Per ball: the sites, the largest |x|^2 included, the five routes of
obs_ball_cut_v1.py (dense 4-component, scalar x 4 by the numerical transform and analytic, sublattice SVD, irrep blocks) and their spread,
balls.py's value on the same (radius, box), and the differences from the review's value.  CPU, instant.
Args: --val (validate JSONs) --balls (balls.py JSONs) --out-json --md."""
import argparse
import json

ROUTES = ("S_dense4", "S_scalar_x4_numerical_transform", "S_scalar_x4", "S_sublattice_svd", "S_irrep_blocks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--balls", nargs="*", default=[])
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    balls = {}
    for f in a.balls:
        r = json.load(open(f))
        balls[(float(r["radius"]), int(r["box"]))] = dict(file=f, entropy=r["entropy"], sites=r["sites"])
    rows = []
    md = ["## S6 (J4) dense controls (obs_ball_cut_controls_v1.py)", "",
          "Site-centred, antiperiodic, m 0.  Ball = {x : |x|^2 <= R^2}: at R = 3.99 and 5.99 it is the strict ball |x| < 4, |x| < 6.", "",
          "| R | L | sites | max |x|^2 | dense 4-component | irrep blocks | spread of the 5 routes | balls.py (lane) | review | dense - review | balls.py - dense |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in a.val:
        r = json.load(open(f))
        vals = [r[k] for k in ROUTES]
        spread = max(vals) - min(vals)
        b = balls.get((float(r["R"]), int(r["L"])))
        ref = r.get("review_ref")
        row = dict(file=f, R=r["R"], L=r["L"], n=r["n"], max_x2=r["max_x2"], routes={k: r[k] for k in ROUTES}, spread=spread,
                   balls=b, review=ref, dense_minus_review=(r["S_dense4"] - ref) if ref is not None else None,
                   balls_minus_dense=(b["entropy"] - r["S_dense4"]) if b else None,
                   checks=dict(kernel_forbidden=r["kernel_forbidden_parity_max"], kernel_direct=r.get("kernel_direct_sum_maxdev"),
                               omega=r["omega_transform"], commutation=r["commutation_VK_minus_KV_max"], frob=r["frob"],
                               svals_top=r["svals_blocks_vs_full_maxdev_top"], group=r["group_checks"]))
        rows.append(row)
        md.append("| %g | %d | %d | %d | %.10f | %.10f | %.1e | %s | %s | %s | %s |" % (
            r["R"], r["L"], r["n"], r["max_x2"], r["S_dense4"], r["S_irrep_blocks"], spread,
            ("%.10f" % b["entropy"]) if b else "(pending)", ("%.8f" % ref) if ref is not None else "-",
            ("%.2e" % row["dense_minus_review"]) if ref is not None else "-", ("%.1e" % row["balls_minus_dense"]) if b else "-"))
    worst = dict(kernel_forbidden=max(r["checks"]["kernel_forbidden"] for r in rows),
                 omega_offdiag=max(r["checks"]["omega"]["internal_offdiag_max"] for r in rows),
                 omega_vs_analytic=max(r["checks"]["omega"]["numerical_minus_analytic_Cplus_max"] for r in rows),
                 commutation=max(r["checks"]["commutation"] for r in rows),
                 frob_rel=max(abs(r["checks"]["frob"]["blocks"] - r["checks"]["frob"]["mask_correlation"]) / r["checks"]["frob"]["mask_correlation"] for r in rows),
                 svals_top=max(r["checks"]["svals_top"] for r in rows))
    md += ["", "Largest check deviations over these balls: kernel parity rule %.1e, Omega transform off-diagonal %.1e, numerical minus analytic "
           "C_+ %.1e, V K - K V %.1e, Frobenius (blocks vs mask correlation, relative) %.1e, singular values blocks vs full sublattice %.1e." % (
               worst["kernel_forbidden"], worst["omega_offdiag"], worst["omega_vs_analytic"], worst["commutation"], worst["frob_rel"], worst["svals_top"])]
    with open(a.out_json, "w") as fh:
        json.dump(dict(script="obs_ball_cut_controls_v1.py", rows=rows, worst=worst), fh, indent=1)
    with open(a.md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
