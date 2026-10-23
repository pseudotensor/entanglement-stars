#!/usr/bin/env python
"""Strand 2b run summary (builder analyst): counts and extremes over a run's case records (law and native control) for
the result file's run record: converged cases, final residuals against the spec's 1e-11 (with the measured noise floor),
d_0 values, the smallest gap and sigma_min(J), the finite-difference step agreements, and control C13 where it ran.
JSON reading and max/min only; emitted, not retyped.
(version 2: records whose final residual stayed above 1e-11 keep their solution (driver v7 on) and are counted and summarized
with the others; version 1 counted them as unconverged and left them out of the residual line)"""
import argparse
import glob
import json
import os


def f(v):
    return "-" if v is None else "%.3e" % float(v)


def summarize(recs, label):
    ok = [r for r in recs if "psi" in r]
    clean = [r for r in ok if r.get("continuation", {}).get("ok")]
    lost = [r for r in recs if "psi" not in r]
    lines = ["- %s: %d records, %d with a solution reached by continuation (positive psi and N at every site: %d); %d of them with the "
             "final Newton residual at or below 1e-11; %d records without a solution%s." % (
        label, len(recs), len(ok), sum(1 for r in ok if min(r["psi"]) > 0 and min(r["N"]) > 0), len(clean), len(lost),
        (" (" + "; ".join("%s m %s k %s: lost at lambda %s" % (r.get("block"), r.get("m"), r.get("k"),
                                                              r.get("continuation", {}).get("lost_at_lam")) for r in lost) + ")") if lost else "")]
    if not ok:
        return lines
    fin = [(r["continuation"].get("final_residual_best"), r) for r in ok]
    above = [(v, r) for v, r in fin if v is not None and v > 1e-11]
    worst = max(fin, key=lambda t: t[0] if t[0] is not None else -1)
    lines.append("- %s: final residual (best iterate) above 1e-11 in %d records (%s); largest %s (block %s, m %s, k %s; its noise floor %s)." % (
        label, len(above), "; ".join("block %s m %s g (%s, %s, %s) k %s: %s, noise floor %s" % (
            r.get("block"), r.get("m"), r.get("gM"), r.get("gB"), r.get("Lambda"), r.get("k"), f(v), f(r.get("residual_noise_floor")))
            for v, r in above) if above else "none", f(worst[0]), worst[1].get("block"), worst[1].get("m"), worst[1].get("k"),
        f(worst[1].get("residual_noise_floor"))))
    if above:
        nf_ok = sum(1 for v, r in above if r.get("residual_noise_floor") is not None and v <= 10 * r["residual_noise_floor"])
        lines.append("- %s: of those, %d have the final residual within 10 times the measured noise floor at the solution." % (label, nf_ok))
    d0s = {}
    for r in ok:
        d0s[(r.get("block"), r.get("d0"))] = d0s.get((r.get("block"), r.get("d0")), 0) + 1
    lines.append("- %s: (block, d_0): count = %s." % (label, ", ".join("(%s, %s): %d" % (k[0], k[1], v) for k, v in sorted(d0s.items(), key=str))))
    g = min(ok, key=lambda r: r.get("gap_block", 1e300))
    s = min(ok, key=lambda r: r.get("sigma_min_J", 1e300))
    lines.append("- %s: smallest gap to the next level %s (block %s, m %s, k %s); smallest sigma_min(J) %s (block %s, m %s, k %s)." % (
        label, f(g.get("gap_block")), g.get("block"), g.get("m"), g.get("k"), f(s.get("sigma_min_J")), s.get("block"), s.get("m"), s.get("k")))
    sa = [r.get("drho_step_agreement_rel") for r in ok if r.get("drho_step_agreement_rel") is not None]
    s4 = [r.get("drho_4th_order_vs_2nd_1e-4_rel") for r in ok if r.get("drho_4th_order_vs_2nd_1e-4_rel") is not None]
    if sa:
        lines.append("- %s: d rho^P/d psi step agreement (relative, max over records) %s; fourth order at h 1e-3 against second order at "
                     "1e-4 (relative, max) %s." % (label, f(max(sa)), f(max(s4)) if s4 else "-"))
    c13 = [r["C13"] for r in ok if isinstance(r.get("C13"), dict) and "C13_rel" in r["C13"]]
    if c13:
        lines.append("- %s: control C13 ran in %d records: max relative (overlap route against c-basis) %s; max determinant-against-vector "
                     "overlap difference %s; max sea-rotation term %s." % (label, len(c13), f(max(c["C13_rel"] for c in c13)),
                                                                        f(max(c.get("det_vs_vector_overlap_max", 0) for c in c13)),
                                                                        f(max(c["sea_rotation_term_max"] for c in c13))))
    else:
        lines.append("- %s: control C13 did not run (it runs on (1,0) and (2,0) when d_0 = 1)." % label)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    law = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.run, "case_*.json")))]
    nat = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.run, "native", "native_q*.json")))]
    lines = summarize(law, "law") + (summarize(nat, "native control") if nat else [])
    open(a.md, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
