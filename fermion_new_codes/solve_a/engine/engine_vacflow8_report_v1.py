#!/usr/bin/env python
"""engine_vacflow8_report_v1.py (builder analyst; spec 18.2, J-VF's decisive rerun): engine_vacflow_v8's record as tables: (1) the channel
sums (weights 16 kappa) of the energy inside and outside R = r_s under NO, R2 and raw at the readouts every 2 r_s of the ramp, at the ramp's
end and after the hold, with N_p = Tr[R2 (1 - R0)] and N_h = Tr[R0 (1 - R2)] (weighted and unweighted); (2) per cut (r_s, 1.5 r_s, 2 r_s)
the outflow fluences (-I integrated over the ramp and over the hold) in the six window bands, the cross-band remainder, the window total
against the narrow-window total, and the local work inside the cut; the groupings: cos 2p < 0 (the middle bands) against cos 2p > 0, the
lead's |p| below and above pi/4, and p > 0 against p < 0; (3) per channel at r_s: the ramp fluence's total and bands, the inside NO energy
at the ramp's end, N_p and N_h after the hold; (4) the record's verdict line. Every number is read from the record (the fluences are
engine_vacflow_v8's trapezoid over the 0.5 r_s readouts). Args: record --md."""
import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    d = json.load(open(a.record))
    args = d["args"]; rs = float(args["rs"])
    bands = d["bands"]
    tot = d["totals"]
    chans = d["channels"]
    t = np.array(tot["t_over_rs"])
    L = ["## J-VF's decisive rerun: engine_vacflow_v8 at r_s %g (t_r %g r_s, hold %g r_s, kappa 1 to %d, %s current, the %s profile, m %g; engine_vacflow8_report_v1.py over %s)" % (
        rs, args["tr"], args["hold"], d.get("kappa_done", args["kmax"]), args["chain_current"], args["profile"], args["m"], a.record.split("/")[-1]), ""]
    if tot.get("left_out"):
        L += ["Channels left out of the sums: %s." % json.dumps(tot["left_out"]), ""]
    L += ["### (1) The energy inside and outside R = r_s (channel sums, weights 16 kappa) and the particle and hole counts", "",
          "| t/r_s | phase | E_in NO | E_out NO | E_in R2 | E_out R2 | E_in raw | E_out raw | N_p (weighted) | N_h (weighted) | N_p, N_h (unweighted) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    ph = [q["phase"] for q in chans[0]["reads"]]
    sel = [i for i in range(len(t)) if (abs(t[i] / 2.0 - round(t[i] / 2.0)) < 1e-9) or i == len(t) - 1 or (ph[i] == "ramp" and (i + 1 < len(ph) and ph[i + 1] == "hold"))]
    for i in sorted(set(sel)):
        L.append("| %.2f | %s | %.6e | %.6e | %.6e | %.6e | %.6e | %.6e | %.4e | %.4e | %.4f, %.4f |" % (
            t[i], ph[i], tot["E_in_NO"][i], tot["E_out_NO"][i], tot["E_in_R2"][i], tot["E_out_R2"][i], tot["E_in_raw"][i], tot["E_out_raw"][i],
            tot["N_p"][i], tot["N_h"][i], tot.get("N_p_unweighted", [float("nan")] * len(t))[i], tot.get("N_h_unweighted", [float("nan")] * len(t))[i]))
    L += ["", "### (2) The outflow fluences through the cuts (channel sums; -I integrated over time: positive = energy leaving the inside)", ""]
    for ck, f in tot["fluences"].items():
        for phase, r in f.items():
            b = r["bands"]
            rev = b[1] + b[4]; nonrev = b[0] + b[2] + b[3] + b[5]
            L.append("- cut %s (%s): total %.6e (narrow window), window %.6e = band-diagonal %.6e + cross %.6e; bands %s; cos 2p < 0 %.6e, cos 2p > 0 %.6e; |p| < pi/4 %.6e, |p| > pi/4 %.6e; p >= 0 %.6e, p < 0 %.6e; work inside %.6e" % (
                ck, phase, r["total"], r["window_total"], sum(b), r["cross"], "; ".join("%s %.4e" % (n, x) for n, x in zip(bands, b)), rev, nonrev, b[0] + b[3],
                b[1] + b[2] + b[4] + b[5], b[0] + b[1] + b[2], b[3] + b[4] + b[5], r["work"]))
    L += ["", "### (3) Per channel at R = r_s", "", "| kappa | weight | outflow fluence over the ramp | its bands (6) | cross | E_in NO at the ramp's end | N_p, N_h after the hold | halvings | rejected |",
          "|---|---|---|---|---|---|---|---|---|"]
    key = "%g" % rs
    for c in chans:
        fr = (c.get("fluences") or {}).get(key, {}).get("ramp", {})
        last_ramp = [q for q in c["reads"] if q["phase"] == "ramp"][-1]
        L.append("| %d | %g | %s | %s | %s | %.4e | %.4f, %.4f | %d | %s |" % (
            c["kappa"], c.get("weight", 16 * c["kappa"]), ("%.4e" % fr["total"]) if "total" in fr else "-", (", ".join("%.2e" % x for x in fr["bands"])) if "bands" in fr else "-",
            ("%.2e" % fr["cross"]) if "cross" in fr else "-", last_ramp["E_in_NO"], c["reads"][-1]["N_p"], c["reads"][-1]["N_h"], c.get("halvings", 0), c.get("rejected") or "-"))
    v = d.get("verdict")
    L += ["", "### (4) The verdict", "", ("- %s" % json.dumps(v)) if v else "- no verdict in the record (the run has not ended)",
          "", "- every number above is read from the record; the fluences are the record's trapezoid over the readouts every %g r_s." % args["every"]]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:3]))


if __name__ == "__main__":
    main()
