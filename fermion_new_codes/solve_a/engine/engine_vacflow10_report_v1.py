#!/usr/bin/env python
"""engine_vacflow10_report_v1.py (builder analyst; engine spec 26.6: vf10 "the cross-band share printed per cut and the band split read only
where it is below 0.1; vf9's profile, ramp and channels, its complete channels the comparator"): the vf10 record as tables. (1) The run and
the continuation per channel (engine_ops_v3's convention, c_LZ as recorded, never rejecting: the readouts, the readouts not verified by
Weyl, the largest sub-step count, pocket count and inversion count, the first unverified s); the complete channels (the full readout count,
not rejected) and any other channel listed apart. (2) Per cut (R = r_s, 1.5 r_s, 2 r_s) over the ramp, on the smoothed cut: the outflow
(-I integrated by the record's trapezoid), the cross-band remainder, the cross-band share |cross|/|total|, the six window bands, and the
two middle bands' share (the cos 2p < 0 bands, pi/4 < |p| < 3 pi/4 at either sign) read only where the cross-band share is below 0.1,
"rides the cos 2p < 0 bands" only where that share is also above 0.9 (engine 19.4's rule); for three channel sets: (a) vf10, every complete
channel (the sums rebuilt here from the per-channel fluences with the record's weights, checked against the record's own readings); (b)
vf10 on vf9's complete channels only; (c) vf9 on its complete channels (the comparator; its smoothed cut is absent where its window does not
fit). (3) Per channel and cut: vf10's sharp and smoothed outflow and smoothed cross-band remainder, vf9's sharp outflow (to its stop), and
over the channels complete in both records the largest relative difference of the sharp outflows (the two records share the profile, the
ramp and the propagation, so the complete channels' sharp outflows agree). Every number is read from the two records. Args: --vf10 --vf9
--md."""
import argparse
import json

MID = (1, 4)  # the bands p>0 pi/4<|p|<3pi/4 and p<0 pi/4<|p|<3pi/4: cos 2p < 0


def complete(rec):
    n = max(len(c["reads"]) for c in rec["channels"])
    return [c for c in rec["channels"] if len(c["reads"]) == n and not c.get("rejected")], n


def weight(c):
    return float(c.get("weight", 16.0 * c["kappa"]))


def sums(chs, cut, phase="ramp"):
    tot = 0.0; cross = 0.0; bands = [0.0] * 6; sharp = 0.0; have = False
    for c in chs:
        f = c["fluences"].get(cut, {}).get(phase)
        if not f:
            continue
        w = weight(c); sharp += w * f["total"]
        if "smooth_total_NO" in f:
            have = True
            tot += w * f["smooth_total_NO"]; cross += w * f["smooth_cross"]
            bands = [b + w * x for b, x in zip(bands, f["smooth_bands"])]
    if not have:
        return {"sharp": sharp, "smoothed": None}
    share = abs(cross) / abs(tot) if tot else float("inf")
    out = {"sharp": sharp, "smoothed": tot, "cross": cross, "cross_share": share, "bands": bands}
    if share < 0.1:
        out["middle_share"] = (bands[MID[0]] + bands[MID[1]]) / tot
        out["rides"] = bool(out["middle_share"] > 0.9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf10", required=True); ap.add_argument("--vf9", required=True); ap.add_argument("--md", required=True)
    a = ap.parse_args()
    A = json.load(open(a.vf10)); B = json.load(open(a.vf9))
    ra = A["args"]; rb = B["args"]; rs = float(ra["rs"])
    cA, nA = complete(A); cB, nB = complete(B)
    kB = {c["kappa"] for c in cB}
    cuts = sorted(A["readings"].keys(), key=float)
    L = ["## J-VF's vf10 on engine_vacflow_v10 (engine spec 26.6; engine_vacflow10_report_v1.py over %s, with vf9's record as the comparator)" % a.vf10.split("/")[-1], "",
         "- vf10: r_s %g, profile %s, m %g, kappa 1 to %d, t_r %g r_s, hold %g r_s, chain current %s; the smoothed cut of width %g sites and window %g sites (the step's tail at the window's edge %.2e); R2 continued by engine_ops_v3's convention (c_LZ %g, never rejecting, the lane's declaration); %.0f s." % (
             rs, ra["profile"], ra["m"], ra["kmax"], ra["tr"], ra["hold"], ra["chain_current"], ra["smooth_width"], ra["smooth_window"], A.get("smooth_tail", float("nan")), ra["c_LZ"], A.get("seconds", float("nan"))),
         "- vf9 (the comparator): the same profile, ramp and channels; smoothed cut width %g sites, window %g sites; its continuation the degeneracy-aware block rule with halving, which stopped kappa %s." % (
             rb["smooth_width"], rb["smooth_window"], ", ".join(str(c["kappa"]) for c in B["channels"] if c not in cB) or "none"),
         "- complete channels (the full readout count, not rejected): vf10 %d of %d (%d readouts each)%s; vf9 %d of %d (kappa %s)." % (
             len(cA), len(A["channels"]), nA, ("; apart: " + ", ".join(str(c["kappa"]) for c in A["channels"] if c not in cA)) if len(cA) < len(A["channels"]) else "",
             len(cB), len(B["channels"]), ",".join(str(k) for k in sorted(kB))), "",
         "### (1) The continuation per channel (vf10)", "",
         "| kappa | weight | readouts | not verified (Weyl) | largest sub-steps | largest pockets | largest inversions | first unverified s |", "|---|---|---|---|---|---|---|---|"]
    for c in A["channels"]:
        q = c.get("continuation") or {}
        L.append("| %d | %g | %s | %s | %s | %s | %s | %s |" % (c["kappa"], weight(c), q.get("readouts", len(c["reads"])), q.get("not_verified", "-"), q.get("substeps_max", "-"),
                 q.get("pockets_max", "-"), q.get("inversions_max", "-"), ("%.6f" % q["first_unverified_s"]) if q.get("first_unverified_s") is not None else "-"))
    L += ["", "### (2) Per cut over the ramp, on the smoothed cut: the cross-band share and the band split where it is below 0.1", "",
          "| cut | channel set | sharp outflow | smoothed outflow | cross-band remainder | cross-band share | middle bands' share (cos 2p < 0) | rides |", "|---|---|---|---|---|---|---|---|"]
    chk = []
    for ck in cuts:
        sets = (("vf10, all %d complete channels" % len(cA), sums(cA, ck)), ("vf10, vf9's %d complete channels" % len(kB), sums([c for c in cA if c["kappa"] in kB], ck)),
                ("vf9, its %d complete channels" % len(cB), sums(cB, ck)))
        for lab, s in sets:
            if s["smoothed"] is None:
                L.append("| %.2f r_s | %s | %.6e | absent (the window does not fit) | - | - | - | - |" % (float(ck) / rs, lab, s["sharp"])); continue
            L.append("| %.2f r_s | %s | %.6e | %.6e | %+.6e | %.4f | %s | %s |" % (float(ck) / rs, lab, s["sharp"], s["smoothed"], s["cross"], s["cross_share"],
                     ("%.4f" % s["middle_share"]) if "middle_share" in s else "not read", ("yes" if s["rides"] else "no") if "rides" in s else "-"))
        own = A["readings"].get(ck, {})
        s0 = sets[0][1]
        if own.get("total_outflow") is not None and s0["smoothed"] is not None:
            chk.append("%.2f r_s: the rebuilt smoothed outflow %.6e against the record's %.6e, the cross-band share %.6f against %.6f" % (
                float(ck) / rs, s0["smoothed"], own["total_outflow"], s0["cross_share"], own["cross_share"]))
    L += ["", "- the record's own readings (all channels) against the sums rebuilt here: " + "; ".join(chk) + ".", "",
          "| cut | channel set | the six smoothed-cut bands over the ramp (%s) |" % "; ".join(A["bands"]), "|---|---|---|"]
    for ck in cuts:
        for lab, chs in (("vf10, all", cA), ("vf10, vf9's complete", [c for c in cA if c["kappa"] in kB]), ("vf9, complete", cB)):
            s = sums(chs, ck)
            if s["smoothed"] is not None:
                L.append("| %.2f r_s | %s | %s |" % (float(ck) / rs, lab, "; ".join("%+.4e" % b for b in s["bands"])))
    L += ["", "### (3) Per channel and cut (the ramp)", "",
          "| kappa | cut | vf10 sharp outflow | vf10 smoothed outflow | vf10 smoothed cross | vf9 sharp outflow (to its stop) | vf9 complete |", "|---|---|---|---|---|---|---|"]
    byB = {c["kappa"]: c for c in B["channels"]}
    worst = {}
    for c in A["channels"]:
        for ck in cuts:
            f = c["fluences"][ck]["ramp"]; g = byB.get(c["kappa"], {}).get("fluences", {}).get(ck, {}).get("ramp")
            L.append("| %d | %.2f r_s | %.4e | %s | %s | %s | %s |" % (c["kappa"], float(ck) / rs, f["total"], ("%.4e" % f["smooth_total_NO"]) if "smooth_total_NO" in f else "-",
                     ("%+.3e" % f["smooth_cross"]) if "smooth_cross" in f else "-", ("%.4e" % g["total"]) if g else "-", "yes" if c["kappa"] in kB else "no"))
            if g and c["kappa"] in kB and c in cA:
                d = abs(f["total"] - g["total"]) / max(abs(g["total"]), 1e-300)
                worst[ck] = max(worst.get(ck, 0.0), d)
    L += ["", "- the channels complete in both records: the largest relative difference of the sharp outflows per cut: " + "; ".join("%.2f r_s %.2e" % (float(ck) / rs, worst[ck]) for ck in cuts if ck in worst) + ".",
          "- the band split is read only where the cross-band share is below 0.1, and \"rides the cos 2p < 0 bands\" only where the middle bands' share is also above 0.9 (engine 19.4, 26.6); every number above is read from the two records by this script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
