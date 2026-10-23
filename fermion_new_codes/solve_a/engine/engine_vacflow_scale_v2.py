#!/usr/bin/env python
"""engine_vacflow_scale_v2.py (builder analyst; version 2 of engine_vacflow_scale_v1.py: --kshow K prints the per-channel rows for kappa <= K and the sum of the rest in one row; nothing else changed. Version 1: J-VF, the r_s = 32 record against r_s = 16): the same quantities of two or more
engine_vacflow_v3 records at the same t_r/r_s, printed side by side, with the ratio of the larger r_s to the smaller one beside the ratio a
scale-covariant (continuum) chain would give (an energy at fixed R/r_s and t_r/r_s scales as 1/r_s, so 16/32 = 0.5; the chain's kappa/R term
keeps kappa fixed under the rescaling). Printed: (1) the annulus energies [1.5, 2], [2, 4], [4, 8], [8, 16] r_s of the stored totals (raw and
NO, after the hold) summed over the channels the records share (the running channel sums at the probes); (2) per channel kappa the NO annulus
energies [1.5, 2] and [2, 4] r_s (differences of successive running sums); (3) the NO shell totals summed over the channels at
R/r_s = 0.5 .. 8 (where the energy sits). Args: --records --md --kshow."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse
import json

import numpy as np

ANN = [(1.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0)]
PROBES = [1.5, 2.0, 4.0, 8.0, 16.0]
XS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def load(f):
    D = json.load(open(f))
    return {"file": os.path.basename(f), "rs": float(D["args"]["rs"]), "D": D}


def annuli(run, kmax, key):
    """the annulus energies from the running channel sums at the probes, channels 1..kmax"""
    r = [x for x in run if x["kappa"] <= kmax][-1]
    E = r["E_raw_at" if key == "raw" else "E_NO_at"]
    return [E[PROBES.index(b)] - E[PROBES.index(a)] for a, b in ANN], r["kappa"]


def per_channel(run, key):
    out = []
    prev = None
    for x in run:
        E = np.array(x["E_raw_at" if key == "raw" else "E_NO_at"])
        d = E - prev if prev is not None else E
        out.append((x["kappa"], d[1] - d[0], d[2] - d[1]))
        prev = E
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--kshow", type=int, default=1000)
    a = ap.parse_args()
    recs = sorted([load(f) for f in a.records], key=lambda r: r["rs"])
    kmax = min(r["D"]["kappa_done"] for r in recs)
    trs = [t for t in recs[0]["D"]["running"] if all(t in r["D"]["running"] for r in recs)]
    L = ["## J-VF at two sizes (engine_vacflow_scale_v2.py over %s)" % ", ".join(r["file"] for r in recs), "",
         "Common channels kappa = 1..%d; every record's own ramp t_r = (2, 8, 32) r_s and hold 2 r_s; a scale-covariant chain gives the ratio %s at fixed R/r_s." % (
             kmax, ", ".join("%g/%g = %g" % (recs[0]["rs"], r["rs"], recs[0]["rs"] / r["rs"]) for r in recs[1:])), "",
         "(1) annulus energies after the hold, summed over kappa <= %d" % kmax, "",
         "| t_r/r_s | key | " + " | ".join("r_s %g: [%g, %g] r_s" % (r["rs"], x, y) for x, y in ANN for r in recs) + " |",
         "|---|---|" + "---|" * (len(ANN) * len(recs))]
    ratios = []
    for t in trs:
        for key in ("raw", "NO"):
            vals = {r["file"]: annuli(r["D"]["running"][t], kmax, key)[0] for r in recs}
            cells = []
            for j in range(len(ANN)):
                for r in recs:
                    cells.append("%.4e" % vals[r["file"]][j])
            L.append("| %s | %s | %s |" % (t, key, " | ".join(cells)))
            for r in recs[1:]:
                ratios.append((t, key, r["rs"], [vals[r["file"]][j] / vals[recs[0]["file"]][j] if vals[recs[0]["file"]][j] != 0 else float("nan") for j in range(len(ANN))]))
    L += ["", "| t_r/r_s | key | r_s ratio | " + " | ".join("[%g, %g] r_s" % ab for ab in ANN) + " | scale-covariant |", "|---|---|---|" + "---|" * len(ANN) + "---|"]
    for t, key, rs, rr in ratios:
        L.append("| %s | %s | %g/%g | %s | %g |" % (t, key, rs, recs[0]["rs"], " | ".join("%.3g" % x for x in rr), recs[0]["rs"] / rs))
    L += ["", "(2) per channel: the NO annulus energies [1.5, 2] and [2, 4] r_s after the hold (weights 16 kappa included)", ""]
    for t in trs:
        L += ["t_r = %s r_s" % t, "", "| kappa | " + " | ".join("r_s %g: [1.5, 2] | r_s %g: [2, 4]" % (r["rs"], r["rs"]) for r in recs) + " |",
              "|---|" + "---|---|" * len(recs)]
        pcs = {r["file"]: {k: (x, y) for k, x, y in per_channel(r["D"]["running"][t], "NO")} for r in recs}
        ks = min(a.kshow, kmax)
        for k in range(1, ks + 1):
            L.append("| %d | %s |" % (k, " | ".join("%.4e | %.4e" % pcs[r["file"]][k] for r in recs)))
        if ks < kmax:
            L.append("| %d..%d summed | %s |" % (ks + 1, kmax, " | ".join("%.4e | %.4e" % (sum(pcs[r["file"]][k][0] for k in range(ks + 1, kmax + 1)),
                     sum(pcs[r["file"]][k][1] for k in range(ks + 1, kmax + 1))) for r in recs)))
        L.append("")
    L += ["(3) the NO shell totals after the hold summed over kappa <= %d (all records' stored totals) at R/r_s" % kmax, "",
          "| t_r/r_s | r_s | " + " | ".join("%g" % x for x in XS) + " |", "|---|---|" + "---|" * len(XS)]
    for t in trs:
        for r in recs:
            if r["D"]["kappa_done"] != kmax:
                L.append("| %s | %g | (the stored totals sum %d channels, not %d: skipped) |" % (t, r["rs"], r["D"]["kappa_done"], kmax))
                continue
            tot = np.array(r["D"]["totals"][t]["NO_hold"])
            L.append("| %s | %g | %s |" % (t, r["rs"], " | ".join("%.3e" % tot[int(round(x * r["rs"])) - 1] for x in XS)))
    L += ["", "- every number above is printed by the script from the records' running channel sums and stored totals."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
