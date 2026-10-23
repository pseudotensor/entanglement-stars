#!/usr/bin/env python
"""engine_vacflow_compare_v1.py (builder analyst; engine 16.6's measurement "the full chain current's effect on E_NO and E_raw per annulus"):
two or more J-VF records compared at one channel count and one ramp from their running sums: E_raw and E_NO at the probes (1.5, 2, 4, 8, 16 r_s),
the annular differences between the probes, and (when both records carry them, engine_vacflow_v6) the weighted inside integrals summed to that
channel count. Args: records... --kappa --tr --md."""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", nargs="+")
    ap.add_argument("--kappa", type=int, default=16)
    ap.add_argument("--tr", default="2")
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    L = ["## J-VF records compared at kappa <= %d, t_r = %s r_s (engine_vacflow_compare_v1.py)" % (a.kappa, a.tr), "",
         "| record | chain current | key | at 1.5, 2, 4, 8, 16 r_s | annuli [1.5, 2], [2, 4], [4, 8], [8, 16] r_s | inside (R < r_s), weighted, summed |", "|---|---|---|---|---|---|"]
    for f in a.records:
        D = json.load(open(f))
        run = D["running"].get(a.tr)
        if run is None:
            L.append("| %s | - | - | ramp %s absent | - | - |" % (os.path.basename(f), a.tr))
            continue
        row = [r for r in run if r["kappa"] == a.kappa]
        if not row:
            L.append("| %s | - | - | kappa %d absent | - | - |" % (os.path.basename(f), a.kappa))
            continue
        r = row[0]
        ins = {}
        for c in D.get("per_channel", []):
            if c["kappa"] <= a.kappa and c.get("inside_weighted"):
                for key, val in c["inside_weighted"].get(a.tr, {}).items():
                    ins[key] = ins.get(key, 0.0) + val
        cc = D["args"].get("chain_current") or "two-step (default)"
        for key, lab in (("E_raw_at", "raw_hold"), ("E_NO_at", "NO_hold")):
            v = r[key]
            ann = [v[i + 1] - v[i] for i in range(len(v) - 1)]
            L.append("| %s | %s | %s | %s | %s | %s |" % (os.path.basename(f), cc, key.split("_")[1], ", ".join("%.4f" % x for x in v), ", ".join("%.4f" % x for x in ann),
                                                         ("%.4f" % ins[lab]) if lab in ins else "-"))
    L += ["", "- every number above is printed by the script from the records' running sums and per-channel inside integrals."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
