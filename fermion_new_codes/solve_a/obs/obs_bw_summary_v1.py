#!/usr/bin/env python3
"""obs_bw_summary_v1.py (builder analyst; B17 of SPEC_final4b_observables.md sec. 12.6, J2's reading of sec. 12.5): the three-size floor table
from J2's records (obs_bw_modular_v2 at L = 2048 and 4096, obs_bw_modular_v3's one row at L = 8192): per (profile, W, cut, m_c) present at
the sizes, R_in(prediction), R_in(uniform), the v = 0 reference R_0 at the same L and m_c, the ratio R_in(pred)/R_0 and the separation
(R_in(uniform) - R_in(pred))/R_0; the rule of observables 12.5: (ii) is "the long-wavelength modular Hamiltonian" only if R_in(pred)/R_0
falls toward 1 over 2048, 4096, 8192. Args: --records (json files) --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = {}
    refs = {}
    for f in a.records:
        D = json.load(open(f))
        for c in D["cases"]:
            key_ref = (c["L"], c["mc"])
            if c["label"].startswith("v = 0"):
                refs[key_ref] = c["uniform"]["R_in"]
                continue
            k = (c["label"], c["mc"])
            rows.setdefault(k, {})[c["L"]] = c
    Ls = sorted({L for v in rows.values() for L in v})
    L_ = ["## J2: the three-size floor table (obs_bw_summary_v1.py over %s)" % ", ".join(x.split("/")[-1] for x in a.records), "",
          "| case | m_c | " + " | ".join("L = %d: R_in pred, R_0, pred/R_0, separation" % L for L in Ls) + " |", "|---|---|" + "---|" * len(Ls)]
    for (lab, mc), byL in sorted(rows.items(), key=lambda t: (t[0][1], t[0][0])):
        cells = []
        for L in Ls:
            c = byL.get(L)
            R0 = refs.get((L, mc))
            if c is None or R0 is None:
                cells.append("-")
                continue
            p = c["pred"]["R_in"]; u = c["uniform"]["R_in"]
            cells.append("%.3e, %.3e, %.4f, %+.3f" % (p, R0, p / R0, (u - p) / R0))
        L_.append("| %s | %g | %s |" % (lab, mc, " | ".join(cells)))
    L_ += ["", "- every number above is printed by the script from the records; the reading's rule is the lead's (observables 12.5)."]
    open(a.md, "w").write("\n".join(L_) + "\n")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
