#!/usr/bin/env python
"""compare_capture_v9_inrun_gate_v3.py (builder analyst): version 2 with the verdict split by what is reproducible: the unitary
readings (A, R_out, inward_inside, inward_transit, beyond_Rmax) and the summed sigma bitwise; the absorbing port (absorbed) within --abs-tol
(1e-14), the run-to-run spread the C-capture version 7 builder measured between two serial runs of one version (9.437e-15,
lead/api/turn41/capture_v7_builder_report.md); nothing else changes. Version 2: version 1 with the differing readings counted per field with their largest
absolute difference (version 1 counted them only); nothing else changes. Version 1: engine_capture_v9's in-run gate (its docstring: the propagation and every
reading are version 8's, so a version 9 run reproduces version 7's or 8's readings to the bit up to their stop): per background and probe,
every per-channel reading of the earlier run (A, R_out, inward_inside, inward_transit, beyond_Rmax, absorbed) against version 9's at the same
reading time, and the summed sigma at those times. Prints the counts and the verdict. Args: --earlier --v9."""
import argparse
import json

FIELDS = ("A", "R_out", "inward_inside", "inward_transit", "beyond_Rmax", "absorbed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--earlier", required=True); ap.add_argument("--v9", required=True)
    ap.add_argument("--abs-tol", type=float, default=1e-14)
    a = ap.parse_args()
    E = json.load(open(a.earlier)); N = json.load(open(a.v9))
    n = d = ns = ds = 0
    per = {}
    for re_, rn in zip(E["results"], N["results"]):
        assert (re_["background"], re_["probe"]) == (rn["background"], rn["probe"])
        cn = {c["kappa"]: c for c in rn["channels"]}
        for c in re_["channels"]:
            c9 = cn.get(c["kappa"])
            for s_ in ("plus", "minus"):
                for f in FIELDS:
                    for k, v in (c[s_].get(f) or {}).items():
                        n += 1
                        w = None if c9 is None else (c9[s_].get(f) or {}).get(k)
                        if w != v:
                            d += 1
                            dd = abs(w - v) if (w is not None and v is not None) else float("inf")
                            q = per.setdefault(f, [0, 0.0]); q[0] += 1; q[1] = max(q[1], dd)
        for k, v in re_["sigma"].items():
            ns += 1
            if rn["sigma"].get(k) != v:
                ds += 1
    print("per-channel readings compared: %d, differing: %d; summed sigma at the earlier run's reading times compared: %d, differing: %d" % (n, d, ns, ds))
    for f, (c_, m_) in sorted(per.items()):
        print("  field %s: %d differing, largest absolute difference %.3e" % (f, c_, m_))
    unit = sum(c_ for f, (c_, m_) in per.items() if f != "absorbed")
    absm = per.get("absorbed", [0, 0.0])[1]
    ok = unit == 0 and ds == 0 and absm <= a.abs_tol
    print("in-run gate (version 3): the unitary readings %s (%d differing), the summed sigma %s, the absorbing port's largest difference %.3e against %.0e: %s"
          % ("bitwise" if unit == 0 else "not bitwise", unit, "bitwise" if ds == 0 else "not bitwise", absm, a.abs_tol, "PASS" if ok else "FAIL"))


if __name__ == "__main__":
    main()
