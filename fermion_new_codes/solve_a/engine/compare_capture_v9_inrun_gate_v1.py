#!/usr/bin/env python
"""compare_capture_v9_inrun_gate_v1.py (builder analyst): engine_capture_v9's in-run gate (its docstring: the propagation and every
reading are version 8's, so a version 9 run reproduces version 7's or 8's readings to the bit up to their stop): per background and probe,
every per-channel reading of the earlier run (A, R_out, inward_inside, inward_transit, beyond_Rmax, absorbed) against version 9's at the same
reading time, and the summed sigma at those times. Prints the counts and the verdict. Args: --earlier --v9."""
import argparse
import json

FIELDS = ("A", "R_out", "inward_inside", "inward_transit", "beyond_Rmax", "absorbed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--earlier", required=True); ap.add_argument("--v9", required=True)
    a = ap.parse_args()
    E = json.load(open(a.earlier)); N = json.load(open(a.v9))
    n = d = ns = ds = 0
    for re_, rn in zip(E["results"], N["results"]):
        assert (re_["background"], re_["probe"]) == (rn["background"], rn["probe"])
        cn = {c["kappa"]: c for c in rn["channels"]}
        for c in re_["channels"]:
            c9 = cn.get(c["kappa"])
            for s_ in ("plus", "minus"):
                for f in FIELDS:
                    for k, v in (c[s_].get(f) or {}).items():
                        n += 1
                        if c9 is None or (c9[s_].get(f) or {}).get(k) != v:
                            d += 1
        for k, v in re_["sigma"].items():
            ns += 1
            if rn["sigma"].get(k) != v:
                ds += 1
    print("per-channel readings compared: %d, differing: %d; summed sigma at the earlier run's reading times compared: %d, differing: %d" % (n, d, ns, ds))
    print("in-run gate: %s" % ("PASS" if d == 0 and ds == 0 else "FAIL"))


if __name__ == "__main__":
    main()
