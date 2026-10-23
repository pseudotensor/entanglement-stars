#!/usr/bin/env python
"""obs_J6_table_v1.py (builder analyst): J6's record (obs_J6_gpu_v1's json) printed for the lead's reading of 19.7 ("J6 read at hold 0
against the record's 0.786658 (C6, the quadrant quadrature), the hold-12 growth of the light channels reported as the planar profile's
overtilted region, its cutoff at m_c 0.34 the check, no coefficient from hold 12"): the assembled values per nk with C6 at hold 0 and at
hold 12 (the script's own column read hold 12), the per-channel S_A per area flat and formed at both holds with the relative excess, the
largest grid mass with a hold-12 excess above the hold-0 one by more than 1e-6 relative and the smallest without, the interpolation
checks with their pass flags, the ramp's step ladder. Args: --json --md."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    J = json.load(open(a.json))
    ar = J["args"]
    L = ["## J6's record for the reading (obs_J6_table_v1.py; obs_J6_gpu_v1.py, kappa a = 1/%g, N = %d, ramp %g at dt %g, holds %s, %d grid masses)" % (
        ar["inv_kappa"], ar["nsites"], ar["ramp_time"], ar["dt"], ar["holds"], ar["ngrid"]), "",
         "| nk | flat | formed, hold 0 | formed, hold 12 | hold 0 minus 0.786658 | hold 12 minus 0.786658 |", "|---|---|---|---|---|---|"]
    for nk, v in J["assembled"].items():
        fl = v["flat"] if not isinstance(v["flat"], list) else v["flat"][-1]
        fo = v["formed"]
        f0 = fo["0.0"] if not isinstance(fo["0.0"], list) else fo["0.0"][-1]
        f12 = fo["12.0"] if not isinstance(fo["12.0"], list) else fo["12.0"][-1]
        L.append("| %s | %.8f | %.8f | %.8f | %+.3e | %+.3e |" % (nk, fl, f0, f12, f0 - 0.786658, f12 - 0.786658))
    L += ["", "| m_c | flat | formed, hold 0 | excess, hold 0 | formed, hold 12 | excess, hold 12 |", "|---|---|---|---|---|---|"]
    last_grow = first_flat = None
    for q in J["grid"]:
        fl = q["flat"][-1]; f0 = q["formed"]["0.0"][-1]; f12 = q["formed"]["12.0"][-1]
        L.append("| %.4f | %.8f | %.8f | %+.3e | %.8f | %+.3e |" % (q["mc"], fl, f0, f0 / fl - 1.0, f12, f12 / fl - 1.0))
        if abs(f12 / f0 - 1.0) > 1e-6:
            last_grow = q["mc"]
        elif first_flat is None and last_grow is not None:
            first_flat = q["mc"]
    L += ["", "- the hold-12 growth (formed at hold 12 above formed at hold 0 by more than 1e-6 relative) at grid masses up to m_c %s; none from m_c %s on." % (
        ("%.4f" % last_grow) if last_grow else "-", ("%.4f" % first_flat) if first_flat else "-"), "",
          "| check m_c | flat | hold 0 | hold 12 | pass (1e-5) |", "|---|---|---|---|---|"]
    for c in J["checks"]:
        e = c["interp_abs_error"]
        L.append("| %.4f | %.1e | %.1e | %.1e | %s |" % (c["mc"], e["flat"], e["0.0"], e["12.0"], c["pass"]))
    L += ["", "| ladder m_c | dt | hold 0 | hold 12 |", "|---|---|---|---|"]
    for d in J["ladder"]:
        for dt in ("2.0", "1.0", "0.5"):
            v = d[dt]["formed"]
            L.append("| %.4f | %s | %.10f | %.10f |" % (d["mc"], dt, v["0.0"], v["12.0"]))
    L += ["", "- every number above is read from the run's json by the script."]
    open(a.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))


if __name__ == "__main__":
    main()
