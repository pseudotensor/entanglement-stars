#!/usr/bin/env python3
"""obs_spline_ring_v1.py (builder analyst): the solver's parity quintic spline (engine_fields_v3.parity_spline's construction) through node
values at spacing ell of the smooth-core function v = -sqrt(r_s) r (r^2 + R_c^2)^(-3/4), against the function: the value error and the ratio of
the first three derivatives, at radii from the core outward (the ringing of an under-resolved core at the node period). Args: --rs --Rc
--ell (list) --out-md."""
import argparse

import numpy as np
from scipy.interpolate import make_interp_spline


def parity_spline(r_nodes, y_nodes, parity):
    rr = np.concatenate([-r_nodes[:0:-1], r_nodes]); yy = np.concatenate([parity * y_nodes[:0:-1], y_nodes])
    return make_interp_spline(rr, yy, k=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs", type=float, default=64.0)
    ap.add_argument("--Rc", type=float, nargs="+", default=[4.0, 12.0])
    ap.add_argument("--ell", type=float, nargs="+", default=[4.0, 8.0])
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    L = ["## The parity quintic spline through node values against the smooth-core function (obs_spline_ring_v1.py; r_s = %g)" % a.rs, "",
         "| R_c | ell | r | (spline - f)/|f| | d1 spline/f | d2 spline/f | d3 spline/f |", "|---|---|---|---|---|---|---|"]
    for Rc in a.Rc:
        f = lambda r: -np.sqrt(a.rs) * np.asarray(r, float) * (np.asarray(r, float) ** 2 + Rc ** 2) ** (-0.75)
        def fd(q, x, h=1e-2):
            if q == 1: return (f(x + h) - f(x - h)) / (2 * h)
            if q == 2: return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
            return (f(x + 2 * h) - 2 * f(x + h) + 2 * f(x - h) - f(x - 2 * h)) / (2 * h ** 3)
        for ell in a.ell:
            rn = np.arange(0.0, 64 * a.rs, ell)
            s = parity_spline(rn, f(rn), -1)
            for r in (1.5 * ell, 2.5 * ell, 3.5 * ell, 5.5 * ell, 8.5 * ell, 0.5 * a.rs, a.rs, 2 * a.rs):
                L.append("| %g | %g | %g | %.2e | %.6f | %.6f | %.6f |" % (Rc, ell, r, float((s(r) - f(r)) / abs(f(r))),
                         float(s.derivative(1)(r) / fd(1, r)), float(s.derivative(2)(r) / fd(2, r)), float(s.derivative(3)(r) / fd(3, r))))
    L += ["", "- every number above is printed by the script."]
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
