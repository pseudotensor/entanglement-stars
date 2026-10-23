#!/usr/bin/env python
"""birth_v1.py (busi, GATECOMP (iii); ReviewA_gatecomp_v1 sec. 3.2): T_birth is the zero of m(T) = min_r(N - |v|), linear through the
two bracketing records; R_birth the minimum's radius r_min linearly interpolated to that clock. Errors from the bracketing records: the clock's
as |T_lin - T_quad| (T_quad the zero of the parabola through the bracketing pair and the record before), the radius's as half the change of
r_min across the bracket plus |R(T_lin) - R(T_quad)|. Reads the records json only (the dense min and r_min the engine stored); prints JSON."""
import json
import sys

import numpy as np


def birth(path, E0=None):
    J = json.load(open(path)); rs = float(J["args"]["rs"]); g = float(J["args"]["g"]); E0 = E0 or float(J["head"]["E0"])
    R = [r for r in J["records"] if r.get("E") is not None and r.get("min_N_minus_absv") is not None]
    T = np.array([r["T"] / rs for r in R]); m = np.array([r["min_N_minus_absv"] for r in R]); rm = np.array([r["r_min_N_minus_absv"] for r in R])
    i = int(np.nonzero(m < 0)[0][0]); lo, hi = i - 1, i
    Tl = T[lo] + (T[hi] - T[lo]) * m[lo] / (m[lo] - m[hi])
    c = np.polyfit(T[lo - 1:hi + 1], m[lo - 1:hi + 1], 2); rts = np.roots(c); rts = rts[np.isreal(rts)].real
    Tq = float(rts[np.argmin(np.abs(rts - Tl))])
    Rl = float(np.interp(Tl, T[lo:hi + 1], rm[lo:hi + 1])); Rq = float(np.interp(Tq, T[lo:hi + 1], rm[lo:hi + 1]))
    rec_hi = R[hi]
    two_gE_hi = 2.0 * g * rec_hi["E"]; two_gE_lo = 2.0 * g * R[lo]["E"]; two_gE_b = two_gE_lo + (two_gE_hi - two_gE_lo) * (Tl - T[lo]) / (T[hi] - T[lo])
    return {"json": path, "rs": rs, "ell": J["args"]["ell"], "E0": E0, "two_gE0": 2.0 * g * E0,
            "bracket": [[float(T[lo]), float(m[lo]), float(rm[lo])], [float(T[hi]), float(m[hi]), float(rm[hi])]],
            "record_before": [float(T[lo - 1]), float(m[lo - 1])],
            "T_birth": float(Tl), "T_birth_quadratic": Tq, "T_birth_err": abs(Tl - Tq),
            "R_birth": Rl, "R_birth_err": 0.5 * abs(rm[hi] - rm[lo]) + abs(Rl - Rq),
            "two_gE_at_birth": two_gE_b, "R_birth_over_2gE0": Rl / (2.0 * g * E0), "R_birth_over_2gE": Rl / two_gE_b,
            "first_root_record": {"T": float(T[hi]), "R_t": rec_hi.get("R_t"), "r_in": rec_hi.get("r_in"), "two_gE": two_gE_hi,
                                  "converged": rec_hi.get("converged")}}


if __name__ == "__main__":
    out = [birth(p) for p in sys.argv[1:]]
    print(json.dumps(out, indent=1))
