#!/usr/bin/env python
"""engine_diag_turn27_dump_v1.py (builder analyst; spec 19.3's diagnosis, step 1): runs engine_collapse_v17 unchanged on a configuration
with its solve wrapped: every solve's site lapse and flow are kept per re-solve (so the radius and the sign pattern of the alternating dN
are read off), and the first failed solve's shell totals (eps_tot, J_tot) and the step's solve history are saved to --dump before the
engine records the rejection. The engine's own arguments follow "--"; this script adds nothing to the physics.
Usage: python engine_diag_turn27_dump_v1.py --dump <npz> -- <engine_collapse_v17 arguments>."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine_collapse_v17 as E17  # noqa: E402

i = sys.argv.index("--")
own = sys.argv[1:i]
dump = own[own.index("--dump") + 1]
sys.argv = [sys.argv[0]] + sys.argv[i + 1:]
_orig = E17.solve
HIST = {"N": [], "v": [], "E": [], "calls": 0}


def solve_wrapped(eps, J, g, R, ell, N0, rtol=1e-12, atol=1e-16):
    out = _orig(eps, J, g, R, ell, N0, rtol=rtol, atol=atol)
    HIST["calls"] += 1
    info = out[3]
    if info.get("failed"):
        np.savez(dump, eps=np.asarray(eps, float), J=np.asarray(J, float), g=g, R=np.asarray(R, float), ell=ell, failed=str(info["failed"]),
                 N_hist=np.array(HIST["N"]), v_hist=np.array(HIST["v"]), E_hist=np.array(HIST["E"]), calls=HIST["calls"])
        print("DUMP: the failed solve's sources and %d earlier solves' fields saved to %s (%s)" % (len(HIST["N"]), dump, info["failed"]), flush=True)
    else:
        HIST["N"].append(np.asarray(out[0], float).copy()); HIST["v"].append(np.asarray(out[1], float).copy()); HIST["E"].append(float(info["E_collar"]))
        np.savez(dump.replace(".npz", "_hist.npz"), N_hist=np.array(HIST["N"]), v_hist=np.array(HIST["v"]), E_hist=np.array(HIST["E"]), R=np.asarray(R, float))
    return out


E17.solve = solve_wrapped
E17.main()
