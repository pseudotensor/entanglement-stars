#!/usr/bin/env python
"""page_interior_init_v6s.py (busi lane, inbox #309): the lean interior mode's initial + sector vacuum Phi_+ = V_+[:, e_+ < 0] of h_+(T_0) computed
on the host and written to --init-cache, so that the GPU run (page_history_v6s.py --lean-interior --init-cache PATH) loads it (its h_+ md5 checked).
Usage: python page_interior_init_v6s.py <page_history_v6s.py arguments, with --device cpu --lean-interior --init-cache PATH>"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_history_v6s as V6S      # noqa: E402
import page_interior_v6s as IR      # noqa: E402


def main():
    a = V6S.parse(sys.argv[1:])
    if not a.lean_interior or not a.init_cache or not a.device.startswith("cpu"):
        raise SystemExit("needs --lean-interior --init-cache PATH --device cpu")
    a.channels = a.kappas.split("-")[0] if "-" in a.kappas else a.kappas.split(",")[0]
    a.tag = a.tag + "_init"
    run = V6S.HistRun(a)
    IR._start(run, {"log": V6S.log, "propagate": V6S.propagate, "tr_blocked": V6S.tr_blocked, "T_START": V6S.T_START})
    V6S.log("init done: %s" % a.init_cache)


if __name__ == "__main__":
    main()
