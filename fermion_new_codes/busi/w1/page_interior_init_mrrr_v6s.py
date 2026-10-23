#!/usr/bin/env python
"""page_interior_init_mrrr_v6s.py (busi lane, inbox #309 item 3): the same host initial vacuum as page_interior_init_v6s.py (Phi_+ = V_+[:, e_+ < 0]
of h_+(T_0), written to --init-cache with the same keys and the same h_+ md5), but with LAPACK zheevr over ALL levels (range 'A', which zheevr
solves by MRRR, zstemr) and the negative columns kept afterwards. The subset route (range 'V') makes zheevr fall back to bisection plus inverse
iteration (dstebz + zstein), whose reorthogonalisation inside the single cluster of 16384 close levels is single-threaded and O(n m^2); the
'A' route needs about 2 n^2 complex of memory (about 40 GB at n 32768) instead of 1.5 n^2. Nothing in page_history_v6s.py or
page_interior_v6s.py is changed: their module's scipy handle is wrapped for this process only, and the cache's info records the route.
Usage: python page_interior_init_mrrr_v6s.py <page_history_v6s.py arguments, with --device cpu --lean-interior --init-cache PATH>"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import page_history_v6s as V6S      # noqa: E402
import page_interior_v6s as IR      # noqa: E402

_SLA = IR.sla
ROUTE = "cpu (scipy zheevr range A = MRRR zstemr, negative columns kept; page_interior_init_mrrr_v6s.py)"


class _ShimSLA:
    """scipy.linalg with eigh(subset_by_value=[-inf, 0]) answered by the all-levels zheevr and a column selection; everything else passed on."""
    def __getattr__(self, name):
        return getattr(_SLA, name)

    @staticmethod
    def eigh(A, subset_by_value=None, driver=None, overwrite_a=False, check_finite=True, **kw):
        if subset_by_value is None or driver != "evr" or kw:
            return _SLA.eigh(A, subset_by_value=subset_by_value, driver=driver, overwrite_a=overwrite_a, check_finite=check_finite, **kw)
        lo, hi = subset_by_value
        e, Z = _SLA.eigh(A, driver="evr", overwrite_a=overwrite_a, check_finite=check_finite)
        keep = (e > lo) & (e <= hi)
        V6S.log("init_mrrr: all %d levels by zheevr range A; %d kept in (%g, %g]" % (len(e), int(keep.sum()), lo, hi))
        Zk = np.ascontiguousarray(Z[:, keep]); del Z
        return e[keep], Zk


_orig_init = IR._dense_init_plus


def _dense_init_plus_mrrr(hp, device, eig_device, threads, log):
    Phi, info = _orig_init(hp, device, eig_device, threads, log)
    if eig_device.type == "cpu":
        info["eig_device"] = ROUTE
    return Phi, info


def main():
    a = V6S.parse(sys.argv[1:])
    if not a.lean_interior or not a.init_cache or not a.device.startswith("cpu"):
        raise SystemExit("needs --lean-interior --init-cache PATH --device cpu")
    IR.sla = _ShimSLA()
    IR._dense_init_plus = _dense_init_plus_mrrr
    a.channels = a.kappas.split("-")[0] if "-" in a.kappas else a.kappas.split(",")[0]
    a.tag = a.tag + "_init_mrrr"
    run = V6S.HistRun(a)
    IR._start(run, {"log": V6S.log, "propagate": V6S.propagate, "tr_blocked": V6S.tr_blocked, "T_START": V6S.T_START})
    V6S.log("init done (MRRR route): %s" % a.init_cache)


if __name__ == "__main__":
    main()
