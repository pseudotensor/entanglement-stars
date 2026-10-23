#!/usr/bin/env python
"""test_w10_defect9_valleys.py (busi W10): defect 9 of REVIEWA_REVIEW_web40i section 5. The literal k > 0 return selector (spec line 263)
misses the second outgoing valley. Gate: the half-chain control vanishing at n = 0.
The chain's bond symbol is sin k sigma_x: on the (1, 1)/sqrt 2 spinor omega = sin k and the group velocity cos k, on (1, -1)/sqrt 2
omega = -sin k and the velocity -cos k; the outgoing modes are the (1, 1) content at |k| < pi/2 and the (1, -1) content at |k| > pi/2.
(1) the conventions checked against the operator itself: packets propagated by the dense exponential of the massless chain move at
    s cos k0 (four cases, both spinors, both valleys); (2) the half-chain control: every ingoing mode filled (the (1, -1) content at
    |k| < pi/2, the (1, 1) content at |k| > pi/2, the boundary bins at cos k = 0 left empty), the outgoing-selected number must vanish:
    w10_lib_v1.outgoing_weights gives n = 0 exactly, the literal k > 0 selector about half the filling; (3) a state in the (1, -1) second
    valley at k < 0 is outgoing in full; the literal selector counts none of it.
Run: qi2 python test_w10_defect9_valleys.py"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main():
    # (1) the velocities from the operator
    n = 400
    R = np.arange(n, dtype=float) + 1.0
    h0, _Q = W10.chain_sparse_ref(R, 0.0, 0.0)                 # massless, kappa 0: the bonds alone, symbol sin k sigma_x
    e, U = np.linalg.eigh(h0.toarray())
    R2 = np.repeat(R, 2)

    def packet(s, k0, n0=200, width=15.0):
        sites = np.arange(n)
        env = np.exp(-((sites - n0) / (2.0 * width)) ** 2) * np.exp(1j * k0 * sites)
        psi = np.zeros(2 * n, complex); psi[0::2] = env / math.sqrt(2.0); psi[1::2] = s * env / math.sqrt(2.0)
        return psi / np.linalg.norm(psi)

    def centroid(psi):
        return float(np.sum(R2 * np.abs(psi) ** 2))
    t = 60.0
    ok = True
    for s, k0, label in ((+1, 0.4, "(1, 1) first valley, k 0.4"), (-1, 0.4, "(1, -1) first valley, k 0.4"),
                         (-1, math.pi - 0.4, "(1, -1) second valley, k pi - 0.4"), (+1, math.pi - 0.4, "(1, 1) second valley, k pi - 0.4")):
        psi = packet(s, k0)
        shift = (centroid(U @ (np.exp(-1j * e * t) * (U.conj().T @ psi))) - centroid(psi)) / t
        expect = s * math.cos(k0)
        good = abs(shift - expect) < 0.02
        ok = ok and good
        print("  %s: measured velocity %+.4f, s cos k0 = %+.4f (%s)" % (label, shift, expect, "outgoing" if expect > 0 else "ingoing"))
    check(ok, "the velocities from the operator: v = s cos k on both spinors; the outgoing modes are (1, 1) at |k| < pi/2 and (1, -1) at |k| > pi/2 (two valleys)")
    # (2) the half-chain control in momentum space
    N = 2048
    k = 2.0 * math.pi * np.fft.fftfreq(N)
    c = np.cos(k)
    w_minus = (c > 1e-9).astype(float)                          # (1, -1): ingoing at cos k > 0, the boundary bins at +-pi/2 left empty
    w_plus = (c < -1e-9).astype(float)                          # (1, 1): ingoing at cos k < 0
    tot = float(w_minus.sum() + w_plus.sum())
    lit = W10.literal_k_positive(k)
    n_literal = float((lit * (w_plus + w_minus)).sum())
    wo_plus, wo_minus = W10.outgoing_weights(k)
    n_both = float((wo_plus * w_plus + wo_minus * w_minus).sum())
    check(n_both == 0.0, "gate (the half-chain control vanishing at n = 0): every ingoing mode of both valleys filled (%d modes): the both-valleys selector returns n = %.1f exactly" % (int(tot), n_both))
    check(n_literal > 0.4 * tot, "the literal k > 0 selector on the same control returns n = %d of %d (the (1, -1) first valley at k in (0, pi/2) and the (1, 1) second valley at k in (pi/2, pi) counted as returning)" % (int(n_literal), int(tot)))
    # (3) the second outgoing valley at k < 0
    w2 = ((k < -math.pi / 2 - 1e-9)).astype(float)              # (1, -1) at k in (-pi, -pi/2): outgoing
    n_lit2 = float((lit * w2).sum()); n_both2 = float((wo_minus * w2).sum())
    check(n_lit2 == 0.0 and n_both2 == float(w2.sum()), "a state in the (1, -1) second valley at k < 0 (%d modes, all outgoing): the literal selector counts %d, the both-valleys selector %d" % (int(w2.sum()), int(n_lit2), int(n_both2)))
    wi_plus, wi_minus = W10.ingoing_weights(k)
    check(np.array_equal(wi_plus + wo_plus, np.ones(N)) and np.array_equal(wi_minus + wo_minus, np.ones(N)), "ingoing plus outgoing weights sum to 1 on every bin for both spinors (the half-bin convention at cos k = 0)")
    print("test_w10_defect9_valleys: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
