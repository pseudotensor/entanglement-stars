#!/usr/bin/env python
"""test_w10_defect5_halfbin.py (busi W10): defect 5 of REVIEWA_REVIEW_web40i section 5. The strict FFT-bin masks of
engine_collapse_v36d.prepare_massless_ball_v33 (sin k < 0; |sin k| > sin(pi/4)) give the boundary bins weight 0 or 1 by the rounding
of the test quantity. Gate: exactly 1/2 on the test.
(1) the quoted number: the strict sin k < 0 mask on a grid with the boundary bin at k = +pi counts 1023 of 2048 bins, 0.49951;
(2) this lane's grid (2 pi fftfreq(2048)) has it at k = -pi, where np.sin(-np.pi) = -1.2e-16 < 0: the strict mask reads 1024/2048 = 1/2
    on the flat spectrum by accident while the k = 0 bin is dropped; the quarter-zone strict mask reads 1022/2048 = 0.49902;
(3) the half-bin convention (w10_lib_v1.half_bin_mask): exactly 0.5 for the flat spectrum in both readouts and in the velocity readout
    of defect 9, 0.5 to 1e-13 for a k-symmetric packet (whose strict readout is off by (w_0 - w_-pi)/2);
(4) the S11 chiral seeds (P28 sec. 5.1's formula at the pathfinder's parameters): the change of the two head diagnostics printed.
Run: qi2 python test_w10_defect5_halfbin.py"""
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


def strict_kneg(k, w):
    return float(w[np.sin(k) < 0].sum())


def strict_quarter(k, w):
    return float(w[np.abs(np.sin(k)) > math.sin(math.pi / 4)].sum())


def halfbin_kneg(k, w):
    return float((w * W10.half_bin_mask(np.sin(k), 0.0, "below")).sum())


def halfbin_quarter(k, w):
    return float((w * W10.half_bin_mask(np.abs(np.sin(k)), math.sin(math.pi / 4), "above")).sum())


def readouts(psi0, psi1, k):
    """the engine's per-orbital readouts from the two spinor components: strict (v33) and half-bin (v36e)."""
    c0 = np.fft.fft(psi0); c1 = np.fft.fft(psi1)
    inw = 0.5 * np.abs(c0 - c1) ** 2
    tot = float((np.abs(c0) ** 2 + np.abs(c1) ** 2).sum())
    spin_tot = np.abs(c0) ** 2 + np.abs(c1) ** 2
    return {"kneg_strict": strict_kneg(k, inw) / tot, "kneg_half": halfbin_kneg(k, inw) / tot,
            "quarter_strict": strict_quarter(k, spin_tot) / tot, "quarter_half": halfbin_quarter(k, spin_tot) / tot}


def main():
    N = 2048
    k_lane = 2.0 * math.pi * np.fft.fftfreq(N)                                   # this lane's grid: the boundary bin at k = -pi
    k_plus = 2.0 * math.pi * np.arange(-N // 2 + 1, N // 2 + 1) / N              # the quoted test's grid: the boundary bin at k = +pi
    flat = np.full(N, 1.0 / N)
    x = strict_kneg(k_plus, flat)
    check(abs(x - 1023.0 / 2048.0) < 1e-15, "the quoted number reproduced: the strict sin k < 0 mask on the grid with the boundary bin at k = +pi counts 1023 of 2048 bins: %.5f" % x)
    y = strict_kneg(k_lane, flat)
    check(y == 0.5 and np.sin(k_lane)[N // 2] < 0.0, "this lane's grid (2 pi fftfreq): np.sin(-np.pi) = %.3e < 0 counts the k = -pi bin, so the strict mask reads %.5f on the flat spectrum by the rounding accident while the k = 0 bin (sin exactly 0) is dropped" % (np.sin(k_lane)[N // 2], y))
    q = strict_quarter(k_lane, flat)
    check(abs(q - 1022.0 / 2048.0) < 1e-15, "the strict quarter-zone mask drops its 4 boundary bins (n = +-256, +-768): %.5f on the flat spectrum (the exact measure is 1/2)" % q)
    h1 = halfbin_kneg(k_lane, flat); h2 = halfbin_kneg(k_plus, flat); h3 = halfbin_quarter(k_lane, flat)
    check(h1 == 0.5 and h2 == 0.5 and h3 == 0.5, "gate (exactly 1/2 on the test): the half-bin convention gives exactly %.17g (k < 0, this grid), %.17g (k < 0, the quoted grid) and %.17g (quarter zone) on the flat spectrum" % (h1, h2, h3))
    w_in_plus, w_in_minus = W10.ingoing_weights(k_lane)
    vel = float((0.5 * flat * w_in_plus + 0.5 * flat * w_in_minus).sum())
    check(vel == 0.5, "defect 9's velocity readout on the flat spectrum (half on each spinor): exactly %.17g" % vel)
    # a k-symmetric packet on the (1, -1) spinor (real envelope: |c(k)|^2 = |c(-k)|^2)
    n = np.arange(N)
    env = np.exp(-((n - 1024.0) / 40.0) ** 2)
    r = readouts(env / math.sqrt(2.0), -env / math.sqrt(2.0), k_lane)
    c0 = np.fft.fft(env); w = np.abs(c0) ** 2 / float((np.abs(c0) ** 2).sum())
    dev_expected = -(w[0] - w[N // 2]) / 2.0
    check(abs(r["kneg_half"] - 0.5) < 1e-13 and abs(r["kneg_strict"] - 0.5) > 1e-6 and abs((r["kneg_strict"] - 0.5) - dev_expected) < 1e-12,
          "a k-symmetric packet: half-bin %.15f, strict %.6f = 1/2 - (w_0 - w_-pi)/2 (w_0 %.4e: the k = 0 bin dropped)" % (r["kneg_half"], r["kneg_strict"], w[0]))
    # the S11 chiral seeds at the pathfinder's parameters (r_s 16: R0 64, x0 32, sigma 16, k0 0.1)
    R = n + 1.0
    x0, sig, k0 = 32.0, 16.0, 0.1
    dmax_k = 0.0; dmax_q = 0.0
    for j in range(6):
        xx = R - x0
        psi = math.sqrt(sig / math.pi) * (sig - 1j * xx) ** j / (sig + 1j * xx) ** (j + 1) * np.exp(-1j * k0 * xx)
        rr = readouts(psi / math.sqrt(2.0), -psi / math.sqrt(2.0), k_lane)
        dmax_k = max(dmax_k, abs(rr["kneg_half"] - rr["kneg_strict"])); dmax_q = max(dmax_q, abs(rr["quarter_half"] - rr["quarter_strict"]))
    print("INFO the S11 chiral seeds j = 0..5 (x0 32, sigma 16, k0 0.1 on 2048 sites): the head's weight_ingoing_spinor_k_negative changes by at most %.2e and weight_above_quarter_zone by at most %.2e between the strict and the half-bin convention (the only numbers the fix moves; none in the evolution)" % (dmax_k, dmax_q))
    print("test_w10_defect5_halfbin: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
