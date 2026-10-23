#!/usr/bin/env python
"""c9_cost_v1.py (busi; inbox #485 optional): zero-card cost estimate of C9 (ReviewA_bpchk2_v1 sec. 1.4: project psi(t) on the
2 L shell |r - 2L| <= a/2 onto the spinor spherical harmonics Omega_kappa_m, |kappa| <= 10, both signs, all m). Counts shell sites on the
actual s = 16 cube (half-side 2.5 L, open, 81^3 sites) and the s = 8 test cube, and prices the projection against MEASURED step costs read
from the run jsons. Nothing is propagated; no card. Emits the note whole."""
import argparse
import json
import math
import os

import numpy as np


def shell_sites(s, half_side_L=2.5, r_shell_L=2.0):
    n = int(round(half_side_L * s))
    x = np.arange(-n, n + 1, dtype=float)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y + Z * Z)
    return int((np.abs(R - r_shell_L * s) <= 0.5).sum()), (2 * n + 1) ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json2c", required=True)
    ap.add_argument("--jsonx", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    j2c = json.load(open(a.json2c)); jx = json.load(open(a.jsonx))
    kmax = 10
    nch = sum(2 * (2 * k) for k in range(1, kmax + 1))  # kappa = +-k, j = k - 1/2, 2j + 1 = 2k values of m
    out = []; P = out.append
    P("# C9 (the 2 L shell projection) zero-card cost estimate, version 1 (busi lane; optional item of inbox #485)")
    P("")
    P("Emitted whole by busi/code/pagetests/v3/c9_cost_v1.py; nothing propagated, no card. C9 is NOT authorized for a card (#485); this is a price only. Spec: ReviewA_bpchk2_v1 sec. 1.4.")
    P("")
    P("## 1. Sizes")
    P("")
    P("| s = L/a | cube sites | shell sites (\\|r - 2L\\| <= a/2) | shell amplitudes (x 4 spinor) | channels (\\|kappa\\| <= %d, both signs, all m) | projector entries | projector memory complex128 (MB) |" % kmax)
    P("|---|---|---|---|---|---|---|")
    sz = {}
    for s in (8, 16, 32):
        ns_, nc = shell_sites(s)
        ent = nch * 4 * ns_
        sz[s] = (ns_, nc)
        P("| %d | %d | %d | %d | %d | %d | %.1f |" % (s, nc, ns_, 4 * ns_, nch, ent, ent * 16 / 1e6))
    P("")
    # cost per sample at s = 16, 256 columns
    ns16 = sz[16][0]
    cols = 256
    flops = 8.0 * nch * 4 * ns16 * cols  # complex MAC = 8 real flops
    tmx = jx["timing"]; t2c = j2c["timing"]
    step_s = tmx["step_median_s"]; dt = tmx["dt"]; batch = min(tmx["col_batch"], jx["K"])
    L = 16.0
    steps_per_L = L / dt
    P("## 2. Overhead at s = 16 (256 columns, sampled every 1 L, as specified)")
    P("")
    P("Projection per sample: %d channels x %d shell amplitudes x %d columns = %.3g real flops (a dense complex GEMM after a gather of the shell sites)." % (nch, 4 * ns16, cols, flops))
    P("")
    P("| assumed sustained fp64 rate | seconds per sample (256 columns) | measured propagation per 1 L, 256 columns (s) | overhead |")
    P("|---|---|---|---|")
    prop_per_L = step_s * steps_per_L * (cols / batch)
    for eff in (1e12, 1e13):
        tsamp = flops / eff
        P("| %.0e flop/s | %.3g | %.0f | %.2g |" % (eff, tsamp, prop_per_L, tsamp / prop_per_L))
    P("")
    P("Measured propagation: x_full step median %.3f s for a batch of %d columns at dt %s a (json timing), so %.0f steps and %.0f s per 1 L for 256 columns. The dt 0.25 a arm (2c) step median is %.3f s." % (step_s, batch, dt, steps_per_L, prop_per_L, t2c["step_median_s"]))
    P("")
    P("## 3. Storage")
    P("")
    P("| run length | samples (every 1 L) | a_kappa_m(t) for 256 columns, complex128 (GB) |")
    P("|---|---|---|")
    for T in (100, 300, 1000):
        P("| %d L | %d | %.3f |" % (T, T + 1, (T + 1) * nch * cols * 16 / 1e9))
    P("")
    P("## 4. The three tests (sec. 1.4), priced")
    P("")
    ns8, nc8 = sz[8]
    P("- (i) Gram of the discretized Omega's on the s = 16 shell (a %d x %d Hermitian from a %d x %d projector): one GEMM, box CPU seconds; orthonormalize if off by more than 1e-2." % (nch, nch, nch, 4 * ns16))
    P("- (ii) s = 8 vacuum box, outgoing kappa = 1 packet: one column on a %d-site cube (%d shell sites), to the packet's transit of the shell (a few L). At the measured s = 16 rate (%.4f s per column per a) scaled by the site ratio %.3f, with 8 a per L at s = 8, one column costs about %.2f s per L on an H100-class device (a rough scaling; small cubes run below peak efficiency); on the box (CPU or its own GPU) minutes to an hour for a few L. Pass: norm recovered to 1e-3, leakage into kappa != 1 below 1e-3." % (
        nc8, ns8, step_s / (batch * dt), nc8 / sz[16][1], step_s / (batch * dt) * (nc8 / sz[16][1]) * 8))
    P("- (iii) Sum over kappa of G_kappa against G on the F1 smoke: the only test that needs the s = 16 engine with projection hooks. A smoke of 32 columns to 10 L at dt 0.5 a costs %.2f h at the measured rate (32 x 160 a x %.4f s per column per a); this is the one piece that would want a card (the item is not authorized for one), or the box GPU at a lower rate." % (32 * 160 * step_s / (batch * dt) / 3600, step_s / (batch * dt)))
    P("")
    P("## 5. Reading")
    P("")
    P("The projection itself is cheap: a dense GEMM per 1 L sample costs a small fraction of the propagation at s = 16 at either assumed rate (sec. 2), and the stored amplitudes are under a gigabyte per 1e3 L run (sec. 3), in line with reviewer A's sec. 1.4 estimate. The cost of C9 is engineering, not card time: the lattice spinor harmonics and the gather, the hook in the engine (one call every L/dt steps, 32 at s = 16 and dt 0.5 a), and tests (i) to (iii), with only (iii) needing the s = 16 engine. Nothing here changes the ruling that C9 waits for authorization.")
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
