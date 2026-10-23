#!/usr/bin/env python
"""packet_v1_dispersion_check.py (busi lane, reviewer A slot, inbox #354): the group velocity of a packet under the engine's clocked chain
h_m = (1/2){N, h0} + (1/2){v, Q} on UNIFORM fields (N0, v0), measured as d<r>/dT over a short run, against the dispersion
    omega(k) = N0 eps(k) + v0 sin(2k)/2,   d omega/dk = N0 sin k cos k / eps(k) + v0 cos(2k),   eps = sqrt(m^2 + sin^2 k),
for packets at several k0, far from the centre (kappa/r negligible). The point checked: beyond |k| = pi/4 the flow coupling cos 2k of the
two-step current reverses sign, so a packet at the zone boundary |k| -> pi/2 moves OUTWARD at |v0| in an infall v0 < 0.
Usage: python packet_v1_dispersion_check.py --out FILE.json [--N0 0.5] [--v0 -0.6]
"""
import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packet_v1_common as PC  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--N0", type=float, default=0.5); ap.add_argument("--v0", type=float, default=-0.6)
    ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--n", type=int, default=4096); ap.add_argument("--kappa", type=int, default=1)
    a = ap.parse_args()
    n = a.n; R = np.arange(n, dtype=float) + 1.0; m = a.m
    h0, Q = PC.chain(R, m, a.kappa)
    N = np.full(n, a.N0); v = np.full(n, a.v0)
    Hm = PC.clocked(h0, Q, N, v)
    r0 = 2000.0; sigma = 12.0; Tend = 60.0; nstep = 12
    rows = []
    for k0 in [0.05, 0.1, 0.2, 0.3, 0.45, 0.6, np.pi / 4, 1.0, 1.2, 1.4, np.pi / 2 - 0.05, -0.45, -1.2, -(np.pi / 2 - 0.05)]:
        g = np.exp(-(R - r0) ** 2 / (4.0 * sigma ** 2)); s = np.sin(k0); e = np.sqrt(m * m + s * s)
        u = np.array([e + m, s], complex); u /= np.linalg.norm(u)
        psi = ((g * np.exp(1j * k0 * R))[:, None] * u[None, :]).reshape(-1)
        # keep the positive band of the uniform chain by filtering with h0 (a short imaginary-time-free route: project with the sign of <h0>?);
        # here the plane-wave spinor is already the positive-energy spinor of m s3 + s1 sin k at k0; the negative-band admixture is O(sigma^-1).
        psi /= np.linalg.norm(psi)
        rm = [float(np.sum(R * PC.site_density(psi, n)))]
        for _ in range(nstep):
            psi = expm_multiply(-1j * (Tend / nstep) * Hm, psi, traceA=0.0)
            rm.append(float(np.sum(R * PC.site_density(psi, n))))
        vmeas = (rm[-1] - rm[0]) / Tend
        vth = a.N0 * s * np.cos(k0) / e + a.v0 * np.cos(2.0 * k0)
        rows.append({"k0": float(k0), "eps": float(e), "v_measured": vmeas, "v_dispersion": float(vth), "N0_eps_prime": float(a.N0 * s * np.cos(k0) / e), "v0_cos2k": float(a.v0 * np.cos(2.0 * k0)),
                     "norm": float(np.linalg.norm(psi) ** 2)})
        print("k0 %+.3f: measured d<r>/dT %+.4f, dispersion %+.4f (N0 eps' %+.4f, v0 cos 2k %+.4f)" % (k0, vmeas, vth, rows[-1]["N0_eps_prime"], rows[-1]["v0_cos2k"]), flush=True)
    json.dump({"N0": a.N0, "v0": a.v0, "m": m, "kappa": a.kappa, "r0": r0, "sigma": sigma, "Tend": Tend, "rows": rows, "script_md5": PC.md5(os.path.abspath(__file__))}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
