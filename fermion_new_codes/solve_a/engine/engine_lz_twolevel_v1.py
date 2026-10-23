#!/usr/bin/env python
"""engine_lz_twolevel_v1.py (builder analyst; engine spec 25.1: "A two-level check (CPU, minutes; BUILDER: the exact sudden-plus-frozen
two-level propagator over a swept crossing at the production's rate and gap against the convention's assignment) is the convention's
labelled model test"): h(t) = [[a t, d], [d, -a t]] (adiabatic gap 2 sqrt(a^2 t^2 + d^2), minimum 2d at t = 0; the record-to-record change
dh = diag(a tau, -a tau), so the convention's Gershgorin rate s_T = ||dh||_G/tau = a). Records at t_k = t_off + k tau, k = -K..K (K = 64
by default, a K tau >= 50 d checked), t_off over --offsets values in [0, tau) (the sampling phase of the crossing). (1) The exact
sudden-plus-frozen step: the state, the lower eigenvector of h(t_-K), propagated psi <- exp(-i h(t_k) tau) psi record by record (the
engine's step: frozen h for tau, a jump at the record); at the end the probability P_up of the upper eigenvector of h(t_K). (2) The
convention (engine_ops_v3 with c_LZ): the reference starts on the lower eigenvector; at each record tau_LZ = 1/(c_LZ sqrt(a)); if tau_LZ
<= tau, nsub = max(1, round(c_LZ sqrt(tau ||dh||_G))) sub-steps along the linear path, the eigenvector of larger weight on the previous
reference assigned at each; else one step at the record; at the end "diabatic" (the reference on the upper eigenvector) or "adiabatic".
(3) The continuous Landau-Zener probability P_LZ = exp(-pi d^2/a). Per (a, d): P_up's range over the offsets, P_LZ, the fraction of offsets
on which the convention ends diabatic, and the agreement count (the convention diabatic where P_up > 1/2, adiabatic where P_up < 1/2);
on a grid spanning the production's pocket-phase rates and gaps (--a-list, --d-list; the production's records' s_T 1.9e-3 to 2.1e-2 and
inter-block gaps 1.9e-5 to 1.3e-2, its minimum gap no larger than the recorded gap). Args: --tau --K --offsets --c-LZ --a-list --d-list --md."""
import argparse
import math

import numpy as np


def eig2(a, d, t):
    h = np.array([[a * t, d], [d, -a * t]], float)
    e, U = np.linalg.eigh(h)
    return e, U, h


def step_exact(psi, h, tau):
    e, U = np.linalg.eigh(h)
    return U @ (np.exp(-1j * e * tau) * (U.conj().T @ psi))


def run_case(a, d, tau, K, t_off, c_LZ):
    ts = [t_off + k * tau for k in range(-K, K + 1)]
    e0, U0, h0 = eig2(a, d, ts[0])
    psi = U0[:, 0].astype(complex)
    ref = U0[:, 0].astype(complex)
    nG = a * tau
    tau_LZ = 1.0 / (c_LZ * math.sqrt(a)) if a > 0 else math.inf
    for k in range(len(ts) - 1):
        _, _, hk = eig2(a, d, ts[k])
        psi = step_exact(psi, hk, tau)
        if tau_LZ <= tau:
            nsub = max(1, int(round(c_LZ * math.sqrt(tau * nG))))
            for j in range(1, nsub + 1):
                _, Uj, _ = eig2(a, d, ts[k] + tau * j / nsub)
                w = np.abs(Uj.conj().T @ ref) ** 2
                ref = Uj[:, int(np.argmax(w))].astype(complex)
        else:
            _, U1, _ = eig2(a, d, ts[k + 1])
            w = np.abs(U1.conj().T @ ref) ** 2
            ref = U1[:, int(np.argmax(w))].astype(complex)
    eK, UK, _ = eig2(a, d, ts[-1])
    P_up = float(abs(UK[:, 1].conj() @ psi) ** 2)
    conv_diabatic = bool(abs(UK[:, 1].conj() @ ref) ** 2 > 0.5)
    return P_up, conv_diabatic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--offsets", type=int, default=8)
    ap.add_argument("--c-LZ", dest="c_LZ", type=float, default=1.0)
    ap.add_argument("--a-list", default="0.001,0.002,0.005,0.01,0.02,0.05")
    ap.add_argument("--d-list", default="1e-5,1e-4,1e-3,0.003,0.01,0.03,0.1,0.3")
    ap.add_argument("--md", required=True)
    x = ap.parse_args()
    al = [float(v) for v in x.a_list.split(",")]; dl = [float(v) for v in x.d_list.split(",")]
    L = ["## The convention against the exact sudden-plus-frozen step on a two-level crossing (engine_lz_twolevel_v1.py)", "",
         "h(t) = [[a t, d], [d, -a t]], records every tau = %g at %d sampling offsets in [0, tau), %d records each side of the crossing; c_LZ %g. P_up: the exact stepped state's weight on the upper adiabatic level at the end (min to max over the offsets); P_LZ = exp(-pi d^2/a); the convention: the fraction of offsets on which its reference ends on the upper level (diabatic); agreement: offsets on which the convention ends diabatic where P_up > 1/2 and adiabatic where P_up < 1/2." % (
             x.tau, x.offsets, x.K, x.c_LZ), "",
         "| a (= s_T) | d (half the minimum gap) | a K tau / d | tau_LZ | sub-steps | P_LZ | P_up (exact step), min to max | convention diabatic (fraction) | agreement (of %d) |" % x.offsets,
         "|---|---|---|---|---|---|---|---|---|"]
    tot = agree_tot = 0
    for a in al:
        for d in dl:
            Ps = []; conv = []; agree = 0
            for io in range(x.offsets):
                t_off = x.tau * io / x.offsets
                P, cd = run_case(a, d, x.tau, x.K, t_off, x.c_LZ)
                Ps.append(P); conv.append(cd)
                agree += int((P > 0.5) == cd)
            tot += x.offsets; agree_tot += agree
            tau_LZ = 1.0 / (x.c_LZ * math.sqrt(a))
            nsub = max(1, int(round(x.c_LZ * math.sqrt(x.tau * a * x.tau)))) if tau_LZ <= x.tau else 1
            L.append("| %g | %g | %.1f | %.3f | %d | %.4e | %.4e to %.4e | %.3f | %d |" % (a, d, a * x.K * x.tau / d, tau_LZ, nsub, math.exp(-math.pi * d * d / a),
                                                                               min(Ps), max(Ps), float(np.mean(conv)), agree))
    L += ["", "- agreement over the grid: %d of %d (offset, case) pairs." % (agree_tot, tot),
          "- the engine's step is the exact frozen-h propagator per record (populations in the new eigenbasis are the endpoint overlaps; the hold changes phases only); the convention is engine_ops_v3's rule on this two-level h; every number above is printed by the script."]
    open(x.md, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
