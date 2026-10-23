#!/usr/bin/env python
"""engine_capture_v1.py (lead's draft; SPEC_final4b_engine.md sec. 5.5): the capture cross section of inward packets on a frozen
background (the solved late-time fields, labelled; the prescribed exterior with the same interior as the control) by the first-passage
definition that survives the return through the centre: the time-integrated bond current through the trapping surface of the state's
ingoing-projected part (the window's Fourier components with negative local group velocity).  sigma = (pi/p^2) sum_kappa |kappa| T_kappa
over both signs of kappa (the chain's two valleys), against 27 pi (g E)^2."""
import json
import os

import numpy as np

import engine_ops_v1 as OPS


def local_vg(k, m, a, v, N=1.0):
    E = np.sqrt(m * m + a * a + np.sin(k) ** 2)
    return N * np.sin(k) * np.cos(k) / E + v * np.cos(2 * k)


def ingoing_projection(psi, R, Rt, m, kappa, v_loc, N_loc, half=32):
    """keep the Fourier components of the window [Rt - half, Rt + half] whose local group velocity is negative; zero outside."""
    sel = (R >= Rt - half) & (R <= Rt + half)
    a = psi.reshape(-1, 2).copy()
    a[~sel] = 0.0
    n = 4 * len(a)
    k = 2 * np.pi * np.fft.fftfreq(n)
    ak = np.fft.fft(a, n=n, axis=0)
    keep = local_vg(k, m, kappa / Rt, v_loc, N_loc) < 0
    ak[~keep] = 0.0
    out = np.fft.ifft(ak, axis=0)[:len(a)]
    out[~sel] = 0.0
    return out.reshape(-1)


def capture_channel(h, Q, R, N, v, m, kappa, omega, Rt, R_launch, sigma_omega, dt_out, T):
    """one channel: build the ingoing packet (BUILDER: flow_chart_v4.branch_momenta with the channel symbol E(k, kappa/R) and the exterior
    v at R_launch, outgoing=False; build_packet with the local spinor), evolve on the frozen h_m, integrate the ingoing-projected cut
    current at Rt; return T_kappa, the largest norm inside Rt (lower bound) and the time series."""
    import flow_chart_v4 as FC  # BUILDER: path
    hm = OPS.clocked(h, Q, N, v)
    e, U = OPS.spectrum(hm)
    vext = float(np.interp(R_launch, R, v)); mc = float(np.sqrt(m * m + (kappa / R_launch) ** 2))
    k, om, a2 = FC.branch_momenta(omega, sigma_omega, mc, vext, outgoing=False)
    psi0 = FC.build_packet(len(R), int(round(R_launch - R[0])), k, a2, mc, m=m, a=kappa / R_launch)
    cut = int(np.searchsorted(R, Rt))
    v_loc, N_loc = float(np.interp(Rt, R, v)), float(np.interp(Rt, R, N))
    times = np.arange(0.0, T + 1e-9, dt_out)
    I_in, inside = [], []
    for t in times:
        psi = FC.propagate(U, e, t, psi0)
        pin = ingoing_projection(psi, R, Rt, m, kappa, v_loc, N_loc)
        I_in.append(-OPS.cut_current(hm, pin[:, None], None, cut))      # inward: the loss of the outside = +inward flux
        inside.append(float(np.sum(np.abs(psi.reshape(-1, 2)[R < Rt]) ** 2)))
    I_in = np.array(I_in)
    return {"T_kappa": float(np.trapz(I_in, times)), "max_inside": float(max(inside)), "times": times.tolist(), "I_in": I_in.tolist()}


def cross_section(results, omega, m, g, E):
    """results: {kappa: T_kappa} for kappa = 1.. (each chain carries +kappa and -kappa once: the sum counts both signs)."""
    p2 = omega ** 2 - m ** 2
    sig = (np.pi / p2) * sum(2 * abs(k) * Tk for k, Tk in results.items())
    return {"sigma": sig, "sigma_over_geometric": sig / (27 * np.pi * (g * E) ** 2), "omega_rs": omega * 2 * g * E}


# BUILDER: the driver: load collapse_<tag>_fields.npz (the last T), E from the record, Rt from trapping_radius; for omega r_s in
# {2, 4, 8, 16} and kappa = 1.. until T_kappa < 1e-3: capture_channel on (i) the solved fields, (ii) N = 1, v = -sqrt(2 g E/r) beyond
# R_t with the solved fields inside (the control); write the tables and the step |kappa| at T = 1/2 against omega b_c, b_c = 3 sqrt 3 g E.
