#!/usr/bin/env python
"""diag_pass0_count_v2.py (builder analyst; version 2 adds --Ts-list, the Fermi-weighted count and energy at several smearings from one
spectrum per channel, and --nsites for the larger families; a diagnostic of the held star test, labelled).  The site-level star test (engine_star_v1.py,
m 0.2, g 0.25, nsites 256, E_F/m 0.95, start from the continuum control) held N_B = 191 at its first pass, about twice what the continuum
star of the start profile holds (M = 19.45).  This counts, on the SAME start profile (N, v): (1) the continuum star's N_B from its own shot;
(2) per channel kappa the WKB count of radial levels below E_F, n_r(kappa) = (1/pi) int sqrt(mu^2 - m^2 - kappa^2/r^2) dr/sqrt(F) with
mu = E_F/N_s, N_s = sqrt(N^2 - v^2), sqrt(F) = N_s/N (the static redshift and the proper radial length of the flat-slice chart); (3) per
kappa the chain's modes below E_F (sharp count and the Fermi-weighted count at T_s = 0.01 m), with the chain's two valleys; (4) the chain at
v = 0 (the lapse alone) and at N = 1 (the flow alone) on the same profile.  The weights: the chain per mode 16 kappa; the continuum
32 kappa n_r(kappa) (8 species, 2 signs of kappa, 2 kappa values of m_j).  Writes JSON and prints the table.
Args: --eos --m --g --EF-over-m --nsites --out."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_ops_v1 as OPS  # noqa: E402
import engine_ovpg_v1 as OV  # noqa: E402
import engine_star_v1 as ST  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eos", required=True)
    ap.add_argument("--m", type=float, default=0.2)
    ap.add_argument("--g", type=float, default=0.25)
    ap.add_argument("--EF-over-m", type=float, default=0.95)
    ap.add_argument("--nsites", type=int, default=256)
    ap.add_argument("--Ts-list", default="0.01,0.005,0.0025,0.00125", help="smearing temperatures as fractions of m")
    ap.add_argument("--kappa-max", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Tss = [float(x) * a.m for x in a.Ts_list.split(",")]
    m, g, EF = a.m, a.g, a.EF_over_m * a.m
    R = np.arange(a.nsites, dtype=float) + 1.0
    N, v, info = ST.ovpg_start(a.eos, m, g, EF, R)
    # (1) the continuum shot's own N_B and M (the same z_c)
    e_of, P_of, n_of = OV.load_eos(a.eos, m)
    z = info["z_c"]
    res = OV.shoot(m, m, g, e_of, P_of, n_of, m * np.sqrt(1 + z * z))
    rr = np.linspace(1e-6, res["R_star"], 20001)
    sol = res["sol"]
    Y = sol.sol(rr)
    mm, Phi = Y[0], Y[1] + res["c"]
    F = 1 - 2 * g * mm / rr
    mu = (res["EF_consistent"]) * np.exp(-Phi)
    n = np.array([float(n_of(x)) for x in mu])
    NB_cont = float(np.trapezoid(4 * np.pi * rr ** 2 * n / np.sqrt(F), rr))
    NB_flat = float(np.trapezoid(4 * np.pi * rr ** 2 * n, rr))
    # (2) WKB per kappa on the lattice profile
    Ns = np.sqrt(np.maximum(N * N - v * v, 1e-300))
    sF = Ns / N
    Rf = np.concatenate([[0.0], R])
    Nsf = interp1d(Rf, np.concatenate([[Ns[0]], Ns]), kind="cubic")
    sFf = interp1d(Rf, np.concatenate([[sF[0]], sF]), kind="cubic")

    def nr(kappa):
        f = lambda r: np.sqrt(max((EF / Nsf(r)) ** 2 - m * m - (kappa / r) ** 2, 0.0)) / sFf(r)
        val, _ = quad(f, 1e-3, R[-1], limit=400, points=[float(x) for x in R[::16]])
        return val / np.pi
    rows = []
    tot = {"wkb": 0.0, "chain_sharp": 0.0, "chain_lapse_only": 0.0, "chain_flow_only": 0.0, "E_sharp": 0.0}
    for T_ in Tss:
        tot["chain_fermi_Ts%.6g" % (T_ / m)] = 0.0
        tot["E_fermi_Ts%.6g" % (T_ / m)] = 0.0
    for kappa in range(1, a.kappa_max + 1):
        h, Q, _ = OPS.chain_parts(a.nsites, m, float(kappa))
        cnt = {}
        for lab, (NN, vv) in (("full", (N, v)), ("lapse_only", (N, 0 * v)), ("flow_only", (0 * N + 1, v))):
            e, U = OPS.spectrum(OPS.clocked(h, Q, NN, vv))
            pos = e > 0
            ep = e[pos]
            cnt[lab] = (int(np.sum(ep < EF)), {T_: (float(OPS.fermi(ep, EF, T_).sum()), float(OPS.fermi(ep, EF, T_) @ ep)) for T_ in Tss},
                        ep[:4].tolist(), float(ep[ep < EF].sum()))
        w = nr(kappa)
        rows.append({"kappa": kappa, "wkb_nr": w, "chain_sharp": cnt["full"][0], "chain_fermi": {"%.6g" % (T_ / m): cnt["full"][1][T_][0] for T_ in Tss},
                     "lowest_levels": cnt["full"][2], "chain_lapse_only": cnt["lapse_only"][0], "chain_flow_only": cnt["flow_only"][0]})
        tot["wkb"] += 32 * kappa * w
        tot["chain_sharp"] += 16 * kappa * cnt["full"][0]
        tot["E_sharp"] += 16 * kappa * cnt["full"][3]
        for T_ in Tss:
            tot["chain_fermi_Ts%.6g" % (T_ / m)] += 16 * kappa * cnt["full"][1][T_][0]
            tot["E_fermi_Ts%.6g" % (T_ / m)] += 16 * kappa * cnt["full"][1][T_][1]
        tot["chain_lapse_only"] += 16 * kappa * cnt["lapse_only"][0]
        tot["chain_flow_only"] += 16 * kappa * cnt["flow_only"][0]
        print("kappa %3d: WKB n_r %.3f (x2 valleys %.3f) | chain below E_F: full %d (Fermi at T_s/m = %s: %s), lapse only %d, flow only %d | lowest %s" % (
            kappa, w, 2 * w, cnt["full"][0], a.Ts_list, " ".join("%.3f" % cnt["full"][1][T_][0] for T_ in Tss), cnt["lapse_only"][0], cnt["flow_only"][0],
            " ".join("%.5f" % x for x in cnt["full"][2])), flush=True)
        if w < 1e-9 and max(cnt["full"][1][T_][0] for T_ in Tss) < 1e-9 and cnt["lapse_only"][0] == 0:
            break
    out = {"args": vars(a), "start": info, "M_cont": res["M"], "R_star_cont": res["R_star"], "NB_cont_proper": NB_cont, "NB_cont_flat_volume": NB_flat,
           "N_centre": float(N[0]), "v_min": float(v.min()), "Ns_centre": float(Ns[0]), "totals": tot, "rows": rows}
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "diag_pass0_count_v2_m%g_g%g_EF%g.json" % (a.m, a.g, a.EF_over_m)), "w"), indent=1, default=float)
    print("continuum: M %.4f, R_star %.2f, N_B (proper volume) %.3f, (flat volume) %.3f; N(0) %.5f, N_s(0) %.5f, min v %.5f" % (
        res["M"], res["R_star"], NB_cont, NB_flat, N[0], Ns[0], v.min()))
    print("totals (weighted): %s" % {k: round(x, 3) for k, x in tot.items()})


if __name__ == "__main__":
    main()
