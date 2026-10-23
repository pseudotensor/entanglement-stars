#!/usr/bin/env python
"""Strand 5 J4 stage 2, the conservative-transport cross-check (spec sec. 10(j)(5); builder analyst): at the dilute, the maximum and the most
compact member of each stage-1 family, the lock pieces of strand5_J4_stage2_v2 recomputed with the matter transported conservatively: the
cumulative number N(< r) = sum_{j' <= j} 4 pi r_j'^2 n_j' psi_j'^6 (monotone PCHIP through the origin, the nodes and N_B beyond the star) is
moved as N_lam(< r) = N(< r/lam), and n_lam at the nodes is its node difference divided by 4 pi r_j^2 psi_lam,j^6, so that N_B is conserved
exactly by the transport (at lam = 1 the nodes' n are returned exactly).  psi, chi, the operator, the EOS inversion and the central differences
(h = 1e-4, 1e-5, Richardson) are those of strand5_J4_stage2_v2.  Per member: Total, R_stat, D_mat, the transport's dN_B/dlam (zero by
construction, printed as the check), the closures Total + 2 pi k (A - B) and R_stat - 2 pi k B - rows.z', and the physics columns beside the
multiplier route's (from the stage-2 JSON of the same family): D_mat (here, at fixed N_B) against D_mat - mu_inf dN_B/dlam (multiplier route),
and Total - D_mat against the multiplier route's, as shares of -(A - B)/G, with their differences.  Writes <out>/cons_<tag>.json.
(version 2: the cumulative interpolated by PCHIP (monotone): version 1's odd cubic spline overshot where N(< r) flattens at the surface, giving
negative node increments that the clip turned into a non-zero dN_B/dlam; the stage-2 functions from strand5_J4_stage2_v5.)"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import json
import sys
import types

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import strand5_J4_stage2_v5 as S2  # noqa: E402

S = S2.S


def cumulative_spline(r, n, psi, J, R):
    """The discrete cumulative number N(< r_j) through the origin and the nodes (N_B beyond), interpolated monotonically (PCHIP), so the
    transported node increments stay non-negative and sum to N_B (version 1's odd cubic spline overshot where N flattens at the surface)."""
    w = 4 * np.pi * r * r * n * psi ** 6
    Nc = np.cumsum(w)
    NB = Nc[-1]
    rx = np.arange(1, J + R + 13, dtype=float)
    Nx = np.concatenate([Nc, np.full(len(rx) - J, NB)])
    return PchipInterpolator(np.concatenate([[0.0], rx]), np.concatenate([[0.0], Nx])), NB, rx[-1]


def phi_parts_cons(lam, F, r_all, J, R, sps, sch, sNc, NB, A, B, rmax, kap_nodes, opx):
    ps, ch, _ = S2.dilated(lam, r_all, sps, sch, lambda x: np.zeros_like(np.asarray(x, float)), A, B, rmax)
    s = lam ** -0.5
    u = r_all * (ps - s)
    Cu = opx.apply(u, np.sqrt(lam) * A)
    w = 4 * np.pi * r_all * r_all
    vac_s = float(np.sum(w * (ch - s) * Cu))
    vac_1 = float(np.sum(w * (ch - 1) * Cu))
    x = r_all[:J] / lam
    Nl = np.where(x <= rmax, sNc(np.minimum(x, rmax)), NB)
    dN = np.diff(np.concatenate([[0.0], Nl]))
    psj, chj = ps[:J], ch[:J]
    nj = np.maximum(dN / (w[:J] * psj ** 6), 0.0)
    b = psj * psj
    eb, _ = F.eb(nj, b, kap_nodes)
    Emat = float(np.sum(w[:J] * (chj / psj) * psj ** 6 * eb))
    NBl = float(np.sum(w[:J] * nj * psj ** 6))
    return vac_s, vac_1, Emat, NBl, float(np.min(dN))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True)
    ap.add_argument("--stage2", required=True, help="the family's stage-2 JSON (multiplier route)")
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--packets", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = json.load(open(a.family))
    M = R["meta"]
    S2J = {round(m["z_c"], 10): m for m in json.load(open(a.stage2))["members"]}
    prof = np.load(a.family.replace(".json", "_profiles.npz"))
    args = types.SimpleNamespace(kernel=a.kernel, packets=a.packets, mlight=M["mlight"], aH=M["aH"], operator=M["operator"])
    stencil, kop, info = S.setup_kernel(args)
    gM, gB = M["couplings"]
    e1 = S.E1Table(M["mlight"], gM, gB, 1.0, 1.0)
    mem = [d for d in R["members"] if d.get("newton_ok")]
    imax = int(np.argmax([d["M"] for d in mem]))
    out = {"meta": {"family": a.family, "stage2": a.stage2, "k_op": kop}, "members": []}
    for i in sorted({0, imax, len(mem) - 1}):
        d = mem[i]
        zc = d["z_c"]
        key = "z%.6f" % zc
        r, psi, N, n, qF = (prof[key + s] for s in ("_r", "_psi", "_N", "_n", "_qF"))
        J = len(r)
        star = S.Star(J, stencil, kop, M["mlight"], (gM, gB, 1.0, 1.0), e1)
        chi = N * psi
        A, B, mu = d["A"], d["B"], d["mu_inf"]
        x = np.concatenate([psi, chi, [A, B, mu]])
        F = S2.Fam(star, M["mlight"], e1)
        kap = psi * psi * qF
        sps, sch, _ = S2.splines(r, psi, chi, n, A, B, star.op.R)
        Rr = star.op.R
        Jx = J + Rr
        r_all = np.arange(1, Jx + 1, dtype=float)
        rmax = J + Rr + 12.0
        opx = S.Radial(Jx, star.op.st)
        sNc, NB, _ = cumulative_spline(r, n, psi, J, Rr)
        D = {}
        mins = []
        for h in S2.HS:
            p = phi_parts_cons(1 + h, F, r_all, J, Rr, sps, sch, sNc, NB, A, B, rmax, kap, opx)
            m_ = phi_parts_cons(1 - h, F, r_all, J, Rr, sps, sch, sNc, NB, A, B, rmax, kap, opx)
            mins += [p[4], m_[4]]
            D[h] = {"Total": ((p[0] + p[2]) - (m_[0] + m_[2])) / (2 * h), "R_stat": ((p[1] + p[2]) - (m_[1] + m_[2])) / (2 * h),
                    "D_mat": (p[2] - m_[2]) / (2 * h), "dNB": (p[3] - m_[3]) / (2 * h)}
        h1, h2 = S2.HS
        rich = {k: (100 * D[h2][k] - D[h1][k]) / 99 for k in D[h1]}
        base = phi_parts_cons(1.0, F, r_all, J, Rr, sps, sch, sNc, NB, A, B, rmax, kap, opx)
        AmBG = 2 * np.pi * kop * (A - B)
        mult = S2J[round(zc, 10)]["lock"]
        rec = {"z_c": zc, "J": J, **rich, "by_h": {str(h): D[h] for h in S2.HS}, "E_mat_base": base[2], "N_B_base": base[3],
               "min_node_increment": float(min(mins)), "AmB_over_G": AmBG,
               "closure_total": rich["Total"] + AmBG, "D_vac_plus_S_inf": rich["Total"] - rich["D_mat"],
               "share_matter": -rich["D_mat"] / AmBG if A != B else None, "share_vac": -(rich["Total"] - rich["D_mat"]) / AmBG if A != B else None,
               "multiplier_share_matter": mult["share_matter_fixed_NB"], "multiplier_share_vac": mult["share_vac"],
               "multiplier_D_mat_fixed_NB": mult["D_mat_fixed_NB"], "multiplier_vac_piece": mult["vac_piece"]}
        rec["diff_D_mat"] = rich["D_mat"] - mult["D_mat_fixed_NB"]
        rec["diff_vac_piece"] = rec["D_vac_plus_S_inf"] - mult["vac_piece"]
        rec["diff_share_matter"] = (rec["share_matter"] - mult["share_matter_fixed_NB"]) if rec["share_matter"] is not None else None
        out["members"].append(rec)
        print("z_c %.4f J %d: Total %.6e (A-B)/G %.6e closure %.1e dN_B %.1e; D_mat cons %.6e mult %.6e (diff %.1e); shares %.5f/%.5f against %.5f/%.5f" % (
            zc, J, rich["Total"], AmBG, rec["closure_total"], rich["dNB"], rich["D_mat"], mult["D_mat_fixed_NB"], rec["diff_D_mat"],
            rec["share_matter"] or float("nan"), rec["share_vac"] or float("nan"), mult["share_matter_fixed_NB"] or float("nan"),
            mult["share_vac"] or float("nan")), flush=True)
    with open(os.path.join(a.out, "cons_" + os.path.basename(a.family)), "w") as fh:
        json.dump(out, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
