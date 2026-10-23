#!/usr/bin/env python
"""dust_tail_v2.py (busi lane, reviewer A; REVIEWA_REVIEW_web40g task 1 on top of DUST_tail_second_route version 1).

Imports version 1 (dust_tail_v1: the slice reader, the operator and sea checks, the per-channel readouts, the run checks, the first-route
loader, the --check mode) and keeps its outputs and json keys; every addition is marked "v2:" and its numbers live under row["v2"].

v2, per dust channel kappa (weight w = 16 kappa) and cut R_c (16 and 24 r_s; beyond = the sites with R > R_c), from version 1's exterior
Killing spectrum (omega_l, phi_l) of h_YY and the dust's occupations n_l:
- the 5-bin split by Killing energy, (-inf, -b), [-b, -m), [-m, m), [m, b), [b, inf), b = sqrt(m^2 + 1/a^2) the free band top (a = 1):
  each bin's number n and energy E = sum omega n (the partition is exact: the 5 n add to N, the 5 E to the block energy); [m, b) and
  [b, inf) separately, with their shares of N;
- the counts: N (the dust's number beyond the cut, fermions, out of the nominal 640 = sum 16 kappa n_kappa; N/640 the number fraction),
  E_sym/m (the Killing energy in units of m, a count), E_sym/E_0 (E_0 the record's charge at T 0, read from the run json or --e0-run, else
  the typed --E0; the fraction converts to a count by E_0/m, never by 640; both forms printed), particles and E_particles/m, and the
  energy per particle E_particles/(particles m) (version 1's);
- the dust-only exterior block's entropy S_dust = tr s(Pi_Y D f D^dagger Pi_Y), s(x) = -x ln x - (1 - x) ln(1 - x) (nats; the nonzero
  eigenvalues nu are those of f^{1/2} D_Y^dagger D_Y f^{1/2}, n_kappa of them), its rank bound n_kappa h2(N/n_kappa) and the dimension bound
  d_Y h2(N/d_Y), d_Y = 2 (n - p); weighted with 16 kappa; beside them the review's form (the weighted count against the unweighted dimension
  summed over the 10 channels, 20,480 beyond 16 r_s) and the weighted-dimension form;
- the site continuity identity d rho_x/dt = F(x) - F(x+1) between version 1's current profile F and the density's time derivative
  2 Im sum_j f_j c_j^dagger P_x h_m c_j (exact at the slice's frozen fields; the residual printed); the current through 16 r_s is version 1's.
v2 with --deviation (one dense eigendecomposition with vectors of h_m per channel, all 10 channels; the Killing vacuum P_- = 1(h_m < 0)):
per cut the readouts of the first route's S_hull stated on their own subsystem: the escaped sector F = the eigenvectors of h_m with weight
>= 1 - eps_loc (0.01) on the sites beyond the cut, F_+ (omega > 0, d_+ modes) and F_- (omega < 0, d_-), d_F = d_+ + d_-; M = F^dagger C F,
C = S S^dagger + D f D^dagger (the stored columns as stored); S_F = tr s(M) (the first route's S_F, equal to its S_hull to 7 digits at
T/r_s 40); the counts n_+ = Tr(F_+^dagger C F_+) (the particles in F; exactly the deviation's, F_+^dagger P_- F_+ = 0) and
n_- = Tr(F_-^dagger (1 - C) F_-) (the holes; exactly the deviation's, F_-^dagger P_- F_- = 1), each split into the dust's part (D f D^dagger)
and the pairs' (S S^dagger - P_-); the bound d_F h2((n_+ + n_-)/d_F) (pinching M to its F_+ and F_- blocks raises tr s, and h2 is concave:
a theorem, tested in the selftest) and the tighter block form d_+ h2(n_+/d_+) + d_- h2(n_-/d_-); E_F = sum_l omega_l (M_ll - 1[l in F_-]);
the dust-only S_F,dust = tr s(F^dagger D f D^dagger F) with its rank bound and what the dust's count alone permits in d_F; the region block's
S(C_YY), S((P_-)_YY) and their difference (the first route's comparator); the exterior block's Killing occupations of the whole deviation
(C - P_-)_YY (particles, holes, energies, the 5 bins; the dust's own and the pairs' beside; the first route's "Killing-basis particles;
holes (P_-)"); version 1's --pairs energies (its table I) and the operator check for all 10 channels from the same decomposition. Every
per-level array goes to the npz (COMMS rule 10).
Outputs: per slice <out>/<tag>_T<T/rs>.json and .npz (version 1's members plus v2's), <tag>_all.json, <tag>_table.md (version 1's tables A
to I, dust_tail_table_v1) and <tag>_table_v2.md (dust_tail_table_v2, composed from <tag>_all.json and the first route's jsons only).
Usage: version 1's arguments plus [--deviation] [--eps-loc 0.01] [--e0-run DIR/TAG (the production json holding the record at T 0)]
[--E0 value (a typed input, labelled as such)]; --selftest (dust_tail_selftest_v2); --check SLICE.npz (version 1's)."""
import os
import sys

_T = "4"
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _T = sys.argv[_i + 1]
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, _T)
import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_tail_v1 as V1  # noqa: E402
import dust_tail_ops_v1 as OPS  # noqa: E402
import dust_tail_table_v1 as TBL1  # noqa: E402
import dust_tail_table_v2 as TBL2  # noqa: E402

fmt = TBL1.fmt
utc = TBL1.utc
md5 = V1.md5
EPS_LOC = 0.01                                   # v2: the first route's localization threshold of the escaped sector
BIN_LABELS = ("(-inf,-b)", "[-b,-m)", "[-m,m)", "[m,b)", "[b,inf)")
MODULES = ("dust_tail_v1.py", "dust_tail_ops_v1.py", "dust_tail_table_v1.py", "dust_tail_table_v2.py", "dust_tail_selftest_v2.py")


# ------------------------------------------------------------------------------------------------------------------- v2: the entropy pieces
def h2(p):
    """the binary entropy in nats, h2(p) = -p ln p - (1 - p) ln(1 - p); 0 at p = 0 and p = 1 (p clipped to [0, 1])."""
    p = float(min(max(float(p), 0.0), 1.0))
    out = 0.0
    if p > 0.0:
        out -= p * math.log(p)
    if p < 1.0:
        out -= (1.0 - p) * math.log(1.0 - p)
    return out


def spectral_entropy(nu):
    """sum_i h2(nu_i) over covariance eigenvalues nu (clipped to [0, 1]): the entropy of a fermionic Gaussian state with that block."""
    nu = np.clip(np.asarray(nu, float).ravel(), 0.0, 1.0)
    out = 0.0
    a = nu[nu > 0.0]
    if a.size:
        out -= float(np.sum(a * np.log(a)))
    c = 1.0 - nu[nu < 1.0]
    if c.size:
        out -= float(np.sum(c * np.log(c)))
    return out


def bound_dh2(n, d):
    """d h2(n/d): the largest entropy d modes can carry at total occupation n (h2 concave: Jensen over the d eigenvalues); 0 when d = 0."""
    d = float(d)
    if d <= 0.0:
        return 0.0
    return d * h2(float(n) / d)


def respected(S, bound, tol=1e-12):
    return bool(S <= bound * (1.0 + 1e-9) + tol)


def herm(A):
    return 0.5 * (A + A.conj().T)


def eigvals01(A):
    """the eigenvalues of a Hermitian covariance block clipped to [0, 1] (the clipping absorbs roundoff of order 1e-15)."""
    if A.shape[0] == 0:
        return np.zeros(0)
    return np.clip(np.linalg.eigvalsh(herm(A)), 0.0, 1.0)


# ------------------------------------------------------------------------------------------------------------------- v2: the 5 bins
def band_top(m, a=1.0):
    """b = sqrt(m^2 + 1/a^2): the free lattice band's top at N = 1, v = 0 (the task's band top)."""
    return math.sqrt(m * m + 1.0 / (a * a))


def five_bins(om, nl, m, b):
    """the 5-bin split by Killing energy: (-inf, -b), [-b, -m), [-m, m), [m, b), [b, inf) (lower edges inclusive); each bin's levels,
    n = sum n_l, E = sum omega_l n_l, E/m and the mean omega/m; the partition is exact (the bins' n add to sum n, the E to sum omega n)."""
    edges = (-math.inf, -b, -m, m, b, math.inf)
    om = np.asarray(om, float)
    nl = np.asarray(nl, float)
    out = []
    for lab, lo, hi in zip(BIN_LABELS, edges[:-1], edges[1:]):
        s = (om >= lo) & (om < hi)
        nn = float(nl[s].sum())
        EE = float((om[s] * nl[s]).sum())
        out.append({"bin": lab, "lo": lo, "hi": hi, "levels": int(s.sum()), "n": nn, "E": EE, "E_over_m": EE / m, "mean_omega_over_m": (EE / nn / m) if nn != 0.0 else None})
    return out


# ------------------------------------------------------------------------------------------------------------------- v2: the dust-only block
def dust_exterior_entropy(colsY, f):
    """the dust-only covariance's exterior block Pi_Y D f D^dagger Pi_Y: its nonzero eigenvalues are those of f^{1/2} (D_Y^dagger D_Y) f^{1/2}
    (n_d x n_d). Returns (nu, S, N): S = sum h2(nu) (for f = 1 the dust Slater state's entanglement across the cut), N = sum nu."""
    colsY = np.asarray(colsY)
    nd = colsY.shape[1]
    if nd == 0:
        return np.zeros(0), 0.0, 0.0
    sf = np.sqrt(np.asarray(f, float))
    G = colsY.conj().T @ colsY
    M = (sf[:, None] * G) * sf[None, :]
    nu = eigvals01(M)
    return nu, spectral_entropy(nu), float(nu.sum())


# ------------------------------------------------------------------------------------------------------------------- v2: the continuity identity
def density_time_derivative(hm, cols, f=None):
    """d rho_x/dt = 2 Im sum_j f_j c_j^dagger P_x h_m c_j from i dC/dt = [h_m, C] (per site); with version 1's cut currents F(p) = dN_Y(p)/dt
    the site continuity d rho_x/dt = F(x) - F(x+1) holds exactly (F(0) = F(n) = 0)."""
    cols = np.asarray(cols)
    if cols.shape[1] == 0:
        return np.zeros(cols.shape[0] // 2)
    w = np.imag(np.conj(cols) * (hm @ cols))
    if f is not None:
        w = w * np.asarray(f, float)[None, :]
    return 2.0 * w.sum(axis=1).reshape(-1, 2).sum(axis=1)


def continuity_residual(drho, Fall):
    """max over sites of |d rho_x/dt - (F(x) - F(x+1))| with F(x) = Fall[x-1] for x >= 1 and F(0) = F(n) = 0."""
    Fin = np.concatenate([[0.0], np.asarray(Fall, float)])
    Fout = np.concatenate([np.asarray(Fall, float), [0.0]])
    return float(np.max(np.abs(np.asarray(drho, float) - (Fin - Fout))))


# ------------------------------------------------------------------------------------------------------------------- v2: the escaped sector
def escaped_sector(V, e, p, eps_loc):
    """the first route's sector F (P01 sec. 2's collector rule): the eigenvectors of h_m with weight >= 1 - eps_loc on the sites >= p, split by
    the sign of the eigenvalue. Returns (w_out per eigenvector, the indices of F_+, the indices of F_-)."""
    w_out = np.sum(np.abs(V[2 * p:, :]) ** 2, axis=0)
    keep = w_out >= 1.0 - eps_loc
    return w_out, np.flatnonzero(keep & (e > 0)), np.flatnonzero(keep & (e < 0))


def deviation_pass(Z, ch, c2, arrays, hm, dust, f, k, nd, m, b, cuts, eps_loc):
    """v2: the Killing vacuum and the readouts on the escaped sector, the region block and the exterior Killing block of C - P_-; also version
    1's spectrum checks and --pairs energies for this channel (from the same decomposition)."""
    sea = np.asarray(Z["sea_%d" % k], np.complex128)
    e_all, V = OPS.full_spectrum(hm, vectors=True)
    neg = e_all < 0
    ch["n_negative"] = int(neg.sum())
    ch["spectrum_symmetry_max_dev"] = float(np.max(np.abs(e_all + e_all[::-1])))
    if ("e_%d" % k) in Z.files:
        ch["spectrum_vs_stored_max_dev"] = float(np.max(np.abs(np.sort(e_all) - np.sort(np.asarray(Z["e_%d" % k], float)))))
    arrays["e_%d" % k] = e_all
    Vn = V[:, neg]
    rho_vac = OPS.site_density(Vn)
    ksym_sea = OPS.op_density(hm, sea)
    ksym_vac = OPS.op_density(hm, Vn)
    ch["vacuum_density_max_dev"] = float(np.max(np.abs(rho_vac - 1.0)))
    ch["pairs_energy_beyond"] = {str(c): float(np.sum(ksym_sea[p:] - ksym_vac[p:])) for c, p in cuts}
    ch["pairs_energy_total"] = float(np.sum(ksym_sea - ksym_vac))
    arrays["ksym_pairs_%d" % k] = ksym_sea - ksym_vac
    for c, p in cuts:
        key = str(c)
        w_out, ip, im_ = escaped_sector(V, e_all, p, eps_loc)
        dplus, dminus = int(len(ip)), int(len(im_))
        dF = dplus + dminus
        idx = np.concatenate([ip, im_])
        Fm = V[:, idx]
        A = Fm.conj().T @ sea
        MA = A @ A.conj().T
        if nd:
            B = Fm.conj().T @ dust
            MB = (B * f[None, :]) @ B.conj().T
        else:
            MB = np.zeros((dF, dF), complex)
        M = herm(MA + MB)
        mu = eigvals01(M)
        S_F = spectral_entropy(mu)
        dA = np.real(np.diag(MA)); dB = np.real(np.diag(MB)); dM = np.real(np.diag(M))
        sp = np.zeros(dF, bool); sp[:dplus] = True
        n_plus_pairs = float(dA[sp].sum()); n_minus_pairs = float(dminus - dA[~sp].sum())
        n_plus_dust = float(dB[sp].sum()); n_minus_dust_occ = float(dB[~sp].sum())
        n_plus = n_plus_pairs + n_plus_dust
        n_minus = n_minus_pairs - n_minus_dust_occ
        eF = e_all[idx]
        E_F = float(np.sum(eF * (dM - (~sp).astype(float))))
        muB = eigvals01(herm(MB)) if (nd and dF) else np.zeros(0)
        S_F_dust = spectral_entropy(muB)
        w_F_dust = float(dB.sum())
        Gp = Fm[:, :dplus].conj().T @ Vn
        vac_plus = float(np.max(np.abs(Gp))) if Gp.size else 0.0
        if dminus:
            Gm = Fm[:, dplus:].conj().T @ Vn
            vac_minus = float(np.max(np.abs(Gm @ Gm.conj().T - np.eye(dminus))))
        else:
            vac_minus = 0.0
        # the region block Y
        SY = sea[2 * p:]; VnY = Vn[2 * p:]; DY = dust[2 * p:]
        CYY = SY @ SY.conj().T
        if nd:
            CYY = CYY + (DY * f[None, :]) @ DY.conj().T
        PYY = VnY @ VnY.conj().T
        eigC = eigvals01(CYY); eigP = eigvals01(PYY)
        S_C = spectral_entropy(eigC); S_P = spectral_entropy(eigP)
        n_C = float(np.real(np.trace(CYY))); n_P = float(np.real(np.trace(PYY)))
        del CYY, PYY
        # the exterior block's Killing occupations of the whole deviation
        om, phi = OPS.exterior_spectrum(hm, p)
        occ_sea = OPS.occupations_in_basis(phi, SY)
        occ_vac = OPS.occupations_in_basis(phi, VnY)
        occ_dust = OPS.occupations_in_basis(phi, DY, f) if nd else np.zeros(len(om))
        del phi
        ndev = occ_sea + occ_dust - occ_vac
        npairs = occ_sea - occ_vac
        pos = om > 0
        N_dust_Y = float(np.sum((np.abs(DY) ** 2) * f[None, :])) if nd else 0.0
        dv = {"d_plus": dplus, "d_minus": dminus, "d_F": dF, "n_plus": n_plus, "n_minus": n_minus, "n_F": n_plus + n_minus, "n_plus_dust": n_plus_dust,
              "n_plus_pairs": n_plus_pairs, "n_minus_pairs": n_minus_pairs, "n_minus_dust_occ": n_minus_dust_occ, "pairs_count_F": n_plus_pairs + n_minus_pairs,
              "S_F": S_F, "bound_F": bound_dh2(n_plus + n_minus, dF), "bound_F_blocks": bound_dh2(n_plus, dplus) + bound_dh2(n_minus, dminus), "E_F": E_F,
              "S_F_dust": S_F_dust, "w_F_dust": w_F_dust, "bound_F_dust_rank": bound_dh2(w_F_dust, nd) if nd else 0.0, "bound_F_dustcount": bound_dh2(w_F_dust, dF),
              "vacuum_plus_overlap_max": vac_plus, "vacuum_minus_identity_dev": vac_minus,
              "S_C": S_C, "S_P": S_P, "region_excess": S_C - S_P, "n_C": n_C, "n_P": n_P, "n_C_minus_n_P_minus_N": n_C - n_P - N_dust_Y,
              "particles_dev": float(ndev[pos].sum()), "holes_dev": float(-ndev[~pos].sum()), "E_particles_dev": float((om[pos] * ndev[pos]).sum()),
              "E_holes_dev": float((om[~pos] * ndev[~pos]).sum()), "E_dev_block": float((om * ndev).sum()),
              "particles_pairs": float(npairs[pos].sum()), "holes_pairs": float(-npairs[~pos].sum()),
              "particles_dust_block": float(occ_dust[pos].sum()), "holes_dust_block": float(occ_dust[~pos].sum()),
              "sum_ndev_minus_excess": float(ndev.sum() - (n_C - n_P)),
              "bins5_dev": five_bins(om, ndev, m, b), "bins5_pairs": five_bins(om, npairs, m, b), "bins5_dust_block": five_bins(om, occ_dust, m, b)}
        dv["bound_F_respected"] = respected(S_F, dv["bound_F"])
        dv["bound_F_blocks_respected"] = respected(S_F, dv["bound_F_blocks"])
        dv["bound_F_dust_rank_respected"] = respected(S_F_dust, dv["bound_F_dust_rank"]) if nd else True
        c2["dev"][key] = dv
        arrays["w_out_%g_%d" % (c, k)] = w_out; arrays["F_plus_idx_%g_%d" % (c, k)] = ip; arrays["F_minus_idx_%g_%d" % (c, k)] = im_
        arrays["M_diag_%g_%d" % (c, k)] = dM; arrays["M_diag_dust_%g_%d" % (c, k)] = dB; arrays["mu_F_%g_%d" % (c, k)] = mu
        arrays["eig_CYY_%g_%d" % (c, k)] = eigC; arrays["eig_PYY_%g_%d" % (c, k)] = eigP
        arrays["omega_dev_%g_%d" % (c, k)] = om; arrays["n_dev_%g_%d" % (c, k)] = ndev; arrays["n_pairs_%g_%d" % (c, k)] = npairs; arrays["n_dust_block_%g_%d" % (c, k)] = occ_dust
    del V, Vn, sea


# ------------------------------------------------------------------------------------------------------------------- one slice
def process_slice_v2(path, m, cuts_rs, prof_rs, spectrum="dust", deviation=False, eps_loc=EPS_LOC, log=print):
    """version 1's process_slice (its outputs unchanged; with --deviation its spectrum checks come from the v2 decomposition of every channel)
    followed by the v2 pass; returns (row, arrays) with row["v2"] and the v2 arrays added."""
    t_s = time.time()
    row, arrays = V1.process_slice(path, m, cuts_rs, prof_rs, "none" if deviation else spectrum, False, log=log)
    b = band_top(m)
    Z = np.load(path)
    R = np.asarray(arrays["R"], float); n = len(R); Napp = arrays["N_applied"]; vapp = arrays["v_applied"]
    cuts = [(c, int(cc["first_site_index"])) for c, cc in zip(cuts_rs, row["cuts"])]
    v2 = {"b": b, "b_over_m": b / m, "eps_loc": eps_loc, "deviation": bool(deviation), "channels": [], "weighted": {}}
    for ch in row["channels"]:
        k = int(ch["kappa"]); w = float(ch["w"]); nd = int(ch["n_dust"]); t0 = time.time()
        hm = OPS.killing_h(R, m, k, Napp, vapp)
        dust = np.asarray(Z["dust_%d" % k], np.complex128); f = np.asarray(Z["f_%d" % k], float)
        c2 = {"kappa": k, "w": w, "n_dust": nd, "beyond": {}, "dev": {}}
        if nd:
            drho = density_time_derivative(hm, dust, f)
            arrays["drho_dust_%d" % k] = drho
            c2["continuity_residual"] = continuity_residual(drho, arrays["F_dust_%d" % k])
            c2["drho_total"] = float(drho.sum())
            for c, p in cuts:
                key = str(c); r1 = ch["beyond"][key]
                bins = five_bins(arrays["omega_%g_%d" % (c, k)], arrays["n_%g_%d" % (c, k)], m, b)
                nu, S_d, N_nu = dust_exterior_entropy(dust[2 * p:], f)
                dY = 2 * (n - p)
                reg = {q: r1[q] for q in ("N", "E_sym", "E_block", "particles", "holes", "E_particles", "E_holes", "E_per_particle_over_m", "E_sym_per_N_over_m")}
                reg.update({"E_sym_over_m": r1["E_sym"] / m, "E_particles_over_m": r1["E_particles"] / m, "bins5": bins,
                            "bins5_sum_n_minus_N": float(sum(x["n"] for x in bins) - r1["N"]), "bins5_sum_E_minus_E_block": float(sum(x["E"] for x in bins) - r1["E_block"]),
                            "S_dust": S_d, "nu": nu.tolist(), "nu_sum_minus_N": N_nu - r1["N"], "rank": nd, "d_Y": dY,
                            "bound_rank": bound_dh2(r1["N"], nd), "bound_dim": bound_dh2(r1["N"], dY)})
                reg["bound_rank_respected"] = respected(S_d, reg["bound_rank"])
                reg["bound_dim_respected"] = respected(S_d, reg["bound_dim"])
                arrays["nu_dust_%g_%d" % (c, k)] = nu
                c2["beyond"][key] = reg
        if deviation:
            deviation_pass(Z, ch, c2, arrays, hm, dust, f, k, nd, m, b, cuts, eps_loc)
        c2["seconds"] = time.time() - t0
        v2["channels"].append(c2)
        b1 = c2["beyond"].get(str(cuts[0][0])); d1 = c2["dev"].get(str(cuts[0][0]))
        msg = "  v2 kappa %d:" % k
        if b1:
            msg += " continuity residual %.1e; beyond %g r_s: 5 bins n [%s]; S_dust %.4e vs rank bound %.4e (%s)" % (
                c2["continuity_residual"], cuts[0][0], ", ".join("%.3e" % x["n"] for x in b1["bins5"]), b1["S_dust"], b1["bound_rank"], "respected" if b1["bound_rank_respected"] else "EXCEEDED")
        if d1:
            msg += "; F: d_F %d, n_+ %.4e (dust %.4e), n_- %.4e, S_F %.4e vs bound %.4e (%s), region excess %.4e, block particles %.4e holes %.4e" % (
                d1["d_F"], d1["n_plus"], d1["n_plus_dust"], d1["n_minus"], d1["S_F"], d1["bound_F"], "respected" if d1["bound_F_respected"] else "EXCEEDED",
                d1["region_excess"], d1["particles_dev"], d1["holes_dev"])
        log(msg + "; %.1f s" % c2["seconds"])
    Z.close()
    weigh_v2(row, v2, cuts, n, m)
    v2["seconds"] = time.time() - t_s
    row["v2"] = v2
    q0 = v2["weighted"][str(cuts[0][0])]
    msg = "  v2 weighted beyond %g r_s: N %.6e (%s of %g), E_sym/m %.6e, bins n [%s], [m,b) share %s, [b,inf) share %s; S_dust %.6e vs rank bound %.6e (%s)" % (
        cuts[0][0], q0["N"], fmt(q0["N_frac_nominal"], "%.4e"), row["dust_number_nominal"], q0["E_sym_over_m"], ", ".join("%.3e" % x["n"] for x in q0["bins5"]),
        fmt(q0["frac_m_b"], "%.4f"), fmt(q0["frac_b_inf"], "%.4e"), q0["S_dust"], q0["bound_rank"], "respected" if q0["all_bound_rank_respected"] else "EXCEEDED")
    if deviation:
        d0 = q0["dev"]
        msg += "; F: d_F %d (w %g), n_+ %.6e, n_- %.6e (pairs' share %s), S_F %.6e vs bound %.6e (%s); region excess %.6e; block particles %.6e holes %.6e" % (
            d0["d_F_unw"], d0["d_F_w"], d0["n_plus"], d0["n_minus"], fmt(d0["pairs_share_of_n_F"], "%.4f"), d0["S_F"], d0["bound_F"], "respected" if d0["all_bound_F_respected"] else "EXCEEDED",
            d0["region_excess"], d0["particles_dev"], d0["holes_dev"])
    log(msg + "; %.0f s" % v2["seconds"])
    return row, arrays


def weigh_v2(row, v2, cuts, n, m):
    """v2: the weighted sums (w = 16 kappa) per cut and the checks' maxima; with --deviation also version 1's pairs_energy and its spectrum
    checks for all channels."""
    chans = v2["channels"]
    dch = [c2 for c2 in chans if c2["n_dust"]]
    wsum_all = float(sum(c2["w"] for c2 in chans))
    nominal = row["dust_number_nominal"]
    for c, p in cuts:
        key = str(c); dY = 2 * (n - p)
        regs = [(c2["w"], c2["beyond"][key]) for c2 in dch]
        W = lambda q: float(sum(w_ * r[q] for w_, r in regs))  # noqa: E731
        q = {x: W(x) for x in ("N", "E_sym", "E_block", "particles", "holes", "E_particles", "E_holes", "E_sym_over_m", "E_particles_over_m", "S_dust", "bound_rank", "bound_dim")}
        q["N_frac_nominal"] = (q["N"] / nominal) if nominal else None
        q["E_per_particle_over_m"] = (q["E_particles"] / q["particles"] / m) if q["particles"] > 0 else None
        q["E_sym_per_N_over_m"] = (q["E_sym"] / q["N"] / m) if q["N"] > 0 else None
        q["rank_w"] = float(sum(w_ * r["rank"] for w_, r in regs)); q["rank_unw"] = int(sum(r["rank"] for _, r in regs))
        q["d_Y"] = dY; q["d_Y_unw_all"] = dY * len(chans); q["d_Y_w_all"] = wsum_all * dY
        q["bound_review_form"] = bound_dh2(q["N"], q["d_Y_unw_all"])
        q["bound_dim_w_form"] = bound_dh2(q["N"], q["d_Y_w_all"])
        q["bins5"] = []
        for i, lab in enumerate(BIN_LABELS):
            nn = float(sum(w_ * r["bins5"][i]["n"] for w_, r in regs)); EE = float(sum(w_ * r["bins5"][i]["E"] for w_, r in regs))
            q["bins5"].append({"bin": lab, "n": nn, "E": EE, "E_over_m": EE / m, "mean_omega_over_m": (EE / nn / m) if nn != 0.0 else None,
                               "levels": int(sum(r["bins5"][i]["levels"] for _, r in regs))})
        q["frac_m_b"] = (q["bins5"][3]["n"] / q["N"]) if q["N"] > 0 else None
        q["frac_b_inf"] = (q["bins5"][4]["n"] / q["N"]) if q["N"] > 0 else None
        q["frac_above_m"] = ((q["bins5"][3]["n"] + q["bins5"][4]["n"]) / q["N"]) if q["N"] > 0 else None
        q["max_bins5_sum_n_minus_N"] = max((abs(r["bins5_sum_n_minus_N"]) for _, r in regs), default=0.0)
        q["max_bins5_sum_E_minus_E_block"] = max((abs(r["bins5_sum_E_minus_E_block"]) for _, r in regs), default=0.0)
        q["max_nu_sum_minus_N"] = max((abs(r["nu_sum_minus_N"]) for _, r in regs), default=0.0)
        q["all_bound_rank_respected"] = all(r["bound_rank_respected"] for _, r in regs)
        q["max_S_dust_over_bound_rank"] = max((r["S_dust"] / r["bound_rank"] for _, r in regs if r["bound_rank"] > 0), default=None)
        if v2["deviation"]:
            dvs = [(c2["w"], c2["dev"][key]) for c2 in chans]
            WD = lambda x: float(sum(w_ * dd[x] for w_, dd in dvs))  # noqa: E731
            d = {x: WD(x) for x in ("n_plus", "n_minus", "n_F", "n_plus_dust", "n_plus_pairs", "n_minus_pairs", "n_minus_dust_occ", "pairs_count_F", "S_F", "bound_F", "bound_F_blocks",
                                    "E_F", "S_F_dust", "w_F_dust", "bound_F_dust_rank", "bound_F_dustcount", "S_C", "S_P", "region_excess", "n_C", "n_P", "particles_dev", "holes_dev",
                                    "E_particles_dev", "E_holes_dev", "E_dev_block", "particles_pairs", "holes_pairs", "particles_dust_block", "holes_dust_block")}
            d["d_F_unw"] = int(sum(dd["d_F"] for _, dd in dvs)); d["d_F_w"] = float(sum(w_ * dd["d_F"] for w_, dd in dvs))
            d["d_plus_unw"] = int(sum(dd["d_plus"] for _, dd in dvs)); d["d_minus_unw"] = int(sum(dd["d_minus"] for _, dd in dvs))
            d["bound_F_w_form"] = bound_dh2(d["n_F"], d["d_F_w"]); d["bound_F_review_form"] = bound_dh2(d["n_F"], d["d_F_unw"])
            d["bound_F_dustcount_w_form"] = bound_dh2(d["w_F_dust"], d["d_F_w"]); d["bound_F_dustcount_review_form"] = bound_dh2(d["w_F_dust"], d["d_F_unw"])
            d["all_bound_F_respected"] = all(dd["bound_F_respected"] for _, dd in dvs)
            d["all_bound_F_blocks_respected"] = all(dd["bound_F_blocks_respected"] for _, dd in dvs)
            d["max_S_F_over_bound_F"] = max((dd["S_F"] / dd["bound_F"] for _, dd in dvs if dd["bound_F"] > 0), default=None)
            d["pairs_share_of_n_F"] = (d["pairs_count_F"] / d["n_F"]) if d["n_F"] > 0 else None
            d["dust_share_of_n_F"] = ((d["n_plus_dust"] - d["n_minus_dust_occ"]) / d["n_F"]) if d["n_F"] > 0 else None
            for name in ("bins5_dev", "bins5_pairs", "bins5_dust_block"):
                d[name] = []
                for i, lab in enumerate(BIN_LABELS):
                    nn = float(sum(w_ * dd[name][i]["n"] for w_, dd in dvs)); EE = float(sum(w_ * dd[name][i]["E"] for w_, dd in dvs))
                    d[name].append({"bin": lab, "n": nn, "E": EE, "E_over_m": EE / m})
            d["max_vacuum_plus_overlap"] = max(dd["vacuum_plus_overlap_max"] for _, dd in dvs)
            d["max_vacuum_minus_identity_dev"] = max(dd["vacuum_minus_identity_dev"] for _, dd in dvs)
            d["max_n_C_minus_n_P_minus_N"] = max(abs(dd["n_C_minus_n_P_minus_N"]) for _, dd in dvs)
            d["max_sum_ndev_minus_excess"] = max(abs(dd["sum_ndev_minus_excess"]) for _, dd in dvs)
            q["dev"] = d
            row["weighted"][key]["pairs_energy"] = float(sum(ch["w"] * ch["pairs_energy_beyond"][key] for ch in row["channels"]))
        v2["weighted"][key] = q
    v2["max_continuity_residual"] = max((c2["continuity_residual"] for c2 in dch), default=0.0)
    v2["max_abs_drho_total"] = max((abs(c2["drho_total"]) for c2 in dch), default=0.0)
    if v2["deviation"]:
        ck = row["checks"]
        ck["max_spectrum_vs_stored"] = max((c.get("spectrum_vs_stored_max_dev") for c in row["channels"] if c.get("spectrum_vs_stored_max_dev") is not None), default=None)
        ck["max_spectrum_symmetry"] = max((c.get("spectrum_symmetry_max_dev") for c in row["channels"] if c.get("spectrum_symmetry_max_dev") is not None), default=None)
        ck["n_negative"] = sorted({c.get("n_negative") for c in row["channels"] if "n_negative" in c})
        ck["max_vacuum_density_dev"] = max((c.get("vacuum_density_max_dev") for c in row["channels"] if c.get("vacuum_density_max_dev") is not None), default=None)


# ------------------------------------------------------------------------------------------------------------------- v2: the energy scale E_0
def energy_scale(run, e0_run, E0_typed, m):
    """E_0 = the record's charge at T 0: from the run json if its earliest record is at T 0, else from --e0-run's json, else the typed --E0
    (labelled as typed); E_0/m converts a fraction of E_0 to a count."""
    out = {"E_0": None, "E_0_source": None, "E_0_over_m": None}
    cands = []
    if run is not None:
        cands.append(("the run json's record at T 0", run[0]))
    if e0_run:
        d_, tag = os.path.split(e0_run)
        pth = os.path.join(d_, "collapse_%s.json" % tag)
        cands.append(("the record at T 0 of %s" % pth, json.load(open(pth))))
    for src, J in cands:
        recs = [r for r in J.get("records", []) if r.get("E") is not None]
        if recs:
            r0 = min(recs, key=lambda r: float(r["T"]))
            if abs(float(r0["T"])) < 1e-9:
                out = {"E_0": float(r0["E"]), "E_0_source": src, "E_0_T": float(r0["T"])}
                break
    if out["E_0"] is None and E0_typed is not None:
        out = {"E_0": float(E0_typed), "E_0_source": "a typed input (--E0), not read from a record"}
    out["E_0_over_m"] = (out["E_0"] / m) if out["E_0"] is not None else None
    return out


def add_energy_fractions(row, e0):
    """v2: the fractions of E_0 beside the counts: a fraction of E_0 converts to a count by E_0/m; the same fraction times the nominal 640
    is printed as the form the review rejects."""
    E0 = (e0 or {}).get("E_0")
    m = row["m"]; nominal = row["dust_number_nominal"]
    for key, q in row["v2"]["weighted"].items():
        q["E_0"] = E0; q["E_0_over_m"] = (E0 / m) if E0 else None
        for lab in ("E_sym", "E_particles"):
            val = q[lab]
            q["%s_frac_E0" % lab] = (val / E0) if E0 else None
            q["%s_count_from_frac_E0" % lab] = ((val / E0) * (E0 / m)) if E0 else None
            q["%s_frac_E0_times_nominal" % lab] = ((val / E0) * nominal) if E0 else None
        q["count_from_frac_E0"] = q["E_sym_count_from_frac_E0"]


# ------------------------------------------------------------------------------------------------------------------- main
def default_args(**kw):
    a = argparse.Namespace(slices="", m=0.2, out_dir="", tag="dust_tail_v2", cuts="16,24", profile="4,8,12,16,20,24,28", run="", e0_run="", E0=None,
                           first_route_tail_json="", first_route_dv2_jsons="", spectrum="dust", deviation=False, eps_loc=EPS_LOC, threads=4, overwrite=False, selftest=False, check="")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def run_main(a, log=print):
    slices = [p for p in a.slices.split(",") if p]
    if not slices or not a.out_dir:
        sys.exit("--slices and --out-dir are required")
    os.makedirs(a.out_dir, exist_ok=True)
    cuts_rs = [float(x) for x in a.cuts.split(",")]; prof_rs = [float(x) for x in a.profile.split(",")]
    for name in ("%s_table.md" % a.tag, "%s_table_v2.md" % a.tag, "%s_all.json" % a.tag):
        if os.path.exists(os.path.join(a.out_dir, name)) and not a.overwrite:
            sys.exit("refusing to overwrite %s (use a new --tag)" % os.path.join(a.out_dir, name))
    try:
        load = open("/proc/loadavg").read().split()[:3]
    except OSError:
        load = None
    prov = {"script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)),
            "modules": ", ".join("%s %s" % (x, md5(os.path.join(HERE, x))) for x in MODULES if os.path.exists(os.path.join(HERE, x))),
            "args": " ".join(sys.argv[1:]), "host": socket.gethostname(), "utc_start": utc(), "load_at_start": load, "threads": a.threads, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": sys.version.split()[0], "m": a.m, "spectrum": a.spectrum, "deviation": bool(a.deviation), "eps_loc": a.eps_loc, "E0_typed": a.E0, "e0_run": a.e0_run, "b": band_top(a.m)}
    run = V1.load_run(a.run) if a.run else None
    if run is not None and abs(float(run[0]["args"]["m"]) - a.m) > 1e-12:
        sys.exit("--m %g differs from the run's m %g" % (a.m, float(run[0]["args"]["m"])))
    e0 = energy_scale(run, a.e0_run, a.E0, a.m)
    prov["E_0"] = e0
    log("E_0: %s (%s); E_0/m %s" % (fmt(e0.get("E_0"), "%.6f"), e0.get("E_0_source"), fmt(e0.get("E_0_over_m"), "%.4f")))
    first = V1.load_first_route(a.first_route_tail_json, a.first_route_dv2_jsons)
    rows = []
    for p in slices:
        row, arrays = process_slice_v2(p, a.m, cuts_rs, prof_rs, a.spectrum, a.deviation, a.eps_loc, log=log)
        if run is not None:
            row["run_checks"] = V1.run_checks(row, arrays["N_applied"], arrays["v_applied"], run[0], run[1], cuts_rs)
            rb = row["run_checks"].get("beyond") or {}
            log("  run checks: applied fields dev %s; K_density - stored K_dust %s; bare energy - E_dust_shells %s; E_G %s, m/T_H %s" % (
                row["run_checks"].get("applied_fields_max_dev"), {c: "%+.3e (rel %s)" % (q["diff"], fmt(q["rel"], "%.2e")) for c, q in rb.items()},
                fmt(row["run_checks"].get("E_dust_shells_diff"), "%+.3e"), fmt(row["run_checks"].get("E_G"), "%.6f"), fmt(row["run_checks"].get("m_over_TH"), "%.3f")))
        row["run_checks_v2"] = dict(e0)
        add_energy_fractions(row, e0)
        stem = os.path.join(a.out_dir, "%s_T%07.2f" % (a.tag, row["T_over_rs"]))
        if (os.path.exists(stem + ".json") or os.path.exists(stem + ".npz")) and not a.overwrite:
            sys.exit("refusing to overwrite %s.*" % stem)
        json.dump(row, open(stem + ".json", "w"), indent=1, default=float)
        np.savez(stem + ".npz", **arrays)
        rows.append(row)
        log("  wrote %s.json and .npz" % stem)
    rows.sort(key=lambda r: r["T_over_rs"])
    prov["utc_end"] = utc()
    md1 = TBL1.compose_table(rows, first, cuts_rs, prof_rs, a.m, prov)
    open(os.path.join(a.out_dir, "%s_table.md" % a.tag), "w").write(md1)
    json.dump({"script": "dust_tail_v2.py", "provenance": prov, "cuts_rs": cuts_rs, "profile_rs": prof_rs, "rows": rows}, open(os.path.join(a.out_dir, "%s_all.json" % a.tag), "w"), indent=1, default=float)
    out2, _ = TBL2.compose_from_outputs(a.out_dir, a.tag, a.first_route_tail_json, a.first_route_dv2_jsons)
    log("wrote %s, %s and %s" % (os.path.join(a.out_dir, "%s_table.md" % a.tag), os.path.join(a.out_dir, "%s_all.json" % a.tag), out2))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=""); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--out-dir", default=""); ap.add_argument("--tag", default="dust_tail_v2")
    ap.add_argument("--cuts", default="16,24"); ap.add_argument("--profile", default="4,8,12,16,20,24,28"); ap.add_argument("--run", default=""); ap.add_argument("--e0-run", default="")
    ap.add_argument("--E0", type=float, default=None); ap.add_argument("--first-route-tail-json", default=""); ap.add_argument("--first-route-dv2-jsons", default="")
    ap.add_argument("--spectrum", default="dust", choices=["dust", "all", "none"]); ap.add_argument("--deviation", action="store_true"); ap.add_argument("--eps-loc", type=float, default=EPS_LOC)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--check", default="")
    a = ap.parse_args()
    if a.selftest:
        import dust_tail_selftest_v2 as ST
        sys.exit(0 if ST.selftest(sys.modules[__name__]) else 1)
    if a.check:
        V1.check(a.check); return
    run_main(a)


if __name__ == "__main__":
    main()
