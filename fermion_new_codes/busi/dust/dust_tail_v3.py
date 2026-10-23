#!/usr/bin/env python
"""dust_tail_v3.py (busi lane, reviewer A; the dust tail's second route, version 3, after the red team of version 2: its defects 1 and 2).

Imports version 2 (dust_tail_v2: version 1's readouts, the 5 bins, the counts, S_dust, the continuity identity and, with --deviation, its set
of single eigenvectors, the region block and the exterior Killing block of the whole deviation) and keeps its outputs, json keys and tables;
every addition is marked "v3:" and its numbers live under row["v3"] of the per-slice json.

v3: the first route's escaped sector, built as engine_massive_channel_D_v2.py builds it (its outside_weight_halves and escaped_sector; this
lane's copy, lines 131 to 152): in each spectral half h of h_m (e >= 0: +, e < 0: -) the eigenvectors Q of the half's outside-weight operator
W_h = V_h^dagger Pi_Y V_h (V_h the half's eigenvectors, Pi_Y the projector on the sites beyond the cut) with eigenvalue w >= 1 - eps_loc
(0.01; 0.001 the first route's control), F_h = V_h Q_keep, F = [F_+, F_-]: 2046 = 1023 + 1023 modes per chain beyond 16 r_s and
1022 = 511 + 511 beyond 24 r_s at every clock of the first route's jsons. The columns of F are wave packets inside one spectral half, not
eigenvectors of h_m. Version 2's set (the individual eigenvectors of the whole chain's h_m with outside weight >= 1 - eps_loc: 0 to 104 modes
per chain at T/r_s 48 to 56) is kept beside under the label "localized set F_loc"; it is not the first route's sector.
On F, per channel and cut, with the stored columns as stored (C = S S^dagger + D f D^dagger): M = F^dagger C F; S_F = tr s(M) (the first
route's S_F); n_+ = Tr F_+^dagger C F_+ (its sector particles) and n_- = Tr F_-^dagger (1 - C) F_- (its sector holes), each split into the
dust's and the pairs' parts; the bound d_F h2(n_F/d_F) and the block form d_+ h2(n_+/d_+) + d_- h2(n_-/d_-); E_F = Re tr[(F^dagger h_m F)
(M - Pi)], Pi = F^dagger P_- F = diag(1 on F_-) (the first route's E_F; F^dagger h_m F is not diagonal); the first route's hull step in the
sector's own coordinates: the eigenvectors of M - Pi with |lambda| > eps_rank (1e-9; 1e-6 the control), per spectral half every left singular
vector of the kept eigenvectors' rows (the fallback form of its hull_coords; page_ops_v6 is not in this lane), L, S_hull = tr s(L^dagger M L),
the check that L reduces Pi, the dropped eigenvalues' bound q h2(min(delta/q, 1/2)) (its docstring's form); the dust-only S_F,dust with its
rank bound; the floors beside every count and entropy: the stored columns' orthonormality defect max|Phi^dagger Phi - 1|, Phi = [S, D] (the
first route's stored_orthonormality), and the eigensolver's (max|F^dagger F - 1|, F_+^dagger P_- F_+, F_-^dagger P_- F_- - 1), eta their
maximum, n_floor = d_F eta, S_floor = d_F h2(eta); the verdict "respected" or "EXCEEDED" is printed only when S_F > S_floor and n_F > n_floor,
else "below floor" (the bound's tolerance is S_floor); the same floor-aware verdict is recomputed for version 2's localized set from its own
json entries. --loewdin (control): the sector readouts repeated on Phi G^(-1/2) (the first route's Loewdin step), beside the as-stored ones.
Every per-mode array goes to the npz (the outside-weight spectra of both halves, the half labels, M's diagonal and eigenvalues, the dust's
diagonal, the sector's Killing energies diag(F^dagger h_m F), the eigenvalues of M - Pi, the hull's singular values).
Outputs: version 2's (<tag>_T<T/rs>.json and .npz with row["v3"] and the v3 arrays added; <tag>_all.json; <tag>_table.md; <tag>_table_v2.md
unchanged, for the comparison with run r2b) plus <tag>_table_v3.md (dust_tail_table_v3: the sector beside the first route's S_hull, d_F,
counts and bound at every clock its jsons carry, T/r_s 40 included when --first-route-t40-json names Dv2_prod10.json, and version 2's sections
with their headers relabelled). The log prints per channel "v3 kappa k: ... F 2046 (1023, 1023; w_min ...), ... S_F ... vs bound ... (ratio;
floors; verdict); S_hull ... (hull, kept, dropped bound) ..." and per slice "v3 weighted beyond 16 r_s: ...".
Usage: version 2's arguments plus [--eps-loc-control 0.001] [--eps-rank 1e-9] [--eps-control 1e-6] [--loewdin]
[--first-route-t40-json FILE]; --selftest (dust_tail_selftest_v3); --check SLICE.npz (version 1's)."""
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
import socket  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg as sla  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dust_tail_v1 as V1  # noqa: E402
import dust_tail_ops_v1 as OPS  # noqa: E402
import dust_tail_table_v1 as TBL1  # noqa: E402
import dust_tail_v2 as V2  # noqa: E402
import dust_tail_table_v2 as TBL2  # noqa: E402
import dust_tail_table_v3 as TBL3  # noqa: E402

fmt = TBL1.fmt
utc = TBL1.utc
md5 = V1.md5
h2 = V2.h2
spectral_entropy = V2.spectral_entropy
bound_dh2 = V2.bound_dh2
eigvals01 = V2.eigvals01
herm = V2.herm
fi = TBL3.fi
minnone = TBL3.minnone
EPS_LOC = V2.EPS_LOC                              # 0.01: the first route's eps_loc (its --eps-loc default)
EPS_LOC_CONTROL = 1e-3                            # v3: the first route's eps_loc_control
EPS_RANK = 1e-9                                   # v3: the first route's eps_rank (the hull's threshold on |lambda|)
EPS_CONTROL = 1e-6                                # v3: the first route's eps_control
THRESH = (1e-12, 1e-9, 1e-7, 1e-6, 1e-4, 1e-2)   # v3: the first route's lambda_counts thresholds
VERDICTS = ("respected", "EXCEEDED", "below floor")
PER_MODE = ("sec_M_diag", "sec_M_diag_dust", "sec_mu_F", "sec_hF_diag", "sec_lam_MD", "sec_hull_sv")
MODULES = ("dust_tail_v1.py", "dust_tail_ops_v1.py", "dust_tail_table_v1.py", "dust_tail_v2.py", "dust_tail_table_v2.py", "dust_tail_selftest_v2.py",
           "dust_tail_table_v3.py", "dust_tail_selftest_v3.py")


# ------------------------------------------------------------------------------------------------------------------- v3: small helpers
def maxabs(A):
    A = np.asarray(A)
    return float(np.max(np.abs(A))) if A.size else 0.0


def gram_defect(X):
    """v3: max|X^dagger X - 1| of a column set (the first route's stored_orthonormality when X = [sea, dust])."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0:
        return 0.0
    G = X.conj().T @ X
    return maxabs(G - np.eye(G.shape[0]))


# ------------------------------------------------------------------------------------------------------------------- v3: the first route's sector
def outside_weight_halves(V, e, r0):
    """v3: the first route's outside_weight_halves (engine_massive_channel_D_v2.py): per spectral half of h_m (e >= 0: +1; e < 0: -1) the
    eigendecomposition (w ascending, Q) of the half's outside-weight operator W_h = V_h^dagger Pi_Y V_h, Pi_Y the projector on the rows r0 on
    (2 rows per site; the sites beyond the cut). Returns {+1: (V_+, w, Q), -1: (V_-, w, Q)}."""
    out = {}
    for hsign, sel in ((1, e >= 0), (-1, e < 0)):
        Vh = V[:, sel]
        VhY = Vh[r0:, :]
        w, Q = sla.eigh(herm(VhY.conj().T @ VhY))
        out[hsign] = (Vh, w, Q)
    return out


def escaped_sector(halves, eps_loc):
    """v3: the first route's escaped_sector: in each spectral half the eigenvectors of W_h with eigenvalue >= 1 - eps_loc, F_h = V_h Q_keep;
    F = [F_+, F_-] with the half labels +1, -1 (every column inside one half, so F^dagger P_- F is diagonal 0/1). Returns (F, half, info)."""
    Fs, hs, info = [], [], {"eps_loc": float(eps_loc)}
    for hsign in (1, -1):
        Vh, w, Q = halves[hsign]
        keep = w >= 1.0 - eps_loc
        Fs.append(Vh @ Q[:, keep])
        hs.append(np.full(int(keep.sum()), hsign, dtype=np.int64))
        lab = "plus" if hsign > 0 else "minus"
        info["dim_%s" % lab] = int(keep.sum())
        info["outside_weight_min_%s" % lab] = float(w[keep].min()) if keep.any() else None
        info["outside_weight_max_dropped_%s" % lab] = float(w[~keep].max()) if (~keep).any() else None
    F = np.concatenate(Fs, axis=1)
    half = np.concatenate(hs)
    info["dim"] = int(F.shape[1])
    return F, half, info


def hull_in_sector(half, Ukeep):
    """v3: the first route's hull of the kept eigenvectors in the sector's own coordinates, in the fallback form of its hull_coords: per spectral
    half the left singular vectors of the kept eigenvectors' rows in that half, every one kept (its "exact" mode keeps every singular vector of
    each nonzero half block: hull dim = sum over halves of min(d_h, kept), as its jsons print). Returns (L, L's half labels, the smallest
    singular value kept, the singular values)."""
    k = Ukeep.shape[0]
    outs, hs, svs = [], [], []
    for hsign in (1, -1):
        rows = np.flatnonzero(half == hsign)
        if rows.size == 0 or Ukeep.shape[1] == 0:
            continue
        Uo, so, _ = sla.svd(Ukeep[rows, :], full_matrices=False, lapack_driver="gesvd")
        if so.size == 0 or not bool((so > 0).any()):
            continue
        B = np.zeros((k, Uo.shape[1]), complex)
        B[rows, :] = Uo
        outs.append(B)
        hs.append(np.full(Uo.shape[1], hsign, dtype=np.int64))
        svs.append(so)
    if not outs:
        return np.zeros((k, 0), complex), np.zeros(0, np.int64), None, np.zeros(0)
    sv = np.concatenate(svs)
    return np.concatenate(outs, axis=1), np.concatenate(hs), float(sv.min()), sv


def rank_sensitive_bound(q, delta):
    """v3: the dropped eigenvalues' bound in the form the first route's docstring states, q h2(min(delta/q, 1/2)) (page_ops_v6.rank_sensitive_bound is not in this lane)."""
    q = int(q)
    if q <= 0:
        return 0.0
    return q * h2(min(float(delta) / q, 0.5))


def hull_entropy(M, Pi_diag, half, lam, U, eps):
    """v3: the first route's hull at one eps: the eigenvectors of M - Pi with |lambda| > eps, their hull L (hull_in_sector), S_hull = tr s(L^dagger M L),
    the check that L reduces Pi (max|mu^2 - mu| of L^dagger Pi L), L's orthonormality, the dropped eigenvalues' trace norm and bound."""
    keep = np.abs(lam) > eps
    kk = int(keep.sum())
    drop = lam[~keep]
    q = int(drop.size)
    delta = float(np.sum(np.abs(drop)))
    res = {"eps": float(eps), "kept": kk, "dropped": q, "dropped_trace_norm": delta, "dropped_bound": rank_sensitive_bound(q, delta), "singular_values": np.zeros(0)}
    if kk == 0:
        res.update({"hull_dim": 0, "S_hull": 0.0, "vacuum_check": 0.0, "hull_orthonormality": 0.0, "s_min_kept": None})
        return res
    L, _hL, smin, sv = hull_in_sector(half, U[:, keep])
    ML = herm(L.conj().T @ (M @ L))
    mv = np.linalg.eigvalsh(herm(L.conj().T @ (Pi_diag[:, None] * L)))
    res.update({"hull_dim": int(L.shape[1]), "S_hull": spectral_entropy(eigvals01(ML)), "vacuum_check": maxabs(mv * mv - mv),
                "hull_orthonormality": gram_defect(L), "s_min_kept": smin, "singular_values": sv})
    return res


# ------------------------------------------------------------------------------------------------------------------- v3: the floors and the verdict
def floors(d, eta_columns, eta_eigensolver):
    """v3: the resolution floor of a d-mode sector: eta = the larger of the stored columns' orthonormality defect and the eigensolver's;
    n_floor = d eta (a count's roundoff), S_floor = d h2(eta) (an eigenvalue error eta at 0 or 1 enters the entropy as h2(eta) = eta (1 - ln eta) + ...)."""
    eta = float(max(float(eta_columns), float(eta_eigensolver), 0.0))
    d = int(d)
    return {"eta_columns": float(eta_columns), "eta_eigensolver": float(eta_eigensolver), "eta": eta, "n_floor": d * eta, "S_floor": (d * h2(eta)) if d else 0.0}


def verdict(S, bound, n, fl):
    """v3: 'respected' or 'EXCEEDED' only when S and n both stand above their floors; 'below floor' otherwise. The bound's tolerance is S_floor
    (version 2's absolute 1e-12 made the printed flag and the printed ratio disagree)."""
    if not (S > fl["S_floor"] and n > fl["n_floor"]):
        return "below floor"
    return "respected" if S <= bound * (1.0 + 1e-9) + fl["S_floor"] else "EXCEEDED"


def localized_set_verdict(dv, eta_columns):
    """v3: the floor-aware verdict for version 2's localized set from its own dev entries (d_F, S_F, bound_F, bound_F_blocks, n_F, the
    F_+^dagger P_- F_+ and F_-^dagger P_- F_- - 1 deviations) and the channel's stored-column defect."""
    eta_eig = max(float(dv.get("vacuum_plus_overlap_max", 0.0)), float(dv.get("vacuum_minus_identity_dev", 0.0)))
    fl = floors(int(dv["d_F"]), eta_columns, eta_eig)
    fl.update({"d_F": int(dv["d_F"]), "S_F": float(dv["S_F"]), "n_F": float(dv["n_F"]), "bound_F": float(dv["bound_F"]),
               "S_F_over_bound_F": (float(dv["S_F"]) / float(dv["bound_F"])) if float(dv["bound_F"]) > 0 else None,
               "verdict": verdict(float(dv["S_F"]), float(dv["bound_F"]), float(dv["n_F"]), fl),
               "verdict_blocks": verdict(float(dv["S_F"]), float(dv["bound_F_blocks"]), float(dv["n_F"]), fl),
               "verdict_printed_by_v2": "respected" if dv.get("bound_F_respected") else "EXCEEDED"})
    return fl


# ------------------------------------------------------------------------------------------------------------------- v3: the readouts on F
def sector_readouts(F, half, Phi, Pf, nsea, hm, Vn, eps_rank, eps_control, with_hull=True):
    """v3: the first route's sector_entropy on F (M = F^dagger C F, S_F = tr s(M), the sector particles and holes, E_F, F_orthonormality,
    lambda_counts, the hulls) with the task's additions (the dust's and the pairs' parts of the counts, the bound and its block form, the vacuum
    identities, the dust-only S_F,dust). C = Phi Pf Phi^dagger with Phi = [sea (nsea columns), dust], Pf the occupations (1 on the sea). Returns a
    dict whose "arrays" entry holds the per-mode arrays (M's diagonal, the dust's diagonal, M's eigenvalues, diag(F^dagger h_m F), the
    eigenvalues of M - Pi, the hull's singular values)."""
    k = int(F.shape[1])
    sp = half > 0
    sm = half < 0
    dplus = int(sp.sum())
    dminus = int(sm.sum())
    out = {"dim": k, "dim_plus": dplus, "dim_minus": dminus}
    if k == 0:
        out.update({"S_F": 0.0, "mu_min": None, "mu_max_minus_1": None, "n_plus": 0.0, "n_minus": 0.0, "n_F": 0.0, "n_plus_dust": 0.0, "n_plus_pairs": 0.0,
                    "n_minus_pairs": 0.0, "n_minus_dust_occ": 0.0, "pairs_count_F": 0.0, "dust_count_F_net": 0.0, "bound_F": 0.0, "bound_F_blocks": 0.0,
                    "S_F_over_bound_F": None, "E_F": 0.0, "F_orthonormality": 0.0, "vacuum_plus_overlap_max": 0.0, "vacuum_minus_identity_dev": 0.0,
                    "S_F_dust": 0.0, "w_F_dust": 0.0, "bound_F_dust_rank": 0.0, "bound_F_dustcount": 0.0, "lambda_counts": {"%g" % t: 0 for t in THRESH},
                    "lambda_max_abs": 0.0, "hulls": {}, "S_hull": 0.0 if with_hull else None, "hull_dim": 0 if with_hull else None, "hull_kept": 0 if with_hull else None,
                    "dropped_bound": 0.0 if with_hull else None, "S_hull_minus_S_F": 0.0 if with_hull else None, "arrays": {nm: np.zeros(0) for nm in PER_MODE}})
        return out
    nd = int(Phi.shape[1] - nsea)
    A = F.conj().T @ Phi
    As = A[:, :nsea]
    Ad = A[:, nsea:]
    MA = herm(As @ As.conj().T)
    MB = herm((Ad * Pf[None, nsea:]) @ Ad.conj().T) if nd else np.zeros((k, k), complex)
    del A, As
    M = herm(MA + MB)
    mu_raw = np.linalg.eigvalsh(M)
    mu = np.clip(mu_raw, 0.0, 1.0)
    S_F = spectral_entropy(mu)
    dM = np.real(np.diag(M))
    dA = np.real(np.diag(MA))
    dB = np.real(np.diag(MB))
    n_plus = float(dM[sp].sum())
    n_minus = float(dminus - dM[sm].sum())
    n_plus_dust = float(dB[sp].sum())
    n_plus_pairs = float(dA[sp].sum())
    n_minus_dust_occ = float(dB[sm].sum())
    n_minus_pairs = float(dminus - dA[sm].sum())
    n_F = n_plus + n_minus
    Pi_diag = sm.astype(float)
    MD = M - np.diag(Pi_diag)
    hF = herm(F.conj().T @ np.asarray(hm @ F))
    E_F = float(np.real(np.sum(hF * MD.T)))
    F_orth = maxabs(F.conj().T @ F - np.eye(k))
    vac_plus = maxabs(F[:, sp].conj().T @ Vn) if dplus else 0.0
    if dminus:
        Gm = F[:, sm].conj().T @ Vn
        vac_minus = maxabs(Gm @ Gm.conj().T - np.eye(dminus))
        del Gm
    else:
        vac_minus = 0.0
    muB = eigvals01(MB) if nd else np.zeros(0)
    S_F_dust = spectral_entropy(muB)
    w_F_dust = float(dB.sum())
    bound_F = bound_dh2(n_F, k)
    out.update({"S_F": S_F, "mu_min": float(mu_raw.min()), "mu_max_minus_1": float(mu_raw.max() - 1.0), "n_plus": n_plus, "n_minus": n_minus, "n_F": n_F,
                "n_plus_dust": n_plus_dust, "n_plus_pairs": n_plus_pairs, "n_minus_pairs": n_minus_pairs, "n_minus_dust_occ": n_minus_dust_occ,
                "pairs_count_F": n_plus_pairs + n_minus_pairs, "dust_count_F_net": n_plus_dust - n_minus_dust_occ,
                "bound_F": bound_F, "bound_F_blocks": bound_dh2(n_plus, dplus) + bound_dh2(n_minus, dminus), "S_F_over_bound_F": (S_F / bound_F) if bound_F > 0 else None,
                "E_F": E_F, "F_orthonormality": F_orth, "vacuum_plus_overlap_max": vac_plus, "vacuum_minus_identity_dev": vac_minus,
                "S_F_dust": S_F_dust, "w_F_dust": w_F_dust, "bound_F_dust_rank": bound_dh2(w_F_dust, nd) if nd else 0.0, "bound_F_dustcount": bound_dh2(w_F_dust, k)})
    lam, U = sla.eigh(herm(MD))
    out["lambda_counts"] = {"%g" % t: int(np.sum(np.abs(lam) > t)) for t in THRESH}
    out["lambda_max_abs"] = maxabs(lam)
    out["hulls"] = {}
    sv = np.zeros(0)
    if with_hull:
        for eps in (eps_rank, eps_control):
            res = hull_entropy(M, Pi_diag, half, lam, U, eps)
            s_ = res.pop("singular_values")
            if eps == eps_rank:
                sv = s_
            out["hulls"]["%g" % eps] = res
        prim = out["hulls"]["%g" % eps_rank]
        out.update({"S_hull": prim["S_hull"], "hull_dim": prim["hull_dim"], "hull_kept": prim["kept"], "dropped_bound": prim["dropped_bound"], "S_hull_minus_S_F": prim["S_hull"] - S_F})
    else:
        out.update({"S_hull": None, "hull_dim": None, "hull_kept": None, "dropped_bound": None, "S_hull_minus_S_F": None})
    out["arrays"] = {"sec_M_diag": dM, "sec_M_diag_dust": dB, "sec_mu_F": mu_raw, "sec_hF_diag": np.real(np.diag(hF)), "sec_lam_MD": lam, "sec_hull_sv": sv}
    return out


# ------------------------------------------------------------------------------------------------------------------- v3: one channel
def channel_pass_v3(Z, c2, c3, arrays, hm, sea, dust, f, k, nd, cuts, eps_loc, eps_loc_control, eps_rank, eps_control, use_loewdin):
    """v3: one dense eigendecomposition of h_m (its own; version 2's eigenvalues compared when its deviation pass ran) and, per cut, the first
    route's escaped sector at eps_loc (with the hull) and at eps_loc_control (without), the readouts, the floors and the verdicts; version 2's
    localized set gets its floor-aware verdict from its own dev entries. The per-mode arrays go to `arrays` under sec_* and w_half_* names."""
    t0 = time.time()
    e_all, V = OPS.full_spectrum(hm, vectors=True)
    e_all = np.asarray(e_all, float)
    V = np.asarray(V)
    neg = e_all < 0
    es = np.sort(e_all)
    c3["n_negative"] = int(neg.sum())
    c3["spectrum_symmetry_max_dev"] = float(np.max(np.abs(es + es[::-1])))
    if ("e_%d" % k) in arrays:
        c3["e_vs_v2_max_dev"] = float(np.max(np.abs(es - np.sort(np.asarray(arrays["e_%d" % k], float)))))
    else:
        c3["e_vs_v2_max_dev"] = None
        arrays["e_%d" % k] = e_all
    c3["e_vs_stored_max_dev"] = float(np.max(np.abs(es - np.sort(np.asarray(Z["e_%d" % k], float))))) if ("e_%d" % k) in Z.files else None
    nsea = int(sea.shape[1])
    Phi = np.concatenate([sea, dust], axis=1) if nd else sea
    Pf = np.concatenate([np.ones(nsea), np.asarray(f, float)]) if nd else np.ones(nsea)
    G = Phi.conj().T @ Phi
    c3["stored_orthonormality"] = maxabs(G - np.eye(G.shape[0]))
    c3["sea_gram_defect"] = maxabs(G[:nsea, :nsea] - np.eye(nsea))
    c3["dust_gram_defect"] = maxabs(G[nsea:, nsea:] - np.eye(nd)) if nd else 0.0
    c3["dust_sea_overlap_max"] = maxabs(G[:nsea, nsea:]) if nd else 0.0
    Phi_o = None
    if use_loewdin:
        gw, gv = sla.eigh(herm(G))
        Phi_o = Phi @ ((gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T)
        c3["loewdin_orthonormality"] = gram_defect(Phi_o)
        c3["gram_eigenvalue_range"] = [float(gw.min()), float(gw.max())]
    del G
    eta_cols = c3["stored_orthonormality"]
    for c, p in cuts:
        key = str(c)
        halves = outside_weight_halves(V, e_all, 2 * p)
        Vn = halves[-1][0]
        arrays["w_half_plus_%g_%d" % (c, k)] = halves[1][1]
        arrays["w_half_minus_%g_%d" % (c, k)] = halves[-1][1]
        for eloc, primary in ((eps_loc, True), (eps_loc_control, False)):
            if not primary and not (eloc and eloc > 0.0):
                c3["sector_control"][key] = None
                continue
            F, half, info = escaped_sector(halves, eloc)
            sec = sector_readouts(F, half, Phi, Pf, nsea, hm, Vn, eps_rank, eps_control, with_hull=primary)
            per_mode = sec.pop("arrays")
            sec.update(info)
            sec["floors"] = floors(sec["dim"], eta_cols, max(sec["F_orthonormality"], sec["vacuum_plus_overlap_max"], sec["vacuum_minus_identity_dev"]))
            sec["verdict"] = verdict(sec["S_F"], sec["bound_F"], sec["n_F"], sec["floors"])
            sec["verdict_blocks"] = verdict(sec["S_F"], sec["bound_F_blocks"], sec["n_F"], sec["floors"])
            if Phi_o is not None:
                lo = sector_readouts(F, half, Phi_o, Pf, nsea, hm, Vn, eps_rank, eps_control, with_hull=False)
                lo.pop("arrays")
                sec["loewdin"] = {q: lo[q] for q in ("S_F", "n_plus", "n_minus", "n_F", "n_plus_dust", "n_minus_dust_occ", "E_F", "bound_F", "S_F_dust", "w_F_dust", "mu_min", "mu_max_minus_1")}
                sec["loewdin"]["S_F_minus_stored"] = lo["S_F"] - sec["S_F"]
                sec["loewdin"]["n_F_minus_stored"] = lo["n_F"] - sec["n_F"]
            if primary:
                c3["sector"][key] = sec
                arrays["sec_half_%g_%d" % (c, k)] = half
                for nm, arr in per_mode.items():
                    arrays["%s_%g_%d" % (nm, c, k)] = arr
            else:
                c3["sector_control"][key] = sec
            del F
        del halves, Vn
        dv = (c2.get("dev") or {}).get(key) if c2 else None
        c3["loc_verdict"][key] = localized_set_verdict(dv, eta_cols) if dv else None
    del V
    c3["seconds"] = time.time() - t0


def channel_line(c3, c0, eps_c):
    key = str(c0)
    s = c3["sector"].get(key) or {}
    sc = (c3.get("sector_control") or {}).get(key) or {}
    fl = s.get("floors") or {}
    return ("  v3 kappa %d: stored Gram - 1 %.1e; beyond %g r_s: F %d (%d, %d; w_min %s), n_+ %.4e (dust %.4e), n_- %.4e, S_F %.6e vs bound %.6e (ratio %s; floors n %.1e, S %.1e; %s); "
            "S_hull %s (hull %s, kept %s, dropped bound %s); E_F %+.4e; F orth %.1e, vac %.1e / %.1e; control eps_loc %s: F %s, S_F %s; %.1f s" % (
                c3["kappa"], c3["stored_orthonormality"], c0, s.get("dim", 0), s.get("dim_plus", 0), s.get("dim_minus", 0), fmt(minnone(s.get("outside_weight_min_plus"), s.get("outside_weight_min_minus")), "%.4f"),
                s.get("n_plus", 0.0), s.get("n_plus_dust", 0.0), s.get("n_minus", 0.0), s.get("S_F", 0.0), s.get("bound_F", 0.0), fmt(s.get("S_F_over_bound_F"), "%.4f"), fl.get("n_floor", 0.0), fl.get("S_floor", 0.0),
                s.get("verdict", "-"), fmt(s.get("S_hull"), "%.6e"), fi(s.get("hull_dim")), fi(s.get("hull_kept")), fmt(s.get("dropped_bound"), "%.1e"), s.get("E_F", 0.0), s.get("F_orthonormality", 0.0),
                s.get("vacuum_plus_overlap_max", 0.0), s.get("vacuum_minus_identity_dev", 0.0), fmt(eps_c, "%g"), fi(sc.get("dim")), fmt(sc.get("S_F"), "%.6e"), c3["seconds"]))


def slice_line(v3, c0):
    q = v3["weighted"][str(c0)]
    vd = q["verdicts"]
    msg = ("  v3 weighted beyond %g r_s: F d_F %d (w %g), n_+ %.6e (dust %.6e), n_- %.6e, n_F %.6e (pairs' share %s), S_F %.6e vs bound %.6e (ratio %s, max channel %s; floors n %.1e, S %.1e; "
           "%d respected / %d EXCEEDED / %d below floor); S_hull %.6e (hull dim %d; dropped bound %.1e); E_F %+.6e" % (
               c0, q["d_F_unw"], q["d_F_w"], q["n_plus"], q["n_plus_dust"], q["n_minus"], q["n_F"], fmt(q["pairs_share_of_n_F"], "%.4f"), q["S_F"], q["bound_F"], fmt(q["S_F_over_bound_F"], "%.4f"),
               fmt(q["max_S_F_over_bound_F"], "%.4f"), q["n_floor_w"], q["S_floor_w"], vd["respected"], vd["EXCEEDED"], vd["below floor"], q["S_hull"], q["hull_dim_unw"], q["dropped_bound"], q["E_F"]))
    if "block_count" in q:
        msg += "; block-basis count %.6e (n_F minus it %+.3e); region excess %.6e" % (q["block_count"], q["n_F_minus_block_count"], q["region_excess"])
    if "control" in q:
        msg += "; control eps_loc %g: d_F %d, S_F %.6e" % (q["control"]["eps_loc"], q["control"]["d_F_unw"], q["control"]["S_F"])
    if "loewdin" in q:
        msg += "; Loewdin control: S_F %.6e (minus as stored %+.3e)" % (q["loewdin"]["S_F"], q["loewdin"]["S_F_minus_stored"])
    if "localized_set" in q:
        lv = q["localized_set"]["verdicts"]
        msg += "; version 2's localized set: %d respected / %d EXCEEDED / %d below floor" % (lv["respected"], lv["EXCEEDED"], lv["below floor"])
    return msg + "; v3 pass %.0f s" % v3["seconds"]


# ------------------------------------------------------------------------------------------------------------------- v3: the weighted sums
def weigh_v3(row, v3, cuts):
    """v3: the weighted sums (w = 16 kappa) per cut, the checks' maxima, the verdict counts, the comparisons with the block basis and the region
    comparator of version 2, the eps_loc control, the Loewdin control and version 2's localized set with its floor-aware verdicts."""
    chans = v3["channels"]
    eps_rank = v3["eps_rank"]
    eps_control = v3["eps_control"]
    for c, p in cuts:
        key = str(c)
        secs = [(c3["w"], c3["sector"][key]) for c3 in chans]

        def W(name, src=secs):
            return float(sum(w_ * float(s[name] or 0.0) for w_, s in src))

        q = {name: W(name) for name in ("n_plus", "n_minus", "n_F", "n_plus_dust", "n_plus_pairs", "n_minus_pairs", "n_minus_dust_occ", "pairs_count_F", "dust_count_F_net",
                                         "S_F", "bound_F", "bound_F_blocks", "E_F", "S_F_dust", "w_F_dust", "bound_F_dust_rank", "bound_F_dustcount", "S_hull", "dropped_bound")}
        q["d_F_unw"] = int(sum(s["dim"] for _, s in secs))
        q["d_F_w"] = float(sum(w_ * s["dim"] for w_, s in secs))
        q["d_plus_unw"] = int(sum(s["dim_plus"] for _, s in secs))
        q["d_minus_unw"] = int(sum(s["dim_minus"] for _, s in secs))
        q["hull_dim_unw"] = int(sum(int(s["hull_dim"] or 0) for _, s in secs))
        q["S_hull_control"] = float(sum(w_ * s["hulls"]["%g" % eps_control]["S_hull"] for w_, s in secs if ("%g" % eps_control) in s["hulls"]))
        q["n_floor_w"] = float(sum(w_ * s["floors"]["n_floor"] for w_, s in secs))
        q["S_floor_w"] = float(sum(w_ * s["floors"]["S_floor"] for w_, s in secs))
        q["verdicts"] = {vv: int(sum(1 for _, s in secs if s["verdict"] == vv)) for vv in VERDICTS}
        q["verdicts_blocks"] = {vv: int(sum(1 for _, s in secs if s["verdict_blocks"] == vv)) for vv in VERDICTS}
        q["all_respected"] = all(s["verdict"] == "respected" for _, s in secs)
        q["verdict"] = "respected" if q["all_respected"] else ("EXCEEDED" if q["verdicts"]["EXCEEDED"] else "below floor")
        q["max_S_F_over_bound_F"] = max((s["S_F_over_bound_F"] for _, s in secs if s["S_F_over_bound_F"] is not None), default=None)
        q["S_F_over_bound_F"] = (q["S_F"] / q["bound_F"]) if q["bound_F"] > 0 else None
        q["bound_F_w_form"] = bound_dh2(q["n_F"], q["d_F_w"])
        q["bound_F_review_form"] = bound_dh2(q["n_F"], q["d_F_unw"])
        q["bound_F_dustcount_w_form"] = bound_dh2(q["w_F_dust"], q["d_F_w"])
        q["bound_F_dustcount_review_form"] = bound_dh2(q["w_F_dust"], q["d_F_unw"])
        q["pairs_share_of_n_F"] = (q["pairs_count_F"] / q["n_F"]) if q["n_F"] > 0 else None
        q["dust_share_of_n_F"] = (q["dust_count_F_net"] / q["n_F"]) if q["n_F"] > 0 else None
        wmins = [minnone(s.get("outside_weight_min_plus"), s.get("outside_weight_min_minus")) for _, s in secs]
        q["outside_weight_min"] = min((x for x in wmins if x is not None), default=None)
        q["max_stored_orthonormality"] = max(c3["stored_orthonormality"] for c3 in chans)
        q["max_F_orthonormality"] = max(s["F_orthonormality"] for _, s in secs)
        q["max_vacuum_plus_overlap"] = max(s["vacuum_plus_overlap_max"] for _, s in secs)
        q["max_vacuum_minus_identity_dev"] = max(s["vacuum_minus_identity_dev"] for _, s in secs)
        hulls = [(s["hulls"].get("%g" % eps_rank) or {}) for _, s in secs]
        q["max_hull_vacuum_check"] = max((h.get("vacuum_check", 0.0) for h in hulls), default=0.0)
        q["max_hull_orthonormality"] = max((h.get("hull_orthonormality", 0.0) for h in hulls), default=0.0)
        q["max_abs_S_hull_minus_S_F"] = max((abs(s["S_hull_minus_S_F"] or 0.0) for _, s in secs), default=0.0)
        q["S_hull_minus_S_F"] = q["S_hull"] - q["S_F"]
        q["max_mu_max_minus_1"] = max((s["mu_max_minus_1"] for _, s in secs if s["mu_max_minus_1"] is not None), default=None)
        q["min_mu_min"] = min((s["mu_min"] for _, s in secs if s["mu_min"] is not None), default=None)
        dev = (((row.get("v2") or {}).get("weighted") or {}).get(key) or {}).get("dev")
        if dev:
            q["block_count"] = float(dev["particles_dev"] + dev["holes_dev"])
            q["n_F_minus_block_count"] = q["n_F"] - q["block_count"]
            q["region_excess"] = float(dev["region_excess"])
            q["S_F_minus_region_excess"] = q["S_F"] - q["region_excess"]
        ctrl = [(c3["w"], c3["sector_control"].get(key)) for c3 in chans]
        if ctrl and all(s is not None for _, s in ctrl):
            q["control"] = {"eps_loc": v3["eps_loc_control"], "d_F_unw": int(sum(s["dim"] for _, s in ctrl)), "d_F_w": float(sum(w_ * s["dim"] for w_, s in ctrl)),
                            "S_F": W("S_F", ctrl), "n_F": W("n_F", ctrl), "n_plus": W("n_plus", ctrl), "n_minus": W("n_minus", ctrl), "E_F": W("E_F", ctrl), "bound_F": W("bound_F", ctrl),
                            "verdicts": {vv: int(sum(1 for _, s in ctrl if s["verdict"] == vv)) for vv in VERDICTS}}
        lws = [(w_, s.get("loewdin")) for w_, s in secs if s.get("loewdin")]
        if v3["loewdin"] and len(lws) == len(secs) and lws:
            q["loewdin"] = {name: float(sum(w_ * lo[name] for w_, lo in lws)) for name in ("S_F", "n_F", "n_plus", "n_minus", "E_F", "bound_F", "S_F_minus_stored", "n_F_minus_stored")}
            q["loewdin"]["max_loewdin_orthonormality"] = max(float(c3.get("loewdin_orthonormality", 0.0)) for c3 in chans)
        locs = [(c3["w"], c3["loc_verdict"].get(key)) for c3 in chans]
        if locs and all(lv is not None for _, lv in locs):
            q["localized_set"] = {"n_floor_w": float(sum(w_ * lv["n_floor"] for w_, lv in locs)), "S_floor_w": float(sum(w_ * lv["S_floor"] for w_, lv in locs)),
                                  "verdicts": {vv: int(sum(1 for _, lv in locs if lv["verdict"] == vv)) for vv in VERDICTS},
                                  "verdicts_blocks": {vv: int(sum(1 for _, lv in locs if lv["verdict_blocks"] == vv)) for vv in VERDICTS},
                                  "max_S_F_over_bound_F": max((lv["S_F_over_bound_F"] for _, lv in locs if lv["S_F_over_bound_F"] is not None), default=None)}
        v3["weighted"][key] = q
    v3["max_e_vs_v2"] = max((c3["e_vs_v2_max_dev"] for c3 in chans if c3.get("e_vs_v2_max_dev") is not None), default=None)
    v3["max_e_vs_stored"] = max((c3["e_vs_stored_max_dev"] for c3 in chans if c3.get("e_vs_stored_max_dev") is not None), default=None)
    v3["max_spectrum_symmetry"] = max(c3["spectrum_symmetry_max_dev"] for c3 in chans)
    v3["max_stored_orthonormality"] = max(c3["stored_orthonormality"] for c3 in chans)
    v3["n_negative"] = sorted({c3["n_negative"] for c3 in chans})


# ------------------------------------------------------------------------------------------------------------------- one slice
def process_slice_v3(path, m, cuts_rs, prof_rs, spectrum="dust", deviation=True, eps_loc=EPS_LOC, eps_loc_control=EPS_LOC_CONTROL, eps_rank=EPS_RANK,
                     eps_control=EPS_CONTROL, loewdin=False, log=print):
    """version 2's process_slice_v2 (its outputs unchanged: version 1's readouts, the v2 readouts and, with deviation, its localized set, the region
    block and the Killing block) followed by the v3 pass on every channel; returns (row, arrays) with row["v3"] and the v3 arrays added."""
    t_s = time.time()
    row, arrays = V2.process_slice_v2(path, m, cuts_rs, prof_rs, spectrum, deviation, eps_loc, log=log)
    t_3 = time.time()
    Z = np.load(path)
    R = np.asarray(arrays["R"], float)
    Napp = arrays["N_applied"]
    vapp = arrays["v_applied"]
    cuts = [(c, int(cc["first_site_index"])) for c, cc in zip(cuts_rs, row["cuts"])]
    v3 = {"eps_loc": float(eps_loc), "eps_loc_control": (float(eps_loc_control) if eps_loc_control else None), "eps_rank": float(eps_rank), "eps_control": float(eps_control),
          "loewdin": bool(loewdin), "deviation_v2": bool(deviation), "channels": [], "weighted": {}}
    for ch, c2 in zip(row["channels"], row["v2"]["channels"]):
        k = int(ch["kappa"])
        w = float(ch["w"])
        nd = int(ch["n_dust"])
        hm = OPS.killing_h(R, m, k, Napp, vapp)
        sea = np.asarray(Z["sea_%d" % k], np.complex128)
        dust = np.asarray(Z["dust_%d" % k], np.complex128)
        f = np.asarray(Z["f_%d" % k], float)
        c3 = {"kappa": k, "w": w, "n_dust": nd, "sector": {}, "sector_control": {}, "loc_verdict": {}}
        channel_pass_v3(Z, c2, c3, arrays, hm, sea, dust, f, k, nd, cuts, eps_loc, eps_loc_control, eps_rank, eps_control, loewdin)
        del sea, dust
        v3["channels"].append(c3)
        log(channel_line(c3, cuts[0][0], eps_loc_control))
    Z.close()
    weigh_v3(row, v3, cuts)
    v3["seconds"] = time.time() - t_3
    v3["seconds_total"] = time.time() - t_s
    row["v3"] = v3
    log(slice_line(v3, cuts[0][0]))
    return row, arrays


# ------------------------------------------------------------------------------------------------------------------- main
def default_args(**kw):
    a = argparse.Namespace(slices="", m=0.2, out_dir="", tag="dust_tail_v3", cuts="16,24", profile="4,8,12,16,20,24,28", run="", e0_run="", E0=None,
                           first_route_tail_json="", first_route_dv2_jsons="", first_route_t40_json="", spectrum="dust", deviation=False, eps_loc=EPS_LOC,
                           eps_loc_control=EPS_LOC_CONTROL, eps_rank=EPS_RANK, eps_control=EPS_CONTROL, loewdin=False, threads=4, overwrite=False, selftest=False, check="")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def run_main(a, log=print):
    slices = [p for p in a.slices.split(",") if p]
    if not slices or not a.out_dir:
        sys.exit("--slices and --out-dir are required")
    os.makedirs(a.out_dir, exist_ok=True)
    cuts_rs = [float(x) for x in a.cuts.split(",")]
    prof_rs = [float(x) for x in a.profile.split(",")]
    for name in ("%s_table.md" % a.tag, "%s_table_v2.md" % a.tag, "%s_table_v3.md" % a.tag, "%s_all.json" % a.tag):
        if os.path.exists(os.path.join(a.out_dir, name)) and not a.overwrite:
            sys.exit("refusing to overwrite %s (use a new --tag)" % os.path.join(a.out_dir, name))
    try:
        load = open("/proc/loadavg").read().split()[:3]
    except OSError:
        load = None
    prov = {"script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)),
            "modules": ", ".join("%s %s" % (x, md5(os.path.join(HERE, x))) for x in MODULES if os.path.exists(os.path.join(HERE, x))),
            "args": " ".join(sys.argv[1:]), "host": socket.gethostname(), "utc_start": utc(), "load_at_start": load, "threads": a.threads, "numpy": np.__version__, "scipy": scipy.__version__,
            "python": sys.version.split()[0], "m": a.m, "spectrum": a.spectrum, "deviation": bool(a.deviation), "eps_loc": a.eps_loc, "eps_loc_control": a.eps_loc_control,
            "eps_rank": a.eps_rank, "eps_control": a.eps_control, "loewdin": bool(a.loewdin), "E0_typed": a.E0, "e0_run": a.e0_run, "b": V2.band_top(a.m),
            "first_route_t40_json": a.first_route_t40_json}
    run = V1.load_run(a.run) if a.run else None
    if run is not None and abs(float(run[0]["args"]["m"]) - a.m) > 1e-12:
        sys.exit("--m %g differs from the run's m %g" % (a.m, float(run[0]["args"]["m"])))
    e0 = V2.energy_scale(run, a.e0_run, a.E0, a.m)
    prov["E_0"] = e0
    log("E_0: %s (%s); E_0/m %s" % (fmt(e0.get("E_0"), "%.6f"), e0.get("E_0_source"), fmt(e0.get("E_0_over_m"), "%.4f")))
    first = V1.load_first_route(a.first_route_tail_json, a.first_route_dv2_jsons)
    rows = []
    for p in slices:
        row, arrays = process_slice_v3(p, a.m, cuts_rs, prof_rs, a.spectrum, a.deviation, a.eps_loc, a.eps_loc_control, a.eps_rank, a.eps_control, a.loewdin, log=log)
        if run is not None:
            row["run_checks"] = V1.run_checks(row, arrays["N_applied"], arrays["v_applied"], run[0], run[1], cuts_rs)
            rb = row["run_checks"].get("beyond") or {}
            log("  run checks: applied fields dev %s; K_density - stored K_dust %s; bare energy - E_dust_shells %s; E_G %s, m/T_H %s" % (
                row["run_checks"].get("applied_fields_max_dev"), {c: "%+.3e (rel %s)" % (q["diff"], fmt(q["rel"], "%.2e")) for c, q in rb.items()},
                fmt(row["run_checks"].get("E_dust_shells_diff"), "%+.3e"), fmt(row["run_checks"].get("E_G"), "%.6f"), fmt(row["run_checks"].get("m_over_TH"), "%.3f")))
        row["run_checks_v2"] = dict(e0)
        V2.add_energy_fractions(row, e0)
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
    json.dump({"script": "dust_tail_v3.py", "provenance": prov, "cuts_rs": cuts_rs, "profile_rs": prof_rs, "rows": rows},
              open(os.path.join(a.out_dir, "%s_all.json" % a.tag), "w"), indent=1, default=float)
    out2, _ = TBL2.compose_from_outputs(a.out_dir, a.tag, a.first_route_tail_json, a.first_route_dv2_jsons)
    out3, _ = TBL3.compose_from_outputs(a.out_dir, a.tag, a.first_route_tail_json, a.first_route_dv2_jsons, a.first_route_t40_json)
    log("wrote %s, %s, %s and %s" % (os.path.join(a.out_dir, "%s_table.md" % a.tag), os.path.join(a.out_dir, "%s_all.json" % a.tag), out2, out3))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=""); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--out-dir", default=""); ap.add_argument("--tag", default="dust_tail_v3")
    ap.add_argument("--cuts", default="16,24"); ap.add_argument("--profile", default="4,8,12,16,20,24,28"); ap.add_argument("--run", default=""); ap.add_argument("--e0-run", default="")
    ap.add_argument("--E0", type=float, default=None); ap.add_argument("--first-route-tail-json", default=""); ap.add_argument("--first-route-dv2-jsons", default="")
    ap.add_argument("--first-route-t40-json", default="", help="v3: the first route's json carrying T/r_s 40 (Dv2_prod10.json), read by the table script only")
    ap.add_argument("--spectrum", default="dust", choices=["dust", "all", "none"]); ap.add_argument("--deviation", action="store_true"); ap.add_argument("--eps-loc", type=float, default=EPS_LOC)
    ap.add_argument("--eps-loc-control", type=float, default=EPS_LOC_CONTROL, help="v3: the first route's eps_loc_control (0 disables)")
    ap.add_argument("--eps-rank", type=float, default=EPS_RANK, help="v3: the first route's eps_rank for the hull")
    ap.add_argument("--eps-control", type=float, default=EPS_CONTROL, help="v3: the first route's eps_control for the hull")
    ap.add_argument("--loewdin", action="store_true", help="v3: control, the sector readouts also on the Loewdin-orthonormalized columns")
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--check", default="")
    a = ap.parse_args()
    if a.selftest:
        import dust_tail_selftest_v3 as ST
        sys.exit(0 if ST.selftest(sys.modules[__name__]) else 1)
    if a.check:
        V1.check(a.check)
        return
    run_main(a)


if __name__ == "__main__":
    main()
