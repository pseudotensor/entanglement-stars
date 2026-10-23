#!/usr/bin/env python
"""engine_single_object_A_v1.py (builder analyst; inbox #41, the revised S4 part A, verbatim in
<internal>/solve_oneobject.md; the owner-approved one-object ruling REVIEWA_RULING_oneobject_MI.md R1):
"A. Correlations across the root (post-processing; no propagation). Records: the 10 stored full states of the r_s/a 64, tau 8, sigma/a 4 history
(T/r_s 12 to 40), the sigma/a 8 member's stored slices, and the r_s/a 32 record at T/r_s 18 used by S7 part 3. Per radial channel kappa as
stored; no 16 kappa weight on any printed correlation (the g_n guard); both the dense root and the applied-field root of N + v per record, all
roots with signed slopes, the split X = sites inside the outer root, Y = outside, and the trapped annulus (between inner and outer root) as a
second X. 1. The full block K = Pi_X C Pi_Y: singular values sigma_i (all of them above 1e-12), sigma_max, ||K||_F. 2. The identity check:
||C^2 - C||_op of the stored state; ||C_X - C_X^2 - K K^dagger||_op; S_X by Tr s(C_X) and by sum_i s(lambda(sigma_i)), lambda(sigma) =
(1 - sqrt(1 - 4 sigma^2))/2, agreement to 1e-10; internal check only, labelled 'per radial channel; not the entropy coefficient'. 3. Slabs:
4-site slabs centred 2, 4 and 8 sites from the root on each side (the slab geometry of P07's DERIVATION.md sec. 10, mutual information not
computed): sigma_max(K_d) and ||K_d||_F for the state C, for the Killing vacuum P_-(h(t_k)) of the record's frozen applied fields with the vacuum
exterior appended (1 dense eigendecomposition per record and channel), for the continued reference P, and the differences C - P_- and C - P.
4. The exact cut current through the outer root, F = -2 Im Tr(h_XY D_YX) with D = C - P and with C, per channel and record, with its sign
(inward negative). 5. The outward-branch content of the model's own matter: the occupied orbitals localized inside the outer root (eigenvectors
of Phi^dagger Pi_X Phi with eigenvalue >= 0.99, Phi the occupied columns) and their weight at |ak| > pi/4 by the discrete radial transform on the
interior sites (window stated), the fraction f_> per record and channel; the same for the whole occupied set as the comparator."
No mutual information is computed anywhere (D15). LABEL (on every output): "the single object's own cross-surface 2-point block and transfer on
the solved history of the massive collapse, per radial channel; the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the
record's own limit; the massless channel, where present, a test channel under owner decision D14" (no massless channel here).

How (builder's conventions, printed in the outputs):
- The state per channel: the stored columns Phi = [sea, dust] with occupations f (1 on the sea, the stored f on the dust); C = Phi f Phi^dagger.
  ||C^2 - C||_op of the stored state = max |mu^2 - mu| over the eigenvalues mu of f^(1/2) G f^(1/2), G = Phi^dagger Phi (exact for the
  stored columns). The slices store complex64 columns, so that number is single-precision; every other readout of C uses the columns
  re-orthonormalized in float64 (Loewdin: Phi G^(-1/2)), labelled, and the identities of item 2 are checked on that projector.
- The roots: the dense root, the zeros of N + v of the dense fields re-solved from the record's stored shell totals by the run's own field
  module (the solve's fine grid to the collar, bisection), slopes by engine_rows_v2.deriv at h = ell/32; the applied-field root, the sign
  changes of N_applied + v_applied between sites, linearly interpolated, slopes the site differences. The outer root: the largest root with a
  positive slope; the inner root: the largest root below it with a negative slope. X: the sites with R < R_outer; Y: the rest; the second X:
  the sites with R_inner < R < R_outer, its complement the second Y (the identity's), its block to the outside Y printed beside.
- The slabs: P07 sec. 10's geometry, "four-site slabs whose nearest sites lie at least d inside/outside the computed root" (primary), and the
  task's wording "centred d sites from the root" (the 4 sites nearest to R_outer -+ d) beside it; d = 2, 4, 8; each slab 4 sites, 8 components.
- The Killing vacuum P_-: the negative-energy projector of h_m = (1/2){N, h} + (1/2){v, Q} at the record's applied fields (the engines' operator,
  engine_vacflow_v1.chain_sparse's stencils) on the stored chain to its end (R = nsites): the engine's exterior (N = 1, v = -(2 g E/R)^(1/2)
  beyond its collar) is already in the applied fields there; the exterior appended to 128 r_s is part B's machinery and is run here once as a
  labelled control (--vacuum-control, one record and channel: the slab blocks against the stored-chain vacuum).
- The cut current: h = h_m at the applied fields; Tr(h_XY D_YX) summed over the pairs within the stencil's reach (2 sites) of the cut;
  F = -2 Im Tr(h_XY D_YX) = dN_Y/dt (outward positive, inward negative), for D = C - P (P the stored continued reference) and for D = C.
- The momentum content: the discrete radial transform on the interior sites 1 .. iX (M = iX sites, per spinor component, orthonormal, no
  taper: a rectangular window), a k_q = 2 pi q/M in (-pi, pi], the zones by engine_s7_twocopy_v3's zone_masks (the boundary |ak| = pi/4 joins the
  upper side); f_> = the weight at |ak| >= pi/4 over the total, with the quarter-zone shares; for the localized occupied orbitals (eigenvalue
  >= 0.99 of Phi^dagger Pi_X Phi), for the whole occupied set (the comparator) and for the dust columns alone (the "whether the dust populates
  the branch" of the ruling's R1, labelled extra).
Args: --family (rs64_sigma4 | rs64_sigma8 | rs32_T18) --slices --run --fields-module --rs32-states --rs32-T0 --kappas --threads
--vacuum-control T_over_rs,kappa --out-json --md; --compose a.json,b.json,c.json --md --out-json. Every number is printed by the script."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import importlib
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_rows_v2 as RW  # noqa: E402
import engine_vacflow_v1 as VF  # noqa: E402

LABEL = ("the single object's own cross-surface 2-point block and transfer on the solved history of the massive collapse, per radial channel; "
         "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit; the massless channel, where present, "
         "a test channel under owner decision D14")
ENT_LABEL = "per radial channel; not the entropy coefficient"
DS = (2, 4, 8)


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def s_ent(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    out = np.zeros_like(x)
    m = (x > 1e-300) & (x < 1.0 - 1e-16)
    out[m] = -(x[m] * np.log(x[m]) + (1.0 - x[m]) * np.log(1.0 - x[m]))
    return out


def lam_of_sigma(s):
    s = np.asarray(s, float)
    return 0.5 * (1.0 - np.sqrt(np.clip(1.0 - 4.0 * s * s, 0.0, None)))


def zone_masks(M):
    """engine_s7_twocopy_v3.zone_masks, verbatim: the signed DFT index and the half- and quarter-zones (|ap| = pi/2 and pi/4 boundaries to the
    upper sides)."""
    qs = np.rint(np.fft.fftfreq(M) * M).astype(np.int64)
    aq = np.abs(qs)
    lo = 4 * aq < M
    quarters = [8 * aq < M, (8 * aq >= M) & (4 * aq < M), (4 * aq >= M) & (8 * aq <= 3 * M), 8 * aq > 3 * M]
    return qs, lo, ~lo, quarters, int(np.sum(4 * aq == M))


def momentum_content(Xrows, M):
    """the weight of the columns' interior components (Xrows: 2M x r, site-major) over the quarter zones of the orthonormal DFT per spinor
    component; returns the shares and f_> (|ak| >= pi/4)."""
    if Xrows.shape[1] == 0 or M == 0:
        return None
    A = Xrows.reshape(M, 2, -1)
    Ak = np.fft.fft(A, axis=0, norm="ortho")
    w = (np.abs(Ak) ** 2).sum(axis=(1, 2))
    tot = float(w.sum())
    _, _, _, quarters, nb = zone_masks(M)
    qsh = [float(w[q].sum()) / tot for q in quarters] if tot > 0 else [None] * 4
    return {"M": M, "columns": int(Xrows.shape[1]), "weight_total": tot, "f_gt": (1.0 - qsh[0]) if tot > 0 else None, "quarters": qsh,
            "outward_band_pi4_pi2": qsh[1] if tot > 0 else None}


def roots_dense(info, ell):
    """the zeros of N + v of the dense fields on the solve's fine grid to the collar, refined by bisection; slopes by engine_rows_v2.deriv."""
    rc = float(info["r_collar"]); rf = np.asarray(info["rfine"], float); rf = rf[(rf > 0) & (rf <= rc)]
    F = lambda r: np.asarray(info["N_dense"](np.atleast_1d(r)), float) + np.asarray(info["v_dense"](np.atleast_1d(r)), float)
    g = F(rf)
    out = []
    for i in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
        r0 = brentq(lambda r: float(F(r)[0]), float(rf[i]), float(rf[i + 1]), xtol=1e-12)
        _, d1, _ = RW.deriv(F, np.array([r0]), ell / 32.0)
        out.append({"R": r0, "slope": float(d1[0])})
    return out


def roots_applied(R, N, v):
    g = np.asarray(N, float) + np.asarray(v, float); R = np.asarray(R, float)
    out = []
    for i in np.nonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)[0]:
        r0 = R[i] + g[i] * (R[i + 1] - R[i]) / (g[i] - g[i + 1])
        out.append({"R": float(r0), "slope": float((g[i + 1] - g[i]) / (R[i + 1] - R[i]))})
    return out


def outer_inner(roots):
    pos = [r for r in roots if r["slope"] > 0]
    if not pos:
        return None, None
    Ro = max(r["R"] for r in pos)
    neg = [r for r in roots if r["slope"] < 0 and r["R"] < Ro]
    Ri = max(r["R"] for r in neg) if neg else None
    return Ro, Ri


def slab_sites(R, Ro, d, kind):
    """four sites: P07 (nearest site at least d from the root) or centred (the 4 sites nearest to Ro -+ d); returns (inside, outside) site indices."""
    R = np.asarray(R, float); n = len(R)
    if kind == "P07":
        ia = int(np.nonzero(R <= Ro - d)[0][-1]); A = list(range(ia - 3, ia + 1))
        ib = int(np.nonzero(R >= Ro + d)[0][0]); B = list(range(ib, ib + 4))
    else:
        ca = Ro - d; cb = Ro + d
        A = sorted(np.argsort(np.abs(R - ca))[:4].tolist()); B = sorted(np.argsort(np.abs(R - cb))[:4].tolist())
    A = [i for i in A if 0 <= i < n]; B = [i for i in B if 0 <= i < n]
    return A, B


def rows_of(sites):
    return np.array([2 * i + c for i in sites for c in (0, 1)], dtype=np.int64)


def block_stats(M):
    s = np.linalg.svd(M, compute_uv=False)
    return {"sigma_max": float(s.max()) if len(s) else 0.0, "fro": float(np.linalg.norm(M))}


def hm_host(R, m, kappa, N, v):
    hs, Qs = VF.chain_sparse(np.asarray(R, float), m, float(kappa))
    Nd = sp.diags(np.repeat(np.asarray(N, float), 2)); Vd = sp.diags(np.repeat(np.asarray(v, float), 2))
    return (0.5 * (Nd @ hs + hs @ Nd) + 0.5 * (Vd @ Qs + Qs @ Vd)).tocsr()


def channel_A(R, m, kappa, N_app, v_app, sea, dust, fdust, ref, splits, need_vacuum=True):
    """items 1 to 5 for one channel and the record's splits (a dict root kind -> (R_outer, R_inner)); returns a dict per root kind."""
    n = len(R)
    Phi = np.concatenate([np.asarray(sea, np.complex128), np.asarray(dust, np.complex128)], axis=1)
    f = np.concatenate([np.ones(sea.shape[1]), np.asarray(fdust, float)])
    G = Phi.conj().T @ Phi
    fh = np.sqrt(np.clip(f, 0.0, None))
    mu = np.linalg.eigvalsh((fh[:, None] * G) * fh[None, :])
    stored_idem = float(np.max(np.abs(mu * mu - mu)))
    gw, gv = np.linalg.eigh(G)
    Phi_o = Phi @ ((gv * (1.0 / np.sqrt(gw))[None, :]) @ gv.conj().T)
    orth_o = float(np.max(np.abs(Phi_o.conj().T @ Phi_o - np.eye(Phi.shape[1]))))
    Pf = Phi_o * f[None, :]
    refc = np.asarray(ref, np.complex128)
    out = {"kappa": int(kappa), "n_occ": int(Phi.shape[1]), "dust_occupations": [float(x) for x in fdust], "stored_idempotency": stored_idem,
           "stored_orthonormality": float(np.max(np.abs(G - np.eye(G.shape[0])))), "loewdin_orthonormality": orth_o,
           "ref_orthonormality": float(np.max(np.abs(refc.conj().T @ refc - np.eye(refc.shape[1])))), "splits": {}}
    H = hm_host(R, m, kappa, N_app, v_app)
    V = None
    if need_vacuum:
        t_ = time.time()
        e, U = np.linalg.eigh(H.toarray())
        V = U[:, e < 0]
        out["vacuum"] = {"n_negative": int(V.shape[1]), "eig_seconds": time.time() - t_, "gap_at_zero": float(e[e >= 0].min() - e[e < 0].max()) if (e >= 0).any() and (e < 0).any() else None}
        del U
    done = {}
    for kind, (Ro, Ri) in splits.items():
        if Ro is None:
            out["splits"][kind] = {"root": None}
            continue
        iX = int(np.searchsorted(np.asarray(R, float), Ro, side="left"))     # sites 0 .. iX-1 have R < Ro
        iI = int(np.searchsorted(np.asarray(R, float), Ri, side="right")) if Ri is not None else None
        key = (iX, iI)
        if key in done:
            out["splits"][kind] = {"same_split_as": done[key], "R_outer": Ro, "R_inner": Ri, "iX": iX}
            continue
        done[key] = kind
        rx = np.arange(2 * iX); ry = np.arange(2 * iX, 2 * n)
        res = {"R_outer": Ro, "R_inner": Ri, "iX": iX, "X_sites": iX, "Y_sites": n - iX}
        # 1, 2: the full block and the identities
        K = Pf[rx] @ Phi_o[ry].conj().T
        sK = np.linalg.svd(K, compute_uv=False)
        CX = Pf[rx] @ Phi_o[rx].conj().T
        CX = 0.5 * (CX + CX.conj().T)
        lx = np.linalg.eigvalsh(CX)
        idres = float(np.max(np.abs(np.linalg.eigvalsh(CX - CX @ CX - K @ K.conj().T))))
        S1 = float(s_ent(lx).sum()); S2 = float(s_ent(lam_of_sigma(sK)).sum())
        res["K"] = {"sigma_above_1e-12": [float(x) for x in sK if x > 1e-12], "n_above_1e-12": int(np.sum(sK > 1e-12)), "sigma_max": float(sK.max()) if len(sK) else 0.0,
                    "fro": float(np.linalg.norm(sK))}
        res["identity"] = {"CX_minus_CX2_minus_KKdag_op": idres, "S_X_trace": S1, "S_X_sigma": S2, "agreement": abs(S1 - S2), "label": ENT_LABEL,
                           "lambda_range": [float(lx.min()), float(lx.max())]}
        # the second X: the trapped annulus, its complement the second Y
        if iI is not None and iI < iX:
            rx2 = np.arange(2 * iI, 2 * iX); ry2 = np.concatenate([np.arange(0, 2 * iI), np.arange(2 * iX, 2 * n)])
            K2 = Pf[rx2] @ Phi_o[ry2].conj().T
            s2 = np.linalg.svd(K2, compute_uv=False)
            CX2 = Pf[rx2] @ Phi_o[rx2].conj().T; CX2 = 0.5 * (CX2 + CX2.conj().T)
            lx2 = np.linalg.eigvalsh(CX2)
            idres2 = float(np.max(np.abs(np.linalg.eigvalsh(CX2 - CX2 @ CX2 - K2 @ K2.conj().T))))
            K2o = Pf[rx2] @ Phi_o[ry].conj().T
            s2o = np.linalg.svd(K2o, compute_uv=False)
            res["annulus"] = {"sites": [int(iI) + 1, int(iX)], "n_sites": int(iX - iI),
                              "K_complement": {"n_above_1e-12": int(np.sum(s2 > 1e-12)), "sigma_max": float(s2.max()), "fro": float(np.linalg.norm(s2)),
                                               "sigma_above_1e-12": [float(x) for x in s2 if x > 1e-12]},
                              "K_to_outside": {"sigma_max": float(s2o.max()), "fro": float(np.linalg.norm(s2o))},
                              "identity": {"op": idres2, "S_X_trace": float(s_ent(lx2).sum()), "S_X_sigma": float(s_ent(lam_of_sigma(s2)).sum()),
                                           "agreement": abs(float(s_ent(lx2).sum()) - float(s_ent(lam_of_sigma(s2)).sum())), "label": ENT_LABEL}}
        else:
            res["annulus"] = None
        # 3: the slabs
        sl = {}
        for geo in ("P07", "centred"):
            for d in DS:
                try:
                    A, B = slab_sites(R, Ro, d, geo)
                except IndexError:
                    sl["%s_%d" % (geo, d)] = None; continue
                ra = rows_of(A); rb = rows_of(B)
                KC = Pf[ra] @ Phi_o[rb].conj().T
                KP = refc[ra] @ refc[rb].conj().T
                row = {"inside_sites_R": [float(R[i]) for i in A], "outside_sites_R": [float(R[i]) for i in B], "C": block_stats(KC), "P": block_stats(KP), "C_minus_P": block_stats(KC - KP)}
                if V is not None:
                    KV = V[ra] @ V[rb].conj().T
                    row["P_minus"] = block_stats(KV); row["C_minus_P_minus"] = block_stats(KC - KV)
                sl["%s_%d" % (geo, d)] = row
        res["slabs"] = sl
        # 4: the cut current through the outer root
        lo = max(2 * iX - 4, 0); hi = min(2 * iX + 4, 2 * n)
        hxy = H[lo:2 * iX, 2 * iX:hi].toarray()
        Cyx = Pf[2 * iX:hi] @ Phi_o[lo:2 * iX].conj().T
        Pyx = refc[2 * iX:hi] @ refc[lo:2 * iX].conj().T
        tC = complex(np.sum(hxy * Cyx.T)); tD = complex(np.sum(hxy * (Cyx - Pyx).T))
        res["cut_current"] = {"F_D": -2.0 * tD.imag, "F_C": -2.0 * tC.imag, "convention": "F = -2 Im Tr(h_XY D_YX) = dN_Y/dt: outward positive, inward negative"}
        # 5: the outward-branch content
        Ux, sx, Vh = np.linalg.svd(Phi_o[rx], full_matrices=False)
        lam = sx ** 2
        keep = lam >= 0.99
        loc = (Ux[:, keep] * sx[keep][None, :]) if keep.any() else np.zeros((2 * iX, 0), complex)
        nd = dust.shape[1]
        res["outward_branch"] = {"localized_count": int(keep.sum()), "localized_lambda_min": float(lam[keep].min()) if keep.any() else None,
                                 "localized": momentum_content(loc, iX), "occupied_all": momentum_content(Phi_o[rx], iX),
                                 "dust": momentum_content(Phi_o[rx][:, Phi.shape[1] - nd:], iX) if nd else None,
                                 "window": "rectangular (no taper) on sites 1 .. iX; orthonormal DFT per spinor component; |ak| >= pi/4 counted in f_>"}
        out["splits"][kind] = res
    return out


def record_A(Z, kappas, m, rec_info, splits, need_vacuum=True, log=print):
    R = np.asarray(Z["R"], float); N_app = np.asarray(Z["N_applied"], float); v_app = np.asarray(Z["v_applied"], float)
    chans = []
    for kap in kappas:
        t_ = time.time()
        ch = channel_A(R, m, kap, N_app, v_app, Z["sea_%d" % kap], Z["dust_%d" % kap], Z["f_%d" % kap], Z["R2_%d" % kap], splits, need_vacuum)
        ch["seconds"] = time.time() - t_
        chans.append(ch)
        sp_ = ch["splits"].get("dense") or {}
        k_ = sp_.get("K") or {}
        log("  kappa %d: stored ||C^2 - C|| %.1e (Loewdin %.1e); dense split: sigma_max(K) %s, ||K||_F %s, %s singular values above 1e-12; identity %s, S_X agreement %s; F_D %s, F_C %s; f_> localized %s, all %s, dust %s; %.1f s" % (
            kap, ch["stored_idempotency"], ch["loewdin_orthonormality"], fmt(k_.get("sigma_max")), fmt(k_.get("fro")), k_.get("n_above_1e-12"),
            fmt((sp_.get("identity") or {}).get("CX_minus_CX2_minus_KKdag_op"), "%.1e"), fmt((sp_.get("identity") or {}).get("agreement"), "%.1e"),
            fmt((sp_.get("cut_current") or {}).get("F_D"), "%+.4e"), fmt((sp_.get("cut_current") or {}).get("F_C"), "%+.4e"),
            fmt(((sp_.get("outward_branch") or {}).get("localized") or {}).get("f_gt"), "%.4f"), fmt(((sp_.get("outward_branch") or {}).get("occupied_all") or {}).get("f_gt"), "%.4f"),
            fmt(((sp_.get("outward_branch") or {}).get("dust") or {}).get("f_gt"), "%.4f"), ch["seconds"]))
    return chans


class Rs32Slice:
    """the r_s/a 32 record at T/r_s 18 (S7 part 3's T0) as a slice: engine_s7_twocopy_v2's stored T0 state (kappa 1 and 2: sea, dust, f, ref)
    and the member's applied fields at that record."""

    def __init__(self, states, Zf, k0, R):
        self.S = np.load(states); self.k0 = k0
        self.d = {"R": np.asarray(R, float), "N_applied": np.asarray(Zf["N_applied"][k0], float), "v_applied": np.asarray(Zf["v_applied"][k0], float),
                  "T": float(Zf["T"][k0])}

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        name, kap = key.rsplit("_", 1)
        m = {"sea": "sea", "dust": "dust", "f": "f", "R2": "ref"}[name]
        return self.S["T0_k%s_%s" % (kap, m)]


def vacuum_control(Z, m, kappa, E, g, rs, Ro, P_V_stored_fn, n_ext_rs=128.0):
    """the Killing vacuum with the vacuum exterior appended to n_ext_rs r_s (N = 1, v = -(2 g E/R)^(1/2) beyond the stored chain), its slab blocks
    against the stored chain's at the same sites."""
    R = np.asarray(Z["R"], float); n = len(R)
    n_ext = int(round(n_ext_rs * rs))
    Re = np.arange(n_ext, dtype=float) + (R[0])
    Ne = np.ones(n_ext); ve = -np.sqrt(2.0 * g * E / Re)
    Ne[:n] = np.asarray(Z["N_applied"], float); ve[:n] = np.asarray(Z["v_applied"], float)
    t_ = time.time()
    He = hm_host(Re, m, kappa, Ne, ve).toarray()
    e, U = np.linalg.eigh(He)
    Ve = U[:, e < 0]
    del U, He
    out = {"n_ext_sites": n_ext, "R_end": float(Re[-1]), "eig_seconds": time.time() - t_, "rows": []}
    for geo in ("P07", "centred"):
        for d in DS:
            A, B = slab_sites(R, Ro, d, geo)
            ra = rows_of(A); rb = rows_of(B)
            KVe = Ve[ra] @ Ve[rb].conj().T
            KVs = P_V_stored_fn(ra, rb)
            out["rows"].append({"slab": "%s_%d" % (geo, d), "appended": block_stats(KVe), "stored_chain": block_stats(KVs), "max_abs_diff": float(np.max(np.abs(KVe - KVs)))})
    return out


def run_family(a):
    t_all = time.time()
    rdir, rtag = os.path.split(a.run)
    J = json.load(open(os.path.join(rdir, "collapse_%s.json" % rtag))); Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
    ra = J["args"]; rs = float(ra["rs"]); g = float(ra["g"]); ell = float(ra["ell"]); m = float(ra["m"])
    FM = importlib.import_module(a.fields_module)
    recs = {round(float(r["T"]), 6): r for r in J["records"] if r.get("E") is not None}
    Tz = np.asarray(Zf["T"], float)
    out = {"script": "engine_single_object_A_v1.py", "family": a.family, "run": a.run, "fields_module": a.fields_module, "label": LABEL, "entropy_label": ENT_LABEL,
           "no_mutual_information": "D15: no mutual information is computed or printed", "records": [], "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    if a.family == "rs32_T18":
        k0 = int(np.argmin(np.abs(Tz / rs - a.rs32_T0)))
        items = [("T/rs %.3f (S7 part 3's T0; engine_s7_twocopy_v2's stored state)" % (Tz[k0] / rs), Rs32Slice(a.rs32_states, Zf, k0, np.asarray(Zf["R"], float)), [1, 2])]
    else:
        items = []
        for p in a.slices.split(","):
            Z = np.load(p)
            ks = [int(k) for k in Z["kappas"]] if a.kappas == "all" else [int(k) for k in a.kappas.split(",")]
            items.append((os.path.basename(p), Z, ks))
    for name, Z, ks in items:
        T = float(Z["T"]); k = int(np.argmin(np.abs(Tz - T)))
        rec = recs.get(round(T, 6), {})
        print("%s: T/rs %.3f, the record's E %s, stored roots %s" % (name, T / rs, fmt(rec.get("E"), "%.6f"), rec.get("trapping_zeros")), flush=True)
        _, _, _, info = FM.solve_constraints_ivp(np.asarray(Zf["eps_tot"][k], float), np.asarray(Zf["J_tot"][k], float), g, np.asarray(Zf["R"], float), ell,
                                                 collar_policy=ra.get("collar_policy", "flag"), support_rel=ra.get("support_rel", 1e-14))
        rd = roots_dense(info, ell) if not info.get("failed") else []
        rap = roots_applied(Z["R"], Z["N_applied"], Z["v_applied"])
        so = {"dense": outer_inner(rd), "applied": outer_inner(rap)}
        E = rec.get("E")
        row = {"name": name, "T_over_rs": T / rs, "E": E, "two_gE": (2.0 * g * float(E)) if E is not None else None, "roots_dense": rd, "roots_applied": rap,
               "stored_trapping_zeros": rec.get("trapping_zeros"), "outer_inner": {kk: list(v_) for kk, v_ in so.items()}, "solve_failed": info.get("failed")}
        print("  roots: dense %s; applied %s; outer/inner: dense %s, applied %s" % (
            ", ".join("%.4f (%+.4f)" % (q["R"], q["slope"]) for q in rd) or "none", ", ".join("%.4f (%+.4f)" % (q["R"], q["slope"]) for q in rap) or "none",
            so["dense"], so["applied"]), flush=True)
        if so["dense"][0] is None and so["applied"][0] is None:
            row["channels"] = None; row["note"] = "no root of N + v (dense and applied): no split; nothing computed on this record"
            out["records"].append(row); json.dump(out, open(a.out_json, "w"), indent=1, default=float)
            continue
        row["channels"] = record_A(Z, ks, m, rec, so, need_vacuum=True, log=lambda s: print(s, flush=True))
        if a.vacuum_control:
            tc, kc = [float(x) for x in a.vacuum_control.split(",")]
            if abs(T / rs - tc) < 1e-9 and so["dense"][0] is not None:
                Hs = hm_host(Z["R"], m, int(kc), Z["N_applied"], Z["v_applied"]).toarray()
                e, U = np.linalg.eigh(Hs); Vs = U[:, e < 0]; del U, Hs
                row["vacuum_control"] = vacuum_control(Z, m, int(kc), float(E), g, rs, so["dense"][0], lambda ra_, rb_: Vs[ra_] @ Vs[rb_].conj().T)
                print("  the vacuum control (kappa %d, the exterior appended to %.0f sites): %s" % (int(kc), row["vacuum_control"]["n_ext_sites"],
                      "; ".join("%s %.1e" % (q["slab"], q["max_abs_diff"]) for q in row["vacuum_control"]["rows"])), flush=True)
                del Vs
        out["records"].append(row)
        json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    out["seconds"] = time.time() - t_all
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    print("done: %d records in %.0f s" % (len(out["records"]), out["seconds"]), flush=True)


def compose(a):
    fams = [json.load(open(p)) for p in a.compose.split(",")]
    L = ["## S4 (revised) part A: the single object's correlations across the trapping root, per record and radial channel (engine_single_object_A_v1.py; inbox #41; post-processing of stored states, no propagation)", "",
         "- %s." % LABEL, "- no mutual information is computed or printed (D15); S_X appears only as the internal check of item 2, labelled \"%s\"." % ENT_LABEL,
         "- no 16 kappa weight on any number below (per radial channel, the g_n guard).",
         "- the stored columns are complex64: ||C^2 - C||_op of the stored state is printed as stored; every other readout of C uses the columns re-orthonormalized in float64 (Loewdin), the continued reference P as stored.",
         "- the Killing vacuum P_-: the negative-energy projector of h_m at the record's applied fields on the stored chain (to its end, where the engine's exterior fields already sit); the exterior appended to 128 r_s is the labelled control where it ran.", ""]
    for F in fams:
        L += ["### Family %s (%s; field module %s; %s to %s)" % (F["family"], F["run"], F["fields_module"], F.get("utc_start"), F.get("utc_end")), ""]
        for r in F["records"]:
            L.append("- %s: T/rs %.3f; E %s, 2GE %s; the dense roots of N + v (slope): %s; the applied-field roots: %s; the stored zeros %s; outer, inner: dense %s, applied %s%s." % (
                r["name"], r["T_over_rs"], fmt(r["E"], "%.6f"), fmt(r["two_gE"], "%.4f"), ", ".join("%.4f (%+.4f)" % (q["R"], q["slope"]) for q in r["roots_dense"]) or "none",
                ", ".join("%.4f (%+.4f)" % (q["R"], q["slope"]) for q in r["roots_applied"]) or "none", r["stored_trapping_zeros"],
                r["outer_inner"]["dense"], r["outer_inner"]["applied"], ("; " + r["note"]) if r.get("note") else ""))
            if r.get("vacuum_control"):
                vc = r["vacuum_control"]
                L.append("  The vacuum control (the exterior appended to %d sites, R %.0f; %.0f s): the slab blocks of P_- against the stored chain's, max abs. difference %s." % (
                    vc["n_ext_sites"], vc["R_end"], vc["eig_seconds"], "; ".join("%s %.1e" % (q["slab"], q["max_abs_diff"]) for q in vc["rows"])))
        L.append("")
        for kind in ("dense", "applied"):
            L += ["#### %s, the %s root: items 1, 2 and 4 per record and channel" % (F["family"], kind), "",
                  "| record | kappa | R_outer (X sites) | n sigma > 1e-12 | sigma_max(K) | norm_F(K) | the top singular values | stored C^2 - C (op) | C_X - C_X^2 - K K^dagger (op) | S_X: trace; sigma route; agreement (%s) | annulus: sites; n sigma; sigma_max; norm_F; to the outside sigma_max | F (D = C - P) | F (C) |" % ENT_LABEL,
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in F["records"]:
                for ch in (r.get("channels") or []):
                    s_ = ch["splits"].get(kind) or {}
                    if s_.get("root", "x") is None:
                        L.append("| %.3f | %d | no root | - | - | - | - | %.1e | - | - | - | - | - |" % (r["T_over_rs"], ch["kappa"], ch["stored_idempotency"])); continue
                    if s_.get("same_split_as"):
                        L.append("| %.3f | %d | %.4f (%d): the same site split as the %s root | | | | | %.1e | | | | | |" % (r["T_over_rs"], ch["kappa"], s_["R_outer"], s_["iX"], s_["same_split_as"], ch["stored_idempotency"])); continue
                    K = s_["K"]; idt = s_["identity"]; an = s_.get("annulus"); cc = s_["cut_current"]
                    ann = ("%d to %d; %d; %.3e; %.3e; %.3e" % (an["sites"][0], an["sites"][1], an["K_complement"]["n_above_1e-12"], an["K_complement"]["sigma_max"], an["K_complement"]["fro"],
                                                            an["K_to_outside"]["sigma_max"])) if an else "-"
                    L.append("| %.3f | %d | %.4f (%d) | %d | %.6e | %.6e | %s | %.1e | %.1e | %.6f; %.6f; %.1e | %s | %+.6e | %+.6e |" % (
                        r["T_over_rs"], ch["kappa"], s_["R_outer"], s_["iX"], K["n_above_1e-12"], K["sigma_max"], K["fro"], " ".join("%.3e" % x for x in K["sigma_above_1e-12"][:8]),
                        ch["stored_idempotency"], idt["CX_minus_CX2_minus_KKdag_op"], idt["S_X_trace"], idt["S_X_sigma"], idt["agreement"], ann, cc["F_D"], cc["F_C"]))
            L.append("")
            for geo in ("P07", "centred"):
                L += ["#### %s, the %s root: item 3, the 4-site slabs (%s), sigma_max; norm_F of the cross block" % (F["family"], kind,
                      "P07 sec. 10: the nearest sites at least d from the root" if geo == "P07" else "the task's wording: centred d sites from the root"), "",
                      "| record | kappa | d | inside sites R | outside sites R | C | P_- (Killing vacuum) | P (continued reference) | C - P_- | C - P |", "|---|---|---|---|---|---|---|---|---|---|"]
                for r in F["records"]:
                    for ch in (r.get("channels") or []):
                        s_ = ch["splits"].get(kind) or {}
                        if not s_.get("slabs"):
                            continue
                        for d in DS:
                            q = s_["slabs"].get("%s_%d" % (geo, d))
                            if not q:
                                continue
                            bs = lambda b: "%.4e; %.4e" % (b["sigma_max"], b["fro"]) if b else "-"
                            L.append("| %.3f | %d | %d | %s | %s | %s | %s | %s | %s | %s |" % (
                                r["T_over_rs"], ch["kappa"], d, ",".join("%.0f" % x for x in q["inside_sites_R"]), ",".join("%.0f" % x for x in q["outside_sites_R"]),
                                bs(q["C"]), bs(q.get("P_minus")), bs(q["P"]), bs(q.get("C_minus_P_minus")), bs(q["C_minus_P"])))
                L.append("")
            L += ["#### %s, the %s root: item 5, the outward-branch content (f_> = the weight at |ak| >= pi/4 of the interior components; the quarter-zone shares [0, pi/4), [pi/4, pi/2), [pi/2, 3pi/4], (3pi/4, pi])" % (F["family"], kind), "",
                  "| record | kappa | localized orbitals (eigenvalue >= 0.99; smallest) | f_> localized | quarters localized | f_> all occupied (comparator) | quarters all | f_> dust (extra) | quarters dust |", "|---|---|---|---|---|---|---|---|---|"]
            for r in F["records"]:
                for ch in (r.get("channels") or []):
                    s_ = ch["splits"].get(kind) or {}
                    ob = s_.get("outward_branch")
                    if not ob:
                        continue
                    qs_ = lambda q: ", ".join("%.4f" % x for x in q["quarters"]) if q else "-"
                    L.append("| %.3f | %d | %d (%s) | %s | %s | %s | %s | %s | %s |" % (
                        r["T_over_rs"], ch["kappa"], ob["localized_count"], fmt(ob["localized_lambda_min"], "%.6f"), fmt((ob["localized"] or {}).get("f_gt"), "%.4f"), qs_(ob["localized"]),
                        fmt((ob["occupied_all"] or {}).get("f_gt"), "%.4f"), qs_(ob["occupied_all"]), fmt((ob.get("dust") or {}).get("f_gt"), "%.4f"), qs_(ob.get("dust"))))
            L.append("")
    worst = max((ch["splits"][k_]["identity"]["agreement"] for F in fams for r in F["records"] for ch in (r.get("channels") or [])
                 for k_ in ch["splits"] if ch["splits"][k_].get("identity")), default=None)
    worst_id = max((ch["splits"][k_]["identity"]["CX_minus_CX2_minus_KKdag_op"] for F in fams for r in F["records"] for ch in (r.get("channels") or [])
                    for k_ in ch["splits"] if ch["splits"][k_].get("identity")), default=None)
    L += ["- item 2 over every record, channel and split: the largest S_X agreement %s and the largest ||C_X - C_X^2 - K K^dagger||_op %s (the target 1e-10); every number above is printed by the script." % (
        fmt(worst, "%.1e"), fmt(worst_id, "%.1e"))]
    open(a.md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump({"script": "engine_single_object_A_v1.py --compose", "families": [F["family"] for F in fams], "worst_S_agreement": worst, "worst_identity": worst_id, "label": LABEL},
                  open(a.out_json, "w"), indent=1, default=float)
    print("composed %s" % a.md, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, choices=["rs64_sigma4", "rs64_sigma8", "rs32_T18"])
    ap.add_argument("--slices", default=None)
    ap.add_argument("--run", default=None, help="the run's dir/tag (its json and fields npz)")
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--rs32-states", default=None, help="engine_s7_twocopy_v2's states npz (the T0 state)")
    ap.add_argument("--rs32-T0", type=float, default=18.0)
    ap.add_argument("--kappas", default="all")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--vacuum-control", default=None, help="T_over_rs,kappa: the exterior appended to 128 r_s on that record and channel")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--compose", default=None)
    a = ap.parse_args()
    if a.compose:
        return compose(a)
    run_family(a)


if __name__ == "__main__":
    main()
