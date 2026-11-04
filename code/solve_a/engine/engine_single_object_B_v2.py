#!/usr/bin/env python
"""engine_single_object_B_v2.py (builder analyst, the HORIZON version 2 builder of the lead's turn 47; engine spec 38.4 (1) (i) to (iv) and
39.5 (1) (v) to (x); brief lead/BRIEF_horizon_v2_builder.md). Version 2 of the revised S4 part B: engine_single_object_B_v1.py's machinery
(imported, never edited: its launch spinor, Killing spectrum, region weights, cuts, dense roots, exterior shells, half-zone split; through it
page_history_v6b's History and Chebyshev action and page_ops_v6's operators) with these changes, each a ruled item:
(i) per fixed-field record a comparator launch at R_t + 2 sigma OUTSIDE the outer root with each k_0 and envelope (both signs of k_0), beside
    v1's two interior depths (columns 0 to 11 are v1's columns, built by the same code; 12 to 17 the comparators); the arrival is v1's (the
    first clock at which the weight beyond 16 r_s reaches half its value at the last clock before the chain-end reflection), resolved here by
    steps of --step-rs (0.25 r_s) with the half-crossing interpolated linearly between steps (v1: whole steps of 1 r_s); the delay is printed
    as the arrival minus the arrival of the comparator with the same k_0 and envelope, beside the two forms 2 r_s ln(1/(a omega_bar)) and
    kappa_in^-1 ln(1/(ak_0)); the comparators also run on the solved history (labelled; the history's arrivals are run-limited);
(iii) "fixed-field" in every label and key (mode fixed_field; v1's word is not used);
(vi) the signed projection on the positive intrinsic Dirac branch with the local group-velocity sign, per column (at launch on the whole chain,
    and inside and beyond R_t(t) every --signed-every-rs): the column's region split into sine windows of --win sites (32) with half overlap
    (sum of the squared windows 1 on the region: the weights add to the region's norm), each window's orthonormal DFT zero-padded to 4 --win;
    per window centre c the spinors u_+-(q) of D(q) = m sigma_z + sin q sigma_x + (kappa/R_c) sigma_y (the positive and negative intrinsic
    branch; the flow term (v/2) sin 2q is proportional to the identity, so h's local eigenvectors are D's) and the local group velocities
    v_+-(q) = +-N_c sin q cos q / E(q) + v_c cos 2q (N_c, v_c the applied fields at the window centre, v <= 0 the inflow); per column the
    weights on (+, v_+ > 0), (+, v_+ <= 0), (-, v_- > 0), (-, v_- <= 0), the + branch on the signed interval pi/4 < q < pi/2 (outward at every
    flow speed for N > 0, v <= 0: inbox #51 and hreduce SO4), the - branch on -pi/2 < q < -pi/4 (its outward counterpart), and the unsigned
    windowed share pi/4 <= |q| < pi/2 (kept, labelled unsigned); controls at --win-controls (16, 64) at launch;
(viii) the finite-box localization classifier (the eigenvectors of the held h with weight >= 0.99 inside R_cut) labelled as such, and each
    column's Killing spectral weight above m (and below -m, and in the gap) printed beside its mean energy (v1 computed them; printed now);
(ii) and (vii) mode dust: the dust-tagged occupied columns (the sector with occupation f) and, as the comparator, the sea-localized orbitals,
    from engine_single_object_B5_prep_v2.py's npz, propagated on the solved history: per record the weight beyond R_t(t) and beyond 2GE(t)
    (occupation-weighted sums and the largest single column), and the readouts of (ix) for each set: the fixed-cut number current
    Tr[X^dag i[h, Pi_{R > R_t}] X] at the record, the moving-cut reassignment over the record step (the weight beyond R_t(t_{k+1}) minus
    the weight beyond R_t(t_k), both of the propagated state at t_{k+1}), the fixed-cut change over the step (at R_t(t_k): the state at
    t_{k+1} minus the state at t_k), the Killing-energy flux Tr[X^dag (i/2)[h^2, Pi_{R > R_t}] X]; printed as separate columns, never summed;
    these are the selected sets' readouts (the full detector density and the reference term are engine_single_object_slices_v1.py's, from the
    stored states); the signed branch readout of each set inside and beyond R_t every --signed-every-rs;
(x) no word for permanent departure is printed: "transfer beyond the moving root within the run".
The solved history may continue across two record files (--record2/--fields2: the continuation of #36 B, its records after --switch-T-rs
used; the production's at and before it), so the history runs read B's continuation as its records land; --resume-npz/--resume-json continue
a history or dust run from its saved end (the saved columns and the series so far), so later records extend it without re-running the start.
Modes: fixed_field (B.1 (i) with the comparators), history (B.1 (ii) with the comparators, the cubic control on --cubic-window-rs), dust (B.5
version 2). Outputs: --out-json (the per-column and per-set results) and --out-npz (the series arrays and, for history and dust, the end
columns). No mutual information (D15); the massive channel (ma 0.2) only (D14). LABEL on every output (v1's). Every number is printed by the
script."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine_single_object_B_v1 as B1   # noqa: E402  (never edited; its import puts page_*_v6 on the path)

OPS = B1.OPS
PH = B1.PH
CD = OPS.CD
RD = OPS.RD
LABEL = B1.LABEL
KS = B1.KS
DEPTHS = ("annulus middle", "2 sigma inside the outer root", "2 sigma outside the outer root (comparator)")
CLASSIFIER = ("the finite-box localization classifier: the eigenvectors of the held h whose weight inside R_cut is at least 0.99; not the "
              "infinite-exterior bound-state projector; a mean energy below m does not exclude support above the threshold (the weight above m "
              "is printed beside it)")
SIGNED = ("the signed projection on the positive intrinsic Dirac branch with the local group-velocity sign: sine windows of %d sites, half "
          "overlap, the DFT zero-padded to 4 window lengths; u_+-(q) of m sigma_z + sin q sigma_x + (kappa/R_c) sigma_y at each window centre; "
          "v_+-(q) = +-N_c sin q cos q / E(q) + v_c cos 2q with the applied fields at the centre; the interval pi/4 < q < pi/2 on the + branch is "
          "outward at every flow speed for N > 0, v <= 0; the unsigned share pi/4 <= |q| < pi/2 is kept as an unsigned windowed share")
WORDING = "transfer beyond the moving root within the run"


def log(msg):
    print(msg, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------- the solved history over two files
class CombinedHistory:
    """page_history_v6b.History over the production's records at and before --switch-T-rs and the continuation's after it (both in the
    engine's layout on the same sites, g, r_s and tau); the same interface B v1 uses (T, E, rec, fields(k), cubic_fields(t), R, g, rs_nom,
    ell, ne, nsites). With no second file it is the first History itself."""

    def __init__(self, rec1, fld1, nsites, rec2=None, fld2=None, switch_T_rs=None):
        self.H1 = PH.History(rec1, fld1, nsites)
        self.files = [{"record": rec1, "fields": fld1}]
        H1 = self.H1
        self.g = H1.g; self.rs_nom = H1.rs_nom; self.R = H1.R; self.ell = H1.ell; self.ne = H1.ne; self.nsites = H1.nsites; self.tau = H1.tau
        if rec2 is None:
            self.src = [(H1, k) for k in range(len(H1.T))]
        else:
            H2 = PH.History(rec2, fld2, nsites)
            if abs(H2.g - H1.g) > 0 or abs((H2.rs_nom or 0) - (H1.rs_nom or 0)) > 0 or H2.ne != H1.ne or abs(H2.tau - H1.tau) > 0:
                raise ValueError("the two record files differ in g, r_s, sites or tau")
            Ts = float(switch_T_rs) * H1.rs_nom
            a_ = [(H1, k) for k in range(len(H1.T)) if H1.T[k] <= Ts + 1e-9]
            b_ = [(H2, k) for k in range(len(H2.T)) if H2.T[k] > Ts + 1e-9]
            self.src = a_ + b_
            self.files.append({"record": rec2, "fields": fld2, "switch_T_over_rs": float(switch_T_rs)})
            self.H2 = H2
        self.T = np.array([H.T[k] for H, k in self.src], float)
        self.E = np.array([H.E[k] for H, k in self.src], float)
        self.rec = [H.rec[k] for H, k in self.src]
        if np.any(np.diff(self.T) <= 0):
            raise ValueError("the combined record clocks are not increasing")

    def __len__(self):
        return len(self.T)

    def fields(self, k):
        H, kk = self.src[k]
        return H.fields(kk)

    def fields_npz_row(self, k):
        """(the fields npz, the row) of record k, for the dense roots."""
        H, kk = self.src[k]
        return (self.files[0]["fields"] if H is self.H1 else self.files[1]["fields"]), kk

    def cubic_fields(self, t):
        k = int(np.clip(np.searchsorted(self.T, t, side="right") - 1, 0, len(self.T) - 2))
        u = (t - self.T[k]) / (self.T[k + 1] - self.T[k])
        ks = [max(k - 1, 0), k, k + 1, min(k + 2, len(self.T) - 1)]
        F = [self.fields(q) for q in ks]
        w = [-0.5 * u ** 3 + u ** 2 - 0.5 * u, 1.5 * u ** 3 - 2.5 * u ** 2 + 1.0, -1.5 * u ** 3 + 2.0 * u ** 2 + 0.5 * u, 0.5 * u ** 3 - 0.5 * u ** 2]
        return sum(wi * Fi[0] for wi, Fi in zip(w, F)), sum(wi * Fi[1] for wi, Fi in zip(w, F))


# --------------------------------------------------------------------------------------------------- the launch set with the comparator
def launch_columns_v2(R, m, kappa, roots, N, v):
    """v1's launch set (the same code for its two depths, so columns 0 to 11 are v1's) and the comparator depth R_t + 2 sigma (12 to 17)."""
    Ro, Ri = roots
    cols, info = [], []
    for depth_name in DEPTHS:
        for k0, sig, kname in KS:
            if depth_name == "annulus middle":
                Rc = 0.5 * (Ri + Ro)
            elif depth_name == "2 sigma inside the outer root":
                Rc = Ro - 2.0 * sig
            else:
                Rc = Ro + 2.0 * sig
            for sgn in (1, -1):
                g = np.exp(-0.5 * ((R - Rc) / sig) ** 2) * np.exp(1j * sgn * k0 * R)
                ic = int(np.argmin(np.abs(R - Rc)))
                u = B1.particle_spinor(m, sgn * k0, kappa, max(Rc, 1.0))
                x = np.zeros(2 * len(R), complex)
                x[0::2] = g * u[0]; x[1::2] = g * u[1]
                x /= np.linalg.norm(x)
                cols.append(x)
                info.append({"depth": depth_name, "comparator": depth_name.startswith("2 sigma outside"), "R_c": float(Rc), "k0": float(sgn * k0), "ak0_name": kname,
                             "sigma": sig, "spinor": [complex(u[0]).real, complex(u[0]).imag, complex(u[1]).real, complex(u[1]).imag], "N_at": float(N[ic]), "v_at": float(v[ic])})
    return np.array(cols).T, info


# --------------------------------------------------------------------------------------------------- the signed branch readout
def signed_branch(X, R, Nf, vf, kappa, m, lo, hi, L=32, pad=4):
    """the signed projection of the docstring's (vi) for the part of the columns X (2n x c, torch) on the sites lo < R <= hi; returns a dict
    of per-column lists (or None when the region is empty)."""
    R = np.asarray(R, float); n = len(R)
    sel = np.nonzero((R > lo) & (R <= hi))[0]
    if len(sel) == 0 or X.shape[1] == 0:
        return None
    i0, i1 = int(sel[0]), int(sel[-1]) + 1
    hL = L // 2
    starts = np.arange(i0 - hL, i1, hL)
    idx = starts[:, None] + np.arange(L)[None, :]                     # W x L site indices
    valid = (idx >= i0) & (idx < i1)
    idc = np.clip(idx, 0, n - 1)
    dev = X.device
    Xs = X.reshape(n, 2, -1)
    seg = Xs[torch.as_tensor(idc, device=dev)]                        # W x L x 2 x c
    gw = np.sin(math.pi * (np.arange(L) + 0.5) / L)
    wt = torch.as_tensor(gw[None, :] * valid, dtype=RD, device=dev)
    seg = seg * wt[:, :, None, None]
    M = pad * L
    F = torch.fft.fft(seg, n=M, dim=1, norm="ortho")                  # W x M x 2 x c
    q = 2.0 * math.pi * np.fft.fftfreq(M)                             # [-pi, pi)
    cen = np.clip(starts + hL, 0, n - 1)
    Nc = np.asarray(Nf, float)[cen]; vc = np.asarray(vf, float)[cen]; Rc = R[cen]
    s_ = np.sin(q)[None, :]; c_ = np.cos(q)[None, :]
    kr = (kappa / Rc)[:, None]
    E = np.sqrt(m * m + s_ ** 2 + kr ** 2)                            # W x M
    nrm = np.sqrt(2.0 * E * (E + m))
    up0 = (m + E) / nrm; up1 = (s_ + 1j * kr) / nrm                   # u_+ = (m + E, sin q + i kappa/R)/norm
    a_p = (torch.as_tensor(np.conj(up0), dtype=CD, device=dev)[:, :, None] * F[:, :, 0, :]
           + torch.as_tensor(np.conj(up1), dtype=CD, device=dev)[:, :, None] * F[:, :, 1, :])
    w_tot = (F.abs() ** 2).sum(dim=2)                                 # W x M x c
    w_p = a_p.abs() ** 2
    w_m = w_tot - w_p
    vp = Nc[:, None] * s_ * c_ / E + vc[:, None] * np.cos(2.0 * q)[None, :]
    vm = -Nc[:, None] * s_ * c_ / E + vc[:, None] * np.cos(2.0 * q)[None, :]
    T_ = lambda a: torch.as_tensor(a.astype(float), dtype=RD, device=dev)[:, :, None]
    out_p = T_(vp > 0); out_m = T_(vm > 0)
    gp = T_(np.broadcast_to((q > math.pi / 4) & (q < math.pi / 2), vp.shape))
    gm = T_(np.broadcast_to((q > -math.pi / 2) & (q < -math.pi / 4), vp.shape))
    un = T_(np.broadcast_to((np.abs(q) >= math.pi / 4) & (np.abs(q) < math.pi / 2), vp.shape))
    S = lambda a: a.sum(dim=(0, 1)).cpu().numpy().tolist()
    return {"window": L, "pad": pad, "region": [float(lo), float(hi)], "sites": [i0 + 1, i1], "windows": int(len(starts)),
            "N_nonpositive_windows": int(np.sum(Nc <= 0)), "v_positive_windows": int(np.sum(vc > 0)),
            "total": S(w_tot), "plus": S(w_p), "minus": S(w_m), "plus_out": S(w_p * out_p), "plus_in": S(w_p * (1.0 - out_p)),
            "minus_out": S(w_m * out_m), "minus_in": S(w_m * (1.0 - out_m)), "plus_pi4_pi2": S(w_p * gp), "minus_negpi2_negpi4": S(w_m * gm),
            "unsigned_pi4_pi2": S(w_tot * un)}


def signed_inside_beyond(X, R, Nf, vf, kappa, m, Rt, L):
    return {"inside_Rt": signed_branch(X, R, Nf, vf, kappa, m, -1.0, Rt, L) if Rt else None,
            "beyond_Rt": signed_branch(X, R, Nf, vf, kappa, m, Rt, 1e30, L) if Rt else None}


# --------------------------------------------------------------------------------------------------- helpers
def arrival_interp(ts, ws, final):
    """the first clock at which ws reaches final/2, linearly interpolated between the samples (ts ascending); None when it never does."""
    if final is None or not (final > 1e-12):
        return None
    half = 0.5 * final
    for j in range(len(ws)):
        if ws[j] >= half:
            if j == 0:
                return float(ts[0])
            w0, w1 = ws[j - 1], ws[j]
            return float(ts[j - 1] + (ts[j] - ts[j - 1]) * (half - w0) / (w1 - w0)) if w1 > w0 else float(ts[j])
    return None


def roots_at(H, k, fields_module):
    fp, kk = H.fields_npz_row(k)
    Zf = np.load(fp, allow_pickle=True)
    return B1.dense_roots(_HRow(H), _ZRow(Zf, kk), 0, fields_module, H.ell)


class _HRow:
    def __init__(self, H):
        self.g = H.g; self.Re = H.H1.Re


class _ZRow:
    def __init__(self, Zf, kk):
        self.d = {"eps_tot": [np.asarray(Zf["eps_tot"][kk], float)], "J_tot": [np.asarray(Zf["J_tot"][kk], float)]}

    def __getitem__(self, key):
        return self.d[key]


def outer_inner_of(rd, rec):
    pos = [q for q in rd if q["slope"] > 0]; Ro = max(q["R"] for q in pos) if pos else rec.get("R_t")
    neg = [q for q in rd if q["slope"] < 0 and q["R"] < (Ro or 0)]; Ri = max(q["R"] for q in neg) if neg else rec.get("r_in")
    kin = abs([q for q in rd if q["R"] == Ri][0]["slope"]) if (neg and Ri is not None) else None
    return Ro, Ri, kin


# --------------------------------------------------------------------------------------------------- fixed-field and history
def fixed_or_history(a, H, dev, mode):
    rs = H.rs_nom; g = H.g; m = a.m; R = H.R; n = len(R)
    out = {"script": "engine_single_object_B_v2.py", "mode": mode, "label": LABEL, "classifier": CLASSIFIER, "signed": SIGNED % a.win, "wording": WORDING,
           "no_mutual_information": "D15", "massless": "not run (owner decision D14)", "files": H.files, "nsites": n, "m": m, "cheb_tol": a.cheb_tol,
           "R_cut_rs": a.R_cut_rs, "step_rs": a.step_rs if mode == "fixed_field" else None, "win": a.win, "launches": [],
           "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    arrays = {}
    prev = json.load(open(a.resume_json)) if a.resume_json else None
    Rz = np.load(a.resume_npz) if a.resume_npz else None
    TT = np.asarray(H.T, float) / rs
    for T_rs in [float(x) for x in a.launch_T.split(",")]:
        k0 = int(np.argmin(np.abs(TT - T_rs)))
        rec = H.rec[k0]
        N0, v0 = H.fields(k0); E0 = float(H.E[k0]); rs0 = 2.0 * g * E0; TH0 = 1.0 / (4.0 * math.pi * rs0)
        rd = roots_at(H, k0, a.fields_module)
        Ro, Ri, kin = outer_inner_of(rd, rec)
        log("launch T/rs %.3f (k %d): E %.6f, 2gE %.4f; the dense roots %s (stored %s); outer %s, inner %s, kappa_in %s" % (
            TT[k0], k0, E0, rs0, ", ".join("%.4f (%+.5f)" % (q["R"], q["slope"]) for q in rd), rec.get("trapping_zeros"), B1.fmt(Ro, "%.4f"),
            B1.fmt(Ri, "%.4f"), B1.fmt(kin, "%.5f")))
        for kap in [int(x) for x in a.kappas.split(",")]:
            t_c = time.time()
            key = ("T%.3f_k%d" % (TT[k0], kap)).replace(".", "p")
            h0, Q = OPS.chain_sparse(R, m, float(kap))
            h = OPS.clocked_csr(h0, Q, N0, v0)
            e, U = OPS.eigh_csr(h, dev)
            i_cut = int(np.searchsorted(R, a.R_cut_rs * rs, side="right"))
            wi = (U[: 2 * i_cut].abs() ** 2).sum(dim=0)
            bidx = torch.nonzero(wi >= 0.99).ravel()
            Ub = U[:, bidx]
            X0np, info = launch_columns_v2(R, m, kap, (Ro, Ri), N0, v0)
            X0 = torch.as_tensor(X0np, dtype=CD, device=dev)
            spec0, W0 = B1.spectrum(e, U, X0, m, TH0, a.bin_TH, a.bin_max_TH)
            bo_launch = (Ub.mH @ X0).abs().pow(2).sum(dim=0).cpu().numpy() if Ub.shape[1] else np.zeros(X0.shape[1])
            sg_launch = {"whole": signed_branch(X0, R, N0, v0, kap, m, -1.0, 1e30, a.win), **signed_inside_beyond(X0, R, N0, v0, kap, m, Ro, a.win)}
            sg_ctrl = {str(Lc): signed_branch(X0, R, N0, v0, kap, m, -1.0, 1e30, Lc) for Lc in [int(x) for x in a.win_controls.split(",") if x.strip()]}
            eig_s = time.time() - t_c
            ext = PH.exterior_shells(B1._R(R), h, a.R_cut_rs * rs, TH0, a.bin_TH, a.bin_max_TH, dev) if mode == "fixed_field" else None
            Rl = np.array([q["R_c"] for q in info])
            t_refl = (128.0 * rs - Rl) + (128.0 - a.R_cut_rs) * rs
            T0 = float(H.T[k0])
            X = X0.clone()
            ser_t, ser_w, ser_norm, ser_Rt, ser_2gE, ser_refl = [], {nm: [] for nm in ("beyond_outer_root", "beyond_2GE", "beyond_16rs", "collar_16_24", "far_32_48", "beyond_Rcut")}, [], [], [], []
            signed_rows = []; ext_rows = []; hz_rows = []
            maxR = np.zeros(X.shape[1]); Kmax = 0; tailacc = 0.0
            cubic = None
            if mode == "fixed_field":
                nst = int(round(a.fixed_rs / a.step_rs)); dt = a.step_rs * rs
                clocks = [T0 + (j + 1) * dt for j in range(nst)]
                Bh = OPS.Banded(h, dev)
                steps = [("fixed", None)] * nst
            else:
                kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
                kstart = k0
                if prev is not None:
                    pl = [L_ for L_ in prev["launches"] if L_["key"] == key]
                    if pl:
                        kstart = int(pl[0]["k_end"]); X = torch.as_tensor(Rz["X_end_" + key], dtype=CD, device=dev)
                        log("  %s: resumed at record %d (T/rs %.3f) from %s" % (key, kstart, TT[kstart], a.resume_npz))
                clocks = [float(H.T[kk + 1]) for kk in range(kstart, kend)]
                steps = [("record", kk) for kk in range(kstart, kend)]
            Xc = X.clone() if (mode == "history" and a.cubic_window_rs and prev is None and T_rs == float(a.launch_T.split(",")[0])) else None
            every = max(1, int(round(a.signed_every_rs * rs / (a.step_rs * rs if mode == "fixed_field" else H.tau))))
            for j, (kind, kk) in enumerate(steps):
                tc = clocks[j]
                if kind == "fixed":
                    X, K, tail = PH.cheb_apply(Bh, X, dt, a.cheb_tol)
                    kk_now = k0; Nn, vn = N0, v0
                else:
                    Nk, vk = H.fields(kk)
                    X, K, tail = PH.cheb_apply(OPS.Banded(OPS.clocked_csr(h0, Q, Nk, vk), dev), X, float(H.T[kk + 1] - H.T[kk]), a.cheb_tol)
                    kk_now = kk + 1; Nn, vn = H.fields(kk_now)
                    if Xc is not None and float(H.T[kk + 1]) <= T0 + a.cubic_window_rs * rs + 1e-9:
                        t_s = float(H.T[kk]); t_e = float(H.T[kk + 1]); nsub = int(round(t_e - t_s))
                        for s_ in range(nsub):
                            Nc_, vc_ = H.cubic_fields(t_s + s_ + 0.5)
                            Xc, _, _ = PH.cheb_apply(OPS.Banded(OPS.clocked_csr(h0, Q, Nc_, vc_), dev), Xc, 1.0, a.cheb_tol)
                        if float(H.T[kk + 1]) >= T0 + a.cubic_window_rs * rs - 1e-9:
                            rc_, _ = B1.region_weights(Xc, R, B1.cuts_of(rs, Ro, rs0, a.R_cut_rs)); rp_, _ = B1.region_weights(X, R, B1.cuts_of(rs, Ro, rs0, a.R_cut_rs))
                            cubic = {"window_rs": [T0 / rs, float(H.T[kk + 1]) / rs], "columns_diff_max": float((Xc - X).abs().max()),
                                     "region_diff": {k_: float(np.max(np.abs(rc_[k_] - rp_[k_]))) for k_ in rp_}}
                            Xc = None
                Kmax = max(Kmax, K); tailacc += tail
                recn = H.rec[kk_now]
                Rt_now = Ro if kind == "fixed" else recn.get("R_t")
                rs_now = rs0 if kind == "fixed" else 2.0 * g * float(H.E[kk_now])
                cw, tail_w = B1.region_weights(X, R, B1.cuts_of(rs, Rt_now, rs0 if kind == "fixed" else rs_now, a.R_cut_rs))
                for c_ in range(X.shape[1]):
                    ix = np.nonzero(tail_w[:, c_] > 1e-3)[0]
                    if len(ix):
                        maxR[c_] = max(maxR[c_], float(R[ix[-1]]))
                norms = (X.abs() ** 2).sum(dim=0).cpu().numpy()
                ser_t.append(tc); ser_norm.append(float(np.max(np.abs(norms - 1.0)))); ser_Rt.append(Rt_now if Rt_now else float("nan")); ser_2gE.append(rs_now)
                for nm in ser_w:
                    ser_w[nm].append(cw[nm] if nm in cw else np.full(X.shape[1], np.nan))
                ser_refl.append([bool(tc - T0 >= t_refl[c_] or tc - T0 >= 208.0 * rs) for c_ in range(X.shape[1])])
                if (j + 1) % every == 0 or j == len(steps) - 1:
                    i_Rt = int(np.searchsorted(R, Rt_now, side="right")) if Rt_now else None
                    signed_rows.append({"t_over_rs": tc / rs, "after_launch_rs": (tc - T0) / rs, "R_t": Rt_now, **signed_inside_beyond(X, R, Nn, vn, kap, m, Rt_now, a.win)})
                    hz_rows.append({"t_over_rs": tc / rs, "halfzone_beyond_Rt": B1.halfzone_beyond(X, i_Rt) if i_Rt is not None else None})
                if kind == "fixed" and (abs(((tc - T0) / rs) % a.ext_every_rs) < 1e-9 or abs(((tc - T0) / rs) % a.ext_every_rs - a.ext_every_rs) < 1e-9 or j == len(steps) - 1):
                    ext_rows.append({"after_launch_rs": (tc - T0) / rs, "shells": B1.ext_spectrum(ext, X)})
            # merge with the resumed series
            t_all = np.array(ser_t); W_all = {nm: np.array(v_) for nm, v_ in ser_w.items()}; refl_all = np.array(ser_refl, bool)
            if prev is not None and ("t_" + key) in Rz.files:
                t_all = np.concatenate([Rz["t_" + key], t_all]); refl_all = np.concatenate([Rz["refl_" + key], refl_all])
                W_all = {nm: np.concatenate([Rz["w_%s_%s" % (nm, key)], W_all[nm]]) for nm in W_all}
                ser_norm = list(Rz["norm_" + key]) + ser_norm; ser_Rt = list(Rz["Rt_" + key]) + ser_Rt; ser_2gE = list(Rz["2gE_" + key]) + ser_2gE
                pl = [L_ for L_ in prev["launches"] if L_["key"] == key][0]
                signed_rows = pl["signed_series"] + signed_rows; hz_rows = pl["halfzone_series"] + hz_rows
                Kmax = max(Kmax, pl["cheb_K_max"]); tailacc += pl["cheb_tail_acc"]
                maxR = np.maximum(maxR, np.array(pl["max_radius_lattice"]))
            arrays["t_" + key] = t_all; arrays["refl_" + key] = refl_all; arrays["norm_" + key] = np.array(ser_norm); arrays["Rt_" + key] = np.array(ser_Rt)
            arrays["2gE_" + key] = np.array(ser_2gE)
            for nm, v_ in W_all.items():
                arrays["w_%s_%s" % (nm, key)] = v_
            specE, WE = B1.spectrum(e, U, X, m, TH0, a.bin_TH, a.bin_max_TH)
            cons = float((WE - W0).abs().max())
            bo = (Ub.mH @ X).abs().pow(2).sum(dim=0).cpu().numpy() if Ub.shape[1] else np.zeros(X.shape[1])
            Rt_end = ser_Rt[-1] if np.isfinite(ser_Rt[-1]) else Ro
            inside = (X[: 2 * int(np.searchsorted(R, Rt_end, side="right"))].abs() ** 2).sum(dim=0).cpu().numpy()
            rows = []
            for c_ in range(X.shape[1]):
                pre = np.nonzero(~refl_all[:, c_])[0]
                fin = float(W_all["beyond_16rs"][pre[-1], c_]) if len(pre) else None
                w16_0 = float(((X0[:, c_].abs() ** 2).reshape(n, 2).sum(dim=1).cpu().numpy()[R > 16.0 * rs]).sum())
                arr = arrival_interp(np.concatenate([[0.0], (t_all[pre] - T0) / rs]), np.concatenate([[w16_0], W_all["beyond_16rs"][pre, c_]]), fin) if len(pre) else None
                arr_step = None
                if fin and fin > 1e-12 and len(pre):
                    jj = np.nonzero(W_all["beyond_16rs"][pre, c_] >= 0.5 * fin)[0]
                    arr_step = float((t_all[pre[jj[0]]] - T0) / rs) if len(jj) else None
                wbar = spec0[c_]["mean"]
                rows.append({"launch": info[c_], "spectrum_launch": spec0[c_], "omega_bar": wbar, "soft": bool(abs(wbar) < m), "w_above_m": spec0[c_]["w_above_m"],
                             "w_below_minus_m": spec0[c_]["w_below_minus_m"], "w_in_gap": spec0[c_]["w_in_gap"],
                             "beyond16_final_before_reflection": fin, "last_pre_reflection_after_launch_rs": float((t_all[pre[-1]] - T0) / rs) if len(pre) else None,
                             "beyond_Rcut_last_pre_reflection": float(W_all["beyond_Rcut"][pre[-1], c_]) if len(pre) else None,
                             "arrival_after_launch_rs": arr, "arrival_first_step_after_launch_rs": arr_step,
                             "pred_2rs_log": (2.0 * rs0 * math.log(1.0 / abs(wbar)) / rs) if 0 < abs(wbar) < 1 else None,
                             "pred_kin_log": (math.log(1.0 / abs(info[c_]["k0"])) / kin / rs) if (kin and kin > 0 and abs(info[c_]["k0"]) < 1.0) else None,
                             "reflection_clock_after_launch_rs": float(t_refl[c_] / rs), "interior_end": float(inside[c_]), "classifier_overlap_end": float(bo[c_]),
                             "classifier_overlap_launch": float(bo_launch[c_]), "max_radius_rs": float(maxR[c_] / rs),
                             "end": {nm: float(W_all[nm][-1, c_]) for nm in W_all}, "spectrum_end": specE[c_] if mode == "history" else None,
                             "end_includes_reflection": bool(refl_all[-1, c_])})
            # the delays against the comparators (same k0 and envelope)
            for c_, r_ in enumerate(rows):
                l_ = r_["launch"]
                if l_["comparator"]:
                    r_["comparator_of"] = None; continue
                cmp_ = [j for j, q in enumerate(rows) if q["launch"]["comparator"] and q["launch"]["k0"] == l_["k0"] and q["launch"]["sigma"] == l_["sigma"]]
                j = cmp_[0] if cmp_ else None
                r_["comparator_column"] = j
                ca = rows[j]["arrival_after_launch_rs"] if j is not None else None
                r_["comparator_arrival_after_launch_rs"] = ca
                r_["delay_minus_comparator_rs"] = (r_["arrival_after_launch_rs"] - ca) if (r_["arrival_after_launch_rs"] is not None and ca is not None) else None
            L_ = {"key": key, "T_launch_over_rs": TT[k0], "k0": k0, "kappa": kap, "E": E0, "two_gE": rs0, "T_H": TH0, "roots_dense": rd, "R_outer": Ro, "R_inner": Ri,
                  "kappa_in": kin, "classifier_states": int(bidx.numel()), "eig_seconds": eig_s, "cheb_K_max": Kmax, "cheb_tail_acc": tailacc, "spectrum_conservation_max": cons,
                  "columns": rows, "signed_launch": sg_launch, "signed_launch_window_controls": sg_ctrl, "signed_series": signed_rows, "halfzone_series": hz_rows,
                  "ext_shells": ext_rows, "cubic_control": cubic, "norm_defect_max": float(np.max(ser_norm)), "steps": len(ser_norm),
                  "max_radius_lattice": maxR.tolist(), "k_end": (k0 + 0) if mode == "fixed_field" else int(kend), "T_end_over_rs": float(t_all[-1] / rs),
                  "seconds": time.time() - t_c}
            if mode == "history":
                kE = int(kend)
                NE, vE = H.fields(kE)
                eE, UE = OPS.eigh_csr(OPS.clocked_csr(h0, Q, NE, vE), dev)
                sE, _ = B1.spectrum(eE, UE, X, m, TH0, a.bin_TH, a.bin_max_TH)
                L_["spectrum_end_on_end_record"] = sE
                del UE
                arrays["X_end_" + key] = X.cpu().numpy()
            out["launches"].append(L_)
            log("  kappa %d: %d classifier states; eig %.0f s; Cheb K max %d; spectrum conservation %.1e; norms' defect %.1e; arrivals %s; delays minus the comparator %s; %.0f s" % (
                kap, int(bidx.numel()), eig_s, Kmax, cons, L_["norm_defect_max"], ", ".join(B1.fmt(r_["arrival_after_launch_rs"], "%.2f") for r_ in rows),
                ", ".join(B1.fmt(r_.get("delay_minus_comparator_rs"), "%+.2f") for r_ in rows if not r_["launch"]["comparator"]), time.time() - t_c))
            json.dump(out, open(a.out_json, "w"), default=float)
            del U, e, ext, X, X0
            torch.cuda.empty_cache()
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if prev is not None:
        out["resumed_from"] = {"json": a.resume_json, "json_md5": md5(a.resume_json), "npz": a.resume_npz, "npz_md5": md5(a.resume_npz)}
    np.savez(a.out_npz, **arrays)
    json.dump(out, open(a.out_json, "w"), default=float)
    return out


# --------------------------------------------------------------------------------------------------- B.5 version 2: the dust and the sea comparator
def set_readouts(X, f, R, h0, Q, Nn, vn, Rt, rs_now, dev):
    S = (X.abs() ** 2).reshape(len(R), 2, -1).sum(dim=1).cpu().numpy()       # sites x c
    fw = np.asarray(f, float)
    by = S[R > Rt].sum(axis=0) if Rt else None
    b2 = S[R > rs_now].sum(axis=0)
    out = {"beyond_Rt_sum": float(np.dot(fw, by)) if by is not None else None, "beyond_Rt_max": float(by.max()) if (by is not None and len(by)) else None,
           "beyond_2gE_sum": float(np.dot(fw, b2)), "beyond_2gE_max": float(b2.max()) if len(b2) else None, "norm_defect": float(np.max(np.abs(S.sum(axis=0) - 1.0))) if len(fw) else 0.0}
    if Rt:
        hn = OPS.clocked_csr(h0, Q, Nn, vn)
        JX = OPS.Banded(OPS.number_current_csr(hn, R, Rt), dev).apply(X)
        EX = OPS.Banded(OPS.current_out_csr(hn, R, Rt), dev).apply(X)
        out["fixed_cut_number_current"] = float(np.dot(fw, (X.conj() * JX).sum(dim=0).real.cpu().numpy()))
        out["killing_energy_flux"] = float(np.dot(fw, (X.conj() * EX).sum(dim=0).real.cpu().numpy()))
    return out, S


def dust(a, H, dev):
    rs = H.rs_nom; g = H.g; m = a.m; R = H.R; n = len(R)
    O = np.load(a.columns_npz)
    tags = sorted({k_.split("_")[2] for k_ in O.files if k_.startswith("cols_dust_")})
    out = {"script": "engine_single_object_B_v2.py", "mode": "dust", "label": LABEL, "wording": WORDING, "signed": SIGNED % a.win, "no_mutual_information": "D15",
           "columns_npz": a.columns_npz, "columns_npz_md5": md5(a.columns_npz), "files": H.files, "launches": [],
           "readouts": ("per record: the occupation-weighted weight beyond R_t(t) and beyond 2GE(t) and the largest single column; the fixed-cut number current at R_t(t) "
                        "(Tr[X^dag i[h, Pi] X], the record's h); the moving-cut reassignment over the step (beyond R_t(t_{k+1}) minus beyond R_t(t_k), both of the state at t_{k+1}); "
                        "the fixed-cut change over the step (beyond R_t(t_k): the state at t_{k+1} minus at t_k); the Killing-energy flux Tr[X^dag (i/2)[h^2, Pi] X] at R_t(t); "
                        "separate columns, never summed; the selected sets' readouts, not the detector density (engine_single_object_slices_v1.py prints that)"),
           "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    arrays = {}
    prev = json.load(open(a.resume_json)) if a.resume_json else None
    Rz = np.load(a.resume_npz) if a.resume_npz else None
    TT = np.asarray(H.T, float) / rs
    kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
    for tg in tags:
        T_rs = float(O["T_%s" % tg]); k0 = int(np.argmin(np.abs(TT - T_rs)))
        for kap in [int(x) for x in a.kappas.split(",")]:
            kd = "cols_dust_%s_k%d" % (tg, kap)
            if kd not in O.files:
                continue
            t_c = time.time()
            key = "%s_k%d" % (tg, kap)
            h0, Q = OPS.chain_sparse(R, m, float(kap))
            sets = {}
            for nm in ("dust", "sealoc"):
                A_ = np.asarray(O["cols_%s_%s_k%d" % (nm, tg, kap)], np.complex128)
                Wc = np.zeros((2 * n, A_.shape[1]), complex); Wc[: A_.shape[0]] = A_
                f_ = np.asarray(O["f_%s_%s_k%d" % (nm, tg, kap)], float)
                sets[nm] = {"X": torch.as_tensor(Wc, dtype=CD, device=dev), "f": f_}
            N0, v0 = H.fields(k0)
            e, U = OPS.eigh_csr(OPS.clocked_csr(h0, Q, N0, v0), dev)
            E0 = float(H.E[k0]); TH0 = 1.0 / (4.0 * math.pi * 2.0 * g * E0)
            launch = {}
            for nm, s_ in sets.items():
                sp0, _ = B1.spectrum(e, U, s_["X"], m, TH0, a.bin_TH, a.bin_max_TH)
                fw = s_["f"]
                launch[nm] = {"n": int(s_["X"].shape[1]), "occupation_sum": float(fw.sum()), "per_column": sp0,
                              "w_below_minus_m": float(np.dot(fw, [q["w_below_minus_m"] for q in sp0])), "w_in_gap": float(np.dot(fw, [q["w_in_gap"] for q in sp0])),
                              "w_above_m": float(np.dot(fw, [q["w_above_m"] for q in sp0])), "mean_sum": float(np.dot(fw, [q["mean"] for q in sp0]))}
            del U, e
            kstart = k0
            if prev is not None:
                pl = [L_ for L_ in prev["launches"] if L_["key"] == key]
                if pl:
                    kstart = int(pl[0]["k_end"])
                    for nm in sets:
                        sets[nm]["X"] = torch.as_tensor(Rz["X_end_%s_%s" % (nm, key)], dtype=CD, device=dev)
            series = {nm: [] for nm in sets}; signed_rows = {nm: [] for nm in sets}
            every = max(1, int(round(a.signed_every_rs * rs / H.tau)))
            Rt_launch = H.rec[k0].get("R_t")
            for nm, s_ in sets.items():
                r0, S0 = set_readouts(s_["X"], s_["f"], R, h0, Q, N0, v0, Rt_launch, 2.0 * g * E0, dev)
                r0.update({"t_over_rs": TT[kstart], "R_t": H.rec[kstart].get("R_t"), "two_gE": 2.0 * g * float(H.E[kstart])})
                if prev is None:
                    series[nm].append(r0)
                    signed_rows[nm].append({"t_over_rs": TT[k0], **signed_inside_beyond(s_["X"], R, N0, v0, kap, m, Rt_launch, a.win)})
                s_["S_prev"] = (s_["X"].abs() ** 2).reshape(n, 2, -1).sum(dim=1).cpu().numpy()
            Kmax = 0; tailacc = 0.0
            for j, kk in enumerate(range(kstart, kend)):
                Nk, vk = H.fields(kk)
                Bk = OPS.Banded(OPS.clocked_csr(h0, Q, Nk, vk), dev)
                tau = float(H.T[kk + 1] - H.T[kk])
                Rt_old = H.rec[kk].get("R_t"); Rt_new = H.rec[kk + 1].get("R_t"); rs_new = 2.0 * g * float(H.E[kk + 1])
                Nn, vn = H.fields(kk + 1)
                for nm, s_ in sets.items():
                    s_["X"], K, tail = PH.cheb_apply(Bk, s_["X"], tau, a.cheb_tol)
                    Kmax = max(Kmax, K); tailacc += tail
                    r_, S1 = set_readouts(s_["X"], s_["f"], R, h0, Q, Nn, vn, Rt_new, rs_new, dev)
                    fw = s_["f"]
                    if Rt_old and Rt_new:
                        b_new_new = S1[R > Rt_new].sum(axis=0); b_new_old = S1[R > Rt_old].sum(axis=0); b_old_old = s_["S_prev"][R > Rt_old].sum(axis=0)
                        r_["moving_cut_reassignment_step"] = float(np.dot(fw, b_new_new - b_new_old))
                        r_["fixed_cut_change_step"] = float(np.dot(fw, b_new_old - b_old_old))
                    r_.update({"t_over_rs": float(H.T[kk + 1]) / rs, "R_t": Rt_new, "two_gE": rs_new, "tau": tau})
                    series[nm].append(r_)
                    s_["S_prev"] = S1
                    if (j + 1) % every == 0 or kk == kend - 1:
                        signed_rows[nm].append({"t_over_rs": float(H.T[kk + 1]) / rs, **signed_inside_beyond(s_["X"], R, Nn, vn, kap, m, Rt_new, a.win)})
            L_ = {"key": key, "tag": tg, "kappa": kap, "T_launch_over_rs": TT[k0], "k0": k0, "k_end": int(kend), "T_end_over_rs": TT[kend], "launch": launch,
                  "cheb_K_max": Kmax, "cheb_tail_acc": tailacc, "seconds": time.time() - t_c, "series": {}, "signed_series": {}}
            for nm in sets:
                ser = series[nm]; sgr = signed_rows[nm]
                if prev is not None:
                    pl = [q for q in prev["launches"] if q["key"] == key]
                    if pl:
                        ser = pl[0]["series"][nm] + ser; sgr = pl[0]["signed_series"][nm] + sgr
                L_["series"][nm] = ser; L_["signed_series"][nm] = sgr
                arrays["X_end_%s_%s" % (nm, key)] = sets[nm]["X"].cpu().numpy()
            out["launches"].append(L_)
            sd = L_["series"]["dust"][-1]; ss = L_["series"]["sealoc"][-1]
            log("dust %s kappa %d: dust %d columns (launch: below -m %.4f, gap %.4f, above m %.4f), sea-localized %d; at T/rs %.3f: the dust beyond R_t %s (launch %s), beyond 2gE %.4e; "
                "the sea-localized beyond R_t %s; the dust's fixed-cut current %s; %.0f s" % (
                    tg, kap, launch["dust"]["n"], launch["dust"]["w_below_minus_m"], launch["dust"]["w_in_gap"], launch["dust"]["w_above_m"], launch["sealoc"]["n"], sd["t_over_rs"],
                    B1.fmt(sd["beyond_Rt_sum"]), B1.fmt(L_["series"]["dust"][0]["beyond_Rt_sum"]), sd["beyond_2gE_sum"], B1.fmt(ss["beyond_Rt_sum"]),
                    B1.fmt(sd.get("fixed_cut_number_current"), "%+.3e"), L_["seconds"]))
            json.dump(out, open(a.out_json, "w"), default=float)
            for s_ in sets.values():
                del s_["X"]
            torch.cuda.empty_cache()
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    if prev is not None:
        out["resumed_from"] = {"json": a.resume_json, "json_md5": md5(a.resume_json), "npz": a.resume_npz, "npz_md5": md5(a.resume_npz)}
    np.savez(a.out_npz, **arrays)
    json.dump(out, open(a.out_json, "w"), default=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["fixed_field", "history", "dust"])
    ap.add_argument("--record", required=True); ap.add_argument("--fields", required=True)
    ap.add_argument("--record2", default=None); ap.add_argument("--fields2", default=None); ap.add_argument("--switch-T-rs", type=float, default=40.0)
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--launch-T", default="15.0,30.0"); ap.add_argument("--T-end-rs", type=float, default=40.0)
    ap.add_argument("--fixed-rs", type=float, default=300.0); ap.add_argument("--step-rs", type=float, default=0.25)
    ap.add_argument("--R-cut-rs", type=float, default=24.0); ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--bin-TH", type=float, default=0.5); ap.add_argument("--bin-max-TH", type=float, default=40.0)
    ap.add_argument("--ext-every-rs", type=float, default=25.0); ap.add_argument("--cubic-window-rs", type=float, default=8.0)
    ap.add_argument("--win", type=int, default=32); ap.add_argument("--win-controls", default="16,64"); ap.add_argument("--signed-every-rs", type=float, default=1.0)
    ap.add_argument("--columns-npz", default=None)
    ap.add_argument("--resume-json", default=None); ap.add_argument("--resume-npz", default=None)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--out-json", required=True); ap.add_argument("--out-npz", required=True)
    a = ap.parse_args()
    for p in (a.out_json, a.out_npz):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu is not None and a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    log("engine_single_object_B_v2 mode %s on %s; %s" % (a.mode, dev, LABEL))
    H = CombinedHistory(a.record, a.fields, a.nsites, a.record2, a.fields2, a.switch_T_rs if a.record2 else None)
    log("history: %d records, T/rs %.3f to %.3f (%s)" % (len(H.T), H.T[0] / H.rs_nom, H.T[-1] / H.rs_nom, json.dumps(H.files)))
    if a.mode in ("fixed_field", "history"):
        fixed_or_history(a, H, dev, a.mode)
    else:
        dust(a, H, dev)


if __name__ == "__main__":
    main()
