#!/usr/bin/env python
"""engine_single_object_B_v1.py (builder analyst; inbox #41, the revised S4 part B (the coordinator's go of 20:2x UTC: page_*_v6b's self-test
passed on the box); the owner-approved one-object ruling R2; verbatim in <internal>/solve_oneobject.md):
"B. Transfer by wavelength and depth. Machinery: the thin-column Chebyshev propagation of page_*_v6 (the quasi-frozen S1 A.4: the record's
applied fields piecewise constant per record, the vacuum Painleve-Gullstrand exterior appended to 128 r_s, 8192 sites, degree set by the
Bessel tail to 1e-14, no dense eigendecomposition in the propagation). 1. Backgrounds: (i) the fields of the records T/r_s 15.0 (first episode,
R_t 51.76) and 30.0 (second surface) of the r_s/a 64 history frozen, to 300 r_s; (ii) the solved history from each of those clocks to T/r_s 40.
2. Channels: the massive radial operator of the record (ma 0.2, the model's own species), kappa = 1 and 2, per channel; the massless channel
only under owner decision D14. 3. Launch set per background and channel: 2 depths (the middle of the trapped annulus between the inner and
outer roots; 2 sigma inside the outer root; the half-zone split of engine spec 36.3 (2) is reused for these) times 3 wavenumbers (ak_0 = 2 pi/64,
wavelength r_s; ak_0 = pi/4, the sign change of cos 2ak; ak_0 = pi/2, lattice scale), Gaussian envelopes of 16, 8 and 4 sites, both signs of k_0
printed, normalized columns; on the frozen record the Killing spectrum of each column at launch (1 dense eigendecomposition per record and
channel). 4. Readouts against the clock, per launch: the norm beyond the outer root, beyond 2GE(t), beyond 16 r_s and in the collar 16 to 24
r_s (R_cut 24; the far collar 32 to 48 as the control); the arrival clock (the clock at which half the final collar weight has arrived) and the
delay from launch, compared with 2 r_s ln(1/(a omega_bar)), omega_bar the column's mean Killing energy, and with kappa_in^(-1) ln(1/(ak_0)) from
the inner root's stored slope kappa_in; the Killing spectrum at arrival in the exterior block's Killing shells, against the launch spectrum:
equal to 1e-10 on the frozen record (a propagator check), shifted on the solved history; the interior weight remaining at the end and its
overlap with the bound states of the frozen h (eigenvectors with weight >= 0.99 inside R_cut), so that any weight that never leaves is printed
as bound-state content and never as trapping; for soft massive columns (omega_bar < m) the maximum radius reached and the weight beyond the
root and beyond 2GE at the end (a bound crossing; exit to infinity is barred by the mass threshold, printed as such), for hard columns
(omega_bar > m) the collar weight. 5. The state's own interior orbitals (eigenvalue >= 0.99 of Phi^dagger Pi_X Phi inside the outer root at
T/r_s 12.125 and 15.0), propagated on the solved history to T/r_s 40: their weight beyond the outer root and beyond 2GE(t) against the clock,
their outward current through the root, their Killing spectrum on the record's frozen h at launch. 6. The sigma/a 8 history as the scale
control." D: "unit norm of every column to 1e-12 per record; the 256-site time-dependent dense oracle with 3 of the launch columns, every
readout of B.4 to 1e-10 against the full propagator; the Killing-spectrum conservation of B.4 on the frozen records; the frozen-record drain as
the positive control (a wavelength-r_s column that neither leaves by 300 r_s nor overlaps the bound states is a defect of the chain or the
collar, to be found); the piecewise-constant against cubic field control on 1 window of 8 r_s around the launch clock, the change in every
readout printed." No mutual information anywhere (D15). LABEL (every output): "the single object's own cross-surface 2-point block and
transfer on the solved history of the massive collapse, per radial channel; the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s
11.75, the record's own limit; the massless channel, where present, a test channel under owner decision D14" (no massless channel here: D14).

The machinery (imported, never edited): page_history_v6b.History (the record's applied fields on the engine's sites, the vacuum PG exterior at
the record's E beyond them, to --nsites 8192 = 128 r_s; the Catmull-Rom fields for the cubic control), page_history_v6b.cheb_apply (the Chebyshev
action exp(-i tau h) X, its order by the Bessel tail below --cheb-tol 1e-14) and exterior_shells (the exterior block beyond R_cut: its dense
Killing eigenbasis, bins of --bin-TH T_H to --bin-max-TH T_H and one overflow bin, particles and holes, the outgoing and incoming parts by the
sign of i[h_X, R] in each bin; T_H = 1/(4 pi 2gE)); page_ops_v6.chain_sparse at m = --m (0.2) and kappa, clocked_csr (h[N, v] = (1/2){N, h_0}
+ (1/2){v, Q}), Banded, eigh_csr, number_current_csr. The builder's conventions (printed): the launch column g(R) e^{i k_0 R} u_+(k_0) with g the
Gaussian of width sigma (16, 8, 4 sites with ak_0 = 2 pi/64, pi/4, pi/2 in that order), centred at the depth, and u_+ the positive-energy spinor
of the local 2x2 symbol N(m sigma_z + sin k_0 sigma_x + (kappa/R_c) sigma_y) + (v/2) sin 2k_0 at the launch site (the model's own particle
branch at k_0), normalized; the depths: (r_in + R_t)/2 and R_t - 2 sigma by the launch record's dense roots (the record's stored zeros); the
regions: R > R_t(t) (the record's stored dense outer root; the frozen record's own), R > 2gE(t), R > 16 r_s, the collar 16 < R/r_s <= 24,
the far collar 32 < R/r_s <= 48; the arrival: the first clock at which the weight beyond 16 r_s reaches half its value at the end of the
window before the chain-end reflection; the reflection clock: a column's front leaving at unit speed from its launch radius reaches the
chain's end (128 r_s) and returns to R_cut at launch + (128 r_s - R_launch) + (128 - 24) r_s, the Page builder's 208 r_s beside it (a column
at R_cut); every readout after the earlier of the two is flagged "includes the chain-end reflection"; the maximum radius reached: the largest
over the clocks of the outermost site whose tail weight exceeds 1e-3; the bound states: the eigenvectors of the frozen h with weight >= 0.99
inside R_cut; the half-zone split beyond the outer root by version 3's DFT convention (|ak| < pi/2 against >= pi/2) as the branch readout.
Modes: --mode frozen (B.1 (i)), history (B.1 (ii) with the cubic control on --cubic-window-rs), orbitals (B.5, the columns from
--orbitals-npz), oracle (D's 256-site dense check). Args: see main(). Every number is printed by the script."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import argparse
import json
import math
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in ([os.environ["PAGE_V6_PATH"]] if os.environ.get("PAGE_V6_PATH") else []) + [os.path.join(HERE, "..", "page"), "<internal>/code_v6"]:
    if os.path.isdir(_p):
        sys.path.insert(1, _p)
import page_ops_v6 as OPS            # noqa: E402
import page_history_v6b as PH        # noqa: E402

LABEL = ("the single object's own cross-surface 2-point block and transfer on the solved history of the massive collapse, per radial channel; "
         "the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit; the massless channel, where present, "
         "a test channel under owner decision D14")
KS = ((2.0 * math.pi / 64.0, 16.0, "2pi/64 (wavelength r_s)"), (math.pi / 4.0, 8.0, "pi/4 (cos 2ak changes sign)"), (math.pi / 2.0, 4.0, "pi/2 (lattice scale)"))
CD = OPS.CD


def fmt(x, f="%.3e"):
    return (f % x) if isinstance(x, (int, float)) and not isinstance(x, bool) and x is not None and x == x else "-"


def particle_spinor(m, k0, kappa, Rc):
    """the positive-energy eigenvector of m sigma_z + sin k0 sigma_x + (kappa/Rc) sigma_y (the local symbol's spinor; N and the flow's identity
    part do not change it)."""
    sz = np.array([[1, 0], [0, -1]], complex); sx = np.array([[0, 1], [1, 0]], complex); sy = np.array([[0, -1j], [1j, 0]], complex)
    A = m * sz + math.sin(k0) * sx + (kappa / Rc) * sy
    e, V = np.linalg.eigh(A)
    return V[:, 1]


def launch_columns(R, m, kappa, roots, N, v):
    """the launch set: 2 depths x 3 (k0, sigma) x 2 signs of k0; returns the columns (2n x 12, normalized) and their descriptions."""
    Ro, Ri = roots
    cols, info = [], []
    for depth_name in ("annulus middle", "2 sigma inside the outer root"):
        for k0, sig, kname in KS:
            Rc = 0.5 * (Ri + Ro) if depth_name == "annulus middle" else Ro - 2.0 * sig
            for sgn in (1, -1):
                g = np.exp(-0.5 * ((R - Rc) / sig) ** 2) * np.exp(1j * sgn * k0 * R)
                ic = int(np.argmin(np.abs(R - Rc)))
                u = particle_spinor(m, sgn * k0, kappa, max(Rc, 1.0))
                x = np.zeros(2 * len(R), complex)
                x[0::2] = g * u[0]; x[1::2] = g * u[1]
                x /= np.linalg.norm(x)
                cols.append(x)
                info.append({"depth": depth_name, "R_c": float(Rc), "k0": float(sgn * k0), "ak0_name": kname, "sigma": sig, "spinor": [complex(u[0]).real, complex(u[0]).imag, complex(u[1]).real, complex(u[1]).imag],
                             "N_at": float(N[ic]), "v_at": float(v[ic])})
    return np.array(cols).T, info


def spectrum(e, U, X, m, TH, bin_TH=0.5, bin_max_TH=40.0):
    """the Killing spectrum of the columns X in the eigenbasis (e, U) of h: per column the weights, the mean energy, the weights at e < 0,
    |e| < m, e > m, and the binned weights in |e|/T_H (the shells' bins, particles and holes apart); returns (summary rows, the weights)."""
    W = (U.mH @ X).abs() ** 2                       # (n2 x c)
    en = e.cpu().numpy(); Wn = W.cpu().numpy()
    nb = int(round(bin_max_TH / bin_TH))
    edges = np.array([b * bin_TH for b in range(nb + 1)] + [1e30])
    rows = []
    for j in range(Wn.shape[1]):
        w = Wn[:, j]
        mean = float(np.sum(en * w)); sp_ = float(math.sqrt(max(np.sum(en * en * w) - mean ** 2, 0.0)))
        b_p = np.histogram(np.abs(en[en > 0]) / TH, bins=edges, weights=w[en > 0])[0]; b_h = np.histogram(np.abs(en[en < 0]) / TH, bins=edges, weights=w[en < 0])[0]
        rows.append({"norm": float(w.sum()), "mean": mean, "spread": sp_, "w_negative": float(w[en < 0].sum()), "w_in_gap": float(w[np.abs(en) < m].sum()),
                     "w_above_m": float(w[en > m].sum()), "w_below_minus_m": float(w[en < -m].sum()), "bins_particles": b_p.tolist(), "bins_holes": b_h.tolist()})
    return rows, W


def region_weights(X, R, cuts):
    """per column the weight beyond each radius in cuts (dict name -> (lo, hi)): sum over the sites lo < R <= hi."""
    S = (X.abs() ** 2).reshape(len(R), 2, -1).sum(dim=1)        # sites x c
    out = {}
    Rt = torch.as_tensor(R, dtype=OPS.RD, device=X.device)
    for name, (lo, hi) in cuts.items():
        msk = (Rt > lo) & (Rt <= hi)
        out[name] = S[msk].sum(dim=0).cpu().numpy()
    tail = torch.flip(torch.cumsum(torch.flip(S, [0]), 0), [0]).cpu().numpy()   # tail[i] = weight at sites >= i
    return out, tail


def halfzone_beyond(X, i0):
    """version 3's DFT split of the columns' part beyond site i0: per column the weights at |ak| < pi/2 and >= pi/2 (and the quarters)."""
    M = X.shape[0] // 2 - i0
    if M <= 0:
        return None
    blk = X[2 * i0:].reshape(M, 2, -1)
    Xk = torch.fft.fft(blk, dim=0, norm="ortho")
    w = (Xk.abs() ** 2).sum(dim=1).cpu().numpy()          # M x c
    qs = np.rint(np.fft.fftfreq(M) * M).astype(np.int64); aq = np.abs(qs)
    lo = 4 * aq < M; q0 = 8 * aq < M
    return {"lo": w[lo].sum(axis=0).tolist(), "hi": w[~lo].sum(axis=0).tolist(), "below_pi4": w[q0].sum(axis=0).tolist()}


class _R:
    def __init__(self, R):
        self.R = R


def log(msg):
    print(msg, flush=True)


def dense_roots(H, Z, k, fields_module, ell):
    """the dense roots of N + v at record k with their slopes (part A's route: the record's stored shell totals re-solved by the run's module)."""
    import importlib
    import engine_single_object_A_v2 as A2
    FM = importlib.import_module(fields_module)
    _, _, _, info = FM.solve_constraints_ivp(np.asarray(Z["eps_tot"][k], float), np.asarray(Z["J_tot"][k], float), H.g, H.Re, ell, collar_policy="flag", support_rel=1e-14)
    return A2.roots_dense(info, ell) if not info.get("failed") else []


def cuts_of(rs, Ro, two_gE, R_cut_rs):
    c = {"beyond_outer_root": (Ro, 1e30) if Ro is not None else None, "beyond_2GE": (two_gE, 1e30), "beyond_16rs": (16.0 * rs, 1e30),
         "collar_16_24": (16.0 * rs, R_cut_rs * rs), "far_32_48": (32.0 * rs, 48.0 * rs), "beyond_Rcut": (R_cut_rs * rs, 1e30)}
    return {k_: v_ for k_, v_ in c.items() if v_ is not None}


def ext_spectrum(ext, X):
    """per column the weight of the part beyond R_cut in the exterior block's shells (per sign and bin, outgoing and incoming)."""
    XX = X[ext["idx"]]
    rows = []
    for bn in ext["bins"]:
        a_o = (bn["O"].mH @ XX).abs().pow(2).sum(dim=0).cpu().numpy(); a_i = (bn["I"].mH @ XX).abs().pow(2).sum(dim=0).cpu().numpy()
        rows.append({"sign": bn["sign"], "bin": bn["bin"], "lo": bn["lo"], "hi": bn["hi"], "out": a_o.tolist(), "in": a_i.tolist()})
    return rows


def frozen_or_history(a, H, Zf, dev, mode):
    rs = H.rs_nom; g = H.g; m = a.m; R = H.R; n = len(R)
    out = {"script": "engine_single_object_B_v1.py", "mode": mode, "label": LABEL, "no_mutual_information": "D15", "massless": "not run (owner decision D14)",
           "record": a.record, "fields": a.fields, "nsites": n, "m": m, "cheb_tol": a.cheb_tol, "R_cut_rs": a.R_cut_rs, "launches": [],
           "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    TT = np.asarray(H.T, float) / rs
    for T_rs in [float(x) for x in a.launch_T.split(",")]:
        k0 = int(np.argmin(np.abs(TT - T_rs)))
        rec = H.rec[k0]; zeros = rec.get("trapping_zeros") or []
        N0, v0 = H.fields(k0); E0 = float(H.E[k0]); rs0 = 2.0 * g * E0; TH0 = 1.0 / (4.0 * math.pi * rs0)
        rd = dense_roots(H, Zf, k0, a.fields_module, H.ell)
        pos = [q for q in rd if q["slope"] > 0]; Ro = max(q["R"] for q in pos) if pos else (rec.get("R_t"))
        neg = [q for q in rd if q["slope"] < 0 and q["R"] < (Ro or 0)]; Ri = max(q["R"] for q in neg) if neg else rec.get("r_in")
        kin = abs([q for q in rd if q["R"] == Ri][0]["slope"]) if (neg and Ri is not None) else None
        log("launch T/rs %.3f (k %d): E %.6f, 2gE %.4f, T_H %.4e; the dense roots %s (stored %s); outer %s, inner %s, kappa_in %s" % (
            TT[k0], k0, E0, rs0, TH0, ", ".join("%.4f (%+.5f)" % (q["R"], q["slope"]) for q in rd), zeros, fmt(Ro, "%.4f"), fmt(Ri, "%.4f"), fmt(kin, "%.5f")))
        for kap in [int(x) for x in a.kappas.split(",")]:
            t_c = time.time()
            h0, Q = OPS.chain_sparse(R, m, float(kap))
            h = OPS.clocked_csr(h0, Q, N0, v0)
            e, U = OPS.eigh_csr(h, dev)
            i_cut = int(np.searchsorted(R, a.R_cut_rs * rs, side="right"))
            wi = (U[: 2 * i_cut].abs() ** 2).sum(dim=0)
            bidx = torch.nonzero(wi >= 0.99).ravel()
            Ub = U[:, bidx]
            X0np, info = launch_columns(R, m, kap, (Ro, Ri), N0, v0)
            X = torch.as_tensor(X0np, dtype=CD, device=dev)
            spec0, W0 = spectrum(e, U, X, m, TH0, a.bin_TH, a.bin_max_TH)
            eig_s = time.time() - t_c
            ext = PH.exterior_shells(_R(R), h, a.R_cut_rs * rs, TH0, a.bin_TH, a.bin_max_TH, dev) if mode == "frozen" else None
            Rl = np.array([q["R_c"] for q in info])
            t_refl = (128.0 * rs - Rl) + (128.0 - a.R_cut_rs) * rs           # lattice units after launch
            series = []
            T0 = float(H.T[k0])
            if mode == "frozen":
                nst = int(round(a.frozen_rs / a.step_rs)); dt = a.step_rs * rs
                clocks = [T0 + (j + 1) * dt for j in range(nst)]
            else:
                kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
                clocks = [float(H.T[kk + 1]) for kk in range(k0, kend)]
            Kmax = 0; tailacc = 0.0
            Bh = OPS.Banded(h, dev) if mode == "frozen" else None
            ext_rows = []
            maxR = np.zeros(X.shape[1])
            Xc = X.clone() if (mode == "history" and a.cubic_window_rs and T_rs == float(a.launch_T.split(",")[0])) else None
            cubic = None
            for j, tc in enumerate(clocks):
                if mode == "frozen":
                    X, K, tail = PH.cheb_apply(Bh, X, dt, a.cheb_tol)
                    kk_now = k0
                else:
                    kk = k0 + j
                    Nk, vk = H.fields(kk)
                    Bk = OPS.Banded(OPS.clocked_csr(h0, Q, Nk, vk), dev)
                    X, K, tail = PH.cheb_apply(Bk, X, float(H.T[kk + 1] - H.T[kk]), a.cheb_tol)
                    kk_now = kk + 1
                    if Xc is not None and float(H.T[kk + 1]) <= T0 + a.cubic_window_rs * rs + 1e-9:
                        t_s = float(H.T[kk]); t_e = float(H.T[kk + 1]); nsub = int(round(t_e - t_s))
                        for s_ in range(nsub):
                            Nc, vc = H.cubic_fields(t_s + s_ + 0.5)
                            Xc, _, _ = PH.cheb_apply(OPS.Banded(OPS.clocked_csr(h0, Q, Nc, vc), dev), Xc, 1.0, a.cheb_tol)
                        if float(H.T[kk + 1]) >= T0 + a.cubic_window_rs * rs - 1e-9:
                            cubic = {"window_rs": [T0 / rs, float(H.T[kk + 1]) / rs], "columns_diff_max": float((Xc - X).abs().max()),
                                     "region_diff": {}}
                            rc_, _ = region_weights(Xc, R, cuts_of(rs, Ro, rs0, a.R_cut_rs)); rp_, _ = region_weights(X, R, cuts_of(rs, Ro, rs0, a.R_cut_rs))
                            cubic["region_diff"] = {k_: float(np.max(np.abs(rc_[k_] - rp_[k_]))) for k_ in rp_}
                            Xc = None
                Kmax = max(Kmax, K); tailacc += tail
                recn = H.rec[kk_now]
                Rt_now = Ro if mode == "frozen" else recn.get("R_t")
                rs_now = rs0 if mode == "frozen" else 2.0 * g * float(H.E[kk_now])
                cw, tail_w = region_weights(X, R, cuts_of(rs, Rt_now, rs_now, a.R_cut_rs))
                for c_ in range(X.shape[1]):
                    idx = np.nonzero(tail_w[:, c_] > 1e-3)[0]
                    if len(idx):
                        maxR[c_] = max(maxR[c_], float(R[idx[-1]]))
                norms = (X.abs() ** 2).sum(dim=0).cpu().numpy()
                i_Rt = int(np.searchsorted(R, Rt_now, side="right")) if Rt_now else None
                hz = halfzone_beyond(X, i_Rt) if i_Rt is not None else None
                series.append({"t": tc, "t_over_rs": tc / rs, "after_launch_rs": (tc - T0) / rs, "R_t": Rt_now, "two_gE": rs_now,
                               "norm_minus_1_max": float(np.max(np.abs(norms - 1.0))), "w": {k_: v_.tolist() for k_, v_ in cw.items()}, "halfzone_beyond_Rt": hz,
                               "reflection_reached": [bool(tc - T0 >= t_refl[c_] or tc - T0 >= 208.0 * rs) for c_ in range(X.shape[1])]})
                if mode == "frozen" and (abs(((tc - T0) / rs) % a.ext_every_rs) < 1e-9 or j == len(clocks) - 1):
                    ext_rows.append({"after_launch_rs": (tc - T0) / rs, "shells": ext_spectrum(ext, X)})
            # the end
            specE, WE = spectrum(e, U, X, m, TH0, a.bin_TH, a.bin_max_TH)
            cons = float((WE - W0).abs().max())
            bo = (Ub.mH @ X).abs().pow(2).sum(dim=0).cpu().numpy() if Ub.shape[1] else np.zeros(X.shape[1])
            inside = (X[: 2 * int(np.searchsorted(R, (series[-1]["R_t"] or Ro), side="right"))].abs() ** 2).sum(dim=0).cpu().numpy()
            rows = []
            for c_ in range(X.shape[1]):
                pre = [s_ for s_ in series if not s_["reflection_reached"][c_]]
                fin = pre[-1]["w"]["beyond_16rs"][c_] if pre else None
                arr = next((s_ for s_ in pre if fin and s_["w"]["beyond_16rs"][c_] >= 0.5 * fin), None) if fin and fin > 1e-12 else None
                wbar = spec0[c_]["mean"]
                rows.append({"launch": info[c_], "spectrum_launch": spec0[c_], "omega_bar": wbar, "soft": bool(abs(wbar) < m),
                             "beyond16_final_before_reflection": fin, "arrival_after_launch_rs": arr["after_launch_rs"] if arr else None,
                             "pred_2rs_log": (2.0 * rs0 * math.log(1.0 / abs(wbar)) / rs) if abs(wbar) > 0 and abs(wbar) < 1 else None,
                             "pred_kin_log": (math.log(1.0 / abs(info[c_]["k0"])) / kin / rs) if (kin and kin > 0) else None,
                             "reflection_clock_after_launch_rs": float(t_refl[c_] / rs), "reflection_page_builder_rs": 208.0,
                             "interior_end": float(inside[c_]), "bound_overlap_end": float(bo[c_]), "bound_overlap_launch": float((Ub.mH @ torch.as_tensor(X0np[:, c_:c_ + 1], dtype=CD, device=dev)).abs().pow(2).sum()) if Ub.shape[1] else 0.0,
                             "max_radius_rs": float(maxR[c_] / rs), "end": {k_: series[-1]["w"][k_][c_] for k_ in series[-1]["w"]},
                             "spectrum_end": specE[c_] if mode == "history" else None})
            L_ = {"T_launch_over_rs": TT[k0], "k0": k0, "kappa": kap, "E": E0, "two_gE": rs0, "T_H": TH0, "roots_dense": rd, "R_outer": Ro, "R_inner": Ri, "kappa_in": kin,
                  "bound_states": int(bidx.numel()), "eig_seconds": eig_s, "cheb_K_max": Kmax, "cheb_tail_acc": tailacc, "spectrum_conservation_max": cons,
                  "columns": rows, "series": series, "ext_shells": ext_rows, "cubic_control": cubic, "seconds": time.time() - t_c}
            if mode == "history":
                kE = k0 + len(clocks)
                NE, vE = H.fields(kE)
                eE, UE = OPS.eigh_csr(OPS.clocked_csr(h0, Q, NE, vE), dev)
                sE, _ = spectrum(eE, UE, X, m, TH0, a.bin_TH, a.bin_max_TH)
                L_["spectrum_end_on_end_record"] = sE; L_["T_end_over_rs"] = TT[kE]
                del UE
            out["launches"].append(L_)
            log("  kappa %d: %d bound states; eig %.0f s; Cheb K max %d, tail %.1e; spectrum conservation %.1e; the norms' largest defect %.1e; collar weights at the end %s; arrival (r_s after launch) %s; %.0f s" % (
                kap, int(bidx.numel()), eig_s, Kmax, tailacc, cons, max(s_["norm_minus_1_max"] for s_ in series),
                ", ".join("%.2e" % r_["end"].get("collar_16_24", float("nan")) for r_ in rows), ", ".join(fmt(r_["arrival_after_launch_rs"], "%.1f") for r_ in rows), time.time() - t_c))
            json.dump(out, open(a.out_json, "w"), default=float)
            del U, e, ext, X
            torch.cuda.empty_cache()
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    json.dump(out, open(a.out_json, "w"), default=float)
    return out


def orbitals(a, H, Zf, dev):
    """B.5: the state's own interior orbitals (from --orbitals-npz: per launch tag and kappa the columns on the engine's sites) on the solved
    history to --T-end-rs: the weight beyond R_t(t) and beyond 2gE(t) per record (summed and the largest single orbital), the outward number
    current through R_t(t) (Tr[W^dagger i[h, Pi_{R > R_t}] W]), the Killing spectrum on the launch record's frozen h."""
    rs = H.rs_nom; g = H.g; m = a.m; R = H.R; n = len(R)
    O = np.load(a.orbitals_npz)
    tags = sorted({k_.split("_")[1] for k_ in O.files if k_.startswith("cols_")})
    out = {"script": "engine_single_object_B_v1.py", "mode": "orbitals", "label": LABEL, "no_mutual_information": "D15", "orbitals_npz": a.orbitals_npz, "launches": [],
           "utc_start": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    TT = np.asarray(H.T, float) / rs
    for tg in tags:
        T_rs = float(O["T_%s" % tg]); k0 = int(np.argmin(np.abs(TT - T_rs)))
        kend = int(np.nonzero(TT <= a.T_end_rs + 1e-9)[0][-1])
        for kap in [int(x) for x in a.kappas.split(",")]:
            key = "cols_%s_k%d" % (tg, kap)
            if key not in O.files:
                continue
            t_c = time.time()
            Wc = np.zeros((2 * n, O[key].shape[1]), complex); Wc[: O[key].shape[0]] = O[key]
            W = torch.as_tensor(Wc, dtype=CD, device=dev)
            h0, Q = OPS.chain_sparse(R, m, float(kap))
            N0, v0 = H.fields(k0)
            e, U = OPS.eigh_csr(OPS.clocked_csr(h0, Q, N0, v0), dev)
            E0 = float(H.E[k0]); TH0 = 1.0 / (4.0 * math.pi * 2.0 * g * E0)
            sp0, _ = spectrum(e, U, W, m, TH0, a.bin_TH, a.bin_max_TH)
            del U, e
            tot = {"mean_energy_sum": float(sum(q["mean"] for q in sp0)), "w_negative": float(sum(q["w_negative"] for q in sp0)), "w_in_gap": float(sum(q["w_in_gap"] for q in sp0)),
                   "w_above_m": float(sum(q["w_above_m"] for q in sp0)), "n": int(W.shape[1])}
            series = []
            for kk in range(k0, kend):
                Nk, vk = H.fields(kk)
                hk = OPS.clocked_csr(h0, Q, Nk, vk)
                W, K, tail = PH.cheb_apply(OPS.Banded(hk, dev), W, float(H.T[kk + 1] - H.T[kk]), a.cheb_tol)
                recn = H.rec[kk + 1]; Rt = recn.get("R_t"); rsn = 2.0 * g * float(H.E[kk + 1])
                S = (W.abs() ** 2).reshape(n, 2, -1).sum(dim=1).cpu().numpy()
                beyond_Rt = S[R > Rt].sum(axis=0) if Rt else None
                beyond_2gE = S[R > rsn].sum(axis=0)
                cur = None
                if Rt:
                    Nn, vn = H.fields(kk + 1)
                    Jc = OPS.number_current_csr(OPS.clocked_csr(h0, Q, Nn, vn), R, Rt)
                    JW = OPS.Banded(Jc, dev).apply(W)
                    cur = (W.conj() * JW).sum(dim=0).real.cpu().numpy()
                series.append({"t_over_rs": float(H.T[kk + 1]) / rs, "R_t": Rt, "two_gE": rsn, "beyond_Rt_sum": float(beyond_Rt.sum()) if beyond_Rt is not None else None,
                               "beyond_Rt_max": float(beyond_Rt.max()) if beyond_Rt is not None else None, "beyond_2gE_sum": float(beyond_2gE.sum()), "beyond_2gE_max": float(beyond_2gE.max()),
                               "current_out_sum": float(cur.sum()) if cur is not None else None, "norm_defect": float(np.max(np.abs(S.sum(axis=0) - 1.0)))})
            L_ = {"tag": tg, "T_launch_over_rs": TT[k0], "kappa": kap, "n_orbitals": int(W.shape[1]), "lambda_min": float(np.min(O["lam_%s_k%d" % (tg, kap)])) if ("lam_%s_k%d" % (tg, kap)) in O.files else None,
                  "spectrum_launch_total": tot, "spectrum_launch": sp0, "series": series, "seconds": time.time() - t_c}
            out["launches"].append(L_)
            s_end = series[-1]
            log("orbitals %s kappa %d: %d orbitals; launch spectrum: weights e < 0 %.4f, |e| < m %.4f, e > m %.4f (of %d); at T/rs %.3f: beyond R_t %s (largest single %s), beyond 2gE %.4e; outward current %s; %.0f s" % (
                tg, kap, W.shape[1], tot["w_negative"], tot["w_in_gap"], tot["w_above_m"], tot["n"], s_end["t_over_rs"], fmt(s_end["beyond_Rt_sum"]), fmt(s_end["beyond_Rt_max"]),
                s_end["beyond_2gE_sum"], fmt(s_end["current_out_sum"], "%+.3e"), L_["seconds"]))
            json.dump(out, open(a.out_json, "w"), default=float)
            del W
            torch.cuda.empty_cache()
    out["utc_end"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    json.dump(out, open(a.out_json, "w"), default=float)
    return out


def oracle(a, dev):
    """D: the 256-site synthetic history of page_history_oracle_v6b (its moving dust-ball profile), 3 of the launch columns (the three
    wavenumbers at the annulus middle of the launch record's applied-field roots), propagated per record by the Chebyshev action and by the
    dense propagator (U exp(-i e tau) U^dagger of each record's dense eigendecomposition): every readout's largest difference."""
    import page_history_oracle_v6b as PO
    SH = PO.synthetic_history()
    H = PH.History(None, None, nsites=256, synthetic=SH)
    rs = H.rs_nom or float(SH["json"]["args"]["rs"]); R = H.R; n = len(R); m = a.m
    TT = np.asarray(H.T, float) / rs
    k0 = int(np.argmin(np.abs(TT - a.oracle_T_rs)))
    N0, v0 = H.fields(k0)
    f = N0 + v0
    rts = [float(R[i] + f[i] * (R[i + 1] - R[i]) / (f[i] - f[i + 1])) for i in np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]]
    Ro = max(rts) if rts else 0.5 * n; Ri = min(rts) if len(rts) > 1 else 0.25 * Ro
    X0np, info = launch_columns(R, m, 1, (Ro, Ri), N0, v0)
    pick = [0, 2, 4]                                    # the three wavenumbers, +k0, at the annulus middle
    X0np = X0np[:, pick]; info = [info[i] for i in pick]
    h0, Q = OPS.chain_sparse(R, m, 1.0)
    Xc = torch.as_tensor(X0np, dtype=CD, device=dev); Xd = Xc.clone()
    two_gE = 2.0 * H.g * float(H.E[k0])
    cuts = {"beyond_outer_root": (Ro, 1e30), "beyond_2GE": (two_gE, 1e30), "beyond_8rs": (8.0 * rs, 1e30), "collar_8_12": (8.0 * rs, 12.0 * rs), "beyond_12rs": (12.0 * rs, 1e30)}
    diffs = {k_: 0.0 for k_ in cuts}; diffs["columns"] = 0.0; diffs["norm"] = 0.0
    for kk in range(k0, len(H.T) - 1):
        Nk, vk = H.fields(kk); hk = OPS.clocked_csr(h0, Q, Nk, vk)
        tau = float(H.T[kk + 1] - H.T[kk])
        Xc, _, _ = PH.cheb_apply(OPS.Banded(hk, dev), Xc, tau, a.cheb_tol)
        e, U = OPS.eigh_csr(hk, dev)
        ph = torch.exp(-1j * e.to(CD) * tau)
        Xd = U @ (ph[:, None] * (U.mH @ Xd))
        wc, _ = region_weights(Xc, R, cuts); wd, _ = region_weights(Xd, R, cuts)
        for k_ in cuts:
            diffs[k_] = max(diffs[k_], float(np.max(np.abs(wc[k_] - wd[k_]))))
        diffs["columns"] = max(diffs["columns"], float((Xc - Xd).abs().max()))
        diffs["norm"] = max(diffs["norm"], float(((Xc.abs() ** 2).sum(dim=0) - 1.0).abs().max()))
    e, U = OPS.eigh_csr(OPS.clocked_csr(h0, Q, N0, v0), dev)
    sc, _ = spectrum(e, U, Xc, m, 1.0 / (4 * math.pi * two_gE)); sd, _ = spectrum(e, U, Xd, m, 1.0 / (4 * math.pi * two_gE))
    diffs["spectrum_mean"] = max(abs(p["mean"] - q["mean"]) for p, q in zip(sc, sd))
    out = {"script": "engine_single_object_B_v1.py", "mode": "oracle", "fixture": "page_history_oracle_v6b.synthetic_history() (256 sites, r_s 12, tau 2)", "launch_T_over_rs": TT[k0],
           "roots": rts, "columns": info, "records": len(H.T) - 1 - k0, "max_abs_diff": diffs, "pass_1e-10": bool(max(diffs.values()) <= 1e-10), "label": LABEL}
    json.dump(out, open(a.out_json, "w"), default=float)
    log("oracle: %d records from T/rs %.3f, roots %s; the largest differences Chebyshev against dense: %s; pass at 1e-10: %s" % (
        out["records"], TT[k0], ["%.3f" % r_ for r_ in rts], {k_: "%.1e" % v_ for k_, v_ in diffs.items()}, out["pass_1e-10"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["frozen", "history", "orbitals", "oracle"])
    ap.add_argument("--record", default=None); ap.add_argument("--fields", default=None)
    ap.add_argument("--fields-module", default="engine_fields_v14")
    ap.add_argument("--nsites", type=int, default=8192); ap.add_argument("--m", type=float, default=0.2); ap.add_argument("--kappas", default="1,2")
    ap.add_argument("--launch-T", default="15.0,30.0"); ap.add_argument("--T-end-rs", type=float, default=40.0)
    ap.add_argument("--frozen-rs", type=float, default=300.0); ap.add_argument("--step-rs", type=float, default=1.0)
    ap.add_argument("--R-cut-rs", type=float, default=24.0); ap.add_argument("--cheb-tol", type=float, default=1e-14)
    ap.add_argument("--bin-TH", type=float, default=0.5); ap.add_argument("--bin-max-TH", type=float, default=40.0)
    ap.add_argument("--ext-every-rs", type=float, default=25.0); ap.add_argument("--cubic-window-rs", type=float, default=8.0)
    ap.add_argument("--orbitals-npz", default=None); ap.add_argument("--oracle-T-rs", type=float, default=6.0)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--out-json", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda:%d" % a.gpu) if (a.gpu is not None and a.gpu >= 0 and torch.cuda.is_available()) else torch.device("cpu")
    log("engine_single_object_B_v1 mode %s on %s; %s" % (a.mode, dev, LABEL))
    if a.mode == "oracle":
        return oracle(a, dev)
    H = PH.History(a.record, a.fields, a.nsites)
    Zf = np.load(a.fields, allow_pickle=True)
    if a.mode in ("frozen", "history"):
        frozen_or_history(a, H, Zf, dev, a.mode)
    else:
        orbitals(a, H, Zf, dev)


if __name__ == "__main__":
    main()
