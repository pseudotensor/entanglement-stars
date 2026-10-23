#!/usr/bin/env python
"""pt3b_v2lib.py (busi; B' v2, the changes the two checks of PAGE_Bprime_design_v1 name: ReviewA_bpchk_v1 and ReviewB_bpchk_v1, inbox #454 (2)).
New functions only; pt3b_cube / pt3b_kernels / pt3b_chain are imported unchanged.

1. The production averaging kernel (reviewer A "wrong" 4, reviewer B "averaging kernel"): the engine's back-average is
   N_n = int 4 pi r^2 K_0(r, R_n) N(r) dr, v_n = int 4 pi r^2 K_1(r, R_n) v(r) dr (engine_fields_v12.site_fields_kernel_average, line 817, with
   engine_fields_v4.kernels, line 37: the shell average of the 3D Gaussian of standard deviation ell). Its 3D form on the cube is the Gaussian
   convolution of std ell of N and of each Cartesian component of v rhat (a radial vector field), computed here on a box padded by 8 ell so
   that no face normalization enters (gauss_cube_fields). For the chains the engine's own 1D formula at the chain sites (engine_avg_radial).
   The v1 cos^2 kernel remains selectable ('cosine') and gap10's compact (1 - s^2/ell^2)^5 kernel ('compact') as a labelled alternative.
2. The clock-scan ramp relative to the baseline averaged root (reviewer A "wrong" 2, reviewer B: zero at and outside the root): f = 1 below
   0.85 R_b, 0 above R_b, R_b the d = 0 averaged outer root on the axis; 'L' keeps the pc13 literal (0.85, 1) L.
3. Exterior-control geometry (reviewer A "wrong" 1, reviewer B): box half-side = onset + 0.5 L at least; the moved splice runs with onset 2.5 L.
4. Deterministic interior windows (G11): eigsh with a fixed seeded start vector; the near-zero cluster printed (largest excluded |e|, smallest
   kept |e|, the threshold).
5. Pole selection (G10): the pole nearest the input's own energy among the pencil's significant poles; an error from the half sample with the
   same rule; poles resolved only when T_obs Gamma >= 0.1 or T_obs >= 100 L; distinct poles deduplicated across inputs.
6. Censored thresholds (reviewer B): N_acc at theta with T_theta beyond the run is 'censored' with a lower bound, never 0.
7. The G4 reader: per bin and per union of bins the output entropy Tr s(C^(1/2) G_S C^(1/2)) (exact for a number-conserving Gaussian input
   covariance C on the propagated span and the dilation's bins, which are orthogonal), S(b_n | R_(n-1)), the early-late mutual information at
   every split, the outgoing number, and the retained-output correlation modes (eigenvectors of G(T) with 0 < p < 1). Guarded: a C with a
   nonzero supply-exit block is refused unless allowed explicitly (reviewer A's Nambu note).
8. The hydrodynamic gate sigma |H| and sigma ||K||_op along r (gate_common_v1.kop_rows, the code of record's definition) and the dense
   re-solve's diagnostics for the manifest."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FM_DIR = "<internal>/code"
GATE_KITS = os.path.normpath(os.path.join(HERE, "..", "..", "gatecomp", "kits"))


# ------------------------------------------------------------------------------------------------ 1. kernels
def engine_kernels(r, R, ell):
    """engine_fields_v4.kernels (line 37), reproduced: K_0, K_1 of the shell-averaged 3D Gaussian of std ell. Unit-tested against the import."""
    r = np.asarray(r, float)[:, None]
    R = np.asarray(R, float)[None, :]
    Em = np.exp(-(r - R) ** 2 / (2.0 * ell ** 2))
    Ep = np.exp(-(r + R) ** 2 / (2.0 * ell ** 2))
    EA = np.exp(-(r ** 2 + R ** 2) / (2.0 * ell ** 2))
    z = r * R / ell ** 2
    small = z < 1e-2
    zs = np.where(small, 1.0, z)
    i0 = np.where(small, EA * (1.0 + z ** 2 / 6.0), (Em - Ep) / (2.0 * zs))
    i1 = np.where(small, EA * (z / 3.0) * (1.0 + z ** 2 / 10.0), ((zs - 1.0) * Em + (zs + 1.0) * Ep) / (2.0 * zs ** 2))
    pref = 1.0 / ((2.0 * np.pi) ** 1.5 * ell ** 3)
    return pref * i0, pref * i1


def engine_avg_radial(Nf, vf, R, ell, half_width=8.0, h_frac=1.0 / 16.0):
    """engine_fields_v12.site_fields_kernel_average (line 817) on callables of r (in sites): the trapezoid of spacing h_frac ell from 0 to
    max R + half_width ell, each site's band |r - R_n| <= half_width ell. Returns (N_n, v_n, norms)."""
    R = np.asarray(R, float)
    if ell <= 0:
        return np.asarray(Nf(R), float), np.asarray(vf(R), float), np.ones_like(R)
    h = ell * h_frac
    rr = np.arange(0.0, R.max() + half_width * ell + 1e-9, h)
    wtr = np.full(len(rr), h); wtr[0] = 0.5 * h
    Nr = np.asarray(Nf(rr), float); vr = np.asarray(vf(rr), float)
    No, vo, no = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R))
    for n, Rn in enumerate(R):
        lo = max(0, int(np.floor((Rn - half_width * ell) / h))); hi = min(len(rr), int(np.ceil((Rn + half_width * ell) / h)) + 1)
        seg = rr[lo:hi]
        K0, K1 = engine_kernels(seg, np.array([Rn]), ell)
        w = 4.0 * np.pi * seg * seg * wtr[lo:hi]
        No[n] = w @ (K0[:, 0] * Nr[lo:hi]); vo[n] = w @ (K1[:, 0] * vr[lo:hi]); no[n] = w @ K0[:, 0]
    return No, vo, no


def compact_kernel(s, ell):
    """gap10_core_v2.kernel_K (line 90): 9009/(1024 pi ell^3) (1 - s^2/ell^2)_+^5."""
    x = np.clip(1.0 - (np.asarray(s, float) / ell) ** 2, 0.0, None)
    return 9009.0 / (1024.0 * math.pi * ell ** 3) * x ** 5


def clock_f_rel(x_over_R):
    t = np.clip((np.asarray(x_over_R, float) - 0.85) / 0.15, 0, 1)
    return 0.5 * (1 + np.cos(math.pi * t))


def gauss_cube_fields(prof, L, H, d=0.0, ell=None, kernel="gauss", ramp="root", R_ramp=None, pad_ells=8.0):
    """the profile point-sampled on a box padded by pad_ells ell (so no face normalization), the clock scan e^(-d f) on N and v before averaging
    (f relative to R_ramp sites when ramp = 'root', to L when 'L'), the 3D average (kernel 'gauss' = the production Gaussian of std ell,
    'compact' = gap10's kernel_K with support ell, 'cosine' = the v1 form, normalized discretely), cropped to the half-side H. Returns
    N, [v_x, v_y, v_z], r, info."""
    import pt3b_cube as CU
    from scipy.signal import fftconvolve
    ell = 0.0 if ell is None else float(ell)
    if kernel == "gauss":
        m = int(math.ceil(pad_ells * ell)) if ell > 0 else 0
    else:
        m = int(math.ceil(ell)) if ell > 0 else 0
    Hp = H + m
    X, Y, Z, r = CU.grid(Hp)
    x = r / L
    Nd = np.asarray(prof["N"](x), float); vr = np.asarray(prof["v"](x), float)
    if d:
        if ramp == "root":
            f = clock_f_rel(r / float(R_ramp))
        else:
            f = CU.clock_f(x)
        s = np.exp(-d * f); Nd = Nd * s; vr = vr * s
    rr = np.where(r > 0, r, 1.0)
    vd = [vr * c / rr * (r > 0) for c in (X, Y, Z)]
    info = {"kernel": kernel, "ell_sites": ell, "pad": m, "ramp": ramp, "R_ramp": R_ramp}
    if ell > 0:
        k = np.arange(-m, m + 1)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        rho = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        if kernel == "gauss":
            Kn = np.exp(-rho ** 2 / (2 * ell * ell))
        elif kernel == "compact":
            Kn = compact_kernel(rho, ell)
        else:
            Kn = np.where(rho < ell, np.cos(0.5 * math.pi * rho / ell) ** 2, 0.0)
        Kn = Kn / Kn.sum()
        info["kernel_discrete_sum"] = float(Kn.sum())
        info["kernel_second_moment_per_axis"] = float((Kn * KX ** 2).sum())
        if kernel == "gauss":
            av = lambda F: fftconvolve(F, Kn, mode="same")
        else:
            norm = fftconvolve(np.ones_like(Nd), Kn, mode="same")
            av = lambda F: fftconvolve(F, Kn, mode="same") / norm
        Nd = av(Nd); vd = [av(c) for c in vd]
    sl = slice(m, m + 2 * H + 1)
    crop = lambda F: F[sl, sl, sl]
    _, _, _, r0 = CU.grid(H)
    return crop(Nd), [crop(c) for c in vd], r0, info


def outer_root_axis(N, v, r, H):
    import pt3b_cube as CU
    ro = CU.averaged_root(N, v, r, H)
    return (max(ro["axis"]) if ro["axis"] else None), ro


def adjoint_defect(H=12, ell=2.0, seed=3):
    """reviewer B's B8 addition: the discrete weighted inner product <W s, u> = <s, W^dag u> for the cube's averaging (W the source smoothing
    s -> K * s, W^dag the field back-average u -> K * u, unit site weights; the Gaussian is symmetric, so W^dag = W^T exactly when the box
    is padded). Returns the relative defect on random s, u with support inside the unpadded box."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    m = int(math.ceil(8 * ell)); n = 2 * (H + m) + 1
    s = np.zeros((n, n, n)); u = np.zeros((n, n, n))
    sl = slice(m, m + 2 * H + 1)
    s[sl, sl, sl] = rng.standard_normal((2 * H + 1,) * 3); u[sl, sl, sl] = rng.standard_normal((2 * H + 1,) * 3)
    k = np.arange(-m, m + 1)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    Kn = np.exp(-(KX ** 2 + KY ** 2 + KZ ** 2) / (2 * ell * ell)); Kn /= Kn.sum()
    Ws = fftconvolve(s, Kn, mode="same"); Wu = fftconvolve(u, Kn[::-1, ::-1, ::-1], mode="same")
    a = float(np.sum(Ws * u)); b = float(np.sum(s * Wu))
    return abs(a - b) / max(abs(a), 1e-300)


# ------------------------------------------------------------------------------------------------ 3. geometry
def exterior_geometry(splice, onset, box_L, auto=True):
    """reviewer A bpchk 'wrong' 1: the moved splice over (2, 2.5) L runs with onset 2.5 L; any onset needs half-side >= onset + 0.5 L."""
    if not auto:
        return onset, box_L
    if splice == "reviewb_out":
        onset = max(onset, 2.5)
    return onset, max(box_L, onset + 0.5)


# ------------------------------------------------------------------------------------------------ 4. windows
def interior_window_det(H_csr, imask, lo, hi, dense_max=6000, k_guess=64, zero_tol=1e-10, seed=20261022, info=None):
    """pt3b_kernels.interior_window with a fixed start vector (reproducible) and the near-zero cluster printed into info."""
    ii = np.where(imask)[0]
    HII = H_csr[ii][:, ii]
    if len(ii) <= dense_max:
        e, V = np.linalg.eigh(HII.toarray())
        sel = (e > lo) & (e < hi)
        e, V = e[sel], V[:, sel]
    else:
        from scipy.sparse.linalg import eigsh
        rng = np.random.default_rng(seed)
        v0 = rng.standard_normal(len(ii)) + 1j * rng.standard_normal(len(ii)); v0 /= np.linalg.norm(v0)
        c = 0.5 * (lo + hi)
        k = k_guess
        while True:
            e, V = eigsh(HII.tocsc(), k=min(k, len(ii) - 2), sigma=c, which="LM", v0=v0, tol=1e-12)
            if (e.min() < lo and e.max() > hi) or k >= len(ii) - 2:
                break
            k *= 2
        sel = (e > lo) & (e < hi)
        o = np.argsort(e[sel])
        e, V = e[sel][o], V[:, sel][:, o]
    scale = float(abs(HII).sum(axis=1).max()) if HII.nnz else 1.0
    z = np.abs(e) < zero_tol * scale
    if info is not None:
        info["n_zero"] = info.get("n_zero", 0) + int(z.sum())
        info.setdefault("zero_cluster", []).append({"window": [float(lo), float(hi)], "threshold": zero_tol * scale,
                                                    "max_abs_excluded": float(np.abs(e[z]).max()) if z.any() else None,
                                                    "min_abs_kept": float(np.abs(e[~z]).min()) if (~z).any() else None})
    e, V = e[~z], V[:, ~z]
    U = np.zeros((H_csr.shape[0], len(e)), complex)
    U[ii] = V
    return e, U, len(ii)


# ------------------------------------------------------------------------------------------------ 5. poles
def select_pole(sig, dt, e_ref, n_poles=None, rel_amp=1e-3):
    """the matrix-pencil pole nearest the input's own energy e_ref among poles with |c| >= rel_amp max |c| (B5 v1.3's rule)."""
    import pt3b_kernels as KR
    w, c, _ = KR.matrix_pencil(np.asarray(sig, complex), dt, n_poles=n_poles)
    ok = np.abs(c) >= rel_amp * np.abs(c).max()
    idx = np.where(ok)[0]
    j = int(idx[np.argmin(np.abs(w[idx].real - e_ref))])
    return w[j], c[j]


def poles_for_inputs(S_diag, dt, e_in, L, T_obs, n_poles=8, min_T_over_L=100.0):
    """per input: the selected pole and its half-sample error; 'resolved' when T_obs Gamma >= 0.1 or T_obs >= min_T_over_L L, and the sample
    has >= 32 points; 'censored' otherwise (width below the run's resolution)."""
    out = []
    n = S_diag.shape[0]
    for j in range(S_diag.shape[1]):
        row = {"input": j, "e": float(e_in[j])}
        if n < 32:
            row.update({"status": "censored", "why": "fewer than 32 samples"}); out.append(row); continue
        w1, c1 = select_pole(S_diag[:, j], dt, e_in[j], n_poles=min(n_poles, n // 4))
        w2, c2 = select_pole(S_diag[: n // 2, j], dt, e_in[j], n_poles=min(n_poles, n // 8))
        g1 = float(-2 * w1.imag); g2 = float(-2 * w2.imag)
        res = (T_obs * max(g1, 0) >= 0.1) or (T_obs >= min_T_over_L * L)
        row.update({"Re": float(w1.real), "Gamma": g1, "Gamma_half": g2, "abs_err": abs(g1 - g2), "residue": float(abs(c1)),
                    "status": "resolved" if res else "censored"})
        out.append(row)
    return out


def dedupe_poles(rows, tol_abs):
    """distinct poles across inputs (|dRe| and |dGamma| within max(tol_abs, the pair's errors)); counts independent directions only once."""
    pts = [(r["Re"], r["Gamma"], r.get("abs_err", 0.0)) for r in rows if r.get("status") == "resolved"]
    uniq = []
    for p in pts:
        if not any(abs(p[0] - q[0]) <= max(tol_abs, p[2], q[2]) and abs(p[1] - q[1]) <= max(tol_abs, p[2], q[2]) for q in uniq):
            uniq.append(p)
    return uniq


def censored_counts(G_by_T, L, thetas, T_max, pole_rows=None):
    """N_acc(theta): time route 'measured' where T_theta = ln 2 L/theta was propagated, else 'censored' with the lower bound #{p(T_max) > 1/2}
    (valid under exponential escape); pole route counts distinct resolved poles with Gamma L > theta, 'censored' below theta < L/T_max."""
    Ts = sorted(G_by_T)
    out = {}
    for th in thetas:
        Tt = math.log(2.0) * L / th
        hit = [T for T in Ts if abs(T - Tt) <= 1e-6 * Tt]
        if hit:
            p = np.linalg.eigvalsh(0.5 * (G_by_T[hit[0]] + G_by_T[hit[0]].conj().T))
            rec = {"time": {"status": "measured", "count": int(np.sum(p > 0.5))}}
        else:
            p = np.linalg.eigvalsh(0.5 * (G_by_T[Ts[-1]] + G_by_T[Ts[-1]].conj().T))
            rec = {"time": {"status": "censored" if Tt > Ts[-1] else "not_sampled", "lower_bound": int(np.sum(p > 0.5)), "T_needed_over_L": Tt / L}}
        if pole_rows is not None:
            uniq = dedupe_poles(pole_rows, 1e-12)
            if th < L / T_max:
                rec["poles"] = {"status": "censored", "lower_bound": int(sum(1 for q in uniq if q[1] * L > th))}
            else:
                rec["poles"] = {"status": "measured", "count_distinct": int(sum(1 for q in uniq if q[1] * L > th))}
        out["%g" % th] = rec
    return out


# ------------------------------------------------------------------------------------------------ 7. the reader (G4)
def gram_assemble(psi_batches, imask=None):
    """reviewer B bpchk cost note (cross-batch coherence): the escape Gram and the retained Gram of ALL columns at one clock from the states of the
    column batches checkpointed at that clock (list of (d, k_b) arrays, columns in input order): G = 1 - Psi^dag Psi (the dilation),
    R = Psi^dag P_I Psi, including the blocks between different batches."""
    P = np.concatenate(psi_batches, axis=1)
    G = np.eye(P.shape[1]) - P.conj().T @ P
    G = 0.5 * (G + G.conj().T)
    if imask is None:
        return G, None
    Pi = P[imask]
    R = Pi.conj().T @ Pi
    return G, 0.5 * (R + R.conj().T)


def bins_reader(G_list, C0, allow_cross=False, ns=None, splits=None, R_list=None):
    """G_list: [(t_n, G(t_n))] cumulative escape Grams (the dilation's bins t_(n-1) < t <= t_n are orthogonal); C0: the input covariance on the
    propagated span (number-conserving, <c_i^dag c_j> in the inputs' basis). Returns per bin S(b_n), S(R_n), S(b_n | R_(n-1)), the early-late
    mutual information I(R_n : b_(n+1..N)) at every split, the outgoing number, and the final correlation modes."""
    import pt3_core as K3
    C0 = 0.5 * (C0 + C0.conj().T)
    if ns is not None and not allow_cross and ns < C0.shape[0] and np.max(np.abs(C0[:ns, ns:])) > 1e-14:
        raise ValueError("supply-exit coherence in C0: refused (reviewer A bpchk: needs the Nambu form or an explicit allowance)")
    w, V = np.linalg.eigh(C0)
    Ch = (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T
    S_of = lambda G: K3.cov_entropy(Ch @ G @ Ch)
    Gs = [G for _, G in G_list]
    dG = [Gs[0]] + [Gs[i] - Gs[i - 1] for i in range(1, len(Gs))]
    rows = []
    for n, (t, G) in enumerate(G_list):
        R = S_of(G); b = S_of(dG[n])
        row = {"t": t, "S_R": R["S"], "S_R_lo": R["S_lo"], "S_R_hi": R["S_hi"], "S_b": b["S"], "number": float(np.real(np.trace(C0 @ G))),
               "number_bin": float(np.real(np.trace(C0 @ dG[n])))}
        if n:
            row["S_b_given_R"] = R["S"] - rows[-1]["S_R"]
        rows.append(row)
    Gend = Gs[-1]
    mi = []
    for n in (splits if splits is not None else range(len(Gs) - 1)):
        E = Gs[n]; Lt = Gend - Gs[n]
        mi.append({"split_t": G_list[n][0], "I_early_late": S_of(E)["S"] + S_of(Lt)["S"] - S_of(Gend)["S"]})
    ro = []
    if R_list is not None:
        # retained-output correlation: the retained sites and the bath are orthogonal, so the joint Gram is R + G and
        # I(retained : output) = S(C^1/2 R C^1/2) + S(C^1/2 G C^1/2) - S(C^1/2 (R + G) C^1/2)
        for (t, G), R in zip(G_list, R_list):
            ro.append({"t": t, "S_retained": S_of(R)["S"], "I_retained_output": S_of(R)["S"] + S_of(G)["S"] - S_of(R + G)["S"]})
    p, Vp = np.linalg.eigh(0.5 * (Gend + Gend.conj().T))
    mid = (p > 1e-8) & (p < 1 - 1e-8)
    return {"bins": rows, "early_late": mi, "retained_output": ro, "corr_mode_p": [float(x) for x in p[mid]], "corr_mode_vectors": Vp[:, mid]}


def bins_reader_json(res):
    out = dict(res); out.pop("corr_mode_vectors", None)
    return out


# ------------------------------------------------------------------------------------------------ 8. gates and manifest
def kop_rows(R, N, v):
    """busi/code/gatecomp/kits/gate_common_v1.kop_rows (line 147), copied verbatim (the box copy of this directory has no gatecomp tree):
    -> (||K||_op, |H|) per site; sites with N <= 0 or R <= 0 get nan."""
    R = np.asarray(R, float); N = np.asarray(N, float); v = np.asarray(v, float)
    dv = np.gradient(v, R)
    with np.errstate(divide="ignore", invalid="ignore"):
        Kr = np.abs(dv) / N
        Kt = np.abs(v) / (R * N)
    K = np.maximum(Kr, Kt)
    K[(N <= 0) | (R <= 0)] = np.nan
    return K, np.abs(dv)


def hydro_gate(prof, L, sigma_sites, r_max_sites, dr=0.25):
    """sigma |H| and sigma ||K||_op along r from the profile (gate_common_v1.kop_rows: K = max(|v'|/N, |v|/(r N)), |H| = |v'|), the maxima over
    r <= r_max and where they sit."""
    R = np.arange(dr, r_max_sites + dr, dr)
    N = np.asarray(prof["N"](R / L), float); v = np.asarray(prof["v"](R / L), float)
    K, Hh = kop_rows(R, N, v)
    sH = sigma_sites * Hh; sK = sigma_sites * K
    iH = int(np.nanargmax(sH)); iK = int(np.nanargmax(sK))
    return {"sigma_sites": sigma_sites, "max_sigma_H": float(sH[iH]), "r_max_sigma_H_over_L": float(R[iH] / L),
            "max_sigma_K": float(sK[iK]), "r_max_sigma_K_over_L": float(R[iK] / L),
            "profile_r_over_L": [float(x) for x in R[::8] / L], "sigma_H": [float(x) for x in sH[::8]], "sigma_K": [float(x) for x in sK[::8]]}


def dense_record_profile_diag(label, Trs):
    """pt3b_cube.dense_record_profile plus the re-solve's scalar diagnostics (the manifest's re-solve residual) and the engine-kernel lift:
    the engine's own back-average of the re-solved dense fields at the record's sites against the stored applied rows."""
    import pt3b_cube as CU
    CU.sys_path_fix()
    import pt_fieldsA_v1 as FA
    import read_history_v1 as RH
    H = FA.load_record(label, Trs)
    FM = RH.field_module()
    _, _, _, info = FM.solve_constraints_ivp(H["eps_tot"], H["J_tot"], H["g"], H["R_sites"], H["ell"], collar_policy="flag", support_rel=1e-14)
    prof = CU.dense_record_profile(label, Trs)
    scal = {k: (float(v_) if isinstance(v_, (int, float, np.floating, np.integer)) else (v_ if isinstance(v_, (str, bool)) or v_ is None else None))
            for k, v_ in info.items() if not callable(v_) and not isinstance(v_, np.ndarray)}
    scal = {k: v_ for k, v_ in scal.items() if v_ is not None}
    Ne, ve, no = FM.site_fields_kernel_average(info, H["R_sites"], H["ell"])
    Nm, vm, _ = engine_avg_radial(info["N_dense"], info["v_dense"], H["R_sites"], H["ell"])
    Na = H["N_app"][H["k_rec"]]; va = H["v_app"][H["k_rec"]]
    n = min(len(Na), len(Ne))
    lift = {"engine_vs_applied_N_max": float(np.max(np.abs(Ne[:n] - Na[:n]))), "engine_vs_applied_v_max": float(np.max(np.abs(ve[:n] - va[:n]))),
            "engine_vs_site_N_max": float(np.max(np.abs(Ne[:n] - H["N_site"][:n]))), "engine_vs_site_v_max": float(np.max(np.abs(ve[:n] - H["v_site"][:n]))),
            "mine_vs_engine_N_max": float(np.max(np.abs(Nm - Ne))), "mine_vs_engine_v_max": float(np.max(np.abs(vm - ve))),
            "norms_range": [float(no.min()), float(no.max())], "ell_sites": float(H["ell"]), "n_sites": int(n)}
    prof["resolve_info"] = scal
    prof["kernel_lift"] = lift
    prof["sigma_rec_sites"] = float(H["ell"])
    return prof


def interior_window_det2(H_csr, imask, lo, hi, dense_max=6000, k_guess=64, zero_tol=1e-10, seed=20261022, info=None):
    """interior_window_det followed by a Rayleigh-Ritz step on the returned span (found 22 Oct: shift-invert eigsh returns NON-orthogonal
    vectors inside the window's exactly degenerate clusters, 8- to 12-fold at s = 16 F1; the v4 prep's 256 columns were full rank and
    normalized but up to 0.98 off orthogonal). The span is orthonormalized (Loewdin, Gram eigenvalues printed) and h_II re-diagonalized on
    it, so the inputs are orthonormal eigenvectors; the count and the span are unchanged. info['orthonormalization'] records the defect
    before and after."""
    e, U, dI = interior_window_det(H_csr, imask, lo, hi, dense_max=dense_max, k_guess=k_guess, zero_tol=zero_tol, seed=seed, info=info)
    if U.shape[1] == 0:
        return e, U, dI
    ii = np.where(imask)[0]
    V = U[ii]
    Gm = V.conj().T @ V
    before = float(np.max(np.abs(Gm - np.eye(len(Gm)))))
    w, P = np.linalg.eigh(0.5 * (Gm + Gm.conj().T))
    Q = V @ (P / np.sqrt(w)[None, :])
    HII = H_csr[ii][:, ii]
    Hq = Q.conj().T @ (HII @ Q)
    e2, W = np.linalg.eigh(0.5 * (Hq + Hq.conj().T))
    V2 = Q @ W
    after = float(np.max(np.abs(V2.conj().T @ V2 - np.eye(V2.shape[1]))))
    resid = float(np.max(np.linalg.norm(HII @ V2 - V2 * e2[None, :], axis=0)))
    U2 = np.zeros_like(U); U2[ii] = V2
    if info is not None:
        info.setdefault("orthonormalization", []).append({"window": [float(lo), float(hi)], "gram_defect_before": before, "gram_defect_after": after,
                                                         "gram_eig_min": float(w.min()), "ritz_residual_max": resid,
                                                         "max_energy_shift": float(np.max(np.abs(np.sort(e2) - np.sort(e))))})
    return e2, U2, dI
