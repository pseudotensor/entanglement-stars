#!/usr/bin/env python
"""pt3b_cube.py (busi; B', the port and return kernel: ReviewA_pc12_v1.md sec. 3.1, ReviewB_pc12_v1.md sec. 3 Computation A; inbox #450).
The four-component massless cubic operator of the paper, h[N, v] = (1/2){N, h_0} + (1/2) sum_i {v^i, Q_i} at m = 0, with gap10_core_v2's
endpoint means (o5_lift_v1.cubic_apply, ported to torch in pt_companion_bprime_v1.Cube; the same arithmetic here):
  (h u)_x = sum_i alpha_i [n^+_i u_{x+e_i} - n^-_i u_{x-e_i}]/(2i) + sum_i [w^+_i u_{x+2e_i} - w^-_i u_{x-2e_i}]/(4i),
  n^+-_i = (N_x + N_{x+-e_i})/2, w^+-_i = (v^i_x + v^i_{x+-2e_i})/2, open box (amplitudes zero outside).
Added for B':
  * fields on the cube from a radial profile (N(r), u(r) = |v|, v = -u rhat; gap10's point sampling, o5_lift_v1.fields_on_cube), with
    (a) the matched dimensionless profile (reviewer B item 2): the rooted record P64_s4_B2n4096 at T/r_s 100 resampled at r/r_s for any r_s/a,
    (b) the per-size stored records (reviewer A item 1), the vacuum exterior u = (r_s/r)^(1/2) appended beyond the stored domain;
  * the clock scan: reviewer A's Delta ln N = -d inside the root (N only), reviewer B's e^(-d f(r/r_s)) on N and v both (w = v/N kept), f smooth
    (f = 1 inside 0.9 R_t, a C^1 ramp to 0 at R_t); labelled a prescribed-background test;
  * the absorbing potential -i eta(r), eta = eta_0 ((r - R_a)/(R_box - R_a))^2 for r > R_a = 2 r_s (beyond the barrier peak at 1.5 r_s), the
    port's far end, and the non-Hermitian step exp(-i (h - i eta) dt) by the Strang split e^(-eta dt/2) e^(-i h dt) e^(-eta dt/2);
  * a sparse CSR of the same operator for small boxes (shift-invert and dense checks), the radial current / flux operators, the interior
    and shell masks, the flow-coupling operator q_r = (1/2) sum_i {rhat_i, Q_i} (Hellmann-Feynman d h/d u at fixed N);
  * the analytic spectrum of the open box at N = 1, v = 0: |E| = (sum_i cos^2(pi n_i/(L+1)))^(1/2), n_i = 1..L, each twice (both signs)."""
import math

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import jv

SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]], complex); SZ = np.array([[1, 0], [0, -1]], complex)
Z2 = np.zeros((2, 2), complex)
ALPHA = [np.block([[Z2, S], [S, Z2]]) for S in (SX, SY, SZ)]
CD = torch.complex128
RD = torch.float64


def grid(H):
    L = 2 * H + 1
    idx = np.arange(L) - H
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z.astype(float))
    return X.astype(float), Y.astype(float), Z.astype(float), r


def fields_from_profile(H, rp, Np, up, rs_vac):
    """point sampling of a radial profile (rp ascending, sites): N, u interpolated; u regular (linear) below rp[0]; vacuum N = 1,
    u = (rs_vac/r)^(1/2) beyond rp[-1]. Returns N (L,L,L), [v_x, v_y, v_z], r."""
    X, Y, Z, r = grid(H)
    rr = np.where(r > 0, r, 1.0)
    u = np.where(r < rp[0], up[0] * r / rp[0], np.where(r > rp[-1], np.sqrt(rs_vac / rr), np.interp(r, rp, up)))
    N = np.where(r < rp[0], Np[0], np.where(r > rp[-1], 1.0, np.interp(r, rp, Np)))
    f = -np.where(r > 0, u / rr, 0.0)
    return N, [f * X, f * Y, f * Z], r


def matched_profile(rs_a, src=None):
    """the rooted record's applied site fields as functions of r/r_s (nominal), resampled at r_s/a = rs_a: (rp, Np, up, rs_vac, meta)."""
    if src is None:
        import test_pt3_stage1 as T
        rc = T.record()
        H = rc["H"]
        R = np.arange(1, len(H["R_sites"]) + 1, dtype=float)
        src = {"x": R / H["rs_nom"], "N": rc["N_app"], "u": np.abs(rc["v_app"]), "rs_eff_over_nom": 2 * H["g"] * H["E_rec"] / H["rs_nom"],
               "label": "P64_s4_B2n4096 T/r_s 100 applied fields", "T_H_rs_nom": 1.0 / (8 * math.pi * H["g"] * H["E_rec"]) * H["rs_nom"],
               "roots_over_rs": [z / H["rs_nom"] for z in (H["rec"]["trapping_zeros"] or [])]}
    rp = src["x"] * rs_a
    return rp, np.asarray(src["N"], float), np.asarray(src["u"], float), src["rs_eff_over_nom"] * rs_a, {
        "label": src["label"] + " resampled at r_s/a %g" % rs_a, "T_H": src["T_H_rs_nom"] / rs_a, "roots": [z * rs_a for z in src["roots_over_rs"]]}


def clock_scan(N, v, r, R_t, d, mode):
    """mode 'reviewa': N -> N e^(-d) inside the root (a C^1 ramp over [0.9, 1] R_t), v held; 'reviewb': N, v -> e^(-d f) (N, v)."""
    if d == 0:
        return N, v
    x = r / R_t
    f = np.where(x <= 0.9, 1.0, np.where(x >= 1.0, 0.0, 0.5 * (1 + np.cos(math.pi * (x - 0.9) / 0.1))))
    s = np.exp(-d * f)
    if mode == "reviewa":
        return N * s, v
    return N * s, [vi * s for vi in v]


def cap(r, R_a, R_box, eta0):
    return np.where(r > R_a, eta0 * ((r - R_a) / max(R_box - R_a, 1e-9)) ** 2, 0.0)


class Cube:
    """torch operator on (4, L, L, L) amplitudes; eta the absorbing potential (or None)."""

    def __init__(self, N, v, eta=None, device="cpu"):
        dev = torch.device(device)
        self.dev = dev
        T = lambda a: torch.as_tensor(np.asarray(a, float), dtype=RD, device=dev)
        self.N = T(N); self.v = [T(x) for x in v]
        self.A = [torch.as_tensor(A, dtype=CD, device=dev) for A in ALPHA]
        self.np_, self.nm_, self.wp_, self.wm_ = [], [], [], []
        for i in range(3):
            self.np_.append(0.5 * (self.N + self.shift(self.N, i, 1)))
            self.nm_.append(0.5 * (self.N + self.shift(self.N, i, -1)))
            self.wp_.append(0.5 * (self.v[i] + self.shift(self.v[i], i, 2)))
            self.wm_.append(0.5 * (self.v[i] + self.shift(self.v[i], i, -2)))
        self.bound = float(3.0 * self.N.abs().max() + 1.5 * max(float(x.abs().max()) for x in self.v)) * 1.01
        self.eta = None if eta is None else T(eta)

    @staticmethod
    def shift(a, axis, s):
        ax = a.ndim - 3 + axis
        out = torch.zeros_like(a)
        L_ = a.shape[ax]
        src = [slice(None)] * a.ndim; dst = [slice(None)] * a.ndim
        if s > 0:
            src[ax] = slice(s, L_); dst[ax] = slice(0, L_ - s)
        else:
            src[ax] = slice(0, L_ + s); dst[ax] = slice(-s, L_)
        out[tuple(dst)] = a[tuple(src)]
        return out

    def apply(self, psi):
        """psi (..., 4, L, L, L) -> h psi (batch dims allowed in front)."""
        out = torch.zeros_like(psi)
        for i in range(3):
            D = self.np_[i] * self.shift(psi, i, 1) - self.nm_[i] * self.shift(psi, i, -1)
            out += (-0.5j) * torch.einsum("ab,...bxyz->...axyz", self.A[i], D)
            del D
            out += (-0.25j) * (self.wp_[i] * self.shift(psi, i, 2) - self.wm_[i] * self.shift(psi, i, -2))
        return out

    def expm(self, psi, tau, tol=1e-15):
        import page_ops_v6 as OPS
        w = self.bound
        z = w * tau
        K, tail = OPS.cheb_order(z, tol)
        kk = np.arange(K + 1)
        ck = jv(kk, z) * ((-1j) ** kk) * np.where(kk == 0, 1.0, 2.0)
        T0 = psi; T1 = self.apply(psi) / w
        Y = complex(ck[0]) * T0 + complex(ck[1]) * T1
        for k in range(2, K + 1):
            T2 = 2.0 * self.apply(T1) / w - T0
            Y += complex(ck[k]) * T2
            T0, T1 = T1, T2
        return Y, K, tail

    def step_cap(self, psi, dt):
        """exp(-i (h - i eta) dt) psi by the Strang split."""
        if self.eta is None:
            return self.expm(psi, dt)[0]
        d = torch.exp(-0.5 * dt * self.eta)
        y, _, _ = self.expm(psi * d, dt)
        return y * d


def cubic_csr(N, v, eta=None):
    """the same operator as a scipy CSR on the flattened index (a, x, y, z) (C order), for small boxes; minus i eta on the diagonal if given."""
    L = N.shape[0]
    n = L ** 3
    idx = np.arange(n).reshape(L, L, L)
    rows, cols, vals = [], [], []
    for i in range(3):
        for s in (+1, -1):
            sl_src = [slice(None)] * 3; sl_dst = [slice(None)] * 3
            if s > 0:
                sl_dst[i] = slice(0, L - 1); sl_src[i] = slice(1, L)
            else:
                sl_dst[i] = slice(1, L); sl_src[i] = slice(0, L - 1)
            xd = idx[tuple(sl_dst)].ravel(); xs = idx[tuple(sl_src)].ravel()
            nm = 0.5 * (N.ravel()[xd] + N.ravel()[xs])
            coef = (-0.5j) * s * nm                        # alpha_i (s n u_{x+s e_i}) /(2i)
            A = ALPHA[i]
            for a in range(4):
                for b in range(4):
                    if A[a, b] != 0:
                        rows.append(a * n + xd); cols.append(b * n + xs); vals.append(coef * A[a, b])
            sl_src2 = [slice(None)] * 3; sl_dst2 = [slice(None)] * 3
            if s > 0:
                sl_dst2[i] = slice(0, L - 2); sl_src2[i] = slice(2, L)
            else:
                sl_dst2[i] = slice(2, L); sl_src2[i] = slice(0, L - 2)
            xd2 = idx[tuple(sl_dst2)].ravel(); xs2 = idx[tuple(sl_src2)].ravel()
            wm = 0.5 * (v[i].ravel()[xd2] + v[i].ravel()[xs2])
            c2 = (-0.25j) * s * wm
            for a in range(4):
                rows.append(a * n + xd2); cols.append(a * n + xs2); vals.append(c2)
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(4 * n, 4 * n))
    if eta is not None:
        H = (H - 1j * sp.diags(np.tile(eta.ravel(), 4))).tocsr()
    return H


def q_r_csr(N, v_unused, r):
    """q_r = (1/2) sum_i {rhat_i, Q_i}: d h / d u at fixed N for v = -u rhat (minus sign kept: d h/d u = -(1/2) sum {rhat_i, Q_i}); Q_i the two-step
    current with unit flow, endpoint means of rhat_i. Small boxes (CSR)."""
    L = N.shape[0]
    X, Y, Z, rr = grid((L - 1) // 2)
    rh = [np.where(rr > 0, c / np.where(rr > 0, rr, 1), 0.0) for c in (X, Y, Z)]
    return cubic_csr(np.zeros_like(N), [-x for x in rh])


def box_spectrum_analytic(L):
    c = np.cos(math.pi * np.arange(1, L + 1) / (L + 1))
    E = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2).ravel()
    return np.sort(np.concatenate([E, E, -E, -E]))


def masks(r, R):
    m = r < R
    return np.tile(m.ravel(), 4)


# ================================================================================= pc13 re-base (ReviewA_pc13 sec. 1, ReviewB_pc13 sec. 1)
REC_F1 = ("P64_ref_c19r11", 30.0)          # reviewer A F1: the sigma/a 4, r_s/a 64 reference collapse at T/r_s 30 (dense root)
REC_A32 = "<internal>/record_slice.npz"   # reviewer B: the nominal r_s/a 32 record at T = 18 x 32 a (set17)


def dense_record_profile(label, Trs):
    """the record's dense fields re-solved from its sources (engine_fields_v16, the inventory's dense route) as callables of r in units of
    L = 2 g E (the declared exterior Schwarzschild length), with the record's back-averaged site fields beside (the production kernel)."""
    sys_path_fix()
    import pt_fieldsA_v1 as FA
    import read_history_v1 as RH
    H = FA.load_record(label, Trs)
    FM = RH.field_module()
    _, _, _, info = FM.solve_constraints_ivp(H["eps_tot"], H["J_tot"], H["g"], H["R_sites"], H["ell"], collar_policy="flag", support_rel=1e-14)
    L = 2 * H["g"] * float(info["E_collar"])
    Nd, vd = info["N_dense"], info["v_dense"]
    rec_ell = H["ell"]
    return {"N": lambda x: np.asarray(Nd(np.asarray(x) * L), float), "v": lambda x: np.asarray(vd(np.asarray(x) * L), float),
            "L_rec": L, "ell_rec_over_L": rec_ell / L, "roots_over_L": [z / L for z in (H["rec"]["trapping_zeros"] or [])],
            "site_R_over_L": H["R_sites"] / L, "site_N": H["N_app"][H["k_rec"]], "site_v": H["v_app"][H["k_rec"]],
            "label": "%s at T/r_s %g (dense re-solve; L = 2gE = %.4f)" % (label, Trs, L), "npz_md5": H["npz_md5"]}


def a32_profile():
    """reviewer B's baseline: the set17 r_s/a 32 record at T/r_s 18 (gap10 record_slice: the engine's applied, back-averaged site fields; its dense
    fields are not in the slice, so the profile is the site rows, linear between sites)."""
    import json as _json
    Z = np.load(REC_A32); J = _json.load(open(REC_A32.replace(".npz", ".json")))
    L = float(J["slice"]["two_gE_json"])
    r = np.asarray(Z["r"], float); N = np.asarray(Z["N"], float); v = np.asarray(Z["v"], float)
    def vf(x):
        R_ = np.asarray(x, float) * L
        return np.where(R_ <= r[-1], np.interp(R_, r, v), v[-1] * np.sqrt(r[-1] / np.maximum(R_, 1e-300)))
    return {"N": lambda x: np.interp(np.asarray(x) * L, r, N, right=1.0), "v": vf,
            "L_rec": L, "roots_over_L": [z / L for z in J["slice"]["dense_record"]["trapping_zeros"]], "label": "set17_rs32 T/r_s 18 (gap10 record_slice; site rows)",
            "ell_rec_over_L": 4.0 / L, "site_rows": True}


def f2_profile():
    """reviewer A F2: the frozen marginally bound dust ball, N = 1, M = E (r/R*)^3 for r < R* = L/2, M = E outside, w = -(2 G M/r)^(1/2), i.e.
    v = -(L (M/E)/r)^(1/2) in units of L: an exact solution of the rows with J = 0, trapped for 0.354 < r/L < 1."""
    Rs = 0.5
    def v(x):
        x = np.asarray(x, float)
        m = np.where(x < Rs, (x / Rs) ** 3, 1.0)
        return -np.sqrt(m / np.maximum(x, 1e-300)) * (x > 0)
    return {"N": lambda x: np.ones_like(np.asarray(x, float)), "v": v, "L_rec": None, "roots_over_L": [Rs ** 1.5, 1.0],
            "label": "F2 frozen dust ball (R* = L/2)", "ell_rec_over_L": 0.0}


def splice(prof, mode):
    """reviewer B: keep the profile through r = 1.5 L, a C^1 cosine splice over (1.5, 2) L to N = 1, v = -(L/r)^(1/2); 'reviewb_out': over (2, 2.5) L;
    'none': reviewer A's (the record's own exterior to its stored domain; vacuum beyond)."""
    if mode == "none":
        return prof
    a, b = (1.5, 2.0) if mode == "reviewb" else (2.0, 2.5)
    def s(x):
        t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
        return 0.5 * (1 - np.cos(math.pi * t))
    N0, v0 = prof["N"], prof["v"]
    vac_v = lambda x: -np.sqrt(1.0 / np.maximum(np.asarray(x, float), 1e-300))
    out = dict(prof)
    out["N"] = lambda x: (1 - s(x)) * N0(x) + s(x) * 1.0
    out["v"] = lambda x: (1 - s(x)) * v0(x) + s(x) * vac_v(x)
    out["label"] = prof["label"] + "; splice %s" % mode
    return out


def clock_f(x):
    """pc13 (both): f = 1 for r/L < 0.85, 0 for r/L > 1, C^1 cosine between."""
    t = np.clip((np.asarray(x, float) - 0.85) / 0.15, 0, 1)
    return 0.5 * (1 + np.cos(math.pi * t))


def cube_fields(prof, L, H, d=0.0, ell_over_L=1.0 / 8, kernel="cosine"):
    """the dense profile point-sampled on the cube at size L (sites), the clock scan e^(-d f) on N and v (w kept, before averaging), then the
    back-average W_ell^dag: a normalized compact 3D kernel K(rho) = cos^2(pi rho/(2 ell)) for rho < ell (builder's 3D form; the K0 constant and K1
    linear identities printed by kernel_identities), convolved by FFT with the box's own normalization. Returns N, [v_i], r, and the raw fields."""
    from scipy.signal import fftconvolve
    X, Y, Z, r = grid(H)
    x = r / L
    Nd = prof["N"](x); vr = prof["v"](x)            # v radial component (negative inward)
    if d:
        s = np.exp(-d * clock_f(x)); Nd = Nd * s; vr = vr * s
    rr = np.where(r > 0, r, 1.0)
    vd = [vr * c / rr * (r > 0) for c in (X, Y, Z)]
    ell = ell_over_L * L
    if ell <= 0.5:
        return Nd, vd, r, (Nd, vd)
    m = int(math.ceil(ell))
    k = np.arange(-m, m + 1)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    rho = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    Kn = np.where(rho < ell, np.cos(0.5 * math.pi * rho / ell) ** 2, 0.0)
    ones = np.ones_like(Nd)
    norm = fftconvolve(ones, Kn, mode="same")
    av = lambda F: fftconvolve(F, Kn, mode="same") / norm
    return av(Nd), [av(c) for c in vd], r, (Nd, vd)


def kernel_identities(L, H, ell_over_L):
    """K0: the averaged constant is 1 to rounding; K1: a linear field is reproduced inside the box (away from its faces)."""
    X, Y, Z, r = grid(H)
    prof = {"N": lambda x: np.ones_like(x), "v": lambda x: 0 * x}
    N, v, _, _ = cube_fields(prof, L, H, 0.0, ell_over_L)
    from scipy.signal import fftconvolve
    ell = ell_over_L * L; m = int(math.ceil(ell))
    k = np.arange(-m, m + 1)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    rho = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    Kn = np.where(rho < ell, np.cos(0.5 * math.pi * rho / ell) ** 2, 0.0)
    norm = fftconvolve(np.ones_like(X), Kn, mode="same")
    lin = fftconvolve(X, Kn, mode="same") / norm
    inner = (np.abs(X) < H - m) & (np.abs(Y) < H - m) & (np.abs(Z) < H - m)
    return {"K0_max_abs": float(np.max(np.abs(N - 1))), "K1_max_abs_inner": float(np.max(np.abs(lin - X)[inner]))}


def averaged_root(N, v, r, H):
    """the outer root of N - |v_r| of the back-averaged fields along the +x axis (and the diagonal), recomputed after the clock scan (reviewer B)."""
    out = {}
    for lab, idx in {"axis": [(H + j, H, H) for j in range(H + 1)], "diagonal": [(H + j, H + j, H + j) for j in range(H + 1)]}.items():
        rr = np.array([r[i] for i in idx]); g = []
        for (i, j, k_) in idx:
            vv = math.sqrt(v[0][i, j, k_] ** 2 + v[1][i, j, k_] ** 2 + v[2][i, j, k_] ** 2)
            g.append(N[i, j, k_] - vv)
        g = np.array(g)
        sc = np.where(np.sign(g[1:]) != np.sign(g[:-1]))[0]
        roots = [float(rr[i] - g[i] * (rr[i + 1] - rr[i]) / (g[i + 1] - g[i])) for i in sc]
        out[lab] = roots
    return out


def cap_reviewa(r, L, onset=2.0, width=0.5, eta_max=1.0):
    """reviewer A pc13 1.2 item 4: -i eta(r) beyond onset L, rising smoothly (sin^2) to eta_max over width L, eta_max ~ 1/a."""
    t = np.clip((r - onset * L) / (width * L), 0, 1)
    return eta_max * np.sin(0.5 * math.pi * t) ** 2


def sys_path_fix():
    import os as _os, sys as _sys
    here = _os.path.dirname(_os.path.abspath(__file__))
    for p in (here, _os.path.dirname(here), _os.path.join(_os.path.dirname(here), "code_v6")):
        if p not in _sys.path:
            _sys.path.insert(0, p)
