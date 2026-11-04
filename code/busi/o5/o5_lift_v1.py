#!/usr/bin/env python
"""o5_lift_v1.py -- O5 (GAP10_T7_final_v1.1 section 9): P21's lift of the chain column to the cubic lattice.

The lift: the column's radial profile f(r) = Z exp(-(r - R_c)^2/(4 sigma^2)) exp(i k_0 (r - R_c)) evaluated at the site radii
r = |x| times the channel's sampled angular functions, with the 4-component structure of the chain's spinor convention
(o5_chain_export_v1.SPINOR_CONVENTION): Psi(x) = (1/r) [ psi_1(r) Omega_{-kappa, m_j}(xhat) ; -psi_2(r) Omega_{+kappa, m_j}(xhat) ],
(psi_1, psi_2) = f(r) u the chain 2-spinor, Omega the spinor spherical harmonics (Condon-Shortley phase, (sigma.L + 1) Omega_kappa =
-kappa Omega_kappa, sigma.rhat Omega_kappa = -Omega_{-kappa}), renormalized on the cube. The lift's own error is the Gram defect of
the sampled angular functions: per shell j (sites with |x| in [j - 1/2, j + 1/2)) the sampled Gram matrix G_j[a, b] = sum_{x in shell}
Omega_a^dag(xhat) Omega_b(xhat)/|x|^2 over the 2 angular functions (identity in the continuum), its deviation ||G_j - I||_2, and the
column-weighted sum delta_G = sum_j p_j ||G_j - I|| (p_j the chain's shell weights), which bounds the lift's shell-weight error at t 0;
the maximum deviation over the support is the lift error for amplitudes. The reduction check applies the numpy copy of
gap10_core_v2's operator (endpoint means, the record's point-sampled fields) to the lifted column and prints <h> and the energy rms
against the chain's for this convention and its 3 alternatives (lower sign, kappa assignment), so a convention error shows as a large
<h> difference while the lattice difference is small. The chain's 2 detector packets (both carrier signs) are lifted the same way
(the same object in both descriptions) and, for the cube's own axis receiver packets (GAP10's packet_mode form at the +nhat 2 r_s
site with the chain's k_det, both signs, both helicities), their channel projections (the adjoint lift) are written for the chain
side. Writes, in the form gap10_run_v2_o5.py reads: <out>/o5_lift_Rc<Rc>_k<kappa>_H<H>_src.npz (coords (n, 3), values (4, n),
meta_json), ..._det.npz (coords, values (2, 4, n), meta_json, axis_proj, axis_labels) and ..._H<H>.json. Resumable (skipped unless
--force). No plots. CPU, about a minute per H.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import scipy.special as sps

SQRT3 = math.sqrt(3.0)
SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
SIGMA = (SX, SY, SZ)
_Z2 = np.zeros((2, 2), dtype=np.complex128)
ALPHA = [np.block([[_Z2, S], [S, _Z2]]) for S in SIGMA]
BETA_DIAG = np.array([1.0, 1.0, -1.0, -1.0])

CONVENTION_NOTE = ("Psi(x) = (1/r)[psi_1 Omega_a(xhat); -lower_sign psi_2 Omega_b(xhat)] with (Omega_a, Omega_b) = (Omega_{-kappa, m_j}, Omega_{+kappa, m_j}) for "
                   "kappa_assign +1 (the convention derived from the chain's representation: psi_1 = G, psi_2 = -i F) and swapped for -1; lower_sign -1 flips the "
                   "lower pair. The reduction check decides between them on the record's fields.")


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


# ------------------------------------------------------------------ angular functions
def sph_Y(l, m, theta, phi):
    """Y_lm(theta, phi) with the Condon-Shortley phase (Y_11 = -sqrt(3/8pi) sin theta e^{i phi}); zero for |m| > l."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    am = abs(int(m))
    if am > l:
        return np.zeros(np.broadcast(theta, phi).shape, dtype=np.complex128)
    norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) * math.factorial(l - am) / math.factorial(l + am))
    Y = norm * sps.lpmv(am, l, np.cos(theta)) * np.exp(1.0j * am * phi)
    if m < 0:
        Y = ((-1.0) ** am) * np.conj(Y)
    return Y


def spinor_harmonic(kappa, mj, theta, phi):
    """Omega_{kappa, m_j}(theta, phi), (2, n): l = kappa for kappa > 0 (j = l - 1/2), l = -kappa - 1 for kappa < 0 (j = l + 1/2)."""
    kappa = int(kappa)
    if kappa == 0:
        raise ValueError("kappa 0 is not a Dirac channel")
    l = kappa if kappa > 0 else -kappa - 1
    jj = abs(kappa) - 0.5
    if abs(mj) > jj + 1e-9 or abs((mj - 0.5) - round(mj - 0.5)) > 1e-9:
        raise ValueError(f"m_j {mj} is not a member of the channel kappa {kappa} (j = {jj})")
    m_up, m_dn = int(round(mj - 0.5)), int(round(mj + 0.5))
    if kappa < 0:
        cu, cd = math.sqrt((l + mj + 0.5) / (2 * l + 1)), math.sqrt((l - mj + 0.5) / (2 * l + 1))
    else:
        cu, cd = -math.sqrt((l - mj + 0.5) / (2 * l + 1)), math.sqrt((l + mj + 0.5) / (2 * l + 1))
    return np.stack([cu * sph_Y(l, m_up, theta, phi), cd * sph_Y(l, m_dn, theta, phi)], 0)


def angles(coords, r):
    c = np.asarray(coords, dtype=float)
    return np.arccos(np.clip(c[:, 2] / r, -1.0, 1.0)), np.arctan2(c[:, 1], c[:, 0])


def angular_pair(coords, r, kappa_H, mj, kappa_assign=1):
    theta, phi = angles(coords, r)
    Oa = spinor_harmonic(-kappa_H, mj, theta, phi)
    Ob = spinor_harmonic(+kappa_H, mj, theta, phi)
    return (Oa, Ob) if kappa_assign > 0 else (Ob, Oa)


# ------------------------------------------------------------------ sites, lift, Gram
def shell_sites(R_lo, R_hi, H):
    """All lattice sites with R_lo <= |x| <= R_hi inside the box |x_i| <= H, the origin excluded; (coords (n, 3) int, r (n,))."""
    hw = min(int(math.ceil(R_hi)), int(H))
    q = np.arange(-hw, hw + 1, dtype=np.int32)
    X, Y, Z = np.meshgrid(q, q, q, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    del X, Y, Z
    r = np.sqrt(np.sum(coords.astype(float) ** 2, 1))
    keep = (r >= R_lo) & (r <= R_hi) & (r > 0.0)
    return coords[keep], r[keep]


def lift_vector(r, f_r, u, Oa, Ob, lower_sign=1.0):
    vals = np.empty((4, r.size), dtype=np.complex128)
    vals[0:2] = ((f_r * u[0]) / r)[None, :] * Oa
    vals[2:4] = ((-lower_sign * f_r * u[1]) / r)[None, :] * Ob
    return vals


def shell_gram_dev(r, Oa, Ob):
    """Per shell j = round(|x|): the sampled Gram matrix of the 2 angular functions with the weights 1/|x|^2 and its deviation from the identity."""
    j = np.rint(r).astype(int)
    jmax = int(j.max())
    w = 1.0 / (r * r)
    gaa = np.bincount(j, weights=w * np.sum(np.abs(Oa) ** 2, 0), minlength=jmax + 1)
    gbb = np.bincount(j, weights=w * np.sum(np.abs(Ob) ** 2, 0), minlength=jmax + 1)
    cab = w * np.sum(np.conj(Oa) * Ob, 0)
    gab = np.bincount(j, weights=cab.real, minlength=jmax + 1) + 1.0j * np.bincount(j, weights=cab.imag, minlength=jmax + 1)
    cnt = np.bincount(j, minlength=jmax + 1)
    p, q = gaa - 1.0, gbb - 1.0
    half = 0.5 * (p + q)
    rad = np.sqrt((0.5 * (p - q)) ** 2 + np.abs(gab) ** 2)
    dev = np.maximum(np.abs(half + rad), np.abs(half - rad))
    return dict(j=np.arange(jmax + 1), gaa=gaa, gbb=gbb, gab=gab, count=cnt, dev=dev)


def lift_shell_column(Rc, sigma, k0, cut, u, Z_chain, H, kappa_H, mj, lower_sign, kappa_assign, p_chain=None, support_floor=1e-6, pair=None):
    """The lift of one Gaussian shell column (the source or a detector); pair = (Oa, Ob) reuses the angular functions on the same sites."""
    R_lo, R_hi = max(0.0, Rc - cut * sigma), Rc + cut * sigma
    coords, r = shell_sites(R_lo, R_hi, H)
    if coords.shape[0] == 0:
        raise ValueError("empty lift support")
    d = r - Rc
    f_r = Z_chain * np.exp(-d * d / (4.0 * sigma * sigma)) * np.exp(1.0j * k0 * d)
    Oa, Ob = angular_pair(coords, r, kappa_H, mj, kappa_assign) if pair is None else pair
    vals = lift_vector(r, f_r, u, Oa, Ob, lower_sign)
    W = float(np.sum(np.abs(vals) ** 2))
    gram = shell_gram_dev(r, Oa, Ob)
    out = dict(coords=coords, r=r, values=vals / math.sqrt(W), values_raw=vals, W_lift=W, gram=gram, R_lo=R_lo, R_hi=R_hi,
               box_clipped=bool(R_hi > H), n_sites=int(coords.shape[0]), Oa=Oa, Ob=Ob, f_r=f_r)
    if p_chain is not None:
        p = np.asarray(p_chain, dtype=float)            # the chain's shell weights, index j - 1 for the shell j = 1 .. n
        devj = np.ones(p.size + 1)                        # shells without lifted sites count as deviation 1
        k = min(gram['dev'].size, devj.size)
        devj[:k] = gram['dev'][:k]
        out['gram_defect'] = float(np.sum(p * devj[1:p.size + 1]))
        sup = p > support_floor * float(p.max())
        out['gram_dev_max_support'] = float(np.max(devj[1:p.size + 1][sup])) if np.any(sup) else float('nan')
        out['weight_defect_bound_check'] = float(abs(W - 1.0))
    return out


def weight_beyond(coords_r, values, R):
    return float(np.sum(np.abs(values[:, coords_r > R]) ** 2))


# ------------------------------------------------------------------ the record's fields on a sub-box and the numpy copy of the cubic operator
def load_table(path):
    z = np.load(path)
    r = np.asarray(z['r'], dtype=float)
    N = np.asarray(z['N'], dtype=float) if 'N' in z.files else np.ones_like(r)
    u = np.abs(np.asarray(z['v'], dtype=float)) if 'v' in z.files else np.abs(np.asarray(z['u'], dtype=float))
    o = np.argsort(r)
    return r[o], N[o], u[o]


def fields_on_cube(Hc, rt, Nt, ut):
    """gap10_core_v2's 'point' sampling of a 'file' profile: v = -u(r) xhat with u continued regularly below r_t[0] and by the vacuum form beyond r_t[-1]."""
    L = 2 * Hc + 1
    idx = (np.arange(L) - Hc).astype(float)
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing='ij')
    r = np.sqrt(X * X + Y * Y + Z * Z)
    rr = np.where(r > 0.0, r, 1.0)
    rs_file = float(rt[-1] * ut[-1] ** 2)
    u = np.where(r < rt[0], ut[0] * r / rt[0], np.where(r > rt[-1], np.sqrt(rs_file / rr), np.interp(r, rt, ut)))
    N = np.where(r < rt[0], Nt[0], np.where(r > rt[-1], Nt[-1], np.interp(r, rt, Nt)))
    f = -np.where(r > 0.0, u / rr, 0.0)
    return N, [f * X, f * Y, f * Z]


def _shift(a, axis, s):
    """out[.., x, ..] = a[.., x + s, ..], zero where x + s is outside."""
    out = np.roll(a, -s, axis=axis)
    L = a.shape[axis]
    sl = [slice(None)] * a.ndim
    sl[axis] = slice(L - s, L) if s > 0 else slice(0, -s)
    out[tuple(sl)] = 0.0
    return out


def cubic_apply(psi, N, v, m):
    """(h u)_x = m N_x beta u_x + sum_i alpha_i [n^+ u_{x+e_i} - n^- u_{x-e_i}]/(2i) + sum_i [w^+ u_{x+2e_i} - w^- u_{x-2e_i}]/(4i), endpoint means (gap10_core_v2's docstring), open box."""
    out = (m * N)[None] * BETA_DIAG[:, None, None, None] * psi
    for i in range(3):
        Np = 0.5 * (N + _shift(N, i, +1))
        Nm = 0.5 * (N + _shift(N, i, -1))
        D = Np[None] * _shift(psi, i + 1, +1) - Nm[None] * _shift(psi, i + 1, -1)
        out += (-0.5j) * np.einsum('ab,bxyz->axyz', ALPHA[i], D)
        del D
        wp = 0.5 * (v[i] + _shift(v[i], i, +2))
        wm = 0.5 * (v[i] + _shift(v[i], i, -2))
        out += (-0.25j) * (wp[None] * _shift(psi, i + 1, +2) - wm[None] * _shift(psi, i + 1, -2))
    return out


def reduction_check(lifts, rt, Nt, ut, m, Hc):
    """<Psi, h Psi> and the energy rms of each convention's lifted (unit) column on the record's point-sampled fields in the sub-box of half-width Hc."""
    L = 2 * Hc + 1
    Nc, vc = fields_on_cube(Hc, rt, Nt, ut)
    out = {}
    for key, lf in lifts.items():
        psi = np.zeros((4, L, L, L), dtype=np.complex128)
        cc = lf['coords']
        psi[:, cc[:, 0] + Hc, cc[:, 1] + Hc, cc[:, 2] + Hc] = lf['values']
        hp = cubic_apply(psi, Nc, vc, m)
        e = float(np.real(np.vdot(psi.reshape(-1), hp.reshape(-1))))
        e2 = float(np.real(np.vdot(hp.reshape(-1), hp.reshape(-1))))
        out[key] = dict(E_mean=e, E_rms=math.sqrt(max(e2 - e * e, 0.0)))
        del psi, hp
    return out


# ------------------------------------------------------------------ GAP10's axis packets and their channel projection (the adjoint lift)
def direction_vector(d):
    return np.array([1.0, 0.0, 0.0]) if d == '100' else np.ones(3) / SQRT3


def site_on_direction(r, d):
    if d == '100':
        pos = np.array([int(round(r)), 0, 0], dtype=int)
    else:
        n = int(round(r / SQRT3))
        pos = np.array([n, n, n], dtype=int)
    return pos, float(np.linalg.norm(pos))


def helicity_spinors(k, n, m):
    """gap10_core_v2.helicity_spinors: positive-energy eigenvectors of h_0(p = k n) with helicity +1 and -1 (in that order)."""
    n = np.asarray(n, dtype=float)
    p = k * n
    d = np.sin(p)
    eps = math.sqrt(m * m + float(d @ d))
    sn = sum(n[i] * SIGMA[i] for i in range(3))
    evals, evecs = np.linalg.eigh(sn)
    chis = [evecs[:, 1], evecs[:, 0]]
    sd = sum(d[i] * SIGMA[i] for i in range(3))
    out = []
    for chi in chis:
        top = math.sqrt((eps + m) / (2.0 * eps)) * chi
        bottom = (sd @ chi) / math.sqrt(2.0 * eps * (eps + m))
        uvec = np.concatenate([top, bottom]).astype(np.complex128)
        out.append(uvec / np.linalg.norm(uvec))
    return out


def gap10_axis_packet(H, pos, nhat, k, spinor, sigma, cut):
    """gap10_core_v2.packet_mode with restriction 'raw': Z exp(-|x - pos|^2/(4 sigma^2)) exp(i k nhat.(x - pos)) chi on |x - pos| <= cut sigma, clipped to the box."""
    R = cut * sigma
    hw = int(math.ceil(R))
    q = np.arange(-hw, hw + 1)
    X, Y, Z = np.meshgrid(q, q, q, indexing='ij')
    delta = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    d2 = (delta ** 2).sum(1)
    ball = d2 <= R * R
    delta, d2 = delta[ball], d2[ball]
    scalar = np.exp(-d2 / (4.0 * sigma * sigma)) * np.exp(1.0j * (delta @ (k * np.asarray(nhat, dtype=float))))
    coords = delta + np.asarray(pos, dtype=int)
    keep = np.all(np.abs(coords) <= H, 1)
    coords, scalar = coords[keep], scalar[keep]
    scalar = scalar / math.sqrt(float(np.sum(np.abs(scalar) ** 2)))
    return coords, np.asarray(spinor, dtype=np.complex128)[:, None] * scalar[None, :]


def channel_projection(coords, values4, kappa_H, mj, n_chain, lower_sign, kappa_assign):
    """(L^dag Phi)_j = sum_{x in shell j} (1/|x|) [Omega_a^dag Phi_up ; -lower_sign Omega_b^dag Phi_low]: the chain 2-spinor whose overlap <g, psi>
    approximates the cube's <Phi, L psi> (exact up to the Gram defect and the radial variation within a shell)."""
    r = np.linalg.norm(np.asarray(coords, dtype=float), axis=1)
    ok = r > 0.0
    coords, r, V = np.asarray(coords)[ok], r[ok], values4[:, ok]
    Oa, Ob = angular_pair(coords, r, kappa_H, mj, kappa_assign)
    j = np.rint(r).astype(int)
    w = 1.0 / r
    g1 = w * np.sum(np.conj(Oa) * V[0:2], 0)
    g2 = -lower_sign * w * np.sum(np.conj(Ob) * V[2:4], 0)
    g = np.zeros((n_chain, 2), dtype=np.complex128)
    for comp, gc in ((0, g1), (1, g2)):
        re = np.bincount(j, weights=gc.real, minlength=n_chain + 1)
        im = np.bincount(j, weights=gc.imag, minlength=n_chain + 1)
        g[:, comp] = (re + 1.0j * im)[1:n_chain + 1]
    return g


# ------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(description="O5: P21's lift of the chain column and its detectors to the cube of half-width H, in the form gap10_run_v2_o5.py reads")
    ap.add_argument('--chain', required=True, help="the chain export's json (o5_chain_export_v1.py; its npz beside it)")
    ap.add_argument('--H', type=int, required=True, help="the cube's half-width (232 for the 10 r_s window, 103 for the 3 r_s check)")
    ap.add_argument('--mj', type=float, default=0.5, help="the channel member m_j (1/2: 1 member of the 2|kappa|-fold channel)")
    ap.add_argument('--lower_sign', type=float, default=1.0, choices=[1.0, -1.0], help="+1 the derived convention; -1 flips the lower pair (the reduction check decides)")
    ap.add_argument('--kappa_assign', type=int, default=1, choices=[1, -1], help="+1 (upper Omega_{-kappa}, lower Omega_{+kappa}), the derived convention; -1 swaps")
    ap.add_argument('--record', default='', help="the record npz for the reduction check (default: the chain json's record path)")
    ap.add_argument('--no_reduction', action='store_true', help="skip the reduction check (it needs about 10 x the column's sub-box in memory)")
    ap.add_argument('--reduction_margin', type=int, default=3, help="the reduction sub-box extends this many sites beyond the column's support")
    ap.add_argument('--no_axis', action='store_true', help="do not write the channel projections of GAP10's axis packets")
    ap.add_argument('--direction', choices=['100', '111'], default='100', help="the cube run's direction (the axis packets' site)")
    ap.add_argument('--axis_sigma', type=float, default=6.0)
    ap.add_argument('--axis_cut', type=float, default=5.0)
    ap.add_argument('--support_floor', type=float, default=1e-6, help="shells with chain weight below this fraction of the maximum do not enter the maximal Gram deviation")
    ap.add_argument('--out', default='o5_lift')
    ap.add_argument('--tag', default='')
    ap.add_argument('--force', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    with open(args.chain) as fh:
        J = json.load(fh)
    npz_path = J['files']['npz'] if os.path.exists(J['files']['npz']) else args.chain[:-5] + '.npz'
    Zc = np.load(npz_path)
    col, det, rec = J['column'], J['detector'], J['record']
    kappa_H = int(J['chain']['kappa'])
    m = float(J['chain']['m'])
    rs = float(J['readout']['rs'])
    H = int(args.H)
    stem = f"o5_lift_Rc{col['Rc']:g}_k{kappa_H}_H{H}{args.tag}"
    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, stem)
    src_path, det_path, json_path = base + '_src.npz', base + '_det.npz', base + '.json'
    if (not args.force) and all(os.path.exists(p) for p in (src_path, det_path, json_path)):
        print(f"[skip] {base}_src.npz, _det.npz and .json exist (--force to redo)", flush=True)
        return 0
    tw = time.time()
    psi0 = np.asarray(Zc['psi0'])
    n_chain = psi0.shape[0]
    p_chain = np.sum(np.abs(psi0) ** 2, 1)
    u = np.array([complex(*col['spinor'][0]), complex(*col['spinor'][1])])
    print(f"[chain] {args.chain} (md5 {md5(args.chain)}; npz {npz_path} md5 {md5(npz_path)}): kappa={kappa_H}, R_c={col['Rc']:g}, sigma={col['sigma']:g}, a k_0={col['ak0']:+.5f}, "
          f"cut {col['cut']:g}, <h>={col['E_mean']:.6f}, rms {col['E_rms']:.5f}, n={n_chain}; lift to H={H} with m_j={args.mj:g}, lower_sign={args.lower_sign:+g}, kappa_assign={args.kappa_assign:+d}", flush=True)

    # ---- the source column, 4 conventions (the chosen one is written)
    t0 = time.time()
    R_lo, R_hi = max(0.0, col['Rc'] - col['cut'] * col['sigma']), col['Rc'] + col['cut'] * col['sigma']
    coords, r = shell_sites(R_lo, R_hi, H)
    Oa, Ob = angular_pair(coords, r, kappa_H, args.mj, +1)
    lifts = {}
    for ls_ in (1.0, -1.0):
        for ka in (1, -1):
            pair = (Oa, Ob) if ka > 0 else (Ob, Oa)
            lifts[f"lower{ls_:+g}_kappa{ka:+d}"] = lift_shell_column(col['Rc'], col['sigma'], col['ak0'], col['cut'], u, col['Z'], H, kappa_H, args.mj, ls_, ka,
                                                                        p_chain=p_chain, support_floor=args.support_floor, pair=pair)
    key = f"lower{args.lower_sign:+g}_kappa{args.kappa_assign:+d}"
    S = lifts[key]
    rad_names = J['readout']['radii_names']
    radii = J['readout']['radii']
    t0_check = []
    for nm, R in zip(rad_names, radii):
        t0_check.append(dict(name=nm, R=float(R), lift_strict=weight_beyond(S['r'], S['values'], R), chain_mid=float(J['sweeps'][0]['p_mid'][nm]),
                             chain_strict=float(J['sweeps'][0]['p_strict'][nm])))
    print(f"[lift source] {S['n_sites']} sites on {R_lo:.1f} <= r <= {R_hi:.1f} a (box clipped: {S['box_clipped']}); weight before renormalization {S['W_lift']:.6f}; "
          f"Gram defect delta_G = {S['gram_defect']:.3e} (column-weighted ||G_j - I||; the lift's own error for shell weights), maximal shell deviation on the support "
          f"{S['gram_dev_max_support']:.3e} (the lift error for amplitudes); ({time.time() - t0:.1f}s)", flush=True)
    for c_ in t0_check:
        print(f"[lift t0] beyond {c_['name']} ({c_['R']:.4f} a): lifted {c_['lift_strict']:.6f} against the chain's cell rule {c_['chain_mid']:.6f} (strict {c_['chain_strict']:.6f}); "
              f"difference to the cell rule {c_['lift_strict'] - c_['chain_mid']:+.2e} (delta_G {S['gram_defect']:.1e})", flush=True)
    sg = S['gram']
    jshow = [j for j in (1, 2, 3, 5, 10, 14, 20, 30, 44, 58, 87) if j < sg['dev'].size]
    print("[lift gram] shell j: sites, ||G_j - I||: " + ', '.join(f"{j}: {int(sg['count'][j])}, {sg['dev'][j]:.2e}" for j in jshow), flush=True)

    # ---- the reduction check
    red = None
    if not args.no_reduction:
        rec_path = args.record or rec['path']
        if os.path.exists(rec_path):
            t0 = time.time()
            rt, Nt, ut = load_table(rec_path)
            Hc = min(int(math.ceil(R_hi)) + int(args.reduction_margin), H)
            rc = reduction_check(lifts, rt, Nt, ut, m, Hc)
            diffs = {k_: abs(v_['E_mean'] - col['E_mean']) for k_, v_ in rc.items()}
            closest = min(diffs, key=diffs.get)
            red = dict(record=rec_path, record_md5=md5(rec_path), Hc=Hc, L=2 * Hc + 1, chain_E_mean=col['E_mean'], chain_E_rms=col['E_rms'],
                       conventions={k_: dict(v_, abs_diff=diffs[k_], rel_diff=diffs[k_] / max(abs(col['E_mean']), 1e-300)) for k_, v_ in rc.items()},
                       default_key=key, closest_key=closest, default_is_closest=(closest == key), wall=time.time() - t0,
                       note="the numpy copy of gap10_core_v2's operator on the record's point-sampled fields; a convention error shows as an O(1) relative <h> difference, the lattice difference between the 2 descriptions as a small one")
            print(f"[reduction] sub-box half-width {Hc} (L {2 * Hc + 1}), {red['wall']:.1f}s: chain <h> {col['E_mean']:.6f} rms {col['E_rms']:.5f}; cube <Psi, h Psi> per convention: "
                  + '; '.join(f"{k_}: {v_['E_mean']:.6f} (rms {v_['E_rms']:.5f}, |diff| {diffs[k_]:.2e})" for k_, v_ in rc.items())
                  + f"; closest {closest}; the chosen convention {key} is the closest: {closest == key}" + ("" if closest == key else " -- WARNING: rerun the lift with the closest convention's --lower_sign/--kappa_assign"), flush=True)
        else:
            print(f"[reduction] record {rec_path} not found here: skipped (give --record)", flush=True)

    # ---- the detectors (the chain's 2 members, both carrier signs), lifted the same way
    det_lifts = []
    coords_d = None
    for md in det['members']:
        ud = np.array([complex(*md['spinor'][0]), complex(*md['spinor'][1])])
        pd_ = np.sum(np.abs(np.asarray(Zc['det_vecs'])[len(det_lifts)]) ** 2, 1)
        D = lift_shell_column(det['r_det'], det['sigma_det'], md['k'], det['cut'], ud, md['Z'], H, kappa_H, args.mj, args.lower_sign, args.kappa_assign,
                              p_chain=pd_, support_floor=args.support_floor, pair=None if coords_d is None else det_lifts[0]['pair'])
        if coords_d is None:
            coords_d = D['coords']
            D['pair'] = (D['Oa'], D['Ob'])
        else:
            assert np.array_equal(coords_d, D['coords'])
        D['label'] = md['label']
        D['meta'] = dict(label=md['label'], sign=int(md['sign']), k=float(md['k']), r_det=float(det['r_det']), sigma_det=float(det['sigma_det']), cut=float(det['cut']),
                         omega_target=float(det['omega_target']), matched=bool(det['matched']), W_lift=D['W_lift'], gram_defect=D['gram_defect'],
                         gram_dev_max_support=D['gram_dev_max_support'], n_sites=D['n_sites'])
        det_lifts.append(D)
        print(f"[lift detector] {md['label']}: {D['n_sites']} sites, weight before renormalization {D['W_lift']:.6f}, Gram defect {D['gram_defect']:.3e}, maximal shell deviation {D['gram_dev_max_support']:.3e}", flush=True)
    amp_lift_error = float(max([S['gram_dev_max_support']] + [D['gram_dev_max_support'] for D in det_lifts]))

    # ---- GAP10's axis packets at the +nhat 2 r_s site with the chain's k_det: their channel projections for the chain side
    axis = None
    axis_proj, axis_labels = np.zeros((0, n_chain, 2), dtype=np.complex128), []
    if not args.no_axis and det['k_det'] > 0.0:
        nhat = direction_vector(args.direction)
        pos, r_site = site_on_direction(det['r_det'], args.direction)
        if int(np.max(np.abs(pos))) <= H:
            projs = []
            for k in (+float(det['k_det']), -float(det['k_det'])):
                spn = helicity_spinors(k, nhat, m)
                for hel, sp in zip((+1, -1), spn):
                    cc, vv = gap10_axis_packet(H, pos, nhat, k, sp, args.axis_sigma, args.axis_cut)
                    projs.append(channel_projection(cc, vv, kappa_H, args.mj, n_chain, args.lower_sign, args.kappa_assign))
                    axis_labels.append(f"o5axis|k={k:+.5f}|h={'+' if hel > 0 else '-'}|sig={args.axis_sigma:g}")
            axis_proj = np.stack(projs, 0)
            axis = dict(direction=args.direction, site=[int(x) for x in pos], r=r_site, nhat=[float(x) for x in nhat], k=float(det['k_det']), sigma=float(args.axis_sigma),
                        cut=float(args.axis_cut), labels=axis_labels, n_sites=int(cc.shape[0]),
                        projection_norms=[float(np.sum(np.abs(g) ** 2)) for g in projs],
                        note="GAP10's own 3-dimensional receiver packets (packet_mode, raw) at the axis site; their channel projection (the adjoint lift) is the chain's reading of them, not a same-object detector")
            print(f"[axis] 4 packets at site {axis['site']} (r {r_site:.3f}, direction {args.direction}), k=+-{det['k_det']:.5f}, width {args.axis_sigma:g}, cut {args.axis_cut:g}; "
                  f"channel-projection norms {['%.4f' % x for x in axis['projection_norms']]} (the channel's share of each packet)", flush=True)
        else:
            print(f"[axis] the 2 r_s site is outside the box H={H}: no axis packets", flush=True)

    # ---- write
    chain_info = dict(json=args.chain, md5_json=md5(args.chain), npz=npz_path, md5_npz=md5(npz_path), stem=os.path.basename(args.chain)[:-5], kappa=kappa_H, Rc=float(col['Rc']),
                      sigma=float(col['sigma']), ak0=float(col['ak0']), cut=float(col['cut']), E_mean=float(col['E_mean']), E_rms=float(col['E_rms']), spinor=col['spinor'],
                      Z=float(col['Z']), n=int(n_chain), r_h=float(rec['r_h']), rs=rs)
    source_info = dict(n_sites=S['n_sites'], R_lo=S['R_lo'], R_hi=S['R_hi'], box_clipped=S['box_clipped'], W_lift=S['W_lift'], gram_defect=S['gram_defect'],
                       gram_dev_max_support=S['gram_dev_max_support'], t0_check=t0_check)
    src_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=CONVENTION_NOTE, chain=chain_info, source=source_info)
    np.savez_compressed(src_path, coords=S['coords'].astype(np.int32), values=S['values'], meta_json=json.dumps(src_meta, default=_json_default))
    det_meta = dict(H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, chain=chain_info, members=[D['meta'] for D in det_lifts],
                    axis_k=(float(det['k_det']) if axis is not None else 0.0), axis_sigma=float(args.axis_sigma), axis_cut=float(args.axis_cut),
                    axis_direction=(args.direction if axis is not None else None), axis_site=(axis['site'] if axis is not None else None), axis_labels=axis_labels)
    np.savez_compressed(det_path, coords=coords_d.astype(np.int32), values=np.stack([D['values'] for D in det_lifts], 0), meta_json=json.dumps(det_meta, default=_json_default),
                        axis_proj=axis_proj, axis_labels=np.array(axis_labels, dtype=str))
    out = dict(script='o5_lift_v1.py', args=vars(args), chain=chain_info, H=H, mj=args.mj, lower_sign=args.lower_sign, kappa_assign=args.kappa_assign, convention=CONVENTION_NOTE,
               spinor_convention_chain=J.get('spinor_convention'),
               source=dict(source_info, shell_gram=dict(j=sg['j'].tolist(), dev=sg['dev'].tolist(), count=sg['count'].tolist(), gaa=sg['gaa'].tolist(), gbb=sg['gbb'].tolist(),
                                                        gab_abs=np.abs(sg['gab']).tolist()),
                           alternatives={k_: dict(W_lift=v_['W_lift'], gram_defect=v_['gram_defect']) for k_, v_ in lifts.items()}),
               reduction=red,
               detector=dict(r_det=float(det['r_det']), sigma_det=float(det['sigma_det']), cut=float(det['cut']), k_det=float(det['k_det']), matched=bool(det['matched']),
                             omega_target=float(det['omega_target']), n_sites=int(coords_d.shape[0]), members=[D['meta'] for D in det_lifts]),
               amp_lift_error=amp_lift_error, axis=axis,
               files=dict(src_npz=src_path, det_npz=det_path, json=json_path, src_md5=md5(src_path), det_md5=md5(det_path)), wall=time.time() - tw)
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=1, default=_json_default)
    print(f"[done] {src_path} (md5 {out['files']['src_md5']}), {det_path} (md5 {out['files']['det_md5']}), {json_path} ({time.time() - tw:.0f}s)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
