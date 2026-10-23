#!/usr/bin/env python
"""o5_shell_native_v1.py -- RUN_O5_controls_v1 section 6: the native cubic build of section 9's fallback shell (the lane's column: a radial Gaussian
of density width 6 a centred at 20 a with the 5 sigma cut, rest spinor, kappa 1, no carrier), written from its closed form on the cube's sites,
and its check against o5_lift_v3's isometric lift of the chain's own shell export.
The closed form: psi(x) = c_j g(x) at every site x with 0 < |x| inside the chain export's cut, g the radial form f(|x|)/|x| (f the chain export's
declared radial profile, the json's column.profile) times the channel's angular member in the lift's 4-component slots (the constant spinor chi in
1 slot for the s_1/2 member; (z/r, (x +- i y)/r) in 2 slots for the p_1/2 member sigma . rhat chi), c_j the per-shell constant that sets the weight
of shell j = rint(|x|) (the chain's midpoint cell) to the chain's cell weight p_chain[j - 1], its phase the lift's own per-shell phase; the origin
site carries 0 (the chain has no r = 0 site). The slots and the angular form are read off the isometric lift's file, whose written vector must be
reproduced to --tol by one of the closed forms (the fit's per-shell constants are least squares; the residual is relative); the native file is
written from that form with the lift's keys (coords, values, meta_json) so gap10_run_v2_o5.py reads it unchanged. --radial cell writes f(j)/j per
shell instead of f(r_site)/r_site (section 6's C3 variation). Prints the fit residual of each candidate form, the residuals of the alternative
radial forms (f(r), a per-shell constant) for the record, the maximal relative difference between the native vector and the lift's, the lift's
origin weight, and the native shell weights beyond the chain's readout radii against the chain's midpoint rule at t 0. CPU, seconds. No digit here
is the object's.
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import o5_chain_export_v2 as CH2  # noqa: E402  (the declared radial profile; no torch import)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


def shell_fit(G, V, j):
    """Per shell j >= 1 the complex least-squares constant c_j with V ~ c_j G over the shell's sites and slots; returns c per site and the relative residual."""
    c = np.zeros(V.shape[0], dtype=np.complex128)
    for jj in np.unique(j):
        if jj < 1:
            continue
        mk = j == jj
        den = float(np.sum(np.abs(G[mk]) ** 2))
        if den > 0.0:
            c[mk] = np.sum(np.conj(G[mk]) * V[mk]) / den
    model = c[:, None] * G
    nv = float(np.linalg.norm(V))
    return c, (float(np.linalg.norm(V - model)) / nv if nv > 0.0 else float('nan'))


def main(argv=None):
    ap = argparse.ArgumentParser(description="RUN_O5_controls_v1: the native cubic shell from its closed form, checked against the isometric lift of the chain's shell export")
    ap.add_argument('--chain', required=True, help="the chain export's json of the shell (--column_source lane --spinor rest --ak0 0 --sigma 6 --cut 5)")
    ap.add_argument('--lift_src', required=True, help="o5_lift_v3's _src.npz of the same chain json with --lift_convention isometric")
    ap.add_argument('--out', required=True, help="the native _src.npz to write (its json beside)")
    ap.add_argument('--radial', choices=['site', 'cell'], default='site', help="site: f(r_site)/r_site; cell: f(j)/j per shell (section 6's C3 variation)")
    ap.add_argument('--tol', type=float, default=1e-9, help="relative residual below which a closed form reproduces the lift's vector")
    args = ap.parse_args(argv)
    with open(args.chain) as fh:
        J = json.load(fh)
    npz = J['files']['npz'] if os.path.exists(J['files']['npz']) else args.chain[:-5] + '.npz'
    Zc = np.load(npz)
    psi0 = np.asarray(Zc['psi0'])
    p_chain = np.sum(np.abs(psi0) ** 2, 1)
    r_chain = np.asarray(Zc['r'], dtype=float)
    prof = dict(J['column']['profile'])
    u = np.array([complex(*J['column']['spinor'][0]), complex(*J['column']['spinor'][1])])
    if abs(u[1]) > 1e-12 or abs(float(J['column'].get('ak0', 0.0))) > 0.0 or J['column'].get('spinor_mode') != 'rest':
        raise SystemExit("the native shell is the rest-spinor (1, 0) column without a carrier (--spinor rest --ak0 0): this chain json is not it")
    Zl = np.load(args.lift_src)
    coords = np.asarray(Zl['coords'])
    vals = np.asarray(Zl['values'])
    meta = json.loads(str(Zl['meta_json']))
    n = coords.shape[0]
    ax = [k for k, d in enumerate(vals.shape) if d == n]
    if not ax:
        raise SystemExit(f"the lift's values (shape {vals.shape}) have no site axis of length {n}")
    ax = ax[0]
    Vfull = np.moveaxis(vals, ax, 0)
    shape_rest = Vfull.shape[1:]
    V2 = Vfull.reshape(n, -1)
    if V2.shape[1] != 4:
        raise SystemExit(f"the lift's values carry {V2.shape[1]} components per site, 4 expected")
    xyz = coords.astype(float)
    r = np.sqrt(np.sum(xyz ** 2, 1))
    j = np.rint(r).astype(int)
    nz = r > 0.5
    f_site = CH2.radial_profile(r, prof, apply_cut=True)
    f_cell = CH2.radial_profile(j.astype(float), prof, apply_cut=True)
    g_site = np.zeros(n, dtype=np.complex128)
    g_site[nz] = f_site[nz] / r[nz]
    g_cell = np.zeros(n, dtype=np.complex128)
    g_cell[nz] = f_cell[nz] / j[nz]
    g_f = np.where(nz, f_site, 0.0).astype(np.complex128)
    g_one = np.where(nz, 1.0, 0.0).astype(np.complex128)
    xr = np.zeros(n)
    yr = np.zeros(n)
    zr = np.zeros(n)
    xr[nz], yr[nz], zr[nz] = xyz[nz, 0] / r[nz], xyz[nz, 1] / r[nz], xyz[nz, 2] / r[nz]
    origin_w = float(np.sum(np.abs(V2[~nz]) ** 2))
    amp = np.max(np.abs(V2), 0)
    slots = [k for k in range(4) if amp[k] > 1e-12 * float(amp.max())]
    print(f"[lift] {args.lift_src} (md5 {md5(args.lift_src)}): {n} sites, values shape {vals.shape} (site axis {ax}), nonzero slots {slots} (per-slot maxima {['%.3e' % a for a in amp]}), "
          f"weight at the origin site {origin_w:.2e}; lift convention {meta.get('lift_convention')}", flush=True)
    forms = []
    if len(slots) == 1:
        forms.append((f"constant member (s_1/2): chi in slot {slots[0]}", slots, lambda g: np.stack([g], 1)))
    if len(slots) == 2:
        for a, b in ((slots[0], slots[1]), (slots[1], slots[0])):
            for sgn in (+1, -1):
                forms.append((f"sigma.rhat chi member (p_1/2): z/r in slot {a}, (x {'+' if sgn > 0 else '-'} i y)/r in slot {b}", [a, b],
                              (lambda sgn_: (lambda g: np.stack([g * zr, g * (xr + sgn_ * 1j * yr)], 1)))(sgn)))
    if not forms:
        raise SystemExit(f"{len(slots)} nonzero slots: neither 1 (the constant member) nor 2 (the sigma.rhat member); read o5_lift_v1's layout before writing the native file")
    best = None
    for name, sl, make in forms:
        G = make(g_site)
        c, res = shell_fit(G, V2[:, sl], j)
        print(f"[fit] {name}: relative residual {res:.2e} with the radial form f(r_site)/r_site", flush=True)
        if best is None or res < best[1]:
            best = (name, res, sl, make, c)
    name, res, sl, make, c_lift = best
    for rname, g in (("f(r_site), no 1/r", g_f), ("per-shell constant", g_one), ("f(j)/j per cell", g_cell)):
        _, r_alt = shell_fit(make(g), V2[:, sl], j)
        print(f"[fit] {name} with the radial form {rname}: relative residual {r_alt:.2e} (for the record)", flush=True)
    if res > args.tol:
        print(f"[stop] no closed form reproduces the lift's vector to {args.tol:g} (best {name}: {res:.2e}): the lift's 4-component layout is not what the closed forms assume; "
              f"read o5_lift_v1.lift_vector before writing the native file by hand", flush=True)
        return 3
    G = make(g_site if args.radial == 'site' else g_cell)
    native = np.zeros_like(V2)
    phases, shells = [], 0
    for jj in np.unique(j):
        if jj < 1:
            continue
        mk = j == jj
        idx = np.nonzero(mk)[0]
        tgt = float(p_chain[jj - 1]) if jj - 1 < p_chain.size else 0.0
        den = float(np.sum(np.abs(G[mk]) ** 2))
        if tgt <= 0.0 or den <= 0.0:
            continue
        ph = np.exp(1j * np.angle(c_lift[idx[0]])) if abs(c_lift[idx[0]]) > 0.0 else 1.0
        phases.append(float(np.angle(c_lift[idx[0]])))
        native[np.ix_(idx, sl)] = (ph * math.sqrt(tgt / den)) * G[mk]
    W = float(np.sum(np.abs(native) ** 2))
    diff = float(np.max(np.abs(native - V2))) / float(np.max(np.abs(V2)))
    phase_spread = (max(phases) - min(phases)) if phases else 0.0
    t0 = []
    for nm, R in zip(J['readout']['radii_names'], J['readout']['radii']):
        wn = float(np.sum(np.abs(native[r > float(R)]) ** 2))
        wc = float(J['sweeps'][0]['p_mid'][nm])
        t0.append(dict(name=nm, R=float(R), native_strict=wn, chain_mid=wc, difference=wn - wc))
    print(f"[native] closed form: {name}; radial form {'f(r_site)/r_site' if args.radial == 'site' else 'f(j)/j per cell'}; {shells if shells else len(phases)} shells set to the chain's cell weights; "
          f"total weight {W:.12f} (the origin 0; the chain's norm {float(np.sum(p_chain)):.12f}); per-shell phase spread of the lift's constants {phase_spread:.1e}; "
          f"maximal relative difference native minus isometric lift {diff:.2e}", flush=True)
    for c_ in t0:
        print(f"[native t0] beyond {c_['name']} ({c_['R']:.4f} a): native {c_['native_strict']:.9f} against the chain's cell rule {c_['chain_mid']:.9f} (difference {c_['difference']:+.1e})", flush=True)
    out_vals = np.moveaxis(native.reshape((n,) + shape_rest), 0, ax)
    meta2 = dict(meta, native=dict(script='o5_shell_native_v1.py', closed_form=name, radial_form=args.radial, fit_residual_to_isometric_lift=res, max_rel_diff_to_isometric_lift=diff,
                                   slots=sl, shells_weight_matched=len(phases), origin_set_to_0=True, lift_origin_weight=origin_w, total_weight=W, phase_spread=phase_spread,
                                   t0_check=t0, chain_json=args.chain, chain_md5=md5(args.chain), chain_npz_md5=md5(npz), lift_src=args.lift_src, lift_src_md5=md5(args.lift_src),
                                   definition="psi(x) = c_j g(x) chi-member on the cube's sites inside the chain export's cut, shell j = rint(|x|) carrying the chain's cell weight p_chain[j - 1]; "
                                              "no sampled angular function, no lift; the origin site 0"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, coords=coords, values=out_vals, meta_json=json.dumps(meta2))
    with open(args.out[:-4] + '.json', 'w') as fh:
        json.dump(meta2['native'], fh, indent=1)
    print(f"[done] {args.out} (md5 {md5(args.out)}) and its json; the detector file for the run is the lift's {args.lift_src.replace('_src.npz', '_det.npz')}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
