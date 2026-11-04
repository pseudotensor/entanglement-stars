#!/usr/bin/env python
"""gap10_run_v2_wl.py -- GAP10 T7, the wavelength-class test (busi/notes/GAP10_wavelength_test_v1.md): one propagation of
long-wavelength packet sources on the record's held fields with the Killing-energy split of each source printed and stored.
gap10_run_v2.py unchanged beside it; this driver shares its field construction, receivers, Chebyshev sweep, certificate,
region Gram matrices, shares, planes and stored projections (same npz keys and json fields, so gap10_analysis_v2.py reads
these runs as before) and adds, each addition marked "WL:" in the comments:
  WL-1 the 2 source classes on the preparation field (--prep fixed or matched, as gap10_run_v2):
       'in'  the incoming-branch carrier k < 0 with Killing energy E_in = --wl_E_in (1.3 m by default) on the preparation
             field: long wavelength (|ak| 0.05 to 0.2 on the T7 record), energy above the threshold;
       'sub' the outgoing-branch minimum over 0 < ak < pi/4 (--wl_sub_mode min; 'mirror' = +|k_in| as in HORIZON v2's
             sign-split columns): the trapped long-wavelength mode, zero group velocity on the preparation field, energy
             in the gap; both with the positive-band projected spinor structure (gap10_sources_wl.packet_mode_band), width
             --sigma (6 a: of the order of the record's 6.6 a annulus), helicity +.
  WL-2 the Killing-energy split of each source: (a) the semiclassical local-symbol estimate; (b) the computed split by
       the Jackson-KPM density from the Chebyshev moments mu_k = <f|T_k(h/dE)|f> of the source on the run's h, read from
       the source's own receiver column during a Chebyshev sweep of order n_f (the moments' box-exactness bound from the
       face residuals; the resolution printed); (c) the threshold filter F_>(h) = smoothed step at m of width --filter_s
       applied by the same sweep: psi_> = F_>(h) f and psi_< = f - psi_>, their weights the filter split.
  WL-2b the five Killing bins (-inf, -b), (-b, -m), (-m, m), (m, b), (b, inf) with b the band top (WEB40G P19 item 1, inbox #54:
       b3 = sqrt(m^2 + 3) for the cubic operator, b1 = sqrt(m^2 + 1) the chain's, both printed): the unbound class is (m, b3),
       content above b3 is bound (the flow lifts the band top); for these long-wavelength sources the weight above b3 is
       expected below 1e-3, and the printed bins say so or not.
  WL-3 --wl_components: the batch members per class, any of raw, above, below (psi_>, psi_<), and with a second filter at b3,
       band (F_>m - F_>b3: the (m, b3) component) and above_b (F_>b3): the components propagate
       like any source, so the escaped share of the above-threshold part and of the sub-threshold part are read directly
       (and the raw share equals their sum plus the cross term of the region Gram matrix, stored when both are in 1 batch;
       with 1 member per run, from the 3 runs by subtraction).
  WL-4 receivers mirrored to -nhat (--rec_mirror): the incoming source leaves through the core on the far side.
  WL-5 the axis WKB kinematic clock of the incoming source to each region radius (gap10_sources_wl.kinematic_clock) and the
       reach radius of the sub-threshold source, printed and stored.
  WL-6 --H sets the parent half-width directly (232 for the T7 test: the certified window must reach the kinematic arrival
       at 2 r_s, about 6.5 r_s after launch; 1 batch member per run at that size on the 80 GB device).
The core is gap10_core_opt2 (the production core; GAP10_OPT2_BACKEND=compiled on the pod, eager on a CPU); its names are
gap10_core_v2's re-exported, so the module name printed is the core used. --selftest runs the checks named in section 10
of the note on the CPU in about 1 minute (34 PASS lines expected).
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gap10_core_opt2 as C          # noqa: E402  the fused core; every v2 name re-exported
import gap10_sources_wl as W         # noqa: E402


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    if not torch.cuda.is_available():
        return torch.device('cpu')
    best, bestfree = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > bestfree:
            best, bestfree = i, free
    return torch.device(f'cuda:{best}')


def pkt_label(role, r, om, sig, res, hel, tag=''):
    return f"pkt|{role}|r={r:.3f}|w={om:.5g}|sig={sig:g}|res={res}|h={'+' if hel > 0 else '-'}{tag}"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _scalar(fn, r):
    return float(np.asarray(fn(np.array([float(r)])), dtype=float).reshape(-1)[0])


def build_parser():
    ap = argparse.ArgumentParser(description="GAP10 T7 wavelength-class test: long-wavelength sources with the Killing-energy split")
    ap.add_argument('--rs', type=int, default=29)
    ap.add_argument('--H', type=int, default=0, help="parent half-width in a (0 = half_frac*rs + margin as gap10_run_v2)")
    ap.add_argument('--half_frac', type=float, default=3.0)
    ap.add_argument('--margin', type=int, default=16)
    ap.add_argument('--ma', type=float, default=0.2)
    ap.add_argument('--ell', type=float, default=4.0)
    ap.add_argument('--Rb_frac', type=float, default=3.86)
    ap.add_argument('--sampling', choices=['averaged', 'cell', 'point'], default='point')
    ap.add_argument('--profile', choices=['ball', 'file'], default='file')
    ap.add_argument('--profile_file', default='')
    ap.add_argument('--direction', choices=['100', '111'], default='100')
    ap.add_argument('--depth', type=float, default=2.0)
    ap.add_argument('--prep', choices=['fixed', 'matched'], default='matched')
    ap.add_argument('--prep_Rb_frac', type=float, default=0.6)
    ap.add_argument('--prep_ell', type=float, default=4.0)
    ap.add_argument('--wl_classes', nargs='+', default=['in', 'sub'], help="source classes in the batch: in, sub")
    ap.add_argument('--wl_components', nargs='+', default=['raw'], help="per class: raw, above, below (psi_> and psi_< of the filter at m), band, above_b (the filter at b3)")
    ap.add_argument('--wl_E_in', type=float, default=W.DEFAULT_E_IN, help="Killing energy label of the incoming-branch carrier on the preparation field")
    ap.add_argument('--wl_sub_mode', choices=['min', 'mirror'], default='min')
    ap.add_argument('--no_band_project', action='store_true', help="fixed carrier spinor (gap10_run_v2's packets) instead of the positive-band projection")
    ap.add_argument('--filter_s', type=float, default=0.04, help="width of the smoothed threshold step F_> at m (in units of 1/a)")
    ap.add_argument('--filter_tol', type=float, default=1e-10)
    ap.add_argument('--filter_nmax', type=int, default=1200)
    ap.add_argument('--sigma', type=float, default=6.0)
    ap.add_argument('--cut', type=float, default=5.0)
    ap.add_argument('--wide_sigma', type=float, default=12.0)
    ap.add_argument('--wide_radii', type=float, nargs='*', default=[2.0, 3.0])
    ap.add_argument('--rec_freqs', type=float, nargs='*', default=None)
    ap.add_argument('--rec_radii', type=float, nargs='*', default=[1.25, 1.5, 2.0, 3.0])
    ap.add_argument('--no_rec_mirror', action='store_true', help="receivers on +nhat only (default: both +nhat and -nhat)")
    ap.add_argument('--regions', type=float, nargs='*', default=[1.5, 2.0])
    ap.add_argument('--common_half', type=int, default=0)
    ap.add_argument('--tmax_rs', type=float, default=10.0)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--tau_frac', type=float, default=1.0)
    ap.add_argument('--dt_read', type=float, default=0.5)
    ap.add_argument('--plane_stride', type=int, default=2)
    ap.add_argument('--no_proj', action='store_true')
    ap.add_argument('--semi_nsamp', type=int, default=200000)
    ap.add_argument('--out', default='runs')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--selftest', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        return selftest()
    dev = pick_device(args.device)
    rs, m = args.rs, args.ma
    H = int(args.H) if args.H > 0 else int(round(args.half_frac * rs)) + args.margin       # WL-6
    L, c = 2 * H + 1, H
    Rb = args.Rb_frac * rs
    tau = float(args.tau) if args.tau is not None else float(args.tau_frac * rs)
    tmax = args.tmax_rs * rs
    nsweeps = int(round(tmax / tau))
    if args.rec_freqs is None:
        args.rec_freqs = [0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.57, 0.8] if m == 0.0 else [0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    args.rec_freqs = [w for w in args.rec_freqs if w > m]
    nhat = C.direction_vector(args.direction)
    common_half = args.common_half if args.common_half > 0 else None
    comps = '-'.join(args.wl_components)
    stem = (f"rs{rs}_m{m:g}_ell{args.ell:g}_Rb{args.Rb_frac:g}_{args.sampling}_{args.profile}_radial_H{H}"
            f"_{args.direction}_d{args.depth:g}_wl{''.join(args.wl_classes)}_{comps}_cheb{args.tag}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, stem)
    print(f"[run] {stem} on {dev} (core {C.__name__}; backend request {C.BACKEND_REQUEST})", flush=True)
    tw = time.time()

    # ---- the configuration and its fields on the halo grid (as gap10_run_v2)
    prof = C.RadialProfile.ball(rs, Rb) if args.profile == 'ball' else C.RadialProfile.from_file(args.profile_file, rs, Rb)
    if args.profile == 'file':
        print(f"[record] {args.profile_file}: rs_file = {prof.rs_file:.4f} against --rs {rs} (relative difference "
              f"{abs(prof.rs_file - rs) / rs:.3e}); N_is_one={prof.N_is_one}", flush=True)
    Lh, ch = L + 4, c + 2
    v_full, N_full, comp = C.build_site_fields(Lh, ch, args.ell, prof, args.sampling)
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    Nint = None if N_full is None else N_full[2:-2, 2:-2, 2:-2].copy()
    ghost = C.ghost_weights_np(v_full, L, N_full)
    nax = np.arange(0, H + 3)
    ax100_u = -v_full[0][ch + nax, ch, ch]
    ax111_u = -(v_full[0][ch + nax, ch + nax, ch + nax] + v_full[1][ch + nax, ch + nax, ch + nax] + v_full[2][ch + nax, ch + nax, ch + nax]) / C.SQRT3
    ax100_N = None if N_full is None else N_full[ch + nax, ch, ch]
    ax111_N = None if N_full is None else N_full[ch + nax, ch + nax, ch + nax]
    del v_full, N_full
    axis = {'100': C.AxisField(nax.astype(float), ax100_u, ax100_N, label='allocated [100]'),
            '111': C.AxisField(nax * C.SQRT3, ax111_u, ax111_N, label='allocated [111]')}
    axf = axis[args.direction]
    u_fn, N_fn, c_fn = axf.u, axf.N, axf.c
    if args.sampling == 'averaged':
        rg_p, V_p = comp['rg'], comp['V']
    else:
        rg_p = V_p = np.zeros(0)
    if args.prep == 'fixed':
        prof_prep = C.RadialProfile.ball(rs, args.prep_Rb_frac * rs)
        r_n, u_n, N_n, _ = C.axis_allocated(prof_prep, args.prep_ell, 'averaged', args.direction, H + 2)
        prep = C.AxisField(r_n, u_n, N_n, label=f'fixed preparation (ball Rb={args.prep_Rb_frac:g} rs, ell={args.prep_ell:g}, averaged, {args.direction})')
    else:
        prep = axf
    wp, wm = C.bond_weights_np(v)
    lp, lm = (None, None) if Nint is None else C.link_weights_np(Nint)
    del v
    bound = C.spectral_bound(m, wp, wm, Nint)
    dE = 1.01 * bound
    Hop = C.LatticeH(m, wp, wm, dev, ghost_np=ghost, N_np=Nint, lp_np=lp, lm_np=lm)
    del wp, wm, lp, lm, ghost
    crossings = C.crossings_between(c_fn, 0.5, min(3.0 * rs, float(H)))
    if crossings:
        r_h, kappa_h = crossings[-1][0], 1.0 / crossings[-1][1]
    else:
        r_h, kappa_h = float('nan'), float('nan')
    print(f"[fields] L={L} H={H} rs={rs} ma={m} ell={args.ell} Rb={Rb} {args.sampling} {args.profile} radial bound={bound:.5f} dE={dE:.5f} "
          f"N_is_one={Nint is None} ({time.time() - tw:.1f}s)", flush=True)
    print(f"[profile] allocated {args.direction}: r_h={r_h:.4f} kappa_h={kappa_h:.6f}; crossings={crossings}", flush=True)
    for rr in (0.5 * rs, float(rs), 1.5 * rs, 2.0 * rs, 3.0 * rs):
        if rr <= H:
            print(f"[profile] r={rr:7.3f}: allocated u={_scalar(u_fn, rr):.6f} N={_scalar(N_fn, rr):.6f} | preparation u={_scalar(prep.u, rr):.6f}", flush=True)

    # ---- positions: the source site (as gap10_run_v2) and the receivers on +nhat and, WL-4, on -nhat
    r_zero = r_h if math.isfinite(r_h) else float(rs)
    src_pos, src_r = C.site_on_direction(r_zero - args.depth, args.direction)
    assert np.max(np.abs(src_pos)) <= H, "source outside the box"
    u_src, N_src = _scalar(u_fn, src_r), _scalar(N_fn, src_r)
    u_src_prep, N_src_prep = _scalar(prep.u, src_r), _scalar(prep.N, src_r)
    print(f"[source] {args.direction} depth {args.depth:g} below r_h={r_zero:.3f}: site {src_pos.tolist()} r={src_r:.4f} u_run={u_src:.5f} "
          f"N_run={N_src:.5f} u_prep={u_src_prep:.5f} N_prep={N_src_prep:.5f} (mismatch {u_src - u_src_prep:+.3e})", flush=True)
    rec = C.Receivers(L, c, dev)
    rec.add_site(src_pos, 'src', src_r)
    rec_sites = []
    sides = [+1] if args.no_rec_mirror else [+1, -1]
    for side in sides:
        for fr in args.rec_radii:
            pos, r = C.site_on_direction(fr * rs, args.direction)
            pos = pos * side
            if np.max(np.abs(pos)) > H:
                print(f"[warn] receiver at {fr:g} rs (side {side:+d}) is outside the box (H={H}); skipped", flush=True)
                continue
            role = f"rec{fr:g}" + ('' if side > 0 else 'm')
            rec_sites.append(dict(frac=fr, side=side, role=role, pos=pos, r=r, u_run=_scalar(u_fn, r), N_run=_scalar(N_fn, r),
                                  u_prep=_scalar(prep.u, r), N_prep=_scalar(prep.N, r)))
            rec.add_site(pos, role, r)

    def add_packet_set(pos, r, role, freqs, sig, restrictions, hels, u_here, N_here, extra, nvec, tag=''):
        for res in restrictions:
            idx_ref, vals, metas = None, [], []
            for om in freqs:
                try:
                    k, nchg = C.outgoing_root(om, u_here, nvec, m, N_here, check_unique=True)
                except ValueError as e:
                    print(f"[warn] {role} sigma={sig:g}: {e}", flush=True)
                    continue
                spn = C.helicity_spinors(k, nvec, m)
                for hel, sp in zip((+1, -1), spn):
                    if hel not in hels:
                        continue
                    try:
                        idx, values, coords, prob, meta = C.packet_mode(L, c, pos, nvec, k, sp, sig, args.cut, rs, restriction=res, common_half=common_half)
                    except ValueError as e:
                        print(f"[warn] {role} res={res}: {e}", flush=True)
                        continue
                    if idx_ref is None:
                        idx_ref = idx
                    else:
                        assert np.array_equal(idx_ref, idx)
                    meta.update(kind='pkt', role=role, r=float(r), omega=float(om), sigma=float(sig), res=res, hel=int(hel), k=float(k),
                                root_sign_changes=int(nchg), vg=float(C.disp_vg(k, u_here, nvec, m, N_here)[0]), u_prep=float(u_here),
                                N_prep=float(N_here), label=pkt_label(role, r, om, sig, res, hel, tag))
                    meta.update(extra)
                    vals.append(values)
                    metas.append(meta)
            if vals:
                rec.add_packet_group(idx_ref, vals, metas)

    for rsite in rec_sites:
        nvec = nhat * rsite['side']          # the outgoing direction at a mirrored receiver is -nhat
        restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.sigma < rs else [])
        add_packet_set(rsite['pos'], rsite['r'], rsite['role'], args.rec_freqs, args.sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'],
                       dict(frac=rsite['frac'], side=rsite['side'], matched_to=-1), nvec)
        if any(abs(rsite['frac'] - w) < 1e-9 for w in args.wide_radii):
            restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.wide_sigma < rs else [])
            add_packet_set(rsite['pos'], rsite['r'], rsite['role'], [args.wl_E_in], args.wide_sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'],
                           dict(frac=rsite['frac'], side=rsite['side'], matched_to=-1), nvec)

    # ---- WL-1: the wavelength-class sources on the preparation field
    hw = int(math.ceil(args.cut * args.sigma))
    classes = []
    k_in_prep = None
    for cls in args.wl_classes:
        if cls == 'in':
            k, nchg = W.incoming_root(args.wl_E_in, u_src_prep, nhat, m, N_src_prep)
            k_in_prep = k
            E_label = args.wl_E_in
        elif cls == 'sub':
            if args.wl_sub_mode == 'mirror':
                if k_in_prep is None:
                    k_in_prep, _ = W.incoming_root(args.wl_E_in, u_src_prep, nhat, m, N_src_prep)
                k = -k_in_prep
                E_label = float(C.disp_omega(k, u_src_prep, nhat, m, N_src_prep)[0])
            else:
                k, E_label = W.outgoing_minimum(u_src_prep, nhat, m, N_src_prep)
            nchg = 0
        else:
            raise ValueError(cls)
        spn = C.helicity_spinors(k, nhat, m)[0]
        idx, values, coords, prob, pmeta = W.packet_mode_band(L, c, src_pos, nhat, k, spn, args.sigma, args.cut, rs, m,
                                                             common_half=common_half, project=not args.no_band_project)
        bw = C.band_weights(values, coords, src_pos, m, -u_src * nhat, nhat, hw, N_src)
        semi = W.semiclassical_split(src_pos, nhat, k, args.sigma, u_fn, N_fn, m, nsamp=args.semi_nsamp)
        tails = W.exterior_tails(coords, prob, [r_zero, 1.25 * rs, 1.5 * rs, 2.0 * rs])
        E_run = float(C.disp_omega(k, u_src, nhat, m, N_src)[0])
        pmeta.update(kind='pkt', role='src', pos=src_pos.tolist(), r=src_r, omega=float(E_label), sigma=float(args.sigma), res='raw', hel=1,
                     k=float(k), root_sign_changes=int(nchg), vg=float(C.disp_vg(k, u_src_prep, nhat, m, N_src_prep)[0]),
                     vg_run=float(C.disp_vg(k, u_src, nhat, m, N_src)[0]), u_prep=u_src_prep, N_prep=N_src_prep, u_run=u_src, N_run=N_src,
                     prep_mismatch=u_src - u_src_prep, predicted_shift=E_run - E_label, E_carrier_run=E_run, band=bw, wl_class=cls,
                     wavelength=2.0 * math.pi / abs(k), semiclassical=semi, initial_tails=tails)
        classes.append(dict(cls=cls, idx=idx, values=values, meta=pmeta))
        print(f"[wl {cls}] k={k:+.5f} (wavelength {2 * math.pi / abs(k):.1f} a, {2 * math.pi / abs(k) / rs:.2f} rs) E_label={E_label:.4f} on the preparation "
              f"(carrier energy on the run field {E_run:.4f}, vg_run {pmeta['vg_run']:+.3f}); band-projected {pmeta['band_projected']} "
              f"(upper band raw {pmeta['upper_band_raw']:.4f}, projected {pmeta.get('upper_band_projected', float('nan')):.4f}); local symbol at the site: "
              f"upper band {bw['upper_band']:.4f} pos. freq {bw['positive_frequency']:.4f} outgoing pos. freq {bw['outgoing_positive_frequency']:.4f} "
              f"lattice scale {bw['lattice_scale']:.4f}; initial tails beyond r_h {tails[f'{r_zero:.4f}']:.4f}, 1.5rs {tails[f'{1.5 * rs:.4f}']:.1e}, "
              f"2rs {tails[f'{2.0 * rs:.4f}']:.1e}", flush=True)
        print(f"[wl {cls}] semiclassical Killing split (local symbol, {semi['nsamp']} samples): E {semi['E_mean']:.4f} +- {semi['E_std']:.4f}; "
              f"above m {semi['w_above']:.4f}, gap {semi['w_gap']:.4f}, below -m {semi['w_below']:.4f}; above m+{semi['shift']:g} {semi['w_above_shift']:.4f}, "
              f"below m-{semi['shift']:g} {semi['w_below_shift']:.4f}; lattice-scale fraction {semi['lattice_scale']:.4f}", flush=True)
    # the source packets as receivers (their own columns carry the Killing moments)
    idx_ref = classes[0]['idx']
    for cl in classes:
        assert np.array_equal(cl['idx'], idx_ref)
        cl['meta']['label'] = pkt_label('src', src_r, cl['meta']['omega'], args.sigma, 'raw', 1, tag=f"|wl={cl['cls']}")
    rec.add_packet_group(idx_ref, [cl['values'] for cl in classes], [dict(cl['meta'], matched_to=-1) for cl in classes])
    return _run_stage2(args, dev, rs, m, H, L, c, tau, tmax, nsweeps, nhat, common_half, stem, path, tw, Hop, dE, bound, rec, rec_sites,
                       classes, add_packet_set, axf, prep, u_fn, N_fn, crossings, r_h, kappa_h, src_pos, src_r, u_src, N_src, u_src_prep,
                       rg_p, V_p, nax, ax100_u, ax111_u, prof)


def _make_state(cl, L, dev):
    psi = torch.zeros((1, 4, L, L, L), dtype=C.CD, device=dev)
    idx_t = torch.tensor(cl['idx'], dtype=torch.int64, device=dev)
    val_t = torch.tensor(cl['values'], dtype=C.CD, device=dev)
    psi[0].view(4, -1).index_copy_(1, idx_t, val_t)
    return psi


def _norm2(x):
    return float((x.real ** 2 + x.imag ** 2).sum().item())


def _run_stage2(args, dev, rs, m, H, L, c, tau, tmax, nsweeps, nhat, common_half, stem, path, tw, Hop, dE, bound, rec, rec_sites, classes,
                add_packet_set, axf, prep, u_fn, N_fn, crossings, r_h, kappa_h, src_pos, src_r, u_src, N_src, u_src_prep, rg_p, V_p, nax,
                ax100_u, ax111_u, prof):
    r_zero = r_h if math.isfinite(r_h) else float(rs)
    state_gb = 4 * L ** 3 * 16 / 1e9
    # ---- WL-2: the threshold filter and the Killing moments, one class at a time (1 state buffer in the sweep)
    cF, n_f = W.filter_coefficients(dE, m, args.filter_s, tol=args.filter_tol, nmax=args.filter_nmax)
    b1, b3 = W.band_tops(m)
    need_b = any(cp in ('band', 'above_b') for cp in args.wl_components)
    cFb, n_fb = (W.filter_coefficients(dE, b3, args.filter_s, tol=args.filter_tol, nmax=args.filter_nmax) if need_b else (None, 0))
    print(f"[filter] F_> = smoothed step at m={m:g} of width s={args.filter_s:g}: Chebyshev order n_f={n_f} (|c_k| < {args.filter_tol:g} beyond; "
          f"the same sweep yields the Killing moments mu_k, k <= {n_f}); band tops b1={b1:.4f} (chain) b3={b3:.4f} (cubic operator)"
          + (f"; second filter at b3: order {n_fb}" if need_b else ''), flush=True)
    rec.finalize()                                   # the source columns exist from here; re-finalized below after the matched receivers
    B_pred = len(classes) * len(args.wl_components)
    fld = (152.0 * L ** 3 + 104.0 * (L + 4) ** 3 + 24.0 * L ** 3) / 1e9
    need_gb = 5.3 * B_pred * state_gb * ((L + 4.0) / L) ** 3 + fld
    free_gb = (torch.cuda.mem_get_info(dev)[0] / 1e9) if dev.type == 'cuda' else float('nan')
    warn = " -- WARNING: the free memory is below the need; run 1 class and 1 component per process" if (dev.type == 'cuda' and free_gb < need_gb) else ""
    print(f"[memory] 1 state buffer = {B_pred * state_gb:.2f} GB at B={B_pred}, L={L}; peak about 5.3 padded buffers (compiled) plus the operator's fields and the "
          f"geometry ({fld:.1f} GB) = {need_gb:.1f} GB (the eager path needs 6 state buffers instead; the filter sweeps run at B=1); free on {dev}: {free_gb:.1f} GB{warn}", flush=True)
    members, kpm = [], {}
    for cl in classes:
        cls = cl['cls']
        col = rec.labels.index(cl['meta']['label'])
        ts = time.time()
        psi0 = _make_state(cl, L, dev)
        acc, P, R, Nr = C.cheb_sweep(Hop, [psi0], dE, cF, n_f, rec.project)      # psi0 is released inside the sweep
        del psi0
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        Pn, Rn, Nrn = P.cpu().numpy(), R.cpu().numpy(), Nr.cpu().numpy()
        del P, R, Nr
        mu_c = Pn[:, 0, col]
        mu, mu_im = mu_c.real.copy(), float(np.max(np.abs(mu_c.imag)))
        resid = Rn[:, 0].copy()
        bnd = W.moments_box_bound(resid, dE)
        ok = np.flatnonzero(bnd < 1e-8)
        k_exact = int(ok[-1]) if ok.size else 0
        ke = np.arange(0, n_f // 2 + 1)
        even_chk = float(np.max(np.abs(mu[2 * ke] - (2.0 * Nrn[ke, 0] ** 2 - mu[0]))))
        kp = W.kpm_split(mu, dE, m, n_use=max(min(n_f, k_exact), 2))
        w_above = _norm2(acc)
        tmp = _make_state(cl, L, dev)
        tmp.sub_(acc)                                                              # psi_< = f - psi_>
        w_below = _norm2(tmp)
        overlap = float((acc.conj() * tmp).sum().real.item())
        filter_box = float(np.sum(np.abs(cF) * bnd))
        kil = dict(n_f=int(n_f), k_exact=k_exact, moments_imag_max=mu_im, even_moment_check=even_chk, resolution=kp['resolution'],
                   n_moments_used=kp['n_moments'], kpm_norm=kp['norm'], w_above=kp['w_above'], w_gap=kp['w_gap'], w_below=kp['w_below'],
                   w_trans_plus=kp['w_trans_plus'], w_trans_minus=kp['w_trans_minus'], E_mean_moments=kp['E_mean'], E_rms_moments=kp['E_rms'],
                   bins_b3=kp['bins_b3'], bins_b1=kp['bins_b1'], b1=b1, b3=b3, w_trans_b3=kp['w_trans_b3'],
                   filter_s=float(args.filter_s), w_filter_above=w_above, w_filter_below=w_below, filter_overlap=overlap, filter_box_bound=filter_box,
                   sweep_wall=time.time() - ts)
        accb = None
        if need_b:                                     # WL-2b: the second filter at the band top b3 (same sweep form, own moments unused)
            psi0 = _make_state(cl, L, dev)
            accb, Pb, Rb_, Nrb = C.cheb_sweep(Hop, [psi0], dE, cFb, n_fb, rec.project)
            del psi0, Pb, Nrb
            kil.update(n_fb=int(n_fb), w_filter_above_b3=_norm2(accb), filter_b3_box_bound=float(np.sum(np.abs(cFb) * W.moments_box_bound(Rb_.cpu().numpy()[:, 0], dE))))
            del Rb_
        cl['meta']['killing'] = kil
        kpm[cls] = dict(mu=mu, resid=resid, norms=Nrn[:, 0].copy(), bound=bnd, density_E=kp['density_E'], density=kp['density'])
        print(f"[wl {cls}] Killing split by the KPM density ({kil['n_moments_used']} moments, box-exact to k={k_exact} at 1e-8, resolution {kil['resolution']:.4f}; "
              f"even-moment check {even_chk:.1e}, Im mu max {mu_im:.1e}, density norm {kil['kpm_norm']:.6f}): above m {kil['w_above']:.4f}, gap {kil['w_gap']:.4f}, "
              f"below -m {kil['w_below']:.4f}; within +-resolution of +m {kil['w_trans_plus']:.4f}, of -m {kil['w_trans_minus']:.4f}; "
              f"<h> from mu_1 {kil['E_mean_moments']:.5f}, rms {kil['E_rms_moments']:.4f}", flush=True)
        bb = kil['bins_b3']
        print(f"[wl {cls}] five Killing bins (b3={b3:.4f}): below -b3 {bb['below_-b']:.2e}, (-b3,-m) {bb['-b_to_-m']:.4f}, gap {bb['gap']:.4f}, (m,b3) {bb['m_to_b']:.4f}, "
              f"above b3 {bb['above_b']:.2e} (within +-resolution of b3 {kil['w_trans_b3']:.2e}); with the chain's b1={b1:.4f}: (m,b1) {kil['bins_b1']['m_to_b']:.4f}, above b1 {kil['bins_b1']['above_b']:.2e}", flush=True)
        print(f"[wl {cls}] threshold filter (s={args.filter_s:g}): weight of psi_> {w_above:.4f}, of psi_< {w_below:.4f}, overlap <psi_>,psi_<> {overlap:.4f} "
              f"(sum {w_above + w_below + 2 * overlap:.6f}); filter's box bound {filter_box:.1e}; sweep {kil['sweep_wall']:.1f}s"
              + (f"; weight of psi_>b3 {kil['w_filter_above_b3']:.2e}" if need_b else ''), flush=True)
        for comp in args.wl_components:
            if comp == 'raw':
                st = _make_state(cl, L, dev)
            elif comp == 'above':
                st = acc.clone() if len(args.wl_components) > 1 else acc
            elif comp == 'below':
                st = tmp.clone() if len(args.wl_components) > 1 else tmp
            elif comp == 'above_b':
                st = accb.clone() if len(args.wl_components) > 1 else accb
            elif comp == 'band':
                st = acc - accb
            else:
                raise ValueError(comp)
            md = dict(cl['meta'])
            md.update(wl_component=comp, label=cl['meta']['label'] + f"|{comp}", matched_to=-1)
            members.append(dict(cls=cls, comp=comp, state=st, meta=md))
        del acc, tmp, accb
    B = len(members)
    psi = members[0]['state'] if B == 1 else torch.cat([mb['state'] for mb in members], 0)
    for mb in members:
        mb['state'] = None
    src_list = [mb['meta'] for mb in members]
    e0, e2 = Hop.energy(psi)
    e0, e2 = e0.cpu().numpy(), e2.cpu().numpy()
    for b in range(B):
        sd = src_list[b]
        sd['energy_mean'] = float(e0[b])
        sd['energy_std'] = float(math.sqrt(max(e2[b] - e0[b] ** 2, 0.0)))
        sd['norm2'] = float(np.real(np.sum(np.abs(psi[b].cpu().numpy()) ** 2))) if L <= 41 else float('nan')
        print(f"[source {b}] {sd['label']}: class {sd['wl_class']} component {sd['wl_component']} k={sd['k']:+.4f} <h>={e0[b]:.5f} (label {sd['omega']:.4f}, "
              f"carrier on the run field {sd['E_carrier_run']:.4f}), energy std {sd['energy_std']:.4f}, exterior prob (beyond rs) {sd['exterior_probability']:.3f}, "
              f"upper band {sd['band']['upper_band']:.3f}, outgoing pos. freq {sd['band']['outgoing_positive_frequency']:.3f}, lattice scale {sd['band']['lattice_scale']:.3f}", flush=True)
        # WL-5: the kinematic clock of the incoming class and the reach of the sub-threshold one, at the member's actual <h>
        E = float(e0[b])
        radii = [1.5 * rs, 2.0 * rs] + [f * rs for f in args.regions if abs(f - 1.5) > 1e-9 and abs(f - 2.0) > 1e-9]
        kin = dict(E=E)
        if sd['wl_class'] == 'in' or E > m:
            try:
                kin.update(W.kinematic_clock(E, src_r, radii, u_fn, N_fn, nhat, m))
                print(f"[kinematics {b}] E={E:.4f}: inward leg {kin['t_in'] / rs:.2f} rs (turning radius {kin['r_turn']}); "
                      f"arrival on the axis: " + ', '.join(f"{float(k) / rs:.2f} rs at {v / rs:.2f} rs" for k, v in kin['t_total'].items()), flush=True)
            except Exception as e:
                print(f"[kinematics {b}] failed: {e!r}", flush=True)
        if sd['wl_class'] == 'sub' or E < m:
            kin['reach_at_E'] = W.reach_radius(E, u_fn, N_fn, nhat, m, 3.0 * rs)
            kin['reach_at_E_plus_std'] = W.reach_radius(E + sd['energy_std'], u_fn, N_fn, nhat, m, 3.0 * rs)
            print(f"[kinematics {b}] E={E:.4f}: outgoing-branch reach {kin['reach_at_E'] / rs:.2f} rs at <h>, {kin['reach_at_E_plus_std'] / rs:.2f} rs at <h> + std", flush=True)
        sd['kinematics'] = kin
    # energy-matched receiver packets (review G10.6) at every receiver site, both sides, for members above the threshold
    for b, sd in enumerate(src_list):
        if sd['energy_mean'] <= m:
            continue
        om_act = float(sd['energy_mean'])
        for rsite in rec_sites:
            nvec = nhat * rsite['side']
            restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.sigma < rs else [])
            add_packet_set(rsite['pos'], rsite['r'], rsite['role'], [om_act], args.sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'],
                           dict(frac=rsite['frac'], side=rsite['side'], matched_to=b, omega_label=sd['omega']), nvec, tag=f"|m{b}")
            if any(abs(rsite['frac'] - w) < 1e-9 for w in args.wide_radii):
                restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.wide_sigma < rs else [])
                add_packet_set(rsite['pos'], rsite['r'], rsite['role'], [om_act], args.wide_sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'],
                               dict(frac=rsite['frac'], side=rsite['side'], matched_to=b, omega_label=sd['omega']), nvec, tag=f"|m{b}")
    rec.finalize()
    if hasattr(rec, '_opt2_cache'):
        rec._opt2_cache = None                         # the padded index cache of the fused projection is stale after new groups
    geo = C.Geometry(L, c, rs, r_zero, dev, region_fracs=tuple(args.regions))
    assert B == B_pred

    # ---- solver setup and the time loop (gap10_run_v2's Chebyshev path, unchanged)
    n_cheb, tr, Cm, Ik, tail = C.cheb_coefficients(dE, tau, args.dt_read)
    c_end = Cm[-1]
    apps_per_sweep = n_cheb + 1
    print(f"[solver] cheb: {apps_per_sweep} applications per sweep, {nsweeps} sweeps of tau={tau:g}, {apps_per_sweep * nsweeps} total; B={B}, nrec={rec.nrec}; "
          f"truncation tail per sweep {tail:.2e}", flush=True)
    times = [np.array([0.0])]
    A_blocks = [rec.project(psi).cpu().numpy()[None]]
    cert_blocks, certb_blocks, certt_blocks = [np.zeros((1, B))], [np.zeros((1, B))], [np.zeros((1, B))]
    d0 = geo.measure(psi)
    res0 = Hop.boundary_residual(psi).cpu().numpy()
    sweep_t, norms, hist, shares, bnd = [0.0], [d0[0]], [d0[1]], [d0[2]], [d0[3]]
    plane_t, planes_z0, planes_dg, res_exact, energies = [0.0], [d0[4]], [d0[5]], [res0], [e0]
    grams, norm_excess = [geo.gram(psi)], []
    proj_P, proj_R, proj_N = [], [], []
    n0 = d0[0].copy()            # WL-3: the components are not unit vectors; unitarity is read against the initial norms
    eps_bnd = np.zeros(B)
    eps_trn = 0.0
    t = 0.0
    for s in range(nsweeps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts = time.time()
        box = [psi]
        del psi
        psi, P, R, Nr = C.cheb_sweep(Hop, box, dE, c_end, n_cheb, rec.project)
        Pn, Rn, Nrn = P.cpu().numpy(), R.cpu().numpy(), Nr.cpu().numpy()
        del P, R, Nr
        A = np.einsum('jk,kbr->jbr', Cm, Pn)
        cb = eps_bnd[None, :] + Ik @ Rn
        ct = (eps_trn + tail) * np.ones((cb.shape[0], B))
        eps_bnd = cb[-1].copy()
        eps_trn += tail
        norm_excess.append(Nrn.max(0) - 1.0)
        if not args.no_proj:
            proj_P.append(Pn)
            proj_R.append(Rn)
            proj_N.append(Nrn)
        tb = t + tr
        t += tau
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        wall = time.time() - ts
        dd = geo.measure(psi)
        err = float(np.max(np.abs(dd[0] / n0 - 1.0)))
        rex = Hop.boundary_residual(psi).cpu().numpy()
        es, _ = Hop.energy(psi)
        es = es.cpu().numpy()
        G = geo.gram(psi)
        print(f"[sweep {s + 1}/{nsweeps}] t={t:.1f} ({t / rs:.2f} rs) wall={wall:.1f}s ({1e3 * wall / apps_per_sweep:.1f} ms/application at B={B}) "
              f"|norm/norm0-1|={err:.2e} |E-E0|max={np.max(np.abs(es - e0)):.2e} cert(min,max)=({(eps_bnd + eps_trn).min():.2e},{(eps_bnd + eps_trn).max():.2e}) "
              f"exact||B psi||max={rex.max():.2e} shares>{[nm for nm, _ in geo.share_radii]} (batch mean)={np.round(dd[2].mean(0), 4).tolist()} "
              f"boundary-layer max={dd[3].max():.2e} s_max^2 beyond {geo.region_names} (max over the batch diagonal)="
              f"{[f'{float(np.max(np.real(np.diag(G[k])))):.4f}' for k in range(G.shape[0])]}", flush=True)
        if norm_excess:
            print(f"[monitor] Chebyshev vector norm excess max over k: {np.max(norm_excess[-1]):.2e}", flush=True)
        if s == 0:
            print(f"[projection] this run about {wall * nsweeps / 60:.1f} min", flush=True)
        if err > 1e-9:
            raise RuntimeError(f"unitarity lost ({err:.2e}): the spectral interval is too small")
        times.append(tb)
        A_blocks.append(A)
        cert_blocks.append(cb + ct)
        certb_blocks.append(cb)
        certt_blocks.append(ct)
        sweep_t.append(t); norms.append(dd[0]); hist.append(dd[1]); shares.append(dd[2]); bnd.append(dd[3])
        res_exact.append(rex); energies.append(es); grams.append(G)
        if ((s + 1) % args.plane_stride == 0) or (s + 1 == nsweeps):
            plane_t.append(t); planes_z0.append(dd[4]); planes_dg.append(dd[5])

    # ---- outputs: gap10_run_v2's keys and fields, plus the WL arrays and fields
    times = np.concatenate(times)
    meta = dict(vars(args))
    meta.update(dict(stem=stem, mode='radial', solver='cheb', L=L, H=H, c=c, Rb=args.Rb_frac * rs, tau=tau, tmax=tmax, nsweeps=nsweeps, dE=dE, bound=bound,
                     n_cheb=n_cheb, J_taylor=None, nst=None, tail_cheb=tail, tail_taylor=None, r_h=r_h, kappa_h=kappa_h, crossings=crossings,
                     src=src_list, src_pos=src_pos.tolist(), src_r=src_r, u_src=u_src, N_src=N_src, u_src_prep=u_src_prep, nhat=nhat.tolist(),
                     prep_label=prep.label, axis_label=axf.label,
                     rec_sites=[dict(frac=d['frac'], side=d['side'], role=d['role'], pos=d['pos'].tolist(), r=d['r'], u_run=d['u_run'], N_run=d['N_run'], u_prep=d['u_prep']) for d in rec_sites],
                     rec_meta=rec.meta, labels=rec.labels, B=B, nrec=rec.nrec, group_cols=rec.group_cols, share_radii=geo.share_radii,
                     region_names=geo.region_names, napply=Hop.napply, wall_total=time.time() - tw, axpy_inplace=C.AXPY_OK,
                     rs_file=(prof.rs_file if prof is not None else None), core=C.__name__, backend=getattr(Hop, 'backend_used', None),
                     state_gb=state_gb, rec_mirror=not args.no_rec_mirror,
                     wl=dict(classes=args.wl_classes, components=args.wl_components, E_in=args.wl_E_in, sub_mode=args.wl_sub_mode,
                             band_project=not args.no_band_project, filter_s=args.filter_s, n_f=int(n_f), sigma=args.sigma,
                             killing={cls: cl['meta']['killing'] for cls, cl in ((cl['cls'], cl) for cl in classes)})))
    arrays = dict(times=times, A=np.concatenate(A_blocks, 0), cert=np.concatenate(cert_blocks, 0), cert_boundary=np.concatenate(certb_blocks, 0),
                  cert_trunc=np.concatenate(certt_blocks, 0), sweep_t=np.array(sweep_t), norms=np.array(norms), hist=np.array(hist), shares=np.array(shares),
                  bnd=np.array(bnd), res_exact=np.array(res_exact), energy=np.array(energies), gram=np.array(grams), plane_t=np.array(plane_t),
                  plane_z0=np.array(planes_z0), plane_diag=np.array(planes_dg), prof_rg=rg_p, prof_V=V_p,
                  axis_r=axf.r, axis_u=axf.u_nodes, axis_N=(axf.N_nodes if axf.N_nodes is not None else np.zeros(0)),
                  ax100_u=ax100_u, ax111_u=ax111_u, ax100_r=nax.astype(float), ax111_r=nax * C.SQRT3,
                  prep_r=prep.r, prep_u=prep.u_nodes, norm_excess=(np.array(norm_excess) if norm_excess else np.zeros((0, B))),
                  filter_c=cF, kpm_classes=np.array([cl['cls'] for cl in classes]),
                  kpm_mu=np.stack([kpm[cl['cls']]['mu'] for cl in classes], 0), kpm_resid=np.stack([kpm[cl['cls']]['resid'] for cl in classes], 0),
                  kpm_norms=np.stack([kpm[cl['cls']]['norms'] for cl in classes], 0), kpm_bound=np.stack([kpm[cl['cls']]['bound'] for cl in classes], 0),
                  kpm_density_E=kpm[classes[0]['cls']]['density_E'], kpm_density=np.stack([kpm[cl['cls']]['density'] for cl in classes], 0))
    for gi, g in enumerate(rec.groups):
        arrays[f'group_S_{gi}'] = g['S']
    if not args.no_proj:
        arrays['proj_P'] = np.stack(proj_P, 0)
        arrays['proj_R'] = np.stack(proj_R, 0)
        arrays['proj_N'] = np.stack(proj_N, 0)
    np.savez_compressed(path + '.npz', **arrays)
    with open(path + '.json', 'w') as f:
        json.dump(meta, f, indent=1, default=_json_default)
    print(f"[done] {path}  applications={Hop.napply}  wall={time.time() - tw:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ the selftest (CPU, about 1 to 3 minutes)
def _test_profile(path, rs=8.0):
    """A record-like table: N < 1 inside, inflow with a trapped annulus (N - u < 0) between about 2 and 9 a, rs_file = rs."""
    r = np.arange(1.0, 61.0)
    N = 1.0 - 0.6 * np.exp(-(r / 6.0) ** 2)
    Rb = 0.6 * rs
    u = np.where(r < Rb, math.sqrt(rs / Rb) * r / Rb, np.sqrt(rs / np.maximum(r, 1e-9)))
    np.savez(path, r=r, N=N, v=-u)


def selftest():
    import tempfile
    import scipy.linalg as sla
    fails = []

    def say(ok, msg):
        print(('PASS ' if ok else 'FAIL ') + msg, flush=True)
        if not ok:
            fails.append(msg)

    dev = torch.device('cpu')
    tmpd = tempfile.mkdtemp(prefix='gap10_wl_')
    tab = os.path.join(tmpd, 'table.npz')
    _test_profile(tab)
    # ---- (1) the dense check: moments, KPM split and the filter against exact diagonalization on a 9^3 box
    rs, m, H = 8.0, 0.2, 4
    L, c = 2 * H + 1, H
    prof = C.RadialProfile.from_file(tab, rs, 0.6 * rs)
    Lh, ch = L + 4, c + 2
    v_full, N_full, _ = C.build_site_fields(Lh, ch, 2.0, prof, 'point')
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    Nint = N_full[2:-2, 2:-2, 2:-2].copy()
    ghost = C.ghost_weights_np(v_full, L, N_full)
    wp, wm = C.bond_weights_np(v)
    lp, lm = C.link_weights_np(Nint)
    dE = 1.01 * C.spectral_bound(m, wp, wm, Nint)
    Hop = C.LatticeH(m, wp, wm, dev, ghost_np=ghost, N_np=Nint, lp_np=lp, lm_np=lm)
    dim = 4 * L ** 3
    Hd = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(dim):
        e = torch.zeros((1, dim), dtype=C.CD)
        e[0, j] = 1.0
        Hd[:, j] = Hop.apply(e.view(1, 4, L, L, L)).reshape(-1).numpy()
    say(float(np.max(np.abs(Hd - Hd.conj().T))) < 1e-12, "dense operator Hermitian")
    w, V = np.linalg.eigh(Hd)
    say(float(np.max(np.abs(w))) < dE, f"spectrum inside the interval (max |E| {np.max(np.abs(w)):.4f} < dE {dE:.4f})")
    nhat = C.direction_vector('100')
    pos = np.array([2, 0, 0])
    u_s, N_s = float(-v[0][c + 2, c, c]), float(Nint[c + 2, c, c])
    k_in, _ = W.incoming_root(0.26, u_s, nhat, m, N_s)
    spn = C.helicity_spinors(k_in, nhat, m)[0]
    idx, vals, coords, prob, pm = W.packet_mode_band(L, c, pos, nhat, k_in, spn, 0.8, 2.0, rs, m)
    say(pm['upper_band_projected'] > pm['upper_band_raw'] and pm['upper_band_projected'] > 0.8,      # a packet of a dozen sites: the re-masking after the projection keeps it below 1
        f"band projection raises the upper-band weight ({pm['upper_band_raw']:.4f} -> {pm['upper_band_projected']:.4f})")
    say(abs(float(np.sum(np.abs(vals) ** 2)) - 1.0) < 1e-12, "projected packet normalized")
    f = np.zeros((4, L ** 3), dtype=np.complex128)
    f[:, idx] = vals
    f = f.reshape(-1)
    coef = V.conj().T @ f
    wts = np.abs(coef) ** 2
    rec = C.Receivers(L, c, dev)
    pm.update(kind='pkt', role='src', r=2.0, omega=0.26, sigma=0.8, res='raw', hel=1, label='pkt|src|test', matched_to=-1)
    rec.add_packet_group(idx, [vals], [pm])
    rec.finalize()
    cF, n_f = W.filter_coefficients(dE, m, 0.08, tol=1e-10, nmax=600)
    psi = torch.tensor(f.reshape(1, 4, L, L, L), dtype=C.CD)
    acc, P, R, Nr = C.cheb_sweep(Hop, [psi], dE, cF, n_f, rec.project)
    mu = P[:, 0, 0].numpy().real
    mu_ex = np.array([float(np.sum(wts * np.cos(k * np.arccos(np.clip(w / dE, -1, 1))))) for k in range(n_f + 1)])
    say(float(np.max(np.abs(mu - mu_ex))) < 1e-10, f"Chebyshev moments from the source column equal the exact spectral moments ({np.max(np.abs(mu - mu_ex)):.1e}, n_f {n_f})")
    kp = W.kpm_split(mu, dE, m)
    ex = dict(w_above=float(wts[w > m].sum()), w_gap=float(wts[np.abs(w) < m].sum()), w_below=float(wts[w < -m].sum()))
    say(abs(kp['norm'] - 1.0) < 1e-8, f"KPM density normalized ({kp['norm']:.8f})")
    for key, tr in (('w_above', kp['w_trans_plus']), ('w_gap', kp['w_trans_plus'] + kp['w_trans_minus']), ('w_below', kp['w_trans_minus'])):
        say(abs(kp[key] - ex[key]) <= tr + 2e-3, f"KPM {key} {kp[key]:.4f} against exact {ex[key]:.4f} within the transition weight {tr:.4f} (resolution {kp['resolution']:.3f})")
    say(abs(kp['E_mean'] - float(np.sum(wts * w))) < 1e-10, "mean energy from mu_1 equals the exact one")
    Fw = 0.5 * (1.0 + sps_erf((w - m) / (math.sqrt(2.0) * 0.08)))
    f_above = V @ (Fw * coef)
    say(float(np.linalg.norm(acc.numpy().reshape(-1) - f_above)) < 1e-8, f"filtered state psi_> equals F_>(h) f by exact diagonalization ({np.linalg.norm(acc.numpy().reshape(-1) - f_above):.1e})")
    say(abs(_norm2(acc) - float(np.sum(wts * Fw ** 2))) < 1e-8, "filter weight ||F_>(h) f||^2 = <f|F_>^2|f> equals the exact one (F_> is a smoothed step, not a projector)")
    bndv = W.moments_box_bound(R[:, 0].numpy(), dE)
    say(bndv.shape == (n_f + 1,) and bndv[0] == 0.0, "moments' box bound computed")
    # ---- (2) the kinematic clock on the test field
    r_n, u_n, N_n, _ = C.axis_allocated(prof, 2.0, 'point', '100', 20)
    axt = C.AxisField(r_n, u_n, N_n)
    kin = W.kinematic_clock(0.4, 6.0, [10.0, 12.0], axt.u, axt.N, nhat, m)
    say(kin['t_in'] > 0.0 and all(np.isfinite(vv) or not kin['ok'][kk] for kk, vv in kin['t_total'].items()), f"kinematic clock runs ({kin})")
    say(np.isfinite(W.reach_radius(0.15, axt.u, axt.N, nhat, m, 24.0)), "reach radius runs")
    # ---- (3) the end-to-end tiny run with all components: the identities raw = above + below
    out = os.path.join(tmpd, 'runs')
    rc = main(['--profile', 'file', '--profile_file', tab, '--rs', '8', '--H', '12', '--ma', '0.2', '--ell', '2', '--Rb_frac', '0.6', '--sampling', 'point',
               '--direction', '100', '--depth', '2', '--prep', 'matched', '--sigma', '1.5', '--cut', '3', '--wl_classes', 'in', 'sub',
               '--wl_components', 'raw', 'above', 'below', '--filter_s', '0.08', '--tmax_rs', '2', '--tau', '8', '--dt_read', '0.5',
               '--rec_radii', '1.25', '1.5', '--regions', '1.25', '1.5', '--wide_radii', '--out', out, '--device', 'cpu', '--semi_nsamp', '20000'])
    say(rc == 0, "tiny run returned 0")
    import glob
    stems = glob.glob(os.path.join(out, '*.npz'))
    say(len(stems) == 1, "one npz written")
    d = np.load(stems[0])
    with open(stems[0][:-4] + '.json') as fh:
        meta = json.load(fh)
    src = meta['src']
    say(len(src) == 6 and [s_['wl_component'] for s_ in src] == ['raw', 'above', 'below'] * 2, "6 batch members in order")
    for ci, cls in enumerate(('in', 'sub')):
        b0, b1, b2 = 3 * ci, 3 * ci + 1, 3 * ci + 2
        dA = float(np.max(np.abs(d['A'][:, b0] - d['A'][:, b1] - d['A'][:, b2])))
        say(dA < 1e-11, f"[{cls}] receiver amplitudes: raw = above + below at every readout ({dA:.1e})")
        G = d['gram']
        dG = float(np.max(np.abs(G[:, :, b0, b0] - (G[:, :, b1, b1] + G[:, :, b2, b2] + 2.0 * G[:, :, b1, b2].real))))
        say(dG < 1e-11, f"[{cls}] region Gram: raw share = above + below + 2 Re cross at every sweep end ({dG:.1e})")
        kil = meta['wl']['killing'][cls]
        say(abs(kil['w_filter_above'] + kil['w_filter_below'] + 2 * kil['filter_overlap'] - 1.0) < 1e-10, f"[{cls}] filter weights sum to 1")
        say(abs(kil['E_mean_moments'] - src[b0]['energy_mean']) < 1e-9, f"[{cls}] <h> from mu_1 equals the eager energy ({kil['E_mean_moments']:.6f} vs {src[b0]['energy_mean']:.6f})")
        say(abs(kil['kpm_norm'] - 1.0) < 1e-6 and kil['w_above'] + kil['w_gap'] + kil['w_below'] > 0.999, f"[{cls}] KPM split normalized (above {kil['w_above']:.4f} gap {kil['w_gap']:.4f} below {kil['w_below']:.4f})")
        say(abs(_n2(d, b1) - kil['w_filter_above']) < 1e-10, f"[{cls}] member 'above' carries the filter weight")
    dn = float(np.max(np.abs(d['norms'] / d['norms'][0][None, :] - 1.0)))
    say(dn < 1e-9, f"unitarity on the tiny run, relative to the initial norms ({dn:.1e})")
    say(float(np.max(np.abs(d['energy'] - d['energy'][0][None, :]))) < 1e-9, "energy conserved on the tiny run")
    say(all(k in d.files for k in ('kpm_mu', 'kpm_resid', 'kpm_norms', 'kpm_bound', 'filter_c', 'kpm_density', 'kpm_density_E')), "WL arrays stored")
    say(src[0]['kinematics'].get('t_in', 0.0) > 0.0 and 'reach_at_E' in src[3]['kinematics'], "kinematics stored per member")
    print(f"[selftest] {len(fails)} failures" + ('' if not fails else ': ' + '; '.join(fails)), flush=True)
    return 0 if not fails else 1


def _n2(d, b):
    """||psi_b(0)||^2 from the stored norms (the tiny run's first row)."""
    return float(d['norms'][0, b])


def sps_erf(x):
    import scipy.special as sps
    return sps.erf(x)


if __name__ == '__main__':
    sys.exit(main())
