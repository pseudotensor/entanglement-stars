#!/usr/bin/env python
"""gap10_run.py -- one propagation: all sources at one position (a batch) against all receivers, with the in-sweep readout,
the parent certificate (boundary and truncation parts, rigorous in exact arithmetic), the region Gram matrices of the
transmissivity at every sweep end, the stored polynomial projections for exact re-evaluation at any time, and the diagnostics.
Repaired from csr_run.py of GAP10 (review R2 to R5): allocated axis fields for every root and ray, cell-only sampling, the
record profile with N weights, depth 0 (the site nearest the trapping zero), energy-matched receiver packets (eq. G10.6),
common clipping for parent comparisons, the preparation mismatch and actual energies printed."""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import gap10_core as C


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


def main():
    ap = argparse.ArgumentParser(description="retarded cross-surface response: one source position (GAP10, repaired)")
    ap.add_argument('--rs', type=int, default=32)
    ap.add_argument('--half_frac', type=float, default=3.0, help="half-width H = half_frac*rs + margin")
    ap.add_argument('--margin', type=int, default=16)
    ap.add_argument('--ma', type=float, default=0.0)
    ap.add_argument('--ell', type=float, default=4.0)
    ap.add_argument('--Rb_frac', type=float, default=0.6)
    ap.add_argument('--sampling', choices=['averaged', 'cell', 'point'], default='averaged',
                    help="averaged = K_ell then cell average; cell = cell average only (the ell -> 0 end); point = site values")
    ap.add_argument('--profile', choices=['ball', 'file'], default='ball')
    ap.add_argument('--profile_file', default='', help="radial table (r, N, v or u) of a solved record, lattice units")
    ap.add_argument('--mode', choices=['radial', 'flat', 'uniform'], default='radial')
    ap.add_argument('--uniform_v', type=float, nargs=3, default=[0.0, 0.0, -0.5])
    ap.add_argument('--direction', choices=['100', '111'], default='100')
    ap.add_argument('--depth', type=float, default=4.0, help="source at the site nearest r_h - depth; 0 = the site nearest the trapping zero")
    ap.add_argument('--source', choices=['site', 'packet', 'both'], default='both')
    ap.add_argument('--src_freqs', type=float, nargs='*', default=None)
    ap.add_argument('--rec_freqs', type=float, nargs='*', default=None)
    ap.add_argument('--rec_radii', type=float, nargs='*', default=[1.25, 1.5, 2.0, 3.0])
    ap.add_argument('--regions', type=float, nargs='*', default=[1.5, 2.0], help="R_det/rs of the transmissivity regions (r_h always)")
    ap.add_argument('--sigma', type=float, default=6.0, help="packet width: standard deviation of |f|^2, in a")
    ap.add_argument('--cut', type=float, default=5.0, help="support radius in units of sigma")
    ap.add_argument('--wide_sigma', type=float, default=12.0)
    ap.add_argument('--wide_radii', type=float, nargs='*', default=[2.0, 3.0])
    ap.add_argument('--spins', type=int, default=1, help="helicity states for the packet sources (1 or 2)")
    ap.add_argument('--inside_twins_max_depth', type=float, default=4.0, help="inside-restricted twin source packets at depths <= this (-1 disables)")
    ap.add_argument('--prep', choices=['fixed', 'matched'], default='fixed',
                    help="packet carriers from the fixed preparation configuration (prep_Rb_frac, prep_ell, averaged, allocated on the direction) or the run's own field")
    ap.add_argument('--prep_Rb_frac', type=float, default=0.6)
    ap.add_argument('--prep_ell', type=float, default=4.0)
    ap.add_argument('--common_half', type=int, default=0, help="clip every packet to |x| <= common_half (0 = box only): common vectors across parents")
    ap.add_argument('--tmax_rs', type=float, default=20.0)
    ap.add_argument('--tau', type=float, default=None, help="sweep length (default tau_frac*rs)")
    ap.add_argument('--tau_frac', type=float, default=1.0)
    ap.add_argument('--dt_read', type=float, default=0.5)
    ap.add_argument('--solver', choices=['cheb', 'taylor'], default='cheb')
    ap.add_argument('--taylor_dt', type=float, default=1.0)
    ap.add_argument('--fft_check_sweeps', type=int, nargs='*', default=[])
    ap.add_argument('--plane_stride', type=int, default=4, help="store the 2 planes every this many sweeps (and at 0 and the end)")
    ap.add_argument('--no_proj', action='store_true', help="do not store the polynomial projections (fine re-evaluation then unavailable)")
    ap.add_argument('--out', default='runs')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    dev = pick_device(args.device)
    rs, m = args.rs, args.ma
    H = int(round(args.half_frac * rs)) + args.margin
    L, c = 2 * H + 1, H
    Rb = args.Rb_frac * rs
    tau = float(args.tau) if args.tau is not None else float(args.tau_frac * rs)
    tmax = args.tmax_rs * rs
    nsweeps = int(round(tmax / tau))
    if args.src_freqs is None:
        args.src_freqs = [0.05, 0.1, 0.2] if m == 0.0 else [0.25, 0.3, 0.4]
    if args.rec_freqs is None:
        args.rec_freqs = [0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.57, 0.8] if m == 0.0 else [0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    args.src_freqs = [w for w in args.src_freqs if w > m]
    args.rec_freqs = [w for w in args.rec_freqs if w > m]
    nhat = C.direction_vector(args.direction)
    radial = (args.mode == 'radial')
    common_half = args.common_half if args.common_half > 0 else None
    stem = (f"rs{rs}_m{m:g}_ell{args.ell:g}_Rb{args.Rb_frac:g}_{args.sampling}_{args.profile}_{args.mode}_H{H}"
            f"_{args.direction}_d{args.depth:g}_{args.source}_{args.solver}{args.tag}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, stem)
    print(f"[run] {stem} on {dev}", flush=True)
    tw = time.time()

    # ---- the configuration and its fields on the halo grid (L+4)^3 (interior L^3, 2 ghost planes per face)
    prof = None
    if radial:
        prof = C.RadialProfile.ball(rs, Rb) if args.profile == 'ball' else C.RadialProfile.from_file(args.profile_file, rs, Rb)
        if args.profile == 'file':
            print(f"[record] {args.profile_file}: rs_file = {prof.rs_file:.4f} against --rs {rs} (relative difference "
                  f"{abs(prof.rs_file - rs) / rs:.3e}); N_is_one={prof.N_is_one}", flush=True)
    Lh, ch = L + 4, c + 2
    v_full, N_full, comp = C.build_site_fields(Lh, ch, args.ell, prof, args.sampling, mode=args.mode, uniform_v=args.uniform_v)
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
    if radial:
        u_fn, N_fn, c_fn = axf.u, axf.N, axf.c
        if args.sampling == 'averaged':
            rg_p, V_p = comp['rg'], comp['V']
            u_rad = lambda r: np.interp(np.asarray(r, dtype=float), rg_p, V_p)
            N_rad = (lambda r: np.interp(np.asarray(r, dtype=float), rg_p, comp['Nl'])) if 'Nl' in comp else (lambda r: np.ones_like(np.asarray(r, dtype=float)))
        else:
            rg_p = V_p = np.zeros(0)
            u_rad, N_rad = prof.u, prof.N
    else:
        rg_p = V_p = np.zeros(0)
        u_fn = u_rad = lambda r: np.zeros_like(np.asarray(r, dtype=float))
        N_fn = N_rad = lambda r: np.ones_like(np.asarray(r, dtype=float))
        c_fn = lambda r: np.ones_like(np.asarray(r, dtype=float))
    if radial and args.prep == 'fixed':
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
    crossings = C.crossings_between(c_fn, 0.5, min(3.0 * rs, float(H))) if radial else []
    crossings_rad = C.crossings_between(lambda r: N_rad(r) - u_rad(r), 0.5, min(3.0 * rs, float(H))) if radial else []
    if crossings:
        r_h, kappa_h = crossings[-1][0], 1.0 / crossings[-1][1]
    else:
        r_h, kappa_h = float('nan'), float('nan')
    print(f"[fields] L={L} H={H} rs={rs} ma={m} ell={args.ell} Rb={Rb} {args.sampling} {args.profile} {args.mode} "
          f"bound={bound:.5f} dE={dE:.5f} N_is_one={Nint is None} ({time.time() - tw:.1f}s)", flush=True)
    print(f"[profile] allocated {args.direction}: r_h={r_h:.4f} kappa_h={kappa_h:.6f} (1/(2 rs)={1 / (2 * rs):.6f}); "
          f"all N-|v|=0 crossings (radius, inverse slope)={crossings}; radial K comparator crossings={crossings_rad}", flush=True)
    if radial:
        for rr in (0.5 * rs, Rb, 0.9 * rs, float(rs), 1.25 * rs, 1.5 * rs, 2.0 * rs, 3.0 * rs):
            if rr <= H:
                print(f"[profile] r={rr:7.3f}: allocated u={_scalar(u_fn, rr):.6f} N={_scalar(N_fn, rr):.6f} | radial comparator "
                      f"u={_scalar(u_rad, rr):.6f} | fixed preparation u={_scalar(prep.u, rr):.6f}", flush=True)

    # ---- positions
    r_zero = r_h if math.isfinite(r_h) else float(rs)
    src_pos, src_r = C.site_on_direction(r_zero - args.depth, args.direction)
    assert np.max(np.abs(src_pos)) <= H, "source outside the box"
    u_src, N_src = _scalar(u_fn, src_r), _scalar(N_fn, src_r)
    u_src_prep, N_src_prep = _scalar(prep.u, src_r), _scalar(prep.N, src_r)
    print(f"[source] {args.direction} depth {args.depth:g} below r_h={r_zero:.3f}: site {src_pos.tolist()} r={src_r:.4f} "
          f"u_run={u_src:.5f} N_run={N_src:.5f} u_prep={u_src_prep:.5f} (mismatch {u_src - u_src_prep:+.3e})", flush=True)
    rec = C.Receivers(L, c, dev)
    rec.add_site(src_pos, 'src', src_r)
    rec_sites = []
    for fr in args.rec_radii:
        pos, r = C.site_on_direction(fr * rs, args.direction)
        if np.max(np.abs(pos)) > H:
            print(f"[warn] receiver at {fr:g} rs is outside the box (H={H}); skipped", flush=True)
            continue
        rec_sites.append(dict(frac=fr, pos=pos, r=r, u_run=_scalar(u_fn, r), N_run=_scalar(N_fn, r), u_prep=_scalar(prep.u, r), N_prep=_scalar(prep.N, r)))
        rec.add_site(pos, f"rec{fr:g}", r)

    def add_packet_set(pos, r, role, freqs, sig, restrictions, hels, u_here, N_here, extra, tag=''):
        for res in restrictions:
            idx_ref, vals, metas = None, [], []
            for om in freqs:
                try:
                    k, nchg = C.outgoing_root(om, u_here, nhat, m, N_here, check_unique=True)
                except ValueError as e:
                    print(f"[warn] {role} sigma={sig:g}: {e}", flush=True)
                    continue
                spn = C.helicity_spinors(k, nhat, m)
                for hel, sp in zip((+1, -1), spn):
                    if hel not in hels:
                        continue
                    try:
                        idx, values, coords, prob, meta = C.packet_mode(L, c, pos, nhat, k, sp, sig, args.cut, rs, restriction=res, common_half=common_half)
                    except ValueError as e:
                        print(f"[warn] {role} res={res}: {e}", flush=True)
                        continue
                    if idx_ref is None:
                        idx_ref = idx
                    else:
                        assert np.array_equal(idx_ref, idx)
                    meta.update(kind='pkt', role=role, r=float(r), omega=float(om), sigma=float(sig), res=res, hel=int(hel),
                                k=float(k), root_sign_changes=int(nchg), vg=float(C.disp_vg(k, u_here, nhat, m, N_here)[0]),
                                u_prep=float(u_here), N_prep=float(N_here), label=pkt_label(role, r, om, sig, res, hel, tag))
                    meta.update(extra)
                    vals.append(values)
                    metas.append(meta)
            if vals:
                rec.add_packet_group(idx_ref, vals, metas)

    if radial:
        for rsite in rec_sites:
            role = f"rec{rsite['frac']:g}"
            restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.sigma < rs else [])
            add_packet_set(rsite['pos'], rsite['r'], role, args.rec_freqs, args.sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'], dict(frac=rsite['frac'], matched_to=-1))
            if any(abs(rsite['frac'] - w) < 1e-9 for w in args.wide_radii):
                restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.wide_sigma < rs else [])
                add_packet_set(rsite['pos'], rsite['r'], role, args.src_freqs, args.wide_sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'], dict(frac=rsite['frac'], matched_to=-1))

    # ---- sources
    src_list, init_ops = [], []
    if args.source in ('site', 'both'):
        for comp in range(4):
            src_list.append(dict(kind='site', role='src', comp=comp, pos=src_pos.tolist(), r=src_r, u_run=u_src, N_run=N_src, res='raw',
                                 exterior_probability=0.0, retained=1.0, label=f"site|src|r={src_r:.3f}|c={comp}"))
            init_ops.append(('site', comp))
    if args.source in ('packet', 'both') and radial:
        twins = args.inside_twins_max_depth >= 0 and args.depth <= args.inside_twins_max_depth + 1e-9
        restr = ['raw'] + (['in'] if twins else [])
        hw = int(math.ceil(args.cut * args.sigma))
        for res in restr:
            idx_ref, vals, metas = None, [], []
            for om in args.src_freqs:
                try:
                    k, nchg = C.outgoing_root(om, u_src_prep, nhat, m, N_src_prep, check_unique=True)
                except ValueError as e:
                    print(f"[warn] source packet a w={om:g}: {e}", flush=True)
                    continue
                spn = C.helicity_spinors(k, nhat, m)
                for hel, sp in list(zip((+1, -1), spn))[:args.spins]:
                    idx, values, coords, prob, meta = C.packet_mode(L, c, src_pos, nhat, k, sp, args.sigma, args.cut, rs, restriction=res, common_half=common_half)
                    bw = C.band_weights(values, coords, src_pos, m, -u_src * nhat, nhat, hw, N_src)
                    meta.update(kind='pkt', role='src', pos=src_pos.tolist(), r=src_r, omega=float(om), sigma=float(args.sigma), res=res,
                                hel=int(hel), k=float(k), root_sign_changes=int(nchg), vg=float(C.disp_vg(k, u_src_prep, nhat, m, N_src_prep)[0]),
                                u_prep=u_src_prep, u_run=u_src, N_run=N_src, prep_mismatch=u_src - u_src_prep,
                                predicted_shift=C.carrier_shift(k, nhat, u_src - u_src_prep), band=bw,
                                label=pkt_label('src', src_r, om, args.sigma, res, hel))
                    src_list.append(meta)
                    init_ops.append(('packet', idx, values))
                    vals.append(values)
                    metas.append(dict(meta, matched_to=-1))
                    if idx_ref is None:
                        idx_ref = idx
            if vals:
                rec.add_packet_group(idx_ref, vals, metas)
    B = len(src_list)
    assert B > 0, "no sources"
    psi = torch.zeros((B, 4, L, L, L), dtype=C.CD, device=dev)
    for b, op in enumerate(init_ops):
        if op[0] == 'site':
            p = src_pos + c
            psi[b, op[1], int(p[0]), int(p[1]), int(p[2])] = 1.0
        else:
            idx_t = torch.tensor(op[1], dtype=torch.int64, device=dev)
            val_t = torch.tensor(op[2], dtype=C.CD, device=dev)
            psi[b].view(4, -1).index_copy_(1, idx_t, val_t)
    e0, e2 = Hop.energy(psi)
    e0, e2 = e0.cpu().numpy(), e2.cpu().numpy()
    for b in range(B):
        src_list[b]['energy_mean'] = float(e0[b])
        src_list[b]['energy_std'] = float(math.sqrt(max(e2[b] - e0[b] ** 2, 0.0)))
        if src_list[b]['kind'] == 'pkt':
            print(f"[source {b}] {src_list[b]['label']}: k={src_list[b]['k']:.4f} <h>={e0[b]:.5f} (label {src_list[b]['omega']:g}, predicted shift "
                  f"{src_list[b]['predicted_shift']:+.2e}), energy std {src_list[b]['energy_std']:.4f}, exterior prob {src_list[b]['exterior_probability']:.3f}, "
                  f"upper band {src_list[b]['band']['upper_band']:.3f}, outgoing pos. freq {src_list[b]['band']['outgoing_positive_frequency']:.3f}, "
                  f"lattice scale {src_list[b]['band']['lattice_scale']:.3f}, zero mode {src_list[b]['band']['zero_mode']:.1e}", flush=True)
    # energy-matched receiver packets (review eq. G10.6: the ray comparison is made at the run's actual carrier energy)
    if radial:
        for b, sd in enumerate(src_list):
            if sd['kind'] != 'pkt' or sd['res'] != 'raw' or sd['energy_mean'] <= m:
                continue
            om_act = float(sd['energy_mean'])
            for rsite in rec_sites:
                role = f"rec{rsite['frac']:g}"
                restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.sigma < rs else [])
                add_packet_set(rsite['pos'], rsite['r'], role, [om_act], args.sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'],
                               dict(frac=rsite['frac'], matched_to=b, omega_label=sd['omega']), tag=f"|m{b}")
                if any(abs(rsite['frac'] - w) < 1e-9 for w in args.wide_radii):
                    restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.wide_sigma < rs else [])
                    add_packet_set(rsite['pos'], rsite['r'], role, [om_act], args.wide_sigma, restr, (+1, -1), rsite['u_prep'], rsite['N_prep'],
                                   dict(frac=rsite['frac'], matched_to=b, omega_label=sd['omega']), tag=f"|m{b}")
    rec.finalize()
    geo = C.Geometry(L, c, rs, (r_h if math.isfinite(r_h) else float(rs)), dev, region_fracs=tuple(args.regions))
    psi0 = psi.clone() if args.fft_check_sweeps else None

    # ---- solver setup
    if args.solver == 'cheb':
        n_cheb, tr, Cm, Ik, tail = C.cheb_coefficients(dE, tau, args.dt_read)
        c_end = Cm[-1]
        apps_per_sweep, J_taylor, nst, tailT = n_cheb + 1, None, None, None
    else:
        J_taylor = C.taylor_order(dE, args.taylor_dt)
        nst = int(round(tau / args.taylor_dt))
        assert abs(nst * args.taylor_dt - tau) < 1e-9
        tailT = C.taylor_tail_bound(J_taylor, dE * args.taylor_dt)
        apps_per_sweep, n_cheb, tail = J_taylor * nst, None, None
    print(f"[solver] {args.solver}: {apps_per_sweep} applications per sweep, {nsweeps} sweeps of tau={tau:g}, "
          f"{apps_per_sweep * nsweeps} total; B={B}, nrec={rec.nrec}; truncation tail per sweep {tail if tail is not None else tailT * nst:.2e}", flush=True)

    # ---- time loop
    times = [np.array([0.0])]
    A_blocks = [rec.project(psi).cpu().numpy()[None]]
    cert_blocks, certb_blocks, certt_blocks = [np.zeros((1, B))], [np.zeros((1, B))], [np.zeros((1, B))]
    d0 = geo.measure(psi)
    res0 = Hop.boundary_residual(psi).cpu().numpy()
    sweep_t, norms, hist, shares, bnd = [0.0], [d0[0]], [d0[1]], [d0[2]], [d0[3]]
    plane_t, planes_z0, planes_dg, res_exact, energies = [0.0], [d0[4]], [d0[5]], [res0], [e0]
    grams, norm_excess = [geo.gram(psi)], []
    proj_P, proj_R, proj_N = [], [], []
    eps_bnd = np.zeros(B)
    eps_trn = 0.0
    t = 0.0
    for s in range(nsweeps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts = time.time()
        if args.solver == 'cheb':
            psi, P, R, Nr = C.cheb_sweep(Hop, psi, dE, c_end, n_cheb, rec.project)
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
        else:
            Al, tl, cbl, ctl, Pl = [], [], [], [], []
            for j in range(nst):
                psi, inc, Pst = C.taylor_step(Hop, psi, args.taylor_dt, J_taylor, project=rec.project)
                eps_bnd = eps_bnd + inc.cpu().numpy()
                eps_trn += tailT
                Al.append(rec.project(psi).cpu().numpy())
                tl.append(t + (j + 1) * args.taylor_dt)
                cbl.append(eps_bnd.copy())
                ctl.append(eps_trn * np.ones(B))
                if not args.no_proj:
                    Pl.append(Pst.cpu().numpy())
            A, tb, cb, ct = np.stack(Al, 0), np.array(tl), np.stack(cbl, 0), np.stack(ctl, 0)
            if not args.no_proj:
                proj_P.append(np.stack(Pl, 0))
        t += tau
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        wall = time.time() - ts
        dd = geo.measure(psi)
        err = float(np.max(np.abs(dd[0] - 1.0)))
        rex = Hop.boundary_residual(psi).cpu().numpy()
        es, _ = Hop.energy(psi)
        es = es.cpu().numpy()
        G = geo.gram(psi)
        print(f"[sweep {s + 1}/{nsweeps}] t={t:.1f} ({t / rs:.2f} rs) wall={wall:.1f}s ({1e3 * wall / apps_per_sweep:.1f} ms/application at B={B}) "
              f"|norm-1|={err:.2e} |E-E0|max={np.max(np.abs(es - e0)):.2e} cert(min,max)=({(eps_bnd + eps_trn).min():.2e},{(eps_bnd + eps_trn).max():.2e}) "
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
        if psi0 is not None and (s + 1) in args.fft_check_sweeps and args.mode in ('flat', 'uniform'):
            vconst = tuple(args.uniform_v) if args.mode == 'uniform' else (0.0, 0.0, 0.0)
            ref = C.free_propagate_fft(psi0, t, m, vconst)
            e = (psi - ref).abs().max().item()
            print(f"[fft check] t={t:.1f}: max|psi(open) - psi(periodic, exact)| = {e:.3e}", flush=True)
            del ref

    times = np.concatenate(times)
    A_all = np.concatenate(A_blocks, 0)
    cert_all = np.concatenate(cert_blocks, 0)
    meta = dict(vars(args))
    meta.update(dict(stem=stem, L=L, H=H, c=c, Rb=Rb, tau=tau, tmax=tmax, nsweeps=nsweeps, dE=dE, bound=bound, n_cheb=n_cheb, J_taylor=J_taylor,
                     nst=nst, tail_cheb=tail, tail_taylor=tailT, r_h=r_h, kappa_h=kappa_h, crossings=crossings, crossings_radial=crossings_rad,
                     src=src_list, src_pos=src_pos.tolist(), src_r=src_r, u_src=u_src, N_src=N_src, u_src_prep=u_src_prep, nhat=nhat.tolist(),
                     prep_label=prep.label, axis_label=axf.label,
                     rec_sites=[dict(frac=d['frac'], pos=d['pos'].tolist(), r=d['r'], u_run=d['u_run'], N_run=d['N_run'], u_prep=d['u_prep']) for d in rec_sites],
                     rec_meta=rec.meta, labels=rec.labels, B=B, nrec=rec.nrec, group_cols=rec.group_cols, share_radii=geo.share_radii,
                     region_names=geo.region_names, napply=Hop.napply, wall_total=time.time() - tw, axpy_inplace=C.AXPY_OK,
                     rs_file=(prof.rs_file if prof is not None else None)))
    arrays = dict(times=times, A=A_all, cert=cert_all, cert_boundary=np.concatenate(certb_blocks, 0), cert_trunc=np.concatenate(certt_blocks, 0),
                  sweep_t=np.array(sweep_t), norms=np.array(norms), hist=np.array(hist), shares=np.array(shares), bnd=np.array(bnd),
                  res_exact=np.array(res_exact), energy=np.array(energies), gram=np.array(grams), plane_t=np.array(plane_t),
                  plane_z0=np.array(planes_z0), plane_diag=np.array(planes_dg), prof_rg=rg_p, prof_V=V_p,
                  axis_r=axf.r, axis_u=axf.u_nodes, axis_N=(axf.N_nodes if axf.N_nodes is not None else np.zeros(0)),
                  ax100_u=ax100_u, ax111_u=ax111_u, ax100_r=nax.astype(float), ax111_r=nax * C.SQRT3,
                  prep_r=prep.r, prep_u=prep.u_nodes, norm_excess=(np.array(norm_excess) if norm_excess else np.zeros((0, B))))
    for gi, g in enumerate(rec.groups):
        arrays[f'group_S_{gi}'] = g['S']
    if not args.no_proj:
        arrays['proj_P'] = np.stack(proj_P, 0)
        if args.solver == 'cheb':
            arrays['proj_R'] = np.stack(proj_R, 0)
            arrays['proj_N'] = np.stack(proj_N, 0)
    np.savez_compressed(path + '.npz', **arrays)
    with open(path + '.json', 'w') as f:
        json.dump(meta, f, indent=1, default=_json_default)
    print(f"[done] {path}  applications={Hop.napply}  wall={time.time() - tw:.0f}s", flush=True)


if __name__ == '__main__':
    main()
