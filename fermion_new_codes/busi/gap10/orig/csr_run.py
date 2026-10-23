#!/usr/bin/env python
"""csr_run.py -- one propagation: all sources at one position (a batch) against all receivers,
with the in-sweep readout, the parent certificate and the diagnostics."""
import argparse
import json
import math
import os
import time
import numpy as np
import torch
import csr_core as C


def direction_vector(d):
    return np.array([1.0, 0.0, 0.0]) if d == '100' else np.ones(3) / math.sqrt(3.0)


def site_on_direction(r, d):
    if d == '100':
        pos = np.array([int(round(r)), 0, 0], dtype=int)
    else:
        n = int(round(r / math.sqrt(3.0)))
        pos = np.array([n, n, n], dtype=int)
    return pos, float(np.linalg.norm(pos))


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


def pkt_label(role, r, om, sig, res, hel):
    return f"pkt|{role}|r={r:.3f}|w={om:g}|sig={sig:g}|res={res}|h={'+' if hel > 0 else '-'}"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser(description="retarded cross-surface response: one source position")
    ap.add_argument('--rs', type=int, default=32)
    ap.add_argument('--half_frac', type=float, default=3.0, help="half-width H = half_frac*rs + margin")
    ap.add_argument('--margin', type=int, default=16)
    ap.add_argument('--ma', type=float, default=0.0)
    ap.add_argument('--ell', type=float, default=4.0)
    ap.add_argument('--Rb_frac', type=float, default=0.6)
    ap.add_argument('--sampling', choices=['averaged', 'point'], default='averaged')
    ap.add_argument('--mode', choices=['radial', 'flat', 'uniform'], default='radial')
    ap.add_argument('--uniform_v', type=float, nargs=3, default=[0.0, 0.0, -0.5])
    ap.add_argument('--direction', choices=['100', '111'], default='100')
    ap.add_argument('--depth', type=float, default=4.0, help="source at r = rs - depth (nearest site)")
    ap.add_argument('--source', choices=['site', 'packet', 'both'], default='both')
    ap.add_argument('--src_freqs', type=float, nargs='*', default=None)
    ap.add_argument('--rec_freqs', type=float, nargs='*', default=None)
    ap.add_argument('--rec_radii', type=float, nargs='*', default=[1.25, 1.5, 2.0, 3.0])
    ap.add_argument('--sigma', type=float, default=6.0, help="packet width: standard deviation of |f|^2, in a")
    ap.add_argument('--cut', type=float, default=5.0, help="support radius in units of sigma")
    ap.add_argument('--wide_sigma', type=float, default=12.0)
    ap.add_argument('--wide_radii', type=float, nargs='*', default=[2.0, 3.0])
    ap.add_argument('--spins', type=int, default=1, help="helicity states for the packet sources (1 or 2)")
    ap.add_argument('--inside_twins_max_depth', type=float, default=4.0,
                    help="inside-restricted twin source packets at depths <= this (0 disables)")
    ap.add_argument('--prep', choices=['fixed', 'matched'], default='fixed',
                    help="packet carriers from the fixed preparation profile (Rb = 0.6 rs, ell = 4, averaged) or the run's")
    ap.add_argument('--tmax_rs', type=float, default=20.0)
    ap.add_argument('--tau', type=float, default=None, help="sweep length (default rs)")
    ap.add_argument('--dt_read', type=float, default=0.5)
    ap.add_argument('--solver', choices=['cheb', 'taylor'], default='cheb')
    ap.add_argument('--taylor_dt', type=float, default=1.0)
    ap.add_argument('--fft_check_sweeps', type=int, nargs='*', default=[])
    ap.add_argument('--out', default='runs')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    dev = pick_device(args.device)
    rs, m = args.rs, args.ma
    H = int(round(args.half_frac * rs)) + args.margin
    L, c = 2 * H + 1, H
    Rb = args.Rb_frac * rs
    tau = float(args.tau) if args.tau is not None else float(rs)
    tmax = args.tmax_rs * rs
    nsweeps = int(round(tmax / tau))
    if args.src_freqs is None:
        args.src_freqs = [0.05, 0.1, 0.2] if m == 0.0 else [0.25, 0.3, 0.4]
    if args.rec_freqs is None:
        args.rec_freqs = [0.05, 0.07, 0.1, 0.14, 0.2, 0.28, 0.4, 0.57, 0.8] if m == 0.0 else [0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    args.src_freqs = [w for w in args.src_freqs if w > m]
    args.rec_freqs = [w for w in args.rec_freqs if w > m]
    nhat = direction_vector(args.direction)
    averaged = (args.sampling == 'averaged')
    radial = (args.mode == 'radial')
    stem = (f"rs{rs}_m{m:g}_ell{args.ell:g}_Rb{args.Rb_frac:g}_{args.sampling}_{args.mode}_H{H}"
            f"_{args.direction}_d{args.depth:g}_{args.source}_{args.solver}{args.tag}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, stem)
    print(f"[run] {stem} on {dev}", flush=True)
    tw = time.time()

    # ---- fields on the halo grid (L+4)^3, interior (L)^3, ghost planes for the certificate
    u_cont = lambda r: C.u_continuum(r, rs, Rb)
    v_full, prof = C.build_site_flow(L + 4, c + 2, args.ell, u_cont, averaged, mode=args.mode, uniform_v=args.uniform_v)
    v = [a[2:-2, 2:-2, 2:-2].copy() for a in v_full]
    ghost = C.ghost_weights_np(v_full, L)
    del v_full
    if radial:
        if averaged:
            rg_p, V_p = prof
            u_fn = lambda r: np.interp(np.asarray(r, dtype=float), rg_p, V_p)
        else:
            rg_p = V_p = np.zeros(0)
            u_fn = lambda r: C.u_continuum(np.asarray(r, dtype=float), rs, Rb)
    else:
        rg_p = V_p = np.zeros(0)
        u_fn = lambda r: np.zeros_like(np.asarray(r, dtype=float))
    if radial and args.prep == 'fixed':
        rg_q, V_q = C.l1_projected_profile(lambda r: C.u_continuum(r, rs, 0.6 * rs), 4.0, 4.0 * rs + 8.0)
        u_prep = lambda r: np.interp(np.asarray(r, dtype=float), rg_q, V_q)
    else:
        u_prep = u_fn
    nax = np.arange(0, H + 1)
    axis_u100 = -v[0][c + nax, c, c]
    axis_u111 = -(v[0][c + nax, c + nax, c + nax] + v[1][c + nax, c + nax, c + nax] + v[2][c + nax, c + nax, c + nax]) / math.sqrt(3.0)
    wp, wm = C.bond_weights_np(v)
    del v
    bound = C.spectral_bound(m, wp, wm)
    dE = 1.01 * bound
    Hop = C.LatticeH(m, wp, wm, dev, ghost_np=ghost)
    del wp, wm, ghost
    crossings = C.crossings_between(u_fn, 1e-3, 3.0 * rs) if radial else []
    if crossings:
        r_h, kappa_h = crossings[-1][0], 1.0 / crossings[-1][1]
    else:
        r_h, kappa_h = float('nan'), float('nan')
    print(f"[fields] L={L} H={H} rs={rs} ma={m} ell={args.ell} Rb={Rb} {args.sampling} {args.mode} "
          f"bound={bound:.5f} dE={dE:.5f} ({time.time() - tw:.1f}s)", flush=True)
    print(f"[profile] r_h={r_h:.4f} (rs={rs}) kappa_h={kappa_h:.6f} (1/(2 rs)={1 / (2 * rs):.6f}) "
          f"all |v|=1 crossings (radius, inverse slope)={crossings}", flush=True)

    # ---- positions and receivers
    src_pos, src_r = site_on_direction(rs - args.depth, args.direction)
    assert np.max(np.abs(src_pos)) <= H, "source outside the box"
    u_src = float(u_fn(np.array([src_r]))[0])
    u_src_prep = float(u_prep(np.array([src_r]))[0])
    rec = C.Receivers(L, c, dev)
    rec.add_site(src_pos, 'src', src_r)
    rec_sites = []
    for fr in args.rec_radii:
        pos, r = site_on_direction(fr * rs, args.direction)
        if np.max(np.abs(pos)) > H:
            print(f"[warn] receiver at {fr:g} rs is outside the box (H={H}); skipped", flush=True)
            continue
        rec_sites.append(dict(frac=fr, pos=pos, r=r, u_run=float(u_fn(np.array([r]))[0]), u_prep=float(u_prep(np.array([r]))[0])))
        rec.add_site(pos, f"rec{fr:g}", r)

    def add_packet_set(pos, r, role, freqs, sig, restrictions, hels, u_here, extra):
        for res in restrictions:
            idx_ref, vals, metas = None, [], []
            for om in freqs:
                try:
                    k = C.outgoing_root(om, u_here, nhat, m)
                except ValueError as e:
                    print(f"[warn] {role} sigma={sig:g}: {e}", flush=True)
                    continue
                spn = C.helicity_spinors(k, nhat, m)
                for hel, sp in zip((+1, -1), spn):
                    if hel not in hels:
                        continue
                    try:
                        idx, values, coords, prob, meta = C.packet_mode(L, c, pos, nhat, k, sp, sig, args.cut, rs, restriction=res)
                    except ValueError as e:
                        print(f"[warn] {role} res={res}: {e}", flush=True)
                        continue
                    if idx_ref is None:
                        idx_ref = idx
                    else:
                        assert np.array_equal(idx_ref, idx)
                    meta.update(kind='pkt', role=role, r=float(r), omega=float(om), sigma=float(sig), res=res, hel=int(hel),
                                k=float(k), vg=float(C.disp_vg(k, u_here, nhat, m)[0]), u_prep=float(u_here),
                                label=pkt_label(role, r, om, sig, res, hel))
                    meta.update(extra)
                    vals.append(values)
                    metas.append(meta)
            if vals:
                rec.add_packet_group(idx_ref, vals, metas)

    if radial:
        for rsite in rec_sites:
            role = f"rec{rsite['frac']:g}"
            restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.sigma < rs else [])
            add_packet_set(rsite['pos'], rsite['r'], role, args.rec_freqs, args.sigma, restr, (+1, -1), rsite['u_prep'], dict(frac=rsite['frac']))
            if any(abs(rsite['frac'] - w) < 1e-9 for w in args.wide_radii):
                restr = ['raw'] + (['out'] if rsite['r'] - args.cut * args.wide_sigma < rs else [])
                add_packet_set(rsite['pos'], rsite['r'], role, args.src_freqs, args.wide_sigma, restr, (+1, -1), rsite['u_prep'], dict(frac=rsite['frac']))

    # ---- sources
    src_list, init_ops = [], []
    if args.source in ('site', 'both'):
        for comp in range(4):
            src_list.append(dict(kind='site', role='src', comp=comp, pos=src_pos.tolist(), r=src_r, u_run=u_src, res='raw',
                                 exterior_probability=0.0, label=f"site|src|r={src_r:.3f}|c={comp}"))
            init_ops.append(('site', comp))
    if args.source in ('packet', 'both') and radial:
        restr = ['raw'] + (['in'] if (args.inside_twins_max_depth > 0 and args.depth <= args.inside_twins_max_depth + 1e-9) else [])
        hw = int(math.ceil(args.cut * args.sigma))
        for res in restr:
            idx_ref, vals, metas = None, [], []
            for om in args.src_freqs:
                k = C.outgoing_root(om, u_src_prep, nhat, m)
                spn = C.helicity_spinors(k, nhat, m)
                for hel, sp in list(zip((+1, -1), spn))[:args.spins]:
                    idx, values, coords, prob, meta = C.packet_mode(L, c, src_pos, nhat, k, sp, args.sigma, args.cut, rs, restriction=res)
                    bw = C.band_weights(values, coords, src_pos, m, -u_src * nhat, nhat, hw)
                    meta.update(kind='pkt', role='src', pos=src_pos.tolist(), r=src_r, omega=float(om), sigma=float(args.sigma), res=res,
                                hel=int(hel), k=float(k), vg=float(C.disp_vg(k, u_src_prep, nhat, m)[0]), u_prep=u_src_prep, u_run=u_src,
                                band=bw, label=pkt_label('src', src_r, om, args.sigma, res, hel))
                    src_list.append(meta)
                    init_ops.append(('packet', idx, values))
                    vals.append(values)
                    metas.append(dict(meta))
                    if idx_ref is None:
                        idx_ref = idx
            if vals:
                rec.add_packet_group(idx_ref, vals, metas)
    rec.finalize()
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
    geo = C.Geometry(L, c, rs, (r_h if math.isfinite(r_h) else float(rs)), dev)
    psi0 = psi.clone() if args.fft_check_sweeps else None

    # ---- solver setup
    if args.solver == 'cheb':
        n_cheb, tr, Cm, Ik = C.cheb_coefficients(dE, tau, args.dt_read)
        Cg = torch.tensor(Cm, dtype=C.CD, device=dev)
        c_end = Cm[-1]
        apps_per_sweep, J_taylor, nst = n_cheb, None, None
    else:
        J_taylor = C.taylor_terms(dE, args.taylor_dt)
        nst = int(round(tau / args.taylor_dt))
        assert abs(nst * args.taylor_dt - tau) < 1e-9
        apps_per_sweep, n_cheb = J_taylor * nst, None
    print(f"[solver] {args.solver}: {apps_per_sweep} applications per sweep, {nsweeps} sweeps, "
          f"{apps_per_sweep * nsweeps} total; B={B}, nrec={rec.nrec}", flush=True)

    # ---- time loop
    times = [np.array([0.0])]
    A_blocks = [rec.project(psi).cpu().numpy()[None]]
    cert_blocks = [np.zeros((1, B))]
    d0 = geo.measure(psi)
    res0 = Hop.boundary_residual(psi).cpu().numpy()
    sweep_t, norms, hist, shares, bnd = [0.0], [d0[0]], [d0[1]], [d0[2]], [d0[3]]
    planes_z0, planes_dg, res_exact, energies = [d0[4]], [d0[5]], [res0], [e0]
    eps_cum = np.zeros(B)
    t = 0.0
    for s in range(nsweeps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts = time.time()
        if args.solver == 'cheb':
            psi, P, R = C.cheb_sweep(Hop, psi, dE, c_end, n_cheb, rec.project)
            A = torch.einsum('jk,kbr->jbr', Cg, P).cpu().numpy()
            Rn = R.cpu().numpy()
            cert = eps_cum[None, :] + Ik @ Rn
            eps_cum = cert[-1].copy()
            del P, R
            tb = t + tr
        else:
            Al, tl, cl = [], [], []
            for j in range(nst):
                psi, inc = C.taylor_step(Hop, psi, args.taylor_dt, J_taylor)
                eps_cum = eps_cum + inc.cpu().numpy()
                Al.append(rec.project(psi).cpu().numpy())
                tl.append(t + (j + 1) * args.taylor_dt)
                cl.append(eps_cum.copy())
            A, tb, cert = np.stack(Al, 0), np.array(tl), np.stack(cl, 0)
        t += tau
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        wall = time.time() - ts
        dd = geo.measure(psi)
        err = float(np.max(np.abs(dd[0] - 1.0)))
        rex = Hop.boundary_residual(psi).cpu().numpy()
        es, _ = Hop.energy(psi)
        es = es.cpu().numpy()
        print(f"[sweep {s + 1}/{nsweeps}] t={t:.1f} ({t / rs:.2f} rs) wall={wall:.1f}s "
              f"({1e3 * wall / apps_per_sweep:.1f} ms/application at B={B}) |norm-1|={err:.2e} "
              f"|E-E0|max={np.max(np.abs(es - e0)):.2e} cert(min,max)=({eps_cum.min():.2e},{eps_cum.max():.2e}) "
              f"exact||B psi||max={rex.max():.2e} shares>{[nm for nm, _ in geo.share_radii]} (batch mean)="
              f"{np.round(dd[2].mean(0), 4).tolist()} boundary-layer max={dd[3].max():.2e}", flush=True)
        if s == 0:
            print(f"[projection] this run about {wall * nsweeps / 60:.1f} min", flush=True)
        if err > 1e-9:
            raise RuntimeError(f"unitarity lost ({err:.2e}): the spectral interval is too small")
        times.append(tb)
        A_blocks.append(A)
        cert_blocks.append(cert)
        sweep_t.append(t); norms.append(dd[0]); hist.append(dd[1]); shares.append(dd[2]); bnd.append(dd[3])
        planes_z0.append(dd[4]); planes_dg.append(dd[5]); res_exact.append(rex); energies.append(es)
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
    meta.update(dict(stem=stem, L=L, H=H, c=c, Rb=Rb, tau=tau, tmax=tmax, nsweeps=nsweeps, dE=dE, bound=bound,
                     n_cheb=n_cheb, J_taylor=J_taylor, r_h=r_h, kappa_h=kappa_h, crossings=crossings,
                     src=src_list, src_pos=src_pos.tolist(), src_r=src_r, u_src=u_src, u_src_prep=u_src_prep, nhat=nhat.tolist(),
                     rec_sites=[dict(frac=d['frac'], pos=d['pos'].tolist(), r=d['r'], u_run=d['u_run'], u_prep=d['u_prep']) for d in rec_sites],
                     rec_meta=rec.meta, labels=rec.labels, B=B, nrec=rec.nrec,
                     share_radii=geo.share_radii, napply=Hop.napply, wall_total=time.time() - tw, axpy_inplace=C.AXPY_OK))
    np.savez_compressed(path + '.npz', times=times, A=A_all, cert=cert_all, sweep_t=np.array(sweep_t), norms=np.array(norms),
                        hist=np.array(hist), shares=np.array(shares), bnd=np.array(bnd), res_exact=np.array(res_exact),
                        energy=np.array(energies), plane_z0=np.array(planes_z0), plane_diag=np.array(planes_dg),
                        prof_rg=rg_p, prof_V=V_p, axis_u100=axis_u100, axis_u111=axis_u111)
    with open(path + '.json', 'w') as f:
        json.dump(meta, f, indent=1, default=_json_default)
    print(f"[done] {path}  applications={Hop.napply}  wall={time.time() - tw:.0f}s", flush=True)


if __name__ == '__main__':
    main()
