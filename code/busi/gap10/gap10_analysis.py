#!/usr/bin/env python
"""gap10_analysis.py -- the printed number (retarded cross-surface transmissivity with its certificate band), arrival times
re-located by exact evaluation of the stored polynomial projections, boundary-clock values and their ray comparison
(eq. G10.7), certificates (Frobenius blocks by the root sum of squares), escaped shares, frequency content, the delay law
against the exact ray integral at the actual energy, the normal form and the asymptotes, control pairs with eps_1 + eps_2,
tables and figures for the runs of gap10_run.py."""
import argparse
import csv
import glob
import json
import math
import os
import numpy as np
import gap10_core as C


def load(stem):
    d = np.load(stem + '.npz')
    with open(stem + '.json') as f:
        meta = json.load(f)
    return d, meta


def group_key(rm):
    if rm['kind'] == 'site':
        return ('site', rm['role'], round(rm['r'], 3), float('nan'), float('nan'), 'raw', -1)
    return ('pkt', rm['role'], round(rm['r'], 3), round(rm['omega'], 6), rm['sigma'], rm['res'], int(rm.get('matched_to', -1)))


def amplitudes_at(d, meta, times, b, cols):
    """A(t) for arbitrary times from the stored polynomial projections: exact within the solver's own polynomial."""
    times = np.asarray(times, dtype=float)
    cols = np.asarray(cols, dtype=int)
    out = np.zeros((times.size, cols.size), dtype=np.complex128)
    if 'proj_P' not in d.files:
        tt, A = d['times'], d['A']
        for j, t in enumerate(times):
            i = int(np.argmin(np.abs(tt - t)))
            out[j] = A[i, b, cols]
        return out
    P = d['proj_P']
    tau, nsweeps = float(meta['tau']), int(meta['nsweeps'])
    if meta['solver'] == 'cheb':
        dE, n = float(meta['dE']), int(meta['n_cheb'])
        for j, t in enumerate(times):
            if t <= 0.0:
                out[j] = d['A'][0, b, cols]
                continue
            s = min(int(math.ceil(t / tau - 1e-12)) - 1, nsweeps - 1)
            tp = t - s * tau
            Cm = C.cheb_readout(dE, n, [tp])[0]
            out[j] = Cm @ P[s][:, b, cols]
    else:
        dt = float(meta['taylor_dt'])
        nst = int(meta['nst'])
        Pf = P.reshape(nsweeps * nst, P.shape[2], P.shape[3], P.shape[4])
        for j, t in enumerate(times):
            if t <= 0.0:
                out[j] = d['A'][0, b, cols]
                continue
            i = min(int(math.ceil(t / dt - 1e-12)) - 1, Pf.shape[0] - 1)
            fr = (t - i * dt) / dt
            w = fr ** np.arange(Pf.shape[1])
            out[j] = np.tensordot(w, Pf[i][:, b, cols], axes=(0, 0))
    return out


def cert_at(tt, cert_col, t):
    """The cumulative certificate is monotone: the value at the first readout time >= t bounds it at t."""
    i = int(np.searchsorted(tt, t - 1e-12, side='left'))
    return float(cert_col[min(i, len(tt) - 1)])


def arrival_stats(tt, amp, cert_col, thr_rel=1e-6):
    peak = float(amp.max())
    out = dict(t_first=np.nan, amp_first=np.nan, t_peak=np.nan, peak=peak, t10=np.nan, t50=np.nan, A0=float(amp[0]), initial_overlap=False,
               cert_first=np.nan, cert_peak=np.nan, resolved=False, resolved_first=False, i_first=-1, i_peak=-1)
    if not np.isfinite(peak) or peak <= 0.0:
        return out
    ipk = int(amp.argmax())
    lm = np.flatnonzero((amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])) + 1
    cand = lm[amp[lm] > thr_rel * peak]
    i_first = int(cand[0]) if cand.size else ipk
    out.update(t_first=float(tt[i_first]), amp_first=float(amp[i_first]), t_peak=float(tt[ipk]), peak=peak,
               t10=float(tt[int(np.flatnonzero(amp > 0.1 * peak)[0])]), t50=float(tt[int(np.flatnonzero(amp > 0.5 * peak)[0])]),
               initial_overlap=bool(amp[0] > 1e-3 * peak), cert_first=float(cert_col[i_first]), cert_peak=float(cert_col[ipk]),
               resolved=bool(peak > 10.0 * cert_col[ipk]), resolved_first=bool(amp[i_first] > 10.0 * cert_col[i_first]), i_first=i_first, i_peak=ipk)
    return out


def refine(d, meta, b, cols, t_c, dt, nfine=50):
    """Re-locate an extremum on [t_c - dt, t_c + dt] by exact evaluation at 2 nfine + 1 points."""
    ts = np.linspace(max(t_c - dt, 0.0), min(t_c + dt, float(meta['tmax'])), 2 * nfine + 1)
    A = amplitudes_at(d, meta, ts, b, cols)
    amp = np.sqrt((np.abs(A) ** 2).sum(1))
    i = int(np.argmax(amp))
    return float(ts[i]), float(amp[i]), bool(i in (0, len(ts) - 1))


def spectrum(tt, x, t_a, t_b):
    """|sum_t x(t) e^{+i w t}|^2 on [t_a, t_b] with a Hann taper; positive w = positive energy."""
    sel = (tt >= t_a) & (tt <= t_b)
    if sel.sum() < 16:
        return None, None, np.nan
    xs = x[sel] * np.hanning(int(sel.sum()))
    dt = float(np.median(np.diff(tt[sel])))
    F = np.conj(np.fft.fft(np.conj(xs)))
    w = 2.0 * math.pi * np.fft.fftfreq(xs.size, d=dt)
    o = np.argsort(w)
    return w[o], np.abs(F[o]) ** 2, 2.0 * math.pi / (tt[sel][-1] - tt[sel][0])


def analyze(stem, rows, trows, figdir=None):
    d, meta = load(stem)
    tt, A, cert = d['times'], d['A'], d['cert']
    rs, m = float(meta['rs']), float(meta['ma'])
    nhat = np.array(meta['nhat'])
    src_r = float(meta['src_r'])
    radial = meta['mode'] == 'radial'
    sweep_t, shares, bnd = d['sweep_t'], d['shares'], d['bnd']
    dt_read = float(np.median(np.diff(tt))) if tt.size > 2 else float(meta['dt_read'])
    if radial:
        axN = d['axis_N'] if ('axis_N' in d.files and d['axis_N'].size) else None
        axf = C.AxisField(d['axis_r'], d['axis_u'], axN)
        u_fn, N_fn = axf.u, axf.N
    else:
        u_fn = lambda r: np.zeros_like(np.asarray(r, dtype=float))
        N_fn = None
    crossings = [tuple(x) for x in meta.get('crossings', [])]
    r_h = float(meta.get('r_h', float('nan')) if meta.get('r_h') is not None else float('nan'))
    kappa_h = float(meta.get('kappa_h', float('nan')) if meta.get('kappa_h') is not None else float('nan'))
    gamma = kappa_h
    b_nf = 0.5 * (float(np.asarray(N_fn(r_h)).reshape(-1)[0]) if (radial and N_fn is not None and np.isfinite(r_h)) else 1.0)
    rec_meta = meta['rec_meta']
    groups = {}
    for j, rm in enumerate(rec_meta):
        groups.setdefault(group_key(rm), []).append(j)
    base = dict(stem=os.path.basename(stem), rs=rs, ma=m, ell=meta['ell'], Rb_frac=meta['Rb_frac'], sampling=meta['sampling'], profile=meta['profile'],
                mode=meta['mode'], H=meta['H'], direction=meta['direction'], depth=meta['depth'], src_r=src_r, u_src=meta['u_src'], solver=meta['solver'],
                tau=meta['tau'], r_h=r_h, kappa_h=kappa_h, n_crossings_total=len(crossings),
                norm_err=float(np.max(np.abs(d['norms'] - 1.0))), energy_err=float(np.max(np.abs(d['energy'] - d['energy'][0][None, :]))),
                norm_excess=(float(np.max(d['norm_excess'])) if d['norm_excess'].size else np.nan), napply=meta.get('napply', -1))
    names = [nm for nm, _ in meta['share_radii']]
    cache = {}
    cross_r = [x[0] for x in crossings]

    def t_ray(r1, om):
        key = (round(r1, 6), round(om, 9))
        if key not in cache:
            try:
                tr_ = C.ray_time(src_r, r1, om, nhat, m, u_fn, N_fn, crossings=cross_r)
                cr = [x for x in crossings if min(src_r, r1) < x[0] < max(src_r, r1)]
                cache[key] = (tr_, float(sum(x[1] for x in cr)), len(cr))
            except Exception:
                cache[key] = (float('nan'), float('nan'), 0)
        return cache[key]

    def t_nf(r1, om):
        if not (radial and np.isfinite(gamma) and gamma > 0 and np.isfinite(r_h) and r1 > r_h and src_r <= r_h + 1e-9):
            return float('nan')
        try:
            return C.nf_delay(om, r_h - src_r, r1 - r_h, gamma, b_nf)
        except Exception:
            return float('nan')

    def chart(row, r1, rm):
        if radial and r1 > rs:
            for nm in ('t_first', 't_peak'):
                if np.isfinite(row[nm]):
                    row['T_' + nm[2:]] = C.chart_T(row[nm], r1, rs)
                    row['u_' + nm[2:]] = C.chart_u(row[nm], r1, rs)
            row.update(T_rms_kinematic=rm.get('T_rms', np.nan), u_rms_kinematic=rm.get('u_rms', np.nan))
            if np.isfinite(row.get('t_ray', np.nan)):
                row['u_ray_rel'] = row['t_ray'] - float(C.clock_offsets(r1, rs)[1])

    for b, sd in enumerate(meta['src']):
        cb = cert[:, b]
        common = dict(base)
        common.update(src_index=b, src_kind=sd['kind'], src_comp=sd.get('comp', -1), omega_src=sd.get('omega', np.nan), hel_src=sd.get('hel', 0),
                      res_src=sd.get('res', 'raw'), k_src=sd.get('k', np.nan), src_exterior=sd.get('exterior_probability', np.nan),
                      src_retained=sd.get('retained', np.nan), src_energy_mean=sd.get('energy_mean', np.nan), src_energy_std=sd.get('energy_std', np.nan),
                      src_prep_mismatch=sd.get('prep_mismatch', np.nan), src_predicted_shift=sd.get('predicted_shift', np.nan),
                      lipschitz_dt=(dt_read * sd.get('energy_std', np.nan)))
        bw = sd.get('band', None)
        if bw:
            common.update(src_upper_band=bw['upper_band'], src_pos_freq=bw['positive_frequency'], src_out_pos_freq=bw['outgoing_positive_frequency'],
                          src_lattice_scale=bw['lattice_scale'], src_zero_mode=bw.get('zero_mode', 0.0))
        for eps_name, eps_val in (('1e-3', 1e-3), ('1e-2', 1e-2), ('1e-1', 1e-1)):
            ok = np.flatnonzero(cb < eps_val)
            common[f't_clean_{eps_name}'] = float(tt[ok[-1]]) if ok.size else 0.0
        over = np.flatnonzero(bnd[:, b] > 1e-4)
        common['t_bnd_1e4'] = float(sweep_t[over[0]]) if over.size else np.nan
        s_clean = np.flatnonzero(sweep_t <= common['t_clean_1e-3'])
        for k, nm in enumerate(names):
            common[f'share_peak_{nm}'] = float(shares[:, b, k].max())
            common[f'share_end_{nm}'] = float(shares[-1, b, k])
            common[f'share_clean_{nm}'] = float(shares[s_clean, b, k].max()) if s_clean.size else 0.0
        om_act = float(sd.get('energy_mean', np.nan)) if sd['kind'] == 'pkt' else np.nan
        for key, cols in groups.items():
            rm = rec_meta[cols[0]]
            amp = np.sqrt(sum(np.abs(A[:, b, j]) ** 2 for j in cols))
            st = arrival_stats(tt, amp, cb)
            r1 = float(rm['r'])
            row = dict(common)
            if np.isfinite(st['t_first']):
                tf, af, edge_f = refine(d, meta, b, cols, st['t_first'], dt_read)
                tp, ap, edge_p = refine(d, meta, b, cols, st['t_peak'], dt_read)
                st.update(t_first=tf, amp_first=af, t_peak=tp, peak=ap, cert_first=cert_at(tt, cb, tf), cert_peak=cert_at(tt, cb, tp),
                          edge_first=edge_f, edge_peak=edge_p)
                st['resolved'] = bool(ap > 10.0 * st['cert_peak'])
                st['resolved_first'] = bool(af > 10.0 * st['cert_first'])
            row.update(receiver=key[1], rec_kind=key[0], r_rec=r1, omega_rec=key[3], sigma_rec=key[4], res_rec=key[5], matched_to=key[6],
                       frac_rec=rm.get('frac', np.nan), rec_retained=rm.get('retained', np.nan), rec_clipped=rm.get('clipped', False),
                       rec_exterior=rm.get('exterior_probability', np.nan), **{k: v for k, v in st.items() if not k.startswith('i_')})
            if radial and key[0] == 'pkt' and key[1].startswith('rec'):
                om_cmp = om_act if (sd['kind'] == 'pkt' and np.isfinite(om_act)) else key[3]
                tr_, gam, ncr = t_ray(r1, om_cmp)
                row.update(t_ray=tr_, Gamma=gam, n_crossings=ncr, omega_cmp=om_cmp, t_nf=t_nf(r1, om_cmp),
                           t_ray_log=(gam * math.log(1.0 / om_cmp) if np.isfinite(gam) and gam > 0 else np.nan),
                           delta_omega=(C.nf_delta(om_cmp, gamma, b_nf) if np.isfinite(gamma) and gamma > 0 else np.nan))
            chart(row, r1, rm)
            jdom = max(cols, key=lambda j: np.abs(A[:, b, j]).max())
            if np.isfinite(st['t_first']):
                if key[0] == 'site':
                    w, S, res = spectrum(tt, A[:, b, jdom], st['t_first'] - rs, st['t_first'] + 3 * rs)
                else:
                    w, S, res = spectrum(tt, A[:, b, jdom], st['t_peak'] - 2 * rs, st['t_peak'] + 2 * rs)
                if w is not None and S.sum() > 0:
                    row.update(omega_peak=float(w[np.argmax(S)]), omega_centroid=float(np.sum(w * S) / np.sum(S)), omega_resolution=res)
                row['phase_at_peak'] = float(np.angle(A[int(np.argmin(np.abs(tt - st['t_peak']))), b, jdom]))
            rows.append(row)
    # the 4 x 4 site block with the root-sum-of-squares certificate (review G10.4)
    site_b = [b for b, sd in enumerate(meta['src']) if sd['kind'] == 'site']
    if len(site_b) == 4:
        cblock = np.sqrt((cert[:, site_b] ** 2).sum(1))
        common = dict(base)
        common.update(src_index=-1, src_kind='site_block', src_comp=-1, omega_src=np.nan, hel_src=0, res_src='raw')
        for key, cols in groups.items():
            if key[0] != 'site':
                continue
            rm = rec_meta[cols[0]]
            amp = np.sqrt(sum(np.abs(A[:, b, j]) ** 2 for b in site_b for j in cols))
            st = arrival_stats(tt, amp, cblock)
            r1 = float(rm['r'])
            row = dict(common)
            if np.isfinite(st['t_first']):
                ts = np.linspace(max(st['t_first'] - dt_read, 0.0), st['t_first'] + dt_read, 101)
                amps = np.sqrt(sum((np.abs(amplitudes_at(d, meta, ts, b, cols)) ** 2).sum(1) for b in site_b))
                i = int(np.argmax(amps))
                st.update(t_first=float(ts[i]), amp_first=float(amps[i]), cert_first=cert_at(tt, cblock, float(ts[i])))
                st['resolved_first'] = bool(st['amp_first'] > 10.0 * st['cert_first'])
            row.update(receiver=key[1], rec_kind='site', r_rec=r1, omega_rec=np.nan, sigma_rec=np.nan, res_rec='raw', matched_to=-1,
                       frac_rec=np.nan, **{k: v for k, v in st.items() if not k.startswith('i_')})
            chart(row, r1, rm)
            rows.append(row)
    # the printed number: s_max^2(P_D U B_X) over the sweep-end times, with the certificate band
    G = d['gram']
    gt = d['sweep_t']
    for k, rname in enumerate(meta['region_names']):
        fams = []
        if len(site_b) == 4:
            fams.append(('site_block', site_b))
        for b, sd in enumerate(meta['src']):
            if sd['kind'] == 'pkt':
                fams.append((sd['label'], [b]))
        for fname, idx in fams:
            best, best_res = (0.0, np.nan, np.nan, None), (0.0, np.nan, np.nan)
            series = []
            for s in range(G.shape[0]):
                sub = G[s, k][np.ix_(idx, idx)]
                lam, vec = C.smax2_from_gram(sub)
                delta = math.sqrt(sum(cert_at(tt, cert[:, b], float(gt[s])) ** 2 for b in idx))
                dR = 2.0 * math.sqrt(lam) * delta + delta * delta
                series.append((float(gt[s]), lam, dR))
                if lam > best[0]:
                    best = (lam, float(gt[s]), dR, vec)
                if lam > 10.0 * dR and lam > best_res[0]:
                    best_res = (lam, float(gt[s]), dR)
            trows.append(dict(stem=base['stem'], rs=rs, ma=m, direction=meta['direction'], depth=meta['depth'], src_r=src_r, r_h=r_h, region=rname,
                              family=fname, kind='region', R_ret=best[0], t_at_R=best[1], band_at_R=best[2], resolved=bool(best[0] > 10.0 * best[2]),
                              R_ret_resolved=best_res[0], t_resolved=best_res[1], band_resolved=best_res[2],
                              optimizer=(None if best[3] is None else [complex(x) for x in best[3]]),
                              series=series, solver=meta['solver'], H=meta['H'], ell=meta['ell'], Rb_frac=meta['Rb_frac'], sampling=meta['sampling']))
    # the receiver-filter transmissivity: Lowdin-orthonormalized receiver packets at a radius against the 4 site inputs
    if len(site_b) == 4:
        for gi, (c0, c1) in enumerate(meta['group_cols']):
            rm = rec_meta[c0]
            if rm['kind'] != 'pkt' or not rm['role'].startswith('rec') or rm['res'] != 'raw' or abs(rm['sigma'] - meta['sigma']) > 1e-9 or int(rm.get('matched_to', -1)) != -1:
                continue
            S = d[f'group_S_{gi}']
            W, rank = C.lowdin_inverse_sqrt(S)
            cols = list(range(c0, c1))
            best, best_res = (0.0, np.nan, np.nan), (0.0, np.nan, np.nan)
            cblock = np.sqrt((cert[:, site_b] ** 2).sum(1))
            for i in range(tt.size):
                M = W @ A[i][np.ix_(site_b, cols)].T
                sv = np.linalg.svd(M, compute_uv=False)
                lam = float(sv[0] ** 2) if sv.size else 0.0
                delta = float(cblock[i])
                dR = 2.0 * math.sqrt(lam) * delta + delta * delta
                if lam > best[0]:
                    best = (lam, float(tt[i]), dR)
                if lam > 10.0 * dR and lam > best_res[0]:
                    best_res = (lam, float(tt[i]), dR)
            trows.append(dict(stem=base['stem'], rs=rs, ma=m, direction=meta['direction'], depth=meta['depth'], src_r=src_r, r_h=r_h,
                              region=f"{rm['role']} packets (rank {rank})", family='site_block', kind='filter', R_ret=best[0], t_at_R=best[1], band_at_R=best[2],
                              resolved=bool(best[0] > 10.0 * best[2]), R_ret_resolved=best_res[0], t_resolved=best_res[1], band_resolved=best_res[2],
                              optimizer=None, series=None, solver=meta['solver'], H=meta['H'], ell=meta['ell'], Rb_frac=meta['Rb_frac'], sampling=meta['sampling']))
    if figdir:
        make_figures(stem, d, meta, figdir, groups)


def make_figures(stem, d, meta, figdir, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    name = os.path.basename(stem)
    tt, A, cert, rs = d['times'], d['A'], d['cert'], float(meta['rs'])
    srcs = meta['src']
    try:
        fig, axs = plt.subplots(len(srcs), 1, figsize=(9, 2.2 * len(srcs) + 1), sharex=True, squeeze=False)
        for b, sd in enumerate(srcs):
            ax = axs[b, 0]
            for key, cols in sorted(groups.items(), key=lambda kv: str(kv[0])):
                if key[0] != 'pkt' or not key[1].startswith('rec') or key[5] != 'raw' or abs(key[4] - meta['sigma']) > 1e-9:
                    continue
                if sd['kind'] == 'pkt' and key[6] != b:
                    continue
                if sd['kind'] == 'site' and (key[1] != 'rec2' or key[6] != -1):
                    continue
                amp = np.sqrt(sum(np.abs(A[:, b, j]) ** 2 for j in cols))
                ax.semilogy(tt / rs, np.maximum(amp, 1e-9), lw=0.8, label=f"{key[1][3:]} rs, a w={key[3]:.3g}")
            ax.semilogy(tt / rs, np.maximum(cert[:, b], 1e-9), 'k--', lw=0.8, label='certificate')
            ax.set_ylim(1e-6, 1.5)
            ax.grid(alpha=0.3)
            ttl = f"source {b}: " + (f"packet a w={sd['omega']:g} h{'+' if sd['hel'] > 0 else '-'} {sd['res']} (<h>={sd['energy_mean']:.4f})" if sd['kind'] == 'pkt' else f"site component {sd['comp']}")
            ax.set_title(ttl, fontsize=8)
            ax.legend(fontsize=6, ncol=4, loc='upper right')
        axs[-1, 0].set_xlabel('t / r_s')
        fig.suptitle(name, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name + '_receivers.png'), dpi=110)
        plt.close(fig)
    except Exception as e:
        print('  figure receivers failed:', repr(e))
    try:
        sh, st = d['shares'], d['sweep_t']
        names = [nm for nm, _ in meta['share_radii']]
        fig, ax = plt.subplots(figsize=(7, 4))
        for b in range(len(srcs)):
            for k, nm in enumerate(names):
                if nm not in ('r_h', '1.5rs', '2rs'):
                    continue
                ax.plot(st / rs, np.maximum(sh[:, b, k], 1e-9), ls={'r_h': '-', '1.5rs': '--', '2rs': ':'}[nm], lw=0.9, label=f"src{b} >{nm}")
        ax.set_yscale('log')
        ax.set_ylim(1e-6, 1.2)
        ax.set_xlabel('t / r_s')
        ax.set_ylabel('share of the norm (= s_max^2 for 1 input mode)')
        ax.legend(fontsize=6, ncol=3)
        ax.grid(alpha=0.3)
        fig.suptitle(name, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name + '_shares.png'), dpi=110)
        plt.close(fig)
    except Exception as e:
        print('  figure shares failed:', repr(e))
    try:
        pz, pdg, H = d['plane_z0'][-1, 0], d['plane_diag'][-1, 0], float(meta['H'])
        tpl = float(d['plane_t'][-1])
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, P, ttl in ((axs[0], pz, 'plane z = 0'), (axs[1], pdg, 'plane y = z ([100] and [111])')):
            im = ax.imshow(np.log10(np.maximum(P.T, 1e-16)), origin='lower', extent=[-H / rs, H / rs, -H / rs, H / rs],
                           vmin=-12, vmax=math.log10(max(float(P.max()), 1e-16)), cmap='viridis')
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='w', lw=0.6))
            ax.set_title(f"{ttl}, t = {tpl / rs:.0f} r_s, source 0", fontsize=8)
            ax.set_xlabel('x / r_s')
            fig.colorbar(im, ax=ax, label='log10 |psi|^2')
        fig.suptitle(name, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, name + '_planes.png'), dpi=110)
        plt.close(fig)
    except Exception as e:
        print('  figure planes failed:', repr(e))
    try:
        from scipy.signal import stft
        key = [k for k in groups if k[0] == 'site' and k[1] == 'rec2']
        if key:
            cols = groups[key[0]]
            j = max(cols, key=lambda j: np.abs(A[:, 0, j]).max())
            x = A[:, 0, j]
            dt = float(np.median(np.diff(tt)))
            f, tseg, Z = stft(np.conj(x), fs=1.0 / dt, nperseg=256, noverlap=224, return_onesided=False)
            o = np.argsort(f)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.pcolormesh(tseg / rs, 2 * math.pi * f[o], np.log10(np.maximum(np.abs(Z[o]), 1e-9)), shading='auto', cmap='magma')
            if meta['mode'] == 'radial':
                axN = d['axis_N'] if ('axis_N' in d.files and d['axis_N'].size) else None
                axf = C.AxisField(d['axis_r'], d['axis_u'], axN)
                r1, src_r, nhat, m = key[0][2], float(meta['src_r']), np.array(meta['nhat']), float(meta['ma'])
                oms = np.geomspace(0.05, 0.9, 12) if m == 0 else np.geomspace(1.1 * m, 0.9, 10)
                tr_ = []
                for om in oms:
                    try:
                        tr_.append(C.ray_time(src_r, r1, om, nhat, m, axf.u, axf.N, crossings=[c[0] for c in meta.get('crossings', [])]))
                    except Exception:
                        tr_.append(np.nan)
                ax.plot(np.array(tr_) / rs, oms, 'w--', lw=1, label='ray flight time (allocated field)')
                ax.legend(fontsize=7)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel('t / r_s')
            ax.set_ylabel('a omega')
            ax.set_title(f"{name}: site receiver at 2 r_s, source 0", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, name + '_spectrogram.png'), dpi=110)
            plt.close(fig)
    except Exception as e:
        print('  figure spectrogram failed:', repr(e))


def _f(x, nd=3):
    try:
        if x is None or not np.isfinite(float(x)):
            return 'nan'
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def write_tables(rows, trows, path):
    out = ["# Retarded cross-surface response (GAP10 build v1): tables. Frozen control background; times in units of r_s unless stated; "
           "every amplitude carries its parent certificate (rigorous in exact arithmetic).\n"]
    out.append("## 0. The printed number: R_ret = s_max^2(P_D U(t) B_X) over the sweep-end times, with the band 2 s delta + delta^2\n")
    out.append("| run | d | region D | input family | R_ret (max) | t/rs | band | resolved | R_ret (resolved max) | t/rs | optimizer |a_c| over the 4 site components |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in trows:
        if r['kind'] != 'region':
            continue
        opt = 'nan' if r['optimizer'] is None else ' '.join(f"{abs(x):.3f}" for x in r['optimizer'])
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['region']} | {r['family']} | {r['R_ret']:.4e} | {_f(r['t_at_R'] / r['rs'], 2)} | {r['band_at_R']:.1e} | "
                   f"{int(bool(r['resolved']))} | {r['R_ret_resolved']:.4e} | {_f(r['t_resolved'] / r['rs'], 2)} | {opt} |")
    out.append("\n## 0b. The receiver-filter transmissivity s_max^2(B_D^dagger U B_X), B_D the Lowdin-orthonormalized receiver packets at a radius, B_X the 4 site components\n")
    out.append("| run | d | receiver set | R (max) | t/rs | band | resolved | R (resolved max) | t/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in trows:
        if r['kind'] != 'filter':
            continue
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['region']} | {r['R_ret']:.4e} | {_f(r['t_at_R'] / r['rs'], 2)} | {r['band_at_R']:.1e} | {int(bool(r['resolved']))} | "
                   f"{r['R_ret_resolved']:.4e} | {_f(r['t_resolved'] / r['rs'], 2)} |")
    out.append("\n## A. Raw width-6a packet sources at the energy-matched raw width-6a receiver packets (omega_cmp = the source's actual <h>)\n")
    out.append("| run | d | a w label | <h> | h | r_rec/rs | t_first | t_peak | t_ray | t_peak - t_ray | t_nf | Gamma/rs | d/delta_w | peak |A| | u_peak/rs | u_ray/rs | eps_B(t_peak) | resolved | overlap0 | share_clean >1.5rs | norm err |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    matched = []
    for r in rows:
        if r.get('src_kind') != 'pkt' or r.get('rec_kind') != 'pkt' or not str(r.get('receiver', '')).startswith('rec'):
            continue
        if r.get('res_rec') != 'raw' or int(r.get('matched_to', -1)) != int(r.get('src_index', -2)):
            continue
        matched.append(r)
        if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        rs = r['rs']
        tray = r.get('t_ray', np.nan)
        ddw = (r['r_h'] - r['src_r']) / r['delta_omega'] if np.isfinite(r.get('delta_omega', np.nan)) and r['delta_omega'] > 0 else np.nan
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['omega_src']:g} | {_f(r['src_energy_mean'], 4)} | {'+' if r['hel_src'] > 0 else '-'} | {r['r_rec'] / rs:.3f} | "
                   f"{_f(r['t_first'] / rs)} | {_f(r['t_peak'] / rs)} | {_f(tray / rs)} | {_f((r['t_peak'] - tray) / rs)} | {_f(r.get('t_nf', np.nan) / rs)} | "
                   f"{_f(r.get('Gamma', np.nan) / rs, 3)} | {_f(ddw, 2)} | {r['peak']:.3e} | {_f(r.get('u_peak', np.nan) / rs)} | {_f(r.get('u_ray_rel', np.nan) / rs)} | "
                   f"{r['cert_peak']:.1e} | {int(bool(r['resolved']))} | {int(bool(r['initial_overlap']))} | {_f(r.get('share_clean_1.5rs'), 4)} | {r['norm_err']:.1e} |")
    out.append("\n## B. Site sources: the 4x4 block at the site receivers (Frobenius norm; certificate = root sum of squares of the 4 columns)\n")
    out.append("| run | receiver | r_rec/rs | t_first | |A| at t_first | eps_B(t_first) | resolved_first | t_peak | peak |A| | eps_B(t_peak) | u_first/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get('src_kind') != 'site_block':
            continue
        rs = r['rs']
        out.append(f"| {r['stem']} | {r['receiver']} | {r['r_rec'] / rs:.3f} | {_f(r['t_first'] / rs)} | {_f(r['amp_first'], 4)} | {r['cert_first']:.1e} | {int(bool(r.get('resolved_first', False)))} | "
                   f"{_f(r['t_peak'] / rs)} | {_f(r['peak'], 4)} | {r['cert_peak']:.1e} | {_f(r.get('u_first', np.nan) / rs)} |")
    out.append("\n## C. Site sources (component 0): arrival at the raw width-6a receiver-packet grid against frequency, receiver at 2 r_s\n")
    out.append("| run | d | a w | t_peak | t_ray | t_peak - t_ray | t_nf | peak |A| | eps_B(t_peak) | resolved | share_clean >1.5rs (all branches) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get('src_kind') != 'site' or r.get('src_comp') != 0 or r.get('rec_kind') != 'pkt' or r.get('receiver') != 'rec2' or int(r.get('matched_to', -1)) != -1:
            continue
        if r.get('res_rec') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        rs = r['rs']
        tray = r.get('t_ray', np.nan)
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['omega_rec']:g} | {_f(r['t_peak'] / rs)} | {_f(tray / rs)} | {_f((r['t_peak'] - tray) / rs)} | {_f(r.get('t_nf', np.nan) / rs)} | "
                   f"{r['peak']:.3e} | {r['cert_peak']:.1e} | {int(bool(r['resolved']))} | {_f(r.get('share_clean_1.5rs'), 4)} |")
    out.append("\n## D. Delay law per depth (ma = 0 runs): slope of t_peak against ln(1/<h>) over the raw packet sources at the matched receiver at 2 r_s, "
               "against the slope of the exact ray integral, of the normal form, and the asymptotes Gamma, 1/kappa_h, 2 r_s\n")
    out.append("| run | d | slope/rs (propagation) | slope/rs (ray) | slope/rs (normal form) | Gamma/rs | 1/(kappa_h rs) | 2 | points |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    byrun = {}
    for r in matched:
        if r.get('res_src') == 'raw' and abs(r['sigma_rec'] - 6.0) < 1e-9 and r.get('receiver') == 'rec2' and r['ma'] == 0.0:
            byrun.setdefault((r['stem'], r['hel_src']), []).append(r)
    for (stem, hel), rr in sorted(byrun.items()):
        if len(rr) < 2:
            continue
        rs = rr[0]['rs']
        x = np.log(1.0 / np.array([r['src_energy_mean'] for r in rr]))
        sp = np.polyfit(x, np.array([r['t_peak'] for r in rr]), 1)[0] / rs
        tr_ = np.array([r.get('t_ray', np.nan) for r in rr])
        sr = np.polyfit(x, tr_, 1)[0] / rs if np.all(np.isfinite(tr_)) else np.nan
        tn = np.array([r.get('t_nf', np.nan) for r in rr])
        sn = np.polyfit(x, tn, 1)[0] / rs if np.all(np.isfinite(tn)) else np.nan
        kh = rr[0]['kappa_h']
        out.append(f"| {stem} h{'+' if hel > 0 else '-'} | {rr[0]['depth']:g} | {_f(sp)} | {_f(sr)} | {_f(sn)} | {_f(rr[0].get('Gamma', np.nan) / rs)} | "
                   f"{_f(1.0 / (kh * rs)) if kh and np.isfinite(kh) and kh > 0 else 'nan'} | 2 | {len(rr)} |")
    out.append("\n## E. Anisotropy: t_peak[111] - t_peak[100] at the same d, a w label and declared receiver fraction (actual radii printed), against the ray's own lag\n")
    out.append("| pair | d | a w | frac | r_rec [100], [111] | lag/rs (propagation) | lag/rs (ray) | Gamma ln sqrt3 / rs |")
    out.append("|---|---|---|---|---|---|---|---|")
    index = {}
    for r in matched:
        if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        key = (r['stem'].replace('_111_', '_DIR_').replace('_100_', '_DIR_'), r['depth'], r['omega_src'], r['hel_src'], r.get('frac_rec', np.nan))
        index.setdefault(key, {})[r['direction']] = r
    for key, dd in sorted(index.items(), key=lambda kv: str(kv[0])):
        if '100' in dd and '111' in dd:
            a, b = dd['100'], dd['111']
            rs = a['rs']
            gam = a.get('Gamma', np.nan)
            out.append(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {key[4]} | {a['r_rec']:.2f}, {b['r_rec']:.2f} | {_f((b['t_peak'] - a['t_peak']) / rs)} | "
                       f"{_f((b.get('t_ray', np.nan) - a.get('t_ray', np.nan)) / rs)} | {_f(gam * math.log(math.sqrt(3.0)) / rs)} |")
    out.append("\n## F. Pairs differing in 1 control only (solver, parent half-width, ell, interior, sampling, profile): matched raw width-6a receivers; "
               "amplitude differences are to be compared with eps_A + eps_B (both parents' certificates)\n")
    out.append("| control | run A | run B | d | a w | r_rec/rs | retained A | retained B | t_peak A | t_peak B | |A| A | |A| B | ||A|-|B|| | eps_A + eps_B |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fld in ['solver', 'H', 'ell', 'Rb_frac', 'sampling', 'profile']:
        grp = {}
        for r in matched:
            if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
                continue
            key = tuple((k, r[k]) for k in ['rs', 'ma', 'ell', 'Rb_frac', 'sampling', 'profile', 'mode', 'H', 'direction', 'depth', 'solver'] if k != fld)
            key += (r['omega_src'], r['hel_src'], r.get('frac_rec', np.nan))
            grp.setdefault(key, []).append(r)
        for key, rr in grp.items():
            if len(rr) < 2:
                continue
            rr = sorted(rr, key=lambda r: str(r[fld]))
            for a, b in zip(rr[:-1], rr[1:]):
                if a[fld] == b[fld]:
                    continue
                rs = a['rs']
                out.append(f"| {fld} | {a['stem']} | {b['stem']} | {a['depth']:g} | {a['omega_src']:g} | {a['r_rec'] / rs:.2f} | {_f(a.get('rec_retained'), 4)} | {_f(b.get('rec_retained'), 4)} | "
                           f"{_f(a['t_peak'] / rs, 4)} | {_f(b['t_peak'] / rs, 4)} | {a['peak']:.4e} | {b['peak']:.4e} | {abs(a['peak'] - b['peak']):.2e} | {a['cert_peak'] + b['cert_peak']:.1e} |")
    out.append("\n## G. Inside-restricted twin against the raw source packet (same run, d, a w, h), matched raw width-6a receivers\n")
    out.append("| run | d | a w | r_rec/rs | exterior prob (raw) | t_peak raw | t_peak in | |A| raw | |A| in | u_peak raw/rs | u_peak in/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tw = {}
    for r in rows:
        if r.get('src_kind') != 'pkt' or r.get('rec_kind') != 'pkt' or not str(r.get('receiver', '')).startswith('rec') or r.get('res_rec') != 'raw':
            continue
        if abs(r['sigma_rec'] - 6.0) > 1e-9 or int(r.get('matched_to', -1)) != -1 or abs(r['omega_rec'] - r['omega_src']) > 1e-9:
            continue
        tw.setdefault((r['stem'], r['omega_src'], r['hel_src'], r.get('frac_rec', np.nan)), {})[r['res_src']] = r
    for key, dd in sorted(tw.items(), key=lambda kv: str(kv[0])):
        if 'raw' in dd and 'in' in dd:
            a, b = dd['raw'], dd['in']
            rs = a['rs']
            out.append(f"| {key[0]} | {a['depth']:g} | {key[1]:g} | {a['r_rec'] / rs:.2f} | {_f(a.get('src_exterior'), 3)} | {_f(a['t_peak'] / rs)} | {_f(b['t_peak'] / rs)} | "
                       f"{a['peak']:.3e} | {b['peak']:.3e} | {_f(a.get('u_peak', np.nan) / rs)} | {_f(b.get('u_peak', np.nan) / rs)} |")
    out.append("\n## H. Per source: certificate clean times, boundary-layer proxy, escaped shares, preparation mismatch, actual energy and weights\n")
    out.append("| run | source | t_clean(1e-3)/rs | t_clean(1e-2)/rs | t_clean(1e-1)/rs | t_bnd(1e-4)/rs | share_clean >1.5rs | share_peak >1.5rs | share_end >2rs | exterior prob | prep mismatch | predicted shift | <h> | energy std | upper band | outgoing pos. freq | lattice scale | zero mode |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in rows:
        k = (r['stem'], r.get('src_index'))
        if k in seen or r.get('src_kind') == 'site_block':
            continue
        seen.add(k)
        rs = r['rs']
        src = f"comp {r['src_comp']}" if r['src_kind'] == 'site' else f"pkt a w={r['omega_src']:g} h{'+' if r['hel_src'] > 0 else '-'} {r['res_src']}"
        out.append(f"| {r['stem']} | {src} | {_f(r['t_clean_1e-3'] / rs, 2)} | {_f(r['t_clean_1e-2'] / rs, 2)} | {_f(r['t_clean_1e-1'] / rs, 2)} | {_f(r.get('t_bnd_1e4', np.nan) / rs, 2)} | "
                   f"{_f(r.get('share_clean_1.5rs'), 4)} | {_f(r.get('share_peak_1.5rs'), 4)} | {_f(r.get('share_end_2rs'), 4)} | {_f(r.get('src_exterior'), 3)} | "
                   f"{_f(r.get('src_prep_mismatch'), 4)} | {_f(r.get('src_predicted_shift'), 4)} | {_f(r.get('src_energy_mean'), 4)} | {_f(r.get('src_energy_std'), 4)} | "
                   f"{_f(r.get('src_upper_band'), 3)} | {_f(r.get('src_out_pos_freq'), 3)} | {_f(r.get('src_lattice_scale'), 3)} | {_f(r.get('src_zero_mode'), 4)} |")
    out.append("\n## I. Boundary clocks (review eq. G10.7): u_peak differences between the matched receivers at 1.5, 2 and 3 r_s against the ray's "
               "int (1/c_g - 1/(1 - S_0)) dr; receiver independence of u is not assumed\n")
    out.append("| run | d | a w label | h | pair | Delta u_peak/rs (propagation) | Delta u/rs (ray) | difference/rs |")
    out.append("|---|---|---|---|---|---|---|---|")
    byp = {}
    for r in matched:
        if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        byp.setdefault((r['stem'], r['src_index']), {})[r.get('frac_rec', np.nan)] = r
    for key, dd in sorted(byp.items(), key=lambda kv: str(kv[0])):
        fr = sorted([f for f in dd if np.isfinite(f) and f >= 1.5])
        for f1, f2 in zip(fr[:-1], fr[1:]):
            a, b = dd[f1], dd[f2]
            rs = a['rs']
            du = b.get('u_peak', np.nan) - a.get('u_peak', np.nan)
            dr_ = b.get('u_ray_rel', np.nan) - a.get('u_ray_rel', np.nan)
            out.append(f"| {key[0]} | {a['depth']:g} | {a['omega_src']:g} | {'+' if a['hel_src'] > 0 else '-'} | {f1:g} -> {f2:g} rs | {_f(du / rs)} | {_f(dr_ / rs)} | {_f((du - dr_) / rs)} |")
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--csv', default='gap10_results.csv')
    ap.add_argument('--tcsv', default='gap10_transmissivity.csv')
    ap.add_argument('--md', default='gap10_results.md')
    ap.add_argument('--figs', default='')
    args = ap.parse_args()
    stems = sorted({p[:-4] for dd in args.dirs for p in glob.glob(os.path.join(dd, '*.npz'))})
    rows, trows = [], []
    for s in stems:
        print('[analysis]', s, flush=True)
        try:
            analyze(s, rows, trows, args.figs or None)
        except Exception as e:
            print('  failed:', repr(e))
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    tkeys = sorted({k for r in trows for k in r if k not in ('series', 'optimizer')})
    with open(args.tcsv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=tkeys, extrasaction='ignore')
        w.writeheader()
        w.writerows(trows)
    write_tables(rows, trows, args.md)
    print(f"[analysis] {len(rows)} readout rows -> {args.csv}; {len(trows)} transmissivity rows -> {args.tcsv}; tables -> {args.md}")


if __name__ == '__main__':
    main()
