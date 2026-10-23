#!/usr/bin/env python
"""csr_analysis.py -- arrival times, boundary-clock values with their spreads, certificates, escaped shares,
frequency content, ray-optics comparison, tables and figures for the runs of csr_run.py."""
import argparse
import csv
import glob
import json
import math
import os
import numpy as np
import csr_core as C


def load(stem):
    d = np.load(stem + '.npz')
    with open(stem + '.json') as f:
        meta = json.load(f)
    return d, meta


def group_key(rm):
    if rm['kind'] == 'site':
        return ('site', rm['role'], round(rm['r'], 3), np.nan, np.nan, 'raw')
    return ('pkt', rm['role'], round(rm['r'], 3), rm['omega'], rm['sigma'], rm['res'])


def arrival_stats(tt, amp, cert, thr_rel=1e-6):
    peak = float(amp.max())
    out = dict(t_first=np.nan, amp_first=np.nan, t_peak=np.nan, peak=peak, t10=np.nan, t50=np.nan,
               A0=float(amp[0]), initial_overlap=False, cert_first=np.nan, cert_peak=np.nan, resolved=False)
    if not np.isfinite(peak) or peak <= 0.0:
        return out
    ipk = int(amp.argmax())
    lm = np.flatnonzero((amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])) + 1
    cand = lm[amp[lm] > thr_rel * peak]
    i_first = int(cand[0]) if cand.size else ipk

    def refine(i):
        if 0 < i < len(amp) - 1:
            den = amp[i - 1] - 2.0 * amp[i] + amp[i + 1]
            if den < 0:
                off = float(np.clip(0.5 * (amp[i - 1] - amp[i + 1]) / den, -0.5, 0.5))
                return float(tt[i] + off * (tt[i + 1] - tt[i])), float(amp[i] - 0.125 * (amp[i - 1] - amp[i + 1]) ** 2 / den)
        return float(tt[i]), float(amp[i])

    tf, af = refine(i_first)
    tp, apk = refine(ipk)
    out.update(t_first=tf, amp_first=af, t_peak=tp, peak=apk,
               t10=float(tt[int(np.flatnonzero(amp > 0.1 * peak)[0])]), t50=float(tt[int(np.flatnonzero(amp > 0.5 * peak)[0])]),
               initial_overlap=bool(amp[0] > 1e-3 * peak), cert_first=float(cert[i_first]), cert_peak=float(cert[ipk]),
               resolved=bool(apk > 10.0 * cert[ipk]))
    return out


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


def profile_fn(d, meta):
    if meta['mode'] != 'radial':
        return lambda r: np.zeros_like(np.asarray(r, dtype=float))
    if meta['sampling'] == 'averaged' and d['prof_rg'].size > 0:
        rg, V = d['prof_rg'], d['prof_V']
        return lambda r: np.interp(np.asarray(r, dtype=float), rg, V)
    rs, Rb = float(meta['rs']), float(meta['Rb'])
    return lambda r: C.u_continuum(np.asarray(r, dtype=float), rs, Rb)


def analyze(stem, rows, figdir=None):
    d, meta = load(stem)
    tt, A, cert = d['times'], d['A'], d['cert']
    rs, m = float(meta['rs']), float(meta['ma'])
    nhat = np.array(meta['nhat'])
    src_r = float(meta['src_r'])
    sweep_t, shares, bnd = d['sweep_t'], d['shares'], d['bnd']
    u_fn = profile_fn(d, meta)
    kappa_h = float(meta.get('kappa_h', float('nan')) or float('nan'))
    radial = meta['mode'] == 'radial'
    rec_meta = meta['rec_meta']
    groups = {}
    for j, rm in enumerate(rec_meta):
        groups.setdefault(group_key(rm), []).append(j)
    base = dict(stem=os.path.basename(stem), rs=rs, ma=m, ell=meta['ell'], Rb_frac=meta['Rb_frac'], sampling=meta['sampling'],
                mode=meta['mode'], H=meta['H'], direction=meta['direction'], depth=meta['depth'], src_r=src_r, u_src=meta['u_src'],
                solver=meta['solver'], r_h=meta.get('r_h', float('nan')), kappa_h=kappa_h,
                norm_err=float(np.max(np.abs(d['norms'] - 1.0))),
                energy_err=float(np.max(np.abs(d['energy'] - d['energy'][0][None, :]))), napply=meta.get('napply', -1))
    names = [nm for nm, _ in meta['share_radii']]
    cache = {}

    def t_ray(r1, om):
        key = (round(r1, 6), om)
        if key not in cache:
            try:
                tr_ = C.ray_time(src_r, r1, om, nhat, m, u_fn)
                cr = C.crossings_between(u_fn, src_r, r1)
                cache[key] = (tr_, float(sum(c[1] for c in cr)), len(cr))
            except Exception:
                cache[key] = (float('nan'), float('nan'), 0)
        return cache[key]

    def chart(row, r1, st, rm):
        if radial and r1 > rs and np.isfinite(st['t_first']):
            row.update(T_first=C.chart_T(st['t_first'], r1, rs), u_first=C.chart_u(st['t_first'], r1, rs),
                       T_peak=C.chart_T(st['t_peak'], r1, rs), u_peak=C.chart_u(st['t_peak'], r1, rs),
                       T_rms=rm.get('T_rms', np.nan), u_rms=rm.get('u_rms', np.nan))

    for b, sd in enumerate(meta['src']):
        cb = cert[:, b]
        common = dict(base)
        common.update(src_index=b, src_kind=sd['kind'], src_comp=sd.get('comp', -1), omega_src=sd.get('omega', np.nan),
                      hel_src=sd.get('hel', 0), res_src=sd.get('res', 'raw'), k_src=sd.get('k', np.nan),
                      src_exterior=sd.get('exterior_probability', np.nan), src_retained=sd.get('retained', np.nan),
                      src_energy_mean=sd.get('energy_mean', np.nan), src_energy_std=sd.get('energy_std', np.nan))
        bw = sd.get('band', None)
        if bw:
            common.update(src_upper_band=bw['upper_band'], src_pos_freq=bw['positive_frequency'],
                          src_out_pos_freq=bw['outgoing_positive_frequency'], src_lattice_scale=bw['lattice_scale'])
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
        for key, cols in groups.items():
            rm = rec_meta[cols[0]]
            amp = np.sqrt(sum(np.abs(A[:, b, j]) ** 2 for j in cols))
            st = arrival_stats(tt, amp, cb)
            r1 = float(rm['r'])
            row = dict(common)
            row.update(receiver=key[1], rec_kind=key[0], r_rec=r1, omega_rec=key[3], sigma_rec=key[4], res_rec=key[5],
                       rec_retained=rm.get('retained', np.nan), rec_clipped=rm.get('clipped', False),
                       rec_exterior=rm.get('exterior_probability', np.nan), **st)
            chart(row, r1, st, rm)
            if radial and key[0] == 'pkt' and key[1].startswith('rec'):
                om = key[3] if sd['kind'] == 'site' else sd['omega']
                tr_, gam, ncr = t_ray(r1, om)
                row.update(t_ray=tr_, Gamma=gam, n_crossings=ncr, omega_ray=om,
                           t_ray_log=(gam * math.log(1.0 / om) if np.isfinite(gam) and gam > 0 else np.nan))
            jdom = max(cols, key=lambda j: np.abs(A[:, b, j]).max())
            if key[0] == 'site':
                w, S, res = spectrum(tt, A[:, b, jdom], st['t_first'] - rs, st['t_first'] + 3 * rs)
            else:
                w, S, res = spectrum(tt, A[:, b, jdom], st['t_peak'] - 2 * rs, st['t_peak'] + 2 * rs)
            if w is not None and S.sum() > 0:
                row.update(omega_peak=float(w[np.argmax(S)]), omega_centroid=float(np.sum(w * S) / np.sum(S)), omega_resolution=res)
            row['phase_at_peak'] = float(np.angle(A[int(np.argmin(np.abs(tt - st['t_peak']))), b, jdom])) if np.isfinite(st['t_peak']) else np.nan
            rows.append(row)
    site_b = [b for b, sd in enumerate(meta['src']) if sd['kind'] == 'site']
    if len(site_b) == 4:
        common = dict(base)
        common.update(src_index=-1, src_kind='site_block', src_comp=-1, omega_src=np.nan, hel_src=0, res_src='raw')
        for key, cols in groups.items():
            if key[0] != 'site':
                continue
            rm = rec_meta[cols[0]]
            amp = np.sqrt(sum(np.abs(A[:, b, j]) ** 2 for b in site_b for j in cols))
            st = arrival_stats(tt, amp, cert[:, site_b].max(1))
            r1 = float(rm['r'])
            row = dict(common)
            row.update(receiver=key[1], rec_kind='site', r_rec=r1, omega_rec=np.nan, sigma_rec=np.nan, res_rec='raw', **st)
            chart(row, r1, st, rm)
            rows.append(row)
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
                if sd['kind'] == 'pkt' and abs(key[3] - sd['omega']) > 1e-9:
                    continue
                if sd['kind'] == 'site' and key[1] != 'rec2':
                    continue
                amp = np.sqrt(sum(np.abs(A[:, b, j]) ** 2 for j in cols))
                ax.semilogy(tt / rs, np.maximum(amp, 1e-9), lw=0.8, label=f"{key[1][3:]} rs, a w={key[3]:g}")
            ax.semilogy(tt / rs, np.maximum(cert[:, b], 1e-9), 'k--', lw=0.8, label='parent certificate')
            ax.set_ylim(1e-6, 1.5)
            ax.grid(alpha=0.3)
            ttl = f"source {b}: " + (f"packet a w={sd['omega']:g} h{'+' if sd['hel'] > 0 else '-'} {sd['res']}" if sd['kind'] == 'pkt' else f"site component {sd['comp']}")
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
        ax.set_ylabel('share of the norm')
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
        fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, P, ttl in ((axs[0], pz, 'plane z = 0'), (axs[1], pdg, 'plane y = z ([100] and [111])')):
            im = ax.imshow(np.log10(np.maximum(P.T, 1e-16)), origin='lower', extent=[-H / rs, H / rs, -H / rs, H / rs],
                           vmin=-12, vmax=math.log10(max(float(P.max()), 1e-16)), cmap='viridis')
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color='w', lw=0.6))
            ax.set_title(f"{ttl}, t = {d['sweep_t'][-1] / rs:.0f} r_s, source 0", fontsize=8)
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
                u_fn = profile_fn(d, meta)
                r1, src_r, nhat, m = key[0][2], float(meta['src_r']), np.array(meta['nhat']), float(meta['ma'])
                oms = np.geomspace(0.05, 0.9, 12) if m == 0 else np.geomspace(1.1 * m, 0.9, 10)
                tr_ = [C.ray_time(src_r, r1, om, nhat, m, u_fn, dr=0.1) for om in oms]
                ax.plot(np.array(tr_) / rs, oms, 'w--', lw=1, label='ray flight time')
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


def write_tables(rows, path):
    out = ["# Retarded cross-surface response: tables (times in units of r_s unless stated)\n"]
    out.append("## A. Packet sources (raw, width 6a) at the matched-frequency raw width-6a receiver packets\n")
    out.append("| run | d | a w | h | r_rec/rs | t_first | t_peak | t_ray | t_peak - t_ray | Gamma/rs | peak |A| | u_peak/rs | u_rms/rs | eps_B(t_peak) | resolved | overlap0 | share_clean >1.5rs | norm err |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    matched = []
    for r in rows:
        if r.get('src_kind') != 'pkt' or r.get('rec_kind') != 'pkt' or not str(r.get('receiver', '')).startswith('rec'):
            continue
        if r.get('res_rec') != 'raw' or abs(r['omega_rec'] - r['omega_src']) > 1e-9:
            continue
        matched.append(r)
        if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        rs = r['rs']
        tray = r.get('t_ray', np.nan)
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['omega_src']:g} | {'+' if r['hel_src'] > 0 else '-'} | {r['r_rec'] / rs:.3f} | "
                   f"{_f(r['t_first'] / rs)} | {_f(r['t_peak'] / rs)} | {_f(tray / rs)} | {_f((r['t_peak'] - tray) / rs)} | "
                   f"{_f(r.get('Gamma', np.nan) / rs, 3)} | {r['peak']:.3e} | {_f(r.get('u_peak', np.nan) / rs)} | {_f(r.get('u_rms', np.nan) / rs)} | "
                   f"{r['cert_peak']:.1e} | {int(bool(r['resolved']))} | {int(bool(r['initial_overlap']))} | {_f(r.get('share_clean_1.5rs'), 4)} | {r['norm_err']:.1e} |")
    out.append("\n## B. Site sources: the 4x4 block at the site receivers (Frobenius norm)\n")
    out.append("| run | receiver | r_rec/rs | t_first | |A| at t_first | eps_B(t_first) | t_peak | peak |A| | u_first/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get('src_kind') != 'site_block':
            continue
        rs = r['rs']
        out.append(f"| {r['stem']} | {r['receiver']} | {r['r_rec'] / rs:.3f} | {_f(r['t_first'] / rs)} | {_f(r['amp_first'], 4)} | {r['cert_first']:.1e} | "
                   f"{_f(r['t_peak'] / rs)} | {_f(r['peak'], 4)} | {_f(r.get('u_first', np.nan) / rs)} |")
    out.append("\n## C. Site sources (component 0): arrival at the raw width-6a receiver packets against frequency, receiver at 2 r_s\n")
    out.append("| run | a w | t_peak | t_ray | t_peak - t_ray | peak |A| | eps_B(t_peak) | share_clean >1.5rs (all branches) |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get('src_kind') != 'site' or r.get('src_comp') != 0 or r.get('rec_kind') != 'pkt' or r.get('receiver') != 'rec2':
            continue
        if r.get('res_rec') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        rs = r['rs']
        tray = r.get('t_ray', np.nan)
        out.append(f"| {r['stem']} | {r['omega_rec']:g} | {_f(r['t_peak'] / rs)} | {_f(tray / rs)} | {_f((r['t_peak'] - tray) / rs)} | "
                   f"{r['peak']:.3e} | {r['cert_peak']:.1e} | {_f(r.get('share_clean_1.5rs'), 4)} |")
    out.append("\n## D. Delay law: slope of t_peak against ln(1/(a w)) over the source frequencies, raw width-6a receiver at 2 r_s (raw packet sources)\n")
    out.append("| run | slope / rs (propagation) | slope / rs (ray integral) | Gamma/rs | 1/(kappa_h rs) | 2 |")
    out.append("|---|---|---|---|---|---|")
    byrun = {}
    for r in matched:
        if r.get('res_src') == 'raw' and abs(r['sigma_rec'] - 6.0) < 1e-9 and r.get('receiver') == 'rec2':
            byrun.setdefault((r['stem'], r['hel_src']), []).append(r)
    for (stem, hel), rr in sorted(byrun.items()):
        if len(rr) < 2:
            continue
        rs = rr[0]['rs']
        x = np.log(1.0 / np.array([r['omega_src'] for r in rr]))
        sp = np.polyfit(x, np.array([r['t_peak'] for r in rr]), 1)[0] / rs
        tr_ = np.array([r.get('t_ray', np.nan) for r in rr])
        sr = np.polyfit(x, tr_, 1)[0] / rs if np.all(np.isfinite(tr_)) else np.nan
        kh = rr[0]['kappa_h']
        out.append(f"| {stem} h{'+' if hel > 0 else '-'} | {_f(sp)} | {_f(sr)} | {_f(rr[0].get('Gamma', np.nan) / rs)} | "
                   f"{_f(1.0 / (kh * rs)) if kh and np.isfinite(kh) and kh > 0 else 'nan'} | 2 |")
    out.append("\n## E. Anisotropy: t_peak[111] - t_peak[100] at the same d, a w and receiver fraction, against Gamma ln sqrt(3)\n")
    out.append("| pair | d | a w | r_rec/rs | lag / rs (propagation) | lag / rs (ray) | Gamma ln sqrt3 / rs |")
    out.append("|---|---|---|---|---|---|---|")
    index = {}
    for r in matched:
        if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        key = (r['stem'].replace('_111_', '_DIR_').replace('_100_', '_DIR_'), r['depth'], r['omega_src'], r['hel_src'], round(r['r_rec'] / r['rs'], 2))
        index.setdefault(key, {})[r['direction']] = r
    for key, dd in sorted(index.items()):
        if '100' in dd and '111' in dd:
            a, b = dd['100'], dd['111']
            rs = a['rs']
            gam = a.get('Gamma', np.nan)
            out.append(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {key[4]} | {_f((b['t_peak'] - a['t_peak']) / rs)} | "
                       f"{_f((b.get('t_ray', np.nan) - a.get('t_ray', np.nan)) / rs)} | {_f(gam * math.log(math.sqrt(3.0)) / rs)} |")
    out.append("\n## F. Pairs differing in 1 control only (solver, parent half-width, ell, interior, sampling): matched raw width-6a receivers\n")
    out.append("| control | run A | run B | d | a w | r_rec/rs | retained A | retained B | t_peak A | t_peak B | |A| A | |A| B | eps_B A | eps_B B |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fld in ['solver', 'H', 'ell', 'Rb_frac', 'sampling']:
        groups = {}
        for r in matched:
            if r.get('res_src') != 'raw' or abs(r['sigma_rec'] - 6.0) > 1e-9:
                continue
            key = tuple((k, r[k]) for k in ['rs', 'ma', 'ell', 'Rb_frac', 'sampling', 'mode', 'H', 'direction', 'depth', 'solver'] if k != fld)
            key += (r['omega_src'], r['hel_src'], round(r['r_rec'] / r['rs'], 2))
            groups.setdefault(key, []).append(r)
        for key, rr in groups.items():
            if len(rr) < 2:
                continue
            rr = sorted(rr, key=lambda r: str(r[fld]))
            for a, b in zip(rr[:-1], rr[1:]):
                if a[fld] == b[fld]:
                    continue
                rs = a['rs']
                out.append(f"| {fld} | {a['stem']} | {b['stem']} | {a['depth']:g} | {a['omega_src']:g} | {a['r_rec'] / rs:.2f} | "
                           f"{_f(a.get('rec_retained'), 4)} | {_f(b.get('rec_retained'), 4)} | "
                           f"{_f(a['t_peak'] / rs, 4)} | {_f(b['t_peak'] / rs, 4)} | {a['peak']:.4e} | {b['peak']:.4e} | {a['cert_peak']:.1e} | {b['cert_peak']:.1e} |")
    out.append("\n## G. Inside-restricted twin against the raw source packet (same run, d, a w, h), matched raw width-6a receivers\n")
    out.append("| run | d | a w | r_rec/rs | exterior prob (raw) | t_peak raw | t_peak in | |A| raw | |A| in | u_peak raw/rs | u_peak in/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tw = {}
    for r in matched:
        if abs(r['sigma_rec'] - 6.0) > 1e-9:
            continue
        tw.setdefault((r['stem'], r['omega_src'], r['hel_src'], round(r['r_rec'] / r['rs'], 2)), {})[r['res_src']] = r
    for key, dd in sorted(tw.items()):
        if 'raw' in dd and 'in' in dd:
            a, b = dd['raw'], dd['in']
            rs = a['rs']
            out.append(f"| {key[0]} | {a['depth']:g} | {key[1]:g} | {key[3]} | {_f(a.get('src_exterior'), 3)} | {_f(a['t_peak'] / rs)} | {_f(b['t_peak'] / rs)} | "
                       f"{a['peak']:.3e} | {b['peak']:.3e} | {_f(a.get('u_peak', np.nan) / rs)} | {_f(b.get('u_peak', np.nan) / rs)} |")
    out.append("\n## H. Per source: certificate clean times, boundary-layer proxy, escaped shares and source weights\n")
    out.append("| run | source | t_clean(1e-3)/rs | t_clean(1e-2)/rs | t_clean(1e-1)/rs | t_bnd(1e-4)/rs | share_clean >1.5rs | share_peak >1.5rs | share_end >2rs | exterior prob | upper band | outgoing pos. freq |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    seen = set()
    for r in rows:
        k = (r['stem'], r.get('src_index'))
        if k in seen or r.get('src_kind') == 'site_block':
            continue
        seen.add(k)
        rs = r['rs']
        src = f"comp {r['src_comp']}" if r['src_kind'] == 'site' else f"pkt a w={r['omega_src']:g} h{'+' if r['hel_src'] > 0 else '-'} {r['res_src']}"
        out.append(f"| {r['stem']} | {src} | {_f(r['t_clean_1e-3'] / rs, 2)} | {_f(r['t_clean_1e-2'] / rs, 2)} | {_f(r['t_clean_1e-1'] / rs, 2)} | "
                   f"{_f(r.get('t_bnd_1e4', np.nan) / rs, 2)} | {_f(r.get('share_clean_1.5rs'), 4)} | {_f(r.get('share_peak_1.5rs'), 4)} | "
                   f"{_f(r.get('share_end_2rs'), 4)} | {_f(r.get('src_exterior'), 3)} | {_f(r.get('src_upper_band'), 3)} | {_f(r.get('src_out_pos_freq'), 3)} |")
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--csv', default='csr_results.csv')
    ap.add_argument('--md', default='csr_results.md')
    ap.add_argument('--figs', default='')
    args = ap.parse_args()
    stems = sorted({p[:-4] for dd in args.dirs for p in glob.glob(os.path.join(dd, '*.npz'))})
    rows = []
    for s in stems:
        print('[analysis]', s, flush=True)
        try:
            analyze(s, rows, args.figs or None)
        except Exception as e:
            print('  failed:', repr(e))
    keys = sorted({k for r in rows for k in r})
    with open(args.csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    write_tables(rows, args.md)
    print(f"[analysis] {len(rows)} rows -> {args.csv}, {args.md}")


if __name__ == '__main__':
    main()
