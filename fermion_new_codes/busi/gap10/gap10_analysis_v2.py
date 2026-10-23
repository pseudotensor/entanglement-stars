#!/usr/bin/env python
"""gap10_analysis_v2.py -- gap10_analysis.py (same command line, same output files, the same tables 0 to I) with the
repairs described in busi/notes/GAP10_analysis_fix_v1.md. Reads the runs of gap10_run_v2.py (the opt2 core writes the
same fields). Replace the script name in the analysis line of matrix.sh; nothing else changes.

Fixes against gap10_analysis.py:
  FIX 1 (table B, the site receivers).
    (a) The receiver group key of a site held float('nan') entries; nan != nan, so each of the 4 spinor components of a
        site receiver became its own group: the "4 x 4 block" rows were 4 identical single-column rows (|A| = one
        column's norm; 1.0000 at the source site, where the Frobenius norm of the identity block is 2). The key now
        holds None and the block is the full 4 x 4 block of the build note (section 'per source and receiver group':
        the group 2-norm over a site's 4 components); its Frobenius norm and its largest singular value are printed.
    (b) The null [111] rows are not an analysis error. The sine lattice with the current as a 2-site hop commutes with
        the 3 sublattice operators (-1)^{x_j} gamma_5 beta alpha_j (j = x, y, z) for any site fields N, v and on the
        open box, so the site block between 2 sites whose coordinate differences are odd on 2 or 3 axes lies in the
        sector generated only by products of 2 or more distinct alphas. On the homogeneous infinite lattice
        exp(-i h t) = e^{-i drag t} [cos(N E t) - i sin(N E t) h_0(k)/(N E)] contains only 1, alpha_i and beta, and the
        block vanishes identically (also on the open box when v = 0); with a current on the open box it is sourced at
        the 2 planes nearest the faces only (box error, inside the parent certificate); on varying fields it is of the
        order of the field gradients (the weighted hops no longer commute), EXCEPT that the sector coefficient is
        antisymmetric under the transposition of 2 odd axes, so for cubic-symmetric (radial, point-sampled) fields and
        2 sites both fixed by such a transposition, the T7 case (sites on the [111] axis, all 3 axes odd), the block
        vanishes exactly, gradients or not (the T7 rows print 0.0000 with resolved_first 0: amplitudes at the 1e-17
        level against a certificate of 4e-18). The selftest verifies each statement on a 7^3 box.
        The T7 [111] site receivers at 1.25, 1.5 and 2 r_s, (21,21,21), (25,25,25), (33,33,33) against the sources
        (12,12,12) and (8,8,8), differ by odd numbers on all 3 axes and read nothing; the 3 r_s receiver (50,50,50)
        differs evenly and reads. The table now prints the parity class of every site receiver and its amplitudes in
        scientific notation; the repair of the rows is a rerun with even-difference receivers (see the note).
    (c) Table B's t_peak now carries the resolved-window maximum of FIX 2 and the global one beside it.
  FIX 2 (table C and every arrival time: the box-face reflection). t_peak was the global maximum of the series over the
    20 r_s. After the certificate has grown past the amplitude the series is the box's own (the return from the +x face
    reaches the 2 r_s receiver after 2 (H - r_rec) + (r_rec - r_src) = 128 a = 4.4 r_s on [100] at d 2, 4.6 r_s at d 8),
    and its maximum was printed with eps_B 3.5 to 4.5 against peaks 0.006 to 0.012. The arrival statistics now also
    take the maximum over the resolved readouts (|A| > 10 eps_B): t_peak_res, peak_res, cert_peak_res, t_cert (the
    last resolved readout). When that maximum sits at the end of the resolved window with the amplitude still rising
    (the certificate overtook a rising flank) it is no maximum: t_peak_res = nan, trunc_res = True. Tables A, B, C, D,
    E, G and I read the resolved maximum; the global maximum and its certificate stay in the csv and in tables B and C.
  FIX 3 (table F, the control pairs). The pairing key omitted the preparation (meta 'prep': fixed or matched) and the
    common clipping (meta 'common_half'); the T7 twins fell into the same group under every control and were skipped as
    equal, and the sort by str() ordered numeric controls lexicographically (124 < 60 < 88). 'prep' is now a control
    (fixed against matched at the same label), 'common_half' is in the key, numeric controls sort numerically.
  Also: the '2 r_s receiver' is selected by its declared fraction (|frac - 2| < 0.1), not by the literal role name, and
    the anisotropy and clock pairs match fractions within 0.1, so that a rerun with moved [111] receivers is readable;
    table 0b prints R(0), the initial-overlap value; the packet width is read from the run (meta 'sigma') instead of
    the literal 6; the core is gap10_core_v2 when present (its functions used here are identical to gap10_core's).
  --selftest runs synthetic checks of the 3 fixes and, when torch and the core import, the lattice-symmetry statement
    of FIX 1b on a small random-field box (commutators, sectors, the exact null on homogeneous fields).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import numpy as np
try:
    import gap10_core_v2 as C
except ImportError:  # the v1 core: identical for everything used here
    import gap10_core as C

CANON_FRACS = (1.25, 1.5, 2.0, 3.0)
FRAC_TOL = 0.1
CONTROLS = ['solver', 'H', 'ell', 'Rb_frac', 'sampling', 'profile', 'prep']
PAIR_KEY = ['rs', 'ma', 'ell', 'Rb_frac', 'sampling', 'profile', 'mode', 'H', 'direction', 'depth', 'solver', 'prep', 'common_half']


def load(stem):
    d = np.load(stem + '.npz')
    with open(stem + '.json') as f:
        meta = json.load(f)
    return d, meta


def group_key(rm):
    """FIX 1a: no float('nan') in the key (nan != nan made every site receiver component its own group)."""
    if rm['kind'] == 'site':
        return ('site', rm['role'], round(float(rm['r']), 3), None, None, 'raw', -1)
    return ('pkt', rm['role'], round(float(rm['r']), 3), round(float(rm['omega']), 6), float(rm['sigma']), rm['res'], int(rm.get('matched_to', -1)))


def build_groups(rec_meta):
    groups = {}
    for j, rm in enumerate(rec_meta):
        groups.setdefault(group_key(rm), []).append(j)
    return groups


def canon_frac(f):
    """The declared receiver fraction snapped to the canonical set within FRAC_TOL (moved sites pair with the others)."""
    try:
        f = float(f)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(f):
        return f
    for cf in CANON_FRACS:
        if abs(f - cf) < FRAC_TOL:
            return cf
    return f


def is_frac(f, target):
    try:
        return abs(float(f) - target) < FRAC_TOL
    except (TypeError, ValueError):
        return False


def _k(x):
    """Dictionary-key form of a value: nan -> None (nan != nan would split equal keys)."""
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return None
        if isinstance(x, (np.floating, np.integer)):
            x = float(x)
            return None if not np.isfinite(x) else x
    except (TypeError, ValueError):
        pass
    return x


def site_frac(meta, rm):
    """The declared fraction of a site receiver from meta['rec_sites'] (matched by position); nan for the source site."""
    pos = rm.get('pos')
    for rsd in meta.get('rec_sites', []) or []:
        if pos is not None and [int(p) for p in rsd['pos']] == [int(p) for p in pos]:
            return float(rsd['frac'])
    return float('nan')


def site_parity(src_pos, rec_pos):
    """FIX 1b: the number of axes on which the receiver site differs from the source site by an odd number of spacings;
    2 or 3 is the null class of the sublattice symmetry (exactly null on the homogeneous infinite lattice; gradient- and
    face-sourced otherwise)."""
    dd = np.asarray(rec_pos, dtype=int) - np.asarray(src_pos, dtype=int)
    n_odd = int(np.sum(np.abs(dd) % 2 == 1))
    return n_odd, ['even', '1 odd', '2 odd (null class)', '3 odd (null class)'][n_odd]


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
    """v1 statistics (t_first, the global t_peak and its certificate, resolved = global peak > 10 eps_B) plus FIX 2:
    the maximum over the resolved readouts {t: |A|(t) > 10 eps_B(t)} as t_peak_res / peak_res / cert_peak_res, the last
    resolved readout t_cert, and trunc_res = True (t_peak_res = nan) when that maximum is the window's last point with the
    amplitude still rising at the next readout: the certificate cut a rising flank, there is no certified maximum."""
    peak = float(amp.max())
    out = dict(t_first=np.nan, amp_first=np.nan, t_peak=np.nan, peak=peak, t10=np.nan, t50=np.nan, A0=float(amp[0]), initial_overlap=False,
               cert_first=np.nan, cert_peak=np.nan, resolved=False, resolved_first=False, i_first=-1, i_peak=-1,
               t_peak_res=np.nan, peak_res=np.nan, cert_peak_res=np.nan, i_peak_res=-1, trunc_res=False, t_cert=np.nan)
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
    ok = amp > 10.0 * np.asarray(cert_col, dtype=float)   # FIX 2: the resolved readouts
    if ok.any():
        idx = np.flatnonzero(ok)
        i_res = int(idx[np.argmax(amp[idx])])
        out['t_cert'] = float(tt[idx[-1]])
        trunc = (i_res + 1 < tt.size) and (not ok[i_res + 1]) and (amp[i_res + 1] >= amp[i_res])
        out['trunc_res'] = bool(trunc)
        if not trunc:
            out.update(t_peak_res=float(tt[i_res]), peak_res=float(amp[i_res]), cert_peak_res=float(cert_col[i_res]), i_peak_res=i_res)
    return out


def refine(d, meta, b, cols, t_c, dt, nfine=50):
    """Re-locate an extremum on [t_c - dt, t_c + dt] by exact evaluation at 2 nfine + 1 points."""
    ts = np.linspace(max(t_c - dt, 0.0), min(t_c + dt, float(meta['tmax'])), 2 * nfine + 1)
    A = amplitudes_at(d, meta, ts, b, cols)
    amp = np.sqrt((np.abs(A) ** 2).sum(1))
    i = int(np.argmax(amp))
    return float(ts[i]), float(amp[i]), bool(i in (0, len(ts) - 1))


def refine_block(d, meta, site_b, cols, t_c, dt, nfine=50):
    """FIX 1a: the same re-location for the 4 x 4 site block (Frobenius norm); returns (t, |A|_F, s_max, edge)."""
    ts = np.linspace(max(t_c - dt, 0.0), min(t_c + dt, float(meta['tmax'])), 2 * nfine + 1)
    blk = np.stack([amplitudes_at(d, meta, ts, b, cols) for b in site_b], 1)   # (times, 4 sources, components)
    frob = np.sqrt((np.abs(blk) ** 2).sum((1, 2)))
    i = int(np.argmax(frob))
    return float(ts[i]), float(frob[i]), float(np.linalg.svd(blk[i], compute_uv=False)[0]), bool(i in (0, len(ts) - 1))


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
    src_pos = [int(p) for p in meta.get('src_pos', [0, 0, 0])]
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
    groups = build_groups(rec_meta)
    base = dict(stem=os.path.basename(stem), rs=rs, ma=m, ell=meta['ell'], Rb_frac=meta['Rb_frac'], sampling=meta['sampling'], profile=meta['profile'],
                mode=meta['mode'], H=meta['H'], direction=meta['direction'], depth=meta['depth'], src_r=src_r, u_src=meta['u_src'], solver=meta['solver'],
                tau=meta['tau'], r_h=r_h, kappa_h=kappa_h, n_crossings_total=len(crossings),
                norm_err=float(np.max(np.abs(d['norms'] - 1.0))), energy_err=float(np.max(np.abs(d['energy'] - d['energy'][0][None, :]))),
                norm_excess=(float(np.max(d['norm_excess'])) if d['norm_excess'].size else np.nan), napply=meta.get('napply', -1),
                prep=meta.get('prep', 'fixed'), common_half=int(meta.get('common_half', 0) or 0), tag=meta.get('tag', ''),   # FIX 3
                sigma_run=float(meta.get('sigma', 6.0)), core=meta.get('core', ''))
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
            for nm in ('t_first', 't_peak', 't_peak_res'):   # FIX 2: the chart values of the resolved maximum too
                if np.isfinite(row.get(nm, np.nan)):
                    row['T_' + nm[2:]] = C.chart_T(row[nm], r1, rs)
                    row['u_' + nm[2:]] = C.chart_u(row[nm], r1, rs)
            row.update(T_rms_kinematic=rm.get('T_rms', np.nan), u_rms_kinematic=rm.get('u_rms', np.nan))
            if np.isfinite(row.get('t_ray', np.nan)):
                row['u_ray_rel'] = row['t_ray'] - float(C.clock_offsets(r1, rs)[1])

    def refine_all(st, b, cols, cb):
        """Re-locate t_first, the global t_peak and the resolved t_peak_res on the fine grid (FIX 2 adds the third)."""
        tf, af, edge_f = refine(d, meta, b, cols, st['t_first'], dt_read)
        tp, ap, edge_p = refine(d, meta, b, cols, st['t_peak'], dt_read)
        st.update(t_first=tf, amp_first=af, t_peak=tp, peak=ap, cert_first=cert_at(tt, cb, tf), cert_peak=cert_at(tt, cb, tp),
                  edge_first=edge_f, edge_peak=edge_p)
        st['resolved'] = bool(ap > 10.0 * st['cert_peak'])
        st['resolved_first'] = bool(af > 10.0 * st['cert_first'])
        if np.isfinite(st['t_peak_res']):
            tr, ar, edge_r = refine(d, meta, b, cols, st['t_peak_res'], dt_read)
            st.update(t_peak_res=tr, peak_res=ar, cert_peak_res=cert_at(tt, cb, tr), edge_peak_res=edge_r)

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
                refine_all(st, b, cols, cb)
            frac = float(rm.get('frac', np.nan)) if key[0] == 'pkt' else site_frac(meta, rm)
            row.update(receiver=key[1], rec_kind=key[0], r_rec=r1, omega_rec=(np.nan if key[3] is None else key[3]),
                       sigma_rec=(np.nan if key[4] is None else key[4]), res_rec=key[5], matched_to=key[6], frac_rec=frac, frac_canon=canon_frac(frac),
                       rec_retained=rm.get('retained', np.nan), rec_clipped=rm.get('clipped', False), rec_exterior=rm.get('exterior_probability', np.nan),
                       n_cols=len(cols), **{k: v for k, v in st.items() if not k.startswith('i_')})
            if key[0] == 'site':   # FIX 1b: the parity class of the site receiver against the source site
                n_odd, plab = site_parity(src_pos, rm.get('pos', src_pos))
                row.update(n_odd_diff=n_odd, parity=plab)
            if radial and key[0] == 'pkt' and key[1].startswith('rec'):
                om_cmp = om_act if (sd['kind'] == 'pkt' and np.isfinite(om_act)) else key[3]
                tr_, gam, ncr = t_ray(r1, om_cmp)
                row.update(t_ray=tr_, Gamma=gam, n_crossings=ncr, omega_cmp=om_cmp, t_nf=t_nf(r1, om_cmp),
                           t_ray_log=(gam * math.log(1.0 / om_cmp) if np.isfinite(gam) and gam > 0 else np.nan),
                           delta_omega=(C.nf_delta(om_cmp, gamma, b_nf) if np.isfinite(gamma) and gamma > 0 else np.nan))
            chart(row, r1, rm)
            jdom = max(cols, key=lambda j: np.abs(A[:, b, j]).max())
            if np.isfinite(st['t_first']):
                tp_ref = st['t_peak_res'] if np.isfinite(st['t_peak_res']) else st['t_peak']   # FIX 2: the window follows the certified peak
                if key[0] == 'site':
                    w, S, res = spectrum(tt, A[:, b, jdom], st['t_first'] - rs, st['t_first'] + 3 * rs)
                else:
                    w, S, res = spectrum(tt, A[:, b, jdom], tp_ref - 2 * rs, tp_ref + 2 * rs)
                if w is not None and S.sum() > 0:
                    row.update(omega_peak=float(w[np.argmax(S)]), omega_centroid=float(np.sum(w * S) / np.sum(S)), omega_resolution=res)
                row['phase_at_peak'] = float(np.angle(A[int(np.argmin(np.abs(tt - tp_ref))), b, jdom]))
            rows.append(row)
    # the 4 x 4 site block with the root-sum-of-squares certificate (review G10.4); FIX 1a: the block is now 4 x 4
    site_b = [b for b, sd in enumerate(meta['src']) if sd['kind'] == 'site']
    if len(site_b) == 4:
        cblock = np.sqrt((cert[:, site_b] ** 2).sum(1))
        common = dict(base)
        common.update(src_index=-1, src_kind='site_block', src_comp=-1, omega_src=np.nan, hel_src=0, res_src='raw')
        for key, cols in groups.items():
            if key[0] != 'site':
                continue
            rm = rec_meta[cols[0]]
            blk = A[:, site_b][:, :, cols]                        # (times, 4 source components, receiver components)
            amp = np.sqrt((np.abs(blk) ** 2).sum((1, 2)))         # Frobenius norm of the block
            smax = np.linalg.svd(blk, compute_uv=False)[:, 0]     # its largest singular value on the readout grid
            st = arrival_stats(tt, amp, cblock)
            r1 = float(rm['r'])
            row = dict(common)
            sm = dict(smax_first=np.nan, smax_peak=np.nan, smax_peak_res=np.nan, smax_grid_peak=float(smax.max()))
            if np.isfinite(st['t_first']):
                for tn, an, cn, sn in (('t_first', 'amp_first', 'cert_first', 'smax_first'), ('t_peak', 'peak', 'cert_peak', 'smax_peak'),
                                       ('t_peak_res', 'peak_res', 'cert_peak_res', 'smax_peak_res')):
                    if np.isfinite(st[tn]):
                        t_, fr_, sm_, edge = refine_block(d, meta, site_b, cols, st[tn], dt_read)
                        st[tn], st[an], st[cn], sm[sn] = t_, fr_, cert_at(tt, cblock, t_), sm_
                        st['edge_' + tn[2:]] = edge
                st['resolved'] = bool(st['peak'] > 10.0 * st['cert_peak'])
                st['resolved_first'] = bool(st['amp_first'] > 10.0 * st['cert_first'])
            n_odd, plab = site_parity(src_pos, rm.get('pos', src_pos))
            frac = site_frac(meta, rm)
            row.update(receiver=key[1], rec_kind='site', r_rec=r1, omega_rec=np.nan, sigma_rec=np.nan, res_rec='raw', matched_to=-1,
                       frac_rec=frac, frac_canon=canon_frac(frac), n_odd_diff=n_odd, parity=plab, n_cols=len(cols), **sm,
                       **{k: v for k, v in st.items() if not k.startswith('i_')})
            chart(row, r1, rm)
            rows.append(row)
    # the printed number: s_max^2(P_D U B_X) over the sweep-end times, with the certificate band (unchanged)
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
                              R_ret_resolved=best_res[0], t_resolved=best_res[1], band_resolved=best_res[2], R0=float(series[0][1]) if series else np.nan,
                              optimizer=(None if best[3] is None else [complex(x) for x in best[3]]),
                              series=series, solver=meta['solver'], H=meta['H'], ell=meta['ell'], Rb_frac=meta['Rb_frac'], sampling=meta['sampling'],
                              prep=base['prep']))
    # the receiver-filter transmissivity: Lowdin-orthonormalized receiver packets at a radius against the 4 site inputs (R(0) added)
    if len(site_b) == 4:
        for gi, (c0, c1) in enumerate(meta['group_cols']):
            rm = rec_meta[c0]
            if rm['kind'] != 'pkt' or not rm['role'].startswith('rec') or rm['res'] != 'raw' or abs(rm['sigma'] - meta['sigma']) > 1e-9 or int(rm.get('matched_to', -1)) != -1:
                continue
            S = d[f'group_S_{gi}']
            W, rank = C.lowdin_inverse_sqrt(S)
            cols = list(range(c0, c1))
            best, best_res, R0 = (0.0, np.nan, np.nan), (0.0, np.nan, np.nan), np.nan
            cblock = np.sqrt((cert[:, site_b] ** 2).sum(1))
            for i in range(tt.size):
                M = W @ A[i][np.ix_(site_b, cols)].T
                sv = np.linalg.svd(M, compute_uv=False)
                lam = float(sv[0] ** 2) if sv.size else 0.0
                if i == 0:
                    R0 = lam
                delta = float(cblock[i])
                dR = 2.0 * math.sqrt(lam) * delta + delta * delta
                if lam > best[0]:
                    best = (lam, float(tt[i]), dR)
                if lam > 10.0 * dR and lam > best_res[0]:
                    best_res = (lam, float(tt[i]), dR)
            trows.append(dict(stem=base['stem'], rs=rs, ma=m, direction=meta['direction'], depth=meta['depth'], src_r=src_r, r_h=r_h,
                              region=f"{rm['role']} packets (rank {rank})", family='site_block', kind='filter', R_ret=best[0], t_at_R=best[1], band_at_R=best[2],
                              resolved=bool(best[0] > 10.0 * best[2]), R_ret_resolved=best_res[0], t_resolved=best_res[1], band_resolved=best_res[2], R0=R0,
                              frac=float(rm.get('frac', np.nan)), optimizer=None, series=None, solver=meta['solver'], H=meta['H'], ell=meta['ell'],
                              Rb_frac=meta['Rb_frac'], sampling=meta['sampling'], prep=base['prep']))
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
    rec_meta = meta['rec_meta']
    try:
        fig, axs = plt.subplots(len(srcs), 1, figsize=(9, 2.2 * len(srcs) + 1), sharex=True, squeeze=False)
        for b, sd in enumerate(srcs):
            ax = axs[b, 0]
            for key, cols in sorted(groups.items(), key=lambda kv: str(kv[0])):
                if key[0] != 'pkt' or not key[1].startswith('rec') or key[5] != 'raw' or abs(key[4] - meta['sigma']) > 1e-9:
                    continue
                if sd['kind'] == 'pkt' and key[6] != b:
                    continue
                if sd['kind'] == 'site' and (not is_frac(rec_meta[cols[0]].get('frac'), 2.0) or key[6] != -1):   # the 2 r_s receiver by fraction
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
                           vmin=-12, vmax=math.log10(max(float(P.max()), 1e-16)), cmap='magma')
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
        cand = [(abs(site_frac(meta, rec_meta[cols[0]]) - 2.0), k) for k, cols in groups.items() if k[0] == 'site' and k[1] != 'src']
        cand = sorted([cd for cd in cand if cd[0] < FRAC_TOL], key=lambda cd: cd[0])
        if cand:
            key = cand[0][1]
            cols = groups[key]
            j = max(cols, key=lambda j: np.abs(A[:, 0, j]).max())
            x = A[:, 0, j]
            dt = float(np.median(np.diff(tt)))
            nper = min(256, max(16, (x.size // 2) // 2 * 2))   # 256 at production length; shorter series (tests) keep a valid window
            f, tseg, Z = stft(np.conj(x), fs=1.0 / dt, nperseg=nper, noverlap=nper - max(1, nper // 8), return_onesided=False)
            o = np.argsort(f)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.pcolormesh(tseg / rs, 2 * math.pi * f[o], np.log10(np.maximum(np.abs(Z[o]), 1e-9)), shading='auto', cmap='magma')
            if meta['mode'] == 'radial':
                axN = d['axis_N'] if ('axis_N' in d.files and d['axis_N'].size) else None
                axf = C.AxisField(d['axis_r'], d['axis_u'], axN)
                r1, src_r, nhat, m = key[2], float(meta['src_r']), np.array(meta['nhat']), float(meta['ma'])
                oms = np.geomspace(0.05, 0.9, 12) if m == 0 else np.geomspace(1.1 * m, 0.9, 10)
                tr_ = []
                for om in oms:
                    try:
                        tr_.append(C.ray_time(src_r, r1, om, nhat, m, axf.u, axf.N, crossings=[cc[0] for cc in meta.get('crossings', [])]))
                    except Exception:
                        tr_.append(np.nan)
                ax.plot(np.array(tr_) / rs, oms, 'w--', lw=1, label='ray flight time (allocated field)')
                ax.legend(fontsize=7)
            ax.set_ylim(-1.5, 1.5)
            ax.set_xlabel('t / r_s')
            ax.set_ylabel('a omega')
            n_odd, plab = site_parity(meta.get('src_pos', [0, 0, 0]), rec_meta[cols[0]].get('pos', [0, 0, 0]))
            ax.set_title(f"{name}: site receiver {key[1]} at r = {key[2]:.2f} a, source 0 (site differences: {plab})", fontsize=8)
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


def _e(x, nd=3):
    try:
        if x is None or not np.isfinite(float(x)):
            return 'nan'
        return f"{float(x):.{nd}e}"
    except Exception:
        return str(x)


def _sortval(x):
    try:
        return (0, float(x), '')
    except (TypeError, ValueError):
        return (1, 0.0, str(x))


def _is_raw6(r):
    """The raw source packet read at a raw receiver packet of the run's own width (the literal 6 of v1 replaced by meta sigma)."""
    return r.get('res_src') == 'raw' and abs(float(r.get('sigma_rec', np.nan)) - float(r.get('sigma_run', 6.0))) < 1e-9


def control_pairs(matched):
    """FIX 3: pairs of matched-receiver rows that differ in exactly 1 control. The key holds every field of PAIR_KEY but the
    control, so a pair differs in that control alone; 'prep' (fixed against matched) is a control and 'common_half' a key
    field (v1 lacked both: the twins fell into one group under every control and were skipped as equal, and T4's clipped
    parent runs would have mixed with the unclipped ones). Controls sort numerically where numeric (v1 sorted str(H))."""
    out = []
    for fld in CONTROLS:
        grp = {}
        for r in matched:
            if not _is_raw6(r):
                continue
            key = tuple((k, _k(r.get(k))) for k in PAIR_KEY if k != fld)
            key += (_k(r.get('omega_src')), _k(r.get('hel_src')), _k(canon_frac(r.get('frac_rec', np.nan))))
            grp.setdefault(key, []).append(r)
        for key, rr in grp.items():
            if len(rr) < 2:
                continue
            rr = sorted(rr, key=lambda r: _sortval(r.get(fld)))
            for a, b in zip(rr[:-1], rr[1:]):
                if a.get(fld) == b.get(fld):
                    continue
                out.append((fld, a, b))
    return out


def write_tables(rows, trows, path):
    out = ["# Retarded cross-surface response (GAP10 build v1, analysis v2): tables. Frozen control background; times in units of r_s unless "
           "stated; every amplitude carries its parent certificate (rigorous in exact arithmetic). Analysis v2 (busi/notes/GAP10_analysis_fix_v1.md): "
           "t_peak in tables A to I is the maximum over the resolved readouts |A| > 10 eps_B (nan where the certificate overtook a rising flank; "
           "the global maximum stands beside it in B and C); the site block is the full 4 x 4 block with its parity class; "
           "table F knows the preparation as a control.\n"]
    out.append("## 0. The printed number: R_ret = s_max^2(P_D U(t) B_X) over the sweep-end times, with the band 2 s delta + delta^2\n")
    out.append("| run | d | region D | input family | R_ret (max) | t/rs | band | resolved | R_ret (resolved max) | t/rs | optimizer |a_c| over the 4 site components |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in trows:
        if r['kind'] != 'region':
            continue
        opt = 'nan' if r['optimizer'] is None else ' '.join(f"{abs(x):.3f}" for x in r['optimizer'])
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['region']} | {r['family']} | {r['R_ret']:.4e} | {_f(r['t_at_R'] / r['rs'], 2)} | {r['band_at_R']:.1e} | "
                   f"{int(bool(r['resolved']))} | {r['R_ret_resolved']:.4e} | {_f(r['t_resolved'] / r['rs'], 2)} | {opt} |")
    out.append("\n## 0b. The receiver-filter transmissivity s_max^2(B_D^dagger U B_X), B_D the Lowdin-orthonormalized receiver packets at a radius, B_X the 4 site components; "
               "R(0) is the initial overlap of the site with the packets' support (a value at t = 0 is not a propagated one)\n")
    out.append("| run | d | receiver set | R (max) | t/rs | band | resolved | R (resolved max) | t/rs | R(0) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in trows:
        if r['kind'] != 'filter':
            continue
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['region']} | {r['R_ret']:.4e} | {_f(r['t_at_R'] / r['rs'], 2)} | {r['band_at_R']:.1e} | {int(bool(r['resolved']))} | "
                   f"{r['R_ret_resolved']:.4e} | {_f(r['t_resolved'] / r['rs'], 2)} | {_e(r.get('R0'), 2)} |")
    out.append("\n## A. Raw width-sigma packet sources at the energy-matched raw receiver packets (omega_cmp = the source's actual <h>). "
               "t_peak, peak |A|, eps_B and u_peak: the resolved-window maximum (FIX 2); resolved = the global maximum is itself resolved "
               "(0: the global maximum was voided by the certificate); t_cert = the last resolved readout\n")
    out.append("| run | d | a w label | <h> | h | r_rec/rs | t_first | t_peak | t_ray | t_peak - t_ray | t_nf | Gamma/rs | d/delta_w | peak |A| | u_peak/rs | u_ray/rs | eps_B(t_peak) | resolved | t_cert | overlap0 | share_clean >1.5rs | norm err |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    matched = []
    for r in rows:
        if r.get('src_kind') != 'pkt' or r.get('rec_kind') != 'pkt' or not str(r.get('receiver', '')).startswith('rec'):
            continue
        if r.get('res_rec') != 'raw' or int(r.get('matched_to', -1)) != int(r.get('src_index', -2)):
            continue
        matched.append(r)
        if not _is_raw6(r):
            continue
        rs = r['rs']
        tray = r.get('t_ray', np.nan)
        tpk = r.get('t_peak_res', np.nan)
        ddw = (r['r_h'] - r['src_r']) / r['delta_omega'] if np.isfinite(r.get('delta_omega', np.nan)) and r['delta_omega'] > 0 else np.nan
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['omega_src']:g} | {_f(r['src_energy_mean'], 4)} | {'+' if r['hel_src'] > 0 else '-'} | {r['r_rec'] / rs:.3f} | "
                   f"{_f(r['t_first'] / rs)} | {_f(tpk / rs)} | {_f(tray / rs)} | {_f((tpk - tray) / rs)} | {_f(r.get('t_nf', np.nan) / rs)} | "
                   f"{_f(r.get('Gamma', np.nan) / rs, 3)} | {_f(ddw, 2)} | {_e(r.get('peak_res'))} | {_f(r.get('u_peak_res', np.nan) / rs)} | {_f(r.get('u_ray_rel', np.nan) / rs)} | "
                   f"{_e(r.get('cert_peak_res'), 1)} | {int(bool(r['resolved']))} | {_f(r.get('t_cert', np.nan) / rs, 2)} | {int(bool(r['initial_overlap']))} | "
                   f"{_f(r.get('share_clean_1.5rs'), 4)} | {r['norm_err']:.1e} |")
    out.append("\n## B. Site sources: the 4 x 4 block at the site receivers (|A|_F = Frobenius norm, s_max = largest singular value; certificate = root sum "
               "of squares of the 4 columns). Parity = the number of axes on which the receiver site differs from the source site by an odd number of "
               "spacings: the sublattice symmetry of the sine lattice makes the block null for 2 or 3 odd axes (exactly on the homogeneous infinite "
               "lattice; gradient- and face-sourced otherwise) (FIX 1b); "
               "t_peak: the resolved-window maximum, the global maximum beside it (FIX 2)\n")
    out.append("| run | receiver | r_rec/rs | parity | t_first | |A|_F(t_first) | eps_B(t_first) | resolved_first | t_peak | |A|_F | s_max | eps_B(t_peak) | t_cert | t_peak glob | |A|_F glob | eps_B glob | u_first/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get('src_kind') != 'site_block':
            continue
        rs = r['rs']
        out.append(f"| {r['stem']} | {r['receiver']} | {r['r_rec'] / rs:.3f} | {r.get('parity', 'nan')} | {_f(r['t_first'] / rs)} | {_e(r['amp_first'])} | {_e(r['cert_first'], 1)} | "
                   f"{int(bool(r.get('resolved_first', False)))} | {_f(r.get('t_peak_res', np.nan) / rs)} | {_e(r.get('peak_res'))} | {_e(r.get('smax_peak_res'))} | "
                   f"{_e(r.get('cert_peak_res'), 1)} | {_f(r.get('t_cert', np.nan) / rs, 2)} | {_f(r['t_peak'] / rs)} | {_e(r['peak'])} | {_e(r['cert_peak'], 1)} | "
                   f"{_f(r.get('u_first', np.nan) / rs)} |")
    out.append("\n## C. Site sources (component 0): arrival at the raw width-sigma receiver-packet grid against frequency, receiver at 2 r_s. "
               "t_peak and peak |A|: the resolved-window maximum (FIX 2: the global maximum of the [100] rows at 4.4 to 5.2 r_s is the box-face "
               "return, voided by the certificate); the global maximum beside it\n")
    out.append("| run | d | a w | t_peak | t_ray | t_peak - t_ray | t_nf | peak |A| | eps_B(t_peak) | t_cert | t_peak glob | peak glob | eps_B glob | resolved glob | share_clean >1.5rs (all branches) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get('src_kind') != 'site' or r.get('src_comp') != 0 or r.get('rec_kind') != 'pkt' or not is_frac(r.get('frac_rec'), 2.0) or int(r.get('matched_to', -1)) != -1:
            continue
        if r.get('res_rec') != 'raw' or abs(r['sigma_rec'] - r.get('sigma_run', 6.0)) > 1e-9:
            continue
        rs = r['rs']
        tray = r.get('t_ray', np.nan)
        tpk = r.get('t_peak_res', np.nan)
        out.append(f"| {r['stem']} | {r['depth']:g} | {r['omega_rec']:g} | {_f(tpk / rs)} | {_f(tray / rs)} | {_f((tpk - tray) / rs)} | {_f(r.get('t_nf', np.nan) / rs)} | "
                   f"{_e(r.get('peak_res'))} | {_e(r.get('cert_peak_res'), 1)} | {_f(r.get('t_cert', np.nan) / rs, 2)} | {_f(r['t_peak'] / rs)} | {_e(r['peak'])} | "
                   f"{_e(r['cert_peak'], 1)} | {int(bool(r['resolved']))} | {_f(r.get('share_clean_1.5rs'), 4)} |")
    out.append("\n## D. Delay law per depth (ma = 0 runs): slope of t_peak against ln(1/<h>) over the raw packet sources at the matched receiver at 2 r_s, "
               "against the slope of the exact ray integral, of the normal form, and the asymptotes Gamma, 1/kappa_h, 2 r_s (resolved-window peaks; rows without one are dropped)\n")
    out.append("| run | d | slope/rs (propagation) | slope/rs (ray) | slope/rs (normal form) | Gamma/rs | 1/(kappa_h rs) | 2 | points |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    byrun = {}
    for r in matched:
        if _is_raw6(r) and is_frac(r.get('frac_rec'), 2.0) and r['ma'] == 0.0 and np.isfinite(r.get('t_peak_res', np.nan)):
            byrun.setdefault((r['stem'], r['hel_src']), []).append(r)
    for (stem, hel), rr in sorted(byrun.items()):
        if len(rr) < 2:
            continue
        rs = rr[0]['rs']
        x = np.log(1.0 / np.array([r['src_energy_mean'] for r in rr]))
        sp = np.polyfit(x, np.array([r['t_peak_res'] for r in rr]), 1)[0] / rs
        tr_ = np.array([r.get('t_ray', np.nan) for r in rr])
        sr = np.polyfit(x, tr_, 1)[0] / rs if np.all(np.isfinite(tr_)) else np.nan
        tn = np.array([r.get('t_nf', np.nan) for r in rr])
        sn = np.polyfit(x, tn, 1)[0] / rs if np.all(np.isfinite(tn)) else np.nan
        kh = rr[0]['kappa_h']
        out.append(f"| {stem} h{'+' if hel > 0 else '-'} | {rr[0]['depth']:g} | {_f(sp)} | {_f(sr)} | {_f(sn)} | {_f(rr[0].get('Gamma', np.nan) / rs)} | "
                   f"{_f(1.0 / (kh * rs)) if kh and np.isfinite(kh) and kh > 0 else 'nan'} | 2 | {len(rr)} |")
    out.append("\n## E. Anisotropy: t_peak[111] - t_peak[100] at the same d, a w label and declared receiver fraction (matched within 0.1; actual radii printed), "
               "against the ray's own lag (resolved-window peaks)\n")
    out.append("| pair | d | a w | frac | r_rec [100], [111] | lag/rs (propagation) | lag/rs (ray) | Gamma ln sqrt3 / rs |")
    out.append("|---|---|---|---|---|---|---|---|")
    index = {}
    for r in matched:
        if not _is_raw6(r):
            continue
        key = (r['stem'].replace('_111_', '_DIR_').replace('_100_', '_DIR_'), r['depth'], r['omega_src'], r['hel_src'], _k(canon_frac(r.get('frac_rec', np.nan))))
        index.setdefault(key, {})[r['direction']] = r
    for key, dd in sorted(index.items(), key=lambda kv: str(kv[0])):
        if '100' in dd and '111' in dd:
            a, b = dd['100'], dd['111']
            rs = a['rs']
            gam = a.get('Gamma', np.nan)
            out.append(f"| {key[0]} | {key[1]:g} | {key[2]:g} | {key[4]} | {a['r_rec']:.2f}, {b['r_rec']:.2f} | {_f((b.get('t_peak_res', np.nan) - a.get('t_peak_res', np.nan)) / rs)} | "
                       f"{_f((b.get('t_ray', np.nan) - a.get('t_ray', np.nan)) / rs)} | {_f(gam * math.log(math.sqrt(3.0)) / rs)} |")
    out.append("\n## F. Pairs differing in 1 control only (solver, parent half-width, ell, interior, sampling, profile, preparation): matched raw receivers; "
               "amplitude differences are to be compared with eps_A + eps_B (both parents' certificates); resolved-window peaks (FIX 3: the preparation is a control, "
               "the clipping is in the key)\n")
    out.append("| control | run A | run B | d | a w | r_rec/rs | retained A | retained B | t_peak A | t_peak B | |A| A | |A| B | ||A|-|B|| | eps_A + eps_B |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fld, a, b in control_pairs(matched):
        rs = a['rs']
        pa, pb = a.get('peak_res', np.nan), b.get('peak_res', np.nan)
        out.append(f"| {fld} | {a['stem']} | {b['stem']} | {a['depth']:g} | {a['omega_src']:g} | {a['r_rec'] / rs:.2f} | {_f(a.get('rec_retained'), 4)} | {_f(b.get('rec_retained'), 4)} | "
                   f"{_f(a.get('t_peak_res', np.nan) / rs, 4)} | {_f(b.get('t_peak_res', np.nan) / rs, 4)} | {_e(pa, 4)} | {_e(pb, 4)} | {_e(abs(pa - pb), 2)} | "
                   f"{_e(a.get('cert_peak_res', np.nan) + b.get('cert_peak_res', np.nan), 1)} |")
    out.append("\n## G. Inside-restricted twin against the raw source packet (same run, d, a w, h), matched raw receivers (resolved-window peaks)\n")
    out.append("| run | d | a w | r_rec/rs | exterior prob (raw) | t_peak raw | t_peak in | |A| raw | |A| in | u_peak raw/rs | u_peak in/rs |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    tw = {}
    for r in rows:
        if r.get('src_kind') != 'pkt' or r.get('rec_kind') != 'pkt' or not str(r.get('receiver', '')).startswith('rec') or r.get('res_rec') != 'raw':
            continue
        if abs(r['sigma_rec'] - r.get('sigma_run', 6.0)) > 1e-9 or int(r.get('matched_to', -1)) != -1 or abs(r['omega_rec'] - r['omega_src']) > 1e-9:
            continue
        tw.setdefault((r['stem'], r['omega_src'], r['hel_src'], _k(r.get('frac_rec', np.nan))), {})[r['res_src']] = r
    for key, dd in sorted(tw.items(), key=lambda kv: str(kv[0])):
        if 'raw' in dd and 'in' in dd:
            a, b = dd['raw'], dd['in']
            rs = a['rs']
            out.append(f"| {key[0]} | {a['depth']:g} | {key[1]:g} | {a['r_rec'] / rs:.2f} | {_f(a.get('src_exterior'), 3)} | {_f(a.get('t_peak_res', np.nan) / rs)} | "
                       f"{_f(b.get('t_peak_res', np.nan) / rs)} | {_e(a.get('peak_res'))} | {_e(b.get('peak_res'))} | {_f(a.get('u_peak_res', np.nan) / rs)} | "
                       f"{_f(b.get('u_peak_res', np.nan) / rs)} |")
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
               "int (1/c_g - 1/(1 - S_0)) dr; receiver independence of u is not assumed (resolved-window peaks; fractions matched within 0.1)\n")
    out.append("| run | d | a w label | h | pair | Delta u_peak/rs (propagation) | Delta u/rs (ray) | difference/rs |")
    out.append("|---|---|---|---|---|---|---|---|")
    byp = {}
    for r in matched:
        if not _is_raw6(r):
            continue
        fc = canon_frac(r.get('frac_rec', np.nan))
        if np.isfinite(fc):
            byp.setdefault((r['stem'], r['src_index']), {})[fc] = r
    for key, dd in sorted(byp.items(), key=lambda kv: str(kv[0])):
        fr = sorted([f for f in dd if f >= 1.5 - 1e-9])
        for f1, f2 in zip(fr[:-1], fr[1:]):
            a, b = dd[f1], dd[f2]
            rs = a['rs']
            du = b.get('u_peak_res', np.nan) - a.get('u_peak_res', np.nan)
            dr_ = b.get('u_ray_rel', np.nan) - a.get('u_ray_rel', np.nan)
            out.append(f"| {key[0]} | {a['depth']:g} | {a['omega_src']:g} | {'+' if a['hel_src'] > 0 else '-'} | {f1:g} -> {f2:g} rs | {_f(du / rs)} | {_f(dr_ / rs)} | {_f((du - dr_) / rs)} |")
    with open(path, 'w') as f:
        f.write('\n'.join(out) + '\n')


def _row(**kw):
    base = dict(rs=32.0, ma=0.0, ell=4.0, Rb_frac=0.6, sampling='averaged', profile='ball', mode='radial', H=112, direction='100', depth=4.0,
                solver='cheb', prep='fixed', common_half=0, res_src='raw', sigma_rec=6.0, sigma_run=6.0, omega_src=0.1, hel_src=1, frac_rec=2.0,
                stem='x', r_rec=64.0, rec_retained=1.0, t_peak_res=50.0, peak_res=0.5, cert_peak_res=1e-9)
    base.update(kw)
    return base


def _lattice_symmetry_test(say):
    """FIX 1b on a small box: h of gap10_core (homogeneous and smoothly varying N, v) commutes with S_j = (-1)^{x_j} gamma_5 beta alpha_j;
    the site block of exp(-i h t) commutes (even difference) or anticommutes (odd) with each W_j; on homogeneous fields without current
    the blocks with 2 or 3 odd differences vanish on the open box; with a homogeneous current they are face-sourced (the commutator of
    the current part with the rest is supported on the 2 planes nearest the faces); on varying fields gradient-sourced. Returns the
    number of failures."""
    import torch
    import scipy.linalg as sla
    fails = 0
    L, c, m, tprop = 7, 3, 0.2, 5.0
    idx = np.arange(L) - c
    X, Y, Z = np.meshgrid(idx, idx, idx, indexing='ij')
    G5 = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]]).astype(np.complex128)
    W = [G5 @ C.BETA @ C.ALPHA[j] for j in range(3)]
    for j in range(3):
        ok = np.allclose(W[j] @ C.ALPHA[j] @ W[j].conj().T, -C.ALPHA[j]) and np.allclose(W[j] @ C.BETA @ W[j].conj().T, C.BETA) \
            and all(np.allclose(W[j] @ C.ALPHA[l] @ W[j].conj().T, C.ALPHA[l]) for l in range(3) if l != j) and np.allclose(W[j] @ W[j].conj().T, np.eye(4))
        say(f"W_{j} = gamma_5 beta alpha_{j} flips alpha_{j} only and is unitary", ok)
        fails += not ok
    n = 4 * L ** 3

    def flat(p):
        return ((p[0] + c) * L + (p[1] + c)) * L + (p[2] + c)

    src = (-1, -1, -1)
    # (receiver, number of odd axes); the last 3 have no axis transposition fixing both sites
    recs = [((1, 1, 1), 0), ((0, -1, -1), 1), ((1, 0, -1), 1), ((0, 0, -1), 2), ((1, 0, 0), 2), ((0, 0, 0), 3), ((2, 0, 0), 3),
            ((0, 1, 2), 2), ((2, 1, 0), 2), ((1, 2, 0), 2)]

    def null_by_permutation(rec):
        """The (2 or 3)-odd sector coefficient is antisymmetric under the transposition of 2 odd axes; when the fields are
        cubic symmetric and both sites are invariant under such a transposition the block vanishes exactly."""
        dd = np.abs(np.asarray(rec) - np.asarray(src)) % 2
        return any(dd[i] == 1 and dd[j] == 1 and src[i] == src[j] and rec[i] == rec[j] for i in range(3) for j in range(i + 1, 3))

    N0 = np.full((L, L, L), 0.8)
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    sinc = np.where(R > 0, np.sin(0.4 * R) / np.maximum(R, 1e-300), 0.4)
    configs = [('homogeneous N, v = 0', N0, [np.zeros((L, L, L)) for _ in range(3)]),
               ('homogeneous N and v', N0, [np.full((L, L, L), s) for s in (0.3, -0.2, 0.15)]),
               ('radial (cubic-symmetric) N and v', 0.7 + 0.25 * np.cos(0.6 * R), [-0.3 * X * sinc, -0.3 * Y * sinc, -0.3 * Z * sinc]),
               ('varying N and v', 0.7 + 0.25 * np.cos(0.5 * X + 0.3 * Y - 0.4 * Z),
                [0.4 * np.sin(0.4 * X - 0.2 * Z + 0.3), 0.3 * np.cos(0.3 * Y + 0.5 * X), -0.35 * np.sin(0.45 * Z + 0.1 * Y - 0.2)])]
    mags, Hms = {}, {}
    for label, N, v in configs:
        wp, wm = C.bond_weights_np(v)
        lp, lm = C.link_weights_np(N)
        Hop = C.LatticeH(m, wp, wm, torch.device('cpu'), ghost_np=None, N_np=N, lp_np=lp, lm_np=lm)
        psi = torch.eye(n, dtype=C.CD).reshape(n, 4, L, L, L)
        Hm = Hop.apply(psi).reshape(n, n).numpy().T
        Hms[label] = Hm
        herm = float(np.abs(Hm - Hm.conj().T).max())
        say(f"{label}: h Hermitian (max |h - h^+| = {herm:.1e})", herm < 1e-13)
        fails += not (herm < 1e-13)
        for j, coord in enumerate((X, Y, Z)):
            sgn = np.where(coord.reshape(-1) % 2 == 0, 1.0, -1.0)
            SW = np.kron(W[j], np.diag(sgn))
            cm = float(np.abs(Hm @ SW - SW @ Hm).max())
            say(f"  [h, (-1)^x_{j} W_{j}] = 0 (max {cm:.1e})", cm < 1e-13)
            fails += not (cm < 1e-13)
        U = sla.expm(-1j * tprop * Hm)
        iy = [s * L ** 3 + flat(src) for s in range(4)]
        for rec, nodd in recs:
            ix = [s * L ** 3 + flat(rec) for s in range(4)]
            M = U[np.ix_(ix, iy)]
            dd = np.asarray(rec) - np.asarray(src)
            n_odd, _ = site_parity(src, rec)
            assert n_odd == nodd, (src, rec, n_odd, nodd)
            nrm = float(np.linalg.norm(M))
            sect = max(float(np.linalg.norm(W[j] @ M - ((-1) ** int(abs(dd[j]) % 2)) * M @ W[j])) for j in range(3))
            mags[(label, rec)] = nrm
            ok = sect < 1e-12 * max(nrm, 1e-300) + 1e-14
            say(f"  block {src} -> {rec} ({nodd} odd): |M|_F = {nrm:.3e}, sector residual {sect:.1e}", ok)
            fails += not ok
            if label == 'homogeneous N, v = 0' and nodd >= 2:
                say(f"    {nodd} odd axes on homogeneous fields without current: the block vanishes (|M|_F = {nrm:.1e})", nrm < 1e-12)
                fails += not (nrm < 1e-12)
            if label.startswith('radial') and nodd >= 2:
                if null_by_permutation(rec):
                    say(f"    radial fields, both sites fixed by a transposition of 2 odd axes (the T7 [111] case): the block vanishes (|M|_F = {nrm:.1e})", nrm < 1e-12)
                    fails += not (nrm < 1e-12)
                else:
                    say(f"    radial fields, no such transposition: the block is gradient-sourced (|M|_F = {nrm:.1e}, no threshold)", True)
    # with a homogeneous current on the OPEN box the 2- and 3-odd blocks are small but not zero: the truncated 1-hop and 2-hop
    # along an axis do not commute at the 2 planes nearest the faces (on the infinite lattice they commute and the null is exact);
    # the commutator of the current part with the rest of h is supported on those planes only
    Hq = Hms['homogeneous N and v'] - Hms['homogeneous N, v = 0']
    Cm = Hms['homogeneous N, v = 0'] @ Hq - Hq @ Hms['homogeneous N, v = 0']
    interior = np.flatnonzero(np.all((np.stack([X, Y, Z], -1).reshape(-1, 3) >= -c + 2) & (np.stack([X, Y, Z], -1).reshape(-1, 3) <= c - 2), 1))
    rows_int = np.concatenate([s * L ** 3 + interior for s in range(4)])
    cin, call = float(np.abs(Cm[rows_int]).max()), float(np.abs(Cm).max())
    say(f"  homogeneous current: [h(v=0), h_current] vanishes on the interior rows (max {cin:.1e}; on the face planes {call:.1e}): the nonzero "
        f"2- and 3-odd blocks on the open box are face-sourced", cin < 1e-13 and call > 1e-3)
    fails += not (cin < 1e-13 and call > 1e-3)
    for label in ('homogeneous N and v', 'radial (cubic-symmetric) N and v', 'varying N and v'):
        ev = max(mags[(label, r)] for r, k in recs if k == 0)
        for rec, nodd in recs:
            if nodd >= 2:
                say(f"  {label}: block {rec} ({nodd} odd) / even-difference block = {mags[(label, rec)] / ev:.2e} ({'face' if label.startswith('homog') else 'gradient'}-sourced, no threshold)", True)
    return fails


def selftest():
    """Synthetic checks of the 3 fixes; exit code 0 when every check passes."""
    fails = [0]

    def say(msg, ok):
        print(('PASS ' if ok else 'FAIL ') + msg)
        fails[0] += not ok

    # FIX 1a: the site receiver groups
    rm_site = [dict(kind='site', role=role, r=r, pos=pos, comp=k) for role, r, pos in (('src', 20.0, [20, 0, 0]), ('rec2', 58.0, [58, 0, 0])) for k in range(4)]
    rm_pkt = [dict(kind='pkt', role='rec2', r=58.0, omega=0.25, sigma=6.0, res='raw', matched_to=-1, hel=h) for h in (1, -1)]
    g2 = build_groups(rm_site + rm_pkt)
    v1 = {}
    for j, rm in enumerate(rm_site + rm_pkt):
        k = ('site', rm['role'], round(rm['r'], 3), float('nan'), float('nan'), 'raw', -1) if rm['kind'] == 'site' else group_key(rm)
        v1.setdefault(k, []).append(j)
    say(f"FIX 1a: 2 site receivers x 4 components + 1 packet pair -> {len(g2)} groups of sizes {sorted(len(c) for c in g2.values())} (v1's nan key gave {len(v1)})",
        len(g2) == 3 and sorted(len(c) for c in g2.values()) == [2, 4, 4] and len(v1) == 9)
    # FIX 1b: the parity classes of the T7 [111] and [100] site receivers
    cases = [((12, 12, 12), (21, 21, 21), 3), ((12, 12, 12), (25, 25, 25), 3), ((12, 12, 12), (33, 33, 33), 3), ((12, 12, 12), (50, 50, 50), 0),
             ((8, 8, 8), (33, 33, 33), 3), ((8, 8, 8), (50, 50, 50), 0), ((20, 0, 0), (58, 0, 0), 0), ((20, 0, 0), (87, 0, 0), 1), ((14, 0, 0), (36, 0, 0), 0),
             ((12, 12, 12), (22, 22, 22), 0), ((12, 12, 12), (34, 34, 34), 0), ((8, 8, 8), (26, 26, 26), 0)]
    say("FIX 1b: parity classes (T7 [111] receivers at 1.25, 1.5, 2 r_s: 3 odd; at 3 r_s and all [100]: even or 1 odd; the even rerun sites 22, 26, 34, 50: even)",
        all(site_parity(s, r)[0] == k for s, r, k in cases))
    # FIX 2: the resolved-window maximum against the global maximum
    rs = 29.0
    tt = np.arange(0.0, 20.0 * rs + 1e-9, 0.5)
    amp = 0.009 * np.exp(-(tt - 1.7 * rs) ** 2 / (2 * (0.15 * rs) ** 2)) + 0.012 * np.exp(-(tt - 4.4 * rs) ** 2 / (2 * (0.2 * rs) ** 2)) + 1e-9
    cert = 1e-13 * 10.0 ** (3.3 * tt / rs)
    cert[0] = 0.0
    st = arrival_stats(tt, amp, cert)
    say(f"FIX 2: direct pulse 0.009 at 1.7 r_s, face return 0.012 at 4.4 r_s under eps_B 30: global t_peak {st['t_peak'] / rs:.2f} r_s (resolved {int(st['resolved'])}), "
        f"resolved-window t_peak {st['t_peak_res'] / rs:.2f} r_s, peak {st['peak_res']:.4f}, t_cert {st['t_cert'] / rs:.2f} r_s",
        abs(st['t_peak'] / rs - 4.4) < 0.01 and (not st['resolved']) and abs(st['t_peak_res'] / rs - 1.7) < 0.01 and abs(st['peak_res'] - 0.009) < 1e-4 and not st['trunc_res'])
    amp2 = 0.3 * np.exp(-(tt - 3.3 * rs) ** 2 / (2 * (0.6 * rs) ** 2)) + 1e-9
    cert2 = 1e-13 * 10.0 ** (4.0 * tt / rs)
    cert2[0] = 0.0
    st2 = arrival_stats(tt, amp2, cert2)
    say(f"FIX 2: a rising flank overtaken by the certificate before its maximum: t_peak_res nan, trunc_res {st2['trunc_res']}, t_cert {st2['t_cert'] / rs:.2f} r_s "
        f"(global t_peak {st2['t_peak'] / rs:.2f}, resolved {int(st2['resolved'])})",
        (not np.isfinite(st2['t_peak_res'])) and st2['trunc_res'] and 2.5 < st2['t_cert'] / rs < 3.2 and not st2['resolved'])
    amp3 = 0.3 * np.exp(-(tt - 1.5 * rs) ** 2 / (2 * (0.2 * rs) ** 2)) + 1e-9
    st3 = arrival_stats(tt, amp3, cert)
    say(f"FIX 2: a resolved single pulse: global and resolved maxima coincide ({st3['t_peak'] / rs:.2f} = {st3['t_peak_res'] / rs:.2f} r_s, resolved {int(st3['resolved'])})",
        st3['t_peak'] == st3['t_peak_res'] and st3['resolved'] and not st3['trunc_res'])
    # FIX 3: the control pairs
    rows = [_row(stem='fixed'), _row(stem='matched', prep='matched'), _row(stem='taylor', solver='taylor'),
            _row(stem='box60', H=60, common_half=60), _row(stem='box88', H=88, common_half=60), _row(stem='box124', H=124, common_half=60),
            _row(stem='rs24', rs=24.0, H=88)]
    pairs = sorted((f, a['stem'], b['stem']) for f, a, b in control_pairs(rows))
    want = sorted([('prep', 'fixed', 'matched'), ('solver', 'fixed', 'taylor'), ('H', 'box60', 'box88'), ('H', 'box88', 'box124')])
    say(f"FIX 3: control pairs {pairs}", pairs == want)
    v1pairs = []
    for fld in ['solver', 'H', 'ell', 'Rb_frac', 'sampling', 'profile']:
        grp = {}
        for r in rows:
            key = tuple((k, r[k]) for k in ['rs', 'ma', 'ell', 'Rb_frac', 'sampling', 'profile', 'mode', 'H', 'direction', 'depth', 'solver'] if k != fld)
            key += (r['omega_src'], r['hel_src'], r.get('frac_rec', np.nan))
            grp.setdefault(key, []).append(r)
        for key, rr in grp.items():
            rr = sorted(rr, key=lambda r: str(r[fld]))
            for a, b in zip(rr[:-1], rr[1:]):
                if a[fld] != b[fld]:
                    v1pairs.append((fld, a['stem'], b['stem']))
    say(f"FIX 3: the v1 pairing on the same rows gives {sorted(v1pairs)} (no twin pair; str-sorted and clipping-blind box pairs)",
        ('prep', 'fixed', 'matched') not in v1pairs)
    # the receiver selection by fraction
    say("fractions: 2.031 -> 2, 1.314 -> 1.25, 1.553 -> 1.5, 2.5 stays; is_frac(1.971 declared 2.0)",
        canon_frac(2.031) == 2.0 and canon_frac(1.314) == 1.25 and canon_frac(1.553) == 1.5 and canon_frac(2.5) == 2.5 and is_frac(2.0, 2.0) and not is_frac(1.5, 2.0))
    # FIX 1b on the lattice itself (needs torch for the core's operator)
    import importlib.util
    if importlib.util.find_spec('torch') is not None:
        fails[0] += _lattice_symmetry_test(say)
    else:
        print("SKIP lattice symmetry test (torch not importable)")
    print(f"[selftest] {'all checks passed' if fails[0] == 0 else str(fails[0]) + ' check(s) FAILED'}")
    return 0 if fails[0] == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--csv', default='gap10_results.csv')
    ap.add_argument('--tcsv', default='gap10_transmissivity.csv')
    ap.add_argument('--md', default='gap10_results.md')
    ap.add_argument('--figs', default='')
    ap.add_argument('--selftest', action='store_true', help="synthetic checks of the fixes (no run directories needed)")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.dirs:
        ap.error("run directories required (or --selftest)")
    print(f"[analysis v2] core {C.__name__}", flush=True)
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
