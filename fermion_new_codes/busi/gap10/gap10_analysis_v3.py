#!/usr/bin/env python
"""gap10_analysis_v3.py -- gap10_analysis_v2.py imported unchanged (the same command line, the same tables 0 to I, the same csv
files) plus table K for the wavelength-class runs of gap10_run_v2_wl.py (busi/notes/GAP10_wavelength_test_v1.md): per run and
batch member the class and component, the carrier, the Killing-energy split (the semiclassical estimate, the KPM five bins
with the band top b3 = sqrt(m^2 + 3) and b1 = sqrt(m^2 + 1), the filter weights), the escaped shares beyond r_h, 1.5 r_s and
2 r_s at the certified sweep ends (the last sweep end whose certificate is below 1e-3, and the resolved R_ret of table 0 with
its band), the share against the above-threshold weight, the kinematic arrival on the axis, and the component reconstruction
raw = above + below + 2 Re cross (within one batch, or across the 3 runs of one stem family by subtraction). A K csv is written
beside the others (--kcsv). Runs of gap10_run_v2.py carry no 'wl' field and get no K row; nothing else changes.
Two cautions on v2's tables for these runs: norm_err there is read against 1, while a component's norm is its filter weight
(table K prints the drift against the initial norm); tables E and I pair receivers by their fraction and so mix the +nhat and
-nhat receivers of mirrored runs (table K reads the receiver rows by side; E and I are not wavelength-test readouts).
--selftest runs v2's selftest and a synthetic check of the reconstruction and the certified-end selector.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import gap10_analysis_v2 as V2    # noqa: E402

CERT_LEVELS = (1e-3, 1e-2)


def _f(x, nd=4):
    return V2._f(x, nd)


def _e(x, nd=2):
    return V2._e(x, nd)


def certified_ends(sweep_t, times, cert_col, level):
    """The sweep ends (t > 0) whose cumulative certificate is below level; the certificate at a sweep end is cert_at."""
    out = []
    for t in sweep_t[1:]:
        if V2.cert_at(times, cert_col, float(t)) < level:
            out.append(float(t))
    return out


def reconstruct(G, ia, ib, iraw=None):
    """Per region and sweep end: the shares of a (above) and b (below), the cross term 2 Re G_ab, and the raw share (G_raw if the
    raw member is in the batch, else the sum). G has shape (nsweeps+1, nregions, B, B)."""
    sa, sb = G[:, :, ia, ia].real, G[:, :, ib, ib].real
    cross = 2.0 * G[:, :, ia, ib].real
    raw = G[:, :, iraw, iraw].real if iraw is not None else sa + sb + cross
    return sa, sb, cross, raw


def k_rows_for_stem(stem, trows):
    d, meta = V2.load(stem)
    if 'wl' not in meta:
        return [], None
    rs, m = float(meta['rs']), float(meta['ma'])
    tt, cert, sweep_t, shares, G = d['times'], d['cert'], d['sweep_t'], d['shares'], d['gram']
    names = [nm for nm, _ in meta['share_radii']]
    rnames = list(meta['region_names'])
    norms = d['norms']
    rows = []
    for b, sd in enumerate(meta['src']):
        kil = meta['wl']['killing'].get(sd.get('wl_class', ''), {})
        semi = sd.get('semiclassical', {})
        kin = sd.get('kinematics', {})
        tails = sd.get('initial_tails', {})
        row = dict(stem=os.path.basename(stem), direction=meta['direction'], depth=meta['depth'], prep=meta.get('prep', ''), H=meta['H'], rs=rs, ma=m,
                   src_index=b, wl_class=sd.get('wl_class'), component=sd.get('wl_component'), k=sd.get('k'), wavelength_rs=(2 * math.pi / abs(sd['k']) / rs if sd.get('k') else np.nan),
                   E_label=sd.get('omega'), E_mean=sd.get('energy_mean'), E_std=sd.get('energy_std'), E_carrier_run=sd.get('E_carrier_run'),
                   norm0=float(norms[0, b]), norm_drift=float(np.max(np.abs(norms[:, b] / norms[0, b] - 1.0))),
                   upper_band=sd.get('band', {}).get('upper_band'), lattice_scale=sd.get('band', {}).get('lattice_scale'),
                   out_pos_freq=sd.get('band', {}).get('outgoing_positive_frequency'), exterior_prob=sd.get('exterior_probability'),
                   tail_rh=(list(tails.values())[0] if tails else np.nan), tail_15=tails.get(f"{1.5 * rs:.4f}", np.nan), tail_2=tails.get(f"{2.0 * rs:.4f}", np.nan),
                   semi_E_mean=semi.get('E_mean'), semi_E_std=semi.get('E_std'), semi_above=semi.get('w_above'), semi_gap=semi.get('w_gap'),
                   semi_m_b3=semi.get('bins_b3', {}).get('m_to_b'), semi_above_b3=semi.get('bins_b3', {}).get('above_b'),
                   kpm_n=kil.get('n_moments_used'), kpm_res=kil.get('resolution'), kpm_k_exact=kil.get('k_exact'), kpm_norm=kil.get('kpm_norm'),
                   kpm_above=kil.get('w_above'), kpm_gap=kil.get('w_gap'), kpm_below=kil.get('w_below'), kpm_trans=kil.get('w_trans_plus'),
                   kpm_m_b3=kil.get('bins_b3', {}).get('m_to_b'), kpm_above_b3=kil.get('bins_b3', {}).get('above_b'), kpm_m_b1=kil.get('bins_b1', {}).get('m_to_b'),
                   kpm_below_minus_m=(kil.get('bins_b3', {}).get('-b_to_-m', 0.0) + kil.get('bins_b3', {}).get('below_-b', 0.0)) if kil else np.nan,
                   filt_above=kil.get('w_filter_above'), filt_below=kil.get('w_filter_below'), filt_overlap=kil.get('filter_overlap'), filt_box=kil.get('filter_box_bound'),
                   filt_s=kil.get('filter_s'), b3=kil.get('b3'), b1=kil.get('b1'),
                   t_in_rs=(kin.get('t_in', np.nan) / rs if kin.get('t_in') is not None else np.nan), r_turn=kin.get('r_turn'),
                   t_kin_15=(kin.get('t_total', {}).get(f"{1.5 * rs:.4f}", np.nan) / rs), t_kin_2=(kin.get('t_total', {}).get(f"{2.0 * rs:.4f}", np.nan) / rs),
                   reach_rs=(kin.get('reach_at_E', np.nan) / rs if kin.get('reach_at_E') is not None else np.nan))
        for lev in CERT_LEVELS:
            ends = certified_ends(sweep_t, tt, cert[:, b], lev)
            tag = f"{lev:.0e}"
            row[f't_cert_end_{tag}'] = (ends[-1] / rs) if ends else np.nan
            for k, nm in enumerate(names):
                if nm not in ('r_h', '1.5rs', '2rs'):
                    continue
                if ends:
                    sel = np.array([int(np.argmin(np.abs(sweep_t - t))) for t in ends])
                    row[f'share_{nm}_end_{tag}'] = float(shares[sel[-1], b, k])
                    row[f'share_{nm}_max_{tag}'] = float(shares[sel, b, k].max())
                    row[f't_share_{nm}_max_{tag}'] = float(sweep_t[sel[np.argmax(shares[sel, b, k])]] / rs)
                else:
                    row[f'share_{nm}_end_{tag}'] = row[f'share_{nm}_max_{tag}'] = row[f't_share_{nm}_max_{tag}'] = np.nan
        for k, nm in enumerate(names):
            if nm in ('r_h', '1.5rs', '2rs'):
                row[f'share_{nm}_last'] = float(shares[-1, b, k])
        # the resolved R_ret of table 0 for this member's family (the packet label)
        for tr in trows:
            if tr['stem'] == row['stem'] and tr['kind'] == 'region' and tr['family'] == sd['label']:
                row[f"Rres_{tr['region']}"] = tr['R_ret_resolved']
                row[f"tres_{tr['region']}"] = tr['t_resolved'] / rs if np.isfinite(tr['t_resolved']) else np.nan
                row[f"band_{tr['region']}"] = tr['band_resolved']
        n0 = row['norm0'] if row['norm0'] else np.nan
        row['cond_share_1.5rs_end_1e-03'] = row.get('share_1.5rs_end_1e-03', np.nan) / n0
        row['cond_share_2rs_end_1e-03'] = row.get('share_2rs_end_1e-03', np.nan) / n0
        wab = row.get('kpm_m_b3', np.nan)
        row['ratio_share15_over_w_m_b3'] = (row.get('share_1.5rs_end_1e-03', np.nan) / wab) if (wab and np.isfinite(wab) and wab > 0) else np.nan
        rows.append(row)
    # the component reconstruction inside one batch
    recon = []
    byc = {}
    for b, sd in enumerate(meta['src']):
        byc.setdefault(sd.get('wl_class'), {})[sd.get('wl_component')] = b
    for cls, dd in byc.items():
        if 'above' in dd and 'below' in dd:
            sa, sb, cross, raw = reconstruct(G, dd['above'], dd['below'], dd.get('raw'))
            dev = float(np.max(np.abs(raw - (sa + sb + cross)))) if 'raw' in dd else np.nan
            for ri, rn in enumerate(rnames):
                for si, t in enumerate(sweep_t):
                    recon.append(dict(stem=os.path.basename(stem), wl_class=cls, region=rn, t_rs=float(t) / rs, share_above=float(sa[si, ri]), share_below=float(sb[si, ri]),
                                      cross=float(cross[si, ri]), share_raw=float(raw[si, ri]), identity_dev=dev, source='batch'))
    return rows, recon


def family_reconstruction(stems_meta):
    """Across runs: the 3 single-member runs raw, above, below of one class and stem family (the stem with the components token
    replaced): cross = raw - above - below per region and sweep end (the 3 propagations are deterministic on one h)."""
    fam = {}
    for stem, (d, meta) in stems_meta.items():
        if 'wl' not in meta or len(meta['src']) != 1:
            continue
        sd = meta['src'][0]
        key = re.sub(r'_wl[a-z]+_[a-z_\-]+_cheb', '_wlX_cheb', os.path.basename(stem)), sd.get('wl_class')
        fam.setdefault(key, {})[sd.get('wl_component')] = (stem, d, meta)
    out = []
    for (fkey, cls), dd in fam.items():
        if not all(c in dd for c in ('raw', 'above', 'below')):
            continue
        dr, da, db = dd['raw'][1], dd['above'][1], dd['below'][1]
        rs = float(dd['raw'][2]['rs'])
        rnames = list(dd['raw'][2]['region_names'])
        if not (dr['sweep_t'].shape == da['sweep_t'].shape == db['sweep_t'].shape):
            continue
        for ri, rn in enumerate(rnames):
            for si, t in enumerate(dr['sweep_t']):
                raw, a, b = dr['gram'][si, ri, 0, 0].real, da['gram'][si, ri, 0, 0].real, db['gram'][si, ri, 0, 0].real
                out.append(dict(stem=fkey, wl_class=cls, region=rn, t_rs=float(t) / rs, share_above=float(a), share_below=float(b), cross=float(raw - a - b),
                                share_raw=float(raw), identity_dev=np.nan, source='3 runs'))
    return out


def write_table_K(rows, recon, path_md):
    out = ["\n## K. The wavelength-class test (gap10_run_v2_wl.py): long-wavelength sources on the held record, their Killing-energy split and the escaped shares "
           "at the certified sweep ends. Bins with the band top b3 = sqrt(m^2 + 3) (the cubic operator) and b1 = sqrt(m^2 + 1) (the chain's); 'semi' = the local-symbol "
           "estimate, 'KPM' = the computed split (Jackson density from the Chebyshev moments, resolution printed), 'filter' = the weights of psi_> and psi_<. "
           "share_end = the share at the last sweep end with certificate < 1e-3 (t given), share_max over those ends; Rres = table 0's resolved maximum with its band; "
           "cond = the component's share divided by its own weight; kin = the axis WKB arrival of the incoming class (reach = the sub-threshold class's farthest outgoing-branch radius).\n"]
    out.append("| run | class | comp | d | prep | k | lambda/rs | E label | <h> | E std | lattice | tail>1.5rs | semi E | semi above m | semi (m,b3) | KPM n, res | KPM above m | KPM gap | KPM below -m | KPM (m,b3) | KPM above b3 | KPM +-res of m | filter > | filter < | overlap | norm0 | t_cert(1e-3) | share>r_h end | share>1.5rs end | share>2rs end | share>1.5rs max (t) | share>2rs max (t) | Rres>1.5rs (band) | Rres>2rs (band) | cond>1.5rs | share>1.5rs / w(m,b3) | t_kin 1.5, 2 rs | reach/rs | norm drift |")
    out.append("|" + "---|" * 39)
    for r in rows:
        out.append(f"| {r['stem']} | {r['wl_class']} | {r['component']} | {r['depth']:g} | {r['prep']} | {_f(r['k'], 4)} | {_f(r['wavelength_rs'], 2)} | {_f(r['E_label'])} | {_f(r['E_mean'])} | "
                   f"{_f(r['E_std'])} | {_f(r['lattice_scale'], 3)} | {_e(r['tail_15'], 1)} | {_f(r['semi_E_mean'], 3)}+-{_f(r['semi_E_std'], 3)} | {_f(r['semi_above'], 3)} | {_f(r['semi_m_b3'], 3)} | "
                   f"{r['kpm_n']}, {_f(r['kpm_res'], 3)} | {_f(r['kpm_above'])} | {_f(r['kpm_gap'])} | {_e(r['kpm_below'], 1)} | {_f(r['kpm_m_b3'])} | {_e(r['kpm_above_b3'], 1)} | {_f(r['kpm_trans'], 3)} | "
                   f"{_f(r['filt_above'])} | {_f(r['filt_below'])} | {_f(r['filt_overlap'])} | {_f(r['norm0'])} | {_f(r.get('t_cert_end_1e-03'), 2)} | {_f(r.get('share_r_h_end_1e-03'))} | "
                   f"{_f(r.get('share_1.5rs_end_1e-03'))} | {_f(r.get('share_2rs_end_1e-03'))} | {_f(r.get('share_1.5rs_max_1e-03'))} ({_f(r.get('t_share_1.5rs_max_1e-03'), 1)}) | "
                   f"{_f(r.get('share_2rs_max_1e-03'))} ({_f(r.get('t_share_2rs_max_1e-03'), 1)}) | {_e(r.get('Rres_1.5rs'), 3)} ({_e(r.get('band_1.5rs'), 1)}) | "
                   f"{_e(r.get('Rres_2rs'), 3)} ({_e(r.get('band_2rs'), 1)}) | {_f(r.get('cond_share_1.5rs_end_1e-03'), 3)} | {_f(r.get('ratio_share15_over_w_m_b3'), 3)} | "
                   f"{_f(r['t_kin_15'], 2)}, {_f(r['t_kin_2'], 2)} | {_f(r['reach_rs'], 2)} | {_e(r['norm_drift'], 1)} |")
    out.append("\n### K2. Component reconstruction: raw share = above + below + 2 Re cross per region and sweep end (identity_dev = the largest deviation when the 3 "
               "members share one batch; for 3 single-member runs the cross term is the difference and carries the 3 certificates)\n")
    out.append("| run family | class | region | t/rs | share above | share below | 2 Re cross | share raw | identity dev | source |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in recon:
        out.append(f"| {r['stem']} | {r['wl_class']} | {r['region']} | {r['t_rs']:.2f} | {_f(r['share_above'])} | {_f(r['share_below'])} | {_f(r['cross'])} | {_f(r['share_raw'])} | "
                   f"{_e(r['identity_dev'], 1)} | {r['source']} |")
    with open(path_md, 'a') as f:
        f.write('\n'.join(out) + '\n')


def selftest():
    rc = V2.selftest()
    fails = []

    def say(ok, msg):
        print(('PASS ' if ok else 'FAIL ') + msg, flush=True)
        if not ok:
            fails.append(msg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2, 3, 7)) + 1j * rng.normal(size=(5, 2, 3, 7))   # (sweeps, regions, members, dim): members a, b and raw = a + b
    X[:, :, 2] = X[:, :, 0] + X[:, :, 1]
    G = np.einsum('srad,srbd->srab', X.conj(), X)
    sa, sb, cross, raw = reconstruct(G, 0, 1, 2)
    say(float(np.max(np.abs(raw - (sa + sb + cross)))) < 1e-10, "synthetic reconstruction raw = above + below + 2 Re cross")
    tt = np.arange(0, 21, 0.5)
    cert = np.where(tt < 10.0, 1e-5, 1e-1)
    ends = certified_ends(np.array([0.0, 4.0, 8.0, 12.0, 16.0]), tt, cert, 1e-3)
    say(ends == [4.0, 8.0], f"certified-end selector ({ends})")
    print(f"[selftest v3] {len(fails)} failures beyond v2's exit code {rc}", flush=True)
    return 0 if (rc == 0 and not fails) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--csv', default='gap10_results.csv')
    ap.add_argument('--tcsv', default='gap10_transmissivity.csv')
    ap.add_argument('--kcsv', default='gap10_wavelength.csv')
    ap.add_argument('--md', default='gap10_results.md')
    ap.add_argument('--figs', default='')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.dirs:
        ap.error("run directories required (or --selftest)")
    print(f"[analysis v3] core {V2.C.__name__} (v2 tables unchanged; table K added)", flush=True)
    stems = sorted({p[:-4] for dd in args.dirs for p in glob.glob(os.path.join(dd, '*.npz'))})
    rows, trows = [], []
    for s in stems:
        print('[analysis]', s, flush=True)
        try:
            V2.analyze(s, rows, trows, args.figs or None)
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
    V2.write_tables(rows, trows, args.md)
    krows, recon, stems_meta = [], [], {}
    for s in stems:
        try:
            kr, rc = k_rows_for_stem(s, trows)
            krows.extend(kr)
            if rc:
                recon.extend(rc)
            if kr:
                stems_meta[s] = V2.load(s)
        except Exception as e:
            print('  table K failed:', s, repr(e))
    recon.extend(family_reconstruction(stems_meta))
    if krows:
        kkeys = sorted({k for r in krows for k in r})
        with open(args.kcsv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=kkeys)
            w.writeheader()
            w.writerows(krows)
        write_table_K(krows, recon, args.md)
    print(f"[analysis] {len(rows)} readout rows -> {args.csv}; {len(trows)} transmissivity rows -> {args.tcsv}; {len(krows)} wavelength rows -> {args.kcsv}; tables -> {args.md}")


if __name__ == '__main__':
    main()
