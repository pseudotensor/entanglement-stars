#!/usr/bin/env python
"""gap10_analysis_v4.py -- the table reissues of GAP10_T7_final_v1.3.md, written from gap10_analysis_v3.py by build_gap10_analysis_v4.py (exact
replacements; no table of gap10_analysis_v2.py or v3 changes in content): table K gains the chain's b1 columns, the normalized component energies, the
3 r_s shell and the energy the stored axis WKB clock was computed at; tables F2 (pairs read at a common clock), L (the parent ladder at common sweep
ends with bands) and S (the solver pairs' single-site 4 x 4 block series on a common time grid) are appended; the md header words are corrected.
Otherwise v3 as it was: gap10_analysis_v2.py imported unchanged (the same command line, the same tables 0 to I, the same csv
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
K_REGIONS = ('r_h', '1.5rs', '2rs', '3rs')    # table K's shells; '3rs' (87 a) exists for the wavelength runs only (the matrix's box ends at 3.55 r_s)
FLOOR_AMP = 2e-7     # the 2 solvers' agreement in a receiver amplitude on the r_s/a 24 ball (GAP10_T7_final_v1.2 section 3): the floor of table S's rule
RES_FACTOR = 10.0    # resolved: a readout above 10 times its certificate band at that clock (gap10_analysis_v2's rule), table F2's window


def _f(x, nd=4):
    return V2._f(x, nd)


def _e(x, nd=2):
    return V2._e(x, nd)


def _v(x):
    """None (an absent json field) read as nan for the printers."""
    return np.nan if x is None else x


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
                if nm not in K_REGIONS:
                    continue
                if ends:
                    sel = np.array([int(np.argmin(np.abs(sweep_t - t))) for t in ends])
                    row[f'share_{nm}_end_{tag}'] = float(shares[sel[-1], b, k])
                    row[f'share_{nm}_max_{tag}'] = float(shares[sel, b, k].max())
                    row[f't_share_{nm}_max_{tag}'] = float(sweep_t[sel[np.argmax(shares[sel, b, k])]] / rs)
                else:
                    row[f'share_{nm}_end_{tag}'] = row[f'share_{nm}_max_{tag}'] = row[f't_share_{nm}_max_{tag}'] = np.nan
        for k, nm in enumerate(names):
            if nm in K_REGIONS:
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
        row['cond_share_3rs_end_1e-03'] = row.get('share_3rs_end_1e-03', np.nan) / n0
        kab, kb1 = row.get('kpm_above'), row.get('kpm_m_b1')
        row['kpm_above_b1'] = (kab - kb1) if (kab is not None and kb1 is not None) else np.nan    # kpm_above is the weight above m
        row['E_mean_norm'], row['E_std_norm'] = norm_moments(row.get('E_mean'), row.get('E_std'), n0)
        row['t_kin_at_E'] = _v(row.get('E_mean'))    # the energy the stored axis WKB clock t_kin was computed at (unnormalized for a component)
        row['regions'] = ' '.join(names)
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
           "at the certified sweep ends. Bins with the cubic free top b3 = sqrt(m^2 + 3/a^2) (the top of the cubic operator's essential spectrum on the held exterior, lifted locally by the flow: Killing weight above it is bound content) and the chain's free top b1 = sqrt(m^2 + 1/a^2) (C5's b; the (m, b1) column and the weight above b1, which is the weight above m minus the (m, b1) weight); 'semi' = the local-symbol "
           "estimate, 'KPM' = the computed split (Jackson density from the Chebyshev moments, resolution printed), 'filter' = the weights of psi_> and psi_<. "
           "share_end = the share at the last sweep end with certificate < 1e-3 (t given), share_max over those ends; Rres = table 0's resolved maximum with its band; "
           "cond = the component's share divided by its own weight; kin = the axis WKB arrival of the incoming class (reach = the sub-threshold class's farthest outgoing-branch radius), the stored clock, which the driver computed at the stored <h>, for a component the unnormalized <psi|h|psi> (printed again as 't_kin at E'), not at '<h> norm' = <h>/norm0; 'E std norm' is sqrt(<h^2>/norm0 - (<h>/norm0)^2) with <h^2> = E std^2 + <h>^2, the raw members (norm0 1) unchanged; '3rs' is the 3 r_s shell (87 a), present for the wavelength runs only.\n"]
    out.append("| run | class | comp | d | prep | k | lambda/rs | E label | <h> | E std | <h> norm | E std norm | lattice | tail>1.5rs | semi E | semi above m | semi (m,b3) | KPM n, res | KPM above m | KPM gap | KPM below -m | KPM (m,b3) | KPM above b3 | KPM (m,b1) | KPM above b1 | KPM +-res of m | filter > | filter < | overlap | norm0 | t_cert(1e-3) | share>r_h end | share>1.5rs end | share>2rs end | share>3rs end | share>1.5rs max (t) | share>2rs max (t) | Rres>1.5rs (band) | Rres>2rs (band) | Rres>3rs (band) | cond>1.5rs | cond>2rs | cond>3rs | share>1.5rs / w(m,b3) | t_kin 1.5, 2 rs | t_kin at E | reach/rs | norm drift |")
    out.append("|" + "---|" * 48)
    for r in rows:
        out.append(f"| {r['stem']} | {r['wl_class']} | {r['component']} | {r['depth']:g} | {r['prep']} | {_f(r['k'], 4)} | {_f(r['wavelength_rs'], 2)} | {_f(r['E_label'])} | {_f(r['E_mean'])} | "
                   f"{_f(r['E_std'])} | {_f(_v(r.get('E_mean_norm')))} | {_f(_v(r.get('E_std_norm')))} | {_f(r['lattice_scale'], 3)} | {_e(r['tail_15'], 1)} | {_f(r['semi_E_mean'], 3)}+-{_f(r['semi_E_std'], 3)} | {_f(r['semi_above'], 3)} | {_f(r['semi_m_b3'], 3)} | "
                   f"{r['kpm_n']}, {_f(r['kpm_res'], 3)} | {_f(r['kpm_above'])} | {_f(r['kpm_gap'])} | {_e(r['kpm_below'], 1)} | {_f(r['kpm_m_b3'])} | {_e(r['kpm_above_b3'], 1)} | {_f(_v(r.get('kpm_m_b1')), 5)} | {_e(_v(r.get('kpm_above_b1')), 2)} | {_f(r['kpm_trans'], 3)} | "
                   f"{_f(r['filt_above'])} | {_f(r['filt_below'])} | {_f(r['filt_overlap'])} | {_f(r['norm0'])} | {_f(_v(r.get('t_cert_end_1e-03')), 2)} | {_f(_v(r.get('share_r_h_end_1e-03')))} | "
                   f"{_f(_v(r.get('share_1.5rs_end_1e-03')))} | {_f(_v(r.get('share_2rs_end_1e-03')))} | {_f(_v(r.get('share_3rs_end_1e-03')))} | {_f(_v(r.get('share_1.5rs_max_1e-03')))} ({_f(_v(r.get('t_share_1.5rs_max_1e-03')), 1)}) | "
                   f"{_f(_v(r.get('share_2rs_max_1e-03')))} ({_f(_v(r.get('t_share_2rs_max_1e-03')), 1)}) | {_e(_v(r.get('Rres_1.5rs')), 3)} ({_e(_v(r.get('band_1.5rs')), 1)}) | "
                   f"{_e(_v(r.get('Rres_2rs')), 3)} ({_e(_v(r.get('band_2rs')), 1)}) | {_e(_v(r.get('Rres_3rs')), 3)} ({_e(_v(r.get('band_3rs')), 1)}) | {_f(_v(r.get('cond_share_1.5rs_end_1e-03')), 3)} | {_f(_v(r.get('cond_share_2rs_end_1e-03')), 3)} | "
                   f"{_f(_v(r.get('cond_share_3rs_end_1e-03')), 3)} | {_f(_v(r.get('ratio_share15_over_w_m_b3')), 3)} | {_f(r['t_kin_15'], 2)}, {_f(r['t_kin_2'], 2)} | {_f(_v(r.get('t_kin_at_E')))} | {_f(r['reach_rs'], 2)} | {_e(r['norm_drift'], 1)} |")
    out.append("\n### K2. Component reconstruction: raw share = above + below + 2 Re cross per region and sweep end (identity_dev = the largest deviation when the 3 "
               "members share one batch; for 3 single-member runs the cross term is the difference and carries the 3 certificates)\n")
    out.append("| run family | class | region | t/rs | share above | share below | 2 Re cross | share raw | identity dev | source |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in recon:
        out.append(f"| {r['stem']} | {r['wl_class']} | {r['region']} | {r['t_rs']:.2f} | {_f(r['share_above'])} | {_f(r['share_below'])} | {_f(r['cross'])} | {_f(r['share_raw'])} | "
                   f"{_e(r['identity_dev'], 1)} | {r['source']} |")
    with open(path_md, 'a') as f:
        f.write('\n'.join(out) + '\n')


def norm_moments(em, es, n0):
    """The normalized mean and spread of a batch member whose stored moments are unnormalized. gap10_run_v2_wl.py stores E_mean = <psi|h|psi>
    and E_std = sqrt(<psi|h^2|psi> - <psi|h|psi>^2) on the member as it is (gap10_core_v2.LatticeH.energy divides by no norm; GAP10_T7_final_v1.2
    section 12 item 3), so with n0 = <psi|psi> the normalized mean is E_mean/n0 and the normalized variance <h^2>/n0 - (E_mean/n0)^2 with
    <h^2> = E_std^2 + E_mean^2; a raw member has n0 1 and keeps its values. The formula is this script's: if the driver's E_std is defined
    otherwise, the spread column is wrong and the mean column is not."""
    if em is None or n0 is None or not np.isfinite(n0) or n0 <= 0:
        return np.nan, np.nan
    mn = float(em) / float(n0)
    if es is None or not np.isfinite(es):
        return mn, np.nan
    return mn, math.sqrt(max((float(es) ** 2 + float(em) ** 2) / float(n0) - mn * mn, 0.0))


def band_of(share, delta):
    """The certificate band 2 s delta + delta^2 of a share s^2 at certificate delta (GAP10_T7_final_v1.2 section 3)."""
    s = math.sqrt(max(float(share), 0.0))
    return 2.0 * s * delta + delta * delta


def _keys(d):
    return list(d.files) if hasattr(d, 'files') else list(d.keys())


def pair_key(stem, kind):
    """The stem with the token that distinguishes the 2 members of a pair removed: 'prep' drops the _matched token and keeps the wavelength
    class and component tokens (a fixed raw run pairs with the matched raw run, never with a component run: GAP10_T7_final_v1.2 section 12
    item 4); 'H' replaces the box token _H<n>_ (the parent ladder); 'solver' replaces the trailing _cheb or _taylor (the second solver)."""
    s = os.path.basename(stem)
    if kind == 'prep':
        return s.replace('_matched', '')
    if kind == 'H':
        return re.sub(r'_H\d+_', '_HX_', s)
    if kind == 'solver':
        return re.sub(r'_(cheb|taylor)$', '_SOLVER', s)
    raise ValueError(kind)


def _solver_token(stem):
    m = re.search(r'_(cheb|taylor)$', os.path.basename(stem))
    return m.group(1) if m else '?'


def find_pairs(stems_meta, kind):
    """The groups of 2 or more stems sharing a pair key, each sorted (by H for the ladder, by name otherwise: a fixed stem sorts before its
    _matched twin and _cheb before _taylor)."""
    groups = {}
    for stem in stems_meta:
        groups.setdefault(pair_key(stem, kind), []).append(stem)
    out = []
    for key in sorted(groups):
        lst = groups[key]
        if len(lst) < 2:
            continue
        if kind == 'H':
            lst = sorted(lst, key=lambda s: int(stems_meta[s][1]['H']))
        else:
            lst = sorted(lst)
        out.append((key, lst))
    return out


def coincident(t1, t2, tol=1e-9):
    """The indices (i1, i2) of the time stamps 2 increasing grids share, |t1 - t2| within tol times max(1, |t|)."""
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    i1, i2, j = [], [], 0
    for i, t in enumerate(t1):
        eps = tol * max(1.0, abs(float(t)))
        while j < len(t2) and t2[j] < t - eps:
            j += 1
        if j < len(t2) and abs(t2[j] - t) <= eps:
            i1.append(i)
            i2.append(j)
    return np.array(i1, int), np.array(i2, int)


def discover_series(d, times, B):
    """The stored series by shape, the npz key printed in the tables: 'rec' = a complex array with a time axis of length len(times) and a member
    axis of length B (the receiver amplitudes, returned as (ntimes, nrec, B)); 'block' = an array with the time axis and 2 trailing axes of 4
    (the site-to-site 4 x 4 blocks at the site receivers, returned as (ntimes, nrec, 4, 4)). The first array of each kind is taken; a reader
    checks the printed key against gap10_run_v2.py's npz writer before quoting a number of F2 or S."""
    nt = len(times)
    found = {}
    for k in _keys(d):
        try:
            a = d[k]
        except Exception:
            continue
        if not isinstance(a, np.ndarray) or a.ndim not in (3, 4) or nt not in a.shape:
            continue
        a = np.moveaxis(a, list(a.shape).index(nt), 0)
        if a.ndim == 4 and a.shape[-2:] == (4, 4) and 'block' not in found:
            found['block'] = (k, a)
        elif a.ndim == 3 and np.iscomplexobj(a) and 'rec' not in found:
            if a.shape[2] == B:
                found['rec'] = (k, a)
            elif a.shape[1] == B:
                found['rec'] = (k, np.moveaxis(a, 1, 2))
    return found


def receiver_labels(meta, n):
    """Labels for n receivers from the json meta: the first list of length n under a key naming receivers, detectors or sites, else any list of
    length n other than 'src', else the indices; the key used is returned (None when none was found)."""
    cands = [k for k in meta if isinstance(meta.get(k), list) and len(meta[k]) == n and k != 'src']
    cands = sorted(cands, key=lambda k: (not any(w in k.lower() for w in ('rec', 'det', 'site')), k))
    for k in cands:
        out = []
        for x in meta[k]:
            if isinstance(x, dict):
                out.append(' '.join(f"{kk}={x[kk]}" for kk in sorted(x) if isinstance(x[kk], (int, float, str)))[:70].replace('|', '/'))
            else:
                out.append(str(x)[:70].replace('|', '/'))
        return k, out
    return None, [str(i) for i in range(n)]


def write_table_F2(stems_meta, path_md, res_factor=RES_FACTOR):
    """F2. Pairs read at a common clock (the re-pairing of table F; GAP10_T7_final_v1.2 section 12 items 4 and 15)."""
    out = ["\n## F2. Pairs read at a common clock (the re-pairing of table F, GAP10_T7_final_v1.2 section 12 items 4 and 15): for the preparation pairs "
           "(fixed against matched; the pair key keeps the wavelength class and component tokens, so a raw run pairs with a raw run), the box pairs (consecutive "
           "boxes of an H ladder on common vectors) and the solver pairs (Chebyshev against Taylor), every member's receiver amplitude in the stored receiver "
           "series is compared at the time stamps the 2 runs share and at which both are resolved (the modulus above RES_FACTOR times the run's own certificate "
           "at that stamp): 'window' is the span of those stamps, 'max abs(a1 - a2)' the largest difference of the moduli over it with its clock, 'eps1 + eps2' "
           "the summed certificates at that clock and 'ratio' their quotient; 'peak1', 'peak2' each run's largest modulus inside the window with its clock; nan "
           "where no common resolved stamp exists (one run's certificate voided the readout before the other's arrival; table F differences such readouts at 2 "
           "clocks). The npz key of the series and the receiver labels found are printed; for the preparation pairs the 2 runs' receiver packets carry different "
           "carriers (the grid is built from the preparation), so their differences are the carriers', not an error.\n"]
    out.append(f"RES_FACTOR {res_factor:g}; stamps coincide within 1e-9 of the clock.\n")
    out.append("| control | pair key | run 1 | run 2 | member (label) | receiver | series key 1 / 2 | common stamps | common resolved | window t/rs | "
               "max abs(a1 - a2) (t/rs) | eps1 + eps2 | ratio | peak1 (t/rs) | peak2 (t/rs) |")
    out.append("|" + "---|" * 15)
    nrows = 0
    for kind in ('prep', 'H', 'solver'):
        for key, lst in find_pairs(stems_meta, kind):
            for i in range(len(lst) - 1):
                s1, s2 = lst[i], lst[i + 1]
                (d1, m1), (d2, m2) = stems_meta[s1], stems_meta[s2]
                b1, b2 = os.path.basename(s1), os.path.basename(s2)
                t1, t2 = np.asarray(d1['times'], float), np.asarray(d2['times'], float)
                rs = float(m1['rs'])
                f1 = discover_series(d1, t1, int(d1['shares'].shape[1]))
                f2 = discover_series(d2, t2, int(d2['shares'].shape[1]))
                if 'rec' not in f1 or 'rec' not in f2:
                    out.append(f"| {kind} | {key} | {b1} | {b2} | - | - | none found (npz keys {' '.join(_keys(d1))}) | - | - | - | - | - | - | - | - |")
                    continue
                (k1, A1), (k2, A2) = f1['rec'], f2['rec']
                c1, c2 = np.asarray(d1['cert'], float), np.asarray(d2['cert'], float)
                i1, i2 = coincident(t1, t2)
                B = min(A1.shape[2], A2.shape[2], c1.shape[1], c2.shape[1])
                nrec = min(A1.shape[1], A2.shape[1])
                lkey, labels = receiver_labels(m1, A1.shape[1])
                src = m1.get('src', [])
                for b in range(B):
                    lab = src[b].get('label', b) if b < len(src) and isinstance(src[b], dict) else b
                    for r in range(nrec):
                        head = f"| {kind} | {key} | {b1} | {b2} | {b} ({lab}) | {r} ({labels[r]}; key {lkey}) | {k1} / {k2} | {len(i1)} |"
                        if len(i1) == 0:
                            out.append(head + " 0 | nan | nan | nan | nan | nan | nan |")
                            continue
                        a1, a2 = np.abs(A1[i1, r, b]), np.abs(A2[i2, r, b])
                        e1, e2 = c1[i1, b], c2[i2, b]
                        ok = (a1 > res_factor * e1) & (a2 > res_factor * e2)
                        if not np.any(ok):
                            out.append(head + " 0 | nan | nan | nan | nan | nan | nan |")
                            continue
                        tt = t1[i1][ok] / rs
                        diff, esum = np.abs(a1[ok] - a2[ok]), e1[ok] + e2[ok]
                        j, p1, p2 = int(np.argmax(diff)), int(np.argmax(a1[ok])), int(np.argmax(a2[ok]))
                        ratio = diff[j] / esum[j] if esum[j] > 0 else np.inf
                        out.append(head + f" {int(ok.sum())} | {tt[0]:.2f} to {tt[-1]:.2f} | {_e(diff[j], 2)} ({tt[j]:.2f}) | {_e(esum[j], 1)} | {_f(ratio, 2)} | "
                                   f"{_e(a1[ok][p1], 3)} ({tt[p1]:.2f}) | {_e(a2[ok][p2], 3)} ({tt[p2]:.2f}) |")
                        nrows += 1
    with open(path_md, 'a') as f:
        f.write('\n'.join(out) + '\n')
    print(f"[analysis v4] table F2: {nrows} rows with a common resolved window", flush=True)


def write_table_L(stems_meta, path_md):
    """L. The parent ladder at common sweep ends (GAP10_T7_final_v1.2 section 17 (c) exception 2)."""
    out = ["\n## L. The parent ladder at common sweep ends (GAP10_T7_final_v1.2 section 17 (c) exception 2): for consecutive boxes of an H ladder on the same stem "
           "family (common vectors), each member's share beyond each region at the sweep ends both boxes reach, with each box's certificate band 2 s delta + delta^2 "
           "(s^2 the share, delta the box's certificate at that end as gap10_analysis_v2.cert_at reads it), the difference of the shares and the summed bands; "
           "'within' is 1 where the difference lies inside the summed bands: the share item of the gap note's check 3 read at a common clock.\n"]
    out.append("| pair key | member (label) | region | t/rs | H_a | share_a | band_a | H_b | share_b | band_b | share_a - share_b | band_a + band_b | within |")
    out.append("|" + "---|" * 13)
    nrows = 0
    for key, lst in find_pairs(stems_meta, 'H'):
        for i in range(len(lst) - 1):
            sa, sb = lst[i], lst[i + 1]
            (da, ma), (db, mb) = stems_meta[sa], stems_meta[sb]
            rs = float(ma['rs'])
            na = [nm for nm, _ in ma['share_radii']]
            nb = [nm for nm, _ in mb['share_radii']]
            Sa, Sb = np.asarray(da['shares'], float), np.asarray(db['shares'], float)
            ta, tb = np.asarray(da['sweep_t'], float), np.asarray(db['sweep_t'], float)
            ia, ib = coincident(ta, tb)
            B = min(Sa.shape[1], Sb.shape[1])
            src = ma.get('src', [])
            for b in range(B):
                lab = src[b].get('label', b) if b < len(src) and isinstance(src[b], dict) else b
                for nm in na:
                    if nm not in nb:
                        continue
                    ka, kb = na.index(nm), nb.index(nm)
                    for ja, jb in zip(ia, ib):
                        t = float(ta[ja])
                        s_a, s_b = float(Sa[ja, b, ka]), float(Sb[jb, b, kb])
                        d_a = float(V2.cert_at(da['times'], da['cert'][:, b], t))
                        d_b = float(V2.cert_at(db['times'], db['cert'][:, b], t))
                        band_a, band_b = band_of(s_a, d_a), band_of(s_b, d_b)
                        out.append(f"| {key} | {b} ({lab}) | {nm} | {t / rs:.2f} | {ma['H']} | {s_a:.5f} | {_e(band_a, 1)} | {mb['H']} | {s_b:.5f} | {_e(band_b, 1)} | "
                                   f"{_e(s_a - s_b, 2)} | {_e(band_a + band_b, 1)} | {int(abs(s_a - s_b) <= band_a + band_b)} |")
                        nrows += 1
    with open(path_md, 'a') as f:
        f.write('\n'.join(out) + '\n')
    print(f"[analysis v4] table L: {nrows} rows", flush=True)


def _grid(t, rs):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan
    return f"{len(t)}, {dt:.4g}, {t[0] / rs:.2f} to {t[-1] / rs:.2f}"


def _first(F, t, rs, thr):
    idx = np.nonzero(F > thr)[0]
    return f"{t[idx[0]] / rs:.3f}; {F[idx[0]]:.2e}" if len(idx) else "none"


def _peak(F, t, rs):
    p = int(np.argmax(F))
    return f"{F[p]:.3e} ({t[p] / rs:.3f})"


def write_table_S(stems_meta, path_md, thr=1e-6, floor=FLOOR_AMP):
    """S. The single-site 4 x 4 block series of the solver pairs on a common time grid (GAP10_T7_final_v1.2 section 17 (d); section 12 item 16)."""
    out = ["\n## S. The single-site 4 x 4 block series at the site receivers, solver pairs on a common time grid (GAP10_T7_final_v1.2 section 17 (d) and section 12 "
           "item 16): for each Chebyshev run and its Taylor twin, the 2 recorded grids (stamps, median spacing in a, span in r_s), the stamps they share, the "
           "Frobenius norm of the block series at the shared stamps with the largest norm of the difference over them and its clock, each series' raw peak (table "
           "B's t_peak and amplitude) and first crossing of thr on its own grid, and both re-read on the shared stamps only. The rule: a difference that vanishes at "
           "the shared stamps while the peaks and crossings move with the grid is the recording grid's; a difference at the shared stamps above the second solver's "
           "floor in amplitude is a difference of the propagation or of the block's definition between the 2 drivers; the verdict prints 'grid' at or below 10 times "
           "the floor, 'not grid' at or above 100 times it, 'undecided' between, and 'undecided' with the interpolated difference when fewer than 3 stamps are shared "
           "(the finer grid interpolated linearly, entry by entry, onto the coarser one's stamps inside the overlap). The 1.5 r_s site receiver is the exception's "
           "locus; the receiver labels found in the json are printed, and where none are, 4 site receivers in increasing radius put it at index 1 (1.25, 1.5, 2, "
           "3 r_s), an assumption printed as such.\n"]
    out.append(f"thr {thr:.0e} (table B's first-arrival threshold); floor {floor:.0e} in amplitude.\n")
    out.append("| pair key | run a (solver) | run b (solver) | receiver | series key a / b | grid a: stamps, dt/a, span t/rs | grid b | shared stamps | "
               "max norm of (A_a - A_b) (t/rs) | / peak | peak a: norm (t/rs) | peak b | peaks on shared stamps a, b | first above thr a (t/rs; norm), b | "
               "first above thr on shared stamps a, b | verdict |")
    out.append("|" + "---|" * 16)
    nrows = 0
    for key, lst in find_pairs(stems_meta, 'solver'):
        for i in range(len(lst) - 1):
            sa, sb = lst[i], lst[i + 1]
            (da, ma), (db, mb) = stems_meta[sa], stems_meta[sb]
            rs = float(ma['rs'])
            ta, tb = np.asarray(da['times'], float), np.asarray(db['times'], float)
            fa = discover_series(da, ta, int(da['shares'].shape[1]))
            fb = discover_series(db, tb, int(db['shares'].shape[1]))
            na_, nb_ = f"{os.path.basename(sa)} ({_solver_token(sa)})", f"{os.path.basename(sb)} ({_solver_token(sb)})"
            if 'block' not in fa or 'block' not in fb:
                out.append(f"| {key} | {na_} | {nb_} | - | none found (npz keys a: {' '.join(_keys(da))}; b: {' '.join(_keys(db))}) | {_grid(ta, rs)} | {_grid(tb, rs)} | "
                           "- | - | - | - | - | - | - | - | no block series: undecided |")
                nrows += 1
                continue
            (ka, Ba), (kb, Bb) = fa['block'], fb['block']
            ia, ib = coincident(ta, tb)
            nrec = min(Ba.shape[1], Bb.shape[1])
            lkey, labels = receiver_labels(ma, Ba.shape[1])
            for r in range(nrec):
                Fa = np.linalg.norm(Ba[:, r].reshape(len(ta), -1), axis=1)
                Fb = np.linalg.norm(Bb[:, r].reshape(len(tb), -1), axis=1)
                pk = max(float(Fa.max()), float(Fb.max()))
                lab = f"{r} ({labels[r]}; key {lkey})" if lkey else f"{r} (index {r}; with 4 site receivers in increasing radius index 1 is 1.5 r_s, an assumption)"
                if len(ia) >= 3:
                    tc = ta[ia]
                    D = np.linalg.norm((Ba[ia, r] - Bb[ib, r]).reshape(len(ia), -1), axis=1)
                    Fac, Fbc = Fa[ia], Fb[ib]
                    how = f"{len(ia)}"
                else:
                    a_is_fine = len(ta) >= len(tb)
                    (Xf, tf), (Xc, tcs) = ((Ba[:, r], ta), (Bb[:, r], tb)) if a_is_fine else ((Bb[:, r], tb), (Ba[:, r], ta))
                    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
                    sel = (tcs >= lo) & (tcs <= hi)
                    tc = tcs[sel]
                    if len(tc) == 0:
                        out.append(f"| {key} | {na_} | {nb_} | {lab} | {ka} / {kb} | {_grid(ta, rs)} | {_grid(tb, rs)} | 0 | - | - | {_peak(Fa, ta, rs)} | {_peak(Fb, tb, rs)} | - | "
                                   f"{_first(Fa, ta, rs, thr)}, {_first(Fb, tb, rs, thr)} | - | undecided (no common clock) |")
                        nrows += 1
                        continue
                    Xf2 = Xf.reshape(len(tf), -1)
                    Xi = np.stack([np.interp(tc, tf, Xf2[:, c].real) + 1j * np.interp(tc, tf, Xf2[:, c].imag) for c in range(Xf2.shape[1])], axis=1)
                    Xc2 = Xc.reshape(len(tcs), -1)[sel]
                    D = np.linalg.norm(Xi - Xc2, axis=1)
                    Fi, Fc = np.linalg.norm(Xi, axis=1), np.linalg.norm(Xc2, axis=1)
                    Fac, Fbc = (Fi, Fc) if a_is_fine else (Fc, Fi)
                    how = f"{len(ia)} (interpolated onto {len(tc)})"
                j = int(np.argmax(D))
                dmax = float(D[j])
                if len(ia) >= 3:
                    verdict = ("grid: the series agree at the shared stamps within 10 times the floor" if dmax <= 10 * floor else
                               "not grid: the series differ at the shared stamps by 100 times the floor or more" if dmax >= 100 * floor else
                               "undecided: between 10 and 100 times the floor at the shared stamps")
                else:
                    verdict = ("grid: the interpolated series agree within 10 times the floor" if dmax <= 10 * floor else
                               "undecided: no shared stamps; the interpolated difference carries the interpolation error of a beating signal")
                out.append(f"| {key} | {na_} | {nb_} | {lab} | {ka} / {kb} | {_grid(ta, rs)} | {_grid(tb, rs)} | {how} | {dmax:.3e} ({tc[j] / rs:.3f}) | "
                           f"{(dmax / pk if pk > 0 else np.nan):.2e} | {_peak(Fa, ta, rs)} | {_peak(Fb, tb, rs)} | {_peak(Fac, tc, rs)}, {_peak(Fbc, tc, rs)} | "
                           f"{_first(Fa, ta, rs, thr)}, {_first(Fb, tb, rs, thr)} | {_first(Fac, tc, rs, thr)}, {_first(Fbc, tc, rs, thr)} | {verdict} |")
                nrows += 1
    with open(path_md, 'a') as f:
        f.write('\n'.join(out) + '\n')
    print(f"[analysis v4] table S: {nrows} rows", flush=True)


def rehead_md(path_md):
    """The header words of gap10_analysis_v2's md (GAP10_T7_final_v1.2 section 12 item 13): in the file header, before the first section,
    'analysis v2' becomes 'analysis v4 (...)'; everywhere, 'Frozen control background' becomes 'held-field control (the record's applied fields
    held fixed)'. The counts are printed; 0 means the words were not found and the header is v2's as written."""
    with open(path_md) as f:
        txt = f.read()
    head, sep, rest = txt.partition('\n## ')
    n1 = head.count('analysis v2')
    head = head.replace('analysis v2', "analysis v4 (gap10_analysis_v2.py's tables, re-headed; table K extended; tables F2, L, S appended)")
    txt = head + sep + rest
    n2 = txt.count('Frozen control background')
    txt = txt.replace('Frozen control background', "held-field control (the record's applied fields held fixed)")
    with open(path_md, 'w') as f:
        f.write(txt)
    print(f"[analysis v4] md header: {n1} 'analysis v2' and {n2} 'Frozen control background' replaced", flush=True)


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
    # v4: the helpers of the extended and appended tables
    w1, w2, ea, eb = 0.3, 0.5, -0.1, 0.4           # a 2-level state with weights w1, w2 at energies ea, eb, unnormalized (norm 0.8)
    n0 = w1 + w2
    em = w1 * ea + w2 * eb
    e2m = w1 * ea ** 2 + w2 * eb ** 2
    mn, sn = norm_moments(em, math.sqrt(e2m - em ** 2), n0)
    say(abs(mn - em / n0) < 1e-12 and abs(sn - math.sqrt(e2m / n0 - (em / n0) ** 2)) < 1e-12, "normalized mean and spread from the stored unnormalized moments (2-level state)")
    say(all(np.isnan(norm_moments(None, None, 1.0))) and norm_moments(0.5, 0.1, 1.0)[0] == 0.5 and abs(norm_moments(0.5, 0.1, 1.0)[1] - 0.1) < 1e-12,
        "normalized moments: an absent field prints nan; a raw member (norm0 1) is unchanged")
    say(pair_key('x_H88_100_d4_site_cheb', 'H') == pair_key('x_H124_100_d4_site_cheb', 'H'), "pair key: the H ladder")
    say(pair_key('x_100_d2_both_cheb', 'prep') == pair_key('x_100_d2_both_matched_cheb', 'prep'), "pair key: the preparation twins")
    say(pair_key('a_wlin_raw_cheb', 'prep') != pair_key('a_wlin_above_matched_cheb', 'prep'), "pair key: a raw run never pairs with a component run")
    say(pair_key('y_d4_site_cheb', 'solver') == pair_key('y_d4_site_taylor', 'solver') != pair_key('y_d8_site_cheb', 'solver'), "pair key: the solver pair")
    i1, i2 = coincident(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 2.0, 4.0]))
    say(list(i1) == [0, 2] and list(i2) == [0, 1], f"coincident stamps ({list(i1)}, {list(i2)})")
    say(abs(band_of(0.25, 1e-3) - (2 * 0.5 * 1e-3 + 1e-6)) < 1e-18, "certificate band 2 s delta + delta^2 from the share s^2")
    Y = rng.normal(size=(6, 2, 4, 4)) + 1j * rng.normal(size=(6, 2, 4, 4))
    dd = {'times': np.arange(6.0), 'shares': np.zeros((3, 5, 2)), 'blk': Y, 'amp': rng.normal(size=(6, 3, 5)) + 0j}
    fnd = discover_series(dd, dd['times'], 5)
    say(fnd.get('block', (None,))[0] == 'blk' and fnd.get('rec', (None,))[0] == 'amp' and fnd['rec'][1].shape == (6, 3, 5),
        "series discovery by shape (block (nt, nrec, 4, 4); receivers (nt, nrec, B))")
    print(f"[selftest v4] {len(fails)} failures beyond v2's exit code {rc}", flush=True)
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
    print(f"[analysis v4] core {V2.C.__name__} (v2 tables unchanged; table K extended; tables F2, L, S appended; md header re-worded)", flush=True)
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
    try:
        rehead_md(args.md)
    except Exception as e:
        print('  md re-heading failed:', repr(e))
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
    all_meta = dict(stems_meta)
    for s in stems:
        if s in all_meta:
            continue
        try:
            all_meta[s] = V2.load(s)
        except Exception as e:
            print('  load failed for the appended tables:', s, repr(e))
    for name, fn in (('F2', write_table_F2), ('L', write_table_L), ('S', write_table_S)):
        try:
            fn(all_meta, args.md)
        except Exception as e:
            print(f'  table {name} failed:', repr(e))
    print(f"[analysis] {len(rows)} readout rows -> {args.csv}; {len(trows)} transmissivity rows -> {args.tcsv}; {len(krows)} wavelength rows -> {args.kcsv}; tables -> {args.md} (K extended; F2, L, S appended)")


if __name__ == '__main__':
    main()
