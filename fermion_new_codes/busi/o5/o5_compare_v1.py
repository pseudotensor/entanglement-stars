#!/usr/bin/env python
"""o5_compare_v1.py -- O5 (GAP10_T7_final_v1.1 section 9): the comparison of the 2 propagations of the same lifted column and the
acceptance rule.

For every pair (chain export, lift, cube run) and every sweep end 1 .. S: the weight beyond r_h and beyond 1.5, 2 and 3 r_s, the
chain's (the cell rule by default; --chain_rule strict) against the cube's (shares), each with its band 2 sqrt(p) delta + delta^2 (the
cube's certificate delta at the sweep end; the chain's stated propagation error), plus the lift's Gram defect: the acceptance rule is
agreement within the sum of the stated errors; a residual beyond it is printed as the difference between the 2 descriptions (the
angular channel against the genuine cubic lattice), labelled physics and not error; where that residual lies within the chain's
cell-rule ambiguity (the half cell at a cut that is an integer radius) this is said beside it; a residual beyond the sum on a column
whose lattice-scale weight exceeds --lattice_scale_min is named the lattice-scale difference between a chain and the cube. The
fixed-cut number current through r_h at the sweep ends is compared with the bands 2 ||h|| (2 delta + delta^2) and the lift term
2 ||h|| delta_G (stated as such); its time integral is the beyond-r_h weight's change on both descriptions (the identity, printed:
exact on the chain, the trapezoid over the sweep ends on the cube). The detector amplitudes (the lifted detector packets, the same
object in both descriptions) are compared in modulus at every readout time within delta_cube + delta_chain + the amplitude lift error,
their peak moduli and peak clocks within --clock_tol_rs r_s; the half-weight crossing clocks of p beyond 1.5 r_s (half the value at
the last sweep end, HORIZON's rule) within --clock_tol_rs r_s (the sweep's 1 r_s is the readout's resolution). The axis packets
(GAP10's own 3-dimensional packets against their channel projections on the chain) are printed without a verdict. The Killing splits
of both descriptions are printed (b1 and b3 both), not compared. Writes <out>.json and <out>.md; no plots. No digit here is the object's.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from o5_chain_export_v1 import half_weight_clock  # noqa: E402

RADII = ['r_h', '1.5rs', '2rs', '3rs']


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def weight_band(p, delta):
    s = math.sqrt(max(float(p), 0.0))
    return 2.0 * s * float(delta) + float(delta) ** 2


def _index_of_time(times, t):
    i = int(np.argmin(np.abs(times - t)))
    if abs(times[i] - t) > 1e-9:
        raise ValueError(f"time {t} is not a readout time of this run")
    return i


# ------------------------------------------------------------------ loading
def load_pair(chain_json, lift_json, cube_json, opts):
    with open(chain_json) as fh:
        J = json.load(fh)
    Zc = np.load(J['files']['npz'] if os.path.exists(J['files']['npz']) else chain_json[:-5] + '.npz')
    with open(lift_json) as fh:
        Lf = json.load(fh)
    with open(cube_json) as fh:
        K = json.load(fh)
    Zk = np.load(cube_json[:-5] + '.npz')
    rs = float(J['readout']['rs'])
    sweep_t_ch = np.asarray(Zc['sweep_t'], dtype=float)
    sweep_t_cu = np.asarray(Zk['sweep_t'], dtype=float)
    S = min(sweep_t_ch.size, sweep_t_cu.size) - 1
    if not np.allclose(sweep_t_ch[:S + 1], sweep_t_cu[:S + 1], atol=1e-9):
        raise ValueError(f"the sweep ends differ: chain {sweep_t_ch[:S + 1]} against cube {sweep_t_cu[:S + 1]}")
    times_ch, times_cu = np.asarray(Zc['times'], dtype=float), np.asarray(Zk['times'], dtype=float)
    nt = min(times_ch.size, times_cu.size)
    if not np.allclose(times_ch[:nt], times_cu[:nt], atol=1e-9):
        raise ValueError("the readout times differ between the 2 descriptions")
    times = times_ch[:nt]
    sweep_idx_ch = np.asarray(Zc['sweep_idx'], dtype=int)[:S + 1]
    sweep_idx_cu = np.array([_index_of_time(times_cu, t) for t in sweep_t_cu[:S + 1]], dtype=int)
    cube_names = [str(x[0]) for x in K['share_radii']]
    chain_names = list(J['readout']['radii_names'])
    radii = [nm for nm in RADII if nm in cube_names and nm in chain_names]
    shares = np.asarray(Zk['shares'], dtype=float)            # (S+1, B, 5)
    cert = np.asarray(Zk['cert'], dtype=float)                # (nt, B)
    p_mid, p_strict = np.asarray(Zc['p_mid'], dtype=float), np.asarray(Zc['p_strict'], dtype=float)
    prim, alt = (p_mid, p_strict) if opts['chain_rule'] == 'midpoint' else (p_strict, p_mid)
    chain_p = {nm: prim[sweep_idx_ch, chain_names.index(nm)] for nm in radii}
    chain_alt = {nm: alt[sweep_idx_ch, chain_names.index(nm)] for nm in radii}
    cube_p = {nm: shares[:S + 1, 0, cube_names.index(nm)] for nm in radii}
    prop_err = np.asarray(Zc['prop_err'], dtype=float)
    chain_delta = prop_err[sweep_idx_ch]
    cube_delta = cert[sweep_idx_cu, 0]
    Jch = np.asarray(Zc['J'], dtype=float)[sweep_idx_ch]
    Jcu = np.asarray(Zk['o5_current_rh'], dtype=float)[:S + 1, 0] if 'o5_current_rh' in Zk.files else None
    labels = list(K['labels'])
    amps = []
    for i, md in enumerate(J['detector']['members']):
        lab = md['label']
        if lab in labels:
            amps.append(dict(label=lab, sign=md['sign'], k=md['k'], chain=np.asarray(Zc['amp'])[:nt, i], cube=np.asarray(Zk['A'])[:nt, 0, labels.index(lab)]))
        else:
            amps.append(dict(label=lab, sign=md['sign'], k=md['k'], chain=np.asarray(Zc['amp'])[:nt, i], cube=None))
    axis = []
    dz_path = Lf['files']['det_npz']
    if os.path.exists(dz_path):
        Zd = np.load(dz_path)
        if 'axis_proj' in Zd.files and Zd['axis_proj'].shape[0] > 0:
            states = np.asarray(Zc['states'])[:nt]
            for g, lab in zip(np.asarray(Zd['axis_proj']), [str(x) for x in Zd['axis_labels']]):
                a_ch = np.einsum('js,tjs->t', np.conj(g), states)
                a_cu = np.asarray(Zk['A'])[:nt, 0, labels.index(lab)] if lab in labels else None
                axis.append(dict(label=lab, chain=a_ch, cube=a_cu, projection_norm=float(np.sum(np.abs(g) ** 2))))
    killing_cube = (K.get('wl', {}).get('killing', {}) or {}).get('o5')
    src0 = (K.get('src') or [{}])[0]
    cut_cube = float(K.get('o5_rh') or 0.0) or float(K.get('r_h'))
    return dict(name=f"Rc{J['column']['Rc']:g}_k{J['chain']['kappa']}_H{K['H']}", chain_json=chain_json, lift_json=lift_json, cube_json=cube_json, rs=rs,
                sweep_t=sweep_t_ch[:S + 1], radii=radii, times=times,
                column=dict(Rc=J['column']['Rc'], sigma=J['column']['sigma'], ak0=J['column']['ak0'], kappa=J['chain']['kappa'], E_mean=J['column']['E_mean'],
                            E_rms=J['column']['E_rms'], spinor_mode=J['column']['spinor_mode']),
                chain=dict(p=chain_p, p_alt=chain_alt, delta=chain_delta, J=Jch, Jint_exact=np.asarray(Zc['Jint_exact_sweeps'], dtype=float)[:S + 1],
                           dp_strict=(p_strict[sweep_idx_ch, 0] - p_strict[0, 0]), wmax=float(J['chain']['eigen']['abs_max']), amp_delta=prop_err[:nt],
                           lattice_scale=float(J['lattice_scale_weight']), killing=J['killing'], r_h=float(J['record']['r_h']),
                           identity_defect_exact_max=float(J['checks']['identity_defect_exact_max']), prop_err_last=float(prop_err[sweep_idx_ch[-1]])),
                cube=dict(p=cube_p, delta=cube_delta, J=Jcu, bound=float(K['bound']), amp_delta=cert[:nt, 0], lattice_scale=float((src0.get('band') or {}).get('lattice_scale', float('nan'))),
                          killing=killing_cube, r_h=cut_cube, r_h_direction=float(K['r_h']), H=int(K['H']), napply=K.get('napply'), wall=K.get('wall_total'),
                          energy_mean=src0.get('energy_mean'), energy_std=src0.get('energy_std'), norm_drift=float(np.max(np.abs(np.asarray(Zk['norms'])[:, 0] / np.asarray(Zk['norms'])[0, 0] - 1.0)))),
                lift=dict(gram_defect=float(Lf['source']['gram_defect']), amp_lift_error=float(Lf['amp_lift_error']), W_lift=float(Lf['source']['W_lift']),
                          reduction=Lf.get('reduction'), det_gram=[m_['gram_defect'] for m_ in Lf['detector']['members']]),
                amps=amps, axis=axis)


# ------------------------------------------------------------------ the comparison
def compare_pair(pair, opts):
    rs = pair['rs']
    sweep_t = np.asarray(pair['sweep_t'], dtype=float)
    S = sweep_t.size - 1
    dG = pair['lift']['gram_defect']
    res = dict(name=pair['name'], column=pair['column'], files=dict(chain=pair.get('chain_json'), lift=pair.get('lift_json'), cube=pair.get('cube_json')),
               cut=dict(chain_r_h=pair['chain']['r_h'], cube_cut=pair['cube']['r_h'], cube_direction_r_h=pair['cube'].get('r_h_direction'),
                        difference=pair['cube']['r_h'] - pair['chain']['r_h']),
               errors=dict(gram_defect=dG, amp_lift_error=pair['lift']['amp_lift_error'], chain_prop_err_last=pair['chain'].get('prop_err_last'),
                           cube_cert_last=float(pair['cube']['delta'][-1]), cert_max=opts['cert_max']),
               shells={}, current=[], detectors=[], axis=[], clocks={}, lattice_scale={}, killing=dict(chain=pair['chain'].get('killing'), cube=pair['cube'].get('killing')), summary={})
    n_agree = n_phys = n_cert = 0
    worst_ratio, worst = 0.0, None
    phys_residuals = []
    for nm in pair['radii']:
        rows = []
        for s in range(1, S + 1):
            p1, p2 = float(pair['chain']['p'][nm][s]), float(pair['cube']['p'][nm][s])
            d1, d2 = float(pair['chain']['delta'][s]), float(pair['cube']['delta'][s])
            b1, b2 = weight_band(p1, d1), weight_band(p2, d2)
            tot = b1 + b2 + dG
            resid = p2 - p1
            amb = abs(float(pair['chain']['p_alt'][nm][s]) - p1)
            certified = d2 <= opts['cert_max']
            if abs(resid) <= tot:
                verdict = 'agree'
            elif abs(resid) <= tot + amb:
                verdict = 'physics (within the chain cell-rule ambiguity)'
            else:
                verdict = 'physics'
            ratio = abs(resid) / tot if tot > 0 else float('inf')
            rows.append(dict(s=s, t=float(sweep_t[s]), t_rs=float(sweep_t[s] / rs), chain=p1, chain_band=b1, cube=p2, cube_band=b2, cube_cert=d2, lift=dG, error_sum=tot,
                             residual=resid, ratio=ratio, cell_rule_ambiguity=amb, certified=certified, verdict=verdict))
            if certified:
                n_cert += 1
                if verdict == 'agree':
                    n_agree += 1
                else:
                    n_phys += 1
                    phys_residuals.append(abs(resid))
                if ratio > worst_ratio:
                    worst_ratio, worst = ratio, (nm, s, resid, tot)
        res['shells'][nm] = rows
    # the half-weight crossing clocks of p beyond 1.5 r_s
    if '1.5rs' in pair['radii']:
        c_ch = half_weight_clock(sweep_t, np.asarray(pair['chain']['p']['1.5rs'], dtype=float))
        c_cu = half_weight_clock(sweep_t, np.asarray(pair['cube']['p']['1.5rs'], dtype=float))
        res['clocks']['half_weight_1p5rs'] = dict(chain=c_ch, cube=c_cu, chain_rs=c_ch / rs, cube_rs=c_cu / rs, difference_rs=(c_cu - c_ch) / rs,
                                                  tolerance_rs=opts['clock_tol_rs'], agree=bool(abs(c_cu - c_ch) <= opts['clock_tol_rs'] * rs) if np.isfinite(c_cu - c_ch) else None)
    # the fixed-cut current
    hb_cu, hb_ch = pair['cube']['bound'], pair['chain']['wmax']
    dp_cu = np.asarray(pair['cube']['p']['r_h'], dtype=float) - float(pair['cube']['p']['r_h'][0]) if 'r_h' in pair['radii'] else None
    Jcu = pair['cube']['J']
    trap = None
    if Jcu is not None:
        trap = np.concatenate([[0.0], np.cumsum(0.5 * (Jcu[1:] + Jcu[:-1]) * np.diff(sweep_t))])
    for s in range(1, S + 1):
        j1 = float(pair['chain']['J'][s])
        d1, d2 = float(pair['chain']['delta'][s]), float(pair['cube']['delta'][s])
        row = dict(s=s, t_rs=float(sweep_t[s] / rs), chain=j1, chain_band=2.0 * hb_ch * (2.0 * d1 + d1 * d1), chain_int_exact=float(pair['chain']['Jint_exact'][s]),
                   chain_dp_strict=float(pair['chain']['dp_strict'][s]), cube_dp=(float(dp_cu[s]) if dp_cu is not None else None), certified=bool(d2 <= opts['cert_max']))
        if Jcu is not None:
            j2 = float(Jcu[s])
            b2 = 2.0 * hb_cu * (2.0 * d2 + d2 * d2)
            lift = 2.0 * hb_cu * dG
            tot = row['chain_band'] + b2 + lift
            row.update(cube=j2, cube_band=b2, lift=lift, error_sum=tot, residual=j2 - j1, verdict=('agree' if abs(j2 - j1) <= tot else 'physics'),
                       cube_trapezoid_int=float(trap[s]), cube_identity_trapezoid_defect=(float(trap[s] - dp_cu[s]) if dp_cu is not None else None))
        res['current'].append(row)
    # the detectors (the same lifted object in both descriptions)
    for a in pair['amps']:
        A1 = np.abs(np.asarray(a['chain']))
        i1 = int(np.argmax(A1))
        d = dict(label=a['label'], sign=a['sign'], k=a['k'], chain_peak=float(A1[i1]), chain_peak_t_rs=float(pair['times'][i1] / rs))
        if a['cube'] is not None:
            A2 = np.abs(np.asarray(a['cube']))
            i2 = int(np.argmax(A2))
            band = np.asarray(pair['cube']['amp_delta'], dtype=float) + np.asarray(pair['chain']['amp_delta'], dtype=float) + pair['lift']['amp_lift_error']
            resid = A2 - A1
            excess = np.maximum(np.abs(resid) - band, 0.0)
            d.update(cube_peak=float(A2[i2]), cube_peak_t_rs=float(pair['times'][i2] / rs), peak_modulus_residual=float(A2[i2] - A1[i1]),
                     peak_modulus_error_sum=float(band[i1] + band[i2]), peak_modulus_verdict=('agree' if abs(A2[i2] - A1[i1]) <= band[i1] + band[i2] else 'physics'),
                     peak_clock_difference_rs=float((pair['times'][i2] - pair['times'][i1]) / rs), peak_clock_agree=bool(abs(pair['times'][i2] - pair['times'][i1]) <= opts['clock_tol_rs'] * rs),
                     max_abs_residual=float(np.max(np.abs(resid))), max_excess_over_error=float(np.max(excess)), fraction_of_times_within_error=float(np.mean(excess == 0.0)),
                     amp_lift_error=pair['lift']['amp_lift_error'])
        res['detectors'].append(d)
    for a in pair['axis']:
        A1 = np.abs(np.asarray(a['chain']))
        i1 = int(np.argmax(A1))
        d = dict(label=a['label'], projection_norm=a['projection_norm'], chain_peak=float(A1[i1]), chain_peak_t_rs=float(pair['times'][i1] / rs))
        if a['cube'] is not None:
            A2 = np.abs(np.asarray(a['cube']))
            i2 = int(np.argmax(A2))
            d.update(cube_peak=float(A2[i2]), cube_peak_t_rs=float(pair['times'][i2] / rs), max_abs_residual=float(np.max(np.abs(A2 - A1))))
        d['note'] = "the cube's own axis packet against its channel projection on the chain (the adjoint lift): not a same-object detector; no verdict"
        res['axis'].append(d)
    # the residual against the lattice-scale weight
    lsc, lsk = pair['chain']['lattice_scale'], pair['cube']['lattice_scale']
    maxphys = max(phys_residuals) if phys_residuals else 0.0
    label = None
    if phys_residuals:
        label = ("the lattice-scale difference between a chain and the cube" if max(lsc, (lsk if np.isfinite(lsk) else 0.0)) > opts['lattice_scale_min']
                 else "the angular channel against the genuine cubic lattice (long-wavelength or threshold column)")
    res['lattice_scale'] = dict(chain_windowed_share_above_pi_over_4=lsc, cube_band_weights_lattice_scale=lsk, threshold=opts['lattice_scale_min'],
                                max_physics_residual_certified=maxphys, label=label)
    fallback = bool(dG > min([weight_band(float(pair['cube']['p'][nm][s]), float(pair['cube']['delta'][s])) for nm in pair['radii'] for s in range(1, S + 1) if pair['cube']['delta'][s] <= opts['cert_max']] or [float('inf')]))
    res['summary'] = dict(certified_sweep_end_readings=n_cert, agree=n_agree, physics=n_phys, worst_ratio_residual_over_error_sum=worst_ratio,
                          worst=(dict(shell=worst[0], s=worst[1], residual=worst[2], error_sum=worst[3]) if worst else None),
                          gram_defect_exceeds_every_certificate_band=fallback,
                          fallback_indicated=("yes: the lift's Gram defect exceeds the shares' certificate bands; section 9's fallback (the spherical shell of radius 20 a and width 6 a, --spinor rest --ak0 0 --sigma 6 on both codes) applies" if fallback else "no"),
                          reduction_convention_ok=((pair['lift'].get('reduction') or {}).get('default_is_closest')),
                          sentence=(f"{n_agree} of {n_cert} certified sweep-end shell readings agree within the sum of the stated errors; {n_phys} exceed it and are the difference between "
                                    f"the 2 descriptions, labelled physics and not error" + (f" ({label})" if label else "") + "; no digit of this comparison is the object's."))
    return res


# ------------------------------------------------------------------ writing
def _f(x, fmt='%.5f'):
    try:
        return fmt % float(x) if x is not None and np.isfinite(float(x)) else '-'
    except (TypeError, ValueError):
        return '-'


def write_md(results, path, opts):
    L = ["# O5 witness exchange: the 2 propagations of the same lifted column (GAP10_T7_final_v1.1 section 9), o5_compare_v1.py", "",
         f"Acceptance rule: agreement at every sweep end within the sum of the stated errors (the cube's certificate band 2 sqrt(p) delta + delta^2, the chain's "
         f"propagation-error band of the same form, the lift's Gram defect); a residual beyond that sum is the difference between the 2 descriptions, labelled physics "
         f"and not error. Chain shells by the {opts['chain_rule']} rule; readings with the cube's certificate above {opts['cert_max']:g} are printed as uncertified and "
         f"not counted. Clocks within {opts['clock_tol_rs']:g} r_s (the sweep). Every number is a reading on 1 held record at 1 averaging width; no digit is the object's.", ""]
    L += ["| pair | column | <h> chain | lift Gram defect | cube cert at last end | chain prop. error at last end | certified readings: agree / physics | worst residual / error sum | half-weight clock chain, cube (r_s) | detector peak clocks chain, cube (r_s) | lattice-scale weight chain; cube | fallback |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for R in results:
        c = R['clocks'].get('half_weight_1p5rs', {})
        dets = R['detectors']
        L.append(f"| {R['name']} | kappa {R['column']['kappa']}, R_c {R['column']['Rc']:g}, sigma {R['column']['sigma']:g}, a k_0 {R['column']['ak0']:+.4f}, {R['column']['spinor_mode']} | "
                 f"{R['column']['E_mean']:.5f} | {R['errors']['gram_defect']:.2e} | {R['errors']['cube_cert_last']:.2e} | {_f(R['errors']['chain_prop_err_last'], '%.1e')} | "
                 f"{R['summary']['agree']} / {R['summary']['physics']} of {R['summary']['certified_sweep_end_readings']} | {R['summary']['worst_ratio_residual_over_error_sum']:.2f} | "
                 f"{_f(c.get('chain_rs'), '%.2f')}, {_f(c.get('cube_rs'), '%.2f')} | " + '; '.join(f"{_f(d.get('chain_peak_t_rs'), '%.2f')}, {_f(d.get('cube_peak_t_rs'), '%.2f')}" for d in dets)
                 + f" | {R['lattice_scale']['chain_windowed_share_above_pi_over_4']:.3f}; {_f(R['lattice_scale']['cube_band_weights_lattice_scale'], '%.3f')} | {R['summary']['fallback_indicated'].split(':')[0]} |")
    L.append("")
    for R in results:
        L += [f"## {R['name']}: kappa {R['column']['kappa']}, R_c {R['column']['Rc']:g} a, sigma {R['column']['sigma']:g} a, a k_0 {R['column']['ak0']:+.5f}, spinor {R['column']['spinor_mode']}; "
              f"chain <h> {R['column']['E_mean']:.6f} (rms {R['column']['E_rms']:.5f})", "",
              f"- Files: chain {R['files']['chain']}; lift {R['files']['lift']}; cube {R['files']['cube']}.",
              f"- The cut: chain r_h {R['cut']['chain_r_h']:.6f} a; the cube's shares and current read the cut {R['cut']['cube_cut']:.6f} a (difference {R['cut']['difference']:+.1e} a; the direction's own crossing {_f(R['cut']['cube_direction_r_h'], '%.6f')} a).",
              f"- Errors: lift Gram defect {R['errors']['gram_defect']:.3e} (shell weights), amplitude lift error {R['errors']['amp_lift_error']:.3e} (the maximal shell Gram deviation on the supports); "
              f"chain propagation error at the last sweep end {_f(R['errors']['chain_prop_err_last'], '%.2e')}; the cube's certificate at the last sweep end {R['errors']['cube_cert_last']:.2e}.",
              f"- The lift's reduction check chose the written convention: {R['summary']['reduction_convention_ok']}.",
              f"- {R['summary']['sentence']}", f"- Fallback: {R['summary']['fallback_indicated']}.", ""]
        for nm, rows in R['shells'].items():
            L += [f"### weight beyond {nm}", "", "| t/r_s | chain p (band) | cube p (certificate band) | lift Gram defect | sum of errors | residual cube - chain | residual/sum | cell-rule ambiguity | certified | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r_ in rows:
                L.append(f"| {r_['t_rs']:.1f} | {r_['chain']:.6f} ({r_['chain_band']:.1e}) | {r_['cube']:.6f} ({r_['cube_band']:.1e}) | {r_['lift']:.1e} | {r_['error_sum']:.2e} | {r_['residual']:+.2e} | "
                         f"{r_['ratio']:.2f} | {r_['cell_rule_ambiguity']:.1e} | {'yes' if r_['certified'] else 'no'} | {r_['verdict']} |")
            L.append("")
        L += ["### the fixed-cut number current through r_h at the sweep ends (outward positive)", "",
              "| t/r_s | chain J (band) | chain exact int J | chain strict weight change | cube J (band) | lift term | sum | residual | verdict | cube weight change | cube trapezoid int J minus weight change |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r_ in R['current']:
            L.append(f"| {r_['t_rs']:.1f} | {r_['chain']:+.4e} ({r_['chain_band']:.1e}) | {r_['chain_int_exact']:+.6f} | {r_['chain_dp_strict']:+.6f} | "
                     + (f"{r_['cube']:+.4e} ({r_['cube_band']:.1e}) | {r_['lift']:.1e} | {r_['error_sum']:.2e} | {r_['residual']:+.2e} | {r_['verdict']}" if 'cube' in r_ else "- | - | - | - | (no cube current: run with --o5_current)")
                     + f" | {_f(r_['cube_dp'], '%+.6f')} | {_f(r_.get('cube_identity_trapezoid_defect'), '%+.2e')} |")
        L += ["", "The chain's exact integral equals its strict weight change to rounding (the fixed-cut identity); the cube's current is read at the sweep ends only, so its trapezoid integral is a crude check of the same identity, not a measurement of it.", "",
              "### the detector amplitudes at 2 r_s (the lifted chain detector, the same object in both descriptions; both carrier signs read against their own members)", "",
              "| member | |A| peak chain (at t/r_s) | |A| peak cube (at t/r_s) | peak modulus residual (error sum) | verdict | peak clock difference (r_s) | clocks agree | max |residual| over the readout | max excess over the error | fraction of readout times within error |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for d in R['detectors']:
            L.append(f"| {d['label']} | {d['chain_peak']:.4e} ({d['chain_peak_t_rs']:.2f}) | {_f(d.get('cube_peak'), '%.4e')} ({_f(d.get('cube_peak_t_rs'), '%.2f')}) | {_f(d.get('peak_modulus_residual'), '%+.2e')} ({_f(d.get('peak_modulus_error_sum'), '%.1e')}) | "
                     f"{d.get('peak_modulus_verdict', '-')} | {_f(d.get('peak_clock_difference_rs'), '%+.2f')} | {d.get('peak_clock_agree', '-')} | {_f(d.get('max_abs_residual'), '%.2e')} | {_f(d.get('max_excess_over_error'), '%.2e')} | {_f(d.get('fraction_of_times_within_error'), '%.3f')} |")
        if R['axis']:
            L += ["", "### GAP10's axis packets at the +nhat 2 r_s site against their channel projections on the chain (not a same-object detector; no verdict)", "",
                  "| packet | channel projection norm | |A| peak chain (t/r_s) | |A| peak cube (t/r_s) | max |residual| |", "|---|---|---|---|---|"]
            for d in R['axis']:
                L.append(f"| {d['label']} | {d['projection_norm']:.4f} | {d['chain_peak']:.4e} ({d['chain_peak_t_rs']:.2f}) | {_f(d.get('cube_peak'), '%.4e')} ({_f(d.get('cube_peak_t_rs'), '%.2f')}) | {_f(d.get('max_abs_residual'), '%.2e')} |")
        c = R['clocks'].get('half_weight_1p5rs')
        if c:
            L += ["", f"### clocks: the half-weight crossing of p beyond 1.5 r_s (half its value at the last sweep end, on the sweep ends): chain {_f(c['chain_rs'], '%.2f')} r_s, cube {_f(c['cube_rs'], '%.2f')} r_s, "
                      f"difference {_f(c['difference_rs'], '%+.2f')} r_s against the tolerance {c['tolerance_rs']:g} r_s: agree {c['agree']}"]
        ls_ = R['lattice_scale']
        L += ["", f"### the residual against the column's lattice-scale weight: chain unsigned windowed share at folded |q| > pi/4 {ls_['chain_windowed_share_above_pi_over_4']:.4f}, the cube's band_weights lattice scale of the lifted column "
                  f"{_f(ls_['cube_band_weights_lattice_scale'], '%.4f')} (threshold {ls_['threshold']:g}); the largest certified physics residual {ls_['max_physics_residual_certified']:.2e}; label: {ls_['label']}", ""]
        kc, kk = (R['killing'] or {}).get('chain'), (R['killing'] or {}).get('cube')
        if kc:
            for nm in ('b1', 'b3'):
                bb = kc[nm]['bins']
                th = kc[nm]['threshold_columns']
                L.append(f"- Chain Killing split (exact), edges +-{kc[nm]['edges']['m']:.4f}, +-{kc[nm]['edges']['b']:.4f} ({nm}): below -b {bb['below_minus_b']:.2e}; (-b, -m) {bb['minus_b_to_minus_m']:.4f}; gap {bb['gap']:.4f}; (m, b) {bb['m_to_b']:.4f}; above b {bb['above_b']:.2e}; "
                         f"threshold columns (assigned to neither) -b {th['-b']:.1e}, -m {th['-m']:.1e}, +m {th['+m']:.1e}, +b {th['+b']:.1e}.")
        if kk:
            for nm, key in (('b3', 'bins_b3'), ('b1', 'bins_b1')):
                bb = kk.get(key) or {}
                L.append(f"- Cube Killing split (Jackson-KPM, resolution {_f(kk.get('resolution'), '%.4f')}, the weight within a resolution of +m {_f(kk.get('w_trans_plus'), '%.4f')} assigned to neither; {nm} = {_f(kk.get(nm), '%.4f')}): "
                         f"below -b {_f(bb.get('below_-b'), '%.2e')}; (-b, -m) {_f(bb.get('-b_to_-m'), '%.4f')}; gap {_f(bb.get('gap'), '%.4f')}; (m, b) {_f(bb.get('m_to_b'), '%.4f')}; above b {_f(bb.get('above_b'), '%.2e')}.")
        L.append("")
    text = "\n".join(L) + "\n"
    with open(path, 'w') as fh:
        fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description="O5: compare the chain's and the cube's propagations of the same lifted column; the acceptance rule of section 9")
    ap.add_argument('--pair', nargs=3, action='append', metavar=('CHAIN_JSON', 'LIFT_JSON', 'CUBE_JSON'), required=True, help="repeatable")
    ap.add_argument('--out', default='o5_compare_v1', help="writes <out>.json and <out>.md")
    ap.add_argument('--chain_rule', choices=['midpoint', 'strict'], default='midpoint', help="the chain's shell rule compared with the cube's site sum")
    ap.add_argument('--cert_max', type=float, default=1e-3, help="a sweep end counts as certified when the cube's certificate is below this")
    ap.add_argument('--clock_tol_rs', type=float, default=1.0, help="clock agreement tolerance in r_s (the sweep)")
    ap.add_argument('--lattice_scale_min', type=float, default=0.05, help="above this lattice-scale weight a physics residual is named the lattice-scale difference between a chain and the cube")
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    opts = dict(chain_rule=args.chain_rule, cert_max=args.cert_max, clock_tol_rs=args.clock_tol_rs, lattice_scale_min=args.lattice_scale_min)
    if (not args.force) and os.path.exists(args.out + '.json') and os.path.exists(args.out + '.md'):
        print(f"[skip] {args.out}.json and .md exist (--force to redo)", flush=True)
        return 0
    results = []
    for ch, lf, cu in args.pair:
        if not all(os.path.exists(p) for p in (ch, lf, cu)):
            print(f"[missing] {ch} / {lf} / {cu}: pair skipped", flush=True)
            continue
        pair = load_pair(ch, lf, cu, opts)
        R = compare_pair(pair, opts)
        results.append(R)
        print(f"[pair {R['name']}] {R['summary']['sentence']} Worst residual/error-sum {R['summary']['worst_ratio_residual_over_error_sum']:.2f} ({R['summary']['worst']}); "
              f"half-weight clocks {R['clocks'].get('half_weight_1p5rs')}; fallback: {R['summary']['fallback_indicated']}", flush=True)
    if not results:
        print("[compare] nothing compared", flush=True)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + '.json', 'w') as fh:
        json.dump(dict(script='o5_compare_v1.py', options=opts, results=results), fh, indent=1, default=_json_default)
    write_md(results, args.out + '.md', opts)
    print(f"[done] {args.out}.json and .md ({len(results)} pairs)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
