#!/usr/bin/env python
"""o5_compare_v2.py -- O5 (GAP10_T7_final_v1.1 section 9): the comparison of the 2 propagations of the same lifted column and the acceptance rule.

Changes against o5_compare_v1.py (from the self-test's 2 failing comparator checks; v1 is never edited):
  * the cell-rule qualification is directional: a residual beyond the sum of the stated errors is 'physics (within the chain cell-rule
    ambiguity)' only when the cube's value lies within the error sum of the interval between the chain's 2 shell rules (the cell rule and the
    strict site rule); v1 accepted |residual| <= error sum + |cell - strict| on either side, which also qualified a residual on the far side of
    the ambiguity. The distance of the cube's value to that interval is stored per reading. (The failing check itself had a wrong premise: its
    residuals of 3 error sums can never lie within error sum + 2e-3 when the error sum is at least the Gram defect 1e-3; o5_selftest_v2 builds
    the readings from explicit residuals on both sides of the interval.)
  * the fallback rule compares the lift's Gram defect with the largest certified certificate band (over the shells and the certified sweep
    ends), not the smallest: v1's smallest band was the first sweep end's (certificate about 1e-5, and delta^2 where the share is still 0), so
    any lift with a nonzero Gram defect 'exceeded the bands' and the rule could never say no. v2 says yes only when the Gram defect exceeds the
    cube's certificate band at every certified reading; the smallest and largest certified bands and the fraction of certified readings whose
    band lies below the Gram defect are stored and printed, so the readings where the lift error dominates the error sum are visible.
  * the chain operator's provenance is read from the chain json (o5_chain_export_v2 records the solve code it imported; a v1 json means the
    reconstructed operator) and printed with the verdicts: a pair run on the reconstructed operator is not a result unless o5_operator_check_v1
    showed the operators equal to rounding (its json given with --operator_check); a lift json without the chain's declared profile, for a chain
    json that carries one, is flagged (the lift must be o5_lift_v2's).
Everything else is v1's. For every pair (chain export, lift, cube run) and every sweep end 1 .. S: the weight beyond r_h and beyond 1.5, 2 and
3 r_s, the chain's (the cell rule by default; --chain_rule strict) against the cube's (shares), each with its band 2 sqrt(p) delta + delta^2
(the cube's certificate delta at the sweep end; the chain's stated propagation error), plus the lift's Gram defect: the acceptance rule is
agreement within the sum of the stated errors; a residual beyond it is the difference between the 2 descriptions (the angular channel against
the genuine cubic lattice), labelled physics and not error; a residual beyond the sum on a column whose lattice-scale weight exceeds
--lattice_scale_min is named the lattice-scale difference between a chain and the cube. The fixed-cut number current through r_h at the
sweep ends is compared with the bands 2 ||h|| (2 delta + delta^2) and the lift term 2 ||h|| delta_G; its time integral is the beyond-r_h
weight's change on both descriptions (exact on the chain, the trapezoid over the sweep ends on the cube). The detector amplitudes (the lifted
detector packets, the same object in both descriptions) are compared in modulus at every readout time within delta_cube + delta_chain + the
amplitude lift error, their peak moduli and peak clocks within --clock_tol_rs r_s; the half-weight crossing clocks of p beyond 1.5 r_s
(HORIZON's rule) within --clock_tol_rs r_s. The axis packets are printed without a verdict. The Killing splits of both descriptions are
printed (b1 and b3), not compared. Writes <out>.json and <out>.md; no plots. No digit here is the object's.
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
RECONSTRUCTED = "reconstructed from HORIZON v2's written definition (o5_chain_export_v1.RadialChain)"


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
    op = (J.get('chain') or {}).get('operator') or {}
    provenance = dict(operator=dict(source=op.get('source', RECONSTRUCTED), solve_code=bool(op.get('solve_code', False)), page_ops_v6_md5=op.get('page_ops_v6_md5'),
                                    engine_md5=op.get('engine_md5'), path=op.get('path')),
                      chain_script=J.get('script'), lift_script=Lf.get('script'), profile=J['column'].get('profile'),
                      lift_has_profile=bool((Lf.get('source') or {}).get('profile')),
                      reflection_flagged=int(((J.get('readout') or {}).get('reflection') or {}).get('n_readout_times_flagged') or 0))
    return dict(name=f"Rc{J['column']['Rc']:g}_k{J['chain']['kappa']}_H{K['H']}", chain_json=chain_json, lift_json=lift_json, cube_json=cube_json, rs=rs,
                sweep_t=sweep_t_ch[:S + 1], radii=radii, times=times,
                column=dict(Rc=J['column']['Rc'], sigma=J['column']['sigma'], ak0=J['column']['ak0'], kappa=J['chain']['kappa'], E_mean=J['column']['E_mean'],
                            E_rms=J['column']['E_rms'], spinor_mode=J['column']['spinor_mode'], profile=J['column'].get('profile'), source=J['column'].get('source')),
                chain=dict(p=chain_p, p_alt=chain_alt, delta=chain_delta, J=Jch, Jint_exact=np.asarray(Zc['Jint_exact_sweeps'], dtype=float)[:S + 1],
                           dp_strict=(p_strict[sweep_idx_ch, 0] - p_strict[0, 0]), wmax=float(J['chain']['eigen']['abs_max']), amp_delta=prop_err[:nt],
                           lattice_scale=float(J['lattice_scale_weight']), killing=J['killing'], r_h=float(J['record']['r_h']),
                           identity_defect_exact_max=float(J['checks']['identity_defect_exact_max']), prop_err_last=float(prop_err[sweep_idx_ch[-1]])),
                cube=dict(p=cube_p, delta=cube_delta, J=Jcu, bound=float(K['bound']), amp_delta=cert[:nt, 0], lattice_scale=float((src0.get('band') or {}).get('lattice_scale', float('nan'))),
                          killing=killing_cube, r_h=cut_cube, r_h_direction=float(K['r_h']), H=int(K['H']), napply=K.get('napply'), wall=K.get('wall_total'),
                          energy_mean=src0.get('energy_mean'), energy_std=src0.get('energy_std'), norm_drift=float(np.max(np.abs(np.asarray(Zk['norms'])[:, 0] / np.asarray(Zk['norms'])[0, 0] - 1.0)))),
                lift=dict(gram_defect=float(Lf['source']['gram_defect']), amp_lift_error=float(Lf['amp_lift_error']), W_lift=float(Lf['source']['W_lift']),
                          reduction=Lf.get('reduction'), det_gram=[m_['gram_defect'] for m_ in Lf['detector']['members']]),
                amps=amps, axis=axis, provenance=provenance)


# ------------------------------------------------------------------ the comparison
def compare_pair(pair, opts):
    rs = pair['rs']
    sweep_t = np.asarray(pair['sweep_t'], dtype=float)
    S = sweep_t.size - 1
    dG = pair['lift']['gram_defect']
    prov = pair.get('provenance') or dict(operator=dict(source=RECONSTRUCTED, solve_code=False))
    op = prov.get('operator') or {}
    oc = opts.get('operator_check')
    if op.get('solve_code'):
        eligible = True
        op_note = f"the chain ran on the solve code's operator ({op.get('source')}; page_ops_v6 md5 {op.get('page_ops_v6_md5')})"
    elif oc is not None and oc.get('equal_to_rounding'):
        eligible = True
        op_note = (f"the chain ran on the reconstructed operator, shown equal to the solve operator to rounding by {oc.get('file')} "
                   f"(max entry difference {float(oc.get('max_entry_diff', float('nan'))):.1e})")
    else:
        eligible = False
        op_note = ("the chain ran on the reconstructed operator (o5_chain_export_v1): not a result (give --operator_check with o5_operator_check_v1's json showing "
                   "equality to rounding, or rerun the chain side with o5_chain_export_v2)")
    lift_mismatch = bool(prov.get('profile')) and not bool(prov.get('lift_has_profile'))
    res = dict(name=pair['name'], column=pair['column'], files=dict(chain=pair.get('chain_json'), lift=pair.get('lift_json'), cube=pair.get('cube_json')),
               cut=dict(chain_r_h=pair['chain']['r_h'], cube_cut=pair['cube']['r_h'], cube_direction_r_h=pair['cube'].get('r_h_direction'),
                        difference=pair['cube']['r_h'] - pair['chain']['r_h']),
               errors=dict(gram_defect=dG, amp_lift_error=pair['lift']['amp_lift_error'], chain_prop_err_last=pair['chain'].get('prop_err_last'),
                           cube_cert_last=float(pair['cube']['delta'][-1]), cert_max=opts['cert_max']),
               provenance=prov, shells={}, current=[], detectors=[], axis=[], clocks={}, lattice_scale={},
               killing=dict(chain=pair['chain'].get('killing'), cube=pair['cube'].get('killing')), summary={})
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
            p_alt = float(pair['chain']['p_alt'][nm][s])
            lo, hi = min(p1, p_alt), max(p1, p_alt)
            dist = max(lo - p2, p2 - hi, 0.0)
            amb = abs(p_alt - p1)
            certified = d2 <= opts['cert_max']
            if abs(resid) <= tot:
                verdict = 'agree'
            elif dist <= tot:
                verdict = 'physics (within the chain cell-rule ambiguity)'
            else:
                verdict = 'physics'
            ratio = abs(resid) / tot if tot > 0 else float('inf')
            rows.append(dict(s=s, t=float(sweep_t[s]), t_rs=float(sweep_t[s] / rs), chain=p1, chain_alt_rule=p_alt, chain_band=b1, cube=p2, cube_band=b2, cube_cert=d2, lift=dG,
                             error_sum=tot, residual=resid, ratio=ratio, cell_rule_ambiguity=amb, distance_to_chain_rule_interval=dist, certified=certified, verdict=verdict))
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
    # the fallback rule: the Gram defect against the certified certificate bands (the largest decides; the smallest and the fraction below are stored)
    cert_rows = [(weight_band(float(pair['cube']['p'][nm][s]), float(pair['cube']['delta'][s])), nm, s)
                 for nm in pair['radii'] for s in range(1, S + 1) if float(pair['cube']['delta'][s]) <= opts['cert_max']]
    if cert_rows:
        bmin, bmax = min(cert_rows), max(cert_rows)
        frac_below = float(sum(1 for b in cert_rows if b[0] < dG)) / len(cert_rows)
        fallback = bool(dG > bmax[0])
        cb = dict(min=bmin[0], min_at=dict(shell=bmin[1], s=bmin[2], t_rs=float(sweep_t[bmin[2]] / rs)), max=bmax[0], max_at=dict(shell=bmax[1], s=bmax[2], t_rs=float(sweep_t[bmax[2]] / rs)),
                  fraction_below_gram_defect=frac_below, n_certified=len(cert_rows))
        if fallback:
            fb_text = (f"yes: the lift's Gram defect {dG:.2e} exceeds the cube's certificate band at every certified sweep-end reading (largest {bmax[0]:.2e} at {bmax[1]}, "
                       f"t/r_s {sweep_t[bmax[2]] / rs:.1f}); section 9's fallback applies (the spherical shell of radius 20 a and width 6 a: the chain with --column_source lane "
                       f"--spinor rest --ak0 0 --sigma 6, the lift and the cube runs with the _shell files)")
        else:
            fb_text = (f"no: the lift's Gram defect {dG:.2e} lies below the largest certified certificate band {bmax[0]:.2e} ({bmax[1]}, t/r_s {sweep_t[bmax[2]] / rs:.1f}); "
                       f"it exceeds the band at a fraction {frac_below:.2f} of the {len(cert_rows)} certified readings (smallest band {bmin[0]:.2e} at {bmin[1]}, "
                       f"t/r_s {sweep_t[bmin[2]] / rs:.1f}), where the lift error dominates the error sum")
    else:
        fallback, cb = None, dict(min=None, max=None, fraction_below_gram_defect=float('nan'), n_certified=0)
        fb_text = "undecided: no certified sweep-end reading (the cube's certificate exceeded cert_max at every sweep end)"
    sentence = (f"{n_agree} of {n_cert} certified sweep-end shell readings agree within the sum of the stated errors; {n_phys} exceed it and are the difference between "
                f"the 2 descriptions, labelled physics and not error" + (f" ({label})" if label else "") + f"; {op_note}"
                + ("; the lift json lacks the chain's declared profile (the lift must be o5_lift_v2's for this chain json): not a result as it stands" if lift_mismatch else "")
                + "; no digit of this comparison is the object's.")
    res['summary'] = dict(certified_sweep_end_readings=n_cert, agree=n_agree, physics=n_phys, worst_ratio_residual_over_error_sum=worst_ratio,
                          worst=(dict(shell=worst[0], s=worst[1], residual=worst[2], error_sum=worst[3]) if worst else None),
                          gram_defect_exceeds_every_certificate_band=fallback, certificate_band_certified=cb, fallback_indicated=fb_text,
                          reduction_convention_ok=((pair['lift'].get('reduction') or {}).get('default_is_closest')),
                          chain_operator=op_note, result_eligible=eligible, lift_profile_mismatch=lift_mismatch, sentence=sentence)
    return res


# ------------------------------------------------------------------ writing
def _f(x, fmt='%.5f'):
    try:
        return fmt % float(x) if x is not None and np.isfinite(float(x)) else '-'
    except (TypeError, ValueError):
        return '-'


def _profile_text(prof):
    if not prof:
        return "v1's (the lane's Gaussian of density width sigma with the cut)"
    cut = prof.get('cut')
    return (f"{prof.get('form', 'gaussian')}: envelope exp(-(r - R_c)^2/({float(prof.get('envelope_denominator', 4.0)):g} sigma^2)), sigma {float(prof.get('sigma', float('nan'))):g} a "
            f"({prof.get('sigma_meaning', '-')}), carrier exp(i k_0 (r - {float(prof.get('carrier_reference', 0.0)):g})), cut {('none' if cut is None else '%g sigma' % float(cut))}; "
            f"source {prof.get('source', '-')}")


def write_md(results, path, opts):
    L = ["# O5 witness exchange: the 2 propagations of the same lifted column (GAP10_T7_final_v1.1 section 9), o5_compare_v2.py", "",
         f"Acceptance rule: agreement at every sweep end within the sum of the stated errors (the cube's certificate band 2 sqrt(p) delta + delta^2, the chain's "
         f"propagation-error band of the same form, the lift's Gram defect); a residual beyond that sum is the difference between the 2 descriptions, labelled physics "
         f"and not error; it is qualified 'within the chain cell-rule ambiguity' only when the cube's value lies within the error sum of the interval between the chain's "
         f"2 shell rules. Chain shells by the {opts['chain_rule']} rule; readings with the cube's certificate above {opts['cert_max']:g} are printed as uncertified and "
         f"not counted. The fallback is indicated only when the Gram defect exceeds the cube's certificate band at every certified reading. Clocks within "
         f"{opts['clock_tol_rs']:g} r_s (the sweep). Every number is a reading on 1 held record at 1 averaging width; no digit is the object's.", ""]
    L += ["| pair | column | <h> chain | lift Gram defect | cube cert at last end | chain prop. error at last end | certified readings: agree / physics | worst residual / error sum | half-weight clock chain, cube (r_s) | detector peak clocks chain, cube (r_s) | lattice-scale weight chain; cube | fallback | chain operator | result eligible |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for R in results:
        c = R['clocks'].get('half_weight_1p5rs', {})
        dets = R['detectors']
        opshort = 'solve code' if R['provenance']['operator'].get('solve_code') else 'reconstructed'
        L.append(f"| {R['name']} | kappa {R['column']['kappa']}, R_c {R['column']['Rc']:g}, sigma {R['column']['sigma']:g}, a k_0 {R['column']['ak0']:+.4f}, {R['column']['spinor_mode']} | "
                 f"{R['column']['E_mean']:.5f} | {R['errors']['gram_defect']:.2e} | {R['errors']['cube_cert_last']:.2e} | {_f(R['errors']['chain_prop_err_last'], '%.1e')} | "
                 f"{R['summary']['agree']} / {R['summary']['physics']} of {R['summary']['certified_sweep_end_readings']} | {R['summary']['worst_ratio_residual_over_error_sum']:.2f} | "
                 f"{_f(c.get('chain_rs'), '%.2f')}, {_f(c.get('cube_rs'), '%.2f')} | " + '; '.join(f"{_f(d.get('chain_peak_t_rs'), '%.2f')}, {_f(d.get('cube_peak_t_rs'), '%.2f')}" for d in dets)
                 + f" | {R['lattice_scale']['chain_windowed_share_above_pi_over_4']:.3f}; {_f(R['lattice_scale']['cube_band_weights_lattice_scale'], '%.3f')} | {R['summary']['fallback_indicated'].split(':')[0]} | "
                 f"{opshort} | {R['summary']['result_eligible']} |")
    L.append("")
    for R in results:
        cb = R['summary'].get('certificate_band_certified') or {}
        L += [f"## {R['name']}: kappa {R['column']['kappa']}, R_c {R['column']['Rc']:g} a, sigma {R['column']['sigma']:g} a, a k_0 {R['column']['ak0']:+.5f}, spinor {R['column']['spinor_mode']}; "
              f"chain <h> {R['column']['E_mean']:.6f} (rms {R['column']['E_rms']:.5f})", "",
              f"- Files: chain {R['files']['chain']}; lift {R['files']['lift']}; cube {R['files']['cube']}.",
              f"- Chain operator: {R['summary']['chain_operator']}. Result eligible: {R['summary']['result_eligible']}."
              + (" The lift json lacks the chain's declared profile: the lift must be o5_lift_v2's for this chain json." if R['summary'].get('lift_profile_mismatch') else ""),
              f"- Column profile: {_profile_text(R['column'].get('profile'))}.",
              f"- The cut: chain r_h {R['cut']['chain_r_h']:.6f} a; the cube's shares and current read the cut {R['cut']['cube_cut']:.6f} a (difference {R['cut']['difference']:+.1e} a; the direction's own crossing {_f(R['cut']['cube_direction_r_h'], '%.6f')} a).",
              f"- Errors: lift Gram defect {R['errors']['gram_defect']:.3e} (shell weights), amplitude lift error {R['errors']['amp_lift_error']:.3e} (the maximal shell Gram deviation on the supports); "
              f"chain propagation error at the last sweep end {_f(R['errors']['chain_prop_err_last'], '%.2e')}; the cube's certificate at the last sweep end {R['errors']['cube_cert_last']:.2e}.",
              f"- The lift's reduction check chose the written convention: {R['summary']['reduction_convention_ok']}.",
              f"- Certified certificate bands: smallest {_f(cb.get('min'), '%.2e')}, largest {_f(cb.get('max'), '%.2e')}; fraction of the {cb.get('n_certified', 0)} certified readings with the band below the Gram defect "
              f"{_f(cb.get('fraction_below_gram_defect'), '%.2f')}.",
              f"- {R['summary']['sentence']}", f"- Fallback: {R['summary']['fallback_indicated']}.", ""]
        for nm, rows in R['shells'].items():
            L += [f"### weight beyond {nm}", "", "| t/r_s | chain p (band) | chain alternative rule | cube p (certificate band) | lift Gram defect | sum of errors | residual cube - chain | residual/sum | cell-rule ambiguity | distance to the chain-rule interval | certified | verdict |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r_ in rows:
                L.append(f"| {r_['t_rs']:.1f} | {r_['chain']:.6f} ({r_['chain_band']:.1e}) | {r_['chain_alt_rule']:.6f} | {r_['cube']:.6f} ({r_['cube_band']:.1e}) | {r_['lift']:.1e} | {r_['error_sum']:.2e} | {r_['residual']:+.2e} | "
                         f"{r_['ratio']:.2f} | {r_['cell_rule_ambiguity']:.1e} | {r_['distance_to_chain_rule_interval']:.2e} | {'yes' if r_['certified'] else 'no'} | {r_['verdict']} |")
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


def load_operator_check(path):
    """o5_operator_check_v1's json: its summary (max entry difference over the kappas, equal_to_rounding) for the provenance of v1 chain exports."""
    with open(path) as fh:
        C = json.load(fh)
    sm = C.get('summary') or {}
    return dict(file=path, max_entry_diff=float(sm.get('max_entry_diff', float('nan'))), equal_to_rounding=bool(sm.get('equal_to_rounding', False)), rounding_tol=sm.get('rounding_tol'))


def main(argv=None):
    ap = argparse.ArgumentParser(description="O5: compare the chain's and the cube's propagations of the same lifted column; the acceptance rule of section 9 (v2 rules)")
    ap.add_argument('--pair', nargs=3, action='append', metavar=('CHAIN_JSON', 'LIFT_JSON', 'CUBE_JSON'), required=True, help="repeatable")
    ap.add_argument('--out', default='o5_compare_v2', help="writes <out>.json and <out>.md")
    ap.add_argument('--chain_rule', choices=['midpoint', 'strict'], default='midpoint', help="the chain's shell rule compared with the cube's site sum")
    ap.add_argument('--cert_max', type=float, default=1e-3, help="a sweep end counts as certified when the cube's certificate is below this")
    ap.add_argument('--clock_tol_rs', type=float, default=1.0, help="clock agreement tolerance in r_s (the sweep)")
    ap.add_argument('--lattice_scale_min', type=float, default=0.05, help="above this lattice-scale weight a physics residual is named the lattice-scale difference between a chain and the cube")
    ap.add_argument('--operator_check', default='', help="o5_operator_check_v1's json; with equality to rounding a v1 chain export (the reconstructed operator) is result eligible")
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    opts = dict(chain_rule=args.chain_rule, cert_max=args.cert_max, clock_tol_rs=args.clock_tol_rs, lattice_scale_min=args.lattice_scale_min,
                operator_check=(load_operator_check(args.operator_check) if args.operator_check else None))
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
        json.dump(dict(script='o5_compare_v2.py', options=opts, results=results), fh, indent=1, default=_json_default)
    write_md(results, args.out + '.md', opts)
    print(f"[done] {args.out}.json and .md ({len(results)} pairs)", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
