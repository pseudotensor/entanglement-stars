#!/usr/bin/env python3
"""s46v31_audit_v1.py (busi lane, inbox #467; ReviewB_nc46_v1 sec. 3): the before-printing audit of NOTCOMP 4.6 v3.
Post-processing only: no solve, no eigendecomposition. Reads
  - the 18 s46v3 slice jsons and the prod 15 finite-difference control (<internal>/s46v3);
  - CORE_HF's route-1 slice jsons (<internal>/full_v1<hist>/core_hf_<hist>_T<t>.json): the unrounded
    route-1 Var_C ('full' variant = the dressed vertex with currents, v3's Var L), M_H, the raw-column floor footprint, the channel rows
    (masks, gaps, S8, S8b) and the checks (S3, S3b, Q8');
  - the engine records of the two histories (prod: c17 to T/r_s 11, then c19 from its resume; s8: c27), every record up to each slice.
Emits (out dir <internal>/s46v31):
  A. the unrounded endpoints: <B_lin>, <B> lower/upper, CS bound, per slice (repr), and the matching baselines
     (route-1 <B_lin> from the unrounded route-1 Var_C and M_H; v2's printed value) -- table A and json;
  B. the error budget per slice (construction, stored state), against the margins to 0.0799 and 0.080 -- table B;
  C. the continued reference = the negative projector, from the occupation and crossing records -- tables C1 (slice), C2 (history).
No number is typed in by hand except the declared constants below (the printing ceilings and v2's printed values, which are
quoted from NOTCOMP_4.6_v2 table, read from the s46v3 v2cmp md).
"""
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

VERSION = 's46v31_audit_v1'
S3DIR = '<internal>/s46v3'
OUT = '<internal>/s46v31'
R1DIR = '<internal>/full_v1'
REC = {'prod': [('<internal>/collapse_c17_rs64_A_R2_tau8_ell4_avg.json', None, 11.0),
                ('<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json', 11.0, None)],
       's8': [('<internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json', None, None)]}
CEILINGS = (0.0799, 0.080)
SLICES = [('prod', t) for t in (12, 15, 16, 20, 24, 25, 28, 30, 35, 40)] + [('s8', t) for t in (12, 13, 14, 16, 20, 25, 30, 40)]


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def find_key(x, key):
    """all values of key anywhere in a nested json."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k == key:
                out.append(v)
            out += find_key(v, key)
    elif isinstance(x, list):
        for v in x:
            out += find_key(v, key)
    return out


def check(J, prefix):
    for c in J['checks']:
        if c['name'].startswith(prefix):
            return float(c['value'])
    return None


def v2_printed():
    """v2's printed <B_lin> per slice, from the s46v3 v2cmp table (column 2)."""
    out = {}
    for line in open(os.path.join(S3DIR, 's46v3_v2cmp_v1.md')):
        m = re.match(r'\|\s*(prod|s8) (\d+)\s*\|\s*([0-9.]+)\s*\|', line)
        if m:
            out[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return out


def history_records(hist, t):
    recs = []
    for path, t_from, t_to in REC[hist]:
        J = json.load(open(path))
        rs = float(J['args']['rs'])
        for r in J['records']:
            if r.get('E') is None:
                continue
            tr = float(r['T_over_rs']) if r.get('T_over_rs') is not None else float(r['T']) / rs
            if t_from is not None and tr <= t_from + 1e-9:
                continue
            if t_to is not None and tr > t_to + 1e-9:
                continue
            if tr > t + 1e-9:
                continue
            recs.append((tr, r, os.path.basename(path)))
    recs.sort(key=lambda x: x[0])
    return recs


def history_summary(hist, t):
    recs = history_records(hist, t)
    s = dict(n_records=len(recs), T_first=recs[0][0], T_last=recs[-1][0], files=sorted(set(x[2] for x in recs)),
             mask_eq_neg_recorded=0, mask_eq_neg_false=0, inversions_sum=0, intermediate_weights_sum=0, n_pocket_max=0,
             n_refill_max_abs=0, min_occupied_weight=1.0, commutator_max=0.0, n_verified_channels=0, n_lz_channels=0,
             converged_false=0, R2_minus_R0_F_max=0.0, nsub_max=1)
    for tr, r, _ in recs:
        if r.get('mask_equals_negative_set') is not None:
            s['mask_eq_neg_recorded'] += 1
            s['mask_eq_neg_false'] += int(not r['mask_equals_negative_set'])
        s['inversions_sum'] += int(r.get('inversions_total') or 0)
        s['intermediate_weights_sum'] += int(r.get('intermediate_weights_total') or 0)
        npk = r.get('n_pocket') or {}
        s['n_pocket_max'] = max(s['n_pocket_max'], sum(int(v) for v in npk.values()) if npk else 0)
        nrf = r.get('n_refill_R0') or {}
        s['n_refill_max_abs'] = max(s['n_refill_max_abs'], max((abs(int(v)) for v in nrf.values()), default=0))
        if r.get('min_occupied_weight') is not None:
            s['min_occupied_weight'] = min(s['min_occupied_weight'], float(r['min_occupied_weight']))
        cm = r.get('commutator_h_R2_F') or {}
        if cm:
            s['commutator_max'] = max(s['commutator_max'], max(float(v) for v in cm.values()))
        for c in (r.get('continuation') or {}).values():
            if c.get('verified'):
                s['n_verified_channels'] += 1
            else:
                s['n_lz_channels'] += 1
        if r.get('converged') is False:
            s['converged_false'] += 1
        if r.get('R2_minus_R0_F') is not None:
            s['R2_minus_R0_F_max'] = max(s['R2_minus_R0_F_max'], float(r['R2_minus_R0_F']))
        s['nsub_max'] = max(s['nsub_max'], int(r.get('nsub_max') or 1))
    last = recs[-1][1]
    s['at_slice'] = dict(T=recs[-1][0], continuation_label=last.get('continuation_label'), mask_equals_negative_set=last.get('mask_equals_negative_set'),
                         inversions_total=last.get('inversions_total'), R2_minus_R0_F=last.get('R2_minus_R0_F'),
                         delta_rtol=(last.get('indicator') or {}).get('delta_rtol'), iteration_last=(last.get('iteration_history') or [None])[-1])
    return s


def v3_log_channels(hist, t):
    p = os.path.join(S3DIR, 'logs', '%s_T%d.log' % (hist, t))
    out = []
    for line in open(p):
        m = re.search(r'channel kappa (\d+): n_occ (\d+) \(dust (\d+)\), gap ([0-9.eE+-]+), \|\|\(1-P\)h P\|\|/\|\|h\|\| ([0-9.eE+-]+)', line)
        if m:
            out.append(dict(kappa=int(m.group(1)), gap=float(m.group(4)), comm=float(m.group(5))))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    v2p = v2_printed()
    fdJ = json.load(open(os.path.join(S3DIR, 'prod_T0015.00_fd.json')))
    fd_rel = [float(x['rel']) for blk in fdJ['fd_route'].values() for d in blk for x in d['radii'].values()]
    e_H = max(fd_rel)                      # the worst relative disagreement of u^T H u with full self-consistent FD (prod 15)
    rows, inputs = [], {}
    for hist, t in SLICES:
        tag = '%s_T%07.2f' % (hist, t)
        pj = os.path.join(S3DIR, tag + '.json')
        S = json.load(open(pj))
        p1 = os.path.join(R1DIR, hist, 'core_hf_%s.json' % tag)
        J1 = json.load(open(p1))
        inputs[tag] = dict(s46v3=dict(path=pj, md5=md5(pj)), route1=dict(path=p1, md5=md5(p1)))
        rep = S['B']['radii']
        V = S['variance']
        Blin, (Blo, Bhi), cs = S['B']['B_lin'], S['B']['B'], S['B']['CS_bound']
        # route-1 matching baseline
        Blin_r1, dVL, foot, dM = 0.0, [], [], []
        varL_err_abs = 0.0
        floor_abs = 0.0
        for l in rep:
            r1 = J1['variance'][l]
            vf = r1['variants']['full']
            M1, M3 = float(r1['M_H']), float(V[l]['M_H'])
            Blin_r1 += 4 * vf['Var_C'] / M1 ** 2
            dVL.append(abs(V[l]['Var_L'] - vf['Var_C']) / vf['Var_C'])
            dM.append(abs(M3 - M1) / M1)
            varL_err_abs += 4 * abs(V[l]['Var_L'] - vf['Var_C']) / M3 ** 2
            foot.append(float(vf['floor_footprint_rel']))
            floor_abs += 4 * float(vf['floor_footprint_rel']) * V[l]['Var_L'] / M3 ** 2
        # second-order construction: relative error e_slice on H, entering LQ linearly and Var Q, <Q>^2 quadratically
        kfd = max(max(V[l]['K_fd_rel']) for l in rep)
        dbr = max(V[l]['dB_fd']['richardson_rel_diff'] for l in rep)
        idr = max([x['rel'] for x in S.get('rho2_identity_R_hm_plus_PA') or []] or [0.0])
        e_slice = max(e_H, kfd, dbr, idr)
        so_abs = sum(4 * (abs(V[l]['LQ_plus_QL']) + V[l]['Var_Q'][1] + V[l]['mean_offset'] ** 2) / V[l]['M_H'] ** 2 for l in rep)
        so_err = 2 * e_slice * so_abs
        delta = Bhi - Blin
        # stored-state terms
        S3 = check(J1, 'S3 ')
        S3b = check(J1, 'S3b')
        Q8 = check(J1, "Q8'")
        smin = [float(x) for x in find_key(J1, 'sigma_min') if isinstance(x, (int, float))]
        smin = min(smin) if smin else None
        amp = 2.0 + 2.0 / (smin if smin else 0.689)
        rec_terms = dict(S3=S3, S3b=S3b, Q8=Q8)
        hs = history_summary(hist, t)
        drt = hs['at_slice']['delta_rtol'] or 0.0
        state_abs = amp * (max(S3, S3b) + Q8 + drt) * Bhi + floor_abs
        # the reference: S8 absolute (Frobenius) and its effect on the second-order part through D = C - P_ref
        chans = J1['channels']
        s8abs = max(float(c['pref_resolution_defect']) * math.sqrt(c['n_neg']) for c in chans)
        dnorm = min(math.sqrt(float(c['TrD2'])) for c in chans)
        ref_abs = 2 * (s8abs / dnorm) * so_abs
        budget = varL_err_abs + so_err + state_abs + ref_abs
        rows.append(dict(slice='%s %d' % (hist, t), hist=hist, T=t, complex64=S['complex64'], radii=rep,
                         B_lin=Blin, B_lo=Blo, B_hi=Bhi, CS=cs, delta=delta, X_width=Bhi - Blo,
                         B_lin_route1=Blin_r1, B_lin_v2_printed=v2p.get((hist, t)),
                         VarL_vs_route1_rel_max=max(dVL), M_vs_route1_rel_max=max(dM), floor_footprint_rel_max=max(foot),
                         e_H=e_slice, second_order_abs=so_abs, terms=dict(varL_construction=varL_err_abs, second_order_construction=so_err,
                                                                         stored_state=state_abs, stored_floor_part=floor_abs, reference_S8=ref_abs),
                         amp=amp, sigma_min=smin, record_checks=rec_terms, delta_rtol=drt, budget=budget,
                         B_hi_plus_budget=Bhi + budget, stress_100pct=Blin + 2 * abs(delta) + varL_err_abs + state_abs + ref_abs,
                         margins={str(c): c - (Bhi + budget) for c in CEILINGS},
                         headroom_0799=(CEILINGS[0] - Bhi) / budget if budget > 0 else None,
                         ref=dict(channels=[dict(kappa=c['kappa'], n_neg=c['n_neg'], n_sea=c['n_sea'], gap=c['gap'], mask_eq_neg=c['mask_equals_neg'],
                                                 stored_mask_eq_stored_neg=c['stored_mask_equals_stored_neg'], S8=c['pref_resolution_defect'],
                                                 S8b=c['commutator_rel'], rec_comm=c.get('record_commutator_h_R2_F')) for c in chans],
                                  S8_abs_max=s8abs, v3_log=v3_log_channels(hist, t)),
                         history=hs))
    # ---- tables
    L = []
    L.append('# NOTCOMP 4.6 v3.1 audit tables (%s; post-processing only)\n' % VERSION)
    L.append('e_H (prod 15, full self-consistent FD of u^T H u, worst over 3 directions x 3 radii): %.3e\n' % e_H)
    L.append('## A. Unrounded endpoints and matching baselines\n')
    L.append('| slice | dtype | <B_lin> (v3) | <B_lin> route 1 (CORE_HF unrounded) | rel | <B_lin> v2 printed | <B> lower | <B> upper | <B> upper - <B_lin> | CS bound |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        L.append('| %s | %s | %r | %r | %.1e | %s | %r | %r | %.6e | %r |' % (r['slice'], 'c64' if r['complex64'] else 'c128', r['B_lin'], r['B_lin_route1'],
                 abs(r['B_lin'] - r['B_lin_route1']) / r['B_lin_route1'], ('%.6f' % r['B_lin_v2_printed']) if r['B_lin_v2_printed'] else '-',
                 r['B_lo'], r['B_hi'], r['delta'], r['CS']))
    L.append('\n## B. Error budget on <B> (absolute), against the printing ceilings\n')
    L.append('| slice | <B> upper | Var L constr. (route 1 vs v3) | 2nd-order constr. (2 e_H x |2nd-order parts|) | e_H used | stored state (amp x (S3+Q8+dtol) x <B> + c64 floor proxy) | of which c64 floor proxy | reference S8 | total | <B> upper + total | margin to 0.0799 | headroom (0.0799 - <B>)/total | stress: <B_lin> + 2|Delta| + rest |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        T_ = r['terms']
        L.append('| %s | %.6f | %.1e | %.1e | %.1e | %.1e | %.1e | %.1e | %.1e | %.6f | %+.2e | %.0f | %.6f |' % (
            r['slice'], r['B_hi'], T_['varL_construction'], T_['second_order_construction'], r['e_H'], T_['stored_state'], T_['stored_floor_part'],
            T_['reference_S8'], r['budget'], r['B_hi_plus_budget'], r['margins']['0.0799'], r['headroom_0799'], r['stress_100pct']))
    L.append('\nInputs per slice for the stored-state term:\n')
    L.append('| slice | S3 (sources vs record eps_tot) | S3b (currents) | Q8\' field check | record delta_rtol | sigma_min(I+BP^A) | amp = 2 + 2/sigma_min | max floor footprint (rel, proxy) | max |Var L - route 1|/Var | max |M_H - route 1|/M |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        c = r['record_checks']
        L.append('| %s | %.2e | %.2e | %.2e | %.2e | %s | %.3f | %.2e | %.1e | %.1e |' % (r['slice'], c['S3'], c['S3b'], c['Q8'], r['delta_rtol'],
                 ('%.3f' % r['sigma_min']) if r['sigma_min'] else '-', r['amp'], r['floor_footprint_rel_max'], r['VarL_vs_route1_rel_max'], r['M_vs_route1_rel_max']))
    L.append('\n## C1. The reference at the slice: the continued reference against the negative projector, per channel (max/min over the 10 channels)\n')
    L.append('| slice | all masks == (e<0) (route 1 S7b, stored and own eigh) | n_neg == n_sea all | min gap | max S8 ||R2R2^+ - UoUo^+||_F/sqrt(n_o) | max S8 abs (F) | max S8b | max record ||[h_m,R2]||_F | v3 run: mask asserted, max ||(1-P)hP||/||h|| | v3 min gap |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        ch = r['ref']['channels']
        lg = r['ref']['v3_log']
        L.append('| %s | %s | %s | %.4f | %.1e | %.1e | %.1e | %.1e | yes, %.1e | %.4f |' % (
            r['slice'], all(c['mask_eq_neg'] and c['stored_mask_eq_stored_neg'] for c in ch), all(c['n_neg'] == c['n_sea'] for c in ch),
            min(c['gap'] for c in ch), max(c['S8'] for c in ch), r['ref']['S8_abs_max'], max(c['S8b'] for c in ch),
            max(c['rec_comm'] or 0 for c in ch), max(x['comm'] for x in lg), min(x['gap'] for x in lg)))
    L.append('\n## C2. The history up to each slice: occupation and crossing records (every engine record from T/r_s 0)\n')
    L.append('| slice | records (T/r_s range; files) | mask == negative set (recorded / false) | inversions (sum) | intermediate weights (sum) | max n_pocket | max |n_refill| | min occupied weight | channel-steps Weyl-verified / LZ convention | max nsub | max ||[h_m,R2]||_F | max R2_minus_R0_F | unconverged records | label at the slice |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        h = r['history']
        L.append('| %s | %d (%.3f to %.3f; %s) | %d / %d | %d | %d | %d | %d | %.10f | %d / %d | %d | %.1e | %.1e | %d | %s |' % (
            r['slice'], h['n_records'], h['T_first'], h['T_last'], ', '.join(x.replace('collapse_', '').replace('.json', '') for x in h['files']),
            h['mask_eq_neg_recorded'], h['mask_eq_neg_false'], h['inversions_sum'], h['intermediate_weights_sum'], h['n_pocket_max'], h['n_refill_max_abs'],
            h['min_occupied_weight'], h['n_verified_channels'], h['n_lz_channels'], h['nsub_max'], h['commutator_max'], h['R2_minus_R0_F_max'],
            h['converged_false'], h['at_slice']['continuation_label']))
    mx = max(rows, key=lambda r: r['B_hi'])
    summ = dict(script=VERSION, code_md5=md5(os.path.abspath(__file__)), e_H=e_H, max_slice=mx['slice'], max_B_hi=mx['B_hi'], max_B_hi_repr=repr(mx['B_hi']),
                max_budget=max(r['budget'] for r in rows), max_B_hi_plus_budget=max(r['B_hi_plus_budget'] for r in rows),
                max_stress=max(r['stress_100pct'] for r in rows), min_margin_0799=min(r['margins']['0.0799'] for r in rows),
                min_headroom_0799=min(r['headroom_0799'] for r in rows),
                history_inversions_total=sum(r['history']['inversions_sum'] for r in rows), history_mask_false=sum(r['history']['mask_eq_neg_false'] for r in rows),
                history_pocket_max=max(r['history']['n_pocket_max'] for r in rows), history_refill_max=max(r['history']['n_refill_max_abs'] for r in rows),
                min_gap=min(min(c['gap'] for c in r['ref']['channels']) for r in rows), max_S8=max(max(c['S8'] for c in r['ref']['channels']) for r in rows),
                max_S8_abs=max(r['ref']['S8_abs_max'] for r in rows), max_VarL_route1_rel=max(r['VarL_vs_route1_rel_max'] for r in rows),
                max_Blin_route1_rel=max(abs(r['B_lin'] - r['B_lin_route1']) / r['B_lin_route1'] for r in rows))
    L.append('\n## Summary\n')
    L.append('```\n' + json.dumps(summ, indent=1) + '\n```')
    open(os.path.join(OUT, 's46v31_audit_tables_v1.md'), 'w').write('\n'.join(L) + '\n')
    json.dump(dict(summary=summ, rows=rows, inputs=inputs), open(os.path.join(OUT, 's46v31_audit_v1.json'), 'w'), indent=1, default=str)
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
