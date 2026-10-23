#!/usr/bin/env python
"""qh_channels_reweight_v1: corrects the channel-weight defect of qh_branch_witness_v1b's --stage channels on saved outputs.

The defect (v1b run_channels): vals[nm] holds the per-copy sea variance ||U_u^dagger A U_o||_F^2 of channel kappa, while
everything downstream (S = cumsum(vals), c = vals/(16 kappa), tail_fit, q_H(K) = 4 sum_j (excess_j + S_j(K))/M_j^2,
the record-channel check against the witness's per_channel w * var_sea) treats vals as the WEIGHTED sea variance
w_kappa * var_sea, w_kappa = 16 kappa (the copy multiplicity; S10 rebuild_channels and the witness g_list). Hence the
record-channel check rel = |v - 16 kappa v|/(16 kappa v) = 1 - 1/(16 kappa) exactly.

Correction (pure factor): vals_corr[kappa] = 16 kappa * vals[kappa]. Everything else of the summary is recomputed from
vals_corr with v1b's own formulas (tail_fit, comparator, marks, q_H at marks, stopping rule). Reads
qh_member<m>_channels.npz / _channels.json / qh_member<m>.json under --out; writes qh_member<m>_channels_reweight_v1.json
and qh_member<m>_channels_reweight_v1.npz beside them. Never overwrites an original (refuses if the target exists unless
--force, and --force only ever touches the _reweight_v1 names).
"""
import argparse
import json
import math
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np

VERSION = 'qh_channels_reweight_v1'
TAG = '_reweight_v1'


def tail_fit(kappas, vals):
    """v1b's tail_fit verbatim (it is correct given weighted vals)."""
    k = np.asarray(kappas, float)
    val = np.asarray(vals, float)
    c = val / (16.0 * k)
    K = k[-1]
    S = float(val.sum())
    sel = (k >= 0.5 * K) & (c > 0)
    if sel.sum() < 20:
        return dict(p=float('nan'), tail=float('inf'), S=S, n_fit=int(sel.sum()), cK_over_c1=float(c[-1] / c[0]) if c[0] > 0 else float('nan'))
    slope = float(np.polyfit(np.log(k[sel]), np.log(c[sel]), 1)[0])
    p = -slope
    tail = 16.0 * float(c[-1]) * K * K / (p - 2.0) if p > 2.0 else float('inf')
    return dict(p=p, tail=tail, S=S, n_fit=int(sel.sum()), cK_over_c1=float(c[-1] / c[0]) if c[0] > 0 else float('nan'))


def lattice_1d_comparator(M):
    """v1b's comparator verbatim."""
    p = np.linspace(-math.pi, math.pi, 40001)
    s2 = np.sin(p) ** 2
    f = s2 * np.cos(p) ** 4 * M * M / (s2 + M * M) ** 3
    return float(np.trapz(f, p))


def stop_rule(kappa, K_rec, fit, stop_tol):
    return (kappa >= max(2 * K_rec + 20, 100) and fit['n_fit'] >= 50 and fit['p'] > 2.2
            and math.isfinite(fit['tail']) and fit['tail'] < stop_tol * fit['S'])


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def reweight_member(out_dir, mem, force=False):
    npz = os.path.join(out_dir, 'qh_member%d_channels.npz' % mem)
    cjs = os.path.join(out_dir, 'qh_member%d_channels.json' % mem)
    wjs = os.path.join(out_dir, 'qh_member%d.json' % mem)
    o_json = os.path.join(out_dir, 'qh_member%d_channels%s.json' % (mem, TAG))
    o_npz = os.path.join(out_dir, 'qh_member%d_channels%s.npz' % (mem, TAG))
    if not (os.path.exists(npz) and os.path.exists(cjs) and os.path.exists(wjs)):
        print('member %d: finished channel outputs missing (npz %s, json %s, witness %s); skipped'
              % (mem, os.path.exists(npz), os.path.exists(cjs), os.path.exists(wjs)))
        return None
    if (os.path.exists(o_json) or os.path.exists(o_npz)) and not force:
        print('member %d: %s exists; skipped (use --force to rewrite the _reweight_v1 files only)' % (mem, o_json))
        return None
    Z = np.load(npz, allow_pickle=False)
    C = json.load(open(cjs))
    W = json.load(open(wjs))
    names = list(C['series'])
    labs = list(W['radius_labels'])
    kappas = [int(k) for k in Z['kappa']]
    kap = np.asarray(kappas, float)
    if kappas != list(range(1, len(kappas) + 1)):
        raise RuntimeError('member %d: kappa list is not 1..K' % mem)
    K = kappas[-1]
    if K != int(C['K_final']):
        raise RuntimeError('member %d: npz K %d != json K_final %s' % (mem, K, C['K_final']))
    raw = {nm: np.asarray(Z['var_' + nm.replace(':', '__')], float) for nm in names}
    wk = 16.0 * kap                                     # the copy weight w_kappa = 16 kappa (S10 / witness g_list)
    w_wit = {int(k): float(w) for k, w in zip(W['kappas'], W['weights'])}
    for k, w in w_wit.items():
        if abs(w - 16.0 * k) > 1e-12:
            raise RuntimeError('member %d: witness weight %s at kappa %d is not 16 kappa' % (mem, w, k))
    vals = {nm: wk * raw[nm] for nm in names}
    K_rec = int(C['K_rec'])
    stop_tol = float(C['stop_tol'])
    # --- the record-channel check against the witness's per_channel (p[3] = w * var_sea)
    rec = []
    for lab in labs:
        for p in W['radii'][lab]['variants']['full']['per_channel']:
            kk = int(p[0])
            if kk <= K:
                ours = float(vals['%s:full' % lab][kk - 1])
                rec.append(dict(lab=lab, kappa=kk, ours=ours, stored=float(p[3]), rel=abs(ours - float(p[3])) / max(abs(float(p[3])), 1e-300),
                                v1b_rel=abs(float(raw['%s:full' % lab][kk - 1]) - float(p[3])) / max(abs(float(p[3])), 1e-300),
                                one_minus_1_over_16kappa=1.0 - 1.0 / (16.0 * kk)))
    # --- the stopping rule replayed on the corrected series at v1b's check points (kappa % 10 == 0, K_rec, K_max)
    primary = '%s:full' % labs[0]
    fired = None
    v1b_fired = None
    K_max = int(C['K_max'])
    for kk in range(1, K + 1):
        if kk % 10 == 0 or kk == K_max or kk == K_rec:
            f_c = tail_fit(kappas[:kk], vals[primary][:kk])
            f_r = tail_fit(kappas[:kk], raw[primary][:kk])
            if fired is None and stop_rule(kk, K_rec, f_c, stop_tol):
                fired = kk
            if v1b_fired is None and stop_rule(kk, K_rec, f_r, stop_tol):
                v1b_fired = kk
    fit_final = tail_fit(kappas, vals[primary])
    if fired is not None:
        status = 'corrected stopping rule fires at K %d (the saved series ends at K %d)' % (fired, K)
    elif K >= K_max:
        status = 'not converged at K_max %d under the corrected weights (slope p %.3f)' % (K_max, fit_final['p'])
    else:
        status = ('NOT converged under the corrected weights: the v1b run stopped at K %d on the mis-weighted rule; the corrected '
                  'rule does not fire up to K %d (slope p %.3f, tail %s, S %.4e); a resume to K_max %d is needed'
                  % (K, K, fit_final['p'], ('%.3e' % fit_final['tail']) if math.isfinite(fit_final['tail']) else 'inf', fit_final['S'], K_max))
    marks = {k: int(v) for k, v in C['marks'].items()}
    per_series = {}
    for nm in names:
        val = vals[nm]
        c = val / (16.0 * kap)
        S = np.cumsum(val)
        f = tail_fit(kappas, val)
        ps = dict(c_1=float(c[0]), c_K_over_c_1=float(c[-1] / c[0]) if c[0] > 0 else None, S_at={}, slope_p=f['p'], tail_estimate=f['tail'], S_K=f['S'],
                  S_plus_tail=f['S'] + f['tail'] if math.isfinite(f['tail']) else float('inf'))
        for mk, kk in marks.items():
            kk = min(max(int(kk), 1), K)
            ps['S_at'][mk] = dict(K=kk, S=float(S[kk - 1]), c_over_c1=float(c[kk - 1] / c[0]) if c[0] > 0 else None)
        per_series[nm] = ps
    R_use = {lab: float(W['radii'][lab]['r']) for lab in labs}
    m = float(W['m'])
    comp = {}
    for lab in labs:
        I1 = lattice_1d_comparator(math.sqrt(m * m + (1.0 / R_use[lab]) ** 2))
        sel = [kk for kk in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000) if kk <= K]
        c = vals['%s:full' % lab] / (16.0 * kap)
        comp[lab] = [dict(kappa=kk, measured_c_over_c1=float(c[kk - 1] / c[0]),
                          comparator_c_over_c1=lattice_1d_comparator(math.sqrt(m * m + (kk / R_use[lab]) ** 2)) / I1) for kk in sel]
    proj = {}
    for mk, kk in marks.items():
        kk = min(max(int(kk), 1), K)
        q = 0.0
        for lab in labs:
            exc = float(W['q_H']['per_radius'][lab]['excess_full_record'])
            MH = float(W['q_H']['per_radius'][lab]['M_H'])
            q += 4.0 * (exc + float(np.cumsum(vals['%s:full' % lab])[kk - 1])) / MH ** 2
        proj[mk] = dict(K=kk, q_H=q, q_H_v1b=float(C['q_H_at_marks'][mk]['q_H']))
    q_tail = 0.0
    for lab in labs:
        ft = tail_fit(kappas, vals['%s:full' % lab])
        q_tail += 4.0 * ft['tail'] / float(W['q_H']['per_radius'][lab]['M_H']) ** 2
    out = dict(version=VERSION, source=dict(npz=npz, json=cjs, witness=wjs, source_version=C['version']), member=mem, label=C['label'],
               correction='vals_corrected[kappa] = 16 kappa * vals_v1b[kappa] (v1b saved the per-copy sea variance; the copy weight 16 kappa was never applied)',
               K_rec=K_rec, K_max=K_max, K_final=K, stop_tol=stop_tol, status=status, status_v1b=C['status'],
               stop_rule_corrected_fires_at=fired, stop_rule_v1b_replayed_fires_at=v1b_fired, series=names, marks=marks,
               record_channel_check=rec, record_channel_check_worst_rel=max([r['rel'] for r in rec], default=0.0),
               ep_min_first_extra=C.get('ep_min_first_extra'), E_top=C.get('E_top'),
               witness_q_H_record=float(W['q_H']['q_H']), per_series=per_series,
               local_1d_comparator=dict(label=C['local_1d_comparator']['label'], values=comp),
               q_H_at_marks=proj, q_H_tail_estimate_beyond_K_final=q_tail,
               q_H_label=C['q_H_label'] + ' (corrected weights 16 kappa)')
    with open(o_json, 'w') as fh:
        json.dump(jsonable(out), fh, indent=1)
    np.savez_compressed(o_npz, kappa=np.asarray(kappas), weight=wk, ep_min=np.asarray(Z['ep_min']), seconds=np.asarray(Z['seconds']),
                        **{'var_' + nm.replace(':', '__'): vals[nm] for nm in names})
    print('member %d (%s): K_final %d; record check worst rel %.2e (v1b %.6f); q_H at K_rec %.6e (witness %.6e, v1b %.6e); '
          'q_H at marks %s; tail beyond K_final %.3e; %s'
          % (mem, C['label'], K, out['record_channel_check_worst_rel'], C['record_channel_check_worst_rel'], proj['K_rec']['q_H'],
             out['witness_q_H_record'], proj['K_rec']['q_H_v1b'], {k: '%.4e (K %d)' % (v['q_H'], v['K']) for k, v in proj.items()}, q_tail, status))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='<internal>/qh_v1b')
    ap.add_argument('--members', default='1,3,5,7,9,11,13,15,17')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    try:
        os.nice(10)
    except Exception:
        pass
    for mem in [int(x) for x in args.members.split(',') if x]:
        reweight_member(args.out, mem, args.force)


if __name__ == '__main__':
    sys.exit(main())
