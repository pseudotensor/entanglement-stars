"""Independent residual summaries, historical masks and plot-array checks."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import mpmath as mp
from compute_LADDERS_v1p3_tables import member_readout
from compute_LADDERS_v1p3_controls import pair
from compute_LADDERS_v1p3_descriptive import ols, RANK, pack


def peak(rows,key):
    selected=[r for r in rows if r.get(key) is not None]
    if not selected:return None
    maximum=max(r[key] for r in selected)
    tied=[r for r in selected if r[key]==maximum]
    return dict(value=maximum,T_over_rs=tied[0]['T_over_rs'],indices=[r['source_index'] for r in tied])


def mask_flags(flags):
    state=True;result=[]
    for flag in flags:
        state=state and flag;result.append(state)
    return result


def stats(rows):
    prefix=[r for r in rows if r['certified']]
    plain_false=[r for r in rows if r['converged'] is False and r['contract'].startswith('plain')]
    confirming=[r for r in rows if r['confirming'] is not None]
    return dict(records=len(rows),stored_true=sum(r['converged'] is True for r in rows),
                stored_false=sum(r['converged'] is False for r in rows),counting=sum(r['counts'] for r in rows),
                kept_above=[r['T_over_rs'] for r in rows if r['kept']>1e-3],
                confirming_above=[r['T_over_rs'] for r in confirming if r['confirming']>1e-3],
                kept_maximum=peak(rows,'kept'),confirming_maximum=peak(confirming,'confirming'),
                prefix_kept_maximum=peak(prefix,'kept'),plain_kept_false=len(plain_false),
                plain_following=sum(r['confirming'] is not None for r in plain_false),
                banded_inside=[r['T_over_rs'] for r in prefix if r['banded']],
                banded_outside=[r['T_over_rs'] for r in rows if r['banded'] and not r['certified']],
                failures=[dict(T_over_rs=r['T_over_rs'],source_index=r['source_index'],reasons=r['reasons'])
                          for r in rows if not r['counts']])


def array_check(own,target):
    if own.shape!=target.shape:return dict(status='shape_mismatch',own_shape=own.shape,target_shape=target.shape)
    same=(own==target)|(np.isnan(own)&np.isnan(target))
    finite=np.isfinite(own)&np.isfinite(target)
    difference=np.abs(own[finite]-target[finite])
    return dict(status='agrees' if bool(np.all(same)) else 'array_mismatch',elements=int(own.size),
                mismatching_indices=np.flatnonzero(~same).tolist(),
                max_difference_index=(int(np.flatnonzero(finite)[np.argmax(difference)])
                                      if difference.size and np.max(difference)>0 else None),
                max_absolute_difference=float(np.max(difference)) if difference.size else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--target-npz',required=True);p.add_argument('--patch-existing',action='store_true')
    args=p.parse_args();out=Path(args.output)
    if args.patch_existing:
        answer=json.loads((out/'record_diagnostics.json').read_text())
        paths=json.loads(Path(args.inputs).read_text())
        for name,path in paths.items():
            raw=json.loads(Path(path).read_text());member=answer['members'][name]
            member['args']=raw['args']
            for row in member['residual_invN']['rows']:
                source=raw['records'][row['source_index']]
                row['gauge']=(source.get('gauge') or {}).get('intervals')
                row['W_negative']=(source.get('solver') or {}).get('W_negative')
            member['residual_invN']['above']=[row for row in member['residual_invN']['rows'] if row['value']>1e-3]
            above=member['residual_invN']['above']
            for row in above:
                coarse=row['ladder'].get('h_ell/32_and_/64')
                if not coarse:continue
                row['coarse_fine_precision']={}
                for precision in [80,160]:
                    mp.mp.dps=precision
                    fine=mp.mpf(row['value']);coarse_value=mp.mpf(coarse[1])
                    row['coarse_fine_precision'][str(precision)]=dict(
                        ratio=mp.nstr(coarse_value/fine,precision),
                        relative_change=mp.nstr(abs(coarse_value-fine)/abs(fine),precision))
            eligible=[row for row in above if 'coarse_fine_precision' in row]
            maximum=max(eligible,key=lambda row:mp.mpf(row['coarse_fine_precision']['160']['ratio'])) if eligible else None
            member['residual_invN']['above_summary']=dict(count=len(above),
                first=above[0]['T_over_rs'] if above else None,last=above[-1]['T_over_rs'] if above else None,
                max_ratio_row=None if maximum is None else dict(source_index=maximum['source_index'],
                    T_over_rs=maximum['T_over_rs'],precision=maximum['coarse_fine_precision']))
            member['npz_clock_gate']=answer['arrays'][name+'__T']['status']=='agrees'
        for check in answer['arrays'].values():
            if check.get('status')=='agrees':check['max_difference_index']=None
        rejected=array_check(np.array([np.nan]),np.array([0.]))['status']=='array_mismatch'
        if not rejected:raise RuntimeError('NaN versus zero rejection failed')
        answer['known_answers']['nan_vs_zero_rejection']=True
        (out/'record_diagnostics.json').write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(patched_members=len(paths),nan_vs_zero_rejection=True)))
        return
    paths=json.loads(Path(args.inputs).read_text());masks=json.loads((out/'residual_series.json').read_text())
    scalar=json.loads((out/'scalar_tables.json').read_text());target=np.load(args.target_npz,allow_pickle=False)
    fixture=array_check(np.array([1.,np.nan]),np.array([1.,np.nan]))
    if fixture['status']!='agrees':raise RuntimeError('NaN-aware array known answer failed')
    if array_check(np.array([1.]),np.array([1.,2.]))['status']!='shape_mismatch':raise RuntimeError('shape fixture failed')
    if mask_flags([True,False,True])!=[True,False,False]:raise RuntimeError('prefix fixture failed')
    answer=dict(known_answers=dict(nan_aware_array=fixture,shape_rejection=True,prefix=True),members={},arrays={},historical={})
    histories={'old_report':{},'intermediate_rule':{}};history_rows={'old_report':{},'intermediate_rule':{}}
    plot_arrays={}
    for name,path in paths.items():
        raw=json.loads(Path(path).read_text());rows=masks[name]['rows'];s=scalar[name]['scalars']
        own=json.loads(Path(scalar[name]['record_table']).read_text())
        lookup={r['source_index']:r for r in own}; old_flags=[];intermediate=[]
        for row in rows:
            is_newton=row['contract']=='Newton explicit'
            old_flags.append(row['converged'] is True and row['kept']<=1e-3 and
                             (not is_newton or row['confirming']<=raw['args']['stop_fields']))
            intermediate.append(row['kept']<=1e-3 and (not is_newton or row['confirming']<=1e-3))
        old_prefix,intermediate_prefix=mask_flags(old_flags),mask_flags(intermediate)
        engines=[]
        for row in rows:
            record=raw['records'][row['source_index']];engine=record.get('engine')
            if not engines or engines[-1]['engine']!=engine:
                engines.append(dict(engine=engine,first=row['T_over_rs'],last=row['T_over_rs'],count=1))
            else:engines[-1].update(last=row['T_over_rs'],count=engines[-1]['count']+1)
        centroids=[r['R_c'] for r in own]
        steps=[b-a for a,b in zip(centroids,centroids[1:])]
        native=[]
        for row in rows:
            solver=raw['records'][row['source_index']].get('solver') or {}
            value=solver.get('residual_invN')
            if value is not None:
                ladder=solver.get('residual_ladder') or {};fine=ladder.get('h_ell/64_and_/128');coarse=ladder.get('h_ell/32_and_/64')
                native.append(dict(source_index=row['source_index'],T_over_rs=row['T_over_rs'],value=value,
                    ladder=ladder,gauge=(raw['records'][row['source_index']].get('gauge') or {}).get('intervals'),
                    W_negative=solver.get('W_negative'),
                    coarse_over_fine=None if not coarse or value==0 else coarse[1]/value,
                    relative_change=None if not coarse or value==0 else abs(coarse[1]-value)/abs(value),
                    fine_scalar_difference=None if not fine else fine[1]-value))
        answer['members'][name]=dict(stats=stats(rows),args=raw['args'],engines=engines,script=raw['script'],mix=raw['args'].get('mix'),
            maxit=raw['args'].get('maxit'),dropped=masks[name]['summary']['dropped_without_E'],
            old_report=dict(counts=old_flags,prefix=old_prefix,
                            end=next((r['T_over_rs'] for r,flag in reversed(list(zip(rows,old_prefix))) if flag),None)),
            intermediate_rule=dict(counts=intermediate,prefix=intermediate_prefix,
                            end=next((r['T_over_rs'] for r,flag in reversed(list(zip(rows,intermediate_prefix))) if flag),None)),
            centroid_steps=dict(values=steps,positive=sum(v>0 for v in steps),negative=sum(v<0 for v in steps),
                                zero=sum(v==0 for v in steps),all_outward=all(v>0 for v in steps)),
            residual_invN=dict(rows=native,above=[r for r in native if r['value']>1e-3],maximum=peak(native,'value')))
        for rule,flags,prefix in [('old_report',old_flags,old_prefix),('intermediate_rule',intermediate,intermediate_prefix)]:
            amended=json.loads(json.dumps(masks[name]));replacement=[]
            for row,flag,in_prefix in zip(rows,flags,prefix):
                r=dict(row);r.update(counts=flag,certified=in_prefix)
                r['banded']=flag and max(r['kept'],r['confirming'] if r['contract']=='Newton explicit' else 0)>1e-4
                reasons=[]
                if rule=='old_report' and row['converged'] is not True:reasons.append('stored flag false')
                if row['kept']>1e-3:reasons.append('kept')
                if row['contract']=='Newton explicit' and row['confirming']>(raw['args']['stop_fields'] if rule=='old_report' else 1e-3):
                    reasons.append('strict confirming' if rule=='old_report' else 'confirming')
                r['reasons']=reasons;replacement.append(r)
            amended['rows']=replacement
            amended['summary'].update(counting=sum(flags),last_certified=next((r['T_over_rs'] for r in reversed(replacement) if r['certified']),None),
                first_failure=next((r for r in replacement if not r['counts']),None),
                banded_inside=[r['T_over_rs'] for r in replacement if r['banded'] and r['certified']],
                banded_outside=[r['T_over_rs'] for r in replacement if r['banded'] and not r['certified']])
            result=member_readout(path,amended);history_rows[rule][name]=result.pop('rows');histories[rule][name]=result
        columns={key:[] for key in ['T','EE0','E','Rt','Rt_rs','Rt_2gE','minNv','r_minNv','Rc','res','conf','slope',
                                    'four_gE_kappa_T','conv_stored','counts','nzeros','ntang','conf_fail','banded','certified']}
        for r in rows:
            record=raw['records'][r['source_index']];a=lookup[r['source_index']]
            values=dict(T=a['T_over_rs'],EE0=a['E_over_E0'],E=a['E'],Rt=a['R_t'],Rt_rs=a['R_t_over_rs'],
                        Rt_2gE=a['R_t_over_2gE'],minNv=a['F'],r_minNv=record.get('r_min_N_minus_absv'),Rc=a['R_c'],
                        res=r['kept'],conf=r['confirming'],slope=a['outer_slope'],
                        four_gE_kappa_T=(None if record.get('kappa_T') is None else 4*s['g']*record['E']*record['kappa_T']),
                        conv_stored=r['converged'],counts=r['counts'],nzeros=len(a['zeros']),ntang=len(a['tangencies'] or []),
                        conf_fail=r['confirming'] is not None and r['confirming']>1e-3,
                        banded=r['banded'],certified=r['certified'])
            for key,value in values.items():columns[key].append(np.nan if value is None else value)
        for key,values in columns.items():
            field=name+'__'+key;array=np.asarray(values,dtype=float);plot_arrays[field]=array
            answer['arrays'][field]=array_check(array,target[field]) if field in target else dict(status='target_unavailable')
        answer['members'][name]['npz_clock_gate']=answer['arrays'][name+'__T']['status']=='agrees'
        for field in target.files:
            if field.startswith(name+'__') and field not in plot_arrays:
                answer['arrays'][field]=dict(status='definition_unresolved',reason='no non-code auxiliary-column contract yet')
    np.savez_compressed(out/'own_plot_series.npz',**plot_arrays)
    (out/'record_diagnostics_base.json').write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
    mp.mp.dps=80
    from compute_LADDERS_v1p3_controls import PAIRS
    for rule,history in histories.items():
        comparisons={}
        for name,controls in PAIRS.items():
            for control in controls:
                result=pair(history[name],history[control],history_rows[rule][name],history_rows[rule][control])
                result.pop('samples');comparisons[name+'__'+control]=result
        answer['historical'][rule]=pack(dict(members=history,control_pairs=comparisons))
    width_names=['P64_s4','P64_s8','S4_128_s8','S4_128_s16'];design=[]
    for name in width_names:
        s=scalar[name]['scalars'];z=mp.log(1/mp.mpf(s['rs']));h=mp.log(s['sigma']);design.append([1,z,h,z*h])
    old_width={}
    for key in ['T_f','R_t_over_rs','R_t_over_2gE']:
        ys=[]
        for name in width_names:
            row=scalar[name]['formation']['row'];s=scalar[name]['scalars']
            ys.append(mp.mpf(row['T'])/s['rs'] if key=='T_f' else mp.mpf(row['R_t'])/s['rs'] if key=='R_t_over_rs'
                      else mp.mpf(row['R_t'])/(2*mp.mpf(s['g'])*mp.mpf(row['E'])))
        old_width[key]=ols(design,ys)
    answer['historical']['old_width_fit']=pack(dict(members=width_names,ordinates=old_width))
    rank_names=[n for n in RANK if n not in ['RS32_s2','P64_s2']];matrix=[]
    for name in rank_names:
        s=scalar[name]['scalars'];x=64/mp.mpf(s['rs']);z=x**(mp.mpf(4)/3);h=(16*mp.mpf(s['sigma'])/s['rs'])**2;j=8*mp.mpf(s['tau'])/s['rs']
        matrix.append([1,z,x*x,h,z*h,j,j*j,z*j])
    sv=mp.svd(mp.matrix(matrix),compute_uv=False);threshold=mp.eps*8*sv[0]
    answer['historical']['old_rank']=pack(dict(members=rank_names,design=matrix,singular_values=list(sv),rank=sum(v>threshold for v in sv),threshold=threshold))
    (out/'record_diagnostics.json').write_text(json.dumps(answer,indent=2,allow_nan=False)+'\n')
    statuses={}
    for result in answer['arrays'].values():statuses[result['status']]=statuses.get(result['status'],0)+1
    print(json.dumps(dict(members=len(answer['members']),array_statuses=statuses)),flush=True)


if __name__=='__main__':main()
