"""Exact represented-array identity on the fixed fifteen-field contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

FIELDS=['T','EE0','E','Rt','Rt_rs','Rt_2gE','minNv','r_minNv','Rc','res','slope','four_gE_kappa_T','conv_stored','nzeros','ntang']
ROOT_FIELDS={'Rt','Rt_rs','Rt_2gE','slope'}
BOOL_FIELDS={'conv_stored'}
COUNT_FIELDS={'nzeros','ntang'}


def clock_gate(values,expected):
    if values.dtype.kind not in 'bifu' or values.ndim!=1:return dict(passed=False,reason='non-real/object or non-vector clock')
    if values.shape!=expected.shape:return dict(passed=False,reason='clock length/shape differs')
    if not np.isfinite(values).all() or not (np.diff(values)>0).all():return dict(passed=False,reason='clock not finite strictly increasing')
    equal=bool(np.array_equal(values,expected))
    return dict(passed=equal,reason='complete ordered raw-record clock identity' if equal else 'clock differs from complete raw-record sequence')


def field_gate(field,array,shape,missing):
    if array.dtype.kind not in 'bifu':return dict(passed=False,reason='non-real or object encoding')
    if array.ndim!=1 or array.shape!=shape:return dict(passed=False,reason='field/full-clock shape differs')
    if np.isinf(array).any():return dict(passed=False,reason='unexpected infinity')
    if not np.array_equal(np.isnan(array),missing):return dict(passed=False,reason='missing pattern differs from raw contract')
    if field in BOOL_FIELDS:
        good=np.isfinite(array).all() and np.logical_or(array==0,array==1).all()
        return dict(passed=bool(good),reason='validated lossless boolean encoding' if good else 'invalid boolean encoding',dtype=str(array.dtype))
    if field in COUNT_FIELDS:
        good=np.isfinite(array).all() and (array>=0).all() and (np.floor(array)==array).all()
        return dict(passed=bool(good),reason='finite nonnegative integer counts' if good else 'invalid count encoding',dtype=str(array.dtype))
    return dict(passed=True,reason='real numeric values and raw-defined missing pattern',dtype=str(array.dtype))


def value(v):return 'NaN' if np.isnan(v) else float(v)


def availability(files,key):
    absent=[label for label,file in files.items() if key not in file]
    return dict(status='input_unavailable' if absent else 'available',missing_from=absent)


def comparison(a,b,normalized,physical,finite_positions):
    same_missing=np.array_equal(np.isnan(a),np.isnan(b))
    both=np.isfinite(a)&np.isfinite(b)
    different=np.flatnonzero((np.isnan(a)!=np.isnan(b))|(both&(a!=b)))
    finite_difference=np.abs(a[both]-b[both])
    maximum=float(np.max(finite_difference)) if finite_difference.size else None
    if maximum is None:
        ties=dict(reason='no finite pairs',count=0)
    elif maximum==0:
        ties=dict(all_finite_positions_reference='field.finite_positions',count=len(finite_positions))
    else:
        positions=np.flatnonzero(both)
        ties=positions[np.flatnonzero(finite_difference==maximum)].tolist()
    details=[dict(position=int(i),T_over_rs=float(normalized[i]),T=float(physical[i]),left=value(a[i]),right=value(b[i])) for i in different]
    return dict(status='exact_identity' if same_missing and not len(different) else 'array_mismatch',
                exact_finite_values=bool(not np.any(both&(a!=b))),exact_missing_pattern=bool(same_missing),
                differing_positions=details,finite_pair_count=int(both.sum()),maximum_finite_difference=maximum,maximum_ties=ties)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--old',required=True);p.add_argument('--current',required=True)
    args=p.parse_args();out=Path(args.output)
    if os.environ.get('QI2_REMOTE_CPU')!='1' or sys.executable!='<internal>/python':raise RuntimeError('prescribed remote CPU interpreter required')
    # Structural controls precede source comparison.
    expected=np.array([0.,1.,2.]);physical=8*expected;good=np.array([1.,2.,3.]);zero_missing=np.zeros(3,dtype=bool)
    controls=[]
    c=lambda name,passed:controls.append(dict(name=name,passed=bool(passed)))
    c('exact three-way match',all(comparison(a,b,expected,physical,[0,1,2])['status']=='exact_identity' for a,b in [(good,good.copy()),(good.copy(),good),(good,good)]))
    c('reordered clock rejected',not clock_gate(np.array([0.,2.,1.]),expected)['passed'])
    c('duplicate clock rejected',not clock_gate(np.array([0.,1.,1.]),expected)['passed'])
    c('changed monotone clock rejected',not clock_gate(np.array([0.,1.,2.1]),expected)['passed'])
    c('shape mismatch rejected',not field_gate('E',good[:2],expected.shape,zero_missing)['passed'])
    missing=np.array([False,True,False]);nan_array=np.array([1.,np.nan,3.])
    c('NaN versus zero rejected',not field_gate('Rt',np.array([1.,0.,3.]),expected.shape,missing)['passed'])
    c('matched authorized NaNs retained',field_gate('Rt',nan_array,expected.shape,missing)['passed'] and comparison(nan_array,nan_array.copy(),expected,physical,[0,2])['status']=='exact_identity')
    absent=availability(dict(old={'member__E':good},current={},own={'member__E':good}),'member__E')
    c('missing field unavailable with source label',absent==dict(status='input_unavailable',missing_from=['current']))
    all_missing=comparison(np.full(3,np.nan),np.full(3,np.nan),expected,physical,[])
    c('all missing exact pattern has no finite maximum',all_missing['status']=='exact_identity' and all_missing['finite_pair_count']==0 and all_missing['maximum_finite_difference'] is None and all_missing['maximum_ties']['count']==0)
    changed=np.array([1.,2.5,3.]);answer=comparison(good,changed,expected,physical,[0,1,2])
    c('changed primitive and physical-clock locator retained',answer['status']=='array_mismatch' and answer['maximum_finite_difference']==.5 and answer['differing_positions'][0]['position']==1 and answer['differing_positions'][0]['T']==8)
    c('invalid boolean rejected',not field_gate('conv_stored',np.array([0.,.5,1.]),expected.shape,zero_missing)['passed'])
    c('negative count rejected',not field_gate('nzeros',np.array([0.,-1.,2.]),expected.shape,zero_missing)['passed'])
    c('fractional count rejected',not field_gate('ntang',np.array([0.,1.5,2.]),expected.shape,zero_missing)['passed'])
    c('object encoding rejected',not field_gate('E',np.array([1.,2.,3.],dtype=object),expected.shape,zero_missing)['passed'])
    c('complex encoding rejected',not field_gate('E',np.array([1.,2.,3.],dtype=complex),expected.shape,zero_missing)['passed'])
    c('infinity rejected',not field_gate('E',np.array([1.,np.inf,3.]),expected.shape,zero_missing)['passed'])
    if not all(r['passed'] for r in controls):raise RuntimeError('historical array structural control failed')
    old_path=Path(args.old);h=hashlib.md5();sha=hashlib.sha256()
    with old_path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk);sha.update(chunk)
    old_digest=h.hexdigest()
    if old_digest!='0de83133b07ca8b2f3533c530e0456f3':raise RuntimeError('old NPZ differs from specified footer identity')
    files=dict(old=np.load(args.old,allow_pickle=False),current=np.load(args.current,allow_pickle=False),own=np.load(out/'own_plot_series.npz',allow_pickle=False))
    summaries=json.loads((out/'scalar_tables.json').read_text());residuals=json.loads((out/'residual_series.json').read_text())
    expected_members=set(summaries)
    result=dict(scope='Exact fifteen-field represented-array identity only; no certification/confirmation/undefined-auxiliary, full raw/profile/parent or full-readout-payload identity claim.',
                known_answers=controls,old_source=dict(path=str(old_path),md5=old_digest,sha256=sha.hexdigest()),
                inventory={label:sorted(f.files) for label,f in files.items()},member_bindings={},fields={})
    for label,file in files.items():
        actual={key.split('__')[0] for key in file.files}
        if actual!=expected_members or any('__' not in key for key in file.files):raise RuntimeError('array member binding differs: '+label)
    totals={};unavailable=[]
    for name,member in summaries.items():
        own_rows=residuals[name]['rows'];rs=member['scalars']['rs']
        normalized=np.asarray([r['T_over_rs'] for r in own_rows]);physical=np.asarray([r['T'] for r in own_rows])
        raw=json.loads(Path(member['source']).read_text());scalar_rows={r['source_index']:r for r in json.loads(Path(member['record_table']).read_text())}
        for r in own_rows:
            if raw['records'][r['source_index']]['T']!=r['T']:raise RuntimeError('raw source-index clock drift')
        gates={}
        for label,file in files.items():
            key=name+'__T';gates[label]=clock_gate(file[key],normalized) if key in file.files else dict(passed=False,reason='clock unavailable')
        result['member_bindings'][name]=dict(source=member['source'],source_indices=[r['source_index'] for r in own_rows],rs=rs,
            full_clock_length=len(normalized),clock_gates=gates)
        admitted=all(g['passed'] for g in gates.values())
        for field in FIELDS:
            key=name+'__'+field;entry=dict(member=name,field=field)
            available=availability(files,key);absent=available['missing_from']
            if absent:
                entry.update(status='input_unavailable',missing_from=absent);unavailable.append(dict(key=key,missing_from=absent));result['fields'][key]=entry;continue
            if not admitted:
                entry.update(status='clock_gate_failed');result['fields'][key]=entry;continue
            missing=[]
            for r in own_rows:
                row=scalar_rows[r['source_index']];record=raw['records'][r['source_index']]
                missing.append(row['R_t'] is None if field in ['Rt','Rt_rs','Rt_2gE'] else row['outer_slope'] is None if field=='slope' else
                               record.get('kappa_T') is None if field=='four_gE_kappa_T' else record.get('r_min_N_minus_absv') is None if field=='r_minNv' else False)
            missing=np.asarray(missing,dtype=bool);arrays={label:file[key] for label,file in files.items()}
            validation={label:field_gate(field,a,normalized.shape,missing) for label,a in arrays.items()}
            entry.update(validations=validation,finite_positions=np.flatnonzero(~missing).tolist())
            if not all(v['passed'] for v in validation.values()):entry['status']='field_gate_failed'
            else:
                comparisons={a+'/'+b:comparison(arrays[a],arrays[b],normalized,physical,entry['finite_positions']) for a,b in [('old','own'),('current','own'),('old','current')]}
                entry.update(comparisons=comparisons,status='exact_identity' if all(c['status']=='exact_identity' for c in comparisons.values()) else 'array_mismatch')
            result['fields'][key]=entry
    for item in result['fields'].values():totals[item['status']]=totals.get(item['status'],0)+1
    result['summary']=totals;result['unavailable']=unavailable
    (out/'historical_physical_arrays.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(members=len(summaries),controls=len(controls),fields=totals,three_way_comparisons=3*totals.get('exact_identity',0))))


if __name__=='__main__':main()
