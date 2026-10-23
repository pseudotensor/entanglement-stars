"""Convergence/conditioning, quoted comparator, and target-diff bookkeeping."""
import argparse
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import mpmath as mp


def compare(low,high,path=''):
    results=[]
    if isinstance(low,dict) or isinstance(high,dict):
        if not isinstance(low,dict) or not isinstance(high,dict) or low.keys()!=high.keys():
            raise ValueError('precision key/type structures differ at '+path)
        for key in sorted(low):
            if key in ['rank_threshold','threshold']:continue
            results.extend(compare(low[key],high[key],path+'/'+key))
    elif isinstance(low,list) or isinstance(high,list):
        if not isinstance(low,list) or not isinstance(high,list) or len(low)!=len(high):
            raise ValueError('precision list structures differ at '+path)
        for index,(a,b) in enumerate(zip(low,high)):results.extend(compare(a,b,path+'/'+str(index)))
    elif low is None or high is None or isinstance(low,bool) or isinstance(high,bool):
        if type(low)!=type(high) or low!=high:raise ValueError('precision null/boolean differs at '+path)
    else:
        if type(low)!=type(high):raise ValueError('precision leaf types differ at '+path)
        if not isinstance(low,(str,float,int)):raise TypeError('unsupported precision leaf at '+path)
        try:a,b=mp.mpf(low),mp.mpf(high)
        except (ValueError,TypeError):
            if not isinstance(low,str) or low!=high:raise ValueError('precision text differs at '+path)
            return []
        if not mp.isfinite(a) or not mp.isfinite(b):raise ValueError('nonfinite precision leaf at '+path)
        scale=max(mp.mpf(1),abs(a),abs(b));difference=abs(a-b)
        results.append(dict(path=path,absolute_difference=mp.nstr(difference,80),
                            scaled_difference=mp.nstr(difference/scale,80)))
    return results


def changed_leaves(old,new,path='',group_changed_length_lists=False):
    if isinstance(old,dict) and isinstance(new,dict):
        result=[]
        for key in sorted(old.keys()|new.keys()):
            child=path+'/'+str(key)
            if key not in old or key not in new:
                result.append(dict(path=child,change='added' if key not in old else 'removed'))
            else:result.extend(changed_leaves(old[key],new[key],child,group_changed_length_lists))
        return result
    if isinstance(old,list) and isinstance(new,list):
        if group_changed_length_lists and len(old)!=len(new):return [dict(path=path,change='list length changed')]
        result=[]
        for index in range(max(len(old),len(new))):
            child=path+'/'+str(index)
            if index>=len(old) or index>=len(new):result.append(dict(path=child,change='added' if index>=len(old) else 'removed'))
            else:result.extend(changed_leaves(old[index],new[index],child,group_changed_length_lists))
        return result
    return [] if type(old)==type(new) and old==new else [dict(path=path,change='changed')]


def diff_bookkeeping(targets):
    old=json.loads((targets/'ladders_readout_v1_2.json').read_text());new=json.loads((targets/'ladders_readout_v1_3.json').read_text())
    artifact=json.loads((targets/'ladders_diff_v1_3.json').read_text());leaves=changed_leaves(old,new)
    entries=artifact['changed'];bands=[entry for entry in entries if entry['path'].endswith('/banded') and entry.get('old')=='<absent>']
    def canonical(path):
        parts=path.split('/')
        if len(parts)>3 and parts[0]=='spreads' and parts[1].isdigit():
            index=int(parts[1]);candidate=new['spreads'][index]
            expected=candidate['member']+' vs '+candidate['control']
            if parts[2]!=expected:raise ValueError('spread path annotation does not match bound pair')
            parts.pop(2)
        return '/'+('/'.join(parts))
    recorded={canonical(entry['path']) for entry in entries+artifact.get('provenance',[])}
    def at(data,path):
        value=data
        for part in path.strip('/').split('/'):
            if isinstance(value,dict):
                if part not in value:return '<absent>'
                value=value[part]
            else:
                index=int(part)
                if index>=len(value):return '<absent>'
                value=value[index]
        return value
    unrecorded=[]
    for leaf in leaves:
        if not any(leaf['path']==path or leaf['path'].startswith(path+'/') for path in recorded):
            unrecorded.append(dict(**leaf,old=at(old,leaf['path']),new=at(new,leaf['path'])))
    return dict(saved_entry_count=len(entries),claimed_count=artifact['n_changed'],
        saved_provenance_count=len(artifact.get('provenance',[])),newly_added_per_record_banded_entries=len(bands),
        all_newly_added_banded_named_entries=sum('banded' in entry['path'] and entry.get('old')=='<absent>' for entry in entries),
        own_structural_changes=len(leaves),own_grouped_list_changes=len(changed_leaves(old,new,group_changed_length_lists=True)),
        unrecorded_changes=unrecorded,
        filtering_convention_unresolved=bool(unrecorded),
        convention='Structural count recurses through common list indices. Grouped count records changed-length lists once. A recorded ancestor covers its changed descendants.',
        physical_payloads_independently_checked=False,own_paths=leaves)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--targets',required=True)
    p.add_argument('--w4',required=True);p.add_argument('--bookkeeping-only',action='store_true');p.add_argument('--precision-only',action='store_true')
    args=p.parse_args();out=Path(args.output);targets=Path(args.targets)
    if args.bookkeeping_only:
        summary=json.loads((out/'precision_bookkeeping.json').read_text())
        summary['target_diff_bookkeeping']=diff_bookkeeping(targets)
        (out/'precision_bookkeeping.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in summary['target_diff_bookkeeping'].items() if k not in ['own_paths','unrecorded_changes']}))
        return
    mp.mp.dps=180;summary=dict(precision={},conditions={},comparators={},historical_null={})
    fixtures=[('added dictionary key',{'a':1},{'a':1,'b':2}),('different list length',[1],[1,2]),
              ('different leaf type',1,'1'),('unequal null',None,0),('unequal boolean',True,False),
              ('unequal text','alpha','beta'),('nonfinite numeric','nan','nan'),('nonfinite float',float('inf'),float('inf'))]
    validation=[]
    for name,low,high in fixtures:
        rejected=False
        try:compare(low,high)
        except (ValueError,TypeError):rejected=True
        validation.append(dict(name=name,passed=rejected))
    validation.append(dict(name='matching nonnumeric leaves',passed=compare({'a':None,'b':True,'c':'label'},{'a':None,'b':True,'c':'label'})==[]))
    if not all(row['passed'] for row in validation):raise RuntimeError('precision structure control failed')
    for file in ['descriptive_arithmetic','controls_catalogs','scalar_precision_checks']:
        data=json.loads((out/(file+'.json')).read_text())
        if file=='scalar_precision_checks':
            deltas=[]
            for name,values in data.items():deltas.extend(compare(values['80'],values['160'],'/'+name))
        else:
            if file=='controls_catalogs':
                control=data['80'].pop('at16_negative_control')
                if control.get('passed') is not True:raise ValueError('declared one-precision at16 fixture failed')
            deltas=compare(data['80'],data['160'])
        maximum=max(deltas,key=lambda r:mp.mpf(r['scaled_difference']))
        passed=mp.mpf(maximum['scaled_difference'])<mp.mpf('1e-60')
        if not passed:raise RuntimeError('precision convergence failed: '+file)
        summary['precision'][file]=dict(compared_numeric_paths=len(deltas),maximum_scaled=maximum,passed=passed,
                                       note='Observed convergence estimate; not a rigorous rounding-error enclosure.')
    if args.precision_only:
        saved=json.loads((out/'precision_bookkeeping.json').read_text());saved['precision']=summary['precision'];saved['precision_validation_controls']=validation
        saved['precision_structure_validation']='Equal dictionary keys, ordered list lengths and compatible leaf types; null, boolean and textual leaves equal; all numeric leaves finite. Declared precision-dependent thresholds excluded explicitly. The controls_catalogs 80-only at16_negative_control fixture is separately required to pass and excluded from the paired scientific-field trees.'
        (out/'precision_bookkeeping.json').write_text(json.dumps(saved,indent=2,allow_nan=False)+'\n')
        print(json.dumps(saved['precision_structure_validation']));return
    descriptive=json.loads((out/'descriptive_arithmetic.json').read_text())['160']
    for fit,block in descriptive['fits'].items():
        summary['conditions'][fit]={}
        for key,value in block['ordinates'].items():
            singular=value['singular_values'];condition=mp.mpf(singular[0])/mp.mpf(singular[-1])
            summary['conditions'][fit][key]=dict(condition_estimate=mp.nstr(condition,40),
                    normal_equation_residual=value['normal_equation_residual'],rank=value['rank'],dof=value['dof'],
                    note='Condition estimate from unscaled design singular values; precision comparison retained separately.')
    scalars=json.loads((out/'scalar_tables.json').read_text());s=scalars['P64_s4']['scalars']
    for precision in [80,160]:
        mp.mp.dps=precision
        def clock(b):
            return (b+2)*mp.sqrt(b-1)*mp.acos(1/mp.sqrt(b))+(mp.sqrt(b)-1)**2+2*mp.log(1+1/mp.sqrt(b))
        rho=2*mp.mpf(s['g'])*mp.mpf(s['E0'])/mp.mpf(s['rs'])
        summary['comparators'][str(precision)]=dict(boundary_clock_b4=mp.nstr(clock(mp.mpf(4)),precision),
            production_rho=mp.nstr(rho,precision),production_nominal_clock=mp.nstr(rho*clock(4/rho),precision),
            source='conditional exterior coordinate-rest boundary-arrival comparator')
        triples=[(16,4,8),(32,4,2),(64,4,8),(64,2,8),(64,8,8),(64,4,4),(64,4,2),(128,16,8)]
        matrix=[]
        for rs,sigma,tau in triples:
            x=mp.mpf(64)/rs;z=x**(mp.mpf(4)/3);h=(mp.mpf(16)*sigma/rs)**2;j=mp.mpf(8)*tau/rs
            matrix.append([1,z,x*x,h,z*h,j,j*j,z*j])
        null=mp.matrix([-mp.mpf(2)/3,mp.mpf(2)/3,0,mp.mpf(7)/24,-mp.mpf(7)/24,-1,0,1])
        residual=mp.matrix(matrix)*null;singular=mp.svd(mp.matrix(matrix),compute_uv=False);threshold=8*mp.eps*singular[0]
        rank=sum(value>threshold for value in singular)
        if rank!=7 or max(abs(v) for v in residual)>mp.mpf(10)**(-precision+10):raise RuntimeError('historical null control failed')
        summary['historical_null'][str(precision)]=dict(triples=triples,rank=rank,
            null=[mp.nstr(v,precision) for v in null],maximum_residual=mp.nstr(max(abs(v) for v in residual),precision),passed=True)
    w4=json.loads(Path(args.w4).read_text());summary['W4_primitives']={key:w4.get(key) for key in ['profile','E0','two_gE0','fold','first_record']}
    summary['target_diff_bookkeeping']=diff_bookkeeping(targets)
    (out/'precision_bookkeeping.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(convergence_checks=3,historical_null=True,
        diff_entries=summary['target_diff_bookkeeping']['saved_entry_count'],
        band_entries=summary['target_diff_bookkeeping']['newly_added_per_record_banded_entries'])),flush=True)


if __name__=='__main__':main()
