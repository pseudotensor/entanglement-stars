"""Post-freeze applied-field comparisons; never used to select own solutions."""
import argparse,json
from pathlib import Path
import numpy as np
from iso128_rebuild_core_v1 import file_identity,require_source_cpu,validate_inventory,source_file_bindings
from control_ISO128_fields_v1 import field_bindings
from iso128_field_readout_v1 import compare_applied


def validate_field_association(complete,source,loaded,clock,current_field,current_source):
    expected={'T0005.00':(640.,5.),'T0011.50':(1472.,11.5)}
    if clock not in expected:raise ValueError('Unknown own field clock')
    T,t=expected[clock]
    if complete['clock']!=clock or complete['T']!=T or complete['T_over_rs']!=t or complete['field_bindings']!=current_field:
        raise ValueError('Wrong field completion clock/current binding')
    if source['T']!=T or source['T_over_rs']!=t or source['spec_source_bindings']!=current_source:
        raise ValueError('Wrong own source clock/current binding')
    associated=[item for item in complete['products'] if Path(item['path']).resolve()==Path(loaded['path']).resolve()]
    if associated!=[loaded]:raise ValueError('Actual loaded primary arrays lack unique frozen-product association')


def main():
    p=argparse.ArgumentParser()
    for key in ('outdir','t5-fields','t115-fields','inventory','field-spec','source-spec'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();runtime=require_source_cpu(a.outdir)
    if any(a.outdir.iterdir()):raise ValueError('Fresh post-freeze comparison output required')
    frozen={}
    current_field=field_bindings(a.field_spec);current_source=source_file_bindings(a.source_spec)
    for clock,directory in [('T0005.00',a.t5_fields),('T0011.50',a.t115_fields)]:
        complete_path=directory/'field_complete.json';complete=json.loads(complete_path.read_text())
        if complete['clock']!=clock or complete['T']!=({'T0005.00':640.,'T0011.50':1472.}[clock]) or complete['field_bindings']!=current_field:raise ValueError('Wrong field-completion clock/current bindings')
        source_identity=complete['source_complete']
        if file_identity(source_identity['path'])!=source_identity:raise ValueError('Frozen source completion changed')
        source=json.loads(Path(source_identity['path']).read_text())
        if source['spec_source_bindings']!=current_source or source['T']!=complete['T']:raise ValueError('Source current-binding/clock mismatch')
        for item in complete['products']:
            if file_identity(item['path'])!=item:raise ValueError('Frozen field product changed')
        frozen[clock]={'directory':directory,'complete':complete,'identity':file_identity(complete_path)}
    inventory=json.loads(a.inventory.read_text());validate_inventory(inventory)
    summaries={};arrays={}
    for clock,row in frozen.items():
        if not row['complete']['variants']['P']['available']:
            summaries[clock]={'available':False,'reason':'primary field reconstruction unavailable'};continue
        entry=inventory['archives'][clock]['identity']
        if file_identity(entry['path'])!=entry:raise ValueError('Original slice identity changed')
        with np.load(entry['path'],allow_pickle=False) as archive:
            raw={key:archive[key] for key in ('T','R','N_applied','v_applied')}
        with np.load(row['directory']/'P'/'field_arrays.npz',allow_pickle=False) as archive:
            ownR=archive['sites_r'];average=archive['average_primary']
        loaded=file_identity(row['directory']/'P'/'field_arrays.npz')
        source=json.loads(Path(row['complete']['source_complete']['path']).read_text())
        validate_field_association(row['complete'],source,loaded,clock,current_field,current_source)
        summary={'available':True,'slice':entry,'fields':{}}
        for column,role in enumerate(('N_applied','v_applied')):
            difference=compare_applied(average[:,column],raw[role],ownR,raw['R'],float(raw['T']),row['complete']['T'],role)
            arrays[clock+'_'+role+'_difference']=difference.pop('difference')
            arrays[clock+'_'+role+'_own']=average[:,column];arrays[clock+'_'+role+'_stored']=raw[role]
            summary['fields'][role]=difference
        arrays[clock+'_R']=ownR;summaries[clock]=summary
        if clock=='T0005.00':
            fields_entry=inventory['archives']['stored_fields']['identity']
            if file_identity(fields_entry['path'])!=fields_entry:raise ValueError('Original fields archive changed')
            with np.load(fields_entry['path'],allow_pickle=False) as archive:
                times=archive['T'];R=archive['R'];indexes=np.flatnonzero(times==640.)
                if indexes.tolist()!=[80] or not np.array_equal(R,raw['R']):raise ValueError('Original duplicate clock/radius association failed')
                for column,role in enumerate(('N_applied','v_applied')):
                    reference=archive[role]
                    if reference.shape!=(len(times),len(R)):raise ValueError('Original duplicate applied shape mismatch')
                    difference=compare_applied(average[:,column],reference[80],ownR,R,640.,640.,role)
                    arrays[clock+'_duplicate_'+role+'_difference']=difference.pop('difference')
                    summary['fields']['duplicate_'+role]=difference
                    summary['fields']['duplicate_'+role]['slice_duplicate_exact']=bool(np.array_equal(raw[role],reference[80]))
    arrays_path=a.outdir/'applied_comparison.npz';np.savez(arrays_path,**arrays)
    run_entry=inventory['run_identity']
    if file_identity(run_entry['path'])!=run_entry:raise ValueError('Original run JSON identity changed')
    record=json.loads(Path(run_entry['path']).read_text())['records'][80]
    if record['T_over_rs']!=5.:raise ValueError('Original T5 log record association failed')
    result={'scope':'post-freeze applied-field comparison only; fixed-reference residual is not a replay',
        'runtime':runtime,'implementation':file_identity(__file__),'own_frozen':{k:v['identity'] for k,v in frozen.items()},
        'comparisons':summaries,'original_T5_record':record,'arrays':file_identity(arrays_path),
        'original_T11_5_log':'not present in the original record archive; no substituted reconstruction'}
    (a.outdir/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'comparison':str(a.outdir/'comparison.json')}))


if __name__=='__main__':main()
