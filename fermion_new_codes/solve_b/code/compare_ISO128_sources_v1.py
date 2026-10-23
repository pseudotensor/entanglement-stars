"""Post-freeze comparison with the original numeric T5 shell-source arrays."""
import argparse
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from iso128_rebuild_core_v1 import file_identity, source_file_bindings, source_comparison_index, difference_details
from iso128_rebuild_core_v1 import validate_inventory, require_source_cpu, validate_bound_comparison_source, validate_saved_arrays


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--outdir',type=Path,required=True);args=p.parse_args();runtime=require_source_cpu(args.outdir)
    if any(args.outdir.iterdir()):raise ValueError('Refusing comparison overwrite/replay')
    current=source_file_bindings(args.spec);complete_path=args.sources/'source_complete.json'
    complete=json.loads(complete_path.read_text())
    if not complete['available'] or complete['T']!=640. or complete['spec_source_bindings']!=current:
        raise ValueError('Expected frozen complete own T5 source artifact')
    for item in complete['products']:
        if file_identity(item['path'])!=item:raise ValueError('Frozen source product changed')
    inventory=json.loads(args.inventory.read_text());fields_entry=inventory['archives']['stored_fields']['identity']
    validate_inventory(inventory)
    if file_identity(fields_entry['path'])!=fields_entry:raise ValueError('Original fields identity changed')
    if file_identity(inventory['run_identity']['path'])!=inventory['run_identity']:raise ValueError('Original JSON changed')
    run=json.loads(Path(inventory['run_identity']['path']).read_text());record=run['records'][80]
    output={};summaries={}
    with np.load(args.sources/'combined_sources.npz',allow_pickle=False) as own, np.load(fields_entry['path'],allow_pickle=False) as archive:
        validate_bound_comparison_source(complete,file_identity(args.sources/'combined_sources.npz'),
                                         {key:own[key] for key in ['T','rs','E0','R']})
        fields={key:archive[key] for key in ['eps_tot','J_tot','eps_pairs','J_pairs']}
        index=source_comparison_index(own['R'],archive['R'],archive['T'],record,fields)
        output['R']=own['R']
        for schedule in ['s64','s127']:
            for key,role,j in [('eps_tot','total',0),('J_tot','total',1),('eps_pairs','pairs',0),('J_pairs','pairs',1)]:
                summary,arrays=difference_details(own[schedule+'_'+role][:,j].real,fields[key][index],own['scale_'+role][:,j],own['R'])
                summary['own_finite_count']=int(np.isfinite(own[schedule+'_'+role][:,j]).sum())
                summary['stored_finite_count']=int(np.isfinite(fields[key][index]).sum())
                summaries[schedule+'_'+key]=summary
                output[schedule+'_'+key+'_own']=own[schedule+'_'+role][:,j].real
                output[schedule+'_'+key+'_stored']=fields[key][index]
                for name,a in arrays.items():output[schedule+'_'+key+'_'+name]=a
    validate_saved_arrays(output)
    arrays_path=args.outdir/'T5_source_comparison.npz';np.savez(arrays_path,**output)
    result={'scope':'post-freeze original T5 source comparison only','available':True,'runtime':runtime,
            'spec_source_bindings':current,'own_complete':file_identity(complete_path),'original_fields':fields_entry,
            'original_run':inventory['run_identity'],'index':index,'T':640.,'T_over_rs':5.,'comparisons':summaries,
            'arrays':file_identity(arrays_path),'T11_5_original_record':'unavailable; no substituted foreign rebuild'}
    (args.outdir/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'comparison':str(args.outdir/'comparison.json')}))


if __name__=='__main__':main()
