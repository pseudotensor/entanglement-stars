"""Expose frozen initial values and exponential metadata without reevaluating observables."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from packet_core_v1 import file_hash, validate_observable_schema
from summarize_PACKET_secondroute_v1 import clean

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();runtime=require_remote_cpu(args.output)
freeze=json.loads((args.output/'result_bindings.json').read_text())
bound={row['path']:row['sha256'] for row in freeze['files']}
folder=args.output/'review';initial=[];exp_rows=[];exp_summary={}
for channel in (1,5):
    for mode in ('E','I16','I32'):
        prefix=f'k{channel}/{mode}'
        for suffix in ('_observables.npz','_complete.json','_exponentials.json'):
            if file_hash(args.output/(prefix+suffix)) != bound[prefix+suffix]:
                raise ValueError('frozen context changed')
        complete=json.loads((args.output/(prefix+'_complete.json')).read_text())
        with np.load(args.output/(prefix+'_observables.npz')) as archive:
            data={key:archive[key] for key in archive.files}
        validate_observable_schema(data,complete['columns'])
        for column,label in enumerate(complete['labels']):
            values={key:value[0,...,column] for key,value in data.items() if value.ndim>=2 and value.shape[0]==224 and value.shape[-1]==complete['columns']}
            initial.append({'channel':channel,'mode':mode,'column':column,'label':label,'record':int(data['record_indices'][0]),'T':float(data['T'][0]),'values':values})
        exp=json.loads((args.output/(prefix+'_exponentials.json')).read_text())
        metadata={'exponential_count':len(exp),'record_count':complete['records'],'interval_count':complete['intervals'],
            'columns':complete['columns'],'ideal_tail_sum_saved':complete['ideal_tail_sum'],'extrema':{}}
        for key in ('L','z','degree','tail'):
            values=[row[key] for row in exp]
            metadata['extrema'][key]={}
            for label,value in (('minimum',min(values)),('maximum',max(values))):
                metadata['extrema'][key][label]={'value':value,'locators':[[row['record'],row['substep']] for row in exp if row[key]==value]}
        norm_entries=[]
        for row in exp:
            for phase in ('before','after'):
                norm_entries.extend((abs(value-1),row['record'],row['substep'],phase,column) for column,value in enumerate(row[phase+'_norm']))
            before=[abs(value-1) for value in row['before_norm']];after=[abs(value-1) for value in row['after_norm']]
            exp_rows.append({'channel':channel,'mode':mode,**{key:row[key] for key in ('record','substep','fraction','L','z','degree','tail')},
                'before_norm_drift_max':max(before),'before_columns':json.dumps([i for i,x in enumerate(before) if x==max(before)],separators=(',',':')),
                'after_norm_drift_max':max(after),'after_columns':json.dumps([i for i,x in enumerate(after) if x==max(after)],separators=(',',':'))})
        maximum=max(row[0] for row in norm_entries)
        metadata['maximum_raw_norm_drift']={'value':maximum,'locators':[list(row[1:]) for row in norm_entries if row[0]==maximum]}
        exp_summary[f'k{channel}_{mode}']=metadata
initial_path=folder/'initial_observables_v1.json';initial_path.write_text(json.dumps(clean(initial),indent=2,allow_nan=False)+'\n')
exp_path=folder/'exponential_metadata_v1.tsv'
with exp_path.open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(exp_rows[0]),delimiter='\t');writer.writeheader();writer.writerows(exp_rows)
summary_path=folder/'exponential_summary_v1.json';summary_path.write_text(json.dumps(exp_summary,indent=2,allow_nan=False)+'\n')
record={'runtime':runtime,'frozen_result_binding_sha256':file_hash(args.output/'result_bindings.json'),
    'scope':'saved scalar serialization and saved-exponential metadata extrema only; no propagation, eigensolver or observable reevaluation',
    'code_sha256':file_hash(__file__),'outputs':{path.name:file_hash(path) for path in (initial_path,exp_path,summary_path)}}
(folder/'context_reduction_bindings.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'initial_rows':len(initial),'saved_exponential_rows':len(exp_rows)}))
