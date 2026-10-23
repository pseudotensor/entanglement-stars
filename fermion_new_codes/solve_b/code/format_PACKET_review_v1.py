"""Serialize frozen packet reductions as review tables; no numerical recomputation."""
import argparse
import csv
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu
from packet_core_v1 import file_hash

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
runtime=require_remote_cpu(args.output)
freeze=json.loads((args.output/'result_bindings.json').read_text())
if file_hash(args.output/'summary.json')!=freeze['summary_sha256']:
    raise ValueError('frozen reduction changed')
s=json.loads((args.output/'summary.json').read_text())
folder=args.output/'review';folder.mkdir(exist_ok=True)
rows=[];reversals={}
for channel,data in s['channels'].items():
    for mode,values in data['modes'].items():
        for index,column in enumerate(values['columns']):
            endpoint=column['endpoint'];maxima=column['maxima']
            row={'channel':channel,'mode':mode,'convention':column['label'][0],'launch':column['label'][1],'k0':column['label'][2],
                'maximum_record_norm_drift':column['norm_drift']['value'],
                'maximum_substep_norm_drift':column['maximum_substep_norm_drift'],
                'E_frozen_energy_max_defect':column['frozen_energy_max_defect'],'E_work_max_defect':column['work_max_defect'],
                'energy_H_endpoint':endpoint['energy_H'],'energy_0_endpoint':endpoint['energy_0'],
                'dense_outer_endpoint':endpoint['dense_shares'][2],'applied_outer_endpoint':endpoint['applied_shares'][2],
                'fixed_initial_outer_endpoint':endpoint['fixed_outer_share'],
                'numerically_admitted':values['trajectory_admitted'][index],
                'failures':json.dumps(column['failures'],separators=(',',':'))}
            for name in ('energy_H','energy_0','homogeneous_mean_excess','negative_band_weight','high_share','negative_cosine_share','opposed_current_share','wall_64','packet_row_max_excess','packet_cauchy_max_excess','radial_mean_band_excess','radial_second_band_excess','H_mean_band_excess','H_second_band_excess'):
                row[name+'_maximum']=maxima[name]['value']
                row[name+'_max_records']=json.dumps(maxima[name]['record_indices'],separators=(',',':'))
            rows.append(row)
            reversals[channel+'_'+mode+'_'+str(index)]={'label':column['label'],**{key:value for key,value in column.items() if key.endswith('_reversals')}}
with (folder/'trajectory_summary_v1.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
(folder/'reversals_v1.json').write_text(json.dumps(reversals,indent=2,allow_nan=False)+'\n')
selected=('norm','energy_H','energy_0','current_Q','velocity_lapse','velocity_flow','velocity_total','centroid','position_variance','dense_shares','applied_shares','fixed_outer_share','negative_band_weight','high_share','negative_cosine_share','opposed_current_share','mean_N','mean_v','wall_64','inside_12','packet_row_max_excess','packet_cauchy_max_excess')
rows=[]
for tag,comparison in s['comparisons'].items():
    for key,packets in comparison['per_packet_differences'].items():
        if key not in selected:continue
        for index,entry in enumerate(packets):
            rows.append({'comparison':tag,'observable':key,'packet_index':index,**{k:v for k,v in entry.items() if k!='locators'},
                'locators':json.dumps(entry['locators'],separators=(',',':'))})
with (folder/'sensitivities_v1.tsv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
(folder/'distance_summary_v1.json').write_text(json.dumps({tag:{k:v for k,v in x.items() if k!='per_packet_differences'} for tag,x in s['comparisons'].items()},indent=2,allow_nan=False)+'\n')
for channel,data in s['channels'].items():
    (folder/('free_preparation_k'+channel+'_v1.json')).write_text(json.dumps({k:v for k,v in data.items() if k!='modes'},indent=2,allow_nan=False)+'\n')
record={'runtime':runtime,'frozen_summary_sha256':freeze['summary_sha256'],'formatting_code_sha256':file_hash(__file__),
    'scope':'literal serialization of frozen saved values; complete scalar and locator data remain in summary.json',
    'outputs':{path.name:{'sha256':file_hash(path),'bytes':path.stat().st_size} for path in sorted(folder.iterdir()) if path.name!='review_reduction_bindings.json'}}
(folder/'review_reduction_bindings.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'review_tables_saved':True,'trajectories':sum(len(mode['columns']) for x in s['channels'].values() for mode in x['modes'].values())}))
