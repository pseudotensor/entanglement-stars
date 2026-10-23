"""Literal saved-field export only; no arithmetic, kernels or aggregation."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'inventory_result.json'
PIN='a487afb751090d7066ac5ed48f3306ae9f1bde5adff0a9b7e072eabe604b74ce'
OUTPUTS=(ROOT/'inventory_summary.json',ROOT/'selected_rows.tsv')
if any(p.exists() for p in OUTPUTS):raise FileExistsError('Literal exports already exist')
raw=SOURCE.read_bytes()
if hashlib.sha256(raw).hexdigest()!=PIN:raise ValueError('Saved inventory changed')
data=json.loads(raw)
summary={k:v for k,v in data.items() if k!='selected_rows'}
summary['replay_prefix']={k:v for k,v in data['replay_prefix'].items() if k!='samples'}
OUTPUTS[0].write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
fields=('id','parent','bounds','depths','method','real','imaginary','width','joint_reason','joint_ref','creation_ref','completion_ref','partial_rule_status','probe_refs','probes','completed_node_indices','node_refs','accumulator')
with OUTPUTS[1].open('x',newline='') as stream:
 writer=csv.writer(stream,delimiter='\t',lineterminator='\n');writer.writerow(fields)
 for row in data['selected_rows']:
  writer.writerow([json.dumps(row[k],separators=(',',':'),allow_nan=False) if isinstance(row[k],(list,dict)) else str(row[k]) for k in fields])
for path in OUTPUTS:print(json.dumps(dict(path=str(path),size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
