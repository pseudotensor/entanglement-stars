"""Hash saved scientific results before the authorized first-route comparison."""
import argparse
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu
from packet_core_v1 import file_hash

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
runtime=require_remote_cpu(args.output)
for channel in (1,5):
    for name in ('producer_complete.json','readout_complete.json'):
        if not (args.output/f'k{channel}'/name).is_file():
            raise ValueError('incomplete scientific output')
summary=args.output/'summary.json'
if not summary.is_file():
    raise ValueError('missing saved reduction')
destination=args.output/'result_bindings.json'
if destination.exists():
    raise ValueError('existing frozen result identities; no overwrite')
files=[]
for path in sorted(args.output.rglob('*')):
    if path.is_file() and path != destination and '__pycache__' not in path.parts:
        files.append({'path':str(path.relative_to(args.output)), 'bytes':path.stat().st_size,'sha256':file_hash(path)})
record={'runtime':runtime,'scope':'scientific saved-output identities before source section 0 comparison; no new propagation or scalar recomputation',
    'summary_sha256':file_hash(summary),'first_route_section0_intake':'not performed',
    'independence_disclosure':'source_input_contract.md; Sol/reviewer incidental section1/2/4 exposure is not erased',
    'files':files,'file_count':len(files),'bytes_excluding_this_binding':sum(row['bytes'] for row in files),
    'binding_code_sha256':file_hash(__file__)}
destination.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'saved_results_bound':True,'files':len(files),'summary_sha256':record['summary_sha256']}))
