"""Split the fixed saved-reduction JSON into reviewable, lossless datasets."""
import argparse
import hashlib
import json
from pathlib import Path
from cpu_runtime import require_remote_cpu
from run_CURVATURE_secondroute_v1 import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    runtime=require_remote_cpu(args.input.parent);raw=args.input.read_bytes();data=json.loads(raw)
    header={key:value for key,value in data.items() if key not in ('comparisons','profile_controls','linear_to_sites')}
    header.update(full_source=str(args.input),full_source_sha256=hashlib.sha256(raw).hexdigest(),formatting_runtime=runtime)
    folder=args.input.parent/'compact_saved';folder.mkdir(exist_ok=True)
    groups={}
    for comparison in data['comparisons']:
        name='cross_histories' if '__' in comparison['label'] else comparison['label'].split('/')[0]
        groups.setdefault(name,[]).append(comparison)
    for name,rows in groups.items():write(folder/('comparisons_'+name+'.json'),{**header,'comparisons':rows})
    for key in ('profile_controls','linear_to_sites'):write(folder/(key+'.json'),{**header,key:data[key]})
    # Exact parse equality verifies every original data item and metadata field.
    restored=[]
    for name in groups:restored.extend(json.loads((folder/('comparisons_'+name+'.json')).read_text())['comparisons'])
    if sorted(restored,key=lambda r:r['label'])!=sorted(data['comparisons'],key=lambda r:r['label']):raise ValueError('split comparisons changed data')
    for key in ('profile_controls','linear_to_sites'):
        if json.loads((folder/(key+'.json')).read_text())[key]!=data[key]:raise ValueError('split dataset changed data')
    print(json.dumps({'comparison_datasets':len(groups),'all_split_values_equal':True}),flush=True)


if __name__=='__main__':main()
