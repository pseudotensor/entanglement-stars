#!/usr/bin/env python3
"""Literal saved-proof supplement; no numerical imports or recomputation."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--kind',choices=('recovery','cost'),required=True);p.add_argument('--run',required=True)
a=p.parse_args();root=Path(a.run)
if a.kind=='recovery':
    original=root/'written_recovery_summary.md';out=root/'written_recovery_summary_with_proof_v1.md'
    result=json.loads((root/'recovered_readouts.json').read_text())
    returned=[json.loads(x) for x in (root/'recovery.ndjson').read_text().splitlines()]
    returned=[x for x in returned if x.get('role')=='recovered_missing_return_readout']
    if len(returned)!=1 or returned[0]['proof']!=result['return_proof'] or returned[0]['bound']!=result['return_bound']:
        raise ValueError('Saved proof/readout association differs')
    command=result['intake']['command'];source=Path(command[command.index('--run')+1])
    records=[json.loads(x) for x in (source/'numerical_actions.ndjson').read_text().splitlines()]
    operators=[x for x in records if x['kind']=='operator']
    entries=[{'label':'recovered original two-site helper, not a serialized original completion',
        'proof':result['return_proof'],'bound':result['return_bound'],'source_action_ordinals':returned[0]['source_ordinals'],
        'primary_assembly_allowance':operators[10]['assembly_error'],'protected_assembly_allowance':operators[12]['assembly_error'],
        'operator_catalog_ordinals':[10,12]}]
else:
    original=root/'written_summary.md';out=root/'written_summary_with_proof_v1.md'
    result=json.loads((root/'controls.json').read_text())
    records=[json.loads(x) for x in (root/'numerical_actions.ndjson').read_text().splitlines()]
    operators=[x for x in records if x['kind']=='operator'];entries=[]
    for i,c in enumerate(result['costs']):
        entries.append({'label':str(c['sites'])+' sites: '+c['measurement_role'],
            'proof':c['return_protection'],'bound':c['return_comparator'],
            'primary_assembly_allowance':operators[3*i]['assembly_error'],
            'protected_assembly_allowance':operators[3*i+2]['assembly_error'],'operator_catalog_ordinals':[3*i,3*i+2]})
if out.exists():raise FileExistsError('Refusing to overwrite proof report')
lines=[original.read_text().rstrip(),'','## Literal saved return proofs','',
    'This section formats retained dictionaries only. No propagation, density eigensolve, integration or numerical bound is recomputed.']
keys=('digest','experiment','initial_identity','schedule','start_clock','end_clock','duration','old_groups',
      'primary_operator','protected_operator','segment_bound','accumulated_bound','difference_operator_error')
for e in entries:
    proof=e['proof'];bound=e['bound']
    literal={k:proof[k] for k in keys if k in proof}
    literal.update({k:v for k,v in e.items() if k not in ('proof','bound')})
    literal.update({'lipschitz_upper':bound['lipschitz'],'difference_operator_error':bound['difference_operator_error'],
                    'protected_operator_error':bound['protected_operator_error'],'return_upper':bound['upper']})
    lines+=['','### '+e['label'],'','```json',json.dumps(literal,indent=2),'```','',
            '| Saved midpoint | Integrand upper |','|---:|---:|']
    if len(bound['midpoints'])!=16 or len(bound['integrand'])!=16:raise ValueError('Incomplete saved midpoint dictionary')
    lines+=['| '+str(t)+' | '+str(f)+' |' for t,f in zip(bound['midpoints'],bound['integrand'])]
with out.open('x') as f:f.write('\n'.join(lines)+'\n')
print('Wrote literal proof supplement:',out)
