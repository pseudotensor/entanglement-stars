#!/usr/bin/env python3
"""Render existing manufactured-control records; no numerical recomputation."""
import argparse
from collections import Counter
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',required=True);p.add_argument('--log',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.run);out=Path(a.output)
if out.exists(): raise FileExistsError('Refusing to overwrite report')
intake=json.loads((root/'intake.json').read_text())
rows=[json.loads(s) for s in (root/'progress.ndjson').read_text().splitlines()]
passed=sum(r['pass'] for r in rows)
lines=['# Partial manufactured attempt: retained-record report','',
       f'{passed}/{len(rows)} recorded assertions passed; attempt incomplete, not an overall pass.',
       'The process exited with a NumPy integer JSON-serialization exception in return_segment/_proof_digest.',
       'The final controls.json, written_summary.md and first-size cost dictionary were not written.',
       'Measured first-size timing categories and process peak memory are unavailable. No values are reconstructed as timings.',
       'Ordinary and registered clock vectors, independent orbitals/wedges, scalar errors, numerical action budgets and operator eigensystems remain on this box.',
       'Registered benchmark decoupling/entropy dictionaries and all cost phase timings were in memory; only their explicitly journaled assertions survived. No numerical readout is recomputed by this formatter.',
       'The second cost size was not started. No axial mock was run.','',
       '## Intake/source/runtime','',json.dumps(intake,indent=2),'',
       '## Last retained assertion','',rows[-1]['name'] if rows else 'none','',
       '## Recorded families','', '| Family | Rows | Passed |','|---|---:|---:|']
families=Counter();good=Counter()
for r in rows:
    words=r['name'].split('_');name='_'.join(words[:2]) if words[0]=='cost' else words[0]
    families[name]+=1;good[name]+=bool(r['pass'])
for name in sorted(families): lines.append(f'| {name} | {families[name]} | {good[name]} |')
lines+=['','## Complete assertion report','',
        '| Assertion | Pass | Value/error | Fixed limit | Retained scalar diagnostic |',
        '|---|---|---:|---:|---|']
keys=('value','lower','upper','state_error','normalization_error','numerical_margin','propagation_margin',
      'negative_mass','eta','eigen_residual','dimension','trace','hermiticity','status','conditional_upper',
      'cumulative_upper','numerical_consistency','disagreement_upper','passes_phase')
for r in rows:
    d=r.get('detail');small={k:d[k] for k in keys if isinstance(d,dict) and k in d}
    if isinstance(d,dict) and 'comparison' in d:
        c=d['comparison'];small['comparison']={k:c[k] for k in
            ('B','B_previous','computed_trace_distance_bound','primary_error','protected_error') if k in c}
    if r.get('rejection') is not None: small['expected_rejection']=r['rejection']
    lines.append('| '+r['name']+' | '+str(r['pass'])+' | '+str(r.get('value'))+' | '+str(r.get('limit'))+' | '+
                 json.dumps(small,separators=(',',':')).replace('|','/')+' |')
lines+=['','## Retained numerical action budget summaries','',
        '| Action | Time | Input | Defect | Residual | Assembly | Coefficient | Phase | Phase multiply | Final | Total |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
operators=actions=0
with (root/'numerical_actions.ndjson').open() as f:
    for line in f:
        r=json.loads(line)
        if r['kind']=='operator': operators+=1;continue
        actions+=1;b=r['budget']
        order=('represented_time','input','spectral_defect','eigen_residual','operator_assembly',
               'coefficient_action','phase_approximation','phase_multiplication','final_action','total')
        lines.append('| '+str(actions)+' | '+' | '.join(str(b[k]) for k in order)+' |')
lines+=['',f'Retained {operators} operator catalog entries and {actions} action records. Their vectors, matrices and phase-component lists remain in numerical_actions.ndjson; only scalar summaries appear here.',
        '', '## Technical exception boundary','', '```text']
lines+=Path(a.log).read_text().splitlines()[-32:]
lines+=['```','', 'This is a formatting-only read of the original retained records; no solver, propagator, entropy or new control was executed.']
with out.open('x') as f: f.write('\n'.join(lines)+'\n')
print(f'Rendered {len(rows)} assertions, {operators} operators and {actions} actions to {out}')
