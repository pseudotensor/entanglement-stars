#!/usr/bin/env python3
"""Only add/difference saved X1 quadrature aggregates; no solver/imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(rb, criterion, convention, cap):
    a=next(x for x in rb['aggregates'] if x['criterion']==criterion and x['convention']==convention)
    c=next(x for x in a['cumulative'] if x['k_cap']==cap)
    if 'complete' in c and not c['complete']:
        raise ValueError('incomplete saved cumulative cannot be reported as full')
    return c['S_N_rsE']


def difference(a,b):
    return [a[j]-b[j] for j in range(3)]


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('preserve saved readout')
    names=['scalar_and_inherited','IR_pilot8','IR_refined12']
    receipts=['execution','IR_pilot8_execution','IR_refined12_execution']
    inputs=[];data=[];elapsed=[]
    for name,receipt in zip(names,receipts):
        path=args.directory/(name+'.json');rp=args.directory/(receipt+'.json')
        d=json.loads(path.read_text());e=json.loads(rp.read_text())
        assert d['complete'] and e['actual_outer_returncode']==0
        assert sha(path)==e['output_sha256'] and d['source_sha256']==e['code_sha256']
        inputs.append({'path':str(path),'sha256':sha(path),'execution_path':str(rp),'execution_sha256':sha(rp)})
        data.append(d);elapsed.append(e['outer_wall_seconds'])
    scalar,pilot,refined=data
    assert pilot['phase']=='pilot8' and pilot['nx']==8
    assert refined['phase']=='refined12' and refined['nx']==12
    assert pilot['source_sha256']==refined['source_sha256']
    assert refined['previous']['sha256']==inputs[1]['sha256']
    out={'task':'X1 saved-row finite-domain content combination',
        'source_sha256':sha(Path(__file__)),'inputs':inputs,
        'scope':'Binary64 saved-aggregate arithmetic only; empirical mesh/endpoint changes, no absolute transmission or angular-tail certificate.',
        'combined':[],'ir_mesh_changes':[],'ir_endpoint_changes':[],
        'frequency_tail_allowances':scalar['content']['frequency_tail_allowances'],
        'prior_whole_outer_wall_seconds':sum(elapsed),
        'prior_ode_solves':pilot['resources']['new_ode_solves_started']+refined['resources']['new_ode_solves_started'],
        'prior_rhs':pilot['resources']['new_rhs_evaluations']+refined['resources']['new_rhs_evaluations']}
    for criterion in ('0.95','0.80','0.50'):
        for convention in ('gap','lapse'):
            for cap in (1,2,4,6,8):
                for i,setting in enumerate(('first','refined')):
                    high=vector(scalar['content']['readbacks'][i],criterion,convention,cap)
                    ir8=vector(pilot['readbacks'][i],criterion,convention,cap)
                    ir12=vector(refined['readbacks'][i],criterion,convention,cap)
                    common={'criterion':criterion,'convention':convention,'k_cap':cap,'setting':setting}
                    out['combined'].append({**common,'inherited_high_S_N_rsE':high,'IR12_S_N_rsE':ir12,
                        'total_S_N_rsE':[high[j]+ir12[j] for j in range(3)]})
                    out['ir_mesh_changes'].append({**common,'IR12_minus_IR8_S_N_rsE':difference(ir12,ir8)})
                for d,nx in ((pilot,8),(refined,12)):
                    a=vector(d['readbacks'][0],criterion,convention,cap)
                    b=vector(d['readbacks'][1],criterion,convention,cap)
                    out['ir_endpoint_changes'].append({'criterion':criterion,'convention':convention,'k_cap':cap,
                        'nx':nx,'refined_minus_first_S_N_rsE':difference(b,a)})
    out['complete']=True
    out['readout_wall_seconds_before_save']=time.monotonic()-started
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    lines=['# X1 combined saved-row content','','Finite k<=8, q in [1e-8,3], size64. '+out['scope'],
        'The separate inherited and IR12 components remain in the JSON. The below1e-8/above3 allowances are conditional finite-k bounds; angular channels above8 remain unresolved.','',
        '| criterion | endpoint | k cap | setting | S | N | rs E |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for x in out['combined']:
        lines.append('| '+' | '.join([x['criterion'],x['convention'],str(x['k_cap']),x['setting'],
                                     *[format(v,'.12g') for v in x['total_S_N_rsE']]])+' |')
    lines+=['','## IR12 minus IR8 at a fixed endpoint setting','',
        '| criterion | endpoint | k cap | setting | delta S | delta N | delta rs E |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for x in out['ir_mesh_changes']:
        lines.append('| '+' | '.join([x['criterion'],x['convention'],str(x['k_cap']),x['setting'],
                                     *[format(v,'.12g') for v in x['IR12_minus_IR8_S_N_rsE']]])+' |')
    lines+=['','## Refined minus first endpoints on a fixed IR mesh','',
        '| criterion | endpoint | k cap | nodes | delta S | delta N | delta rs E |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for x in out['ir_endpoint_changes']:
        lines.append('| '+' | '.join([x['criterion'],x['convention'],str(x['k_cap']),str(x['nx']),
                                     *[format(v,'.12g') for v in x['refined_minus_first_S_N_rsE']]])+' |')
    lines+=['','Every input and actual execution receipt is identified in the JSON. No solved-history entropy, new ODE, independent numerical route or error enclosure is generated by this readout.','']
    args.report.write_text('\n'.join(lines))
    print(json.dumps({'complete':True,'output_sha256':sha(args.output),'prior_whole_outer_wall_seconds':out['prior_whole_outer_wall_seconds'],
        'prior_ode_solves':out['prior_ode_solves'],'prior_rhs':out['prior_rhs']}))


if __name__=='__main__':
    main()
