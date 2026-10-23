"""Explicit collar balance from saved own ledger primitives; no first-route inputs."""
import argparse,json,math
from pathlib import Path
from cpu_runtime import require_remote_cpu

LINEAR=('endpoint_change_D','delta_C','current','jumps','vacuum_change','jumps_minus_vacuum',
        'balance_second_term_D','epsilon_tel','endpoint_minus_balance','endpoint_inventory_closure',
        'inventory_identity_residual','closure_evaluation_difference','balance_decomposition_difference')
PRIMITIVE=('endpoint_change_D','delta_C','current','jumps','vacuum_change')
DIAGNOSTICS=('epsilon_tel','endpoint_minus_balance','endpoint_inventory_closure')
PRIMARY=((26,40),(36,50),(46,60),(56,70),(66,80),(76,90),(86,100))
SETS=((1,),(1,2),(1,2,3),tuple(range(1,19)))
UNIT_ROUNDOFF=2.0**-53


def derive(D,C,F,W,P):
    J=math.fsum((W,-P));B=math.fsum((F,W,-P));t=math.fsum((C,-F,-W))
    r=math.fsum((D,-B));c=math.fsum((r,-t));direct=math.fsum((D,-C,P))
    scale=max(1.,math.fsum(abs(x) for x in (D,C,F,W,P)))
    return dict(zip(LINEAR,(D,C,F,W,P,J,B,t,r,c,direct,math.fsum((c,-direct)),
                           math.fsum((B,-math.fsum((F,J))))))),scale


def aggregate(rows):
    out={name:math.fsum(row['weight']*row[name] for row in rows) for name in LINEAR}
    scale=max(1.,math.fsum(row['weight']*math.fsum(abs(row[k]) for k in PRIMITIVE) for row in rows))
    out['arithmetic_scale']=scale;out['arithmetic_comparison_threshold']=64*UNIT_ROUNDOFF*scale
    out['aggregate_balance_evaluation_difference']=math.fsum((out['balance_second_term_D'],
        -math.fsum((out['current'],out['jumps'],-out['vacuum_change']))))
    out['aggregate_closure_evaluation_difference']=math.fsum((out['endpoint_inventory_closure'],
        -math.fsum((out['endpoint_change_D'],-out['delta_C'],out['vacuum_change']))))
    out['weighted_absolute_diagnostics']={k:math.fsum(row['weight']*abs(row[k]) for row in rows) for k in DIAGNOSTICS}
    out['arithmetic_comparisons_pass']=all(abs(out[k])<=out['arithmetic_comparison_threshold'] for k in
        ('aggregate_balance_evaluation_difference','aggregate_closure_evaluation_difference'))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();runtime=require_remote_cpu(a.output)
    if not a.input.resolve().is_relative_to('<internal>/solve_b_bulk') or a.input.name!='ledger.json':
        raise ValueError('Input must be this lane\'s saved bulk ledger.json')
    if a.output.resolve()==a.input.resolve().parent:raise ValueError('Do not write into the original ledger directory')
    a.output.mkdir(parents=True,exist_ok=True)
    def write(name,obj):(a.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    fixtures=[]
    cases=(('zero',0.,0.,0.,0.,0.,(0.,0.,0.,0.,0.)),
           ('positive_telescope',.5,.25,.125,1.,.875,(.125,.625,.25,.25,0.)),
           ('negative_telescope',.5,.25,.125,.5,.375,(.125,.625,-.25,-.25,0.)),
           ('cancelled_jump_signed_balance',-.5,.25,.25,-.25,-.5,(0.,-.5,0.,0.,0.)),
           ('inconsistent_inventory',.5,.25,.125,1.,1.,(.125,.625,.25,.375,.125)))
    expected_names=('jumps_minus_vacuum','balance_second_term_D','epsilon_tel','endpoint_minus_balance','endpoint_inventory_closure')
    for label,F,W,P,C,D,expected_tuple in cases:
        result,scale=derive(D,C,F,W,P);expected=dict(zip(expected_names,expected_tuple))
        passed=all(result[k]==v for k,v in expected.items()) and result['closure_evaluation_difference']==0 and result['balance_decomposition_difference']==0
        fixtures.append({'name':label,'inputs':{'F':F,'W':W,'P':P,'C':C,'D':D},'expected':expected,'result':result,'pass':passed})
        if not passed:raise RuntimeError('Exact scalar fixture failed: '+label)
    first,_=derive(.875,1.,.5,.25,.125);first.update(kappa=1,weight=16)
    second,_=derive(.1875,.25,.25,.125,.0625);second.update(kappa=2,weight=32)
    agg=aggregate([first,second]);expected_agg={'current':16.,'jumps':8.,'vacuum_change':4.,'delta_C':24.,
        'endpoint_change_D':20.,'balance_second_term_D':20.,'jumps_minus_vacuum':4.,
        'epsilon_tel':0.,'endpoint_minus_balance':0.,'endpoint_inventory_closure':0.}
    agg_pass=all(agg[k]==v for k,v in expected_agg.items()) and agg['weighted_absolute_diagnostics']['epsilon_tel']==8.
    if not agg_pass:raise RuntimeError('Exact weighted fixture failed')
    controls={'scalar_fixtures':fixtures,'aggregate_fixture':{'inputs':[first,second],'expected':expected_agg,
        'expected_weighted_absolute_telescope':8.,'result':agg,'pass':agg_pass}}
    write('known_answers.json',controls)
    source=json.loads(a.input.read_text());rows=[];index={}
    for original in source['rows']:
        if original['cut_rs'] not in (16,30):continue
        k=original['kappa'];weight=original['weight']
        if not isinstance(k,int) or not 1<=k<=18 or weight!=16*k:raise ValueError('Wrong channel or weight')
        key=(k,original['clock_start_rs'],original['clock_end_rs'],original['cut_rs'])
        if key in index:raise ValueError('Duplicate saved row '+str(key))
        required=('second_term_D','delta_C','current','jumps','vacuum_change',
                  'inventory_D_start','inventory_D_end','inventory_C_start','inventory_C_end')
        if not all(math.isfinite(original[name]) for name in required) or not all(math.isfinite(x) for x in key):
            raise ValueError('Nonfinite required scalar')
        D,C,F,W,P=(original[name] for name in required[:5]);values,scale=derive(D,C,F,W,P)
        row={'kappa':k,'weight':weight,'clock_start_rs':key[1],'clock_end_rs':key[2],'cut_rs':key[3],
             'source_input':original['input'],**{name:original[name] for name in required[5:]},**values,
             'arithmetic_scale':scale,'arithmetic_comparison_threshold':64*UNIT_ROUNDOFF*scale}
        # These are old-versus-new arithmetic evaluations within this lane only.
        for old,new in (('balance_residual','endpoint_minus_balance'),('telescoping_residual','epsilon_tel')):
            if old in original:
                if not math.isfinite(original[old]):raise ValueError('Nonfinite old arithmetic check')
                row['old_'+old+'_evaluation_difference']=math.fsum((values[new],-original[old]))
        checks=('closure_evaluation_difference','balance_decomposition_difference')+tuple(x for x in row if x.startswith('old_'))
        row['arithmetic_comparisons_pass']=all(abs(row[x])<=row['arithmetic_comparison_threshold'] for x in checks)
        index[key]=row;rows.append(row)
    rows.sort(key=lambda r:(r['kappa'],r['cut_rs'],r['clock_start_rs'],r['clock_end_rs']))
    windows=sorted({key[1:] for key in index});aggregates=[]
    for channels in SETS:
        for start,end,cut in windows:
            subset=[index.get((k,start,end,cut)) for k in channels]
            base={'channels':list(channels),'clock_start_rs':start,'clock_end_rs':end,'cut_rs':cut}
            if any(row is None for row in subset):
                aggregates.append({**base,'status':'missing channel; no numerical aggregate',
                    'missing_channels':[k for k,row in zip(channels,subset) if row is None]});continue
            aggregates.append({**base,'status':'formed',**aggregate(subset)})
    def key_for(row):return {k:row[k] for k in ('kappa','channels','clock_start_rs','clock_end_rs','cut_rs') if k in row}
    def maxima(items,fields,arithmetic=True):
        result={}
        for name in fields:
            available=[r for r in items if name in r]
            if not available:continue
            best=max(available,key=lambda r:abs(r[name]))
            result[name]={'maximum_absolute_value':abs(best[name]),'signed_value':best[name],'key':key_for(best)}
            if arithmetic:result[name]['threshold']=best['arithmetic_comparison_threshold']
        return result
    row_fields=('closure_evaluation_difference','balance_decomposition_difference',
                'old_balance_residual_evaluation_difference','old_telescoping_residual_evaluation_difference')
    agg_fields=('aggregate_balance_evaluation_difference','aggregate_closure_evaluation_difference')
    summary={'unweighted_arithmetic_maxima':maxima(rows,row_fields),'weighted_aggregate_arithmetic_maxima':maxima(aggregates,agg_fields),
             'unweighted_measured_diagnostic_maxima':maxima(rows,DIAGNOSTICS,False),
             'weighted_aggregate_measured_diagnostic_maxima':maxima(aggregates,DIAGNOSTICS,False),
             'unweighted_failed_arithmetic_keys':[key_for(r) for r in rows if not r['arithmetic_comparisons_pass']],
             'weighted_failed_arithmetic_keys':[key_for(r) for r in aggregates if r.get('status')=='formed' and not r['arithmetic_comparisons_pass']]}
    result={'scope':'Controlled arithmetic of saved own primitives; no regenerated traces/currents, covariance or beta validation, or first-route comparison.',
        'source':str(a.input.resolve()),'runtime':runtime,'schema_definitions':{
            'endpoint_change_D':'Legacy own second_term_D: endpoint inventory change, unaltered.',
            'balance_second_term_D':'fsum(current,jumps,-vacuum_change)',
            'epsilon_tel':'fsum(delta_C,-current,-jumps)',
            'endpoint_minus_balance':'endpoint_change_D-balance_second_term_D',
            'endpoint_inventory_closure':'endpoint_minus_balance-epsilon_tel; measured, never imposed as zero',
            'arithmetic_comparison_threshold':'64*2^-53*declared primitive scale; not a physical error certificate'},
        'known_answers':controls,'rows':rows,'aggregates':aggregates,'summary':summary}
    write('balance_readout.json',result)
    lines=['# Explicit collar balance from saved own primitives','',result['scope'],'',
        'Individual rows are unweighted; aggregates use 16 kappa once. Telescope and inventory closure are measured signed diagnostics, not arithmetic tolerances.','',
        '| Channels | Cut/rs | Window | Balance second term B_D | Endpoint change Delta E_D | Current F | Jumps minus vacuum J | Telescope t | Endpoint minus balance r | Inventory closure c |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    fields=('balance_second_term_D','endpoint_change_D','current','jumps_minus_vacuum','epsilon_tel','endpoint_minus_balance','endpoint_inventory_closure')
    for row in aggregates:
        if (row['clock_start_rs'],row['clock_end_rs']) not in PRIMARY:continue
        values=[','.join(map(str,row['channels'])),str(row['cut_rs']),f"{row['clock_start_rs']:g} to {row['clock_end_rs']:g}"]
        values += [format(row[name],'.17g') for name in fields] if row['status']=='formed' else ['unavailable']*len(fields)
        lines.append('| '+' | '.join(values)+' |')
    lines += ['','Arithmetic and measured-diagnostic maxima, with responsible keys, are in balance_readout.json.','',
              f"Unweighted arithmetic failures: {len(summary['unweighted_failed_arithmetic_keys'])}; weighted failures: {len(summary['weighted_failed_arithmetic_keys'])}."]
    (a.output/'balance_readout.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'saved':str(a.output/'balance_readout.json'),'rows':len(rows),'aggregates':len(aggregates),
                     'arithmetic_failure_count':len(summary['unweighted_failed_arithmetic_keys'])+len(summary['weighted_failed_arithmetic_keys'])}),flush=True)


if __name__=='__main__':main()
