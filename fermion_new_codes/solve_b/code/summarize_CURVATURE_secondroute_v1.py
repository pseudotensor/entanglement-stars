"""Saved-array curvature sensitivities and own-history comparisons; no derivatives."""
import argparse
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from curvature_core_v1 import maximum, ordered_grid
from run_CURVATURE_secondroute_v1 import COMPONENTS, EXPECTED, VARIANTS, write


def exact_join(t1,r1,t2,r2):
    t1,r1,t2,r2=[ordered_grid(x,1) for x in (t1,r1,t2,r2)]
    times,ta,tb=np.intersect1d(t1,t2,return_indices=True)
    radii,ra,rb=np.intersect1d(r1,r2,return_indices=True)
    positive=radii>0
    return times,ta,tb,radii[positive],ra[positive],rb[positive]


def comparison_fields(first,second,threshold=None):
    """Production pair validity, signed sensitivity and optional gate disagreement."""
    first,second=np.broadcast_arrays(first,second)
    paired=np.isfinite(first)&np.isfinite(second)
    with np.errstate(over='ignore',invalid='ignore'):
        raw=second-first
    valid=paired&np.isfinite(raw)
    delta=np.where(valid,raw,np.nan)
    result={'signed_difference':delta,'absolute_difference':np.abs(delta),
            'available_pair':paired,'available_difference':valid}
    if threshold is not None:
        result['threshold_disagreement']=paired&((first<=threshold)!=(second<=threshold))
    return result


def compare(first,second,destination,label):
    destination.mkdir(parents=True,exist_ok=True)
    one=json.loads((first/'records.json').read_text());two=json.loads((second/'records.json').read_text())
    with np.load(first/'profiles.npz') as a,np.load(second/'profiles.npz') as b:
        times,ta,tb,radii,ra,rb=exact_join(a['T'],a['R'],b['T'],b['R'])
        if not len(times) or not len(radii):
            write(destination/'comparison.json',{'label':label,'unavailable':'no exact common positive grid and clock'});return
        saved={'T':times,'R':radii,'first_record_indices':ta,'second_record_indices':tb,
               'first_radius_indices':ra,'second_radius_indices':rb}
        per_record=[{'T':float(t),'T_over_r0':float(t/64),'fields':{}} for t in times]
        for i,row in enumerate(per_record):
            aa,bb=one['rows'][int(ta[i])],two['rows'][int(tb[i])]
            row['first_evaluation_pre_gate']=aa['evaluation_pre_gate']
            row['second_evaluation_pre_gate']=bb['evaluation_pre_gate']
            row['first_solver_prefix']=aa['solver']['prefix'];row['second_solver_prefix']=bb['solver']['prefix']
            row['mutual_evaluation_prefix']=all(value is True for value in [aa['evaluation_pre_gate'],bb['evaluation_pre_gate'],aa['solver']['prefix'],bb['solver']['prefix']])
            row['first_full_variant_sample_sphere']=aa['sphere'];row['second_full_variant_sample_sphere']=bb['sphere']
            row['first_full_variant_sample_frame']=aa['frame'];row['second_full_variant_sample_frame']=bb['frame']
        for field in COMPONENTS+['gK','gH','gA']:
            aa=a[field][np.ix_(ta,ra)];bb=b[field][np.ix_(tb,rb)]
            compared=comparison_fields(aa,bb,.1 if field in ('gK','gH','gA') else None)
            for key,value in compared.items():saved[field+'_'+key]=value
            delta=compared['signed_difference'];paired=compared['available_pair']
            difference_valid=compared['available_difference']
            for i,row in enumerate(per_record):
                row['fields'][field]={'max_absolute_difference':maximum(np.abs(delta[i]),radii),
                                     'first_common_domain_maximum':maximum(aa[i],radii),
                                     'second_common_domain_maximum':maximum(bb[i],radii),
                                     'requested_pair_count':len(radii),'available_pair_count':int(np.count_nonzero(paired[i])),
                                     'unavailable_pair_count':int(np.count_nonzero(~paired[i])),
                                     'available_difference_count':int(np.count_nonzero(difference_valid[i])),
                                     'unavailable_difference_count':int(np.count_nonzero(~difference_valid[i]))}
                if field in ('gK','gH','gA'):
                    disagree=compared['threshold_disagreement'][i]
                    row['fields'][field]['threshold_disagreement_count']=int(np.count_nonzero(disagree))
                    row['fields'][field]['threshold_disagreement_radii']=radii[disagree].tolist()
            del aa,bb,delta
        strict=a['frame_all_support_restricted'][np.ix_(ta,ra)]&b['frame_all_support_restricted'][np.ix_(tb,rb)]
        saved['mutual_all_support_restricted']=strict
        differences=saved['frame_norm_absolute_difference']
        for i,row in enumerate(per_record):
            row['all_support_restricted_frame_difference']=maximum(differences[i],radii,strict[i])
        np.savez(destination/'profile_differences.npz',**saved)
    write(destination/'comparison.json',{'scope':'Exact common own-history clocks/radii; second minus first. Finite differences are sensitivities, not enclosures.',
                                         'label':label,'first':str(first),'second':str(second),
                                         'first_metadata':one['metadata'],'second_metadata':two['metadata'],
                                         'common_clock_count':len(times),'common_radius_count':len(radii),
                                         'rows':per_record})


def global_maximum(rows,field,selected=None):
    if selected is None:selected=[True]*len(rows)
    available=[(i,row[field]['value']) for i,row in enumerate(rows) if selected[i] and row[field]['value'] is not None]
    requested=int(np.count_nonzero(selected));evaluated=len(available)
    partial=sum(bool(rows[i][field]['partial']) for i,value in available)
    accounting={'selected_records':requested,'evaluated_records':evaluated,
                'unavailable_records':requested-evaluated,'partially_available_records':partial,
                'partial':bool(requested>evaluated or partial)}
    if not available:return {'value':None,'T':[],'radii':[],**accounting}
    largest=max(value for i,value in available);ids=[i for i,value in available if value==largest]
    return {'value':largest,'T':[rows[i]['T'] for i in ids],
            'radii':[rows[i][field]['radii'] for i in ids],**accounting}


def summarize_history(root,name):
    directory=root/name
    source=json.loads((directory/'metric_nodes/primary/records.json').read_text())
    rows=source['rows'];positive_prefix=[row['evaluation_pre_gate'] is True for row in rows]
    post=[row['evaluation_pre_gate'] is False for row in rows]
    first=next((i for i,row in enumerate(rows) if row['evaluation_pre_gate'] is False),None)
    local=json.loads((directory/'gate_prefix.json').read_text())['local']
    summary={'history':name,'metadata':source['metadata'],'records':len(rows),
             'first_gate_failure':None if first is None else {'T':rows[first]['T'],'previous_T':rows[first-1]['T'] if first else None,
                     'previous_local_gate':local[first-1] if first else None,
                     'previous_prefix':rows[first-1]['evaluation_pre_gate'] if first else None,
                     'bracket_status':'initial_observed_failure' if first==0 else 'passing_prefix_to_failure' if all(v is True for v in local[:first]) else 'unresolved_prior_prefix'},
             'pre_gate_record_count':sum(positive_prefix),'post_gate_record_count':sum(post),
             'unresolved_gate_record_count':sum(row['evaluation_pre_gate'] is None for row in rows)}
    for prefix,selection in [('all',[True]*len(rows)),('pre_gate_evaluation',positive_prefix),('post_gate_evaluation',post)]:
        for field in ('sphere','frame','centered_radial_and_time_frame','all_support_restricted_frame','all_support_restricted_sphere'):
            summary[prefix+'_'+field]=global_maximum(rows,field,selection)
    with np.load(directory/'linear_to_sites_sphere.npz') as site:
        sampling=[]
        for i,row in enumerate(rows):
            site_peak=maximum(site['V'][i],site['R'])
            sampling.append({'T':row['T'],'node_sphere':row['sphere'],'site_sphere':site_peak,
                             'site_minus_node_maximum':None if site_peak['value'] is None or row['sphere']['value'] is None else site_peak['value']-row['sphere']['value'],
                             'center_estimates':row['center']})
    write(directory/'sphere_sampling_sensitivity.json',{'rows':sampling})
    write(directory/'summary.json',summary)
    return summary


def near_ties(root,name,role):
    comparisons=[]
    for variant,*_ in VARIANTS[1:]:
        item=json.loads((root/'sensitivities'/name/role/variant/'comparison.json').read_text())
        comparisons.append({row['T']:row for row in item.get('rows',[])})
    rows=[]
    with np.load(root/name/role/'primary/profiles.npz') as baseline:
        radii=baseline['R'];positive=radii>0
        for field,peak_name in [('V','sphere'),('frame_norm','frame')]:
            values=baseline[field]
            for i,clock in enumerate(baseline['T']):
                peak=maximum(values[i],radii,positive);spreads=[]
                for comparison in comparisons:
                    if float(clock) not in comparison:continue
                    row=comparison[float(clock)]
                    difference=row['fields'][field]['max_absolute_difference']['value']
                    if difference is not None:spreads.append(difference)
                    a=row['first_full_variant_sample_'+peak_name]['value'];b=row['second_full_variant_sample_'+peak_name]['value']
                    if a is not None and b is not None:spreads.append(abs(b-a))
                spread=max(spreads) if spreads else None
                if spread is None or peak['value'] is None:ids=np.array([],dtype=int)
                else:ids=np.flatnonzero(positive&np.isfinite(values[i])&(peak['value']-values[i]<=spread))
                rows.append({'T':float(clock),'field':field,'maximum':peak,'observed_sensitivity_scale':spread,
                             'candidate_indices':ids.tolist(),'candidate_radii':radii[ids].tolist()})
    write(root/name/role/'near_ties.json',{'scope':'Descriptive near-tie set within the largest available prescribed profile/maximum sensitivity, not an error enclosure.', 'rows':rows})


def gate_sensitivity(root,name,role):
    with np.load(root/name/role/'primary/profiles.npz') as baseline:
        times,radii=baseline['T'],baseline['R']
    saved={'T':times,'R':radii};rows=[]
    for gate in ('gK','gH','gA'):
        requested=np.zeros((len(times),len(radii)),dtype=np.uint8)
        available=requested.copy();straddled=np.zeros_like(requested,dtype=bool)
        for variant,*_ in VARIANTS[1:]:
            with np.load(root/'sensitivities'/name/role/variant/'profile_differences.npz') as comparison:
                ids=np.ix_(comparison['first_record_indices'],comparison['first_radius_indices'])
                requested[ids]+=1
                available[ids]+=comparison[gate+'_available_pair']
                straddled[ids]|=comparison[gate+'_threshold_disagreement']
        saved[gate+'_requested_comparisons']=requested;saved[gate+'_available_comparisons']=available
        saved[gate+'_unavailable_comparisons']=requested-available
        saved[gate+'_variants_without_exact_pair']=len(VARIANTS)-1-requested
        saved[gate+'_stencil_sensitive']=straddled
        for i,t in enumerate(times):
            positive=radii>0;ids=np.flatnonzero(positive&straddled[i])
            rows.append({'T':float(t),'gate':gate,'stencil_sensitive_radii':radii[ids].tolist(),
                         'requested_pairs':int(np.sum(requested[i,positive])),
                         'available_pairs':int(np.sum(available[i,positive])),
                         'unavailable_pairs':int(np.sum((requested-available)[i,positive])),
                         'variants_without_exact_pair':int(np.sum((len(VARIANTS)-1-requested)[i,positive]))})
    np.savez(root/name/role/'gate_sensitivity.npz',**saved)
    write(root/name/role/'gate_sensitivity.json',{'scope':'Observed threshold straddles; unavailable or absent exact pairs are not agreement and no robustness certificate is inferred.','rows':rows})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    runtime=require_remote_cpu(args.input)
    summaries=[]
    for name in EXPECTED:
        summaries.append(summarize_history(args.input,name))
        for role in ('metric_nodes','applied_sites'):
            baseline=args.input/name/role/'primary'
            for variant,*_ in VARIANTS[1:]:
                compare(baseline,args.input/name/role/variant,args.input/'sensitivities'/name/role/variant,
                        name+'/'+role+'/'+variant)
        compare(args.input/name/'metric_nodes/primary',args.input/name/'applied_sites/primary',
                args.input/'sensitivities'/name/'applied_vs_metric',name+'/applied_minus_metric')
        for role in ('metric_nodes','applied_sites'):
            near_ties(args.input,name,role);gate_sensitivity(args.input,name,role)
        print(json.dumps({'saved_history_sensitivities':name}),flush=True)
    for first,second in [('prod_r11','B2'),('prod_r11','P64_s4'),('prod_r11','P64_s4_tau4'),
                         ('prod_r11','P64_s4_tau2'),('P64_s4','P64_s4_tau4'),('P64_s4','P64_s4_tau2'),
                         ('P64_s4','P64_s2'),('P64_s4','P64_s8')]:
        label=first+'__'+second
        compare(args.input/first/'metric_nodes/primary',args.input/second/'metric_nodes/primary',
                args.input/'cross_histories'/label,label)
        print(json.dumps({'saved_cross_history':label}),flush=True)
    write(args.input/'summaries.json',{'runtime':runtime,'rows':summaries,
                                     'native_per_site_covariance_check':'unavailable',
                                     'scope':'Own sampled metric diagnostics and prescribed sensitivities; no first-route comparison or global singularity conclusion.'})


if __name__=='__main__':main()
