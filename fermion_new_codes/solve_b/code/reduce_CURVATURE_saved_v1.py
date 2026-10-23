"""Compact complete reduction of fixed curvature sensitivities and saved controls."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from curvature_core_v1 import maximum, ordered_grid
from run_CURVATURE_secondroute_v1 import COMPONENTS, EXPECTED, VARIANTS, write
from compact_CURVATURE_accounting_v1 import encode

HASHES={}
FIELDS=COMPONENTS+['gK','gH','gA']
PRODUCTS=['B_r_product_discrepancy','C_r_product_discrepancy','B_t_product_discrepancy','Y_product_discrepancy']


def binding(path):
    path=Path(path)
    if str(path) not in HASHES:
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
        HASHES[str(path)]=h.hexdigest()
    return {'path':str(path),'sha256':HASHES[str(path)]}


def groups(locations,signed=None):
    """Lossless equal-radius membership groups, preserving signed values if supplied."""
    result={}
    for clock,ids in locations:
        ids=list(map(int,ids));item={'radius_indices':ids}
        if signed is not None:
            values=np.asarray(signed[clock,ids],dtype=float)
            if len(values) and np.all(values==values[0]):item['constant_signed_value']=float(values[0])
            else:item['signed_values']=values.tolist()
        key=json.dumps(item,sort_keys=True)
        if key not in result:result[key]={**item,'clock_indices':[],'points_per_clock':len(ids)}
        result[key]['clock_indices'].append(int(clock))
    return list(result.values())


def peak_summary(peaks,selected,signed=None):
    if len(peaks)!=len(selected):raise ValueError('peak/selection length mismatch')
    ids=[i for i,v in enumerate(selected) if v]
    available=[i for i in ids if peaks[i]['value'] is not None]
    requested=sum(peaks[i]['requested_points'] for i in ids)
    evaluated=sum(peaks[i]['available_points'] for i in ids)
    largest=max((peaks[i]['value'] for i in available),default=None)
    locators=[(i,peaks[i]['indices']) for i in available if peaks[i]['value']==largest]
    return {'value':largest,'selection_status':'empty_selection' if requested==0 else 'available' if available else 'missing_requested_samples',
            'requested_records':len(ids),'evaluated_records':len(available),'unavailable_records':len(ids)-len(available),
            'empty_selection_records':sum(peaks[i]['requested_points']==0 for i in ids),
            'partially_available_records':sum(bool(peaks[i]['partial']) for i in available),
            'requested_points':requested,'evaluated_points':evaluated,'unavailable_points':requested-evaluated,
            'partial':requested>evaluated,'exact_tie_count':sum(len(j) for i,j in locators),
            'extremal_membership':groups(locators,signed)}


def array_peaks(values,radii,masks=None):
    if masks is None:masks=np.broadcast_to(radii>0,values.shape)
    return [maximum(row,radii,masks[i]) for i,row in enumerate(values)]


def selections(rows):
    return {'all':[True]*len(rows),'mutual_evaluation_prefix':[r['mutual_evaluation_prefix'] is True for r in rows]}


def absent_exact_pairs(clock_count,positive_radius_count,common_clock_count,common_radius_count):
    total=clock_count*positive_radius_count
    joined=common_clock_count*common_radius_count
    if joined>total:raise ValueError('joined domain exceeds source domain')
    return total-joined


def validate_comparison_association(rows,arrays,first,second):
    clocks=ordered_grid(arrays['T'],1);radii=ordered_grid(arrays['R'],1)
    if radii[0]<=0 or not np.array_equal([r['T'] for r in rows],clocks):raise ValueError('comparison row clock mismatch')
    for prefix,source in [('first',first),('second',second)]:
        ti=arrays[prefix+'_record_indices'];ri=arrays[prefix+'_radius_indices']
        if not np.array_equal(source['T'][ti],clocks) or not np.array_equal(source['R'][ri],radii):
            raise ValueError('comparison source index/clock/radius association mismatch')


def validate_site_association(rows,array_clocks,primary):
    clocks=ordered_grid(array_clocks,1)
    if not np.array_equal([r['T'] for r in rows],clocks) or not np.array_equal([r['T'] for r in primary],clocks):
        raise ValueError('site/primary row clock mismatch')


def reduce_comparison(path):
    saved=json.loads(path.read_text());report={'source':binding(path),'label':saved['label']}
    if 'unavailable' in saved:
        report['status']=saved['unavailable'];return report
    rows=saved['rows'];array_path=path.with_name('profile_differences.npz')
    report['arrays_source']=binding(array_path)
    first=Path(saved['first'])/'profiles.npz';second=Path(saved['second'])/'profiles.npz'
    report['first_profile']=binding(first);report['second_profile']=binding(second)
    with np.load(array_path) as arrays,np.load(first) as aa,np.load(second) as bb:
        validate_comparison_association(rows,arrays,aa,bb)
        report['T']=arrays['T'];report['R']=arrays['R']
        report['common_clock_count']=len(arrays['T']);report['common_radius_count']=len(arrays['R'])
        for prefix,source in [('first',aa),('second',bb)]:
            ti=arrays[prefix+'_record_indices'];ri=arrays[prefix+'_radius_indices']
            report[prefix+'_variant_record_indices']=ti
            report[prefix+'_original_record_indices']=source['original_record_indices'][ti]
            report[prefix+'_variant_radius_indices']=ri
            report[prefix+'_original_radius_indices']=source['original_radius_indices'][ri]
            report[prefix+'_full_variant_R']=source['R']
            report[prefix+'_absent_exact_pair_count']=absent_exact_pairs(len(source['T']),int(np.count_nonzero(source['R']>0)),len(arrays['T']),len(arrays['R']))
        report['fields']={}
        for field in FIELDS:
            signed=arrays[field+'_signed_difference']
            result={}
            for population,selected in selections(rows).items():
                ids=[i for i,v in enumerate(selected) if v]
                item={'difference':peak_summary([r['fields'][field]['max_absolute_difference'] for r in rows],selected,signed),
                      'first_common_domain_maximum':peak_summary([r['fields'][field]['first_common_domain_maximum'] for r in rows],selected),
                      'second_common_domain_maximum':peak_summary([r['fields'][field]['second_common_domain_maximum'] for r in rows],selected)}
                for key in ['requested_pair_count','available_pair_count','unavailable_pair_count','available_difference_count','unavailable_difference_count']:
                    item[key]=sum(rows[i]['fields'][field][key] for i in ids)
                if field.startswith('g'):
                    disagreement=arrays[field+'_threshold_disagreement']
                    locations=[(i,np.flatnonzero(disagreement[i])) for i in ids if np.any(disagreement[i])]
                    item['threshold_disagreement_count']=sum(len(j) for i,j in locations)
                    item['threshold_membership']=groups(locations)
                result[population]=item
            report['fields'][field]=result
        strict=arrays['mutual_all_support_restricted'];difference=arrays['frame_norm_absolute_difference']
        report['mutual_all_support_restricted_frame_difference']=peak_summary(array_peaks(difference,arrays['R'],strict),[True]*len(rows),arrays['frame_norm_signed_difference'])
        report['full_variant_sample_maxima_on_joined_clocks']={}
        for prefix in ('first','second'):
            for field in ('sphere','frame'):
                peaks=[r[prefix+'_full_variant_sample_'+field] for r in rows]
                report['full_variant_sample_maxima_on_joined_clocks'][prefix+'_'+field]={population:peak_summary(peaks,selected) for population,selected in selections(rows).items()}
    return report


def validate_profile_association(rows,arrays):
    clocks=ordered_grid(arrays['T'],1)
    if not np.array_equal([r['T'] for r in rows],clocks):raise ValueError('profile row/array clock mismatch')
    if not np.array_equal([r['record_index'] for r in rows],arrays['original_record_indices']):raise ValueError('profile original record-index mismatch')


def reduce_profiles(root):
    reports=[]
    for history in EXPECTED:
        for role in ('metric_nodes','applied_sites'):
            for variant,*_ in VARIANTS:
                folder=root/history/role/variant;record=folder/'records.json';raw=json.loads(record.read_text());rows=raw['rows']
                source=folder/'profiles.npz'
                result={'history':history,'role':role,'variant':variant,'records_source':binding(record),'arrays_source':binding(source),
                        'T':[r['T'] for r in rows],'original_record_indices':[r['record_index'] for r in rows],'products':{},'center':{}}
                populations={'all':[True]*len(rows),'evaluation_pre_gate':[r['evaluation_pre_gate'] is True for r in rows]}
                with np.load(source) as arrays:
                    validate_profile_association(rows,arrays)
                    result['R']=arrays['R'];result['original_radius_indices']=arrays['original_radius_indices']
                    for field in PRODUCTS:
                        signed=arrays[field];peaks=array_peaks(np.abs(signed),arrays['R'])
                        result['products'][field]={name:peak_summary(peaks,selected,signed) for name,selected in populations.items()}
                for center in rows[0]['center']:
                    result['center'][center]={};rr=np.array([0.])
                    center_pops={**populations,'conditional_all_support':[r['center'][center]['conditional_all_support_restricted_frame'] is True for r in rows]}
                    for field in ('V','frame_norm'):
                        values=np.asarray([[np.nan if r['center'][center][field] is None else r['center'][center][field]] for r in rows])
                        peaks=array_peaks(values,rr,np.ones(values.shape,dtype=bool))
                        result['center'][center][field]={name:peak_summary(peaks,selected) for name,selected in center_pops.items()}
                if rows[0]['center']:
                    result['center_two_minus_three_absolute_spreads']={}
                    for field in rows[0]['center_proxy_sensitivity']:
                        if field=='conditional_gate_straddled':
                            result['center_gate_straddle_record_indices']=[i for i,r in enumerate(rows) if r['center_proxy_sensitivity'][field] is True]
                            result['center_gate_straddle_unavailable_record_indices']=[i for i,r in enumerate(rows) if r['center_proxy_sensitivity'][field] is None]
                            continue
                        signed=np.asarray([[np.nan if r['center_proxy_sensitivity'][field] is None else r['center_proxy_sensitivity'][field]] for r in rows])
                        peaks=array_peaks(np.abs(signed),np.array([0.]),np.ones(signed.shape,dtype=bool))
                        result['center_two_minus_three_absolute_spreads'][field]={name:peak_summary(peaks,selected,signed) for name,selected in populations.items()}
                reports.append(result)
                print(json.dumps({'saved_profile_controls':history+'/'+role+'/'+variant}),flush=True)
    return reports


def reduce_sites(root):
    results=[]
    for history in EXPECTED:
        directory=root/history;path=directory/'linear_to_sites_records.json';rows=json.loads(path.read_text())['rows']
        primary=json.loads((directory/'metric_nodes/primary/records.json').read_text())['rows']
        with np.load(directory/'linear_to_sites_sphere.npz') as arrays:
            validate_site_association(rows,arrays['T'],primary)
            radii=arrays['R']
        results.append({'history':history,'source':binding(path),'arrays_source':binding(directory/'linear_to_sites_sphere.npz'),
                        'T':[r['T'] for r in rows],'R':radii,
                        'maxima':{name:peak_summary([r['sphere'] for r in rows],selected) for name,selected in
                                  {'all':[True]*len(rows),'evaluation_pre_gate':[r['evaluation_pre_gate'] is True for r in primary]}.items()}})
    return results


def controls():
    peaks=[maximum([0,0],[1,2]),maximum([0,0],[1,2])]
    zero=peak_summary(peaks,[True,True],np.zeros((2,2)))
    empty=peak_summary(peaks,[False,False])
    missing=peak_summary([maximum([np.nan,2],[1,2])],[True])
    signed=peak_summary([maximum([2,2],[1,2])],[True],np.array([[-2.,2.]]))
    restricted=peak_summary(array_peaks(np.array([[1.,2.],[3.,np.nan]]),np.array([1.,2.]),np.array([[False,True],[False,False]])),[True,True])
    tests={'zero_exact_ties_preserved':zero['value']==0 and zero['exact_tie_count']==4 and zero['extremal_membership']==[{'radius_indices':[0,1],'constant_signed_value':0.,'clock_indices':[0,1],'points_per_clock':2}],
           'empty_is_null_not_zero':empty['value'] is None and empty['selection_status']=='empty_selection' and empty['requested_records']==0,
           'partial_and_unavailable_points':missing['partial'] and missing['requested_points']==2 and missing['evaluated_points']==1,
           'signed_values_at_extrema':signed['extremal_membership'][0]['signed_values']==[-2.,2.],
           'restricted_mask_empty_record':restricted['value']==2 and restricted['empty_selection_records']==1 and restricted['unavailable_points']==0,
           'absent_exact_pairs_separate':absent_exact_pairs(3,4,2,2)==8}
    all_missing=peak_summary([maximum([np.nan,np.nan],[1,2])],[True])
    tests['all_requested_values_missing_is_not_empty']=all_missing['value'] is None and all_missing['selection_status']=='missing_requested_samples' and all_missing['requested_points']==2 and all_missing['unavailable_points']==2
    def rejects(function):
        try:function();return False
        except ValueError:return True
    tests['shifted_site_clock_rejected']=rejects(lambda:validate_site_association([{'T':0},{'T':8}],np.array([0.,8.]),[{'T':0},{'T':9}]))
    tests['missing_site_clock_rejected']=rejects(lambda:validate_site_association([{'T':0}],np.array([0.,8.]),[{'T':0},{'T':8}]))
    tests['selection_length_mismatch_rejected']=rejects(lambda:peak_summary(peaks,[True]))
    fixture_arrays={'T':np.array([0.,8.]),'R':np.array([1.,2.]),'first_record_indices':np.array([0,1]),'second_record_indices':np.array([0,1]),'first_radius_indices':np.array([0,1]),'second_radius_indices':np.array([0,1])}
    fixture_source={'T':np.array([0.,8.]),'R':np.array([1.,2.])}
    tests['shifted_comparison_row_clock_rejected']=rejects(lambda:validate_comparison_association([{'T':0},{'T':9}],fixture_arrays,fixture_source,fixture_source))
    wrong_source={**fixture_source,'R':np.array([1.,3.])}
    tests['wrong_comparison_radius_map_rejected']=rejects(lambda:validate_comparison_association([{'T':0},{'T':8}],fixture_arrays,fixture_source,wrong_source))
    profile_arrays={'T':np.array([0.,8.]),'original_record_indices':np.array([0,4])}
    tests['shifted_profile_clock_rejected']=rejects(lambda:validate_profile_association([{'T':0,'record_index':0},{'T':9,'record_index':4}],profile_arrays))
    tests['shifted_profile_index_rejected']=rejects(lambda:validate_profile_association([{'T':0,'record_index':0},{'T':8,'record_index':5}],profile_arrays))
    if not all(tests.values()):raise RuntimeError('compact saved-reduction control failed')
    return tests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    runtime=require_remote_cpu(args.input);tests=controls()
    comparisons=[]
    paths=sorted((args.input/'sensitivities').rglob('comparison.json'))+sorted((args.input/'cross_histories').rglob('comparison.json'))
    if len(paths)!=99:raise ValueError('expected 91 within-history/role comparisons and eight cross-history comparisons')
    for path in paths:
        comparisons.append(reduce_comparison(path));print(json.dumps({'reduced_comparison':str(path.relative_to(args.input))}),flush=True)
    result={'scope':'Reduction of frozen saved quantities only; differences are sensitivities, not error enclosures. Radius-index memberships refer to the explicitly named common or full-variant radius vector. All ties and signed extremal values are preserved.',
            'runtime':runtime,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'manufactured_controls':tests,
            'comparison_count':len(comparisons),'comparisons':comparisons,'profile_controls':reduce_profiles(args.input),'linear_to_sites':reduce_sites(args.input)}
    # Convert NumPy values before the purely representational lossless index encoding.
    from run_CURVATURE_secondroute_v1 import clean
    write(args.input/'saved_reduction.json',encode(clean(result)))
    print(json.dumps({'reduced_comparisons':len(comparisons),'profile_controls':len(result['profile_controls'])}),flush=True)


if __name__=='__main__':main()
