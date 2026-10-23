"""Compact complete accounting of frozen curvature JSON and NPZ headers only."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from cpu_runtime import require_remote_cpu
from run_CURVATURE_secondroute_v1 import EXPECTED, VARIANTS, write
from summarize_CURVATURE_secondroute_v1 import global_maximum


def membership(values):
    return {name:[i for i,value in enumerate(values) if value is target]
            for name,target in [('true',True),('false',False),('unavailable',None)]}


def finite_membership(values):
    return {'finite':[i for i,value in enumerate(values) if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value)],
            'unavailable':[i for i,value in enumerate(values) if not isinstance(value,(int,float)) or isinstance(value,bool) or not np.isfinite(value)]}


def audit_rows(rows):
    report={'records':len(rows),'evaluation_pre_gate':membership([r['evaluation_pre_gate'] for r in rows]),
            'solver_prefix':membership([r['solver']['prefix'] for r in rows]),'invalid_locations':{},'center':{}}
    for key in rows[0]['unavailable_or_invalid_locations']:
        entries=[r['unavailable_or_invalid_locations'][key] for r in rows]
        report['invalid_locations'][key]={'record_indices':[i for i,value in enumerate(entries) if value],
                                          'total_locations':sum(map(len,entries))}
    for label,field in [('sphere','sphere'),('frame','frame'),('strict_sphere','all_support_restricted_sphere'),
                        ('strict_frame','all_support_restricted_frame')]:
        records=[r[field] for r in rows]
        report[label+'_availability']={
            'fully_available':[i for i,r in enumerate(records) if r['value'] is not None and not r['partial']],
            'partial':[i for i,r in enumerate(records) if r['value'] is not None and r['partial']],
            'unavailable':[i for i,r in enumerate(records) if r['value'] is None],
            'empty_restricted_selection':[i for i,r in enumerate(records) if r['requested_points']==0],
            'missing_requested_physical_samples':[i for i,r in enumerate(records) if r['value'] is None and r['requested_points']>0]}
        for name,selected in [('all',[True]*len(rows)),('pre',[r['evaluation_pre_gate'] is True for r in rows]),
                              ('post',[r['evaluation_pre_gate'] is False for r in rows])]:
            report[name+'_'+label+'_maximum']=global_maximum(rows,field,selected)
    for center in rows[0]['center']:
        available=[r['center'][center] for r in rows]
        flags=[key for key,value in available[0].items() if isinstance(value,bool) or value is None and key.endswith('maximum')]
        report['center'][center]={'flags':{key:membership([r.get(key) for r in available]) for key in flags},
                                 'finite':{key:finite_membership([r[key] for r in available]) for key in ['N0','v1','N2','k','k_t','X','V','frame_norm','gK_center_proxy']}}
    report['frame_peak_original_wall_records']=[i for i,r in enumerate(rows) if r.get('frame_maximum_original_archive_wall') is True]
    report['frame_peak_numerical_endpoint_records']=[i for i,r in enumerate(rows) if r.get('frame_maximum_numerical_grid_endpoint') is True]
    return report


def support_population(value,positive,original_indices,seam=False):
    selected=value[:,positive]
    counts=np.count_nonzero(selected,axis=1)
    item={'scope':'Positive stored radii only; r=0 excluded.',
          'true_samples':int(np.sum(counts)),'false_samples':int(selected.size-np.sum(counts)),
          'records_all_true':int(np.count_nonzero(np.all(selected,axis=1))),
          'records_any_true':int(np.count_nonzero(np.any(selected,axis=1)))}
    if seam:
        item['variant_record_indices_with_any_true']=np.flatnonzero(np.any(selected,axis=1)).tolist()
        item['original_record_indices_with_any_true']=original_indices[np.any(selected,axis=1)].tolist()
    return item


def support_accounting(path):
    output={}
    with np.load(path) as archive:
        positive=archive['R']>0
        original=archive['original_record_indices']
        for key in archive.files:
            if '_support_' not in key and key!='centroid_full_support':continue
            value=archive[key]
            if value.shape!=(len(archive['T']),len(positive)) or value.dtype!=bool:raise ValueError('malformed targeted support mask '+key)
            output[key]=support_population(value,positive,original,'known_seam' in key or 'unknown_seam' in key)
    return output


def headers(path):
    output={}
    with zipfile.ZipFile(path) as archive:
        for entry in archive.namelist():
            with archive.open(entry) as stream:
                version=np.lib.format.read_magic(stream)
                shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
            output[entry[:-4]]={'shape':list(shape),'dtype':str(dtype)}
    return output


def controls():
    peak={'value':1.,'partial':False,'radii':[4.],'indices':[1],'requested_points':2}
    row={'T':0.,'evaluation_pre_gate':True,'solver':{'prefix':None},'unavailable_or_invalid_locations':{'bad':[]},
         'sphere':peak,'frame':peak,'all_support_restricted_sphere':peak,'all_support_restricted_frame':peak,'center':{}}
    other={**row,'T':8.,'evaluation_pre_gate':False,'solver':{'prefix':False},'unavailable_or_invalid_locations':{'bad':[2,3]},
           'frame':{'value':None,'partial':True,'radii':[],'indices':[],'requested_points':2}}
    answer=audit_rows([row,other])
    tests={'three_state_membership':membership([True,False,None])=={'true':[0],'false':[1],'unavailable':[2]},
           'invalid_location_population':answer['invalid_locations']['bad']=={'record_indices':[1],'total_locations':2},
           'unavailable_not_partial_evaluation':answer['frame_availability']=={'fully_available':[0],'partial':[],'unavailable':[1],'empty_restricted_selection':[],'missing_requested_physical_samples':[1]},
           'partial_global_coverage':answer['all_frame_maximum']['partial'] and answer['all_frame_maximum']['unavailable_records']==1,
           'distinct_pre_post':answer['pre_frame_maximum']['selected_records']==1 and answer['post_frame_maximum']['value'] is None,
           'finite_zero_is_available':finite_membership([0.,None,float('inf')])=={'finite':[0],'unavailable':[1,2]}}
    population=support_population(np.array([[True,False,True],[True,False,False]],dtype=bool),np.array([False,True,True]),np.array([3,7]),True)
    tests['support_population_excludes_center']=population['true_samples']==1 and population['false_samples']==3 and population['records_all_true']==0 and population['records_any_true']==1
    tests['support_original_seam_locators']=population['variant_record_indices_with_any_true']==[0] and population['original_record_indices_with_any_true']==[3]
    empty={**row,'frame':{'value':None,'partial':False,'radii':[],'indices':[],'requested_points':0}}
    empty_report=audit_rows([empty])
    tests['empty_restricted_selection_is_not_missing_physical_samples']=empty_report['frame_availability']['empty_restricted_selection']==[0] and empty_report['frame_availability']['missing_requested_physical_samples']==[]
    if not all(tests.values()):raise RuntimeError('output accounting controls failed')
    return tests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    runtime=require_remote_cpu(args.input);tests=controls();profiles=[]
    for history in EXPECTED:
        for role in ('metric_nodes','applied_sites'):
            for variant,*_ in VARIANTS:
                folder=args.input/history/role/variant
                saved=json.loads((folder/'records.json').read_text())
                report=audit_rows(saved['rows'])
                report.update(history=history,role=role,variant=variant,metadata=saved['metadata'],
                              original_record_indices=[r['record_index'] for r in saved['rows']],
                              physical_T=[r['T'] for r in saved['rows']],
                              records_source=str(folder/'records.json'),profiles_source=str(folder/'profiles.npz'),
                              first_T=saved['rows'][0]['T'],last_T=saved['rows'][-1]['T'],
                              array_headers=headers(folder/'profiles.npz'),support_accounting=support_accounting(folder/'profiles.npz'))
                profiles.append(report)
    write(args.input/'output_accounting.json',{'scope':'Complete frozen producer row accounting and NPZ shape/dtype headers; no derivatives or new physical readout.',
          'runtime':runtime,'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'manufactured_controls':tests,'profile_count':len(profiles),'profiles':profiles})
    print(json.dumps({'accounted_profiles':len(profiles)}),flush=True)


if __name__=='__main__':main()
