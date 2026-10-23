"""Own-code stored-node curvature and prescribed sensitivity readout; no evolution."""
import argparse
import gc
import hashlib
import json
import numbers
import shutil
from pathlib import Path
import numpy as np
from cpu_runtime import require_remote_cpu
from curvature_core_v1 import (calculate, full_frame_domain_support, linear_sphere_samples, maximum, ordered_grid,
                              nominal_gate, prefix, support_masks)
from inventory_CURVATURE_inputs_v1 import SOURCES
from run_LADDERS_v1p3_arithmetic_v1 import classify

COMPONENTS=['X','Y','Z','U','V','frame_norm','kretschmann']
EXPECTED={'prod_r11':(4,8,321),'B2':(4,8,801),'P64_s4':(4,8,321),
          'P64_s2':(2,8,321),'P64_s8':(8,8,321),'P64_s4_tau4':(4,4,641),
          'P64_s4_tau2':(4,2,689)}
VARIANTS=[('primary',1,1,3,3),('radial_five',1,1,5,3),
          ('radial_stride2',1,2,3,3),('radial_stride4',1,4,3,3),
          ('time_five',1,1,3,5),('cadence2',2,1,3,3),('cadence4',4,1,3,3)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')


def numeric_object_matrix(array):
    array=np.asarray(array)
    if array.dtype==object:
        if not all(isinstance(x,numbers.Real) and not isinstance(x,(bool,np.bool_)) for x in array.flat):
            raise ValueError('object field contains a non-real numeric scalar')
    elif not np.issubdtype(array.dtype,np.number) or np.iscomplexobj(array):
        raise ValueError('field is not a real numeric array')
    return np.asarray(array,dtype=float)


def bound_scalar(array):
    array=np.asarray(array)
    if array.shape!=() or np.iscomplexobj(array) or not np.issubdtype(array.dtype,np.number):
        raise ValueError('parameter must be a real numeric scalar of shape ()')
    value=array.item()
    if not np.isfinite(value):raise ValueError('nonfinite parameter')
    return value


def validate_contract(raw,t,r,n,v,role,name):
    if role not in ('metric_nodes','applied_sites'):raise ValueError('unknown field role')
    args=raw['args'];ell,tau,count=EXPECTED[name]
    for key,expected in [('m',.2),('g',.25),('rs',64.),('ell',ell),('tau',tau)]:
        if args.get(key)!=expected:raise ValueError('wrong physical parameter '+key)
    if args.get('site_fields')!='averaged':raise ValueError('unbound site-field role')
    t,r=ordered_grid(t),ordered_grid(r,5)
    if len(t)!=count or n.shape!=(len(t),len(r)) or v.shape!=n.shape:raise ValueError('source dimensions mismatch')
    if t[0]!=0 or t[-1]!=(count-1)*tau:raise ValueError('wrong full stored time extent')
    if args.get('nsites')!=(4096 if name=='B2' else 2048):raise ValueError('wrong declared full site domain')
    if len(raw['records'])!=len(t):raise ValueError('JSON record count mismatch')
    if not np.array_equal(t,np.asarray([record['T'] for record in raw['records']])):raise ValueError('JSON/NPZ clock mismatch')
    if not np.array_equal(np.diff(t),np.full(len(t)-1,float(tau))):raise ValueError('missing or changed record cadence')
    for record,clock in zip(raw['records'],t):
        if record.get('T_over_rs')!=clock/64:raise ValueError('nominal/physical clock mismatch')
    if role=='metric_nodes' and r[0]!=0:raise ValueError('missing metric center node')
    if role=='applied_sites' and r[0]!=1:raise ValueError('wrong applied site origin')
    if 'nsites' in args:
        expected_grid=np.arange(0,args['nsites']+ell,ell,dtype=float) if role=='metric_nodes' else np.arange(1,args['nsites']+1,dtype=float)
        if not np.array_equal(r,expected_grid):raise ValueError('radial grid differs from declared stored grid')
    return True


def contract_controls():
    t=np.arange(321)*8.;r=np.arange(513)*4.;n=np.ones((321,513));v=n*0
    raw={'args':{'m':.2,'g':.25,'rs':64.,'ell':4,'tau':8.,'site_fields':'averaged','nsites':2048},
         'records':[{'T':float(a),'T_over_rs':float(a/64)} for a in t]}
    cases=[]
    cases.append({'name':'valid bound field contract','passed':validate_contract(raw,t,r,n,v,'metric_nodes','prod_r11')})
    def rejects(label,fn):
        try:fn();ok=False
        except (ValueError,TypeError):ok=True
        cases.append({'name':label,'passed':ok})
    rejects('wrong role',lambda:validate_contract(raw,t,r,n,v,'guessed_alias','prod_r11'))
    wrong=json.loads(json.dumps(raw));wrong['args']['rs']=1
    rejects('wrong units',lambda:validate_contract(wrong,t,r,n,v,'metric_nodes','prod_r11'))
    missing=t.copy();missing[10]+=1
    rejects('missing clock',lambda:validate_contract(raw,missing,r,n,v,'metric_nodes','prod_r11'))
    rejects('malformed object entry',lambda:numeric_object_matrix(np.array([['1.0']],dtype=object)))
    rejects('altered grid shape',lambda:validate_contract(raw,t,r[:-1],n,v,'metric_nodes','prod_r11'))
    rejects('complex grid',lambda:ordered_grid(np.arange(5)+.1j))
    rejects('one-element vector scalar',lambda:bound_scalar(np.array([.25])))
    # Source-specific seam controls use the actual metadata and support helpers.
    def seam_fixture():
        return {'args':{'resume_T_over_rs':3/64},'head':{'resume':{'T_over_rs':3/64}},
                'records':[{'engine':'a','solver':{'module':'field'},'inversions':i} for i in range(9)]}
    seam_raw=seam_fixture();seam_raw['records'][4]['resumed_from']={'T_over_rs':3/64}
    boundaries,events=seam_rows(seam_raw,np.arange(9,dtype=float))
    cases.append({'name':'duplicate restart declarations deduplicated','passed':boundaries==[False]*3+[True]+[False]*4 and len(events)==1})
    cases.append({'name':'one-shot annotation disappearance creates no seam','passed':boundaries[4] is False})
    no_resume=seam_fixture();no_resume['args']={};no_resume['head']={}
    ordinary,_=seam_rows(no_resume,np.arange(9,dtype=float))
    cases.append({'name':'ordinary evolving counter creates no seam','passed':ordinary==[False]*8})
    no_resume['args']['resume_slice']='opaque-non-clock-locator'
    unknown,_=seam_rows(no_resume,np.arange(9,dtype=float))
    cases.append({'name':'unlocated declared restart unresolved','passed':all(value is None for value in unknown)})
    from curvature_core_v1 import derivative_table
    tables={'radial_first':derivative_table(np.arange(9)+1.,1),'radial_second':derivative_table(np.arange(9)+1.,2),
            'time_first':derivative_table(np.arange(9,dtype=float),1)}
    valid=np.ones((9,9),dtype=bool)
    support=support_masks(tables,valid,valid,[True]*9,[True]*9,np.arange(9),boundaries)
    cases.extend([{'name':'support ending at k touches without crossing','passed':bool(support['frame_support_seam_touch_only'][2,4] and not support['frame_support_known_seam'][2,4])},
                  {'name':'support including k and later crosses','passed':bool(support['frame_support_known_seam'][3,4])},
                  {'name':'support wholly after k does not cross','passed':bool(not support['frame_support_known_seam'][5,4])}])
    indices=np.array([0,2,4,6,8]);tables['time_first']=derivative_table(indices.astype(float),1)
    coarse=support_masks(tables,valid[:5],valid[:5],[True]*5,[True]*5,indices,boundaries)
    cases.append({'name':'coarsened stencil skipping k still crosses','passed':bool(coarse['frame_support_known_seam'][1,4])})
    after_indices=np.arange(3,8);tables['time_first']=derivative_table(after_indices.astype(float),1)
    starts=support_masks(tables,valid[:5],valid[:5],[True]*5,[True]*5,after_indices,boundaries)
    cases.append({'name':'one-sided support starting at k crosses','passed':bool(starts['frame_support_known_seam'][0,4])})
    tables['time_first']=derivative_table(np.arange(9,dtype=float),1)
    unresolved=support_masks(tables,valid,valid,[True]*9,[True,None]+[True]*7,np.arange(9),[False]*8)
    cases.append({'name':'unresolved supporting prefix retained','passed':bool(unresolved['frame_support_solver_prefix_unresolved'][1,4])})
    no_wall=support_masks(tables,valid,valid,[True]*9,[True]*9,np.arange(9),[False]*8,wall_index=None)
    cases.append({'name':'numerical subset endpoint is not automatically archive wall','passed':bool(no_wall['frame_support_numerical_grid_endpoint_touch'][4,8] and not no_wall['frame_support_wall_touch'][4,8])})
    nested=seam_fixture();nested['args']={};nested['head']={'resume_slice_v34':{'slice':'opaque'}}
    unknown_nested,_=seam_rows(nested,np.arange(9,dtype=float))
    cases.append({'name':'nested unlocated declared restart unresolved','passed':all(value is None for value in unknown_nested)})
    from summarize_CURVATURE_secondroute_v1 import exact_join,global_maximum,comparison_fields
    joined=exact_join([0,8,16],[0,4,8],[0,16,32],[0,8,16])
    cases.append({'name':'saved-array exact join uses no nearest replacement','passed':joined[0].tolist()==[0,16] and joined[3].tolist()==[8] and joined[1].tolist()==[0,2]})
    synthetic=[{'T':0,'frame':{'value':2.,'radii':[4.],'partial':False}},
               {'T':8,'frame':{'value':None,'radii':[],'partial':True}},
               {'T':16,'frame':{'value':1.,'radii':[8.],'partial':True}}]
    summary=global_maximum(synthetic,'frame')
    cases.append({'name':'global maximum preserves unavailable selected record','passed':summary['value']==2 and summary['partial'] and summary['selected_records']==3 and summary['evaluated_records']==2 and summary['unavailable_records']==1})
    comparison=comparison_fields(np.array([.1,.11,np.nan,-1e308]),np.array([.11,.1,.12,1e308]),.1)
    cases.extend([
        {'name':'production signed differences are second minus first','passed':bool(comparison['signed_difference'][0]>0 and comparison['signed_difference'][1]<0)},
        {'name':'production threshold equality and straddle','passed':comparison['threshold_disagreement'].tolist()==[True,True,False,True]},
        {'name':'missing pair is unavailable rather than zero agreement','passed':bool(not comparison['available_pair'][2] and np.isnan(comparison['signed_difference'][2]))},
        {'name':'finite input overflow has unavailable difference but available input pair','passed':bool(comparison['available_pair'][3] and not comparison['available_difference'][3] and np.isnan(comparison['signed_difference'][3]))}])
    return cases


def load_history(name):
    stem=SOURCES[name];record_path=Path(stem+'.json');field_path=Path(stem+'_fields.npz')
    binding={'record_path':str(record_path),'fields_path':str(field_path),
             'record_sha256':digest(record_path),'fields_sha256':digest(field_path)}
    raw=json.loads(record_path.read_text())
    with np.load(field_path,allow_pickle=True) as archive:
        for key in ('g','ell','tau'):
            if bound_scalar(archive[key])!=raw['args'][key]:raise ValueError('archive/JSON scalar mismatch '+key)
        t=numeric_object_matrix(archive['T']);r=numeric_object_matrix(archive['R'])
        nodes=numeric_object_matrix(archive['r_nodes'])
        n=numeric_object_matrix(archive['N_nodes']);v=numeric_object_matrix(archive['v_nodes'])
        if nodes.ndim!=2 or not np.array_equal(nodes,np.broadcast_to(nodes[0],nodes.shape)):
            raise ValueError('node grids differ between records')
        rn=nodes[0]
        validate_contract(raw,t,rn,n,v,'metric_nodes',name)
        na,va=numeric_object_matrix(archive['N_applied']),numeric_object_matrix(archive['v_applied'])
        validate_contract(raw,t,r,na,va,'applied_sites',name)
        binding['arrays']={key:{'shape':list(archive[key].shape),'dtype':str(archive[key].dtype)} for key in archive.files}
    if binding['record_sha256']!=digest(record_path) or binding['fields_sha256']!=digest(field_path):
        raise ValueError('input mutated during binding')
    return raw,binding,t,rn,n,v,r,na,va


def residual_rows(raw):
    rows=[];locals=[]
    for i,record in enumerate(raw['records']):
        try:
            result=classify(record,raw['args'].get('mix')=='newton');local=bool(result['counts'])
        except (ValueError,TypeError,KeyError) as error:
            result={'unresolved':str(error)};local=None
        locals.append(local);rows.append({'record_index':i,'T':record['T'],'local':local,**result})
    ancestry=prefix(locals)
    for row,p in zip(rows,ancestry):row['prefix']=p
    return rows,ancestry


def seam_rows(raw,t):
    boundaries=[False]*(len(t)-1);events={}
    def mark(k,state,locator):
        if not 0<=k<len(boundaries):return
        if state is True:boundaries[k]=True
        elif boundaries[k] is not True:boundaries[k]=None
        event=events.setdefault(k,{'boundary_k':k,'locators':[]})
        if locator not in event['locators']:event['locators'].append(locator)
    previous=None
    for i,record in enumerate(raw['records']):
        signature=(record.get('engine'),record.get('solver',{}).get('module'))
        if i and signature!=previous:
            known=any(a!=b and a is not None and b is not None for a,b in zip(previous,signature))
            mark(i-1,True if known else None,{'kind':'method_identifier_transition',
                                            'left':i-1,'right':i,'old':previous,'new':signature,
                                            'missing_identifier_transition':any((a is None)!=(b is None) for a,b in zip(previous,signature))})
        previous=signature
    clocks=[];locators=[];unlocated=[]
    args=raw['args'];head=raw.get('head',{})
    declared=[('args.resume_T_over_rs',{'T_over_rs':args.get('resume_T_over_rs')}),
              ('head.resume',head.get('resume',{})),('head.resume_slice_v34',head.get('resume_slice_v34',{})),
              ('runner.resume',(raw.get('runner') or {}).get('resume',{}))]
    declared.extend(('records[%s].resumed_from'%i,record['resumed_from']) for i,record in enumerate(raw['records']) if isinstance(record.get('resumed_from'),dict))
    for locator,item in declared:
        if not isinstance(item,dict):continue
        clock=item.get('resume_T_over_rs',item.get('T_over_rs'))
        if isinstance(clock,(float,int)) and not isinstance(clock,bool):
            physical=float(clock)*64;clocks.append(physical)
            ids=np.flatnonzero(t==physical)
            if len(ids)!=1:raise ValueError('declared restart clock is absent: '+locator)
            k=int(ids[0])
            if 'record_index' in item and item['record_index']!=k:raise ValueError('restart index/clock disagreement')
            if 'stored_records_kept' in item and item['stored_records_kept']!=k+1:raise ValueError('restart kept-count/clock disagreement')
            locators.append((k,{'kind':'declared_resume','source':locator,'T':physical,'record_index':k}))
        elif item and locator!='args.resume_T_over_rs':unlocated.append(locator)
    if len(set(clocks))>1:raise ValueError('inconsistent scalar restart locators')
    for k,locator in locators:mark(k,True,locator)
    labels=list(events.values())
    if (any(args.get(key) for key in ('resume_slice','resume_run','resume_from')) and not clocks) or unlocated:
        boundaries=[None if value is False else value for value in boundaries]
        labels.append({'kind':'resume_declaration_without_clock','unlocated_sources':unlocated})
    if not labels:labels=[{'kind':'no_declared_seam_in_inspected_metadata'}]
    return boundaries,labels


def gate_local(values,positive):
    result=[]
    for row in values[:,positive]:
        finite=np.isfinite(row)
        result.append(False if np.any(row[finite]>.1) else None if not np.all(finite) else True)
    return result


def intervals(mask,r):
    ids=np.flatnonzero(mask)
    if not len(ids):return []
    groups=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [{'first_index':int(g[0]),'last_index':int(g[-1]),'first_radius':float(r[g[0]]),'last_radius':float(r[g[-1]])} for g in groups]


def record_table(data,centers,raw,tids,quality,pre,cap,gate_cap,role,variant):
    rows=[];positive=data['R']>0;r=data['R']
    for i,original in enumerate(tids):
        clock=float(data['T'][i]);n=data['N'][i,positive]
        row={'record_index':int(original),'T':clock,'T_over_r0':clock/64,'field_role':role,
             'variant':variant,'sphere':maximum(data['V'][i],r,positive),
             'frame':maximum(data['frame_norm'][i],r,positive),
             'centered_radial_frame':maximum(data['frame_norm'][i],r,positive&data['radial_centered']),
             'minimum_lapse':float(np.min(n[np.isfinite(n)])) if np.any(np.isfinite(n)) else None,
             'time_centered':bool(data['time_centered'][i]),'solver':quality[int(original)],
             'evaluation_pre_gate':pre[int(original)],'center':{},
             'source_R_c':raw['records'][int(original)].get('R_c')}
        row['variant_radial_domain']=[float(r[0]),float(r[-1])]
        row['original_archive_radial_domain']=[float(data['source_first_radius']),float(data['source_wall_radius'])]
        row['unavailable_or_invalid_locations']={
            'nonpositive_lapse_indices':np.flatnonzero(np.isfinite(data['N'][i])&(data['N'][i]<=0)).tolist(),
            'nonfinite_lapse_indices':np.flatnonzero(~np.isfinite(data['N'][i])).tolist(),
            'nonfinite_flow_indices':np.flatnonzero(~np.isfinite(data['v'][i])).tolist(),
            'positive_radius_unavailable_sphere_indices':np.flatnonzero(positive&~np.isfinite(data['V'][i])).tolist(),
            'positive_radius_unavailable_frame_indices':np.flatnonzero(positive&~np.isfinite(data['frame_norm'][i])).tolist()}
        for field in COMPONENTS:
            row[field+'_absolute_max']=maximum(np.abs(data[field][i]),r,positive)
        for field in ('gK','gH','gA'):
            row[field]=maximum(data[field][i],r,positive)
            row[field+'_failing_intervals']=intervals(positive&np.isfinite(data[field][i])&(data[field][i]>.1),r)
        for name in ('direct','space1','Y','frame'):
            selected=positive&data[name+'_all_support_restricted'][i]
            row[name+'_all_support_points']=int(np.count_nonzero(selected))
        row['all_support_restricted_frame']=maximum(data['frame_norm'][i],r,positive&data['frame_all_support_restricted'][i])
        row['all_support_restricted_sphere']=maximum(data['V'][i],r,positive&data['direct_all_support_restricted'][i])
        rc=row['source_R_c']
        if isinstance(rc,(int,float)) and np.isfinite(rc) and rc>0:
            selected=positive&(r<=rc)
            row['centroid_domain']={key:maximum(data[key][i],r,selected) for key in ('V','frame_norm','gK','gH','gA')}
            row['centroid_domain']['all_support_restricted_frame']=maximum(data['frame_norm'][i],r,selected&data['frame_all_support_restricted'][i]&data['centroid_full_support'][i])
        else:row['centroid_domain']={'unavailable':'missing finite positive source-defined R_c'}
        if not row['time_centered']:row['centered_radial_and_time_frame']=maximum(data['frame_norm'][i],r,np.zeros(len(r),dtype=bool))
        else:row['centered_radial_and_time_frame']=row['centered_radial_frame']
        row['sphere_over_center_cap']=None if row['sphere']['value'] is None else row['sphere']['value']/cap
        row['sphere_over_gate_scale']=None if row['sphere']['value'] is None else row['sphere']['value']/gate_cap
        row['endpoint_values']={key:[float(data[key][i,1 if r[0]==0 else 0]),float(data[key][i,-1])] for key in ('V','frame_norm','N')}
        if row['frame']['indices']:
            j=row['frame']['indices'][0];row['frame_maximum_components']={k:float(data[k][i,j]) for k in COMPONENTS}
            row['frame_maximum_positive_sample_endpoint']=bool(j==len(r)-1 or j==(1 if r[0]==0 else 0))
            row['frame_maximum_numerical_grid_endpoint']=bool(j in (0,len(r)-1))
            row['frame_maximum_original_archive_wall']=bool(r[j]==float(data['source_wall_radius']))
        for k,values in centers.items():
            if values is not None:
                row['center'][k]={key:clean(value[i]) for key,value in values.items()}
                for quantity,peak in [('V','sphere'),('frame_norm','frame')]:
                    value=values[quantity][i]
                    row['center'][k][quantity+'_exceeds_positive_sample_maximum']=(bool(value>row[peak]['value']) if np.isfinite(value) and row[peak]['value'] is not None else None)
        if centers.get('2') is not None and centers.get('3') is not None:
            row['center_proxy_sensitivity']={key+'_two_minus_three':clean(centers['2'][key][i]-centers['3'][key][i]) for key in ('V','X','frame_norm','gK_center_proxy')}
            two,three=centers['2']['gK_center_proxy'][i],centers['3']['gK_center_proxy'][i]
            row['center_proxy_sensitivity']['conditional_gate_straddled']=(bool((two<=.1)!=(three<=.1)) if np.isfinite(two) and np.isfinite(three) else None)
        rows.append(row)
    return rows


def write_variant(folder,result,tables,centers,rows,metadata):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez(folder/'profiles.npz',**result)
    stencil_arrays={name+'__'+key:value for name,table in tables.items() for key,value in table.items()}
    np.savez_compressed(folder/'stencils.npz',**stencil_arrays)
    np.savez_compressed(folder/'centers.npz',**{k+'__'+name:value for k,values in centers.items() if values is not None for name,value in values.items()})
    write(folder/'records.json',{'scope':'Sampled finite-difference metric diagnostics, not continuum error enclosures.',
                               'metadata':metadata,'rows':rows})


def run_history(name,out):
    raw,binding,t,r,n,v,rsites,napplied,vapplied=load_history(name)
    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(binding['record_path'],folder/'input_records.json')
    quality,solver= residual_rows(raw)
    seams,seam_labels=seam_rows(raw,t)
    write(folder/'input_bindings.json',{'binding':binding,'args':raw['args'],'head':raw.get('head'),
                                     'runner':raw.get('runner'),'solver_rows':quality,'seams':seam_labels})
    sigma=float(raw['args']['ell']);cap=(32*np.pi*.25/3)*np.sqrt(.2**2+1.5);gate_cap=(.1/sigma)**2
    own_pre=None;account=[]
    for role,grid,nn,vv in [('metric_nodes',r,n,v),('applied_sites',rsites,napplied,vapplied)]:
        for variant,ts,radial_stride,rwidth,twidth in VARIANTS:
            tids=np.arange(0,len(t),ts);rids=np.arange(0,len(grid),radial_stride)
            result,tables,centers=calculate(t[tids],grid[rids],nn[np.ix_(tids,rids)],vv[np.ix_(tids,rids)],sigma,rwidth,twidth)
            if role=='metric_nodes' and variant=='primary':
                own_pre=prefix(gate_local(result['gK'],result['positive_radius']))
                write(folder/'gate_prefix.json',{'local':gate_local(result['gK'],result['positive_radius']),'prefix':own_pre})
            local_pre=[own_pre[int(i)] for i in tids];local_solver=[solver[int(i)] for i in tids]
            strain=result['gK_pass'].copy()
            # Center support remains explicitly conditional and cannot enter the strict set.
            if grid[0]==0:strain[:,0]=True
            supports=support_masks(tables,result['field_domain_valid'],strain,local_pre,local_solver,
                                   tids,seams,0 if grid[0]==0 else None,
                                   len(rids)-1 if rids[-1]==len(grid)-1 else None)
            result.update(supports)
            result['source_first_radius']=np.asarray(grid[0]);result['source_wall_radius']=np.asarray(grid[-1])
            for count,proxy in centers.items():
                if proxy is None:continue
                table=tables['time_first']['indices']
                positive_count=int(count)
                proxy['gK_center_proxy']=sigma*np.abs(proxy['k'])
                proxy['conditional_center_gate_pass']=nominal_gate(proxy['gK_center_proxy'])
                proxy['center_gate_unavailable']=~np.isfinite(proxy['gK_center_proxy'])
                proxy['spatial_positive_support_node_gate_pass']=np.all(result['gK_pass'][:,1:positive_count+1],axis=1)
                proxy['full_time_positive_support_node_gate_pass']=np.all(proxy['spatial_positive_support_node_gate_pass'][table],axis=1)
                proxy['full_time_conditional_center_gate_pass']=np.all(proxy['conditional_center_gate_pass'][table],axis=1)
                proxy['full_time_support_pre_prefix']=np.all((np.asarray(local_pre,dtype=object)==True)[table],axis=1)
                proxy['full_time_support_solver_prefix']=np.all((np.asarray(local_solver,dtype=object)==True)[table],axis=1)
                for key,values in [('pre',local_pre),('solver',local_solver)]:
                    objects=np.asarray(values,dtype=object)[table]
                    proxy['full_time_support_'+key+'_prefix_unresolved']=np.any(objects==None,axis=1)&~np.any(objects==False,axis=1)
                proxy['full_time_support_known_seam']=supports['frame_support_known_seam'][:,0]
                proxy['full_time_support_unknown_seam']=supports['frame_support_unknown_seam'][:,0]
                proxy['spatial_sphere_support_valid']=proxy['parity_valid']&np.isfinite(proxy['V'])
                proxy['conditional_all_support_restricted_frame']=(proxy['full_time_support_valid']&proxy['full_time_support_pre_prefix']
                    &proxy['full_time_support_solver_prefix']&proxy['full_time_positive_support_node_gate_pass']
                    &proxy['full_time_conditional_center_gate_pass']&~proxy['full_time_support_known_seam']&~proxy['full_time_support_unknown_seam'])
            centroid=np.zeros_like(result['field_domain_valid'])
            for i,original in enumerate(tids):
                rc=raw['records'][int(original)].get('R_c')
                if isinstance(rc,(int,float)) and np.isfinite(rc) and rc>0:centroid[i]=result['R']<=rc
            result['centroid_full_support']=full_frame_domain_support(centroid,tables)
            result['original_record_indices']=tids;result['original_radius_indices']=rids
            result['sphere_over_center_cap']=result['V']/cap
            result['sphere_over_gate_scale']=result['V']/gate_cap
            rows=record_table(result,centers,raw,tids,quality,own_pre,cap,gate_cap,role,variant)
            meta={'history':name,'role':role,'variant':variant,'sigma':sigma,'cap_center':cap,'gate_scale':gate_cap,
                  'original_archive_radial_domain':[float(grid[0]),float(grid[-1])],
                  'variant_radial_domain':[float(result['R'][0]),float(result['R'][-1])],
                  'source_binding':binding,'seams':seam_labels,
                  'support_rule':'V direct; U/Z radial-first; Y current radial-first union time-direct; X/full-frame current radial-second union time-first x radial-first. Stencil indices locate all primitive samples. Strict support excludes any center touch and known/unknown seam.',
                  'center_status':'parity-conditioned extrapolation only','native_per_site_covariance_check':'unavailable'}
            write_variant(folder/role/variant,result,tables,centers,rows,meta)
            if role=='metric_nodes' and variant=='primary':
                sampled=linear_sphere_samples(r,nn,vv,rsites)
                np.savez(folder/'linear_to_sites_sphere.npz',T=t,**sampled)
                write(folder/'linear_to_sites_records.json',{'rows':[{'T':float(t[i]),'sphere':maximum(q,rsites)} for i,q in enumerate(sampled['V'])]})
            account.append({'role':role,'variant':variant,'requested_records':len(tids),'evaluated_records':len(rows),
                            'fully_available_frame_records':sum(row['frame']['value'] is not None and not row['frame']['partial'] for row in rows),
                            'partial_frame_records':sum(row['frame']['value'] is not None and row['frame']['partial'] for row in rows),
                            'unavailable_frame_records':sum(row['frame']['value'] is None for row in rows)})
            print(json.dumps({'history':name,'role':role,'variant':variant,'rows_saved':len(rows)}),flush=True)
            del result,tables,centers,rows,supports;gc.collect()
    write(folder/'accounting.json',{'history':name,'accounting':account,'native_per_site_covariance_check':'unavailable'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--controls-only',action='store_true')
    parser.add_argument('--histories',nargs='*',default=list(SOURCES))
    args=parser.parse_args();runtime=require_remote_cpu(args.output)
    from control_CURVATURE_v1 import run_known_answers
    source_files=['curvature_core_v1.py','control_CURVATURE_v1.py','run_CURVATURE_secondroute_v1.py','summarize_CURVATURE_secondroute_v1.py',
                  'run_LADDERS_v1p3_arithmetic_v1.py','inventory_CURVATURE_inputs_v1.py','cpu_runtime.py']
    hashes={name:digest(Path(__file__).parent/name) for name in source_files}
    hashes['frozen_specification']=digest(args.spec)
    if args.controls_only:
        answers=run_known_answers(contract_controls);answers.update(runtime=runtime,code_sha256=hashes)
        write(args.output/'known_answers.json',answers)
        if not answers['all_passed']:raise RuntimeError('manufactured curvature control failed')
        print(json.dumps({'controls':len(answers['tests']),'all_passed':True}));return
    controls=json.loads((args.output/'known_answers.json').read_text())
    if not controls.get('all_passed') or controls.get('code_sha256')!=hashes:
        raise RuntimeError('passing actual-source controls required before physical evaluation')
    write(args.output/'runtime.json',{'runtime':runtime,'code_sha256':hashes,'frozen_spec_source':str(args.spec)})
    for name in args.histories:
        if name not in SOURCES:raise ValueError('unknown requested history')
        try:run_history(name,args.output)
        except Exception as error:
            write(args.output/name/'unavailable.json',{'history':name,'error':repr(error)})
            raise


if __name__=='__main__':main()
