#!/usr/bin/env python3
"""Metadata-only price of retained Step-1 plans; no operator evaluation."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import traceback
ROOT=Path('<internal>/vacuum_ref_u2_v2')
OLD=ROOT/'prefix_v1'
NEW=ROOT/'prefix_missing_v1'
START=time.monotonic()
PHASE='metadata'
SIGNALS=[]
CAP=48*1024**2
class Interrupted(Exception):pass
def handler(sig,frame):
    SIGNALS.append({'signal':int(sig),'phase':PHASE,'elapsed':time.monotonic()-START})
    if PHASE!='metadata':raise Interrupted('price signal '+str(sig)+' during '+PHASE)
def boundary():
    if SIGNALS:raise Interrupted('pending main-work signal')
def metadata(st):
    return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}


def inspect(path,expected=None,capture=False):
    path=Path(path);o={'path':str(path),'expected_sha256':expected,'bytes_read':0,'phase':'resolve'}
    digest=None;blocks=[]
    try:
        resolved=str(path.resolve(strict=True));o['resolved_path']=resolved
        digest=hashlib.sha256()
        with path.open('rb') as f:
            o['phase']='initial_fstat'
            before=metadata(os.fstat(f.fileno()));o['before']=before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular source/output required')
            o['phase']='read'
            while True:
                block=f.read(8*1024**2)
                if not block:break
                digest.update(block);o['bytes_read']+=len(block)
                if capture:blocks.append(block)
            o['phase']='final_fstat'
            after=metadata(os.fstat(f.fileno()));o['after']=after
            o['phase']='close'
        o['phase']='path_stat'
        current=metadata(path.stat());o['path_now']=current
        o['phase']='final_resolve'
        resolved_now=str(path.resolve(strict=True));o['resolved_path_now']=resolved_now
        actual=digest.hexdigest();o['sha256']=actual;o['phase']='compare'
        if before!=after or before!=current or resolved!=resolved_now or (expected is not None and actual!=expected):
            raise ValueError('unstable or mismatched file: '+str(path))
        pin={'path':str(path),'resolved_path':resolved,'sha256':actual,**before}
        return (pin,b''.join(blocks)) if capture else pin
    except BaseException as exc:
        if digest is not None:o['partial_sha256']=digest.hexdigest()
        exc.observation=o
        raise


def nodup(rows):
    d={}
    for k,v in rows:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d
def badconstant(token):raise ValueError('nonfinite JSON '+token)
def loads(data):return json.loads(data,object_pairs_hook=nodup,parse_float=str,parse_constant=badconstant)
def dump(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pair(value):return [value.numerator,value.denominator]
def cpu(value):
    a=Fraction(str(value['user_seconds']));b=Fraction(str(value['system_seconds']))
    if a<0 or b<0:raise ValueError('negative CPU metadata')
    return a+b
def rowref(path,line,row):
    return {'file':str(path),'sequence':row['sequence'],'object_id':row['object_id'],'sha256_line':hashlib.sha256(line).hexdigest()}
def refs(value):
    if isinstance(value,dict):
        out=[value] if set(('file','sequence','object_id','sha256_line')).issubset(value) else []
        return out+[r for v in value.values() for r in refs(v)]
    if isinstance(value,list):return [r for v in value for r in refs(v)]
    return []
def validate_times(value):
    if isinstance(value,dict):
        if 'user_seconds' in value and 'system_seconds' in value:
            cpu(value)
            if 'wall_seconds' in value and Fraction(str(value['wall_seconds']))<0:raise ValueError('negative wall receipt')
        for v in value.values():validate_times(v)
    elif isinstance(value,list):
        for v in value:validate_times(v)
class Record:
    def __init__(self,out):
        if out!=ROOT/'pricing_v1' or out.resolve()!=out or not out.is_dir():raise ValueError('literal owned pricing output path')
        self.out=out;self.bytes=0;self.seq=0;self.done=set();self.expected=[];self.admitted={};self.visited=set();self.failed=[];self.postchecks={};self.error=None
        self.streams={n:(out/(n+'.jsonl')).open('xb') for n in ('admission','pricing')}
        self.stage_plan=['admission','reuse_index','sample_views','applicability','samples','roster','resources','spend','aggregate','postchecks','assessment'];self.roles={('runtime','runtime'),('disk','disk'),('input_plan','input_plan'),('reuse_index','reuse_index'),('logical_role_plan','logical_role_plan'),('logical_role_plan','admitted_operation_role_map'),('source_applicability','source_applicability'),('excluded_segmented_view','excluded_segmented_view'),('resource_forecast','resource_forecast'),('spend_allowance','spend_allowance'),('aggregate','aggregate'),('assessment','assessment'),('merge_class','n2:N2:cache')}
        self.roles.update(('input_pin',f'input:{i}') for i in range(216));self.roles.update(('postcheck',f'postcheck:{i}') for i in range(216))
        self.roles.update(('continuous_sample_view',f'sample:{i}') for i in range(180));self.roles.update(('sample_class',f'n{n}:N{N}:CAR') for n in (2,3) for N in range(4))
        self.roles.update(('missing_matrix_price',f'price:{i}') for i in range(199))
        self.input_indices={};self.secondary_failures=[];self.roster_resolved=False
        self.write('pricing','logical_role_plan','logical_role_plan',{'ordered_roles':[list(v) for v in sorted(self.roles)],'unresolved_remaining_identity_count':199,'unresolved_continuous_sample_count':180,'remaining_identity_status':'not_admitted'})
        self.stages=[]
    def write(self,stream,kind,name,payload,operands=()):
        row={'sequence':self.seq+1,'type':kind,'object_id':name,'payload':payload,'operand_refs':list(operands)};data=dump(row)
        if self.bytes+len(data)>CAP:raise Interrupted('48-MiB producer metadata output limit')
        self.bytes+=len(data)  # Conservative attempted-byte reservation is never refunded.
        f=self.streams[stream];f.write(data);f.flush();os.fsync(f.fileno());self.seq+=1
        ref=rowref(self.out/(stream+'.jsonl'),data,row);self.done.add((kind,name));return ref
    def atomic(self,name,value):
        data=dump(value)
        if self.bytes+len(data)>CAP:raise Interrupted('48-MiB producer metadata output limit')
        self.bytes+=len(data)
        path=self.out/name;temp=path.with_name(name+'.partial.'+str(time.monotonic_ns()))
        with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        fd=os.open(self.out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def admit(rec):
    rec.expected=loads(os.environ['VACUUM_U2_REVIEWED_PINS'])
    if len(rec.expected)!=216 or len({p['path'] for p in rec.expected})!=216:raise ValueError('216 distinct current input pins')
    rec.write('admission','input_plan','input_plan',rec.expected)
    documents={}
    for i,expected in enumerate(rec.expected):
        boundary();path=Path(expected['path']);rec.visited.add(str(path));actual=None
        try:
            actual,data=inspect(path,expected['sha256'],True)
            rec.write('admission','input_pin',f'input:{i}',{'expected':expected,'returned':actual,'matched':actual==expected})
            if actual!=expected:
                exc=ValueError('guardian-worker input descriptor join');exc.observation={'expected':expected,'returned':actual};raise exc
            rec.admitted[str(path)]=actual;rec.input_indices[str(path)]=i
            if path.parent in (OLD,NEW) and path.suffix in ('.json','.jsonl'):documents[str(path)]=data
        except BaseException as exc:
            item={'expected':expected,'returned':actual,'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
            rec.failed.append(item)
            try:rec.write('admission','failed_input','failed:'+str(i),item)
            except BaseException as secondary:rec.secondary_failures.append({'phase':'admission_failure_journal_write','type':type(secondary).__name__,'message':str(secondary),'observation':getattr(secondary,'observation',None)})
            raise
    old=loads(documents[str(OLD/'execution.json')]);new=loads(documents[str(NEW/'execution.json')])
    for obj,root,count,rc in ((old,OLD,171,1),(new,NEW,24,0)):
        if obj['returncode']!=rc or obj['errors'] or any(obj[k] is not True for k in ('terminal_established','child_reaped','inventory_complete','owned_sources_unchanged')):raise ValueError('terminal prefix custody')
        if len(obj['inventory'])!=count or obj['missing_fixed_files'] or obj.get('missing_source_admissions') or obj.get('missing_owned_source_postchecks') or obj.get('missing_inventory_files'):raise ValueError('prefix incomplete custody metadata')
        for pin in obj['inventory']:
            actual=rec.admitted[pin['path']]
            if pin['unfinished_temporary'] or any(pin[k]!=actual[k] for k in actual):raise ValueError('current prefix inventory join')
        for role,pin in obj['source_pins'].items():
            if rec.admitted[pin['path']]!=pin or obj['owned_source_postchecks'][role]['matched'] is not True:raise ValueError('prefix source/postcheck join')
    if len(new['source_pins'])!=188 or new.get('missing_dynamic_matrix_files') or new.get('failed_inventory_observations'):raise ValueError('new complete prefix cut')
    final=loads(documents[str(NEW/'vacuum_ref_u2_prefix_missing_result.json')])
    if any(final[k] is not True for k in ('terminal_checkpoint','target_complete','scientific_missing_cut_complete','combined_prefix_mathematical_complete')) or final['full_package_complete'] is not False or final['balance_test_run'] is not False:raise ValueError('prefix scope flags')
    rows={};references={};kinds={};line_metadata=[]
    for root,names in ((OLD,('admission','coefficients','aliases','metrics','norms')),(NEW,('admission','metrics','norms'))):
        for name in names:
            path=root/(name+'.jsonl')
            for line in documents[str(path)].splitlines(keepends=True):
                boundary();row=loads(line);seq=row['sequence']
                if type(seq) is not int or seq in rows:raise ValueError('unique global journal sequence')
                rows[seq]=row;references[seq]=rowref(path,line,row);kinds.setdefault(row['type'],[]).append(row)
                if name=='admission' or row['type'] in ('sector_matrix','sector_norm','complete_matrix_norm_unit','segmented_matrix_norm_completion','face_stack_norm'):line_metadata.append((len(line),references[seq]))
    if sorted(rows)!=list(range(1,2729)):raise ValueError('2728 contiguous closed records')
    for row in rows.values():
        validate_times(row)
        for ref in refs(row):
            if references.get(ref['sequence'])!=ref:raise ValueError('direct record reference')
    expected_counts={'public_alias':199,'canonical_body':325,'canonical_product':194,'canonical_audit':519,'canonical_operation':519,'sector_basis':8,'candidate_work_plan':2,'sector_matrix':181,'sector_norm':181,'complete_matrix_norm_unit':180,'segmented_matrix_norm_completion':1,'face_stack_norm':8}
    if any(len(kinds.get(k,[]))!=v for k,v in expected_counts.items()):raise ValueError('exact combined prefix role counts')
    if any(r['payload']['number_conserved'] is not True or r['payload']['violations'] or r['payload']['max_rank']>4 for r in kinds['canonical_audit']):raise ValueError('inherited number/rank audit failed')
    body={r['object_id']:r for r in kinds['canonical_body']+kinds['canonical_product']};aliases={};reverse={};aliasrefs={};canonicalrefs={}
    for row in kinds['public_alias']:
        v=row['payload'];n=v['parent_sites'];label=v['logical_name'];digest=v['canonical_sha256']
        if body[f'n{n}:'+v['canonical_body']]['payload']['canonical_sha256']!=digest:raise ValueError('alias canonical join')
        aliases[n,label]=digest;reverse.setdefault((n,digest),[]).append(label);aliasrefs[n,label]=references[row['sequence']]
        original=body[f'n{n}:'+v['canonical_body']];resolved=original
        while 'canonical_ref' in resolved['payload']:resolved=rows[resolved['payload']['canonical_ref']['sequence']]
        canonicalrefs[n,label]=[references[original['sequence']]]+([references[resolved['sequence']]] if resolved['sequence']!=original['sequence'] else [])
    matrices={};norms={};units={};plans={};work={};dims={}
    for row in kinds['sector_matrix']:
        v=row['payload'];m=v['matrix'];key=m['parent_sites'],m['sector'],m['canonical_sha256']
        if key in matrices:raise ValueError('duplicate matrix identity')
        actual=rec.admitted[m['path']]
        if actual['sha256']!=m['sha256'] or actual['size']!=m['size']:raise ValueError('matrix metadata/input identity')
        matrices[key]=row
    for kind,dest in (('sector_norm',norms),('complete_matrix_norm_unit',units)):
        for row in kinds[kind]:
            fields=row['object_id'].split(':');key=int(fields[0][1:]),int(fields[1][1:]),fields[-1]
            if key not in matrices or key in dest:raise ValueError('matrix/norm/unit identity join')
            dest[key]=row
    for row in kinds['candidate_work_plan']:
        v=row['payload'];n=v['parent_sites'];plans[n]=row;dims[n]=v['sector_dimensions']
        if len(v['public_plan'])!={2:70,3:129}[n] or set(v['public_plan'])!={label for parent,label in aliases if parent==n}:raise ValueError('fixed public plan')
        for label,counts in v['per_public_body'].items():
            for sector,w in counts.items():
                if type(w) is not int or w<0:raise ValueError('native nonnegative work metadata')
                key=n,int(sector),aliases[n,label]
                if key in work and work[key]!=w:raise ValueError('aliased candidate counts disagree')
                work[key]=w
    full={(n,N,digest) for n,digest in reverse for N in range(4)};missing=full-set(matrices)
    if len(full)!=380 or len(missing)!=199 or set(norms)!=set(matrices):raise ValueError('380 full /181 prefix /199 absent')
    for n,counts in ((2,(13,13,12,13)),(3,(37,37,37,37))):
        if tuple(sum(key[:2]==(n,N) for key in missing) for N in range(4))!=counts:raise ValueError('exact remaining sector roster')
    context={'old':old,'new':new,'rows':rows,'refs':references,'kinds':kinds,'aliases':aliases,'reverse':reverse,'aliasrefs':aliasrefs,'matrices':matrices,'norms':norms,'units':units,'plans':plans,'work':work,'dims':dims,'full':full,'missing':missing,'canonicalrefs':canonicalrefs,'line_metadata':line_metadata}
    rec.write('pricing','reuse_index','reuse_index',{'full_identities':380,'prefix_matrix_norm_pairs':181,'remaining':199,'continuous_units':180,'excluded_segmented_refs':[references[r['sequence']] for r in kinds['segmented_matrix_norm_completion']],'inherited_matrices':[references[r['sequence']] for r in matrices.values()],'inherited_norms':[references[r['sequence']] for r in norms.values()],'inherited_stacks':[references[r['sequence']] for r in kinds['face_stack_norm']],'no_coefficient_or_matrix_evaluation':True})
    rec.stages.extend(['admission','reuse_index']);return context
def price(rec,c):
    boundary();samples={};class_refs={};table={};basisrefs={(r['payload']['parent_sites'],r['payload']['sector']):c['refs'][r['sequence']] for r in c['kinds']['sector_basis']}
    cache=c['kinds']['selected_cache'][0];cache_inputs=cache['payload']['reuse_refs']
    if tuple(cache_inputs[name]['nnz'] for name in ('W','W0','W1'))!=(40,20,20):raise ValueError('cached matrix nnz metadata')
    p1key=2,2,c['aliases'][2,'P1[0]'];p1=c['matrices'][p1key];p1meta=p1['payload']['matrix'];p1nnz=p1['payload']['nnz']
    p1input={'matrix_receipt_ref':c['refs'][p1['sequence']],'matrix_identity':p1meta,'nnz':p1nnz}
    exceptions={c['aliases'][2,'V[f]']:('cached_W_reference_norm_only',60,40,[cache_inputs['W']]),c['aliases'][2,'V[n1]']:('cached_W_plus_W1_merge',60,60,[cache_inputs['W'],cache_inputs['W1']]),c['aliases'][2,'V[v0]']:('cached_W_plus_P1_merge',40+p1nnz,40+p1nnz,[cache_inputs['W'],p1input])}
    if len(exceptions)!=3 or c['aliases'][2,'V[v0]']!=c['aliases'][2,'V[v1]']:raise ValueError('three cache identity dispositions')
    roster=[]
    for index,(n,N,digest) in enumerate(sorted(c['missing'])):
        labels=sorted(c['reverse'][n,digest]);w=c['work'][n,N,digest];mode,charged,actual,inputs=exceptions[digest] if (n,N)==(2,2) and digest in exceptions else ('CAR_spectator_assembly',w,w,[])
        roster.append({'role_id':f'price:{index}','parent_sites':n,'sector':N,'canonical_sha256':digest,'logical_aliases':labels,'candidate_work':w,'candidate_work_all_sectors':[c['work'][n,k,digest] for k in range(4)],'shape':[c['dims'][n][N]]*2,'basis_ref':basisrefs[n,N],'alias_refs':[c['aliasrefs'][n,label] for label in labels],'canonical_identity_refs':[ref for label in labels for ref in c['canonicalrefs'][n,label]],'candidate_plan_ref':c['refs'][c['plans'][n]['sequence']],'operation_class':mode,'actual_class_work':actual,'charged_class_work':charged,'direct_cache_inputs':inputs,'scientific_assembly_required':mode!='cached_W_reference_norm_only','requires_norm':True,'norm_already_present':False,'intended_filename':None if mode=='cached_W_reference_norm_only' else f'n{n}_N{N}_{digest}.json'})
    c['remaining_roster']=roster
    c['rolemap_ref']=rec.write('pricing','logical_role_plan','admitted_operation_role_map',{'remaining_roster_before_derived_prices':roster,'input_plan':rec.expected,'roster_resolved':True})
    rec.roster_resolved=True;boundary()
    c['applicability_ref']=rec.write('pricing','source_applicability','source_applicability',{'old_source_pins':c['old']['source_pins'],'new_source_pins':c['new']['source_pins'],'old_and_new_CAR_costs_retained_with_source_labels':True,'no_optimized_implementation_timing_claim':True,'segmented_clock_excluded':True})
    rec.stages.append('applicability')
    for sample_index,(key,row) in enumerate(sorted(c['units'].items())):
        boundary();v=row['payload'];m=c['matrices'][key]['payload'];duration=cpu(v['inclusive_total_including_norm_before_unit_metric_write'])
        if 'rate_sample_eligible' in v and v['rate_sample_eligible'] is not True:raise ValueError('ineligible continuous sample')
        if v['candidate_work']!=c['work'][key] or v['matrix_ref']['path']!=m['matrix']['path']:raise ValueError('unit/matrix candidate join')
        source_label='old' if Path(c['refs'][row['sequence']]['file']).parent==OLD else 'new'
        label=v['logical_aliases'][0]
        value={'key':list(key),'work':v['candidate_work'],'nnz':m['nnz'],'method':v['method'],'CPU':pair(duration),'CPU_original':v['inclusive_total_including_norm_before_unit_metric_write'],'unit_ref':c['refs'][row['sequence']],'matrix_ref':c['refs'][c['matrices'][key]['sequence']],'norm_ref':c['refs'][c['norms'][key]['sequence']],'matrix_bytes':m['matrix']['size'],'source_label':source_label,'producer_pin':c[source_label]['source_pins']['producer'],'source_applicability_ref':c['applicability_ref'],'basis_ref':basisrefs[key[:2]],'canonical_identity_refs':c['canonicalrefs'][key[0],label],'norm_operation_receipt':c['norms'][key]['operation_receipt'],'lagged_writer_receipt':row.get('previous_completed_writer_receipt'),'lagged_checkpoint_receipt':row.get('previous_completed_checkpoint_receipt'),'journal_writer_totals_before_unit':row.get('journal_writer_totals_before_this_write'),'checkpoint_totals_before_unit':row.get('checkpoint_totals_before_this_write'),'cost_subintervals':{name:value for name,value in m.items() if isinstance(value,dict) and 'user_seconds' in value}}
        view_ref=rec.write('pricing','continuous_sample_view',f'sample:{sample_index}',value,[value['unit_ref'],value['matrix_ref'],value['norm_ref'],c['applicability_ref']])
        value.update(sample_view_ref=view_ref,duration_internal=duration);samples.setdefault(key[:2],[]).append(value)
    if sum(map(len,samples.values()))!=180:raise ValueError('180 continuous sample views')
    boundary();segmented=c['kinds']['segmented_matrix_norm_completion'][0]
    rec.write('pricing','excluded_segmented_view','excluded_segmented_view',{'original_ref':c['refs'][segmented['sequence']],'payload_verbatim':segmented['payload'],'rate_sample_eligible':False})
    rec.stages.append('sample_views')
    for n in (2,3):
        for N in range(4):
            boundary();values=samples[n,N];empty=[v for v in values if v['nnz']==0];positive=[v for v in values if v['work']>0 and v['method'].startswith('CAR_spectator_assembly')]
            remaining=[row for row in roster if (row['parent_sites'],row['sector'])==(n,N)]
            if not empty:raise ValueError('missing empty-unit sample')
            h=max(v['duration_internal'] for v in empty);need_positive=any(row['candidate_work']>0 for row in remaining if row['operation_class']=='CAR_spectator_assembly')
            if need_positive and not positive:raise ValueError('missing positive CAR sample')
            rate=max(v['duration_internal']/v['work'] for v in positive) if need_positive else Fraction(0)
            byte_rate=max(Fraction(v['matrix_bytes'],v['work']) for v in positive) if need_positive else Fraction(0)
            b0=max(v['matrix_bytes'] for v in empty);rss=max(int(v['CPU_original']['peak_RSS_KiB']) for v in values);ws=max(v['work'] for v in values);wm=max(row['candidate_work'] for row in remaining)
            resource={'parent_sites':n,'sector':N,'shape':[c['dims'][n][N]]*2,'measured_peak_RSS_KiB':rss,'RSS_maximizer_sample_refs':[v['sample_view_ref'] for v in values if int(v['CPU_original']['peak_RSS_KiB'])==rss],'max_sampled_candidate_work':ws,'sample_work_maximizer_refs':[v['sample_view_ref'] for v in values if v['work']==ws],'max_missing_candidate_work':wm,'missing_work_maximizers':[row for row in remaining if row['candidate_work']==wm],'remaining_rolemap_ref':c['rolemap_ref'],'memory_forecast_KiB':pair(Fraction(rss)*max(Fraction(1),Fraction(wm,max(1,ws))))}
            out={'parent_sites':n,'sector':N,'dimension':c['dims'][n][N],'h':pair(h),'h_tie_refs':[v['unit_ref'] for v in empty if v['duration_internal']==h],'r':pair(rate),'r_tie_refs':[v['unit_ref'] for v in positive if need_positive and v['duration_internal']/v['work']==rate],'matrix_bytes_per_work':pair(byte_rate),'byte_rate_tie_refs':[v['matrix_ref'] for v in positive if need_positive and Fraction(v['matrix_bytes'],v['work'])==byte_rate],'max_empty_file_bytes':b0,'empty_byte_tie_refs':[v['matrix_ref'] for v in empty if v['matrix_bytes']==b0],'resource_forecast':resource,'sample_refs':[v['sample_view_ref'] for v in values]}
            class_refs[n,N]=rec.write('pricing','sample_class',f'n{n}:N{N}:CAR',out,out['sample_refs']+[c['rolemap_ref']])
            table[n,N]={'h':h,'r':rate,'byte_rate':byte_rate,'row':out,'samples':values,'resource':resource}
    rec.stages.append('samples');boundary()
    merge=c['rows'][1882];mv=merge['payload'];merge_duration=cpu(mv['inclusive_total_including_norm_before_unit_metric_write'])
    if merge['type']!='complete_matrix_norm_unit' or mv['merge_input_nnz_sum']!=60 or c['rows'][1880]['payload']['matrix']['size']!=588:raise ValueError('fixed merge sample')
    merge_rate=merge_duration/60
    merge_ref=rec.write('pricing','merge_class','n2:N2:cache',{'sample_ref':c['refs'][1882],'matrix_ref':c['refs'][1880],'norm_ref':c['refs'][1881],'CPU':pair(merge_duration),'input_nnz_sum':60,'rate':pair(merge_rate),'matrix_bytes':588,'cache_refs':cache_inputs,'P1_input':p1input,'source_applicability_ref':c['applicability_ref']},[c['refs'][1882],c['refs'][p1['sequence']],c['applicability_ref']])
    priced=[]
    for row in roster:
        boundary();n,N=row['parent_sites'],row['sector'];t=table[n,N];w=row['candidate_work'];mode=row['operation_class'];charged=row['charged_class_work']
        operands=[class_refs[n,N],row['candidate_plan_ref'],row['basis_ref'],c['rolemap_ref']]+row['alias_refs']+row['canonical_identity_refs']+row['direct_cache_inputs']
        if mode!='CAR_spectator_assembly':
            estimate=4*(t['h']+merge_rate*max(1,charged));byte_base=Fraction(588,60)*max(1,charged);operands.append(merge_ref)
        else:estimate=4*(t['h']+t['r']*max(1,w));byte_base=t['byte_rate']*max(1,w)
        header=0 if row['intended_filename'] is None else len(dump({'shape':row['shape'],'denominator':1,'entries':[],'logical_aliases':row['logical_aliases']}))+len(row['intended_filename'].encode())
        output=0 if mode=='cached_W_reference_norm_only' else math.ceil(4*(t['row']['max_empty_file_bytes']+byte_base))+header
        item={**row,'CPU_forecast':pair(estimate),'matrix_bytes_base_forecast':pair(byte_base if mode!='cached_W_reference_norm_only' else Fraction(0)),'matrix_bytes_forecast':output,'literal_header_filename_alias_allowance':header,'no_matrix_file':mode=='cached_W_reference_norm_only','operand_refs':operands}
        ref=rec.write('pricing','missing_matrix_price',row['role_id'],item,operands);item['ref']=ref;priced.append((item,estimate))
    if len(priced)!=199 or sum(v['operation_class']=='CAR_spectator_assembly' for v,cost in priced)!=196:raise ValueError('196 CAR/two merges/one direct W norm')
    rec.stages.append('roster');return table,priced,class_refs
def aggregate(rec,c,table,priced,class_refs):
    boundary();per_matrix=[{'price_ref':item['ref'],'matrix_bytes_forecast':item['matrix_bytes_forecast'],'literal_header_filename_alias_allowance':item['literal_header_filename_alias_allowance'],'no_matrix_file':item['no_matrix_file']} for item,cost in priced]
    matrix_forecast=sum(v['matrix_bytes_forecast'] for v in per_matrix)
    mem=[{**table[key]['resource'],'class_ref':class_refs[key]} for key in sorted(table)]
    peaks=[{'attempt':name,'kind':field,'peak_RSS_KiB':int(c[name][field]),'execution_pin':rec.admitted[str((OLD if name=='old' else NEW)/'execution.json')]} for name in ('old','new') for field in ('child_peak_RSS_KiB','wrapper_peak_RSS_KiB')]
    memory=max([Fraction(v['peak_RSS_KiB']) for v in peaks]+[Fraction(*v['memory_forecast_KiB']) for v in mem])
    L=max(length for length,ref in c['line_metadata']);Lrefs=[ref for length,ref in c['line_metadata'] if length==L]
    checkpoints=[rec.admitted[str(OLD/'vacuum_ref_u2_v2_result.json')],rec.admitted[str(NEW/'vacuum_ref_u2_prefix_missing_result.json')]];C=max(v['size'] for v in checkpoints);Cties=[v for v in checkpoints if v['size']==C]
    metadata_allowance=4*((3*199+2*216+64)*L+(199+2)*C)
    header_object={'expected_pins':rec.expected,'remaining_operation_roster':c['remaining_roster']};final_plans_header=len(dump(header_object))
    existing_outputs=sum(v['size'] for path,v in rec.admitted.items() if Path(path).parent in (OLD,NEW,OLD/'matrices',NEW/'matrices'))
    output_forecast=matrix_forecast+metadata_allowance+final_plans_header
    resource_reasons=[]
    if memory*1024>4*1024**3:resource_reasons.append('memory_forecast_exceeds_4_GiB')
    if output_forecast>8*1024**3:resource_reasons.append('output_forecast_exceeds_8_GiB')
    resource={'matrix_forecasts':per_matrix,'sector_resource_forecasts':mem,'memory_forecast_KiB':pair(memory),'memory_maximizer_sector_refs':[v['class_ref'] for v in mem if Fraction(*v['memory_forecast_KiB'])==memory],'whole_attempt_peak_inputs':peaks,'memory_maximizer_whole_inputs':[v for v in peaks if v['peak_RSS_KiB']==memory],'output_forecast_bytes':output_forecast,'new_matrix_forecast_bytes':matrix_forecast,'metadata_written_bytes_allowance':metadata_allowance,'metadata_L_bytes':L,'metadata_L_tie_refs':Lrefs,'checkpoint_C_bytes':C,'checkpoint_C_tie_pins':Cties,'checkpoint_C_inputs':checkpoints,'literal_final_plan_header_bytes':final_plans_header,'literal_header_plan_ref':c['rolemap_ref'],'existing_owned_output_bytes':existing_outputs,'existing_output_is_occupancy_not_new_write_charge':True,'resource_stop_report_reasons':resource_reasons,'forecast_is_not_certified_bound':True}
    resource_ref=rec.write('pricing','resource_forecast','resource_forecast',resource,[item['ref'] for item,cost in priced]+list(class_refs.values())+Lrefs+Cties+[c['rolemap_ref']])
    rec.stages.append('resources');boundary()
    total=sum((cost for item,cost in priced),Fraction(0));soft=math.ceil(Fraction(5,4)*total+30);hard=soft+15
    spent=sum((Fraction(str(c[k][q])) for k in ('old','new') for q in ('complete_attempt_user_seconds','complete_attempt_system_seconds')),Fraction(0));provisional_spend=spent+4+159
    hard_aggregate=provisional_spend+hard+60;forecast_aggregate=provisional_spend+total;reasons=list(resource_reasons)
    if hard_aggregate>7200:reasons.append('proposed_hard_aggregate_exceeds_7200_CPU_seconds')
    if forecast_aggregate>7200:reasons.append('scheduling_forecast_exceeds_7200_CPU_seconds')
    spend={'sum_missing_CPU_forecasts':pair(total),'proposed_finish_CPU_soft_hard':[soft,hard],'proposed_finish_external_handled_wall':[hard+60,hard+45],'prefix_actual_CPU':pair(spent),'old_guardian_final_write_reserves_CPU':4,'pricing_reserved_worker_and_guardian_CPU':159,'pricing_complete_actual_cost_requires_terminal_execution_record':True,'provisional_hard_aggregate_CPU':pair(hard_aggregate),'provisional_forecast_aggregate_CPU':pair(forecast_aggregate),'finish_guardian_metadata_CPU_allowance':60,'finish_eligibility':'STOP_REPORT' if reasons else 'PROVISIONAL_PENDING_TERMINAL_PRICING_COST_AND_FINISH_SOURCE','stop_report_reasons':reasons,'omitted_required_operation_classes':[]}
    spend_ref=rec.write('pricing','spend_allowance','spend_allowance',spend,[item['ref'] for item,cost in priced]+[rec.admitted[str(OLD/'execution.json')],rec.admitted[str(NEW/'execution.json')],resource_ref])
    rec.stages.append('spend');boundary()
    value={**resource,**spend,'resource_forecast_ref':resource_ref,'spend_allowance_ref':spend_ref,'unperformed_scientific_matrix_files':198,'unperformed_scientific_norms':199,'existing_W_reference_reused':True,'full_package_complete':False,'balance_test_run':False,'closure':'not_established'}
    rec.write('pricing','aggregate','aggregate',value,[resource_ref,spend_ref]);return value

def main():
    global PHASE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--outdir',type=Path,required=True);args=parser.parse_args()
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU):signal.signal(sig,handler)
    launch=float(os.environ['VACUUM_U2_CHILD_LAUNCH']);signal.setitimer(signal.ITIMER_REAL,max(.001,55-(time.monotonic()-launch)))
    rec=Record(args.outdir);budget={'pricing_complete':False,'full_package_complete':False,'balance_test_run':False,'closure':'not_established','finish_eligibility':'not_established','budget_is_not_a_launch_instruction':True};c=None
    try:
        limits=loads(os.environ['VACUUM_U2_LIMITS']);runtime={'executable':sys.executable,'affinity':sorted(os.sched_getaffinity(0)),'nice':os.getpriority(os.PRIO_PROCESS,0),'CPU_limits':list(resource.getrlimit(resource.RLIMIT_CPU)),'address_space_limits':list(resource.getrlimit(resource.RLIMIT_AS)),'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS','CUDA_VISIBLE_DEVICES')},'resident_threads':len(list(Path('/proc/self/task').iterdir()))}
        rec.write('admission','runtime','runtime',runtime)
        st=os.statvfs(ROOT);free=st.f_bavail*st.f_frsize
        rec.write('admission','disk','disk',{'available_bytes':free,'required_bytes':16*1024**3})
        if free<16*1024**3:raise ValueError('metadata disk minimum')
        if len(runtime['affinity'])!=1 or runtime['nice']<10 or runtime['resident_threads']!=1 or runtime['CPU_limits']!=[45,50] or runtime['address_space_limits']!=[1024**3]*2 or any(v!='1' for k,v in runtime['environment'].items() if k!='CUDA_VISIBLE_DEVICES') or runtime['environment']['CUDA_VISIBLE_DEVICES']!='':raise ValueError('metadata runtime limits')
        c=admit(rec);table,priced,class_refs=price(rec,c)
        budget.update(aggregate(rec,c,table,priced,class_refs));rec.stages.append('aggregate');boundary()
    except BaseException as exc:
        rec.error={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None),'traceback':traceback.format_exc()}
        try:rec.write('pricing','failure','failure',rec.error)
        except BaseException as secondary:rec.secondary_failures.append({'phase':'primary_failure_journal_write','type':type(secondary).__name__,'message':str(secondary),'observation':getattr(secondary,'observation',None)})
    finally:
        PHASE='terminal_metadata';signal.setitimer(signal.ITIMER_REAL,max(.001,min(3,60-(time.monotonic()-launch)-2)))
        try:
            for path,old in rec.admitted.items():
                check={'expected':old,'matched':False};rec.postchecks[path]=check
                try:
                    now=inspect(path,old['sha256']);check['returned']=now
                    if now!=old:raise ValueError('postcheck descriptor mismatch')
                    check['matched']=True
                except BaseException as exc:
                    check['failure']={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                    if isinstance(exc,Interrupted):raise
                rec.write('admission','postcheck',f'postcheck:{rec.input_indices[path]}',check)
            rec.stages.append('postchecks')
        except BaseException as exc:
            budget['terminal_metadata_failure']={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
        finally:
            PHASE='final_budget';final_deadline=min(time.monotonic()+2,launch+60)
            signal.setitimer(signal.ITIMER_REAL,max(.001,final_deadline-time.monotonic()))
        budget.update(error=rec.error,signals=SIGNALS,expected_input_plan=rec.expected,failed_input_observations=rec.failed,unvisited_source_admissions=[p for p in rec.expected if p['path'] not in rec.visited],unvisited_postchecks_not_admitted=[p for p in rec.expected if p['path'] not in rec.admitted],missing_sample_classes=[f'n{n}:N{N}:CAR' for n in (2,3) for N in range(4) if ('sample_class',f'n{n}:N{N}:CAR') not in rec.done],missing_source_admissions=[p for p in rec.expected if p['path'] not in rec.admitted],missing_postchecks=[p for p in rec.admitted if p not in rec.postchecks],failed_postchecks=[v for v in rec.postchecks.values() if not v['matched']],completed_metadata_stages=rec.stages,missing_metadata_stages=[s for s in rec.stage_plan if s not in rec.stages],roster_resolved=rec.roster_resolved,unresolved_remaining_identity_count=0 if rec.roster_resolved else 199,secondary_failures=rec.secondary_failures,attempted_producer_bytes_reserved_before_assessment_and_budget=rec.bytes,returned_price_rows=[list(v) for v in sorted(rec.done) if v[0]=='missing_matrix_price'],missing_price_rows=[f'price:{i}' for i in range(199) if ('missing_matrix_price',f'price:{i}') not in rec.done],no_automatic_retry=True,metadata_only=True)
        try:
            rec.write('pricing','assessment','assessment',{'metadata_error':rec.error,'missing_roles_before_assessment':[list(v) for v in sorted(rec.roles-rec.done-{('assessment','assessment')})],'unperformed_science':True})
            rec.stages.append('assessment')
        except BaseException as exc:budget['assessment_failure']={'type':type(exc).__name__,'message':str(exc)}
        budget['missing_metadata_stages']=[s for s in rec.stage_plan if s not in rec.stages]
        budget['missing_roles']=[list(v) for v in sorted(rec.roles-rec.done)]
        budget['pricing_complete']=rec.error is None and not budget.get('assessment_failure') and not budget['missing_roles'] and not budget.get('terminal_metadata_failure') and not budget['missing_metadata_stages'] and not budget['missing_source_admissions'] and not budget['missing_postchecks'] and not budget['failed_postchecks'] and not budget['missing_price_rows']
        if not budget['pricing_complete']:budget['finish_eligibility']='not_established'
        try:
            remaining=final_deadline-time.monotonic()
            if remaining>0:
                signal.setitimer(signal.ITIMER_REAL,remaining)
                rec.atomic('measured_budget.json',budget)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for f in rec.streams.values():f.close()
    return 0 if budget['pricing_complete'] else 2
if __name__=='__main__':sys.exit(main())
