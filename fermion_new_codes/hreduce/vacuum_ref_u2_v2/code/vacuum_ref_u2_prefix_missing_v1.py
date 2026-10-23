#!/usr/bin/env python3
"""Absent-only completion of fixed Step-1 prefix; no balance expression/test."""
import argparse
from fractions import Fraction as F
import hashlib
import itertools
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
OLD_EXEC_SHA='c1e05134da0aa24a46821d9b6912c38d3574a2b190f6a68e32733fdff334412a'
LAST='7a1281744f655a5ab569712b4580cc5c0ba5c6cf45708e607726a9d8df36bfcb'
TOP_FILES=['admission.jsonl','metrics.jsonl','norms.jsonl','vacuum_ref_u2_prefix_missing_result.json','execution.json','stdout.log','stderr.log']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
STOP=[]
PHASE='scientific'
ENTRY=time.monotonic()
Z=(F(0),F(0))
class Stopped(Exception):pass
def handler(sig,frame):
    STOP.append(int(sig))
    if PHASE!='scientific':raise Stopped('interruptible '+PHASE+' signal '+str(sig))
def boundary():
    if STOP:raise Stopped('handled signals '+str(STOP))
def addz(a,b):return (a[0]+b[0],a[1]+b[1])


def put(d,key,z):
    v=addz(d.get(key,Z),z)
    if v!=Z:d[key]=v
    else:d.pop(key,None)

def native(v):
    if type(v) is not int:raise ValueError('native integer required')
    return v

def nodup(rows):
    d={}
    for k,v in rows:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d

def finitefloat(s):
    x=float(s)
    if not math.isfinite(x):raise ValueError('float overflow')
    return x

def badconstant(s):raise ValueError('nonfinite JSON token '+s)

def loads(b):return json.loads(b,object_pairs_hook=nodup,parse_constant=badconstant,parse_float=finitefloat)

def dump(obj):return (json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def resource_failures(obj):
    if isinstance(obj,dict):
        out=[obj['resource_observation_failure']] if 'resource_observation_failure' in obj else []
        for key,value in obj.items():
            if key!='resource_observation_failure':out.extend(resource_failures(value))
        return out
    if isinstance(obj,(list,tuple)):
        return [failure for value in obj for failure in resource_failures(value)]
    return []

def identity(st):return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}

def readpin(path,expected):
    observation={'path':str(path),'expected_sha256':expected,'phase':'resolve','bytes_read':0}
    try:
        resolved=str(path.resolve(strict=True));observation['resolved_path']=resolved
        with path.open('rb') as f:
            before=identity(os.fstat(f.fileno()));observation['before']=before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular file required')
            observation['phase']='read';data=f.read();observation['bytes_read']=len(data)
            after=identity(os.fstat(f.fileno()));observation['after']=after
        current=identity(path.stat());observation['path_now']=current
        now=str(path.resolve(strict=True));observation['resolved_path_now']=now
        digest=hashlib.sha256(data).hexdigest();observation['sha256']=digest;observation['phase']='compare'
        if before!=after or before!=current or resolved!=now or digest!=expected:raise ValueError('unstable/mismatched input '+str(path))
        return data,{'path':str(path),'resolved_path':resolved,'sha256':digest,**before}
    except BaseException as exc:
        exc.observation=observation;raise

def usage():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return {'wall':time.monotonic(),'user':r.ru_utime,'system':r.ru_stime,'peak_RSS_KiB':r.ru_maxrss}

def currentrss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])
    raise ValueError('missing current RSS')

def receipt(before):
    try:
        after=usage()
        out={'wall_seconds':after['wall']-before['wall'],'user_seconds':after['user']-before['user'],
             'system_seconds':after['system']-before['system'],'peak_RSS_KiB':after['peak_RSS_KiB']}
        try:out['current_RSS_KiB']=currentrss()
        except BaseException as exc:
            if isinstance(exc,Stopped):raise
            out['resource_observation_failure']={'type':type(exc).__name__,'message':str(exc)}
        return out
    except BaseException as exc:
        if isinstance(exc,Stopped):raise
        return {'resource_observation_failure':{'type':type(exc).__name__,'message':str(exc)}}

def assemble(a,m,sector,basis):
    index={v:i for i,v in enumerate(basis)};out={}
    for (ii,jj),z in a.items():
        count=sector-len(ii)
        if count<0:continue
        available=[x for x in range(m) if x not in ii and x not in jj]
        for spectators in itertools.combinations(available,count):
            source=sum(1<<x for x in jj+spectators);target=source;sgn=1
            for x in jj:
                sgn*=(-1 if (target&((1<<x)-1)).bit_count()%2 else 1);target^=1<<x
            for x in reversed(ii):
                sgn*=(-1 if (target&((1<<x)-1)).bit_count()%2 else 1);target^=1<<x
            put(out,(index[target],index[source]),(sgn*z[0],sgn*z[1]))
    return out

def matrixpayload(a,dim,aliases):
    den=math.lcm(*(v.denominator for z in a.values() for v in z)) if a else 1
    rows=[[i,j,int(z[0]*den),int(z[1]*den)] for (i,j),z in sorted(a.items())]
    g=math.gcd(den,*(abs(v) for row in rows for v in row[2:])) if rows else 1
    return {'shape':[dim,dim],'denominator':den//g,'entries':[[i,j,r//g,s//g] for i,j,r,s in rows],'logical_aliases':sorted(aliases)}

class Record:
    def __init__(self,out):
        self.out=out;self.sequence=2291;self.bytes=0;self.pins=[];self.postchecks=[];self.error=None;self.metadata_failures=[]
        self.streams={name:(out/(name+'.jsonl')).open('xb') for name in ('admission','metrics','norms')}
        (out/'matrices').mkdir()
        self.plan=set();self.done=set();self.matrix_refs={};self.norm_refs={};self.unit_refs={};self.segmented=None;self.stack=None
        self.writer_previous=None;self.writer_totals={'user_seconds':0.,'system_seconds':0.,'wall_seconds':0.,'bytes_written':0}
        self.checkpoint_previous=None;self.checkpoint_totals={'user_seconds':0.,'system_seconds':0.,'wall_seconds':0.,'bytes_written':0};self.generation=0
        self.expected_pins=[];self.visited_admissions=set();self.admitted_paths=set();self.failed_admissions=[]
    def expect(self,stream,kind,name):self.plan.add((stream,kind,name))
    def write(self,stream,kind,name,payload,refs=(),measure=None):
        before=usage()
        for failure in resource_failures([payload,measure]):self.metadata_failures.append({'object_id':name,'failure':failure})
        row={'sequence':self.sequence+1,'type':kind,'object_id':name,'payload':payload,'operand_refs':list(refs),'units':'a=1; fixed on-site Step-1 coefficient','operation_receipt':measure,'previous_completed_writer_receipt':self.writer_previous,'journal_writer_totals_before_this_write':dict(self.writer_totals),'previous_completed_checkpoint_receipt':self.checkpoint_previous,'checkpoint_totals_before_this_write':dict(self.checkpoint_totals)}
        data=dump(row)
        if self.bytes+len(data)>8*1024**3:raise Stopped('8-GiB cumulative output limit')
        f=self.streams[stream];f.write(data);f.flush();os.fsync(f.fileno());self.bytes+=len(data);self.sequence+=1
        ref={'file':str(self.out/(stream+'.jsonl')),'sequence':self.sequence,'object_id':name,'sha256_line':hashlib.sha256(data).hexdigest()}
        self.done.add((stream,kind,name));cost=receipt(before)
        self.writer_previous={'record_ref':ref,'serialization_write_flush_fsync':cost,'bytes_written':len(data)}
        for key in ('user_seconds','system_seconds','wall_seconds'):self.writer_totals[key]+=cost.get(key,0.)
        self.writer_totals['bytes_written']+=len(data)
        for failure in resource_failures(cost):self.metadata_failures.append({'object_id':name,'writer_failure':failure})
        return ref
    def atomic(self,path,data):
        if self.bytes+len(data)>8*1024**3:raise Stopped('8-GiB cumulative output limit')
        temporary=path.with_name(path.name+'.partial')
        with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        self.bytes+=len(data)
    def close(self):
        for f in self.streams.values():f.close()

def structural_matrix(obj,shape,aliases):
    if set(obj)!= {'shape','denominator','entries','logical_aliases'} or obj['shape']!=shape or obj['logical_aliases']!=sorted(aliases):raise ValueError('bound matrix header/aliases')
    den=native(obj['denominator']);previous=(-1,-1)
    if den<=0:raise ValueError('positive matrix denominator')
    for row in obj['entries']:
        if type(row) is not list or len(row)!=4:raise ValueError('matrix coordinate schema')
        i,j,r,s=map(native,row)
        if not(0<=i<shape[0] and 0<=j<shape[1]) or (i,j)<=previous or not(r or s):raise ValueError('matrix coordinate order/range')
        previous=i,j
def structural_canonical(obj):
    if set(obj)!= {'denominator','entries'} or native(obj['denominator'])<=0:raise ValueError('canonical native header')
    previous=None
    for row in obj['entries']:
        if type(row) is not list or len(row)!=4:raise ValueError('canonical entry header')
        ii,jj,r,s=row
        if type(ii) is not list or type(jj) is not list or any(type(v) is not int or not(0<=v<24) for v in ii+jj):raise ValueError('canonical mode schema')
        native(r);native(s);key=tuple(ii),tuple(jj)
        if ii!=sorted(set(ii)) or jj!=sorted(set(jj)) or (previous is not None and key<=previous) or not(r or s):raise ValueError('canonical native ordering')
        previous=key
def decode_canonical(obj):
    # Only called for the eighteen absent matrix assemblies.
    den=obj['denominator']
    return {(tuple(ii),tuple(jj)):(F(r,den),F(s,den)) for ii,jj,r,s in obj['entries']}
def rowreference(path,line,row):
    return {'file':str(path),'sequence':row['sequence'],'object_id':row['object_id'],'sha256_line':hashlib.sha256(line).hexdigest()}
def journalrefs(obj):
    if isinstance(obj,dict):
        result=[obj] if set(('file','sequence','object_id','sha256_line')).issubset(obj) else []
        for value in obj.values():result.extend(journalrefs(value))
        return result
    if isinstance(obj,list):return [r for value in obj for r in journalrefs(value)]
    return []

def admit(rec):
    reviewed=rec.expected_pins
    if type(reviewed) is not list or len(reviewed)!=188 or len({p['path'] for p in reviewed})!=188:raise ValueError('188 distinct reviewed paths required')
    documents={};pinby={}
    for index,expected in enumerate(reviewed):
        boundary();path=Path(expected['path']);rec.visited_admissions.add(str(path))
        try:data,actual=readpin(path,expected['sha256'])
        except BaseException as exc:
            failed={'expected':expected,'failure':{'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}}
            rec.failed_admissions.append(failed);rec.write('admission','failed_input_observation',f'failed-input:{index}',failed)
            raise
        observation={'expected':expected,'observed':actual,'matched':actual==expected}
        ref=rec.write('admission','input_pin',f'input:{index}',observation)
        rec.pins.append(actual);pinby[str(path)]=actual
        if actual!=expected:
            rec.failed_admissions.append(observation)
            raise ValueError('guardian/child descriptor join')
        rec.admitted_paths.add(str(path))
        if path.parent==OLD or path.parent==OLD/'matrices':documents[str(path)]=(data,ref)
    execution=loads(documents[str(OLD/'execution.json')][0])
    if pinby[str(OLD/'execution.json')]['sha256']!=OLD_EXEC_SHA or execution['returncode']!=1 or any(execution.get(k) is not True for k in ('terminal_established','child_reaped','inventory_complete','owned_sources_unchanged')) or execution['errors'] or execution['missing_fixed_files']:raise ValueError('exact partial terminal lineage')
    inventory=execution['inventory'];by_path={item['path']:item for item in inventory}
    if len(inventory)!=171 or len(by_path)!=171 or set(documents)!=(set(by_path)|{str(OLD/'execution.json')}):raise ValueError('exact 172-file old output set')
    if len(execution['source_pins'])!=13:raise ValueError('thirteen inherited source/input pins')
    for role,pin in execution['source_pins'].items():
        if pinby[pin['path']]!=pin or execution['owned_source_postchecks'][role]['matched'] is not True:raise ValueError('inherited current source/input lineage')
    for path,item in by_path.items():
        actual=pinby[path]
        if any(actual[k]!=item[k] for k in ('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns')) or item['unfinished_temporary']:raise ValueError('old closed inventory join')
    rows={};references={};kinds={};counts={}
    for stream in ('admission','coefficients','aliases','norms','metrics'):
        path=OLD/(stream+'.jsonl');data=documents[str(path)][0]
        for line in data.splitlines(keepends=True):
            row=loads(line);seq=native(row['sequence'])
            if seq in rows:raise ValueError('duplicate old global sequence')
            rows[seq]=row;references[seq]=rowreference(path,line,row)
            key=stream,row['type'];counts[key]=counts.get(key,0)+1
            kinds.setdefault(row['type'],[]).append(row)
    if sorted(rows)!=list(range(1,2292)):raise ValueError('2291 contiguous closed records required')
    for row in rows.values():
        for ref in journalrefs(row):
            if ref!=references.get(ref['sequence']):raise ValueError('old direct journal reference')
    required={'canonical_body':325,'canonical_product':194,'canonical_audit':519,'canonical_operation':519,'public_alias':199,'sector_basis':8,'sector_matrix':163,'sector_norm':162,'complete_matrix_norm_unit':162,'face_stack_norm':7,'candidate_work_plan':2,'postcheck':13}
    if any(len(kinds.get(kind,[]))!=count for kind,count in required.items()):raise ValueError('exact retained cut')
    if any(row['payload']['matched'] is not True for row in kinds['postcheck']):raise ValueError('old producer postchecks')
    if any(row['payload']['number_conserved'] is not True or row['payload']['violations'] for row in kinds['canonical_audit']):raise ValueError('inherited audit status')
    canonical={};bodyrows={};identities={}
    for seq,row in sorted(rows.items()):
        if row['type'] not in ('canonical_body','canonical_product'):continue
        payload=row['payload'];digest=payload['canonical_sha256']
        if 'canonical' in payload:
            obj=payload['canonical'];structural_canonical(obj)
            if hashlib.sha256(dump(obj)).hexdigest()!=digest:raise ValueError('one-time native JSON identity')
            identities[digest]=obj
        else:
            ancestor=payload['canonical_ref']['sequence']
            if ancestor>=seq or ancestor not in canonical or canonical[ancestor][0]!=digest:raise ValueError('backward canonical identity')
            obj=canonical[ancestor][1]
        canonical[seq]=(digest,obj);bodyrows[row['object_id']]=seq
    aliases={};alias_refs={};reverse={};coefficient_refs={}
    for row in kinds['public_alias']:
        value=row['payload'];n=native(value['parent_sites']);label=value['logical_name'];key=n,label
        if key in aliases:raise ValueError('duplicate public alias')
        seq=bodyrows[f'n{n}:'+value['canonical_body']];digest,obj=canonical[seq]
        if value['canonical_sha256']!=digest:raise ValueError('public canonical identity join')
        aliases[key]=(digest,obj,seq);alias_refs[key]=references[row['sequence']];reverse.setdefault((n,digest),[]).append(label)
        resolved=seq
        while 'canonical_ref' in rows[resolved]['payload']:resolved=rows[resolved]['payload']['canonical_ref']['sequence']
        coefficient_refs[key]=[references[seq]]+([references[resolved]] if resolved!=seq else [])
    for labels in reverse.values():labels.sort()
    matrices={};norms={};units={};stacks={}
    for row in kinds['sector_matrix']:
        value=row['payload'];m=value['matrix'];key=m['parent_sites'],m['sector'],m['canonical_sha256'];path=m['path']
        if key in matrices or path not in documents or pinby[path]['sha256']!=m['sha256'] or pinby[path]['size']!=m['size']:raise ValueError('matrix receipt/current identity')
        obj=loads(documents[path][0]);structural_matrix(obj,m['shape'],reverse[key[0],key[2]])
        matrices[key]={'metadata':m,'row':row,'ref':references[row['sequence']]}
        if key==(3,3,LAST):last_matrix=obj
    for row in kinds['sector_norm']+kinds['complete_matrix_norm_unit']:
        parts=row['object_id'].split(':');key=int(parts[0][1:]),int(parts[1][1:]),parts[-1]
        dest=norms if row['type']=='sector_norm' else units
        if key in dest or key not in matrices:raise ValueError('norm/unit to matrix join')
        dest[key]={'payload':row['payload'],'ref':references[row['sequence']]}
    for row in kinds['face_stack_norm']:
        value=row['payload'];stacks[value['parent_sites'],value['sector']]={'payload':value,'ref':references[row['sequence']]}
    old_result=loads(documents[str(OLD/'vacuum_ref_u2_v2_result.json')][0])
    if old_result['terminal_checkpoint'] is not False or old_result['target_complete'] is not False:raise ValueError('preserve stale incomplete checkpoint scope')
    expected_missing=set()
    for label,digest,seq in MISSING:
        expected_missing.update([('metrics','sector_matrix',f'n3:N3:matrix:{digest}'),('norms','sector_norm',f'n3:N3:norm:{digest}'),('metrics','complete_matrix_norm_unit',f'n3:N3:unit:{digest}')])
    expected_missing.update([('norms','sector_norm',f'n3:N3:norm:{LAST}'),('metrics','complete_matrix_norm_unit',f'n3:N3:unit:{LAST}'),('norms','face_stack_norm','n3:N3:Pi2_stack')])
    required_roles={(v['stream'],v['type'],v['object_id']) for v in old_result['required_role_plan']}
    present_roles={(Path(references[seq]['file']).stem,row['type'],row['object_id']) for seq,row in rows.items()}
    if len(required_roles)!=2346 or required_roles-present_roles!=expected_missing:raise ValueError('exact 57 missing old roles')
    planned=set();work_index={};candidate_refs={}
    for row in kinds['candidate_work_plan']:
        value=row['payload'];n=value['parent_sites']
        candidate_refs[n]=references[row['sequence']]
        for label,per_sector in value['per_public_body'].items():
            for sector,count in per_sector.items():work_index[n,int(sector),label]=native(count)
        for sector,labels in value['prefix_plans'].items():
            planned.update((n,int(sector),aliases[n,label][0]) for label in labels)
    missing_keys={(3,3,digest) for label,digest,seq in MISSING}
    if len(planned)!=181 or planned-set(matrices)!=missing_keys or planned-set(norms)!=(missing_keys|{(3,3,LAST)}) or planned-set(units)!=(missing_keys|{(3,3,LAST)}):raise ValueError('exact matrix/norm/unit cuts')
    if set(stacks)!={(n,k) for n in (2,3) for k in range(4)}-{(3,3)}:raise ValueError('exact stack cut')
    failure=kinds.get('failure',[])
    if len(failure)!=1 or failure[0]['payload']['type']!='Stopped' or failure[0]['payload']['message']!='handled signals [14]':raise ValueError('handled scientific-stop lineage')
    basisrow=rows[2231];basis=basisrow['payload']['basis']
    if basisrow['type']!='sector_basis' or basisrow['payload']['parent_sites']!=3 or basisrow['payload']['sector']!=3 or len(basis)!=2024 or basis!=sorted(set(basis)) or any(type(v) is not int or not(0<=v<1<<24) or v.bit_count()!=3 for v in basis):raise ValueError('saved exact chain three-particle basis')
    for label,digest,seq in MISSING:
        if aliases[3,label][0]!=digest or aliases[3,label][2]!=seq or (3,3,digest) in matrices or (3,3,digest) in norms:raise ValueError('literal missing matrix cut')
    if (3,3,LAST) not in matrices or (3,3,LAST) in norms or (3,3,LAST) in units or (3,3) in stacks:raise ValueError('norm-only and stack cut')
    if matrices[3,3,LAST]['row']['sequence']!=2277 or len(last_matrix['entries'])!=2880:raise ValueError('last returned matrix identity')
    alias_group_refs={(n,digest):[alias_refs[n,label] for label in labels] for (n,digest),labels in reverse.items()}
    context={'rows':rows,'references':references,'aliases':aliases,'alias_refs':alias_refs,'reverse':reverse,'matrices':matrices,'norms':norms,'units':units,'stacks':stacks,'basis':basis,'basis_ref':references[2231],'last_matrix':last_matrix,'execution':execution,'pinby':pinby,'planned':planned,'work_index':work_index,'candidate_refs':candidate_refs,'alias_group_refs':alias_group_refs,'coefficient_refs':coefficient_refs}
    rec.write('admission','reuse_index','reuse_index',{'old_execution_ref':pinby[str(OLD/'execution.json')],'old_attempt_permanently_incomplete':True,'inherited_counts':required,'coefficient_roles':[references[seq] for seq in canonical],'public_aliases':[{'parent_sites':n,'logical_name':label,'canonical_sha256':value[0],'ref':alias_refs[n,label]} for (n,label),value in aliases.items()],'existing_matrices':[value['ref'] for value in matrices.values()],'existing_norms':[value['ref'] for value in norms.values()],'existing_units':[value['ref'] for value in units.values()],'existing_stacks':[value['ref'] for value in stacks.values()],'basis_ref':references[2231],'historical_continuous_timing_incomplete':[LAST]},[pinby[str(OLD/'execution.json')]])
    return context

def norm_return(rec,digest,obj,matrix_ref,body_zero,coefficient_refs):
    boundary();before=usage()
    total=sum(r*r+s*s for i,j,r,s in obj['entries']);square=F(total,obj['denominator']**2)
    value={'squared_norm':[square.numerator,square.denominator],'norm_exact':f'sqrt({square.numerator}/{square.denominator})','identically_zero':body_zero,'zero_in_sector':not obj['entries'],'vacuous_body':body_zero}
    ref=rec.write('norms','sector_norm',f'n3:N3:norm:{digest}',value,[matrix_ref]+coefficient_refs,receipt(before))
    rec.norm_refs[digest]={'payload':value,'ref':ref}
    return ref

def complete_missing(rec,context):
    old=context['matrices'][3,3,LAST];before=usage()
    alias=context['aliases'][3,'F2[n0,0]'];norm_ref=norm_return(rec,LAST,context['last_matrix'],old['metadata'],not alias[1]['entries'],context['coefficient_refs'][3,'F2[n0,0]'])
    value={'parent_sites':3,'sector':3,'canonical_sha256':LAST,'original_matrix_receipt':old['ref'],'original_matrix_timing_verbatim':old['row']['payload'],'new_norm_ref':norm_ref,'new_norm_only_interval_including_write':receipt(before),'continuous_historical_unit_timing_available':False,'rate_sample_eligible':False,'historical_gap':'original continuous end-to-end clock unavailable; disjoint actual costs are not an invented continuous unit'}
    rec.segmented=rec.write('metrics','segmented_matrix_norm_completion',f'n3:N3:segmented:{LAST}',value,[old['ref'],norm_ref])
    checkpoint(rec,context)
    for label,digest,seq in MISSING:
        boundary();unit_start=usage()
        actual_digest,obj,role_seq=context['aliases'][3,label]
        source_refs=context['coefficient_refs'][3,label]+[context['basis_ref'],context['candidate_refs'][3]]+context['alias_group_refs'][3,digest]
        operand_start=usage();a=decode_canonical(obj);decode_cost=receipt(operand_start)
        math_start=usage();matrix=assemble(a,24,3,context['basis']);math_cost=receipt(math_start)
        # No pending-stop boundary may discard the returned matrix.
        encode_start=usage();payload=matrixpayload(matrix,2024,context['reverse'][3,digest]);data=dump(payload);encoding_cost=receipt(encode_start)
        write_start=usage();path=rec.out/'matrices'/f'n3_N3_{digest}.json';rec.atomic(path,data);write_cost=receipt(write_start)
        matrix_ref={'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'size':len(data),'shape':[2024,2024],'canonical_sha256':digest,'parent_sites':3,'sector':3,'logical_aliases':context['reverse'][3,digest]}
        reference=rec.write('metrics','sector_matrix',f'n3:N3:matrix:{digest}',{'matrix':matrix_ref,'method':'CAR_spectator_assembly_missing_only','candidate_work':context['work_index'][3,3,label],'merge_input_nnz_sum':None,'nnz':len(matrix),'denominator':payload['denominator'],'maximum_integer_bits':max((abs(v).bit_length() for row in payload['entries'] for v in row[2:]),default=0),'cached_operand_decode':decode_cost,'mathematical_assembly':math_cost,'matrix_payload_serialization':encoding_cost,'durable_matrix_write':write_cost,'inclusive_total_through_write':receipt(unit_start)},source_refs)
        matrix_ref['receipt_ref']=reference;rec.matrix_refs[digest]=matrix_ref
        norm_ref=norm_return(rec,digest,payload,matrix_ref,not obj['entries'],context['coefficient_refs'][3,label])
        value={'parent_sites':3,'sector':3,'canonical_sha256':digest,'logical_aliases':context['reverse'][3,digest],'candidate_work':context['work_index'][3,3,label],'merge_input_nnz_sum':None,'method':'CAR_spectator_assembly_missing_only','matrix_ref':matrix_ref,'norm_ref':norm_ref,'rate_sample_eligible':True,'inclusive_total_including_norm_before_unit_metric_write':receipt(unit_start)}
        rec.unit_refs[digest]=rec.write('metrics','complete_matrix_norm_unit',f'n3:N3:unit:{digest}',value,source_refs+[reference,norm_ref])
        checkpoint(rec,context)
    boundary();before=usage();squares=[];refs=[]
    for b in (0,1):
        digest=context['aliases'][3,f'Pi2[{b}]'][0];item=context['norms'][3,3,digest]
        pair=item['payload']['squared_norm'];squares.append(F(native(pair[0]),native(pair[1])));refs.append(item['ref'])
    square=sum(squares,F(0));value={'parent_sites':3,'sector':3,'squared_norm':[square.numerator,square.denominator],'norm_exact':f'sqrt({square.numerator}/{square.denominator})'}
    rec.stack=rec.write('norms','face_stack_norm','n3:N3:Pi2_stack',value,refs,receipt(before))

def checkpoint(rec,context,terminal=False):
    before=usage();rec.generation+=1
    missing=[{'stream':stream,'type':kind,'object_id':name} for stream,kind,name in sorted(rec.plan-rec.done)]
    scientific=all(digest in rec.matrix_refs and digest in rec.norm_refs and digest in rec.unit_refs for label,digest,seq in MISSING) and LAST in rec.norm_refs and rec.segmented is not None and rec.stack is not None
    observed={str(path):path.stat().st_size for path in rec.out.rglob('*') if path.is_file()}
    complete=scientific and terminal and not missing and rec.error is None and not rec.metadata_failures
    value={'status':'prefix_missing_complete' if complete else 'incomplete','terminal_checkpoint':terminal,'target_complete':complete,'scientific_missing_cut_complete':scientific,'old_attempt_status_permanently':'incomplete','combined_prefix_mathematical_complete':scientific,'full_package_complete':False,'balance_test_run':False,'closure':'not_established','historical_continuous_timing_incomplete':[LAST],'continuous_unit_sample_count_if_union_complete':180,'segmented_completion_count':int(rec.segmented is not None),'segmented_record_rate_sample_eligible':False,'fixed_file_plan':TOP_FILES,'missing_top_files_before_this_replacement':[name for name in TOP_FILES if not(rec.out/name).is_file()],'file_observation_scope':'own-process pre-checkpoint observation; terminal wrapper inventory is separate','dynamic_matrix_file_plan':[{'path':str(rec.out/'matrices'/f'n3_N3_{digest}.json'),'canonical_sha256':digest} for label,digest,seq in MISSING],'required_role_plan':[{'stream':stream,'type':kind,'object_id':name} for stream,kind,name in sorted(rec.plan)],'missing_required_roles':missing,'returned_matrices':rec.matrix_refs,'returned_norms':rec.norm_refs,'new_full_units':rec.unit_refs,'segmented_completion':rec.segmented,'new_stack':rec.stack,'input_pins':rec.pins,'postchecks':rec.postchecks,'error':rec.error,'resource_metadata_failures':rec.metadata_failures,'handled_signals':STOP,'checkpoint_generation':rec.generation,'cumulative_charged_written_bytes':rec.bytes,'current_tree_bytes_observed':sum(observed.values()),'current_files_observed':observed,'journal_writer_totals_before_this_write':rec.writer_totals,'last_completed_writer_receipt':rec.writer_previous,'checkpoint_totals_before_this_write':rec.checkpoint_totals,'last_completed_checkpoint_receipt':rec.checkpoint_previous,'final_checkpoint_tail':'bounded two-second unassigned child tail included in reaped child CPU; guardian own final record is a separate two-second unmeasured reserve' if terminal else None,'elapsed_seconds':time.monotonic()-ENTRY}
    by_kind={}
    for item in missing:by_kind.setdefault((item['stream'],item['type']),[]).append(item['object_id'])
    expected_matrix_paths={digest:str(rec.out/'matrices'/f'n3_N3_{digest}.json') for label,digest,seq in MISSING}
    post_visited={row['expected']['path'] for row in rec.postchecks}
    admitted=rec.admitted_paths
    value['unadmitted_input_paths']=[pin['path'] for pin in rec.expected_pins if pin['path'] not in admitted]
    value.update(expected_input_plan_before_admission=rec.expected_pins,expected_input_plan_available=len(rec.expected_pins)==188,failed_admission_observations=rec.failed_admissions,unvisited_admission_paths=[pin['path'] for pin in rec.expected_pins if pin['path'] not in rec.visited_admissions],failed_postcheck_observations=[row for row in rec.postchecks if not row['matched']],unvisited_postcheck_paths=[pin['path'] for pin in rec.expected_pins if pin['path'] not in post_visited],unvisited_postchecks_not_admitted=[pin['path'] for pin in rec.expected_pins if pin['path'] not in post_visited and pin['path'] not in admitted],missing_admission_roles=[item['object_id'] for item in missing if item['stream']=='admission' and item['type']!='postcheck']+by_kind.get(('metrics','admission_complete'),[]),missing_postcheck_roles=by_kind.get(('admission','postcheck'),[]),missing_matrix_receipt_roles=by_kind.get(('metrics','sector_matrix'),[]),missing_norm_roles=by_kind.get(('norms','sector_norm'),[]),missing_full_unit_roles=by_kind.get(('metrics','complete_matrix_norm_unit'),[]),missing_segmented_unit_roles=by_kind.get(('metrics','segmented_matrix_norm_completion'),[]),missing_stack_roles=by_kind.get(('norms','face_stack_norm'),[]),missing_new_matrix_files=[path for path in expected_matrix_paths.values() if path not in observed],raw_matrix_file_present_receipt_missing=[path for digest,path in expected_matrix_paths.items() if path in observed and ('metrics','sector_matrix',f'n3:N3:matrix:{digest}') not in rec.done],observed_incomplete_temporaries=[path for path in observed if '.partial' in Path(path).name])
    encoding_start=usage();data=dump(value);encoding_cost=None if terminal else receipt(encoding_start);write_start=None if terminal else usage();rec.atomic(rec.out/'vacuum_ref_u2_prefix_missing_result.json',data)
    if not terminal:
        cost=receipt(before);rec.checkpoint_previous={'generation':rec.generation,'serialization':encoding_cost,'atomic_write_flush_fsync':receipt(write_start),'inclusive_checkpoint_total':cost,'bytes_written':len(data)}
        for key in ('user_seconds','system_seconds','wall_seconds'):rec.checkpoint_totals[key]+=cost.get(key,0.)
        rec.checkpoint_totals['bytes_written']+=len(data)
        for failure in resource_failures(rec.checkpoint_previous):rec.metadata_failures.append({'checkpoint_generation':rec.generation,'failure':failure})
    return value

MISSING=[('A2[n0,1]', '58c6d42fb97aac719cb089e94ef3a7108930d8acd4d2b48ea506d7647d858ec4', 1528), ('F2[n0,1]', '16fd3c28e2cbe763beceb37c26a3ef62129bacb4f5152e05ed9c681598d19002', 1525), ('A2[n1,0]', '4f39a482029c027f3ba56254b133a2a43b0ebd6e026cef2acb91b456f4b9058b', 1444), ('A2[n1,1]', 'e53834d64c9c29a6c3918ab021187213b7f27f5c9464777c100470eca639370e', 1541), ('F2[n1,1]', 'b63f488a52a88494c11b61014794b3a4d94ee1be8730b740d5737e99abb9f975', 1538), ('A2[n1,2]', 'af4d6e1555aefe2de1492ef7b1dcae7b4092f7411ae10f1c3abf04c6f0151462', 1638), ('F2[n1,2]', '31600e031abbdf6237c548f023d767772a27e8a71d39d3119024ab76085ffa89', 1635), ('A2[n2,1]', '0568c28d830367e261f17a2355725d4ec4944988b4699c6b370a3ca967413a52', 1554), ('F2[n2,1]', 'd023ba3f36fd0a3420397aa17c581c1b35bd33b0ee1c9409ef805bc1948fc367', 1551), ('F2[n2,2]', '762bb630e13bd5bee4ce19f4c470a73f09d5cea032227ef0dbc6d86255d38b17', 1648), ('A2[v0,1]', 'ea2a5c3b947fdd919a7cc0896c9667c98a70defc86df26bcf0827e29e20ed262', 1567), ('F2[v0,1]', 'ad9158e96652d62712bac9422a55cc8a9148cb29fee28ae6cb2baa6487945b7e', 1564), ('A2[v0,2]', 'de36a53b84d196049d75b283534f46dcb9086883a17730c3b2890624353f4688', 1664), ('F2[v0,2]', '8ac2d559952e2ab954d2fc6899ff8505dd6a29a98c786b6e15ef870882ee9c47', 1661), ('A2[v1,0]', 'e87c5e91cd481049fcdf92a82125b20ef9ec2187b7f3bc94a164aca8fbcb430a', 1483), ('F2[v1,0]', 'e253df69836859f39b6891c884dbcab6678aef4d074a5600cd93a9f0896a3122', 1480), ('A2[v2,1]', 'f74a5abbaace8fb4317346fa4e636eb78bfdb5b2d35cf410c9ee889658d754e6', 1593), ('F2[v2,1]', '237a9edc294ecc96355014ff42fcd7063d10447a09cdcab4859369d3456f6f6d', 1590)]

def install_plan(rec):
    for index in range(188):
        rec.expect('admission','input_pin',f'input:{index}')
        rec.expect('admission','postcheck',f'postcheck:{index}')
    rec.expect('admission','runtime','runtime');rec.expect('admission','reuse_index','reuse_index')
    rec.expect('metrics','admission_complete','admission_complete')
    for label,digest,seq in MISSING:
        rec.expect('metrics','sector_matrix',f'n3:N3:matrix:{digest}')
        rec.expect('norms','sector_norm',f'n3:N3:norm:{digest}')
        rec.expect('metrics','complete_matrix_norm_unit',f'n3:N3:unit:{digest}')
    rec.expect('norms','sector_norm',f'n3:N3:norm:{LAST}')
    rec.expect('metrics','segmented_matrix_norm_completion',f'n3:N3:segmented:{LAST}')
    rec.expect('norms','face_stack_norm','n3:N3:Pi2_stack')

def runtime(rec,limits):
    env={k:os.environ.get(k) for k in THREADS+('CUDA_VISIBLE_DEVICES',)}
    free=os.statvfs(ROOT)
    observed={'nice':os.getpriority(os.PRIO_PROCESS,0),'affinity':sorted(os.sched_getaffinity(0)),'resident_threads':len(list(Path('/proc/self/task').iterdir())),'CPU':list(resource.getrlimit(resource.RLIMIT_CPU)),'AS':list(resource.getrlimit(resource.RLIMIT_AS)),'environment':env,'available_disk_bytes':free.f_bavail*free.f_frsize,'executable':str(Path(sys.executable).resolve()),'aggregate_planning_reservation_CPU_seconds':1750.221279,'original_task_ceiling_CPU_seconds':7200,'guardians_own_final_write_unmeasured_reserve_seconds':2}
    rec.write('admission','runtime','runtime',observed)
    if observed['nice']<10 or len(observed['affinity'])!=1 or observed['resident_threads']!=1 or observed['CPU']!=[540,560] or observed['AS']!=[4*1024**3]*2 or limits['CPU_soft_hard']!=[540,560] or any(env[k]!='1' for k in THREADS) or env['CUDA_VISIBLE_DEVICES']!='' or observed['available_disk_bytes']<16*1024**3 or observed['executable']!=str(Path('<internal>/python').resolve()):raise ValueError('fixed missing-prefix runtime guard')

def main():
    global PHASE
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--outdir',required=True);args=parser.parse_args()
    for sig in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT,signal.SIGXCPU):signal.signal(sig,handler)
    limits=loads(os.environ['VACUUM_U2_LIMITS']);launch=float(os.environ['VACUUM_U2_CHILD_LAUNCH'])
    signal.setitimer(signal.ITIMER_REAL,max(.001,580-(time.monotonic()-launch)))
    rec=None;context=None;result=None
    try:
        out=Path(args.outdir)
        if out!=ROOT/'prefix_missing_v1' or not out.is_dir():raise ValueError('fresh owned missing-prefix directory')
        rec=Record(out);install_plan(rec)
        expected=loads(os.environ['VACUUM_U2_REVIEWED_PINS'])
        if type(expected) is not list or len(expected)!=188 or any(type(pin) is not dict or type(pin.get('path')) is not str for pin in expected):raise ValueError('native complete expected 188-path plan')
        rec.expected_pins=expected
        rec.write('admission','logical_plan','logical_plan',{'missing_matrices':MISSING,'norm_only_matrix':LAST,'new_matrix_count':18,'new_norm_count':19,'new_stack_count':1,'fixed_files':TOP_FILES,'expected_input_plan_before_admission':rec.expected_pins,'required_role_plan':[{'stream':s,'type':k,'object_id':n} for s,k,n in sorted(rec.plan)],'balance_test_run':False,'closure':'not_established'})
        runtime(rec,limits);admission_start=usage();context=admit(rec)
        rec.write('metrics','admission_complete','admission_complete',{'current_pins':188,'metadata_only_reuse_admission':True,'scope':'descriptor/hash/native-JSON/schema/reference checks; no scientific reconstruction'},rec.pins,receipt(admission_start))
        complete_missing(rec,context)
    except BaseException as exc:
        if rec is None:raise
        rec.error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'observation':getattr(exc,'observation',None)}
        rec.write('admission','failure','primary_failure',rec.error)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if rec is not None:
            PHASE='metadata';signal.setitimer(signal.ITIMER_REAL,15)
            try:
                for index,pin in enumerate(rec.pins):
                    observation={'expected':pin,'matched':False};rec.postchecks.append(observation)
                    try:
                        data,actual=readpin(Path(pin['path']),pin['sha256']);observation['observed']=actual
                        if actual!=pin:raise ValueError('current input postcheck changed')
                        observation['matched']=True
                    except BaseException as exc:
                        observation['failure']={'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)}
                        if isinstance(exc,Stopped):raise
                    rec.write('admission','postcheck',f'postcheck:{index}',observation)
            except BaseException as exc:
                if rec.error is None:rec.error={}
                rec.error.setdefault('metadata_failures',[]).append({'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None)})
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);PHASE='final_record';signal.setitimer(signal.ITIMER_REAL,2)
                try:
                    missing=[pin['path'] for pin in rec.pins[len(rec.postchecks):]]
                    failed=[row for row in rec.postchecks if not row['matched']]
                    if missing or failed:
                        if rec.error is None:rec.error={}
                        rec.error.update(missing_postchecks=missing,failed_postchecks=failed)
                    result=checkpoint(rec,context,True);rec.close()
                finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if result is not None and result['target_complete'] and rec.error is None and not rec.metadata_failures else 2

if __name__=='__main__':sys.exit(main())
