#!/usr/bin/env python3
"""One absent-only full K10/c0 frame and F1 native block; no propagation."""
import argparse
import contextlib
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
ROOT=Path('<internal>/bprime_channel_lead_v1')
BUSI=Path('<internal>/Bprime_channel_lead_v1')
OUT=ROOT/'frame_K10_c0_F1_v1'
ENTRY=time.monotonic()
PHASE='scientific'
STOP=[]
np=None
class Stopped(Exception):pass
class Withheld(Exception):pass
def handler(sig,frame):
    STOP.append({'signal':int(sig),'phase':PHASE,'elapsed':time.monotonic()-ENTRY})
    if PHASE!='scientific':raise Stopped('interruptible '+PHASE+' signal '+str(sig))
def boundary():
    if STOP:raise Stopped('pending handled signal')
def metadata(st):
    return {k:getattr(st,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}


def inspect(path,expected=None,capture=False):
    path=Path(path);o={'path':str(path),'expected_sha256':expected,'bytes_read':0,'phase':'resolve'}
    digest=None;blocks=[]
    try:
        resolved=str(path.resolve(strict=True));o['resolved_path']=resolved
        digest=hashlib.sha256();md5=hashlib.md5()
        with path.open('rb') as f:
            o['phase']='initial_fstat'
            before=metadata(os.fstat(f.fileno()));o['before']=before
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('regular source/output required')
            o['phase']='read'
            while True:
                block=f.read(8*1024**2)
                if not block:break
                digest.update(block);md5.update(block);o['bytes_read']+=len(block)
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
        pin={'path':str(path),'resolved_path':resolved,'sha256':actual,'md5':md5.hexdigest(),**before}
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


def dump(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def finite_float(token):
    value=float(token)
    if not math.isfinite(value):raise ValueError('nonfinite JSON floating token')
    return value
def loads(data):return json.loads(data,object_pairs_hook=nodup,parse_constant=badconstant,parse_float=finite_float)
def resource_state():
    r=resource.getrusage(resource.RUSAGE_SELF)
    fields={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','VmSize:','VmPeak:','Threads:')):fields[line.split(':')[0]]=int(line.split()[1])
    return {'wall':time.monotonic(),'user':r.ru_utime,'system':r.ru_stime,'peak_RSS_KiB':r.ru_maxrss,**fields}
def elapsed(before):
    try:
        after=resource_state()
        return {'wall_seconds':after['wall']-before['wall'],'user_seconds':after['user']-before['user'],'system_seconds':after['system']-before['system'],'peak_RSS_KiB':after['peak_RSS_KiB'],'current_RSS_KiB':after.get('VmRSS'),'VmSize_KiB':after.get('VmSize'),'VmPeak_KiB':after.get('VmPeak'),'resident_threads':after.get('Threads')}
    except BaseException as exc:
        if isinstance(exc,Stopped):raise
        return {'resource_observation_failure':{'type':type(exc).__name__,'message':str(exc)}}
def failure(exc):return {'type':type(exc).__name__,'message':str(exc),'observation':getattr(exc,'observation',None),'secondary_observation_writer_failure':getattr(exc,'secondary_observation_writer_failure',None),'traceback':traceback.format_exc()}

FROZEN='d993d2dab0002e28dae3c3834d872c2b771258fc82bff6f78d48d2c3d0151c8d'
SEEDS={'controls': ('controls_952e9e6e_v1/controls.json', 'bc3f5093671243721c05eaa7f2f7013ff8a4f9405c9ae89a3c7bdd54f3a6f45c'), 'native': ('native_952e9e6e_v1/native.json', '477e6dcee52372949dd4a6314cdacfc6c8e288611f5c1764d4f63989daeed4a3'), 'fields': ('fields_952e9e6e_v1/fields.json', '1b4428fceb8f580c0b5bb3c409f6dbfdb0c738299500bd1b7496489e2368fb94'), 'b11': ('b11_inputs_952e9e6e_v1/b11.json', '661d8c8df963aa758e448465dd2bbed0e6153046a383cc659b091153b5ab7aee'), 'k10': ('setup_benchmark_K10_952e9e6e_v1/run.json', '0841a6e86400afa64c0022725b20902212ac82066b5cf949c786f6e0f9bfbef4'), 'setup10': ('setup_benchmark_K10_952e9e6e_v1/setup_benchmark.json', '6181a094748fbc47555e478422d7adb3244a510beb1271854f41e2b215a4e07f')}
HANDOFF={'b11_detail_v1.py': '7cd98186b5f61877f26f4e6fca58db99ec3d91a3ac0d8c0fd8e0f56a3b345897', 'controls_detail_v1.py': '92973ac23fa9de62f82542840fab43178c81922607c8e7a8bff7c28937a23994', 'input_evidence_v1.py': '06dca79b96156aea09b9c44427d931030888415bbbb32b7e21f43c513301a696', 'lead_channels.py': 'a3afcd29f3385fd78c84378c5cb2fc38aa2a10b546f800388186895c9f538f88', 'lead_controls.py': '4848c30d398d781176d3301fac126a2a7a5705ca55b634c0aedba6e10c46a05a', 'lead_field_controls.py': '24acadfd8ac5220851a1de10e02170b74a3758122dcad6c21b6bb906d3e502dd', 'lead_field_core.py': '9b8ae1582d52ab15785b0c1f0b1ce56d092f0060a3261412a11fa68d4b92921a', 'lead_fields.py': 'e8c50abc55b4a26b0d9b39deb646d9a16a90ff0d67e2de666d5ea524c0370fa9', 'lead_gaussian.py': '2134c05b822804ec6e40c9021930c560ea487fedfa0dc2ad69acddeb22ccbbf7', 'lead_inputs.py': '0fd8857f0d931ac65a7d9cb85577a4d7a482d497206d744d4c4938c58980aaf3', 'lead_native.py': '478ac530ecd05146628f2b86d35290b7cefba28e74066bbca3649e7cab2875f1', 'lead_numerics.py': '635fa566ff1a63c924979beb17484ac8c4ff67a85e204523dc6002b6202c781d', 'lead_operator.py': '1add3a6b80f3b86aaf5dbd2fe8838301cecd748fd6e8cdb3dd9a86bd03405fb3', 'lead_prepare.py': '78ee18dd9c6b1952aac4c98fb386a389ceaf6c5dbcd48a993276d9d27793603c', 'lead_read.py': '5ad883fb5bde565b5df4068adcb36e6ca38ec927c48af9d6660163895aa8ec5d', 'lead_references.py': '620f3faef683879537cdf6161b7839d470a0495c5512dac87caca2fcff9b2f99', 'lead_run.py': 'dc863cad4273938cf73039de38b1c39aa1092f157025cf1c7d945d12e715695f', 'lead_runtime.py': '5336a3981ea090ec4ebf088a00d44b0e238364afe3120c3c5da8d32d51d4c8e7', 'lead_setup.py': '8b753d789f01f6599160dfa7a022fb53d63909d0cd81cbb9d71d30616fcc05ff', 'lead_symmetry.py': '90759a1bdd7eab894da3245375774d6cb18a99851f270350dfe564014cf4299f', 'run_evidence_v1.py': '9f1dab88ae48dba4d759e246607f4f65a4894f03b71576a12e8354f415f9e966'}
PARENTS={
 'D_result':('b11_diagnosis_v1/b11_diagnosis_result.json','37d83c9b74d47c3d178eb9c794f4d88e0ebb61e7029f938c79643f8009e1a70d'),
 'D_execution':('b11_diagnosis_v1/execution.json','1ce7034f12b4495375a27ad79d49b88d75f87eef35c123d69df85c440d154925'),
 'K_result':('setup_increment_K12_v1/setup_increment_K12_result.json','54585ed5fb04117014937a48c83b73e1f931e1cc96f597b17b3aa3f7d45502b0'),
 'K_execution':('setup_increment_K12_v1/execution.json','572d2705ff296998836476adc154ce08c864197e50cb88e4e0858bb0a15f342c')}
SELECTED={
 'K_journal':('setup_increment_K12_v1/admission.jsonl','ebc9cfa0b2e47637d240a40a7ef46d51c91b4b537eb468cd1c4aca098755f1b8'),
 'labels':('setup_increment_K12_v1/label_context.npz','d166fbaaf36c75c203bf64a8d1385d510d6d363af4c3bd1ae12c6e880298ca47'),
 'H_TT':('setup_increment_K12_v1/c0_H.npz','2be049bf2598115528338e5a29207c3c856d0649f501d6ba235dd6dc3e1d58db')}
TILE=BUSI/'setup_benchmark_K10_952e9e6e_v1/setup_tile_c0.npz'
TILE_SHA='18439867d4755ba3ea45f08ed289d4138c15a381e433437c424bfd1e38f39739'
NATIVE=BUSI/'native_952e9e6e_v1/native.npz'
NATIVE_SHA='ae2aa1fd52c59bc13fc75a13879762eabce3831f2cb98b2498cefccc871bdf23'
RAW_FILES=('geometry.npz A_R.npy G_R.npy G_raw.npy G_defect.npy G_H.npy G_realization_difference.npy eigensystem.npz inverse_eigenvalues_sqrt.npy scaled_eigenvectors.npy inverse_half.npy E.npy E_gram.npy E_gram_defect.npy H_delta.npz H_delta_defect.npz native_bound.npz U_delta.npy P_delta.npy P_raw.npy BP.npy D_raw.npy D_defect.npy D_H.npy D_realization_difference.npy').split()
DECISIONS=('F.geometry F.A_finite F.Gram_finite F.Gram_Hermitian F.spectrum F.inverse_half_finite F.E_finite F.E_Gram P.H_schema P.H_delta_Hermitian P.bound P.raw_finite P.diagonal_Hermitian').split()
TEXT_FILES=['admission.jsonl','objects.jsonl','metrics.jsonl','frame_unit_result.json','execution.json','stdout.log','stderr.log']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
NPOINTS=27452
RPOINTS=26428
ROWS=109808
RESTROWS=105712
WIDTH=3520
NUMERIC_CAP=64*1024**3
METADATA_CAP=64*1024**2

def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return {'nonfinite_float':repr(value)}
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

def encode(value):return dump(clean(value))
def journal_ref(path,line,row):return {'file':str(path),'sequence':row['sequence'],'kind':row['kind'],'name':row['name'],'sha256_line':hashlib.sha256(line).hexdigest()}
def member_ref(pin,key):return {'file_pin':pin,'array_key':key}
def logical_hash(a):
    a=np.ascontiguousarray(a)
    return {'shape':list(a.shape),'dtype':a.dtype.str,'sha256_C_bytes':hashlib.sha256(memoryview(a).cast('B')).hexdigest()}

def source_plan(here,worker_sha,wrapper_sha,accepted_parents=None,observer=None):
    plan={};parents={};initial=[]
    def add(path,sha):
        path=str(path)
        if path in plan and plan[path]!=sha:
            value={'path':path,'first_expected_sha256':plan[path],'conflicting_expected_sha256':sha}
            exc=ValueError('conflicting source/input SHA: '+path);exc.observation=value
            try:
                if observer:observer('conflict',value)
            except BaseException as secondary:exc.secondary_observation_writer_failure=failure(secondary)
            raise exc
        plan[path]=sha
    for role,(relative,sha) in PARENTS.items():
        path=ROOT/relative
        if accepted_parents is None:
            expected={'role':role,'path':str(path),'sha256':sha}
            if observer:observer('expected_parent',expected)
            try:
                pin,data=inspect(path,sha,True);initial.append(pin)
                if observer:observer('returned_parent',{'expected':expected,'returned':pin})
                parents[role]=loads(data)
            except BaseException as exc:
                try:
                    if observer:observer('failed_parent',{'expected':expected,'failure':failure(exc)})
                except BaseException as secondary:exc.secondary_observation_writer_failure=failure(secondary)
                raise
        else:parents[role]=accepted_parents[role]
        add(path,sha)
    for role in ('D','K'):
        result=parents[role+'_result'];execution=parents[role+'_execution']
        if len(result['input_pins'])!={'D':57,'K':59}[role]:raise ValueError('literal parent input count')
        for pin in result['input_pins']:add(pin['path'],pin['sha256'])
        for pin in execution['source_pins'].values():add(pin['path'],pin['sha256'])
    for relative,sha in SELECTED.values():add(ROOT/relative,sha)
    for relative,sha in SEEDS.values():add(BUSI/relative,sha)
    add(TILE,TILE_SHA);add(NATIVE,NATIVE_SHA)
    for name,sha in HANDOFF.items():add(here/name,sha)
    add(here/'bprime_frame_unit_v1.py',worker_sha);add(here/'run_bprime_frame_unit_v1.py',wrapper_sha)
    add(here.parent/'inputs/first_frame_contract_v1.tex',FROZEN)
    return [{'path':path,'sha256':sha} for path,sha in sorted(plan.items())],initial

class Records:
    def __init__(self,out,expected):
        if out!=OUT or out.resolve()!=out or not out.is_dir():raise ValueError('literal fresh output directory')
        allowed={'execution.json','stdout.log','stderr.log'}
        if any(p.name not in allowed for p in out.iterdir()):raise ValueError('foreign/preexisting output; no replay')
        self.out=out;self.expected=expected;self.pinby={};self.visited=set();self.failed_inputs=[];self.postchecks={};self.secondary=[];self.error=None;self.withheld=None
        self.seq=0;self.meta_reserved=0;self.numeric_reserved=0;self.saved={};self.partial={};self.decisions={};self.phases={};self.timings=[];self.documents={};self.row_refs={}
        self.writer_totals={'user_seconds':0.,'system_seconds':0.,'wall_seconds':0.,'attempted_bytes':0};self.generation=0
        self.streams={name:(out/(name+'.jsonl')).open('xb') for name in ('admission','objects','metrics')}
        self.chunks={'A_R.npy':[(i,min(i+512,RPOINTS)) for i in range(0,RPOINTS,512)],'E.npy':[(i,min(i+4096,ROWS)) for i in range(0,ROWS,4096)],'U_delta.npy':[(i,min(i+128,WIDTH)) for i in range(0,WIDTH,128)]};self.completed_chunks={k:[] for k in self.chunks}
        self.postcheck_plan=[x['path'] for x in expected]
        self.return_failures={};self.metadata_intervals={};self.primary_phase_failures=[]
        self.checkpoint_totals={'attempted':0,'completed':0,'user_seconds':0.,'system_seconds':0.,'wall_seconds':0.,'one_write_lag':True}
        self.event('admission','fixed_plan','fixed_plan',{'raw_files':RAW_FILES,'decisions':DECISIONS,'expected_inputs':expected,'chunk_plan':self.chunks,'cost_classes':list('AGLBEOHUPD'),'no_physical_clearance':True})
    def event(self,stream,kind,name,value,refs=()):
        before=resource_state();row={'sequence':self.seq+1,'kind':kind,'name':name,'payload':clean(value),'operand_references':list(refs)};data=encode(row)
        if self.meta_reserved+len(data)>METADATA_CAP-16*1024**2:raise Stopped('48-MiB producer metadata allowance')
        self.meta_reserved+=len(data);self.writer_totals['attempted_bytes']+=len(data)
        f=self.streams[stream];f.write(data);f.flush();os.fsync(f.fileno());self.seq+=1
        ref=journal_ref(self.out/(stream+'.jsonl'),data,row)
        cost=elapsed(before)
        for k in ('user_seconds','system_seconds','wall_seconds'):self.writer_totals[k]+=cost.get(k,0.)
        self.row_refs[kind,name]=ref;return ref
    def reserve(self,count):
        if self.numeric_reserved+count>NUMERIC_CAP:raise Stopped('64-GiB attempted numeric output cap')
        self.numeric_reserved+=count
    def commit_file(self,name,temp,refs,operation,extra=None):
        path=self.out/name
        if path.exists():raise ValueError('immutable returned file already exists')
        os.replace(temp,path)
        fd=os.open(self.out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        before=resource_state();pin=inspect(path);hash_cost=elapsed(before)
        value={'file':pin,'operation_receipt':operation,'hash_receipt':hash_cost,'extra':extra}
        ref=self.event('objects','returned_raw',name,value,refs);self.saved[name]={'pin':pin,'ref':ref,'extra':extra};self.partial.pop(name,None);self.checkpoint();return pin
    def save(self,name,value,refs=(),operation=None):
        try:return self._save(name,value,refs,operation)
        except BaseException as exc:
            self.return_failures[name]={'failure':failure(exc),'final_path':str(self.out/name),'final_exists':(self.out/name).exists(),'partial_exists':(self.out/(name+'.partial')).exists()}
            raise
    def _save(self,name,value,refs=(),operation=None):
        if name not in RAW_FILES or name in self.saved or name in self.partial:raise ValueError('planned single raw return required')
        before=resource_state();path=self.out/name;temp=path.with_name(name+'.partial')
        arrays=value if isinstance(value,dict) else {'value':value}
        size=sum(int(a.nbytes) for a in arrays.values());self.reserve(size+4096)
        self.partial[name]={'path':str(temp),'status':'serialization_started','reserved_bytes':size+4096}
        with temp.open('xb') as f:
            if name.endswith('.npz'):np.savez(f,**arrays)
            else:np.save(f,value,allow_pickle=False)
            f.flush();os.fsync(f.fileno())
        return self.commit_file(name,temp,refs,operation,{'arrays':{k:{'shape':list(a.shape),'dtype':a.dtype.str} for k,a in arrays.items()},'serialization_write_fsync':elapsed(before)})
    def calculate(self,name,fn,refs=()):
        boundary();self.event('metrics','operation_begin',name,{'raw_output':name},refs);before=resource_state();value=fn();operation=elapsed(before)
        pin=self.save(name,value,refs,operation);self.timings.append({'name':name,'computation':operation,'raw_ref':self.saved[name]['ref']});resources(self,'after_'+name);return value
    def create_map(self,name,shape):
        boundary()
        if name in self.saved or name in self.partial:raise ValueError('single mmap construction')
        size=math.prod(shape)*16;self.reserve(size+4096);path=self.out/(name+'.partial')
        if path.exists():raise ValueError('preserve existing partial')
        self.partial[name]={'path':str(path),'shape':list(shape),'dtype':'<c16','reserved_bytes':size+4096,'status':'filling','completed_intervals':[]}
        return np.lib.format.open_memmap(path,mode='w+',dtype=np.complex128,shape=shape)
    def tile(self,name,key,array,target,write_fn,refs,operation):
        before=resource_state();write_fn();target.flush()
        with Path(self.partial[name]['path']).open('rb') as f:os.fsync(f.fileno())
        value={'interval':list(key),'returned':logical_hash(array),'operation_receipt':operation,'write_flush_hash_receipt':elapsed(before),'partial_path':self.partial[name]['path']}
        ref=self.event('objects','returned_tile',name+':'+str(key[0]),value,refs)
        self.completed_chunks[name].append(list(key));self.partial[name]['completed_intervals'].append(list(key));self.checkpoint();return ref
    def finish_map(self,name,array,refs):
        try:return self._finish_map(name,array,refs)
        except BaseException as exc:
            self.return_failures[name]={'failure':failure(exc),'final_path':str(self.out/name),'final_exists':(self.out/name).exists(),'partial_exists':(self.out/(name+'.partial')).exists()}
            raise
    def _finish_map(self,name,array,refs):
        if self.completed_chunks[name]!=[list(v) for v in self.chunks[name]]:raise ValueError('all exact tile intervals required')
        array.flush();temp=Path(self.partial[name]['path'])
        self.commit_file(name,temp,refs,None,{'shape':list(array.shape),'dtype':array.dtype.str,'completed_intervals':self.completed_chunks[name]})
    def control(self,name,actual_fn,predicate,expected,refs=()):
        boundary()
        if name not in DECISIONS or name in self.decisions:raise ValueError('fixed single decision')
        self.event('metrics','control_begin',name,{'expected':expected},refs)
        before=resource_state();actual=actual_fn();cost=elapsed(before)
        self.event('objects','control_actual',name,actual,refs)
        self.event('metrics','control_actual_computation_cost',name,cost,refs)
        self.event('objects','control_expected',name,expected,refs)
        boundary()
        before=resource_state();truth=bool(predicate(actual));cost=elapsed(before)
        self.event('objects','control_decision',name,truth,refs);self.decisions[name]=truth
        self.event('metrics','control_truth_computation_cost',name,cost,refs);self.checkpoint()
        if not truth:self.withheld=name;raise Withheld(name)
    @contextlib.contextmanager
    def phase(self,label):
        boundary();before=resource_state();self.event('metrics','phase_begin',label,{'definition':'disjoint phase interval including raw retention and controls'})
        original=None
        try:yield
        except BaseException as exc:
            original=exc;self.primary_phase_failures.append({'phase':label,'failure':failure(exc)})
            raise
        finally:
            try:
                cost=elapsed(before);self.phases[label]=cost;self.event('metrics','phase_end',label,cost);self.checkpoint()
            except BaseException as exc:
                if original is None:raise
                self.secondary.append({'phase_finalization':label,'failure':failure(exc)})
    def atomic_result(self,value,before):
        data=encode(value)
        if self.meta_reserved+len(data)>METADATA_CAP-16*1024**2:raise Stopped('producer metadata cap')
        self.meta_reserved+=len(data);path=self.out/'frame_unit_result.json';temp=path.with_name(path.name+'.partial.'+str(time.monotonic_ns()))
        with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
        fd=os.open(self.out,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        cost=elapsed(before);self.checkpoint_totals['completed']+=1
        for key in ('user_seconds','system_seconds','wall_seconds'):self.checkpoint_totals[key]+=cost.get(key,0.)
    def checkpoint(self,terminal=False):
        before=resource_state();self.checkpoint_totals['attempted']+=1
        self.generation+=1;missing=[n for n in RAW_FILES if n not in self.saved]
        presence={n:(self.out/n).is_file() for n in RAW_FILES}
        missing_intervals={n:[list(v) for v in plan if list(v) not in self.completed_chunks[n]] for n,plan in self.chunks.items()}
        missing_decisions=[n for n in DECISIONS if n not in self.decisions]
        complete=not missing and not missing_decisions;passed=complete and all(self.decisions.values())
        final_inputs_ok=bool(self.postcheck_plan) and set(self.postchecks)==set(self.postcheck_plan) and all(v.get('matched') for v in self.postchecks.values())
        admitted=terminal and passed and final_inputs_ok and self.error is None and not self.secondary and self.withheld is None and not STOP
        self.atomic_result({'generation':self.generation,'terminal':terminal,'target_complete':complete,'raw_complete':not missing,'validation_complete':not missing_decisions,'all_controls_passed':passed,'frame_admitted':admitted,'native_block_admitted':admitted,'all_final_inputs_unchanged':final_inputs_ok,'physical_setup_complete':False,'performed_propagation':False,'F2_association_repaired':False,'status':('completed' if passed else 'completed_with_control_failure') if complete else ('withheld' if self.withheld else 'incomplete'),'raw_plan':RAW_FILES,'decision_plan':DECISIONS,'chunk_plan':self.chunks,'saved_files':self.saved,'partial_arrays':self.partial,'completed_chunks':self.completed_chunks,'missing_raw_receipts':missing,'missing_files':[n for n,present in presence.items() if not present],'producer_physical_file_presence_before_checkpoint_write':presence,'missing_tile_intervals':missing_intervals,'raw_return_failures':self.return_failures,'primary_phase_failures':self.primary_phase_failures,'metadata_intervals':self.metadata_intervals,'checkpoint_writer_costs_one_write_lag':dict(self.checkpoint_totals),'missing_decisions':missing_decisions,'false_decisions':[n for n,v in self.decisions.items() if not v],'decisions':self.decisions,'phase_costs':self.phases,'operation_costs':self.timings,'expected_input_plan':self.expected,'admitted_inputs':self.pinby,'failed_input_observations':self.failed_inputs,'unvisited_input_paths':[v['path'] for v in self.expected if v['path'] not in self.visited],'postcheck_plan':self.postcheck_plan,'postchecks':self.postchecks,'missing_postchecks':[p for p in self.postcheck_plan if p not in self.postchecks],'failed_postchecks':[v for v in self.postchecks.values() if not v.get('matched')],'error':self.error,'secondary_failures':self.secondary,'withheld':self.withheld,'handled_signals':STOP,'numeric_attempted_reserved_bytes':self.numeric_reserved,'metadata_attempted_reserved_before_this_write':self.meta_reserved,'journal_writer_totals':self.writer_totals,'source_bound_receipts':True},before)

def resources(rec,label,workspace=0):
    state=resource_state();available=None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
    from threadpoolctl import threadpool_info
    pools=threadpool_info();state.update(Python_version=sys.version,NumPy_version=getattr(np,'__version__',None),SciPy_version=getattr(sys.modules.get('scipy'),'__version__',None),CPU_soft_hard=list(resource.getrlimit(resource.RLIMIT_CPU)),AS_soft_hard=list(resource.getrlimit(resource.RLIMIT_AS)),MemAvailable_bytes=available,workspace_bytes=workspace,library_pools=pools,affinity=sorted(os.sched_getaffinity(0)),nice=os.getpriority(os.PRIO_PROCESS,0),numba_loaded=any(k=='numba' or k.startswith('numba.') for k in sys.modules))
    rec.event('metrics','resources',label,state)
    if state.get('Threads',100)>16 or state['nice']<10 or len(state['affinity'])!=8 or state['numba_loaded']:raise Stopped('runtime CPU/thread admission')
    if state.get('VmSize',0)*1024>60000000000 or state.get('VmRSS',0)*1024>60000000000:raise Stopped('scientific memory cap')
    if available is None or available-workspace<64*1024**3:raise Stopped('standing MemAvailable reserve')
    if any(os.environ.get(k)!='8' for k in THREADS) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise Stopped('thread/CUDA environment')
    if any(p.get('num_threads',0)>8 for p in pools):raise Stopped('BLAS pool cap')
    return state

def pin_subset_equal(actual,previous):
    return all(actual.get(k)==v for k,v in previous.items() if k in ('path','resolved_path','sha256','dev','ino','size','mtime_ns','ctime_ns'))
def strict_schema(a,shape,dtype):return a.shape==tuple(shape) and a.dtype==np.dtype(dtype) and not a.dtype.hasobject

def admit(rec):
    needed_json={str(ROOT/rel) for rel,sha in PARENTS.values()}|{str(BUSI/rel) for rel,sha in SEEDS.values()}
    needed_binary={str(TILE),str(NATIVE),str(ROOT/SELECTED['labels'][0]),str(ROOT/SELECTED['H_TT'][0])}
    journal_path=str(ROOT/SELECTED['K_journal'][0])
    for i,expected in enumerate(rec.expected):
        boundary();path=expected['path'];rec.visited.add(path);actual=None
        before=resource_state()
        try:
            if path in needed_json|needed_binary|{journal_path}:actual,data=inspect(path,expected['sha256'],True)
            else:actual=inspect(path,expected['sha256']);data=None
            ref=rec.event('admission','input_pin',path,{'expected':expected,'returned':actual,'matched':actual==expected,'read_hash_receipt':elapsed(before)})
            if actual!=expected:
                exc=ValueError('guardian/worker descriptor mismatch');exc.observation={'expected':expected,'returned':actual};raise exc
            rec.pinby[path]=actual
            if data is not None:rec.documents[path]=data
        except BaseException as exc:
            value={'expected':expected,'returned':actual,'failure':failure(exc)};rec.failed_inputs.append(value)
            try:rec.event('admission','failed_input',path,value)
            except BaseException as secondary:rec.secondary.append(failure(secondary))
            raise
    docs={role:loads(rec.documents[str(BUSI/rel)]) for role,(rel,sha) in SEEDS.items()}
    parents={role:loads(rec.documents[str(ROOT/rel)]) for role,(rel,sha) in PARENTS.items()}
    calculated,_=source_plan(Path(__file__).resolve().parent,rec.pinby[str(Path(__file__).resolve())]['sha256'],rec.pinby[str(Path(__file__).resolve().with_name('run_bprime_frame_unit_v1.py'))]['sha256'],parents)
    if {v['path']:v['sha256'] for v in calculated}!={v['path']:v['sha256'] for v in rec.expected}:raise ValueError('exact independently rebuilt source/input plan')
    for role in ('D','K'):
        result=parents[role+'_result'];execution=parents[role+'_execution']
        if any(result.get(k) is not True for k in ('raw_complete','validation_complete','all_controls_passed','target_complete','terminal','all_final_inputs_unchanged')) or result['error'] is not None or result['errors'] or result['withheld'] is not None:raise ValueError('accepted owned predecessor result '+role)
        if any(execution.get(k) is not True for k in ('terminal_established','child_reaped','inventory_complete','owned_sources_unchanged')) or execution['returncode']!=0 or execution['errors'] or execution.get('missing_owned_source_postchecks') or execution.get('missing_inventory_files'):raise ValueError('accepted owned predecessor custody '+role)
        for pin in result['input_pins']:
            if not pin_subset_equal(rec.pinby[pin['path']],pin) or result['final_postchecks'][pin['path']]['matched'] is not True:raise ValueError('parent input/current interval')
        for key,pin in execution['source_pins'].items():
            if not pin_subset_equal(rec.pinby[pin['path']],pin) or execution['owned_source_postchecks'][key]['matched'] is not True:raise ValueError('parent source/current interval')
    for role in ('controls','native','fields'):
        if docs[role].get('all_passed') is not True:raise ValueError('historical accepted prerequisite '+role)
    codes=docs['controls']['code_bindings']
    if len(codes)!=17 or any(not n.startswith('lead_') or n not in HANDOFF for n in codes):raise ValueError('exact historical seventeen-code map')
    def historical_tuple(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                pin=rec.pinby[value['path']]
                for k in ('sha256','resolved_path'):
                    if k in value and pin[k]!=value[k]:raise ValueError('historical literal tuple '+k)
                if 'bytes' in value and pin['size']!=value['bytes']:raise ValueError('historical byte count')
                if 'md5' in value and pin['md5']!=value['md5']:raise ValueError('historical same-open MD5 tuple')
            else:
                for v in value.values():historical_tuple(v)
        elif isinstance(value,list):
            for v in value:historical_tuple(v)
    for role in ('controls','native','fields','b11','k10','setup10'):historical_tuple(docs[role])
    if docs['native']['bindings']['code']!=codes or docs['fields']['bindings']['code']!=codes or docs['fields']['bindings']['native']['sha256']!=SEEDS['native'][1]:raise ValueError('historical source/field bindings')
    if docs['b11']['all_passed'] is not False or docs['b11']['bindings']['fields']['sha256']!=SEEDS['fields'][1] or docs['b11']['bindings']['code']!=codes:raise ValueError('preserved B11 refusal')
    core=('lead_runtime.py','lead_native.py','lead_channels.py','lead_setup.py','lead_symmetry.py','lead_operator.py','lead_numerics.py','lead_controls.py')
    if docs['k10']['all_passed'] is not True or docs['k10'].get('failure') is not None or docs['k10']['bindings']['code']!=codes or docs['setup10']['code_bindings']!={k:codes[k] for k in core}:raise ValueError('old complete K10 source map')
    if docs['k10']['arguments']['K']!=10 or docs['k10']['arguments']['field']!='F1':raise ValueError('K10/F1 fixture')
    here=Path(__file__).resolve().parent
    for name,old in codes.items():
        if old['path']!=str(BUSI/'src_952e9e6eb8c9/code'/name) or rec.pinby[str(here/name)]['sha256']!=old['sha256']:raise ValueError('historical-to-owned source bridge')
    k=parents['K_result'];inventory={v['path']:v for v in parents['K_execution']['inventory']}
    for role in ('labels','H_TT'):
        path=str(ROOT/SELECTED[role][0]);old=k['saved_files'][Path(path).name]
        if not pin_subset_equal(rec.pinby[path],old) or any(inventory[path][key]!=old[key] for key in ('path','sha256','size')):raise ValueError('selected K raw custody')
    krows={}
    for line in rec.documents[journal_path].splitlines(keepends=True):
        row=loads(line)
        if row['kind']=='returned_raw' and row['name'] in ('label_context.npz','c0_H.npz'):
            if row['payload']['file']['sha256']!=k['saved_files'][row['name']]['sha256']:raise ValueError('direct K raw journal reference')
            krows[row['name']]=journal_ref(Path(journal_path),line,row)
    if set(krows)!={'label_context.npz','c0_H.npz'}:raise ValueError('two direct K raw receipts')
    rec.event('admission','lineage','lineage',{'historical_source_map':codes,'current_owned_sources':{name:rec.pinby[str(here/name)] for name in HANDOFF},'parent_records':{role:rec.pinby[str(ROOT/rel)] for role,(rel,sha) in PARENTS.items()},'selected_K_raw_refs':krows,'B11_F2_association_still_refused':True,'accepted_controls_replayed':False})
    rec.checkpoint();return docs,parents,krows

def decode_inputs(rec):
    import io
    def selected(path,names):
        with np.load(io.BytesIO(rec.documents[str(path)]),allow_pickle=False) as archive:
            rec.event('admission','archive_members',str(path),{'actual_members':list(archive.files),'required_members':list(names)},[rec.pinby[str(path)]])
            if (path==TILE and set(archive.files)!=set(names)) or any(name not in archive.files for name in names):raise ValueError('required NPZ members')
            return {name:archive[name] for name in names}
    tile=selected(TILE,('points','global_dofs','A','G','projected_native_tile'))
    labels=selected(ROOT/SELECTED['labels'][0],('labels10',))['labels10']
    native=selected(NATIVE,('native_N','native_flow'))
    h=selected(ROOT/SELECTED['H_TT'][0],('data','indices','indptr','shape'))
    rec.event('admission','selected_member_schemas','selected_member_schemas',{'tile':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in tile.items()},'labels':{'shape':list(labels.shape),'dtype':labels.dtype.str},'native':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in native.items()},'H_TT':{k:{'shape':list(v.shape),'dtype':v.dtype.str} for k,v in h.items()}},[rec.pinby[str(path)] for path in (TILE,NATIVE,ROOT/SELECTED['labels'][0],ROOT/SELECTED['H_TT'][0])])
    if any(a.dtype.hasobject for a in list(tile.values())+[labels]+list(native.values())+list(h.values())):raise ValueError('no object arrays')
    if not(strict_schema(tile['points'],(1024,3),'int64') and strict_schema(tile['global_dofs'],(4096,),'int64') and strict_schema(tile['A'],(4096,WIDTH),'complex128') and strict_schema(tile['G'],(WIDTH,WIDTH),'complex128') and strict_schema(tile['projected_native_tile'],(WIDTH,WIDTH),'complex128') and strict_schema(labels,(WIDTH,7),'int64') and strict_schema(native['native_N'],(81,81,81),'float64') and strict_schema(native['native_flow'],(3,81,81,81),'float64')):raise ValueError('literal selected array schemas')
    if list(h['shape'])!=[4096,4096] or h['indptr'].shape!=(4097,):raise ValueError('cached H shape')
    for name,a in native.items():
        observed=logical_hash(a);association=rec.source_docs['k10']['field_association']['native_N' if name=='native_N' else 'native_v']
        rec.event('admission','native_logical_identity',name,{'actual':observed,'expected':association},[member_ref(rec.pinby[str(NATIVE)],name),{'file_pin':rec.pinby[str(BUSI/SEEDS['k10'][0])],'json_pointer':'/field_association/'+('native_N' if name=='native_N' else 'native_v')}])
        if observed['shape']!=association['shape'] or observed['dtype']!=association['dtype'] or observed['sha256_C_bytes']!=association['sha256']:raise ValueError('native logical-C field association')
    refs={'tile':rec.pinby[str(TILE)],'native':rec.pinby[str(NATIVE)],'labels':rec.pinby[str(ROOT/SELECTED['labels'][0])],'H_TT':rec.pinby[str(ROOT/SELECTED['H_TT'][0])]}
    rec.event('admission','selected_members','selected_members',{'refs':refs,'keys':{'tile':list(tile),'native':list(native),'H_TT':list(h)},'labels_shape':list(labels.shape)})
    return tile,labels,native,h,refs

def ref(rec,name,key=None):
    value=rec.saved[name]['ref']
    return {'return':value,'array_key':key} if key else value

def scalar(rec,name,fn,refs=()):
    boundary();before=resource_state();value=fn()
    cost=elapsed(before);rec.event('metrics','returned_scalar',name,value,refs)
    rec.event('metrics','scalar_computation_cost',name,cost,refs);return value

def finite(*arrays):return all(bool(np.all(np.isfinite(a))) for a in arrays)

def csr_arrays(h):
    return {'data':h.data.astype(np.complex128),'indices':h.indices.astype(np.int64),'indptr':h.indptr.astype(np.int64),'shape':np.array(h.shape,dtype=np.int64)}

def missing_h(points,lapse,flow):
    from scipy.sparse import coo_matrix
    from lead_native import spin_matrices
    alpha,_,_=spin_matrices();count=len(points)
    lookup={tuple(map(int,p)):i for i,p in enumerate(points)}
    rows=[];cols=[];values=[]
    for i,p in enumerate(points):
        if i%512==0:boundary()
        here=tuple(map(int,p+40))
        for axis in range(3):
            for distance in (1,2):
                target=p.copy();target[axis]+=distance;j=lookup.get(tuple(map(int,target)))
                if j is None or (i<1024 and j<1024):continue
                there=tuple(map(int,target+40))
                if distance==1:
                    coefficient=-1j*(lapse[here]+lapse[there])/4
                    for a in range(4):
                        for b in range(4):
                            value=coefficient*alpha[axis,a,b]
                            if value!=0:
                                r,c=a*count+i,b*count+j
                                rows.extend((r,c));cols.extend((c,r));values.extend((value,value.conjugate()))
                else:
                    value=-1j*(flow[(axis,)+here]+flow[(axis,)+there])/8
                    if value!=0:
                        for a in range(4):
                            r,c=a*count+i,a*count+j
                            rows.extend((r,c));cols.extend((c,r));values.extend((value,value.conjugate()))
    h=coo_matrix((np.asarray(values,dtype=np.complex128),(rows,cols)),shape=(ROWS,ROWS)).tocsr()
    h.sum_duplicates();h.eliminate_zeros();h.sort_indices();return h

def native_bound_arrays(native):
    # The literal NativeHamiltonian coefficient convention, without constructing
    # its neighbor arrays, importing its JIT kernel, or applying it to a column.
    n=81;sites=n**3;indices=np.arange(sites);coord=np.indices((n,n,n)).reshape(3,sites)
    kept=np.sum((coord-40)**2,axis=0)<=40**2
    nf=native['native_N'].reshape(-1);vf=native['native_flow'].reshape(3,sites)
    row=np.zeros(sites,np.float64);column=row.copy()
    for axis,stride in enumerate((n*n,n,1)):
        for distance in (1,2):
            for direction in (1,-1):
                boundary();target_coord=coord[axis]+direction*distance
                valid=(target_coord>=0)&(target_coord<n)
                target=indices+((target_coord%n)-coord[axis])*stride
                valid &= kept & kept[target]
                field=nf if distance==1 else vf[axis]
                magnitude=np.abs(-direction*(field[valid]+field[target[valid]])/(4. if distance==1 else 8.))
                row[valid]+=magnitude;np.add.at(column,target[valid],magnitude)
    return {'row_sum':row,'column_sum':column}

def scientific_unit(rec):
    from lead_channels import sampled_frame
    from scipy.sparse import csr_matrix
    tile,labels,native,h_TT,inputs=decode_inputs(rec)
    old_ref=lambda key:member_ref(inputs['tile'],key)
    geom_refs=[old_ref('points'),old_ref('global_dofs'),member_ref(inputs['labels'],'labels10')]
    with rec.phase('A'):
        coords=(np.indices((81,81,81),dtype=np.int64)-40).reshape(3,-1).T
        r2=np.sum(coords*coords,axis=1);sites=np.flatnonzero((r2>32**2)&(r2<=34**2)).astype(np.int64)
        points=coords[sites];del coords,r2
        n=len(points);r=n-1024
        dofs=(np.arange(4,dtype=np.int64)[:,None]*81**3+sites[None]).reshape(-1)
        trows=(np.arange(4,dtype=np.int64)[:,None]*n+np.arange(1024,dtype=np.int64)[None]).reshape(-1)
        rrows=(np.arange(4,dtype=np.int64)[:,None]*n+np.arange(1024,n,dtype=np.int64)[None]).reshape(-1)
        g={'points':points,'sites':sites,'dofs':dofs,'T_rows':trows,'R_rows':rrows,'points_R':points[1024:].copy(),'dofs_R':dofs[rrows]}
        rec.save('geometry.npz',g,geom_refs)
        rec.control('F.geometry',lambda:{'n':n,'r':r,'m':4*n,'p':len(labels),'tile_points_equal':bool(np.array_equal(points[:1024],tile['points'])),'tile_dofs_equal':bool(np.array_equal(dofs[trows],tile['global_dofs']))},lambda d:d=={'n':NPOINTS,'r':RPOINTS,'m':ROWS,'p':WIDTH,'tile_points_equal':True,'tile_dofs_equal':True},{'n':NPOINTS,'r':RPOINTS,'m':ROWS,'p':WIDTH,'tile_points_equal':True,'tile_dofs_equal':True},[ref(rec,'geometry.npz')]+geom_refs)
        ar=rec.create_map('A_R.npy',(RESTROWS,WIDTH));batch_refs=[]
        for start,stop in rec.chunks['A_R.npy']:
            boundary();resources(rec,'before_angular_'+str(start));before=resource_state()
            chunk=sampled_frame(g['points_R'][start:stop],10)
            cost=elapsed(before);length=stop-start
            def write():
                for spin in range(4):ar[spin*r+start:spin*r+stop]=chunk[spin*length:(spin+1)*length]
            batch_refs.append(rec.tile('A_R.npy',(start,stop),chunk,ar,write,[ref(rec,'geometry.npz','points_R'),member_ref(inputs['labels'],'labels10')],cost))
            del chunk
        rec.finish_map('A_R.npy',ar,batch_refs)
        rec.control('F.A_finite',lambda:{'all_finite':finite(ar,tile['A'])},lambda d:d['all_finite'],{'all_finite':True},[ref(rec,'A_R.npy'),old_ref('A')])
    with rec.phase('G'):
        gr=rec.calculate('G_R.npy',lambda:ar.conj().T@ar,[ref(rec,'A_R.npy')])
        raw=rec.calculate('G_raw.npy',lambda:tile['G']+gr,[old_ref('G'),ref(rec,'G_R.npy')]);del gr
        defect=rec.calculate('G_defect.npy',lambda:raw-raw.conj().T,[ref(rec,'G_raw.npy')])
        rec.control('F.Gram_finite',lambda:{'all_finite':finite(raw,defect,tile['G'])},lambda d:d['all_finite'],{'all_finite':True},[old_ref('G'),ref(rec,'G_raw.npy'),ref(rec,'G_defect.npy')])
        norm1=scalar(rec,'G_norm_one',lambda:(float(np.linalg.norm(raw,1))),[ref(rec,'G_raw.npy')]);norminf=scalar(rec,'G_norm_infinity',lambda:(float(np.linalg.norm(raw,np.inf))),[ref(rec,'G_raw.npy')])
        bg=scalar(rec,'b_G',lambda:(math.sqrt(norm1*norminf)),[rec.row_refs['returned_scalar','G_norm_one'],rec.row_refs['returned_scalar','G_norm_infinity']])
        fd=scalar(rec,'G_defect_Frobenius',lambda:(float(np.linalg.norm(defect))),[ref(rec,'G_defect.npy')]);boundary();limit=5e-12*max(1.,bg)
        rec.control('F.Gram_Hermitian',lambda:{'defect_Frobenius':fd,'b_G':bg},lambda d:math.isfinite(d['b_G']) and math.isfinite(d['defect_Frobenius']) and d['defect_Frobenius']<=limit,{'maximum':limit},[ref(rec,'G_defect.npy'),rec.row_refs['returned_scalar','b_G'],rec.row_refs['returned_scalar','G_defect_Frobenius']]);del defect
        gh=rec.calculate('G_H.npy',lambda:(raw+raw.conj().T)/2,[ref(rec,'G_raw.npy')])
        difference=rec.calculate('G_realization_difference.npy',lambda:gh-raw,[ref(rec,'G_H.npy'),ref(rec,'G_raw.npy')]);del difference,raw
    with rec.phase('L'):
        def eigensystem():
            values,vectors=np.linalg.eigh(gh);return {'values':values,'vectors':vectors}
        eig=rec.calculate('eigensystem.npz',eigensystem,[ref(rec,'G_H.npy')]);del gh
        values=eig['values'];vectors=eig['vectors']
        rec.control('F.spectrum',lambda:{'all_finite':finite(values,vectors),'minimum':float(np.min(values)),'maximum':float(np.max(values))},lambda d:d['all_finite'] and d['minimum']>=.25 and d['maximum']<=4.,{'all_finite':True,'minimum_at_least':.25,'maximum_at_most':4.},[ref(rec,'eigensystem.npz')])
    with rec.phase('B'):
        roots=rec.calculate('inverse_eigenvalues_sqrt.npy',lambda:values**(-.5),[ref(rec,'eigensystem.npz','values')])
        scaled=rec.calculate('scaled_eigenvectors.npy',lambda:vectors*roots[None,:],[ref(rec,'eigensystem.npz','vectors'),ref(rec,'inverse_eigenvalues_sqrt.npy')])
        inverse=rec.calculate('inverse_half.npy',lambda:scaled@vectors.conj().T,[ref(rec,'scaled_eigenvectors.npy'),ref(rec,'eigensystem.npz','vectors')])
        rec.control('F.inverse_half_finite',lambda:{'all_finite':finite(roots,scaled,inverse)},lambda d:d['all_finite'],{'all_finite':True},[ref(rec,n) for n in ('inverse_eigenvalues_sqrt.npy','scaled_eigenvectors.npy','inverse_half.npy')]);del roots,scaled,vectors,values,eig
    with rec.phase('E'):
        resources(rec,'before_virtual_A_copy',workspace=ROWS*WIDTH*16)
        before=resource_state();a=np.empty((ROWS,WIDTH),np.complex128);a[trows]=tile['A'];a[rrows]=ar
        rec.event('metrics','virtual_row_scatter','A',{'copy_cost':elapsed(before),'shape':list(a.shape),'public_full_A_file':False},[old_ref('A'),ref(rec,'A_R.npy'),ref(rec,'geometry.npz')]);del ar
        e=rec.create_map('E.npy',(ROWS,WIDTH));e_refs=[]
        for start,stop in rec.chunks['E.npy']:
            boundary();before=resource_state();chunk=a[start:stop]@inverse;cost=elapsed(before)
            e_refs.append(rec.tile('E.npy',(start,stop),chunk,e,lambda:e.__setitem__(slice(start,stop),chunk),[old_ref('A'),ref(rec,'A_R.npy'),ref(rec,'inverse_half.npy'),ref(rec,'geometry.npz')],cost));del chunk
        rec.finish_map('E.npy',e,e_refs)
        rec.control('F.E_finite',lambda:{'all_finite':finite(e)},lambda d:d['all_finite'],{'all_finite':True},[ref(rec,'E.npy')])
    with rec.phase('O'):
        ge=rec.calculate('E_gram.npy',lambda:e.conj().T@e,[ref(rec,'E.npy')]);del e
        de=rec.calculate('E_gram_defect.npy',lambda:ge-np.eye(WIDTH,dtype=np.complex128),[ref(rec,'E_gram.npy')]);del ge
        epsilon=scalar(rec,'E_Gram_maximum_absolute_defect',lambda:(float(np.max(np.abs(de)))),[ref(rec,'E_gram_defect.npy')])
        rec.control('F.E_Gram',lambda:{'all_finite':finite(de),'maximum_absolute_defect':epsilon},lambda d:d['all_finite'] and math.isfinite(d['maximum_absolute_defect']) and d['maximum_absolute_defect']<=5e-12,{'all_finite':True,'maximum':5e-12},[ref(rec,'E_gram_defect.npy'),rec.row_refs['returned_scalar','E_Gram_maximum_absolute_defect']]);del de
        be=scalar(rec,'b_E',lambda:(math.sqrt(1+WIDTH*epsilon)),[rec.row_refs['returned_scalar','E_Gram_maximum_absolute_defect']])
    with rec.phase('H'):
        boundary();before=resource_state();h=missing_h(points,native['native_N'],native['native_flow']);h_arrays=csr_arrays(h)
        rec.save('H_delta.npz',h_arrays,[ref(rec,'geometry.npz'),member_ref(inputs['native'],'native_N'),member_ref(inputs['native'],'native_flow'),inputs['H_TT']],elapsed(before))
        rec.control('P.H_schema',lambda:{'canonical_sorted':bool(h.has_sorted_indices and h.has_canonical_format),'finite_nonzero':finite(h.data) and bool(np.all(h.data!=0)),'TT_nnz':int(h[trows][:,trows].nnz),'nnz':int(h.nnz),'shape':list(h.shape),'indices_dtype':h_arrays['indices'].dtype.str,'indptr_dtype':h_arrays['indptr'].dtype.str},lambda d:d['canonical_sorted'] and d['finite_nonzero'] and d['TT_nnz']==0 and d['nnz']<=12*ROWS and d['shape']==[ROWS,ROWS] and d['indices_dtype']=='<i8' and d['indptr_dtype']=='<i8',{'canonical_sorted':True,'finite_nonzero':True,'TT_nnz':0,'nnz_at_most':12*ROWS,'shape':[ROWS,ROWS],'indices_dtype':'<i8','indptr_dtype':'<i8'},[ref(rec,'H_delta.npz'),ref(rec,'geometry.npz','T_rows')]);del h_arrays
        boundary();before=resource_state();hd=h-h.conj().T;hd.sum_duplicates();hd.eliminate_zeros();hd.sort_indices()
        rec.save('H_delta_defect.npz',csr_arrays(hd),[ref(rec,'H_delta.npz')],elapsed(before))
        maximum=scalar(rec,'H_delta_adjoint_maximum_absolute',lambda:(float(np.max(np.abs(hd.data))) if hd.nnz else 0.),[ref(rec,'H_delta_defect.npz')]);del hd
        rec.control('P.H_delta_Hermitian',lambda:{'maximum_absolute':maximum},lambda d:math.isfinite(d['maximum_absolute']) and d['maximum_absolute']==0.,{'maximum_absolute':0.},[ref(rec,'H_delta_defect.npz'),rec.row_refs['returned_scalar','H_delta_adjoint_maximum_absolute']])
        bound=rec.calculate('native_bound.npz',lambda:native_bound_arrays(native),[member_ref(inputs['native'],'native_N'),member_ref(inputs['native'],'native_flow')])
        b_finite=scalar(rec,'native_bound_all_finite',lambda:finite(*bound.values()),[ref(rec,'native_bound.npz')])
        normrow=scalar(rec,'native_maximum_row_sum',lambda:(float(np.max(bound['row_sum']))),[ref(rec,'native_bound.npz','row_sum')]);normcol=scalar(rec,'native_maximum_column_sum',lambda:(float(np.max(bound['column_sum']))),[ref(rec,'native_bound.npz','column_sum')])
        bh=scalar(rec,'b_H',lambda:(math.sqrt(normrow*normcol)),[rec.row_refs['returned_scalar','native_maximum_row_sum'],rec.row_refs['returned_scalar','native_maximum_column_sum']]);del bound
        rec.control('P.bound',lambda:{'all_finite':b_finite,'b_H':bh},lambda d:d['all_finite'] and math.isfinite(d['b_H']) and d['b_H']>0,{'all_finite':True,'b_H_positive':True},[ref(rec,'native_bound.npz'),rec.row_refs['returned_scalar','b_H'],rec.row_refs['returned_scalar','native_bound_all_finite']])
    with rec.phase('U'):
        u=rec.create_map('U_delta.npy',(ROWS,WIDTH));u_refs=[]
        for start,stop in rec.chunks['U_delta.npy']:
            boundary();before=resource_state();chunk=h@a[:,start:stop];cost=elapsed(before)
            u_refs.append(rec.tile('U_delta.npy',(start,stop),chunk,u,lambda:u.__setitem__((slice(None),slice(start,stop)),chunk),[ref(rec,'H_delta.npz'),old_ref('A'),ref(rec,'A_R.npy'),ref(rec,'geometry.npz')],cost));del chunk
        rec.finish_map('U_delta.npy',u,u_refs);del h
    with rec.phase('P'):
        pd=rec.calculate('P_delta.npy',lambda:a.conj().T@u,[old_ref('A'),ref(rec,'A_R.npy'),ref(rec,'geometry.npz'),ref(rec,'U_delta.npy')]);del a
        praw=rec.calculate('P_raw.npy',lambda:tile['projected_native_tile']+pd,[old_ref('projected_native_tile'),ref(rec,'P_delta.npy')])
        rec.control('P.raw_finite',lambda:{'all_finite':finite(praw,tile['projected_native_tile'],pd,u)},lambda d:d['all_finite'],{'all_finite':True},[ref(rec,n) for n in ('U_delta.npy','P_delta.npy','P_raw.npy')]+[old_ref('projected_native_tile')]);del pd,u
    with rec.phase('D'):
        bp=rec.calculate('BP.npy',lambda:inverse.conj().T@praw,[ref(rec,'inverse_half.npy'),ref(rec,'P_raw.npy')]);bp_finite=scalar(rec,'BP_all_finite',lambda:finite(bp),[ref(rec,'BP.npy')]);del praw
        draw=rec.calculate('D_raw.npy',lambda:bp@inverse,[ref(rec,'BP.npy'),ref(rec,'inverse_half.npy')]);del bp,inverse
        dd=rec.calculate('D_defect.npy',lambda:draw-draw.conj().T,[ref(rec,'D_raw.npy')])
        defect=scalar(rec,'D_defect_Frobenius',lambda:(float(np.linalg.norm(dd))),[ref(rec,'D_defect.npy')]);boundary();limit=5e-12*max(1.,bh*be*be)
        rec.control('P.diagonal_Hermitian',lambda:{'all_finite':bp_finite and finite(draw,dd),'defect_Frobenius':defect},lambda d:d['all_finite'] and math.isfinite(d['defect_Frobenius']) and d['defect_Frobenius']<=limit,{'all_finite':True,'maximum':limit},[ref(rec,'BP.npy'),ref(rec,'D_raw.npy'),ref(rec,'D_defect.npy'),rec.row_refs['returned_scalar','b_H'],rec.row_refs['returned_scalar','b_E'],rec.row_refs['returned_scalar','BP_all_finite'],rec.row_refs['returned_scalar','D_defect_Frobenius']]);del dd
        dh=rec.calculate('D_H.npy',lambda:(draw+draw.conj().T)/2,[ref(rec,'D_raw.npy')])
        diff=rec.calculate('D_realization_difference.npy',lambda:dh-draw,[ref(rec,'D_H.npy'),ref(rec,'D_raw.npy')]);del draw,dh,diff
    rec.checkpoint()

def main():
    global np,PHASE
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--outdir',required=True);args=p.parse_args()
    launch=float(os.environ['BPRIME_CHILD_LAUNCH']);external_end=launch+3600
    for sig in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM,signal.SIGXCPU):signal.signal(sig,handler)
    signal.setitimer(signal.ITIMER_REAL,max(.001,launch+3480-time.monotonic()))
    rec=None;startup=resource_state()
    try:
        if Path(args.outdir)!=OUT:raise ValueError('fixed output path')
        expected=loads(os.environ['BPRIME_FRAME_PINS']);rec=Records(OUT,expected)
        if os.getpriority(os.PRIO_PROCESS,0)<10 or len(os.sched_getaffinity(0))!=8:raise ValueError('CPU admission before scientific imports')
        if any(os.environ.get(k)!='8' for k in THREADS) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('thread admission before imports')
        import numpy as numpy
        np=numpy
        resources(rec,'entry',48*1024**3)
        n=NPOINTS;m=ROWS;mr=RESTROWS;mt=4096;r0=RPOINTS;p0=WIDTH
        storage=16*((mr+2*m)*p0+17*p0*p0)+16*p0+16*81**3+8*(4*n+m+mt+2*mr+3*r0)+880*m+48+25*4096+METADATA_CAP
        disk=os.statvfs(ROOT);free=disk.f_bavail*disk.f_frsize
        rec.event('metrics','storage_preflight','storage_preflight',{'fixed_dtype_storage_upper_bound':storage,'numeric_cap':NUMERIC_CAP,'metadata_cap':METADATA_CAP,'free_disk':free,'disk_margin':1024**3,'aggregate_reserved_CPU':11299.310702,'aggregate_cap_CPU':28800})
        if storage>NUMERIC_CAP+METADATA_CAP or free<NUMERIC_CAP+METADATA_CAP+1024**3:raise Stopped('dtype/disk storage admission')
        rec.source_docs,parents,krefs=admit(rec)
        if rec.source_docs['k10']['field_association']['mass']!=0. or rec.source_docs['k10']['field_association']['spacing']!=1.:raise ValueError('fixed native mass/spacing')
        rec.metadata_intervals['startup_import_admission']=elapsed(startup);rec.event('metrics','metadata_interval','startup_import_admission',rec.metadata_intervals['startup_import_admission'])
        boundary();scientific_unit(rec)
    except BaseException as exc:
        if rec is None:traceback.print_exc();return 2
        rec.error=failure(exc)
        if 'startup_import_admission' not in rec.metadata_intervals:rec.metadata_intervals['startup_import_admission']=elapsed(startup)
        try:rec.checkpoint()
        except BaseException as secondary:rec.secondary.append(failure(secondary))
    finally:
        if rec is not None:
            PHASE='postchecks';post_before=resource_state();end=min(time.monotonic()+90,external_end);remaining=end-time.monotonic()
            if remaining>0:
                signal.setitimer(signal.ITIMER_REAL,remaining)
                try:
                    for expected in rec.expected:
                        if time.monotonic()>=end:raise Stopped('absolute postcheck deadline')
                        path=expected['path'];receipt={'expected':expected,'matched':False};rec.postchecks[path]=receipt
                        try:
                            returned=inspect(path,expected['sha256']);receipt['returned']=returned
                            if returned!=expected:raise ValueError('input changed across attempt')
                            receipt['matched']=True
                        except BaseException as exc:
                            receipt['failure']=failure(exc)
                            if isinstance(exc,Stopped):raise
                            rec.secondary.append(failure(exc))
                except BaseException as exc:rec.secondary.append(failure(exc))
                finally:signal.setitimer(signal.ITIMER_REAL,0)
            rec.metadata_intervals['producer_terminal_postchecks']=elapsed(post_before)
            PHASE='final_record';end=min(time.monotonic()+10,external_end);remaining=end-time.monotonic()
            if remaining>0:
                signal.setitimer(signal.ITIMER_REAL,remaining)
                try:rec.checkpoint(terminal=True)
                except BaseException as exc:rec.secondary.append(failure(exc));traceback.print_exc()
                finally:signal.setitimer(signal.ITIMER_REAL,0)
            for f in rec.streams.values():f.close()
    if rec is None:return 2
    return 0 if len(rec.saved)==25 and len(rec.decisions)==13 and all(rec.decisions.values()) and rec.error is None and not rec.secondary and all(v.get('matched') for v in rec.postchecks.values()) and len(rec.postchecks)==len(rec.expected) and not STOP else 2

if __name__=='__main__':sys.exit(main())
