"""Frozen CORE section 43: first missing channel-one Gram, three checkpoints."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import time

ROOT=Path('<internal>/rung4_T13_gram_k01_v1')
PREVIOUS=ROOT.parent/'rung4_T13_number_density_v1'
DEPENDENCIES=ROOT.parent/'profiles_v1'
SOURCE=Path('<internal>/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_T0013.00.npz')
INHERITED_SOURCE_SHA='072549f046e22084e19a4cf8f48abe750f84ae0a5a5f414c6716902e00870c78'
EXPECTED_ID=dict(size=2685802106,device=2048,inode=17891544,mtime_ns=1792305468347378926,ctime_ns=1792454095501898249)
PINS={'context.npz':'9cf47d19ed14128edd0964ac851c29b1f1fe71fa74354122fa3971f9ee33aa53',
 'rung4_T13_number_density_result.json':'8a9f30a392b05c5faecdf2bb96865cb83ef33137df84974de9c7a1622db4703c',
 'execution.json':'7a0f9f375ac29ed88bcb8648a0f5f1a81b4b059abb3c61dc42e320ebd84ab7fb'}
MODULE_PINS={'core_well_profiles_v1.py':'8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54',
 'core_well_input_v1.py':'294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b'}
PRODUCTS={'raw':'gram_raw_k01.npz','defect':'gram_hermitian_defect_k01.npz','sym':'gram_sym_k01.npz'}
VALIDATIONS=('input_binding','joint_columns_finite','raw_gram_finite','hermitian_defect_finite','sym_gram_finite','final_source_identity')
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
SPEC_SHA='5567137dd04b26ab785e7887f3414d1b30c02939b4f22c19578b3fd18590f367'
REPORT={}
START=0


class StageStop(Exception):pass


@contextmanager
def protected():
    old=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:yield
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,old)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def identity(state):
    return dict(size=state.st_size,device=state.st_dev,inode=state.st_ino,mtime_ns=state.st_mtime_ns,ctime_ns=state.st_ctime_ns)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not math.isfinite(value):return dict(nonfinite=repr(value))
    return value


def save():
    with protected():
        REPORT.update(elapsed_seconds=time.monotonic()-START,process_cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        REPORT['missing_products']=[name for name in PRODUCTS.values() if not (ROOT/name).is_file()]
        REPORT['missing_validations']=[name for name in VALIDATIONS if not REPORT['validations'][name]]
        temp=ROOT/'rung4_T13_gram_k01_result.json.tmp'
        with temp.open('x') as stream:
            json.dump(clean(REPORT),stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        temp.replace(ROOT/'rung4_T13_gram_k01_result.json')


def safe_boundary():
    if time.monotonic()-START>=160:
        REPORT['signal_flags']['interruption_requested']=True
        REPORT['signal_flags']['elapsed_deadline_observed']=True
    if REPORT['signal_flags']['interruption_requested']:
        save();raise StageStop('Handled interruption at completed-stage boundary')


def begin(name):
    safe_boundary()
    REPORT['phase']=name;REPORT['active_stage']=name
    state=REPORT['stages'].setdefault(name,{})
    state.update(status='started',wall_start=time.monotonic()-START,cpu_start=time.process_time())
    save();return time.monotonic(),time.process_time()


def end(name,started,status):
    REPORT['stages'][name].update(status=status,wall_elapsed=time.monotonic()-started[0],cpu_elapsed=time.process_time()-started[1])
    REPORT['active_stage']=None;save()


def current_source_identity(source,when):
    row=dict(when=when,descriptor=identity(os.fstat(source.stream.fileno())),path=identity(SOURCE.stat()))
    REPORT['source_identity_checks'].append(row);save()
    if row['descriptor']!=EXPECTED_ID or row['path']!=EXPECTED_ID:raise ValueError('Original source identity changed: '+when)
    source.verify_identity()


def read_bound(path,pin,label):
    with path.open('rb') as stream:
        before=identity(os.fstat(stream.fileno()));raw=stream.read();after=identity(os.fstat(stream.fileno()))
    row=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),expected_sha256=pin,before=before,after=after,path_after=identity(path.stat()))
    REPORT['bindings'][label]=row;save()
    if before!=after or after!=row['path_after'] or row['sha256']!=pin:raise ValueError('Pinned input changed: '+label)
    return raw


def validate(name,passed):
    REPORT['validations'][name]=bool(passed);save()
    if not passed:raise ValueError('Finite/binding check failed: '+name)


def checkpoint(name,key,value,validation):
    state=REPORT['stages'][name]
    state.update(returned=True,returned_wall=time.monotonic()-START,returned_cpu=time.process_time(),
        schema=dict(key=key,keys=[key],storage='ZIP_STORED',shape=list(value.shape),dtype=value.dtype.str,c_contiguous=bool(value.flags.c_contiguous),f_contiguous=bool(value.flags.f_contiguous)),saved=False,validated=False)
    save()
    path=ROOT/PRODUCTS[name];temporary=ROOT/(PRODUCTS[name]+'.tmp')
    try:
        with protected():
            with temporary.open('xb') as stream:
                np.savez(stream,**{key:value});stream.flush();os.fsync(stream.fileno())
            temporary.replace(path)
            state.update(saved=True,path=str(path),size=path.stat().st_size,sha256=digest(path),saved_wall=time.monotonic()-START,saved_cpu=time.process_time())
            save()
    except Exception as error:
        REPORT['preservation_error']=dict(stage=name,type=type(error).__name__,message=str(error),temporary=str(temporary),temporary_exists=temporary.exists(),final_exists=path.exists())
        if temporary.exists():REPORT['preservation_error']['temporary_size']=temporary.stat().st_size
        raise
    finite=bool(np.isfinite(value).all())
    state['finite']=finite;state['validated']=finite
    if name=='defect':REPORT['hermitian_defect']=float(value)
    validate(validation,finite)


def produce(name,key,operation,validation):
    started=begin(name)
    value=operation()
    checkpoint(name,key,np.asarray(value),validation)
    end(name,started,'validated')
    return value


def main():
    global np,REPORT,START
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box root only')
    allowed={'rung4_T13_gram_k01_v1.py','run_rung4_T13_gram_k01_v1.py','execution.json','execution.json.tmp','stdout.log','stderr.log'}
    if any(p.name not in allowed for p in ROOT.iterdir()):raise FileExistsError('Prior output or temporary; no retry')
    if shutil.disk_usage('/').free<100_000_000_000 or shutil.disk_usage('/data').free<8*2**30:raise RuntimeError('Standing disk reserve')
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    if available<8*2**30:raise RuntimeError('Available RAM reserve')
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
    for name in names:os.environ[name]='4'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    priority=os.getpriority(os.PRIO_PROCESS,0)
    if priority<10:os.nice(10-priority)
    cpus=sorted(os.sched_getaffinity(0))[:4]
    if len(cpus)!=4:raise RuntimeError('Four allocated CPUs required')
    os.sched_setaffinity(0,cpus)
    resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));resource.setrlimit(resource.RLIMIT_CPU,(700,720))
    START=time.monotonic()
    REPORT=dict(status='incomplete',target_complete=False,definition_spec_sha256=SPEC_SHA,
        label='v36d width-four from T=0, T/rs=13, kappa=1, Gram-only finite stored-state operand',
        quantity='H=Phi.conj().T @ Phi; delta_H=float(np.max(np.abs(H-H.conj().T))); G=(H+H.conj().T)*.5',
        units='dimensionless discrete-column Gram; original interleaved row/archived column order',
        declared_units=dict(a=1,mu=.2,g=.25,sigma_over_a=4,tau_over_a=8,rs=64,T=832,kappa=1,dust_columns=5,weight=16,weights_applied_to_Gram=False),
        source=dict(path=str(SOURCE),sha256_inherited=INHERITED_SOURCE_SHA,fresh_whole_file_hash_verified=False,expected_identity=EXPECTED_ID),
        phase='initial',active_stage=None,bindings={},source_identity_checks=[],operand_members={},operand=None,hermitian_defect=None,
        stages={name:dict(status='unvisited',returned=False,saved=False,validated=False) for name in ('raw','defect','sym')},
        validations={name:False for name in VALIDATIONS},missing_products=list(PRODUCTS.values()),missing_validations=list(VALIDATIONS),
        downstream_missing_products=['polar_k%02d.npz'%k for k in range(1,6)]+['number_k%02d.npz'%k for k in range(1,6)]+['combined.npz'],
        excluded=dict(positive_Gram_validation=False,eigensolve=False,inverse_root=False,selected_columns=False,number=False,density=False,radius=False,other_channels=False,context_regenerated=False,new_fixture=False),
        signal_flags=dict(interruption_requested=False,first_signal=None,first_signal_wall=None,elapsed_deadline_observed=False),
        resources=dict(external_wall_seconds=180,handled_wall_seconds=160,cpu_soft_seconds=700,cpu_hard_seconds=720,address_space_bytes=4*2**30,available_memory_bytes=available,threads=4),
        own_sources={name:digest(ROOT/name) for name in ('rung4_T13_gram_k01_v1.py','run_rung4_T13_gram_k01_v1.py')},
        hard_stop_limitation='Returned-stage preservation requires checkpoint completion before external/hard-CPU termination; unfinished operations/writes cannot be guaranteed',error=None,preservation_error=None)
    def request_stop(signum,frame):
        flags=REPORT['signal_flags']
        if flags['first_signal'] is None:flags['first_signal']=signum;flags['first_signal_wall']=time.monotonic()-START
        flags['interruption_requested']=True
    signal.signal(signal.SIGALRM,request_stop);signal.signal(signal.SIGXCPU,request_stop);signal.alarm(160)
    save();source=None
    try:
        started=begin('input_binding')
        previous={name:read_bound(PREVIOUS/name,pin,'predecessor:'+name) for name,pin in PINS.items()}
        prior=json.loads(previous['rung4_T13_number_density_result.json']);execution=json.loads(previous['execution.json'])
        if prior['status']!='incomplete' or prior['phase']!='joint_polar' or prior['polar_completed'] or prior['number_completed']:raise ValueError('Prior missing-only extent changed')
        if prior['source']['actual_sha256']!=INHERITED_SOURCE_SHA or prior['source']['path']!=str(SOURCE) or prior['source']['descriptor_before']!=EXPECTED_ID or prior['source']['descriptor_after']!=EXPECTED_ID:raise ValueError('Inherited source admission binding')
        if execution['status']!='finished' or execution['exit_code']!=1 or execution['rung4_T13_number_density_result.json']['sha256']!=PINS['rung4_T13_number_density_result.json']:raise ValueError('Predecessor execution binding')
        context=next(row for row in prior['arrays'] if row['path']==str(PREVIOUS/'context.npz'))
        if context['sha256']!=PINS['context.npz'] or context['size']!=len(previous['context.npz']):raise ValueError('Context binding')
        REPORT['inherited_admission']=dict(source=prior['source'],context=context,original_error=prior['error'],meaning='Inherited input/context admission only; no joint-polar result or scientific control')
        for name,pin in MODULE_PINS.items():read_bound(DEPENDENCIES/name,pin,'dependency:'+name)
        import numpy as _np
        np=_np
        spec=importlib.util.spec_from_file_location('owned_gram_numeric_reader',DEPENDENCIES/'core_well_input_v1.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        REPORT['implementation']=dict(numpy=np.__version__,nice=os.getpriority(os.PRIO_PROCESS,0),affinity=sorted(os.sched_getaffinity(0)),environment={name:os.environ.get(name) for name in names+('CUDA_VISIBLE_DEVICES',)})
        source=module.StoredNpz(SOURCE)
        current_source_identity(source,'opened_before_mapping')
        validate('input_binding',True);end('input_binding',started,'validated')
        started=begin('assemble_joint_columns')
        header_rows={};mapped={}
        for name,shape in (('sea_1',(4096,2048)),('dust_1',(4096,5))):
            current_source_identity(source,'before_'+name)
            header=source.member_header(name)
            header_rows[name]=dict(shape=list(header['shape']),dtype=header['dtype'].str,fortran_order=header['fortran_order'],offset=header['offset'],data_bytes=header['data_bytes'],zip_crc32=header['zip_crc32'])
            REPORT['operand_members']=header_rows;save()
            if header['shape']!=shape or header['dtype']!=np.dtype('complex128'):raise ValueError('Original column header mismatch: '+name)
            mapped[name]=source.array(name)
            current_source_identity(source,'after_'+name)
            safe_boundary()
        Phi=np.empty((4096,2053),dtype=np.complex128,order='F')
        Phi[:,:2048],Phi[:,2048:]=mapped['sea_1'],mapped['dust_1']
        del mapped
        REPORT['operand']=dict(shape=list(Phi.shape),dtype=Phi.dtype.str,c_contiguous=bool(Phi.flags.c_contiguous),f_contiguous=bool(Phi.flags.f_contiguous),column_order=['sea_1:0:2048','dust_1:0:5'],persisted=False,complete_original_member_references=True)
        current_source_identity(source,'after_joint_copy');save();safe_boundary()
        validate('joint_columns_finite',bool(np.isfinite(Phi).all()));end('assemble_joint_columns',started,'validated')
        H=produce('raw','gram_raw',lambda:Phi.conj().T @ Phi,'raw_gram_finite')
        del Phi
        defect=produce('defect','hermitian_defect',lambda:float(np.max(np.abs(H-H.conj().T))),'hermitian_defect_finite')
        G=produce('sym','gram_sym',lambda:(H+H.conj().T)*.5,'sym_gram_finite')
        del H,G,defect
    except StageStop as error:REPORT['stop_reason']=str(error)
    except Exception as error:REPORT['error']=dict(type=type(error).__name__,message=str(error),phase=REPORT['phase'])
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN)
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(720,720))
        try:
            if source is not None:
                current_source_identity(source,'terminal_checkpoint');REPORT['validations']['final_source_identity']=True
        except Exception as error:REPORT['final_source_identity_error']=dict(type=type(error).__name__,message=str(error))
        REPORT['surviving_final_files']={name:dict(path=str(ROOT/name),size=(ROOT/name).stat().st_size) for name in PRODUCTS.values() if (ROOT/name).is_file()}
        REPORT['unfinished_temporaries']={p.name:dict(size=p.stat().st_size) for p in ROOT.glob('*.npz.tmp')}
        REPORT['target_complete']=all(REPORT['validations'].values()) and all(REPORT['stages'][name]['saved'] and REPORT['stages'][name]['validated'] for name in PRODUCTS)
        REPORT['status']='completed_rung4_T13_gram_k01_only' if REPORT['target_complete'] and REPORT['error'] is None and REPORT['preservation_error'] is None else 'incomplete'
        save()
        if source is not None:source.close()
    print(json.dumps(dict(status=REPORT['status'],target_complete=REPORT['target_complete'],missing_products=REPORT['missing_products'],missing_validations=REPORT['missing_validations'],error=REPORT['error'])))
    if REPORT['status']!='completed_rung4_T13_gram_k01_only':raise SystemExit(1)


if __name__=='__main__':main()
