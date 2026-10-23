"""One missing fixed rung slice: joint-dust half-number radius and first-eight radial densities."""
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

ROOT=Path('<internal>/rung4_T13_number_density_v1')
BASE=ROOT.parent/'profiles_v1'
SOURCE=Path('<internal>/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_T0013.00.npz')
SOURCE_SHA='072549f046e22084e19a4cf8f48abe750f84ae0a5a5f414c6716902e00870c78'
PINS={BASE/'core_well_profiles_v1.py':'8a55cd10be12b2b0b4c36f8b364c31dd198ccc8bdf114ea4bb0c2d78ec86fd54',BASE/'core_well_input_v1.py':'294ebd7e68dd8f260f4be2a70c8d250db0f00106679edbea8e2274d83c4f326b'}
SIGNALS={signal.SIGALRM,signal.SIGXCPU}
COUNTS=(5,4,3,2,2,0,0,0,0,0)


def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def main():
 if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory only')
 target=ROOT/'rung4_T13_number_density_result.json'
 if target.exists() or target.with_suffix('.json.tmp').exists() or any(ROOT.glob('*.npz*')):raise FileExistsError('Never retry/overwrite a bounded allocation')
 if shutil.disk_usage('/').free<100_000_000_000 or shutil.disk_usage('/data').free<8*2**30:raise RuntimeError('Disk reserve')
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[name]='4'
 os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
 available_memory=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
 if available_memory<8*2**30:raise RuntimeError('Available RAM reserve')
 cpus=sorted(os.sched_getaffinity(0))[:4]
 if len(cpus)!=4:raise RuntimeError('Four CPUs required')
 os.sched_setaffinity(0,cpus)
 if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
 resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));resource.setrlimit(resource.RLIMIT_CPU,(700,720))
 def deadline(signum,frame):
  if report.get('deadline_seen'):return
  report['deadline_seen']=True;signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  raise TimeoutError('Fixed one-slice allocation deadline')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(160)
 start=time.monotonic();np=None
 report=dict(status='incomplete',purpose='Task3 named missing width-four T13 half-number radius and first-eight radial-cell densities',
  label='v36d width-four rung, another engine from T=0, T/rs=13, finite stored state; half-number radius and first-eight radial-cell densities',
  scope=dict(original_c27=False,r11=False,history=False,static_three_width_HF=False,native_HF=False,native_cartesian_saturation=False),
  source=dict(path=str(SOURCE),sha256=SOURCE_SHA),dependencies={str(p):h for p,h in PINS.items()},
  declared_units=dict(a=1,mu=.2,g=.25,sigma_over_a=4,tau_over_a=8,clock_over_rs=13),
  resources=dict(available_memory_bytes=available_memory,minimum_available_memory_bytes=8*2**30,threads=4,cpu_affinity=cpus,nice=10,address_space_bytes=4*2**30,cpu_soft=700,cpu_hard=720,wall_internal=160,wall_external=180),
  phase='input_binding',current_channel=None,completion_list_meaning='completed lists count constructed and saved products, not validated products',polar_validated=[],number_validated=[],
  density_definition='first eight q_j/(4*pi*(R_j/a)^2), in one-based site order; units a^3 n; no Cartesian/central interpretation',
  number_definition='interleaved adjacent spin rows; q=sum(abs(Z.reshape(2048,2,-1))**2*f,axis=(1,2)); weights once',combined_saved=False,combined_validated=False,
  polar_completed=[],number_completed=[],missing_polar=list(range(1,6)),missing_number=list(range(1,6)),empty_channels=list(range(6,11)),channels={},arrays=[])
 def clean(x):
  if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [clean(v) for v in x]
  if np is not None and isinstance(x,np.ndarray):return clean(x.tolist())
  if np is not None and isinstance(x,np.generic):return clean(x.item())
  if isinstance(x,float) and not math.isfinite(x):return None
  return x
 def save():
  mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
  try:
   normal=['context.npz']+['polar_k%02d.npz'%k for k in range(1,6)]+['number_k%02d.npz'%k for k in range(1,6)]+['combined.npz']
   saved={Path(row['path']).name for row in report['arrays']}
   report['missing_products']=[name for name in normal if name not in saved]
   report['missing_validations']=['polar_k%02d'%k for k in range(1,6) if k not in report['polar_validated']]+['number_k%02d'%k for k in range(1,6) if k not in report['number_validated']]+([] if report['combined_validated'] else ['combined'])
   report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=digest(Path(__file__)))
   tmp=target.with_suffix('.json.tmp')
   with tmp.open('x') as stream:
    stream.write(json.dumps(clean(report),indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
   tmp.replace(target)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 def arrays(name,**values):
  p=ROOT/(name+'.npz');tmp=p.with_suffix('.npz.tmp')
  if p.exists() or tmp.exists():raise FileExistsError(str(p))
  with tmp.open('xb') as stream:np.savez(stream,**values);stream.flush();os.fsync(stream.fileno())
  tmp.replace(p)
  row=dict(path=str(p),size=p.stat().st_size,sha256=digest(p),members={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in values.items()})
  report['arrays'].append(row);return row
 def require(label,value,limit):
  if not math.isfinite(float(value)) or float(value)>=limit:raise ValueError(label+' exceeds fixed control')
 def half_radius(R,F):
  i=int(np.searchsorted(F,.5,side='left'))
  if not 0<=i<len(R):raise ValueError('Half-number bracket absent')
  if i==0:
   if not F[0]>=.5:raise ValueError('First-site bracket does not contain target')
   value=float(R[0]);indices=np.array([0],dtype=np.int64)
  else:
   denominator=float(F[i]-F[i-1])
   if not (math.isfinite(denominator) and denominator>0 and F[i-1]<.5<=F[i]):raise ValueError('Invalid half-number interpolation bracket')
   value=float(R[i-1]+(R[i]-R[i-1])*(.5-F[i-1])/denominator)
   indices=np.array([i-1,i],dtype=np.int64)
  return value,dict(indices=indices,R=R[indices],F=F[indices],target=.5)

 try:
  for p,h in PINS.items():
   if digest(p)!=h:raise ValueError('Changed own dependency '+str(p))
  import numpy as numpy
  np=numpy
  import scipy
  from scipy.linalg import eigh
  sys.path.insert(0,str(BASE))
  from core_well_input_v1 import StoredNpz,file_identity
  spec=importlib.util.spec_from_file_location('own_profile_primitives',BASE/'core_well_profiles_v1.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine);engine.np=np
  def tracked_eigh(*args,**kwargs):
   eigenvalues,eigenvectors=eigh(*args,**kwargs)
   finite_values=bool(np.isfinite(eigenvalues).all());positive=bool(finite_values and len(eigenvalues)>0 and eigenvalues[0]>0)
   if not positive:
    k=report['current_channel'];report['channels'][k]=dict(failed_gram_control=dict(all_eigenvalues_finite=finite_values,strictly_positive=positive,minimum=eigenvalues[0] if len(eigenvalues) else None))
    arrays('failed_gram_k%02d'%k,joint_gram_eigenvalues=eigenvalues,kappa=np.array(k,dtype=np.int64));save()
   return eigenvalues,eigenvectors
  engine.eigh=tracked_eigh
  report['implementation']=dict(actual_nice=os.getpriority(os.PRIO_PROCESS,0),python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__)
  with StoredNpz(SOURCE) as source:
   before=os.fstat(source.stream.fileno());hasher=hashlib.sha256();source.stream.seek(0)
   for part in iter(lambda:source.stream.read(2**20),b''):hasher.update(part)
   after=os.fstat(source.stream.fileno());source.verify_identity()
   observed=hasher.hexdigest();report['source'].update(actual_sha256=observed,descriptor_before=dict(size=before.st_size,device=before.st_dev,inode=before.st_ino,mtime_ns=before.st_mtime_ns,ctime_ns=before.st_ctime_ns),descriptor_after=dict(size=after.st_size,device=after.st_dev,inode=after.st_ino,mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns));save()
   if before.st_size!=2685802106 or observed!=SOURCE_SHA:raise ValueError('Changed fixed archive')
   if (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev,after.st_ctime_ns):raise ValueError('Archive changed during descriptor-bound hash')
   expected_keys={'R','T','rs','kappas','weights','dtype','N_applied','v_applied'}|{prefix+str(k) for k in range(1,11) for prefix in ('sea_','dust_','f_','R2_','e_','mask_')}
   if set(source.entries)!={key+'.npy' for key in expected_keys}:raise ValueError('Unexpected exact archive member set')
   for key in ('T','rs'):
    header=source.member_header(key)
    if header['shape']!=() or header['dtype']!=np.dtype('float64'):raise ValueError('Stored clock dtype/shape mismatch')
   header=source.member_header('dtype')
   if header['shape']!=() or header['dtype'].kind!='U':raise ValueError('Dtype declaration shape/type mismatch')
   dtype_declaration=str(source.array('dtype').item());report['source']['dtype_declaration']=dtype_declaration
   if dtype_declaration!='complex128':raise ValueError('Archive dtype declaration mismatch')
   if float(source.array('T'))!=832 or float(source.array('rs'))!=64:raise ValueError('Clock/scale mismatch')
   R=np.asarray(source.array('R')).copy();weights=np.asarray(source.array('weights')).copy();kappas=np.asarray(source.array('kappas')).copy()
   if R.dtype!=np.dtype('float64') or kappas.dtype!=np.dtype('int64') or weights.dtype!=np.dtype('float64') or not np.array_equal(R,np.arange(1,2049)) or not np.array_equal(kappas,np.arange(1,11)) or not np.array_equal(weights,16*kappas):raise ValueError('Grid/channel/multiplicity mismatch')
   N,v=[np.asarray(source.array(k)).copy() for k in ('N_applied','v_applied')]
   if any(x.shape!=(2048,) or x.dtype!=np.dtype('float64') or not np.isfinite(x).all() for x in (N,v)):raise ValueError('Applied field metadata')
   fs={}
   for k,d in enumerate(COUNTS,1):
    f=np.asarray(source.array('f_'+str(k))).copy();fs[k]=f
    if f.dtype!=np.dtype('float64') or f.shape!=(d,) or not np.array_equal(f,np.ones(d)):raise ValueError('Occupations/counts changed')
    for key,shape in [('sea_'+str(k),(4096,2048)),('dust_'+str(k),(4096,d))]:
     h=source.member_header(key)
     if h['shape']!=shape or h['dtype']!=np.dtype('complex128'):raise ValueError('Column header mismatch '+key)
   report['source'].update(identity=file_identity(SOURCE),keys=source.keys,clock_T=832,rs=64,input_clock_dtype='float64',saved_clock_dtype='int64',weights=weights,kappas=kappas,dust_counts=COUNTS)
   mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
   try:arrays('context',R=R,N_applied=N,v_applied=v,weights=weights,kappas=kappas);save()
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
   number_profiles={}
   for k in range(1,6):
    f=fs[k];report.update(phase='joint_polar',current_channel=k);save();mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     Z,B,eigenvalues,info=engine.joint_dust(source,k,f)
     row=arrays('polar_k%02d'%k,dust=Z,joint_inverse_root_columns=B,joint_gram_eigenvalues=eigenvalues,f=f,kappa=np.array(k,dtype=np.int64),weight=np.array(weights[k-1]),T=np.array(832,dtype=np.int64),rs=np.array(64,dtype=np.int64),gram_min=np.array(info['gram_min']),gram_max=np.array(info['gram_max']),hermitian_defect=np.array(info['hermitian_defect']),dust_orthogonality_max=np.array(info['dust_orthogonality_max']))
     report['channels'][k]=dict(polar=info,polar_archive=row['path']);report['polar_completed'].append(k);report['missing_polar'].remove(k);save()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    require('dust Gram error',info['dust_orthogonality_max'],1e-10)
    report['polar_validated'].append(k);report['phase']='number_readout';save()
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     q=np.sum(np.abs(Z.reshape(2048,2,-1))**2*f,axis=(1,2))
     total=float(q.sum());residual=total-COUNTS[k-1];minimum=float(q.min());finite=bool(np.isfinite(q).all())
     controls=dict(total_number=total,count_residual=residual,count_error=abs(residual),site_min=minimum,all_finite=finite)
     arrays('number_k%02d'%k,q=q,f=f,kappa=np.array(k,dtype=np.int64),weight=np.array(weights[k-1]),number=np.array(total),trace_error=np.array(residual),site_min=np.array(minimum))
     report['channels'][k]['number_controls']=controls;report['number_completed'].append(k);report['missing_number'].remove(k);number_profiles[k]=q;save()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    require('unweighted count error',abs(residual),1e-9)
    if not finite or minimum<0:raise ValueError('Invalid radial number profile')
    source.verify_identity();report['number_validated'].append(k);save();del Z,B,eigenvalues,q;gc.collect()
   report.update(phase='combined_number_readout',current_channel=None);save()
   mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
   combined={};combined_error=None
   try:
    combined.update(R=R,target_fraction=np.array(.5))
    combined['q']=np.zeros(2048)
    weighted_channels=[]
    for k in range(1,6):
     combined['q']+=weights[k-1]*number_profiles[k];weighted_channels.append(k)
     report['combined_channels']=weighted_channels.copy()
    q=combined['q'];report['combined_stage']='weighted_profile'
    combined['radial_density_first8']=q[:8]/(4*np.pi*R[:8]**2)
    report['radial_density_first8']=combined['radial_density_first8'].copy();report['combined_stage']='density_before_normalization';save()
    if not np.isfinite(combined['radial_density_first8']).all() or np.any(combined['radial_density_first8']<0):raise ValueError('Invalid first-eight radial densities')
    report['density_normalization_residual_first8']=4*np.pi*R[:8]**2*combined['radial_density_first8']-q[:8]
    combined['cumulative_count']=q.cumsum();report['combined_stage']='cumulative_count'
    combined['weighted_count']=np.array(q.sum());total=float(combined['weighted_count'])
    controls=dict(weighted_count=total,weighted_trace_error=abs(total-640),site_min=float(q.min()),all_finite=bool(np.isfinite(q).all()),cumulative_all_finite=bool(np.isfinite(combined['cumulative_count']).all()),cumulative_min=float(combined['cumulative_count'].min()),cumulative_max=float(combined['cumulative_count'].max()),cumulative_monotone=bool(np.all(np.diff(combined['cumulative_count'])>=0)),cumulative_endpoint_minus_total=float(combined['cumulative_count'][-1]-total))
    report['controls']=controls
    if not math.isfinite(total) or total<=0:raise ValueError('Invalid total for cumulative normalization')
    combined['cumulative_fraction']=combined['cumulative_count']/total;report['combined_stage']='normalized_cumulative'
    if not controls['all_finite'] or controls['site_min']<0 or not controls['cumulative_all_finite'] or not controls['cumulative_monotone'] or not np.isfinite(combined['cumulative_fraction']).all():raise ValueError('Weighted count/cumulative domain')
    r50,bracket=half_radius(R,combined['cumulative_fraction'])
    combined.update(R50=np.array(r50),quantile50_indices=bracket['indices'],quantile50_R=bracket['R'],quantile50_F=bracket['F'])
    report['combined_stage']='half_number_radius';report['quantile']=dict(R50=r50,bracket=bracket)
    require('weighted count error',controls['weighted_trace_error'],1e-7)
    if not math.isfinite(r50) or not float(bracket['R'][0])<=r50<=float(bracket['R'][-1]):raise ValueError('Interpolated radius outside bracket')
    if controls['cumulative_min']<0 or controls['cumulative_max']>640+1e-7:raise ValueError('Cumulative count outside domain')
    source.verify_identity();report['combined_validated']=True
   except Exception as error:
    combined_error=error;report['combined_error']=type(error).__name__+': '+str(error);raise
   finally:
    try:
     if combined:
      arrays('combined',**combined);report['combined_saved']=True;save()
    except Exception as error:
     if combined_error is None:raise
     report['combined_preservation_error']=type(error).__name__+': '+str(error);save()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
   report['status']='completed_rung4_T13_number_density_only';report['phase']='complete';save()
 except Exception as error:report.update(error=type(error).__name__+': '+str(error),status='incomplete')
 finally:
  signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS);signal.signal(signal.SIGALRM,signal.SIG_IGN);signal.signal(signal.SIGXCPU,signal.SIG_IGN);signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(720,720));save()
 print(json.dumps(dict(status=report['status'],polar_completed=report['polar_completed'],number_completed=report['number_completed'])))
 if report['status']!='completed_rung4_T13_number_density_only':raise SystemExit(1)

if __name__=='__main__':main()
