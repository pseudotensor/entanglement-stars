"""One fixed stored rung slice: missing joint-dust profiles and window spectra."""
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

ROOT=Path('<internal>/rung8_T16_v1')
BASE=ROOT.parent/'profiles_v1'
SOURCE=Path('<internal>/orbitals_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_T0016.00.npz')
SOURCE_SHA='5ed863eb348d1d40805d662c6d92017a18a9861ebbef96ad1f2d0ec256d8674b'
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
 target=ROOT/'rung8_T16_result.json'
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
 def deadline(signum,frame):raise TimeoutError('Fixed one-slice allocation deadline')
 signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(160)
 start=time.monotonic();np=None
 report=dict(status='incomplete',purpose='Task3 missing width-eight rung dust density, number radii and shell momentum',
  label='v36d width-eight rung, another engine from T=0, T/rs=16, finite stored state',
  scope=dict(original_c27=False,r11=False,history=False,static_three_width_HF=False,native_HF=False,native_cartesian_saturation=False),
  source=dict(path=str(SOURCE),sha256=SOURCE_SHA),dependencies={str(p):h for p,h in PINS.items()},
  declared_units=dict(a=1,mu=.2,g=.25,sigma_over_a=8,tau_over_a=8,clock_over_rs=16),
  resources=dict(available_memory_bytes=available_memory,minimum_available_memory_bytes=8*2**30,threads=4,cpu_affinity=cpus,nice=10,address_space_bytes=4*2**30,cpu_soft=700,cpu_hard=720,wall_internal=160,wall_external=180),
  phase='input_binding',current_channel=None,completion_list_meaning='completed lists count constructed and saved products, not validated products',polar_validated=[],readout_validated=[],
  measure_rows=['q: unweighted canonical dust count per radial site','e: signed bare energy per radial site, a^-1','j: signed two-step source current per radial site, a^-1'],density_units='q/(4piR^2) in a^-3; e,j/(4piR^2) in a^-4',
  polar_completed=[],readout_completed=[],missing_polar=list(range(1,6)),missing_readout=list(range(1,6)),empty_channels=list(range(6,11)),channels={},arrays=[])
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
   report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=digest(Path(__file__)))
   tmp=target.with_suffix('.json.tmp');tmp.write_text(json.dumps(clean(report),indent=2,allow_nan=False)+'\n');tmp.replace(target)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
 def arrays(name,**values):
  p=ROOT/(name+'.npz');tmp=p.with_suffix('.npz.tmp')
  if p.exists() or tmp.exists():raise FileExistsError(str(p))
  with tmp.open('xb') as stream:np.savez(stream,**values)
  tmp.replace(p)
  row=dict(path=str(p),size=p.stat().st_size,sha256=digest(p),members={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in values.items()})
  report['arrays'].append(row);return row
 def require(label,value,limit):
  if not math.isfinite(float(value)) or float(value)>=limit:raise ValueError(label+' exceeds fixed control')
 def quantile(R,q,p):
  cumulative=q.cumsum()/q.sum();i=int(np.searchsorted(cumulative,p,side='left'))
  if not 0<=i<len(R):raise ValueError('Quantile missing bracket')
  if i==0:return float(R[0]),dict(indices=[0],R=[float(R[0])],F=[float(cumulative[0])],target=p)
  value=R[i-1]+(R[i]-R[i-1])*(p-cumulative[i-1])/(cumulative[i]-cumulative[i-1])
  return float(value),dict(indices=[i-1,i],R=R[i-1:i+1],F=cumulative[i-1:i+1],target=p)
 try:
  for p,h in PINS.items():
   if digest(p)!=h:raise ValueError('Changed own dependency '+str(p))
  before=SOURCE.stat()
  if before.st_size!=2685802106 or digest(SOURCE)!=SOURCE_SHA:raise ValueError('Changed fixed archive')
  after=SOURCE.stat()
  if (before.st_size,before.st_mtime_ns,before.st_ino,before.st_dev)!=(after.st_size,after.st_mtime_ns,after.st_ino,after.st_dev):raise ValueError('Archive changed during hash')
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
    arrays('failed_gram_k%02d'%k,joint_gram_eigenvalues=eigenvalues,kappa=np.array(k));save()
   return eigenvalues,eigenvectors
  engine.eigh=tracked_eigh
  report['implementation']=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__)
  with StoredNpz(SOURCE) as source:
   if file_identity(SOURCE)!=dict(path=str(SOURCE),size=after.st_size,mtime_ns=after.st_mtime_ns,device=after.st_dev,inode=after.st_ino):raise ValueError('Archive pathname changed after hash')
   if len(source.keys)!=68:raise ValueError('Unexpected archive schema')
   dtype_declaration=str(source.array('dtype').item());report['source']['dtype_declaration']=dtype_declaration
   if dtype_declaration!='complex128':raise ValueError('Archive dtype declaration mismatch')
   if float(source.array('T'))!=1024 or float(source.array('rs'))!=64:raise ValueError('Clock/scale mismatch')
   R=np.asarray(source.array('R')).copy();weights=np.asarray(source.array('weights')).copy();kappas=np.asarray(source.array('kappas')).copy()
   if not np.array_equal(R,np.arange(1,2049)) or not np.array_equal(kappas,np.arange(1,11)) or not np.array_equal(weights,16*kappas):raise ValueError('Grid/channel/multiplicity mismatch')
   N,v=[np.asarray(source.array(k)).copy() for k in ('N_applied','v_applied')]
   if any(x.shape!=(2048,) or not np.isfinite(x).all() for x in (N,v)):raise ValueError('Applied field metadata')
   fs={}
   for k,d in enumerate(COUNTS,1):
    f=np.asarray(source.array('f_'+str(k))).copy();fs[k]=f
    if f.shape!=(d,) or not np.array_equal(f,np.ones(d)):raise ValueError('Occupations/counts changed')
    for key,shape in [('sea_'+str(k),(4096,2048)),('dust_'+str(k),(4096,d))]:
     h=source.member_header(key)
     if h['shape']!=shape or h['dtype']!=np.dtype('complex128'):raise ValueError('Column header mismatch '+key)
   report['source'].update(identity=file_identity(SOURCE),keys=source.keys,clock_T=1024,rs=64,weights=weights,kappas=kappas,dust_counts=COUNTS)
   mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
   try:arrays('context',R=R,N_applied=N,v_applied=v,weights=weights,kappas=kappas);save()
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
   total=np.zeros((3,2048));spectra=np.zeros((16,4096));channel_info={}
   for k in range(1,6):
    f=fs[k];report.update(phase='joint_polar',current_channel=k);save();mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     Z,B,eigenvalues,info=engine.joint_dust(source,k,f)
     row=arrays('polar_k%02d'%k,dust=Z,joint_inverse_root_columns=B,joint_gram_eigenvalues=eigenvalues,f=f,kappa=np.array(k),weight=np.array(weights[k-1]),T=np.array(1024),rs=np.array(64),gram_min=np.array(info['gram_min']),gram_max=np.array(info['gram_max']),hermitian_defect=np.array(info['hermitian_defect']),dust_orthogonality_max=np.array(info['dust_orthogonality_max']))
     report['channels'][k]=dict(polar=info,polar_archive=row['path']);report['polar_completed'].append(k);report['missing_polar'].remove(k);save()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    require('dust Gram error',info['dust_orthogonality_max'],1e-10)
    report['polar_validated'].append(k);report['phase']='channel_readout';save()
    mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
    try:
     site=engine.measures(Z,f,R,k);spectrum,folded,shell_count,shell_high=engine.momentum(Z,f)
     lowmask=np.abs(folded)<=512;highmask=~lowmask;low=spectrum[:,lowmask].sum(axis=1)
     empty_shell=shell_count==0;high_share=np.full(16,np.nan);low_share=np.full(16,np.nan)
     np.divide(shell_high,shell_count,out=high_share,where=~empty_shell);np.divide(low,shell_count,out=low_share,where=~empty_shell)
     parseval=np.max(np.abs(shell_count-site[0].reshape(16,128).sum(axis=1)));trace_error=abs(site[0].sum()-COUNTS[k-1]);closure=np.max(np.abs(shell_count-shell_high-low))
     global_energy=np.sum(Z.conj()*engine.bare(Z,R,k)*f);global_current=np.sum(Z.conj()*engine.flow(Z)*f)
     energy_gap=abs(site[1].sum()-global_energy.real);current_gap=abs(site[2].sum()-global_current.real)
     controls=dict(trace=site[0].sum(),trace_error=trace_error,parseval_error=parseval,complement_error=closure,site_min=site[0].min(),high_min=shell_high.min(),high_excess_max=(shell_high-shell_count).max(),global_bare_energy_real=global_energy.real,global_bare_energy_imaginary=global_energy.imag,global_source_current_real=global_current.real,global_source_current_imaginary=global_current.imag,energy_trace_closure=energy_gap,current_trace_closure=current_gap)
     arrays('readout_k%02d'%k,measures=site,shell_fft_weights=spectrum,folded_index=folded,high_mask=highmask,low_mask=lowmask,shell_number=shell_count,shell_high_number=shell_high,shell_low_number=low,shell_high_share=high_share,shell_low_share=low_share,empty_shell=empty_shell,global_bare_energy=np.array(global_energy),global_source_current=np.array(global_current),f=f,kappa=np.array(k),weight=np.array(weights[k-1]))
     report['channels'][k].update(controls=controls,shell_high_share=high_share,shell_low_share=low_share,empty_shell=empty_shell);report['readout_completed'].append(k);report['missing_readout'].remove(k)
     total+=weights[k-1]*site;spectra+=weights[k-1]*spectrum;channel_info[k]=controls;save()
    finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    require('unweighted count error',trace_error,1e-9);require('shell Parseval',parseval,1e-9);require('complement closure',closure,1e-9)
    if energy_gap>1e-9*max(1,abs(global_energy.real)) or current_gap>1e-9*max(1,abs(global_current.real)):raise ValueError('Bare-vertex trace closure')
    if not np.isfinite(site).all() or not np.isfinite(spectrum).all() or site[0].min()<0 or shell_high.min()<0 or (shell_high-shell_count).max()>1e-9:raise ValueError('Invalid new dust measure')
    source.verify_identity();report['readout_validated'].append(k);save();del Z,B,eigenvalues,site,spectrum;gc.collect()
   report.update(phase='combined_readout',current_channel=None);save()
   q,e,j=total;cumulative=q.cumsum();r50,b50=quantile(R,q,.5);r90,b90=quantile(R,q,.9)
   density=q/(4*np.pi*R**2);shell_count=spectra.sum(axis=1);shell_high=spectra[:,highmask].sum(axis=1);shell_low=spectra[:,lowmask].sum(axis=1)
   share=np.full(16,np.nan);np.divide(shell_high,shell_count,out=share,where=shell_count>0)
   complement=np.full(16,np.nan);np.divide(shell_low,shell_count,out=complement,where=shell_count>0)
   controls=dict(weighted_count=q.sum(),weighted_trace_error=abs(q.sum()-640),weighted_parseval_error=np.max(np.abs(shell_count-q.reshape(16,128).sum(axis=1))),weighted_complement_error=np.max(np.abs(shell_high+shell_low-shell_count)),cumulative_min=cumulative.min(),cumulative_max=cumulative.max(),quantile_order=bool(r50<=r90),site_min=q.min())
   maxima={}
   for label,selected in [('full',np.ones(2048,dtype=bool)),('R_le_128',R<=128),('R_le_R50',R<=r50)]:
    indices=np.flatnonzero(selected);i=int(indices[np.argmax(density[selected])]);maxima[label]=dict(index=i,site=i+1,R=R[i],density=density[i])
   mask=signal.pthread_sigmask(signal.SIG_BLOCK,SIGNALS)
   try:
    arrays('combined',R=R,N_applied=N,v_applied=v,weights=weights,kappas=kappas,measures=total,q=q,e=e,j=j,cumulative_count=cumulative,density=density,energy_density=e/(4*np.pi*R**2),current_density=j/(4*np.pi*R**2),density_over_pi24=density/(np.pi/24),density_over_four=density/4,density_over_two=density/2,count_over_chain_capacity=q/1760,R50=np.array(r50),R90=np.array(r90),shell_fft_weights=spectra,folded_index=folded,high_mask=highmask,low_mask=lowmask,shell_number=shell_count,shell_high_number=shell_high,shell_low_number=shell_low,shell_high_share=share,shell_low_share=complement,quantile50_indices=np.asarray(b50['indices']),quantile50_R=np.asarray(b50['R']),quantile50_F=np.asarray(b50['F']),quantile90_indices=np.asarray(b90['indices']),quantile90_R=np.asarray(b90['R']),quantile90_F=np.asarray(b90['F']))
    report.update(controls=controls,quantiles=dict(R50=r50,R90=r90,bracket50=b50,bracket90=b90),density_maxima=maxima,inner_eight_density=density[:8],shell_high_share=share,shell_low_share=complement);save()
   finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
   require('weighted trace',controls['weighted_trace_error'],1e-7);require('weighted shell closure',controls['weighted_parseval_error'],1e-7);require('weighted complements',controls['weighted_complement_error'],1e-7)
   if not controls['quantile_order'] or q.min()<0 or cumulative.min()<0 or cumulative.max()>640+1e-7:raise ValueError('Count/quantile domain')
   source.verify_identity()
   report.update(status='completed_rung8_T16_dust_only',phase='completed')
 except Exception as error:report.update(error=type(error).__name__+': '+str(error),status='incomplete')
 finally:
  signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(720,720));save()
 print(json.dumps(dict(status=report['status'],polar_completed=report['polar_completed'],readout_completed=report['readout_completed'])))
 if report['status']!='completed_rung8_T16_dust_only':raise SystemExit(1)

if __name__=='__main__':main()
