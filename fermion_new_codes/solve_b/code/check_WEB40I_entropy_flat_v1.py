#!/usr/bin/env python3
"""The single X3 ET1/ET17--21 flat gate, on the authorized CPU box.

One L32 identity fixture, one L8192 Jacobi negative subspace, four cuts.
Raw arrays stay on /data. Floating residuals are explicitly uncertified.
"""
from __future__ import annotations
import time
START = time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import signal
import socket
STARTUP_AS_LIMIT=resource.getrlimit(resource.RLIMIT_AS)

os.environ['CUDA_VISIBLE_DEVICES']=''
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='4'
import numpy as np
import scipy
from scipy.linalg import eigh, eigvalsh, eigh_tridiagonal

LARGE_L=8192
CUTS=(128,1024,1536,3072)
TARGETS=(2.0544,2.7464,2.8815,3.1109)
WALL_LIMIT=1200.0
SAVE_RESERVE=60.0
NO_NEW_DENSE_BELOW=120.0
AS_LIMIT=4*1024**3
BLOCK=128
RESULT={}
ROOT=None
STOP_REQUESTED=False


class BudgetStop(Exception):
    pass


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def check_floors():
    mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                 if x.startswith('MemAvailable:')))*1024
    if mem<64*1024**3 or shutil.disk_usage('/').free<=100_000_000_000:
        raise BudgetStop('owner memory/root-disk floor')


def guard(dense=False):
    remaining=WALL_LIMIT-(time.monotonic()-START)
    if STOP_REQUESTED or remaining < (NO_NEW_DENSE_BELOW if dense else SAVE_RESERVE):
        raise BudgetStop('reserve reached; completed products preserved')
    check_floors()


def request_stop(*_args):
    global STOP_REQUESTED
    STOP_REQUESTED=True


def save_result():
    u=resource.getrusage(resource.RUSAGE_SELF)
    RESULT['resources']={'phase_wall_seconds':time.monotonic()-START,
        'process_user_cpu_seconds':u.ru_utime,'process_system_cpu_seconds':u.ru_stime,
        'peak_rss_kib':u.ru_maxrss}
    tmp=ROOT/'flat_gate.json.tmp'
    tmp.write_text(json.dumps(RESULT,indent=2,allow_nan=False)+'\n')
    tmp.replace(ROOT/'flat_gate.json')


def index_raw():
    lines=['path\tsize_bytes\tsha256_or_file_count\tbox_location\treason\thistory']
    for r in RESULT['raw_files']:
        lines.append('\t'.join([r['relative_path'],str(r['size_bytes']),r['sha256'],
            r['box_path'],'raw array retained on box; never copied to Git','single fixed flat gate']))
    (ROOT/'LARGE_FILES.tsv').write_text('\n'.join(lines)+'\n')


def save_array(name,array):
    # The reserve prevents starting work, not preserving a completed result.
    check_floors()
    path=ROOT/'raw'/name
    if path.exists():
        raise FileExistsError('refuse raw overwrite: '+str(path))
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        np.save(f,array,allow_pickle=False)
    tmp.replace(path)
    entry={'relative_path':'raw/'+name,'box_path':str(path),'size_bytes':path.stat().st_size,
        'sha256':sha(path),'shape':list(array.shape),'dtype':str(array.dtype)}
    RESULT['raw_files'].append(entry)
    index_raw();save_result()
    return entry


def entropy(values,factor=2):
    raw=np.asarray(values)
    clipped=np.clip(raw,0,1)
    inside=(clipped>0)&(clipped<1)
    x=clipped[inside]
    value=float(factor*np.sum(-x*np.log(x)-(1-x)*np.log1p(-x),dtype=np.float64))
    return {'entropy_nats':value,'factor':factor,'unclipped_min':float(np.min(raw)),
        'unclipped_max':float(np.max(raw)),'clipped_count':int(np.count_nonzero(raw!=clipped)),
        'clipping_l2':float(np.linalg.norm(raw-clipped)),
        'epsilon_log_certified':None,'certified':False}


def norm_squared(a):
    if a.ndim==2 and not np.iscomplexobj(a):
        return float(np.einsum('ij,ij->',a,a,optimize=False))
    return float(np.vdot(a,a).real)


def orthogonality(columns):
    size=columns.shape[1];total=0.0
    trans=columns.conj().T if np.iscomplexobj(columns) else columns.T
    for i in range(0,size,BLOCK):
        guard(dense=True)
        stop=min(size,i+BLOCK)
        g=trans[i:stop,:]@columns
        g[np.arange(stop-i),np.arange(i,stop)]-=1
        total+=norm_squared(g)
    return math.sqrt(total)


def jacobi_residual(diagonal,off,values,columns):
    total=0.0
    for i in range(0,len(values),BLOCK):
        guard(dense=True)
        v=columns[:,i:i+BLOCK]
        r=diagonal[:,None]*v-v*values[None,i:i+BLOCK]
        r[:-1]+=off[:,None]*v[1:]
        r[1:]+=off[:,None]*v[:-1]
        total+=norm_squared(r)
    return math.sqrt(total)


def reconstruction_residual(matrix,values,vectors):
    total=0.0;size=len(values)
    trans=vectors.conj().T if np.iscomplexobj(vectors) else vectors.T
    for i in range(0,size,BLOCK):
        guard(dense=True)
        stop=min(size,i+BLOCK)
        r=(vectors[i:stop,:]*values[None,:])@trans
        r-=matrix[i:stop,:]
        total+=norm_squared(r)
    return math.sqrt(total)


def fixture():
    length=32;parity=np.where(np.arange(length)%2==0,1.0,-1.0)
    diagonal=parity/(np.arange(length)+1);off=np.full(length-1,0.5)
    sigma_y=np.array([[0,-1j],[1j,0]])
    sigma_x=np.array([[0,1],[1,0]],dtype=complex)
    h=np.zeros((2*length,2*length),dtype=complex)
    basis=np.zeros_like(h);g=1.0
    for j in range(length):
        h[2*j:2*j+2,2*j:2*j+2]=sigma_y/(j+1)
        u=g*np.array([1,1j*parity[j]])/math.sqrt(2)
        basis[2*j:2*j+2,j]=u
        basis[2*j:2*j+2,length+j]=u.conjugate()
        if j+1<length:
            h[2*j:2*j+2,2*j+2:2*j+4]=-0.5j*sigma_x
            h[2*j+2:2*j+4,2*j:2*j+2]=0.5j*sigma_x
            g=-parity[j]*g
    jac=np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
    expected=np.zeros_like(h);expected[:length,:length]=jac;expected[length:,length:]=-jac
    guard(dense=True)
    transformed=basis.conj().T@h@basis
    error=float(np.max(np.abs(transformed-expected)))
    w,v=eigh(h,driver='evr',check_finite=True)
    occupied=v[:,w<0];projector=occupied@occupied.conj().T
    native_cut_values=eigvalsh(projector[:32,:32],driver='evr')
    native=entropy(native_cut_values,factor=1)
    jw,jv=eigh_tridiagonal(diagonal,off,select='i',select_range=(0,15),lapack_driver='stemr')
    q=jv[:16]@jv[:16].T
    reduced_cut_values=eigvalsh(q,driver='evr')
    reduced=entropy(reduced_cut_values)
    difference=native['entropy_nats']-reduced['entropy_nats']
    for name,a in [('fixture_native.npy',h),('fixture_sector_basis.npy',basis),
                   ('fixture_jacobi.npy',jac),('fixture_native_eigenvalues.npy',w),
                   ('fixture_native_eigenvectors.npy',v),('fixture_native_cut_covariance.npy',projector[:32,:32]),
                   ('fixture_native_cut_eigenvalues.npy',native_cut_values),
                   ('fixture_jacobi_eigenvalues.npy',jw),('fixture_jacobi_eigenvectors.npy',jv),
                   ('fixture_jacobi_cut_covariance.npy',q),('fixture_jacobi_cut_eigenvalues.npy',reduced_cut_values)]:
        save_array(name,a)
    return {'L':length,'cut_sites':16,'operator_entrywise_error':error,
        'native_entropy':native,'reduced_entropy':reduced,'entropy_difference':difference,
        'native_negative_count':int(np.count_nonzero(w<0)),'jacobi_negative_count':len(jw),
        'passed':error<1e-13 and abs(difference)<1e-10}


def cut_stage(q,target,values,columns,phases,parent_bound):
    entry={'cut_sites':q,'cut_over_rs':q/64,'target':target,'state':'forming'}
    RESULT['cuts'].append(entry);save_result()
    view=columns[:q,:]
    guard(dense=True)
    covariance=view@view.T
    covariance=(covariance+covariance.T)/2
    save_array(f'cut{q}_covariance.npy',covariance)
    entry['state']='cut eigensolve';save_result();guard(dense=True)
    ew,ev=eigh(covariance,driver='evr',check_finite=True,overwrite_a=False)
    save_array(f'cut{q}_eigenvalues.npy',ew);save_array(f'cut{q}_eigenvectors.npy',ev)
    entry['raw_entropy']=entropy(ew)
    entry['target_difference']=entry['raw_entropy']['entropy_nats']-target
    entry['state']='cut residuals';save_result()
    zeta=orthogonality(ev);recon=reconstruction_residual(covariance,ew,ev)
    eig_error=None if zeta>=1 else recon+float(np.max(np.abs(ew)))*(
        (1+math.sqrt(1+zeta))*zeta/(1+math.sqrt(1-zeta)))
    u=2**-53;n=columns.shape[1]
    gamma_n=n*u/(1-n*u)
    formation_model=gamma_n*norm_squared(view)
    eq=None if parent_bound is None or eig_error is None else parent_bound+formation_model+eig_error
    modulus=None
    if eq is not None:
        x=min(eq/math.sqrt(q),0.5)
        h=0.0 if x==0 else -x*math.log(x)-(1-x)*math.log1p(-x)
        modulus=2*q*h
    entry['error_ingredients']={'orthogonality_frobenius_observed':zeta,
        'reconstruction_frobenius_observed':recon,'eig_error_observed_formula':eig_error,
        'formation_roundoff_model':formation_model,'e_q_uncertified_formula':eq,
        'entropy_allowance_without_log_uncertified':modulus,'epsilon_log_certified':None,
        'residual_norm_rounding_enclosure':None,'certified_entropy_interval':None,
        'rounding_cell_half_width':5e-5,'rounding_cell_resolved_certified':False}
    del ev
    entry['state']='held phase covariance';save_result()
    phased=np.zeros((q,q),dtype=complex)
    for i in range(0,len(values),BLOCK):
        guard(dense=True)
        z=view[:,i:i+BLOCK]*phases[None,i:i+BLOCK]
        phased+=z@z.conj().T
    phased=(phased+phased.conj().T)/2
    save_array(f'cut{q}_phase_covariance.npy',phased)
    delta=phased-covariance
    entry['held_covariance_difference_frobenius']=float(np.linalg.norm(delta))
    entry['held_covariance_difference_max']=float(np.max(np.abs(delta)))
    save_array(f'cut{q}_phase_covariance_difference.npy',delta)
    del delta
    entry['state']='held phase eigensolve';save_result();guard(dense=True)
    pw,pv=eigh(phased,driver='evr',check_finite=True,overwrite_a=False)
    save_array(f'cut{q}_phase_eigenvalues.npy',pw);save_array(f'cut{q}_phase_eigenvectors.npy',pv)
    entry['held_phase_entropy']=entropy(pw)
    entry['held_entropy_difference']=entry['held_phase_entropy']['entropy_nats']-entry['raw_entropy']['entropy_nats']
    entry['held_abs_1e-10_check']=abs(entry['held_entropy_difference'])<=1e-10
    entry['state']='complete';save_result()


def main():
    global ROOT,RESULT
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    ROOT=args.directory
    if (ROOT/'flat_gate.json').exists() or (ROOT/'raw').exists():
        raise FileExistsError('saved gate exists; no automatic repeat')
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'raw').mkdir()
    resource.setrlimit(resource.RLIMIT_AS,(AS_LIMIT,AS_LIMIT))
    resource.setrlimit(resource.RLIMIT_CPU,(4800,4800))
    signal.signal(signal.SIGALRM,request_stop)
    signal.alarm(max(1,math.ceil(WALL_LIMIT-SAVE_RESERVE-(time.monotonic()-START))))
    RESULT={'task':'X3 native flat vacuum and held-phase gate','source_sha256':sha(Path(__file__)),
        'host':socket.gethostname(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'nice':os.getpriority(os.PRIO_PROCESS,0),'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'threads':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'address_space_limit_bytes':AS_LIMIT,
        'startup_address_space_limits_observed':list(STARTUP_AS_LIMIT),
        'whole_group_wall_limit_seconds':1200,'cpu_limit_seconds':4800,'save_reserve_seconds':60,
        'no_new_dense_below_remaining_seconds':120,'L':8192,'negative_count_required':4096,
        'cut_sites':list(CUTS),'time_over_a':64,'copy_weight_applied':False,
        'scope':'One exact native flat preparation; observed arithmetic/error estimates, no certified decimals or production propagator/tail clearance.',
        'complete':False,'reason':'running','stage':'fixture','raw_files':[],'cuts':[]}
    save_result()
    try:
        RESULT['fixture']=fixture();save_result()
        if not RESULT['fixture']['passed']:
            raise ValueError('L32 operator/entropy identity fixture failed; large solve not started')
        j=np.arange(LARGE_L)
        diagonal=np.where(j%2==0,1.0,-1.0)/(j+1)
        off=np.full(LARGE_L-1,0.5)
        save_array('diagonal.npy',diagonal);save_array('off_diagonal.npy',off)
        RESULT['stage']='single negative Jacobi eigensolve';save_result();guard(dense=True)
        values,columns=eigh_tridiagonal(diagonal,off,select='i',select_range=(0,LARGE_L//2-1),
            lapack_driver='stemr',check_finite=True)
        if columns.shape!=(8192,4096) or not np.all(np.isfinite(columns)) or not np.all(np.isfinite(values)):
            raise ValueError('invalid negative Ritz output')
        if not np.all(values<0) or not np.all(np.diff(values)>=0):
            raise ValueError('wrong negative signs/order')
        # Release a possible full-size LAPACK backing store behind a slice.
        columns=columns.copy(order='F')
        save_array('negative_ritz_values.npy',values);save_array('negative_ritz_columns.npy',columns)
        phases=np.exp(-1j*64*values)
        save_array('held_phase_factors.npy',phases)
        RESULT['stage']='parent residual and orthogonality';save_result()
        eta=orthogonality(columns);rho=jacobi_residual(diagonal,off,values,columns)
        b=float(-values[-1]);bound=None if eta>=1 else eta+math.sqrt(2)*rho/((1/LARGE_L+b)*math.sqrt(1-eta))
        RESULT['parent_error']={'eta_frobenius_observed':eta,'rho_frobenius_observed':rho,'b':b,
            'exact_gap_lower_bound':1/LARGE_L,'e_P_observed_formula':bound,
            'gram_rounding_enclosure':None,'native_diagonal_rounding_enclosure':None,
            'residual_norm_rounding_enclosure':None,'certified':False}
        save_result()
        for q,target in zip(CUTS,TARGETS):
            RESULT['stage']=f'cut {q}';save_result();cut_stage(q,target,values,columns,phases,bound)
        RESULT['complete']=True;RESULT['reason']='one fixed fixture/parent/four cuts completed'
    except Exception as error:
        RESULT['reason']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0);index_raw();save_result()
        print(json.dumps({'complete':RESULT['complete'],'reason':RESULT['reason'],
            'completed_cuts':[x['cut_sites'] for x in RESULT['cuts'] if x['state']=='complete'],
            'resources':RESULT['resources']}))
    raise SystemExit(0 if RESULT['complete'] else 2)


if __name__=='__main__':
    main()
