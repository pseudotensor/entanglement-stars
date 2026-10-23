"""Task5 bounded first phase: independent array roots and raw recorded gates.

No foreign solver/readout, row certificate, reference reconstruction or evolution.
"""
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import resource
import signal
import shutil
import sys
import time
import zipfile

ROOT=Path('<internal>/roots')
INVENTORY=ROOT.parent/'input_inventory'/'residual_input_inventory_v1_1.json'
INVENTORY_SHA='f2b123a6312eeb793c489c58bf8489e6e9bf5026e50694ad8e49f04f6a21acb5'
CURRENT=Path('<internal>/core_well_current_v1.json')
CURRENT_SHA='5798c16fc31fb2fdf640341b15362cbeb9c3e7fe1c7b658c1b96f22e330ef99f'
RAW_T16=Path('<internal>/raw_T16.npz')
RAW_SHA='ee49209a0e9999f825ac5ff3c178a70727db5f7b3ff9b09721c97807e6f3ee52'
np=None


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    s=path.stat()
    return dict(path=str(path),resolved_path=str(path.resolve()),size=s.st_size,
                mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if np is not None and isinstance(x,np.ndarray):return clean(x.tolist())
    if np is not None and isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return dict(nonfinite_literal=repr(x))
    return x


def save(path,data):
    blob=json.dumps(clean(data),indent=2,allow_nan=False)+'\n'
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('x') as stream:stream.write(blob)
    if path.exists():raise FileExistsError(path)
    temporary.rename(path)


class NumericArrayUnpickler(pickle.Unpickler):
    """Only NumPy array construction for historical numeric object arrays."""
    def find_class(self,module,name):
        if module=='numpy' and name in ('ndarray','dtype'):
            return getattr(np,name)
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name in ('_reconstruct','scalar'):
            return getattr(np.core.multiarray,name)
        raise pickle.UnpicklingError('Unapproved array class: '+module+'.'+name)


def nodes_member(archive,key):
    with archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
        elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
        else:raise ValueError('Unsupported NPY version')
        if not dtype.hasobject:
            if dtype.kind not in 'fiu':
                raise ValueError('Non-numeric node dtype')
            raw=stream.read()
            if len(raw)!=math.prod(shape)*dtype.itemsize:
                raise ValueError('Numeric node payload length differs')
            return np.frombuffer(raw,dtype=dtype).reshape(shape,order='F' if order else 'C').astype(float)
        value=NumericArrayUnpickler(stream).load()
        if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype:
            raise ValueError('Node-array header/payload disagreement')
        for x in value.flat:
            if not isinstance(x,(int,float,np.integer,np.floating)):
                raise ValueError('Non-numeric node object')
        return np.asarray(value,dtype=float)


def roots(R,N,v):
    R,N,v=map(lambda x:np.asarray(x,dtype=float),(R,N,v))
    if (R.ndim!=1 or len(R)<2 or N.shape!=R.shape or v.shape!=R.shape
            or not np.isfinite(R).all() or not np.isfinite(N).all() or not np.isfinite(v).all()
            or np.any(np.diff(R)<=0)):
        raise ValueError('Invalid root samples')
    f=N+v;isolated=[];zero_intervals=[];interval_nodes=set()
    i=0
    while i<len(R)-1:
        if f[i]==0 and f[i+1]==0:
            start=i
            while i+1<len(R) and f[i+1]==0:i+=1
            zero_intervals.append(dict(bounds=[R[start],R[i]],indices=[start,i]))
            interval_nodes.update(range(start,i+1))
        else:i+=1
    for i in np.flatnonzero(f==0):
        if int(i) not in interval_nodes:
            isolated.append(dict(radius=R[i],bracket=[R[i],R[i]],indices=[int(i),int(i)],kind='exact_node',
                left_slope=(f[i]-f[i-1])/(R[i]-R[i-1]) if i>0 else None,
                right_slope=(f[i+1]-f[i])/(R[i+1]-R[i]) if i+1<len(R) else None))
    changes=np.flatnonzero((f[:-1]!=0)&(f[1:]!=0)&(np.signbit(f[:-1])!=np.signbit(f[1:])))
    for i in changes:
        width=R[i+1]-R[i];slope=(f[i+1]-f[i])/width
        radius=R[i]-f[i]*width/(f[i+1]-f[i])
        isolated.append(dict(radius=radius,bracket=[R[i],R[i+1]],indices=[int(i),int(i+1)],
            slope=slope,orientation='outward' if slope>0 else 'inward',kind='linear_sign_change'))
    isolated.sort(key=lambda x:x['radius'])
    outer=isolated[-1]['radius'] if isolated else None
    if zero_intervals and (outer is None or max(x['bounds'][1] for x in zero_intervals)>=outer):
        outer=None
    return dict(isolated_roots=isolated,zero_intervals=zero_intervals,outer_isolated_root=outer,
        minimum_lapse=float(N.min()),nonpositive_lapse_count=int((N<=0).sum()),
        positive_shift_count=int((v>0).sum()),domain=[R[0],R[-1]],
        scope='Piecewise-linear saved-array roots; no smooth-cell tangency or error certificate')


def prefix(states):
    last=None
    for i,state in enumerate(states):
        if state is not True:break
        last=i
    return last


def fixtures():
    R=np.linspace(.25,4,31);N=np.ones_like(R)
    assert not roots(R,N,-.2*N)['isolated_roots']
    assert not roots(R,N,np.zeros_like(R))['isolated_roots']
    annulus_R=np.array([1.,2.,3.])
    annulus=roots(annulus_R,np.ones(3),-np.sqrt(2/annulus_R))
    assert len(annulus['isolated_roots'])==1 and annulus['outer_isolated_root']==2.
    assert 'left_slope' in annulus['isolated_roots'][0] and 'right_slope' in annulus['isolated_roots'][0]
    r=roots(np.array([1.,2.]),np.ones(2),np.array([-1.5,-.5]))
    assert r['outer_isolated_root']==1.5
    R=np.linspace(.5,3.5,49);speed=.01*(R-1)*(R-2)*(R-3)
    r=roots(R,np.ones_like(R),speed-1)
    assert [x['radius'] for x in r['isolated_roots']]==[1.,2.,3.]
    r=roots(np.array([1.,2.,3.,4.]),np.ones(4),np.array([-.9,-1.,-1.,-.9]))
    assert len(r['zero_intervals'])==1 and r['zero_intervals'][0]['bounds']==[2.,3.]
    assert r['outer_isolated_root'] is None and not r['isolated_roots']
    r=roots(np.arange(1.,6.),np.ones(5),np.array([-.9,-1.,-1.,-1.,-.9]))
    assert len(r['zero_intervals'])==1 and r['zero_intervals'][0]['bounds']==[2.,4.]
    r=roots(np.array([1.,2.,3.]),np.ones(3),np.array([-1.5,-1.,0.]))
    assert r['isolated_roots'][0]['left_slope']==.5 and r['isolated_roots'][0]['right_slope']==1.
    args=dict(stop_fields=1e-6,stop_E=1e-7)
    gate=raw_gates(dict(converged=True,solve_v32=dict(kept_residual=9e-7,confirming_residual=3e-6,confirming_dE=1e-9)),args)
    assert gate['kept_field_gate_recalculated'] is True and gate['confirming_gate_recalculated'] is False
    for cf,expected in ((1e-6,False),(float('nan'),None),(None,None),(9e-7,True)):
        gate=raw_gates(dict(solve_v32=dict(kept_residual=9e-7,confirming_residual=cf,confirming_dE=1e-9)),args)
        assert gate['confirming_gate_recalculated'] is expected and gate['raw_flag_pass'] is None
    assert raw_gates({},args)['confirming_gate_recalculated'] is None
    assert prefix([True,False,True])==0 and prefix([True,None,True])==0
    assert prefix([False,True]) is None and prefix([True,True])==1
    assert not 1e-3>1e-3 and 1.001e-3>1e-3
    assert 3e-6>1e-6 and 9e-7<1e-6
    # Independent analytic row fixture: no production source is involved.
    r=np.linspace(.1,4,40);g=.25;h=.3;alpha=.02
    N=np.exp(alpha*(r*r-16));v=-h*r*N;Np=2*alpha*r*N
    vp=-h*N-h*r*Np;rho=h*h*(3+4*alpha*r*r)/(8*np.pi*g)
    J=h*alpha*r/(2*np.pi*g)
    RW=v*v+2*r*v*vp-8*np.pi*g*r*r*N*N*rho
    RM=v*Np+4*np.pi*g*r*N*N*J
    assert np.max(np.abs(RW))<2e-14 and np.max(np.abs(RM))<2e-14
    # Opposite-sign source injections leave U while violating both rows.
    dJ=.01*np.ones_like(r);drho=-(v/N)*dJ
    dRW=-8*np.pi*g*r*r*N*N*drho;dRM=4*np.pi*g*r*N*N*dJ
    conversion=dRW/(N*N)-2*r*v*dRM/(N**3)
    assert np.max(np.abs(conversion))<2e-14 and np.max(np.abs(dRW))>1e-4
    return dict(status='pass',root_and_zero_interval=True,prefix_and_threshold=True,
        analytic_row_max=float(max(np.max(np.abs(RW)),np.max(np.abs(RM)))),
        cancellation_max=float(np.max(np.abs(conversion))),
        deferred_controls='Gaussian source/back-map fixtures and derivative stencils belong to the unimplemented row phase')


def raw_gates(record,args):
    solver=record.get('solver') or {};newton=record.get('solve_v32')
    result=dict(converged=record.get('converged'),unconverged=record.get('unconverged'),
        iterations=record.get('iterations'),stop_fields=args.get('stop_fields'),stop_E=args.get('stop_E'),
        solver_residual_W=solver.get('residual_W'),solver_residual_invN=solver.get('residual_invN'))
    result['native_kept_residual']=None
    result['native_kept_residual_source']=None
    result['native_last_iterate_residual']=None
    history=record.get('iteration_history')
    result['raw_last_iteration_tuple']=history[-1] if isinstance(history,list) and history else None
    result['raw_last_iteration_tuple_source']='iteration_history[-1]' if isinstance(history,list) and history else None
    result['raw_iteration_history']=history
    unconverged=record.get('unconverged')
    if isinstance(unconverged,dict) and 'residual' in unconverged:
        result['native_kept_residual']=unconverged['residual']
        result['native_kept_residual_source']='unconverged.residual'
    result['native_last_iterate_unavailable']='Plain iteration_history component norm/scale contract not bound'
    if isinstance(newton,dict):
        keys=('kept_residual','confirming_residual','confirming_dE','meets_stop_rule','kept_from','ends_member')
        result['solve_v32']={k:newton.get(k) for k in keys}
        kept=newton.get('kept_residual')
        result['native_kept_residual']=kept
        result['native_kept_residual_source']='solve_v32.kept_residual'
        cf,ce=newton.get('confirming_residual'),newton.get('confirming_dE')
        tf,te=args.get('stop_fields'),args.get('stop_E')
        if all(isinstance(x,(int,float)) and math.isfinite(x) for x in (cf,ce,tf,te)):
            result['confirming_gate_recalculated']=bool(cf<tf and ce<te)
        else:result['confirming_gate_recalculated']=None
    else:
        result['confirming_gate_recalculated']=None
        result['confirming_unavailable_reason']='No explicit solve_v32 confirming scalar contract'
    kept=result['native_kept_residual'];threshold=args.get('stop_fields')
    result['kept_field_gate_recalculated']=(bool(kept<threshold) if all(isinstance(x,(int,float)) and math.isfinite(x) for x in (kept,threshold)) else None)
    flag=record.get('converged')
    result['raw_flag_pass']=flag if type(flag)==bool else None
    return result


def main():
    global np
    if Path(__file__).resolve().parent!=ROOT:
        raise RuntimeError('Only named box directory')
    if any(ROOT.glob('member_*.json*')) or any(ROOT.glob('summary.json*')):
        raise FileExistsError('Existing root phase requires assessment, never retry')
    if shutil.disk_usage('/').free<100_000_000_000 or shutil.disk_usage('/data').free<2*2**30:
        raise RuntimeError('Required storage reserve unavailable')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[key]='4'
    os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    cpus=sorted(os.sched_getaffinity(0));assigned=cpus[4:8] if len(cpus)>=8 else cpus[:4]
    if len(assigned)!=4:raise RuntimeError('Four CPUs unavailable')
    os.sched_setaffinity(0,assigned)
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(2*2**30,2*2**30))
    resource.setrlimit(resource.RLIMIT_CPU,(2360,2400))
    def deadline(signum,frame):raise TimeoutError('Root phase allocation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(580)
    import numpy as numpy
    np=numpy
    start=time.monotonic();report=dict(status='incomplete',purpose='Task5 partial roots and raw recorded gates',
        producer_sha256=digest(Path(__file__)),members=[],remaining_member_indices=list(range(32)),
        resources=dict(wall_seconds=600,internal_deadline_seconds=580,cpu_seconds=2400,
            address_space_bytes=2*2**30,threads=4,cpu_affinity=assigned,nice=os.getpriority(os.PRIO_PROCESS,0)),
        row_diagnostics='Not evaluated: source width/cell, native source spline, node representation and same-stage contract not bound',
        certified_row_prefix=None)
    try:
        if digest(INVENTORY)!=INVENTORY_SHA or digest(CURRENT)!=CURRENT_SHA or digest(RAW_T16)!=RAW_SHA:
            raise ValueError('Changed owned provenance input')
        inventory=json.loads(INVENTORY.read_text());current=json.loads(CURRENT.read_text())
        report['fixtures']=fixtures();report['inventory_sha256']=INVENTORY_SHA
        with np.load(RAW_T16,allow_pickle=False) as z:
            raw_t16={k:z[k] for k in ('R','N_applied','v_applied')}
        for member_index,item in enumerate(inventory['inputs']):
            path=Path(item['path']);initial=identity(path)
            if initial!=item['identity'] or digest(path)!=item['sha256']:
                raise ValueError('Changed JSON input: '+str(path))
            source=json.loads(path.read_text());records=source['records'];args=source.get('args') or {}
            if len(records)!=item['record_count']:raise ValueError('Changed record count')
            clocks=[float(r['T']) for r in records]
            if not all(math.isfinite(t) for t in clocks) or any(b<a for a,b in zip(clocks,clocks[1:])):
                raise ValueError('Unordered record clocks')
            clock_counts=Counter(clocks)
            duplicates=[i for i,t in enumerate(clocks) if clock_counts[t]>1]
            fields_path=Path(item['fields']['path']);fields=None;field_identity=None;field_sha=None;clock_indices={}
            if item['fields']['exists']:
                field_identity=identity(fields_path)
                if field_identity!=item['fields']['identity']:raise ValueError('Changed fields identity')
                field_sha=digest(fields_path)
                with np.load(fields_path,allow_pickle=False) as z:
                    keys=('R','T','N','v','N_applied','v_applied','g')
                    fields={k:z[k] for k in keys if k in z.files}
                if not all(k in fields for k in ('R','T','N','v','N_applied','v_applied','g')):
                    raise ValueError('Missing required numeric root members')
                with zipfile.ZipFile(fields_path) as z:
                    names=set(z.namelist())
                    if all(k+'.npy' in names for k in ('r_nodes','N_nodes','v_nodes')):
                        try:
                            nodes={k:nodes_member(z,k) for k in ('r_nodes','N_nodes','v_nodes')}
                            fields.update(nodes)
                        except (ValueError,pickle.UnpicklingError,TypeError) as error:
                            fields['node_unavailable']=type(error).__name__+': '+str(error)
                for i,t in enumerate(fields['T']):clock_indices.setdefault(float(t),[]).append(i)
                for k in ('N','v','N_applied','v_applied'):
                    if fields[k].shape!=(len(fields['T']),len(fields['R'])):
                        raise ValueError('Root array shape mismatch')
            result=dict(member=item['label'],source=initial,json_sha256=item['sha256'],
                fields_identity=field_identity,fields_sha256=field_sha,record_count=len(records),
                duplicate_record_indices=duplicates,records=[],missing=[])
            if fields is None:result['missing'].append('Literal fields counterpart absent; no substitute member used')
            for record_index,record in enumerate(records):
                clock=float(record['T']);row=dict(index=record_index,T=clock,T_over_rs=record.get('T_over_rs'),
                    r_collar=record.get('r_collar'),record_R_t=record.get('R_t'),
                    record_r_in=record.get('r_in'),record_trapping_zeros=record.get('trapping_zeros'),
                    record_tangencies=record.get('tangencies'),record_E=record.get('E'),
                    raw_gates=raw_gates(record,args),roots={})
                matches=clock_indices.get(clock,[])
                if fields is None or len(matches)!=1 or clock_counts[clock]!=1:
                    row['root_unavailable']='No unique JSON-to-NPZ clock join; no lineage substitution'
                else:
                    j=matches[0];R=fields['R'];N=fields['N_applied'][j];v=fields['v_applied'][j]
                    # Preserve the prior own T16 applied-root calculation, binding its arrays.
                    reuse=(member_index==10 and record.get('T_over_rs')==16)
                    if reuse:
                        if (field_sha!='a7d74d1b5e8516caa68f8249e7794572ffa315ed92eccde824552e9b8d3f32bb'
                                or not np.array_equal(R,raw_t16['R'])
                                or not np.array_equal(N,raw_t16['N_applied'])
                                or not np.array_equal(v,raw_t16['v_applied'])):
                            raise ValueError('Historical applied-root reuse binding failed')
                        row['roots']['applied_sites']=dict(reused=True,source=str(CURRENT),source_sha256=CURRENT_SHA,
                            exact_saved_roots=current['applied_zeros'],exact_saved_outer=current['outer_applied_zero'],
                            outer_isolated_root=int(current['outer_applied_zero']['radius']['numerator'])/int(current['outer_applied_zero']['radius']['denominator']))
                    else:row['roots']['applied_sites']=roots(R,N,v)
                    row['roots']['solve_output_sites']=roots(R,fields['N'][j],fields['v'][j])
                    if 'r_nodes' in fields:
                        row['roots']['stored_nodes']=roots(fields['r_nodes'][j],fields['N_nodes'][j],fields['v_nodes'][j])
                    elif 'node_unavailable' in fields:
                        row['stored_node_roots_unavailable']=fields['node_unavailable']
                    E=record.get('E');g=float(fields['g']);args_g=args.get('g')
                    normalize=(isinstance(E,(float,int)) and math.isfinite(E) and E>0
                        and math.isfinite(g) and g>0 and isinstance(args_g,(float,int))
                        and math.isfinite(args_g) and g==args_g)
                    row['normalization_contract']='finite positive E and g bound to args.g' if normalize else 'unavailable: energy/g/argument binding'
                    for value in row['roots'].values():
                        outer=value.get('outer_isolated_root');rt=record.get('R_t')
                        value['outer_linear_minus_record_R_t']=outer-rt if outer is not None and isinstance(rt,(int,float)) and math.isfinite(rt) else None
                        value['outer_R_over_record_2gE']=outer/(2*g*E) if outer is not None and normalize else None
                        record_list=record.get('trapping_zeros')
                        actual_list=([x['radius'] for x in value['isolated_roots']] if 'isolated_roots' in value else
                            [int(x['radius']['numerator'])/int(x['radius']['denominator']) for x in value.get('exact_saved_roots',[])])
                        if (isinstance(record_list,list) and all(isinstance(x,(int,float)) and math.isfinite(x) for x in record_list)
                                and record_list==sorted(record_list) and not value.get('zero_intervals')
                                and len(actual_list)==len(record_list)):
                            value['ordered_linear_minus_record_root_list']=[a-b for a,b in zip(actual_list,record_list)]
                            value['list_comparison_scope']='Ordered equal-length saved-array and record lists; different interpolation representations'
                        else:
                            value['ordered_linear_minus_record_root_list']=None
                            value['list_comparison_scope']='Unresolved correspondence: unequal/unknown lists or zero intervals; no guessed match'
                    row['field_index']=j
                result['records'].append(row)
            if identity(path)!=initial or (fields_path.exists() and identity(fields_path)!=field_identity):
                raise ValueError('Input changed during audit')
            def first(predicate):
                for r in result['records']:
                    if predicate(r):return dict(index=r['index'],T=r['T'],T_over_rs=r['T_over_rs'])
                return None
            flag_states=[r['raw_gates']['raw_flag_pass'] for r in result['records']]
            confirming_states=[r['raw_gates']['confirming_gate_recalculated'] for r in result['records']]
            kept_states=[r['raw_gates']['kept_field_gate_recalculated'] for r in result['records']]
            combined=[False if a is False or b is False else True if a is True and b is True else None
                      for a,b in zip(flag_states,confirming_states)]
            all_recorded=[False if any(x is False for x in values) else True if all(x is True for x in values) else None
                for values in zip(flag_states,kept_states,confirming_states)]
            endpoints={}
            for key,states in (('raw_flag_prefix',flag_states),('explicit_confirming_prefix',confirming_states),
                               ('raw_flag_and_explicit_confirming_prefix',combined),
                               ('all_explicit_recorded_gates_prefix',all_recorded)):
                index=prefix(states)
                endpoints[key]=None if index is None else dict(index=index,T=clocks[index],T_over_rs=records[index].get('T_over_rs'))
            def threshold_status(key):
                available=[isinstance(r['raw_gates'].get(key),(int,float)) and math.isfinite(r['raw_gates'][key]) for r in result['records']]
                unavailable=first(lambda r:not (isinstance(r['raw_gates'].get(key),(int,float)) and math.isfinite(r['raw_gates'][key])))
                crossing=first(lambda r:isinstance(r['raw_gates'].get(key),(int,float)) and math.isfinite(r['raw_gates'][key]) and r['raw_gates'][key]>1e-3)
                return dict(coverage_status='complete' if all(available) else 'unresolved',
                    available_records=sum(available),unavailable_records=len(available)-sum(available),
                    first_unavailable=unavailable,first_observed_above_1e_3=crossing)
            result['summary']=dict(**endpoints,
                recorded_threshold_metrics={key:threshold_status(key) for key in ('native_kept_residual','native_last_iterate_residual','solver_residual_W','solver_residual_invN')},
                first_capped_record='unavailable: no literal cap-marker contract bound; raw flag failure remains separate',
                joined_field_records=sum('field_index' in r for r in result['records']),
                first_raw_flag_failure=first(lambda r:r['raw_gates']['raw_flag_pass'] is False),
                first_explicit_confirming_failure=first(lambda r:r['raw_gates']['confirming_gate_recalculated'] is False),
                first_native_kept_above_1e_3=first(lambda r:isinstance(r['raw_gates']['native_kept_residual'],(float,int)) and r['raw_gates']['native_kept_residual']>1e-3),
                native_plain_last_iterate_crossing='unavailable: format norm not bound',
                first_root_unavailable=first(lambda r:'root_unavailable' in r),
                first_recorded_W_above_1e_3=first(lambda r:isinstance(r['raw_gates']['solver_residual_W'],(float,int)) and r['raw_gates']['solver_residual_W']>1e-3),
                first_recorded_invN_above_1e_3=first(lambda r:isinstance(r['raw_gates']['solver_residual_invN'],(float,int)) and r['raw_gates']['solver_residual_invN']>1e-3),
                independent_row_prefix=None)
            target=ROOT/f'member_{member_index:02d}.json';save(target,result)
            report['members'].append(dict(member_index=member_index,member=item['label'],path=str(target),
                size=target.stat().st_size,sha256=digest(target),summary=result['summary'],missing=result['missing']))
            report['remaining_member_indices'].remove(member_index)
            # A separate exclusive checkpoint per completed member preserves progress.
            save(ROOT/f'checkpoint_{member_index:02d}.json',report)
            del fields,result
        report['status']='completed_roots_and_raw_gates_only'
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(2400,2400));report['elapsed_seconds']=time.monotonic()-start
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save(ROOT/'summary.json',report)
    if report['status']!='completed_roots_and_raw_gates_only':raise SystemExit(1)


if __name__=='__main__':main()
