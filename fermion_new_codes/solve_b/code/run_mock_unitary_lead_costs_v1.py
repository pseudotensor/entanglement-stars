#!/usr/bin/env python3
"""Box-only missing-cost continuation; no replay of the completed fixture prefix."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--authorchecks',required=True)
    parser.add_argument('--continuation-contract',required=True)
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing run directory')
    expected_spec='495e59d68d57061246eb82dfe76ca26bceede021d29c1ba9fffcd63024184b6c'
    if hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()!=expected_spec:
        raise ValueError('Frozen mathematical spec identity mismatch')
    if hashlib.sha256(Path(args.authorchecks).read_bytes()).hexdigest()!='02aed9d4e45d5b1ec359a096338039803fc61f38f66536a29922e372f20b1446':
        raise ValueError('Frozen source addendum identity mismatch')
    if hashlib.sha256(Path(args.continuation_contract).read_bytes()).hexdigest()!='d78e9ee13c1f03c1cbb774e6c02ee73cb5486b05cd6f7a011460ce1ae3229b62':
        raise ValueError('Frozen continuation contract identity mismatch')
    from cpu_runtime import require_remote_cpu
    env=require_remote_cpu(out)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        if os.environ.get(name)!='1':
            raise RuntimeError('Exactly one thread required: '+name)
    import numpy as np
    import scipy
    import mpmath
    import ctypes
    import io
    import contextlib
    from scipy.linalg import expm
    from scipy.sparse.linalg import expm_multiply
    import mock_unitary_lead_v1 as q
    if len(list(Path('/proc/self/task').iterdir()))!=1:
        raise RuntimeError('Exactly one observed resident thread required')
    env['threads']=len(list(Path('/proc/self/task').iterdir()))
    if np.finfo(float).eps!=2**-52 or np.finfo(float).smallest_subnormal!=2**-1074:
        raise RuntimeError('Unsupported IEEE binary64 environment')
    if ctypes.CDLL(None).fegetround()!=0:
        raise RuntimeError('FE_TONEAREST required')
    tiny=np.float64(np.finfo(float).tiny)
    if tiny*np.float64(.5)==0 or np.float64(2)*np.float64(2**-1074)!=np.float64(2**-1073):
        raise RuntimeError('Gradual underflow required')
    config=io.StringIO()
    with contextlib.redirect_stdout(config): np.show_config()
    identities={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__),Path(q.__file__),Path(args.spec),Path(args.authorchecks),Path(args.continuation_contract),Path(__file__).with_name('cpu_runtime.py')]}
    intake={'runtime':env,'identities':identities,'versions':{'numpy':np.__version__,
        'scipy':scipy.__version__,'mpmath':mpmath.__version__},'backend':config.getvalue(),
        'command':sys.argv,'thread_environment':{name:os.environ.get(name) for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')},
        'rounding':'FE_TONEAREST','gradual_underflow':True,'python':sys.version,'platform':platform.platform()}
    with (out/'intake.json').open('x') as f: json.dump(intake,f,indent=2,allow_nan=False)
    rows=[]; arrays={}; costs=[]
    started=time.perf_counter()

    def clean(x):
        if isinstance(x,np.ndarray):
            if np.iscomplexobj(x): return clean({'real':x.real.tolist(),'imag':x.imag.tolist()})
            return clean(x.tolist())
        if isinstance(x,complex): return clean({'real':x.real,'imag':x.imag})
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items() if not str(k).startswith('_')}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,np.generic): return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x): return None
        return x

    journal=(out/'progress.ndjson').open('x')
    timing_telemetry=(out/'timing_only_measurements.ndjson').open('x')
    current_role='continuation_admission'
    action_journal=(out/'numerical_actions.ndjson').open('x');action_summaries=[]
    def numerical_recorder(record):
        action_journal.write(json.dumps(clean(record),allow_nan=False)+'\n');action_journal.flush()
        if record['kind']=='action':
            action_summaries.append({k:v for k,v in record['budget'].items() if k!='phase_component_errors'})
    q.set_recorder(numerical_recorder)
    def persist(row):
        journal.write(json.dumps(clean(row),allow_nan=False)+'\n');journal.flush()

    def assertion_row(name,value,limit=0.,detail=None):
        try: val=float(value);bound=float(limit)
        except (TypeError,ValueError,OverflowError): val=float('nan');bound=float('nan')
        passed=np.isfinite(val) and np.isfinite(bound) and bound>=0 and val<=bound
        return {'name':name,'pass':bool(passed),'value':val,'limit':bound,'detail':detail}

    def check(name,value,limit=0.,detail=None):
        if current_role=='timing_only_remeasurement':
            metric={'role':current_role,'metric':name,'value':float(value),'reference_limit':float(limit),'detail':detail}
            timing_telemetry.write(json.dumps(clean(metric),allow_nan=False)+'\n');timing_telemetry.flush()
            if not np.isfinite(float(value)) or not np.isfinite(float(limit)):
                raise ValueError('Nonfinite timing-workload telemetry')
            return None
        row=assertion_row(name,value,limit,detail);row['role']=current_role;rows.append(row);persist(row)
        return row['pass']

    def expect_reject(name,fun):
        try: fun()
        except (ValueError,PermissionError,RuntimeError) as e:
            rows.append({'name':name,'pass':True,'rejection':str(e)})
        else: rows.append({'name':name,'pass':False,'rejection':None})
        persist(rows[-1])

    def save(name,**data):
        with (out/('arrays_'+name+'.npz')).open('xb') as f:
            np.savez_compressed(f,**data)
        for key,value in data.items(): arrays[name+'__'+key]=np.asarray(value)

    def ent(b,psi,A,err=0.):
        r=q.reduced(b,psi,A); e=q.entropy(r,err)
        return e

    def interval_violation(lo,hi,target):
        if q.I(lo).b<=target.a and q.I(hi).a>=target.b: return 0.
        return max(q.upper(q.I(lo)-target.a),q.upper(target.b-q.I(hi)))

    def vector_target_error(vector,targets):
        vector=np.asarray(vector)
        if vector.shape!=(len(targets),) or any(len(t)!=2 for t in targets):
            raise ValueError('analytic vector target shape mismatch')
        total=q.I(0)
        for z,(re,im) in zip(vector,targets):
            total+=abs(q.I(float(z.real))-re)**2+abs(q.I(float(z.imag))-im)**2
        return q.upper(q.iv.sqrt(total))

    def ent_known(name,b,psi,A,expected,err=0.):
        e=ent(b,psi,A,err)
        target=q.I(0) if expected==0 else q.iv.ln(q.I(2))
        check(name,q.upper(abs(q.I(e['value'])-target)),5e-10,e)
        check(name+'_enclosure',interval_violation(e['lower'],e['upper'],target),0.)
        return e

    def direct_ladder(modes,mode,dagger):
        # Independent tensor-factor Jordan-Wigner construction in integer order.
        ident=np.eye(2); z=np.diag([1.,-1.])
        op=np.array([[0.,0.],[1.,0.]]) if dagger else np.array([[0.,1.],[0.,0.]])
        result=np.ones((1,1))
        for j in reversed(range(modes)):
            result=np.kron(result,z if j<mode else op if j==mode else ident)
        return result

    cost_journal=(out/'cost_stages.ndjson').open('x')
    def cost_checkpoint(record):
        cost_journal.write(json.dumps(clean(record),allow_nan=False)+'\n');cost_journal.flush()
    # Two predeclared free-control cost sizes; no interacting mock propagation.
    for sites in (2,3):
        measurement_role='timing_only_remeasurement' if sites==2 else 'new_n3_control'
        current_role=measurement_role
        wall0=time.perf_counter(); cpu0=time.process_time()
        begin=time.perf_counter(); begin_cpu=time.process_time()
        basis=q.Basis.make(8*sites,2)
        h=q.free_one_body(sites,1.,1.,np.ones(sites),np.zeros(sites))
        q.verify_dyadic_free_fixture(h,sites,1.,1.,np.ones(sites),np.zeros(sites))
        sparse,he=q.assemble(basis,q.quadratic_terms(h)); dense=sparse.toarray()
        timings={'basis_assembly_wall':time.perf_counter()-begin,'basis_assembly_cpu':time.process_time()-begin_cpu}
        cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'basis_assembly','timings':timings})
        begin=time.perf_counter(); begin_cpu=time.process_time(); solver=q.Spectral(dense,he)
        orbital_solver=q.Spectral(h)
        timings['eigensystem_wall']=time.perf_counter()-begin
        timings['eigensystem_cpu']=time.process_time()-begin_cpu
        cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'eigensystem','timings':timings})
        initial=basis.ket([0,1]); ts=np.linspace(0,1,17)
        states=[]; errors=[]; independent=[]; entropies=[]; core_entropies=[]; state_budgets=[]; independent_budgets=[]
        timing_prop=0.; timing_red=0.; timing_prop_cpu=0.; timing_red_cpu=0.;independent_wall=0.;independent_cpu=0.; max_state=0.; max_norm=0.; max_energy=0.; max_nl=0.
        energies=[]; counts=[]; mode_counts=[]; light_entropies=[]
        for t in ts:
            begin=time.perf_counter(); begin_cpu=time.process_time(); psi,e=solver.apply(initial,float(t))
            state_budgets.append(dict(solver.last_budget))
            timing_prop+=time.perf_counter()-begin
            timing_prop_cpu+=time.process_time()-begin_cpu
            cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'propagation_subphase','clock':float(t),'completed_wall':timing_prop,'completed_cpu':timing_prop_cpu,'budget':state_budgets[-1]})
            independent_start=time.perf_counter();independent_start_cpu=time.process_time()
            orbitals=[];orbital_errors=[];orbital_budgets=[]
            for j in (0,1):
                one=np.eye(basis.modes,dtype=complex)[j]
                evolved,oe=orbital_solver.apply(one,float(t))
                orbitals.append(evolved);orbital_errors.append(oe);orbital_budgets.append(dict(orbital_solver.last_budget))
            orbital=np.stack(orbitals,axis=1)
            determinant,det_error=q.wedge_state(basis,*orbitals,*orbital_errors)
            independent_budgets.append({'clock':float(t),'orbital_budgets':orbital_budgets,'determinant_error':det_error})
            independent_wall+=time.perf_counter()-independent_start;independent_cpu+=time.process_time()-independent_start_cpu
            cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'independent_subphase','clock':float(t),'completed_wall':independent_wall,'completed_cpu':independent_cpu,'budget':independent_budgets[-1]})
            save(f'cost_{sites}_clock_{t}',state=psi,determinant=determinant,orbitals=orbital,state_error=e,independent_error=det_error,orbital_errors=orbital_errors)
            deviation=q.exact_vector_distance(psi,determinant)
            check(f'cost_{sites}_state_{t}',deviation,2e-11)
            check(f'cost_{sites}_independent_within_bound_{t}',q.upper(q.I(deviation)-q.I(e)-q.I(det_error)),0.)
            max_state=max(max_state,deviation); max_norm=max(max_norm,abs(np.vdot(psi,psi)-1))
            energy=float(np.vdot(psi,dense@psi).real); energies.append(energy)
            max_energy=max(max_energy,abs(energy-np.vdot(initial,dense@initial).real))
            nl=sum(abs(z)**2*sum(1 for j in range(basis.modes) if mask>>j&1 and j%8>=4) for mask,z in zip(basis.states,psi))
            max_nl=max(max_nl,nl); counts.append(nl)
            observed=np.array([sum(abs(z)**2*((mask>>j)&1) for mask,z in zip(basis.states,psi)) for j in range(basis.modes)])
            expected_counts=np.sum(abs(orbital)**2,axis=1)
            check(f'cost_{sites}_occupations_{t}',np.max(abs(observed-expected_counts)),2e-11)
            check(f'cost_{sites}_independent_energy_{t}',abs(energy-np.trace(orbital.conj().T@h@orbital)),2e-11*max(1,np.linalg.norm(dense)))
            mode_counts.append(observed)
            begin=time.perf_counter(); begin_cpu=time.process_time()
            output=list(range(8,basis.modes))
            er=ent(basis,psi,output,e); ei=ent(basis,determinant,output,det_error)
            core=ent(basis,psi,list(range(8)),e)
            light_entropy=ent(basis,psi,[j for j in range(basis.modes) if j%8>=4],e)
            check(f'cost_{sites}_light_entropy_{t}',abs(light_entropy['value']),5e-10)
            light_entropies.append(light_entropy);core_entropies.append(core)
            check(f'cost_{sites}_entropy_independent_{t}',abs(er['value']-ei['value']),5e-10)
            check(f'cost_{sites}_purity_complement_{t}',abs(er['value']-core['value']),5e-10)
            timing_red+=time.perf_counter()-begin
            timing_red_cpu+=time.process_time()-begin_cpu
            entropies.append(er);states.append(psi);independent.append(determinant);errors.append(e)
            cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'ordinary_clock','clock':float(t),'output_entropy':er,'core_entropy':core,'light_entropy':light_entropy,'state_error':e,'independent_error':det_error,'budget':state_budgets[-1],'completed_clock_timings':{'propagation_wall':timing_prop,'propagation_cpu':timing_prop_cpu,'independent_wall':independent_wall,'independent_cpu':independent_cpu,'reduction_wall':timing_red,'reduction_cpu':timing_red_cpu}})
        timings['propagation_wall']=timing_prop;timings['reduction_entropy_wall']=timing_red
        timings['propagation_cpu']=timing_prop_cpu;timings['reduction_entropy_cpu']=timing_red_cpu
        timings['independent_orbital_wedge_wall']=independent_wall;timings['independent_orbital_wedge_cpu']=independent_cpu
        cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'ordinary_complete','timings':timings})
        check(f'cost_{sites}_norm',max_norm,2e-12)
        check(f'cost_{sites}_energy',max_energy,2e-11*max(1,np.linalg.norm(dense)))
        check(f'cost_{sites}_zero_conversion',max_nl,2e-11)
        # Same free control with independent fixed pair code, reference dimension2.
        begin=time.perf_counter(); begin_cpu=time.process_time()
        words=np.stack([basis.ket([0,1]),basis.ket([2,3])]);q.validate_code(basis,words)
        registered_states=[]; register_diagnostics=[]
        register_schedule=q.schedule_identity({'operator':q.array_identity(dense),'clocks':ts.tolist(),'code':[[0,1],[2,3]]})
        register_max_norm=0.;register_max_energy=0.;register_max_wedge=0.
        reference_initial=words/np.sqrt(2)
        register_plan=q.RegisterPlan.make(basis,f'cost_{sites}_register',reference_initial,register_schedule,ts,[j for j in range(8,basis.modes) if j%8>=4],list(range(8,basis.modes)))
        reference_energy=sum(np.vdot(row,dense@row).real for row in reference_initial)
        reference_initial_error=q.upper(q.iv.sqrt(2)*abs(q.I(float(1/np.sqrt(2)))-1/q.iv.sqrt(2)))
        for t in ts:
            registered,re=solver.apply(reference_initial,float(t),reference_initial_error)
            registered_budget=dict(solver.last_budget)
            orbitals=[];orbital_errors=[];orbital_budgets=[]
            for j in range(4):
                one=np.eye(basis.modes,dtype=complex)[j]
                evolved,oe=orbital_solver.apply(one,float(t))
                orbitals.append(evolved);orbital_errors.append(oe);orbital_budgets.append(dict(orbital_solver.last_budget))
            branches=[];branch_errors=[]
            for j in (0,2):
                w,we=q.wedge_state(basis,orbitals[j],orbitals[j+1],orbital_errors[j],orbital_errors[j+1])
                branches.append(w);branch_errors.append(we)
            branches=np.stack(branches);weight=float(1/np.sqrt(2));wedge=weight*branches
            wedge_error=q.upper(q.iv.sqrt(sum((q.I(x)**2 for x in branch_errors),q.I(0)))/q.iv.sqrt(2)+
                abs(q.I(weight)-1/q.iv.sqrt(2))*q.I(q.fnorm(branches))+q.I(q.real_product_round(wedge)))
            save(f'cost_{sites}_register_clock_{t}',state=registered,wedge=wedge,orbitals=orbitals,state_error=re,independent_error=wedge_error,orbital_errors=orbital_errors)
            disagreement=q.exact_vector_distance(registered,wedge)
            norm_drift=abs(np.vdot(registered,registered)-1)
            reference_energy_now=sum(np.vdot(row,dense@row).real for row in registered)
            energy_drift=abs(reference_energy_now-reference_energy)
            check(f'cost_{sites}_register_wedge_{t}',disagreement,2e-11)
            check(f'cost_{sites}_register_combined_bounds_{t}',q.upper(q.I(disagreement)-q.I(re)-q.I(wedge_error)),0.)
            check(f'cost_{sites}_register_norm_{t}',norm_drift,2e-12)
            check(f'cost_{sites}_register_energy_{t}',energy_drift,2e-11*max(1,np.linalg.norm(dense)))
            register_max_norm=max(register_max_norm,norm_drift)
            register_max_energy=max(register_max_energy,energy_drift)
            register_max_wedge=max(register_max_wedge,disagreement)
            regsnap=q.Snapshot.make(basis,registered,f'cost_{sites}_register','primary',reference_initial,register_schedule,float(t),re,dense)
            dl=q.register_diagnostic(register_plan,regsnap,'light')
            da=q.register_diagnostic(register_plan,regsnap,'all')
            Sr=q.entropy(q.reduced(basis,registered,[],True),re)
            ent_records={}
            for label,diagnostic in [('light',dl),('all',da)]:
                O=diagnostic['output']; K=diagnostic['complement']
                so=ent(basis,registered,O,re);sk=ent(basis,registered,K,re)
                sro=q.entropy(q.reduced(basis,registered,O,True),re)
                srk=q.entropy(diagnostic['joint'],re)
                check(f'cost_{sites}_register_pure_accounting_{label}_{t}',abs(so['value']+sk['value']-sro['value']-srk['value']),5e-10)
                ent_records[label]={'output':so,'complement':sk,'register_output':sro,'register_complement':srk}
            register_diagnostics.append({'clock':float(t),'light':dl,'all':da,'state_error':re,
                'error_budget':registered_budget,'orbital_budgets':orbital_budgets,'norm_drift':float(norm_drift),'energy':float(reference_energy_now),
                'wedge_disagreement':float(disagreement),'independent_state_error':wedge_error,'register_entropy':Sr,'entropies':ent_records})
            registered_states.append(registered)
            cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'register_clock','record':register_diagnostics[-1],'elapsed_registered_wall':time.perf_counter()-begin,'elapsed_registered_cpu':time.process_time()-begin_cpu})
        timings['register_inclusive_wall']=time.perf_counter()-begin
        timings['register_inclusive_cpu']=time.process_time()-begin_cpu
        cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'register_complete','timings':timings})
        begin=time.perf_counter(); begin_cpu=time.process_time()
        terms=q.quadratic_terms(h);protected=q.protect(terms,[list(range(8,basis.modes))],basis.modes)
        kh,ke=q.assemble(basis,protected);K=kh.toarray(); kp=q.Spectral(K,ke)
        ret,retproof=q.return_segment(basis,f'cost_{sites}_return',terms,[list(range(8,basis.modes))],solver,kp,initial,1.,0.,16,schedule=q.schedule_identity({'operator':q.array_identity(dense),'protected':q.array_identity(K),'start':0.,'end':1.}))
        timings['return_comparator_wall']=time.perf_counter()-begin
        timings['return_comparator_cpu']=time.process_time()-begin_cpu
        cost_checkpoint({'sites':sites,'role':measurement_role,'phase':'return_complete','timings':timings,'return_bound':ret,'protection':retproof})
        cost={'measurement_role':measurement_role,'sites':sites,'modes':basis.modes,'number':2,'dimension':basis.dimension,
            'reference_dimension':2,'nnz':int(sparse.nnz),'dense_bytes':dense.nbytes,
            'sparse_bytes':sparse.data.nbytes+sparse.indices.nbytes+sparse.indptr.nbytes,
            'clocks':17,'timings':timings,'total_wall':time.perf_counter()-wall0,
            'total_cpu':time.process_time()-cpu0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'max_state_disagreement':max_state,'max_norm_drift':float(max_norm),
            'register_max_norm_drift':float(register_max_norm),'register_max_energy_drift':float(register_max_energy),
            'register_max_wedge_disagreement':float(register_max_wedge),
            'max_energy_drift':max_energy,'max_light_count':max_nl,'max_state_error':max(errors),
            'max_entropy_width':max(z['upper']-z['lower'] for z in entropies),
            'register_diagnostics':register_diagnostics,'return_comparator':ret,'return_protection':retproof,
            'spectral_eta':solver.eta,'spectral_residual':solver.residual,'state_budgets':state_budgets,'independent_budgets':independent_budgets,
            'entropy_diagnostics':entropies,'light_entropy_diagnostics':light_entropies,'core_entropy_diagnostics':core_entropies}
        costs.append(cost)
        cost_checkpoint({'phase':'size_complete','cost':cost})
        save(f'cost_{sites}',states=states,independent=independent,clocks=ts,
             state_errors=errors,energies=energies,light_counts=counts,
             registered=registered_states,occupations=mode_counts,hamiltonian=dense,one_body=h,protected=K,
             basis=np.array(basis.states,dtype=np.uint64),eigenvalues=solver.values,eigenvectors=solver.vectors,
             onebody_eigenvalues=orbital_solver.values,onebody_eigenvectors=orbital_solver.vectors)
    current_role='final_runtime_source_checks'
    for path,digest in identities.items():
        check('source_unchanged_'+Path(path).name,0. if hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest else 1.,0.)
    env['threads_final']=len(list(Path('/proc/self/task').iterdir()))
    check('resident_threads_final',abs(env['threads_final']-1),0.)
    result={'scope':'missing-only costs: n2 timing remeasurement, new n3 control; no fixture replay or mock','runtime':env,
        'intake':intake,'identities':identities,
        'controls':rows,'costs':costs,'all_passed':all(r['pass'] for r in rows),
        'total_wall':time.perf_counter()-started}
    with (out/'controls.json').open('x') as f: json.dump(clean(result),f,indent=2,allow_nan=False)
    with (out/'raw_arrays.npz').open('xb') as f: np.savez_compressed(f,**arrays)
    journal.close();action_journal.close();cost_journal.close();timing_telemetry.close();q.set_recorder(None)
    lines=['# Manufactured harness self-tests','',f"Passed {sum(r['pass'] for r in rows)}/{len(rows)} controls.",
           'No physical axial mock was run. Raw records stay on this box.','']
    lines+=['## Runtime and scientific/source bindings','',json.dumps(clean(intake),indent=2),'',
        '## Complete manufactured control report','',
        '| Control | Passed | Observed error/value | Fixed limit | Diagnostic |',
        '|---|---|---:|---:|---|']
    scalar_keys=('value','lower','upper','state_error','normalization_error','numerical_margin',
        'propagation_margin','negative_mass','eta','eigen_residual','dimension','trace','hermiticity',
        'status','conditional_upper','cumulative_upper','numerical_consistency','disagreement_upper',
        'passes_phase','total','input','spectral_defect','operator_assembly','coefficient_action',
        'phase_approximation','phase_multiplication','final_action')
    for row in rows:
        detail=row.get('detail')
        readable={k:detail[k] for k in scalar_keys if isinstance(detail,dict) and k in detail}
        if isinstance(detail,dict) and 'comparison' in detail:
            comp=detail['comparison']; readable['comparison']={k:comp[k] for k in
                ('B','B_previous','computed_trace_distance_bound','primary_error','protected_error','direct_trace_distance') if k in comp}
        if row.get('rejection') is not None: readable['expected_rejection']=row['rejection']
        text=json.dumps(clean(readable),separators=(',',':')).replace('|','/')
        lines.append('| '+row['name']+' | '+str(row['pass'])+' | '+str(clean(row.get('value')))+' | '+str(clean(row.get('limit')))+' | '+text+' |')
    lines+=['','## Propagation component summaries','',
        'Every cost-workload action, including independent orbitals and return quadrature, is retained in numerical_actions.ndjson.',
        '| Action | Time | Input | Spectral defect | Residual | Assembly | Coefficient action | Phase enclosure | Phase multiply | Final action | Total |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i,a in enumerate(action_summaries):
        keys=('represented_time','input','spectral_defect','eigen_residual','operator_assembly','coefficient_action','phase_approximation','phase_multiplication','final_action','total')
        lines.append('| '+str(i)+' | '+' | '.join(str(a[k]) for k in keys)+' |')
    for c in costs:
        lines+=['',f"### Size {c['sites']} ({c['measurement_role']}): ordinary entropy intervals",'',
            '| Clock | Lead S [lo,hi] | Core S [lo,hi] | Light S [lo,hi] |','|---:|---|---|---|']
        for t,lead,core,light in zip(np.linspace(0,1,17),c['entropy_diagnostics'],c['core_entropy_diagnostics'],c['light_entropy_diagnostics']):
            lines.append('| '+str(t)+' | '+' | '.join(str([r['lower'],r['upper']]) for r in (lead,core,light))+' |')
        lines+=['',f"### Size {c['sites']} ({c['measurement_role']}): reference readout intervals",'',
            '| Clock | Light D [lower,upper] | Complete D [lower,upper] | S(R) | Light S(O),S(K) | Complete S(O),S(K) | State / independent error |',
            '|---:|---|---|---:|---|---|---|']
        for r in c['register_diagnostics']:
            lines.append('| '+str(r['clock'])+' | '+str([r['light']['lower'],r['light']['upper']])+' | '+
                str([r['all']['lower'],r['all']['upper']])+' | '+str(r['register_entropy']['value'])+' | '+
                str([r['entropies']['light'][k]['value'] for k in ('output','complement')])+' | '+
                str([r['entropies']['all'][k]['value'] for k in ('output','complement')])+' | '+
                str([r['state_error'],r['independent_state_error']])+' |')
        lines+=['','| Clock/role | S(O) interval | S(K) interval | S(RO) interval | S(RK) interval |',
            '|---|---|---|---|---|']
        for r in c['register_diagnostics']:
            for kind in ('light','all'):
                e0=r['entropies'][kind]
                lines.append('| '+str(r['clock'])+'/'+kind+' | '+' | '.join(str([e0[k]['lower'],e0[k]['upper']]) for k in ('output','complement','register_output','register_complement'))+' |')
        lines+=['', 'Return-helper upper bound: '+str(c['return_comparator']['upper'])+
            '; Lipschitz upper: '+str(c['return_comparator']['lipschitz'])+
            '; difference-operator allowance: '+str(c['return_comparator']['difference_operator_error'])+'.']
        lines+=['',f"## {c['sites']} sites ({c['measurement_role']}), {c['modes']} modes, sector dimension {c['dimension']}",
                json.dumps(clean({k:v for k,v in c.items() if k not in ('register_diagnostics','return_comparator','return_protection','entropy_diagnostics','light_entropy_diagnostics','core_entropy_diagnostics','state_budgets','independent_budgets')}),indent=2)]
    (out/'written_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(out),'controls':len(rows),'passed':sum(r['pass'] for r in rows),
        'all_passed':result['all_passed'],'total_wall':result['total_wall']}))
    if not result['all_passed']: raise SystemExit(1)

if __name__=='__main__':
    main()
