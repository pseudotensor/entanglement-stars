#!/usr/bin/env python3
"""Box-only manufactured self-tests; never a physical axial mock runner."""
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
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing run directory')
    expected_spec='495e59d68d57061246eb82dfe76ca26bceede021d29c1ba9fffcd63024184b6c'
    if hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()!=expected_spec:
        raise ValueError('Frozen mathematical spec identity mismatch')
    if hashlib.sha256(Path(args.authorchecks).read_bytes()).hexdigest()!='02aed9d4e45d5b1ec359a096338039803fc61f38f66536a29922e372f20b1446':
        raise ValueError('Frozen source addendum identity mismatch')
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
        [Path(__file__),Path(q.__file__),Path(args.spec),Path(args.authorchecks),Path(__file__).with_name('cpu_runtime.py')]}
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
        row=assertion_row(name,value,limit,detail);rows.append(row);persist(row)
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

    nonfinite_row={'array':np.array([np.nan,np.inf]),'complex':complex(np.inf,np.nan)}
    serialized=json.dumps(clean(nonfinite_row),allow_nan=False)
    check('null_safe_serialization',0. if 'NaN' not in serialized and 'Infinity' not in serialized else 1.,0.,clean(nonfinite_row))
    for name,value,limit in [('nonfinite',float('nan'),0.),('overflow',float('inf'),0.),('invalid_limit',0.,float('nan'))]:
        failed=assertion_row('manufactured_'+name,value,limit)
        json.dumps(clean(failed),allow_nan=False)
        check('assertion_rejects_'+name,0. if not failed['pass'] else 1.,0.,failed)
    expect_reject('analytic_target_length',lambda:vector_target_error(np.array([1.,0.]),[(q.I(1),q.I(0))]))
    expect_reject('analytic_target_shape',lambda:vector_target_error(np.array([[1.,0.]]),[(q.I(1),q.I(0)),(q.I(0),q.I(0))]))
    # Full-Fock CAR and literal independent tensor monomials.
    for m in (2,4):
        b=q.Basis.make(m)
        a=[direct_ladder(m,j,False) for j in range(m)]
        for j in range(m):
            for k in range(m):
                check(f'CAR_{m}_{j}_{k}',np.max(abs(a[j]@a[k].T+a[k].T@a[j]-(np.eye(2**m) if j==k else 0))),0.)
                check(f'CAR_ann_{m}_{j}_{k}',np.max(abs(a[j]@a[k]+a[k]@a[j])),0.)
        term=q.Term(1j,((0,True),(m-1,False)))
        h,eh=q.assemble(b,[term,term.adjoint()])
        exact=1j*a[0].T@a[-1]-1j*a[-1].T@a[0]
        check(f'full_fock_lift_{m}',np.max(abs(h.toarray()-exact)),0.)
        save(f'CAR_{m}',lift=h.toarray(),independent=exact)
    expect_reject('odd_monomial',lambda:q.assemble(q.Basis.make(2),[q.Term(1,((0,True),))]))
    expect_reject('sector_leakage',lambda:q.assemble(q.Basis.make(4,1),[q.Term(1,((0,True),(1,True)))]))
    leaking_ops=((0,True),(1,True))
    expect_reject('leakage_cancellation_mutation',lambda:q.assemble(q.Basis.make(4,1),
        [q.Term(c,leaking_ops) for c in (1e16,1.,-1e16)]))
    cancelled,ce0=q.assemble(q.Basis.make(4,0),[q.Term(1.,leaking_ops),q.Term(-1.,leaking_ops)])
    check('exact_omitted_sector_cancellation',float(cancelled.nnz),0.)
    expect_reject('continuity_fractional_dimension',lambda:q.continuity(.1,2.5))
    expect_reject('decoupling_missing_reference_axis',lambda:q.decoupling(q.Basis.make(2,1),np.array([1.,0.]),[1]))
    expect_reject('full_dimension_cap',lambda:q.Basis.make(11))
    expect_reject('sector_dimension_cap',lambda:q.Basis.make(24,4))

    # Explicit nonconstant fields and open one/two-step links.
    h=q.free_one_body(3,1,1,[.5,1,1],[-1,0,0])
    sx=np.array([[0,1],[1,0]]); alpha=np.kron(sx,sx); beta=np.diag([1,1,-1,-1])
    for species in (0,1):
        ix=lambda x:list(range(8*x+4*species,8*x+4*species+4))
        for x,N in enumerate((.5,1,1)):
            check(f'field_mass_{species}_{x}',np.max(abs(h[np.ix_(ix(x),ix(x))]-(N*beta if species==0 else 0))),0.)
        for x,y,answer in ((0,1,-3j*alpha/8),(1,2,-1j*alpha/2),(0,2,1j*np.eye(4)/8)):
            check(f'field_link_{species}_{x}_{y}',np.max(abs(h[np.ix_(ix(x),ix(y))]-answer)),0.)
    q.verify_dyadic_free_fixture(h,3,1,1,[.5,1,1],[-1,0,0])
    flat=q.free_one_body(3,1,1,[1,1,1],[0,0,0])
    q.verify_dyadic_free_fixture(flat,3,1,1,[1,1,1],[0,0,0])
    check('open_no_wrap',np.max(abs(flat[:8,16:])),0.)
    for name,fun in [('complex_lapse',lambda:q.free_one_body(2,1,1,[1+0j,1],[0,0])),
        ('scalar_field',lambda:q.free_one_body(2,1,1,1,[0,0])),
        ('zero_lapse',lambda:q.free_one_body(2,1,1,[0,1],[0,0])),
        ('complex_mass',lambda:q.free_one_body(2,1,1j,[1,1],[0,0]))]:
        expect_reject(name,fun)
    save('fields',nonconstant=h,flat=flat)

    # Two-mode exact free lead and spectral action, including reverse time.
    b=q.Basis.make(2,1); H=np.array([[0,1],[1,0]],complex)
    prop=q.Spectral(H); start=b.ket([0])
    save('free_eigensystem',hamiltonian=H,eigenvalues=prop.values,eigenvectors=prop.vectors)
    clocks=[0.,np.pi/8,np.pi/4,np.pi/2]
    for t in clocks:
        psi,e=prop.apply(start,t)
        expected=np.array([np.cos(t),-1j*np.sin(t)])
        deviation=np.linalg.norm(psi-expected)
        check(f'free_amplitude_{t}',deviation,2e-11)
        exact_target=[(q.iv.cos(q.I(t)),q.I(0)),(q.I(0),-q.iv.sin(q.I(t)))]
        check(f'free_analytic_observed_{t}',vector_target_error(psi,exact_target),2e-11)
        check(f'free_state_bound_{t}',q.upper(q.I(vector_target_error(psi,exact_target))-q.I(e)),0.)
        check(f'free_norm_{t}',abs(np.vdot(psi,psi)-1),2e-12)
        inverse,ei=prop.apply(psi,-t,e)
        check(f'free_inverse_{t}',np.linalg.norm(inverse-start),2e-11)
        independent=expm_multiply(-1j*t*H,start)
        check(f'free_independent_{t}',np.linalg.norm(psi-independent),2e-11)
        save('free_'+str(t),state=psi,expected=expected,bound=e)
    x,e=prop.apply(start,.25); y,e=prop.apply(x,.5,e); z,ez=prop.apply(start,.75)
    check('semigroup',np.linalg.norm(y-z),2e-11)

    # Manufactured quartet converter, signs, coherence and dark inputs.
    term=q.Term(1,((2,True),(3,True),(1,False),(0,False)))
    pair_terms=[term,term.adjoint()]
    b4=q.Basis.make(4); vp,ve=q.assemble(b4,pair_terms); V=vp.toarray()
    HH=b4.ket([0,1]); LL=b4.ket([2,3])
    check('pair_matrix_element',abs(np.vdot(LL,V@HH)-1),0.)
    for g in (0.,1.,-1.):
        solver=q.Spectral(g*V,ve*abs(g))
        save('pair_eigensystem_'+str(g),hamiltonian=solver.h,eigenvalues=solver.values,eigenvectors=solver.vectors)
        for t in clocks:
            psi,e=solver.apply(HH,t)
            ex=np.cos(g*t)*HH-1j*np.sin(g*t)*LL
            check(f'pair_amplitude_{g}_{t}',np.linalg.norm(psi-ex),2e-11)
            theta=q.I(g)*q.I(t)
            exact_target=[(q.I(0),q.I(0)) for _ in b4.states]
            exact_target[b4.states.index(3)]=(q.iv.cos(theta),q.I(0))
            exact_target[b4.states.index(12)]=(q.I(0),-q.iv.sin(theta))
            check(f'pair_analytic_observed_{g}_{t}',vector_target_error(psi,exact_target),2e-11)
            check(f'pair_bound_{g}_{t}',q.upper(q.I(vector_target_error(psi,exact_target))-q.I(e)),0.)
            a=np.vdot(HH,psi); c=np.vdot(LL,psi)
            rhoHL=a*c.conjugate()
            check(f'pair_coherence_{g}_{t}',vector_target_error(np.array([rhoHL]),[(q.I(0),q.iv.cos(theta)*q.iv.sin(theta))]),2e-11)
            nh=sum(abs(z)**2*((s&3).bit_count()) for s,z in zip(b4.states,psi))
            nl=sum(abs(z)**2*((s&12).bit_count()) for s,z in zip(b4.states,psi))
            check(f'pair_occupations_{g}_{t}',max(q.upper(abs(q.I(nh)-2*q.iv.cos(theta)**2)),q.upper(abs(q.I(nl)-2*q.iv.sin(theta)**2))),2e-11)
            check(f'pair_energy_{g}_{t}',abs(np.vdot(psi,g*V@psi)),2e-11*max(1,np.linalg.norm(g*V)))
            save(f'pair_{g}_{t}',state=psi,density=np.outer(psi,psi.conj()),bound=e)
        coherent=(HH+1j*LL)/np.sqrt(2)
        amp=q.I(float(1/np.sqrt(2)))
        initerr=q.upper(q.iv.sqrt(2)*abs(amp-1/q.iv.sqrt(2)))
        psi,e=solver.apply(coherent,np.pi/8,initerr)
        theta=q.I(g)*q.I(float(np.pi/8))
        exact_target=[(q.I(0),q.I(0)) for _ in b4.states]
        exact_target[b4.states.index(3)]=((q.iv.cos(theta)+q.iv.sin(theta))/q.iv.sqrt(2),q.I(0))
        exact_target[b4.states.index(12)]=(q.I(0),(q.iv.cos(theta)-q.iv.sin(theta))/q.iv.sqrt(2))
        check(f'pair_complex_analytic_observed_{g}',vector_target_error(psi,exact_target),2e-11)
        check(f'pair_complex_bound_{g}',q.upper(q.I(vector_target_error(psi,exact_target))-q.I(e)),0.)
        independent=expm_multiply(-1j*np.pi/8*g*V,coherent)
        check(f'pair_complex_input_{g}',np.linalg.norm(psi-independent),2e-11)
        inv,ei=solver.apply(psi,-np.pi/8,e)
        check(f'pair_inverse_{g}',np.linalg.norm(inv-coherent),2e-11)
        for name,ket in [('odd',[0]),('blocked',[0,1,2])]:
            psi,e=solver.apply(b4.ket(ket),np.pi/4)
            check(f'pair_dark_{name}_{g}',np.linalg.norm(psi-b4.ket(ket)),2e-11)
    # The incoherent mixture has the same populations but fails phase check.
    mixed=.5*np.outer(HH,HH.conj())+.5*np.outer(LL,LL.conj())
    pure=(HH-1j*LL)/np.sqrt(2);pure_density=np.outer(pure,pure.conj())
    incoherent=q.coherence_readout(mixed,b4.states.index(3),b4.states.index(12),.5j,2e-11)
    coherent=q.coherence_readout(pure_density,b4.states.index(3),b4.states.index(12),.5j,2e-11)
    check('coherence_same_populations',np.max(abs(np.diag(mixed)-np.diag(pure_density))),2e-11)
    check('coherence_rejects_incoherent',0. if not incoherent['passes_phase'] and coherent['passes_phase'] else 1.,0.,incoherent)
    save('incoherent',mixed=mixed,pure=pure_density)
    wrong=V.copy(); wrong[b4.states.index(12),b4.states.index(3)]*=-1
    expect_reject('pair_conjugation_mutation',lambda:q.Spectral(wrong))

    # Graded noncontiguous reduction, including the known wrong unsigned trace.
    graded=(b4.ket([0,1])+b4.ket([0,3])-b4.ket([1,2])+b4.ket([2,3]))/2
    eg=ent_known('graded_pure',b4,graded,[0,2],0.)
    reordered=ent_known('graded_reordered',b4,graded,[2,0],0.)
    ent_known('graded_empty',b4,graded,[],0.)
    ent_known('graded_full',b4,graded,list(range(4)),0.)
    unsigned=np.zeros((4,4),complex)
    for s,z in zip(b4.states,graded):
        aa=(s&1)|((s>>1)&2); cc=((s>>1)&1)|((s>>2)&2)
        unsigned[aa,cc]=z
    wrongrho=unsigned@unsigned.conj().T
    uw=np.linalg.eigvalsh(wrongrho); us=-sum(x*np.log(x) for x in uw if x>0)
    check('unsigned_mutation_detected',q.upper(abs(q.I(us)-q.iv.ln(2))),5e-10)
    save('graded',state=graded,correct=q.reduced(b4,graded,[0,2])['matrix'],unsigned=wrongrho)
    expect_reject('duplicate_subsystem',lambda:q.reduced(b4,graded,[0,0]))
    expect_reject('out_of_range_subsystem',lambda:q.reduced(b4,graded,[4]))
    ent_known('product_entropy',b4,b4.ket([0,2]),[0,1],0.)

    # Exact prescribed two-collection circuit; no physical lead substituted.
    bc=q.Basis.make(8,2)
    circuit=(bc.ket([0,2])+bc.ket([1,3]))/np.sqrt(2)
    q.validate_schedule(bc,[1.,2.],[[4,5],[6,7]])
    stage1=q.swaps(bc,circuit,[(0,4),(1,5)])
    stage2=q.swaps(bc,stage1,[(2,6),(3,7)])
    first=ent_known('circuit_first',bc,stage1,[4,5],np.log(2))
    old=ent_known('circuit_old_at_second',bc,stage2,[4,5],np.log(2))
    both=ent_known('circuit_final_union',bc,stage2,[4,5,6,7],0.)
    increment=q.conditional(both,old); cumulative=q.conditional(both,first)
    check('circuit_conditional',q.upper(abs(q.I(increment['value'])+q.iv.ln(2))),5e-10,increment)
    check('circuit_cumulative',q.upper(abs(q.I(cumulative['value'])+q.iv.ln(2))),5e-10,cumulative)
    check('circuit_old_entropy_invariant',abs(old['value']-first['value']),5e-10)
    protected_stage2=q.swaps(bc,stage1,[(2,6),(3,7)])
    check('circuit_primary_protected_identical',np.max(abs(stage2-protected_stage2)),0.)
    save('circuit',initial=circuit,stage1=stage1,stage2=stage2,protected=protected_stage2)
    # Numerical F_01 realization on its full two-mode Fock space.
    bs=q.Basis.make(2)
    sh=np.diag([0.,1.,1.,2.]).astype(complex)
    sh[1,2]=sh[2,1]=-1
    sp=q.Spectral(sh)
    for mask in bs.states:
        ket=bs.ket([j for j in range(2) if mask>>j&1])
        got,ge=sp.apply(ket,np.pi/2)
        exact=q.swaps(bs,ket,[(0,1)])
        clock_allowance=q.upper(abs(q.I(float(np.pi/2))-q.iv.pi/2)*q.I(q.fnorm(sh)))
        check('numeric_fswap_'+str(mask),np.linalg.norm(got-exact),2e-11)
        check('numeric_fswap_bound_'+str(mask),q.upper(q.I(vector_target_error(got,[(q.I(float(z.real)),q.I(float(z.imag))) for z in exact]))-q.I(ge)-q.I(clock_allowance)),0.)
    # Static Bell, product and empty-block readouts.

    bell=(b4.ket([0,2])+b4.ket([1,3]))/np.sqrt(2)
    EA=ent_known('bell_A',b4,bell,[0,1],np.log(2))
    EAB=ent_known('bell_AB',b4,bell,[0,1,2,3],0.)
    cond=q.conditional(EAB,EA)
    check('negative_conditional',q.upper(abs(q.I(cond['value'])+q.iv.ln(2))),5e-10,cond)
    check('negative_conditional_enclosure',cond['upper'],0.)
    pos=q.conditional(EA,ent(b4,bell,[]))
    check('positive_conditional',q.upper(abs(q.I(pos['value'])-q.iv.ln(2))),5e-10,pos)
    zero=q.conditional(ent(b4,b4.ket([0,2]),[0,1,2,3]),ent(b4,b4.ket([0,2]),[0,1]))
    check('zero_conditional',abs(zero['value']),5e-10,zero)

    for name,ivalue in [('touch',{'value':0.,'lower':-1.,'upper':0.}),('overlap',{'value':0.,'lower':-1.,'upper':.1})]:
        r=q.purification_margin(ivalue,0.,0.,2,2)
        check('guard_'+name,0. if r['status']=='return_or_precision_unresolved' else 1.,0.,r)

    # Returning free particle: entropy fall without later arrival, guard fails.
    br=q.Basis.make(3,1); initial=(br.ket([0])+br.ket([1]))/np.sqrt(2)
    returning=np.array([[0,-1j,0],[1j,0,0],[0,0,0]],complex)
    ps=q.Spectral(returning); ks=q.Spectral(np.zeros((3,3),complex))
    returned,pe=ps.apply(initial,np.pi/4)
    check('return_entirely_C',abs(abs(returned[1])**2-1),2e-11)
    ent_known('return_initial_entropy',br,initial,[0],np.log(2))
    return_clock_error=q.upper(abs(q.I(float(np.pi/4))-q.iv.pi/4)*q.I(q.fnorm(returning)))
    er=ent_known('return_final_entropy',br,returned,[0],0.,q.upsum([pe,return_clock_error]))
    rc=q.conditional(ent(br,returned,[0,2],pe),er)
    check('return_empty_B_conditional',abs(rc['value']),5e-10)
    bound=q.midpoint_bound(returning,ks.h,ks,initial,np.pi/4,16)
    distance=np.linalg.norm(returned-initial)
    check('return_duhamel_contains_distance',distance-bound['upper'],0.,bound)

    zb=q.midpoint_bound(np.zeros((3,3)),ks.h,ks,initial,1.)
    check('zero_removed_term',zb['upper'],0.)
    # Lipschitz f(t)=|t-.5| on [0,1]: midpoint0, exact integral1/4.
    upper_lipschitz=q.lipschitz_midpoint(1.,0.,1.)
    check('lipschitz_midpoint_correction',abs(upper_lipschitz-.25),1e-15)
    save('return',initial=initial,primary=returned,protected=initial,bound=bound['upper'])
    # Production record pipeline for the exact two-collection circuit.
    circop=np.zeros((bc.dimension,bc.dimension),complex)
    circ_schedule=q.schedule_identity({'gates':[[[0,4],[1,5]],[[2,6],[3,7]]],
        'clocks':[1.,2.],'operator':q.array_identity(circop)})
    circ_initial_error=q.reduced(bc,circuit,[])['normalization_error']
    cplan=q.CollectionPlan.make(bc,[1.,2.],[[4,5],[6,7]])
    snaps={}
    for role in ('primary','protected'):
        for clock,vector in [(0.,circuit),(1.,stage1),(2.,stage2)]:
            snaps[(role,clock)]=q.Snapshot.make(bc,vector,'swap_circuit',role,circuit,circ_schedule,clock,circ_initial_error,circop)
    pc=q.conditional_at_clock(cplan,1,q.entropy_record(snaps['primary',2.],[4,5,6,7]),q.entropy_record(snaps['primary',2.],[4,5]))
    qc=q.conditional_at_clock(cplan,1,q.entropy_record(snaps['protected',2.],[4,5,6,7]),q.entropy_record(snaps['protected',2.],[4,5]))
    pchange=q.cumulative_change(cplan,1,pc['_union'],q.entropy_record(snaps['primary',1.],[4,5]))
    firstproof=q.verify_circuit_step('swap_circuit',[],[(0,4),(1,5)],circop,circop,snaps['primary',0.],snaps['primary',1.])
    proof=q.verify_circuit_step('swap_circuit',[[4,5]],[(2,6),(3,7)],circop,circop,snaps['primary',1.],snaps['primary',2.])
    if not np.array_equal(q.swaps(bc,stage1,proof['gates']),stage2): raise RuntimeError('Circuit gate record mismatch')
    ce=q.comparison_evidence(snaps['primary',2.],snaps['protected',2.],proof,firstproof)
    cert=q.certificate(cplan,1,pc,qc,ce,snaps['primary',1.])
    check('isolated_output_guard',0. if cert['status']=='bounded_return_excluded' else 1.,0.,cert)
    check('circuit_record_cumulative',q.upper(abs(q.I(pchange['value'])+q.iv.ln(2))),5e-10,pchange)
    # Production bindings reject real record substitutions through the helpers.
    for label,vector,clock,operator in [('clock',stage2,2.1,circop),
        ('state',stage1,2.,circop),('operator',stage2,2.,np.eye(bc.dimension,dtype=complex))]:
        badsnap=q.Snapshot.make(bc,vector,'swap_circuit','primary',circuit,circ_schedule,clock,circ_initial_error,operator)
        badold=q.entropy_record(badsnap,[4,5])
        expect_reject('conditional_actual_'+label,lambda badold=badold:q.conditional_at_clock(cplan,1,pc['_union'],badold))
    expect_reject('conditional_actual_partition',lambda:q.conditional_at_clock(cplan,1,pc['_union'],q.entropy_record(snaps['primary',2.],[6,7])))
    expect_reject('cumulative_actual_clock',lambda:q.cumulative_change(cplan,1,pc['_union'],q.entropy_record(snaps['primary',2.],[4,5])))
    expect_reject('later_disjoint_gate_recouples_old',lambda:q.verify_circuit_step('swap_circuit',[[4,5]],[(4,6),(3,7)],circop,circop))

    changed=dict(pc['_union']);changed['upper']=1.
    expect_reject('entropy_actual_interval_mutation',lambda:q.conditional_at_clock(cplan,1,changed,pc['_old']))
    changedproof=dict(proof);changedproof['end_clock']=1.5
    expect_reject('return_proof_clock_mutation',lambda:q.comparison_evidence(snaps['primary',2.],snaps['protected',2.],changedproof,firstproof))
    changed_evidence=dict(ce);changed_evidence['B']=.1
    expect_reject('return_bound_substitution',lambda:q.certificate(cplan,1,pc,qc,changed_evidence,snaps['primary',1.]))
    previous_changed=dict(ce);previous_changed['B_previous']=.1
    expect_reject('return_previous_bound_substitution',lambda:q.certificate(cplan,1,pc,qc,previous_changed,snaps['primary',1.]))
    wrong_prefix=q.verify_circuit_step('swap_circuit',[[6,7]],[(0,4),(1,5)],circop,circop,snaps['primary',0.],snaps['primary',1.])
    wrong_prefix_evidence=q.comparison_evidence(snaps['primary',2.],snaps['protected',2.],proof,wrong_prefix)
    expect_reject('return_earlier_protection_prefix',lambda:q.certificate(cplan,1,pc,qc,wrong_prefix_evidence,snaps['primary',1.]))
    badzero=q.Snapshot.make(bc,stage1,'swap_circuit','primary',circuit,circ_schedule,0.,circ_initial_error,circop)
    expect_reject('circuit_wrong_clock_zero_state',lambda:q.verify_circuit_step('swap_circuit',[],[(0,4),(1,5)],circop,circop,badzero,snaps['primary',1.]))
    alternative=q.swaps(bc,circuit,[(0,6),(1,7)])
    altone=q.Snapshot.make(bc,alternative,'swap_circuit','primary',circuit,circ_schedule,1.,circ_initial_error,circop)
    disconnected=q.verify_circuit_step('swap_circuit',[],[(0,6),(1,7)],circop,circop,snaps['primary',0.],altone)
    expect_reject('return_valid_disconnected_prefix',lambda:q.comparison_evidence(snaps['primary',2.],snaps['protected',2.],proof,disconnected))
    # Same production pipeline, now using the actual returning experiment.
    rt=[q.Term(1j,((1,True),(0,False))),q.Term(-1j,((0,True),(1,False)))]
    rh,rhe=q.assemble(br,rt);rk,rke=q.assemble(br,q.protect(rt,[[0]],3))
    rps=q.Spectral(rh.toarray(),rhe);rqs=q.Spectral(rk.toarray(),rke)
    rinit_error=q.reduced(br,initial,[])['normalization_error']
    rp,rpe=rps.apply(initial,np.pi/4,rinit_error);rq,rqe=rqs.apply(initial,np.pi/4,rinit_error)
    rschedule=q.schedule_identity({'primary':q.array_identity(rh.toarray()),'protected':q.array_identity(rk.toarray()),'clocks':[0.,float(np.pi/4)],'groups':[[0]]})
    rb,rproof=q.return_segment(br,'returning',rt,[[0]],rps,rqs,initial,np.pi/4,rinit_error,16,schedule=rschedule)
    rplan=q.CollectionPlan.make(br,[0.,float(np.pi/4)],[[0],[2]])
    rs={}
    for role,vector,error,operator in [('primary',rp,rpe,rps.h),('protected',rq,rqe,rqs.h)]:
        rs[role]=q.Snapshot.make(br,vector,'returning',role,initial,rschedule,np.pi/4,error,operator)
    rpc=q.conditional_at_clock(rplan,1,q.entropy_record(rs['primary'],[0,2]),q.entropy_record(rs['primary'],[0]))
    rqc=q.conditional_at_clock(rplan,1,q.entropy_record(rs['protected'],[0,2]),q.entropy_record(rs['protected'],[0]))
    rp0=q.Snapshot.make(br,initial,'returning','primary',initial,rschedule,0.,rinit_error,rps.h)
    rchange=q.cumulative_change(rplan,1,rpc['_union'],q.entropy_record(rp0,[0]))
    expect_reject('continuous_nonzero_start',lambda:q.return_segment(br,'returning',rt,[[0]],rps,rqs,initial,np.pi/4,rinit_error,16,schedule=rschedule,start_clock=1.))
    revidence=q.comparison_evidence(rs['primary'],rs['protected'],rproof)
    rcert=q.certificate(rplan,1,rpc,rqc,revidence,rp0)
    check('return_guard_fails',0. if rcert['status']=='return_or_precision_unresolved' else 1.,0.,rcert)
    check('return_actual_common_conditional',abs(rpc['value']),5e-10,rpc)
    check('return_actual_cumulative_fall',q.upper(abs(q.I(rchange['value'])+q.iv.ln(2))),5e-10,rchange)
    expect_reject('return_unrelated_conditional',lambda:q.certificate(rplan,1,rpc,qc,revidence,rp0))
    # Crossing quadratic/quartic removal and retained internal terms.
    cross=q.Term(1,((0,True),(2,False)))
    internal=q.Term(1,((0,True),(1,False)))
    testterms=[cross,cross.adjoint(),internal,internal.adjoint(),term,term.adjoint()]
    protected=q.protect(testterms,[[0,1]],4)
    check('protection_crossing_terms',0. if protected==[internal,internal.adjoint()] else 1.,0.)
    laterprotected=q.protect(testterms,[[0,1],[2,3]],4)
    lh,le=q.assemble(b4,laterprotected)
    injected=lh.toarray()+q.assemble(b4,[cross,cross.adjoint()])[0].toarray()
    expect_reject('later_disjoint_term_recouples_old',lambda:q.verify_protected_terms(b4,testterms,[[0,1],[2,3]],injected))
    expect_reject('protected_block_recoupling',lambda:q.protect(testterms,[[0,1],[1,2]],4))
    check('continuity_zero',q.continuity(0.,16),0.)
    check('continuity_d1',q.continuity(1.,1),0.)
    check('continuity_max',q.upper(abs(q.I(q.continuity(1.,4))-q.iv.ln(4))),5e-10)

    # Fixed-code register controls, exact fermionic swaps/permutations.
    permute=q.fermion_permute

    bre=q.Basis.make(12,2)
    code=np.stack([bre.ket([0,1]),bre.ket([2,3])])
    q.validate_code(bre,code)
    ref=code/np.sqrt(2)
    light_perm=list(range(12))
    for i in range(4): light_perm[i],light_perm[i+4]=i+4,i
    lightstate=permute(bre,ref,light_perm)
    alloutput=list(range(4,12)); lightoutput=list(range(4,8))
    for name,state,Ol,OA,expected_l,expected_a in [
        ('light',lightstate,lightoutput,alloutput,0.,0.),
        ('heavy',permute(bre,ref,list(range(8,12))+list(range(4,8))+list(range(4))),lightoutput,alloutput,.75,0.),
        ('retained',ref,lightoutput,alloutput,.75,.75)]:
        rsc=q.schedule_identity({'fixture':name,'clock':1.,'state':q.array_identity(state)})
        regplan=q.RegisterPlan.make(bre,'register_'+name,ref,rsc,[1.],Ol,OA)
        regsnap=q.Snapshot.make(bre,state,'register_'+name,'primary',ref,rsc,1.,q.reduced(bre,state,[])['normalization_error'],np.zeros((bre.dimension,bre.dimension),complex))
        dl=q.register_diagnostic(regplan,regsnap,'light'); da=q.register_diagnostic(regplan,regsnap,'all')
        for label,d,expected in [('light',dl,expected_l),('all',da,expected_a)]:
            check(f'register_{name}_{label}',abs(d['value']-expected),5e-10,d)
            check(f'register_enclosure_{name}_{label}',interval_violation(d['lower'],d['upper'],q.I(expected)),0.)
        # I(R:O)+I(R:K)=2S(R), so S(O)+S(K)-S(RO)-S(RK)=0.
        Sr=q.entropy(q.reduced(bre,state,[],True))
        so=ent(bre,state,OA); sk=ent(bre,state,[i for i in range(12) if i not in OA])
        sro=q.entropy(q.reduced(bre,state,OA,True)); srk=q.entropy(q.reduced(bre,state,[i for i in range(12) if i not in OA],True))
        check(f'register_pure_accounting_{name}',abs(so['value']+sk['value']-sro['value']-srk['value']),5e-10)
        save('register_'+name,state=state,rho_RK_light=dl['joint']['matrix'],rho_RK_all=da['joint']['matrix'])
    expect_reject('register_plan_invalid_code',lambda:q.RegisterPlan.make(bre,'bad',np.stack([code[0],code[0]])/np.sqrt(2),rsc,[1.],lightoutput,alloutput))
    expect_reject('register_plan_unequal_reference',lambda:q.RegisterPlan.make(bre,'bad',np.stack([np.sqrt(.75)*code[0],.5*code[1]]),rsc,[1.],lightoutput,alloutput))
    expect_reject('register_actual_output_role',lambda:q.register_diagnostic(regplan,regsnap,'light',OA))
    shifted=q.Snapshot.make(bre,ref,'register_retained','primary',ref,rsc,1.1,0.,np.zeros((bre.dimension,bre.dimension),complex))
    expect_reject('register_actual_clock',lambda:q.register_diagnostic(regplan,shifted,'light'))
    # Correctly tests R versus output (opposite complement) as the positive control.
    dro=q.decoupling(bre,lightstate,[i for i in range(12) if i not in lightoutput])
    check('register_R_output_positive',abs(dro['value']-.75),5e-10)
    odd=q.Basis.make(6,1); oddcode=np.stack([odd.ket([0]),odd.ket([1])]);q.validate_code(odd,oddcode)
    ho,eo=q.assemble(odd,pair_terms); op=q.Spectral(ho.toarray(),eo)
    for t in clocks:
        oddstate,oe=op.apply(oddcode/np.sqrt(2),t)
        for name,output in [('light',[2,3]),('all',[2,3,4,5])]:
            od=q.decoupling(odd,oddstate,output,oe)
            check(f'register_odd_dark_{name}_{t}',abs(od['value']-.75),5e-10,od)
    expect_reject('invalid_code_rank',lambda:q.validate_code(bre,np.stack([code[0],code[0]])))
    full=q.Basis.make(2)
    expect_reject('invalid_code_parity',lambda:q.validate_code(full,np.stack([full.ket([]),full.ket([0])])) )
    expect_reject('duplicate_output_schedule',lambda:q.validate_schedule(b4,[0.,1.],[[0,1],[1,2]]))
    expect_reject('clock_order',lambda:q.validate_schedule(b4,[1.,0.],[[0],[1]]))
    expect_reject('missing_complement',lambda:q.validate_partition(4,[0],[1,2]))
    expect_reject('overlapping_complement',lambda:q.validate_partition(4,[0],[0,1,2,3]))
    expect_reject('negative_state_error',lambda:prop.apply(start,1.,-1.))
    expect_reject('zero_propagation_input',lambda:prop.apply(np.zeros(2),1.))
    expect_reject('negative_entropy_error',lambda:q.entropy(q.reduced(b4,HH,[0]),-1.))
    expect_reject('negative_decoupling_error',lambda:q.decoupling(bre,ref,alloutput,-1.))
    expect_reject('zero_norm',lambda:q.reduced(b4,np.zeros(16),[0]))
    bad=np.zeros(16); bad[0]=np.nan
    expect_reject('nonfinite_state',lambda:q.reduced(b4,bad,[0]))
    expect_reject('small_negative_without_allowance',lambda:q.entropy({'matrix':np.diag([1+1e-10,-1e-10]).astype(complex),'assembly_error':0.}))
    expect_reject('gross_density_negative',lambda:q.entropy({'matrix':np.diag([1.1,-.1]).astype(complex),'assembly_error':0.}))
    small=q.entropy({'matrix':np.diag([1+1e-14,-1e-14]).astype(complex),'assembly_error':2e-14})
    check('tiny_negative_paid',0. if small['negative_mass']>0 and small['numerical_margin']>0 else 1.,0.,small)
    expect_reject('nonhermitian_density',lambda:q.entropy({'matrix':np.array([[1,1],[0,0]],complex),'assembly_error':0.}))
    expect_reject('nonhermitian_H',lambda:q.Spectral(np.array([[0,1],[0,0]],complex)))
    expect_reject('nonorthogonal_eigenvectors',lambda:q.Spectral(H,eigenvectors=np.ones((2,2)),eigenvalues=np.array([1.,-1.])))
    expect_reject('complex_supplied_eigenvalues',lambda:q.Spectral(H,eigenvectors=np.eye(2),eigenvalues=np.array([1+1j,-1.])))
    expect_reject('bad_eigen_residual',lambda:q.Spectral(H,eigenvectors=np.eye(2),eigenvalues=np.zeros(2)))
    expected={'basis':'fixture','state':'initial','hamiltonian':'free','clock':1.,
        'light_output':(2,3),'all_output':(2,3,4,5)}
    for field,value in [('clock',1.1),('state','other'),('hamiltonian','other'),
                        ('light_output',(2,3,4,5)),('all_output',(2,3))]:
        changed=dict(expected);changed[field]=value
        expect_reject('association_'+field,lambda changed=changed:q.validate_association(expected,changed))
    expect_reject('mock_without_gate',lambda:q.axial_mock())
    expect_reject('mock_with_gate',lambda:q.axial_mock(gate={'pass':True}))

    # Two predeclared free-control cost sizes; no interacting mock propagation.
    for sites in (2,3):
        wall0=time.perf_counter(); cpu0=time.process_time()
        begin=time.perf_counter(); begin_cpu=time.process_time()
        basis=q.Basis.make(8*sites,2)
        h=q.free_one_body(sites,1.,1.,np.ones(sites),np.zeros(sites))
        q.verify_dyadic_free_fixture(h,sites,1.,1.,np.ones(sites),np.zeros(sites))
        sparse,he=q.assemble(basis,q.quadratic_terms(h)); dense=sparse.toarray()
        timings={'basis_assembly_wall':time.perf_counter()-begin,'basis_assembly_cpu':time.process_time()-begin_cpu}
        begin=time.perf_counter(); begin_cpu=time.process_time(); solver=q.Spectral(dense,he)
        orbital_solver=q.Spectral(h)
        timings['eigensystem_wall']=time.perf_counter()-begin
        timings['eigensystem_cpu']=time.process_time()-begin_cpu
        initial=basis.ket([0,1]); ts=np.linspace(0,1,17)
        states=[]; errors=[]; independent=[]; entropies=[]; core_entropies=[]; state_budgets=[]; independent_budgets=[]
        timing_prop=0.; timing_red=0.; timing_prop_cpu=0.; timing_red_cpu=0.;independent_wall=0.;independent_cpu=0.; max_state=0.; max_norm=0.; max_energy=0.; max_nl=0.
        energies=[]; counts=[]; mode_counts=[]; light_entropies=[]
        for t in ts:
            begin=time.perf_counter(); begin_cpu=time.process_time(); psi,e=solver.apply(initial,float(t))
            state_budgets.append(dict(solver.last_budget))
            timing_prop+=time.perf_counter()-begin
            timing_prop_cpu+=time.process_time()-begin_cpu
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
        timings['propagation_wall']=timing_prop;timings['reduction_entropy_wall']=timing_red
        timings['propagation_cpu']=timing_prop_cpu;timings['reduction_entropy_cpu']=timing_red_cpu
        timings['independent_orbital_wedge_wall']=independent_wall;timings['independent_orbital_wedge_cpu']=independent_cpu
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
        timings['register_inclusive_wall']=time.perf_counter()-begin
        timings['register_inclusive_cpu']=time.process_time()-begin_cpu
        begin=time.perf_counter(); begin_cpu=time.process_time()
        terms=q.quadratic_terms(h);protected=q.protect(terms,[list(range(8,basis.modes))],basis.modes)
        kh,ke=q.assemble(basis,protected);K=kh.toarray(); kp=q.Spectral(K,ke)
        ret,retproof=q.return_segment(basis,f'cost_{sites}_return',terms,[list(range(8,basis.modes))],solver,kp,initial,1.,0.,16,schedule=q.schedule_identity({'operator':q.array_identity(dense),'protected':q.array_identity(K),'start':0.,'end':1.}))
        timings['return_comparator_wall']=time.perf_counter()-begin
        timings['return_comparator_cpu']=time.process_time()-begin_cpu
        cost={'sites':sites,'modes':basis.modes,'number':2,'dimension':basis.dimension,
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
        save(f'cost_{sites}',states=states,independent=independent,clocks=ts,
             state_errors=errors,energies=energies,light_counts=counts,
             registered=registered_states,occupations=mode_counts,hamiltonian=dense,one_body=h,protected=K,
             basis=np.array(basis.states,dtype=np.uint64),eigenvalues=solver.values,eigenvectors=solver.vectors,
             onebody_eigenvalues=orbital_solver.values,onebody_eigenvectors=orbital_solver.vectors)
    for path,digest in identities.items():
        check('source_unchanged_'+Path(path).name,0. if hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest else 1.,0.)
    env['threads_final']=len(list(Path('/proc/self/task').iterdir()))
    check('resident_threads_final',abs(env['threads_final']-1),0.)
    result={'scope':'manufactured self-tests only; no axial mock run','runtime':env,
        'intake':intake,'identities':identities,
        'controls':rows,'costs':costs,'all_passed':all(r['pass'] for r in rows),
        'total_wall':time.perf_counter()-started}
    with (out/'controls.json').open('x') as f: json.dump(clean(result),f,indent=2,allow_nan=False)
    with (out/'raw_arrays.npz').open('xb') as f: np.savez_compressed(f,**arrays)
    journal.close();action_journal.close();q.set_recorder(None)
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
        'Every action, including free/pair/swap/inverse and return quadrature, is retained in numerical_actions.ndjson.',
        '| Action | Time | Input | Spectral defect | Residual | Assembly | Coefficient action | Phase enclosure | Phase multiply | Final action | Total |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i,a in enumerate(action_summaries):
        keys=('represented_time','input','spectral_defect','eigen_residual','operator_assembly','coefficient_action','phase_approximation','phase_multiplication','final_action','total')
        lines.append('| '+str(i)+' | '+' | '.join(str(a[k]) for k in keys)+' |')
    for c in costs:
        lines+=['',f"### Size {c['sites']}: reference readout intervals",'',
            '| Clock | Light D [lower,upper] | Complete D [lower,upper] | S(R) | Light S(O),S(K) | Complete S(O),S(K) | State / independent error |',
            '|---:|---|---|---:|---|---|---|']
        for r in c['register_diagnostics']:
            lines.append('| '+str(r['clock'])+' | '+str([r['light']['lower'],r['light']['upper']])+' | '+
                str([r['all']['lower'],r['all']['upper']])+' | '+str(r['register_entropy']['value'])+' | '+
                str([r['entropies']['light'][k]['value'] for k in ('output','complement')])+' | '+
                str([r['entropies']['all'][k]['value'] for k in ('output','complement')])+' | '+
                str([r['state_error'],r['independent_state_error']])+' |')
        lines+=['', 'Return-helper upper bound: '+str(c['return_comparator']['upper'])+
            '; Lipschitz upper: '+str(c['return_comparator']['lipschitz'])+
            '; difference-operator allowance: '+str(c['return_comparator']['difference_operator_error'])+'.']
        lines+=['',f"## {c['sites']} sites, {c['modes']} modes, sector dimension {c['dimension']}",
                json.dumps(clean({k:v for k,v in c.items() if k not in ('register_diagnostics','return_comparator','return_protection','entropy_diagnostics','light_entropy_diagnostics','core_entropy_diagnostics','state_budgets','independent_budgets')}),indent=2)]
    (out/'written_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(out),'controls':len(rows),'passed':sum(r['pass'] for r in rows),
        'all_passed':result['all_passed'],'total_wall':result['total_wall']}))
    if not result['all_passed']: raise SystemExit(1)

if __name__=='__main__':
    main()
