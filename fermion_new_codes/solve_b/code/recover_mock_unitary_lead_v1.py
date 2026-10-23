#!/usr/bin/env python3
"""Box-only missing readouts from the interrupted attempt; never propagate."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import os
import sys
from types import SimpleNamespace

CONTRACT='d78e9ee13c1f03c1cbb774e6c02ee73cb5486b05cd6f7a011460ce1ae3229b62'
PRESERVATION='497a45e799a7333580d90cdc91de7e12cfff444a821b766d88b955dc6e07976f'
SPEC='495e59d68d57061246eb82dfe76ca26bceede021d29c1ba9fffcd63024184b6c'
ADDENDUM='02aed9d4e45d5b1ec359a096338039803fc61f38f66536a29922e372f20b1446'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_digest(path,expected):
    if digest(path)!=expected: raise ValueError('Preserved file/source identity mismatch: '+str(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('run','preservation','output','spec','authorchecks','continuation-contract'):
        p.add_argument('--'+flag,required=True)
    args=p.parse_args();root=Path(args.run);out=Path(args.output)
    if out.exists(): raise FileExistsError('Refusing to overwrite recovery output')
    for path,sha in [(args.preservation,PRESERVATION),(args.spec,SPEC),(args.authorchecks,ADDENDUM),(args.continuation_contract,CONTRACT)]:
        require_digest(path,sha)
    frozen=json.loads(Path(args.preservation).read_text())['files']
    for path,meta in frozen.items(): require_digest(path,meta['sha256'])
    if str(root/'intake.json') not in frozen: raise ValueError('Unbound original run')
    from cpu_runtime import require_remote_cpu
    runtime=require_remote_cpu(out)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
        if os.environ.get(name)!='1':raise ValueError('One thread required')
    import numpy as np
    import mock_unitary_lead_v1 as q
    import ctypes
    import scipy
    import mpmath
    if ctypes.CDLL(None).fegetround()!=0 or np.finfo(float).eps!=2**-52 or np.finfo(float).smallest_subnormal!=2**-1074:
        raise ValueError('Required IEEE binary64 round-to-nearest model')
    if np.float64(np.finfo(float).tiny)*np.float64(.5)==0:
        raise ValueError('Gradual underflow required')
    if len(list(Path('/proc/self/task').iterdir()))!=1:raise ValueError('One resident thread required')
    intake={'runtime':runtime,'command':sys.argv,'scope':'saved-data readout only; no propagation or propagation-eigensystem construction',
        'rounding':'FE_TONEAREST','gradual_underflow':True,'versions':{'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mpmath.__version__},
        'identities':{str(path):digest(path) for path in [Path(__file__),Path(q.__file__),Path(__file__).with_name('cpu_runtime.py'),
            Path(args.spec),Path(args.authorchecks),Path(args.continuation_contract),Path(args.preservation)]}}
    with (out/'intake.json').open('x') as f:json.dump(intake,f,indent=2)
    def clean(x):
        if isinstance(x,np.ndarray):return clean({'real':x.real.tolist(),'imag':x.imag.tolist()}) if np.iscomplexobj(x) else clean(x.tolist())
        if isinstance(x,complex):return clean({'real':x.real,'imag':x.imag})
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items() if not str(k).startswith('_')}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,np.generic):return clean(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    journal=(out/'recovery.ndjson').open('x');checks=[];results=[]
    def emit(row):journal.write(json.dumps(clean(row),allow_nan=False)+'\n');journal.flush()
    def check(name,passed,detail=None):
        row={'name':name,'pass':bool(passed),'scope':'new recovery association/consistency','detail':detail}
        checks.append(row);emit(row)
        if not passed:raise ValueError('Recovery check failed: '+name)
    def reject(name,call):
        try:call()
        except (ValueError,RuntimeError) as e:check(name,True,str(e))
        else:check(name,False)
    def decode(z):
        if not isinstance(z,dict) or set(z)!= {'real','imag'}:raise ValueError('Complex array encoding')
        a,b=np.asarray(z['real'],float),np.asarray(z['imag'],float)
        if a.shape!=b.shape:raise ValueError('Complex component shape')
        z=np.empty(a.shape,dtype=np.complex128);z.real=a;z.imag=b
        return q.finite(z)
    records=[json.loads(line) for line in (root/'numerical_actions.ndjson').read_text().splitlines()]
    operators=[r for r in records if r['kind']=='operator'];actions=[r for r in records if r['kind']=='action']
    if len(operators)!=13 or len(actions)!=230:raise ValueError('Original execution roster mismatch')
    def bound_action(ordinal,operator,initial,clock,output=None,error=None,role='ordinary',action_roster=None):
        roster=actions if action_roster is None else action_roster
        if not 0<=ordinal<len(roster):raise ValueError('Missing ordinal')
        a=roster[ordinal];x,y=decode(a['input_state']),decode(a['output_state'])
        if role=='register':
            if x.ndim!=2 or x.shape[0]!=2 or y.shape!=x.shape:raise ValueError('Registered action shape/role mismatch')
        elif role in ('ordinary','orbital','midpoint'):
            if x.ndim!=1 or y.shape!=x.shape:raise ValueError('Ordinary/orbital action shape/role mismatch')
        else:raise ValueError('Unrecognized saved action role')
        if (a['operator']!=q.array_identity(operator) or q.array_identity(x)!=q.array_identity(initial) or
            a['budget']['represented_time']!=clock):raise ValueError('Operator/input/clock action association')
        if output is not None and q.array_identity(y)!=q.array_identity(output):raise ValueError('Saved state/action output association')
        if error is not None and a['budget']['total']!=error:raise ValueError('Saved error/action association')
        return {'ordinal':ordinal,'stage':'return_midpoint' if role=='midpoint' else role,'clock':clock,'operator':a['operator'],
            'initial':q.array_identity(x),'output':y,'output_identity':q.array_identity(y),
            'error':a['budget']['total'],'input_error':a['budget']['input']}
    class SavedAdapter:
        def __init__(self,h,assembly_error,initial,roster,expected,stage='return_midpoint'):
            self.h=h;self.assembly_error=assembly_error;self.initial=q.array_identity(initial);self.expected=expected;self.used=[]
            if len(roster)!=len(expected):raise ValueError('Incomplete or extra saved-action roster')
            self.roster=[]
            for r,e in zip(roster,expected):
                if (r['ordinal']!=e['ordinal'] or r['clock']!=e['clock'] or r['stage']!=stage or
                    r['operator']!=q.array_identity(h) or r['initial']!=self.initial or
                    r['output_identity']!=q.array_identity(r['output']) or r['output_identity']!=e['output_identity'] or
                    r['error']!=e['error'] or r['input_error']!=e['input_error']):
                    raise ValueError('Saved action stage/ordinal/clock/operator/state/budget mismatch')
                self.roster.append(r)
        def apply(self,initial,time,input_error=0.):
            index=len(self.used)
            if index>=len(self.roster):raise ValueError('Saved action requested twice/extra')
            r=self.roster[index]
            if q.array_identity(initial)!=self.initial or time!=r['clock'] or input_error!=r['input_error']:
                raise ValueError('Saved adapter called with a different initial/clock/error')
            self.used.append(r['ordinal']);return r['output'].copy(),r['error']
    signed=np.empty(2,dtype=np.complex128);signed.real=[-0.,0.];signed.imag=[0.,-0.]
    check('exact_complex_json_round_trip',q.array_identity(decode(json.loads(json.dumps(clean(signed)))))==q.array_identity(signed))
    # Tiny typed-index and actual proof round-trip guard, without any propagation.
    b=q.Basis.make(2,1);hh=np.array([[0.,1.],[1.,0.]],complex);tiny_terms=q.quadratic_terms(hh)
    literal=[q.Term(1+0j,((0,True),(1,False))),q.Term(1+0j,((1,True),(0,False)))]
    check('native_integer_labels',tiny_terms==literal and all(type(i) is int for t in tiny_terms for i,_ in t.operators))
    lifted,he=q.assemble(b,tiny_terms);check('unchanged_tiny_lift',np.array_equal(lifted.toarray(),hh))
    zero=np.zeros((2,2),complex);initial=b.ket([0]);times=[(2*j+1)/32 for j in range(16)]
    tiny=[{'ordinal':j,'stage':'return_midpoint','clock':t,'operator':q.array_identity(zero),
        'initial':q.array_identity(initial),'output':initial.copy(),'output_identity':q.array_identity(initial),'error':0.,'input_error':0.} for j,t in enumerate(times)]
    expected=[dict(r) for r in tiny]
    adapter=SavedAdapter(zero,0.,initial,tiny,expected)
    _,proof=q.return_segment(b,'typed_fixture',tiny_terms,[[0]],SimpleNamespace(h=hh,assembly_error=he),adapter,initial,1.,0.,16,schedule='typed_fixture')
    check('proof_json_round_trip',q._verify_proof(json.loads(json.dumps(proof))))
    changed=json.loads(json.dumps(proof));changed['segment_bound']+=.1
    reject('proof_bound_mutation',lambda:q._verify_proof(changed))
    changed=json.loads(json.dumps(proof));changed['primary_terms'][0][2][0][0]=1
    reject('proof_mode_mutation',lambda:q._verify_proof(changed))
    reject('missing_midpoint',lambda:SavedAdapter(zero,0.,initial,tiny[:-1],expected))
    duplicated=[dict(r) for r in tiny];duplicated[1]=dict(duplicated[0])
    reject('duplicate_midpoint',lambda:SavedAdapter(zero,0.,initial,duplicated,expected))
    for field,value in [('clock',.123),('operator',q.array_identity(hh)),('stage','ordinary')]:
        bad=[dict(r) for r in tiny];bad[0][field]=value
        reject('wrong_'+field,lambda bad=bad:SavedAdapter(zero,0.,initial,bad,expected))
    bad=[dict(r) for r in tiny];bad[0]['output']=b.ket([1])
    reject('same_shape_wrong_state',lambda:SavedAdapter(zero,0.,initial,bad,expected))
    reject('changed_original_hash',lambda:require_digest(root/'intake.json','0'*64))
    # Test the actual ordinary/register file-action join, separately from adapters.
    fake={'operator':q.array_identity(zero),'input_state':clean(initial),
        'output_state':clean(initial),'budget':{'represented_time':0.,'total':0.,'input':0.}}
    check('tiny_actual_ordinary_join',bound_action(0,zero,initial,0.,initial,0.,action_roster=[fake])['clock']==0.)
    for name,op,clock,output,error in [('clock',zero,.1,initial,0.),('operator',hh,0.,initial,0.),
        ('output',zero,0.,b.ket([1]),0.),('error',zero,0.,initial,.1)]:
        reject('actual_join_wrong_'+name,lambda op=op,clock=clock,output=output,error=error:
            bound_action(0,op,initial,clock,output,error,action_roster=[fake]))
    reject('actual_join_missing_ordinal',lambda:bound_action(1,zero,initial,0.,action_roster=[fake]))
    reject('actual_join_duplicate_clock',lambda:bound_action(1,zero,initial,.5,action_roster=[fake,fake]))
    reject('ordinary_substituted_as_register',lambda:bound_action(0,zero,initial,0.,role='register',action_roster=[fake]))
    reg=np.stack([b.ket([0]),b.ket([1])])/np.sqrt(2)
    fake_reg={'operator':q.array_identity(zero),'input_state':clean(reg),'output_state':clean(reg),
        'budget':{'represented_time':0.,'total':0.,'input':0.}}
    check('tiny_actual_register_join',bound_action(0,zero,reg,0.,reg,0.,role='register',action_roster=[fake_reg])['clock']==0.)
    reject('register_substituted_as_ordinary',lambda:bound_action(0,zero,reg,0.,role='ordinary',action_roster=[fake_reg]))
    # Frozen original stage ordinals: prefix78, ordinary51, register85, midpoint16.
    basis=q.Basis.make(16,2);initial=basis.ket([0,1]);code=np.stack([basis.ket([0,1]),basis.ket([2,3])]);ref=code/np.sqrt(2)
    H=decode(operators[10]['hamiltonian']);h=decode(operators[11]['hamiltonian']);K=decode(operators[12]['hamiltonian'])
    if H.shape!=(120,120) or h.shape!=(16,16) or K.shape!=(120,120):raise ValueError('Preserved operator stage roster')
    clocks=[j/16 for j in range(17)]
    def load_clock(kind,t):
        path=root/f'arrays_cost_2_{kind}clock_{float(t)}.npz'
        if str(path) not in frozen:raise ValueError('Unbound clock array')
        require_digest(path,frozen[str(path)]['sha256'])
        with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}
    def number(z):
        z=np.asarray(z)
        if z.shape!=():raise ValueError('Scalar saved budget required')
        return q.error_allowance(float(z))
    for j,t in enumerate(clocks):
        d=load_clock('',t);psi=q.finite(d['state']);e=number(d['state_error'])
        if psi.shape!=(120,):raise ValueError('Ordinary/register state role mismatch')
        bound_action(78+3*j,H,initial,t,psi,e)
        orbitals=q.finite(d['orbitals']);errs=np.asarray(d['orbital_errors'])
        if orbitals.shape!=(16,2) or errs.shape!=(2,):raise ValueError('Ordinary orbital roster')
        for k in range(2):bound_action(79+3*j+k,h,np.eye(16,dtype=complex)[k],t,orbitals[:,k],float(errs[k]),role='orbital')
        wedge,we=q.wedge_state(basis,orbitals[:,0],orbitals[:,1],float(errs[0]),float(errs[1]))
        check('ordinary_saved_wedge_'+str(t),np.array_equal(wedge,d['determinant']) and we==number(d['independent_error']))
        entropies={name:q.entropy(q.reduced(basis,psi,modes),e) for name,modes in
                   [('lead',list(range(8,16))),('core',list(range(8))),('light',[4,5,6,7,12,13,14,15])]}
        result={'role':'recovered_original_readout','kind':'ordinary','clock':t,'ordinal':78+3*j,'state_error':e,'entropies':entropies}
        results.append(result);emit(result)
    schedule=q.schedule_identity({'operator':q.array_identity(H),'clocks':clocks,'code':[[0,1],[2,3]]})
    plan=q.RegisterPlan.make(basis,'cost_2_register',ref,schedule,clocks,list(range(12,16)),list(range(8,16)))
    for j,t in enumerate(clocks):
        d=load_clock('register_',t);psi=q.finite(d['state']);e=number(d['state_error'])
        if psi.shape!=(2,120):raise ValueError('Register/ordinary state role mismatch')
        bound_action(129+5*j,H,ref,t,psi,e,role='register')
        orbitals=q.finite(d['orbitals']);errs=np.asarray(d['orbital_errors'])
        if orbitals.shape!=(4,16) or errs.shape!=(4,):raise ValueError('Registered orbital roster')
        for k in range(4):bound_action(130+5*j+k,h,np.eye(16,dtype=complex)[k],t,orbitals[k],float(errs[k]),role='orbital')
        branches=[];branch_errors=[]
        for k in (0,2):
            w,we=q.wedge_state(basis,orbitals[k],orbitals[k+1],float(errs[k]),float(errs[k+1]));branches.append(w);branch_errors.append(we)
        branches=np.stack(branches);weight=float(1/np.sqrt(2));wedge=weight*branches
        we=q.upper(q.iv.sqrt(sum((q.I(x)**2 for x in branch_errors),q.I(0)))/q.iv.sqrt(2)+
            abs(q.I(weight)-1/q.iv.sqrt(2))*q.I(q.fnorm(branches))+q.I(q.real_product_round(wedge)))
        check('registered_saved_wedge_'+str(t),np.array_equal(wedge,d['wedge']) and we==number(d['independent_error']))
        snap=q.Snapshot.make(basis,psi,'cost_2_register','primary',ref,schedule,t,e,H)
        readouts={kind:q.register_diagnostic(plan,snap,kind) for kind in ('light','all')}
        entropies={}
        for kind,r in readouts.items():
            entropies[kind]={'output':q.entropy(q.reduced(basis,psi,r['output']),e),
                'complement':q.entropy(q.reduced(basis,psi,r['complement']),e),
                'register_output':q.entropy(q.reduced(basis,psi,r['output'],True),e),
                'register_complement':q.entropy(r['joint'],e)}
        accounting={kind:{'value':e0['output']['value']+e0['complement']['value']-e0['register_output']['value']-e0['register_complement']['value'],
            'lower':q.lower(q.I(e0['output']['lower'])+q.I(e0['complement']['lower'])-q.I(e0['register_output']['upper'])-q.I(e0['register_complement']['upper'])),
            'upper':q.upper(q.I(e0['output']['upper'])+q.I(e0['complement']['upper'])-q.I(e0['register_output']['lower'])-q.I(e0['register_complement']['lower']))} for kind,e0 in entropies.items()}
        result={'role':'recovered_original_readout','kind':'register','clock':t,'ordinal':129+5*j,'state_error':e,
            'readouts':readouts,'entropies':entropies,'accounting':accounting,'register_entropy':q.entropy(q.reduced(basis,psi,[],True),e)}
        results.append(result);emit(result)
    midpoint=[bound_action(214+j,K,initial,t,role='midpoint') for j,t in enumerate(times)]
    expected=[dict(r) for r in midpoint]
    adapter=SavedAdapter(K,float(operators[12]['assembly_error']),initial,midpoint,expected)
    primary=SimpleNamespace(h=H,assembly_error=float(operators[10]['assembly_error']))
    primary_terms=q.quadratic_terms(h)
    # Build the many-body primary from the retained one-body matrix; no eigensolve.
    lifted,allowance=q.assemble(basis,primary_terms)
    check('retained_primary_matrix_association',np.array_equal(lifted.toarray(),H) and allowance==primary.assembly_error)
    return_schedule=q.schedule_identity({'operator':q.array_identity(H),'protected':q.array_identity(K),'start':0.,'end':1.})
    rb,proof=q.return_segment(basis,'recovered_cost_2_return',primary_terms,[list(range(8,16))],primary,adapter,initial,1.,0.,16,schedule=return_schedule)
    check('all_sixteen_original_midpoints_consumed',adapter.used==list(range(214,230)))
    emit({'role':'recovered_missing_return_readout','bound':rb,'proof':proof,'source_ordinals':adapter.used})
    for path,meta in frozen.items():require_digest(path,meta['sha256'])
    for path,sha in intake['identities'].items():require_digest(path,sha)
    final_threads=len(list(Path('/proc/self/task').iterdir()));check('recovery_final_one_thread',final_threads==1)
    result={'scope':'recovered missing dictionaries only; original timings/final checks unavailable','intake':intake,
        'checks':checks,'readouts':results,'return_bound':rb,'return_proof':proof,'complete':True,
        'original_passed_assertions':547,'new_original_assertions':0,'final_threads':final_threads}
    with (out/'recovered_readouts.json').open('x') as f:json.dump(clean(result),f,indent=2,allow_nan=False)
    journal.close()
    lines=['# Recovered missing readouts','',
        'No propagation or propagation-eigensystem construction. Original547 passed predicates remain original evidence.',
        'Original first-size timing and final process/source checks remain unavailable. This process reports only its own final checks.',
        '',json.dumps(intake,indent=2),'',f'{len(checks)}/{len(checks)} new association/consistency guards passed.','',
        '| Clock | Lead S [lo,hi] | Core S [lo,hi] | Light S [lo,hi] |','|---:|---|---|---|']
    for r in results:
        if r['kind']=='ordinary':
            lines.append('| '+str(r['clock'])+' | '+' | '.join(str([r['entropies'][k]['lower'],r['entropies'][k]['upper']]) for k in ('lead','core','light'))+' |')
    lines+=['','| Clock | Light D [lo,hi] | Complete D [lo,hi] | Register S | State error |','|---:|---|---|---:|---:|']
    for r in results:
        if r['kind']=='register':
            lines.append('| '+str(r['clock'])+' | '+' | '.join(str([r['readouts'][k]['lower'],r['readouts'][k]['upper']]) for k in ('light','all'))+' | '+str(r['register_entropy']['value'])+' | '+str(r['state_error'])+' |')
    lines+=['','| Clock/role | S(O) interval | S(K) interval | S(RO) interval | S(RK) interval | Accounting interval |',
        '|---|---|---|---|---|---|']
    for r in results:
        if r['kind']=='register':
            for kind in ('light','all'):
                e0=r['entropies'][kind];account=r['accounting'][kind]
                lines.append('| '+str(r['clock'])+'/'+kind+' | '+' | '.join(str([e0[k]['lower'],e0[k]['upper']]) for k in ('output','complement','register_output','register_complement'))+' | '+str([account['lower'],account['upper']])+' |')
    lines+=['','Recovered original return-helper bound: '+str(rb['upper'])+'; Lipschitz upper '+str(rb['lipschitz'])+'.','',
        '## New recovery guards','']+[r['name']+': '+str(r['pass']) for r in checks]
    (out/'written_recovery_summary.md').write_text('\n'.join(lines)+'\n')
    print('Recovered missing readouts; no new trajectories or original assertions.')

if __name__=='__main__':main()
