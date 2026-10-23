#!/usr/bin/env python3
"""Box-only exact filtered static admission. No propagation entry point."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import argparse
import hashlib
import math
import json
import time
from types import MappingProxyType
import run_mock_vertex_admission_v1 as e
import mock_filtered_kernel_v1 as kernel
import mock_filtered_vertex_v1 as v

SPEC_SHA='a4ef1500a0af2fb37872e18893e8f805e1074d51b2eb01505b9fb40aec13a867'
FROZEN={
 'mock_unitary_lead_v1.py':'6fe256e6940c35db8ebfcdce8e3e2228d8ed2361a2b37b0728d799b00133e21a',
 'mock_axial_vertex_adapter_v1.py':'482b1b7a3a7d4600d06cdd5cc6030d63cdc22d30876006ac0214a712da579e7b',
 'run_mock_vertex_admission_v1.py':'6f74f133d01070c25c32c96b5231e78f6840c216cb9d0ef94f8b925c6a148265',
 'mock_unitary_lead_records_v1.py':'a18f1461ce08b2ccfbeedb57a15deddb8f430628c441bf308c5125edb3f04536',
 'cpu_runtime.py':'207c7a16432e722fa0b77c1e9d05e0d2ef712b540a95f25762bcb13cd10feb85',
 'ReviewA_fctl_v1.md':'967c11c1909ff34fce665e2cae5e3dd33547a0879ade7178f728709d7481ac07',
 'ReviewB_fctl_v1.md':'8920df798d880efec8310eac0eccf1e2078e3037773e1fcb57f13392e96445f9',
 'VERTEX_gate_filtered_v1.md':'2ed43ffb2adeb3e3d5eb05a11d138793502f2b05741bc8b23cd4b4e1ba6ddd69'}


def verify_frozen(code,contracts):
    for name,digest in FROZEN.items():
        path=(contracts if name.endswith('.md') else code)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('frozen source/contract identity '+name)


def validate_kernel_export(G,export,budget):
    if set(export)!=set(G):raise ValueError('kernel export keys')
    error=sum((abs(export[r]-G[r]) for r in G),F(0))
    if budget<0 or error>budget:raise ValueError('kernel export error budget')
    return error


def control_gate(directory,review,review_sha,sources):
    directory=Path(directory);review=Path(review)
    if hashlib.sha256(review.read_bytes()).hexdigest()!=review_sha:raise PermissionError('sole review identity')
    with (directory/'records.ndjson').open() as f:records=[json.loads(line) for line in f]
    if not records or records[-1]['kind']!='termination':raise PermissionError('unfinished controls')
    end=records[-1]['data']
    if end['status']!='complete' or not end['all_pass'] or not end['sources_unchanged'] or end['threads']!=1:raise PermissionError('failed controls/runtime')
    complete=[row['data'] for row in records if row['kind']=='controls_complete']
    if complete!=[{'groups':['kernel','helpers','actual_mutations']}]:raise PermissionError('missing manufactured obligations')
    intake=next(row['data'] for row in records if row['kind']=='intake')
    before={Path(k).name:v for k,v in intake['sources'].items()}
    now={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    if before!=now:raise PermissionError('matrix source differs from controls')
    digest=hashlib.sha256((directory/'written_report.md').read_bytes()).hexdigest()
    if digest not in review.read_text():raise PermissionError('sole review does not bind complete actual controls')
    return dict(directory=str(directory),report_sha256=digest,review=str(review),review_sha256=review_sha)


def prior_size_costs(paths,sources):
    results=[]
    for n,path in zip((2,3),paths):
        with (Path(path)/'records.ndjson').open() as f:rows=[json.loads(line) for line in f]
        end=rows[-1]
        if end['kind']!='termination' or end['data']['status']!='complete' or not end['data']['all_pass'] or not end['data']['sources_unchanged']:raise PermissionError('earlier size incomplete')
        summary=[row['data'] for row in rows if row['kind']=='matrix_admission_summary']
        if len(summary)!=1 or summary[0]['n']!=n:raise PermissionError('earlier size association')
        before={Path(k).name:v for k,v in end['data']['sources'].items()}
        now={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
        if before!=now:raise PermissionError('earlier size source differs')
        results.append(dict(path=str(path),summary=summary[0],cost=end['data']))
    if len(results)!=2:raise PermissionError('both earlier size costs required')
    return results


def exact(bundle,name,a,b):
    def compare():
        equal=a==b
        bundle.save('comparison_inputs',{'name':name,'_exact':{'actual':a,'expected':b}})
        scalar=(e.C,F,str,int,float,bool,type(None))
        def describe(z):
            if isinstance(z,scalar):return z
            if isinstance(z,(tuple,list)) and len(z)<=16 and all(isinstance(v,scalar) for v in z):return z
            if isinstance(z,dict) and len(z)<=8 and all(isinstance(k,(str,int)) and isinstance(v,scalar) for k,v in z.items()):return z
            return dict(type=type(z).__name__,entries=len(z))
        detail=dict(exact_equal=equal,actual=describe(a),expected=describe(b))
        def difference(x,y,path=()):
            if x==y:return None
            if isinstance(x,dict) and isinstance(y,dict):
                for key in sorted(set(x)|set(y),key=repr):
                    if key not in x or key not in y:return dict(path=path+(str(key),),actual=describe(x.get(key)),expected=describe(y.get(key)))
                    if x[key]!=y[key]:return difference(x[key],y[key],path+(str(key),))
            if isinstance(x,(tuple,list)) and isinstance(y,(tuple,list)):
                for j,(u,w) in enumerate(zip(x,y)):
                    if u!=w:return difference(u,w,path+(j,))
            return dict(path=path,actual=describe(x),expected=describe(y))
        if not equal:detail['first_difference']=difference(a,b)
        bundle.check(name,equal,detail)
    return bundle.phase('comparison_'+name,compare)


def kernel_phase(bundle):
    result=bundle.phase('kernel_two_route_construction',kernel.construct)
    bundle.save('kernel_all_routes',result)
    G=result['G'];other=result['independent_G'];q=result['q'];Q=result['Q']
    exact(bundle,'kernel_Q_at_one',kernel.evaluate(Q,F(1)),F(1,2))
    exact(bundle,'kernel_integral_q',kernel.integrate(q,F(-1),F(1)),F(1))
    exact(bundle,'kernel_second_moment_q',kernel.integrate((F(0),F(0))+q,F(-1),F(1)),F(1,15))
    exact(bundle,'kernel_I',result['anchors']['I'],F(429,4096))
    for r in range(-4,5):
        exact(bundle,'kernel_overlap_route_'+str(r),G[r],other.get(abs(r),F(0)))
    for lo,hi,parts,total in result['partition']:exact(bundle,'kernel_partition_'+str(lo),total,(F(1),))
    exact(bundle,'kernel_partition_sum',sum(G.values(),F(0)),F(1))
    for r in (0,1,2):bundle.check('kernel_positive_cauchy_'+str(r),0<G[r]<=G[0],dict(r=r,value=G[r],G0=G[0]))
    compact={r:G[r] for r in range(-2,3)};v.validate_kernel(compact)
    kappa=G[0]-2*G[1]+2*G[2];h=2*G[1]+8*G[2]
    bundle.check('kernel_kappa_range',0<=kappa<=1,dict(kappa=kappa,h=h))
    exported={r:F(float(z)) for r,z in compact.items()}
    export_difference={r:exported[r]-compact[r] for r in compact}
    error=validate_kernel_export(compact,exported,F(1,10**12))
    bundle.save('kernel_scalars',dict(G=compact,kappa_pi=kappa,h=h,binary64_export=exported,binary64_signed_difference=export_difference,binary64_error=error))
    bundle.check('kernel_binary64_budget',error<=F(1,10**12),dict(error=error))
    return compact


def helper_controls(bundle,G):
    for n in (2,3,7):
        b=v.Basis.make(n)
        exact(bundle,'basis_complete_'+str(n),len(b.states),math.comb(8*n,2))
        bundle.reject('basis_wrong_modes_'+str(n),lambda n=n:v.Basis.make(n,modes=8*n-1))
        bundle.reject('basis_wrong_sector_'+str(n),lambda n=n:v.Basis.make(n,number=1))
        bundle.reject('basis_truncation_'+str(n),lambda n=n,b=b:v.Basis.make(n,states=b.states[:-1]))
        bundle.reject('allocation_overcap_'+str(n),lambda b=b:b.allocation((2049,1)))
        bundle.reject('basis_cap_change_'+str(n),lambda n=n:v.Basis.make(n,cap=1024 if n==7 else 2048))
    for n in (True,2.,4):bundle.reject('basis_bad_size_'+str(n),lambda n=n:v.Basis.make(n))
    for U in (True,.5,1.,complex(1)):bundle.reject('U_guard_'+str(U),lambda U=U:v.validate_U(U))
    for U in (-1,0,1):exact(bundle,'U_valid_'+str(U),v.validate_U(U),U)
    for label,bad in [('negative',{**G,0:-G[0]}),('asymmetric',{**G,1:G[1]+1}),('missing',{r:c for r,c in G.items() if r!=2})]:
        bundle.reject('kernel_guard_'+label,lambda bad=bad:v.validate_kernel(bad))
    bundle.reject('kernel_export_mismatch',lambda:validate_kernel_export(G,{r:c+(1 if r==0 else 0) for r,c in G.items()},F(1,10**12)))
    bundle.reject('kernel_export_budget_failure',lambda:validate_kernel_export(G,{r:F(float(c)) for r,c in G.items()},F(0)))
    bundle.reject('kernel_export_missing',lambda:validate_kernel_export(G,{},F(1)))
    b=v.Basis.make(2)
    for bad in ((-1,True),(16,True),(True,True),(0,1)):
        bundle.reject('CAR_mode_guard_'+str(bad),lambda bad=bad:v.car(1,(bad,),16))
    for p in b.pairs[:8]:
        i,j=p;state=(1<<i)|(1<<j)
        exact(bundle,'CAR_pair_annihilation_'+str(p),v.car(state,((j,False),(i,False)),16),(0,1))
        exact(bundle,'CAR_reversed_pair_'+str(p),v.car(state,((i,False),(j,False)),16),(0,-1))
    def inner(a,b):return sum((z*b.get(x,F(0)) for x,z in a.items()),F(0))
    for n,j,norm,cross in ((2,-1,F(1,16),F(-1,16)),(2,0,F(5,16),F(3,16)),(7,3,F(3,8),F(1,8))):
        a=v.stencil(n,0,j);b=v.stencil(n,1,j)
        exact(bundle,'filter_norm_'+str((n,j)),inner(a,a),norm)
        exact(bundle,'filter_cross_'+str((n,j)),inner(a,b),cross)
    bundle.check('reject_canonical_filter',inner(v.stencil(2,0,0),v.stencil(2,0,0))!=1,{'reason':'filtered modes are not canonical'})
    for j in (0,1):
        bad=v.stencil(2,0,j,renormalize=True)
        bundle.check('reject_endpoint_renormalization_'+str(j),inner(bad,bad)!=F(5,16),dict(actual=inner(bad,bad),expected=F(5,16)))
    bundle.reject('stencil_bad_label',lambda:v.stencil(2,2,0))
    bundle.reject('stencil_missing_halo',lambda:v.stencil(2,0,-2))


def sparse_dot(left,right):
    return sum((left[i].conj()*z for i,z in right.items()),e.ZERO)


class ContractionCache:
    """Exact full-sector actions reused only for an identical full input vector.

    This changes traversal cost, never the operator, sector, state or arithmetic.
    Every new action is retained before its first measured contraction.
    """
    def __init__(self,matrix,centres,basis,name,declared_pairs=None):
        self.matrix=dict(matrix);self.centres={j:dict(m) for j,m in centres.items()}
        self.dimension=len(basis.states);self.name=name;self.actions={}
        self.expected=e.canonical_pairs(basis.modes)
        self.declared=tuple(basis.pairs if declared_pairs is None else declared_pairs)
        e.association(self.declared,self.expected)
        if any(type(i) is not int or type(j) is not int or not 0<=i<self.dimension or not 0<=j<self.dimension for m in (self.matrix,*self.centres.values()) for i,j in m):raise ValueError('matrix/basis index association')

    def contract(self,left,right,bundle,label):
        if len(left)!=self.dimension or len(right)!=self.dimension:raise ValueError('contraction full-sector vector length')
        e.bound_element({},left,right,self.declared,self.expected,1,0)
        key=tuple(right)
        if key not in self.actions:
            output=e.action(self.matrix,right,self.dimension)
            centre_outputs={j:e.action(m,right,self.dimension) for j,m in self.centres.items()}
            bundle.save('exact_cached_actions',dict(operator=self.name,label=label,input=right,output=output,centres=centre_outputs,dimension=self.dimension))
            self.actions[key]=(output,centre_outputs)
        output,centre_outputs=self.actions[key]
        def exact_dot(state):
            return sum((a.conj()*bb for a,bb in zip(left,state) if a and bb),e.ZERO)
        return exact_dot(output),{j:exact_dot(state) for j,state in centre_outputs.items()}


def local_norm(ts):
    q=math.prod(e.dot(t,t).r for t in ts);a=math.isqrt(q.numerator);b=math.isqrt(q.denominator)
    if a*a!=q.numerator or b*b!=q.denominator:raise ValueError('unexpected irrational local norm')
    return F(a,b)


def display_payload(z,den=F(1)):
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    e.normalizable(den)
    iv.dps=60;out=[]
    for a in (z.r,z.i):
        if not a:out.append((0.,0.));continue
        value=q.I(a)/iv.sqrt(q.I(den));out.append((q.lower(value),q.upper(value)))
    return dict(numerator=z,sqrt_denominator=den,intervals=out,status=e.classify_display(out))


def normalized_payload(z,Qh,Ql):
    e.normalizable(Qh);e.normalizable(Ql)
    return display_payload(z,Qh*Ql)


def interval(bundle,label,z,den=F(1),pair_norms=None):
    result=display_payload(z,den) if pair_norms is None else normalized_payload(z,*pair_norms)
    bundle.save('outward_display',dict(label=label,**result))
    bundle.check('display_resolved_'+label,result['status']=='resolved',dict(status=result['status']))
    return result['intervals']


def matrices(bundle,n,G):
    import mock_unitary_lead_v1 as q
    import mock_axial_vertex_adapter_v1 as adapter
    b=v.Basis.make(n);Ms=e.pairing_matrices()
    if n in (2,3):exact(bundle,'basis_matches_frozen',b.states,q.Basis.make(8*n,2).states)
    bundle.save('resource_preflight',dict(n=n,modes=b.modes,dimension=len(b.states),dense_cap=2048 if n==7 else 1024,
        dense_complex128_bytes=16*len(b.states)**2,earlier_costs='preceding size reports; no physical cost inference',truncation=False))
    rows=bundle.phase('term_expansion',lambda:v.expand_terms(b,G,adapter.PAIR_COEFFICIENTS,adapter.PAIR_ORDER))
    bundle.save('actual_terms',dict(rows=rows,basis=b.states,pairs=b.pairs))
    W,centres,forward=bundle.phase('CAR_assembly',lambda:v.assemble(b,rows))
    bundle.save('tested_vertex',{'basis':b.states,'pairs':b.pairs,'_exact':dict(W=W,centres=centres,forward=forward)})
    direct,dcentres,frows=bundle.phase('independent_M_pair_assembly',lambda:v.independent_matrix(b,G,Ms))
    bundle.save('independent_vertex',{'_exact':dict(W=direct,centres=dcentres,rows=frows)})
    exact(bundle,'full_vertex_matrix',W,direct);exact(bundle,'all_centre_matrices',centres,dcentres)
    total={}
    for m in centres.values():total=e.matrix_add(total,m)
    exact(bundle,'centre_partition',total,W);exact(bundle,'Hermiticity',W,e.adjoint(W))
    nh=[sum(j%8<4 for j in pair) for pair in b.pairs]
    bundle.check('species_parity',all(nh[i]%2==nh[j]%2 for i,j in W),{})
    bundle.check('forward_conversion_charge',all(nh[i]==0 and nh[j]==2 for i,j in forward),{})
    bundle.check('mixed_conversion_zero',all(nh[i]!=1 and nh[j]!=1 for i,j in W),{})
    bundle.check('separate_numbers_not_conserved',any(nh[i]!=nh[j] for i,j in W),{})
    bundle.save('charge_commutators',{'_exact':dict(NH={k:(nh[k[0]]-nh[k[1]])*z for k,z in W.items()},NL={k:(nh[k[1]]-nh[k[0]])*z for k,z in W.items()})})
    mass=F(float(F(3,5)));ht=bundle.phase('free_CAR_lift',lambda:v.lift_tested(b,v.onebody_terms(b,mass)))
    hd=bundle.phase('independent_four_delta_lift',lambda:e.lift_literal(e.literal_h(n,mass),b.pairs))
    ideal=e.lift_literal(e.literal_h(n,F(3,5)),b.pairs)
    exact(bundle,'free_full_matrix',ht,hd)
    exact(bundle,'represented_mass_difference',mass-F(3,5),F(-1,5*2**53))
    massdelta=e.matrix_add(ht,ideal,-1)
    bundle.save('mass_error',dict(delta_m=mass-F(3,5),operator_bound=2*abs(mass-F(3,5)),_exact=massdelta))
    bundle.check('mass_error_bound',all(i==j and not z.i and abs(z.r)<=2*abs(mass-F(3,5)) for (i,j),z in massdelta.items()),{})
    bundle.check('free_cross_species_zero',all(nh[i]==nh[j] for i,j in ht),{})
    actual={}
    for U in (0,1,-1):
        v.validate_U(U);H=e.matrix_add(ht,W,U);independent=e.matrix_add(hd,direct,U)
        export,delta,frob2=bundle.phase('export_'+str(U),lambda H=H:v.rounded(H))
        expected=v.independent_rounded(independent)
        bundle.save('Hamiltonian',dict(U=U,basis=b.states,pairs=b.pairs,_exact=dict(exact_represented=H,independent=independent,export=export,assembly_difference=delta,frob2=frob2)))
        exact(bundle,'rounded_H_'+str(U),export,expected)
        exact(bundle,'exact_H_'+str(U),H,independent)
        bounds=interval(bundle,'assembly_Frobenius_'+str(U),e.C(frob2),frob2 if frob2 else F(1))
        allowance=F(bounds[0][1])
        bundle.check('assembly_allowance_'+str(U),frob2<=allowance**2,dict(frob2=frob2,upper=allowance))
        bundle.save('total_operator_allowance',dict(U=U,assembly=allowance,mass=2*abs(mass-F(3,5)),total=allowance+2*abs(mass-F(3,5))))
        actual[U]=H
    exact(bundle,'positive_U_difference',e.matrix_add(actual[1],actual[0],-1),W)
    exact(bundle,'negative_U_difference',e.matrix_add(actual[-1],actual[0],-1),{key:-z for key,z in W.items()})
    index={p:i for i,p in enumerate(b.pairs)};factor=(9*G[0]+8*G[1]+G[2])/64
    for x in range(n):
        for pair,K in (((0,1),4),((0,2),-8),((0,3),-4)):
            h=index[8*x+pair[0],8*x+pair[1]];l=index[8*x+4+pair[0],8*x+4+pair[1]]
            exact(bundle,'same_site_'+str((x,pair)),W.get((l,h),e.ZERO),e.C(K*factor))
    on=ons=None
    if n==7:
        on_direct,orows=bundle.phase('independent_onsite_comparison',lambda:v.onsite_matrix(b,Ms))
        (on,ons_all,on_forward),on_terms=bundle.phase('tested_onsite_translation',lambda:v.onsite_tested(b,adapter.onsite_axial_pair_terms(0,1)))
        ons={x:ons_all[x] for x in range(n)}
        exact(bundle,'onsite_complete_matrix',on,on_direct)
        bundle.save('onsite_matrices',{'_exact':dict(W=on,centres=ons,rows=orows,terms=on_terms)})
    else:
        base=adapter.onsite_axial_pair_terms(0,1)
        for x in range(n):
            expected=adapter.onsite_axial_pair_terms(x,n)
            translated=[(e.cc(t.coefficient),tuple((mode+8*x,creation) for mode,creation in t.operators)) for t in base]
            exact(bundle,'onsite_translation_'+str(x),translated,[(e.cc(t.coefficient),t.operators) for t in expected])
    return b,Ms,W,centres,actual,on,ons,rows


class LegCache:
    def __init__(self,basis):
        self.basis=basis;self.cache={};slots={}
        for label,(hs,ls,ih,il,expected_null) in e.fixed_roster(basis.n).items():
            for suffix,species,spec,spin in (('H1',0,hs[0],ih[0]),('H2',0,hs[1],ih[1]),('L1',1,ls[0],il[0]),('L2',1,ls[1],il[1])):
                slots[label+suffix]=(label+suffix,species,spec,spin)
        self.expected_slots=MappingProxyType(slots)

    def make(self,descriptor,expected,provided=None):
        e.verify_leg_descriptor(descriptor,expected)
        label,species,spec,spin=expected
        if self.expected_slots.get(label)!=expected:raise ValueError('intrinsic size/roster/slot/parameter association')
        z,r,m,E=spec;n=self.basis.n
        manual=[e.ZERO]*(8*n);a,b=(1,1) if spin==1 else (-1,1)
        local=(e.C(a),e.C(b),e.C(spin*r*a),e.C(spin*r*b))
        for x in range(n):
            for c in range(4):manual[8*x+4*species+c]=(z**x)*local[c]
        key=(n,species,spec,spin)
        if key not in self.cache:self.cache[key]=e.template(n,species,z,r,spin)
        actual=self.cache[key] if provided is None else provided
        e.validate_leg(actual,8*n,species,label,expected[0])
        if actual!=manual:raise ValueError('same-length leg content mismatch')
        return actual,tuple(local)


def mutation_controls(bundle,G):
    import mock_axial_vertex_adapter_v1 as adapter
    b=v.Basis.make(2);rows=v.expand_terms(b,G,adapter.PAIR_COEFFICIENTS,adapter.PAIR_ORDER)
    W,centres,forward=v.assemble(b,rows);index={pair:i for i,pair in enumerate(b.pairs)}
    locator=(index[(4,5)],index[(0,1)]);reverse=locator[::-1];target=e.C(4*(9*G[0]+8*G[1]+G[2])/64)
    exact(bundle,'mutant_baseline_entry',W.get(locator,e.ZERO),target)
    choices=[('half_pair',rows,dict(half_pair=True)),('reversed_annihilators',rows,dict(reversed_annihilators=True)),
        ('doubled_adjoint',rows,dict(doubled_adjoint=True)),('missing_G2',[r for r in rows if abs(r[0]-r[1])!=2],{})]
    for name,kw in [('omitted_halo',dict(omit_halo=True)),('wrap',dict(wrap=True)),('endpoint_renormalization',dict(renormalize=True)),('independent_b',dict(independent_b=True)),('wrong_kernel_offset',dict(kernel_offset=1))]:
        choices.append((name,v.expand_terms(b,G,adapter.PAIR_COEFFICIENTS,adapter.PAIR_ORDER,**kw),{}))
    for name,mutrows,kw in choices:
        mut,mc,mf=bundle.phase('mutation_'+name,lambda mutrows=mutrows,kw=kw:v.assemble(b,mutrows,**kw))
        loc=reverse if name=='doubled_adjoint' else locator
        actual=mut.get(loc,e.ZERO)
        bundle.save('term_mutation',dict(name=name,locator=loc,actual=actual,expected=target,_exact=dict(matrix=mut,difference=e.matrix_add(mut,W,-1),terms=mutrows)))
        bundle.check('reject_term_mutation_'+name,actual!=target,dict(locator=loc,actual=actual,expected=target,reason='decisive endpoint pair target failed'))
    pairs=b.pairs;left=[e.ZERO]*len(pairs);right=list(left);left[locator[0]]=e.ONE;right[locator[1]]=e.ONE
    bundle.reject('basis_permutation_mismatch',lambda:e.bound_element(W,left,right,tuple(reversed(pairs)),pairs,1,0))
    perm=tuple(reversed(range(len(pairs))));inv={old:new for new,old in enumerate(perm)}
    mapped={(inv[i],inv[j]):z for (i,j),z in W.items()}
    exact(bundle,'consistent_basis_permutation',e.matrix_element(mapped,[left[i] for i in perm],[right[i] for i in perm]),target)
    cache=LegCache(b);manifest=e.fixed_roster(2);label=next(iter(manifest));hs,ls,ih,il,null=manifest[label]
    desc=(label+'H1',0,hs[0],ih[0]);good,_=cache.make(desc,desc)
    bundle.reject('cached_wrong_slot',lambda:cache.make((label+'H2',0,hs[0],ih[0]),desc))
    bundle.reject('cached_wrong_size',lambda:LegCache(v.Basis.make(3)).make(desc,desc,provided=good))
    bundle.reject('fresh_wrong_size_label',lambda:LegCache(v.Basis.make(3)).make(desc,desc))
    fresh3=e.template(3,0,hs[0][0],hs[0][1],ih[0])
    bundle.reject('fresh_correct_shape_wrong_size_label',lambda:LegCache(v.Basis.make(3)).make(desc,desc,provided=fresh3))
    badslot=(label+'H1',0,hs[1],ih[1])
    bundle.reject('fresh_wrong_slot_parameters',lambda:LegCache(b).make(badslot,badslot))
    badparameter=(label+'H1',0,(e.ONE,*hs[0][1:]),ih[0])
    bundle.reject('fresh_wrong_parameter_label',lambda:LegCache(b).make(badparameter,badparameter))
    bad=list(good);bad[0]+=1
    bundle.reject('cached_wrong_same_length_vector',lambda:cache.make(desc,desc,provided=bad))
    badspecies=[e.ZERO]*16;badspecies[4]=e.ONE
    bundle.reject('cached_wrong_species',lambda:cache.make(desc,desc,provided=badspecies))
    for badroster in (('a',),('a','a'),('b','a')):bundle.reject('roster_guard_'+str(badroster),lambda badroster=badroster:e.roster_check(badroster,('a','b')))
    bundle.reject('null_status_guard',lambda:e.enforce_pair_classification(F(0),F(1),False))
    # Same actual bound contraction handles complex rephasing, pair sign and rescaling.
    actual=e.bound_element(W,left,right,pairs,pairs,1,0)
    for label,a,bb,factor in [('real_rescale',e.scale(left,3),e.scale(right,2),e.C(6)),('phase_left',e.scale(left,e.J),right,-e.J),('reverse_right',left,e.scale(right,-1),e.C(-1))]:
        exact(bundle,'pair_transform_'+label,e.bound_element(W,a,bb,pairs,pairs,1,0),actual*factor)
    modes=b.modes
    single=[]
    for index_ in (0,1,4,5):
        leg=[e.ZERO]*modes;leg[index_]=e.ONE;single.append(leg)
    h1,h2,l1,l2=single;scaledh=e.scale(h1,2);scaledl=e.scale(l1,3)
    wh=e.wedge(scaledh,h2,pairs);wl=e.wedge(scaledl,l2,pairs)
    Qh=e.gram(scaledh,h2)[0];Ql=e.gram(scaledl,l2)[0]
    transformed=e.bound_element(W,wl,wh,pairs,pairs,1,0)
    interval(bundle,'manufactured_normalized_original',actual,pair_norms=(F(1),F(1)))
    interval(bundle,'manufactured_normalized_rescaled',transformed,pair_norms=(Qh,Ql))
    exact(bundle,'rescaled_actual_Grams',(Qh,Ql),(F(4),F(9)))
    exact(bundle,'normalized_rescaling',transformed/6,actual)
    for qh,ql in ((F(0),F(1)),(F(-1),F(1)),(F(-1),F(-1))):
        bundle.reject('normalization_norm_guard_'+str((qh,ql)),lambda qh=qh,ql=ql:normalized_payload(actual,qh,ql))
    wide=normalized_payload(e.C(10**100),F(1),F(1))
    exact(bundle,'normalization_unresolved_control',wide['status'],'precision_unresolved')
    action_cache=ContractionCache(W,centres,b,'manufactured_W')
    bundle.reject('production_cache_basis_mismatch',lambda:ContractionCache(W,centres,b,'wrong',tuple(reversed(pairs))))
    bundle.reject('production_cache_species',lambda:action_cache.contract(right,left,bundle,'wrong_species'))
    cached,cvalues=action_cache.contract(left,right,bundle,'manufactured_first')
    exact(bundle,'cached_complete_contraction',cached,e.matrix_element(W,left,right))
    exact(bundle,'cached_all_centre_contractions',cvalues,{j:e.matrix_element(m,left,right) for j,m in centres.items()})
    other=e.scale(right,e.J)
    changed,cv=action_cache.contract(left,other,bundle,'manufactured_changed')
    exact(bundle,'cached_changed_vector',changed,e.J*actual)
    exact(bundle,'cached_repeated_vector',action_cache.contract(left,right,bundle,'manufactured_repeated'),(cached,cvalues))
    bundle.reject('cache_wrong_length',lambda:action_cache.contract(left[:-1],right,bundle,'wrong'))
    # A one-particle input cannot survive the two annihilators, through actual CAR.
    for mode in range(16):
        bundle.check('one_particle_zero_'+str(mode),all(v.car(1<<mode,ops,16)[1]==0 and v.car(1<<mode,v.adjoint_ops(ops),16)[1]==0 for j,k,c,ops in rows),{})


def ordinary_probes(bundle,b,Ms,W,centres,actual,on,ons,G):
    manifest=e.fixed_roster(b.n);declared=tuple(manifest);bundle.save('ordinary_roster',manifest)
    cache=LegCache(b);done=[];counts=dict(null=0,normalized=0);interior={};onsite_interior={};intervalley_found=False
    actual_cache=ContractionCache(actual[1],centres,b,'actual_H_plus1')
    onsite_cache=ContractionCache(on,ons,b,'tested_onsite_W') if b.n==7 else None
    for label,(hs,ls,ih,il,expected_null) in manifest.items():
        def inputs():
            descriptors=[(label+'H1',0,hs[0],ih[0]),(label+'H2',0,hs[1],ih[1]),(label+'L1',1,ls[0],il[0]),(label+'L2',1,ls[1],il[1])]
            legs,ts=zip(*(cache.make(d,d) for d in descriptors))
            wh=e.wedge(legs[0],legs[1],b.pairs);wl=e.wedge(legs[2],legs[3],b.pairs)
            gh=e.gram(legs[0],legs[1]);gl=e.gram(legs[2],legs[3])
            return legs,ts,wh,wl,gh,gl
        legs,ts,wh,wl,gh,gl=bundle.phase(label+'_construction',inputs)
        Qh,Ql=gh[0],gl[0];lc=e.local_C(ts,Ms);nf=local_norm(ts);phases=tuple(t[0] for t in hs+ls)
        S,Sj=v.envelope(b,G,phases)
        bundle.save('ordinary_inputs',dict(label=label,legs=legs,wedgeH=wh,wedgeL=wl,gramH=gh,gramL=gl,C=lc,local_norm=nf,S=S,centre_envelopes=Sj))
        e.enforce_pair_classification(Qh,Ql,expected_null)
        exact(bundle,label+'_gramH',e.dot(wh,wh),e.C(Qh));exact(bundle,label+'_gramL',e.dot(wl,wl),e.C(Ql))
        raw,values=bundle.phase(label+'_actual_contraction',lambda:actual_cache.contract(wl,wh,bundle,label))
        bundle.save('actual_ordinary_raw_before_other_work',dict(label=label,raw=raw,centres=values))
        onraw,onvalues=onsite_cache.contract(wl,wh,bundle,label) if onsite_cache else (None,None)
        free=e.matrix_element(actual[0],wl,wh)
        bundle.save('ordinary_contractions',dict(label=label,raw=raw,centres=values,expected=lc*S,baseline_raw=onraw,baseline_centres=onvalues,free=free))
        exact(bundle,label+'_full_raw',raw,lc*S)
        exact(bundle,label+'_centre_raw',values,{j:lc*s for j,s in Sj.items()})
        exact(bundle,label+'_free_zero',free,e.ZERO)
        totalphase=(phases[0]*phases[1])/(phases[2]*phases[3]);geometric=sum((totalphase**x for x in range(b.n)),e.ZERO)
        if onsite_cache:
            exact(bundle,label+'_onsite_raw',onraw,lc*geometric)
            for x in range(b.n):exact(bundle,label+'_onsite_centre_'+str(x),onvalues[x],lc*(totalphase**x))
        if expected_null:
            counts['null']+=1;exact(bundle,label+'_Pauli_null',wh,[e.ZERO]*len(wh))
        else:
            counts['normalized']+=1;interval(bundle,label+'_filtered_normalized',raw,pair_norms=(Qh,Ql))
            if onsite_cache:interval(bundle,label+'_onsite_normalized',onraw,pair_norms=(Qh,Ql))
        if '_collinear_' in label:
            sign=1 if '_collinear_1_' in label else -1
            target=e.C(F(-sign*32,5)) if ih==il and ih[0]==ih[1] else e.ZERO
            exact(bundle,label+'_unfiltered_local_target',lc/nf,target)
            if b.n==7:
                centre=values[3]/nf;wanted=target*F(17,100)
                exact(bundle,label+'_filtered_interior_target',centre,wanted)
                interior[label]=centre;onsite_interior[label]=onvalues[3]/nf
                exact(bundle,label+'_onsite_interior_target',onvalues[3]/nf,target)
        else:
            bits=label.split('_valley_')[1][:4]
            if bits[2]==bits[3]:exact(bundle,label+'_same_outgoing_valley_threshold_zero',lc,e.ZERO)
            if bits=='0101' and ih==(1,-1) and il==(1,-1):
                intervalley_found=True
                bundle.save('decisive_intervalley',dict(n=b.n,label=label,raw=raw,gramH=gh,gramL=gl,centres=values,normalized=not expected_null,authority='#143 full-support zero; report all finite boundary leakage'))
                if b.n==2:
                    # Full template norms differ from separately normalized leg Grams;
                    # express the inherited normalized pair amplitude with its exact sqrt.
                    target=(10*G[0]-3*G[1]-6*G[2])/64
                    product_norm=Qh*Ql;rootn=math.isqrt(product_norm.numerator);rootd=math.isqrt(product_norm.denominator)
                    exact(bundle,'intervalley_norm_is_square',F(rootn**2,rootd**2),product_norm)
                    exact(bundle,'intervalley_n2_positive_target',raw/F(rootn,rootd),e.C(target))
                    bundle.check('intervalley_n2_positive',target>0,dict(target=target))
                if b.n==7:exact(bundle,'intervalley_full_support_zero',values[3],e.ZERO)
        done.append(label)
    e.roster_check(tuple(done),declared)
    exact(bundle,'ordinary_row_count',len(done),288)
    #64 null incoming threshold probes;32 non-null collinear rows.
    exact(bundle,'ordinary_pair_counts',counts,dict(null=64,normalized=224))
    bundle.check('decisive_intervalley_present',intervalley_found,{})
    if b.n==7:
        for sign in (1,-1):
            selected=[z for label,z in interior.items() if ('_collinear_'+str(sign)+'_') in label]
            exact(bundle,'interior_spin_average_'+str(sign),sum((z.abs2() for z in selected),F(0))/4,F(9248,15625))
            old_selected=[z for label,z in onsite_interior.items() if ('_collinear_'+str(sign)+'_') in label]
            exact(bundle,'onsite_interior_spin_average_'+str(sign),sum((z.abs2() for z in old_selected),F(0))/4,F(512,25))


def independent_tangent_envelope(b,G,Ms,il,ih,z1,z2):
    """Literal derivative of filtered pair bilinears, including phase derivatives."""
    t1=e.local_template(F(0),ih[0]);t2=e.local_template(F(0),ih[1])
    l1=e.local_template(F(1),il[0]);l2=e.local_template(F(-1),il[1])
    def bilinear(a,M,bb):return sum((a[i]*M[i][j]*bb[j] for i in range(4) for j in range(4)),e.ZERO)
    result={j:e.ZERO for j in range(-1,b.n+1)}
    for band in (0,1):
        for j in range(-1,b.n+1):
            L=v.stencil(b.n,band,j);p1=sum((c*z1**x for x,c in L.items()),e.ZERO);p2=sum((c*z2**x for x,c in L.items()),e.ZERO)
            for k in range(-1,b.n+1):
                if abs(j-k)>2:continue
                R=v.stencil(b.n,band,k);p=sum(R.values(),F(0));dx=sum((c*x for x,c in R.items()),F(0))
                d1=e.local_template(F(1),ih[0]);d1=e.vecadd(d1,e.scale(t1,-1))
                d2=e.scale(e.vecadd(e.local_template(F(1),ih[1]),e.scale(t2,-1)),-1)
                h1=e.scale(t1,p);h2=e.scale(t2,p)
                dh1=e.vecadd(e.scale(d1,p),e.scale(t1,e.J*F(6,5)*dx))
                dh2=e.vecadd(e.scale(d2,p),e.scale(t2,-e.J*F(6,5)*dx))
                value=4*G[j-k]*sum(((1 if mu==0 else -1)*(bilinear(l1,M,l2)*p1*p2).conj()*(bilinear(dh1,M,h2)+bilinear(h1,M,dh2)) for mu,M in enumerate(Ms)),e.ZERO)
                result[j]+=value/2;result[k]+=value/2
    return sum(result.values(),e.ZERO),result


def tangent_probes(bundle,b,Ms,W,centres,on,ons,G):
    spins=((1,1),(1,-1),(-1,1),(-1,-1));declared=tuple((branch,il,ih) for branch in (0,1) for il in spins for ih in spins)
    bundle.save('tangent_roster',declared);done=[];z=e.C(F(4,5),F(3,5))
    actual_cache=ContractionCache(W,centres,b,'actual_W_tangent')
    onsite_cache=ContractionCache(on,ons,b,'tested_onsite_W_tangent') if b.n==7 else None
    for branch,il,ih in declared:
        compare_onsite=onsite_cache is not None and branch==0
        label='tangent_'+str((branch,il,ih))
        def construct_tangent():
            z1,z2=(z,z.conj()) if branch==0 else (-z.conj(),-z)
            h1=e.template(b.n,0,e.ONE,F(0),ih[0]);h2=e.template(b.n,0,e.ONE,F(0),ih[1])
            l1=e.template(b.n,1,z1,F(1),il[0]);l2=e.template(b.n,1,z2,F(-1),il[1])
            dh1=e.derivative_template(b.n,0,ih[0],0);dh2=e.derivative_template(b.n,0,ih[1],1)
            dw=e.vecadd(e.wedge(dh1,h2,b.pairs),e.wedge(h1,dh2,b.pairs));wl=e.wedge(l1,l2,b.pairs)
            return z1,z2,h1,h2,l1,l2,dh1,dh2,dw,wl
        z1,z2,h1,h2,l1,l2,dh1,dh2,dw,wl=bundle.phase(label+'_construction',construct_tangent)
        bundle.save('tangent_inputs',dict(label=label,heavy=(h1,h2),light=(l1,l2),heavy_derivative=(dh1,dh2),light_derivative=([e.ZERO]*b.modes,[e.ZERO]*b.modes),raw_wedge_derivative=dw))
        exact(bundle,label+'_full_derivative',dw,e.tangent_literal(b.n,ih[0],b.pairs,ih[1]))
        raw,values=bundle.phase(label+'_actual_contraction',lambda:actual_cache.contract(wl,dw,bundle,label))
        bundle.save('actual_tangent_raw_before_other_work',dict(label=label,raw=raw,centres=values))
        reference,rvalues=bundle.phase(label+'_independent_reference',lambda:independent_tangent_envelope(b,G,Ms,il,ih,z1,z2))
        onraw,onvalues=bundle.phase(label+'_onsite_contraction',lambda:onsite_cache.contract(wl,dw,bundle,label)) if compare_onsite else (None,None)
        bundle.save('tangent_contractions',dict(label=label,raw=raw,centres=values,independent=reference,independent_centres=rvalues,onsite=onraw,onsite_centres=onvalues))
        exact(bundle,label+'_whole_raw',raw,reference);exact(bundle,label+'_centre_raw',values,rvalues)
        target=e.C(-16) if il==ih and ih[0]==ih[1] else e.ZERO
        if compare_onsite:exact(bundle,label+'_onsite_local',onraw/F(8*b.n),target)
        if b.n==7:
            expected=target*(F(81,100) if branch==0 else F(1,100))
            exact(bundle,label+'_interior',values[3]/8,expected)
        # Raw null-family derivatives are labelled as derivatives, never normalized states.
        bundle.save('tangent_scope',dict(label=label,prepared_state_amplitude=False,status='raw derivative',onsite_local=onraw/F(8*b.n) if compare_onsite else None,filtered_central=values.get(3,e.ZERO)/8 if b.n==7 else None))
        done.append((branch,il,ih))
    e.roster_check(tuple(done),declared);exact(bundle,'tangent_count',len(done),32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('controls','matrix'))
    parser.add_argument('--n',required=True,type=int,choices=(2,3,7));parser.add_argument('--output',required=True);parser.add_argument('--spec',required=True)
    parser.add_argument('--contracts',required=True)
    parser.add_argument('--controls-output');parser.add_argument('--control-review');parser.add_argument('--control-review-sha')
    parser.add_argument('--prior-sizes',nargs=2)
    args=parser.parse_args()
    if args.stage=='controls' and args.n!=2:raise ValueError('small helper controls only')
    spec=Path(args.spec)
    if hashlib.sha256(spec.read_bytes()).hexdigest()!=SPEC_SHA:raise ValueError('frozen specification identity')
    from mock_filtered_records_v1 import StaticBundle
    code=Path(__file__).parent
    names=('run_mock_filtered_vertex_admission_v1.py','mock_filtered_records_v1.py','mock_filtered_kernel_v1.py','mock_filtered_vertex_v1.py',
        'run_mock_vertex_admission_v1.py','mock_unitary_lead_v1.py','mock_axial_vertex_adapter_v1.py','mock_unitary_lead_records_v1.py','cpu_runtime.py')
    contracts=Path(args.contracts);verify_frozen(code,contracts)
    sources=[spec,*(code/name for name in names),*(contracts/name for name in FROZEN if name.endswith('.md'))]
    gate=None;previous=None
    if args.stage=='matrix':
        if not all((args.controls_output,args.control_review,args.control_review_sha)):raise PermissionError('actual completed controls and sole review required')
        gate=control_gate(args.controls_output,args.control_review,args.control_review_sha,sources)
        if args.n==7:
            if not args.prior_sizes:raise PermissionError('earlier size costs required before n7')
            previous=prior_size_costs(args.prior_sizes,sources)
    bundle=StaticBundle(args.output,sources,args.n)
    status='incomplete';error=None
    try:
        if gate:bundle.save('control_gate',gate)
        if previous:bundle.save('earlier_filtered_size_costs',previous)
        G=kernel_phase(bundle)
        for name,z in [('kappa_pi',G[0]-2*G[1]+2*G[2]),('h',2*G[1]+8*G[2]),*((str(r),c) for r,c in G.items())]:interval(bundle,'kernel_'+name,e.C(z))
        if args.stage=='controls':
            bundle.phase('helper_controls',lambda:helper_controls(bundle,G))
            bundle.phase('actual_mutation_controls',lambda:mutation_controls(bundle,G))
            bundle.save('controls_complete',dict(groups=['kernel','helpers','actual_mutations']))
        else:
            b,Ms,W,centres,actual,on,ons,rows=bundle.phase('matrices_and_exports',lambda:matrices(bundle,args.n,G))
            bundle.phase('complete_ordinary_roster',lambda:ordinary_probes(bundle,b,Ms,W,centres,actual,on,ons,G))
            bundle.phase('complete_tangent_roster',lambda:tangent_probes(bundle,b,Ms,W,centres,on,ons,G))
            bundle.save('matrix_admission_summary',dict(n=b.n,modes=b.modes,dimension=len(b.states),terms=len(rows),W_nnz=len(W),H_nnz=len(actual[1]),centre_nnz={j:len(m) for j,m in centres.items()},ordinary=288,tangent=32,
                channels='all axial valleys and four M components; perpendicular256/25 and transverse valleys absent',scope='flat static matrix only; no propagation or current'))
        status='complete'
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);bundle.save('incomplete',dict(error=error))
    finally:bundle.finish(status,error)
    if status!='complete':raise SystemExit(1)


if __name__=='__main__':main()
