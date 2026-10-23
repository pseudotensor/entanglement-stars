"""Own S10 fixture implementation, CORE_missing_work_spec_v1 section22.

Candidate source: full literal review is required before the bounded allocation.
No scientific code in this file is executed on the local host.
"""
from fractions import Fraction
from itertools import combinations, product
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

ROOT=Path('<internal>/s10_fixture_v1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph(np):
    """Independent open axial graph, site-major standard Dirac convention."""
    sigma=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],dtype=complex)
    zero=np.zeros((2,2),complex)
    alpha=np.array([np.block([[zero,s],[s,zero]]) for s in sigma])
    beta=np.diag([1,1,-1,-1]).astype(complex)
    positions=[(0,0,0)]
    for radius in (1,2):
        for axis in range(3):
            for sign in (1,-1):
                p=[0,0,0];p[axis]=sign*radius;positions.append(tuple(p))
    h=np.kron(np.eye(13),beta/5)
    index={p:i for i,p in enumerate(positions)}
    edges=[]
    for p,j in index.items():
        for axis in range(3):
            q=list(p);q[axis]+=1;q=tuple(q)
            if q not in index:continue
            k=index[q];h[4*j:4*j+4,4*k:4*k+4]=-1j*alpha[axis]/2
            h[4*k:4*k+4,4*j:4*j+4]=1j*alpha[axis]/2;edges.append((j,k,axis))
    shell=np.array([np.repeat([sum(abs(x) for x in p)==r for p in positions],4) for r in range(3)],dtype=float)
    vertices=np.array([(s[:,None]*h+h*s[None,:])/2 for s in shell])
    return dict(positions=np.array(positions),alpha=alpha,beta=beta,h0=h,A=vertices,shell_masks=shell,edges=np.array(edges))


def fixed_number_bilinears(np,occupation):
    """Three-mode CAR with least-significant-bit mode zero."""
    labels=sorted(sum(1<<i for i in modes) for modes in combinations(range(3),occupation))
    lookup={mask:i for i,mask in enumerate(labels)}
    E=np.zeros((3,3,len(labels),len(labels)),complex)
    for col,mask in enumerate(labels):
        for j in range(3):
            if not mask&(1<<j):continue
            sign1=(-1)**((mask&((1<<j)-1)).bit_count());removed=mask^(1<<j)
            for i in range(3):
                if removed&(1<<i):continue
                sign2=(-1)**((removed&((1<<i)-1)).bit_count())
                E[i,j,lookup[removed|(1<<i)],col]=sign1*sign2
    return np.array(labels,dtype=int),E


def radial_factors(np):
    """Literal free eigenvectors in Eq46; shell products use full matrices."""
    mu=1/5;eps=math.sqrt(mu*mu+7/4);ed=math.sqrt(mu*mu+1/4)
    b=np.array([math.sqrt(6),0,1])/math.sqrt(7)
    d=np.array([1,0,-math.sqrt(6)])/math.sqrt(7);f=np.array([0,1,0])
    factors=[]
    for s in (1,-1):
        c=math.sqrt((eps+s*mu)/(2*eps));a=math.sqrt((eps-s*mu)/(2*eps))
        U=np.column_stack((-a*b+c*f,d,c*b+a*f)).astype(complex)
        h=np.array([[s*mu,math.sqrt(6)/2,0],[math.sqrt(6)/2,-s*mu,1/2],[0,1/2,s*mu]],complex)
        As=[]
        for r in range(3):
            pi=np.zeros((3,3));pi[r,r]=1;As.append(U.conj().T@((pi@h+h@pi)/2)@U)
        factors.append(dict(sign=s,U=U,h=h,A=np.array(As),energies=np.array([-eps,s*mu,eps]),reference=np.array([1,int(s<0),0])))
    return mu,eps,ed,factors


def tensor_bilinears(np,occupations):
    """Four physical copies; last factor varies fastest."""
    labels=[];local=[];embedded=[]
    for n in occupations:
        masks,E=fixed_number_bilinears(np,n);labels.append(masks);local.append(E)
    for copy in range(4):
        E=np.zeros((3,3,81,81),complex)
        for i in range(3):
            for j in range(3):
                value=np.array([[1]],complex)
                for factor in range(4):
                    value=np.kron(value,local[factor][i,j] if copy==factor else np.eye(3))
                E[i,j]=value
        embedded.append(E)
    determinant_labels=np.array(list(product(*labels)),dtype=int)
    return determinant_labels,np.array(embedded),local


def main():
    if Path(__file__).resolve().parent!=ROOT:raise RuntimeError('Named box directory required')
    if any(ROOT.glob('*.npz*')) or (ROOT/'s10_result.json').exists():raise FileExistsError('Missing-only allocation; never retry')
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(4*2**30,4*2**30));resource.setrlimit(resource.RLIMIT_CPU,(2300,2400))
    def deadline(s,f):raise TimeoutError('Bounded S10 allocation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.signal(signal.SIGXCPU,deadline);signal.alarm(580)
    start=time.monotonic();report=dict(status='incomplete',scope='S10 scalar fixture only; no native HF',
        specification_snapshot_sha256='7953ae8f1d02a0f25fdba6840bfc24f5add6842b99230c9817073ca1d2b17a58',units=dict(a=1,energy='1/a',variance='1/a^2',counts='dimensionless'),inputs=dict(m='1/5',g='1/4096',K=[['12/5','1','1/2'],['1','1','1/2'],['1/2','1/2','1/2']]),
        products={},controls={},scf={},exact={},completion={},resources=dict(threads=4,wall=600,cpu=2400,address_space=4*2**30))
    def save():
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGALRM,signal.SIGXCPU})
        try:
            report.update(elapsed_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,producer_sha256=digest(Path(__file__)))
            tmp=ROOT/'s10_result.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(ROOT/'s10_result.json')
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    try:
        import numpy as np
        import scipy.linalg as la
        def real(z):return float(np.real(z))
        def norm(z):return float(la.norm(z))
        def maximum(z):return float(np.max(np.abs(z)))
        def check(name,residual,scale=1.0,tolerance=1e-12):
            report['controls'][name]=dict(residual=float(residual),scale=float(scale),threshold=tolerance,passed=bool(residual<=tolerance))
            if not residual<=tolerance:raise ArithmeticError('Control failed: '+name)
        def product_file(name,**arrays):
            path=ROOT/(name+'.npz');tmp=ROOT/(name+'.npz.tmp')
            if path.exists() or tmp.exists():raise FileExistsError('Completed product exists')
            with tmp.open('xb') as stream:np.savez_compressed(stream,**arrays)
            tmp.replace(path);report['products'][name]=dict(path=str(path),size=path.stat().st_size,sha256=digest(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()})
            save()
        graph_data=graph(np);h,A=graph_data['h0'],graph_data['A'];mu,eps,ed,factors=radial_factors(np);g=1/4096
        K=np.array([[12/5,1,1/2],[1,1,1/2],[1/2,1/2,1/2]],float)
        w,V=la.eigh(h);P0=V[:,:26]@V[:,:26].conj().T;Rplus=V[:,26:28]@V[:,26:28].conj().T
        expected=np.sort(np.array([-mu]*2+[mu]*2+[-eps]*4+[eps]*4+[-ed]*20+[ed]*20))
        check('graph_edges',abs(len(graph_data['edges'])-12));check('graph_spectrum',maximum(w-expected))
        check('graph_hermiticity',maximum(h-h.conj().T));check('vertex_sum',maximum(A.sum(axis=0)-h))
        check('free_projector_rank',abs(real(np.trace(P0))-26));check('free_eigen_residual',maximum(h@V-V*w))
        check('vertex_squared_traces',maximum(np.array([np.trace(a@a) for a in A])-np.array([79,174,99])/25))
        bright=[];Ts=[];hop=h-mu*np.kron(np.eye(13),graph_data['beta'])
        for c in range(4):
            e0=np.zeros(52,complex);e0[c]=1;e1=2/math.sqrt(6)*hop@e0;e2=2*(hop@e1-math.sqrt(6)/2*e0)
            T=np.column_stack((e0,e1,e2));factor=factors[c//2];L=T@factor['U'];bright.append(L);Ts.append(T)
            check('bright_free_intertwiner_'+str(c),maximum(h@L-L*factor['energies']))
            check('graph_radial_intertwiner_'+str(c),maximum(h@T-T@factor['h']))
            for r in range(3):check('graph_shell_vertex_'+str(c)+'_'+str(r),maximum(L.conj().T@A[r]@L-factor['A'][r]))
        bright_matrix=np.column_stack(bright);check('bright_isometry',maximum(bright_matrix.conj().T@bright_matrix-np.eye(12)))
        for r in range(3):check('vertex_bright_offblock_'+str(r),maximum((np.eye(52)-bright_matrix@bright_matrix.conj().T)@A[r]@bright_matrix))
        Pdark=P0-sum((L@np.diag(factors[c//2]['reference'])@L.conj().T for c,L in enumerate(bright)),np.zeros((52,52),complex))
        check('dark_projector_rank',abs(real(np.trace(Pdark))-20));check('dark_projector_idempotency',maximum(Pdark@Pdark-Pdark))
        dark=[]
        for s in (1,-1):
            hd=np.array([[s*mu,1/2],[1/2,-s*mu]],complex);dc=math.sqrt((ed+s*mu)/(2*ed));ds=math.sqrt((ed-s*mu)/(2*ed));Ud=np.array([[-ds,dc],[dc,ds]],complex)
            da=np.array([np.zeros((2,2)),(hd+s*mu*np.eye(2))/2,(hd-s*mu*np.eye(2))/2])
            check('dark_free_diagonalization_'+str(s),maximum(hd@Ud-Ud*np.array([-ed,ed])))
            dark.append(dict(U=Ud,A=np.array([Ud.conj().T@ar@Ud for ar in da]),h=hd,p=np.diag([1,0])))
        for j,darkfactor in enumerate(dark):
            for r in range(3):
                ar=darkfactor['A'][r];check('dark_vertex_diagonal_'+str(j)+'_'+str(r),maximum(ar-np.diag(np.diag(ar))))
                check('dark_vertex_conservation_'+str(j)+'_'+str(r),maximum(ar@np.diag([-ed,ed])-np.diag([-ed,ed])@ar))
        block_trace=np.array([[2*sum(np.trace(f['A'][r]@f['A'][t]) for f in factors)+10*sum(np.trace(d['A'][r]@d['A'][t]) for d in dark) for t in range(3)] for r in range(3)])
        check('graph_block_trace_products',maximum(block_trace-np.array([[np.trace(ar@at) for at in A] for ar in A])))
        brightA=np.array([[factors[c//2]['A'][r] for c in range(4)] for r in range(3)])
        brightP=np.array([np.diag(factors[c//2]['reference']) for c in range(4)])
        alpha=np.array([2*sum(np.trace(np.diag(f['reference'])@f['A'][r]) for f in factors)+10*sum(np.trace(d['p']@d['A'][r]) for d in dark) for r in range(3)])
        beta=np.array([[2*sum(np.trace(np.diag(f['reference'])@f['A'][t]@np.diag(f['reference'])@f['A'][r]) for f in factors)+10*sum(np.trace(d['p']@d['A'][t]@d['p']@d['A'][r]) for d in dark) for t in range(3)] for r in range(3)])
        check('reference_alpha_graph_blocks',maximum(alpha-np.array([np.trace(P0@a) for a in A])))
        check('reference_beta_graph_blocks',maximum(beta-np.array([[np.trace(P0@at@P0@ar) for at in A] for ar in A])))
        product_file('graph_v1',sites=graph_data['positions'],alpha=graph_data['alpha'],beta=graph_data['beta'],h0=h,A=A,P0=P0,Rplus=Rplus,Pdark=Pdark,
            free_eigenvalues=w,free_eigenvectors=V,T=np.array(Ts),L=np.array(bright),shell_masks=graph_data['shell_masks'],edges=graph_data['edges'])
        masks1,car1=fixed_number_bilinears(np,1);masks2,car2=fixed_number_bilinears(np,2)
        product_file('factors_v1',h_bright=np.array([f['h'] for f in factors]),U_bright=np.array([f['U'] for f in factors]),a_bright=np.array([f['A'] for f in factors]),
            p_bright=np.array([np.diag(f['reference']) for f in factors]),h_dark=np.array([d['h'] for d in dark]),U_dark=np.array([d['U'] for d in dark]),
            p_dark=np.array([d['p'] for d in dark]),a_dark=np.array([d['A'] for d in dark]),masks_n1=masks1,masks_n2=masks2,bilinears_n1=car1,bilinears_n2=car2,
            alpha_ref=alpha,beta_ref=beta,K=K)
        report['completion'].update(graph=True,factors=True);save()
        k=132*mu*mu/245;keff=19/10;tcoef=math.sqrt(3)*(eps-mu)**1.5/(14*math.sqrt(eps))
        c2target=-27*keff**2*(eps-mu)**6/(38416*eps**2*(eps+mu))
        pplus=V[:,26:27];D1=pplus@pplus.conj().T
        cancellations=np.array([[np.trace(ar@D1)*np.trace(at@D1)-np.trace(ar@D1@at@D1) for at in A] for ar in A])
        check('rank_one_termwise_cancel',maximum(cancellations))
        frozenq=np.array([np.trace(a@Rplus) for a in A]);frozenX=sum(K[r,t]*np.trace(A[r]@Rplus@A[t]@Rplus) for r in range(3) for t in range(3))
        check('frozen_H_binding',abs(real(g/2*frozenq@K@frozenq)-2*g*k));check('frozen_HF_binding',abs(real(g/2*(frozenq@K@frozenq-frozenX))-g*k))
        report['known_answers']=dict(c2_reference_radical=c2target,pair_norm_squared=6*keff**2*tcoef**4,free_reference_variance=0.0,
            pair_reference_variance=g*g*6*keff**2*tcoef**4,computed_linear_charge_variance=norm((np.eye(52)-P0)@h@P0)**2,frozen_H_binding=2*g*k,frozen_HF_binding=g*k,
            disclosed_HF_ratio=0.5000102342,disclosed_exact_ratio=0.5000377984)
        report['completion']['frozen_rank_one']=True
        report['analytic_bounds']=dict(omitted_sector_margin=(ed-mu)-2458231/8192000,W_norm_rational='2458231/4000',twice_g_W='2458231/8192000',free_gap=ed-mu,
            sectors='Nf0 and Nf2; fixed dark occupations, whole-sector identity by analytic norm bound')
        # Full graph SCF; use its fixed free basis to avoid diagonal cancellation.
        Afree=np.array([V.conj().T@a@V for a in A]);Pfree=np.diag([1.0]*26+[0.0]*26);Rfree=np.diag([0.0]*26+[1.0]*2+[0.0]*24)
        S=70477/2000;Bbound=9*g*S;Lbound=2*g*S;gapbound=ed-mu-2*Bbound;qmap=Lbound/gapbound
        report['analytic_bounds'].update(S=S,field_bound=Bbound,Lipschitz=Lbound,gap_bound=gapbound,projector_contraction=qmap,
            interpolation='common h0 neighborhood; no arbitrary interpolation-gap inference')
        if not qmap<9/91:raise ArithmeticError('Contraction premise failed')
        def source_and_exchange(D):
            q=np.array([real(np.trace(ar@D)) for ar in Afree]);direct=float(q@K@q)
            exchange=sum(K[r,t]*real(np.trace(Afree[r]@D@Afree[t]@D)) for r in range(3) for t in range(3))
            return q,direct,exchange
        def field(C,exchange):
            D=C-Pfree;q,direct,X=source_and_exchange(D);F=np.diag(w).astype(complex)-g*sum((Afree[r]*sum(K[r,t]*q[t] for t in range(3)) for r in range(3)),np.zeros((52,52),complex))
            if exchange:F+=g*sum((K[r,t]*Afree[r]@D@Afree[t] for r in range(3) for t in range(3)),np.zeros((52,52),complex))
            return F
        for label,do_exchange in (('H',False),('HF',True)):
            C=Pfree+Rfree;history=[];iteration_C=[];iteration_q=[];iteration_pieces=[];done=False
            for iteration in range(128):
                F=field(C,do_exchange);ew,Q=la.eigh(F);new=Q[:,:28]@Q[:,:28].conj().T
                distance=norm(new-C);history.append(dict(iteration=iteration,projector_change=distance,gap=float(ew[28]-ew[27])))
                C=new;generated_Q=Q.copy()
                qo,qe=Q[:,:28],Q[:,28:];dc=np.zeros((52,52),complex)
                dc[:28,:28]=-qe[:28]@qe[:28].conj().T;dc[28:,28:]=qo[28:]@qo[28:].conj().T
                dc[:28,28:]=qo[:28]@qo[28:].conj().T;dc[28:,:28]=dc[:28,28:].conj().T
                iq,idir,ix=source_and_exchange(Rfree+dc);chem=(mu+ed)/2
                ik=float(np.dot(w[28:]-chem,np.sum(np.abs(qo[28:])**2,axis=1))+np.dot(chem-w[:28],np.sum(np.abs(qe[:28])**2,axis=1)))
                iteration_C.append(C.copy());iteration_q.append(iq.copy());iteration_pieces.append([ik,idir,ix,2*mu+ik-g/2*(idir-(ix if do_exchange else 0))])
                if distance<=1e-13:done=True;break
            F=field(C,do_exchange);ew,Q=la.eigh(F);Qo,Qe=Q[:,:28],Q[:,28:];mapped=Qo@Qo.conj().T
            residual=norm(mapped-C)
            # Final readout uses C and factors from its generating eigensolve, rather than silently accepting the new mapped state.
            Qocc,Qempty=generated_Q[:,:28],generated_Q[:,28:]
            dC=np.zeros((52,52),complex);dC[:28,:28]=-Qempty[:28]@Qempty[:28].conj().T
            dC[28:,28:]=Qocc[28:]@Qocc[28:].conj().T;dC[:28,28:]=Qocc[:28]@Qocc[28:].conj().T;dC[28:,:28]=dC[:28,28:].conj().T
            D=Rfree+dC;xi=(mu+ed)/2
            gain=np.sum(np.abs(Qocc[28:])**2,axis=1);loss=np.sum(np.abs(Qempty[:28])**2,axis=1)
            kinetic=float(np.dot(w[28:]-xi,gain)+np.dot(xi-w[:28],loss));q,direct,X=source_and_exchange(D)
            binding=g/2*(direct-(X if do_exchange else 0))-kinetic;distance_bound=residual/(1-qmap)
            truncation=(2*eps+2*Bbound+Lbound)*distance_bound**2/2
            diagnostics=dict(converged=done and residual<=1e-12,iterations=history,projector_residual=residual,
                commutator=norm(F@C-C@F),idempotency=norm(C@C-C),rank_defect=abs(real(np.trace(C))-28),
                hermiticity=norm(C-C.conj().T),gap=float(ew[28]-ew[27]),eigen_residual=norm(F@Q-Q*ew),
                eigen_orthogonality=norm(Q.conj().T@Q-np.eye(52)),gain_minus_loss=float(gain.sum()-loss.sum()),
                kinetic_deformation=kinetic,shell_sources=q.tolist(),direct=direct,exchange=X,pair_binding=g/2*(direct-(X if do_exchange else 0)),binding=binding,
                conditional_truncation_energy_bound=truncation,full_matrix_arithmetic_enclosure_available=False)
            report['scf'][label]=diagnostics
            exchange_matrix=np.array([[np.trace(Afree[r]@D@Afree[t]@D) for t in range(3)] for r in range(3)])
            product_file('scf_'+label.lower()+'_v1',C=V@C@V.conj().T,D_conditioned=V@D@V.conj().T,F=V@F@V.conj().T,mapped_C=V@mapped@V.conj().T,
                occupied_factor=V@Qocc,empty_factor=V@Qempty,q=q,exchange_matrix=exchange_matrix,
                C_free=C,D_free_conditioned=D,F_free=F,field_eigenvectors=V@Q,field_eigenvalues=ew,
                conditioned_deltaC=dC,occupation_gain=gain,occupation_loss=loss,
                iteration_C_free=np.array(iteration_C),iteration_q=np.array(iteration_q),iteration_energy_pieces=np.array(iteration_pieces))
            diagnostics['iteration_energy_columns']=['kinetic_cost','direct','exchange','total_energy']
            if not diagnostics['converged']:raise ArithmeticError('SCF incomplete '+label)
            report['completion']['SCF_'+label]=True;save()
        for Nf,occupations in ((0,(1,1,2,2)),(2,(2,2,2,2))):
            labels,E,local=tensor_bilinears(np,occupations);eye=np.eye(81)
            for ci,loc in enumerate(local):
                n=occupations[ci];check('CAR_number_'+str(Nf)+'_'+str(ci),maximum(sum(loc[i,i] for i in range(3))-n*np.eye(3)))
                cr=0.0
                for i,j,k0,l in product(range(3),repeat=4):
                    rhs=(loc[i,l] if j==k0 else 0)-(loc[k0,j] if l==i else 0)
                    cr=max(cr,maximum(loc[i,j]@loc[k0,l]-loc[k0,l]@loc[i,j]-rhs))
                check('CAR_commutator_'+str(Nf)+'_'+str(ci),cr)
            def O(br,dk):
                out=sum((np.einsum('ij,ijab->ab',br[c],E[c]) for c in range(4)),np.zeros((81,81),complex))
                return out+10*sum(np.trace(dark[s]['p']@dk[s]) for s in range(2))*eye
            Or=[O(brightA[r],np.array([d['A'][r] for d in dark])) for r in range(3)]
            W=np.zeros((81,81),complex)
            for r,t in product(range(3),repeat=2):
                br1=np.array([brightA[r,c]@(np.eye(3)-brightP[c])@brightA[t,c] for c in range(4)])
                br2=np.array([brightA[t,c]@brightP[c]@brightA[r,c] for c in range(4)])
                dk1=np.array([d['A'][r]@(np.eye(2)-d['p'])@d['A'][t] for d in dark])
                dk2=np.array([d['A'][t]@d['p']@d['A'][r] for d in dark])
                W-=K[r,t]/2*((Or[r]-alpha[r]*eye)@(Or[t]-alpha[t]*eye)-O(br1,dk1)+O(br2,dk2)-beta[r,t]*eye)
            check('W_hermiticity_'+str(Nf),maximum(W-W.conj().T));W=(W+W.conj().T)/2
            free=np.zeros(81)
            for ci in range(4):
                energy=factors[ci//2]['energies'];reference_mask=1 if occupations[ci]==1 else 3
                reference=sum(energy[j] for j in range(3) if reference_mask&(1<<j))
                for v,mask in enumerate(labels[:,ci]):free[v]+=sum(energy[j] for j in range(3) if mask&(1<<j))-reference
            if not free[0]==0 or not np.all(free[1:]>0):raise ArithmeticError('Free ground indexing')
            c1=real(W[0,0]);wvec=W[1:,0];Ddiag=free[1:];coeff=-real(np.vdot(wvec,wvec/Ddiag))
            if Nf==2:check('Nf2_frozen_W00',abs(c1+k))
            if Nf==0:
                check('reference_scalar',abs(c1));check('reference_coefficient_relative',abs((coeff-c2target)/c2target),tolerance=1e-10)
                check('reference_pair_norm',abs(real(np.vdot(wvec,wvec))-6*keff**2*tcoef**4))
                singles=[]
                for idx in range(1,81):
                    degree=sum((int(labels[idx,c])^int(labels[0,c])).bit_count() for c in range(4))//2
                    if degree==1:singles.append(abs(W[idx,0]))
                check('reference_singles',max(singles,default=0.0))
            product_file('exact_nf'+str(Nf)+'_construction_v1',copy_counts=np.array(occupations),basis_masks=labels,h0_excitation=free,W=W)
            Cblock=np.diag(free[1:])+g*W[1:,1:];bvec=g*wvec;a0=g*c1
            # The declared whole-space norm supplies an initial scalar bracket.
            bound=g*2458231/4000;lo,hi=-bound,bound;root_trace=[]
            def root_eval(delta):
                M=Cblock-delta*np.eye(80);x=la.solve(M,bvec,assume_a='her');r=M@x-bvec
                gamma=float(np.min(np.real(np.diag(M))-(np.sum(np.abs(M),axis=1)-np.abs(np.diag(M)))))
                if gamma<=0:raise ArithmeticError('Nonpositive Schur complement lower bound')
                value=delta-a0+real(np.vdot(bvec,x));error=norm(bvec)*norm(r)/gamma
                return value,x,error,gamma,norm(r)
            flo,*_=root_eval(lo);fhi,*_=root_eval(hi)
            if not flo<=0<=fhi:raise ArithmeticError('Schur bracket')
            delta=a0;root_stop='iteration_cap'
            for iteration in range(96):
                fv,x,err,gamma,lr=root_eval(delta);root_trace.append(dict(iteration=iteration,delta=delta,f=fv,linear_residual=lr,conditional_scalar_error=err,gamma=gamma,lo=lo,hi=hi))
                if fv<0:lo=delta
                else:hi=delta
                correction=fv/(1+real(np.vdot(x,x)));proposal=delta-correction
                if abs(fv)<=1e-24 or proposal==delta:
                    root_stop='residual' if abs(fv)<=1e-24 else 'floating_stagnation';break
                delta=proposal if lo<proposal<hi else (lo+hi)/2
            fv,x,err,gamma,lr=root_eval(delta)
            if root_stop=='iteration_cap':raise ArithmeticError('Schur iteration incomplete')
            flo_final,*_=root_eval(lo);fhi_final,*_=root_eval(hi)
            if abs(fv)+err>1e-18:raise ArithmeticError('Conditional Schur accuracy failed')
            vector=np.concatenate(([1+0j],-x));vector/=la.norm(vector)
            H=np.diag(free)+g*W;eres=norm(H@vector-delta*vector);evals,evecs=la.eigh(H)
            Dbright=[];Dsector=[];copyC=[]
            for ci in range(4):
                pc=np.array([(int(labels[0,ci])>>i)&1 for i in range(3)]);Dlocal=np.zeros((3,3),complex)
                for i,j in product(range(3),repeat=2):
                    if i==j:
                        # Exact integer occupation differences condition both particles and holes.
                        Dlocal[i,i]=sum(float(abs(vector[v])**2)*(((int(mask)>>i)&1)-pc[i]) for v,mask in enumerate(labels[:,ci]))
                    else:Dlocal[j,i]=np.vdot(vector,E[ci,i,j]@vector)
                Dsector.append(Dlocal);copyC.append(np.diag(pc)+Dlocal)
                Dbright.append(Dlocal+np.diag(pc-factors[ci//2]['reference']))
            Dgraph=sum((bright[c]@Dbright[c]@bright[c].conj().T for c in range(4)),np.zeros((52,52),complex));CQ=P0+Dgraph
            qQ=np.array([real(np.trace(ar@Dgraph)) for ar in A]);XQ=sum(K[r,t]*real(np.trace(A[r]@Dgraph@A[t]@Dgraph)) for r in range(3) for t in range(3))
            pair_gaussian=-g/2*(float(qQ@K@qQ)-XQ);interaction=g*real(np.vdot(vector,W@vector));cumulant=interaction-pair_gaussian
            kinetic=real(np.vdot(vector,np.diag(free)@vector));rayleigh=real(np.vdot(vector,H@vector));variance=norm(H@vector-rayleigh*vector)**2;schur_residual_squared=norm(H@vector-delta*vector)**2
            report['exact'][str(Nf)]=dict(shift=delta,total_energy=Nf*mu+delta,c1=c1,c2=coeff,root_iterations=root_trace,
                scalar_residual=fv,root_stop_reason=root_stop,root_bracket=[lo,hi],root_bracket_values=[flo_final,fhi_final],conditional_accuracy_pass=True,conditional_root_error=abs(fv)+err,conditional_gamma=gamma,eigen_residual=eres,
                ordinary_eigensolve_difference=float(evals[0]-delta),variance=variance,rayleigh_shift=rayleigh,rayleigh_energy=Nf*mu+rayleigh,schur_residual_squared=schur_residual_squared,kinetic_deformation=kinetic,
                interaction_energy=interaction,gaussian_pair_energy=pair_gaussian,interaction_cumulant=cumulant,
                full_matrix_arithmetic_enclosure_available=False,sector='81-state fixed-copy CAR; whole-sector by analytic norm bound')
            if Nf==0:
                computed_pair_variance=g*g*real(np.vdot(wvec,wvec))
                report['exact']['0'].update(reference_shift_minus_c2g2=delta-c2target*g*g,cumulant_minus_2c2g2=cumulant-2*c2target*g*g,
                    computed_reference_pair_variance=computed_pair_variance,known_reference_pair_variance=g*g*6*keff**2*tcoef**4,
                    computed_linear_charge_variance=norm((np.eye(52)-P0)@h@P0)**2)

            extra={}
            if Nf==0:
                pair_indices=np.array([36,30,28,12,10,4],dtype=int)
                check('reference_pair_amplitudes',maximum(np.abs(W[pair_indices,0])-keff*tcoef**2))
                check('reference_pair_denominators',maximum(free[pair_indices]-2*(eps+mu)))
                outside=[i for i in range(81) if i not in set(pair_indices)]
                check('reference_pair_offsupport',maximum(W[outside,0]))
                extra=dict(W_reference=W[:,0],pair_indices=pair_indices,pair_amplitudes=W[pair_indices,0],pair_denominators=free[pair_indices],c2_terms=-np.abs(W[pair_indices,0])**2/free[pair_indices])
            root_columns=list(root_trace[0]);report['exact'][str(Nf)]['root_history_columns']=root_columns
            exchange_matrix=np.array([[np.trace(A[r]@Dgraph@A[t]@Dgraph) for t in range(3)] for r in range(3)])
            product_file('exact_nf'+str(Nf)+'_v1',copy_counts=np.array(occupations),basis_indices=np.array(list(product(range(3),repeat=4))),basis_masks=labels,
                h0_excitation=free,W=W,H_shifted=H,ground_vector=vector,eigen_residual=H@vector-delta*vector,rayleigh_residual=H@vector-rayleigh*vector,W_ground_vector=W@vector,
                copy_covariance=np.array(copyC),copy_covariance_delta=np.array(Dsector),copy_reference_delta=np.array(Dbright),
                C=CQ,D_conditioned=Dgraph,q=qQ,exchange_matrix=exchange_matrix,
                root_history=np.array([[row[k] for k in root_columns] for row in root_trace]),root_bracket=np.array([lo,hi]),solve_vector=x,solve_residual=(Cblock-delta*np.eye(80))@x-bvec,
                ordinary_eigenvalues=evals,ordinary_eigenvectors=evecs,reference_excitation_vector=wvec,reference_denominators=Ddiag,**extra)
            report['completion']['exact_Nf'+str(Nf)]=True;save()
        BH=report['scf']['H']['binding'];BHF=report['scf']['HF']['binding'];BQ=report['exact']['0']['shift']-report['exact']['2']['shift']
        eBH=report['scf']['H']['conditional_truncation_energy_bound'];eHF=report['scf']['HF']['conditional_truncation_energy_bound']
        eBQ=sum(report['exact'][str(n)]['conditional_root_error'] for n in (0,2))
        if not BH>eBH:raise ArithmeticError('Hartree denominator not separated from conditional error')
        report['conditional_ratio_errors']=dict(HF=(eHF+abs(BHF/BH)*eBH)/(BH-eBH),exact_sub=(eBQ+abs(BQ/BH)*eBH)/(BH-eBH),
            scope='conditional Schur solve and SCF truncation only; full matrix/arithmetic evaluation errors unavailable; no rigorous enclosure')
        report['ratios']=dict(B_H=BH,B_HF=BHF,B_exact_sub=BQ,HF_over_H=BHF/BH,exact_sub_over_H=BQ/BH,
            disclosed_target_HF=0.5000102342,disclosed_target_exact_sub=0.5000377984,
            difference_HF=BHF/BH-0.5000102342,difference_exact_sub=BQ/BH-0.5000377984,
            rigorous_enclosure_available=False,scope='controlled binary64 arithmetic; conditional solve/truncation errors exclude full assembly/evaluation errors')
        report['completion'].update(graph=True,frozen_rank_one=True,SCF_H=True,SCF_HF=True,exact_Nf0=True,exact_Nf2=True,native_HF=False)
        report['status']='completed_controlled_scalar_fixture'
    except Exception as e:report.update(status='incomplete',error=type(e).__name__+': '+str(e))
    finally:
        signal.alarm(0);resource.setrlimit(resource.RLIMIT_CPU,(2400,2400));save()
    print(json.dumps(dict(status=report['status'],completion=report['completion'])))
    if report['status']!='completed_controlled_scalar_fixture':raise SystemExit(1)


if __name__=='__main__':main()
