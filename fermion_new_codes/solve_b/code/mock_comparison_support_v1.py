"""UNRUN exact charge/support algebra from addendum section four.

Definition-only draft. No numerical spectral/export implementation, controls,
physical context adapter or execution entry point is supplied here. It is not
part of the size-two/three comparison runner or its control admission.
"""
from fractions import Fraction as F
import run_mock_vertex_admission_v1 as e


def _zero_matrix(size):
    if type(size) is not int or not 0<=size<=112:raise ValueError('small charge-block allocation cap')
    return [[e.ZERO for _ in range(size)] for _ in range(size)]


def exact_charge_support(basis,state,output):
    """Complete canonical N2 coefficients -> exact k0/k1 and k2 Gram.

    Every retained row is determined by the complete sector, never its observed
    nonzeros. R is an ordinary qubit preceding the graded O/K permutation.
    The unconstructed k2 factor is represented only by two linear vectors.
    """
    import math
    import mock_unitary_lead_v1 as q
    from mock_comparison_readouts_v1 import _coefficients
    if basis.number!=2 or basis.modes>56 or len(basis.states)!=math.comb(basis.modes,2):
        raise ValueError('complete N2 sector up to the separately authorized size-seven sector required')
    if len(set(basis.states))!=len(basis.states) or any(mask<0 or mask>>basis.modes or mask.bit_count()!=2 for mask in basis.states):
        raise ValueError('noncanonical/incomplete occupation roster')
    values,norm=_coefficients(basis,state)
    O=q.labels(output,basis.modes);K=tuple(j for j in range(basis.modes) if j not in O)
    rank={j:i for i,j in enumerate(O+K)}
    maps=[];keys={0:set(),1:set(),2:set()}
    for mask in basis.states:
        occupied=[j for j in range(basis.modes) if mask>>j&1]
        sign=-1 if rank[occupied[0]]>rank[occupied[1]] else 1
        o=sum(((mask>>j)&1)<<i for i,j in enumerate(O))
        k=sum(((mask>>j)&1)<<i for i,j in enumerate(K))
        number=k.bit_count();keys[number].add(k);maps.append((o,k,number,sign))
    rr=[[sum((values[r][j]*values[s][j].conj() for j in range(len(basis.states))),e.ZERO)/norm
         for s in range(2)] for r in range(2)]
    if rr[0][0]+rr[1][1]!=e.C(1):raise ValueError('whole-state reference marginal trace')
    small={};vectors=[{},{}]
    for number in (0,1):
        retained=tuple(sorted(keys[number]));dimension=len(retained);size=2*dimension
        matrix=_zero_matrix(size);index={k:i for i,k in enumerate(retained)};groups={}
        for r in range(2):
            for j,(o,k,count,sign) in enumerate(maps):
                if count!=number or not values[r][j]:continue
                groups.setdefault(o,[]).append((r*dimension+index[k],sign*values[r][j]))
        for group in groups.values():
            for i,a in group:
                for j,b in group:matrix[i][j]+=a*b.conj()/norm
        marginal=[[sum((matrix[r*dimension+i][r*dimension+j] for r in range(2)),e.ZERO)
                   for j in range(dimension)] for i in range(dimension)]
        difference=[[matrix[i][j]-rr[i//dimension][j//dimension]*marginal[i%dimension][j%dimension]
                     for j in range(size)] for i in range(size)]
        if sum((difference[i][i] for i in range(size)),e.ZERO)!=e.ZERO:
            raise ValueError('exact charge-block difference trace')
        small[number]=dict(particle_number=number,retained_keys=retained,dimension=size,
            joint_exact=matrix,complement_exact=marginal,difference_exact=difference,
            reference_exact=rr,groups=groups)
    for r in range(2):
        for j,(o,k,count,sign) in enumerate(maps):
            if count==2:
                if o:raise ValueError('k2 block must have O vacuum')
                vectors[r][k]=sign*values[r][j]
    retained2=tuple(sorted(keys[2]))
    if any(set(vector)!=set(retained2) for vector in vectors):raise ValueError('incomplete exact k2 vector coordinates')
    a=sum((vectors[0][k].abs2() for k in retained2),F(0))/norm
    b=sum((vectors[0][k].conj()*vectors[1][k] for k in retained2),e.ZERO)/norm
    c=sum((vectors[1][k].abs2() for k in retained2),F(0))/norm
    determinant=a*c-b.abs2()
    if min(a,c,determinant)<0:raise ValueError('negative exact Gram')
    if a:
        span=2 if determinant else 1;pivot='v0'
        coordinates=(('sqrt(a)','0'),('b/sqrt(a)','sqrt(Delta/a)' if determinant else '0'))
    elif c:
        if b or determinant:raise ValueError('zero a with nonzero Gram cross/determinant')
        span=1;pivot='v1';coordinates=(('0',),('sqrt(c)',))
    else:
        if b or determinant:raise ValueError('zero Gram diagonal with nonzero cross/determinant')
        span=0;pivot='none';coordinates=((),())
    compressed_dimension=2*span;full_dimension=2*len(retained2)
    if compressed_dimension>4 or compressed_dimension>full_dimension:raise ValueError('exact support dimension')
    block2=dict(particle_number=2,retained_keys=retained2,vectors_unnormalized_exact=vectors,
        normalization_exact=norm,a=a,b=b,c=c,Delta=determinant,span_dimension=span,pivot=pivot,
        coordinate_formula=coordinates,compressed_dimension=compressed_dimension,
        full_dimension=full_dimension,unconstructed_zero_multiplicity=full_dimension-compressed_dimension,
        reference_exact=rr,density_nonzero_eigenvalue=a+c,
        difference_formula='|psi2><psi2| - rho_R_whole tensor (v0 v0^dag + v1 v1^dag)',
        entropy_formula='-(a+c)*log(a+c), with 0log0=0')
    return dict(output=O,complement=K,norm_exact=norm,reference_exact=rr,small_blocks=small,k2=block2,
        complete_joint_dimension=sum(block['dimension'] for block in small.values())+full_dimension,
        truncated=False,rank_rule='exact normalized dyadic Gram; no tolerance or physical-state rank inference')
