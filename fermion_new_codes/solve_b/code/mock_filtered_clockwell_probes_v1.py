"""UNRUN #145 bounded probe inputs and independent weighted envelopes.

No stage/propagation entry point. Full source and manufactured-output readings
by the original author/reviewer are prerequisites for numerical use.
"""
from fractions import Fraction as F
from types import MappingProxyType
import run_mock_vertex_admission_v1 as e
import run_mock_filtered_vertex_admission_v1 as admitted
import mock_filtered_clockwell_v1 as clock


def ordinary_manifest(basis):
    """Exactly the preregistered sixteen collinear and three valley rows."""
    if basis.role!='weighted' or basis.n not in (2,3,7):raise ValueError('weighted probe basis')
    rows={}
    for label,parameters in e.fixed_roster(basis.n).items():
        hs,ls,ih,il,null=parameters
        collinear='_collinear_1_' in label
        valley='_valley_' in label and label.split('_valley_')[1][:4] in ('0000','0101','0001') and ih==il==(1,-1)
        if collinear or valley:
            if null:raise ValueError('bounded ordinary roster unexpectedly null')
            rows[label]=parameters
    if len(rows)!=19:raise ValueError('independently fixed nineteen-row roster')
    return MappingProxyType(rows)


def tangent_manifest(basis):
    if basis.role!='weighted' or basis.n not in (2,3,7):raise ValueError('weighted tangent basis')
    return tuple((branch,spin) for branch in (0,1) for spin in (1,-1))


class OrdinaryInputs:
    def __init__(self,basis):
        self.basis=basis;self.manifest=ordinary_manifest(basis);self.cache=admitted.LegCache(basis)

    def make(self,label,parameters,*,provided_legs=None):
        if label not in self.manifest or parameters!=self.manifest[label]:raise ValueError('intrinsic ordinary label/parameters')
        if type(parameters) is not tuple or len(parameters)!=5 or type(parameters[4]) is not bool or any(type(spin) is not int for pair in parameters[2:4] for spin in pair):raise ValueError('literal ordinary parameter types')
        hs,ls,ih,il,null=self.manifest[label]
        descriptors=((label+'H1',0,hs[0],ih[0]),(label+'H2',0,hs[1],ih[1]),
                     (label+'L1',1,ls[0],il[0]),(label+'L2',1,ls[1],il[1]))
        if provided_legs is not None and len(provided_legs)!=4:raise ValueError('four physical legs')
        pairs=[self.cache.make(d,d,provided=None if provided_legs is None else provided_legs[i]) for i,d in enumerate(descriptors)]
        legs=tuple(x[0] for x in pairs);ts=tuple(x[1] for x in pairs)
        wh=e.wedge(legs[0],legs[1],self.basis.pairs);wl=e.wedge(legs[2],legs[3],self.basis.pairs)
        gh=e.gram(legs[0],legs[1]);gl=e.gram(legs[2],legs[3]);e.enforce_pair_classification(gh[0],gl[0],False)
        return dict(label=label,parameters=parameters,descriptors=descriptors,legs=legs,local=ts,
                    heavy=wh,light=wl,gramH=gh,gramL=gl,C=e.local_C(ts,e.pairing_matrices()),
                    local_norm=admitted.local_norm(ts),phases=tuple(spec[0] for spec in hs+ls))


def envelope(basis,phases):
    """Independent spatial numerator; reports N_j times the BARE allocation."""
    if len(phases)!=4:raise ValueError('four ordered phases')
    G=clock.kernel();bare={j:e.ZERO for j in range(-1,basis.n+1)};total=e.ZERO
    p={}
    for leg,z in enumerate(phases):
        for band in (0,1):
            for j in bare:p[leg,band,j]=sum((c*z**x for x,c in clock.stencil(basis,band,j).items()),e.ZERO)
    for band in (0,1):
        for j in bare:
            for k in bare:
                if abs(j-k)>2:continue
                value=G[j-k]*(p[2,band,j]*p[3,band,j]).conj()*p[0,band,k]*p[1,band,k]
                bare[j]+=value/2;bare[k]+=value/2
                total+=value*(clock.lapse(basis,j)+clock.lapse(basis,k))/2
    weighted={j:value*clock.lapse(basis,j) for j,value in bare.items()}
    return dict(total=total,centres=weighted,flat_centres=bare,header=clock.TABLE_HEADER)


def tangent_inputs(basis,descriptor,*,provided_derivative=None,provided_light=None):
    if type(descriptor) is not tuple or len(descriptor)!=2 or any(type(x) is not int for x in descriptor) or descriptor not in tangent_manifest(basis):raise ValueError('intrinsic tangent branch/spin')
    branch,spin=descriptor;z=e.C(F(4,5),F(3,5));z1,z2=(z,z.conj()) if branch==0 else (-z.conj(),-z)
    h1=e.template(basis.n,0,e.ONE,F(0),spin);h2=e.template(basis.n,0,e.ONE,F(0),spin)
    l1=e.template(basis.n,1,z1,F(1),spin);l2=e.template(basis.n,1,z2,F(-1),spin)
    d1=e.derivative_template(basis.n,0,spin,0);d2=e.derivative_template(basis.n,0,spin,1)
    dw=e.vecadd(e.wedge(d1,h2,basis.pairs),e.wedge(h1,d2,basis.pairs));wl=e.wedge(l1,l2,basis.pairs)
    literal=e.tangent_literal(basis.n,spin,basis.pairs,spin)
    if dw!=literal:raise ValueError('complete derivative including phase disagrees with literal')
    if provided_derivative is not None and list(provided_derivative)!=dw:raise ValueError('detached full derivative')
    if provided_light is not None and list(provided_light)!=wl:raise ValueError('detached outgoing tangent pair')
    return dict(descriptor=descriptor,phases=(z1,z2),heavy=(h1,h2),light_legs=(l1,l2),
                derivative_legs=(d1,d2),heavy_derivative=dw,light=wl,literal_derivative=literal,
                prepared_state_amplitude=False,status='raw derivative; no normalization of null family')


def tangent_envelope(basis,descriptor):
    """Differentiate the literal M bilinears and their full spatial phases."""
    if type(descriptor) is not tuple or len(descriptor)!=2 or any(type(x) is not int for x in descriptor) or descriptor not in tangent_manifest(basis):raise ValueError('fixed tangent envelope descriptor')
    branch,spin=descriptor;z=e.C(F(4,5),F(3,5));z1,z2=(z,z.conj()) if branch==0 else (-z.conj(),-z)
    Ms=e.pairing_matrices();G=clock.kernel();bare={j:e.ZERO for j in range(-1,basis.n+1)};total=e.ZERO
    t1=e.local_template(F(0),spin);t2=e.local_template(F(0),spin)
    l1=e.local_template(F(1),spin);l2=e.local_template(F(-1),spin)
    delta1=e.vecadd(e.local_template(F(1),spin),e.scale(t1,-1))
    delta2=e.scale(e.vecadd(e.local_template(F(1),spin),e.scale(t2,-1)),-1)
    def bilinear(a,M,b):return sum((a[i]*M[i][j]*b[j] for i in range(4) for j in range(4)),e.ZERO)
    for band in (0,1):
        for j in bare:
            L=clock.stencil(basis,band,j)
            p1=sum((c*z1**x for x,c in L.items()),e.ZERO);p2=sum((c*z2**x for x,c in L.items()),e.ZERO)
            for k in bare:
                if abs(j-k)>2:continue
                R=clock.stencil(basis,band,k);p=sum(R.values(),F(0));dx=sum((c*x for x,c in R.items()),F(0))
                h1=e.scale(t1,p);h2=e.scale(t2,p)
                dh1=e.vecadd(e.scale(delta1,p),e.scale(t1,e.J*F(6,5)*dx))
                dh2=e.vecadd(e.scale(delta2,p),e.scale(t2,-e.J*F(6,5)*dx))
                value=4*G[j-k]*sum(((1 if mu==0 else -1)*(bilinear(l1,M,l2)*p1*p2).conj()*
                    (bilinear(dh1,M,h2)+bilinear(h1,M,dh2)) for mu,M in enumerate(Ms)),e.ZERO)
                bare[j]+=value/2;bare[k]+=value/2
                total+=value*(clock.lapse(basis,j)+clock.lapse(basis,k))/2
    return dict(total=total,centres={j:value*clock.lapse(basis,j) for j,value in bare.items()},
                flat_centres=bare,header=clock.TABLE_HEADER,descriptor=descriptor)
