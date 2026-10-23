"""UNRUN #145 exact source association and normalized dyadic source action.

No physical midpoint evaluation or propagation. Source creation requires the
separate full sparse reference; the action requires an independently supplied
expected sample identity, not a metadata object copied from its argument.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction as F
import hashlib
import math
import run_mock_vertex_admission_v1 as e
import mock_filtered_clockwell_v1 as clock
import mock_filtered_joining_v1 as joining


def matrix_rows(matrix,rows,columns):
    out=[]
    for key,value in matrix.items():
        if not isinstance(key,tuple) or len(key)!=2 or any(type(x) is not int for x in key):raise ValueError('exact matrix locator')
        i,j=key
        if not 0<=i<rows or not 0<=j<columns or not isinstance(value,e.C) or not value:raise ValueError('coalesced exact sparse matrix')
        out.append((i,j,value))
    return tuple(sorted(out))


def fingerprint(value):
    return hashlib.sha256(repr(value).encode('ascii')).hexdigest()


def exact_vector(vector,dimension):
    if len(vector)!=dimension:raise ValueError('complete source input dimension')
    # Only exact rationals are admitted. IEEE raw values must first be exported
    # as their literal dyadic Fractions, with their own saved state error.
    if any(not isinstance(z,e.C) for z in vector):raise ValueError('literal exact dyadic source input')
    return tuple(vector)


@dataclass(frozen=True)
class SampleIdentity:
    operator: str
    basis: str
    preparation: str
    clock: F
    state: str
    epsilon: F

    def __post_init__(self):
        if any(type(v) is not str or not v for v in (self.operator,self.basis,self.preparation,self.state)):raise ValueError('sample identity fields')
        if not isinstance(self.clock,F) or self.clock<0 or self.clock>4:raise ValueError('exact sample clock')
        if not isinstance(self.epsilon,F) or self.epsilon<0:raise ValueError('nonnegative exact state allowance')


@dataclass(frozen=True)
class SourceState:
    identity: SampleIdentity
    vector: tuple

    @classmethod
    def bind(cls,vector,*,operator,basis,preparation,clock_value,epsilon):
        vector=tuple(vector)
        if not vector or any(not isinstance(z,e.C) for z in vector):raise ValueError('exact source state')
        identity=SampleIdentity(operator,basis,preparation,clock_value,fingerprint(vector),epsilon)
        return cls(identity,vector)


@dataclass(frozen=True)
class JoiningSource:
    n: int
    U: int
    basis_states: tuple
    long_states: tuple
    embedding: tuple
    H_rows: tuple
    Hlong_rows: tuple
    S_rows: tuple
    Q_rows: tuple
    patterns: tuple
    operator: str
    basis: str
    seal: str

    def payload(self):
        return (self.n,self.U,self.basis_states,self.long_states,self.embedding,self.H_rows,self.Hlong_rows,self.S_rows,self.Q_rows,self.patterns,self.operator,self.basis)

    def verify(self):
        if self.seal!=fingerprint(self.payload()):raise ValueError('mutated bound joining source')
        return self

    @classmethod
    def bind(cls,tested,reference):
        short=tested['short'];long=tested['long'];U=tested['U']
        joining.flat.validate_U(U);joining.flat.validate_U(reference['U'])
        if short.role!='weighted' or short.n not in (2,3):raise ValueError('source short role/size')
        if long!=clock.StaticBasis.make(short.n+4,'joining_reference'):raise ValueError('full extended basis')
        if reference['short']!=short or reference['long']!=long or reference['U']!=U:raise ValueError('independent source size/strength')
        for route in (tested,reference):
            if route['empty_extension'] is not True or type(route['added_sites']) is not int or route['added_sites']!=4:raise ValueError('four-site empty source extension')
            if route['profile_short']!=clock.profile(short) or route['profile_long']!=clock.profile(long):raise ValueError('declared source clocks')
        if tested['internal_difference']:raise ValueError('short compression mismatch')
        if tested['S']!=reference['S'] or tested['Q']!=reference['Q']:raise ValueError('actual patterned source differs from independent full reference')
        expected_embedding=tuple({mask:i for i,mask in enumerate(long.states)}[mask] for mask in short.states)
        if tuple(reference['embedding'])!=expected_embedding:raise ValueError('canonical empty embedding')
        Hlong_rows=matrix_rows(reference['H_long'],long.dimension,long.dimension)
        matrix_rows(reference['H_short'],short.dimension,short.dimension)
        reverse={j:i for i,j in enumerate(expected_embedding)};retained_S={}
        for (i,j),value in reference['H_long'].items():
            if j in reverse:e.add_entry(retained_S,(i,reverse[j]),value)
        for (i,j),value in reference['H_short'].items():e.add_entry(retained_S,(expected_embedding[i],j),-value)
        if retained_S!=reference['S'] or joining._gram(retained_S,long.dimension)!=reference['Q']:
            raise ValueError('retained Hamiltonians/embedding detached from actual reference S/Q')
        validate_Q_axis(reference['Q'])
        # Reconstruct every pattern contribution. This prevents a detached Q/S
        # from masking omitted patterns, crossing quartics or CAR mutations.
        S={};Q={};index={mask:i for i,mask in enumerate(long.states)};patterns=[]
        lower={0:(0,),1:tuple(1<<j for j in range(short.modes))}
        if tested['pattern_target_bases']!=lower:raise ValueError('complete lower-number pattern target bases')
        for P,O in sorted(tested['patterns'].items()):
            if not isinstance(P,tuple) or len(P) not in (1,2) or P!=tuple(sorted(set(P))) or any(type(j) is not int or not short.modes<=j<long.modes for j in P):raise ValueError('canonical nonempty outside pattern')
            rows=matrix_rows(O,len(lower[2-len(P)]),short.dimension)
            if not rows:raise ValueError('empty pattern must be coalesced away')
            outside=sum(1<<j for j in P)
            for i,j,z in rows:e.add_entry(S,(index[lower[2-len(P)][i]|outside],j),z)
            Q=e.matrix_add(Q,joining._gram(O,len(lower[2-len(P)])))
            patterns.append((P,rows))
        if S!=reference['S'] or Q!=reference['Q']:raise ValueError('pattern/operator interference association')
        H_tested=joining.monomial_matrix(short,tested['short_terms'])
        if H_tested!=reference['H_short']:raise ValueError('actual short target operator')
        Hrows=matrix_rows(H_tested,short.dimension,short.dimension)
        Srows=matrix_rows(S,long.dimension,short.dimension);Qrows=matrix_rows(Q,short.dimension,short.dimension)
        basis=fingerprint((short.n,short.modes,short.states,short.pairs,short.role))
        operator=fingerprint((clock.SPEC_SHA,clock.TABLE_HEADER,U,basis,Hrows))
        payload=(short.n,U,short.states,long.states,expected_embedding,Hrows,Hlong_rows,Srows,Qrows,tuple(patterns),operator,basis)
        return cls(*payload,fingerprint(payload))

    def norm_ratio(self,state,expected,save=None):
        self.verify()
        if type(state) is not SourceState or type(expected) is not SampleIdentity or state.identity!=expected:raise ValueError('independent planned source sample association')
        if expected.operator!=self.operator or expected.basis!=self.basis:raise ValueError('source operator/basis identity')
        raw=exact_vector(state.vector,len(self.basis_states))
        if fingerprint(raw)!=expected.state:raise ValueError('detached source state content')
        if save:save('exact_source_action_input',dict(source=self.seal,identity=asdict(expected),raw=raw))
        norm=e.dot(raw,raw)
        if norm.i or norm.r<=0:raise ValueError('positive exact raw norm')
        pattern_norms={};total=F(0)
        for P,rows in self.patterns:
            output={}
            for i,j,z in rows:e.add_entry(output,i,z*raw[j])
            value=sum((z.abs2() for z in output.values()),F(0));pattern_norms[P]=value;total+=value
        Q={(i,j):z for i,j,z in self.Q_rows};quadratic=e.matrix_element(Q,raw,raw)
        if save:save('exact_source_action_before_identity_check',dict(raw_norm=norm,pattern_squared_norms=pattern_norms,total=total,Q_quadratic=quadratic))
        if quadratic!=e.C(total):raise ValueError('actual grouped action disagrees with Q quadratic form')
        return dict(raw_norm=norm.r,pattern_squared_norms=pattern_norms,source_squared_norm=total,
                    f_hat_squared=total/norm.r,identity=asdict(expected),source=self.seal,
                    exact_source_products=True,source_export_error=F(0),header=clock.TABLE_HEADER)


def panel_manifest():
    return tuple((F(2*j+1,128),F(1,64)) for j in range(256))


def validate_panels(samples,prefixes):
    clocks=tuple(clock_value for clock_value,allowance in samples)
    if clocks!=tuple(t for t,d in panel_manifest()):raise ValueError('ordered complete fixed wall midpoints')
    for t,allowance in samples:
        if not isinstance(t,F) or not isinstance(allowance,F) or allowance<0:raise ValueError('exact nonnegative panel input')
    wanted=tuple((F(j,4),16*j) for j in range(17))
    if tuple(prefixes)!=wanted:raise ValueError('fixed quarter-clock prefix association')
    return wanted


def source_norm_upper(source,state,expected,save=None):
    """Exact products plus paid normalized-state error; no physical action."""
    from mpmath import iv
    import mock_unitary_lead_v1 as q
    ratio=source.norm_ratio(state,expected,save=save)
    Q={(i,j):z for i,j,z in source.Q_rows};H={(i,j):z for i,j,z in source.H_rows}
    Csq=clock.schur(Q,len(source.basis_states))['maximum'];Hbound=clock.schur(H,len(source.basis_states))['maximum']
    previous=iv.dps
    try:
        iv.dps=60;C=iv.sqrt(q.I(Csq));L=C*q.I(Hbound)
        measured=iv.sqrt(q.I(ratio['f_hat_squared']));upper=measured+C*q.I(expected.epsilon)
        def enclosure(value):
            lo,hi=q.lower(value),q.upper(value)
            if not math.isfinite(lo) or not math.isfinite(hi) or not 0<=hi-lo<=1e-12:raise ValueError('source display precision unresolved')
            return lo,hi
        return dict(**ratio,Cjoin_squared=Csq,H_bound=Hbound,Cjoin_interval=enclosure(C),L_interval=enclosure(L),
            measured_norm_interval=enclosure(measured),f_upper_interval=enclosure(upper),
            error_charge='Cjoin*epsilon; exact source products; no source export error',
            interval_precision_decimal=60,status='resolved')
    finally:iv.dps=previous


def panel_prefix_bounds(samples,prefixes,L_upper):
    """Outward binary64 endpoints enter as exact dyadics, then exact sums."""
    validate_panels(samples,prefixes)
    if not isinstance(L_upper,F) or L_upper<0:raise ValueError('nonnegative exact Lipschitz upper bound')
    width=F(1,64);pieces=tuple(width*allowance+L_upper*width**2/4 for t,allowance in samples)
    return dict(panel_contributions=pieces,prefixes=tuple((t,sum(pieces[:count],F(0))) for t,count in prefixes),
        meaning='Duhamel state-distance bound; not a return probability',physical_midpoints_evaluated=False)


def validate_Q_axis(Q):
    if any(z.r and z.i for z in Q.values()):raise ValueError('this actual Q roster must be real or pure imaginary')
    return True


def operator_allowances(represented,intended,rounded,assembly_upper,mass_upper):
    for value in (assembly_upper,mass_upper):
        if not isinstance(value,F) or value<0:raise ValueError('exact nonnegative exported operator allowance')
    export_delta=joining._subtract(rounded,represented);mass_delta=joining._subtract(represented,intended)
    square=sum((z.abs2() for z in export_delta.values()),F(0))
    if square>assembly_upper**2:raise ValueError('understated actual assembly allowance')
    mass_actual=clock.schur(mass_delta,max((max(key) for key in represented),default=0)+1)['maximum']
    if mass_actual>mass_upper:raise ValueError('understated intended-mass allowance')
    return dict(assembly_difference=export_delta,represented_mass_difference=mass_delta,
        assembly_Frobenius_squared=square,assembly_upper=assembly_upper,mass_actual_bound=mass_actual,
        mass_allowance=mass_upper,total_allowance=assembly_upper+mass_upper,
        actual_operator=fingerprint(matrix_rows(rounded,max((max(k) for k in represented),default=0)+1,max((max(k) for k in represented),default=0)+1)),
        represented_operator=fingerprint(represented),intended_operator=fingerprint(intended))
