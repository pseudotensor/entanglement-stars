# On-site conversion, allocated balance and resonant obstruction

This document defines the direct on-site heavy-light converter and proves
the operator statements used with it. A separate filtered control is
defined in Section 7. Neither construction supplies a gravitational
closure, a production rate, a mechanical deformation rule or a bound on
the force in a collapsing state.

Sources. The operator, its symmetries and the first-order current are
proofs/hreduce/VERTEX_gate_v1.md, Sections 1 and 3; the allocation, flux, force,
routing, field coupling and the two witnesses are proofs/hreduce/VERTEX_stress_v1.md,
(1)-(10) and (14)-(17); the normalized spinors and the tensor (8) are
proofs/hreduce/VERTEX_amplitudes_v1.md, (1); the resonant common-boost family and
its derivatives (10) are Part A of proofs/hreduce/VERTEX_force_moving_v1.md; Section 5
restates Lemmas 1-4 and Theorem 1 of proofs/busi/NONCONS_universal_v1.2.md, whose
hypothesis (A) is verified there exactly at ma=3/5 (checks C1, C4) and here at the
Part A point; the filtered operator of Section 7 is
proofs/hreduce/VERTEX_gate_filtered_v1.md, Sections 1 and 3. This file adds no result
beyond those files; it collects them in the paper's order.

## 1. Operator and valley content

Use dimensionless site CAR on x in aZ^3, component order
(upper up, upper down, lower up, lower down), and

alpha_i=tau_x tensor sigma_i, beta=tau_z tensor I2,
h0,A=m_A beta+sum_i alpha_i(T_i-T_i^dagger)/(2ia),
(T_i u)_x=u_{x+ae_i},
m_H=m>0, m_L=0.

The definitions below are those of the supplied
"VERTEX_gate_v1.md", Sections 1 and 3, and
"VERTEX_stress_v1.md", equations (1), (4)--(7):

M0=tau_z tensor sigma_y,
M1=tau_y tensor sigma_z,
M2=i tau_y tensor I2,
M3=-tau_y tensor sigma_x,
eta=diag(1,-1,-1,-1),

Phi_A^mu(x)=c_A,x^T M^mu c_A,x,
W_x=sum_mu eta_mumu Phi_L,x^{mu dagger} Phi_H,x^mu+h.c.,
H=H0+UV, V=sum_x W_x, U=lambda_X/a^3 real.              (1)

There is no factor 1/2 in Phi. lambda_X has dimensions length squared;
U has dimensions energy. No relation between lambda_X and G is inferred.

The matrices M are antisymmetric. CAR gives

Phi a^dagger(u)a^dagger(v)|0>=-2u^T M v |0>.

Thus the vertex changes species numbers by (-2,+2) or (+2,-2), preserves
their parities and total number, and is Hermitian. Operators W_x at
distinct sites commute because they are even and have disjoint support.

An on-site Fourier transform enforces crystal-momentum conservation
modulo 2pi/a. At valley corners K_b=pi b/a, the leading corner
quartets satisfy b1 xor b2 xor b3 xor b4=0; all such quartets occur
subject to their spin matrix elements. No separate canonical valley
field or filter is part of (1). The common-spinor contraction does not
prove a common physical multivalley Lorentz symmetry.

## 2. Energy allocation and current

Let

e0,x=dGamma({Pi_x,h0}/2), e_x=e0,x+UW_x,
D_i^0=sum_x x_i e0,x, D_i^W=sum_x x_i W_x.

dGamma sums the two species. On the infinite lattice the dipole
identities are first defined on finite-support vectors. On a periodic
parent use consistent local displacements, not a discontinuous global
position function.

Expanding the dipole commutator gives exactly

Q_i=i[H,D_i^0+UD_i^W]
 =Q0,i+UQ1,i+U^2Q2,i,
Q1,i=i[H0,D_i^W]+i[V,D_i^0],
Q2,i=i[V,D_i^W].                                       (2)

For (1), Q2=0 by the disjoint-site commutation just proved.
The free onsite mass terms cancel in Q1 because their two dipole
weights are equal. Allocating a hopping bond T_xy equally to x and y
leaves

Q1,i=(i/2)sum_<x,y> (y_i-x_i)[T_xy,W_y-W_x].             (3)

The free part is

Q0,i=dGamma(q_i), q_i=(T_i^2-T_i^{dagger 2})/(4ia).

Define the energy transfer and momentum allocation

jE_xy=i[e_x,e_y],
P_xi=P0_xi+(1/2)sum_y(y_i-x_i)(jE_xy-jE0_xy),
P0_xi=dGamma({Pi_x,q_i}/2).                             (4)

Antisymmetry of jE and the dipole identity show sum_x P_xi=Q_i.
The interaction bond moment is divided equally between its endpoints.

## 3. Exact flux, force and routing

Define

J0_xyi=dGamma((i/2){Pi_x h0 Pi_y-Pi_y h0 Pi_x,q_i}),
JI_xyi=i([e_x,P_yi]-[e_y,P_xi]
          -[e0,x,P0_yi]+[e0,y,P0_xi]),
sI_xi=i[e_x,Q_i]-i[e0,x,Q0,i].                          (5)

All these operators have finite range. J0 and JI are antisymmetric
under x,y exchange. Since [h0,q_i]=0,

i[H0,P0_xi]+sum_y J0_xyi=0.

Summing the definition of JI over y gives
i[e_x,Q_i]-i[H,P_xi]-i[e0,x,Q0,i]+i[H0,P0_xi].
Hence

dot P_xi+sum_y(J0_xyi+JI_xyi)=sI_xi,
sum_x sI_xi=i[H,Q_i].                                  (6)

This is an algebraic identity, not source-free conservation.

A literal face flux can be defined without a metric variation.
Route x-to-y transfer equally over all shortest cubic paths.
Let chi_xy(z,j) be the signed average traversal of the edge
(z,z+ae_j). Then

sum_j[chi_xy(z,j)-chi_xy(z-ae_j,j)]=delta_zx-delta_zy.

Consequently

T_ij(z+ae_j/2)=(1/(2a^2))sum_xy chi_xy(z,j)(J0_xyi+JI_xyi)

obeys the cell-divergence form of (6), after division by a^3.
Its integrated first moment is
(1/2)sum_xy(y_j-x_j)(J0_xyi+JI_xyi).
This routing need not give a symmetric tensor, and is not a definition
of mechanical stress by virtual work.

For classical site fields the declared generator is

G[N,v]=sum_x N_x e_x+sum_xj v_xj P_xj.

Writing Aext=G-H gives

dot P_xi+sum_y J_xyi=sI_xi+i[Aext,P_xi].                 (7)

If the source subtracts a time-dependent reference mean p_ref,x,
its derivative includes -dot p_ref,x. An interacting stationary
reference, when it exists differentiably, must be used consistently.
Neither (6) nor (7) proves a hydrodynamic relation between an energy
source and v times this force.

## 4. Matrix elements and an on-shell witness

At sine momentum d, use the normalized positive-band spinors

u_m(d)=sqrt((E+m)/(2E)) ( I2 ; sigma.d/(E+m) ),
E=sqrt(m^2+|d|^2).

On V_s sites, ordered two-particle states have the stripped tensor

A_t1t2;s1s2=4sum_mu eta_mumu
 (u_L1^T Mmu u_L2)^*_{t1t2}
 (u_H1^T Mmu u_H2)_{s1s2}.                              (8)

The physical element of UV is U A/V_s times the crystal-momentum
delta. Equation (8) follows directly from the two factors -2 in
Section 1. It is not flux normalized.

For opposite heavy momenta +-d zhat, write Y=sigma_y, Z=sigma_z.
Direct multiplication gives

J_H=(mY/E, i d I2/E, -d Z/E, 0),
J_L=(0,iI2,-Z,0)

for opposite light z directions. Substitution in (8) gives

A=-(8d/E)diag(1,0,0,1).                                (9)

At heavy rest this tensor is zero. Nonzero momentum, not the rest
threshold, is used below.

Set mu=ma=3/5, sin h=9/20 and sin k=3/4 (the Part A point of
proofs/hreduce/VERTEX_force_moving_v1.md). Then both species have
energy 3/(4a) per leg and

A=-(24/5)diag(1,0,0,1).

For B=aP_z, keep h fixed and set
theta_H+-=B/2+-h, theta_L+-=B/2+-k(B).
The shell equation is the equality of the two energy sums.
At B=0 its derivative in k is sqrt(7)/2, nonzero. The implicit
function theorem gives a real smooth even k(B), with k'(0)=0.
The amplitude remains nonzero near zero.

The pair charges are

a q_HH=sin B cos(2h),
a q_LL=sin B cos(2k(B)).

Thus

partial_P q_HH|0=119/200,
partial_P q_LL|0=-1/8,
partial_P Delta q|0=-18/25.                            (10)

There are therefore exactly resonant connected configurations at
arbitrarily small nonzero P with Delta q!=0.

For any differentiable proposed conserved charge
Q(U)=Q0+UQ1+o(U), [H0,Q0]=0, its first-order equation is

[H0,Q1]+[V,Q0]=0.

Between simultaneous free energy/charge eigenstates,

(E_f-E_i)(Q1)_fi=(q_f-q_i)V_fi,
<f|i[V,Q0]|i>=-i(q_f-q_i)V_fi.                          (11)

Equations (9)--(10) violate the on-shell condition in (11).
For a local charge required to conserve on all sufficiently large
periodic parents, its finite Fourier identity holds at all real
momenta: after eliminating the momentum-constrained leg, a local
term of support diameter D has degree at most D in each free
momentum coordinate. Vanishing on a grid of more than 2D points in
each coordinate forces each coefficient to vanish. This justifies
using the real-momentum shell, without pretending that every such
point is an allowed momentum on one small torus.

## 5. General scalar-band odd-symbol obstruction

This section restates the argument of proofs/busi/NONCONS_universal_v1.2.md.
Assume a finite-range one-body leading symbol, scalar on each free
band and commuting with h0. It has the form

q_A(p)=f_A(p)I+g_A(p)h0,A(p),                            (12)

where f_A,g_A are real trigonometric polynomials.
For the heavy species, trace with beta extracts m g_A.
For the light species, trace with alpha_i extracts g_A sin(ap_i)/a.
The sine factors in distinct variables are coprime in the Laurent
polynomial ring; the identities t_i sin(ap_j)=t_j sin(ap_i) imply
that g_A itself is a Laurent polynomial. Hermiticity makes it real
on the momentum torus.

Let q_A^+=f_A+g_A E_A and let q_A^e,q_A^o be its even and odd parts.
For ma<1 consider the zero-total-momentum shell

Sigma0={(p,k): |ap_i|,|ak_i|<pi/2,
                 E_H(p)=E_L(k) in (m,1/a)}.

It is connected. Along every direction, sum_i sin^2(a t n_i) is
strictly increasing until the ray leaves the open valley cube,
and reaches at least one there. Each allowed energy surface is
therefore a connected star-shaped sphere, varying continuously
with energy; their pair product over an interval is connected.
Their energy gradients do not vanish.

Suppose the stripped conversion tensor is nonzero at one point of
Sigma0. Its squared norm is a frame-independent analytic function.
It is therefore nonzero on a dense open subset of this connected
shell. The point of Section 4 supplies this hypothesis at ma=3/5.

Use coordinates p1=P/2+p, p2=P/2-p, and similarly for k.
Let G be the outgoing minus incoming energy sum and Dq the
corresponding charge difference. Conservation implies Dq A=0 on
G=0. At P=0, some derivative of G in k is nonzero, while
partial_P G=0 by energy evenness. The P directions are tangent
to the shell. Evaluating and differentiating Dq A=0 there gives

q_L^e(k)=q_H^e(p),
grad q_L^o(k)=grad q_H^o(p).                             (13)

The derivatives of A disappear because Dq=0 at the base point.
Equivalently apply Hadamard's lemma to Dq A=G R.
Density and continuity extend (13) over Sigma0.

At a fixed energy, fixing one heavy momentum in (13) shows that
grad q_L^o is constant on the light energy surface. Thus

grad q_L^o(k)=b(E_L(k)).

Vanishing curl implies b'(E) is parallel to every normal of that
energy surface. Its normals at the two diagonal directions
(1,1,1) and (-1,1,1) are nonparallel. Hence b'(E)=0.
The odd eigenvalue is affine, b.k+c, on an open shell region.
Real analyticity extends this identity to R^3 minus the isolated
light nodes, a connected set. Periodicity forces b=0; oddness
forces c=0. Equation (13) then forces the heavy odd eigenvalue to
vanish by the same analytic continuation.

For completeness, vanishing positive-band odd eigenvalue also
eliminates odd polynomial coefficients in (12). If f^o=-g^o E,
squaring gives (f^o)^2=(g^o)^2 E^2. Fix the other two momentum
coordinates generically. As a Laurent polynomial in z=e^{iap_1},
a^2 E^2=mu'^2+sin^2(ap_1), with mu'^2>0, has four distinct simple
zeros. The right side has odd multiplicity at those zeros while
the left side has even multiplicity, unless g^o=0. Then f^o=0.
This also treats a putative odd contribution confined to the
negative band.

Therefore no such finite-range leading momentum with a nonzero
parity-odd part has a differentiable finite-range conserved
correction at the stated test parameter. Spin-dependent leading
symbols, many-body leading terms and unbounded-range corrections
are not classified here. No complete classification of even
charges is needed: number and H are explicit conserved exceptions
to any unrestricted "nonconstant symbol" statement.

## 6. Quadratic witness and expectation scope

The total on-site momentum rate is

i[H,Q]=U i([H0,Q1]+[V,Q0])+U^2 i[V,Q1].                 (14)

Its quadratic term is nonzero. One can check it on one z bond,
a=1. Number modes 0,...,7 at the first site and 8,...,15 at the
second, with heavy components before light components.
Let

A=|0,2>, B=|8,10>, C=|0,8>,
T=-(i/2)sum_A c_A,0^dagger alpha_z c_A,1+h.c.,
Q1=(i/2)[T,W1-W0],
F2=i[W0+W1,Q1], J2=(i/2)[W0-W1,Q1].

Direct use of (1) and CAR gives

W0 A=-8|4,6>, W1 B=-8|12,14>,
T C=-i(A+B)/2,
Q1 C=-2|4,6>+2|12,14>,
F2 C=16i(A-B), J2 C=8i(A+B).

For the normalized species-product Slater state
psi=(A+iC)/sqrt(2), Hermiticity then gives

< F2 > =-16, < J2 > =-8.

Restoring units, these are -16 U^2 for the force and -8a U^2
for the integrated bond stress. Additional bonds on a larger
periodic embedding do not connect the two occupied support
configurations in this matrix element; diagonal terms vanish
in this real basis. Thus the witness is not an open-wall effect.

Any source-free antisymmetric periodic flux would have zero
lattice sum, contradicting (14) and this witness. Bond rerouting
cannot remove the integrated obstruction.

On rho_H tensor rho_L with each factor conserving its own number,
operators of odd converter degree have zero expectation by the
species-number selection rule. This proves vanishing first-order
interaction energy, current and flux expectations. The quadratic
witness proves that neutral even-degree terms need not vanish.
These statements do not evaluate the interacting sea, its stress,
or its reference response. In a state stationary under the full H,
the expectation of every full commutator i[H,Q] is zero by trace
cyclicity; stationarity under H0 alone is insufficient.

## 7. The filtered control is different

The attached "VERTEX_gate_filtered_v1.md", Sections 1 and 3, defines

F_b=prod_i [1+(-1)^{b_i}(T_i+T_i^dagger)/2]/2,
Phi_A,b,x^mu=(F_b c_A)_x^T Mmu(F_b c_A)_x,
O_b,xy=sum_mu eta_mumu Phi_L,b,x^{mu dagger}Phi_H,b,y^mu,

W_x=(1/2)sum_b,y g_{(x-y)/a}
 (O_b,xy+O_b,yx+O_b,xy^dagger+O_b,yx^dagger).            (15)

Here

B_a(z)=a^-3 1_{[-a/2,a/2]^3}(z),
K_ell(z)=9009(1-|z|^2/ell^2)_+^5/(1024pi ell^3),
k=K_ell*B_a,
g_r=a^3 int k(y)k(y-ar)d^3y,
kappa(P)=sum_r g_r e^{iaP.r}.

The normalized cell partition gives sum_r g_r=1.
The leg symbol is
f_b(p)=prod_i[1+(-1)^{b_i}cos(ap_i)]/2.
The tensor (8) is multiplied by

kappa(P) sum_b prod_{l=1}^4 f_b(p_l).                    (16)

At zero pair momentum kappa=1, and the multiplier is strictly
positive inside the zero valley. Thus the odd-symbol proof applies
to this control too at the test parameter.

Equations (2), (4)--(7) remain exact, but now

Q2,i=(i/2)sum_xy(y_i-x_i)[W_x,W_y]

cannot be discarded: filtered supports overlap. The total rate
also includes U^2 i[H0,Q2] and U^3 i[V,Q2].
This document does not assign the on-site absence of Q2 to (15),
nor transfer any on-site mock result to the filtered operator.
