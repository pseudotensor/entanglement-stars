# Weak-field lattice probe calculations

The calculation uses only the positive-band principal symbol of the
sine-stencil fermion and the prescribed stationary exterior
N=1, v=-sqrt(2mu/r) rhat, mu=GE. It is not a finite-zone trajectory
calculation. The probe mass is m>0.
On the exterior the probe's Hamiltonian is v.q(p)+epsilon(p) at clock N=1.

## 1. Expansion and canonical coordinates

Near any valley,

epsilon(p)=m+p^2/(2m)-p^4/(8m^3)
            -a^2 sum_i p_i^4/(6m)+higher orders,
q_i(p)=p_i-(2a^2/3)p_i^3+higher orders.

The lattice contribution at the required weak-field and
low-momentum order is therefore

delta H_a=-a^2 sum_i[p_i^4/(6m)+(2/3)v_i p_i^3].          (1)

Because the stationary exterior v is a gradient, the shift
P=p+mv is canonical. It turns the Newtonian part into
P^2/(2m)-m mu/r. Expanding (1) after p=P-mv gives

delta H_a=-(a^2/(6m))sum_i[
 P_i^4-6m^2v_i^2P_i^2+8m^3v_i^3P_i-3m^4v_i^4].         (2)

Terms beyond the required order are not retained.

## 2. Average over orientation in a fixed plane

Let Lhat be the plane normal and define

B=(1+sum_i Lhat_i^4)/8, u=P/m.

Rotating two fixed vectors in this plane uniformly through an angle
gives

<sum_i u_i^4>_orient=3B u^4,
<sum_i v_i^2u_i^2>_orient
   =B[v^2u^2+2(v.u)^2],
<sum_i v_i^3u_i>_orient=3B v^2(v.u),
<sum_i v_i^4>_orient=3B v^4.

For example the projection of coordinate axis i into the plane has
squared length 1-Lhat_i^2; the fourth angular moment is
3(1-Lhat_i^2)^2/8. Summing uses
sum_i(1-Lhat_i^2)^2=1+sum_i Lhat_i^4.
Polarization gives the mixed moments.

Equation (2) becomes

<delta H_a>_orient/m
 =-(am)^2 B/2[
 u^4-2v^2u^2-4(v.u)^2+8v^2(v.u)-3v^4].                 (3)

The term odd in radial velocity has zero orbital average.
Indeed v is radial and stationary, so its time integral is a
closed integral of a function of r times dr.

## 3. Secular Kepler average

For an ellipse of semimajor axis A_o, eccentricity e, and
j=sqrt(1-e^2), let the specific angular momentum be ell.
Then

u^2=2mu/r-mu/A_o, ell^2=mu A_o j^2,
u_r^2=u^2-ell^2/r^2,
v^2=2mu/r, (v.u)^2=(2mu/r)u_r^2.

The Kepler time averages are

<1/r>=1/A_o,
<1/r^2>=1/(A_o^2 j),
<1/r^3>=1/(A_o^3 j^3).

They follow immediately from r=A_o(1-e cos E) and
dt=(1-e cos E)dE/n.

The even bracket in (3) simplifies before averaging to

mu^2/A_o^2-32mu^2/r^2+8mu^2/(A_o r)+8mu ell^2/r^3,
the coefficients -32=4-8-16-12 from u^4, -2v^2u^2, -4(v.u)^2 and -3v^4.

Its average is mu^2 A_o^-2(1-32/j+8+8/j)=mu^2 A_o^-2(9-24/j). Thus

<delta H_a>_sec/m
 =(am)^2 B mu^2 A_o^-2(-9/2+12/j).                      (4)

## 4. Schwarzschild comparison and precession

In static Schwarzschild areal coordinates, with canonical specific
radial momentum u_r and transverse momentum u_t, the Hamiltonian
per unit mass is

H_GR/m=sqrt[(1-2mu/r)
              (1+(1-2mu/r)u_r^2+u_t^2)].

Its first post-Newtonian correction is

-u^4/8-(mu/(2r))u^2-(mu/r)u_r^2-mu^2/(2r^2).

Using the averages above gives

<H_GR,1PN>/m=mu^2 A_o^-2(15/8-3/j).                    (5)

At fixed Kepler principal action sqrt(mu A_o), differentiation
with respect to the angular action sqrt(mu A_o)j gives the
secular apsidal rate. The j-independent terms in (4)--(5) drop out.
Therefore

Delta varpi/Delta varpi_GR
 =1-4B(am)^2
 =1-(1/2)(1+sum_i Lhat_i^4)(am)^2.                      (6)

The in-plane orientation average is essential. Equations (4)--(6)
do not give the unaveraged anisotropic torque or a finite-zone
orbit. The exterior field has not been modified.

## 5. Fixed isotropic composite approximation

This is a kinematic fixed-internal-state approximation, not a
construction of a bound state of the free model.

For n identical constituents let the internal momentum distribution
be isotropic and inversion symmetric, with zero total internal
momentum. Give each constituent the common small shift P/n and
keep that distribution fixed. The expectation of the free internal
energy has a scalar quadratic center term P^2/(2M_*).
Inversion symmetry eliminates the constant and quadratic terms in
the expectation of the odd current sum. Its linear term is

sum_a <q_i(k_a+P/n)>=Upsilon P_i+O(P^3),
Upsilon=<cos(2ak_i)>,

with the average including constituents and their fixed internal
state. Isotropy makes Upsilon independent of i.

For a small object on which the flow is effectively uniform,
at constant clock N=1 (a varying clock would add -E_int grad N to the force), its
center Hamiltonian to this order is

H_c=E_int+P^2/(2M_*)+Upsilon v(X).P.

Hamilton's equations give

dot X=P/M_*+Upsilon v,
dot P=-Upsilon (grad v)^T P.

At stationary irrotational v and constant Upsilon, elimination of P
gives

ddot X=Upsilon^2 (grad v)^T v
      =Upsilon^2 grad(v^2/2).

The curl term cancels only for irrotational flow.
For v^2=2mu/r this is -Upsilon^2 mu rhat/r^2.
The result requires the fixed-state, small-size, isotropic,
quadratic-center-momentum assumptions. It is not assigned to
the elementary dust orbitals used in the collapse.
