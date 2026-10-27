# Reference perturbation energy, band flux and radiation entropy

## 1. Connected reference energy

Work first on a finite parent. Let h be Hermitian and gapped at zero,
P its negative projector, and |Omega> its filled sea.
Write occupied levels i,j and empty levels alpha,beta.

The specified second-order Hamiltonian, apart from its scalar constant,
is split as

H(lambda)=H0+lambda V,
H0=:c^dagger h c:_P,
V=:c^dagger Bc:_P+V4,
B=-h_H[bar Delta],

where V4 is fully Wick-ordered quartic relative to P.
The parameter lambda is perturbative bookkeeping, not a new coupling
in the model.

Normal ordering makes <Omega|V|Omega>=0.
The one-body term acting on the sea gives precisely

sum_alpha,i B_alpha i c_alpha^dagger c_i |Omega>.

A fully Wick-ordered number-conserving quartic acting on the sea
can create only two particles and two holes. Denote its
antisymmetrized amplitudes, with ordered indices, by
T_alpha beta;ij. It creates no single particle-hole state:
every term with an annihilator of the quasiparticle vacuum on
the right vanishes.

These single and double sectors are orthogonal eigenstates of H0,
with excitation energies respectively

epsilon_alpha-epsilon_i,
epsilon_alpha+epsilon_beta-epsilon_i-epsilon_j.

Second-order Rayleigh--Schrodinger perturbation theory therefore gives

Delta E_ref^(2)
 =-sum_alpha,i |B_alpha i|^2/(epsilon_alpha-epsilon_i)
  -sum_alpha<beta,i<j |T_alpha beta;ij|^2/
       (epsilon_alpha+epsilon_beta-epsilon_i-epsilon_j). (1)

Both terms are nonpositive. Strict negativity requires at least one
nonzero amplitude. The single term disappears exactly when
(I-P)BP=0, equivalently PB(I-P)=0 for Hermitian B.
Commutation of h with P says nothing about this separate B.

At fixed finite parent, if both amplitudes are O(G) and excitation
denominators stay uniformly positive, (1) is O(G^2).
An O(sqrt(G)) amplitude can instead give an O(G) contribution.
An infinite-parent or volume-uniform assertion additionally needs
convergence and uniform bounds on both sums. Equation (1) does
not establish such bounds or the continuation of a gapless branch.

## 2. Finite-band energy flux for arbitrary fermionic covariance

Consider one freely propagating outgoing positive-frequency channel
with one-particle space L2([omega1,omega2],domega), 0<=omega1<omega2.
Normal order against the empty band. Let 0<=C<=I be its covariance;
Gaussianity and stationarity are unnecessary.

Put

f_t(omega)=exp(i omega t)/sqrt(2pi),
g_t(omega)=omega f_t(omega),
A_t=(|f_t><g_t|+|g_t><f_t|)/2.

The band-restricted local energy current is j_E(t)=Tr(C A_t).
It is the symmetrized time-derivative current; its off-diagonal
frequency kernel contains (omega+omega')/2.

Define m_j=int_omega1^omega2 omega^j domega/(2pi).
A_t has range in span(f_t,g_t). Its trace is m1 and its two
possibly nonzero eigenvalues are

lambda_+-=(m1+-sqrt(m0 m2))/2.                           (2)

This follows by diagonalizing its two-dimensional Gram
representation. Cauchy--Schwarz gives sqrt(m0m2)>=m1, with strict
inequality for a band of positive width. Hence the negative
eigenvalue is generally nonzero.

For any Hermitian finite-rank A and 0<=C<=I,

Tr(CA)<=Tr A_+.

Apply this to (2), and integrate the pointwise inequality over
any arrival interval of length Delta t:

E_Delta t<=Delta t (m1+sqrt(m0m2))/2.                    (3)

For n_ch channels the current is the direct sum of these operators.
Off-diagonal interchannel covariance blocks do not contribute to
the trace; (3) therefore multiplies by n_ch even in their presence.

With omega_i=x_i T_H, T_H=1/(4pi r_s),
E=r_s/(2G), g=G/a^2, chi=a/r_s, (3) becomes

E_Delta t/E
 <= B12 n_ch (Delta t/r_s)(x2^2-x1^2)g chi^2/(32pi^3),

B12=(1/2)[1+
 2sqrt((x1^2+x1x2+x2^2)/3)/(x1+x2)].                    (4)

At x1=25,x2=100 this is 1.02915... times the filled-band coefficient.
For frequency-diagonal C the expectation is instead
int omega n(omega)domega/(2pi)<=m1, so B12=1 is valid in that
restricted class.

The distinction is real: filling the positive eigenvector of A_t
gives lambda_+>m1 at that instant, and continuity gives violations
of the filled-band bound on sufficiently short intervals.
Equation (4) is a valid, possibly nonoptimal, finite-time bound.
It concerns the stated freely propagating band current, not an
interacting detector observable or an unlimited-frequency burst.

## 3. Entropy growth without a geometric horizon premise

For any tripartite state on earlier radiation R, new radiation b,
and retained system c, weak monotonicity gives

S(Rb)+S(bc)>=S(R)+S(c).                                 (5)

It follows from strong subadditivity by purifying the tripartite
state and replacing complementary entropies. Thus

S(Rb)-S(R)>=S(c)-S(bc).

If S(c)>=log 2-epsilon and S(bc)<=epsilon, the increase is at least
log 2-2epsilon. No assumption about a global event horizon or
an impermeable geometric surface appears.

Transfer across a surface can change the hypotheses about a retained
independent partner, but nonzero transfer probability is not itself
a statement about S(bc), purification of R, or recovery of an unknown
input. Those require the actual joint state and its correlations.
