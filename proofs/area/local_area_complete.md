# Local area estimates: complementary complete proofs

All entropies are in nats. This document proves the planar convexity,
massive reference estimate, finite-parent estimate, a sufficient frozen-gap
certificate, and finite-pocket transfer used in the local-area appendix.
No numerical history is certified here.

## 1. Definitions and entropy estimates

Set a=1. On l2(Z^3) tensor C^4 let

h0(k)=zeta beta+sum_i alpha_i sin k_i,
E(k)=sqrt(zeta^2+sum_i sin^2 k_i),
q_i(k)=sin(2k_i)/2,
P0(k)=(I-h0(k)/E(k))/2.

Values at the isolated massless nodes do not affect the infinite-parent
operator. The matrices obey the four-component Clifford relations. The
inhomogeneous operator is

h={N,h0}/2+sum_i {v_i,q_i}/2.

Site fields are real. All gradients below are gradients of specified
smooth extensions of those site fields. A finite cut A has projection
chi_A and satisfies

#{x in A: dist(x,A^c)<=d} <= kappa A_lat(1+d).

In particular its displacement crossing count is at most
kappa A_lat(1+|z|).

For an orthogonal projection P put X_P=chi_A P chi_Ac. Purity gives

S_A(P)=sum_j F(s_j(X_P)),
F(s)=s_bin((1-sqrt(1-4s^2))/2),  0<=s<=1/2,

where s_bin(t)=-t log t-(1-t)log(1-t).
The identity follows from
P_A(I-P_A)=X_P X_P^dagger. Zero singular values contribute zero.

The derivative obeys 0<=F'<=2. Indeed, with x=sqrt(1-4s^2),

F'(s)=sqrt(1-x^2) log((1+x)/(1-x))/x,

and artanh(x)<=x/sqrt(1-x^2), as follows by differentiating both sides
from equality at zero. Also, for every fixed 0<b<1,

|F'(s)|<=C_b s^b.

Mirsky's inequality therefore gives

|S_A(P)-S_A(Q)|<=2||chi_A(P-Q)chi_Ac||_1<=||P-Q||_1.       (1)

For the second inequality set U=chi_A-chi_Ac. The off-diagonal part
(D-UDU)/2 of D=P-Q has trace norm 2||chi_A D chi_Ac||_1, at most ||D||_1.

If contractions A0,A1 differ by rank r, then

|Tr s_bin(A1)-Tr s_bin(A0)|<=r log 2.                     (2)

To see the constant, decompose A1-A0 into positive and negative parts
of ranks r_+,r_-. Eigenvalue interlacing bounds its spectral-shift
function between -r_- and r_+. The positive and negative integrals of
s_bin' each have magnitude log 2. Integration gives
(r_++r_-)log 2, not twice that value. Finite truncation proves the
corresponding convergent infinite-system assertions.

We also use the following surface bound. For 0<p<=2, a block matrix X
satisfies

||X||_p^p <= sum_xy Tr |X_xy|^p.                         (3)

Apply trace subadditivity of positive powers p/2 first to the column
decomposition of XX^dagger, and then to the four-dimensional block
column Gram matrices. Infinite exteriors follow by finite truncation.

The flat kernel satisfies

||P0(z)||<=C(1+|z|)^-3,                                  (4)

uniformly on a fixed compact mass interval. Away from a node it is
smooth. On an annulus of radius b about a node, its jth derivative is
bounded by C_j b^-j, uniformly in that mass interval. A smooth dyadic
partition gives the annular Fourier bound

C_M b^3(1+b|z|)^-M.

Summing separately over b|z|<=1 and b|z|>1 proves (4), for M>3.
Consequently (3) and the crossing count give

||X_P0||_p^p<=C_p kappa A_lat,  p>4/3.                   (5)

For singular values x_j,y_j and D=X-Y, the mean-value estimate and
Cauchy--Schwarz give

|Tr F(|X|)-Tr F(|Y|)|
 <=C_b ||D||_2 [sum_j(x_j^(2b)+y_j^(2b))]^1/2.           (6)

This follows by bounding each difference by
C_b |x_j-y_j|(x_j^b+y_j^b), then using Mirsky in Hilbert--Schmidt norm.

## 2. Convexity and identification of the limit

Here is the missing identification of the convex finite-volume
functional with the actual planar entropy coefficient.

First, entropy can be approximated uniformly in a surface norm.
Choose alpha in (2/3,1). There are polynomials p_j with
p_j(0)=p_j(1)=0 and

sup_{0<t<1}|s_bin(t)-p_j(t)|/[t(1-t)]^alpha ->0.           (7)

One construction first changes s_bin only in endpoint intervals of
width delta so that the changed function divided by t(1-t) is
continuous, and then uniformly approximates that quotient by a
polynomial. The endpoint weighted error tends to zero because
t^(1-alpha)|log t| tends to zero.

For C_A=chi_A P0 chi_A, operator concavity and diagonal Jensen give

Tr[C_A(1-C_A)]^alpha
 <=sum_{x in A,spin} [(X_P0 X_P0^dagger)_{x spin,x spin}]^alpha
 <=C_alpha kappa A_lat.                                (8)

The last inequality also follows directly from (3) with p=2alpha
and (5). Thus (7) controls the entropy error by its weighted
approximation error times the boundary area.

A polynomial trace is a sum of closed paths of kernels P0(z).
Its volume term cancels because p_j(P0)=0 on the full lattice.
For each fixed path length the sum of absolute coefficients times
path diameter is finite. To prove this explicitly, put
w(z)=(1+|z|)^-3. Splitting a convolution into
|y|<=|z|/2, |z-y|<=|z|/2 and the remaining region proves inductively

w^{*n}(z)<=C_n(1+|z|)^-3[log(2+|z|)]^(n-1).

Marking one path edge to bound the diameter therefore reduces the
closed-path sum to a constant times

sum_z |z|(1+|z|)^-6[log(2+|z|)]^(n-2)<infinity.           (9)

For one half-space, integration over its normal offset counts each
path's permitted interval of anchor depths. Those lengths are
continuous, positively homogeneous functions of the normal, bounded
by the path diameter. Equation (9) gives dominated convergence of
the polynomial surface functionals. Equation (8) then removes the
polynomial approximation uniformly in the normal. This identifies
the coefficient with the depth integral of the actual half-space
entropy density, rather than with a separately postulated surface
energy. The primitive-layer construction gives the same coefficient
at rational normals: layer width is 1/|b| and tangent cell area |b|.
The irrational strip count is obtained by smoothing the normal strip
and a large tangential test, applying Poisson summation, and taking
the tangential limit. After smoothing, reciprocal terms have a
summable bound C_M(1+|n.k|)^-M(1+|Pi_tan k|)^-M, M>3;
only reciprocal vectors parallel to n survive. Rational normals are
handled by their exact layer count. Removing the strip smoothing
recovers the depth integral.

Now let Omega be a large lattice cube and define on its subsets

m_Omega(A)=[S(A)+S(Omega\A)-S(Omega)]/2.

Strong subadditivity makes S submodular; hence m_Omega is submodular.
It is symmetric under A <-> Omega\A and vanishes at the empty set
and at Omega. Its Lovasz extension can be written

L_Omega(u)=int_R m_Omega({x in Omega:u(x)>t}) dt.          (10)

For negative levels this notation includes the usual zero endpoint
subtraction; it is finite because m_Omega(empty)=m_Omega(Omega)=0.
Submodularity proves that (10) is convex. Symmetry proves evenness,
and rescaling t proves positive homogeneity.

Set u(x)=xi.x and divide (10) by |Omega|. We claim

lim_{Omega increasing cubes} L_Omega(xi.x)/|Omega|
 = |xi| eta_{xi/|xi|}.                                  (11)

For a fixed finite path, anchors farther than its diameter from the
cube boundary see the infinite-lattice half-space. Integration over
t gives the same normal-depth length just derived, multiplied by
|xi|. The two entropy terms in m_Omega give the two sides of this
internal cut; their factor 1/2 counts it once. Terms associated only
with an exterior cube face cancel against S(Omega). The remaining
anchors within a fixed path diameter of the cube boundary are
O(side(Omega)^2); their normalized contribution tends to zero.

The integrated absolute contribution of any path, including such
boundary paths, is bounded by C|xi| times its diameter per anchor.
Thus (9) justifies removing the path cutoff after the cube limit.
For the entropy approximation, the intersection of a cube and a
half-space has a uniform boundary-tube bound O(side^2(1+d)).
The t interval has length O(|xi| side), so (8), divided by the cube
volume, bounds the approximation error by C|xi| times the error in
(7). This proves (11) for entropy itself.

Pointwise limits of finite convex functions preserve convexity.
Equation (11) therefore proves that Phi(xi)=|xi|eta_n, Phi(0)=0,
is convex. It is finite, continuous, even and homogeneous. Thus it
is the support function of a compact convex body. Cubic symmetry
comes from the flat operator.

If h_K(e_i)=eta_100, reflection symmetry makes the point eta_100 e_i
belong to K: average a supporting point over reflections in the
other coordinates. Conversely every coordinate of a point of K
has modulus at most eta_100. Therefore

conv{+-eta_100 e_i} subset K subset [-eta_100,eta_100]^3,

which proves the axial sandwich used in the paper. Adding other
enclosed support values is an ordinary convex-geometric operation;
this proof certifies no numerical support value.

## 3. Massive reference estimate

Assume zeta>0; 0<n_*<=N<=n^*, |v|<=v^*; first derivatives bounded
by C1 epsilon and second derivatives by C2 epsilon^2. Assume a
supremum-norm continuous path from the flat coefficients through
this class, along which h has a uniform gap (-delta,delta).
The reference is its negative spectral projection P.

Define the analytic Fourier multipliers

B_ij(k)=-i h0(k)q_j(k)partial_i P0(k)/(2E(k)^2),
a_ij(x)=partial_i v_j(x)/N(x),
P1=(1/2)sum_ij {a_ij(X),B_ij(D)}.                        (12)

Fourier convention: for a translation-invariant operator,
A_{x,x+z}=K_A(z)=int exp(-ik.z)A(k)d^3k/(2pi)^3.
Thus z=y-x, the shift (T_i u)_x=u_{x+e_i} has symbol exp(ik_i),
and multiplying K_A(z) by z_i gives the symbol -i partial_{k_i}A.
In particular [f,A] has first-gradient symbol
+i sum_i (partial_i f)(partial_{k_i}A), fixing the sign in (14).

The Clifford identities give
{h0,partial_i P0}=0,
[h0,B_ij]=-i q_j partial_i P0.
Thus B_ij is Hermitian. Its kernels and those of P0 decay
exponentially, because zeta>0 makes their symbols analytic in a
fixed complex strip.

For clarity, all remainder statements below use the weighted Schur
norm

||A||_gamma=max(sup_x sum_y e^(gamma|x-y|)||A_xy||,
               sup_y sum_x e^(gamma|x-y|)||A_xy||).

It is a Banach algebra. Choose gamma smaller than the analyticity
strip. Taylor's exact integral remainder for a coefficient f is

f(y)-f(x)-(y-x).grad f(x)
 =int_0^1(1-t)(y-x)^T Hess f(x+t(y-x))(y-x)dt.            (13)

Every remainder is bounded in the weighted norm by a second-gradient
bound times a finite second-moment sum of an exponentially decaying
kernel. Products use the Banach-algebra inequality.

Expanding the exact endpoint-symmetrized kernels with (13) gives

[h,P0]=i sum_ij (partial_i v_j)q_j partial_i P0
          +O_gamma(epsilon^2).                         (14)

The first lapse-gradient term vanishes by
{h0,partial_i P0}=0. Replacing left ordering by symmetric ordering
in (14) costs O_gamma(epsilon^2). Using (12), the leading term of
[h,P1] is the negative of (14). Terms in which a derivative strikes
a_ij are O(epsilon^2); terms containing its product with a field
gradient are also O(epsilon^2). The projector equation gives

[h,P0+P1]=O_gamma(epsilon^2),
(P0+P1)^2-(P0+P1)=O_gamma(epsilon^2).                    (15)

Indeed the zeroth-order band-diagonal part of P1 vanishes, and
commuting a_ij through P0 costs its O(epsilon^2) gradient.

Put A=P0+P1. For small epsilon its spectrum lies in disjoint
neighborhoods of zero and one. The Riesz projection Q around the
neighborhood of one satisfies

Q=A+O_gamma(epsilon^2).                                 (16)

This follows by the convergent inverse expansion on circles separated
from {0,1}; the inverse for P0 is explicit and the weighted norm of
P1 is O(epsilon). Form

h_d=QhQ+(1-Q)h(1-Q).

Equations (15)--(16) give h_d-h=O_gamma(epsilon^2).
At the beginning of the path, Q=P0. The actual gap of h and the
small perturbation imply that h_d remains gapped. Its Q block stays
negative and its complementary block stays positive by continuity;
a sign change would cross zero. Supremum-norm continuity of the
coefficients implies continuity of the first derivatives here:
the uniform second-derivative bound and the elementary interpolation
bound ||grad f||^2<=C||f||||Hess f|| suffice. Hence Q is continuous.

The negative projection of h_d is therefore Q. Resolvent identities
between h and h_d give

P=P0+P1+R2,  ||R2||_gamma'<=C epsilon^2                 (17)

for a fixed smaller gamma'>0. To justify the weighted resolvents,
conjugate the finite-range operator by truncated exponential distance
weights. Its change has norm at most
sum_d ||h_d|| (e^(gamma'|d|)-1).
Choose gamma' so this is less than half the contour distance to the
spectrum. A Neumann inverse bounds the conjugated resolvent uniformly;
remove the truncated weight by monotone bounds. Constants depend
only on the stated mass, gap, amplitude and derivative bounds.

In particular the cross-cut remainder has trace norm
C kappa A_lat epsilon^2, and the cross-cut P-P0 has squared
Hilbert--Schmidt norm C kappa A_lat epsilon^2, by summing their
exponential kernels with the crossing count.

Let Z_P be the Hermitian double
[[0,X_P],[X_P^dagger,0]]. Then
S_A(P)=Tr F(Z_P)/2, using the even extension of F.
For 0<p<=1, (3) and exponential decay give

Tr |Z_P|^p+Tr |Z_P0|^p <= C kappa A_lat p^-4.            (18)

The p^-4 is the three-dimensional displacement sum with one crossing
factor: sum_z(1+|z|)e^(-c p|z|)<=C p^-4.

Choose an even smooth F_d agreeing with F outside |x|<d,
F_d(0)=0, such that

|F-F_d|(x)<=C d^(2-p)|x|^p,
||F_d'||_infinity<=C,
||F_d''||_infinity<=C(1+|log d|).                       (19)

For example replace the logarithm in the small-x expansion
F(x)=-x^2 log x^2+x^2+O(x^4|log x|)
by a smoothly joined logarithm cut off at d. The bound in (19)
uses sup_{0<z<1} z^(2-p)|log z|<infinity uniformly for p<=1.

The trace Taylor remainder for self-adjoint matrices is bounded by
one half ||F_d''|| times the squared Hilbert--Schmidt perturbation.
This follows in an eigenbasis from the divided differences of F_d';
the same statement follows by finite truncation here.

Time reversal fixes P0 and the cut and sends P1 to -P1.
Consequently the real trace Tr F_d'(Z_P0) Z_P1 is its own negative
and vanishes. The R2 part of the linear term is bounded by its
trace norm. Equations (17)--(19) yield

|S_A(P)-S_A(P0)|
 <=C kappa A_lat[
 epsilon^2(1+|log d|)+d^(2-p)p^-4].

Take d=epsilon and p=1/|log epsilon| for epsilon<=e^-1.
Then epsilon^-p=e and

|S_A(P)-S_A(P0)|
 <=C kappa A_lat epsilon^2(1+|log epsilon|)^4.            (20)

Increasing C covers the remaining compact small-epsilon interval.
No gap of the restricted covariance away from zero or one was used.

## 4. Finite antiperiodic parent

Let L be even, epsilon=1/L, with antiperiodic shifts in all three
directions. Fields have periodic Fourier transfers, represented in
the native Brillouin cell, and satisfy

sum_q |q|(|Nhat(q)|+sum_i|vhat_i(q)|)<=F1 epsilon,
n_*<=N<=n^*,  |v|<=v^*,  n_*-v^*>=c_*>0.                (21)

All statements are uniform for 0<=zeta<=zeta_*.
P0^L is the flat projector at the same zeta.
Write E=|h0| and

h_d=P0^L h P0^L+(I-P0^L)h(I-P0^L).

The exact lapse-transfer commutator has norm at most C|q|: after
substitution of P0 its potentially singular factor is
(E_+-E_-)[I-h_+h_-/(E_+E_-)]/4.
For flow it is
[q_i(k_+)+q_i(k_-)] [P0(k_-)-P0(k_+)]/2.
If endpoint energies exceed a fixed multiple of |q|, differentiate
P0 and use |q_i|<=E; otherwise both energies and the averaged q_i
are O(|q|). This proves, including at massless nodes,

||h-h_d||<=C epsilon.                                  (22)

The diagonal band forms obey

h_d|positive >=c_* E-C epsilon,
h_d|negative <=-c_* E+C epsilon.                        (23)

For the lapse, the difference between {N,E}/2 and
E^(1/2)NE^(1/2) has multiplier
(sqrt(E_+)-sqrt(E_-))^2/2=O(|q|).
For flow put t_i=q_i/E. The difference between {v_i,q_i}/2 and
E^(1/2){v_i,t_i}E^(1/2)/2 has multiplier

(sqrt(E_+)-sqrt(E_-))
(q_i(k_+)/sqrt(E_+)-q_i(k_-)/sqrt(E_-))/2,

also O(|q|), since both factors are uniformly one-half Hölder.
Finally ||sum_i v_i t_i||<=v^*, by Cauchy--Schwarz and
sum_i t_i^2<=1. Integrating against (21) proves (23).

Endpoint commutator bounds also give
||h psi||>=c_*||E psi||-C epsilon||psi||,
and the same for h_d. The antiperiodic momentum grid has

rank 1_{E<=r}<=C L^3(r+epsilon)^3.                      (24)

It follows by counting grid points in the eight valley balls.
Coercivity and min--max applied to E^2 bound the low-energy ranks
of h and h_d by C L^3(delta+epsilon)^3. Equation (23) bounds
the wrong-sign subspaces of h_d by C L^3 epsilon^3.

Take a smooth negative-energy selector f_delta changing only in
(-delta,delta). Its Fourier/Duhamel representation gives

||f_delta(h)-f_delta(h_d)||<=C epsilon/delta.

Thus, for P=P_-(h) plus any orthogonal zero-mode filling,

P-P0^L=B_delta+T_delta,
||B_delta||<=C epsilon/delta,
||T_delta||<=3,
rank T_delta<=C L^3(delta+epsilon)^3.                    (25)

For cuts with n_A comparable to L^3, (25) gives the cross-block
Hilbert--Schmidt bound C sqrt(n_A)d, where

d=epsilon/delta+(delta+epsilon)^(3/2).

Choose delta=epsilon^(2/5), giving d=O(epsilon^(3/5)).
The flat surface Schatten bound, (6) with b=5/6, and rank
interpolation give

|Delta S|<=C d sqrt[n_A(A_lat+n_A d^(5/3))].             (26)

Since n_A=O(epsilon^-3) and A_lat is comparable to epsilon^-2,
division by area yields C epsilon^(1/10).

For completeness, the flat finite-parent comparison used here does
not require an assumed thermodynamic limit. If |z|<=0.9L, the
antiperiodic flat kernel differs from the infinite kernel by O(L^-3).
To prove this, use the dyadic decomposition from (4).
Annuli with radius b>=c/L have image error bounded by
C_M b^3(bL)^-M, summed over nonzero reciprocal images.
Summing these annuli gives O(L^-3), for M>3.
Choose the final smooth node neighborhood below the smallest
antiperiodic node distance, of order L^-1. It has no grid samples
and continuum Fourier norm O(L^-3). The smooth complement has
arbitrarily faster image decay. Even L ensures identical valley
phases in these estimates. Constants are uniform on the compact
mass interval.

For a nonwrapping ball r<=0.45L, its restricted matrix difference
therefore has Hilbert--Schmidt norm O(n_A L^-3)=O(1), trace norm
O(sqrt(n_A))=O(r^(3/2)). Eigenvalue trace-norm perturbation and
binary-entropy concavity give

|S_A(P0^L)-S_A(P0)|<=C r^(3/2)(1+log r).                 (27)

The same dyadic estimates give the parent surface bound used in
(26): for representative displacements up to the parent diameter,
the flat kernel is O((1+dist_torus(z,0))^-3); summing crossings
as in (5) is uniform. This completes both finite-parent steps.
Application to a continued reference requires that it actually has
the occupations stipulated above.

## 5. A directly proved gap certificate

Assume each of N,v_1,v_2,v_3 has Euclidean Lipschitz constant at most
C1 epsilon, N<=n^*, |v_i|<=v^*, and every frozen Hermitian symbol
has gap at least g_*>0.

For integer R>=1 define
chi_j^(1)(x)=cos(pi(x-jR)/(2R)) for |x-jR|<=R, zero otherwise,
and take tensor products in three dimensions. Then

sum_j chi_j(x)^2=1,
sum_j |chi_j(x)-chi_j(y)|^2<=pi^2|x-y|^2/(4R^2).         (28)

The second statement follows by viewing the functions as a unit
vector-valued map: its squared derivative in each coordinate is
pi^2/(4R^2), and integrating its derivative along the line segment.

Freeze all coefficients at each support center, giving h_j.
On the support, the onsite mass coefficient changes by at most
zeta C1 epsilon sqrt(3)R. The six nearest-neighbor hopping norms
sum to 3; their midpoint coefficient error is at most
C1 epsilon(sqrt(3)R+1/2).
The six two-step hopping norms sum to 3/2; their midpoint error is
at most C1 epsilon(sqrt(3)R+1).
Row and column Schur bounds therefore give

||(h-h_j)chi_j psi||
 <=E_R||chi_j psi||,
E_R=C1 epsilon[sqrt(3)(zeta+9/2)R+3].                  (29)

The same estimate holds after summing squared norms over j.
Using (28), the direct-sum commutator norm is at most

(sum_j ||[h,chi_j]psi||^2)^1/2
 <=D_R||psi||,
D_R=3pi(n^*+v^*)/(2R).                                (30)

Indeed the sum of nearest-neighbor hopping norm times hop length is
3n^*, and the corresponding two-step sum is 3v^*.
The onsite term does not contribute.

Plancherel gives ||h_j chi_j psi||>=g_*||chi_j psi||.
Triangle inequalities in the direct sum now prove

||h psi|| >= [g_*-E_R-D_R]||psi||.                      (31)

Only a positive right side is a certificate. For uniform fields use
the exact frozen gap directly. The same certificate must hold along
the preparation path when it is used to identify the continued sea
with P_-(h).

The frozen-gap lower bound at a massive trapping root follows from
t_i=sin^2 k_i, S=sum t_i:

|q|^2=sum t_i(1-t_i)<=S-S^2/3,
E^2=zeta^2+S.

For |v|<=sN this gives
g(x)>=N(x) min_{0<=S<=3}[sqrt(zeta^2+S)-s sqrt(S-S^2/3)].
At s=1 and zeta>0 the continuous minimum is strictly positive.
For a linear path from (1,0), each expression NE-|v.q| is a
concave function of the path parameter and is bounded below by
the smaller endpoint bound. This proves the path statement used
with (31), not a numerical threshold for any stored profile.

Illustrative evaluation used in the paper. The root estimate is
g(x)>=N(x) g_1(zeta), where
g_1(zeta)=min_{0<=S<=3}[sqrt(zeta^2+S)-sqrt(S-S^2/3)].
It implies a global bound n_* g_1(zeta) only where N>=n_* and
|v|<=N throughout the domain. A root condition at one location
does not supply the global frozen-gap infimum.

At zeta=1/5 the minimizer satisfies
100S^3-221S^2-12S+9=0 and lies between 0.1831904 and 0.1831905.
The minimum lies between 0.05769 and 0.05770.

For the illustrative numerical inputs C1=1, n^*=v^*=1, separately
assume that g_*>=g_1(1/5) throughout the preparation path. Put
A=sqrt(3)(zeta+9/2)=4.7sqrt(3), B=3pi, and
R=ceil(sqrt(B/(A epsilon))). Equations (29)-(31) then give

E_R+D_R<=2sqrt(A B epsilon)+(A+3)epsilon.

Here A=8.14064..., B=9.42478..., and 2sqrt(A B)=17.5184....
For 0<epsilon<=1e-5 this upper bound is below 0.05551, so it is
strictly below the assumed frozen-gap lower bound. This is a
sufficient illustrative scale, not a necessary gap condition.

No stored profile or preparation path is certified by this substitution.
Their actual frozen-gap infima and field derivative bounds must be
enclosed before (31) is applied. The length L in epsilon=a/L is the
field-variation scale and is not automatically the nominal r_s.

## 6. Transfer from a finite clipped region

Let h_c be a comparison operator satisfying Section 3, with gap
(-delta_c,delta_c), and let V=h-h_c be supported in a finite site set S.
For the clipping construction one may take S within two sites of U.
All operators here are bounded and self-adjoint.

Let P commute with h. Choose 0<delta1<delta_c and suppose

r_b=rank[(P-P_-(h))1_{|h|>=delta1}]<infinity.

Correct those r_b occupations to their ground-state values, producing
a commuting orthogonal projection P'. Equation (2) gives

|S_A(P)-S_A(P')|<=r_b log 2.                            (32)

The remaining non-ground occupations lie in the inner window.
Choose delta2 strictly between delta1 and delta_c so that neither
+-delta2 is an eigenvalue of h. Such a choice exists: V is finite
rank and the spectrum in each closed strict subinterval of the
comparison gap is finite. In fact the subspace |h|<delta2 has
dimension n0<=rank V. Otherwise it contains a nonzero vector in
ker V, on which ||h psi||=||h_c psi||>=delta_c||psi||,
contradicting its spectral bound delta2.

Let Gamma be a rectangle enclosing the spectra of h and h_c below
-delta2, but no other real spectrum. Put

c=min_{z in Gamma} dist(z,spec h_c)>0,
b=min_{z in Gamma} dist(z,spec h)>0.

Reduce c, if necessary, so c<=delta_c-delta2.
Choose gamma>0 such that exponential conjugation of h_c changes
its norm by at most c/2. Its finite hopping range gives explicitly
the sufficient condition

sum_d ||(h_c)_d||_infinity(e^(gamma|d|)-1)<=c/2.

Combes--Thomas, proved by the same conjugation and Neumann argument
as in Section 3, gives

||chi_S(z-h_c)^-1 e_{y,s}||<=2c^-1 e^(-gamma dist(y,S))

both on Gamma and at real |z|<delta2.

The resolvent identity, integrated around Gamma, therefore bounds
each exterior column of P_{h,<-delta2}-P_-(h_c) by

A1 e^(-gamma dist(y,S)),
A1=|Gamma| ||V||/(pi b c).                              (33)

Every normalized eigenvector phi of h in |lambda|<delta2 satisfies

phi=-(h_c-lambda)^-1 V phi.

Its site components are bounded by
2||V||c^-1 e^(-gamma dist(y,S)).
Consequently every orthogonal subprojection of that inner spectral
subspace has column norm at most

A2 e^(-gamma dist(y,S)),
A2=2sqrt(n0)||V||/c.                                   (34)

P' is the far negative projection plus such an inner subprojection.
Combining (33)--(34), with A0=A1+A2, gives a column bound for
D=P'-P_-(h_c).

Define, for d>=0,

Ecal(d)=8A0 sum_{y:dist(y,S)>=max(0,d-2)}
                     e^(-gamma dist(y,S)).             (35)

This function is finite and nonincreasing. Since S is finite,
shell counting bounds it by C_E exp(-gamma d/2).
If dist(A^c,U)>=d, (1), the four spin columns and (35) give

|S_A(P')-S_A(P_-(h_c))|<=Ecal(d).                        (36)

Combining (20), (32) and (36) proves

|S_A(P)-S_A(P0)|
 <=C_loc^c kappa A_lat epsilon^2(1+|log epsilon|)^4
    +r_b log 2+Ecal(d).                                (37)

One may use the continuous nonincreasing extension of (35), or its
integer-distance version, to define d0 as the first distance where
Ecal<=log 2. It exists. For d>=d0 the final two terms are at most
(r_b+1)log 2.

The constants in (35) explicitly depend on S, ||V||, the comparison
gap, the chosen separating contour and n0. No uniform estimate for
growing pockets or collapsing contour separations is claimed.
There need be no gap of h at zero. Without finite r_b the finite
rank conclusion is not asserted.

## 7. Correspondence and audit of the supplied proofs

The manuscript's B.1 assertion is supported by
"declared_area_law.tex", SA1--SA9. In particular its line
"|S(A_r)-S_L(A_r)|<=C_mu r^2 sqrt(log(2+L))/L" supplies an
area-uniform entropy approximation, and SA5 supplies the digital
sphere count. Those arguments retain fixed mass, the infinite parent
and arbitrary center offset. They do not certify any decimal value.

For B.2, that file does not prove the stated Lovasz-extension limit.
Section 2 above supplies it.

For B.3, "declared_local_area.tex", SC6, states
"P=P0+P1+R2, |(R2)_xy|<=Cepsilon^2 exp(-c|x-y|)" and gives the
entropy regularization mechanism. Section 3 above makes the
corrector, weighted norm, projection construction and endpoint
estimate explicit with the manuscript's hypotheses. No G dependence
enters except through the chosen fields and their stated bounds.

For B.4, the same file's SC4--SC5 gives the low-rank/small-norm
decomposition and exponent, but says that the "flat finite-parent
leading-coefficient limit is still a required separately established
limit." Section 4 supplies that comparison and makes clear that both
flat projectors have the same mass.

The supplied material does not derive the original B.5 constants
c1,c3. Section 5 proves different explicit sufficient constants.
The paper prints (31); the old expression and its constants are withdrawn.

No complete finite-pocket transfer proof with the manuscript's
constant dependence appears in the two supplied area files.
Section 6 supplies a finite-rank version and its actual dependencies.
It does not claim a pocket-uniform constant.
