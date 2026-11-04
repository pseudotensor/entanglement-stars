# Derivation text moved from ms_fermion_new to the companion repository (owner ruling, 2 Nov 2026)

Each section holds, verbatim, the LaTeX of one earned-edit passage as it stood in the paper
before the 31-page cut: first the full text as first committed today (the complete
derivation), then the shortened form committed in 96280dffc7c. The paper now keeps only the
statement, its hypotheses, the essential constants, every hedge sentence and every todo mark,
with "derivation in Ref. McKinneyCode2026" pointing here. These sections are meant to be
committed to the companion repository (github.com/pseudotensor/entanglement-stars) as the
derivation notes behind Secs. II D, IV C and IV D.

Symbol conventions follow the paper: \mathsf Q propagated sea, \mathsf Q_1 first adiabatic
corrector, \mathsf d_mu band-reference defect, \mathsf M source-rate generator, \varPi
polarization operator, \mathsf E remaining sea departure, \mathcal D_mu[B] = i Tr(B[hbar, A_mu]),
c_P the polarization current constant, \Delta_\star = C - P_\star.


## 1. Bond-gradient expansion (Sec. II D): full text as committed in 25f859c0d12

```latex
The first declaration's endpoint allocation gives every hop \(d\)
of \(h[u]\) the coefficient \(\tfrac12(u_x+u_{x+d})\) of its field.
Replacing it by the midpoint value \(u(x+d/2)\) gives the midpoint
form, whose spatial derivative expansion is the lattice Moyal
expansion, an asymptotic series under the derivative-weighted hop
bounds below. The endpoint-minus-midpoint generator difference obeys
\(\|\delta h\|\le\tfrac{3a}8\|\nabla^2N\|_\infty+\tfrac a4\sum_i\|\partial_i^2v^i\|_\infty+\tfrac{a^2}4\sum_k\|\nabla^2b_k\|_\infty\),
where the Hessian norm is the supremum of second directional
derivatives; this bounds the exact difference by Taylor's integral
formula. After momentum integration, its second-gradient flat-sea
energy density is
\(-\tfrac14\sum_i\partial_i^2N\int_p\sin^2(ap_i)/\varepsilon(p)\),
the leading midpoint-symbol density, not the complete local endpoint
defect. On a periodic parent, or under the summability conditions
for Poisson summation, the integrated exact flat-sea defect vanishes
for fields filtered with \(K_\ell*B_a\): the cell factor kills every
nonzero reciprocal-lattice contribution to the translated sampling
sums. With a Gaussian kernel and no cell factor the multiplier at
reciprocal vector \(\mathsf G\) (sans-serif, not Newton's constant)
is \(e^{-\sigma^2|\mathsf G|^2/2}\) and the alias terms survive; an
absolute alias bound then needs the field's Fourier amplitudes and
the sum over reciprocal vectors. Two-step hops have lattice
midpoints and no alias term. Open boundaries retain explicit
boundary terms.

For a translation-invariant vertex \(A=\sum_dA_dT^d\),
\((T^du)_x=u_{x+d}\), the midpoint-hop covariance
\(C_d(y)=C_{y+d/2,y-d/2}\) with controlled spatial derivatives, and
its Wigner symbol \(\mathcal C(x,p)\),
\begin{equation*}
a^{-3}\Tr\bigl(C\{\Pi_x,A\}/2\bigr)
=\int_p\tr(A\mathcal C)
-\frac18\int_p\tr(A_{,p_ip_j}\mathcal C_{,x_ix_j})+\mathsf R_4,
\end{equation*}
with
\(|\mathsf R_4|\le\frac{a^{-3}}{384}\sum_d\|A_d\|_1|d|^4\sup_y\|\partial_{\widehat d}^{\,4}C_d(y)\|_{\rm op}\).
Odd orders cancel in the symmetric endpoint average. This estimate
requires the displayed derivatives; mass alone neither supplies nor
excludes them. Each bond has its own shifted source kernel, with
second-moment increment \(d_id_j/4\). Unit, two-step and
face-diagonal hops therefore have different increments, including
the signed mixed moments of face-diagonal hops.

On a controlled massive prepared-band expansion, write the symbol
of \(h[u]\) as \(\mathsf h=\mathsf h_0+\mathsf h_1+\mathsf h_2+\cdots\),
\(\mathsf h_0=N\hat h_0+v\cdot q\), \(\mathsf h_1=b\cdot\Sigma^{\rm lat}\),
with \(\hat h_0\) the symbol of \(h_0\) and \(\mathsf h_2\) the
second-gradient symbol of the endpoint-minus-midpoint difference
\(\delta h\). The first two Moyal projector correctors reproduce the
bulk response \(W_2=-\tfrac14\int F:\mathsf T:F/N\) of
Sec.~\ref{sec:model-sources}, \(F_{ij}=\partial_iv_j-2\epsilon_{ijk}b_k\).
Coefficient derivatives are taken with \(N,v,b\) independent;
\(F=NK\) only after imposing \(b=\nabla\times v/4\). Holding the
lower-order symbol fixed, \(\mathsf h_2\) changes the off-diagonal
second corrector by
\(\delta\pi_2=-[P\mathsf h_2(I-P)+(I-P)\mathsf h_2P]/(2N\varepsilon)\),
\(P\) the prepared band of the frozen symbol. Its integrated linear
flat-sea contribution is the divergence and alias term above, with
a field-independent coefficient, because below the pocket threshold
the prepared band of the frozen symbol is \(P_0\) at every clock and
flow. This does not imply that general local moments or
higher-order reference derivatives are unchanged.

The endpoint clock vertex has an \(O(k^2)\) interband amplitude and
hence no second-gradient quadratic energy response. On the lattice
its difference from the midpoint symbol, \(\mathsf h_2\), enters the
fourth-order coefficient and must be retained there; in the
continuum limit the two vertices coincide. For weak static
perturbations about \(N=1\), \(v=b=0\), with
\(u(x)=\int_ku_ke^{ik\cdot x}\), \(\int_k=\int d^3k/(2\pi)^3\),
\(v_{T,k}\) the part of \(v_k\) transverse to \(k\) and \(\Lambda\)
the momentum cutoff, of order \(1/a\), the leading continuum
logarithmic part of the quadratic energy, with eight valleys and no
\(\tfrac12\) prefactor in the Fourier quadratic form, is
\begin{equation*}
W^{[2]}_{\log}=\int_k|k|^4\log\frac{\Lambda}{\mu_{\rm IR}}
\Bigl[-\frac{|\delta N_k|^2}{30\pi^2}+\frac{|v_{T,k}|^2}{40\pi^2}\Bigr],
\qquad b=\tfrac14\nabla\times v,
\end{equation*}
where \(\mu_{\rm IR}\sim\max(m,|k|)\) denotes the leading-logarithmic
infrared scale. The longitudinal quadratic logarithmic coefficient
vanishes. With independent \(b=0\), the orbital transverse flow
coefficient is instead \(1/(15\pi^2)\). These are quadratic-response
coefficients, not a complete nonlinear fourth-gradient term of the
vacuum energy, a massless retained-order bound, or a bound on
reference derivatives.

For homogeneous preparation and a covariance
\(C=\sum_\pm P_\pm\mathsf F_\pm P_\pm\), with \(P_\pm(p)\) the band
projectors of \(\hat h_0\), slowly varying \(\mathsf F_\pm\)
(sans-serif, distinct from the field \(F\)), band blocks
\(f_\pm=P_\pm\mathsf F_\pm P_\pm\) and band curvature
\(\varOmega^\pm_{jk}=iP_\pm[\partial_{p_j}P_\pm,\partial_{p_k}P_\pm]P_\pm\)
(the variant distinguishes it from the rotation \(\Omega\)), the
unweighted kinetic flux has leading term
\(S^{(0)}_{ij}=\sum_\pm\int_p(\pm q_iq_j/\varepsilon)\tr f_\pm\),
momentum \(q_i\) carried at velocity \(\pm q_j/\varepsilon\) with the
hole sign of Sec.~\ref{sec:results-filling} and Eq.~\eqref{eq:C-rate},
and first-gradient curvature term
\(S^{(1)}_{ij}=-\sum_\pm\int_pq_i(\pm\varepsilon)\tr[(\partial_{x_k}f_\pm)\varOmega^\pm_{jk}]\).
The latter is a momentum-magnetization curl and vanishes for
occupations scalar within each Dirac band because the band-curvature
trace vanishes. Angular averaging alone does not establish that
cancellation.

General preparation and evolution require the exact split
\(\Delta_\star=C-P_\star\),
\(\Delta_\star^{\rm off}=P_\star\Delta_\star(I-P_\star)+(I-P_\star)\Delta_\star P_\star\),
\(a^3(S^{\rm op}_{x,ij}-S^{\rm kin}_{x,ij})=\Tr(\Delta_\star^{\rm off}\mathsf s_{x,ij})\),
which includes sea and matter coherence. For homogeneous preparation
and smooth small transfers, such coherence can enter the local
current at first order and the energy at second order, without a
universal spacing-squared suppression. A spin-resolved occupation
can also have a zeroth-order spin moment whose curl contributes to
physical momentum at first order. Source rates are the exact lattice
commutator rates minus \(\mathcal W''[\bar u]\dot{\bar u}\), followed
by the averaging of Eq.~\eqref{eq:dm-average} and the spin
improvement \(\tfrac14\nabla\times W_\ell Z\). The vanishing
integrated generator-weighted reference response is not local
cancellation. Neither covariant source conservation, polarization
stress, nor small retained-order balance defects follows from smooth
fields or the preceding quadratic coefficients alone.
\todo{the two-route record check of the bond-gradient expansion}
```

## 1b. Bond-gradient expansion: shortened form committed in 96280dffc7c

```latex
The first declaration's endpoint allocation, with hop coefficients
\(\tfrac12(u_x+u_{x+d})\), differs from the midpoint form
\(u(x+d/2)\) by a generator bounded by the spacing times the fields'
second derivatives (constants \(3a/8\), \(a/4\), \(a^2/4\) for
clock, flow and spin), and the midpoint form's spatial derivative
expansion is the lattice Moyal expansion, asymptotic under
derivative-weighted hop bounds with a fourth-order remainder of
coefficient \(1/384\) in the fourth midpoint-hop derivatives of the
covariance, which mass alone neither supplies nor excludes
(Ref.~\cite{McKinneyCode2026}). On a periodic parent the integrated
exact flat-sea endpoint defect vanishes for fields filtered with
\(K_\ell*B_a\), the cell factor killing every reciprocal-lattice
term; a Gaussian kernel without it leaves alias terms, and open
boundaries retain boundary terms. On a controlled massive
prepared-band expansion the first two Moyal projector correctors
reproduce \(W_2\) of Sec.~\ref{sec:model-sources} with \(N,v,b\)
independent, the endpoint difference changing only the off-diagonal
second corrector with a field-independent integrated flat-sea
contribution, since below the pocket threshold the prepared band of
the frozen symbol is \(P_0\) at every clock and flow; general local
moments and higher-order reference derivatives are not thereby
fixed, and the endpoint clock vertex, with no second-gradient
quadratic response, enters only at fourth order on the lattice. For
weak static perturbations about flat fields the leading continuum
logarithm of the quadratic energy for eight valleys, per
\(|k|^4\log(\Lambda/\mu_{\rm IR})\) with \(\Lambda\) the momentum
cutoff of order \(1/a\) and \(\mu_{\rm IR}\sim\max(m,|k|)\), is
\(-1/(30\pi^2)\) for the clock, \(+1/(40\pi^2)\) for the transverse
flow at \(b=\tfrac14\nabla\times v\), zero for the longitudinal flow
and \(+1/(15\pi^2)\) for the orbital transverse flow at independent
\(b=0\): quadratic-response coefficients, not a complete nonlinear
fourth-gradient vacuum energy, a massless retained-order bound or a
bound on reference derivatives. For homogeneous preparation and a
covariance slowly varying within the band projectors of the symbol
of \(h_0\), the first-gradient kinetic flux is a
momentum-magnetization curl in the band curvature that vanishes for
occupations scalar within each Dirac band, not by angular averaging,
the leading flux carrying momentum \(q_i\) at velocity
\(\pm q_j/\varepsilon\) with the hole sign of
Sec.~\ref{sec:results-filling} and Eq.~\eqref{eq:C-rate}; general
preparation and evolution need the exact split
\(a^3(S^{\rm op}_{x,ij}-S^{\rm kin}_{x,ij})=\Tr(\Delta_\star^{\rm off}\mathsf s_{x,ij})\),
\(\Delta_\star=C-P_\star\) and \(\Delta_\star^{\rm off}\) its part
off-diagonal in \(P_\star\), whose sea and matter coherence can
enter the current at first and the energy at second order without a
universal spacing-squared suppression, a spin-resolved occupation
adding a zeroth-order spin moment whose curl enters physical
momentum at first order. Source rates are the exact lattice
commutator rates minus \(\mathcal W''[\bar u]\dot{\bar u}\), then
the averaging of Eq.~\eqref{eq:dm-average} and the spin improvement
\(\tfrac14\nabla\times W_\ell Z\); the vanishing integrated
generator-weighted reference response is no local cancellation, and
neither covariant source conservation, polarization stress nor small
retained-order defects follows from smooth fields or these
coefficients.
\todo{the two-route record check of the bond-gradient expansion}
```

## 2. Stress exact algebra (Sec. II D): full text as committed in 4fd34e4c06d

```latex
The exact bookkeeping of the sources and of the stress split is
stated here for later use. On a finite parent, for the
field-independent vertices \(A_\mu\) (the allocations \(e_x\),
\(p_{x,i}\) and \(\mathsf z_{x,k}\)), define
\(\mathcal D_\mu[B]=i\Tr(B[\bar h,A_\mu])\) for any one-body \(B\),
equivalently its allocated bond flux plus local force; this equals
the rate of \(\Tr(BA_\mu)\) whenever \(i\dot B=[\bar h,B]\). With
\(\mathsf Q\) the propagated sea and \(\Delta=C-\mathsf Q\), exactly
\begin{equation*}
\dot s^{\rm sea}_\mu=\mathcal D_\mu[\mathsf Q]-(W_2''\dot{\bar u})_\mu,\qquad
\dot s_\mu-\dot s^{\rm sea}_\mu=\mathcal D_\mu[\Delta],\qquad
\langle\bar u,W_2''\dot{\bar u}\rangle=0 .
\end{equation*}
On a smooth time-dependent field path where the level-following sum
\(P\) of eigenprojectors of \(\bar h\), continued from \(P_\star\),
is twice differentiable and has positive occupied-unoccupied
separation \(g_{\rm sep}\), define the band-reference defect, first
adiabatic corrector, source-rate generator and polarization operator
as follows,
\begin{equation*}
\mathsf d_\mu=\Tr(PA_\mu)-r_\mu,\quad
[\bar h,\mathsf Q_1]=i\dot P,\ P\mathsf Q_1P=(I-P)\mathsf Q_1(I-P)=0,\quad
\mathsf M=C-P-\mathsf Q_1,\quad
\varPi=P-P_\star+\mathsf Q_1,
\end{equation*}
with \(\|\mathsf Q_1\|_2\le\|\dot P\|_2/g_{\rm sep}\); the sans-serif
\(\mathsf Q\), \(\mathsf M\) and the variant \(\varPi\) distinguish
these from the current \(Q_i\), the enclosed mass \(\mathcal M\) and
the branch projector \(\Pi\) of Sec.~\ref{sec:model-operators}. Then,
exactly,
\begin{equation*}
\dot s_\mu=\mathcal D_\mu[\mathsf M]+\dot{\mathsf d}_\mu,\qquad
s_\mu=\Tr(\mathsf MA_\mu)+\Tr(\mathsf Q_1A_\mu)+\mathsf d_\mu,
\end{equation*}
with the difference between the band-moment rate and the reference
rate retained in \(\dot{\mathsf d}_\mu\). The remaining departure
\(\mathsf E=\mathsf Q-P-\mathsf Q_1\), sans-serif to distinguish it
from the boundary charge \(E\), obeys
\(i\dot{\mathsf E}=[\bar h,\mathsf E]-i\dot{\mathsf Q}_1\) from
\(\mathsf E(t_0)=-\mathsf Q_1(t_0)\); \(\mathsf Q_1\), \(\mathsf E\)
and \(\mathsf d\) all gravitate. The clock-bond stresses split
exactly as
\begin{equation*}
S^{\rm op}[C-P_\star]=S^{\rm op}[\mathsf M]+S^{\rm op}[\varPi],\qquad
\delta S=S^{\rm op}[\mathsf M^{\rm off}]+S^{\rm op}[\varPi^{\rm off}],
\end{equation*}
off-diagonal parts relative to \(P_\star\), where for identical
fields, sources, averaging, clock and normalization
\(\delta S_{ij}=S^{\rm op}_{ij}[C-P_\star]-S^{\rm kin}_{ij}[C-P_\star]\)
and, exactly, the evolution residuals with the two stresses differ by
\(R^{\rm op}_{ij}-R^{\rm kin}_{ij}=-8\pi GN(\delta S_{ij}-\tfrac12\delta_{ij}\operatorname{tr}\delta S)\),
an algebraic identity. Here \(S^{\rm op}[\varPi]\) is the
polarization-flux diagnostic and \(S^{\rm op}[\mathsf M]\) the flux
of the source-rate generator; \(S_{ij}=S^{\rm kin}_{ij}[C-P_\star]\)
remains the paper's stress, and \(S^{\rm op}[\mathsf M]\) and
\(S^{\rm op}[\varPi]\) print as diagnostics. Conditional on the
corresponding local bulk expansions, with preparation terms included
consistently, \(\mathsf d_N,\mathsf d_v\) are formally fourth
spatial-gradient order and \(\mathsf d_Z\) third; the energy and
momentum moments of \(\mathsf Q_1\) are formally fourth
spacetime-derivative order. None of these remainder bounds is
established here. For an approximate projector and corrector, put
\(\mathsf R_1=[\bar h,\mathsf Q_1]-i\dot P\); the source-rate identity
acquires the additional term
\(-i\Tr(([\bar h,P]+\mathsf R_1)A_\mu)\). Reading
\(S^{\rm op}[\mathsf M]\) as a conserving continuum stress requires
bounds on \(\mathsf d\), \(\dot{\mathsf d}\), \(\Tr(\mathsf Q_1A_\mu)\)
and the bond-gradient, advection and allocation defects. The held
projector of Sec.~\ref{sec:results-lattice} is a comparator intended
to follow the same occupations as \(P\); the audit checks that
continuation and measures \(P_{\rm held}-P\), rather than assuming
their identification. A trapping root or a positive frozen gap
implies none of these bounds. Averaging, endpoint, higher-gradient
and radial-channel terms must also be retained; the measured split is
read in Sec.~\ref{sec:star}. \todo{the record audit of the stored
histories on two routes} The balance of a directly coupled species is
stated with its converter in Sec.~\ref{sec:page}.
```

## 2b. Stress exact algebra: shortened form committed in 96280dffc7c

```latex
The exact bookkeeping of the sources and of the stress split is
stated here for later use (Ref.~\cite{McKinneyCode2026}). On a
finite parent, for the field-independent vertices \(A_\mu\) (the
allocations \(e_x\), \(p_{x,i}\), \(\mathsf z_{x,k}\)) and any
one-body \(B\), \(\mathcal D_\mu[B]=i\Tr(B[\bar h,A_\mu])\) equals
the rate of \(\Tr(BA_\mu)\) whenever \(i\dot B=[\bar h,B]\); with \(\mathsf Q\)
the propagated sea and \(\Delta=C-\mathsf Q\), exactly
\(\dot s^{\rm sea}_\mu=\mathcal D_\mu[\mathsf Q]-(W_2''\dot{\bar u})_\mu\),
\(\dot s_\mu-\dot s^{\rm sea}_\mu=\mathcal D_\mu[\Delta]\) and
\(\langle\bar u,W_2''\dot{\bar u}\rangle=0\). On a smooth field path
where the level-following sum \(P\) of eigenprojectors of \(\bar h\)
continued from \(P_\star\) is twice differentiable with positive
occupied-unoccupied separation \(g_{\rm sep}\), the band-reference
defect \(\mathsf d_\mu=\Tr(PA_\mu)-r_\mu\), the first adiabatic
corrector \(\mathsf Q_1\) with \([\bar h,\mathsf Q_1]=i\dot P\),
\(P\mathsf Q_1P=(I-P)\mathsf Q_1(I-P)=0\) and
\(\|\mathsf Q_1\|_2\le\|\dot P\|_2/g_{\rm sep}\), the source-rate
generator \(\mathsf M=C-P-\mathsf Q_1\) and the polarization operator
\(\varPi=P-P_\star+\mathsf Q_1\) (letters distinct from the current
\(Q_i\), the mass \(\mathcal M\) and the branch projector \(\Pi\))
give, exactly,
\(\dot s_\mu=\mathcal D_\mu[\mathsf M]+\dot{\mathsf d}_\mu\) and
\(s_\mu=\Tr(\mathsf MA_\mu)+\Tr(\mathsf Q_1A_\mu)+\mathsf d_\mu\),
the difference between band-moment and reference rates retained in
\(\dot{\mathsf d}_\mu\); the remaining departure
\(\mathsf E=\mathsf Q-P-\mathsf Q_1\) (not the boundary charge
\(E\)) obeys \(i\dot{\mathsf E}=[\bar h,\mathsf E]-i\dot{\mathsf Q}_1\)
from \(\mathsf E(t_0)=-\mathsf Q_1(t_0)\), and \(\mathsf Q_1\),
\(\mathsf E\) and \(\mathsf d\) all gravitate. The clock-bond
stresses split exactly as
\(S^{\rm op}[C-P_\star]=S^{\rm op}[\mathsf M]+S^{\rm op}[\varPi]\) and
\(\delta S=S^{\rm op}[\mathsf M^{\rm off}]+S^{\rm op}[\varPi^{\rm off}]\),
off-diagonal parts relative to \(P_\star\), where
\(\delta S_{ij}=S^{\rm op}_{ij}[C-P_\star]-S^{\rm kin}_{ij}[C-P_\star]\)
at identical fields, sources, averaging, clock and normalization and
the two stresses' evolution residuals differ exactly by
\(R^{\rm op}_{ij}-R^{\rm kin}_{ij}=-8\pi GN(\delta S_{ij}-\tfrac12\delta_{ij}\operatorname{tr}\delta S)\);
\(S^{\rm op}[\varPi]\) is the polarization-flux diagnostic,
\(S^{\rm op}[\mathsf M]\) the flux of the source-rate generator, and
\(S_{ij}=S^{\rm kin}_{ij}[C-P_\star]\) remains the paper's stress
with both printed as diagnostics. Conditional on the corresponding
local bulk expansions with preparation terms included,
\(\mathsf d_N,\mathsf d_v\) are formally fourth spatial-gradient
order, \(\mathsf d_Z\) third, and the energy and momentum moments of
\(\mathsf Q_1\) fourth spacetime-derivative order; none of these
remainder bounds is established here, and an approximate projector
and corrector add \(-i\Tr(([\bar h,P]+\mathsf R_1)A_\mu)\) to the
rate identity, \(\mathsf R_1=[\bar h,\mathsf Q_1]-i\dot P\). Reading
\(S^{\rm op}[\mathsf M]\) as a conserving continuum stress requires
bounds on \(\mathsf d\), \(\dot{\mathsf d}\), \(\Tr(\mathsf Q_1A_\mu)\)
and the bond-gradient, advection and allocation defects; the held
projector of Sec.~\ref{sec:results-lattice} is a comparator meant to
follow the same occupations as \(P\), and the audit checks that
continuation and measures \(P_{\rm held}-P\) rather than assuming
their identification; a trapping root or a positive frozen gap
implies none of these bounds. Averaging, endpoint, higher-gradient
and radial-channel terms must also be retained; the measured split is
read in Sec.~\ref{sec:star}. \todo{the record audit of the stored
histories on two routes} The balance of a directly coupled species is
stated with its converter in Sec.~\ref{sec:page}.
```

## 3. Polarization-moment bound (Sec. IV D): full text as committed in 88e8ac50f96

```latex
The polarization moments can be bounded without assuming a cap on
them in advance: their energy contribution has a favourable sign,
and averaging controls their current by that energy. With
\(X=F/N\), the current-time polarization energy density is
\(p=\tfrac14X:\mathsf T:X\). For any real matrix \(A\), writing
\(z_i=\cos(ap_i)(Aq)_i\), the integrand of \(A:\mathsf T:A\) is
\((m^2|z|^2+|d\times z|^2)/\varepsilon^5\), so \(\mathsf T\) is
positive semidefinite throughout the zone, without a valley
approximation. On a fixed finite parent realizing this functional
by positive quadrature and exact adjoint differentiation, with
boundary contributions retained, let \(P\) and \(j\) be its
smoothed, angularly averaged polarization energy and radial
current. Cauchy-Schwarz in the positive tensor form gives
\(j^2\le c_P^2P\), and Jensen's inequality preserves this estimate
under angular averaging; here \(c_P\) is computed from the actual
differentiation operator and kernel, whose differentiated tests
must be dominated by the kernel at every contributing site. Write
the spherical sources as \(\rho=R-P\) and \(J=B-j\), absorbing the
fixed preparation moments and any kernel-normalization allowance
into \(|R|\le B_\rho\) and \(|B|\le B_J\). Completing the square
gives \(\rho+wJ\le B_\rho+B_J|w|+c_P^2w^2/4\). Thus, for any
\(\theta>0\), the charge row and \(w^2=2G\mathcal M/r\) imply
\(\mathcal M'\le4\pi A_\theta r^2+8\pi GD_\theta r\mathcal M\),
where \(A_\theta=B_\rho+B_J^2/(4\theta)\) and
\(D_\theta=\theta+c_P^2/4\); comparison from the regular centre
gives \(\mathcal M(r)\le Y_L^2r^3\) and
\(|w(r)|\le\sqrt{2G}\,Y_Lr\) on \(0\le r\le L\), with
\(Y_L^2=(4\pi A_\theta/3)e^{4\pi GD_\theta L^2}\). Retaining half
the negative polarization-energy term and integrating the charge
row gives \(\int_{B_L}P\le Q_L\) over the ball \(B_L\) of radius
\(L\), where \(Q_L=(4\pi L^3/3)(2B_\rho+2B_JW_L+c_P^2W_L^2)\) and
\(W_L=\sqrt{2G}\,Y_LL\). If every functional site has positive
kernel mass \(\mu_x=\int_{B_L}k(y-x)\,d^3y\), \(k\) the averaging
kernel, positivity then gives
\(P_{\max}\le\|k\|_\infty Q_L/\min_x\mu_x\) and
\(|j|\le c_P\sqrt{P_{\max}}\); the fixed adjoint maps likewise
bound every site's energy, current and spin moments, including
their fixed preparation subtraction. At a regular centre the
sharper sign estimate gives \(K_t^2(0)\le(8\pi G/3)B_\rho\); when
\(F_\star=0\) and averaging is normalized, \(B_\rho=\bar\rho\),
giving the upper bound \(10.4/a^2\) at \(ma=0.2\) and
\(G/a^2=1/4\). These are finite-parent bounds for the displayed
cubic reference, with rapidly growing constants, not estimates of
attained densities or certification of the chain subtraction. They
control the clock integral on trapped intervals but establish
neither its integrability near vanishing enclosed energy, an
invertible constraint response, nor the missing tidal-stress bound
or persistence of the regular branch.
```

## 3b. Polarization-moment bound: shortened form committed in 96280dffc7c

```latex
The polarization moments can be bounded without assuming a cap on
them in advance (Ref.~\cite{McKinneyCode2026}): with \(X=F/N\) the
current-time polarization energy density is
\(p=\tfrac14X:\mathsf T:X\), and the integrand of \(A:\mathsf T:A\)
for any real matrix \(A\) is \((m^2|z|^2+|d\times z|^2)/\varepsilon^5\)
with \(z_i=\cos(ap_i)(Aq)_i\), so \(\mathsf T\) is positive
semidefinite throughout the zone without a valley approximation. On
a fixed finite parent realizing this functional by positive
quadrature and exact adjoint differentiation, boundary contributions
retained, Cauchy-Schwarz in the positive tensor form bounds the
smoothed, angularly averaged polarization current by the square root
of its energy, \(j^2\le c_P^2P\), with \(c_P\) from the actual
differentiation operator and kernel, whose differentiated tests must
be dominated by the kernel at every contributing site, Jensen's
inequality preserving the bound under angular averaging. Absorbing
the fixed preparation moments and any kernel-normalization allowance
into bare caps \(B_\rho\), \(B_J\) on the remaining energy and
current, completing the square in the charge row and comparing from
the regular centre bounds the enclosed energy by a cubic,
\(\mathcal M(r)\le Y_L^2r^3\), and the normalized flow by
\(\sqrt{2G}\,Y_Lr\) on \(0\le r\le L\), hence the integrated
polarization energy and, through the kernel's positivity, every
site's energy, current and spin moments with their preparation
subtraction, with constants growing exponentially in \(GL^2\). At a
regular centre the sharper sign estimate gives
\(K_t^2(0)\le(8\pi G/3)B_\rho\), and \(B_\rho=\bar\rho\) when
\(F_\star=0\) with normalized averaging, the upper bound
\(10.4/a^2\) at \(ma=0.2\), \(G/a^2=1/4\). These are finite-parent
bounds for the displayed cubic reference with rapidly growing
constants, not estimates of attained densities or certification of
the chain subtraction; they control the clock integral on trapped
intervals but establish neither its integrability near vanishing
enclosed energy, an invertible constraint response, nor the missing
tidal-stress bound or persistence of the regular branch.
```

## 4. Dressing energy about the nearest determinant (Sec. IV C): full text as committed in 88e8ac50f96

```latex
Ordering is fixed by the prepared sea, whereas stationary dressing
is evaluated about the fixed-occupation determinant of the slice
Hamiltonian nearest the stored covariance in the sea-plus-dust
number sector. We form the boundary-charge Hessian with the full
polarization response, diagonalize the one-body Hamiltonian
exactly, and rewrite the same prepared-sea-ordered operator using
contractions in that determinant. The Hartree reordering terms
cancel, leaving an exchange correction to the single-excitation
amplitude and a first-order exchange scalar; the double-excitation
vertex is unchanged. The connected second-order energy is the
signed single- and double-excitation sum, while the squared
first-order state correction uses squared denominators. Positive
denominators make the connected term nonpositive; zero denominators
require degenerate treatment. These are retained-order diagnostics,
not controlled dressed energies. The earlier held-projector values
evaluate a different comparator and do not bound these corrections,
and no dressing energy is assigned to the subtraction.
```

## 4b. Dressing energy: shortened form committed in 96280dffc7c

```latex
Ordering is fixed by the prepared sea, whereas stationary dressing
is evaluated about the fixed-occupation determinant of the slice
Hamiltonian nearest the stored covariance in the sea-plus-dust
number sector: forming the boundary-charge Hessian with the full
polarization response, diagonalizing the one-body Hamiltonian
exactly and rewriting the prepared-sea-ordered operator with
contractions in that determinant, the Hartree reordering terms
cancel, leaving an exchange correction to the single-excitation
amplitude and a first-order exchange scalar with the
double-excitation vertex unchanged; the connected second-order
energy is the signed single- and double-excitation sum, nonpositive
for positive denominators, zero denominators requiring degenerate
treatment, and the squared first-order state correction uses
squared denominators (Ref.~\cite{McKinneyCode2026}). These are
retained-order diagnostics, not controlled dressed energies; the
earlier held-projector values evaluate a different comparator and do
not bound these corrections, and no dressing energy is assigned to
the subtraction.
```

## 5. Continuation certificate (Sec. IV D): full text as committed in 88e8ac50f96

```latex
A sufficient certificate of continuation on a specified finite
realization combines a current bound with an inverse-Jacobian
bound, together with verified centre, source-cap and
reference-domain conditions. With \(F(C,u)=0\) the implemented
constraint rows, including their regular-centre closure and the
outer normalization in use, and \(F_u\) their Jacobian in the
fields, the joint equations are \(i\dot C=[\bar h,C]\) and
\(\dot u=-F_u^{-1}F_C\dot C\) wherever \(F_u\), evaluated with
Eq.~\eqref{eq:dm-reference}'s full reference response, is
invertible. The covariance remains in \(0\le C\le I\); the
certificate must additionally verify that the solution stays on an
admitted inward branch and that the rows have a regular extension
at every represented radius where the enclosed energy vanishes.

The current bounds the clock cell by cell once the implemented form
of the clock law is verified. For cells labelled by their outer
radii \(r_i=ia\), \(L=n_{\rm ch}a\), let \(\mathcal J_c\) satisfy
\(r_iJ_{\ell r,i}^2\le\mathcal J_c\mathcal M_i\) at every cell, with
the same full current as the clock law; on the chain, \(r\le L\)
turns the preceding paragraph's witness into
\(\mathcal J_c\le Lq_*^2S_\kappa/(m\delta)\) at the scanned records.
If the implemented clock law advances \(\ln N\) across cell \(i\) by
\(4\pi Ga\,J_{\ell r,i}\sqrt{r_i^3/(2G\mathcal M_i)}\), then for
positive cell masses each increment is at most
\(2\pi r_ia\sqrt{2G\mathcal J_c}\) in absolute value, and a cell of
zero enclosed energy requires the verified regular extension. Thus
the discrete total variation satisfies
\(A_N^{\rm d}\le A_c=\pi L(L+a)\sqrt{2G\mathcal J_c}\), and
\(N(L)e^{-A_c}\le N_i\le N(L)e^{A_c}\). Unit normalization at the
last site gives \(N(L)=1\); normalization at infinity instead
requires an enclosed exterior clock contribution. Conversely, on
this fixed quadrature, uniform upper and positive lower clock bounds
bound the increments and hence the witness, mass zeros treated the
same way; neither direction says anything about the centre inside
the first cell.

The charge row supplies the flow bound. The certificate must verify
that the clock component of \(DW_2[u]\), before the preparation
value \(DW_2[u_\star]\) is subtracted, is nonnegative for the
implemented allocation and averaging, so that exclusion and the
fixed preparation moment cap the full energy source from above.
Together with
\(|wJ_{\ell r}|\le\sqrt{2G\mathcal J_c}\,\mathcal M/r\), this gives
a regular-centre comparison bound on enclosed energy and normalized
flow, and the clock bound then bounds \(v=Nw\). The required
reference derivatives and the extension of the constraint map must
be enclosed on the resulting regular domain rather than inferred
from bounded field values alone.

On a connected enclosure of the solution, a tube on which these
conditions hold at every time, uniform bounds on \(\mathcal J_c\)
and \(\|F_u^{-1}\|\) give unique joint continuation over the
certified interval. Failure to enclose any condition is
inconclusive, and even a certified loss of invertibility is not by
itself a dynamical singularity. Exclusion and charge conservation
do not by themselves keep the regular branch, and the leading laws
supply no lattice-scale geometric law once their gradient expansion
fails, so a Pauli-supported interior is a possible outcome, not an
assumption that closes the question. The stored-record diagnostics
above do not supply this whole-path certificate under
Eq.~\eqref{eq:dm-reference}. The deciding calculation is a validated
enclosure of the joint evolution from the archived preparation of
the two-field history with nominal \(r_s=80a\), its five heavy
channels and the full-weight light field of 16 full chains evolved
together under Eq.~\eqref{eq:dm-reference}'s sources, each species
with its own reference and preparation calibration, over
\(0\le t/a\le1112\), \(13.9\) of its nominal \(r_s\), on two
independent implementations sharing only that preparation and its
inputs. A pass decides unique continuation of that finite history,
not of the reference collapse's longer history or of every admitted
history. \todo{validate the connected solution tube of the two-field history with nominal \(r_s=80a\) under Eq.~\eqref{eq:dm-reference} for both species, from its archived preparation at \(t=0\) through \(t=1112a\), the light field at full weight with its own reference and preparation calibration, enclosing preparation and propagated errors, full-reference residuals and derivatives of both species, admitted-branch and centre closure, the local source cap, the witness and inverse-Jacobian bounds, the clock quadrature and outer normalization with the exterior tail, and the flow and reference-domain bounds; require checkpoint agreement between the two independently validated routes}
```

## 5b. Continuation certificate: shortened form committed in 96280dffc7c

```latex
A sufficient certificate of continuation on a specified finite
realization combines a current bound with an inverse-Jacobian bound
and verified centre, source-cap and reference-domain conditions
(Ref.~\cite{McKinneyCode2026}). With \(F(C,u)=0\) the implemented
constraint rows, including their regular-centre closure and the
outer normalization in use, and \(F_u\) their Jacobian in the
fields, the joint equations are \(i\dot C=[\bar h,C]\) and
\(\dot u=-F_u^{-1}F_C\dot C\) wherever \(F_u\), evaluated with
Eq.~\eqref{eq:dm-reference}'s full reference response, is
invertible; the covariance stays in \(0\le C\le I\), and the
certificate must also verify that the solution stays on an admitted
inward branch and that the rows extend regularly at every
represented radius where the enclosed energy vanishes. The current
bounds the clock cell by cell once the implemented form of the clock
law is verified: if \(\mathcal J_c\) satisfies
\(r_iJ_{\ell r,i}^2\le\mathcal J_c\mathcal M_i\) at every cell of
outer radius \(r_i=ia\), \(L=n_{\rm ch}a\), with the same full
current as the clock law (on the chain, \(r\le L\) turns the
preceding paragraph's witness into
\(\mathcal J_c\le Lq_*^2S_\kappa/(m\delta)\) at the scanned
records), the discrete total variation of \(\ln N\) obeys
\(A_N^{\rm d}\le A_c=\pi L(L+a)\sqrt{2G\mathcal J_c}\) and
\(N(L)e^{-A_c}\le N_i\le N(L)e^{A_c}\), \(N(L)=1\) for
unit normalization at the last site and an enclosed exterior clock
contribution for normalization at infinity. The charge row supplies the
flow bound once the certificate verifies that the clock component of
\(DW_2[u]\), before the preparation value \(DW_2[u_\star]\) is
subtracted, is nonnegative for the implemented allocation and
averaging, so that exclusion and the fixed preparation moment cap
the full energy source from above; with
\(|wJ_{\ell r}|\le\sqrt{2G\mathcal J_c}\,\mathcal M/r\) this bounds
enclosed energy and normalized flow by comparison from the regular
centre, the clock bound then bounds \(v=Nw\), and the reference
derivatives and the extension of the constraint map must be
enclosed on the resulting regular domain rather than inferred from
bounded field values. On a connected enclosure of the solution, a
tube on which these conditions hold at every time, uniform bounds on
\(\mathcal J_c\) and \(\|F_u^{-1}\|\) give unique joint continuation
over the certified interval; failure to enclose any condition is
inconclusive, and even a certified loss of invertibility is not by
itself a dynamical singularity. Exclusion and charge conservation do
not by themselves keep the regular branch, and the leading laws
supply no lattice-scale geometric law once their gradient expansion
fails, so a Pauli-supported interior is a possible outcome, not an
assumption that closes the question. The stored-record diagnostics
above do not supply this whole-path certificate under
Eq.~\eqref{eq:dm-reference}; the deciding calculation is a validated
two-route enclosure of the joint evolution of the two-field history
with nominal \(r_s=80a\) under Eq.~\eqref{eq:dm-reference} for both
species over \(0\le t/a\le1112\), \(13.9\) of its nominal \(r_s\),
and a pass decides unique continuation of that finite history, not
of the reference collapse's longer history or of every admitted
history. \todo{validate the connected solution tube of the two-field history with nominal \(r_s=80a\) under Eq.~\eqref{eq:dm-reference} for both species, from its archived preparation at \(t=0\) through \(t=1112a\), the light field at full weight with its own reference and preparation calibration, enclosing preparation and propagated errors, full-reference residuals and derivatives of both species, admitted-branch and centre closure, the local source cap, the witness and inverse-Jacobian bounds, the clock quadrature and outer normalization with the exterior tail, and the flow and reference-domain bounds; require checkpoint agreement between the two independently validated routes}
```

## 6. Appendix B proof-sketch sentences moved from the paper (verbatim; the statements and hypotheses stay in Appendix B)

B.1 The sphere coefficient:
```latex
Polynomial approximants to \(s\), vanishing at \(0,1\), have
error controlled against \([u(1-u)]^{1/2}\) by the operator Jensen inequality~\cite{HansenPedersen1982}. Their closed-path expansion reduces the entropy to boundary deficits; path-diameter-weighted summability gives continuity in the normal and the ball limit by dominated convergence.
```

B.2 Convexity of the planar coefficient:
```latex
Gaussian strong subadditivity on finite regions
(Koteljanskii's inequality~\cite{Koteljanskii1950}) makes \(\tfrac12[S_A+S_{{\rm cube}\setminus A}-S_{\rm cube}]\) a symmetric submodular set function of \(A\). Its Lov\'asz extension~\cite{Lovasz1983}
is convex, even and positively homogeneous. On \(\boldsymbol\xi\cdot x\), divided by the cube's volume, it converges to \(\Phi\) by B.1's path expansion, the outer faces cancelling.
```

B.3 The massive estimate:
```latex
Taylor expansion of commutator kernels fixes \(P_1\) from \([h,P_0]\).
Gapped-path Combes-Thomas bounds~\cite{CombesThomas1973} control
\(R_2\) in an exponentially weighted Schur norm. Second-order trace
Taylor expansion~\cite{Bhatia1997} of cross-block entropy and the
area-factor Schatten bound~\cite{Rotfeld1969} finish the estimate.
```

B.4 The massless finite-parent estimate:
```latex
The proof
(Ref.~\cite{McKinneyCode2026})
splits at \(\delta=\epsilon^{2/5}\) into rank \(O(L^3\delta^3)\)
and norm \(O(\epsilon/\delta)\); the cross-block entropy estimate
gives exponent \(1/10\).
```

B.5 The gap at the trapping root:
```latex
A tensor cosine partition of support half-width \(R\) compares
localized operators with frozen symbols by Plancherel.
Freezing costs \(E_R\), partition commutators \(D_R\);
```

B.6 Local gap and local area:
```latex
A contour and comparison resolvent localize the remaining projector
difference and finite inner-window subspace near the clipping
support. Cross-block trace norm gives \(\mathcal E(d)\);
B.3 gives the comparison term.
```

Bibliography keys that leave the paper with these sentences: HansenPedersen1982, Koteljanskii1950,
Lovasz1983, CombesThomas1973, Bhatia1997, Rotfeld1969 (entries stay in ms_fermion_new.bib).

## 7. Singularity subsection replaced (lead-routed, inbox 43)

Sec. IV D "Singularities without a smooth geometry" of fermion_new/05_star.tex was replaced in
full by <internal>/SINGULARITY_SUBSECTION_agreed_v1.tex
(md5 0e27e004348939ddb7b3328c082a5fe3), agreed by reviewer A and reviewer B; the text stands until the M1
certificate lands. The replaced subsection text, including the one-sentence continuation
statement and its todo whose full specification is section 5 above, the one-sentence
polarization statement (section 3 above) and the pre-today paragraphs (definition, exclusion
bound, three-level continuation, computed verification, clock floor), is recoverable verbatim
from git: commit cced475d5e6, fermion_new/05_star.tex, from the subsection heading to the line
before "Diagnostics of the computed geometry". The new text carries no todo mark; it prints a
limitation sentence instead.

## 8. Batch 6 (owner, 2 Nov 2026, "the paper is not a log file"): passages moved verbatim from the body, cited as Ref. McKinneyCode2026

Each entry gives the body location (the paragraph it came from) and the text exactly as it stood before the batch-6 commit; the body keeps the key result and the numbers used later.

### 8.1 Sec. IV A dust benchmark: derivation line before Eq. star-peeling-control

```latex
Using Schwarzschild time \(t_s\) from
Eq.~\eqref{eq:dr-exterior-coordinates}, retarded time
\(u=t_s-r_{\rm tort}(R_b)\), \(dr_{\rm tort}/dR_b=\alpha_s^{-2}\),
and surface Killing lapse \(\alpha_s=\sqrt{1-r_s/R_b}\), it gives
```

### 8.2 Sec. IV B formation: averaging-scale dependence paragraph

```latex
The formation depends on the averaging scale: the first surface forms
at \(T/r_s=12.000\), 12.125 and 12.750 on the three members at 0.65,
0.66 and 0.76 of the record's \(2GE\); the \(\sigma/a=4\) and 8
members are \(0.625r_s\) apart and 15.5 percent apart in radius (18.0
percent at matched clocks), while the \(\sigma/a=2\) and 4 members
meet both criteria at formation only
(22.2 percent at matched clocks): the factor-4 control fails on both
pairs.
```

### 8.3 Sec. IV B formation: episode gaps sentence

```latex
Episode gaps are 1.0, 3.75, and \(4.9r_s\) at \(\tau=8\), 4, and 2,
the last beyond verification. Neither held-source nor widened
```

### 8.4 Sec. IV B formation: interior readings paragraph

```latex
Interior readings share this dependence: at \(\sigma/a=2\), 4, and 8, centroid minima are 10.9, 14.5, and
\(22.4a\) at 12.875, 13.25 and 14: centroid motion, not interior
rebound. Collar charge at 16 is 0.847, 0.919 and 0.975 of \(E_0\).
This held-step deficit, not a charge-loss law, halves with step
length and is 0.0005 on the conserving \(\sigma/a=4\) step
(verified to 12.5, read beyond).
Increasing channels from 10 to 14 and 20 at \(\sigma/a=4\),
\(\tau=8\) and 4 leaves formation's record unchanged and radius
within \(0.09a\); twenty-channel readings at 14 are unverified
and weighted sea pair energy unconverged. Conserving-step depth
and centroid, and all three under more channels, remain open; formation itself is verified at all three widths, and nothing below uses the depth or centroid values.
Post-formation step differences diminish, but width differences
remain order one, so no closing time, lifetime, late-time law, receiver-time redshift law, or second surface is established.
```

### 8.5 Sec. IV B formation: size and mass ladders paragraph

```latex
The size and mass ladders, the families of runs varying \(r_s/a\) or
\(\zeta\), give first-record dependences, not fitted laws.
At \(\zeta=0.2\), nominal sizes \(r_s/a=32,64,128\) form at
\(T/r_s=15.81,12.125,11.19\)--11.56, at 0.73, 0.66 and
0.63--0.68 of \(2GE\). The 128 values use \(\sigma/a=8,16\);
its well-filled width-4 solve fails. That preparation has six,
not five, columns in its fullest channel. Its isotropic width-4
control forms at 11.5 and 0.70, in one calculation. These isotropic
preparations fill eight states per momentum, not the sixteen of the
continuum fluid.
From \((\zeta,r_s/a)=(0.4,64)\) to \((0.1,96)\), formation moves
from 5.375 to 21.17 and from 0.37 to 0.73 of \(2GE\).
The \((0.05,326)\) member has no root through its verified
\(0.49r_s\). The \(E_0/E_{\max}=1.14\) members have no root
through 40 from \(4r_s\); the \((0.2,64)\) and \((0.1,64)\)
counts released from \(2r_s\) form.
```

### 8.6 Sec. IV B formation: width-agreement and size-256 paragraph

```latex
At size 64, isotropic preparation forms \(0.25r_s\) later;
matched-clock radii over 12.75--16 agree within 1.4 percent.
Wider averaging delays formation and raises its root fraction;
halving the step advances it by fractions of \(r_s\).
Size 32 and \(\zeta=0.4\) fail width agreement; the 128 pair
passes only at formation, width 16 forming after the hydrodynamic test fails.
At nominal 256, formation within the hydrodynamic description is reported only at width 8,
without a width span or law. A documented size-256 width-8 run
uses tapered flat-box preparation, fifteen 4096-site chains, and
\(\tau=16a\), not a well-filling preparation; its link to that reading is
unestablished because the preparations differ; the size-256 formation stays a reported, not verified, reading.
```

### 8.7 Sec. IV B view from far away: headline redshift values

```latex
Through \(T/r_s=40\), selected clock events give 10.3
(\(3\times10^4\)) for the reference collapse and 15.6 at width 2.
The reference collapse's event leaves \(0.30r_s\) at \(T/r_s=10.5356\),
before formation, and arrives at 40.
Elsewhere selected launches exceed clock readings (3.1--6.0):
3.6 at width 8, 6.5 at \(\tau=4a\), and 7.6 and 3.5 at size 128,
widths 8 and 16. The two completions give 3.7 and 4.1 across
initial and later clock releases and launches.
Deepest arrivals still grow at the endpoint.
```

### 8.8 Sec. IV B view from far away: clock-shift and sensitivity sentences

```latex
Clock shifts reach 2.3 for dense
fields, \(10^{-7}\) for tested clock integration, and 0.05 for
other propagation variants at headline events and fixed arrival.
Field sensitivity is 0.4 per \(10^{-3}\) at the verification
ceiling, not an error bound; two propagation tests shift launches
by up to 1.4.
```

### 8.9 Sec. IV B view from far away: lattice-ray crossover values

```latex
\(\ln(1/g_D)\) at most 6.6, 5.4, 4.5 and 3.7, near the crossover scales \(\ln[2/(ak)^2]=6.7,5.3,3.9,3.1\). This crossover is not a bound; some exceed it.
```

### 8.10 Sec. III E vacuum exterior: stationary axial perturbation paragraph

```latex
The leading constraints admit stationary axial perturbations of
Eq.~\eqref{eq:dr-exterior}. Rotations and inversion separate polar
and axial sectors. Without an imposed polar perturbation,
axisymmetric toroidal \(\delta v\) closes the axial sector at \(\delta N=0\);
full-response uniqueness is unproved. At unit clock,
\((1+b)\nabla^2\delta v=-16\pi G\,\delta J_\ell\), with
\(b\) from Eq.~\eqref{eq:dm-density}.
For a weak stationary source with conserved momentum density and
unit interior clock to leading order, the regular decaying solution
has \(\delta v=2G\,L\times x/[(1+b)r^3]\) far away,
\(L=\int y\times\delta J_\ell\,d^3y\) the source's angular momentum, spin included: general relativity's dipole
divided by \(1+b\), Kerr's to first order in \(L\) at
\(b=0\)~\cite{Doran2000}, and no regular linear response at \(b=-1\).
Coupled rotating sources and nonspherical evolution remain
unestablished, outside the spherical dynamics established here, and
no result below uses them.
```

### 8.11 Sec. V C light field on the collapse: opening paragraph and channel-tail details

```latex
On the computed collapse background the light field develops
outgoing energy flow and a vacuum-subtracted exterior-cut entropy
still rising at the final record, an early-time radiation
diagnostic, not a measured Page turnover or a once-counted escaping
entropy; the readings follow.
Weighted held-vacuum entropy diverges with channel cutoff.
Excess converges for temporally continuous, piecewise smooth
fields, but its computed tail is open: the sums end at \(k=18\) with the error floor stated below, an early-time diagnostic that enters no result beyond this subsection.
For per-copy excess \(x_k\), the weighted terms \(16k\,x_k\) share one level from \(k=12\),
also in probes to \(k=48\). At \(T/r_s=50\), \(R=16r_s\),
a natural cubic time spline changes \(k=12,16,24,48\) terms by
1.1--7.3 percent, cause unknown. Sums end at \(k=18\).
Their error floor is at most \(1.2\times10^{-4}\) of a sum above 3 nats.
The common-mode statistic \(18\max_{12\le k\le18}|16k\,x_k|\),
without lower-channel comparison, is 0.02--6.2 percent of formed
sums from \(T/r_s=80\). Smaller absolute sums are unformed,
not tail-converged.
```

### 8.12 Sec. V C: setup and readings paragraph

```latex
The test light field evolves from its prepared sea at \(T=0\) to
\(T/r_s=100\) on a separate doubled-domain collapse: 4096 matter
sites, 8192 light sites, vacuum appended beyond \(64r_s\).
Back-averaged fields are linearly interpolated in eight midpoint
substeps per record. Light gravity is omitted; the interior is
past the limit of its hydrodynamic description. Dense roots occur throughout 18.375--40 and at 56, 76
and 100. At \(2,16,24,30r_s\), sampled maxima are 8.26--9.50 nats
at 100. All five cuts \(2,16,24,30,48r_s\) then span
\(1.7\times10^{-4}\) to \(2.7\times10^{-4}\) of \(A(E)/(4G)\)
(Fig.~\ref{fig:page-v4}).
At \(30r_s\), excess falls from 6.76 nats at 60 to 4.96 at 70,
despite outward net \(k\le8\) energy over 56--70: not net return.
Piecewise-constant propagation adds 1.4 nats through \(k=8\)
at 60; field jumps can leave nonsummable reference mismatch.
Interior-spectrum purity is a single-calculation reading; arithmetic on
the same spectra reconstructs neither covariance nor held projector.
```

### 8.13 Sec. V C: excitation fraction sentence

```latex
and comparable over 80--100 except one negative checkpoint.
Excitation is 0.16--2.5 percent of \(E\) at checkpoints 15--100.
```

### 8.14 Sec. V C: entropy-per-energy diagnostic paragraph

```latex
Two post-processing calculations on shared exports give the diagnostic
\(\overline T_H\,\delta(\Delta S_{\rm cut})/\delta E_{\rm col}\),
with endpoint-mean \(\overline T_H\) and the \(2r_s\) cut
advanced by the ray map: 0.020--0.023 over 40--46,
0.05--0.08 over 46--100, and at most 0.38 on windows at least
\(6r_s\) long on the \(2r_s\) checkpoint grid over 40--100,
for \(k\le1,2,18\) at both cuts.
These lie below \(\mathcal R=1.64\) but classify neither emitted
entropy nor thermality. The \(k=1,2\) formation segment gives
5.6 nats over 14--40, 1.4--1.7 times the thermal budget;
its departing burst carries 120 times the benchmark energy
over collar clocks 36--62.
```

### 8.15 Sec. V C: histories-match-only-at-formation paragraph

```latex
The histories match only at formation. All four form at 12.125
within \(1.0a\) in radius.
At 40, reference and long completions have dust energy below
\(2.6\times10^{-3}\) lattice units beyond \(16r_s\).
Their \(k=1,2\) excess at \(2r_s\) spans 95 percent of the
reference collapse's eighteen-channel sum. Solver, charge, and matter-chain length also differ: 4096 sites for the light background, 2048 for
completions, so core treatment is not isolated. Completions end at
40 before dust readings at 48--56; collar-flux and cross-root-transfer comparisons are absent.
```

### 8.16 Sec. V C: end-at-100 paragraph

```latex
The end at 100 is only \(2.5\times10^{-4}t_*\), using the
formation charge. Over 80--100 the excess rises by
0.030--0.040 nats per \(r_s\), 0.23--0.31 of the benchmark
rate at 100.
```

### 8.17 Sec. V C: held-field paragraph

```latex
Holding fields at 15 or 30 for \(300r_s\) gives lowest-channel
flux plateaus 5.4--6.3 and \(-0.017\)--0.076 in units of
\(L_H(\kappa_T)\), before far-end returns.
Holding from 100 to 500 preserves \(E_{\rm inv}\) to seven
digits. At every \(10r_s\) checkpoint from 140 the \(k=1\)
per-chain flow is 0.9--1.5 \(L_H(T_H)\); cut excess grows by
1.18--1.28 nats, still about 0.0023 nats per \(r_s\) at 500.
```
Correction (lead inbox 90): the growth of 1.18--1.28 nats is the cut excess per chain over the whole hold, from 100 to 500, not from 140 (growth from 140 would be 1.08 to 1.11); per chain the excess is 0.5149/0.4727/0.4584/0.4482/0.2924 at 100 and 1.7142/1.6659/1.6430/1.6260/1.5764 at 500 at the cuts 2, 16, 24, 30, 48 r_s, from the interior table <internal>/v6s_B2hold_k1_interior.md (md5 45ab942d) and the readout B2_hold_drain_readout_v2.2.md (md5 e7c4310e), verified by the lead on the box. The printed sentence was corrected in the same commit.

### 8.18 Sec. V C: isolation budget values

```latex
verified on the first calculation and fails two budgets on the second,
the narrow-block subspace-angle sine (0.056--0.057 against 0.05)
and the broad-block drain fraction (0.147 against 0.05). The verdict
```

### 8.19 Sec. V C: no-evaporation paragraph

```latex
Single-calculation charge ladders retain no root to their final record
below size 32 at \(ma=0.4\), or 64 at 0.2; the next smaller
members touch. Thus \(t_*\) is compared only at
\(r_{s0}/a\ge32\). No evaporation, turnover, Page time, or approach
to \(A/(4G)\) is computed. Over 40--100, incoming collar flow
is at most 2.7 percent of outgoing at the 22 positive-outgoing
checkpoints; outgoing-mode energy flow is inward at the other
nine. Late field supply is not separated from excitation drain.
```

### 8.20 Sec. V C: light-gravity control paragraph

```latex
At \(r_s/a=32\), \(\sigma=4a\), light gravity in \(k=1,2\)
shifts the first dense outer root by \(+0.0042a\) from the source-off control.
Reported differences through 80 are relative chain-end charge
at most \(3.4\times10^{-5}\) and light energy beyond \(16r_s\), relative to the held sea projector, lower by 0.2--0.4 percent over 40--80. These finite-channel, single-calculation differences lack verified
errors. Matched comparison ends at 50; control continuation to 80
does not establish matching beyond 50.
Charge change is a conservation residual, not a feedback bound;
the deficit roughly matches energy already inside the cut at 40.
```

### 8.21 Sec. V E: long-hold fit values

```latex
Reported long-hold fits at the last slope maximum and minimum,
at \(\omega r_s=0.5,0.25\), give
\(2\pi T_{\rm fit}/\kappa_T=1.00\)--1.03 and 0.89--0.92,
within about 12 percent of the slope temperature and 12--30
percent from \(T_H\), without an error bound.
```

### 8.22 Sec. VI A: matching-constant values C_c

```latex
has retarded duration
\(\tfrac43r_s\ln[1/(2\chi q)]+C_cr_s+O(r_s(2\chi q)^{2/3})\) where it is
not empty, with \(C_c=-7.63\), \(-4.43\), and \(-1.38\) within 5, 20, and
50 percent of \(\kappa_H\). At
```

### 8.23 Sec. IV D singularity subsection before the batch-6 reorder (reordered to definition, analytic conditions, computed records, what remains; one sentence rewritten; nothing removed)

```latex
\subsection{Singularities without a smooth geometry}\label{sec:singularity}

Here a dynamical singularity is failure of unique joint continuation
of \(C,N,v\) at finite lattice time, with norm-preserving state
evolution, under unchanged laws, reference convention, and boundary
conditions, without new interior-boundary data. The test concerns
the specified coupled lattice equations, even where their motivating
derivative expansion is uncontrolled; failure of that expansion,
or of a gauge or clock choice alone, is not a singularity.
General relativity tests inextendible causal geodesics of finite
proper or affine length~\cite{ChoquetBruhat2009}; fixed-spacetime
quantum probes~\cite{HorowitzMarolf1995} and quantum-cosmology
bounces~\cite{AshtekarPawlowskiSingh2006} provide other tests,
not a universal definition.

A slice is admitted when it carries a spherical solution of the
rows with \(N>0\), a regular centre, and nonnegative, absolutely
continuous enclosed energy, \(\mathcal M(0)=0\). Negative enclosed
energy, or zero enclosed energy on a sphere carrying radial
momentum, obstructs such a solution. Exclusion bounds the
prepared-sea part of the sources, not the polarization moments.
At a regular centre,
\(K_t^2(0)=8\pi G\rho_\ell(0)/3\); the bare bound
\(\bar\rho\) of Sec.~\ref{sec:results-filling} therefore bounds
this angular curvature only after the polarization energy is
also bounded. Full spacetime curvature additionally needs the
tidal contribution, including stress and the evolution residual.
Bounded density establishes neither bounded full curvature nor
continuation.

Normal ordering does not ensure positive local energy, even for
excitations confined to the prepared bands. The flat symbol bound
\(|q(p)|\le\varepsilon(p)\) of Appendix~\ref{app:charge} supplies
no universal local bound on the full sources, which also contain
coherence and polarization response. Exclusion does not establish
persistence of admission. No invariant regular domain, and no
self-consistent counterexample to continuation over all admitted
data, is established here.

On a finite realization, unique joint continuation follows while
the solution stays in a compact regular positive-clock domain
with bounded required reference derivatives and a uniformly
invertible boundary-normalized constraint Jacobian. The reduced
covariance evolution is then locally Lipschitz and preserves its
spectrum. A prescribed Hermitian generator with locally
time-integrable norm already has a unitary propagator; this
weaker statement does not establish coupled-field continuation.

A sufficient clock condition needs neither local energy positivity
nor a band decomposition. On a regular inward branch with
\(\mathcal M>0\) for \(r>0\), put
\(\mathsf k=(2G\mathcal M/r^3)^{1/2}\) and normalize \(N(L)=1\).
The rows give \((\ln N)'=4\pi GJ_{\ell r}/\mathsf k\). If the full sources
satisfy, with a finite enclosed \(\Gamma_j\ge0\),
\begin{equation}
 rJ_{\ell r}(r,t)^2\le\Gamma_j\mathcal M(r,t),
 \qquad t\in[t_j,t_{j+1}],
 \label{eq:sg-current-mass}
\end{equation}
then, integrating \(|(\ln N)'|\le2\pi r\sqrt{2G\Gamma_j}\) from the
centre,
\begin{equation}
 |\ln N(r,t)|\le B_j,\qquad
 B_j=\pi L^2\sqrt{2G\Gamma_j}.
 \label{eq:sg-clock-bound}
\end{equation}
This bounds the clock above and away from zero without a density
floor; it is sufficient, not necessary, need not be uniform in
lattice size, and must be proved throughout an interval, not
inferred from endpoint samples.

The reference collapse has positive stored clocks through
\(T/r_s=40\), with sampled minimum \(0.068\).
At 573 records of three histories, the implemented constraint
Jacobian has verified inverse norm at most \(2.90\), with bounded
double-precision rounding and assembly errors.
Two reported independent, no-density-floor current--mass scans
cover 894 records in four histories. Their reported clock floor
is \(e^{-50}\), far below the sampled minima of about
0.07--0.11; these are record tests, not interval enclosures.
The reference collapse uses a held projector without light;
other histories have different light rules, one carrying the law's
block-form light counterterm. These tests verify no trajectory
under Eq.~\eqref{eq:dm-reference}, and only one history carries the
law's light counterterm. The stored-slice agreement of
Sec.~\ref{sec:results-lattice} remains a consistency check.

Two independent calculations reconstruct reference-collapse rays and angular
curvature from stored fields: all 36 tested ingoing rays cross
the regular centre at finite affine length, and the sampled
angular-curvature maximum is \(0.14/a^2\) near the centre at
\(T/r_s=13.25\). In two independent calculations, test packets without back-reaction
over \(T/r_s=12.125\)--40 preserve norm to \(2\times10^{-11}\).
No dynamical singularity is observed within the computed time;
these probes, beyond the hydrodynamic description's tested limit,
establish neither unique joint continuation between records,
affine completeness nor an infinite-time extension.

The deciding calculation is a validated coupled-trajectory
enclosure on every consecutive record interval.
Let \(F(C,u)=0\), \(u=(N,v)\), denote the boundary-normalized
constraints with the stated averaging and full reference response.
The enclosure must satisfy \(i\dot C=[\bar h(u),C]\) and
\(F_u\dot u=-F_C\dot C\), and connect to the next interval.
Its four sufficient conditions are:
\begin{enumerate}[label=(\roman*),leftmargin=*]
\item
\emph{Source and trajectory consistency.}
Enclose a solution under Eq.~\eqref{eq:dm-reference}, including
the propagated sea, all active species and copy weights, and
initial-data, assembly, quadrature, and evolution errors.
Constraint and evolution defects, boundary charge and covariance
spectra must be enclosed consistently.

\item
\emph{Admission and clock control.}
Throughout \([t_j,t_{j+1}]\), verify a regular inward branch and
a positive margin
\(\mu_3=\mathcal M(r,t)/r^3\ge\mu_{*,j}>0\) for \(0<r\le L\).
Its regular-centre limit is \(4\pi\rho_\ell(0,t)/3\), implying
\(\rho_\ell(0,t)\ge3\mu_{*,j}/(4\pi)\) and
\(\mathsf k^2\ge2G\mu_{*,j}\).
Verify Eq.~\eqref{eq:sg-current-mass} on the same enclosure,
including every tail and low-density cell. At the centre use
\(j_1=J_{\ell r}/r\) and \(j_1^2\le\Gamma_j\mu_3\), with enclosed
regular limits. Use the full constraint mass, not enclosed rest
mass. Discrete rows require the corresponding clock estimate
or a bounded radial reconstruction defect.

\item
\emph{Field solvability.}
In fixed, printed coordinates and norms, verify a uniform
inverse bound for the full \(F_u\), including reference response.
For example, an approximate inverse \(Z_j\) with
\(\sup\|I-Z_jF_u\|\le q_j<1\) throughout the enclosure gives
\(\|F_u^{-1}\|\le\|Z_j\|/(1-q_j)\).
A clock bound alone does not establish this condition, nor do
the inverses verified at stored records.

\item
\emph{No other exit.}
Bound the fields, required spatial regularity, and reference
derivatives, and close the trajectory enclosure on itself
throughout the interval, not only at its endpoints.
Successive enclosures must continue the same branch.
\end{enumerate}
The positive margin in (ii) selects a sufficient nondegenerate
domain; regular degenerate centres and mass-zero vacuum components
need separate matching and uniqueness arguments. Failure of a
sufficient inequality means not verified, not singular.

Stored held-projector histories may seed a verification only with
enclosed defects under the printed law; otherwise re-evolution is
required. Interpolating stored fields and propagating test matter
is not this calculation. No complete between-record verification
under the printed law, or general bound on polarization moments
over admitted cubic slices, is established.
The leading laws supply no lattice-scale geometric law once
their gradient expansion fails, so a Pauli-supported interior
remains a possible outcome, not an assumption that closes the
continuation question.


```

### 8.24 Sec. IV C interior: compression and density readings paragraph

```latex
The compression is read on the dust's half-count radius \(R_{50}\): on
the \(\sigma/a=8\) member it falls to \(19.1a\) at 14, the inner
trapping root's minimum, 1.8 times the benchmark radius \(R_\pi\)
(\(10.5a\) for the isotropic ball) and above the filling-volume
comparison radius (\(2.67a\) on the chains), and rises to \(34.8a\) at
16; on the
\(\sigma/a=4\) history, which stores no slice at 13 to 14, its
compression minimum unsampled, it is 31.4, 28.9 and \(31.4a\) at 12, 15
and 16. The largest central density on any stored slice is
\(0.063a^{-3}\) at 28, 3.2 percent of the exclusion density \(2a^{-3}\);
at 15 the inner root at \(7.05a\) encloses 33.6 particles at 1.1 percent
of it, with \(J/\rho=0.25\) there, and the dust's bare energy per
particle \(a\rho/n\), written \(\epsilon_c\) below, is 0.674 and 0.548
inside \(R_{50}\) at 15 and 16, 2.7 to 3.4 times \(ma\). The inner
shell's dust (\(R\le128a\)) at \(|ap|>\pi/4\) is 15.1 and 11.0 percent
at 15 and 16 on the reference collapse and 0.3 and 4.3 percent at 14 and
16 on the \(\sigma/a=8\) member, lattice momenta the rows' drift can
supply before saturation; the \(\pi/4\) split, a plane-wave criterion,
reads neither bin as outgoing content. The central density peaks at the
compression minima on both members, and the dust fills to \(\pi/4\) on
no stored slice.
```

### 8.25 Sec. IV C interior: same-source clock slopes

```latex
On the stored slices at 15
and 16 the clock is no power of the enclosed count, the least-squares
slope of \(\ln N\) on \(\ln\mathcal N_<\) over \(R\le R_{50}\) being
0.228 and 0.215 against the same-source 1.18 and 0.87, and at 15 on the
kernel-averaged sources \((\lambda-1)/2\) runs from 0.07 at the inner
root to 3.0 at \(R_{50}\) with no constant-\(\lambda\) annulus, so no
exponent is calibrated; Eq.~\eqref{eq:sc-clock-exact} integrated on the
dense fields matches the clock's fall from \(R_{50}\) to the outer root
within 1 percent at 15 and 16 and 3.1 percent on the 13 rooted slices,
in two independent calculations.
```

### 8.26 Sec. IV C interior: burst-bin values

```latex
In the Killing split above
the \((m,b_r]\) bin, the burst, gains a net 58.4 particles (9.1 percent)
between 16 and 20 against each slice's own fields, its mean 1.12 to
\(1.16m\) from 20 to 40, at the bottom of \((m,b_r)\) (\(b_r/m=5.1\)).
```

### 8.27 Sec. IV C interior: pair-energy and mass-fluctuation values

```latex
With the native density-current pair kernel, the whole slice's signed exchange weakens the pair energy by 0.14 to 0.16 percent, 0.90 to 1.01 times \(1/N_B\), and the
root-mean-square (rms) fluctuation of the linearized mass observable is 0.9 to 14 percent
of \(\mathcal M\) over the inner root, \(R_{50}\), and \(R_t\), 0.23 to
0.84 times \(\mathcal N_<^{-1/2}\), neither count comparison a bound,
the kernels being indefinite.
```

### 8.28 Sec. IV C interior: Fock amplitude values

```latex
In two independent calculations, with \(\widehat E_2\) Wick ordered instead about the Hartree
state the model evolves, the Fock term's single particle-hole
amplitude from that state has norm \(7.1\times10^{-3}\) and
\(1.5\times10^{-3}\) times that of \(h[\bar C]\) at \(T/r_s=11.0\)
and 15.0 on the ten channels, the instantaneous relative Fock correction
\(|\delta\dot M_j/\dot M_j|\), at fixed radius per unit clock,
is below \(10^{-2}\) on the first calculation at all represented radii
and at most \(3.9\times10^{-3}\) on the second calculation at the inner
root, \(R_{50}\), and \(R_t\) at \(T/r_s=15.0\), and the two-dust-hole double amplitude changes
by under 1 percent from ten to 20 channels. On the 14- and
20-channel histories at 11.0 this Fock-to-Hartree single ratio is
\(9.0\times10^{-3}\) and \(1.30\times10^{-2}\); 97 to 98 percent of
its growth, and all of the contribution above ten channels, come from the exchange field of the propagated sea's mismatch with the held projector, which grows faster than the copy weights
\(16k\): no channel limit is established on the chains
```

### 8.29 Sec. IV E diagnostics: first-law term values

```latex
On the \(\sigma=2a\) member's 224 consecutive
record pairs with a simple
dense root, to \(T/r_s=40\), on the dense fields in finite-collar form,
with the stored history's sources, the terms of
Eq.~\eqref{eq:page-first-law-rows} sum to \(\delta E=-18.36\),
\((\kappa_R/8\pi G)\delta A=-6.51\) and \(W_{\rm ext}=-11.93\) lattice
units, residual 0.08, a diagnostic with no error bound: the work term
is above its numerical floor in 221 of the pairs, these records lacking
a vacuum exterior beyond the root.
```

### 8.30 Sec. IV E diagnostics: stress-residual medians

```latex
kernel-averaged stress, by a median 0.17, 0.04 and 0.06 of the row's
largest term at \(\sigma/a=2\), 4, and 8, not proportional to the
averaging fractions \((\sigma/R)^2\) of 0.004, 0.011 and 0.033, and by 0.18
to 0.74 at inner roots above \(2\sigma\), outside the hydrodynamic description.
```

### 8.31 Sec. VI D: pocket-ceiling values

```latex
Two independent ceilings on the pocket sector's entropy, the Gibbs
entropy of unrefilled pockets at fixed energy \(E_0\) on the
constant-charge vacuum profile taken as proxy for the object's
fields, are, in units of \(fA_R/(4G)\) with \(f\) at \(ma=0.2\), 0.30 and 0.27
at \(R/a=32,64\) from light-field finite-lattice spectra and
0.28--0.31 at \(R/a=64,128,256\) from the direction-averaged frozen
symbol over the full zone, heavy field included. On the stored
slices past the limit of the hydrodynamic description the same ceilings are 0.09--0.10 on the first calculation
and 0.08--0.10 on the second. These pocket ceilings scale with area
and lie below the assumed store capacity; without the sector
restriction both grow with volume (exponent 2.99 over fourfold size
on the second calculation) and exceed the store at every computed size.
Neither identifies the accessible sector
(Ref.~\cite{McKinneyCode2026}).
```

### 8.32 Sec. VI D: two-field diagnostic setup and enclosure values

```latex
The two-field diagnostic of the heavy-light reference terms, at a
single stored record, has nominal
\(r_{s,\mathrm{nom}}/a=80\), \(m_Ha=0.2\), \(m_La=0\),
\(G/a^2=1/4\), \(\eta_E(0)\simeq0.919285\) at release,
averaging width \(\sigma=4a\), heavy columns \(6,5,4,3,2\)
in \(k=1,\ldots,5\), weights \(16k\), one light channel
\(k=1\), full-chain weight 16, and
Eq.~\eqref{eq:dm-reference}'s polarization term subtracted for
the heavy field only.
At \(t/a=1104\), outer dense root \(R/a\simeq69.47\) (both
calculations within \(0.03a\); heavy matter still lies outside it),
two independent implementations on shared stored inputs agree
to about a part in a thousand on the reference term's
contribution at fixed radius to the rate of change of the mass
inside it: it has the rate's sign and is about 8.6 percent of
it (7.9--9.5 with record-spacing errors), 0.7 percent of which
is the light field's term, absent from the history. On this
history's stored fields at \(t/a=1104\), two independent
implementations enclose the preparation-calibrated
analytic-\(f'\) light bulk polarization functional rigorously at
\(-5.7101733\times10^{-4}\) per full chain, or
\(-9.13628\times10^{-3}\) at full-chain weight 16, with interval
radii below \(10^{-28}\); endpoint completion and the massless
approximation are outside this enclosure.
Following the stored root, both calculations give
\(-(3.27\pm0.10)\times10^{-3}\) in lattice units.
Both find that the light reference response, also absent,
would lower the heavy channels' unweighted sum of squared
heavy-light Hessian elements, heavy moments unsubtracted, by
\((2.4043\pm0.0004)\times10^{-4}\) of itself.
```

### 8.33 Sec. V A: direct on-site converter definition (matrices)

```latex
A direct heavy-light coupling, absent from every history, is defined
for later reference. The direct on-site converter
\(H=H_0+UV\), \(U\in\mathbb R\), uses the basis
\(\alpha_i=\tau_x\otimes\sigma_i\), \(\beta_{\rm D}=\tau_z\otimes I_2\):
\begin{equation}
\begin{gathered}
V=\sum_{x,\mu}\eta_{\mu\mu}
 \Phi_{L,x}^{\mu\dagger}\Phi_{H,x}^{\mu}+\mathrm{h.c.},\\
\Phi_{A,x}^\mu=c_{A,x}^{T}M^\mu c_{A,x},\quad
\eta=\operatorname{diag}(1,-1,-1,-1),\\
M^0=\tau_z\otimes\sigma_y,\quad M^1=\tau_y\otimes\sigma_z,\\
M^2=i\tau_y\otimes I_2,\quad M^3=-\tau_y\otimes\sigma_x .
\end{gathered}
\end{equation}
Here \(A=L,H\) labels the light and heavy fields. The same clock and
flow couple to its site energy and assigned
energy-dipole current with equally divided bond corrections,
```

### 8.34 Sec. II C: requirements of the directly coupled extension

```latex
An interacting generator \(H_U\) would
need a reference of its own from the same Hamiltonian,
\(W_U=\langle H_U\rangle_{\rm ref}\) and
\(A_{\rm ref}=A-\langle A\rangle_{\rm ref}I\) on a differentiable
stationary branch of fixed-occupation exact eigenstates or
self-consistent Hartree states, where stationarity gives
\(\delta W_U=\langle\delta H_U\rangle_{\rm ref}\); it would add
quartic site interaction energy and bond-current corrections
referenced against that family, and would need its energy,
corrected flow-coupled momentum, stress, and lattice momentum force
from that family, their closure through second order in the
interaction with all quadratic terms and reference response, and an
interacting form of the boundary-generator identity (Sec.~\ref{sec:model-operators}).
```

### 8.35 Sec. IV A method: dust release momentum

```latex
no coordinate-rest
condition is imposed, and the dust's total momentum at release is
55.78 in lattice units.
```

## 9. Batch 7 (owner, 2 Nov 2026): Sec. VI D record-level specifics and the exchange-current response, moved verbatim

### 9.1 Sec. VI D: partner-accounting record specifics

```latex
For the represented channels (channels \(k=6\)--10 omitted) with
the trapping root held at \(2000a\), both frozen partner-accounting calculations give positive
\(S(B|X)\) for a later escaped block \(B\) and the earlier escaped
output \(X\) on the common checkpoint,
two independent implementations agreeing within their enclosures;
neither supplies a history-verified entropy-reducing addition of
later escaped radiation to earlier escaped output.
```

### 9.2 Sec. VI D: firewall paragraph with the exchange-current response definition (Eq. page-effective-weight)

```latex
The firewall question is which side holds a late partner~\cite{AMPS2013}.
For emitted mode \(d\) and either earlier radiation or retained
region \(X\), set \(K_{dr}=\langle c_d^\dagger c_r\rangle\).
In the product \(\varpi\) of their marginals, define the
Bogoliubov--Kubo--Mori response of the exchange current:
\begin{equation}
\begin{aligned}
\widehat\jmath_{dX}&=i\sum_{r\in X}\bigl(K_{dr}^{*}c_d^\dagger c_r
-K_{dr}c_r^\dagger c_d\bigr),\\
\mathcal W_X(d)&=\int_0^1\Tr\bigl[\varpi^{\varsigma}\,
\widehat\jmath_{dX}\,\varpi^{1-\varsigma}\,\widehat\jmath_{dX}\bigr]
\,d\varsigma .
\end{aligned}
\label{eq:page-effective-weight}
\end{equation}

```

### 9.3 Sec. VI D: properties of the exchange-current response

```latex
The current's mean is zero. For positive \(\varpi\), \(\mathcal W_X(d)\) is \(\partial_\lambda^2\ln\Tr e^{\ln\varpi+\lambda\widehat\jmath_{dX}}|_0\)
at fixed current; the integral extends to singular cases.
The comparison product is neither source reference nor conductance;
the pure pair's own covariance vanishes.


```

### 9.4 Sec. VI D: pair-model response weights

```latex
For the caption's independent fixed-number pairs, a new partner
is retained and a released partner lies in earlier radiation.
At emitted occupation \(n\), the weight
\(\mathcal W(n)=n(1-n)(2n-1)/\ln[n/(1-n)]\), \(\mathcal W(1/2)=\tfrac18\),
lies on the partner's side and equals ordinary variance at
half occupation. Even as \(aT_H\to0\),
\(n=\Gamma/(e^{\omega/T_H}+1)\) does not approach one half.
Thus \(\mathcal W/s=1/(8\ln2)\) at half occupation but
\(\mathcal W/s\sim1/\ln^2n\) for small \(n\):
response and entropy weight cold spectra differently.
Dephasing removes this response but preserves number
anticorrelation, so it is no entanglement or monogamy bound.
```

### 9.5 Sec. VI D: terminal expansion stage values

```latex
The benchmark reaches a terminal expansion stage when port
depth \(111\delta=10^{-4}R\), with
\(\delta=(R\ell_H^2)^{1/3}\) and stretched light spacing \(\ell_H\).
Exactly, \(\delta/R=10^{-4}/111\) and
\(\ell_H/R=(10^{-4}/111)^{3/2}\).
Two independent calculations give the terminal radius \(R_T/R_0\simeq0.012285\) nominally,
0.0126 at \(a=2\ell_P\), and 0.0118--0.0128 across the
verified spacing interval, scaling as \(\chi_0^{2/9}\)
at fixed \(G/a^2,\mathcal L\).
```

### 9.6 Sec. VI D: two-field record, condensed form before batch 7

```latex
The two-field diagnostic of the heavy-light reference terms is a
single stored record (nominal \(r_{s,\mathrm{nom}}/a=80\), \(m_Ha=0.2\),
\(m_La=0\), one light channel; setup in Ref.~\cite{McKinneyCode2026}).
At \(t/a=1104\), with the outer dense root at \(R/a\simeq69.47\) and
heavy matter still outside it, two independent implementations on
shared stored inputs agree to about a part in a thousand that the
reference term's contribution to the rate of change of the mass
inside the root has the rate's sign and is about 8.6 percent of it
(7.9--9.5 with record-spacing errors), 0.7 percent of which is the
light field's term, absent from the history; both enclose the light
bulk polarization functional rigorously on the stored fields, and
both find that the light reference response, also absent, would
lower the heavy channels' unweighted sum of squared heavy-light
Hessian elements by \(2.4\times10^{-4}\) of itself (enclosures in
Ref.~\cite{McKinneyCode2026}).
All are one stored record and none is a re-evolved balance; the
complete moving-root and history balances stay open.
```

### 9.7 Sec. IV C: pair-kernel exchange, mass-observable fluctuation and the grid rows of the held-source masses (before the inbox-51 q_inf insertion)

```latex
With the native density-current pair kernel, the whole slice's signed exchange weakens the pair energy by about \(1/N_B\), and the
root-mean-square fluctuation of the linearized mass observable over
the inner root, \(R_{50}\), and \(R_t\) is below \(\mathcal N_<^{-1/2}\),
neither count comparison a bound, the kernels being indefinite (slice
values in Ref.~\cite{McKinneyCode2026}). The held-source operators' full second-order masses
\(\widehat M_j\) at \(R_{50}\) and the roots are read at
the nearest \(0.1a\) grid rows as diagnostics (Sec.~\ref{sec:model-operators}). \fnd{For
```

### 9.8 Sec. IV C: Fock amplitude sentences before the q_inf insertion (before the inbox-51 q_inf insertion)

```latex
and the correlation beyond Hartree-Fock is uncontrolled. In two independent calculations, with \(\widehat E_2\) Wick ordered instead about the Hartree
state the model evolves, the Fock term's single particle-hole
amplitude from that state is below \(10^{-2}\) of that of \(h[\bar C]\)
on the ten channels at \(T/r_s=11.0\) and 15.0, as is the
instantaneous relative Fock correction \(|\delta\dot M_j/\dot M_j|\)
at the inner root, \(R_{50}\), and \(R_t\), and the two-dust-hole
double amplitude changes by under 1 percent from ten to 20 channels.
The single ratio grows with the channel count, almost all of the
growth coming from the exchange field of the propagated sea's
mismatch with the held projector, which grows faster than the copy
weights \(16k\): no channel limit is established on the chains
(amplitude values in Ref.~\cite{McKinneyCode2026})
```

### 9.9 Sec. IV C: same-source closure formula (lambda and the explicit ratio form) (before the inbox-51 commit)

```latex
The same-source assignment \(\mathcal M=m\mathcal N_<\), \(\mathcal N_<(r)\) the enclosed count, at a constant bare energy per particle \(\epsilon_c/a\), a flat-slice benchmark assumption, closes the law with \(\lambda=\epsilon_c/\zeta\) and \(N(r)/N(R_\pi)=[\mathcal N_<(r)/\mathcal N_<(R_\pi)]^{\nu}\), \(\nu=(\epsilon_c-\zeta)/(2\zeta)\); at a regular centre it fails by the ratio of bare energy per particle to \(m\).
```

## 10. Inbox 52 (lead-routed earned edits, reviewer A nefcheck as amended by reviewer B nealt1): passages replaced or moved verbatim

### 10.1 Sec. V E: long-hold fit statement before inbox 52 (its 12 percent and 12-30 percent values)

```latex
Reported long-hold fits at the last slope maximum and minimum give \(2\pi T_{\rm fit}/\kappa_T\) within about 12 percent of the slope temperature and 12--30 percent from \(T_H\), without an error bound (fit values in Ref.~\cite{McKinneyCode2026}). These reported fits remain unverified, their normalization against \(T_H\) unreconciled across readings, and are no evidence for the evolving object's emission. The \(\tau=4\) history sustains neither packet at its last slope maximum. No conversion spectrum is read at this size.
```

### 10.2 Sec. V E: evolved-cut paragraph before inbox 52 (r_ref open; B-transfer premises)

```latex
On the evolved trapping cut, the outer dense root or \(2GE\) without one, the heavy field's weighted \(k\le10\) exterior entropy exceeds the held vacuum's by 109--311 nats over 12--40 at width 4 (two independent calculations at six of ten clocks), and 116--305 at width 8 (one calculation). It lies within the excitation term in one calculation, but \(r_{\rm ref}\) is open: no small evolved-area error follows at this size. Appendix~\ref{app:local}'s transfer estimate, Eq.~\eqref{eq:B-transfer}, has gap and distance premises not verified on this cut, past the hydrodynamic description, so it establishes neither transfer nor the flat coefficient here.
```

### 10.3 Sec. V D: arrival-time clause (half arrive 39.45 r_s after formation)

```latex
Current is outward at all five clocks 48--56; half arrive \(39.45r_s\) after formation, with count still rising.
```

### 10.4 Sec. V D: sweep-count detail of the chain-versus-cubic column comparison

```latex
The one column read in both descriptions agrees at sweeps one to three only because both fractions lie below tolerance, then differs by up to 0.34 at nine to ten; the lift of the chain description to the cubic operator is not validated.
```

### 10.5 Sec. V C: substep interpolation of the back-averaged fields (eight midpoint substeps per record)

```latex
Back-averaged fields are linearly interpolated in eight midpoint substeps per record. Light gravity is omitted;
```

### 10.6 Sec. V D: run length T = 18 x 32a of the cubic transfer test

```latex
At size 32, \(T=18\times32a\), the four-component cubic operator on held point-sampled fields sends interior-source weights 0.68--0.90 beyond 1.5 sweeps after two to four sweeps, a sweep being \(29a\) of lattice time.
```

### 10.7 Sec. V C: the held P_- fixing eligible levels and inventory

```latex
The drain inventory assigns collar flux to these pole blocks, with the held \(P_-\) fixing both the eligible levels and the inventory; attributing
```

### 10.8 Sec. VI A: window paragraph before inbox 52 (matching constant "remains open")

```latex
The window \(T_d\) ends at \(\alpha_s=\alpha_d\) in Eq.~\eqref{eq:sc-approach}. The vacuum exterior \eqref{eq:dr-exterior}, incoming sea, and sine dispersion fix its leading logarithm, with \(2\chi q=2a\omega\). Its \(O(r_s)\) matching constant from the formation transient and channel tails remains open.
```

### 10.9 Sec. VI A: dust-benchmark constant paragraph before inbox 52

```latex
The constant is explicit on the dust benchmark of Eq.~\eqref{eq:star-dust-control}: the segment of the exact ray map on which the peeling rate \(\kappa_u\) lies within a given fraction of \(\kappa_H=1/(4GE)\), closed by the sine dispersion where \((R_b-r_s)/r_s\) reaches \((2\chi q)^{2/3}\), has retarded duration \(\tfrac43r_s\ln[1/(2\chi q)]+C_cr_s+O(r_s(2\chi q)^{2/3})\) where it is not empty, with explicit \(C_c\) at 5, 20, and 50 percent of \(\kappa_H\) (values in Ref.~\cite{McKinneyCode2026}). At \(r_s/a=64\) and \(q=1\) the segment is empty within 5 and 20 percent and carries about one e-fold of peeling within 50 percent, and no packet resolving its own frequency fits inside it at any of the three criteria.
```

### 10.10 Sec. VI A: entropy-mark paragraph before inbox 52

```latex
Over the window the light field carries, at the rates of Eq.~\eqref{eq:sc-luminosity}, \(\tfrac83g\mathcal L\chi^2\ln[1/(2\chi q)]\) of \(E\) at leading order; its outgoing modes carry a marginal entropy of \(0.144\ln[1/(2\chi q)]\) nats on the retarded clock, \(O(\chi^2\ln(1/\chi))\) of \(A/(4G)\), 0.50 nats at \(r_s/a=64\), \(q=1\) and zero matching constant: a mark on the entropy axis of the Page curve; it is not the computed history's entropy.
```

### 10.11 Sec. VI C: burst-bound derivation sentences before inbox 52

```latex
Its band-restricted one-particle energy-current kernel has two nonzero eigenvalues, \((m_1\pm\sqrt{m_0m_2})/2\); \(0\le C\le I\) bounds every fermionic state's current expectation by the positive one. The multichannel kernel is a direct sum, so interchannel coherence leaves the bound unchanged.
```

### 10.12 Sec. VI D: single two-field record paragraph before inbox 52 (part-in-a-thousand agreement; 2.4e-4 Hessian-norm effect of the absent light reference response)

```latex
A single stored two-field record (nominal \(r_{s,\mathrm{nom}}/a=80\), one light channel) finds, in two independent implementations agreeing to a part in a thousand, that the heavy-light reference term is about 8.6 percent of the rate of change of the mass inside the root, with the light field's term 0.7 percent of it and absent from the history, and that the absent light reference response would lower the heavy-light Hessian norm by \(2.4\times10^{-4}\) of itself (setup and enclosures in Ref.~\cite{McKinneyCode2026}). It is one record, not a re-evolved balance; the complete moving-root and history balances stay open.
```

### 10.13 Sec. VI C: solar-mass versus size-64 evaluation paragraph before inbox 52 (size-64 column: alpha_d 0.315, delta T/T_H 2.82e-4, m/T_H 161, t_*/r_s 5.21e5)

```latex
Evaluated, not measured, at \(q=1\), \(m=0\) unless stated, for \(r_s/a=64\) and one solar mass: the validity scale \(\alpha_d\) is 0.315 and \(2.80\times10^{-13}\); the formal axial correction \(\delta T/T_H\) is \(2.82\times10^{-4}\) and \(1.39\times10^{-76}\); \(m/T_H\) at \(\zeta=0.2\) is 161 and \(2.30\times10^{38}\); and if Eq.~\eqref{eq:sc-luminosity} holds throughout evaporation, \(t_*/r_s\) is \(5.21\times10^{5}\) and \(1.06\times10^{78}\), \(t_*=3.3\times10^{65}\) years at one solar mass; at fixed \(g,q,\zeta\) these scale as \(\chi^{1/3}\), \(\chi^2\), \(\chi^{-1}\), and \(\chi^{-2}\), changes of 12 to 73 orders of magnitude.
```

### 10.14 Sec. VI B: post-crossing paragraph before inbox 52 item 3 (fountain "leaves that vanishing unestablished")

```latex
After crossing, a vacuum exterior with \(E\) conserved (Appendix~\ref{app:charge}) keeps the root at \(r_s=2GE\) and the work in Eq.~\eqref{eq:page-first-law-rows} zero; the computed fountain, matter outside the root, leaves that vanishing unestablished. Inferring recession also requires the light field's signed null-source contribution, a \(\chi\)-uniform positive margin \(1-2G\mathcal M'>0\), and controlled finite-zone and moving-reference defects. Its energy budget alone gives neither autonomous recession nor a shed energy fraction.
```

### 10.15 Sec. VI C: heavy-particle mode budget sentence (duplicate of the Sec. VI D statement "Original heavy particles alone supply only O(N_B ln(1/chi_0)) nats")

```latex
establishing neither saturation nor any nonzero rate. Conversions that merely exhaust the original heavy particles provide only \(O(N_B)\) events; with no additional sea excitations assumed, a mode budget gives \(S_{\rm esc}\lesssim N_B\ln(1/\chi_0)\), \(\chi_0=a/r_{s0}\).
```

### 10.16 Sec. VI C: burst-bound kernel derivation (band moments m_j, eigenvalues (m_1 +- sqrt(m_0 m_2))/2, direct-sum kernel) before inbox 52 item 3

```latex
Exclusion bounds bounded-band bursts even with frequency coherence. For one freely propagating outgoing channel, normal ordered against the empty band \(0\le\omega_1<\omega_2\), define \(m_j=\int_{\omega_1}^{\omega_2}\omega^j\,d\omega/(2\pi)\). Its band-restricted energy-current kernel has the two nonzero eigenvalues \((m_1\pm\sqrt{m_0m_2})/2\), so \(0\le C\le I\) bounds every state's current by the positive one, interchannel coherence included (direct-sum kernel). In infinity-arrival interval \(\Delta t\), with \(\omega_i=x_iT_H\), \(n_{\rm ch}\) channels carry at most
```

### 10.17 Sec. VI D: response-weight sentence before inbox 52 item 3 (variance at half occupation; emitted occupation n = Gamma/(e^{omega/T_H}+1))

```latex
the response weight lies on the partner's side, equals the ordinary variance at half occupation, and weights cold spectra differently from entropy, since the emitted occupation \(n=\Gamma/(e^{\omega/T_H}+1)\) does not approach one half even as \(aT_H\to0\) (weights in Ref.~\cite{McKinneyCode2026}).
```

### 10.18 Sec. V B: size and mass ladder member readings (formation clocks and fractions of 2GE) before inbox 52 item 3

```latex
form no root within the computed time: at \(\zeta=0.2\) the nominal sizes 32, 64, and 128 form at \(T/r_s=15.8\), 12.1, and 11.2 to 11.6, at 0.73, 0.66, and 0.63 to 0.68 of \(2GE\); from \((\zeta,r_s/a)=(0.4,64)\) to \((0.1,96)\) formation moves from 5.4 to 21.2 and from 0.37 to 0.73 of \(2GE\); the \((0.05,326)\) member and the \(E_0/E_{\max}=1.14\) members form no root within their computed time (member details in Ref.~\cite{McKinneyCode2026}).
```

### 10.19 Sec. V B: clock-shift sentence before inbox 52 item 3 (field sensitivity 0.4 per 1e-3 at the verification ceiling)

```latex
Late redshift changes at order one with width, step, completion and propagation background. Clock shifts reach 2.3 for dense fields at headline events and fixed arrival, and field sensitivity is 0.4 per \(10^{-3}\) at the verification ceiling; the sensitivity is not an error bound (shift values in Ref.~\cite{McKinneyCode2026}).
```

### 10.20 Sec. IV C: mass-function fluctuation paragraph (Slater N_r^{-1/2} bound; the Hartree-Fock paragraph keeps the statement with the repository pointer), removed as verbosity-audit row 57, payment for inbox 53

```latex
The mass-function fluctuation is read with the variance formula of Sec.~\ref{sec:model-operators}, \(A=\nabla_C\mathcal M(r)\) at the Hartree covariance; in the positive rest-mass Slater approximation the mass inside \(r\) is \(m\) times the number inside, whose Slater variance is at most its expectation \(N_r>0\), so the relative fluctuation is at most \(N_r^{-1/2}\).
```

### 10.21 Sec. II C: directly coupled extension, requirement sentence (verbosity-audit row 12)

```latex
The directly coupled extension is outside this paper: no result below uses it, and its converter (Sec.~\ref{sec:page}) is absent from every computed history. An interacting generator \(H_U\) would need a reference of its own from the same Hamiltonian on a differentiable stationary branch, and its energy, momentum, stress, second-order closure, and boundary-generator identity would all have to come from that one family (the requirements are listed in Ref.~\cite{McKinneyCode2026}).
```

### 10.22 Sec. III A: composite free-fall factor, explanatory clause (verbosity-audit row 19)

```latex
it holds at quadratic centre-momentum order only up to the free-fall factor \(\Upsilon^2\), \(\Upsilon=\langle\cos(2ak_i)\rangle\) over internal momentum: the factor is the internal-momentum average of the flow's advection weight \(\cos(2ap_i)\) in Eq.~\eqref{eq:dr-comoving} (centre Hamiltonian in Ref.~\cite{McKinneyCode2026}). Section~\ref{sec:star}'s elementary dust orbitals have no composite factor.
```

### 10.23 Sec. III C: cross-block bound, collar-truncation mechanism (verbosity-audit row 26)

```latex
gives entropy difference \(O(L^{3/2}\sqrt{\log L})\) after truncating the massive flat sea's cross block to a logarithmic collar, the Lipschitz bound controlling the collar and the remaining singular-value weight contributing \(O(L\log L)\) (Ref.~\cite{McKinneyCode2026}).
```

### 10.24 Sec. III C: reference-split comparison, 640-particle rank-term values (verbosity-audit row 29)

```latex
their entropy difference costs at most \(N_B\log2\), leaving the comparison between \(\mathsf Q\) and \(P\): 640 particles give 443.6 nats, or 0.0101 per lattice area on a sphere of radius \(59.07a\), the reference collapse's count and \(2GE_0\) (Sec.~\ref{sec:star}), a rank term not added again to the direct budget, which already includes these particles.
```

### 10.25 Sec. III D: exclusion site-count sentence (verbosity-audit row 30)

```latex
Two positive-band states per site require at least \(N_B/2\) sites for \(N_B\) particles, equivalent to a ball of radius \((3N_B/8\pi)^{1/3}a=0.49N_B^{1/3}a\).
```

### 10.26 Sec. III E: periastron-advance derivation sentence (verbosity-audit row 32)

```latex
\(\gamma_{\rm PN}=\beta_{\rm PN}=1\)~\cite{Will2014}. A massive probe's principal symbol, Eq.~\eqref{eq:dr-cone-expansion}, adds \(\delta H_a=-a^2\sum_i[p_i^4/(6m)+2v_ip_i^3/3]\) to its geodesic Hamiltonian at leading weak-field, low-momentum order; averaged over a Kepler orbit with plane normal \(\hat L\), it multiplies the periastron advance of general relativity by \(1-\tfrac12(1+\sum_i\hat L_i^4)(am)^2\), by differentiating the orbit-averaged correction with respect to the angular action (Ref.~\cite{McKinneyCode2026}): the exterior's lattice corrections are the probe's own.
```

### 10.27 Sec. III E: null-circle clause and the Appendix A pointer (verbosity-audit row 34; the pointer was wrong, Appendix A held only the Cavendish calibration)

```latex
At fixed \(E\), the vacuum exterior beyond averaging support is independent of \(\ell\). Averaging preserves \(E=a^3\sum_x\rho_x\) for fixed bare sources with \(J=0\), \(N=1\), and a regular centre (Appendix~\ref{app:example}). Where contained in that exterior, the trapping root \(r_s=2GE\) of \(N+v=1-\sqrt{r_s/r}\) and long-wavelength null circle \(3GE\), the extremum of \((1-r_s/r)/r^2\), are likewise independent. The back-averaged flow of Eq.~\eqref{eq:dr-boundary-variation}, for \(r\gg\ell\) with exterior averaging support, is
```

### 10.28 Sec. III F: axial-term comparison factors (fiftyfold, 1.2-fold) (verbosity-audit row 36)

```latex
Resolving it needs independent slope calibration and matching, bandwidth, reference, and endpoint errors below this order. On a full vacuum collar, the matter-root and slope averaging shift \(9(\sigma/a)^2(\kappa_Ha)^2\) exceeds the constant axial term fiftyfold for Sec.~\ref{sec:star}'s kernels and 1.2-fold for cell variance \(a^2/12\) alone. The histories lack that collar, so they test temperature only at order one, not this coefficient.
```

### 10.29 Appendix A (A1_worked_example.tex) in full, folded into Sec. III E (Cavendish sentence) and III F (escape-fraction hedge) (verbosity-audit rows 34 and 115)

```latex
\section{A point-source exterior}\label{app:example}

% assumption: the Cavendish calibration treats laboratory test bodies as slow normal-comoving valley rays; composites carry the free-fall factor Upsilon^2 of Sec. III A, negligible only because (ak)^2 is Planck-suppressed.
% risk: low
% GR slippage: no
% fold into text: no
For a weak, slow source and a slow normal-comoving continuum test body,
\(\ddot r=vv'=-GE/r^2\) and, for a test mass \(m_{\rm t}\),
\(F_r=-GEm_{\rm t}/r^2\),
the Cavendish match to \(G\), \(E\) the source's rest energy to
leading order. No escape fraction follows from the point-sampled
lattice control of Eq.~\eqref{eq:dr-crossing-ray}.

```

### 10.30 Sec. IV A: compactness 0.229 and the chi < 1.84 zeta^2 / 1.9 zeta^2 size criterion (verbosity-audit row 41)

```latex
The compactness is \(2GE/R=2c_E/c_R=0.229\). Below, the count sets \(mN_B/E_{\max}=\zeta^2(r_s/a)\sqrt g/(2c_E)\), \(r_s=2GmN_B\) nominal, 4.71 at the nominal \((\zeta,r_s/a)=(0.2,64)\), and the charge \(E/E_{\max}=\eta_E\,mN_B/E_{\max}\), 4.35 at the collapse's own charge with the binding fraction \(\eta_E=E/(mN_B)=0.923\) at release (Sec.~\ref{sec:scaling}); \(mN_B\) exceeds \(E_{\max}\) where the size ratio \(\chi=a/r_s\) is below \(\zeta^2\sqrt g/(2c_E)\), \(1.84\zeta^2\) at \(g=\tfrac14\) and \(1.9\zeta^2\) at the spacing of Eq.~\eqref{eq:page-share}, a criterion comparing \(mN_B\) with \(E_{\max}\), not an identification of \(E_{\max}/m\) with the family's maximum count, whose binding fraction is not computed.
```

### 10.31 Sec. IV A: chain polarization counting sentence (verbosity-audit row 42)

```latex
Their polarization counts derivatives of \((k/r,m)\) in the total gradient order, with mass-gradient parameter \(k/(k^2+m^2r^2)\); the cubic operator has no corresponding mass-vector term.
```

### 10.32 Sec. IV A: critical averaging width 0.2609 of 2GE (verbosity-audit row 46)

```latex
\(\sigma/a=2\), 4, and 8: a factor 4 between \(a\ll\sigma\) and the critical width 0.2609 of \(2GE\), above which any \(0\le\mathcal M\le E\) at \(N=1\) has no back-averaged trapping zero.
```

### 10.33 Sec. IV B: comoving-surface redshift sentence, duplicate of the Sec. VI B law (verbosity-audit row 48)

```latex
Outgoing continuum light from this comoving surface is received at fraction \(1-\sqrt{r_s/R_b}\to0\): flow redshift at \(N=1\).
```

### 10.34 Sec. IV B: width-agreement paragraph before the no-loss set (isotropic size-64 0.25 r_s and 1.4 percent; width 16 forming after the hydrodynamic test fails) (verbosity-audit row 52)

```latex
Width agreement holds only near the reference size. Wider averaging delays formation and raises its root fraction, halving the step advances it by fractions of \(r_s\), and isotropic preparation at size 64 forms \(0.25r_s\) later with matched-clock radii agreeing within 1.4 percent. Size 32 and \(\zeta=0.4\) fail width agreement; the 128 pair passes only at formation, width 16 forming after the hydrodynamic test fails; the size-256 formation, reported only at width 8 without a width span or law, stays a reported, not verified, reading, the only documented size-256 run differing in preparation (Ref.~\cite{McKinneyCode2026}).
```

### 10.35 Sec. IV C: second-order comparator energy, order counting in G and the exact slice values 8.06e-4, 4.21e-3, 0.136, 1.095 (verbosity-audit row 55)

```latex
nonpositive, strictly negative if either numerator is nonzero, the first sum absent only if \(P_gB(I-P_g)=0\); with the denominators bounded below by the gap uniformly in \(G\) and controlled sums, amplitudes of order \(G\) give order \(G^2\), while a mixed density-current Hessian element of order \(\sqrt G\) can give order \(G\) (Ref.~\cite{McKinneyCode2026}). On the reference collapse's stored slices at \(T/r_s=11.0\) and 15.0, two independent calculations expand about each slice's Hartree covariance \(\bar C\), ordered against the stored history's held projector as comparator, and find \(|\Delta E_{\rm ref}^{(2)}|/E=8.06\times10^{-4}\) and \(4.21\times10^{-3}\). The comparator's squared first-order dressing norm is 0.136 and 1.095, almost all of it from the \(B\) singles, which nothing balances on the matter-free comparator. The expansion about it therefore has no small-dressing control at 15.0, it is not stationary under \(\widehat E_2\), and its dressed state's branch weight is untested.
```

### 10.36 Sec. IV C: stationary-dressing paragraph, contraction mechanism (verbosity-audit row 56)

```latex
Stationary dressing is evaluated about the fixed-occupation determinant of the slice Hamiltonian nearest the stored covariance in the sea-plus-dust number sector, where the Hartree reordering terms cancel exactly and the connected second-order energy is the signed single- and double-excitation sum, nonpositive for positive denominators (contractions in that determinant reorder the prepared-sea-ordered operator, the induced field cancelling the Hartree terms and leaving the exchange correction to the single-excitation amplitude; Ref.~\cite{McKinneyCode2026}); these are retained-order diagnostics, not controlled dressed energies, the earlier held-projector values evaluate a different comparator, and no dressing energy is assigned to the subtraction.
```

### 10.37 Sec. IV C: exclusion-saturated filling curvature sentence (Ha about 2.3), the strain-bound paragraph makes the same exclusion (verbosity-audit row 58)

```latex
Exclusion-saturated filling, as a filling-only benchmark with the polarization moments set aside, would have curvature length \(1/H\) below one spacing at every size (\(Ha\approx2.3\) at Eq.~\eqref{eq:page-share}'s spacing), outside hydrodynamics; it is not imposed.
```

### 10.38 Sec. IV C: same-source benchmark paragraph before the no-loss set (regular-centre failure ratio; 1 percent at 15 and 16, 3.1 percent on the 13 rooted slices) (verbosity-audit row 60)

```latex
The same-source assignment \(\mathcal M=m\mathcal N_<\), \(\mathcal N_<(r)\) the enclosed count, at constant bare energy per particle \(\epsilon_c/a\), a flat-slice benchmark, closes the law with \(N\propto\mathcal N_<^{\nu}\), \(\nu=(\epsilon_c-\zeta)/(2\zeta)\), and fails at a regular centre by the ratio of bare energy per particle to \(m\). On the stored slices at 15 and 16 the clock is no power of the enclosed count and \((\lambda-1)/2\) has no constant annulus, so no exponent is calibrated; Eq.~\eqref{eq:sc-clock-exact} integrated on the dense fields matches the clock's fall from \(R_{50}\) to the outer root within 1 percent at 15 and 16 and 3.1 percent on the 13 rooted slices, in two independent calculations (slopes and \(\lambda\) values in Ref.~\cite{McKinneyCode2026}).
```

### 10.39 Sec. V B: error-floor composition sentence (verbosity-audit row 75)

```latex
weights \(16k\) count identical copies. Every sum includes measured light-state impurity, truncation, and rounding in its error floor. Excess is not radiation entropy:
```

### 10.40 Sec. V C: channel-tail error floor 1.2e-4 and tail statistic 0.02-6.2 percent (verbosity-audit row 78)

```latex
and the computed sums end at \(k=18\) with an error floor of at most \(1.2\times10^{-4}\) of a sum above 3 nats and a common-mode tail statistic of 0.02--6.2 percent of the formed sums from \(T/r_s=80\) (channel-level details in Ref.~\cite{McKinneyCode2026}); the tail enters no result beyond this subsection.
```

### 10.41 Sec. V C: doubled-domain setup detail (4096 matter sites, 8192 light sites, vacuum appended beyond 64 r_s) (verbosity-audit row 79)

```latex
The exterior-cut excess reaches about nine nats by \(T/r_s=100\), a few parts in \(10^4\) of \(A(E)/(4G)\). The test light field evolves from its prepared sea at \(T=0\) to \(T/r_s=100\) on a separate doubled-domain collapse: 4096 matter sites, 8192 light sites, vacuum appended beyond \(64r_s\). Light gravity is omitted; the interior is past the limit of its hydrodynamic description. At \(2,16,24,30r_s\), sampled maxima are 8.26--9.50 nats at 100, and all five cuts \(2,16,24,30,48r_s\) then span \(1.7\times10^{-4}\) to \(2.7\times10^{-4}\) of \(A(E)/(4G)\) (Fig.~\ref{fig:page-v4}); field jumps can leave nonsummable reference mismatch (the root record, the non-monotonic \(30r_s\) excess, and the propagation and purity readings are in Ref.~\cite{McKinneyCode2026}).
```

### 10.42 Sec. VI C: burst-bound display and the B_12 formula (verbosity-audit row 99)

```latex
\(n_{\rm ch}\) channels carry at most
\begin{equation}
 \frac{E_{\Delta t}}E\le
 B_{12}\,\frac{n_{\rm ch}(\Delta t/r_s)(x_2^2-x_1^2)g\chi^2}{32\pi^3}
\end{equation}
of \(E\), with \(B_{12}=\tfrac12[1+2\sqrt{(x_1^2+x_1x_2+x_2^2)/3}/(x_1+x_2)]\) (Ref.~\cite{McKinneyCode2026}); frequency-diagonal occupations give \(B_{12}=1\), the filled-band count. For \(25\le x\le100\) (\(B_{12}=1.029\)) the one-solar-mass bound with \(n_{\rm ch}=192\) (\(k=\pm1,\pm2\)) and \(\Delta t=40r_s\) is near \(2\times10^{-72}\), escaped only by hardening to lattice energies, more channels, or a longer arrival.
```

### 10.43 Sec. VI D: pocket-count crossing radius in the form O(R_0^{2/3} a^{1/3}) at fixed G/a^2 (verbosity-audit row 100)

```latex
accumulated partner count at $R=O(aS_0^{1/3})$, or $O(R_0^{2/3}a^{1/3})$ at fixed $G/a^2$; channel and energy
```

### 10.44 Sec. VI D: three common inputs, component lists (verbosity-audit row 108)

```latex
Three common inputs still need independent calculations: the full heavy-light boundary-charge Hessian (energy, current, light-transport, and reference derivatives); the prepared charge/number sectors, weights, and multiplicities; and the root source balance (physical against reference transport through zero crossings; clock, background-current, moving-root, and exterior responses).
```

### 10.45 Sec. VII A: held one-ended-field mixed-packet remark (verbosity-audit row 110; Sec. V B keeps the bound-block statement)

```latex
The information paradox needs no impermeable surface~\cite{Hawking1976}. For earlier radiation \(R\), a new mode \(b\), and its retained partner \(c\), weak monotonicity gives \(S(Rb)-S(R)\ge S(c)-S(bc)\), so \(S(c)\ge\ln2-\epsilon\) and \(S(bc)\le\epsilon\) force a radiation-entropy increase of at least \(\ln2-2\epsilon\) whatever the surface's permeability (Ref.~\cite{McKinneyCode2026}); the same monogamy underlies the firewall argument~\cite{AMPS2013}. Transfer of information must therefore carry partner correlations, not just probability, and on held one-ended fields a packet mixed with an above-band bound level stays mixed.
```

### 10.46 Appendix D: finite-moment realization, verification setup values (no surface-crossing hopping, epsilon 1e-4, Gaussian width 2a, positive sources, N to 1e-4, flow to 0.2 percent, gap 0.12, coefficient norm 1.1e-7) (verbosity-audit row 117)

```latex
Two independent calculations verify a candidate using the full reference of Eq.~\eqref{eq:dm-reference}, with the spin-curl Hamiltonian, cubic gapped \(W_2[N,v,b]\) and preparation \(u_\star=u_0\), \(P_\star=P_-(h[u_0])\) (Ref.~\cite{McKinneyCode2026}). The ball has radius \(8a\), no surface-crossing hopping, \(ma=0.2\), \(G/a^2=1/4\), \(\epsilon=10^{-4}\) before anchor correction and Gaussian per-axis width \(2a\), with positive sources, \(N=1\) to \(10^{-4}\) and uniform-source flow to 0.2 percent. The gap exceeds 0.12 along the straight path from \(h_0\). On 1028 tests of the four densities with compact clock and flow bumps, both calculations verify the corrected anchor and positive definite Gram matrix; the moments match exactly, and uniform enclosures including rounding and quadrature give strict mixedness for Euclidean coefficient norm at most \(1.1\times10^{-7}\).
```

### 10.47 Sec. III G: closing paragraph before inbox 55 item 1 (light-field statistical temperature "by about half if the two cut entropies match; this is uncomputed")

```latex
At the matched spacing Lorentz violation is Planckian (local Lorentz invariance holds at long wavelength, the dimension-six dispersion coefficient is Planckian at \(a\) near \(2\ell_P\), and composite-particle dispersion bounds lie outside this one-species model) and the cut's statistical temperature is \(T_H\): with controlled subleading entropy derivatives at fixed physical \(m,a,G\), \(T_{\rm cut}=(dS_{\rm cut}/dE)^{-1}=(T_H/f)[1+o(1)]\), so the entropy match and the statistical temperature select the same spacing at leading order in the one-field model. Including the light field (the massless species of Sec.~\ref{sec:page}) as Gauss-law matter lowers the total cut's statistical temperature at this spacing below \(T_H/f\), by about half if the two cut entropies match; this is uncomputed. The object's accessible state count remains separately open.
```

### 10.48 Sec. VII C: microcanonical question before inbox 55 item 1

```latex
and controlled reduced quantum evolution through an evaporation endpoint, which must realize the accessible charge sectors and statistical weights that microcanonical emission and absorption at \(T_H\) require; cut entropy and a Page maximum do not decide it.
```

### 10.49 Sec. III B: uniqueness example for the two-step current (payment for inbox 55 item 1)

```latex
No current depending only on \(u=f\nabla g-g\nabla f\) reproduces this bracket: \(f=\cos(\xi x_i)\) and \(g=\sin(\xi x_i)/\xi\) give \(u=e_i\) for every \(\xi\) while the bracket varies as \(F_a(\xi)/\xi\).
```

### 10.50 Sec. VI D: "Fig. 3 early cut excess tests neither its height nor its turnover" clause (payment for inbox 55; Secs. V C and VI C state it)

```latex
so the curve below is conditional on assumed laws, not derived object dynamics, and Fig.~\ref{fig:page-v4}'s early cut excess tests neither its height nor its turnover.
```

### 10.51 Sec. II D: lattice-corrections paragraph before the inbox 55 payment (vacuum-energy logarithm coefficients -1/(30 pi^2), +1/(40 pi^2), +1/(15 pi^2); exact off-diagonal split formula; Moyal mechanism) (verbosity-audit row 13)

```latex
The sources' lattice corrections are controlled: the first declaration's endpoint assignment has a lattice Moyal expansion whose first two projector correctors reproduce \(W_2\) of Sec.~\ref{sec:model-sources} with \(N,v,b\) independent, the endpoint difference entering only at fourth order; whose leading quadratic vacuum-energy logarithms for eight valleys, per \(|k|^4\log(\Lambda/\mu_{\rm IR})\) with \(\Lambda\) the momentum cutoff of order \(1/a\) and \(\mu_{\rm IR}\sim\max(m,|k|)\), are \(-1/(30\pi^2)\) for the clock, \(+1/(40\pi^2)\) for the transverse flow at \(b=\tfrac14\nabla\times v\), zero for the longitudinal flow and \(+1/(15\pi^2)\) for the orbital transverse flow at independent \(b=0\), quadratic-response coefficients only; and whose first-gradient kinetic flux is a momentum-magnetization curl vanishing for occupations scalar within each Dirac band, general preparation requiring the exact off-diagonal split \(a^3(S^{\rm op}_{x,ij}-S^{\rm kin}_{x,ij})=\Tr(\Delta_\star^{\rm off}\mathsf s_{x,ij})\), \(\Delta_\star=C-P_\star\): the endpoint assignment is an exact translated-coefficient product whose midpoint-hop expansion is the lattice Moyal expansion, the logarithms are the second-order symbol traced against the band projector, and the curl is the first Moyal term of the band-projected covariance (remainder bounds and the alias and hop estimates in Ref.~\cite{McKinneyCode2026}; neither covariant source conservation nor polarization stress follows from smooth fields).
```

### 10.52 Sec. III G: finite-part coefficients paragraph before the inbox 55 payment (b_100 = -0.4175, b_111 = -0.4464, b_110 = -0.4355 +- 0.0023, b-bar = -0.464 +- 0.002, [100] remainder order) (verbosity-audit row 38)

```latex
At fixed physical \(m\), the match uses the flat sea's massless coefficient and enclosure. Between momenta \(m\) and \(a^{-1}\), each of the eight valleys is a continuum Dirac field with half-space entropy singular as \(-\tfrac16\log(m^2+k_\perp^2)\), \(k_\perp\) the transverse momentum. Thus every planar cut has \(\eta(\zeta)-\eta(0)=\frac{\zeta^2}{3\pi}\log\zeta^2+O(\zeta^2)\). Where the second term is \(b_{\boldsymbol n}\zeta^2+o(\zeta^2)\), \(b_{\boldsymbol n}\) is the cut's finite part; the exact [100] remainder is \(O(\zeta^4|\log\zeta|)\). Interval arithmetic encloses \(b_{100}=-0.4175\) and \(b_{111}=-0.4464\). The single-calculation fit \(b_{110}=-0.4355\) uses five masses from 0.07 to 0.28, with propagated input error \(\pm0.0023\) and truncation about \(10^{-4}\), calibrated by the same fit to exact [100] and verified [111] values. For the sphere, \(\bar b\) is the angular mean of \(b_{\boldsymbol n}\) if the remainder is uniformly \(o(\zeta^2)\) over normals, a uniformity not proved here; fifteen-direction quadrature estimates \(\bar b=-0.464\pm0.002\) without a bound, entering only the \(10^{-32}\)-level spacing shift that follows.
```

### 10.53 Sec. VI D: inbox-52 collar sentence and its todo, superseded by the inbox-55 balance block

```latex
With exterior storage and apparatus work, \(dE_R/du=-L\) remains a separate root-source requirement: for a collar \(b\) outside the root, exactly \(E_R=E_0-\int L+\Delta\), with \(\Delta\) the collar balance remainder \(\mathcal M(b,u)-\mathcal M(b,u_a)+\int L\) less the change of the exterior inventory \(\mathcal M(b)-E_R\), and the curve below holds at leading order only if \(\sup|\Delta|=o(E_0)\) through turnover, which no two-field history yet tests. \todo{root-to-collar balance on a two-field history with the declared reference response, between-record evolution, and arrival clock}
```

### 10.54 Sec. VI D: single two-field record paragraph before inbox 55 item 2

```latex
A single stored two-field record (nominal \(r_{s,\mathrm{nom}}/a=80\), one light channel) finds, in two independent implementations, that the heavy-light reference term is about 8.6 percent of the rate of change of the mass inside the root, with the light field's term, absent from the history, 0.7 percent of it (Ref.~\cite{McKinneyCode2026}). It is one record, not a re-evolved balance; the complete moving-root and history balances stay open.
```

### 10.55 Sec. VI D: three-inputs sentence before inbox 55 item 2 (its todo is kept)

```latex
Three common inputs still need independent calculations: the full heavy-light boundary-charge Hessian, the prepared charge and number sectors with their weights and multiplicities, and the root source balance through zero crossings and the moving root (components in Ref.~\cite{McKinneyCode2026}).
```

### 10.56 Sec. IV D: between-record enclosure, four-item list before the inbox 55 payment (regular-centre limits, j_1 = J/r, approximate inverse Z_j, discrete-row clock estimate) (verbosity-audit row 70)

```latex
The deciding calculation is a validated coupled-trajectory enclosure on every consecutive record interval. Let \(F(C,u)=0\), \(u=(N,v)\), denote the boundary-normalized constraints with the stated averaging and full reference response. The enclosure must satisfy \(i\dot C=[\bar h(u),C]\) and \(F_u\dot u=-F_C\dot C\), and connect to the next interval. Its four sufficient conditions are:
\begin{enumerate}[label=(\roman*),leftmargin=*]
\item
\emph{Source and trajectory consistency.}
Enclose a solution under Eq.~\eqref{eq:dm-reference}, including the propagated sea, all active species and copy weights, and initial-data, assembly, quadrature, and evolution errors. Constraint and evolution defects, boundary charge and covariance spectra must be enclosed consistently.

\item
\emph{Admission and clock control.}
Throughout \([t_j,t_{j+1}]\), verify a regular inward branch and a positive margin \(\mu_3=\mathcal M(r,t)/r^3\ge\mu_{*,j}>0\) for \(0<r\le L\). Its regular-centre limit is \(4\pi\rho_\ell(0,t)/3\), implying \(\rho_\ell(0,t)\ge3\mu_{*,j}/(4\pi)\) and \(\mathsf k^2\ge2G\mu_{*,j}\). Verify Eq.~\eqref{eq:sg-current-mass} on the same enclosure, including every tail and low-density cell. At the centre use \(j_1=J_{\ell r}/r\) and \(j_1^2\le\Gamma_j\mu_3\), with enclosed regular limits. Use the full constraint mass, not enclosed rest mass. Discrete rows require the corresponding clock estimate or a bounded radial reconstruction defect.

\item
\emph{Field solvability.}
In fixed, printed coordinates and norms, verify a uniform inverse bound for the full \(F_u\), including reference response. For example, an approximate inverse \(Z_j\) with \(\sup\|I-Z_jF_u\|\le q_j<1\) throughout the enclosure gives \(\|F_u^{-1}\|\le\|Z_j\|/(1-q_j)\). A clock bound alone does not establish this condition, nor do the inverses verified at stored records.

\item
\emph{No other exit.}
Bound the fields, required spatial regularity, and reference derivatives, and close the trajectory enclosure on itself throughout the interval, not only at its endpoints. Successive enclosures must continue the same branch.
\end{enumerate}
```

### 10.57 Sec. VII A: transfer clause before inbox 55 item 3

```latex
the monogamy behind the firewall argument~\cite{AMPS2013}, and transfer of information must therefore carry partner correlations, not probability alone.
```

### 10.58 Sec. V C: conformal companion paragraph with Eq. page-conformal before the inbox 55 payment (verbosity-audit row 80)

```latex
A conformal companion reproduces the \(2r_s\) formation excess until hard formation light arrives. Before flux reaches the collar, it describes that excess as follows. Removing the massless channel's centrifugal term on the same fields and unfolding the regular centre gives one chiral Dirac field on a ray circle of circumference \(L_{\rm c}=265r_s\), labelled by entry advanced time \(p\). Cut entropy is the arc entropy between outgoing and incoming labels \(p_{\rm out},p_{\rm in}\):
\begin{equation}
 \begin{aligned}
 S={}&\frac16\ln\Big[\frac{L_{\rm c}}\pi\sin\frac{\pi(p_{\rm in}-p_{\rm out})}{L_{\rm c}}\Big]\\
 &-\frac1{12}\ln\Big[\frac{dp_{\rm out}}{dt}\,\frac{dp_{\rm in}}{dt}\Big]
 +\frac1{12}\ln\big[N^2-v^2\big]_{R}+{\rm const},
\end{aligned}
 \label{eq:page-conformal}
\end{equation}
per sector, two per chain. This is exact for the antiperiodic conformal vacuum with fixed reflection phases and a spacelike cut, and a scaling description of the chain~\cite{HolzheyLarsenWilczek1994,CalabreseCardy2004,BianchiSmerlak2014}. Exponential stretching adds \(\kappa_u/12\) per sector per unit time, the thermal entropy flux at \(\kappa_u/(2\pi)\). The propagated chain's growth agrees within \(4\times10^{-4}\) nats per sector below two e-foldings and \(6\times10^{-3}\) at three. Hard formation light, the light emitted when the trapping surface forms, then breaks this agreement: growth at \(2r_s\) levels at 1.28--1.40 nats per sector over 60--100, with about half that excess over the held vacuum. The late level's mechanism and size or trapped-region dependence remain uncomputed.
```

### 10.59 Sec. VII B: one-clause characterizations of the cited approaches (verbosity-audit row 113; every citation kept)

```latex
Thermodynamic derivations on a smooth spacetime~\cite{Jacobson1995,Jacobson2016,Verlinde2011,Padmanabhan2010}, holographic~\cite{RyuTakayanagi2006,Faulkner2014,VanRaamsdonk2010} and non-holographic~\cite{CaoCarrollMichalakis2017,CaoCarroll2018} reconstructions of geometry from entanglement, island Page curves from quantum extremal surfaces~\cite{Penington2020,AEMM2019}, and geometry read from relative-entropy actions~\cite{Bianconi2025}, conditional expectations~\cite{AhmadKlinger2025}, or correlations~\cite{LeEtAl2025,VedlEtAl2025,Ansel2024} all start further along the emergence direction; Regge calculus~\cite{Regge1961}, causal dynamical triangulations, and loop quantum gravity~\cite{BojowaldEtAl2016} discretize geometry or deform its algebra, and regular black holes~\cite{SimpsonVisser2019,Hayward2006}, gravastars~\cite{MazurMottola2004}, horizonless collapse~\cite{BarceloEtAl2008,KawaiMatsuoYokokura2013,Yokokura2025}, Planck stars~\cite{RovelliVidotto2014}, and fuzzballs~\cite{Mathur2009} fix interiors, whereas here only matter is on the lattice and geometry is hydrodynamic.
```

### 10.60 Sec. III: table-like TikZ schematic (fig:schematic) replaced by the visual overview (owner comment); the removed figure source follows verbatim

```latex
\begin{figure*}[t]
\centering
\begin{tikzpicture}[x=1cm,y=1cm,
  box/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,inner sep=3pt,text width=3.0cm},
  wide/.style={draw,rounded corners=2pt,align=center,font=\footnotesize,inner sep=3pt,text width=7.7cm},
  lab/.style={font=\footnotesize\bfseries,anchor=south west,inner sep=1pt},
  note/.style={font=\footnotesize,align=left}]
\node[box] (d1) at (1.7,7.0) {\textbf{1. The matter}\\ one Dirac fermion on the fixed cubic lattice; each bond's energy and current assigned to its two sites (\ref{sec:model-matter})};
\node[box] (d2) at (5.05,7.0) {\textbf{2. The principle}\\ energy is a charge: a clock rate and a flow velocity weight energy and momentum; one constant $G$ (\ref{sec:model-dictionary})};
\node[box] (d3) at (8.4,7.0) {\textbf{3. The hydrodynamic regime}\\ sources and fields averaged over a scale $\ell$ with $a\ll\ell\ll L_{\rm geom}$ (\ref{sec:model-dictionary})};
\node[box] (d4) at (11.75,7.0) {\textbf{4. Vacuum subtraction}\\ sources referenced against the prepared sea, so uniform fields source nothing (\ref{sec:model-sources})};
\node[box] (d5) at (15.1,7.0) {\textbf{5. The spherical quantum form}\\ Wick ordering against the prepared sea; evolution on the branch $\Pi$ (\ref{sec:model-operators})};
\node[lab] at ([yshift=3pt]d1.north west) {Assumed: the five declarations (Sec.~\ref{sec:model})};
\node[box] (s1) at (1.7,3.8) {\textbf{Gauss laws}\\ Einstein's constraints on flat slices, Eq.~\eqref{eq:dm-h2}};
\node[box] (s2) at (5.05,3.8) {\textbf{Metric}\\ the fermion's light cone: $ds^2=-N^2dt^2+|dx-v\,dt|^2$};
\node[box] (s3) at (8.4,3.8) {\textbf{Spherical rows}\\ $\mathcal M$ and $N$ from the sources; evolution rows where the sources are conserved};
\node[box] (s4) at (11.75,3.8) {\textbf{Continuum limit}\\ one parameter $g=G/a^2$; no tensor modes on the fixed lattice};
\node[box] (s5) at (15.1,3.8) {\textbf{Consequences}\\ Schwarzschild exterior, $T_H$, first law, spacing near $2\ell_P$ (\ref{sec:results-exterior} to \ref{sec:results-firstlaw})};
\node[lab] at ([yshift=3pt]s1.north west) {Derived: four steps (Sec.~\ref{sec:model-gravity}) and their consequences (Sec.~\ref{sec:results})};
\draw[->] (s1) -- (s2); \draw[->] (s2) -- (s3); \draw[->] (s3) -- (s4); \draw[->] (s4) -- (s5);
\draw[->] (8.4,6.05) -- (8.4,4.95) node[midway,right,note] {the principle, averaged and vacuum-subtracted, selects the laws};
\node[wide,draw=blue,text=blue] (c1) at (4.05,1.1) {\textbf{Cubic lattice: statics and entanglement}\\ flow threshold, area coefficient $\eta$ and cut entropies (\ref{sec:results-vacuum}); filling and momentum flux (\ref{sec:results-filling}); cubic-operator transfer test (\ref{sec:page-transfer})};
\node[wide,draw=red,text=red] (c2) at (12.75,1.1) {\textbf{Angular-channel chains: dynamics}\\ collapse histories and their diagnostics at Hartree order within the branch $\Pi$ (\ref{sec:star}); the light field on the collapse and held-field drains (\ref{sec:page}); the two-field record (\ref{sec:page-effective})};
\node[lab] at ([yshift=3pt]c1.north west) {Computed: two lattice descriptions};
\draw[->] (8.4,2.95) -- (8.4,2.15) node[midway,right,note] {solved on};
\node[draw,dashed,rounded corners=2pt,align=center,font=\footnotesize,inner sep=3pt,text width=16.4cm] (a1) at (8.4,-0.45) {\textbf{Assumed for the conditional Page curve only (\ref{sec:page-effective}):} emission, storage, and release laws; no result outside Sec.~\ref{sec:page-effective} uses them};
\end{tikzpicture}
\caption{What is assumed, derived, and computed. Top: the five
declarations of Sec.~\ref{sec:model}. Middle: the four derivation
steps of Sec.~\ref{sec:model-gravity} and their consequences in
Sec.~\ref{sec:results}. Bottom: the two lattice descriptions used in
the computations, the full cubic lattice for statics and entanglement
(blue) and the angular-channel chains, one radial chain per
angular-momentum channel, for the dynamics (red), with the sections
that use each; the emission, storage, and release laws of
Sec.~\ref{sec:page-effective} (dashed) are assumed, and only the
conditional Page curve uses them.}
\label{fig:schematic}
\end{figure*}
```

### 10.61 Sec. IV A: zero-mass chain-subtraction sentence before inbox 58 ("at any radius at zero mass")

```latex
Their polarization counting differs from the cubic operator's (Ref.~\cite{McKinneyCode2026}). For channels \(k\le3\), the chain subtraction is not verified below about \(2/m\), \(10a\) at \(ma=0.2\), or at any radius at zero mass. This radius is a working exclusion scale; it is not a remainder bound.
```

### 10.62 Sec. V A: light-field counterterm sentence before inbox 58

```latex
Each is normal ordered against its own prepared sea and carries its own polarization counterterms; averaged sources add in Eq.~\eqref{eq:dr-radial}, and the extension enters nowhere else.
```

### 10.63 Introduction: spherical-symmetry and tensor-mode block (owner batch 8 item 40)

```latex
what follows is derived from them or computed with them. The laws are not restricted to spherical symmetry, since the constraints hold on every flow, but the dynamics established is spherical. Spherical general relativity has no independent gravitational degrees of freedom, and the fixed lattice carries no tensor modes, so the one nonspherical result below, a stationary axial perturbation of the exterior (Sec.~\ref{sec:results-exterior}), establishes neither general nonspherical evolution nor gravitational waves.
```

### 10.64 Introduction: singularity roadmap sentence with the open-item clause (owner batch 8 item 41)

```latex
Without a smooth geometry a singularity needs a new definition. The section adopts failure of unique joint continuation of state, clock, and flow, states the domain on which continuation is unique, and checks its clock condition and the implemented Jacobians at every stored record of the computed collapse; uniform persistence of the domain is left open.
```

### 10.65 Sec. II B: third-declaration sentence and its explanation before owner batch 8 item 43 ("instantaneous")

```latex
\textbf{The third declaration is this hydrodynamic regime and its scale \(\ell\).} \fnd{Its Gauss laws are local in the smooth fields and instantaneous, with no field rates: locality and instantaneity belong to this regime, as does the regular dependence on finitely many spatial derivatives at zero gradients that admits the derivative expansion, in which the principle's four identities hold order by order.
```

### 10.66 Sec. II E: fifth-declaration sentences before owner batch 8 item 44

```latex
\fnd{\textbf{The fifth declaration, the spherical quantum form, fixes how the boundary charge's quantum fluctuations are counted and on which states the evolution acts: the charge operator is Wick ordered against the sea the collapse prepared, so that the prepared vacuum itself carries no pair energy, and the evolution acts on the branch, the subspace of states on which the boundary-charge operator stays within a factor two of its expectation, the range of the spectral projector \(\Pi\) of that operator.} The branch is part of the quantum theory, not an approximation to it, and the computed histories work at Hartree order within it. The quantum spherical sector is thus a small-fluctuation theory about the Hartree collapse: \(E[C]=\mathcal M(\infty)\) is expanded to second order in bare moments about the admitted Hartree covariance \(\bar C\), retaining the reference's polarization response, and fully Wick ordered against the prepared sea \(P_\star\); the polarization counterterms are scalars and order nothing. The pair vertex has zero Gaussian expectation and zero first covariance derivative at \(P_\star\) (the no-tadpole condition); vacuum subtraction alone does not fix this ordering.}
```

### 10.67 Sec. IV A: solver tolerances and verification rule (verbosity-audit row 44, owner-approved payment)

```latex
The solve requires relative charge change below \(10^{-7}\) and both field changes below \(10^{-6}\). Verification ends before the first kept field residual above \(10^{-3}\), or confirming residual where a further iterate is stored; later readings are labelled. Neither rule bounds field error. Records use flat-slice \(T\), not arrival time. Exact exponentials preserve purity; the massive fermion's purity residuals are not stored. The collar charge \(E\), the charge read at a fixed large radius outside the matter (the collar), has on held steps a second-order per-step drift with no fixed sign.
```

### 10.68 Sec. IV B: members median kappa_R range 0.74-1.21 (verbosity-audit row 53, owner-approved payment)

```latex
On seven of ten reference-collapse clocks, hovering redshift tracks \(\int\kappa_Rdt\) with slope 0.85--1.25; members' median \(\kappa_R/[1/(4GE)]\) spans 0.74--1.21. Blueshifted arrivals at widths at least 8 at both sizes and both completions rule out a uniform sampled redshift bound.
```

### 10.69 Sec. IV B: continuum-map remarks (ak = 48 emission remark; dispersion increase 0.16) (verbosity-audit row 53, owner-approved payment)

```latex
These continuum-map values are not spectra: the reference collapse's maximum at \(\omega_Dr_s=0.1\) would need emission at \(ak=48\). Same-event lattice rays at \(ak=0.05\)--0.3 arrive earlier and, on the four most redshifted histories, less redshifted: \(\ln(1/g_D)\) near the crossover scale \(\ln[2/(ak)^2]\) (values in Ref.~\cite{McKinneyCode2026}). This crossover is not a bound; some exceed it. After early root closure, dispersion can instead increase redshift by 0.16. No packet or finite-band spectrum is computed.
```

### 10.70 Sec. IV C: R_pi = 10.5a (isotropic ball) and 2.67a filling-volume radius (verbosity-audit row 59, owner-approved payment)

```latex
1.8 times the benchmark radius \(R_\pi\) (\(10.5a\) for the isotropic ball) and above the filling-volume comparison radius (\(2.67a\) on the chains), and rises again by 16;
```

### 10.71 Sec. IV C: 51-record hydrodynamic-test norm sentence (verbosity-audit row 61, owner-approved payment)

```latex
The hydrodynamic test's norm \(\sigma\|K\|_{\rm op}\ge0.1\) at all 51 reference outer-root records, its inner roots, and every \(\sigma/a=8\) root.
```

### 10.72 Sec. III C: eta(0) estimates paragraph before inbox 55 payment (-44/45 benchmark; digital-shell plateau remark) (verbosity-audit row 24)

```latex
\fnd{The numerical estimates of \(\eta(0)\) agree, but only the convexity bound is rigorous. A numerical bound on the area coefficient from interpolating upper and lower estimates over 205 lattice directions gives \(0.91905664<\eta(0)<0.91929748\), numerical evidence under an empirical quadratic-convergence premise that its 32 edge directions fail. The two exact axis values and convexity alone establish \(0.8164<\eta(0)<1.0203\). A ten-radius ball fit using the continuum sphere logarithm \(-44/45\) as a benchmark estimates \(\eta(0)=0.9191\) with fit spread 0.0017; the spread is not an error bound. Digital shells are deterministic: the entropy is constant while the site set is unchanged. Width-three squared-radius plateaus imply \(\limsup_{r\to\infty}r^2|S_A(P_0)/(4\pi r^2)-\eta| \ge3\eta/2\), a statement about some radii; it does not cover every cut. Neither the logarithm nor the extrapolation is established by this fit.}
```

### 10.73 Sec. III C: eta(0.2) paragraph before inbox 55 payment (0.7567805, 0.89239, 0.838636 +- 0.000004, 0.8073-0.9016, 0.8843-0.8880, -0.0320 +- 0.0042) (verbosity-audit row 25)

```latex
At \(ma=0.2\) the same chain of bounds and estimates gives \(\eta(0.2)\) near 0.887. There \(\eta_{100}=0.7567805\) and \(\eta_{111}=0.89239\) are enclosed to the printed digits by interval arithmetic on their layer chains, and \(\eta_{110}=0.838636\pm0.000004\) is measured under the massless interpolation bound's error rule and reproduced to seven digits by its layer chain's half-chain spectrum. The two enclosed values and the convexity establish \(0.7883<\eta(0.2)<0.9831\); with the measured [110] interval as a further premise, \(0.8073<\eta(0.2)<0.9016\); the 22-direction interpolation bound, numerical evidence under the massless bound's premise, gives \(0.8843<\eta(0.2)<0.8880\); and the paired ball fit, \(\eta(0.2)-\eta(0)=-0.0320\pm0.0042\) (a model envelope), gives 0.887, inside all three.
```

### 10.74 Sec. III C: gap-criterion inputs (C_1 = n^* = v^* = 1, g_1(0.2) about 0.0577, epsilon <= 1e-5) (verbosity-audit row 28)

```latex
At \(ma=0.2\), with Appendix~\ref{app:local}'s field bounds \(C_1=n^*=v^*=1\) and an independently supplied pathwise frozen-gap lower bound \(\mathsf g_*\ge\mathfrak g_1(0.2)\simeq0.0577\), that criterion is met for \(\epsilon\le10^{-5}\) with the integer localization length specified in the proof.
```

### 10.75 Sec. III C: Unruh paragraph before inbox 55 payment (M_perp, b(M) expansion, Eq. dr-unruh display) (verbosity-audit row 23)

```latex
The flat sea's half-space modular operator is a boost, and its temperature is the acceleration temperature. For \(C=P_0\) the infinite midbond [100] half-space has the exact one-particle modular operator \(k_A=\log(C_A^{-1}-1)=b(M_\perp)B_A\)~\cite{Eisler2025}, \(B_A\) the half-space boost generator, \(M_\perp^2=(am)^2+\sum_{j=2}^3\sin^2(ap_j)\), \(b(M)=2\pi[1-M^2/4+O(M^4)]\); on constant fields, in \(X=x-vt\), \(T=Nt\), as \(a\to0\) at fixed physical mass, acceleration, and resolved momenta each valley has the Lorentz-invariant Dirac vacuum, whose boost modular generator~\cite{BisognanoWichmann1976} an observer of constant proper acceleration \(\mathfrak a\) reads as the temperature~\cite{Unruh1976}
\begin{equation}
T_U=\frac{\mathfrak a}{2\pi};
\label{eq:dr-unruh}
\end{equation}
the lattice \(b\) corrects this modular statement and is no detector temperature for curved flow.
```

### 10.76 Sec. III D: kinetic-flux pressure paragraph with Eq. dr-pressure (p_Q below p_th) (verbosity-audit row 31)

```latex
For the stress \(S_{ij}\) of Eq.~\eqref{eq:dm-evolution}, the kinetic flux of \(Q\), cubic symmetry, and periodic integration by parts give
\begin{equation}
\begin{aligned}
p_Q&=\frac23\int_p\frac{|q|^2}{\varepsilon}f_\mu,\\
p_{\rm th}-p_Q&=\frac{4a^2}{3}\int_p|d|^2(\mu-\varepsilon)_+>0\quad(\mu>m),
\end{aligned}
\label{eq:dr-pressure}
\end{equation}
so its homogeneous isotropic part is \(p_Q\), below \(p_{\rm th}\).
```

### 10.77 Sec. III E: monotonicity sentence behind "an outgoing branch exists at every inward speed" (verbosity-audit row 35)

```latex
On \(0<\cos z<\min\{1,1/u_r\}\), \(a\omega\) rises from 0 to 1; \(\dot r\), concave in \(\cos z\), exceeds \(\min\{u_r,|1-u_r|\}\) and vanishes only at \(u_r=1\), \(z=0\). Thus an outgoing branch exists at every inward speed:
```

### 10.78 Sec. II B: dipole-improvement uniqueness argument for the current (verbosity-audit row 8; owner batch 8 payment)

```latex
\fnd{In the uncoupled model the flow multiplies this exact lattice current and the spin curl below. A translation-invariant finite-range Hermitian dipole improvement \(B_i\) adds the purely interband \(i[h_0,B_i]\); since \(Q_i\) commutes with the flat sea, the filled negative-energy states of \(h_0\) (Sec.~\ref{sec:model-sources}), the improved current commutes only if this addition vanishes.}
```

### 10.79 Sec. II C: rigid-turn surface-term sentence (verbosity-audit row 10c; owner batch 8 payment)

```latex
The completed rigid turn \(\delta v=\omega\times x\), \(\delta b=\omega/2\) leaves \(F\) unchanged pointwise, so \(W_2\) and its bulk functional derivatives are invariant; the surface term \(\tfrac12\oint n_i(\omega\times x)_j(\mathsf T:F)_{ij}/N\) arises only in the integrated bulk Euler pairing on a bounded region, so Euler's identity is used on a lattice with no outgoing bonds or with that boundary functional included.
```

### 10.80 Sec. IV E (from III C): direct-state budget counts K <= 7.07 over 44 histories, K <= 7.49 over 20 records, 2 percent agreement (verbosity-audit row 27; owner batch 8 payment)

```latex
\(K\le7.07\) over 44 histories and \(K\le7.49\) over 20 records at \(r_s/a=32,64,256\), with no observed growth in \(r_s/a\), matched-clock values agreeing within 2 percent.
```

### 10.81 Sec. IV B: collar-charge deficit value 0.0005 on the conserving sigma/a = 4 step (verbosity-audit row 50; owner batch 8 payment)

```latex
Centroid minima move inward with width (centroid motion, not interior rebound), and the collar charge at 16 shows a held-step deficit that halves with step length and is 0.0005 on the conserving \(\sigma/a=4\) step; the deficit is a held-step effect, not a charge-loss law.
```

### 10.82 Sec. VII A: restatement of the Sec. III consequences in the Scope paragraph (verbosity-audit row 109)

```latex
The fermion's own light cone then combines clock rate, flow velocity, and lattice spacing into a metric. Outside a spherical positive-energy source that metric is exactly Schwarzschild's; the exponential stretching of its outgoing rays converts the incoming vacuum into Fermi radiation at the Hawking temperature; and its first law assigns the exterior the Bekenstein-Hawking entropy, which the flat vacuum's cut entanglement supplies at a lattice spacing near two Planck lengths. The computed collapse
```

### 10.83 Fig. 3 caption: inset range, edge and endpoint sentences before the owner batch 8 payment (verbosity-audit row 104)

```latex
spread the switch is not computed. Inset: \(0.5036\le t/t_*\le0.5245\), approximately, fractions 0.35--0.65. Bands are attainable interval averages for release orders obeying the entropy law, nonnegative inventories, and complete formal release; edges need not share one history. Endpoint averages are one-sided, not final-quantum properties. Both panels assume constant coefficients, no pre-activation entropy, and formal continuation to \(t_*\). Bands show coefficient and ordering freedom, not physical error.}
```

### 10.84 Sec. II E: diagnostics-split paragraph, "The split behind the printed diagnostics is exact on smooth paths" (per-paragraph review note 30, CUT; owner batch 8 part 5 payment; the todo mark it carried stays in the paper after the preceding paragraph)

```latex
The split behind the printed diagnostics is exact on smooth paths:
where the level-following band projector
\(P\) continued from \(P_\star\) is twice differentiable with
positive occupied-unoccupied separation, the sources split exactly
into the moments of a source-rate generator
\(\mathsf M=C-P-\mathsf Q_1\), of the first adiabatic corrector
\(\mathsf Q_1\) and of the band-reference defect
\(\mathsf d_\mu=\Tr(PA_\mu)-r_\mu\) (\(A_\mu\) the assignment vertices), all
of which gravitate, and the clock-bond stresses split exactly as
\(S^{\rm op}[C-P_\star]=S^{\rm op}[\mathsf M]+S^{\rm op}[\varPi]\)
with \(\varPi=P-P_\star+\mathsf Q_1\) the polarization operator
(letters distinct from \(Q_i\), \(\mathcal M\), and the branch
projector \(\Pi\)), the operator-minus-kinetic difference being the
operator stress of the parts off-diagonal in \(P_\star\);
\(S_{ij}=S^{\rm kin}_{ij}[C-P_\star]\) is the paper's stress by
definition, and the other two print as diagnostics. Reading
\(S^{\rm op}[\mathsf M]\) as a conserving continuum stress would need
bounds on the band-reference defect, the corrector moments, and the
defect terms of the bond-gradient expansion (the \(\cos(2ap_i)\)
advection weight at finite spacing and the assignment of bond energy
and current to sites); these have no proved bounds and are measured,
not bounded, on the stored histories (Sec.~\ref{sec:star}; two-calculation
check in the to-do below), and no result here uses that reading (the
flux-force functional \(i\Tr(B[\bar h,A_\mu])\) gives the rate of
every Hartree-propagated moment, the level-following corrector
solving \([\bar h,\mathsf Q_1]=i\dot P\) cancels the reference rate
up to the defect's rate, and the split is linearity of
\(S^{\rm op}\) in \(C-P_\star=\mathsf M+\varPi\);
Ref.~\cite{McKinneyCode2026}). \todo{the stored slice
audit of the stored histories in two independent calculations} The balance of a
directly coupled species is stated with its converter in
Sec.~\ref{sec:page}.
```

### 10.85 Sec. VII A: direct-coupling toy-model paragraph, "The direct-coupling toy model establishes conversion, not release" (per-paragraph review note 175, CUT; owner batch 8 part 5 payment, second cut needed by layout hysteresis)

```latex
The direct-coupling toy model establishes conversion, not release: Sec.~\ref{sec:page}'s finite-chain model fixes a dimensionless strength of one before any computation, independently of entropy or gravity (controls at one half, two, and zero), and its two- and three-site tests establish heavy-light pair conversion and nothing further, not release, return-excluded purification, encoded-input recovery, conversion time, Page time, turnover, or maximum.
```

### 10.86 Appendix C: "Finite-moment realization" paragraph before shortening (per-paragraph review note 190, SHORTEN; owner batch 8 part 5 payment, third lever needed by layout hysteresis). The explicit right inverse, the corrected covariance, the mixedness margin, and the preparation of the verified candidate are kept here verbatim.

```latex
\paragraph{Finite-moment realization.}
Let a regular strictly mixed covariance \(C_0\) satisfy
\(\epsilon I\le C_0\le(1-\epsilon)I\) on a differentiable reference
branch. Smoothly weighted density assignments define test vertices
\(A_\mu\). If the Gram matrix \(\Gamma_{\mu\nu}=\Tr(A_\mu A_\nu)\) is positive definite, \(L(X)_\mu=\Tr(A_\mu X)\) has right inverse
\(R(s)=\sum_{\mu\nu}A_\mu(\Gamma^{-1})_{\mu\nu}s_\nu\). For fields \(u=(N,v)\) near a regular \(u_0\), with \(r_{\rm test}(u)\) the reference moments of Eq.~\eqref{eq:dm-reference} on the tests and \(s_g(u)\) the sources the laws assign to \(u\) there, the anchor obeying \(L(C_0)-r_{\rm test}(u_0)=s_g(u_0)\), set \(C(u)=C_0+R[s_g(u)-s_g(u_0)+r_{\rm test}(u)-r_{\rm test}(u_0)]\); then \(L(C(u))-r_{\rm test}(u)=s_g(u)\) exactly, the reference's response included, and \(C(u)\) stays strictly mixed while the added operator's norm is below \(d_0=\min\{\lambda_{\min}(C_0),1-\lambda_{\max}(C_0)\}>0\). The
construction tests the realization of the imposed identities on the
retained moments only; it establishes neither the full smoothed source
equations, the inverse of the laws beyond spherical symmetry, nor
any deformation-algebra closure. Two independent calculations verify a candidate using the full
reference of Eq.~\eqref{eq:dm-reference}, with the spin-curl
Hamiltonian, cubic gapped \(W_2[N,v,b]\), and preparation
\(u_\star=u_0\), \(P_\star=P_-(h[u_0])\), on a ball of radius \(8a\)
at \(ma=0.2\), \(G/a^2=1/4\), over 1028 tests of the four densities
with compact clock and flow bumps: both verify the corrected anchor
and positive definite Gram matrix, the moments match exactly, and
uniform rigorous bounds including rounding and quadrature give strict
mixedness (setup values, gap, and coefficient-norm bound in
Ref.~\cite{McKinneyCode2026}).
```

### 10.87 Appendix A.6 "Local gap and local area" paragraph before shortening (per-paragraph review note 185, SHORTEN, mechanism to repository; owner batch 8 part 5 payment, fourth lever needed by layout hysteresis). The construction of the localized estimate (contour, comparison resolvent, cross-block trace norm, A.3 comparison term), the E_D remark, and the closing remarks are kept here verbatim.

```latex
Where a sign-reversed region is open inside the cut, the massive bound is
transferred from a clipped comparison. For \(\zeta>0\), fields obeying (H1)
and a finite sign-reversed region \(\mathcal U=\{x:|\bar v_x|>s_0\bar N_x\}\),
\(0<s_0<s_c\) with \(\mathfrak g_{s_c}(\zeta)>0\), let \(h^{\rm cmp}\) have
the flow clipped smoothly to \(|\bar v^{\rm cmp}_x|<s_c\bar N_x\) and
unchanged where \(|\bar v_x|\le s_0\bar N_x\): \(h-h^{\rm cmp}\) is
supported within two sites of \(\mathcal U\), and the clipped fields
obey (H1) and have a frozen gap by the Cauchy step of A.5, hence a gap
\(\delta^{\rm cmp}\) by Eq.~\eqref{eq:B-gapcert} for \(\epsilon\) small.
For an orthogonal projection \(P\) commuting with \(h\), let \(E_D\)
be the supremum of the absolute energies with non-ground-state
occupation. For \(0<\delta_1<\delta^{\rm cmp}\), require
\(r_b=\operatorname{rank}[(P-P_-(h))
\mathbf1_{\{|h|\ge\delta_1\}}]<\infty\).
Thus \(r_b=0\) if \(\delta_1>E_D\).
Reference~\cite{McKinneyCode2026}
constructs a nonincreasing
\(\mathcal E(d)\le C_{\mathcal E}e^{-\vartheta d}\) by localizing
the remaining projector difference and finite inner-window subspace
near the clipping support with a contour and comparison resolvent,
the cross-block trace norm giving \(\mathcal E(d)\) and A.3 the
comparison term.
Its constants depend on the finite clipping support, the operator
difference, the comparison gap, and a contour separating a chosen
inner spectral window; they are not uniform over growing sign-reversed regions.
The rate is a reduced Combes--Thomas rate of the comparison operator.
With \(d_0=\min\{d\ge0:\mathcal E(d)\le\log2\}\), the resulting
bound, for sufficiently small \(\epsilon\) and every
\(\varkappa\)-regular finite cut whose complement is at distance at
least \(d\) from \(\mathcal U\), is
\begin{equation}
\begin{aligned}
 |S_A(P)-S_A(P_0)|&\le C^{\rm cmp}_{\rm loc}\varkappa A_{\rm lat}\epsilon^2(1+|\log\epsilon|)^4\\
 &\quad+r_b\log2+\mathcal E(d),
\end{aligned}
\label{eq:B-transfer}
\end{equation}
the last two terms being at most \((r_b+1)\log2\) for \(d\ge d_0\).
Separate \(r_b\) occupation changes at cost \(r_b\log2\).
Unclipped \(h\) needs no zero gap.
```

### 10.88 Appendix A.4 "The massless finite-lattice estimate" paragraph before shortening (per-paragraph review note 182, SHORTEN, proof detail; applied early as the layout lever for the introduction commit)

```latex
Use the finite lattice \(\Lambda_L=(\mathbb Z/L\mathbb Z)^3\),
\(L\) even, with antiperiodic shifts and \(\epsilon=1/L\).
For each \(0\le\zeta\le\zeta_*\), \(P_0^{(L)}\) denotes the flat
projector at that same mass; it has no zero mode, including at
\(\zeta=0\). Assume \(n_*\le\bar N\le n^*\),
\(|\bar v|\le v^*\), \(n_*-v^*\ge c_*>0\), and
\(\sum_{\mathbf q}|\mathbf q|(|\widehat N(\mathbf q)|+\sum_i|\widehat v_i(\mathbf q)|)\le\mathcal F_1\epsilon\).
For a nonwrapping ball \(c_rL\le r\le0.45L\), and
\(P=\mathbb 1_{(-\infty,0)}(h)\) with any orthogonal zero-mode
filling,
\(|S_A(P)-S_A(P_0^{(L)})|\le CA_{\rm lat}\epsilon^{1/10}\),
where \(C\) depends only on
\((\zeta_*,\mathcal F_1,c_*,n^*,v^*,c_r)\).
This applies to a comparator projector with the stated spectral occupations. The proof
(Ref.~\cite{McKinneyCode2026}) splits at \(\delta=\epsilon^{2/5}\)
into rank \(O(L^3\delta^3)\) and norm \(O(\epsilon/\delta)\), the
cross-block entropy estimate giving the exponent \(1/10\); it also
bounds the flat finite-lattice entropy
difference by \(Cr^{3/2}(1+\log r)\), so A.1 gives the coefficient
\(\eta(\zeta)\) in the fixed-mass limit.
```

### 10.89 Appendix A.5 "At and across the trapping radius" paragraph before shortening (per-paragraph review note 184, SHORTEN, proof detail; the explicit Cauchy step and the Appendix B bound citation are kept here verbatim)

```latex
At and across the trapping radius. With \(t_i=\sin^2k_i\) and \(S=\sum_it_i\),
Cauchy's inequality gives \(|q|^2\le S-S^2/3\) while
\(\varepsilon^2=\zeta^2+S\), so where \(|\bar v(x)|\le s\bar N(x)\) the
frozen gap is at least \(\bar N(x)\mathfrak g_s(\zeta)\),
\(\mathfrak g_s(\zeta)=\min_{0\le S\le3}[\sqrt{\zeta^2+S}-s\sqrt{S-S^2/3}]\);
at \(s=1\), the trapping radius included, this is positive for \(\zeta>0\),
while where \(|\bar v(x)|>\bar N(x)\) the frozen gap is
positive exactly below the sign-reversal threshold of
Eq.~\eqref{eq:dr-flow-threshold} in the flow's direction, which is at least
\(w_c^{(3)}(\zeta)=1/\nu_3>1\) by the bound \(|q|/\varepsilon\le\nu_3\) of
Appendix~\ref{app:charge}. Along the linear path from the flat sea, in (H1) with
\(n_*\to\min\{1,n_*\}\), \(n^*\to\max\{1,n^*\}\), the frozen gap is at
least \(\min\{\zeta,\mathsf g_*\}\), so a massive configuration obeying
(H1) with \(\mathsf g_*>0\) meets (H2) and (H3) once the right side of
Eq.~\eqref{eq:B-gapcert} at these bounds and
\(\mathsf g_*\to\min\{\zeta,\mathsf g_*\}\) is positive;
for \(\zeta=0\) the frozen gap vanishes at the valleys for every
flow, which leaves the massless estimate.
```

### 10.90 Sec. IV A: solver-tolerance paragraph (per-paragraph review note 76, CUT; payment for the inbox 62/63 heavy-outer-endpoint sentence; the flat-slice-time clause is kept in the paper at the start of the next paragraph)

```latex
Solver tolerances and the verification rule, which bounds no field error, are in Ref.~\cite{McKinneyCode2026}; the stored slices use flat-slice \(T\), not arrival time; exact exponentials preserve purity (the massive fermion's purity residuals are not stored); and the shell charge \(E\), the charge read on a fixed shell at radius \(16r_s\) outside the matter, has on held steps a second-order per-step drift with no fixed sign.
```

### 10.91 Sec. IV B: capture-cross-section paragraph (per-paragraph review note 91, CUT, scope note only; payment for the inbox 62/63 sentence)

```latex
No capture cross section is reported: for quasi-static incidence \(m\ll\omega\), \(a\ll\omega^{-1}\ll r_s\), the leading capture term is \(27\pi G^2E^2f_{\rm nr}\), \(f_{\rm nr}\) the entering disk flux not returned within infinity-time \(\Delta T\), held-exterior probes remain in transit with unverified channel tails, and there is no free-fermion absorption.
```

### 10.92 Sec. II C: directly-coupled-extension scope paragraph (per-paragraph review note 15, CUT)

```latex
The directly coupled extension (Sec.~\ref{sec:page}) is outside this paper: no result uses it, its converter is absent from every computed history, and the requirements on an interacting generator's own reference are listed in Ref.~\cite{McKinneyCode2026}.
```

### 10.93 Sec. II D: first-step symmetry and dimension counting paragraph before shortening (review note 19, SHORTEN; the dimension counting, the time-reversal action on h, and the 13-to-5 coefficient counting are kept here)

```latex
% assumption: "with one constant" excludes quadratic-gradient terms whose coefficients are built from m or 1/a^2, which the matter's own polarization supplies (Sec. II C); the reduction to five constants therefore rests on the fourth declaration's subtraction and on the third declaration's clause that the lattice enters only through rulers and G.
% risk: medium
% GR slippage: no
% fold into text: no; declared in Sec. II B and II C; cross-reference
Uniform fields have constant fluxes, giving \(J_\ell=0\) and
\(\rho_\ell+(v/N)\cdot J_\ell=0\), hence \(\rho_\ell=0\):
no derivative-free term, consistent with flat-vacuum subtraction.
Time reversal \(\Theta\) (Appendix~\ref{app:local}) sends
\(h[N,v]\) to \(h[N,-v]\), preserving energy density and reversing
momentum density. Clock is even, flow odd, and \(\mathcal A_0\)
even in flow. This excludes terms linear in \(K\) or \(\Omega\),
their first derivatives, and their products with \(\nabla N/N\).
These three gradients have inverse-length dimension,
\(\mathcal A_0\) inverse fourth-power dimension. With one constant,
the leading term is quadratic with coefficient dimension \(1/G\);
clock second derivatives reduce to it modulo a divergence:
\(N\nabla\cdot(\nabla N/N)=\nabla^2N-N|\nabla N/N|^2\).
Inversion and time reversal exclude odd derivative orders.
The next terms have two more derivatives, relative factors
\((\ell_P/L_{\rm geom})^2\) and \((a/L_{\rm geom})^2\), with
undetermined coefficients and no bound. Counted on the laws, cubic
symmetry allows thirteen second-gradient source coefficients;
reciprocal response fixes the seven momentum ones from the four
flow-gradient energy ones and relates the two clock ones, so the laws
are exactly the variations, modulo a divergence, of
\begin{equation}
\begin{aligned}
\mathcal A={}&\frac{N}{16\pi G}\Bigl[a_1(\operatorname{tr}K)^2
+a_2\operatorname{tr}(K^2)+a_3\sum_i(e_i\cdot Ke_i)^2\Bigr.\\
&\Bigl.{}+b\operatorname{tr}(\Omega^{\mathsf T}\Omega)
+c\,\Bigl|\frac{\nabla N}{N}\Bigr|^2\Bigr],
\end{aligned}
\label{eq:dm-density}
\end{equation}
\(e_i\) the lattice axes, with five dimensionless constants, \(b\)
already zero by the third fact. The energy law is
\(\delta\mathcal A/\delta N\) and the momentum law
\(\delta\mathcal A/\delta v\) (Appendix~\ref{app:rows} for the
expressions); the charge requirement fixes the other constants.
```

### 10.94 Sec. II D: lattice-corrections control paragraph before shortening (review note 26, SHORTEN)

```latex
The sources' lattice corrections are controlled: the first
declaration's endpoint assignment has a lattice Moyal expansion whose
first two projector correctors reproduce \(W_2\) of
Sec.~\ref{sec:model-sources} with \(N,v,b\) independent, the endpoint
difference entering only at fourth order, and whose first-gradient
kinetic flux is a momentum-magnetization curl vanishing for
occupations scalar within each Dirac band (the vacuum-energy
logarithms, the exact off-diagonal split for general preparation,
and the remainder, alias, and hop estimates in
Ref.~\cite{McKinneyCode2026}); neither covariant source conservation
nor polarization stress follows from smooth fields.
\todo{the two-calculation stored slice check of the bond-gradient expansion}
```

### 10.95 Sec. II D: source-tensor paragraph before shortening (review note 27, SHORTEN; the restated coefficient selection is kept here)

```latex
% assumption: identifying (rho_l, -J_l, S_ij) with the normal-normal, normal-tangential and tangential components of a stress tensor in the derived line element presumes the matter is covariant in that metric at long wavelength (valley-supported, leading order in a|k|); the identification is a comparison with the ADM form, not a derivation of covariance.
% risk: medium
% GR slippage: possible (this is where the GR form is read off; it is a comparison, stated as such)
% fold into text: no; the conditions are listed in the third step
The source tensor
\(T\) has \(\rho_\ell\), \(-J_{\ell i}\), and \(S_{ij}\) as its
normal-normal, normal-tangential, and tangential components, so that
\(T_{0i}=-J_i\) at \(N=1\), \(v=0\). Equation~\eqref{eq:dm-h2} is
then the Hamiltonian and the momentum constraint of general
relativity on flat slices~\cite{ADM1962}: in the hydrodynamic limit
the Gauss laws take Einstein's form on every flow. After vacuum
subtraction uniform fields source nothing, so the laws have no
constant term, which is where a cosmological constant would enter;
and the momentum law carries the sign general relativity assigns
for this source tensor \(T\). What selects Einstein's constraints is the energy requirement on
flows with a potential, the fourth part of the principle: it fixes
\(a_1=-a_2\), \(a_3=0\), and \(c=0\). Without it, \(a_1\),
\(a_2\), \(a_3\), and \(c\) stay free, a family of flat-slice laws
whose isotropic members have the strain terms of Ho\v{r}ava's
gravity~\cite{Horava2009} and the clock-gradient term of its
extension~\cite{BlasPujolasSibiryakov2010}; the moment's flux fixes
\(b=0\), the value general relativity owes to the covariance of its
slices. Beyond the leading order each correction, at every higher
even order in the derivatives, must contribute to the charge
\(\rho_\ell+w\cdot J_\ell\) the divergence of a local field on flows
with a potential at any clock: a constraint on the sources, not the
density, not classified here.
```

### 10.96 Sec. II D: evolution-equation paragraph before shortening (review note 29, SHORTEN; the residual identities moved to Appendix C)

```latex
% assumption: the evolution rows follow from constraints plus conservation only with a regular centre and nonzero flow for r>0; where the flow vanishes at some r>0 the division by r^2 v fails and the evolution rows are not implied. Eq. dm-evolution itself is general relativity's equation, shown for comparison.
% risk: low
% GR slippage: no
% fold into text: no; "nonzero flow for r>0" is stated
Only the two spherical components of Einstein's evolution equation
on these slices~\cite{Dirac1958,ADM1962} are established, and only
where the sources are conserved; its
general form is shown for comparison, with \(\mathcal L_v\) the Lie
derivative along the flow:
\begin{equation}
\begin{aligned}
(\partial_t+\mathcal L_v)K_{ij}
&=\partial_i\partial_jN
 -N[(\operatorname{tr}K)K_{ij}-2K_{ik}K^k{}_j]\\
&\quad+8\pi GN
 \left[S_{ij}-\tfrac12\delta_{ij}(S-\rho_\ell)\right].
\end{aligned}
\label{eq:dm-evolution}
\end{equation}
Let \(R_r\) and \(R_t\) be the radial and transverse parts of its
left side minus its right side, and \(\epsilon_E\), \(\epsilon_M\) the
energy and momentum parts of the source tensor's divergence per unit proper time. With
\(\rho_\ell\) and \(J_\ell\) taken from the rows at every time,
\(4\pi GN\epsilon_E=K_tR_r+(K_r+K_t)R_t\) and
\(4\pi GN\epsilon_M=R_t'+(R_t-R_r)/r\) identically
(Appendix~\ref{app:rows}). Hence \((r^2vR_t)'=4\pi Gr^2N(N\epsilon_E+v\epsilon_M)\): \(r^2vR_t\) is the cumulative weighted defect plus \(r_b^2v(r_b)R_t(r_b)\) from an inner matching radius \(r_b\), a constant a regular centre sets to zero, so conservation gives \(R_t=R_r=0\). So the
spherical components of Eq.~\eqref{eq:dm-evolution}, the evolution
rows, hold exactly when the sources are conserved, without a field
evolution law of their own. Einstein identification at retained order requires a regular inward spherical branch (regular at the centre, with positive clock, nonnegative enclosed energy, and inward flow, Sec.~\ref{sec:singularity}), nonzero flow for \(r>0\), and a massive frozen symbol with positive gap. Appendix~\ref{app:local}'s gap criterion below the sign-reversal threshold suffices and can include the trapping radius (Sec.~\ref{sec:results-cone}). Matter must be valley-supported, prepared in \(\operatorname{Ran}(I-P_\star)\), with the propagated sea's departure from the prepared band and the added matter's \(P_\star\)-coherence small at retained order: history conditions, not consequences of a trapping radius or preparation projection. Finite-momentum, averaging and reference remainders need separate
control.
\todo{two-calculation reference-remainder checks above; scaling-family
gap margins, valley support, normalized sea/dust-coherence balances
and matched kinetic residuals} \todo{the stored slice audit of the stored histories in two independent calculations}
```

### 10.97 Sec. II E: quantum-form paragraph before shortening (review note 32, SHORTEN)

```latex
% assumption: Hartree order means clock and flow are c-number fields sourced by one-body expectation values (semiclassical gravity); the quantum form is an expansion about this branch whose convergence is not established, and the branch restriction of the fifth declaration is a projection postulate that keeps the quantum theory near the Hartree solution by construction.
% risk: medium
% GR slippage: no
% fold into text: no; declared
% assumption: the boundary-generator identity and conservation of E need the rows to be solvable differentiably in C (invertible constraint Jacobian) along the whole history; energy conservation is conditional on staying on the admitted branch.
% risk: low
% GR slippage: no
% fold into text: no; stated in Appendix C's hypotheses
Last comes the quantum form the computed collapse uses. Computations use the spherical quantum form's leading Hartree term,
the exact covariance evolution of Sec.~\ref{sec:model-dictionary}.
On a differentiable field branch
it determines Gaussian cut entropies; sources also require
Eq.~\eqref{eq:dm-reference}'s field-dependent reference moments.
The anticommutation algebra precedes the sea-generated Fock
representation; the pair vertex below additionally needs quartic correlations.
On a differentiable regular inward branch, varying Eq.~\eqref{eq:dr-radial} with \(\langle u,\delta_us\rangle=0\) from Eq.~\eqref{eq:dm-contact} gives the boundary-generator identity
for Sec.~\ref{sec:model-gravity}'s laws
(Appendix~\ref{app:charge}):
\begin{equation}
\delta E=\Tr(\bar h\,\delta C),\qquad
\bar h=h[W_\ell^\dagger N,W_\ell^\dagger v],
\label{eq:dr-boundary-variation}
\end{equation}
for every differentiable variation of \(C\) on the inward branch.
Along the Hartree evolution \(i\dot C=[\bar h,C]\) the boundary
charge, rather than \(E_m\), generates the matter's evolution and is
conserved by it (Appendix~\ref{app:charge}).
```

### 10.98 Sec. II E: fifth-declaration paragraph before shortening of its technical tail (review note 34, SHORTEN)

```latex
\fnd{\textbf{The fifth declaration is the quantum spherical sector:
the boundary-charge operator, Wick ordered against the sea the
collapse prepared, restricted to the states on which it stays within
a factor two of its expectation. The computations work at Hartree
order inside it.} That restriction is the branch, the range of the
spectral projector \(\Pi\) of the charge operator, and the ordering
leaves the prepared vacuum with no pair energy; the sector is part of
the quantum theory, not an approximation to it, a small-fluctuation
theory about the Hartree collapse in which \(E[C]=\mathcal M(\infty)\)
is expanded to second order in bare moments about the regular Hartree covariance \(\bar C\), retaining the reference's polarization
response, and fully Wick ordered against the prepared sea
\(P_\star\); the polarization counterterms are scalars and order
nothing, the pair vertex has zero Gaussian expectation and zero first
covariance derivative at \(P_\star\) (the no-tadpole condition), and
vacuum subtraction alone does not fix this ordering.} The branch holds the quantized boundary charge within a factor two
of its Hartree value \(E=E[\bar C]>0\): the deviation operator
\(\mathcal B=4(\widehat E_2/E-I)^2\) and its spectral projector
\(\Pi=\mathbf1_{[0,1]}(\mathcal B)=\mathbf1_{[E/2,3E/2]}(\widehat E_2)\)
select the states of small departure, invariant under
\(\widehat E_2\); \(H_+=\Pi\widehat E_2\Pi\) equals \(\widehat E_2\) on
\(\operatorname{Ran}\Pi\), satisfies
\(\tfrac12E\,\Pi\le H_+\le\tfrac32E\,\Pi\), and defines the
spherical evolution on the branch where it is nonempty, with leakage \(\|(I-\Pi)\widehat E_2\Pi\|=0\). \fnd{A normalized state has branch weight at least
\(1-\langle\mathcal B\rangle\), so \(\langle\mathcal B\rangle<1\)
establishes nonemptiness.} Interior masses \(\widehat M(r)\), expanded
likewise about \(M(r)>0\) with reference response, are diagnostics
outside the branch projector \(\Pi\): in any normalized state,
\(\Pr(\widehat M(r)\le0)\le\langle(\widehat M(r)/M(r)-I)^2\rangle\),
for each mass separately. Nonemptiness,
Hartree, and re-expansion errors must be measured.
Neither convergence nor divergence of the full series is established. The ordering leaves the second-order dressing energy, read in Sec.~\ref{sec:star} against the ground projector of the Hartree Hamiltonian, rather than subtracting it by declaration.
```

### 10.99 Sec. III A: packet-velocity paragraph before shortening (review note 38, SHORTEN; composite free-fall factor conditions kept here)

```latex
% assumption: "the equivalence principle holds exactly" refers only to the massive valley ray at k=0 and constant coupled clock; composites violate it at order (ak)^2 through Upsilon^2, so universality of free fall is not a property of the model beyond this measure-zero case.
% risk: low
% GR slippage: no
% fold into text: no; the qualification is in the paragraph
A positive-band packet has the velocity
\begin{equation}
\begin{aligned}
\dot x_i=\partial_{p_i}\omega_+&=N\frac{q_i}{\varepsilon}+v_i\cos(2ap_i),\\
\dot x_i|_{p=p_\star}&=v_i\qquad(m>0),
\end{aligned}
\label{eq:dr-comoving}
\end{equation}
so the massive valley ray moves with a uniform flow; at \(k=0\),
\(\dot k=-m\nabla N\), so marginally bound continuum dust with
constant \(N=1\) is comoving, the massless band having no rest point.
At constant coupled clock, the equivalence principle holds exactly for the
elementary massive valley ray comoving at zero relative momentum.
For a small isotropic inversion-symmetric composite of fixed
internal state in stationary irrotational coupled flow, it holds at quadratic centre-momentum order only up to the free-fall factor \(\Upsilon^2\), \(\Upsilon=\langle\cos(2ak_i)\rangle\) over internal momentum (centre Hamiltonian in Ref.~\cite{McKinneyCode2026}); Sec.~\ref{sec:star}'s elementary dust orbitals have no composite factor.
```

### 10.100 Sec. III B: Dirac-Schwinger bracket paragraph before shortening (review note 40, SHORTEN; remainder conditions kept here)

```latex
At long wavelength, for continuum-spinor smearings \(f,g\) (valley sum and valley-rotated spin matrices
understood), the bare bracket is
\begin{equation}
\begin{aligned}
i[H[f],H[g]]&=\mathcal P[u]+\mathcal S[u]+\mathcal R_a,\\
\mathcal P[u]&=\int d^3x\,\psi^\dagger\tfrac12\{u_i,-i\partial_i\}\psi,\\
\mathcal S[u]&=\tfrac14\int d^3x\,\psi^\dagger\boldsymbol\Sigma\cdot(\nabla\times u)\psi,
\end{aligned}
\label{eq:dr-dirac-schwinger}
\end{equation}
the Dirac-Schwinger form~\cite{Dirac1962,Schwinger1962} of a
relativistic energy density; for matter momenta of size \(p_c\) about
a valley and transfers of size \(q_c\), with \(ap_c,aq_c\ll1\),
bounded Fourier-smearing norms, and unwrapped transfers, the
one-particle remainder \(r_a\) on these modes obeys
\(\|r_a\|=O(a^2[p_c^3q_c+p_cq_c^3+p_c^2q_c^2+q_c^4])\),
an absolute estimate. Radial \(f,g\) have zero curl, so their
continuum bracket closes on the momentum density.
```

### 10.101 Sec. III C: flat-sea stability paragraph before shortening (review note 43, SHORTEN)

```latex
The flat sea is stable under uniform flow up to an exact threshold.
For homogeneous preparation
\(P_0=P_-(h_0)\), \(P_-\) the negative
spectral projection, the spin-scalar two-step current \(Q_i\) commutes with
\(P_0\). Uniform fields therefore preserve the sea and its cut entropies
at every flow speed, without refilling at zero-energy crossings or
overcritical flow. The ground projector
\(P_g=P_-(h)\), below a comparator projector (one against which cut
entropies are compared), equals \(P_0\) for
\(|v|<N\), and for \(j=1,2,3\)
equal nonzero Cartesian flow components below the exact bulk threshold
\begin{equation}
\begin{aligned}
w_c^{(j)}(am)&=\sqrt{1+(am)^2/j}+am/\sqrt j,\\
|v|/N<w_c^{(j)}(am)&\ \Longrightarrow\ P_g=P_0;
\end{aligned}
\label{eq:dr-flow-threshold}
\end{equation}
the massless threshold is 1 in every direction, with sign-reversed states for
\(|v|>N\), and no pointwise speed test fixes the ground projector on a nonuniform history.
```

### 10.102 Sec. III C: trapping-cut comparison paragraph before shortening (review note 48, SHORTEN)

```latex
% provenance: K<=7.07 / 7.49: <internal>/TRAPPING_CUT_EF_R2_v1.md (route 2, l.12 and l.125); spec WEBGPT40O_REDUCTION_v1 P02; route 1 not pinned.
\fnd{The trapping-cut entropy is compared with the flat sea's through
bounds that need no gap or energy premise, either through a
comparator projector or directly. For a pure
number-conserving Gaussian covariance it is \(\sum g(\sigma)\)
over the cross block's singular values, with
\(g(\sigma)=s(\tfrac12[1-\sqrt{1-4\sigma^2}])\) and
\(0\le g'\le2\). Mirsky's inequality~\cite{Mirsky1960} gives
\(|S_A(C)-S_A(P)|\le\nu=\|C-P\|_1\) on the same lattice with
convergent traces; a restricted difference of rank \(r_A\) costs
at most \(r_A\log2\), without a gap or energy premise.
These are sufficient, not necessary, controls: at fixed \(ma>0\),
for a ball of radius \(La\), a cross-block bound
\(\|\Pi_A(C-P_0)\Pi_{A^c}\|_2^2=O(L)\) gives entropy difference \(O(L^{3/2}\sqrt{\log L})\) (collar truncation in Ref.~\cite{McKinneyCode2026}).
A sufficient direct-state test, in lattice units with
\(\Omega_0=|h_0|\), is \(E_f(C)=\|\Omega_0^{1/2}(fC-P_0f)\|_2^2\le KL\),
\(f\) a smooth window bounded below on the cut region with support
and derivative bounds scaled to \(L\); the massive
gap and the flat-projector commutator bound give the cross-block condition from it; its pass on the computed collapse histories is reported in Sec.~\ref{sec:results-diagnostics}.}
```

### 10.103 Sec. III C: appendix-estimates paragraph before shortening (review note 49, SHORTEN)

```latex
% assumption: at ma=0 the gap closes at the valleys and no transfer estimate covers the trapping root, so at the physical spacing (ma<1e-16) the flat coefficient eta(0) on a trapping cut, which the match of Sec. III G uses, is a condition and not a theorem.
% risk: high
% GR slippage: no
% fold into text: no; stated here ("the massless estimate's margin excludes the root") and in Sec. III G; cross-reference at the match
% provenance: g_1(0.2)~0.0577: printed in <internal>/ROADMAP_STEP4_R1_FableAPI_v1.md and TO_CODEX_task4_cut_coefficients_v2.md; entered 255ec9f93d9; script not pinned.
Appendix~\ref{app:local}'s two estimates bound a comparator's effect
on the area coefficient under a gap premise. Under their respective
hypotheses, its infinite-lattice massive and finite-lattice massless
estimates bound a comparator projector's entropy change per area by
\(C_{\rm loc}\epsilon^2(1+|\log\epsilon|)^4\) and
\(C\epsilon^{1/10}\), \(\epsilon=a/L\).
\fnd{For \(ma>0\) the frozen gap is positive below the sign-reversal threshold,
including the trapping radius. A global gap requires a pathwise
estimate: an open sign-reversed region makes \(\mathsf g_*<0\), so
Appendix~\ref{app:local}'s criterion A.5 cannot verify that
path. This does not imply an exact zero
eigenvalue on every finite lattice.} At \(ma=0.2\) that criterion is met with Appendix~\ref{app:local}'s field bounds and an independently supplied pathwise frozen-gap lower bound (values in Ref.~\cite{McKinneyCode2026}).
\fnd{The trapping-radius condition alone does not supply this global lower bound,
and these gap-criterion inputs have not been rigorously bounded on the
computed histories. The direct-state budget above uses the gap
of \(h_0\), not a gap of the evolving \(h\).}
At \(ma=0\) the frozen gap vanishes at the valleys and the
massless estimate's margin excludes the trapping radius.
Neither constant, \(C_{\rm loc}\) nor \(C\), is evaluated: the two
estimates fix the \(\epsilon\) exponents, not the sizes.
```

### 10.104 Sec. III C: reference-split comparison paragraph before shortening (review note 50, SHORTEN; the site-set statement kept here)

```latex
% provenance: 443.6 = 640 ln2; 0.0101 = 443.6/(4 pi 59.07^2); 59.07a = two_gE0 of P64_s4 in ladders_readout_v1_3.json; same row <internal>/TRAPPING_CUT_EF_R2_v1.md:200.
\fnd{The reference-split comparison bounds the trapping-cut entropy per
area in terms of these pieces. It gives
\(|S_A(C)/A_{\rm lat}-\eta(ma)|\le e_{\rm flat}+r_{\rm ref}
+\min\{\nu,r_A\log2\}/A_{\rm lat}\),
where \(e_{\rm flat}\) is the flat finite-cut remainder and \(r_{\rm ref}=|S_A(P)-S_A(P_0)|/A_{\rm lat}\) for a comparator projector \(P\), the ground projector or a stored history's fixed reference projector; the subtraction has no reference state.
For a trapping cut, \(A_{\rm lat}=4\pi(R_t/a)^2\) uses the fine-grid trapping radius \(R_t\), the zero of the fields solved on the sites before
back-averaging (Sec.~\ref{sec:star}), and the same digital site set in every
comparison, not the nominal or charge radius. If \(\mathsf Q\) is the propagated preparation
sea and \(C-\mathsf Q\) consists of \(N_B\) occupied particle orbitals, their entropy difference costs at most \(N_B\log2\), a rank term already inside the direct budget, leaving the comparison between \(\mathsf Q\) and \(P\) (values in Ref.~\cite{McKinneyCode2026}).
An excess over the held ground projector is a different reference
comparison.} 
```

### 10.105 Sec. III D: filling-bound paragraph before shortening (review note 53, SHORTEN; the thermodynamic pressure formula kept here)

```latex
The filling bounds the matter the collapse can compress. In the prescribed lattice frame \(N=1,v=0\), the
two positive-band spin states filled
to chemical potential \(\mu\) without antiparticle excitations have,
with \(\int_p=\int_{[-\pi/a,\pi/a)^3}d^3p/(2\pi)^3\),
\(f_\mu=\mathbf1_{\{\varepsilon<\mu\}}\) and
\(\varepsilon_{\max}=\sqrt{m^2+3/a^2}\), the capped density \(2\int_pf_\mu\) and
energy density \(2\int_p\varepsilon f_\mu\),
\begin{equation}
n_{\max}=\frac2{a^3},\qquad
\rho_{\max}=2\int_p\varepsilon\leq\frac{2\varepsilon_{\max}}{a^3},
\label{eq:dr-filling-bound}
\end{equation}
while for \(\mu>\varepsilon_{\max}\), the thermodynamic pressure \(p_{\rm th}=2\int_p(\mu-\varepsilon)_+=\mu n_{\max}-\rho_{\max}\)
grows at fixed filling: exclusion caps density and energy density, not pressure. It bounds the matter and, on regular cubic slices, the prepared-sea part of the central angular curvature's source:
Each bare particle energy is at most \(\varepsilon_{\max}\), and the prepared-sea normal-ordered part of \(\rho_x\), \(a^{-3}\Tr[(C-P_\star)e_x]\), is bounded by \(\bar\rho=(4/a^4)\sqrt{(ma)^2+3/2}\) for any state and preparation; the polarization moments add to it (Sec.~\ref{sec:singularity}).
```

### 10.106 Sec. III E: rotational-scope paragraph before shortening (review note 56, SHORTEN)

```latex
The laws hold on every flow, but nothing rotational is computed
here: the leading constraints admit a stationary axial perturbation
of Eq.~\eqref{eq:dr-exterior} whose far field is general relativity's
frame-dragging dipole divided by \(1+b\), Kerr's to first order at \(b=0\)~\cite{Doran2000} (Ref.~\cite{McKinneyCode2026}); coupled rotating
sources and nonspherical evolution are outside the spherical dynamics
established here, and no result below uses them.
```

### 10.107 Sec. III E: outgoing lattice branch paragraph before shortening (review note 59, SHORTEN; wavelength window and flight conditions kept here)

```latex
% assumption: the outward lattice branch and its delay are computed on a static constant-charge exterior held fixed during a flight of duration about 2 r_s ln(1/(a omega)); back-reaction and evolution of the fields during the crossing are neglected.
% risk: low
% GR slippage: no
% fold into text: no; "Holding the quasi-static constant-charge exterior throughout flight" is stated
In the point-sampled axial control with the fields held fixed at
\(N=1\), for \(z=ak\in(0,\pi/2)\) and \(u_r=\sqrt{r_s/r}\), the
massless positive-frequency outgoing carrier obeys
\begin{equation}
\begin{aligned}
a\omega&=\sin z(1-u_r\cos z),\quad \dot r=\cos z-u_r\cos(2z),\\ \dot r|_{u_r=1} &=\tfrac32z^2+O(z^4)
=\tfrac32(2a\omega)^{2/3}+O((a\omega)^{4/3}).
\end{aligned}
\label{eq:dr-crossing-ray}
\end{equation}
Thus an outgoing branch exists at every inward speed (monotonicity in Ref.~\cite{McKinneyCode2026}): the positive-frequency branch of the lattice dispersion relation at mesoscopic wavelength, whose group velocity exceeds the long-wavelength light speed by the cubic correction of Eq.~\eqref{eq:dr-speeds}.
Its wavelength is \(O((a^2/\omega)^{1/3})\) at the trapping surface,
mesoscopic when \(a\ll(a^2/\omega)^{1/3}\ll L_{\rm geom}\), but
lattice scale at fixed \(u_r>1\).
Holding the quasi-static constant-charge exterior throughout flight
between fixed \(0<r_{\rm in}/r_s<1<r_{\rm out}/r_s\), as
\(a\omega\to0\) at fixed \(m/\omega<1\), gives this branch's delay
\begin{equation}
 \Delta t_{\rm cross}=2r_s\ln\frac{1}{a\omega}+O(r_s),
 \label{eq:star-crossing-delay}
\end{equation}
a third of it inside the surface; no escape fraction follows from this point-sampled lattice control.
```

### 10.108 Sec. III F: axial temperature-correction paragraph before shortening (review note 62, SHORTEN)

```latex
On the exact exterior, axial endpoint propagating modes at fixed
\(\omega/\kappa_u\), \(m/\kappa_u\) away from threshold have
\(\delta_T=(\kappa_ua)^2[5/8+(\omega/\kappa_u)^2-\tfrac34(m/\kappa_u)^2]
+o((\kappa_ua)^2)\), formally warming if unit-flux channels match
to relative error \(o((\kappa_ua)^2)\), unproved.
Resolving it needs independent slope calibration and matching, bandwidth, reference, and endpoint errors below this order; on a full vacuum shell the matter-trapping-radius and slope averaging shift \(9(\sigma/a)^2(\kappa_Ha)^2\) exceeds the constant axial term for Sec.~\ref{sec:star}'s kernels (factors in Ref.~\cite{McKinneyCode2026}), and the histories lack that shell, so they test temperature only at order one, not this coefficient. No spherical correction is established, and it need not share the axial sign.
```

### 10.109 Sec. III G: massive-coefficient paragraph before shortening (review note 66, SHORTEN; the half-space entropy singularity and the fifteen-direction mean kept here)

```latex
At fixed physical \(m\), the match uses the flat sea's massless coefficient and its interval bound: between momenta \(m\) and \(a^{-1}\) each
of the eight valleys is a continuum Dirac field with half-space
entropy singular as \(-\tfrac16\log(m^2+k_\perp^2)\), \(k_\perp\)
the transverse momentum, so every planar cut has
\(\eta(\zeta)-\eta(0)=\frac{\zeta^2}{3\pi}\log\zeta^2+b_{\boldsymbol n}\zeta^2+o(\zeta^2)\),
with directional finite parts \(b_{\boldsymbol n}\) bounded or fitted between \(-0.42\) and \(-0.46\) and, if the remainder is
uniformly \(o(\zeta^2)\) over normals (a uniformity not proved
here), a fifteen-direction sphere mean near \(-0.46\) without a
bound, entering only the \(10^{-32}\)-level spacing shift that
follows (values in Ref.~\cite{McKinneyCode2026}). \fnd{At the physical spacing, \(ma<10^{-16}\) for every known
elementary fermion. If the angular expansion holds, the estimated
coefficient change is about \(-8\times10^{-32}\), moving the
spacing by about \(-5\times10^{-32}\) of itself.
This estimate uses the massless match; its angular remainder
still needs a bound.}
```

### 10.110 Sec. IV A: maximum-mass threshold paragraph before shortening (review note 70, SHORTEN)

```latex
% assumption: hydrostatic equilibrium needs the source conservation (Euler) equation together with the constraint rows, i.e. the evolution rows' hypothesis, and the normalization is related to the Oppenheimer-Volkoff result of general relativity by a degeneracy rescaling without independent check.
% risk: medium
% GR slippage: possible (Oppenheimer-Volkoff is a GR result used for the threshold benchmark)
% fold into text: no; "a numerical cross-check is not established here" is stated
% provenance: (c_E,c_R)=(0.135858,1.18689): <internal>/strand5_star_v3.py (md5 b19b7304) -> <internal>/star_v3.json; refined in <internal>/RESULT_J4_items12_v1.md:51.
% provenance: 0.229, 4.71, 4.35, 1.84, 1.9 are arithmetic on c_E, c_R, eta_E = 59.0689/64. sqrt8 OV check: no producer.
The threshold the collapsing clouds exceed is the maximum mass of the
cold continuum family: the fermion as a cold degenerate isotropic fluid with 16 states per momentum, in
hydrostatic equilibrium by the rows \eqref{eq:dr-radial} with a regular
centre and vacuum matching, has its first mass maximum, \(E_{\max}\), at
\begin{equation}
\begin{aligned}
E&=c_E/(G^{3/2}m^2),&Ea&=c_E/(g^{3/2}\zeta^2),\\
R&=c_R/(\sqrt G\,m^2),&R/a&=c_R/(\sqrt g\,\zeta^2),
\end{aligned}
\label{eq:sc-mass-radius}
\end{equation}
with \(\zeta=ma\), \(g=G/a^2\), and \((c_E,c_R)=(0.135858,1.18689)\).
The degeneracy scaling relates this normalization to the two-state
Oppenheimer-Volkoff~\cite{OppenheimerVolkoff1939} values by division
by \(\sqrt8\), a relation quoted, not cross-checked numerically.
Below, the count sets \(mN_B/E_{\max}=\zeta^2(r_s/a)\sqrt g/(2c_E)\),
\(r_s=2GmN_B\) nominal, 4.71 at the nominal
\((\zeta,r_s/a)=(0.2,64)\), and the charge
\(E/E_{\max}=\eta_E\,mN_B/E_{\max}\), 4.35 at the collapse's own
charge with the binding fraction \(\eta_E=E/(mN_B)=0.923\) at release
(Sec.~\ref{sec:scaling}): a comparison of \(mN_B\) with
\(E_{\max}\), not an identification of \(E_{\max}/m\) with the
family's maximum count, whose binding fraction is not computed
(compactness and size criterion in Ref.~\cite{McKinneyCode2026}).
```

### 10.111 Sec. IV A: angular-channel chain paragraph before shortening and splitting (review note 74, SHORTEN; the chain operator definitions and the polarization-counting remark are kept here)

```latex
% assumption: every collapse number is computed on radial angular-channel chains with copy weights 16k, a different lattice model from the declared cubic fermion and related to it only through the continuum spherical limit; the chain lift is not validated against the cubic operator (Sec. V D) and the chain subtraction is uncertified below about 2/m.
% risk: high
% GR slippage: no
% fold into text: no; stated here; recommend one clause in the Introduction's roadmap ("computed on a spherical chain reduction")
The collapse is computed on angular-channel chains, the lattice description the dynamics uses (the static and entanglement results of Sec.~\ref{sec:results} use the full cubic lattice): in spherical symmetry the cubic-lattice problem separates into one radial chain per angular-momentum channel, a partial-wave decomposition; the chains share the cubic operator's continuum spherical
limit but not its zone, and every reading below is tested against
step, averaging-width, and channel-count controls. Channel \(k\) has a two-component spinor at
\(r_n=na\), \(n=1,\dots,n_{\rm ch}\), shift \((Tu)_n=u_{n+1}\),
no origin-crossing hop, and an open last site. With radial momentum \(\widehat p_r=(T-T^\dagger)/(2ia)\),
operator \(h_{0,k}=m\sigma_3+\sigma_1\widehat p_r+\sigma_2k/r_n\) and spin-scalar current \(Q_r=(T^2-T^{\dagger2})/(4ia)\), the assigned densities couple to back-averaged
radial fields. One chain per channel carries the eight valleys' magnetic weight
\(16k\); its second radial valley carries \(-k\), filled valleys
uncounted. Their polarization counting differs from the cubic operator's
(Ref.~\cite{McKinneyCode2026}). For \(m>0\) and channels \(k\le3\),
the chain subtraction is not verified below about \(2/m\), \(10a\) at
\(ma=0.2\), a working exclusion scale, not a remainder bound.
% provenance: massive-chain outer endpoint cutoff tests and acceptance rule: <internal>/AstraAPI_boutcmp2_v1.md (md5 a1c9f8ef) and AstraAPI_boutpred_v1.md (md5 19cbef01); lead-routed inbox 62 and 63.
For the massive chain's outer endpoint, the stored cutoff sums pass
every prescribed test on their changes between successive cutoffs at
\(10^{-6}\), with propagated numerical error bounds below
\(10^{-8}\), and every comparison supports a fixed-radius subtraction
rather than an omitted gradient; no outer endpoint coefficient of the
massive chain is established, because for every one of the 40
coefficients (eight at each of five channels) the change from cutoff
512 to 1024 exceeds the change from 256 to 512, although every change
is below \(10^{-6}\) and the numerical bounds are below \(10^{-8}\)
(Ref.~\cite{McKinneyCode2026}). At \(m=0\), \(N=1\), \(v=0\), the site-flow kernel \(K=\tfrac12E_{vv}\) has exact range three, \(K_{rs}=0\) for
\(|r-s|>3\), and for every \(k=1,\ldots,5\), separately on even and
odd rows, the verified limits are
\(c_0=\lim_r\lim_{L_c}r^2(K\mathbf 1)_r=0\) and
\(c_1=\lim_r\lim_{L_c}r(Kx)_r=0\), \(x_n=n-1\), parent removal
preceding the row limit; with \(\alpha=1/(12\pi)\) the constant-coefficient endpoint rule gives, for the massless tail's endpoint coefficients \(e_{11}\) and \(e_{01}\) (Ref.~\cite{McKinneyCode2026}), \(a_{11}=c_1-\alpha=-\alpha\) and
\(b_{e01}=(c_1+c_0)/2=0\), so \(e_{11}\) diverges and \(e_{01}\) is
undecided, with no cancellation established: this constant-coefficient light-chain prescription has no finite massless endpoint. The 80 finite-parent interval rows at \(L_c=4096\), 4097, 8192, 8193
and \(r=32\), 33, 64, 65, from two independent numerical
calculations, are implementation cross-checks, not coefficient bounds, the first calculation's finite-lattice values now
verified at the target tolerance \(\alpha/10\) on all 40 chains
(\(k=1,\ldots,5\), lattices \(2^{12}\) to \(2^{15}\) with partners),
falling from about \(10^{-3}\) at \(r=32\) to about \(10^{-5}\) at the
window end: finite lattice, not a limit.
```

### 10.112 Sec. IV B: later-trapping-episodes paragraph (review note 83, CUT; closing times 17.375 and 16.375 kept here)

```latex
Later trapping episodes (intervals with a trapping radius) depend on width and step. The reference's first episode closes at 17.375, with a zero at every stored slice over 18.375 to 40;
\(\sigma/a=8\) closes at 16.375, with only two later intervals with a trapping radius.
Neither held-source nor widened
completion reproduces the second episode; the late trapping radius and its
fraction of \(2GE\) depend on interior completion.
```

### 10.113 Sec. IV B: interior-readings paragraph before shortening (review note 84, SHORTEN)

```latex
% provenance: collar charge 0.847/0.919/0.975: <internal>/SCALING_ladder_v1.3.md:198; channel control <internal>/W5_W4_formation_attribution_v1.md sec. 3, attrib_readout_v1.json.
% provenance: W5 members kappa_sea 10/14/20, tau 8 and 4, nsites 2048, ell 4, to 14 (<internal>/lad_W5_*). Centroid minima line not pinned.
The interior readings depend on the width at order one while
formation is robust under the channel count, so no closing time,
lifetime, late-time law, receiver-time redshift law, or second
surface is established. Centroid minima move inward with width (centroid motion, not interior rebound), and the shell charge at 16 shows a held-step deficit that halves with step length (value in Ref.~\cite{McKinneyCode2026}), a held-step effect, not a charge-loss law. Increasing the channel count from 10
to 20 leaves formation's slice unchanged and its radius within
\(0.09a\), with twenty-channel readings at 14 unverified and the
weighted sea pair energy unconverged. Conserving-step depth and
centroid, and all three under more channels, remain open; formation
itself is verified at all three widths, and nothing below uses the
depth or centroid values (run values in Ref.~\cite{McKinneyCode2026}).
```

### 10.114 Sec. IV B: redshift-definition paragraph before shortening (review note 87, SHORTEN)

```latex
% assumption: the received redshift uses stationary fields at emission and a static receiver at 4 r_s, with no transfer to infinity.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: redshift: <internal>/REDSHIFT_received_v4.1.md + REDSHIFT_received_v4.1_scripts/ (rz_v4_head_v3.py, rz_v4_variants_v4.py, rz_v3_law_v2.py, rz_v4_dispev_tables_v3.py).
% provenance: per-number map <internal>/SOURCES.md (H1: P64_s4 L_D 10.338, P64_s2 15.569; H2: T_e 10.5356, r_e 19.057a).
% provenance: registry <internal>/registry_v1.json (md5 efcfad78); receiver 4 r_s; RK4 h=0.5a, DOP853 rtol 1e-11 check.
For outward continuum light from a comoving emitter, stationary
fields give \(g_\infty=\omega_\infty/\omega_{\rm em}=N+v\) at
emission; the computed values instead use
\(g_D=\omega_D/\omega_{\rm em}\) at a static receiver at \(4r_s\)
(subscript \(D\) for receiver quantities),
\(\ln(1/g_D)=-\ln(d\tau_{\rm em}/dT_D)+\ln\sqrt{N_D^2-v_D^2}\), with
no transfer to infinity, back-averaged fields, clocks comoving from
\(T=0\), and launches beside the trapping radius. The history tracks no
material surface and its trapping radius stays below \(2GE\), so
Eq.~\eqref{eq:star-peeling-control} describes none of its stored slices.
```

### 10.115 Sec. IV B: late-redshift paragraph before shortening (review note 89, SHORTEN)

```latex
Late redshift changes at order one with width, step, completion, and
propagation background; clock shifts reach 2.3 for fine-grid fields at
headline events and fixed arrival, and the field sensitivity at the
verification ceiling is not an error bound (values in Ref.~\cite{McKinneyCode2026}).
On seven of ten reference-collapse clocks, hovering redshift tracks \(\int\kappa_Rdt\) with slope 0.85--1.25 (median slopes in Ref.~\cite{McKinneyCode2026}), and blueshifted arrivals at widths at least 8 at both sizes and both completions rule out a uniform sampled redshift bound.
```

### 10.116 Sec. IV B: continuum-map-versus-lattice-ray paragraph (review note 90, CUT; the ak = 0.05 to 0.3 crossover remark kept here)

```latex
These continuum-map values are not spectra: same-event lattice rays at \(ak=0.05\)--0.3 arrive earlier and, on the four most redshifted histories, less redshifted, \(\ln(1/g_D)\) near the crossover scale \(\ln[2/(ak)^2]\), which is not a bound, some exceeding it, and after early closure of the trapping radius dispersion can instead increase redshift (values in Ref.~\cite{McKinneyCode2026}); no packet or finite-band spectrum is computed.
```

### 10.117 Sec. IV C: interior-definitions paragraph before shortening (review note 92, SHORTEN; the explicit E_H constant and the Gaussian pair expectation kept here)

```latex
Inside the surface the computation runs past the hydrodynamic regime; this subsection reads the interior to the limit of its description. Let \(A_\mu\) be the energy and current vertices and \(V_{\mu\nu}\)
the Hessian of \(E[C]\) in bare moments \(\Tr(A_\mu C)\) at
\(\bar C\), including reference response. Set \(\Delta_\star=C-P_\star\), the Wick difference against the prepared sea (the source \(\Tr(A_\mu C)-r_\mu\) is \(\Tr(A_\mu\Delta_\star)\) less the polarization moment), \(\bar\Delta_\star=\bar C-P_\star\),
\(h_H[X]=\sum_{\mu\nu}V_{\mu\nu}A_\mu\Tr(A_\nu X)\) and
\(\widehat V_\star=\tfrac12\sum_{\mu\nu}V_{\mu\nu}
:(c^\dagger A_\mu c)(c^\dagger A_\nu c):_{P_\star}\).
With \(\mathcal E_H=E[\bar C]-\Tr(h[\bar C]\bar\Delta_\star)
+\tfrac12\sum_{\mu\nu}V_{\mu\nu}\Tr(A_\mu\bar\Delta_\star)\Tr(A_\nu\bar\Delta_\star)\),
the reduced second-order Hamiltonian is
\(\widehat E_2=\mathcal E_H
+:c^\dagger\bigl(h[\bar C]-h_H[\bar\Delta_\star]\bigr)c:_{P_\star}+\widehat V_\star\),
\(h[\bar C]\) using \(\bar C\)'s fields. Its number-conserving Gaussian
pair expectation is
\(\tfrac12\sum_{\mu\nu}V_{\mu\nu}[\Tr(A_\mu\Delta_\star)\Tr(A_\nu\Delta_\star)
-\Tr(A_\mu\Delta_\star A_\nu\Delta_\star)]\).
These Hartree and Fock terms cancel for
\(\Delta_\star=|u\rangle\langle u|\), \(P_\star u=0\): no one-particle pair
self-attraction.
```

### 10.118 Sec. IV C: second-order comparator energy paragraph before shortening (review note 93, SHORTEN; the explicit single and double sums kept here)

```latex
% provenance: |dE_ref|/E 8.06e-4, 4.21e-3 and dressing 0.136, 1.095: <internal>/VREF_R2_solve_v1.md (<internal>/vref_r2) and <internal>/VREF_R1_v1.md (route 1, blind).
% provenance: slices 11.0 (rebuilt from W5 record 88) and 15.0 (B2n4096, rebuilt from record 120); spec <internal>/AstraAPI_vref_v1.md; printed in ad1228fec97.
For \(H_0=:c^\dagger h[\bar C]c:_{P_\star}\), \(B=-h_H[\bar\Delta_\star]\), and
\(V=:c^\dagger Bc:_{P_\star}+\widehat V_\star\), full Wick ordering makes the first-order expectation in \(P_\star\) zero. With the ground projector \(P_g\) of \(h[\bar C]\) in place of \(P_\star\), a ground comparator, on a gapped path the one-body and quartic terms create orthogonal single and double particle-hole states. With antisymmetrized quartic elements \(T_{\alpha\beta;ij}\), \(i,j\) occupied and \(\alpha,\beta\) empty, the connected comparator energy \(\Delta E_{\rm ref}^{(2)}\), apart from \(\mathcal E_H\), is minus the sum of \(|B_{\alpha i}|^2\) over single excitations and of \(|T_{\alpha\beta;ij}|^2\) over double excitations, each divided by its excitation energy:
nonpositive, strictly negative if either numerator is nonzero, the
first sum absent only if \(P_gB(I-P_g)=0\) (order counting in \(G\)
in Ref.~\cite{McKinneyCode2026}).
```

### 10.119 Sec. IV C: stationary-dressing paragraph (review note 94, CUT)

```latex
Stationary dressing about the fixed-occupation determinant of the
slice Hamiltonian nearest the stored covariance in the sea-plus-dust
number sector, where the Hartree reordering terms cancel exactly,
gives a nonpositive connected second-order energy for positive
denominators (contraction mechanism in Ref.~\cite{McKinneyCode2026});
these are retained-order diagnostics, not controlled dressed
energies, the earlier fixed-reference-projector values evaluate a different
comparator, and no dressing energy is assigned to the subtraction.
```

### 10.120 Sec. IV C: outward-mode sorting paragraph before shortening (review note 98, SHORTEN)

```latex
% assumption: the Killing split uses stationary fields; on the evolving collapse Killing energy is not conserved and the bins are read against each slice's own fields.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: 4.96/a^4 = 4 sqrt((ma)^2+3/2) at ma 0.2; b_r/m = sqrt(1+25); R_90 40.0a->499a, burst 58.4: <internal>/ASYMPTOTICS_checks_v1.9.md series and CORE_well_v1.2 (script not pinned).
Equation~\eqref{eq:dr-comoving} sends modes at
\(\pi/4<ap<\pi/2\) outward at every inward flow speed, not the
whole filling; a receding trapping radius provides a second exit.
In stationary fields, positive-branch Killing energy
\(N(\varepsilon-|w|q_r)\) is conserved. Beyond the fountain
(outward-thrown matter that falls back), stationary fields sort content at 0, \(m\), and the radial band top \(b_r=\sqrt{m^2+a^{-2}}\), the Killing split below.
Propagation in \((m,b_r)\) is necessary, not sufficient, for
reaching infinity. Above \(b_r\) and between 0 and \(m\) content
is bound, the latter fountaining; below 0 it turns inside the trapping radius.
```

### 10.121 Sec. IV C: stored-slice dressing paragraph before shortening (review note 100, SHORTEN)

```latex
On the reference collapse's stored
slices at \(T/r_s=11.0\) and 15.0, two independent calculations
expanding about each slice's Hartree covariance \(\bar C\), ordered
against the stored history's fixed reference projector as comparator, find
\(|\Delta E_{\rm ref}^{(2)}|/E\) below \(5\times10^{-3}\) but a
squared first-order dressing norm of 0.14 and 1.1, almost all from
the \(B\) singles, which nothing balances on the matter-free
comparator; the expansion about it has no small-dressing control at
15.0, is not stationary under \(\widehat E_2\), and its dressed
state's branch weight is untested (values in
Ref.~\cite{McKinneyCode2026}).
```

### 10.122 Sec. IV C: Hartree-Fock-level paragraph before shortening and splitting (review note 105, SHORTEN; the pair-kernel exchange remark and the sensitivity radii (3.46-4.96)e-3 and (3.81-5.11)e-2 kept here)

```latex
% provenance: exchange 0.14-0.16% and variance: <internal>/core_hf_v1.py + <internal>/route2 (route2_exchange.py, route2_variance.py); <internal>/ROUTE2_core_B29_v1.md; commit 1a297312728.
% provenance: <B> 0.0798 at 15.0: <internal>/VREF_R1_v1.md:27; route 2 <internal>/VREF_R2_solve_v1.md gives 0.0817 (reconciliation doc not located).
% provenance: Fock ratios 7.1e-3/1.5e-3/3.9e-3, 9.0e-3/1.30e-2: <internal>/VDRESS_RESIDUAL_solve_v1.2.md (<internal>/*.json) and <internal>/VDRESS_RESIDUAL_R1_v1.1.md.
The interior's Hartree-Fock level is read on 18 stored slices of the \(\sigma/a=4\) and 8 runs past the limit of the hydrodynamic description, in two independent calculations on four. Every operator of this paragraph is a held-source operator, built from the fixed reference projector's response (the comparator projector of Sec.~\ref{sec:results-lattice}) instead of Eq.~\eqref{eq:dm-reference}'s reference response \todo{rebuild the boundary-charge derivatives, masses, branch projector, pair kernel, and Fock corrections with \(D\mathcal W\) and ordering against \(P_\star\), and redo these readings}. The pair-kernel exchange, about \(1/N_B\) of the pair energy, and the linearized mass observable's fluctuation, below \(\mathcal N_<^{-1/2}\), are in Ref.~\cite{McKinneyCode2026}; with indefinite kernels neither is a bound. The held-source second-order masses \(\widehat M_j\) at \(R_{50}\) and the trapping radii are diagnostics (Sec.~\ref{sec:model-operators}). \fnd{For the interior deviation \(4\sum_j(\widehat M_j/M_j-I)^2\), two independent calculations give \(\langle\mathcal B\rangle=0.0798\) and 0.0817 at 15.0 on the ten channels, so the bound \(\langle\mathcal B\rangle\le0.080\) on all 18 slices holds in one calculation only; both are conditional on the bound for the computed Hessian's error, proved only for errors parallel to the computed Hessian. The larger value implies overlap at least 0.918 with that test subspace, not the branch, and probability at most 0.021 for each mass to be nonpositive. \todo{prove the bound for the computed Hessian's error, including errors not parallel to the computed Hessian, and propagate it on two independent calculations to branch weights and mass biases on all 18 slices}} The branch weight does not inherit 0.918 \todo{two-calculation \(4\langle(\widehat E_2/E-I)^2\rangle\) on the 18 slices and at 11.0}
and the correlation beyond Hartree-Fock is uncontrolled. For the boundary charge itself, \(q_\infty=4\langle(\widehat E_2/E-I)^2\rangle\), two independent calculations with independent full-tail comparators give matched nominal upper endpoints from \(2.93\times10^{-5}\) to \(9.27\times10^{-4}\) on five ten-channel stored slices; single-calculation sensitivity estimates, not verified robustness ranges, place the 0.080 and 1 crossings at relative perturbation radii \((3.46\text{--}4.96)\times10^{-3}\) and \((3.81\text{--}5.11)\times10^{-2}\); a validated \(q_\infty\le0.080\) would give branch weight at least 0.92, but neither actual-error inclusion nor the declared-reference branch calculation is established, and the five slices cover neither all 18 nor \(T/r_s=11.0\). In two independent calculations with \(\widehat E_2\) Wick ordered about the evolved Hartree state, the Fock single amplitude and the relative Fock correction to the mass rates stay below \(10^{-2}\) of their Hartree counterparts on ten channels at \(T/r_s=11.0\) and 15.0; the single ratio grows with channel count through the propagated sea's exchange mismatch with the fixed reference projector, faster than the copy weights \(16k\), so no channel limit is established on the chains (values in Ref.~\cite{McKinneyCode2026})
\todo{two-calculation sea-reference exchange singles on a cubic lattice}.
On the cubic lattice the sea variance obeys an area law (relative enclosed-charge fluctuation \(\sigma^{-1/2}\) at leading order in \(a/\sigma\), outside the hydrodynamic premise); on the chains it grows quadratically with the channel cutoff.
```

### 10.123 Sec. IV D: stored-clock and Jacobian paragraph before shortening (review note 111, SHORTEN)

```latex
Stored time slices show positive clocks and nonsingular implemented Jacobians at every sample: the reference collapse's stored clocks stay positive through \(T/r_s=40\) (sampled minimum 0.068); two reported independent no-density-floor current--mass scans of 894 stored slices in four histories give a clock floor \(e^{-50}\), far below the sampled minima of 0.07--0.11, with normalization and centre coverage unspecified; and the implemented constraint Jacobian has verified inverse norm at most 2.90, with bounded rounding and assembly errors, on a separately identified set of 573 stored slices in three histories. The scans test clock conditions on their slices and the inverse checks their own set, establishing neither exact trapping radii nor the intervening trajectory.
The reference collapse uses a fixed reference projector without light, other
histories different light rules, one with the law's block-form light counterterm, whose constant-coefficient massless-chain endpoint fails through \(e_{11}\) while \(e_{01}\) remains undecided (Sec.~\ref{sec:results-lattice}); these tests verify no trajectory under Eq.~\eqref{eq:dm-reference}. The stored-slice agreement of
Sec.~\ref{sec:results-lattice} remains a consistency check.
```

### 10.124 Sec. IV D: between-slice bound paragraph before shortening (review note 113, SHORTEN; the four conditions (i)-(iv) in full are kept here)

```latex
What remains is a between-slice bound. No regular case is known where the coupled evolution fails to continue, and no
invariant regular domain is proved; continuation for all regular data is open, and the deciding two-field re-evolution is in
progress. The deciding calculation is a validated rigorous bound on the coupled trajectory on every consecutive slice interval: with \(F(C,u)=0\),
\(u=(N,v)\), the boundary-normalized constraints with the stated
averaging and full reference response, the bound must satisfy
\(i\dot C=[\bar h(u),C]\) and \(F_u\dot u=-F_C\dot C\), connect to
the next interval, and meet four sufficient conditions: (i) a solution rigorously bounded under Eq.~\eqref{eq:dm-reference} with the
propagated sea, every active species and copy weight, and all
initial-data, assembly, quadrature, and evolution errors, defects and
spectra bounded consistently; (ii) throughout \([t_j,t_{j+1}]\) a
regular inward branch with a positive margin
\(\mathcal M(r,t)/r^3\ge\mu_{*,j}>0\) for \(0<r\le L\) and
Eq.~\eqref{eq:sg-current-mass} verified on the same bound, every
tail and low-density cell included, with the full constraint mass,
not enclosed rest mass; (iii) in fixed, printed coordinates and
norms, a uniform inverse bound on the full \(F_u\), reference
response included, which neither a clock bound nor the inverses verified at stored time slices supplies; and (iv) a self-closing rigorous bound on the fields, required spatial regularity, and reference
derivatives throughout the interval that continues the same branch
(centre handling, approximate-inverse recipe, and discrete-row clock
estimate in Ref.~\cite{McKinneyCode2026}).
The positive margin in (ii) selects a sufficient nondegenerate
domain; regular degenerate centres and mass-zero vacuum components
need separate matching and uniqueness arguments. Failure of a
sufficient inequality means not verified, not singular.
Stored fixed-reference-projector histories may seed a verification only with rigorously bounded defects under the printed law; otherwise re-evolution is
required. Interpolating stored fields and propagating test matter
is not this calculation. No complete between-slice verification
under the printed law, or general bound on polarization moments
over regular cubic slices, is established.
The leading laws supply no lattice-scale geometric law once
their gradient expansion fails, so a Pauli-supported interior
remains a possible outcome, not an assumption that closes the
continuation question.
```

### 10.125 Sec. IV E: first-law and stress diagnostics paragraph before shortening (review note 114, SHORTEN; the inbox 61 sentence and the todo unchanged in the paper)

```latex
% assumption: the first-law and evolution-row diagnostics use the held-projector sources and a stress of unclosed provenance; agreement or disagreement with the no-work first law is not established by them.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: first-law sums -18.36/-6.51/-11.93 over 224 pairs (sigma/a 2, P64_s2): <internal>/FIRSTLAW_records_v2.2.md (l.55, 233); <internal>/firstlaw_v2, <internal>/firstlaw (function not pinned).
% provenance: row-residual medians 0.17/0.04/0.06 per the stress lineage (FABLE_write_stress_v2.1); landed 2cb9ac21f64, 21e8dd01028.
Finally the computed geometry is checked against the first law and the stress it should obey. On the stored time slices the terms of the first law do not add up to zero, and the stress does not meet its expected balance either; neither check comes with an error bound, so both are indications, not tests.
On the \(\sigma=2a\) run's 224 consecutive pairs of stored slices with a simple fine-grid trapping radius (to \(T/r_s=40\), fine-grid fields in finite-shell form, stored sources), the terms of Eq.~\eqref{eq:page-first-law-rows} sum to a residual of 0.08 lattice units against terms of order 10 (values in Ref.~\cite{McKinneyCode2026}), the work term above its numerical floor in 221 pairs, these stored slices lacking a vacuum exterior beyond the trapping radius. The stored \(\delta E\) carries the held step's charge deficit, up to 0.7 percent of \(E\) per slice near formation (fixed-scheme control in Ref.~\cite{McKinneyCode2026}), comparable to
these terms, so the diagnostic does not by itself exclude the no-work
form. With the stress's identity verified only on sampled slices (below) and normalization, clock, and between-node sign unclosed, between stored slices the mass inside each fixed outer trapping radius changes at
a rate differing from the evolution row's, with the stored
kernel-averaged stress, by medians of a few to 20 percent of the
row's largest term at the three widths and by up to 0.74 at inner trapping radii
above \(2\sigma\), outside the hydrodynamic description (medians in
Ref.~\cite{McKinneyCode2026}).
Independent reconstructions support identifying the stored stress as
\(S^{\rm op}[C-P_{\rm held}]\) on the shared stored time slices
\((r_s/a,\sigma/a,T/r_s)=(64,4,17)\), \((64,2,17)\), \((64,8,14)\),
and the checked \((128,8,30)\) anchor; completed four-point matched
diagnostics show small differences between the two independent
calculations in prepared-operator and kinetic residuals on all three
\(r_s/a=64\) slices but an unresolved prepared-reference and
projection discrepancy at the anchor, while validated joint
kinetic/reference agreement and complete size, width, and step
controls remain unestablished. So identified, with \(P_{\rm held}\) the fixed reference projector
of Sec.~\ref{sec:results-lattice}, these are held-comparator
operator-stress residuals, not those of the kinetic \(S_{ij}\) of
Sec.~\ref{sec:model-gravity} with Eq.~\eqref{eq:dm-reference}'s
sources, and the split of Sec.~\ref{sec:model-gravity} remains
unclosed. A polarization coefficient needs two independent cubic-lattice calculations with its definition, sign, normalization, invariants, and error, beside the chains' separately normalized value.
\todo{matched kinetic and reference residuals on identical stored slices,
separated terms and residual identities before medians; the identification failures at \(r_s/a=32\) (\(T/r_s=12\), 14, 20) and
\(\sigma/a=2\) (12, 14, 16); size, width, and step controls;
independent polarization-coefficient audit}
```

### 10.126 Sec. V A: direct heavy-light coupling (toy converter) paragraph (review note 117, CUT)

```latex
A direct heavy-light coupling, absent from every history, is defined
for later reference: an on-site converter \(H=H_0+UV\),
\(U\in\mathbb R\), bilinear in light and heavy pair operators (its
matrices are in Ref.~\cite{McKinneyCode2026}). The same clock and
flow couple to its site energy and assigned
energy-dipole current with equally divided bond corrections,
\(\mathcal Q_i=\mathcal Q_{0,i}+U\mathcal Q_{1,i}\), the free part
summed over species; disjoint on-site terms commute, so there is no
quadratic correction. The free charge \(c^\dagger Q_ic\) commutes
with \(h_0\), the interacting current's rate has nonzero linear and
quadratic terms, and at \(ma=3/5\) a free resonance has \(E_f=E_i\)
but \((q_f-q_i)V_{fi}\ne0\), obstructing every finite-range
correction to this charge differentiable at zero coupling, with
range and coefficients independent of lattice volume and conservation
through first order on all sufficiently large periodic lattices
(Ref.~\cite{McKinneyCode2026}); its
identification with displacement momentum needs deformation-response
matching (Sec.~\ref{sec:model-sources}).
```

### 10.127 Sec. V A: coupled-species momentum balance paragraph (review note 118, CUT)

```latex
The coupled species has an exact lattice balance for its assigned
momentum density under the unweighted \(H\), with summed source the
force \(i[H,\mathcal Q_i]\); the explicit balance with its
field-weighting and reference-subtraction terms is given in
Ref.~\cite{McKinneyCode2026}. If its hydrodynamic source
contribution \(X\) satisfies \(N\epsilon_E^X+v\epsilon_M^X=0\), an
additional force-work condition, then, since
\((r^2vR_t)'=4\pi Gr^2N(N\epsilon_E+v\epsilon_M)\)
(Sec.~\ref{sec:model-gravity}), a regular centre where the rows
hold, or zero enclosed-charge rate response below the support, gives
\(R_t^X=0\) and \(R_r^X=-4\pi GNr\epsilon_M^X\): on a regular branch the interacting sources fix the fields through the rows at
each time, and the balance identities give the evolution residual,
reference response included. Clock and flow store no momentum to
absorb this force; it must be carried or its neglect bounded, and
its referenced source contribution, interacting sea, and response
included, is uncomputed.
```

### 10.128 Sec. V C: four-light-field-histories caveat (review note 129, CUT)

```latex
% provenance: completion light runs v6o_W3fs/v6o_W3pa k1, k2 (8192 sites, to 40) <internal>/runs; readout <internal>/PAGE_solved_history_v4.1.md (T4); 2.5e-4 t_* arithmetic.
The four light-field histories agree only at formation (12.125, radii within \(1.0a\)), so later readings carry no cross-history control: solver, charge, and matter-chain length differ, core treatment is not isolated, and the completions end at 40, before the dust readings at 48--56, leaving no shell-flux or cross-trapping-radius transfer comparison (Ref.~\cite{McKinneyCode2026}).
```

### 10.129 Sec. V E: width-control caveat (review note 140, CUT)

```latex
Width control remains partial, the light flux, evolved-cut excess, and interior using two scales on different objects and the cross-surface block and first-law slices one; area-regime transfer is uncomputed.
```

### 10.130 Sec. V A: radiating-species paragraph before shortening (review note 116, SHORTEN)

```latex
% assumption: adding the light field is a model extension beyond the five declarations; its vacuum subtraction uses the fourth declaration's gapped band expansion at m=0, where that expansion is uncontrolled (Sec. IV D), and its pocket threshold coincides with the trapping root, so inside the surface its prepared band is not the ground band.
% risk: high
% GR slippage: no
% fold into text: no; "its m=0 expansion uncontrolled" is stated in Sec. IV D; cross-reference here
% provenance: a = 2.71 l_P = sqrt(8 eta(0)) l_P with eta(0)=0.9191 (two massless ball cuts); m/T_H bound analytic.
Radiation needs a species that can leave: this section adds a massless light field and reads it on the computed collapse. Thermal radiation of mass \(m\) at \(T_H\) reaches infinity with
factor \(e^{-m/T_H}=e^{-4\pi mr_s}\). Above the continuum maximum,
\(m/T_H\ge8\pi c_Em_P/m\), \(m_P=G^{-1/2}\): the forming species
is thermally dark; a massless rate needs \(4\pi mr_s\ll1\).
For radiation alone we add one massless four-component Dirac field, the light field, with the same operator, clock, and flow; the massive fermion is then the heavy field. Each is normal ordered against its own prepared sea; formally, their polarization-subtracted sources add in Eq.~\eqref{eq:dr-radial}, and the extension enters nowhere else. This declaration supplies no finite massless-chain endpoint: the constant-coefficient \(e_{11}\) diverges and \(e_{01}\) establishes no cancellation (Sec.~\ref{sec:results-lattice}); finite-parent light calculations retain their stated scope.
On the regular positive-mass trace domain the two fields' energy brackets, remainders, and reference central terms add and close on the total current; the light field's sign-reversed states open at the matter's trapping radius, and exclusion bounds its normal-ordered energy below on a finite lattice at fixed fields.
```

### 10.131 Sec. V B: held-field unitarity and purity bounds before shortening (review note 121, SHORTEN)

```latex
% assumption: the held-field results need stationary one-ended fields, incoming vacuum and absolutely continuous non-bound spectrum.
% risk: low
% GR slippage: no
% fold into text: no; stated
For held stationary regular one-ended fields with incoming vacuum,
scattering unitarity on the absolutely continuous spectrum gives
outgoing occupation \(\Theta(-\omega)\), zero stationary flux, and a
once-counted positive outgoing-energy budget at most
\(E_{\rm inv}=\Tr[h(C_0-P_-)]\); if all non-bound spectrum is
absolutely continuous, the continuum part of a compact deviation
occupying finitely many bound levels leaves every finite site set,
its bound block and phases remaining, and purity bounds the exterior
excess's upper limit by \(\ln2\) per occupied bound level, even
without convergence. Only occupied deviation levels count (the
appended exterior's band-edge bound-level series hold constant
occupations); no drain rate follows.
```

### 10.132 Sec. V B: evolving-field caveats before shortening (review note 122, SHORTEN)

```latex
On evolving Hartree fields, \(\Tr[\dot h(C-P_-)]\) is work, not
production, and a luminosity also needs reference transport, storage,
and interface flux; exclusion bounds outgoing occupations by one (by
\(\Gamma(\omega)\) only for reflection-independent ports), ray-map
conversion bounds no driven sidebands, a weak modulation's transition
count is bounded by the squared time integral of the interband norm,
and a preformed excitation can drain at rate \(L\) for
\(E_{\rm inv}/L\), so finite late flux alone proves no sustained
supply.
```

### 10.133 Sec. V C: conformal-companion paragraph before shortening (review note 126, SHORTEN)

```latex
% assumption: the conformal companion applies the 1+1 conformal-field-theory entropy formula on the derived background after dropping the centrifugal term and unfolding the centre; it is a scaling description, not the chain's exact entropy.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: conformal control v6s_B2_kz (kappa_zero) <internal>/v6s_B2_kz_k1_d13.json (md5 8db1de11), 8192 sites, k=1, to 100.
% provenance: 4e-4, 6e-3, 1.28-1.40: <internal>/NOTES_D1_conformal_control_v1.2.md (l.27, 41, 48); L_c = 265 r_s is the ray-circle label.
A conformal companion reproduces the \(2r_s\) formation excess until
hard formation light arrives: one chiral Dirac field on a ray circle
of circumference \(265r_s\), obtained by removing the massless
channel's centrifugal term on the same fields and unfolding the
regular centre, with cut entropy the arc entropy between outgoing and
incoming entry labels (exact for the antiperiodic conformal vacuum
with fixed reflection phases and a spacelike cut, a scaling
description of the
chain~\cite{HolzheyLarsenWilczek1994,CalabreseCardy2004,BianchiSmerlak2014};
formula and labels in Ref.~\cite{McKinneyCode2026}), reproduces the
propagated chain's growth within \(4\times10^{-4}\) nats per sector
below two e-foldings and \(6\times10^{-3}\) at three, exponential
stretching adding \(\kappa_u/12\) per sector per unit time, the
thermal entropy flux at \(\kappa_u/(2\pi)\). Hard formation light,
the light emitted when the trapping surface forms, then breaks this
agreement: growth at \(2r_s\) levels at 1.28--1.40 nats per sector
over 60--100, with about half that excess over the held vacuum, and
the late level's mechanism and size or trapped-region dependence
remain uncomputed.
```

### 10.134 Sec. V C: tunnelling-drain paragraph before shortening (review note 131, SHORTEN)

```latex
Read as tunnelling, the held-field drain comes from one quasi-bound
level pair, but its isolation is not verified. The pair, one \(k=1\)
pole each for hole and particle, has width \(5.65\times10^{-6}/a\),
e-folding time about \(3\times10^3r_s\); the hold spans about 0.14 e-fold, so
neither the line nor its proximity to \(L_H(T_H)\) has thermal
meaning. Isolation, attributing the inventory's shell flux to these
pole blocks alone, requires their residue subspaces to stay within
predeclared stability budgets: the width-limited drain statistics
pass in both calculations, but isolation fails two budgets in the second (named, with values, in Ref.~\cite{McKinneyCode2026}), so the
verdict is ``drain statistics pass, isolation not verified'', not a
verified two-calculation width-limited drain of sign-reversed states. Channel \(k=2\)
carries no drain of sign-reversed states; the first return is near 1000, and none of
these stored slices completes the drain.
```

### 10.135 Sec. V C: light-gravity control paragraph before rewording (review note 133, SHORTEN)

```latex
% provenance: light gravity +0.0042a, 3.4e-5, 0.2-0.4%: <internal>/compC engine_twofield_v8; readout <internal>/C_readout_s4_v3c.md; <internal>/C_sigma4_coupling_effect_v1.2.md.
% provenance: runs C32_s4_ell4_tau2_twofield_v3 and _v3_off (nsites 2048, tau 2, ell 4, light k=1,2, to 80); launch <internal>/pod_launch_C_v11.sh. One route.
Light gravity shifts the trapping radius by a fraction of a lattice spacing in a single-calculation control with unverified errors: at \(r_s/a=32\), \(\sigma=4a\), light gravity in \(k=1,2\) shifts the first fine-grid outer trapping radius by \(+0.0042a\) from the source-off control, with small reported charge and energy differences through 80 (values in Ref.~\cite{McKinneyCode2026}); matched comparison ends at 50, continuation to 80 establishing no matching beyond it, and the charge change is a conservation residual, not a feedback bound, its deficit roughly matching energy already inside the cut at 40.
At \(r_s/a=64\), the two-field history is not formed \todo{re-evolve the two-field size-64 history with Eq.~\eqref{eq:dm-reference}'s sources}.
Escaping entropy is not compared with the control.
```

### 10.136 Sec. V E: evolved-trapping-cut paragraph before shortening (review note 139, SHORTEN; the explicit window-constant conditions kept here)

```latex
% provenance: evolved-cut excess 109-311 / 116-305 nats: <internal>/evolvedcut_v1 (ec_recompute.py, ec_spectra.py) and <internal>/solve run_EVOLVEDCUT_secondroute_v1.py; compare <internal>/EVOLVEDCUT_route_compare_v1.md.
On the evolved trapping cut, the outer fine-grid trapping radius or \(2GE\) without
one, the heavy field's weighted \(k\le10\) exterior entropy exceeds
the held vacuum's by 109--311 nats over 12--40 at width 4 (two
independent calculations at six of ten clocks) and 116--305 at width
8 (one calculation), angular-channel readings, not the cubic cut. At
fixed \(ma>0\), Sec.~\ref{sec:results-vacuum}'s direct comparison
would give the flat leading coefficient on evolved cubic cuts if the
cubic budget \(E_f(C)\le KL\) held uniformly in size, with \(K\), \(f_*^{-1}\), the scaled support, and \(L\|\nabla f\|_\infty\) size-independent and site-diagonal \(f\ge f_*>0\) on the cut region; neither a comparator's \(r_{\rm ref}\) nor
Eq.~\eqref{eq:B-transfer}'s gap and distance premises enter. The reported budgets are chain readings, not an established uniform cubic premise, so no small evolved-area error follows at this size. \todo{paired cubic cut at the fine-grid trapping radius: flat ball entropy and cubic sea plus dust on the stored back-averaged fields, same site set, budget and cross-block norm on both; a one-way cubic control on the stored geometry, not the coupled cubic object}
```

### 10.137 Sec. VI C: Slater-burst bound (review note 155, CUT)

```latex
A pure Slater burst of \(N_B\) occupied heavy orbitals with mean
exterior fraction \(f_N\) has straddling entropy at most \(N_Bs(f_N)\),
which establishes neither a later plateau nor a turnover.
```

### 10.138 Sec. VI D: conditional-inputs paragraph (review note 162, CUT)

```latex
The four two-calculation inputs are conditional: the vacuum values \(\kappa_R=1/(2R)\) and \(T_H=1/(4\pi R)\), Fermi transmission at \(\kappa_R/(2\pi)\) through a stationary dispersive layer at a simple trapping radius, full incoming occupation from the stretched moving-site sea on a tested shrinking comparison segment, and the exterior scattering estimates \(\mathcal L,\mathcal R\); post-crossing vacuum is proved for the dust solution, not the object's fountain, supply tests omit initial and terminal stages, and these inputs establish neither a complete channel nor sustained post-window emission, delay not being duration.
```

### 10.139 Sec. VI D: terminal-radius paragraph (review note 166, CUT)

```latex
% provenance: R_T/R_0 ~ 0.012285: <internal>/ROADMAP_STEP1_R2_v1.md (D7, l.239) and ROADMAP_STEP1_R2_v1_fable.md; 0.0126 and 0.0118-0.0128 not matched; landed a55713ff364.
The comparison case's expansion ends at a terminal radius
\(R_T/R_0\simeq0.012\), scaling as \(\chi_0^{2/9}\) at fixed
\(G/a^2,\mathcal L\) (two independent calculations; port-depth
criterion and values in Ref.~\cite{McKinneyCode2026}), which
establishes neither port operation nor an endpoint; release of the
remaining \(10^{-4}S_0\) is assumed, and separate store-algebra,
energy-supply, and carrier results do not establish joint operation
on the prepared state.
```

### 10.140 Sec. VI B: late-stage paragraph before shortening (review note 148, SHORTEN)

```latex
The late stage is read from far away, where only the exterior rays matter. At fixed \(\zeta,g,\eta_E\), a fixed-shape cold cloud of release
radius \(O(r_s)\) has Fermi momentum \(k_Fa=O(\chi^{2/3})\), so the leading approximation is pressureless. Marginally bound
dust with constant-charge vacuum exterior preserves \(k=0\) at
\(N=1\) (Eq.~\eqref{eq:dr-comoving}); Eq.~\eqref{eq:star-dust-control}
gives \(R_b-r_s=t_c-t+O((t_c-t)^2/r_s)\), the surface crossing
\(r_s\) at finite lattice and proper time \(t_c\). The rows do not halt compression: formal continuation beyond the hydrodynamic limit reaches the comparison radius \(R_\pi\), \(R_\pi/r_s\propto\chi^{2/3}\),
within \(\tfrac23r_s\) of lattice time after \(t_c\). Arrival, halting, and subsequent density there remain unestablished; no halting density is imposed.
```

### 10.141 Sec. VI B: received-redshift and cutoff paragraph before shortening (review note 149, SHORTEN)

```latex
For \(1\gg\delta/r_s\gg\max\{(2\chi q)^{2/3},\sigma^2/r_s^2\}\),
\(\delta=R_e-r_s\), an outgoing long-wave front in the exterior of
Eq.~\eqref{eq:dr-exterior}, emitted from the surface at \((t_e,R_e)\), reaches a static
receiver at retarded time
\(u=t_e+2r_s\ln(2r_s/\delta)-3r_s+O(\delta)\).
Here \(u=t-r-2\sqrt{r_sr}-2r_s\ln(\sqrt{r/r_s}-1)\) is constant
along the ray, with Sec.~\ref{sec:star}'s normalization.
Since \(\delta=t_c-t_e+O((t_c-t_e)^2/r_s)\),
\(R_e(u)-r_s=2e^{-3/2}r_se^{-(u-t_c)/(2r_s)}[1+O(e^{-(u-t_c)/(2r_s)})]\);
launch enters only through \(t_c\) on this branch. This gives the surface's approach, not the emitting shell's thickness or distribution.
The surface's emission is received at \(g_\infty=N+v=1-\sqrt{r_s/R_e}\):
\(\ln(1/g_\infty)=(u-t_c)/(2r_s)+\tfrac32+O(\delta/r_s)\), with
Eq.~\eqref{eq:sc-ray-temperature}'s exponent \(\kappa_H\).
Interior light escapes this fixed-charge continuum comparison solution only
if it crosses the surface before \(t_c\); afterwards the surface
lies inside the trapping radius and outgoing long waves move inward, so none
from filling at \(R_\pi\) reaches any \(u\).
At fixed \(\chi\), these relations are not uniform at infinite
reception time. Mode \(q\) reaches the prescribed dispersive cutoff at \(1-r_s/R_e=\delta_d\), equivalently \((R_e-r_s)/r_s=\delta_d/(1-\delta_d)\), where \(u-t_c=\tfrac43r_s\ln[1/(2\chi q)]+O(r_s)\), with continuum estimate \(1/g_\infty\simeq2/\alpha_d^2\); beyond it propagation follows Eq.~\eqref{eq:dr-crossing-ray}'s lattice branch with evolving sources, not followed here.
At a smooth, simple, held trapping radius, the radial massless operator's near-surface limit gives Fermi-shaped two-channel transmission at
\(\kappa_R/(2\pi)\). This local scattering result fixes neither incoming occupation, sustained supply, escape yield on evolving fields, nor heavy-information transfer. Equation~\eqref{eq:star-crossing-delay} is a delay; it is not a release law.
```

### 10.142 Sec. VI C: bounded-band burst bound before shortening (review note 154, SHORTEN)

```latex
% provenance: B_12 = 1.029 at x in [25,100] and the 2e-72 bound are closed-form evaluations of the displayed inequality (n_ch 192, Delta t 40 r_s, chi 1.09e-38).
Exclusion bounds bounded-band bursts even with frequency coherence:
for freely propagating outgoing channels normal ordered against the
empty band \(0\le\omega_1<\omega_2\), the band-restricted
energy-current kernel's positive eigenvalue bounds every state's
current, interchannel coherence included (eigenvalues in
Ref.~\cite{McKinneyCode2026}), so in infinity-arrival interval
\(\Delta t\), with \(\omega_i=x_iT_H\), \(n_{\rm ch}\) channels carry at most
\(B_{12}n_{\rm ch}(\Delta t/r_s)(x_2^2-x_1^2)g\chi^2/(32\pi^3)\) of
\(E\), with \(B_{12}=1\) for frequency-diagonal occupations, the
filled-band count, and \(B_{12}=1.029\) for \(25\le x\le100\)
(formula in Ref.~\cite{McKinneyCode2026}); the one-solar-mass bound
with \(n_{\rm ch}=192\) (\(k=\pm1,\pm2\)) and \(\Delta t=40r_s\) is
near \(2\times10^{-72}\), escaped only by hardening to lattice
energies, more channels, or a longer arrival.
```

### 10.143 Sec. VI D: Page-curve setup paragraph before shortening (review note 157, SHORTEN)

```latex
% assumption: the store's activation time is part of the assumed laws (the store itself is introduced as assumed only in the next paragraph), and in this extension the light field's sites move with the flow, a departure from the fixed-lattice declaration.
% risk: medium
% GR slippage: no
% fold into text: yes, in the sentence ("In the assumed laws, store loading activates ..."); done in this pass
The trapping radius \(R\) encloses \(E_R\), with \(A_R=4\pi R^2\). In the assumed laws, store loading activates
\(O(R_0\ln(1/\chi_0))\) in lattice time after matter crosses \(r_s\).
At activation, define \(R_0,E_0\), \(S_0=\pi R_0^2/G\),
\(\chi_0=a/R_0\), and elapsed retarded time \(t=u-u_a\).
Take \(\chi_0\to0\) at fixed \(ma,G/a^2,\eta_E\).
In this extension, light-field radial sites move with the flow,
\(\dot r_n=v(r_n)\); preceding fixed-site results keep their scope.
```

### 10.144 Sec. VI D: first-assumption paragraph before shortening (review note 159, SHORTEN)

```latex
% provenance: partner ledger S(B|X)>0 at a 2000a held root: <internal>/PARTNER_LEDGER_S2_webgpt_v1.md and <internal>/PL2_MANIFEST_routeA_solve_v1_ADDENDUM8.md; folded in a81972666e9.
The Page curve below rests on assumptions stated here once. First,
two properties not supplied by the existence of sign-reversed states: an
accessible retained sector with
logarithmic dimension at most $fA_R/(4G)$ and $o(S_0)$ retained
entropy outside it, and transport of partner correlations to
the no-return output as the trapping radius recedes under emission, a
recession the computed histories do not show (Sec.~\ref{sec:page}),
at the rate required
by the minimum law below, followed by complete release.
On held fields two independent partner-accounting calculations find positive conditional entropy \(S(B|X)\) of a later escaped block given earlier output, so neither verifies an entropy-reducing addition (channel set, trapping radius, and checkpoint in Ref.~\cite{McKinneyCode2026}).
\todo{the partner accounting: establish the accessible-sector bound
from sign-reversed-state occupation, deposition, and drift, including omitted
sectors; compute no-return conditional entropies, residual retained
entropy, and coherent heavy-input recovery; close the windowed trapping-radius energy budget and branch errors. Test the first controlled blocks
at sizes 32 and 64, then derive the thin-shell continuation through
turnover with controlled matching to the microscopic dynamics and
uniform error bounds.}
```

### 10.145 Sec. VI D: third-assumption paragraph before shortening (review note 161, SHORTEN)

```latex
Third, the laws require one linear autonomous charge- and
number-conserving generator supplying all referenced sources in
Eq.~\eqref{eq:dr-radial}, with heavy transitions (sea transitions
included) supplying the energy for every loading, control, and
release operation beforehand, escaping energy counted once (through
inward partners or equivalent matter-to-output transfer, not both),
\(dE_R/du=-L\) as a separate source requirement at the trapping radius under exterior
storage and apparatus work (stated as a shell balance below), and an
input-independent outgoing channel that assigns the received
occupations and saturates the smaller of marginal budget and store
capacity up to \(o(S_0)\): a minimum law assumed, not realized, since
heavy-copy number conservation hides an occupation-pattern store's
cross-sector coherences from the light output, so a realization
needs a charge-matched code.
% RED MARK 1 (stretch goal, owner 1 Nov): construct the channel from the
% model in a charge-matched, copy-twirl-correctable code, store and
% capacity realized, not assumed. Frozen occupation-pattern codes are
% ruled out (E_dec >= 13/16 route 1, >= 0.174 route 2, tolerance 5e-4).
All retained content is released, not deleted (apparatus-entropy, pre-activation, and unintended-output assumptions in Ref.~\cite{McKinneyCode2026}).
```

### 10.146 Sec. VI D: firewall paragraph before shortening (review note 165, SHORTEN; figure and caption unchanged in the paper)

```latex
The firewall question is which side, the earlier radiation or the
retained region, holds a late emitted quantum's
partner~\cite{AMPS2013}. The exchange-current response between the
emitted mode and either side measures that side (the
Bogoliubov--Kubo--Mori response and its weights are defined in
Ref.~\cite{McKinneyCode2026}); dephasing removes the response but not
the number anticorrelation, so it is no entanglement or monogamy
bound.
\begin{figure*}[t]
\centering
\includegraphics[width=0.80\textwidth]{fermion_new/figures/page_curve_amps.pdf}
\caption{Leading-order comparison curve from the assumed laws, not an object history.
(a) Eq.~\eqref{eq:page-effective-curve}: \(f=1\) (black),
\(0.887\) (dashed blue), \(0.7883<f<0.9831\) (band).
Dots: maxima; dotted lines: Page times; open circle: formal end
outside the expansion, not verified completion.
(b) At \(f=1\), earlier-radiation (orange) and retained (blue)
fractions of the exchange-current response (defined in
Ref.~\cite{McKinneyCode2026}), an exact logarithmic-mean response,
summing to one, averaged over 0.04 in
\(S_{\rm marg}/S_0\).
Additional assumptions: independent fixed-number pure pairs with single partners; ideal charge- and number-matched encoding, decoding, and release; release at pair occupation with particle-hole relabelling;
comparison spectrum; no action on escaped quanta; divisible,
delay-free inventories; product-marginal exchange-probe states.
Solid curves use uniform release. The switch in (b) is sharp because
the assumed release law is a minimum of two branches, nothing stored
being released until the store saturates at the Page time; the 0.04
averaging is the only smoothing, and how a realized channel would
spread the switch is not computed. Bands are attainable interval averages for release orders obeying the entropy law, nonnegative inventories, and complete formal release. Both panels assume constant coefficients, no pre-activation entropy, and formal continuation to \(t_*\); bands show coefficient and ordering freedom, not physical error.}
\label{fig:page-conditional}
\end{figure*}
For independent pairs at fixed particle number, each with a single
partner, a new partner is retained and a released partner lies in
earlier radiation; the response weight lies on the partner's side and weights cold
spectra differently from entropy, since the emitted occupation does
not approach one half even as \(aT_H\to0\) (weights in Ref.~\cite{McKinneyCode2026}). New pairs add marginal entropy and released stored halves remove it, so
the pair model releases none before Page time and then fraction
\(\tfrac12(1+f/\mathcal R)\) of emitted marginal entropy. Every
admissible release order has this zero instantaneous pre-Page
fraction and complete-formal-post-Page entropy-weighted mean; uniform
release also gives it as the earlier-radiation response fraction, and
other orders fill inventory-constrained bands. These statements need
the pair model, not the entropy law alone; centred averages crossing
Page time can be nonzero earlier.
```

### 10.147 Sec. VI D: finite-parent response cross-check before shortening (review note 169, SHORTEN; the one-slice diagnostic values kept here)

```latex
A finite-parent response cross-check, not endpoint evidence: at the stored nominal-size-80 slice with light weight 16 and heavy-only polarization subtraction, two independent calculations
agree to about a part in a thousand on the fixed-radius reference
contribution: it carries the mass rate's sign and about 8.6 percent
of it, with 0.7 percent from the light term absent from that history.
The trapping-radius-following diagnostic is \(-(3.27\pm0.10)\times10^{-3}\) in
lattice units; the absent light response lowers the specified
unweighted heavy-light Hessian-square sum by
\((2.4043\pm0.0004)\times10^{-4}\). These are one-slice diagnostics, not a re-evolved balance, and establish no finite light-endpoint coefficients; the \(e_{11}\) obstruction remains.
```

### 10.148 Sec. VII A: information-paradox paragraph before shortening (review note 173, SHORTEN; the transfer-diagnostic sentences in their original form)

```latex
% provenance: L ~ 0.005237 = C_0 of page_curve_amps R1/R2 (see Eq. sc-luminosity provenance).
The information paradox needs no impermeable
surface~\cite{Hawking1976}: for earlier radiation \(R\), a new mode
\(b\), and its retained partner \(c\), weak monotonicity gives
\(S(Rb)-S(R)\ge S(c)-S(bc)\), so a nearly pure partner
(\(S(c)\ge\ln2-\epsilon\), \(S(bc)\le\epsilon\)) forces a
radiation-entropy increase of at least \(\ln2-2\epsilon\) whatever
the surface's permeability (Ref.~\cite{McKinneyCode2026}), the monogamy behind the firewall argument~\cite{AMPS2013}. Transfer
must carry partner correlations, not just probability. For the pure
Hartree light covariance and complementary once-counted output and
retained projectors \(E\) and \(Q\), \(QCQ-(QCQ)^2=QCECQ\): escaped
entropy equals entanglement with the whole retained complement. A
later output block reduces accumulated entropy exactly when its
conditional entropy is negative. Propagating the retained singular
vectors of the earlier-output cross block measures partner transfer
and censored delays. On a stationary held future, after complete
continuous output is accounted for, the residual is the bound-sector
entropy; finite-chain holds require a return budget and do not
establish that limit by themselves. The logarithmic branch delay
bounds no transferred weight. These finite-history quantities are
\todo{C1/C2 conditional entropies, partner-transfer matrix, and
held-future residual controls}. Complete release through turnover
remains open.
```

### 10.149 Sec. VI D: five-slice root-debit mark paragraph and finite-parent cross-check before the reviewer B rdcmp2 text (lead-routed inbox 64, red mark 3; the half-weight light counterterm sentence, the balance/debit-share hedge, the open-items sentence, and the second todo are kept in the paper; the cross-check paragraph is cut, its numbers used by no later sentence)
```latex

A finite-parent response cross-check, not endpoint evidence: at the stored nominal-size-80 slice with light weight 16 and heavy-only polarization subtraction, two independent calculations agree to about a part in a thousand on the fixed-radius reference contribution, which carries the mass rate's sign and about 8.6 percent of it (one-slice diagnostics in Ref.~\cite{McKinneyCode2026}, not a re-evolved balance, establishing no finite light-endpoint coefficients; the \(e_{11}\) obstruction remains).

The five-slice test uses the stored two-field trajectory with
light-reference weight 16, not the paper's weight 32: \todo{signed trapping-radius-to-shell accounting over \(T/r_s=14.9\) to 15.1, with increment
and scheme errors}. It tests balances and debit shares, not Hawking
depletion. A scaling-uniform \(\sup|\mathfrak d_c|=o(E_0)\) through
turnover remains open, as do the full heavy-light boundary-charge
Hessian, prepared-sector multiplicities, and funded complete release.
\todo{a trapping radius clear of heavy matter, shell remainder bounded;
other-clock reference terms; energy-supplied complete release; uniform
decoding and response-error bounds}
```

### 10.150 Sec. VI D: pair-model response-weight parenthetical (payment for inbox 64; the sentence around it is kept)
```latex
For independent pairs at fixed particle number, each with a single partner, a new partner is retained and a released partner lies in earlier radiation, so the response weight lies on the partner's side (weighting cold spectra differently from entropy, since the emitted occupation does not approach one half even as \(aT_H\to0\); weights in Ref.~\cite{McKinneyCode2026}); new pairs add marginal entropy and released stored halves remove it, so the pair model releases none before the Page time and then the fraction \(\tfrac12(1+f/\mathcal R)\) of emitted marginal entropy, every admissible release order sharing this zero pre-Page fraction and post-Page mean while other orders fill inventory-constrained bands. These statements need the pair model, not the entropy law alone, and centred averages crossing the Page time can be nonzero earlier.
```

### 10.151 Sec. VI D: fixed-lattice sign-reversed-state count estimate (payment for inbox 64; the first clause and "exclusion supplies no release law" are kept)
```latex
heavy information by itself. We therefore assume a store of \(\lfloor e^{fA_R/(4G)}\rfloor\) states carried at the trapping radius (computed ceilings on the sign-reversed sector close this subsection),
\(f=4\eta(ma)G/a^2\), the identification of the store coefficient with the cut coefficient being part of the assumption.
In stationary fields, partners at negative Killing energy turn inside the trapping radius (Sec.~\ref{sec:star}) into the light field's sign-reversed states. On the fixed lattice, a trapped profile with a nonzero,
size-independent fraction of sign-reversed states has of order $(R/a)^3$ sign-reversed states; if
independently produced pairs keep their partners there,
$S_{\rm esc}=S_{\rm marg}$, and the count of sign-reversed states meets the
accumulated partner count at $R=O(aS_0^{1/3})$; channel and energy
restrictions may require earlier release, redistribution, or
emission suppression, but exclusion supplies no release law.
```

### 10.152 Sec. VI D: firewall paragraph opening with the Bogoliubov--Kubo--Mori parenthetical (payment for inbox 64; the sentences are kept with the parenthetical reduced to the reference)
```latex
The firewall question is which side, the earlier radiation or the
retained region, holds a late emitted quantum's
partner~\cite{AMPS2013}. The exchange-current response between the
emitted mode and either side measures that side (the
Bogoliubov--Kubo--Mori response and its weights are defined in
Ref.~\cite{McKinneyCode2026}); dephasing removes the response but not
the number anticorrelation, so it is no entanglement or monogamy
bound.
```

### 10.153 Sec. VI D: store-assumption paragraph with the fresh-sea clause and the subsection orientation sentence before trimming (payment for inbox 64; everything but "fresh-sea emission with inward partners drains no heavy information by itself" and "The Page curve comes last because its laws are assumed" is kept)
```latex
The Page curve comes last because its laws are assumed: a Page curve tracks the no-return entropy \(S_{\rm esc}\) in time at infinity~\cite{Page1993}; the model fixes its temperature and area normalization but no laws of emission, storage, and release, so the curve below is conditional on assumed laws, not derived object dynamics.

% GR slippage: no
% fold into text: no; "We therefore assume" is stated
Purity makes a vanishing retained entropy ceiling force turnover,
but the model provides no accessible count of size \(fA_R/(4G)\).
Original heavy particles alone supply only
\(O(N_B\ln(1/\chi_0))\) nats, about \(10^{-36}S_0\) at one
solar mass; fresh-sea emission with inward partners drains no
heavy information by itself. We therefore assume a store of \(\lfloor e^{fA_R/(4G)}\rfloor\) states carried at the trapping radius (computed ceilings on the sign-reversed sector close this subsection),
```

### 10.154 Sec. VI D: inner-comparison and condition (a) sentences with the narrowed root-debit todo, as landed in 2bbcd4c4ae7 (lead-routed inbox 65 and 66; replaced by the inner-overlap statement, the interval-Newton uniqueness sentence, and the limitation sentence; the todo removed on the lead's instruction, mark closed for the implemented law)
```latex
The second calculation bounds the inner trapping radius's debit negative (outward motion) at all five slices; agreement with the first calculation's intervals is being checked. Condition (a), full reference response in the local fields and a unique trapping radius, is pending at all five slices; the Hartree reading of the debit and its sign depends on it: \todo{condition (a), its uncertainty propagated into the debit intervals; the unrounded inner-trapping-radius interval comparison under P and H}. The test checks balances and debit shares, not Hawking
depletion.
```

### 10.155 Sec. I: the "None starts" and "Where the older approach" sentences (owner comment cmw026dyp2bah, lead-routed; replaced by one positive motivating sentence, the prior-work sentences before them kept)
```latex
emergent-gravity models. None starts from a specific matter
Hamiltonian on fixed sites with no metric and derives from it the
gravitational constraints and Schwarzschild's problem, a spherical
source, its exterior, and its collapse, posed for a quantum lattice fermion. Where the older approach to emergent gravity put in an action, a deformation algebra of constraints, or constraint theorems, the present one puts in a single charge requirement.
```

### 10.156 Sec. IV A: massless-chain endpoint paragraph before shortening (structural pass, STRUCTURE_PLAN_20261103 payment P1; the no-finite-endpoint statement, the divergence of e_11, the undecided e_01, and the cross-check status of the 80 interval rows are kept)
```latex
The massless light chain has no finite endpoint under the constant-coefficient prescription. At \(m=0\), \(N=1\), \(v=0\), the site-flow kernel \(K=\tfrac12E_{vv}\) has exact range three, and for every \(k=1,\ldots,5\), separately on even and odd rows, the verified limits \(c_0=\lim_r\lim_{L_c}r^2(K\mathbf 1)_r\) and \(c_1=\lim_r\lim_{L_c}r(Kx)_r\), \(x_n=n-1\), parent removal preceding the row limit, both vanish. With \(\alpha=1/(12\pi)\) the endpoint rule then gives \(a_{11}=c_1-\alpha=-\alpha\) and \(b_{e01}=(c_1+c_0)/2=0\) for the massless tail's endpoint coefficients \(e_{11}\) and \(e_{01}\) (Ref.~\cite{McKinneyCode2026}), so \(e_{11}\) diverges and \(e_{01}\) is undecided, with no cancellation established. The 80 finite-parent interval rows at \(L_c=4096\), 4097, 8192, 8193 and \(r=32\), 33, 64, 65, from two independent numerical calculations, are implementation cross-checks, not coefficient bounds, the first calculation's finite-lattice values verified at the target tolerance \(\alpha/10\) on all 40 chains
(\(k=1,\ldots,5\), lattices \(2^{12}\) to \(2^{15}\) with partners),
falling from about \(10^{-3}\) at \(r=32\) to about \(10^{-5}\) at the
window end: finite lattice, not a limit.
```

### 10.157 Sec. IV B: size and mass ladders paragraph before shortening (structural pass, payment P2; the first-slice dependences, the no-trapping-radius cases, the isotropic-preparation sentence, and the width-agreement statements are kept, the ladder ranges and the width and step directions moved to the reference)
```latex
The size and mass ladders, the families of runs varying \(r_s/a\) or \(\zeta\), give first-slice dependences, not fitted laws: larger sizes form earlier in units of \(r_s\) at a smaller fraction
of \(2GE\), lighter masses later at a larger fraction, and runs near the maximum mass (\(E_0/E_{\max}=1.14\)) or at small \(\zeta\)
form no trapping radius within the computed time (the ladders run over nominal
sizes 32 to 128 at \(\zeta=0.2\) and over \(\zeta=0.4\) to 0.05 at
sizes 64 to 326, run values in Ref.~\cite{McKinneyCode2026}). The isotropic preparations fill eight
states per momentum, not the sixteen of the continuum fluid.
Width agreement holds only near the reference size: wider averaging
delays formation and raises its trapping-radius fraction, halving the step
advances it by fractions of \(r_s\), size 32 and \(\zeta=0.4\) fail
width agreement, the 128 pair passes only at formation, and the
size-256 formation, reported at one width from a run differing in
preparation, stays a reported, not verified, reading (run values in Ref.~\cite{McKinneyCode2026}).
```

### 10.158 Sec. IV C: the closed-form clock integral of Eq. sc-clock-exact (structural pass A.9 demotion; the first line of the display is kept with its label, the integral named in words)
```latex
\begin{equation}
\begin{aligned}
(\ln N)'&=\frac{2\pi r^2\rho}{\mathcal M}-\frac{\mathcal M'}{2\mathcal M}
 =\frac{\lambda-1}{2}(\ln\mathcal M)',\\
N(r)&=N(r_1)\sqrt{\mathcal M(r_1)/\mathcal M(r)}\\
&\quad\times\exp\!\Bigl[-\!\int_r^{r_1}\!
 \frac{2\pi r'^2\rho(r')}{\mathcal M(r')}\,dr'\Bigr]
\end{aligned}
\label{eq:sc-clock-exact}
\end{equation}
so \(N\propto\mathcal M^{(\lambda-1)/2}\) on any annulus of constant \(\lambda\). The lapse values \(N(R_\pi)\) and \(N(R_t)\) are close only
```

### 10.159 Secs. II B, II D, and III E: four displays made inline (structural pass A.1 and A.4 demotions; the content is kept inline, the labels eq:dm-current, eq:dm-gauss, eq:dr-exterior-coordinates, and eq:dr-average-generator are retired and their two references reworded)
```latex
\begin{equation}
Q_i=\frac{1}{2i}[X_i,D^2]
   =\frac{T_i^2-(T_i^\dagger)^2}{4ia},
\label{eq:dm-current}
\end{equation}

\begin{equation}
\rho_\ell+w\cdot J_\ell=\nabla\cdot\Bigl[\frac{F}{16\pi GN^2}
-\frac{c}{8\pi G}\frac{\nabla N}{N}\Bigr]
-\frac{c}{16\pi G}\Bigl|\frac{\nabla N}{N}\Bigr|^2
\label{eq:dm-gauss}
\end{equation}

\begin{equation}
dt_s=dt+\frac{v\,dr}{1-v^2},\qquad
r=\tilde r(1+GE/(2\tilde r))^2,
\label{eq:dr-exterior-coordinates}
\end{equation}

\begin{equation}
W_\ell^\dagger v=v[1-9\sigma^2/(8r^2)]+O(|v|(\ell/r)^4).
\label{eq:dr-average-generator}
\end{equation}
```

### 10.160 Sec. IV B: width-dependence paragraph before shortening (structural pass C, payment P3; every claim and hedge kept, the unverified-reading list named in the reference)
```latex
% provenance: collar charge 0.847/0.919/0.975: <internal>/SCALING_ladder_v1.3.md:198; channel control <internal>/W5_W4_formation_attribution_v1.md sec. 3, attrib_readout_v1.json.
% provenance: W5 members kappa_sea 10/14/20, tau 8 and 4, nsites 2048, ell 4, to 14 (<internal>/lad_W5_*). Centroid minima line not pinned.
The interior readings depend on the width at order one while formation is robust under the channel count (10 to 20 channels leave formation's slice unchanged and its radius within \(0.09a\)), so no closing time, lifetime, late-time law, receiver-time redshift law, or second surface is established. Centroid minima move inward with width (centroid motion, not interior rebound), the shell charge at \(16r_s\) shows a held-step deficit that halves with step length (a held-step effect, not a charge-loss law), the twenty-channel readings at 14 and the weighted sea pair energy are unverified, conserving-step depth and centroid remain open, and nothing below uses the depth or centroid values (run values in Ref.~\cite{McKinneyCode2026}).
```

### 10.161 Sec. IV C: boundary-charge deviation paragraph before shortening (structural pass C, payment P4; the endpoint range and perturbation radii moved to the reference, every claim and the todo kept)
```latex
The boundary charge's own deviation is small on the sampled slices but not validated. For \(q_\infty=4\langle(\widehat E_2/E-I)^2\rangle\), two independent calculations with independent full-tail comparators give matched nominal upper endpoints from \(2.93\times10^{-5}\) to \(9.27\times10^{-4}\) on five ten-channel stored slices, with single-calculation sensitivity estimates, not verified robustness ranges, placing the 0.080 and 1 crossings at relative perturbation radii of a few \(10^{-3}\) and a few \(10^{-2}\) (values in Ref.~\cite{McKinneyCode2026}). A validated \(q_\infty\le0.080\) would give branch weight at least 0.92, but neither actual-error inclusion nor the declared-reference branch calculation is established, and the five slices cover neither all 18 nor \(T/r_s=11.0\). In two independent calculations with \(\widehat E_2\) Wick ordered about the evolved Hartree state, the Fock single amplitude and the relative Fock correction to the mass rates stay below \(10^{-2}\) of their Hartree counterparts on ten channels at \(T/r_s=11.0\) and 15.0. The single ratio grows with channel count through the propagated sea's exchange mismatch with the fixed reference projector, faster than the copy weights \(16k\), so no channel limit is established on the chains (values in Ref.~\cite{McKinneyCode2026})
\todo{two-calculation sea-reference exchange singles on a cubic lattice}.
On the cubic lattice the sea variance obeys an area law (relative enclosed-charge fluctuation \(\sigma^{-1/2}\) at leading order in \(a/\sigma\), outside the hydrodynamic premise). On the chains it grows quadratically with the channel cutoff.
```

### 10.162 Sec. IV B: formation-control sentences before shortening (structural pass C; the formation clocks and 2GE fractions of the width pairs moved to the reference)
```latex
% provenance: Fig. 1 = fermion_new/figures/item5_collapse_v3.1.pdf (md5 643fd822b0cae873cbba8c84546ad866), installed e804b9c5fc8.
% provenance: plot <internal>/figure_item5_collapse_v3_1.py (md5 8eab4a5f); data figures_data_v1.py (md5 e2469b43)
% provenance: npz <internal>/fig_item5_collapse_v3_data.npz (md5 941fa621); log figures_data_v1.log there.
% provenance: (a,b) <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11{.json,_fields.npz} (sigma/a 4, tau 8, 2048 sites).
% provenance: (b,c) <internal>/ladders_readout_v1_3.{npz,json}, members P64_s4/P64_s2/P64_s8.
% provenance: params m a 0.2, G/a^2 1/4, r_s 64a, N_B 640, R_0 4 r_s, kappa_sea 10; run on BOX qi2 env, matplotlib 3.10.8.
\begin{figure*}
\includegraphics[width=0.80\textwidth]{fermion_new/figures/item5_collapse_v3.1.pdf}
\caption{Collapse of 640 fermions prepared at \(4r_s\). Radii and flat-slice clocks use nominal \(r_s=64a\).
(a) \(N\) (solid) and \(|v|\) (dashed), before release (black) and
at 15 (red). In the grey trapped region outgoing long waves move
inward, out to the trapping surface (circle).
(b) Trapped region (grey) and its outer edge (black), \(2GE\) of
the whole object (dotted) and the matter's mean radius (red). The
surface forms at 12.1 (dashed line), inside \(2GE\). The trapped region closes at 17.4 and reopens at 18.4, and the matter later
spreads outside \(2GE\).
Field panels use a separate production history checked against
the reference collapse.
(c) \(\min_r(N-|v|)\), negative where trapped, through verified
initial history segments at halved (blue), reference (black), and doubled (red)
averaging widths (2, 4, and 8 spacings). Formation and depth
depend on width.}
\label{fig:star-collapse}
\end{figure*}
% provenance: formation clocks/radii: ladder readout <internal>/ladders_readout_v1_3.json (formation 12.125, closes 17.375, reopens 18.375).
% provenance: second route <internal>/FORMATION_newton_route_v1.1.md (Newton, collapse_lad_P64_s4_newton_v36nq_h100_1.json, md5 8f31394b); readers <internal>/.
% provenance: members P64_s4, _tau4, _tau2, P64_s2, P64_s8, W2_dg (conserving step) per <internal>/PFIX_VALUES_v1.md markers 2a-2c.
The trapping radius forms 0.5 to \(1.3r_s\) before the dust solution's trapping radius. The first stored slices across the steps and on a charge-conserving step (11.875) put the zero-step formation near 11.8 to \(11.9r_s\).
The factor-4 averaging control fails on both width pairs: the first
surface forms at \(T/r_s=12.0\) to 12.75 on the three runs, at 0.65
to 0.76 of the slice's \(2GE\), and the runs differ by 15 to 22
percent in radius at matched clocks (run values in
Ref.~\cite{McKinneyCode2026}).
```

### 10.163 Sec. IV B: received-redshift paragraph before the removal of the repeated factor and endpoint clause (structural pass C; the result-first head sentence of pass A is already in it)
```latex
The computed collapse reaches a received redshift factor of \(3\times10^4\) by \(T/r_s=40\), still rising. For each sampled class and verified arrival interval the headline reports \(\ln(1/g_D)\) at the maximum of
\(L_T=-\ln(d\tau_{\rm em}/dT_D)\): the sampled redshifts reach
\(\ln(1/g_D)=10.3\) (a factor \(3\times10^4\)) on the reference
collapse through \(T/r_s=40\), from an event leaving \(0.30r_s\)
before formation, and 3.1 to 15.6 across the controls and completions
(run values in Ref.~\cite{McKinneyCode2026}), the deepest arrivals
still growing at the endpoint. Flights lie almost wholly beyond the
ray form of the hydrodynamic test, which maximizes
\(\sigma|\partial_r\bar v|\) over the entire chain rather than only
to the dust centroid.
```

### 10.164 Sec. IV B: late-redshift sentence before shortening (structural pass C; the clock-shift value moved to the reference)
```latex
Late redshift is not robust: it changes at order one with width, step, completion, and propagation background (clock shifts up to 2.3 for fine-grid fields at the headline events, the field sensitivity not an error bound), hovering redshift tracks \(\int\kappa_Rdt\) with slope 0.85 to 1.25 on seven of ten reference-collapse clocks, and blueshifted arrivals at widths of 8 and above at both sizes and completions rule out a uniform sampled redshift bound (values in Ref.~\cite{McKinneyCode2026}).
```

### 10.165 Sec. IV C: same-source comparison case before shortening (structural pass C; the exponent formula moved to the reference)
```latex
The same-source comparison case \(\mathcal M=m\mathcal N_<\),
\(\mathcal N_<(r)\) the enclosed count, at constant bare energy per particle \(\epsilon_c/a\), \(\epsilon_c\) the dust's bare energy per particle \(a\rho/n\) inside the half-count radius \(R_{50}\), closes the law with
\(N\propto\mathcal N_<^{\nu}\), \(\nu=(\epsilon_c-\zeta)/(2\zeta)\),
and fails at a regular centre. On the stored slices the clock is no power of the enclosed count and no exponent is calibrated, while Eq.~\eqref{eq:sc-clock-exact} integrated on the fine-grid fields matches the clock's fall from \(R_{50}\) to the outer trapping radius within 1 to 3 percent in two independent calculations (slopes in Ref.~\cite{McKinneyCode2026}).
```

### 10.166 Sec. IV E: first-law and evolution-row diagnostic paragraph and the stress-identification paragraph before shortening (structural pass C, plan item P10; the slice list, the pair count above the floor, and the width qualifiers moved to the reference)
```latex
% assumption: the first-law and evolution-row diagnostics use the held-projector sources and a stress of unclosed provenance; agreement or disagreement with the no-work first law is not established by them.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: first-law sums -18.36/-6.51/-11.93 over 224 pairs (sigma/a 2, P64_s2): <internal>/FIRSTLAW_records_v2.2.md (l.55, 233); <internal>/firstlaw_v2, <internal>/firstlaw (function not pinned).
% provenance: row-residual medians 0.17/0.04/0.06 per the stress lineage (FABLE_write_stress_v2.1); landed 2cb9ac21f64, 21e8dd01028.
Finally the computed geometry is checked against the first law and the stress it should obey. On the stored time slices the terms of the first law do not add up to zero, and the stress does not meet its expected balance either. Neither check comes with an error bound, so both are indications, not tests.
On the \(\sigma=2a\) run's 224 consecutive pairs of stored slices with a simple fine-grid trapping radius (to \(T/r_s=40\), these slices lacking a vacuum exterior beyond it), the terms of Eq.~\eqref{eq:page-first-law-rows} sum to a residual of 0.08 lattice units against terms of order 10, the work term above its numerical floor in 221 pairs, but the stored \(\delta E\) carries the held step's charge deficit, up to 0.7 percent of \(E\) per slice near formation and comparable to these terms, so the diagnostic does not by itself exclude the no-work form (values and the fixed-scheme control in Ref.~\cite{McKinneyCode2026}). Between stored slices the mass inside each fixed outer trapping radius changes at a rate differing from the evolution row's with the stored kernel-averaged stress by medians of a few to 20 percent of the row's largest term at the three widths, and by up to 0.74 at inner trapping radii above \(2\sigma\), outside the hydrodynamic description, with the stress's identity verified only on sampled slices (below) and normalization, clock, and between-node sign unclosed (medians in Ref.~\cite{McKinneyCode2026}).
Independent reconstructions support identifying the stored stress as
\(S^{\rm op}[C-P_{\rm held}]\) on the shared stored time slices
\((r_s/a,\sigma/a,T/r_s)=(64,4,17)\), \((64,2,17)\), \((64,8,14)\),
and the checked \((128,8,30)\) anchor. Completed four-point matched
diagnostics show small differences between the two independent
calculations in prepared-operator and kinetic residuals on all three
\(r_s/a=64\) slices but an unresolved prepared-reference and
projection discrepancy at the anchor, while validated joint
kinetic/reference agreement and complete size, width, and step
controls remain unestablished. So identified, with \(P_{\rm held}\) the fixed reference projector
of Sec.~\ref{sec:results-lattice}, these are held-comparator
operator-stress residuals, not those of the kinetic \(S_{ij}\) of
Sec.~\ref{sec:model-gravity} with Eq.~\eqref{eq:dm-reference}'s
sources, and the split of Sec.~\ref{sec:model-gravity} remains
unclosed. A polarization coefficient needs two independent cubic-lattice calculations with its definition, sign, normalization, invariants, and error, beside the chains' separately normalized value.
\todo{matched kinetic and reference residuals on identical stored slices,
separated terms and residual identities before medians. The identification failures at \(r_s/a=32\) (\(T/r_s=12\), 14, 20) and \(\sigma/a=2\) (12, 14, 16). Size, width, and step controls. Independent polarization-coefficient audit}
```

### 10.167 Sec. IV E: stress-identification paragraph before shortening (structural pass C, plan item P10)
```latex
% assumption: the first-law and evolution-row diagnostics use the held-projector sources and a stress of unclosed provenance; agreement or disagreement with the no-work first law is not established by them.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: first-law sums -18.36/-6.51/-11.93 over 224 pairs (sigma/a 2, P64_s2): <internal>/FIRSTLAW_records_v2.2.md (l.55, 233); <internal>/firstlaw_v2, <internal>/firstlaw (function not pinned).
% provenance: row-residual medians 0.17/0.04/0.06 per the stress lineage (FABLE_write_stress_v2.1); landed 2cb9ac21f64, 21e8dd01028.
Finally the computed geometry is checked against the first law and the stress it should obey. On the stored time slices the terms of the first law do not add up to zero, and the stress does not meet its expected balance either. Neither check comes with an error bound, so both are indications, not tests.
On the \(\sigma=2a\) run's 224 consecutive pairs of stored slices with a simple fine-grid trapping radius (to \(T/r_s=40\), these slices lacking a vacuum exterior beyond it), the terms of Eq.~\eqref{eq:page-first-law-rows} sum to a residual of 0.08 lattice units against terms of order 10, the work term above its numerical floor in 221 pairs, but the stored \(\delta E\) carries the held step's charge deficit, up to 0.7 percent of \(E\) per slice near formation and comparable to these terms, so the diagnostic does not by itself exclude the no-work form (values and the fixed-scheme control in Ref.~\cite{McKinneyCode2026}). Between stored slices the mass inside each fixed outer trapping radius changes at a rate differing from the evolution row's with the stored kernel-averaged stress by medians of a few to 20 percent of the row's largest term at the three widths, and by up to 0.74 at inner trapping radii above \(2\sigma\), outside the hydrodynamic description, with the stress's identity verified only on sampled slices (below) and normalization, clock, and between-node sign unclosed (medians in Ref.~\cite{McKinneyCode2026}).
Independent reconstructions support identifying the stored stress as
\(S^{\rm op}[C-P_{\rm held}]\) on the shared stored time slices
\((r_s/a,\sigma/a,T/r_s)=(64,4,17)\), \((64,2,17)\), \((64,8,14)\),
and the checked \((128,8,30)\) anchor. Completed four-point matched
diagnostics show small differences between the two independent
calculations in prepared-operator and kinetic residuals on all three
\(r_s/a=64\) slices but an unresolved prepared-reference and
projection discrepancy at the anchor, while validated joint
kinetic/reference agreement and complete size, width, and step
controls remain unestablished. So identified, with \(P_{\rm held}\) the fixed reference projector
of Sec.~\ref{sec:results-lattice}, these are held-comparator
operator-stress residuals, not those of the kinetic \(S_{ij}\) of
Sec.~\ref{sec:model-gravity} with Eq.~\eqref{eq:dm-reference}'s
sources, and the split of Sec.~\ref{sec:model-gravity} remains
unclosed. A polarization coefficient needs two independent cubic-lattice calculations with its definition, sign, normalization, invariants, and error, beside the chains' separately normalized value.
\todo{matched kinetic and reference residuals on identical stored slices,
separated terms and residual identities before medians. The identification failures at \(r_s/a=32\) (\(T/r_s=12\), 14, 20) and \(\sigma/a=2\) (12, 14, 16). Size, width, and step controls. Independent polarization-coefficient audit}
```

### 10.168 Sec. V D: transfer-test paragraph before shortening (structural pass C; sweep lengths and the number-conservation identification moved to the reference)
```latex
On the computed histories, \(k=1\) four-site slabs straddling the trapping radius have largest singular values 4.7 and 5.8 times the held vacuum's at 15 and 30, past the limit of the hydrodynamic description.
Two tests past that limit find outward transfer in fractions that
depend on the preparation: radial \(k=1\) columns reach order-one
weight beyond \(16r_s\) by 47.5, and at size 32 the four-component
cubic operator on held point-sampled fields sends interior-source
weights 0.68--0.90 beyond 1.5 sweeps of \(29a\) after two to four
sweeps (number conservation identifying net crossing of the trapping radius), matched
long waves above \(m\) transferring order-one fractions and gap
preparations much less. The one column read in both descriptions
agrees only where both fractions lie below tolerance and differs by
up to 0.34 at later sweeps, so the lift of the chain description to
the cubic operator is not validated.
```

### 10.169 Sec. V E: evolved-cut paragraph before shortening (structural pass C; the width-8 values and the premise parenthetical moved to the reference)
```latex
% provenance: evolved-cut excess 109-311 / 116-305 nats: <internal>/evolvedcut_v1 (ec_recompute.py, ec_spectra.py) and <internal>/solve run_EVOLVEDCUT_secondroute_v1.py; compare <internal>/EVOLVEDCUT_route_compare_v1.md.
On the evolved trapping cut (the outer fine-grid trapping radius, or \(2GE\) without one), the heavy field's weighted \(k\le10\) exterior entropy exceeds the held vacuum's by 109 to 311 nats over 12 to 40 at width 4 (two independent calculations at six of ten clocks) and 116 to 305 at width 8 (one calculation), angular-channel readings, not the cubic cut. At fixed \(ma>0\), Sec.~\ref{sec:results-vacuum}'s direct comparison would give the flat leading coefficient on evolved cubic cuts if the cubic budget \(E_f(C)\le KL\) held uniformly in size with size-independent window constants (no comparator \(r_{\rm ref}\) and none of Eq.~\eqref{eq:B-transfer}'s premises enter). The reported budgets are chain readings, not an established uniform cubic premise, so no small evolved-area error follows at this size. \todo{paired cubic cut at the fine-grid trapping radius: flat ball entropy and cubic sea plus dust on the stored back-averaged fields, same site set, budget and cross-block norm on both, a one-way cubic control on the stored geometry, not the coupled cubic object}
```

### 10.170 Sec. V C: entropy-per-energy sentence before shortening (structural pass C; the definition and channel sets moved to the reference)
```latex
% provenance: collar flow ratios: PAGE v4.1 (ratio12_kT/TH in <internal>/PAGE_fig_v4.1_data.npz); readout <internal>/PAGE_solved_history_v4.1.md.
% provenance: diagnostic 0.020-0.023/0.05-0.08/0.38: route 2 <internal>/PAGE_secondroute_v1, <internal>/SECONDROUTE_PAGE_excess_v1.md; compare page_balance_compare_v1.
The shell energy flow lies below the Hawking comparison case before
formation light arrives, two orders above it during the passage, and
near it afterwards: for \(k=1,2\) the shell flow \(L_{\rm col}\),
energy flow through \(16r_s\) per flat-slice clock plotted at slice time, compared in Fig.~\ref{fig:page-v4}(b) with the two-channel
greybody flux \(L_H\) at \(\kappa_T/(2\pi)\), \(\kappa_T\) the slice's fine-grid surface slope, and at the stored slice charge's \(T_H\), is
at most 0.15 of the slope comparison flux before 36, two orders above over
42--46, and comparable over 80--100 except one negative checkpoint
(excitation fractions in Ref.~\cite{McKinneyCode2026}).
The entropy carried per unit of shell energy,
\(\overline T_H\,\delta(\Delta S_{\rm cut})/\delta E_{\rm col}\) with
endpoint-mean \(\overline T_H\) and the \(2r_s\) cut advanced by the
ray map, is 0.02 to 0.38 over 40--100 for \(k\le1,2,18\) at both cuts
in two post-processing calculations on shared exports
(Ref.~\cite{McKinneyCode2026}), far below \(\mathcal R=1.64\), and
classifies neither emitted entropy nor thermality.
```

### 10.171 Sec. V C: no-evaporation paragraph before shortening (structural pass C; the size comparison clause and the inward checkpoints clause moved to the reference)
```latex
% provenance: ladder roots from ladders_readout_v1_3 members; inflow 2.7% at 22 checkpoints from the PAGE v4.1 collar-flow series (<internal>/PAGE_solved_history_v4.1.md).
No evaporation, turnover, Page time, or approach to \(A/(4G)\) is
computed: single-calculation charge ladders retain a trapping radius to their final stored slice only from size 32 at \(ma=0.4\) and 64 at 0.2 (so
\(t_*\) is compared only at \(r_{s0}/a\ge32\)), over 40--100 the
incoming shell flow is at most 2.7 percent of the outgoing at the
positive-outgoing checkpoints with the outgoing-mode flow inward at
the others (checkpoint counts in Ref.~\cite{McKinneyCode2026}), and
late field supply is not separated from excitation drain.
```

### 10.172 Sec. V C: light-gravity control before shortening (structural pass C; the continuation and deficit clauses moved to the reference)
```latex
% provenance: light gravity +0.0042a, 3.4e-5, 0.2-0.4%: <internal>/compC engine_twofield_v8; readout <internal>/C_readout_s4_v3c.md; <internal>/C_sigma4_coupling_effect_v1.2.md.
% provenance: runs C32_s4_ell4_tau2_twofield_v3 and _v3_off (nsites 2048, tau 2, ell 4, light k=1,2, to 80); launch <internal>/pod_launch_C_v11.sh. One route.
Light gravity shifts the trapping radius by a fraction of a lattice spacing in a single-calculation control with unverified errors: at \(r_s/a=32\), \(\sigma=4a\), light gravity in \(k=1,2\) shifts the first fine-grid outer trapping radius by \(+0.0042a\) from the source-off control, with small reported charge and energy differences through 80 (values in Ref.~\cite{McKinneyCode2026}), the matched comparison ending at 50 (continuation to 80 establishes no matching beyond it) and the charge change a conservation residual, not a feedback bound, its deficit roughly matching energy already inside the cut at 40. At \(r_s/a=64\) the two-field history is not formed \todo{re-evolve the two-field size-64 history with Eq.~\eqref{eq:dm-reference}'s sources}.
Escaping entropy is not compared with the control.
```

### 10.173 Sec. V C: conformal companion before shortening (structural pass E payment P5; the circumference, the three-e-folding agreement, and the stretching rate moved to the reference)
```latex
% assumption: the conformal companion applies the 1+1 conformal-field-theory entropy formula on the derived background after dropping the centrifugal term and unfolding the centre; it is a scaling description, not the chain's exact entropy.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: conformal control v6s_B2_kz (kappa_zero) <internal>/v6s_B2_kz_k1_d13.json (md5 8db1de11), 8192 sites, k=1, to 100.
% provenance: 4e-4, 6e-3, 1.28-1.40: <internal>/NOTES_D1_conformal_control_v1.2.md (l.27, 41, 48); L_c = 265 r_s is the ray-circle label.
A conformal companion reproduces the \(2r_s\) formation excess until hard formation light arrives: one chiral Dirac field on a ray circle of circumference \(265r_s\), the massless channel without its centrifugal term unfolded through the regular centre, with cut entropy the arc entropy between outgoing and incoming entry labels (a scaling description of the chain~\cite{HolzheyLarsenWilczek1994,CalabreseCardy2004,BianchiSmerlak2014}, construction in Ref.~\cite{McKinneyCode2026}), reproduces the propagated chain's growth within \(4\times10^{-4}\) nats per sector below two e-foldings and \(6\times10^{-3}\) at three, exponential stretching adding \(\kappa_u/12\) per sector per unit time, the thermal entropy flux at \(\kappa_u/(2\pi)\). Hard formation light, emitted when the trapping surface forms, then breaks this agreement: growth at \(2r_s\) levels at 1.28 to 1.40 nats per sector over 60 to 100, about half of it excess over the held vacuum, with the late level's mechanism and its size or trapped-region dependence uncomputed.
```

### 10.174 Sec. V C: held-field drain and tunnelling paragraphs before merging and shortening (structural pass E payment P6; the level-pair width and the first-return time moved to the reference)
```latex
% provenance: holds at 15/30: v6kA_c15/c30 (page_history_v6k, 8192 sites, 300 r_s), plateaus from v3b holds <internal>/PAGE_v41_part3.md:12.
% provenance: hold 100->500: v6s_B2hold (<internal>/v6s_B2hold), <internal>/B2_hold_drain_readout_v2.2.md.
% provenance: pocket width 5.65e-6/a and isolation 0.056/0.147: PK5 routes (<internal>/PAGE_PK5_R1_v1_pocket_validation.csv, <internal>/PK5_R2_busi_v1.md); landed 73da652e7d6.
On held fields the flow settles to a slow drain of order the Hawking comparison case. It is a drain, not radiation. Holding from 100 to 500
preserves \(E_{\rm inv}\) to seven digits, and at every \(10r_s\)
checkpoint from 140 the \(k=1\) per-chain flow is 0.9--1.5
\(L_H(T_H)\) while the cut excess grows by 1.18--1.28 nats (hold plateaus in Ref.~\cite{McKinneyCode2026}).
```
Correction (lead inbox 90): in the sentence above the growth of 1.18--1.28 nats belongs to the whole hold, from 100 to 500, and is per chain; the printed sentence now reads "and over the whole hold, from 100 to 500, the cut excess grows by 1.18--1.28 nats per chain" (interior table md5 45ab942d, readout md5 e7c4310e, see the note under 8.17).

### 10.175 Sec. V C: tunnelling reading before merging (with 10.174)
```latex
Read as tunnelling, the held-field drain comes from one quasi-bound \(k=1\) level pair of width \(5.65\times10^{-6}/a\), e-folding time about \(3\times10^3r_s\), so that the hold spans about 0.14 e-fold and neither the line nor its proximity to \(L_H(T_H)\) has thermal meaning. Its isolation is not verified, since the width-limited drain statistics pass in both calculations but the pair's residue subspaces fail two stability budgets in the second (values in Ref.~\cite{McKinneyCode2026}), and channel \(k=2\) carries no such drain, the first return lying near 1000 and none of these stored slices completing the drain.
```

### 10.176 Sec. III C: eta(0) and eta(0.2) numerical paragraphs before shortening (structural pass, payment P7; the edge-direction failure, the logarithm and extrapolation remark, and the [110] interval moved to the reference)
```latex
\fnd{The numerical estimates of \(\eta(0)\) agree, but only the
convexity bound is rigorous: the two exact axis values give
\(0.8164<\eta(0)<1.0203\). A 205-direction interpolation sandwich
gives \(0.91905664<\eta(0)<0.91929748\), numerical evidence under an
empirical quadratic-convergence premise that its 32 edge directions
fail. A ten-radius ball fit estimates \(\eta(0)=0.9191\) with
spread 0.0017, not an error bound, establishing neither the logarithm
nor the extrapolation (comparison value and plateau remark in
Ref.~\cite{McKinneyCode2026}).}
```

### 10.177 Sec. III C: eta(0.2) paragraph before shortening (with 10.176)
```latex
At \(ma=0.2\) the same chain gives \(\eta(0.2)\) near 0.887:
interval-arithmetic bounds on \(\eta_{100}\) and \(\eta_{111}\)
with convexity establish \(0.7883<\eta(0.2)<0.9831\), and a
22-direction interpolation bound and a paired ball fit, numerical
evidence, give 0.884 to 0.888 (directional values, the measured
[110] interval, and the intermediate bound in
Ref.~\cite{McKinneyCode2026}).
```

### 10.178 Sec. III F: axial delta_T paragraph before shortening (structural pass, payment P8; the fixed-ratio qualifier and the averaging-shift factor moved to the reference)
```latex
On the exact exterior, axial endpoint modes at fixed \(\omega/\kappa_u\), \(m/\kappa_u\) away from threshold have \(\delta_T=(\kappa_ua)^2[5/8+(\omega/\kappa_u)^2-\tfrac34(m/\kappa_u)^2]+o((\kappa_ua)^2)\), formally warming if unit-flux channels match to relative error \(o((\kappa_ua)^2)\), unproved. The computed histories cannot resolve this coefficient, since their averaging shift \(9(\sigma/a)^2(\kappa_Ha)^2\) exceeds the axial term and they lack a full vacuum shell (factors in Ref.~\cite{McKinneyCode2026}), so they test temperature only at order one. No spherical correction is established, and it need not share the axial sign.
```

### 10.179 Sec. III E: full-vacuum-shell parenthetical before shortening (structural pass C)
```latex
At fixed \(E\), the vacuum exterior beyond averaging support and,
where contained in it, the trapping radius \(r_s=2GE\) of
\(N+v=1-\sqrt{r_s/r}\) are independent of \(\ell\), and averaging
preserves \(E=a^3\sum_x\rho_x\) for fixed bare sources with
\(J=0\), \(N=1\), and a regular centre. The back-averaged flow of Eq.~\eqref{eq:dr-boundary-variation}, for \(r\gg\ell\) with exterior averaging support, is \(W_\ell^\dagger v=v[1-9\sigma^2/(8r^2)]+O(|v|(\ell/r)^4)\). At a simple zero \(R\) of \(N+v\), a trapping radius, the rows give the surface slope
\(\kappa_R=\partial_r(N+v)|_R=(N_R/2R)[1-8\pi GR^2(\rho_R-J_R)]\),
\(1/(4GE)\) in the vacuum exterior. On a full vacuum shell
(a vacuum exterior from the trapping radius past the averaging support to the charge-reading shell, Sec.~\ref{sec:results-lattice}'s charge-reading radius),
this back-averaged flow with \(\sigma^2\) the kernel's per-axis variance puts the matter's own trapping radius and slope, read in the
fields the matter sees, at
\(r_{H,m}/r_s=2r_s\kappa_m=1-\tfrac94\sigma^2/r_s^2+O(\sigma^4/r_s^4)\),
the geometric trapping radius staying at \(r_s\).
```

### 10.180 Sec. VI A: exact dust ray-map paragraph before shortening (structural pass D, plan item D.5a; the exact map, the leading-logarithm duration, the C_c range, and the size-64 segment statements are kept, the s_c equation, F(s), the three C_c values, and the e-fold formula moved to the reference)
```latex
For Eq.~\eqref{eq:dr-crossing-ray}'s axial symbol at continuum
momentum \(\omega/(1+w)\), cubic and linear terms agree at
\(\alpha_s^2=\delta_d=(2\chi q)^{2/3}\), \(\alpha_d=\delta_d^{1/2}\),
with reduced wavelength \(\lambda_h=a/\alpha_d\).
This validity scale is not small at the computed sizes and is neither
a stopping surface nor a maximal redshift. This section's series
require \(1\gg\alpha_s^2\gg\max(\delta_d,\sigma^2/r_s^2)\).
The window ends where the sine dispersion is no longer small,
\(\alpha_s=\alpha_d\) in Eq.~\eqref{eq:sc-approach}, with
\(2\chi q=2a\omega\). The vacuum exterior \eqref{eq:dr-exterior},
incoming sea, and dispersion fix its leading logarithm, and its
\(O(r_s)\) constant is a functional of the complete incoming ray map,
through the interior and out, and of the peeling-rate tolerance. The Schwarzschild-time window \(T_d\) has half the logarithm and its own constant.
% provenance: 0.144 = (4/3) x 0.1079 and 0.50 = 0.144 ln 32 (arithmetic on Eq. sc-luminosity); C_c = -7.63/-4.43/-1.38 from the exact dust ray map (icloud window, ff3fd92bdee): producer not located.
% source: C_c = -7.63, -4.43, -1.38: SOLVEB check_WEB40I_ray_window_v1.py (md5 0811daec), scalar_and_inherited.md (c066d4d4) :14-16, FABLE_web40i_x1fix_v1.md (eec34eba) :14; PROVENANCE_TRACE_v1 (317c6d92), lead inbox 69.
On the dust solution the map is exact: with \(s=\sqrt{R_b/r_s}\),
a ray entering the ball at \(s+1\) leaves it at \(s\), so
\(dU/du=(s+1)^3(s-1)/[s^3(s+2)]\) and
\(\kappa_u/\kappa_H=6/[s^4(s+1)(s+2)]\). The segment on which
\(\kappa_u\) lies within a fraction \(c\) of \(\kappa_H\) starts at
\(s_c\), \(s_c^4(s_c+1)(s_c+2)=6/(1-c)\), and ends at
\(s_d=(1-\delta_d)^{-1/2}\). With \(F(s)=-\tfrac23(s^3-1)-s^2-2s-2\ln(s-1)\) its retarded duration is
\(r_s[F(s_d)-F(s_c)]\) when \(s_d<s_c\) and zero otherwise. For a
nonempty window, its asymptotic duration is
\(\tfrac43r_s\ln[1/(2\chi q)]+C_cr_s+O(r_s\delta_d)\) with
\(C_c=\ln4-3-F(s_c)=-7.63\), \(-4.43\), and \(-1.38\) at \(c=0.05\),
0.2, and 0.5, and it carries \(\ln[(dU/du)(s_c)/(dU/du)(s_d)]\)
e-folds of peeling. At \(r_s/a=64\) and \(q=1\) the 5 and 20 percent
segments are empty, the 50 percent segment lasts \(2.76r_s\) with 0.89
e-folds, and no packet resolving its own frequency fits inside it.
```

### 10.181 Sec. VI C: solar-mass evaluation list before shortening (structural pass D, plan item D.5b; m/T_H and the chi scaling exponents moved to the reference)
```latex
% provenance: solar-mass column and r_s/a=64 column are closed-form evaluations (alpha_d=(2chi q)^{1/3}, m/T_H=4 pi m r_s, t_*/r_s=(r_s/a)^2/(6 g L)); cross-listed in <internal>/ASYMPTOTICS_checks_v1.9.md.
The radiation budget follows from the exterior temperature and exclusion, the Page bound from purity. The Hawking comparison case has an adiabatic evaporation scale \(E/L\), \(r_s/(E/L)=\tfrac12|\dot\kappa_H|/\kappa_H^2=2g\mathcal L\chi^2\), using quasi-steady emission at the instantaneous mass, not a steady supply. At \(g=\tfrac14\), \(a=2\ell_P\), one solar mass has
\(\chi=5.47\times10^{-39}g^{-1/2}(M_\odot/E)=1.09\times10^{-38}\),
\(r_s=2.953\,\mathrm{km}\) and \(T_H=6.17\times10^{-8}\,\mathrm K\).
Evaluated, not measured, at one solar mass with \(q=1\) and \(m=0\)
unless stated: the validity scale \(\alpha_d\) is
\(2.80\times10^{-13}\), the formal axial correction \(\delta T/T_H\)
is \(1.39\times10^{-76}\), \(m/T_H\) at \(\zeta=0.2\) is
\(2.30\times10^{38}\), and if Eq.~\eqref{eq:sc-luminosity} holds
throughout evaporation, \(t_*/r_s\) is \(1.06\times10^{78}\),
\(t_*=3.3\times10^{65}\) years. At fixed \(g,q,\zeta\) these scale as
\(\chi^{1/3}\), \(\chi^2\), \(\chi^{-1}\), and \(\chi^{-2}\), changes
of 12 to 73 orders of magnitude from the computed size \(r_s/a=64\),
where \(\alpha_d=0.315\) (the size-64 values in
Ref.~\cite{McKinneyCode2026}).
```

### 10.182 Sec. VI C: bounded-band burst bound before shortening (structural pass D, plan item D.5c; the bound formula and the channel count moved to the reference, the 2e-72 value and the escape hedge kept)
```latex
At unit lapse the flow's redshift near the trapping radius permits soft output but establishes no spectral weight, count, or rate (Eq.~\eqref{eq:dr-crossing-ray}). For a channel count bounded in \(\chi\), the channel-capacity bound permits \(p\ge2\) in the slowly evolving object's assumed luminosity \(L\propto\chi^p/a^2\) without establishing saturation or any nonzero rate.
% provenance: B_12 = 1.029 at x in [25,100] and the 2e-72 bound are closed-form evaluations of the displayed inequality (n_ch 192, Delta t 40 r_s, chi 1.09e-38).
Exclusion bounds bounded-band bursts even with frequency coherence: for outgoing channels in the band \(0\le\omega_1<\omega_2\), \(\omega_i=x_iT_H\), \(n_{\rm ch}\) channels carry in an arrival interval \(\Delta t\) at most \(B_{12}n_{\rm ch}(\Delta t/r_s)(x_2^2-x_1^2)g\chi^2/(32\pi^3)\) of \(E\), \(B_{12}=1\) for frequency-diagonal occupations and 1.029 for \(25\le x\le100\) with coherence (kernel eigenvalues in Ref.~\cite{McKinneyCode2026}). At one solar mass with \(n_{\rm ch}=192\) (\(k=\pm1,\pm2\)) and \(\Delta t=40r_s\) this is near \(2\times10^{-72}\), escaped only by hardening to lattice energies, more channels, or a longer arrival.
```

### 10.183 Sec. VI D: Table I, the rigorous interval bounds on the outer trapping radius energy debit (structural pass D, plan item D.5e, lead-decides, applied as the announced default; the text keeps the sign statement under both interpolants and points to the reference)
```latex
\begin{table}[t]
\caption{Rigorous interval bounds (second calculation, Ref.~\cite{McKinneyCode2026}) on the outer trapping radius's energy debit \(H_R\), units \(10^{-4}\), under the P and H interpolants.}
\label{tab:root-debit}
% P column: busi rdcheck_v3 Route B outer-root H_R (rdcheck_v3 a4c1bf10, all five clocks passing every binding gate, exp verified at 160 bits), outward decimals, in units of 1e-4 by exact decimal shift; cleared (rdcheck_v3, reviewer B round 4, lead inbox 92); H column and inner roots: cleared (reviewer B rdprint_check4 4fd1d257, values unchanged from busi's rdcheck_v7 table 78a36851, lead inbox 106)
% H column: busi cubic-Hermite outer-root H_R (rdcheck_v5 ba091c48 on interp_v5 ae4386fa, 5/5 pass, both interpolants, inner and outer roots, exhaustive root counts, exactly two roots per record), outward decimals, in units of 1e-4 by exact decimal shift; cleared: all 20 intervals (outer and inner roots, P and H, five clock records) cleared to print by reviewer B rdprint_check4 (4fd1d257), values unchanged from busi's rdcheck_v7 table (78a36851), lead inbox 106
\footnotesize
\begin{ruledtabular}
\begin{tabular}{lcc}
\(t/a\) & P & H \\
1096 & \([-69.15718140,\,-68.99568799]\) & \([-69.41016303,\,-69.24819924]\) \\
1100 & \([138.8348135,\,139.0207002]\) & \([139.0982565,\,139.2860939]\) \\
1104 & \([422.4865081,\,422.7061558]\) & \([420.7226502,\,420.9322324]\) \\
1108 & \([810.5060050,\,810.7089990]\) & \([810.9657400,\,811.1841460]\) \\
1112 & \([1373.011735,\,1373.190276]\) & \([1371.288298,\,1371.497756]\) \\
\end{tabular}
\end{ruledtabular}
\end{table}
```
Correction (lead inbox 80): the P-interpolant endpoints above are busi's corrected Route B values from the lib_v5 re-run (the delivered values had a float band subtraction before interval wrapping), converted exactly from H_R at stored steps 17536, 17600, 17664, 17728, 17792 to the table's units of 1e-4; the corrected intervals contain the delivered ones, outward by at most 1.8e-9 in H_R, and agree with them at the three decimals the table printed (delivered: [-69.157, -68.996], [138.835, 139.021], [422.487, 422.706], [810.506, 810.709], [1373.012, 1373.190]). The H-interpolant column carries busi's corrected lib_v5 values for the outer trapping radius as well (busi commit 5a426b79407, posted in the busi STATUS under the heading for lead item 1118 and copied from it, lead inbox 82), converted exactly to the table's units, every interval containing the delivered one, with shifts at most 1.04e-9 in H_R at the clock records (delivered H: [-69.410, -69.248], [139.098, 139.286], [420.723, 420.932], [810.966, 811.184], [1371.288, 1371.498]) and at most 3.1e-12 at the inner trapping radius; the Hermite neighbour-cell flags in the text come from the delivered runs, the corrected jsons not carrying them. Known-answer tests of the re-run: step 13752 [-33.4414312143, -33.4404485003], step 13824 [-1.6133522467, -1.6133333183]. Uniqueness at the five slices now holds on two independent routes (icloud interval-Newton, busi Krawczyk kraw_v7 with nested boxes). Round 3 (lead inbox 85): the Krawczyk route is not cleared (kraw_v7 checked the old lib_v2 box, the endpoint export rounded to nearest, not outward, evaluator modules unreviewed), so uniqueness is credited to the interval-Newton certificate alone pending the re-check, and the printed five-slice wording stays softened. lib_v6 (lead inbox 88, copied from the busi STATUS block "root-debit printing linkage CLOSED", 05:46Z 3 Nov): lib_v6 a98a2fd4 with nextafter-outward endpoint export, run_v4 b44efc2d, interp_v3 99d42743, rdcheck_v2 7001e003, all gates binding, 7/7 pass; Route B outer-root H_R (outward): 17536 [-0.006915718139860501, -0.006899568799552365] q 0.805 (json md5 cc5a563d); 17600 [0.013883481355555418, 0.013902070011011295] q 0.807 (json md5 f9c01946); 17664 [0.04224865081216819, 0.0422706155790809] q 0.810 (json md5 c62a92ad); 17728 [0.08105060050757842, 0.08107089989645704] q 0.812 (json md5 e2b98f87); 17792 [0.137301173590925, 0.13731902753279937] q 0.814 (json md5 a87f775d); KAT 13752 [-33.441431214316516, -33.44044850028014], q 0.841 (877413d5); KAT 13824 [-1.6133522467157269, -1.6133333182809226], q 0.838 (3061981b); every new interval contains the delivered one, shifts at most 1.8e-9, no sign change; supersedes the lib_v5 values, which agree with these at the six decimals the P column printed before; the P column now carries the lib_v6 strings shifted exactly to the table's units; run_v4, interp_v3, and lib_v6 are not yet reviewer B-reviewed (round 4 pending), so both columns remain provisional. rdcheck_v3 (lead inbox 92, copied from the busi STATUS block "root-debit printing items (1)-(4) DONE, 5/5 PASS (rdcheck_v3 a4c1bf10)", 06:29Z 3 Nov): all five clocks pass every binding gate, exp verified at 160 bits, outward decimals 17536 [-0.006915718140, -0.006899568799] (json md5 26e6da9b); 17600 [0.01388348135, 0.01390207002] (json md5 8e9aff94); 17664 [0.04224865081, 0.04227061558] (json md5 a9efac01); 17728 [0.08105060050, 0.08107089990] (json md5 8444f018); 17792 [0.1373011735, 0.1373190276] (json md5 a7b05aa8); the P column now carries these strings shifted exactly to the table's units and is cleared (reviewer B round 4, lead inbox 92); the cubic-Hermite column stays provisional. Hermite path closed (lead inbox 95, copied from the busi STATUS block "Hermite path closed: 5/5 PASS, both interpolants, inner and outer roots, exhaustive root counts", 06:44Z 3 Nov; rdcheck_v5 ba091c48 on interp_v5 ae4386fa, json md5s d11d0989 a7a6c1ce f532d0e8 8dc49af7 9ecb6d3f): all rdprint gates, interp box bitwise identical, every H interval certified, exhaustive Hermite count with 0 unresolved of 32,957 cells, exhaustive linear count with 0 uncertain nodes, root list identical, exactly two roots per record under both interpolants; outer cubic-Hermite H_R (outward decimals) 17536 [-0.006941016303, -0.006924819924]; 17600 [0.01390982565, 0.01392860939]; 17664 [0.04207226502, 0.04209322324]; 17728 [0.08109657400, 0.08111841460]; 17792 [0.1371288298, 0.1371497756], now the H column by exact decimal shift; the inner-root intervals at the same steps, not printed in Table I: 17536 P [-0.07328744796, -0.07326132949] H [-0.07386858904, -0.07385332009]; 17600 P [-0.07749536256, -0.07747730649] H [-0.07611118308, -0.07609750922]; 17664 P [-0.07861446867, -0.07858957663] H [-0.07850088804, -0.07848842153]; 17728 P [-0.08018562431, -0.08016764453] H [-0.08105693244, -0.08104546762]; 17792 P [-0.08473741459, -0.08472026938] H [-0.08381312706, -0.08380303155]; both columns cleared (both interpolants, lead inbox 95), the square-root validation (rdcheck_v6) and reviewer B's confirmation on the complete packet pending for the push.

### 10.184 Sec. VI A: validity-window paragraph before shortening (structural pass D; the 2 chi q = 2 a omega identity and the "through the interior and out" clause folded)
```latex
For Eq.~\eqref{eq:dr-crossing-ray}'s axial symbol at continuum
momentum \(\omega/(1+w)\), cubic and linear terms agree at
\(\alpha_s^2=\delta_d=(2\chi q)^{2/3}\), \(\alpha_d=\delta_d^{1/2}\),
with reduced wavelength \(\lambda_h=a/\alpha_d\).
This validity scale is not small at the computed sizes and is neither
a stopping surface nor a maximal redshift. This section's series
require \(1\gg\alpha_s^2\gg\max(\delta_d,\sigma^2/r_s^2)\).
The window ends where the sine dispersion is no longer small,
\(\alpha_s=\alpha_d\) in Eq.~\eqref{eq:sc-approach}, with
\(2\chi q=2a\omega\). The vacuum exterior \eqref{eq:dr-exterior},
incoming sea, and dispersion fix its leading logarithm, and its
\(O(r_s)\) constant is a functional of the complete incoming ray map,
through the interior and out, and of the peeling-rate tolerance. The Schwarzschild-time window \(T_d\) has half the logarithm and its own constant.
% provenance: 0.144 = (4/3) x 0.1079 and 0.50 = 0.144 ln 32 (arithmetic on Eq. sc-luminosity); C_c = -7.63/-4.43/-1.38 from the exact dust ray map (icloud window, ff3fd92bdee): producer not located.
% source: C_c = -7.63, -4.43, -1.38: SOLVEB check_WEB40I_ray_window_v1.py (md5 0811daec), scalar_and_inherited.md (c066d4d4) :14-16, FABLE_web40i_x1fix_v1.md (eec34eba) :14; PROVENANCE_TRACE_v1 (317c6d92), lead inbox 69.
On the dust solution the map is exact: with \(s=\sqrt{R_b/r_s}\),
a ray entering the ball at \(s+1\) leaves it at \(s\), so
\(dU/du=(s+1)^3(s-1)/[s^3(s+2)]\) and
\(\kappa_u/\kappa_H=6/[s^4(s+1)(s+2)]\). The segment on which
\(\kappa_u\) lies within a fraction \(c\) of \(\kappa_H\) starts at
\(s_c\), \(s_c^4(s_c+1)(s_c+2)=6/(1-c)\), and ends at
\(s_d=(1-\delta_d)^{-1/2}\). With \(F(s)=-\tfrac23(s^3-1)-s^2-2s-2\ln(s-1)\) its retarded duration is
\(r_s[F(s_d)-F(s_c)]\) when \(s_d<s_c\) and zero otherwise. For a
nonempty window, its asymptotic duration is
\(\tfrac43r_s\ln[1/(2\chi q)]+C_cr_s+O(r_s\delta_d)\) with
\(C_c=\ln4-3-F(s_c)=-7.63\), \(-4.43\), and \(-1.38\) at \(c=0.05\),
0.2, and 0.5, and it carries \(\ln[(dU/du)(s_c)/(dU/du)(s_d)]\)
e-folds of peeling. At \(r_s/a=64\) and \(q=1\) the 5 and 20 percent
segments are empty, the 50 percent segment lasts \(2.76r_s\) with 0.89
e-folds, and no packet resolving its own frequency fits inside it.
```

### 10.185 Sec. VI B: late-stage paragraph before the retarded-coordinate formula moved to the reference (structural pass D)
```latex
For \(1\gg\delta/r_s\gg\max\{(2\chi q)^{2/3},\sigma^2/r_s^2\}\),
\(\delta=R_e-r_s\), an outgoing long-wave front in the exterior of Eq.~\eqref{eq:dr-exterior}, emitted from the surface at \((t_e,R_e)\), reaches a static receiver at retarded time \(u=t_e+2r_s\ln(2r_s/\delta)-3r_s+O(\delta)\), with \(u=t-r-2\sqrt{r_sr}-2r_s\ln(\sqrt{r/r_s}-1)\) constant along the ray in Sec.~\ref{sec:star}'s normalization. Since \(\delta=t_c-t_e+O((t_c-t_e)^2/r_s)\), \(R_e(u)-r_s=2e^{-3/2}r_se^{-(u-t_c)/(2r_s)}[1+O(e^{-(u-t_c)/(2r_s)})]\), launch entering only through \(t_c\), and the surface's emission is received at \(g_\infty=N+v=1-\sqrt{r_s/R_e}\), \(\ln(1/g_\infty)=(u-t_c)/(2r_s)+\tfrac32+O(\delta/r_s)\), with Eq.~\eqref{eq:sc-ray-temperature}'s exponent \(\kappa_H\). This gives the surface's approach, not the emitting shell's thickness or distribution. Interior light escapes this fixed-charge continuum comparison solution only if it crosses the surface before \(t_c\). Afterwards the surface lies inside the trapping radius and outgoing long waves move inward, so none from filling at \(R_\pi\) reaches any \(u\). At fixed \(\chi\) these relations are not uniform at infinite reception time: mode \(q\) reaches the prescribed dispersive cutoff at \(1-r_s/R_e=\delta_d\), where \(u-t_c=\tfrac43r_s\ln[1/(2\chi q)]+O(r_s)\) and \(1/g_\infty\simeq2/\alpha_d^2\), beyond which propagation follows Eq.~\eqref{eq:dr-crossing-ray}'s lattice branch with evolving sources, not followed here. At a smooth, simple, held trapping radius the radial massless operator's near-surface limit gives Fermi-shaped two-channel transmission at \(\kappa_R/(2\pi)\), a local scattering result that fixes neither incoming occupation, sustained supply, escape yield on evolving fields, nor heavy-information transfer. Eq.~\eqref{eq:star-crossing-delay} is a delay, not a release law.
```

### 10.186 Sec. VI D: depletion display made inline (structural pass D; content kept, label eq:page-effective-depletion retired, it was unreferenced)
```latex
% assumption: the depletion, marginal-entropy and minimum channel laws are assumed; the Page time and maximum of Eqs. page-effective-curve and page-effective-maximum follow from these assumptions and the first-law normalization, not from the object's dynamics.
% risk: high
% GR slippage: no
% fold into text: no; stated at every step of this subsection
% provenance: 0.6211/0.5105, 0.5755/0.4773, [0.5323,0.6146]: page_curve_amps R1 table (<internal>/PAGE_FIGURE_R1_v1.md:159-161), agreed by R2 (<internal>/PAGE_FIGURE_COMPARE_v1.md).
Assume quasi-steady \(L=\mathcal L/R^2\),
\(dE_R/du=-L\), and \(dS_{\rm marg}/du=\mathcal RL/T_H\).
With \(t_*=R_0^3/(6G\mathcal L)\) and \(y\equiv(E_R/E_0)^2\), integration gives
\begin{equation} \begin{aligned} R^3&=R_0^3-6G\mathcal L\,t,\qquad y=(1-t/t_*)^{2/3},\\ S_{\rm marg}/S_0&=\mathcal R\,(1-y), \end{aligned} \label{eq:page-effective-depletion} \end{equation}
\(A_R/(4G)=S_0y\) is retained first-law entropy. Eq.~\eqref{eq:page-conditional}'s budget starts at activation.
Subadditivity gives \(S_{\rm esc}\le S_{\rm marg}\). Purity bounds it by the store's capacity \(fA_R/(4G)\) plus other retained entropy.
The assumed channel law gives
\begin{equation}
S_{\rm esc}/S_0=\min\{\mathcal R(1-y),\;f\,y\}+o(1),
\label{eq:page-effective-curve}
\end{equation}
uniformly through turnover, without a numerical remainder bound. Thermal marginals supply only an upper bound (Fig.~\ref{fig:page-conditional}).
\todo{the curve from
a realized channel: saturated bounds, bounded remainder}
The branch intersection is
\begin{equation}
\frac{t_P}{t_*}=1-\Big(\frac{\mathcal R}{\mathcal R+f}\Big)^{3/2},
\qquad
\frac{S_P}{S_0}=\frac{f\,\mathcal R}{\mathcal R+f}.
\label{eq:page-effective-maximum}
\end{equation}
```

### 10.187 Sec. VI D: Page-curve and Page-time displays before merging into one boxed display (structural pass D, plan item D.6; label eq:page-effective-maximum retired, its one reference retargeted)
```latex
% assumption: the depletion, marginal-entropy and minimum channel laws are assumed; the Page time and maximum of Eqs. page-effective-curve and page-effective-maximum follow from these assumptions and the first-law normalization, not from the object's dynamics.
% risk: high
% GR slippage: no
% fold into text: no; stated at every step of this subsection
% provenance: 0.6211/0.5105, 0.5755/0.4773, [0.5323,0.6146]: page_curve_amps R1 table (<internal>/PAGE_FIGURE_R1_v1.md:159-161), agreed by R2 (<internal>/PAGE_FIGURE_COMPARE_v1.md).
Assume quasi-steady \(L=\mathcal L/R^2\),
\(dE_R/du=-L\), and \(dS_{\rm marg}/du=\mathcal RL/T_H\).
With \(t_*=R_0^3/(6G\mathcal L)\) and \(y\equiv(E_R/E_0)^2\), integration gives
\begin{equation} \begin{aligned} R^3&=R_0^3-6G\mathcal L\,t,\qquad y=(1-t/t_*)^{2/3},\\ S_{\rm marg}/S_0&=\mathcal R\,(1-y), \end{aligned} \label{eq:page-effective-depletion} \end{equation}
\(A_R/(4G)=S_0y\) is retained first-law entropy. Eq.~\eqref{eq:page-conditional}'s budget starts at activation.
Subadditivity gives \(S_{\rm esc}\le S_{\rm marg}\). Purity bounds it by the store's capacity \(fA_R/(4G)\) plus other retained entropy.
The assumed channel law gives
\begin{equation}
S_{\rm esc}/S_0=\min\{\mathcal R(1-y),\;f\,y\}+o(1),
\label{eq:page-effective-curve}
\end{equation}
uniformly through turnover, without a numerical remainder bound. Thermal marginals supply only an upper bound (Fig.~\ref{fig:page-conditional}).
\todo{the curve from
a realized channel: saturated bounds, bounded remainder}
The branch intersection is
\begin{equation}
\frac{t_P}{t_*}=1-\Big(\frac{\mathcal R}{\mathcal R+f}\Big)^{3/2},
\qquad
\frac{S_P}{S_0}=\frac{f\,\mathcal R}{\mathcal R+f}.
\label{eq:page-effective-maximum}
\end{equation}
```

### 10.188 Sec. III A: dispersion-speed display made inline (structural pass D payment; label eq:dr-speeds retired, its one reference reworded)
```latex
The light cone comes first, since it fixes the metric. Flat-sea statements hold on the infinite lattice or on a periodic lattice unless otherwise specified. On uniform fields the bands of
\(h=Nh_0+v\cdot Q\) are
\(\omega_\pm(p)=v\cdot q(p)\pm N\varepsilon(p)\). Near a valley,
\(p=p_\star+k\) with \(a|k|\ll1\),
\begin{equation}
\begin{aligned}
\varepsilon^2&=m^2+|k|^2-\frac{a^2}{3}\sum_i k_i^4+O(a^4|k|^6),\\
q_i&=k_i-\frac{2a^2}{3}k_i^3+O(a^4k_i^5),
\end{aligned}
\label{eq:dr-cone-expansion}
\end{equation}
and for \(m=0,N=1,v=0\), \(k=k\boldsymbol n\), \(A_4=\sum_i n_i^4\),
\begin{equation}
\begin{aligned}
\frac{\varepsilon}{k}&=1-\frac{(ak)^2}{6}A_4+O((ak)^4),\\
\boldsymbol n\cdot\nabla_k\varepsilon&=1-\frac{(ak)^2}{2}A_4+O((ak)^4).
\end{aligned}
\label{eq:dr-speeds}
\end{equation}
With \(A_4=1,1/2,1/3\) along [100], [110], [111], anisotropy starts
at \((ak)^2\).
```

### 10.189 Sec. III C: flow-threshold display made inline (structural pass D payment; label eq:dr-flow-threshold retired, the Appendix A reference reworded)
```latex
The flat sea is stable under uniform flow up to an exact threshold: for homogeneous preparation \(P_0=P_-(h_0)\), \(P_-\) the negative spectral projection, the two-step current \(Q_i\) commutes with \(P_0\), so uniform fields preserve the sea and its cut entropies at every flow speed, and the ground projector \(P_g=P_-(h)\), one comparator projector against which cut entropies are compared, equals \(P_0\) for \(|v|<N\) and, for \(j=1,2,3\) equal nonzero Cartesian flow components, below the exact bulk threshold
\begin{equation}
\begin{aligned}
w_c^{(j)}(am)&=\sqrt{1+(am)^2/j}+am/\sqrt j,\\
|v|/N<w_c^{(j)}(am)&\ \Longrightarrow\ P_g=P_0;
\end{aligned}
\label{eq:dr-flow-threshold}
\end{equation}
the massless threshold is 1 in every direction, with sign-reversed states for
\(|v|>N\), and no pointwise speed test fixes the ground projector on a nonuniform history.
```

### 10.190 Sec. III C: cut-entropy display made inline (structural pass D payment; label eq:dr-cut retired, the Appendix A reference reworded)
```latex
Cut entropies are functionals of the covariance alone. Quadratic
matter evolution preserves number-conserving Gaussian states. A region \(A\) with site projection \(\Pi_A\) has entropy,
with \(s(t)=-t\log t-(1-t)\log(1-t)\),
\begin{equation}
C_A=(\Pi_A C\Pi_A)|_A,\qquad S_A=\Tr s(C_A),
\label{eq:dr-cut}
\end{equation}
the entanglement entropy of a pure sea in
nats~\cite{Peschel2003,PeschelEisler2009}.
```

### 10.191 Sec. III D: filling-bound display made inline (structural pass D payment; label eq:dr-filling-bound retired, it was unreferenced)
```latex
The filling bounds the matter the collapse can compress. In the prescribed lattice frame \(N=1,v=0\), the
two positive-band spin states filled
to chemical potential \(\mu\) without antiparticle excitations have,
with \(\int_p=\int_{[-\pi/a,\pi/a)^3}d^3p/(2\pi)^3\),
\(f_\mu=\mathbf1_{\{\varepsilon<\mu\}}\) and
\(\varepsilon_{\max}=\sqrt{m^2+3/a^2}\), the capped density \(2\int_pf_\mu\) and
energy density \(2\int_p\varepsilon f_\mu\),
\begin{equation}
n_{\max}=\frac2{a^3},\qquad
\rho_{\max}=2\int_p\varepsilon\leq\frac{2\varepsilon_{\max}}{a^3},
\label{eq:dr-filling-bound}
\end{equation}
exclusion caps density and energy density, not pressure, which grows with \(\mu\) at fixed filling. It bounds the matter and, on regular cubic slices, the prepared-sea part of the central angular curvature's source:
Each bare particle energy is at most \(\varepsilon_{\max}\), and the prepared-sea normal-ordered part of \(\rho_x\), \(a^{-3}\Tr[(C-P_\star)e_x]\), is bounded by \(\bar\rho=(4/a^4)\sqrt{(ma)^2+3/2}\) for any state and preparation. The polarization moments add to it (Sec.~\ref{sec:singularity}).
```

### 10.192 Sec. IV A: hydrodynamic-test paragraph before removing the sentence that repeats Sec. II B (structural pass D payment)
```latex
% provenance: completions W3 (engine v36g) under <internal>/{w3_frozen-sources,w3_paired-avg,w3_point-core}, restart from P64_s4 T0011.00, arm 11.125.
% provenance: gate readings <internal>/PAGE_solved_history_v4.1.md:479-482; nsites 2048, tau 8, ell 4; held-source projection tol 1e-9.
The hydrodynamic test, the check that the fields stay smooth on the
averaging scale, first fails on the reference collapse in its
one-term form at \(T/r_s=11.75\), before formation. The test compares \(\sigma|H|\), \(H=-\partial_rv\),
and a six-term geometry diagnostic against 0.1 out to the dust-energy centroid, the dust's energy-weighted mean radius. Either failure ends the hydrodynamic description, the regime in which the laws were derived (Sec.~\ref{sec:model-dictionary}). The computation continues exactly on the same lattice equations with well-defined fields, but their reading as a geometry is no longer guaranteed, so readings past the failure are labelled statements about the lattice equations. The six-term test fails on every stored slice from trapping-radius-free \(T/r_s=12\) to 40.
Three completions restart at the reference collapse's \(T/r_s=11.0\) slice, activating at the first stored slice failing the test at or after 11.125: inside the failing radius they hold sources with charge projected onto the slice's, widen averaging to \(\sigma/a=8\), or point-sample core coupling (the last ends at 12.5).
```

### 10.193 Sec. IV A: 18-slice reference-law paragraph before shortening (structural pass D payment; "typically under 1 percent" and the row names folded)
```latex
% assumption: the reference collapse uses a held level-following projector and the three two-field histories (ON, OFF, C_M1nl23) a sea projector frozen at release with a counterterm relative to the release fields, neither being the declared reference of Eq. dm-reference; all collapse diagnostics inherit this provenance and are uncertified under the fourth declaration.
% risk: high
% GR slippage: no
% fold into text: no; stated with its todo
% provenance: 18-slice reference-law screen: <internal>/REF_SCREEN_routeA_solve_v1.3.md (3.2%, 0.6%); scripts <internal>/ref707_stage1_v2.py, ref707_spread_v1.py.
% provenance: members r11 (c19, sigma/a 4) and c27 (sigma/a 8), T/r_s 12-40; outputs <internal>/s2_{r11,c27}.json; folded in a81972666e9.
\label{sec:results-lattice}%
The stepping is exact for the state and re-solves the fields at every step: for step length \(\tau_j\), the accepted back-averaged fields
\(W_\ell^\dagger u_j\), \(u_j=(N_j,v_j)\), are held in
Eq.~\eqref{eq:dm-hamiltonian}'s chain operator, \(W_\ell^\dagger\) with the history's kernel, each covariance being conjugated by the exponential of that operator over the step, and the next fields solve the rows from the kernel-averaged moments of that state (Sec.~\ref{sec:model-dictionary}).
% provenance: per-history reference rule: <internal>/PERSIST_THREE_HISTORY_MANIFEST_v1.md (md5 e851799a, sections 0 and "Reference rule as the paper states it") and <internal>/HELDPROJ_CODEFACT_webgpt_v1.md (md5 75035019, Conclusion); reference collapse P64_s4 (321 slices) level-following, the three two-field histories ON, OFF, C_M1nl23 (159, 159, 255 slices) sea projector frozen at release plus counterterm, T_held = 0; lead inbox 68.
The reference collapse uses step-start covariance minus a level-following projector continued from the previous accepted fields to each trial field, called the fixed reference projector below. The three two-field histories of the Jacobian check in Sec.~\ref{sec:singularity} instead hold the sea projector frozen at release and add a counterterm relative to the release fields, with no continued projector. The level following is checked at single stored slices, not along the history, and no stored history is verified under Eq.~\eqref{eq:dm-reference}'s source prescription.
On all 18 stored interior slices of the reference collapse, at two
averaging widths over flat-slice lattice time \(T/r_s=12\)--40, its fixed-reference-projector histories' sources and fields
agree with Eq.~\eqref{eq:dm-reference}'s law far inside the step
controls' spread: the energy source differs by at most 3.2 percent of
the spread on the worst slice, typically under 1 percent, the clock and flow after re-solving both Gauss-law rows lie
within 0.6 percent of it, and the fixed reference projector's departure from
the prepared band is almost exactly the second-gradient polarization
that Eq.~\eqref{eq:dm-reference} subtracts. This is consistency at
the stored slices, not verification at all times.
\todo{evaluate the stored histories' energy and momentum source residuals under Eq.~\eqref{eq:dm-reference}, re-solve the narrow passages with the new sources under substep refinement until \(E_2-E_1-\int\Tr(\bar h\,dC)\) lies within the source and solve budget with no passage-associated jump, and re-evolve where residuals exceed the step controls' spread}
% method (moved from the text 2026-11-02, page budget): only specified Newton-solver controls use midpoint covariance, in two-stage source continuation.
```

### 10.194 Sec. V C: channel-tail and exterior-cut paragraphs before shortening (structural pass D payment; convergence conditions, site counts, and late rise rates moved to the reference)
```latex
% assumption: the light field is evolved as a test field (its gravity omitted) on stored back-averaged fields interpolated between records, on an interior past the hydrodynamic limit.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: tail statistic and floors: <internal>/page_v4_data_v1b.py (share=18c, l.155-157), <internal>/w1_postpass_v1f.py (md5 f6466903); readout <internal>/PAGE_solved_history_v4.1.md.
The computed collapse shows early outgoing light flow and a rising
exterior-cut entropy, an early radiation diagnostic. It does not demonstrate a Page curve, and the Page curve printed in
Sec.~\ref{sec:page-effective} is conditional on assumed laws. The
readings follow. The channel sums converge in principle but their
tail is not controlled numerically: weighted held-vacuum entropy
diverges with channel cutoff, the vacuum-subtracted excess converges
for temporally continuous, piecewise smooth fields, and the computed sums end at \(k=18\) (error floor, tail statistic, and channel-level details in Ref.~\cite{McKinneyCode2026}). The tail enters no result beyond this subsection.
```

### 10.195 Sec. V C: exterior-cut excess paragraph before shortening (with 10.194)
```latex
% provenance: light run v6o_B2 (page_history_v6o.py md5 4c793ad7) on B2 (4096 matter sites), 8192 light sites, interp_sub 8, k=1..18; <internal>/.
% provenance: maxima 9.504/8.738/8.259 nats at 100 (<internal>/figures_data_v1.log); readout <internal>/PAGE_solved_history_v4.1.md.
The exterior-cut excess reaches about nine nats by \(T/r_s=100\), a
few parts in \(10^4\) of \(A(E)/(4G)\): the test light field evolves
from its prepared sea at \(T=0\) to \(T/r_s=100\) on a separate
doubled-domain collapse, light gravity omitted and the interior past
the limit of its hydrodynamic description (site counts and appended
vacuum in Ref.~\cite{McKinneyCode2026}). The five cuts \(2,16,24,30,48r_s\) span a few parts in \(10^4\) of \(A(E)/(4G)\) at 100 (Fig.~\ref{fig:page-v4}, maxima and spans in Ref.~\cite{McKinneyCode2026}), while
field jumps can leave nonsummable reference mismatch (the trapping-radius slice, the non-monotonic \(30r_s\) excess, and the propagation and
purity readings in Ref.~\cite{McKinneyCode2026}). The end at 100 is
\(2.5\times10^{-4}t_*\) at the formation charge, a negligible
fraction of the evaporation time (late rise rates in
Ref.~\cite{McKinneyCode2026}).
```

### 10.196 Sec. V B: Hawking comparison case paragraph before removing the Page-maximum numbers duplicated in Sec. VI D (structural pass D payment; the depletion law sentence merged)
```latex
% assumption: the luminosity, number and entropy rates assume thermal supply at T_H from the inner port on the constant-charge exterior; nothing in the computed history supplies this occupation, and the model's own ray map gives an exponential segment of only about 0.2 thermal periods at the computed size and about 5 at solar mass (Sec. III F).
% risk: medium
% GR slippage: possible (Page's greybody fluxes on a fixed Schwarzschild background)
% fold into text: no; "thermally supplied" and "benchmark" are stated
% provenance: L=0.005237, S-dot=0.1079, R=1.639: page_curve_amps R1/R2 (C_0 0.00523731547127, D_0 0.107876529936), <internal>/PAGE_FIGURE_COMPARE_v1.md; earlier <internal>/.
% provenance: 0.621 S_0 at 0.511 t_*: Eq. page-effective-maximum at f=1 (PAGE_FIGURE_R1_v1.md:159). n-dot 0.01554 and 0.350 S_0 at 0.566 t_*: producer not located.
The Hawking comparison case, a thermally supplied exterior, feeds each
Dirac channel \(k\) of the
constant-charge exterior, Eq.~\eqref{eq:dr-exterior}, thermally at
\(T_H\) from its inner port, the outer port empty, with
unit-current transmission \(\Gamma_k(\omega)\). Summed over the
eight valleys, particles and antiparticles, \(|k|\le8\),
% source: n-dot = 0.01554/r_s: SOLVE/results/page/scat32/scat32_scattering.json (md5 443e546d) N_dot_rs, a1_rates_curve.py (192d242c) log :28; PROVENANCE_TRACE_v1 (317c6d92), lead inbox 69.
\(10^{-3}\le\omega r_s\le3\), the outgoing ports carry~\cite{Page1976}
\begin{equation}
L=\mathcal L/r_s^2,\quad
\dot n=0.01554/r_s,\quad
\dot S_{\rm marg}=0.1079/r_s,
\label{eq:sc-luminosity}
\end{equation}
with \(\mathcal L=0.005237\). Two independent numerical calculations reproduce the
luminosity and entropy-rate coefficients. The number rate is
reported but unverified. Neither the flux at infinity nor the
non-returned fraction \(f_{\rm nr}\) is read on the solved history.
If this quasi-steady emission persists and alone depletes \(E\),
the charge formally reaches zero at \(t_*=r_{s0}^3/(6G\mathcal L)\),
\(r_{s0}=2GE_0\). The leading first-law entropy \(S(E)=A(E)/(4G)\)
of Eq.~\eqref{eq:page-first-law} falls from \(S_0=S(E_0)\) as
\(S_0(1-t_\infty/t_*)^{2/3}\). Using the unrounded coefficients,
the flux carries \(\mathcal R=T_H\dot S_{\rm marg}/L=1.639\)
nats per unit of energy over \(T_H\)~\cite{Page1983}, giving the
marginal budget
\begin{equation}
S_{\rm marg}(t_\infty)=\mathcal R[S_0-S(E(t_\infty))].
\label{eq:page-conditional}
\end{equation}
Purity with a retained capacity at most \(S(E)\) bounds the escaped
entropy by \(\min(S_{\rm marg},S(E))\). Typicality on the counts
\(S_{\rm marg}/\ln2\) escaped and \(S(E)/\ln2\) retained gives the
maximum \(0.621S_0\) at \(0.511t_*\)~\cite{Page1993} (a typical fermionic Gaussian state on the same counts gives a lower, later maximum~\cite{BianchiHacklKieburg2021}, values in Ref.~\cite{McKinneyCode2026}): curves of the comparison case under assumed ensembles, not of the object.
```

### 10.197 Sec. IV D: between-slice bound paragraph before shortening (structural pass D payment, plan item P9 applied lightly; the four conditions kept, the error list, the tail clause, the norm clause, and the degenerate-centre clause shortened)
```latex
What remains is a between-slice bound. No regular case is known where the coupled evolution fails to continue, and no
invariant regular domain is proved. Continuation for all regular data is open, and the deciding two-field re-evolution is in
progress. The deciding calculation is a validated rigorous bound on the coupled trajectory on every consecutive slice interval, satisfying \(i\dot C=[\bar h(u),C]\) and \(F_u\dot u=-F_C\dot C\) for the boundary-normalized constraints \(F(C,u)=0\), \(u=(N,v)\), with the full reference response, connecting to the next interval, under four sufficient conditions. First, a solution rigorously bounded under Eq.~\eqref{eq:dm-reference} with the propagated sea, every active species and copy weight, and every initial-data, assembly, quadrature, and evolution error. Second, throughout the interval a regular inward branch with a positive margin \(\mathcal M(r,t)/r^3\ge\mu_{*,j}>0\) and Eq.~\eqref{eq:sg-current-mass} verified on the same bound with the full constraint mass, every tail and low-density cell included. Third, a uniform inverse bound on the full \(F_u\), reference response included, in fixed printed norms, which neither a clock bound nor the inverses verified at stored time slices supplies. Fourth, a self-closing bound on the fields, their regularity, and the reference derivatives that continues the same branch (full statement, centre handling, approximate-inverse recipe, and discrete-row clock estimate in Ref.~\cite{McKinneyCode2026}). The positive margin selects a sufficient nondegenerate domain, regular degenerate centres and mass-zero vacuum components needing separate matching and uniqueness arguments, and failure of a sufficient inequality means not verified, not singular. Stored fixed-reference-projector histories may seed a verification only with rigorously bounded defects under the printed law, otherwise re-evolution is required, and interpolating stored fields and propagating test matter is not this calculation. No complete between-slice verification under the printed law, or general bound on polarization moments over regular cubic slices, is established.
The leading laws supply no lattice-scale geometric law once
their gradient expansion fails, so a Pauli-supported interior
remains a possible outcome, not an assumption that closes the
continuation question.
```

### 10.198 Sec. IV D: stored-slice paragraph before removing the rounding-and-assembly clause (structural pass D payment)
```latex
Stored time slices show positive clocks and nonsingular implemented Jacobians at every sample, which tests the slices, not the trajectory between them: the reference collapse's stored clocks stay positive through \(T/r_s=40\) (sampled minimum 0.068), two reported independent no-density-floor current--mass scans of 894 stored slices in four histories give a clock floor \(e^{-50}\), far below the sampled minima, with normalization and centre coverage unspecified, and the implemented constraint Jacobian has verified inverse norm at most 2.90, with bounded rounding and assembly errors, on 573 stored slices in three histories, establishing neither exact trapping radii nor the intervening trajectory. The reference collapse uses the fixed reference projector without light, and the three two-field histories hold the sea projector frozen at release under different light rules (one with the block-form light counterterm whose constant-coefficient massless-chain endpoint fails through \(e_{11}\), Sec.~\ref{sec:results-lattice}), so these tests verify no trajectory under Eq.~\eqref{eq:dm-reference}, and the stored-slice agreement of Sec.~\ref{sec:results-lattice} remains a consistency check.
```

### 10.199 Bibliography: eprint fields removed from three published-article entries (structural pass D payment, the page lever recorded on 2 Nov; journal references unchanged)
```bibtex
@article{BianchiSmerlak2014, author = {Bianchi, Eugenio and Smerlak, Matteo}, title = {Entanglement entropy and negative energy in two dimensions}, journal = {Phys. Rev. D}, volume = {90}, pages = {041904}, year = {2014}, eprint = {1404.0602}, archivePrefix = {arXiv}, primaryClass = {gr-qc}}
@article{HolzheyLarsenWilczek1994, author = {Holzhey, Christoph and Larsen, Finn and Wilczek, Frank}, title = {Geometric and renormalized entropy in conformal field theory}, journal = {Nucl. Phys. B}, volume = {424}, pages = {443--467}, year = {1994}, eprint = {hep-th/9403108}, archivePrefix = {arXiv}}
@article{Will2014, author = {Will, Clifford M.}, title = {The Confrontation between General Relativity and Experiment}, journal = {Living Rev. Relativ.}, volume = {17}, pages = {4}, year = {2014}, eprint = {1403.7377}, archivePrefix = {arXiv}, primaryClass = {gr-qc}}
```

### 10.200 Appendix A.6: local gap and local area, the flow-clipped transfer estimate with Eq. B-transfer (structural pass F, final payment; replaced by a three-line statement, the display unreferenced since commit 086faf4b2f5)
```latex
Where a sign-reversed region is open inside the cut, the massive bound is
transferred from a clipped comparison. For \(\zeta>0\), fields obeying (H1)
and a finite sign-reversed region \(\mathcal U=\{x:|\bar v_x|>s_0\bar N_x\}\),
\(0<s_0<s_c\) with \(\mathfrak g_{s_c}(\zeta)>0\), the comparison
operator \(h^{\rm cmp}\) has the flow clipped smoothly to
\(|\bar v^{\rm cmp}_x|<s_c\bar N_x\) and unchanged where
\(|\bar v_x|\le s_0\bar N_x\). It differs from \(h\) only within two
sites of \(\mathcal U\) and has a gap \(\delta^{\rm cmp}\) by the
Cauchy step of A.5 and Eq.~\eqref{eq:B-gapcert} for \(\epsilon\)
small. For an orthogonal projection \(P\) commuting with \(h\),
\(0<\delta_1<\delta^{\rm cmp}\), and
\(r_b=\operatorname{rank}[(P-P_-(h))\mathbf1_{\{|h|\ge\delta_1\}}]<\infty\),
Ref.~\cite{McKinneyCode2026} constructs a nonincreasing
\(\mathcal E(d)\le C_{\mathcal E}e^{-\vartheta d}\) by localizing the
remaining projector difference near the clipping support, with a
reduced Combes--Thomas rate of the comparison operator and constants
that depend on the clipping support and the comparison gap and are
not uniform over growing sign-reversed regions. The resulting bound,
for sufficiently small \(\epsilon\) and every \(\varkappa\)-regular
finite cut whose complement is at distance at least \(d\) from
\(\mathcal U\), is
\begin{equation}
\begin{aligned}
 |S_A(P)-S_A(P_0)|&\le C^{\rm cmp}_{\rm loc}\varkappa A_{\rm lat}\epsilon^2(1+|\log\epsilon|)^4\\
 &\quad+r_b\log2+\mathcal E(d),
\end{aligned}
\label{eq:B-transfer}
\end{equation}
the last two terms being at most \((r_b+1)\log2\) for
\(d\ge d_0=\min\{d\ge0:\mathcal E(d)\le\log2\}\). Separate \(r_b\) occupation changes cost \(r_b\log2\), and unclipped \(h\) needs no
zero gap.

```

### 10.201 Appendix A.3: proof-steps sentence before shortening (structural pass F; the Combes-Thomas 1973, Bhatia 1997, and Rotfeld 1969 citations dropped with it, each cited only there)
```latex
\paragraph{A.3 The massive estimate.}
The corrector, weighted-remainder, and endpoint-entropy steps are
proved in Ref.~\cite{McKinneyCode2026}: Taylor expansion of
commutator kernels fixes \(P_1\) from \([h,P_0]\), gapped-path
Combes-Thomas bounds~\cite{CombesThomas1973} control \(R_2\) in an
exponentially weighted Schur norm, and a second-order trace Taylor
expansion~\cite{Bhatia1997} of the cross-block entropy with the
area-factor Schatten bound~\cite{Rotfeld1969} finishes the estimate.
```

### 10.202 Appendix A.1, A.2, A.4, A.5: sketch sentences before tightening (structural pass F; the closed-path expansion, the finite-volume limit, the rank and norm orders of the massless proof split, and the Plancherel clause folded)
```latex
% assumption: the sphere-coefficient and convexity proofs are delegated to the companion repository; the paper states hypotheses and sketches only, and the massive estimate's hypotheses (A1)-(A3) are not verified on the computed histories (Sec. III C).
% risk: low
% GR slippage: no
% fold into text: no
\paragraph{A.1 The sphere coefficient.}
Sketch (full proof, with evenness and positivity of
\(\eta_{\boldsymbol n}\), in Ref.~\cite{McKinneyCode2026}):
polynomial approximants to \(s\) with
error controlled by the operator Jensen
inequality~\cite{HansenPedersen1982} reduce the entropy through a
closed-path expansion to boundary deficits, whose
path-diameter-weighted summability gives continuity in the normal
and the ball limit by dominated convergence.
```

### 10.203 Appendix A.2: finite-volume-limit sentence before tightening (with 10.202)
```latex
\paragraph{A.2 Convexity of the planar coefficient.}
\(\Phi(\boldsymbol\xi)=|\boldsymbol\xi|\eta_{\boldsymbol\xi/|\boldsymbol\xi|}\),
\(\Phi(0)=0\), is convex, even, and positively homogeneous:
the support function \(\max_{x\in\mathsf K}x\cdot\boldsymbol\xi\)
of a compact cubic-symmetric convex body
\(\mathsf K\)~\cite{Schneider2014}. The finite-volume limit
(Ref.~\cite{McKinneyCode2026}) follows from Gaussian strong
subadditivity (Koteljanskii's inequality~\cite{Koteljanskii1950}),
which makes \(\tfrac12[S_A+S_{{\rm cube}\setminus A}-S_{\rm cube}]\)
submodular in \(A\), so that its Lov\'asz extension~\cite{Lovasz1983}
is convex and converges to \(\Phi\) by A.1's path expansion.
```

### 10.204 Appendix A.4: massless estimate before tightening the proof-split sentence (with 10.202)
```latex
On the finite lattice \(\Lambda_L=(\mathbb Z/L\mathbb Z)^3\), \(L\)
even, with antiperiodic shifts and \(\epsilon=1/L\), let
\(P_0^{(L)}\) be the flat projector at the same mass
\(0\le\zeta\le\zeta_*\) (it has no zero mode, including at
\(\zeta=0\)). For \(n_*\le\bar N\le n^*\), \(|\bar v|\le v^*\),
\(n_*-v^*\ge c_*>0\),
\(\sum_{\mathbf q}|\mathbf q|(|\widehat N(\mathbf q)|+\sum_i|\widehat v_i(\mathbf q)|)\le\mathcal F_1\epsilon\),
a nonwrapping ball \(c_rL\le r\le0.45L\), and
\(P=\mathbb 1_{(-\infty,0)}(h)\) with any orthogonal zero-mode
filling, \(|S_A(P)-S_A(P_0^{(L)})|\le CA_{\rm lat}\epsilon^{1/10}\)
with \(C\) depending only on
\((\zeta_*,\mathcal F_1,c_*,n^*,v^*,c_r)\). The same holds for a
comparator projector with the stated spectral occupations. The proof
(Ref.~\cite{McKinneyCode2026}) splits at \(\delta=\epsilon^{2/5}\)
into a rank \(O(L^3\delta^3)\) part and a norm \(O(\epsilon/\delta)\)
part, giving the exponent \(1/10\), and bounds the flat
finite-lattice entropy difference by \(Cr^{3/2}(1+\log r)\), so A.1
gives \(\eta(\zeta)\) in the fixed-mass limit.
```

### 10.205 Appendix A.5: gap criterion before tightening the partition clause (with 10.202)
```latex
To verify the gap premise (H3), assume
\(0<n_*\le\bar N\le n^*\), \(|\bar v_i|\le v^*\), and
Lipschitz constants at most \(C_1\epsilon\). The symbol frozen at \(x\),
\(\bar N(x)\hat h_0(k)+\sum_i\bar v_i(x)q_i(k)\), has frozen gap
\(\mathsf g(x)=\min_k[\bar N(x)\varepsilon(k)-|\bar v(x)\cdot q(k)|]\). Set \(\mathsf g_*=\inf_x\mathsf g(x)\). For every integer \(R\ge1\) and every \(\psi\),
\begin{equation}
\begin{split}
 \|h\psi\|&\ge(\mathsf g_*-E_R-D_R)\|\psi\|,\\
 E_R&=C_1\epsilon[\sqrt3(\zeta+9/2)R+3],\\
 D_R&=\frac{3\pi(n^*+v^*)}{2R}.
\end{split}
\label{eq:B-gapcert}
\end{equation}
Positive \(\delta=\mathsf g_*-E_R-D_R\) establishes the gap
(Ref.~\cite{McKinneyCode2026}): a tensor cosine partition of
support half-width \(R\) compares localized operators with frozen
symbols by Plancherel, freezing costing \(E_R\) and partition
commutators \(D_R\), and \(R=O(\epsilon^{-1/2})\) gives sufficient
gap loss \(O(\sqrt\epsilon)\).
```

### 10.206 Appendix C: finite-moment realization paragraph before compression (structural pass F; the Gram-matrix mechanics and the strict-mixedness condition on the operator norm moved to the reference, the 1028-test verification kept)
```latex
% assumption: the realization tests the imposed identities on finitely many moments about a strictly mixed covariance; it establishes neither the smoothed source equations nor their inverse beyond spherical symmetry, and the null-Lagrangian step below inherits the potential-flow restriction of the fourth requirement.
% risk: low
% GR slippage: no
% fold into text: no; stated
% provenance: Appendix D checks (1028 tests, 1.1e-7, gap>0.12): <internal>/APPD_SPINCURL_RA_busi_v1.md, SPINCURL_CHECKS89_R1_busi_v1.md and <internal>/APPD_SPINCURL_webgpt_v1.md; commit c76303833b3 (gap >= 0.1966).
\paragraph{Finite-moment realization.}
On a differentiable reference branch, a regular strictly mixed
covariance \(C_0\) with \(\epsilon I\le C_0\le(1-\epsilon)I\) can be
corrected to satisfy the imposed identities exactly on the retained
moments: when the Gram matrix of the test vertices is positive
definite, its right inverse supplies an operator with the required
moment changes (assigned sources and reference moments of
Eq.~\eqref{eq:dm-reference} between \(u_0\) and nearby \(u\)), and
the corrected covariance stays strictly mixed while that operator's
norm is below \(C_0\)'s distance from 0 and 1 (construction in
Ref.~\cite{McKinneyCode2026}). This tests the realization on the
retained moments only. It establishes neither the full smoothed
source equations, the inverse of the laws beyond spherical symmetry,
nor any deformation-algebra closure. Two independent calculations
verify such a candidate with the full reference, the spin-curl
Hamiltonian, and cubic gapped \(W_2[N,v,b]\) on a ball of radius
\(8a\) at \(ma=0.2\), \(G/a^2=1/4\), over 1028 tests of the four
densities: the moments match exactly, and uniform rigorous bounds
including rounding and quadrature give strict mixedness (setup
values, gap, and coefficient-norm bound in
Ref.~\cite{McKinneyCode2026}).
```

### 10.207 Sec. VII A: the E.13 and E.14 sentences as first landed in 433edc7c9cc, before the clause trims of pass F
```latex
% provenance: L ~ 0.005237 = C_0 of page_curve_amps R1/R2 (see Eq. sc-luminosity provenance).
Permeability alone does not dissolve the information paradox, which needs no impermeable surface~\cite{Hawking1976}: for earlier radiation \(R\), a new mode
\(b\), and its retained partner \(c\), weak monotonicity gives
\(S(Rb)-S(R)\ge S(c)-S(bc)\), so a nearly pure partner
(\(S(c)\ge\ln2-\epsilon\), \(S(bc)\le\epsilon\)) forces a
radiation-entropy increase of at least \(\ln2-2\epsilon\) whatever
the surface's permeability (Ref.~\cite{McKinneyCode2026}), the monogamy behind the firewall argument~\cite{AMPS2013}. Transfer
must carry partner correlations, not just probability. On the computed histories the surface is not where entanglement stops: the cross-surface correlations exceed the vacuum's severalfold (Sec.~\ref{sec:page-transfer}), the surface forms inside matter whose central density stays at a few percent of the exclusion density, and no curvature barrier is met at it, so whatever decides the firewall question here is the partner accounting, not a decoupling of the two sides. For the pure Hartree light covariance and complementary once-counted output and
retained projectors \(E\) and \(Q\), \(QCQ-(QCQ)^2=QCECQ\): escaped
entropy equals entanglement with the whole retained complement. A
later output block reduces accumulated entropy exactly when its
conditional entropy is negative. Propagating the retained singular vectors of the earlier-output cross block measures partner transfer and censored delays. On a stationary held future the residual after complete continuous output is the bound-sector entropy. Finite-chain holds require a return budget and do not establish that limit by themselves. The logarithmic branch delay bounds no transferred weight. These finite-history quantities are
\todo{C1/C2 conditional entropies, partner-transfer matrix, and
held-future residual controls}. Complete release through turnover
remains open.
```

### 10.208 Sec. VI: opening paragraph as landed in 2f386647ed6, before the two closing sentences were merged (pass F)
```latex
% assumption: the astrophysical extrapolation expands the dust benchmark (comoving dust at N=1 carrying all charge, the rows' Oppenheimer-Snyder solution), not the computed object; every late-stage law below is a property of that benchmark.
% risk: medium
% GR slippage: no (the benchmark is the rows' own solution, though it is also general relativity's dust collapse)
% fold into text: no; "Expand the dust benchmark's surface approach" is stated
% provenance: eta_E = 59.0689/64 = 0.923 (two_gE0 of P64_s4 in <internal>/ladders_readout_v1_3.json).
The computed object has a nominal \(r_s\) of 64 lattice spacings and a stellar-mass black hole has \(10^{38}\), so this section asks what of the computed collapse survives that extrapolation and what a distant observer would see. Four questions are answered in turn, each on the exterior rays, where the lattice enters only through averaging, mode conversion, and the probe's own dispersion: how long the exterior emits at Hawking's temperature after the surface crosses \(r_s\) (Sec.~\ref{sec:scaling-rays}), what redshift a far observer receives from the surface and where the long-wavelength description of its light ends (Sec.~\ref{sec:scaling-far}), whether an external observer could tell the object at one solar mass from a Schwarzschild black hole (Sec.~\ref{sec:scaling-budget}), and, under assumed emission, storage, and release laws, when the radiation entropy turns over (Sec.~\ref{sec:page-effective}). The first three questions use only the derived exterior and the dust comparison solution. The fourth is the one place in the paper where laws are assumed rather than derived.
```

### 10.209 Sec. II E: boundary-generator why-it-matters sentence as landed in 5bcc2d221a1, before the clause trim (pass F)
```latex
% assumption: Hartree order means clock and flow are c-number fields sourced by one-body expectation values (semiclassical gravity); the quantum form is an expansion about this branch whose convergence is not established, and the branch restriction of the fifth declaration is a projection postulate that keeps the quantum theory near the Hartree solution by construction.
% risk: medium
% GR slippage: no
% fold into text: no; declared
% assumption: the boundary-generator identity and conservation of E need the rows to be solvable differentiably in C (invertible constraint Jacobian) along the whole history; energy conservation is conditional on staying on the admitted branch.
% risk: low
% GR slippage: no
% fold into text: no; stated in Appendix C's hypotheses
Last comes the quantum form the computed collapse uses: the spherical sector's leading Hartree term, the exact covariance evolution of Sec.~\ref{sec:model-dictionary}, which on a differentiable field branch determines Gaussian cut entropies, the sources also requiring Eq.~\eqref{eq:dm-reference}'s field-dependent reference moments and the pair vertex below quartic correlations. On a differentiable regular inward branch, varying Eq.~\eqref{eq:dr-radial} with \(\langle u,\delta_us\rangle=0\) from Eq.~\eqref{eq:dm-contact} gives the boundary-generator identity (Appendix~\ref{app:charge}):
\begin{equation}
\boxed{\;\delta E=\Tr(\bar h\,\delta C),\qquad
\bar h=h[W_\ell^\dagger N,W_\ell^\dagger v]\;}\,,
\label{eq:dr-boundary-variation}
\end{equation}
for every differentiable variation of \(C\) on the inward branch, so along the Hartree evolution \(i\dot C=[\bar h,C]\) the boundary charge, rather than \(E_m\), generates the matter's evolution and is conserved by it. One charge read on the bonds leaving the whole system therefore drives everything inside it, which is where the principle's surface reading first becomes dynamics and the statement on which the soft holographic reading of Sec.~\ref{sec:discussion} rests.
```

### 10.210 Sec. II D: first-step closer as landed in 086faf4b2f5, before the trim (pass F)
```latex
% assumption: the spherical rows need a regular centre M(0)=0, unit clock at infinity and the inward branch v<0; the energy row fixes only v^2, so the sign is a choice of time orientation (inward is the black-hole branch; v>0 would give the outgoing, white-hole branch), and N is fixed only where the flow is nonzero.
% risk: medium
% GR slippage: no
% fold into text: no; the three conditions are stated after Eq. dr-radial
The first step ends with its result, the two Gauss laws of the hydrodynamic limit, the energy law and the momentum law:
\begin{equation}
\boxed{\begin{aligned}
(\operatorname{tr}K)^2-\operatorname{tr}(K^2)&=16\pi G\rho_\ell,\\
-\nabla\cdot\bigl[K-(\operatorname{tr}K)I\bigr]&=8\pi GJ_\ell ,
\end{aligned}}
\label{eq:dm-h2}
\end{equation}
selected by the principle's four requirements on the declared domain, smooth fields over the fixed flat lattice at leading order in gradients. These two equations carry the whole gravitational content of the paper: the second step identifies them with Einstein's constraints, and the exterior, the temperature, the entropy, and the computed collapse all solve them and nothing else, with \(G\) the only constant. Their isotropy at \(a_3=0\) concerns these
constraints, not matter dispersion (Sec.~\ref{sec:results-cone}) or
any tensor mode. In spherical symmetry they are the rows (quantum
form in Sec.~\ref{sec:model-operators}),
\begin{equation}
w=\frac vN,\quad
\mathcal M=\frac{rw^2}{2G},\quad
\mathcal M'=4\pi r^2(\rho_\ell+wJ_{\ell r}),
\label{eq:dr-charge}
\end{equation}
and, the energy row written for \(v\),
\begin{equation}
(rv^2)'=8\pi Gr^2N^2\rho_\ell,\qquad
\frac{vN'}{4\pi GrN^2}=-J_{\ell r},
\label{eq:dr-radial}
\end{equation}
With a
regular centre, \(\mathcal M(0)=0\), a unit clock at infinity, and the
inward branch (\(v<0\) for \(r>0\)), the rows fix \(\mathcal M\) and
\(N\) from the sources wherever the enclosed energy is not zero: in
this sector the geometry has no data of its own,
\(E=\mathcal M(\infty)\) is the boundary charge and \(r_s=2GE\) the length it defines. This completes the first step. What the rows do not say is what \(N\) and \(v\) mean geometrically, which the second step supplies. 
```

### 10.211 Sec. III E and III F: crossing-delay and temperature why-it-matters sentences as landed in 5bcc2d221a1, before the clause trims (pass F)
```latex
% assumption: the outward lattice branch and its delay are computed on a static constant-charge exterior held fixed during a flight of duration about 2 r_s ln(1/(a omega)); back-reaction and evolution of the fields during the crossing are neglected.
% risk: low
% GR slippage: no
% fold into text: no; "Holding the quasi-static constant-charge exterior throughout flight" is stated
In the point-sampled axial control with the fields held fixed at
\(N=1\), for \(z=ak\in(0,\pi/2)\) and \(u_r=\sqrt{r_s/r}\), the
massless positive-frequency outgoing carrier obeys
\begin{equation}
\begin{aligned}
a\omega&=\sin z(1-u_r\cos z),\quad \dot r=\cos z-u_r\cos(2z),\\ \dot r|_{u_r=1} &=\tfrac32z^2+O(z^4)
=\tfrac32(2a\omega)^{2/3}+O((a\omega)^{4/3}).
\end{aligned}
\label{eq:dr-crossing-ray}
\end{equation}
Thus an outgoing branch exists at every inward speed (monotonicity in Ref.~\cite{McKinneyCode2026}): the lattice dispersion's positive-frequency branch at the mesoscopic wavelength \(O((a^2/\omega)^{1/3})\), whose group velocity exceeds the long-wavelength light speed by the cubic correction of Sec.~\ref{sec:results-cone}, lattice scale at fixed \(u_r>1\). On the quasi-static constant-charge exterior, as \(a\omega\to0\) at fixed \(m/\omega<1\), this branch's delay between fixed \(r_{\rm in}<r_s<r_{\rm out}\) is
\begin{equation}
 \boxed{\;\Delta t_{\rm cross}=2r_s\ln\frac{1}{a\omega}+O(r_s)\;}\,,
 \label{eq:star-crossing-delay}
\end{equation}
a third of it inside the surface. No escape fraction follows from this point-sampled lattice control. This is the sense in which the surface is no horizon: a lattice mode of mesoscopic wavelength leaves the trapped region with a delay only logarithmic in the spacing, so the trapping surface is a long-wavelength statement, and the information questions of Secs.~\ref{sec:page} and \ref{sec:discussion} are posed on a surface that lattice modes cross. The exterior is therefore Schwarzschild's, with \(r_s=2GE\), surface slope \(\kappa_H=1/(4GE)\), and a lattice branch that crosses it outward with a logarithmic delay.
```

### 10.212 Sec. III F: temperature why-it-matters sentence as landed (with 10.211)
```latex
% assumption: the conversion factor is Hawking's fixed-background result applied to the derived exterior; it needs the incoming state to be the prepared sea in the regular incoming coordinate, geometric-optics transfer through the collapsing interior ("transverse surface crossing and regular interior transfer"), adiabaticity, and a surface that crosses r_s, which the computed collapse does not resolve (Sec. V E); the segment lasts only about 0.2 thermal periods at the computed size and about 5 at solar mass.
% risk: medium
% GR slippage: possible (quantum field theory on the derived background, legitimate after Sec. II D; the exponential segment is a property of the collapse benchmark, not of the computed object)
% fold into text: no; the conditions are in the paragraph and Sec. V E
% provenance: R_pi = 4 a n_rad,max (1.222 N_B^{1/3} a isotropic); T_d window 2.31 r_s and 57.8 r_s are closed-form leading logs; cross-listed in <internal>/ASYMPTOTICS_checks_v1.9.md.
On that exterior the ray map gives the conversion temperature, the temperature the first law needs to turn area into entropy. For outgoing retarded time \(u\) normalized at infinity, let \(U(u)\)
be the received ray's incoming vacuum coordinate and
\(\kappa_u=-\partial_u\ln|\partial_uU|\) the peeling rate.
On an exponential segment with \(\kappa_u>0\),
\(|\partial_u\kappa_u|\ll\kappa_u^2\) and \(\kappa_ua\ll1\),
modes \(m<\omega\ll a^{-1}\) incident in the regular incoming
vacuum leave with leading Fermi conversion factor
\([e^{2\pi\omega/\kappa_u}+1]^{-1}\)~\cite{Hawking1975,BarceloEtAl2006,BarceloEtAl2011}. On the
constant-charge exterior of Eq.~\eqref{eq:dr-exterior}, with
transverse surface crossing and regular interior transfer, the map
from the affine incoming coordinate \(U\) has the late exponent
\(\kappa_H=1/(4GE)\), the vacuum surface slope:
\begin{equation}
\boxed{\begin{gathered}
U_H-U=C_Ue^{-\kappa_Hu}+\cdots,\qquad\kappa_H=1/(2r_s),\\
T_{\rm conv}=\kappa_H/(2\pi)=T_H=1/(8\pi GE).
\end{gathered}}
\label{eq:sc-ray-temperature}
\end{equation}
Hawking's temperature therefore follows from the derived exterior and the prepared sea with no bath imposed, and it is the temperature that the first law of Sec.~\ref{sec:results-firstlaw} converts into the area entropy, while how long the emission lasts is a separate question, answered next. The exponential segment lasts
\(T_d=\tfrac23r_s\ln[1/(2a\omega)]+O(r_s)\) in Schwarzschild time \(t_s\) for a surface falling through it (Sec.~\ref{sec:scaling}), twice \(T_d\) at leading logarithmic order in retarded time. The leading-logarithmic estimates are
\(2.31r_s\) at \(\omega r_s=1\) and \(r_s/a=64\), 0.18 of the thermal
period \(1/T_H=4\pi r_s\), and \(57.8r_s\), 4.6 periods, at one solar
mass. Within it the stated modes leave at leading order at \(T_H\). The subsequent emission is not fixed by this ray-map result.
At a fixed profile, finite spacing changes the conversion temperature
at each frequency, not \(\kappa_u\):
\(T_{\rm conv}(\omega)=T_H[1+\delta_T(\omega;E,a,m)]\).
An unsuppressed channel at infinity needs \(m<T_H\).
```

### 10.213 Sec. I: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_02 pair 1), beginning "Geometry is then bookkeeping of how the matter's energy and momentum a"
Replaced text:
```latex
Geometry is then bookkeeping of how the matter's energy and momentum are weighted, derived in four steps (Sec.~\ref{sec:model-gravity}): the laws the principle allows, the metric read from clock, flow, and spacing, the two spherical rows that follow a source in time, and what survives the continuum limit, where one parameter remains, \(G\) in lattice units.
```
Replacement:
```latex
Geometry is then bookkeeping of how the matter's energy and momentum are weighted (Sec.~\ref{sec:model-gravity}).
```

### 10.214 Sec. II: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_03 pair 1), beginning "Potential flows have no rotation and do not test \(b\). A flux require"
Replaced text:
```latex
Potential flows have no rotation
and do not test \(b\). A flux requirement on every flow would require
\(b=-1\), contradicting the third fact (Appendix~\ref{app:rows}). The momentum law,
```
Replacement:
```latex
The momentum law,
```

### 10.215 Sec. II: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_03 pair 2), beginning "The laws are derived in this regime. The computation extends beyond it"
Replaced text:
```latex
The laws are derived in this regime. The computation extends beyond it.
```
Replacement:
```latex
The computation extends beyond it.
```

### 10.216 Sec. III: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_04 pair 1), beginning "angular-channel chains (one radial chain per angular-momentum channel)"
Replaced text:
```latex
angular-channel chains (one radial chain per angular-momentum channel).
```
Replacement:
```latex
angular-channel chains.
```

### 10.217 Sec. III: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_04 pair 2), beginning "of itself. This estimate uses the massless match. Its angular remainde"
Replaced text:
```latex
of itself.
This estimate uses the massless match. Its angular remainder
still needs a bound.}
```
Replacement:
```latex
of itself.}
```

### 10.218 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 1), beginning "The reference collapse uses the fixed reference projector without ligh"
Replaced text:
```latex
The reference collapse uses the fixed reference projector without light, and the three two-field histories hold the sea projector frozen at release under different light rules (one with the block-form light counterterm whose constant-coefficient massless-chain endpoint fails through \(e_{11}\), Sec.~\ref{sec:results-lattice}), so these tests verify no trajectory under Eq.~\eqref{eq:dm-reference}, and the stored-slice agreement of Sec.~\ref{sec:results-lattice} remains a consistency check.
```
Replacement:
```latex
None of these histories runs under Eq.~\eqref{eq:dm-reference} (Sec.~\ref{sec:results-lattice}), so these tests verify no trajectory under it.
```

### 10.219 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 2), beginning "These probes, beyond the hydrodynamic description's tested limit, esta"
Replaced text:
```latex
These probes, beyond the hydrodynamic description's tested limit, establish
neither unique joint continuation between stored slices, affine
completeness, nor an infinite-time extension.
```
Replacement:
```latex

```

### 10.220 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 3), beginning "Centroid minima move inward with width, which is centroid motion and n"
Replaced text:
```latex
Centroid minima move inward with width, which is centroid motion and not interior rebound, the shell charge at \(16r_s\) shows a held-step deficit that halves with step length, a held-step effect and not a charge-loss law, and nothing below uses the depth or centroid values (the unverified twenty-channel and sea-pair readings and the open conserving-step items in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
Centroid minima move inward with width, centroid motion and not interior rebound (depth, centroid, shell-charge, and the unverified twenty-channel and sea-pair readings in Ref.~\cite{McKinneyCode2026}, none used below).
```

### 10.221 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 4), beginning "with no continued projector. The level following is checked at single "
Replaced text:
```latex
with no continued projector. The level following is checked at single stored slices, not along the history, and no stored history is verified under Eq.~\eqref{eq:dm-reference}'s source prescription.
```
Replacement:
```latex
with no continued projector.
```

### 10.222 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 5), beginning "solution. Exclusion bounds the prepared-sea part of the sources, not t"
Replaced text:
```latex
solution. Exclusion bounds the prepared-sea part of the sources, not
the polarization moments. At a regular centre,
```
Replacement:
```latex
solution. At a regular centre,
```

### 10.223 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 6), beginning "of Sec.~\ref{sec:results-filling} therefore bounds"
Replaced text:
```latex
of Sec.~\ref{sec:results-filling} therefore bounds
```
Replacement:
```latex
of Sec.~\ref{sec:results-filling} bounds
```

### 10.224 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 7), beginning "Bounded density establishes neither bounded full curvature nor continu"
Replaced text:
```latex
Bounded density establishes neither bounded full curvature nor
continuation.
Positivity offers
```
Replacement:
```latex
Positivity offers
```

### 10.225 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 8), beginning "A validated \(q_\infty\le0.080\) would give branch weight at least 0.9"
Replaced text:
```latex
A validated \(q_\infty\le0.080\) would give branch weight at least 0.92, but neither actual-error inclusion nor the declared-reference branch calculation is established, and the five slices cover neither all 18 nor \(T/r_s=11.0\).
```
Replacement:
```latex
The five slices cover neither all 18 nor \(T/r_s=11.0\).
```

### 10.226 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 9), beginning "and rises again by 16. The \(\sigma/a=4\) history stores no slice at 1"
Replaced text:
```latex
and rises again by 16. The \(\sigma/a=4\) history stores no slice at 13 to
14, so its compression minimum is unsampled.
```
Replacement:
```latex
and rises again by 16.
% sampling provenance (pagebud item 17, text kept here): The \(\sigma/a=4\) history stores no slice at 13 to 14, so its compression minimum is unsampled.

```

### 10.227 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 10), beginning "the one-term test failing at gradient length \(1/|H|<10\sigma=40a\). S"
Replaced text:
```latex
the one-term test failing at gradient length \(1/|H|<10\sigma=40a\). Sec.~\ref{sec:singularity} discusses singularity and its formation beyond the sampled description limit.
```
Replacement:
```latex
the one-term test failing at gradient length \(1/|H|<10\sigma=40a\).
```

### 10.228 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 11), beginning "slice, activating at the first stored slice failing the test at or aft"
Replaced text:
```latex
slice, activating at the first stored slice failing the test at or after 11.125: inside the failing radius they hold sources
```
Replacement:
```latex
slice: inside the failing radius they hold sources
```

### 10.229 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 12), beginning "or point-sample core coupling (the last ends at 12.5)."
Replaced text:
```latex
or point-sample core coupling (the last ends at 12.5).
```
Replacement:
```latex
or point-sample core coupling (activation and end slices in Ref.~\cite{McKinneyCode2026}).
```

### 10.230 Sec. IV: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_05 pair 13), beginning "Field panels use a separate production history checked against the ref"
Replaced text:
```latex
Field panels use a separate production history checked against
the reference collapse.
```
Replacement:
```latex
% Field panels use a separate production history checked against the reference collapse. (pagebud item 18, caption sentence kept here as a comment)
```

### 10.231 Sec. VI: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_06 pair 1), beginning "fountain modulation of the interior (its weak-modulation coefficient u"
Replaced text:
```latex
fountain modulation of the interior (its weak-modulation coefficient unmeasured), or evolving-background
```
Replacement:
```latex
fountain modulation of the interior, or evolving-background
```

### 10.232 Sec. VI: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_06 pair 2), beginning "output correlations. The stored time slices show no lifetime, late pla"
Replaced text:
```latex
output correlations. The stored time slices show no lifetime, late plateau, end burst, turnover, or evaporation endpoint. The scale
```
Replacement:
```latex
output correlations. The scale
```

### 10.233 Sec. VI: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_06 pair 3), beginning "nor heavy-information transfer. Eq.~\eqref{eq:star-crossing-delay} is "
Replaced text:
```latex
nor heavy-information transfer. Eq.~\eqref{eq:star-crossing-delay} is a delay, not a release law.
```
Replacement:
```latex
nor heavy-information transfer.
```

### 10.234 Sec. V: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_07 pair 1), beginning "Reported long-hold fits at the last slope maximum and minimum (values "
Replaced text:
```latex
Reported long-hold fits at the last slope maximum and minimum (values in Ref.~\cite{McKinneyCode2026}) are unverified, unreconciled against \(T_H\) across readings, and no evidence for the evolving object's emission. The \(\tau=4\) history sustains neither packet at its last slope maximum, and no conversion spectrum is read at this size.
```
Replacement:
```latex
No conversion spectrum is read at this size (unverified long-hold fits in Ref.~\cite{McKinneyCode2026}).
```

### 10.235 Sec. V: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_07 pair 2), beginning "\(\Delta S_{\rm cut}=S_{\rm cut}-\Tr s(P_{-,r>R})\), and no-return out"
Replaced text:
```latex
\(\Delta S_{\rm cut}=S_{\rm cut}-\Tr s(P_{-,r>R})\), and no-return
output entropy \(S_{\rm esc}\), counting modes once.
The held full-chain negative projector \(P_-\) is formed before
restriction. Weights \(16k\) count identical copies (error floors in Ref.~\cite{McKinneyCode2026}). Excess is not radiation entropy:
```
Replacement:
```latex
\(\Delta S_{\rm cut}=S_{\rm cut}-\Tr s(P_{-,r>R})\), \(P_-\) the held full-chain negative projector, and no-return output entropy \(S_{\rm esc}\), counting modes once (restriction order, copy weights, and error floors in Ref.~\cite{McKinneyCode2026}). Excess is not radiation entropy:
```

### 10.236 Sec. V: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_07 pair 3), beginning "output-return and omitted-mode errors}, and \(P_h\), the held vacuum a"
Replaced text:
```latex
output-return and omitted-mode errors}, and \(P_h\), the held vacuum
at stored slice 800, is the fixed comparator, not an entropy subtraction.
```
Replacement:
```latex
output-return and omitted-mode errors}.
```

### 10.237 Sec. V: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_07 pair 4), beginning "It does not demonstrate a Page curve, and the Page curve printed in Se"
Replaced text:
```latex
It does not demonstrate a Page curve, and the Page curve printed in
Sec.~\ref{sec:page-effective} is conditional on assumed laws. The readings follow.
```
Replacement:
```latex

```

### 10.238 Sec. V: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_07 pair 5), beginning "The five cuts from \(2r_s\) to \(48r_s\) span a few parts in \(10^4\) "
Replaced text:
```latex
The five cuts from \(2r_s\) to \(48r_s\) span a few parts in \(10^4\) of \(A(E)/(4G)\) at 100 (Fig.~\ref{fig:page-v4}), while field jumps can leave nonsummable reference mismatch (maxima, spans, the non-monotonic \(30r_s\) excess, and the propagation and purity readings in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The five cuts from \(2r_s\) to \(48r_s\) are shown in Fig.~\ref{fig:page-v4}, and field jumps can leave nonsummable reference mismatch (cut values in Ref.~\cite{McKinneyCode2026}).
```

### 10.239 Sec. V: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_07 pair 6), beginning "Finite stored slices verify crossing and outward motion, not escape to"
Replaced text:
```latex
Finite stored slices verify crossing and outward motion, not escape
to infinity. Whether
```
Replacement:
```latex
Whether
```

### 10.240 Appendix C: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_C pair 1), beginning "\paragraph{Finite-moment realization.} On a differentiable reference b"
Replaced text:
```latex
\paragraph{Finite-moment realization.}
On a differentiable reference branch, a regular strictly mixed covariance can be corrected to satisfy the imposed identities exactly on the retained moments, the assigned sources and reference moments of Eq.~\eqref{eq:dm-reference} between \(u_0\) and nearby \(u\), when the Gram matrix of the test vertices is positive definite (construction in Ref.~\cite{McKinneyCode2026}). This tests the realization on the retained moments only and establishes neither the full smoothed source equations, the inverse of the laws beyond spherical symmetry, nor any deformation-algebra closure. Two independent calculations verify such a candidate with the full reference, the spin-curl Hamiltonian, and cubic gapped \(W_2[N,v,b]\) on a ball of radius \(8a\) at \(ma=0.2\), \(G/a^2=1/4\), over 1028 tests of the four densities, the moments matching exactly and uniform rigorous bounds giving strict mixedness (setup values, gap, and coefficient-norm bound in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
% repository pointer (pagebud item 1, lead inbox 96): the finite-moment realization on a differentiable reference branch (a regular strictly mixed covariance corrected to satisfy the imposed identities exactly on the retained moments when the Gram matrix of the test vertices is positive definite; two-calculation verification with the full reference, the spin-curl Hamiltonian, and cubic gapped W_2[N,v,b] on a ball of radius 8a at ma = 0.2, G/a^2 = 1/4, 1028 tests, the moments matching exactly and uniform rigorous bounds giving strict mixedness) is in Ref. McKinneyCode2026; the paragraph is archived verbatim in review_output/paperapi/REPO_DERIVATIONS_20261102.md. No result in the body cites it.
```

### 10.241 Appendix C: replaced text, verbatim (pagebud Tier A, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): duplicates, unused definitions, and unused diagnostics; replaced text verbatim, replacement shown, spec TA_C pair 2), beginning "requires \(b=-1\) (conservation is not used here, and the spherical bo"
Replaced text:
```latex
requires \(b=-1\) (conservation is not used here, and the spherical boundary-generator identity follows from the rows, Appendix~\ref{app:charge}).
```
Replacement:
```latex
requires \(b=-1\).
```

### 10.242 Sec. VI: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_06 pair 1), beginning "The five-slice test uses the stored two-field trajectory with light-re"
Replaced text:
```latex
The five-slice test uses the stored two-field trajectory with
light-reference weight 16, not the paper's weight 32 (a half-weight light counterterm with flagged, zero endpoint coefficients, its identification with \(W_2\) not established). At the five stored slices \(t/a=1096\) to 1112, \(T/r_s=14.9\) to 15.1, with \(N=1\) at the grid's outer edge, the trapping radius's energy debit per lattice time is read under a piecewise-linear (P) and a cubic-Hermite (H) interpolant \(\widehat{\mathcal M}\) of the stored enclosed energy, \(\widehat D_R=1-2G\,\partial_r\widehat{\mathcal M}(R)\), \(H_R=-\dot{\widehat R}/(2G)=-\partial_t\widehat{\mathcal M}(R)/\widehat D_R\). The second calculation's rigorous interval bounds at the outer trapping radius (tabulated in Ref.~\cite{McKinneyCode2026}) are negative at \(t/a=1096\) (expansion) and positive after (recession) under both interpolants. All ten intervals overlap the first calculation's provisional intervals, numerical corroboration by the second calculation, not a second rigorous bound. Three cubic-Hermite neighbour cells are not shown zero-free (outer trapping radius at \(t/a=1096\) and 1100, inner at 1108). The P interpolant is clean. The second calculation bounds the inner trapping radius's debit negative (outward motion) at all five slices, and all ten inner intervals overlap the first calculation's provisional values (numerical corroboration). At each of the five stored slices the implemented field solve has a unique solution in a verified box around the stored fields, proved by a separate interval-Newton certificate (contraction about \(10^{-4}\)), and the second calculation's intervals enclose the debit at the implemented fixed points in its box (uniqueness re-check with directed-rounding export pending), so the Hartree reading of the debit and its sign holds there for the implemented half-weight law: two independent calculations agree on all twenty trapping-radius and interpolant pairs, one with rigorous bounds and one corroborating. The test checks balances and debit shares and establishes no between-slice continuation, continuum trapping surface, Hawking depletion, or Page curve. A scaling-uniform \(\sup|\mathfrak d_c|=o(E_0)\) through
turnover remains open, as do the full heavy-light boundary-charge
Hessian, prepared-sector multiplicities, and funded complete release.
```
Replacement:
```latex
The five-slice test on the stored two-field trajectory (light-reference weight 16, not the paper's 32, under a half-weight counterterm whose identification with \(W_2\) is not established) reads the trapping radius's energy debit per lattice time at \(t/a=1096\) to 1112, \(T/r_s=14.9\) to 15.1, with rigorous interval bounds: the outer trapping radius's debit is negative at 1096 (expansion) and positive after (recession), the inner one negative (outward motion) at all five slices, under two interpolants of the stored enclosed energy, with a unique implemented field solve in a verified box at each slice (interpolant, cell, certificate, and corroborating-calculation details in Ref.~\cite{McKinneyCode2026}). The test checks balances and debit shares and establishes no between-slice continuation, continuum trapping surface, Hawking depletion, or Page curve. A scaling-uniform \(\sup|\mathfrak d_c|=o(E_0)\) through turnover remains open, as do the full heavy-light boundary-charge Hessian, prepared-sector multiplicities, and funded complete release.
```

### 10.243 Sec. VI: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_06 pair 2), beginning "two independent partner-accounting calculations find"
Replaced text:
```latex
two independent partner-accounting calculations find
```
Replacement:
```latex
partner accounting on held fields finds
```

### 10.244 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 1), beginning "The deciding calculation is a validated rigorous bound on the coupled "
Replaced text:
```latex
The deciding calculation is a validated rigorous bound on the coupled trajectory on every consecutive slice interval, satisfying \(i\dot C=[\bar h(u),C]\) and \(F_u\dot u=-F_C\dot C\) for the boundary-normalized constraints \(F(C,u)=0\), \(u=(N,v)\), with the full reference response, connecting to the next interval, under four sufficient conditions. First, a solution rigorously bounded under Eq.~\eqref{eq:dm-reference} with the propagated sea, every active species and copy weight, and every numerical error. Second, throughout the interval a regular inward branch with a positive margin \(\mathcal M(r,t)/r^3\ge\mu_{*,j}>0\) and Eq.~\eqref{eq:sg-current-mass} verified on the same bound with the full constraint mass. Third, a uniform inverse bound on the full \(F_u\), reference response included, which neither a clock bound nor the inverses verified at stored time slices supplies. Fourth, a self-closing bound on the fields, their regularity, and the reference derivatives that continues the same branch (full statement, centre handling, approximate-inverse recipe, and discrete-row clock estimate in Ref.~\cite{McKinneyCode2026}). The positive margin selects a sufficient nondegenerate domain, degenerate centres and mass-zero vacuum components needing separate arguments, and failure of a sufficient inequality means not verified, not singular. Stored fixed-reference-projector histories may seed a verification only with rigorously bounded defects under the printed law, otherwise re-evolution is required, and interpolating stored fields and propagating test matter is not this calculation. No complete between-slice verification under the printed law, or general bound on polarization moments over regular cubic slices, is established.
```
Replacement:
```latex
The deciding calculation, in progress, is a validated rigorous bound on the coupled trajectory \(i\dot C=[\bar h(u),C]\), \(F_u\dot u=-F_C\dot C\) for the boundary-normalized constraints \(F(C,u)=0\), \(u=(N,v)\), over every consecutive slice interval under Eq.~\eqref{eq:dm-reference} with the full reference response, verifying throughout each interval a regular inward branch with margin \(\mathcal M/r^3\ge\mu_*>0\), Eq.~\eqref{eq:sg-current-mass}, a uniform inverse bound on the full \(F_u\), and a self-closing bound on the fields and reference derivatives (full statement, centre handling, and seeding rule in Ref.~\cite{McKinneyCode2026}); failure of a sufficient inequality means not verified, not singular, and no complete between-slice verification under the printed law is established.
```

### 10.245 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 2), beginning "with the stress's identity verified only on sampled slices (below) and"
Replaced text:
```latex
with the stress's identity verified only on sampled slices (below) and normalization, clock, and between-node sign unclosed
```
Replacement:
```latex
with normalization, clock, and between-node sign unclosed
```

### 10.246 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 3), beginning "Independent reconstructions support identifying the stored stress as \"
Replaced text:
```latex
Independent reconstructions support identifying the stored stress as \(S^{\rm op}[C-P_{\rm held}]\) on four shared stored time slices (listed in Ref.~\cite{McKinneyCode2026}), with small differences between the two independent calculations in prepared-operator and kinetic residuals on the three \(r_s/a=64\) slices but an unresolved prepared-reference and projection discrepancy at the size-128 anchor, while validated joint kinetic/reference agreement and complete size, width, and step controls remain unestablished. So identified, with \(P_{\rm held}\) the fixed reference projector
of Sec.~\ref{sec:results-lattice}, these are held-comparator
operator-stress residuals, not those of the kinetic \(S_{ij}\) of
Sec.~\ref{sec:model-gravity} with Eq.~\eqref{eq:dm-reference}'s
sources, and the split of Sec.~\ref{sec:model-gravity} remains
unclosed. A polarization coefficient needs two independent cubic-lattice calculations with its definition, sign, normalization, invariants, and error, beside the chains' separately normalized value.
```
Replacement:
```latex
The stored stress is the held-comparator operator stress \(S^{\rm op}[C-P_{\rm held}]\), \(P_{\rm held}\) the fixed reference projector of Sec.~\ref{sec:results-lattice}, not the kinetic \(S_{ij}\) of Sec.~\ref{sec:model-gravity} with Eq.~\eqref{eq:dm-reference}'s sources, so that split remains unclosed (identification on shared slices, its unresolved size-128 discrepancy, and the polarization-coefficient audit in Ref.~\cite{McKinneyCode2026}).
```

### 10.247 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 4), beginning "For the massive chain's outer endpoint, the stored cutoff sums pass ev"
Replaced text:
```latex
For the massive chain's outer endpoint, the stored cutoff sums pass
every prescribed test on their changes between successive cutoffs at
\(10^{-6}\), with propagated numerical error bounds below
\(10^{-8}\), and every comparison supports a fixed-radius subtraction
rather than an omitted gradient. No outer endpoint coefficient of the
massive chain is established, because for every one of the 40
coefficients (eight at each of five channels) the change from cutoff
512 to 1024 exceeds the change from 256 to 512, although every change
is below \(10^{-6}\) and the numerical bounds are below \(10^{-8}\)
(Ref.~\cite{McKinneyCode2026}).

The massless light chain has no finite endpoint under the constant-coefficient prescription. At \(m=0\), \(N=1\), \(v=0\), the two verified limits of the site-flow kernel that the endpoint rule needs both vanish for every \(k=1,\ldots,5\), so the massless tail's endpoint coefficient \(e_{11}\) diverges and \(e_{01}\) is undecided, with no cancellation established (kernel limits, endpoint rule, and the 80 finite-parent interval rows from two independent numerical calculations, which are implementation cross-checks and not coefficient bounds, in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The massive chain's outer endpoint is a fixed-radius subtraction, supported by every stored cutoff comparison, but no endpoint coefficient is established, the cutoff changes not yet decreasing (cutoff tests in Ref.~\cite{McKinneyCode2026}). The massless light chain has no finite endpoint under the constant-coefficient prescription: its endpoint coefficient \(e_{11}\) diverges for every \(k=1,\ldots,5\), with no cancellation established (kernel limits and endpoint rule in Ref.~\cite{McKinneyCode2026}).
```

### 10.248 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 5), beginning "Two-calculation statements name each calculation and its shared inputs"
Replaced text:
```latex
Two-calculation statements name each calculation and its shared inputs, two post-processing calculations on the same stored data not being two independent calculations.
```
Replacement:
```latex
Which calculation produced each number, and which numbers two independent calculations reproduce, is recorded in Ref.~\cite{McKinneyCode2026}, and readings resting on one calculation are labelled single-calculation or reported.
```

### 10.249 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 6), beginning "two independent calculations expanding about each slice's Hartree cova"
Replaced text:
```latex
two independent calculations expanding about each slice's Hartree covariance against the fixed reference projector find
```
Replacement:
```latex
expansions about each slice's Hartree covariance against the fixed reference projector find
```

### 10.250 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 7), beginning "on all 18 slices on the ten channels, 0.0798 at 15.0, is reproduced in"
Replaced text:
```latex
on all 18 slices on the ten channels, 0.0798 at 15.0, is reproduced in two independent calculations. It is conditional
```
Replacement:
```latex
on all 18 slices on the ten channels, 0.0798 at 15.0. It is conditional
```

### 10.251 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 8), beginning "two independent calculations with independent full-tail comparators gi"
Replaced text:
```latex
two independent calculations with independent full-tail comparators give
```
Replacement:
```latex
independent full-tail comparators give
```

### 10.252 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 9), beginning "In two independent calculations with \(\widehat E_2\) Wick ordered"
Replaced text:
```latex
In two independent calculations with \(\widehat E_2\) Wick ordered
```
Replacement:
```latex
With \(\widehat E_2\) Wick ordered
```

### 10.253 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 10), beginning "over \(T/r_s=12.125\)--40 in two independent calculations."
Replaced text:
```latex
over \(T/r_s=12.125\)--40 in two independent calculations.
```
Replacement:
```latex
over \(T/r_s=12.125\)--40.
```

### 10.254 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 11), beginning "at \(r_s/a=32\), 64, 256 in two independent calculations with no"
Replaced text:
```latex
at \(r_s/a=32\), 64, 256 in two independent calculations with no
```
Replacement:
```latex
at \(r_s/a=32\), 64, 256 with no
```

### 10.255 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 12), beginning "readings past it are labelled statements about the lattice equations ("
Replaced text:
```latex
readings past it are labelled statements about the lattice equations (Sec.~\ref{sec:model-dictionary}). The six-term test
```
Replacement:
```latex
readings past it are labelled statements about the lattice equations. The six-term test
```

### 10.256 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 13), beginning "The same-source comparison case \(\mathcal M=m\mathcal N_<\), \(\mathc"
Replaced text:
```latex
The same-source comparison case \(\mathcal M=m\mathcal N_<\), \(\mathcal N_<(r)\) the enclosed count, at constant bare energy per particle closes the law with \(N\) a power of \(\mathcal N_<\) and fails at a regular centre (exponent in Ref.~\cite{McKinneyCode2026}). On the stored slices the clock is no power of the enclosed count and no exponent is calibrated, while Eq.~\eqref{eq:sc-clock-exact} integrated on the fine-grid fields matches the clock's fall from \(R_{50}\) to the outer trapping radius within 1 to 3 percent in two independent calculations (slopes in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
Eq.~\eqref{eq:sc-clock-exact} integrated on the fine-grid fields matches the clock's fall from \(R_{50}\) to the outer trapping radius within 1 to 3 percent (slopes and the same-source comparison case in Ref.~\cite{McKinneyCode2026}).
```

### 10.257 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 14), beginning "Although the same-source gradient length grows as \((r_s/a)^{1/3}\) at"
Replaced text:
```latex
Although the same-source gradient length grows as
\((r_s/a)^{1/3}\) at fixed \(\sigma/a,ma,g,\epsilon_c\), the strain bound excludes
```
Replacement:
```latex
The strain bound excludes
```

### 10.258 Sec. IV: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_05 pair 15), beginning "a comparison radius. With \(\mathsf k=(2G\mathcal M/r^3)^{1/2}\) the r"
Replaced text:
```latex
a comparison radius. With \(\mathsf k=(2G\mathcal M/r^3)^{1/2}\) the rows give \((\ln N)'=4\pi GJ_{\ell r}/\mathsf k\), so the boundary-normalized central clock is finite and positive exactly when the signed integral \(\int_0^{r_1}J_{\ell r}/\mathsf k\,dr\) converges. For smooth finite-order centres, \(\mathsf k=r^{2n}A(r^2)\) with \(A(0)>0\), this is \(J_{\ell r}=O(r^{2n+1})\). Positive central density is unnecessary.
```
Replacement:
```latex
a comparison radius (a central-clock finiteness criterion for smooth centres in Ref.~\cite{McKinneyCode2026}).
```

### 10.259 Sec. II: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_03 pair 1), beginning "\(\mathsf T\) the momentum-space response tensor of Appendix~\ref{app:"
Replaced text:
```latex
\(\mathsf T\) the momentum-space response tensor of Appendix~\ref{app:charge}, a tensor response, not a renormalization of \(G\).
```
Replacement:
```latex
\(\mathsf T\) a momentum-space response tensor (Ref.~\cite{McKinneyCode2026}), a tensor response, not a renormalization of \(G\).
```

### 10.260 Sec. II: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_03 pair 2), beginning "A longitudinal flow \(v_1(x_1)\) gives \(-\widehat\gamma_{11}a^{-2}\in"
Replaced text:
```latex
A longitudinal flow \(v_1(x_1)\) gives \(-\widehat\gamma_{11}a^{-2}\int(\partial_1v_1)^2/N\), \(\widehat\gamma_{11}=a^2\mathsf T_{11,11}/4>0\) (two independent calculations agree at \(ma=0.2\), 0.009834, and at \(m=0\), 0.0105). The completed rigid turn
```
Replacement:
```latex
The completed rigid turn
```

### 10.261 Sec. II: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_03 pair 3), beginning "(the surface term of the integrated Euler pairing on a bounded region,"
Replaced text:
```latex
(the surface term of the integrated Euler pairing on a bounded region, and where Euler's identity is used, in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
(the longitudinal-flow coefficient \(\widehat\gamma_{11}\), the surface term of the integrated Euler pairing, and the use of Euler's identity in Ref.~\cite{McKinneyCode2026}).
```

### 10.262 Sec. II: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_03 pair 4), beginning "one even kernel \(k_{\ell,a}\), the cell function convolved with a nor"
Replaced text:
```latex
one even kernel \(k_{\ell,a}\), the cell function convolved with a normalized \(C^4\) bump of radius \(\ell\) (Eq.~\eqref{eq:dm-average}, Appendix~\ref{app:charge}), smooths site densities
```
Replacement:
```latex
one even kernel \(k_{\ell,a}\) (Eq.~\eqref{eq:dm-average}, Appendix~\ref{app:charge}) smooths site densities
```

### 10.263 Sec. II: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_03 pair 5), beginning "with per-axis variance \(\sigma^2=\ell^2/15+a^2/12\) and zero mixed mo"
Replaced text:
```latex
with per-axis variance \(\sigma^2=\ell^2/15+a^2/12\) and zero mixed moments, so that \(|W_\ell^\dagger Y-Y|\le\tfrac32\sigma^2A_*/L_{\rm geom}^2\) by Taylor's formula. The computation extends beyond it.
```
Replacement:
```latex
with per-axis variance \(\sigma^2=\ell^2/15+a^2/12\). The computation extends beyond it.
```

### 10.264 Sec. II: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_03 pair 6), beginning "The computed histories of Sec.~\ref{sec:star} evolve the same lattice "
Replaced text:
```latex
The computed histories of Sec.~\ref{sec:star} evolve the same lattice equations through this kernel:
the matter is a full one-body quantum state, the covariance, evolved
exactly under the quadratic generator at each step, and the clock and
flow at each step solve the constraint rows sourced by the
kernel-averaged moments of that state, quantum matter at Hartree
order on hydrodynamic fields with the second-order pair term of
Sec.~\ref{sec:model-operators} read as a diagnostic, not propagated.
Past the point where the smoothness test fails the evolution is still
exact and the fields well defined, but their reading as a geometry is
no longer guaranteed, so results there are stated as statements about
the lattice equations, and the singularity question of
Sec.~\ref{sec:singularity} is posed on those equations for that
reason.
```
Replacement:
```latex
The computed histories of Sec.~\ref{sec:star} evolve the same lattice equations through this kernel. Past the point where the smoothness test fails the evolution is still exact and the fields well defined, but their reading as a geometry is no longer guaranteed, so results there are stated as statements about the lattice equations (Sec.~\ref{sec:singularity}).
```

### 10.265 Sec. III: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_04 pair 1), beginning "\fnd{The reference-split comparison bounds the trapping-cut entropy pe"
Replaced text:
```latex
\fnd{The reference-split comparison bounds the trapping-cut entropy per
area in terms of these pieces. It gives
\(|S_A(C)/A_{\rm lat}-\eta(ma)|\le e_{\rm flat}+r_{\rm ref}
+\min\{\nu,r_A\log2\}/A_{\rm lat}\),
where \(e_{\rm flat}\) is the flat finite-cut remainder and \(r_{\rm ref}=|S_A(P)-S_A(P_0)|/A_{\rm lat}\) for a comparator projector \(P\), the ground projector or a stored history's fixed reference projector. The subtraction has no reference state.
For a trapping cut, \(A_{\rm lat}=4\pi(R_t/a)^2\) uses the fine-grid trapping radius \(R_t\) (Sec.~\ref{sec:star}) and the same site set in every comparison. If \(\mathsf Q\) is the propagated preparation sea and \(C-\mathsf Q\) consists of \(N_B\) occupied orbitals, their entropy difference costs at most \(N_B\log2\), a rank term already inside the direct budget, leaving the comparison between \(\mathsf Q\) and \(P\) (values in Ref.~\cite{McKinneyCode2026}). An excess over the held ground projector is a different reference comparison.}
```
Replacement:
```latex
\fnd{The reference-split comparison bounds \(|S_A(C)/A_{\rm lat}-\eta(ma)|\) by the flat finite-cut remainder \(e_{\rm flat}\), the comparator's own shift \(r_{\rm ref}=|S_A(P)-S_A(P_0)|/A_{\rm lat}\) for a comparator projector \(P\), and \(\min\{\nu,r_A\log2\}/A_{\rm lat}\), with \(A_{\rm lat}=4\pi(R_t/a)^2\) on the fine-grid trapping radius \(R_t\) (Sec.~\ref{sec:star}) and the same site set in every comparison, and the dust's \(N_B\) occupied orbitals cost at most \(N_B\log2\) (values in Ref.~\cite{McKinneyCode2026}).}
```

### 10.266 Sec. V: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_07 pair 1), beginning "at both cuts in two post-processing calculations on shared exports (de"
Replaced text:
```latex
at both cuts in two post-processing calculations on shared exports (definition
```
Replacement:
```latex
at both cuts (definition
```

### 10.267 Sec. V: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_07 pair 2), beginning "(the pair's residue subspaces fail two stability budgets in the second"
Replaced text:
```latex
(the pair's residue subspaces fail two stability budgets in the second calculation, values in Ref.~\cite{McKinneyCode2026})
```
Replacement:
```latex
(Ref.~\cite{McKinneyCode2026})
```

### 10.268 B: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_B pair 1), beginning "\paragraph{The bulk second-gradient term.} For the reference of Eq.~\e"
Replaced text:
```latex
\paragraph{The bulk second-gradient term.}
For the reference of Eq.~\eqref{eq:dm-reference}, in the bulk,
\begin{equation*}
\begin{aligned}W_2&=-\frac14\!\int\!\frac{F:\mathsf T:F}{N},\qquad F_{ij}=\partial_iv_j-2\epsilon_{ijk}b_k,\\
\mathsf T_{ij,kl}&=\!\int_p\!\Bigl[\frac{\delta_{ik}\cos^2(ap_i)q_jq_l}{\varepsilon^3}-\frac{q_iq_jq_kq_l}{\varepsilon^5}\Bigr],\end{aligned}
\end{equation*}
with \(\int_p\) the momentum integral over the zone at the prepared mass.
```
Replacement:
```latex
% repository pointer (pagebud item 25, lead inbox 96): the bulk second-gradient term W_2 = -(1/4) int F:T:F/N, F_ij = d_i v_j - 2 eps_ijk b_k, with the momentum-space response tensor T_ij,kl = int_p [delta_ik cos^2(a p_i) q_j q_l / eps^3 - q_i q_j q_k q_l / eps^5] over the zone at the prepared mass, is in Ref. McKinneyCode2026; the paragraph is archived verbatim in review_output/paperapi/REPO_DERIVATIONS_20261102.md. Sec. II C now names it as a momentum-space response tensor with the Ref. pointer.
```

### 10.269 B: replaced text, verbatim (pagebud Tier B, lead inbox 96 (FableAPI_pagebud_v1.md md5 a3703d8a): method and provenance detail kept by repository pointer; replaced text verbatim, replacement shown, spec TB_B pair 2), beginning "by Taylor's formula and the kernel's second moments. For \(F\in C^4\),"
Replaced text:
```latex
by Taylor's formula and the kernel's second moments. For \(F\in C^4\),
\((k*F)(x)=F(x)+\tfrac12\sigma^2\mathrm{tr}\,\nabla^2F(x)+R(x)\),
\(|R|\le\tfrac1{24}\sup_{y,|e|=1}|\partial_e^4F(y)|\int|z|^4k\).
```
Replacement:
```latex
by Taylor's formula and the kernel's second moments. The \(C^4\) expansion with its fourth-moment remainder, which supports the \(O((\ell/r)^4)\) remainder of Sec.~\ref{sec:results-exterior}, is in Ref.~\cite{McKinneyCode2026}.
```

### 10.270 Sec. VI D: the conditional Page curve subsection as it stood on the Tier B state before the emisF6 block replaced it (lead inbox 96 batch, commit (4'); verbatim, its 23 comment lines kept in the paper)
```latex
\subsection{A conditional Page curve and the correlations of its radiation}
\label{sec:page-effective}

A Page curve tracks the no-return entropy \(S_{\rm esc}\) in time at infinity~\cite{Page1993}. The fourth question is when the radiation entropy turns over and how high it reaches, if emission, storage, and release follow the laws assumed below. The model fixes its temperature and area normalization but no laws of emission, storage, and release, so the curve below is conditional on assumed laws, not derived object dynamics.

% provenance: Fig. 3 = fermion_new/figures/page_curve_amps.pdf (md5 1b053d204f4194719f405c226473dc60) = codex_solve page_curve_amps_R1.pdf; installed a55713ff364.
% provenance: producer <internal>/page_curve_amps_R1.py (md5 737b822b), staged runner, base <internal>/page_fig_r1.
% provenance: data page_curve_amps_R1.csv (md5 94f31b21); report <internal>/PAGE_FIGURE_R1_v1.md (maxima l.159-161).
% provenance: route 2 <internal>/page_curve_amps_R2{,_data}.py; comparison <internal>/PAGE_FIGURE_COMPARE_v1.md (all printed digits agree).
% provenance: comparator q in [0.001,3], k=1..8 x 64k, 1920 nodes; C_0 0.00523731547127, D_0 0.107876529936; spec WRITE_PAGE_SIGNED_v1 sec. 3.

% assumption: the store's activation time is part of the assumed laws (the store itself is introduced as assumed only in the next paragraph), and in this extension the light field's sites move with the flow, a departure from the fixed-lattice declaration.
% risk: medium
% GR slippage: no
% fold into text: yes, in the sentence ("In the assumed laws, store loading activates ..."); done in this pass
In the assumed laws the trapping radius \(R\) encloses \(E_R\), \(A_R=4\pi R^2\), store loading activates \(O(R_0\ln(1/\chi_0))\) in lattice time after matter crosses \(r_s\), and at activation \(R_0,E_0\), \(S_0=\pi R_0^2/G\), \(\chi_0=a/R_0\), and the elapsed retarded time \(t=u-u_a\) are defined, with \(\chi_0\to0\) at fixed \(ma,G/a^2,\eta_E\) and light-field radial sites moving with the flow, \(\dot r_n=v(r_n)\) (preceding fixed-site results keep their scope).

% assumption: the root-carried store of exp(fA_R/4G) states is assumed; the model supplies no accessible count of that size, so the turnover of the conditional curve is built in by this assumption.
% risk: high
% GR slippage: no
% fold into text: no; "We therefore assume" is stated
Purity makes a vanishing retained entropy ceiling force turnover,
but the model provides no accessible count of size \(fA_R/(4G)\).
Heavy particles supply only
\(O(N_B\ln(1/\chi_0))\) nats, about \(10^{-36}S_0\) at one
solar mass. We therefore assume a store of \(\lfloor e^{fA_R/(4G)}\rfloor\) states carried at the trapping radius (computed ceilings on the sign-reversed sector close this subsection),
\(f=4\eta(ma)G/a^2\), identifying the store coefficient with the cut coefficient as part of the assumption.
In stationary fields, partners at negative Killing energy turn inside the trapping radius (Sec.~\ref{sec:star}) into the light field's sign-reversed states. Exclusion supplies no release law.

% provenance: partner ledger S(B|X)>0 at a 2000a held root: <internal>/PARTNER_LEDGER_S2_webgpt_v1.md and <internal>/PL2_MANIFEST_routeA_solve_v1_ADDENDUM8.md; folded in a81972666e9.
The Page curve below rests on assumptions stated here once. First, two properties sign-reversed states do not supply: an accessible retained sector of logarithmic dimension at most $fA_R/(4G)$ with $o(S_0)$ retained entropy outside it, and transport of partner correlations to the no-return output, at the rate the minimum law below requires and followed by complete release, as the trapping radius recedes under emission, a recession the computed histories do not show (Sec.~\ref{sec:page}). On held fields partner accounting on held fields finds positive conditional entropy \(S(B|X)\) of a later escaped block given earlier output, so neither verifies an entropy-reducing addition (Ref.~\cite{McKinneyCode2026}).
\todo{the partner accounting: establish the accessible-sector bound
from sign-reversed-state occupation, deposition, and drift, including omitted sectors. Compute no-return conditional entropies, residual retained entropy, and coherent heavy-input recovery. Close the windowed trapping-radius energy budget and branch errors. Test the first controlled blocks
at sizes 32 and 64, then derive the thin-shell continuation through
turnover with controlled matching to the microscopic dynamics and
uniform error bounds.}

Second, connection to the object requires one massless
four-component Dirac radiation field and pure evolution leaving
a pure, input-independent retained state beyond the endpoint region \(1-t/t_*\lesssim\chi_0^3\).
The first law fixes area normalization, not state
count~\cite{Bekenstein1973,Hawking1975}. No completion time follows. Completion-independent maxima require disturbances within the
stated remainder bounds.

Third, the laws require one linear autonomous charge- and number-conserving generator supplying all referenced sources in Eq.~\eqref{eq:dr-radial}, heavy transitions funding every loading, control, and release operation beforehand, escaping energy counted once, \(dE_R/du=-L\) as a separate source requirement at the trapping radius (the shell balance below), and an input-independent outgoing channel that saturates the smaller of marginal budget and store capacity up to \(o(S_0)\): a minimum law assumed, not realized, since heavy-copy number conservation hides an occupation-pattern store's cross-sector coherences from the light output, so a realization needs a charge-matched code.
% RED MARK 1 (stretch goal, owner 1 Nov): construct the channel from the
% model in a charge-matched, copy-twirl-correctable code, store and
% capacity realized, not assumed. Frozen occupation-pattern codes are
% ruled out (E_dec >= 13/16 route 1, >= 0.174 route 2, tolerance 5e-4).
All retained content is released, not deleted (further assumptions in Ref.~\cite{McKinneyCode2026}).

% assumption: the depletion, marginal-entropy and minimum channel laws are assumed; the Page time and maximum of Eqs. page-effective-curve and page-effective-maximum follow from these assumptions and the first-law normalization, not from the object's dynamics.
% risk: high
% GR slippage: no
% fold into text: no; stated at every step of this subsection
% provenance: 0.6211/0.5105, 0.5755/0.4773, [0.5323,0.6146]: page_curve_amps R1 table (<internal>/PAGE_FIGURE_R1_v1.md:159-161), agreed by R2 (<internal>/PAGE_FIGURE_COMPARE_v1.md).
Assume quasi-steady \(L=\mathcal L/R^2\),
\(dE_R/du=-L\), and \(dS_{\rm marg}/du=\mathcal RL/T_H\).
With \(t_*=R_0^3/(6G\mathcal L)\) and \(y\equiv(E_R/E_0)^2\), integration gives \(R^3=R_0^3-6G\mathcal L\,t\), \(y=(1-t/t_*)^{2/3}\), and \(S_{\rm marg}/S_0=\mathcal R\,(1-y)\), and \(A_R/(4G)=S_0y\) is retained first-law entropy. Eq.~\eqref{eq:page-conditional}'s budget starts at activation.
Subadditivity gives \(S_{\rm esc}\le S_{\rm marg}\). Purity bounds it by the store's capacity \(fA_R/(4G)\) plus other retained entropy.
The assumed channel law gives the curve and its maximum,
\begin{equation}
\boxed{\begin{gathered}
S_{\rm esc}/S_0=\min\{\mathcal R(1-y),\;f\,y\}+o(1),\\
\frac{t_P}{t_*}=1-\Big(\frac{\mathcal R}{\mathcal R+f}\Big)^{3/2},\qquad
\frac{S_P}{S_0}=\frac{f\,\mathcal R}{\mathcal R+f},
\end{gathered}}
\label{eq:page-effective-curve}
\end{equation}
the first line holding uniformly through turnover, without a numerical remainder bound, and the second being the branch intersection, the Page time and the maximum. Thermal marginals supply only an upper bound (Fig.~\ref{fig:page-conditional}).
\todo{the curve from
a realized channel: saturated bounds, bounded remainder}
At \(f=1\) this is the Page time of a thermal emitter with entropy ratio \(\mathcal R\), and the cut share \(f\) is the one place the model's own number enters.

The finite-band estimates \(\mathcal L\simeq0.0052373\),
\(\mathcal R\simeq1.6391\) have no narrow absolute bound.
At \(f=1\) the maximum \(0.6211S_0\) occurs at \(0.5105t_*\) and
equals the remaining \(A_R/(4G)\). At \(a=2\ell_P,ma=0.2\), nominal
\(f=0.887\) gives \(0.5755S_0\) at \(0.4773t_*\) and the verified
\(0.7883<f<0.9831\) heights \([0.5323,0.6146]S_0\).
Stage-dependent rates require reintegration, not substitution in Eq.~\eqref{eq:page-effective-curve}.
```

### 10.271 Sec. VI D: the sign-reversed-state ceilings paragraph with its two provenance comments, removed when the emisF6 block absorbed its numbers (commit (4'))
```latex
% provenance: pocket ceilings 0.30/0.27, 0.28-0.31, 0.09-0.10/0.08-0.10, exponent 2.99: two routes per commit 73da652e7d6; candidates <internal>/STORE_DOS_R2_busi_v1.md, PK5 R1/R2; line-level producer not located.
% source: pocket ceilings about 0.3 (R/a 32 to 256) and about 0.1 (stored slices): route 1 SOLVE/results/capacity_r1<internal>/capII_*_v1.json (md5 58aebc27, 5b57d9d6, 29331480, aa52b8c5) and capsym_v1.json (6d41c4e8); route 2 CAPACITY_UPPER_webgpt_v1.md (383e5b62) :14-15; PROVENANCE_TRACE_v1 (317c6d92), lead inbox 69.
Two independent ceilings on the sign-reversed sector's Gibbs entropy at the sector's measured energy (a Gibbs variational ceiling, not an accessible state count), on the constant-charge vacuum profile as proxy for the object, are about 0.3 of \(fA_R/(4G)\) (\(f\) at \(ma=0.2\)) at \(R/a=32\) to 256 and about 0.1 on the stored slices
past the hydrodynamic description. They scale with area and lie below
the assumed store, whereas without the sector restriction both grow
with volume and exceed it at every computed size, and neither identifies the accessible sector (values in Ref.~\cite{McKinneyCode2026}). So under the assumed laws the radiation entropy peaks at \(0.62S_0\) near half the evaporation time, and the computed checks above test the budget's pieces on stored slices while establishing neither the store nor the release.

```

### 10.272 Sec. VI: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_06b pair 1), beginning "and, under assumed emission, storage, and release laws, when the radia"
Replaced text:
```latex
and, under assumed emission, storage, and release laws, when the radiation entropy turns over (Sec.~\ref{sec:page-effective}).
```
Replacement:
```latex
and, with the emission defined by the light field's own propagator and the supply, balance, capacity, and release of a comparator assumed, when that comparator's radiation entropy turns over (Sec.~\ref{sec:page-effective}).
```

### 10.273 Sec. VI: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_06b pair 2), beginning "\caption{Leading-order comparison curve from the assumed laws, not an "
Replaced text:
```latex
\caption{Leading-order comparison curve from the assumed laws, not an object history.
```
Replacement:
```latex
\caption{Comparator under the saturated-supply, balance, store, and release assumptions of Eq.~\eqref{eq:page-effective-curve}, not an object history, and the stored histories measure neither its lifetime nor its turnover.
```

### 10.274 07b: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_07b pair 1), beginning "Thermal marginals ensure neither monotone accumulated entropy nor a Pa"
Replaced text:
```latex
Thermal marginals ensure neither monotone accumulated entropy nor a Page curve. Here \(S_{\rm esc}\) is \todo{C1/C2 escaped-mode entropy from
two blindly and independently re-propagated pure covariances, with
output-return and omitted-mode errors}.
```
Replacement:
```latex
Thermal marginals ensure neither monotone accumulated entropy nor a Page curve. \(S_{\rm esc}\) is defined from the output covariance in Sec.~\ref{sec:page-effective}, its value on the stored histories is \todo{C1/C2 escaped-mode entropy from
two blindly and independently re-propagated pure covariances, with
output-return and omitted-mode errors}, and identification of a reported accounting with it requires documented packet and subtraction conventions.
```

### 10.275 Sec. V: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I2_07 pair 1), beginning "Each new output block adds its entropy minus its mutual information wi"
Replaced text:
```latex
Each new output block adds
its entropy minus its mutual information with earlier output to
\(S_{\rm esc}\).
```
Replacement:
```latex
Each new output block adds its conditional entropy given the earlier output to \(S_{\rm esc}\).
```

### 10.276 abstract: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_main5 pair 1), beginning "With emission, storage, and release laws assumed, stellar-mass radiati"
Replaced text:
```latex
With emission, storage, and release laws assumed, stellar-mass radiation entropy follows a Page curve peaking near half the evaporation time.
```
Replacement:
```latex
With supply, balance, area-sized capacity, and release assumed, a comparator, not the computed history, peaks near half its evaporation time.
```

### 10.277 Sec. I: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_02 pair 1), beginning "Section~\ref{sec:scaling} gives the astrophysical scaling and the Page"
Replaced text:
```latex
Section~\ref{sec:scaling} gives the astrophysical scaling and the Page curve conditional on assumed emission, storage, and release laws, the only place where laws are assumed rather than derived, and
```
Replacement:
```latex
Section~\ref{sec:scaling} gives the astrophysical scaling and the radiation entropy defined by the light field's own emission, beside a comparator whose supply, balance, capacity, loading, and release are assumed, the only place where laws are assumed rather than derived, and
```

### 10.278 Sec. VII: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_08 pair 1), beginning "and with them the Page time of Eq.~\eqref{eq:page-effective-curve}, th"
Replaced text:
```latex
and with them the Page time of Eq.~\eqref{eq:page-effective-curve}, the same formula as for any thermal emitter with that entropy ratio, with the cut share \(f\) as the one ingredient the model adds. Turnover of the conditional Page curve proves no complete evaporation, since full release requires decoupling from every retained sector, and the curve remains conditional on assumed laws, not a completion of the divergent constant-coefficient light-chain endpoint.
```
Replacement:
```latex
and with them the Page time of Eq.~\eqref{eq:page-effective-maximum}, the same formula as for any thermal emitter with that entropy ratio, with the cut share \(f\) as the one ingredient the model adds. Turnover of the conditional Page curve proves no complete evaporation, since full release requires decoupling from every retained sector, and Sec.~\ref{sec:page-effective} defines the radiation entropy from the full output covariance, its minimum-capacity curve a conditional comparator, not a computed object history and not a completion of the divergent constant-coefficient light-chain endpoint.
```

### 10.279 Sec. II: replaced text, verbatim (commit (4'), parts (ii)-(v) of the emisF6 delivery as signed with emisF4, lead inbox 91 and 96, spec I_03 pair 1), beginning "apart from the emission laws of Sec.~\ref{sec:page-effective}."
Replaced text:
```latex
apart from the emission laws of Sec.~\ref{sec:page-effective}.
```
Replacement:
```latex
apart from the comparator's supply, balance, capacity, and release assumptions of Sec.~\ref{sec:page-effective}.
```

### 10.280 Sec. VI: replaced text, verbatim (commit (4'), payment (b): the Sec. VI D firewall and pair-model paragraphs replaced by a one-sentence pointer before Fig. 3, spec I10_06 pair 1), beginning "The firewall question is which side, earlier radiation or retained reg"
Replaced text:
```latex
The firewall question is which side, earlier radiation or retained region, holds a late quantum's partner~\cite{AMPS2013}. The exchange-current response between the
emitted mode and either side measures that side (defined in Ref.~\cite{McKinneyCode2026}). Dephasing removes the response but not
the number anticorrelation, so it is no entanglement or monogamy
bound.
```
Replacement:
```latex
The firewall question~\cite{AMPS2013}, the exchange-current response that locates a late quantum's partner, and the independent-pair release model behind Fig.~\ref{fig:page-conditional}(b) are set out in Ref.~\cite{McKinneyCode2026}.
```

### 10.281 Sec. VI: replaced text, verbatim (commit (4'), payment (b): the Sec. VI D firewall and pair-model paragraphs replaced by a one-sentence pointer before Fig. 3, spec I10_06 pair 2), beginning "For independent pairs at fixed particle number, each with a single par"
Replaced text:
```latex
For independent pairs at fixed particle number, each with a single partner, a new partner is retained and a released partner lies in earlier radiation, so the response weight lies on the partner's side. New pairs add marginal entropy and released stored halves remove it, so the pair model releases none before the Page time and then the fraction \(\tfrac12(1+f/\mathcal R)\) of emitted marginal entropy, every admissible release order sharing this pre-Page zero and post-Page mean, other orders filling inventory-constrained bands. These statements need the pair model, not the entropy law alone, and centred averages crossing the Page time can be nonzero earlier.
```
Replacement:
```latex

```

### 10.282 B: replaced text, verbatim (commit (4'), payment (c): Appendix B compressed to the conservation statement, a two-sentence sketch, and the reference, spec TC_B pair 1), beginning "\paragraph{Euler's identity for the reference.} \(\mathcal W[\lambda u"
Replaced text:
```latex
\paragraph{Euler's identity for the reference.} \(\mathcal W[\lambda u]=\lambda\mathcal W[u]\) for \(\lambda>0\). Differentiating in \(\lambda\) at one gives \(\mathcal W=\langle u,D\mathcal W\rangle=\langle u,r\rangle\), and differentiating that identity in \(u\) gives \(\langle u,\mathcal W''\delta u\rangle=0\): the Hessian annihilates the fields, so \(\langle u,\delta r\rangle=0\) for every field variation.
```
Replacement:
```latex
% provenance: Appendix B compressed to the conservation statement, a two-sentence sketch, and the reference (lead's payment (c) for the 31-page cap, lead inbox 96 batch), keeping the two displays other sections cite (Eq. C-rate in Sec. IV A, Eq. dm-average in Sec. II D) and the nu_3 bound used in Appendix A; Euler's identity is folded into the sketch. The full appendix as it stood is archived verbatim in review_output/paperapi/REPO_DERIVATIONS_20261102.md.
```

### 10.283 B: replaced text, verbatim (commit (4'), payment (c): Appendix B compressed to the conservation statement, a two-sentence sketch, and the reference, spec TC_B pair 2), beginning "\paragraph{The Hartree flow conserves the boundary charge.} On a latti"
Replaced text:
```latex
\paragraph{The Hartree flow conserves the boundary charge.}
On a lattice with no outgoing bonds, at fixed kernel, let the radial fields solve
Eq.~\eqref{eq:dr-radial} differentiably in \(C\), with the angularly averaged sources of Eq.~\eqref{eq:dm-contact}, a regular centre, \(\mathcal M>0\) for \(r>0\), and \(N(\infty)=1\), the reference moments \(r=D_u\mathcal W\) taken at the back-averaged fields \(\bar u\). Varying the rows gives \(\partial_r(N\delta\mathcal M)=4\pi r^2(N\delta\rho_\ell+v\delta J_{\ell r})\). Integration and the adjoint averaging give \(\delta E=\langle\bar u,\delta s\rangle=\Tr(\bar h\,\delta C)-\langle\bar u,\mathcal W''[\bar u]\delta\bar u\rangle=\Tr(\bar h\,\delta C)\), the last term zero by Euler's identity.
Thus \(i\dot C=[\bar h,C]\) conserves \(E=\mathcal M(\infty)\):
\(dE/dt=-i\Tr(\bar h[\bar h,C])=0\).
```
Replacement:
```latex
On a lattice with no outgoing bonds, at fixed kernel, let the radial fields solve Eq.~\eqref{eq:dr-radial} differentiably in \(C\), with the angularly averaged sources of Eq.~\eqref{eq:dm-contact}, a regular centre, \(\mathcal M>0\) for \(r>0\), and \(N(\infty)=1\), the reference moments \(r=D_u\mathcal W\) taken at the back-averaged fields \(\bar u\). Then the Hartree flow \(i\dot C=[\bar h,C]\) conserves the boundary charge \(E=\mathcal M(\infty)\). Sketch: varying the rows gives \(\partial_r(N\delta\mathcal M)=4\pi r^2(N\delta\rho_\ell+v\delta J_{\ell r})\), and integration with the adjoint averaging gives \(\delta E=\langle\bar u,\delta s\rangle=\Tr(\bar h\,\delta C)-\langle\bar u,\mathcal W''[\bar u]\delta\bar u\rangle\). The last term vanishes by Euler's identity for the degree-one reference, \(\mathcal W[\lambda u]=\lambda\mathcal W[u]\) for \(\lambda>0\) giving \(\mathcal W=\langle u,r\rangle\) and \(\langle u,\mathcal W''\delta u\rangle=0\), so \(\delta E=\Tr(\bar h\,\delta C)\) and \(dE/dt=-i\Tr(\bar h[\bar h,C])=0\), the full derivation in Ref.~\cite{McKinneyCode2026}.
```

### 10.284 B: replaced text, verbatim (commit (4'), payment (c): Appendix B compressed to the conservation statement, a two-sentence sketch, and the reference, spec TC_B pair 3), beginning "\paragraph{The rate of the mass function at a trapping radius.} With \"
Replaced text:
```latex
\paragraph{The rate of the mass function at a trapping radius.}
With \(t_i=\sin^2(ap_i)\),
\(a^2|q|^2=\sum_it_i(1-t_i)<\sum_it_i=a^2(\varepsilon^2-m^2)\) except at
the eight valleys, where both sides vanish, so \(q_r=n\cdot q<\varepsilon\)
for every unit \(n\) wherever \(\varepsilon(p)>0\). Hence, for a
band-diagonal occupation \(f\ge0\) with \(\int_pf>0\), \(\rho=2\int_pf\varepsilon\),
\(J=2\int_pfq_r\) and the radial kinetic stress \(p_r=2\int_pfq_r^2/\varepsilon\),
the rate at a trapping radius \(v=-N\) is
```
Replacement:
```latex
At a trapping radius \(v=-N\), for a band-diagonal occupation \(f\ge0\) with \(\int_pf>0\), \(\rho=2\int_pf\varepsilon\), \(J=2\int_pfq_r\), and the radial kinetic stress \(p_r=2\int_pfq_r^2/\varepsilon\), the rate of the mass function is
```

### 10.285 B: replaced text, verbatim (commit (4'), payment (c): Appendix B compressed to the conservation statement, a two-sentence sketch, and the reference, spec TC_B pair 4), beginning "the valleys being a null set. This positive square is the rate supplie"
Replaced text:
```latex
the valleys being a null set. This positive square is the rate supplied
by the continuum evolution rows. The lattice transport weight and the
changing-reference source must be retained when identifying it with the
actual mass rate. For \(m>0\), maximizing the Cauchy bound
\((S-S^2/3)/[(ma)^2+S]\),
\(S=\sum_i t_i\in[0,3]\), gives
\(|q|/\varepsilon\le\nu_3=\sqrt{1+(ma)^2/3}-ma/\sqrt3<1\),
\(q_r\le\nu_3\varepsilon\), and \(\rho+p_r-2J\ge(1-\nu_3)(\rho-J)\). At
\(m=0\) only the strict inequality stands.
```
Replacement:
```latex
since \(q_r<\varepsilon\) wherever \(\varepsilon(p)>0\), the valleys being a null set. This positive square is the rate supplied by the continuum evolution rows, the lattice transport weight and the changing-reference source being retained when it is identified with the actual mass rate. For \(m>0\) the Cauchy bound \(|q|/\varepsilon\le\nu_3=\sqrt{1+(ma)^2/3}-ma/\sqrt3<1\) gives \(\rho+p_r-2J\ge(1-\nu_3)(\rho-J)\), and at \(m=0\) only the strict inequality stands (Ref.~\cite{McKinneyCode2026}).
```

### 10.286 B: replaced text, verbatim (commit (4'), payment (c): Appendix B compressed to the conservation statement, a two-sentence sketch, and the reference, spec TC_B pair 5), beginning "\paragraph{Back-averaging error.} The averaging kernel"
Replaced text:
```latex
\paragraph{Back-averaging error.}
The averaging kernel
```
Replacement:
```latex
The averaging kernel
```

### 10.287 Sec. VI: replaced text, verbatim (commit (4'), typographic overfull fix in the emisF6 block, spec I13d_06 pair 1), beginning "on the vacuum profile at \(R/a=32\)--256, in two independent calculati"
Replaced text:
```latex
on the vacuum profile at \(R/a=32\)--256, in two independent calculations,
```
Replacement:
```latex
on the vacuum profile at \(R/a=32\) to 256, in two independent calculations,
```

### 10.288 Sec. VI: replaced text, verbatim (commit (4'), typographic overfull fix in the emisF6 block, spec I13d_06 pair 2), beginning "What the delivery moved to Ref. McKinneyCode2026 is listed in the deli"
Replaced text:
```latex
What the delivery moved to Ref. McKinneyCode2026 is listed in the delivery file under "Moved to Ref.".
```
Replacement:
```latex
What the delivery moved to Ref. McKinneyCode2026 is listed in the delivery file under "Moved to Ref.". Typographic: "R/a = 32--256" set as "32 to 256" to remove a 1.5 pt overfull line.
```

### 10.289 Sec. II: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_03 pair 1), beginning "With \(\mathsf j_{xy}[h]=i(\Pi_xh\Pi_y-\Pi_yh\Pi_x)\), its unweighted "
Replaced text:
```latex
With
\(\mathsf j_{xy}[h]=i(\Pi_xh\Pi_y-\Pi_yh\Pi_x)\), its unweighted clock-flux vertex is \(\mathsf s_{x,ij}=\tfrac a4\{Q_i,\mathsf j_{x,x+ae_j}[h_0]+\mathsf j_{x-ae_j,x}[h_0]\}\) and its prepared-band projection \(\mathsf s^{\rm kin}_{x,ij}=P_\star\mathsf s_{x,ij}P_\star+(I-P_\star)\mathsf s_{x,ij}(I-P_\star)\).
This equally assigns adjacent bond fluxes, absent bonds zero:
the exact \(p_{x,i}\) rate is minus the bond sum of
\(\{Q_i,\mathsf j_{xy}[h]\}/2\), plus the local force
\(i\{\Pi_x,[h,Q_i]\}/2\). The clock weights each \(h_0\) flux by
\((N_x+N_y)/2\), the flow retains its stated
\(\cos(2ap_j)\) advection on uniform fields, and this bond divergence,
not a site difference, must be used in the balance
(bond-balance check in Ref.~\cite{McKinneyCode2026}).
For any one-body \(B\), set
\(a^3S^{\rm op}_{x,ij}[B]=\Tr(B\mathsf s_{x,ij})\) and
\(a^3S^{\rm kin}_{x,ij}[B]=\Tr(B\mathsf s^{\rm kin}_{x,ij})\).
Both receive the source kernel and spherical angular average.
```
Replacement:
```latex
Its vertex \(\mathsf s_{x,ij}\) is the symmetrized flux of \(Q_i\) on the two bonds at \(x\) along \(j\), equally assigned, absent bonds zero, each bond flux weighted by the mean of its two site clocks, and \(\mathsf s^{\rm kin}_{x,ij}\) is its prepared-band projection, so that the exact \(p_{x,i}\) rate is minus the bond sum of this flux plus a local force, a bond divergence and not a site difference, which the balance must use (explicit vertex, projection, and bond-balance check in Ref.~\cite{McKinneyCode2026}). For any one-body \(B\), set \(a^3S^{\rm op}_{x,ij}[B]=\Tr(B\mathsf s_{x,ij})\) and \(a^3S^{\rm kin}_{x,ij}[B]=\Tr(B\mathsf s^{\rm kin}_{x,ij})\), both receiving the source kernel and spherical angular average.
```

### 10.290 Sec. III: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_04 pair 1), beginning "At fixed physical \(m\), the match uses the flat sea's massless coeffi"
Replaced text:
```latex
At fixed physical \(m\), the match uses the flat sea's massless coefficient and its interval bound: every planar cut has \(\eta(\zeta)-\eta(0)=\frac{\zeta^2}{3\pi}\log\zeta^2+b_{\boldsymbol n}\zeta^2+o(\zeta^2)\), the logarithm from each valley's continuum Dirac field between momenta \(m\) and \(a^{-1}\), with directional finite parts \(b_{\boldsymbol n}\) between \(-0.42\) and \(-0.46\) and a sphere mean near \(-0.46\) without a bound (its angular remainder not proved uniform), entering only the \(10^{-32}\)-level spacing shift that follows (values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
At fixed physical \(m\), the match uses the flat sea's massless coefficient and its interval bound, every planar cut having the mass correction of Eq.~\eqref{eq:dr-area}, the logarithm from each valley's continuum Dirac field between momenta \(m\) and \(a^{-1}\), with directional finite parts between \(-0.42\) and \(-0.46\) and a sphere mean near \(-0.46\) without a bound (its angular remainder not proved uniform), entering only the \(10^{-32}\)-level spacing shift that follows (values in Ref.~\cite{McKinneyCode2026}).
```

### 10.291 Sec. III: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_04 pair 2), beginning "At \(k=0\), \(\dot k=-m\nabla N\), so marginally bound continuum dust "
Replaced text:
```latex
At \(k=0\),
\(\dot k=-m\nabla N\), so marginally bound continuum dust with
constant \(N=1\) is comoving, the massless band having no rest point.
At constant coupled clock, the equivalence principle holds exactly for the
elementary massive valley ray comoving at zero relative momentum.
For a small composite of fixed internal state it holds only up to the lattice free-fall factor \(\Upsilon^2\), \(\Upsilon=\langle\cos(2ak_i)\rangle\) over internal momentum (centre Hamiltonian in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
At \(k=0\), \(\dot k=-m\nabla N\), so marginally bound continuum dust with constant \(N=1\) is comoving, the massless band having no rest point. At constant coupled clock, the equivalence principle holds exactly for the elementary massive valley ray comoving at zero relative momentum, and for a small composite of fixed internal state only up to a lattice free-fall factor \(\Upsilon^2\), the mean of \(\cos(2ak_i)\) over internal momentum (centre Hamiltonian in Ref.~\cite{McKinneyCode2026}).
```

### 10.292 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_05 pair 1), beginning "Width controls re-evolve the same preparation at \(\tau=8\), averaging"
Replaced text:
```latex
Width controls re-evolve the same
preparation at \(\tau=8\), averaging sources and fields at
\(\sigma/a=2\), 4, and 8, between \(a\ll\sigma\) and the critical width above which any \(0\le\mathcal M\le E\) at \(N=1\) has no back-averaged trapping zero (its value in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
Width controls re-evolve the same preparation at \(\tau=8\), averaging sources and fields at \(\sigma/a=2\), 4, and 8, below the critical width above which no back-averaged trapping zero exists (its definition and value in Ref.~\cite{McKinneyCode2026}).
```

### 10.293 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_05 pair 2), beginning "Let \(A_\mu\) be the energy and current vertices and \(V_{\mu\nu}\) th"
Replaced text:
```latex
Let \(A_\mu\) be the energy and current vertices and \(V_{\mu\nu}\)
the Hessian of \(E[C]\) in bare moments \(\Tr(A_\mu C)\) at
\(\bar C\), including reference response. Set \(\Delta_\star=C-P_\star\), the Wick difference against the prepared sea (the source \(\Tr(A_\mu C)-r_\mu\) is \(\Tr(A_\mu\Delta_\star)\) less the polarization moment), \(\bar\Delta_\star=\bar C-P_\star\),
\(h_H[X]=\sum_{\mu\nu}V_{\mu\nu}A_\mu\Tr(A_\nu X)\) and
\(\widehat V_\star=\tfrac12\sum_{\mu\nu}V_{\mu\nu}
:(c^\dagger A_\mu c)(c^\dagger A_\nu c):_{P_\star}\).
The reduced second-order Hamiltonian is \(\widehat E_2=\mathcal E_H+:c^\dagger\bigl(h[\bar C]-h_H[\bar\Delta_\star]\bigr)c:_{P_\star}+\widehat V_\star\), \(h[\bar C]\) using \(\bar C\)'s fields and \(\mathcal E_H\) the constant fixed by the expansion point (Ref.~\cite{McKinneyCode2026}). Its Gaussian pair expectation's Hartree and Fock terms cancel for a single particle outside the sea, \(\Delta_\star=|u\rangle\langle u|\), \(P_\star u=0\): no one-particle pair self-attraction.
```
Replacement:
```latex
Let \(A_\mu\) be the energy and current vertices and \(V_{\mu\nu}\) the Hessian of \(E[C]\) in bare moments \(\Tr(A_\mu C)\) at \(\bar C\), including reference response, with \(\Delta_\star=C-P_\star\) the Wick difference against the prepared sea and \(\bar\Delta_\star=\bar C-P_\star\). The reduced second-order Hamiltonian \(\widehat E_2\) is the constant fixed by the expansion point plus the Wick-ordered one-body term \(h[\bar C]-h_H[\bar\Delta_\star]\), \(h_H[X]=\sum_{\mu\nu}V_{\mu\nu}A_\mu\Tr(A_\nu X)\), plus the Wick-ordered pair vertex \(\widehat V_\star\) built from \(V\) (explicit form in Ref.~\cite{McKinneyCode2026}). Its Gaussian pair expectation's Hartree and Fock terms cancel for a single particle outside the sea: no one-particle pair self-attraction.
```

### 10.294 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_05 pair 3), beginning "The second-order comparator energy is nonpositive. Full Wick ordering "
Replaced text:
```latex
The second-order comparator energy is nonpositive. Full Wick ordering makes the first-order expectation in \(P_\star\) zero. Against the ground projector \(P_g\) of \(h[\bar C]\) as comparator, on a gapped path the one-body term \(B=-h_H[\bar\Delta_\star]\) and the quartic term create orthogonal single and double particle-hole states, and the connected comparator energy \(\Delta E_{\rm ref}^{(2)}\), apart from \(\mathcal E_H\), is minus a sum of squared single and double amplitudes over their excitation energies, strictly negative unless both vanish, the single sum absent only if \(P_gB(I-P_g)=0\) (order counting in \(G\) in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The second-order comparator energy is nonpositive: full Wick ordering makes the first-order expectation in \(P_\star\) zero, and against the ground projector \(P_g\) of \(h[\bar C]\) on a gapped path the one-body term \(B=-h_H[\bar\Delta_\star]\) and the quartic term create orthogonal single and double particle-hole states, so the connected comparator energy \(\Delta E_{\rm ref}^{(2)}\), apart from \(\mathcal E_H\), is minus a sum of squared amplitudes over excitation energies, strictly negative unless both vanish, the single sum absent only if \(P_gB(I-P_g)=0\) (order counting in \(G\) in Ref.~\cite{McKinneyCode2026}).
```

### 10.295 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_05 pair 4), beginning "but the first-order dressing large (squared norm 0.14 and 1.1, almost "
Replaced text:
```latex
but the first-order dressing large (squared norm 0.14 and 1.1, almost all from the \(B\) singles, which nothing balances on the matter-free comparator), so the expansion about that comparator has no small-dressing control at 15.0, is not stationary under \(\widehat E_2\), and its dressed state's branch weight is untested (values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
but the first-order dressing large, almost all from the \(B\) singles, which nothing balances on the matter-free comparator, so the expansion about that comparator has no small-dressing control at 15.0, is not stationary under \(\widehat E_2\), and its dressed state's branch weight is untested (squared norms in Ref.~\cite{McKinneyCode2026}).
```

### 10.296 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_05 pair 5), beginning "The hydrodynamic hierarchy bounds the averaged source \(\rho\), not \("
Replaced text:
```latex
The hydrodynamic hierarchy bounds the
averaged source \(\rho\), not \(\rho_x\): the strain bound
\(\sigma\mathcal K\leq0.1\) caps \(\rho a^4\) at \(0.0044(a/\sigma)^2\)
at the spacing of Eq.~\eqref{eq:page-share}, below \(10^{-4}\) at
\(\sigma=4a\) of exclusion's kinematic bound \(4.96/a^4\) on the
prepared-sea normal-ordered local energy density at \(ma=0.2\).
```
Replacement:
```latex
The hydrodynamic hierarchy bounds the averaged source \(\rho\), not \(\rho_x\), the strain bound \(\sigma\mathcal K\leq0.1\) capping \(\rho a^4\) four orders of magnitude below exclusion's kinematic bound at \(\sigma=4a\) and the spacing of Eq.~\eqref{eq:page-share} (values in Ref.~\cite{McKinneyCode2026}).
```

### 10.297 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_05 pair 6), beginning "The computed cloud lies above it: \(mN_B/E_{\max}=\zeta^2(r_s/a)\sqrt "
Replaced text:
```latex
The computed cloud lies above it: \(mN_B/E_{\max}=\zeta^2(r_s/a)\sqrt g/(2c_E)=4.71\) at the nominal
```
Replacement:
```latex
The computed cloud lies above it: \(mN_B/E_{\max}=4.71\) at the nominal
```

### 10.298 Sec. VI: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_06b pair 1), beginning "0.50 nats at \(r_s/a=64\), \(q=1\), and 0.30 on the exact 50 percent s"
Replaced text:
```latex
0.50 nats at
\(r_s/a=64\), \(q=1\), and 0.30 on the exact 50 percent segment, a
mark on the entropy axis of the Page curve, not the computed
history's entropy.
```
Replacement:
```latex
0.50 nats at \(r_s/a=64\), \(q=1\) (0.30 on the exact 50 percent segment), a mark on the entropy axis of the Page curve, not the computed history's entropy.
```

### 10.299 Sec. V: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_07 pair 1), beginning "and purity bounds the exterior excess's upper limit by \(\ln2\) per oc"
Replaced text:
```latex
and purity bounds the exterior excess's upper limit by \(\ln2\) per occupied bound level of the deviation, whose continuum part leaves every finite site set when the non-bound spectrum is absolutely continuous (details in Ref.~\cite{McKinneyCode2026}). No drain rate follows.

On evolving Hartree fields,
```
Replacement:
```latex
and purity bounds the exterior excess's upper limit by \(\ln2\) per occupied bound level of the deviation (the continuum part's escape under an absolutely continuous non-bound spectrum in Ref.~\cite{McKinneyCode2026}). No drain rate follows. On evolving Hartree fields,
```

### 10.300 Sec. V: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_07 pair 2), beginning "for \(k=1,2\) the shell flow \(L_{\rm col}\), energy flow through \(16"
Replaced text:
```latex
for \(k=1,2\) the shell flow \(L_{\rm col}\),
energy flow through \(16r_s\) per flat-slice clock plotted at slice time, compared in Fig.~\ref{fig:page-v4}(b) with the two-channel
greybody flux \(L_H\) at \(\kappa_T/(2\pi)\), \(\kappa_T\) the slice's fine-grid surface slope, and at the stored slice charge's \(T_H\), is
at most 0.15 of the slope comparison flux before 36,
```
Replacement:
```latex
for \(k=1,2\) the shell flow through \(16r_s\) per flat-slice clock, compared in Fig.~\ref{fig:page-v4}(b) with the two-channel greybody flux at the slice's fine-grid surface slope and at the stored slice charge's \(T_H\), is at most 0.15 of the slope comparison flux before 36,
```

### 10.301 Sec. V: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I4_07 pair 3), beginning "the count still rising. At 15, 611 were inside and 29 outside the trap"
Replaced text:
```latex
the count still rising. At 15, 611 were inside and 29 outside the
trapping radius, so unitarity puts at least \((\sqrt{50.7}-\sqrt{29})^2=3.0\),
or 0.49 percent, of that interior component beyond \(16r_s\) at 56.
```
Replacement:
```latex
the count still rising, and since 611 were inside and 29 outside the trapping radius at 15, unitarity puts at least 0.49 percent of that interior component beyond \(16r_s\) at 56 (the bound in Ref.~\cite{McKinneyCode2026}).
```

### 10.302 Appendix A: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I5_A2 pair 1), beginning "On the finite lattice \(\Lambda_L=(\mathbb Z/L\mathbb Z)^3\), \(L\) ev"
Replaced text:
```latex
On the finite lattice \(\Lambda_L=(\mathbb Z/L\mathbb Z)^3\), \(L\)
even, with antiperiodic shifts and \(\epsilon=1/L\), let
\(P_0^{(L)}\) be the flat projector at the same mass
\(0\le\zeta\le\zeta_*\) (it has no zero mode, including at
\(\zeta=0\)). For \(n_*\le\bar N\le n^*\), \(|\bar v|\le v^*\),
\(n_*-v^*\ge c_*>0\),
\(\sum_{\mathbf q}|\mathbf q|(|\widehat N(\mathbf q)|+\sum_i|\widehat v_i(\mathbf q)|)\le\mathcal F_1\epsilon\),
a nonwrapping ball \(c_rL\le r\le0.45L\), and
\(P=\mathbb 1_{(-\infty,0)}(h)\) with any orthogonal zero-mode
filling, \(|S_A(P)-S_A(P_0^{(L)})|\le CA_{\rm lat}\epsilon^{1/10}\)
with \(C\) depending only on
\((\zeta_*,\mathcal F_1,c_*,n^*,v^*,c_r)\). The same holds for a
comparator projector with the stated spectral occupations. The proof (Ref.~\cite{McKinneyCode2026}) splits at \(\delta=\epsilon^{2/5}\) into a rank part and a norm part, giving the exponent \(1/10\), and bounds the flat finite-lattice entropy difference so that A.1 gives \(\eta(\zeta)\) in the fixed-mass limit.
```
Replacement:
```latex
On the finite lattice \(\Lambda_L=(\mathbb Z/L\mathbb Z)^3\), \(L\) even, with antiperiodic shifts and \(\epsilon=1/L\), let \(P_0^{(L)}\) be the flat projector at the same mass \(0\le\zeta\le\zeta_*\), which has no zero mode. For fields bounded as in (H1) with \(n_*-v^*\ge c_*>0\), a weighted Fourier norm of their gradients at most \(\mathcal F_1\epsilon\), a nonwrapping ball \(c_rL\le r\le0.45L\), and \(P=\mathbb 1_{(-\infty,0)}(h)\) with any orthogonal zero-mode filling, \(|S_A(P)-S_A(P_0^{(L)})|\le CA_{\rm lat}\epsilon^{1/10}\) with \(C\) depending only on \((\zeta_*,\mathcal F_1,c_*,n^*,v^*,c_r)\), and the same holds for a comparator projector with the stated spectral occupations (hypotheses in full and the proof, which splits at \(\delta=\epsilon^{2/5}\) into a rank part and a norm part and bounds the flat finite-lattice entropy difference so that A.1 gives \(\eta(\zeta)\) in the fixed-mass limit, in Ref.~\cite{McKinneyCode2026}).
```

### 10.303 Appendix A: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I5_A2 pair 2), beginning "At and across the trapping radius. By Cauchy's inequality on the latti"
Replaced text:
```latex
At and across the trapping radius. By Cauchy's inequality on the
lattice momenta (Ref.~\cite{McKinneyCode2026}), where
\(|\bar v(x)|\le s\bar N(x)\) the frozen gap is at least
\(\bar N(x)\mathfrak g_s(\zeta)\),
\(\mathfrak g_s(\zeta)=\min_{0\le S\le3}[\sqrt{\zeta^2+S}-s\sqrt{S-S^2/3}]\),
positive at \(s=1\), the trapping radius included, for \(\zeta>0\). Where \(|\bar v(x)|>\bar N(x)\) it is positive exactly below the
sign-reversal threshold of Sec.~\ref{sec:results-vacuum} in the
flow's direction, which is at least \(w_c^{(3)}(\zeta)=1/\nu_3>1\).
```
Replacement:
```latex
At and across the trapping radius, by Cauchy's inequality on the lattice momenta (Ref.~\cite{McKinneyCode2026}), where \(|\bar v(x)|\le s\bar N(x)\) the frozen gap is at least \(\bar N(x)\mathfrak g_s(\zeta)\), positive at \(s=1\), the trapping radius included, for \(\zeta>0\) (the closed form of \(\mathfrak g_s\) in Ref.~\cite{McKinneyCode2026}), and where \(|\bar v(x)|>\bar N(x)\) it is positive exactly below the sign-reversal threshold of Sec.~\ref{sec:results-vacuum} in the flow's direction.
```

### 10.304 Appendix C: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec G_C pair 1), beginning "Varying \(\mathcal A\) in \(N\) and in \(v\) gives \(\rho_\ell=-\mathc"
Replaced text:
```latex
Varying \(\mathcal A\) in \(N\) and in \(v\) gives
\(\rho_\ell=-\mathcal A/N-(c/8\pi G)\nabla\cdot(\nabla N/N)\) and
\(J_\ell=-\tfrac{1}{8\pi G}\nabla\cdot\bigl[a_1(\operatorname{tr}K)I+a_2K
+a_3\operatorname{Diag}K+b\,\Omega\bigr]\), \(\operatorname{Diag}K\)
the diagonal part of \(K\) and \((\nabla\cdot M)_j=\sum_i\partial_iM_{ij}\). At \(a_1=-1\), \(a_2=1\), \(a_3=b=c=0\) they are Eq.~\eqref{eq:dm-h2}.
Euler's identity for the degree-one \(\mathcal A\),
\(N\rho_\ell+v\cdot J_\ell=\mathcal A+\nabla\cdot F_{\mathcal A}\) with
\(F_{\mathcal A,i}=-\bigl(N\,\partial\mathcal A/\partial(\partial_iN)
+\sum_jv_j\,\partial\mathcal A/\partial(\partial_iv_j)\bigr)\),
is the divergence by which \(\mathcal A_0\) and \(\mathcal A\) differ.
```
Replacement:
```latex
Varying \(\mathcal A\) in \(N\) and in \(v\) gives the energy law \(\rho_\ell=-\mathcal A/N-(c/8\pi G)\nabla\cdot(\nabla N/N)\) and the momentum law as the divergence of the strain and rotation terms with their five coefficients (explicit form in Ref.~\cite{McKinneyCode2026}), which at \(a_1=-1\), \(a_2=1\), \(a_3=b=c=0\) are Eq.~\eqref{eq:dm-h2}. Euler's identity for the degree-one \(\mathcal A\), \(N\rho_\ell+v\cdot J_\ell=\mathcal A+\nabla\cdot F_{\mathcal A}\), with \(F_{\mathcal A}\) built from the derivatives of \(\mathcal A\) with respect to the field gradients, is the divergence by which \(\mathcal A_0\) and \(\mathcal A\) differ.
```

### 10.305 Sec. IV: replaced text, verbatim (commit (4'), payment ladder rungs 1-3 (the pairs still present after Tiers A and B), method details to Ref. McKinneyCode2026, spec I5_05 pair 1), beginning "The interior's Hartree-Fock level is read on 18 stored slices of the \"
Replaced text:
```latex
The interior's Hartree-Fock level is read on 18 stored slices of the \(\sigma/a=4\) and 8 runs past the limit of the hydrodynamic description, in two independent calculations on four. Every operator of this paragraph is a held-source operator, built from the fixed reference projector's response (the comparator projector of Sec.~\ref{sec:results-lattice}) instead of Eq.~\eqref{eq:dm-reference}'s reference response
```
Replacement:
```latex
The interior's Hartree-Fock level is read on 18 stored slices of the \(\sigma/a=4\) and 8 runs past the limit of the hydrodynamic description, in two independent calculations on four, every operator of this paragraph being a held-source operator built from the fixed reference projector's response (the comparator projector of Sec.~\ref{sec:results-lattice}) instead of Eq.~\eqref{eq:dm-reference}'s reference response
```

### 10.306 Sec. III: replaced text, verbatim (commit (5'), lead inbox 78, evolved-cut budget three-term split from FableAPI_evcutF2_v1.md (md5 7bbe777e) with AstraAPI_evcutA2_v1.md (md5 08c917ed), spec I6_04 pair 1), beginning "gives that cross-block condition through the massive gap (details in R"
Replaced text:
```latex
gives that cross-block condition through the massive gap (details in Ref.~\cite{McKinneyCode2026}). Its pass on the computed collapse histories is reported in Sec.~\ref{sec:results-diagnostics}.}
```
Replacement:
```latex
gives that cross-block condition through the massive gap (details in Ref.~\cite{McKinneyCode2026}).}

% provenance: Sec. III C three-term split of the direct-state budget, the Sec. IV E pointer, the Sec. V E sentence, and the Discussion sentences from <internal>/FableAPI_evcutF2_v1.md (md5 7bbe777e, section (b)) with every correction of AstraAPI_evcutA2_v1.md (md5 08c917ed) accepted (K_0 size-dependent, gamma_1 = 0.392 at ma = 0.2, (H1)-(H3) with the field-rate bounds); lead inbox 78. The delivery's anchor "The trapping-cut budget passes" no longer existed verbatim, the equivalent span being the Sec. III C pointer sentence "Its pass on the computed collapse histories is reported in Sec. IV E" and the Sec. IV E direct-state paragraph, both replaced; plain-word map applied (records -> stored slices, routes -> independent calculations, dense root -> fine-grid trapping radius, collar -> shell, certifies -> establishes), the one prose semicolon split.
For pure \(C\), in the same lattice units, the test is exactly \(E_f(P_0)+\Tr[(C-P_0)fh_0f]\), with \(E_f(P_0)=\gamma_1\sum_{x,i}|f_{x+e_i}-f_x|^2\) and \(\gamma_1=\int_p\sin^2p_1/\varepsilon\), 0.392 at \(ma=0.2\), the flat sea's own window energy, and the massive gap gives \(\|\Pi_A(C-P_0)\|_2^2\le[\sqrt{E_f(C)}+\sqrt{E_f(P_0)}]^2/m\), the cross-block condition above. For the harmonic window, equal to one on the ball of radius \(L\) and \(2L/|x|-1\) out to \(2L\), the gradient sum is \(8\pi L+O(1)\), so the window term is \(8\pi\gamma_1L+O(1)\), about \(9.9L\) at large \(L\), with its finite-size value computed exactly at each size, and the \(N_B\) dust orbitals contribute at most \(\varepsilon_{\max}N_B=\varepsilon_{\max}L/(2g\,ma\,\eta_E)\), \(18.9L\) at the reference parameters (Sec.~\ref{sec:star}), for every state and every size of the family. The budget therefore rests on the propagated sea's window energy \(\Tr[(\mathsf Q-P_0)fh_0f]\). This term is \(O(L)\) on intervals \(t\le T_*L\) along which the fields obey Appendix~\ref{app:local}'s bounds on the scale \(L\) together with matching bounds on their time derivatives, by the prepared band's adiabatic expansion, while the computed histories leave the spatial bounds before formation (Sec.~\ref{sec:results-lattice}) and the held fields' step jumps lie outside the rate bounds, so the expansion establishes none of them. On the angular-channel chains the full budget reads \(K\le7.07\) over 44 histories and \(K\le7.49\) over 20 stored slices at \(r_s/a=32\), 64, 256, with no growth in \(1/\chi\) and matched-clock values agreeing within 2 percent, and the sea's term has no cubic reading. \todo{the propagated sea's window energy and cross block on the cubic digital ball at the fine-grid trapping radius, two blind independent calculations at \(r_s/a=32\) and 64 with the quiet exterior start, box and stored-slice controls of the predeclared gate, paired with the chain readings}
```

### 10.307 05b: replaced text, verbatim (commit (5'), lead inbox 78, evolved-cut budget three-term split from FableAPI_evcutF2_v1.md (md5 7bbe777e) with AstraAPI_evcutA2_v1.md (md5 08c917ed), spec I6_05b pair 1), beginning "\fnd{On the computed collapse histories the direct-state test of Sec.~"
Replaced text:
```latex
\fnd{On the computed collapse histories the direct-state test of Sec.~\ref{sec:results-vacuum} passes with a bounded constant, \(K\) at most 7.5 at \(r_s/a=32\), 64, 256 with no observed growth in \(r_s/a\) (counts and matched-clock agreement in Ref.~\cite{McKinneyCode2026}). The tested covariance includes the propagated sea, so no separate reference-transfer estimate is needed, and these are bounds on the sampled histories, not a uniform theorem over every regular history or a massless-limit estimate.}
```
Replacement:
```latex
\fnd{The direct-state budget of Sec.~\ref{sec:results-vacuum} passes on the angular-channel chains at every tested size, its propagated-sea term unread on the cubic cut, and its bounds hold for the sampled histories, not as a uniform theorem over every regular history or a massless-limit estimate.}
```

### 10.308 Sec. V: replaced text, verbatim (commit (5'), lead inbox 78, evolved-cut budget three-term split from FableAPI_evcutF2_v1.md (md5 7bbe777e) with AstraAPI_evcutA2_v1.md (md5 08c917ed), spec I6_07 pair 1), beginning "At fixed \(ma>0\), Sec.~\ref{sec:results-vacuum}'s direct comparison w"
Replaced text:
```latex
At fixed \(ma>0\), Sec.~\ref{sec:results-vacuum}'s direct comparison would give the flat leading coefficient on evolved cubic cuts if the cubic budget \(E_f(C)\le KL\) held uniformly in size with size-independent window constants, and the reported budgets are chain readings, not an established uniform cubic premise, so no small evolved-area error follows at this size.
```
Replacement:
```latex
The budget's sea term is unread on this cubic cut, and at this size the budget's cross-block bound exceeds the area term through its shell rank in any case, so no small evolved-area error follows.
```

### 10.309 Sec. VII: replaced text, verbatim (commit (5'), lead inbox 78, evolved-cut budget three-term split from FableAPI_evcutF2_v1.md (md5 7bbe777e) with AstraAPI_evcutA2_v1.md (md5 08c917ed), spec I6_08 pair 1), beginning "the trapping surface is marginally trapped by its null expansions in t"
Replaced text:
```latex
the trapping surface is marginally trapped by its null expansions in the derived line element~\cite{ChoquetBruhat2009}. This is not an analogue model:
```
Replacement:
```latex
the trapping surface is marginally trapped by its null expansions in the derived line element~\cite{ChoquetBruhat2009}. A direct-state criterion for retaining the flat leading coefficient is satisfied on the sampled histories in their angular-channel description, and of its three terms the flat sea's own is explicit and the dust's is uniform over the scaling family, while the propagated sea's has no cubic reading and no uniform extension. This is not an analogue model:
```

### 10.310 Sec. II: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K3_03 pair 1), beginning "The propagated sea's departure from the prepared band, including its r"
Replaced text:
```latex
The propagated sea's departure from the prepared band, including its
reversible first adiabatic correction, gravitates. Its exact bookkeeping and the stress split follow below. Continuum conservation holds
```
Replacement:
```latex
The propagated sea's departure from the prepared band, including its reversible first adiabatic correction, gravitates. Continuum conservation holds
```

### 10.311 Sec. II: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K3_03 pair 2), beginning "Spherical geometry requires two fixed-shape energy tests at radius \(r"
Replaced text:
```latex
Spherical geometry requires two fixed-shape energy tests at radius \(r\), related by a cubic rotation about the centre, to agree. Their difference
does not commute with the sea projector. Its positive sea variance
tends to twice one test's as their separation grows with \(r\).
By clustering, normal states of the infinite incoming-sea
representation approach the sea on distant tests, excluding every
nonzero normal state from the strong unprojected spherical Gauss
constraints.
```
Replacement:
```latex
Spherical geometry requires two fixed-shape energy tests at radius \(r\), related by a cubic rotation about the centre, to agree, and their difference does not commute with the sea projector, its positive sea variance tending to twice one test's as their separation grows with \(r\), so by clustering every nonzero normal state of the infinite incoming-sea representation is excluded from the strong unprojected spherical Gauss constraints.
```

### 10.312 Sec. III: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K3_04 pair 1), beginning "\(H_s=H[\sin(\xi x_i)]\) have the bracket \begin{equation} i[H_c,H_s]="
Replaced text:
```latex
\(H_s=H[\sin(\xi x_i)]\) have the bracket
\begin{equation}
i[H_c,H_s]=F_a(\xi)c^\dagger Q_ic,\quad
F_a(\xi)=\frac{\sin(2a\xi)+2\sin(a\xi)}{4a},
\label{eq:dr-longitudinal-algebra}
\end{equation}
with \(F_a(\xi)/\xi=1-\tfrac5{12}(a\xi)^2+O((a\xi)^4)\): the energy
```
Replacement:
```latex
\(H_s=H[\sin(\xi x_i)]\) have the bracket \(i[H_c,H_s]=F_a(\xi)c^\dagger Q_ic\), \(F_a(\xi)=[\sin(2a\xi)+2\sin(a\xi)]/(4a)\), with \(F_a(\xi)/\xi=1-\tfrac5{12}(a\xi)^2+O((a\xi)^4)\): the energy
```

### 10.313 Sec. III: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K3_04 pair 2), beginning "\fnd{For \(ma>0\) the frozen gap is positive below the sign-reversal t"
Replaced text:
```latex
\fnd{For \(ma>0\) the frozen gap is positive below the sign-reversal threshold,
including the trapping radius. A global gap requires a pathwise
estimate: an open sign-reversed region makes \(\mathsf g_*<0\), so
Appendix~\ref{app:local}'s criterion A.5 cannot verify that
path. This does not imply an exact zero
eigenvalue on every finite lattice.} At \(ma=0.2\) that criterion is met with Appendix~\ref{app:local}'s field bounds and an independently supplied pathwise frozen-gap lower bound (values in Ref.~\cite{McKinneyCode2026}).
\fnd{The trapping-radius condition alone does not supply this global lower bound,
and these gap-criterion inputs have not been rigorously bounded on the
computed histories. The direct-state budget above uses the gap
of \(h_0\), not a gap of the evolving \(h\).}
```
Replacement:
```latex
\fnd{For \(ma>0\) the frozen gap is positive below the sign-reversal threshold, including the trapping radius, while a global gap requires a pathwise estimate, which an open sign-reversed region (\(\mathsf g_*<0\)) prevents Appendix~\ref{app:local}'s criterion A.5 from verifying, without implying an exact zero eigenvalue on every finite lattice.} At \(ma=0.2\) that criterion is met with Appendix~\ref{app:local}'s field bounds and an independently supplied pathwise frozen-gap lower bound (values in Ref.~\cite{McKinneyCode2026}). \fnd{The trapping-radius condition alone does not supply this global lower bound, these gap-criterion inputs have not been rigorously bounded on the computed histories, and the direct-state budget above uses the gap of \(h_0\), not a gap of the evolving \(h\).}
```

### 10.314 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K4_05 pair 1), beginning "for outward continuum light from a comoving emitter the computed value"
Replaced text:
```latex
for outward continuum light from a comoving emitter the computed value is \(g_D=\omega_D/\omega_{\rm em}\), \(\ln(1/g_D)=-\ln(d\tau_{\rm em}/dT_D)+\ln\sqrt{N_D^2-v_D^2}\) (subscript \(D\) for receiver quantities, while stationary fields would give \(N+v\) at emission for a receiver at infinity), on back-averaged fields
```
Replacement:
```latex
for outward continuum light from a comoving emitter the computed value is \(g_D=\omega_D/\omega_{\rm em}\), the emitter's clock rate against the receiver's static clock (formula in Ref.~\cite{McKinneyCode2026}, stationary fields giving \(N+v\) at emission for a receiver at infinity), on back-averaged fields
```

### 10.315 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K4_05 pair 2), beginning "reports \(\ln(1/g_D)\) at the maximum of \(L_T=-\ln(d\tau_{\rm em}/dT_"
Replaced text:
```latex
reports \(\ln(1/g_D)\) at the maximum of
\(L_T=-\ln(d\tau_{\rm em}/dT_D)\): the sampled redshifts reach
```
Replacement:
```latex
reports \(\ln(1/g_D)\) at the maximum of the emitter's clock rate: the sampled redshifts reach
```

### 10.316 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K4_05 pair 3), beginning "The inner shell's dust at \(|ap|>\pi/4\) is at most 15 percent, lattic"
Replaced text:
```latex
The inner shell's dust at \(|ap|>\pi/4\) is at most 15 percent, lattice
momenta the rows' drift can supply before saturation, and the
\(\pi/4\) split, a plane-wave criterion, reads neither bin as outgoing
content.
```
Replacement:
```latex
The inner shell's dust at \(|ap|>\pi/4\) is at most 15 percent, and the \(\pi/4\) split, a plane-wave criterion, reads neither bin as outgoing content.
```

### 10.317 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K5_05 pair 1), beginning "Channel \(k\) is a two-component chain at \(r_n=na\), \(n=1,\dots,n_{\"
Replaced text:
```latex
Channel \(k\) is a two-component chain at \(r_n=na\), \(n=1,\dots,n_{\rm ch}\), with operator \(h_{0,k}=m\sigma_3+\sigma_1\widehat p_r+\sigma_2k/r_n\), the two-step radial current, no origin-crossing hop, an open last site, and the eight valleys' weight \(16k\) (details and the chains' polarization counting in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
Channel \(k\) is a two-component radial chain with operator \(h_{0,k}=m\sigma_3+\sigma_1\widehat p_r+\sigma_2k/r_n\), an open last site, and the eight valleys' weight \(16k\) (the two-step current, the hopping rules, and the chains' polarization counting in Ref.~\cite{McKinneyCode2026}).
```

### 10.318 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K5_05 pair 2), beginning "The stepping is exact for the state and re-solves the fields at every "
Replaced text:
```latex
The stepping is exact for the state and re-solves the fields at every step: for step length \(\tau_j\), the accepted back-averaged fields
\(W_\ell^\dagger u_j\), \(u_j=(N_j,v_j)\), are held in
Eq.~\eqref{eq:dm-hamiltonian}'s chain operator, \(W_\ell^\dagger\) with the history's kernel, each covariance being conjugated by the exponential of that operator over the step, and the next fields solve the rows from the kernel-averaged moments of that state (Sec.~\ref{sec:model-dictionary}).
```
Replacement:
```latex
The stepping is exact for the state and re-solves the fields at every step: the accepted back-averaged fields are held in Eq.~\eqref{eq:dm-hamiltonian}'s chain operator over the step, each covariance being conjugated by its exponential, and the next fields solve the rows from the kernel-averaged moments of that state (Sec.~\ref{sec:model-dictionary}, step details in Ref.~\cite{McKinneyCode2026}).
```

### 10.319 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K5_05 pair 3), beginning "while formation is robust under the channel count (10 to 20 channels l"
Replaced text:
```latex
while formation is robust under the channel count (10 to 20 channels leave formation's slice unchanged and its radius within \(0.09a\)), so
```
Replacement:
```latex
while formation is robust under the channel count (values in Ref.~\cite{McKinneyCode2026}), so
```

### 10.320 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K5_05 pair 4), beginning "A prescribed Hermitian generator with locally time-integrable norm alr"
Replaced text:
```latex
A
prescribed Hermitian generator with locally time-integrable norm
already has a unitary propagator. This weaker statement does not
establish coupled-field continuation.
```
Replacement:
```latex
A prescribed Hermitian generator with locally time-integrable norm already has a unitary propagator, a weaker statement that does not establish coupled-field continuation.
```

### 10.321 Sec. VI: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K5_06 pair 1), beginning "At unit lapse the flow's redshift near the trapping radius permits sof"
Replaced text:
```latex
At unit lapse the flow's redshift near the trapping radius permits soft output but establishes no spectral weight, count, or rate (Eq.~\eqref{eq:dr-crossing-ray}). For a channel count bounded in \(\chi\), the channel-capacity bound permits \(p\ge2\) in the slowly evolving object's assumed luminosity \(L\propto\chi^p/a^2\) without establishing saturation or any nonzero rate.
```
Replacement:
```latex
At unit lapse the flow's redshift near the trapping radius permits soft output but establishes no spectral weight, count, or rate (Eq.~\eqref{eq:dr-crossing-ray}), and for a channel count bounded in \(\chi\) the channel-capacity bound permits \(p\ge2\) in the slowly evolving object's assumed luminosity \(L\propto\chi^p/a^2\) without establishing saturation or any nonzero rate.
```

### 10.322 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K6_05 pair 1), beginning "All three are verified through formation of the trapping surface, \(\s"
Replaced text:
```latex
All three are verified through formation of the trapping surface,
\(\sigma/a=2\) through 40.
Agreement requires formation clocks within \(0.5r_s\) and trapping radii within 10 percent at matched clocks. Point sampling the fields at the sites instead of back-averaging changes formation. All collapse statements below
use back-averaged coupling.
```
Replacement:
```latex
All three are verified through formation of the trapping surface, \(\sigma/a=2\) through 40, agreement requiring formation clocks within \(0.5r_s\) and trapping radii within 10 percent at matched clocks, and all collapse statements below use back-averaged coupling, since point sampling the fields at the sites instead changes formation.
```

### 10.323 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K6_05 pair 2), beginning "The cold formation comparison case is a pressureless cloud released fr"
Replaced text:
```latex
The cold formation comparison case is a pressureless cloud released from
coordinate rest at \(4r_s\), with \(2GE=59.05a\) and no shell
crossing. Its outer shell reaches \(R=2GE\) at \(T/r_s=13.22\),
where \(t_s\) diverges.
```
Replacement:
```latex
The cold formation comparison case, a pressureless cloud released from coordinate rest at \(4r_s\) with \(2GE=59.05a\) and no shell crossing, reaches \(R=2GE\) at \(T/r_s=13.22\), where \(t_s\) diverges.
```

### 10.324 Sec. IV: replaced text, verbatim (commit (5'), inbox-78 method trims in the pairs still present after the tiers, every claim, number, hedge, and todo kept, spec K6_05 pair 3), beginning "Width agreement holds only near the reference size, wider averaging de"
Replaced text:
```latex
Width agreement holds only near the reference size, wider averaging delaying formation and halving the step advancing it, size 32 and
```
Replacement:
```latex
Width agreement holds only near the reference size, size 32 and
```

### 10.325 Sec. VI: replaced text, verbatim (lead inbox 94, closeF2 mass-loss closure text (FableAPI_closeF2_v1.md md5 71937f98, AstraAPI_closeA_v1.md eb335afb): the emisF6 block sentence replaced and the Appendix B paragraph added, spec I15_06 pair 1), beginning "Assumed, pending the full-reference balance test \todo{B: two-field fu"
Replaced text:
```latex
Assumed, pending the full-reference balance test \todo{B: two-field full-reference shell balance with reference response, exterior storage, trapping-radius-gradient factor and sea mismatch below the balance accuracy}: sea and reference mismatch, exterior storage, the trapping-radius-gradient correction and the arrival-time conversion controlled well enough to use \(dE_R/du=-L\), \(L\) the light field's outgoing Killing-energy flux, with the trapping radius at \(R=2GE_R\), \(E_R\) the charge inside it, and a shell clock \(u=t-\Phi(R_c)\) of unit Jacobian only for the stationary exterior and clock convention specified. The omitted terms are not bounded at the balance accuracy, the vacuum shell holding for the dust comparison solution and unproved for the object's fountain.
```
Replacement:
```latex
% provenance: mass-loss closure (the balance between trapping radius and shell, the root form with the margin alpha_R and the defect, the window bound, and the stored-slice verdict) from <internal>/FableAPI_closeF2_v1.md (md5 71937f98, 96 lines; reviewer A accepted all four corrections of AstraAPI_closeA_v1.md, md5 eb335afb), lead inbox 94, replacing the block's "Assumed, pending the full-reference balance test" sentence; plain-word map applied (root -> trapping radius, collar -> shell, records -> stored histories, routes -> calculations), prose semicolons split, the verdict table of the delivery's section 5 archived in REPO_DERIVATIONS with a Ref. pointer and printed as one sentence; the delivery's own placement (Sec. VI E and an appendix paragraph) replaced by the lead's placement here, the appendix paragraph kept (Appendix B).
Between the trapping radius and a shell radius \(r_c\) the rows give, on histories that solve both constraint rows, the exact balance \(dE_R/dt=-L_c-dE_{\rm st}/dt+\mathcal D_c\) (Appendix~\ref{app:charge}), where \(L_c\) is the charge's flux through the shell, the Killing-energy flux where the shell is in vacuum, \(E_{\rm st}=\mathcal M(r_c)-E_R\) the energy stored between trapping radius and shell, and \(\mathcal D_c\) the source defect, which is the weighted defect beyond the shell when the charge is conserved there. The law \(dE_R/du=-L\), \(L\) the light field's outgoing Killing-energy flux, with the trapping radius at \(R=2GE_R\), \(E_R\) the charge inside it, and the shell clock \(u=t-\Phi(r_c)\) of unit Jacobian only for the stationary exterior and clock convention specified, therefore holds to relative accuracy \(\epsilon\) exactly when the storage rate, the defect, the charge drift and the mismatch between the shell clock and \(u\) each stay below \(\epsilon L\) over the window. A quasi-stationary stored state of the massless field puts the first at \(2\ell_c/t_{\rm em}\), with \(\ell_c\) the shell's crossing time and \(t_{\rm em}=E_R/L\), of order \(\chi^2\), and over any window with \(Q=\int L_c\,dt>0\), relaxed storage \(\mathcal S(R)\) with \(s=\sup|2G\mathcal S'|<1\) and transient \(|\delta_{\rm st}|\le\delta_0\), and \(\int|\mathcal D_c|\,dt\le cQ\), the bound is \(|\Delta E_R/Q+1|\le(s+c+2\delta_0/Q)/(1-s)\). The stored histories measure none of the four below the thermal rate, both calculations finding them storage-dominated, the shell drift per stored slice \(10^4\) to \(10^5\) times the thermal debit, so the closure is untestable on the stored runs (the condition-by-condition readout in Ref.~\cite{McKinneyCode2026}) \todo{B: two-field full-reference shell balance with reference response, exterior storage, trapping-radius-gradient factor and sea mismatch below the balance accuracy}. At the trapping radius the law reads \(4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R=-(2R\kappa_R/N_R)L\), an inward flux of negative null energy, equal to \(-L\) at a trapping radius clear of heavy matter with negligible defect \todo{the light field's null source at a dense trapping radius clear of heavy matter on a coupled two-field history, long after formation light, with \(\mathcal D_R\) and \(2R\kappa_R/N_R\) measured, against \(-L/(4\pi R^2N_R)\)}. The vacuum shell holds for the dust comparison solution and is unproved for the object's fountain.
```

### 10.326 B: replaced text, verbatim (lead inbox 94, closeF2 mass-loss closure text (FableAPI_closeF2_v1.md md5 71937f98, AstraAPI_closeA_v1.md eb335afb): the emisF6 block sentence replaced and the Appendix B paragraph added, spec I15_B pair 1), beginning "since \(q_r<\varepsilon\) wherever \(\varepsilon(p)>0\), the valleys b"
Replaced text:
```latex
since \(q_r<\varepsilon\) wherever \(\varepsilon(p)>0\), the valleys being a null set. This positive square is the rate supplied by the continuum evolution rows, the lattice transport weight and the changing-reference source being retained when it is identified with the actual mass rate. For \(m>0\) the Cauchy bound \(|q|/\varepsilon\le\nu_3=\sqrt{1+(ma)^2/3}-ma/\sqrt3<1\) gives \(\rho+p_r-2J\ge(1-\nu_3)(\rho-J)\), and at \(m=0\) only the strict inequality stands (Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
since \(q_r<\varepsilon\) wherever \(\varepsilon(p)>0\), the valleys being a null set. This positive square is the rate supplied by the continuum evolution rows, the lattice transport weight and the changing-reference source being retained when it is identified with the actual mass rate. For \(m>0\) the Cauchy bound \(|q|/\varepsilon\le\nu_3=\sqrt{1+(ma)^2/3}-ma/\sqrt3<1\) gives \(\rho+p_r-2J\ge(1-\nu_3)(\rho-J)\), and at \(m=0\) only the strict inequality stands (Ref.~\cite{McKinneyCode2026}).

% provenance: the mass balance between trapping radius and shell from FableAPI_closeF2_v1.md (md5 71937f98) with AstraAPI_closeA_v1.md (md5 eb335afb), lead inbox 94; plain-word map applied.
At fixed \(r\) the transverse evolution row, with the charge and momentum rows eliminating \(w'\) and \(N'\), gives \(\partial_t\mathcal M=-4\pi r^2\mathcal F+\mathcal D\), \(\mathcal F=N[w\rho_\ell+(1+w^2)J_{\ell r}+wp_r]=-T^r{}_t\), so that \(4\pi r^2\mathcal F\) is the flux through the sphere per unit \(t\) of the conserved current whose charge is \(\mathcal M\), the Killing-energy flux where \(N=1\), and \(\mathcal D=r^2vR_t/(GN)\). Where the rows carry residuals \(\epsilon_{\mathcal M}=\mathcal M'-4\pi r^2(\rho_\ell+wJ_{\ell r})\) and \(\epsilon_N=(\ln N)'+4\pi GrJ_{\ell r}/w\), the defect is \(\mathcal D=r^2vR_t/(GN)-2v\epsilon_{\mathcal M}+rv(1-w^2)\epsilon_N/G\). By \((N\mathcal D)'=4\pi r^2N(N\epsilon_E+v\epsilon_M)\), \(N\mathcal D\) at any radius is \(N_b\mathcal D_b\) at an outer boundary \(b\) minus the weighted defect between them, and beyond all sources \(\mathcal D_b=\dot E\), zero on the Hartree flow and the step's charge drift on stored histories. For the trapping radius \(R(t)\), a fixed shell \(r_c\), \(E_{\rm st}=\mathcal M(r_c)-E_R\) and \(E_{\rm out}=E-\mathcal M(r_c)\) this gives \(\dot E_R=-L_c-\dot E_{\rm st}+\mathcal D_c\) and \(\dot E_{\rm out}=\dot E+L_c-\mathcal D_c\) with \(L_c=4\pi r_c^2\mathcal F(r_c)\), and at the trapping radius \((2R\kappa_R/N_R)\dot E_R=4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R\), Eq.~\eqref{eq:C-rate} completed by the sweep of the moving trapping radius, \(2R\kappa_R/N_R=1-2G\mathcal M'_R\) being the trapping-radius-gradient factor.
```

### 10.327 The stored-slice verdict table of FableAPI_closeF2_v1.md section 5 (lead inbox 94), archived verbatim; printed in the paper as one sentence with the Ref. pointer (both calculations storage-dominated, closure untestable on the stored runs)
```markdown
| Condition or term | Readout | Records | Verdict |
|---|---|---|---|
| Identity at fixed radius | $\partial_t\mathcal M+4\pi r^2\mathcal F$ against $r^2vR_t/(GN)$ plus constraint residuals | 4–17 percent of the heavy flux at fixed outer-root radii; provenance unclosed | tested at the percent level of the heavy flux, $\gtrsim10^3L$; not certified |
| Anchoring, $\dot E$ | collar drift per record | $7\times10^{-3}E$ to $5\times10^{-4}E$ | drift is $10^4$–$10^5Q$ per record; $c\gg1$ |
| (V) heavy collar flux | heavy current through $16r_s$ over $L_c$ | charge beyond $16r_s<2.6\times10^{-3}$ through 40; outward current 48–56, 50.7/640 beyond at 56; ratio unread | clear in inventory through about 40; flux ratio not read; fails from 48 |
| $\alpha_R\to1$, root clear | heavy charge at the root; $\kappa_R$ | 29/640 outside at 15, 4–5 percent of $E$; $\kappa_R/[1/4GE]$ median 0.74–1.21 | fails on every rooted record |
| (S) relaxed storage | $E_{\rm st}$ against $R$; $|\Delta E_{\rm st}|/|\Delta E_R|$ | order one or larger; episodic roots; light inventory absent from $\mathcal M$ on light-off-background histories | fails; untestable for the light field without light gravity |
| time-scale separation | longest relaxation time | one $k=1$ pole, $1/\Gamma=2.8\times10^3r_s$ (lower bound on $t_{\rm rel}$); broad-block drain 0.147 on route two; first return near 1000 | $t_{\rm rel}$ unbounded above; not certified |
| (C) exterior defect | $\Delta\mathcal M(16r_s)+\int L_c-\int\mathcal D_c$ with light gravity on | size 32 only, $k=1,2$, one route; charge difference $3.4\times10^{-5}$, comparable to the light energy inside the cut | not certified; parameter estimate only |
| held fields | $\mathcal D_c=L_c$ identically | collar flow $0.9$–$1.5L_H$; $E_{\rm inv}$ to seven digits | certifies the inventory drain $\int L_c\le E_{\rm inv}$, nothing about the closure |
| root source $T_{ll}(R)$ | light's $\rho_\ell+p_r-2J_{\ell r}$ at the dense root against $-\alpha_RL/(4\pi R^2N_R)$ | two-field 32, single records; formation light; heavy matter at the root | readable, not certifying |
```

### 10.328 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 1), beginning "and the shell clock \(u=t-\Phi(r_c)\) of unit Jacobian only for the st"
Replaced text:
```latex
and the shell clock \(u=t-\Phi(r_c)\) of unit Jacobian only for the stationary exterior and clock convention specified, therefore holds to relative accuracy
```
Replacement:
```latex
and at a fixed shell \(u=t-\Phi(r_c)\) with unit Jacobian only for the stationary exterior and clock convention specified, therefore holds to relative accuracy
```

### 10.329 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 2), beginning "Identifying the channel-one statistic with the unsubtracted \(S_{\rm e"
Replaced text:
```latex
Identifying the channel-one statistic with the unsubtracted \(S_{\rm esc}\) requires its packet basis and comparator to be documented, and its late-slope normalization against the comparison-case rate, 0.28--0.37 here against the separate normalization of Sec.~\ref{sec:page}'s diagnostic, is unverified until the stored history's shell charge at 100 and the comparison rate, clock and units used are printed beside it \todo{C: accounting conventions and normalization}.
```
Replacement:
```latex
Identifying the channel-one statistic with the unsubtracted \(S_{\rm esc}\) requires its packet basis and comparator to be documented. The reported late slope over 80--100, 0.030--0.040 nats per nominal \(r_s\), is 0.28--0.37 of the nominal comparison-case rate \(0.1079/r_s\). The separate 0.23--0.31 normalization of that diagnostic (Ref.~\cite{McKinneyCode2026}) remains unverified until the stored history's shell charge at 100 and the comparison rate, clock and units actually used are printed beside it \todo{C: accounting conventions and normalization}.
```

### 10.330 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 3), beginning "The arrival clock from the stationary-exterior flight, 40.8--44.1 for "
Replaced text:
```latex
The arrival clock from the stationary-exterior flight, 40.8--44.1 for \(\omega r_s=2\)--0.4 up to an undetermined \(O(r_s)\) matching term,
```
Replacement:
```latex
The arrival clock from the stationary-exterior flight, \(T/r_s=40.8\)--44.1 for \(\omega r_s=2\)--0.4 up to an undetermined \(O(r_s)\) matching term,
```

### 10.331 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 4), beginning "With \(n_k^H=\Gamma_k/(e^{\omega/T_H}+1)\) and \(\mathcal R_\nu=T_H\su"
Replaced text:
```latex
With \(n_k^H=\Gamma_k/(e^{\omega/T_H}+1)\) and \(\mathcal R_\nu=T_H\sum_k64k\int\frac{d\omega}{2\pi}s(\nu n_k^H)\big/(\nu\mathcal L/R^2)\), the emitted energy and the marginal budget are supply integrals,
```
Replacement:
```latex
With \(n_k^H=\Gamma_k/(e^{\omega/T_H}+1)\), \(\mathcal R_\nu=T_H\sum_k64k\int\frac{d\omega}{2\pi}s(\nu n_k^H)\big/(\nu\mathcal L/R^2)\), and the budget integrand continued by its limit, zero, at zero supply, the emitted energy and the marginal budget are supply integrals,
```

### 10.332 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 5), beginning "the Fermi-shaped local two-channel conversion at \(\kappa_R/(2\pi)\) r"
Replaced text:
```latex
the Fermi-shaped local two-channel conversion at \(\kappa_R/(2\pi)\) reported in Ref.~\cite{McKinneyCode2026} on fixed and on flow-carried sites.
```
Replacement:
```latex
the Fermi-shaped local two-channel conversion at \(\kappa_R/(2\pi)\) reported in Ref.~\cite{McKinneyCode2026} on fixed and on flow-carried sites (in the interior Wentzel-Kramers-Brillouin overlap region the two outward-moving branches and their positions, valid as \(\omega\to0\) and not their exact positions at positive \(\omega\), in Ref.~\cite{McKinneyCode2026}).
```

### 10.333 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 6), beginning "and the stored histories through \(T/r_s=100\) fix neither the late su"
Replaced text:
```latex
and the stored histories through \(T/r_s=100\) fix neither the late supply nor the turnover
```
Replacement:
```latex
and the stored histories through \(T/r_s=100\), \(2.5\times10^{-4}\) of the saturated lifetime at this size, fix neither the late supply nor the turnover
```

### 10.334 Sec. VI: replaced text, verbatim (lead inbox 97, the six emisA7 qualifications restored from emisF4 in the emisF6 block (AstraAPI_emisA7_v1.md md5 f2dac01c); replaced text verbatim, replacement shown, spec I17_06 pair 7), beginning "What the delivery moved to Ref. McKinneyCode2026 is listed in the deli"
Replaced text:
```latex
What the delivery moved to Ref. McKinneyCode2026 is listed in the delivery file under "Moved to Ref.". Typographic:
```
Replacement:
```latex
What the delivery moved to Ref. McKinneyCode2026 is listed in the delivery file under "Moved to Ref.". Six emisA7 qualifications restored from emisF4 (AstraAPI_emisA7_v1.md, md5 f2dac01c, lead inbox 97): "at a fixed shell" for the clock u = t - Phi(r_c); the late slope "over 80--100" with F4's split, 0.28--0.37 of the nominal comparison-case rate and the separate 0.23--0.31 normalization remaining unverified, the latter pointed to Ref. McKinneyCode2026 because Sec. V C no longer prints it; T/r_s on the arrival estimate 40.8--44.1; the budget integrand continued by its limit, zero, at zero supply, in the printed comparator; the interior Wentzel-Kramers-Brillouin overlap region and the two outward-moving branches named in the printed repository pointer; and "2.5e-4 of the saturated lifetime at this size" on the stored histories through T/r_s = 100. Typographic:
```

### 10.335 Sec. III C: trapping-cut comparison bounds, the singular-value form of the cut entropy moved to the repository (owner readability residual (1), lead message of 3 Nov, "readability beats exhaustive detail"; kept in the paper: Mirsky's inequality, the rank cost, the cross-block bound, and the direct-state test, every bound as stated)

Old (HEAD before this commit), verbatim with its provenance comment:

```latex
% provenance: K<=7.07 / 7.49: <internal>/TRAPPING_CUT_EF_R2_v1.md (route 2, l.12 and l.125); spec WEBGPT40O_REDUCTION_v1 P02; route 1 not pinned.
Whether a collapse's trapping cut keeps this coefficient is the next question. \fnd{The trapping-cut entropy is compared with the flat sea's through bounds that need no gap or energy premise, through a comparator projector or directly: for a pure number-conserving Gaussian covariance the entropy is \(\sum g(\sigma)\) over the cross block's singular values, \(g(\sigma)=s(\tfrac12[1-\sqrt{1-4\sigma^2}])\), \(0\le g'\le2\). Mirsky's inequality~\cite{Mirsky1960} gives \(|S_A(C)-S_A(P)|\le\nu=\|C-P\|_1\) on the same lattice with convergent traces, and a restricted difference of rank \(r_A\) costs at most \(r_A\log2\). These are sufficient, not necessary, controls: at fixed \(ma>0\), for a ball of radius \(La\), a cross-block bound \(\|\Pi_A(C-P_0)\Pi_{A^c}\|_2^2=O(L)\) gives entropy difference \(O(L^{3/2}\sqrt{\log L})\), and the sufficient direct-state test \(E_f(C)=\|\Omega_0^{1/2}(fC-P_0f)\|_2^2\le KL\), \(\Omega_0=|h_0|\), \(f\) a smooth window on the cut region scaled to \(L\), gives that cross-block condition through the massive gap (details in Ref.~\cite{McKinneyCode2026}).}
```

New:

```latex
% provenance: K<=7.07 / 7.49: <internal>/TRAPPING_CUT_EF_R2_v1.md (route 2, l.12 and l.125); spec WEBGPT40O_REDUCTION_v1 P02; route 1 not pinned.
Whether a collapse's trapping cut keeps this coefficient is the next question. \fnd{The trapping-cut entropy is compared with the flat sea's through bounds that need no gap or energy premise, through a comparator projector or directly: Mirsky's inequality~\cite{Mirsky1960} gives \(|S_A(C)-S_A(P)|\le\nu=\|C-P\|_1\) on the same lattice with convergent traces, and a restricted difference of rank \(r_A\) costs at most \(r_A\log2\). These are sufficient, not necessary, controls: at fixed \(ma>0\), for a ball of radius \(La\), a cross-block bound \(\|\Pi_A(C-P_0)\Pi_{A^c}\|_2^2=O(L)\) gives entropy difference \(O(L^{3/2}\sqrt{\log L})\), and the sufficient direct-state test \(E_f(C)=\|\Omega_0^{1/2}(fC-P_0f)\|_2^2\le KL\), \(\Omega_0=|h_0|\), \(f\) a smooth window on the cut region scaled to \(L\), gives that cross-block condition through the massive gap (the singular-value form of the cut entropy behind these bounds, and the details, in Ref.~\cite{McKinneyCode2026}).}
```

Moved to the repository: for a pure number-conserving Gaussian covariance the entropy is the sum of g(sigma) over the cross block's singular values, g(sigma) = s((1/2)[1 - sqrt(1 - 4 sigma^2)]), 0 <= g' <= 2 (Ref. McKinneyCode2026).

### 10.336 Sec. III C: three-term split of the direct-state budget, derivation mechanics moved to the repository (owner readability residual (1); kept in the paper: the three-term split as stated, the window term 8 pi gamma_1 L + O(1) with gamma_1 = 0.392 at ma = 0.2 and 9.9L, the orbital bound epsilon_max N_B and 18.9L, the propagated-sea statement with its conditions, K <= 7.07 over 44 histories and K <= 7.49 over 20 stored slices, and the todo)

Old (HEAD before this commit), verbatim with its provenance comment:

```latex
% provenance: Sec. III C three-term split of the direct-state budget, the Sec. IV E pointer, the Sec. V E sentence, and the Discussion sentences from <internal>/FableAPI_evcutF2_v1.md (md5 7bbe777e, section (b)) with every correction of AstraAPI_evcutA2_v1.md (md5 08c917ed) accepted (K_0 size-dependent, gamma_1 = 0.392 at ma = 0.2, (H1)-(H3) with the field-rate bounds); lead inbox 78. The delivery's anchor "The trapping-cut budget passes" no longer existed verbatim, the equivalent span being the Sec. III C pointer sentence "Its pass on the computed collapse histories is reported in Sec. IV E" and the Sec. IV E direct-state paragraph, both replaced; plain-word map applied (records -> stored slices, routes -> independent calculations, dense root -> fine-grid trapping radius, collar -> shell, certifies -> establishes), the one prose semicolon split.
The test splits into three terms, the flat sea's window energy, the dust's, and the propagated sea's, and only the last lacks a bound holding for every state and size. For pure \(C\), in the same lattice units, the test is exactly \(E_f(P_0)+\Tr[(C-P_0)fh_0f]\), with \(E_f(P_0)=\gamma_1\sum_{x,i}|f_{x+e_i}-f_x|^2\) and \(\gamma_1=\int_p\sin^2p_1/\varepsilon\), 0.392 at \(ma=0.2\), the flat sea's own window energy, and the massive gap gives \(\|\Pi_A(C-P_0)\|_2^2\le[\sqrt{E_f(C)}+\sqrt{E_f(P_0)}]^2/m\), the cross-block condition above. For the harmonic window, equal to one on the ball of radius \(L\) and \(2L/|x|-1\) out to \(2L\), the gradient sum is \(8\pi L+O(1)\), so the window term is \(8\pi\gamma_1L+O(1)\), about \(9.9L\) at large \(L\), with its finite-size value computed exactly at each size, and the \(N_B\) dust orbitals contribute at most \(\varepsilon_{\max}N_B=\varepsilon_{\max}L/(2g\,ma\,\eta_E)\), \(18.9L\) at the reference parameters (Sec.~\ref{sec:star}), for every state and every size of the family. The budget therefore rests on the propagated sea's window energy \(\Tr[(\mathsf Q-P_0)fh_0f]\). This term is \(O(L)\) on intervals \(t\le T_*L\) along which the fields obey Appendix~\ref{app:local}'s bounds on the scale \(L\) together with matching bounds on their time derivatives, by the prepared band's adiabatic expansion, while the computed histories leave the spatial bounds before formation (Sec.~\ref{sec:results-lattice}) and the held fields' step jumps lie outside the rate bounds, so the expansion establishes none of them. On the angular-channel chains the full budget reads \(K\le7.07\) over 44 histories and \(K\le7.49\) over 20 stored slices at \(r_s/a=32\), 64, 256, with no growth in \(1/\chi\) and matched-clock values agreeing within 2 percent, and the sea's term has no cubic reading. \todo{the propagated sea's window energy and cross block on the cubic digital ball at the fine-grid trapping radius, two blind independent calculations at \(r_s/a=32\) and 64 with the quiet exterior start, box and stored-slice controls of the predeclared gate, paired with the chain readings}
```

New:

```latex
% provenance: Sec. III C three-term split of the direct-state budget, the Sec. IV E pointer, the Sec. V E sentence, and the Discussion sentences from <internal>/FableAPI_evcutF2_v1.md (md5 7bbe777e, section (b)) with every correction of AstraAPI_evcutA2_v1.md (md5 08c917ed) accepted (K_0 size-dependent, gamma_1 = 0.392 at ma = 0.2, (H1)-(H3) with the field-rate bounds); lead inbox 78. The delivery's anchor "The trapping-cut budget passes" no longer existed verbatim, the equivalent span being the Sec. III C pointer sentence "Its pass on the computed collapse histories is reported in Sec. IV E" and the Sec. IV E direct-state paragraph, both replaced; plain-word map applied (records -> stored slices, routes -> independent calculations, dense root -> fine-grid trapping radius, collar -> shell, certifies -> establishes), the one prose semicolon split.
The test splits into three terms, the flat sea's window energy, the dust's, and the propagated sea's, and only the last lacks a bound holding for every state and size. For pure \(C\), in the same lattice units, the test is exactly \(E_f(P_0)+\Tr[(C-P_0)fh_0f]\), the flat sea's own window energy plus the state's, and the massive gap turns it into the cross-block condition above (the gap inequality in Ref.~\cite{McKinneyCode2026}). The flat sea's window energy is explicit, \(8\pi\gamma_1L+O(1)\) for the harmonic window with \(\gamma_1=0.392\) at \(ma=0.2\), about \(9.9L\) at large \(L\) and computed exactly at each size, and the \(N_B\) dust orbitals contribute at most \(\varepsilon_{\max}N_B\), \(18.9L\) at the reference parameters (Sec.~\ref{sec:star}), for every state and every size of the family (the window, the gradient sum, and the orbital count in Ref.~\cite{McKinneyCode2026}). The budget therefore rests on the propagated sea's window energy \(\Tr[(\mathsf Q-P_0)fh_0f]\). This term is \(O(L)\) on intervals \(t\le T_*L\) along which the fields obey Appendix~\ref{app:local}'s bounds on the scale \(L\) together with matching bounds on their time derivatives, by the prepared band's adiabatic expansion, while the computed histories leave the spatial bounds before formation (Sec.~\ref{sec:results-lattice}) and the held fields' step jumps lie outside the rate bounds, so the expansion establishes none of them. On the angular-channel chains the full budget reads \(K\le7.07\) over 44 histories and \(K\le7.49\) over 20 stored slices at \(r_s/a=32\), 64, 256, with no growth in \(1/\chi\) and matched-clock values agreeing within 2 percent, and the sea's term has no cubic reading. \todo{the propagated sea's window energy and cross block on the cubic digital ball at the fine-grid trapping radius, two blind independent calculations at \(r_s/a=32\) and 64 with the quiet exterior start, box and stored-slice controls of the predeclared gate, paired with the chain readings}
```

Moved to the repository: E_f(P_0) = gamma_1 sum_{x,i} |f_{x+e_i} - f_x|^2 with gamma_1 = int_p sin^2 p_1 / epsilon; the gap inequality ||Pi_A (C - P_0)||_2^2 <= [sqrt(E_f(C)) + sqrt(E_f(P_0))]^2 / m; the harmonic window (one on the ball of radius L, 2L/|x| - 1 out to 2L) and its gradient sum 8 pi L + O(1); the orbital count epsilon_max N_B = epsilon_max L / (2 g ma eta_E) (Ref. McKinneyCode2026).

### 10.337 Sec. III C: orientation sentence and subsection closer (owner readability residual (1), the thesis-to-conclusion line; additions, nothing moved)

Old closer, verbatim:

```latex
% provenance: 443.6 = 640 ln2; 0.0101 = 443.6/(4 pi 59.07^2); 59.07a = two_gE0 of P64_s4 in ladders_readout_v1_3.json; same row <internal>/TRAPPING_CUT_EF_R2_v1.md:200.
\fnd{The reference-split comparison bounds \(|S_A(C)/A_{\rm lat}-\eta(ma)|\) by the flat finite-cut remainder \(e_{\rm flat}\), the comparator's own shift \(r_{\rm ref}=|S_A(P)-S_A(P_0)|/A_{\rm lat}\) for a comparator projector \(P\), and \(\min\{\nu,r_A\log2\}/A_{\rm lat}\), with \(A_{\rm lat}=4\pi(R_t/a)^2\) on the fine-grid trapping radius \(R_t\) (Sec.~\ref{sec:star}) and the same site set in every comparison, and the dust's \(N_B\) occupied orbitals cost at most \(N_B\log2\) (values in Ref.~\cite{McKinneyCode2026}).} With the coefficient and its comparison bounds fixed, what remains before the exterior is how much matter a collapse can compress. 
```

New closer:

```latex
% provenance: 443.6 = 640 ln2; 0.0101 = 443.6/(4 pi 59.07^2); 59.07a = two_gE0 of P64_s4 in ladders_readout_v1_3.json; same row <internal>/TRAPPING_CUT_EF_R2_v1.md:200.
\fnd{The reference-split comparison bounds \(|S_A(C)/A_{\rm lat}-\eta(ma)|\) by the flat finite-cut remainder \(e_{\rm flat}\), the comparator's own shift \(r_{\rm ref}=|S_A(P)-S_A(P_0)|/A_{\rm lat}\) for a comparator projector \(P\), and \(\min\{\nu,r_A\log2\}/A_{\rm lat}\), with \(A_{\rm lat}=4\pi(R_t/a)^2\) on the fine-grid trapping radius \(R_t\) (Sec.~\ref{sec:star}) and the same site set in every comparison, and the dust's \(N_B\) occupied orbitals cost at most \(N_B\log2\) (values in Ref.~\cite{McKinneyCode2026}).} So the flat coefficient is fixed within rigorous bounds, and a collapse's cut keeps it where the direct-state budget passes uniformly, which the computed chains show at every tested size while the cubic cut is unread. What remains before the exterior is how much matter a collapse can compress. 
```

### 10.338 Sec. V: replaced text, verbatim (borrowF B3b, purity and pairs (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label; replaced span and replacement, spec BF_B3b_07 pair 1), beginning "and the tail enters no result beyond this subsection. Within that cont"
Replaced text:
```latex
and the tail enters no result beyond this subsection. Within that control, the cut entropy is read first.
```
Replacement:
```latex
and the tail enters no result beyond this subsection. Unitary quadratic evolution preserves the global purity of the initially pure light covariance, while the drive may create excitations and cut entanglement, and the numerical purity defect remains to be measured \todo{B3: \(\Tr s(C_L)\) per channel on the stored clocks, two independent calculations}. Against a declared equal-rank comparator projector \(P\), the difference \(C_L-P\) has paired positive and negative eigenvalues, its trace norm is a state-comparator distance, and the particle and hole excitation numbers are \(N_p=N_h=\Tr[(I-P)C_L]=\tfrac12\Tr(C_L-P)^2\), whose interpretation depends on the comparator, a changing band distance alone not separating production from reference motion or transport \todo{B3: the comparator-departure trace norm and \(N_p\), \(N_h\) per channel on the stored slices against a declared equal-rank projector, with its continuation, rank, and idempotency validated}.
% provenance: B3b, purity and pairs, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: the purity sentence replaced by reviewer B's (purity preserved by the unitary evolution, the numerical defect to be measured), the pair sentence replaced by reviewer B's for a declared equal-rank projector (trace norm a distance, N_p = N_h = Tr[(I-P)C_L], interpretation comparator dependent), the eigenvalue-sum equality not presented as a purity test, the prepared-band projector P_band(t) not assumed; inserted inside the paragraph before its closing sentence; lead inbox 98, 104, 105.
% status (lead inbox 105): APPLIED WITH LABEL (purity derived, its defect and the pair counts pending, two B3 marks). borrowA ids applied: A-B3b.1 purity derived, pair count not; A-B3b.2 purity sentence replacement; A-B3b.3 P_band(t) not assumed, a declared projector instead; A-B3b.4 pair sentence replacement with N_p = N_h; A-B3b.5 eigenvalue sums not a purity test; A-B3b.6 validation items in the mark.
Within that control, the cut entropy is read first.
```

### 10.339 Sec. VI: replaced text, verbatim (borrowF B3a, the register identity (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label; replaced span and replacement, spec BF_B3a_06 pair 1), beginning "so no entropy-reducing arrival is established on the stored slices."
Replaced text:
```latex
so no entropy-reducing arrival is established on the stored slices.
```
Replacement:
```latex
so no entropy-reducing arrival is established on the stored slices. For the complete pure partition \(B\), \(\mathcal E_u\), \(\mathcal S_u\) of the light covariance, \(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\) exactly, so the sign of the accumulated output's increment is the sign of the block's cross-cut correlation imbalance between what remains and what has left, and a change from positive to negative marks a turnover of \(S_{\rm esc}\) \todo{B3: \(I(B{:}\mathcal E_u)\), \(I(B{:}\mathcal S_u)\), and \(S(B|\mathcal E_u)\) per received block on the C1/C2 re-propagated pure covariances with the documented no-return packet basis and complete complement, the identity residual as a consistency check alongside an independent global purity measurement}. What the propagator itself fixes about the emission, before any such register is read, comes next.
% provenance: B3a, the register identity, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: reviewer A's partner and crossing wording replaced by reviewer B's (the identity on the complete pure partition, the sign of the increment, a positive-to-negative change as the turnover), "the identity's residual as the purity check" replaced by "a consistency check alongside an independent global purity measurement", no positivity claim for the new register; the prose uses cross-cut correlations, not the information-measure wording, per the block rule; appended to the first paragraph after its last sentence with a closer that motivates the next paragraph; lead inbox 98, 104, 105.
% status (lead inbox 105): APPLIED WITH LABEL (exact identity, the register pending, mark B3). borrowA ids applied: A-B3a.1 partner and purity claims withdrawn; A-B3a.2 replacement text; A-B3a.3 residual a consistency check beside an independent purity measurement; A-B3a.4 mixed-spectator caveat respected (no purity inference); A-B3a.5 no positivity claim for the new register.

```

### 10.340 Sec. VI: replaced text, verbatim (borrowF B1b, the late branch as an entropy bound (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label; replaced span and replacement, spec BF_B1b_06 pair 1), beginning "For the Page curve we take an idealized depletion comparator with thes"
Replaced text:
```latex
For the Page curve we take an idealized depletion comparator with these assumptions:
```
Replacement:
```latex
% provenance: B1b, the measured late branch as an entropy bound, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: the insertion replaced by reviewer B's entropy-bound statement (complete orthogonal partition, an upper bound and not a measured capacity, the light coefficient eta(0) and the (R_t/R_0)^2 factor, the store still assumed), reviewer A's store-clause rewrite and its use of the 109 to 311 and about nine nats readings dropped (different measurements), the cubic cut marked as a one-way cubic control; lead inbox 98 and 104.
% status (lead inbox 105): APPLIED WITH LABEL (an entropy bound, the store still assumed, the cubic cut a pending one-way control, mark B1). borrowA ids applied: A-B1b.1 bound not a measured capacity; A-B1b.2 light coefficient eta(0) and the (R_t/R_0)^2 factor; A-B1b.3 replacement text; A-B1b.4 the 109 to 311 and about nine nats readings not used; A-B1b.5 one-way cubic control label. The standing qualification that no stored history runs under Eq. dm-reference is untouched.
For the Page curve we take an idealized depletion comparator with these assumptions:
```

### 10.341 Sec. VI: replaced text, verbatim (borrowF B1b, the late branch as an entropy bound (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label; replaced span and replacement, spec BF_B1b_06 pair 2), beginning "supplies no area-sized accessible-sector bound."
Replaced text:
```latex
supplies no area-sized accessible-sector bound.
```
Replacement:
```latex
supplies no area-sized accessible-sector bound. If the received light modes lie wholly outside the trapping cut and the remaining modes form a complete orthogonal complement, purity and subadditivity give \(S_{\rm esc}(u)\le S_{{\rm cut},L}(R_t(u))+S(\mathcal X_u)\), \(S_{{\rm cut},L}\) the light field's entropy across the trapping cut and \(\mathcal X_u\) the exterior modes not yet received, an entropy bound, not an accessible-sector count or a saturation law. If the evolved light cut retains its flat coefficient, \(S_{{\rm cut},L}/S_0=[4\eta(0)G/a^2](R_t/R_0)^2+o(1)\), and the accessible store of logarithmic dimension at most \(fA_R/(4G)\) remains assumed \todo{B1: \(S_{\rm cut}\) of the evolved cubic covariance at \(R_t(t)\) on the stored slices, both fields, two independent calculations, with the flat ball on the same site set, as a one-way cubic control}.
```

### 10.342 Sec. VI: replaced text, verbatim (borrowF B1c, the achievability channel (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label; replaced span and replacement, spec BF_B1c_06 pair 1), beginning "turns the bound in Eq.~\eqref{eq:page-effective-curve} into equality."
Replaced text:
```latex
turns the bound in Eq.~\eqref{eq:page-effective-curve} into equality.
```
Replacement:
```latex
turns the bound in Eq.~\eqref{eq:page-effective-curve} into equality. An enqueue and swap circuit of Gaussian Schmidt pairs across the trapping cut, with stored sign-reversed-state partners released into outgoing packets, is an externally controlled Gaussian proposal, not a realization of the model's autonomous boundary generator. Attainability with the prescribed outgoing marginals, the measured sign-reversed-state energy, charge-matched encoding, and complete release remains unproved, and an unresolved block costs at most its entropy, which does not generally vanish when its frequency width is narrowed \todo{B1c: the permitted autonomous generator and its funding for a capacity-saturating history, with energy and number conservation, outgoing occupations, the sign-reversed-state inventory against the Gibbs ceiling, and \(S_{\rm out}\) checked along the curve}.
% provenance: B1c, the achievability channel, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: reviewer A's "exists inside the model's own class" claim replaced by reviewer B's statement (an externally controlled Gaussian proposal, attainability unproved, the unresolved-block entropy not vanishing with narrow packets; the enqueue rotation costs 2 omega sin^2 theta and the swap gives occupation 1 - n), the verification items as the mark; lead inbox 98 and 104.
% status (lead inbox 105): APPLIED WITH LABEL (an externally controlled proposal, attainability unproved, mark B1c). borrowA ids applied: A-B1c.1 conflict with the autonomous generator stated; A-B1c.2 rotation energy cost and swap occupation (not printed, the claim withdrawn); A-B1c.3 replacement text; A-B1c.4 verification items as the mark.

```

### 10.343 Sec. III: replaced text, verbatim (borrowF B8, the discrete exterior of a flux discretization and its remainder (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label, the numerical check held; replaced span and replacement, spec BF_B8_04 pair 1), beginning "The geometric trapping radius stays at \(r_s\). % assumption: the outw"
Replaced text:
```latex
The geometric trapping radius stays at \(r_s\).

% assumption: the outward lattice branch and its delay are computed on a static constant-charge exterior
```
Replacement:
```latex
The geometric trapping radius stays at \(r_s\).

% provenance: B8, the exact discrete exterior and its lattice remainder, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: the opening replaced by reviewer B's illustrative-discretization statement, the discretization not identified with the production solver and the trigamma profile a different cumulative scheme, the effective sources a stationary continuum interpretation of the scheme's error and not lattice corrections of the model, the "largest lattice correction above the probe corrections" claim and the solar-mass arithmetic not printed, beta_PN conditional on the scheme (reviewer B's corrected intermediate expression in the derivation notes, not printed), the back-averaged prediction corrected to c-bar_1 = c_1 and c-bar_2 = c_2 - (9/4)(sigma/a)^2; reviewer A's Sec. II D clause edit not applied (caveat kept); inserted as its own paragraph with a thesis and a closer, the paragraph pass's sentences kept; lead inbox 98, 104, 105.
% status (lead inbox 105): APPLIED WITH LABEL for the analytic part (illustrative scheme, derived, not the solver's); the numerical check on the stored slices HELD (solve lane), so no fit sentence and no B8 mark are printed. borrowA ids applied: A-B8.1 not established for the implemented solver; A-B8.2 opening replaced; A-B8.3 effective sources an interpretation of the scheme's error, beta_PN conditional, no "largest correction" claim, solar number dropped; A-B8.4 back-averaging c-bar_1 = c_1, c-bar_2 = c_2 - (9/4)(sigma/a)^2; A-B8.5 vacuum fixture and unrestricted fit recorded in BF_B9_B10.md for the lane.
A discretized energy law has an exterior of its own, exact on the chain, whose lattice remainder is a property of the scheme and not of the model. An illustrative flux discretization, the charge inside the sites \(r_m\le r_n\) read as the flux through the bond leaving them, is \((r_n+\theta a)(v_n^2+v_{n+1}^2)/(4G)=E\), the radius read at \(r_n+\theta a\), so that in vacuum every exterior bond carries \(E\) and the exterior is algebraic. Its smooth vacuum solution is
\begin{equation}
\begin{gathered}
v_n^2=\frac{2GE}{r_n}\Bigl[1+c_1\frac{a}{r_n}+c_2\frac{a^2}{r_n^2}+O\Bigl(\frac{a^3}{r_n^3}\Bigr)\Bigr],\\
c_1=\tfrac12-\theta,\qquad c_2=-\theta(1-\theta),
\end{gathered}
\label{eq:dr-discrete-exterior}
\end{equation}
a radius read at the bond midpoint having no first-order remainder and one read at a node \(c_1=\pm\tfrac12\), while a node-weighted cumulative scheme gives the exact trigamma profile \(v_n^2=2GE\,\psi_1(n)/a\), a different scheme whose expansion is the \(\theta=0\) case through first order. This discretization has not been identified with the production solver. Read through the rows at \(N=1\) as a mass function, \(2G\mathcal M_{\rm eff}=rv^2\), the first-order term is a stationary continuum interpretation of the scheme's error, an effective exterior source \(\rho_{\rm lat}=-c_1r_sa/(8\pi Gr^4)\) with \(p_r=-\rho_{\rm lat}\) and \(p_\perp=\rho_{\rm lat}\), traceless and falling as \(r^{-4}\) like a radial Maxwell field, with \(\beta_{\rm PN}=1-2c_1a/r_s\) conditional on the scheme, orders and signs being scheme dependent and not lattice corrections of the model. Under the back-averaging of Appendix~\ref{app:charge} the coefficients become \(\bar c_1=c_1\) and \(\bar c_2=c_2-\tfrac94(\sigma/a)^2\), averaging not erasing a nonzero \(c_1\). The implemented chain's coefficients are not read here, and what crosses this exterior's surface outward comes next.

% assumption: the outward lattice branch and its delay are computed on a static constant-charge exterior
```

### 10.344 Sec. V: replaced text, verbatim (borrowF B7 with B1a (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label, B2 held; replaced spans and replacements, spec BF_B7_07 pair 1), beginning "The Hawking comparison case, a thermally supplied exterior, feeds each"
Replaced text:
```latex
The Hawking comparison case, a thermally supplied exterior, feeds each
Dirac channel \(k\) of the
constant-charge exterior, Eq.~\eqref{eq:dr-exterior}, thermally at
\(T_H\) from its inner port, the outer port empty, with
unit-current transmission \(\Gamma_k(\omega)\). Summed over the
eight valleys, particles and antiparticles, \(|k|\le8\),
```
Replacement:
```latex
% provenance: B7 with B1a and B2, the model's own barrier, the two currents, and the microscopic check, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: the barrier lead-in replaced by reviewer B's (amplitude rescaling alpha_s^{1/2} and the stationary phase e^{i omega F}, the transmission a continuum comparison transmission with finite-spacing and back-averaging corrections separate), the thermal supply stated as assumed, the sharp-count sentence replaced by reviewer B's binary-transmission comparison limits, the six-digit convergence, 1/r_max law, Page 1976 and 1983 benchmarks, and flat-chain quench percentages replaced by reviewer B's finite-band statement with the checks left unverified as marks, "The number rate is reported but unverified" kept, the B2 quench HELD (hreduce lane, lead inbox 105) and not printed, its runnable spec corrected per reviewer B in BF_B9_B10.md, the "Purity with a retained capacity" sentence folded to one clause that keeps its citations (B1a); lead inbox 98 and 104.
% status (lead inbox 105): APPLIED WITH LABEL. borrowA ids applied: A-B7.1 barrier lead-in with rescaling and phase; A-B7.2 thermal supply labelled assumed; A-B7.3 binary-transmission limits instead of the sharp-count sentence; A-B7.4 finite-band provenance statement, convergence and Page-benchmark checks labelled unverified (mark B7); A-B7.5 number rate kept unverified, new benchmark and quench numbers not printed; A-B7.6 and A-B7.7 B2 quench held (spec invalid as written, corrected spec recorded); A-B7.8 telescoping identity not a second calculation. The standing qualification that no stored history runs under Eq. dm-reference is untouched.
The Hawking comparison case is a thermally supplied exterior whose transmission is the model's own. On the exterior of Eq.~\eqref{eq:dr-exterior} the operator of channel \(k\) is \(\bar h_k=h_{0,k}+\tfrac12\{v,Q_r\}\) at \(m=0\), \(N=1\), \(v=-\sqrt{r_s/r}\). For the leading continuum limit on the unaveraged vacuum profile, write the stationary chain spinor as \(e^{-i\omega t}\psi_\omega(r)\), set \(t_s=t+F(r)\) with \(F'=v/\alpha_s^2\), \(\alpha_s=\sqrt{1-r_s/r}\) the Killing lapse, and define \(\chi=\alpha_s^{1/2}e^{\eta\sigma_1/2}e^{i\omega F}\psi_\omega\) with \(\tanh\eta=v\). Then
\begin{equation}
\omega\chi=\bigl[-i\sigma_1\partial_{r_*}+W\sigma_2\bigr]\chi,\qquad
W=\frac{\alpha_sk}{r},\qquad dr_*=\frac{dr}{\alpha_s^2},
\label{eq:page-dirac-barrier}
\end{equation}
and since \(\sigma_2\) anticommutes with \(\sigma_1\) the two components see the partner barriers \(W^2\pm\partial_{r_*}W\), rising to the centrifugal \(k(k\pm1)/r^2\) far out and vanishing at \(r_s\) with \(\alpha_s\). The unit-flux transmission \(\Gamma_k(\omega)\) of this barrier, a ratio of asymptotic fluxes and so the same in either time, the same for the two partners and for both signs of \(\kappa=\pm k\), is the continuum comparison transmission, finite-spacing and back-averaging corrections requiring separate calculation. Feeding each channel thermally at \(T_H\) from the inner port with the outer port empty, an assumed supply not established on the collapse, and counting the eight valleys, both signs of \(\kappa\), particles and holes, \(64k\) modes per unit of \(\omega/2\pi\),
\begin{equation}
\begin{gathered}
L=\sum_k64k\!\int_0^\infty\!\frac{d\omega}{2\pi}\,\omega\,n_k,\\
\dot S_{\rm marg}=\sum_k64k\!\int_0^\infty\!\frac{d\omega}{2\pi}\,s(n_k),\qquad
n_k=\frac{\Gamma_k(\omega)}{e^{\omega/T_H}+1},
\end{gathered}
\label{eq:page-currents}
\end{equation}
the entropy taken at the transmitted occupation because tracing out the reflected part leaves the transmitted mode Gaussian at \(\Gamma_kf\). Summed over the eight valleys, particles and antiparticles, \(|k|\le8\),
```

### 10.345 Sec. V: replaced text, verbatim (borrowF B7 with B1a (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label, B2 held; replaced spans and replacements, spec BF_B7_07 pair 2), beginning "with \(\mathcal L=0.005237\). Two independent numerical calculations r"
Replaced text:
```latex
with \(\mathcal L=0.005237\). Two independent numerical calculations reproduce the
luminosity and entropy-rate coefficients. The number rate is
reported but unverified.
```
Replacement:
```latex
with \(\mathcal L=0.005237\). Two independent calculations reproduce the finite-band estimates \(\mathcal L\simeq0.0052373\) and \(\mathcal R\simeq1.6391\), without a narrow absolute bound, and the channel-sum convergence, Page-benchmark, and quench checks remain unverified \todo{B7: the Page benchmark and partner check on both of this paper's integrators}. The number rate is reported but unverified. The ratio \(\mathcal R=T_H\dot S_{\rm marg}/L\) is fixed by the one transmission and the one occupation and exceeds one mode by mode, since \(s(x)-(\omega/T_H)x\) vanishes at \(x=0\) and increases up to \(x=f\). For binary transmissions, \(\Gamma_k\in\{0,1\}\), the currents depend on the open-channel count \(C(\omega)\), a constant count giving \(\mathcal R=2\) and \(C\propto\omega^2\) giving \(4/3\), analytic comparison limits, not bounds on a soft barrier, for which \(\sum_ks(\Gamma_kf)\) is not determined by \(\sum_k\Gamma_k\).
```

### 10.346 Sec. V: replaced text, verbatim (borrowF B7 with B1a (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied with label, B2 held; replaced spans and replacements, spec BF_B7_07 pair 3), beginning "Purity with a retained capacity at most \(S(E)\) bounds the escaped en"
Replaced text:
```latex
Purity with a retained capacity at most \(S(E)\) bounds the escaped entropy by \(\min(S_{\rm marg},S(E))\), whose maximum under typicality is the Page maximum of Sec.~\ref{sec:page-effective}~\cite{Page1993} (a typical fermionic Gaussian state on the same counts gives a lower, later maximum~\cite{BianchiHacklKieburg2021}, values in Ref.~\cite{McKinneyCode2026}): curves of the comparison case under assumed ensembles, not of the object.
```
Replacement:
```latex
A typical fermionic Gaussian state on the same counts gives a lower, later maximum than the Page maximum of Sec.~\ref{sec:page-effective}~\cite{Page1993,BianchiHacklKieburg2021} (values in Ref.~\cite{McKinneyCode2026}), curves of the comparison case under assumed ensembles, not of the object.
```

### 10.347 Sec. III: replaced text, verbatim (borrowF B4, the energy-energy bracket on the cubic lattice (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied; replaced spans and replacements, spec BF_B4_04 pair 1), beginning "No current depending only on \(u=f\nabla g-g\nabla f\) reproduces this"
Replaced text:
```latex
No current depending only on \(u=f\nabla g-g\nabla f\) reproduces this bracket (example in Ref.~\cite{McKinneyCode2026}).

At long wavelength the bracket takes the Dirac-Schwinger form
```
Replacement:
```latex
No current depending only on \(u=f\nabla g-g\nabla f\) reproduces this bracket (example in Ref.~\cite{McKinneyCode2026}).

% provenance: B4, the energy-energy bracket on the cubic lattice, from <internal>/FableAPI_borrowF_v1.md (md5 0c2e9b65) with the corrections of AstraAPI_borrowA_v1.md (md5 23a55ce4) applied: the opening assertion replaced by reviewer B's path-decomposition statement, the background-clock conclusion replaced by reviewer B's leading-order statement, the spin-expectation argument replaced by reviewer B's time-reversal sentence, "the factor F_a(xi)/xi" set as F_a(xi) and the retired label eq:dr-longitudinal-algebra re-pointed to the inline F_a(xi) above, the constant-flow mixed closure stated as long-wavelength O(a^2), the sentence "No current depending only on u ..." and the full-deformation-algebra caveats of Secs. II D and VII C kept unchanged (reviewer A's two clause edits there not applied); derivation notes in the delivery file; inserted as its own paragraph with a thesis and a closer, the paragraph pass's sentences kept; lead inbox 98, 104, 105.
% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids applied: A-B4.1 "No current depending only on u" sentence kept; A-B4.2 opening assertion replaced ("closes for any two smearings" not printed); A-B4.3 background-lapse statement leading order only; A-B4.4 grade-two expectation by time reversal; A-B4.5 F_a(xi) not F_a(xi)/xi; A-B4.6 "continuum bracket" retained; A-B4.7 constant-flow mixed closure long-wavelength O(a^2); A-B4.8 Secs. II D and VII C caveats kept; A-B4.9 dictionary clause in reviewer B's words (BF_B4_03.txt).
The lattice bracket itself has an exact path decomposition, and for affine smearings it reproduces the model's own flow and spin couplings with no remainder. Its scalar part contains axial two-step paths, while its cross-direction part is grade two, and general smearings have additional lattice dependence not determined by \(u\). Write \(P^I_f=\tfrac12\{f,T_I-T_I^\dagger\}=\bar f_IT_I-T_I^\dagger\bar f_I\), \(\bar f_I(x)=\tfrac12[f(x)+f(x+ae_I)]\), so that \(e[f]=\tfrac12\{f,h_0\}=\sum_I\alpha_IP^I_f/(2ia)+mf\beta_{\rm D}\). The mass terms drop out of \([e[f],e[g]]\) because \(\{P^I_f,g\}=\{P^I_g,f\}\) identically, same-direction terms are internal scalars, and cross-direction terms carry \(\alpha_I\alpha_J=-\alpha_J\alpha_I\), so
\begin{equation}
\begin{aligned}
i[e[f],e[g]]&=\sum_I\frac{\mathsf u_IT_I^2-T_I^{\dagger2}\mathsf u_I}{4ia^2}\\
&\quad+\sum_{I<J}\frac{\alpha_I\alpha_J\bigl(\{P^I_f,P^J_g\}-\{P^I_g,P^J_f\}\bigr)}{4ia^2},\\
\mathsf u_I(x)&=\bar f_I(x)\,\bar g_I(x+ae_I)-\bar g_I(x)\,\bar f_I(x+ae_I),
\end{aligned}
\label{eq:dr-lattice-bracket}
\end{equation}
exact on the lattice. The first sum is the two-step current \(Q_I\) of Sec.~\ref{sec:model-dictionary} smeared by the bond field \(\mathsf u_I/a\), and the second is the spin term with the lattice vertex \(\Sigma^{\rm lat}_K=\Sigma_K\mathsf c_I\mathsf c_J\) of Eq.~\eqref{eq:dm-hamiltonian}. For smearings affine in \(x\) both sums are exact couplings: with \(u=f\nabla g-g\nabla f\), itself affine, \(i[H[f],H[g]]=c^\dagger\bigl[\tfrac12\sum_I\{u_I,Q_I\}+\tfrac18\sum_K\{(\nabla\times u)_K,\Sigma^{\rm lat}_K\}\bigr]c\) with no remainder. For smooth smearings the bond field expands about the two-step midpoint as \(u_I+\tfrac{a^2}{4}(\partial_I^2f\,\partial_Ig-\partial_If\,\partial_I^2g)+\tfrac{a^2}{6}(f\partial_I^3g-g\partial_I^3f)+O(a^4)\), which for the harmonic pair above is \(\xi[1-\tfrac5{12}(a\xi)^2]\), the \(F_a(\xi)\) above. For real smearings the grade-two term is odd under the time reversal of Appendix~\ref{app:local}, so its expectation vanishes in \(P_0\) and in any covariance \(F(h[N,0])\), but not necessarily in a flow-prepared or propagated state. The long-wavelength form of this decomposition is the relativistic bracket that follows.

At long wavelength the bracket takes the Dirac-Schwinger form
```

### 10.348 Sec. III: replaced text, verbatim (borrowF B4, the energy-energy bracket on the cubic lattice (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied; replaced spans and replacements, spec BF_B4_04 pair 2), beginning "Radial \(f,g\) have zero curl, so their continuum bracket closes on th"
Replaced text:
```latex
Radial \(f,g\) have zero curl, so their
continuum bracket closes on the momentum density.
```
Replacement:
```latex
Radial \(f,g\) have zero curl, so their continuum bracket closes on the momentum density. Smearing about a background clock, \(f=N\mathcal N\) and \(g=N\mathcal M\), the long-wavelength bracket has \(u=N^2(\mathcal N\nabla\mathcal M-\mathcal M\nabla\mathcal N)\), the clock's own gradient cancelling, which identifies the background-lapse structure function at leading order, \(N^2\delta^{IJ}\), the clock squared times the flat co-metric of Eq.~\eqref{eq:dm-metric} at \(v=0\), not an exact finite-lattice constraint algebra. For a constant flow the mixed bracket closes at long wavelength, \(i[\mathcal P[v],H[N]]=H[v\cdot\nabla N]+O(a^2)\) with \(\mathcal P[v]=c^\dagger\tfrac12\sum_I\{v_I,Q_I\}c\), and for a flow with strain it does not, as the dilation shows next.
```

### 10.349 Sec. II: replaced text, verbatim (borrowF B4, the energy-energy bracket on the cubic lattice (borrowF 0c2e9b65, borrowA 23a55ce4, lead inbox 98, 104, 105), applied; replaced spans and replacements, spec BF_B4_03 pair 1), beginning "Neither coefficient is free: the bracket of Sec.~\ref{sec:results-curr"
Replaced text:
```latex
Neither coefficient is free: the bracket of Sec.~\ref{sec:results-current} fixes the flow term's, and the prepared sea's rest under a bulk rigid turn \(v=\omega\times x\) fixes the spin vertex within \(\Sigma_kf_k(p)\).
```
Replacement:
```latex
Neither coefficient is free: both follow from the exact bulk bracket for affine smearings (Sec.~\ref{sec:results-current}), and the completed rigid turn independently commutes with \(h_0\), the prepared sea's rest under a bulk rigid turn \(v=\omega\times x\) fixing the spin vertex within \(\Sigma_kf_k(p)\).
```

### 10.350 Sec. III: replaced text, verbatim (owner comment cmw0g42tp1m26, Sec. III opener clause reworded (lead-routed), spec I21_04 pair 1), beginning "The order is the order of need: the light cone fixes the metric, the e"
Replaced text:
```latex
The order is the order of need: the light cone fixes the metric, the energy algebra fixes the current,
```
Replacement:
```latex
Each step uses the one before it: the light cone fixes the metric, the energy algebra fixes the current,
```

### 10.351 Sec. IV: replaced text, verbatim (lead inbox 107, the Sec. IV C Hessian-error bound sentence and the narrowed todo (gapHessF d2636c36, gapHessA 3cf70dbf); replaced spans and replacements, spec I22_05 pair 1), beginning "The interior's Hartree-Fock level is read on 18 stored slices"
Replaced text:
```latex
The interior's Hartree-Fock level is read on 18 stored slices
```
Replacement:
```latex
% provenance: the computed Hessian's error bound for every direction: <internal>/FableAPI_gapHessF_v1.md (md5 d2636c36) and AstraAPI_gapHessA_v1.md (md5 3cf70dbf), two independent proofs that the restriction to errors parallel to the computed Hessian is unnecessary (the error enters the deviation through Gaussian traces of the Hartree state, an all-direction bound follows from the validated constraint solve and a residual bound for its linear response, the Wick Gram matrix giving the optimal constant for a given Hessian-error norm); scope taken from gapHessA: the implemented held-source operators, not Eq. dm-reference, the interior 0.080 a test-subspace overlap and not a branch certificate, the numerical enclosure on the eighteen slices not certified from the files (response residuals, inverse bounds, Wick sensitivity constants, and non-Hessian error terms to be measured, the hreduce pilot running); lead inbox 107.
The interior's Hartree-Fock level is read on 18 stored slices
```

### 10.352 Sec. IV: replaced text, verbatim (lead inbox 107, the Sec. IV C Hessian-error bound sentence and the narrowed todo (gapHessF d2636c36, gapHessA 3cf70dbf); replaced spans and replacements, spec I22_05 pair 2), beginning "It is conditional on the bound for the computed Hessian's error, prove"
Replaced text:
```latex
It is conditional on the bound for the computed Hessian's error, proved only for errors parallel to the computed Hessian.
```
Replacement:
```latex
The computed Hessian's error of any direction enters this deviation through Gaussian traces of the Hartree state and is enclosed by the constraint response residual times the verified inverse norm of the implemented Jacobian, with the Wick Gram matrix giving the optimal constant for a given error norm, so the bound no longer needs errors parallel to the computed Hessian, holds for the implemented held-source operators, and awaits its numerical enclosure on the 18 slices.
```

### 10.353 Sec. IV: replaced text, verbatim (lead inbox 107, the Sec. IV C Hessian-error bound sentence and the narrowed todo (gapHessF d2636c36, gapHessA 3cf70dbf); replaced spans and replacements, spec I22_05 pair 3), beginning "\todo{prove the bound for the computed Hessian's error, including erro"
Replaced text:
```latex
\todo{prove the bound for the computed Hessian's error, including errors not parallel to the computed Hessian, and propagate it on two independent calculations to branch weights and mass biases on all 18 slices}
```
Replacement:
```latex
\todo{the numerical enclosure of the computed Hessian's error on the 18 slices from the response residuals, inverse bounds, Wick sensitivity constants, and non-Hessian error terms, propagated on two independent calculations to branch weights and mass biases}
```

### 10.354 Sec. II: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_03 pair 1), beginning "The sources' lattice corrections are controlled:"
Replaced text:
```latex
The sources' lattice corrections are controlled:
```
Replacement:
```latex
All three sources are now defined, and their lattice corrections are controlled:
```

### 10.355 Sec. III: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_04 pair 1), beginning "Appendix~\ref{app:local}'s two estimates bound a comparator's effect o"
Replaced text:
```latex
Appendix~\ref{app:local}'s two estimates bound a comparator's effect on the area coefficient under a gap premise, by
```
Replacement:
```latex
The comparator-projector comparison needs a gap premise, and Appendix~\ref{app:local}'s two estimates bound its effect on the area coefficient by
```

### 10.356 Sec. IV: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_05 pair 1), beginning "fix the lapse \(N\) on a slice, not a received"
Replaced text:
```latex
fix the lapse \(N\) on a slice, not a received
```
Replacement:
```latex
also fix the lapse \(N\) on a slice, not a received
```

### 10.357 Sec. IV: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_05 pair 2), beginning "Which modes are thrown outward follows from Sec.~\ref{sec:results-cone"
Replaced text:
```latex
Which modes are thrown outward follows from Sec.~\ref{sec:results-cone}:
```
Replacement:
```latex
Which modes are thrown outward, the last analytic ingredient, follows from Sec.~\ref{sec:results-cone}:
```

### 10.358 Sec. IV: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_05 pair 3), beginning "Finally the computed geometry is checked against the first law and the"
Replaced text:
```latex
Finally the computed geometry is checked against the first law and the stress it should obey.
```
Replacement:
```latex
Finally the computed geometry is checked against the first law, the stress it should obey, and the direct-state budget of Sec.~\ref{sec:results-vacuum}.
```

### 10.359 Sec. VI: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_06 pair 1), beginning "After crossing, a vacuum exterior with \(E\) conserved"
Replaced text:
```latex
After crossing, a vacuum exterior with \(E\) conserved
```
Replacement:
```latex
After the surface crosses \(r_s\), a vacuum exterior with \(E\) conserved
```

### 10.360 Sec. VI: replaced text, verbatim (join verification FIX rows (PARAGRAPH_STRUCTURE_20261103.md, owner ruling 2026-11-04), the seven next-opener rewordings as proposed; replaced text and replacement, spec J_06 pair 2), beginning "A finite-time crossing of the branches, \(\mathcal B(u_P)=fy(u_P)S_0\)"
Replaced text:
```latex
A finite-time crossing of the branches, \(\mathcal B(u_P)=fy(u_P)S_0\) at finite \(u_P\)
```
Replacement:
```latex
A finite-time crossing of the two branches of Eq.~\eqref{eq:page-effective-curve}, \(\mathcal B(u_P)=fy(u_P)S_0\) at finite \(u_P\)
```

### 10.361 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 1), beginning "% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids"
Replaced text:
```latex
% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids applied: A-B4.1
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B4 (b)): the definitions of P^I_f and e[f], the mass-term cancellation, the display eq:dr-lattice-bracket (label retired, cited nowhere), the bond-field expansion and the explicit midpoint formula moved to Ref. McKinneyCode2026 behind one pointer sentence; the thesis, the scope statement, the affine-smearing result, the F_a(xi) link, the time-reversal condition and the closer kept; replaced text archived verbatim in REPO_DERIVATIONS 10.362.
% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids applied: A-B4.1
```

### 10.362 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 2), beginning "Its scalar part contains axial two-step paths, while its cross-directi"
Replaced text:
```latex
Its scalar part contains axial two-step paths, while its cross-direction part is grade two, and general smearings have additional lattice dependence not determined by \(u\). Write \(P^I_f=\tfrac12\{f,T_I-T_I^\dagger\}=\bar f_IT_I-T_I^\dagger\bar f_I\), \(\bar f_I(x)=\tfrac12[f(x)+f(x+ae_I)]\), so that \(e[f]=\tfrac12\{f,h_0\}=\sum_I\alpha_IP^I_f/(2ia)+mf\beta_{\rm D}\). The mass terms drop out of \([e[f],e[g]]\) because \(\{P^I_f,g\}=\{P^I_g,f\}\) identically, same-direction terms are internal scalars, and cross-direction terms carry \(\alpha_I\alpha_J=-\alpha_J\alpha_I\), so
\begin{equation}
\begin{aligned}
i[e[f],e[g]]&=\sum_I\frac{\mathsf u_IT_I^2-T_I^{\dagger2}\mathsf u_I}{4ia^2}\\
&\quad+\sum_{I<J}\frac{\alpha_I\alpha_J\bigl(\{P^I_f,P^J_g\}-\{P^I_g,P^J_f\}\bigr)}{4ia^2},\\
\mathsf u_I(x)&=\bar f_I(x)\,\bar g_I(x+ae_I)-\bar g_I(x)\,\bar f_I(x+ae_I),
\end{aligned}
\label{eq:dr-lattice-bracket}
\end{equation}
exact on the lattice. The first sum is the two-step current \(Q_I\) of Sec.~\ref{sec:model-dictionary} smeared by the bond field \(\mathsf u_I/a\), and the second is the spin term with the lattice vertex \(\Sigma^{\rm lat}_K=\Sigma_K\mathsf c_I\mathsf c_J\) of Eq.~\eqref{eq:dm-hamiltonian}. For smearings affine in \(x\) both sums are exact couplings: with \(u=f\nabla g-g\nabla f\), itself affine, \(i[H[f],H[g]]=c^\dagger\bigl[\tfrac12\sum_I\{u_I,Q_I\}+\tfrac18\sum_K\{(\nabla\times u)_K,\Sigma^{\rm lat}_K\}\bigr]c\) with no remainder. For smooth smearings the bond field expands about the two-step midpoint as \(u_I+\tfrac{a^2}{4}(\partial_I^2f\,\partial_Ig-\partial_If\,\partial_I^2g)+\tfrac{a^2}{6}(f\partial_I^3g-g\partial_I^3f)+O(a^4)\), which for the harmonic pair above is \(\xi[1-\tfrac5{12}(a\xi)^2]\), the \(F_a(\xi)\) above.
```
Replacement:
```latex
Its exact decomposition in two-step shifts (Ref.~\cite{McKinneyCode2026}) has a scalar part, axial two-step paths forming the current \(Q_I\) of Sec.~\ref{sec:model-dictionary} smeared by a bond field built from \(f\) and \(g\), and a grade-two cross-direction part, the spin term with the lattice vertex \(\Sigma^{\rm lat}_K=\Sigma_K\mathsf c_I\mathsf c_J\) of Eq.~\eqref{eq:dm-hamiltonian}, while general smearings have additional lattice dependence not determined by \(u\). For smearings affine in \(x\) both parts are exact couplings: with \(u=f\nabla g-g\nabla f\), itself affine, \(i[H[f],H[g]]=c^\dagger\bigl[\tfrac12\sum_I\{u_I,Q_I\}+\tfrac18\sum_K\{(\nabla\times u)_K,\Sigma^{\rm lat}_K\}\bigr]c\) with no remainder, and for smooth smearings the bond field's midpoint expansion gives, for the harmonic pair above, the \(F_a(\xi)\) above.
```

### 10.363 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 3), beginning "At long wavelength the bracket takes the Dirac-Schwinger form~\cite{Di"
Replaced text:
```latex
At long wavelength the bracket takes the Dirac-Schwinger form~\cite{Dirac1962,Schwinger1962} of a relativistic energy density: for continuum-spinor smearings \(f,g\) (valley sum and valley-rotated spin matrices
understood), the bare bracket is
\begin{equation}
\begin{aligned}
i[H[f],H[g]]&=\mathcal P[u]+\mathcal S[u]+\mathcal R_a,\\
\mathcal P[u]&=\int d^3x\,\psi^\dagger\tfrac12\{u_i,-i\partial_i\}\psi,\\
\mathcal S[u]&=\tfrac14\int d^3x\,\psi^\dagger\boldsymbol\Sigma\cdot(\nabla\times u)\psi,
\end{aligned}
\label{eq:dr-dirac-schwinger}
\end{equation}
with a one-particle lattice remainder bounded by
```
Replacement:
```latex
At long wavelength the bare bracket takes the Dirac-Schwinger form~\cite{Dirac1962,Schwinger1962} of a relativistic energy density, a momentum density \(\mathcal P[u]\) plus a spin term \(\mathcal S[u]\) at \(u=f\nabla g-g\nabla f\) for continuum-spinor smearings \(f,g\) (valley sum and valley-rotated spin matrices understood), its one-particle lattice remainder bounded by
```

### 10.364 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 4), beginning "the counter-advected set for inward flow is \(\{\pi/4<|ak|\le\pi/2\}\)"
Replaced text:
```latex
the counter-advected set for
inward flow is \(\{\pi/4<|ak|\le\pi/2\}\), \(k\) the momentum from
a valley, giving the comparison radius
```
Replacement:
```latex
the counter-advected momentum set (Ref.~\cite{McKinneyCode2026}) gives the comparison radius
```

### 10.365 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 5), beginning "Outside \(r=2GE\), the Schwarzschild time \(dt_s=dt+v\,dr/(1-v^2)\) an"
Replaced text:
```latex
Outside \(r=2GE\), the Schwarzschild time \(dt_s=dt+v\,dr/(1-v^2)\) and the isotropic radius \(r=\tilde r(1+GE/2\tilde r)^2\) put the exterior in the standard isotropic form of the Schwarzschild metric, whose post-Newtonian (PN) parameters are
\(\gamma_{\rm PN}=\beta_{\rm PN}=1\)~\cite{Will2014}.
```
Replacement:
```latex
Outside \(r=2GE\) the exterior is Schwarzschild's, with Schwarzschild time \(dt_s=dt+v\,dr/(1-v^2)\), and in isotropic form (Ref.~\cite{McKinneyCode2026}) its post-Newtonian (PN) parameters are \(\gamma_{\rm PN}=\beta_{\rm PN}=1\)~\cite{Will2014}.
```

### 10.366 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 6), beginning "% status (lead inbox 105): APPLIED WITH LABEL for the analytic part (i"
Replaced text:
```latex
% status (lead inbox 105): APPLIED WITH LABEL for the analytic part (illustrative scheme, derived, not the solver's)
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B8 (a)): the trigamma expansion cited to Olver1993, the effective-source formula with its pressures and the PN bookkeeping moved to Ref. McKinneyCode2026 behind one clause; the thesis, the discretization sentence, the display eq:dr-discrete-exterior, the production-solver sentence, beta_PN conditional on the scheme, the back-averaged coefficients and the closer kept; replaced text archived verbatim in REPO_DERIVATIONS 10.367.
% status (lead inbox 105): APPLIED WITH LABEL for the analytic part (illustrative scheme, derived, not the solver's)
```

### 10.367 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 7), beginning "a radius read at the bond midpoint having no first-order remainder and"
Replaced text:
```latex
a radius read at the bond midpoint having no first-order remainder and one read at a node \(c_1=\pm\tfrac12\), while a node-weighted cumulative scheme gives the exact trigamma profile \(v_n^2=2GE\,\psi_1(n)/a\), a different scheme whose expansion is the \(\theta=0\) case through first order. This discretization has not been identified with the production solver. Read through the rows at \(N=1\) as a mass function, \(2G\mathcal M_{\rm eff}=rv^2\), the first-order term is a stationary continuum interpretation of the scheme's error, an effective exterior source \(\rho_{\rm lat}=-c_1r_sa/(8\pi Gr^4)\) with \(p_r=-\rho_{\rm lat}\) and \(p_\perp=\rho_{\rm lat}\), traceless and falling as \(r^{-4}\) like a radial Maxwell field, with \(\beta_{\rm PN}=1-2c_1a/r_s\) conditional on the scheme, orders and signs being scheme dependent and not lattice corrections of the model.
```
Replacement:
```latex
a radius read at the bond midpoint having no first-order remainder and one read at a node \(c_1=\pm\tfrac12\), while a node-weighted cumulative scheme gives the trigamma profile \(v_n^2=2GE\,\psi_1(n)/a\)~\cite{Olver1993}, the \(\theta=0\) case through first order. This discretization has not been identified with the production solver. Read through the rows at \(N=1\) as a mass function, the first-order term is a stationary continuum interpretation of the scheme's error, a traceless effective exterior source falling as \(r^{-4}\) like a radial Maxwell field (its form and PN bookkeeping in Ref.~\cite{McKinneyCode2026}), with \(\beta_{\rm PN}=1-2c_1a/r_s\) conditional on the scheme, orders and signs being scheme dependent and not lattice corrections of the model.
```

### 10.368 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 8), beginning "For \(\kappa_R\ne0\) it has the exact area primitive \((\delta E-W_{\r"
Replaced text:
```latex
For \(\kappa_R\ne0\) it has the exact area primitive \((\delta E-W_{\rm ext})/(\kappa_R/2\pi)=\delta(A/4G)\) whatever the exterior work. The no-work form \(\delta E/(\kappa_R/2\pi)\) is integrable only where \(d\kappa_R\wedge dE=0\) on the regular family.
```
Replacement:
```latex
For \(\kappa_R\ne0\) its exact primitive is \(\delta(A/4G)\) whatever the exterior work, and the no-work form is integrable only where \(d\kappa_R\wedge dE=0\) on the regular family (Ref.~\cite{McKinneyCode2026}).
```

### 10.369 Sec. III: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_04 pair 9), beginning "One thermodynamic temperature \(T(E)=T_H[1+\delta(E)]\), not supplied "
Replaced text:
```latex
One thermodynamic temperature \(T(E)=T_H[1+\delta(E)]\), not
supplied by mode-resolved \(\delta_T\), would integrate
\(dE=T\,dS_{\rm th}\) at fixed other charges to~\cite{BardeenCarterHawking1973}
\begin{equation}
 S_{\rm th}(E)=\frac{A(E)}{4G}+C-8\pi G\int^E dE'\,
 \frac{E'\delta(E')}{1+\delta(E')} ,
 \label{eq:page-first-law}
\end{equation}
\(A=4\pi r_s^2\), with integration constant \(C\). The leading
first-law entropy \(A/(4G)+C\) is the Bekenstein-Hawking
value~\cite{Bekenstein1973,Hawking1975}, supplied by the fermion's
entanglement at one spacing, not by state counting.
```
Replacement:
```latex
One thermodynamic temperature, not supplied by the mode-resolved \(\delta_T\), would integrate \(dE=T\,dS_{\rm th}\) at fixed other charges~\cite{BardeenCarterHawking1973}, at \(T=T_H\) to
\begin{equation}
 S_{\rm th}(E)=\frac{A(E)}{4G}+C,
 \label{eq:page-first-law}
\end{equation}
\(A=4\pi r_s^2\), with integration constant \(C\) (a correction \(\delta(E)\) to \(T_H\) adds an integral, Ref.~\cite{McKinneyCode2026}). This is the Bekenstein-Hawking value~\cite{Bekenstein1973,Hawking1975}, supplied by the fermion's entanglement at one spacing, not by state counting.
```

### 10.370 Sec. IV: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_05 pair 1), beginning "centre and vacuum matching, has its first mass maximum, \(E_{\max}\), "
Replaced text:
```latex
centre and vacuum matching, has its first mass maximum, \(E_{\max}\), at
\begin{equation}
\begin{aligned}
E&=c_E/(G^{3/2}m^2),&Ea&=c_E/(g^{3/2}\zeta^2),\\
R&=c_R/(\sqrt G\,m^2),&R/a&=c_R/(\sqrt g\,\zeta^2),
\end{aligned}
\label{eq:sc-mass-radius}
\end{equation}
% source: sqrt8 Oppenheimer-Volkoff cross-check: not located (PROVENANCE_TRACE_v1 317c6d92 has no entry); lead inbox 69.
with \(\zeta=ma\), \(g=G/a^2\), and \((c_E,c_R)=(0.135858,1.18689)\), the two-state Oppenheimer-Volkoff~\cite{OppenheimerVolkoff1939} values divided by \(\sqrt8\) (a relation quoted, not cross-checked numerically). The computed cloud lies above it:
```
Replacement:
```latex
centre and vacuum matching, has its first mass maximum at \(E_{\max}=c_E/(G^{3/2}m^2)\), \(E_{\max}a=c_E/(g^{3/2}\zeta^2)\),
% source: sqrt8 Oppenheimer-Volkoff cross-check: not located (PROVENANCE_TRACE_v1 317c6d92 has no entry); lead inbox 69.
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part B Sec. IV A (a)): the four-line display eq:sc-mass-radius (label retired, cited nowhere) set inline for E_max, the radius line and c_R = 1.18689 to Ref. McKinneyCode2026 (R/a is used nowhere else); replaced text archived verbatim in REPO_DERIVATIONS 10.370.
with \(\zeta=ma\), \(g=G/a^2\), and \(c_E=0.135858\), the two-state Oppenheimer-Volkoff~\cite{OppenheimerVolkoff1939} value divided by \(\sqrt8\) (a relation quoted, not cross-checked numerically, the radius at the maximum in Ref.~\cite{McKinneyCode2026}). The computed cloud lies above it:
```

### 10.371 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 1), beginning "% status (lead inbox 105): APPLIED WITH LABEL. borrowA ids applied: A-"
Replaced text:
```latex
% status (lead inbox 105): APPLIED WITH LABEL. borrowA ids applied: A-B7.1
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B7 (a)+(b)): the static-frame reduction of the massless Dirac equation and the display eq:page-dirac-barrier (label retired, cited nowhere) to one clause with Unruh1976 and Chandrasekhar1983 and a Ref. pointer, the transmission sentence tightened, the Landauer currents display eq:page-currents (label retired, cited nowhere) to one clause with Datta1995 and Peschel2003, the mode-by-mode ratio argument to the Ref., the binary-transmission limits said in words (elementary Fermi integrals); the thesis, the channel operator, the assumed-supply qualification, the finite-band statement with mark B7, the number-rate sentence, the typical-Gaussian clause with its citations and the closer kept; replaced text archived verbatim in REPO_DERIVATIONS 10.372 and 10.373.
% status (lead inbox 105): APPLIED WITH LABEL. borrowA ids applied: A-B7.1
```

### 10.372 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 2), beginning "For the leading continuum limit on the unaveraged vacuum profile, writ"
Replaced text:
```latex
For the leading continuum limit on the unaveraged vacuum profile, write the stationary chain spinor as \(e^{-i\omega t}\psi_\omega(r)\), set \(t_s=t+F(r)\) with \(F'=v/\alpha_s^2\), \(\alpha_s=\sqrt{1-r_s/r}\) the Killing lapse, and define \(\chi=\alpha_s^{1/2}e^{\eta\sigma_1/2}e^{i\omega F}\psi_\omega\) with \(\tanh\eta=v\). Then
\begin{equation}
\omega\chi=\bigl[-i\sigma_1\partial_{r_*}+W\sigma_2\bigr]\chi,\qquad
W=\frac{\alpha_sk}{r},\qquad dr_*=\frac{dr}{\alpha_s^2},
\label{eq:page-dirac-barrier}
\end{equation}
and since \(\sigma_2\) anticommutes with \(\sigma_1\) the two components see the partner barriers \(W^2\pm\partial_{r_*}W\), rising to the centrifugal \(k(k\pm1)/r^2\) far out and vanishing at \(r_s\) with \(\alpha_s\). The unit-flux transmission \(\Gamma_k(\omega)\) of this barrier, a ratio of asymptotic fluxes and so the same in either time, the same for the two partners and for both signs of \(\kappa=\pm k\), is the continuum comparison transmission, finite-spacing and back-averaging corrections requiring separate calculation. Feeding each channel thermally at \(T_H\) from the inner port with the outer port empty, an assumed supply not established on the collapse, and counting the eight valleys, both signs of \(\kappa\), particles and holes, \(64k\) modes per unit of \(\omega/2\pi\),
\begin{equation}
\begin{gathered}
L=\sum_k64k\!\int_0^\infty\!\frac{d\omega}{2\pi}\,\omega\,n_k,\\
\dot S_{\rm marg}=\sum_k64k\!\int_0^\infty\!\frac{d\omega}{2\pi}\,s(n_k),\qquad
n_k=\frac{\Gamma_k(\omega)}{e^{\omega/T_H}+1},
\end{gathered}
\label{eq:page-currents}
\end{equation}
the entropy taken at the transmitted occupation because tracing out the reflected part leaves the transmitted mode Gaussian at \(\Gamma_kf\). Summed over the eight valleys, particles and antiparticles, \(|k|\le8\),
```
Replacement:
```latex
In the leading continuum limit on the unaveraged vacuum profile the stationary chain spinor, rescaled by \(\alpha_s^{1/2}\), boosted to the static frame and stripped of its stationary phase, obeys the massless Dirac equation on Schwarzschild in the tortoise coordinate \(r_*\)~\cite{Unruh1976,Chandrasekhar1983}, \(\alpha_s=\sqrt{1-r_s/r}\) the Killing lapse, so its two components see the partner barriers \(W^2\pm\partial_{r_*}W\), \(W=\alpha_sk/r\), rising to the centrifugal \(k(k\pm1)/r^2\) far out and vanishing at \(r_s\) with \(\alpha_s\) (the reduction in Ref.~\cite{McKinneyCode2026}). The unit-flux transmission \(\Gamma_k(\omega)\) of this barrier, a flux ratio and so the same in either time, for both partners and for both signs of \(\kappa=\pm k\), is the continuum comparison transmission, finite-spacing and back-averaging corrections requiring separate calculation. Feeding each channel thermally at \(T_H\) from the inner port with the outer port empty, an assumed supply not established on the collapse, the energy and entropy currents are the Landauer currents~\cite{Datta1995} at the transmitted occupation \(n_k=\Gamma_k(\omega)/(e^{\omega/T_H}+1)\), with \(64k\) modes per unit of \(\omega/2\pi\) for the eight valleys, both signs of \(\kappa\), particles and holes, the entropy taken at that occupation because the transmitted mode stays Gaussian once the reflected part is traced out~\cite{Peschel2003}. Summed over the eight valleys, particles and antiparticles, \(|k|\le8\),
```

### 10.373 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 3), beginning "The ratio \(\mathcal R=T_H\dot S_{\rm marg}/L\) is fixed by the one tr"
Replaced text:
```latex
The ratio \(\mathcal R=T_H\dot S_{\rm marg}/L\) is fixed by the one transmission and the one occupation and exceeds one mode by mode, since \(s(x)-(\omega/T_H)x\) vanishes at \(x=0\) and increases up to \(x=f\). For binary transmissions, \(\Gamma_k\in\{0,1\}\), the currents depend on the open-channel count \(C(\omega)\), a constant count giving \(\mathcal R=2\) and \(C\propto\omega^2\) giving \(4/3\), analytic comparison limits, not bounds on a soft barrier, for which \(\sum_ks(\Gamma_kf)\) is not determined by \(\sum_k\Gamma_k\).
```
Replacement:
```latex
The ratio \(\mathcal R=T_H\dot S_{\rm marg}/L\) is fixed by the one transmission and the one occupation and exceeds one mode by mode (Ref.~\cite{McKinneyCode2026}). For binary transmissions the currents depend only on the open-channel count, a constant count giving \(\mathcal R=2\) and a count growing as \(\omega^2\) giving \(4/3\) by elementary Fermi integrals, comparison limits and not bounds on a soft barrier, whose entropy sum is not fixed by its transmission sum.
```

### 10.374 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 4), beginning "If this quasi-steady emission persists and alone depletes \(E\), the c"
Replaced text:
```latex
If this quasi-steady emission persists and alone depletes \(E\), the charge formally reaches zero at \(t_*=r_{s0}^3/(6G\mathcal L)\), \(r_{s0}=2GE_0\), and the leading first-law entropy \(S(E)=A(E)/(4G)\) of Eq.~\eqref{eq:page-first-law} falls from \(S_0=S(E_0)\) as \(S_0(1-t_\infty/t_*)^{2/3}\).
```
Replacement:
```latex
If this quasi-steady emission persisted and alone depleted \(E\), the charge would reach zero at \(t_*=r_{s0}^3/(6G\mathcal L)\)~\cite{Page1976}, \(r_{s0}=2GE_0\), with \(S(E)=A(E)/(4G)\) the first-law entropy of Eq.~\eqref{eq:page-first-law} and \(S_0=S(E_0)\).
```

### 10.375 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 5), beginning "Unitary quadratic evolution preserves the global purity of the initial"
Replaced text:
```latex
Unitary quadratic evolution preserves the global purity of the initially pure light covariance, while
```
Replacement:
```latex
Unitary quadratic evolution preserves the global purity of the initially pure light covariance~\cite{Peschel2003}, while
```

### 10.376 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 6), beginning "Against a declared equal-rank comparator projector \(P\), the differen"
Replaced text:
```latex
Against a declared equal-rank comparator projector \(P\), the difference \(C_L-P\) has paired positive and negative eigenvalues, its trace norm is a state-comparator distance, and the particle and hole excitation numbers are \(N_p=N_h=\Tr[(I-P)C_L]=\tfrac12\Tr(C_L-P)^2\), whose interpretation depends on the comparator,
```
Replacement:
```latex
Against a declared equal-rank comparator projector \(P\), the trace norm of \(C_L-P\) is a state-comparator distance and the particle and hole excitation numbers are \(N_p=N_h=\Tr[(I-P)C_L]\)~\cite{PeschelEisler2009}, whose interpretation depends on the comparator,
```

### 10.377 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 7), beginning "% status (lead inbox 105): APPLIED WITH LABEL (purity derived"
Replaced text:
```latex
% status (lead inbox 105): APPLIED WITH LABEL (purity derived
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B3b (a)): Gaussian purity under quadratic evolution cited to Peschel2003, the paired spectrum and the second form of the excitation count cited to PeschelEisler2009; both B3 marks and the comparator-dependence statement kept; replaced text archived verbatim in REPO_DERIVATIONS 10.375 and 10.376.
% status (lead inbox 105): APPLIED WITH LABEL (purity derived
```

### 10.378 Sec. V: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_07 pair 8), beginning "The companion is one chiral Dirac field on a ray circle, the massless "
Replaced text:
```latex
The companion is one chiral Dirac field on a ray circle, the massless channel without its centrifugal term unfolded through the regular centre (a scaling description
```
Replacement:
```latex
The companion is one chiral Dirac field on a ray circle, the unfolded massless channel (a scaling description
```

### 10.379 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 1), beginning "The dust solution's surface approaches the Schwarzschild radius expone"
Replaced text:
```latex
The dust solution's surface approaches the Schwarzschild radius exponentially: on the dust solution, Eq.~\eqref{eq:star-dust-control}, with \(R_b>r_s\), \(\alpha_s\), \(t_s\), and \(u\) as in Sec.~\ref{sec:star}
and \(w=v(R_b)/N(R_b)=-\sqrt{r_s/R_b}\),
\begin{equation}
\begin{aligned}
d\alpha_s^2/dt_s&=-\alpha_s^2(1-\alpha_s^2)^{5/2}/r_s,\\
\alpha_s^2&=\theta-\tfrac52\theta^2+O(\theta^3),\qquad
\theta=\theta_0e^{-t_s/r_s},
\end{aligned}
\label{eq:sc-approach}
\end{equation}
with \(\theta_0\) set by the launch: the Schwarzschild factor
\(\alpha_s^2\) falls as
\(e^{-t_s/r_s}\), Eq.~\eqref{eq:star-peeling-control}, with no
positive dynamical floor.
```
Replacement:
```latex
The dust solution's surface approaches the Schwarzschild radius exponentially: on the dust solution, Eq.~\eqref{eq:star-dust-control}, with \(R_b>r_s\), \(\alpha_s\), \(t_s\), and \(u\) as in Sec.~\ref{sec:star} and \(w=v(R_b)/N(R_b)=-\sqrt{r_s/R_b}\), the Schwarzschild factor \(\alpha_s^2\) of the surface falls as \(e^{-t_s/r_s}\)~\cite{OppenheimerSnyder1939}, Eq.~\eqref{eq:star-peeling-control}, with no positive dynamical floor.
```

### 10.380 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 2), beginning "the window ends where the sine dispersion is no longer small, \(\alpha"
Replaced text:
```latex
the window ends where the sine dispersion is no longer small, \(\alpha_s=\alpha_d\) in Eq.~\eqref{eq:sc-approach}.
```
Replacement:
```latex
the window ends where the sine dispersion is no longer small, \(\alpha_s=\alpha_d\) on the surface's approach.
```

### 10.381 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 3), beginning "On the dust solution the map is exact: with \(s=\sqrt{R_b/r_s}\), a ra"
Replaced text:
```latex
On the dust solution the map is exact: with \(s=\sqrt{R_b/r_s}\), a ray entering the ball at \(s+1\) leaves it at \(s\), so \(dU/du=(s+1)^3(s-1)/[s^3(s+2)]\) and \(\kappa_u/\kappa_H=6/[s^4(s+1)(s+2)]\). The segment on which \(\kappa_u\) lies within a fraction \(c\) of \(\kappa_H\) ends at \(s_d=(1-\delta_d)^{-1/2}\), and for a nonempty window
```
Replacement:
```latex
On the dust solution the map is exact (Ref.~\cite{McKinneyCode2026}), and the segment on which \(\kappa_u\) lies within a fraction \(c\) of \(\kappa_H\) ends at the dispersive cutoff \(\alpha_s^2=\delta_d\), so for a nonempty window
```

### 10.382 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 4), beginning "The received redshift grows at Hawking's exponent until dispersion end"
Replaced text:
```latex
The received redshift grows at Hawking's exponent until dispersion ends the long-wave description: for \(1\gg\delta/r_s\gg\max\{(2\chi q)^{2/3},\sigma^2/r_s^2\}\),
\(\delta=R_e-r_s\), an outgoing long-wave front in the exterior of Eq.~\eqref{eq:dr-exterior}, emitted from the surface at \((t_e,R_e)\), reaches a static receiver at retarded time \(u=t_e+2r_s\ln(2r_s/\delta)-3r_s+O(\delta)\) in Sec.~\ref{sec:star}'s normalization (the exterior's retarded coordinate in Ref.~\cite{McKinneyCode2026}). Since \(\delta=t_c-t_e+O((t_c-t_e)^2/r_s)\), \(R_e(u)-r_s=2e^{-3/2}r_se^{-(u-t_c)/(2r_s)}[1+O(e^{-(u-t_c)/(2r_s)})]\), launch entering only through \(t_c\), and the surface's emission is received at \(g_\infty=N+v=1-\sqrt{r_s/R_e}\), \(\ln(1/g_\infty)=(u-t_c)/(2r_s)+\tfrac32+O(\delta/r_s)\), with Eq.~\eqref{eq:sc-ray-temperature}'s exponent \(\kappa_H\).
```
Replacement:
```latex
The received redshift grows at Hawking's exponent until dispersion ends the long-wave description: for \(1\gg\delta/r_s\gg\max\{(2\chi q)^{2/3},\sigma^2/r_s^2\}\), \(\delta=R_e-r_s\), a long-wave front emitted from the surface at \((t_e,R_e)\) in the exterior of Eq.~\eqref{eq:dr-exterior} reaches a static receiver at a retarded time \(u\) with \(R_e(u)-r_s\propto e^{-(u-t_c)/(2r_s)}\), launch entering only through \(t_c\), and its redshift \(\ln(1/g_\infty)=(u-t_c)/(2r_s)+\tfrac32+O(\delta/r_s)\), \(g_\infty=N+v\), grows with Eq.~\eqref{eq:sc-ray-temperature}'s exponent \(\kappa_H\)~\cite{Hawking1975,MTW1973} (the retarded-time arithmetic in Ref.~\cite{McKinneyCode2026}).
```

### 10.383 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 5), beginning "The Hawking comparison case has an adiabatic evaporation scale \(E/L\)"
Replaced text:
```latex
The Hawking comparison case has an adiabatic evaporation scale \(E/L\), \(r_s/(E/L)=\tfrac12|\dot\kappa_H|/\kappa_H^2=2g\mathcal L\chi^2\), using quasi-steady emission at the instantaneous mass, not a steady supply.
```
Replacement:
```latex
The Hawking comparison case's adiabatic evaporation scale is \(E/L\), with \(r_s/(E/L)=2g\mathcal L\chi^2\) under quasi-steady emission at the instantaneous mass, not a steady supply.
```

### 10.384 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 6), beginning "For the ideal stationary, regulated covariance with this symbol the Sz"
Replaced text:
```latex
For the ideal stationary, regulated covariance with this symbol the Szeg\H{o} limit gives the received entropy rate, \(64k\) modes per unit of \(\omega/2\pi\) in channel \(k\), as the rates of Eq.~\eqref{eq:sc-luminosity}, in two independent calculations, \(\mathcal R=1.639\) nats per unit of energy over \(T_H\): basis-invariant within the received-mode subspace, not under arbitrary packet selection, and on a finite adiabatic segment assuming control of boundary, regulator, channel-tail and nonstationarity errors.
```
Replacement:
```latex
The Szeg\H{o} limit for the stationary regulated covariance with this symbol~\cite{Widom1980} gives the received entropy rate (\(64k\) modes per unit of \(\omega/2\pi\) in channel \(k\)) as the rates of Eq.~\eqref{eq:sc-luminosity}, \(\mathcal R=1.639\) nats per unit of energy over \(T_H\) in two independent calculations, basis-invariant within the received-mode subspace, not under arbitrary packet selection, and on a finite adiabatic segment conditional on control of boundary, regulator, channel-tail, and nonstationarity errors.
```

### 10.385 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 7), beginning "% status (lead inbox 105): APPLIED WITH LABEL (exact identity, the reg"
Replaced text:
```latex
% status (lead inbox 105): APPLIED WITH LABEL (exact identity, the register pending, mark B3).
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B3a (a)): the pure-partition identity cited to NielsenChuang2000, "exactly" dropped; the mark B3, the turnover statement and the closer kept; archived in REPO_DERIVATIONS 10.386.
% status (lead inbox 105): APPLIED WITH LABEL (exact identity, the register pending, mark B3).
```

### 10.386 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 8), beginning "\(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\) exactly"
Replaced text:
```latex
\(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\) exactly, so the sign
```
Replacement:
```latex
\(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\)~\cite{NielsenChuang2000}, so the sign
```

### 10.387 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 9), beginning "% status (lead inbox 105): APPLIED WITH LABEL (an entropy bound, the s"
Replaced text:
```latex
% status (lead inbox 105): APPLIED WITH LABEL (an entropy bound, the store still assumed, the cubic cut a pending one-way control, mark B1).
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B1b (a)): purity plus subadditivity cited to EisertCramerPlenio2010; eta(0), (R_t/R_0)^2, the store label and mark B1 kept; archived in REPO_DERIVATIONS 10.388.
% status (lead inbox 105): APPLIED WITH LABEL (an entropy bound, the store still assumed, the cubic cut a pending one-way control, mark B1).
```

### 10.388 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 10), beginning "the remaining modes form a complete orthogonal complement, purity and "
Replaced text:
```latex
the remaining modes form a complete orthogonal complement, purity and subadditivity give
```
Replacement:
```latex
the remaining modes form a complete orthogonal complement, purity and subadditivity~\cite{EisertCramerPlenio2010} give
```

### 10.389 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 11), beginning "% status (lead inbox 105): APPLIED WITH LABEL (an externally controlle"
Replaced text:
```latex
% status (lead inbox 105): APPLIED WITH LABEL (an externally controlled proposal, attainability unproved, mark B1c).
```
Replacement:
```latex
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part A B1c (b)): the unresolved-block clause shortened with a Ref. pointer, its claim kept; reviewer B's proposal statement and mark B1c kept; archived in REPO_DERIVATIONS 10.390.
% status (lead inbox 105): APPLIED WITH LABEL (an externally controlled proposal, attainability unproved, mark B1c).
```

### 10.390 Sec. VI: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Parts A and B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_06 pair 12), beginning "and an unresolved block costs at most its entropy, which does not gene"
Replaced text:
```latex
and an unresolved block costs at most its entropy, which does not generally vanish when its frequency width is narrowed \todo{B1c:
```
Replacement:
```latex
and an unresolved block costs at most its entropy, not reduced by narrowing its frequency width (Ref.~\cite{McKinneyCode2026}) \todo{B1c:
```

### 10.391 Appendix A: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_A2 pair 1), beginning "At a rational normal \(\mathbf b/|\mathbf b|\), \(\mathbf b\) primitiv"
Replaced text:
```latex
At a rational normal \(\mathbf b/|\mathbf b|\), \(\mathbf b\) primitive,
the Fourier transform along the lattice layers \(\mathbf b\cdot x=j\),
of cell area \(|\mathbf b|\), reduces the cut to half-chains of the
layer chain \(h_{\mathbf b}(k_\perp)\), so
\(\eta_{\boldsymbol n}=|\mathbf b|^{-1}\int_{\mathbb T^2}\frac{d^2k_\perp}{(2\pi)^2}S_{\rm half}[h_{\mathbf b}(k_\perp)]\),
\(S_{\rm half}\) the half-chain entropy. For \(\mathbf b=(1,0,0)\) this
is Eq.~\eqref{eq:dr-area-chain}, with \(S_{\rm half}=s_4(M_\perp)\) by
Ref.~\cite{Eisler2025}.
```
Replacement:
```latex
At a rational normal \(\mathbf b/|\mathbf b|\), \(\mathbf b\) primitive, the layer Fourier transform (Ref.~\cite{McKinneyCode2026}) reduces the cut to half-chains of the layer chain \(h_{\mathbf b}(k_\perp)\), \(\eta_{\boldsymbol n}=|\mathbf b|^{-1}\int_{\mathbb T^2}\frac{d^2k_\perp}{(2\pi)^2}S_{\rm half}[h_{\mathbf b}(k_\perp)]\) with \(S_{\rm half}\) the half-chain entropy, Eq.~\eqref{eq:dr-area-chain} for \(\mathbf b=(1,0,0)\) with \(S_{\rm half}=s_4(M_\perp)\) by Ref.~\cite{Eisler2025}.
```

### 10.392 Appendix A: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_A2 pair 2), beginning "and sends \(\bar v\) to \(-\bar v\). For fields of Lipschitz constant "
Replaced text:
```latex
and sends \(\bar v\) to \(-\bar v\). For fields of Lipschitz
constant at most \(C_1\epsilon\), \(h\) is within \(3C_1\epsilon\) in
norm of \(\bar N(X)h_0+\sum_i\bar v_i(X)Q_i\) (the Schur
test). For a cut
```
Replacement:
```latex
and sends \(\bar v\) to \(-\bar v\). For a cut
```

### 10.393 Appendix A: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_A2 pair 3), beginning "For every integer \(R\ge1\) and every \(\psi\), \begin{equation} \begi"
Replaced text:
```latex
For every integer \(R\ge1\) and every \(\psi\),
\begin{equation}
\begin{split}
 \|h\psi\|&\ge(\mathsf g_*-E_R-D_R)\|\psi\|,\\
 E_R&=C_1\epsilon[\sqrt3(\zeta+9/2)R+3],\\
 D_R&=\frac{3\pi(n^*+v^*)}{2R}.
\end{split}
\label{eq:B-gapcert}
\end{equation}
Positive \(\delta=\mathsf g_*-E_R-D_R\) establishes the gap
(Ref.~\cite{McKinneyCode2026}, a tensor cosine partition of half-width \(R\) comparing localized operators with frozen symbols, freezing costing \(E_R\) and partition commutators \(D_R\), with \(R=O(\epsilon^{-1/2})\)).
```
Replacement:
```latex
For every integer \(R\ge1\) and every \(\psi\), \(\|h\psi\|\ge(\mathsf g_*-E_R-D_R)\|\psi\|\), with \(E_R=C_1\epsilon[\sqrt3(\zeta+9/2)R+3]\) the freezing cost and \(D_R=3\pi(n^*+v^*)/(2R)\) the partition cost of a tensor cosine partition of half-width \(R=O(\epsilon^{-1/2})\) comparing localized operators with frozen symbols (the Schur-test and partition estimates in Ref.~\cite{McKinneyCode2026}), so positive \(\delta=\mathsf g_*-E_R-D_R\) establishes the gap.
```

### 10.394 Appendix A: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_A2 pair 4), beginning "once the right side of Eq.~\eqref{eq:B-gapcert} at these bounds and"
Replaced text:
```latex
once the right side of
Eq.~\eqref{eq:B-gapcert} at these bounds and
```
Replacement:
```latex
once \(\mathsf g_*-E_R-D_R\) at these bounds and
```

### 10.395 B: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_B pair 1), beginning "The last term vanishes by Euler's identity for the degree-one referenc"
Replaced text:
```latex
The last term vanishes by Euler's identity for the degree-one reference, \(\mathcal W[\lambda u]=\lambda\mathcal W[u]\) for \(\lambda>0\) giving \(\mathcal W=\langle u,r\rangle\) and \(\langle u,\mathcal W''\delta u\rangle=0\), so \(\delta E=\Tr(\bar h\,\delta C)\) and \(dE/dt=-i\Tr(\bar h[\bar h,C])=0\), the full derivation in Ref.~\cite{McKinneyCode2026}.
```
Replacement:
```latex
The last term vanishes by Euler's identity for the degree-one reference, so \(\delta E=\Tr(\bar h\,\delta C)\) and \(dE/dt=-i\Tr(\bar h[\bar h,C])=0\), the full derivation in Ref.~\cite{McKinneyCode2026}.
```

### 10.396 B: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_B pair 2), beginning "and vanishing mixed moments. For an even probability kernel \(k\) on \"
Replaced text:
```latex
and vanishing mixed moments. For an even probability kernel \(k\) on \(\mathbb R^3\) with per-axis
variance \(\sigma^2\) and vanishing mixed second moments, and
\(F\in C^2(\mathbb R^3)\) with bounded Hessian,
\(|(k*F)(x)-F(x)|\le\tfrac32\sigma^2\|\nabla^2F\|_\infty\),
by Taylor's formula and the kernel's second moments. The \(C^4\) expansion
```
Replacement:
```latex
and vanishing mixed moments, so for \(F\in C^2(\mathbb R^3)\) with bounded Hessian the second-moment Taylor bound \(|(k_{\ell,a}*F)(x)-F(x)|\le\tfrac32\sigma^2\|\nabla^2F\|_\infty\) holds (Ref.~\cite{McKinneyCode2026}). The \(C^4\) expansion
```

### 10.397 B: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_B pair 3), beginning "% provenance: the mass balance between trapping radius and shell from "
Replaced text:
```latex
% provenance: the mass balance between trapping radius and shell from FableAPI_closeF2_v1.md (md5 71937f98) with AstraAPI_closeA_v1.md (md5 eb335afb), lead inbox 94; plain-word map applied.
```
Replacement:
```latex
% provenance: the mass balance between trapping radius and shell from FableAPI_closeF2_v1.md (md5 71937f98) with AstraAPI_closeA_v1.md (md5 eb335afb), lead inbox 94; plain-word map applied.
% tightened (owner cite-not-repeat rule, lead GO 2026-11-04, BORROWF_TIGHTEN Part B Appendix B (b)): the residual expressions epsilon_M, epsilon_N, the explicit defect and the (N D)' anchoring identity to Ref. McKinneyCode2026; the flux identity, the defect on exact rows, the anchoring statement (D_b = E-dot beyond all sources), the shell balance and the trapping-radius form kept; archived in REPO_DERIVATIONS 10.398.
```

### 10.398 B: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B, owner cite-not-repeat rule, lead GO 2026-11-04, spec T_B pair 4), beginning "and \(\mathcal D=r^2vR_t/(GN)\). Where the rows carry residuals \(\eps"
Replaced text:
```latex
and \(\mathcal D=r^2vR_t/(GN)\). Where the rows carry residuals \(\epsilon_{\mathcal M}=\mathcal M'-4\pi r^2(\rho_\ell+wJ_{\ell r})\) and \(\epsilon_N=(\ln N)'+4\pi GrJ_{\ell r}/w\), the defect is \(\mathcal D=r^2vR_t/(GN)-2v\epsilon_{\mathcal M}+rv(1-w^2)\epsilon_N/G\). By \((N\mathcal D)'=4\pi r^2N(N\epsilon_E+v\epsilon_M)\), \(N\mathcal D\) at any radius is \(N_b\mathcal D_b\) at an outer boundary \(b\) minus the weighted defect between them, and beyond all sources
```
Replacement:
```latex
and \(\mathcal D=r^2vR_t/(GN)\) plus terms linear in the rows' residuals (their form and the anchoring identity in Ref.~\cite{McKinneyCode2026}), so that \(N\mathcal D\) at any radius is its value at an outer boundary \(b\) minus the weighted defect between them, and beyond all sources
```

### 10.399 Sec. VI: replaced text, verbatim (payment (a) on the emission block, option-2 reduction, lead GO 2026-11-04, spec T_PA_06 pair 1), beginning "are \(w_{k\pm}=\Gamma_ke^{\pm\pi\omega/\kappa_R}/[2\cosh(\pi\omega/\ka"
Replaced text:
```latex
are \(w_{k\pm}=\Gamma_ke^{\pm\pi\omega/\kappa_R}/[2\cosh(\pi\omega/\kappa_R)]\). For \(0\le\nu_{k\pm}\le1\),
\begin{equation}
\begin{gathered}
n_k(\omega,u)=w_{k+}\nu_{k+}+w_{k-}\nu_{k-}\le\Gamma_k(\omega),\\
w_{k-}=\frac{\Gamma_k(\omega)}{e^{2\pi\omega/\kappa_R}+1},
\end{gathered}
\label{eq:page-valve}
\end{equation}
so \(n_k\le\Gamma_k\). The pair \((\nu_+,\nu_-)=(0,1)\) is Hawking's occupation,
```
Replacement:
```latex
are \(w_{k\pm}=\Gamma_ke^{\pm\pi\omega/\kappa_R}/[2\cosh(\pi\omega/\kappa_R)]\), so that for \(0\le\nu_{k\pm}\le1\) the occupation is \(n_k(\omega,u)=w_{k+}\nu_{k+}+w_{k-}\nu_{k-}\le\Gamma_k(\omega)\), with \(w_{k-}=\Gamma_k(\omega)/(e^{2\pi\omega/\kappa_R}+1)\). The pair \((\nu_+,\nu_-)=(0,1)\) is Hawking's occupation,
```

### 10.400 Sec. VI: replaced text, verbatim (payment (a) on the emission block, option-2 reduction, lead GO 2026-11-04, spec T_PA_06 pair 2), beginning "Hawking's feed (the incoming vacuum's under Eq.~\eqref{eq:page-valve}'"
Replaced text:
```latex
Hawking's feed (the incoming vacuum's under Eq.~\eqref{eq:page-valve}'s assumptions),
```
Replacement:
```latex
Hawking's feed (the incoming vacuum's under the valve assumptions above),
```

### 10.401 Sec. VI: replaced text, verbatim (payment (a) on the emission block, option-2 reduction, lead GO 2026-11-04, spec T_PA_06 pair 3), beginning "The arrival clock from the stationary-exterior flight, \(T/r_s=40.8\)-"
Replaced text:
```latex
The arrival clock from the stationary-exterior flight, \(T/r_s=40.8\)--44.1 for \(\omega r_s=2\)--0.4 up to an undetermined \(O(r_s)\) matching term, is a consistency check, not an independently predicted jump, and evolving or dispersive histories require their actual packet transfer.
```
Replacement:
```latex
The arrival clock from the stationary-exterior flight (its \(T/r_s\) estimates in Ref.~\cite{McKinneyCode2026}) is a consistency check up to an undetermined \(O(r_s)\) matching term, not an independently predicted jump, and evolving or dispersive histories require their actual packet transfer.
```

### 10.402 Sec. VI: replaced text, verbatim (payment (a) on the emission block, option-2 reduction, lead GO 2026-11-04, spec T_PA_06 pair 4), beginning "What the delivery moved to Ref. McKinneyCode2026 is listed in the deli"
Replaced text:
```latex
What the delivery moved to Ref. McKinneyCode2026 is listed in the delivery file under "Moved to Ref.".
```
Replacement:
```latex
What the delivery moved to Ref. McKinneyCode2026 is listed in the delivery file under "Moved to Ref.". Payment (a) (the lead's option-2 reduction, GO 2026-11-04): the valve display set inline with its label eq:page-valve retired (the one reference now reads "the valve assumptions above"), and the arrival-clock flight numbers T/r_s = 40.8-44.1 for omega r_s = 2 to 0.4 moved to Ref. McKinneyCode2026 with the claim and the T/r_s clock kept; archived in REPO_DERIVATIONS 10.399-10.401.
```

### 10.403 Sec. II: replaced text, verbatim (page-budget cut list Tier C items 57, 58, 53 (FableAPI_pagebud_v1.md a3703d8a), lead GO 2026-11-04, spec TC_03 pair 1), beginning "Sec.~\ref{sec:model-gravity}. The clock and flow of Sec.~\ref{sec:mode"
Replaced text:
```latex
Sec.~\ref{sec:model-gravity}. The clock and flow of
Sec.~\ref{sec:model-dictionary} store no momentum on the sites
and, on flows with a potential, no energy. This section states
```
Replacement:
```latex
Sec.~\ref{sec:model-gravity}. This section states
```

### 10.404 Sec. II: replaced text, verbatim (page-budget cut list Tier C items 57, 58, 53 (FableAPI_pagebud_v1.md a3703d8a), lead GO 2026-11-04, spec TC_03 pair 2), beginning "On potential flows \(v=\nabla\phi\), including every spherical flow, i"
Replaced text:
```latex
On potential flows \(v=\nabla\phi\), including every
spherical flow, it
expresses at any clock the generator per local-clock tick on any
set of sites as a flux through its outgoing bonds built from
clock, flow, and their first derivatives there (Sec.~\ref{sec:model-dictionary}). Its density is
```
Replacement:
```latex
On potential flows \(v=\nabla\phi\), including every spherical flow, the fourth requirement of Sec.~\ref{sec:model-dictionary} applies to the generator per local-clock tick, whose density is
```

### 10.405 Sec. II: replaced text, verbatim (page-budget cut list Tier C items 57, 58, 53 (FableAPI_pagebud_v1.md a3703d8a), lead GO 2026-11-04, spec TC_03 pair 3), beginning "slices. Beyond the leading order each correction at every higher even "
Replaced text:
```latex
slices. Beyond the leading order each correction at every higher even order in the derivatives must contribute to the charge
\(\rho_\ell+w\cdot J_\ell\) the divergence of a local field on flows
with a potential at any clock: a constraint on the sources, not the density, not classified here.
```
Replacement:
```latex
slices.
% Tier C item 53 of the page-budget cut list, not printed (lead GO 2026-11-04, archived in REPO_DERIVATIONS 10.405): Beyond the leading order each correction at every higher even order in the derivatives must contribute to the charge rho_l + w . J_l the divergence of a local field on flows with a potential at any clock: a constraint on the sources, not the density, not classified here.
```

### 10.406 Sec. III: replaced text, verbatim (page-budget cut list Tier C item 60 (FableAPI_pagebud_v1.md a3703d8a), lead GO 2026-11-04, spec TC_04 pair 1), beginning "(centre Hamiltonian in Ref.~\cite{McKinneyCode2026}). Sec.~\ref{sec:st"
Replaced text:
```latex
(centre Hamiltonian in Ref.~\cite{McKinneyCode2026}). Sec.~\ref{sec:star}'s elementary dust orbitals have no such factor. So the cone
```
Replacement:
```latex
(centre Hamiltonian in Ref.~\cite{McKinneyCode2026}). So the cone
```

### 10.407 Sec. VI: replaced text, verbatim (page-budget cut list Tier C item 54 (FableAPI_pagebud_v1.md a3703d8a), lead GO 2026-11-04, spec TC_06 pair 1), beginning "at fixed \(r_sm\) or \(r_s/a\) leaves it. The computed \(\eta_E=0.923\"
Replaced text:
```latex
at fixed \(r_sm\) or \(r_s/a\) leaves it.
The computed \(\eta_E=0.923\) equals
\(\langle\sqrt{1-2G\mathcal M(r)/r}\rangle\) only for pure
coordinate-rest dust carrying all charge, an unestablished preparation
condition, so the equality is an interpretation, not an input.
```
Replacement:
```latex
at fixed \(r_sm\) or \(r_s/a\) leaves it.
% Tier C item 54 of the page-budget cut list, not printed (lead GO 2026-11-04, archived in REPO_DERIVATIONS 10.407): The computed eta_E = 0.923 equals <sqrt(1 - 2 G M(r)/r)> only for pure coordinate-rest dust carrying all charge, an unestablished preparation condition, so the equality is an interpretation, not an input.
```

### 10.408 Sec. VII: replaced text, verbatim (tightening list BORROWF_TIGHTEN_20261104 Part B Sec. VII A, owner cite-not-repeat rule, lead GO 2026-11-04, writer stopped per STATUS 639a9b3776a, spec T_08 pair 1), beginning "adds nearly \(\ln2\) to the radiation entropy however permeable the su"
Replaced text:
```latex
adds nearly \(\ln2\) to the radiation entropy however permeable the surface (Ref.~\cite{McKinneyCode2026}), the monogamy behind the firewall argument~\cite{AMPS2013}.
```
Replacement:
```latex
adds nearly \(\ln2\) to the radiation entropy however permeable the surface~\cite{Mathur2009} (Ref.~\cite{McKinneyCode2026}), the monogamy behind the firewall argument~\cite{AMPS2013}.
```

### 10.409 Sec. III: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_04 pair 1), beginning "the Painlev\'e-Gullstrand form of the Schwarzschild"
Replaced text:
```latex
the Painlev\'e-Gullstrand form of the Schwarzschild
```
Replacement:
```latex
% provenance: lead inbox 108, FableAPI_embedRF_v1.md (md5 04a8f94b) text A (exterior scope): the Schwarzschild result's scope clause ("where that exterior contains r_s = 2GE") and the computed-collapse qualification placed in the result sentence below, in place; the abstract carries the same clause.
the Painlev\'e-Gullstrand form of the Schwarzschild
```

### 10.410 Sec. III: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_04 pair 2), beginning "and the trapping radius \(r_s=2GE\), the temperature, and the first la"
Replaced text:
```latex
and the trapping radius \(r_s=2GE\), the temperature, and the first law below are all read from this one solution.
```
Replacement:
```latex
and where that exterior contains \(r_s=2GE\) the trapping radius, the temperature, and the first law below are all read from this one solution, properties of the exterior that do not by themselves establish a vacuum trapping radius or an exponential segment on the computed collapse.
```

### 10.411 Sec. IV: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_05 pair 1), beginning "The cold formation comparison case is a pressureless cloud released fr"
Replaced text:
```latex
The cold formation comparison case is a pressureless cloud released from coordinate rest at \(4r_s\) with \(2GE=59.05a\) and no shell crossing. It reaches \(R=2GE\) at \(T/r_s=13.22\), where \(t_s\) diverges.
```
Replacement:
```latex
% provenance: lead inbox 108, FableAPI_embedRF_v1.md (md5 04a8f94b) text D (crossing and lattice-stage clocks), in place of the two comparison-case sentences; the optional closed form of T_c not printed.
The cold formation comparison case is a pressureless outer boundary released from coordinate rest at \(4r_s\) outside a constant charge of radius \(2GE=59.05a\), with no shell crossing. It crosses that radius at \(T/r_s=13.22\), where \(t_s\) diverges, a comparison clock and not a trajectory of either cloud, since coordinate rest is not imposed on the computed preparation, and it carries to no other size. The marginally bound dust comparison solution, Eq.~\eqref{eq:star-dust-control}, has its own crossing clock and reaches the filling radius \(R_\pi\) a time \(\tfrac23r_s[1-(R_\pi/r_s)^{3/2}]\) after it, and since \(R_\pi\) is a comparison radius and not an observed endpoint, these clocks establish neither the computed core's arrival there nor any duration of a vacuum trapping radius. Whether outward-moving matter reaches the trapping radius in a controlled astrophysical limit is open.
```

### 10.412 Sec. IV: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_05 pair 2), beginning "Thus \(\mathcal M(R_t)=R_t/(2G)\) leaves 4 to 5 percent of \(E\) outsi"
Replaced text:
```latex
Thus \(\mathcal M(R_t)=R_t/(2G)\) leaves 4 to 5 percent of \(E\)
outside. Both readings are now compared with the dust solution and tested against the width.
```
Replacement:
```latex
% provenance: lead inbox 108 and 109, FableAPI_embedRF_v1.md (md5 04a8f94b) text B (the stored-trapping-radius paragraph, authoritative for every statement about the trapping radius and f_ext) with the solve root audit (table 9c3cec12): the size-64 and size-128 histories behind the f_ext readouts are one-field (P64_s4/s8, S4_128_s8/s16, the only two-field history being C_M1nl23), so the readouts stay inside the todo with the history named as one-field; the clock integral for ln N_R to Ref. McKinneyCode2026 (its symbol is not defined in the paper); plain-word map (root -> trapping radius).
Thus \(f_{\rm ext}=1-R_t/(2GE)\), which the rows identify with the net signed charge outside the trapping radius, sea departure and reference response included and not a particle fraction, is 0.04 to 0.05 at \(T/r_s=15\), while a slice without a trapping radius has no \(f_{\rm ext}\) and its source support reaches \(2GE\) \todo{the readouts of the stored histories at later clocks, on the one-field size-64 history with certified simple trapping radii \(f_{\rm ext}=0.065\) at \(T/r_s=15\), none at 17 to 18.5, and 0.31 to 0.34 at 20 to 40, about plus or minus 0.03 from the charge-identity residual, print only with the history, reference law, and copy weights named and the trapping radii audited}. None of these readouts establishes a source-free trapping radius or local clearance at any clock. At every differentiable trapping radius the rows give \(\kappa_R=N_R\beta_R/(2R)\) with \(\beta_R=1-2G\mathcal M'_R=1-8\pi GR^2(\rho_\ell-J_{\ell r})_R\), so \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\), and the clock at the trapping radius, fixed by the weighted current outside it (the integral in Ref.~\cite{McKinneyCode2026}), lies below one when that current lags the inward flow and above one when it leads. A simple outer trapping radius has \(\kappa_R>0\) and \(\beta_R>0\), and values of \(\kappa_R/\kappa_H\) above one are not excluded \todo{on the stored histories, certify the outer trapping radii with sign changes and no farther trapping radii, bound the slopes and the three evaluations of \(\beta_R\) with interpolation, solve, source, and tail errors, and decompose the exterior charge into heavy, propagated-sea, light, and reference parts, the reported range \(-0.31\) to 1.04 of \(\kappa_R/\kappa_H\) waiting on this audit}. Both readings are now compared with the dust solution and tested against the width.
```

### 10.413 Sec. IV: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_05 pair 3), beginning "The size and mass ladders, the families of runs varying"
Replaced text:
```latex
The size and mass ladders, the families of runs varying
```
Replacement:
```latex
% provenance: lead inbox 111, ISO128_T14_PROVENANCE_v1.md (md5 728536f0, solve): the stored size-128 history has unconverged records through its formation passage T/r_s 11.6 to 12.9 and around T 14 (E falls about 13 percent non-monotonically, the T12 resume point inside the passage), so its formation-window reading below is labelled unconverged; numbers from that run outside the window stay; no size-128 formation time or table column is printed elsewhere.
The size and mass ladders, the families of runs varying
```

### 10.414 Sec. IV: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_05 pair 4), beginning "the 128 pair passes only at formation, and the size-256 formation"
Replaced text:
```latex
the 128 pair passes only at formation, and the size-256 formation
```
Replacement:
```latex
the 128 pair passes only at formation (unconverged records there), and the size-256 formation
```

### 10.415 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 1), beginning "the 50 percent segment lasts \(2.76r_s\) with 0.89 e-folds, and no pac"
Replaced text:
```latex
the 50 percent segment lasts \(2.76r_s\) with 0.89 e-folds, and no packet resolving its own frequency fits inside it.
```
Replacement:
```latex
the 50 percent segment lasts \(2.76r_s\) with 0.89 e-folds, and no packet resolving its own frequency fits inside it. On the dust comparison solution the exact map expands in the received exponential of the surface's approach (coefficients in Ref.~\cite{McKinneyCode2026}), and its first-order term is a fractional-linear deformation of the exponential, which leaves the chiral vacuum two-point function unchanged, so for the dust solution's conformal companion of Sec.~\ref{sec:page} no outgoing quantity changes at first order and its stress flux falls below the thermal value only at second order in that exponential, a conformal stress result and not a particle rate or a greybody-filtered luminosity. At one solar mass the segment ends \(114r_s\) after the surface's crossing, about nine thermal periods of retarded time, and the comparison rates of Eq.~\eqref{eq:sc-luminosity} integrated over it give \(3.5\times10^{-77}\) of \(E\), about 1.8 quanta by the reported but unverified number rate, and 12 nats of marginal entropy. These coefficients describe the comparison solution's propagation and its companion's stress and fix neither the occupation of the outgoing branch nor its supply, which need the evolving incident covariance at the trapping radius and a once-counted output basis, the received-clock reparametrization being no rate multiplier.
```

### 10.416 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 2), beginning "The received redshift grows at Hawking's exponent until dispersion end"
Replaced text:
```latex
The received redshift grows at Hawking's exponent until dispersion ends the long-wave description:
```
Replacement:
```latex
% provenance: lead inbox 108, FableAPI_embedRF_v1.md (md5 04a8f94b) texts C (valve and luminosity comparisons) and E (Page-model assumptions and work), and FableAPI_nuRF_v1.md (md5 26573649) supply paragraphs for Secs. scaling-rays (above, its expansion coefficients, segment thresholds and closed forms to Ref. McKinneyCode2026) and scaling-far (below, the g_D ray integral and the axial-symbol expansions to the Ref.); nuRF's sentence placing the trapping radius "clear of heavy matter only near T/r_s = 15 ... embedded in the outgoing fountain afterwards" dropped per the lead (fountain chronology not established), embedRF text B's wording used instead (Sec. IV B); in place at the paragraph ends, no sentence compressed.
The received redshift grows at Hawking's exponent until dispersion ends the long-wave description:
```

### 10.417 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 3), beginning "At a smooth, simple, held trapping radius the radial massless operator"
Replaced text:
```latex
At a smooth, simple, held trapping radius the radial massless operator's near-surface limit gives Fermi-shaped two-channel transmission at \(\kappa_R/(2\pi)\), a local scattering result that fixes neither incoming occupation, sustained supply, escape yield on evolving fields, nor heavy-information transfer.
```
Replacement:
```latex
At a smooth, simple, held trapping radius the radial massless operator's near-surface limit gives Fermi-shaped two-channel transmission at \(\kappa/2\pi\), \(\kappa\) the slope of the back-averaged fields the operator couples to, which the fine-grid slope \(\kappa_R=N_R\beta_R/(2R)\) of Sec.~\ref{sec:star} replaces only with the back-averaging shift controlled, a local scattering result that fixes neither incoming occupation, sustained supply, escape yield on evolving fields, nor heavy-information transfer. A connected vacuum exterior containing the trapping radius gives \(\kappa_R=\kappa_H\), equality of the two slopes not by itself establishing vacuum, and the shell-flow comparisons at the trapping-radius slope and at \(T_H(E)\) are comparison cases, neither a measured emission temperature of the evolving object. On evolving fields the received factor is the \(g_D\) of Sec.~\ref{sec:star}, its ray-integral form and the static-receiver conversion in Ref.~\cite{McKinneyCode2026}, the carrier and group velocity at the dispersive cutoff of a held trapping radius are local kinematics, not a received redshift and not a redshift floor, and the slope at a simple trapping radius is \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\), equal to one only on a vacuum shell with unit clock, so on the stored histories the slope and the received map are read from \(N_R\), \(\beta_R\), \(f_{\rm ext}\) and the complete flight through the evolving fields, not from the comparison constants.
```

### 10.418 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 4), beginning "On a differentiable interval with a simple trapping radius, with \(E_R"
Replaced text:
```latex
On a differentiable
interval with a simple trapping radius, with \(E_R=R/(2G)\) and
\(2R\kappa_R=N_R(1-2G\mathcal M'_R)\),
Eq.~\eqref{eq:page-first-law-rows} reads
\(\dot E=2R\kappa_R\dot E_R+P_{\rm ext}\), \(P_{\rm ext}\) the rate
form of \(W_{\rm ext}\) with the reference response included, so at
\(\dot E=0\) the exterior work vanishes exactly when the trapping radius stands still, motion of the trapping radius implies nonzero total exterior work, and a
receding trapping radius needs positive exterior work where
\(1-2G\mathcal M'_R>0\).
```
Replacement:
```latex
On a differentiable interval with a simple trapping radius, with \(E_R=R/(2G)\) and \(2R\kappa_R=N_R\beta_R\), Eq.~\eqref{eq:page-first-law-rows} reads \(\dot E=N_R\beta_R\dot E_R+P_{\rm ext}\), \(P_{\rm ext}\) the rate form of \(W_{\rm ext}\) with the reference response included, so at \(\dot E=0\) the exterior work vanishes exactly when the trapping radius stands still, motion of the trapping radius implies nonzero total exterior work, and with the boundary charge conserved and \(dE_R/du=-L\) a receding trapping radius has \(P_{\rm ext}=N_R\beta_RL\,du/dT>0\) where \(\beta_R>0\), the balance's assignment of the radiated energy to the sources outside the trapping radius, controlled by the shell remainder with storage, reference response, and the moving trapping radius's sweep included.
```

### 10.419 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 5), beginning "Evaluated, not measured, at one solar mass with \(q=1\) and \(m=0\) un"
Replaced text:
```latex
Evaluated, not measured, at one solar mass with \(q=1\) and \(m=0\) unless stated: the validity scale \(\alpha_d\) is \(2.80\times10^{-13}\), against 0.315 at the computed size, the formal axial correction \(\delta T/T_H\) is \(1.39\times10^{-76}\), and if Eq.~\eqref{eq:sc-luminosity} holds throughout evaporation, \(t_*/r_s\) is \(1.06\times10^{78}\), \(t_*=3.3\times10^{65}\) years (\(m/T_H\), the \(\chi\) scaling exponents, and the size-64 values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
% provenance: lead inbox 108, FableAPI_embedRF_v1.md (md5 04a8f94b) text G (one-solar-mass conclusion), in place.
Evaluated, not measured, for the constant-charge vacuum comparison at one solar mass with \(q=1\) and \(m=0\) unless stated: the validity scale \(\alpha_d\) is \(2.80\times10^{-13}\), against 0.315 at the computed size, the \(10^{-76}\)-level shift \(\delta T/T_H=1.39\times10^{-76}\) is the formal axial fixed-profile correction under its matching condition, not an established spherical-object temperature shift, and the evaporation scale \(t_*/r_s=1.06\times10^{78}\), \(t_*=3.3\times10^{65}\) years, follows only if the assumed luminosity of Eq.~\eqref{eq:sc-luminosity} persists, no sustained emission law or lifetime being established for the computed object (\(m/T_H\), the \(\chi\) scaling exponents, and the size-64 values in Ref.~\cite{McKinneyCode2026}).
```

### 10.420 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 6), beginning "AstraAPI_pccmpAB_v1.md (md5 ea3685a0), the two routes disagree beyond "
Replaced text:
```latex
AstraAPI_pccmpAB_v1.md (md5 ea3685a0), the two routes disagree beyond their floors on most channels,
```
Replacement:
```latex
AstraAPI_pccmpAB_v1.md (md5 ea3685a0), superseded by lead inbox 110 and 112: the two independent calculations agree on all 69 common channel-clock pairs of the 14 ruled channels (occupations identical, largest E-bar difference 6.0e-9 nats, the 16k-weighted subtotal at T/r_s 100 differing by 1.6e-6 nats against a numerical floor of 6.7e-5, correlation matrices ||chi_A - chi_B||_F = 5.6e-11 at k = 1, 8.8e-13 relative, eigenvalues to 3.4e-14), the earlier 0.038-nat gap from the second calculation's old static-projector construction, no pending item; earlier reading, superseded: the two routes disagree beyond their floors on most channels,
```

### 10.421 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 7), beginning "its two calculations reading 8.790 and 8.744 nats on the common channe"
Replaced text:
```latex
its two calculations reading 8.790 and 8.744 nats on the common channels at 100 with agreement between them not established.
```
Replacement:
```latex
its two independent calculations agreeing on all 69 common channel-clock pairs of the 14 ruled channels at 100, occupations identical and subtotals within \(1.6\times10^{-6}\) nats against a floor of \(6.7\times10^{-5}\), a finite-time diagnostic on the first calculation's operator as the declared convention.
```

### 10.422 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 8), beginning "the vacuum exterior's \(R=2GE_R\) and \(\kappa_R=1/(2R)\), and the red"
Replaced text:
```latex
the vacuum exterior's \(R=2GE_R\) and \(\kappa_R=1/(2R)\), and the reduced balance
```
Replacement:
```latex
the vacuum exterior's \(R=2GE_R\) and \(\kappa_R=1/(2R)\), that is the vacuum comparison temperature \(T_H(E_R)=1/(4\pi R)\) and the channel coefficients of the thermally supplied vacuum profile, properties of the exterior solution whose identification with the evolving object requires the source, transport, and shell-remainder calculation below and which establish no vacuum trapping radius on the computed collapse, and the reduced balance
```

### 10.423 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 9), beginning "and a variable-supply lifetime fraction exists only with a finite evap"
Replaced text:
```latex
and a variable-supply lifetime fraction exists only with a finite evaporation time \(t_*\). The cut share \(f\) is the one place
```
Replacement:
```latex
and a variable-supply lifetime fraction exists only with a finite evaporation time \(t_*\), the area fraction and the height holding independently of the luminosity normalization while the time fraction uses \(L=\mathcal L/R^2\) and changes with the rate law. The cut share \(f\) is the one place
```

### 10.424 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 10), beginning "simple fine-grid trapping radius, \(E_R=R/(2G)\), \(\beta_R=1-2G\mathc"
Replaced text:
```latex
simple fine-grid trapping radius, \(E_R=R/(2G)\), \(\beta_R=1-2G\mathcal M'_R>0\),
and \(N_R\beta_R\dot E_R=\Xi_R-\Xi_R^{\rm ref}\). Exterior work is
\(\dot E-N_R\beta_R\dot E_R\), so conserved charge and recession do
not imply zero work.
```
Replacement:
```latex
simple fine-grid trapping radius, \(E_R=R/(2G)\), \(\beta_R>0\), and \(N_R\beta_R\dot E_R=\Xi_R-\Xi_R^{\rm ref}\). Exterior work is \(\dot E-N_R\beta_R\dot E_R\), so conserved charge and recession do not imply zero work, clearance of heavy matter from the trapping radius does not by itself set \(N_R\beta_R=1\), and no zero-exterior-work condition is imposed.
```

### 10.425 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_06 pair 11), beginning "\todo{a trapping radius clear of heavy matter with the shell remainder"
Replaced text:
```latex
\todo{a trapping radius clear of heavy matter with the shell remainder bounded, other-clock reference terms, energy-supplied complete release, and uniform decoding and response-error bounds}
```
Replacement:
```latex
\todo{establish source clearance at the trapping radius or retain the measured \(N_R\), \(\beta_R\), and \(f_{\rm ext}\), bound the heavy, light, and reference shell remainder uniformly in the scaling family, and match the received no-return output to the assumed emission and release laws}
```

### 10.426 Sec. V: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_07 pair 1), beginning "Held fields bound what can leave but fix no rate:"
Replaced text:
```latex
Held fields bound what can leave but fix no rate:
```
Replacement:
```latex
% provenance: lead inbox 108, FableAPI_nuRF_v1.md (md5 26573649) page supply paragraph at this paragraph's end (its coherent two-port occupation formula to Ref. McKinneyCode2026) and FableAPI_embedRF_v1.md (md5 04a8f94b) text F in the shell-entropy sentence below; text F's greybody and assumed-store sentences are already carried by Sec. V B's "an assumed supply not established on the collapse" and "curves of the comparison case under assumed ensembles, not of the object".
Held fields bound what can leave but fix no rate:
```

### 10.427 Sec. V: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_07 pair 2), beginning "so finite late flux alone proves no sustained supply (bounds in Ref.~\"
Replaced text:
```latex
so finite late flux alone proves no sustained supply (bounds in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
so finite late flux alone proves no sustained supply (bounds in Ref.~\cite{McKinneyCode2026}). The Hawking comparison case below is a supplied exterior, its inner port fed thermally at \(T_H\), so its rates are a comparison and not a result about the object's supply: for a stationary matched two-port channel at a held trapping radius the outgoing occupation with an empty outgoing branch is at most \(\Gamma_kF\), \(F=[e^{2\pi\omega/\kappa_R}+1]^{-1}\) the local conversion probability, the comparison case at the same \(\kappa_R\), and with that branch filled it exceeds it (the coherent two-port form in Ref.~\cite{McKinneyCode2026}), so the comparison bounds the supply only given the incident state, which the computed collapse does not fix once formation light has passed. On evolving fields the scattering map mixes frequencies and the once-counted output needs its basis, and the exponential segment of Sec.~\ref{sec:scaling-rays} fixes the temperature of what is converted, not the amount.
```

### 10.428 Sec. V: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_07 pair 3), beginning "The unit-flux transmission \(\Gamma_k(\omega)\) of this barrier, a flu"
Replaced text:
```latex
The unit-flux transmission \(\Gamma_k(\omega)\) of this barrier, a flux ratio and so the same in either time, for both partners and for both signs of \(\kappa=\pm k\), is the continuum comparison transmission,
```
Replacement:
```latex
The unit-flux transmission \(\Gamma_k(\omega)\) of this barrier, a ratio of asymptotic fluxes and so the same in either time, the same for the two partners and for both signs of \(\kappa=\pm k\), is the continuum comparison transmission,
```

### 10.429 Sec. V: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_07 pair 4), beginning "The companion is one chiral Dirac field on a ray circle, the unfolded "
Replaced text:
```latex
The companion is one chiral Dirac field on a ray circle, the unfolded massless channel (a scaling description
```
Replacement:
```latex
The companion is one chiral Dirac field on a ray circle, the massless channel without its centrifugal term unfolded through the regular centre (a scaling description
```

### 10.430 Sec. V: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_07 pair 5), beginning "The entropy carried per unit of shell energy is 0.02 to 0.38 over 40--"
Replaced text:
```latex
The entropy carried per unit of shell energy is 0.02 to 0.38 over 40--100 at both cuts (definition and channel sets in Ref.~\cite{McKinneyCode2026}), far below \(\mathcal R=1.64\), and classifies neither emitted entropy nor thermality.
```
Replacement:
```latex
The entropy carried per unit of shell energy is 0.02 to 0.38 over 40--100 at both cuts (definition and channel sets in Ref.~\cite{McKinneyCode2026}), a shell diagnostic far below \(\mathcal R=1.64\) and not the no-return output ratio \(\mathcal R\), which classifies neither emitted entropy nor thermality and fixes no alternative maximum.
```

### 10.431 abstract: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main pair 1), beginning "Outside a spherical positive-energy source the constraints give exactl"
Replaced text:
```latex
Outside a spherical positive-energy source the constraints give exactly Schwarzschild's vacuum and trapping surface, and its ray map's exponential segment carries Hawking's temperature.
```
Replacement:
```latex
Outside a spherical positive-energy source the constraints give exactly Schwarzschild's vacuum, a trapping surface where that vacuum reaches \(2GE\), and an exponential ray-map segment carrying Hawking's temperature.
```

### 10.432 abstract: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main pair 2), beginning "is a flux of these fields through the bonds leaving it,"
Replaced text:
```latex
is a flux of these fields through the bonds leaving it,
```
Replacement:
```latex
is a flux of these fields through its outgoing bonds,
```

### 10.433 abstract: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main pair 3), beginning "forms an entanglement star with a trapping surface inside the matter."
Replaced text:
```latex
forms an entanglement star with a trapping surface inside the matter.
```
Replacement:
```latex
forms an entanglement star, its trapping surface inside the matter.
```

### 10.434 abstract: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main pair 4), beginning "the radiation entropy follows a Page curve peaking"
Replaced text:
```latex
the radiation entropy follows a Page curve peaking
```
Replacement:
```latex
its entropy follows a Page curve peaking
```

### 10.435 abstract: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main pair 5), beginning "% "the Page curve's four laws are stipulated, not derived" (Sec. VI D)"
Replaced text:
```latex
% "the Page curve's four laws are stipulated, not derived" (Sec. VI D)
```
Replacement:
```latex
% "the Page curve's four laws are stipulated, not derived" (Sec. VI D)
% lead inbox 108 (FableAPI_embedRF_v1.md md5 04a8f94b, text A abstract form): "a trapping surface where that vacuum reaches 2GE" in the Schwarzschild sentence; three words of the abstract rephrased (its outgoing bonds, its trapping surface, its entropy) to hold the 300-word count.
```

### 10.436 main2: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main2 pair 1), beginning "measured against the Dirac sea the collapse prepares,"
Replaced text:
```latex
measured against the Dirac sea the collapse prepares,
```
Replacement:
```latex
measured against the collapse's own Dirac sea,
```

### 10.437 main3: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108_main3 pair 1), beginning "as Gauss's law reads charge from electric flux."
Replaced text:
```latex
as Gauss's law reads charge from electric flux.
```
Replacement:
```latex
as Gauss's law reads charge from flux.
```

### 10.438 Sec. IV: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108b_05 pair 1), beginning "The marginally bound dust comparison solution, Eq.~\eqref{eq:star-dust"
Replaced text:
```latex
The marginally bound dust comparison solution, Eq.~\eqref{eq:star-dust-control}, has its own crossing clock and reaches the filling radius \(R_\pi\) a time \(\tfrac23r_s[1-(R_\pi/r_s)^{3/2}]\) after it, and since \(R_\pi\) is a comparison radius and not an observed endpoint, these clocks establish neither the computed core's arrival there nor any duration of a vacuum trapping radius.
```
Replacement:
```latex
The dust comparison solution has its own crossing clock (its arrival at the filling radius \(R_\pi\) in Ref.~\cite{McKinneyCode2026}), and since \(R_\pi\) is a comparison radius and not an observed endpoint, these clocks establish neither the computed core's arrival there nor any duration of a vacuum trapping radius.
```

### 10.439 Sec. IV: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108b_05 pair 2), beginning "At every differentiable trapping radius the rows give \(\kappa_R=N_R\b"
Replaced text:
```latex
At every differentiable trapping radius the rows give \(\kappa_R=N_R\beta_R/(2R)\) with \(\beta_R=1-2G\mathcal M'_R=1-8\pi GR^2(\rho_\ell-J_{\ell r})_R\), so \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\), and the clock at the trapping radius, fixed by the weighted current outside it (the integral in Ref.~\cite{McKinneyCode2026}), lies below one when that current lags the inward flow and above one when it leads. A simple outer trapping radius has \(\kappa_R>0\) and \(\beta_R>0\), and values of \(\kappa_R/\kappa_H\) above one are not excluded
```
Replacement:
```latex
At every differentiable trapping radius the rows give \(\kappa_R=N_R\beta_R/(2R)\), \(\beta_R=1-2G\mathcal M'_R\), so \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\) (the clock and slope identities in Ref.~\cite{McKinneyCode2026}), and a simple outer trapping radius has \(\kappa_R>0\) and \(\beta_R>0\), values of \(\kappa_R/\kappa_H\) above one not being excluded
```

### 10.440 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108b_06 pair 1), beginning "On the dust comparison solution the exact map expands in the received "
Replaced text:
```latex
On the dust comparison solution the exact map expands in the received exponential of the surface's approach (coefficients in Ref.~\cite{McKinneyCode2026}), and its first-order term is a fractional-linear deformation of the exponential, which leaves the chiral vacuum two-point function unchanged, so for the dust solution's conformal companion of Sec.~\ref{sec:page} no outgoing quantity changes at first order and its stress flux falls below the thermal value only at second order in that exponential, a conformal stress result and not a particle rate or a greybody-filtered luminosity. At one solar mass the segment ends \(114r_s\) after the surface's crossing, about nine thermal periods of retarded time, and the comparison rates of Eq.~\eqref{eq:sc-luminosity} integrated over it give \(3.5\times10^{-77}\) of \(E\), about 1.8 quanta by the reported but unverified number rate, and 12 nats of marginal entropy. These coefficients describe the comparison solution's propagation and its companion's stress and fix neither the occupation of the outgoing branch nor its supply, which need the evolving incident covariance at the trapping radius and a once-counted output basis, the received-clock reparametrization being no rate multiplier.
```
Replacement:
```latex
On the dust comparison solution the exact map's first-order correction to the received exponential is a fractional-linear deformation that leaves the chiral vacuum two-point function unchanged, so the conformal companion's stress flux departs from the thermal value only at second order (the coefficients, the segment thresholds, and the one-solar-mass segment in Ref.~\cite{McKinneyCode2026}), and these coefficients fix neither the occupation of the outgoing branch nor its supply, which need the evolving incident covariance at the trapping radius and a once-counted output basis.
```

### 10.441 Sec. VI: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108b_06 pair 2), beginning "On evolving fields the received factor is the \(g_D\) of Sec.~\ref{sec"
Replaced text:
```latex
On evolving fields the received factor is the \(g_D\) of Sec.~\ref{sec:star}, its ray-integral form and the static-receiver conversion in Ref.~\cite{McKinneyCode2026}, the carrier and group velocity at the dispersive cutoff of a held trapping radius are local kinematics, not a received redshift and not a redshift floor, and the slope at a simple trapping radius is \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\), equal to one only on a vacuum shell with unit clock, so on the stored histories the slope and the received map are read from \(N_R\), \(\beta_R\), \(f_{\rm ext}\) and the complete flight through the evolving fields, not from the comparison constants.
```
Replacement:
```latex
Since \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\) equals one only on a vacuum shell with unit clock, on the stored histories the slope and the received map are read from \(N_R\), \(\beta_R\), \(f_{\rm ext}\) and the complete flight through the evolving fields (the received factor \(g_D\) of Sec.~\ref{sec:star} and the dispersive-cutoff kinematics in Ref.~\cite{McKinneyCode2026}), not from the comparison constants.
```

### 10.442 Sec. V: replaced text, verbatim (lead inbox 108 to 112: embedRF texts A to H (FableAPI_embedRF_v1.md 04a8f94b) with the inbox 109 root audit, nuRF supply paragraphs (FableAPI_nuRF_v1.md 26573649, fountain sentence dropped), the C1 two-calculation agreement (inbox 110, 112), the size-128 unconverged-records label (inbox 111); second pass: the appended supply mechanics trimmed to their conclusions with the mechanics in Ref. McKinneyCode2026 (owner selection rule), spec T_108b_07 pair 1), beginning "The Hawking comparison case below is a supplied exterior, its inner po"
Replaced text:
```latex
The Hawking comparison case below is a supplied exterior, its inner port fed thermally at \(T_H\), so its rates are a comparison and not a result about the object's supply: for a stationary matched two-port channel at a held trapping radius the outgoing occupation with an empty outgoing branch is at most \(\Gamma_kF\), \(F=[e^{2\pi\omega/\kappa_R}+1]^{-1}\) the local conversion probability, the comparison case at the same \(\kappa_R\), and with that branch filled it exceeds it (the coherent two-port form in Ref.~\cite{McKinneyCode2026}), so the comparison bounds the supply only given the incident state, which the computed collapse does not fix once formation light has passed. On evolving fields the scattering map mixes frequencies and the once-counted output needs its basis, and the exponential segment of Sec.~\ref{sec:scaling-rays} fixes the temperature of what is converted, not the amount.
```
Replacement:
```latex
The Hawking comparison case below is a supplied exterior, its inner port fed thermally at \(T_H\), so its rates are a comparison and not a result about the object's supply: a stationary two-port channel at a held trapping radius gives at most the comparison occupation with an empty outgoing branch and more with that branch filled (the coherent two-port form in Ref.~\cite{McKinneyCode2026}), so the comparison bounds the supply only given the incident state, which the computed collapse does not fix once formation light has passed, and the exponential segment of Sec.~\ref{sec:scaling-rays} fixes the temperature of what is converted, not the amount.
```

### 10.443 abstract: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_main pair 1), beginning "We study one Dirac fermion on a fixed cubic lattice, assuming no light"
Replaced text:
```latex
We study one Dirac fermion on a fixed cubic lattice, assuming no light cone, vacuum, or metric.
```
Replacement:
```latex
We study many particles of a Dirac fermion species on a fixed cubic lattice, assuming no light cone, vacuum, or metric.
```

### 10.444 abstract: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_main pair 2), beginning "a trapping surface where that vacuum reaches \(2GE\),"
Replaced text:
```latex
a trapping surface where that vacuum reaches \(2GE\),
```
Replacement:
```latex
a trapping surface where it reaches \(2GE\),
```

### 10.445 abstract: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_main pair 3), beginning "and modes prepared inside later reach the exterior."
Replaced text:
```latex
and modes prepared inside later reach the exterior.
```
Replacement:
```latex
and modes prepared inside reach the exterior.
```

### 10.446 abstract: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_main pair 4), beginning "a dynamical singularity is failure of unique joint continuation of sta"
Replaced text:
```latex
a dynamical singularity is failure of unique joint continuation of state, clock, and flow.
```
Replacement:
```latex
a dynamical singularity is failure of unique continuation of state, clock, and flow.
```

### 10.447 abstract: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_main pair 5), beginning "% lead inbox 108 (FableAPI_embedRF_v1.md md5 04a8f94b, text A abstract"
Replaced text:
```latex
% lead inbox 108 (FableAPI_embedRF_v1.md md5 04a8f94b, text A abstract form):
```
Replacement:
```latex
% owner comment cmw0j97q2855c ("We study one Dirac fermion ... It's not just one"): the species is one, the particles many; abstract "many particles of a Dirac fermion species", Secs. I, II (prose) and VII "a single Dirac fermion field", the first declaration's bold text "a single species of Dirac fermion" (only "one" changed inside the owner-approved bold); three word-neutral rephrasings (where it reaches, prepared inside reach, unique continuation) hold the abstract at 300 words.
% lead inbox 108 (FableAPI_embedRF_v1.md md5 04a8f94b, text A abstract form):
```

### 10.448 main2: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_main2 pair 1), beginning "The flux leaving the whole system generates the evolution."
Replaced text:
```latex
The flux leaving the whole system generates the evolution.
```
Replacement:
```latex
The flux leaving the whole system generates evolution.
```

### 10.449 Sec. I: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_02 pair 1), beginning "The source is one lattice fermion on fixed cubic sites,"
Replaced text:
```latex
The source is one lattice fermion on fixed cubic sites,
```
Replacement:
```latex
The source is a single Dirac fermion field on fixed cubic sites,
```

### 10.450 Sec. II: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_03 pair 1), beginning "The one lattice fermion is the matter that generates the geometry,"
Replaced text:
```latex
The one lattice fermion is the matter that generates the geometry,
```
Replacement:
```latex
The single lattice fermion field is the matter that generates the geometry,
```

### 10.451 Sec. II: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_03 pair 2), beginning "The first declaration is the matter: one Dirac fermion on the fixed cu"
Replaced text:
```latex
The first declaration is the matter: one Dirac fermion on the fixed cubic lattice,
```
Replacement:
```latex
The first declaration is the matter: a single species of Dirac fermion on the fixed cubic lattice,
```

### 10.452 Sec. VII: replaced text, verbatim (owner comment cmw0j97q2855c (one Dirac fermion read as one particle): one species, many particles, spec OF_08 pair 1), beginning "Take one Dirac fermion on a fixed cubic lattice,"
Replaced text:
```latex
Take one Dirac fermion on a fixed cubic lattice,
```
Replacement:
```latex
Take a single Dirac fermion field on a fixed cubic lattice,
```

### 10.453 Sec. V: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B7 held behind a pointer, lead GO), spec T_SEL_07 pair 1), beginning "The Hawking comparison case is a thermally supplied exterior whose tra"
Replaced text:
```latex
The Hawking comparison case is a thermally supplied exterior whose transmission is the model's own. On the exterior of Eq.~\eqref{eq:dr-exterior} the operator of channel \(k\) is \(\bar h_k=h_{0,k}+\tfrac12\{v,Q_r\}\) at \(m=0\), \(N=1\), \(v=-\sqrt{r_s/r}\). In the leading continuum limit on the unaveraged vacuum profile the stationary chain spinor, rescaled by \(\alpha_s^{1/2}\), boosted to the static frame and stripped of its stationary phase, obeys the massless Dirac equation on Schwarzschild in the tortoise coordinate \(r_*\)~\cite{Unruh1976,Chandrasekhar1983}, \(\alpha_s=\sqrt{1-r_s/r}\) the Killing lapse, so its two components see the partner barriers \(W^2\pm\partial_{r_*}W\), \(W=\alpha_sk/r\), rising to the centrifugal \(k(k\pm1)/r^2\) far out and vanishing at \(r_s\) with \(\alpha_s\) (the reduction in Ref.~\cite{McKinneyCode2026}). The unit-flux transmission
```
Replacement:
```latex
% selection (owner rule 2026-11-04, lead GO: B7 held in the repository behind a pointer): the static-frame Dirac reduction with its partner barriers and the binary-transmission limits moved to Ref. McKinneyCode2026; the channel operator, the transmission statement, the Landauer currents, the finite-band statement with mark B7, the number-rate sentence and the ratio statement kept; the Sec. V C shell-flow checkpoint numbers and the hold's invariant, flow and excess numbers moved to the Ref. with their conclusions kept in place; archived verbatim in REPO_DERIVATIONS.
The Hawking comparison case is a thermally supplied exterior whose transmission is the model's own. On the exterior of Eq.~\eqref{eq:dr-exterior} the operator of channel \(k\) is \(\bar h_k=h_{0,k}+\tfrac12\{v,Q_r\}\) at \(m=0\), \(N=1\), \(v=-\sqrt{r_s/r}\), whose continuum limit is the massless Dirac equation on Schwarzschild~\cite{Unruh1976,Chandrasekhar1983} with its centrifugal partner barriers (the static-frame reduction in Ref.~\cite{McKinneyCode2026}). The unit-flux transmission
```

### 10.454 Sec. V: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B7 held behind a pointer, lead GO), spec T_SEL_07 pair 2), beginning "For binary transmissions the currents depend only on the open-channel "
Replaced text:
```latex
For binary transmissions the currents depend only on the open-channel count, a constant count giving \(\mathcal R=2\) and a count growing as \(\omega^2\) giving \(4/3\) by elementary Fermi integrals, comparison limits and not bounds on a soft barrier, whose entropy sum is not fixed by its transmission sum. Neither the flux
```
Replacement:
```latex
Neither the flux
```

### 10.455 Sec. V: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B7 held behind a pointer, lead GO), spec T_SEL_07 pair 3), beginning "at the stored slice charge's \(T_H\). It is at most 0.15 of the slope "
Replaced text:
```latex
at the stored slice charge's \(T_H\). It is at most 0.15 of the slope comparison flux before 36, two orders above over
42--46, and comparable over 80--100 except one negative checkpoint
(excitation fractions in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
at the stored slice charge's \(T_H\) (checkpoint values and excitation fractions in Ref.~\cite{McKinneyCode2026}).
```

### 10.456 Sec. V: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B7 held behind a pointer, lead GO), spec T_SEL_07 pair 4), beginning "Holding from 100 to 500 preserves \(E_{\rm inv}\) to seven digits, and"
Replaced text:
```latex
Holding from 100 to 500 preserves \(E_{\rm inv}\) to seven digits, and at every \(10r_s\) checkpoint from 140 the \(k=1\) per-chain flow is 0.9--1.5 \(L_H(T_H)\), and over the whole hold the cut excess grows by 1.18--1.28 nats per chain.
```
Replacement:
```latex
Holding from 100 to 500, the \(k=1\) per-chain flow stays near \(L_H(T_H)\) at every checkpoint from 140 (the invariant, flow, and excess values in Ref.~\cite{McKinneyCode2026}).
```

### 10.457 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 1), beginning "The massive chain's outer endpoint is a fixed-radius subtraction, supp"
Replaced text:
```latex
The massive chain's outer endpoint is a fixed-radius subtraction, supported by every stored cutoff comparison, but no endpoint coefficient is established, the cutoff changes not yet decreasing (cutoff tests in Ref.~\cite{McKinneyCode2026}). The massless light chain has no finite endpoint under the constant-coefficient prescription: its endpoint coefficient \(e_{11}\) diverges for every \(k=1,\ldots,5\), with no cancellation established (kernel limits and endpoint rule in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
% selection (owner rule 2026-11-04): method and measurement specifics moved to Ref. McKinneyCode2026 with their conclusions kept in place (endpoint subtractions, source and field agreement percentages, column occupations, width-agreement criteria, completion rules, the redshift headline procedure, the comparator-energy Wick mechanics, the hierarchy's numerical margin, the per-particle energy and shell fractions); archived verbatim in REPO_DERIVATIONS.
The chains' endpoint subtractions are a fixed-radius subtraction for the massive chain, supported by every stored cutoff comparison but with no endpoint coefficient established, and none for the massless light chain under the constant-coefficient prescription (cutoff tests, kernel limits, and the endpoint rule in Ref.~\cite{McKinneyCode2026}).
```

### 10.458 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 2), beginning "The energy source differs by at most 3.2 percent of the spread, the re"
Replaced text:
```latex
The energy source differs by at most 3.2 percent of the spread, the re-solved clock and flow lie within 0.6 percent of it, and the fixed reference projector's departure from the prepared band is almost exactly the second-gradient polarization that Eq.~\eqref{eq:dm-reference} subtracts.
```
Replacement:
```latex
The fixed reference projector's departure from the prepared band is almost exactly the second-gradient polarization that Eq.~\eqref{eq:dm-reference} subtracts (source and field differences in Ref.~\cite{McKinneyCode2026}).
```

### 10.459 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 3), beginning "The dust, the \(N_B\) added positive-band fermions, occupies \(5,4,3,2"
Replaced text:
```latex
The dust, the \(N_B\) added positive-band fermions, occupies
\(5,4,3,2,2\) columns (occupied radial orbitals) in \(k=1,\ldots,5\),
with weights \(16k\). It is prepared in a self-consistent confining
well at \(R_0=4r_s\), projected on the positive band and
orthonormalized, above the filled negative sea. No coordinate-rest
condition is imposed (the dust's release momentum is in
Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The dust, the \(N_B\) added positive-band fermions in \(k=1,\ldots,5\), is prepared in a self-consistent confining well at \(R_0=4r_s\), projected on the positive band and orthonormalized above the filled negative sea, with no coordinate-rest condition imposed (column occupations and the release momentum in Ref.~\cite{McKinneyCode2026}).
```

### 10.460 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 4), beginning "All three are verified through formation of the trapping surface, \(\s"
Replaced text:
```latex
All three are verified through formation of the trapping surface, \(\sigma/a=2\) through 40, agreement requiring formation clocks within \(0.5r_s\) and trapping radii within 10 percent at matched clocks.
```
Replacement:
```latex
All three are verified through formation of the trapping surface, \(\sigma/a=2\) through 40 (agreement criteria in Ref.~\cite{McKinneyCode2026}).
```

### 10.461 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 5), beginning "Three completions restart at the reference collapse's \(T/r_s=11.0\) s"
Replaced text:
```latex
Three completions restart at the reference collapse's \(T/r_s=11.0\) slice: inside the failing radius they hold sources with charge projected onto the slice's, widen averaging to \(\sigma/a=8\), or point-sample core coupling (activation and end slices in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
Three completions restart at the reference collapse's \(T/r_s=11.0\) slice with held, widened, or point-sampled core coupling (their rules and end slices in Ref.~\cite{McKinneyCode2026}).
```

### 10.462 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 6), beginning "For each sampled class and verified arrival interval the headline repo"
Replaced text:
```latex
For each sampled class and verified arrival interval the headline reports \(\ln(1/g_D)\) at the maximum of the emitter's clock rate: the sampled redshifts reach
```
Replacement:
```latex
The sampled redshifts reach
```

### 10.463 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 7), beginning "The second-order comparator energy is nonpositive: full Wick ordering "
Replaced text:
```latex
The second-order comparator energy is nonpositive: full Wick ordering makes the first-order expectation in \(P_\star\) zero, and against the ground projector \(P_g\) of \(h[\bar C]\) on a gapped path the one-body term \(B=-h_H[\bar\Delta_\star]\) and the quartic term create orthogonal single and double particle-hole states. The connected comparator energy \(\Delta E_{\rm ref}^{(2)}\), apart from \(\mathcal E_H\), is then minus a sum of squared amplitudes over excitation energies, strictly negative unless both vanish, the single sum absent only if \(P_gB(I-P_g)=0\) (order counting in \(G\) in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The second-order comparator energy \(\Delta E_{\rm ref}^{(2)}\) is nonpositive: against the ground projector \(P_g\) of \(h[\bar C]\) on a gapped path it is minus a sum of squared single and double particle-hole amplitudes over excitation energies, strictly negative unless both vanish, the one-body term \(B=-h_H[\bar\Delta_\star]\) driving the singles (Wick ordering and order counting in \(G\) in Ref.~\cite{McKinneyCode2026}).
```

### 10.464 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 8), beginning "The hydrodynamic hierarchy bounds the averaged source \(\rho\), not \("
Replaced text:
```latex
The hydrodynamic hierarchy bounds the averaged source \(\rho\), not \(\rho_x\), the strain bound \(\sigma\mathcal K\leq0.1\) capping \(\rho a^4\) four orders of magnitude below exclusion's kinematic bound at \(\sigma=4a\) and the spacing of Eq.~\eqref{eq:page-share} (values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The hydrodynamic hierarchy bounds the averaged source \(\rho\), not \(\rho_x\), far below exclusion's kinematic bound (values in Ref.~\cite{McKinneyCode2026}).
```

### 10.465 Sec. IV: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_05 pair 9), beginning "The dust's bare energy per particle \(\epsilon_c\) inside \(R_{50}\) i"
Replaced text:
```latex
The dust's bare energy per particle \(\epsilon_c\) inside \(R_{50}\) is 2.7 to 3.4 times \(ma\) at 15 and 16. The inner shell's dust at \(|ap|>\pi/4\) is at most 15 percent, and the \(\pi/4\) split, a plane-wave criterion, reads neither bin as outgoing content.
```
Replacement:
```latex
The inner shell's dust at \(|ap|>\pi/4\) is a small fraction, and the \(\pi/4\) split, a plane-wave criterion, reads neither bin as outgoing content (the per-particle energy and shell fractions in Ref.~\cite{McKinneyCode2026}).
```

### 10.466 Sec. VI: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_06 pair 1), beginning "It therefore holds to relative accuracy \(\epsilon\) exactly when the "
Replaced text:
```latex
It therefore holds to relative accuracy \(\epsilon\) exactly when the storage rate, the defect, the charge drift, and the mismatch between the shell clock and \(u\) each stay below \(\epsilon L\) over the window. A quasi-stationary stored state of the massless field puts the first at \(2\ell_c/t_{\rm em}\), with \(\ell_c\) the shell's crossing time and \(t_{\rm em}=E_R/L\), of order \(\chi^2\), and over any window with \(Q=\int L_c\,dt>0\), relaxed storage \(\mathcal S(R)\) with \(s=\sup|2G\mathcal S'|<1\) and transient \(|\delta_{\rm st}|\le\delta_0\), and \(\int|\mathcal D_c|\,dt\le cQ\), the bound is \(|\Delta E_R/Q+1|\le(s+c+2\delta_0/Q)/(1-s)\).
```
Replacement:
```latex
It therefore holds to relative accuracy \(\epsilon\) exactly when the storage rate, the defect, the charge drift, and the mismatch between the shell clock and \(u\) each stay below \(\epsilon L\) over the window, a quasi-stationary stored state of the massless field putting the first at order \(\chi^2\) (the window bound in Ref.~\cite{McKinneyCode2026}).
```

### 10.467 Sec. VI: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_06 pair 2), beginning "The reported late slope over 80--100, 0.030--0.040 nats per nominal \("
Replaced text:
```latex
The reported late slope over 80--100, 0.030--0.040 nats per nominal \(r_s\), is 0.28--0.37 of the nominal comparison-case rate \(0.1079/r_s\). The separate 0.23--0.31 normalization of that diagnostic (Ref.~\cite{McKinneyCode2026}) remains unverified until the stored history's shell charge at 100 and the comparison rate, clock and units actually used are printed beside it
```
Replacement:
```latex
The reported late slope over 80--100 is about a third of the nominal comparison-case rate, and its separate normalization remains unverified until the stored history's shell charge at 100 and the comparison rate, clock and units actually used are printed beside it (the slope and both normalizations in Ref.~\cite{McKinneyCode2026})
```

### 10.468 Sec. VI: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_06 pair 3), beginning "At \(a=2\ell_P\), \(ma=0.2\), nominal \(f=0.887\) gives \(0.5755S_0\) "
Replaced text:
```latex
At \(a=2\ell_P\), \(ma=0.2\), nominal \(f=0.887\) gives \(0.5755S_0\) at \(0.4773t_*\) and the interval bound \(0.7883<f<0.9831\) of Sec.~\ref{sec:results-vacuum} the heights \([0.5323,0.6146]S_0\) at saturation, while the finite-band \(\mathcal L\) and \(\mathcal R\) have no narrow bound.
```
Replacement:
```latex
At the nominal cut share \(f=0.887\) and across its interval bound of Sec.~\ref{sec:results-vacuum} the maximum moves by a few percent (values in Ref.~\cite{McKinneyCode2026} and Fig.~\ref{fig:page-conditional}), while the finite-band \(\mathcal L\) and \(\mathcal R\) have no narrow bound.
```

### 10.469 Sec. VI: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, lead GO), spec T_SEL_06 pair 4), beginning "Three statements follow from the propagator on the exterior of Eq.~\eq"
Replaced text:
```latex
Three statements follow from the propagator on the exterior of Eq.~\eqref{eq:dr-exterior}, each with its own hypotheses.
```
Replacement:
```latex
% selection (owner rule 2026-11-04): the closure's window-bound mechanics, the late-slope and normalization numbers, and the Page-maximum variant values moved to Ref. McKinneyCode2026 with their conclusions and marks kept in place; archived verbatim in REPO_DERIVATIONS.
Three statements follow from the propagator on the exterior of Eq.~\eqref{eq:dr-exterior}, each with its own hypotheses.
```

### 10.470 Sec. III: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B4 held behind a pointer, lead GO), spec T_SEL_04 pair 1), beginning "Its exact decomposition in two-step shifts (Ref.~\cite{McKinneyCode202"
Replaced text:
```latex
Its exact decomposition in two-step shifts (Ref.~\cite{McKinneyCode2026}) has a scalar part, axial two-step paths forming the current \(Q_I\) of Sec.~\ref{sec:model-dictionary} smeared by a bond field built from \(f\) and \(g\), and a grade-two cross-direction part, the spin term with the lattice vertex \(\Sigma^{\rm lat}_K=\Sigma_K\mathsf c_I\mathsf c_J\) of Eq.~\eqref{eq:dm-hamiltonian}, while general smearings have additional lattice dependence not determined by \(u\). For smearings affine in \(x\) both parts are exact couplings: with \(u=f\nabla g-g\nabla f\), itself affine, \(i[H[f],H[g]]=c^\dagger\bigl[\tfrac12\sum_I\{u_I,Q_I\}+\tfrac18\sum_K\{(\nabla\times u)_K,\Sigma^{\rm lat}_K\}\bigr]c\) with no remainder, and for smooth smearings the bond field's midpoint expansion gives, for the harmonic pair above, the \(F_a(\xi)\) above.
```
Replacement:
```latex
Its exact decomposition in two-step shifts has a scalar part, the current \(Q_I\) of Sec.~\ref{sec:model-dictionary} smeared by a bond field, and a grade-two spin part with the lattice vertex of Eq.~\eqref{eq:dm-hamiltonian}, both exact couplings for affine smearings, the \(F_a(\xi)\) above for the harmonic pair, while general smearings have additional lattice dependence not determined by \(u\) (the decomposition and the affine coupling in Ref.~\cite{McKinneyCode2026}).
```

### 10.471 Sec. III: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B4 held behind a pointer, lead GO), spec T_SEL_04 pair 2), beginning "% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids"
Replaced text:
```latex
% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids applied: A-B4.1
```
Replacement:
```latex
% held (owner rule 2026-11-04, lead GO: B4 held in the repository behind a pointer): the decomposition's explicit form, the affine coupling formula and the vertex definition moved to Ref. McKinneyCode2026, the thesis, the scope, the F_a(xi) link, the time-reversal condition and the closer kept; archived verbatim in REPO_DERIVATIONS.
% status (lead inbox 105): APPLIED (derived algebra only). borrowA ids applied: A-B4.1
```

### 10.472 Sec. III: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B4 held behind a pointer, lead GO), spec T_SEL_04 pair 3), beginning "The flat sea's window energy is explicit, \(8\pi\gamma_1L+O(1)\) for t"
Replaced text:
```latex
The flat sea's window energy is explicit, \(8\pi\gamma_1L+O(1)\) for the harmonic window with \(\gamma_1=0.392\) at \(ma=0.2\), about \(9.9L\) at large \(L\) and computed exactly at each size, and the \(N_B\) dust orbitals contribute at most \(\varepsilon_{\max}N_B\), \(18.9L\) at the reference parameters (Sec.~\ref{sec:star}), for every state and every size of the family (the window, the gradient sum, and the orbital count in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The flat sea's window energy is explicit and linear in \(L\), computed exactly at each size, and the \(N_B\) dust orbitals contribute at most \(\varepsilon_{\max}N_B\), for every state and every size of the family (the window, the gradient sum, and the orbital count in Ref.~\cite{McKinneyCode2026}).
```

### 10.473 Sec. III: replaced text, verbatim (selection pass (owner rule 2026-11-04: excessive detail to the repository, B4 held behind a pointer, lead GO), spec T_SEL_04 pair 4), beginning "with directional finite parts between \(-0.42\) and \(-0.46\) and a sp"
Replaced text:
```latex
with directional finite parts between \(-0.42\) and \(-0.46\) and a sphere mean near \(-0.46\) without a bound (its angular remainder not proved uniform), and it enters
```
Replacement:
```latex
with a sphere-mean finite part near \(-0.46\) without a bound (its angular remainder not proved uniform, directional values in Ref.~\cite{McKinneyCode2026}), and it enters
```

### 10.474 abstract: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_main pair 1), beginning "We study many particles of a Dirac fermion species on a fixed cubic la"
Replaced text:
```latex
We study many particles of a Dirac fermion species on a fixed cubic lattice, assuming no light cone, vacuum, or metric.
```
Replacement:
```latex
We study lattice fermions on fixed cubic sites, a Dirac fermion field, assuming no light cone, vacuum, or metric.
```

### 10.475 abstract: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_main pair 2), beginning "% owner comment cmw0j97q2855c ("We study one Dirac fermion ... It's no"
Replaced text:
```latex
% owner comment cmw0j97q2855c ("We study one Dirac fermion ... It's not just one"):
```
Replacement:
```latex
% owner comments cmw0jjt7lhe38, cmw0jkfinuec6 (lead-routed, superseding the one-species wording): "lattice fermions on fixed cubic sites, a Dirac fermion field" in the abstract, "the lattice fermions ... one Dirac fermion field", "the fermions", "the same fermions, whose energy and current source the fields, form the cloud that collapses" in Secs. I, II and VII, "lattice fermions on the fixed cubic sites, one Dirac fermion field" inside the first declaration's bold (only the one-fermion words changed).
% owner comment cmw0j97q2855c ("We study one Dirac fermion ... It's not just one"):
```

### 10.476 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 1), beginning "Sindoni~\cite{Sindoni2012} reviews microscopic emergent-gravity models"
Replaced text:
```latex
Sindoni~\cite{Sindoni2012} reviews microscopic
emergent-gravity models.
```
Replacement:
```latex
Sindoni~\cite{Sindoni2012} reviews other microscopic emergent-gravity models.
```

### 10.477 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 2), beginning "Here the input is one matter Hamiltonian on fixed sites with no metric"
Replaced text:
```latex
Here the input is one matter Hamiltonian on fixed sites with no metric and a single charge requirement, from which the gravitational constraints and the spherical collapse problem are derived.
```
Replacement:
```latex
% owner comments cmw0jfb2o3t74 (the Sindoni sentence), cmw0jig6nodrk (the transition after it, basic and logically ordered), cmw0jjt7lhe38 and cmw0jkfinuec6 (lattice fermions, a Dirac fermion field, the same fermions form the cloud), cmw0k15basrtz (the charge principle named), cmw0k2pq2vpb1 and the owner's follow-ups (the structural-versus-hydrodynamic story told as the introduction goes and in the roadmap, at the introduction's prior length: the four-step list of Sec. II D and the lattice-description sentence of the roadmap, both restated later, dropped to pay); lead-routed 2026-11-04.
Here the input is a matter Hamiltonian for fermions on fixed sites with no metric, from which the gravitational equations of motion are derived.
```

### 10.478 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 3), beginning "The source is a single Dirac fermion field on fixed cubic sites, a com"
Replaced text:
```latex
The source is a single Dirac fermion field on fixed cubic sites, a complete quantum theory with no light cone, vacuum, or metric assumed, and the same fermion is the cloud that collapses. The one principle is that energy is a charge. Two fields on the sites are the only new quantities: a clock rate that weights the fermion's energy and a flow velocity that weights its momentum. The principle says that the energy inside any set of sites is a flux of these two fields through the bonds leaving it, as Gauss's law reads charge from electric flux. Like Gauss's law, this relation is a constraint, not an equation of motion. In the hydrodynamic limit, on a declared averaging scale, the clock and flow are fixed at each instant by the matter's averaged energy and momentum, as the electrostatic field is fixed by its charges. The fermion's symmetries and the charge requirement then leave only Einstein's constraint equations, with one constant, \(G\), that calibrates the flux. The sources are measured against the vacuum the collapse itself prepares, so uniform fields source nothing. Geometry is then bookkeeping of how the matter's energy and momentum are weighted. Section~\ref{sec:model-gravity} derives it in four steps: the laws the principle allows, the metric read from clock, flow, and spacing, the two spherical rows that follow a source in time, and what survives the continuum limit, where one parameter remains, \(G\) in lattice units.
```
Replacement:
```latex
The source is lattice fermions on fixed cubic sites, a Dirac fermion field with no light cone, vacuum, or metric assumed, and the same fermions, whose energy and current source the fields, form the cloud that collapses. The one new input is the charge principle: energy is a charge. Two fields on the sites are the only new quantities, a clock rate that weights the fermions' energy and a flow velocity that weights their momentum, both defined at every step from the matter's energy and momentum averaged over a declared scale, since these fluctuate from site to site. The charge principle says that the energy inside any set of sites is a flux of these two fields through the bonds leaving it, as Gauss's law reads charge from electric flux, and like Gauss's law this relation is a constraint, not an equation of motion. The sources are measured against the vacuum the collapse itself prepares, so uniform fields source nothing, the same subtraction for every kind of matter, which is why the laws carry one constant. Where the matter varies slowly over the averaging scale, the hydrodynamic limit, the fermions' symmetries and the charge principle leave only Einstein's constraint equations, with one constant, \(G\), that calibrates the flux. Geometry is then bookkeeping of how the matter's energy and momentum are weighted.
```

### 10.479 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 4), beginning "And its boundary charge defines the quantum spherical sector on the st"
Replaced text:
```latex
And its boundary charge defines the quantum spherical sector on the states where it stays close to its expectation (Sec.~\ref{sec:model}).
```
Replacement:
```latex
And its boundary energy, measured against the prepared vacuum, is sharp on a stated set of states, the quantum spherical sector on which that energy serves as the charge (Sec.~\ref{sec:model}).
```

### 10.480 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 5), beginning "Section~\ref{sec:model} states these five declarations and derives the"
Replaced text:
```latex
Section~\ref{sec:model} states these five declarations and derives the laws from them, and the sections that follow put those laws to work on the exterior, the collapse, and the radiation.
```
Replacement:
```latex
Section~\ref{sec:model} states these five declarations and derives the laws from them. Four declarations are structural inputs, and so is the third's averaging scale, while the slow-variation premise on that scale is the hydrodynamic limit, in which the laws take Einstein's form.
```

### 10.481 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 6), beginning "the Bekenstein-Hawking entropy to the entanglement across the surface "
Replaced text:
```latex
the Bekenstein-Hawking entropy to the entanglement across the surface at a Planck-scale spacing. Section~\ref{sec:star}
```
Replacement:
```latex
the Bekenstein-Hawking entropy to the entanglement across the surface at a Planck-scale spacing. Sections~\ref{sec:model} and \ref{sec:results} work in that limit. Sections~\ref{sec:star} and \ref{sec:page} do not assume it: they evolve the exact lattice fermions with the clock and flow re-solved at every step from the same constraint equations, so the computation continues where the hydrodynamic reading of the fields ends. Section~\ref{sec:star}
```

### 10.482 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 7), beginning "Without a smooth geometry a singularity needs a new definition: the se"
Replaced text:
```latex
Without a smooth geometry a singularity needs a new definition: the section adopts failure of unique joint continuation of state, clock, and flow and tests its conditions at every stored time slice of the computed collapse.
```
Replacement:
```latex
Without a smooth geometry a singularity needs a new definition, which the section supplies and tests at every stored time slice.
```

### 10.483 Sec. I: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_02 pair 8), beginning "and reads its early entropy outside a cut radius. The static and entan"
Replaced text:
```latex
and reads its early entropy outside a cut radius. The static and entanglement results are computed on the full cubic lattice, the collapse and its light field on one radial chain per angular-momentum channel. Section~\ref{sec:scaling}
```
Replacement:
```latex
and reads its early entropy outside a cut radius. Section~\ref{sec:scaling}
```

### 10.484 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 1), beginning "The matter comes first. The single lattice fermion field is the matter"
Replaced text:
```latex
The matter comes first. The single lattice fermion field is the matter that generates the geometry, its energy and current being the sources, as distinct from matter that would only contribute a stress tensor on a given geometry. Here there is no second kind, since the same fermion is both the source and the collapsing cloud.
```
Replacement:
```latex
% owner comments (lead-routed 2026-11-04): cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c (lattice fermions, one Dirac fermion field, the same fermions form the cloud; inside the first declaration's bold only the one-fermion words changed); cmw0jqxyz4zbm with the owner's follow-up (the Sec. II A doubling passage cut to the symbols later sections use: d_i(p) and epsilon(p) for Secs. II D, III A, IV D and Appendix B, the valleys p_star for Secs. II to V and the appendices, the mass-m Dirac species and the 16 states per momentum for Secs. IV A and V B, the doubling cited to NielsenNinomiya1981, the expansion sentence dropped since Eq. dr-cone-expansion restates it); cmw0jusflcthw (the second-declaration connector in order); cmw0k15basrtz ("the charge principle" wherever the principle stood alone, defined at the second declaration); cmw0k2pq2vpb1 (one statement of the hydrodynamic limit at the Sec. II D opener); cmw0k5ogtc6fe (the step-two opener said straight); cmw0kar9upu6c with the owner's follow-up (the fifth-declaration passage in plain words, the owner lifting its verbatim protection); the owner's structural-versus-hydrodynamic sentence at the Sec. II closer.
The matter comes first. The lattice fermions, one Dirac fermion field on the fixed cubic sites, are the matter that generates the geometry, their energy and current being the sources, as distinct from matter that would only contribute a stress tensor on a given geometry. Here there is no second kind, since the same fermions, whose energy and current source the fields, form the cloud that collapses.
```

### 10.485 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 2), beginning "On a plane wave of momentum \(p\), \(\widehat p_i\to d_i(p)=\sin(ap_i)"
Replaced text:
```latex
On a plane wave of momentum
\(p\), \(\widehat p_i\to d_i(p)=\sin(ap_i)/a\) and
\(h_0^2\to\varepsilon(p)^2=m^2+|d(p)|^2\). The eight zeros \(p_\star\) of \(d\), with \(p_{\star i}\in\{0,\pi/a\}\), are the valleys. Near each, \(d_i=\pm k_i+O(a^2k_i^3)\) for \(p=p_\star+k\), so \(h_0\) is a Dirac operator of mass \(m\): eight light Dirac species of one lattice
field, \(8\times2=16\) particle states per low-energy momentum.
```
Replacement:
```latex
On a plane wave of momentum \(p\), \(\widehat p_i\to d_i(p)=\sin(ap_i)/a\) and \(h_0^2\to\varepsilon(p)^2=m^2+|d(p)|^2\), whose eight zeros \(p_\star\), \(p_{\star i}\in\{0,\pi/a\}\), are the valleys: by fermion doubling~\cite{NielsenNinomiya1981} each is a Dirac species of mass \(m\), \(8\times2=16\) particle states per low-energy momentum.
```

### 10.486 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 3), beginning "The first declaration is the matter: a single species of Dirac fermion"
Replaced text:
```latex
The first declaration is the matter: a single species of Dirac fermion on the fixed cubic lattice, with its energy
```
Replacement:
```latex
The first declaration is the matter: lattice fermions on the fixed cubic sites, one Dirac fermion field, with its energy
```

### 10.487 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 4), beginning "because the principle needs something for the matter's energy and mome"
Replaced text:
```latex
because the principle needs something for the matter's energy and momentum
```
Replacement:
```latex
because the charge principle needs something for the matter's energy and momentum
```

### 10.488 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 5), beginning "Two fields now weight the matter's energy and momentum, and nothing ye"
Replaced text:
```latex
Two fields now weight the matter's energy and momentum, and nothing yet says what fixes the fields, which is the principle's job. \fnd{\textbf{The second declaration is the principle: energy is a charge.}
```
Replacement:
```latex
Two fields now weight the matter's energy and momentum. What fixes them is the second declaration, the charge principle. \fnd{\textbf{The second declaration is the charge principle: energy is a charge.}
```

### 10.489 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 6), beginning "What the principle still needs is the scale on which the fields are"
Replaced text:
```latex
What the principle still needs is the scale on which the fields are
```
Replacement:
```latex
What the charge principle still needs is the scale on which the fields are
```

### 10.490 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 7), beginning "expansion in which the principle's four identities hold order by"
Replaced text:
```latex
expansion in which the principle's four identities hold order by
```
Replacement:
```latex
expansion in which the charge principle's four identities hold order by
```

### 10.491 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 8), beginning "The first finds the laws the principle allows, without which no geomet"
Replaced text:
```latex
The first finds the laws the principle allows, without which no geometry can be read. The second reads clock, flow, and lattice spacing as a metric, which the third needs to follow a source in time through the spherical rows. The fourth asks what of this survives the continuum limit.
```
Replacement:
```latex
The first finds the laws the charge principle allows, without which no geometry can be read. The second reads clock, flow, and lattice spacing as a metric, which the third needs to follow a source in time through the spherical rows. The fourth asks what of this survives the continuum limit. All four work in the hydrodynamic limit, the regime in which the matter varies slowly over the averaging scale, and the computed collapse of Sec.~\ref{sec:star} does not assume it.
```

### 10.492 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 9), beginning "The first step finds the laws, the local functional the principle"
Replaced text:
```latex
The first step finds the laws, the local functional the principle
```
Replacement:
```latex
The first step finds the laws, the local functional the charge principle
```

### 10.493 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 10), beginning "\int(\rho_\ell\,\delta N+J_\ell\cdot\delta v)\,d^3y\), the principle"
Replaced text:
```latex
\int(\rho_\ell\,\delta N+J_\ell\cdot\delta v)\,d^3y\), the principle
```
Replacement:
```latex
\int(\rho_\ell\,\delta N+J_\ell\cdot\delta v)\,d^3y\), the charge principle
```

### 10.494 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 11), beginning "selected by the principle's four requirements on the declared domain"
Replaced text:
```latex
selected by the principle's four requirements on the declared domain
```
Replacement:
```latex
selected by the charge principle's four requirements on the declared domain
```

### 10.495 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 12), beginning "The second step reads the geometry the laws describe, since without a "
Replaced text:
```latex
The second step reads the geometry the laws describe, since without a metric the words trapping surface, redshift, and horizon have no meaning here: clock, flow, and lattice spacing assemble into a metric, the flat lattice an input. In slowly varying
```
Replacement:
```latex
The second step reads the geometry the laws describe: clock, flow, and lattice spacing assemble into a metric, and that metric is what defines the trapping surface and the redshift used below. In slowly varying
```

### 10.496 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 13), beginning "The strong constraints exclude every normal state, so the quantum theo"
Replaced text:
```latex
The strong constraints exclude every normal state, so the quantum theory needs a stated domain on which the spherical sector lives, and the fifth declaration gives it. \fnd{\textbf{The fifth declaration is the quantum spherical sector:
the boundary-charge operator, Wick ordered against the sea the
collapse prepared, restricted to the states on which it stays within
a factor two of its expectation.} The computations work at Hartree order inside it, the working approximation rather than a declared input. That restriction is the branch, the range of the
spectral projector \(\Pi\) of the charge operator, and the ordering
leaves the prepared vacuum with no pair energy. The sector is part of
the quantum theory, not an approximation to it, a small-fluctuation
theory about the Hartree collapse in which \(E[C]=\mathcal M(\infty)\)
is expanded to second order in bare moments about the regular Hartree covariance \(\bar C\), retaining the reference's polarization
response, and fully Wick ordered against the prepared sea \(P_\star\). The polarization counterterms are scalars and order
nothing, the pair vertex has zero Gaussian expectation and zero first
covariance derivative at \(P_\star\) (the no-tadpole condition), and
vacuum subtraction alone does not fix this ordering.} The branch is the range of the spectral projector \(\Pi=\mathbf1_{[E/2,3E/2]}(\widehat E_2)\) of the second-order charge operator about its Hartree value \(E=E[\bar C]>0\).
```
Replacement:
```latex
The first four declarations fix the clock and flow from the matter, but the matter is quantum, and the constraints cannot be imposed as exact operator equations on all states, since the strong constraints exclude every normal state. The quantum problem therefore needs a stated domain, and the fifth declaration gives it. \fnd{\textbf{The fifth declaration is the quantum spherical sector: the states on which the total energy, read as the flux through the outer boundary and measured against the prepared vacuum, is sharp, within a factor of two of its mean.} On those states the boundary energy behaves as a definite charge, so the charge principle applies to quantum matter. Measuring against the prepared vacuum is the Wick ordering: the energy \(E[C]=\mathcal M(\infty)\) is expanded to second order in bare moments about the regular Hartree covariance \(\bar C\), retaining the reference's polarization response, and fully Wick ordered against the prepared sea \(P_\star\), which leaves the prepared vacuum with no pair energy. The polarization counterterms are scalars and order nothing, the pair vertex has zero Gaussian expectation and zero first covariance derivative at \(P_\star\) (the no-tadpole condition), and vacuum subtraction alone does not fix this ordering.} The computations work inside this sector at Hartree order, the working approximation and not part of the declaration. The sector is the branch, the range of the spectral projector \(\Pi=\mathbf1_{[E/2,3E/2]}(\widehat E_2)\) of the second-order charge operator about its Hartree value \(E=E[\bar C]>0\).
```

### 10.497 Sec. II: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_03 pair 14), beginning "Section~\ref{sec:model} is complete: five declarations, two constraint"
Replaced text:
```latex
Section~\ref{sec:model} is complete: five declarations, two constraint equations, one metric, and one quantum sector. From here on
```
Replacement:
```latex
Section~\ref{sec:model} is complete: five declarations, two constraint equations, one metric, and one quantum sector. Four declarations and the averaging scale of the third are structural inputs, while the slow-variation premise on that scale is the hydrodynamic limit in which the two constraint equations take Einstein's form. From here on
```

### 10.498 Sec. III: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_04 pair 1), beginning "With the laws in hand, this section derives their consequences before "
Replaced text:
```latex
With the laws in hand, this section derives their consequences before any collapse is computed. Each step uses the one before it:
```
Replacement:
```latex
% owner comment (lead-routed 2026-11-04) on this opener: it read as if the collapse used the hydrodynamic-limit laws; reworded, and the paper swept for the same implication (the introduction's "put those laws to work on the collapse" reworded in the same commit, no other instance found).
This section works out what the constraint equations imply in the hydrodynamic limit: the exterior of a spherical source, its temperature, and its entropy. The computed collapse of Sec.~\ref{sec:star} does not rest on these results, since it evolves the same lattice equations directly. Each step uses the one before it:
```

### 10.499 Sec. IV: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_05 pair 1), beginning "Everything to this point is analytical or quasi-analytical. With conti"
Replaced text:
```latex
Everything to this point is analytical or quasi-analytical. With continuum transport and vanishing source defects, every positive
```
Replacement:
```latex
% owner comment cmw0k2pq2vpb1 with the owner's follow-up (lead-routed 2026-11-04): the opener says which sections were the hydrodynamic continuum limit and how the computation proceeds; the hydrodynamic-test paragraph's restatement of the regime folded to pay.
Everything to this point, Secs.~\ref{sec:model} and \ref{sec:results}, was derived in the hydrodynamic continuum limit. The computation starts here and does not assume that limit: it evolves the exact lattice fermions, with the clock and flow re-solved at every step from the same constraint rows applied to the actual averaged matter and measured against the vacuum the collapse prepares, so it stays valid where the slow-variation premise fails, and what stops there is only the reading of clock and flow as a geometry, which the hydrodynamic test flags, readings past it being labelled statements about the lattice equations. With continuum transport and vanishing source defects, every positive
```

### 10.500 Sec. IV: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_05 pair 2), beginning "Either failure ends the hydrodynamic description, the regime in which "
Replaced text:
```latex
Either failure ends the hydrodynamic description, the regime in which the laws were derived, and readings past it are labelled statements about the lattice equations.
```
Replacement:
```latex
Either failure ends the geometric reading of the fields, and readings past it are labelled statements about the lattice equations.
```

### 10.501 Sec. IV: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_05 pair 3), beginning "The boundary charge's own deviation is small on the sampled slices but"
Replaced text:
```latex
The boundary charge's own deviation is small on the sampled slices but not validated.
```
Replacement:
```latex
% provenance: lead inbox 113 (solve, q_inf on all 18 slices held and under the dm-reference law, one calculation, float64 comparator grade): the maximum is 1.0104e-3 (size 64, T/r_s 35), so "below 1e-3" holds only on the five printed slices, which reproduce to print precision; the sentence below already scopes the bound to those five, and no other sentence or comment implies it on all 18 (Sec. IV D and the Outlook do not mention q_inf); the 0.11 percent reference-law shift is not printed for lack of room.
The boundary charge's own deviation is small on the sampled slices but not validated.
```

### 10.502 Sec. VII: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_08 pair 1), beginning "The lattice fermion and the one principle, with nothing geometric adde"
Replaced text:
```latex
The lattice fermion and the one principle, with nothing geometric added, lead to
```
Replacement:
```latex
% owner comments cmw0jjt7lhe38, cmw0jkfinuec6 and cmw0k15basrtz (lead-routed 2026-11-04): lattice fermions, a Dirac fermion field, and the charge principle named.
The lattice fermions and the charge principle, with nothing geometric added, lead to
```

### 10.503 Sec. VII: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_08 pair 2), beginning "Take a single Dirac fermion field on a fixed cubic lattice, weight its"
Replaced text:
```latex
Take a single Dirac fermion field on a fixed cubic lattice, weight its energy and momentum by a clock rate and a flow velocity, and require that energy be a charge
```
Replacement:
```latex
Take lattice fermions on fixed cubic sites, a Dirac fermion field, weight their energy and momentum by a clock rate and a flow velocity, and require that energy be a charge
```

### 10.504 Sec. VII: replaced text, verbatim (owner comments cmw0jjt7lhe38, cmw0jkfinuec6, cmw0j97q2855c, cmw0jfb2o3t74, cmw0jig6nodrk, cmw0jqxyz4zbm, cmw0jusflcthw, cmw0k15basrtz, cmw0k2pq2vpb1, cmw0k5ogtc6fe, cmw0kar9upu6c and the owner follow-ups (lead-routed 2026-11-04), lead inbox 113, spec OC_08 pair 3), beginning "the principle's origin from weaker assumptions"
Replaced text:
```latex
the principle's origin from weaker assumptions
```
Replacement:
```latex
the charge principle's origin from weaker assumptions
```

### 10.505 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 1), beginning "Sec.~\ref{sec:model-gravity}. This section states the model's five dec"
Replaced text:
```latex
Sec.~\ref{sec:model-gravity}. This section states the model's five declarations, each set in bold where it is made within this section, and derives the laws from them.
```
Replacement:
```latex
Sec.~\ref{sec:model-gravity}. This section states the model's five declarations, each set in bold where it is made within this section, and derives the laws from them. Secs.~\ref{sec:model-matter} to \ref{sec:model-sources} hold exactly and in general, the matter, the clock and flow defined from the averaged matter at every step, and the vacuum subtraction. Section~\ref{sec:model-gravity} is the hydrodynamic limit, where the laws are derived and read as a geometry, and Sec.~\ref{sec:model-operators} states the quantum sector, which rests on the constraint rows as equations and not on that limit.
% owner comment (lead-routed 2026-11-04, "section 2 includes both foundations and hydrodynamic limit ... how is vacuum handled in general case?"): the section mapped as foundations (II A to II C), the limit (II D, retitled, its opener saying the laws are derived there and the computation of Sec. IV uses the resulting rows as equations), and the quantum sector (II E, retitled, its opener saying it rests on the rows and not on the limit's premise); the sector cannot precede II D, since its construction varies the spherical rows and uses their boundary charge, so the limit sits fourth of five and is labelled; the general handling of the vacuum stated at the fourth declaration; the coarse-field paragraph's restatement of the computation folded to pay.
```

### 10.506 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 2), beginning "The scale is a control whose effect on every long-wavelength result mu"
Replaced text:
```latex
The scale is a control whose effect on every long-wavelength result must be \(O(\ell^2/L_{\rm geom}^2)\), the order a smooth field changes under averaging: one even kernel \(k_{\ell,a}\) (Eq.~\eqref{eq:dm-average}, Appendix~\ref{app:charge}) smooths site densities, \(W_\ell\), and averages fields back onto the sites, \(W_\ell^\dagger\), with per-axis variance \(\sigma^2=\ell^2/15+a^2/12\). The computation extends beyond it. The computed histories of Sec.~\ref{sec:star} evolve the same lattice equations through this kernel. Past the point where the smoothness test fails the evolution is still exact and the fields well defined, but their reading as a geometry is no longer guaranteed, so results there are stated as statements about the lattice equations (Sec.~\ref{sec:singularity}). The last ingredient
```
Replacement:
```latex
The scale is a control whose effect on every long-wavelength result must be \(O(\ell^2/L_{\rm geom}^2)\), the order a smooth field changes under averaging: one even kernel \(k_{\ell,a}\) (Eq.~\eqref{eq:dm-average}, Appendix~\ref{app:charge}) smooths site densities, \(W_\ell\), and averages fields back onto the sites, \(W_\ell^\dagger\), with per-axis variance \(\sigma^2=\ell^2/15+a^2/12\). The computed histories of Sec.~\ref{sec:star} evolve the same lattice equations through this kernel beyond that regime. The last ingredient
```

### 10.507 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 3), beginning "The reference is independent of the excitation state at fixed fields a"
Replaced text:
```latex
The reference is independent of the excitation state at fixed fields and preparation. \(C\) carries the evolving one-body correlations.
```
Replacement:
```latex
In general the vacuum is this concrete state, the Dirac sea the collapse itself prepares, and the computation of Sec.~\ref{sec:star} measures the sources against it exactly at the operator level, the hydrodynamic limit only reading the subtracted sources as smooth fields. The reference is independent of the excitation state at fixed fields and preparation. \(C\) carries the evolving one-body correlations.
```

### 10.508 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 4), beginning "\subsection{The Gauss laws, the metric, and the continuum limit}\label"
Replaced text:
```latex
\subsection{The Gauss laws, the metric, and the continuum limit}\label{sec:model-gravity}\label{hyp:fc}
```
Replacement:
```latex
\subsection{The hydrodynamic limit: the Gauss laws, the metric, and the continuum limit}\label{sec:model-gravity}\label{hyp:fc}
```

### 10.509 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 5), beginning "With matter, fields, and sources declared, the laws follow. The deriva"
Replaced text:
```latex
With matter, fields, and sources declared, the laws follow. The derivation proceeds in four steps, with the fermion's own
```
Replacement:
```latex
This subsection is the hydrodynamic limit: with matter, fields, and sources declared, the laws follow at leading gradient order, where the matter varies slowly over the averaging scale, and the computation of Sec.~\ref{sec:star} uses the resulting constraint rows as equations without that premise. The derivation proceeds in four steps, with the fermion's own
```

### 10.510 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 6), beginning "The fourth asks what of this survives the continuum limit. All four wo"
Replaced text:
```latex
The fourth asks what of this survives the continuum limit. All four work in the hydrodynamic limit, the regime in which the matter varies slowly over the averaging scale, and the computed collapse of Sec.~\ref{sec:star} does not assume it.
```
Replacement:
```latex
The fourth asks what of this survives the continuum limit.
```

### 10.511 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 7), beginning "\subsection{The spherical sector: quantum form used in the computation"
Replaced text:
```latex
\subsection{The spherical sector: quantum form used in the computations}\label{sec:model-operators}
```
Replacement:
```latex
\subsection{The quantum spherical sector}\label{sec:model-operators}
```

### 10.512 Sec. II: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03 pair 8), beginning "Last comes the quantum form the computed collapse uses: the spherical "
Replaced text:
```latex
Last comes the quantum form the computed collapse uses: the spherical sector's leading Hartree term,
```
Replacement:
```latex
Last comes the quantum sector, exact and general, which rests on the constraint rows as the model's equations and not on the limit's premise. The computed collapse uses its leading Hartree term,
```

### 10.513 Sec. III: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_04 pair 1), beginning "\section{Properties of the fermion and of the vacuum exterior}\label{s"
Replaced text:
```latex
\section{Properties of the fermion and of the vacuum exterior}\label{sec:results}
```
Replacement:
```latex
\section{Properties of the fermion and the vacuum exterior in the hydrodynamic limit}\label{sec:results}
```

### 10.514 Sec. IV: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_05 pair 1), beginning "applied to the actual averaged matter and measured against the vacuum "
Replaced text:
```latex
applied to the actual averaged matter and measured against the vacuum the collapse prepares, so it stays valid
```
Replacement:
```latex
applied to the actual averaged matter and measured against the vacuum the collapse prepares (Sec.~\ref{sec:model-sources}), so it stays valid
```

### 10.515 03b: replaced text, verbatim (owner reorganization request (lead-routed 2026-11-04): Sec. II mapped as foundations, the hydrodynamic limit (retitled II D) and the quantum sector (retitled II E), Sec. III retitled, the general vacuum stated at the fourth declaration, spec RO_03b pair 1), beginning "Secs.~\ref{sec:model-matter} to \ref{sec:model-sources} hold exactly a"
Replaced text:
```latex
Secs.~\ref{sec:model-matter} to \ref{sec:model-sources} hold exactly and in general, the matter, the clock and flow defined from the averaged matter at every step, and the vacuum subtraction. Section~\ref{sec:model-gravity} is the hydrodynamic limit, where the laws are derived and read as a geometry, and Sec.~\ref{sec:model-operators} states the quantum sector, which rests on the constraint rows as equations and not on that limit.
```
Replacement:
```latex
Secs.~\ref{sec:model-matter} to \ref{sec:model-sources} and \ref{sec:model-operators} hold exactly and in general, and Sec.~\ref{sec:model-gravity} is the hydrodynamic limit, in which the laws are derived and read as a geometry.
```

### 10.516 Sec. III: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_04 pair 1), beginning "and where that exterior contains \(r_s=2GE\) the trapping radius, the "
Replaced text:
```latex
and where that exterior contains \(r_s=2GE\) the trapping radius, the temperature, and the first law below are all read from this one solution, properties of the exterior that do not by themselves establish a vacuum trapping radius or an exponential segment on the computed collapse.
```
Replacement:
```latex
and where that exterior contains \(r_s=2GE\) the trapping radius, the temperature, and the first law below are all read from this one solution, which by itself establishes no vacuum trapping radius or exponential segment on the computed collapse.
```

### 10.517 Sec. IV: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_05 pair 1), beginning "It crosses that radius at \(T/r_s=13.22\), where \(t_s\) diverges, a c"
Replaced text:
```latex
It crosses that radius at \(T/r_s=13.22\), where \(t_s\) diverges, a comparison clock and not a trajectory of either cloud, since coordinate rest is not imposed on the computed preparation, and it carries to no other size. The dust comparison solution has its own crossing clock (its arrival at the filling radius \(R_\pi\) in Ref.~\cite{McKinneyCode2026}), and since \(R_\pi\) is a comparison radius and not an observed endpoint, these clocks establish neither the computed core's arrival there nor any duration of a vacuum trapping radius. Whether outward-moving matter reaches the trapping radius in a controlled astrophysical limit is open.
```
Replacement:
```latex
It crosses that radius at \(T/r_s=13.22\), where \(t_s\) diverges, a comparison clock and not a trajectory of either cloud, since coordinate rest is not imposed on the computed preparation, and it carries to no other size. Neither it nor the dust solution's own crossing clock (Ref.~\cite{McKinneyCode2026}) establishes the computed core's arrival at the comparison radius \(R_\pi\) or any duration of a vacuum trapping radius, and whether outward-moving matter reaches the trapping radius in a controlled astrophysical limit is open.
```

### 10.518 Sec. IV: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_05 pair 2), beginning "Thus \(f_{\rm ext}=1-R_t/(2GE)\), which the rows identify with the net"
Replaced text:
```latex
Thus \(f_{\rm ext}=1-R_t/(2GE)\), which the rows identify with the net signed charge outside the trapping radius, sea departure and reference response included and not a particle fraction, is 0.04 to 0.05 at \(T/r_s=15\), while a slice without a trapping radius has no \(f_{\rm ext}\) and its source support reaches \(2GE\) \todo{
```
Replacement:
```latex
Thus \(f_{\rm ext}=1-R_t/(2GE)\), by the rows the net signed charge outside the trapping radius (sea departure and reference response included, not a particle fraction), is 0.04 to 0.05 at \(T/r_s=15\), and a slice without a trapping radius has no \(f_{\rm ext}\) \todo{
```

### 10.519 Sec. IV: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_05 pair 3), beginning "None of these readouts establishes a source-free trapping radius or lo"
Replaced text:
```latex
None of these readouts establishes a source-free trapping radius or local clearance at any clock. At every differentiable trapping radius the rows give \(\kappa_R=N_R\beta_R/(2R)\), \(\beta_R=1-2G\mathcal M'_R\), so \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\) (the clock and slope identities in Ref.~\cite{McKinneyCode2026}), and a simple outer trapping radius has \(\kappa_R>0\) and \(\beta_R>0\), values of \(\kappa_R/\kappa_H\) above one not being excluded \todo{
```
Replacement:
```latex
No readout establishes a source-free trapping radius or local clearance at any clock. The rows give \(\kappa_R=N_R\beta_R/(2R)\), \(\beta_R=1-2G\mathcal M'_R\), so \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\) (Ref.~\cite{McKinneyCode2026}), and a simple outer trapping radius has \(\kappa_R>0\) and \(\beta_R>0\), values above one not excluded \todo{
```

### 10.520 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 1), beginning "On the dust comparison solution the exact map's first-order correction"
Replaced text:
```latex
On the dust comparison solution the exact map's first-order correction to the received exponential is a fractional-linear deformation that leaves the chiral vacuum two-point function unchanged, so the conformal companion's stress flux departs from the thermal value only at second order (the coefficients, the segment thresholds, and the one-solar-mass segment in Ref.~\cite{McKinneyCode2026}), and these coefficients fix neither the occupation of the outgoing branch nor its supply, which need the evolving incident covariance at the trapping radius and a once-counted output basis.
```
Replacement:
```latex
The exact dust map's corrections to the received exponential enter the conformal companion's stress flux only at second order (coefficients and the one-solar-mass segment in Ref.~\cite{McKinneyCode2026}), and they fix neither the occupation of the outgoing branch nor its supply, which need the evolving incident covariance at the trapping radius and a once-counted output basis.
```

### 10.521 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 2), beginning "At a smooth, simple, held trapping radius the radial massless operator"
Replaced text:
```latex
At a smooth, simple, held trapping radius the radial massless operator's near-surface limit gives Fermi-shaped two-channel transmission at \(\kappa/2\pi\), \(\kappa\) the slope of the back-averaged fields the operator couples to, which the fine-grid slope \(\kappa_R=N_R\beta_R/(2R)\) of Sec.~\ref{sec:star} replaces only with the back-averaging shift controlled, a local scattering result that fixes neither incoming occupation, sustained supply, escape yield on evolving fields, nor heavy-information transfer. A connected vacuum exterior containing the trapping radius gives \(\kappa_R=\kappa_H\), equality of the two slopes not by itself establishing vacuum, and the shell-flow comparisons at the trapping-radius slope and at \(T_H(E)\) are comparison cases, neither a measured emission temperature of the evolving object. Since \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\) equals one only on a vacuum shell with unit clock, on the stored histories the slope and the received map are read from \(N_R\), \(\beta_R\), \(f_{\rm ext}\) and the complete flight through the evolving fields (the received factor \(g_D\) of Sec.~\ref{sec:star} and the dispersive-cutoff kinematics in Ref.~\cite{McKinneyCode2026}), not from the comparison constants.
```
Replacement:
```latex
At a smooth, simple, held trapping radius the radial massless operator's near-surface limit gives Fermi-shaped two-channel transmission at \(\kappa/2\pi\), \(\kappa\) the back-averaged slope, which the fine-grid \(\kappa_R\) of Sec.~\ref{sec:star} replaces only with the back-averaging shift controlled, a local scattering result that fixes neither incoming occupation, sustained supply, escape yield on evolving fields, nor heavy-information transfer. A vacuum exterior containing the trapping radius gives \(\kappa_R=\kappa_H\), equal slopes not establishing vacuum, the shell-flow comparisons at the trapping-radius slope and at \(T_H(E)\) are comparison cases and not a measured temperature, and since \(\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})\) is one only on a vacuum shell with unit clock, the stored histories' slope and received map are read from \(N_R\), \(\beta_R\), \(f_{\rm ext}\) and the complete flight (Ref.~\cite{McKinneyCode2026}), not from the comparison constants.
```

### 10.522 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 3), beginning "so at \(\dot E=0\) the exterior work vanishes exactly when the trappin"
Replaced text:
```latex
so at \(\dot E=0\) the exterior work vanishes exactly when the trapping radius stands still, motion of the trapping radius implies nonzero total exterior work, and with the boundary charge conserved and \(dE_R/du=-L\) a receding trapping radius has \(P_{\rm ext}=N_R\beta_RL\,du/dT>0\) where \(\beta_R>0\), the balance's assignment of the radiated energy to the sources outside the trapping radius, controlled by the shell remainder with storage, reference response, and the moving trapping radius's sweep included.
```
Replacement:
```latex
so at \(\dot E=0\) the exterior work vanishes exactly when the trapping radius stands still, and with the charge conserved and \(dE_R/du=-L\) a receding trapping radius has \(P_{\rm ext}=N_R\beta_RL\,du/dT>0\) where \(\beta_R>0\), the radiated energy assigned to the sources outside it, controlled by the shell remainder (Ref.~\cite{McKinneyCode2026}).
```

### 10.523 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 4), beginning "the \(10^{-76}\)-level shift \(\delta T/T_H=1.39\times10^{-76}\) is th"
Replaced text:
```latex
the \(10^{-76}\)-level shift \(\delta T/T_H=1.39\times10^{-76}\) is the formal axial fixed-profile correction under its matching condition, not an established spherical-object temperature shift, and the evaporation scale \(t_*/r_s=1.06\times10^{78}\), \(t_*=3.3\times10^{65}\) years, follows only if the assumed luminosity of Eq.~\eqref{eq:sc-luminosity} persists, no sustained emission law or lifetime being established for the computed object (\(m/T_H\), the \(\chi\) scaling exponents, and the size-64 values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
the formal axial shift \(\delta T/T_H=1.39\times10^{-76}\) is no established spherical-object temperature shift, and \(t_*/r_s=1.06\times10^{78}\), \(t_*=3.3\times10^{65}\) years, follows only if the assumed luminosity of Eq.~\eqref{eq:sc-luminosity} persists, no emission law or lifetime being established for the computed object (\(m/T_H\), the \(\chi\) exponents, and the size-64 values in Ref.~\cite{McKinneyCode2026}).
```

### 10.524 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 5), beginning "that is the vacuum comparison temperature \(T_H(E_R)=1/(4\pi R)\) and "
Replaced text:
```latex
that is the vacuum comparison temperature \(T_H(E_R)=1/(4\pi R)\) and the channel coefficients of the thermally supplied vacuum profile, properties of the exterior solution whose identification with the evolving object requires the source, transport, and shell-remainder calculation below and which establish no vacuum trapping radius on the computed collapse, and the reduced balance
```
Replacement:
```latex
that is the vacuum comparison temperature \(T_H(E_R)=1/(4\pi R)\) and the thermally supplied vacuum profile's channel coefficients, whose identification with the evolving object requires the source, transport, and shell-remainder calculation below, and the reduced balance
```

### 10.525 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 6), beginning ", the area fraction and the height holding independently of the lumino"
Replaced text:
```latex
, the area fraction and the height holding independently of the luminosity normalization while the time fraction uses \(L=\mathcal L/R^2\) and changes with the rate law. The cut share
```
Replacement:
```latex
, the area fraction and height independent of the luminosity normalization, the time fraction not. The cut share
```

### 10.526 Sec. VI: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_06 pair 7), beginning "so conserved charge and recession do not imply zero work, clearance of"
Replaced text:
```latex
so conserved charge and recession do not imply zero work, clearance of heavy matter from the trapping radius does not by itself set \(N_R\beta_R=1\), and no zero-exterior-work condition is imposed.
```
Replacement:
```latex
so conserved charge and recession do not imply zero work, clearance of heavy matter does not set \(N_R\beta_R=1\), and no zero-work condition is imposed.
```

### 10.527 Sec. V: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_07 pair 1), beginning "The Hawking comparison case below is a supplied exterior, its inner po"
Replaced text:
```latex
The Hawking comparison case below is a supplied exterior, its inner port fed thermally at \(T_H\), so its rates are a comparison and not a result about the object's supply: a stationary two-port channel at a held trapping radius gives at most the comparison occupation with an empty outgoing branch and more with that branch filled (the coherent two-port form in Ref.~\cite{McKinneyCode2026}), so the comparison bounds the supply only given the incident state, which the computed collapse does not fix once formation light has passed, and the exponential segment of Sec.~\ref{sec:scaling-rays} fixes the temperature of what is converted, not the amount.
```
Replacement:
```latex
The Hawking comparison case below is a supplied exterior, its inner port fed thermally at \(T_H\), so its rates bound the object's supply only given the incident state, which the computed collapse does not fix once formation light has passed (the two-port occupation in Ref.~\cite{McKinneyCode2026}), and the exponential segment of Sec.~\ref{sec:scaling-rays} fixes the temperature of what is converted, not the amount.
```

### 10.528 Sec. V: replaced text, verbatim (lead inbox 114: the inbox 108 to 112 inserted texts compressed toward the lengths of the sentences they replaced, scope qualifiers, labels, todos and numbers kept, derivation steps to a clause plus pointer, spec T_114_07 pair 2), beginning "a shell diagnostic far below \(\mathcal R=1.64\) and not the no-return"
Replaced text:
```latex
a shell diagnostic far below \(\mathcal R=1.64\) and not the no-return output ratio \(\mathcal R\), which classifies neither emitted entropy nor thermality and fixes no alternative maximum.
```
Replacement:
```latex
a shell diagnostic far below \(\mathcal R=1.64\), not the no-return ratio, classifying neither emitted entropy nor thermality and fixing no alternative maximum.
```

### 10.529 Bibliography: removed fields, verbatim (lead's ladder item 7, the bibliography eprint fields, GO 2026-11-04)
Removed text:
```bibtex
Ansel2024:
  eprint       = {2405.02380},
  archiveprefix= {arXiv},
  primaryclass = {gr-qc},
YangDingYan2025:
  eprint  = {2511.00950},
  archiveprefix = {arXiv},
```
Replacement: none (the two entries keep their journal, volume, pages and year fields; the printed "arXiv:2405.02380 [gr-qc]" tails disappear).

### 10.530 Sec. IV: replaced text, verbatim (second selection pass (owner rule 2026-11-04, the lead ladder item 8: remaining measurement specifics to the repository), spec T_SEL2_05 pair 1), beginning "the zero-step formation lying near 11.8 to \(11.9r_s\) across the step"
Replaced text:
```latex
the zero-step formation lying near 11.8 to \(11.9r_s\) across the steps and on a charge-conserving step. The factor-4 averaging control fails on both width pairs, the three runs differing by 15 to 22 percent in radius at matched clocks (formation clocks and fractions in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
the zero-step formation near \(11.9r_s\) across the steps and on a charge-conserving step. The factor-4 averaging control fails on both width pairs (formation clocks, radii, and fractions in Ref.~\cite{McKinneyCode2026}).
```

### 10.531 Sec. IV: replaced text, verbatim (second selection pass (owner rule 2026-11-04, the lead ladder item 8: remaining measurement specifics to the repository), spec T_SEL2_05 pair 2), beginning "Hovering redshift tracks \(\int\kappa_Rdt\) with slope 0.85 to 1.25 on"
Replaced text:
```latex
Hovering redshift tracks \(\int\kappa_Rdt\) with slope 0.85 to 1.25 on seven of ten reference-collapse clocks, and blueshifted arrivals at widths of 8 and above rule out a uniform sampled redshift bound (values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
Hovering redshift tracks \(\int\kappa_Rdt\) on most reference-collapse clocks, and blueshifted arrivals at widths of 8 and above rule out a uniform sampled redshift bound (slopes and values in Ref.~\cite{McKinneyCode2026}).
```

### 10.532 Sec. IV: replaced text, verbatim (second selection pass (owner rule 2026-11-04, the lead ladder item 8: remaining measurement specifics to the repository), spec T_SEL2_05 pair 3), beginning "The Killing burst bin above \(m\) gains about 9 percent of the particl"
Replaced text:
```latex
The Killing burst bin above \(m\) gains about 9 percent of the particles over 16 to 20 against each slice's own fields, at energies just above \(m\) (values in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
The Killing burst bin above \(m\) gains a few percent of the particles over 16 to 20, at energies just above \(m\) (values in Ref.~\cite{McKinneyCode2026}).
```

### 10.533 Sec. VI: replaced text, verbatim (second selection pass (owner rule 2026-11-04, the lead ladder item 8: remaining measurement specifics to the repository), spec T_SEL2_06 pair 1), beginning "with \(C_c\) between \(-7.6\) and \(-1.4\) for \(c\) from 0.05 to 0.5 "
Replaced text:
```latex
with \(C_c\) between \(-7.6\) and \(-1.4\) for \(c\) from 0.05 to 0.5 (the exact segment and its e-fold count in Ref.~\cite{McKinneyCode2026}).
```
Replacement:
```latex
with \(C_c\) of order minus a few (its values, the exact segment, and its e-fold count in Ref.~\cite{McKinneyCode2026}).
```

### 10.534 Sec. V: replaced text, verbatim (second selection pass (owner rule 2026-11-04, the lead ladder item 8: remaining measurement specifics to the repository), spec T_SEL2_07 pair 1), beginning "growth at \(2r_s\) levels at 1.28 to 1.40 nats per sector over 60 to 1"
Replaced text:
```latex
growth at \(2r_s\) levels at 1.28 to 1.40 nats per sector over 60 to 100, about half of it excess over the held vacuum,
```
Replacement:
```latex
growth at \(2r_s\) levels off over 60 to 100, about half of it excess over the held vacuum (levels in Ref.~\cite{McKinneyCode2026}),
```

### 10.535 Every red todo mark converted to a LaTeX comment (owner decision 2026-11-04, lead-routed), with the sentences closed where the mark was part of a sentence
The 36 marks' texts are preserved verbatim in the "% TODO[n]:" comment lines placed between paragraphs and in review_output/paperapi/TODO_LEDGER_20261104.md (number, section, placement, paragraph anchor, handling). Rendered text changed only where a mark carried the sentence:
```latex
TODO[5] (Sec. III G, owner-approved typical-subspace paragraph): "The corresponding two-field statistical temperature is \todo{two-field cut-entropy derivative}." -> "The corresponding two-field statistical temperature remains to be derived."
TODO[8] (Sec. IV B): "values above one not excluded \todo{...}." -> "values above one not excluded, pending the audit of the stored trapping radii."
TODO[11] (Sec. IV C): "The branch weight does not inherit 0.918 \todo{...} and the correlation beyond Hartree-Fock is uncontrolled." -> "The branch weight does not inherit 0.918, and the correlation beyond Hartree-Fock is uncontrolled."
TODO[14] (Sec. V B): "its value on the stored histories is \todo{...}, and identification" -> "its value on the stored histories is not yet established, and identification"
TODO[26] (Sec. VI D): "the fixed/moving equivalence \todo{...}: the outer" -> "the fixed/moving equivalence: the outer"
TODO[31] (Sec. VI D): "is not established \todo{...}" (paragraph end, no period) -> "is not established."
TODO[34] (Sec. VI D, owner-approved shell-accounting paragraph): "The received-time map and once-counted output term are \todo{received-output matching}." -> "The received-time map and once-counted output term remain to be matched."
TODO[36] (Sec. VII A): "These finite-history quantities are \todo{...}." -> "These finite-history quantities remain to be computed."
```
Also in this commit (lead inbox 116): "is 0.04 to 0.05 at \(T/r_s=15\), and a slice" -> "is 0.04 to 0.05 (plus or minus 0.03) at \(T/r_s=15\) and \(\sigma=4a\), and a slice", the width dependence recorded in the TODO[7] comment; and owner comment cmw0kx24xgp8n: "two analytic comparison solutions, stated first." -> "two analytic comparison solutions, which come next." (the only such tic; "as stated below" inside the first declaration's bold is not the tic and stays).

### 10.536 Sec. II: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_03 pair 1), beginning "\subsection{The hydrodynamic limit: the Gauss laws, the metric, and th"
Replaced text:
```latex
\subsection{The hydrodynamic limit: the Gauss laws, the metric, and the continuum limit}\label{sec:model-gravity}\label{hyp:fc}
```
Replacement:
```latex
\section{The hydrodynamic limit: Einstein's constraints and the metric}\label{sec:model-gravity}\label{hyp:fc}
```

### 10.537 Sec. II: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_03 pair 2), beginning "This subsection is the hydrodynamic limit: with matter, fields, and so"
Replaced text:
```latex
This subsection is the hydrodynamic limit: with matter, fields, and sources declared, the laws follow at leading gradient order, where the matter varies slowly over the averaging scale, and the computation of Sec.~\ref{sec:star} uses the resulting constraint rows as equations without that premise.
```
Replacement:
```latex
This section is the hydrodynamic limit: with the matter, fields, and sources of Sec.~\ref{sec:model} declared, the laws follow at leading gradient order, where the matter varies slowly over the averaging scale, and the computation of the collapse (Sec.~\ref{sec:star}) uses the resulting constraint rows as equations without that premise.
```

### 10.538 Sec. II: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_03 pair 3), beginning "\subsection{The quantum spherical sector}\label{sec:model-operators}"
Replaced text:
```latex
\subsection{The quantum spherical sector}\label{sec:model-operators}
```
Replacement:
```latex
\section{The quantum spherical sector}\label{sec:model-operators}
```

### 10.539 Sec. II: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_03 pair 4), beginning "This section states the model's five declarations, each set in bold wh"
Replaced text:
```latex
This section states the model's five declarations, each set in bold where it is made within this section, and derives the laws from them. Secs.~\ref{sec:model-matter} to \ref{sec:model-sources} and \ref{sec:model-operators} hold exactly and in general, and Sec.~\ref{sec:model-gravity} is the hydrodynamic limit, in which the laws are derived and read as a geometry.
```
Replacement:
```latex
This section states the model's first four declarations, each set in bold where it is made, all exact and general: the matter, the clock and flow defined from the averaged matter at every step, and the vacuum subtraction. Section~\ref{sec:model-gravity} is the hydrodynamic limit, in which the laws are derived from them and read as a geometry, and Sec.~\ref{sec:model-operators} states the fifth declaration, the quantum sector, which rests on the constraint rows as equations and not on that limit.
```

### 10.540 Sec. II: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_03 pair 5), beginning "Section~\ref{sec:model} is complete: five declarations, two constraint"
Replaced text:
```latex
Section~\ref{sec:model} is complete: five declarations, two constraint equations, one metric, and one quantum sector. Four declarations and the averaging scale of the third are structural inputs, while the slow-variation premise on that scale is the hydrodynamic limit in which the two constraint equations take Einstein's form.
```
Replacement:
```latex
The model is complete: five declarations, two constraint equations, one metric, and one quantum sector. Four declarations and the averaging scale of the third are structural inputs, while the slow-variation premise on that scale is the hydrodynamic limit of Sec.~\ref{sec:model-gravity}, in which the two constraint equations take Einstein's form.
```

### 10.541 Sec. I: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_02 pair 1), beginning "the quantum spherical sector on which that energy serves as the charge"
Replaced text:
```latex
the quantum spherical sector on which that energy serves as the charge (Sec.~\ref{sec:model}).
```
Replacement:
```latex
the quantum spherical sector on which that energy serves as the charge (Sec.~\ref{sec:model-operators}).
```

### 10.542 Sec. I: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_02 pair 2), beginning "Section~\ref{sec:model} states these five declarations and derives the"
Replaced text:
```latex
Section~\ref{sec:model} states these five declarations and derives the laws from them. Four declarations are structural inputs, and so is the third's averaging scale, while the slow-variation premise on that scale is the hydrodynamic limit, in which the laws take Einstein's form.
```
Replacement:
```latex
Section~\ref{sec:model} states four of these declarations as structural inputs, the third's averaging scale among them. Section~\ref{sec:model-gravity} derives the laws in the hydrodynamic limit, the slow-variation premise on that scale under which they take Einstein's form, and Sec.~\ref{sec:model-operators} states the fifth declaration, the quantum sector.
```

### 10.543 Sec. I: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_02 pair 3), beginning "Sections~\ref{sec:model} and \ref{sec:results} work in that limit. Sec"
Replaced text:
```latex
Sections~\ref{sec:model} and \ref{sec:results} work in that limit. Sections~\ref{sec:star} and \ref{sec:page} do not assume it:
```
Replacement:
```latex
Sections~\ref{sec:model-gravity} and \ref{sec:results} work in that limit. Sections~\ref{sec:star} and \ref{sec:page} do not assume it:
```

### 10.544 Sec. III: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_04 pair 1), beginning "This section works out what the constraint equations imply in the hydr"
Replaced text:
```latex
This section works out what the constraint equations imply in the hydrodynamic limit:
```
Replacement:
```latex
This section works out what the constraint equations of Sec.~\ref{sec:model-gravity} imply in the hydrodynamic limit:
```

### 10.545 Sec. IV: replaced text, verbatim (owner request (lead-routed 2026-11-04): the hydrodynamic limit as its own section, the quantum sector as its own section, pointers updated, spec RS_05 pair 1), beginning "Everything to this point, Secs.~\ref{sec:model} and \ref{sec:results},"
Replaced text:
```latex
Everything to this point, Secs.~\ref{sec:model} and \ref{sec:results}, was derived in the hydrodynamic continuum limit.
```
Replacement:
```latex
Everything analytic to this point, Secs.~\ref{sec:model-gravity} and \ref{sec:results}, was derived in the hydrodynamic continuum limit.
```

### 10.546 Sec. II: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_03 pair 1), beginning "These two equations carry the whole gravitational content of the paper"
Replaced text:
```latex
These two equations carry the whole gravitational content of the paper: the second step identifies them with Einstein's constraints, and the exterior, the temperature, the entropy, and the computed collapse all solve them and nothing else, with \(G\) the only constant.
```
Replacement:
```latex
These two equations carry the whole gravitational content of the paper: the second step identifies them with Einstein's constraints, the exterior, temperature, and entropy follow from them in the limit, and the computed collapse solves them at every step, with \(G\) the only constant.
```

### 10.547 Sec. II: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_03 pair 2), beginning "it is at long wavelength the covariant conservation of the source tens"
Replaced text:
```latex
it is at long wavelength the covariant conservation of the source tensor in the line element on the flows the paper computes (Secs.~\ref{sec:results-current} and \ref{sec:results-filling}).
```
Replacement:
```latex
it is at long wavelength the covariant conservation of the source tensor in the line element on the potential flows the paper treats (Secs.~\ref{sec:results-current} and \ref{sec:results-filling}).
```

### 10.548 Sec. III: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_04 pair 1), beginning "and its acceleration temperature, on which the conversion temperature "
Replaced text:
```latex
and its acceleration temperature, on which the conversion temperature of Sec.~\ref{sec:results-conversion} rests.
```
Replacement:
```latex
and its acceleration temperature, the flat-sea counterpart of the conversion temperature of Sec.~\ref{sec:results-conversion}.
```

### 10.549 Sec. III: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_04 pair 2), beginning "The filling bounds the matter the collapse can compress, which Sec.~\r"
Replaced text:
```latex
The filling bounds the matter the collapse can compress, which Sec.~\ref{sec:star} needs before any interior density can be read against a ceiling.
```
Replacement:
```latex
The filling bounds the matter the collapse can compress, the ceiling against which Sec.~\ref{sec:star} reads the computed interior densities.
```

### 10.550 Sec. III: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_04 pair 3), beginning "That is the ceiling a collapse meets, and the exterior it must produce"
Replaced text:
```latex
That is the ceiling a collapse meets, and the exterior it must produce comes next.
```
Replacement:
```latex
That is the ceiling a collapse meets, and the exterior the rows give any spherical source comes next.
```

### 10.551 Sec. III: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_04 pair 4), beginning "The exterior is the first geometric result, solved exactly, and everyt"
Replaced text:
```latex
The exterior is the first geometric result, solved exactly, and everything the paper says about a black hole seen from outside rests on it.
```
Replacement:
```latex
The exterior is the first geometric result, solved exactly, and every astrophysical statement about a black hole seen from outside (Sec.~\ref{sec:scaling}) rests on it.
```

### 10.552 Sec. IV: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_05 pair 1), beginning "The computed histories begin in this section, after the analytic thres"
Replaced text:
```latex
The computed histories begin in this section, after the analytic threshold they must exceed.
```
Replacement:
```latex
The computed histories begin in this section, with the analytic threshold their clouds lie above.
```

### 10.553 Sec. IV: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_05 pair 2), beginning "Which modes are thrown outward, the last analytic ingredient, follows "
Replaced text:
```latex
Which modes are thrown outward, the last analytic ingredient, follows from Sec.~\ref{sec:results-cone}:
```
Replacement:
```latex
Which modes are thrown outward, the last analytic reading, follows from Sec.~\ref{sec:results-cone}:
```

### 10.554 Sec. IV: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_05 pair 3), beginning "The lapse law holds on the computed interior to a few percent: Eq.~\eq"
Replaced text:
```latex
The lapse law holds on the computed interior to a few percent: Eq.~\eqref{eq:sc-clock-exact} integrated on the fine-grid fields matches the clock's fall
```
Replacement:
```latex
The lapse law, an identity of the solved rows, holds on the computed interior to a few percent: Eq.~\eqref{eq:sc-clock-exact} integrated on the fine-grid fields reproduces the clock's fall
```

### 10.555 Sec. IV: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_05 pair 4), beginning "Finally the computed geometry is checked against the first law, the st"
Replaced text:
```latex
Finally the computed geometry is checked against the first law, the stress it should obey, and the direct-state budget of Sec.~\ref{sec:results-vacuum}.
```
Replacement:
```latex
Finally the computed clock and flow are checked against the first law, the stress they should obey, and the direct-state budget of Sec.~\ref{sec:results-vacuum}.
```

### 10.556 Sec. VI: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_06 pair 1), beginning "So at the computed size the exponential segment is shorter than one th"
Replaced text:
```latex
So at the computed size the exponential segment is shorter than one thermal period and carries half a nat, which is why the computed collapse cannot show Hawking emission, while at one solar mass
```
Replacement:
```latex
So at the computed size the exponential segment is shorter than one thermal period and carries half a nat, consistent with the computed collapse resolving no Hawking-rate segment (Sec.~\ref{sec:page}), while at one solar mass
```

### 10.557 Sec. VI: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_06 pair 2), beginning "The received redshift grows at Hawking's exponent until dispersion end"
Replaced text:
```latex
The received redshift grows at Hawking's exponent until dispersion ends the long-wave description:
```
Replacement:
```latex
On the dust solution the received redshift grows at Hawking's exponent until dispersion ends the long-wave description:
```

### 10.558 Sec. VI: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_06 pair 3), beginning "The answer, then, is that the surface's light arrives with a redshift "
Replaced text:
```latex
The answer, then, is that the surface's light arrives with a redshift growing as
```
Replacement:
```latex
The answer, for the dust comparison solution, is that the surface's light arrives with a redshift growing as
```

### 10.559 Sec. VI: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_06 pair 4), beginning "To an external observer, then, the object is Schwarzschild's exterior "
Replaced text:
```latex
To an external observer, then, the object is Schwarzschild's exterior radiating at \(T_H=6\times10^{-8}\,\mathrm K\), the lattice entering
```
Replacement:
```latex
To an external observer, then, the exterior at one solar mass is Schwarzschild's, radiating at \(T_H=6\times10^{-8}\,\mathrm K\) on its exponential segment, the lattice entering
```

### 10.560 Sec. VII: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_08 pair 1), beginning "The crossing delay is only logarithmic in the spacing, Eq.~\eqref{eq:s"
Replaced text:
```latex
The crossing delay is only logarithmic in the spacing, Eq.~\eqref{eq:star-crossing-delay}, so the surface is permeable on the time scale of a light crossing.
```
Replacement:
```latex
On the vacuum exterior the crossing delay is only logarithmic in the spacing, Eq.~\eqref{eq:star-crossing-delay}, so such a surface is permeable on the time scale of a light crossing.
```

### 10.561 Sec. VII: replaced text, verbatim (hydrodynamic-confusion sweep (paperapi-hydrosweep c339728bd4a, HYDRO_CONFUSION_SWEEP_20261104.md items 4 to 19; items 1 to 3 superseded by the owner-comment commit c6fd1d7a59b), lead-routed 2026-11-04, spec HS_08 pair 2), beginning "To an external observer the exterior is Schwarzschild's with no post-N"
Replaced text:
```latex
To an external observer the exterior is Schwarzschild's with no post-Newtonian departure within the leading laws,
```
Replacement:
```latex
To an external observer the derived exterior is Schwarzschild's with no post-Newtonian departure within the leading laws,
```

### 10.562 Sec. VII: replaced text, verbatim (owner comments cmw0la0obo1xf (regularization and toy model), cmw0lio5hdqvr (no lattice-scale law for the geometry), cmw0lovgbdkqy (the Outlook cut to the big issues), lead-routed 2026-11-04, spec OL_08 pair 1), beginning "is the spacetime itself, not an effective metric for excitations insid"
Replaced text:
```latex
is the spacetime itself, not an effective metric for excitations inside a spacetime that was put in.
```
Replacement:
```latex
is the spacetime itself, not an effective metric for excitations inside a spacetime that was put in. Nor is it a regularization model: the only scale put in is the lattice spacing, the first law then fixes that spacing at about two Planck lengths (the match \(f=1\) of Sec.~\ref{sec:results-firstlaw}), no term is added by hand to tune anything, and the lattice corrections to the exterior are computed rather than absorbed, so the discreteness is a physical scale and not an arbitrary cutoff. Nor is it a toy model. Its inputs are a few straightforward principles about ordinary objects, fermions on fixed sites, energy behaving as a charge as it must for general relativity, and a vacuum to measure against, and everything else in the paper is derived from them or computed with them. The object computed is the full quantum collapse of that matter at Hartree order, not a model of one.
% owner comment cmw0la0obo1xf (lead-routed 2026-11-04): the regularization-model and toy-model questions answered beside the not-an-analogue sentence, in the paper's own facts.
```

### 10.563 Sec. VII: replaced text, verbatim (owner comments cmw0la0obo1xf (regularization and toy model), cmw0lio5hdqvr (no lattice-scale law for the geometry), cmw0lovgbdkqy (the Outlook cut to the big issues), lead-routed 2026-11-04, spec OL_08 pair 2), beginning "% fold into text: no; stated in Sec. VII A Three groups of questions r"
Replaced text:
```latex
% fold into text: no; stated in Sec. VII A
Three groups of questions remain open. The first is what the fixed lattice, with no tensor modes, cannot yet give: tensor propagation, the full deformation algebra of the constraints, a lattice-scale law for the geometry, and the laws' coupled realization beyond spherical symmetry. The second is the coupled continuation beyond the hydrodynamic regime: uniform persistence of the regular domain between the stored time slices, decided by the between-slice bound of Sec.~\ref{sec:singularity}. No endpoint, halting mechanism, redshift floor, or lattice-scale interior structure is established, so a Pauli-supported interior remains a possible outcome, not an assumption. The third is the evaporation: whether radiation is sustained after the formation transients, how much partner correlation is transferred, how fast, and what entropy stays behind, and any evaporation, Page time, or turnover of the computed object, decided first by the two-field re-evolution under the reference prescription. Also open are the directly coupled source prescription (Ref.~\cite{McKinneyCode2026}), the charge principle's origin from weaker assumptions, and controlled reduced quantum evolution through an evaporation endpoint. Neither the temperature nor the area entropy here counts states:
```
Replacement:
```latex
% fold into text: no; stated in Sec. VII A
% owner comments cmw0lio5hdqvr and cmw0lovgbdkqy (lead-routed 2026-11-04): the Outlook cut to the big open issues (tensor propagation and the laws beyond spherical symmetry, the full deformation algebra, beyond leading gradient order, the state count); "a lattice-scale law for the geometry" removed, since geometry is a reading of the fields in the hydrodynamic limit and needs no law of its own at the lattice scale; the Pauli-supported-interior sentence removed; the persistence, evaporation, Page-curve, re-evolution, source-prescription and principle-origin items dropped here, each stated in the body with its mark (Secs. VI D, VII C to VII E, VIII D); no TODO comment lived only in the Outlook. Old Outlook 263 words.
Three questions remain open above the others. The first is dynamics beyond spherical symmetry, which the fixed lattice without tensor modes cannot yet give: tensor propagation, the laws' coupled realization on general flows, and with them the full deformation algebra of the constraints, whose Dirac-Schwinger relations are not realized on fixed coordinates. The second is the order of the derivation: the laws hold at leading gradient order, and what the charge principle fixes beyond it, where the fields' reading as a geometry ends, is not worked out. The third is the deepest. Neither the temperature nor the area entropy here counts states:
```

### 10.564 Sec. IV: replaced text, verbatim (owner comments cmw0la0obo1xf (regularization and toy model), cmw0lio5hdqvr (no lattice-scale law for the geometry), cmw0lovgbdkqy (the Outlook cut to the big issues), lead-routed 2026-11-04, spec OL_05 pair 1), beginning "The leading laws supply no lattice-scale geometric law once their grad"
Replaced text:
```latex
The leading laws supply no lattice-scale geometric law once
their gradient expansion fails, so a Pauli-supported interior
remains a possible outcome, not an assumption that closes the
continuation question.
```
Replacement:
```latex
Once their gradient expansion fails the leading laws no longer read as a geometry, so a Pauli-supported interior remains a possible outcome, not an assumption that closes the continuation question.
```

### 10.567 Appendix B: the explicit averaging-kernel display (box function B_a, quintic C^4 kernel K_ell with normalization 9009/(1024 pi ell^3), k_{ell,a}=K_ell*B_a, the averaging map W_ell and its adjoint) and its lead-in sentence (owner, 2026-11-04, "Reduce appendix if can", the kernel display is "just basic math"; lead-routed; kept here: a one-sentence statement of the properties used, a one-line display of W_ell and its adjoint carrying the label eq:dm-average cited from Sec. II D, and the variance and Taylor-bound sentence)

```latex
The averaging kernel of Sec.~\ref{sec:model-dictionary} is the cell function convolved with a normalized \(C^4\) bump,
\begin{equation}
\begin{aligned}
B_a(z)&=a^{-3}\mathbf1_{[-a/2,a/2]^3}(z),\\
K_\ell(z)&=\frac{9009}{1024\pi\ell^3}(1-|z|^2/\ell^2)_+^5,\\
k_{\ell,a}&=K_\ell*B_a,\\
(W_\ell F)(y)&=a^3\sum_xk_{\ell,a}(y-x)F_x,\\
(W_\ell^\dagger F)_x&=\int d^3y\,k_{\ell,a}(y-x)F(y),
\end{aligned}
\label{eq:dm-average}
\end{equation}
```

### 10.568 Appendix A.2: the finite-volume argument for convexity of the planar coefficient (Gaussian strong subadditivity making the cut-entropy combination submodular, its Lovasz extension convex and converging to Phi) (owner, 2026-11-04, "Reduce appendix if can"; lead-routed; kept here: the support-function statement with Ref. Schneider2014, convexity from Gaussian strong subadditivity with Ref. Koteljanskii1950 and the repository pointer, the axial bounds Eq. B-axial)

```latex
\(\mathsf K\)~\cite{Schneider2014}. The finite-volume limit follows from Gaussian strong subadditivity~\cite{Koteljanskii1950}, which makes \(\tfrac12[S_A+S_{{\rm cube}\setminus A}-S_{\rm cube}]\) submodular in \(A\), so that its Lov\'asz extension~\cite{Lovasz1983} is convex and converges to \(\Phi\) (Ref.~\cite{McKinneyCode2026}).
```

### 10.569 Appendix C: the residual relations paragraph with its two intermediate relations 4 pi G N eps_E = K_t R_r + (K_r+K_t) R_t and 4 pi G N eps_M = R_t' + (R_t-R_r)/r (owner, 2026-11-04, "Reduce appendix if can"; lead-routed; kept here: the definitions, the weighted-defect relation (r^2 v R_t)' used by Appendix B and Sec. II F, the regular-centre statement, and the zero-divergence relation for R_r)

```latex
\paragraph{The residual relations.}
With \(R_r\), \(R_t\) the radial and transverse parts of Eq.~\eqref{eq:dm-evolution}'s left side minus its right side and \(\epsilon_E\), \(\epsilon_M\) the energy and momentum parts of the source tensor's divergence per unit proper time, eliminating sources by the spherical rows and differentiating gives, at each time, \(4\pi GN\epsilon_E=K_tR_r+(K_r+K_t)R_t\) and \(4\pi GN\epsilon_M=R_t'+(R_t-R_r)/r\), hence \((r^2vR_t)'=4\pi Gr^2N(N\epsilon_E+v\epsilon_M)\): \(r^2vR_t\) is the cumulative weighted defect plus its value at an inner matching radius, which a regular centre sets to zero. At zero divergence \(R_r=-(K_r+K_t)R_t/K_t\).
```

### 10.570 Sec. VI C: the exchange-single growth clause (single ratio growing with channel count through the propagated sea's exchange mismatch with the fixed reference projector, faster than the copy weights 16k) replaced by the earned statement of lead inbox 119 (reviewer B xsingleX, AstraAPI_xsingleX_v1.md md5 6516e295, sec. b; sources: solve xsingle route A, alt XSINGLE_B_RESULT_v1, solve kmax s11 f10dec49, hsplit s11k10 b8cc24e3); the old growth mechanism was an overclaim, since channels above k=10 use rates-off adiabatic seas; also removed in the same paragraph for room: the restated pointer "(endpoints and perturbation radii in Ref. McKinneyCode2026)" after the q_inf endpoints sentence

```latex
The single ratio grows with channel count through the propagated sea's exchange mismatch with the fixed reference projector, faster than the copy weights \(16k\), so no channel limit is established on the chains (values in Ref.~\cite{McKinneyCode2026})
```
