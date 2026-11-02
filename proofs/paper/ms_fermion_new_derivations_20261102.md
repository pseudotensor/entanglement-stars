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
full by /home/jon/qi2_fable_20261013/deliveries/lead_gaps/SINGULARITY_SUBSECTION_agreed_v1.tex
(md5 0e27e004348939ddb7b3328c082a5fe3), agreed by Fable and Astra; the text stands until the M1
certificate lands. The replaced subsection text, including the one-sentence continuation
statement and its todo whose full specification is section 5 above, the one-sentence
polarization statement (section 3 above) and the pre-today paragraphs (definition, exclusion
bound, three-level continuation, computed verification, clock floor), is recoverable verbatim
from git: commit cced475d5e6, fermion_new/05_star.tex, from the subsection heading to the line
before "Diagnostics of the computed geometry". The new text carries no todo mark; it prints a
limitation sentence instead.
