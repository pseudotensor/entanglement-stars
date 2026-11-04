## Replacements

**A.** Accept; set as a plain sentence (bold marks the five declarations only), mark text as given.

**B.** Accept all three; the component's operator is named as its referent, Wentzel-Kramers-Brillouin is spelled out (no undefined acronyms), the continuation is named where "the stated continuation" had no antecedent, and "part of V" is written as the scattering calculation marked below.

**C.** Accept; the diagnostic keeps its Sec. page reference.

**D.** Accept; the instruction is written as the pending condition carrying the mark, and no inferred charge is printed.

**E.** Accept.

**F.** Accept; the abstract stays at 293 words.

**Labels.** Proved: accept. Derived: accept. Reported: partial rebuttal: $\tau_k=\Gamma_k$ moves to Assumed as asked, but the Fermi-shaped local conversion ratio stays listed as reported with its scope (not flux-normalized weights), because the block and the retained Sec. VI C sentence print it as reported in Ref. McKinneyCode2026 on two routes and the label list must match the text; the $f$ enclosure signed as taken from Sec. results-vacuum, not independently certified in this section. Assumed: accept. Outstanding tests: signed.

## Final paper text (page-neutral, to be verified by compiling in the same layout)

**Instructions.** (i) replaces the subsection title and the eleven paragraphs from "A Page curve tracks the no-return entropy" through "…not substitution in Eq.~\eqref{eq:page-effective-maximum}." in Sec. VI D; keep Fig. page-conditional with (v); keep everything from "The firewall question…" onward. Page neutrality is decided by comparing page breaks after same-layout compilation; if a break moves, the designated trims are, in order, the arrival-clock sentence of paragraph four (from "The arrival clock is a consistency check" to "actual packet transfer") and the sentence "The test light of Sec.~\ref{sec:page} omitted its own gravity, so no debit is realized in those histories.", nothing else.

### (i) Replacement block

```latex
\subsection{The emission law and the Page curve}
\label{sec:page-effective}

A Page curve tracks the no-return entropy $S_{\rm esc}$ of the
radiation against the clock at infinity~\cite{Page1993}. The model
fixes the emission without a law placed by hand. The light field's
quadratic Hartree generator induces a Gaussian unitary; its
one-particle propagator $\mathsf U$ from preparation carries the
prepared sea to the pure covariance $C_L=\mathsf UP_\star\mathsf
U^\dagger$. Let $D_u$ contain orthonormal, once-counted outgoing
packets of both energy signs, received through retarded time $u$
and represented on a common slice with the earlier packets
retained; their identification with no-return radiation assumes
that return contamination from beyond the receiver and omitted-mode
errors are controlled. Then
\begin{equation}
n(\phi)=\|P_\star\mathsf U^\dagger\phi\|^2,\qquad
A_u=D_u^\dagger C_LD_u,\qquad
S_{\rm esc}(u)=\Tr s(A_u),
\label{eq:page-emission-law}
\end{equation}
the occupation of a packet being the weight of the prepared sea in
the packet traced back to preparation, and a negative-energy packet's
vacancy $1-n$ its hole occupation. Both are identities of the Hartree
evolution, Gaussian state counting under the model's own Gaussian
unitary, with no subtraction. A block $B$ received after $u$ adds
$S(B|\mathcal E_u)=S(B)-I(B{:}\mathcal E_u)$, nonnegative when the
blocks are independent and smaller, or negative, when a block
purifies earlier output. On the represented channels with the root
held at $2000a$ both frozen ledgers give positive $S(B|X)$, two
routes, so no entropy-reducing arrival is certified in the records.

Three statements follow from the propagator on the exterior of
Eq.~\eqref{eq:dr-exterior}; each has its own hypotheses and none
fixes a Page maximum. First, on the exponential segment of the ray
map the emission is derived. Traced back, a channel-$k$ packet of
frequency $\omega$ splits at the centrifugal barrier near $3GE$: the
part reflected toward large radius came in from the prepared sea at
positive frequency and is empty; the part transmitted with
probability $\Gamma_k(\omega)$ hugs the surface and leaves the
formation region as the incoming amplitude
$(du/dU)^{1/2}e^{-i\omega u(U)}$, the square root being the chiral
component's weight under the map $U_H-U=C_Ue^{-\kappa_Hu}$. On
$U<U_H$ the pulled-back amplitude is proportional to
$(U_H-U)^{-1/2+i\omega/\kappa_H}$, whose regular incoming-vacuum
continuation is analytic in a half plane; projection of normalized
resolving packets then gives the negative-to-positive frequency
weight ratio $e^{-2\pi\omega/\kappa_H}$ in the controlled
exponential limit, the real exponent changing only their common
spectral factor:
\begin{equation}
n_k(\omega)=\frac{\Gamma_k(\omega)}{e^{2\pi\omega/\kappa_H}+1},
\qquad\kappa_H=\frac1{4GE},
\label{eq:page-hawking-factor}
\end{equation}
Hawking's factor with the exponent of the exterior's own ray map
(Sec.~\ref{sec:results-conversion}) and the transmission of its own
channel operator, under the adiabatic long-wavelength conditions
stated there. For the ideal stationary, regulated covariance with
this symbol the Szeg\H{o} limit gives the entropy of the packets
received over a duration $T$ as
$T\sum_k64k\int_0^\infty\frac{d\omega}{2\pi}s(n_k)+o(T)$, channel
$k$ carrying $64k$ modes per unit of $\omega/2\pi$ (eight valleys,
$2k$ magnetic states, two signs of $\kappa$, particles and holes);
the limit is invariant under basis changes within the received-mode
subspace, not under arbitrary packet selection, and its use on a
finite adiabatic segment assumes control of boundary, regulator,
channel-tail and nonstationarity errors. No packet-independent
exponential coherence decay follows; inter-arrival covariance and
mutual information require measurement \todo{C: the output
covariance on the stored histories, inter-arrival blocks,
conditional entropies and energy spectrum}. The sums are the rates
of Eq.~\eqref{eq:sc-luminosity}, two routes, $\mathcal R=1.639$ nats
per unit of energy over $T_H$. Second, on the dust benchmark
dispersion limits the continuum exponential approximation to the
estimated frequency-dependent window
$2T_d=\tfrac43r_s\ln[1/(2a\omega)]+C_cr_s$ (Sec.~\ref{sec:scaling}),
usable only when positive and long enough for a resolving packet;
the common-window, zero-matching estimates of its content, about
$0.5$ nats at $r_s/a=64$ and $12.5$ nats at one solar mass, are
illustrative, not bounds on total emission or certified
finite-window entropies. Third, the energy balance. With both
fields' averaged sources in Eq.~\eqref{eq:dr-radial} the boundary
charge is conserved (Appendix~\ref{app:charge}), and the charge
inside a collar radius $R_c$ beyond the barrier obeys the windowed
identity $N\dot{\mathcal M}(R_c)=\Tr(\bar h_{R_c}\dot C)
-\langle\bar u_{R_c},\mathcal W''\dot{\bar u}\rangle$, the state's
energy flux through the collar and the reference's response to the
field rates. Assumption, pending the full-reference balance test
\todo{B: the collar balance on a two-field full-reference history,
with reference response, exterior storage, root-gradient factor and
sea mismatch below the balance accuracy}: sea and reference
mismatch, exterior storage, the root-gradient correction and the
arrival-time conversion are controlled well enough to use
$dE_R/du=-L$, $L$ the light field's outgoing Killing-energy flux,
with the root at $R=2GE_R$, $E_R$ the charge inside it. At a fixed
collar $u=t-\Phi(R_c)$ has unit Jacobian only for the stationary
exterior and clock convention specified; evolving histories require
their actual time conversion. The quantity $\pi R^2/G$ is a
first-law quantity, not an established retained entropy or
accessible capacity; the omitted terms have not been bounded at the
balance accuracy, the vacuum collar holding for the dust benchmark
and unproved for the object's fountain. The test light of
Sec.~\ref{sec:page} omitted its own gravity, so no debit is realized
in those histories.

The segment does not supply its own continuation. Outside it the
full dispersive propagator determines the contributing incident
branches and stored inventories; the ray map alone determines
neither their occupations nor their coherences. Assuming stationary
scattering completeness and incoming vacuum, the absolutely
continuous sector has no stationary excitation flux
(Sec.~\ref{sec:page}); prepared transients and resonant inventories
may drain. A root receding at speed $\varepsilon$ because of its own
emission excites the light field at approximately second order in
$\varepsilon$, with an excitation-to-required-power ratio at most
$6.5\times10^{-5}$, which excludes self-funding only on the tested
profiles and speed range, one route \todo{C: the recession
excitation on a second route, prescribed-recession histories with a
held-history subtraction}. What the interior supplies passes the
root, which acts as a valve. Near a smooth simple root the outgoing
chiral component of the radial massless operator has the frozen
symbol $(N/a)\sin x\,(1-|w|\cos x)$, $x=ap$, $w=v/N$. Neglecting
its coupling to the ingoing component, through $k/r$, and higher
layer terms gives the cubic approximation
$\omega=\kappa_R(r-R)p+c_3p^3$, $c_3=N_Ra^2/2$. In the interior
Wentzel-Kramers-Brillouin overlap region the two outward-moving
branches approach $x=\pm x_c$, with $\cos x_c=1/|w|$, as
$\omega\to0$; these are not their exact positions at positive
$\omega$. For the symmetrically ordered, isolated cubic-layer
model, $\tilde\psi(p)\propto p^{-1/2-i\omega/\kappa_R}
e^{ic_3p^3/(3\kappa_R)}$, and continuing its power of $p$ around
$p=0$ on the side regular outside the root gives the formal
two-branch factor $e^{-2\pi\omega/\kappa_R}$. In a flux-normalized
two-port reduction, unitarity fixes the weight sum to $\tau_k$;
establishing that reduction and its physical weights for the full
radial operator is part of the scattering calculation marked below.
The resulting Fermi-shaped local two-channel conversion at
$\kappa_R/(2\pi)$ is reported in Ref.~\cite{McKinneyCode2026} on
fixed and on flow-carried sites, but its flux-normalized weights and
the fixed/moving equivalence are not established by that derivation.
Assumptions, pending the full scattering and incident-covariance
calculations \todo{V: the flux-normalized scattering matrix from
root layer to barrier with every propagating branch, and the
moving-site operator related to the fixed-site one by its
time-dependent unitary}: the outer contributing inputs are empty;
exactly these two root-side ports contribute; their relevant
coherences vanish; and the fully matched weights, including survival
of the local conversion ratio and the identification
$\tau_k=\Gamma_k$, are assumed to be
$w_{k\pm}=\Gamma_ke^{\pm\pi\omega/\kappa_R}/[2\cosh(\pi\omega/\kappa_R)]$.
For $0\le\nu_{k\pm}\le1$ these assumptions give
\begin{equation}
n_k(\omega,u)=w_{k+}\nu_{k+}+w_{k-}\nu_{k-}\le\Gamma_k(\omega),
\qquad
w_{k-}=\frac{\Gamma_k(\omega)}{e^{2\pi\omega/\kappa_R}+1}.
\label{eq:page-valve}
\end{equation}
Under them $n_k\le\Gamma_k$, and $(\nu_+,\nu_-)=(0,1)$ gives
Hawking's occupation, the incident state a regular prepared sea
supplies when both branches trace back adiabatically to it, as on
the controlled segment. Exciting the other port can increase the
occupation up to $\Gamma_k$, but the reported formation-burst energy
and entropy diagnostics do not establish that branch occupation or
mechanism.

The histories report entropy diagnostics and collar flux, not
incident occupations. At the $16r_s$ cut they report a channel-one
statistic of about 0.473 nats per copy by $T/r_s=100$, two routes,
a rise at 42--44 coincident with the collar flux burst of
Fig.~\ref{fig:page-v4}(b), and a whole-field exterior excess of
about 8.8 nats. The illustrative 0.5-nat common-window estimate
neither bounds the segment's contribution to that excess nor
establishes its emission mechanism. The excess-per-energy diagnostic
of Sec.~\ref{sec:page}, 20 to 80 times below the thermal
$\mathcal R$, likewise does not identify the mechanism.
Identification of the channel-one statistic with the unsubtracted
$S_{\rm esc}$ requires its packet basis and comparator to be
documented. The reported late slope over 80--100, 0.030--0.040 nats
per nominal $r_s$, is 0.28--0.37 of the nominal benchmark
$0.1079/r_s$. The separate 0.23--0.31 normalization of
Sec.~\ref{sec:page} remains unverified until the record's collar
charge at 100 and the benchmark rate, clock and units actually used
are printed beside it \todo{C: ledger conventions and
normalization}. The arrival clock is a consistency check, not an
independently predicted jump: for a stationary exterior of charge
radius $r_s$, formation at 12.1, the exterior flight of $23.2r_s$
from the root to $16r_s$ with its root logarithm replaced by the
crossing delay $2r_s\ln[1/(2a\omega)]$ of
Eq.~\eqref{eq:star-crossing-delay} gives $T/r_s=40.8$--44.1 for
$\omega r_s=2$--0.4, with an undetermined $O(r_s)$ matching term,
and at the cut $u=T-26.2r_s$; evolving or dispersive histories
require their actual packet transfer.

For the Page curve we take an idealized depletion comparator with
these assumptions: initially uncorrelated output with
$S_{\rm esc}(0)=0$; the two-port depletion law $\nu_{k+}=0$,
$\nu_{k-}=\bar\nu(u)$, so $n_k=\bar\nu n_k^H$ with $0\le\bar\nu\le1$
independent of channel and frequency; a self-similar transmission
spectrum $\Gamma_k(\omega R)$; the vacuum exterior's $R=2GE_R$ and
$\kappa_R=1/(2R)$; and the reduced balance $dE_R/du=-\bar L$,
$\bar L=\bar\nu\mathcal L/R^2$. With $n_k^H=\Gamma_k/(e^{\omega/T_H}+1)$,
$\mathcal R_\nu=T_H\sum_k64k\int\frac{d\omega}{2\pi}s(\nu n_k^H)
\big/(\nu L_H)$ and the budget integrand continued by its limit,
zero, at zero supply, the emitted energy and the marginal budget are
supply integrals,
\begin{equation}
\begin{aligned}
R^3&=R_0^3-6G\mathcal L\!\int_0^u\!\bar\nu\,dv,\qquad
y\equiv\Big(\frac{E_R}{E_0}\Big)^2=\frac{R^2}{R_0^2},\\
\mathcal B(u)&=\int_0^u\frac{\mathcal R_{\bar\nu(v)}\bar L(v)}{T_H(v)}\,dv
=\int_{S(E_R(u))}^{S_0}\mathcal R_{\bar\nu}\,dS,\\
\frac{S_{\rm esc}}{S_0}&\le\min\Big\{\frac{\mathcal B(u)}{S_0},\;f\,y(u)\Big\}+o(1),
\end{aligned}
\label{eq:page-effective-curve}
\end{equation}
Here $S_0=\pi R_0^2/G$, and $u=0$ is the comparator's initial output
slice, placed at the first arrival of root-emitted light,
$O(R_0\ln(1/\chi_0))$ after the matter crosses $r_s$ by the crossing
delay of Eq.~\eqref{eq:star-crossing-delay} on the dust benchmark;
its identification with an actual collapse-arrival time requires
the packet-transfer calculation, and the remainder refers to
$\chi_0=a/R_0\to0$ at fixed $ma$, $G/a^2$ and $\eta_E$.
Subadditivity gives
$S_{\rm esc}\le\mathcal B_{\rm pkt}\equiv\sum_{i\in D_u}s((A_u)_{ii})$
in any orthonormal basis of the same received-mode subspace. An
additional comparator assumption is that a resolving basis exists
for which $\mathcal B_{\rm pkt}=\mathcal B+o(S_0)$ uniformly over
the history where the envelope is used; a slowly varying-symbol
derivation of it requires control of supply variation, packet
boundaries, regulators, channel tails and the endpoint region. The
covariance calculation must evaluate
$\sup_u|\mathcal B_{\rm pkt}(u)-\mathcal B(u)|/S_0$, including
omitted-mode errors, and test its convergence to zero in the stated
scaling limit \todo{C: uniform packet-budget discrepancy}. The
second branch assumes the stated retained capacity. Purity equates
$S_{\rm esc}$ with the entropy of the actual complement and supplies
no area-sized accessible-sector bound: original heavy particles give
$O(N_B\ln(1/\chi_0))$ nats, about $10^{-36}S_0$ at one solar mass,
and the partners at negative Killing energy that turn inside the
root (Sec.~\ref{sec:star}) fill the light field's pockets, of count
$O((R/a)^3)$, whose Gibbs ceilings at fixed energy on the vacuum
profile are 0.27--0.31 of $fA_R/(4G)$, $f=4\eta(ma)G/a^2$, at
$R/a=32$--256, two routes, and 0.08--0.10 on the stored slices,
scaling with area while the unrestricted pocket entropy grows with
volume; these finite-size estimates establish neither an asymptotic
accessible capacity $fA_R/(4G)$ nor its dynamical attainability.
The assumptions are therefore: a root-carried store of logarithmic
dimension at most $fA_R/(4G)$ with $o(S_0)$ retained entropy outside
it; independent pairs with partners stored until it saturates; then
release carrying the partner correlations to the output as the root
recedes, complete at the end; heavy transitions funding every
operation, each escaping energy debited once, apparatus entropy at
most 2000 nats and unintended output $o(E_0)$, $o(S_0)$; and the
capacity-saturating history that turns the bound in
Eq.~\eqref{eq:page-effective-curve} into equality. Under a joint
generator conserving each heavy-copy number, tracing out the heavy
field removes sensitivity to coherence between different
conserved-number sectors, not necessarily coherence within a sector.
A charge-matched code removes only that obstruction and supplies no
coherent heavy-to-light transfer mechanism; these assumptions do not
establish recovery of arbitrary heavy-input coherence under product
Hartree evolution \todo{the partner ledger: the accessible-sector
bound from pocket occupation, deposition and drift, including
omitted sectors; no-return conditional entropies, residual retained
entropy and coherent heavy-input recovery; the windowed root energy
budget and branch errors; the first controlled blocks at sizes 32
and 64}. A finite-time crossing requires a finite $u_P$ with
$\mathcal B(u_P)=fy(u_P)S_0$ under the supply-integral evolution of
$R$; the supply condition
$6G\mathcal L\int_0^\infty\bar\nu\,dv\ge R_0^3(1-y_P^{3/2})$ is
necessary, not sufficient, and establishes no saturation of the
entropy bound. For constant $0<\bar\nu\le1$, under the assumed
capacity-saturating minimum law, the branches meet at
\begin{equation}
y_P=\frac{\mathcal R_{\bar\nu}}{\mathcal R_{\bar\nu}+f},\quad
\frac{S_P}{S_0}=\frac{f\mathcal R_{\bar\nu}}{\mathcal R_{\bar\nu}+f},\quad
\frac{u_P}{t_*}=1-y_P^{3/2},\quad
t_*=\frac{R_0^3}{6G\mathcal L\bar\nu},
\label{eq:page-effective-maximum}
\end{equation}
and for variable supply the crossing solves
$\mathcal B(u_P)=fy(u_P)S_0$; if a finite evaporation time $t_*$
exists,
$u_P/t_*=(1-y_P^{3/2})\langle\bar\nu\rangle_{0:t_*}/\langle\bar\nu\rangle_{0:u_P}$
with $y_P=y(u_P)$, and otherwise no lifetime fraction is defined. At
$\bar\nu=f=1$ the comparator peaks at $0.6211S_0$ and $0.5105t_*$,
$t_*=r_{s0}^3/(6G\mathcal L)$: the assumed minimum-capacity
comparator with Hawking-rate feed, the feed the incoming vacuum
would supply through Eq.~\eqref{eq:page-valve} under its
assumptions, not an entropy history determined by that feed alone.
Concavity of $s$ gives $\mathcal R_\nu\ge\mathcal R$; for the dilute
benchmark occupations
$\mathcal R_\nu\simeq\mathcal R+(T_H\dot n/L)\ln(1/\nu)$ with the
one-route number rate, so constant quarter supply gives
$\mathcal R_{1/4}\simeq1.97$, height $0.66S_0$ at $0.46$ of a
lifetime four times longer \todo{Q: $\mathcal R_\nu$ by quadrature on
the benchmark transmissions, two routes}. At $a=2\ell_P$, $ma=0.2$,
nominal $f=0.887$ gives $0.5755S_0$ at $0.4773t_*$, and the enclosure
of Sec.~\ref{sec:results-vacuum}, $0.7883<f<0.9831$, gives heights
$[0.5323,0.6146]S_0$ at saturation; the finite-band
$\mathcal L\simeq0.0052373$, $\mathcal R\simeq1.6391$ have no narrow
enclosure. These are comparator values under the emission, balance,
capacity and release assumptions above, not a computed maximum; the
records through $T/r_s=100$, $2.5\times10^{-4}$ of the saturated
lifetime at this size, fix neither the late supply nor the turnover
\todo{$\bar\nu(u)$ beyond the records: the thin-shell asymptotic
expansion nearest the trapping root at astrophysical $\chi$,
received redshift with lapse and flow, matched to the stored
histories}. Figure~\ref{fig:page-conditional} plots the saturated
comparator; panel (b) adds the caption's assumptions.
```

### (ii) Sec. VI B, "Emission and held-field controls"

Replace "Thermal marginals ensure neither monotone accumulated entropy nor a Page curve; $S_{\rm esc}$ is uncomputed." by:

```latex
Thermal marginals ensure neither monotone accumulated entropy nor a
Page curve; $S_{\rm esc}$ is defined from the output covariance in
Sec.~\ref{sec:page-effective}, and identification of a reported
ledger with it requires documented packet and subtraction
conventions.
```

### (iii) Abstract

Replace the two sentences "A second, massless Dirac field with the same lattice operator develops outgoing flow on the computed collapse. At stellar mass, with emission, storage and release laws assumed, the radiation entropy peaks near half the evaporation time." by:

```latex
A second, massless Dirac field with the same lattice operator
develops outgoing flow on the computed collapse, its computed
exterior-entropy diagnostic still rising at the last record; its
propagator defines emission and gives Hawking's factor on a
controlled exponential ray-map segment. With saturated supply,
reduced mass balance, the benchmark area-sized capacity,
capacity-saturating loading and complete release assumed, a
comparator, not the computed history, peaks near half its
evaporation time.
```

### (iv) Introduction map and Discussion

Introduction: "…and the Page curve that follows once emission, storage and release laws are assumed;" $\to$ "…and the radiation entropy defined by the light field's own emission, beside a comparator whose supply, balance, capacity, loading and release are assumed;".

Discussion: "…and Sec.~\ref{sec:page-effective}'s curve is a benchmark under assumed laws." $\to$ "…and Sec.~\ref{sec:page-effective} defines the radiation entropy from the full output covariance, its minimum-capacity curve a conditional comparator, not a computed object history."

### (v) Fig. page-conditional, first caption sentence

"Leading-order comparison curve from the assumed laws, not an object history." $\to$ "Comparator under the saturated-supply, balance, store and release assumptions of Eq.~\eqref{eq:page-effective-curve}, not an object history; the records measure neither its lifetime nor its turnover."

## Final status labels

**Proved:** $n(\phi)=\|P_\star\mathsf U^\dagger\phi\|^2$, $S_{\rm esc}=\Tr s(A_u)$, the increment $S(B|\mathcal E_u)$ and $S_{\rm esc}\le\mathcal B_{\rm pkt}$ in any orthonormal basis of the received-mode subspace; boundary-charge conservation and the windowed collar identity; $n\le\tau_k$ for a root-fed mode under incoming vacuum within the two-port reduction with empty outer inputs; the envelope $S_{\rm esc}\le\min\{\mathcal B_{\rm pkt},fyS_0\}+o(S_0)$, conditional on purity and the stated retained-capacity assumption, which permits $o(S_0)$ entropy outside the store.

**Derived:** Hawking's factor with $\kappa_H$ and $\Gamma_k$ on the controlled exponential segment, Jacobian included; the stationary Szegő rate; the isolated cubic-layer continuation calculation, the formal two-branch factor $e^{-2\pi\omega/\kappa_R}$ and the sum $\tau_k$ within a flux-normalized two-port reduction, not full-operator flux-normalized weights; the supply-integral budget, crossing formulas and $\mathcal R_\nu\ge\mathcal R$; the window $2T_d$ and the arrival clock as controlled estimates.

**Reported computations (pending reproduction):** $\mathcal L$, $\dot S_{\rm marg}$, $\mathcal R$ (two routes); $\dot n$ (one route); the 0.473 statistic, the 8.8 nats and the rise at 42–44 (two routes); the excess-per-energy diagnostic; the late slope, its normalization pending C; the recession ratio (one route, C); the Fermi-shaped local root-layer conversion ratio at $\kappa_R/(2\pi)$ on fixed and flow-carried sites (Ref. McKinneyCode2026, two routes), not flux-normalized weights; the pocket ceilings (two routes); the enclosure of $f$, taken from Sec. results-vacuum and not independently certified in this section.

**Assumed (labelled in the text):** no-return identification at a finite receiver; $S_{\rm esc}(0)=0$ with uncorrelated initial output; the two-port depletion law $\nu_+=0$, $\nu_-=\bar\nu$ independent of channel and frequency; self-similar $\Gamma_k(\omega R)$; the vacuum exterior's $\kappa_R=1/(2R)$; the reduced balance pending B, with control of the actual arrival-time conversion pending B and unit Jacobian only for the specified stationary exterior; empty outer inputs, exactly two root ports, vanishing port coherences and the fully matched weights, including survival of the local conversion ratio and $\tau_k=\Gamma_k$ on fixed and moving sites, pending V; the uniform comparator packet-budget premise $\mathcal B_{\rm pkt}=\mathcal B+o(S_0)$ over the history where the envelope is used, pending C; the store, its capacity, independent-pair loading, complete release and the saturating minimum law.

**Outstanding tests (marks and vehicles), not completed gates:** collar balance (B); recession second route, output covariance, ledger conventions and normalization, uniform packet-budget discrepancy (C); scattering matrix and moving-site unitary (V); $\mathcal R_\nu$ quadrature (Q); partner ledger (existing); $\bar\nu(u)$ beyond the records (P3). None closes by cutting a claim.