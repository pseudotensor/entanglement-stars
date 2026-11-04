**(i), shorter again.** Same scope as signed (replaces the subsection title and the eleven paragraphs from "A Page curve tracks the no-return entropy" through "…not substitution in Eq.~\eqref{eq:page-effective-maximum}."; Fig. page-conditional kept with (v); "The firewall question…" onward kept); the six emisA6 restorations are in the text, one clause each; same five displays and the same nine marks, verbatim. Against emisF5 plus the restorations this is about 300 words (about 35 column lines) shorter; every remaining sentence carries a claim, a named hypothesis or a mark, so the rest of the 70 lines is not available in this block without deleting one of those.

```latex
\subsection{The emission law and the Page curve}
\label{sec:page-effective}

A Page curve tracks the no-return entropy $S_{\rm esc}$ of the
radiation against the clock at infinity~\cite{Page1993}. The model
fixes the emission without a law placed by hand: the light field's
quadratic Hartree generator induces a Gaussian unitary whose
one-particle propagator $\mathsf U$ from preparation carries the
prepared sea to the pure covariance $C_L=\mathsf UP_\star\mathsf
U^\dagger$. Let $D_u$ contain orthonormal, once-counted outgoing
packets of both energy signs received through retarded time $u$,
on a common slice with the earlier packets retained; identifying
them with no-return radiation assumes controlled return
contamination from beyond the receiver and omitted-mode errors.
Then
\begin{equation}
n(\phi)=\|P_\star\mathsf U^\dagger\phi\|^2,\qquad
A_u=D_u^\dagger C_LD_u,\qquad
S_{\rm esc}(u)=\Tr s(A_u),
\label{eq:page-emission-law}
\end{equation}
with $n$ the prepared sea's weight in a packet traced back to
preparation and $1-n$ a negative-energy packet's hole occupation:
Gaussian identities under the model's own unitary, with no
subtraction. A block $B$ received after $u$ adds
$S(B|\mathcal E_u)=S(B)-I(B{:}\mathcal E_u)$, nonnegative for
independent blocks and smaller or negative when it purifies earlier
output; on the represented channels with the root held at $2000a$
both frozen ledgers give positive $S(B|X)$, two routes, so the
records certify no entropy-reducing arrival.
Ref.~\cite{McKinneyCode2026} details the derivations behind this
subsection and its secondary diagnostics.

Three statements follow from the propagator on the exterior of
Eq.~\eqref{eq:dr-exterior}, each with its own hypotheses. First, on
the exponential segment of the ray map the emission is derived: a
channel-$k$ packet of frequency $\omega$ is empty in the part
reflected at the centrifugal barrier near $3GE$, and its part
transmitted with probability $\Gamma_k(\omega)$, traced back along
the map of Eq.~\eqref{eq:sc-ray-temperature} with the regular
incoming-vacuum continuation, has on normalized resolving packets,
in the controlled exponential limit, the occupation
\begin{equation}
n_k(\omega)=\frac{\Gamma_k(\omega)}{e^{2\pi\omega/\kappa_H}+1},
\qquad\kappa_H=\frac1{4GE},
\label{eq:page-hawking-factor}
\end{equation}
Hawking's factor, Jacobian included, with the exterior's own
exponent and transmission, under the adiabatic long-wavelength
conditions of Sec.~\ref{sec:results-conversion}. For the ideal
stationary, regulated covariance with this symbol the Szeg\H{o}
limit gives the received entropy rate, $64k$ modes per unit of
$\omega/2\pi$ in channel $k$, as the rates of
Eq.~\eqref{eq:sc-luminosity}, two routes, $\mathcal R=1.639$ nats
per unit of energy over $T_H$: basis-invariant within the
received-mode subspace, not under arbitrary packet selection, and
on a finite adiabatic segment assuming control of boundary,
regulator, channel-tail and nonstationarity errors. No
packet-independent exponential coherence decay follows;
inter-arrival covariance and mutual information require
measurement \todo{C: output covariance on the stored histories,
inter-arrival blocks, conditional entropies and energy spectrum}.
Second, on the dust benchmark dispersion limits the continuum
exponential approximation to the estimated window $2T_d$ of
Sec.~\ref{sec:scaling}, usable only when positive and long enough
for a resolving packet, whose common-window, zero-matching
estimates, about $0.5$ nats at $r_s/a=64$, are illustrative, not
bounds on total emission or certified finite-window entropies.
Third, the energy balance: with both fields' averaged sources in
Eq.~\eqref{eq:dr-radial} the boundary charge is conserved
(Appendix~\ref{app:charge}), the charge inside a collar radius
$R_c$ beyond the barrier changing by collar flux plus reference
response to the field rates. Assumed, pending the full-reference
balance test \todo{B: two-field full-reference collar balance with
reference response, exterior storage, root-gradient factor and sea
mismatch below the balance accuracy}: sea and reference mismatch,
exterior storage, the root-gradient correction and the
arrival-time conversion controlled well enough to use
$dE_R/du=-L$, $L$ the light field's outgoing Killing-energy flux,
with the root at $R=2GE_R$, $E_R$ the charge inside it, and a
collar clock $u=t-\Phi(R_c)$ of unit Jacobian only for the
stationary exterior and clock convention specified. The omitted
terms are not bounded at the balance accuracy, the vacuum collar
holding for the dust benchmark and unproved for the object's
fountain.

Outside the segment the full dispersive propagator, not the ray
map, determines the contributing incident branches and stored
inventories, their occupations and coherences; under stationary
scattering completeness and incoming vacuum the absolutely
continuous sector carries no stationary excitation flux
(Sec.~\ref{sec:page}), while prepared transients and resonant
inventories may drain. A root receding at speed $\varepsilon$ under
its own emission excites the light field at about second order in
$\varepsilon$, at most $6.5\times10^{-5}$ of the required power,
which excludes self-funding only on the tested profiles and speed
range, one route \todo{C: recession excitation on a second route,
prescribed-recession histories with a held-history subtraction}.
Near a smooth simple root, a valve on what the interior supplies,
the symmetrically ordered isolated cubic-layer model of the
outgoing chiral component, neglecting its coupling to the ingoing
component, through $k/r$, and higher layer terms and continued on
the side regular outside the root, gives the formal two-branch
factor $e^{-2\pi\omega/\kappa_R}$ with weight sum $\tau_k$ by
flux-normalized two-port unitarity: the Fermi-shaped local
two-channel conversion at $\kappa_R/(2\pi)$ reported in
Ref.~\cite{McKinneyCode2026} on fixed and on flow-carried sites.
Assumed, pending the full scattering and incident-covariance
calculations, which must also fix that reduction's physical
weights for the full radial operator and the fixed/moving
equivalence \todo{V: flux-normalized scattering matrix from root
layer to barrier with every propagating branch, and the
moving-site operator related to the fixed-site one by its
time-dependent unitary}: the outer contributing inputs are empty;
exactly these two root-side ports contribute; their relevant
coherences vanish; and the fully matched weights, including
survival of the local conversion ratio and the identification
$\tau_k=\Gamma_k$, are
$w_{k\pm}=\Gamma_ke^{\pm\pi\omega/\kappa_R}/[2\cosh(\pi\omega/\kappa_R)]$.
For $0\le\nu_{k\pm}\le1$,
\begin{equation}
n_k(\omega,u)=w_{k+}\nu_{k+}+w_{k-}\nu_{k-}\le\Gamma_k(\omega),
\qquad
w_{k-}=\frac{\Gamma_k(\omega)}{e^{2\pi\omega/\kappa_R}+1},
\label{eq:page-valve}
\end{equation}
so $n_k\le\Gamma_k$, with $(\nu_+,\nu_-)=(0,1)$ Hawking's
occupation, the incident state of a regular prepared sea when both
branches trace back adiabatically to it, as on the controlled
segment, and up to $\Gamma_k$ when the other port is excited.

The histories report entropy diagnostics and collar flux, not
incident occupations: at the $16r_s$ cut a channel-one statistic of
about 0.473 nats per copy by $T/r_s=100$, two routes, rising at
42--44 with the collar flux burst of Fig.~\ref{fig:page-v4}(b), and
a whole-field exterior excess of about 8.8 nats. Neither the
illustrative 0.5-nat window estimate, which does not bound the
segment's contribution to that excess, nor Sec.~\ref{sec:page}'s
excess-per-energy diagnostic, 20 to 80 times below the thermal
$\mathcal R$, nor the formation-burst energy and entropy
diagnostics identifies the emission mechanism or the excited port;
identifying the channel-one statistic with the unsubtracted
$S_{\rm esc}$ requires its packet basis and comparator to be
documented, and its late-slope normalization against the benchmark
rate, 0.28--0.37 here against Sec.~\ref{sec:page}'s 0.23--0.31, is
unverified until the record's collar charge at 100 and the
benchmark rate, clock and units used are printed beside it
\todo{C: ledger conventions and normalization}. The arrival clock
from the stationary-exterior flight, 40.8--44.1 for
$\omega r_s=2$--0.4 up to an undetermined $O(r_s)$ matching term,
is a consistency check, not an independently predicted jump;
evolving or dispersive histories require their actual packet
transfer.

For the Page curve we take an idealized depletion comparator with
these assumptions: initially uncorrelated output, $S_{\rm esc}(0)=0$;
the two-port depletion law $\nu_{k+}=0$, $\nu_{k-}=\bar\nu(u)$, so
$n_k=\bar\nu n_k^H$ with $0\le\bar\nu\le1$ independent of channel and
frequency; a self-similar transmission spectrum $\Gamma_k(\omega R)$;
the vacuum exterior's $R=2GE_R$ and $\kappa_R=1/(2R)$; and the
reduced balance $dE_R/du=-\bar L$, $\bar L=\bar\nu\mathcal L/R^2$.
With $n_k^H=\Gamma_k/(e^{\omega/T_H}+1)$ and
$\mathcal R_\nu=T_H\sum_k64k\int\frac{d\omega}{2\pi}s(\nu n_k^H)
\big/(\nu\mathcal L/R^2)$, the emitted energy and the marginal
budget are supply integrals,
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
Here $S_0=\pi R_0^2/G$, $R_0=2GE_0$, the remainder refers to
$\chi_0=a/R_0\to0$ at fixed $ma$, $G/a^2$ and $\eta_E$, and $u=0$,
the first arrival of root-emitted light, lies $O(R_0\ln(1/\chi_0))$
after the matter crosses $r_s$ on the dust benchmark
(Eq.~\eqref{eq:star-crossing-delay}), an actual collapse-arrival
time only through the packet-transfer calculation. A further
comparator assumption is a resolving packet basis whose diagonal
entropy sum $\sum_{i\in D_u}s((A_u)_{ii})$, an upper bound on
$S_{\rm esc}$ by subadditivity in any orthonormal basis of the
received-mode subspace, equals $\mathcal B+o(S_0)$ uniformly
wherever the envelope is used, requiring control of supply
variation, packet boundaries, regulators, channel tails and the
endpoint region; whether the discrepancy over $S_0$, omitted-mode
errors included, vanishes uniformly in the stated scaling limit is
untested \todo{C: uniform packet-budget discrepancy}. The second
branch assumes the stated retained capacity: $\pi R^2/G$ is a
first-law quantity, not an established retained entropy or
accessible capacity, and purity, equating $S_{\rm esc}$ with the
entropy of the actual complement, supplies no area-sized
accessible-sector bound. Original heavy particles give only
$O(N_B\ln(1/\chi_0))$ nats, and the light field's pockets inside
the root, $O((R/a)^3)$ in number and filled by the partners at
negative Killing energy that turn there (Sec.~\ref{sec:star}), have
fixed-energy Gibbs ceilings of 0.27--0.31 of $fA_R/(4G)$ on the
vacuum profile at $R/a=32$--256, two routes, and 0.08--0.10 on the
stored slices, $A_R=4\pi R^2$, $f=4\eta(ma)G/a^2$
(Eq.~\eqref{eq:page-share}), while the unrestricted pocket entropy
grows with volume: finite-size area-scaling estimates establishing
neither an asymptotic accessible capacity $fA_R/(4G)$ nor its
dynamical attainability. The assumptions are therefore a
root-carried store of logarithmic dimension at most $fA_R/(4G)$
with $o(S_0)$ retained entropy outside it; independent pairs with
partners stored until it saturates; then release carrying the
partner correlations to the output as the root recedes, complete
at the end; heavy transitions funding every operation and each
escaping energy debited once (the test light of Sec.~\ref{sec:page}
omitted its own gravity, so no debit is realized in those
histories), apparatus entropy at most 2000 nats and unintended
output $o(E_0)$, $o(S_0)$; and the capacity-saturating history that
turns the bound in Eq.~\eqref{eq:page-effective-curve} into
equality. Under a joint generator conserving each heavy-copy
number, tracing out the heavy field hides from the light output
the coherence between conserved-number sectors, not necessarily
within one; a charge-matched code removes only that obstruction
and supplies no coherent heavy-to-light transfer, so recovery of
arbitrary heavy-input coherence under product Hartree evolution is
not established \todo{partner ledger: accessible-sector bound from
pocket occupation, deposition and drift including omitted sectors;
no-return conditional entropies, residual retained entropy and
coherent heavy-input recovery; windowed root energy budget and
branch errors; first controlled blocks at sizes 32 and 64}. A
finite-time crossing of the branches, $\mathcal B(u_P)=fy(u_P)S_0$
at finite $u_P$ under the supply-integral evolution of $R$, has a
supply condition that is necessary, not sufficient, and
establishes no saturation of the entropy bound; for constant
$0<\bar\nu\le1$, under the assumed minimum law, they meet at
\begin{equation}
y_P=\frac{\mathcal R_{\bar\nu}}{\mathcal R_{\bar\nu}+f},\quad
\frac{S_P}{S_0}=\frac{f\mathcal R_{\bar\nu}}{\mathcal R_{\bar\nu}+f},\quad
\frac{u_P}{t_*}=1-y_P^{3/2},\quad
t_*=\frac{R_0^3}{6G\mathcal L\bar\nu},
\label{eq:page-effective-maximum}
\end{equation}
and a variable-supply lifetime fraction exists only with a finite
evaporation time $t_*$. At $\bar\nu=f=1$, Hawking's feed (the
incoming vacuum's under Eq.~\eqref{eq:page-valve}'s assumptions),
the comparator peaks at $0.6211S_0$ and $0.5105t_*$: an assumed
minimum-capacity comparator, not a computed maximum or an entropy
history that feed determines. Concavity of $s$ gives
$\mathcal R_\nu\ge\mathcal R$ \todo{Q: $\mathcal R_\nu$ by
quadrature on the benchmark transmissions, two routes}. At
$a=2\ell_P$, $ma=0.2$, nominal $f=0.887$ gives $0.5755S_0$ at
$0.4773t_*$ and the enclosure $0.7883<f<0.9831$ of
Sec.~\ref{sec:results-vacuum} the heights $[0.5323,0.6146]S_0$ at
saturation; the finite-band $\mathcal L$ and $\mathcal R$ have no
narrow enclosure, and the records through $T/r_s=100$ fix neither
the late supply nor the turnover \todo{$\bar\nu(u)$ beyond the
records: thin-shell asymptotic expansion nearest the trapping root
at astrophysical $\chi$, received redshift with lapse and flow,
matched to the stored histories}.
Figure~\ref{fig:page-conditional} plots the saturated comparator;
panel (b) adds the caption's assumptions.
```

**Moved to Ref. McKinneyCode2026 (one pointer, end of first paragraph).** Items already moved in emisF5 stay there; three of them are re-described because their qualifications are now back in the text:

- the pulled-back amplitude $(U_H-U)^{-1/2+i\omega/\kappa_H}$, the traced-back incoming amplitude $(du/dU)^{1/2}e^{-i\omega u(U)}$ under the map $U_H-U=C_Ue^{-\kappa_Hu}$ (the map itself is Eq.~sc-ray-temperature), the negative-to-positive weight-ratio step $e^{-2\pi\omega/\kappa_H}$ on normalized resolving packets, the detail of the regular incoming-vacuum half-plane continuation (the condition itself is now stated in the text) and the remark that the Jacobian's real exponent changes only the common spectral factor;
- the $64k$ mode count (eight valleys, $2k$ magnetic states, two signs of $\kappa$, particles and holes) and the Szeg\H{o} finite-duration form $T\sum_k64k\int_0^\infty\frac{d\omega}{2\pi}s(n_k)+o(T)$;
- the solar-mass window estimate, $12.5$ nats, and the explicit window formula $2T_d=\tfrac43r_s\ln[1/(2a\omega)]+C_cr_s$ (cross-referenced to Sec.~V~A) with the $2.5\times10^{-4}t_*$ restatement (already in Sec.~VI~C);
- the windowed collar identity $N\dot{\mathcal M}(R_c)=\Tr(\bar h_{R_c}\dot C)-\langle\bar u_{R_c},\mathcal W''\dot{\bar u}\rangle$;
- the root-layer frozen symbol $(N/a)\sin x\,(1-|w|\cos x)$, the size of the neglected $k/r$ coupling and higher layer terms (the neglect itself is now stated in the text), the WKB branch positions $\pm x_c$ with $\cos x_c=1/|w|$, valid as $\omega\to0$ and not their exact positions at positive $\omega$, the cubic-layer symbol $\omega=\kappa_R(r-R)p+c_3p^3$, $c_3=N_Ra^2/2$, its continuation around $p=0$ and the solution $\tilde\psi(p)\propto p^{-1/2-i\omega/\kappa_R}e^{ic_3p^3/(3\kappa_R)}$;
- the arrival-clock flight pieces (flight from the root to $16r_s$, $23.2r_s$ exterior flight, root logarithm replaced by the crossing delay of Eq.~star-crossing-delay, $u=T-26.2r_s$ at the cut, formation at 12.1) and the late slope's absolute value, 0.030--0.040 nats per nominal $r_s$ against the benchmark $0.1079/r_s$ (both already in Sec.~VI~C and Eq.~sc-luminosity);
- the slowly varying-symbol derivation behind the packet-budget prerequisites (the prerequisites themselves are now stated in the text), the symbol $\mathcal B_{\rm pkt}$ for the packet budget and the zero-supply continuation of the budget integrand by its limit, zero;
- the heavy-particle budget's solar-mass value, about $10^{-36}S_0$, and the volume-growth exponent of the unrestricted pocket entropy, 2.99 over fourfold size on the second route (the growth itself is now stated in the text);
- the explicit supply condition $6G\mathcal L\int_0^\infty\bar\nu\,dv\ge R_0^3(1-y_P^{3/2})$ and the variable-supply lifetime fraction $u_P/t_*=(1-y_P^{3/2})\langle\bar\nu\rangle_{0:t_*}/\langle\bar\nu\rangle_{0:u_P}$, $y_P=y(u_P)$;
- the dilute formula $\mathcal R_\nu\simeq\mathcal R+(T_H\dot n/L)\ln(1/\nu)$ and its one-route quarter-supply evaluation, $\mathcal R_{1/4}\simeq1.97$ at constant quarter supply, height $0.66S_0$ at 0.46 of a lifetime four times longer;
- the finite-band digits $\mathcal L\simeq0.0052373$, $\mathcal R\simeq1.6391$ (printed as 0.005237 and 1.639 in the paper) and the $t_*=r_{s0}^3/(6G\mathcal L)$ restatement at $\bar\nu=1$ (already in Sec.~VI~C).

**Merged in place, nothing moved:** the two status clauses on the two-port reduction's physical weights for the full radial operator and on the fixed/moving equivalence, folded into the "Assumed, pending" sentence of the valve paragraph; the formation-burst diagnostics clause, folded into the histories' mechanism sentence; "none fixing a Page maximum" and "these are comparator values under the assumptions above, not a computed maximum", folded into the minimum-capacity comparator clause; "evolving histories require their actual time conversion", folded into the controlled arrival-time conversion item; the first-law-quantity clause ($\pi R^2/G$), carried from the balance paragraph to the capacity sentence where it is used.