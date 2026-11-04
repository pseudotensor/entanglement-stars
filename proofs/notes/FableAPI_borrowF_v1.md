# Borrowed items, ready to apply to P3

Conventions used below: P3's notation throughout (\(N\), \(v\), \(w=v/N\), \(\mathcal M\), \(r_s=2GE\), \(\kappa_H=1/(4GE)\), \(\mathcal L\), \(\mathcal R\), \(f\), \(S_{\rm cut}\), \(S_{\rm esc}\), \(P_\star\), \(C_L\), channels \(k\) with weight \(16k\), \(\sigma=4a\) the kernel width); P3's labels for \ref and \eqref; the \todo macro for every sentence whose P3 evidence has not landed. No em-dashes, counts one to nine in words. Each item states where it goes, what it replaces, the net line change, and what is computed, measured with a mark, or assumed. Line counts are two-column estimates.

## 0. Map

| Item | Where | Replaces | Net lines | Status of content |
|---|---|---|---|---|
| B4 | Sec. results-current, the Dirac-Schwinger paragraph; one clause each in Sec. model-gravity step four, Sec. model-dictionary, Sec. discussion outlook | the lead-in sentence of eq dr-dirac-schwinger and the sentence "No current depending only on \(u\)..."; three clauses | +20 | derived (exact lattice identity, checkable below) |
| B7 + B1a + B2 | Sec. page, "Emission and held-field controls", the Hawking comparison case passage | the seven lines from "The Hawking comparison case..." to "...reported but unverified." and the five-line sentence "Purity with a retained capacity ... not of the object." | +14 | derived (barrier), computed (rates, limits), one P2-computed check quoted, two marks |
| B1b | emisF6, after "...supplies no area-sized accessible-sector bound." plus one rewritten clause | the clause "a root-carried store of logarithmic dimension at most \(fA_R/(4G)\)" | +8 | measured on chains, cubic cut marked |
| B1c | emisF6, end of the assumptions sentence | the clause "...into equality." | +9 | kinematic, inside the Gaussian class; realization marked |
| B3a | emisF6 first paragraph, after "...certify no entropy-reducing arrival." | nothing | +6 | exact identity; register marked |
| B3b | Sec. page, "The light field on the computed collapse", after the channel-tail sentence | nothing | +5 | two marks |
| B8 | Sec. results-exterior, after the back-averaged-flow sentence; one clause in Sec. model-gravity | the clause "with undetermined coefficients and no bound" | +18 | derived; one mark (stored-slice fit) |
| B9 | Figs. star-collapse (d), page-conditional (a) restyled and (c) new, page-v4 (d) optional | nothing | captions +8 | data listed |
| B10 | specs only; two optional one-sentence hooks | nothing | +2 | marks |

Compensating slack, if the owner wants strict neutrality: Sec. page's "Purity with a retained capacity..." sentence (already removed above), the Kerr sentence in Sec. results-exterior (3 lines, result used nowhere), and 5 lines of the citation list in Sec. discussion.

---

## 1. B4: the energy-energy bracket on the cubic lattice

**Where.** Sec. results-current. Replace the paragraph beginning "At long wavelength, for continuum-spinor smearings \(f,g\) ... the bare bracket is" and ending "...closes on the momentum density." (keep the display eq dr-dirac-schwinger inside the new text), and delete the sentence "No current depending only on \(u=f\nabla g-g\nabla f\) reproduces this bracket (example in Ref.)", with the text below. Then three clause edits: in Sec. model-gravity step four, "so neither tensor propagation nor the full deformation algebra is established" becomes "so neither tensor propagation nor the mixed bracket of the deformation algebra is established, the energy-energy bracket closing on the lattice with the metric's structure function (Sec.~\ref{sec:results-current})"; in Sec. model-dictionary, "Neither coefficient is free: the bracket of Sec.~\ref{sec:results-current} fixes the flow term's, and the prepared sea's rest under a bulk rigid turn fixes the spin vertex within \(\Sigma_kf_k(p)\)" becomes "Neither coefficient is free: the bracket of Sec.~\ref{sec:results-current} produces both terms, current and spin vertex, exactly for affine smearings, and the prepared sea's rest under a bulk rigid turn"; in Sec. discussion outlook, "the deformation algebra of the constraints (the Dirac-Schwinger commutation relations, not realized on fixed spatial coordinates)" becomes "the mixed bracket of the deformation algebra, not realized on fixed spatial coordinates (Sec.~\ref{sec:results-current})".

```latex
The bracket closes on the lattice itself, for any two smearings, and what
it produces is the flow coupling of Eq.~\eqref{eq:dm-hamiltonian}. Write
$P^I_f=\tfrac12\{f,T_I-T_I^\dagger\}=\bar f_IT_I-T_I^\dagger\bar f_I$,
$\bar f_I(x)=\tfrac12[f(x)+f(x+ae_I)]$, so that
$e[f]=\tfrac12\{f,h_0\}=\sum_I\alpha_IP^I_f/(2ia)+mf\beta_{\rm D}$. The
mass terms drop out of $[e[f],e[g]]$ because $\{P^I_f,g\}=\{P^I_g,f\}$
identically, same-direction terms are internal scalars, and
cross-direction terms carry $\alpha_I\alpha_J=-\alpha_J\alpha_I$, so
\begin{equation}
\begin{aligned}
i[e[f],e[g]]&=\sum_I\frac{\mathsf u_IT_I^2-T_I^{\dagger2}\mathsf u_I}{4ia^2}
+\sum_{I<J}\frac{\alpha_I\alpha_J\bigl(\{P^I_f,P^J_g\}-\{P^I_g,P^J_f\}\bigr)}{4ia^2},\\
\mathsf u_I(x)&=\bar f_I(x)\,\bar g_I(x+ae_I)-\bar g_I(x)\,\bar f_I(x+ae_I),
\end{aligned}
\label{eq:dr-lattice-bracket}
\end{equation}
exact on the lattice. The first sum is the two-step current $Q_I$ of
Sec.~\ref{sec:model-dictionary} smeared by the bond field
$\mathsf u_I/a$; cross-direction paths carry no internal scalar, so the
scalar sector has two-step axial displacements and nothing else, and
the current the energy algebra selects is $Q_I$ exactly, not up to a
stencil choice. The second sum is the spin term with the lattice vertex
$\Sigma^{\rm lat}_K=\Sigma_K\mathsf c_I\mathsf c_J$ of
Eq.~\eqref{eq:dm-hamiltonian}. For smearings affine in $x$ both sums
are exact couplings: with $u=f\nabla g-g\nabla f$, itself affine,
$i[H[f],H[g]]=c^\dagger\bigl[\tfrac12\sum_I\{u_I,Q_I\}
+\tfrac18\sum_K\{(\nabla\times u)_K,\Sigma^{\rm lat}_K\}\bigr]c$
with no remainder, the flow and spin couplings of the model at the
flow $u$. For smooth smearings the bond field expands about the
two-step midpoint as
$u_I+\tfrac{a^2}{4}(\partial_I^2f\,\partial_Ig-\partial_If\,\partial_I^2g)
+\tfrac{a^2}{6}(f\partial_I^3g-g\partial_I^3f)+O(a^4)$, which for the
harmonic pair above is $\xi[1-\tfrac5{12}(a\xi)^2]$, the factor
$F_a(\xi)/\xi$ of Eq.~\eqref{eq:dr-longitudinal-algebra}. The spin
term's expectation vanishes in the prepared sea and in every state that
is a function of a flow-free coupled operator, since
$\omega=\alpha_1\alpha_2\alpha_3\beta_{\rm D}$ anticommutes with every
term of $h_0$ while $\alpha_I\alpha_J$ commutes with it; on flows it is
the spin coupling itself. At long wavelength, for continuum-spinor
smearings (valley sum and valley-rotated spin matrices understood),
Eq.~\eqref{eq:dr-lattice-bracket} is
[eq dr-dirac-schwinger, the display, unchanged]
the Dirac-Schwinger form~\cite{Dirac1962,Schwinger1962} of a
relativistic energy density, with the one-particle lattice remainder
[the existing bound and its conditions, unchanged]. Radial $f,g$ have
zero curl, so their bracket closes on the momentum density. Smearing
about a background clock, $f=N\mathcal N$ and $g=N\mathcal M$, gives
$u=N^2(\mathcal N\nabla\mathcal M-\mathcal M\nabla\mathcal N)$, the
clock's own gradient cancelling: the energy-energy bracket of the
clock-coupled generator closes with structure function $N^2\delta^{IJ}$,
the clock squared times the flat co-metric of Eq.~\eqref{eq:dm-metric}
at $v=0$, a second route to the light cone, from the algebra and not
the dispersion. For a constant flow the mixed bracket closes as well,
$i[\mathcal P[v],H[N]]=H[v\cdot\nabla N]+O(a^2)$ with
$\mathcal P[v]=c^\dagger\tfrac12\sum_I\{v_I,Q_I\}c$, from
$[T_I^2,\tfrac12\{N,h_0\}]=\tfrac12\{N(\cdot+2ae_I)-N,h_0\}T_I^2$; for
a flow with strain it does not, as the dilation shows next.
```

The existing paragraph "The mixed bracket of momentum and energy does not close on the fixed flat lattice..." follows, with its first clause changed to "does not close for flows with strain:".

**Derivation notes (checkable).**
1. Shifts: \((T_Iu)(x)=u(x+ae_I)\), so \(T_I\phi=\phi^{(+I)}T_I\), \(\phi T_I^\dagger=T_I^\dagger\phi^{(+I)}\), \(T_I^\dagger\phi T_I=\phi^{(-I)}\). Then \(\tfrac12\{f,T_I-T_I^\dagger\}=\bar f_IT_I-T_I^\dagger\bar f_I\).
2. \(\{P^I_f,g\}=2\bar f_I\bar g_IT_I-2T_I^\dagger\bar f_I\bar g_I\), symmetric in \(f,g\): mass-kinetic cross terms \(\frac{m}{2ia}\alpha_I\beta_{\rm D}(\{P^I_f,g\}-\{P^I_g,f\})\) vanish exactly; \([\beta_{\rm D}f,\beta_{\rm D}g]=0\).
3. \([P^I_f,P^I_g]=\mathsf u_IT_I^2-T_I^{\dagger2}\mathsf u_I\) by expanding the four products; \(i/(2ia)^2=1/(4ia^2)\).
4. Cross terms: \([\alpha_IP^I_f,\alpha_JP^J_g]+[\alpha_JP^J_f,\alpha_IP^I_g]=\alpha_I\alpha_J(\{P^I_f,P^J_g\}-\{P^I_g,P^J_f\})\), purely grade two.
5. Affine \(f\): \(P^I_f=f\Delta_I+af_{,I}\mathsf c_I\) exactly, \(\Delta_I=T_I-T_I^\dagger\); \([\Delta_I,g]=2ag_{,I}\mathsf c_I\), \([\mathsf c_J,f]=\tfrac a2f_{,J}\Delta_J\). The derivative pieces of the grade-two bracket cancel exactly, leaving \(2a^2(f_{,I}g_{,J}-f_{,J}g_{,I})\mathsf c_I\mathsf c_J\); with \(\alpha_I\alpha_J=i\Sigma_K\) this is \(\tfrac14\Sigma^{\rm lat}\cdot(\nabla\times u)\). For affine \(f,g\): \(\partial_Iu_I=0\) (no sum) and \(\mathsf u_I(x)/a=u_I(x+ae_I)=\tfrac12[u_I(x)+u_I(x+2ae_I)]\), so the scalar sum is \(\tfrac12\{u_I,Q_I\}\) exactly.
6. Midpoint expansion of \(\mathsf u_I/a\) about \(x+ae_I\): \(\bar f_I(x)=F(y)\), \(F=f+\tfrac{a^2}{8}f''\), \(y=x+\tfrac a2e_I\); \(F(z-\tfrac a2)G(z+\tfrac a2)-G(z-\tfrac a2)F(z+\tfrac a2)=a(FG'-GF')+\tfrac{a^3}{24}(FG'''-GF''')+\tfrac{a^3}{8}(F''G'-F'G'')+O(a^5)\). Check: \(f=\cos\xi x\), \(g=\sin\xi x\) gives \(\xi(1-\tfrac{5}{12}a^2\xi^2)\).
7. \(\omega\) oddness: \(\omega h_0\omega^{-1}=-h_0\Rightarrow\omega(P_0-\tfrac12)\omega^{-1}=-(P_0-\tfrac12)\); grade-two operators commute with \(\omega\) and are internally traceless, so their \(P_0\) expectation is zero. The propagated state on flows is not \(\omega\)-odd (the flow term is internal-scalar), hence the qualification.
8. Constant flow: \([Q_I,\tfrac12\{N,h_0\}]=\frac{1}{4ia}[\tfrac12\{N^{(+2I)}-N,h_0\}T_I^2-\tfrac12\{N^{(-2I)}-N,h_0\}T_I^{\dagger2}]\); expanding \(N^{(\pm2I)}-N=\pm2a\partial_IN+2a^2\partial_I^2N+\ldots\) gives \(\tfrac14\{\partial_IN,h_0\}(T_I^2+T_I^{\dagger2})+a^2\{\partial_I^2N,h_0\}Q_I\), i.e. \(e[\partial_IN]\) up to \(O(a^2)\) on long-wavelength matter.

**Softened claim as agreed:** the energy-energy bracket closes with the metric's structure function; the mixed bracket is open except for constant flows.

---

## 2. B7 with B1a and B2: the model's own barrier, the two currents, and the microscopic check

**Where.** Sec. page, subsection "Emission and held-field controls". Replace from "The Hawking comparison case, a thermally supplied exterior, feeds each Dirac channel \(k\)..." through "The number rate is reported but unverified." with the text below (eq sc-luminosity stays as the display inside it). Delete the sentence "Purity with a retained capacity at most \(S(E)\) bounds the escaped entropy by \(\min(S_{\rm marg},S(E))\), ... not of the object." (emisF6 carries the comparator). Keep eq page-conditional and its sentence.

```latex
The Hawking comparison case is a thermally supplied exterior whose
transmission is the model's own. On the exterior of
Eq.~\eqref{eq:dr-exterior} the operator of channel $k$ is
$\bar h_k=h_{0,k}+\tfrac12\{v,Q_r\}$ at $m=0$, $N=1$, $v=-\sqrt{r_s/r}$.
A mode stationary in the lattice time at Killing frequency $\omega$ is
stationary in the time $t_s$ of Sec.~\ref{sec:results-exterior} as well,
and in the static frame there, the spinor boosted by $e^{\eta\sigma_1/2}$
with $\tanh\eta=v$, its long-wavelength radial equation is
\begin{equation}
\omega\chi=\bigl[-i\sigma_1\partial_{r_*}+W\sigma_2\bigr]\chi,\qquad
W=\frac{\alpha_sk}{r},\qquad dr_*=\frac{dr}{\alpha_s^2},
\label{eq:page-dirac-barrier}
\end{equation}
$\alpha_s=\sqrt{1-r_s/r}$ the Killing lapse. Because $\sigma_2$
anticommutes with $\sigma_1$ the angular term enters the square in
quadrature,
$\omega^2\chi=[-\partial_{r_*}^2+W^2+\sigma_3\partial_{r_*}W]\chi$: the
two components see the partner barriers $W^2\pm\partial_{r_*}W$, rising
to the centrifugal $k(k\pm1)/r^2$ far out and vanishing at $r_s$ with
$\alpha_s$. The transmission $\Gamma_k(\omega)$ is that of this barrier,
unit-current normalized, a ratio of asymptotic fluxes and so the same in
either time, the same for the two partners and for both signs of
$\kappa=\pm k$. Feeding each channel thermally at $T_H$ from the inner
port with the outer port empty, and counting the eight valleys, both
signs of $\kappa$, particles and holes, $64k$ modes per unit of
$\omega/2\pi$,
\begin{equation}
L=\sum_k64k\!\int_0^\infty\!\frac{d\omega}{2\pi}\,\omega\,n_k,\qquad
\dot S_{\rm marg}=\sum_k64k\!\int_0^\infty\!\frac{d\omega}{2\pi}\,s(n_k),\qquad
n_k=\frac{\Gamma_k(\omega)}{e^{\omega/T_H}+1},
\label{eq:page-currents}
\end{equation}
the entropy taken at the transmitted occupation because tracing out the
reflected part leaves the transmitted mode Gaussian at $\Gamma_kf$,
give
[eq sc-luminosity, unchanged]
with $\mathcal L=0.005237$, two independent calculations, the channel
sums converged to six digits in $\mathcal L$ and three in $\mathcal R$
by $k\le4$ (the centrifugal tail falls as $r^{-2}$, so the
outer-boundary error of $\mathcal R$ dies only as $1/r_{\max}$); the
same integrator returns Page's emission coefficients for spins
$\tfrac12$, 1 and 2, $8.20$, $3.36$ and $0.384$ against $8.18$, $3.37$
and $0.380$ in units of $10^{-5}/(G^2E^2)$, and identical transmission
for the two partners~\cite{Page1976} \todo{B7: the Page benchmark and
partner check on both of this paper's integrators}. The number rate is
reported but unverified. The ratio $\mathcal R=T_H\dot S_{\rm marg}/L
=1.639$ is fixed by the one transmission and the one occupation. It
exceeds one mode by mode: $s(x)-(\omega/T_H)x$ vanishes at $x=0$ and
has derivative $\log[(1-x)/x]-\omega/T_H\ge0$ for $x\le f$, and
$\Gamma_kf\le f$. For a sharp count $C(\omega)=\sum_k64k\,\Gamma_k$ it
depends on the $\omega$ dependence of $C$ alone, exactly 2 for a flat
count and $4/3$ for $C\propto\omega^2$, and it returns Page's $1.4847$
for photons and gravitons as $1.48474$~\cite{Page1983}; the soft Dirac
barrier sits between the two. The counting is checked without
Landauer input by an exact Gaussian quench. On a uniform hopping chain,
whose low-energy counting is dispersion independent, a half at
temperature $T_L$ joined to a half at zero and evolved exactly carries
energy and entropy into the cold half at rates within $1.2$ and $1.3$
percent of Eq.~\eqref{eq:page-currents} at $\Gamma=1$ for $T_L$ from
$0.03$ to $0.3$ of the hopping, with $\mathcal R=1.97$ to $2.01$
against 2, the remainder the logarithmic entanglement term a finite fit
window leaves. On the held exterior chain itself, a uniform-flow
reservoir at $T_H$ inside the barrier and the vacuum outside, the same
quench reproduces $L$ and $\dot S_{\rm marg}$ through a cut beyond the
barrier \todo{B2: the held-exterior quench, two routes}.
```

**Derivation notes.** (i) Both coordinates share the Killing vector \(\partial_t\); a stationary scattering problem's transmission is a ratio of asymptotic conserved fluxes, independent of the time coordinate and of the local spinor frame. (ii) In the static tetrad the massless radial Dirac system in Schwarzschild reads \(\partial_{r_*}\chi=(W\sigma_3+i\omega\sigma_2)\chi\), unitarily \(\omega\chi=(-i\sigma_1\partial_{r_*}+W\sigma_2)\chi\); squaring uses \(\sigma_1\sigma_2=i\sigma_3\). (iii) Limits: \(\int_0^\infty\log(1+e^{-x})dx=\int_0^\infty x/(e^x+1)dx=\pi^2/12\) gives \(\mathcal R=2\); \(\int_0^\infty x^3/(e^x+1)dx=7\pi^4/120\) and \(\int_0^\infty x^2\log(1+e^{-x})dx=7\pi^4/360\) give \(4/3\). (iv) The mode-by-mode bound uses \(\log[(1-x)/x]\ge\log[(1-f)/f]=\omega/T_H\) on \([0,f]\).

**Status.** Barrier: derived. \(\mathcal L,\mathcal R\), limits, Page 1983 anchor: computed (P3's own two calculations for the rates; the limits analytic). Page 1976 spins check and partner check: computed on the earlier integrator, marked until both P3 integrators reproduce it (a five-minute run). Flat-chain quench: computed (one exact route against the Landauer route). Held-exterior quench: marked.

**B2 runnable spec (held-exterior quench).**
- Chain: channel \(k=1\) (then \(k=2\)), massless two-component chain \(h_{0,k}\), \(n_{\rm ch}\ge8192\) sites; fields \(N=1\), \(v_n=-\sqrt{r_s/r_n}\) for \(r_n\ge r_{\rm in}\); \(r_{\rm in}=1.2r_s\) (so \(|v_{\rm in}|=0.913<1\), both branches propagate). Reservoir: sites \(r_n<r_{\rm in}\), length \(\ge\) run time times the maximal group speed, uniform flow \(v_{\rm in}\), centrifugal term frozen at \(k/r_{\rm in}\), so the reservoir operator \(h_{\rm res}=h_{0,k}|_{k/r_{\rm in}}+v_{\rm in}Q_r\) is translation invariant and its eigenvalues are Killing energies.
- Initial covariance: \(C(0)=f(h_{\rm res}/T_H)\oplus P_-(\bar h_{\rm ext})\), Fermi function at \(T_H=1/(4\pi r_s)\) on the reservoir, the ground projector of the decoupled exterior chain outside (below the sign-reversal threshold it is the prepared band).
- Evolution: \(C(t)=e^{-i\bar ht}C(0)e^{i\bar ht}\) with \(\bar h\) the joined chain, exact (one diagonalization).
- Readouts at the cut \(r_c=3r_s\), beyond the barrier near \(1.5r_s\): energy flux \(\Tr(C\,i[\bar h,\bar h_{>r_c}])\); entropy \(S_{>r_c}(t)=\Tr s(C_{>r_c})\), fitted after the front passes to \(\dot S\,t+c\log t+d\) on two windows (window shift is a control), plus the block form \(\sum_j S(B_j|\text{earlier})\) as the second entropy readout.
- Targets per two-component copy: \(L_{\rm copy}=2\int_0^\infty\frac{d\omega}{2\pi}\omega n_k\), \(\dot S_{\rm copy}=2\int_0^\infty\frac{d\omega}{2\pi}s(n_k)\), with \(\Gamma_k\) from the transfer matrix on eq page-dirac-barrier (the second route). Acceptance: both rates within 2 percent, their ratio within 2 percent of the single-channel \(T_H\dot S_{\rm copy}/L_{\rm copy}\).
- Controls: reservoir length doubled, \(r_{\rm in}=1.4r_s\), chain length doubled, \(T_H\) halved (the ratio must not move).

---

## 3. B1b, B1c, B3a, B3b: the measured late branch, the achievability channel, the register identity, purity and pairs

### B1b (into emisF6)

**Where.** emisF6, paragraph "Here \(S_0=\pi R_0^2/G\) ...", after the sentence ending "...supplies no area-sized accessible-sector bound." Insert:

```latex
The branch does have a measured form. For the pure light covariance,
purity and subadditivity give
$S_{\rm esc}(u)\le S_{\rm cut}(R_t(u))+S(\mathcal X_u)$, with
$S_{\rm cut}(R_t)$ the entropy of the evolved covariance across the
trapping cut and $\mathcal X_u$ the exterior modes not yet received, so
wherever the retained entropy outside the trapped region is $o(S_0)$
the second branch is the measured ratio $S_{\rm cut}(R_t(u))/S_0$, not
an assumed store, and $f$ in Eq.~\eqref{eq:page-effective-curve} is
that ratio's leading value, $4\eta(ma)G/a^2$ where the evolved cut
keeps the flat coefficient, which the direct-state budget supports on
the histories (Sec.~\ref{sec:results-diagnostics}) without a uniform
cubic bound at this size. On the stored slices the chain readings give
the exterior excess over the held vacuum, 109 to 311 nats for the heavy
field and about nine for the light field (Sec.~\ref{sec:page}); the cut
entropy itself is the paired cubic cut \todo{B1: $S_{\rm cut}$ of the
evolved cubic covariance at $R_t(t)$ on the stored slices, both fields,
two routes, with the flat ball on the same site set}.
```

Then rewrite the clause in the next sentence: "The assumptions are therefore a root-carried store of logarithmic dimension at most \(fA_R/(4G)\) with \(o(S_0)\) retained entropy outside it;" becomes "The assumptions are therefore that the trapped region's accessible logarithmic dimension is its measured cut entanglement \(S_{\rm cut}(R_t)\), with \(o(S_0)\) retained entropy outside it;".

Derivation: \(S_{\rm esc}=S(D_u)=S(\overline{D_u})\) by purity; \(\overline{D_u}=\mathrm{int}(R_t)\oplus\mathcal X_u\) when the received packets lie outside \(R_t\) on the common slice; subadditivity and \(S(\mathrm{int})=S(\mathrm{ext})=S_{\rm cut}\).

### B1c (into emisF6)

**Where.** emisF6, the assumptions sentence ending "...and the capacity-saturating history that turns the bound in Eq.~\eqref{eq:page-effective-curve} into equality." Replace that final clause with:

```latex
and the capacity-saturating history that turns the bound in
Eq.~\eqref{eq:page-effective-curve} into equality. Such a history
exists inside the model's own class, as a channel and not as the
collapse: a number-conserving quadratic circuit that, while the output
entropy lies below the cut capacity, sends Gaussian Schmidt pairs
across the trapping cut, a received packet at Killing energy $\omega$
paired with a pocket mode at $-\omega$ inside the root, and once the
capacity is reached releases stored pocket partners into outgoing
packets by Gaussian swaps at the rate that keeps the output entropy
equal to the shrinking capacity, its only error one unresolved packet's
entropy, vanishing in the narrow-packet limit. It establishes that the
bound is attainable within the Gaussian class with the pockets as the
store; holding $fA_R/(4G)$ there needs pocket energy beyond the
fixed-energy ceilings below, and whether the collapse's own evolution
follows such a history is the register question of the first paragraph,
open here.
```

### B3a (into emisF6)

**Where.** emisF6, first paragraph, after "...so the records certify no entropy-reducing arrival." Insert:

```latex
Which side a block's partner is on is the same statement: with
$\mathcal S_u$ the complement of $B\cup\mathcal E_u$ in the pure light
covariance, $I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)$
exactly, so a positive conditional entropy says the block is more
correlated with what remains than with what has left, a negative one
the reverse, and the turnover of $S_{\rm esc}$ is where the two mutual
informations cross, which no partition that does not accumulate the
output can show. On the stored histories the register deciding it is
\todo{B3: $I(B{:}\mathcal E_u)$, $I(B{:}\mathcal S_u)$ and
$S(B|\mathcal E_u)$ per received block on the C1/C2 re-propagated pure
covariances, the identity's residual as the purity check}.
```

Derivation: \(I(B{:}\mathcal S)-I(B{:}\mathcal E)=S(\mathcal S)-S(B\mathcal S)-S(\mathcal E)+S(B\mathcal E)\); purity gives \(S(\mathcal S)=S(B\mathcal E)\) and \(S(B\mathcal S)=S(\mathcal E)\), hence \(2[S(B\mathcal E)-S(\mathcal E)]=2S(B|\mathcal E)\).

### B3b (into Sec. page, "The light field on the computed collapse")

**Where.** After the sentence ending "...and the tail enters no result beyond this subsection." Insert:

```latex
The propagated light covariance is pure at every stored slice to
\todo{B3: $\Tr s(C_L)$ per channel on the stored clocks, two routes}
nats, so its exterior entropies are entanglement and the drive produces
none. The pairs the changing fields make are the departure
$C_L-P_{\rm band}(t)$ from the prepared-band projector at the slice's
fields, whose positive and negative eigenvalues match by number
conservation; its trace norm per channel, read at the formation burst
and on the quiet slices, separates formation light from transport
\todo{B3: the band-departure trace norm per channel on the stored
slices}.
```

### Computed versus assumed, for the Page construction after adoption

| Element | Status |
|---|---|
| \(\Gamma_k(\omega)\) from eq page-dirac-barrier; \(L\), \(\dot S_{\rm marg}\), \(\mathcal R=1.639\); limits 2 and 4/3; \(\mathcal R>1\) mode by mode | computed / derived |
| Page 1976 and partner benchmarks | computed on one integrator, marked for P3's two |
| Flat-chain quench (1.2, 1.3 percent, \(\mathcal R=1.97\) to 2.01) | computed (exact evolution against Landauer) |
| Held-exterior quench | marked (spec above) |
| Identity \(I(B{:}\mathcal S)-I(B{:}\mathcal E)=2S(B|\mathcal E)\); crossing = turnover | exact |
| \(S(B|X)>0\) on the frozen ledgers (no crossing reached) | computed (emisF6) |
| Register on re-propagated covariances | marked (spec below) |
| Purity defect; band-departure pair count | marked (spec below) |
| Second branch as \(S_{\rm cut}(R_t)\) measured; chain excesses 109 to 311 and about nine nats | measured on chains; cubic cut marked |
| \(f\) as the leading value of the measured share | measured where the flat coefficient holds; uniform cubic budget open |
| Achievability circuit | kinematic, inside the Gaussian class (spec below) |
| Quasi-steady emission at the instantaneous \(T_H\); depletion law; accessibility and complete release of the cut entanglement; \(o(S_0)\) retained entropy outside the trapped region | assumed |

### Specs

**Register (B3a).** On the eighteen-channel light histories at the \(16r_s\) cut, stored clocks 40 to 100 and the held continuation 100 to 500: packet basis \(D_u\) as documented for emisF6's ledger; \(B_j\) the packets received in \((u_j,u_{j+1}]\), \(\mathcal E_j=\oplus_{i<j}B_i\), \(\mathcal S_j\) the orthogonal complement. From the pure \(C_L\) (both C1 and C2 re-propagated routes) compute \(S(B_j)\), \(S(\mathcal E_j)\), \(S(B_j\mathcal E_j)\), \(S(\mathcal S_j)\), \(S(B_j\mathcal S_j)\); report \(I(B{:}\mathcal E)\), \(I(B{:}\mathcal S)\), \(S(B|\mathcal E)\) per channel with weights \(16k\), and the residual \(I(B{:}\mathcal S)-I(B{:}\mathcal E)-2S(B|\mathcal E)\), which must be at round-off (it is zero for any pure state; a nonzero value measures the covariance's purity loss). Expected at this size: \(S(B|\mathcal E)>0\) at every block, no crossing.

**Purity and pairs (B3b).** Per stored clock and channel: \(\Tr s(C_L)\) (purity defect, expected below \(10^{-6}\) nats; compare with the test packets' \(2\times10^{-11}\) norm defect); \(P_{\rm band}(t)\) the prepared band of \(\bar h(t)\) continued through tilted crossings as Sec. model-sources prescribes; \(\|C_L-P_{\rm band}\|_1\) and its positive and negative eigenvalue sums (must agree to the purity defect). Read at \(T/r_s=11,12.125,15,20,40,60,100\): the jump at formation is formation light; the slope on quiet slices is the drive's production, to be compared with emisF6's recession excitation bound.

**Cubic cut (B1b).** As P3's existing todo: on the stored back-averaged fields at the fine-grid \(R_t\), the cubic covariance of sea plus dust and the flat ball on the same site set; add the ratio \(S_{\rm cut}(R_t)/(\pi R_t^2/G)\) as the measured \(f\), and the direct-state budget \(E_f(C)\) on the same cut.

**Achievability circuit (B1c).** On the held exterior chain with its pocket modes: (i) choose the receiver's packet basis and a pocket basis (sign-reversed one-body states inside the root, Killing energies \(-\omega\)); (ii) enqueue: a one-particle rotation by angle \(\theta\) between an empty outgoing packet and a filled pocket mode with \(\sin^2\theta=n_k(\omega)\), one pair per step while \(S_{\rm out}<S_{\rm cut}(R_t)\) with \(R_t\) following the comparator's \(E_R(u)\); (iii) release: a swap between a stored pocket partner and a fresh empty outgoing packet, at the rate holding \(S_{\rm out}=S_{\rm cut}(R_t)\); (iv) report \(S_{\rm out}(u)\) against \(\min\{\mathcal B(u),S_{\rm cut}(R_t(u))\}\) (agreement within one packet's entropy) and the pocket energy the store carries against the Gibbs ceiling at the sector's measured energy.

---

## 4. B8: the exact discrete exterior of the rows and its lattice remainder

**Where.** Sec. results-exterior, after the sentence ending "...the geometric trapping radius staying at \(r_s\)." Insert the text below. In Sec. model-gravity, "with undetermined coefficients and no bound" becomes "with undetermined coefficients and no bound; the one computable lattice remainder of the exterior, the chain scheme's own, is in Sec.~\ref{sec:results-exterior}".

```latex
The rows also have an exact discrete exterior, whose lattice remainder
is a property of the scheme. On the chain the energy law is read as the
principle states it: the charge inside the sites $r_m\le r_n$ is the
flux through the bond leaving them, a reading $\mathcal F_{n+\frac12}$
of $rv^2/N^2$ built from the bond's radius and its two endpoint flows,
so in vacuum every exterior bond carries $E$ and the exterior is
algebraic. For the symmetric reading
$\mathcal F=\tfrac12(v_n^2+v_{n+1}^2)\,r_{n+\theta}$, the radius read at
$r_n+\theta a$, the smooth solution is
\begin{equation}
v_n^2=\frac{2GE}{r_n}\Bigl[1+c_1\frac{a}{r_n}+c_2\frac{a^2}{r_n^2}
+O\Bigl(\frac{a^3}{r_n^3}\Bigr)\Bigr],\qquad
c_1=\tfrac12-\theta,\qquad c_2=-\theta(1-\theta),
\label{eq:dr-discrete-exterior}
\end{equation}
so a radius read at the bond midpoint has no first-order remainder and
one read at a node has $c_1=\pm\tfrac12$. A node-weighted cumulative
scheme, the squared flow accumulated inward from infinity with the
increments $2GEa/r_m^2$, gives instead the exact profile
$v_n^2=2GE\,\psi_1(n)/a$, $\psi_1$ the trigamma function, whose
expansion $\psi_1(n)=1/n+1/(2n^2)+1/(6n^3)+\dots$ is the $\theta=0$
case through first order. Read through the rows at $N=1$ as a mass
function, $2G\mathcal M_{\rm eff}=rv^2$, the first-order term is an
effective exterior source $\rho_{\rm lat}=-c_1r_sa/(8\pi Gr^4)$ with
$p_r=-\rho_{\rm lat}$ and $p_\perp=\rho_{\rm lat}$, traceless and
falling as $r^{-4}$ like a radial Maxwell field of $GQ^2=-2c_1GEa$; the
second-order term gives $\rho_{\rm lat}=-c_2r_sa^2/(4\pi Gr^5)$ with
$p_\perp=\tfrac32\rho_{\rm lat}$ and trace $\rho_{\rm lat}$. Orders and
signs are scheme dependent; what is not is that a node-read scheme
carries the exterior's largest lattice correction, $O(a/r_s)$ at the
trapping radius and $\beta_{\rm PN}=1-2c_1a/r_s$, above the
$O(a^2/r_s^2)$ of the kernel and of the probe's dispersion, while a
midpoint scheme reduces it to that order. At one solar mass $c_1a/r_s$
is $5\times10^{-39}$ for $|c_1|=\tfrac12$. The implemented chain's
$c_1$ and $c_2$ are read on the stored slices \todo{B8: fit of
$r_nv_n^2/(2GE)-1$ on the exterior nodes of the stored slices to
$c_1a/r_n+c_2a^2/r_n^2$, fine-grid and back-averaged flows, the latter
against $-\tfrac94\sigma^2/r_n^2$}.
```

**Derivation notes.**
1. Ansatz \(v_n^2=2GE\,\phi(r_n)\), \(\phi(r)=r^{-1}(1+c_1a/r+c_2a^2/r^2)\); \(\tfrac12[\phi(r)+\phi(r+a)]=\phi+\tfrac a2\phi'+\tfrac{a^2}{4}\phi''\); multiply by \(r+\theta a\) and set equal to one: order \(a/r\): \(c_1-\tfrac12+\theta=0\); order \(a^2/r^2\): \(c_2-c_1+\tfrac12+\theta c_1-\tfrac\theta2=0\), giving \(c_2=-\theta(1-\theta)\). Numerical check at \(\theta=\tfrac12\), \(n=10\): residual \(3\times10^{-5}\).
2. Cumulative scheme: \(v_n^2=\sum_{m\ge n}2GEa/r_m^2=(2GE/a)\sum_{m\ge n}m^{-2}=(2GE/a)\psi_1(n)\). With midpoint radii the sum is \(\psi_1(n+\tfrac12)=1/n-1/(12n^3)+\dots\): no first-order term, the same statement as \(c_1=0\) at \(\theta=\tfrac12\).
3. Effective source: \(\mathcal M_{\rm eff}=E(1+c_1a/r+c_2a^2/r^2)\), \(\rho_{\rm lat}=\mathcal M'_{\rm eff}/(4\pi r^2)\) by eq dr-charge at \(N=1\), \(J=0\); \(p_r=-\rho\) from the stationary rows; \(p_\perp=-\rho-\tfrac r2\rho'\) from conservation; for \(\rho\propto r^{-4}\) this is \(p_\perp=\rho\) (traceless), for \(r^{-5}\) it is \(\tfrac32\rho\).
4. Maxwell reading: \(2G\mathcal M_{\rm eff}=r_s+c_1r_sa/r\) against \(r_s-GQ^2/r\). PPN: \(-g_{t_st_s}=1-2U+2U^2(2c_1a/r_s)\) with \(U=GE/r\), so \(\beta_{\rm PN}=1-2c_1a/r_s\).
5. The kernel term from Appendix charge: \((W_\ell^\dagger v)^2r/(2GE)=1-\tfrac94\sigma^2/r^2\), i.e. \(c_1=0\), \(c_2=-\tfrac94(\sigma/a)^2=-36\) at \(\sigma=4a\).

**Cheap check on a stored slice (B8).** Reference collapse, slices \(T/r_s=15\) and 40 (and the size-32 and size-128 members at matched clocks): take the exterior nodes \(r_n>R_{90}+3\sigma\) up to the charge-reading shell at \(16r_s\); \(E\) the charge read there. Form \(\epsilon_n=r_nv_n^2/(2GE)-1\) on the fine-grid flow and \(\bar\epsilon_n\) on the back-averaged flow; also \(N_n-1\). Least-squares fit of \(\epsilon_n\) and \(\bar\epsilon_n\) to \(c_1a/r_n+c_2a^2/r_n^2\). Expected: \(N_n-1=0\) to solver tolerance; fine grid \(|c_1|\le\tfrac12\) with the value identifying the scheme (an algebraic solve returns \(c_1=c_2=0\)); back-averaged \(c_1=0\), \(c_2=-36\pm\) fit error at \(\sigma=4a\) and \(-9\) at \(\sigma=2a\). A nonzero fine-grid \(c_1\) decides whether eq dr-discrete-exterior's first-order term is present in the production scheme. This fit is panel (d) of Fig. star-collapse below.

---

## 5. B9: figures to restore, as panels of the three existing figures

**Fig. star-collapse, new panel (d)** (format of P1's post-Newtonian residual panel, log radius). Curves: \(r_nv_n^2/(2GE)-1\) on the fine-grid flow (solid) and on the back-averaged flow (dashed) at \(T/r_s=15\) (black) and 40 (red), reference collapse \(\sigma/a=4\); the kernel prediction \(-\tfrac94\sigma^2/r^2\) (dotted); \(N_n-1\) (thin grey); one curve each from the size-32 and size-128 members at matched clocks. Data: stored fine-grid \(N_n,v_n\) and the back-averaged fields at those slices; \(E\) at the \(16r_s\) shell; \(R_{90}\) at each slice to set the inner edge. Caption sentence: "(d) Exterior residuals against Eq. dr-exterior on the vacuum nodes: the fine-grid flow (solid) shows the scheme's own remainder, Eq. dr-discrete-exterior, the back-averaged flow (dashed) the kernel's \(-\tfrac94\sigma^2/r^2\) (dotted)."

**Fig. page-v4, optional panel (d)** (the B2 quench). Curves: energy and entropy entering the exterior beyond \(3r_s\) against time for \(k=1\) and 2, with the Landauer slopes of eq page-currents as straight lines; inset the fitted \(\mathcal R\) per channel. Data: the new quench run. Caption sentence: "(d) Exact Gaussian quench on the held exterior: a uniform-flow reservoir at \(T_H\) inside the barrier, vacuum outside; slopes are the rates of Eq. page-currents."

**Fig. page-conditional, panel (a) restyled** (P2's format). Curves of \(S_{\rm esc}/S_0=\min\{\mathcal R(1-y),fy\}\), \(y=(1-t/t_*)^{2/3}\), at \(f=1\) for \(\mathcal R=1\) (reversible emission), \(4/3\) (a count \(\propto\omega^2\)), \(1.4847\) (Page's photons and gravitons) and \(1.639\) (this model's channels), with the \(f\) band \(0.7883<f<0.9831\) and the dashed \(f=0.887\) kept at \(\mathcal R=1.639\); dots at the maxima \(S_P/S_0=f\mathcal R/(\mathcal R+f)=0.500,0.571,0.598,0.621\), dotted Page times \(0.646,0.568,0.538,0.5105\), and \(0.4773\) for \(f=0.887\). Data: none from runs. Caption adds: "Radiation carrying more entropy per unit of energy fills the bound sooner, so larger \(\mathcal R\) turns over earlier; neither \(\eta\) nor \(a/\ell_P\) enters the time, only \(f\) and \(\mathcal R\)."

**Fig. page-conditional, new panel (c)** (P2's register format). Curves: \(I(B{:}\mathcal E_u)\) (orange), \(I(B{:}\mathcal S_u)\) (blue) and \(S(B|\mathcal E_u)\) (black) per received block against \(u\) at the \(16r_s\) cut, channels \(k=1,2\) weighted \(16k\), stored clocks 40 to 100 and the held continuation to 500; grey line the identity residual. Data: the C1/C2 re-propagated pure covariances and the documented packet basis (the register spec). Caption sentence: "(c) The register on the computed histories: \(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\) exactly, so a crossing of the two would be the turnover; on these records the conditional entropy stays positive." Panel (b) stays as agreed.

---

## 6. B10: two cheap checks

**Check 1: modular slope on stored slices (\(\beta_{\rm ent}(1)=\pi\)).**
- Flat control: the massless light chain \(h_{0,k}\), \(k=1\), \(v=0\), \(C=P_0\) on \(n_{\rm ch}=4096\); half-space \(A=\{r_n>R\}\), \(R=16r_s=1024a\); \(k_A=\log(C_A^{-1}-1)\) with eigenvalues clipped at \(10^{-14}\); \(\beta_{\rm ent}(n):=(k_A)^{\sigma_1}_{n,n+1}/(h_{0,k})^{\sigma_1}_{n,n+1}\), the \(\sigma_1\) component of the \(2\times2\) block of the bond \(n\) steps inside the cut. Expected: \(\beta_{\rm ent}(1)=\pi a\) to \(10^{-4}\) and slope \(2\pi a\) per bond for \(n\ll R\) (the \(\sigma_1\hat p_r\) hop is two decoupled half-filled hopping chains after the gauge \(c_n\to i^nc_n\); the centrifugal term \(k/R\sim0.002/a\) is negligible). This reproduces P3's half-space boost statement on the chain.
- Stored slices: \(C_L(t)\) at \(T/r_s=15,40,100\), cuts \(R=8,16,30\,r_s\); same readout. Report \(\delta\beta(n)=\beta_{\rm ent}(n)/[2\pi(n-\tfrac12)a]-1\) for \(n\le20\), beside the Mirsky control \(\|C_A-P_{0,A}\|_1\) restricted to those bonds. Expected: \(\delta\beta\) at the level of the state's local departure from the sea; no flow factor appears, since a uniform flow leaves the sea and its half-space modular operator unchanged (Sec. results-vacuum), so the exterior's redshift sits in the time map and not in the lattice-frame modular slope.
- Diagnostic variant at the trapping cut, \(R=R_t\) (fine grid), slices 15 to 40: compare \((k_A)_{n,n+1}\) bondwise with (i) \(2\pi(n-\tfrac12)a\,(h_0)_{n,n+1}\), the vacuum boost about the cut, and (ii) \((2\pi/\kappa_H)(\bar h)_{n,n+1}=4\pi r_s(\bar h)_{n,n+1}\), thermal at \(T_H\) in the Killing generator; report the depth at which the ratio to (ii) departs from constancy. An incoming-vacuum (Unruh-type) exterior is thermal in the outgoing sector only, so (ii) should hold at best on the outgoing branch; the result is a diagnostic of the conversion state on the computed slices, not a test of \(T_H\).
- Cost: one diagonalization of \(C_A\) per slice, cut and channel.

**Check 2: tidal eigenvalues at and inside the trapping surface.**
- Formula (derived from the Ricci equation of the flat slicing with P3's \(K_{ij}=+\tfrac12\mathcal L_n\gamma_{ij}\) and the evolution rows eq dm-evolution): for the normal observer, \(E_{ij}=n^\rho n^\sigma R_{\rho i\sigma j}=(\operatorname{tr}K)K_{ij}-K_{ik}K_{kj}-8\pi G[S_{ij}-\tfrac12\delta_{ij}(S-\rho_\ell)]\), the "constraint reading", holding where the sources are conserved; in spherical symmetry \(E_{\hat r\hat r}=2K_rK_t-8\pi G[S_{rr}-\tfrac12(S-\rho_\ell)]\), \(E_{\hat\theta\hat\theta}=K_t(K_r+K_t)-8\pi G[S_{\theta\theta}-\tfrac12(S-\rho_\ell)]\), \(S=S_{rr}+2S_{\theta\theta}\). Direct reading from consecutive slices: \(E_{ij}=-\tfrac1N(\partial_t+\mathcal L_v)K_{ij}+\tfrac1N\partial_i\partial_jN+K_{ik}K_{kj}\) with \((\mathcal L_vK)_{\hat r\hat r}=vK_r'+2v'K_r\), \((\mathcal L_vK)_{\hat\theta\hat\theta}=vK_t'+2vK_t/r\). Checks: vacuum PG gives \(E_{\hat r\hat r}=-r_s/r^3\), \(E_{\hat\theta\hat\theta}=+r_s/(2r^3)\) (Schwarzschild, boost invariant); trace \(=4\pi G(\rho_\ell+S)\).
- Inputs: stored fine-grid \(N_n,v_n\) (centred differences for \(K_r=v'/N\), \(K_t=v/(rN)\)), the kernel-averaged \(\rho_\ell\), \(J_{\ell r}\), and the stored kinetic stress components at the slices where they exist (Sec. results-diagnostics); consecutive slices for the direct reading.
- Readouts: both readings at \(R_t\), at \(R_{50}\) and at \(r=2a\), on \(T/r_s=12.125,13.25,15,20,40\) of the reference collapse and the \(\sigma/a=8\) run; their difference, which is the evolution residual already reported as medians; units \(1/r_s^2\) and \(1/a^2\). Expected: at \(R_t\) the Schwarzschild values \(-1/r_s^2\) and \(+1/(2r_s^2)\) (\(2.4\times10^{-4}/a^2\) and half of it at \(r_s=64a\)) within the matter's correction \(8\pi G[\ldots]\); inside, values to be set beside P3's sampled angular curvature \(0.14/a^2\) and the lattice scale \(1/a^2\). Report also \(K_t^2(0)\) against \(8\pi G\rho_\ell(0)/3\) (regular centre).
- Optional hooks (one sentence each, with marks): Sec. singularity, after "...a sampled angular-curvature maximum of \(0.14/a^2\) near the centre at \(T/r_s=13.25\)": "and tidal eigenvalues of the normal observer at the trapping radius and at \(R_{50}\) of \todo{B10: tidal eigenvalues on the stored slices, constraint and direct readings} in units of \(1/a^2\)"; Sec. discussion, after "no curvature barrier is met at it": "the tidal eigenvalues there being the Schwarzschild values within the matter's share \todo{B10}". Cost: finite differences on stored arrays.