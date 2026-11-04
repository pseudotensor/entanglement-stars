# Derivations and proof details removed from the paper for length

This file holds derivations and proof details that were removed from the printed paper for length. They support results the paper states and cites to this repository, but the paper does not depend on them. Each section names the paper section and the result it supports. The text is copied as it stood in the paper source, in LaTeX or in the plain notation of the source comment.

Part A holds the text that stayed in the paper's LaTeX source as comments next to the printed pointer. Part B explains where the longer passages removed before 4 November 2026 are archived. Part C holds three passages from earlier drafts.

Section numbers refer to the submitted version: II the model, III the hydrodynamic limit, IV exterior, temperature and entropy, V the quantum spherical sector, VI collapse, VII radiation, VIII astrophysical scaling, and Appendices A (local area estimates), B (boundary charge) and C (Gauss laws and spherical rows).

## Part A. Text kept next to the pointers

### A1. Sec. III, bond averaging of the clock and the second-gradient polarization

1/2[N(x)+N(x+d)] = N(x+d/2) + (1/8) d^2 N''(x+d/2) + O(d^4). Equation~\eqref{eq:dm-reference} therefore retains the bulk polarization \(W_2\) of Sec.~\ref{sec:model-sources} through second gradient order on slowly varying fields, the first omitted bulk terms having fourth gradient order. For flat preparation the polarization's first-gradient kinetic flux is a curl, zero for occupations scalar within each Dirac band, so the stress is kinetic. Covariant source conservation is not a consequence of smooth fields but a premise, which the third step states.

### A2. Sec. VI A, continuation of the followed reference and its selection margin

Supports the printed statement that the continuation is a single maximum-overlap assignment with selection margin at least 0.806.

Ordering of the followed levels below the rest is unnecessary. A reference-selection switch is not covered by this differential identity and must be distinguished from held-step and field-solve errors.

The stored float64 spectra report ordering inversions on 49 channel-slice pairs in channels 1-4, within T/r_s=12.75-14.5 and, for channel 1, 20.875-21.0. The smallest stored followed-unfollowed energy difference in absolute value is 1.8662e-5. This is a stored-data minimum, not an established pathwise gap. The spectra alone establish neither identification with the prepared band nor indefiniteness of the Hessian restricted to clock and flow variations.

Selection: without sub-stepping, block rotation or intermediate projector; smallest squared projection of a followed eigenvector onto the reference 0.903, largest of an unfollowed one 0.097 (channel 2, T/r_s=13.375); the margin is unmeasured along the path between slices.

### A3. Sec. VI A, the chain second-gradient functional

Supports the comparison with "the chain second-gradient functional" cited to this repository.

This statement is at fixed positive mass and does not establish the finite-chain comparison. For that comparison the independently derived chain second-gradient polarization is the bulk functional W_{2,ch} = -(1/8) int dr [A v'^2 + 2B v v' + C v^2]/N per two-component channel, with A, B, C zone integrals of the frozen symbol at the local angular mass k/r; source and Hessian sums carry the copy weights 16k. The angular-gradient parameter obeys |d_r(k/r)|/(m^2+k^2/r^2) <= 1/(2mr) <= 0.05 at r >= r_min = 10/m = 50a for ma=0.2, against 0.96 at the first site. The bulk functional is not a prescription at the open end, where the missing bond polarizes even a uniform flow.

Both norms are in units a^-1, using the Euclidean operator norm of the finite-site energy Hessians in dimensionless applied clock and flow fields, restricted to the tested probe spaces. The probes have sin^2 windows supported in [r_s,5r_s] and [r_s,9r_s] at the tested wavevectors; the core r<r_min is read on the followed band alone, against no bulk law.

### A4. Sec. VI A, Newton-solver controls

Only specified Newton-solver controls use midpoint covariance, in two-stage source continuation.

### A5. Sec. VI C, the dust's radial momentum and the lapse law in the interior

Supports the interior readings cited to this repository (half-count radius, central density, lattice momentum weight).

The radial momentum of the dust inside r_s is read by a windowed transform of the stored columns with the window's own norm as weight; it carries 15.4 percent of its weight at pi/4 < a|p_r| < 3pi/4; this is a reading of the weight and not of the source-conservation residual. The clock on these slices obeys the lapse law: Eq.~\eqref{eq:sc-clock-exact} integrated on the fine-grid fields reproduces the clock's fall from \(R_{50}\) to the outer trapping radius within 1 to 3 percent.

### A6. Sec. VI D, the Hessian of the reference functional and the gap-based bound

Supports "\(W_2''\) is negative semidefinite wherever \(N\) is bounded below" and the bound \(2\|\delta h\|_{\rm HS}^2/\delta_{\rm b}\).

For Eq. dm-reference the Hessian is W''=W_2'', the other two terms being linear, with D^2 W_2[u](du,du) = -(1/2) int |dF - dN F/N|^2_T N^{-1} d^3x, |X|^2_T = X_ij T_ij,kl X_kl <= tau_m |X|^2, tau_m <= 2/(3 sqrt3 m a^3). For each locally fixed isolated followed set D^2 E_cont[u](du,du) = -2 sum_{i in F, a notin F} |<a|dh|i>|^2/(e_a - e_i), so |D^2 E_cont| <= (2/delta_b) ||dh||_HS^2 with delta_b = min |e_a - e_i| > 0. Channel responses carry their copy weights 16k. Ordering guarantees a negative-semidefinite response; mixed denominators alone do not establish indefiniteness on the allowed field variations. The stored spectral minimum (1.8662e-5, Sec. A2) is a stored-data minimum; established lower bounds require spectral and assembly error bounds. Channel-1 check: eigen-residual at most 5.4e-15, sine of the angle to the stored columns at most 1.5e-11, an a posteriori float64 residual check, not an interval bound.

### A7. Sec. VI D, conditions of the continuation statement and the full-Jacobian test

Supports "full statement, centre handling, and seeding rule" and the reduction of the full-Jacobian test to an inverse bound on \(J=I-D_xG\).

The reference-derivative bound through the isolation delta_b applies on a single continuation branch. The isolation margin must remain positive throughout the established field neighbourhood and trajectory tube, and validated selection margins must keep the actual continuation rule, at every sub-step it takes, on that branch. The quadratic response bound alone is not a bound on the full constraint Jacobian. A followed-unfollowed crossing prevents this gap-based bound, not necessarily continuation itself. A change of selected projector requires separate jump accounting. Failure of a sufficient inequality means not verified, not singular. The full-Jacobian test can use the self-consistency map G(x) on the applied site fields x, the back-averaged fields the rows return from the state's sources with the reference evaluated at x: within one differentiable realization, provided the held-source row Jacobian is invertible and G solves those same rows, F_u is invertible exactly when J = I - D_x G is. For a finite realization an approximate inverse Y, with a validated residual q = ||I - YJ||_inf < 1, establishes ||J^-1||_inf <= ||Y||_inf/(1-q). The same construction for Eq. dm-reference uses W_2'' as the reference response on its declared realization. No stored slice carries this bound. The stored histories pass the clock tests above.

Remaining need, as stated for the continuation result: to establish continuation by two independent calculations, we still need seed, total-defect and growth bounds valid throughout each interval.

### A8. Sec. VIII B, conditions of the recession inference

Supports the positive exterior work of a receding trapping radius.

The recession inference also requires the light field's signed null-source contribution, a chi-uniform positive margin 1-2G M' > 0, and controlled finite-zone and moving-reference defects.

### A9. Sec. VIII D, the retained-spectrum entropy bound

Supports "at most \(D_{\rm ret}e\) positive levels lie at or below any energy \(e\) (derivation in Ref.)".

Define \(E_s^B=\sum_j\mu_j(n_j-b_j)\) and let \({\rm SR}\) contain the levels with \(b_j\ne\mathbf1_{\mu_j<0}\). Then
\begin{equation*}
E_{\rm dep}=E_s^B+2E_{\rm SR}^{\rm occ}\ge|E_s^B|,\qquad
E_{\rm SR}^{\rm occ}=\sum_{j\in{\rm SR}}|\mu_j|\,|n_j-b_j|.
\end{equation*}
The signed term is reconciled with the charge accounting at the same weighting, remainders retained. To use the once-counted received energy \(E_{\rm rec}\) as the Gibbs budget, the comparison case additionally assumes positive-deposit affordability,
\[
E_{\rm dep}\le E_{\rm rec},
\quad\text{equivalently}\quad
E_{\rm SR}^{\rm occ}\le\tfrac12(E_{\rm rec}-E_s^B).
\]
This is not a consequence of signed charge conservation. If only an interior mode subset is counted, an entropy bound for every other retained mode, including radiation in flight, must be added. The retained spectrum turns this bound into one number. Let \(n_0\) be the number of zero levels, \(0<\epsilon_1\le\dots\le\epsilon_M\) the positive values of \(|\mu_j|\), and \(D_{\rm ret}=\max_{j\le M}j/\epsilon_j\) (zero if \(M=0\)), so that at most \(D_{\rm ret}e\) positive levels lie at or below any energy \(e\). The level sum equals \(\int_0^\infty\beta N_+(e)(1+e^{\beta e})^{-1}de\), \(N_+\) the positive-level counting function, at most \(\pi^2D_{\rm ret}/(12\beta)\), and minimizing over \(\beta\) gives
\begin{equation*}
S_{\rm esc}=S_{\rm ret}\le n_0\ln2+\frac{\pi}{\sqrt3}\sqrt{D_{\rm ret}E_{\rm dep}},
\end{equation*}
with \(E_{\rm rec}\) in place of \(E_{\rm dep}\) under the affordability assumption. It establishes neither that budget for the complete retained field nor the comparison case's accessible store.

### A10. Sec. VIII D, the copy-twirl argument (fragment)

Only this fragment of the argument stayed in the source: "the 16k copies of a channel enter the sources and the second-order charge only through collective bilinears commuting with the copy rotations, so ..." The printed paper completes the sentence: the light output is invariant under every U(16k) rotation of the heavy copies. The full argument is not in this file.

### A11. Sec. VIII D, scope of the comparison case and the firewall question

The finite-band L and R have no narrow bound. The figure of the Page curve plots the saturated comparison case. The firewall question, the exchange-current response that locates a late quantum's partner, and the independent-pair release model behind panel (b) of that figure are set out in the archived passages 9.2 to 9.4 of `notes/REPO_DERIVATIONS_20261102.md`.

### A12. Sec. VIII D, what the once-counted shell output still needs

The once-counted shell output, the light field's outgoing-branch Killing-energy current through the shell with each crossing counted once, and the received-time map that assigns crossings to receiver intervals remain to be matched to the independently read received energy \(\int L\,du\), and net shell transport is not substituted for them.

### A13. Sec. VIII D, conditions of the five-slice recession test

t/a = 1096 to 1112, light-reference weight 16 not the paper's 32, half-weight light counterterm taken as W_2 (an assumption of the test), two interpolants of the stored enclosed energy, Krawczyk-box and interval-Newton verifications.

### A14. Appendix A.4, full hypotheses of the finite-lattice estimate

L even; P_0^(L) has no zero mode; weighted Fourier norm of the gradients at most F_1 eps; nonwrapping ball c_r L <= r <= 0.45 L; P = 1_(-inf,0)(h) with any orthogonal zero-mode filling; C depends only on (zeta_*, F_1, c_*, n^*, v^*, c_r).

### A15. Appendix A.5, the localization argument for the gap

Supports "the gap \(\delta=\mathsf g_*-E_R-D_R\) with costs \(E_R,D_R=O(\epsilon^{1/2})\)".

A tensor cosine partition of half-width R=O(eps^-1/2), comparing localized operators with frozen symbols, gives ||h psi|| >= (g_* - E_R - D_R)||psi|| for every psi, with freezing and partition costs E_R, D_R = O(eps^1/2), so positive delta = g_* - E_R - D_R establishes the gap. A massive configuration obeying (H1) with g_* > 0 meets (H2) and (H3) along the linear path from the flat sea once g_* - E_R - D_R stays positive at its bounds.

### A16. Appendix B, the rate at the trapping radius and the anchored defect

Supports the derivation of the defect \(\mathcal D_b\) and the shell balance (the residual expressions are in archived passage 10.398).

This positive square is the rate supplied by the continuum evolution rows, the lattice transport weight and the changing-reference source being retained when it is identified with the actual mass rate.

At fixed r the transverse evolution row, with the charge and momentum rows eliminating w' and N', gives the balance; F = N[w rho_l + (1+w^2) J_lr + w p_r]; D = r^2 v R_t/(G N) plus terms linear in the rows' residuals; N D at any radius is its value at an outer boundary b minus the weighted defect between them.

### A17. Appendix C, interior masses as diagnostics

No result of the paper uses this. The linearized mass is N delta M = Tr(h_r delta C) - <u_r, delta r>, and positivity does not follow.

### A18. Appendix C, the uniform-shift and rigid-turn invariances, the momentum law's sign, and b = 0

Supports "Proofs of the uniform-shift and rigid-turn invariances, of the momentum law's sign, and of \(b=0\)".

Momentum's outgoing-bond flux makes its integral invariant under finitely supported field changes, and symmetry of the linearized laws then makes the sources invariant under uniform flow shifts. The same two steps for the moment's flux give invariance under the rigid turn v -> v + omega x x. The +v.P coupling fixes the momentum law's sign by stationarity in the flow.

The response identity and the rigid turn. For the rigid flow v = omega x x, relative to zero flow, Delta rho_l = -b|omega|^2/(8 pi G N^2), so invariance requires b = 0.

## Part B. Longer passages archived before 4 November 2026

The note `notes/REPO_DERIVATIONS_20261102.md` holds, verbatim, the LaTeX of every passage removed from the printed paper between 1 and 4 November 2026. That covers the full derivations, the earlier longer forms of shortened paragraphs, and the record-level values. Its section numbers follow the draft of that week, which maps to the submitted version as follows.

| Draft section | Submitted section |
|---------------|-------------------|
| II A to II C, the model | II A to II C |
| II D, the Gauss laws, the metric and the continuum limit | III |
| II E, the spherical sector | V |
| III, properties of the fermion and of the vacuum exterior | IV |
| IV, collapse | VI |
| V, radiation, correlations and entropy | VII |
| VI, astrophysical scaling | VIII |
| VII, discussion | IX |
| Appendices A, B, C | Appendices A, B, C |

The paper source names archived passages for these pointers: 10.155 (introduction), 10.183 (Sec. VIII D, recession test), 10.362 (Appendix C, the bracket and its decomposition), 10.367 (Sec. IV D, scheme and post-Newtonian bookkeeping), 10.370 (Sec. VI, maximum mass and radius), 10.372 and 10.373 (Sec. VII B, the static-frame Dirac reduction and mode counting), 10.375 and 10.376 (Sec. VII C), 10.386, 10.388, 10.390 (Sec. VIII D), 10.398 (Appendix B), 10.399 to 10.401 (Sec. VIII D), 10.405, 10.407, 10.567 (Appendix B, the averaging kernel) and 10.568 (Appendix A).

## Part C. Text from earlier drafts of the paper

These passages stood in earlier drafts of the paper and were removed for length. They are copied verbatim.

### C1. Appendix C, the six energy-row and seven momentum-row second-gradient terms

Supports "the energy row six second-gradient terms and the momentum row seven (listed in Ref.)".

With the flow entering only through its derivatives and the laws of degree zero (Sec.~\ref{sec:model-gravity}), cubic symmetry, inversion, and time reversal leave the energy row six second-gradient terms, \(\theta^2/N^2\), \(S_{ij}S_{ij}/N^2\), \(\sum_iA_{ii}^2/N^2\), \(W_{ij}W_{ij}/N^2\), \(N_iN_i/N^2\), and \(\nabla^2N/N\), and the momentum row seven, \(\partial_j\theta/N\), \(\nabla^2v_j/N\), \(\partial_j^2v_j/N\), \(N_j\theta/N^2\), \(N_i\partial_iv_j/N^2\), \(N_i\partial_jv_i/N^2\), and \(N_j\partial_jv_j/N^2\), with no sum over \(j\) in \(\partial_j^2v_j\) and \(N_j\partial_jv_j\): thirteen coefficients, which reciprocity reduces to the five of Eq.~\eqref{eq:dm-density}.

### C2. Sec. VIII D, the windowed charge identity

Supports "Between consecutive solved configurations the windowed charge identity (Ref.)". The symbolic derivation of the identity in its differential form, \((N\dot{\mathcal M})'=4\pi r^2(N\dot\rho+v\dot J)\), is in `notes/FableSub_clockgrowth_v1.md`.

Between consecutive solved configurations the windowed charge identity gives \(N_{r,j}\delta_j\mathcal M(r)=\sum_ag_a\Tr(h_{a,r,j}\delta_jC_a)-\langle\bar u_{<r,j},\delta_jr_{\rm tot}\rangle+e_{r,j}\), with \(\delta_jX=X_{j+1}-X_j\), \(g_a\) the copy weight, \(h_{a,r,j}\) the windowed generator of species \(a\) at stored slice \(j\), \(r_{\rm tot}\) the copy-weighted reference, and \(e_{r,j}\) the increment remainder with the source, solve, and scheme errors. In the differentiable limit the state and reference terms are \(\Xi_r\) and \(\Xi_r^{\rm ref}\), so at a simple fine-grid trapping radius \(N_R\beta_R\dot E_R=\Xi_R-\Xi_R^{\rm ref}\) with \(E_R=R/(2G)\), \(\beta_R>0\), and the exterior work \(\dot E-N_R\beta_R\dot E_R\) is not zero by conservation alone.

### C3. Appendix A, the direct test supplies d_*

Supports "The direct test of Sec.~\ref{sec:page} supplies \(d_*=2\kappa_f/\zeta+O(C_\varphi^2)\) (sketch and constants in Ref.)". The constants themselves are not in this repository.

Mirsky's inequality with the 2-Lipschitz singular-value function bounds the boundary rows by \(2\sqrt{m_B}\|X-X_0\|_2\). The direct test of Sec.~\ref{sec:page} supplies \(d_*\): \(\|\varphi C-P_0\varphi\|_2^2\le E_\varphi(C)/\zeta\) and the kernel bound on the window's edge give \(d_*=2\kappa_f/\zeta+O(C_\varphi^2)\). Only the gap of \(h_0\) enters, no gap of the evolving generator.
