> Owner word rule (#1287) applied to the NEW blocks by code/wordrule_v3.pl (OLD blocks are the paper's current text, untouched). Word substitutions only: certified/certifies/certificate -> established/establishes/bound; enclosures -> bounds. The three remaining \todo marks (pairs 9 and 10) are items13final's signed lookup marks (its 'Lookups' section), still open.

# Items 1-3: FINAL paper text for paperapi (v1)

Assembled by busi code/items13_assemble_v1.pl from the signed sources, verbatim (no retyping):
- pairs 2-7, 9-12, 16, 19: reviewer A items13final (5fc294db), OLD and NEW as printed there;
- pair 1, 17, 18: reviewer B I3routeA2 (9cb6b109); pair 14: reviewer A I3routeF (144b4d71) with reviewer B's one-sentence amendment;
- pair 13: reviewer A sepF2 (85276a30); pair 8 and 15: reviewer B merge (c9d10358) with reviewer A's amendments (33b2e297);
- pair-8 numbers from codeB I1_weighted_v8_v1.json (6f514c44), copy-weighted totals, rounded outward (Delta r: max over slices of value plus the a-vs-2a realization difference; norms: min-down/max-up over T/r_s 11, 15, 40, widened by the stated error terms).
For each pair: OLD = the text to replace (from items13final); NEW = the final text.

### 1. Abstract

OLD
```latex
On every stored slice the computed collapse has a positive clock and an invertible constraint Jacobian, and rays cross the regular centre.
```
NEW (FINAL)
```latex
On every stored slice the computed collapse keeps a positive clock, and traced rays cross the regular centre.
```

### 2. Introduction, section map

OLD
```latex
Sections~\ref{sec:star} and \ref{sec:page} do not assume it: they evolve the lattice fermions on the partial-wave chain model of Sec.~\ref{sec:star} with the clock and flow re-solved at every step from the same constraint equations, so the computation continues where the hydrodynamic reading of the fields ends.
```
NEW
```latex
Sections~\ref{sec:star} and \ref{sec:page} do not assume it: they evolve the lattice fermions on the partial-wave chain model of Sec.~\ref{sec:star} with the clock and flow re-solved at every step from the same constraint rows and the sources that section specifies, so the computation continues where the hydrodynamic reading of the fields ends.
```


### 3. Sec. model-gravity, the \(W_2\) sentence (item 1)

OLD
```latex
All three sources are now defined, and their lattice corrections are controlled: on slowly varying fields the lattice reproduces \(W_2\) of Sec.~\ref{sec:model-sources} through second order, its own corrections entering only at fourth order in gradients (Ref.~\cite{McKinneyCode2026}).
```
NEW
```latex
All three sources are now defined, and their lattice corrections are controlled: on a bond of displacement \(d\) the assignment of declaration one carries \(\tfrac12[N(x)+N(x+d)]=N(x+\tfrac d2)+\tfrac18d^2N''(x+\tfrac d2)+O(d^4)\), an even series, and in the smooth massive cubic bulk its second-order term changes the prepared band's energy only by a boundary divergence, so endpoint and midpoint assignment give the same second-gradient band-energy functional. Equation~\eqref{eq:dm-reference} therefore retains the bulk polarization \(W_2\) of Sec.~\ref{sec:model-sources} through second gradient order on slowly varying fields, the first omitted bulk terms having fourth gradient order (Ref.~\cite{McKinneyCode2026}).
```


### 4. Sec. results, opening

OLD
```latex
The computed collapse of Sec.~\ref{sec:star} does not rest on these results, since it evolves the same lattice equations directly.
```
NEW
```latex
The computed collapse of Sec.~\ref{sec:star} does not rest on these results, since it evolves the chain fermions under the same constraint rows directly.
```


### 5. Sec. results-vacuum, the gap criterion on the computed history (item 3)

OLD
```latex
At \(ma=0.2\) that criterion is met with Appendix~\ref{app:local}'s field bounds and an independently supplied pathwise frozen-gap lower bound (values in Ref.~\cite{McKinneyCode2026}). \fnd{The trapping-radius condition alone does not supply this global lower bound, these gap-criterion inputs have not been rigorously bounded on the computed histories, and the direct-state budget above uses the gap of \(h_0\), not a gap of the evolving \(h\).}
```
NEW
```latex
On the reference collapse the frozen-symbol envelope over flow orientations, the worst case of Appendix~\ref{app:local}'s \(\mathsf g(x)\) on each sphere, is positive at every stored site at release and at the first stored slice with a trapping radius, \(T/r_s=12.125\), and first turns negative at \(T/r_s=12.375\): through \(T/r_s=40\) it is negative at 1731 of the 657\,408 stored site--slice pairs, each where the back-averaged flow exceeds the back-averaged clock, and at \(T/r_s=15\) at 22 of 2048 sites, with minimum between \(-9.775114\times10^{-3}\) and \(-9.775112\times10^{-3}\) in lattice units. A positive envelope therefore cannot supply the gap premise on the computed histories, which the direct-state budget covers instead; the result decides neither the chains' spectral gap nor the prepared band's separation \(2N\varepsilon\) from the upper band, positive at every positive clock. \fnd{The trapping-radius condition alone does not supply a global lower bound, and the direct-state budget above uses the gap of \(h_0\), not a gap of the evolving \(h\).}
```


### 6. Sec. star, opening

OLD
```latex
Everything analytic to this point, Secs.~\ref{sec:model-gravity} and \ref{sec:results}, was derived in the hydrodynamic continuum limit, and the computation does not assume that limit: it evolves the lattice fermions on the partial-wave chain model below, with the clock and flow re-solved at every step from the same constraint rows applied to the averaged matter, measured against the vacuum it prepares (Sec.~\ref{sec:model-sources}). It therefore stays valid where the slow-variation premise fails.
```
NEW
```latex
Everything analytic to this point, Secs.~\ref{sec:model-gravity} and \ref{sec:results}, was derived in the hydrodynamic continuum limit, and the computation does not assume that limit: it evolves the lattice fermions on the partial-wave chain model below, with the clock and flow re-solved at every step from the same constraint rows applied to the averaged matter, measured against the level-following reference of Sec.~\ref{sec:results-lattice}. It therefore continues where the slow-variation premise fails.
```


### 7. Sec. star, chain definition (item 2)

OLD
```latex
Its sites are \(r_n=na\), \(n\ge1\), with no site at the origin and no bond inward from the first site, and the flow multiplies the two-step radial current \(Q_r=[T_r^2-(T_r^\dagger)^2]/(4ia)\), \(T_r\) the shift by one site, with the bond-midpoint clock and flow weights of the cubic declaration (Ref.~\cite{McKinneyCode2026}). The assigned densities couple to back-averaged radial fields.
```
NEW
```latex
Its sites are \(r_n=na\), \(n\ge1\), with no site at the origin and no bond inward from the first site, and the flow multiplies the two-step radial current \(Q_r=[T_r^2-(T_r^\dagger)^2]/(4ia)\), \(T_r\) the shift by one site, with the endpoint-averaged clock and flow weights of Eq.~\eqref{eq:dm-hamiltonian} (Ref.~\cite{McKinneyCode2026}). The assigned densities couple to back-averaged radial fields. The chains are not a compression of the cubic operator: a normalized spinor \(\psi\) on the first site of channel \(k\) has \(\|h_{0,k}\psi\|^2=m^2+(4k^2+1)/(4a^2)\), above the cubic operator's bound \(m^2+3/a^2\) for \(k\ge2\), so no isometry intertwines the two operators, and no cubic state is reconstructed from the chains below.
```


### 8. Sec. star, method: the reference (item 2)

OLD
```latex
The reference collapse uses step-start covariance minus a level-following projector continued from the previous accepted fields to each trial field, called the fixed reference projector below.On all 18 stored interior slices of the reference collapse, at two
averaging widths over flat-slice lattice time \(T/r_s=12\)--40, its fixed-reference-projector histories' sources and fields
agree with Eq.~\eqref{eq:dm-reference}'s law far inside the step controls' spread. The fixed reference projector's departure from the prepared band is almost exactly the second-gradient polarization that Eq.~\eqref{eq:dm-reference} subtracts (Ref.~\cite{McKinneyCode2026}). This is consistency at
the stored slices, not verification at all times.
```
NEW (FINAL)
```latex
The reference collapse uses step-start covariance minus a level-following spectral projector continued from the preceding accepted fields to each trial field, called the fixed reference projector below. On a locally fixed, isolated spectral branch, \(E_{\rm cont}[u]=\Tr(P_{\rm cont}[u]h[u])\) is locally degree one and \(DE_{\rm cont}[u](\delta u)=\Tr(P_{\rm cont}[u]\delta h)\). Under the other regularity, boundary and adjoint-averaging hypotheses of Appendix~\ref{app:charge}, the associated differentiable constrained Hartree flow conserves the boundary charge. Ordering of the followed levels below the rest is unnecessary. A reference-selection switch is not covered by this differential identity and must be distinguished from held-step and field-solve errors.

The stored float64 spectra report ordering inversions on 49 channel--slice pairs in channels 1--4, within \(T/r_s=12.75\)--14.5 and, for channel 1, 20.875--21.0. The smallest stored followed--unfollowed energy difference in absolute value is \(1.8662\times10^{-5}\). This is a stored-data minimum, not an established pathwise gap. The spectra alone establish neither identification with the prepared band nor indefiniteness of the Hessian restricted to clock and flow variations. At every stored slice where the followed index set changes, the continuation from the preceding slice's reference is a single maximum-overlap assignment, without sub-stepping, block rotation or intermediate projector, and the smallest squared projection of a followed eigenvector onto that reference is 0.903, the largest of an unfollowed one 0.097 (channel 2, \(T/r_s=13.375\)): the selection margin is at least 0.806 at the stored slices and unmeasured along the path between them.

On an identified smooth massive prepared-band branch on a common realization, the bulk expansion agrees with Eq.~\eqref{eq:dm-reference} through second gradient order. The first omitted bulk terms have fourth gradient order, including the nonpositive clock polarization about homogeneous massive rest where \(W_2[N,0,0]=0\). This statement is at fixed positive mass and does not establish the finite-chain comparison. For that comparison the independently derived chain second-gradient polarization is the bulk functional \(W_{2,\rm ch}=-\tfrac18\int dr\,[A\,v'^2+2B\,vv'+C\,v^2]/N\) per two-component channel, with \(A\), \(B\), \(C\) zone integrals of the frozen symbol at the local angular mass \(k/r\) (Ref.~\cite{McKinneyCode2026}); source and Hessian sums carry the copy weights \(16k\). The angular-gradient parameter obeys \(|\partial_r(k/r)|/(m^2+k^2/r^2)\le1/(2mr)\le0.05\) at \(r\ge r_{\min}=10/m=50a\) for \(ma=0.2\), against 0.96 at the first site. The bulk functional is not a prescription at the open end, where the missing bond polarizes even a uniform flow. On the stored slices at \(T/r_s=11\), 15 and 40, the maximum absolute discrepancy on \(r_s\le r\le9r_s\) between the followed-band reference's source densities \((\rho,J_r)\) and those of Eq.~\eqref{eq:dm-reference} with this functional, prepared-sea moments and release calibration included, is at most \(3.2\times10^{-9}\) in units \(a^{-4}\).
 The Hessian remainder after subtracting the second-gradient response has window-restricted norm \(0.225\)--\(0.232\) and \(0.0153\)--\(0.0156\), against a second-gradient Hessian of norm \(1.66\)--\(1.72\) and \(0.417\)--\(0.429\) on the two windows.
 Both norms are in units \(a^{-1}\), using the Euclidean operator norm of the finite-site energy Hessians in dimensionless applied clock and flow fields, restricted to the tested probe spaces. The probes have \(\sin^2\) windows supported in \([r_s,5r_s]\) and \([r_s,9r_s]\) at the tested wavevectors; the core \(r<r_{\min}\) is read on the followed band alone, against no bulk law. These are stored-slice comparisons outside the core, not verification at all times.
```

### 9. Sec. star, averaging kernel (item 1 and 3 binding)

OLD
```latex
Sources and fields use a Gaussian of width \(\sigma=4a\), whose
variance enters the leading averaging bounds
(Appendix~\ref{app:charge}).
```
NEW
```latex
Sources and fields use a Gaussian of width \(\sigma=4a\), whose variance enters the leading averaging bounds (Appendix~\ref{app:charge}) \todo{bind the numerical averaging map and its adjoint, with their centre, outer-boundary and Gaussian-tail errors, to the compactly supported kernel of Appendix~\ref{app:charge}}.
```


### 10. Sec. star, step and width controls (item 3)

OLD
```latex
Step controls repeat the reference collapse at \(\tau=4\) and 2.
```
NEW
```latex
Step controls repeat the reference collapse at \(\tau=4\) through \(T/r_s=14\) and at \(\tau=2\) \todo{bind the \(\tau=2\) archive at the reference collapse's settings, which the pinned inventory does not locate}.
```

OLD
```latex
All three are verified through formation of the trapping surface, \(\sigma/a=2\) through 40 (Ref.~\cite{McKinneyCode2026}).
```
NEW
```latex
All three are verified through formation of the trapping surface, \(\sigma/a=2\) through 40 \todo{bind the \(\sigma/a=2\) field archive of the from-release history plotted in Fig.~\ref{fig:star-collapse}(c), absent from the pinned inventory} (Ref.~\cite{McKinneyCode2026}).
```


### 11. Sec. star, formation ladders (item 3)

OLD
```latex
Runs near the maximum mass (\(E_0/E_{\max}=1.14\), \(E_{\max}\) the cold continuum maximum mass) or at small \(\zeta\) form no trapping radius within the computed time (ladder ranges and run values in Ref.~\cite{McKinneyCode2026}).
```
NEW
```latex
Runs near the maximum mass (\(E_0/E_{\max}=1.14\), \(E_{\max}\) the cold continuum maximum mass) form no trapping radius by \(T/r_s=40\), and the \(\zeta=0.05\) members end before \(T/r_s=0.5\) and decide nothing about formation (ladder ranges and run values in Ref.~\cite{McKinneyCode2026}).
```


### 12. Sec. star, trapped interior: insertion (item 2 and 3)

Insert immediately after
```latex
The largest central density on any stored slice is 3.2 percent of the exclusion density \(2a^{-3}\), and the dust fills to \(\pi/4\) on no slice (slice values in Ref.~\cite{McKinneyCode2026}).
```
NEW
```latex
At \(T/r_s=15\) the radial momentum of the dust inside \(r_s\), read by a windowed transform of the stored columns with the window's own norm as weight, has mean \(\cos(2ap_r)\) of 0.403 and carries 15.4 percent of its weight at \(\pi/4<a|p_r|<3\pi/4\): the balance's lattice weight \(\cos(2ap_i)\) of Sec.~\ref{sec:model-gravity} averages to less than half its continuum value over the interior dust, a reading of the weight and not of the source-conservation residual, so the interior is read on the lattice equations, as the hydrodynamic test already says. The dust's coherence with the release sea, \(\|SD+DS-2SDS\|_{\rm HS}^2/\Tr D\) for the release sea factor \(S\) and the dust covariance \(D\), is \(2.15\times10^{-4}\) on the same slice, and with the propagated sea in place of \(S\) the same expression is at most \(4.3\times10^{-20}\).
```


### 13. Sec. singularity: insertion (items 2 and 3)

Insert immediately before the paragraph beginning `Every sampled slice passes these tests`:
NEW (FINAL)
```latex
On a fixed finite realization with bounded fields and \(N\ge n_*>0\), the required derivatives of the massive second-gradient reference are bounded. For Eq.~\eqref{eq:dm-reference} the Hessian is \(\mathcal W''=W_2''\), the other two terms being linear, with \(D^2W_2[u](\delta u,\delta u)=-\tfrac12\int|\delta\mathsf F-\delta N\,\mathsf F/N|^2_{\mathsf T}\,N^{-1}d^3x\), \(|X|^2_{\mathsf T}=X_{ij}\mathsf T_{ij,kl}X_{kl}\le\tau_m|X|^2\) and \(\tau_m\le2/(3\sqrt3\,ma^3)\), negative semidefinite and finite wherever \(N\) is bounded below. For each locally fixed isolated followed set,
\[
D^2E_{\rm cont}[u](\delta u,\delta u)
=-2\sum_{i\in F,\ a\notin F}
 \frac{|\langle a|\delta h|i\rangle|^2}{e_a-e_i},
\qquad
|D^2E_{\rm cont}[u](\delta u,\delta u)|
\le \frac{2}{\delta_{\rm b}}\|\delta h\|_{\rm HS}^2,
\]
where \(\delta_{\rm b}=\min_{i\in F,a\notin F}|e_a-e_i|>0\). Channel responses carry their copy weights \(16k\). Ordering guarantees a negative-semidefinite response; mixed denominators alone do not establish indefiniteness on the allowed field variations. The stored spectral minimum is reported in Sec.~\ref{sec:results-lattice}; established lower bounds require spectral and assembly error bounds. The channel-1 calculation at \(T/r_s=15\) finds the followed eigenspace of the stored columns isolated by at least \(1.898\times10^{-2}\), with eigen-residual at most \(5.4\times10^{-15}\) and the sine of its angle to those columns at most \(1.5\times10^{-11}\), an a posteriori float64 residual check, not an interval bound.
```

### 14. Sec. singularity, the stored-slice tests (item 3)

OLD
```latex
Every sampled slice passes these tests, which checks the slices and not the trajectory between them. The reference collapse's stored clocks stay positive through \(T/r_s=40\) (sampled minimum 0.068), independent current-mass scans of 894 stored slices in four histories give a clock floor \(e^{-50}\), and the implemented constraint Jacobian has verified inverse norm at most 2.90 on 573 slices in three histories, all under the held-source operators of Sec.~\ref{sec:results-lattice} rather than Eq.~\eqref{eq:dm-reference}. Rays and packets traced through the stored fields, a probe and not a continuation statement, show no singular behaviour: all 36 tested ingoing rays cross the regular centre at finite affine length, so no focusing singularity forms there, with angular curvature at most \(0.14/a^2\), and test packets without back-reaction keep their norm to \(2\times10^{-11}\) over \(T/r_s=12.125\)--40 (two independent calculations, Ref.~\cite{McKinneyCode2026}).
```
NEW (FINAL)
```latex
Every sampled slice passes the clock tests, which checks the slices and not the trajectory between them. On all 321 stored slices of the reference collapse through \(T/r_s=40\) the back-averaged clock the matter sees is positive, with minimum 0.1020, and independent current-mass scans of 894 stored slices in four histories give a clock floor \(e^{-50}\). The constraint-Jacobian test is specified below, with the between-slice bound. Rays and packets traced through the stored fields, a probe and not a continuation statement, show no singular behaviour: all 36 tested ingoing rays cross the regular centre at finite affine length, so the traced rays meet no focusing singularity there, with angular curvature at most \(0.14/a^2\), and test packets without back-reaction keep their norm to \(2\times10^{-11}\) over \(T/r_s=12.125\)--40 (two independent calculations, Ref.~\cite{McKinneyCode2026}).
```

### 15. Sec. singularity, the between-slice bound (item 3)

OLD
```latex
The bound runs over every consecutive slice interval under Eq.~\eqref{eq:dm-reference} with the full reference response, and verifies throughout each interval a regular inward-flow solution with margin \(\mathcal M/r^3\ge\mu_*>0\), Eq.~\eqref{eq:sg-current-mass}, a uniform inverse bound on the full \(F_u\), and a self-closing bound on the fields and reference derivatives (full statement, centre handling, and seeding rule in Ref.~\cite{McKinneyCode2026}). Failure of a sufficient inequality means not verified, not singular. The stored histories pass the slice tests above, and the between-slice bound is what the continuation calculation must certify.
```
NEW (FINAL)
```latex
The bound runs over every consecutive slice interval under the computed law, the followed band's energy as its reference with its full response, and verifies throughout each interval a regular inward-flow solution with margin \(\mathcal M/r^3\ge\mu_*>0\), Eq.~\eqref{eq:sg-current-mass}, a uniform inverse bound on the full \(F_u\), and a self-closing bound on the fields and reference derivatives (full statement, centre handling, and seeding rule in Ref.~\cite{McKinneyCode2026}). The reference-derivative bound through the isolation \(\delta_{\rm b}\) applies on a single continuation branch. The isolation margin must remain positive throughout the established field neighbourhood and trajectory tube, and validated selection margins must keep the actual continuation rule, at every sub-step it takes, on that branch. The quadratic response bound alone is not a bound on the full constraint Jacobian. A followed--unfollowed crossing prevents this gap-based bound, not necessarily continuation itself. A change of selected projector requires separate jump accounting. Failure of a sufficient inequality means not verified, not singular. The full-Jacobian test can use the self-consistency map \(G(x)\) on the applied site fields \(x\), the back-averaged fields the rows return from the state's sources with the reference evaluated at \(x\): within one differentiable realization, provided the held-source row Jacobian is invertible and \(G\) solves those same rows, \(F_u\) is invertible exactly when \(J=I-D_xG\) is. For a finite realization an approximate inverse \(Y\), with a validated residual \(q=\|I-YJ\|_\infty<1\), establishes \(\|J^{-1}\|_\infty\le\|Y\|_\infty/(1-q)\). The same construction for Eq.~\eqref{eq:dm-reference} uses \(W_2''\) as the reference response on its declared realization. No stored slice carries this bound. The stored histories pass the clock tests above, and the between-slice bound is what the continuation calculation must establish.
```

### 16. Sec. page-transfer, the one cubic column (item 2)

OLD
```latex
The one column read in both descriptions differs by up to 0.34 at later sweeps, so the lift of the chain description to the cubic operator is a structural input of the chain computations, that column being its only test.
```
NEW
```latex
The one column read in both descriptions differs by up to 0.34 at later sweeps, a comparison of the two dynamics and the chains' only cubic contact; no cubic state is reconstructed from the chains.
```


### 17. Discussion, first paragraph

OLD
```latex
and the quantum test evolution has a positive clock and an invertible constraint Jacobian at every stored time slice.
```
NEW (FINAL)
```latex
and the quantum test evolution keeps a positive clock at every stored time slice.
```

### 18. Discussion, singularity paragraph

OLD
```latex
The computed collapse meets these conditions on every stored slice, and its rays cross the regular centre, so no focusing singularity forms within the computed time (Sec.~\ref{sec:star}).
```
NEW (FINAL)
```latex
On every stored slice the computed collapse keeps a positive clock, and its traced rays cross the regular centre, meeting no focusing singularity within the computed time (Sec.~\ref{sec:star}); the constraint Jacobian with the reference's response, and the between-slice bound, remain to be established.
```

### 19. Outside items 1–3, both-signed direction (Sec. page-effective)

OLD
```latex
That region can hold it:
```
NEW
```latex
That region can hold two thirds of it:
```

---


