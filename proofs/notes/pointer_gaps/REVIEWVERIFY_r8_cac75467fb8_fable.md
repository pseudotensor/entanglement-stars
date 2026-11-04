# Verification of review round 8 (manuscript commit cac75467fb8)

## Verdict table

| # | Item (duplicates merged) | File / section | Claim in a few words | Verdict (class) | One-line reason |
|---|---|---|---|---|---|
| 1 | E1 = Edit 1 = census row "Outlook fourth order" | 08_discussion.tex, Outlook, fourth-gradient paragraph | $\lambda a^2\lvert B\rvert^2/(16\pi GN)$ is "the only isotropic fourth-gradient flow term", one free coefficient, one transverse response fixes it | VALID-APPLY-MODIFIED (FIX) | $C_{ij}C_{ji}/N$, $C_{ij}=\partial_i(\nabla\times v)_j$, meets all four requirements and is independent of $\lvert B\rvert^2/N$ modulo divergences at variable clock (torus integral $-4\epsilon\pi^3$); the uniform-clock momentum response is blind to it. Reviewer's text correct; apply condensed. |
| 2 | E2 = Edit 2 = census row "E2 simplification" | 06_scaling.tex, sec:page-effective, "Third, the energy balance" | balance "equal to $-L$ at a trapping radius clear of heavy matter with negligible defect" | VALID-APPLY-MODIFIED (FIX) | The factor is $\beta_R=1-2G\mathcal M'_R$; an ingoing null light flux alone gives $\beta_R=1/(1-GL/N_R)\neq1$; unit factor holds only at first order in $GL/N_R$. Apply condensed. |
| 3 | E3 = Edit 3 = census row "Microcanonical Boltzmann claim" | 08_discussion.tex, Outlook, microcanonical test | ratio is Hawking's "exactly when $\log\mathcal N$ has derivative $8\pi GE$" | VALID-APPLY-MODIFIED (FIX) | Exact ratio is the finite difference $e^{S(E-\omega,Q-q)-S(E,Q)}$; the derivative fixes it to first order only; $S=4\pi GE^2$ adds $4\pi G\omega^2$ to the exponent. If this sentence is the previous round's "exactly when" fix, that fix is incomplete. |
| 4 | Edit 4 (DERIVATION) | A2_local_area.tex (app:local), new paragraph; cited in sec:results-vacuum | massive cross-block bound $\Rightarrow$ entropy error $O(L^{3/2}\sqrt{\log L})$, with $d_*$ from the direct test | VALID-APPLY-MODIFIED (FIX, derivation) | Every step rederived (strip analyticity, kernel bound, $T_d$, Rotfel'd, Mirsky, concavity bound, $d_*$); correct. Print as a sketch per owner rule B4, full constants to the held supplement. |
| 5 | Edit 5 | 04_results.tex, eq:page-first-law-rows | "its exact primitive is $\delta(A/4G)$" | VALID-APPLY-MODIFIED (CLARIFICATION) | The exact one-form is $(\delta E-W_{\rm ext})/(\kappa_R/2\pi)=\delta(A/4G)$; apply inline and keep the integrability reference. |
| 6 | Edit 6 | 05_star.tex, formation paragraph | $f_{\rm ext}$ called "the net signed charge" | VALID-APPLY-MODIFIED (CLARIFICATION) | $f_{\rm ext}=[E-\mathcal M(R_t)]/E$ with $\mathcal M(R_t)=R_t/2G$; dimensionless fraction; apply inline. |
| 7 | Edit 7 = review Sec. C/E (proofs delegated) | A2_local_area.tex A.1, A.2, A.4, A.5 | nonstandard proofs and the rigorous $\eta$ interval unavailable in the text | OPEN | Correct as a proof-availability request; owner rule B4 puts full proofs in the held supplement; lemmas (i)–(iv) and the [111] input data listed below. No manuscript text change. |
| 8 | Edit 8 | 05_star.tex, sec:singularity | between-slice continuation uncertified | OPEN | Manuscript already says so; the proposed Neumann/Picard enclosure is sound; no text change. |
| 9 | Edit 9 | 06_scaling.tex, sec:page-effective | supply, depletion, release unproved; $K\gtrsim0.593\sqrt{R_0/a}$ | OPEN (no text change) | Arithmetic checks and restates the manuscript's own channel obstruction; no edit proposed. |
| 10 | Edit 10 | 08_discussion.tex, Outlook | TT tidal projection of $(\delta N,\delta v)$ vanishes | NON-ISSUE | Reviewer agrees; linearized $R_{0i0j}$ verified; text already correct. |
| 11 | Edit 11 | 08_discussion.tex, Outlook | accessible count depends on the expansion centre | OPEN (no text change) | Manuscript already states it; nothing beyond item 3 to change. |
| 12 | Unverified numerics and repository values (review Secs. A, E) | various | numbers, intervals, two-solver agreement unverified by the reviewer | NON-ISSUE for the text | No error alleged; the reproducibility manifest is part of item 7. |

No item is ALREADY-FIXED; none of the eight previous-round corrections is re-reported. No proposed edit is a HEDGE.

---

## Item 1: E1 / Edit 1 (fourth-gradient uniqueness)

**1. Passage** (Outlook, sec:discussion): "at the next order the four requirements leave one coefficient free: with $B=\nabla\times\nabla\times v$, the density $\mathcal A+\lambda a^2|B|^2/(16\pi GN)$ meets all four for every real $\lambda$, is the only isotropic fourth-gradient flow term they admit modulo divergences, and on a transverse flow of wavevector $q$ its momentum law is $(q^2/16\pi G)(1+2\lambda a^2q^2)v_T$ … Only a fifth input, the law's own transverse response at order $(aq)^2$, fixes it." Quoted correctly by the reviewer.

**2. Mathematics.** Write $\varpi=\nabla\times v$, $C_{ij}=\partial_i\varpi_j$, $T_{ijk}=\partial_i\partial_jv_k$. Then $C_{ij}=2\epsilon_{jab}\partial_aS_{ib}$ with $S$ the symmetric strain (from $\partial_i\partial_jv_k=\partial_iS_{jk}+\partial_jS_{ik}-\partial_kS_{ij}$, which I checked term by term), $C_{ii}=\nabla\cdot\varpi=0$, and $|B|^2=|\nabla\times\varpi|^2=C_{ij}C_{ij}-C_{ij}C_{ji}$. So there are exactly two isotropic quadratic forms in $C$: $C_{ij}C_{ij}$ and $C_{ij}C_{ji}$, hence two densities $\lvert B\rvert^2/N$ and $C_{ij}C_{ji}/N$.

Requirements for $\mathcal A_4=a^2[\lambda_1|B|^2+\lambda_2C_{ij}C_{ji}]/(16\pi GN)$: degree one in $(N,v)$ ✓ (so the response identity follows from Euler's identity as for $\mathcal A$); local, cubic (indeed $O(3)$) invariant, even under inversion ($\varpi$ pseudovector, $C\to-C$, $CC$ even) and time reversal ($v\to-v$, $CC$ even) ✓; invariant under uniform flow shifts and the rigid turn ($\varpi\to\varpi+2\omega$, $C$ unchanged) ✓; zero at uniform fields ✓. Requirement 3: for a density $Q(\partial_kS_{ij})/N$, two integrations by parts give $\delta\mathcal A_4/\delta v_j=\partial_i\sigma_{ij}$ with $\sigma_{ij}=\partial_k[N^{-1}\partial Q/\partial(\partial_kS_{ij})]$ symmetric, so $\epsilon_{abc}x_bJ_c=\partial_i(\epsilon_{abc}x_b\sigma_{ic})$ is a flux ✓ (valid at every clock). Requirement 4: on $v=\nabla\phi$, $C=0$, and since $\mathcal A_4$ is quadratic in $C$ both $\delta\mathcal A_4/\delta N=-a^2[\ldots]/(16\pi GN^2)$ and $\delta\mathcal A_4/\delta v$ vanish identically, so $\rho_\ell+w\cdot J_\ell$ receives nothing on potential flows at every positive clock ✓.

Independence: at constant $N$, $C_{ij}C_{ji}=\partial_i(\varpi_j\partial_j\varpi_i)$ (using $\partial_i\varpi_i=0$), a divergence; at variable $N$ it is not. For $v=(0,0,\cos x+\cos y)$: $\varpi=(-\sin y,\sin x,0)$, $C_{xy}=\cos x$, $C_{yx}=-\cos y$, $C_{ij}C_{ji}=-2\cos x\cos y$, $B=(0,0,\cos x+\cos y)$. With $1/N=1+\epsilon\cos x\cos y$ on $\mathbb T^3$: $\int C_{ij}C_{ji}/N=-2\epsilon\cdot2\pi\cdot\pi\cdot\pi=-4\epsilon\pi^3$, $\int|B|^2/N=8\pi^3+4\epsilon\pi^3$. If $\int CC/N=c\int|B|^2/N$ for all configurations, $\epsilon=0$ forces $c=0$ and then $-4\epsilon\pi^3=0$, contradiction. The two densities are independent modulo divergences on the declared variable-clock domain. The energy law at unit clock already distinguishes them ($-\lambda_2a^2C_{ij}C_{ji}/16\pi G$ is nonzero pointwise in this example), while the momentum law at uniform clock does not see $\lambda_2$ at all (variation of a divergence). The manuscript's "one coefficient free", "only … term", and "only a fifth input … fixes it" are all false; the displayed transverse law $(q^2/16\pi G)(1+2\lambda a^2q^2)v_T$ is correct for $\lambda_1$ (I recomputed: $\nabla\cdot K=-\tfrac12q^2v_T$, $\nabla\times\nabla\times B=q^4v_T$). This error was not in the previous round's list; it is a new finding.

**Script.** verify_03.py checks the local invariant, the vanishing constant-clock integral, the value $-4\epsilon\pi^3$, the $8\pi^3$ of the $B$ term, and $\nabla\times\nabla\phi=0$. It tests the independence counterexample; it does not test the four requirements (checked by hand above). PASS is consistent with my derivation.

**3. Real problem**: yes, a false uniqueness statement, consequential for the Outlook's "two coefficients give two laws / fifth input" conclusion, not for any spherical result (both terms vanish with their variations on potential flows).

**4. Replacement correctness/compatibility.** The reviewer's block is mathematically correct, including the torus numbers and the $\lambda_1$-only statement. Other sites: Sec. model-gravity ("next terms … undetermined coefficients and no bound") and Sec. results-exterior ("higher gradient order, where … corrections are undetermined") are already compatible; nothing else carries the uniqueness claim. One refinement: the $\lambda_2$ term does not vanish on the exterior's frame-dragging field ($\varpi$ is a gradient there, so $B=0$ but $C_{ij}$ is symmetric and nonzero); at unit clock its momentum law vanishes identically and its energy law is second order in the perturbation, so "Kerr's to first order" is unaffected.

**5. Scope**: the correction replaces a false claim with a true weaker one (at least two coefficients, no complete classification claimed); no scope label dropped. **6.** No bridge invented; the full fourth-order classification stays open (item O1 below).

**Recommended replacement** (whole paragraph, condensed from the reviewer's, page cost about five lines):
```latex
The second is the order of the derivation. The laws hold at leading gradient order, and at the next order the four requirements leave coefficients free, at least two: with \(\varpi=\nabla\times v\), \(C_{ij}=\partial_i\varpi_j\) and \(B=\nabla\times\varpi\), the density \(\mathcal A+a^2[\lambda_1|B|^2+\lambda_2C_{ij}C_{ji}]/(16\pi GN)\) meets all four for every real \(\lambda_1,\lambda_2\). Each term is local, isotropic, degree one, invariant under uniform shifts and rigid turns of the flow, and quadratic in second derivatives of \(v\), so its momentum law is the divergence of a symmetric local tensor and the moment has its flux; on \(v=\nabla\phi\) both vanish with their first variations, so the potential-flow charge requirement holds at every positive clock. The two are independent modulo divergences only because the clock varies: at constant clock \(C_{ij}C_{ji}=\partial_i(\varpi_j\partial_j\varpi_i)\) is a divergence, while for \(v=(0,0,\cos x+\cos y)\) and \(1/N=1+\epsilon\cos x\cos y\) on a \(2\pi\)-periodic box \(\int C_{ij}C_{ji}/N\,d^3x=-4\epsilon\pi^3\neq0\). On a transverse flow of wavevector \(q\) at unit clock the momentum law is \((q^2/16\pi G)(1+2\lambda_1a^2q^2)v_T\), the \(\lambda_2\) term dropping out, so the law's own transverse response at order \((aq)^2\) fixes \(\lambda_1\) alone; \(\lambda_2\) needs the energy law on superposed transverse flows or the momentum law at nonuniform clock. Nothing declared selects either. Both terms vanish on every radial or potential flow, \(B\) also on the exterior's frame-dragging field and the \(\lambda_2\) term there at first order in the perturbation, so no result here depends on them, and elsewhere their effect is of relative order \((aq)^2\), below the averaging control.
```

---

## Item 2: E2 / Edit 2 (moving-root factor)

**1. Passage** (sec:page-effective): "At the trapping radius it is an inward flux of negative null energy, $4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R=-(2R\kappa_R/N_R)L$ with $\mathcal D_R$ the source defect there, equal to $-L$ at a trapping radius clear of heavy matter with negligible defect." Quoted correctly.

**2. Mathematics.** Appendix app:charge gives $(2R\kappa_R/N_R)\dot E_R=4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R$ and $2R\kappa_R/N_R=1-2G\mathcal M'_R$. From Eq. dr-charge at the root ($w=-1$), $\mathcal M'_R=4\pi R^2(\rho_\ell-J_{\ell r})_R$, consistent with $\kappa_R=(N_R/2R)[1-8\pi GR^2(\rho_R-J_R)]$ in sec:results-exterior (verify_02 "root slope": $N'_R=4\pi GRN_RJ$ from the momentum row, $v'_R=N_R(1-8\pi GR^2\rho)/2R$ from the energy row; PASS and I rederived both). The factor is one iff $\mathcal M'_R=0$, i.e. iff the full source combination $\rho_\ell-J_{\ell r}$ vanishes at $R$, which a light-only null flux violates: for ingoing null stress $p_r=\rho_\ell$, $J_{\ell r}=-\rho_\ell$ (negative $\rho_\ell$ for the negative-energy inflow), $\rho_\ell+p_r-2J_{\ell r}=4\rho_\ell$, so with negligible defect $16\pi R^2N_R\rho_\ell=-\beta_RL$ and $\beta_R=1-16\pi GR^2\rho_\ell=1+\beta_RGL/N_R$, hence $\beta_R=1/(1-GL/N_R)$. The unit factor holds to first order in $GL/N_R$ (of order $g\chi^2\mathcal L$, about $10^{-5}$ at the computed size), not exactly. The reviewer is right; "clear of heavy matter with negligible defect" is not a source-free shell.

**Script.** verify_02 "light-only root factor is not one" merely checks that $(1+16\pi GR^2\rho)-1=16\pi GR^2\rho$, a tautology; it does not test the claim. The "root slope" and "radial charge" checks do support the $\beta_R$ formula.

**3.** Real error, local; the general balance, Appendix app:charge, and the two-window identity (which cancels $\beta_R$ by Leibniz's rule) retain the factor correctly. The Page crossing assumes the depletion law and is unaffected.

**4. Replacement.** Correct, including $\beta_R=1/(1-GL/N_R)$. Compatible: $\beta_R=1-2G\mathcal M'_R$ is already defined in sec:star and used in sec:scaling-far ($P_{\rm ext}=N_R\beta_RL\,du/dT$). Time normalization: the display writes $L$ without the $du/dT$ factor; the replacement should keep "same normalization".

**5.** Factual correction of an exact claim to its correct first-order form: FIX, not a hedge. **6.** None needed.

**Recommended replacement** (for the quoted sentence):
```latex
At the trapping radius it is an inward flux of negative null energy, \(4\pi R^2N_R(\rho_\ell+p_r-2J_{\ell r})_R+\mathcal D_R=-\beta_RL\) with \(\mathcal D_R\) the source defect there and \(\beta_R=2R\kappa_R/N_R=1-2G\mathcal M'_R=1-8\pi GR^2(\rho_\ell-J_{\ell r})_R\) the moving-root factor of Sec.~\ref{sec:star}, equal to one exactly when \(\mathcal M'_R=0\). Clearing the root of heavy matter does not remove the light flux's own part of \(\mathcal M'_R\): for an ingoing null light flux with negligible defect, \(p_r=\rho_\ell\) and \(J_{\ell r}=-\rho_\ell\), the debit \(dE_R/du=-L\) in the same time normalization gives \(\beta_R=1/(1-GL/N_R)\), so the balance reads \(-L\) at first order in \(GL/N_R\) and keeps \(\beta_R\) exactly.
```

---

## Item 3: E3 / Edit 3 (microcanonical ratio)

**1. Passage** (Outlook): "… obey $\mathcal N_a\mathsf W_{a\to b}=\mathcal N_b\mathsf W_{b\to a}$, so their ratio is $\mathcal N_b/\mathcal N_a$, and it is Hawking's ratio $e^{-8\pi GE(\omega-\mu q)}$, chemical potential $\mu$, exactly when $\log\mathcal N$ has derivative $8\pi GE$ in $E$ and the chemical-potential term in $Q$." Quoted correctly.

**2. Mathematics.** $\mathcal N_b/\mathcal N_a=\exp[S_{\rm mc}(E-\omega,Q-q)-S_{\rm mc}(E,Q)]$, $S_{\rm mc}=\log\mathcal N$, a finite difference. With $\partial_ES=\beta$, $\partial_QS=-\beta\mu$: $S(E-\omega,Q-q)-S(E,Q)=-\int_0^1\beta(E-s\omega,Q-sq)[\omega-\mu(E-s\omega,Q-sq)q]\,ds$ (fundamental theorem along the segment; I checked the integrand sign). The derivative condition at the initial shell fixes only the first-order term $-8\pi GE(\omega-\mu q)$. Counterexample: $S=4\pi GE^2$ (the area entropy, since $A/4G=4\pi(2GE)^2/4G$), $q=0$: $S(E-\omega)-S(E)=-8\pi GE\omega+4\pi G\omega^2$, so the ratio is $e^{-8\pi GE\omega+4\pi G\omega^2}$, not Hawking's. Neither direction of "exactly when" holds. The reviewer is right. The reciprocity $\mathcal N_a\mathsf W_{a\to b}=\sum_{\alpha\in a,\beta\in b}|\langle\beta|V|\alpha\rangle|^2=\mathcal N_b\mathsf W_{b\to a}$ needs equal weights in each shell and $|V_{\beta\alpha}|=|V_{\alpha\beta}|$ (Hermitian $V$), which the manuscript's "exact form" implicitly assumes; stating it is a clarification, not a hedge.

**Script.** verify_05 "finite microcanonical ratio" checks the quadratic expansion; trivial algebra, but it does test the counterexample.

**3.** Real error in the Outlook's test criterion. The first-law integration and the Hawking factor elsewhere do not use it. Provenance: the previous round installed an "exactly when" criterion; if it is this sentence, that fix replaced one imprecision with another (finite transition versus derivative) and is incomplete; if it is one of the other "exactly when" sentences (sources conserved; $f=1$; remainder within $\epsilon L$; $K_{XY}\ne0$), those are all genuine biconditionals and are unaffected. Either way the current sentence must change.

**4. Replacement.** Correct throughout. Length: three displays in the Outlook is more than the paper's style and budget need; the content fits in two sentences.

**5.** Correction of a false exactness claim: FIX. **6.** The open state-count problem stays open as the manuscript already says.

**Recommended replacement** (for "so their ratio is … in $Q$."):
```latex
so their ratio is \(\mathcal N_b/\mathcal N_a=\exp[S_{\rm mc}(E-\omega,Q-q)-S_{\rm mc}(E,Q)]\), \(S_{\rm mc}=\log\mathcal N\), a finite entropy difference, for equally weighted shells and a Hermitian transition operator. It is Hawking's ratio \(e^{-8\pi GE(\omega-\mu q)}\), chemical potential \(\mu\), to first order in \(\omega\) and \(q\) exactly when \(S_{\rm mc}\) has derivative \(8\pi GE\) in \(E\) and \(-8\pi GE\mu\) in \(Q\) at the initial shell; the finite difference adds the higher terms, \(4\pi G\omega^2\) to the exponent for \(S_{\rm mc}=4\pi GE^2\) and \(q=0\).
```
Also change "The test has an exact form:" to "The test has an exact form, finite-shell reciprocity:".

---

## Item 4: Edit 4 (massive direct-state transfer estimate)

**1. Passage** (sec:results-vacuum): "at fixed $ma>0$, for a ball of radius $La$, a cross-block bound $\|\Pi_A(C-P_0)\Pi_{A^c}\|_2^2=O(L)$ gives entropy difference $O(L^{3/2}\sqrt{\log L})$. The sufficient direct-state test is $E_\varphi(C)=\|\Omega_0^{1/2}(\varphi C-P_0\varphi)\|_2^2\le\kappa_fL$ … It gives that cross-block condition through the massive gap (details in Ref.~\cite{McKinneyCode2026})." Quoted correctly (with ellipsis).

**2. Mathematics, step by step.**
- Strip: $\sin(x+iy)=\sin x\cosh y+i\cos x\sinh y$, $\operatorname{Re}\sin^2(x+iy)=\sin^2x(1+2\sinh^2y)-\sinh^2y\ge-\sinh^2y$, so $\operatorname{Re}\varepsilon^2\ge\zeta^2-3\sinh^2b=\zeta^2/2$ for $\sinh b=\zeta/\sqrt6$ ✓. Hence $\sqrt{\varepsilon^2}$ analytic and $|\varepsilon|\ge\sqrt{\operatorname{Re}\varepsilon^2}\ge\zeta/\sqrt2$; $|\sin(x+iy)|\le\cosh y$ gives $\|\hat h_0\|\le\zeta+3\cosh b$; contour shift by $ib\,\operatorname{sgn}(x_i-y_i)$ in each $k_i$ gives $\|(P_0)_{xy}\|\le\tfrac12[1+\sqrt2(\zeta+3\cosh b)/\zeta]e^{-b|x-y|_1}=K_\zeta e^{-b|x-y|_1}$ ✓.
- $T_d$: for $x\in I$ every $y\in A^c$ has $|x-y|_1\ge|x-y|_2>d$; $\|(P_0)_{xy}\|_2^2\le4K_\zeta^2e^{-2b|x-y|_1}\le4K_\zeta^2e^{-bd}e^{-b|x-y|_1}$; $\sum_{z\in\mathbb Z^3}e^{-b|z|_1}=\coth^3(b/2)$; total $\le4|I|K_\zeta^2e^{-bd}\coth^3(b/2)\le T_d$ ✓.
- $F(X)=S_A(C)$: $C_A-C_A^2=XX^\dagger$ for a projector, eigenvalues $\nu=(1\pm\sqrt{1-4\sigma^2})/2$, $s$ symmetric under $\nu\to1-\nu$, $f(0)=s(1)=0$ ✓. $f'(t)=2\operatorname{artanh}(z)/z$, $z=\sqrt{1-4t}$, positive and decreasing: increasing, concave ✓; $g(\sigma)=f(\sigma^2)$ has $g'=4\sigma\operatorname{artanh}(z)/z\in[0,2]$ ✓ (the manuscript's own derivative).
- Rotfel'd ($\operatorname{Tr}f(P+Q)\le\operatorname{Tr}f(P)+\operatorname{Tr}f(Q)$ for concave $f\ge0$, $f(0)=0$) plus trace monotonicity for increasing $f$ give $0\le F(X)-F(X_B)\le F(X_I)$ ✓ ($f$ extended linearly past $1/4$ with slope 2 to meet the hypothesis on $[0,\infty)$).
- Mirsky: $\sum_j|\sigma_j-\sigma_j'|^2\le\|Y-Y'\|_2^2$, at most $m_B$ singular values for row-restricted blocks, Cauchy–Schwarz: $|F(X_B)-F(X_{0B})|\le2\sqrt{m_B}\,\|X-X_0\|_2\le2\sqrt{m_Bd_*L}$ ✓.
- $f(t)\le2t\log(e/t)$: with $\mu=1-\nu\in[0,\tfrac12]$, $s\le\mu\log(e/\mu)$ (from $-(1-\mu)\log(1-\mu)\le\mu$), $t\le\mu\le2t$, and $\mu\log(e/\mu)$ increasing ✓; Jensen for the concave $2t\log(e/t)$ over $M$ singular values gives $F(Y)\le g_M(\|Y\|_2^2)$, $g_M$ increasing on $t<M$ ✓. $\|X_I\|_2^2\le2d_*L+2T_d$, $\|X_{0I}\|_2^2\le T_d$ ✓. Boxed estimate ✓.
- $M=4|A|\le4(2L+1)^3\le108L^3$; $d=2b^{-1}\log L$ gives $T_d=O(L)$; $m_B\le4\cdot8\cdot4\pi L^2(1+d)=O(L^2\log L)$; total $O(L^{3/2}\sqrt{1+\log L})$, matching the manuscript's exponent ✓.
- $d_*$: $\Omega_0\ge\zeta$ gives $\|D\|_2^2\le E_\varphi/\zeta$ ✓; $\Pi_A\varphi=\Pi_A$ gives $X-X_0=\Pi_AD\Pi_{A^c}-\Pi_AP_0(1-\varphi)\Pi_{A^c}$ ✓ (expanded both sides); $1-\varphi(y)=\varphi(x)-\varphi(y)\le C_\varphi|x-y|/L$ and the kernel bound give $\|\Pi_AP_0(1-\varphi)\|_2^2\le MK_\zeta^2C_\varphi^2\Lambda_b/L^2\le108K_\zeta^2C_\varphi^2\Lambda_bL$ with $\Lambda_b=3(\tfrac{1+r}{1-r})^2\cdot\tfrac{2r(1+r)}{(1-r)^3}=\tfrac{6r(1+r)^3}{(1-r)^5}$, $r=e^{-2b}$ ✓; $d_*=2\kappa_f/\zeta+216K_\zeta^2C_\varphi^2\Lambda_b$ ✓. The declared $2L$-to-$3L$ window has $\varphi=1$ on $A$ and Lipschitz constant $O(1/L)$ ✓. Only the gap of $h_0$ is used ✓.

**Script.** verify_06 "massive direct cross-block bound" tests the decomposition inequality $\|X-X_0\|_2^2\le2E_\varphi/\lambda_{\min}+2\|\Pi_AP_0(1-\varphi)\Pi_{A^c}\|_2^2$ on one random 14-mode instance; the "window identity" test checks the manuscript's pure-state identity (I rederived it: $E_\varphi(C)-E_\varphi(P)=\operatorname{Tr}[(C-P)\varphi h\varphi]$ needs $C^2=C$). Both test the stated steps on one instance; the proof is the hand derivation above.

**3.** It fills a real gap (the manuscript delegates the proof). **4.** Correct and compatible with the notation of sec:results-vacuum and app:local ($\varkappa$-regularity, $A_{\rm lat}$, $s$, $P_0$). **5.** Adds no qualifier; it is a derivation. **6.** It does not supply the history-uniform budget $E_\varphi(C)\le\kappa_fL$ (a measured/assumed input) or the massless $L^{7/4}$ theorem; the manuscript already labels both.

**Recommendation.** Apply as a sketch in app:local (owner rule B4; the reviewer's full block would add about two thirds of a page), change "(details in Ref.~\cite{McKinneyCode2026})" in sec:results-vacuum to "(sketch in Appendix~\ref{app:local}, constants in Ref.~\cite{McKinneyCode2026})", and put the reviewer's full-constant version in the held supplement:
```latex
\paragraph{A.6 The massive direct-state estimate.}
Shown: for \(\zeta>0\), a projector \(C\), the ball \(A\) of radius \(L\ge1\), \(X=\Pi_AC\Pi_{A^c}\) and \(X_0=\Pi_AP_0\Pi_{A^c}\) with \(\|X-X_0\|_2^2\le d_*L\),
\(|S_A(C)-S_A(P_0)|\le2\sqrt{m_Bd_*L}+g_M(t_C)+g_M(t_0)\), where \(m_B\le4\varkappa A_{\rm lat}(1+d)\) counts the modes within depth \(d\) of \(A^c\), \(M=4|A|\le108L^3\), \(g_M(t)=2t\log(eM/t)\), \(t_C=\min\{2d_*L+2T_d,M/4\}\), \(t_0=\min\{T_d,M/4\}\), \(T_d=K_\zeta^2Me^{-bd}\coth^3(b/2)\), \(b=\operatorname{arsinh}(\zeta/\sqrt6)\) and \(K_\zeta=\tfrac12[1+\sqrt2(\zeta+3\cosh b)/\zeta]\); at \(d=2b^{-1}\log L\) the right side is \(O(L^{3/2}\sqrt{1+\log L})\).
Sketch: in the strip \(|\operatorname{Im}k_i|\le b\), \(\operatorname{Re}\varepsilon^2\ge\zeta^2/2\), so the symbol of \(P_0\) is analytic there and contour shifting gives \(\|(P_0)_{xy}\|\le K_\zeta e^{-b|x-y|_1}\), hence \(\|\Pi_IX_0\|_2^2\le T_d\) for the interior \(I=A\setminus B\) of the boundary layer \(B\). With \(f(t)=s((1+\sqrt{1-4t})/2)\), increasing and concave with \(f(0)=0\), \(S_A(C)=\Tr f(X^\dagger X)\); Rotfel'd's inequality on \(X^\dagger X=X_B^\dagger X_B+X_I^\dagger X_I\) bounds the interior rows by \(\Tr f(X_I^\dagger X_I)\le g_M(\|X_I\|_2^2)\), from \(f(t)\le2t\log(e/t)\) and concavity, and Mirsky's inequality with the \(2\)-Lipschitz singular-value function \(f(\sigma^2)\) bounds the boundary rows by \(2\sqrt{m_B}\,\|X-X_0\|_2\). The direct test supplies \(d_*\): for a window with \(\varphi=1\) on \(A\) and Lipschitz constant \(C_\varphi/L\), as the \(2L\)-to-\(3L\) window of Sec.~\ref{sec:page}, \(D=\varphi C-P_0\varphi\) has \(\|D\|_2^2\le E_\varphi(C)/\zeta\) and \(X-X_0=\Pi_AD\Pi_{A^c}-\Pi_AP_0(1-\varphi)\Pi_{A^c}\), whose second term the kernel bound makes \(O(L)\), so \(E_\varphi(C)\le\kappa_fL\) gives \(d_*=2\kappa_f/\zeta+216K_\zeta^2C_\varphi^2\Lambda_b\), \(\Lambda_b=\sum_{z}|z|^2e^{-2b|z|_1}=6r(1+r)^3/(1-r)^5\), \(r=e^{-2b}\). Only the gap of \(h_0\) enters, no gap of the evolving generator.
```

---

## Item 5: Edit 5 ("exact primitive")

**1. Passage** (sec:results-firstlaw): "For $\kappa_R\ne0$ its exact primitive is $\delta(A/4G)$ whatever the exterior work (integrability conditions in Ref.~\cite{McKinneyCode2026})." Quoted correctly.

**2.** From the displayed identity, $(\delta E-W_{\rm ext})/(\kappa_R/2\pi)=\tfrac{2\pi}{\kappa_R}\tfrac{\kappa_R}{8\pi G}\delta A=\delta(A/4G)$ ✓. "Its" has no clear antecedent and a primitive is $A/4G$, not $\delta(A/4G)$. **3.** Mathematics right, wording ambiguous: CLARIFICATION. **4.** Reviewer's text correct; drops the integrability reference, which should stay (it is what justifies the general nonvacuum family). No other site carries the phrase. **5–6.** No qualifier added.

**Recommended replacement** (inline, page-neutral):
```latex
For \(\kappa_R\ne0\) the one-form \((\delta E-W_{\rm ext})/(\kappa_R/2\pi)\) is exact, equal to \(\delta(A/4G)\) whatever the exterior work (integrability conditions in Ref.~\cite{McKinneyCode2026}).
```

---

## Item 6: Edit 6 ($f_{\rm ext}$)

**1. Passage** (sec:star, formation): "Thus $f_{\rm ext}=1-R_t/(2GE)$, by the rows the net signed charge outside the trapping radius (sea departure and reference response included, not a particle fraction), is 0.04 to 0.05 …" Quoted correctly.

**2.** At the fine-grid root $w=-1$, so $\mathcal M(R_t)=R_tw^2/2G=R_t/2G$ and $1-R_t/(2GE)=[E-\mathcal M(R_t)]/E$: a fraction, dimensionless, while "charge" is an energy. Downstream uses ($\kappa_R/\kappa_H=N_R\beta_R/(1-f_{\rm ext})$ in sec:star and sec:scaling-far) are consistent with the fraction: $\kappa_R/\kappa_H=N_R\beta_R\cdot2GE/R=N_R\beta_R/(1-f_{\rm ext})$ ✓. **3.** CLARIFICATION. **4.** Reviewer's text correct; a display is unnecessary; keep the parenthetical. 

**Recommended replacement** (inline):
```latex
Thus \(f_{\rm ext}=1-R_t/(2GE)=[E-\mathcal M(R_t)]/E\), by the rows the fraction of the boundary charge carried outside the trapping radius, where \(\mathcal M(R_t)=R_t/2G\) (sea departure and reference response included, not a particle fraction), is 0.04 to 0.05 (plus or minus 0.03) at \(T/r_s=15\) and \(\sigma=4a\).
```

---

## Item 7: Edit 7 and review Secs. C/E (proofs delegated to the repository)

**1. Passages**: app:local A.1 "full proof … in Ref.", A.2 "through a finite-volume limit (Ref.)", A.3 "proved in Ref.", A.4 "hypotheses in full and the proof in Ref.", A.5 partition constants "(Ref.)", and sec:results-vacuum "verified interval bound". Quoted correctly.

**2.** The reviewer's partial verifications are correct: $D=(I-P_0)\varphi C-P_0\varphi(I-C)$ (expands to $\varphi C-P_0\varphi$), orthogonal domains and ranges give $\|D\|\le1$, and $\|\Pi_AD\|_2^2\le2\operatorname{Tr}(\Pi_AP_{\le\lambda})+2E_\varphi/\lambda$, with $\lambda\propto L^{-1/2}$ giving $O(L^{3/2})$ as the manuscript says. The reviewer is also right that the Rotfel'd–Mirsky route of item 4 does not close the massless $L^{7/4}$ claim: for $\zeta=0$ the flat cross block decays only as $|x-y|^{-3}$, so $\|\Pi_IX_0\|_2^2\sim L^2d^{-3}$ is not $O(L)$ at bounded depth; the endpoint estimate with the massless projector is a genuine missing step. **3.** Not an error claim; a proof-availability request. **4–6.** Owner rule B4: full proofs are a held supplement sent with the referee response, so the paper text needs no change; the supplement must contain (i) the summable closed-path majorant of A.1, (ii) the finite-volume SSA-to-convexity limit of A.2, (iii) the massless $L^{7/4}$ endpoint estimate with its constant $C_*$, (iv) the partition and freezing constants of A.5, and (v) the $[111]$ input interval and the outward-rounded angular enclosure that yield $0.8164<\eta(0)<1.0203$ and $0.7883<\eta(0.2)<0.9831$ (the coarse bound $M_1\eta_{100}\le\eta\le\tfrac32\eta_{100}$ alone gives only $0.654<\eta(0)<1.180$, so the sharpened interval rests on the [111] data). Verdict OPEN; no replacement text.

---

## Item 8: Edit 8 (between-slice continuation)

**1. Passage** (sec:singularity): "The stored histories pass the slice tests above, and the between-slice bound is what the continuation calculation must certify." Quoted correctly. **2.** The reviewer's route is sound: if $\|F_u(z)-F_u(z_0)\|\le A_Cr_C+A_ur_u$ in the box and $\eta_0(A_Cr_C+A_ur_u)<\theta<1$, the Neumann series gives $\|F_u^{-1}\|\le\eta_0/(1-\theta)$; $\|[\bar h,C]\|_2\le2\|\bar h\|\,\|C\|_2=2H\sqrt n$ for a rank-$n$ projector; $\|\dot u\|\le\eta_0(1-\theta)^{-1}\|F_C\|\,2H\sqrt n$; a Picard interval closes the box. **3–6.** The manuscript already states exactly this certificate as outstanding, including that the held-source Jacobians are not the full-reference ones. OPEN; no text change.

## Item 9: Edit 9 (supply, depletion, release)

**1. Passage** (sec:page-effective): "its attainability with the prescribed marginals and complete release is assumed, not proved". **2.** From $S^{(K)}_{\rm esc}\le32K(K+1)(r/a)\ln2+S(\mathcal X)$ with $S_{\rm esc}=\alpha S_0=\alpha\pi R_0^2/G$, $r=cR_0$, $G=ga^2$: $K(K+1)\ge\alpha\pi R_0/(32gc\ln2\,a)$; at $g=\tfrac14$, $c=1$, $\alpha=0.6211$ the coefficient is $0.352$, so $K\gtrsim0.593\sqrt{R_0/a}$ ✓. This restates the manuscript's own "$K^2a/R_0\to0$ excludes an area-sized peak". **3–6.** OPEN research direction; nothing to change in the text.

## Item 10: Edit 10 (tensor modes)

**1. Passages**: Outlook "which the fixed lattice without tensor modes does not give" and sec:model-gravity fourth step "whose tidal tensor has no transverse-traceless part". **2.** Linearizing $ds^2=-N^2dt^2+|dx-v\,dt|^2$: $h_{00}=-2\delta N$, $h_{0i}=-\delta v_i$, $h_{ij}=0$; $R_{0i0j}=\tfrac12(\partial_i\partial_th_{0j}+\partial_j\partial_th_{0i}-\partial_t^2h_{ij}-\partial_i\partial_jh_{00})=\partial_i\partial_j\delta N-\tfrac12(\partial_i\partial_t\delta v_j+\partial_j\partial_t\delta v_i)$ ✓; in Fourier space every term carries a $k_i$ or $k_j$, so the TT projection vanishes ✓. **3.** NON-ISSUE: the manuscript already states this correctly; no change.

## Item 11: Edit 11 (accessible count)

The Outlook already says "The only charge operator defined here is second order about one Hartree covariance, so its shell dimensions depend on that centre" and lists the count as open. OPEN; nothing to change beyond item 3.

## Item 12: unverified numerics

The reviewer states repeatedly that no implementation defect is inferred. No manuscript change follows; the run-to-figure manifest belongs with the held supplement (item 7). Script outputs that do bear on printed numbers all agree with the text: verify_07 reproduces $\eta_{100}(0)=0.78658$ (0.786583) and the half-chain formula to $10^{-10}$; verify_05 reproduces the Page crossings $(0.5105,0.6211)$ and $(0.4773,0.5755)$, the ratio $4.71$, the $0.49$ percent bound, $r_s=2.953$ km, $T_H=6.17\times10^{-8}$ K, $\alpha_d=2.80\times10^{-13}$, $\delta T/T_H=1.39\times10^{-76}$, $3.3\times10^{65}$ yr, $T_d=57.8r_s$; verify_04 reproduces $-9\sigma^2/8r^2$ and the horizon expansions $z^3/2$, $\tfrac32z^2$; verify_01 confirms the Clifford square, the rigid-turn cancellation $[h_0,L_Q+\tfrac12\Sigma^{\rm lat}]=0$ (I checked the sign convention $[X_i,T_i]=-aT_i$), the $-\tfrac5{12}$ and $-\tfrac23$ series, and the positivity identity; verify_02 confirms the radial rows, first variation, clock identity, flux sign and root slope. These test what they claim except the tautological "light-only root factor" check noted under item 2.

---

## Open items left after the corrections

- **O1 (from item 1).** Complete fourth-gradient classification: enumerate all isotropic, degree-one, fourth-order densities (quadratic in $\nabla\nabla v$, $\nabla\nabla N$, and mixed $\nabla N\,\nabla v$, $(\nabla N)^2\nabla v$ terms), apply the four requirements including the potential-flow divergence test at variable clock (the analogue of the step that removed $c$), and count the free coefficients. Closing calculation: the Euler–Lagrange null test on $v=\nabla\phi$ at variable $N$ for the full quadratic basis.
- **O2 (item 3 / 11).** The accessible microcanonical count: construct, for one fixed second-order charge operator, the reachable shell components from the collapse subspace and their weights, and compare $\Delta S_{\rm mc}$ with $4\pi G[(E-\omega)^2-E^2]$.
- **O3 (item 4 residual).** The history-uniform budget $E_\varphi(C)\le\kappa_fL$ on cubic cuts (now measured only on chains) and the massless $L^{7/4}$ endpoint estimate with its constant.
- **O4 (item 7).** The held supplement with lemmas (i)–(v) above and the numerical manifest.
- **O5 (item 8).** The between-slice certificate under the full $W_2$-dependent reference: interval enclosures of $\eta_0$, $A_C$, $A_u$, $H$, $\|F_C\|$ and the margins $\mathcal M/r^3\ge\mu_*$, Eq. sg-current-mass, on every consecutive slice interval.
- **O6 (item 9).** Autonomous supply, depletion and release from the boundary generator; the two-window remainder $\mathfrak d_c$ compared with the light debit.
- **O7 (item 2 residual).** The coupled depletion law itself remains assumed; the corrected sentence makes the exact balance and its first-order reduction explicit but establishes no debit rate.

## Adversarial self-check (three least certain verdicts)

**Item 4 (apply the derivation as a sketch).** Strongest counterargument: the owner's rule B4 keeps full proofs out of the paper and the page budget is binding; the manuscript's existing sentence already states the result with a repository citation, so the right action is supplement-only with no manuscript change. Response: a twelve-line sketch naming the key steps (strip analyticity, Rotfel'd, Mirsky, the concavity bound, the $d_*$ decomposition) is exactly what B4 asks for, and it closes a reviewer-flagged gap in a result the paper's spacing claim leans on; if the page limit is already saturated, the sketch should still replace the "details in Ref." parenthetical at least by its one-sentence core. Verdict stands.

**Item 2 (FIX rather than CLARIFICATION).** Counterargument: the author plainly meant the leading-order reduction, and $GL/N_R\sim10^{-5}$ at the computed size, so "equal to $-L$" is a harmless shorthand. Response: the sentence is unqualified, sits beside an exact identity whose factor it drops, and the manuscript retains the factor everywhere else, so the rendered text asserts an exact equality that is false; the correction changes the claim's content (exact to first order), which is the owner's definition of FIX. The recommended text is the same under either label, so the classification does not change the action.

**Item 3 (new error versus ALREADY-FIXED).** Counterargument: the previous round installed "the 'exactly when' criterion", possibly this very sentence, and the intended reading was infinitesimal (the shells are thin in $\omega$). Response: as rendered, the sentence equates a finite-shell ratio with Hawking's factor for finite $\omega$ and $q$; the quadratic area entropy itself violates it by $e^{4\pi G\omega^2}$, so the text is wrong regardless of which round wrote it; the only effect of the provenance is whether the previous fix is recorded as incomplete, which I have flagged. Verdict stands.