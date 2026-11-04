# Round 9 check of review_review.md / review_edits.md against commit 89c4e21db31

**Preliminaries.** The supplied review_edits.md contains seven edits (1–7); there are no edits 8–10 to verify. P1 = edit 1, C1 = edit 2, C2 = edit 3; edits 4–7 are the reviewer's own OPEN items. No round‑8 correction is re‑reported: the census marks the two fourth‑gradient terms, \(\beta_R\), the finite‑shell reciprocity ratio, the A.6 cross‑block bound, the first‑law one‑form and \(f_{\rm ext}\) as confirmed. I spot‑checked them independently (\(\kappa_R=N_R\beta_R/2R\) from \(w'=(1-2G\mathcal M')/2r\) at \(w=-1\); \(f_{\rm ext}=1-R_t/2GE=[E-\mathcal M(R_t)]/E\) from \(\mathcal M(R_t)=R_t/2G\); transverse response \((q^2/16\pi G)(1+2\lambda_1a^2q^2)v_T\) with the \(\lambda_2\) term vanishing because \(q\cdot\varpi=0\); \(\mathcal N_a\mathsf W_{a\to b}=\sum_{i\in a,j\in b}|T_{ij}|^2=\mathcal N_b\mathsf W_{b\to a}\); the A.6 window identity \(E_\varphi(C)-E_\varphi(P_0)=\Tr[(C-P_0)\varphi h_0\varphi]\) for projector \(C\); the Leibniz cancellation of \(-N_R\mathcal M'_R\dot R\) against the \(\beta_R\) term in Eq. (page-two-window)). All hold in the current text.

## 1. Verdict table

| # | Item | File / section | Claim in a few words | Verdict | Class | One‑line reason |
|---|---|---|---|---|---|---|
| 1 | P1 / edit 1 | 06_scaling.tex, sec:page-effective, sentence after Eq. (page-effective-maximum) | "the time fraction not" independent of the luminosity normalization | VALID-APPLY-MODIFIED | FIX | \(u_P/t_*=1-y_P^{3/2}\) depends only on \(\mathcal R_{\bar\nu},f\); a common rescaling of the emission currents rescales \(u_P\) and \(t_*\) together. Reviewer's text is right but adds a redundant display; page‑neutral sentence below. |
| 2 | C1 / edit 2 | 08_discussion.tex, Scope of the results | "no slicing is put in … flat‑slice form … is an output" | VALID-APPLY-MODIFIED | CLARIFICATION | The lattice and its time parameter supply the slices (requirements_new.txt lists "flat slices at the spacing a" among the declared rulers); the output is Einstein's form on them. Keep the true "no spacetime metric, no piece of GR" clause the reviewer's text drops. |
| 3 | C2 / edit 3 | 08_discussion.tex, area‑entropy paragraph | "window energy … grows no faster than the cut" | VALID-APPLY-MODIFIED | CLARIFICATION | Sufficient budget is \(E_\varphi(C)\le\kappa_fL\), linear in the radius; an area‑sized budget fed through A.6 gives \(O(L^2\sqrt{\log L})\) and would not keep the coefficient. |
| 4 | edit 4 | 05_star.tex, sec:singularity | between‑slice continuation certificate | NON-ISSUE (no text change) | OPEN | Already stated open and specifically scoped; the reviewer's enclosure route and velocity bounds are valid; goes to breakthrough tasks. |
| 5 | edit 5 | 06_scaling.tex, sec:page-effective | autonomous supply, depletion, release | NON-ISSUE (no text change) | OPEN | Already stated "assumed, not proved"; the reviewer's diagnostic and the \(K^2a/R_0\to0\) obstruction are consistent with the text. |
| 6 | edit 6 | 08_discussion.tex, Outlook, first and second questions | tensor sector and \(\lambda_1,\lambda_2\) cannot be selected | NON-ISSUE (no text change) | OPEN | Linearization \(\delta g_{ij}=0,\ \delta g_{0i}=-\delta v_i,\ \delta g_{00}=-2\delta N\) verified; \(\lambda\) family verified; already stated. |
| 7 | edit 7 | 08_discussion.tex, Outlook, third question | accessible charge‑sector weights | NON-ISSUE (no text change) | OPEN | Reciprocity identity verified; the paper already says the cut entropy is not the count. |
| 8 | edits 8–10 | — | — | — | — | Not present in the supplied file. |
| 9 | Withdrawn findings V4–V12, V14, V15 | various | sign/factor, ADM convention, first law, 0.49 % bound, Page/horizon/singularity overclaims | NON-ISSUE | — | Withdrawals are correct; spot‑checked the \(K\) convention on the PG vacuum, the mass row, the Minkowski form of the 0.49 % bound. |

## 2. Script audit (what each script actually tests)

- **verify_01.py**: symbolic momentum‑space checks of \(h_0^2=\varepsilon^2\), \(\tfrac12\partial_{p_j}h_0^2=q_j\), the completed rigid turn (I rederived \([\tfrac12\mathsf c_x\mathsf c_y\Sigma_z,h]=\tfrac ia\mathsf c_x\mathsf c_y(\alpha_ys_x-\alpha_xs_y)\), which cancels the orbital term exactly), the bracket \(i[e_c,e_s]=\tfrac18[(h_++h)^2-(h_-+h)^2]=F_aq_x\) (I verified this encoding of the periodic‑lattice bracket and the closed form), the \(-5/12\), and the cone expansions. These genuinely test the attached census rows.
- **verify_02.py**: symbolic checks of \(\mathcal M'=4\pi r^2(\rho+wJ)\), Eq. (sc-clock-exact), the row‑variation first‑law identity, the PG time‑transformation sign, \(\nabla^2(f\hat r)=-\tfrac94 v/r^2\) for \(f\propto r^{-1/2}\), the averaged root \(1-\tfrac94\sigma^2/r_s^2\) and slope, and \(T_H\,d(A/4G)/dE=1\). All test what they are attached to (rederived by hand; see item 9).
- **verify_03.py**: the crossing‑ray expansions, the dust ODE \(dg/du=-g(1-g)^4/2r_s\) (valid given \(dR/du=vg\), which follows from \(du/dt=1/(1+v)\) along the surface, verified), and the clock constant \(\pi L^2\sqrt{2G\Gamma}\) are genuine tests. The periastron block checks only the arithmetic of the reviewer's own averaged quartic bracket with an orientation factor inserted by hand; it does not independently derive the manuscript's multiplier. No edit depends on it.
- **verify_04.py**: random‑instance checks (evidence, not proofs) of the trace‑norm and rank entropy bounds, the cross‑block singular‑value formula, the pure‑covariance window identity (rederived above) and Wick's variance on four modes. They test the identities as stated.
- **verify_05.py**: quadrature of Eq. (dr-area-chain) at \(\zeta=0\), reproducing 0.78658. It tests the quoted planar number given the imported half‑chain formula; it tests nothing about \(\eta(0)=0.9191\) or the interval bound.
- **verify_06.py**: the first block tests \(\partial_\lambda(u_P/t_*)=0\) symbolically with \(\mathcal R\) held fixed, which is the P1 claim under the common‑normalization reading; the line "PASS: absolute uP scales as inverse normalization" is a bare print, not a test. **Discrepancy:** the re‑run output as supplied starts at "PASS coherent interior-transfer lower bound"; the two normalization prints and the four preceding checks (Page points at \(f=1\), \(0.887\); OV ratio; binding fraction) are absent although exit code 0 implies all asserts passed. The printed record therefore does not evidence those checks; I rederived them by hand below (all agree). Request a clean re‑run for the record. The solar‑mass arithmetic lines are genuine arithmetic checks and agree with my own.

## 3. Per‑item verification

### Item 1 — P1 / edit 1 (FIX, VALID-APPLY-MODIFIED)

1. **Passage** (sec:page-effective, immediately after Eq. (page-effective-maximum)): "and a lifetime fraction exists only with a finite evaporation time \(t_*\), the area fraction and height independent of the luminosity normalization, the time fraction not." The reviewer quotes it exactly.

2. **Mathematics.** Eq. (page-effective-curve) with constant \(\bar\nu\): \(R^3=R_0^3-6G\mathcal L\bar\nu u\). With \(t_*=R_0^3/(6G\mathcal L\bar\nu)\) (the box), \(R^3/R_0^3=1-u/t_*\), and since \(y=R^2/R_0^2\), \(y^{3/2}=1-u/t_*\). With constant \(\mathcal R_{\bar\nu}\) the first branch is \(\mathcal B/S_0=\mathcal R_{\bar\nu}(1-y)\); crossing \(fy\) gives \(y_P=\mathcal R_{\bar\nu}/(\mathcal R_{\bar\nu}+f)\), height \(f\mathcal R_{\bar\nu}/(\mathcal R_{\bar\nu}+f)\), \(u_P/t_*=1-y_P^{3/2}\). The "luminosity normalization" can only mean the common prefactor of the emission currents (\(\mathcal L\) and \(\dot S_{\rm marg}\) together): \(\mathcal R=T_H\dot S_{\rm marg}/L\) "is fixed by the one transmission and the one occupation" (Sec. VII.B), and this is the only reading under which the sentence's first claim (area fraction independent) is true. Under \(\mathcal L\to\lambda\mathcal L\): \(t_*\to t_*/\lambda\), \(u_P=t_*(1-y_P^{3/2})\to u_P/\lambda\), so \(u_P/t_*\) is invariant. "The time fraction not" is false; the absolute Page time is what depends on the normalization. Under the only alternative reading (normalization \(=\bar\nu\)): since \(s\) is concave with \(s(0)=0\), \(s(\nu n)\ge\nu s(n)\), so \(\mathcal R_{\bar\nu}\) increases as \(\bar\nu\) decreases and the area fraction, height and time fraction all change together; the sentence is false under that reading too. Numbers: \(\mathcal R=1.639,f=1\): \(y_P=0.62107\), \(S_P/S_0=0.6211\), \(u_P/t_*=1-0.62107^{3/2}=1-0.48945=0.5105\); \(f=0.887\): \(y_P=0.64885\), \(S_P/S_0=0.5755\), \(u_P/t_*=0.4773\) — the paper's quoted maxima, containing no \(\mathcal L\). Script: see audit; the symbolic test is the right test but its printed output is missing from the re‑run.

3. **Status.** A real, local prose error. The equation, maxima and figure are right.

4. **Replacement.** The reviewer's text is mathematically correct (\(u_P=\frac{R_0^3}{6G\mathcal L\bar\nu}[1-(\frac{\mathcal R_{\bar\nu}}{\mathcal R_{\bar\nu}+f})^{3/2}]\) is just \(t_*\) times the boxed fraction). But it adds a display and a sentence that restate the box ("a result prints once"; the paper sits at the 29‑page cap), and it drops the leading "and", breaking the sentence that continues from the box. Other sites carrying the claim: Discussion ("the Page time of Eq. (page-effective-maximum), the same formula as for any thermal emitter with that entropy ratio and the cut share \(f\)"), abstract and Sec. VII.D ("near half the evaporation time"), Fig. (page-conditional) caption — all consistent with the corrected statement and need no change.

5. **Scope.** No scope change; the constant‑\(\bar\nu\), self‑similar comparison‑case scope is in the preceding sentence and stays.

6. **No bridge needed.**

Replacement (paste in place of the quoted clause):
```latex
and a lifetime fraction exists only with a finite evaporation time \(t_*\); the area fraction, the height, and the time fraction \(u_P/t_*\) are all independent of a common normalization of the emission currents, which leaves \(\mathcal R_{\bar\nu}\) fixed and rescales \(t_*\) and \(u_P\) together, so only the Page time \(u_P\) itself depends on it.
```

### Item 2 — C1 / edit 2 (CLARIFICATION, VALID-APPLY-MODIFIED)

1. **Passage** (Discussion, Scope of the results): "No spacetime metric, no piece of general relativity, and no slicing is put in, since the fixed lattice and its time parameter are all there is, and the flat-slice form of the constraints is an output of the derivation." Quoted exactly by the reviewer.

2. **Mathematics.** Sec. III: the line element is "the unique quadratic form in \((dt,dx)\) with this cone, normalized to the lattice spacing", and "\(\gamma_{ij}=\delta_{ij}\) the metric of the slices"; the fourth step: a slice‑curvature term "vanishes on the flat lattice, so its coefficient … escapes every variation of \(N\) and \(v\)". So no spatial metric is varied anywhere; the slices are the constant‑lattice‑time site sets, flat at spacing \(a\) because the declared uniform sine hopping on cubic sites has an isotropic leading cone and the spacing is the ruler. requirements_new.txt, declaration (ii): "the rulers are the fixed lattice itself, flat slices at the spacing a, with no clock of the lattice's own." What the charge principle outputs is the coefficient combination \(a_1=-a_2,\ a_3=b=c=0\), i.e. Einstein's form of the constraints on those slices. Hence "the flat‑slice form … is an output" is true only read as "Einstein's form on the flat slices", and "no slicing is put in" is ambiguous: no slicing of a pre‑given spacetime is chosen (true), but the foliation by lattice time and the flat rulers are supplied (so the clause reads as false to a GR reader). The reviewer's reading is the correct one; no circularity.

3. **Status.** Presentational ambiguity in a summary sentence that the body and the requirements resolve: a clarification.

4. **Replacement.** The reviewer's text is correct in content but (i) drops "No spacetime metric, no piece of general relativity … is put in", which is true (the metric is read from the fermion's cone; GR enters only as vocabulary at the identification) and is the thesis of the Scope paragraph, and (ii) introduces "spatial kinematics", a term the paper does not use. Modified version below keeps the true clause. Other sites with the same claim, all consistent with the modified sentence: abstract ("assuming no light cone, vacuum, or metric"); Sec. I ("no light cone, vacuum, or metric assumed"); Sec. II opening ("Only the fixed cubic lattice and its spacing are assumed: no spacetime metric, no action"); Sec. II.B ("geometry is read from clock and flow, not assumed for them"); Sec. III ("the Hamiltonian and the momentum constraint of general relativity on flat slices"); Discussion later paragraph ("No background metric is assumed anywhere"). None says "no slicing".

5. **Scope.** The derived result (Einstein's form of the constraints) stays unqualified; stating that the slices come from the lattice is the owner's own declaration, not a relabelling of a result as an assumption.

6. **No bridge needed.**

Replacement:
```latex
Neither a spacetime metric nor any piece of general relativity is put in: the fixed cubic lattice and its time parameter are all there is, and they supply the slices, flat at the spacing \(a\) once the fermion's cone is read as Eq.~\eqref{eq:dm-metric}; the output of the derivation is Einstein's form of the constraints on those slices.
```

### Item 3 — C2 / edit 3 (CLARIFICATION, VALID-APPLY-MODIFIED)

1. **Passage** (Discussion, end of the "Three facts" paragraph): "A collapse keeps that coefficient while its state's window energy around the trapping cut grows no faster than the cut, as the sampled histories do on the angular-channel chains (Sec.~\ref{sec:results-vacuum})." Quoted exactly.

2. **Mathematics.** Sec. IV.B: "The sufficient direct-state test is \(E_\varphi(C)=\|\Omega_0^{1/2}(\varphi C-P_0\varphi)\|_2^2\le\kappa_f L\)" for a ball of radius \(La\), and "The flat coefficient therefore holds on any history whose propagated sea keeps its window energy linear in its size." A.6: \(\|X-X_0\|_2^2\le d_*L\) with \(d_*=2\kappa_f/\zeta+O(C_\varphi^2)\) gives \(|S_A(C)-S_A(P_0)|\le2\sqrt{m_Bd_*L}+\dots=O(L^{3/2}\sqrt{\log L})=o(L^2)\). With an area‑sized budget \(E_\varphi\le\kappa L^2\) the same chain gives \(\|X-X_0\|_2^2=O(L^2)\) and \(2\sqrt{m_BL^2}=O(L^2\sqrt{\log L})\), not \(o(A)\): the coefficient would not be preserved. Sec. VII.A (massless) likewise needs \(\kappa_fL\) to reach \(L^{7/4}\). "The cut" is a surface whose natural size is \(\propto L^2\), so the summary can be read as the weaker, insufficient condition. The reviewer is right.

3. **Status.** Clarification; nothing in the estimates is wrong.

4. **Replacement.** The reviewer's wording is correct. I keep the prose and name the radius and the budget; I also point the "sampled histories" clause at the section where \(\kappa_f\le7.07/7.49\) is reported. Other sites, all linear in \(L\) and unchanged: Sec. IV.B (definition), Sec. IV.F ("uniform budget of Sec. IV.B's direct-state form"), Sec. VI.F (values), Sec. VII.A (massless theorem), App. A.6.

5. **Scope.** Unchanged: still "as the sampled histories do on the angular-channel chains".

6. **No bridge needed.**

Replacement:
```latex
A collapse keeps that coefficient while its state's window energy around the trapping cut grows no faster than the cut's radius in lattice units, the budget \(E_\varphi(C)\le\kappa_fL\) of Sec.~\ref{sec:results-vacuum}, as the sampled histories do on the angular-channel chains (Sec.~\ref{sec:results-diagnostics}).
```

### Item 4 — edit 4 (OPEN; no text change)

1. **Passage** (sec:singularity, last paragraph): "The stored histories pass the slice tests above, and the between-slice bound is what the continuation calculation must certify." Exact.
2. **Mathematics.** The reviewer's bounds: for a projector covariance of rank \(\le d\), \(\|C\|_F\le\sqrt d\), so \(\|{-i}[\bar h,C]\|_F\le2\|\bar h\|_{\rm op}\|C\|_F\le2H\sqrt d\); \(V_u=-F_u^{-1}F_CV_C\) gives \(\|V_u\|\le2K_CH\sqrt d/s\) in compatible norms. Valid (crude) displacement bounds; the requirement to first correct each seed to a zero of the full‑reference \(F\) by a validated Newton–Kantorovich step is correct, since the stored Jacobian and residual checks are "under the held-source operators … rather than Eq. (dm-reference)" (the paper says so).
3. **Status.** The manuscript already states the open certificate with its components (regular branch with margin \(\mathcal M/r^3\ge\mu_*\), Eq. (sg-current-mass), uniform inverse of the full \(F_u\), self‑closing field and reference‑derivative bound). Nothing to change; "must certify" is specific scope wording.
4–6. No replacement; no bridge. Closing calculation: the first consecutive stored‑slice interval spanning formation, with full‑reference residual after seed correction, interval enclosure of \(F_u^{-1}\) and of the reference derivatives, and the positive‑clock and enclosed‑mass margins reported.

### Item 5 — edit 5 (OPEN; no text change)

1. **Passage** (sec:page-effective): "The only release mechanism in hand, an enqueue-and-swap circuit of Gaussian Schmidt pairs across the trapping cut, is an externally controlled Gaussian proposal, not a realization of the model's autonomous boundary generator, and its attainability with the prescribed marginals and complete release is assumed, not proved". Exact.
2. **Mathematics.** The identity the reviewer's diagnostic rests on holds: for the pure tripartition \(B,\mathcal E_u,\mathcal S_u\), \(S(B\mathcal S)=S(\mathcal E)\) and \(S(\mathcal S)=S(B\mathcal E)\) give \(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\). The asymptotic obstruction: \(32K(K+1)(r/a)\ln2/S_0\) with \(S_0=\pi R_0^2/G\), fixed \(G/a^2\), \(r=O(R_0)\) is \(O(K^2a/R_0)\), as the paper states. Neither the Gaussian identities nor Eq. (page-two-window) force a nonzero sustained supply or a negative block increment; the stored slices report positive increments only.
3. **Status.** Already stated as assumed; "assumed, not proved" is a specific label. No change.
4–6. No replacement. Closing calculation: on a coupled history, the once‑counted packet matrix \(D_u\) with \(S(B|\mathcal E_u)\), \(I(B{:}\mathcal E_u)\), \(I(B{:}\mathcal S_u)\), the received Killing energy and the two‑window storage/work terms computed together, plus a statement of how the channel/store scaling \(K^2a/R_0\) is accessed.

### Item 6 — edit 6 (OPEN; no text change)

1. **Passage** (Outlook): "tensor propagation, the laws' coupled realization on general flows, and the deformation algebra of the constraints, whose Dirac-Schwinger relations are not realized on fixed coordinates." and the second question's \(\lambda_1,\lambda_2\) paragraph. Exact.
2. **Mathematics.** \(ds^2=-N^2dt^2+|dx-v\,dt|^2\) gives \(g_{00}=-(N^2-|v|^2)\), \(g_{0i}=-v_i\), \(g_{ij}=\delta_{ij}\); linearizing at \(N=1,v=0\): \(\delta g_{00}=-2\delta N\), \(\delta g_{0i}=-\delta v_i\), \(\delta g_{ij}=0\). The transverse‑traceless sector of linearized gravity lives in \(h^{TT}_{ij}\), which vanishes identically, confirming the paper's "no tensor mode". The \(\lambda\) family: on a transverse flow \(B=\nabla\times\nabla\times v=q^2v\), \(|B|^2=q^4|v|^2\), \(C_{ij}C_{ji}=0\) since \(q\cdot\varpi=0\), so the momentum law is \((q^2/16\pi G)(1+2\lambda_1a^2q^2)v_T\) as printed; both terms vanish on potential flows, so the four requirements cannot fix them.
3–6. Already stated; no replacement; closing requires additional microscopic structure or an independently justified dynamical requirement, not available here.

### Item 7 — edit 7 (OPEN; no text change)

1. **Passage** (Outlook): "The cut entropy is not this count." Exact.
2. **Mathematics.** With equally weighted shells and Hermitian \(T\), \(\mathsf W_{a\to b}=\mathcal N_a^{-1}\sum_{i\in a,j\in b}|T_{ij}|^2\), so \(\mathcal N_a\mathsf W_{a\to b}=\mathcal N_b\mathsf W_{b\to a}\); the ratio is \(\mathcal N_b/\mathcal N_a\) and equals Hawking's factor to first order exactly when \(\partial_ES_{\rm mc}=8\pi GE\). Reciprocity constrains a ratio once dimensions and weights are given; it does not supply them. The reviewer's two‑centre stability diagnostic is a sensible first test.
3–6. Already stated open; no replacement. Closing calculation: a centre‑independent accessible ensemble from the collapse dynamics with its weights, or the two‑centre re‑expansion stability test as a first step.

### Item 8 — edits 8–10

Not present in the supplied review_edits.md; nothing to verify.

### Item 9 — withdrawn findings V4–V12, V14, V15 (NON-ISSUE)

Spot checks: (V5) with shift \(-v\) and \(K_{ij}=\tfrac1{2N}(\partial_iv_j+\partial_jv_i)\), the PG vacuum \(N=1,\ v=-\sqrt{r_s/r}\) gives \(K_r=\tfrac12\sqrt{r_s}r^{-3/2}\), \(K_t=-\sqrt{r_s}r^{-3/2}\), \((\mathrm{tr}K)^2-\mathrm{tr}K^2=\tfrac94-\tfrac94=0\) and radial \(\nabla\cdot[K-(\mathrm{tr}K)I]=T_r'+\tfrac2r(T_r-T_t)=-3+3=0\): both rows solved with the paper's signs; and \((\mathrm{tr}K)^2-\mathrm{tr}K^2=\tfrac{2}{N^2r^2}(rv^2)'\) reproduces Eq. (dr-radial) and \(\mathcal M'=4\pi r^2(\rho+wJ)\). (V12) Minkowski on the orbital sum: \(\sqrt{\sum_j\|PU\phi_j^{\rm in}\|^2}\ge\sqrt{50.7}-\sqrt{29}=1.735\), squared \(3.01\), over 611 gives \(0.49\%\). (V4) see script audit. The remaining withdrawals (V6–V11, V14, V15) correctly recognise statements the text already scopes. No action.

## 4. Open items left after the corrections

1. **Between‑slice continuation certificate** (item 4). Unresolved: a validated enclosure of the coupled trajectory under Eq. (dm-reference) across one stored‑slice interval. Closes with: seed correction to the full‑reference zero of \(F\) (Newton–Kantorovich), interval bounds \(\|\bar h\|\le H\), \(\sigma_{\min}(F_u)\ge s\), \(\|F_C\|\le K_C\), reference‑derivative and clock/mass margins verified on a box containing the displacement bounds, first on the interval spanning formation.
2. **Autonomous supply, depletion and release** (item 5). Closes with the simultaneous computation of \(S(B|\mathcal E_u)\), \(I(B{:}\mathcal E_u)\), \(I(B{:}\mathcal S_u)\), received Killing energy and two‑window storage/work on a coupled history, together with the channel/store scaling statement that evades \(K^2a/R_0\to0\).
3. **Tensor sector and \(\lambda_1,\lambda_2\)** (item 6). Closes only with an additional microscopic input or an independently justified dynamical requirement; none is in the paper.
4. **Accessible charge‑sector count** (item 7). Closes with a centre‑independent accessible ensemble and weights; first step, the two‑centre re‑expansion stability test.
5. **Evidence housekeeping (not a manuscript item):** the supplied verify_06.py output is missing its first six expected lines; a clean re‑run should be filed with this round. The missing checks are confirmed by hand above.
6. **Wording‑rule flag (pre‑existing, outside these edits):** the bare phrase "is not established" appears twice in sec:scaling-far ("whether the surface arrives or halts there, and at what density, is not established"; "recession with vanishing exterior work in the astrophysical limit is not established"). Under the owner rule these should carry specific scope words at the paper lane's next pass (e.g., "is a computation this paper does not make" for the first; "is an assumption for the object's fountain", which the following sentence already says, for the second). No claim changes.

## 5. Adversarial self‑check (three least certain verdicts)

**C1 (clarification, modified).** Strongest counter: in GR usage "slicing" means a foliation chosen within a given spacetime; here no spacetime pre‑exists, so "no slicing is put in" is literally true, "flat‑slice form" plainly means the form of the constraints, and the body is unambiguous — hence STYLISTIC-SKIP; or, conversely, the reviewer's "supplied spatial kinematics" relabels a derived output (flatness) as an input, a hedge. Response: the owner's requirements declare "flat slices at the spacing a" as the rulers of declaration (ii), so calling them supplied is the owner's own framing, not a relabelling; and lattice time is a preferred foliation, which the sentence's own "since" clause concedes, so a GR reader can take "no slicing" as false. The modified sentence keeps every true claim (no metric, no GR, Einstein's form as output). Verdict unchanged.

**C2 (clarification, modified).** Strongest counter: the sentence cites Sec. IV.B, where the budget is defined linear in \(L\), so no attentive reader is misled, and the owner's rule against restating conditions in summary sentences argues for leaving it. Response: the summary as written can be read as an area‑sized budget, which the paper's own A.6 chain shows is insufficient; naming the radius costs no length and restates no new condition, only the one the sentence already invokes. Low stakes either way; verdict unchanged.

**Edits 4–7 as NON-ISSUE for text.** Strongest counter: the reviewer's routes (interval enclosure; the conditional‑entropy triad) are more specific than the paper's open statements, and a reader would benefit from one sentence naming the decisive calculation; or an OPEN classification should trigger a text mark. Response: the owner's rules exclude method and lane detail from the manuscript, the paper already names each certificate's components and labels each assumption specifically ("must certify", "assumed, not proved", "lie outside this derivation", "is not this count"), and the submitted paper is to carry no marks for work not planned in it; the routes belong in breakthrough tasks, where I have placed them. Verdict unchanged.