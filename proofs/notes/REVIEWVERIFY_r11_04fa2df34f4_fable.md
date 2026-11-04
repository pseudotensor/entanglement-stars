# Verification of referee round 11 (manuscript commit 04fa2df34f4)

Note: the task names "edits 1-10"; the supplied review_edits.md contains nine edits. There is no edit 10.

## 1. Verdict table

| # | Item | File / section | Claim in a few words | Verdict | Class | One-line reason |
|---|---|---|---|---|---|---|
| 1 | B1 = edit 1 | 05_star, "The trapped interior", sentence after eq:sc-clock-exact | "close only at bounded λ and small charge fraction outside R_π" is false | VALID-APPLY-MODIFIED | FIX | Necessity claim is false (J=0 ⇒ N constant, λ=1, any outside charge; verified, and the dust comparison solution is an in-paper instance); reviewer's text is correct but six display lines at the page cap; same correction fits in two sentences. |
| 2 | B2.1 = edit 2 | 03b_quantum_sector, definition of H_+ | "no leakage" should name the fixed expansion point | VALID-APPLY | CLARIFICATION | Π is a spectral projector of the same Ê_2, so the statement is true exactly at the fixed C̄; naming it matches requirement (v) ("its leakage the measured error" along the history) and the paper's next sentence on re-expansion errors. |
| 3 | B2.2 = edit 3 | figure files for fig:star-collapse, fig:page-v4 (axis text) | "time at infinity T/r_s" → "flat-slice time T/r_s" | STYLISTIC-SKIP | CLARIFICATION | Label lives in the PDFs, not in the source given; N(∞)=1 makes both labels correct as rates; captions and text already define T as flat-slice time; harmless to relabel if the figures are regenerated anyway. |
| 4 | C2 | 05_star formation paragraph; App. B | ~7–8 % charge change by T/r_s=15 | NON-ISSUE | OPEN (repository check) | Arithmetic confirmed (2GE ≈ 54.1–54.7a vs 59.07a); the paper makes no integrator-conservation claim and names "the step's charge drift on stored histories"; f_ext uses contemporaneous E and is step-robust. No text change; τ-convergence of E(T) is a repository query. |
| 5 | edit 4 | 05_star, sec:singularity | between-slice tube strategy | NON-ISSUE (no edit) | OPEN | Grönwall, Neumann inverse and first-exit steps are correct and coincide with what the paper says the between-slice bound must verify; no text supplied or needed (but see word-rule item: "must certify"). |
| 6 | edit 5 | 06_scaling, sec:page-effective | energy closure cannot fix a turnover; measure inter-arrival covariance | NON-ISSUE (no edit) | OPEN | Four-mode ±ln2 example verified (script 06 tests it) and is already printed in the paper; the ΔS=Tr s(A_{u+B})−Tr s(A_u) computation is the open calculation, as the paper states. |
| 7 | edit 6 | 05_star, sec:results-diagnostics | chain sea budget has no cubic counterpart | NON-ISSUE (no edit) | OPEN | Duhamel identity and the HS bound ≤2∫‖(h_cub V−Vh_ch)U_ch P‖_2 verified; identifies the defect to bound; paper already says "no cubic reading". |
| 8 | edit 7 | 08_discussion, Outlook 1 | no TT mode in the clock-flow field space | NON-ISSUE (no edit) | OPEN | h_00=−2δN, h_0i=−δv_i, h_ij=0 ⇒ R_0i0j=∂_i∂_jδN−½∂_t(∂_iδv_j+∂_jδv_i), TT projection zero for every k≠0; confirms the paper's sentence. |
| 9 | edit 8 | 08_discussion, Outlook 2 | λ_1, λ_2 invisible on potential flows | NON-ISSUE (no edit) | OPEN | B=C=0 on v=∇φ and both quadratic invariants have zero first variation there; confirms the paper's sentence. |
| 10 | edit 9 | 08_discussion, Outlook 3 | microcanonical count not supplied by cut entropy | NON-ISSUE (no edit) | OPEN | Reciprocity ratio d(E−ω)/d(E) and S′_micro=8πGE verified; matches the paper's third Outlook question. |

Script status (all PASS reproduced; my own arithmetic agrees):
- verify_01: tests Clifford square, completed-turn commutator at symbol level, axial bracket F_a and the −5/12 term, cone coefficients, root expansions. Tests the B-table rows it is attached to.
- verify_02: tests M′ identity, first-variation identity, surface slope, clock equation, F=−T^r_t sign, vector Laplacian −9/(4r²), averaged root and slope. Tests its rows.
- verify_03: tests edit 1's counterexample (rows at N=1, J=0; root at 2b; outer root; zero integrand; 98.6 % outside b/4). Tests exactly the B1 claim. It does not test the replacement's inequality |e^x−1|≤e^{|x|}−1, which is elementary and I checked.
- verify_04: tests the orbit average of a quartic term the reviewer wrote down; it does not derive that term from the lattice Hamiltonian, so it checks the averaging arithmetic, not the full PN claim. Attached to no issue.
- verify_05: arithmetic on printed inputs (E_max, Page peaks, spacings, solar numbers, ℛ, strain, coherence-sensitive 0.49 %, ceilings, drift). Tests C1/C2 arithmetic only, as labelled.
- verify_06: tests edit 5's example (projectors, equal diagonals, ±ln2, equal diagonal energies). Tests the claim.

## 2. Per-item verification

### Item 1 (B1 / edit 1): the lapse-comparison sentence

1. Passage (sec:star, "The trapped interior"), verbatim:
"The lapse values \(N(R_\pi)\) and \(N(R_t)\) are close only
at bounded \(\lambda\) and a small charge fraction outside \(R_\pi\), a comparison radius."
The reviewer's quote is accurate (line break joined).

2. Mathematics. From eq:dr-charge and eq:dr-radial: v=Nw, rv²=2GN²M, so (rv²)′=2G(2NN′M+N²M′)=8πGr²N²ρ gives (ln N)′=(4πr²ρ−M′)/(2M)=2πr²ρ/M−M′/(2M); with 4πr²ρ=λM′ this is ½(λ−1)(ln M)′. Both equalities of eq:sc-clock-exact hold. Now set J=0: the momentum row vN′=−4πGrN²J gives N′=0 wherever v≠0 (the paper's domain, w≠0), so N(R_π)=N(R_t) exactly and λ=1 wherever M′≠0, with no condition on how much charge lies outside R_π. The reviewer's smooth profile checks: M=Er³/(r²+b²)^{3/2} gives M′=3Er²b²/(r²+b²)^{5/2}=4πr²ρ; v²=2GE r²/(r²+b²)^{3/2} has its maximum (5/3)^{3/2}/2≈1.08>1 at r=√2 b and equals 1 at r=2b on the decreasing branch, so 2b is the outer trapping root; 1−M(b/4)/E=1−17^{−3/2}=0.986. The marginally bound dust comparison solution (N=1, comoving, J=0) is a second counterexample already in the paper. The necessity claim is false; the sufficient direction (bounded λ, M nondecreasing, small annulus fraction) is what the author could have meant. A second defect: "outside R_π" counts charge beyond R_t, which drops out of N(R_t)/N(R_π); only the annulus enters.

3. Real problem: a sufficient control printed as necessary. FIX.

4. Reviewer's replacement: N(r_1)/N(r_0)=e^{I_{01}} and |e^{I}−1|≤e^{D}−1 are correct (|e^x−1|≤e^{|x|}−1 for all real x), and the J=0 remark is correct. It costs about six lines of display at a 29-page cap; the same content fits in two sentences. No other site carries the claim: R_π appears in sec:results-filling (definition), the method paragraph ("arrival at the comparison radius R_π"), the interior paragraph ("1.8 times the comparison radius"), and sec:scaling-far ("reaches the comparison radius R_π∝χ^{2/3}r_s"); none restates closeness of lapse values.

5. Scope: a false "only if" becomes a true "if" plus a true identity; no scope label is added or dropped.

6. No bridge needed.

Replacement (paste in place of the quoted sentence):
```latex
Integrating Eq.~\eqref{eq:sc-clock-exact} between \(R_\pi\), a comparison radius, and \(R_t\) gives \(N(R_t)/N(R_\pi)=\exp\int_{R_\pi}^{R_t}(\ln N)'\,dr\), so the two lapse values are close whenever \(\int_{R_\pi}^{R_t}|(\ln N)'|\,dr\) is small, as it is at bounded \(\lambda\) when \(\mathcal M\) is nondecreasing between the two radii and the annulus holds a small fraction of \(\mathcal M(R_t)\). The condition is sufficient, not necessary: at \(J=0\) the clock is constant, so the two values coincide whatever the charge outside \(R_\pi\).
```
(Check of the sufficient case: with M′≥0 on the annulus, ∫|(ln N)′|dr ≤ ½ sup|λ−1| ln[M(R_t)/M(R_π)] = ½ sup|λ−1| ln[1/(1−f_ann)].) The reviewer's longer version may be used instead if the page budget allows; it is correct.

### Item 2 (B2.1 / edit 2): "no leakage"

1. Passage (sec:model-operators), verbatim: "Then \(H_+=\Pi\widehat E_2\Pi\) defines the spherical evolution there, with \(\tfrac12E\,\Pi\le H_+\le\tfrac32E\,\Pi\) and no leakage, and the deviation operator \(\mathcal B=4(\widehat E_2/E-I)^2\) measures the departure." Quote accurate.

2. Mathematics. Π=1_{[E/2,3E/2]}(Ê_2) is a spectral projector of the finite-dimensional self-adjoint Ê_2, so [Π,Ê_2]=0, H_+=Ê_2Π, spec(H_+|_{Ran Π})⊂[E/2,3E/2] gives the two-sided bound, and e^{−iH_+t} (equivalently e^{−iÊ_2t}) leaves Ran Π invariant: no leakage at the expansion point C̄ where Ê_2 and Π are defined. Correct. Along a history the expansion point moves and Π changes; the paper's next sentence already lists "re-expansion errors" as measured quantities, and requirement (v) describes the branch as "a fixed projector, its leakage the measured error". So the fixed-point reading is the intended one; making it explicit removes the dynamical misreading.

3. CLARIFICATION, not a hedge: the paper never claimed no leakage along a history.

4–6. No other site states "no leakage"; the Discussion's "the sector where the boundary charge is sharp" is unaffected. No scope change, no bridge.

Replacement:
```latex
At the fixed expansion point \(\bar C\), \(H_+=\Pi\widehat E_2\Pi\) defines the spherical evolution in \(\operatorname{Ran}\Pi\), with \(\tfrac12E\,\Pi\le H_+\le\tfrac32E\,\Pi\) and no leakage from it, since \(\Pi\) is a spectral projector of the same \(\widehat E_2\), and the deviation operator \(\mathcal B=4(\widehat E_2/E-I)^2\) measures the departure.
```
(The reviewer's shorter wording is also acceptable; the added "since" clause is what prevents the misread.)

### Item 3 (B2.2 / edit 3): figure axis labels

1. Passage: the axis text is inside fermion_new/figures/item5_collapse_v3.1.pdf and page_v5.pdf, not in the source supplied, so I cannot confirm "time at infinity" is printed there. The captions read "Radii and flat-slice clocks use nominal \(r_s=64a\)" and "against slice time"; the text reads "Times are the flat-slice lattice time \(T\) in units of \(r_s\) unless a clock is named ... and the receiver the infinity clock \(t_\infty\)" and "Stored times are flat-slice \(T\), not arrival times."

2. Mathematics. In the flat-slice line element N(∞)=1 and v(∞)=0, so dT is the proper time of a static far observer, and dt_s=dT at fixed r. "Time at infinity" is therefore correct as a rate; the only possible confusion is with the receiver clock t_∞, which labels reception events rather than slices.

3. Stylistic. No LaTeX change is needed; captions already fix the meaning. If the figure scripts are run for any other reason, relabel the horizontal axes "flat-slice time \(T/r_s\)". Otherwise skip.

### Item 4 (C2): charge drift

1. Passage: "the outer trapping radius enclosed in \([51.750,51.771]\,a\)" with f_ext "4.4 to 5.3 percent at \(T/r_s=15\)", and "\(2GE_0=59.07a\) (\(E_0\) the charge at release...)".

2. Arithmetic: 2GE=R_t/(1−f_ext) ∈ [51.750/0.956, 51.771/0.947]a=[54.13, 54.67]a, i.e. 7.5–8.4 % below 59.07a (script 05 agrees). Correct.

3. The paper claims conservation for the differentiable Hartree flow (App. B) and for the hydrodynamic exterior, and explicitly names the finite-step departure: App. B "zero on the Hartree flow and the step's charge drift on stored histories"; sec:results-diagnostics "at the level of the held step's charge deficit"; Fig. star-collapse(b) plots 2GE as a varying dotted curve. f_ext is computed with the contemporaneous E and is step-robust (4.1–5.3 % over τ=8,4,2), so the formation reading does not rest on E_0. NON-ISSUE for the text. The repository question stands: how E(T/r_s=15) changes with τ. If the drift shrinks ∝τ it is a step error; if it does not, it needs a physical or implementation account. Listed under open items; nothing to change in the text now (section 4b: location-only).

### Item 5 (edit 4): between-slice continuation

Passage (sec:singularity): "The stored histories pass the slice tests above, and the between-slice bound is what the continuation calculation must certify." Mathematics of the proposal: with F(C,u)=0 and F_u invertible, u̇=−F_u^{−1}F_CĊ (as in the paper); for an approximate path with defect d and ‖Dg‖≤L on a tube, e′≤Le+d gives e(t)≤e_0e^{L(t−t_j)}+(d/L)(e^{L(t−t_j)}−1), limit e_0+d(t−t_j) at L=0; ‖I−BF_u‖≤θ<1 gives ‖F_u^{−1}‖≤‖B‖/(1−θ) by the Neumann series; first exit closes the tube. All correct and standard, and this is exactly the paper's stated between-slice bound (margin M/r³≥μ_*, eq:sg-current-mass, uniform inverse bound on the full F_u, self-closing field and reference bounds). No edit; class OPEN. What closes it: the full-reference residual d and derivative bound L on each stored interval with the inverse bound via a reference inverse B. Owner word rule: this sentence uses "certify"; see word-rule list.

### Item 6 (edit 5): entropy-reducing received block

Passage (sec:page-effective): "its attainability with the prescribed marginals and complete release is assumed, not proved". Mathematics: with occupied orbitals (E+S_1)/√2, (B+S_2)/√2 the {E,B} block is diag(½,½), S(EB)=2ln2, S(E)=ln2, S(B|E)=+ln2; with (E+B)/√2, (S_1+S_2)/√2 the block is ½[[1,1],[1,1]], S(EB)=0, S(B|E)=−ln2; all diagonal occupations are ½ so every one-body diagonal energy agrees. Correct, script 06 tests it, and the paper already prints it ("two four-mode pure histories with identical received energies give block increments +ln2 and −ln2"). The paper also says "inter-arrival covariance and cross-cut correlations require measurement" and "The stored histories carry no computed value of S_esc". No edit; class OPEN. What closes it: ΔS=Tr s(A_{u+B})−Tr s(A_u) on the stored histories with the D_u construction and bounded return and omitted-mode errors.

### Item 7 (edit 6): cubic sea budget

Passage (sec:results-diagnostics): "its propagated-sea term has no cubic reading". Mathematics: d/ds[U_cub(t,s)VU_ch(s)]=iU_cub(t,s)(h_cubV−Vh_ch)U_ch(s), integrated from 0 to t gives U_cub(t)V−VU_ch(t)=−i∫_0^tU_cub(t,s)(h_cubV−Vh_ch)U_ch(s)ds (sign as displayed). With A=U_cubV, B=VU_ch isometries, APA†−BPB†=(A−B)PA†+BP(A−B)†, so the Hilbert–Schmidt norm is ≤2‖(A−B)P‖_2≤2∫‖(h_cubV−Vh_ch)U_ch(s)P‖_2ds. Correct. Identifies the defect; bounds nothing yet. No edit; OPEN. What closes it: a matched-mass, matched-window cubic propagated-sea budget on a size ladder, or a bound on the displayed integral for the occupied zone modes.

### Item 8 (edit 7): tensor sector

Passage (Outlook): "its clock-flow field space having no transverse-traceless perturbation". Linearizing eq:dm-metric about N=1, v=0: g_00=−N²+|v|²⇒h_00=−2δN; g_0i=−v_i⇒h_0i=−δv_i; h_ij=0. Linearized R_0i0j=½(∂_i∂_0h_0j+∂_0∂_jh_0i−∂_0²h_ij−∂_i∂_jh_00)=∂_i∂_jδN−½∂_t(∂_iδv_j+∂_jδv_i); both terms are longitudinal in at least one index, so the TT projection vanishes for every k≠0. Confirms the paper. No edit; OPEN as the paper already states.

### Item 9 (edit 8): fourth-gradient coefficients

Passage (Outlook): "which nothing declared selects". On v=∇φ, ∇×v=0 so B=∇×∇×v=0 and C_ij=∂_i(∇×v)_j=0; |B|² and C_ijC_ji are quadratic in vanishing quantities, so their first variations vanish on that sector. Matches the paper's "Both terms vanish with their first variations on every radial or potential flow". No edit; OPEN.

### Item 10 (edit 9): microcanonical count

Passage (Outlook): "a spectral count of a charge quantization the five declarations do not fix". With uniformly weighted shells and reciprocal |M_if|=|M_fi|, Γ_em(E,ω)/Γ_abs(E−ω,ω)=d(E−ω)/d(E)=exp[S(E−ω)−S(E)]; matching e^{−ω/T_H} to first order needs S′=1/T_H=8πGE, which A/4G=4πGE² satisfies but which no cut-entropy or first-law statement derives. The affine-versus-nonlinear remark is the paper's own. No edit; OPEN.

## 3. Open items left behind

1. Between-slice continuation (edit 4): compute, on every consecutive stored interval, the full-reference residual d and derivative bound L of the reduced system z=(C,u) and the inverse bound on the full F_u (via ‖I−BF_u‖≤θ<1); the paper's sentence stands with "verify" in place of "certify".
2. Inter-arrival entropy increment (edit 5): ΔS=Tr s(A_{u+B})−Tr s(A_u) with the once-counted D_u on the stored history, return and omitted-mode errors bounded.
3. Cubic propagated-sea budget (edit 6): matched-mass, matched-window cubic budget on a size ladder; the Duhamel integral is the quantity to bound.
4. Charge drift (C2): E(T/r_s=15) under τ=8,4,2 and its scaling with τ; if not ∝τ, an explanation beyond "held step" is needed. Text unchanged until then.
5. Outlook questions (edits 7–9): remain open as printed; the reviewer's derivations confirm the obstructions.

Owner word-rule findings in the reviewed text (not raised by the referee; exact replacements, meaning unchanged):
- sec:star, interior paragraph: "the full normalized error budget not yet certified" → "the full normalized error budget not yet verified".
- sec:singularity: "is what the continuation calculation must certify" → "is what the continuation calculation must verify".
- sec:page-effective: "The certified ceiling of \(5\,256.3\) nats" → "The verified ceiling of \(5\,256.3\) nats".
- App. A.2: "with both enclosures rounded outward" → "with both bounds rounded outward".
- sec:page-effective: "reconciled with the charge ledger at the same weighting" → "reconciled with the charge accounting at the same weighting".
- sec:results-vacuum: "the projector route when a comparator state is at hand and the direct route when only the state's energy is" → "the projector comparison when a comparator state is at hand and the direct comparison when only the state's energy is".
- sec:scaling-far (bare "not established"): "and whether the surface arrives or halts there, and at what density, is not established." → "and whether the surface arrives or halts there, and at what density, is fixed neither by the pressureless dust comparison solution nor by the computed collapse, whose interior radius stays above \(R_\pi\) (Sec.~\ref{sec:star})."
- sec:scaling-far (bare "not established"): "so recession with vanishing exterior work in the astrophysical limit is not established, and" → "so recession with vanishing exterior work, exact on the dust comparison solution, is an assumption for the computed object in the astrophysical limit, and" (the same statement the paper already makes two paragraphs later: "The vacuum shell holds for the dust comparison solution and is an assumption for the object's fountain").
No occurrence of "certificate", "readout", "decider" or "vehicle" in the body text.

## 4. Adversarial self-check

- Item 3 (STYLISTIC-SKIP). Strongest counter: the paper reserves "infinity clock \(t_\infty\)" for the receiver, so an axis saying "time at infinity" names a different clock than the one plotted, and a referee flagged it. This is a fair point, but both labels are correct as rates (N(∞)=1, v(∞)=0), the captions already say flat-slice clocks, and no LaTeX edit fixes a PDF label. Verdict stands: optional relabel if the figures are regenerated; not a required edit.
- Item 1 (modified replacement). Counter: adding "when \(\mathcal M\) is nondecreasing" and keeping "bounded λ" could look like new conditions, and λ is undefined where M′=0. Response: the general criterion (small ∫|(ln N)′|) needs no monotonicity; the specific λ-form needs M′≥0, since otherwise the integral is bounded by the total variation of ln M rather than by the net annulus fraction, so dropping it would reinstate a false sufficiency claim; λ is used inside the paper's own "where M′≠0" clause. Verdict stands.
- Item 4 (NON-ISSUE). Counter: an 8 % charge loss over roughly 120 steps is large, and if it is not a step error the first-law residual "at the level of the held step's charge deficit" and the sec:scaling-far recession discussion inherit an unexplained energy sink. Response: the manuscript nowhere claims the integrator conserves E, labels the drift, and reads formation with the contemporaneous charge; the decisive check (E(15) versus τ) is a repository computation, not a text correction. Verdict stands, with the query recorded as open.