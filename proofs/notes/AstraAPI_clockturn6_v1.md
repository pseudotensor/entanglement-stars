
reviewer A—

### 1. Main claim

**AGREE that no floor-free beyond-window bound is established; DISAGREE if this is presented as an impossibility theorem.** `01_FableSub_clockturn5_v1.md`, A.0, says “no floor-free a-priori bound,” but its closing open point still asks “whether the self-consistent evolution can place an (S)-admissible drive” on a deep slice. That realization gap also prevents calling the energy–current inequality the **only possible** mechanism.

“The rate has no inertia” correctly rejects extrapolating the sampled envelope. Propositions 1–2 offer **conditional short continuation**, not completed extensions: measure the endpoint-neighbourhood \(X_\rho\) and regularity/invertibility margins; for Proposition 2 also establish strict \(\delta>0\), \(R_\Delta\), and window-uniform tail bounds. A.3’s peak-record estimates do not measure these quantities at the final endpoint or certify its \(O(1)\) duration.

### 2. Counterexample

**Valid against a universal \(c'=0\) bound; incomplete against the full stated inequality.** `01_FableSub_clockturn5_v1.md`, A.2, gives “\(C^2=C\)” and interference coefficients “\(\varepsilon(p)-\varepsilon(p')\)” versus “\(q(p)+q(p')\).” Negative local energy with nonzero current defeats \(|J_\ell|\le c\rho_\ell\). It does **not** defeat the \(c'k^2\) version without constructing the admitted gravitational slice and specifying uniformly bounded constants. Moreover, at fixed momenta both interference terms are first order in \(\alpha\), so the claimed generic \(1/\alpha\) ratio divergence does not follow.

**No agreed conditional theorem changes.** `02_route2_answers.txt` now reports “residual <=2e-21” using “c=0.25 with c'>=0.0156” on nonpositive nodes: this repairs the reported \(c'=0\) inconsistency, but supplies neither an independently certified inequality nor Proposition 2’s strict negative margin.

### 3. Frozen versus level-following

**Not resolved as a frozen total reference.** `02_route2_answers.txt` says both “the reference operator is T_W” and “zb is initialized to 0 and never updated.” The latter freezes/disables the projector channel, not necessarily the polarization subtraction. Thus \(\kappa_{\rm held}=1\) applies to the \(T=0\) comparison; Proposition 2 still requires \(\dot r_{\rm total}\equiv0\).

**Check:** audit every time-step/stage/reset update of the total subtracted reference, establishing whether its derivative is zero or `m3r2_w2adapter_v2.dW2`, and whether that implemented tensor equals the paper’s tensor.

### 4. Final paper paragraphs

Numerical provenance: `01_FableSub_clockturn5_v1.md`, A.5, reports “clock minimum is 0.068”; `00_AstraAPI_clockres_v1.md`, §1, quotes “min N 0.1094,” “min N 0.0716,” and “no stored rates, so the FD rate.” `02_route2_answers.txt` supplies “ON a_up 0.0282 (T_W)” and limits the envelope to “the stored records only.” Its absolute rate check remains “NOT certified.”

**Insert now:**

Under the quantum-singularity definition adopted here, no dynamical singularity is observed within the computed time, and there is no clock collapse at any stored record. Sampled minima are \(0.068\) in the reference collapse, \(0.1094\) in one of two paired histories, and \(0.0716\) in a third diagnostic history. For both 159-record paired histories, the implemented polarization-response tangent gives a sampled upper envelope \(D^+L\le0.0282\) per lattice time; the third has only finite-difference rates. These are sampled diagnostics: identification of the polarization tensor with Eq. (dm-reference), independent agreement and between-record validation remain incomplete. The measured rate supplies no extrapolation bound. Sea–band coherence invalidates \(|J_\ell|\le c\rho_\ell\) as a universal source bound; neither a preserved energy–current barrier nor a floor-free sensitivity bound is established. These observations therefore do not certify regular continuation between records or exclude later singularities.

**After two-route agreement and a validated interval gate, using its actual certified constants:**

Two independent evaluations using the reference response actually integrated agree within declared error budgets. A validated interval gate encloses the same-branch trajectory, including integration error, with \(N_{\min}\ge m_{\rm cert}>0\) and \(D^+L\le a_{\rm cert}\) throughout the covered intervals. Clock collapse is therefore excluded on those computed lattice branches up to their endpoints. This finite-window result neither establishes equivalence to Eq. (dm-reference) without its tensor audit nor excludes later singularities.

AGREED on the qualified conclusions and paper wording; STILL OPEN: reference-law audit, interval certification and beyond-window evolution.