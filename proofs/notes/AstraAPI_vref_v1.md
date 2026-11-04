
# Executable specification: spherical-sector vertex contractions

**Scope.** Post-process the reference collapse at \(T/r_s=11.0\) and \(15.0\), using the stored states and accepted fields. Do not rerun the coupled collapse. Return separate results for each slice, with numerical errors and applicability flags. If either state must be reconstructed, replay only the recorded quadratic propagators.

**Source anchors.** Here \(P_{\rm text}\) denotes `"00_PAPER_flat_3f3f40a2fc3.tex"`. The governing passages are:

- **Ordering:** \(P_{\rm text}\), `sec:model-operators`: “Expand \(E[C]=\mathcal M(\infty)\) … about the admitted Hartree covariance \(\bar C\), retaining the reference's spectral response, and fully Wick order against the continued reference \(P\).”
- **Vertex and perturbative split:** \(P_{\rm text}\), `sec:star`, paragraph beginning “In the reduced quantum form”: “\(V_{\mu\nu}\) the Hessian of \(E[C]\) in the bare moments \(\Tr(A_\mu C)\)” and “\(B=-h_H[\bar\Delta]\)”.
- **Reference:** \(P_{\rm text}\), `eq:dm-reference`: “Levels retain their preparation occupations through zero-energy crossings”; `eq:dm-contact` specifies its derivative.
- **Branch:** \(P_{\rm text}\), `sec:model-operators`: “\(\mathcal B=4\sum_j(\widehat M_j/M_j-I)^2\)” and “\(\lambda_\Pi=\|(I-\Pi)\widehat E_2\Pi\|\)”.
- **Existing target:** \(P_{\rm text}\), `sec:star`: “the reported one-route bound \(\langle\mathcal B\rangle\le0.080\) on all 18 slices on the ten channels”.
- **Independence:** `"01_requirements_new.txt"`, §5: “two independent routes agreeing within their stated errors”.

No implementation functions or numerical data files are supplied here. The interfaces and output names below are requirements for the implementations, not claims about existing code.

## 1. Input contract and hard gates

Implement `evaluate_slice(manifest)`, with the following inputs.

| Input | Required content |
|---|---|
| Identity | History identifier, checksums, record clock, \(a,m,G\), step convention, channel count and boundary conditions. Require \(ma=0.2,\ G/a^2=1/4\), ten open 2048-site chains. |
| State | Full \(\bar C_k\), or an occupied panel \(U_k\) with \(\bar C_k=U_kU_k^\dagger\), **including the propagated sea**, not dust columns alone. |
| Reference | \(P_k=Z_kZ_k^\dagger\), occupation labels and continuation provenance; alternatively preparation plus the actual configuration path and crossing/degeneracy convention needed to reconstruct it. |
| Fields | Dense solved \(N,v,\mathcal M\), stored back-averaged fields, radial grids, quadrature, averaging and angular-projection matrices, boundary/collar charge \(E\). |
| Discretization | Explicit site allocations \(A_\mu\), source normalization, channel-copy convention and the discrete spherical-row residual. |
| Masses | Fixed represented radii and their nearest \(0.1a\) rows: \(R_{50}\), plus existing inner/outer roots. Freeze these radii when differentiating; do not differentiate the rule selecting \(R_{50}\) or a root. |

Use the actual \(\sigma=4a\) Gaussian map of this history, not the compact kernel of `eq:dm-average`, nor a newly chosen one-dimensional Gaussian.

The chain operator is, from \(P_{\rm text}\), “\(h_{0,k}=m\sigma_3+\sigma_1\widehat p_r+\sigma_2k/r_n\)”:
\[
h_k=\tfrac12\{\bar N,h_{0,k}\}+\tfrac12\{\bar v,Q_r\},
\qquad Q_r=\frac{T^2-T^{\dagger2}}{4ia}.
\]
Its dimension is \(d=4096\); it is block-banded with two-site range.

**Normalization gate.** Represent the magnetic/valley multiplicities as genuine identical copies, \(g_k=16k\), without adding another valley factor. Recover
\[
\Tr(\bar C-P)=640
\]
and the stored sources and charge before contracting anything. The paper’s weighted-chain description and the requirement file’s §5 sentence “no degeneracy weight on any source in the radial codes” must be resolved by an explicit export convention: identify where multiplicity has already been incorporated. Never multiply an already aggregated source again.

**Reference gate.** Check \(P^2=P\), \([h,P]=0\), fixed rank and occupation continuation. Endpoint fields alone do not determine \(P\) through ambiguous crossings. Do not substitute \(P_-(h)\), a propagated sea, or a new interpolation path.

**State gate.** Measure Hermiticity, panel orthogonality, \(\|\bar C^2-\bar C\|\), and trace errors. The exclusive single/double-pair formulas below require a Slater projector. Do not silently purify an impure export; bound its effect or evaluate the general Gaussian Wick moments instead.

If no complete state snapshot exists, replay
\[
U_{k,j+1}=e^{-i\tau_jh_{k,j}}U_{k,j}
\]
on the **recorded accepted fields**, with the recorded preparation. This is not a new collapse calculation. Missing reference-path information produces `INPUT_BLOCKED`.

## 2. Independently construct the Hessians

Let
\[
z_\mu=\Tr(A_\mu C),\qquad D=\bar C-P,\qquad d_\mu=\Tr(A_\mu D).
\]
All traces below include the declared copies.

Construct the local admitted map \(u(z)\), \(u=(N,v)\), from the stored discrete rows
\[
R(u,z)=0,
\]
with sources built from \(C-P(u)\). Differentiate this map, including the reference, rather than holding the subtracted sea fixed:
\[
R_u\,\delta u=-R_z\,\delta z,\qquad
\delta h=\sum_{\mu\nu}A_\mu V_{\mu\nu}\delta z_\nu .
\]
Thus \(V_{\mu\nu}=\partial^2E/\partial z_\mu\partial z_\nu\). Check independently that
\[
\delta E=\Tr(h\,\delta C).
\]

For unequal energies,
\[
(\delta P)_{ab}
=\frac{n_a-n_b}{\varepsilon_a-\varepsilon_b}(\delta h)_{ab}.
\]
Equal-energy elements must follow the specified differentiable continuation. A pseudoinverse or arbitrary eigenvector matching is not a definition of that continuation.

For every represented fixed radius, construct
\[
G_j=\nabla_C M_j,\qquad V^{(j)}_{\mu\nu}
=\frac{\partial^2M_j}{\partial z_\mu\partial z_\nu}.
\]
Check the mass derivative against the paper’s window identity:
\[
N(r_j)\delta M_j
=\Tr\!\left[h_{r_j}(\delta C-\delta P)\right].
\]
Here \(h_{r_j}\) uses fields cut off **before back-averaging**, as specified in \(P_{\rm text}\), `sec:model-operators`. Differentiate this identity, including \(N(r_j)\), the windowed fields and reference response, to check the mass Hessian.

Keep all \(\rho\rho,\rho J,J\rho,JJ\) blocks. In particular, retain the mixed element identified in \(P_{\rm text}\) as “of order \(\sqrt G\)”. No \(G^2\)-only truncation is permitted.

## 3. Common contraction definitions

For any symmetric real moment Hessian \(K\), define
\[
\begin{aligned}
H_K(X)&=\sum_{\mu\nu}K_{\mu\nu}A_\mu\Tr(A_\nu X),\\
X_K(X)&=\sum_{\mu\nu}K_{\mu\nu}A_\mu X A_\nu,\\
f_K(X)&=\tfrac12\sum_{\mu\nu}K_{\mu\nu}
 \left[\Tr(A_\mu X)\Tr(A_\nu X)-\Tr(A_\mu X A_\nu X)\right],\\
\Gamma^K_{ab;cd}
&=\sum_{\mu\nu}K_{\mu\nu}
 \left[(A_\mu)_{ac}(A_\nu)_{bd}
       -(A_\mu)_{ad}(A_\nu)_{bc}\right].
\end{aligned}
\]
The quartic convention is
\[
\widehat V_{K,P}
=\frac14\sum_{abcd}\Gamma^K_{ab;cd}
:c_a^\dagger c_b^\dagger c_d c_c:_P .
\]
This fixes all antisymmetrization factors.

Verify on small Fock spaces:

1. \(\langle\widehat V_{K,P}\rangle_C=f_K(C-P)\);
2. its expectation and first covariance derivative vanish at \(C=P\);
3. Hartree and exchange cancel for \(C-P=|u\rangle\langle u|\), \(Pu=0\);
4. explicit Fock-space action agrees with the single/double-pair decomposition.

These are mandatory unit tests, not physical approximations.

## 4. Reference energy and source response

Use exactly
\[
\begin{aligned}
\widehat E_2&=\mathcal E_H+
:c^\dagger(h-H_V(D))c:_P+\widehat V_{V,P},\\
\mathcal E_H&=E-\Tr(hD)+\tfrac12d^{\mathsf T}Vd,\\
B&=-H_V(D).
\end{aligned}
\]
Diagonalize the held \(h\). Indices \(i,j\) denote reference-occupied levels; \(\alpha,\beta\) denote reference-empty levels. Set
\[
d_{\alpha i}=\varepsilon_\alpha-\varepsilon_i,\qquad
d_{\alpha\beta ij}=\varepsilon_\alpha+\varepsilon_\beta-\varepsilon_i-\varepsilon_j.
\]
Compute
\[
\boxed{
F\equiv\Delta E_{\rm ref}^{(2)}
=-\sum_{\alpha i}\frac{|B_{\alpha i}|^2}{d_{\alpha i}}
-\sum_{\alpha<\beta,\ i<j}
 \frac{|\Gamma^V_{\alpha\beta;ij}|^2}{d_{\alpha\beta ij}} .
}
\]
Return both terms, their sum and \(F/E\), excluding \(\mathcal E_H\). Also return
\[
\mathcal Z_2=
\sum_{\alpha i}\frac{|B_{\alpha i}|^2}{d_{\alpha i}^2}
+\sum_{\alpha<\beta,\ i<j}
 \frac{|\Gamma^V_{\alpha\beta;ij}|^2}{d_{\alpha\beta ij}^2},
\]
the first-order reference-state dressing norm.

**Denominator gate.** The paper states this energy result “on a gapped path with \(P\) the ground projector”. Measure \(P-P_-(h)\), pathwise crossing information, and the smallest relevant single and double denominators. On that branch \(F\le0\). If occupations have crossed, retain them: signed nonresonant sums may be reported as formal continued-eigenstate contractions, but they do not pass the paper’s gapped-ground-reference control. A coupled zero denominator requires degenerate perturbation theory; do not regularize it into a claimed answer.

### Source derivatives

At fixed \(\bar C\), fixed kernel and fixed preparation/continuation convention, differentiate \(F\) with respect to the coupled fields. Treat the Hessian as the local response Hessian constructed from the rows at the trial fields, not as a frozen numerical matrix. Include \(\delta V,\delta P,\delta D\), eigenvector response and denominators:
\[
\delta F
=-2\Re\sum_q\frac{t_q^*\delta t_q}{d_q}
+\sum_q\frac{|t_q|^2\delta d_q}{d_q^2},
\]
where \(q\) runs over both sectors. In particular,
\[
\delta B=-H_{\delta V}(D)+H_V(\delta P).
\]
This requires differentiating the row response once further. Compare the resulting derivative with centered finite differences on the same reference branch.

Define site-integrated contributions by
\[
q_{\rho,x}=\frac{\partial F}{\partial N_x},\qquad
q_{J,x}=\frac{\partial F}{\partial v_x}.
\]
On cubic sites, \(\delta\rho_x=q_{\rho,x}/a^3\) and similarly for \(J\); on chains use the exported shell-volume/angular map, not \(a^3\) by analogy. Apply the actual averaging adjoint relationship:
\[
\delta F=\int
(\delta\rho_\ell\,\delta N+\delta J_\ell\,\delta v)\,d^3y .
\]

Insert these extra sources into the **linearized** coupled rows, including \(P\)'s response, to obtain \(\delta N,\delta v,\delta M\). Report source profiles and field shifts. At each simple root,
\[
\delta R_t=-\frac{\delta N+\delta v}{\partial_r(N+v)}.
\]
A small \(F/E\) alone is not evidence that fields are unchanged.

## 5. Mass bias, fully reordered norms and branch weight

Reorder the full second-order mass operator about the actual Hartree state, not about dust added to \(P\). For \(K=V^{(j)}\), define
\[
\beta_j=-\tfrac12\sum_{\mu\nu}K_{\mu\nu}
                 \Tr(A_\mu D A_\nu D),\qquad
L_j=G_j-X_K(D).
\]
Then
\[
\widehat M_j-M_j
=\beta_j I+:c^\dagger L_jc:_{\bar C}
+\widehat V_{K,\bar C}.
\]
Consequently, for the pure Hartree state,
\[
\begin{aligned}
\langle\widehat M_j\rangle-M_j&=\beta_j,\\
S_{1,j}&=\Tr[\bar C L_j(I-\bar C)L_j],\\
S_{2,j}&=\sum_{a<b,\ i<l}
 |\Gamma^{K,\bar C}_{ab;il}|^2,\\
\boxed{\langle\mathcal B\rangle
&=4\sum_j\frac{\beta_j^2+S_{1,j}+S_{2,j}}{M_j^2}.}
\end{aligned}
\]
Here occupied/empty indices in \(S_2\) refer to \(\bar C\), not \(P\). Return \(\beta_j/M_j\), \(\sqrt{S_{1,j}}/M_j\), \(\sqrt{S_{2,j}}/M_j\) separately. The bias squared must not be omitted.

Apply the same reordering to \(\widehat E_2\). Its full single-pair matrix is \(h-X_V(D)\). Relative to the Hartree tangent Hamiltonian, the residual single-pair matrix is instead \(-X_V(D)\); label these two norms distinctly.

**Hole accounting.**
\[
H=\Tr[P(I-\bar C)],\qquad
N_{\rm p}=\Tr[(I-P)\bar C]=640+H,
\]
and verify independently
\[
\boxed{\Tr(\bar C-P)^2=640+2H.}
\]
For a nonprojector covariance the right side has the additional term
\(-\Tr[\bar C(I-\bar C)]\). Report that impurity term rather than attributing it to holes. No norm may replace \(D\) by the 640 dust orbitals.

With an upper error bound \(b_+\) on \(\langle\mathcal B\rangle\),
\[
\langle\Pi\rangle\ge\max(0,1-b_+),\qquad
\Pr(\widehat M_j\le0)\le b_+/4 .
\]
Confirming \(b_+\le0.080\) gives the stated \(0.92\) branch-weight and \(0.020\) individual-mass bounds **on these slices only**. Confirmation of “all 18 slices” requires processing those 18 inputs.

### Leakage is a separate output

Single/double-pair norms are not
\(\lambda_\Pi=\|(I-\Pi)\widehat E_2\Pi\|\). Do not relabel them as leakage.

A cheap, rigorous but potentially loose certificate is allowed. Rewrite
\[
\widehat E_2=cI+c^\dagger t c+\widehat V_{V,\mathrm{bare}},
\quad
t=h-H_V(D)-[H_V(P)-X_V(P)].
\]
Then
\[
\lambda_\Pi\le L_{\rm global}
=\|t\|_1+\frac12\sum_{\mu\nu}|V_{\mu\nu}|
 \left(\|A_\mu\|_1\|A_\nu\|_1+\|A_\mu A_\nu\|_1\right).
\]
Evaluate or upper-bound these trace norms with the full copy multiplicity.

If this certificate is too large, return `LEAKAGE_UNRESOLVED`. Closing it requires a tighter certified bound on the specified compressed operator, or its action/singular norm on a demonstrably sufficient branch subspace. A finite excitation-space calculation without a complement bound is not an operator-norm certificate. Test \(\tau L\) and \(64a\,L\); these are fixed-slice controls, not bounds for a changing projector throughout collapse.

## 6. GPU algorithm and cost

Do not construct the replicated one-particle matrix or a four-index vertex.

1. **Channel parallelism.** Process native channels independently for diagonalization, occupied panels and one-body responses. Reduce moments with the declared copy counts.
2. **Response solve.** Factor or iteratively solve \(R_u\), reusing it for Hessian actions and source derivatives. Measure its conditioning.
3. **Signed factorization.** Write
   \[
   K_{\mu\nu}=\sum_{s=1}^{R}\eta_sL_{\mu s}L_{\nu s},
   \qquad F_s=\sum_\mu L_{\mu s}A_\mu .
   \]
   Keep negative eigenvalues. Each \(F_s\) remains banded.
4. **Pair contractions.** For occupied/empty panels of either \(P\) or \(\bar C\), form
   \(X_{ks}=U_{E,k}^\dagger F_{ks}U_{O,k}\). The exact double-pair norm is
   \[
   S_2=\frac12\Re\sum_{st}\eta_s\eta_t
   \left[
   \left(\sum_k g_k\Tr X_{ks}^\dagger X_{kt}\right)^2
   -\sum_k g_k\Tr\!\left((X_{ks}^\dagger X_{kt})^2\right)
   \right].
   \]
   This handles distinct-copy pairs and same-copy exchange correctly. Multiplying every pair term by either \(g_k\) or \(g_k^2\) does not.
5. **Positive-denominator energy.** Use
   \[
   X_s(t)=e^{-t h_E/2}X_s e^{t h_O/2},
   \qquad
   F=-\int_0^\infty[S_B(t)+S_2(t)]\,dt .
   \]
   The same integral weighted by \(t\) gives \(\mathcal Z_2\). Shift energies by a chemical potential inside the verified occupied/empty gap for stable exponentials. Signed denominators require spectral sums or an independently controlled resolvent, not this Laplace integral.

Diagonalization costs \(O(10d^3)\). Banded application to an occupied panel costs \(O(bdo)\), with fixed bandwidth \(b\). Dense transformed panels cost \(O(Rdeo)\) per channel. The deterministic Gram route has worst-case cost
\[
O\!\left(QR^2\sum_k e_k o_k^2\right)
\]
for \(Q\) quadrature nodes, with tiled device storage rather than \(O(Rd^2)\) resident storage. One native complex-double occupied panel is about 134 MB at half filling.

**Measure effective rank \(R\), quadrature count and actual GPU time.** The supplied material establishes no low-rank Hessian or inexpensive full-rank contraction. Rank truncation is permitted only with propagated residual bounds; randomized contractions require confidence bounds and independent seeds.

## 7. Error controls and decision rules

Each output must carry numerical and applicability errors separately.

- Check row residuals, \(V=V^{\mathsf T}\), mass-Hessian symmetry, projector identities and independent directional derivatives.
- Resolve small denominators explicitly; report their contribution versus denominator cutoff.
- Bound Laplace tails using the measured minimum denominators. For example,
  \[
  \text{tail}\le
  S_1(0)e^{-d_1t_{\max}}/d_1+
  S_2(0)e^{-d_2t_{\max}}/d_2 .
  \]
- Control cancellation in signed-factor contractions with double precision, compensated reductions and selected higher-precision checks.
- Propagate discarded Hessian factors into \(B,\Gamma,F\), their derivatives and mass moments. Convergence of \(F\) does not certify its derivative.
- Measure endpoint sensitivity and channel contributions on the fixed data. Ten-channel results are ten-channel results, not a cutoff limit.
- Attach the stored hydrodynamic/certification flags. \(P_{\rm text}\) states that the one-term test first fails at \(T/r_s=11.75\); a successful quantum test at \(15.0\) does not restore that hydrodynamic premise.

For a concrete screening run, preregister **1%** as the small-correction threshold; this is an execution criterion, not a bound supplied by the paper.

| Reading | Controlled Hartree-level result | Failure or unresolved result |
|---|---|---|
| Reference correlation | \(|F|/E<0.01\), \(\mathcal Z_2<0.01\), errors well below thresholds | Order-one correction/dressing fails; resonant or unsupported branch is unresolved |
| Sources/fields | Induced relative clock, flow, mass and simple-root shifts below 1%; absolute profiles reported near zeros | Large field/root changes fail even if integrated \(F\) is small |
| Mass bias | \(|\beta_j|/M_j<0.01\), also small against measured mass fluctuation and the branch margin \(M_j/2\) | Bias or reordered norms comparable to \(M_j\) fail |
| Branch | Certified \(\langle\mathcal B\rangle<1\); \(0.080\) is the stronger reproduction target | A bound above one is inconclusive, not proof of an empty branch |
| Leakage | Certified \(64a\,\lambda_\Pi<0.01\) for the held slice | A loose upper bound is unresolved; measured large leakage fails |

Report the signed exchange separately. The paper’s “0.14 to 0.16 percent” is relative to **pair energy**, not automatically to \(E\), and is not a substitute for this correlation calculation.

## 8. Independent implementations and deliverable

Both teams must independently implement:

- operator allocations, normalization and reference reconstruction/response;
- boundary and mass Hessians;
- Wick reordering, antisymmetrization, hole accounting and all branch terms;
- reference energy, source derivatives and induced field response;
- error estimates and final classifications.

Use distinct derivative routes: one differentiates the discrete admitted solve; the other derives tangent/adjoint row equations and spectral response. Validate contractions by explicit small-Fock-space action versus independent Wick/Slater algebra. Full-size contraction implementations may share the mathematical factorization, but not computed Hessians, pair matrices, spectra of reduced vertices or final scalar exports as their independent input.

Share only immutable raw data, the normalization manifest, conventions and agreed test cases. Reconcile intermediate residuals and channel-resolved outputs before comparing totals.

Write `vertex_control.json` plus radial profile arrays containing every quantity above, checksums, thresholds, error bounds and one of `PASS`, `FAIL`, `UNRESOLVED`, or `INPUT_BLOCKED` for each test. No numerical value can be assigned from the supplied text alone.

=== SIGNED ===