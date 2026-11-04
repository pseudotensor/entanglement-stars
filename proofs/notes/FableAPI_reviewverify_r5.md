# Independent check of review round 5 (manuscript a36d08149e4)

Preliminary: none of the previous round's corrections is re-reported as an error. The B.1 table re-checks several of them (projector entropy bound; nonvacuum-shell slope counterexample; crossing-ray dispersion; exterior averaging; thermal-period arithmetic) and passes them. No ALREADY-FIXED verdict is therefore needed on the issues list; the only items that touch prior fixes are confirmations.

## 1. Verdict table

| # | Item | File / section | Claim in brief | Verdict | Class | Reason |
|---|---|---|---|---|---|---|
| R1 (P1 = Edit 2) | 03_model, eq:dm-reference paragraph | bulk tensor $\mathsf T$ not displayed; reviewer derives it | VALID-APPLY-MODIFIED | FIX (derivation) | Interband vertex, $2N\varepsilon$ denominator, four-component trace, $-\tfrac14$ normalization, variational densities all rederive; insert condensed boxed form (page budget) and confirm normalization against the repository's $c_\rho$ before commit |
| R2 (P2 = Edit 1) | 04_results, sec:results-firstlaw, after eq:page-share | "two independent readings, not fitted by one" | VALID-APPLY-MODIFIED | FIX | $dS_{\rm cut}/dE=f/T_H+r'$, so $T_{\rm cut}=T_H$ at leading order iff $f=1$: same condition; statement is false; use shorter replacement |
| R3 (P3 = Edit 5) | figures/star_structure.tex | label $r_s=2GE=64a$ | VALID-APPLY | FIX | $2GE_0=59.07a\neq64a$; the figure's own linear scale already places $2GE$ at $0.92\,r_s$; only the label is wrong |
| R4 (P4a = Edit 6) | 05_star, trapped interior | bin above $m$ called "the fountaining class" | VALID-APPLY-MODIFIED | FIX | fountaining was defined as $0<E<m$; name the interval $(m,b_r)$ rather than a new phrase |
| R5 (P4b = Edit 7) | 05_star, trapped interior | "well above the exclusion ceiling" | VALID-APPLY | FIX | density is 3.2 percent of the ceiling: below, not above |
| R6 (Edit 3) | 05_star, sec:singularity | explicit finite-interval continuation bound | VALID-APPLY-MODIFIED | FIX (derivation) | Neumann, Hilbert–Schmidt and Picard bounds correct; manuscript sentence already correct; add one condensed sentence if pages allow, else supplement |
| R7 (Edit 4) | sec:singularity | between-slice bound uncertified | NON-ISSUE (text) | OPEN | manuscript states exactly this; task goes to the continuation calculation |
| R8 (Edit 8) | Intro, appendices | proofs not in submitted text | NON-ISSUE (text) | OPEN | requirements already make the full proofs a held supplement; supply it |
| R9 (Edit 9) | 08_discussion Outlook; sec:model-gravity step 4 | tensor sector and next-gradient freedom | NON-ISSUE | OPEN | both already stated as open; reviewer's TT and $a^2\lvert\nabla\times\nabla\times v\rvert^2$ checks agree with text |
| R10 (Edit 10) | Outlook; sec:page-effective | state count, supply, copy obstruction | NON-ISSUE | OPEN | already open in text; copy-twirl invariance already printed |
| R11 (B.3 last paragraph, Val. 14) | 03_model, sec:model-dictionary | "no finite-range improvement of $Q_i$ compatible with the sea" | VALID-APPLY-MODIFIED | CLARIFICATION | scalar trigonometric polynomials commute with $P_0$, so the natural reading is false; print what fixes $Q_i$ and point to the theorem's admissible class (open) |
| R12 (E, presentation) | 08_discussion | "Nor is it a toy model" | VALID-APPLY-MODIFIED | CLARIFICATION | owner rules forbid the phrase; delete the sentence, keep the content sentence |
| R13 (D census conclusion) | abstract/introduction | "one principle" is a package | NON-ISSUE | CLARIFICATION (none needed) | sec:model-dictionary already states "Its content is four requirements" |
| R14 (Val. 15) | bibliography | entries absent from package | NON-ISSUE | OPEN (package) | review-package omission, not a manuscript error |
| R15 (B.3 bullets, Val. 13, 16) | repository numerics | unverified, not errors | NON-ISSUE | OPEN (provenance) | no text change; belongs to the supplement index |
| R16 (Val. 1–6) | withdrawn findings | sign, lapse, circularity, Page, singularity, Lipschitz | NON-ISSUE | n/a | withdrawals are correct; none re-opens a prior-round fix |

## 2. Verification, item by item

### R1 (P1 / Edit 2): the bulk response tensor

**1. Passage** (sec:model-sources): "In the bulk \(W_2=-\tfrac14\int\mathsf F:\mathsf T:\mathsf F/N\) with \(\mathsf F_{ij}=\partial_iv_j-2\epsilon_{ijk}b_k\) and \(\mathsf T\) a momentum-space response tensor (its form and the directional coefficients \(c_\rho=0.438\), 0.420, 0.413 per \(a\) along [100], [110], [111] at \(ma=0.2\) in Ref.~\cite{McKinneyCode2026}), a tensor response, not a renormalization of \(G\)." Quoted correctly.

**2. Mathematics.** I rederived the reviewer's tensor from the manuscript's own definition of $W_2$ (second-gradient term of the prepared-band vacuum energy of the frozen symbol $h=Nh_0+v\cdot q+b\cdot\Sigma^{\rm lat}$, midpoint assignment).

- Projector overlap: with $\partial_{p_i}\varepsilon=q_i/\varepsilon$ and $\partial_{p_i}P_+=\tfrac12(D_i/\varepsilon-h_0q_i/\varepsilon^3)$, $D_i=\partial_{p_i}h_0=\alpha_ic_i$, one gets $P_+(p+\xi/2)P_-(p-\xi/2)=\xi_iP_+\partial_iP_+P_-+O(\xi^2)=(\xi_i/2\varepsilon)P_+D_iP_-+O(\xi^2)$ (the $h_0q_i$ term dies on $P_+h_0P_-=0$). Multiplying by the scalar flow vertex $v_jq_j(p)$ gives the reviewer's flow element.
- Spin vertex: $\{h_0,\alpha_i\}=2d_i$ gives $[h_0,\alpha_1\alpha_2]=2(d_1\alpha_2-d_2\alpha_1)$, and $P_+[h_0,X]P_-=2\varepsilon P_+XP_-$, hence $P_+\Sigma_3^{\rm lat}P_-=(i/\varepsilon)c_1c_2P_+(d_2\alpha_1-d_1\alpha_2)P_-=(i/\varepsilon)\epsilon_{ij3}q_jP_+D_iP_-$. Correct.
- Combined vertex: $-(i/2\varepsilon)\mathsf F_{ij}q_jP_+D_iP_-$ with $\mathsf F_{ij}=i\xi_iv_j-2\epsilon_{ijk}b_k$ reproduces both pieces, and the completed rigid turn ($\partial_iv_j=\epsilon_{ijl}\omega_l$, $2\epsilon_{ijk}b_k=\epsilon_{ijk}\omega_k$) gives $\mathsf F=0$, as the manuscript states.
- Clock vertex: the lattice anticommutator $\tfrac12\{N,h_0\}$ has element $\tfrac12N(\xi)[h_0(p+\xi/2)+h_0(p-\xi/2)]$; projecting gives $\tfrac12[\varepsilon(p+\xi/2)-\varepsilon(p-\xi/2)]\,P_+(p+\xi/2)P_-(p-\xi/2)=O(\xi)\cdot O(\xi)$. No first-gradient clock vertex; its second-order energy is fourth order, and cross terms with the flow vertex are odd under $\Theta$. Correct.
- Trace: $\tr[(1-h_0/\varepsilon)D_i(1+h_0/\varepsilon)D_k]=\tr(D_iD_k)-\tr(h_0D_ih_0D_k)/\varepsilon^2$ (the odd-$\alpha$ traces vanish); with $h_0\alpha_ih_0=2d_ih_0-\varepsilon^2\alpha_i$, $\tr(h_0D_ih_0D_k)=c_ic_k(8d_id_k-4\varepsilon^2\delta_{ik})$, so $\tr(P_-D_iP_+D_k)=2(\delta_{ik}c_ic_k-q_iq_k/\varepsilon^2)$. Correct.
- Second-order energy of the filled lower band, denominator $2N\varepsilon$ (the $O(\xi)$ correction from a uniform flow background is odd and drops), Parseval over $\xi$: $W_2=-\tfrac14\int d^3x\,\mathsf F_{ij}\mathsf T_{ij,kl}\mathsf F_{kl}/N$ with $\mathsf T_{ij,kl}=\int_{\rm BZ}\tfrac{d^3p}{(2\pi)^3}\tfrac{q_jq_l}{\varepsilon^3}(\delta_{ik}c_ic_k-\tfrac{q_iq_k}{\varepsilon^2})$. The $-\tfrac14$ matches the displayed prefactor; the form is nonnegative (it is $\sum\tr V^\dagger V$), symmetric under $(ij)\leftrightarrow(kl)$, dimension $a^{-2}$, degree one in $u$.
- Completeness of the bulk term: the midpoint symbol's second-order corrections ($-\tfrac18\partial_i\partial_jN\,\partial_{p_i}\partial_{p_j}h_0$ and the $v,b$ analogues) contribute $-\tfrac{a^2}{4}\nabla^2N\int_p d_i^2/\varepsilon$ and zone integrals of total $p$-derivatives: divergences or zero in the bulk, so they change no bulk law (they do matter for chain endpoints, which the reviewer rightly excludes). Zeroth-order eigenvectors are $x$-independent, so no intraband Berry-type term enters at this order.
- Variations with $N,v,b$ independent: $r_N^{(2)}=\mathsf F_{ij}\mathsf B_{ij}/(4N^2)$, $r_{v_j}^{(2)}=\tfrac12\partial_i(\mathsf B_{ij}/N)$, $r_{b_k}^{(2)}=\epsilon_{ijk}\mathsf B_{ij}/N$ with $\mathsf B=\mathsf T\mathsf F$: correct (the factor 2 uses the symmetry of $\mathsf T$).

Script: verify_04.py tests the trace identity (first loop) and so supports the tensor's matrix structure. Its second loop checks $\{h_0,\partial_{p_i}P_+\}=0$, a general projector identity; it does not test the $O(\xi^2)$ clock-vertex claim, which I verified analytically above.

**3. Real problem?** Yes: an establishment/presentation gap, not an error. The reviewer is right that the sources, Jacobian and Hessian all rest on $\mathsf T$, and the reader cannot reconstruct it from the text.

**4. Replacement correct and compatible?** The derivation is correct and uses only objects the paper defines. Two cautions: (i) the formula follows from eq:dm-reference's definition; whether the repository's implemented $\mathsf T$ carries this normalization is a check the paper lane must do (the quoted $c_\rho$ values are the natural test; note $c_\rho$ is quoted "per $a$" while $\mathsf T$ has dimension $a^{-2}$, so $c_\rho$ is some derived coefficient the text never defines); (ii) the reviewer's full text is far too long for a 29-page paper and uses $i$ both as index and imaginary unit. Other sites carrying the same claim: sec:model-gravity ("the lattice reproduces $W_2$ ... through second order, its own corrections entering only at fourth order") is consistent with the divergence argument above; the rigid-turn invariance sentence is consistent.

**5. Scope.** The insertion adds no qualifier and drops none.

**6. No bridge fabricated**: the tensor is the evaluation of the paper's own definition; the open residual is the implementation match.

**Recommended replacement** (for the sentence quoted in step 1):

```latex
In the bulk the expansion is second-order perturbation theory in the
gradients' interband vertex. With \(D_i=\partial_{p_i}h_0\),
\(P_\pm=\tfrac12(I\pm h_0/\varepsilon)\), \(c_i=\cos(ap_i)\) and
\(\mathsf F_{ij}=\partial_iv_j-2\epsilon_{ijk}b_k\), a flow or spin
component of transfer \(\xi\) couples the two bands through
\((\mathsf F_{ij}q_j/2\varepsilon)\,P_+D_iP_-\) up to a phase, the
clock term has no first-gradient interband element, and the prepared
band lies \(2N\varepsilon\) below the upper one, so
\begin{equation}
W_2=-\frac14\!\int\! d^3x\,\frac{\mathsf F_{ij}\mathsf T_{ij,kl}\mathsf F_{kl}}{N},
\quad
\mathsf T_{ij,kl}=\int_{\rm BZ}\!\frac{d^3p}{(2\pi)^3}\,
\frac{q_jq_l}{\varepsilon^3}\Bigl(\delta_{ik}c_ic_k-\frac{q_iq_k}{\varepsilon^2}\Bigr),
\label{eq:dm-response}
\end{equation}
the four-component trace \(\tr(P_-D_iP_+D_k)=2(\delta_{ik}c_ic_k-q_iq_k/\varepsilon^2)\)
already taken and no valley factor to add: a nonnegative quadratic
form of dimension \(a^{-2}\), a tensor response and not a
renormalization of \(G\) (directional coefficients \(c_\rho=0.438\),
0.420, 0.413 per \(a\) along [100], [110], [111] at \(ma=0.2\) in
Ref.~\cite{McKinneyCode2026}). With \(\mathsf B_{ij}=\mathsf T_{ij,kl}\mathsf F_{kl}\)
its bulk variations at independent \(N,v,b\) are
\(r^{(2)}_N=\mathsf F_{ij}\mathsf B_{ij}/(4N^2)\),
\(r^{(2)}_{v_j}=\tfrac12\partial_i(\mathsf B_{ij}/N)\) and
\(r^{(2)}_{b_k}=\epsilon_{ijk}\mathsf B_{ij}/N\).
```

Condition for applying: the paper lane reproduces at least one of the quoted $c_\rho$ from this $\mathsf T$ (or confirms the repository's $\mathsf T$ equals it); if the implemented normalization differs, the formula must be adjusted to the implemented one, not the reverse.

### R2 (P2 / Edit 1): the two spacing readings

**1. Passage** (sec:results-firstlaw): "Second, with controlled subleading entropy derivatives at fixed physical \(m,a,G\), the cut's statistical temperature \(T_{\rm cut}=(dS_{\rm cut}/dE)^{-1}=(T_H/f)[1+o(1)]\), a derivative along the family of cut states, selects the same spacing as the entropy match at leading order in the one-field model, so the spacing is selected by two independent readings, not fitted by one." Quoted correctly.

**2. Mathematics.** $A/4G=4\pi G E^2$, so $d(A/4G)/dE=8\pi GE=1/T_H$. With $S_{\rm cut}=fA/(4G)+r_{\rm cut}(E)$, $1/T_{\rm cut}=f/T_H+r_{\rm cut}'$ and $T_{\rm cut}/T_H=[f+T_Hr'_{\rm cut}]^{-1}$. Under the stated derivative control $T_Hr'_{\rm cut}=o(1)$, "$T_{\rm cut}=T_H$ at leading order" is the condition $f=1$, which is the entropy match. The temperature reading is the entropy reading differentiated with the first law: a consistency identity, not an independent determination. verify_06.py's first assertion checks exactly $dS_{\rm cut}/dE=(4\eta G/a^2)/T_H$ at leading order, so it does test the dependence claim.

**3. Real problem.** Yes; the sentence asserts independence that the derivation contradicts. (It does not affect eq:page-share, the selected value, or the law-selection chain.)

**4. Replacement.** The reviewer's is correct but introduces $S_{\rm BH}$ and three displays. Other sites: sec:page's two-field paragraph says only "select the same spacing at leading order" (fine); Discussion and Abstract make no independence claim; the requirements file itself says "select the same spacing at leading order". No other site changes.

**5. Scope.** The premise "controlled subleading entropy derivatives" is unchanged. A false statement corrected: FIX, not hedge.

**Recommended replacement:**

```latex
Second, with controlled subleading entropy derivatives at fixed
physical \(m,a,G\), the cut's statistical temperature
\(T_{\rm cut}=(dS_{\rm cut}/dE)^{-1}\), a derivative along the family
of cut states, is \((T_H/f)[1+o(1)]\): writing
\(S_{\rm cut}=fA/(4G)+r_{\rm cut}(E)\) and using
\(d(A/4G)/dE=1/T_H\), the first law's own identity, gives
\(T_{\rm cut}/T_H=[f+T_Hr_{\rm cut}'(E)]^{-1}\), and the derivative
control is \(T_Hr_{\rm cut}'=o(1)\). The statistical temperature
therefore equals \(T_H\) at leading order exactly when \(f=1\), the
entropy match's own condition: the two readings are one condition,
the area law differentiated with the first law, the second a
consistency check of the first and not a second determination of the
spacing.
```

### R3 (P3 / Edit 5): structure-diagram label

**1. Passage** (figures/star_structure.tex, in 05_star): `\node at (0,-0.60) {linear to $r_s$, log beyond, $r_s=2GE=64a$};`. Body: "nominal \(r_s=64a\), \(N_B=640\), \(2GE_0=59.07a\)"; "\(r_s=2GmN_B\)" at the nominal; Fig. star-collapse caption "Radii and flat-slice clocks use nominal \(r_s=64a\)".

**2. Mathematics.** $2GE_0=0.923\times64a=59.07a$ (the stated $\eta_E=0.923$). The figure's own coordinates confirm the distinction: the $2GE$ ring is drawn at $2.77$ and $R_t$ at $2.63$; with the nominal $r_s$ at $3.0$ on the linear scale, $2.77/3.0=0.923=\eta_E$ and $2.63/3.0=0.877=0.95\times0.923$, exactly the text's "$R_t$ is 0.95 of $2GE$". The drawing uses the nominal scale and places $2GE$ inside it; the label equating $r_s$, $2GE$ and $64a$ contradicts the drawing.

**3. Real problem.** Yes, a normalization error in one label; no data are implicated.

**4. Replacement.** `nominal $r_s=2GmN_B=64a$` matches the collapse section's convention. Incidental observation (not raised by the reviewer): the cold formation control uses "constant charge of radius \(2GE=59.05a\)" while the reference run has \(2GE_0=59.07a\); the paper lane should confirm the two numbers are meant to differ.

**5. Scope.** None affected.

**Recommended replacement:**

```latex
\node at (0,-0.60) {linear to $r_s$, log beyond, nominal $r_s=2GmN_B=64a$};
```

### R4 (P4a / Edit 6): the fountaining label

**1. Passages** (05_star, trapped interior). Definition: "above \(b_r\) and between 0 and \(m\) content is bound, the latter fountaining (outward-thrown matter that falls back). Below 0 it turns inside the trapping radius, and propagation in \((m,b_r)\) is necessary, not sufficient, for reaching infinity." Later: "The Killing burst bin above \(m\), the fountaining class of the split above, gains a few percent of the particles over 16 to 20, at energies just above \(m\)". Quoted correctly.

**2. Check.** The definition assigns "fountaining" to $0<E_{\rm Killing}<m$. A bin "above $m$" lies in $(m,b_r)$, the propagating class. The later label is wrong; the number and the bin are untouched.

**3.** A real local error (a false label).

**4. Replacement.** The reviewer's wording is correct but introduces "propagation-allowed interval", a new phrase; naming the interval $(m,b_r)$ already defined is tighter. "Killing burst bin" is itself undefined at first use; the replacement should make its meaning clear.

**5. Scope.** Unchanged.

**Recommended replacement:**

```latex
The Killing burst bin, the particles just above \(m\) in the
propagating interval \((m,b_r)\) of the split above, gains a few
percent of the particles over 16 to 20 (values in
Ref.~\cite{McKinneyCode2026}).
```

### R5 (P4b / Edit 7): the exclusion ceiling

**1. Passage:** "The interior stays well above the exclusion ceiling and the comparison radius." Followed by "\(R_{50}\) falls to \(19.1a\) ... a minimum 1.8 times the comparison radius \(R_\pi\)" and "The largest central density on any stored slice is 3.2 percent of the exclusion density \(2a^{-3}\)".

**2. Check.** Density $0.032\times2a^{-3}$ is below the ceiling; radius $1.8R_\pi$ is above the comparison radius. The sentence applies "above" to both. False as written for the density.

**3.** Real local error. Discussion ("central density stays at a few percent of the exclusion density") is consistent with the corrected reading.

**4. Replacement.** Reviewer's is correct. Slightly shorter, keeping the thesis-sentence form:

```latex
The interior's density stays well below the exclusion ceiling and its
radius well above the comparison radius.
```

### R6 (Edit 3): finite-interval continuation constants

**1. Passage** (sec:singularity): "Unique continuation holds on a compact regular domain. On a finite realization, unique joint continuation follows while the solution stays in a compact regular positive-clock domain with bounded required reference derivatives and a uniformly invertible boundary-normalized constraint Jacobian. The reduced covariance evolution is then locally Lipschitz and preserves its spectrum."

**2. Mathematics of the proposal.** Neumann series: $\|F_u^{-1}\|\le\beta/(1-\beta\omega)$ when $\|F_u-J_0\|\le\omega$, $\|J_0^{-1}\|\le\beta$, $\beta\omega<1$: correct. $\|[h,C]\|_2\le2\|h\|_{\rm op}\|C\|_2$ and $\|C\|_2\le\sqrt{\dim}$ for $0\le C\le I$: correct. $\dot u=-F_u^{-1}F_C\dot C$ preserves $F=0$ by the chain rule; unitary conjugation preserves $0\le C\le I$; the lifted field is Lipschitz on the box given bounded reference derivatives (the manuscript's premise); the a priori speed bounds keep the trajectory in the box for $t<\min\{d_C/V,d_u/(MAV)\}$, and Picard–Lindelöf gives existence and uniqueness there. All correct. The bound is crude ($\sqrt{\dim}$ is large) but sufficient.

**3. Real problem?** No error in the text; the manuscript's qualitative statement is correct and the between-slice paragraph already lists the ingredients. The addition makes the "self-closing bound" explicit.

**4. Compatibility.** The reviewer's $A$ collides with the area $A$, $d$ with $d_i=\sin(ap_i)/a$, $h$ with the unbarred generator (the trajectory is written with $\bar h(u)$). The full theorem (half a page, theorem-like) conflicts with the page cap and the no-theorem-environment rule; the content belongs in the held supplement, with at most one sentence in the paper.

**5. Scope.** No claim changes.

**Recommended addition** (after "preserves its spectrum."), if pages allow; otherwise place in the supplement:

```latex
The premises give the interval explicitly: on a box
\(\|C-C_0\|_2\le d_C\), \(\|u-u_0\|\le d_u\) about an enclosed
solution, with \(\|F_C\|\le\mathsf f_C\), \(\|F_u^{-1}\|\le\mathsf m\)
(which \(\|F_u(C_0,u_0)^{-1}\|\le\beta\) and
\(\|F_u-F_u(C_0,u_0)\|\le\omega\), \(\beta\omega<1\), give as
\(\mathsf m=\beta/(1-\beta\omega)\)) and \(\|\bar h(u)\|\le\mathsf h\),
one has \(\|\dot C\|_2\le V=2\mathsf h\sqrt{\mathsf n}\) for a
one-particle space of dimension \(\mathsf n\) and
\(\|\dot u\|\le\mathsf m\,\mathsf f_C V\), so the coupled trajectory
exists uniquely on \(0\le t<\min\{d_C/V,\,d_u/(\mathsf m\,\mathsf f_CV)\}\),
the lifted system \(\dot C=-i[\bar h(u),C]\),
\(\dot u=-F_u^{-1}F_C\dot C\) preserving \(F=0\) and \(0\le C\le I\).
```

### R7 (Edit 4): between-slice certification

**1. Passage:** "The stored histories pass the slice tests above, and the between-slice bound is what the continuation calculation must certify."
**2–3.** The reviewer agrees with the text. No claim is contradicted. **4.** No replacement. **Class OPEN**; see open items.

### R8 (Edit 8): the proof supplement

**1. Passage** (Introduction): "Proofs beyond the sketches given here and the numerical sources are in the companion repository~\cite{McKinneyCode2026}."
**2–3.** Not a mathematical issue. The requirements make the full proofs a held supplement sent with the referee response, so the reviewer's request matches the project's own plan. **4.** No text change required; optionally the sentence could say "a versioned supplement in the companion repository". **Class OPEN** (process).

### R9 (Edit 9): tensor sector and next-gradient freedom

**1. Passages:** sec:model-gravity fourth step ("the fixed lattice carries no independent transverse-traceless perturbations ... whose tidal tensor has no transverse-traceless part"); Outlook ("\(a^2|\nabla\times\nabla\times v|^2/(16\pi GN)\) with any coefficient, vanishing with its variations on potential flows").
**2. Check of the reviewer's reasoning.** With $h_{ij}=0$, linearized $R_{0i0j}=\tfrac12(\partial_0\partial_ih_{0j}+\partial_0\partial_jh_{0i}-\partial_i\partial_jh_{00})$, whose TT projection vanishes (symmetrized gradients and a Hessian have no TT part): agrees with the text. The next-gradient example is degree one under common rescaling ($\lambda^2/\lambda$), kills constants and $\omega\times x$ under the double curl, is $\Theta$- and inversion-even, and is quadratic in $\nabla\times\nabla\times v$ so vanishes with its first variation on potential flows: agrees with the text.
**3.** Nothing to change. **Class OPEN** as the manuscript already prints it.

### R10 (Edit 10): state count, supply, copy obstruction

**1. Passages:** Outlook third question; sec:page-effective "the light output is invariant under every \(U(16k)\) rotation of the heavy copies (the copy-twirl argument in Ref.)".
**2.** The reviewer's argument ($[U_H,\mathcal U]=0$ and cyclicity of the partial trace over $H$ give identical light outputs for $\rho_H$ and $U_H\rho_HU_H^\dagger$) is correct and is what the text states. **3.** Nothing to change. **Class OPEN** as printed.

### R11 (B.3 final paragraph, Validation 14): the current-uniqueness clause

**1. Passage** (sec:model-dictionary): "In the uncoupled model the flow multiplies this exact lattice current and the spin curl below, no finite-range improvement of \(Q_i\) being compatible with the filled negative-energy sea of \(h_0\), the flat sea (Ref.~\cite{McKinneyCode2026})."

**2. Mathematics.** The reviewer's counterexample is right and I extend it. Any scalar function of momentum commutes with $P_0=\mathbf 1_{(-\infty,0)}(h_0)$. Take $q'_i=q_i+c\,[\sin(4ap_i)/(4a)-\sin(2ap_i)/(2a)]$: finite range, odd in $p_i$ (so $\Tr(P_0p'_{x,i})=0$), unit slope at all eight valleys ($\cos4ap_\star=1$), and $[Q'_i,P_0]=0$. So "preservation of the flat sea under uniform flow" cannot be the hypothesis of a uniqueness theorem. I also checked the rigid-turn rest within spin completions $\Sigma_kf_k(p)$: with $Q'_k=Q_k+\delta Q_k$ the extra orbital commutator $-i(D_j\delta q_k-D_k\delta q_j)$ is cancelled by an extra $f_i$ iff $f_i=c_j\delta q_k/(2d_k)=c_k\delta q_j/(2d_j)$. For single-axis $\delta q_k(p_k)$ this forces $\delta q_k=\lambda q_k$, so unit valley slope leaves $Q_k$ unique; for general $\delta q_k=q_k\,g(p)$ with cubic-symmetric $g$ vanishing at the valleys (e.g. $g=c\sum_i[\cos(2ap_i)-1]$) the completion $f_i=gc_jc_k/2$ exists, so the class must be named. A true theorem of the manuscript's shape plausibly exists (rigid-turn rest plus single-axis finite range), but the text names neither hypothesis nor class.

**3. Real problem?** The sentence is ambiguous, and under its most natural reading false. The reviewer's downgrade to "state the theorem" is the right call; since the repository statement is not available, I cannot decide the intended theorem (CANNOT-DECIDE on content; CLARIFICATION on text).

**4. Compatibility.** The paper does support: $Q_i$ is the energy-dipole rate; the energy-density algebra closes on it (sec:results-current, with "no current depending only on $u_{fg}$ reproduces this bracket"); the rigid-turn rest fixes the spin vertex given $Q_i$. The clarification should rest on these.

**5–6. Scope and bridge.** No qualifier is added; the unsupported clause is replaced by what is supported plus a pointer whose hypotheses the supplement must print. The weakest true statement is given; the admissible class is an open item.

**Recommended replacement** (last sentence of that paragraph):

```latex
In the uncoupled model the flow multiplies this exact lattice current
and the spin curl below. The current is fixed by the matter, not
chosen: it is the rate of the energy dipole and the energy-density
algebra closes on it (Sec.~\ref{sec:results-current}), while
preservation of the flat sea under uniform flow alone would fix
nothing, every scalar function of momentum commuting with \(P_0\)
(the uniqueness theorem for finite-range currents, with its admissible
class and its rigid-turn hypothesis, in Ref.~\cite{McKinneyCode2026}).
```

If the repository theorem states its class (for instance single-axis finite-range currents with unit valley slope), the paper lane should print that class in place of "with its admissible class".

### R12 (E, presentation): "Nor is it a toy model"

**1. Passage** (sec:discussion): "Nor is it a toy model. Its inputs are a few straightforward principles about ordinary objects, ..."
**2–3.** Not mathematics. The reviewer's objection is rhetorical; the owner's standing rules independently forbid the phrase "toy model". Deleting the one sentence loses nothing (the next sentence carries the content).
**4. Replacement:** delete "Nor is it a toy model." so the paragraph continues "Its inputs are a few straightforward principles about ordinary objects, ...".
**5.** No scope change.

### R13–R16

R13: sec:model-dictionary already says "Its content is four requirements" before they are used; no change. R14: the `\bibliography{ms_fermion_new}` entries are absent from the review package only; supply the compiled bibliography with the next package. R15: the unverified-numerics list is a provenance request; the supplement index (R8) answers it; no text change. R16: the six withdrawals are correct on my own rederivation (momentum-row sign with $K_r=v'/N$, $K_t=v/(rN)$; $\mathcal F=-T^r{}_t$ with $T_{ne}=-J$; the cross-block entropy derivative bounded by 2 with the pinching factor), and none re-reports a previous-round fix.

## 3. Do the scripts test the claims they are attached to?

- verify_01.py: yes for $h_0^2=\varepsilon^2$, for the rigid-turn cancellation (its `orbital_comm` is $[h_0,\epsilon_{ijk}X_jQ_k]$ in symbol form, matching my derivation), for the series coefficients of eq:dr-cone-expansion and the $3z^2/2$ of eq:dr-crossing-ray. The harmonic-bracket identity is a symbol-level reduction that confirms the arithmetic of $F_a(\xi)$ and $-5/12$, not the full operator bracket.
- verify_02.py: yes; the radial rows, variation identity, root slope, null-Lagrangian coefficients and the clock-bound integral $\pi L^2\sqrt{2G\Gamma}$ are the manuscript's statements.
- verify_03.py: yes; I confirmed its $T_{ne}=-J$ encoding ($n_\mu=(-N,0)$, $e_\mu=(-V,1)$, $n\cdot e=0$), the $-9/4$ vector Laplacian and the root/slope shifts, the discrete exterior, the delay logarithm and the dust redshift ODE.
- verify_04.py: the trace identity, yes; the second loop is a generic projector identity and does not test the $O(\xi^2)$ clock-vertex claim (verified analytically in R1).
- verify_05.py: yes at the level of one random finite instance for the window-energy identity, cross-block entropy formula, trace-norm bound and Wick variance; evidence, not proof.
- verify_06.py: yes for the spacing dependence (R2) and the quoted arithmetic (mass ratios 4.71/4.35, comparator $0.6211S_0$, $0.5105t_*$, two-field intervals, 0.49 percent, solar numbers); the threshold check tests a one-parameter reduction of the bulk threshold, not the full three-component statement.

## 4. Open items left after the corrections

1. **Between-slice continuation certificate (R7).** Closes with a validated box propagation under eq:dm-reference with the full $F_u$ including $\mathcal W''$ over each stored interval: enclosed starting root, derivative bounds, positive-clock and centre-regularity margins, a self-closing box (the explicit interval of R6), and an enclosure of the exponential-step defect relative to the continuous trajectory for seeding.
2. **Versioned proof supplement (R8, R15).** Theorem statements with hypotheses and an index mapping each claim to a file: directional-area/convexity (A.1–A.2), massive and massless estimates (A.3–A.5), the strong-sector obstruction, the current-uniqueness theorem with its admissible class and rigid-turn hypothesis (R11), the orbital and axial conversion-temperature coefficients.
3. **Normalization of the displayed $\mathsf T$ (R1).** Closes by reproducing at least one quoted $c_\rho$ (and defining $c_\rho$ in the text or dropping the numbers), and by confirming the repository's $\mathsf T$ equals eq:dm-response.
4. **Tensor sector and next-gradient classification (R9)** and **microcanonical count, autonomous supply, heavy-to-light transfer (R10)**: as printed in the Outlook; the reviewer's suggested first calculations (next-order invariant classification; finite $\widehat E_2$ charge-window state counts with packet matrix elements; two-port incident covariance during a driven interval) are the resolving computations.
5. **Minor consistency flags (not raised by the reviewer):** "Killing burst bin" undefined at first use (fixed by the R4 wording); $2GE=59.05a$ (cold formation control) versus $2GE_0=59.07a$ (reference run) to be confirmed as intentional.

## 5. Adversarial self-check

**R1 (insert the boxed $\mathsf T$).** Strongest case against: the Moyal expansion might contain second-gradient bulk terms the perturbative calculation misses (intraband or Berry-type terms, or non-divergence pieces of the midpoint symbol corrections), or the repository's $\mathsf T$ may differ in normalization, so the displayed formula would be inconsistent with the quoted $c_\rho$ and the implemented sources. I checked that the zeroth-order eigenvectors are $x$-independent (no mixed Berry curvature at this order), that $\tr(\pi_1h_0)=0$, and that the symbol corrections are bulk divergences; the residual risk is the implementation match, which is why the verdict is "apply modified, conditional on reproducing $c_\rho$". The verdict stands.

**R11 (clarify the uniqueness clause).** Strongest case against: the author's intended theorem may be exactly "no finite-range improvement compatible with the sea" under hypotheses I have not guessed, making my rewording a weakening of a true, proved claim. Against this, the clause as printed admits an explicit counterexample under its natural reading, and my own check shows the admissible class matters (single-axis versus multi-axis improvements), so the hypotheses must appear in the text either way. The verdict (CLARIFICATION, with the theorem's content undecided until the supplement states it) stands; the paper lane should restore the stronger wording only together with its printed hypotheses.

**R6 (add the explicit interval).** Strongest case against: the addition is redundant with a correct sentence and the repository's "full statement", costs lines against a 29-page cap, and its $\sqrt{\mathsf n}$ bound is too crude to certify any real interval, so it adds no usable content to the paper. This is a page-budget judgment, not a mathematical one; if the budget forbids it, the correct outcome is to move the sentence to the supplement, which changes the verdict's placement, not its substance. (R12 is the other low-certainty call: the reviewer's point is purely rhetorical, and I apply it only because the owner's rules forbid the phrase.)