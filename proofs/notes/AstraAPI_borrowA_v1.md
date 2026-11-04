
**Do not apply borrowF wholesale.** Some identities are sound, but several proposed “computed” results are unsupported, and several changes remove qualifications that P3 needs.

Below, **F** means `00_FableAPI_borrowF_v1.md`; **P** means `01_P3_current_nc.tex`. Quoted text and equation labels are search anchors. No implementation functions, numerical exports, PDFs, or `emisF6` were supplied. P also expressly says **“no stored history is verified under Eq.~\eqref{eq:dm-reference}'s source prescription”**; new post-processing does not remove that qualification.

## B4: cubic energy–energy bracket

**Verdict/status.** The displayed path identity `eq:dr-lattice-bracket`, mass cancellation, and **bulk affine-smearing result are derived correctly** for P3’s `e[f]` and two-step `Q_i`. The claim **“closes on the lattice itself, for any two smearings”** is false if closure means P3’s `h[0,u]`.

P’s sentence **“No current depending only on \(u=f\nabla g-g\nabla f\) reproduces this bracket”** must stay. Indeed,
\[
f_\xi=\cos(\xi x_i),\qquad g_\xi=\frac{\sin(\xi x_i)}{\xi}
\]
give the same \(u=e_i\), but their bracket is \([F_a(\xi)/\xi]Q_i\), which depends on \(\xi\). For general smearings, the cross-direction sum is likewise not merely the model’s curl coupling.

**Exact replacements.**

Replace F’s opening assertion with:

> “The lattice bracket has the exact path decomposition below. Its scalar part contains axial two-step paths, while its cross-direction part is grade two. For affine smearings in the bulk it equals the model’s flow and spin coupling at \(u=f\nabla g-g\nabla f\); general smearings have additional lattice dependence not determined by \(u\).”

Replace the background-clock conclusion with:

> “For \(f=N\mathcal N\) and \(g=N\mathcal M\), the long-wavelength bracket has \(u=N^2(\mathcal N\nabla\mathcal M-\mathcal M\nabla\mathcal N)\). This identifies the background-lapse structure function at leading order, not an exact finite-lattice constraint algebra.”

Replace the spin-expectation argument with:

> “For real smearings the grade-two term is odd under the time reversal of Appendix~\ref{app:local}; its expectation therefore vanishes in \(P_0\) and in any covariance \(F(h[N,0])\), but not necessarily in a flow-prepared or propagated state.”

The stated \(\omega\)-argument alone does not prove the claim for arbitrary \(F\); time reversal does. Also:

- Change **“the factor \(F_a(\xi)/\xi\)”** following the midpoint expansion to **“\(F_a(\xi)\)”**.
- Retain **“continuum bracket”** in the radial-smearing sentence.
- Constant-flow mixed closure is **long-wavelength, \(O(a^2)\)**, not exact.
- Keep P’s full-deformation-algebra caveats in `sec:model-gravity` and the outlook.
- The dictionary may say: **“Both coefficients follow from the exact bulk bracket for affine smearings; the completed rigid turn independently commutes with \(h_0\).”**

**Computation before print:** none for these corrected algebraic statements. Any claimed numerical validation should compare direct commutators with the path formula for random smearings, and separately verify affine equality and generic non-equality with `h[0,u]`.

## B7/B1a/B2: barrier, currents, microscopic check

### Barrier and currents

**Verdict/status.** The partner potentials and their equal transmission are **derived correctly in the continuum limit**. They are not the exact transmission of P’s finite-spacing, back-averaged `h_{0,k}+\{\bar v,Q_r\}/2`.

F’s **“spinor boosted by \(e^{\eta\sigma_1/2}\)”** omits the amplitude rescaling and stationary-coordinate phase needed for the displayed equation.

**Replace the barrier lead-in with:**

> “For the leading continuum limit on the unaveraged vacuum profile, write the stationary chain spinor as \(e^{-i\omega t}\psi_\omega(r)\), set \(t_s=t+F(r)\), \(F'=v/\alpha_s^2\), and define \(\chi=\alpha_s^{1/2}e^{\eta\sigma_1/2}e^{i\omega F}\psi_\omega\), with \(\tanh\eta=v\). Then \(\omega\chi=(-i\sigma_1\partial_{r_*}+k\alpha_s\sigma_2/r)\chi\). Its unit-flux transmission is the continuum comparison transmission; finite-spacing and back-averaging corrections require separate calculation.”

The currents using \(s(\Gamma f)\), and the modewise \(s(\Gamma f)>(\omega/T_H)\Gamma f\) for nonzero transmission, are correct. Thermal supply remains **assumed**, not established on the collapse.

Replace F’s sharp-count sentence with:

> “For binary transmissions, \(\Gamma_k\in\{0,1\}\), the currents depend on the open-channel count \(C(\omega)\); a constant count gives \(\mathcal R=2\), and \(C\propto\omega^2\) gives \(4/3\). These are analytic comparison limits, not bounds on a soft barrier, for which \(\sum_k s(\Gamma_k f)\) is not determined by \(\sum_k\Gamma_k\).”

### Numerical provenance

P reports two calculations of the rates, but also says **“finite-band estimates … have no narrow absolute bound.”** F supplies no P3 evidence for **“six digits … by \(k\le4\)”**, the \(1/r_{\max}\) entropy-error law, the new Page benchmarks, or the flat-chain percentages.

**Replace those claims with:**

> “Two independent P3 calculations reproduce the finite-band estimates \(\mathcal L\simeq0.0052373\) and \(\mathcal R\simeq1.6391\), without a narrow absolute bound. The additional convergence, Page-benchmark, and quench checks remain unverified for P3.”

The limits are **derived**. New benchmark/quench numbers are **reported by F, not verified P3 computations**. Keep the number rate unverified.

### B2 runnable specification: not valid as written

F freezes \(k/r_{\rm in}\) in a lead at \(r_{\rm in}=1.2r_s\). That lead is gapped:
\[
\Delta_{\rm res}\simeq \frac{k}{r_{\rm in}}\sqrt{1-v_{\rm in}^2}
       =4.28\,k\,T_H .
\]
Thus **“both branches propagate”** does not hold throughout the thermally relevant frequency range. This lead does not supply the horizon-port distribution assumed by the barrier calculation.

There is also a counting mismatch. P specifies **“weight \(16k\)”** for the full sine chain. An unfiltered massless chain has two radial valleys and particle/hole channels. Its target is
\[
2\sum_{\alpha=1}^{2}\int_0^\infty\frac{d\omega}{2\pi}
 \bigl(\omega n_\alpha,\ s(n_\alpha)\bigr),
\]
reducing to **four**, not two, times the integral when the two transmissions coincide.

**Required computation/specification:**

1. Construct the actual P3 joined operator, including two-step `Q_r`, back-averaging and the centrifugal term. Compute its transmission eigenvalues independently of covariance evolution.
2. Compare the quench with **that joined operator’s** Landauer targets. To identify them with Hawking-barrier rates, additionally demonstrate continuum and inner-port matching convergence, with the missing low-frequency contribution and splice reflections bounded.
3. Repeat lead/domain lengths, cut position, fit windows and lattice resolution; include the zero-temperature joining transient and entropy logarithm.
4. Replace **“\(T_H\) halved (the ratio must not move)”** with:  
   > “Change \(T_H\) with \(r_s\) so that \(T_Hr_s=1/(4\pi)\), or compare against the recomputed temperature-dependent ratio.”
5. Do not identify the nonuniform exterior’s ground projector with the prepared band from a pointwise speed test. P explicitly says no such test fixes the projector.
6. For printed rate precision, vary frequency endpoints/quadrature, \(k_{\max}\), asymptotic boundaries and matching order on both integrators; state benchmark degeneracies explicitly.

The block-entropy telescoping identity is a consistency readout, not a second independent evolution calculation.

## B1b/B1c/B3a/B3b: late branch, channel, register, purity and pairs

### B1b: an entropy bound, not a measured capacity

**Verdict/status.** The subadditivity argument is correct only when the received modes, trapped interior and remaining exterior form a complete orthogonal partition of the same pure state. It gives an **upper bound**, not a measured late branch or an accessible Hilbert-space dimension.

F’s **“\(f\) … that ratio’s leading value”** also uses the wrong normalization: \(S_{\rm cut}/S_0\) contains the changing area factor. For pure light, the coefficient is \(\eta(0)\), not the heavy field’s \(\eta(ma)\). P’s direct-state budget is expressly **not a massless-limit estimate**.

**Replace the proposed insertion and capacity reinterpretation with:**

> “If the received light modes lie wholly outside the trapping cut and the remaining modes form a complete orthogonal complement, purity and subadditivity give \(S_{\rm esc}(u)\le S_{{\rm cut},L}(R_t(u))+S(\mathcal X_u)\). This is an entropy bound, not an accessible-sector count or a saturation law. If the evolved light cut retains its flat coefficient, \(S_{{\rm cut},L}/S_0=[4\eta(0)G/a^2](R_t/R_0)^2+o(1)\). The accessible store of logarithmic dimension at most \(fA_R/(4G)\) remains assumed.”

P’s **“109 to 311 nats”** are heavy-chain trapping-cut **excesses**. Its **“about nine nats by \(T/r_s=100\)”** concern light exterior-cut excesses, not the same trapping-cut measurement. Neither supplies the full cubic \(S_{\rm cut}\).

**Before print:** propagate the cubic covariance on the specified stored geometry; measure full cut entropies for each field and flat controls on identical site sets, with numerical/size errors. Measure \(\mathcal X_u\), packet–interior overlap, return and omitted-mode entropy. Label this a one-way cubic control, not a coupled cubic history. An accessible-sector bound and uniform massless/scaling premise require additional arguments, not merely finite-cut measurements.

### B1c: no P3 achievability construction yet

F’s **“exists inside the model’s own class”** conflicts with P’s required **“one linear autonomous charge- and number-conserving generator”** and funded, charge-matched release.

The proposed rotation from an occupied one-body level \(-\omega\) to an empty level \(+\omega\) costs
\[
\Delta E=2\omega\sin^2\theta .
\]
It does not create a negative-energy hole partner at zero cost. That hole would require removing an occupied **positive** Killing-energy level. Moreover, swapping the stored half into an empty outgoing mode gives occupation \(1-n\), not automatically the prescribed \(n\).

**Replace the entire achievability claim with:**

> “The enqueue/swap circuit is an externally controlled Gaussian proposal, not a realization of P3’s autonomous boundary generator. Attainability with the prescribed outgoing marginals, measured pocket energy, charge-matched encoding and complete release remains unproved. An unresolved block costs at most its entropy; that entropy does not generally vanish when its frequency width is narrowed.”

**Before print:** supply the actual permitted generator and funding mechanism. Verify energy and species/copy-number conservation, outgoing occupations, available pocket inventory, Gibbs ceiling and \(S_{\rm out}\) throughout the proposed curve. Test reference-entangled charge-matched inputs and retained decoupling, not only one entropy trajectory. A successful arbitrary Gaussian circuit still does not establish collapse realization.

### B3a: identity correct; “partner” and purity claims too strong

The identity is **derived correctly**. Mutual information compares total correlations, not a unique partner mode. Equality of the two mutual informations means a zero increment; a maximum requires the conditional entropy to change **positive to negative**.

**Replace with:**

> “For the complete pure partition \(B,\mathcal E_u,\mathcal S_u\), \(I(B{:}\mathcal S_u)-I(B{:}\mathcal E_u)=2S(B|\mathcal E_u)\). Thus the sign of the accumulated output’s increment equals the sign of this correlation imbalance; a change from positive to negative marks a turnover of \(S_{\rm esc}\).”

Replace **“the identity’s residual as the purity check”** with:

> “the identity residual as a consistency check, alongside an independent global purity measurement.”

A mixed spectator in \(\mathcal S_u\) can leave the residual zero, so zero residual does not certify purity.

**Before print:** supply the actual orthonormal no-return packet basis and complete complement; independently re-propagate C1/C2; compute the listed entropies, return/tail errors and global purity. Do not infer complementary entropies from purity and then present the resulting zero residual as a test. No positivity result for this new register is supplied.

### B3b: purity is derived; the proposed “pair count” is not

P’s `i\dot C=[h,C]` preserves the spectrum. Numerical purity requires measurement, but the theorem does not.

**Replace the purity sentence with:**

> “Unitary quadratic evolution preserves the global purity of the initially pure light covariance, while the drive may create excitations and cut entanglement. The numerical purity defect remains to be measured.”

P says **“The subtraction has no reference state.”** Its `\mathcal W` and prepared-band expansion do not specify the exact global projector `P_band(t)` assumed in F.

For a **declared exact equal-rank projector** \(P\), replace the pair sentence with:

> “The difference of the two projectors has paired positive and negative eigenvalues. Its trace norm is a state–comparator distance, whereas the particle and hole excitation numbers are \(N_p=N_h=\Tr[(I-P)C_L]=\tfrac12\Tr(C_L-P)^2\). Their interpretation depends on the comparator; a changing band distance alone does not separate production from reference motion or transport.”

The equality of positive/negative eigenvalue **sums** tests
\(\Tr C_L-\Tr P\), not purity.

**Before print:** define and validate \(P\), its continuation, rank and idempotency; report \(\Tr s(C_L)\), \(\|C_L^2-C_L\|\), particle number, \(N_p,N_h\), and both distance norms. Compare evolving, held and adiabatic controls, including \(\dot P\). The test-packet norm defect does not bound the full covariance’s entropy defect.

## B8: discrete exterior and remainder

**Verdict/status.** The stated \(c_1,c_2\) and trigamma expansions are correct **for the separately chosen discretizations**. They are not established for P3’s implemented constraint function `F(C,u)`. P says **“The model specifies no lattice-scale geometric law.”**

**Replace the opening with:**

> “An illustrative flux discretization is \((r_n+\theta a)(v_n^2+v_{n+1}^2)/(4G)=E\). Its smooth vacuum solution has the expansion below, with \(c_1=\tfrac12-\theta\) and \(c_2=-\theta(1-\theta)\). This discretization has not been identified with the production solver; the trigamma profile belongs to a different cumulative scheme.”

The effective sources are a **stationary continuum interpretation of numerical error**, not new P3 matter. The quoted \(\beta_{\rm PN}\) is conditionally correct, but derivation note four has the wrong intermediate expression. Replace it with
\[
-g_{t_st_s}=1-2U+2\left(1-\frac{2c_1a}{r_s}\right)U^2+\cdots,
\qquad U=\frac{GE}{\tilde r}.
\]
Do not print these as physical P3 corrections or describe them as larger than all probe corrections; P’s probe corrections depend on momentum, not universally \(a/r_s\).

**The back-averaging prediction needs correction:**
\[
\boxed{\bar c_1=c_1,\qquad
\bar c_2=c_2-\frac94(\sigma/a)^2.}
\]
Averaging does not erase a nonzero \(c_1\). Values \(-36\) and \(-9\) require zero fine-grid \(c_2\).

**Before print:** obtain the actual discrete rows and boundary normalization; run a genuinely vacuum fixture; fit unrestricted coefficients with radial-window and resolution controls. On histories, measure/subtract or bound \(\rho_\ell,J_{\ell r}\), charge variation and \(N-1\). **\(R_{90}+3\sigma\) is not a vacuum certificate**: dust remains outside \(R_{90}\), and the Gaussian has tails. Audit the implemented vector back-averaging, then compare fine/back-averaged residuals jointly. The solar number is only arithmetic conditional on an assumed \(c_1\).

## B9: figures

P names `fermion_new/figures/item5_collapse_v3.1.pdf`, `fermion_new/figures/page_v5.pdf`, and `fermion_new/figures/page_curve_amps.pdf`; the actual PDFs/data are absent.

- **Conditional panel (a):** the supplied maxima and normalized Page times follow correctly from `eq:page-effective-curve`. Keep its assumed-law status and identify the \(f\) band as a heavy-field coefficient interval under the store assumption, not a measured capacity/error band.
- Replace **“neither \(\eta\) nor \(a/\ell_P\) enters the time”** with:
  > “The fraction \(t_P/t_*\) depends on \(\mathcal R\) and \(f=4\eta G/a^2\); \(\eta\) and \(a/\ell_P\) enter through \(f\), while the dimensional scale \(t_*\) also depends on \(\mathcal L\).”
- **Exterior panel (d):** needs the corrected B8 audit. Its caption cannot already assert the production scheme’s remainder.
- **Quench panel (d):** needs corrected B2.
- **Register panel (c):** needs B3a. Delete **“on these records the conditional entropy stays positive”** until that register has actually been computed.
- Identify each plotted history; P’s existing collapse field panels use **“a separate production history”**.

## B10: cheap checks

### Modular slope

F’s **“\(\beta_{\rm ent}(1)=\pi a\)”** uses a site weight for a hopping bond. With a midbond cut, sites at \((n-\tfrac12)a\), and the first bond wholly inside \(A\),
\[
\boxed{\beta_{\rm bond}^{\rm BW}(n)=2\pi na,}
\]
not \(2\pi(n-\tfrac12)a\). More generally use the bond midpoint’s distance from the cut.

**Replace the expected-value sentence with:**

> “The uniform massless midbond control is compared with the bond boost weight \(2\pi na\). Corrections from the radial \(k/r\) term, finite boundaries and the lattice coefficient \(b(M)\) are measured rather than presumed negligible.”

The proposed clipping also invalidates the depth test: clipping at \(10^{-14}\) bounds \(\|k_A\|\) by approximately \(32.24\), whereas the \(n=20\) ideal hopping magnitude is \(20\pi\simeq62.83\). That clipped matrix cannot reproduce the proposed deep slope.

**Before print:** benchmark the uniform chain; vary precision, clipping, length and cut location, with errors on each reported bond. P’s Mirsky entropy bound does **not** control the singular matrix logarithm from a local covariance difference. Uniform flow preserves \(P_0\), but this supplies no modular-slope prediction for the evolved nonuniform state. An outgoing thermal comparison needs an outgoing-mode projection, not merely a spatial hopping ratio.

### Tidal eigenvalues

The two tensor formulas have the correct sign for P’s `K_{ij}=+\tfrac12\mathcal L_n\gamma_{ij}` and `eq:dm-evolution`. The source expression is conditional on that evolution equation, not a constraint-only curvature identity. If \(\mathscr R_{ij}\) is its left-minus-right residual,
\[
E^{\rm direct}_{ij}-E^{\rm source}_{ij}=-\mathscr R_{ij}/N .
\]

F’s **“stored kinetic stress components”** is not supported: P identifies the available readings as **“held-comparator operator-stress residuals, not those of the kinetic \(S_{ij}\)”**.

Replace the Schwarzschild expectation with:

> “The vacuum fixture must reproduce \((-r_s/r^3,r_s/(2r^3))\). On stored trapping cuts the matter terms and evolution residual are retained explicitly; Schwarzschild values are not imposed as targets.”

P’s trapping surface is inside matter, and \(R_t\) is not generally \(r_s\).

**Before print:** reconstruct matched P3 sources and kinetic stress on identical field/covariance slices; evaluate both readings at the actual \(R_t\), \(R_{50}\) and declared core points; refine temporal/spatial differences and report constraint/evolution residuals separately. At \(r=2a\), below P’s quoted massive-chain subtraction exclusion scale, label results as lattice/interpolated-metric diagnostics. Delete both affirmative optional hooks until these measurements exist.

**Bottom line:** corrected identities and conditional derivations can enter now. New benchmark, quench, cubic-capacity, channel-realization, register, pair-production, solver-remainder and modular/tidal numerical claims cannot.