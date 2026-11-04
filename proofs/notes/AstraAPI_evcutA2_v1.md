
**The split is sound; the finite-size constant and the measurement gate need corrections.** References below are to `00_FableAPI_evcutF_v1.md`; the underlying paper and numerical records were not supplied.

### 1. Theorem 1

**(i)–(iv) are correct**, assuming the stated orthogonal-projector purity, \(\Delta_B\ge0\), and \(\zeta>0\). In particular, “\(E_f(P_0)=\gamma_1D[f]\)” has the correct spin multiplicity.

- \(K_{\rm dust}=\sqrt{3.04}/0.0923=18.890\ldots\) is correct.
- \(\gamma_1(0.2)\) is approximately **\(0.392\)**; §2’s “\(0.393\)” is slightly high. The quoted \(8\pi\gamma_1\simeq9.9\) remains correct at that precision.
- **Not correct as a fixed all-size constant:** “\(K_0=\gamma_1D[f_L]/L\to8\pi\gamma_1\)” defines a size-dependent value and supplies its limit, not a uniform cap. Use \(K_0(L)\), or prove a supremum over the specified size range. Literally, \(D[f_{0.1}]=6>8\pi(0.1)\).

Thus the exact split and sea lower bound hold; the advertised uniform \(K_0\simeq9.9\) requires a finite-size bound.

### 2. Proposition 2

**Yes, as a conditional statement for the stipulated smooth, slowly varying lattice Hamiltonian.** The pointwise band gap survives scalar tilt; the first-order corrected projector has the quoted “defect … \(\le c\epsilon^2\).” Duhamel over \(O(\epsilon^{-1})\) time gives \(O(\epsilon)\), and the projection-block argument correctly converts this into \(O(L)\) window energy.

It does **not** follow from spatial hydrodynamic tests alone. Nor do discontinuously held fields automatically satisfy H3. The supplied histories have not been certified by this proposition.

### 3. Calculation and gate

**The quantity is right, but only for the prescribed-field cubic evolution.** The caveat “without cubic back-reaction” is essential: this does not validate a self-consistent cubic collapse. The trace estimator is unbiased, and its stated variance upper bound is valid.

**The gate is not sound as written:**

- G2’s “the uniform budget is refuted” is false. Finite-size growth exceeding \(0.5\) rejects that tolerance, not the existence of some uniform \(K\).
- G3’s “size-256 sea term stands as the cubic proxy within \(\pm0.5L\)” does not follow from agreement at 32 and 64. Size 256 remains an extrapolation unless independently validated.
- G1 must use an **upper uncertainty endpoint**, not the sample mean. Failed precision at the sampling cap is inconclusive. Uncertainty control must accommodate stopping and multiple records.
- Failure of the sea cap alone does not imply failure of the total \(30.8L\) cap: the dust cap may have substantial slack.

Before accepting tested records, measure convergence of \(\mathcal E_{\rm sea}/L\) and \(\mathcal N_A\) under **box enlargement/taper displacement, interpolation refinement, and propagator-tolerance tightening**, and include those errors. A “box of side \(10L\)” is not automatically safe through time \(40L\). Also verify \(R_t\le L\), required by G4’s window bound.

### 4. Section 4: exact mathematical replacements

Replace:

> “For a window equal to one on the cut and falling to zero over a shell of width \(L\), the window term is at most \(8\pi\gamma_1L\)”

with:

> “For the specified harmonic window \(f_L\), the window term is exactly \(\gamma_1D[f_L]=(8\pi\gamma_1+o(1))L\); a finite-size uniform cap requires a bound on \(D[f_L]/L\) over the stated size range”

Replace the clause beginning “which is \(O(L)\) wherever” and ending “(Sec. IV A)” with:

> “which is \(O(L)\) on intervals \(t\le T_*L\) satisfying Proposition 2’s full (H1)–(H3), including its additional field-rate assumptions; the reported spatial-bound failures prevent this proposition from certifying the computed histories”

Keep the paragraph explicitly in **\(a=1\) units**. If restoring physical energy units while retaining its dimensionless \(\gamma_1\), the baseline is \(\gamma_1D[f_L]/a\), and the reference dust cap is \(18.9L/a\).

The shorter alternative and Discussion wording are appropriately conditional. The quoted history counts, chain budgets, “exceeds the area term,” and unseen Sec. III G sentence cannot independently be checked from these inputs; the area claim requires evaluating the final entropy-error bound against the area term on the stated cut.