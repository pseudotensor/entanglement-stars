
## Ruling

**Concede the physical covector and the physical vertex/rate agreement at the closing-check precision. Do not certify the total collar replacement. The massless cubic coefficients agree within the reported numerical enclosures.**

I am comparing the supplied reports, not rerunning their referenced scripts or inspecting their JSON/NPZ files. Per your disclosure, these are separate implementations on shared inputs, **not a mutually blind comparison**.

### (a) Physical vertex and signed rate: **AGREE**, with the broader closing-check errors

The physical heavy leg is \(s_{C,H}\), as specified in your question. The \(s_C-t\) contraction is a test direction. **No further covector-selection calculation is needed for this ruling.**

`03_VRC_routeA_solve_v1.8.md`, “ROUND 5,” reporting `vrc_direct_v3.py`, states:

> “Solve D_0 du_0 = B j_L, then D_full dz = -B V^-1 chi_L du_0”

and reports five refinement levels with inverse-operator-propagated solve errors.

`06_VRC_ROUND5_1104_webgpt_v1.md`, §3, reporting `vrcb_close_v1.py`, explicitly adopts:

> “dV error 2.5e-9: the method spread 2.05e-9 plus the fixed-point Richardson change 4.0e-10.”

> “Rate error 3.2e-8 and V_0 error 1.1e-8, both method spread.”

Using those **stated closing-check errors**:

| Physical quantity | A: direct correction | B: quoted closing result | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| \(dV\) | \(1.56880104335\times10^{-5}\pm2.0\times10^{-11}\) | \(1.56863348732\times10^{-5}\pm2.5\times10^{-9}\) | \(1.6756\times10^{-9}\) | \(1.068\times10^{-4}\) |
| \(\Delta\Gamma/\Gamma_0\) | \(-2.40452963397\times10^{-4}\pm3.3\times10^{-10}\) | \(-2.40426503947\times10^{-4}\pm3.2\times10^{-8}\) | \(2.6459\times10^{-8}\) | \(1.100\times10^{-4}\) |

**Both agree within summed closing errors and pass \(10^{-3}\) relative.**

The requested exported-covector check also passes. B’s §3 “tangent solves” value differs from A’s v1.3 \(dV\) by \(1.5112\times10^{-9}\); B’s quoted direct value differs by \(1.6886\times10^{-9}\). Both are below the approximately \(2\times10^{-9}\) target.

**The narrow direct-error claims do not agree.** B’s §2 quotes \(3.4\times10^{-11}\) for \(dV\) and \(6.1\times10^{-10}\) for the rate. Against A, the physical discrepancies are approximately **31 and 28 times the summed narrow errors**. The \(s_C-t\) test likewise retains discrepancies \(1.6816\times10^{-9}\) and \(2.6537\times10^{-8}\), outside its narrow errors.

Thus the dominant percent-level difference was the covector; the remaining approximately \(10^{-4}\) relative difference is covered by B’s **explicit broader numerical budget**, not explained by the narrow direct-error estimates. No further calculation is needed to print the broader result.

### (b) Cross residuals: **consistent on the tested tangents**, not a completed reciprocal audit

`06_VRC_ROUND5_1104_webgpt_v1.md`, §1, reporting `vrcb_round5_v1.py`, prints:

> “D_0 on zero_A: 1.3e-7 / 6e-7.”

> “D_full on full_A: 1.9e-7 / 5e-7.”

Its bare-response comparison is \(1.2\times10^{-7}\) to \(6\times10^{-7}\). These support consistency of the tested actions at B’s reported finite-difference floor. They are **not an operator-norm bound**, and a residual normalized to the baseline response does not certify the much smaller correction to that same relative accuracy.

Only **B acting on A’s exported tangents** is supplied. A’s round-five section still says:

> “Cross-evaluated residuals: PENDING”

and names `vrc_xres_v1.py` as the reverse check.

**Calculation needed only to recover the narrow precision:** export B’s baseline and correction tangents; apply A’s operators; compare the two right-hand sides and operator actions on identical vectors; propagate the cross residuals through the inverse operators into **both covectors and the unrounded channel-rate sum**. That would distinguish a small systematic action difference from understated errors. Another baseline re-solve would not.

### (c) Collar: **no two-route, error-controlled total replacement**

The controlling A result is `04_VRC_routeA_solve_v1.9.1.md`, “COLLAR TEST, LITERAL FORM,” not the superseded interpretations in v1.8. It defines:

> “eps_remainder = |Q(hd) - Q(hd/2)| + |Q(D8) - Q(D16)|/3”

and applies the requested majorant test.

| A, width \(8a\) | Ratio of left to right side, both grids | Verdict |
|---|---:|---|
| Heavy | 1.144 | **FAIL** |
| Light | 0.760 | **PASS** |
| Total | 1.107 | **FAIL** |

The same file identifies the cause:

> “The majorant's refinement error is dominated by its dependence on the u_dot stencil (13 % of Q).”

B’s §4 instead prints the heading:

> “|W-Q| + eps_rem vs 1e-3 (|W| - eps_W)”

where its \(Q\) denotes the **collar**, not the outside majorant. It supplies only the fine-grid table and describes errors as including grid and “polarization Richardson” changes. It does **not** print the majorant’s own \(D_8/D_{16}\) dependence or the complete requested two-grid audit.

Changing B’s remainder to its almost identical outside majorant would not, by itself, erase its reported pass. **The substantive omission is the unprinted majorant-specific temporal error**, not merely the symbol choice.

For A’s total at the coarse grid, the remaining allowance for \(\epsilon_{\rm remainder}\) is approximately \(4.82\times10^{-8}\); the reported value is \(2.514\times10^{-7}\). Spatial refinement alone does not resolve that.

**Disposition:** retain the full contraction. A establishes a light-only pass under the stated test, but B supplies no species-resolved confirmation. Neither report establishes a two-route total replacement.

**Exact missing B calculation:** print \(W,C,Q\) on both grids for \(D_8,D_{16}\) and their Richardson combination, then apply the literal rule with the majorant’s own refinement error. A narrower temporal bound requires closer records; a different collar requires a separately specified width test. Even a pass would localize the Gaussian-window quantity, not identify it with the sharp-root debit.

### (d) Massless cubic tensor: **AGREE within the reported enclosures**

`05_CUBIC_K0_routeA_v1.md`, reporting `vrc_k0_v1.py`, defines the off-doubler grid and the shear convention:

> “c44(sym) = (kappa_12,12 + kappa_12,21)/2”

`06_VRC_ROUND5_1104_webgpt_v1.md`, §5, reporting `vrcb_cubicK0_v1.py`, supplies the matching massless refinement calculation. Their extrapolated values agree to at most \(2\times10^{-17}\):

| Entry | Common numerical limit, rounded |
|---|---:|
| \(c_{11}\) | \(0.010517254745\) |
| \(c_{12}\) | \(-0.003907418876\) |
| \(c_{44,\mathrm{sym}}\) | \(0.006199438345\) |

B’s printed enclosures contain A’s direct-\(m=0\) enclosures. Agreement also holds using A’s wider **final frozen enclosures**.

Keep that distinction: A explicitly labels its widening as the consequence of inconsistent positive-mass polynomial extrapolations. Its proposed \(m^2\log m\) explanation does not silently replace the frozen rule. These are **numerical, convergence-based enclosures**, not proved interval bounds on the integral.

The massless numerical calculation is closed for the requested comparison. It does not establish the radial-chain-to-cubic multiplicity map.

## What earns paper space?

**Earned:** the physical signed correction with the broader numerical error, and the numerical massless-tensor agreement.

**Not earned:** narrow direct precision, a total collar replacement, the full heavy-light Hessian, or a completed root/history balance.

At the page limit, I would print **one new rate sentence**, keep the cubic coefficients in the supporting delivery, and remove both completed numerical tasks from the mark. This follows `01_requirements_new.txt`’s requirement:

> “two independent routes agreeing within their stated errors before anything enters the paper”

without presenting numerical-method agreement as a physical emission law.

### Same-budget paragraph swap

Replace the entire paragraph beginning in `00_06_scaling.tex`:

> “This diagnostic uses a two-field history”

with:

```latex
This two-field diagnostic has nominal
\(r_{s,\mathrm{nom}}/a=80\), \(m_Ha=0.2\), \(m_La=0\),
\(G/a^2=1/4\), release \(\eta_E\simeq0.919285\), and
averaging width \(\sigma=4a\). Heavy columns \(6,5,4,3,2\)
in \(k=1,\ldots,5\) have weights \(16k\); the light channel
\(k=1\) has full-chain weight 16.
Equation~\eqref{eq:dm-reference}'s polarization is subtracted
only for the heavy field.
At \(t/a=1104\), \(R/a\simeq69.47\), the light reference response
changes the unit-weight sum of squared heavy-light vertices by
a fraction \(-(2.4043\pm0.0004)\times10^{-4}\), using heavy
moments without reference subtraction; the numerical error
includes the spread between methods.
The fixed-radius reference term has the mass rate's sign and
is 8.6 percent of it (7.9--9.5 with record-spacing errors),
the light part 0.7 percent. Following the recorded root gives
\(-(3.27\pm0.10)\times10^{-3}\) in lattice units.
These single-record tests leave the moving-root and history
balances open.
```

The rounded rate and error conservatively cover both routes’ closing intervals. “Sum of squared heavy-light vertices” avoids calling this diagnostic an established emission rate.

Leave “Three common inputs…” intact: selected contractions do not close those full inputs. Replace its `\todo` with:

```latex
\todo{complete two independent calculations of the three inputs
above: bound the full root debit and collar remainder under
record-spacing and width refinement; include other-clock
reference terms; establish funded complete release and uniform
decoding and response-error bounds}
```

These swaps stay within the existing text budget. **Rendered paragraph height and the final PDF page count must still be checked by rebuilding**; the supplied excerpt alone cannot establish the 29-page fit.