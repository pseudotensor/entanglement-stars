
**No blanket two-route pass: all 10 outer-root interval pairs overlap; the 10 inner-root interval comparisons cannot be decided from the extract.** Route B supplies only rounded inner-root values, not their enclosures.

### 1. Comparison

Sources: Route A’s `00_ROOTDEBIT_M1_CLOCK_BATCH_v1.md`, using `do_record_v1d.sh`/`v3f`, explicitly says **“H_R is PROVISIONAL: the reference response Q_R and the tangent are not enclosed.”** Route B’s final clock block in `01_routeB.md`, using `run_v3`/`lib_v2`/`interp_v2`, reports **“ALL CERTIFIED.”** These are comparisons of the supplied printed numbers, not independently inspected output files.

**Outer root — agree numerically in every row.** The decisive intersections are:

| \(t\) | B: P | A ∩ B: P | B: H | A ∩ B: H |
|---|---|---|---|---|
| 1096 | `[-0.0069157,-0.0068996]` | `[-0.0069149,-0.0068996]` | `[-0.0069410,-0.0069248]` | `[-0.0069404,-0.0069248]` |
| 1100 | `[+0.0138835,+0.0139021]` | `[+0.013884,+0.0139021]` | `[+0.0139098,+0.0139286]` | `[+0.013910,+0.0139286]` |
| 1104 | `[+0.0422487,+0.0422706]` | `[+0.042250,+0.0422706]` | `[+0.0420723,+0.0420932]` | `[+0.042073,+0.0420932]` |
| 1108 | `[+0.0810506,+0.0810709]` | `[+0.081052,+0.0810709]` | `[+0.0810966,+0.0811184]` | `[+0.081097,+0.0811184]` |
| 1112 | `[+0.1373012,+0.1373190]` | `[+0.137303,+0.1373190]` | `[+0.1371288,+0.1371498]` | `[+0.137130,+0.1371498]` |

These are overlaps, **not containment of B in A**, and not two certificates.

**Inner root — interval verdict unresolved in every row; reported signs agree.**

| \(t\) | P: A interval; B rounded value | H: A interval; B rounded value |
|---|---|---|
| 1096 | `[-0.073261,-0.068273]`; `-0.07329` | `[-0.073853,-0.068805]`; `-0.07386` |
| 1100 | `[-0.077477,-0.068678]`; `-0.07749` | `[-0.076098,-0.067448]`; `-0.07610` |
| 1104 | `[-0.078590,-0.072259]`; `-0.07860` | `[-0.078488,-0.072145]`; `-0.07850` |
| 1108 | `[-0.080168,-0.076083]`; `-0.08018` | `[-0.081045,-0.076905]`; `-0.08105` |
| 1112 | `[-0.084720,-0.076500]`; `-0.08473` | `[-0.083803,-0.075661]`; `-0.08381` |

B’s printed representatives lie \(2\)–\(29\times10^{-6}\) below A’s lower endpoints. **Without B’s widths, this proves neither overlap nor disjointness.** Export both routes’ unrounded inner-root \(H_R\) endpoints and test  
\(\max(A_-,B_-)\leq\min(A_+,B_+)\) separately for all ten pairs.

### 2. Paper text supportable now

The fully interval-filled, all-root FINAL template cannot be completed: `02_AstraAPI_rdtext_v1.md` requires **“obtain unrounded enclosures”** and **“Check interval overlap before rounding.”** The following replacement is defensible now, **with the tables above included**; it does not claim that the missing comparisons passed.

> At the five stored M1 clock records \(t=1096,1100,1104,1108,1112\), over \(T/r_s=14.9\)–15.1, we evaluate the instantaneous Hartree root debit per infinity-normalised lattice time for the implemented **32-source/16-reference light law (half-weight light counterterm; the paper’s reference weight is 32)**. All four endpoint slots are flagged and contribute zero; identification with the paper’s \(W_2\) is not certified. The clock is normalised by \(N=1\) at \(y=8240\).
>
> We report separately **P**, piecewise-linear \(\widehat M\), and **H**, cubic Hermite of \((f_j,f'_j=-D_j)\), with
> \[
> \widehat D_R=1-2G\,\partial_r\widehat M(R),\qquad
> H_R=-\frac{\dot{\widehat R}}{2G}
> =-\frac{\partial_t\widehat M(R)}{\widehat D_R}.
> \]
> Route B’s outer-root enclosures are listed above: under both interpolants the debit is negative at \(t=1096\), indicating expansion, and positive at the four subsequent records, indicating recession, increasing from approximately \(+0.0139\) to \(+0.1373\). All ten printed outer-root intervals overlap Route A’s provisional intervals. This is numerical corroboration of Route B, not a second certificate.
>
> Route B also reports certified negative inner-root debits at all five records, indicating outward motion; its rounded P/H values are listed above. Their interval agreement with Route A remains unresolved because the supplied extract omits the inner-root enclosures.
>
> **Condition (a): pending at all five records.** The Hartree-branch interpretation remains conditional on that verification; this label may be lifted when the corresponding certificates pass.
>
> **[to do: complete condition (a), propagate its uncertainty where required, and compare the unrounded inner-root debit enclosures under P and H; reconcile any disjoint pairs]**

### 3. Red-mark disposition

**OPEN for the requested all-root two-route closure; the outer-root component is narrowed to the pending verification.** Neither the template’s blanket-agreement case nor its established-disagreement case is justified yet. Passing (a) alone will not resolve the missing inner comparisons.

These stored-record results establish neither a continuum horizon or globally outermost root without exterior exclusion, nor between-record continuation or depletion, a heavy-matter-free root with bounded collar remainder, full-weight-light evolution, a retarded-time Hawking rate, funded complete release, or Page certification.