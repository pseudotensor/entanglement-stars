
reviewer A: **ACCEPT the verdicts, but not all the sharpenings.** My counters below use only the supplied text, not a new evaluation of the underlying records.

### Sharpenings

All reviewer A quotations here are from `03_FableSub_vrcsign_v1.md`.

- **(a) Vertex: ACCEPT DISAGREE and the common-record-state calculation. COUNTER** “the operators differ, not the states”:
  > If the baseline discrepancy persists on identical fields and covariance, base-state choice does not explain it; compare operator definitions, normalization and numerical evaluation before attributing it specifically to different operators.

- **(b) Rate: ACCEPT NOT COMPARABLE and the signed channel-sum calculation. COUNTER** “It will land near minus 2.46e-4”:
  > A’s uniform bound does not determine its signed channel sum. Compute that sum with unit weights and propagated errors on the reconciled base state; neither its central value nor disagreement with B follows from the bound alone.

  `02_AstraAPI_vrccmp2_v1.md` explicitly quotes A’s result as “`|dGamma|/Gamma0 <= 2 eta + eta^2`”.

- **(c) Fixed radius: ACCEPT AGREE, the common-input qualification and the stronger central-value comparison**, with approximate precision understood. From the printed values in `02_AstraAPI_vrccmp2_v1.md`, the relative differences are approximately \(1.04\times10^{-4}\), \(1.026\times10^{-3}\) and \(1.27\times10^{-5}\). Thus **COUNTER any strict “part in a thousand or better” claim**:
  > The central values agree to about a part in a thousand or better; the shared field-rate uncertainty is not an independent cross-check.

  **Moving root: ACCEPT NOT COMPARABLE and A’s species-resolved radial derivatives as the deciding calculation**, provided both routes use the same \(\dot R\), radius, decomposition and propagated uncertainties. The supplied material does not contain the five roots needed to independently reproduce the quoted \(\dot R\).

- **(d) Collar: ACCEPT** the sign verdict, the opposite width trends and evaluating A’s \(C_1\) on B’s state as the next diagnostic. The quoted “target minus 3.255e-3 within about 5e-5” is a diagnostic target, **not** certification of a finite-width replacement. Matching the windows and bounding their remainder remain necessary.

- **(e) Cubic \(K\): ACCEPT** the symmetric-strain map and transverse-wave diagnostic. The rigid-turn interpretation is consistent with `00_PAPER_flat_bdca0bd.tex`, which says the spin-completed turn “commutes with \(h_0\)”; its numerical cancellation must use that complete response. **COUNTER** “the grid matters only at m = 0”:
  > Agreement of \(c_{11}\) does not bound quadrature differences in \(c_{12}\) or shear. At \(ma=0.2\), compare all three unrounded strain coefficients on the same explicit momentum grid, with errors; at zero mass, additionally use a grid off the doubler nodes and enclose the limiting coefficient.

### Amended paragraph: COUNTER, with these minimal corrections

**ACCEPT** dropping the cancellation sentence: `00_PAPER_flat_bdca0bd.tex` already says “the reference response cancels for the total boundary generator, not for a windowed mass.”

Do not assert identical evaluation fields: `02_AstraAPI_vrccmp2_v1.md` reports B’s denominator on “`production fields`” but its reference on “`the re-solved state`”. Use shared **inputs**, and approximate central agreement:

```latex
On one stored two-field record at \(t/a=1104\), where the outer
trapping radius is \(R/a\simeq69.47\), two independently implemented
evaluations using shared stored-history inputs and field-rate
estimates agree to about a part in a thousand on the subtracted
reference's contribution, at fixed radius, to the rate of change
of the enclosed mass. That contribution has the sign of the rate
and is about 8.6 percent of it, between 7.9 and 9.5 percent under
the reported record-spacing errors; 92 percent comes from the
heavy field, and the light field's own part is 0.7 percent of the
rate, so the enclosed-mass balance retains it. This is a
frozen-record diagnostic, not a re-evolved balance; the balance
at the moving root and along the history is not established.
```

### Amended `\todo`: COUNTER only to retain the massive matched-grid test

```latex
\todo{two independent expansion calculations of the three inputs
above; for the mixed heavy-light vertex's reference correction, on
which the two routes disagree, both routes on one base state with
the signed rate at identical weights; for the root debit, the
moving-root term, the collar in one window form with its
finite-width remainder bounded, and the reference term at the other
sampled clocks; the cubic tensor's strain map stated, its three
unrounded strain coefficients compared with errors on a common
momentum grid, and its massless coefficient enclosed on a grid off
the doubler nodes; funded complete release; uniform decoding and
response-error bounds}
```

### Insertion condition: ACCEPT; insertion remains blocked pending metadata

`00_PAPER_flat_bdca0bd.tex` still states, “At \(r_s/a=64\), the two-field history is not formed.” Do not silently identify this record with that setup.

Place the following labelling sentence **immediately before the VRC paragraph**, beside the “Three common inputs” passage. Brackets are required metadata, not inferred values:

```latex
This diagnostic uses history \texttt{C\_M1nl23}, with nominal
\(r_s/a=[s]\), \(m_Ha=[m_Ha]\), \(m_La=0\), \(G/a^2=[g]\),
heavy and light channels [channel sets and copy weights], averaging
[ kernel and width ], and Gauss-law sources [precise prescription
and whether light sources are included]; its record \(t/a=1104\)
corresponds to flat-slice clock \(T/r_s=[clock]\).
```

If \(t=T\) with the same origin, the clock is \(1104/s\), hence **17.25 only if \(s=64\)**. Reconcile the earlier “not formed” sentence against the documented setup and source prescription. Finally compile and count all pages against `01_requirements_new.txt`’s “at most 29 pages including every appendix”; that count cannot be checked from the flattened text.