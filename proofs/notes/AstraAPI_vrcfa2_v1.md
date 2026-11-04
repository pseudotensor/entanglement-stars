
reviewer A:

### Decisions on your three counters

1. **(e): ACCEPT dropping “on a common momentum grid.”** I withdraw that requirement at \(ma=0.2\). Independent quadratures with controlled errors suffice.

   One qualification: `03_FableSub_vrcfa2_v1.md` says the strip implies that “every equispaced 128-point periodic rule, shifted or not, is converged to about 1e-11.” The strip width supplies the exponential rate, not an absolute error bound without its prefactor. I have not independently measured those errors. To substantiate that numerical precision, supply a quadrature bound for each coefficient, or report all three unrounded coefficients under refinement, for example \(128^3\) to \(256^3\), with the error model stated. The final `\todo` retains **“compared with errors”**; it need not prescribe identical grids.

2. **Paragraph wording: ACCEPT both changes.** Use “a diagnostic on one stored record” and “the mass inside the surface.” This avoids a second meaning of “frozen” beside `00_PAPER_flat_bdca0bd.tex`’s “lower band of the frozen principal symbol,” and expresses its defined mass function, `\mathcal M=\frac{rw^2}{2G}`, plainly.

3. **Labelling sentence: ACCEPT parameter labels instead of a run name.** `01_requirements_new.txt` requires “every run labelled by its dimensionless parameters” and excludes “card facts and code detail.” The supplied metadata now resolves the missing labels, including the source convention.

   In particular, `04_solve_meta.txt` states **“LIGHT SOURCES ARE INCLUDED”** and **“the light carries no reference term.”** Thus the light normal-ordered density gravitates in this history; the light polarization correction evaluated by the diagnostic was not included in that evolution. I make that distinction explicit below.

### Binding fraction and clock

The **release** binding fraction is derivable. `04_solve_meta.txt` gives “release charge radius 2GE0 = 73.5428 a (E0 147.08566),” “m_H a 0.2,” and “N_B 800.” Therefore
\[
\eta_E(0)=\frac{E_0a}{(m_Ha)N_B}
=\frac{147.08566}{0.2\times800}
=0.919285375.
\]
Print \(\eta_E(0)\simeq0.919285\), explicitly at release. A **record-time** binding fraction would require \(E(t/a=1104)a\), which this metadata does not supply.

The same file gives “NOMINAL r_s/a = 2 G m N_B = 80” and confirms the same clock origin, so \(1104/80=13.80\), **not 17.25**.

## AGREED

The following is the final insertion, with the labelling sentence immediately before the diagnostic paragraph. The paragraph retains the reported numerical findings from `03_FableSub_vrcfa2_v1.md`; this response does not independently recompute them.

### FINAL labelling sentence

```latex
This diagnostic uses a two-field history at nominal
\(r_{s,\mathrm{nom}}/a=80\), \(m_Ha=0.2\), \(m_La=0\),
\(4\pi m_Hr_{s,\mathrm{nom}}=64\pi\), \(G/a^2=1/4\) and
release binding fraction
\(\eta_E(0)=E_0/(m_HN_B)\simeq0.919285\), with \(N_B=800\),
evolved at step \(\tau/a=2\) with Gaussian averaging of sources
and back-averaging of fields at \(\sigma/a=4\), five heavy
channels \(k=1,\ldots,5\) with occupied-column counts
\(n_k=(6,5,4,3,2)\) and copy weights \(16k\), and one light
channel \(k=1\) with full-chain copy weight \(w_l=16\);
the Gauss-law sources contain both fields' densities normal
ordered against their release seas and subtract the heavy
field's change in second-gradient polarization from preparation,
as in Eq.~\eqref{eq:dm-reference}, but contain no light
polarization counterterm, which this diagnostic evaluates;
the record \(t/a=1104\) corresponds to flat-slice clock
\(T/r_{s,\mathrm{nom}}=13.80\).
```

The full-chain light weight is deliberately **16**, not 32: `04_solve_meta.txt` distinguishes “w_l 16 on the full chain” from the evolved symmetry block’s “2 w_l = 32.” Those are alternative representations, not additional multiplicities.

### FINAL paragraph

```latex
On one stored two-field record at \(t/a=1104\), where the outer
trapping radius is \(R/a\simeq69.47\), two independently
implemented evaluations, sharing the stored history and its
field-rate estimates as inputs, agree to about a part in a
thousand on the subtracted reference's contribution, at fixed
radius, to the rate of change of the mass inside the surface.
That contribution has the sign of the rate and is about
8.6 percent of it, between 7.9 and 9.5 percent under the reported
record-spacing errors; 92 percent of this contribution comes
from the heavy field, and the light field's reference part is
0.7 percent of the rate.
This is a diagnostic on one stored record, not a re-evolved
balance; the balance at the moving root and along the history
is not established.
```

### FINAL `\todo`

```latex
\todo{two independent expansion calculations of the three inputs
above; for the mixed heavy-light vertex's reference correction, on
which the two routes disagree, both routes on one base state with
the signed rate at identical weights; for the root debit, the
moving-root term, the collar in one window form with its
finite-width remainder bounded, and the reference term at the other
sampled clocks; the cubic tensor's strain map stated, its three
unrounded strain coefficients compared with errors, and its
massless coefficient enclosed on a grid off the doubler nodes;
funded complete release; uniform decoding and response-error
bounds}
```

### The size-64 sentence

**No substantive change is required, and its `\todo` remains open.** `04_solve_meta.txt` explicitly answers the setup-identification question **“NO”** and states, “it does NOT literally discharge the paper's size-64 todo.” Its comparison identifies the size-64 setup as \(N_B=640,\tau=8a\), whereas this history has \(N_B=800,\tau=2a\) and nominal size 80.

Consequently, your conditional rewrite to “the two-field history with the light field's reference sources is not formed” is unnecessary. The smallest optional clarification is to add **“nominal”**:

```latex
At nominal \(r_s/a=64\), the two-field history is not formed
\todo{re-evolve the two-field size-64 history with
Eq.~\eqref{eq:dm-reference}'s sources}.
```

The remaining production check is to compile the revised manuscript and count all pages, including appendices, against `01_requirements_new.txt`’s “at most 29 pages including every appendix”; the flattened text cannot establish that count.