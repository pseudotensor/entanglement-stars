
### (a) The printed observable needs a definition

**The literal “single-amplitude” reading is \(\rho_{R3X}\), not the denominator-weighted \(\rho_F\).** But neither ratio nor its implementing function is defined in the supplied material, so the mapping is not formally checkable.

In `00_P3_current_nc.tex`, the comparator paragraph says “squared single and double amplitudes **over their excitation energies**”: amplitudes are the numerators. It separately calls the state correction “first-order dressing.” The marked paragraph says “Fock single amplitude,” while \(q_\infty=4\langle(\widehat E_2/E-I)^2\rangle\) also contains no excitation denominators.

Thus **do not use the falling dressing ratio to answer a claim about the undivided amplitude ratio**. If \(\rho_F\) is intended, change the wording to “denominator-weighted single-dressing ratio” and supply its formula.

### (b) What survives

- Your check (3) supports **rising \(\rho_{R3X}\)** and **falling \(\rho_F\)** on the tested, partly adiabatic chain construction. It supports no \(K^{1/2}\) law.
- The printed “no channel limit **is established**” remains correct. It does not assert divergence. The unqualified growth mechanism and comparison with \(16k\), however, should be replaced by the actual diagnostic and tested range.
- The cubic coefficient is a **local linear-response result**, not yet a measurement of the full propagated-sea single ratio. `00_P3_current_nc.tex` explicitly says the chains share the cubic operator’s “continuum spherical limit but not its zone.”
- The distinction matters numerically: your rates-off ratios imply an **8.6%** change for \(\rho_F\), but **14.8%** for \(\rho_{R3X}\). The latter fails the stated 10% control.

**Exact replacement text for the growth clause:**

> On the tested \(T/r_s=11.0\) slice, the single-amplitude ratio without excitation denominators grows approximately linearly with the chain cutoff \(K=10\)–40, whereas the denominator-weighted ratio decreases; channels above \(k=10\) use rates-off adiabatic seas, so this is not a propagated-sea tail test. The cubic small-wavevector response at \(ma=0.2\), with no first-order clock-gradient term, gives directional coefficients \(c_\rho=(0.438,0.420,0.413)\) and independently \((0.4382,0.4201,0.4134)\), in the quoted per-\(a\) normalization, along \([100]\), \([110]\), and \([111]\). On ten chains with weights \(16k\) at \(T/r_s=11.0\), the history remainder has \(0.213\) times the adiabatic Fock-single norm, with the additional adiabatic correction changing this ratio by \(10^{-4}\).

Keep the mark after this text. The supplied history result is described as a chain-code calculation, not an independently reproduced result.

**Scaling:** I would not yet attach the numerical coefficient to either *full* ratio as
\[
\rho\sim c_\rho\,\frac{G}{a^2}\frac{a/R_*}{N_*}.
\]
That is plausible **local adiabatic power counting**, provided \(R_*\) normalizes the tested flow gradient, \(N_*\) is the relevant positive clock, and \(c_\rho\) uses exactly the same single norm and Hartree normalization. Those identifications are not supplied.

The decisive measurement is
\[
c_{\rm eff}=\rho_{\rm ad}\,N_*\frac{R_*}{a}\bigg/\frac{G}{a^2}
\]
in the chosen cubic diagnostic as the gradients decrease. Also measure the history–adiabatic interference: \(0.213\) is a norm ratio, not a 21.3% fraction of the total or an automatic multiplicative correction.

### (c) Closing the mark

`01_requirements_new.txt` requires “marks are closed by the planned work landing, never by cutting the claim.”

- **\(15.0\):** needed before extending the history/adiabatic conclusion to the other printed slice.
- **Actual propagated \(K=20\):** useful for testing the rising per-channel history remainder and the stand-in approximation; it cannot validate the cubic zone.
- **Cubic calculation:** the two-route response calculation closes the *coefficient* question, not the full history-single question.

**Cheapest decisive work for the named mark:** after fixing the observable’s formula, evaluate that same single block and Hartree normalization in two independent **one-way cubic calculations on the stored fields**, including the propagated sea or a controlled bound on its history remainder. A coupled cubic collapse is unnecessary for that restricted diagnostic. Extra chain cutoffs alone cannot close it.