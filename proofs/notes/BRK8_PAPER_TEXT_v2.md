# BRK-8 storage / positive-deposit paragraph: FINAL paper text for paperapi (v2: ONE token patched, 4,711.5 -> 5,256.3, per reviewer A 458bd4d5 and reviewer B 8e9c3e2b)

From busi, per lead #1267 and #1270. Both reviewer B and reviewer A signed it.
- Text: reviewer B round 4 (api/brk8_deposit/answer/astra_busi_brk8depA4_20261104_083717.md, 42969a17), sec. 3, VERBATIM.
- Exception, one clause: reviewer A round 4 (fable_busi_brk8depF3_20261104_091749.md, b203d218, sec. 1) replaced the box antecedent, and reviewer B signed that clause in astra_busi_brk8depA5_20261104_093044.md (e432d348).
- The text below was produced from those files by sed and perl. Nothing was retyped.

Where it goes:
- It replaces the paper's BRK-8 storage-comparator paragraph.
- Delete the floor sentences (18,510 nats, 0.678, 6.78) and the Q_V "common bound" clause from this storage argument. All rounds agreed that.
- Do not write "the comparator's storage law is this inequality".
- No clock label goes beside the 4,711.5: record k108 is T/r_s = 15 in BRK-4 and 13.5 in the D3 note. The lookup to reconcile them is running (pmixA), and the text names the slice by R_t.

----- BEGIN TEXT -----
At product Hartree order the light field’s state is pure. For a complete orthogonal received–retained partition, with retained generator levels \(\mu_j\), occupations \(n_j\), and a declared binary filling \(b_j\),
\[
S_{\rm esc}=S_{\rm ret}\le
\inf_{\beta\ge0}\left\{\beta E_{\rm dep}
+\sum_j\ln(1+e^{-\beta|\mu_j|})\right\},
\qquad
E_{\rm dep}=\sum_j|\mu_j|\,|n_j-b_j|.
\]
Define \(E_s^B=\sum_j\mu_j(n_j-b_j)\) and let \({\rm SR}\) contain the levels with \(b_j\ne\mathbf1_{\mu_j<0}\). Then
\[
E_{\rm dep}=E_s^B+2E_{\rm SR}^{\rm occ},\qquad
E_{\rm SR}^{\rm occ}=\sum_{j\in{\rm SR}}|\mu_j|\,|n_j-b_j|,
\qquad E_{\rm dep}\ge|E_s^B|.
\]
The signed term must be reconciled with the reference-calibrated charge ledger using the same weighting and retaining reference, work, interface and numerical remainders. To use the once-counted received energy \(E_{\rm rec}\) as the Gibbs budget, the comparator additionally assumes positive-deposit affordability,
\[
E_{\rm dep}\le E_{\rm rec},
\quad\text{equivalently}\quad
E_{\rm SR}^{\rm occ}\le\tfrac12(E_{\rm rec}-E_s^B).
\]
This is not a consequence of signed charge conservation. If only an interior mode subset is counted, an entropy bound for every other retained mode, including radiation in flight, must be added. The certified ceiling of \(5\,256.3\) nats at the assigned total positive-cost budget \(23.9/a\), at 99.9 percent confidence, applies to the massless light generator on an open \(131^3\)-site box at the stored fields of the slice with fine-grid trapping radius \(R_t=49.48a\), every one-particle level of the box counted at its \(|\bar h|\) cost, and establishes neither that budget for the complete retained field nor the comparator’s area-sized accessible store [to do: BRK-8 D-a on the light history with the received-packet basis after first arrival].
----- END TEXT -----
