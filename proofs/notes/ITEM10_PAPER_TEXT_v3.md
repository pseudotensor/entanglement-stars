# Item 10 Hessian-term text (v3, final): conditional result from the S2 combine

reviewer B astra_busi_item10finalA (88a922d2) VERBATIM, extracted by awk. It supersedes v1 and v2.
- Basis: s2r_combine_v4.json 73a76ab3: e_ub = 1.27e-7 against e_print,B = 3.45e-5 (gate a passes), [A_B, B_B] = [0.079788944765, 0.079791443410]. The endpoints are printed outward.
- Run (1): eps_V,F <= 7.1e-9 at 99.9% joint. L_safe <= 430.52 (per-channel composition, signed by reviewer A and reviewer B). The fully centered nominal operator was checked in code.
- The owner word rule and the one-hedge-sentence rule are both met: the conditions are in one sentence.

----- BEGIN TEXT -----
At \(T/r_s=15\), \(\sigma/a=4\), the separately evaluated finite-map diagnostic \(\langle\mathcal B\rangle\), using the reference of Eq.~\eqref{eq:dm-reference}, rounds to \(0.0798\) (four decimal places). The S2 bounds \([A_B,B_B]\subseteq[0.079788944765,\,0.079791443411]\) and \(e_{\rm ub}<e_{\rm print,B}\) establish this conditionally on float64 AD-reference accuracy without an AD-error bound (A1), a valid measured-slope-\(\times10^3\) allowance for the root shift and its propagated effects (A2), the stored slice orbitals defining the state (A3), and a valid mass-replay error allowance (A4), with \(99.9\%\) joint confidence only for Hessian discrepancy bounds relative to AD and otherwise float64 a-posteriori, non-interval bounds.
----- END TEXT -----
