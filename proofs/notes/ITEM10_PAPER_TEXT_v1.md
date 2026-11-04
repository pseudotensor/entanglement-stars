# Item 10 Hessian-term sentence: paper text for paperapi (v1, status (b))

From busi, per lead #1273. It replaces the clause "the computed Hessian's error entering ... assumption of the reading".
- Text: reviewer B astra_busi_item10compA_20261104_100911 (645d3b4b), sec. 3, sentence (b) VERBATIM, followed by reviewer B's sec.-4 appendix sentence. Both were extracted by awk.
- busi changed ONE phrase in the appendix to match the run actually performed: "from 60 Gaussian probes at 99.9% confidence per row" became "from 54 to 60 Gaussian probes per row at 99.9\% joint confidence over the three rows".
  - Run facts: codeB run (1), probe_combine_v1.json dfcb6257. The K probes are m = 54 and the dB probes m = 60. The joint bound is a union over 6 estimators at a = 1.667e-4 each, which meets reviewer B's joint-confidence rule. The result is eps_V,F <= 7.09e-9.
- Premises checked: the per-channel composition (reviewer A b9d7441a, reviewer B 645d3b4b). codeB confirmed from code s46v3_mass2_v1.py 44f4761f that the nominal operator is fully centered with the full trace and that its quadratic grade is the Fock term alone. So L_safe <= 431 applies.
- Status (b) prints now. A reduced S2 may upgrade this to (b'), "certified <= 0.080", by about 14:00Z; busi will send a replacement if so.

----- BEGIN TEXT -----
The reported 18-slice inequality \(\langle\mathcal B\rangle\le0.080\) and the associated 0.92 overlap are computed-operator readings. At \(T/r_s=15\) on the \(\sigma/a=4\) run, the analytic bound \(\|Q_{\delta V_j}\|_{\bar C}\le431\|\delta V_j\|_F\) controls the complete centered Hessian perturbation, but the full normalized error budget has not been certified; no enclosure for the exact finite-map diagnostic is claimed.

The closed-form–AD discrepancies have Frobenius upper bounds obtained from 54 to 60 Gaussian probes per row at 99.9\% joint confidence over the three rows, with AD taken as the numerical reference; these are probabilistic discrepancy bounds, not enclosures of AD’s own error or of the complete finite-map error.
----- END TEXT -----
