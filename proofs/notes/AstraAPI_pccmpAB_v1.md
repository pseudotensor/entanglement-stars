
## 1. Comparison: **not blanket two-route agreement**

**Normalization matters.** In `"00_routeA_all.md"` [A], the columns `"w_k"` and `"w_k E_bar[P_h2]"` give \(w_A=16k\). `"01_C1_ROUTE_B_RESULT_v6_v1.md"` [B] specifies `"sum_k 32k x + sector"`. Thus compare weighted contributions—or, equivalently, **A/2 against B**, subject to confirming the sector identification.

The rule in `"02_FableAPI_c1thr_v1.md"` [R] is `"agreement within the union of the two envelopes plus numerical floors"`; the large \(2n_A\ln2\) caps are **not** agreement tolerances.

### Channel-matched pairs: \(T/r_s=100\)

All numbers below are **\(10^{-6}\) nats**, using [A]’s `"E_bar[P_h2] (ruled, headline)"`, `"env min"`, `"env max"` and [B]’s `"E-bar v6"`, `"envelope"`, `"bound"`.

| \(k\) | A/2: average [envelope] | B: average [envelope] | B bound | Result |
|---:|---:|---:|---:|:---|
| 1 | 236445.066 [236443.509, 236445.634] | 237900 [238000, 238000] | 0.16 | D |
| 2 | 14018.696 [13959.187, 14049.224] | 13990 [14000, 14000] | 0.18 | O |
| 3 | 2141.236 [2141.180, 2141.392] | 2186 [2180, 2190] | 0.27 | D |
| 4 | 430.885 [430.519, 431.022] | 397.7 [397, 399] | 0.27 | D |
| 5 | 57.376 [54.421, 63.185] | 77.01 [76.8, 77.3] | 0.33 | D |
| 6 | 29.915 [29.907, 29.941] | 22.83 [22.7, 23.0] | 0.30 | D |
| 7 | 3.3777 [3.3472, 3.4717] | 7.076 [7.04, 7.11] | 0.35 | D |
| 8 | −0.0332 [−1.2992, 0.0525] | 2.203 [2.18, 2.24] | 0.38 | D |
| 9 | 5.6407 [5.6106, 5.6765] | 2.205 [2.16, 2.24] | 0.38 | D |
| 13 | 0.5394 [0.4852, 2.5606] | 1.307 [1.29, 1.31] | 0.54 | O |
| 14 | 3.3958 [3.0152, 3.4879] | 1.310 [1.28, 1.32] | 0.59 | D* |
| 15 | 0.2492 [−0.7089, 0.3526] | 1.129 [1.13, 1.14] | 0.60 | D* |
| 16 | 2.7894 [2.6154, 3.1173] | 1.127 [1.12, 1.13] | 0.64 | D* |
| 17 | −0.5040 [−0.6766, 0.1966] | 1.146 [1.14, 1.15] | 0.64 | D* |

- **O:** envelopes overlap: **\(k=2,13\)**.
- **D:** envelopes remain disjoint after adding B’s tabulated one-bound floor. B’s coarse endpoint rounding does not change these classifications.
- **D\*:** overlap becomes possible using **three** B bounds. But [B]’s `"RESOLVED means |E| > 3 x the numerical bound"` is a significance test, not a declared three-bound agreement tolerance.
- **A supplies no numerical floor values.** Therefore a definitive combined-floor verdict requires unrounded envelopes and \(b_A(k,100)\), testing envelope separation against \(b_A/2+b_B\). `THR_KAT` is not that floor.

### Shared clocks and coverage

Let \(K=\{1,\ldots,9,13,\ldots,17\}\). Summing [A]’s `"w_k E_bar[P_h2]"` gives:

| \(T/r_s\) | A subtotal, \(K\) | B whole field, \(1\ldots18\) |
|---:|---:|---:|
| 12 | −0.235665 | −0.2074 |
| 20 | −0.158927 | −0.0745 |
| 40 | −0.111152 | +0.1112 |
| 100 | +8.744466 | +8.7923 |

**These are unequal-coverage numbers, not valid agreement pairs.** B supplies no channel breakdown at 12, 20 or 40. At the other shared clocks—30, 44, 46, 50, 56, 60, 66, 70, 76, 80, 86, 90, 96—A supplies only \(k=1\), whereas B supplies whole-field totals.

At **100**, a matched subtotal *can* be reconstructed:

- **A, \(K\):** \(8.744466\), envelope \([8.738906,8.748939]\).
- **B, \(K\):** approximately \(8.7899\), envelope approximately \([8.7867,8.7921]\), subtracting B’s \(k=10,11,12,18\) contributions from its whole-field result.

Those envelopes are disjoint by about **0.0378 nats**, versus B’s quoted whole-field numerical bound **0.0028 nats**. Exact reconstruction requires B’s unrounded reduction output, identified in [B] as `"pcb_reduce_v6_v1.py 71a8dc10"`.

## 2. Disagreement and source

**Do not average it away.** The printed weights explain a bookkeeping factor of two, **not the remaining mismatch**.

The leading audit target is the **implemented escaped subspace/reference alignment**, not the energy convention:

- [A], computed by `"pc_c1_thr_v1 d36f58a7"`, reports at \(k=1,100\): `"n_+” = 8072`, `"S_bar[P_h2]" = 3.7254579976`.
- [B] reports `"S_ref v6" = 5.905012e-02` there. Although this B entry is hard-cut rather than window-averaged, the reference-entropy scale difference survives the weight normalization and demands investigation.
- [R] requires `"the same χ_far, X, s and P_h"`; [B] still labels A1–A3 `"open"`.
- [B] measures `"|shift E| <= 3.0e-6 nats"` whole-field and approximately \(2\times10^{-9}\) per copy at 100: far too small to explain the principal discrepancies.

**The exact cause is not established by these extracts.** Measure, on identical sector-normalized inputs: escaped-projector ranks, spectra and principal angles; the held800 reference operator; and every ladder value \(S_i[C]\), \(S_i[P_h]\). Also obtain A’s numerical floors and the missing channel/clock outputs. This distinguishes projector, reference, covariance/propagation and multiplicity errors.

## 3. Printable wording

> On the assumed stored implemented background, two routes evaluate the c1thr-window escaped-mode excess; these diagnostics are not certified self-consistent Page curves. At \(T/r_s=100\), Route B gives a whole-field excess of \(8.7923\) nats, with threshold envelope \([8.7891,8.7945]\). On the commonly reported channels \(k=1\!:\!9,13\!:\!17\), Route A gives \(8.74447\,[8.73891,8.74894]\) nats, whereas Route B gives approximately \(8.7899\,[8.7867,8.7921]\) nats; two-route agreement therefore remains unestablished.

This retains the requested **assumed-store** qualification; the actual emission-defined manuscript passage was not supplied for a textual consistency check.

## 4. What the pair does not support

**Neither a Page turnover nor a Page time.** [B] explicitly states `"E_esc can be negative and is not an entanglement entropy"` and says the 66 bump `"stays as a feature of the V2 subspace ... to be looked at under A3"`. Local peaks or decreases in these uncertified implemented-background diagnostics are not evidence of an emission-defined Page turnover.