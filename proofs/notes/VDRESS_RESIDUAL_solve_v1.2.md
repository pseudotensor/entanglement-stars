# VDRESS beyond-Hartree residual, route 2 (solve), v1

- Spec: lead_gaps/VDRESS_CALC_SPEC_v1 (preregistered thresholds and predictions).
- Notes: lane `results/vref_r2/vdress/NOTES_vdress.md` (md5 0213c2db3775666946b9e660b63143a2). JSON: vdress_S11_v1.json, vdress_S15_v1.json, fock_check_seed7/8.json.
- Blind to route 1.
- Slices: stored 11.0 and 15.0, interior rows, collar excluded.

## Identity checks (PASS to roundoff)
**Exact re-Wick reconstruction of E_2 about C-bar.**
- Setting: explicit Jordan-Wigner Fock space, 6 modes, seeds 7 and 8, with normal ordering done independently about P and about C-bar.
- The P-ordered and C-bar-ordered forms agree to 9e-14.
- <E_2>_Cbar = E_HF to 3e-14.
- The residual identity (residual norm squared = r_F^2 + r_D^2) holds to 3e-13.

**Cancellation of r_F against the direct singles.**
- In the Fock space, the quartic's singles exactly cancel the direct Hartree singles, leaving h - F, to 3e-14.
- On the stored slices (channels 1 and 10, both slices), the P-ordered gradient equals the C-bar-ordered one to 1e-15.
- The cancelled direct-singles norm is 0.34 at 11.0 and 1.21 at 15.0. The remaining r_F is 1.6e-3 and 1.7e-3.

## Readouts vs preregistration

| readout | 11.0 | 15.0 | threshold | prediction |
|---|---|---|---|---|
| (a) r_F/r_H | 7.1e-3 | 1.5e-3 | PASS | missed (predicted < 1e-3) |
| (b) r_F/r_B | 5.5e-3 | 3.6e-3 | not refuted (refutation >= 0.3) | inside the 1.6e-3..0.04 band |
| (c) mass rate at R50 | 1.1e-3 | 1.3e-3 | PASS | slightly missed |
| (c) mass rate at r_in | n/a (no roots) | 0.8e-3 | PASS | met |
| (c) mass rate at R_t | n/a | 2.5e-3..3.9e-3 | PASS | missed |
| <P\|B\|P> | >= 3.79 | >= 7.61 | - | reference outside the branch, as expected |

Notes on the readouts:
- 11.0 has no trapping roots, so the clause "all three radii on both slices" is only fully testable at 15.0.
- The R_t range reflects a small-denominator sensitivity.
- (a) at 11.0 is the closest call: a factor of 1.4 inside its threshold.

## (d) r_D by hole content at channel cutoffs 10/14/20: NEEDS A RULING
Class by class:
- **Two dust holes:** 0.0215 (11.0) and 0.049 (15.0), flat with cutoff. The flatness holds by construction, so it is not a real test: the stored history has 10 channels, channels 11-20 were added as dust-free reference sea, and V was held at its 10-channel value.
- **One sea + one dust hole:** grows 1.7-1.8x from cutoff 10 to 20 (0.047 -> 0.082 at 11.0; 0.036 -> 0.066 at 15.0).
- **Two sea holes:** grows with cutoff, as printed; no threshold applies.

Outcome by reading of the spec:
- If "dust-hole" includes the mixed class, (d) FAILS on both slices.
- If it means two dust holes only, (d) passes trivially.

A genuine cutoff test needs histories actually run with 14 and 20 channels.

## Overall
- The Hartree level is NOT REFUTED on either slice.
- The preregistered pass holds on (a) and (c) wherever the radii exist.
- (d) depends on the reading of "dust-hole".
- The self red-team in the notes covers the extension-channel choice, the R_t denominator sensitivity, and the fact that a snapshot pass is not a trajectory bound.

---
# v1.1 addendum (ruling #556; notes s8, md5 58754e5a52228849c0730e3575a9b183). Supersedes the (a) and (d) statements above where they differ.

## (1) Zero-cost decomposition of the mixed sea+dust class: EQUAL
The cutoff 10 -> 20 increment of r_mix^2 equals the contraction with Sigma(20) - Sigma(10) to roundoff:

| slice | increment (both sides) | relative difference |
|---|---|---|
| 11.0 | 0.0044660481727902 | 8e-15 |
| 15.0 | 0.0030797010023085 vs ...084 | 1.3e-14 |

- The flat (same-copy) exchange part has the same bound at both cutoffs (2.4e-6, 1.4e-6).
- Reading: the growth is the sea's fluctuation dressing the dust excitation; there is no excess dust correlation.
- Caveat: with the extension channels as reference sea and V frozen, this equality follows from linearity. It checks the decomposition and the code, not physics.

## (2) Genuine 14- and 20-channel formation-control histories
These are the W5 k14/k20, tau 8. They DO store the 11.0 slice, so they were evaluated without any run, each on its own slice, fields and rows. All gates pass (640.0, sources 1e-13, reference = negative projector, 0 inversions).

| readout at 11.0 | k10 | k14 | k20 |
|---|---|---|---|
| (a) r_F/r_H | 7.10e-3 | 8.98e-3 | **1.30e-2 FAIL** (> 1e-2) |
| (b) r_F/r_B | 5.5e-3 | 5.3e-3 | 5.9e-3 |
| (c) R50 mass rate | ~-1.0e-3 | ~-1.0e-3 | ~-1.0e-3 |
| (d) dust-dust r_D | 0.0215 | 0.0215 | 0.0217 |
| sea-dust at own cutoff | 0.047 | 0.063 | 0.083 |
| <P\|B\|P> | >= 3.78 | >= 3.78 | >= 3.78 |

- **(a):** r_F grows 1.56 -> 2.10 -> 3.42 (x1e-3) while r_H grows only 19 %. **The earlier (a) pass holds only for channel cutoff <= 14.**
- **(b):** far from the 0.3 refutation level.
- **(c):** cutoff-stable; there are no trapping roots at 11.0.
- **(d):** stable to 1 % across genuine histories. Within k20, the sea-dust 10 -> 20 growth is again Sigma dressing.

## (3) Dust-only doubles beside r_H

| slice | dust-dust r_D | r_H | ratio |
|---|---|---|---|
| 11.0 (k10 / k14 / k20) | 0.0215 / 0.0215 / 0.0217 | 0.220 / 0.234 / 0.262 | 0.098 / 0.092 / 0.083 |
| 15.0 | 0.049 | 1.170 | 0.042 |

## Status of (d)
As ruled, (d) on the 10-channel records is UNTESTED. On the genuine 14/20-channel histories at 11.0, the dust-dust r_D is cutoff-stable.

## Worker slip
A pgrep listing printed route-1 process command lines. No route-1 file was opened.

---
# v1.2 addendum (#564; four-block split, read-only; notes s9, md5 0211287d8b7e5ba554e226bfaa8b8e1f)

The four blocks plus interference reproduce r_F exactly. Values are weighted sums of squares, sum_k 16k ||.||^2.

**11.0 histories:**

| history, channels | U F[Q-P] Q | U F[D] D | U F[D] Q | r_F^2 |
|---|---|---|---|---|
| k10, old | 1.048e-6 | 1.221e-6 | 1.71e-7 | 2.435e-6 |
| k14, old | 1.366e-6 | 1.272e-6 | 1.72e-7 | 2.805e-6 |
| k14, added | 1.610e-6 | 0 | 0 | 1.610e-6 |
| k20, old | 1.933e-6 | 1.359e-6 | 1.75e-7 | 3.461e-6 |
| k20, added | 8.203e-6 | 0 | 0 | 8.203e-6 |

- The U F[Q-P] D block is ~3e-13.
- Interference is -6e-9 (sea columns) and +4e-10 (dust columns).

**15.0, k10:** U F[D] D = 2.52e-6, which is 84 % of r_F^2 = 2.997e-6. The sea block is 4.2e-7.

**Dust-hole residual** r_Fd = sqrt(sum 16k ||U F[Dbar] Pi_D||_F^2):
- values: 1.105e-3 / 1.128e-3 / 1.166e-3 at 11.0 (k10/14/20), and 1.588e-3 at 15.0;
- per sqrt(640): 4.4e-5 to 6.3e-5;
- r_Fd/r_H: 4.4e-3 to 5.0e-3 at 11.0, and 1.4e-3 at 15.0;
- cutoff-stable (+5.5 %).

**reviewer A's predictions:**
1. "Channels above 10 carry only U F[Q-P] Q": **HOLDS.** The other blocks are exactly zero there, structurally, because there is no dust above k = 5.
2. "r_F^2 tracks sum 16k": **DOES NOT HOLD.**
   - Per unit weight, r_F^2 is 2.77e-9, 2.63e-9 and 3.47e-9. The sea block alone is 1.19e-9, 1.77e-9 and 3.02e-9, which is superlinear.
   - The added channels' per-copy sea block shows no trend in k.
   - The old channels' own sea block rises with the history's cutoff (+84 % from 10 to 20), while their dust blocks rise only ~11 %.
   - **So the (a) failure at k = 20 comes from the sea-reference mismatch block U F[Q-P] Q: 1.01e-5 of r_F^2 = 1.17e-5.**
