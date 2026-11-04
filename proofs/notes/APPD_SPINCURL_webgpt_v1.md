# APPENDIX D SPIN-CURL CORRECTION, route B (route2), v1

Lead #392. Manifest: APPD_SPINCURL_MANIFEST_webgpt_v1.md (md5 77df28286e148e4d12c93ad7d2799f88), frozen with pass rules before any number. Blind to route A.

## Verdict

- **The correction is real for an eq. 5-only reference.** Eq. 5 on a rigid turn gives exactly -S|omega|^2/4.
- **This route's App. D certification (v1.1) already contains the correction.** Its 12 x 12 tensor M carries reviewer A's rotation block exactly:
  - M_bb = S to 7e-16;
  - the v-b cross block is S/4;
  - the rotation sector cancels to 1e-16 on b = curl v/4.
- **The recomputed strict-mixedness result is unchanged.** At ||c|| = 1.1e-7 the sup is 6.945e-5 < 1e-4 (margin 3.06e-5), and r* = 2.38e-7, for both the FULL and the EQ5-ONLY variants.
- **The delta r_v shift of the sup is <= 5.6e-9**, against reviewer A's estimate of <= 5e-6.

| Item | Pre-registered rule | Measured | Result |
|---|---|---|---|
| (0) Rigid turn, eq. 5 alone, against -S|omega|^2/4 | ratio within 1e-6 | W_2^eq5 = -0.0344089974131, -S/4 = -0.0344089974131, ratio 1.0000000000 | **PASS-0** (the correction is needed for eq. 5-only) |
| (1) S, and S against M_bb | rel. enclosure < 1e-6; M_bb = S | S = 0.137635989652265 (trapezoid N = 96 to 256, converged to 3e-16, float); M_bb = S to 7e-16; 4|M_vb| = S | **PASS-1** |
| (2) Route B exact energies, transverse modes, at 1e-5 | accepted fits (residual < 1e-9), enclosure < 1e-5 | No fit met the 1e-9 residual: the transfers reach k = 0.39 (L = 256) and 0.52 (L = 192), and the k^6+ terms there exceed 1e-9 | **FAIL by the pre-registered rule** (diagnostic below: agreement at <= 1e-8) |
| (3) Curl-carrying compact bump at the ball size | diagnostic, no rule | exact/full-M = 0.856; exact/eq5-only = 0.472 | full M right; eq. 5-only off by 2.1x |
| (4) Strict mixedness, FULL margin > 0 at 1.1e-7 | margin > 0 | margin 3.06e-5 (FULL = s88); EQ5-ONLY the same to 4e-12 | **PASS-4** |

## Details

### (0) and (1): the rotation sector

On v = omega x x (N = 1, omega = e_3), W_2 densities:

| Tensor used | Density |
|---|---|
| eq. 5 flow block alone | -0.0344089974131 = -S/4 |
| full M with b = omega/2 | 1.2e-16 (cancels) |
| spin block alone, b = omega/2 | -0.0344089974131 = -S|b|^2 |

So eq. 5 alone carries the orbital -(S/16)|curl v|^2, which the spin block cancels. **This route's M was the full W_2 throughout** (s72, s75, s88). The tensor had been validated against exact responses that included the spin term (s72, curl-carrying modes).

### (2) Route B exact second-order energies (s90; json md5 51f012e44030815f63cfac7ab70e23a6)

**Setup.** Transverse flow mode dv = e_j cos(k x_i), (i, j) = (1, 2) and (1, 3), ma = 0.2. Closed-form projector-trace sums on periodic parents L = 256 and 192, at n = 1, 2, 4, 8, 16. Three runs:
- (a) orbital only, against eq. 5;
- (b) spin only, against the b-b block (S|b|^2, b = curl/4);
- (c) both, against the full M.

**Raw data.** A/A_2 - 1 is smooth and goes as k^2. At the lowest transfer (L = 256, k = 0.0245):

| Run | A/A_2 - 1 |
|---|---|
| (a) orbital | -3.163e-4 |
| (b) spin | -4.796e-4 |
| (c) both | -3.863e-4 |

**The pre-registered acceptance (fit residual < 1e-9) admitted no fit, so PASS-2 is FAILED by the rule.** This is a fault of my rule design (the k range was too wide for a 1e-9 residual), not of the data.

**Diagnostic** (s93, not the pre-registered rule; json edeab5ea21405a24a5c023f4572f74a1). The interpolating Richardson intercepts converge monotonically:

| Run | L = 256: 3 / 4 / 5 points | L = 192: 3 / 4 / 5 points |
|---|---|---|
| orbital | -1.66e-8 / -1.63e-9 / -5.5e-10 | -9.1e-8 / -1.4e-8 / -6.7e-9 |
| spin | -1.71e-8 / -1.58e-9 / -5.2e-10 | -9.3e-8 / -1.4e-8 / -6.4e-9 |
| both | -1.90e-8 / -1.88e-9 / -6.4e-10 | -1.04e-7 / -1.7e-8 / -7.8e-9 |

(Identical for (i, j) = (1, 3).)

So the exact energies agree with eq. 5, S|b|^2 and the full M at the 1e-9 to 1e-8 level, against the 1e-5 target. This is printed as a diagnostic, not as a pass.

### (3) Ball-size compact bump (s92; json md5 254a55114794e03d86d6b374dee6c2d2)

Open ball R_b = 8, flat h_0, ma = 0.2. Bump dv = psi e_2 (w = 4), back-averaged; curl term w = grad Phi x e_2.

| Exact E2 | W_2 prediction | ratio |
|---|---|---|
| orbital 0.0029698 | eq. 5: 0.0034791 | 0.854 |
| spin 0.0013055 | S|b|^2: 0.0017242 | 0.757 |
| both 0.0016412 | full M: 0.0019174 | **0.856** |
| both 0.0016412 | eq. 5 only: 0.0034791 | **0.472** |

**Reading.**
- The full-M prediction carries the same 15 percent gradient-expansion and boundary error as the orbital comparison: the bump is only 2a wider than the kernel.
- The eq. 5-only prediction is wrong by 2.1x for a curl-carrying bump.
- This is the delta r_v effect, seen directly.

### (4) Strict mixedness (s91, EQ5-ONLY; json md5 d69158556688e076a630dc07b9d4f75f; FULL = s88, md5 592ec0a67223eb7d3e4af3c945cfd59c)

| | FULL (12 x 12 M; s88) | EQ5-ONLY (9 x 9; s91) |
|---|---|---|
| H_W Frobenius | 0.012545 | 0.022415 |
| Sensitivity sqrt(||S||) (Lanczos + Cholesky) | 232.96904 | 232.96901 |
| ||X_0||_op (certified) | 4.36152e-5 | 4.36152e-5 |
| Corrected X_0 bound (v1.1 method: + 2 eps sqrt(dim) ||P_hat - P_star|| + s_g rounding) | 4.3623e-5 | 4.3623e-5 |
| Bound at 1.1e-7 (X_0 + lin + rem) | 6.945e-5 | 6.945e-5 |
| Margin at 1.1e-7 | 3.06e-5 | 3.06e-5 |
| Certified radius | 2.38e-7 | 2.38e-7 |

**Measured delta r_v.**
- H_W(full) - H_W(eq5) has Frobenius norm 0.01210. It sits entirely in the flow-bump columns (0.01210); the clock-bump columns carry 2.2e-7.
- That is 0.37 percent of ||A_2||_F.
- Its own Hilbert-Schmidt-mapped sup is <= 0.0511 per unit ||c||, so <= 5.6e-9 at 1.1e-7.
- The sensitivity changes by 3e-5 out of 233.
- The sup is dominated by the laws' A_2. The reference response on these tests is sub-percent.

**Background discrete-curl residual.** max |curl_D v_0| = 6.45e-4 on the sites, against max |grad v_0| = 0.031. Its rotation-sector W_2 at u_0 is -1.2e-6. It enters H_W only through the g_0 terms, already included in both variants.

The anchor, Gram and path gap are unchanged (curl-free), as stated.

## Self red-team

| Risk | Check |
|---|---|
| M's spin block mis-normalized | M_bb = S to 7e-16 by two independent formulas. The rotation sector cancels to 1e-16. The ball bump shows full-M/exact the same as orbital/exact (0.856 against 0.854) |
| PASS-2 failure hides a real disagreement | The interpolating intercepts converge to <= 6e-10 (L = 256) and 7e-9 (L = 192). The rule failed on its residual threshold, not on the intercept |
| delta r_v under-weighted by near-null Gram directions | The full operator-norm sensitivity is recomputed with the change (232.96904 against 232.96901), not estimated |

## Record

| File (box <internal>/appd) | md5 |
|---|---|
| spincurl_s90_v1.json | 51f012e44030815f63cfac7ab70e23a6 |
| spincurl_post_s93_v1.json | edeab5ea21405a24a5c023f4572f74a1 |
| cert_s91_eq5only_v1.json | d69158556688e076a630dc07b9d4f75f |
| ballbump_s92_v1.json | 254a55114794e03d86d6b374dee6c2d2 |
| scripts s90, s91, s92, s93 | ce7715b2, 0c0e239e, a60b2240, 4b9420bd |
