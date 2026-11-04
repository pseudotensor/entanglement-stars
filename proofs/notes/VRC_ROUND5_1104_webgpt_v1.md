# VRC route B -- ROUND 5 at 1104 of C_M1nl23 (route 2, route2), v1, with the #480 closing check and the amend7 massless cubic K

Manifests:
- VRC_MANIFEST_webgpt_v1_amend6.md (cd5c4833) and amend7 (961b41ef), each frozen before any of its numbers.
- reviewer B vrccmp4 was not opened.
- Route A's files were used as arrays only.

## 1. Arrays and cross-evaluation (round5_1104_v1.json 54c70295, vrcb_round5_v1.py 6b271c3a)
Arrays:
- Route A's exported j_L, j_H, j_H_s_only, V, u_b and s_b equal route B's at roundoff (relative 2.4e-19 to 1.1e-16).
- s_only vs route B's s_C,c: 1.7e-18 absolute.

Cross-residuals of route B's operators on route A's exported tangents, relative:
- ||B j_L - bare_A||: 1.2e-7 (JVP step 5e-6) and 6e-7 (2.5e-6).
- D_0 on zero_A: 1.3e-7 / 6e-7.
- D_full on full_A: 1.9e-7 / 5e-7.
- Against A's own bare: 1.5e-8 to 1.4e-7.
These sit at the FD floor of route B's JVPs, so the tangent solutions are mutually consistent at about 1e-7.

## 2. Direct correction D_full delta = -B T_L du_0 (steps 1e-5, 5e-6, 2.5e-6)
Final step 2.5e-6, unrounded. Errors are the step change plus the conditioning-propagated solve error.

| covector | dV | dV/V0 | unit-weight signed rate |
|---|---|---|---|
| s_C - t | 1.5345028481517413e-05 +- 3.4e-11 | -1.2016021672679294e-4 +- 2.8e-10 | -2.346787114171107e-4 +- 6.1e-10 |
| s_C only | 1.5686334873191136e-05 +- 3.4e-11 | -1.2291770833404184e-4 +- 2.8e-10 | -2.4042650394732708e-4 +- 6.1e-10 |

The earlier inter-route 2.2% (vertex) and 2.4% (rate) gaps are the covector choice. Operators and solves agree at about 1e-7.

## 3. #480 closing check: physical heavy leg = s_C only, route A's exported key (close_1104_v1.json 3a359709, vrcb_close_v1.py a228839a)

| method | V_0 | V_full | dV | rate |
|---|---|---|---|---|
| tangent solves (amend5) | -0.1276165696481761 | -0.1276008831359446 | 1.5686512231525424e-05 | -2.4042928421562322e-4 |
| **direct correction (quoted)** | **-0.12761655814930967** | **-0.12760087181443647** | **1.5686334873187338e-05** | **-2.4042650394716746e-4** |
| perturbed fixed points | -0.12761656280577027 | -0.12760087441667853 | 1.568838909173434e-05 | -2.404589687633824e-4 |

- dV error 2.5e-9: the method spread 2.05e-9 plus the fixed-point Richardson change 4.0e-10.
- Rate error 3.2e-8 and V_0 error 1.1e-8, both method spread.
- dV/V0 = -1.2291770833401208e-4.

## 4. Width-2/4/8 density-cutoff collar (debit units; hd 0.125 values; eps includes the 0.25 -> 0.125 change and the polarization Richardson change)

| sigma_w | W | Q (density collar) | remainder W - Q | outside majorant | |W-Q| + eps_rem vs 1e-3 (|W| - eps_W) |
|---|---|---|---|---|---|
| 2 | -2.9524054653e-3 | -2.5594272296e-3 | -3.92978e-4 | 4.0188e-4 | 3.9301e-4 vs 2.9522e-6: FAIL |
| 4 | -2.6942136210e-3 | -2.7187222799e-3 | +2.45087e-5 | 2.8337e-5 | 2.4518e-5 vs 2.6941e-6: FAIL |
| 8 | -1.9260027569e-3 | -1.9278597815e-3 | +1.85702e-6 | 1.8571e-6 | 1.8586e-6 vs 1.9259e-6: PASS |

Only width 8 passes the one-per-mille omission test.

## 5. Massless cubic K, m_L a = 0 (amend7; cubicK_m0_v1.json f34ce082, vrcb_cubicK0_v1.py 811ff548; run on pod2)
- Grid: shifted midpoint (off the doublers), N = 128/256/512/1024.
- Fitted exponent p = 2.000 for every entry (between triples 2.0001 vs 2.0006), so the convergence is algebraic O(N^-2).
- Enclosure = K_inf +- (|K_inf - K_1024| + |previous-triple extrapolation - K_inf| + class spread).
- The literal-vs-reduced check at N = 128 is in the JSON.

| entry | N = 1024 | K_inf | enclosure |
|---|---|---|---|
| c11 | 0.010517417597355033 | 0.010517254745127151 | [0.010517091617551926, 0.010517417872702377] |
| c12 | -0.0039075003016594235 | -0.003907418875517711 | [-0.003907500437572025, -0.003907337313463396] |
| kappa_12,12 | 0.01630613297802983 | 0.016306295565907653 | [0.016306132866483292, 0.016306458265332013] |
| kappa_12,21 | = c12 | | |
| c44_sym | 0.006199316338185204 | 0.006199438345193926 | [0.00619931632593629, 0.0061995603644515625] |

ma -> 0 cross-check (N = 512, m = 0.2/0.1/0.05/0.025), extrapolating in m with the fitted p_m:
- c11 -> 0.0105177 (p_m 1.95);
- c12 -> -0.0039131 (p_m 1.52);
- c44_sym -> 0.0062015 (p_m 1.63).
These are consistent with the m = 0 limit at the 1e-3 level. The m-sequence is not cleanly algebraic, so the m = 0 grid limit above is the deciding number.
