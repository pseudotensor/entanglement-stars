# VRC route A (closed form), solve, v1.9.1: vertex reference contraction on mark 3 (#620, #621, #662, #665, #669, #671, A7, #673, #677, #678)

**v1.9.1 = v1.9 (bc6717d8) plus "CORRECTION 31 Oct 2026: the collar verdict under lead #678 (labels and record)". This is the v1.8.1 correction the lead requested; v1.9 already carried the literal verdict, so the correction is issued on top of it. v1-v1.9 stay exactly as issued.**

**v1.9 = v1.8 (12421fb3) plus the final section "COLLAR TEST, LITERAL FORM (#678, A9c)": the same stored collar numbers, re-tested under the lead's literal definition. Where they differ, it supersedes v1.8's collar verdict. v1-v1.8 stay exactly as issued.**

**v1.8 = v1.7 (0036d977) plus the final section "ROUND 5 (#677, A9)": the direct correction with a JVP refinement sequence for both covectors, and the width-8 density-cutoff collar test. Cross-evaluated residuals are PENDING because route B's solution has not been exported. v1-v1.7 stay exactly as issued.**

**v1.7 = v1.6 (94799267) plus the final section "COMMON-ARRAY TANGENT AUDIT AND COLLAR CUTOFF (#673, A8)".** On the common arrays, route A's heavy covector is s_C,c - t_c(u_b); v1-v1.6 used s_C,c only. The common-definition values of V_HL0, dV and the rate in that section SUPERSEDE the earlier ones as common-definition values. v1-v1.6 stay exactly as issued.

**v1.6 = v1.5 (aa43c002) plus the final section "CHI_L WEIGHTING CHECK (A7)".** The factor-2 flag is CLOSED: the weighting is confirmed, so no light number changes. The section also gives the exact averaging-kernel formula and the clock origin from code. v1-v1.5 stay exactly as issued.

**v1.5 = v1.4 (ba6fa2d4) plus the final section "METADATA AND CUBIC K RESTATED (#671; reviewer B vrcfa)".** It adds the exact C_M1nl23 metadata, with file and md5 for each value, the answer on the paper's r_s/a = 64 statement, and c11/c12/c44 restated on the explicit grid. Where they differ, it supersedes the #669 LABEL. v1-v1.4 stay exactly as issued.

**v1.4 = v1.3 (bf2469b1) plus the final section "DECIDING CALCS (#669)", which includes the LABEL for history C_M1nl23.** It is frozen under ADDENDUM_A6 (e9044543) and was computed BLIND to route B. v1-v1.3 stay exactly as issued.

**v1.3 = v1.2 (e86f7cf3) plus the final section "ROUND 3 (lead #665): clock 1104 on the common base".** That section is frozen under ADDENDUM_A5 (670b540f) and was computed BLIND to route B. v1-v1.2 stay exactly as issued.

**v1.2 = v1.1 (8534b429) plus the final section "CORRECTIONS 31 Oct 2026 (reviewer B vrccmp, lead #664)".** Where they conflict, that section supersedes the text above it.

**v1.1 = v1 (7cd07541) plus section 2.4:** the c19 LIGHT row on CL_snapshots_v3, which replaces the withdrawn stage-2 (ii) light block (ADDENDUM_A4, md5 2979d860). Sections 1-5 are v1's text; section 6 is updated.

**Governing:** AstraAPI_vrc1a_v1 sections 1-2; FableAPI_vrc1f_v1 (3)(a)-(d).

**Manifest:** vrc620/MANIFEST_vrc620_v1.md (86adb5f7, with A1 and A2), plus ADDENDUM_A3_vrc620_rooted_v1.md (d65a4536, frozen before any rooted number).

BLIND to route B: no route2 VRC file was opened.

**Scripts** (vrc620/; box <internal>/vrc620):
- vrc_chi_v2 (59a5e099)
- vrc_cubic_v1 (3d7628d5)
- vrc_vertex_v3 (2780c4ff)
- vrc_w2direct_v2
- vrc_lightir_v2 (2ea07bd6)
- vrc_vertex_v4 (89027978)
- vrc_kspec_v1 (2a3f98b7)

**Outputs:** vrc620/a3/ (JSON md5s at the end).

## 1. What was already done (earlier sections and STATUS; unchanged)

- **Stage 1, PASS.** The chain closed form chi_ref = -2 sum (1/N) xi_H^T [[A, -B], [-B, C]] xi_L:
  - vs FD of dW2: 2.8-4.8e-11;
  - quadratic truncation: ratios 4.00-4.04;
  - symmetry <= 8e-16; Euler <= 1.5e-16;
  - assembled heavy 4e-11.
- **Pre-formation vertex** (records 688/696/704). dV_HL/V_HL = 2.1-2.5e-5 and eta 4.3-5.5e-5. The 1% omission test PASSES.
- **A2 (a), direct W_2 on the chain** (record 704, k 1-5). A_2/A_exact - 1 = -8.4e-4 / -2.1e-4 at L 40/80, quadratic in L. The Richardson extrapolation is +8e-7 +- 5e-8: target 1e-5 MET.
- **A2 (b), light m = 0 IR prescription** (finite open 8192-site parent, k_l/r the only regulator), on the engine-native light. Dependence on m_L a = 1e-4: <= 2.5e-6; on m_L a = 1e-3: <= 2.5e-4; on parent 4096 vs 8192: <= 3e-8. PASS.
- **c19 LIGHT block:** withdrawn with the old C_L. The optional redo on CL_snapshots_v3 is NOT done in v1.

## 2. Rooted mark 3 records (A3)

**Runs:** PRIMARY C_M1nl23 (light on) and TWIN C_M1nl25off (light does not source the rows).

**Clocks:** 1104, 1152, 1200, 1408, 1984, each with its +-8a and +-16a records (box pl2 mirrors).

**1408 has NO OUTER ROOT in either run.** It is reported as such and not substituted (the frozen rule). The primary vertex there is still computed.

**Map check at every clock:** closed-form adjoint vs FD of dW2 = 7e-12 to 1.5e-11 (threshold 1e-5).

### 2.1 Stage 2 (i) and A2 (d): the full V_HL insertion (PRIMARY)

| t | V_HL0 (chi_L off) | dV_HL = V_full - V_0 | dV/V_0 | raw vs linear-form (rel) | first order rel | max GMRES res | eta uniform | max eta per channel |
|---|---|---|---|---|---|---|---|---|
| 1104 | -0.127617 | +1.569e-5 | -1.23e-4 | 5.1e-5 | 3.1e-4 | 1.6e-12 | 1.23e-4 | 1.51e-4 |
| 1152 | -0.136679 | +1.166e-5 | -8.5e-5 | 3.1e-5 | 2.1e-4 | 9.4e-13 | 8.5e-5 | 9.5e-5 |
| 1200 | -0.141668 | +6.88e-6 | -4.9e-5 | 8.3e-5 | 1.9e-4 | 1.8e-12 | 4.9e-5 | 5.4e-5 |
| 1408 | -0.125645 | +1.82e-6 | -1.4e-5 | 6.8e-4 | 9.9e-4 | 7.0e-13 | 2.6e-5 | 4.1e-5 |
| 1984 | -0.102900 | +1.015e-5 | -9.9e-5 | 7.8e-5 | 1.3e-3 | 2.3e-13 | 1.68e-4 | 2.59e-4 |

- **Discrepancy** (raw difference vs the difference-free linear form, plus the JVP step uncertainty): <= 6.8e-4 of |dV|, within the 1e-3 threshold. PASS at every clock.
- **Solves:** GMRES <= 2e-12.
- **One-percent omission, vertex:** PASS at every clock (|dV| + eps <= 1.3e-4 |V_0|).
- **A2 (d) rate:** eta is measured against the re-solved baseline (J0 = chi_H in the new-law rows), with the chi_L-off control at the same covariance. eta_uniform is 2.6e-5 to 1.7e-4, so |dGamma|/Gamma0 <= 2 eta + eta^2 <= 3.4e-4. The uniform weights are labelled; per-channel values are in the JSONs. One-percent omission PASS.
- **Twin:** NOT APPLICABLE. Its light does not source the rows, so no heavy-light row vertex exists there.

### 2.2 Root debit (both runs)

(d_t M)_ref,R = -chi_ref(u_bar_R, u_dot)/N_R, with u_bar_R = W^dag(1_{y<R} u) (production sigma = 4a). u_dot and d_t M|_R are Richardson rates from the +-8a and +-16a records.

| run t | R | d_t M|_R (fd err) | ref total | ref heavy | ref light | eps_ref | share | 1% test | D_R enclosed | dt/du enclosed |
|---|---|---|---|---|---|---|---|---|---|---|
| on 1104 | 69.47 | -0.03555 (0.0025) | -3.060e-3 | -2.813e-3 | -2.47e-4 | 4.9e-5 | +8.6% | FAIL | [0.554, 0.930] | [1, 1.0022] |
| off 1104 | 69.47 | -0.03654 (0.0025) | -3.074e-3 | -2.825e-3 | (-2.48e-4)* | 4.9e-5 | +8.4% | FAIL | [0.548, 0.930] | [1, 1.0022] |
| on 1152 | 57.25 | -0.18669 (0.0025) | +1.076e-3 | +1.002e-3 | +7.4e-5 | 2.0e-5 | -0.58% | pass | [0.392, 0.558] | [1, 1.0017] |
| off 1152 | 57.18 | -0.18833 (0.0023) | +1.015e-3 | +0.946e-3 | (+6.9e-5)* | 2.3e-5 | -0.54% | pass | [0.400, 0.560] | [1, 1.0017] |
| on 1200 | 46.67 | -0.15066 (0.0008) | +6.38e-4 | +5.98e-4 | +3.9e-5 | 1.7e-6 | -0.42% | pass | [0.247, 0.432] | [1, 1.0013] |
| off 1200 | 46.45 | -0.14825 (0.0011) | +6.13e-4 | +5.76e-4 | (+3.7e-5)* | 1.4e-6 | -0.41% | pass | [0.209, 0.422] | [1, 1.0013] |
| on 1984 | 34.83 | +0.04844 (0.0003) | +1.301e-3 | +1.186e-3 | +1.15e-4 | 2.9e-6 | +2.7% | FAIL | [0.716, 0.822] | [1, 1.1259] |
| off 1984 | 34.52 | +0.05148 (0.0002) | +1.381e-3 | +1.259e-3 | (+1.22e-4)* | 2.1e-6 | +2.7% | FAIL | [0.684, 0.795] | [1, 1.1260] |

\* The twin's light part is printed only: it does not enter that run's balance (only chi_H does). The twin "share" uses the total; with chi_H alone the twin shares are +7.7%, -0.50%, -0.39% and +2.4%.

**Reading:**
- The reference term in the root debit is NOT a one-percent correction. It is 8.4-8.6% at the onset of retreat (1104) and 2.7% at 1984, and 0.4-0.6% at 1152/1200.
- About 92% of it is the HEAVY reference (chi_H). The LIGHT reference part alone, -0.7% / +0.04% / +0.03% / +0.24% of d_t M|_R in the primary, passes the 1% test at every clock.
- Per the rule, failure means the correction is RETAINED in the root balance. Nothing is rejected.
- The margin D_R is enclosed positive at every rooted clock, so the signs are claimable. dE_R/du enclosures are in the JSONs (dEdu_*_enclosure).

### 2.3 A2 (c): windowed contraction vs the collar forms (both runs)

- **C1, the chain collar:** the exact local limit of the same chain Hessian, -(2 v_R/N_R^2) sum_ch w (A xi_dot1 - B xi_dot2)|_R.
- **C2, the cubic collar:** -(8 pi R^2 v_R/N_R^2) <K_rr,kl> E_kl(u_dot), with K orientation-averaged.

| run t | windowed sigma/a = 2 / 4 / 8 | C1 | dist to C1 (2/4/8) | sign = C1 | monotone | C2 heavy / light (per unit mult.) |
|---|---|---|---|---|---|---|
| on 1104 | -3.359e-3 / -3.060e-3 / -1.789e-3 | -3.255e-3 | 1.0e-4 / 2.0e-4 / 1.5e-3 | yes | yes | -0.338 / -0.359 |
| off 1104 | -3.377e-3 / -3.074e-3 / -1.795e-3 | -3.274e-3 | 1.0e-4 / 2.0e-4 / 1.5e-3 | yes | yes | -0.340 / -0.361 |
| on 1152 | 7.89e-4 / 1.076e-3 / 1.230e-3 | 7.00e-4 | 9e-5 / 3.8e-4 / 5.3e-4 | yes | yes | +0.045 / +0.046 |
| off 1152 | 6.90e-4 / 1.015e-3 / 1.216e-3 | 6.42e-4 | 5e-5 / 3.7e-4 / 5.7e-4 | yes | yes | +0.041 / +0.041 |
| on 1200 | 6.43e-4 / 6.38e-4 / 7.01e-4 | 8.12e-4 | 1.7e-4 / 1.8e-4 / 1.1e-4 | yes | NO | +0.035 / +0.036 |
| off 1200 | 6.23e-4 / 6.13e-4 / 6.79e-4 | 7.96e-4 | 1.7e-4 / 1.8e-4 / 1.2e-4 | yes | NO | +0.034 / +0.035 |
| on 1984 | 1.479e-3 / 1.301e-3 / 3.81e-4 | 1.297e-3 | 1.8e-4 / 5e-6 / 9.2e-4 | yes | NO | +0.036 / +0.039 |
| off 1984 | 1.538e-3 / 1.381e-3 / 4.84e-4 | 1.363e-3 | 1.8e-4 / 1.8e-5 / 8.8e-4 | yes | NO | +0.037 / +0.040 |

- **SIGN: PASS** at every rooted clock, run and sigma. The windowed contraction, C1 and C2 (heavy and light) all agree in sign. The sign follows the strain rate at the root: negative at 1104, positive at 1152-1984.
- **Convergence (frozen criterion): FAIL at 1200 and 1984** (4 of 8 clock-runs), PASS at 1104 and 1152.
- The collar form reproduces the windowed contraction to 3-21% at sigma = 2a, and to 0.4-58% at the production sigma = 4a. It is a sign and order-of-magnitude form, not a quantitative replacement on these records.
- C2's magnitude is per unit species multiplicity. The chain-to-cubic multiplicity is not established, so C2 counts for sign only, as frozen.
- Recomputing at production sigma reproduces the production value exactly (difference 0).

### 2.4 c19 LIGHT row on CL_snapshots_v3: REPLACEMENT of the withdrawn stage-2 (ii) light block (v1.1)

**Frozen:** ADDENDUM_A4_vrc620_c19light_v1.md (2979d860).

**Script:** vrc_c19_v4.py (4b75a7f4) = vrc_c19_v2's block() text, light only.
- Inputs: the published v3 k = 1 meta files only, each md5-checked against MD5SUMS (88380b2e). The withdrawn directory was not opened.
- Object: per copy (weight 1), k = 1, m = 0, 8192 sites, UNCONTROLLED (E29), under the A2 (b) finite-parent IR prescription.

| T/r_s | u_dot | chi_L(u_dot, u_dot) per copy | FD rel (h/8) | FD halving ratios | symmetry | Euler |
|---|---|---|---|---|---|---|
| 20 | D1 backward 18.5 -> 20 (replacement value) | -7.376e-8 | 4.7e-9 | 4.00 / 4.02 / 4.06 | 4.8e-15 | 0 |
| 20 | D2 centred 18.5 / 20 / 24 | -2.959e-8 | 2.9e-9 | 4.00 / 4.02 / 4.06 | 8.2e-16 | 0 |
| 40 | D1 backward 30 -> 40 (replacement value) | -1.065e-9 | 8.7e-10 | 4.00 / 3.98 / 4.50 | 6.0e-16 | 0 |
| 40 | D2 centred 30 / 40 / 56 | -2.355e-10 | 9.4e-10 | 4.00 / 3.98 / 4.50 | 2.5e-15 | 0 |

**Thresholds** (FD <= 1e-5 with quadratic convergence; symmetry and Euler <= 1e-10): PASS at T20 and T40, for both directions.

**DIRECTION SENSITIVITY: |D1 - D2|/|D1| = 0.60 (T20) and 0.78 (T40).** The published snapshot spacing (1.5-16 r_s) does not resolve the field rate. So the contraction VALUE is uncertain at order one, even though the closed form itself is verified to 1e-8. It is quoted as -7.4e-8 (T20) and -1.1e-9 (T40) per copy, with that spread. Only the sign (negative) and the order of magnitude are robust.

**Script note.** v3 (61a1e5b5; output c19light_v3.json, labelled SUPERSEDED, never quoted) took the meta field T to be in r_s units, but it is already in a. That made u_dot 64x too small. v4 fixes this and asserts meta T = 64 x label.

The withdrawn v2 row used the same "(T - T') x 64" construction on the old meta. If the old meta also stored T in a, its value carried the same factor. It is withdrawn either way and was not reopened.

## 3. Cubic K: independent spectral-sum extraction (A3 item 4)

**Method:** the exact second-order lower-band energy of h0 + (1/2){v, q} for plane-wave flows v = e cos(k x) on the naive cubic Dirac parent. No projector derivative is taken.
- Richardson in k^2 (h = 0.08 to 0.01).
- 9 waves: longitudinal and transverse along [100], [110] and [111].
- Compared wave by wave, and by a least-squares fit of the cubic components, with vaclaw kappa_cubic on the same p grid.

| m | npts | max wave rel diff | fit rel diff (K1, K2a, K2b, K3) | c11 | c12 | c44 |
|---|---|---|---|---|---|---|
| 0.2 | 64 | 4.0e-10 | <= 3.0e-10 | 0.0098341 | -0.0030481 | 0.0055542 |
| 0.2 | 128 | 1.5e-12 | <= 3.3e-12 | 0.0098342 | -0.0030481 | 0.0055542 |
| 0.05 | 64 | 6.0e-6 | <= 7.4e-6 | 0.0104391 | -0.0037882 | 0.0061311 |
| 0.05 | 128 | 1.2e-6 | <= 1.2e-6 | 0.0104681 | -0.0037973 | 0.0061300 |

**PASS at the 1e-5 target** on all three invariants, at both masses and both grids. K2a = K2b, as the cubic symmetry requires.

The light (m = 0) value is not taken on this grid, because eps vanishes at the doubler nodes; m 0.05 is a labelled proxy.

**For the paper's bracket "16 pi G K_L"** at G/a^2 = 1/4 (m 0.05 proxy):
- uniaxial K(e1 e1, e1 e1) = c11 = 0.01047, giving 16 pi G K = 0.132;
- pure shear K(S, S)/|S|^2 = 2 c44 = 0.01226, giving 0.154.

At m 0.2: 0.124 and 0.140. This sits at the lower end of reviewer A's a-priori 0.15-0.25.

## 4. Thresholds summary (manifest)

| test | result |
|---|---|
| closed form vs FD <= 1e-5 | PASS (<= 1.5e-11 on rooted records) |
| symmetry / Euler <= 1e-10 | PASS (stage 1) |
| vertex discrepancy <= 1e-3, solves | PASS (<= 6.8e-4; GMRES <= 2e-12) |
| 1% omission, vertex | PASS (all clocks) |
| 1% omission, rate (eta) | PASS (eta <= 1.7e-4) |
| 1% omission, root debit (total reference) | FAIL at 1104 (8.4-8.6%) and 1984 (2.7%); pass at 1152 and 1200. Correction RETAINED |
| 1% omission, root debit (light reference alone, primary) | PASS (<= 0.7%) |
| A2 (c) sign | PASS |
| A2 (c) monotone convergence to the collar | FAIL at 1200 and 1984; pass at 1104 and 1152 |
| cubic spectral sum 1e-5 | PASS |
| root margin enclosed positive | yes at every rooted clock |

## 5. reviewer A's mark 3 sentence, filled where measured

"... with ratio 16 pi G K_L = 0.13-0.15 (m a = 0.05 proxy of the massless chain; 0.12-0.14 at m a = 0.2) on uniaxial-to-shear strains. On the re-solved slices it changes the rate by eta <= 1.7e-4 (|dGamma|/Gamma <= 3.4e-4) and the root debit by +8.6%, -0.6%, -0.4% and +2.7% of d_t M|_R at t = 1104, 1152, 1200 and 1984 (light part alone <= 0.7%), under the stated infrared prescription of the massless chain."

The error bars are in sections 2-3.

## 6. Not done

- The optional c19 light row is now DONE (section 2.4, v1.1).
- An m = 0 cubic K on a shifted grid.
- A C_L-based j_L insertion on c19, which needs a combined heavy + light rows model that does not exist.
- A finer-T u_dot for c19: published T only, per instruction.

## 7. Output md5s (vrc620/a3/)

- kspec_v1.json: 6448097a
- v4_on_t1104 / 1152 / 1200 / 1408 / 1984: d7912206 / 20525bd3 / 4c06e00b / a818bd05 / 734ef63f
- v4_off_t1104 / 1152 / 1200 / 1408 / 1984: 010f3d3b / 419809d1 / 5c1bc026 / 4c285c47 / a154942d
- c19light_v4.json: cb8389d3 (v1.1)
- c19light_v3.json: 5c90be31 (superseded, not quoted)

## CORRECTIONS 31 Oct 2026 (reviewer B vrccmp, lead #664)

Source: AstraAPI_vrccmp_v1, which compared v1 only. v1 (7cd07541) and v1.1 (8534b429) stay exactly as issued; this section supersedes the affected statements in both. All values below are recomputed from vrc620/a3/v4_{on,off}_t<clock>.json (ref / d_t M|_R).

**(1) The twin's PHYSICAL root-reference shares are heavy-only.** The twin's light does not source its rows, so only chi_H enters that run's balance. The "share" column of the twin rows in section 2.2 includes a light response that is NOT in the twin's balance.

| clock | twin heavy-only share, this recompute | reviewer B | twin displayed total (not physical) |
|---|---|---|---|
| 1104 | +7.73 % | +7.7 % | +8.41 % |
| 1152 | -0.502 % | -0.50 % | -0.539 % |
| 1200 | -0.389 % | -0.39 % | -0.414 % |
| 1984 | +2.445 % | +2.4 % | +2.683 % |

They agree within rounding. The 1% omission test on the twin's physical share FAILS at 1104 and 1984 and passes at 1152 and 1200, the same verdicts as before.

**(2) The light-only signed shares in the section 2.2 prose were WRONG at the first three clocks.** v1 printed "-0.7% / +0.04% / +0.03% / +0.24%"; the signs were inverted at 1104, 1152 and 1200. The correct values are ref_light / d_t M|_R in the primary:

| clock | light-only share, this recompute | reviewer B |
|---|---|---|
| 1104 | +0.695 % | +0.695 % |
| 1152 | -0.040 % | -0.040 % |
| 1200 | -0.026 % | -0.026 % |
| 1984 | +0.237 % | +0.237 % |

They agree within rounding. The conclusion |light| <= 0.7 % (light-alone 1% test passes, using the total's eps; red-team item 3) is unchanged. For reference, the primary's heavy-only shares are +7.91, -0.537, -0.397 and +2.449 %.

**(3) The C2 collar form KEEPS the factor v_R,** as derived in ADDENDUM_A3 and section 2.3: (d_t M)_ref,R = -(8 pi R^2 v_R / N_R^2) <K_rr,kl> E_kl(u_dot). The printed FableAPI_vrc1f form without v_R is not used.

**(4) 16 pi G K_L = 0.13-0.15 is a MASSIVE proxy** (m a = 0.05), NOT a massless value. The m = 0 cubic K was not computed (the doubler nodes have eps = 0 on the grid). Section 5's sentence must read "16 pi G K at m a = 0.05 (massive proxy) = 0.13-0.15; at m a = 0.2, 0.12-0.14; massless value not computed", and must not be cited as K_L at m = 0.

**Comparability (reviewer B):** route B's frozen-record bridging replay at 1104 is not yet matched to these records, so the routes are NOT yet comparable. Nothing here is a two-route agreement claim.

## ROUND 3 (lead #665): clock 1104 on the common base

**Frozen:** ADDENDUM_A5_vrc620_round3_v1.md (670b540f), frozen before any round-3 number.

**Blindness:** BLIND to route B. No route B file was opened. The route B values quoted in AstraAPI_vrccmp2_v1 are used nowhere: not in any definition, tolerance, verdict or comparison below.

**Scripts:** vrc_round3_v1.py (cc96c0ca) and vrc_kmid_v1.py (e2a8db58), unchanged since launch. They ran on box CPUs 12-23 at nice 10:
- primary on CPUs 12-17, 836 s;
- twin, then cubic, on CPUs 18-23.
All three .out files are free of Tracebacks. Every A5 item was produced, so no new script or addendum was needed.

**Records:**
- PRIMARY C_M1nl23: r_0003190, with r_0003062/3126/3254/3318 for the rates.
- TWIN C_M1nl25off: r_0003156, with r_0003028/3092/3220/3284.

**Common-base check:** fields(x) reproduces the stored Nb and vb to:
- primary: 1.67e-15 and 8.88e-15;
- twin: 2.33e-15 and 1.71e-14.

### R3.1 Vertex reconciliation (primary, A5 item 1)

The stored record is used exactly, with no re-solve. All three operators run in one code and share B (the rows' JVP, Richardson at eps_rel 1e-4 and 5e-5). err = eps_JVP + eps_GMRES.

| operator | V_HL (eps 1e-4) | V_HL (eps 5e-5) | eps_JVP | eps_GMRES | GMRES rel res (1e-4, 5e-5) |
|---|---|---|---|---|---|
| J_full = I + B(chi_H + chi_L)/V | -0.12760087968198544 | -0.1276008788747034 | 8.07e-10 | 3.5e-14 | 2.7e-13, 9.8e-13 |
| J_0 = I + B chi_H/V | -0.1276165677054504 | -0.12761656692669618 | 7.79e-10 | 1.0e-13 | 7.8e-13, 9.7e-13 |
| J_bare = I | -0.1280298404683894 | -0.1280298396814483 | 7.87e-10 | 0 | 0, 0 |

**Differences** (first two columns are values from the 1e-4 run):

| difference | value | err |
|---|---|---|
| dV = V_full - V_0 | +1.5688023464965895e-5 | 2.87e-11 |
| V_full - V_bare | +4.2896078640394797e-4 | 2.04e-11 |
| V_0 - V_bare | +4.132727629389821e-4 | 8.3e-12 |
| dV, difference-free linear form | +1.5688007911794022e-5 | 2.5e-12 (linear-form GMRES res 1.6e-12) |
| dV/V_0 | -1.2293093088959383e-4 | 2.2e-10 |

- Raw dV and the linear form differ by 1.56e-11. That is inside the raw error of 2.9e-11.
- **Change from my current base (v4, a3/v4_on_t1104.json, d7912206) to the common base: 0, 0 and 0** for V_HL_full, V_HL0 and dV, bit for bit. v4 already ran on the stored record with no re-solve, so the round-3 base and the v1-v1.2 base are the same object.
- In v1-v1.2, "re-solved baseline" in section 2.1 means the J_0 operator, not a re-solve of the fields.
- The heavy chi_H reference Hessian moves V_HL by +4.13e-4. Adding chi_L moves it by a further +1.57e-5.

### R3.2 Signed rate (primary, A5 item 2)

dGamma/Gamma0 = [sum_c V_full,c^2 - sum_c V_0,c^2] / sum_c V_0,c^2. The channel set is the heavy channels h1-h5 (cfg chans k = 1..5), with w_c = 1, on the common base.

| | value |
|---|---|
| Gamma0 = sum_c V_0,c^2 | 3.4535337353648935e-3 |
| Gamma_full | 3.4527033222380506e-3 |
| **dGamma/Gamma0 (SIGNED)** | **-2.404531678203417e-4** |
| eps_JVP, eps_GMRES | 4.52e-10, 2.1e-12 |
| total error | 4.54e-10 |

**Per channel (V_0 -> V_full):**
- h1: -0.01409045358686142 -> -0.014088327276649758
- h2: -0.02373311082875623 -> -0.023729556050446188
- h3: -0.029782233265058175 -> -0.029778019543478345
- h4: -0.031437974579193104 -> -0.03143452496809075
- h5: -0.028572795445581483 -> -0.028570451843320407

**Checks and reading:**
- An independent jq re-sum of the per-channel values gives -2.4045316782047e-4, which agrees to 1e-16.
- |V| falls in EVERY channel. So the sign (a rate decrease) holds for ANY positive channel weights, not only w_c = 1.
- The value sits inside v1.2's bound |dGamma|/Gamma0 <= 2 eta + eta^2 = 2.46e-4.
- One-percent omission (manifest): PASS.

### R3.3 Moving-root reference contribution (A5 item 3)

**Definition:** moving_s = fixed_s + R_dot dM_ref,s/dR, where:
- M_ref,s(R) = G_R . (-t_s/V), with G_R the exact rows adjoint;
- d/dR is central differences at hd and hd/2, Richardson-combined;
- R_dot = (4 D8 - D16)/3.

**Light source:** the law's w_l dW2_L(u), labelled "NOT in this L0 run, and not in the twin balance".

**Root quantities:**

| | primary (on) | twin (off) |
|---|---|---|
| R, N_R | 69.4658301566053, 0.9814540935848048 | 69.46507538581614, 0.9813429733487186 |
| R track (t-16, t-8, t, t+8, t+16) | 69.4396, 69.5275, 69.4658, 69.1276, 68.1970 | 69.4439, 69.5302, 69.4651, 69.1188, 68.1655 |
| R_dot (D8, D16) | -0.020380686984556935 +- 0.0046126 (-0.0249933, -0.0388312) | -0.020964053614344873 +- 0.0047462 (-0.0257102, -0.0399488) |
| d_t M\|_R | -0.03555491313876106 +- 0.00246594 | -0.03654311473663796 +- 0.00248862 |

**Primary, per species:**

| species | fixed (eps_fixed) | dM_ref/dR (err) | moving (err) | moving - fixed | share of d_t M\|_R (err) |
|---|---|---|---|---|---|
| heavy | -2.812700064753619e-3 (4.49e-5) | 9.596158292526358e-3 (4.3e-7) | **-3.0082763631678593e-3 (8.92e-5)** | -1.956e-4 | **+8.46 % (0.84)** |
| light (law's) | -2.472536404356286e-4 (3.9e-6) | 7.189960869540256e-4 (3.4e-8) | **-2.6190727462695987e-4 (7.2e-6)** | -1.47e-5 | +0.737 % (0.071) |
| total | -3.059953705189248e-3 (4.88e-5) | 1.0315154379480383e-2 (4.7e-7) | **-3.2701836377948194e-3 (9.64e-5)** | -2.102e-4 | +9.20 % (0.91) |

**Twin, per species** (heavy is PHYSICAL; light and total are printed only):

| species | fixed (eps_fixed) | dM_ref/dR (err) | moving (err) | moving - fixed | share (err) |
|---|---|---|---|---|---|
| heavy | -2.825320546275228e-3 (4.55e-5) | 9.64021519673088e-3 (4.5e-7) | **-3.027418534513316e-3 (9.12e-5)** | -2.021e-4 | **+8.28 % (0.81)** |
| light (not in balance) | -2.48392771242588e-4 (3.9e-6) | 7.22925103577478e-4 (3.5e-8) | -2.635482118731421e-4 (7.4e-6) | -1.52e-5 | (+0.721 %) |
| total (not physical) | -3.073713317517816e-3 (4.94e-5) | 1.0363140300308358e-2 (4.8e-7) | -3.2909667463864585e-3 (9.86e-5) | -2.173e-4 | (+9.01 %) |

**Errors:**
- err(moving) is propagated as frozen.
- The share errors (my propagation, not frozen) add err(moving)/|d_t M| and |moving| err(d_t M)/d_t M^2.

**Verdicts (manifest one-percent omission):**
- **Heavy and total: FAIL.** The moving-root reference is 8.3-9.2 % of d_t M|_R, so it is RETAINED in the root balance.
- **The law's light alone: PASS.** It is 0.737 +- 0.071 %, upper edge 0.81 % < 1 %.
- Moving vs fixed: the R_dot term adds about 7 % to the fixed-radius value in each species. Its own error is |dM/dR| err(R_dot) = 4.4e-5 (heavy), so the shift is 4.4 times that error.
- The fixed-radius values reproduce v1.2's table, which serves as an internal reproduction check.

**Cross-check (printed, not used):** the local form 4 pi R^2 (rho_ref + w J_ref)|_R without the w feedback gives dM_ref/dR:
- heavy 1.0527e-2 / light 7.80e-4 (primary);
- heavy 1.0574e-2 / light 7.84e-4 (twin).
That is about 9.7 % (heavy) and 8.5 % (light) above the adjoint. The w feedback is a 10 % effect on dM_ref/dR, and therefore about a 0.7 % effect on the moving value.

### R3.4 Collar on the explicit Gaussian window (A5 item 4)

**Window and sums:**
- Window: lambda = erfc((y - R)/(sqrt2 sigma_w))/2.
- Production back-averaging (sigma 4a); root, fields and u_dot as in R3.3.
- WINDOWED = the full site sum.
- COLLAR = the sum over |r - R| <= 3 sigma_w.
- REMAINDER = WINDOWED - COLLAR (exact).
- BOUND = sum over |r - R| > 3 sigma_w of |site density|.

**Frozen PASS rule:** |remainder| <= bound AND bound <= 1e-3 |windowed|.

**Primary:**

| sigma_w/a | species | windowed | collar | remainder | bound | bound/\|windowed\| | n sites in | replacement |
|---|---|---|---|---|---|---|---|---|
| 2 | heavy | -2.7135038395171597e-3 | -2.352977889254591e-3 | -3.605259502625687e-4 | 3.6885649921400226e-4 | 0.1359 | 12 | FAIL |
| 2 | light | -2.3874433013415953e-4 | -2.0632310273735748e-4 | -3.242122739680205e-5 | 3.3020060570105096e-5 | 0.1383 | 12 | FAIL |
| 2 | total | -2.9522481696513192e-3 | -2.5593009919919483e-3 | -3.92947177659371e-4 | 4.018439929432453e-4 | 0.1361 | 12 | FAIL |
| 4 | heavy | -2.4755512868892353e-3 | -2.4983739620231026e-3 | +2.2822675133867248e-5 | 2.638109645311158e-5 | 0.01066 | 24 | FAIL |
| 4 | light | -2.1851245094968932e-4 | -2.2018886393836913e-4 | +1.6764129886798099e-6 | 2.0470246030206177e-6 | 0.00937 | 24 | FAIL |
| 4 | total | -2.6940637378389245e-3 | -2.718562825961472e-3 | +2.449908812254733e-5 | 2.8329099171842172e-5 | 0.01052 | 24 | FAIL |
| 8 | heavy | -1.7679773099238114e-3 | -1.7697479605337388e-3 | +1.7706506099273746e-6 | 1.770651347010228e-6 | 1.0015e-3 | 48 | **FAIL (knife edge)** |
| 8 | light | -1.5793943933774019e-4 | -1.5802730424317832e-4 | +8.786490543813679e-8 | 9.964732502148211e-8 | 6.31e-4 | 48 | **PASS** |
| 8 | total | -1.9259167492615516e-3 | -1.9277752647769175e-3 | +1.8585155153658366e-6 | 1.858519153295268e-6 | 9.65e-4 | 48 | **PASS** |

**Twin:**

| sigma_w/a | heavy (PHYSICAL) windowed / collar / remainder / bound / rel | replacement | light, total (printed) |
|---|---|---|---|
| 2 | -2.7252434658267977e-3 / -2.3648859895391435e-3 / -3.6035747628765423e-4 / 3.689652860485682e-4 / 0.1354 | FAIL | FAIL, FAIL |
| 4 | -2.4846565751836546e-3 / -2.5079900025628188e-3 / +2.3333427379164124e-5 / 2.6933312686254677e-5 / 0.01084 | FAIL | FAIL, FAIL |
| 8 | -1.7722473096273437e-3 / -1.7739711600635408e-3 / +1.723850436197074e-6 / 1.723851052963131e-6 / 9.73e-4 | **PASS** | PASS, PASS |

**For reference, the v4 local chain collar C1** (a3/v4_*_t1104.json):
- on: heavy -2.993419424137266e-3, light -2.6140302921428735e-4, total -3.2548224533515534e-3;
- off: heavy -3.0114066424567695e-3, light -2.6300253450671257e-4, total -3.274409176963482e-3.

**Verdicts:**
- |remainder| <= bound holds in every case, so the finite-width remainder is BOUNDED everywhere.
- The collar REPLACES the windowed contraction only at sigma_w = 8a: primary light and total, and twin heavy (physical).
- At 2a and 4a, and for the primary heavy at 8a (bound 1.0015e-3, over the threshold by 1.5e-6 relative), the WINDOWED contraction is KEPT, as frozen.
- The windowed value itself depends on sigma_w at the order-one level. The total goes -2.95e-3 / -2.69e-3 / -1.93e-3 at 2/4/8a, against the sharp-window fixed value -3.06e-3 and C1 -3.25e-3.
- These Gaussian-window numbers are a different object from v1.2's section 2.3. There, sigma was the back-averaging width under a sharp cut; here the cut itself is Gaussian at sigma 4a production averaging.

### R3.5 Cubic K at m a = 0.2, unrounded (A5 item 5)

**Grid:** MIDPOINT, p_i = -pi + (2i + 1) pi/128, i = 0..127. All 128 points are listed in kmid_v1.json (grid_points_128), from -3.117048960983623 to +3.117048960983623 in steps of pi/64.

| entry | formula, 128^3 mid | formula, 256^3 mid | spectral, 128^3 mid | \|formula - spectral\| | \|128 - 256\| |
|---|---|---|---|---|---|
| c11 = kappa_11,11 | 0.009834182248158777 | 0.009834182247987322 | 0.009834182248156572 | 2.2e-15 | 1.7e-13 |
| c12 = kappa_11,22 | -0.003048063693959298 | -0.0030480636939466885 | -0.0030480636939491535 | 1.0e-14 | 1.3e-14 |
| kappa_12,12 (grad 1, vel 2, grad 1, vel 2) | 0.014156435012591707 | 0.014156435012586416 | 0.01415643501257193 | 2.0e-14 | 5.3e-15 |
| kappa_12,21 | -0.003048063693959298 | -0.0030480636939466885 | -0.0030480636939491535 | 1.0e-14 | 1.3e-14 |
| c44(sym) = (kappa_12,12 + kappa_12,21)/2 | 0.005554185659316205 | 0.005554185659319864 | 0.005554185659311388 | 4.8e-15 | 3.7e-15 |

**Shear convention:** for the symmetric strain S = (e1 e2 + e2 e1)/2, K(S, S)/|S|^2 = 2 c44(sym) = 0.01110837131863241.

**Reading:**
- On this grid, kappa_12,21 = c12 exactly.
- c44(sym) is the symmetrised shear modulus. kappa_12,12 alone (0.01416) is NOT c44.
- The values agree with section 3's rounded m 0.2 row (0.0098342 / -0.0030481 / 0.0055542).
- 16 pi G K at G/a^2 = 1/4: uniaxial 4 pi c11 = 0.12358, shear 8 pi c44 = 0.13959 (m a = 0.2; massive, not K_L at m = 0).
- Both error estimates are <= 1.7e-13, so PASS at the 1e-5 target.

### R3.6 CORRECTION 31 Oct 2026: the dE_R/du enclosures were not printed

v1-v1.2 said only that "dE_R/du enclosures are in the JSONs". The values are:
- dEdu_ref = ref_total x dt/du / D_R;
- dEdu_total = d_t M|_R x dt/du / D_R;
- each enclosed over the three dt/du values and the five-record D_R values.

| run t (JSON md5) | dE_R/du reference part [min, max] | dE_R/du total [min, max] |
|---|---|---|
| on 1104 (d7912206) | [-5.538557e-3, -3.289199e-3] | [-6.435487e-2, -3.821862e-2] |
| on 1152 (20525bd3) | [+1.927189e-3, +2.749674e-3] | [-4.771740e-1, -3.344413e-1] |
| on 1200 (4c06e00b) | [+1.476181e-3, +2.580639e-3] | [-6.097154e-1, -3.487703e-1] |
| on 1984 (734ef63f) | [+1.583054e-3, +2.045089e-3] | [+5.892669e-2, +7.612521e-2] |
| off 1104 (010f3d3b) | [-5.619008e-3, -3.305052e-3]* | [-6.680391e-2, -3.929348e-2] |
| off 1152 (419809d1) | [+1.813444e-3, +2.541803e-3]* | [-4.716450e-1, -3.364940e-1] |
| off 1200 (5c1bc026) | [+1.452145e-3, +2.936333e-3]* | [-7.098965e-1, -3.510748e-1] |
| off 1984 (a154942d) | [+1.737732e-3, +2.272614e-3]* | [+6.477277e-2, +8.471013e-2] |

1408 has no root in either run, so it has no enclosure.

\* The twin's reference part uses the TOTAL (heavy + light). Per v1.2 correction (1), only heavy is physical there.

The fixed-radius values are those in the JSONs; the moving-root analogue at 1104 follows from R3.3 by the same factors.

### R3.7 SELF RED-TEAM

1. **Collar heavy at 8a is a knife edge.** bound/|windowed| = 1.0015e-3 against 1e-3. The FAIL is the frozen verdict, and I do not reinterpret it.
   - The remainder equals the bound to 7 digits at 8a (and to about 2 % at 2a). The out-of-collar sites are single-signed, so the bound is tight, not conservative.
   - The bound is an exact sum on the dense grid and carries no discretisation error estimate.
2. **The window is not a harmless convention.** The windowed contraction moves 35 % between sigma_w = 2a and 8a. A collar "passing" at 8a means only that the 8a-windowed quantity is localised. It does not mean that quantity equals the sharp-root debit (-3.06e-3).
3. **The error on R_dot is large (23 %)**, and the R track is not monotone: R rises from t-16 to t-8, then falls.
   - The Richardson (4 D8 - D16)/3 assumes an h^2 error. With D8 and D16 differing by 55 %, that assumption is weak.
   - The moving-minus-fixed shift is 4.4 times its R_dot-propagated error. Its size, about 7 %, is therefore uncertain at the 25 % level.
   - None of this changes any verdict: heavy fails at both fixed and moving, and light passes at both.
4. **Light in R3.3/R3.4 is the LAW's light reference** (w_l dW2_L), not a stored engine t. It is labelled as not in this L0 run. The twin's light and total are not physical.
5. **The vertex errors (8e-10) are numerical only:** JVP step plus GMRES. They take the stored fields and covariance as exact. The base change is exactly 0, so any route A / route B vertex difference at the 1e-3 level cannot come from my base drifting. Whether it comes from operator definitions, I cannot say blind.
6. **The signed rate uses w_c = 1 over h1-h5.** Its sign is weight-independent (every |V_c| decreases); its magnitude is not.
7. **The cubic formula-vs-spectral agreement (1e-14) is on the SAME grid,** so it is a check of the algebra, not of quadrature. The 128-vs-256 difference is the quadrature check. The kappa_12,21 = c12 identity is structural, not an independent confirmation.
8. **The share errors** (the R3.3 parentheses) are my own linear propagation, not a frozen rule.
9. **Scope:** a single clock (1104). Nothing here is a two-route agreement claim; comparison is for the lead.

### R3.8 Output md5s (vrc620/r3/; box <internal>/r3 identical)

- r3_on_t1104.json: 88019ef63ab66aaa3ccd8a8bb3a3d51c
- r3_off_t1104.json: f3ffd6fd48fd44d9f6ac2a9991f9ac40
- kmid_v1.json: df483f0b4f71e406e3b1c484cedaa643
- logs: r3_on_t1104.out d8118b60, r3_off_t1104.out 441e356e, kmid_v1.out 41918349
- vrc_round3_v1.py cc96c0ca; vrc_kmid_v1.py e2a8db58; ADDENDUM_A5 670b540f

## DECIDING CALCS (#669)

**Frozen:** ADDENDUM_A6_vrc620_deciding_v1.md (e9044543), frozen before any new number.

**Blindness:** still BLIND to route B. FableSub_vrcsign_v1 was not opened; only the lead's #669 text, which has route B's numbers removed, was used.

**Clock:** 1104. Primary C_M1nl23 r_0003190; twin C_M1nl25off r_0003156.

**What is new and what is extracted:**
- (a), (b) and (c) are extracted from r3_on_t1104.json (88019ef6) and r3_off_t1104.json (f3ffd6fd). Item (a) is confirmed by rerunning vrc_probe_tsign_v2.py (405297f1) unchanged.
- (e) is new: vrc_kwave_v1.py (8c86a2aa), run on box CPUs 12-23 at nice 10. No Tracebacks.

### (a) Vertex on the record's OWN state, no light t

**The round-3 "stored record" vertex IS exactly this object.** Evidence:
- **Record contents.** r_0003190.npz holds x, Nb and vb, plus per heavy channel no_rho, no_J, w, tN and tv, plus light no_rho_l0, no_J_l0 and w_l0. It holds NO light tN/tv (unzip -l key list).
- **Sources.** The probe rerun (r4/probe_v2_r3190.json, daa9da72) gives max|x - (sum_s w_s no_s - sum_h t_h)/V| = 3.47e-18 (N row) and 1.73e-18 (v row), against |x| max 0.0216 and 0.0083. The no_s sum includes the light no_l0; the t sum is heavy only. The opposite sign misses by 1.1e-3 and 2.7e-3.
- **Fields.** fields(x) reproduces the stored Nb/vb to 1.67e-15 and 8.88e-15.
- **Operator.** The vertex solves use this x, Nb and vb with no re-solve (vrc_round3_v1.py lines 92 and 165-167). V_HL0 uses J_0 = I + B chi_H/V. chi_L enters only J_full, as a response Hessian, never as a source.

**Values (unrounded, primary):**
- **V_HL0 = -0.1276165677054504 +- 7.79e-10**, where the error is eps_JVP 7.79e-10 plus eps_GMRES 1.0e-13.
- **dV = V_full - V_0 = +1.5688023464965895e-5 +- 2.87e-11.**
- The difference-free linear form gives +1.5688007911794022e-5 (+- 2.5e-12).
- dV/V_0 = -1.2293093e-4.

The twin has no heavy-light vertex: its light does not source the rows.

### (b) Rate, one line

**sum_c (|V_full,c|^2 - |V_0,c|^2) / sum_c |V_0,c|^2 = -2.404531678203417e-4 +- 4.5e-10** (unit weights, h1-h5, from the stored per-channel vertices; a jq re-sum gives -2.4045316782047e-4).

Per-channel spread of (V_full,c^2 - V_0,c^2)/V_0,c^2:

| channel | ratio |
|---|---|
| h1 | -3.0179e-4 |
| h2 | -2.9954e-4 |
| h3 | -2.8295e-4 |
| h4 | -2.1944e-4 |
| h5 | -1.6404e-4 |

- The range is [-3.018e-4, -1.640e-4], with unweighted mean -2.536e-4.
- All five channels are negative, so the sign holds for any positive weights.
- The value changes by at most a factor 1.8 depending on the weights.

### (c) Moving root: dM_ref,s/dR at R, per a

M_ref,s(R) = G_R . (-t_s/V), with G_R the exact rows adjoint. The derivative is Richardson-combined from central differences at hd and hd/2 (hd = 0.25 a); the error is |D(hd) - D(hd/2)|.

| | primary C_M1nl23 (R = 69.4658 a) | twin C_M1nl25off (R = 69.4651 a) |
|---|---|---|
| **heavy** | **+9.596158292526358e-3 +- 4.3e-7** | **+9.64021519673088e-3 +- 4.5e-7** (physical) |
| light (the law's w_l dW2_L; NOT in this L0 run) | +7.189960869540256e-4 +- 3.4e-8 | +7.22925103577478e-4 +- 3.5e-8 (not in balance) |
| total | +1.0315154379480383e-2 +- 4.7e-7 | +1.0363140300308358e-2 +- 4.8e-7 (not physical) |

**Sign and size:**
- All are POSITIVE: the reference mass enclosed grows outward at the root.
- The relative errors are about 5e-5, so the sign is decided by about 2e4 sigma.
- With R_dot = -0.02038 +- 0.0046 (primary), the moving term R_dot dM/dR is NEGATIVE: heavy -1.956e-4, about 4.4 sigma.

**Cross-check (printed, not used).** The local form 4 pi R^2 (rho_ref + w J_ref)|_R without the w feedback gives heavy 1.0527e-2 and light 7.80e-4 (primary). That is the same sign, about 10 % larger.

### (e) Cubic K: the [100] transverse plane wave

**Setup.** m a = 0.2, MIDPOINT grid. The wave is v = e2 cos(k x1): k = e1, e = e2 (100Ty; 100Tz is identical). Its coefficient is c(h) = -2 E2(h k)/h^2 at h = 0.08, 0.04, 0.02, 0.01, Richardson in h^2.

| quantity | n = 128 | n = 256 |
|---|---|---|
| c(h) | 0.014109178941655023, 0.01414455693188527, 0.014153461439478193, 0.014155691365224978 | 0.014109178941646235, 0.014144556931871187, 0.014153461439467753, 0.014155691365218137 |
| **wave coefficient (Richardson)** | **0.014156435012571264** (Richardson err 1.09e-12) | **0.014156435012565912** (1.09e-12) |
| kappa_12,12 (full-gradient contraction) | 0.014156435012591707 | 0.014156435012586416 |
| kappa_12,21 | -0.003048063693959298 | -0.0030480636939466885 |
| c44(sym) = (kappa_12,12 + kappa_12,21)/2 (symmetric contraction) | 0.005554185659316205 | 0.005554185659319864 |
| kappa_12,12 - kappa_12,21 | 0.017204498706551004 | 0.017204498706533105 |
| \|wave - kappa_12,12\|/kappa_12,12 | 1.44e-12 | 1.45e-12 |
| \|wave - c44(sym)\|/c44(sym) | 1.549 | 1.549 |

**MAP VERDICT (frozen rule): the exact [100] transverse wave REPRODUCES the FULL-GRADIENT contraction kappa_12,12. It does NOT reproduce c44(sym)**, which it exceeds by a factor 2.549.

**Supporting results:**
- The longitudinal control 100L gives 0.009834182248156175, matching c11 to 2.6e-13.
- The wave's grid error (128 vs 256) is 5.4e-15.
- The formula values are bit-identical to kmid_v1.json.
- The earlier endpoint-grid spectral sum (a3/kspec_v1.json, n 128) gave 0.014156435012560567 for the same wave, against its own kappa contraction of 0.014156435012581254.

**The map, stated explicitly.** grad v has the single entry (1, 2) = dv. Split it as grad v = S + A, with S = (dv/2)(e1 e2 + e2 e1) and A = (dv/2)(e1 e2 - e2 e1). Cubic symmetry gives K(S, A) = 0. Then:
- **symmetric (strain) part:** K(S, S)/|S|^2 = kappa_12,12 + kappa_12,21 = 2 c44(sym) = 0.01110837131863241.
- **antisymmetric (rigid-turn) part:** K(A, A)/|A|^2 = kappa_12,12 - kappa_12,21 = 0.017204498706551. Per unit rotation angle omega (A = omega(e1 e2 - e2 e1)), K(A, A)/omega^2 = 2(kappa_12,12 - kappa_12,21) = 0.034408997413102, which is reviewer A's "cost of a rigid turn".
- **the transverse wave carries both:** kappa_12,12 = c44(sym) + (kappa_12,12 - kappa_12,21)/2, i.e. 0.0055541857 + 0.0086022494 = 0.0141564350.

**Consequences:**
- The plane-wave spectral sum confirms the FULL tensor entry, not the strain modulus.
- The root debit's physical object is the symmetric restriction, c44(sym) for shear and c11/c12 for normal strain (reviewer A's point, adopted).
- Any sentence or collar form that uses a transverse-wave coefficient as "the shear K" overstates the strain stiffness by 2.55x at m a = 0.2.
- This K is NOT rotation-invariant: the lattice charges a rigid turn, at 61 % of the [100] transverse-wave coefficient.

Check on my own earlier text:
- v1.2 section 3 and R3.5 quote 16 pi G K for shear as 8 pi c44(sym), the symmetric object. That is correct as stated.
- The C2 cubic collar (v1.2 section 2.3) uses <K_rr,kl> E_kl(u_dot) with E the strain, also symmetric. It is counted for sign only.

### LABEL for history C_M1nl23 (print before any VRC sentence)

"History C_M1nl23 is a spherically symmetric lattice-chain evolution on a radial chain with lattice spacing a, from the release of preparation F_M1_R4 under the NEW vacuum law. Its parameters:
- **Grid:** heavy block 2048 sites, light block 8192 sites; dense field grid 0.25 a, extension 10.
- **Coupling and charge:** G/a^2 = 1/4. Release energy E = 147.0857, so r_s = 2GE = 73.5428 a. The energy drift is |C| <= 1e-7 at the four clocks below.
- **Heavy matter:** m a = 0.2, five channels (k, n_k, w) = (1, 6, 16), (2, 5, 32), (3, 4, 48), (4, 3, 64), (5, 2, 80), so sum n_k w = N_B = 800.
- **Light matter:** a massless chain (m = 0), angular k_l = 1, weight w_l = 16 on the full chain (the engine evolves the Gamma = + block with 2 w_l = 32).
- **Smoothing:** sources and fields are back-averaged with the history's kernel of width sigma = 4 a.
- **Sources:** S(C - P_star) - [rpol(u) - rpol(u_star)], with P_star and u_star fixed at release. W_2 comes from the vaclaw_v3 rule (eq (2) on the chain symbol d = (sin p, k/r, m), q = sin(2p)/2, kappa-gradients counted). The reference moments t are never clipped. The light block is L0, with no light counterterm (uncontrolled, E29).
- **Code:** engine m3l_eng_v14, run script m3l_run_v24 from t = 862 a (earlier m3l_run_v21/v22 in the C_M1nl20/22 lineage).
- **Records:** every 8 a.
- **Clocks:** t/a = 1104, 1152, 1200 and 1984, i.e. t/r_s = 15.012, 15.664, 16.317 and 26.977 (record steps 3190, 3574, 3958, 8734).
- **Twin:** C_M1nl25off is the same law and grid with the light sourcing nothing."

**Sources for each value:**
- **release_meta.json** (box zone640/C_M1nl23, md5 c86d0a27862a6d7b1b9ced8f3c48678b):
  - cfg: ns_h 2048, ns_l 8192, hd 0.25, ext 10, sigma 4.0, m 0.2, k_l 1, w_l 16, chans, NB 800 (also R0 320, tau 2.0, rule "paper", uout 0.5, rec_every 4);
  - E0 = 147.08566295226302 and twoGE = 73.54283147613151, so G = 0.25;
  - prep dir F_M1_R4.
- **m3l_run_v10.py** (8ae13e5c, base_cfg): the same cfg literals. **m3l_eng_v8.py** (11e15323): chans = (k, n_k, w) and wl_eff = 2 w_l. Record r_0003190 stores w_l0 = 32 and w_h0 = 16.
- **MANIFEST_m3nl607_v1.md** (9afb2510):
  - engine sources, vaclaw_v3 (vaclaw_w2_v3 6f9ab685), adapter m3l_w2adapt_v1 (7f2bb86e), light L0 / E29, records on the exact 8 a grid;
  - m3l_run_v22 (e5d8ce0c), m3l_eng_v14 (493d4b7c), m3l_run_v24 (7716577d);
  - C_M1nl23 from C_M1nl22/ckpt_0001372 (md5 79d03742, t 862);
  - the twin C_M1nl25off with m3l_run_v25 (31d7e264).
- **records.jsonl** (box pl2/rec_C_M1nl23, md5 0f1d0ec48aceafa873b4f974168fd672):
  - t = 1104.000000000004 at step 3190, 1152 at 3574, 1200 at 3958, 1984.0000000000114 at 8734;
  - C_drift -7.89e-8, -6.74e-8, -6.66e-8, +9.75e-8.
- t/r_s = t/73.54283147613151.

### SELF RED-TEAM (#669)

1. **(a) is a statement about MY object.** The round-3 vertex sits on the stored x, which carries the light no_l0 moments but no light t. If route B's "own state" re-solves fields, or includes a light t, it is a different object. I have not seen B and do not infer which it uses.
2. **Light weight convention: a possible factor-2 trap.** My chi_L (and the law's light t in (c)) uses w_l = 16 on vaclaw's full two-band light chain (d = (sin p, k/r, 0)). The engine evolves one Gamma block with 2 w_l = 32, and NOTES (window split) records that the two blocks contribute equally.
   - Those are the same weighting only if vaclaw's light chain is the full chain, which is how it is defined.
   - No direct FD check of chi_L against the engine's block Hessian was made in this round.
   - A factor 2 here would double dV, the light part of the rate, and every light reference number.
   - **Flagged for the comparison; not resolved here.**
3. **The derivative in (c) is precise but one-sided in method.** Its 5e-5 relative error is the finite-difference error of G_R in R only. It does not include the linear interpolation of G_R in y (hd = 0.25 a). The no-w-feedback local form differs by 10 %, so the w feedback is material.
4. **(e) is exact for the parent symbol** (naive cubic Dirac, m a = 0.2), not for the radial chain. The 2.55x wave/strain ratio and the nonzero rigid-turn cost are properties of that parent's K.
5. **Label caveats:**
   - R0, tau, rule and uout are quoted as cfg values without interpretation.
   - The runs before t = 862 a used m3l_run_v21/v22. I cite the manifest for that lineage; I did not re-hash those checkpoints.
6. **Scope:** single clock. No two-route claim.

### Output md5s (vrc620/r4/; box identical)

- probe_v2_r3190.json: daa9da7240ba2a8d820ea301f39bd4a2
- kwave_v1.json: 0286130de8d7233d8931bc26958f3498
- kwave_v1.out: de8ee842ec8deea40b45fbbfe23e2a61
- vrc_kwave_v1.py: 8c86a2aab75e7f714df321d6f7125ae1
- ADDENDUM_A6: e90445431731ba57d20ce3093e9b8e93

## METADATA AND CUBIC K RESTATED (#671; reviewer B vrcfa)

**Scope.** This section is lookup and restatement only. No new computation was run. Every value comes from a file cited with its md5, and nothing is inferred. It supersedes the LABEL in "DECIDING CALCS (#669)" where they differ: that label gave r_s only as 2GE0 and gave the clock in those units.

**Files:**
- [RM] box <internal>/release_meta.json, c86d0a27862a6d7b1b9ced8f3c48678b (run cfg and release).
- [P6] scripts/m3l_prepM_v6.py, e6af5202c196dafd0e78c909d5f22735 (preparation).
- [E8] scripts/m3l_eng_v8.py, 11e1532380fef80e1dbe64eaecdceeca.
- [LB] scripts/m3l_lib_v2.py, 4e0dba4708744357c27ec209f4bdcb59.
- [MF] m3nl607/MANIFEST_m3nl607_v1.md, 9afb2510f492ff9a37e18fd77fe337bc.
- [RJ] box <internal>/records.jsonl, 0f1d0ec48aceafa873b4f974168fd672.
- [PR] vrc620/r4/probe_v2_r3190.json, daa9da7240ba2a8d820ea301f39bd4a2.
- [PA] deliveries/lead_gaps/PAPER_flat_bdca0bd.tex, c9e3439d45702b0fe13d2d966f1d9c3a.

### Metadata for history C_M1nl23

| item | value | source |
|---|---|---|
| nominal r_s/a | **80** = 2 G m N_B = 2 x 0.25 x 0.2 x 800 | [P6] rs_nom = 2*0.25*m*NB; [RM] NB 800 |
| release charge radius 2GE0/a | 73.54283147613151 (E0 = 147.08566295226302; eta_E = 0.9192853934516438) | [RM] twoGE, E0, eta_E |
| m_H a | 0.2 | [RM] cfg m |
| m_L a | **0**: the light is "Gamma = S sigma_2 block of the massless chain" | [LB] light_block_band; [E8] light D(key) built with no mass |
| G/a^2 | 1/4 | [RM] twoGE/(2 E0) = 0.25; [P6] 2*0.25 |
| heavy channels (k, n_k, w) | (1,6,16) (2,5,32) (3,4,48) (4,3,64) (5,2,80); sum n_k w = N_B = 800; 2048 sites each | [RM] cfg chans, ns_h; [E8] "(k, n_k, w)" |
| light channel | k_l = 1; w_l = 16 on the full massless chain. The engine evolves the Gamma = + block with wl_eff = 2 w_l = 32 (record w_l0 = 32). 8192 sites | [RM] k_l, w_l, ns_l; [E8] wl_eff |
| averaging kernel | the history's back-averaging Gaussian, **sigma = 4a**; dense grid hd = 0.25a, ext 10 | [RM] sigma, hd, ext; [MF] "the history's kernel" |
| step tau | 2.0 a | [RM] tau |
| release radius R0 | 320 a = 4 x nominal r_s; preparation rule "paper" (binding rule, prepare_dust v36d as implemented), uout 0.5, prep dir F_M1_R4 | [RM] R0, rule, uout, prep; [P6] R0 = r0fac*rs_nom; NOTES_mark3_layer_r1 sec 5 for "binding rule" |
| Gauss-law sources | S(C - P_star) - [rpol(u) - rpol(u_star)], with P_star and u_star from release; W_2 = vaclaw_v3 (vaclaw_w2_v3 6f9ab685) through m3l_w2adapt_v1 (7f2bb86e); t never clipped | [MF] Engine |
| light sources included? | **YES** in C_M1nl23. x = (sum_s w_s no_s - sum_h t_h)/V to 3.47e-18, where the no sum includes the light no_l0. The light has NO reference term (L0, no light counterterm, uncontrolled E29; the record has no light t). The twin C_M1nl25off has light_off (the light evolves but sources nothing) | [PR]; [MF] Light block, TWIN |
| code at the clocks | m3l_eng_v14 (493d4b7c), m3l_run_v24 (7716577d) from t = 862a, from C_M1nl22/ckpt_0001372 (79d03742) | [MF] Amendment A2 |

### Clock map

**Clock and origin:**
- The run's t is its flat-slice evolution time in units of a, with **t = 0 at release**: [RJ] has step 0 at t 0.0, with E equal to the release E.
- The paper's records also use flat-slice T, and its collapse figures quote clocks in units of NOMINAL r_s ([PA] lines 1280 and 1362).
- So the convention is the same (flat slice, origin at release), as far as the paper states it.

| t/a | record step [RJ] | T/r_s,nominal (r_s = 80a) | t/(2GE0) (73.5428a) |
|---|---|---|---|
| 1104 | 3190 | **13.80** | 15.012 |
| 1152 | 3574 | **14.40** | 15.664 |
| 1200 | 3958 | **15.00** | 16.317 |
| 1984 | 8734 | **24.80** | 26.977 |

### Is C_M1nl23 the r_s/a = 64 setup that the paper says "is not formed"? NO.

**The paper's statement.** [PA] line 2044: "At r_s/a=64, the two-field history is not formed \todo{re-evolve the two-field size-64 history with Eq. (dm-reference)'s sources}."

**The paper's reference collapse** ([PA] lines 1285-1296):
- ma 0.2, G/a^2 1/4, nominal r_s 64a, N_B 640, 2GE0 59.07a;
- tau 8a, ten 2048-site chains;
- dust columns 5,4,3,2,2 in k = 1..5, weights 16k;
- prepared at R0 = 4 r_s; sigma = 4a.

**SAME in C_M1nl23:**
- ma 0.2, G/a^2 1/4, sigma 4a, weights 16k;
- 2048-site heavy chains in k = 1..5;
- R0 = 4 x nominal r_s;
- flat-slice clock with its origin at release.

**CHANGED in C_M1nl23:**
- **size:** N_B 640 -> 800 (columns 6,5,4,3,2), so nominal r_s 64a -> 80a and 2GE0 59.07a -> 73.54a [RM, P6];
- **step:** tau 8a -> 2a [RM];
- **law and sources:** the new vacuum law, with sources S(C - P*) - [rpol(u) - rpol(u*)] through vaclaw_v3. [MF] is governed by eq:dm-reference (2927a70d9c9);
- **fields:** two fields, the heavy plus a sourcing light block (L0, no light counterterm).

**Reading:**
- C_M1nl23 is a two-field history with eq:dm-reference's sources at NOMINAL SIZE 80, not 64. It does not literally discharge the paper's size-64 todo.
- The paper may cite it only under its own label, with r_s/a = 80 nominal.

### (e) Cubic K at m a = 0.2: c11, c12, c44 unrounded on ONE explicit grid

**Grid:** MIDPOINT 128^3, the same set in each axis. p_i = -pi + (2i + 1) pi/128, i = 0..127, i.e. pi/128 times the odd integers from -127 to +127. Explicitly (from kmid_v1.json, df483f0b):

-3.117048960983623, -3.067961575771282, -3.018874190558942, -2.9697868053466014, -2.9206994201342606, -2.8716120349219203, -2.82252464970958, -2.773437264497239, -2.724349879284899, -2.675262494072558, -2.6261751088602177, -2.5770877236478773, -2.5280003384355365, -2.478912953223196, -2.4298255680108554, -2.380738182798515, -2.3316507975861747, -2.2825634123738343, -2.2334760271614935, -2.184388641949153, -2.1353012567368124, -2.086213871524472, -2.0371264863121317, -1.9880391010997909, -1.9389517158874505, -1.88986433067511, -1.8407769454627694, -1.7916895602504288, -1.7426021750380885, -1.6935147898257479, -1.6444274046134073, -1.5953400194010667, -1.5462526341887264, -1.4971652489763858, -1.4480778637640452, -1.3989904785517047, -1.3499030933393643, -1.3008157081270237, -1.2517283229146832, -1.2026409377023426, -1.1535535524900022, -1.1044661672776614, -1.055378782065321, -1.0062913968529807, -0.95720401164064, -0.9081166264282996, -0.8590292412159592, -0.8099418560036185, -0.7608544707912781, -0.7117670855789373, -0.662679700366597, -0.6135923151542566, -0.5645049299419158, -0.5154175447295755, -0.4663301595172351, -0.4172427743048943, -0.36815538909255396, -0.31906800388021317, -0.2699806186678728, -0.22089323345553247, -0.17180584824319167, -0.12271846303085132, -0.07363107781851097, -0.024543692606170175, 0.024543692606170175, 0.07363107781851097, 0.12271846303085132, 0.17180584824319167, 0.22089323345553247, 0.2699806186678728, 0.31906800388021317, 0.36815538909255396, 0.4172427743048943, 0.4663301595172351, 0.5154175447295755, 0.5645049299419158, 0.6135923151542566, 0.662679700366597, 0.7117670855789373, 0.7608544707912781, 0.8099418560036185, 0.8590292412159588, 0.9081166264282992, 0.9572040116406404, 1.0062913968529807, 1.055378782065321, 1.1044661672776614, 1.1535535524900018, 1.202640937702343, 1.2517283229146834, 1.3008157081270237, 1.349903093339364, 1.3989904785517044, 1.4480778637640457, 1.497165248976386, 1.5462526341887264, 1.5953400194010667, 1.644427404613407, 1.6935147898257474, 1.7426021750380887, 1.791689560250429, 1.8407769454627694, 1.8898643306751097, 1.93895171588745, 1.9880391010997913, 2.0371264863121317, 2.086213871524472, 2.1353012567368124, 2.1843886419491527, 2.233476027161494, 2.2825634123738343, 2.3316507975861747, 2.380738182798515, 2.4298255680108554, 2.4789129532231957, 2.528000338435537, 2.5770877236478773, 2.6261751088602177, 2.675262494072558, 2.7243498792848984, 2.7734372644972396, 2.82252464970958, 2.8716120349219203, 2.9206994201342606, 2.969786805346601, 3.0188741905589422, 3.0679615757712826, 3.117048960983623


| invariant | value (formula, 128^3 midpoint) | spectral sum, same grid | \|formula - spectral\| | grid error \|128 - 256 midpoint\| |
|---|---|---|---|---|
| c11 = kappa_11,11 | **0.009834182248158777** | 0.009834182248156572 | 2.2e-15 | 1.7e-13 |
| c12 = kappa_11,22 | **-0.003048063693959298** | -0.0030480636939491535 | 1.0e-14 | 1.3e-14 |
| c44(sym) = (kappa_12,12 + kappa_12,21)/2 | **0.005554185659316205** | 0.005554185659311388 | 4.8e-15 | 3.7e-15 |

**Related entries:**
- The full-gradient entries are kappa_12,12 = 0.014156435012591707 and kappa_12,21 = -0.003048063693959298.
- The shear convention is K(S, S)/|S|^2 = 2 c44(sym) for S = (e1 e2 + e2 e1)/2.
- The [100] transverse plane wave reproduces kappa_12,12, not c44(sym) (see #669 (e)).
- Sources: r3/kmid_v1.json (df483f0b) and r4/kwave_v1.json (0286130d). This repeats the values already in R3.5.

### SELF RED-TEAM (#671)

1. **"Same origin"** rests on the paper's own wording (flat-slice T, quoted in nominal r_s, clocks from release). I did not trace the code of the paper's size-64 history. If its T = 0 is not the release, the nominal T/r_s column shifts by that offset.
2. **Light weight convention.** The light chain is massless and split into two Gamma blocks; the engine evolves one block at 2 w_l. This supports reading w_l = 16 on the full chain, which is the convention my chi_L uses. It is still not a direct finite-difference check of chi_L against the engine block (#669 red-team 2 stands, weakened).
3. **"Ten 2048-site chains" in the paper versus C_M1nl23's five heavy two-component 2048-site channels plus one 8192-site light block.** I report both as stated and do not claim the counts correspond.
4. **The paper version** is the most recent PAPER_flat in lead_gaps (bdca0bd). If reviewer A cited another version, its line numbers may differ.

## CHI_L WEIGHTING CHECK (A7): the factor-2 flag is CLOSED

**Frozen:** ADDENDUM_A7_vrc620_chiL_v1.md (817b1623). It was revised once before any number existed: T2's direction became pure flow, with the reason stated in the file.

**Script and run:** vrc_chiL_v1.py (12533587), on box CPUs 12-23 at nice 10, 3530 s, no Traceback.

**Output:** r5/chiL_v1.json (24d193a8) and chiL_v1.out (b7e60256).

**Inputs:**
- Clock 1104, record r_0003190, with its +-8a and +-16a neighbours for u_dot.
- Directions ud = u_dot (Richardson) and ub = the sharp-window back-averaged u at R = 69.4658a.

**Method.** The exact sea Hessian chi_exact(a, b) = -2 Re sum_{i<0, p>0} conj(<p|dh_a|i>)<p|dh_b|i>/(l_p - l_i) is taken from a dense eigh of each Hamiltonian. h is linear in (N, v), so dh = band(dN, dv) exactly.

**VERDICT: CONFIRMED.** w_l = 16 on vaclaw's full light chain is the same weighting as the engine's wl_eff = 2 w_l = 32 on its one Gamma = + block. No light number changes. dV, dGamma/Gamma0, and every light row (fixed, moving, dM_ref,L/dR, collar, and the c19 light row) stand as printed.

### T0: vaclaw chi_dens is the exact Hessian of vaclaw W_2 (light, w = 1)

| direction | chi_dens | FD (Richardson) | rel |
|---|---|---|---|
| ud | -1.2242432472579942e-7 | -1.2242432464649593e-7 | 6.5e-10 |
| ub | -3.51171501971711e-3 | -3.5117150184973155e-3 | 3.5e-10 |
| far flow bump (1500a) | -2.0475421110885698e-4 | -2.0475420848434003e-4 | 1.3e-8 |

The worst case is 1.3e-8, against the frozen 1e-6, so **PASS**.

### T1 (decisive): the engine's block at weight 32 equals the full massless chain at weight 16

**Spectra:**
- The full chain (light_full_band = heavy_band at m = 0) has n = 16384, 8192 negative states, gap 4.93e-4.
- Each Gamma block has n = 8192, 4096 negative states, gap 7.05e-4.
- spec(full) = spec(+) U spec(-) to 3.3e-15.

| pair | 16 x chi_exact[full] | 32 x chi_exact[Gamma = +] | 32 x chi_exact[Gamma = -] | rel (+ vs full) | rel (- vs +) |
|---|---|---|---|---|---|
| (ud, ud) | -2.142652691789804e-4 | -2.1426526917896796e-4 | -2.142652691789668e-4 | 5.8e-14 | 5.4e-15 |
| (ud, ub) | +1.7444009632979043e-4 | +1.7444009632941741e-4 | +1.7444009632941964e-4 | 2.1e-12 | 1.3e-14 |
| (ub, ub) | -5.690912999923685e-2 | -5.6909129999237126e-2 | -5.690912999923728e-2 | 4.9e-15 | 2.7e-15 |

Every relative difference is <= 2.1e-12, against the frozen 1e-8, so **PASS**. Each Gamma block carries exactly half of the full chain's sea Hessian, so 2 w_l on one block = w_l on the full chain.

### T2: vaclaw's normalization is per FULL chain

**Setup:** the heavy-type full chain heavy_band(8192, k = 1, m a = 0.2), weight 1, on the record's fields, with far bumps in the controlled regime. Its gap is 0.0167.

| bump | vaclaw chi | exact chi | ratio |
|---|---|---|---|
| pure flow, centre 4000a, width 400a | -9.152352940721111e-5 | -9.15229250834664e-5 | 1.0000066 |
| pure flow, centre 1500a, width 200a | -1.830450451948213e-4 | -1.8304073407220093e-4 | 1.0000236 |
| mixed (dN = 0.3 g), 4000a (information) | -9.146720534101565e-5 | -9.146701554798346e-5 | 1.0000021 |

|ratio - 1| <= 2.4e-5, so the normalization is **per FULL chain**. Together with T1, this closes the factor-2 question: the same vaclaw.coeffs code serves the heavy (validated here, and the engine's own stored heavy t = w_ch dW_2 to 0.1 %) and the light at m = 0.

### T3 (information, NOT a weighting factor): the law's massless second-gradient Hessian vs the exact sea Hessian at 1104

| pair | vaclaw chi_L (w 16) | 16 x exact full massless chain | vaclaw/exact |
|---|---|---|---|
| (ub, ub) | -5.618744031547376e-2 | -5.690912999923685e-2 | 0.987 |
| (ud, ub) | +2.4266809755929314e-4 | +1.7444009632979043e-4 | 1.391 |
| (ud, ud) | -1.9587891956127907e-6 | -2.142652691789804e-4 | 0.009 |
| far flow bump 1500a | -3.2760673777417117e-3 | -3.9026201903168568e-3 | 0.839 |
| far flow bump 4000a | -5.771893929682613e-4 | -1.9020278337657187e-3 | 0.303 |

**Fixed-R light term -chi_L(ub, ud)/N_R:**
- law (round 3): -2.472536404356286e-4, reproduced bit for bit here;
- exact sea Hessian: -1.777363785733882e-4.

**Reading:**
- For the MASSLESS light, the second-gradient law is uncontrolled (eta_M = 1/k = 1 at every r; E29). On the record's own directions it differs from the exact filled-sea Hessian by -1 % to +39 %, and by a factor of about 100 on (ud, ud).
- (ud, ud) is dominated by the clock rate dN. W_2 = -sum F(v)/N carries no N-gradient term; the exact sea responds to clock gradients.
- This is the law's DEFINITION of the light reference (vaclaw_v3 is the counterterm), not a weighting error. So no number is "corrected".
- It is a material caveat on any physical reading of the light rows: an exact-sea light reference would move the fixed light term by -28 %, in the same sign.
- The heavy (m a = 0.2) agrees with exact to about 1e-5 in its controlled regime (T2).

### Averaging kernel: the exact formula in the engine

**Source:** m3l_lib_v2.py (4e0dba4708744357c27ec209f4bdcb59). The kernels K0 (lines 245-252) and K1 (254-263) are used by Grid (265-289).

- **Density kernel.** K0(y, r, sigma) = [r/(sqrt(2 pi) sigma y)] exp(-(y - r)^2/(2 sigma^2)) (1 - exp(-2 y r/sigma^2)), with limit (r/(sqrt(2 pi) sigma)) exp(-r^2/(2 sigma^2)) 2r/sigma^2 at y = 0. This is the angular average of a 3-D isotropic Gaussian of width sigma placed at radius r, i.e. r/(sqrt(2 pi) sigma y) [e^{-(y-r)^2/2sigma^2} - e^{-(y+r)^2/2sigma^2}].
- **Current kernel.** K1(y, r, sigma) = [r/(sqrt(2 pi) sigma y)] exp(-(y - r)^2/(2 sigma^2)) [(1 + e^{-t}) - (1 - e^{-t})/(t/2)], with t = 2yr/sigma^2 (series t^2/6 - t^3/12 + t^4/40 for t < 1e-3). This is the radial-vector projection of the same Gaussian.
- **Truncation and grid.** Both are truncated at |y - r| <= ext sigma = 40a and sampled on the dense grid y = i hd, hd = 0.25a, with W0[i, n] = a K0(y_i, r_n, sigma) and W1[i, n] = a K1(y_i, r_n, sigma).
- **Forward smoothing** (sources to dense; Grid.smooth, line 286): rho_dense = W0 rho_site, J_dense = W1 J_site.
- **Back-averaging** (fields to sites; Grid.back, line 288): N_b,n = [W0^T (w N_d)]_n/V_n and v_b,n = [W1^T (w v_d)]_n/V_n, where w_i = 4 pi y_i^2 hd c_i (trapezoid) and V_n = 4 pi r_n^2 a.
- **Parameters:** sigma = 4a, hd = 0.25a, ext = 10 ([RM] cfg). Used by fields_from_sources (lines 338-348) at every step.

### Clock origin from code

- **The run's t.** m3l_run_v10.py (8ae13e5c), release(), line 107: "st.t = 0.0; st.step = 0; ..." is executed right after the release_meta.json of F_M1_R4 is written (line 106). records.jsonl ([RJ] 0f1d0ec4) has step 0 at t 0.0 with E = E_release. The C_M1nl23 lineage resumes checkpoints with their stored t (C_M1nl22/ckpt_0001372 at t 862), so t/a is time since that release.
- **The paper's T.** Not traced in code. The paper's size-64 histories come from a different engine lineage (the prepare_dust v36d ladders), which I did not open. The paper's text says records use flat-slice T, with clocks quoted in nominal r_s ([PA] lines 1280 and 1362).
- **"Same origin" therefore stands on:** (i) code for the run (t = 0 at release); (ii) the paper's text for the paper. It is not code-verified on the paper side.

### SELF RED-TEAM (A7)

1. T1 and T2 compare against the EXACT filled-sea Hessian of the static h[u] (the ground-state sea of the instantaneous h). The engine's light state is dynamic (the covariance C evolves). That does not affect the weighting question, which is a property of h and the block decomposition. It does mean T3 compares the law to the static sea, not to the run's evolved light state.
2. T2's ratio (1 + 2.4e-5) shows that the normalization is per full chain. Its small residual is the higher-gradient correction in the controlled regime.
3. The T3 discrepancy is large and physically relevant: the massless second-gradient law misses clock-gradient (N') response and is uncontrolled. Every light number in v1-v1.5 is "the law's light reference (vaclaw_v3, E29)", not the exact sea's.
4. A7 was revised once before any number existed (T2 direction). That is disclosed above and in the file.

### Output md5s (vrc620/r5/; box identical)

- chiL_v1.json: 24d193a8b95ff708d8a23fbf35b0df63
- chiL_v1.out: b7e6025643bb75e8e32ed94b2addb0c1
- vrc_chiL_v1.py: 125335874379f2cfd9b79bd053dcdb5b
- ADDENDUM_A7: 817b1623426ca5aa57c8ad9989037f72

## COMMON-ARRAY TANGENT AUDIT AND COLLAR CUTOFF (#673, A8)

**Frozen:** ADDENDUM_A8_vrc620_commonarray_v1.md (f62d855e), frozen before any number.

**Blindness (#674):**
- AstraAPI_vrccmp3_v1 was NOT opened.
- From route B's common file (<internal>/arrays_1104_v1.npz, a3a800b3) only the definition metadata was printed (vrc_readB_meta_v1.py, 789ebe3f), and its INPUT arrays were read for the elementwise check.
- No route B result was opened. The only route-B-side figures used are the lead's relative gaps quoted in #673 (2.17 %, 0.069 %, 2.39 %), and only in the reading at the end.

**Scripts and run:**
- vrc_export_v1.py (f8bdf157);
- vrc_audit_v1.py (e638a3c7), on box CPUs 12-23 at nice 10, with no Traceback.

**Outputs:**
- r6/audit_v1.json (c3f3e482) and audit_v1.out (6784427b);
- tangent solutions r6/audit_solutions_B.npz (021486ad), also exported as <internal>/solutions_1104_routeA_v1.npz (same md5).

### Step 1: export

<internal>/arrays_1104_routeA_v1.npz (df176562), with an .md5 file. It holds:
- j_L;
- j_H = (s_C - t)/V;
- the old s-only covector;
- s_C, t, V and the weights;
- u_b, s_b and a readme.

### Step 2: mapping onto route B's definitions

- **Same:** the pairing sum V (a_N du_N + a_v du_v), the light source s_C,L (no light t; block 32 = full chain 16, closed in A7), u_b, s_b, V and the operator form.
- **DIFFERENT, the heavy contraction covector.** Route A v1-v1.6 used s_C,c only. The common definition is j_H,c = s_C,c - t_c(u_b), with t_c the stored engine reference moment.

**Elementwise check** (mine against route B's inputs, max|diff|/max|B|):

| array | rel diff |
|---|---|
| j_L | 2.4e-19 |
| V, u_b, s_b | 0 |
| j_H[c], c = 1..5 | <= 1.1e-16 |
| my OLD s-only covector against B's j_H | 0.042, 0.044, 0.040, 0.034, 0.034 |

So the arrays are COMMON, and the old covector differs by 3.4-4.4 % of max.

### Step 3: blind solves on the common arrays (stored base, rhs B j_L)

**GMRES relative residuals:**
- full: 2.7e-13 and 9.8e-13;
- zero: 7.8e-13 and 9.7e-13;
- bare: exact (no solve).

**Errors:**
- eps_JVP = |eps 1e-4 - eps 5e-5|;
- eps_cond = |<j_H[c], D^-1 r>|, from one refinement solve D e = r. This is conditioning-aware and includes D^-1.

**Per channel, common covector j_H,c = (s - t)/V:**

| channel | V_bare | V_0 | V_full | dV = V_full - V_0 (err) |
|---|---|---|---|---|
| h1 | -0.014140696343912717 | -0.01409002202289544 | -0.014087915171648757 | +2.1068512466825007e-6 (2.9e-12) |
| h2 | -0.023826822285570197 | -0.023742139428680777 | -0.023738632205968693 | +3.507222712084701e-6 (5.0e-12) |
| h3 | -0.02990441790242568 | -0.029801635502443732 | -0.02979749497179506 | +4.140530648674007e-6 (6.4e-12) |
| h4 | -0.031561417123860465 | -0.031465461584507554 | -0.03146210523880493 | +3.3563457026222765e-6 (7.3e-12) |
| h5 | -0.02868156974819002 | -0.02860548477905871 | -0.028603249006236058 | +2.2357728226528317e-6 (7.2e-12) |
| **total** | **-0.12811492340395908** | **-0.12770474331758622** | **-0.1276893965944535** | **+1.5346723132719786e-5 (2.9e-11)** |

**Totals:**
- The per-V errors are eps_JVP of about 8e-10 (total) and eps_cond <= 4e-14.
- dV/V_0 = -1.201734777740741e-4.
- My exported (s - t) covector gives identical values to 1e-16.

**Unit-weight signed rate** (h1-h5, common covectors): **dGamma/Gamma0 = -2.3470545388819998e-4 +- 4.5e-10** (eps_JVP 4.5e-10, eps_cond 1.0e-12). Per channel: -2.990e-4, -2.954e-4, -2.779e-4, -2.133e-4 and -1.563e-4.

**The SAME solves contracted with my OLD s-only covector** reproduce v1.3 exactly:
- V_0 = -0.1276165677054504;
- V_full = -0.12760087968198544;
- dV = +1.5688023464965895e-5;
- rate = -2.404531678203417e-4.

**Covector effect alone** (same tangents, s - t against s-only):

| quantity | change |
|---|---|
| V_0 | +0.0691 % |
| dV | -2.177 % |
| rate | -2.391 % |

Per-channel dV ratio, new/old:

| h1 | h2 | h3 | h4 | h5 |
|---|---|---|---|---|
| 0.9908 | 0.9866 | 0.9826 | 0.9730 | 0.9540 |

The ratio varies by channel, so it is not a common factor.

**Reading.** The lead's #673 gaps are V_HL0 0.069 %, dV 2.17 % and rate 2.39 %, with a channel pattern that is not a common factor. They coincide with the covector-definition effect computed here. This is a comparison of relative sizes only; I hold no route B values. The vertex and rate disagreement is therefore fully accounted for by route A's v1-v1.6 heavy covector, which omitted -t_c(u_b), not by the solves or the operator.
- On the common definition, route A's values are the bolded ones above.
- They supersede v1.3/v1.4's V_HL0, dV and rate as "common-definition" values. The old values stay valid as the s-only contraction, which is what they were.

**Cross-evaluated residuals: PENDING.** Route B's solution is not yet in <internal>/routeB; only its input arrays are there. My tangents are exported for route B to cross-evaluate. When B's solution appears, I will apply my D_0 and D_full to it with a separate script (vrc_xres_v1.py), as A8 (vi) specifies.

### Calc 2: collar cutoff

**Definitions:** f = u_bar_R,w - u_b = back((lambda - 1) Nd, (lambda - 1) vd), g = u_dot, P_w = 1{|r - R| <= 3 sigma_w}. The Euler null holds: max over sites of |b_x(u_b, g)| = 0 exactly. Values are raw chi, with the debit -(.)/N_R in brackets, on hd = 0.25a.

| sigma_w | species | A = chi(P_w f, g) | C = sum P_w b(f, g) | A - C (cutoff-boundary term) | \|A - C\|/\|C\| |
|---|---|---|---|---|---|
| 2 | heavy | 1.7729698242038724e-3 [-1.80647e-3] | 2.309339781523455e-3 [-2.35298e-3] | -5.363699573195826e-4 | 23.2 % |
| 2 | light | 1.577209997472512e-4 | 2.0249665378269794e-4 | -4.477565403544672e-5 | 22.1 % |
| 2 | total | 1.9306908239511234e-3 [-1.96717e-3] | 2.511836435306152e-3 [-2.55930e-3] | -5.811456113550288e-4 | 23.1 % |
| 4 | heavy | 2.262484788757998e-3 [-2.30524e-3] | 2.452039352333267e-3 [-2.49837e-3] | -1.8955456357526912e-4 | 7.7 % |
| 4 | light | 2.0116826488809062e-4 | 2.1610526187410028e-4 | -1.4936996986009653e-5 | 6.9 % |
| 4 | total | 2.4636530536460885e-3 [-2.51021e-3] | 2.668144614207367e-3 [-2.71856e-3] | -2.044915605612786e-4 | 7.7 % |
| 8 | heavy | 1.750718503135032e-3 [-1.78380e-3] | 1.7369263804792008e-3 [-1.76975e-3] | +1.3792122655831122e-5 | 0.79 % |
| 8 | light | 1.571933002869071e-4 | 1.55096544647639e-4 | +2.096755639268105e-6 | 1.35 % |
| 8 | total | 1.9079118034219392e-3 [-1.94396e-3] | 1.8920229251268399e-3 [-1.92778e-3] | +1.5888878295099336e-5 | 0.84 % |

**Dense-grid error** (|hd 0.25 - hd 0.125|): at most 2.6e-8 (A), 1.5e-8 (C) and 1.3e-8 (A - C), at every sigma_w and species.

**Reading:**
- C times -1/N_R reproduces round 3's "collar" (R3.4) bit for bit.
- The two restrictions differ by a stencil boundary term: 23 % at 2a, 7.7 % at 4a and 0.8 % at 8a. So "the collar" is defined only together with WHERE the projector acts (on f, or on the density). Any comparison must use the same one.

### SELF RED-TEAM (#673)

1. The decisive finding is a DEFINITIONAL one: my heavy covector omitted -t_c. I did not choose between the definitions on physics grounds. The common definition was set by the lead and route B, and I report both.
2. Matching the lead's quoted relative gaps is circumstantial, not a value-level comparison. The cross-evaluated residual (pending) and route B's own values on the common arrays are the real test.
3. The eps_cond is a one-step refinement estimate. It is tiny (<= 4e-14) because GMRES converged to 1e-12. The dominant error is the finite-difference JVP (about 8e-10), which is shared by every route A number.
4. t_c(u_b) uses the stored engine t. The convention probe showed it equals w_c[DW2(u_b) - DW2(u*)], not w_c DW2(u_b). If route B's t_c(u_b) were the latter, the arrays would not have matched to 1e-16; they did.

## ROUND 5 (#677, A9)

**Frozen addenda:**
- A9 (7845a85a);
- A9a (e649af53): GMRES rtol 1e-13 -> 1e-12;
- A9b (8cbc28d4): rtol 1e-10, maxiter 30.

All three were dated and written before any round-5 number. vrc_direct_v1 and v2 stalled at the JVP residual floor without writing a number; I killed both (own pids) and used neither. A9b's "per-solve print" was not implemented in v3 (an edit failed before launch). The residuals are printed per level instead.

**Script and run:** vrc_direct_v3.py (0382d1e8), on box CPUs 12-23 at nice 10, no Traceback.

**Output:** r7/direct_v3.json (175c4415) and direct_v3.out (b1446225).

**Blindness:** vrccmp4 was not opened, nor any route B result.

### (1) Cross-evaluated residuals: PENDING

Route B's tangent solution is not in <internal>/routeB; only arrays_1104_v1.npz is there (checked with find -name). vrc_xres_v1.py (cba63116) is on the box and runs as soon as the file appears. My tangents are exported (solutions_1104_routeA_v1.npz, 021486ad).

### (2) Direct correction

**Method:**
- Solve D_0 du_0 = B j_L, then D_full dz = -B V^-1 chi_L du_0, and take dV_c = <cov_c, dz>.
- Refinement sequence: JVP eps_rel 2e-4, 1e-4, 5e-5 (quoted), 2.5e-5 and 1.25e-5. Each level has its own B in every action and its own du_0.
- GMRES achieved relative residuals: du_0 2.6e-12 to 7.8e-11, dz 6.2e-12 to 6.8e-11, all info 0.

**Errors:**
- eps_JVP = the maximum deviation over the five levels from the 5e-5 value;
- eps_solve = |<cov, D_full^-1 r_dz>| + |<cov, D_full^-1 B V^-1 chi_L D_0^-1 r_0>|, from refinement solves;
- err = eps_JVP + eps_solve.

**Per channel, common covector s_C - t:**

| ch | V_0 (err) | dV direct (err) |
|---|---|---|
| h1 | -0.014090021945152787 (6.6e-10) | 2.1068499435410418e-6 (2.0e-12) |
| h2 | -0.023742139293203093 (1.1e-9) | 3.5072204458587677e-6 (3.5e-12) |
| h3 | -0.02980163532819733 (1.5e-9) | 4.140527715973821e-6 (4.5e-12) |
| h4 | -0.03146546138909133 (1.7e-9) | 3.3563423886675195e-6 (5.1e-12) |
| h5 | -0.028605484590850855 (1.7e-9) | 2.23576956010256e-6 (5.0e-12) |

**Per channel, old covector s_C only:**

| ch | V_0 (err) | dV direct (err) |
|---|---|---|
| h1 | -0.014090453509317652 (6.6e-10) | 2.1263089119681698e-6 (2.0e-12) |
| h2 | -0.023733110693670643 (1.1e-9) | 3.5547760506013154e-6 (3.5e-12) |
| h3 | -0.029782233091376748 (1.5e-9) | 4.213718656910207e-6 (4.5e-12) |
| h4 | -0.03143797438449389 (1.7e-9) | 3.4496078007796632e-6 (5.1e-12) |
| h5 | -0.028572795258224377 (1.7e-9) | 2.3435990132194192e-6 (5.0e-12) |

**Totals and rate.** The rate is sum(2 V_0 dV + dV^2)/sum V_0^2 from the unrounded values, with errors propagated linearly.

| covector | V_0 (err) | dV (err; rel) | rate (err; rel) | rate over the 5 levels |
|---|---|---|---|---|
| **s_C - t** | -0.1277047425464954 (6.6e-9) | **+1.534671005414371e-5** (2.0e-11; 1.3e-6) | **-2.347052488254761e-4** (3.3e-10; 1.4e-6) | -2.3470518e-4 ... -2.3470555e-4 (spread 3.7e-10) |
| s_C only | -0.12761656693708331 (6.6e-9) | +1.5688010433478774e-5 (2.0e-11; 1.3e-6) | -2.4045296339666403e-4 (3.3e-10; 1.4e-6) | spread 3.7e-10 |

**Consistency:**
- Direct minus difference-of-solves at eps 5e-5 is -5e-12 to -1.3e-11 per channel, and -5.2e-11 in total. That is 2.5x the direct error. The difference-of-solves carries the larger GMRES stopping error of two separate solves (residual 7.8e-11). The direct value is the one quoted.
- Against the #673 audit (dV 1.5346723132719786e-5, rate -2.3470545388819998e-4), the changes are 1.3e-11 (8.5e-7 relative) and 2.1e-10 (8.8e-7 relative). Both are within the new errors.

**Do the new error bars explain the inter-route gap? NO.** The lead says the routes agree on dV and the rate to about 1e-4 relative. Route A's measured relative errors are now 1.3e-6 (dV) and 1.4e-6 (rate), from the full JVP-step refinement sequence plus conditioning-aware solve errors. A 1e-4 gap is about 75x these. It is not route A's FD-JVP or solve error, as measured.
- What remains is either route B's error budget or a sub-1e-4 definitional or discretization difference in the operator B or chi. Candidates are B by finite differences (route A) against an analytic tangent, and the stencil or endpoints in chi.
- The cross residuals in (1) will separate these. They are the next test, not a claim.

### (3) Width-8 density-cutoff collar (debit units -(.)/N_R)

| species, grid | W | Q | signed rem W - Q | eps_rem | eps_W | eps_Q |
|---|---|---|---|---|---|---|
| heavy, 0.25a | -1.7679773099238136e-3 | -1.769747960533742e-3 | +1.7706506e-6 | 1.7706513e-6 | 1.768e-5 | 1.745e-5 |
| heavy, 0.125a | -1.7679673462665563e-3 | -1.7697379302747233e-3 | +1.7705840e-6 | 1.7705847e-6 | 1.768e-5 | 1.745e-5 |
| light, 0.25a | -1.5793943933774038e-4 | -1.5802730424317854e-4 | +8.786e-8 | 9.965e-8 | 1.547e-6 | 1.527e-6 |
| light, 0.125a | -1.5793847202171706e-4 | -1.580263345682422e-4 | +8.786e-8 | 9.965e-8 | 1.547e-6 | 1.527e-6 |
| total, 0.25a | -1.925916749261554e-3 | -1.9277752647769207e-3 | +1.8585155e-6 | 1.8585192e-6 | 1.923e-5 | 1.898e-5 |
| total, 0.125a | -1.9259058182882734e-3 | -1.9277642648429655e-3 | +1.8584466e-6 | 1.8584502e-6 | 1.923e-5 | 1.898e-5 |

**Where eps_W comes from:**
- The grid part |W(0.25) - W(0.125)| is about 1e-8 (heavy 1.0e-8).
- eps_W is dominated by the u_dot rate term |W(D8) - W(D16)|/3, which is about 1.0 % of W.
- |rem| <= eps_rem everywhere.

**Test (my reading: |W - Q| + eps_rem <= 1e-3 (|W| - eps_W)):**

| species | lhs/rhs ratio | result |
|---|---|---|
| heavy | 2.02 | FAIL |
| light | 1.20 | FAIL |
| total | 1.95 | FAIL |

The verdict is the same on both grids.

**Other readings, from the same printed numbers:**
- **eps_rem alone <= 1e-3 (|W| - eps_W):** heavy 1.012 FAIL; light 0.637 PASS; total 0.975 PASS.
- **Literal Q + eps_rem <= ...:** true only vacuously, because Q < 0.

At width 8, the density-cutoff collar is a quantitative replacement for the light only, or for the total under the eps_rem-only reading. The heavy misses by 1 % under the eps_rem-only reading and by 2x under mine. **The full contraction W is retained.**

### SELF RED-TEAM (#677)

1. eps_JVP is the maximum deviation over a five-level eps sequence spanning 16x. It captures both the truncation (large eps) and the roundoff (small eps) sides. The rate drifts monotonically by 3.7e-10 across the sequence, which sets the error. A JVP that is wrong in a step-independent way, for example a systematic error of the forward rows solve, would not show here. Only the cross residual against an independent route can expose that.
2. Every GMRES solve stopped at <= 7.8e-11. The conditioning-aware terms (<= 1.5e-15 in dV) confirm that this is negligible in the covectors.
3. A9 needed two dated tolerance amendments (A9a, A9b), both before any number. The frozen error RULES did not change.
4. The collar test reading is mine, as stated and sent to main with a default. Every reading's numbers are printed.

## COLLAR TEST, LITERAL FORM (#678, A9c)

**Note:** ADDENDUM_A9c_vrc620_collar_literal_v1.md (9870590e).

**Definitions (lead #678):**
- Q = the outside majorant of |W - C|;
- C = the collar sum;
- eps_remainder = |Q(hd) - Q(hd/2)| + |Q(D8) - Q(D16)|/3;
- PASS iff Q + eps_remainder <= 1e-3 (|W| - eps_W).

**Disclosure:** the collar numbers existed before this clarification. v1.8 used A9's reading. This section re-tests the SAME stored numbers (r7/direct_v3.json, 175c4415) under the literal form, with no recomputation.

**Results** at width 8, in debit units:

| species, grid | W | eps_W | Q (outside majorant) | eps_remainder | Q + eps_rem | 1e-3 (\|W\| - eps_W) | ratio | VERDICT | W - C (reference) |
|---|---|---|---|---|---|---|---|---|---|
| heavy, 0.25a | -1.7679773099238136e-3 | 1.7683e-5 | 1.7706513470104984e-6 | 2.3117e-7 | 2.001822e-6 | 1.750294e-6 | 1.144 | **FAIL** | +1.7706506e-6 |
| heavy, 0.125a | -1.7679673462665563e-3 | 1.7683e-5 | 1.770584744740443e-6 | 2.3116e-7 | 2.001745e-6 | 1.750284e-6 | 1.144 | **FAIL** | +1.7705840e-6 |
| light, 0.25a | -1.5793943933774038e-4 | 1.5468e-6 | 9.964732502140618e-8 | 1.9145e-8 | 1.187921e-7 | 1.563926e-7 | 0.760 | **PASS** | +8.786e-8 |
| light, 0.125a | -1.5793847202171706e-4 | 1.5468e-6 | 9.96451397121727e-8 | 1.9144e-8 | 1.187889e-7 | 1.563916e-7 | 0.760 | **PASS** | +8.786e-8 |
| total, 0.25a | -1.925916749261554e-3 | 1.9230e-5 | 1.8585191532954692e-6 | 2.5140e-7 | 2.109920e-6 | 1.906687e-6 | 1.107 | **FAIL** | +1.8585155e-6 |
| total, 0.125a | -1.9259058182882734e-3 | 1.9230e-5 | 1.8584501914188929e-6 | 2.5139e-7 | 2.109840e-6 | 1.906676e-6 | 1.107 | **FAIL** | +1.8584466e-6 |

**|W - C|** equals the signed remainder (all positive here) and is below Q everywhere.

**Verdict (literal, #678):**
- At sigma_w = 8a the density-cutoff collar PASSES for the light only.
- It FAILS for the heavy (1.14x) and the total (1.11x), identically on both grids.
- The failure is driven by eps_remainder, the u_dot rate part of the majorant's refinement error (D8 vs D16), and by eps_W reducing |W|.
- This supersedes v1.8's collar verdict under A9's reading (heavy 2.02, light 1.20, total 1.95, all FAIL). The light changes from FAIL to PASS.

## CORRECTION 31 Oct 2026: the collar verdict under lead #678 (labels and record)

This section completes the #678 correction. The verdict itself was already given in v1.9's section "COLLAR TEST, LITERAL FORM (#678, A9c)". No number is recomputed: every value is read from r7/direct_v3.json (175c4415).

**(a) The majorant's own refinement error, per grid and per u_dot stencil.** Q is the outside majorant sum_{|r - R| > 24a} |b|/N_R, in debit units.

| species | grid | Q (Richardson u_dot) | Q (D8) | Q (D16) | grid term \|Q(0.25) - Q(0.125)\| | stencil term \|Q(D8) - Q(D16)\|/3 | eps_remainder |
|---|---|---|---|---|---|---|---|
| heavy | 0.25a | 1.7706513470e-6 | 2.0017553e-6 | 2.6950673e-6 | 6.66e-11 | 2.3110e-7 | 2.3117e-7 |
| heavy | 0.125a | 1.7705847447e-6 | 2.0016782e-6 | 2.6949587e-6 | 6.66e-11 | 2.3109e-7 | 2.3116e-7 |
| light | 0.25a | 9.964732502e-8 | 1.1171677e-7 | 1.6914452e-7 | 2.2e-12 | 1.9143e-8 | 1.9145e-8 |
| light | 0.125a | 9.964513971e-8 | 1.1171391e-7 | 1.6913849e-7 | 2.2e-12 | 1.9142e-8 | 1.9144e-8 |
| total | 0.25a | 1.8585191533e-6 | 2.1098514e-6 | 2.8638483e-6 | 6.90e-11 | 2.5133e-7 | 2.5140e-7 |
| total | 0.125a | 1.8584501914e-6 | 2.1097711e-6 | 2.8637337e-6 | 6.90e-11 | 2.5132e-7 | 2.5139e-7 |

The majorant's refinement error is dominated by its dependence on the u_dot stencil (13 % of Q). The grid term is about 1e-10.

**(b) THE verdict: lead #678, literal.** The test is majorant + its refinement error <= 1e-3 (|W| - eps_W).

| species | 0.25a: lhs / rhs (ratio) | 0.125a: lhs / rhs (ratio) | VERDICT |
|---|---|---|---|
| heavy | 2.001822e-6 / 1.750294e-6 (1.144) | 2.001745e-6 / 1.750284e-6 (1.144) | **FAIL** |
| light | 1.187921e-7 / 1.563926e-7 (0.760) | 1.187889e-7 / 1.563916e-7 (0.760) | **PASS** |
| total | 2.109920e-6 / 1.906687e-6 (1.107) | 2.109840e-6 / 1.906676e-6 (1.107) | **FAIL** |

The total's 2.5 % margin without the refinement error (0.975) is erased by the majorant's stencil dependence, which brings it to 1.107. This verdict is THE collar verdict for v1.8 (3).

**(c) Relabelled, for reference only.** v1.8's form |W - Q| + eps_rem <= 1e-3 (|W| - eps_W), where v1.8's "Q" was the collar C, is the **"|W - C| replacement test (not the lead's test)"**. Its ratios were heavy 2.02, light 1.20 and total 1.95. The signed remainder W - C (= |W - C| here) is +1.77065e-6 (heavy), +8.786e-8 (light) and +1.85852e-6 (total), all below the majorant.

**(d) The record.** A9b (8cbc28d4) did NOT record #678: it concerns the GMRES tolerance only. #678 is recorded in A9c (9870590e). A9c states that it was written AFTER the collar numbers existed, and that it changes only the test applied to the stored numbers.
