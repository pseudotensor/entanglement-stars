# B_SECOND_ROUTE_solve_v1.3 (solve lane, second route; DRAFT for the lead; v1.3 = v1.2 plus DETERMINISTIC storage enclosures on the 8 s8 slices, #597 / AstraAPI_bclose2a_v1)

The <B> second route and mass-bias audit for the paper at fecb5e4f0b6 (05_star.tex: "the bound <B> <= 0.080 on all 18 slices ... the second route on the other 17 slices and the mass-bias audit on all 18").

R_50 is the half-count radius (reviewer A r50f, reviewer B r50a/r50s). This lane was BLIND to the other route's files and per-slice values. Full record, frozen manifest (md5 c33d441e), gates, the declared v3 deviation, and SELF RED-TEAM: NOTES_bvref2_17.md beside this file.

**Coverage: all 18 slices.**
- prod (sigma/a 4): T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35, 40.
- s8 (sigma/a 8): 12, 13, 14, 16, 20, 25, 30, 40. Restored from the Drive backup and md5-verified.

**A/B and checks.**
- The driver on our B2 rec-120 state reproduces our 15.0 values: M 80.775, R_50 term 0.0048426, r_in 0.074209, R_t 0.00075, <B> 0.0798.
- The prod history's own 15.0 slice agrees to 4e-7.
- The half-count R_50 reproduces every R_50 the paper prints: 31.4 / 28.9 / 31.4a (prod 12 / 15 / 16) and 19.1 / 34.8a (s8 14 / 16).

| slice | r_in (r, M, term) | R_50 (r, M, term) | R_t (r, M, term) | <B> | <= 0.080 | max \|beta\|/M |
|---|---|---|---|---|---|---|
| prod 12 | none | 31.4, 59.555, 0.00384 | none | 0.0038 | yes | 2.3e-4 |
| prod 15 | 7.0, 13.887, 0.07421 | 28.9, 80.775, 0.00484 | 51.8, 103.525, 0.00075 | 0.0798 | yes | 3.6e-4 |
| prod 16 | 23.4, 46.766, 0.02592 | 31.4, 71.015, 0.00534 | 50.8, 101.644, 0.00086 | 0.0321 | yes | 3.8e-4 |
| prod 20 | 7.8, 15.489, 0.02435 | 17.9, 56.384, 0.00375 | 38.4, 76.857, 0.00133 | 0.0294 | yes | 3.3e-4 |
| prod 24 | 18.4, 36.773, 0.01665 | 24.6, 57.901, 0.00376 | 38.9, 77.826, 0.00109 | 0.0215 | yes | 3.9e-4 |
| prod 25 | 17.1, 34.259, 0.01729 | 24.6, 57.095, 0.00400 | 37.3, 74.586, 0.00145 | 0.0227 | yes | 3.6e-4 |
| prod 28 | 5.5, 10.914, 0.02704 | 16.0, 58.431, 0.00314 | 37.6, 75.111, 0.00102 | 0.0312 | yes | 4.4e-4 |
| prod 30 | 12.8, 25.562, 0.02912 | 21.4, 57.187, 0.00310 | 37.5, 75.020, 0.00102 | 0.0332 | yes | 4.1e-4 |
| prod 35 | 6.4, 12.755, 0.04120 | 17.2, 59.461, 0.00319 | 37.2, 74.437, 0.00102 | 0.0454 | yes | 4.2e-4 |
| prod 40 | 7.5, 15.107, 0.02828 | 16.8, 54.587, 0.00348 | 36.3, 72.619, 0.00121 | 0.0330 | yes | 4.0e-4 |
| s8 12 | none | 41.2, 56.533, 0.00291 | none | 0.0029 | yes | 1.7e-4 |
| s8 13 | 20.0, 40.030, 0.00345 | 23.3, 52.413, 0.00252 | 48.2, 96.414, 0.00091 | 0.0069 | yes | 2.7e-4 |
| s8 14 | 10.7, 21.372, 0.00579 | 19.1, 61.527, 0.00249 | 52.9, 105.815, 0.00045 | 0.0087 | yes | 3.7e-4 |
| s8 16 | 31.3, 62.575, 0.00466 | 34.8, 72.098, 0.00266 | 52.9, 105.815, 0.00034 | 0.0077 | yes | 3.9e-4 |
| s8 20 | none | 54.2, 59.779, 0.00334 | none | 0.0033 | yes | 2.4e-4 |
| s8 25 | none | 40.0, 54.580, 0.00258 | none | 0.0026 | yes | 2.4e-4 |
| s8 30 | none | 31.4, 56.188, 0.00230 | none | 0.0023 | yes | 3.3e-4 |
| s8 40 | 24.8, 49.603, 0.00279 | 25.9, 52.977, 0.00246 | 41.0, 81.960, 0.00077 | 0.0060 | yes | 3.7e-4 |

Term = 4(beta^2 + S1 + S2)/M^2, upper end. Per-mass beta, S1, S2 ranges and K = 0 values are in logs/summary.json and logs/summary_s8.json.

**Readings.**
- **<B> second route: <= 0.080 on all 18 slices.**
  - The maximum is 0.0798 at prod 15.0. Under a 100 percent Hessian error it lies in [0.07974, 0.07998].
  - Next is 0.0454 (prod 35). The s8 member stays <= 0.0087.
  - The inner root dominates where it exists.
- **Mass-bias audit on all 18 slices** (|beta_j|/M_j < 1%, the vref spec criterion): PASS on all 44 masses. The maximum is 4.4e-4, and the bias share of any term is <= 6.9e-4.
- **Declared items:**
  - The complex64 storage gate is now closed and quantified (sections below). On prod, the fp64 factors pass the gate (0.0) and storage moves no <B> by more than 6.6e-9. On s8, the deterministic enclosures (v1.3) keep every combined upper <= 0.0230.
  - The v3 change: the rows RK4 step = ell/16, required for s8 (ell 8). Prod is unchanged.
  - The route reuses our vref algebra. It is independent relative to the other lane, not within ours.

## v1.2: storage-precision closure (AstraAPI_bclosea_v1; full record in NOTES addendum 2)
**fp64 factors.**
- The five complex64 prod slices (15, 25, 30, 35, 40) were rebuilt in fp64 by exact forward replay from the nearest stored complex128 slice of the same history (12 -> 15, 24 -> 25, 28 -> 30, 28 -> 35, 28 -> 40). The replay step reproduces stored complex128 slices to 3.4e-14.
- R2 was rebuilt in fp64 from the fp64 eigh negative-set eigenvectors: ||R2 R2^dag - P||_F = 0.0. **The gate passes.**
- beta_j, S1_j and S2_j were recomputed on these factors:
  - over all 15 prod masses: |d beta| <= 1.6e-10, |d S1| <= 5.1e-8, |d S2| <= 1.5e-10, |d term| <= 9.0e-10;
  - prod 15: <B> 0.0798005591 (c64) -> 0.0798005583 (fp64).
- The 8 s8 slices have no complex128 slice. v1.2 bounded them by sampled re-rounding runs, which are NOT an upper bound (reviewer B bclose2a). **v1.3 replaces that with deterministic enclosures; see the v1.3 section below and the table rows marked v1.3.**

**Combined upper bound** = Hessian bracket (a 100% Hessian error, the vref 8b rule) + storage bound:

| slice | combined upper <B> | <= 0.080 | bias upper (2x for a 100% K error) | < 0.01 |
|---|---|---|---|---|
| prod 12 / 15 / 16 / 20 / 24 | 0.0038582 / **0.0799761** / 0.0323169 / 0.0295245 / 0.0216783 | yes | 4.6e-4 / 7.2e-4 / 7.6e-4 / 6.6e-4 / 7.8e-4 | yes |
| prod 25 / 28 / 30 / 35 / 40 | 0.0229056 / 0.0312890 / 0.0334658 / 0.0455327 / 0.0330889 | yes | 7.3e-4 / 8.9e-4 / 8.2e-4 / 8.5e-4 / 8.0e-4 | yes |
| s8 12 / 13 / 14 / 16 (v1.3, deterministic) | 0.0111397 / 0.0104565 / 0.0100835 / 0.0115984 | yes | 4.2e-4 / 5.8e-4 / 7.7e-4 / 8.3e-4 | yes |
| s8 20 / 25 / 30 / 40 (v1.3, deterministic) | 0.0229635 / 0.0101092 / 0.0045947 / 0.0099772 | yes | 6.1e-4 / 5.6e-4 / 7.1e-4 / 7.7e-4 | yes |

- **Every combined upper bound is <= 0.080.** The tightest is prod 15 at 0.0799761: margin 2.4e-5 against a storage bound of 1.6e-9.
- **Every bias bound is < 0.01** (max 8.9e-4).
- Storage: on prod, the changes are measured by fp64 replay (<= 6.6e-9 on <B>, <= 1.2e-11 on bias). On s8 they are deterministic enclosures (v1.3: d<B>(K) <= 4.9e-3, d<B>(K=0) <= 1.5e-5, d beta <= 4.1e-3; worst-case, not estimates).

## v1.3: deterministic storage enclosure on the 8 s8 slices (AstraAPI_bclose2a_v1, option (b); full record in NOTES addendum 3)
- **Option (a) is not available.** There is no fp64 or complex128 c27 source anywhere: not on the box (s8_slices are complex64; stage_s8_rec holds fields, json and handoff), not in the Drive backup (core_hf_v1 = the same complex64 slices; every MANIFEST was grepped), and not in the inventory.
- **Method.** Frozen note B_v13_method_note_v1.md (md5 814bab26; with amendment A1, 44344262).
  - Only sea_k and dust_k are complex64. Everything else entering <B> and the bias (f, the applied fields, P, chi, J, G, K, the rows, the roots) is fp64.
  - The storage model |dX| <= 2^-24 |X| componentwise gives a rigorous polar-factor (Loewdin) bound: ||dCbar||_F <= 1.08e-5 per channel.
  - This is propagated through beta, S1, S2, the terms and the bias, with operator-norm, Cauchy-Schwarz and exact vertex-trace bounds (W = sum A_mu^2 assembled exactly).
  - The rest is first order plus explicit remainders.
  - The combined upper is Hessian upper + 5 d<B>(K=0) + 4 d<B>(K).
  - A1: after the frozen coarse form failed on the T14 test (0.159), ||W||_2 ||X||_F^2 was replaced by the exact Tr(X W X). That is still rigorous, and declared.
- **Checks.** The v6 readings are bitwise equal to v1.2. The corr_blocks and x_op identities hold to <= 3.3e-14. Every R_50 keeps its 0.1a row under the enclosed dust-number change.

| s8 slice | <B> | combined upper (deterministic) | <= 0.080 | bias upper | < 0.01 |
|---|---|---|---|---|---|
| 12 | 0.0029068 | 0.0111397 | yes | 4.15e-4 | yes |
| 13 | 0.0068817 | 0.0104565 | yes | 5.75e-4 | yes |
| 14 | 0.0087292 | 0.0100835 | yes | 7.70e-4 | yes |
| 16 | 0.0076627 | 0.0115984 | yes | 8.26e-4 | yes |
| 20 | 0.0033445 | 0.0229635 | yes | 6.13e-4 | yes |
| 25 | 0.0025796 | 0.0101092 | yes | 5.63e-4 | yes |
| 30 | 0.0023017 | 0.0045947 | yes | 7.11e-4 | yes |
| 40 | 0.0060267 | 0.0099772 | yes | 7.66e-4 | yes |

**v1.3 reading: every combined upper bound is <= 0.080 and every bias upper is < 0.01, on all 18 slices.**
- The s8 slices are now enclosed deterministically: maximum 0.0230 (T20, margin 0.057), bias at most 8.3e-4.
- prod is unchanged from v1.2: fp64 replay; tightest prod 15 at 0.0799761; bias at most 8.9e-4.
- The enclosure is for storage error. fp64 evaluation error is separately measured (3.4e-14 replay agreement, 6e-13 Loewdin residual).
- Logs: logs/v13/ (v13b_enclosure.json md5 6e0f1321, per-slice v6 jsons, the failed coarse-form record v13_test.json).
- Scripts: scripts/bv2_slice_v5.py, bv2_slice_v6.py, bv2_enclose_v1.py, bv2_enclose_v2.py, bv2_v13_run_v1.sh, bv2_v13_run_v2.sh, bv2_hdr_v1.py.
