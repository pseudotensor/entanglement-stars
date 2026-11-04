# ROUTE2 core B29 v1: the second route of the dynamic core Hartree-Fock level on the stored collapse-core slices (busi, inbox #349)

Built and run by busi's analyst builder on 20 Oct 2026. This is the core stage of B29 (FABLE_REVIEW_web40j_s1, B29, fix 31, FS3), the run that FableAPI_hf_replace_v1 item (3) and AstraAPI_hf_replace_v1 section 2 name as the one missing check before the dynamic core numbers can stand beside Sec. IV C 1's fixed-field control.

## Result in one paragraph

On prod T15 and T16 and s8 T14 and T16, at R_in, R_50 and R_t, an independent second code reproduces every core number of CORE_HF_full v1.2 that it was compared on. That is 180 comparisons: 45 per slice, covering the native signed exchange (E_F and E_pair, bare and dressed, their ratio and the dressing's shift), the differentiated-vertex mass variance (Var_C, the sea part, the signed excess and sqrt(Var_C)/M_H, for the dressed and the undressed vertex), M_H, the reference response, sigma_min(I + B P^A), the kernel spectra, N_B and R_50. 178 agree within error budgets declared before the comparison. The worst relative differences are 1.1e-15 on M_H, 4e-11 on the variances, 8e-14 on the dressed E_F and 5e-13 on E_pair. The other 2 are the sea part at R_in on prod T15, for the dressed and the undressed vertex: they differ by 2.7e-10 relative against a declared budget of 5e-11. That is a defect of the budget, not of either result. The contraction part of the budget was taken from route 1's own Gram-against-direct check, which route 1 measures at R_50 and on a random vertex. At the bounce's inner root the cancellation in a Gram quadratic form is 19 times larger. A rebuild of route 1's Gram form in route 2's code puts the scatter at 1.2e-10, and the scale-aware Gram budget is 8e-10 (the diagnosis below). Every gate passes, and so does the fixture on which both routes must agree to roundoff.

## What was computed (setup)

- **Slices.** prod T15 (complex64) and T16 (complex128) from <internal>/c19_rs64_A_R2_tau8_ell4_avg_r11, and s8 T14 and T16 (complex64) from the staged copies in <internal>/. These are the same files route 1 read.
- **Records and fields.** The records and fields npz under <internal>/C3prod19 and C3lad27, read only.
- **Channels and regime.** All 10 channels with weights 16 kappa, at m a 0.2, g 0.25, ell 4 (prod) and 8 (s8).
- **Radii.** R_in and R_t are the record's dense roots. R_50 is recomputed from the dust columns and equals the well json's R_50 to 1e-9.
- **Compute.** One box GPU, cuda:0, an RTX A6000 that nvidia-smi showed free (12 MiB used, 0 percent) at the start. All arithmetic was complex128 or float64. The host side ran at nice 10 with at most 8 threads; no dense LAPACK call ran on the box CPU.
- **Wall times.** About 26 to 30 minutes per slice, plus 13 minutes for each dr 0.05 rerun.
- **Code** (busi/code/route2/):
  - route2_core_b29_v1.py (md5 5f88c7bc), the data script;
  - route2_core_b29_table_v1.py (7ab72826), the tables and the comparison;
  - route2_core_b29_diag_v1.py (3dcf7738), the diagnosis;
  - run_route2_core_b29_v1.sh (68b929a4), the run chain;
  - route2_common.py (3b79f86a), unchanged, which supplies route 2's chain, kernel and rows conventions.

What route 2 is, quantity by quantity:

- **The state.** Phi = [sea | dust] and P_ref = R2 R2^+, both Loewdin-orthonormalized. The continued occupations are the stored mask, asserted equal to e < 0 of route 2's own eigh of h_m.
- **The mixed-state variance.** Tr[C A (1 - C) A] for C = Phi diag(f) Phi^+. It is computed by the general formula (sum f ||A phi||^2 - sum f f |Phi^+ A Phi|^2) and, on these pure states (f = 1 on every stored column), by its residual form ||A Phi - Phi Phi^+ A Phi||^2, which is the headline value. Beside them are printed:
  - the sea part Tr[P A (1 - P) A];
  - the signed excess;
  - the split of Var_C into the propagated sea columns, the dust columns and the blocking term -2 ||Phi_dust^+ A Phi_sea||^2;
  - on the complex64 slices, the raw-column covariance as the storage-floor control.
- **The rows.** Route 2's RK4 at dr 0.1 on [0, R_out], with R_out = R_last + 8 ell + 1 and N(R_out) = 1. The windows b(R) = dM(R)/ds are cumulative Simpson integrals, and the Hessian B is the closed-form enclosure product. Each is a device port of route2_common, equal to it to 7e-16 (a gate).
- **The reference response.** P^A = L^* R_P L, with (R_P X)_ou = (p_o - p_u)/(e_o - e_u) X_ou. It is applied exactly in the eigenbasis as an operator, and GMRES uses only that operator. It is also formed as an explicit matrix from the SVD of the Cauchy kernel 1/(e_o - e_u), truncated at 1e-15 of its largest singular value (rank 15 to 21 per channel). A rigorous bound on the dropped part is printed: 4e-10 to 6e-10 of ||P^A||_2. The measured matrix-against-operator difference is 7e-14 to 3e-11.
- **The dressed vertex.** beta = (I + B P^A)^-1 b, by full GMRES from zero on the matrix-free exact operator. ROUTE2's ruling 4 sets the criterion: the true residual must be below 1e-10 of ||b||. Here it is 8.5e-16 to 9.6e-13, in 2 to 6 iterations. A dense solve with the explicit matrix agrees with it to 9e-14 to 4e-13 on Var_C.
- **The exchange.**
  - E_F = -(1/2) sum_k w_k <V, T_k>, with T_k = Tr(A_mu D A_nu D) and D = C - P_ref; E_pair = (1/2) s^T V s.
  - Both are computed for V = B (bare) and V = (I + B P^A)^-1 B (dressed), with the energy, mixed and current blocks.
  - The Gram bound |E_F| <= (1/2) ||V||_2 sum_k w_k Tr T_k^{|D|} is printed.
  - The dressed E_pair is computed a second time by GMRES on the exact operator; it agrees with the dense kernel to 1e-13 to 4e-13.
- **sigma_min(I + B P^A).** The computed value and a certified lower bound (the computed value minus ||B||_2 times the dropped-part bound).
- **||[h_m, P_ref]||.** Printed per channel as ||(1 - P_ref) h_m P_ref||_F/||h_m||_F. The response is refused (UNRESOLVED) above 1e-6, or when the stored mask is not e < 0. The values are 7.7e-15 to 7.8e-15 on prod T16 and 2.3e-8 on the complex64 slices, so all 40 channel-slices are resolved.

How route 2 differs from route 1, and what they share:

- **New code everywhere.**
  - The slice reader is numpy's npz loader, not a zip-offset memmap.
  - The eigensolver and every channel contraction are route 2's own: the four-term expansion of Tr(A_mu X A_nu Y) on banded operators, and the residual-form variances in place of the Gram quadratic forms.
  - The response is the SVD of the Cauchy kernel and the exact eigenbasis operator, in place of the exponential sum and S10's operator.
  - The dressing is GMRES on the exact operator, in place of LU with the exponential-sum matrix.
  - The rows are route2_common's formulas, in place of S10's radial_solve and linearized.
  - Nothing of route 1 or of the code route 1 imports (hf_star_variance_v3, qh_branch_witness_v1b, engine_ops_v1, engine_fields_v2, slices144_io_v1) is imported.
- **Shared by both.** The stored slices, the record's eps_tot and J_tot, the chain and kernel definitions (route2_common was checked against engine_ops_v1 and engine_fields_v2 in ROUTE2's K-S0), and the discrete scheme of the rows. The fixture shows the two codes realize the same discrete rows to 1.4e-15. So this comparison tests the codes, not the discretization. The discretization itself is measured separately (the dr study below) and is common to both.

## Known-answer gates (selftest_route2_core_b29_v1.json: 20 PASS, 0 FAIL)

- **Z17 (the angular regulator's statement).** The specification states it through the two-shell control (the C16 reading): V_kappa = kappa^2/(72 kappa^2 + 32).
  - The block and residual forms reproduce it to 7e-15 at kappa 1, 2, 3, 10, 100 and 1000 (0.009615384615 at kappa 1).
  - At m a 0.2 and kappa 1 the value is 0.0086357, against the reader's 0.008636.
  - The weighted channel sum 16 kappa V_kappa reaches 11.9713, 102.9773 and 1121.7485 at K = 10, 30 and 100, against the reader's 11.97, 102.98 and 1121.7. Over K^2/9 that is 1.077, 1.030 and 1.010: the sum grows as the square of the cutoff.
  - On the real slices the per-copy sea variance of the R_50 vertex is flat over the 10 stored channels, as the gate's statement says (the regulator values below).
  - Test defect, stated as such: gate version 1 also held the unreduced covariance form (sum ||A phi||^2 - ||Phi^+ A Phi||^2) to 1e-13, and it failed at kappa 1000 with 7.7e-9. That form's roundoff is eps ||A Phi||^2/Var, which grows as kappa^2. Version 2 holds it to 100 eps ||A Phi||^2/Var and passes at 0.02 of that.
- **Z16 (the counts fix neither).** Two 4-shell preparations with equal counts per shell and zero mean currents have window-count variances 0 and 1/2. The covariance formula and a Fock-space brute force give the same values to 2e-16.
- **The mixed-state formulas (FS3).** These were checked against the Fock space on 6 modes with a thermal Gaussian state.
  - Tr[C A (1 - C) A] equals the Fock variance to 5e-16.
  - The reference-ordered pair expectation equals Tr(DA)Tr(DB) - Tr(DADB) for D = C - P to 2.4e-14.
  - The ordering's own zero, <W_P>_P = 0, holds to 4e-15.
- **The contraction code (fixture of 12 sites, kappa 1 and 2, a non-eigen state with dust and a rotated reference).**
  - The banded bilinear Tr(A_mu X A_nu Y) matches the dense trace to 2e-16.
  - P^A by the SVD of the Cauchy kernel matches the eigenbasis double sum to 3.8e-15, and the exact operator matches it to 1.9e-16.
  - P^A on the generator (N, v) is 1.7e-16 of ||P^A|| ||(N, v)||.
  - P^A is negative semidefinite to 8e-18.
  - The variance's residual and covariance forms match the dense trace to 5e-16.
  - The one-particle identity E_F = -E_pair (y = -1) holds to 7e-16.
  - GMRES matches the dense solve to 4e-14.
- **Fix 31.**
  - The pair and exchange energies with and without Pi are printed on the fixture, with the sign change recorded. On that fixture's random kernel E_F changes sign and E_pair does not. That is a property of a random kernel, not of the lane's seed: this delivery contains no seed run.
  - A reassigned block returns UNRESOLVED in both of its forms. A reference with the top occupied and the lowest empty mode rotated by 0.3 rad has ||(1 - P) h_m P||/||h_m|| = 3.6e-2. A reference with the two modes exchanged is a spectral projector whose mask is not e < 0, and the mask test catches it. The spectral reference itself stays resolved (4e-15).
  - On the four core slices both energies keep their sign under the dressing: E_F is positive and E_pair negative, bare and dressed.

## The fixture where both routes must agree to roundoff

Both routes were run on the same slice, prod T16 restricted to channel kappa 1:

- route 1: its own driver core_hf_slices_v1_1.py (box copy <internal>/qi2_corehf_v1_1_code, run as is, with --kappas 1 --save-kernels --no-evolution on cuda:0, output in my directory);
- route 2: route2_core_b29_v1.py with --kappas 1.

Route 1's saved arrays were then fed into route 2's contractions:

| quantity | relative difference |
|---|---|
| P^A, explicit matrices (exponential sum against Cauchy-kernel SVD) | 1.7e-13 |
| B; V | 4.4e-14; 4.4e-14 |
| M on the 20811-point grid; the back-averaged fields; the windows b at the radii | 1.4e-15; 4.2e-16; 3.7e-16 to 5.5e-16 |
| sources s; the exchange block diagonal | 3.7e-14; 7.4e-15 |
| E_F and E_pair, bare and dressed, with route 1's kernels on route 2's T and s | 3.4e-15 to 5.9e-15 |
| Var_C of route 1's vertices on route 2's contraction (R_in, R_50, R_t; noref and full) | 4.7e-15 to 4.2e-13 |
| Var_C0, the same | 5e-15 to 1.7e-12 |
| sigma_min(I + B P^A) | 1.5e-12 |

At R_out the sea part is an analytic zero. Route 1's Gram value is -1.5e-12 and route 2's residual form gives 9.0e-14.

## The numbers (route 2; the stored channel set kappa <= 10; every slice past the record's own hydrodynamic limit)

The exchange of the whole slice (dressed: the continued reference's response; bare: B):

| slice | dtype | E_F dressed | E_pair dressed | E_F/abs(E_pair) dressed | E_F bare | E_F/abs(E_pair) bare | dressed/bare E_F - 1 | sigma_min (certified lower) |
|---|---|---|---|---|---|---|---|---|
| prod 15 | c64 | 3.83826e-02 | -27.2067 | 1.4108e-03 | 3.94934e-02 | 1.4274e-03 | -2.81% | 0.689324 (0.689324) |
| prod 16 | c128 | 4.12431e-02 | -27.0771 | 1.5232e-03 | 4.20408e-02 | 1.5342e-03 | -1.90% | 0.689589 (0.689589) |
| s8 14 | c64 | 3.97040e-02 | -27.4976 | 1.4439e-03 | 4.01006e-02 | 1.4475e-03 | -0.99% | 0.694921 (0.694920) |
| s8 16 | c64 | 4.40647e-02 | -28.5066 | 1.5458e-03 | 4.45039e-02 | 1.5490e-03 | -0.99% | 0.696563 (0.696563) |

The blocks are new for three of the four slices (route 1 has the bare blocks of prod T16 only, its C1, and route 2 reproduces them to 1.7e-14 to 2.2e-13). The dressed blocks are new on all four (route 1's v1.2 says the dressed split is not computed).

| slice | block | E_F dressed | E_F bare | E_pair dressed | E_pair bare |
|---|---|---|---|---|---|
| prod 15 | energy | -7.0595e-02 | -6.9273e-02 | +41.4641 | +40.8773 |
| prod 15 | mixed | +2.6951e-01 | +2.6791e-01 | -164.4354 | -163.5090 |
| prod 15 | current | -1.6053e-01 | -1.5914e-01 | +95.7647 | +94.9636 |
| prod 15 | abs(total)/sum abs(blocks) | 0.077 | 0.080 | 0.090 | 0.092 |
| prod 16 | energy | -5.5311e-02 | -5.4368e-02 | +32.9943 | +32.5740 |
| prod 16 | mixed | +2.1417e-01 | +2.1321e-01 | -130.8637 | -130.2961 |
| prod 16 | current | -1.1761e-01 | -1.1680e-01 | +70.7923 | +70.3196 |
| prod 16 | abs(total)/sum abs(blocks) | 0.107 | 0.109 | 0.115 | 0.118 |
| s8 14 | energy | -5.2437e-02 | -5.1395e-02 | +31.7029 | +31.0846 |
| s8 14 | mixed | +2.0502e-01 | +2.0370e-01 | -125.1495 | -124.3385 |
| s8 14 | current | -1.1288e-01 | -1.1220e-01 | +65.9490 | +65.5508 |
| s8 14 | abs(total)/sum abs(blocks) | 0.107 | 0.109 | 0.123 | 0.125 |
| s8 16 | energy | -5.6647e-02 | -5.5956e-02 | +33.6229 | +33.2575 |
| s8 16 | mixed | +2.1898e-01 | +2.1819e-01 | -133.4924 | -133.0299 |
| s8 16 | current | -1.1827e-01 | -1.1773e-01 | +71.3629 | +71.0419 |
| s8 16 | abs(total)/sum abs(blocks) | 0.112 | 0.114 | 0.120 | 0.121 |

On the blocks:

- **Cancellation.** The blocks cancel by a factor of 9 to 13: the mixed block is the largest and has the opposite sign to the other two.
- **The dressing's effect.** The dressing increases each block's magnitude, by 0.4 to 2.0 percent, on all four slices.
- **The Gram bound.** It is 254 to 420 times |E_F|. The bound is valid but uninformative, because the native kernel is indefinite (eig B from -1.065 to 1.686, eig V from -1.059 to 1.698).

The differentiated-vertex variance (full: the dressed vertex; M_H at the 0.1a grid row; the reference response is Var_C(full)/Var_C(noref) - 1):

| slice | radius | r/a (grid row) | M_H | Var_C | sea part Var_C0 | signed excess | sqrt(Var_C)/M_H | N_<^-1/2 | ratio | reference response | propagated sea / dust / blocking |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prod 15 | R_in | 7.048 (7.0) | 13.887 | 3.5757 | 0.1475 | +3.4282 | 0.1362 | 0.1726 | 0.79 | -1.0% | 0.2004 / 3.3772 / -0.0020 |
| prod 15 | R_50 | 28.854 (28.9) | 80.775 | 7.8785 | 1.2718 | +6.6067 | 0.0347 | 0.0559 | 0.62 | +5.8% | 1.3635 / 6.6265 / -0.1115 |
| prod 15 | R_t | 51.760 (51.8) | 103.525 | 2.0009 | 0.9232 | +1.0778 | 0.0137 | 0.0404 | 0.34 | +0.8% | 0.9481 / 1.0572 / -0.0043 |
| prod 16 | R_in | 23.444 (23.4) | 46.766 | 14.1455 | 1.0297 | +13.1158 | 0.0804 | 0.0956 | 0.84 | +4.1% | 1.1352 / 13.0802 / -0.0699 |
| prod 16 | R_50 | 31.350 (31.4) | 71.015 | 6.7154 | 0.9495 | +5.7659 | 0.0365 | 0.0559 | 0.65 | +3.9% | 0.9932 / 5.8393 / -0.1171 |
| prod 16 | R_t | 50.837 (50.8) | 101.644 | 2.2049 | 0.8722 | +1.3327 | 0.0146 | 0.0406 | 0.36 | +1.3% | 0.8788 / 1.3499 / -0.0238 |
| s8 14 | R_in | 10.711 (10.7) | 21.372 | 0.6605 | 0.0545 | +0.6059 | 0.0380 | 0.1046 | 0.36 | -0.3% | 0.0545 / 0.6089 / -0.0030 |
| s8 14 | R_50 | 19.140 (19.1) | 61.527 | 2.3528 | 0.2391 | +2.1137 | 0.0249 | 0.0559 | 0.45 | +0.5% | 0.2358 / 2.1417 / -0.0247 |
| s8 14 | R_t | 52.908 (52.9) | 105.815 | 1.2659 | 0.4303 | +0.8357 | 0.0106 | 0.0412 | 0.26 | +0.7% | 0.4343 / 0.8325 / -0.0009 |
| s8 16 | R_in | 31.333 (31.3) | 62.575 | 4.5554 | 0.3651 | +4.1903 | 0.0341 | 0.0656 | 0.52 | +2.0% | 0.3784 / 4.2007 / -0.0237 |
| s8 16 | R_50 | 34.841 (34.8) | 72.098 | 3.4527 | 0.3675 | +3.0851 | 0.0258 | 0.0559 | 0.46 | +2.2% | 0.3783 / 3.1027 / -0.0284 |
| s8 16 | R_t | 52.913 (52.9) | 105.815 | 0.9425 | 0.3447 | +0.5978 | 0.0092 | 0.0405 | 0.23 | +1.6% | 0.3390 / 0.6122 / -0.0087 |

Reading the split:

- The propagated-sea column (the stored sea columns, pair content included) is not the reference's sea part Var_C0. It stays close to Var_C0 everywhere, within 0.04 to 36 percent.
- The dust columns carry the excess.
- The blocking term is small and negative: 0.06 to 1.7 percent of Var_C.

## Error budgets (declared before the comparison; all relative)

Every compared number carries an agreement budget built only from stored quantities, as the sum of these parts:

- **rows.** 1e-12 for every rows-dependent number. Declared from the fixture: the two codes realize the same discrete rows to at most 4.4e-14, so the declared budget is 100 times the largest fixture difference.
- **dressing.** Route 2's two solves (GMRES on the exact operator against the dense solve with the explicit matrix), at 9e-14 to 4e-13 on Var_C. Plus 2 rho/sigma_min, where rho is the larger of the two routes' residuals: route 2's GMRES at most 9.6e-13, route 1's LU at most 4.4e-15. Together about 1e-12 to 3e-12 on Var_C. For the dressed exchange, this part is the explicit response's relative error (5.1e-10 at most, the dropped-SVD bound) times the dressing's share of the number.
- **contraction.** For the variances, route 1's Gram-against-direct check Q6' on that slice: 4.8e-11 to 4.2e-10 on these four slices. For the exchange, 1e-13 times the block cancellation factor (9 to 13).
- **quadrature.** Route 1's exponential sum: its matrix-against-operator test, 4.5e-15 to 1.3e-14.

Not in the budget, and printed beside it:

- **The shared discretization.** It was measured by route 2's dr 0.05 rerun of each slice at the same evaluation rows, with the explicit response reused: 2|X(0.1) - X(0.05)|/|X| is at most 1.2e-8 on Var_C, 3.4e-8 on the sea part, 1.6e-9 on M_H and 3.8e-10 on E_F. Both routes carry it, because they use the same scheme.
- **The radius offset.** The grid row against the nominal radius is a separate and larger systematic: up to 1.5 percent on M_H at small R_in, per v1.2 C5. The M_interp column of the json carries it.
- **The complex64 storage floor.** It is common to both routes, which read the same stored columns and Loewdin-orthonormalize them. Route 1's C2 control puts its effect at 2.4e-8 on the variances. Route 2's own raw-column covariance control gives footprints of 3.2e-6 to 7.1e-5, which reproduces route 1's proxy column. That control is a proxy, not the effect.

## The comparison (180 numbers; route 1 at full precision from its per-slice card jsons, the source of v1.2's tables)

| quantity (at R_in, R_50, R_t where it applies) | numbers | worst relative difference | smallest budget | verdict |
|---|---|---|---|---|
| M_H (the same grid rows on both routes) | 12 | 1.1e-15 | 1.0e-12 | 12 agree |
| Var_C, noref and full | 24 | 3.7e-11 | 4.9e-11 | 24 agree |
| sea part Var_C0, noref and full | 24 | 2.7e-10 | 4.9e-11 | 22 agree, 2 DISAGREE (prod T15 R_in) |
| signed excess, noref and full | 24 | 3.8e-11 | 5.4e-11 | 24 agree |
| sqrt(Var_C)/M_H, noref and full | 24 | 1.8e-11 | 2.5e-11 | 24 agree |
| reference response | 12 | 1.1e-10 | 1.8e-9 | 12 agree |
| E_F, dressed and bare | 8 | 1.3e-13 | 1.9e-12 | 8 agree |
| E_pair, dressed and bare | 8 | 5.0e-13 | 1.8e-12 | 8 agree |
| E_F/abs(E_pair), dressed and bare | 8 | 3.8e-13 | 3.7e-12 | 8 agree |
| dressed/bare E_F - 1 | 4 | 2.8e-12 | 6.9e-10 | 4 agree |
| sigma_min(I + B P^A); ||V - B||_F/||B||_F | 8 | 1.9e-12 | 4.0e-10 | 8 agree |
| eig B and eig V, min and max | 16 | 1.5e-14 | 1.0e-12 | 16 agree |
| N_B; R_50 | 8 | 1.5e-12 | 1.0e-12 | 8 agree |

Route 1's per-slice arrays were also fed into route 2's code on each of the four full slices:

- M on the grid agrees to 6.2e-16 to 1.4e-15;
- the windows b to 2.6e-16 to 6.1e-16;
- the dressed vertices to 2.4e-14 to 9.6e-13;
- the sources to 3.8e-13 to 6.0e-13;
- eig B to 2.5e-15 to 1.9e-14.

The 6 per-slice tables, with each budget's parts, are in ROUTE2_core_B29_tables_v1.md beside this file.

**Diagnosis of the 2 disagreements** (route2_core_b29_diag_v1.py; the declared verdicts stand as printed). For the R_in vertices of prod T15, route 2's code was given route 1's stored vertices, which equal route 2's to 3e-13. It evaluated the sea part three ways: its own residual form, the unreduced difference, and route 1's Gram quadratic form rebuilt.

- **The cancellation scale.** s = ||A R2||^2/Var_P reaches 3.7e5 per channel at R_in, against 2.0e4 at R_50.
- **The scatter.** The unreduced and Gram forms scatter around the residual form by 6.7e-11 to 1.2e-10. Route 1's json value lies 2.7e-10 below the residual form and 1.5e-10 below the rebuilt Gram value.
- **The comparison scale.** The Gram form's roundoff is about eps s per channel, so 10 eps max(s) = 8e-10. Both differences sit inside it.
- **Which form is more accurate.** The residual form's own roundoff is about 2 eps sqrt(s), or 3e-13. Route 2's value is therefore the more accurate one, and route 1's is right to its own roundoff.
- **R_50 on the same slice.** The same three forms agree to 1.5e-11 to 3.6e-11, inside 10 eps max(s) = 4.4e-11. That is consistent with route 1's Q6' check, which is measured there.
- **The other three slices.** All of their Var_C0 numbers agree within the declared budget.

## The controls (route 2's, every slice)

- **The rows.**
  - Field check against N_applied and v_applied: dN 1.1e-7 to 8.4e-7 and dv 1.1e-7 to 7.6e-7 (label CORE). These equal route 1's field check (8.36e-7 and 7.60e-7 on T16).
  - E(R_out) against the record's E: 3e-8 to 1.1e-7 relative.
  - The first law b(R_out) against the gradient: 4.5e-16 to 6.4e-16.
- **The reference and its response.**
  - The continued reference against route 2's own eigenvectors: 5.9e-15 (complex128) and 2.3e-8 to 2.4e-8 (complex64).
  - The stored e_k against route 2's eigh: at most 2.2e-15. The mask equals e < 0 bitwise on all 40 channel-slices.
  - ||P^A (N, v)||/||(N, v)|| with the exact operator: 6.3e-15 to 6.5e-15.
- **The sea part at R_out.** 5.3e-12 (complex128) and 1.06e-9 (complex64) of a Var_C of 1.32 to 1.37: the analytic zero to the storage floor.
- **The pair content.** 775 to 898 eigenvalues of D above 1e-6 per channel. D = C - P_ref is far from a dust-only projector.
- **The angular regulator.** These are per-copy sea variances of the R_50 full vertex for kappa = 1 to 10; every channel-summed variance above is at the production's regulator kappa <= 10.
  - prod T15: 1.35e-3 to 1.47e-3.
  - prod T16: 1.02e-3 to 1.09e-3.
  - s8 T14: 2.92e-4 falling to 2.44e-4.
  - s8 T16: 3.88e-4 to 4.25e-4.
  - The flat per-copy value is Z17's statement: without the regulator the channel sum grows as K^2. The extension beyond kappa = 10 was not run here.
- **The discretization (the dr 0.05 reruns at the same rows).** The variances move by at most 1.7e-8, M_H by 8.2e-10 and E_F by 1.9e-10.

## What this supports in the paper, and what it does not

It supports:

- **Two routes on these four slices.** The core's dressed native signed exchange and the differentiated-vertex variance on prod T15 and T16 and s8 T14 and T16, at R_in, R_50 and R_t, now have two independent codes agreeing far below every printed digit. This is the condition FableAPI_hf_replace_v1 item (3) names. On these four slices:
  - E_F/|E_pair| is 0.141 to 0.155 percent dressed (0.143 to 0.155 bare);
  - the dressing lowers E_F by 1.0 to 2.8 percent;
  - sqrt(Var_C)/M_H is 0.79 to 0.84 of N_<^{-1/2} at R_in, 0.62 to 0.65 at R_50 and 0.34 to 0.36 at R_t on prod;
  - the same ratios on s8 are 0.36 to 0.52, 0.45 to 0.46 and 0.23 to 0.26.
- **The exchange's block structure.** The energy, mixed and current blocks, bare and dressed, are now measured on four slices, the dressed ones for the first time.

It does not support:

- **The other 14 slices.** reviewer A R12's conditional paste prints ranges over all 18 slices, for example "0.14 to 0.16 percent" and "lowered by the dressing by 0.6 to 4.4 percent". Those ranges stay one-route outside the four slices run here: prod T12, T20 to T40 and s8 T12, T13, T20 to T40. A sentence that cites two routes must name the four clocks or wait for the rest. Each further slice costs about 30 minutes on a free card with this code.
- **The control's replacement.** reviewer A and reviewer B list conditions this run does not touch:
  - every slice is past the hydrodynamic gate (sigma |dv| 0.12 to 0.26 against 0.1), with no certified history prefix and no completion spread;
  - no time-step spread (the tau 4 and 2 slices are unread);
  - the scale spread is two members only;
  - the correlation beyond Hartree-Fock is uncontrolled, since B29's native cluster (L 12, kappa 1, 134596 states, Lanczos) is not part of this delivery;
  - Var_C is the linearized dressed observable's variance, not the second-order mass operator's;
  - E_F is the whole slice's exchange, not the inner core's;
  - the kernels are indefinite, so 1/N_B and 5/(4 N_B) remain comparators and bound nothing.
- **Testing the discretization.** The second route shares the rows' discrete scheme, so it confirms the code, not the discretization. Route 2's dr study gives at most 3.4e-8 on any compared number, far below every printed digit; that is the shared accuracy, not a second-route statement.
- **Sec. IV C 1 and the discussion.** Sec. IV C 1's fixed-field control stays as printed, and so do the discussion's and the letter's citations of it. The HF sentence at 05_star.tex:366 can name these four slices as two-route if the paper lane chooses. Its Z16 mark ("against the step and scale spreads, with the small-cluster exact control; both evolution rows in the core") is not discharged by this run. This delivery edits no paper file.

## Files

**Beside this file** (ROUTE2_core_B29_v1_json/, every file under 1 MB, MD5SUMS inside):
- The data jsons, one per slice and one per dr 0.05 rerun: prod_T0015.00, prod_T0016.00, s8_T0014.00 and s8_T0016.00, each with a _dr05 twin.
- The fixture: prod_T0016.00_k1x.json (route 2, with route 1's arrays on route 2's code) and fixture_route1_core_hf_prod_T0016.00_k1.json (route 1's driver on the same fixture).
- selftest_route2_core_b29_v1.json (the gates).
- diag_prod_T0015.00.json (the diagnosis).
- ROUTE2_core_B29_compare_v1.json (every compared number with its budget parts and verdict).
- ROUTE2_core_B29_tables_v1.md (the per-slice tables, budgets first).

**Bulk on the box** (<internal>/route2_core, 3.7 GB, listed in busi/LARGE_FILES.tsv):
- the arrays npz per slice, with B, P^A, V and every vertex: 540 MB each, 70 MB for the dr 0.05 reruns;
- the fixture arrays (fixture/, fixture_route1/);
- the logs (logs/), including chain.log with every run's start and end;
- the code copies (code/), with md5s equal to the repository's.
