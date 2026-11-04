# VREF route 1: vertex reference contractions on the reference collapse (v1)

Webgpt lane, route 1: differentiate the discrete admitted solve (spec sec. 8). The spec is deliveries/lead_gaps/AstraAPI_vref_v1.md and the paper deliveries/lead_gaps/PAPER_flat_3f3f40a2fc3.tex. Route 2 was kept blind to this route and this route to route 2. The slices are T/r_s 11.0 (matched rerun W5) and T/r_s 15.0 (the reference collapse on the doubled domain B2n4096). Every number in the tables was copied from this route's result files by the assembly script s38; none was retyped. Table 1 gives every test's status. vertex_control_R1_v1.json carries the same statuses with their values, errors, rules and the sha256 of every result file.

## 1. Summary

- **Gates pass at both slices** (Table 2). With g_k = 16k genuine copies, Tr(C-bar - P) recovers 640, and the replayed state recovers the stored sources and the record's charge E. The reference is the ground projector, equal to the record's continued reference. The state is Slater up to its measured impurity. The spec's small-Fock-space unit tests all pass (Table 12).
- **Reference correlation FAILS at both slices** (Table 4). |F|/E is below 1%, but the dressing Z_2 exceeds 1% by more than an order of magnitude at both: the reference state is dressed at order one. The interior shares carry the failure for every vertex choice. At 15.0 the denominator gate also fails on the path, because crossings are continued by the record's LZ convention.
- **Signed exchange** (Table 5): it weakens the pair energy by an amount inside the paper's printed 0.14 to 0.16 percent at both slices, and inside 0.90 to 1.01 times 1/N_B.
- **Sources and fields** (Tables 6 and 7). The source derivative of F_1 is exact, confirmed by the scaling identity and by centered differences on the same branch. Its induced field changes pass the 1% screen at 11.0 and FAIL at 15.0, where two represented masses and the inner root move by more than 1%. The double-pair source profile is UNRESOLVED.
- **Masses** (Table 8). Every mass gradient satisfies the paper's window identity to rounding. The complete mass Hessians agree with centered differences along the self-consistent admitted map and are clearly told apart from K0. That passes at 11.0. At 15.0 the inner root's residual sits just above the preregistered 1e-4, so that test is UNRESOLVED (sec. 4.5).
- **Mass bias, reordered norms and branch weight PASS at both slices** (Table 9). The bias is far below 1% and below the measured rms fluctuation at every radius. <B> reproduces the paper's 0.080 with the complete Hessians: far below it at 11.0, and just below it at 15.0. There the three masses sum to a bound whose upper end lies within a fraction of a percent of 0.080, with the inner root carrying most of it. Those statements hold for these two slices only.
- **E_2 reordered about C-bar** (Table 10). The residual single-pair matrix -X_V(D) is far below 1% of E at both slices. The full single-pair matrix h - X_V(D) exceeds 1% at 15.0, so the test FAILS there by the 1% screening rule fixed before the 15.0 numbers. That norm is dominated by the Hartree rate of the held h (C-bar is not stationary under h; see VDRESS_RESIDUAL_R1_v1.md), not by a correction.
- **Hole accounting** (Table 2): Tr D^2 = 640 + 2H - Tr[C-bar(1 - C-bar)] holds to rounding at both slices.
- **Leakage: LEAKAGE_UNRESOLVED** at both slices (Table 11). Only the cheap certificate was evaluated, and it is loose by many orders of magnitude.
- **Premises** (sec. 5). The dressed reference differs from P by an order-one Frobenius norm at 15.0. The spec's two definitions of V differ in the collar, where the stored rows are not variational.

**Table 0. Headline numbers**

| quantity | T/r_s 11.0 (W5) | T/r_s 15.0 (B2n4096) |
|---|---|---|
| \|F\|/E (bracket) | [0.0008058, 0.0008058] | [0.004206, 0.004206] |
| Z_2 (bracket) | [0.1362, 0.1362] | [1.095, 1.095] |
| interior share (R <= r_c - 10 ell): \|F_1\|/E, Z_1 | [0.0004737, 0.08275] | [0.004178, 1.091] |
| largest induced relative field, mass or root shift (F_1 sources) | 0.003309 | 0.01774 |
| largest \|beta_j\|/M_j | 0.0001034 | 0.0003725 |
| <B> upper end b_+ | 0.004138 | 0.0798 |
| signed Fock exchange / Hartree pair energy | -0.001471 | -0.001411 |
| holes H | 0.007354 | 0.5099 |
| 64 a L_global (leakage certificate) | 1.745e+11 | 4.733e+11 |

**Table 1. Status per test**

| test | T/r_s 11.0 (W5) | T/r_s 15.0 (B2n4096) |
|---|---|---|
| gate_normalization | PASS | PASS |
| gate_reference | PASS | PASS |
| gate_state_slater | PASS | PASS |
| dE_identity_interior | PASS | PASS |
| dE_identity_collar | FAIL | FAIL |
| vertex_V | PASS | PASS |
| mass_window_identity | PASS | PASS |
| mass_hessian_fd | PASS | UNRESOLVED |
| denominator_gate | PASS | FAIL |
| reference_correlation | FAIL | FAIL |
| signed_exchange | PASS | PASS |
| source_derivative_exact | PASS | PASS |
| induced_fields_single_pair | PASS | FAIL |
| induced_fields_double_pair | UNRESOLVED | UNRESOLVED |
| induced_scaling_dE | PASS | PASS |
| hole_accounting | PASS | PASS |
| mass_bias | PASS | PASS |
| branch_certified_lt_1 | PASS | PASS |
| branch_target_0080 | PASS | PASS |
| E2_reordered_norms | PASS | FAIL |
| leakage | UNRESOLVED (LEAKAGE_UNRESOLVED) | UNRESOLVED (LEAKAGE_UNRESOLVED) |
| unit_tests_fock_space (both) | PASS | PASS |

## 2. Scope, labels and inputs

Two slices of the reference collapse, as the spec names them:

- **T/r_s 11.0**: the matched rerun W5 (record 88). No stored state exists at 11.0. The T0011.00 slice is listed in md5_done.txt but is absent from the box and from the HF dataset (QUESTIONS.md Q1). So the state was **replayed** back from the stored T/r_s 11.5 slice over records 91 to 88 with the recorded quadratic propagators on the recorded accepted fields.
- **T/r_s 15.0**: the **reference collapse on the doubled domain B2n4096** (record 120), replayed back from the HF T/r_s 16.0 slice over records 127 to 120 in the same way. B2n4096 has 4096 sites per chain (d = 8192 per channel), not the spec's 2048. That is the doubled domain the brief prescribes past W5's end.

All inputs are read-only. Their md5s are in Table 13, and the HF slice's sha256 equals its LFS pointer. Nothing else under <internal>/claude_alt_solve_bulk or <internal>/busi_bulk was opened. All of this route's arrays are on the box under <internal>/webgpt_vref_r1 (box record). The heavy 15.0 stages and an independent 11.0 recompute ran on the lane's H100 pod (lead #280, Q3 = a), and their small outputs are copied to the box under from_h100/.

## 3. Method (route 1: differentiate the discrete admitted solve)

**State and normalization.** Each channel k = 1..10 carries g_k = 16k genuine identical copies. There is no further valley factor. The stored sources are already copy-aggregated: they are recovered as sum_k g_k [moments(C-bar_k) - moments(P_k)] and never multiplied again. The replay is a Chebyshev expansion of exp(-i tau h_k) on each recorded field pair, with tails below 1e-41. Its step convention is fixed by replaying forward from 11.5 to 12.0 against the stored 12.0 slice, and by recovering the stored shell totals and charge E at the target record.

**Reference.** At both slices P_k is the ground projector of the held h_k (the negative set). It equals the record's continued reference, by the record's own continuation flags. All per-channel reference-side work uses the dense eigendecomposition of h_k (CPU on the box; torch.linalg.eigh on the H100, A/B against the box eigenvalues).

**Discrete admitted solve.** This route implements the history's discrete rows. That means the sigma = 4a Gaussian source rows, the centre series, a fixed-step 12-stage Runge-Kutta march on the radial grid (h = ell/16) to the collar at the chain end, and the back-averaging to the applied site fields u = (N-bar, v-bar). This implementation matches the engine's own solve on the same sources. Exact forward tangents give the Jacobian S' = du/ds. Reverse adjoints give dE/ds and dM_j/ds at fixed P, and forward-over-reverse gives the exact Hessians of any scalar of the solve. Each is checked against central differences.

**Vertex.** The reference response is chi = dm_P/du, the static polarization of the reference moments with copies. It is computed by a 32-node exponential-sum (Laplace) quadrature of 1/x and 1/x^2 over the measured gapped spectrum, and cross-checked by Zolotarev rational poles. The coupled-row linearization gives V = (I + S' chi)^{-1} S' = du/dz, with all rho-rho, rho-J and J-J blocks kept, including the mixed element of order sqrt(G). The contractions use its symmetrization. The Hessian-of-E vertex V_E = (I + H_E chi)^{-1} H_E is carried as a cross-check. The two definitions of the spec's sec. 2 coincide wherever delta E = Tr(h delta C) holds. They differ only in the collar, where the stored rows are not variational (sec. 4.2).

**Reference energy.** B = -H_V(D) gives F_1 = -sum |B_{alpha i}|^2/d_{alpha i} = (1/2) c^T chi c with c = V d. It is evaluated both in the eigenbasis and in closed form. The double pair F_2 and the dressing Z_2 use the Laplace representation. The ring (direct) term Tr(V Pi(t) V Pi(t)^T) is exact. The same-copy exchange is bracketed by its per-copy Cauchy-Schwarz bound: per copy the exchange is the inner product of the K-contracted pair tensor with its occupied-index swap, so its magnitude is at most sum_k g_k Tr(K Pi_k K Pi_k^T). The spec's small-Fock-space unit test confirms the bracket (Table 12, T5).

**Source derivatives.** The exact gradient q_1 = dF_1/du at fixed C-bar, kernel and continuation includes dV, dP, dD, eigenvector response and denominators. It is assembled from one Hessian-vector product of the discrete solve, one second-order reference contraction by Zolotarev poles, and chi V chi. It is checked along the scaling direction exactly and by centered differences on the same branch along two smooth directions. Inserting q_1 as extra sources into the linearized rows (P's response included) gives delta u = V q_1, delta M_j = G_j . q_1 and the simple-root shifts delta R = -(delta N + delta v)/d_r(N + v) on the raw grid fields. The double-pair source profile was not computed (Q4, default b). Only its scaling-direction derivative is.

**Masses.** M_j = U(r_j)/(2g) of the discrete solve at fixed represented radii. At 11.0 that is R_50 only, since no trapping root exists. At 15.0 it is R_50 and the record's inner and outer roots, at their nearest 0.1a rows. R_50 comes from the replayed dust columns with copies. The radii are frozen when differentiating. The gradient at fixed P is checked against the paper's window identity, evaluated by independent quadrature on the engine's dense fields. Then G_j = (I - chi V)^T M'_j, and K_j = K0_j - V Xi_j V. Here K0_j = S^T Hess(M_j - lambda_j . u) S with S = I - chi V, and Xi_j is the second-order response of Tr(Omega_j P) (Zolotarev poles). The complete K_j is checked by centered differences of G_j(z) along the self-consistent admitted map u(z) = U(z - m_P(u)).

**Reordering about C-bar and branch.** beta_j = -(1/2) sum K Tr(A D A D) and L_j = G_j - X_K(D). S_1 = sum_k g_k ||(1 - C-bar_k) L_j C-bar_k||^2 is exact. S_2 is bracketed as above, with the occupied/empty split of C-bar (Pi^C = Cov_C-bar), and <B> = 4 sum_j (beta_j^2 + S_1j + S_2j)/M_j^2. The same reordering applied to E_2 gives its two labelled single-pair norms. The pieces that need C-bar itself (D-kernel, Cov_C, S_1) ran on the box. At 15.0 the complete K_j and V came from the H100 as exactly symmetric upper triangles; every full matrix was rebuilt and verified by its sha256.

**Leakage.** Only the spec's cheap global certificate L_global is computed. Trace norms carry the full copy multiplicity: ||t_k||_1 from the eigenvalues of each dense t_k, and ||A_mu A_nu||_1 bounded per copy and zero beyond four sites.

**Hole accounting.** H = Tr[P(1 - C-bar)] and N_p are computed with copies, and the identity Tr D^2 = 640 + 2H - Tr[C-bar(1 - C-bar)] is checked directly. No norm replaces D by the dust orbitals.

## 4. Results per test

The values in every table were copied from this route's result files by the assembly script s38. None was retyped.

### 4.1 Gates (reported first) and unit tests

All three gates pass at both slices (Table 2).

- **Normalization.** With g_k = 16k, Tr(C-bar - P) recovers 640. With the same weights, the replayed state recovers the stored shell totals at the target record. This route's solve on those totals reproduces the record's charge E.
- **Reference.** P is idempotent, commutes with h, and has fixed rank (half filling in every channel). It equals the negative set, which the record's continuation flags confirm is the continued reference at both records.
- **State.** The replayed panels are orthonormal. The covariance is a Slater projector up to the impurity in Table 2, which is reported rather than purified. The exclusive single/double-pair formulas are used as they stand.

The hole identity Tr D^2 = 640 + 2H - Tr[C-bar(1 - C-bar)] holds to rounding at both slices. The hole count H is small at 11.0 and about one half at 15.0. The spec's mandatory small-Fock-space unit tests all pass (Table 12). They cover: the quartic expectation; its vanishing value and first derivative at C = P; the one-particle Hartree-exchange cancellation; the Fock-space action against the single/double-pair decomposition; the Gram-route S_2 against brute force with the exchange bracket; and the reordering identity, second moment and hole identity.

**Table 2. Gates**

| quantity | 11.0 | 15.0 |
|---|---|---|
| Tr(C-bar - P), copies g_k = 16k | 639.999999975 | 639.999999909 |
| stored sources: eps, rel. max | 1.886e-11 | 2.989e-12 |
| stored sources: J, rel. max | 9.828e-13 | 4.604e-14 |
| charge E: this solve minus record | -2.029e-08 | 2.162e-08 |
| max \|\|P^2 - P\|\| | 1.168e-14 | 1.811e-14 |
| max \|\|[h, P]\|\| | 3.112e-15 | 2.006e-15 |
| mask = negative set, all channels | True | True |
| min gap | 0.2713 | 0.01899 |
| impurity Tr[C(1 - C)] | 2.534e-08 | 8.543e-08 |
| max panel orthogonality error | 5.829e-13 | 1.112e-12 |
| holes H = Tr[P(1 - C-bar)] | 0.00735428 | 0.509897 |
| N_p = Tr[(1 - P)C-bar] = Tr D + H | 640.0073543 | 640.5098964 |
| Tr D^2 | 640.0147085 | 641.0197929 |
| Tr D^2 - (Tr D + 2H - Tr[C-bar(1 - C-bar)]) | 5.457e-11 | -4.366e-11 |

**Table 12. Small Fock-space unit tests (spec sec. 3; shared by both slices)**

| test | value |
|---|---|
| V_hermitian | 5.329e-15 |
| Wick_closed_form_vs_generic_P | 1.735e-18 |
| T1_slater_cov_check | 6.661e-16 |
| T1_slater | 5.329e-15 |
| T1_thermal | 8.882e-16 |
| T2_value_at_P | 3.574e-16 |
| T2_first_derivative | 1.636e-07 |
| T2_second_derivative_nonzero | 9.392 |
| T3_one_particle | 4.941e-15 |
| T3_fK_one_particle | 7.118e-16 |
| T4_V_on_ref_norm2_vs_S2 | 1.155e-14 |
| T4_F_resolvent | -1.497 |
| T4_F_formula | -1.497 |
| T4_F_diff | 2.887e-15 |
| T4_first_order_ref_expectation | 1.03e-15 |
| T5_S2_gram_vs_bruteforce | 6.217e-15 |
| T5_exchange | 23.14 |
| T5_bracket_sum_gk_D2k | 23.93 |
| T5_bracket_holds | True |
| T6_reordering_operator_identity | 1.779e-14 |
| T6_second_moment | 89 |
| T6_beta2_S1_S2 | 89 |
| T6_diff | 8.527e-14 |
| T6_mean_minus_M_vs_beta | 7.105e-15 |
| T6_hole_identity | 8.882e-16 |

### 4.2 The rows identity, the vertex and the collar

This route's discrete solve reproduces the engine's on the same sources. Its Jacobian and the Hessian-of-E actions agree with centered differences (Table 3).

The independent check delta E = Tr(h delta C) of spec sec. 2 holds to rounding on R <= r_c - 10 ell. It **fails in the collar**, the last 10 ell of the chain, which is the history's collar node, by up to one half of the field at the last site. This is a property of the stored rows there, not of this implementation, because the engine's own solve agrees with this route's on the same sources. Its consequence is that the spec's two definitions of V differ in the collar. The coupled-row Jacobian du/dz, its symmetrization and the Hessian of E agree to rounding in the interior and disagree only at the collar sites (Table 3: the asymmetry row and the Hessian-vs-Jacobian row). Sections 4.3 to 4.5 carry this as the vertex-choice applicability error, with interior-only shares beside every total that depends on it. cond(R_u) is close to one at both slices, so the response solve is well conditioned. V is indefinite and its numerical rank is well below d. Every rho-rho, rho-J and J-J block is kept.

**Table 3. Field solve and vertex V**

| quantity | 11.0 | 15.0 |
|---|---|---|
| dE/ds = u-bar, max rel, R <= r_c - 10 ell | 5.64e-14 | 2.316e-13 |
| dE/ds = u-bar, max rel, all sites | 0.5008 | 0.5004 |
| Jacobian vs central differences (h = 1e-5), max rel | 1.596e-09 | 2.316e-09 |
| Hessian of E action vs differences, rel | 1.639e-09 | 1.434e-09 |
| Hessian of E vs Jacobian, all / interior | [0.006569, 8.227e-14] | [0.05088, 2.072e-13] |
| cond(R_u) | 1.368 | 1.374 |
| V_raw asymmetry, rel | 0.006415 | 0.05072 |
| max \|V\| blocks rho-rho / rho-J / J-J | [0.03182, 0.05611, 0.008004] | [0.001459, 0.004283, 0.005222] |
| numerical rank of V (1e-12) | 1470 | 2892 |
| eigenvalue range of V | [-1.002, 1.646] | [-1.578, 2.238] |

### 4.3 Connected reference energy and dressing

**Denominators.** At both slices every single and double denominator is positive and F <= 0. At 11.0 the path is gapped: record 88's continuation is verified in every channel, the mask is the negative set and there are no pockets. The denominator gate passes there. At 15.0 the slice's P is again the ground projector, but the path from preparation is not gapped. Crossings are continued by the record's LZ convention from T/r_s 11.125, and at record 112 the mask is not the negative set (pockets in channels 1 to 3). The 15.0 numbers are therefore a ground-state contraction of the slice that does not satisfy the paper's gapped-path premise, and the gate is recorded as FAIL at 15.0.

**Reference correlation: FAIL at both slices.** |F|/E is below the 1% screening threshold at both slices (Table 4). But the first-order dressing norm Z_2 exceeds the threshold by more than an order of magnitude at both. That is an order-one dressing of the reference state, not a small correction. Three checks show the failure is not a collar artifact:

- The interior shares (R <= r_c - 10 ell) carry most of Z_1. They are the same for every vertex choice and exceed the threshold on their own (Table 4).
- Z_2 is almost entirely the single pair. The double pair is a small fraction of F at both slices.
- At 15.0 most of Z_1 sits at R <= 32, the trapped core (vertex_control json: deep_interior_R_le_32). At 11.0 the single-pair amplitude is carried mainly by the current channel (radial decomposition s29, json on the box).

F_1 itself depends on the vertex choice at 11.0, where the collar's share of F_1 is large (Table 4). It depends much less on the choice at 15.0. F/E stays below 1% for every choice.

**Signed exchange.** The signed Fock exchange of the pair energy is reported separately in Table 5, against the paper's statement that it "weakens the pair energy by 0.14 to 0.16 percent". At both slices it weakens the pair energy by an amount inside that range, and inside the paper's 0.90 to 1.01 times 1/N_B, so the reproduction test passes (Table 1). It is relative to the Hartree pair energy, not to E, and it is no substitute for the correlation calculation above.

**Table 4. Connected reference energy (F excludes E_H)**

| quantity | 11.0 | 15.0 |
|---|---|---|
| single pair F_1 | -0.09458376 | -0.4576126 |
| double pair F_2 (exchange bracket) | [-0.0003947, -0.0003937] | [-0.0001141, -0.0001115] |
| F = F_1 + F_2 | [-0.0949785, -0.09497748] | [-0.4577267, -0.4577241] |
| F/E | [-0.0008058, -0.0008058] | [-0.004206, -0.004206] |
| Z_1 (single-pair dressing) | 0.136 | 1.095 |
| Z_2 (single + double) | [0.136158, 0.136159] | [1.09526, 1.09526] |
| F_1, unsymmetrized V = du/dz | -0.1171 | -0.4593 |
| F_1, V = Hessian of E | -0.07905 | -0.4564 |
| interior share (R <= r_c - 10 ell): F_1, Z_1 | [-0.05584, 0.08275] | [-0.4547, 1.091] |
| interior share with V = Hessian of E: F_1, Z_1 | [-0.05584, 0.08275] | [-0.4547, 1.091] |
| min single / double denominator | [0.2713, 0.5426] | [0.01899, 0.03797] |
| eigenbasis sum vs closed form (1/2) c^T chi c, rel | 8.385e-10 | 7.762e-10 |
| Laplace fit errors (1/x, 1/x^2) | [4.839e-08, 8.004e-07] | [1.992e-07, 4.286e-06] |
| chi: Zolotarev poles vs Laplace nodes, rel | 1.201e-08 | 2.926e-08 |

**Table 5. Signed exchange of the pair energy**

| quantity | 11.0 | 15.0 |
|---|---|---|
| Hartree pair energy (1/2) s^T V s | -7.776981 | -27.21339 |
| signed Fock exchange | 0.01143775 | 0.03839449 |
| Fock / Hartree | -0.001471 | -0.001411 |
| (Fock / Hartree) x 640 | -0.9413 | -0.903 |

### 4.4 Source derivatives and the induced field change

**The derivative is exact (Table 6).** q_1 = dF_1/du agrees with the exact scaling-direction derivative to the Laplace quadrature's rounding. That direction tests the second-order reference term. q_1 also agrees with centered differences of F_1 on the same branch along the interior bump, to the differences' truncation error, at both slices. That direction tests every term, including the solve's Hessian. The difference route's base point reproduces F_1, and no occupation changes in any trial. Along the far-edge bump the derivative is below the evaluation noise of F_1 at both slices, so those rows show consistency only within that noise.

**Induced field change (Table 7).** q_1 is inserted as extra sources into the linearized rows, with P's response included.

- **15.0: FAIL.** The represented masses at R_50 and r_in, and the inner root, shift by more than 1%, from the single pair alone.
- **11.0: PASS for the single pair.** Every induced relative clock, flow and mass shift is below 1%, and there is no root at 11.0.
- **Double pair: UNRESOLVED.** Its source profile is not computed (QUESTIONS.md Q4, default b). Its scaling-direction derivative is a small fraction of F_1's at both slices (Table 7). Cancelling the 15.0 shifts would need a double-pair profile orders of magnitude larger than its scaling projection. Nothing computed here excludes that, so the double-pair entry stays UNRESOLVED at both slices rather than being folded into the single-pair status.

The scaling-direction (boundary-charge) shift of the whole F is below 1% of E at both slices (Table 7). A small F/E is not evidence that the fields are unchanged; the 15.0 root and mass shifts show that.

The source profiles q_rho = dF_1/dN_x and q_J = dF_1/dv_x and the field shifts dN, dv are in profiles_R1_v1.npz, together with the local shares of F_1 and Z_1. They are derivatives with respect to the applied (back-averaged) site fields of the chain. As extra moments, they enter the rows through the history's own source rows: the sigma = 4a Gaussian map with its shell normalization, which is the discrete solve's actual averaging adjoint. No a^3 cubic-site conversion is used.

**Table 6. Source derivative q_1 = dF_1/du (exact) and its checks**

| quantity | 11.0 | 15.0 |
|---|---|---|
| scaling direction: q_1 . u-bar vs exact, rel | 8.811e-07 | 6.34e-07 |
| difference route reproduces F_1 at the base, rel | 4.634e-09 | 1.899e-09 |
| interior bump: q_1 . x | 0.0005234 | 0.01408 |
| interior bump: central difference rel (e = 1e-3 / 5e-4 / Richardson) | [6.452e-05, 4.963e-06, 1.489e-05] | [1.044e-06, 8.884e-07, 8.365e-07] |
| edge bump: q_1 . x | 1.259e-08 | 5.735e-10 |
| edge bump: central difference rel (e = 1e-3 / 5e-4 / Richardson) | [0.0003441, 0.00258, 0.003325] | [5.202, 0.4631, 1.116] |
| term norms: solve Hessian / Xi / chi V chi / q_1 | [0.002542, 0.04482, 0.002279, 0.04519] | [0.01305, 0.7086, 0.01226, 0.7107] |

**Table 7. Induced field change from F_1's sources (linearized rows, P's response included)**

| quantity | 11.0 | 15.0 |
|---|---|---|
| dE = u-bar . q_1, and dE/E | [0.1165, 0.0009884] | [0.4647, 0.00427] |
| max \|dN\|, max \|dN\|/N | [0.0008547, 0.001194] | [0.001675, 0.007799] |
| max \|dv\|, max \|dv\|/\|v\| (\|v\| > 1e-3) | [0.0004871, 0.003309] | [0.004788, 0.01666] |
| dM/M at R50 | 0.0008206 | 0.0171 |
| dM/M at r_in | no such radius | 0.01774 |
| dM/M at R_t | no such radius | 0.004462 |
| root 7.0: N + v there, slope, dN, dv, dR, dR/R | no roots | [0.0006987, -0.01461, 0.0008154, -0.002338, -0.1042, -0.01489] |
| root 51.8: N + v there, slope, dN, dv, dR, dR/R | no roots | [0.0003583, 0.009047, 1.737e-05, -0.002225, 0.244, 0.004711] |
| double pair: dF_2/de / dF_1/de along u-bar | 0.004259 | 0.0002453 |
| scaling-direction dE of F (boundary charge), dE/E bracket | [0.0008012, 0.0008012] | [0.004255, 0.004255] |

### 4.5 Represented masses and the mass Hessian

The represented radii are frozen at their nearest 0.1a rows: R_50 at 11.0, and R_50 and the record's inner and outer roots at 15.0 (Table 8). The mass gradients at fixed P satisfy the paper's window identity to rounding at every radius. The represented masses equal the engine's Misner-Sharp values on the same fields (vertex_control json).

The complete Hessians K_j = K0_j - V Xi_j V were checked as the spec asks ("differentiate this identity ... to check the mass Hessian"). The mass gradient G_j(z) was recomputed at displaced moments z = z-bar +/- e dz along the self-consistent admitted map u(z) = U(z - m_P(u)). At every trial point the reference was rebuilt, as were chi, S' and the exact du/dz. The centered differences were compared with K_j dz (Table 8). The difference route's dM agrees with G_j . dz (json), and its step sizes agree with each other. The second-order reference term V Xi V is not small against K0 (vertex_control json, mass_hessian_fd), and the differences resolve it: they agree with K_j far better than with K0_j at every radius.

By the rule fixed before the check (Richardson against K_j dz below 1e-4 on interior rows), R_50 and R_t pass at 15.0 and r_in does not. The r_in mismatch is small and lies between the rule's two thresholds, so the 15.0 entry is UNRESOLVED (Table 1). Its step sizes agree with each other, so the residual is not truncation error. The residual is not explained here. The candidates not ruled out are the stored applied fields' small departure from the self-consistent fields at the base point, where K_j was built, and the Laplace quadrature inside K0_j's chi. At 11.0, R_50 is the only radius (Table 8).

G_j barely depends on the vertex choice, because the masses' windows are interior. The collar ambiguity therefore does not reach the mass gradients or the branch moments.

**Table 8. Represented masses and the mass Hessian**

| quantity | 11.0 | 15.0 |
|---|---|---|
| R50: r_j, M_j, window identity rel | [59.8, 61.17, 1.806e-13] | [28.9, 80.78, 5.622e-13] |
| R50: d(G_j)/dz vs K_j dz, rel: all / interior / vs K0 interior | [0.0001843, 3.464e-05, 0.0474] | [0.0002315, 4.116e-05, 0.1017] |
| R50: G_j with du/dz vs with symmetrized V, rel: all / interior | [2.166e-09, 2.166e-09] | [2.459e-09, 2.457e-09] |
| r_in: r_j, M_j, window identity rel | no such radius | [7, 13.89, 2.486e-13] |
| r_in: d(G_j)/dz vs K_j dz, rel: all / interior / vs K0 interior | no such radius | [0.001008, 0.0001811, 1.047] |
| r_in: G_j with du/dz vs with symmetrized V, rel: all / interior | no such radius | [7.524e-10, 7.522e-10] |
| R_t: r_j, M_j, window identity rel | no such radius | [51.8, 103.5, 1.341e-12] |
| R_t: d(G_j)/dz vs K_j dz, rel: all / interior / vs K0 interior | no such radius | [1.286e-05, 2.312e-06, 0.01919] |
| R_t: G_j with du/dz vs with symmetrized V, rel: all / interior | no such radius | [7.9e-11, 7.899e-11] |

### 4.6 Mass bias, fully reordered norms and the branch weight

The full second-order mass operators are reordered about the actual Hartree state C-bar (spec sec. 5), with the complete mass Hessians K_j (Table 9). The occupied/empty split in S_1 and S_2 is that of C-bar. The hole count, from the hole identity, is in Table 2. No norm replaces D by the dust orbitals.

- **11.0.** The represented mass is M(R_50). Its bias beta/M is far below 1% and below the measured rms fluctuation, which the single-pair norm sqrt(S_1)/M dominates; the double-pair norm is more than an order of magnitude smaller. The upper end of <B> lies far below 0.080.
- **15.0.** Three masses are represented: R_50, the inner root and the outer root. Every bias is far below 1% and below its rms fluctuation. The inner root's single-pair norm is the largest relative fluctuation, and it carries most of <B>. The sum's upper end lies just below 0.080, within a fraction of a percent. The S_2 exchange at 15.0 is bracketed by the looser unweighted channel sum (s40c), not the per-copy sum used at 11.0. Both are rigorous, and the 15.0 brackets are narrow at the shown digits (Table 9).

So the paper's one-route bound <B> <= 0.080, and with it the branch weight at least 0.92 and Pr(M_j <= 0) at most 0.020 (Table 9, last row), is reproduced by this route on these two slices with the complete second-order masses, at 15.0 with almost no margin. These slices are two of the paper's 18. The rms fluctuations of the linearized masses (sqrt(S_1)/M) lie inside the paper's printed 0.9 to 14 percent at every represented radius.

**E_2 reordered about C-bar (Table 10).** beta/E is the signed Fock exchange over E. The full single-pair matrix h - X_V(D) carries the Hartree rate of the held h: C-bar is not stationary under h, so this norm measures the state's rate of change rather than a correction. At 15.0 it exceeds 1% of E, and the test is recorded as FAIL by the 1% screening rule fixed in advance. The residual matrix -X_V(D), relative to the Hartree tangent Hamiltonian, is far below 1% at both slices. Its S_1 is exactly the beyond-Hartree singles norm r_F^2 of VDRESS_RESIDUAL_R1_v1.md. The two norms are labelled separately, as the spec requires.

**Table 9. Mass bias, fully reordered norms (about C-bar) and branch weight**

| quantity | 11.0 | 15.0 |
|---|---|---|
| R50: beta/M | 0.0001034 | 0.0003725 |
| R50: sqrt(S_1)/M | 0.03215 | 0.03477 |
| R50: sqrt(S_2)/M | [0.001072, 0.001075] | [0.001289, 0.001318] |
| R50: 4(beta^2 + S_1 + S_2)/M^2 | [0.004138, 0.004138] | [0.004843, 0.004843] |
| r_in: beta/M | no such radius | 8.563e-05 |
| r_in: sqrt(S_1)/M | no such radius | 0.1362 |
| r_in: sqrt(S_2)/M | no such radius | [0.002152, 0.0022] |
| r_in: 4(beta^2 + S_1 + S_2)/M^2 | no such radius | [0.07421, 0.07421] |
| R_t: beta/M | no such radius | 0.0003709 |
| R_t: sqrt(S_1)/M | no such radius | 0.01367 |
| R_t: sqrt(S_2)/M | no such radius | [0.0005977, 0.0006104] |
| R_t: 4(beta^2 + S_1 + S_2)/M^2 | no such radius | [0.0007496, 0.0007497] |
| <B> (sum over represented masses) | [0.004138, 0.004138] | [0.0798, 0.0798] |
| <Pi> lower bound 1 - b_+; Pr(M_j <= 0) upper b_+/4 | [0.9959, 0.001035] | [0.9202, 0.01995] |

**Table 10. E_2 reordered about C-bar, normalized by E**

| quantity | 11.0 | 15.0 |
|---|---|---|
| beta/E (both labels) | 9.703e-05 | 0.0003528 |
| sqrt(S_1)/E, full single-pair matrix h - X_V(D) | 0.001861 | 0.01075 |
| sqrt(S_1)/E, residual single-pair matrix -X_V(D) | 1.324e-05 | 1.591e-05 |
| sqrt(S_2)/E | [0.0005258, 0.0005271] | [0.0005682, 0.0005801] |

### 4.7 Leakage (a separate output)

The single/double-pair norms above are not lambda_Pi and are not relabelled as leakage. The spec's cheap global certificate L_global uses the full copy multiplicity. At both slices it is many orders of magnitude above every threshold (Table 11), for tau L and for 64 a L alike, so the result is **LEAKAGE_UNRESOLVED** at both slices. This is the certificate's looseness, not a measured leakage: the bound adds the trace norms of every copy of every channel. A tighter certified bound on the compressed operator, or its action on a demonstrably sufficient branch subspace, was not attempted.

**Table 11. Global leakage certificate (copies included)**

| quantity | 11.0 | 15.0 |
|---|---|---|
| t_trace_norm_full | 7.186e+08 | 1.969e+09 |
| quartic_term | 2.008e+09 | 5.426e+09 |
| L_global | 2.727e+09 | 7.395e+09 |
| tau_L | 2.182e+10 | 5.916e+10 |
| 64a_L | 1.745e+11 | 4.733e+11 |

## 5. Notes on the open premises (amendment 1)

1. **Is the subtracted object P or P + X?** Under E_2 the first-order correction of the reference's one-body matrix is delta P = X + X^dag, with X = -sum_{alpha i} B_{alpha i}/(eps_alpha - eps_i) |alpha><i|. So ||delta P||_F^2 = 2 Z_1 in the copy-weighted norm. Z_1 is about one tenth at 11.0 and of order one at 15.0 (Table 4). The dressed reference therefore differs from P by a Frobenius norm that is not small at either slice. Any premise that treats the subtracted object as P with a perturbatively small correction is contradicted by these numbers. At 15.0 the correction sits mostly in the trapped core, R <= 32. This bears on the paper's ordering prescription ("fully Wick order against the continued reference P") but does not decide it: the computation measures the size of X, not which object the theory should subtract.
2. **Band projection of the evolution-row stress.** This route has no finding. It post-processes held slices through the static rows only (energy and momentum constraints of the discrete admitted solve), and no evolution-row stress enters.
3. **The vertex is ambiguous in the collar.** The spec's two definitions of V, the Hessian of E and delta h = sum_mu A_mu V_mu nu delta z_nu, coincide only where delta E = Tr(h delta C). The stored rows break that identity in the collar (sec. 4.2). Any collar-sensitive quantity, F_1 at 11.0 above all, needs the collar's convention fixed before two routes can be compared there. The interior shares do not have this ambiguity.

## 6. Applicability flags (separate from the numerical errors)

- **Hydrodynamic gate.** The paper's one-term test first fails at T/r_s 11.75. 11.0 lies before the gate. 15.0 lies past it, and a quantum-level result there does not restore the hydrodynamic premise.
- **Path at 15.0.** The path is not gapped: crossings are continued by the record's LZ convention from 11.125, there are pockets at record 112, and the record's own continuation "verified" flags are false at record 120 (vertex_control json, gate_reference). The slice's P is the ground projector, but the paper's gapped-path premise for the energy result fails there.
- **Collar.** The stored rows are not variational in the last 10 ell of the chain. Totals that reach the collar (F_1 at 11.0 above all) depend on the vertex choice there; the interior shares do not (sec. 4.2, Table 4).
- **Domain.** 15.0 is evaluated on the doubled domain B2n4096 (4096 sites per chain), not on 2048-site chains. The brief prescribes this, and B2n4096 matches W5 on 0 to 14.
- **Record self-consistency at 11.0.** The applied fields of record 88 differ from the admitted solve of the replayed sources by the amount in the vertex_control json (applicability flags). This route evaluates P and h at the record's applied fields, as stored. The mass-Hessian check runs the self-consistent map and shows the resulting change in M_j is far below every reported digit (fdK json).
- **Scope.** These are ten-channel results, not a cutoff limit. They cover two of the paper's 18 slices. The branch reproduction against 0.080 applies to these two slices only, and confirming "all 18 slices" requires processing those 18 inputs.

## 7. Self red-team: what would make each result wrong, and what rules it out

| Risk | What it would corrupt | Check that rules it out (where) |
|---|---|---|
| Wrong replay step convention (sign of tau, which record's fields drive a step) | every output | Forward replay from 11.5 to 12.0 matches the stored 12.0 slice. Each replayed target record recovers the stored shell totals and the charge E (Table 2; gates_s08/s13 json) |
| Copy normalization applied twice or not at all | sources, V, F, all norms | Tr(C-bar - P) = 640 and the stored sources recovered with the same g_k = 16k (Table 2) |
| Wrong reference (propagated sea, P_-(h) through a crossing, new interpolation) | F, Z_2, D, holes | P equals the record's continued reference by the record's own flags. P^2 = P, [h, P] = 0, fixed rank, positive denominators (Tables 2 and 4). At 15.0 the path information is reported and the gate fails on it (sec. 4.3) |
| Impure state treated with Slater formulas | single/double pairs, S_1, S_2 | Impurity measured and reported (Table 2). The hole identity carries it explicitly |
| Operator allocations A_mu or moment kernels wrong | sources, chi, D-kernel, everything | Stored sources recovered. Small-chain brute-force tests of the moment kernels and of moments_of_matrix (s07 unit json; h16/hw2 self-test). Small Fock-space tests (Table 12) |
| Discrete solve not the history's rows | V, masses, source response | Matches the engine's solve on the same sources. Jacobian and Hessian actions match centered differences. The adjoint gradient matches the tangents (Table 3; fld json) |
| Vertex ambiguity in the collar | F_1 at 11.0, collar parts of everything | Three vertices compared. Interior shares are identical for all three and carry the FAIL readings on their own (Table 4) |
| Laplace quadrature or Zolotarev pole error | chi, F, Z_2, Xi | Fit errors measured on the actual spectral range. chi agrees between the Laplace and Zolotarev routes. The eigenbasis sum matches the closed form. The ring integrand decays by many decades over the node range (Table 4; energy json D2_nodes) |
| Exchange bracket invalid | F_2, S_2, <B> | Per-copy Cauchy-Schwarz proof (sec. 3) and the brute-force unit test T5 (Table 12). Bracket widths are small (Tables 4 and 9) |
| q_1 incomplete (a missing dV, dP, eigenvector or denominator term) | sources, induced fields | Scaling-direction identity (exact). Centered differences on the same branch along an interior direction, which test all terms at both slices (Table 6). Base point reproduced |
| Linearized response inconsistent with q_1 | dN, dv, dM, dR | dE = u-bar . q_1 equals the exact scaling derivative (Table 6). The same V gives both. Root shifts use the raw grid rows with a three-point slope (q1 json) |
| Mass gradient or Hessian wrong | beta, S_1, S_2, <B> | Window identity (Table 8). Centered differences of G_j along the self-consistent admitted map against K_j dz. The difference route's dM matches G_j . dz. G_j is insensitive to the vertex choice (Table 8) |
| Matrices corrupted between H100 and box | 15.0 branch, Fock | sha256 of every transferred file, and of every rebuilt full matrix against the pod's (branch_s40c json, transfer_check) |
| Trace-product shortcut wrong | S_2 bracket | Two-product form checked against the four-product form once per run (branch_s40c json) |
| Hole identity accidentally satisfied | holes | Computed per channel and summed with copies, independently of every contraction above (gates json) |
| Leakage misread | lambda_Pi | Only a certificate is claimed. UNRESOLVED is the result, not a small value |
| Over-reading the slices | branch and paper claims | Two slices of the paper's 18 and ten channels. The hydrodynamic gate (11.75) and the 15.0 path are attached as applicability flags (sec. 6) |

Not checked by this route: the full double-pair source profile; the exact same-copy exchange in S_2 and F_2 (bracketed only); any slice other than these two; convergence in the channel cutoff.

## 8. Deviations, open items and the questions file

- **Q1 (no stored state at 11.0 or 15.0).** Reading (a), replay, was used and validated (sec. 3). The W5 T0011.00 slice listed in md5_done.txt does not exist on the box or on HF.
- **Q2 (vertex in the collar).** Reading (a) is used, the coupled-row Jacobian, symmetrized. The unsymmetrized du/dz and the Hessian of E are reported beside it, and the interior shares remove the ambiguity (sec. 4.2).
- **Q3 (15.0 on the GPU).** Answered (a) by the lead (#280). The 15.0 bubble passes, vertex, energies, masses, second-order mass term, source derivatives, leakage and difference checks ran on the lane's H100. The items that need C-bar (D-kernel, Cov_C, S_1, Fock exchange) ran on the box CPU with the H100's matrices transferred and verified.
- **Q4 (source profile).** Default (b) is used: the exact F_1 profile and its induced fields, plus the exact scaling-direction shift of the whole F. The double-pair profile is UNRESOLVED.
- **Exact S_2 and F_2 exchange.** Bracketed by a rigorous per-copy bound, not evaluated exactly. Table 9 shows the brackets are narrow compared with every threshold.
- **Leakage.** LEAKAGE_UNRESOLVED. Only the cheap certificate was evaluated.
- **Route comparison.** This is route 1 only. Agreement with route 2 within stated errors (requirements sec. 5) is for the lead to reconcile. Per the spec, intermediate residuals and channel-resolved outputs should be reconciled before totals. The per-channel F_1 and Z_1 are in the energy json files on the box.

## 9. Files

- **Deliverables** (<internal>/route2): this report, VREF_R1_v1_box_record.md and vertex_control_R1_v1.json. Their md5s are in the box record.
- **Box** (<internal>/webgpt_vref_r1):
  - final/: vertex_control_R1_v1.json, tables_R1_v1.md, tables/ and profiles_R1_v1.npz, which holds the radial profiles of q_rho, q_J, dN, dv, c and the local shares of F_1 and Z_1 at both slices.
  - out/w5_t11/ and out/b2_t15/: every result json and npz.
  - from_h100/: the H100's results, logs and scripts.
  - scripts/ and logs/.
- The input md5s are in Table 13. The sha256 of every result file the tables were built from is in vertex_control_R1_v1.json (result_files_sha256).

**Table 13. Inputs (read only)**

| input | md5 |
|---|---|
| W5_json | 6d04fb0698c2a87d6f5f739456f26b3c |
| W5_fields | 22cc05a1149d124cf55a1d15b656cdfb |
| W5_spectra | 7e4642bdd91c4d424c1b27fd9fbab13f |
| W5_handoff | 83c91538038c4384844290af03d45254 |
| W5_slice_T11.50 | 93cc1c016b64a82641375a512bbf67c8 |
| W5_slice_T12.00 | fa8b784265bd9d5d078fb03d6cf7cc83 |
| B2_json | 417a4018cf6497f9b0020408783f44f5 |
| B2_fields | cb26fcfaf3d6e80a17f0e73bac687301 |
| HF slice T/r_s 16.0 (sha256 = LFS) | bb5f1dd93bcec2ef577a4a3cc364f187f84cc84f9575736945dfc4bc3506bb13 |
| HF slice T/r_s 16.0 (md5) | b5be00b2ab42f70e17f87fa019cc2388 |

