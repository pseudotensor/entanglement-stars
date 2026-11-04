# VDRESS route 1: the beyond-Hartree residual about the stored Hartree state (v1.1)

Webgpt lane, route 1, blind to route 2. This version applies the lead's ruling #307 (amendment 5), which both reviewers signed (deliveries/lead_gaps/AstraAPI_vdress3_v1.md, FableAPI_vdress3_v1.md), to VDRESS_RESIDUAL_R1_v1.md. v1 is kept unchanged beside it. The calculation, its preregistered thresholds and its stored inputs are those of v1 (VDRESS_CALC_SPEC_v1; VREF_R1_v1.md). Every number in the tables was copied from the result files by script.

## 1. What changed from v1

1. **(d) is printed as UNTESTED on the 10-channel records.** It is neither "pass" nor "not passed". Channels 11-20 added as dust-free sea with the 10-channel vertex held are a frozen-background extension, not a genuine history-cutoff comparison. The preregistered reading Q6 and its tables are kept below as the record (Tables V1-V5, unchanged from v1).
2. **The ruling's zero-cost check** on the mixed sea+dust class, at both slices (sec. 3, Table V6).
3. **The genuine 14- and 20-channel histories.** The existing formation-control histories W5_k14 and W5_k20 store the T/r_s 11.0 slice, so (a)-(d) are evaluated on them, read only and with no run, each with its own C-bar, P and V (sec. 4, Table V7). Their slices stop at T/r_s 14.0, so there is no 15.0 counterpart.
4. **The dust-only doubles are printed beside r_H** (Table V8, and in Table V7 for every history).

## 2. Summary (printed status under the ruling)

**The Hartree level is not refuted at second order on these slices. Correlation beyond Hartree-Fock is uncontrolled.** No refutation criterion is met on any record evaluated here: neither slice at ten channels, nor the genuine 14- and 20-channel histories at 11.0. r_F/r_B stays far below 0.3 and every Fock-induced mass rate stays far below 1e-2 (Tables V1, V2, V7).

**(d) is UNTESTED on the 10-channel records**, at both slices. Under the ruling, the padded extension is not a genuine history-cutoff comparison. The preregistered Q6 tables are kept as the record (sec. 9).

**The ruling's zero-cost check: equal at both slices** (Table V6). The cutoff growth of the mixed sea+dust class is exactly the contraction of the dust's single-excitation Gram with the added sea covariance, to roundoff. It is the sea's fluctuation dressing the dust, reported without a threshold. There is no excess growth.

**The genuine 14- and 20-channel histories at 11.0** (Table V7), each with its own state, reference and vertex:

- The gates pass.
- **(a) r_F/r_H grows with the channel cutoff and crosses the 1e-2 pass line at 20 channels.** The preregistered pass condition therefore fails on the genuine 20-channel history at 11.0. This is a failed pass, not a refutation.
- **(c)** is stable across the three histories and well below 1e-2.
- **(b)** stays inside the predicted window.
- **The two-dust class moves by less than 1% from 10 to 14 channels and by just over 1% from 14 to 20.** At the preregistered 1% tolerance it is not cutoff-stable, though marginally.
- The sea-dust and sea-sea classes grow with the cutoff as in the padded extension, which tracked the genuine histories closely at this slice.
- The vertex itself depends strongly on the cutoff: cond(R_u), r_B and the single-pair screen F_1, Z_1 all grow from 10 to 20 channels. This is the "readouts' sensitivity to V" the ruling asked to see.

**Dust-only doubles beside r_H** (Tables V7, V8): the two-dust doubles residual is about ten percent of r_H at 11.0 and about four percent at 15.0 on the 10-channel records.

**What is controlled and what is not.** Control is established for the one-body (Fock) correction on these slices at ten channels, and it weakens with the cutoff (a). The dust's second-order coupling to the sea's fluctuation is measured, not bounded, and grows with the cutoff. The total residual gives no leakage bound. Not a trajectory bound, not uniform in chi.

## 3. The ruling's zero-cost check on the mixed sea+dust class (Table V6)

The ruling decomposes the mixed class exactly:

r_mix^2 = (flat part: the same-copy exchange, with the sea and dust holes in one copy) + sum V_mu nu V_mu'nu' D_mu mu' Sigma_nu nu'

- D_mu mu' = Tr[(I - C-bar) A_mu Pi_d A_mu'] is the dust's single-excitation Gram. It lives in the channels that carry dust and does not depend on the cutoff.
- Sigma_nu nu' = Tr[(I - C-bar) A_nu S A_nu'] is the sea's source covariance. S is the replayed sea panel for k <= 10 and the reference P for the added channels.

**The test.** s42 recomputed D in its own pass and contracted it with the added channels' sea covariance dSigma alone (the H100's buckets for channels 11-14 and 15-20). It compared the result with the 10 -> 14, 14 -> 20 and 10 -> 20 increments of r_mix^2 that s39b printed as differences of its cutoff totals.

**Result.** At both slices the increments equal the direct contractions to roundoff. The flat exchange part is the same at every cutoff, and the recomputed D reproduces s39b's dust-dust direct part. So the mixed class's growth with the cutoff is the sea's fluctuation dressing the dust's single excitation. As the ruling directs, it is reported without a threshold, and there is no excess growth that would signal genuine dust correlation.

What the check does and does not establish: on the padded records V, C-bar and P are held, so the equality is also linear algebra. Its content is that the computed classes carry no cutoff dependence other than the dSigma contraction. It says nothing about how the classes would move in genuine histories, where V and the dust orbitals change with the cutoff. Sec. 4 measures that directly at 11.0.

**Table V6. The ruling's zero-cost check: r_mix^2 = (flat exchange part) + sum V V D Sigma (s42)**

| quantity | value |
|---|---|
| T11.0: increment 10 -> 14 of r_mix^2 (s39b) / direct contraction with dSigma / rel. difference | [0.001651, 0.001651, 2.626e-15] |
| T11.0: increment 14 -> 20 of r_mix^2 (s39b) / direct contraction with dSigma / rel. difference | [0.002815, 0.002815, 3.697e-15] |
| T11.0: increment 10 -> 20 of r_mix^2 (s39b) / direct contraction with dSigma / rel. difference | [0.004466, 0.004466, 3.107e-15] |
| T11.0: flat (same-copy exchange) part, per-copy bound at cutoffs 10, 14, 20 | [2.401e-06, 2.401e-06, 2.401e-06] |
| T11.0: direct part of r_mix^2: Sigma(10) part, + channels 11-14, + channels 15-20 | [0.002245, 0.001651, 0.002815] |
| T11.0: D2_dd recomputed against s39b (rel.) | 1.175e-16 |
| T15.0: increment 10 -> 14 of r_mix^2 (s39b) / direct contraction with dSigma / rel. difference | [0.001111, 0.001111, 1.366e-14] |
| T15.0: increment 14 -> 20 of r_mix^2 (s39b) / direct contraction with dSigma / rel. difference | [0.001973, 0.001973, 8.794e-15] |
| T15.0: increment 10 -> 20 of r_mix^2 (s39b) / direct contraction with dSigma / rel. difference | [0.003084, 0.003084, 1.055e-14] |
| T15.0: flat (same-copy exchange) part, per-copy bound at cutoffs 10, 14, 20 | [1.39e-06, 1.39e-06, 1.39e-06] |
| T15.0: direct part of r_mix^2: Sigma(10) part, + channels 11-14, + channels 15-20 | [0.001329, 0.001111, 0.001973] |
| T15.0: D2_dd recomputed against s39b (rel.) | 3.576e-16 |

## 4. The genuine 14- and 20-channel histories at T/r_s 11.0 (Table V7)

The formation-control histories W5_k14 and W5_k20 store their T/r_s 11.0 slices (md5 listed in slices/md5_done.txt). Both are evaluated read only, with no run, and with the 10-channel W5 history at 11.0 (v1) beside them. Each history enters with its own:

- stored state C-bar (sea panels and dust columns, read directly; no replay);
- reference P (the slice's R2 panel, the negative set of the held h at 11.0);
- applied fields and stored sources at the 11.0 record;
- vertex V (its own chi over 14 or 20 channels and its own discrete solve);
- mass gradient G at its own R_50 row.

The interior-row restriction and every formula are those of v1.

**Gates.** The slices' md5s match. Tr(C-bar - P) = 640 with g_k = 16k. The stored shell totals are recovered to roundoff. R2 commutes with h, has half-filling rank and only negative levels in every channel. The dust occupations are one. R_50 falls on the 59.8 row at 14 channels and on the 59.9 row at 20; G was prepared for both.

**Readouts.**

- **(a)** r_F grows faster than r_H with the cutoff, and r_F/r_H crosses 1e-2 at 20 channels.
- **(b)** r_F/r_B stays inside the predicted window. r_B itself grows with the vertex.
- **(c)** The Fock-induced mass rate at R_50 hardly changes.
- **(d)** The two-dust class shifts by less than 1% at the first step and by just over 1% at the second. The sea-dust and sea-sea classes grow as the sea's added fluctuation predicts, close to the padded extension's values.

**Reading.** On the genuine ladder at 11.0:

- The preregistered pass condition fails at 20 channels through (a), not through (c).
- The two-dust clause is marginally outside the preregistered 1% per step. reviewer A's ruling calls this a test of the channel stability of the Hartree-level inputs, the vertex and the dust orbitals, and the vertex visibly is not channel-stable: cond(R_u), r_B and the single-pair screen grow strongly.
- No refutation criterion is met.

The 15.0 slice has no genuine counterpart.

**Table V7. The genuine 10-, 14- and 20-channel histories at T/r_s 11.0 (each with its own C-bar, P and V; interior rows)**

| quantity | 10 channels (W5) | 14 channels | 20 channels |
|---|---|---|---|
| slice md5 matches md5_done.txt | replayed (VREF gates) | True | True |
| Tr(C-bar - P), copies g_k = 16k | VREF Table 2 | 640 | 640 |
| stored sources recovered (eps, J rel.) | VREF Table 2 | [5.438e-13, 3.308e-14] | [3.002e-12, 8.281e-14] |
| continuation at the 11.0 record | verified (Weyl) at record 88 | LZ convention (c_LZ = 1) from T/rs 10.750 (channels 11,12,13,14); inversions 0; n_pocket 0; intermediate weights 0 | LZ convention (c_LZ = 1) from T/rs 10.375 (channels 11,12,13,14,15,16,17,18,19,20); inversions 0; n_pocket 0; intermediate weights 0 |
| cond(R_u) of the history's own vertex | 1.368 | 1.806 | 3.117 |
| single pair on the history's own vertex: F_1, Z_1 | [-0.09458, 0.136] | [-0.1754, 0.2387] | [-0.3373, 0.4332] |
| r_H | 0.2197 | 0.234 | 0.2622 |
| dust-only doubles r_D (two dust holes), beside r_H | [0.02135, 0.02162] | [0.0215, 0.02178] | [0.02175, 0.02203] |
| r_F | 0.00156 | 0.002101 | 0.003415 |
| r_B | 0.3306 | 0.4637 | 0.6652 |
| (a) r_F/r_H  [1e-2] | 0.007103 | 0.008979 | 0.01302 |
| (b) r_F/r_B  [refute >= 0.3] | 0.004719 | 0.004531 | 0.005134 |
| (c) dMdot/Mdot at R_50 (row r_j)  [1e-2] | -0.001048 (r_j 59.8) | -0.001022 (r_j 59.8) | -0.0009829 (r_j 59.9) |
| (d) r_D sea-sea | [0.03396, 0.034] | [0.05948, 0.05951] | [0.1055, 0.1056] |
| (d) r_D sea-dust (mixed) | [0.04736, 0.04741] | [0.06281, 0.06285] | [0.0832, 0.08323] |
| (d) r_D dust-dust | [0.02135, 0.02162] | [0.0215, 0.02178] | [0.02175, 0.02203] |
| (d) r_D with a dust hole | [0.05195, 0.0521] | [0.06639, 0.06652] | [0.086, 0.0861] |
| (d) r_D total | [0.06206, 0.06221] | [0.08914, 0.08925] | [0.1361, 0.1362] |
| two-dust r_D relative change from the previous history (genuine histories; 1% per step) |  | 0.007382 | 0.01161 |
| dust-hole r_D relative change from the previous history (genuine histories; 1% per step) |  | 0.2767 | 0.2944 |

## 5. Dust-only doubles beside the Hartree rate

As the ruling asks, the two-dust-hole doubles residual is printed beside r_H at both 10-channel slices. Table V7 prints it for each genuine history at 11.0.

**Table V8. Dust-only doubles beside the Hartree rate (both slices, 10 channels)**

| quantity | T/r_s 11.0 | T/r_s 15.0 |
|---|---|---|
| r_H | 0.2197 | 1.17 |
| r_D, two dust holes (bracket) | [0.02135, 0.02162] | [0.04913, 0.04936] |
| r_F | 0.00156 | 0.001731 |

## 6. Method additions in v1.1

- **s42 (box): the zero-cost check.** D, the dust kernel, is recomputed per channel from the replayed panels and saved. It is contracted with the H100's sea covariances of the added channels on interior rows. The result is compared with s39b's increments and its flat-part bounds (sec. 3).
- **hk1 (H100): the 14- and 20-channel histories without C-bar.** From each history's own fields and json at the 11.0 record (read only):
  - eigh of every channel (14 or 20);
  - chi by the 32-node Laplace quadrature;
  - the discrete admitted solve at the stored sources, with its Jacobian and V = (I + S' chi)^{-1} S';
  - the interior-row restriction;
  - the mass gradient G = (I - chi V)^T M' at the candidate R_50 rows;
  - r_B;
  - the single-pair F_1 and Z_1 on that history's own vertex.
- **s44 (box): the 14- and 20-channel histories with C-bar.** It reads the stored 11.0 slice (no replay) and checks its md5 against slices/md5_done.txt. Gates: Tr(C-bar - P) = 640; the stored shell totals recovered; P = the slice's reference panel R2, shown to be the negative set of the held h by the commutator, the rank and the spectrum of R2^dag h R2. It finds R_50 from the stored dust and then runs s39b's VDRESS pass with every channel genuine. V comes from hk1, rebuilt from its exact upper triangle and verified by sha256.

## 7. Self red-team, additions for v1.1 (v1's table, sec. 5 of v1, still applies)

| Risk | Readout at stake | Check that rules it out (where) |
|---|---|---|
| The zero-cost equality holds only because one array was reused | the mixed-class reading | D was recomputed in its own pass and contracted with dSigma alone. The s39b side is a difference of two independently contracted cutoff totals. D's dust-dust direct part reproduces s39b's (Table V6) |
| The flat (exchange) part hides cutoff dependence | the mixed-class reading | Its per-copy bound is identical at the three cutoffs (Table V6). Same-copy exchange needs both holes in one copy, and the added channels carry no dust |
| The 14/20-channel inputs misread (wrong record, wrong weights, wrong reference) | Table V7 | Record index from each history's own clock (T/r_s = 11.0). Copy weights 16k checked. Tr(C-bar - P) = 640. The stored shell totals are recovered. R2 commutes with the held h, has full half-filling rank and only negative levels. Each slice's md5 matches md5_done.txt (Table V7, json) |
| The 14/20-channel vertex wrong | Table V7 (a)-(d) | It is built by the code that reproduced the 10-channel vertex and F_1 on the box (VREF; H100 A/B). For each history: chi symmetric, chi annihilates the applied fields, the discrete solve reproduces the record's charge E, and the transfer to the box is verified by sha256 (json) |
| The R_50 row differs between histories | (c) in Table V7 | Each history's R_50 is computed from its own stored dust and represented at its own nearest 0.1 a row, with G prepared for the neighbouring rows (json) |
| Over-reading the genuine comparison | sec. 4 | Three histories at one slice. The 14- and 20-channel histories' added channels follow the LZ convention from before 11.0 (record flags in Table V7), and the 20-channel history is uncertified from 14.0 (reviewer A's ruling). Only 11.0 is compared |

## 8. Applicability, additions for v1.1 (v1's sec. 6 still applies)

- **The genuine comparison covers one slice.** The 14- and 20-channel histories store slices up to T/r_s 14.0 only, so the genuine cutoff comparison is at 11.0. At 15.0, (d) stays UNTESTED.
- **The added channels' path.** In the 14- and 20-channel histories, the added channels are continued by the LZ convention from before 11.0, at 10.75 and 10.375 (record flags, Table V7). At the 11.0 record their references are still the negative sets of the held h (gates, Table V7).
- **Certification.** reviewer A's ruling records the 20-channel history as uncertified from T/r_s 14. 11.0 lies inside the certified range.
- **What the genuine comparison decides.** It tests the channel stability of the Hartree-level inputs at the slice (the vertex and the dust orbitals) through the dust-containing classes, at three cutoffs. It is not a convergence proof in the channel cutoff.

## 9. The v1 record (preregistered reading Q6), unchanged

The tables below are v1's: hv3's output file, copied unchanged. Their (d) rows and the "NOT PASSED, NOT REFUTED" classification in Table V4 are the preregistered Q6 record. Under the ruling, (d) on these 10-channel records is printed UNTESTED (sec. 2). v1's method, reading, red-team and applicability sections (VDRESS_RESIDUAL_R1_v1.md, secs. 2 and 4-6) still apply.

**Table V1. Preregistered readouts (interior rows; full V in brackets where computed)**

| readout | T/r_s 11.0 (W5) | T/r_s 15.0 (B2n4096) |
|---|---|---|
| r_H = \|\|(1 - C) h C\|\|_2 | 0.2197 | 1.17 |
| r_F = \|\|(1 - C) F C\|\|_2 | 0.00156 (0.00156) | 0.001731 |
| r_B = \|\|(1 - P) h_H[D] P\|\|_2 | 0.3306 (0.2836) | 0.485 (0.4828) |
| (a) r_F/r_H  [pass < 1e-2; predicted < 1e-3] | 0.007103 | 0.001479 |
| (b) r_F/r_B  [refute >= 0.3; predicted 1.6e-3 to 0.040] | 0.004719 | 0.003569 |
| paper's printed exchange-to-direct ratio of the pair energy (the prediction's comparison) | 0.0014 to 0.0016 (all 18 slices, printed) | 0.0014 to 0.0016 (all 18 slices, printed) |
| this route's signed Fock/Hartree pair-energy ratio at the slice | -0.001471 | in VREF_R1_v1.md Table 5 (s40) |

**Table V2. Fock-induced mass rate through the linearized rows (G_j = dM_j/dz along the admitted map)**

| readout | 11.0 | 15.0 |
|---|---|---|
| (c) dMdot/Mdot at R50  [pass < 1e-2; predicted < 1e-3] | -0.001048 (-0.001048) | 0.001224 |
|    Hartree mass rate Mdot at R50 (per unit clock) | 0.5195 | -0.3063 |
| (c) dMdot/Mdot at r_in  [pass < 1e-2; predicted < 1e-3] | no such radius | 0.000842 |
|    Hartree mass rate Mdot at r_in (per unit clock) | no such radius | -0.1474 |
| (c) dMdot/Mdot at R_t  [pass < 1e-2; predicted < 1e-3] | no such radius | 0.002506 |
|    Hartree mass rate Mdot at R_t (per unit clock) | no such radius | 0.03195 |

**Table V3. (d) r_D by hole content and channel cutoff (brackets: per-copy exchange bounds)**

| readout | 11.0 | 15.0 |
|---|---|---|
| cutoff 10: r_D, two sea holes | [0.03396, 0.034] | [0.01252, 0.01254] |
| cutoff 10: r_D, one sea, one dust | [0.04736, 0.04741] | [0.03643, 0.03647] |
| cutoff 10: r_D, two dust | [0.02135, 0.02162] | [0.04913, 0.04936] |
| cutoff 10: r_D with a dust hole (sea-dust + dust-dust) | [0.05195, 0.0521] | [0.06117, 0.06137] |
| cutoff 10: r_D total | [0.06206, 0.06221] | [0.06244, 0.06264] |
| cutoff 14: r_D, two sea holes | [0.05909, 0.05913] | [0.0232, 0.02321] |
| cutoff 14: r_D, one sea, one dust | [0.0624, 0.06244] | [0.04939, 0.04941] |
| cutoff 14: r_D, two dust | [0.02135, 0.02162] | [0.04913, 0.04936] |
| cutoff 14: r_D with a dust hole (sea-dust + dust-dust) | [0.06595, 0.06608] | [0.06966, 0.06984] |
| cutoff 14: r_D total | [0.08855, 0.08867] | [0.07343, 0.0736] |
| cutoff 20: r_D, two sea holes | [0.1036, 0.1036] | [0.04478, 0.04479] |
| cutoff 20: r_D, one sea, one dust | [0.08191, 0.08194] | [0.06642, 0.06644] |
| cutoff 20: r_D, two dust | [0.02135, 0.02162] | [0.04913, 0.04936] |
| cutoff 20: r_D with a dust hole (sea-dust + dust-dust) | [0.08465, 0.08474] | [0.08262, 0.08277] |
| cutoff 20: r_D total | [0.1337, 0.1338] | [0.09397, 0.09411] |
| dust-hole r_D relative change 10->14, 14->20  [stable < 1e-2] | [0.2682, 0.2825] | [0.138, 0.1851] |

**Table V4. Zero-cost adjunct and classification**

| quantity | 11.0 | 15.0 |
|---|---|---|
| <P\|B\|P> (bracket) | [3.797, 3.797] | [7.48, 7.48] |
| refutes the Hartree level at the slice | False | False |
| slice classification | NOT PASSED, NOT REFUTED | NOT PASSED, NOT REFUTED |
| (a) pass / (c) pass / (d) cutoff-stable | True / True / False | True / True / False |
| overall (both slices) | NOT PASSED, NOT REFUTED | NOT PASSED, NOT REFUTED |

**Table V5. Identity checks (V1-V6: explicit small Fock space, hv2b; then full size, s39b)**

| check | value |
|---|---|
| V1 re-Wick operator identity, max \|difference\| (operator scale) | [7.124e-15, 19.75] |
| V1 <E_2>_Cbar vs E_HF[Cbar] | 2.22e-15 |
| V2 residual norm^2 vs r_F^2 + r_D^2 (difference; values) | [1.599e-14, 26.88, 26.88] |
| V3 singles vs -(1 - C)F C (max \|difference\|; scale) | [1.798e-15, 3.014] |
| V3 direct singles of the P-ordered one-body part vs -H_V(D); quartic's singles vs H_V(D) - F | [8.006e-16, 1.776e-15] |
| V3 size of the cancelled direct singles | 1.414 |
| V4 hole-content classes sum to r_D^2 (difference); brackets hold | 2.665e-15; {'ss': True, 'sd': True, 'dd': True} |
| V5 <P\|(M - M_j)^2\|P> formula (difference) | 2.842e-14 |
| V6 rate identity i<[E_2, A]> = Tr(A(-i[h - F, C])) (max \|difference\|; scale) | [5.107e-15, 7.811] |
| T11.0 full size: identity_singles_rewick_rel | 5.154e-13 |
| T11.0 full size: identity_rF_full_vs_s14_E2_residual_S1_rel | 5.219e-15 |
| T11.0 full size: identity_rD_full_direct_vs_s14_E2_S2_direct_rel | 1.235e-12 |
| T11.0 full size: identity_E2_expectation_P_ordered_vs_E_plus_Ex | {"P_ordered": 117.88558657804889, "E_plus_Ex": 117.88558657804887, "diff": 1.4210854715202004e-14} |
| T11.0 full size: AB_PiP_vs_s11b_cov0_rel | 3.429e-14 |
| T11.0 full size: AB_PiP_vs_hv1 | {"fro": [2252.017858102245, 2252.0178581022637], "trace_real": [78091.3434683946, 78091.34346839465]} |
| T11.0 full size: AB_rB_int_box_vs_hv1 | [0.3306, 0.3306] |
| T15.0 full size: identity_singles_rewick_rel | 1.921e-13 |

## 10. Files

- **Deliverables** (<internal>/route2): VDRESS_RESIDUAL_R1_v1.1.md (this file); VDRESS_RESIDUAL_R1_v1.md (unchanged). The process record for both (scripts, md5s, kills, transfers) is in VREF_R1_v1_box_record.md, secs. 5-7.
- **Box** <internal>/webgpt_vref_r1:
  - out/vdress/: rmix_s42_v1.json and Pid_T11.0_s42_v1.npy / Pid_T15.0_s42_v1.npy (s42); k1420_s44_v1.json (s44); vdress_control_R1_v1.1.json and vdress_tables_R1_v1.1.md (hv4, copied from the pod); and v1's files.
  - from_h100/out/k1420/: hk1_k14.json and hk1_k20.json, the vertices as exact upper triangles (sha256 in the jsons), and the small arrays (sources, applied fields, G at the R_50 rows).
- **Read-only inputs added in v1.1.** The 14- and 20-channel histories' T0011.00 slices under <internal>/slices (md5 checked against md5_done.txt, Table V7), and their run json and fields npz under ladders/r1/runs/. The md5s are in the box record.
- vdress_control_R1_v1.1.json carries the printed status, the zero-cost check, the genuine-history readouts and the sha256 of every result file used.

