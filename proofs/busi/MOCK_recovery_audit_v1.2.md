# MOCK recovery audit v1.2: channel-optimal decoders on the saved n = 3 mock states (#490; v1.1 per #491; v1.2 per #492)

Version 1.2 (sec. 9 lists the v1.1 changes, sec. 11 the v1.2 changes; v1 and v1.1 are kept). Final delivery after the coordinator-run independent second read (sec. 7). Author: busi (analyst engineer), busi inbox #490, following deliveries/lead_gaps/ReviewA_mock1x_v1.md item B1 and ReviewB_mock1x_v1.md item B. Inputs are icloud saved registered states, read in place and read-only; nothing was propagated.

Frozen scope: n = 3; g = 1 and the g = 0 control; clocks t = 1, 2, 4; both converter forms: on-site baseline and filtered control with the endpoint-copy halo: N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1 (declared convention, not a measurement). The registered state is (|0_R>|B> + |1_R>|D>)/sqrt 2 evolved, R an ordinary reference qubit. Output (i) whole-lead light modes (sites 1, 2; light components; the receiver); output (ii) all light modes including the core (transport diagnostic, not an expanded receiver). Every O sector (vacuum, one and two particles) is kept; no postselection, no branch renormalization. Null value of the entanglement fidelity: 1/4.

## 1. Admission (before any division or optimization)

Squared norm N2 of each saved raw state, exact (binary64 entries are dyadic rationals), with certified decimal enclosure; |N2 - 1| against the addendum normalized-state tolerance (1 + eps)^2 - 1 (eps = saved state error of that action) and against the run spec norm-drift threshold 2e-12; the only trace divisor used downstream is N2 (rho_RO = C C^dagger / N2; no block or branch divisor); the code Gram matrix G_rs = 2 <v_r, v_s> / N2 of the two evolved codeword branches and its certified Frobenius distance from I_2 against 2 (2 sqrt 2 eps + 2 eps^2).

| arm | g | t | raw array (md5) | N2 certified enclosure | N2 > 0 | abs(N2 - 1) | addendum tolerance | within | within 2e-12 | Gram distance from I_2 (upper) | Gram tolerance | admitted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| on-site | 0 | 1 | `array_123.npz` (`b8a38a65d0aa05f255830d67c562e0bd`) | [1.0000000000000002441408, 1.0000000000000002441408] | True | 2.441e-16 | 3.826e-10 | True | True | 6.344e-16 | 1.082e-09 | True |
| on-site | 0 | 2 | `array_231.npz` (`d3efbe6a94d9bb2e25609f158978d9dd`) | [1.0000000000000002609867, 1.0000000000000002609867] | True | 2.61e-16 | 5.498e-10 | True | True | 5.398e-16 | 1.555e-09 | True |
| on-site | 0 | 4 | `array_405.npz` (`8e77284ff16d6e5d58b19c6d97bdd11f`) | [1.0000000000000001742687, 1.0000000000000001742687] | True | 1.743e-16 | 8.84e-10 | True | True | 7.097e-16 | 2.5e-09 | True |
| on-site | 1 | 1 | `array_123.npz` (`29d9c0642f243a2e5911b43ec18a00c0`) | [0.9999999999999990447171, 0.9999999999999990447171] | True | 9.553e-16 | 6.418e-10 | True | True | 1.578e-15 | 1.815e-09 | True |
| on-site | 1 | 2 | `array_231.npz` (`44025b771dcc567323b1ae16b34602c1`) | [0.9999999999999991701854, 0.9999999999999991701854] | True | 8.298e-16 | 1.068e-09 | True | True | 1.731e-15 | 3.022e-09 | True |
| on-site | 1 | 4 | `array_405.npz` (`b64daa5e58dbde4642799f9e2116f20f`) | [0.9999999999999991995224, 0.9999999999999991995224] | True | 8.005e-16 | 1.922e-09 | True | True | 1.343e-15 | 5.435e-09 | True |
| filtered (endpoint-copy halo) | 0 | 1 | `array_121.npz` (`b8a38a65d0aa05f255830d67c562e0bd`) | [1.0000000000000002441408, 1.0000000000000002441408] | True | 2.441e-16 | 3.826e-10 | True | True | 6.344e-16 | 1.082e-09 | True |
| filtered (endpoint-copy halo) | 0 | 2 | `array_229.npz` (`d3efbe6a94d9bb2e25609f158978d9dd`) | [1.0000000000000002609867, 1.0000000000000002609867] | True | 2.61e-16 | 5.498e-10 | True | True | 5.398e-16 | 1.555e-09 | True |
| filtered (endpoint-copy halo) | 0 | 4 | `array_403.npz` (`8e77284ff16d6e5d58b19c6d97bdd11f`) | [1.0000000000000001742687, 1.0000000000000001742687] | True | 1.743e-16 | 8.84e-10 | True | True | 7.097e-16 | 2.5e-09 | True |
| filtered (endpoint-copy halo) | 1 | 1 | `array_121.npz` (`7a91f9a0983ed4b406de070872f3891d`) | [1.0000000000000007671597, 1.0000000000000007671597] | True | 7.672e-16 | 3.972e-10 | True | True | 2.235e-15 | 1.123e-09 | True |
| filtered (endpoint-copy halo) | 1 | 2 | `array_229.npz` (`a787dd40e81c0597cab591b71f2d41cb`) | [1.0000000000000007764162, 1.0000000000000007764162] | True | 7.764e-16 | 5.786e-10 | True | True | 2.146e-15 | 1.637e-09 | True |
| filtered (endpoint-copy halo) | 1 | 4 | `array_403.npz` (`5dd4adbced52187eab4c8b7b82925b92`) | [1.0000000000000007622036, 1.0000000000000007622036] | True | 7.622e-16 | 9.415e-10 | True | True | 2.204e-15 | 2.663e-09 | True |

Admission result: every state admitted; optimization proceeded.

## 2. Known-answer tests of the SDP and certification (run first)

| test | certified lower | certified upper | known value | pass |
|---|---|---|---|---|
| a_perfect_copy | 0.999999999793 | 1.0000000002 | 1 | True |
| b_no_information | 0.24999999995 | 0.2500000005 | 0.25 | True |
| c_rotated_copy | 0.999999999799 | 1.0000000002 | 1 | True |
| d_classical | 0.499999999898 | 0.500000000201 | 0.5 | True |
| e_rank_deficient_copy | 0.999999999799 | 1.0000000006 | 1 | True |

## 3. Certified optimal entanglement fidelity per state

Gate on the physical-target bounds (saved-state bounds widened by 2 eps): lower > 1/2 PASS (a one-shot transfer witness only, not recovery, not return exclusion); upper <= 1/2 FAIL; otherwise UNRESOLVED. Rows for the filtered control carry the endpoint-copy halo: N_-1 = N_0 = 1/2, N_n = N_(n-1) = 1 (declared convention, not a measurement).

| arm | g | t | output | certified F* (saved state) [lower, upper] | physical target [lower, upper] | numerical primal / dual | gate | blocks k: O-weight | fixed receiver F_e (busi, sec. 4 of the clock-well file) | saved D_RK (same output, (ii): n/a) |
|---|---|---|---|---|---|---|---|---|---|---|
| on-site | 0 | 1 | i whole-lead light (receiver) | [0.24999999995, 0.2500000037] | [0.249999999567, 0.250000004083] | 0.25 / 0.25 | FAIL | 0: 1; 1: 1.66683e-30; 2: 4.69745e-31 | [0.2499999998, 0.2500000002] | [0.749999998, 0.750000002] |
| on-site | 0 | 1 | ii all light incl. core (diagnostic) | [0.24999999995, 0.2500000079] | [0.249999999567, 0.250000008283] | 0.25 / 0.249999999999 | FAIL (diagnostic only) | 0: 1; 1: 2.39376e-30; 2: 7.62332e-31 | n/a | n/a |
| on-site | 0 | 2 | i whole-lead light (receiver) | [0.24999999995, 0.2500000037] | [0.2499999994, 0.25000000425] | 0.25 / 0.25 | FAIL | 0: 1; 1: 4.96421e-30; 2: 8.74022e-31 | [0.2499999997, 0.2500000003] | [0.7499999978, 0.7500000022] |
| on-site | 0 | 2 | ii all light incl. core (diagnostic) | [0.24999999995, 0.2500000079] | [0.2499999994, 0.25000000845] | 0.25 / 0.249999999999 | FAIL (diagnostic only) | 0: 1; 1: 7.69881e-30; 2: 1.41097e-30 | n/a | n/a |
| on-site | 0 | 4 | i whole-lead light (receiver) | [0.24999999995, 0.2500000037] | [0.249999999066, 0.250000004584] | 0.25 / 0.25 | FAIL | 0: 1; 1: 1.05836e-29; 2: 1.00051e-30 | [0.2499999996, 0.2500000004] | [0.7499999973, 0.7500000027] |
| on-site | 0 | 4 | ii all light incl. core (diagnostic) | [0.24999999995, 0.2500000079] | [0.249999999066, 0.250000008784] | 0.25 / 0.249999999999 | FAIL (diagnostic only) | 0: 1; 1: 1.6655e-29; 2: 2.26475e-30 | n/a | n/a |
| on-site | 1 | 1 | i whole-lead light (receiver) | [0.25586845323, 0.255868456984] | [0.255868452588, 0.255868457626] | 0.255868453281 / 0.255868453282 | FAIL | 0: 0.988263; 1: 0.0115176; 2: 0.000219308 | [0.2500219488, 0.2500219494] | [0.7470396345, 0.7470396392] |
| on-site | 1 | 1 | ii all light incl. core (diagnostic) | [0.342275912612, 0.34227592065] | [0.34227591197, 0.342275921292] | 0.342275912682 / 0.342275912681 | FAIL (diagnostic only) | 0: 0.815448; 1: 1.8772e-30; 2: 0.184552 | n/a | n/a |
| on-site | 1 | 2 | i whole-lead light (receiver) | [0.255874063743, 0.255874067496] | [0.255874062674, 0.255874068564] | 0.255874063794 / 0.255874063794 | FAIL | 0: 0.988252; 1: 0.010078; 2: 0.00167017 | [0.2501373977, 0.2501373988] | [0.7470367783, 0.7470367843] |
| on-site | 1 | 2 | ii all light incl. core (diagnostic) | [0.361478944694, 0.36147895273] | [0.361478943626, 0.361478953798] | 0.361478944768 / 0.361478944767 | FAIL (diagnostic only) | 0: 0.777042; 1: 4.72285e-30; 2: 0.222958 | n/a | n/a |
| on-site | 1 | 4 | i whole-lead light (receiver) | [0.268462553911, 0.268462557667] | [0.26846255199, 0.268462559589] | 0.268462553965 / 0.268462553965 | FAIL | 0: 0.963075; 1: 0.0291136; 2: 0.00781154 | [0.2504434246, 0.2504434266] | [0.7405031943, 0.7405032028] |
| on-site | 1 | 4 | ii all light incl. core (diagnostic) | [0.328218183825, 0.328218191861] | [0.328218181904, 0.328218193783] | 0.328218183893 / 0.328218183892 | FAIL (diagnostic only) | 0: 0.843564; 1: 1.00072e-29; 2: 0.156436 | n/a | n/a |
| filtered (endpoint-copy halo) | 0 | 1 | i whole-lead light (receiver) | [0.24999999995, 0.2500000037] | [0.249999999567, 0.250000004083] | 0.25 / 0.25 | FAIL | 0: 1; 1: 1.66683e-30; 2: 4.69745e-31 | [0.2499999998, 0.2500000002] | [0.749999998, 0.750000002] |
| filtered (endpoint-copy halo) | 0 | 1 | ii all light incl. core (diagnostic) | [0.24999999995, 0.2500000079] | [0.249999999567, 0.250000008283] | 0.25 / 0.249999999999 | FAIL (diagnostic only) | 0: 1; 1: 2.39376e-30; 2: 7.62332e-31 | n/a | n/a |
| filtered (endpoint-copy halo) | 0 | 2 | i whole-lead light (receiver) | [0.24999999995, 0.2500000037] | [0.2499999994, 0.25000000425] | 0.25 / 0.25 | FAIL | 0: 1; 1: 4.96421e-30; 2: 8.74022e-31 | [0.2499999997, 0.2500000003] | [0.7499999978, 0.7500000022] |
| filtered (endpoint-copy halo) | 0 | 2 | ii all light incl. core (diagnostic) | [0.24999999995, 0.2500000079] | [0.2499999994, 0.25000000845] | 0.25 / 0.249999999999 | FAIL (diagnostic only) | 0: 1; 1: 7.69881e-30; 2: 1.41097e-30 | n/a | n/a |
| filtered (endpoint-copy halo) | 0 | 4 | i whole-lead light (receiver) | [0.24999999995, 0.2500000037] | [0.249999999066, 0.250000004584] | 0.25 / 0.25 | FAIL | 0: 1; 1: 1.05836e-29; 2: 1.00051e-30 | [0.2499999996, 0.2500000004] | [0.7499999973, 0.7500000027] |
| filtered (endpoint-copy halo) | 0 | 4 | ii all light incl. core (diagnostic) | [0.24999999995, 0.2500000079] | [0.249999999066, 0.250000008784] | 0.25 / 0.249999999999 | FAIL (diagnostic only) | 0: 1; 1: 1.6655e-29; 2: 2.26475e-30 | n/a | n/a |
| filtered (endpoint-copy halo) | 1 | 1 | i whole-lead light (receiver) | [0.260715660894, 0.260715664667] | [0.260715660497, 0.260715665064] | 0.260715660946 / 0.260715660946 | FAIL | 0: 0.978569; 1: 0.00478239; 2: 0.0166489 | [0.2520760121, 0.2520760125] | [0.7445541545, 0.7445541585] |
| filtered (endpoint-copy halo) | 1 | 1 | ii all light incl. core (diagnostic) | [0.285875817278, 0.285875825262] | [0.285875816881, 0.285875825659] | 0.285875817336 / 0.285875817336 | FAIL (diagnostic only) | 0: 0.928248; 1: 1.50096e-30; 2: 0.0717516 | n/a | n/a |
| filtered (endpoint-copy halo) | 1 | 2 | i whole-lead light (receiver) | [0.261625151808, 0.261625155561] | [0.26162515123, 0.26162515614] | 0.261625151861 / 0.261625151861 | FAIL | 0: 0.97675; 1: 0.0132511; 2: 0.00999925 | [0.2511691992, 0.2511691998] | [0.74408364, 0.7440836445] |
| filtered (endpoint-copy halo) | 1 | 2 | ii all light incl. core (diagnostic) | [0.311620772234, 0.311620780261] | [0.311620771656, 0.311620780839] | 0.311620772299 / 0.311620772298 | FAIL (diagnostic only) | 0: 0.876758; 1: 3.5605e-30; 2: 0.123242 | n/a | n/a |
| filtered (endpoint-copy halo) | 1 | 4 | i whole-lead light (receiver) | [0.286715210946, 0.286715214703] | [0.286715210004, 0.286715215645] | 0.286715211003 / 0.286715211003 | FAIL | 0: 0.92657; 1: 0.055212; 2: 0.0182184 | [0.2520465822, 0.2520465832] | [0.7305499769, 0.7305499823] |
| filtered (endpoint-copy halo) | 1 | 4 | ii all light incl. core (diagnostic) | [0.296511157126, 0.296511165132] | [0.296511156185, 0.296511166074] | 0.296511157189 / 0.296511157188 | FAIL (diagnostic only) | 0: 0.906978; 1: 9.44447e-30; 2: 0.0930223 | n/a | n/a |

## 4. Per-state summary

- on-site, g = 0, t = 1: receiver F* in [0.25, 0.25] -> FAIL; transport diagnostic F* in [0.25, 0.25000001] -> FAIL (diagnostic only).
- on-site, g = 0, t = 2: receiver F* in [0.25, 0.25] -> FAIL; transport diagnostic F* in [0.25, 0.25000001] -> FAIL (diagnostic only).
- on-site, g = 0, t = 4: receiver F* in [0.25, 0.25] -> FAIL; transport diagnostic F* in [0.25, 0.25000001] -> FAIL (diagnostic only).
- on-site, g = 1, t = 1: receiver F* in [0.25586845, 0.25586846] -> FAIL; transport diagnostic F* in [0.34227591, 0.34227592] -> FAIL (diagnostic only).
- on-site, g = 1, t = 2: receiver F* in [0.25587406, 0.25587407] -> FAIL; transport diagnostic F* in [0.36147894, 0.36147895] -> FAIL (diagnostic only).
- on-site, g = 1, t = 4: receiver F* in [0.26846255, 0.26846256] -> FAIL; transport diagnostic F* in [0.32821818, 0.32821819] -> FAIL (diagnostic only).
- filtered (endpoint-copy halo), g = 0, t = 1: receiver F* in [0.25, 0.25] -> FAIL; transport diagnostic F* in [0.25, 0.25000001] -> FAIL (diagnostic only).
- filtered (endpoint-copy halo), g = 0, t = 2: receiver F* in [0.25, 0.25] -> FAIL; transport diagnostic F* in [0.25, 0.25000001] -> FAIL (diagnostic only).
- filtered (endpoint-copy halo), g = 0, t = 4: receiver F* in [0.25, 0.25] -> FAIL; transport diagnostic F* in [0.25, 0.25000001] -> FAIL (diagnostic only).
- filtered (endpoint-copy halo), g = 1, t = 1: receiver F* in [0.26071566, 0.26071567] -> FAIL; transport diagnostic F* in [0.28587582, 0.28587583] -> FAIL (diagnostic only).
- filtered (endpoint-copy halo), g = 1, t = 2: receiver F* in [0.26162515, 0.26162516] -> FAIL; transport diagnostic F* in [0.31162077, 0.31162078] -> FAIL (diagnostic only).
- filtered (endpoint-copy halo), g = 1, t = 4: receiver F* in [0.28671521, 0.28671522] -> FAIL; transport diagnostic F* in [0.29651116, 0.29651117] -> FAIL (diagnostic only).

Overall: the receiver gate is FAIL at every sampled clock, both converter forms and both couplings; the largest certified physical upper bound over all receiver states is 0.2867152156 and over all diagnostic states 0.3614789538. In the reviewer A B1 wording, a FAIL at every clock licenses 'for every decoder' in place of 'declared receiver' in the recovery finding, with this scope only: decoders acting on the whole-lead LIGHT modes, at the sampled clocks t = 1, 2, 4, at n = 3. It is not a statement about decoders acting on the full heavy-plus-light lead output (not tested by this audit), about other clocks, or about larger chains. The empty-output (vacuum, k = 0) sector contributes between [0.24999999995, 0.2500000001] (certified block bounds; within 1e-15 of 1/4 in the second read; exactly 1/4 by the theorem of sec. 9) to F* in every receiver state, which is why the FAILs are clear-cut.

## 5. Cost (all busi runs for this audit, including the two stopped versions)

| item | CPU s | wall | peak RSS KiB |
|---|---|---|---|
| cw18 admission | 0.42 user | 0.61 s | 56480 |
| cw19 (v1; full-space SDP; stopped by busi after 5 of 24 solves to speed up assembly) | 1021.9 (sum of per-solve process CPU) | | not recorded |
| cw21 (v2; same full-space SDP; stopped by busi on its first g = 1 transport-diagnostic block because of memory and the 2 CPU-hour limit) | 4294.24 user + 166.58 sys | 28:49.02 | 62358384 |
| cw24 (v3; SDP on supp(rho_O), certified on the full space; superseded: its ball-arithmetic squares returned NaN on every rank-one block, so those blocks were reported UNCERTIFIED) | 36.08 user + 0.89 sys | 0:12.55 | 185524 |
| cw27 (v4 = cw24 with x*x squares; the reported results) | 36.0345 (process CPU) / 35.22 user + 1.00 sys | 0:12.23 | 187012 |
| total CPU of all versions | 5555.91 | | |

Agreement of the superseded runs with cw27 on every block they certified (numerical primal and dual optima): cw19: 12 certified blocks, max |numerical optimum difference| 2.84e-12; cw21: 18 certified blocks, max |numerical optimum difference| 2.84e-12; cw24: 61 certified blocks, max |numerical optimum difference| 0. All stdout files are kept.

Per-solve CPU (process time, all threads):

| arm | g | t | output | CPU s | wall s |
|---|---|---|---|---|---|
| baseline | 0 | 1 | i_whole_lead_light | 0.48112 | 0.16008 |
| baseline | 0 | 1 | ii_all_light_incl_core | 1.3772 | 0.57822 |
| baseline | 0 | 2 | i_whole_lead_light | 0.32871 | 0.098169 |
| baseline | 0 | 2 | ii_all_light_incl_core | 1.3261 | 0.54738 |
| baseline | 0 | 4 | i_whole_lead_light | 0.28333 | 0.07654 |
| baseline | 0 | 4 | ii_all_light_incl_core | 1.3373 | 0.54594 |
| baseline | 1 | 1 | i_whole_lead_light | 0.27732 | 0.07871 |
| baseline | 1 | 1 | ii_all_light_incl_core | 3.9009 | 1.2186 |
| baseline | 1 | 2 | i_whole_lead_light | 0.29176 | 0.095249 |
| baseline | 1 | 2 | ii_all_light_incl_core | 3.8735 | 1.233 |
| baseline | 1 | 4 | i_whole_lead_light | 0.24927 | 0.083879 |
| baseline | 1 | 4 | ii_all_light_incl_core | 3.8975 | 1.2067 |
| filtered | 0 | 1 | i_whole_lead_light | 0.27016 | 0.079262 |
| filtered | 0 | 1 | ii_all_light_incl_core | 1.3678 | 0.56582 |
| filtered | 0 | 2 | i_whole_lead_light | 0.282 | 0.081053 |
| filtered | 0 | 2 | ii_all_light_incl_core | 1.4025 | 0.59526 |
| filtered | 0 | 4 | i_whole_lead_light | 0.32823 | 0.096609 |
| filtered | 0 | 4 | ii_all_light_incl_core | 1.3766 | 0.57206 |
| filtered | 1 | 1 | i_whole_lead_light | 0.28209 | 0.08242 |
| filtered | 1 | 1 | ii_all_light_incl_core | 4.0847 | 1.2614 |
| filtered | 1 | 2 | i_whole_lead_light | 0.23871 | 0.08435 |
| filtered | 1 | 2 | ii_all_light_incl_core | 3.8982 | 1.2085 |
| filtered | 1 | 4 | i_whole_lead_light | 0.26332 | 0.088173 |
| filtered | 1 | 4 | ii_all_light_incl_core | 3.9327 | 1.2053 |

Box CPU only (compute-host), nice 10, BLAS and Rayon threads 4, no GPU. Packages: Clarabel and python-flint installed in a private venv <internal>/venv_cw (system site packages inherited; the shared qi2 environment untouched).

## 6. For the second reader

Inputs (icloud saved outputs on compute-host, read-only): for each state the registered primary action raw vector (npz, shape 2 x 276, R first) listed with its md5 in sec. 1, and the basis row keys of the same physical job report (row_keys of any `coherent_conversion` row: (register index, occupation bitmask), modes 0-23, mode m = 8x + 4A + c, A = 1 light). Reports: `<internal>/written_report.md` md5 `84db4c2a9fe340d41bac5750db83a8c5`; `<internal>/written_report.md` md5 `3325aebbc1cd4042b89d1ee0f7d9ec22`; `<internal>/written_report.md` md5 `62df5e285a7763662983a0a4eaa9ddcc`; `<internal>/written_report.md` md5 `ae7f8968500fdaab89fb0407ee02bdb6`.

Exact formulation (cw27). For output mode set O (i: {12,13,14,15,20,21,22,23}; ii: {4,...,7} added), write each basis state |m1 m2> (m1 < m2) as |a>_O |t>_K with sign -1 when m1 is outside O and m2 inside (graded reordering, O first). C[(r, a), t] = sign * psi[r, (m1, m2)]; rho_RO = C C^dagger / N2. rho_RO is block diagonal in k = |a| in {0, 1, 2}. For each block: dual SDP  min Tr sigma_k  s.t.  I_2 x sigma_k - rho_k >= 0 (sigma_k Hermitian d_k x d_k); primal SDP  max Tr[rho_k Z_k]  s.t.  Z_k >= 0, Tr_R Z_k = I. F* = (1/2) sum_k (optimum). (Koenig-Renner-Schaffner: this is the optimal entanglement fidelity over all CPTP decoders O -> qubit; block decomposition is exact because rho is block diagonal and pinching preserves dual feasibility and trace.) Null check: rho = I/2 x rho_O gives 1/4. Numerical solve: each block is compressed to Q = orthonormal basis of supp(rho_O^(k)) (left singular vectors of [C_0 | C_1] with singular value > 1e-13 of the largest); compression to the exact support would preserve feasibility and the objective, but the numerical support uses a 1e-13 singular-value cutoff, so the compressed solve is only a numerical candidate; the reported bounds do not rely on the compression because the solution is lifted back (sigma -> Q sigma Q^dagger, Z -> (I x Q) Z (I x Q)^dagger) and certified on the full space (the upper certificate absorbs any discarded component through delta, the lower certificate is exactly PSD on the full space).

Certification. Upper: sigma' = sigma_num + delta I; I x sigma' - rho = B B^dagger + eta I + E with dyadic B, eta = 1e-10; ball arithmetic (python-flint acb, 256 bits) shows ||E||_F < eta, hence I x sigma' - rho >= 0 exactly, and UB = (1/2) sum_k Tr sigma'_k. Lower: Z_k = A A^dagger with dyadic A (exactly PSD); s_k = certified upper bound of lambda_max(Tr_R Z_k) (s I - T = B B^dagger + eta I + E with ||E||_F < eta in ball arithmetic; Gershgorin as fallback); Z_k / s_k has Tr_R <= I (completion to a channel adds a nonnegative term); LB = (1/2) sum_k ||A^dagger C_k||_F^2 / (N2 s_k), evaluated in ball arithmetic. Physical-target bounds widen by 2 eps (normalized-vector distance bound; F is a POVM-element expectation). A second reader can re-derive F* with any SDP solver from the inputs above; the gate needs only the comparison with 1/2.

## 7. Independent second read and repairs

Second read: `<internal>/MOCK_recovery_secondread_v1.md` (md5 `795be5e9d77da4126369712f311ed606`, copy of <internal>/MOCK_recovery_secondread_v1.md), coordinator-run, different agent: all 12 states AGREE; its own certified bounds lie inside these. Repairs, regenerated by cw31_write_audit_v5.py (no hand edits): (1) the 'for every decoder' sentence in sec. 4 now states its scope (whole-lead light modes only, t = 1, 2, 4, n = 3; not decoders on the full heavy-plus-light lead output) and cites the empty-output sector contribution; (2) sec. 6 no longer calls the support compression exact (1e-13 cutoff) and states why the bounds stand (re-certified on the full space).

## 8. Files (box <internal>/recovery, MD5SUMS there; code busi/code/clockwell/)

| file | md5 |
|---|---|
| `<internal>/cw18_admission.json` | `276e9137b45c0323c9fd4f6120a7f366` |
| `<internal>/cw28_selftest_v4.json` | `de585326f2a41ff24b17ad589acf2deb` |
| `<internal>/cw27_sdp.json` | `b4a7134d603477f1818fb40bd467702e` |
| `<internal>/cw19_stdout.txt` | `77fda38bf50b250c7fda198b6312d0bb` |
| `<internal>/cw21_stdout.txt` | `203d1e7af5f8017ae2e2276de5a502ba` |
| `<internal>/cw18_recovery_admission.py` | `84584359e7881dc2eefb70f1994dc796` |
| `<internal>/cw19_recovery_sdp.py` | `1dc8f1d2de6ed59d90ed67293ec59d1f` |
| `<internal>/cw20_sdp_selftest.py` | `9663cd658285e258c638e67a24d074a4` |
| `<internal>/cw21_recovery_sdp_v2.py` | `2bf49494b111fc403b0d723fa68aeb84` |
| `<internal>/cw22_sdp_selftest_v2.py` | `508e497d80efd945e35ac48aae96dbad` |
| `<internal>/cw23_write_audit.py` | `d61a56a80e73282e4df2f54cd563f98a` |
| `<internal>/cw24_recovery_sdp_v3.py` | `f585951dcefb15ae3f5178639241074d` |
| `<internal>/cw25_sdp_selftest_v3.py` | `080cfa4d2664e621ee308c5f6edc6625` |
| `<internal>/cw26_write_audit_v2.py` | `3d84752a6ead10fc0fb53c518f6e5664` |
| `<internal>/cw27_recovery_sdp_v4.py` | `fbe35c66573b49486014b1270ab982a6` |
| `<internal>/cw28_sdp_selftest_v4.py` | `d92b38228c755f5c760f904e0216ada9` |
| `<internal>/cw29_write_audit_v3.py` | `4d4f42fec130b876a9a0cfb2a8e1cfca` |
| `<internal>/cw30_write_audit_v4.py` | `c20f6dee0a38a9996219ff86cc3d0c46` |
| `<internal>/cw31_write_audit_v5.py` | `5f03c8b24379d2d1d2f99e8b3f6c140b` |
| `<internal>/cw39_write_audit_v1_1.py` | `e77cb31b5772cad1b0ee75f03d1f2e96` |
## 9. v1.1 additions (#491; both models accepted v1)

(a) Final v1: `<internal>/MOCK_recovery_audit_v1.md` md5 `236ddfb4e864dccef4dcacfb92c1d741` (kept unchanged). cw27_sdp.json md5 now `b4a7134d603477f1818fb40bd467702e`; md5 recorded in the .wip the second reader checked `b4a7134d603477f1818fb40bd467702e`: unchanged. Only the sec. 4 and sec. 6 wording was regenerated between the .wip and v1.

(b) The g = 0 arrays coincide across the two converter forms (expected: at g = 0 the halo multiplies a zero interaction): t = 1: on-site md5 `b8a38a65d0aa05f255830d67c562e0bd`, filtered md5 `b8a38a65d0aa05f255830d67c562e0bd`, byte-identical; t = 2: on-site md5 `d3efbe6a94d9bb2e25609f158978d9dd`, filtered md5 `d3efbe6a94d9bb2e25609f158978d9dd`, byte-identical; t = 4: on-site md5 `8e77284ff16d6e5d58b19c6d97bdd11f`, filtered md5 `8e77284ff16d6e5d58b19c6d97bdd11f`, byte-identical.

(c) Why the empty-output sector gives 1/4: a symmetry of the vertex, and its consequence at every n. Let q = diag(+1, -1, -1, +1) on the four components of every mode (sigma_z x sigma_z) and r = 1 x sigma_x (c0 <-> c1, c2 <-> c3), extended to Q = dGamma(q) and the unitary R = Gamma(r) on Fock space (both species). (1) beta = sigma_z x 1 and alpha = sigma_x x sigma_x commute with q and with r, and both act site by site, so dGamma(h) commutes with Q and R for every clock profile N, mass and size n. (2) Every converter monomial (on-site; or filtered, any stencil L, kernel G and halo weight, since the fields d_{b,j} mix sites but not components) moves a component pair Q to a pair P with K_PQ nonzero; K is nonzero only where q(P) = +-q(Q), so H conserves Q mod 4. (3) The heavy pairs with |q| = 2 are 03 (q = +2) and 12 (q = -2), and K's columns for 03 and 12 are equal; so on any heavy state with |Q| = 2 the heavy factor of every converter monomial is proportional to H_03 + H_12, which is R-even (r maps 03 to 12). The dark codeword D = (|03> - |12>)/sqrt 2 has |Q| = 2 and RD = -D; H0 preserves both, so the H0-evolved D stays R-odd with |Q| = 2, every converter heavy factor annihilates it (its image is a multiple of the R-even vacuum with an R-odd coefficient, i.e. zero), V exp(-iH0 t) D = 0 for all t, and exp(-iHt) D = exp(-iH0 t) D: D never acquires light, at every clock, n, profile, mass, coupling and form. (4) B = |01>_H has Q = 0, so the B and D branches lie in different Q mod 4 sectors for all t; their components with an empty light output O are orthogonal (Q is additive and the empty O carries Q = 0). Hence for any output O made of light modes only (the receiver (i) or the diagnostic (ii)), with p0 the probability that the B branch has no particle in O: rho_RO's empty-O block is (1/2) diag(p0, 1) (so lambda_max(G0) = 1 exactly, G0 the Gram matrix of the two branches' empty-O components), the k >= 1 blocks are (1/2)|0><0| x X_k (only the B branch), and F* = (1/2)[lambda_max of the empty block + (1/2)(1 - p0)] = 1/4 + (1 - p0)/4 <= 1/2. So the light-only FAIL (upper <= 1/2) holds at every n, every clock, both forms and every g for this code: it is a property of the code (one codeword is dark by a symmetry of the vertex), not of n = 3; the n = 5 light-only receiver is therefore not informative and is not added (reviewer A mock2 (3)).

Script-emitted checks (cw34, cw35, busi's own builder cw32 which reproduces the saved n = 3 Hamiltonians and states, see MOCK_n5_plan_v1.md): K column 03 = column 12: True; K charge rule q(P) = +-q(Q): True; max over n in {2, 3, 5, 7} and both forms of |[Q, H0]|, |[R, H0]| and of the H entries between Q = 0 and Q = 2 mod 4: 0; Krylov test (n = 2, 3, 5, 7; both forms; clock-well and random profiles; two masses): max ||V q_j|| / ||V|| over the Krylov basis of D = 0; the closed form 1/4 + (1 - p0)/4 lies inside all 24 certified intervals of sec. 3: True.

(d) reviewer B's sign repair: the phrase '(-1)^(k(2-k)) = +1' occurs in the second read's sec. 1 (not in this file); (-1)^(k(2-k)) is +1 at k = 0, 2 and -1 at k = 1, and a uniform sign on a block cancels in rho_k = C_k C_k^dagger, so no number changes. The one 'exactly' that rested on a 1e-15 display (sec. 4) now cites the theorem in (c). 

## 10. Files added in v1.1

| file | md5 |
|---|---|
| `<internal>/cw32_mockmodel.py` | `fe5b3830cf1871eb6c8799594c08252b` |
| `<internal>/cw34_dark_codeword.py` | `cdc455f7fb00eea70d9e49e191ca4ece` |
| `<internal>/cw35_dark_symmetry.py` | `921babaa66ac09d0b4529b5109e1973c` |
| `<internal>/cw34_dark.json` | `1ed753512b6a38cbfcdbf90109107aaf` |
| `<internal>/cw35_dark_symmetry.json` | `04e136833b4f7f9b49995c6398515a5a` |
| `<internal>/cw43_write_audit_v1_2.py` | `54e7907a06b35fe5dc74d3a977b5f246` |
| `<internal>/cw42_dark_subspace_v2.py` | `ffad5827425e04909991a2cef29b7659` |
| `<internal>/cw42_dark_subspace.json` | `2d7a7a4c041e5293ae5a441030b063b4` |

## 11. v1.2 additions (#492; both models confirmed the theorem of sec. 9(c))

(a) The converter matrix K, exact integers (rows: light creation pair P, columns: heavy annihilation pair Q, both in the antisymmetric component-pair order 01, 02, 03, 12, 13, 23; the same integers as the frozen adapter and as busi's builder: True):

| P \ Q | 01 | 02 | 03 | 12 | 13 | 23 |
|---|---|---|---|---|---|---|
| 01 | 4 | 0 | 0 | 0 | 0 | -4 |
| 02 | 0 | -8 | 0 | 0 | 0 | 0 |
| 03 | 0 | 0 | -4 | -4 | 0 | 0 |
| 12 | 0 | 0 | -4 | -4 | 0 | 0 |
| 13 | 0 | 0 | 0 | 0 | -8 | 0 |
| 23 | -4 | 0 | 0 | 0 | 0 | 4 |

Pair charges q(P) = q(a) + q(b) with q = (+1, -1, -1, +1): 01: +0, 02: +0, 03: +2, 12: -2, 13: +0, 23: +0. Exact integer arithmetic (Python int, no floating point): column 03 equals column 12 entry by entry: True; every nonzero K_PQ (10 of 36) has q(P) = +-q(Q): True. These two facts are exact properties of the integer matrix; the phrase 'symmetry of the vertex' in sec. 9 is a conclusion drawn from this matrix, not from a separately stated principle.

(b) Hypotheses of the 'every n' statement: (H1) the one-body generator is h = sum over site operators tensored with 1, beta = sigma_z x 1 and alpha = sigma_x x sigma_x only (any real clock profile, mass and size); (H2) the converter is built from this K with component-blind spatial fields (each field d_{b,j} mixes sites with scalar stencil weights, never components) and scalar kernel and halo weights. Both hold for the on-site and filtered forms as implemented; the zero commutators and Krylov leakage at n = 2, 3, 5, 7 (sec. 9) are evidence for the implementation, and the algebra under (H1)-(H2) is the proof.

(c) Scope of R: the proof uses only that R commutes with H0 and that the heavy annihilation factor on the |Q| = 2 sector is R-even. R is not claimed to be a symmetry of the full H. (Measured, not claimed: max |[Gamma(r), H]| at g = 1 is saved n=3 onsite: 0, busi n=3 onsite: 0, busi n=5 onsite: 0, saved n=3 filtered: 0, busi n=3 filtered: 0, busi n=5 filtered: 0.)

(d) N = 2 only. Nothing is claimed for many-body states: for example c0^dag c1^dag c3^dag |0> at one site is not annihilated by H_03 + H_12 (h_3 h_0 removes c0 and c3 and leaves -c1^dag|0>), so no many-body dark remnant follows from this theorem.

(f) The closed form F* = 1/4 + (1 - p0)/4 is an identity for the exact state; the statement that it lies inside all 24 certified intervals of sec. 3 is what covers the computed states.

Wording (reviewer B): '03 and 12' denote the converter's antisymmetric component-pair columns (same-centre field pairs), not every spatially separated two-particle basis state. 'Every n' means this two-particle, initially light-vacuum code under the stated Hamiltonians; it is not a general many-body or arbitrary-intervention no-go. F* = 1/2 (attained only at p0 = 0) would be the perfect transmission of a classical bit, not qubit recovery. p0 is conditional on the normalized B branch; the registered state's empty-output (vacuum) weight is (1 + p0)/2.

Dark-subspace readout (reviewer A A, cw42; W = largest H_H-invariant subspace of ker T, T the heavy-to-light converter block at g = 1): saved n=3 onsite: dim W = 27 (n(2n-1) = 15, heavy sector 66), ||P_W B|| = 9.9e-16, ||P_W B'|| = 1.2e-15, ||P_W D|| = 1.000000000000; busi n=3 onsite: dim W = 27 (n(2n-1) = 15, heavy sector 66), ||P_W B|| = 9.9e-16, ||P_W B'|| = 1.2e-15, ||P_W D|| = 1.000000000000; busi n=5 onsite: dim W = 85 (n(2n-1) = 45, heavy sector 190), ||P_W B|| = 3.6e-15, ||P_W B'|| = 1.8e-15, ||P_W D|| = 1.000000000000; saved n=3 filtered: dim W = 15 (n(2n-1) = 15, heavy sector 66), ||P_W B|| = 4e-15, ||P_W B'|| = 3.1e-15, ||P_W D|| = 1.000000000000; busi n=3 filtered: dim W = 15 (n(2n-1) = 15, heavy sector 66), ||P_W B|| = 3.7e-16, ||P_W B'|| = 5e-15, ||P_W D|| = 1.000000000000; busi n=5 filtered: dim W = 45 (n(2n-1) = 45, heavy sector 190), ||P_W B|| = 2.3e-14, ||P_W B'|| = 1.4e-13, ||P_W D|| = 1.000000000000. For the filtered form W is exactly the symmetry-protected sector; for the on-site form W is larger than it (additional exactly dark directions), with B and B' outside W in both forms.
