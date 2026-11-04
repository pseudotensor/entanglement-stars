# Computation A on the solved history: PAGE_solved_history version 4.1 (the light channel's exterior-cut excess and collar flux on the B2 history to 100 r_s, the sigma/a 8 history beside; the readout Sec. VI prints; version 4 corrected after its correctness reading)

Version 4 stays as issued; this version applies the corrections of the correctness reading (deliveries/lead_gaps/FableAPI_reading_PAGE_v4_v1.md, both parts) that the stored data support, recomputed from the producer's saved arrays, and folds in the kappa 18 sub-step control, now complete to 100 r_s. The section "Changes from v4" lists each correction with its source. Composed by solve/code/page_v4/page_v4_data_v1b.py (the producer; it runs on the box and reads stored files only), page_v4_tables_v1b.py (the tables and the per-pair export) and page_v4_fig_v1b.py (the figure) from the files listed in SOURCES. The generated numerical tables are derived from the producer's stored-data output; the numerical v1f verdict column is copied from the stated post-pass compositions (the busi lane's K = 18 composition W1_B2_composed_v3 from 40 r_s; this lane's reruns of the same post-pass for the rows before 40 r_s and for the sigma/a 8 history at K = 18), and the final tail status, OPEN at every pair, is this composition's reading under the probes and the absence of a completed remainder certificate, reported separately from the automatic verdict. Prose values taken from prior gate or comparison reports (S5's flat-gate numbers; version 3's piecewise-constant run) are identified with their source where they appear. The consistency checks (T9) compare this composition with the post-pass outputs; they are not an independent rerun of the propagation nor a certification of every physical error source.

**Label on every number.** The light channel of the extension at first order (owner decision D16): the massless 4-component Dirac field of the same lattice operator, by which the model is extended for the Page curve and the radiation seen from infinity only, propagated at first order (D16) on the solved history of the heavy field's collapse, its sources not fed back; the history B2 (r_s/a 64, m a 0.2, sigma/a 4, tau 8; engine_collapse_v19's record of the production member continued to T/r_s 100 on 4096 engine sites, md5 417a4018; the formation record clock T/r_s 12.125 at R_t 38.56 a; the trapped episodes 12.125 to 17.25 and 18.375 to 100 r_s; R_t/2gE 0.684 at 40 and 0.685 at 100; E/E0 0.899 at 40 and 0.898 at 100); the sigma/a 8 history (c27 prep4, its record to 40 r_s, md5 ef0ae520; formation 12.75 at R_t 44.53 a; episodes 12.75 to 16.25, 26.625 to 29.625 and 38.875 to 40) beside as a different object, never averaged with it; the interior beyond the hydrodynamic gate H sigma > 0.1 from T/r_s 11.75, the record's own limit (the gate's wording in the owner's O2 form once ruled); the propagation page_history_v6o (the fields interpolated linearly between records in 8 sub-steps, each at its midpoint; Chebyshev tolerance 1e-16; the massless-sector halving; 8192 sites with the vacuum exterior appended beyond the engine's domain and the join checked at 32 and 64 r_s; the cuts 2, 16, 24, 30 and 48 r_s; the collar 16 r_s; R_cut 24 r_s; the readout clocks every 2 r_s from 40 to 100 and the checkpoints every 2 r_s before, with 15 r_s). The exterior-cut excess S_X(C) - S_X(P_-(h)) is the signed excess of the exterior-cut entropy over the held Killing vacuum of the same fields at the same cut and record, summed over kappa = 1 to 18 with the weights 16 kappa: a finite-channel partial sum, a cut's excess, not an entanglement measure of radiation and not a von Neumann entropy (it can be negative, and is, before the formation's light reaches the cut); the raw cut entropy and the held vacuum's print only as regulated columns (T1c), since on the radial-channel reduction their weighted channel sums have no limit in kappa. No selected-archive or collector quantity prints (version 3b; engine spec 47.1): S_col, its spread, the outside weights, the collected counts, S_B, S(M) - S(M'), S_vac(archive), the E_K/E_F/E_cross/E_remove/M_chi split, W_R and the held-field entropies are absent from this version. E_rad^(0)/E, the net signed collar flow of the kappa 1 and 2 channels over E, prints as a labelled diagnostic (T5): 7.557e-6 through 40 r_s and 3.816e-3 through 100, neither permanently escaped energy, charge loss nor a bound on the omitted feedback.

## 0. What this version prints, in plain terms

1. **The channel sum (T1).** At the readout clocks 40 to 100 r_s the 18-channel excess prints at the cuts 2, 16, 24, 30 and 48 r_s. At 100 r_s it reads 9.504, 8.738, 8.454, 8.259 and 5.867 nats; at 40 r_s 5.874 and 0.0919 nats at the 2 and 16 r_s cuts, with the 24, 30 and 48 r_s cuts not formed there (|S| 5e-3 to 8e-3, below the common mode's share K c of 1.2e-2 to 2.3e-2). Two errors print beside every pair: the propagation floor F_meas (the measured impurity's entropy with the truncation's theorem bound), 1.68e-4 to 4.63e-4 nats, at most 1.19e-4 of |S| where |S| exceeds 3 nats, and the exterior common mode's share K c = 18 c (c the largest |16 kappa x_kappa| over kappa 12 to 18 at the pair), 0.04 to 1.22 percent of |S| at 100 r_s and 0.02 to 6.19 percent over the formed pairs from 80 r_s; its reading as the common mode's share within the sum assumes that the common mode below kappa 12 is no larger than at 12 to 18, and it bounds nothing beyond kappa 18. The verdicts apply to |S|: over the 31 readout clocks the 2 and 16 r_s cuts print formed, tail open at all 31; the 24 r_s cut not formed at 4 (40 to 46 r_s); the 30 r_s cut not formed at 7 (40 to 52) and within the floor at 2 (54, 56); the 48 r_s cut not formed at 19 (40 to 70 and 74 to 78), within the floor at 1 (72) and formed from 80 (0.01154 nats at 70, 3.1384 at 80). The automatic v1f rule prints one pair converged (76 r_s, the 24 r_s cut); this composition prints it formed, tail open, with the final tail status OPEN like every other pair (item 2, T2c). The a priori allowance F_apriori prints beside as the bound the impurity was checked against, not as an error: 0.343 to 0.402 nats at 100 r_s, 3.6 to 6.8 percent of the sum. The record-interpolation control (T8) carries the cuts 16 to 48 r_s and kappa 1 to 8 only, so the 2 r_s cut's sums, the largest values among them, print with no record-interpolation spread.
2. **The tail beyond kappa 18 is OPEN at every pair, as an uncertified remainder (T2, T2b, T2c, the figure's panel c).** From kappa 8 to 12 on, the weighted excess terms 16 kappa x_kappa stop falling and sit near the exterior's common mode: at 100 r_s and the 24 r_s cut the terms are -4.5e-4 to -6.8e-4 from kappa 8 to 18 (a spread of a factor 1.40 over kappa 12 to 18) while the formed-energy terms 16 kappa E_kappa fall from 1.7e-4 to 7.0e-5; the spread over kappa 12 to 18 is within a factor 1.12 to 16.6 at the 17 every-10 pairs where |288 x_18| exceeds 5e-4 (within 17 percent at 100/24, 11 percent at 80/30) and a factor 2.75 at 100/2, where the terms are 8e-5 to 2.1e-4, so the terms are near one value where they are large and not uniformly flat to 10 percent. The single-channel probes at kappa 24, 32 and 48 carry a term of the same size: the largest kappa 18 term over the 155 readout pairs is at 50 r_s, the 16 r_s cut, -1.230e-2, where the probes read -9.33e-3, -7.64e-3 and -5.30e-3 (0.43 of the kappa 18 term at 48); at 80 r_s, the 30 r_s cut (the second largest among the every-10 pairs) they are flat: 3.05e-3, 3.25e-3, 3.18e-3, 2.82e-3 (0.92 at 48); at 100 r_s, the 24 r_s cut -4.82e-4, -4.85e-4, -4.51e-4, -2.48e-4. Over the 17 every-10 pairs with |288 x_18| above 5e-4 the per-step ratios r_a = (|512 x_32|/|384 x_24|)^(1/8) run 0.637 to 1.094 (median 0.967) and r_b = (|768 x_48|/|512 x_32|)^(1/16) 0.838 to 1.164 (median 0.971); the four terms share a sign at 14 of them, and where they do the kappa 48 term is 0.026 to 0.924 of the kappa 18 term with the fitted exponent of |16 kappa x_kappa| against kappa -3.71 to -0.09 (over all 24 same-sign pairs of the 35, -3.71 to +1.21; the +1.21 is at 90/2, outside the 17); the terms change sign between kappa 18 and 48 at 3 of the 17 and at 8 of the other 18. The geometric criterion of W1 ruling v2 section 2(b) (the terms fall from kappa 18 through 24, 32 and 48 with one sign and both per-step ratios below 1) is met at 10 of the 17 pairs (40/16, 50/2, 50/16, 60/48, 70/16, 70/24, 70/30, 90/16, 90/30, 100/30); its conditional majorant |288 x_18| q/(1 - q), q the larger ratio, would read 0.64, 0.78, 0.20 and 0.46 percent of the sum at 70/24, 70/30, 90/16 and 100/30, 1.6, 3.9, 4.2, 11.9 and 17.7 percent at 90/30, 50/2, 70/16, 50/16 and 40/16, and 3.6 times the sum at 60/48 (a not-formed pair). This composition applies it nowhere and prints no percentage: the extrapolation is licensed by the ruling on a continuation premise that the sparse samples (four channels, kappa 18 to 48) do not establish for the intervening channels or beyond kappa 48; the medians of the per-step ratios are those of a power law near 1/kappa (0.965 and 0.975 for 16 kappa x_kappa proportional to 1/kappa), whose partial sums grow as ln K; the fitted exponents show no steepening in kappa over the sampled range; and at the same cut the terms rise or change sign at neighbouring readout clocks (80/24, 80/30, 60/24, 100/48). These finite-range samples establish neither divergence nor the absence of eventual geometric decay; no evaluated, proved remainder bound is supplied, and no unconditional percentage accuracy is certified for the completed sum. At 76/24 the v1f rule's clauses hold (q_x 0.964 over kappa 16 to 18, the neighbours 74 and 78 falling, the geometric tail 5.67e-2 and K c 5.69e-2 within 0.01 |S| = 6.59e-2) and the probes at that clock fall with one sign (1.64e-3, 1.21e-3, 9.00e-4 against 2.14e-3 at kappa 18; r_a 0.963, r_b 0.982; the conditional majorant 1.7 percent); its tail status is OPEN for the same reason as every other pair's. The shifted-vacuum comparison of the ruling's run (a) is in T8b: where 288 |x_18| exceeds 5e-4 the vacuum's cut entropy moves by 124 to 1.8e5 times less than x_18 under the one-sub-step field shift (median 420; the signs agree at 10 of the 17), the ruling's branch for the exterior's physics, not the construction's reference lag; and the halved sub-step (item 5) moves the kappa 18 term by at most 9.0 percent of itself where it exceeds 1e-3. The energy majorant over the last three channels (q 0.878 to 0.897 from 40 r_s, the tail at most 1.1e-3 of the partial sum at K = 18) is a within-K statement: the probes' energies 16 kappa E_kappa at 100 r_s are 6.02e-5, 5.30e-5 and 8.65e-5 at kappa 24, 32 and 48 against 7.04e-5 at kappa 18, so the envelope is not continued by them and no energy tail beyond kappa 18 is certified.
3. **The second averaging scale (T3).** On the sigma/a 8 history, a different object whose record ends at 40 r_s, S_18 is 1.9127e-2 nats at the 2 r_s cut at 40 r_s (F_meas 1.77e-4; K c 9.64e-3), formed with an open tail; at the cuts 16 to 48 r_s |S_18| is 1.2497e-3 to 5.6943e-3 nats and every pair is not formed under the common-mode rule (K c 2.29e-3 to 1.01e-2); B2 at the same clock gives 5.87374 and 0.0918789 nats at the cuts 2 and 16 r_s. Its 2 r_s cut is not quiet before 40: -0.675 nats at 22 r_s (K c 0.177), +1.417 at 24 (K c 0.059), 0.077 at 26, -0.192 at 36 (K c 0.082), 0.019 at 40, all formed, tail open by the rule (the figure's dashed green curve). At 40 r_s the s8 weighted energies 16 kappa E_kappa rise from 2.55e-6 at kappa 5 to 1.01e-5 at kappa 16, then 9.11e-6 and 8.70e-6 at 17 and 18 (the energy fall is intact at 14 r_s); no completed channel-tail certificate is supplied for s8 either. The s8 record's R_t (the dense root the record reports) and the propagation row's min(N - |v|) (the applied fields on the chain) follow different conventions and are not used together as a trapping certificate.
4. **The completion spread (T4).** W3's two completions of the member past the hydrodynamic gate arm at the first record failing the gate at or after T/r_s 11.125, restarted from the rung-4 run's stored slice at 11.0 on engine v36g with 2048 engine sites (the sources held inside the gate radius at the gated record's values with the collar charge projected to the record's E; the paired averaging widened to 8 a inside, which runs at a charge 6.4 percent above the record's, E 113.01 against 106.21 at 40 r_s and 114.43 against 109.28 at 14, so its exterior differs at zeroth order), and carry the light channel's kappa 1 and 2 channels to 40 r_s. The held-sources record has a root at 14 r_s and none at the sampled checkpoints 16, 20, 30 and 40 (min(N - |v|) +0.50 at 40); the paired-avg record has a root at 14 and 16 and none at 20, 30 and 40 (+0.06 at 40); B2 keeps its root from 18.375 r_s on. The kappa 1, 2 excess (16 x_1 + 32 x_2) at the 2 r_s cut at 40 r_s reads 5.5330 nats on B2, -0.029302 on the held-sources completion and +0.74290 on the paired-avg completion: the three-history spread, 5.5623 nats, is 94.7 percent of B2's 18-channel excess at that pair; at the 16, 24, 30 and 48 r_s cuts the spreads are 0.0683, 0.00475, 0.00347 and 0.00625 nats, 74.3, 58.3, 67.6 and 118.2 percent of the 18-channel sums there (comparisons of a two-channel spread with B2's 18-channel magnitude, including not-formed pairs; not measured errors of the full sum); at 30 r_s the 2 r_s cut's spread is 4.0 nats against the K = 18 sum 4.36, and at 14 r_s the spreads are 0.2 to 66 percent of the sums. The two completions alone differ by 0.7722 nats at the 2 r_s cut at 40 r_s (0.041, 0.0026, 0.0014 and 0.0063 at the other cuts). The two-channel excitation E_inv at 40 r_s is 1.4378, 0.24327 and 0.14132; the accumulated collar flows 8.0263e-4, -2.2190e-7 and 4.2940e-5. The third completion (the core coupling point-sampled) did not run: its history ended at 12.75 r_s. The spread is measured only for kappa 1 and 2 through 40 r_s and carries the completions' engine and chain (v36g on 2048 sites against engine_collapse_v19 on 4096) and the paired-avg charge with the interior's; no spread exists at the readout clocks 42 to 100 r_s or for kappa 3 to 18, so the numbers there are the record's own continuation's.
5. **The sub-step controls (T1b, T3).** On B2 the kappa 18 channel at 16 sub-steps per record is complete to 100 r_s. Its term's change D16 = 288 [x_18(16) - x_18(8)] is 9.8e-8 to 1.3e-4 nats over the 155 readout pairs; the whole-set proxy 18 |D16| (conditional on the other 17 channels carrying no larger change; not the measured change of the full sum) is below 0.5 percent of the sum at 119 of the 122 formed pairs and 2.0 to 3.5 percent at 40/16, 50/24 and 48/24, and 0.02 to 8.8 percent at the 33 not-formed and within-the-floor pairs, where the sums are small; where the kappa 18 term exceeds 1e-3 (47 pairs) the halved sub-step moves it by at most 9.0 percent of itself, so the construction's one-sub-step reference lag (W1 ruling v2 section 1(b)) is at most that fraction of the common mode there; the kappa 18 formed energy 288 E_18 rises by 7.0 to 9.9 percent under the finer sub-step over 40 to 100 r_s. On the sigma/a 8 history the kappa 14 control gives 18 |D16|/|S_18| at most 0.72 percent over the 15 formed pairs (11 at the 2 r_s cut, 2 at 16, 1 at 24, 1 at 30 r_s) and |D16|/|224 x_14| at most 0.197 there (at 18 r_s, the 2 r_s cut; 0.013 or less at 14, 16, 20, 30 and 40), 0.074 at the not-formed pair 40/16 and 0.055 at 40/48; the kappa 14 formed energy 224 E_14 falls by 12 percent (7.97e-6 to 6.98e-6 at 40 r_s), opposite in sign to B2's rise. Each control is one channel's measured change, not a certified error of the 18-channel sum.
6. **The collar flux (T5, the figure's panel b).** The kappa 1 and 2 channels' flow L at the collar 16 r_s (the collar energy per coordinate interval Delta T at the collar's recordwise stationary-chart clock) against the two-channel continuum greybody comparators at the record's dense root and slope, L_H(kappa_T), and at 1/(8 pi G E), L_H(T_H), not an observed thermal flux, with the kappa 3 guard's change beside (a truncation diagnostic, not a bound on every omitted channel): |L|/L_H(kappa_T) is at most 0.1451 over 14 to 34 r_s where that comparator exists (0.2636 with kappa 3; L negative at 9 of the 12 checkpoints; the comparator is absent at 18 r_s, where the record has no root); the slope ratios are 0.0179, -0.0270 and 1.1985 at 36, 38 and 40 r_s (0.0494, -0.0232 and 3.4105 against the T_H comparator; 0.0332, 0.1296 and 1.4403 with kappa 3), 190.2556, 272.3820 and 61.4497 at 42, 44 and 46 (202, 294 and 67 with kappa 3: a 6 to 9 percent change), 3.07 to 26.15 over 48 to 60 and -0.025 to 19.37 over 62 to 100 (19.4 at 66, 10.1 at 72, 0.08 to 1.3 at 80 to 100 but for -0.025 at 84; negative at 1 of the 30 checkpoints from 42); the comparator L_H(kappa_T) is 1.89 to 3.28 times L_H(T_H) over 42 to 100, so the T_H ratios are that much larger at the same clock. The formation record's light reaches the collar at 35.08 to 37.00 r_s on the record clock (35.57 to 37.49 on the collar's clock) by the fixed-charge flight comparison from 2 and 1.5 x 2gE_f. The net signed flow through 100 r_s is 3.816e-3 of E (0.4047 in lattice units); its half-value crossing is bracketed by the checkpoints 42 and 44 (the linear-interpolation estimate 43.04; the within-interval crossing is not stored). E_inv/E of the two channels is 9.80 percent at 14 r_s (15.79 with kappa 3; 21.67 with every channel to 18), 0.16 to 2.52 percent over 15 to 40 and 0.42 to 1.43 percent over 42 to 100 (K = 18: 0.45 to 1.46; the kappa 3 guard changes it by 1.8 to 6.2 percent of itself from 42 r_s). The join's largest relative deviation in v is 4.4e-6 to 7.1e-6 over 14 to 56 r_s, 5.84e-5 at 58, 4.57e-4 at 60, 8.70e-4 at 64, first above 1e-3 at the 66 r_s checkpoint, 1.54e-2 at 80 and 7.18e-2 at 100. The propagation variant: version 3's piecewise-constant s4 run (page_history_v6b on the r11 record) read 4.0141 and 5.1180 at 36 and 38 r_s where this run reads 0.0179 and -0.0270 (the piecewise-constant propagation forms energy at every record boundary, W1 ruling v2 section 5), and its continuation (page_history_v6i, the first chain, through its return marker at 58) against this run at 42 to 56 r_s differs by -7.6 to +14.6 percent (187.065/190.256, 278.914/272.382, 66.089/61.450, 12.446/12.652, 7.635/8.265, 26.008/26.155, 16.843/16.025, 3.524/3.075; the 14.6 percent at 56 r_s). The accumulated collar flow prints as an emitted energy nowhere beyond this kappa 1, 2 diagnostic: the high channels' stored accumulators carry the exterior's first-order term, the balance discrepancy d = (E_out - flow)/(16 kappa) = W_exc - R (the moving fields' work on the exterior excess minus the held vacuum's rearrangement, not measured separately here), 7.31e-7 to 7.33e-7 per channel over kappa 11 to 18 at 100 r_s, above E_kappa from kappa 13 (d/E_kappa 0.954 at kappa 12, 1.103 at kappa 13), which is not emission; the K = 18 accumulator over E and the per-channel accumulator column are removed from the paper-facing tables (they stay in the audit data); E_out/E_kappa, the exterior share of each channel's formed energy beyond the collar, prints instead: 0.412 at kappa 1, 0.991 at kappa 2, 0.759 to 1.050 at kappa 11 to 18 at 100 r_s (values above 1 are possible: a region may carry negative excess energy).
7. **The D13 protocol's stored observables (T7)** print as numbers with no supply and no darkness statement (section 3): the detector-energy ledger's second term at the cuts 16 and 30 r_s per window with the agreement between the current's crossing and the inventory change printed per window, L_old and L_sea by the 16 back-propagated probes at every probe clock, the windowed changes of the exterior shells' hard and soft energy inventories, the identified source modes' same-packet transfer with their vacuum occupations, the signed collar split per checkpoint, and the first-return estimate (beyond 100 r_s at every cut: 162 r_s at the 48 r_s cut, 228 at the collar); the holds' drain readings are copied from version 3b.
8. **The marks (T6).** The formation's record clock 12.125; T_d 2.31 r_s (the paper's table, the leading term with an unfixed matching constant); the formation's light at the 2, 16, 24, 30 and 48 r_s radii from 2 and 1.5 x 2gE_f: 12.70 to 14.62, 35.08 to 37.00, 45.27 to 47.18, 52.62 to 54.54 and 73.90 to 75.81 r_s on the record clock (the shifted values in T6 use the 16 r_s collar's recordwise offset at the nearest checkpoint and are not clock conversions at the other radii); the 48 r_s cut's sum prints not formed through 70 and at 74 to 78, within the floor at 72 and formed from 80 (0.01154 nats at 70, 3.1384 at 80), the 30 r_s cut's not formed through 52, within the floor at 54 and 56 and formed from 58 (0.599 at 58, 6.756 at 60), the 24 r_s cut's not formed through 46 and formed from 48 (0.0669 at 50, 4.669 at 60), the 16 r_s cut's formed from 40 (0.0919 at 40, 4.500 at 50); the half-collar-flow clock between 42 and 44 (estimate 43.04).
9. **The record-interpolation control (T8).** The piecewise-constant propagation's (page_history_v6n) kappa 1 to 8 sum exceeds the production's by 0.31 to 0.57 nats at the cuts 16 to 48 r_s at 100 r_s (3.6 to 9.7 percent of the production's partial sum there), by 2.10 nats at the 16 r_s cut at 40 r_s and 1.99 at the 24 r_s cut at 50 r_s (just behind the front), its quench entropy; its formed energy exceeds the production's by 0.234 at 100 r_s, rising from kappa 4 (16 kappa (E_ctrl - E_prod) 1.9e-2 at kappa 4 to 7.1e-2 at kappa 8): the quench floor of the record-wise propagation and the reason the production interpolates. The control carries no 2 r_s cut and no channel above 8, so the 2 r_s cut's sums (the largest value 9.504 nats among them) print with no record-interpolation spread, and the kappa 9 to 18 spread is unmeasured at every cut.
10. **The largest value against the levels; no peak on the epoch.** The largest sampled 18-channel excess from 40 to 100 r_s is 9.50449 nats, at 100 r_s and the 2 r_s cut: 2.68986e-4 of A/(4G) = 35334.4 nats and 2.92631e-4 of eta A/a^2 = 32479.4 nats (eta 0.9192, G/a^2 1/4; A at the record's 2gE, not at the trapping root or the cut's radius); at 100 r_s the five cuts read 1.66e-4 to 2.69e-4 of A/(4G). These compare the radial-channel excess with the stated area levels and identify it with no three-dimensional cut entropy. No saturation of the levels is observed or expected on this epoch. The sums are not monotone: at the cuts 2, 16, 24 and 30 r_s the row at 100 r_s is the largest of the 31 readout rows, but at the 48 r_s cut the largest is 82 r_s's (6.405 nats; 100 r_s reads 5.867, after falls of 1.196 between 82 and 84 and 0.698 between 84 and 86); between consecutive readout clocks the sums fall at 6, 9, 9, 11 and 8 of the 30 steps at the cuts 2, 16, 24, 30 and 48 r_s (the largest falls 0.162 at 46 to 48, 1.431 at 44 to 46, 1.413 at 54 to 56, 1.182 at 62 to 64 and 1.196 at 82 to 84), and from 70 r_s on at 2, 3, 4, 5 and 6 of the 15 steps; the 30 r_s cut falls from 6.756 nats at 60 r_s to 4.960 at 70 and rises again (5.911, 7.353, 8.259 at 80, 90, 100). The 30 r_s fall follows the front's passage (the formation's light reaches 30 r_s at 52.62 to 54.54 r_s by the flight comparison; the cut is within the floor at 54 and 56 and formed from 58) at the pair where the record-interpolation control differs most from the production (1.38 nats, 20 percent of the sum at 60/30, T8), and is not read as content returning through the cut: this finite-channel excess establishes no radiation-entropy turnover.

**What is open.** (i) The tail beyond kappa 18: an uncertified remainder at every pair, no percentage; the share K c is its stated error within K and bounds nothing beyond K. (ii) The completion spread is order one at every cut at 40 r_s on kappa 1 and 2 (item 4); the third completion did not run; no spread exists for kappa 3 to 18 or beyond 40 r_s. (iii) The 16-sub-step control exists for one channel per history (kappa 18 on B2, kappa 14 on s8); the other channels' sub-step error is inferred from it, not measured. (iv) The join's largest relative deviation in v is 8.7e-4 at 64 r_s and rises to 7.2e-2 at 100, past 1e-3 at the 66 r_s checkpoint (at most 7.1e-6 through 56, 5.8e-5 at 58, 4.6e-4 at 60): every row from 58 r_s carries that departure, printed beside it in T5 and T7; a distant first return (an estimate) does not certify the growing join error. (v) The heavy field's exterior-cut entropy and the two fields' sum (W8) are not in this version; the accumulated output entropy S_esc is not computed. (vi) The D13 classification (supply or darkness) is not made here; the four observables print as stored and their complete fluence error budget is not demonstrated. (vii) The record-interpolation control exists for kappa 1 to 8 at the cuts 16 to 48 r_s only (no 2 r_s cut; its kappa 9 to 18 run is queued behind the ruling2 jobs). (viii) The shifted-vacuum comparison exists for kappa 18 at the every-10 clocks (T8b), not at every readout pair.

## 1. The readout sentences (S1 to S3, S6 to S8 as the correctness reading replaces them, with this composition's numbers; S4 and S5 the W1 ruling v2's texts, which stand)

S1. "The exterior-cut excess S_X(C) - S_X(P_-(h)) on B2 is reported as the partial sum over kappa 1 to 18 with weights 16 kappa, at the five stated cuts and the readout clocks 40 to 100 r_s. The production uses page_history_v6o, linear field interpolation with 8 sub-steps, Chebyshev tolerance 1e-16, the massless-sector halving and 8192 sites. The measured-impurity and truncation contribution F_meas ranges from 1.68e-4 to 4.63e-4 nats and is at most 1.19e-4 of |S_18| where |S_18| exceeds 3 nats. The quantity c is the largest absolute weighted excess term over kappa 12 to 18. The allowance 18 c ranges from 0.02 to 6.19 percent of |S_18| over the formed pairs from 80 r_s. Its interpretation as a within-sum common-mode allowance assumes that the common-mode component below kappa 12 is no larger; it bounds nothing beyond kappa 18. The high-channel excess contains slowly varying and oscillatory components, but is not uniformly flat to 10 percent (within 17 percent over kappa 12 to 18 at 100 r_s and the 24 r_s cut, 11 percent at 80 r_s and the 30 r_s cut, a factor 2.75 at 100 r_s and the 2 r_s cut). Single-channel probes at kappa 24, 32 and 48 carry a term of the same size: at the two pairs with the largest kappa 18 term it is 0.43 and 0.92 of that term at kappa 48 (the 16 r_s cut at 50 r_s: -1.23e-2, -9.33e-3, -7.64e-3, -5.30e-3; the 30 r_s cut at 80 r_s, flat: 3.05e-3, 3.25e-3, 3.18e-3, 2.82e-3), 0.08 of it or of the opposite sign at the next two (the 24 r_s cut at 70 r_s, the 48 r_s cut at 100 r_s), and over the 14 same-sign pairs of the 17 with |288 x_18| above 5e-4 it falls as a power of kappa with exponent -3.71 to -0.09 and per-step ratios 0.838 to 1.008 whose medians (0.967, 0.971) are a 1/kappa law's, changing sign at 3 of the 17; the geometric criterion of W1 ruling v2 section 2(b) is met at 10 of the 17 pairs and is not applied, its conditional remainder there being 0.20 to 17.7 percent of the sum (0.46 percent at 100 r_s and the 30 r_s cut) on a continuation premise the samples do not establish. The shifted-vacuum comparison of the ruling's run (a) moves the held vacuum's cut entropy by 124 to 1.8e5 times less than x_18 where the term exceeds 5e-4 (median 420), the ruling's branch for an exterior term of the moving fields and not the construction's reference lag, and the halved sub-step moves the kappa 18 term by at most 9 percent of itself where it exceeds 1e-3. The last-three-channel energy ratios fall (q 0.878 to 0.897 from 40 r_s), but the higher-channel energy probes (6.02e-5, 5.30e-5, 8.65e-5 at kappa 24, 32, 48 against 7.04e-5 at kappa 18 at 100 r_s) do not continue that geometric envelope. Both the completed excess sum and its unconditional percentage accuracy remain uncertified: the tail beyond kappa 18 is OPEN at every pair, as an uncertified remainder, and no pair prints a percentage. The a priori allowance at 100 r_s is 0.343 to 0.402 nats, 3.6 to 6.8 percent of the partial excess, not a bound on all error sources."

S2. "The verdicts are applied to the magnitude of the signed excess. A pair is not formed when |S_18| < 18 c. Otherwise it is within the floor when 0.01 |S_18| < F_meas. Over the 155 readout pairs, 30 are not formed: four at the 24 r_s cut from 40 to 46 r_s, seven at the 30 r_s cut from 40 to 52 r_s, and nineteen at the 48 r_s cut from 40 to 70 r_s and from 74 to 78 r_s. Three are within the floor: 54/30, 56/30 and 72/48. The other 122 pairs are formed and have an uncertified tail beyond kappa 18; the 16 r_s cut at 50 r_s has the front at the cut (18 c 6.3 percent of the sum). The automatic v1f test labels 76/24 converged (its clauses hold there, and the probes at 76 r_s fall with one sign: 1.64e-3, 1.21e-3, 9.00e-4 against 2.14e-3 at kappa 18), but no complete tail certificate is supplied there, so its final tail status is OPEN like every other pair's. These verdicts are resolution labels for the finite-channel excess, not certificates of radiation arrival."

S3. "The high-channel accumulated collar flow is not reported as emitted energy. The labelled kappa 1, 2 flux and net accumulator remain D13 diagnostics (S8). For a fixed collar, the balance discrepancy d = Delta E_out - F_collar obeys d = W_exc - R, where W_exc is the moving fields' work on the exterior excess and R is the held vacuum's rearrangement. The stored values give d 7.31e-7 to 7.33e-7 per channel over kappa 11 to 18 at 100 r_s. It exceeds E_kappa from kappa 13, not kappa 12 (d/E_kappa 0.954 at kappa 12, 1.103 at kappa 13). The two terms W_exc and R are not separately measured in this delivery. The ratio E_out/E_kappa ranges from 0.759 to 1.050 over kappa 11 to 18; values above one are possible because a region can carry negative excess energy."

S4 (stands; the W1 ruling v2's text). "The piecewise-constant control (page_history_v6n, kappa 1 to 8) rises from kappa 4 (16 kappa E_kappa 2.3e-2 to 7.1e-2 at 100 r_s against the production's 3.9e-3 to 1.7e-4), the quench floor of the record-wise propagation and the reason the production interpolates; its excess at 100 r_s exceeds the production's by 0.31 to 0.57 nats at the cuts 16 to 48 r_s, its extra quench entropy, and its kappa sum has no majorant at K = 8; the control for kappa 9 to 18 is pending." (Computed here: 0.313 to 0.571 at 100 r_s; 2.10 and 1.99 nats behind the front at 40/16 and 50/24; the control carries no 2 r_s cut, so the 2 r_s cut's sums, the largest value 9.504 nats among them, print with no record-interpolation spread.)

S5 (stands; the W1 ruling v2's text, copied from the busi lane's composition W1_B2_composed_v3 and its flat gate: this composition makes no flat-vacuum check of its own). "The flat held vacuum's kappa 1 cut entropy on the 8192-site chain agrees with icloud X3's at 2, 16, 24 and 48 r_s to 4.6e-6 nats (sector route; the dense route within 1.8e-9 of it)", followed by the RELABEL text as printed: the smoke test's S_X values 2.74643, 2.88147, 2.95571, 3.11088 nats at 16, 24, 30 and 48 r_s are the flat held vacuum's raw kappa 1 exterior-cut entropy on the 8192-site chain; they are not an escaping entropy and print nowhere as one. (The flat gate's values 2.054355, 2.746430, 2.881472, 2.955715 and 3.110877 nats at 2, 16, 24, 30 and 48 r_s against X3's comparators 2.05436, 2.74643, 2.88147 and 3.11088 at 2, 16, 24 and 48 r_s, differences at most 4.6e-6; the 30 r_s value has no X3 comparator and prints for the relabel; the dense route within 1.8e-9 of the sector route.)

S6 (the scale beside). "On the sigma/a 8 history, a different object whose record ends at 40 r_s, S_18 is 1.9127e-2 nats at the 2 r_s cut at 40 r_s, with F_meas = 1.77e-4 and 18 c = 9.64e-3. It is formed with an open tail. At cuts 16 to 48 r_s, |S_18| ranges from 1.2497e-3 to 5.6943e-3 nats and every pair is not formed under the common-mode rule. B2 at the same clock gives 5.87374 and 0.0918789 nats at cuts 2 and 16 r_s. Before 40 r_s the s8 2 r_s cut reads -0.675 nats at 22 r_s and +1.417 at 24 (18 c 0.177 and 0.059), 0.077 at 26 and -0.192 at 36, formed, tail open by the rule. At 40 r_s, the s8 weighted energies rise from 2.55e-6 at kappa 5 to 1.01e-5 at kappa 16, then fall to 9.11e-6 and 8.70e-6 at 17 and 18. No completed channel-tail certificate is supplied. The kappa 14 sub-step control gives the scaled diagnostic 18 |D16|/|S_18| of at most 0.72 percent over the 15 formed pairs (11 at cut 2, two at cut 16, and one each at cuts 24 and 30 r_s). This scaled one-channel diagnostic is not a measured change or a certified error bound for the full 18-channel sum."

S7 (the completion spread). "Every history-dependent number beyond the hydrodynamic gate is conditional on the chosen interior continuation, and the completion spread exists only to 40 r_s and only for the kappa 1 and 2 channels: on W3's two completions of the member past the gate (the sources held inside the gate radius with the collar charge projected to the record's E; the paired averaging widened inside, which runs at a charge 6.4 percent above the record's, E 113.0 against 106.2 at 40 r_s, so its exterior differs at zeroth order), both without a trapping root at the sampled checkpoints from 16 and 20 r_s on where B2 keeps its root from 18.375, the kappa 1, 2 excess at 40 r_s and the 2 r_s cut is 5.5330 nats on B2, -0.029302 on the held-sources completion and +0.74290 on the paired-averaging completion: their three-history spread is 5.5623 nats, 94.7 percent of B2's 18-channel excess, and at the cuts 16, 24, 30 and 48 r_s the spread ratios are 74.3, 58.3, 67.6 and 118.2 percent (a two-channel spread against B2's 18-channel magnitude, including pairs labelled not formed, not measured errors of the full 18-channel sum). At 40 r_s the two-channel excitation energies are 1.4378, 0.24327 and 0.14132, and the net collar flows 8.0263e-4, -2.2190e-7 and 4.2940e-5. B2 uses engine v19 on 4096 sites, the completions v36g on 2048 sites, and the paired-averaging charge differs, so the spread includes those changes. The third completion did not run, and no completion spread is supplied for channels 3 to 18 or clocks beyond 40 r_s, so the numbers there are the record's own continuation's."

S8 (the flux, the D13 protocol's labelled diagnostic). "The kappa 1, 2 collar flux on B2 is reported per coordinate interval Delta T, with its recordwise stationary-chart clock label. The comparison curves are continuum greybody fluxes for the same channel set at the record's dense surface slope and at T_H = 1/(8 pi G E), not measurements of a thermal spectrum. Over 14 to 34 r_s, |L|/L_H of the slope comparator is at most 0.1451 where that comparator exists, or 0.2636 with kappa 3 included; L is negative at nine of the twelve checkpoints. The slope ratios are 0.0179, -0.0270 and 1.1985 at 36, 38 and 40 r_s, then 190.2556, 272.3820 and 61.4497 at 42, 44 and 46 r_s. They range from 3.07 to 26.15 over 48 to 60 r_s and from -0.025 to 19.37 over 62 to 100 r_s. The formation-light mark at the collar, 35.08 to 37.00 r_s on the record clock, is a fixed-charge flight comparison. The net signed flow through 100 r_s is 3.816e-3 of E. Its half-value crossing is bracketed by 42 and 44 r_s, with a linear-interpolation estimate of 43.04 r_s. E_inv/E is 9.80 percent at 14 r_s, 0.16 to 2.52 percent over 15 to 40 r_s, and 0.42 to 1.43 percent over 42 to 100 r_s. The join's maximum relative deviation in v is 8.70e-4 at 64 r_s, 1.34e-3 at 70 r_s and 7.18e-2 at 100 r_s. These finite-channel diagnostics are neither a charge loss nor a feedback bound; the kappa 3 change is a truncation diagnostic, not a proved bound on every omitted channel."

## 2. The tables

Every table below is page_v4_tables_v1b.md as the table script wrote it (its md5 in SOURCES); the per-pair export page_v4_readout_export_v1b.tsv beside it holds one row per readout clock and cut.

### T1. The exterior-cut excess of the B2 history summed over kappa = 1 to 18 (weights 16 kappa), nats, at the readout clocks and the cuts; its errors beside

S = sum_k 16 k x_k (x_k the channel's excess S_X(C) - S_X(P_-(h)) at the cut); F_meas the propagation floor (the measured impurity's entropy with the truncation's theorem bound, weighted and summed); F_apriori the a priori allowance (the bound the impurity was checked against, not an error); c the largest |16 k x_k| over kappa 12 to 18 at the pair; K c = 18 c the exterior common mode's share within K (its reading as the common mode's share assumes the common mode below kappa 12 is no larger than at 12 to 18; it bounds nothing beyond kappa 18). The verdict column is the post-pass v1f's automatic verdict (not formed if |S| < K c; within the floor if 0.01 |S| < F_meas; converged if the last three terms fall at the clock and its neighbours, are resolved, the geometric tail and K c are within 0.01 |S|; else formed, tail open), with the one automatic 'converged' (76 r_s, the 24 r_s cut) printed as formed, tail open: its final tail status is OPEN like every other pair's (T2c). The tail beyond kappa 18 is OPEN at every pair: no evaluated remainder bound is supplied (T2). Rows at 14 to 30 r_s are before the readout clocks (the post-pass rerun from 14 r_s); rows from 40 r_s are the busi lane's K = 18 composition (W1_B2_composed_v3).

| T/r_s | cut r_s | S (nats) | F_meas | F_meas/\|S\| | F_apriori | c | K c | K c/\|S\| (percent) | q_x (last 3) | verdict (v1f; the final tail status is OPEN) | reason | tail beyond kappa 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 2 | -6.48147e-02 | 6.59e-05 | 1.0e-03 | 0.053 | 1.62e-03 | 2.92e-02 | 45.05 | 16.058 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |
| 14 | 16 | -1.09009e-01 | 7.12e-05 | 6.5e-04 | 0.059 | 3.51e-03 | 6.31e-02 | 57.92 | 1.324 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |
| 14 | 24 | -1.45980e-01 | 7.23e-05 | 5.0e-04 | 0.060 | 9.38e-03 | 1.69e-01 | 115.62 | 0.765 | not formed | - | OPEN |
| 14 | 30 | -1.48426e-01 | 7.30e-05 | 4.9e-04 | 0.060 | 1.04e-02 | 1.87e-01 | 125.74 | 0.891 | not formed | - | OPEN |
| 14 | 48 | -1.19656e-01 | 7.45e-05 | 6.2e-04 | 0.061 | 8.96e-03 | 1.61e-01 | 134.72 | 0.984 | not formed | - | OPEN |
| 16 | 2 | 3.66961e-02 | 7.26e-05 | 2.0e-03 | 0.060 | 1.06e-03 | 1.92e-02 | 52.22 | 0.417 | formed, tail open | the terms do not fall at a neighbouring clock (15: q 1.091, 18: q 2.071); the common mode's share n c above 0.01 of the sum | OPEN |
| 16 | 16 | -4.54226e-02 | 7.85e-05 | 1.7e-03 | 0.066 | 3.50e-03 | 6.31e-02 | 138.87 | 0.737 | not formed | - | OPEN |
| 16 | 24 | -3.70019e-02 | 7.97e-05 | 2.2e-03 | 0.068 | 5.61e-03 | 1.01e-01 | 272.78 | 0.414 | not formed | - | OPEN |
| 16 | 30 | -4.81573e-02 | 8.04e-05 | 1.7e-03 | 0.068 | 5.05e-03 | 9.09e-02 | 188.83 | 0.912 | not formed | - | OPEN |
| 16 | 48 | -1.01099e-01 | 8.21e-05 | 8.1e-04 | 0.070 | 5.84e-03 | 1.05e-01 | 103.96 | 3.749 | not formed | - | OPEN |
| 20 | 2 | 7.19888e-01 | 9.04e-05 | 1.3e-04 | 0.075 | 6.06e-02 | 1.09e+00 | 151.58 | 0.956 | not formed | - | OPEN |
| 20 | 16 | -2.04313e-02 | 9.77e-05 | 4.8e-03 | 0.082 | 2.21e-03 | 3.98e-02 | 194.73 | 3.085 | not formed | - | OPEN |
| 20 | 24 | -2.35676e-02 | 9.92e-05 | 4.2e-03 | 0.084 | 2.44e-03 | 4.40e-02 | 186.51 | 1.871 | not formed | - | OPEN |
| 20 | 30 | -2.98964e-02 | 1.00e-04 | 3.3e-03 | 0.085 | 3.93e-03 | 7.07e-02 | 236.54 | 2.579 | not formed | - | OPEN |
| 20 | 48 | -2.57130e-02 | 1.02e-04 | 4.0e-03 | 0.086 | 4.85e-03 | 8.73e-02 | 339.36 | 7.098 | not formed | - | OPEN |
| 30 | 2 | 4.36124e+00 | 1.24e-04 | 2.8e-05 | 0.109 | 2.56e-03 | 4.60e-02 | 1.05 | 1.002 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |
| 30 | 16 | 5.94211e-03 | 1.34e-04 | 2.3e-02 | 0.121 | 1.04e-03 | 1.88e-02 | 316.51 | 0.350 | not formed | - | OPEN |
| 30 | 24 | -2.34927e-03 | 1.36e-04 | 5.8e-02 | 0.123 | 7.16e-04 | 1.29e-02 | 548.48 | 1.621 | not formed | - | OPEN |
| 30 | 30 | -7.66651e-03 | 1.37e-04 | 1.8e-02 | 0.125 | 1.62e-03 | 2.91e-02 | 379.60 | 4.205 | not formed | - | OPEN |
| 30 | 48 | -1.65523e-02 | 1.40e-04 | 8.4e-03 | 0.127 | 1.91e-03 | 3.43e-02 | 207.38 | 1.213 | not formed | - | OPEN |
| 40 | 2 | 5.87374e+00 | 1.68e-04 | 2.9e-05 | 0.144 | 4.07e-04 | 7.33e-03 | 0.12 | 1.327 | formed, tail open | the last 3 terms do not fall | OPEN |
| 40 | 16 | 9.18789e-02 | 1.82e-04 | 2.0e-03 | 0.159 | 2.25e-03 | 4.04e-02 | 44.01 | 1.648 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |
| 40 | 24 | 8.14153e-03 | 1.85e-04 | 2.3e-02 | 0.162 | 8.28e-04 | 1.49e-02 | 183.01 | 1.913 | not formed | - | OPEN |
| 40 | 30 | 5.12573e-03 | 1.86e-04 | 3.6e-02 | 0.164 | 6.84e-04 | 1.23e-02 | 240.22 | 0.947 | not formed | - | OPEN |
| 40 | 48 | -5.28734e-03 | 1.89e-04 | 3.6e-02 | 0.167 | 1.26e-03 | 2.27e-02 | 429.07 | 5.137 | not formed | - | OPEN |
| 50 | 2 | 6.17869e+00 | 2.08e-04 | 3.4e-05 | 0.178 | 1.35e-03 | 2.43e-02 | 0.39 | 1.001 | formed, tail open | the last 3 terms do not fall | OPEN |
| 50 | 16 | 4.49983e+00 | 2.25e-04 | 5.0e-05 | 0.197 | 1.57e-02 | 2.82e-01 | 6.27 | 0.962 | formed, tail open | the terms do not fall at a neighbouring clock (52: q 1.052); the common mode's share n c above 0.01 of the sum | OPEN |
| 50 | 24 | 6.68715e-02 | 2.29e-04 | 3.4e-03 | 0.201 | 2.30e-03 | 4.14e-02 | 61.85 | 1.262 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |
| 50 | 30 | 9.64806e-03 | 2.30e-04 | 2.4e-02 | 0.203 | 6.37e-04 | 1.15e-02 | 118.87 | 1.287 | not formed | - | OPEN |
| 50 | 48 | 1.90803e-03 | 2.35e-04 | 1.2e-01 | 0.207 | 1.07e-03 | 1.93e-02 | 1010.93 | 3.576 | not formed | - | OPEN |
| 60 | 2 | 8.19063e+00 | 2.49e-04 | 3.0e-05 | 0.211 | 4.79e-04 | 8.63e-03 | 0.11 | 1.108 | formed, tail open | the last 3 terms do not fall | OPEN |
| 60 | 16 | 5.66813e+00 | 2.70e-04 | 4.8e-05 | 0.234 | 1.56e-03 | 2.81e-02 | 0.50 | 1.024 | formed, tail open | the last 3 terms do not fall | OPEN |
| 60 | 24 | 4.66868e+00 | 2.74e-04 | 5.9e-05 | 0.239 | 3.26e-03 | 5.87e-02 | 1.26 | 0.955 | formed, tail open | the terms do not fall at a neighbouring clock (58: q 3.541); the common mode's share n c above 0.01 of the sum | OPEN |
| 60 | 30 | 6.75600e+00 | 2.76e-04 | 4.1e-05 | 0.241 | 1.28e-03 | 2.31e-02 | 0.34 | 0.849 | formed, tail open | the terms do not fall at a neighbouring clock (58: q 1.043, 62: q 2.861) | OPEN |
| 60 | 48 | 1.01968e-02 | 2.81e-04 | 2.8e-02 | 0.247 | 8.37e-04 | 1.51e-02 | 147.74 | 5.181 | not formed | - | OPEN |
| 70 | 2 | 8.41699e+00 | 2.89e-04 | 3.4e-05 | 0.244 | 3.28e-04 | 5.90e-03 | 0.07 | 0.811 | formed, tail open | the terms do not fall at a neighbouring clock (68: q 1.116, 72: q 1.540) | OPEN |
| 70 | 16 | 6.18005e+00 | 3.13e-04 | 5.1e-05 | 0.271 | 5.78e-04 | 1.04e-02 | 0.17 | 1.363 | formed, tail open | the last 3 terms do not fall | OPEN |
| 70 | 24 | 5.73254e+00 | 3.17e-04 | 5.5e-05 | 0.277 | 4.52e-03 | 8.14e-02 | 1.42 | 0.942 | formed, tail open | the terms do not fall at a neighbouring clock (68: q 1.085); the common mode's share n c above 0.01 of the sum | OPEN |
| 70 | 30 | 4.96032e+00 | 3.20e-04 | 6.5e-05 | 0.280 | 3.06e-03 | 5.50e-02 | 1.11 | 1.027 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |
| 70 | 48 | 1.15436e-02 | 3.26e-04 | 2.8e-02 | 0.286 | 7.57e-04 | 1.36e-02 | 117.98 | 3.045 | not formed | - | OPEN |
| 80 | 2 | 8.70687e+00 | 3.30e-04 | 3.8e-05 | 0.277 | 4.25e-04 | 7.64e-03 | 0.09 | 2.162 | formed, tail open | the last 3 terms do not fall | OPEN |
| 80 | 16 | 8.14319e+00 | 3.57e-04 | 4.4e-05 | 0.308 | 1.55e-03 | 2.78e-02 | 0.34 | 0.981 | formed, tail open | the terms do not fall at a neighbouring clock (78: q 1.552) | OPEN |
| 80 | 24 | 6.20763e+00 | 3.63e-04 | 5.8e-05 | 0.314 | 1.81e-04 | 3.26e-03 | 0.05 | 1.673 | formed, tail open | the last 3 terms do not fall | OPEN |
| 80 | 30 | 5.91070e+00 | 3.66e-04 | 6.2e-05 | 0.318 | 3.05e-03 | 5.49e-02 | 0.93 | 1.062 | formed, tail open | the last 3 terms do not fall | OPEN |
| 80 | 48 | 3.13840e+00 | 3.72e-04 | 1.2e-04 | 0.325 | 9.44e-04 | 1.70e-02 | 0.54 | 2.081 | formed, tail open | the last 3 terms do not fall | OPEN |
| 90 | 2 | 9.08845e+00 | 3.70e-04 | 4.1e-05 | 0.310 | 2.77e-04 | 4.99e-03 | 0.05 | 1.612 | formed, tail open | the last 3 terms do not fall | OPEN |
| 90 | 16 | 8.45786e+00 | 4.00e-04 | 4.7e-05 | 0.345 | 5.84e-04 | 1.05e-02 | 0.12 | 1.113 | formed, tail open | the last 3 terms do not fall | OPEN |
| 90 | 24 | 8.11655e+00 | 4.06e-04 | 5.0e-05 | 0.352 | 1.63e-04 | 2.94e-03 | 0.04 | 2.535 | formed, tail open | the last 3 terms do not fall; kappa 16 not resolved against F_meas | OPEN |
| 90 | 30 | 7.35338e+00 | 4.10e-04 | 5.6e-05 | 0.355 | 2.90e-03 | 5.22e-02 | 0.71 | 1.055 | formed, tail open | the last 3 terms do not fall | OPEN |
| 90 | 48 | 4.94953e+00 | 4.17e-04 | 8.4e-05 | 0.363 | 6.36e-03 | 1.14e-01 | 2.31 | 0.772 | formed, tail open | the terms do not fall at a neighbouring clock (92: q 1.556); the common mode's share n c above 0.01 of the sum | OPEN |
| 100 | 2 | 9.50449e+00 | 4.11e-04 | 4.3e-05 | 0.343 | 2.10e-04 | 3.78e-03 | 0.04 | 0.752 | formed, tail open | the terms do not fall at a neighbouring clock (98: q 1.137) | OPEN |
| 100 | 16 | 8.73770e+00 | 4.44e-04 | 5.1e-05 | 0.381 | 3.42e-04 | 6.15e-03 | 0.07 | 2.263 | formed, tail open | the last 3 terms do not fall | OPEN |
| 100 | 24 | 8.45397e+00 | 4.51e-04 | 5.3e-05 | 0.389 | 6.76e-04 | 1.22e-02 | 0.14 | 1.216 | formed, tail open | the last 3 terms do not fall | OPEN |
| 100 | 30 | 8.25867e+00 | 4.55e-04 | 5.5e-05 | 0.393 | 1.26e-03 | 2.26e-02 | 0.27 | 1.007 | formed, tail open | the last 3 terms do not fall | OPEN |
| 100 | 48 | 5.86743e+00 | 4.63e-04 | 7.9e-05 | 0.402 | 3.99e-03 | 7.19e-02 | 1.22 | 2.568 | formed, tail open | the last 3 terms do not fall; the common mode's share n c above 0.01 of the sum | OPEN |

Tally over the 31 readout clocks 40 to 100 r_s (not formed / within the floor / formed, tail open): 2 r_s: 0 / 0 / 31; 16 r_s: 0 / 0 / 31; 24 r_s: 4 / 0 / 27; 30 r_s: 7 / 2 / 22; 48 r_s: 19 / 1 / 11. The automatic v1f tally has the 24 r_s cut as 4 / 0 / 1 / 26 (one pair, 76/24, converged by the v1f rule alone; T2c prints the probes at that pair).
F_meas over the readout pairs: 1.68e-04 to 4.63e-04 nats; F_meas/|S| where |S| exceeds 3 nats: at most 1.19e-04 (118 pairs). K c/|S| over the formed pairs from 80 r_s: 0.02 to 6.19 percent; at 100 r_s: 0.04 to 1.22 percent. F_apriori at 100 r_s: 0.343 to 0.402 nats (3.6 to 6.8 percent of the sum; the allowance, not the error).
The sums against the levels (the rows' levels block: A/(4G) = 4 pi (2gE)^2 / (4 G) at the record's instantaneous 2gE with G/a^2 = 1/4, and eta A/a^2 with eta = 0.9192 (the settings' eta), in nats): the largest |S| from 40 r_s is 9.50449 nats at T/r_s 100, the cut 2 r_s (the last clock's value, not a peak: the 30 r_s cut falls from 6.75600 at 60 r_s to 4.96032 at 70 and rises again), which is 2.68986e-04 of A/(4G) (35334.4 nats) and 2.92631e-04 of eta A/a^2 (32479.4 nats) at that record's 2gE (A evaluated at the record's 2gE, not at the trapping root or the cut's radius); at 100 r_s the five cuts read 2 r_s: 2.69e-04, 16 r_s: 2.47e-04, 24 r_s: 2.39e-04, 30 r_s: 2.34e-04, 48 r_s: 1.66e-04 of A/(4G). These compare the radial-channel excess with the stated area levels; they identify it with no three-dimensional cut entropy.
The spread of the high channels' terms |16 kappa x_kappa| over kappa 12 to 18 (largest over smallest) at the 35 every-10 readout pairs: 1.12 to 16.57 at the 17 pairs where |288 x_18| exceeds 5e-4 (within 17 percent at 100/24, 11 percent at 80/30), 1.40 to 23.11 at the other 18 (a factor 2.75 at 100/2, where the terms are 8e-5 to 2.1e-4): the terms are near one value where they are large and not uniformly flat to 10 percent.
The largest kappa 18 term over the 155 readout pairs is at 50 r_s, the 16 r_s cut: 288 x_18 = -1.230e-02 (the probes there -9.329e-03, -7.638e-03, -5.303e-03 at kappa 24, 32, 48); the second largest among the every-10 pairs is 80/30 (+3.050e-03).
The record-interpolation control (T8) carries the cuts 16 to 48 r_s and kappa 1 to 8 only: the 2 r_s cut's sums, the largest value among them, print with no record-interpolation spread.

### T1c. The regulated columns behind T1 (computed here from the channel files): the raw exterior-cut entropy and the held vacuum's, summed over kappa 1 to 18 with the weights 16 kappa (nats; the raw cut entropy of the radial truncation has no limit in kappa and prints only as a regulated column; S = raw - vacuum)

| T/r_s | cut r_s | raw sum_k 16 k S_X | held vacuum sum_k 16 k S_X(P_-(h)) | S (their difference) |
|---|---|---|---|---|
| 14 | 2 | 3467.37035 | 3467.43516 | -6.48147e-02 |
| 14 | 16 | 5377.52948 | 5377.63849 | -1.09009e-01 |
| 14 | 24 | 5747.31243 | 5747.45841 | -1.45980e-01 |
| 14 | 30 | 5950.80466 | 5950.95309 | -1.48426e-01 |
| 14 | 48 | 6379.40571 | 6379.52537 | -1.19656e-01 |
| 20 | 2 | 3468.48906 | 3467.76917 | 7.19888e-01 |
| 20 | 16 | 5377.61350 | 5377.63393 | -2.04313e-02 |
| 20 | 24 | 5747.43202 | 5747.45559 | -2.35676e-02 |
| 20 | 30 | 5950.92099 | 5950.95089 | -2.98964e-02 |
| 20 | 48 | 6379.49833 | 6379.52404 | -2.57130e-02 |
| 30 | 2 | 3471.28831 | 3466.92707 | 4.36124e+00 |
| 30 | 16 | 5377.63818 | 5377.63224 | 5.94211e-03 |
| 30 | 24 | 5747.45165 | 5747.45400 | -2.34927e-03 |
| 30 | 30 | 5950.94190 | 5950.94956 | -7.66651e-03 |
| 30 | 48 | 6379.50663 | 6379.52318 | -1.65523e-02 |
| 40 | 2 | 3472.73683 | 3466.86309 | 5.87374e+00 |
| 40 | 16 | 5377.73170 | 5377.63982 | 9.18789e-02 |
| 40 | 24 | 5747.46325 | 5747.45511 | 8.14153e-03 |
| 40 | 30 | 5950.95479 | 5950.94966 | 5.12573e-03 |
| 40 | 48 | 6379.51764 | 6379.52293 | -5.28734e-03 |
| 50 | 2 | 3472.99659 | 3466.81789 | 6.17869e+00 |
| 50 | 16 | 5382.39590 | 5377.89607 | 4.49983e+00 |
| 50 | 24 | 5747.52568 | 5747.45881 | 6.68715e-02 |
| 50 | 30 | 5950.96231 | 5950.95266 | 9.64806e-03 |
| 50 | 48 | 6379.52508 | 6379.52317 | 1.90803e-03 |
| 60 | 2 | 3475.07272 | 3466.88209 | 8.19063e+00 |
| 60 | 16 | 5383.26772 | 5377.59959 | 5.66813e+00 |
| 60 | 24 | 5752.22694 | 5747.55825 | 4.66868e+00 |
| 60 | 30 | 5957.71447 | 5950.95847 | 6.75600e+00 |
| 60 | 48 | 6379.53429 | 6379.52409 | 1.01968e-02 |
| 70 | 2 | 3475.26127 | 3466.84428 | 8.41699e+00 |
| 70 | 16 | 5383.76516 | 5377.58511 | 6.18005e+00 |
| 70 | 24 | 5753.08450 | 5747.35196 | 5.73254e+00 |
| 70 | 30 | 5956.00474 | 5951.04442 | 4.96032e+00 |
| 70 | 48 | 6379.53789 | 6379.52635 | 1.15436e-02 |
| 80 | 2 | 3475.53088 | 3466.82401 | 8.70687e+00 |
| 80 | 16 | 5385.69366 | 5377.55047 | 8.14319e+00 |
| 80 | 24 | 5753.62481 | 5747.41717 | 6.20763e+00 |
| 80 | 30 | 5956.82162 | 5950.91092 | 5.91070e+00 |
| 80 | 48 | 6382.67747 | 6379.53907 | 3.13840e+00 |
| 90 | 2 | 3475.93761 | 3466.84915 | 9.08845e+00 |
| 90 | 16 | 5386.03644 | 5377.57858 | 8.45786e+00 |
| 90 | 24 | 5755.54146 | 5747.42492 | 8.11655e+00 |
| 90 | 30 | 5958.22132 | 5950.86794 | 7.35338e+00 |
| 90 | 48 | 6384.49005 | 6379.54052 | 4.94953e+00 |
| 100 | 2 | 3476.35096 | 3466.84648 | 9.50449e+00 |
| 100 | 16 | 5386.30677 | 5377.56907 | 8.73770e+00 |
| 100 | 24 | 5755.88265 | 5747.42867 | 8.45397e+00 |
| 100 | 30 | 5959.20289 | 5950.94422 | 8.25867e+00 |
| 100 | 48 | 6385.47815 | 6379.61072 | 5.86743e+00 |

### T1b. The other error sources per pair (nats): the 16-sub-step control (kappa 18), the record-interpolation control (kappa 1 to 8, the piecewise-constant propagation), the completion spread (kappa 1 and 2, to 40 r_s)

D16 = 288 [x_18(16 sub-steps) - x_18(8 sub-steps)] is the measured change of one weighted channel term under sub-step halving; 18 |D16| is a whole-retained-set proxy, conditional on the other 17 channels carrying no larger change, not the measured change of the full sum (a sum restricted to kappa 12 to 18 would contain seven such terms). The record-interpolation column is the production's kappa 1 to 8 partial sum minus the piecewise-constant control's (page_history_v6n) at the same pair (no control exists at the 2 r_s cut); the completion column is the spread (largest minus smallest) of the kappa 1 and 2 weighted excess over the three histories of T4 (B2 and W3's two completions), present only to 40 r_s.

| T/r_s | cut r_s | S (K = 18) | 288 x_18 (8 sub) | 288 x_18 (16 sub) | D16 | 18 \|D16\|/\|S\| (percent) | prod - ctrl, kappa 1 to 8 | ctrl - prod over \|S\| (percent) | completion spread (kappa 1, 2) | spread over \|S\| (percent) |
|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 2 | -6.48147e-02 | 1.591e-03 | 1.640e-03 | 4.96e-05 | 1.378 | - | - | 4.300e-02 | 66.3 |
| 14 | 16 | -1.09009e-01 | 3.507e-03 | 3.508e-03 | 6.59e-07 | 0.011 | -8.311e-04 | 0.76 | 4.386e-03 | 4.0 |
| 14 | 24 | -1.45980e-01 | -2.590e-03 | -2.589e-03 | 3.02e-07 | 0.004 | -1.887e-03 | 1.29 | 1.994e-03 | 1.4 |
| 14 | 30 | -1.48426e-01 | -5.919e-03 | -5.918e-03 | 2.58e-07 | 0.003 | -1.897e-03 | 1.28 | 1.184e-03 | 0.8 |
| 14 | 48 | -1.19656e-01 | -8.521e-03 | -8.520e-03 | 5.13e-07 | 0.008 | -1.397e-03 | 1.17 | 2.724e-04 | 0.2 |
| 16 | 2 | 3.66961e-02 | 7.822e-05 | 1.292e-04 | 5.09e-05 | 2.499 | - | - | 6.937e-02 | 189.1 |
| 16 | 16 | -4.54226e-02 | -1.258e-03 | -1.257e-03 | 1.34e-07 | 0.005 | 1.792e-03 | -3.95 | 1.409e-02 | 31.0 |
| 16 | 24 | -3.70019e-02 | -3.032e-04 | -3.030e-04 | 2.31e-07 | 0.011 | 1.146e-03 | -3.10 | 8.757e-03 | 23.7 |
| 16 | 30 | -4.81573e-02 | 3.820e-03 | 3.821e-03 | 4.59e-07 | 0.017 | 4.396e-04 | -0.91 | 6.450e-03 | 13.4 |
| 16 | 48 | -1.01099e-01 | 1.316e-03 | 1.317e-03 | 6.48e-07 | 0.012 | -4.637e-04 | 0.46 | 3.115e-03 | 3.1 |
| 20 | 2 | 7.19888e-01 | -4.213e-02 | -4.260e-02 | -4.72e-04 | 1.180 | - | - | 1.661e+00 | 230.7 |
| 20 | 16 | -2.04313e-02 | 1.333e-04 | 1.332e-04 | -1.37e-07 | 0.012 | -1.499e-04 | 0.73 | 1.346e-02 | 65.9 |
| 20 | 24 | -2.35676e-02 | -7.370e-04 | -7.372e-04 | -1.99e-07 | 0.015 | -1.612e-04 | 0.68 | 1.385e-02 | 58.8 |
| 20 | 30 | -2.98964e-02 | 1.263e-03 | 1.263e-03 | -1.44e-07 | 0.009 | 3.537e-04 | -1.18 | 1.208e-02 | 40.4 |
| 20 | 48 | -2.57130e-02 | -2.093e-03 | -2.093e-03 | -2.92e-07 | 0.020 | 4.578e-04 | -1.78 | 7.483e-03 | 29.1 |
| 30 | 2 | 4.36124e+00 | -1.775e-03 | -1.801e-03 | -2.57e-05 | 0.011 | - | - | 4.015e+00 | 92.1 |
| 30 | 16 | 5.94211e-03 | -4.903e-05 | -4.890e-05 | 1.28e-07 | 0.039 | -4.319e-04 | 7.27 | 2.356e-03 | 39.7 |
| 30 | 24 | -2.34927e-03 | 3.595e-04 | 3.594e-04 | -2.92e-08 | 0.022 | 6.571e-05 | -2.80 | 2.329e-03 | 99.1 |
| 30 | 30 | -7.66651e-03 | 6.303e-04 | 6.302e-04 | -4.21e-08 | 0.010 | 1.009e-04 | -1.32 | 7.522e-03 | 98.1 |
| 30 | 48 | -1.65523e-02 | -1.107e-03 | -1.107e-03 | -1.03e-07 | 0.011 | -5.513e-05 | 0.33 | 1.122e-02 | 67.8 |
| 40 | 2 | 5.87374e+00 | 3.810e-04 | 3.843e-04 | 3.24e-06 | 0.001 | - | - | 5.562e+00 | 94.7 |
| 40 | 16 | 9.18789e-02 | 1.154e-03 | 1.257e-03 | 1.03e-04 | 2.015 | -2.104e+00 | 2289.81 | 6.828e-02 | 74.3 |
| 40 | 24 | 8.14153e-03 | 2.862e-04 | 2.864e-04 | 1.95e-07 | 0.043 | -4.937e-04 | 6.06 | 4.750e-03 | 58.3 |
| 40 | 30 | 5.12573e-03 | 1.722e-04 | 1.723e-04 | 1.40e-07 | 0.049 | -1.774e-05 | 0.35 | 3.465e-03 | 67.6 |
| 40 | 48 | -5.28734e-03 | 7.418e-04 | 7.420e-04 | 1.96e-07 | 0.067 | 4.248e-05 | -0.80 | 6.250e-03 | 118.2 |
| 50 | 2 | 6.17869e+00 | 1.040e-03 | 1.052e-03 | 1.18e-05 | 0.003 | - | - | - | - |
| 50 | 16 | 4.49983e+00 | -1.230e-02 | -1.227e-02 | 2.56e-05 | 0.010 | -4.495e-01 | 9.99 | - | - |
| 50 | 24 | 6.68715e-02 | 8.119e-04 | 8.905e-04 | 7.86e-05 | 2.115 | -1.990e+00 | 2975.77 | - | - |
| 50 | 30 | 9.64806e-03 | -6.762e-05 | -6.609e-05 | 1.54e-06 | 0.287 | -1.658e-03 | 17.19 | - | - |
| 50 | 48 | 1.90803e-03 | -7.713e-05 | -7.684e-05 | 2.88e-07 | 0.272 | 1.166e-05 | -0.61 | - | - |
| 60 | 2 | 8.19063e+00 | -4.207e-04 | -4.273e-04 | -6.69e-06 | 0.001 | - | - | - | - |
| 60 | 16 | 5.66813e+00 | -1.420e-03 | -1.409e-03 | 1.09e-05 | 0.003 | -4.024e-01 | 7.10 | - | - |
| 60 | 24 | 4.66868e+00 | -1.664e-03 | -1.628e-03 | 3.59e-05 | 0.014 | -5.100e-01 | 10.92 | - | - |
| 60 | 30 | 6.75600e+00 | 4.114e-04 | 4.763e-04 | 6.49e-05 | 0.017 | -1.381e+00 | 20.45 | - | - |
| 60 | 48 | 1.01968e-02 | 5.726e-04 | 5.731e-04 | 5.10e-07 | 0.090 | -5.484e-05 | 0.54 | - | - |
| 70 | 2 | 8.41699e+00 | 1.868e-04 | 1.883e-04 | 1.57e-06 | 0.000 | - | - | - | - |
| 70 | 16 | 6.18005e+00 | -5.687e-04 | -5.638e-04 | 4.86e-06 | 0.001 | -3.742e-01 | 6.05 | - | - |
| 70 | 24 | 5.73254e+00 | 2.820e-03 | 2.843e-03 | 2.37e-05 | 0.007 | -3.574e-01 | 6.23 | - | - |
| 70 | 30 | 4.96032e+00 | -1.291e-03 | -1.246e-03 | 4.53e-05 | 0.016 | -5.071e-01 | 10.22 | - | - |
| 70 | 48 | 1.15436e-02 | -1.818e-04 | -1.807e-04 | 1.06e-06 | 0.166 | -9.161e-04 | 7.94 | - | - |
| 80 | 2 | 8.70687e+00 | 4.247e-04 | 4.544e-04 | 2.96e-05 | 0.006 | - | - | - | - |
| 80 | 16 | 8.14319e+00 | 1.282e-03 | 1.285e-03 | 3.19e-06 | 0.001 | -3.163e-01 | 3.88 | - | - |
| 80 | 24 | 6.20763e+00 | -9.564e-05 | -8.150e-05 | 1.41e-05 | 0.004 | -3.600e-01 | 5.80 | - | - |
| 80 | 30 | 5.91070e+00 | 3.050e-03 | 3.083e-03 | 3.27e-05 | 0.010 | -3.825e-01 | 6.47 | - | - |
| 80 | 48 | 3.13840e+00 | -1.803e-04 | -1.019e-04 | 7.84e-05 | 0.045 | -1.198e+00 | 38.18 | - | - |
| 90 | 2 | 9.08845e+00 | 7.884e-05 | 8.989e-05 | 1.10e-05 | 0.002 | - | - | - | - |
| 90 | 16 | 8.45786e+00 | -5.833e-04 | -5.747e-04 | 8.63e-06 | 0.002 | -3.116e-01 | 3.68 | - | - |
| 90 | 24 | 8.11655e+00 | 1.625e-04 | 1.807e-04 | 1.82e-05 | 0.004 | -3.238e-01 | 3.99 | - | - |
| 90 | 30 | 7.35338e+00 | 2.119e-03 | 2.146e-03 | 2.72e-05 | 0.007 | -3.385e-01 | 4.60 | - | - |
| 90 | 48 | 4.94953e+00 | -2.751e-04 | -1.987e-04 | 7.64e-05 | 0.028 | -6.272e-01 | 12.67 | - | - |
| 100 | 2 | 9.50449e+00 | -7.656e-05 | -6.971e-05 | 6.84e-06 | 0.001 | - | - | - | - |
| 100 | 16 | 8.73770e+00 | 2.878e-04 | 3.120e-04 | 2.42e-05 | 0.005 | -3.138e-01 | 3.59 | - | - |
| 100 | 24 | 8.45397e+00 | -4.821e-04 | -4.628e-04 | 1.94e-05 | 0.004 | -3.125e-01 | 3.70 | - | - |
| 100 | 30 | 8.25867e+00 | -1.027e-03 | -1.007e-03 | 2.00e-05 | 0.004 | -3.251e-01 | 3.94 | - | - |
| 100 | 48 | 5.86743e+00 | -2.405e-03 | -2.351e-03 | 5.39e-05 | 0.017 | -5.713e-01 | 9.74 | - | - |

The 16-sub-step control of kappa 18 (B2): complete to 100 r_s (ruling2 run (c): --interp-sub 16, kappa 18 only, --lean; wall 3.87 h). Over the 155 readout pairs |D16| is 9.8e-08 to 1.3e-04 nats; as 18 |D16| it is below 0.5 percent of the sum at 119 of the 122 formed pairs and 2.0 to 3.5 percent at the rest (48/24 3.49, 50/24 2.11, 40/16 2.02), and 0.02 to 8.8 percent at the 33 not-formed and within-the-floor pairs, where the sums are small. Where the kappa 18 term exceeds 1e-3 (47 pairs) the halved sub-step moves it by 0.090 of itself at most (|D16|/|288 x_18|), so the construction's one-sub-step reference lag (W1 ruling v2 section 1(b)) is at most that fraction of the common mode there. The kappa 18 formed energy 288 E_18 under 16 sub-steps is 7.0 to 9.9 percent above its 8-sub-step value over 40 to 100 r_s (a rise, opposite in sign to the s8 kappa 14 control's 12 percent fall).

### T2. The channel sum's convergence in kappa at the readout pairs: the partial sums, the last three terms, the probes kappa 24, 32, 48, the geometric criterion and its conditional majorant, and the tail

Partial sums S_K = sum_{k <= K} 16 k x_k. The probes are single-channel runs of the same production line (--kappas 24-24, 32-32, 48-48, --no-kappa-extend, --lean) on the same record, final at every one of the 52 checkpoints (every 2 r_s with 15); their terms 16 K x_K are listed with the kappa 18 term. r_a = (|512 x_32|/|384 x_24|)^(1/8) and r_b = (|768 x_48|/|512 x_32|)^(1/16) are per-step ratios of the weighted terms (ratios of the unweighted x differ by (4/3)^(1/8) and (3/2)^(1/16)); 'criterion' says whether the four terms fall from kappa 18 through 24, 32 and 48 with one sign and both per-step ratios below 1 (W1 ruling v2 section 2(b)); where it holds, the conditional majorant |288 x_18| q/(1 - q), q the larger ratio, is printed as the remainder that extrapolation would give, with its percent of |S|: it is a conditional estimate on a continuation premise, never applied here, and no pair prints a percentage. The slope is the least-squares exponent of |16 kappa x_kappa| against kappa over the four channels where they share a sign (a finite-range fit, not an asymptotic law). The tail beyond kappa 18 is OPEN at every pair: no evaluated, proved remainder bound is supplied.

| T/r_s | cut r_s | S_2 | S_8 | S_12 | S_18 | 256 x_16 | 272 x_17 | 288 x_18 | 384 x_24 | 512 x_32 | 768 x_48 | r_a | r_b | same sign | criterion | conditional majorant (percent of \|S\|) | slope | c | tail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 2 | 5.5330e+00 | 5.8697e+00 | 5.8717e+00 | 5.87374e+00 | +3.39e-04 | +2.87e-04 | +3.81e-04 | +1.56e-04 | +1.93e-04 | +4.84e-04 | 1.026 | 1.059 | yes | not met | - | 0.38 | 4.07e-04 | OPEN |
| 40 | 16 | 5.3114e-02 | 7.4399e-02 | 8.4877e-02 | 9.18789e-02 | +1.14e-03 | +7.00e-04 | +1.15e-03 | +8.69e-04 | +5.03e-04 | +2.95e-05 | 0.934 | 0.838 | yes | met | 1.63e-02 (17.73) | -3.71 | 2.25e-03 | OPEN |
| 40 | 24 | 4.4640e-03 | 7.5819e-03 | 7.3318e-03 | 8.14153e-03 | +1.37e-04 | -2.62e-04 | +2.86e-04 | -2.59e-04 | +2.65e-06 | -6.41e-05 | 0.564 | 1.220 | no | not met | - | - | 8.28e-04 | OPEN |
| 40 | 30 | 8.1395e-03 | 5.1230e-03 | 4.8448e-03 | 5.12573e-03 | -4.28e-04 | +4.05e-04 | +1.72e-04 | +2.47e-04 | -1.97e-04 | -8.88e-05 | 0.972 | 0.951 | no | not met | - | - | 6.84e-04 | OPEN |
| 40 | 48 | -1.0222e-02 | -4.7461e-03 | -6.3973e-03 | -5.28734e-03 | -9.14e-04 | -1.44e-04 | +7.42e-04 | +4.67e-04 | -2.52e-04 | -1.64e-04 | 0.926 | 0.973 | no | not met | - | - | 1.26e-03 | OPEN |
| 50 | 2 | 5.8959e+00 | 6.1659e+00 | 6.1721e+00 | 6.17869e+00 | +1.14e-03 | +1.04e-03 | +1.04e-03 | +8.73e-04 | +6.22e-04 | +5.80e-04 | 0.959 | 0.996 | yes | met | 2.38e-01 (3.86) | -0.63 | 1.35e-03 | OPEN |
| 50 | 16 | 4.2927e+00 | 4.6497e+00 | 4.5816e+00 | 4.49983e+00 | -1.34e-02 | -1.29e-02 | -1.23e-02 | -9.33e-03 | -7.64e-03 | -5.30e-03 | 0.975 | 0.977 | yes | met | 5.33e-01 (11.85) | -0.84 | 1.57e-02 | OPEN |
| 50 | 24 | 3.0932e-02 | 4.7897e-02 | 5.9086e-02 | 6.68715e-02 | +1.18e-03 | +6.43e-04 | +8.12e-04 | +3.50e-04 | -9.49e-06 | +1.07e-04 | 0.637 | 1.164 | no | not met | - | - | 2.30e-03 | OPEN |
| 50 | 30 | 2.4164e-03 | 7.3795e-03 | 8.6217e-03 | 9.64806e-03 | -3.00e-04 | +3.87e-04 | -6.76e-05 | +1.34e-05 | -5.89e-06 | -1.99e-05 | 0.902 | 1.079 | no | not met | - | - | 6.37e-04 | OPEN |
| 50 | 48 | 2.6158e-03 | -5.6500e-04 | 5.7212e-04 | 1.90803e-03 | +2.09e-04 | +7.49e-04 | -7.71e-05 | +2.68e-04 | +2.06e-04 | +1.19e-04 | 0.968 | 0.966 | no | not met | - | - | 1.07e-03 | OPEN |
| 60 | 2 | 7.9423e+00 | 8.1946e+00 | 8.1927e+00 | 8.19063e+00 | -3.49e-04 | -3.87e-04 | -4.21e-04 | -1.61e-04 | -1.05e-04 | -3.90e-05 | 0.948 | 0.940 | yes | met | 7.71e-03 (0.09) | -2.34 | 4.79e-04 | OPEN |
| 60 | 16 | 5.3548e+00 | 5.6823e+00 | 5.6770e+00 | 5.66813e+00 | -1.46e-03 | -1.50e-03 | -1.42e-03 | -1.48e-03 | -1.10e-03 | -6.82e-04 | 0.963 | 0.971 | yes | not met | - | -0.80 | 1.56e-03 | OPEN |
| 60 | 24 | 4.2724e+00 | 4.6970e+00 | 4.6812e+00 | 4.66868e+00 | -1.90e-03 | -1.82e-03 | -1.66e-03 | -7.08e-04 | -2.21e-04 | -2.57e-04 | 0.865 | 1.009 | yes | not met | - | -2.00 | 3.26e-03 | OPEN |
| 60 | 30 | 6.1615e+00 | 6.7440e+00 | 6.7513e+00 | 6.75600e+00 | +7.77e-04 | +6.59e-04 | +4.11e-04 | +4.29e-04 | +3.78e-04 | +1.21e-04 | 0.984 | 0.932 | yes | not met | - | -1.24 | 1.28e-03 | OPEN |
| 60 | 48 | 9.5621e-03 | 7.9604e-03 | 9.3130e-03 | 1.01968e-02 | +5.05e-05 | -2.62e-04 | +5.73e-04 | +2.48e-04 | +2.19e-04 | +2.26e-05 | 0.985 | 0.868 | yes | met | 3.68e-02 (361.24) | -3.13 | 8.37e-04 | OPEN |
| 70 | 2 | 8.1429e+00 | 8.4141e+00 | 8.4155e+00 | 8.41699e+00 | +3.16e-04 | +2.56e-04 | +1.87e-04 | +2.13e-04 | +1.69e-04 | +1.46e-04 | 0.971 | 0.991 | yes | not met | - | -0.31 | 3.28e-04 | OPEN |
| 70 | 16 | 5.9286e+00 | 6.1847e+00 | 6.1831e+00 | 6.18005e+00 | -4.42e-04 | -4.17e-04 | -5.69e-04 | -5.57e-04 | -5.48e-04 | -3.36e-04 | 0.998 | 0.970 | yes | met | 2.59e-01 (4.19) | -0.52 | 5.78e-04 | OPEN |
| 70 | 24 | 5.3309e+00 | 5.6915e+00 | 5.7114e+00 | 5.73254e+00 | +3.35e-03 | +3.15e-03 | +2.82e-03 | +1.76e-03 | +9.68e-04 | +2.14e-04 | 0.928 | 0.910 | yes | met | 3.65e-02 (0.64) | -2.63 | 4.52e-03 | OPEN |
| 70 | 30 | 4.5809e+00 | 4.9863e+00 | 4.9716e+00 | 4.96032e+00 | -1.64e-03 | -1.68e-03 | -1.29e-03 | -6.62e-04 | -4.07e-04 | -2.40e-04 | 0.941 | 0.968 | yes | met | 3.86e-02 (0.78) | -1.69 | 3.06e-03 | OPEN |
| 70 | 48 | 3.3986e-03 | 8.7089e-03 | 1.0252e-02 | 1.15436e-02 | -1.78e-04 | +5.43e-04 | -1.82e-04 | -2.31e-05 | +2.25e-05 | +8.90e-06 | 0.997 | 0.944 | no | not met | - | - | 7.57e-04 | OPEN |
| 80 | 2 | 8.4239e+00 | 8.7034e+00 | 8.7050e+00 | 8.70687e+00 | +1.74e-04 | +1.96e-04 | +4.25e-04 | +2.01e-04 | +1.88e-04 | +2.42e-04 | 0.992 | 1.016 | yes | not met | - | -0.49 | 4.25e-04 | OPEN |
| 80 | 16 | 7.8704e+00 | 8.1283e+00 | 8.1346e+00 | 8.14319e+00 | +1.49e-03 | +1.31e-03 | +1.28e-03 | +1.29e-03 | +1.17e-03 | +9.73e-04 | 0.988 | 0.988 | yes | not met | - | -0.29 | 1.55e-03 | OPEN |
| 80 | 24 | 5.9423e+00 | 6.2073e+00 | 6.2079e+00 | 6.20763e+00 | -1.08e-04 | -1.81e-04 | -9.56e-05 | +7.29e-06 | +5.30e-05 | +2.18e-04 | 1.281 | 1.092 | no | not met | - | - | 1.81e-04 | OPEN |
| 80 | 30 | 5.5496e+00 | 5.8851e+00 | 5.8937e+00 | 5.91070e+00 | +2.79e-03 | +2.96e-03 | +3.05e-03 | +3.25e-03 | +3.18e-03 | +2.82e-03 | 0.997 | 0.993 | yes | not met | - | -0.09 | 3.05e-03 | OPEN |
| 80 | 48 | 2.8097e+00 | 3.1321e+00 | 3.1367e+00 | 3.13840e+00 | -7.23e-05 | +8.66e-05 | -1.80e-04 | -4.33e-04 | -4.62e-04 | -3.88e-04 | 1.008 | 0.989 | yes | not met | - | 0.67 | 9.44e-04 | OPEN |
| 90 | 2 | 8.7860e+00 | 9.0873e+00 | 9.0881e+00 | 9.08845e+00 | +1.72e-04 | +2.77e-04 | +7.88e-05 | +9.28e-05 | +8.33e-05 | +2.79e-04 | 0.987 | 1.079 | yes | not met | - | 1.21 | 2.77e-04 | OPEN |
| 90 | 16 | 8.1972e+00 | 8.4629e+00 | 8.4610e+00 | 8.45786e+00 | -5.25e-04 | -5.84e-04 | -5.83e-04 | -4.75e-04 | -3.63e-04 | -1.67e-04 | 0.967 | 0.953 | yes | met | 1.70e-02 (0.20) | -1.27 | 5.84e-04 | OPEN |
| 90 | 24 | 7.8686e+00 | 8.1166e+00 | 8.1161e+00 | 8.11655e+00 | +2.84e-05 | +6.41e-05 | +1.62e-04 | +1.37e-04 | +1.61e-04 | +1.13e-04 | 1.020 | 0.978 | yes | not met | - | -0.30 | 1.63e-04 | OPEN |
| 90 | 30 | 7.0459e+00 | 7.3263e+00 | 7.3392e+00 | 7.35338e+00 | +2.21e-03 | +2.01e-03 | +2.12e-03 | +1.66e-03 | +1.41e-03 | +1.06e-03 | 0.980 | 0.982 | yes | met | 1.16e-01 (1.58) | -0.70 | 2.90e-03 | OPEN |
| 90 | 48 | 4.4277e+00 | 4.9436e+00 | 4.9510e+00 | 4.94953e+00 | -3.18e-03 | -2.45e-03 | -2.75e-04 | -9.67e-04 | +3.35e-04 | +1.13e-03 | 0.876 | 1.079 | no | not met | - | - | 6.36e-03 | OPEN |
| 100 | 2 | 9.1784e+00 | 9.5061e+00 | 9.5054e+00 | 9.50449e+00 | -1.57e-04 | -1.18e-04 | -7.66e-05 | -2.65e-05 | +9.35e-06 | +2.94e-04 | 0.878 | 1.241 | no | not met | - | - | 2.10e-04 | OPEN |
| 100 | 16 | 8.4544e+00 | 8.7356e+00 | 8.7366e+00 | 8.73770e+00 | +1.69e-04 | +1.27e-04 | +2.88e-04 | +7.02e-05 | +3.59e-05 | +1.68e-04 | 0.920 | 1.101 | yes | not met | - | -0.52 | 3.42e-04 | OPEN |
| 100 | 24 | 8.1955e+00 | 8.4596e+00 | 8.4573e+00 | 8.45397e+00 | -5.16e-04 | -6.28e-04 | -4.82e-04 | -4.85e-04 | -4.51e-04 | -2.48e-04 | 0.991 | 0.963 | yes | not met | - | -0.67 | 6.76e-04 | OPEN |
| 100 | 30 | 8.0120e+00 | 8.2707e+00 | 8.2654e+00 | 8.25867e+00 | -1.05e-03 | -1.02e-03 | -1.03e-03 | -7.99e-04 | -6.47e-04 | -4.06e-04 | 0.974 | 0.971 | yes | met | 3.83e-02 (0.46) | -0.94 | 1.26e-03 | OPEN |
| 100 | 48 | 5.5978e+00 | 5.8883e+00 | 5.8774e+00 | 5.86743e+00 | +7.78e-04 | -9.36e-04 | -2.40e-03 | -2.54e-04 | -5.21e-04 | +1.56e-04 | 1.094 | 0.927 | no | not met | - | - | 3.99e-03 | OPEN |

Over the 17 pairs at the every-10 readout clocks where |288 x_18| exceeds 5e-4: r_a 0.637 to 1.094 (median 0.967), r_b 0.838 to 1.164 (median 0.971); the four terms share a sign at 14 of them, and the criterion (the terms fall from kappa 18 through 24, 32 and 48 with one sign, both per-step ratios below 1) is met at 10: 40/16, 50/2, 50/16, 60/48, 70/16, 70/24, 70/30, 90/16, 90/30, 100/30. The conditional majorant there reads 0.20 to 361.24 percent of the sum (40/16 17.73; 50/2 3.86; 50/16 11.85; 60/48 361.24; 70/16 4.19; 70/24 0.64; 70/30 0.78; 90/16 0.20; 90/30 1.58; 100/30 0.46); it is not applied. For a 1/kappa law of the weighted terms the per-step ratios would be 0.965 and 0.975 (the medians are those of a power law near 1/kappa, whose partial sums grow as ln K); the fitted exponents over the 14 same-sign pairs run -3.71 to -0.09, over all 24 same-sign pairs of the 35 -3.71 to 1.21; the terms change sign between kappa 18 and 48 at 3 of the 17 selected pairs and at 8 of the other 18; at the same cut the terms rise or change sign at neighbouring readout clocks (80/24, 80/30, 60/24, 100/48). These finite-range samples establish neither divergence nor the absence of eventual geometric decay; no unconditional percentage is certified for the completed sum.
Where the signs agree among the selected pairs, the kappa 48 term is 0.026 to 0.924 of the kappa 18 term; the kappa 24 term 0.43 to 1.07; the kappa 32 term 0.13 to 1.04. Over all 155 readout pairs (every 2 r_s) the criterion is met at 42 of the 73 pairs where |288 x_18| exceeds 5e-4 and at 44 of the 155.

### T2c. The pair the v1f rule prints converged (76 r_s, the 24 r_s cut) with the probes at that clock and its neighbours

| T/r_s | cut | S_18 | 256 x_16 | 272 x_17 | 288 x_18 | q_x (last 3) | 384 x_24 | 512 x_32 | 768 x_48 | r_a | r_b | criterion | conditional majorant (percent of \|S\|) | v1f verdict | final tail status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70 | 24 | 5.73254e+00 | +3.35e-03 | +3.15e-03 | +2.82e-03 | 0.942 | +1.76e-03 | +9.68e-04 | +2.14e-04 | 0.928 | 0.910 | met | 3.65e-02 (0.64) | formed, tail open | OPEN |
| 72 | 24 | 6.11462e+00 | +7.74e-03 | +7.52e-03 | +7.38e-03 | 0.982 | +6.33e-03 | +5.20e-03 | +3.98e-03 | 0.976 | 0.983 | met | 4.37e-01 (7.15) | formed, tail open | OPEN |
| 74 | 24 | 6.51471e+00 | +4.26e-03 | +3.98e-03 | +3.82e-03 | 0.960 | +2.82e-03 | +2.12e-03 | +1.43e-03 | 0.965 | 0.976 | met | 1.54e-01 (2.36) | formed, tail open | OPEN |
| 76 | 24 | 6.59347e+00 | +2.41e-03 | +2.23e-03 | +2.14e-03 | 0.964 | +1.64e-03 | +1.21e-03 | +9.00e-04 | 0.963 | 0.982 | met | 1.14e-01 (1.73) | converged | OPEN |
| 78 | 24 | 6.33950e+00 | +1.04e-03 | +7.81e-04 | +6.57e-04 | 0.841 | +4.34e-04 | +2.84e-04 | +2.63e-04 | 0.949 | 0.995 | met | 1.31e-01 (2.07) | formed, tail open | OPEN |
| 80 | 24 | 6.20763e+00 | -1.08e-04 | -1.81e-04 | -9.56e-05 | 1.673 | +7.29e-06 | +5.30e-05 | +2.18e-04 | 1.281 | 1.092 | not met | - | formed, tail open | OPEN |

At 76/24 the v1f rule's clauses hold (q_x 0.964 over kappa 16 to 18, the neighbours 74 and 78 falling, the last three terms resolved, the geometric tail 5.67e-02 and K c 5.69e-02 within 0.01 |S| = 6.59e-02); the probes at the same clock give the row above. The composition prints the pair formed, tail open with the final tail status OPEN like every other pair: the v1f rule's majorant is the last three channels' extrapolation and no remainder bound beyond kappa 18 is supplied at this pair either (T2).

### T2b. Every channel's term at three pairs (the figure's panel c): 16 kappa x_kappa, its propagation floor 16 kappa F_meas,kappa, and the formed-energy term 16 kappa E_kappa

At T/r_s 100, the cut 24 r_s (S_18 = 8.45397e+00 nats; c = 6.76e-04; F_meas = 4.51e-04; the spread of |16 kappa x_kappa| over kappa 12 to 18 a factor 1.40):

| kappa | 16 kappa x_kappa | 16 kappa F_meas,kappa | resolved | 16 kappa E_kappa | partial sum S_kappa |
|---|---|---|---|---|---|
| 1 | +7.334e+00 | 1.7e-06 | yes | 6.44e-01 | 7.33409e+00 |
| 2 | +8.614e-01 | 3.1e-06 | yes | 1.42e-01 | 8.19550e+00 |
| 3 | +1.980e-01 | 5.1e-06 | yes | 2.90e-02 | 8.39355e+00 |
| 4 | +5.107e-02 | 7.8e-06 | yes | 3.88e-03 | 8.44461e+00 |
| 5 | +1.130e-02 | 1.3e-05 | yes | 1.41e-03 | 8.45591e+00 |
| 6 | +3.430e-03 | 1.9e-05 | yes | 5.55e-04 | 8.45934e+00 |
| 7 | +7.179e-04 | 2.5e-05 | yes | 2.45e-04 | 8.46006e+00 |
| 8 | -4.532e-04 | 1.8e-05 | yes | 1.69e-04 | 8.45961e+00 |
| 9 | -5.610e-04 | 1.5e-05 | yes | 1.58e-04 | 8.45905e+00 |
| 10 | -5.567e-04 | 1.4e-05 | yes | 1.55e-04 | 8.45849e+00 |
| 11 | -6.067e-04 | 2.1e-05 | yes | 1.51e-04 | 8.45788e+00 |
| 12 | -5.906e-04 | 3.3e-05 | yes | 1.47e-04 | 8.45729e+00 |
| 13 | -6.762e-04 | 3.9e-05 | yes | 1.38e-04 | 8.45662e+00 |
| 14 | -4.951e-04 | 4.1e-05 | yes | 1.26e-04 | 8.45612e+00 |
| 15 | -5.221e-04 | 5.0e-05 | yes | 1.11e-04 | 8.45560e+00 |
| 16 | -5.160e-04 | 4.5e-05 | yes | 9.49e-05 | 8.45508e+00 |
| 17 | -6.277e-04 | 5.4e-05 | yes | 8.35e-05 | 8.45445e+00 |
| 18 | -4.821e-04 | 4.9e-05 | yes | 7.04e-05 | 8.45397e+00 |
| 24 (probe) | -4.850e-04 | 7.6e-05 | yes | 6.02e-05 | - |
| 32 (probe) | -4.508e-04 | 1.2e-04 | yes | 5.30e-05 | - |
| 48 (probe) | -2.479e-04 | 1.3e-04 | yes | 8.65e-05 | - |

At T/r_s 80, the cut 30 r_s (S_18 = 5.91070e+00 nats; c = 3.05e-03; F_meas = 3.66e-04; the spread of |16 kappa x_kappa| over kappa 12 to 18 a factor 1.25):

| kappa | 16 kappa x_kappa | 16 kappa F_meas,kappa | resolved | 16 kappa E_kappa | partial sum S_kappa |
|---|---|---|---|---|---|
| 1 | +4.645e+00 | 1.3e-06 | yes | 6.63e-01 | 4.64474e+00 |
| 2 | +9.049e-01 | 2.5e-06 | yes | 1.41e-01 | 5.54962e+00 |
| 3 | +2.295e-01 | 3.9e-06 | yes | 2.86e-02 | 5.77917e+00 |
| 4 | +6.955e-02 | 6.0e-06 | yes | 3.78e-03 | 5.84872e+00 |
| 5 | +2.414e-02 | 9.4e-06 | yes | 1.39e-03 | 5.87286e+00 |
| 6 | +7.061e-03 | 1.4e-05 | yes | 5.47e-04 | 5.87992e+00 |
| 7 | +3.442e-03 | 2.0e-05 | yes | 2.37e-04 | 5.88336e+00 |
| 8 | +1.720e-03 | 1.7e-05 | yes | 1.61e-04 | 5.88508e+00 |
| 9 | +1.850e-03 | 1.4e-05 | yes | 1.49e-04 | 5.88693e+00 |
| 10 | +2.065e-03 | 1.2e-05 | yes | 1.47e-04 | 5.88899e+00 |
| 11 | +2.286e-03 | 1.9e-05 | yes | 1.44e-04 | 5.89128e+00 |
| 12 | +2.445e-03 | 2.6e-05 | yes | 1.40e-04 | 5.89372e+00 |
| 13 | +2.577e-03 | 3.0e-05 | yes | 1.32e-04 | 5.89630e+00 |
| 14 | +2.750e-03 | 3.2e-05 | yes | 1.20e-04 | 5.89905e+00 |
| 15 | +2.854e-03 | 3.7e-05 | yes | 1.06e-04 | 5.90190e+00 |
| 16 | +2.785e-03 | 3.7e-05 | yes | 9.11e-05 | 5.90469e+00 |
| 17 | +2.958e-03 | 4.2e-05 | yes | 8.03e-05 | 5.90765e+00 |
| 18 | +3.050e-03 | 4.2e-05 | yes | 6.83e-05 | 5.91070e+00 |
| 24 (probe) | +3.251e-03 | 6.1e-05 | yes | 5.98e-05 | - |
| 32 (probe) | +3.179e-03 | 9.2e-05 | yes | 5.30e-05 | - |
| 48 (probe) | +2.819e-03 | 1.1e-04 | yes | 8.70e-05 | - |

At T/r_s 50, the cut 16 r_s (S_18 = 4.49983e+00 nats; c = 1.57e-02; F_meas = 2.25e-04; the spread of |16 kappa x_kappa| over kappa 12 to 18 a factor 1.27):

| kappa | 16 kappa x_kappa | 16 kappa F_meas,kappa | resolved | 16 kappa E_kappa | partial sum S_kappa |
|---|---|---|---|---|---|
| 1 | +3.061e+00 | 7.5e-07 | yes | 4.22e-01 | 3.06071e+00 |
| 2 | +1.232e+00 | 1.5e-06 | yes | 1.40e-01 | 4.29269e+00 |
| 3 | +2.774e-01 | 2.8e-06 | yes | 2.79e-02 | 4.57012e+00 |
| 4 | +8.850e-02 | 3.5e-06 | yes | 3.61e-03 | 4.65862e+00 |
| 5 | +2.903e-02 | 5.1e-06 | yes | 1.37e-03 | 4.68765e+00 |
| 6 | -3.150e-03 | 6.8e-06 | yes | 5.36e-04 | 4.68450e+00 |
| 7 | -1.630e-02 | 9.7e-06 | yes | 2.24e-04 | 4.66820e+00 |
| 8 | -1.854e-02 | 1.1e-05 | yes | 1.47e-04 | 4.64966e+00 |
| 9 | -1.823e-02 | 1.1e-05 | yes | 1.37e-04 | 4.63143e+00 |
| 10 | -1.746e-02 | 1.1e-05 | yes | 1.35e-04 | 4.61397e+00 |
| 11 | -1.667e-02 | 1.3e-05 | yes | 1.33e-04 | 4.59730e+00 |
| 12 | -1.568e-02 | 1.7e-05 | yes | 1.29e-04 | 4.58162e+00 |
| 13 | -1.514e-02 | 1.8e-05 | yes | 1.21e-04 | 4.56648e+00 |
| 14 | -1.380e-02 | 1.9e-05 | yes | 1.11e-04 | 4.55269e+00 |
| 15 | -1.434e-02 | 2.1e-05 | yes | 9.87e-05 | 4.53834e+00 |
| 16 | -1.336e-02 | 2.3e-05 | yes | 8.46e-05 | 4.52498e+00 |
| 17 | -1.286e-02 | 2.6e-05 | yes | 7.50e-05 | 4.51212e+00 |
| 18 | -1.230e-02 | 2.6e-05 | yes | 6.48e-05 | 4.49983e+00 |
| 24 (probe) | -9.329e-03 | 3.5e-05 | yes | 5.92e-05 | - |
| 32 (probe) | -7.638e-03 | 5.7e-05 | yes | 5.24e-05 | - |
| 48 (probe) | -5.303e-03 | 8.3e-05 | yes | 8.97e-05 | - |

At T/r_s 100, the cut 2 r_s (S_18 = 9.50449e+00 nats; c = 2.10e-04; F_meas = 4.11e-04; the spread of |16 kappa x_kappa| over kappa 12 to 18 a factor 2.75):

| kappa | 16 kappa x_kappa | 16 kappa F_meas,kappa | resolved | 16 kappa E_kappa | partial sum S_kappa |
|---|---|---|---|---|---|
| 1 | +8.239e+00 | 1.5e-06 | yes | 6.44e-01 | 8.23865e+00 |
| 2 | +9.397e-01 | 2.8e-06 | yes | 1.42e-01 | 9.17836e+00 |
| 3 | +2.554e-01 | 4.6e-06 | yes | 2.90e-02 | 9.43378e+00 |
| 4 | +5.577e-02 | 7.1e-06 | yes | 3.88e-03 | 9.48955e+00 |
| 5 | +1.266e-02 | 1.1e-05 | yes | 1.41e-03 | 9.50222e+00 |
| 6 | +3.455e-03 | 1.7e-05 | yes | 5.55e-04 | 9.50567e+00 |
| 7 | +6.170e-04 | 2.2e-05 | yes | 2.45e-04 | 9.50629e+00 |
| 8 | -1.517e-04 | 1.7e-05 | yes | 1.69e-04 | 9.50614e+00 |
| 9 | -1.982e-04 | 1.4e-05 | yes | 1.58e-04 | 9.50594e+00 |
| 10 | -1.794e-04 | 1.3e-05 | yes | 1.55e-04 | 9.50576e+00 |
| 11 | -1.763e-04 | 2.0e-05 | yes | 1.51e-04 | 9.50558e+00 |
| 12 | -1.849e-04 | 3.0e-05 | yes | 1.47e-04 | 9.50540e+00 |
| 13 | -1.422e-04 | 3.5e-05 | yes | 1.38e-04 | 9.50526e+00 |
| 14 | -2.081e-04 | 3.7e-05 | yes | 1.26e-04 | 9.50505e+00 |
| 15 | -2.102e-04 | 4.5e-05 | yes | 1.11e-04 | 9.50484e+00 |
| 16 | -1.573e-04 | 4.1e-05 | yes | 9.49e-05 | 9.50468e+00 |
| 17 | -1.183e-04 | 4.9e-05 | yes | 8.35e-05 | 9.50456e+00 |
| 18 | -7.656e-05 | 4.5e-05 | yes | 7.04e-05 | 9.50449e+00 |
| 24 (probe) | -2.652e-05 | 6.9e-05 | no | 6.02e-05 | - |
| 32 (probe) | +9.353e-06 | 1.1e-04 | no | 5.30e-05 | - |
| 48 (probe) | +2.943e-04 | 1.2e-04 | yes | 8.65e-05 | - |

The formed-energy majorant over the last three channels (q 0.878 to 0.897 from 40 r_s; the energy tail at most 1.1e-3 of the partial sum at K = 18, W1_B2_composed_v3) is a within-K statement: the probes' energies 16 kappa E_kappa at 100 r_s are 6.02e-05, 5.30e-05 and 8.65e-05 at kappa 24, 32 and 48 against 7.04e-05 at kappa 18, so the last-three-channel envelope (which would put the kappa 24 term below 3.7e-5) is not continued by the probes; no energy tail beyond kappa 18 is certified.

### T3. The two averaging scales side by side at K = 18: the sigma/a 4 history (B2) and the sigma/a 8 history (s8), 14 to 40 r_s

The sigma/a 8 history is a different object printed beside, never averaged with B2 (its record ends at 40 r_s; its formation record clock is 12.75 with R_t 44.53 a; B2's 12.125 with R_t 38.56 a). Both sums are the v1f post-pass's at K = 18 (the s8 composition is this lane's rerun of the busi post-pass on the 18 s8 channels; the row at 40 is its from-40 run, the rows before it its from-14 run). R_t is the root the history record reports (the dense fields); min(N - |v|) is copied from the propagation row (the applied fields on the 8192-site chain): their conventions differ and the two are not used together as a trapping certificate here.

| T/r_s | cut r_s | B2: S | F_meas | K c | verdict | s8: S | F_meas | K c | verdict | B2: 288 E_18 | s8: 288 E_18 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 2 | -6.4815e-02 | 6.59e-05 | 2.92e-02 | formed, tail open | -4.0156e-02 | 6.77e-05 | 3.01e-02 | formed, tail open | 4.29e-05 | 5.03e-06 |
| 14 | 16 | -1.0901e-01 | 7.12e-05 | 6.31e-02 | formed, tail open | -4.8354e-02 | 7.32e-05 | 4.09e-02 | formed, tail open | 4.29e-05 | 5.03e-06 |
| 14 | 24 | -1.4598e-01 | 7.23e-05 | 1.69e-01 | not formed | -4.9350e-02 | 7.43e-05 | 5.64e-02 | not formed | 4.29e-05 | 5.03e-06 |
| 14 | 30 | -1.4843e-01 | 7.30e-05 | 1.87e-01 | not formed | -4.6873e-02 | 7.50e-05 | 5.74e-02 | not formed | 4.29e-05 | 5.03e-06 |
| 14 | 48 | -1.1966e-01 | 7.45e-05 | 1.61e-01 | not formed | -3.5892e-02 | 7.66e-05 | 4.78e-02 | not formed | 4.29e-05 | 5.03e-06 |
| 16 | 2 | 3.6696e-02 | 7.26e-05 | 1.92e-02 | formed, tail open | 6.0668e-02 | 7.58e-05 | 1.26e-02 | formed, tail open | 3.97e-05 | 9.47e-06 |
| 16 | 16 | -4.5423e-02 | 7.85e-05 | 6.31e-02 | not formed | -1.3698e-02 | 8.18e-05 | 1.81e-02 | not formed | 3.97e-05 | 9.47e-06 |
| 16 | 24 | -3.7002e-02 | 7.97e-05 | 1.01e-01 | not formed | -1.0473e-02 | 8.31e-05 | 3.28e-02 | not formed | 3.97e-05 | 9.47e-06 |
| 16 | 30 | -4.8157e-02 | 8.04e-05 | 9.09e-02 | not formed | -1.5604e-02 | 8.39e-05 | 2.96e-02 | not formed | 3.97e-05 | 9.47e-06 |
| 16 | 48 | -1.0110e-01 | 8.21e-05 | 1.05e-01 | not formed | -3.2565e-02 | 8.56e-05 | 3.38e-02 | not formed | 3.97e-05 | 9.47e-06 |
| 20 | 2 | 7.1989e-01 | 9.04e-05 | 1.09e+00 | not formed | 1.8169e-01 | 9.30e-05 | 1.25e-02 | formed, tail open | 6.48e-05 | 8.51e-06 |
| 20 | 16 | -2.0431e-02 | 9.77e-05 | 3.98e-02 | not formed | -9.2409e-03 | 1.00e-04 | 1.11e-02 | not formed | 6.48e-05 | 8.51e-06 |
| 20 | 24 | -2.3568e-02 | 9.92e-05 | 4.40e-02 | not formed | -1.1188e-02 | 1.02e-04 | 1.57e-02 | not formed | 6.48e-05 | 8.51e-06 |
| 20 | 30 | -2.9896e-02 | 1.00e-04 | 7.07e-02 | not formed | -1.4481e-02 | 1.03e-04 | 2.40e-02 | not formed | 6.48e-05 | 8.51e-06 |
| 20 | 48 | -2.5713e-02 | 1.02e-04 | 8.73e-02 | not formed | -1.3782e-02 | 1.05e-04 | 2.01e-02 | not formed | 6.48e-05 | 8.51e-06 |
| 30 | 2 | 4.3612e+00 | 1.24e-04 | 4.60e-02 | formed, tail open | 4.8607e-02 | 1.33e-04 | 9.45e-03 | formed, tail open | 6.26e-05 | 9.37e-06 |
| 30 | 16 | 5.9421e-03 | 1.34e-04 | 1.88e-02 | not formed | 1.6210e-03 | 1.43e-04 | 5.46e-03 | not formed | 6.26e-05 | 9.37e-06 |
| 30 | 24 | -2.3493e-03 | 1.36e-04 | 1.29e-02 | not formed | -2.1076e-03 | 1.46e-04 | 7.85e-03 | not formed | 6.26e-05 | 9.37e-06 |
| 30 | 30 | -7.6665e-03 | 1.37e-04 | 2.91e-02 | not formed | -4.4866e-03 | 1.47e-04 | 1.38e-02 | not formed | 6.26e-05 | 9.37e-06 |
| 30 | 48 | -1.6552e-02 | 1.40e-04 | 3.43e-02 | not formed | -9.5453e-03 | 1.50e-04 | 6.68e-03 | within the floor | 6.26e-05 | 9.37e-06 |
| 40 | 2 | 5.8737e+00 | 1.68e-04 | 7.33e-03 | formed, tail open | 1.9127e-02 | 1.77e-04 | 9.64e-03 | formed, tail open | 6.42e-05 | 8.70e-06 |
| 40 | 16 | 9.1879e-02 | 1.82e-04 | 4.04e-02 | formed, tail open | 1.6255e-03 | 1.91e-04 | 2.29e-03 | not formed | 6.42e-05 | 8.70e-06 |
| 40 | 24 | 8.1415e-03 | 1.85e-04 | 1.49e-02 | not formed | -1.7791e-03 | 1.94e-04 | 7.04e-03 | not formed | 6.42e-05 | 8.70e-06 |
| 40 | 30 | 5.1257e-03 | 1.86e-04 | 1.23e-02 | not formed | -1.2497e-03 | 1.96e-04 | 1.01e-02 | not formed | 6.42e-05 | 8.70e-06 |
| 40 | 48 | -5.2873e-03 | 1.89e-04 | 2.27e-02 | not formed | -5.6943e-03 | 1.99e-04 | 6.98e-03 | not formed | 6.42e-05 | 8.70e-06 |

s8 tally over its 15 clocks 14 to 40 r_s (not formed / within the floor / converged / formed, tail open): 2 r_s: 3 / 1 / 0 / 11; 16 r_s: 12 / 1 / 0 / 2; 24 r_s: 13 / 1 / 0 / 1; 30 r_s: 14 / 0 / 0 / 1; 48 r_s: 14 / 1 / 0 / 0. At 40 r_s |S| at the 16 to 48 r_s cuts is 1.2497e-03 to 5.6943e-03 nats against K c 2.29e-03 to 1.01e-02.
s8's weighted energy terms 16 kappa E_kappa at 40 r_s over kappa 1 to 18: 1: 1.93e-02, 2: 8.66e-04, 3: 1.83e-05, 4: 3.19e-06, 5: 2.55e-06, 6: 2.85e-06, 7: 3.37e-06, 8: 4.00e-06, 9: 4.45e-06, 10: 4.93e-06, 11: 6.12e-06, 12: 6.77e-06, 13: 7.41e-06, 14: 7.97e-06, 15: 8.33e-06, 16: 1.01e-05, 17: 9.11e-06, 18: 8.70e-06 (B2's at 40 r_s over kappa 12 to 18: 12: 1.23e-04, 13: 1.17e-04, 14: 1.07e-04, 15: 9.60e-05, 16: 8.26e-05, 17: 7.41e-05, 18: 6.42e-05).

The sigma/a 8 history's collar flux (kappa 1, 2; L per coordinate interval; the comparator where the record has a root):

| T/r_s | s8: E | R_t (a) | min(N - \|v\|) | L (kappa 1, 2) | L_H(kappa_T) | L/L_H(kappa_T) | L/L_H(T_H) | E_inv (kappa 1, 2) | E_inv/E, percent | collar flow acc. (kappa 1, 2) | flow over E | join max rel dv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 115.1727 | 52.908 | -0.0885 | -5.966e-09 | 1.650e-06 | -0.0036 | -0.0038 | 8.0136e-01 | 0.70 | -8.6808e-07 | -7.537e-09 | 1.72e-05 |
| 15 | 115.2263 | 54.578 | -0.0439 | -8.921e-09 | 1.178e-06 | -0.0076 | -0.0057 | 4.8459e-02 | 0.04 | -1.4391e-06 | -1.249e-08 | 1.72e-05 |
| 16 | 115.1394 | 52.913 | +0.0142 | -5.731e-09 | 7.439e-08 | -0.0770 | -0.0036 | 3.9750e-03 | 0.00 | -1.8058e-06 | -1.568e-08 | 1.72e-05 |
| 20 | 114.4759 | - | +0.1783 | +7.833e-08 | - | - | +0.0490 | 1.4913e-02 | 0.01 | +6.1987e-06 | 5.415e-08 | 1.72e-05 |
| 30 | 113.7658 | - | +0.0605 | -4.952e-08 | - | - | -0.0306 | 1.9714e-02 | 0.02 | -3.5658e-06 | -3.134e-08 | 1.72e-05 |
| 36 | 113.5094 | - | +0.1614 | +2.688e-08 | - | - | +0.0165 | 2.0038e-02 | 0.02 | -5.7612e-07 | -5.076e-09 | 1.72e-05 |
| 38 | 113.4263 | - | +0.0937 | +8.530e-09 | - | - | +0.0052 | 2.0108e-02 | 0.02 | +5.1576e-07 | 4.547e-09 | 1.72e-05 |
| 40 | 113.2621 | 40.963 | +0.0384 | +8.115e-08 | 1.878e-07 | +0.4320 | +0.0497 | 2.0117e-02 | 0.02 | +1.0903e-05 | 9.626e-08 | 1.72e-05 |

The s8 16-sub-step control (kappa 14, to 40 r_s, complete): D16 = 224 [x_14(16 sub) - x_14(8 sub)] against the kappa 14 term 224 x_14, and the formed energy 224 E_14 under both sub-steps (one weighted channel term's change; 18 |D16| the whole-set proxy of T1b). Over the 15 formed pairs (2 r_s: 11; 16 r_s: 2; 24 r_s: 1; 30 r_s: 1) 18 |D16|/|S_18| is at most 0.72 percent and |D16|/|224 x_14| at most 0.197 (at 18/2); at the not-formed pair 40/16 the term ratio is 0.074 and at 40/48 0.055.

| T/r_s | cut r_s | 224 x_14 (8 sub) | 224 x_14 (16 sub) | D16 | D16 / (224 x_14) | s8 S_18 | 18 \|D16\|/\|S_18\| (percent) | 224 E_14 (8 sub) | 224 E_14 (16 sub) |
|---|---|---|---|---|---|---|---|---|---|
| 14 | 2 | +1.421e-03 | +1.437e-03 | +1.60e-05 | 0.011 | -4.0156e-02 | 0.72 | 6.25e-06 | 5.55e-06 |
| 14 | 16 | -1.580e-03 | -1.579e-03 | +4.62e-07 | -0.000 | -4.8354e-02 | 0.02 | 6.25e-06 | 5.55e-06 |
| 14 | 24 | -2.710e-03 | -2.709e-03 | +7.48e-07 | -0.000 | -4.9350e-02 | 0.03 | 6.25e-06 | 5.55e-06 |
| 14 | 30 | -2.955e-03 | -2.954e-03 | +9.89e-07 | -0.000 | -4.6873e-02 | 0.04 | 6.25e-06 | 5.55e-06 |
| 14 | 48 | -2.655e-03 | -2.654e-03 | +1.70e-06 | -0.001 | -3.5892e-02 | 0.09 | 6.25e-06 | 5.55e-06 |
| 16 | 2 | +5.591e-04 | +5.663e-04 | +7.20e-06 | 0.013 | 6.0668e-02 | 0.21 | 1.20e-05 | 1.20e-05 |
| 16 | 16 | -4.223e-04 | -4.218e-04 | +4.82e-07 | -0.001 | -1.3698e-02 | 0.06 | 1.20e-05 | 1.20e-05 |
| 16 | 24 | +1.729e-03 | +1.730e-03 | +8.56e-07 | 0.000 | -1.0473e-02 | 0.15 | 1.20e-05 | 1.20e-05 |
| 16 | 30 | +1.429e-03 | +1.430e-03 | +1.06e-06 | 0.001 | -1.5604e-02 | 0.12 | 1.20e-05 | 1.20e-05 |
| 16 | 48 | -1.088e-03 | -1.087e-03 | +1.66e-06 | -0.002 | -3.2565e-02 | 0.09 | 1.20e-05 | 1.20e-05 |
| 20 | 2 | -6.267e-04 | -6.204e-04 | +6.27e-06 | -0.010 | 1.8169e-01 | 0.06 | 8.49e-06 | 7.27e-06 |
| 20 | 16 | -6.177e-04 | -6.174e-04 | +2.77e-07 | -0.000 | -9.2409e-03 | 0.05 | 8.49e-06 | 7.27e-06 |
| 20 | 24 | +3.058e-04 | +3.062e-04 | +4.83e-07 | 0.002 | -1.1188e-02 | 0.08 | 8.49e-06 | 7.27e-06 |
| 20 | 30 | -1.147e-03 | -1.147e-03 | +5.76e-07 | -0.001 | -1.4481e-02 | 0.07 | 8.49e-06 | 7.27e-06 |
| 20 | 48 | +5.987e-04 | +5.997e-04 | +9.50e-07 | 0.002 | -1.3782e-02 | 0.12 | 8.49e-06 | 7.27e-06 |
| 30 | 2 | +2.383e-04 | +2.393e-04 | +9.73e-07 | 0.004 | 4.8607e-02 | 0.04 | 9.83e-06 | 9.27e-06 |
| 30 | 16 | -3.035e-04 | -3.038e-04 | -2.41e-07 | 0.001 | 1.6210e-03 | 0.27 | 9.83e-06 | 9.27e-06 |
| 30 | 24 | +3.256e-04 | +3.258e-04 | +2.42e-07 | 0.001 | -2.1076e-03 | 0.21 | 9.83e-06 | 9.27e-06 |
| 30 | 30 | +5.714e-04 | +5.717e-04 | +3.03e-07 | 0.001 | -4.4866e-03 | 0.12 | 9.83e-06 | 9.27e-06 |
| 30 | 48 | +2.527e-04 | +2.532e-04 | +4.73e-07 | 0.002 | -9.5453e-03 | 0.09 | 9.83e-06 | 9.27e-06 |
| 40 | 2 | +4.617e-04 | +4.663e-04 | +4.60e-06 | 0.010 | 1.9127e-02 | 0.43 | 7.97e-06 | 6.98e-06 |
| 40 | 16 | +1.271e-04 | +1.365e-04 | +9.38e-06 | 0.074 | 1.6255e-03 | 10.39 | 7.97e-06 | 6.98e-06 |
| 40 | 24 | -1.935e-05 | -1.958e-05 | -2.33e-07 | 0.012 | -1.7791e-03 | 0.24 | 7.97e-06 | 6.98e-06 |
| 40 | 30 | -5.167e-05 | -5.209e-05 | -4.20e-07 | 0.008 | -1.2497e-03 | 0.61 | 7.97e-06 | 6.98e-06 |
| 40 | 48 | -1.224e-05 | -1.290e-05 | -6.68e-07 | 0.055 | -5.6943e-03 | 0.21 | 7.97e-06 | 6.98e-06 |

### T4. The completion spread: the kappa 1 and 2 channels (weights 16 and 32) on B2 and on W3's two completions, to 40 r_s

- B2 (the record's own continuation, engine_collapse_v19, 4096 sites): engine sites 4096, records 801; the record's label (end): lled); economy: lagged first iterate, stop /dE/E/ < 1e-07 and fields < 1e-06, plain iteration, at most 20 iterations (the smallest-residual iterate returned, flagged); tau = 8; collar policy flag; dictionary: H = N E + v.p, ds^2 = -N^2 dt^2 + (dx - v dt)^2, the ADM sign of the momentum law (J_i = -(1/8 pi G) d_j (K^j_i - delta^j_i K) for K_ij = (d_i v_j + d_j v_i)/(2N)); the goal's display's sign of v amended per engine 12.2; the owner's confirmation pending; J_ell = <P_r> = -J_r of the goal's display (engine 20.3)
- held-sources (W3fs): engine sites 2048, records 321; the record's label (end):  test fails); v36g W3: completion held-sources (the run tag W3fs) from the first record failing the hydrodynamic gate sigma max(/v'/N/, /v/(r N)/) <= 0.1 (sigma = ell = 4 a) at or after T/rs 11.125, inside the gate radius the arming record's own, rounded up to the next multiple of ell, the blend 1 -> 0 over [R_* - ell, R_* + ell]; the sources held inside at the gated record's values, the charge projected by one scalar lambda per field solve so that the collar charge equals the stored member's E at the same clock (tolerance 1e-09)
- paired-avg (W3pa): engine sites 2048, records 321; the record's label (end): tion where the Weyl test fails); v36g W3: completion paired-avg from the first record failing the hydrodynamic gate sigma max(/v'/N/, /v/(r N)/) <= 0.1 (sigma = ell = 4 a) at or after T/rs 11.125, inside the gate radius the arming record's own, rounded up to the next multiple of ell, the blend 1 -> 0 over [R_* - ell, R_* + ell]; the paired averaging widened to 8 a inside (the shell mixing of width sqrt(8^2 - 4^2) a on the sources before the ell = 4 a solve, its transpose on the back-averaged fields: adjoint-paired)

The completions arm at the first record failing the hydrodynamic gate at or after T/r_s 11.125 (the label above), restarted from the rung-4 run's stored slice at T/r_s 11.0 on engine v36g with 2048 engine sites; B2 is engine_collapse_v19's record on 4096 sites. The paired-avg completion runs at a charge above the record's (E 113.01 against 106.21 at 40 r_s, 6.4 percent; 114.43 against 109.28 at 14), so its exterior differs at zeroth order; the held-sources completion's charge is projected to the record's E. The spread below carries those differences with the interior's.

| T/r_s | cut r_s | B2: 16 x_1 + 32 x_2 | held-sources (W3fs) | paired-avg (W3pa) | spread (all three) | spread (the two completions) | spread over \|S_18\| (percent) |
|---|---|---|---|---|---|---|---|
| 14 | 2 | -9.6199e-02 | -5.3201e-02 | -5.9412e-02 | 4.2998e-02 | 6.2107e-03 | 66.3 |
| 14 | 16 | -1.1916e-02 | -1.1432e-02 | -7.5303e-03 | 4.3860e-03 | 3.9013e-03 | 4.0 |
| 14 | 24 | -5.2923e-03 | -5.1316e-03 | -3.2982e-03 | 1.9941e-03 | 1.8334e-03 | 1.4 |
| 14 | 30 | -3.2728e-03 | -3.2145e-03 | -2.0885e-03 | 1.1843e-03 | 1.1260e-03 | 0.8 |
| 14 | 48 | -9.8152e-04 | -9.6117e-04 | -7.0916e-04 | 2.7237e-04 | 2.5202e-04 | 0.2 |
| 16 | 2 | +2.0618e-02 | +8.9993e-02 | +5.9011e-02 | 6.9375e-02 | 3.0981e-02 | 189.1 |
| 16 | 16 | -2.6964e-02 | -2.6114e-02 | -1.2875e-02 | 1.4089e-02 | 1.3238e-02 | 31.0 |
| 16 | 24 | -1.6214e-02 | -1.5911e-02 | -7.4572e-03 | 8.7571e-03 | 8.4535e-03 | 23.7 |
| 16 | 30 | -1.1857e-02 | -1.1687e-02 | -5.4069e-03 | 6.4498e-03 | 6.2799e-03 | 13.4 |
| 16 | 48 | -5.7052e-03 | -5.6609e-03 | -2.5898e-03 | 3.1154e-03 | 3.0711e-03 | 3.1 |
| 20 | 2 | +1.8639e+00 | +2.0279e-01 | +4.2177e-01 | 1.6611e+00 | 2.1898e-01 | 230.7 |
| 20 | 16 | -2.3851e-02 | -2.3352e-02 | -1.0393e-02 | 1.3457e-02 | 1.2959e-02 | 65.9 |
| 20 | 24 | -2.5257e-02 | -2.5094e-02 | -1.1403e-02 | 1.3854e-02 | 1.3691e-02 | 58.8 |
| 20 | 30 | -2.1837e-02 | -2.1749e-02 | -9.7536e-03 | 1.2084e-02 | 1.1996e-02 | 40.4 |
| 20 | 48 | -1.3445e-02 | -1.3426e-02 | -5.9619e-03 | 7.4835e-03 | 7.4639e-03 | 29.1 |
| 30 | 2 | +3.9898e+00 | -2.4717e-02 | +1.2659e+00 | 4.0146e+00 | 1.2906e+00 | 92.1 |
| 30 | 16 | +6.9730e-03 | +5.3641e-03 | +4.6168e-03 | 2.3562e-03 | 7.4726e-04 | 39.7 |
| 30 | 24 | -2.5124e-03 | -2.8764e-03 | -5.4761e-04 | 2.3288e-03 | 2.3288e-03 | 99.1 |
| 30 | 30 | -1.3218e-02 | -1.3387e-02 | -5.8650e-03 | 7.5224e-03 | 7.5224e-03 | 98.1 |
| 30 | 48 | -1.9776e-02 | -1.9812e-02 | -8.5937e-03 | 1.1219e-02 | 1.1219e-02 | 67.8 |
| 40 | 2 | +5.5330e+00 | -2.9302e-02 | +7.4290e-01 | 5.5623e+00 | 7.7220e-01 | 94.7 |
| 40 | 16 | +5.3114e-02 | -1.5167e-02 | +2.5880e-02 | 6.8282e-02 | 4.1047e-02 | 74.3 |
| 40 | 24 | +4.4640e-03 | -2.8564e-04 | +2.3452e-03 | 4.7497e-03 | 2.6308e-03 | 58.3 |
| 40 | 30 | +8.1395e-03 | +6.0583e-03 | +4.6741e-03 | 3.4654e-03 | 1.3842e-03 | 67.6 |
| 40 | 48 | -1.0222e-02 | -1.0644e-02 | -4.3941e-03 | 6.2501e-03 | 6.2501e-03 | 118.2 |

| T/r_s | run | E (charge) | R_t (a) | min(N - \|v\|) | E_inv (kappa 1, 2) | collar flow acc. (kappa 1, 2) | L (kappa 1, 2) | L_H(kappa_T) | join max dN | join max rel dv |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | B2 | 116.9336 | - | +0.0404 | 5.5220e-04 | -1.7327e-07 | -1.212e-09 | - | 7.80e-07 | 5.21e-06 |
| 12 | held-sources | 116.9337 | - | +0.0407 | 5.8131e-04 | -1.7413e-07 | -1.215e-09 | - | 3.88e-07 | 4.58e-06 |
| 12 | paired-avg | 117.2406 | - | +0.0455 | 5.3846e-04 | -1.3447e-07 | -9.047e-10 | - | 3.20e-07 | 4.53e-06 |
| 14 | B2 | 109.2800 | 50.575 | -0.1835 | 1.0709e+01 | -2.7799e-06 | -2.036e-08 | 1.932e-06 | 3.02e-07 | 4.76e-06 |
| 14 | held-sources | 109.2800 | 49.762 | -0.0476 | 1.1990e-01 | -2.7799e-06 | -2.036e-08 | 7.749e-07 | 1.56e-07 | 4.39e-06 |
| 14 | paired-avg | 114.4280 | 53.396 | -0.1339 | 4.2057e+00 | -1.3571e-06 | -9.552e-09 | 1.669e-06 | 3.14e-08 | 4.31e-06 |
| 16 | B2 | 108.5480 | 50.837 | -0.0284 | 1.7089e-01 | -5.5572e-06 | -1.720e-08 | 1.186e-07 | 9.27e-08 | 4.49e-06 |
| 16 | held-sources | 108.5480 | - | +0.0523 | 6.4590e-02 | -5.5560e-06 | -1.719e-08 | - | 4.53e-08 | 4.32e-06 |
| 16 | paired-avg | 114.0443 | 52.157 | -0.0044 | 4.8797e-02 | -2.4077e-06 | -6.383e-09 | 7.345e-09 | 4.85e-08 | 4.32e-06 |
| 20 | B2 | 107.7251 | 38.432 | -0.1057 | 1.4242e+00 | +4.1240e-07 | +6.746e-08 | 2.167e-06 | 2.84e-07 | 4.72e-06 |
| 20 | held-sources | 107.7251 | - | +0.2219 | 1.9198e-01 | +4.1278e-07 | +6.746e-08 | - | 1.43e-07 | 4.39e-06 |
| 20 | paired-avg | 113.5156 | - | +0.0874 | 1.1958e-01 | +3.3977e-06 | +6.289e-08 | - | 1.27e-08 | 4.30e-06 |
| 30 | B2 | 106.6753 | 37.511 | -0.0600 | 4.5275e-01 | -1.2361e-05 | -6.486e-08 | 3.491e-06 | 3.81e-08 | 4.42e-06 |
| 30 | held-sources | 106.6753 | - | +0.4043 | 2.3857e-01 | -1.2364e-05 | -6.493e-08 | - | 1.95e-08 | 4.30e-06 |
| 30 | paired-avg | 113.1325 | - | +0.0333 | 1.3884e-01 | -4.7162e-06 | -3.291e-08 | - | 8.94e-09 | 4.30e-06 |
| 40 | B2 | 106.2057 | 36.310 | -0.1118 | 1.4378e+00 | +8.0263e-04 | +6.329e-06 | 5.281e-06 | 6.41e-08 | 4.45e-06 |
| 40 | held-sources | 106.2058 | - | +0.4976 | 2.4327e-01 | -2.2190e-07 | +7.813e-08 | - | 3.24e-08 | 4.31e-06 |
| 40 | paired-avg | 113.0110 | - | +0.0563 | 1.4132e-01 | +4.2940e-05 | +2.875e-07 | - | 4.60e-09 | 4.30e-06 |

The spread of the charge E over the three histories at the sampled clocks 14 to 40: 5.148 to 6.805 (the two completions alone 5.148 to 6.805; 0 at the start).

### T5. The collar flux of the kappa 1 and 2 channels on B2 to 100 r_s, with the kappa 3 guard's change and the K = 18 sum beside

L is the collar energy flow per coordinate interval Delta T (the increment of the stored accumulator over the checkpoint interval divided by its length; identity checked to 0.0e+00 relative), placed at the collar's recordwise stationary-chart clock t_inf (t_inf - T = 0.366 to 0.508 r_s over 14 to 100); L_H(kappa_T) and L_H(T_H) are the channel set's continuum greybody comparators at the record's dense root and slope and at 1/(8 pi G E), not an observed thermal flux. The ratios with kappa 3 add the truncation guard (a truncation diagnostic, not a bound on every omitted channel); the K = 18 ratio sums every channel's L (the high channels' accumulators carry the moving exterior's first-order term, W1 ruling S3, so that ratio is a diagnostic only). 'v6b' is version 3's two-channel ratio on the piecewise-constant propagation of the r11 record (through 40), the propagation variant beside. The E_inv/E columns are percentages.

| T/r_s | t_inf/r_s | E | L (kappa 1, 2) | L_H(kappa_T) | L_H(T_H) | L/L_H(kappa_T) | L/L_H(T_H) | with kappa 3: L/L_H(kappa_T) | L/L_H(T_H) | K = 18: L/L_H(kappa_T) | E_inv/E, percent, kappa 1 and 2 | E_inv/E, percent, kappa 1 to 3 | E_inv/E, percent, K = 18 | join max rel dv | v6b L/L_H(kappa_T) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | 14.366 | 109.280 | -2.036e-08 | 1.932e-06 | 1.753e-06 | -0.0105 | -0.0116 | -0.0268 | -0.0295 | -1.8081 | 9.80 | 15.79 | 21.67 | 4.76e-06 | -0.0101 |
| 15 | 15.385 | 108.827 | -2.619e-08 | 1.645e-06 | 1.768e-06 | -0.0159 | -0.0148 | -0.0358 | -0.0333 | +0.0889 | 1.98 | 2.50 | 2.50 | 4.57e-06 | -0.0162 |
| 16 | 16.397 | 108.548 | -1.720e-08 | 1.186e-07 | 1.777e-06 | -0.1451 | -0.0097 | -0.2636 | -0.0176 | -1.6342 | 0.16 | 0.17 | 0.17 | 4.49e-06 | -0.1498 |
| 18 | 18.407 | 108.323 | -2.082e-08 | - | 1.784e-06 | - | -0.0117 | - | -0.0137 | - | 0.25 | 0.27 | 0.28 | 4.41e-06 | - |
| 20 | 20.432 | 107.725 | +6.746e-08 | 2.167e-06 | 1.804e-06 | +0.0311 | +0.0374 | +0.0245 | +0.0294 | -0.0525 | 1.32 | 1.34 | 1.35 | 4.72e-06 | +0.0313 |
| 22 | 22.450 | 107.321 | -3.314e-08 | 3.548e-06 | 1.818e-06 | -0.0093 | -0.0182 | -0.0112 | -0.0220 | -0.0743 | 1.75 | 1.77 | 1.78 | 4.41e-06 | -0.0103 |
| 24 | 24.460 | 107.088 | -5.475e-09 | 1.340e-06 | 1.825e-06 | -0.0041 | -0.0030 | -0.0107 | -0.0079 | -0.0602 | 0.32 | 0.35 | 0.35 | 4.41e-06 | -0.0022 |
| 26 | 26.463 | 107.009 | +3.354e-08 | 1.651e-06 | 1.828e-06 | +0.0203 | +0.0183 | -0.0028 | -0.0026 | -0.0143 | 0.33 | 0.35 | 0.36 | 4.41e-06 | +0.0214 |
| 28 | 28.474 | 106.756 | -2.986e-08 | 4.735e-06 | 1.837e-06 | -0.0063 | -0.0163 | +0.0071 | +0.0182 | -0.0046 | 2.52 | 2.55 | 2.55 | 4.39e-06 | -0.0057 |
| 30 | 30.478 | 106.675 | -6.486e-08 | 3.491e-06 | 1.840e-06 | -0.0186 | -0.0353 | -0.0090 | -0.0171 | -0.0340 | 0.42 | 0.45 | 0.46 | 4.42e-06 | -0.0265 |
| 32 | 32.480 | 106.609 | +2.792e-09 | 1.767e-06 | 1.842e-06 | +0.0016 | +0.0015 | +0.0057 | +0.0055 | -0.0215 | 0.35 | 0.37 | 0.38 | 4.38e-06 | +0.0158 |
| 34 | 34.487 | 106.456 | -1.333e-08 | 4.922e-06 | 1.847e-06 | -0.0027 | -0.0072 | -0.0310 | -0.0827 | -0.0698 | 1.89 | 1.91 | 1.92 | 4.43e-06 | +0.0043 |
| 36 | 36.490 | 106.396 | +9.128e-08 | 5.103e-06 | 1.849e-06 | +0.0179 | +0.0494 | +0.0332 | +0.0920 | +0.5504 | 0.65 | 0.67 | 0.68 | 4.41e-06 | +4.0141 |
| 38 | 38.494 | 106.303 | -4.304e-08 | 1.594e-06 | 1.852e-06 | -0.0270 | -0.0232 | +0.1296 | +0.1115 | +1.5178 | 0.38 | 0.40 | 0.41 | 4.38e-06 | +5.1180 |
| 40 | 40.498 | 106.206 | +6.329e-06 | 5.281e-06 | 1.856e-06 | +1.1985 | +3.4105 | +1.4403 | +4.1116 | +1.9274 | 1.35 | 1.38 | 1.39 | 4.45e-06 | +1.4110 |
| 42 | 42.501 | 106.132 | +1.079e-03 | 5.671e-06 | 1.858e-06 | +190.2556 | +580.4975 | +202.4509 | +619.9252 | +202.1689 | 0.76 | 0.78 | 0.79 | 4.39e-06 | - |
| 44 | 44.503 | 106.083 | +9.556e-04 | 3.508e-06 | 1.860e-06 | +272.3820 | +513.7237 | +294.4211 | +556.0367 | +297.1049 | 0.42 | 0.45 | 0.45 | 4.37e-06 | - |
| 46 | 46.506 | 106.018 | +2.970e-04 | 4.834e-06 | 1.862e-06 | +61.4497 | +159.4925 | +66.9790 | +174.3074 | +71.2275 | 1.43 | 1.46 | 1.46 | 4.41e-06 | - |
| 48 | 48.505 | 106.031 | +6.581e-05 | 5.201e-06 | 1.862e-06 | +12.6517 | +35.3422 | +15.4788 | +43.3720 | +15.0539 | 0.68 | 0.70 | 0.71 | 4.42e-06 | - |
| 50 | 50.506 | 106.018 | +3.508e-05 | 4.245e-06 | 1.862e-06 | +8.2647 | +18.8375 | +8.6018 | +19.6461 | +5.7856 | 0.53 | 0.56 | 0.56 | 4.45e-06 | - |
| 52 | 52.506 | 106.007 | +1.482e-04 | 5.666e-06 | 1.863e-06 | +26.1546 | +79.5444 | +26.6523 | +81.3475 | +26.1546 | 0.77 | 0.79 | 0.80 | 4.62e-06 | - |
| 54 | 54.506 | 106.008 | +8.759e-05 | 5.466e-06 | 1.863e-06 | +16.0245 | +47.0207 | +16.6619 | +49.0545 | +18.4845 | 0.99 | 1.02 | 1.02 | 5.16e-06 | - |
| 56 | 56.507 | 106.001 | +1.463e-05 | 4.758e-06 | 1.863e-06 | +3.0750 | +7.8532 | +3.3378 | +8.5465 | +4.6898 | 0.66 | 0.69 | 0.70 | 7.12e-06 | - |
| 58 | 58.507 | 105.996 | +4.457e-05 | 5.076e-06 | 1.863e-06 | +8.7790 | +23.9184 | +8.6435 | +23.6180 | +9.4817 | 0.63 | 0.65 | 0.66 | 5.84e-05 | - |
| 60 | 60.507 | 105.987 | +8.484e-05 | 5.783e-06 | 1.864e-06 | +14.6706 | +45.5279 | +14.9899 | +46.6910 | +14.9779 | 1.06 | 1.09 | 1.09 | 4.57e-04 | - |
| 62 | 62.507 | 105.984 | +1.978e-05 | 5.365e-06 | 1.864e-06 | +3.6877 | +10.6159 | +3.9096 | +11.2911 | +3.7780 | 0.70 | 0.73 | 0.74 | 7.39e-04 | - |
| 64 | 64.508 | 105.983 | +2.762e-05 | 4.960e-06 | 1.864e-06 | +5.5691 | +14.8219 | +5.4043 | +14.4236 | +5.4055 | 0.74 | 0.77 | 0.78 | 8.70e-04 | - |
| 66 | 66.507 | 105.985 | +1.095e-04 | 5.654e-06 | 1.864e-06 | +19.3676 | +58.7602 | +19.6456 | +59.8156 | +19.7222 | 0.66 | 0.68 | 0.69 | 1.00e-03 | - |
| 68 | 68.507 | 105.987 | +3.165e-05 | 5.603e-06 | 1.864e-06 | +5.6488 | +16.9845 | +5.7637 | +17.3909 | +5.6038 | 0.69 | 0.72 | 0.72 | 1.19e-03 | - |
| 70 | 70.507 | 105.988 | +1.240e-05 | 5.729e-06 | 1.864e-06 | +2.1651 | +6.6563 | +2.1943 | +6.7707 | +2.2748 | 0.82 | 0.85 | 0.85 | 1.34e-03 | - |
| 72 | 72.507 | 105.988 | +5.387e-05 | 5.311e-06 | 1.864e-06 | +10.1441 | +28.9080 | +10.1236 | +28.9413 | +10.2681 | 0.61 | 0.64 | 0.65 | 1.45e-03 | - |
| 74 | 74.507 | 105.989 | +2.351e-05 | 5.741e-06 | 1.863e-06 | +4.0946 | +12.6153 | +4.1534 | +12.8433 | +4.2642 | 0.90 | 0.92 | 0.93 | 1.65e-03 | - |
| 76 | 76.507 | 106.002 | +1.640e-05 | 5.602e-06 | 1.863e-06 | +2.9281 | +8.8045 | +2.9150 | +8.7957 | +2.7732 | 0.70 | 0.73 | 0.74 | 2.86e-03 | - |
| 78 | 78.507 | 106.003 | +7.208e-06 | 5.364e-06 | 1.863e-06 | +1.3437 | +3.8690 | +1.4310 | +4.1338 | +1.4172 | 0.59 | 0.62 | 0.63 | 7.05e-03 | - |
| 80 | 80.507 | 106.004 | +5.613e-06 | 5.815e-06 | 1.863e-06 | +0.9653 | +3.0131 | +1.0060 | +3.1518 | +1.1898 | 0.76 | 0.79 | 0.79 | 1.54e-02 | - |
| 82 | 82.507 | 106.005 | +2.757e-06 | 6.118e-06 | 1.863e-06 | +0.4507 | +1.4800 | +0.7133 | +2.3521 | +0.7997 | 0.65 | 0.68 | 0.68 | 2.62e-02 | - |
| 84 | 84.506 | 106.013 | -1.449e-07 | 5.723e-06 | 1.863e-06 | -0.0253 | -0.0778 | -0.1305 | -0.4025 | -0.0271 | 0.73 | 0.75 | 0.76 | 3.83e-02 | - |
| 86 | 86.506 | 106.023 | +4.670e-06 | 5.427e-06 | 1.862e-06 | +0.8605 | +2.5078 | +0.8790 | +2.5703 | +0.9457 | 0.80 | 0.83 | 0.84 | 4.81e-02 | - |
| 88 | 88.505 | 106.032 | +5.087e-06 | 5.616e-06 | 1.862e-06 | +0.9058 | +2.7319 | +0.9408 | +2.8475 | +0.9184 | 0.78 | 0.80 | 0.81 | 5.71e-02 | - |
| 90 | 90.505 | 106.033 | +2.470e-06 | 5.460e-06 | 1.862e-06 | +0.4524 | +1.3268 | +0.5872 | +1.7276 | +0.4012 | 0.65 | 0.68 | 0.69 | 6.23e-02 | - |
| 92 | 92.505 | 106.037 | +3.536e-06 | 5.997e-06 | 1.862e-06 | +0.5896 | +1.8992 | +0.5647 | +1.8262 | +0.5894 | 0.75 | 0.78 | 0.78 | 6.60e-02 | - |
| 94 | 94.505 | 106.038 | +6.589e-06 | 5.454e-06 | 1.862e-06 | +1.2080 | +3.5391 | +1.2661 | +3.7215 | +1.5001 | 0.78 | 0.81 | 0.82 | 6.84e-02 | - |
| 96 | 96.505 | 106.042 | +4.154e-07 | 5.266e-06 | 1.862e-06 | +0.0789 | +0.2231 | +0.1905 | +0.5406 | +0.2824 | 0.69 | 0.72 | 0.73 | 6.98e-02 | - |
| 98 | 98.505 | 106.048 | +7.214e-06 | 5.558e-06 | 1.861e-06 | +1.2978 | +3.8755 | +1.2744 | +3.8188 | +1.3715 | 0.69 | 0.72 | 0.73 | 7.10e-02 | - |
| 100 | 100.504 | 106.053 | +3.288e-06 | 5.604e-06 | 1.861e-06 | +0.5868 | +1.7668 | +0.6307 | +1.9054 | +0.6841 | 0.74 | 0.77 | 0.77 | 7.18e-02 | - |

The accumulated collar flow of kappa 1 and 2 over E (E_rad^(0)/E, a diagnostic): 7.557e-06 at 40 r_s, 3.816e-03 at 100 r_s. E_inv (kappa 1, 2) over E: 9.80 percent at 14 r_s, 0.16 to 2.52 percent over 15 to 40, 0.42 to 1.43 percent over 42 to 100 (K = 18: 21.67 percent at 14, 0.45 to 1.46 percent over 42 to 100). The flux ratio (kappa 1, 2, slope comparator): -0.1451 to +0.0311 over 14 to 34 where the comparator exists (negative L at 9 of the 12 checkpoints 14 to 34; the comparator is absent at 18 r_s, where the record has no root), -0.0270 to +1.1985 over 36 to 40, +61.45 to +272.38 over 42 to 46, +3.07 to +26.15 over 48 to 60, -0.025 to +19.368 over 62 to 100 (negative at 1 of the 30 checkpoints from 42); the comparator L_H(kappa_T) is 1.89 to 3.28 times L_H(T_H) over 42 to 100. The join's largest relative deviation in v: 4.37e-06 to 7.12e-06 over 14 to 56, 5.84e-05 at 58, 4.57e-04 at 60, 8.70e-04 at 64, first above 1e-3 at 66 r_s (1.00e-03), 1.54e-02 at 80, 7.18e-02 at 100.

The propagation variant on the continuation: version 3's A-to-100 run (page_history_v6i, piecewise-constant, the first chain; its rows after 58 r_s carry the return marker) against this run's kappa 1, 2 ratio L/L_H(kappa_T) at 42 to 56 r_s: 42: 187.065 against 190.256 (-1.7 percent); 44: 278.914 against 272.382 (+2.4 percent); 46: 66.089 against 61.450 (+7.5 percent); 48: 12.446 against 12.652 (-1.6 percent); 50: 7.635 against 8.265 (-7.6 percent); 52: 26.008 against 26.155 (-0.6 percent); 54: 16.843 against 16.025 (+5.1 percent); 56: 3.524 against 3.075 (+14.6 percent); the largest difference 14.6 percent (at 56 r_s). At 36 and 38 r_s version 3's s4 run (v6b) read 4.0141 and 5.1180 against this run's 0.0179 and -0.0270.

E_out/E_kappa (the exterior share of the formed energy beyond the collar) per channel at 100 r_s: 1: 0.412, 2: 0.991, 3: 0.964, 4: 0.919, 5: 0.959, 6: 0.951, 7: 0.926, 8: 0.904, 9: 0.914, 10: 0.919, 11: 0.924, 12: 0.855, 13: 0.976, 14: 0.759, 15: 0.814, 16: 0.934, 17: 1.050, 18: 0.835 (values above 1 are possible: a region may carry negative excess energy).

The first-order term of W1 ruling v2 section 3 per channel at 100 r_s (unweighted; from the stored rows: d = (E_out - flow)/(16 kappa), the balance discrepancy between the exterior deviation energy beyond the collar and the accumulated collar flow, equal to W_exc - R, the moving fields' work on the exterior excess minus the held vacuum's rearrangement, neither measured separately here; E_kappa = E_inv/(16 kappa); the accumulated flow itself prints as an emitted energy nowhere beyond the kappa 1, 2 diagnostic above and stays in the audit data):

| kappa | 16 kappa E_kappa (weighted) | 16 kappa E_out (weighted) | d = (E_out - flow)/(16 kappa) | E_kappa (unweighted) | d/E_kappa | L at 100 (weighted) | L_H(kappa_T) at 100 |
|---|---|---|---|---|---|---|---|
| 1 | 6.438e-01 | 2.654e-01 | 4.526e-05 | 4.024e-02 | 0.001 | +2.582e-06 | 5.13e-06 |
| 2 | 1.415e-01 | 1.402e-01 | 4.146e-06 | 4.423e-03 | 0.001 | +7.063e-07 | 4.78e-07 |
| 3 | 2.898e-02 | 2.792e-02 | 1.243e-06 | 6.037e-04 | 0.002 | +2.606e-07 | 2.37e-08 |
| 4 | 3.883e-03 | 3.570e-03 | 8.379e-07 | 6.067e-05 | 0.014 | -2.997e-07 | 4.41e-10 |
| 5 | 1.410e-03 | 1.353e-03 | 7.623e-07 | 1.763e-05 | 0.043 | +1.361e-07 | 7.93e-13 |
| 6 | 5.548e-04 | 5.275e-04 | 7.361e-07 | 5.779e-06 | 0.127 | +8.702e-09 | 8.43e-16 |
| 7 | 2.451e-04 | 2.269e-04 | 7.309e-07 | 2.188e-06 | 0.334 | +1.135e-08 | 6.50e-19 |
| 8 | 1.694e-04 | 1.531e-04 | 7.310e-07 | 1.323e-06 | 0.552 | +2.640e-09 | 3.81e-22 |
| 9 | 1.576e-04 | 1.440e-04 | 7.317e-07 | 1.094e-06 | 0.669 | +6.957e-09 | 1.75e-25 |
| 10 | 1.546e-04 | 1.420e-04 | 7.321e-07 | 9.660e-07 | 0.758 | +2.055e-08 | 6.49e-29 |
| 11 | 1.515e-04 | 1.400e-04 | 7.325e-07 | 8.608e-07 | 0.851 | +7.191e-08 | 1.98e-32 |
| 12 | 1.474e-04 | 1.261e-04 | 7.326e-07 | 7.679e-07 | 0.954 | +3.401e-08 | 5.06e-36 |
| 13 | 1.382e-04 | 1.349e-04 | 7.326e-07 | 6.645e-07 | 1.103 | +7.867e-08 | 1.10e-39 |
| 14 | 1.256e-04 | 9.539e-05 | 7.325e-07 | 5.607e-07 | 1.306 | -1.283e-07 | 2.05e-43 |
| 15 | 1.107e-04 | 9.002e-05 | 7.323e-07 | 4.611e-07 | 1.588 | +4.848e-09 | 3.33e-47 |
| 16 | 9.494e-05 | 8.865e-05 | 7.320e-07 | 3.709e-07 | 1.974 | +2.278e-08 | 4.74e-51 |
| 17 | 8.349e-05 | 8.765e-05 | 7.318e-07 | 3.070e-07 | 2.384 | +9.353e-08 | 5.98e-55 |
| 18 | 7.043e-05 | 5.881e-05 | 7.314e-07 | 2.445e-07 | 2.991 | +2.371e-07 | 6.71e-59 |

d is 7.314e-07 to 7.326e-07 per channel over kappa 11 to 18 at 100 r_s and exceeds E_kappa from kappa 13 on (d/E_kappa 0.954 at kappa 12, 1.103 at kappa 13).

The signed collar split per checkpoint (kappa 1, 2; the collar_split block as stored: the energy flux through the collar window at the checkpoint split by the radial current's sign and by the threshold 25 T_H^ref; window 32 sites about the site 1025; weights 16 and 32 summed; the ratio |in|/|out| beside):

| T/r_s | E flux out (kappa 1, 2) | E flux in | \|in\|/\|out\| | out soft | in soft | out hard | in hard | kappa 1: out, in | kappa 2: out, in |
|---|---|---|---|---|---|---|---|---|---|
| 14 | +6.678e-09 | -1.486e-08 | 2.226 | +0.00e+00 | +4.97e-09 | +6.68e-09 | -1.98e-08 | -4.97e-10, -3.82e-09 | +7.18e-09, -1.10e-08 |
| 20 | -7.017e-08 | +2.283e-09 | 0.033 | +0.00e+00 | +2.01e-09 | -7.02e-08 | +2.70e-10 | +1.24e-08, +1.86e-09 | -8.26e-08, +4.24e-10 |
| 30 | -1.695e-08 | -1.824e-09 | 0.108 | +0.00e+00 | -2.39e-10 | -1.70e-08 | -1.58e-09 | -6.07e-09, -2.60e-09 | -1.09e-08, +7.78e-10 |
| 36 | +1.428e-07 | -3.870e-09 | 0.027 | +0.00e+00 | +1.97e-10 | +1.43e-07 | -4.07e-09 | +8.64e-09, -3.60e-09 | +1.34e-07, -2.66e-10 |
| 38 | -2.809e-08 | +9.614e-09 | 0.342 | +0.00e+00 | -7.42e-10 | -2.81e-08 | +1.04e-08 | +2.47e-07, +4.49e-09 | -2.75e-07, +5.12e-09 |
| 40 | +8.675e-05 | +1.640e-08 | 0.000 | +0.00e+00 | +1.61e-09 | +8.68e-05 | +1.48e-08 | +3.58e-05, +3.04e-09 | +5.10e-05, +1.34e-08 |
| 42 | +1.164e-03 | -4.907e-08 | 0.000 | +0.00e+00 | -4.66e-09 | +1.16e-03 | -4.44e-08 | +5.87e-04, -2.83e-08 | +5.77e-04, -2.08e-08 |
| 44 | +5.104e-04 | -5.331e-08 | 0.000 | +0.00e+00 | -1.29e-08 | +5.10e-04 | -4.04e-08 | +2.18e-04, -6.10e-08 | +2.93e-04, +7.73e-09 |
| 46 | -4.463e-06 | -2.129e-08 | 0.005 | +0.00e+00 | -4.58e-09 | -4.46e-06 | -1.67e-08 | +2.61e-05, +2.84e-08 | -3.05e-05, -4.97e-08 |
| 48 | +2.871e-05 | -4.604e-07 | 0.016 | +0.00e+00 | +3.82e-09 | +2.87e-05 | -4.64e-07 | +9.36e-06, -2.05e-07 | +1.93e-05, -2.55e-07 |
| 50 | +1.566e-04 | -5.669e-07 | 0.004 | +0.00e+00 | +1.05e-08 | +1.57e-04 | -5.77e-07 | +1.17e-04, -2.16e-07 | +3.96e-05, -3.51e-07 |
| 54 | +5.197e-05 | +5.900e-07 | 0.011 | +0.00e+00 | -3.31e-09 | +5.20e-05 | +5.93e-07 | +5.36e-05, +2.38e-07 | -1.65e-06, +3.52e-07 |
| 56 | -9.782e-06 | +2.290e-07 | 0.023 | +0.00e+00 | -1.04e-08 | -9.78e-06 | +2.39e-07 | -9.98e-06, +6.98e-08 | +1.97e-07, +1.59e-07 |
| 60 | +6.731e-05 | -1.120e-07 | 0.002 | +0.00e+00 | +1.51e-09 | +6.73e-05 | -1.13e-07 | +5.69e-05, -3.68e-08 | +1.04e-05, -7.51e-08 |
| 70 | +3.361e-06 | -9.091e-08 | 0.027 | +0.00e+00 | -2.80e-09 | +3.36e-06 | -8.81e-08 | +3.36e-06, -3.83e-08 | -1.63e-09, -5.26e-08 |
| 78 | -3.425e-06 | -1.140e-07 | 0.033 | +0.00e+00 | -5.22e-09 | -3.42e-06 | -1.09e-07 | -5.15e-06, -6.12e-08 | +1.73e-06, -5.28e-08 |
| 80 | +8.087e-06 | +9.285e-08 | 0.011 | +0.00e+00 | +1.54e-10 | +8.09e-06 | +9.27e-08 | +6.40e-06, +3.63e-08 | +1.69e-06, +5.66e-08 |
| 90 | +9.518e-06 | -4.690e-08 | 0.005 | +0.00e+00 | -9.39e-10 | +9.52e-06 | -4.60e-08 | +7.24e-06, -1.98e-08 | +2.28e-06, -2.71e-08 |
| 92 | -1.240e-06 | +3.455e-08 | 0.028 | +0.00e+00 | -2.90e-09 | -1.24e-06 | +3.75e-08 | +4.45e-06, +7.39e-09 | -5.69e-06, +2.72e-08 |
| 100 | +8.218e-06 | -2.834e-08 | 0.003 | +0.00e+00 | -2.50e-09 | +8.22e-06 | -2.58e-08 | +5.97e-06, -1.91e-08 | +2.25e-06, -9.24e-09 |

Over the checkpoints 40 to 100 r_s the ratio |in|/|out| of the kappa 1, 2 split runs 0.000 to 0.033; it is below 0.01 at 24 of the 31 checkpoints and above it at 7: 48 (0.016), 54 (0.011), 56 (0.023), 70 (0.027), 78 (0.033), 80 (0.011), 92 (0.028).

### T6. The marks

| mark | value | how obtained |
|---|---|---|
| the formation's record clock | T/r_s 12.125, R_t 38.556 a, E 116.6044 (2gE_f 58.3022 a) | the B2 record's first record with a trapping root |
| the trapped episodes of B2 | 12.125 to 17.25 r_s (42 records, R_t 37.93 to 52.02 a); 18.375 to 100 r_s (654 records, R_t 34.35 to 39.17 a) | runs of consecutive records with a root, from the record json (801 records to 100 r_s) |
| R_t over 2gE | 40.0 r_s: 0.684 (R_t 36.310 a, E/E0 0.8990); 56.0 r_s: 0.694 (R_t 36.789 a, E/E0 0.8973); 76.0 r_s: 0.687 (R_t 36.402 a, E/E0 0.8973); 100.0 r_s: 0.685 (R_t 36.307 a, E/E0 0.8977) | the record json |
| T_d | 2.31 r_s | <internal>/06_scaling.tex:338 (2.31 r_s at q = 1 and r_s/a = 64; T_d = (2/3) r_s ln[1/(2 chi q)] + O(r_s)) |
| the formation's light at the 2 r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock 12.70 to 14.62 r_s; shifted by the 16 r_s collar's recordwise offset at the nearest checkpoint: 12.75 to 15.00 (not a clock conversion at that radius); plus T_d on the record clock: 15.01 to 16.93 | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: 0.6324 and 2.7341) |
| the formation's light at the 16 r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock 35.08 to 37.00 r_s; shifted by the 16 r_s collar's recordwise offset at the nearest checkpoint: 35.57 to 37.49 (not a clock conversion at that radius); plus T_d on the record clock: 37.39 to 39.31 | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: 25.2004 and 27.3022) |
| the formation's light at the 24 r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock 45.27 to 47.18 r_s; shifted by the 16 r_s collar's recordwise offset at the nearest checkpoint: 45.78 to 47.69 (not a clock conversion at that radius); plus T_d on the record clock: 47.58 to 49.49 | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: 36.3833 and 38.4851) |
| the formation's light at the 30 r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock 52.62 to 54.54 r_s; shifted by the 16 r_s collar's recordwise offset at the nearest checkpoint: 53.13 to 55.04 (not a clock conversion at that radius); plus T_d on the record clock: 54.93 to 56.85 | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: 44.4549 and 46.5567) |
| the formation's light at the 48 r_s radius (fixed-charge flight comparison from 2 and 1.5 x 2gE_f) | record clock 73.90 to 75.81 r_s; shifted by the 16 r_s collar's recordwise offset at the nearest checkpoint: 74.41 to 76.32 (not a clock conversion at that radius); plus T_d on the record clock: 76.21 to 78.12 | 2 r_s,f [F(x1) - F(x0)], F(x) = x^2/2 + x + ln(x - 1), x = (R/r_s,f)^(1/2), r_s,f = 2gE_f at the formation record, from c = 2 and 1.5 to R; the version 3 formula (travel over 2gE_f: 67.8110 and 69.9128) |
| the half-collar-flow clock (kappa 1, 2) through 100 r_s | bracketed by the checkpoints 42 and 44 (the linear-interpolation estimate 43.04 r_s; the accumulated flow at 100: 4.0473e-01) | the clock at which the kappa 1, 2 accumulated signed collar flow reaches half of its value at the end clock, by linear interpolation between the stored checkpoints (2 r_s apart); the per-record flows are not stored in the v6o rows, so the clock is bracketed by the two checkpoints named |
| the half-collar-flow clock (kappa 1, 2) through 40 r_s | bracketed by 38 and 40 (the estimate 39.01 r_s; the flow at 40: 8.0263e-04); version 3's on the s4 run (per-record flows, v6b): 35.725 on the record clock, 36.214 on the collar clock | the same, through 40 only |
| the first return (estimate) | cut 2 r_s: 259.1 r_s; cut 16 r_s: 227.7 r_s; cut 24 r_s: 210.9 r_s; cut 30 r_s: 198.5 r_s; cut 48 r_s: 161.6 r_s | the continuum light round trip in the record's fields (outgoing N - /v/, ingoing N + /v/): from the cut at the run's start, and the formation's light from the root to the chain end and back to the cut; the earlier of the two; an estimate, not a measured arrival |
| the sigma/a 8 record | formation 12.75 r_s at R_t 44.53 a; episodes 12.75 to 16.25 (29 records); 26.625 to 29.625 (25 records); 38.875 to 40 (10 records); E0 118.2131 | the s8 record json |

### T7. The D13 protocol's stored observables on B2 (kappa 1 and 2, weights 16 and 32), rows before the first return

(a) The detector-energy ledger at the cuts 16 and 30 r_s per window (the windows ending at the readout clocks): stored per channel by page_d13_readout_v1d (the detector-energy ledger at the cuts 16 and 30 r_s over windows of at least 13 r_s: the deviation's Killing-energy inventory beyond the cut at the window's start and end, the crossing of the state's current and of the record jumps, the held vacuum's change; second_term_D = inventory_D_end - inventory_D_start; telescoping_residual = change_C - crossing_total); here the kappa 1 and 2 channels summed with the weights 16 and 32 (kappa 3 beside where marked 123). The agreement column is |crossing current - second term| over |second term|; the last column is the full-precision difference crossing jumps minus vacuum change, which is part of the balance and is kept.

| window T/r_s | cut | u_start, u_end (r_s) | inventory_D start | inventory_D end | second term (D end - D start) | crossing current (C) | agreement (percent) | crossing jumps (C) | vacuum change | jumps - vacuum change | kappa 1 second term | kappa 2 second term | with kappa 3: second term |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26 to 40 | 16 | 0.7, 14.7 | +9.279e-06 | +8.093e-04 | +8.0001e-04 | +8.0287e-04 | 0.36 | +6.0336e-01 | +6.0336e-01 | -2.861e-06 | +2.746e-04 | +5.254e-04 | +1.004e-03 |
| 26 to 40 | 30 | -16.7, -2.7 | +2.265e-06 | +1.372e-06 | -8.8984e-07 | -9.5515e-07 | 7.34 | +2.5962e-03 | +2.5961e-03 | +6.531e-08 | -1.953e-07 | -6.945e-07 | -5.474e-06 |
| 36 to 50 | 16 | 10.7, 24.7 | +7.468e-06 | +3.123e-01 | +3.1232e-01 | +3.1215e-01 | 0.05 | +5.4286e+00 | +5.4284e+00 | +1.673e-04 | +1.809e-01 | +1.314e-01 | +3.379e-01 |
| 36 to 50 | 30 | -6.7, 7.4 | +4.043e-06 | +1.952e-06 | -2.0915e-06 | -1.5173e-06 | 27.45 | -2.0379e-03 | -2.0373e-03 | -5.742e-07 | -4.713e-06 | +2.621e-06 | -7.115e-06 |
| 46 to 60 | 16 | 20.7, 34.7 | +2.993e-01 | +3.611e-01 | +6.1792e-02 | +6.1530e-02 | 0.42 | +8.2481e+01 | +8.2481e+01 | +2.626e-04 | +5.133e-02 | +1.046e-02 | +6.527e-02 |
| 46 to 60 | 30 | 3.4, 17.4 | -1.600e-06 | +1.845e-01 | +1.8451e-01 | +1.8451e-01 | 0.00 | +3.9530e-01 | +3.9530e-01 | +2.499e-07 | +1.143e-01 | +7.025e-02 | +1.957e-01 |
| 56 to 70 | 16 | 30.7, 44.7 | +3.444e-01 | +3.870e-01 | +4.2568e-02 | +4.2288e-02 | 0.66 | +1.1831e+02 | +1.1831e+02 | +2.790e-04 | +3.997e-02 | +2.596e-03 | +4.327e-02 |
| 56 to 70 | 30 | 13.4, 27.4 | +3.178e-06 | +3.355e-01 | +3.3548e-01 | +3.3547e-01 | 0.00 | +2.0766e+00 | +2.0766e+00 | +1.340e-05 | +2.010e-01 | +1.345e-01 | +3.618e-01 |
| 66 to 80 | 16 | 40.7, 54.7 | +3.813e-01 | +4.008e-01 | +1.9533e-02 | +1.9284e-02 | 1.27 | +1.0648e+02 | +1.0648e+02 | +2.487e-04 | +1.829e-02 | +1.238e-03 | +1.984e-02 |
| 66 to 80 | 30 | 23.4, 37.4 | +3.089e-01 | +3.650e-01 | +5.6056e-02 | +5.5982e-02 | 0.13 | +1.6530e+01 | +1.6530e+01 | +7.459e-05 | +4.946e-02 | +6.600e-03 | +5.772e-02 |
| 76 to 90 | 16 | 50.7, 64.7 | +3.991e-01 | +4.028e-01 | +3.7232e-03 | +3.5406e-03 | 4.90 | +9.5438e+01 | +9.5438e+01 | +1.826e-04 | +3.267e-03 | +4.561e-04 | +4.095e-03 |
| 76 to 90 | 30 | 33.4, 47.4 | +3.592e-01 | +3.954e-01 | +3.6219e-02 | +3.6151e-02 | 0.19 | +7.0021e+01 | +7.0021e+01 | +6.858e-05 | +3.492e-02 | +1.302e-03 | +3.674e-02 |
| 86 to 100 | 16 | 60.7, 74.7 | +4.018e-01 | +4.056e-01 | +3.7744e-03 | +3.6608e-03 | 3.01 | +8.5641e+01 | +8.5641e+01 | +1.137e-04 | +3.278e-03 | +4.966e-04 | +4.023e-03 |
| 86 to 100 | 30 | 43.4, 57.4 | +3.858e-01 | +4.015e-01 | +1.5770e-02 | +1.5676e-02 | 0.59 | +8.5470e+01 | +8.5470e+01 | +9.367e-05 | +1.481e-02 | +9.605e-04 | +1.616e-02 |

The deviation's inventory between the cuts 16 and 30 r_s at the window's start (kappa 1, 2; with kappa 3): 26: +7.014e-06 (-4.779e-07); 36: +3.425e-06 (+4.782e-06); 46: +2.993e-01 (+3.227e-01); 56: +3.444e-01 (+3.711e-01); 66: +7.237e-02 (+7.424e-02); 76: +3.990e-02 (+4.048e-02); 86: +1.604e-02 (+1.635e-02).

(b) L_old and L_sea by the 16 back-propagated probes at the cuts 16 and 30 r_s, every probe clock (40 to 100 every 4 r_s): L(t) = Tr[C(t) J_x(t)] = sum_a j_a <w_a(t_ref)/C(t_ref)/w_a(t_ref)>, the probes back-propagated through the stored fields without collection; L_old on C - P_-(t_ref), L_sea on P_-(t_ref), L_src on the source modes; the rank-16 bound //J_x//_1 beside. L_H(kappa_T) of kappa 1, 2 at the clock beside; before the first return at every row (the first return at the 16 r_s cut is estimated at 227.7 r_s, at 30 r_s 198.5). The ratio columns are |L_old| over |L| (the direct current) and, at the reference clock 12 r_s, |L_sea - L| over |L|.

| T/r_s | cut | L (direct, kappa 1, 2) | t_ref | L_old | L_sea | L_src | \|L_old\|/\|L\| | \|L_sea - L\|/\|L\| | kappa 1: L_old, L_sea | kappa 2: L_old, L_sea | L_H(kappa_T) (kappa 1, 2) | join max rel dv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | 16 | +7.521e-05 | 40 | +7.521e-05 | +3.841e-15 | +1.368e-08 | 1.00 | 1.000 | +3.31e-05, -8.03e-16 | +4.21e-05, +4.64e-15 | 5.281e-06 | 4.45e-06 |
| 40 | 16 | +7.521e-05 | 20 | +7.526e-05 | -4.319e-08 | +1.118e-07 | 1.00 | 1.001 | +3.31e-05, -2.80e-08 | +4.21e-05, -1.52e-08 | 5.281e-06 | 4.45e-06 |
| 40 | 16 | +7.521e-05 | 12 | +1.003e-06 | +7.421e-05 | -3.048e-01 | 0.01 | 0.013 | +1.35e-06, +3.17e-05 | -3.47e-07, +4.25e-05 | 5.281e-06 | 4.45e-06 |
| 40 | 30 | -8.717e-09 | 40 | -8.717e-09 | +5.220e-15 | +1.849e-12 | 1.00 | 1.000 | +4.10e-09, +1.75e-16 | -1.28e-08, +5.05e-15 | 5.281e-06 | 4.45e-06 |
| 40 | 30 | -8.717e-09 | 20 | -8.766e-09 | +4.904e-11 | +2.074e-12 | 1.01 | 1.006 | +3.92e-09, +1.77e-10 | -1.27e-08, -1.28e-10 | 5.281e-06 | 4.45e-06 |
| 40 | 30 | -8.717e-09 | 12 | -8.405e-10 | -7.877e-09 | +2.359e-12 | 0.10 | 0.096 | +1.12e-09, +2.97e-09 | -1.96e-09, -1.09e-08 | 5.281e-06 | 4.45e-06 |
| 44 | 16 | +6.141e-04 | 40 | +6.140e-04 | +2.522e-08 | +7.273e-06 | 1.00 | 1.000 | +2.03e-04, +8.63e-09 | +4.11e-04, +1.66e-08 | 3.508e-06 | 4.37e-06 |
| 44 | 16 | +6.141e-04 | 20 | +6.095e-04 | +4.532e-06 | -8.852e-01 | 0.99 | 0.993 | +2.04e-04, -6.11e-07 | +4.06e-04, +5.14e-06 | 3.508e-06 | 4.37e-06 |
| 44 | 16 | +6.141e-04 | 12 | +1.533e-05 | +5.987e-04 | -2.179e-01 | 0.02 | 0.025 | +1.09e-05, +1.92e-04 | +4.42e-06, +4.06e-04 | 3.508e-06 | 4.37e-06 |
| 44 | 30 | -3.082e-07 | 40 | -3.081e-07 | -6.592e-11 | +1.055e-11 | 1.00 | 1.000 | -1.03e-07, -1.96e-11 | -2.05e-07, -4.63e-11 | 3.508e-06 | 4.37e-06 |
| 44 | 30 | -3.082e-07 | 20 | -3.080e-07 | -2.453e-10 | +1.255e-11 | 1.00 | 0.999 | -1.03e-07, +2.72e-10 | -2.05e-07, -5.17e-10 | 3.508e-06 | 4.37e-06 |
| 44 | 30 | -3.082e-07 | 12 | -3.019e-07 | -6.308e-09 | +1.324e-11 | 0.98 | 0.980 | -1.04e-07, +9.62e-10 | -1.98e-07, -7.27e-09 | 3.508e-06 | 4.37e-06 |
| 48 | 16 | +2.918e-05 | 40 | +2.789e-05 | +1.290e-06 | +3.431e-07 | 0.96 | 0.956 | +8.67e-06, +4.63e-07 | +1.92e-05, +8.27e-07 | 5.201e-06 | 4.42e-06 |
| 48 | 16 | +2.918e-05 | 20 | +2.653e-05 | +2.653e-06 | -3.001e-01 | 0.91 | 0.909 | +8.16e-06, +9.74e-07 | +1.84e-05, +1.68e-06 | 5.201e-06 | 4.42e-06 |
| 48 | 16 | +2.918e-05 | 12 | -1.333e-08 | +2.919e-05 | -1.469e-01 | 0.00 | 0.000 | +1.51e-07, +8.98e-06 | -1.65e-07, +2.02e-05 | 5.201e-06 | 4.42e-06 |
| 48 | 30 | -4.184e-08 | 40 | -4.180e-08 | -3.779e-11 | +1.886e-11 | 1.00 | 0.999 | -1.10e-08, -1.98e-11 | -3.08e-08, -1.79e-11 | 5.201e-06 | 4.42e-06 |
| 48 | 30 | -4.184e-08 | 20 | -4.127e-08 | -5.683e-10 | +2.096e-10 | 0.99 | 0.986 | -1.14e-08, +3.34e-10 | -2.99e-08, -9.02e-10 | 5.201e-06 | 4.42e-06 |
| 48 | 30 | -4.184e-08 | 12 | -6.706e-08 | +2.522e-08 | +1.340e-10 | 1.60 | 1.603 | -1.42e-08, +3.16e-09 | -5.29e-08, +2.21e-08 | 5.201e-06 | 4.42e-06 |
| 52 | 16 | +1.838e-04 | 40 | +1.831e-04 | +7.529e-07 | +3.862e-06 | 1.00 | 0.996 | +1.76e-04, +3.16e-07 | +6.99e-06, +4.37e-07 | 5.666e-06 | 4.62e-06 |
| 52 | 16 | +1.838e-04 | 20 | -2.099e-02 | +2.117e-02 | -3.821e-02 | 114.16 | 114.163 | -6.41e-03, +6.58e-03 | -1.46e-02, +1.46e-02 | 5.666e-06 | 4.62e-06 |
| 52 | 16 | +1.838e-04 | 12 | +7.700e-07 | +1.831e-04 | -6.114e-01 | 0.00 | 0.004 | +9.00e-07, +1.76e-04 | -1.30e-07, +7.56e-06 | 5.666e-06 | 4.62e-06 |
| 52 | 30 | +1.384e-06 | 40 | +1.388e-06 | -3.645e-09 | +1.351e-10 | 1.00 | 1.003 | +3.69e-07, -1.35e-09 | +1.02e-06, -2.29e-09 | 5.666e-06 | 4.62e-06 |
| 52 | 30 | +1.384e-06 | 20 | +1.571e-06 | -1.869e-07 | +1.743e-09 | 1.14 | 1.135 | +2.81e-07, +8.66e-08 | +1.29e-06, -2.74e-07 | 5.666e-06 | 4.62e-06 |
| 52 | 30 | +1.384e-06 | 12 | +6.259e-07 | +7.581e-07 | -1.945e+00 | 0.45 | 0.452 | -2.80e-09, +3.71e-07 | +6.29e-07, +3.88e-07 | 5.666e-06 | 4.62e-06 |
| 56 | 16 | -8.447e-06 | 40 | -8.397e-06 | -4.982e-08 | +1.849e-06 | 0.99 | 0.994 | -1.13e-05, -3.07e-08 | +2.89e-06, -1.91e-08 | 4.758e-06 | 7.12e-06 |
| 56 | 16 | -8.447e-06 | 20 | -1.356e-03 | +1.348e-03 | -1.514e-02 | 160.57 | 160.573 | -2.35e-04, +2.24e-04 | -1.12e-03, +1.12e-03 | 4.758e-06 | 7.12e-06 |
| 56 | 16 | -8.447e-06 | 12 | +3.499e-08 | -8.482e-06 | -2.693e-01 | 0.00 | 0.004 | +2.35e-08, -1.13e-05 | +1.15e-08, +2.86e-06 | 4.758e-06 | 7.12e-06 |
| 56 | 30 | -3.011e-07 | 40 | -3.042e-07 | +3.030e-09 | +1.450e-09 | 1.01 | 1.010 | -2.39e-08, +2.20e-09 | -2.80e-07, +8.34e-10 | 4.758e-06 | 7.12e-06 |
| 56 | 30 | -3.011e-07 | 20 | -3.146e-07 | +1.346e-08 | +1.539e-08 | 1.04 | 1.045 | -1.96e-08, -2.05e-09 | -2.95e-07, +1.55e-08 | 4.758e-06 | 7.12e-06 |
| 56 | 30 | -3.011e-07 | 12 | -1.578e-06 | +1.277e-06 | -5.293e-01 | 5.24 | 5.240 | -4.27e-07, +4.06e-07 | -1.15e-06, +8.71e-07 | 4.758e-06 | 7.12e-06 |
| 60 | 16 | +7.036e-05 | 40 | +7.035e-05 | +6.767e-09 | +3.439e-06 | 1.00 | 1.000 | +5.93e-05, -4.56e-09 | +1.10e-05, +1.13e-08 | 5.783e-06 | 4.57e-04 |
| 60 | 16 | +7.036e-05 | 20 | -1.315e-03 | +1.386e-03 | -3.011e-02 | 18.69 | 18.695 | -2.16e-04, +2.75e-04 | -1.10e-03, +1.11e-03 | 5.783e-06 | 4.57e-04 |
| 60 | 16 | +7.036e-05 | 12 | +2.216e-08 | +7.034e-05 | -1.158e-01 | 0.00 | 0.000 | +2.11e-08, +5.93e-05 | +1.09e-09, +1.10e-05 | 5.783e-06 | 4.57e-04 |
| 60 | 30 | +2.262e-03 | 40 | +2.262e-03 | -2.867e-09 | +6.402e-06 | 1.00 | 1.000 | +1.23e-03, -1.72e-09 | +1.03e-03, -1.15e-09 | 5.783e-06 | 4.57e-04 |
| 60 | 30 | +2.262e-03 | 20 | +2.342e-03 | -8.011e-05 | -7.443e-01 | 1.04 | 1.035 | +1.26e-03, -2.59e-05 | +1.09e-03, -5.42e-05 | 5.783e-06 | 4.57e-04 |
| 60 | 30 | +2.262e-03 | 12 | +2.171e-04 | +2.045e-03 | -3.617e-01 | 0.10 | 0.096 | +1.22e-04, +1.11e-03 | +9.50e-05, +9.37e-04 | 5.783e-06 | 4.57e-04 |
| 64 | 16 | +4.746e-05 | 40 | +4.662e-05 | +8.441e-07 | -8.136e-01 | 0.98 | 0.982 | +4.51e-05, +2.30e-07 | +1.50e-06, +6.14e-07 | 4.960e-06 | 8.70e-04 |
| 64 | 16 | +4.746e-05 | 20 | -1.542e-03 | +1.590e-03 | -3.952e-02 | 32.49 | 32.495 | -8.47e-04, +8.92e-04 | -6.95e-04, +6.97e-04 | 4.960e-06 | 8.70e-04 |
| 64 | 16 | +4.746e-05 | 12 | +7.024e-08 | +4.739e-05 | -8.517e-02 | 0.00 | 0.001 | +6.80e-08, +4.53e-05 | +2.26e-09, +2.11e-06 | 4.960e-06 | 8.70e-04 |
| 64 | 30 | +1.450e-04 | 40 | +1.455e-04 | -4.722e-07 | +7.626e-07 | 1.00 | 1.003 | +7.54e-05, -3.08e-07 | +7.01e-05, -1.64e-07 | 4.960e-06 | 8.70e-04 |
| 64 | 30 | +1.450e-04 | 20 | +1.395e-04 | +5.565e-06 | -4.719e-01 | 0.96 | 0.962 | +7.44e-05, +6.37e-07 | +6.50e-05, +4.93e-06 | 4.960e-06 | 8.70e-04 |
| 64 | 30 | +1.450e-04 | 12 | -5.007e-07 | +1.455e-04 | -1.737e-01 | 0.00 | 0.003 | -3.92e-07, +7.55e-05 | -1.09e-07, +7.01e-05 | 4.960e-06 | 8.70e-04 |
| 68 | 16 | -4.465e-06 | 40 | -3.317e-05 | +2.870e-05 | -2.259e-01 | 7.43 | 7.428 | -1.04e-05, +6.16e-06 | -2.28e-05, +2.25e-05 | 5.603e-06 | 1.19e-03 |
| 68 | 16 | -4.465e-06 | 20 | -2.433e-05 | +1.987e-05 | -1.245e-02 | 5.45 | 5.450 | -2.40e-05, +1.98e-05 | -3.39e-07, +1.13e-07 | 5.603e-06 | 1.19e-03 |
| 68 | 16 | -4.465e-06 | 12 | -7.153e-09 | -4.458e-06 | -6.552e-02 | 0.00 | 0.002 | +2.90e-09, -4.24e-06 | -1.00e-08, -2.16e-07 | 5.603e-06 | 1.19e-03 |
| 68 | 30 | -3.254e-04 | 40 | -3.253e-04 | -1.475e-07 | +2.860e-06 | 1.00 | 1.000 | -1.44e-04, -1.30e-07 | -1.82e-04, -1.70e-08 | 5.603e-06 | 1.19e-03 |
| 68 | 30 | -3.254e-04 | 20 | -1.092e-02 | +1.059e-02 | -2.009e-01 | 33.55 | 33.550 | -3.62e-03, +3.48e-03 | -7.30e-03, +7.11e-03 | 5.603e-06 | 1.19e-03 |
| 68 | 30 | -3.254e-04 | 12 | +2.249e-07 | -3.256e-04 | -1.334e-01 | 0.00 | 0.001 | +4.31e-07, -1.44e-04 | -2.06e-07, -1.81e-04 | 5.603e-06 | 1.19e-03 |
| 72 | 16 | +3.269e-05 | 40 | -1.253e-03 | +1.286e-03 | -8.951e-02 | 38.34 | 38.343 | -3.69e-04, +3.98e-04 | -8.85e-04, +8.88e-04 | 5.311e-06 | 1.45e-03 |
| 72 | 16 | +3.269e-05 | 20 | -3.309e-04 | +3.636e-04 | -2.500e-03 | 10.12 | 10.123 | -8.89e-05, +1.18e-04 | -2.42e-04, +2.45e-04 | 5.311e-06 | 1.45e-03 |
| 72 | 16 | +3.269e-05 | 12 | -4.204e-09 | +3.269e-05 | -4.972e-02 | 0.00 | 0.000 | -9.20e-10, +2.93e-05 | -3.28e-09, +3.39e-06 | 5.311e-06 | 1.45e-03 |
| 72 | 30 | +2.589e-05 | 40 | +2.584e-05 | +4.432e-08 | +1.053e-06 | 1.00 | 0.998 | +8.76e-06, -4.54e-08 | +1.71e-05, +8.97e-08 | 5.311e-06 | 1.45e-03 |
| 72 | 30 | +2.589e-05 | 20 | -3.234e-03 | +3.260e-03 | -1.464e-01 | 124.94 | 124.937 | -6.75e-04, +6.84e-04 | -2.56e-03, +2.58e-03 | 5.311e-06 | 1.45e-03 |
| 72 | 30 | +2.589e-05 | 12 | +7.656e-09 | +2.588e-05 | -9.157e-02 | 0.00 | 0.000 | +2.65e-08, +8.69e-06 | -1.88e-08, +1.72e-05 | 5.311e-06 | 1.45e-03 |
| 76 | 16 | -2.186e-05 | 40 | +2.632e-05 | -4.819e-05 | -5.402e-02 | 1.20 | 1.204 | +9.84e-06, -2.30e-05 | +1.65e-05, -2.52e-05 | 5.602e-06 | 2.86e-03 |
| 76 | 16 | -2.186e-05 | 20 | -1.149e-05 | -1.037e-05 | -1.424e-03 | 0.53 | 0.526 | -6.60e-06, -6.58e-06 | -4.89e-06, -3.79e-06 | 5.602e-06 | 2.86e-03 |
| 76 | 16 | -2.186e-05 | 12 | +4.787e-10 | -2.186e-05 | -4.220e-02 | 0.00 | 0.000 | +1.03e-09, -1.32e-05 | -5.55e-10, -8.69e-06 | 5.602e-06 | 2.86e-03 |
| 76 | 30 | +1.166e-04 | 40 | +1.167e-04 | -1.065e-07 | +1.504e-05 | 1.00 | 1.001 | +1.25e-04, +1.15e-07 | -8.42e-06, -2.21e-07 | 5.602e-06 | 2.86e-03 |
| 76 | 30 | +1.166e-04 | 20 | -9.322e-04 | +1.049e-03 | -5.142e-02 | 8.00 | 7.997 | -1.22e-04, +2.47e-04 | -8.10e-04, +8.01e-04 | 5.602e-06 | 2.86e-03 |
| 76 | 30 | +1.166e-04 | 12 | +1.409e-07 | +1.164e-04 | -1.015e-01 | 0.00 | 0.001 | +1.53e-07, +1.25e-04 | -1.19e-08, -8.63e-06 | 5.602e-06 | 2.86e-03 |
| 80 | 16 | +9.784e-06 | 40 | -3.606e-04 | +3.704e-04 | -3.513e-02 | 36.86 | 36.857 | -5.45e-05, +6.16e-05 | -3.06e-04, +3.09e-04 | 5.815e-06 | 1.54e-02 |
| 80 | 16 | +9.784e-06 | 20 | -5.565e-05 | +6.543e-05 | -1.303e-03 | 5.69 | 5.688 | -3.18e-06, +1.03e-05 | -5.25e-05, +5.51e-05 | 5.815e-06 | 1.54e-02 |
| 80 | 16 | +9.784e-06 | 12 | +2.174e-09 | +9.782e-06 | -2.056e-02 | 0.00 | 0.000 | +4.07e-11, +7.17e-06 | +2.13e-09, +2.62e-06 | 5.815e-06 | 1.54e-02 |
| 80 | 30 | +5.574e-05 | 40 | +5.439e-05 | +1.352e-06 | -9.528e-01 | 0.98 | 0.976 | +5.11e-05, +3.56e-07 | +3.32e-06, +9.95e-07 | 5.815e-06 | 1.54e-02 |
| 80 | 30 | +5.574e-05 | 20 | -1.261e-03 | +1.316e-03 | -1.059e-02 | 22.62 | 22.618 | -2.14e-04, +2.65e-04 | -1.05e-03, +1.05e-03 | 5.815e-06 | 1.54e-02 |
| 80 | 30 | +5.574e-05 | 12 | -4.025e-09 | +5.574e-05 | -4.595e-01 | 0.00 | 0.000 | -5.35e-09, +5.14e-05 | +1.32e-09, +4.32e-06 | 5.815e-06 | 1.54e-02 |
| 84 | 16 | -3.511e-06 | 40 | -6.075e-04 | +6.040e-04 | -2.856e-02 | 173.04 | 173.039 | -5.03e-05, +4.60e-05 | -5.57e-04, +5.58e-04 | 5.723e-06 | 3.83e-02 |
| 84 | 16 | -3.511e-06 | 20 | -1.031e-04 | +9.962e-05 | -1.244e-03 | 29.38 | 29.375 | +7.85e-07, -5.04e-06 | -1.04e-04, +1.05e-04 | 5.723e-06 | 3.83e-02 |
| 84 | 16 | -3.511e-06 | 12 | +1.100e-09 | -3.512e-06 | -1.509e-02 | 0.00 | 0.000 | +8.69e-10, -4.25e-06 | +2.31e-10, +7.42e-07 | 5.723e-06 | 3.83e-02 |
| 84 | 30 | +4.073e-05 | 40 | +3.509e-05 | +5.639e-06 | -4.221e-01 | 0.86 | 0.862 | +3.88e-05, +1.37e-06 | -3.75e-06, +4.27e-06 | 5.723e-06 | 3.83e-02 |
| 84 | 30 | +4.073e-05 | 20 | -7.121e-05 | +1.119e-04 | -5.657e-03 | 1.75 | 1.748 | -5.23e-05, +9.25e-05 | -1.89e-05, +1.94e-05 | 5.723e-06 | 3.83e-02 |
| 84 | 30 | +4.073e-05 | 12 | +5.902e-09 | +4.072e-05 | -3.148e-01 | 0.00 | 0.000 | +9.20e-09, +4.02e-05 | -3.30e-09, +5.15e-07 | 5.723e-06 | 3.83e-02 |
| 88 | 16 | +2.246e-05 | 40 | -2.938e-04 | +3.162e-04 | -2.139e-02 | 13.08 | 13.080 | -3.44e-05, +5.06e-05 | -2.59e-04, +2.66e-04 | 5.616e-06 | 5.71e-02 |
| 88 | 16 | +2.246e-05 | 20 | -5.251e-05 | +7.497e-05 | -1.142e-03 | 2.34 | 2.338 | -9.86e-07, +1.72e-05 | -5.15e-05, +5.78e-05 | 5.616e-06 | 5.71e-02 |
| 88 | 16 | +2.246e-05 | 12 | +1.043e-09 | +2.246e-05 | -1.439e-02 | 0.00 | 0.000 | -2.20e-11, +1.62e-05 | +1.07e-09, +6.26e-06 | 5.616e-06 | 5.71e-02 |
| 88 | 30 | +8.522e-05 | 40 | -1.176e-03 | +1.261e-03 | -1.432e-01 | 13.80 | 13.800 | -3.62e-04, +4.38e-04 | -8.14e-04, +8.23e-04 | 5.616e-06 | 5.71e-02 |
| 88 | 30 | +8.522e-05 | 20 | -2.670e-04 | +3.522e-04 | -5.058e-03 | 3.13 | 3.133 | -4.89e-05, +1.24e-04 | -2.18e-04, +2.28e-04 | 5.616e-06 | 5.71e-02 |
| 88 | 30 | +8.522e-05 | 12 | -4.710e-09 | +8.522e-05 | -2.287e-01 | 0.00 | 0.000 | -2.65e-09, +7.56e-05 | -2.06e-09, +9.66e-06 | 5.616e-06 | 5.71e-02 |
| 92 | 16 | -2.576e-06 | 40 | -1.564e-04 | +1.538e-04 | -1.643e-02 | 60.72 | 60.723 | +4.31e-06, -1.66e-07 | -1.61e-04, +1.54e-04 | 5.997e-06 | 6.60e-02 |
| 92 | 16 | -2.576e-06 | 20 | -3.085e-05 | +2.828e-05 | -7.797e-04 | 11.98 | 11.978 | -5.02e-06, +9.16e-06 | -2.58e-05, +1.91e-05 | 5.997e-06 | 6.60e-02 |
| 92 | 16 | -2.576e-06 | 12 | +1.459e-10 | -2.576e-06 | -1.389e-02 | 0.00 | 0.000 | -1.66e-10, +4.14e-06 | +3.12e-10, -6.72e-06 | 5.997e-06 | 6.60e-02 |
| 92 | 30 | +2.062e-05 | 40 | -3.886e-04 | +4.092e-04 | -9.212e-02 | 18.84 | 18.845 | -8.44e-05, +1.04e-04 | -3.04e-04, +3.06e-04 | 5.997e-06 | 6.60e-02 |
| 92 | 30 | +2.062e-05 | 20 | -1.364e-04 | +1.570e-04 | -4.538e-03 | 6.61 | 6.613 | -3.22e-05, +5.14e-05 | -1.04e-04, +1.06e-04 | 5.997e-06 | 6.60e-02 |
| 92 | 30 | +2.062e-05 | 12 | +1.508e-09 | +2.062e-05 | -1.473e-01 | 0.00 | 0.000 | +1.84e-10, +1.92e-05 | +1.32e-09, +1.39e-06 | 5.997e-06 | 6.60e-02 |
| 96 | 16 | -3.796e-06 | 40 | -1.480e-04 | +1.443e-04 | -1.333e-02 | 39.00 | 38.997 | -2.56e-05, +2.74e-05 | -1.22e-04, +1.17e-04 | 5.266e-06 | 6.98e-02 |
| 96 | 16 | -3.796e-06 | 20 | -1.949e-05 | +1.569e-05 | -8.625e-04 | 5.13 | 5.133 | -2.13e-06, +3.86e-06 | -1.74e-05, +1.18e-05 | 5.266e-06 | 6.98e-02 |
| 96 | 16 | -3.796e-06 | 12 | -2.878e-10 | -3.796e-06 | -1.198e-02 | 0.00 | 0.000 | -3.40e-10, +1.73e-06 | +5.18e-11, -5.53e-06 | 5.266e-06 | 6.98e-02 |
| 96 | 30 | +5.786e-06 | 40 | -5.370e-04 | +5.428e-04 | -6.676e-02 | 92.81 | 92.806 | -9.51e-05, +1.01e-04 | -4.42e-04, +4.42e-04 | 5.266e-06 | 6.98e-02 |
| 96 | 30 | +5.786e-06 | 20 | -7.809e-05 | +8.388e-05 | -4.955e-03 | 13.50 | 13.496 | -4.71e-06, +1.09e-05 | -7.34e-05, +7.30e-05 | 5.266e-06 | 6.98e-02 |
| 96 | 30 | +5.786e-06 | 12 | +5.696e-10 | +5.786e-06 | -1.127e-01 | 0.00 | 0.000 | +4.48e-10, +6.20e-06 | +1.22e-10, -4.13e-07 | 5.266e-06 | 6.98e-02 |
| 100 | 16 | +8.900e-06 | 40 | -4.001e-04 | +4.090e-04 | -9.754e-03 | 44.95 | 44.952 | -1.97e-05, +2.60e-05 | -3.80e-04, +3.83e-04 | 5.604e-06 | 7.18e-02 |
| 100 | 16 | +8.900e-06 | 20 | -8.748e-05 | +9.638e-05 | -5.697e-04 | 9.83 | 9.829 | -5.03e-06, +1.14e-05 | -8.24e-05, +8.50e-05 | 5.604e-06 | 7.18e-02 |
| 100 | 16 | +8.900e-06 | 12 | -9.812e-10 | +8.901e-06 | -9.398e-03 | 0.00 | 0.000 | +7.65e-11, +6.33e-06 | -1.06e-09, +2.57e-06 | 5.604e-06 | 7.18e-02 |
| 100 | 30 | +1.390e-05 | 40 | -1.287e-04 | +1.426e-04 | -4.804e-02 | 9.26 | 9.262 | +5.40e-06, +5.28e-06 | -1.34e-04, +1.37e-04 | 5.604e-06 | 7.18e-02 |
| 100 | 30 | +1.390e-05 | 20 | -1.808e-05 | +3.198e-05 | -7.234e-03 | 1.30 | 1.301 | -7.59e-07, +1.14e-05 | -1.73e-05, +2.05e-05 | 5.604e-06 | 7.18e-02 |
| 100 | 30 | +1.390e-05 | 12 | +1.940e-07 | +1.371e-05 | -7.495e-02 | 0.01 | 0.014 | +7.82e-08, +1.06e-05 | +1.16e-07, +3.10e-06 | 5.604e-06 | 7.18e-02 |

Over all 32 probes (16 clocks 40 to 100 every 4 r_s, 2 cuts): with the reference clock 12 r_s |L_old| exceeds 2e-7 at 9 of the 32 and is at most 2.2e-04 (at 60 r_s, the 30 r_s cut, where L is 2.3e-03); |L_sea - L|/|L| is at most 5.24 (at 56 r_s, the 30 r_s cut, where L is -3.0e-07) and at most 0.096 over the 19 probes where |L| exceeds 1e-5 (there |L_old|/|L| is at most 0.096); with the reference clock 40 r_s at the 16 r_s cut |L_old|/|L| is 1.2 to 173.0 over 72 to 100 r_s (every 4 r_s), and with the reference clock 20 r_s it is 0.5 to 160.6 over 56 to 100 r_s (161 at 56 r_s, where the join's departure is 7.1e-06).

(c) Windowed changes of the hard and soft exterior-shell energy inventories per retarded window (the windows ending at the readout clocks): stored per channel: the exterior Killing shells' energy change per retarded window split by the radial current's sign, above (hard) and below (soft) the threshold 25 T_H^ref (hard_threshold_energy), and above the band top (above); the _P entries are the held vacuum's; weights 16 and 32 summed; the threshold 25 T_H^ref = 3.7464e-02 (T_H^ref 1.4986e-03). These are signed inventory changes of the exterior Killing shells, not a crossing through a fixed detector.

| window T/r_s | u_start, u_end | hard E out | hard E in | out + in (hard) | soft E out | soft E in | above-band E out | above-band E in | hard N out | hard E out (held vacuum) | soft E out (held vacuum) | with kappa 3: hard E out, soft E out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 26 to 40 | -9.4, 4.6 | +3.202e-01 | -3.216e-01 | -1.45e-03 | +2.945e-03 | -2.874e-03 | -1.37e-04 | -1.46e-04 | -2.336e-02 | +1.94e-07 | -1.95e-06 | +6.404e-01, +7.730e-04 |
| 36 to 50 | 0.6, 14.6 | +1.374e-01 | -1.383e-01 | -8.19e-04 | +6.656e-04 | -5.996e-04 | -9.89e-05 | -1.05e-04 | -6.098e-02 | +2.58e-04 | +1.11e-05 | +7.132e-01, +3.083e-03 |
| 46 to 60 | 10.6, 24.6 | +3.807e-01 | -3.601e-02 | +3.45e-01 | +1.079e-03 | +9.161e-03 | -1.35e-04 | -1.42e-04 | -5.515e-02 | +3.34e-01 | +4.58e-03 | +4.563e-01, -1.043e-03 |
| 56 to 70 | 20.6, 34.6 | -1.462e-01 | +1.530e-01 | +6.74e-03 | +2.287e-02 | +1.054e-02 | -3.67e-03 | -3.83e-03 | +6.879e-02 | +3.81e-02 | +1.79e-02 | -7.668e-01, +2.552e-02 |
| 66 to 80 | 30.6, 44.6 | +2.711e-02 | -4.163e-02 | -1.45e-02 | +2.581e-02 | +4.577e-03 | -4.51e-03 | -4.70e-03 | -8.913e-03 | +1.08e-02 | +2.03e-02 | +4.393e-02, +2.343e-02 |
| 76 to 90 | 40.6, 54.6 | +3.941e-01 | -4.019e-01 | -7.86e-03 | +1.049e-02 | +1.424e-02 | -1.04e-03 | -1.06e-03 | -5.612e-02 | +5.31e-03 | +1.80e-02 | +7.846e-01, +9.901e-03 |
| 86 to 100 | 50.6, 64.6 | -1.425e-01 | +1.404e-01 | -2.18e-03 | +1.125e-02 | -3.306e-03 | -2.50e-04 | -2.61e-04 | +1.085e-01 | +4.68e-03 | +1.23e-02 | -2.918e-01, +2.079e-02 |

The identified source modes (the eigenvectors of C inside the region at the reference clock with eigenvalue at least 0.99, propagated with the sea without collection) and their same-packet transfer tau (summed over the set, its maximum over the set, and the average tau sum / modes) and inferred occupation at the collar (16 r_s), the cut (24 r_s) and the far radius (48 r_s), per channel; the last column is the set's occupation in the held vacuum of the reference clock (the stored occ_vac_extremes, smallest and largest over the set):

| kappa | reference clock (region, modes) | at T/r_s | tau sum collar | tau max collar | tau average collar | occupation (collar) | tau sum cut | tau average cut | occupation (cut) | tau sum far | occupation (far) | vacuum occupation of the set (min, max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 12 (inside 2gE (no root at the reference clock); 55 modes) | 40 | 30.241 | 0.9947 | 0.550 | 0.9960 | 0.000 | 0.000 | 0.3769 | 2.234e-08 | 0.4698 | 0.999956, 1.000000 |
| 1 | 20 (inside the outer root; 34 modes) | 40 | 0.000 | 0.0000 | 0.000 | 0.1290 | 0.000 | 0.000 | 0.2524 | 4.489e-08 | 0.4659 | 0.484753, 1.000000 |
| 1 | 40 (inside the outer root; 32 modes) | 40 | 0.000 | 0.0000 | 0.000 | 0.3826 | 0.000 | 0.000 | 0.4963 | 1.876e-07 | 0.4906 | 0.243767, 1.000000 |
| 1 | 12 (inside 2gE (no root at the reference clock); 55 modes) | 60 | 49.654 | 0.9995 | 0.903 | 0.9997 | 39.748 | 0.723 | 0.9994 | 1.288e-07 | 0.4449 | 0.999956, 1.000000 |
| 1 | 20 (inside the outer root; 34 modes) | 60 | 28.898 | 0.9981 | 0.850 | 0.9983 | 24.392 | 0.717 | 0.9946 | 1.833e-07 | 0.4612 | 0.484753, 1.000000 |
| 1 | 40 (inside the outer root; 32 modes) | 60 | 0.006 | 0.0045 | 0.000 | 0.0141 | 0.001 | 0.000 | 0.0294 | 2.693e-07 | 0.4945 | 0.243767, 1.000000 |
| 1 | 12 (inside 2gE (no root at the reference clock); 55 modes) | 80 | 51.634 | 0.9998 | 0.939 | 0.9999 | 48.848 | 0.888 | 0.9998 | 2.506e+01 | 0.9949 | 0.999956, 1.000000 |
| 1 | 20 (inside the outer root; 34 modes) | 80 | 30.020 | 0.9989 | 0.883 | 0.9995 | 29.374 | 0.864 | 0.9993 | 1.395e-04 | 0.1802 | 0.484753, 1.000000 |
| 1 | 40 (inside the outer root; 32 modes) | 80 | 26.630 | 0.9928 | 0.832 | 0.9979 | 23.192 | 0.725 | 0.9934 | 1.217e-05 | 0.1548 | 0.243767, 1.000000 |
| 1 | 12 (inside 2gE (no root at the reference clock); 55 modes) | 100 | 52.022 | 0.9998 | 0.946 | 0.9999 | 50.744 | 0.923 | 0.9999 | 3.926e+01 | 0.9997 | 0.999956, 1.000000 |
| 1 | 20 (inside the outer root; 34 modes) | 100 | 30.083 | 0.9989 | 0.885 | 0.9995 | 30.007 | 0.883 | 0.9995 | 2.735e+01 | 0.9988 | 0.484753, 1.000000 |
| 1 | 40 (inside the outer root; 32 modes) | 100 | 27.451 | 0.9981 | 0.858 | 0.9981 | 26.645 | 0.833 | 0.9982 | 6.721e-03 | 0.0144 | 0.243767, 1.000000 |
| 2 | 12 (inside 2gE (no root at the reference clock); 56 modes) | 40 | 30.346 | 0.9919 | 0.542 | 0.9974 | 0.000 | 0.000 | 0.4289 | 3.533e-12 | 0.4886 | 0.999995, 1.000000 |
| 2 | 20 (inside the outer root; 35 modes) | 40 | 0.000 | 0.0000 | 0.000 | 0.0581 | 0.000 | 0.000 | 0.0926 | 2.820e-12 | 0.5197 | 0.216429, 1.000000 |
| 2 | 40 (inside the outer root; 33 modes) | 40 | 0.000 | 0.0000 | 0.000 | 0.1551 | 0.000 | 0.000 | 0.0917 | 4.630e-12 | 0.0302 | 0.274113, 1.000000 |
| 2 | 12 (inside 2gE (no root at the reference clock); 56 modes) | 60 | 49.591 | 0.9994 | 0.886 | 0.9999 | 39.400 | 0.704 | 0.9997 | 6.048e-11 | 0.4690 | 0.999995, 1.000000 |
| 2 | 20 (inside the outer root; 35 modes) | 60 | 28.776 | 0.9985 | 0.822 | 0.9992 | 24.075 | 0.688 | 0.9965 | 4.583e-11 | 0.3565 | 0.216429, 1.000000 |
| 2 | 40 (inside the outer root; 33 modes) | 60 | 0.001 | 0.0002 | 0.000 | 0.0045 | 0.000 | 0.000 | 0.0041 | 1.690e-10 | 0.0661 | 0.274113, 1.000000 |
| 2 | 12 (inside 2gE (no root at the reference clock); 56 modes) | 80 | 51.477 | 0.9996 | 0.919 | 1.0000 | 48.643 | 0.869 | 0.9999 | 2.490e+01 | 0.9967 | 0.999995, 1.000000 |
| 2 | 20 (inside the outer root; 35 modes) | 80 | 29.824 | 0.9988 | 0.852 | 1.0000 | 29.281 | 0.837 | 0.9999 | 4.420e-06 | 0.1290 | 0.216429, 1.000000 |
| 2 | 40 (inside the outer root; 33 modes) | 80 | 25.913 | 0.9975 | 0.785 | 0.9997 | 22.761 | 0.690 | 0.9972 | 3.375e-06 | 0.0673 | 0.274113, 1.000000 |
| 2 | 12 (inside 2gE (no root at the reference clock); 56 modes) | 100 | 51.862 | 0.9996 | 0.926 | 1.0000 | 50.569 | 0.903 | 1.0000 | 3.906e+01 | 0.9999 | 0.999995, 1.000000 |
| 2 | 20 (inside the outer root; 35 modes) | 100 | 29.933 | 0.9988 | 0.855 | 0.9999 | 29.841 | 0.853 | 1.0000 | 2.723e+01 | 0.9996 | 0.216429, 1.000000 |
| 2 | 40 (inside the outer root; 33 modes) | 100 | 26.772 | 0.9979 | 0.811 | 0.9998 | 25.946 | 0.786 | 0.9999 | 6.825e-04 | 0.0033 | 0.274113, 1.000000 |

The enclosed Killing energy of the deviation within the outer root at 100 r_s (kappa 1): I_L 2.253e-02, M_root 69.810, I_L/M_root 3.23e-04, delta R_L/R 4.36e-04 (I_L the Killing energy of D = C - P_- enclosed within the outer root; kappa_R the applied (N + v)'s slope at the root; delta R_L = G I_L/(R kappa_R) (TWOFIELD CF3's source-only coefficient; the heavy response at the same order is not included)).

### T8. The record-interpolation control (kappa 1 to 8): the production's partial sum against the piecewise-constant propagation's (page_history_v6n), and the quench energy

| T/r_s | cut r_s | production S_8 | control S_8 | prod - ctrl | 16 kappa E_kappa, kappa 1 to 8: production | control | quench E_ctrl - E_prod |
|---|---|---|---|---|---|---|---|
| 14 | 2 | - | - | - | 2.3680e+01 | 2.3814e+01 | 1.340e-01 |
| 14 | 16 | -9.2090e-02 | -9.1259e-02 | -8.311e-04 | 2.3680e+01 | 2.3814e+01 | 1.340e-01 |
| 14 | 24 | -7.0693e-02 | -6.8807e-02 | -1.887e-03 | 2.3680e+01 | 2.3814e+01 | 1.340e-01 |
| 14 | 30 | -5.7355e-02 | -5.5458e-02 | -1.897e-03 | 2.3680e+01 | 2.3814e+01 | 1.340e-01 |
| 14 | 48 | -3.3247e-02 | -3.1850e-02 | -1.397e-03 | 2.3680e+01 | 2.3814e+01 | 1.340e-01 |
| 20 | 2 | - | - | - | 1.4531e+00 | 1.6247e+00 | 1.715e-01 |
| 20 | 16 | -1.8931e-02 | -1.8781e-02 | -1.499e-04 | 1.4531e+00 | 1.6247e+00 | 1.715e-01 |
| 20 | 24 | -1.3238e-02 | -1.3076e-02 | -1.612e-04 | 1.4531e+00 | 1.6247e+00 | 1.715e-01 |
| 20 | 30 | -1.9329e-02 | -1.9683e-02 | +3.537e-04 | 1.4531e+00 | 1.6247e+00 | 1.715e-01 |
| 20 | 48 | -4.8662e-02 | -4.9120e-02 | +4.578e-04 | 1.4531e+00 | 1.6247e+00 | 1.715e-01 |
| 30 | 2 | - | - | - | 4.8549e-01 | 6.8156e-01 | 1.961e-01 |
| 30 | 16 | 5.1135e-03 | 5.5454e-03 | -4.319e-04 | 4.8549e-01 | 6.8156e-01 | 1.961e-01 |
| 30 | 24 | -3.6877e-03 | -3.7534e-03 | +6.571e-05 | 4.8549e-01 | 6.8156e-01 | 1.961e-01 |
| 30 | 30 | -6.7533e-03 | -6.8542e-03 | +1.009e-04 | 4.8549e-01 | 6.8156e-01 | 1.961e-01 |
| 30 | 48 | -1.2123e-02 | -1.2068e-02 | -5.513e-05 | 4.8549e-01 | 6.8156e-01 | 1.961e-01 |
| 40 | 2 | - | - | - | 1.4712e+00 | 1.6829e+00 | 2.117e-01 |
| 40 | 16 | 7.4399e-02 | 2.1783e+00 | -2.104e+00 | 1.4712e+00 | 1.6829e+00 | 2.117e-01 |
| 40 | 24 | 7.5819e-03 | 8.0756e-03 | -4.937e-04 | 1.4712e+00 | 1.6829e+00 | 2.117e-01 |
| 40 | 30 | 5.1230e-03 | 5.1407e-03 | -1.774e-05 | 1.4712e+00 | 1.6829e+00 | 2.117e-01 |
| 40 | 48 | -4.7461e-03 | -4.7886e-03 | +4.248e-05 | 1.4712e+00 | 1.6829e+00 | 2.117e-01 |
| 60 | 2 | - | - | - | 1.1592e+00 | 1.3835e+00 | 2.243e-01 |
| 60 | 16 | 5.6823e+00 | 6.0847e+00 | -4.024e-01 | 1.1592e+00 | 1.3835e+00 | 2.243e-01 |
| 60 | 24 | 4.6970e+00 | 5.2069e+00 | -5.100e-01 | 1.1592e+00 | 1.3835e+00 | 2.243e-01 |
| 60 | 30 | 6.7440e+00 | 8.1253e+00 | -1.381e+00 | 1.1592e+00 | 1.3835e+00 | 2.243e-01 |
| 60 | 48 | 7.9604e-03 | 8.0152e-03 | -5.484e-05 | 1.1592e+00 | 1.3835e+00 | 2.243e-01 |
| 80 | 2 | - | - | - | 8.3938e-01 | 1.0704e+00 | 2.310e-01 |
| 80 | 16 | 8.1283e+00 | 8.4446e+00 | -3.163e-01 | 8.3938e-01 | 1.0704e+00 | 2.310e-01 |
| 80 | 24 | 6.2073e+00 | 6.5673e+00 | -3.600e-01 | 8.3938e-01 | 1.0704e+00 | 2.310e-01 |
| 80 | 30 | 5.8851e+00 | 6.2676e+00 | -3.825e-01 | 8.3938e-01 | 1.0704e+00 | 2.310e-01 |
| 80 | 48 | 3.1321e+00 | 4.3303e+00 | -1.198e+00 | 8.3938e-01 | 1.0704e+00 | 2.310e-01 |
| 100 | 2 | - | - | - | 8.2053e-01 | 1.0548e+00 | 2.343e-01 |
| 100 | 16 | 8.7356e+00 | 9.0494e+00 | -3.138e-01 | 8.2053e-01 | 1.0548e+00 | 2.343e-01 |
| 100 | 24 | 8.4596e+00 | 8.7721e+00 | -3.125e-01 | 8.2053e-01 | 1.0548e+00 | 2.343e-01 |
| 100 | 30 | 8.2707e+00 | 8.5958e+00 | -3.251e-01 | 8.2053e-01 | 1.0548e+00 | 2.343e-01 |
| 100 | 48 | 5.8883e+00 | 6.4595e+00 | -5.713e-01 | 8.2053e-01 | 1.0548e+00 | 2.343e-01 |

The quench per channel (16 kappa (E_ctrl - E_prod) at 100 r_s): 1: 2.21e-03, 2: 6.32e-03, 3: 1.14e-02, 4: 1.89e-02, 5: 2.91e-02, 6: 4.07e-02, 7: 5.50e-02, 8: 7.07e-02. The control holds no 2 r_s cut and no channel above 8.

### T8b. Run (a) of W1 ruling v2 section 2: the held vacuum's cut entropy under a sub-step field shift (kappa 18; the busi lane's ruling2/runA_v1_results.jsonl, as stored, every row)

D_half and D_full are S_X^vac at the fields shifted by Delta_sub/2 and Delta_sub (the record interval over 8 sub-steps; the linear mode samples each sub-step at its midpoint, so Delta_sub/2 is the shift that applies) minus the stored S_X^vac at the record's fields; x_18 the stored kappa 18 excess (unweighted); the vacuum's reproduction at the record's fields ('repro') beside. The ruling's branch: D below x_18 by 10 or more where the weighted term 288 |x_18| exceeds 5e-4 reads the common mode as the exterior's physics, not the construction's reference lag.

| T/r_s | cut | 288 x_18 (stored) | 288 D_half | 288 D_full | \|x_18/D_half\| | same sign | repro | dN/dt | dv/dt |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 2 | +3.810e-04 | -5.79e-08 | -1.16e-07 | 6578 | no | 5.8e-13 | +2.77e-05 | -1.36e-05 |
| 40 | 16 | +1.154e-03 | -2.50e-07 | -5.01e-07 | 4618 | no | 6.2e-12 | -1.49e-07 | +2.81e-06 |
| 40 | 24 | +2.862e-04 | +3.59e-08 | +6.29e-08 | 7980 | yes | 1.7e-11 | -1.04e-09 | +2.07e-06 |
| 40 | 30 | +1.722e-04 | +2.83e-08 | +5.43e-08 | 6081 | yes | 4.5e-11 | -6.49e-10 | +1.85e-06 |
| 40 | 48 | +7.418e-04 | +2.85e-08 | +4.63e-08 | 26039 | yes | 8.0e-11 | -6.48e-10 | +1.46e-06 |
| 50 | 2 | +1.040e-03 | +2.11e-06 | +4.21e-06 | 494 | yes | -1.5e-12 | -1.42e-05 | +2.99e-05 |
| 50 | 16 | -1.230e-02 | +4.45e-05 | +8.91e-05 | 276 | no | -2.0e-12 | -2.86e-05 | +4.07e-05 |
| 50 | 24 | +8.119e-04 | -5.00e-07 | -1.00e-06 | 1624 | no | 1.1e-11 | -2.63e-07 | +2.93e-07 |
| 50 | 30 | -6.762e-05 | -1.43e-08 | -3.73e-08 | 4741 | yes | 2.1e-11 | -8.99e-10 | -8.13e-08 |
| 50 | 48 | -7.713e-05 | -1.06e-08 | -1.13e-08 | 7274 | yes | 1.1e-10 | -3.14e-10 | -6.48e-08 |
| 60 | 2 | -4.207e-04 | -2.72e-06 | -5.44e-06 | 155 | yes | -6.5e-13 | +1.15e-05 | -1.10e-05 |
| 60 | 16 | -1.420e-03 | -3.93e-06 | -7.85e-06 | 361 | yes | 5.4e-12 | -1.08e-07 | +4.18e-06 |
| 60 | 24 | -1.664e-03 | -1.25e-05 | -2.50e-05 | 133 | yes | 1.9e-11 | -2.06e-07 | +3.17e-07 |
| 60 | 30 | +4.114e-04 | -6.04e-07 | -1.21e-06 | 682 | no | 4.0e-11 | -1.10e-07 | +1.63e-07 |
| 60 | 48 | +5.726e-04 | +3.10e-09 | +2.92e-10 | 184498 | yes | 7.6e-11 | -9.26e-12 | -1.10e-08 |
| 70 | 2 | +1.868e-04 | -1.42e-06 | -2.83e-06 | 132 | no | 1.6e-12 | +4.91e-06 | +8.32e-07 |
| 70 | 16 | -5.687e-04 | +4.60e-06 | +9.18e-06 | 124 | no | 1.8e-11 | -9.55e-07 | +5.71e-06 |
| 70 | 24 | +2.820e-03 | -6.72e-06 | -1.34e-05 | 420 | no | 3.0e-11 | -9.55e-06 | +1.66e-05 |
| 70 | 30 | -1.291e-03 | -9.45e-06 | -1.89e-05 | 137 | yes | 4.1e-11 | -1.76e-07 | +2.81e-07 |
| 70 | 48 | -1.818e-04 | +1.02e-08 | -1.79e-09 | 17836 | no | 9.2e-11 | -3.45e-11 | -2.01e-11 |
| 80 | 2 | +4.247e-04 | -4.89e-07 | -9.78e-07 | 868 | no | -1.2e-12 | +9.14e-06 | -9.41e-06 |
| 80 | 16 | +1.282e-03 | +5.45e-07 | +1.09e-06 | 2352 | yes | 1.8e-11 | +7.87e-08 | +3.48e-06 |
| 80 | 24 | -9.564e-05 | +2.46e-06 | +4.92e-06 | 39 | no | 2.6e-11 | +1.89e-10 | +2.08e-06 |
| 80 | 30 | +3.050e-03 | +6.30e-07 | +1.27e-06 | 4845 | yes | 4.4e-11 | -8.95e-06 | +1.47e-05 |
| 80 | 48 | -1.803e-04 | -3.71e-07 | -7.23e-07 | 486 | yes | 1.3e-10 | -1.37e-07 | +1.88e-07 |
| 90 | 2 | +7.884e-05 | +1.37e-06 | +2.73e-06 | 58 | yes | 1.2e-12 | +2.36e-06 | -4.60e-06 |
| 90 | 16 | -5.833e-04 | +1.81e-06 | +3.63e-06 | 322 | no | 1.4e-11 | +6.70e-08 | +3.04e-06 |
| 90 | 24 | +1.625e-04 | +3.45e-06 | +6.92e-06 | 47 | yes | 5.5e-11 | -6.09e-07 | +3.24e-06 |
| 90 | 30 | +2.119e-03 | -6.56e-06 | -1.31e-05 | 323 | no | 3.1e-11 | -2.55e-07 | +1.99e-06 |
| 90 | 48 | -2.751e-04 | +6.61e-07 | +1.30e-06 | 416 | no | 9.8e-11 | -1.14e-07 | +1.61e-07 |
| 100 | 2 | -7.656e-05 | -3.11e-06 | -6.22e-06 | 25 | yes | 2.4e-13 | +1.95e-06 | +3.09e-08 |
| 100 | 16 | +2.878e-04 | -4.12e-07 | -8.28e-07 | 699 | no | 8.5e-12 | +1.05e-06 | +4.03e-07 |
| 100 | 24 | -4.821e-04 | +9.74e-07 | +1.95e-06 | 495 | no | 2.3e-11 | -2.04e-07 | +2.40e-06 |
| 100 | 30 | -1.027e-03 | -2.94e-06 | -5.88e-06 | 349 | yes | 1.7e-11 | +1.80e-07 | +1.11e-06 |
| 100 | 48 | -2.405e-03 | -4.92e-06 | -9.83e-06 | 489 | yes | 7.4e-11 | -2.22e-07 | +3.10e-07 |

Where 288 |x_18| exceeds 5e-4 (17 pairs): |x_18/D_half| is 124 to 184498 (median 420); the signs agree at 10 of them. D_full is about 2 D_half (linear in the shift).

### T9. Consistency checks of this composition

- This script's sums against the post-pass rows (K = 18, B2, 225 pairs): largest |difference| in S 0.0e+00, F_meas 1.1e-19, c 0.0e+00, F_apriori 5.6e-17 nats.
- The same for s8: largest |difference| in S 0.0e+00 nats.
- The verdicts at 40 to 100 r_s of the busi composition (from 40 r_s) against this lane's rerun from 14 r_s: 0 pairs differ.
- s8 at 40 r_s, the from-40 run against the from-14 run: 0 pairs differ.
- L_interval against the accumulator's increment per coordinate interval: largest relative deviation 0.0e+00 (the accumulator-difference definition; not a check of the exterior work or reference balance).
- The light-travel formula against version 3's marks (the 16 r_s radius; the dimensionless travel over 2gE_f): 25.200393 and 27.302162 (this script) against 25.200389 and 27.302158 (version 3): differences 4.5e-06 and 4.5e-06 absolute, 1.8e-07 and 1.7e-07 relative; 2gE_f 58.30220 against 58.30221 (the B2 record's formation E against r11's).
- The export page_v4_readout_export_v1b.tsv holds one row per readout clock and cut (155 rows from 40 r_s) with S, F_meas, F_apriori, c, K c, the last three weighted terms and their floors, q_x with its neighbours, the v1f tail, the probe terms, r_a, r_b, the criterion, the conditional majorant, the automatic verdict and the final tail status, for recomputing every filter of this delivery from stored data.


## 3. The D13 readouts: the sayable set, what the stored observables show, what is not said

The protocol (FABLE_RULING_O1_D13_protocol_v1_final.md, section 1) names four observables and fixes what may be said before they are read on the non-reflecting continuation: the theorem and window sentences; E_inv, the collar flow and L/L_H as labelled kappa 1, 2 diagnostics with the kappa 3 guard's change as their truncation error; the holds as drains in progress with their extrapolations. Not sayable: supplies, dark, an occupation, an emptying time, a matched-work sentence, a selected-archive digit. This version prints accordingly; the observables below are diagnostics as stored, their complete fluence error budget (the allowance of at most 1e-4 of the comparator's luminosity times the window) is not demonstrated, and no classification of the late flux as supply or darkness is made.

- **The labelled diagnostics on B2 to 100 r_s** are T5 and S8 (E_inv, the accumulated collar flow and L/L_H of kappa 1 and 2 with the kappa 3 guard's change). The marks the protocol carries with them: the formation record's light reaches the collar at 16 r_s at 35.08 to 37.00 on the record clock (35.57 to 37.49 on the collar's clock); the half-flow crossing through 100 is bracketed by the checkpoints 42 and 44 (the estimate 43.04; through 40 r_s on this propagation it is bracketed by 38 and 40, against version 3's 35.7 on the piecewise-constant run with per-record flows); R_cut 24 r_s near 45.3 to 47.2. All rows to 100 r_s are before the first return by the stored estimate (the continuum light round trip in the record's fields: 228 r_s at the collar, 211 at R_cut, 162 at the 48 r_s cut; an estimate, not a measured chain-end certificate); the join's departure from the appended vacuum form is printed beside each row (at most 7.12e-6 in v through 56 r_s, 5.84e-5 at 58, 4.57e-4 at 60, 8.70e-4 at 64, above 1e-3 from the 66 r_s checkpoint, 1.54e-2 at 80, 7.18e-2 at 100).
- **(a) The detector-energy ledger at the cuts 16 and 30 r_s** (T7 (a), kappa 1 and 2 summed, windows of 13 to 14 r_s). The second term at the 16 r_s cut (the deviation's inventory beyond the cut at the window's end minus at its start) is 8.0e-4 in the window 26 to 40, 0.312 in 36 to 50, 0.062 in 46 to 60, 0.043 in 56 to 70, 0.020 in 66 to 80, 0.0037 in 76 to 90 and 0.0038 in 86 to 100; at the 30 r_s cut -8.9e-7, -2.1e-6, 0.185, 0.336, 0.056, 0.036 and 0.016 in the same windows; the inventory beyond 16 r_s reaches 0.406 at 100 r_s (0.402 beyond 30 r_s), the inventory between the cuts falling from 0.344 at 56 to 0.016 at 86. The current's crossing and the inventory change are close at several windows, but not uniformly within 1 percent: their relative differences are 0.36 and 7.3 percent in the window 26 to 40 (the cuts 16 and 30), 0.05 and 27.45 percent in 36 to 50 (the 30 r_s cut's second term there is 2e-6), 0.42 and 0.00 in 46 to 60, 0.66 and 0.00 in 56 to 70, 1.27 and 0.13 in 66 to 80, 4.9 and 0.19 in 76 to 90, and 3.0 and 0.59 percent in 86 to 100. The large record-jump and held-vacuum changes (-2.0e-3 to 118 per window; 0.6 and below before the light reaches the cut) nearly cancel, but their full-precision difference (6.5e-8 to 2.8e-4, printed) is part of the balance and is kept. These differences are not by themselves numerical errors or evidence of supply; a D13 classification requires the complete balance and its total error allowance.
- **(b) L_old and L_sea by the 16 back-propagated probes** (T7 (b), every probe clock 40 to 100 every 4 r_s). With the reference clock 12 r_s the split reconstructs the direct current as L_old + L_sea, but L_old is not uniformly small and L_sea is not identically the direct current: |L_old| exceeds 2e-7 at 9 of the 32 probes and is at most 2.2e-4 (60 r_s, the 30 r_s cut, where L is 2.3e-3); |L_sea - L|/|L| is at most 0.096 over the 19 probes where |L| exceeds 1e-5 (there |L_old|/|L| is at most 0.096) and 5.2 at 56 r_s, the 30 r_s cut, where L is 3.0e-7. Preparation in the vacuum at T = 0 does not imply equality with the instantaneous held vacuum at T = 12 (the deviation at 12 r_s is small but not zero: E_inv 5.5e-4). With the reference clock 40 r_s the two components at the 16 r_s cut have opposite signs at the clocks from 72 to 100 r_s and magnitudes 1.2 to 173 times the direct current (38, 1.2, 37, 173, 13, 61, 39 and 45 at 72, 76, 80, 84, 88, 92, 96 and 100; L is 8.9e-6 at 100); with the reference clock 20 r_s the same opposite-sign parts appear from 56 r_s (161 times the direct current at 56, where the join's departure is 7e-6; 0.5 to 33 times afterwards), so the split is not attributed to the join here. This cancellation is a property of the reference-dependent split, not a supply classification; the comparator L_H(kappa_T) of the two channels is 5.2e-6 to 5.8e-6 over those clocks.
- **(c) Windowed changes of the hard and soft exterior-shell energy inventories** (T7 (c)). These entries are signed changes of the exterior Killing shells' energy inventories per retarded window, split by the radial current's sign and by the threshold 25 T_H^ref (3.75e-2), not certified fluences through a fixed detector. The hard outgoing and ingoing changes (kappa 1 and 2) are +0.32 and -0.32 in the window 26 to 40, +0.14 and -0.14 (36 to 50), +0.38 and -0.036 (46 to 60), -0.15 and +0.15 (56 to 70), +0.027 and -0.042 (66 to 80), +0.39 and -0.40 (76 to 90), -0.14 and +0.14 (86 to 100): they nearly cancel in every window but 46 to 60 (their sum +0.34) and 66 to 80 (-0.0145); the held vacuum's own hard outgoing change is 0.334 against the state's 0.381 in 46 to 60 and 1.9e-7 to 0.038 in the other windows, so the table does not show the hard entries dominated by the held vacuum's rearrangement outside 46 to 60; the soft changes are 6.0e-4 to 2.6e-2 in magnitude over both directions. A hard-emission or burst-tail reading requires the corresponding crossing ledger, reference terms and error enclosure; none is made here.
- **The identified source modes** (T7's last table): the eigenvectors of C inside the region at the reference clocks 12, 20 and 40 r_s with eigenvalue at least 0.99 (55, 34 and 32 modes for kappa 1; 56, 35 and 33 for kappa 2). Their occupation in the held vacuum of the reference clock (the stored occ_vac_extremes, printed per set) is 0.99996 to 1.0 for the sets referenced at 12 r_s and 0.24 to 1.0 (kappa 1) and 0.22 to 1.0 (kappa 2) for the sets referenced at 20 and 40 r_s inside the outer root, so those sets include modes that are not the sea's own; version 4's statement that they are the sea's own occupied modes is withdrawn. At 100 r_s the kappa 1 sets have total collar transfers 27 to 52 over sets of 32 to 56 modes; their maximum single-mode transfers are 0.998 to 0.9998, and for the 32-mode set referenced at 40 r_s the average transfer is 0.858 at the collar and 0.833 at the cut (kappa 2's 33-mode set: 0.811 and 0.786). No occupation or return of a radiation partner is inferred from these source-mode diagnostics.
- **The signed collar split per checkpoint** (T5's last table): the kappa 1, 2 energy flux through the collar window split by the radial current's sign; over 40 to 100 r_s the incoming part is 0.000 to 0.033 of the outgoing, below 0.01 at 24 of the 31 checkpoints and above it at 7 (0.016 at 48, 0.011 at 54, 0.023 at 56, 0.027 at 70, 0.033 at 78, 0.011 at 80, 0.028 at 92). The drain rule's no-return clause (the incoming part below 1 percent of the outgoing on every pre-reflection row) holds at 24 of 31 rows; the 7 rows above the threshold print with no attribution among the four causes the protocol admits (the centrifugal return of channel content below kappa/r_collar, the late arrival of slow lattice carriers, the interference of scattering packets, a collector or chain defect), none assigned without the Killing-energy-resolved split and the chain-end diagnostic.
- **(d) The holds** (copied from version 3b, md5 561e18ab, and the protocol): the records of T/r_s 15 and 30 held fixed drain at 5.4 to 6.3 times the slope comparator from 77 to the reflection clock 242.6 (after the burst's passage) and at -0.017 to 0.076 from 148 to 257.5; the emptying clocks are extrapolations only, 2,242 and 315 on the chord from the hold's start to the last clean row, T/r_s 3,547 and 16,182 on the plateau's slope over 127 to 241 and 142 to 256: drains in progress, no emptying time.
- **Not said anywhere in this version:** that the interior supplies or does not supply the channel; that the object is dark; an occupation read from a soft-bin ratio or from the source modes; an emptying time; a matched-work identity (W against Tr[h-dot (C - P)] is not printed as one); a selected-archive digit; "radiates at T_H", "agrees with" or "departs from Hawking's" at any clock.

## 4. The figure

PAGE_fig_v4.1.png, drawn by page_v4_fig_v1b.py from page_v4_data_v1.json and .npz; the plotted arrays are in PAGE_fig_v4.1_data.npz. Colours black, red, blue and tab10 only.

Caption. (a) The light channel's exterior-cut excess over the held Killing vacuum on the B2 history (r_s/a 64, m a 0.2, sigma/a 4, tau 8; page_history_v6o, 8 sub-steps, 8192 sites), summed over kappa 1 to 18 with the weights 16 kappa, at the cuts 2 (green), 16 (blue), 24 (black), 30 (purple) and 48 r_s (red) against the record clock T/r_s (over 14 to 100 r_s the collar's recordwise clock offset t_inf - T is 0.37 to 0.51 r_s, a stationary-chart substitution); the band is the exterior common mode's share K c; filled markers: formed, tail open; open markers: not formed or within the floor; the sigma/a 8 history's sums dashed to 40 r_s (a different object; its 2 r_s cut swings between -0.68 and +1.42 nats at 22 to 24 r_s); the shaded bands are the fixed-charge flight comparison of the formation's light from 2 and 1.5 x 2gE_f to each cut's radius; the dashed line the formation's record clock 12.125; the dash-dot line the formation's light at the 24 r_s cut plus T_d (T_d/r_s 2.31, a leading term with an unfixed matching constant, not a measured end). The tail beyond kappa 18 is open (an uncertified remainder) at every pair; the two errors F_meas and K c are T1's; the 2 r_s cut carries no record-interpolation spread; the completion spread (kappa 1, 2) is order one at every cut at 40 r_s (T4) and is not drawn; the sums are not monotone (the 30 r_s cut falls from 6.76 at 60 to 4.96 at 70 r_s; the 48 r_s cut's largest readout row is 82 r_s's, 6.41, not 100 r_s's, 5.87). (b) The collar flux at 16 r_s of the kappa 1 and 2 channels (weights 16 and 32), |L|/L_H against the record clock for the two comparators (red: the record's slope; blue: 1/(8 pi G E)); L is the collar energy per coordinate interval Delta T, L_H the two-channel continuum greybody, not an observed thermal flux; open markers: L negative; dotted: with the kappa 3 guard; grey squares: version 3's piecewise-constant run through 40; the shaded band the formation's light at the collar; the solid lines the half-collar-flow clocks through 100 (bracketed by 42 and 44; the estimate 43.0) and through 40 (bracketed by 38 and 40; the estimate 39.0); the dash-dot line the collar's arrival plus T_d. (c) The weighted channel terms |16 kappa x_kappa| against kappa at 100 r_s, the 24 r_s cut (black circles) and at 80 r_s, the 30 r_s cut (red squares), filled where x > 0 and open where x < 0, with the single-channel probes at kappa 24, 32 and 48, the propagation floor 16 kappa F_meas,kappa (dotted) and the formed-energy terms 16 kappa E_kappa (dashed): from kappa 12 the excess terms sit near the exterior's common mode (within a factor 1.4 at 100/24) while the energy terms fall, and the probes carry a term of the same size to kappa 48; no remainder bound beyond kappa 18 is supplied. Label: the light channel of the extension at first order (D16), propagated at first order on the solved history of the heavy field's collapse, its sources not fed back; the interior beyond the hydrodynamic gate from T/r_s 11.75. No Page curve of evaporation, thermal emission or partner return is claimed.

## 5. What the paper's marked sentences can take from this version (the correctness reading's R11 pastes, with the numbers of this version where they differ)

- 07_page_curve.tex line 46 (the pending curve): "The computed curve of the light channel at first order on the B2 history is the 18-channel excess Delta S_cut at the cuts 2, 16, 24, 30 and 48 r_s from T/r_s 40 to 100, with the propagation floor (1.7e-4 to 4.6e-4 nats) and the exterior common mode's share K c (0.02 to 6.2 percent of the sum from 80 r_s) beside each value, the raw cut entropy and the held Killing vacuum's as regulated columns (3467 to 6385 nats, with no limit in kappa), the tail beyond kappa 18 open at every pair as an uncertified remainder, the completion spread order one at 40 r_s and absent after, and no record-interpolation spread at the 2 r_s cut. Its largest value is the last clock's, 9.50 nats at the 2 r_s cut at 100 r_s, 2.7e-4 of A(E)/(4G) and 2.9e-4 of eta(0) A/a^2 at the record's charge (the five cuts 1.7e-4 to 2.7e-4 of A/(4G) at 100 r_s); the sums are not monotone (the 30 r_s cut falls from 6.76 at 60 to 4.96 at 70 as the front passes it; the 48 r_s cut's largest readout row is 82 r_s's, 6.41), no saturation is expected on the epoch and none is seen, and the finite-channel excess establishes no radiation-entropy turnover. [to do: the heavy field's exterior-cut entropy at the same cuts and the 2 fields' sum; the record-interpolation control at the 2 r_s cut and for kappa 9 to 18.]" (The reading's clause "the sums rise from 70 r_s on at every formed cut" is not taken: at the 2 r_s resolution the sums fall at 2 to 6 of the 15 steps from 70 at every cut.)
- The channel-set and propagation sentence (R11(a)): "On the B2 history (the production member continued to T/r_s 100; r_s/a 64, sigma 4a, tau 8, trapped episodes 12.125 to 17.25 and 18.375 to 100 r_s) the light channel is read at first order, its sources not fed back, propagated from its continued sea at T = 0 with the fields interpolated linearly between the records on 8192 sites with the vacuum exterior appended, through T/r_s 100, past the record's limit at 11.75: the exterior-cut excess in the channels kappa 1 to 18 and the collar flux in kappa 1 and 2 with kappa 3 as the truncation guard (the guard changes the flux ratio by 6 to 9 percent at 42 to 46 r_s and by order 1 before 36, the excitation by 1.8 to 6.2 percent from 42 r_s and from 9.8 to 15.8 percent at 14). The net signed collar flow of the 2 channels at 16 r_s over the charge, E_rad^(0)/E, is 7.6e-6 through 40 and 3.8e-3 through 100, a diagnostic, neither permanently escaped energy, charge loss nor a bound on the omitted feedback. The channels' excitation is 9.8 percent of E at 14, 0.16 to 2.5 percent at the checkpoints 15 to 40 and 0.42 to 1.4 percent over 42 to 100, its fall from 14 to 16 coming as the first trapped episode's pocket phase closes and P_- refills, a change the fields' evolution between the records carries, not outward energy loss."
- The flux sentence (R11(b)): "It is at most 0.15 of either comparator in magnitude before 36 (0.26 with the guard; negative at 9 of 12 checkpoints), 0.018, -0.027 and 1.20 times L_H(kappa_T) (0.049, -0.023 and 3.4 times L_H(T_H)) at 36, 38 and 40, 190, 272 and 61 times L_H(kappa_T) at 42, 44 and 46, 3.1 to 26 over 48 to 60 and -0.03 to 19 over 62 to 100 (0.08 to 1.3 at 80 to 100 but for 84), the 1/(8 pi G E) comparator 1.9 to 3.3 times smaller than the slope comparator from 42 on; the 4.0 and 5.1 at 36 and 38 of the piecewise-constant propagation were that propagation's (it forms energy at every record boundary). Light leaving 1.5 and 2 times the formation's 2GE at its record clock reaches the collar at 35.6 to 37.5 on the collar's clock; the accumulated flow through 100 is 0.405 in lattice units and half of it has arrived between the checkpoints 42 and 44 r_s on the record clock; from 64 r_s the join to the appended vacuum departs from the vacuum form by 8.7e-4 to 7.2e-2 in v."
- The sigma = 8a sentence (R11(c)): "On the sigma = 8a history, a different object, the excitation is 0.70 percent of E at 14 (the two channels; 0.80 in lattice units), the flux at 40 is 0.43 and 0.050 times the 2 comparators, the net collar flow through 40 is 9.6e-8 of E, and the 18-channel exterior-cut excess at 40 is 1.9e-2 nats at the 2 r_s cut and within the common mode at the 16 to 48 r_s cuts (|S| 1.2e-3 to 5.7e-3 against K c 2.3e-3 to 1.0e-2), against B2's 5.87 and 0.092 nats at the 2 and 16 r_s cuts; its 2 r_s cut swings between -0.68 and +1.42 nats at 22 to 24 r_s."
- Line 207 (the completion spread): S7 above, keeping the to-do for the transfer across the root, the late root, the far dust and the third completion.
- The tail sentences (R11(f), (g)): after "with a tail of order ln K/K ...": "On the computed history that asymptotic tail is not reached: the weighted terms 16 kappa x_kappa sit near the moving exterior's common mode from kappa 12 (about 5e-4 at 100 r_s and the 24 r_s cut, first order in the state's deviation, changing sign with the clock while the formed-energy terms fall) and single-channel probes carry a term of the same size to kappa 48, so the tail beyond K = 18 prints as open, an uncertified remainder, and the common mode's share K c as its error within K." "with its channel table and tail bound" becomes "with its channel table and its tail open"; "The paper's Page curve is S_cut against t_infty ... Its peak is compared" becomes "The paper's Page curve is Delta S_cut against t_infty at stated cuts, S_cut and its baseline printed beside it as regulated columns, since on the radial-channel reduction the held vacuum's weighted channel sum has no limit in kappa; its largest value is compared with ...".
- Line 227 (R11(h), reviewer B R19): the first two clauses are met (the channel from 40 to 100 on the 8192-site chain; the no-return clause read on the stored signed split: below 0.01 at 24 of 31 checkpoints, the seven flagged clocks 48, 54, 56, 70, 78, 80 and 92 r_s printed without attribution); "the flux after T_d read for the interior's supply by the retarded decomposition above, and the self-consistent 2-field history" stay to do: the detector inventories, back-propagated-current splits and shell-energy changes are available as diagnostics, but their complete fluence error budget is not demonstrated, so the late flux is not classified. The paper's "That late flux is classified by the detector-energy ledger" must not imply that this delivery performed the classification (a classification requires the ledger's complete pre-return-window allowance, at most 1e-4 of the comparator luminosity times the window; the collar-charge drift is not that test).
- reviewer B R20 (for the paper lane, with recheck337 R6): "The transmitted Pauli bound constrains the outgoing occupation, not the signed collar or detector current. It supplies no luminosity coefficient, and the signed current can be negative at a checkpoint." and "The cross-surface block and first-law records are read at one scale. The finite-K light-field exterior-cut excess is now reported on the sigma/a 4 and 8 histories through their common clock 40 r_s. They are different computed objects with substantially different readouts, so this comparison does not establish averaging-scale independence."

## 6. Checks

T9 lists them: this composition's sums equal the post-pass rows to 1e-16 (225 B2 pairs, 75 s8 pairs); the verdicts of the busi composition from 40 r_s and this lane's rerun from 14 r_s agree at every pair from 40; L per interval equals the accumulator's increment per interval exactly (the accumulator-difference definition; not a check of the exterior work or reference balance); the dimensionless flight times agree with version 3's marks to 4e-6 absolutely and under 2e-7 relatively (2gE_f 58.30220 here against 58.30221, the B2 record's formation E against r11's). The per-pair export (page_v4_readout_export_v1b.tsv, 155 rows) lets every filter of this delivery be recomputed from stored data.

## Changes from v4 (each with its source in deliveries/lead_gaps/FableAPI_reading_PAGE_v4_v1.md; "the data" means recomputed here from the producer's saved arrays)

1. The tail: OPEN stated as an uncertified remainder; "not geometric" and "the stored terms give neither" withdrawn; the geometric criterion of W1 ruling v2 section 2(b) evaluated per pair on the stored probe terms (met at 10 of the 17 selected pairs: 40/16, 50/2, 50/16, 60/48, 70/16, 70/24, 70/30, 90/16, 90/30, 100/30) with its conditional majorant printed and not applied (0.20 to 17.7 percent of the sum at the formed ones; 3.6 times the sum at 60/48); "area-like" dropped; the 1/kappa-law medians, the no-steepening of the exponents and the neighbouring-clock sign changes given as observations (reviewer A R2; reviewer B R1, R2; the T2 columns 'criterion' and 'conditional majorant').
2. The largest kappa 18 term: 50/16 (-1.230e-2; the probes -9.33e-3, -7.64e-3, -5.30e-3), 80/30 the second largest among the every-10 pairs; the exponent range -3.71 to -0.09 over the 14 same-sign selected pairs (+1.21 at 90/2, outside the 17); "do not continue that fall" at 70 r_s withdrawn (the probes fall there at 0.928 and 0.910) (reviewer A R2, R9; reviewer B R2).
3. S1's "equal across channels to about 10 percent" scoped by the spread over kappa 12 to 18 printed per pair (a factor 1.12 to 16.6 at the 17 selected pairs; 2.75 at 100/2); the premise of K c stated; the energy majorant labelled within-K with the probes' energies beside; run (a)'s rows supplied (T8b) and the attribution to the exterior's physics given with them; the sub-step bound (9.0 percent of the kappa 18 term where it exceeds 1e-3) (reviewer A M15, M2; reviewer B R3).
4. S2 with |S|; the automatic v1f verdict kept in T1's column with 76/24 printed formed, tail open and its final status OPEN; the tally 4 / 0 / 27 with the automatic count noted; T2c with the probes at 76 r_s (which exist: the probes are final at every 2 r_s) and its neighbours (reviewer A R9; reviewer B R4, R16).
5. S3: "prints nowhere" replaced by reviewer B's S3; d defined as the balance discrepancy (E_out - flow)/(16 kappa) = W_exc - R; the threshold kappa 13 (d/E_kappa 0.954 at 12, 1.103 at 13); the K = 18 accumulator over E and the per-channel accumulator column removed from the paper-facing tables (T5 prints d and E_kappa per channel instead) (reviewer A R3; reviewer B R5).
6. s8: |S| 1.2497e-3 to 5.6943e-3 at the 16 to 48 r_s cuts at 40 r_s; the 15 formed pairs' distribution (11, 2, 1, 1 at the cuts 2, 16, 24, 30); the kappa 14 control's term ratio 0.197 at 18/2 (0.013 or less at 14, 16, 20, 30, 40), 0.074 at 40/16 and 0.055 at 40/48; "at any K" dropped; the 2 r_s cut's swings at 22 to 26 r_s added from the data (-0.675, +1.417, 0.077 nats) (reviewer A R4, M1; reviewer B R6).
7. S7 with the scope (to 40 r_s, kappa 1 and 2 only), the paired-avg charge (6.4 percent above the record's), the engine and chain confounding, the sampled-checkpoint wording for the roots, and "order one at every cut at 40 r_s" (reviewer A R8; reviewer B R7); the charge spread printed in T4.
8. S8 replaced by reviewer B's; the join numbers corrected (8.7e-4 at 64, above 1e-3 from the 66 r_s checkpoint, at most 7.12e-6 through 56; the pre-60 range 4.4e-6 to 5.8e-5) (reviewer A M3; reviewer B R8).
9. Item 10 rewritten: the sums are not monotone; the 30 r_s cut's fall 6.756 to 4.960 between 60 and 70 with the 30 r_s flight mark (52.62 to 54.54) and the control's 1.38 nats at 60/30; the largest value is the last clock's, not a peak; A at the record's 2gE; no identification with a three-dimensional cut (reviewer A R1; reviewer B R9). The data contradict two clauses of the reading's replacement, which was read from the every-10 table: at the 48 r_s cut the largest readout row is 82 r_s's (6.405 nats; 100 r_s reads 5.867), and the sums fall at 2 to 6 of the 15 steps from 70 r_s at every cut; the delivery prints the data.
10. The v6i comparison at full precision from version 3's A-to-100 rows: -7.6 to +14.6 percent, the 14.6 at 56 r_s (reviewer A R7; reviewer B R10).
11. Section 3 (a): the "1 percent" claim withdrawn; the agreement per window printed (0.00 to 27.45 percent) and the jumps-minus-vacuum difference kept at full precision; the record-jump range -2.0e-3 to 118 (reviewer A R5, M6; reviewer B R11).
12. Section 3 (b): L_old at the reference clock 12 as the data say (above 2e-7 at 9 of 32 probes, up to 2.2e-4 at 60/30 where L is 2.3e-3; |L_sea - L|/|L| at most 0.096 where |L| exceeds 1e-5 and 5.2 at 56/30 where L is 3e-7); the reference-40 ratios at every probe clock (1.2 to 173 over 72 to 100 at the 16 r_s cut; the every-8 subset's 13 to 46 in the reading); the attribution to the join withdrawn (the reference-20 split shows the same parts from 56 r_s, where the join departure is 7e-6) (reviewer A R6; reviewer B R12).
13. Section 3 (c): the heading and interpretation as inventory changes; the 66 to 80 exception; the held vacuum's entries 1.9e-7 to 0.038 outside 46 to 60, so the "dominated by the sea's rearrangement" statement of v4 is withdrawn; the soft range 6.0e-4 to 2.6e-2 over both directions (reviewer B R14; reviewer A M6).
14. The source modes: the average transfers (0.858 at the collar and 0.833 at the cut for kappa 1's 32-mode set; 0.811 and 0.786 for kappa 2's) beside the maxima; the sets' vacuum occupations printed (0.99996 to 1.0 at the reference clock 12; 0.22 to 1.0 at 20 and 40), so v4's statement that the sets are the sea's own modes is withdrawn; "32 to 56 modes" (reviewer B R13; reviewer A M10).
15. T6: the shifted clocks at the 2, 24, 30 and 48 r_s radii labelled as the 16 r_s collar's offset at the nearest checkpoint, not conversions at those radii; the 2 and 30 r_s flight marks added (12.70 to 14.62; 52.62 to 54.54); the half-flow clocks as brackets with the estimate (reviewer B R15, R8; reviewer A R1, M14).
16. T5: the E_inv/E headings say percent; the comparator is absent at 18 r_s (no root) (reviewer B R15, R8).
17. The figure: the x-label's offset range 0.37 to 0.51 r_s; the 2 and 30 r_s cuts in panel (a) with their bands; panel (c)'s "flat to about 10 percent" replaced (reviewer A M5, M14; reviewer B R3, R15).
18. The provenance statement rewritten (the verdict column copied, the final tail status this composition's; S5 and the v6i rows identified with their sources; the checks not an independent rerun) (reviewer B R16); S1's "within 3.6 to 6.8 percent" to "3.6 to 6.8 percent of the sum" (reviewer A M4); T1b's definition of 18 |D16| (reviewer A M7; reviewer B R6); the 48 r_s cut's within-the-floor row at 72 (reviewer A M8; reviewer B R4); S5's source and the X3 comparators listed (reviewer A M9); the arming clock 11.125 with the restart slice 11.0 (reviewer A M11); T1c's header (reviewer A M12); the duplicated "(iii)" removed, the probes' checkpoint set and finality stated (reviewer A M13); the flight-formula agreement as 4e-6 absolute and under 2e-7 relative (reviewer B M3); the R_t and min(N - |v|) conventions (reviewer B M4).
19. The B2 kappa 18 sub-step control folded in, complete to 100 r_s (item 5, T1b; the licensed readings of reviewer A M2: at most 9.0 percent of the kappa 18 term where it exceeds 1e-3, so the one-sub-step reference lag is at most that fraction of the common mode; the energy rise 7.0 to 9.9 percent against s8's 12 percent fall).
20. The 2 r_s cut's missing record-interpolation spread stated in items 1 and 9, S4 and "What is open" (reviewer A R10); the paper pastes of section 5 updated with this version's numbers (reviewer A R11; reviewer B R17 to R20).

## SOURCES

Scripts (this lane; the box copies under <internal>/code are byte-identical):
- 26f909ef3cecee6506b32a6ffcd9ce45  <internal>/page_v4_data_v1b.py
- 13fd1cc05ed0caaacb846bed0470cfa5  <internal>/page_v4_tables_v1b.py
- 54753621a327c49f3fa556c0a839361e  <internal>/page_v4_fig_v1b.py
- b610037d5d256bd5c3891a88ff7a3d2f  <internal>/run_page_v4_box_v1b.sh
- c8b3fcf8ea71a9bd0d530e6a5fc494de  <internal>/assemble_page_v41_v1.sh

The run lines (the box, nice 10, OMP/MKL 16 threads; run_page_v4_box_v1b.sh): the producer `page_v4_data_v1b.py OUT`, the tables `page_v4_tables_v1b.py OUT/page_v4_data_v1.json OUT/page_v4_tables_v1b.md OUT/page_v4_readout_export_v1b.tsv`, the figure `page_v4_fig_v1b.py OUT/page_v4_data_v1.json OUT/page_v4_data_v1.npz OUT/PAGE_fig_v4.1.png OUT/PAGE_fig_v4.1_data.npz`, OUT = <internal>/out_v41 (version 4's outputs stay in <internal>/out). The post-pass reruns before them (the busi lane's w1_postpass_v1f.py, md5 f6466903, with its two self-test gates, both pass_all): `--prod <internal>/s8_prod_K18 v6o_s8 --from-rs 40 --kappa-max 18` and `--from-rs 14`, and `--prod <internal>/prod v6o_B2 --ctrl <internal>/v6n_B2_pod7_k1to8 v6n_B2 --from-rs 14 --kappa-max 18 --sentences-b2`; their outputs under <internal>/. The s8 K = 18 channel folder <internal>/s8_prod_K18 is symlinks to <internal>/v6o_s8_k*to* (its manifest s8_prod_K18_MANIFEST.md5 beside it).

Outputs (the box OUT folder; the small ones copied to solve/results/page_v4.1/ and the delivery folder):
- 0b36f0bf5f27cc240bc9911b7817cd6e  page_v4_data_v1.json
- 0ddef18391770b4eaa2e6a3fbb6405e4  page_v4_data_v1.npz
- 98af43efc28c545f2248d4176dec7a23  page_v4_tables_v1b.md
- 8636ce15063259bcf8502c90aed028ca  page_v4_readout_export_v1b.tsv
- d091cf5681a66339dd60ad5dad731a22  PAGE_fig_v4.1.png
- 5119ffa229c003ebb7eb69e13b96962c  PAGE_fig_v4.1_data.npz

Inputs on the box, by resolved path (the producer's MANIFEST.md5; the channel runs' launch line is job_v6o_r2.sh's / the pod card scripts': page_history_v6o.py --fields-key applied --nsites 8192 --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs 100 (40 for s8 and W3); the probes and the sub-step controls with --kappas K-K --no-kappa-extend --lean (--interp-sub 16 for the controls)):
- b8c7edc73dc90619561674eb6b1e726f  <internal>/v6o_B2_k1_summary.json  [B2 production kappa 1 summary.json]
- 50fe59a03edc93837980127cbbcec51b  <internal>/v6o_B2_k1_d13.json  [B2 production kappa 1 d13.json]
- 207cc3146a31f307d7931b7a0d624378  <internal>/v6o_B2_k2_summary.json  [B2 production kappa 2 summary.json]
- d19ccb67b023255cf99afbac22d0eb4a  <internal>/v6o_B2_k2_d13.json  [B2 production kappa 2 d13.json]
- 936565a2ffbf367e17fc642a8667552d  <internal>/v6o_B2_k3_summary.json  [B2 production kappa 3 summary.json]
- 78f26e42db7d7b1e6443c25e66a4ba90  <internal>/v6o_B2_k3_d13.json  [B2 production kappa 3 d13.json]
- e4c11f76b11b854c3055665a40e2fdeb  <internal>/v6o_B2_k4_summary.json  [B2 production kappa 4 summary.json]
- 2909f8466f226c711f937dad59e03dd8  <internal>/v6o_B2_k4_d13.json  [B2 production kappa 4 d13.json]
- d66989e517a0bf0e3e1acd5afa946d31  <internal>/v6o_B2_k5_summary.json  [B2 production kappa 5 summary.json]
- 25ade5a6b7c217f5b8868fac7f00ccde  <internal>/v6o_B2_k5_d13.json  [B2 production kappa 5 d13.json]
- b3f9a248e399cd9e98c2cf1a0143768a  <internal>/v6o_B2_k6_summary.json  [B2 production kappa 6 summary.json]
- 86bb53519c02ac98b479332a5a48f275  <internal>/v6o_B2_k6_d13.json  [B2 production kappa 6 d13.json]
- b17c1ac47a3f84b1f7452fdf434403a1  <internal>/v6o_B2_k7to9_k7_summary.json  [B2 production kappa 7 summary.json]
- a74a4341dcba6995b27097d58dbcc055  <internal>/v6o_B2_k7to9_k7_d13.json  [B2 production kappa 7 d13.json]
- c4b42932ee0d03216093f4d17cfa8238  <internal>/v6o_B2_k7to9_k8_summary.json  [B2 production kappa 8 summary.json]
- da624eeabffa35698e4a6da2cde7efd4  <internal>/v6o_B2_k7to9_k8_d13.json  [B2 production kappa 8 d13.json]
- baa90047463d02e09cdad85ae55bc203  <internal>/v6o_B2_k7to9_k9_summary.json  [B2 production kappa 9 summary.json]
- b4e9825e91f995e10b87f6494cfd7acd  <internal>/v6o_B2_k7to9_k9_d13.json  [B2 production kappa 9 d13.json]
- 4239c19c7a5e71368ce2a29aa0a42be2  <internal>/v6o_B2_k10to12_k10_summary.json  [B2 production kappa 10 summary.json]
- 8b67b3072305545bdede1752466e5bd0  <internal>/v6o_B2_k10to12_k10_d13.json  [B2 production kappa 10 d13.json]
- 00b048e20846435760612cc361c1b41d  <internal>/v6o_B2_k10to12_k11_summary.json  [B2 production kappa 11 summary.json]
- 27e5b89d1d07de8ddd55c4ec19f5ec5f  <internal>/v6o_B2_k10to12_k11_d13.json  [B2 production kappa 11 d13.json]
- 75c2c362febdd21964e773e7981c68a4  <internal>/v6o_B2_k10to12_k12_summary.json  [B2 production kappa 12 summary.json]
- c961ce48a701ec467dbdefbed6bf79a4  <internal>/v6o_B2_k10to12_k12_d13.json  [B2 production kappa 12 d13.json]
- ea3914375a21468e87c809b255b34486  <internal>/v6o_B2_k13to16_k13_summary.json  [B2 production kappa 13 summary.json]
- 4593725f9c75dd3635333782df5450bd  <internal>/v6o_B2_k13to16_k13_d13.json  [B2 production kappa 13 d13.json]
- eddd1551ad0798b62fa228020f9bb011  <internal>/v6o_B2_k13to16_k14_summary.json  [B2 production kappa 14 summary.json]
- cb34915964f65e2392b8f4b43c4d3f33  <internal>/v6o_B2_k13to16_k14_d13.json  [B2 production kappa 14 d13.json]
- 34c7583ab2d382167c0e892227b99f67  <internal>/v6o_B2_k13to16_k15_summary.json  [B2 production kappa 15 summary.json]
- b393dcf077afe320eaa04775cf215a45  <internal>/v6o_B2_k13to16_k15_d13.json  [B2 production kappa 15 d13.json]
- a867cfda6ba12d65690f871a0f2c33af  <internal>/v6o_B2_k13to16_k16_summary.json  [B2 production kappa 16 summary.json]
- f520414e587eea81a4f94079c6cf4be5  <internal>/v6o_B2_k13to16_k16_d13.json  [B2 production kappa 16 d13.json]
- 78a71baba71cd2ad4a5445680402de8e  <internal>/v6o_B2_k17to18_k17_summary.json  [B2 production kappa 17 summary.json]
- 03bfaba1b42a2afb10207cf68e3fc958  <internal>/v6o_B2_k17to18_k17_d13.json  [B2 production kappa 17 d13.json]
- f8c7fda38db5e836473dc455470e333f  <internal>/v6o_B2_k17to18_k18_summary.json  [B2 production kappa 18 summary.json]
- 37951baa8c72c66e6612759b007a5400  <internal>/v6o_B2_k17to18_k18_d13.json  [B2 production kappa 18 d13.json]
- 32a4752f37ccddadc8d4983d6089e1dd  <internal>/W1_B2_composed_v3.json  [post-pass B2_from40]
- cc8ec7aa54e7336783a04554eb1f50ed  <internal>/W1_B2_v1f_K18_from14.json  [post-pass B2_from14]
- 1481f6a39db70e0774f7c170e34eda77  <internal>/W1_s8_v1f_K18_from40.json  [post-pass s8_from40]
- 0ccbab3d2b708b6b98e7174446f37b7c  <internal>/W1_s8_v1f_K18_from14.json  [post-pass s8_from14]
- 89cee3169d439c0ac1dbb69a0a4bedc2  <internal>/v6o_B2r2b_k24_k24_summary.json  [probe kappa 24 summary.json]
- df6e78196e9c096935e747319bfe589d  <internal>/v6o_B2r2b_k24_k24_d13.json  [probe kappa 24 d13.json]
- 54a52212c5a81dab1d5777f99ec187fa  <internal>/v6o_B2r2b_k32_k32_summary.json  [probe kappa 32 summary.json]
- 80fc3c76e199197a5cb22e3df1b13e13  <internal>/v6o_B2r2b_k32_k32_d13.json  [probe kappa 32 d13.json]
- 5fd4ef20f299c3aa5944e9450c297967  <internal>/v6o_B2r2b_k48_k48_summary.json  [probe kappa 48 summary.json]
- 54dcac45b753b0c04b260911338cbbdc  <internal>/v6o_B2r2b_k48_k48_d13.json  [probe kappa 48 d13.json]
- 7dfc6157a9a3ae20504af1fa20ba8982  <internal>/v6o_B2r2c_sub16_k18_summary.json  [B2 sub16 kappa 18 summary.json]
- beff9d5bc47b7106135430776b8ca245  <internal>/v6o_B2r2c_sub16_k18_d13.json  [B2 sub16 kappa 18 d13.json]
- 2c5f1d65ca72877e883c71ca4afe8b8f  <internal>/v6o_s8_k1to1_k1_summary.json  [s8 production kappa 1 summary.json]
- 3ff51aa06e5e0d9b3ee50bcd9980491f  <internal>/v6o_s8_k1to1_k1_d13.json  [s8 production kappa 1 d13.json]
- 21badaf597d9233b81e039df949a059c  <internal>/v6o_s8_k2to2_k2_summary.json  [s8 production kappa 2 summary.json]
- c5fb6e1ec732371ea9d753369c2c64ff  <internal>/v6o_s8_k2to2_k2_d13.json  [s8 production kappa 2 d13.json]
- ccefb50b9a9721cc637a39d1150559d0  <internal>/v6o_s8_k3to4_k3_summary.json  [s8 production kappa 3 summary.json]
- 615e5e625ed9de7f49ada483e6a88150  <internal>/v6o_s8_k3to4_k3_d13.json  [s8 production kappa 3 d13.json]
- ee9fd961e87c0fa462b20f307e292282  <internal>/v6o_s8_k3to4_k4_summary.json  [s8 production kappa 4 summary.json]
- d7d20584532c5fc12dc920c833f98092  <internal>/v6o_s8_k3to4_k4_d13.json  [s8 production kappa 4 d13.json]
- 4b39a5ff83c2dadc6a4385ae992941b0  <internal>/v6o_s8_k5to5_k5_summary.json  [s8 production kappa 5 summary.json]
- 5b38f03333057b775763666b8f636bac  <internal>/v6o_s8_k5to5_k5_d13.json  [s8 production kappa 5 d13.json]
- f96bce24a5ae929c36fa1241040d02c8  <internal>/v6o_s8_k6to6_k6_summary.json  [s8 production kappa 6 summary.json]
- 4467b7084ce66b34741a8e3a55afa01e  <internal>/v6o_s8_k6to6_k6_d13.json  [s8 production kappa 6 d13.json]
- b4dd27824f9c03d25a07e41d974746d5  <internal>/v6o_s8_k7to7_k7_summary.json  [s8 production kappa 7 summary.json]
- 105c1668afe3091fe8d50f7ff3baa6c4  <internal>/v6o_s8_k7to7_k7_d13.json  [s8 production kappa 7 d13.json]
- 81f0743980a3b7aa7dbda6df78510e04  <internal>/v6o_s8_k8to8_k8_summary.json  [s8 production kappa 8 summary.json]
- ea8e76971d85c6808e45c8892497e2ce  <internal>/v6o_s8_k8to8_k8_d13.json  [s8 production kappa 8 d13.json]
- d03cd4c95a1fb8723128e9f8826dd5e2  <internal>/v6o_s8_k9to9_k9_summary.json  [s8 production kappa 9 summary.json]
- 7506b24a26a1707b7be68f24cfacd04a  <internal>/v6o_s8_k9to9_k9_d13.json  [s8 production kappa 9 d13.json]
- 02dfbb4bce6ea9101cc053e1a4e68882  <internal>/v6o_s8_k10to10_k10_summary.json  [s8 production kappa 10 summary.json]
- c5269739873b3d368af0c915d4a679ed  <internal>/v6o_s8_k10to10_k10_d13.json  [s8 production kappa 10 d13.json]
- 855ed16db9ddd36ffa847264d5c5160d  <internal>/v6o_s8_k11to11_k11_summary.json  [s8 production kappa 11 summary.json]
- 507177c86125822f2adaf0f0036fcd0c  <internal>/v6o_s8_k11to11_k11_d13.json  [s8 production kappa 11 d13.json]
- aa43cc9bb91e0e0221d164a872a443c9  <internal>/v6o_s8_k12to12_k12_summary.json  [s8 production kappa 12 summary.json]
- 745d376a99af059213911797118c6064  <internal>/v6o_s8_k12to12_k12_d13.json  [s8 production kappa 12 d13.json]
- e9ccff30cd99622b7f26b3ddd8342e0f  <internal>/v6o_s8_k13to13_k13_summary.json  [s8 production kappa 13 summary.json]
- ed2b3fef8ab0e46cb33ecd2385487d25  <internal>/v6o_s8_k13to13_k13_d13.json  [s8 production kappa 13 d13.json]
- ba5a0d6d7303ee52c1c30594c8562cdd  <internal>/v6o_s8_k14to14_k14_summary.json  [s8 production kappa 14 summary.json]
- 69c5d4a6d11e7290e478cac793afa152  <internal>/v6o_s8_k14to14_k14_d13.json  [s8 production kappa 14 d13.json]
- 2442f5c84b373d044fe56b98f55071b9  <internal>/v6o_s8_k15to15_k15_summary.json  [s8 production kappa 15 summary.json]
- b483f3194696941fe6124b32dbc70783  <internal>/v6o_s8_k15to15_k15_d13.json  [s8 production kappa 15 d13.json]
- 2b665655454576e2e7a80d3e22828315  <internal>/v6o_s8_k16to16_k16_summary.json  [s8 production kappa 16 summary.json]
- 47fc801b6536ec1406b4069ba54acf00  <internal>/v6o_s8_k16to16_k16_d13.json  [s8 production kappa 16 d13.json]
- 00c54e73a39e25ac479d2d6600443d82  <internal>/v6o_s8_k17to17_k17_summary.json  [s8 production kappa 17 summary.json]
- 696070ff28171ae4dbb81b3b3aa22b4d  <internal>/v6o_s8_k17to17_k17_d13.json  [s8 production kappa 17 d13.json]
- ddc34d8d38af1ff45aa5f2de469849fe  <internal>/v6o_s8_k18to18_k18_summary.json  [s8 production kappa 18 summary.json]
- f27bebc91f2483fa084aba720592e9a9  <internal>/v6o_s8_k18to18_k18_d13.json  [s8 production kappa 18 d13.json]
- 4f9d37b95cb4a4d32bab6fe05f1696d6  <internal>/v6o_s8r2c_sub16_k14_summary.json  [s8 sub16 kappa 14 summary.json]
- 12a2a052850a70cb34b3f413d2c707c8  <internal>/v6o_s8r2c_sub16_k14_d13.json  [s8 sub16 kappa 14 d13.json]
- 2dace52ff1436c3220b2c23a8528b7bb  <internal>/v6o_W3fs_k1to1_k1_summary.json  [W3 held-sources (W3fs) kappa 1 summary.json]
- 61401632e459fa90a7d4fc4fbfc6dd7e  <internal>/v6o_W3fs_k1to1_k1_d13.json  [W3 held-sources (W3fs) kappa 1 d13.json]
- ad129b4e2bb9c52c417a4eb76a10e4c3  <internal>/v6o_W3fs_k2to2_k2_summary.json  [W3 held-sources (W3fs) kappa 2 summary.json]
- 3407af10e16b9e209f21e203133a3398  <internal>/v6o_W3fs_k2to2_k2_d13.json  [W3 held-sources (W3fs) kappa 2 d13.json]
- d37c2b25f0a68a40acef7c53744b6698  <internal>/v6o_W3pa_k1to1_k1_summary.json  [W3 paired-avg (W3pa) kappa 1 summary.json]
- e53f9d1007aa6c4f29080223c02cad60  <internal>/v6o_W3pa_k1to1_k1_d13.json  [W3 paired-avg (W3pa) kappa 1 d13.json]
- d171ef26b6a58d6ed3b2bda8e2ec51fd  <internal>/v6o_W3pa_k2to2_k2_summary.json  [W3 paired-avg (W3pa) kappa 2 summary.json]
- fc690e65c6c78e28811a26ab8ec671ba  <internal>/v6o_W3pa_k2to2_k2_d13.json  [W3 paired-avg (W3pa) kappa 2 d13.json]
- 491db6fbb0037abdc2877c9e5ac4b558  <internal>/in_v3_summary.json  [version 3 summary json]
- 417a4018cf6497f9b0020408783f44f5  <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json  [B2 record json]
- cb26fcfaf3d6e80a17f0e73bac687301  <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz  [B2 fields npz]
- ef0ae52090957ae6749c83d2db958ce8  <internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json  [s8 record json]
- a9d6472d9180a331543b183b019e6fd4  <internal>/collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg_fields.npz  [s8 fields npz]
- 1e62a345abb47d1fd44522ace298f8b7  <internal>/v6n_B2_k1_summary.json  [B2 piecewise-constant control (v6n) kappa 1 summary.json]
- 3cb0a854368137f1ee20c78fbeffa957  <internal>/v6n_B2_k1_d13.json  [B2 piecewise-constant control (v6n) kappa 1 d13.json]
- 036f4447870e3d76f9c5d2b336f0b65c  <internal>/v6n_B2_k2_summary.json  [B2 piecewise-constant control (v6n) kappa 2 summary.json]
- 6c056f72c6c5cbd7004758224f25c04e  <internal>/v6n_B2_k2_d13.json  [B2 piecewise-constant control (v6n) kappa 2 d13.json]
- 962e044adfc363e7fc753f9252b011e5  <internal>/v6n_B2_k3_summary.json  [B2 piecewise-constant control (v6n) kappa 3 summary.json]
- ae462eb75f48d53546878dde0049d885  <internal>/v6n_B2_k3_d13.json  [B2 piecewise-constant control (v6n) kappa 3 d13.json]
- 02583b38fb6e3292e602b0bce5d4252e  <internal>/v6n_B2_k4_summary.json  [B2 piecewise-constant control (v6n) kappa 4 summary.json]
- d2858395b8f2e61e45feca0e03c6962b  <internal>/v6n_B2_k4_d13.json  [B2 piecewise-constant control (v6n) kappa 4 d13.json]
- 64e8f4fc368daa534bc1b89690273d24  <internal>/v6n_B2_k5_summary.json  [B2 piecewise-constant control (v6n) kappa 5 summary.json]
- b30c75dffda78a780aea4d87fad025a8  <internal>/v6n_B2_k5_d13.json  [B2 piecewise-constant control (v6n) kappa 5 d13.json]
- 995a458cfa969cc99a68a6a31342a878  <internal>/v6n_B2_k6_summary.json  [B2 piecewise-constant control (v6n) kappa 6 summary.json]
- 04e372c96459881468d62885cef65ac4  <internal>/v6n_B2_k6_d13.json  [B2 piecewise-constant control (v6n) kappa 6 d13.json]
- 5e985cedd6e88d8949d6f5120d569e2b  <internal>/v6n_B2_k7_summary.json  [B2 piecewise-constant control (v6n) kappa 7 summary.json]
- 91b8cd4889d35e786810b35b4882d4ce  <internal>/v6n_B2_k7_d13.json  [B2 piecewise-constant control (v6n) kappa 7 d13.json]
- 418bdfbbc52d85be6b16e92c446208b9  <internal>/v6n_B2_k8_summary.json  [B2 piecewise-constant control (v6n) kappa 8 summary.json]
- 217a6d11f6743b060ab9b360035a0c19  <internal>/v6n_B2_k8_d13.json  [B2 piecewise-constant control (v6n) kappa 8 d13.json]
- 1c296ce476a805de63774ad06c787ae5  <internal>/runA_v1_results.jsonl  [run (a) results jsonl]
- 4c793ad7110ec0e4dcf3881b84d2115b  <internal>/page_history_v6o.py  [code page_history_v6o.py]
- 0eaed28a4caaa809ea03cc92e2baf4f2  <internal>/page_w1_sector_v1d.py  [code page_w1_sector_v1d.py]
- 70383bb15fa5c642a1ac7d570e2f923e  <internal>/page_d13_readout_v1d.py  [code page_d13_readout_v1d.py]
- f64669038e16b94e9cdf7a8bcc5a1205  <internal>/w1_postpass_v1f.py  [code w1_postpass_v1f.py]
- 74c340d62f77747989f24360713d7a25  <internal>/job_v6o_r2.sh  [code job_v6o_r2.sh]
- a720f4f44b76859d89b39e073b89f263  <internal>/runA_v1.py  [code runA_v1.py]
- c2ec12771d5b45b3decef107bb22e4ce  <internal>/W1_B2_composed_v3.md  [composition file W1_B2_composed_v3.md]
- 0ceebb9140146d549ebad75bb5ee3f53  <internal>/compose_manifest.txt  [composition file compose_manifest.txt]
- c59e4dde88da2188bbc140f9c14fc73c  <internal>/s8_prod_K18_MANIFEST.md5  [composition file s8_prod_K18_MANIFEST.md5]
- 26f909ef3cecee6506b32a6ffcd9ce45  <internal>/page_v4_data_v1b.py  [this script]
- 0b36f0bf5f27cc240bc9911b7817cd6e  <internal>/page_v4_data_v1.json  [output]
- 0ddef18391770b4eaa2e6a3fbb6405e4  <internal>/page_v4_data_v1.npz  [output]

Text inputs read for this composition (local):
- 810494a626e6aee569c2586957f481ab  <internal>/FableAPI_reading_PAGE_v4_v1.md  (the correctness reading of version 4 (both parts; the corrections applied here))
- c8035e56039bb1fa8dadeddd45a299fd  <internal>/PAGE_solved_history_v4.md  (version 4 as issued)
- b1000b0e57c606ed99e50bde4ccc3d96  <internal>/W1_B2_composed_v3.md  (the busi lane's B2 K = 18 composition (its sections 1, 4 to 6: run (a), the tally, the sentences S1 to S5 with numbers))
- 9efc82355c38dc81c5b0217c876125ac  <internal>/W1_compose_ruling_v2.md  (the W1 compose ruling v2 (the v1f rule, S1 to S3, runs (a) to (d), the geometric criterion of section 2(b)))
- f64669038e16b94e9cdf7a8bcc5a1205  <internal>/w1_postpass_v1f.py  (the post-pass (the lane copy; the box copy has the same md5))
- 04400d63a0e97e2c6f4229c37811d469  <internal>/FableAPI_reading_B2_k24_k32_k48_probes_J7_v1.md  (the reading of the kappa 24, 32, 48 probes (its section 3; 76/24 ruled formed, tail open))
- fb18b286a1905eed637338e2de7e9b31  <internal>/FABLE_RULING_O1_D13_protocol_v1_final.md  (the D13 protocol ruling (the sayable set, section 1))
- b827418abf5e4071356be411b0275c48  <internal>/PAGE_solved_history_v3.md  (version 3 (the marks' definitions, section 6))
- 561e18abf58ff60a9ca0153ccc2d590e  <internal>/PAGE_solved_history_v3b.md  (version 3b (the holds' readings; the withheld list))
- af0029e7ebef812690dfda9140855cc3  <internal>/FableAPI_cold_reading_PAGE_solved_history_v3.md  (the cold reading of version 3)
- 7af1a7f3b279200c3dc52103b5b09c91  <internal>/FableAPI_cold_reading_PAGE_solved_history_v3b.md  (the cold reading of version 3b)
- 491db6fbb0037abdc2877c9e5ac4b558  <internal>/PAGE_solved_history_v3_summary.json  (version 3's summary json (copied to the box as in_v3_summary.json: the figure block's marks, the s4 run's flux and the A-to-100 rows))
- e9857e83085a50a860bc764d7bf09331  <internal>/requirements_new.txt  (the requirements (the Page-curve paragraphs of sections 2 and 5))
- 24fb3be0680fbaaf0d24bcbd3decd576  <internal>/07_page_curve.tex  (the paper's section (the marked sentences at lines 46, 207, 227))
- e70ebeeaa89ff5f8b8afec1c93c366d4  <internal>/06_scaling.tex  (the paper's T_d table (line 338))
