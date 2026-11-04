# W9 cold read: raymap_w9_B2_w9 (RAYMAP_solved_flow_v3 over v2b) with raymap_w9_selftest_v1

## 1. What was computed

Three outgoing unit packets in the kappa = 1 light channel, received at 16 r_s (site 1023) at omega r_s = 0.0796 (0.83 T_H at the launch charge), 1 and 2, were launched at 139 retarded clocks u/r_s 6.044 to 74.693 (records 251 to 800, T_L/r_s 31.375 to 100) and propagated backward through the stored B2 records ((0.2, 64), sigma/a 4, tau 8, v6m fields) to T = 0, where they are ingoing packets at U/r_s 7 to 87 (f(k<0) 0.99 to 1.00 at omega r_s 1 and 2). From the T = 0 readouts it forms 7 peeling-rate estimates per clock (the primary k = d ln<|k|>_T0/du, then E, Ede, U, Uin, rayk, rayU), their ratio to kappa_H = 1/(4GE_launch) (0.6003 to 0.6039 per nominal r_s; 2gE_k 53.0 to 53.3 over the launch records), band runs within 5, 20 and 50 percent, onset, the dispersive end at alpha_d = (2 a omega)^(1/3), the e-fold count, and v2b's record table with 2gE_k and R_t/2gE_k added. It is the task spec's readout (3 jets, onset, dispersive end, e-folds within 5 and 20 percent, 50 added) on 1 history, 1 receiver, 1 channel. Against the requirement: the cycle count is not computed, the comparator block is the withdrawn rate form, and no averaging-scale, step or first-route control is attached, so "Y1's second route" is half a pair as delivered.

## 2. Numbers

**Can print, labelled computed on the (0.2, 64) sigma/a 4, tau 8 history at 16 r_s, kappa 1, against 1/(4GE) at the launch record's collar charge:**

- Peak traced-back-wavenumber exponent 0.459 kappa_H (omega r_s 1) and 0.482 kappa_H (omega r_s 2), both at u/r_s 12.557, 1-step/2-step error 0.006 and 0.013; minima -0.485 (u 24.944) and -0.476 (23.944); the oscillation's amplitude falls to ±0.07 and ±0.13 after u/r_s 60. Normalization floor: against 1/(4GE_0) (2gE_0 59.069, 0.5417 per nominal r_s) the same peaks read 0.51 and 0.54; against 1/(2 r_s nominal) 0.55 and 0.58.
- k-readout band runs (2-clock rule, band error ≤ frac): none at 0.05, 0.20 or 0.50 on any jet; onset none; e-folds inside each band 0.
- Dispersive end not reached by any jet: max <|k|>_T0 0.1202 per spacing = 0.382 of alpha_d 0.3150 at u/r_s 18.941 (omega r_s 1), 0.1450 = 0.365 of 0.3969 at 15.439 (omega r_s 2), 0.0190 = 0.140 of 0.1355 at the last clock 74.693 (T_H, still rising); ray |k_ray| max 0.0330, 0.1224, 0.1442; the 1/3 and pi/4 thresholds never reached.
- The T_H jet: 139 of 139 rows near the branch edge (cut fraction 0.220, edge distance 0.94 sigma_k), sigma_u median 54.6 r_s, so 2 sigma_u = 109 r_s exceeds the 68.6 r_s grid; smeared exponent 0.005 to 0.046 kappa_H; total ln blueshift over the grid 1.07.
- Checks: norm and forward-check defects ≤ 3.8e-12 on all 417 rows, 0 refused; the summary and all-runs tables reconcile (clock counts and e-fold sums checked on all 3 jets).
- Record table (v2b's dense columns): 2gE_k 59.069 at T 0, 54.274 at 16 (0.9188 of E_0, the requirement's own figure), 52.99 to 53.03 from 50 on; kappa_T × 4GE_k 0.97 to 1.31 from T/r_s 27 on (0.32 to 0.63 at 16 to 19); R_t dense/2gE_k 0.67 to 0.71 (35.8 to 37.6 a) from 27 on; the dense slope first at 0.9 of 1/(4GE_k) at records 99 to 100 (T/r_s 12.375 to 12.5), the light from 1.5 and 2 × 2gE_k then arriving at u(arrival) 11.15 to 11.25 and 9.48 to 9.64.
- Self-test 15 of 15: the k readout within 0.0018 (LIN) and 0.0047 (PG) of kappa_H on exact segments, e-fold count 2.0373 against 2.0387, forward-backward 4.6e-12; the onset is biased 2 clocks (1 r_s) late by construction (T0b), the floor of any onset.

**Cannot print:**

- Every comparator column (f_s, f_d, extents, e-folds, constants): withdrawn form (defect 1).
- Any onset, segment or e-fold from U, Uin, rayU, rayk rows: centroid swings by the readout's own rule (defect 2); rayU, rayk and Uin have no self-test row.
- The T_H jet's exponent as the thermal frequency's own: it is a 55 r_s smear.
- A cycle count: no column; 0 only by arithmetic on the empty k segments.
- Row 17's received-redshift or late-time law e^(-u/(2 r_s)) against an emission clock: the readout gives the ray map's exponent (which never sustains kappa_H), not a redshift-against-emission table; the multistream run is not W9's.
- The ratio to each record's own dense slope kappa_T (the requirement's test scale): ingredients present, per-clock composition absent.
- The 50-percent emptiness at omega r_s 1 and 2 without the normalization named (defect 5).
- Any ell-, step- or 2-route-controlled statement.

## 3. Defects, ranked

1. **Comparator is the withdrawn rate.** 6/[(1+f)^4(2+f)(3+f)], f = (R − r_s)/r_s, is "the radial gap in place of y − 1" the requirement withdraws; the y-form 6/(y^4(y+1)(y+2)) gives 0.78 to 0.80 at f_d for omega r_s 1 (readout 0.632), constants −7.63/−4.43/−1.38 r_s (readout −9.08/−6.11/−3.75), 1.4 to 1.5 nominal e-folds within 50 percent at omega r_s 1 (readout 0.434), and 0.3/3.5/6.5 r_s at T_H (readout empty/1.886/4.242). The self-test's COMP row passes against the retired −6.0/−3.7/0.63, and its "f_s 0.0107 against the quoted 0.021" is f against y − 1 (sqrt(1.0214) − 1 = 0.0107). The readout says the solved-flow columns do not depend on it; they do not.
2. **Position-map runs printed as onsets and e-folds with no flag.** Every non-empty row is U, Uin, rayU or rayk; there the k ratio is low or negative and ln(k_end/k_start) contradicts the e-fold column (omega r_s 1, U 0.20: 0.9736 against −0.0110; U 0.50: 1.0318 against −0.5107; omega r_s 2, the only "resolved" runs, U 0.50 at 47.446–49.445 and 54.445–56.445: 1.568 and 1.357 with r_k +0.13 to −0.23 and r_E falling; T_H rayU at 8.550: ratio 1.018 while r_rayk is 0.108 and r_k 0.043). The failure rules name these centroid swings and make k the record value; the tables do not.
3. **Missing columns.** No cycle count; no rayU column and no error column but err_k in the per-clock tables, though each readout is defined with its own, so rayU runs and the "err mean" of U, E, Ede and rayk runs cannot be checked per clock.
4. **The T_H jet carries a frequency label** ("1 T_H at the nominal r_s") in the Y1 header while its cut removes 22 percent of the Gaussian on every row.
5. **kappa_H normalization unstated as a sensitivity.** 1/(4GE_launch) uses the collar charge after its step deficit (0.83 of nominal), which the requirement calls a step artefact, never mass loss; against 1/(4GE_0) the k readout enters the 50-percent band at 12.557–13.057 (omega r_s 1, by 1 percent) and 12.057–13.057 (omega r_s 2, by 1 to 4 percent).
6. **Small slips.** "A blueshift of 10.63 against 75.4", "5.96 against 15.8", "3.59 against 9.9" imply launch wavenumbers about 1 percent apart within each pair; "139 clocks at step 0.499994" while the grid has a 0.378 step (14.560 to 14.938) and a 0.250 last step, with the U readout "exact on a uniform grid" and the omega r_s 2 U 0.50 onset 15.439 at that step; the per-jet bullets' "no run of 2 clocks (0 isolated in-band clocks)" are the k readout's but do not say so above tables listing rayU runs.
7. **Inherited labels.** "The goal's display's sign of v amended per engine 12.2; the owner's confirmation pending" (no W9 number depends on it, |v_0| throughout); the formation root "34.509 spacings" at record 97 is 0.59 of the interpolated 2gE_k 58.2 against the requirement's 0.66, unreconciled.

## 4. Paper sentence (06_scaling:375)

On the solved (0.2, 64) history (sigma/a 4, tau 8), outgoing probes at omega r_s = 1 and 2 received at 16 r_s in the kappa = 1 light channel and traced back through the stored records have a peeling rate that peaks at 0.46 and 0.48 of 1/(4GE) at the launch record's charge (u/r_s 12.6 on the retarded clock) and holds no run of 2 clocks within 5, 20 or 50 percent of it over u/r_s 6 to 75, so the records' own exponential segment is empty at those criteria with 0 e-folds and 0 mode cycles, and their traced-back wavenumber reaches 0.38 and 0.37 of the dispersive cutoff (2 a omega)^(1/3), no dispersive end being read. At omega = T_H the probe at this receiver spans about 55 r_s in u with its spectrum cut at the branch edge, and its smeared exponent is 0.005 to 0.046 of 1/(4GE). These are the readings of 1 history, 1 receiver and 1 channel, against the drifted collar charge's 1/(4GE), with no averaging-scale or step control and no second route.

For 05_star:190 (row 17) the fallback stands: [to do: a received-redshift law against the emission clock and the released ball's multistream continuation are not read; W9 supplies only the ray map's exponent, which sustains no segment at 1/(4GE) at the collar].