# NOTES_D1_conformal_control v1.2 (v1 below, corrected in place where marked v1.1 or v1.2)

## Changes from v1.1 (per deliveries/lead_gaps/FableAPI_reading_D1_v1.1_delta.md, inbox #190)

v1.1 recorded the reading's R1 to R4 in its header but left the body unchanged. v1.2 carries them through the body, adds the missed R5 and M1, and also applies the delta's carried-over minors M2 and M5. Nothing was recomputed beyond reading three stored files: the harness npz/log, the ray map d1_raymap_v1.npz and its summary json.

- **R1 (the 2.2 plateau ratio).** Replaced at the three body sites (sec. 1 condition 5, sec. 5, sec. 10) by the like-for-like numbers: the lattice's converted part 0.92 +- 0.04 per copy is 1.45 to 1.5 times the sheet's c = pi value. The growth's 1.34 is named as its two parts.
- **R2 (the label).** Sec. 1's heading "Verdict: PASS WITH CONDITIONS" and its closing sentence are replaced by the restricted label and the plan sentence's restricted form. The v1.1 header's "CONFIRMS D1 = PASS WITH CONDITIONS" is marked as superseded.
  - One number in the reading's paste is corrected: "a quarter of its growth by 40 r_s" is about a third (0.36 of the growth 1.13 at 40 r_s). The 0.36 is a quarter of the plateau 1.34.
- **R3 (the band-edge mechanism).** Stated as a reading, not established, at sec. 1 condition 3 and sec. 6.
- **R4 ("stops growing").** It is now "levels off" (sec. 1 condition 4, sec. 5, sec. 10), with the lattice's converted part flat while the circle term still rises.
  - The rise is 0.0018 per r_s at the 2 r_s cut over 60 to 100 r_s, from the stored ray map (circle term +0.372 at 60, +0.445 at 100). The delta's paste says 0.003; the stored map gives 0.002.
- **R5 (missed in v1.1).** The harness's late rate is about a tenth of kappa/12, not zero. From a2_chain_hold_n1200.npz (67b6d191) and its log: DS/2 per sector rises 2.53341 -> 2.56239 over 1750 to 2000 a, which is 0.029, or 1.16e-4 per a.
  - Against kappa/12 = 0.0135784/12 = 1.13e-3 per a, that is 0.10 kappa/12.
  - Over 1800 to 2000 a it is 0.023 per sector (the delta's number), also 0.10 kappa/12. Over single 25 a intervals it ranges from 0.07 to 0.13 kappa/12.
- **M1 (the first return).** From the stored ray map d1_raymap_v1.npz (4bc3232e): the round trip from the 2 r_s cut to the wall and back is 259.1 to 260.7 r_s (D_out + D_in at each record's frozen fields; 260.2 on the T = 0 fields, 259.1 on the T = 100 fields).
  - Light leaving the 2 r_s cut at the start returns at 260.2 r_s.
  - The formation's light, crossing the cut at 28 r_s (Lam 5.4), returns at 287.6 r_s. Over its crossing clocks 8 to 33 r_s (Lam 0.1 to 7.9) it returns at 268 to 293 r_s.
  - So the delta's 259 and 288 are reproduced. v1's 266 is replaced; every cut stays clean to 100 r_s.
- **M2.** The static residual is one constant across the four outer cuts: 1.0e-3 to 1.1e-3, constant to 1e-4 at the printed digits (sec. 3).
- **M5.** The trunc column is labelled as the withdrawn mechanism's curve (sec. 4).

## Changes from v1 (v1.1)

(per deliveries/lead_gaps/FableAPI_reading_D1_v1.md, inbox #179)

- (v1.1's wording, superseded in v1.2 by the restricted label of sec. 1) The reading CONFIRMS D1 = PASS WITH CONDITIONS: the propagation certified to 4e-4 to 6.4e-3 nats per copy over the first ~3 e-foldings at every cut; both formula corrections (the finite ray-circle form; the held-vacuum observable) right; the window ~3 e-foldings right.
- PLATEAU CORRECTED (reading item 4): v1's '2.2 times the sheet's (1/12) ln(c/(T_H a))' compared the GROWTH (which contains the circle term this note introduced, +0.37 to +0.45 over 60 to 100 r_s) with a number that does not. Like for like, the lattice's CONVERTED PART (the growth less the circle and cutoff terms and the ingoing blueshift) levels off at 0.92 +- 0.04 per copy at the 2 r_s cut from 60 r_s (0.98 +- 0.09 at 16 r_s, 0.92 +- 0.15 at 24, both still drifting up), 11 to 12 e-foldings at the conformal rate: 1.45 to 1.5 times the sheet's c = pi value (0.637 at the record's charge) and 1.7 times its c = 1 value. The sheet's truncation is refuted either way (no c between 1 and pi gives 0.92; the sheet's own harness reaches 2.56 per sector). THE FACTOR TO PRINT IS 1.5, NOT 2.2; the growth's 1.34 +- 0.04 is to be named as its two parts (converted 0.92 plus the circle and cutoff terms).
- The band-edge explanation of the ~3 e-folding window (the partners of modes near 0.05/a reaching the band edge) is a READING, NOT ESTABLISHED (reading section 3).
- LABEL (reading item 6): PASS on the propagation and on the corrected formula within 3 e-foldings; FAIL for the sheet's formula as applied, its PASS criterion, its truncation, saturation and size law. The plan's D1 PASS sentence prints only in the restricted form (they certify each other to 3 e-foldings at every cut).
- Nothing moves a printed kappa = 1 number (reading item 5): the 18-channel sums and kappa 1's 8.24 nats at 100 r_s stand.

---

# NOTES_D1_conformal_control_v1: the conformal control (kappa = 0 chain on B2's fields) against the exact 1+1 Gaussian formula on the model's own ray map

solve lane, 20 Oct 2026 (box clock 19 Oct 21:xx). Plan item D1 (LEAD_fable_page_verify_routes_v1.md sec. A2 and the day-1 D1 bullet; the formula sheet FABLE_write_A2_v1.md; the engine busi ready_jobs_v3 sec. 16). The control run is v6s_B2_kz (page_history_v6s --kappa-zero, kappa 1's production line on B2's 8192-site chain, cuts 2, 16, 24, 30, 48 r_s, readouts every 2 r_s to 100 r_s; completed 03:47 UTC 20 Oct on the H100 card, exit 0, 0 lines of AUDIT FAIL / FLOOR WARNING / Traceback; the box mirror under <internal>/D1_kz_bench1b is byte-identical to the card, MIRROR_STATUS "D1_MIRROR_OK 15 files"). Everything numerical here was computed on the box (env qi2, nice 10) by the three scripts in solve/code/page_verify/ (sec. 9); the stored run was not re-propagated. Per copy means per S_stag sector = half the printed chain number (S_X = 2 S_Xc(C_+); each sector at kappa = 0 is one c = 1 Dirac fermion, busi sec. 16, confirmed by the vacuum check of sec. 4).

## 1. Verdict (v1.2): PASS on the propagation and on the finite-chain formula within 3 e-foldings at every cut; FAIL for the sheet's formula as applied, its PASS criterion, its truncation, saturation and size law

**PASS on the propagation.** Read on the state's own entropy growth per copy, [S_X(t) - S_X(0)]/2, the production code's massless chain agrees with the exact conformal prediction on the traced ray map to 4e-4 nats per copy at the 2 r_s cut from 0 to 18 r_s (blueshift Lam = -ln p' up to 2.0 e-foldings) and to 6e-3 at 20 r_s (Lam = 2.9); at the 16, 24, 30 and 48 r_s cuts to 7e-4, 3e-4, 1.4e-3 and 2.5e-3 through Lam = 2.1, 2.1, 2.5 and 2.8 (40, 50, 58 and 80 r_s). The five cuts' departures collapse onto one curve in Lam (figure, lower right). A separate static check at T = 0 (the circle form's prediction for the differences of the vacuum cut entropy between the five cuts) agrees with the lattice to 1.0e-3 to 1.1e-3 per copy. The propagation and the map certify each other wherever the continuum premise (partners inside the chain's band) holds.

**The conditions.**
1. The agreement is obtained only after two corrections to the sheet's formula, both on the formula's side (sec. 3, 7): (a) the in-state is the Killing vacuum of a finite chain, i.e. the conformal vacuum on a ray circle of circumference L = 265.31 r_s, and the exterior block is an arc covering most of that circle, so the line form (1/6) ln[(p(u) - p(u_0))/(u - u_0)] must be replaced by (1/6) ln[(L/pi) sin(pi l_int/L)] with l_int the p-arc of the rays inside the cut; this term is +0.11 per copy at 18 r_s, +0.24 at 30, +0.45 at 100 at the 2 r_s cut and the sheet omitted it; (b) the sheet's observable, the excess over the held Killing vacuum of the record, is ill-conditioned at kappa = 0: the massless chain's held vacuum on a trapped record jumps by up to 0.60 nats per copy between adjacent 2 r_s clocks (1.29 at T = 0, 1.47 at 12 r_s, between 1.49 and 2.09 from 20 r_s on), while the production's kappa = 1 held vacuum moves by at most 0.010 (2 r_s) and 4e-4 (16 r_s). The formula's reference (the conformal u-vacuum) changes by only (1/6) ln[D(t)/D(0)] = -0.003 over the run, so the formula's statement is about the state's growth; the lattice held vacuum's behaviour is a separate kappa = 0 interior effect.
2. **The sheet's PASS criterion, as written, FAILS** (excess per copy within 0.05 of its line formula from 8 through 28 r_s at 2 r_s): max|excess - sheet| = 0.23 at 16 r_s, mean -0.07 over 2 to 30 r_s; and the sheet's line formula misses the state's growth by 0.10 at 18 r_s, 0.13 at 20, 0.28 at 24, 0.31 at 28. The side that is wrong is the formula's application (sec. 7), not the propagation.
3. **The window is shorter than the plan's.** The formula applies for Lam <~ 2.5 to 3 (u - u_f about 10 r_s; through 20 r_s on the record clock at the 2 r_s cut, 42 at 16 r_s, 52 at 24, 58 at 30, 80 at 48), not the plan's Lam = 6.6 to 7.7 (u - u_f ~ 14 r_s, 26 to 30 r_s). (v1.2) The formula holds to about 3 e-foldings at every cut. That is where the formation's hard light reaches the cut (the kappa = 1 chain's formation jump, 0.02 to 0.11 per copy over 18 to 22 r_s at 2 r_s, coincides), and it is also where the partners of modes near 0.05/a would reach the band edge. Which of these ends the segment, and why the departure is positive, is open; the band-edge explanation is a reading, not established. The thermal count 6.6 to 7.7 is where the control crosses back through the formula.
4. **The departure is not the sheet's signature.** From Lam ~ 3 to 7.7 the lattice chain runs ABOVE the continuum formula by +0.04 to +0.12 per copy (the mean over the band-edge regime +0.06 at 2 r_s, +0.10 at 16, 24 and 48 r_s, +0.09 at 30 r_s), crosses it at Lam ~ 8 to 9 (34 r_s at 2 r_s; 54 to 56 at 16; 64 to 66 at 24; 72 to 74 at 30; 94 at 48), and falls below after: -0.17 at Lam 12, -0.64 at 18, -1.15 at 25, -3.37 at 100 r_s (Lam 51.7) at the 2 r_s cut, while the formula continues at 0.055 nats per r_s per copy. The chain's growth then levels off (sec. 5; v1.2: levels off, not stops).
5. **The saturation height is not (1/12) ln(c/(T_H a)).** The growth per copy at the 2 r_s cut is 1.34 +- 0.04 over 60 to 100 r_s (1.28 to 1.40; 1.39 at 100), the printed excess 0.72 +- 0.14 (0.52 to 1.01; 0.59 at 100); at 16 r_s 1.08 +- 0.09 and 0.77 +- 0.14 (both still drifting up). The sheet predicted 0.53, 0.57, 0.62 (c = 1, pi/2, pi) from 37 to 39 r_s. (v1.2) Of the growth, 0.37 to 0.45 is the circle term's own slow rise and 0.92 +- 0.04 the lattice's converted part. That is 11 to 12 e-foldings at the conformal rate: 1.45 to 1.5 times the sheet's c = pi value (0.637 at the record's charge) and 1.7 times its c = 1 value. No cutoff between 1 and pi gives it; it would need c of about 90. The excess over the held vacuum is 0.72 +- 0.14, and the reference itself is unsteady by 0.6. reviewer A's own chain known-answer run (a2_chain_hold_n1200, finished on the box at 19:49) also shows no saturation at (1/12) ln(c/(Ta)) = 0.51 to 0.61 per sector: that chain follows its line formula to within 0.03 up to 6 e-foldings, then rises at up to 2.7 times kappa/12 and reaches 2.56 per sector at 2000 a against the formula's 1.72 and the predicted saturation 0.6 (the sheet itself says that if the chain does not saturate, its TRUNCATION part is withdrawn).

(v1.2) So the plan's D1 sentence holds only in this form: the propagation and the map certify each other to 3 e-foldings at every cut. The companion, the finite-chain conformal formula on the state's growth, covers the control's first 0.36 nats per copy: about a third of its growth by 40 r_s (1.13) and a quarter of its plateau (1.34). The kappa = 1 term's relation to it is the measured ratio 0.13 to 0.19 by growth, not a prediction. Nothing in the control questions the printed kappa = 1 numbers (sec. 6).

## 2. What was compared

- Control: v6s_B2_kz_k1_d13.json (md5 8db1de1196f40b617d4b3c26816e9725), per clock and cut: S_X (the exterior block's entropy, both sectors), S_X_vac (the held vacuum P_-(h(t)) of the same record), excess, excess_floor_high. Args checked in the json: kappa_zero True, kappas 1-1, nsites 8192, interp linear, interp_sub 8, cheb_tol 1e-16, sector auto, sector_propagate on, cuts 2,16,24,30,48, T_end 100; log pass lines present ("--kappa-zero: channel(s) [1] run with the mass term kappa/R switched off", "sector check: max|h + conj h| 0.0e+00, max|[S_stag, h]| 0.0e+00 ... the halving USED", "8192 occupied of 16384, 0 levels within 1e-12 of zero"). The floors (high) are 6.5e-9 to 3.3e-5 per copy.
- Production kappa = 1 on the same history: v6o_B2_k1_d13.json (50fe59a03edc93837980127cbbcec51b), same readout, same cuts and clocks.
- The map: traced by d1_raymap_data_v1.py (my own code following the sheet's recipe: the applied fields bilinear in (t, R), the centre column N(0) = N(1), v(0) = 0, the PG exterior at the record's charge beyond 4096 a to the wall at 8192 a as the production appends it, fields frozen before T = 0; RK4 step 0.25 a; every Jacobian from the separation equation). It reproduces the sheet's a2_raymap_B2.npz (65498c6cb0ef54801859740c4e7cb7b1): p_H = -43.4118 r_s, the affine offset u - p = 136.0208 r_s, the lag u_c - t = 77.16 r_s at 50 r_s, Lam(100 r_s, 2 r_s cut) = 51.735, and the sheet's dS_wall at every clock and cut to 4.5e-5 (the sheet's table: 0.235, 0.524, 0.985 at 20, 30, 40 r_s; mine 0.2346, 0.5237, 0.9851). Step-size test: dt 0.5 against 0.25 changes dS_circ by at most 7e-5 per copy and Lam by 7e-4.
- First return (v1.2, from the stored ray map d1_raymap_v1.npz): the round trip from the 2 r_s cut to the wall and back is 259.1 to 260.7 r_s. Light leaving the cut at the start is back at 260.2 r_s. The formation's light, crossing the cut at 28 r_s (Lam 5.4), is back at 287.6 r_s (268 to 293 for crossings at 8 to 33 r_s). At the 48 r_s cut the round trip is 161.7 r_s, so the formation's light, crossing it from 67 r_s on, is back after 228 r_s. All cuts are clean to 100 r_s. (v1 printed 266.)

## 3. The formula as applied (the finite-chain form)

Per copy, with p the crossing time of x = -R_ref (R_ref = 4096 a) of the in-leg of a ray (the sheet's p), the state's exterior-block entropy at the cut R_c and record clock t is

S(t) = (1/6) ln[(L/pi) sin(pi l_int(t)/L)] + (1/12)[Lam_out(t) + Lam_in(t)] + (1/12) ln[f_out(R_c, t) f_in(R_c, t)] + const,

L = 265.31 r_s the round trip of the initial static chain (the circumference of the ray circle on which the initial Killing vacuum is the conformal vacuum; the chain has no zero level at T_0, NS sector), l_int = p_in - p_out the p-arc of the rays currently inside the cut (p_out of the out-ray at the cut traced back through the centre, p_in of the in-ray at the cut traced back to R_ref; l_int(0) = 5.13, 37.4, 54.2, 66.6, 103.5 r_s at the five cuts, growing by about 1 per unit t once the in-rays are trapped), Lam = -ln(dp/dt_cut) at each endpoint (the out-endpoint's Lam_out is the sheet's -ln p'(u_c) up to du_c/dt; Lam_in <= 0.03 over the run), f_out = N - |v| and f_in = N + |v| at the cut (the lattice spacing is a (dp/dt)/f wide in p at each endpoint; the cut sits inside the initial diffuse star, N = 0.856, v = -0.276 at 128 a at T = 0 against 0.98, -0.63 at 20 r_s, so this term is -0.013 per copy at 22 r_s and +0.002 at 100). The prediction for the growth is dS_circ = S(t) - S(0); the sheet's dS_wall is the same expression with the line form (1/6) ln[(L - l_int(t))/(L - l_int(0))] in place of the sine term and without the cutoff term; its semi-infinite form drops the finite-size terms altogether. The sheet's u_0 = u_c - 2D uses the outgoing travel time for the ingoing leg (D_out = 148 to 150 r_s against D_in = 110 to 112 r_s at the 2 r_s cut); the variant with the actual ingoing leg (dS_wall_in in the npz) differs from dS_wall by at most 0.01 and is not what fixes the discrepancy.

Static check at T = 0: the lattice's S_vac(R_c) - S_vac(2 r_s) per copy = 0.3568, 0.4143, 0.4433, 0.4914 at 16, 24, 30, 48 r_s; the circle form gives 0.3558, 0.4132, 0.4422, 0.4903; residuals 0.0010, 0.0011, 0.0011, 0.0011: (v1.2) one constant across the four outer cuts, to 1e-4 at the printed digits. The line form predicts nothing here (its reference is the interval length 2D, which is nearly the whole circle).

## 4. Tables (nats per copy; the full 52-clock tables for all five cuts are in d1_conformal_control_v1_tables.md)

Columns: Lam = e-foldings of blueshift at the cut's outgoing endpoint; growth = the control's [S_X(t) - S_X(0)]/2; circle = the finite-chain formula; excess = the control's printed [S_X - S_X^vac]/2; sheet = the sheet's dS_wall; trunc = the sheet's truncated curve at c = pi (v1.2: the withdrawn mechanism's curve, printed for reference only); vac = the control's held vacuum per copy; k1 = the production kappa = 1 chain, growth and excess per copy.

### 2 r_s cut (the sheet's window: 8 to 28 r_s)

| t | Lam | growth | circle | growth - circle | excess | sheet | excess - sheet | trunc | vac | k1 growth | k1 excess |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | 0.10 | 0.0112 | 0.0110 | +0.0002 | 0.0114 | 0.0120 | -0.0006 | 0.0120 | 1.2921 | 0.0028 | 0.0033 |
| 10 | 0.26 | 0.0288 | 0.0285 | +0.0003 | 0.0073 | 0.0208 | -0.0135 | 0.0208 | 1.3139 | 0.0075 | 0.0034 |
| 12 | 0.47 | 0.0605 | 0.0601 | +0.0004 | -0.1194 | 0.0344 | -0.1538 | 0.0344 | 1.4722 | 0.0136 | -0.0006 |
| 14 | 0.72 | 0.1069 | 0.1065 | +0.0004 | -0.0823 | 0.0589 | -0.1413 | 0.0590 | 1.4815 | 0.0166 | -0.0030 |
| 16 | 1.23 | 0.1749 | 0.1746 | +0.0004 | -0.1296 | 0.1007 | -0.2303 | 0.1007 | 1.5969 | 0.0198 | 0.0005 |
| 18 | 2.02 | 0.2686 | 0.2683 | +0.0003 | 0.0245 | 0.1651 | -0.1406 | 0.1651 | 1.5364 | 0.0227 | 0.0060 |
| 20 | 2.90 | 0.3630 | 0.3694 | -0.0064 | 0.1672 | 0.2346 | -0.0674 | 0.2343 | 1.4881 | 0.0533 | 0.0369 |
| 22 | 3.47 | 0.4916 | 0.4404 | +0.0512 | 0.2101 | 0.2764 | -0.0663 | 0.2763 | 1.5738 | 0.1144 | 0.1000 |
| 24 | 3.96 | 0.5958 | 0.5058 | +0.0899 | 0.2182 | 0.3194 | -0.1012 | 0.3193 | 1.6699 | 0.1042 | 0.0905 |
| 26 | 4.48 | 0.6205 | 0.5699 | +0.0505 | 0.3855 | 0.3622 | +0.0233 | 0.3625 | 1.5273 | 0.0922 | 0.0778 |
| 28 | 5.44 | 0.7540 | 0.6685 | +0.0855 | 0.4679 | 0.4416 | +0.0263 | 0.4416 | 1.5784 | 0.1083 | 0.0941 |
| 30 | 6.44 | 0.8215 | 0.7683 | +0.0532 | 0.5069 | 0.5237 | -0.0168 | 0.5229 | 1.6069 | 0.1009 | 0.0872 |
| 32 | 7.37 | 0.8995 | 0.8615 | +0.0381 | 0.6450 | 0.6009 | +0.0441 | 0.5838 | 1.5468 | 0.1264 | 0.1129 |
| 34 | 8.53 | 0.9311 | 0.9717 | -0.0405 | 0.7382 | 0.6965 | +0.0418 | 0.6145 | 1.4852 | 0.1195 | 0.1048 |
| 36 | 9.77 | 1.0005 | 1.0876 | -0.0871 | 0.3011 | 0.7987 | -0.4976 | 0.6232 | 1.9917 | 0.1229 | 0.1097 |
| 40 | 12.02 | 1.1292 | 1.2984 | -0.1692 | 0.9139 | 0.9851 | -0.0712 | 0.6249 | 1.5076 | 0.1576 | 0.1442 |
| 50 | 18.22 | 1.2233 | 1.8591 | -0.6358 | 0.8739 | 1.4961 | -0.6222 | 0.6192 | 1.6416 | 0.1733 | 0.1603 |
| 60 | 24.85 | 1.2986 | 2.4445 | -1.1459 | 1.0132 | 2.0427 | -1.0295 | 0.6128 | 1.5777 | 0.2355 | 0.2221 |
| 70 | 31.47 | 1.3373 | 3.0213 | -1.6840 | 0.8854 | 2.5877 | -1.7023 | 0.6061 | 1.7442 | 0.2406 | 0.2274 |
| 80 | 38.20 | 1.3157 | 3.6031 | -2.2874 | 0.6367 | 3.1424 | -2.5058 | 0.5991 | 1.9713 | 0.2481 | 0.2352 |
| 90 | 44.98 | 1.3455 | 4.1833 | -2.8377 | 0.7806 | 3.6996 | -2.9190 | 0.5917 | 1.8572 | 0.2591 | 0.2461 |
| 100 | 51.73 | 1.3872 | 4.7588 | -3.3716 | 0.5941 | 4.2552 | -3.6611 | 0.5840 | 2.0854 | 0.2705 | 0.2575 |

The clocks 0 to 6 r_s agree to <= 1e-4 (not listed). At 2 r_s the e-foldings pass 1, 3, 6.6, 7.05, 7.7, 10, 20 at t = 15.2, 20.4, 30.3, 31.3, 32.6, 36.4, 52.7 r_s.

### 16 r_s cut (the sheet's window: 28 to 48 r_s; the lag behind 2 r_s is 20.5 r_s)

| t | Lam | growth | circle | growth - circle | excess | sheet | excess - sheet | k1 growth | k1 excess |
|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.16 | 0.0145 | 0.0145 | -0.0000 | -0.0757 | 0.0131 | -0.0888 | 0.0001 | 0.0002 |
| 34 | 0.44 | 0.0422 | 0.0423 | -0.0001 | 0.0010 | 0.0365 | -0.0355 | 0.0003 | 0.0004 |
| 36 | 0.76 | 0.0722 | 0.0723 | -0.0001 | -0.3021 | 0.0625 | -0.3646 | 0.0002 | 0.0003 |
| 38 | 1.30 | 0.1220 | 0.1223 | -0.0002 | -0.0872 | 0.1068 | -0.1940 | 0.0002 | 0.0002 |
| 40 | 2.12 | 0.1958 | 0.1965 | -0.0007 | 0.1480 | 0.1741 | -0.0261 | 0.0011 | 0.0011 |
| 42 | 2.85 | 0.3107 | 0.2633 | +0.0474 | 0.0921 | 0.2331 | -0.1410 | 0.0983 | 0.0983 |
| 44 | 3.54 | 0.4287 | 0.3277 | +0.1010 | 0.3592 | 0.2889 | +0.0702 | 0.1049 | 0.1048 |
| 46 | 4.03 | 0.4597 | 0.3747 | +0.0850 | 0.4193 | 0.3296 | +0.0897 | 0.0831 | 0.0829 |
| 48 | 4.75 | 0.5311 | 0.4404 | +0.0907 | 0.0601 | 0.3879 | -0.3277 | 0.0831 | 0.0829 |
| 50 | 5.88 | 0.6486 | 0.5403 | +0.1083 | 0.5442 | 0.4806 | +0.0636 | 0.0957 | 0.0956 |
| 52 | 6.82 | 0.7302 | 0.6231 | +0.1070 | 0.4635 | 0.5566 | -0.0931 | 0.1038 | 0.1042 |
| 54 | 7.93 | 0.7491 | 0.7205 | +0.0287 | 0.6655 | 0.6475 | +0.0181 | 0.1032 | 0.1038 |
| 56 | 9.20 | 0.8041 | 0.8316 | -0.0275 | 0.4099 | 0.7526 | -0.3427 | 0.1128 | 0.1134 |
| 60 | 11.50 | 0.9501 | 1.0323 | -0.0821 | 0.8776 | 0.9412 | -0.0636 | 0.1382 | 0.1388 |
| 70 | 17.79 | 1.0330 | 1.5761 | -0.5431 | 0.8714 | 1.4583 | -0.5869 | 0.1606 | 0.1612 |
| 80 | 24.43 | 1.0988 | 2.1443 | -1.0455 | 0.7512 | 2.0037 | -1.2525 | 0.2193 | 0.2200 |
| 90 | 31.10 | 1.1743 | 2.7127 | -1.5384 | 0.9268 | 2.5521 | -1.6253 | 0.2286 | 0.2292 |
| 100 | 37.87 | 1.1413 | 3.2856 | -2.1443 | 0.6629 | 3.1076 | -2.4446 | 0.2357 | 0.2363 |

### 24, 30 and 48 r_s cuts (the regime boundaries)

| cut | agreement regime (Lam < ~2.5-3): clocks, max|growth - circle| | band-edge regime (3 <= Lam < 7.7): clocks, mean / max (growth - circle) | zero crossing | at 100 r_s: growth, circle, excess |
|---|---|---|---|---|
| 24 r_s | 2 to 50 r_s, 3e-4 (then +0.040 at 52 r_s, Lam 2.85) | 54 to 62 r_s, +0.103 / +0.114 | 64 to 66 r_s (Lam 7.9 to 9.2) | 1.152, 2.675, 0.752 |
| 30 r_s | 2 to 58 r_s, 1.4e-3 (then +0.064 at 60 r_s, Lam 3.00) | 62 to 70 r_s, +0.086 / +0.103 | 72 to 74 r_s (Lam 8.3 to 9.6) | 1.178, 2.244, 0.828 |
| 48 r_s | 2 to 80 r_s, 2.5e-3 (then +0.088 at 82 r_s, Lam 3.38) | 82 to 92 r_s, +0.099 / +0.121 | 94 r_s (Lam 9.0) | 0.952, 1.057, 0.723 |

## 5. The departure and the plateau against the sheet's numbers

- Onset: the first departure beyond 0.01 per copy is at Lam = 2.85 to 3.4 at every cut (22 r_s at 2 r_s; 42 at 16; 52 at 24; 60 at 30; 82 at 48). The sheet's boundary (Lam = 6.6 to 7.7; 30.5 to 33 r_s at 2 r_s) is where the control crosses back through the formula, not where it leaves it.
- Sign: above the formula by +0.04 to +0.12 per copy over 3 <~ Lam <~ 7.7 at all cuts (the sheet's truncation predicted below). The oscillation between adjacent clocks in this regime (+0.05, +0.09, +0.05, +0.09, +0.05 at 22 to 30 r_s) is in the control's growth, not in the formula.
- Shortfall: growth - circle = -0.04, -0.09, -0.13, -0.17 at Lam = 8.5, 9.8, 10.9, 12.0 (34 to 40 r_s at 2 r_s); -0.64 at Lam 18; -1.15 at 25; -1.68 at 31; -3.37 at 52 (100 r_s). The curve growth - circle against Lam is the same at the five cuts to about 0.05 (figure).
- Plateau of the growth per copy: 2 r_s cut 1.34 +- 0.04 over 60 to 100 r_s (the last four clocks 1.398, 1.398, 1.354, 1.387; the drift over 60 to 100 about 0.002 per r_s, within the +-0.04 wiggle); 16 r_s 1.08 +- 0.09 (0.95 at 60, 1.14 at 100, still rising at about 0.005 per r_s); 24 r_s 0.98 +- 0.15 (1.15 at 100); 30 and 48 r_s still forming (1.18 and 0.95 at 100). The excess over the held vacuum: 0.72 +- 0.14 (2 r_s), 0.77 +- 0.14 (16 r_s), 0.73 +- 0.19 (24 r_s) over 60 to 100 r_s. Per chain double these; for the sixteen chains of kappa = 1 at the 2 r_s cut: 43 nats (growth) or 23 nats (excess) at 60 to 100 r_s, against the sheet's "16 to 19 at 100 (truncated)" and its 0.53 to 0.62 per copy. The sheet's (1/12) ln(pi/(T_H a)) = 0.64 per copy (0.58 to 0.62 on its curves at 40 to 100 r_s) is (v1.2) 1.45 to 1.5 times below the lattice's converted part (0.92 +- 0.04) and 1.2 to 1.4 times below the mean excess. The growth's 1.34 also holds the circle term's +0.37 to +0.45.
- The late rise of the control at the 2 r_s cut is at most 0.002 per r_s per copy (from 60 r_s on), i.e. (v1.2) the lattice's converted part is flat there while the circle term still rises by 0.002 per r_s (0.0018 over 60 to 100 r_s on the stored map), and the production's kappa = 1 chain still rises at 0.0009 per r_s per copy (0.2355 at 60 to 0.2705 at 100; 0.028 nats per r_s for the channel).
- reviewer A's own harness (a2_chain_hold_n1200: 1200 sites, flow held from 60 a, kappa = 0.01358/a, Ta = 2.16e-3, cut at 200 a): DS_chain/2 over its line formula = 0.97, 1.01, 1.04, 1.07, 1.08, 1.07, 1.07, 1.08 at 4.1 to 6.4 e-foldings (the 3 to 8 percent excess is of the size of the finite-chain term for that geometry, L = 2400 a and l_int(0) = 400 a, estimated 0.04 to 0.07 per sector; not computed exactly here), then 1.10, 1.20, 1.33, 1.46, 1.58, 1.73, 1.90, 1.95 at 6.8 to 13.9 e-foldings, and 2.56 per sector at 2000 a against the formula's 1.72: the chain's rate rose to 2.7 kappa/12 at 8 to 12 e-foldings and fell to about a tenth of kappa/12 by 1800 a (v1.2: 0.029 per sector over 1750 to 2000 a and 0.023 over 1800 to 2000 a, i.e. 1.15e-4 to 1.16e-4 per a = 0.10 kappa/12; 0.07 to 0.13 kappa/12 over single 25 a intervals; a2_chain_hold_n1200.npz). The predicted saturation 0.51 to 0.61 per sector did not occur. That harness has a cavity between an inner root at 17.9 a and the outer root at 87.6 a with |v| up to 1.5 and N = 1; B2's interior differs (N down to 0.14 at the centre, a thin trapped shell at 10 to 35 a), so its burst is not expected in B2 and is not seen: B2's control shows the +0.1 overshoot and then the plateau.

## 6. The production kappa = 1 chain beside the control (per copy)

| clock | k1 growth 2 r_s | k1 excess 2 r_s | copy growth 2 r_s | copy excess 2 r_s | ratio k1/copy (growth, excess) | k1 excess 16 r_s | copy growth 16 r_s |
|---|---|---|---|---|---|---|---|
| 20 | 0.053 | 0.037 | 0.363 | 0.167 | 0.15, 0.22 | -0.0005 | -0.001 |
| 30 | 0.101 | 0.087 | 0.822 | 0.507 | 0.12, 0.17 | 0.0002 | 0.015 |
| 40 | 0.158 | 0.144 | 1.129 | 0.914 | 0.14, 0.16 | 0.0011 | 0.196 |
| 60 | 0.236 | 0.222 | 1.299 | 1.013 | 0.18, 0.22 | 0.139 | 0.950 |
| 100 | 0.271 | 0.258 | 1.387 | 0.594 | 0.19, 0.43 (0.36 against the 60-100 mean excess 0.72) | 0.236 | 1.141 |

The kappa = 1 excess at 100 r_s, 0.2575 per copy, times 32 is 8.24 nats: the channel's printed 8.2 is reproduced from the stored run. The kappa = 1 held vacuum is smooth (1.0276 to 1.0473 at 2 r_s, 1.3748 to 1.3757 at 16 r_s), so the production's excess is a well-conditioned observable; only the kappa = 0 control's is not. The sheet's "a quarter and a half of the sixteen chains' conformal saturation at 40 and 100 r_s" becomes, with the measured copy: 0.13 to 0.16 at 40 r_s and 0.19 (growth) to 0.36 (mean excess) at 100 r_s. The kappa = 1 chain's formation jump (0.0227 to 0.1144 per copy over 18 to 22 r_s at 2 r_s; 0.0011 to 0.0983 over 40 to 42 r_s at 16 r_s) coincides with the control's departure from the formula at Lam ~ 3 (v1.2: both arrive when the formation's hard light reaches the cut; that this ends the formula's segment, and the band-edge explanation, are a reading, not established).

## 7. Which side is wrong and how I know

The propagation is not in question:
1. With the finite-chain formula the state's growth agrees to 4e-4 per copy over 10 clocks at the 2 r_s cut while three distinct pieces vary (Lam/12 up to 0.17, the circle term up to 0.11, the cutoff term -0.01 as the fields at the cut change from N = 0.86, v = -0.28 to 0.99, -0.63), and to <= 2.5e-3 at the other four cuts through Lam ~ 2.5 to 2.8; the departure beyond is a function of Lam alone, the same at five cuts whose clocks differ by up to 60 r_s.
2. The static check at T = 0 (sec. 3) ties the circle form to the lattice vacuum at 1e-3 across five cuts.
3. The ray map is reproduced by independent code to 4.5e-5 in dS_wall and is converged in the step size to 7e-5.
The sheet's application is what fails:
4. Its line form treats an interval that is 98 percent of the ray circle (2D = 258 to 300 r_s against L = 265 r_s) with the line formula; the error is the circle term, +0.11 per copy at 18 r_s, +0.24 at 30, +0.45 at 100 (2 r_s cut), smaller at the outer cuts (+0.02 at 40 r_s, +0.13 at 100 at 16 r_s). The sheet's known-answer checks (affine, Carlitz-Willey, Walker-Davies) were on the line formula's own closed forms and do not see this.
5. Its observable, the excess over the lattice held vacuum, carries the kappa = 0 held vacuum's jumps (0.60 per copy between adjacent clocks); the formula's own reference changes by 0.003.
6. Its lattice-scale prediction (saturation at (1/12) ln(c/(Ta)) by a sharp truncation of the thermal integral at omega e^Lam = c/a) is contradicted by this control (overshoot, then a plateau whose converted part is 1.45 to 1.5 times higher (v1.1)) and by its own harness run (no saturation, 2.56 per sector).
Nothing here tests the physical channels' propagation beyond what the control shares with them (the same code path at kappa = 0); the kappa = 1 held-vacuum smoothness and the reproduced 8.2 nats are consistency statements.

## 8. Error budget for the comparison in the agreement regime (per copy)

| item | size | how established |
|---|---|---|
| propagation floor (excess_floor_high/2) | <= 3.3e-5 (<= 1.2e-5 before 40 r_s) | the run's own floors |
| ray-map numerics | <= 7e-5 (dS_circ, dt 0.5 vs 0.25); 4.5e-5 (dS_wall, this tracer vs the sheet's) | d1_raymap_v1_dt05.npz; a2_raymap_B2.npz |
| the circle form's lattice constant across cuts | 1.0e-3 to 1.1e-3 | the static check at T = 0 |
| the ingoing endpoint (exterior evolution) | Lam_in <= 0.03, i.e. <= 2.5e-3, included explicitly | the trace |
| the local cutoff term (1/12) ln(f_out f_in) | up to -0.013, included; its residual is bounded by the 4e-4 agreement over 8 to 18 r_s where the cut's fields change most | the data |
| the ray-label correction for in-rays with p_in > 0 (t > 50 r_s at 2 r_s) | <= 2e-3 (the exterior round trip changes by <~ 0.4 r_s against l_int >= 50 r_s) | estimate, not computed |
| record interpolation | common to both (linear in t in both) | by construction |
| defensible total in the agreement regime | about 1.5e-3 | |

The observed max|growth - circle| in that regime: 4e-4 (2 r_s, to 18 r_s; 6.4e-3 at 20 r_s where Lam = 2.9 is already at the edge), 7e-4 (16 r_s, to 40 r_s), 3e-4 (24 r_s, to 50), 1.4e-3 (30 r_s, to 58), 2.5e-3 (48 r_s, to 80). The sheet's tolerance of 0.05 is met by a factor 20 to 100 in this regime and violated by a factor 2 in the band-edge regime.

## 9. Files (all md5 recorded at the time of use)

Inputs read whole: deliveries/lead_gaps/LEAD_fable_page_verify_routes_v1.md fa7e963d3e8ca7bea533c999aebfcab4; deliveries/lead_gaps/FABLE_write_A2_v1.md e8b608064079b7b99f57774135819ab2; busi/results/ready_jobs_v3.md at origin/builder-numerics bc7c62c5df25bfe9c583d974bb264549df48eaf2 (content md5 ad892a7359087cbb48fea46cebf9823d; last changed in 3d6c7a7c62daf2e22fdc0f574fe5b5e7c489a6ba), section 16; deliveries/claude_alt_solve/PAGE_v4.1_numbers_for_paper.md 571e89b5b331ff055b9f71fa18937a93; PAGE_solved_history_v4.1.md f6a614240f163485fd5aa45b9bc4083b (searched for the kappa 1 and per-copy statements: it contains none; its "no saturation ... on this epoch" is about the 18-channel sum, which the kappa = 1 run confirms is still rising).
The sheet's scripts and data (deliveries/lead_gaps/FABLE_write_A2_scripts/, box <internal>/fable_A2): a2_raymap_B2.npz 65498c6cb0ef54801859740c4e7cb7b1; a2_raymap_B2.py 7b56f606b0f837208578bef497745aa5; a2_summary_B2.log a19f5decabd5cd2dcde47362a828a65e; a2_raymap_B2.log 4bafe493ced330b13856068b28e917e3; a2_chain_mirror.py e66992a8cd324b04ed99c5eed6f126c2; the completed box log a2_chain_hold_n1200.log 02039a9cd08741a48be0c00a12521c69 (16137 bytes; the deliveries copy 5ebf2c5684261d979ffe1dc78848d3aa is the 307-byte stub written while it ran) and a2_chain_hold_n1200.npz 67b6d19101d77f8752442f8dd5bd778e.
The control run (card /workspace/claude_alt_solve/page/ and the identical box mirror <internal>/D1_kz_bench1b): runs/v6s_B2_kz/v6s_B2_kz_k1_d13.json 8db1de1196f40b617d4b3c26816e9725; v6s_B2_kz_k1_d13_tables.md 27804cde5d06caee8a1664e015309e64; v6s_B2_kz_k1_d13.npz 85a3908e7d75b54b3a0590b4fe5e5d17; v6s_B2_kz_k1_summary.json b33a3531de1d76d18cd647bc39f6629d; v6s_B2_kz_k1_tables.md adc956169b8a07dbf3969e5b056c3ec5; v6s_B2_kz_ksum.json 2913a6373d0302a009ed0ab2527d6ef4; v6s_B2_kz_ksum.md 8db6cdd59124c23ba242d64ba4a7db08; logs/v6s_B2_kz.log 3d49bc4a5b390ab5827fb62123ff33d7.
The engine (card and box identical): page_history_v6s.py 611d190f5047d0a65dfc1c72866cc69e; page_interior_v6s.py 15e08e2807a3bce6e565306de023e80f; launch_v6s_B2_card_v2.sh 1f6944666adecdd1eaced8c5218336e1; page_d13_readout_v1d.py 70383bb15fa5c642a1ac7d570e2f923e; page_w1_sector_v1d.py 0eaed28a4caaa809ea03cc92e2baf4f2; page_ops_v6.py 31a85c000cecb4425f5af66828392ab6; page_core_v6.py c77b8d0df7d00feeb970e206250746b7.
The production and the history (box): <internal>/v6o_B2_k1_d13.json 50fe59a03edc93837980127cbbcec51b; v6o_B2_k1_d13_tables.md 35339b4951b0d095107e764d28e2d166; <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096_fields.npz cb26fcfaf3d6e80a17f0e73bac687301; ..._B2n4096.json 417a4018cf6497f9b0020408783f44f5.
Scripts (solve/code/page_verify/, also in the box folder .../D1_kz_bench1b/analysis/): d1_raymap_data_v1.py fba717fbe248ef62aaffb971c58dca27 (data: the map and both formulas); d1_control_data_v1.py 0612fd8a858896669a8f2637c1d008fd (data: the comparison); d1_conformal_control_fig_v1.py 9770f33dfcf31a6839d988e361fff6e9 (figure).
Results (solve/results/page_verify/ and the box folder): d1_raymap_v1.npz 4bc3232ee524ce770c1d233680ded991, d1_raymap_v1_summary.json 77ed934a5b6c9391596142907ba57dad, d1_raymap_v1.log d37451d8b3dcfe596efa55c5be4b93d6; d1_raymap_v1_dt05.npz a9d1bb5d45e1f1a5eb77661cd08e9dd0, d1_raymap_v1_dt05_summary.json d24c05a085845129c7c80f3a61426447, d1_raymap_v1_dt05.log eaf7f5dec5608351c6a14af3b534fe8f; d1_conformal_control_v1.npz cf933e0231e0e05b7fcaba78d1628b2e, d1_conformal_control_v1_summary.json 25575ed0e8fa05a6a0c317c0f2041755, d1_conformal_control_v1_tables.md 09737304477747024ff44c2b09d60c9e, d1_control_data_v1.log 341ab34dda6ca3035a7fbde1bf759281; d1_conformal_control_v1.png a7177e97e7ba0eb918380e3fb49cc4fd (five panels: control growth and excess, the circle formula, the sheet's line and truncated curves, the kappa = 1 chain, against the record clock, the Lam < 3 window shaded and the band-edge regime hatched; sixth panel: growth - formula against Lam at all cuts).

## 10. What the paper must not print, and what the control supports instead

Must not print (from the sheet's draft paragraph and outcome sentences):
- "the copy's excess saturates at (1/12) ln(c/T_H a), 0.55 to 0.65 nats per copy at r_s/a = 64" and the size law "0.53, 0.59, 0.64 at r_s/a = 16, 32, 64 (0.06 per copy per doubling)": the measured plateau per copy is 1.34 +- 0.04 (growth; v1.2: of which the lattice's converted part is 0.92 +- 0.04, 1.45 to 1.5 times the sheet's c = pi value, and the circle term +0.37 to +0.45) and 0.72 +- 0.14 (excess) at the 2 r_s cut; the mechanism is an overshoot followed by a levelling off, not a truncation; the size law is untested (B4).
- "the chain on the formula through 28 to 30 r_s (u - u_f <~ 14 r_s), i.e. a static lattice horizon converts a mode for the span (1/kappa) ln(c/omega a)": the formula holds to Lam ~ 3 (20 r_s at 2 r_s; about 10 r_s of u - u_f); the lattice leaves it at Lam ~ 3 to 3.5 and crosses back at Lam ~ 8 to 9.
- "17 to 20 nats at 40 r_s and 16 to 19 at 100 r_s for the sixteen chains" and "the kappa = 1 term is a quarter and a half of the conformal saturation": the measured sixteen-chain copy numbers are 36 (growth) or 29 (excess) at 40 r_s and 43 or 23 at 60 to 100 r_s; the kappa = 1 ratios are 0.13 to 0.16 at 40 r_s and 0.19 to 0.36 at 100 r_s.
- Eq. (page-conformal) in its line form as the companion of a lattice number, and "the excess over the held Killing vacuum" as the quantity it predicts at kappa = 0.
- Any "0.64 per copy, 10.3 for the 16 copies" (the plan's) or "per chain 1.28, 20.5 for sixteen chains" (the sheet's).
Supported by the control:
- The conformal copy on the same fields follows the finite-chain conformal formula on the traced ray map to 1e-3 nats per copy while the blueshift at the cut is below about 3 e-foldings (0.36 nats per copy at 20 r_s at the 2 r_s cut; the same at the 16, 24, 30 and 48 r_s cuts at their own clocks), runs 0.04 to 0.12 above it while the burst's partners cross the band edge (3 to 8 e-foldings), and (v1.2) levels off at 1.34 +- 0.04 nats per copy (2.7 per chain) from 60 r_s at the 2 r_s cut. Within that, the lattice's converted part is flat at 0.92 +- 0.04 while the circle term still rises by 0.002 per r_s, and the 16 and 24 r_s cuts still rise by 0.005 and 0.003 per r_s at 100. Meanwhile the formula continues at 0.055 nats per r_s per copy; its excess over the held vacuum of the record is 0.72 +- 0.14 per copy there, the held vacuum of the massless chain on a trapped record being itself unsteady by up to 0.6 per copy between records.
- The ray map's numbers (p_H, the peeling rate 0.1 to 0.9 kappa_H through the formation and 1.12 kappa_H from 35 r_s, the 20.5 r_s lag of the 16 r_s cut) are reproduced and can be printed as the sheet gives them.
- The kappa = 1 chain's printed numbers are unchanged (8.24 nats at 100 r_s at 2 r_s from the stored run); its held vacuum is smooth, so its excess is well-conditioned.
