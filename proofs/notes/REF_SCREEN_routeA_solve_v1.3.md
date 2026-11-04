# REF_SCREEN_routeA_solve_v1.3: v1.2 (md5 4ca4b8a9, unchanged) + section 11 (the c17 early-leg stage-1 screen, records 0-88, by forward exact replay, with the end-to-end gate at record 96). Lead #707 / #735, 2026-11-01.

**Frozen.** results/mark3_layer_r1/zone640/ADDENDUM_707_ref_v1.md (md5 87ef1164c2a527645b01f3a899fc8455), frozen before any number. Amendment v1a (744c5013a211b4c7b6f32bb301203a67) changed ONLY the V0 tolerance to roundoff, before any W2-inclusive number (section 5).

**Scripts** (box <internal>/code, md5-matched local copies in results/mark3_layer_r1/scripts/):
- ref707_stage1_v1.py 7a92560c (P*-part only, because of the over-strict V0);
- ref707_stage1_v2.py dce918f5 (the result);
- ref707_spread_v1.py 857bc220;
- ref707_inspect_v1/v2 (inventory only).

**Outputs.** Box ref707/: s1_*, s2_*, spread_v1.txt, MD5SUMS_ref707_s1_v1.txt. Copies in the worktree under results/mark3_layer_r1/ref707/: s2_r11.json 2d9d2c60, s2_c27.json a316c605.

**Run.** Read-only, no evolution. Box CPU, 2 x 4 threads, nice 10; about 6 min per member.

## 1. Answer

On every one of the 18 stored interior slices (r11, sigma/a 4: T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35, 40; c27, sigma/a 8: 12, 13, 14, 16, 20, 25, 30, 40), the held-projector history's sources and fields agree with the paper's Eq. dm-reference law **far inside the step-control spread** (tau 4 vs tau 8 at the same T):

| quantity | dm minus held, all 18 slices | spread (tau4 - tau8), all slices | worst per-slice ratio (dm - held)/spread |
|---|---|---|---|
| energy source, max over r | 0.0022 to 0.062 | 1.52 to 14.9 | 0.51 % (r11 T 12), so >= 196x |
| momentum source, max over r | 0.0069 to 0.134 | 0.85 to 9.46 | 3.2 % (r11 T 24), so >= 31x |
| N after re-solving both Gauss rows | 1.5e-5 to 8.0e-5 | 0.028 to 0.177 | 0.26 % (r11 T 24), so >= 380x |
| v after re-solving both Gauss rows | 3.3e-6 to 9.9e-5 | 0.016 to 0.295 | 0.55 % (r11 T 24), so >= 180x |
| charge E (about 106-118) | -0.0044 to +0.0002 | | <= 4e-5 relative |

**Verdict under the frozen rule: WITHIN SPREAD on all 18 slices. No slice differs.** The held-projector and dm-reference laws are indistinguishable at the step-control level on the stored interior (T/r_s 12-40, both widths). That includes the formation slice at T/r_s 12 (record 96, the first trapped-surface record is 97).

**Why.** The held projector's departure from P* (the prepared band) is almost exactly the second-gradient polarization W_2 that the paper subtracts. The two parts of dm - held are each 0.01-0.20 in the energy source, and they cancel to 0.002-0.06 (section 3).

## 2. Definitions (as frozen)

**Held sources** (engine_collapse_v19 sources(), l.1196-1204):
eps_held = sum_kappa w_kappa [dens(h0, sea) - dens(h0, R2) + dens(h0, dust, f)], with J_held likewise with Q.

**dm-reference sources:**
eps_dm = sum_kappa w_kappa [dens(h0, sea) - dens(h0, P*) + dens(h0, dust, f) - (dW2_N(u) - dW2_N(u*))], with J likewise with Q and dW2_v.
- P* = the negative band of the clocked chain at (N_cinit, v_cinit): 2048 levels per channel, as engine_collapse_v19 l.1002-1007 builds sea and ref0;
- u = the record's site fields;
- W_2 = vaclaw_w2_v3 (6f9ab685), the full-flow chain W_2 with k = kappa, m = 0.2.

**Inputs.** Engine <internal>/code: v19 9e4e6abf, v5 fc6faf26, fields_v14 dc864dfe, vacflow 5225e959, all md5-checked at start. Slices: per kappa sea, dust, f and R2 (complex128). Fields: C3prod19 r11 tau8 (and tau4 for the spread), C3lad27 c27 (a9d6472d).

**The Gauss rows under dm.** Re-solve with V19.solve (the record's own solver options: averaged site fields, collar flag, support 1e-14) from (eps_dm, J_dm), and compare with the stored fields. The dm solves' own row residuals are residual_W <= 4.8e-4 and residual_invN <= 3.0e-5, with no failure.

## 3. Per slice (key-summed; full repr in spread_v1.txt and the s2 JSONs)

dS_eps is split into its P* part [dens(R2) - dens(P*)] and its W2 part [-(dW2(u) - dW2(u*))]. Both parts were checked in v1 and v2. The P* part alone is 0.010-0.203; the W2 part cancels it.

| member T/r_s | dS_eps max (P* part) | dS_J max | dN max | dv max | dE | spread S_eps / S_J / S_N / S_v |
|---|---|---|---|---|---|---|
| r11 12 | 0.0077 (0.030) | 0.0135 | 4.4e-5 | 4.7e-5 | -8.1e-4 | 1.52 / 0.85 / 0.038 / 0.094 |
| r11 15 | 0.0120 (0.105) | 0.104 | 6.4e-5 | 8.4e-5 | +1.4e-4 | 6.16 / 4.44 / 0.099 / 0.120 |
| r11 16 | 0.0106 (0.054) | 0.107 | 6.4e-5 | 7.2e-5 | +2.3e-4 | 8.60 / 6.00 / 0.085 / 0.110 |
| r11 20 | 0.0372 (0.133) | 0.116 | 7.0e-5 | 6.6e-5 | -3.5e-3 | 9.26 / 6.27 / 0.153 / 0.294 |
| r11 24 | 0.0132 (0.083) | 0.124 | 7.3e-5 | 8.8e-5 | +1.2e-4 | 5.08 / 3.86 / 0.028 / 0.016 |
| r11 25 | 0.0157 (0.068) | 0.124 | 7.4e-5 | 7.1e-5 | -1.4e-3 | 6.34 / 4.78 / 0.101 / 0.141 |
| r11 28 | 0.0615 (0.203) | 0.127 | 7.6e-5 | 7.9e-5 | -4.4e-3 | 12.9 / 8.66 / 0.091 / 0.149 |
| r11 30 | 0.0295 (0.126) | 0.128 | 7.6e-5 | 9.9e-5 | -2.0e-4 | 14.9 / 9.46 / 0.137 / 0.206 |
| r11 35 | 0.0594 (0.195) | 0.131 | 7.8e-5 | 9.0e-5 | -3.5e-3 | 10.9 / 6.87 / 0.069 / 0.072 |
| r11 40 | 0.0506 (0.162) | 0.134 | 8.0e-5 | 9.2e-5 | -3.4e-3 | 13.4 / 8.74 / 0.177 / 0.295 |
| c27 12 | 0.0022 (0.010) | 0.0069 | 1.5e-5 | 3.3e-6 | -2.7e-4 | r11's: 1.52 / 0.85 / 0.038 / 0.094 |
| c27 13 | 0.0077 (0.034) | 0.0191 | 1.6e-5 | 8.8e-6 | -1.0e-3 | r11's: 6.11 / 2.80 / 0.043 / 0.038 |
| c27 14 | 0.0159 (0.066) | 0.0326 | 1.8e-5 | 1.4e-5 | -1.8e-3 | r11's: 5.65 / 4.38 / 0.103 / 0.128 |
| c27 16 | 0.0073 (0.031) | 0.0330 | 1.8e-5 | 1.8e-5 | -1.3e-4 | r11's: 8.60 / 6.00 / 0.085 / 0.110 |
| c27 20 | 0.0041 (0.0085) | 0.0403 | 2.2e-5 | 1.7e-5 | +2.1e-5 | r11's: 9.26 / 6.27 / 0.153 / 0.294 |
| c27 25 | 0.0043 (0.0115) | 0.0422 | 2.3e-5 | 1.6e-5 | -3.2e-4 | r11's: 6.34 / 4.78 / 0.101 / 0.141 |
| c27 30 | 0.0086 (0.029) | 0.0482 | 2.6e-5 | 2.2e-5 | -2.0e-4 | r11's: 14.9 / 9.46 / 0.137 / 0.206 |
| c27 40 | 0.0109 (0.034) | 0.0538 | 2.9e-5 | 2.4e-5 | -7.4e-4 | r11's: 13.4 / 8.74 / 0.177 / 0.295 |

The c27 member has no tau pair, so the frozen rule uses r11's spread at the same T/r_s, as labelled. The c27 differences are 2-10x smaller than r11's in every column.

## 4. Gates

- **V0 PASS** (after v1a): the engine chain (Chan.hs / Qs via chain_sparse) equals m3l heavy_band(k = kappa) to <= 4.4e-16 for every kappa, with R = n + 1. The W2 module is the right chain.
- **V1** (held sources rebuilt from the slice = stored eps_tot / J_tot, frozen 1e-10):
  - PASS on r11 T 12, 16, 20, 24 and 28 (<= 2.8e-13);
  - FAIL, labelled UNVALIDATED, on r11 T 15, 25, 30, 35 and 40 (5e-6 to 1.3e-5 relative) and on all 8 c27 slices (1e-6 to 5.4e-5).
  - Information only: these mismatches sit at the engine's own fixed-point tolerances (stop_fields 1e-6, stop_E 1e-7 in the records' solver JSON). They are 10^3-10^4 below dm - held, and 10^5 below the spread. On the 5 fully validated slices the verdict is the same as on the other 13.
- **V2** (the held re-solve reproduces the stored fields, frozen 1e-9): PASS on the same 5 r11 slices (<= 7.8e-11); 2e-8 to 9e-7 elsewhere, which tracks V1.

## 5. Deviation record

- v1 (7a92560c) implemented V0 "exactly equal" as == 0. The chains differ at 4.4e-16 by floating-point assembly, so v1 excluded W2 from dm and its numbers are the P*-part only (kept: s1_*.json).
- v1a (dated, before any W2-inclusive number) set the V0 tolerance to 1e-14. v2 differs from v1 in that one comparison (diff in the addendum). Nothing else was changed.

## 6. What this does and does not settle

- **Settled:** at the 18 stored interior slices, replacing the held projector by the paper's dm-reference changes the sources by <= 3.2 % of the step-control spread (energy <= 0.51 %, momentum <= 3.2 %, slice by slice) and the solved fields by <= 0.55 % of it. On this screen no record here needs re-evolution because of the source law.
- **Not settled:**
  - (i) The formation passage's substep-refined energy budget E_2 - E_1 - int Tr(h_bar dC) across records 96 -> 97 (stage 2, separate addendum).
  - (ii) The early c17 leg (records 0-88), which has no slices and needs the exact backward rebuild from T/r_s 12 (about 4-8 CPU-h).
  - (iii) Histories other than these two members.

  The paper's todo (l.1272) asks for (i) explicitly.

## 7. Self red-team

1. **Is P* right?** It is built exactly as the engine built its own initial sea and ref0 (the clocked chain at the stored N_cinit, v_cinit). The held R2 started from it.
2. **Is W2 right?** V0 shows the same chain. vaclaw_w2_v3 is route 1's W_2 for this chain (reviewer B w2fulla). The u for W2 is the record's site fields; a sensitivity variant with N/v_applied is in the JSONs (dS_eps_W2_applied_variant).
3. **Is the cancellation suspicious?** It is the expected design: a level-following projector equals the band expansion through second order where the band stays gapped. The P* and W2 parts are each reported, so the reader can see both.
4. **Spread choice.** tau 4 vs tau 8 is the coarsest available step-control pair. A tighter spread (e.g. tau 8 vs tau 16) would be smaller, but the margins (>= 31x on sources, >= 180x on fields) leave room for a spread about 30x tighter.
5. **Rules.** Read-only on all stored data; box CPU only (cpus 0-7, nice 10, 8 threads); no GPU; outputs only in ref707/.

## 8. Stage 2: the formation passage 96 -> 97 under dm-reference sources, substep refinement (NEW in v1.1)

**Frozen.** ADDENDUM_707_ref_stage2_v1.md, md5 c2df51a2cca022cb85cb13d4e3b9a2b0, frozen before any stage-2 number.

**Script.** ref707_stage2_v2.py, md5 af91153164599e2df4325bcd5967eb1c. It is v1 (a3295dec) with ONLY the matrix-product implementation changed: the same Chebyshev series, applied as banded shifts with the channels threaded. v1 was stopped by exact pid before any substep number; its held-drift and u0 numbers are identical to v2's.

**Output.** Box ref707/stage2_v2/stage2.json, md5 5c2526b7143960d70313b7a539b58ff1 (worktree copy results/mark3_layer_r1/ref707/stage2_v2.json); log s2.out a9a14dbd. The directory ref707/stage2/ holds the stopped v1 (its s2.done was written by the driver after the stop; it holds no substep numbers).

**Run.** Box CPU, 8 threads, nice 10; 4.9 h (box 11:05-16:00Z).

**Scheme.** Member r11. Start from the stored slice at T/r_s 12 (record 96), u_0 = its dm-solved fields (E_0 = 116.9328467, 5 iterations to 2.7e-11). Over the record interval (tau 8):
- n = 1, 2, 4, 8 implicit-midpoint substeps;
- per substep: exp(-i tau_s h[u_app]) applied to every sea and dust orbital (Chebyshev, K = 72-123 terms);
- then the dm sources at the trial fields and the Gauss rows re-solved, with u_app = the midpoint, iterated to 1e-10.
- Every substep converged (6-12 iterations); orbital unitarity <= 5.4e-13; the dm solves' residual_W <= 4.8e-5.

**The budget E_2 - E_1 - int Tr(h_bar dC)** (E = the rows' charge; RESIDUAL uses the endpoint-trapezoid generator; the substep-generator form is printed beside it):

| n (tau_s) | E_2 - E_1 | sum I (substep generator) | sum I^trap | RESIDUAL | max abs(N_n - N_8), max abs(v_n - v_8) at 12.125 |
|---|---|---|---|---|---|
| 1 (8) | -1.07159e-2 | 9.3e-10 | 9.3e-10 | -1.07159e-2 | 8.4e-4, 6.1e-4 |
| 2 (4) | -2.71955e-3 | -4.7e-10 | -7.0e-10 | -2.71955e-3 | 5.6e-4, 1.2e-4 |
| 4 (2) | -6.87657e-4 | 0 | -1.2e-10 | -6.87657e-4 | 2.3e-5, 2.7e-5 |
| 8 (1) | -1.70754e-4 | 0 | 1.2e-10 | -1.70754e-4 | 0, 0 |

- **Trace integrals.** Both sit at the floating-point floor of the about-1e6 traces (<= 1e-9). That is exact for a unitary midpoint propagator, so the residual equals the record's energy drift.
- **Convergence order 2.** Successive ratios are 3.940, 3.955 and 4.027 (observed order 1.98, 1.98, 2.01).
- **Richardson limit.** E_8 + (E_8 - E_4)/3 = +1.5e-6, i.e. 1.3e-8 of E. Under the paper's law the record conserves the charge to the step-control truncation.
- **Fields.** They converge with refinement (n = 4 vs 8: 2-3e-5).

**Frozen verdict: PASS.**
- Budget = |E_dm - E_held| at T 12 (8.1e-4) + max residual_W x |E| (5.6e-3) + |held drift 96 -> 97| (0.329) = 0.336.
- |RESIDUAL_8| = 1.7e-4 <= 0.336, and |R_4 - R_8| = 5.2e-4 <= |R_2 - R_4| = 2.0e-3.

**Jump test: no passage-associated jump.** The held history's per-record charge drift grows smoothly through the passage:
- 92 -> 100: -0.110, -0.141, -0.185, -0.244, **-0.329 (96 -> 97)**, -0.448, -0.611, -0.816;
- 96 -> 97 is 1.35x the median of the others, against the 3x jump threshold.

**Finding (labelled as an inference).**
- The stored held history loses 0.329 of the charge over the formation record, and its loss accelerates (-0.82 per record by 99 -> 100).
- The dm-law re-solve over the same record loses 0.0107 at the same step (n = 1, tau 8), 30x less, and converges to zero at second order.
- Stage 1 showed that the held and dm SOURCES differ by <= 0.5 % of the step-control spread. So the held history's energy loss is attributable to its stepping scheme: rows from the STEP-START covariance with a lagged, level-following reference (engine_collapse_v19 l.1214-1258), against my implicit midpoint. It is not attributable to the source law.
- The clean separating control is the held law under the same implicit-midpoint scheme (n = 1, about 40 min CPU). It was not run, being outside the frozen stage 2. Recommended if the attribution is to be printed.
- **Implication for #707's purpose:** the held histories' FIELDS sit inside the step-control spread of the paper's law (stage 1), but their charge bookkeeping across the passage carries a scheme drift of order 0.3 % of E per record at T/r_s ~ 12, growing. Any reading that depends on charge conservation at the 1e-3 level through the formation passage needs either that control or re-evolution under a midpoint scheme.

**The c17 early leg (records 0-88) is now feasible and cheap.** m3-adaptive's SINGCERT Stage 0 finished the exact backward rebuild to T = 0 (box <internal>/stage0_full, per-record checkpoints; record 0 = ckpt/state_latest.npz md5 d8bbc506; source checks <= 5.4e-10; R2 selection unambiguous). The stage-1 screen runs on those states READ-ONLY at about 15 s per record after a 3-min P* build, i.e. about 25 min CPU for all 89 records. It is not started unasked.

## 9. #735 separating control: the HELD law under the SAME implicit-midpoint scheme (NEW in v1.2)

**Frozen.** ADDENDUM_735_v1.md, md5 32606fec33d03695567cf1b4061f8b5d, frozen before any number.

**Script.** ref707_s735_v1.py, md5 518bd3dcf1df9e4f401a9ba753f95a47. It imports ref707_stage2_v2 (af911531) for the identical propagation, traces and solver set-up.

**Output.** Box ref707/s735/control.json, md5 fa58556e32f7ac7ab0d8b8b492d19008 (worktree copy results/mark3_layer_r1/ref707/control_s735.json); log control.out 4b2894ad.

**Run.** Box 16:15-17:30Z (date -u), 8 threads, cpus 0-7.

**Definition.** Stage 2 with n = 1 on r11 record 96 -> 97 (tau 8), changing ONLY the sources:
- eps_held = sum_kappa w [dens(h0, sea) + dens(h0, dust, f) - dens(h0, R_held(u))];
- R_held(u) = the 2048 eigenvectors of h_kappa[u] with the largest overlap with the stored R2_96 (the engine's level-following selection, as SINGCERT Stage 0 reconstructed it);
- fixed point to 1e-10.

**Gate H0** (at the stored record-96 fields):
- The selected subspace equals the stored R2_96 to 1 - s_min^2 <= 5.1e-15 for every kappa (selected weights 1 - 1e-14; largest rejected 2e-17).
- The held sources reproduce eps_tot[96] to 1.2e-8 relative, against the frozen 1e-10, so the gate is formally FAIL and the control is labelled UNVALIDATED on that one item.
- 1.2e-8 relative in the sources is about 10^6 below the decision margins (0.032 vs 0.110 in E) and cannot move the verdict.
- At the converged fields the selection stays unambiguous (min selected weight 0.999997, max rejected 3.0e-6).

**Result.**

| law, scheme | E(96) | E(97) | E_2 - E_1 (residual; trace integrals <= 5e-10) |
|---|---|---|---|
| HELD, implicit midpoint (this control; 12 iterations, 5.1e-11) | 116.9336611 | 116.9229385 | **-1.07226e-2** |
| dm-reference, implicit midpoint (stage 2, n = 1) | 116.9328467 | 116.9221308 | -1.07159e-2 |
| HELD, the stored history's own scheme | | | -0.32925 |

**Frozen verdict: SCHEME.** |D_held| = 0.0107 <= 3 |D_dm| = 0.032.
- Under the same stepping scheme the two source laws give the same record energy change to 6.6e-6 (0.06 % of it).
- The stored history's 30x larger loss (-0.329 on this record, growing to -0.82 by 99 -> 100) is therefore a property of its STEPPING SCHEME, not of the held source law. That scheme solves the rows from the step-start covariance with a lagged reference (engine_collapse_v19 l.1214-1258).
- With stage 2 (second-order convergence to +1.5e-6 under refinement), the paper's law conserves the charge across the formation record. The held law does too, under a midpoint scheme at tau 8, to 1.07e-2 per record, which is itself O(tau^2) truncation.

## 10. Paper statements that depend on charge conservation through the formation passage (#735 item 3; no computation)

**Paper:** <internal>/PAPER_flat_bdca0bd.tex, md5 c9e3439d (the version m3-adaptive cited). Line numbers are this file's.

The measured scale: the stored histories' held-step charge deficit is 0.11-0.82 per record over T/r_s 11.5-12.5, i.e. 0.1-0.7 % of E per record and growing. It is a scheme effect (section 9). The paper's law conserves the charge to <= 1e-8 of E per record in the refinement limit (section 8).

**DEPEND on charge conservation at or beyond the 1e-3 level:**

1. **l.1749-1756, the first-law diagnostic.** "On the sigma = 2a member's 224 consecutive record pairs ... with the stored history's sources, the terms ... sum to delta E = -18.36, (kappa_R/8 pi G) delta A = -6.51 and W_ext = -11.93 ... the work term ... excludes the no-work form."
   - delta E here is the stored (held-scheme) charge change over T/r_s ~ 12-40 (about 15 % of E). Section 9 shows the held scheme's per-record charge change on the formation record is about 30x the law's.
   - So delta E, and with it the inferred W_ext and the "excludes the no-work form" conclusion, carry a scheme deficit of the same order as the terms. This depends on charge conservation at far above 1e-3.
   - (Member sigma = 2a was not itself re-solved here; same engine and scheme.)
2. **l.1397-1401, collar charge.** "Collar charge at 16 is 0.847, 0.919 and 0.975 of E_0. This held-step deficit, not a charge-loss law, halves with step length and is 0.0005 on the conserving sigma/a = 4 step (certified to 12.5, read beyond)."
   - The attribution "held-step deficit, not a charge-loss law" is now CONFIRMED on the formation record by a law-vs-law control at fixed scheme (section 9). The 0.847-0.975 values are scheme-dependent readings.
   - The "0.0005 on the conserving step" figure is a 5e-4 statement that relies on that conserving step's own conservation through formation (not re-tested here).
3. **l.1375-1378, the formation clock.** "The first trapped root forms inside the matter ... The first records across the steps and on a charge-conserving step (11.875) put the zero-step formation near 11.8 to 11.9 r_s."
   - This uses a charge-conserving step to place formation, so it depends on conservation through the passage.
   - By section 8 the paper's law gives second-order-convergent conservation there, which supports using a conserving step. The 11.8-11.9 value itself was not recomputed.
4. **l.1381-1382, formation radius in units of the record's 2GE** ("... at 0.65, 0.66 and 0.76 of the record's 2GE"). E is the stored record's charge, so the fractions inherit the held-step deficit accumulated by formation (about 0.1-0.3 % per record near T 12). At the two printed digits this is marginal (<= 1e-2) rather than decisive.
5. **l.2228, the post-crossing root.** "After crossing, a vacuum exterior with E conserved (Appendix charge) keeps the root at r_s = 2GE." This relies on the analytic conservation (l.2856-2859, "the Hartree flow conserves the boundary charge"). Section 8 is a numerical confirmation of that conservation across the formation record under the paper's law. The stored held histories do NOT satisfy it at their own step (section 9).

**NOT dependent on it** (checked in context):
- l.2042, size-32 two-field "charge change is a conservation residual": a different history, already labelled a residual.
- The source-law readings: stage 1 shows held and dm fields agree within 0.55 % of the step-control spread.

**The todo at l.1272** ("re-solve the narrow passages with the new sources under substep refinement until E_2 - E_1 - int Tr(h_bar dC) lies within the source and solve budget with no passage-associated jump") is MET for the r11 formation record (section 8). Section 9 adds that the stored held histories' charge drift is a stepping-scheme effect. Readings 1-4 above would need re-reading on a conserving scheme to remove it.

## 11. The c17 early leg, records 0-88 (T/r_s 0-11), stage-1 screen (NEW in v1.3)

**Frozen.** ADDENDUM_735_v1.md (32606fec), item (2).

**Script.** ref707_s735_v1.py (518bd3dc), mode c17.

**Output.** Box ref707/s735/c17.json, md5 0d30e846b130ca26f165c9d0ec016652 (worktree copy results/mark3_layer_r1/ref707/c17_s735.json); log c17.out 7396f7f6.

**Run.** Box 17:31Z 31 Oct to about 23:40Z (date -u), 8 threads, cpus 0-7, about 4.2 min per record. No restart was needed.

**Method.**
- The states are regenerated by FORWARD exact unitary replay from SINGCERT Stage 0's record-0 state (box <internal>/state_latest.npz, md5 d8bbc5066f4362b17c13023a84dc78a5, verified), with the sc_stage0_v2 convention state(T_{n+1}) = exp(-i tau h_m(u_n)) state(T_n), u_n = the stored applied fields, tau 8.
- At every record 0-88: dm sources vs the stored (held) eps_tot / J_tot; the dm re-solve of both Gauss rows; and the tau4-vs-tau8 r11 spread at the same T.

**Gates.**
- **RE (end-to-end) PASS:** the state replayed through all 96 records matches the stored T/r_s 12 slice to 1 - s_min^2 <= 2.9e-12 for every kappa (frozen <= 1e-8). The replay is the production history.
- **R0:** at record 0 (R2 = P* by construction, u = u*), dens(sea) - dens(P*) + dust reproduces eps_tot[0] to 4.7e-10 relative, against the frozen 1e-10, so it is formally FAIL. That equals Stage 0's own record-0 source check (4.7e-10), i.e. the reproduction floor of the rebuilt state.

**Result: WITHIN SPREAD on all 88 records 1-88. Record 0 is DIFFERS by the frozen label only.**
- At T = 0 the tau4 and tau8 histories share the identical initial data, so the "spread" there is 1e-13 (round-off).
- The record-0 difference (6.8e-10 in eps) is the state's own reproduction error, not a law difference: at record 0 P* = R2 and u = u*, so dm and held coincide exactly by construction.

**Worst per-record ratios (dm - held)/spread, records 1-88:**

| quantity | worst ratio | where |
|---|---|---|
| energy source | 6.0 % | |
| momentum source | 30.0 % | record 76, T/r_s 9.5: 1.94e-3 vs 6.46e-3. Records 74-78 are 28-30 %. |
| N after the dm re-solve | 6.7 % | |
| v after the dm re-solve | 13.7 % | record 18, T/r_s 2.25 |

- The dm rows' charge differs from the held by at most 1.4e-4 (of about 118). The dm solves' residual_W is <= 2.0e-5, with no failures.
- **Margins.** The early-leg spread is 10^2-10^4 smaller than the interior one (S_eps 1e-4 to 0.18 vs 1.5-15), so this is the stringent part of the screen. Its margins (>= 3.3x on every quantity) are much tighter than the interior's (>= 31x), but every record passes.

**Selected records** (dm - held max | spread; held E = the rows' charge from the stored sources):

| record (T/r_s) | dS_eps / S_eps | dS_J / S_J | dN / S_N | dv / S_v | dE | E_held |
|---|---|---|---|---|---|---|
| 8 (1) | 4.5e-6 / 9.6e-5 | 9.2e-6 / 2.5e-4 | 3.0e-7 / 8.1e-5 | 8.3e-7 / 1.2e-5 | 8.3e-8 | 118.1387 |
| 24 (3) | 7.5e-6 / 7.3e-4 | 6.5e-5 / 8.6e-4 | 1.5e-6 / 1.9e-4 | 1.3e-6 / 5.0e-5 | 1.2e-6 | 118.1438 |
| 40 (5) | 4.9e-5 / 2.9e-3 | 1.8e-4 / 1.4e-3 | 9.9e-6 / 4.6e-4 | 1.0e-5 / 3.6e-4 | 3.9e-6 | 118.1490 |
| 56 (7) | 6.5e-5 / 7.2e-3 | 4.4e-4 / 2.4e-3 | 7.8e-6 / 3.2e-4 | 1.3e-5 / 1.3e-3 | 4.9e-6 | 118.1483 |
| 72 (9) | 1.9e-5 / 1.7e-2 | 1.4e-3 / 5.1e-3 | 4.2e-5 / 1.0e-3 | 4.2e-6 / 4.9e-3 | -7.7e-6 | 118.1204 |
| 80 (10) | 3.3e-4 / 4.4e-2 | 2.6e-3 / 1.2e-2 | 2.4e-5 / 3.3e-3 | 2.0e-5 / 1.1e-2 | -3.9e-5 | 118.0612 |
| 88 (11) | 1.3e-3 / 1.8e-1 | 5.0e-3 / 7.5e-2 | 2.5e-5 / 1.1e-2 | 3.6e-5 / 2.8e-2 | -1.4e-4 | 117.8741 |

**By-product (held charge history, from the stored sources).**
- E_held is flat at 118.138-118.150 over T/r_s 0-8, then falls: 118.120 (9), 118.061 (10), 117.874 (11), and 116.934 at 12 (stage 2).
- That is the onset of the held-step deficit identified in section 9. It is negligible (< 1e-4 relative) until T/r_s ~ 8 and reaches 0.2 % per record by formation.

**Verdict for #707's purpose** (with sections 1-10):
- The held-projector histories' SOURCES and FIELDS agree with the paper's dm-reference law within the step-control spread on every examined record: records 1-88 here, the 18 interior slices T/r_s 12-40 (stage 1), and the formation record 96 -> 97 (stage 2).
- Re-evolution for the SOURCE LAW is not indicated anywhere examined.
- What does need attention is the held histories' STEPPING SCHEME's charge deficit from T/r_s ~ 9 on (section 9) and the paper statements listed in section 10.
