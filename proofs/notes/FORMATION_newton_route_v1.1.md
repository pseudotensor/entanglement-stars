# FORMATION_newton_route_v1.1: the formation clock's second route (P64_s4 by Newton from T = 0)

Solve lane, 21 Oct 2026 (lead inbox #262 to #264; reviewer A's ruling deliveries/lead_gaps/FableAPI_h100job_v1.md). For the write lane's "on two routes" in Sec. IV B and the conclusions, after a cold check by route2-busi.

## 1. Result

P64_s4 run again from T = 0 by Newton (engine_collapse_v36nq 8549c542 with engine_fields_v16b d7250a37, exact J.v, --pair-workers 2, H2 on) on the production member's own preparation and stepping holds its first trapping root at **T/r_s 12.125**, the production's first-root record, with **R_t 38.556 a (0.6024 r_s, 0.6613 of 2GE = 58.302 a)**, r_in 23.239 a, min(N - |v|) -0.0279, E/E0 0.9870198: equal to the production's to the printed digits. By the paper's criteria (0.5 r_s in the clock, 10 percent in R_t at matched clocks) the two routes **AGREE**: the clock differs by 0 r_s, R_t by less than 4e-7 relative at every matched clock 12.125 to 12.5 (2.5e-7, 2.4e-7, 3.0e-7, 3.9e-7). Record by record over 0 to 12.5 (101 records) the largest differences are 2.3e-7 in E/E0 and 3.4e-7 in min(N - |v|).

**Certification.** busi's audited confirm (rc_audit_v1.py 708f2743): all 101 records certified, kept residual at most 7.5e-8, confirming residual at most 1.5e-8 (at 6.125; 2.1e-9 to 2.6e-9 over 11.5 to 12.5), no record failing at the engine floor 1e-6, at 1e-4 or at 1e-3; the formation record (kept 7.39e-8, confirming 2.63e-9) and the 3 certifying records 12.25, 12.375, 12.5 are certified under the production rule.

**What the second route establishes, and what not.** It is a different solver (Newton-Krylov with the exact Jacobian product, a different engine version, field build, Python stack and card) on the same discretization, preparation and record stepping as the production plain-iteration member, and it lands on the same fixed point: the printed readouts agree to 4e-7 (E/E0 2.3e-7, min(N - |v|) 3.4e-7, R_t 4e-7 relative) and the stored fields to 1.3e-6 in N and 1.0e-6 in v on the nodes (6.4e-7 and 8.6e-7 on the sites), the level of the production's own kept residuals (8e-7 at 0.75 and 6e-7 to 9.4e-7 over 10.875 to 12.5, where the field differences peak) against the Newton run's 2e-11 and 4e-8 to 7.5e-8 there. So the formation clock and radius are solver-independent: no plain-iteration artifact. It does not test discretization dependence (that is the ladders' job).

## 2. Readout

| run | first root T/r_s | R_t/a | R_t/r_s | R_t/2GE | r_in/a | min(N-|v|) | E/E0 | kept | confirming | zero crossing of min(N-|v|) |
|---|---|---|---|---|---|---|---|---|---|---|
| Newton v36nq (this run) | 12.125 | 38.5562 | 0.6024 | 0.6613 (2GE 58.302 a) | 23.239 | -0.02793 | 0.9870198 | 7.39e-08 | 2.63e-09 | 12.0553 |
| production P64_s4 v36d plain | 12.125 | 38.5562 | 0.6024 | 0.6613 | 23.239 | -0.02793 | 0.9870198 | (plain) | (plain) | 12.0553 |

R_t at matched clocks (Newton / production): 12.125 38.5562 / 38.5562; 12.25 42.2227 / 42.2228; 12.375 44.2085 / 44.2085; 12.5 45.4669 / 45.4669 (all rel. diff < 1e-5).
Paper criteria: clock |dT| = 0 r_s (<= 0.5): AGREE; R_t within 10 percent at matched clocks: AGREE (differences below 4e-7 relative).
Record by record 0 to 12.5 (101 records): max |dE/E0| 2.25e-7, max |d min(N-|v|)| 3.37e-7, at 1e-12 to 1e-11 with changing sign through 8.0, through 1e-10 at 9 to 10 and 1e-9 at 11, first above 1e-8 at 11.5 and 11.625 and above 1e-7 at 12.25, to their maxima at 12.5 (11.5: -1.3e-8 / +6.8e-9; 12.125: -7.3e-8 / +7.4e-8; 12.5: -2.3e-7 / +3.4e-7).
busi's audited confirm (rc_audit_v1.py 708f2743, readers engine_ladder_compare_v5): 101 records, kept below 1e-4 on all 101 (max 7.5e-8), r_c missing 0, r_c sources disagree 0, engine stop rule met on 101; certified through 12.5 as printed (30.3), with r_c at the engine floor 1e-6, at 1e-4 and at 1e-3: no record fails. The formation record and the 3 certifying records (12.25, 12.375, 12.5) are certified under the production rule (kept and confirming <= 1e-3).

The kept residual is reproduced from the stored fields: on every one of the 101 records max(|N_applied - N|, |v_applied - v|) over the 2048 sites equals the stored kept residual to the last digit (and on the production's 101 records its last-iterate residual). From the stored node fields (513 nodes at 4 a) a cubic through four nodes recomputes R_t at 12.125 as 38.564 a against the record's dense 38.556 (0.008 a, the grid's interpolation), and the production's node fields give the same to 1e-5 a; the first negative minimum is at 12.125 on the sites, the nodes and the records of both runs. (reviewer A's reading.)

Reading for the lead (not a ruling): the second route is a different SOLVER (Newton-Krylov with exact J.v, v36nq, H2 field build, pair workers, np1263 stack, H100) on the SAME discretization, preparation and stepping as the production plain-mix v36d member; it lands on the same fixed point: the printed readouts agree to 4e-7 (E/E0 2.3e-7, min(N - |v|) 3.4e-7, R_t 4e-7 relative) and the stored fields to 1.3e-6 in N and 1.0e-6 in v on the nodes (6.4e-7 and 8.6e-7 on the sites), the level of the production's own kept residuals (8e-7 at 0.75 and 6e-7 to 9.4e-7 over 10.875 to 12.5, where the field differences peak) against the Newton run's 2e-11 and 4e-8 to 7.5e-8 there, so it certifies that the formation clock and R_t are solver-independent (no plain-iteration artifact), not that they are discretization-independent.

## 3. Arguments against P64_s4 production (json args of collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json; parsed args of this launch)
Every key not listed is equal (m 0.2, g 0.25, rs 64, R0_over_rs 4, reading A, nsites 2048, ell 4, prep_ell 4, tau 8, kappa_sea 10, reference R2, site_fields averaged, stop_E 1e-7, stop_fields 1e-6, maxit 20, field_rtol 1e-12, field_atol 1e-16, field_kernel_cache, v36_fixes, collar_policy flag, support_rel 1e-14, gpu 0, c_LZ 1, window_max 4, s11 8,16 / 1, nk_inner 15, mix_alpha 0.5, delta_rtol_every 10, commutator_every 1, streams 0, m1_simpson_steps auto, half_offset 0, ball_* and weyl_* defaults).

| key | P64_s4 production | this run | reason |
|---|---|---|---|
| mix | plain | newton | the job: the Newton route (ISO_128's block) |
| nk_maxiter | 8 | 30 | ISO_128's Newton block |
| nk_krylov | None | 40 | ISO_128's Newton block |
| nk_jvp | None | exact | ISO_128's Newton block |
| nk_line_search | None | halving | ISO_128's Newton block |
| nk_halvings | None | 6 | ISO_128's Newton block |
| nk_restarts | None | 2 | ISO_128's Newton block |
| nk_plain_fallback | (absent in v36d) | on | ISO_128's Newton block (v36q option) |
| nk_host_release | (absent) | on | ISO_128's Newton block (v36m option) |
| nk_set | None | resolve | ISO_128's Newton block |
| iter_cache | (absent) | drop | ISO_128's Newton block (v36n memory option) |
| prep_memory | (absent) | lean | ISO_128's Newton block (v36n memory option) |
| end_above | None | 0.001 | ISO_128's Newton block: a record with kept residual > 1e-3 ends the member |
| pair_workers | 0 | 2 | SYM-N1 launch rule (bitwise option, busi #370) |
| dust_set / dust_kappa_max / dust_multiplicity | (absent in v36d) | well / None / 1 | v36n defaults = P64's well filling (NOT ISO_128's isotropic) |
| T_over_rs | 40 | 13 | stop near formation; the stopper ends it earlier |
| resume_run / resume_slice | the from0 run / its T0018.00 slice | None | FRESH from T = 0 (the production json is the T18 resume of the from-0 member) |
| slices_c128 | 19..40 | 1..11 every 1, 11.5..13 every 0.125 | the job's slice plan |
| slice_dir / out / tag | production paths | /workspace/solve_p64newton/{states,run}, lad_P64_s4_newton_v36nq_h100_1 | own dir |
Environment (not args): OMP/MKL/OPENBLAS/NUMEXPR 14 (H1, vCPU - 2), H2_PW_ONE=1 H2_LOCAL_SPLINE=1 (H2), NUMPY_MADVISE_HUGEPAGE=0, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; no nice. Engine: production P64 ran v36d; this runs v36nq (ISO_128's Newton engine).

## 4. Provenance

- Card: qi2_h100_1 (AWS p5.4xlarge, H100 80 GB). START 2026-10-21T04:54:09Z, pid 6428; the stopper (pid 6431) ended the engine at 05:24:59 after the run json and fields held 12.5 (json 8f31394b, fields e50744ed) (the stopper's stop lines are in the card's STOPPER.txt, quoted in the operator notes; the box copy ends at the formation line of 05:22:39; on the box the final json and fields carry the card's write times 05:23:54 and 05:23:56 UTC, and the log's last projection line puts the end about 05:25 UTC); 0.51 engine card-hours (about 18 s a record).
- The card was stopped at 05:34 by the AWS stop-watcher before the box pull finished (its trigger text was in this lane's STATUS; reworded); the volume was mounted read-only on 10.66.121.10 and the run, logs and every slice (T 1 to 11, 11.5 to 12.75; 27 files) are on the box under <internal>/final with the card's md5 on every file (MD5_card.txt). The tar pull (pull_final_v1.sh, log pull_final_v1.log) ended at 06:11:04 UTC with 'md5 DIFFER or empty' (11 of 27 files, the T0012.50 slice among them partial); the file-by-file pull pull_missing_v2.sh (log pull_missing_v2.log) reused 8 slices from the live mirror by md5, pulled the other 8 and ended at 06:34:09 UTC with '27 of 27 md5-equal, 0 failed'. The readout used the card json 8f31394b (box readout/newton_run.json).
- Readout files: box <internal>/readout (readout_v1.md 05170781, rc_audit_v1.md 02a19a2c, rc_audit_v1.json 76ec264c). Scripts: solve/code/ops/p64newton/ (launch, stopper, mirror, readout_v1.py, pull_final_v1.sh, pull_missing_v2.sh). Full operator notes: solve/results/NOTES_P64_s4_newton_h100_1_v1.md.

## 5. Changes from v1

reviewer A's cold reading (deliveries/lead_gaps/FableAPI_reading_FORMATION_newton_v1.md) CONFIRMS every printed number; its section 10 fixes 1 to 8 are applied word for word: (1) the pull record (the tar pull ended 'md5 DIFFER or empty'; the file-by-file pull ended 27 of 27 md5-equal); (2) pull_missing_v2.sh added to the scripts; (3) the stop times; (4) the record-by-record growth described as measured; (5) 'the same fixed point to 1e-7' replaced by the readouts' 4e-7 and the fields' 1.3e-6, at the production's own residual level; (6) R_t agreement below 4e-7 relative, not 'about 1e-6'; (7) the confirming residuals' maximum 1.5e-8; (8) the kept residual and R_t reproduced from the stored fields. No printed number changes.
