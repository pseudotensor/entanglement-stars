# PFIX values v1: numerical-run provenance for the twelve [VALUE NEEDED] markers

This is a read-only extraction for the correction set of the lattice-fermion paper (ms_fermion_new.tex with fermion_new/*.tex at commit 24434791d9f; flattened copy deliveries/lead_gaps/PAPER_flat_24434791.tex). The marker texts are those of AstraAPI_pfix1_v1.md (proofs/provenance/current_paper.md), quoted verbatim. Markers are numbered in the line order of value_needed.txt; pfix1 sec. 5 uses a different numbering, so its number is given in brackets below. FableAPI_pfix2_v1.md Patch 3 extends marker 6 to the ladder members, and Patch 5 extends marker 4 to the Hawking comparator rates; both extensions are covered.

Values were read from code, launch scripts, run records and logs. Nothing was run or recomputed, apart from hashing files (md5 or sha256), and no physics judgment is made. Where code and paper disagree, both are reported and the disagreement is not resolved. "NOT FOUND" lists the places searched.

Read access note: no file anywhere was written or changed, and no job was touched. A few box reads (the ladder-readout json and some run-record heads) used short `python3 -c` one-liners to print json fields. That is read-only, but heavier than the cat/head/grep allowed.

## Where things live (path map used throughout)

- Paper: <internal>/.wt-fermion-minimal (branch PR3, HEAD fc938764ee4; the paper source is unchanged since 24434791d9f).
- Companion archive: https://github.com/pseudotensor/entanglement-stars, main = f690b45eceee4a2dcddcc4f1bb456300881e87c5. The fermion material is fermion_new_codes/, data/, proofs/ and SKIPPED_LARGE.tsv. The local clone <internal>/entanglement-stars has a different branch checked out (sync-corrected-observable), so the archive was read with `git show origin/main:<path>`.
- Collapse engine and Page code, "solve_a": <internal>/solve (code/engine, code/page*, results/, STATUS.md, RUN_RECORD.md).
- Readouts, figures, light runs and redshift, "busi": <internal>/busi.
- Second-route arithmetic, "solve_b": <internal>/solve.
- hreduce: <internal>/.wt-route2-hreduce.
- Deliveries: <internal>/deliveries.
- Run records: the compute box 192.168.1.93, under <internal>/claude_alt_solve_bulk, <internal>/busi_bulk and <internal>/engine.
- The reference collapse is "P64_s4": run tag c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0(_to40), run on engine_collapse_v36d.py (md5 0315e082...). Fig. 1(a,b) is drawn from the separate c19 production record collapse_c19_rs64_A_R2_tau8_ell4_avg_r11 (engine v19 from 11.125; records 0 to 11 from v17). The light-field figures run on the B2 history, which is c19 continued on 4096 sites.
- Companion file: PFIX_VALUES_v1_sha256.tsv (same folder) gives the SHA256 of every file at archive commit f690b45 (marker 12).

## Status summary

| # | marker (opening words) | pfix1 sec. 5 no. | status |
|---|---|---|---|
| 1 | actual numerical precision, exponential/eigensolver ... | 7 | PARTIAL |
| 2 | actual radial chain lengths ... | 4 | PARTIAL |
| 3 | actual reference path, overlap/assignment rule ... | 6 | PARTIAL (FOUND for the reference collapse, c19, the sigma/a 2 and 8 runs and the c_LZ controls) |
| 4 | assertion-to-artifact map ... | 11 | PARTIAL |
| 5 | complete event selection and receiver-frequency normalization ... | 10 | FOUND |
| 6 | complete initial occupied-column construction ... | 3 | PARTIAL |
| 7 | entropy and flux payload selectors ... | 9 | PARTIAL |
| 8 | exact spatial and temporal interpolation ... | 8 | PARTIAL |
| 9 | final rendered page count ... | 12 | SKIPPED (the lead fills it) |
| 10 | immutable current-paper commit and companion-archive commit ... | 1 | SKIPPED (the lead fills it; the HEADs seen are recorded) |
| 11 | literal equations or pseudocode for S_j, R_j and T_j ... | 5 | FOUND (pod-side per-file code md5 not recorded) |
| 12 | SHA256 of distributed source, data, figure and manifest bytes ... | 2 | PARTIAL |

Cross-marker code/paper disagreements are reported, not resolved. Where to find each:
- (a) Sources at the step's start, not "midpoint sources": markers 11 and 3.
- (b) Certification: the engine stops at 1e-7 (dE/E) and 1e-6 (fields); the 1e-3 level is the readout's certification rule: markers 1 and 11.
- (c) The per-step continuation choice is logged per record and channel, against the paper's "the per-step choice is not read": marker 3.
- (d) The run json "script" field says engine_collapse_v34.py and the "engine" field engine_collapse_v25, while the file executed is v36d: markers 6 and 11.
- (e) "Released at rest" against a recorded dust_momentum_total of 55.78: marker 6.
- (f) Feedback "on two routes" against the busi report "no second numerical route": marker 4.
- (g) The headline redshifts are normalized at a static receiver at 4r_s: marker 5.
- Further items are inside each section.


## Marker 1

> [VALUE NEEDED: actual numerical precision, exponential/eigensolver algorithms and tolerances, software versions, randomization settings, and measured purity/charge residual definitions; obtain from each run's execution metadata and the executed propagation and eigensolver functions.]

Path abbreviations: SOLVE = <internal>/solve; BUSI = <internal>/busi; BOX = ssh 192.168.1.93; LIGHTCODE = <internal>/code_d13 (page_history_v6o.py md5 4c793ad7110ec0e4dcf3881b84d2115b, the md5 recorded as "v6o untouched, 4c793ad7..." at BUSI/STATUS.md:1160).

### 1a. Matter (heavy-field) engine: precision and algorithms

Executed engine files (md5 of the local copies in SOLVE/code/engine): engine_collapse_v19.py 9e4e6abf4f3a23d97ac8ba79dda85785 (c19 production from T/r_s 11.125, B2, the tau 4 control); engine_collapse_v36d.py 0315e0826ee872ee194474876ef523f9 (P64_s4 "rung 4 from T = 0", the W5 channel-count rungs); shared imports engine_collapse_v5.py fc6faf26a353f6d904a5242874e25586, engine_fields_v16.py 1ffcf5e88c7953d7bfee025d57f4fb86 (v36d line), engine_fields_v14.py dc864dfeafbf233f5669b961a51cb7fc (v19 line), engine_mixing_v1.py d142154a1e454fed344f1b0ddacf71e3, engine_ops_v2.py cdabf12b56be61d1d26879b57f073650, engine_ops_v3.py 460664cd52b8babd42f9a9c227499662, engine_ops_v34_cert.py 3db7b1dbb174a7a1b79c6b865d535ade, engine_vacflow_v1.py 5225e959b9e0bc730062041d60d800c8, engine_stress_v2.py c5759e7852fa5446ba22b6f72ab4d869, engine_collapse_v17.py 8fdca1df50817c9f885e64c17cbddc50 (imports: SOLVE/code/engine/engine_collapse_v36d.py:367-383, engine_collapse_v19.py:171-185). The W3 completions ran engine v36g: <internal>/engine_collapse_v36g.py md5 6224748b2651d4fd0a97125f1e89c79b (the only copy found; not in SOLVE/code/engine).

- Precision: complex128 for every state, operator and slice; fields float64. T_ (SOLVE/code/engine/engine_collapse_v5.py:92-93), channel CSR complex128 (engine_collapse_v5.py:105-107), dense clocked operator built on the device in complex128 from float64 N, v (engine_collapse_v36d.py:928-944, clocked_gpu). Slices stored "complex128" (run log line "slice resume: ... (complex128)", <internal>/B2r50_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.log; launch_B2_pod4_v1.sh header "complex128 slices").
- Eigensolver: dense Hermitian eigendecomposition torch.linalg.eigh of the clocked chain operator h_m = (1/2){N,h} + (1/2){v,Q}, one per channel per field evaluation (engine_collapse_v5.py:115-117 eigT; engine_collapse_v36d.py:388 eigT = C5.eigT, :921-925 eigT_t timing wrapper). Run on CUDA (args "gpu": 0); the backend library (cuSOLVER or MAGMA) and any eigh driver flag are NOT recorded in run metadata. No tolerance argument is passed to eigh (it is a direct dense solver).
- Exponential: not a matrix-exponential routine; the step is the spectral exponential from the same eigendecomposition, U diag(exp(-i e tau)) U^dagger applied to the sea and dust columns, with the fields held over tau: engine_collapse_v36d.py:2652-2660 (main loop), engine_collapse_v19.py:1565-1573, engine_collapse_v5.py:375-382; the half-step variant exp(-i e tau/2) for the M1 Simpson diagnostic only (engine_collapse_v36d.py:2416, engine_collapse_v19.py:1405).
- Field (Gauss-row) ODE solver: scipy.integrate.solve_ivp method DOP853, rtol 1e-12, atol 1e-16 (engine_fields_v16.py:173, 217, 229, 249, 383-384; defaults at :265; engine_fields_v14.py:168, 184, 294-295, 350; engine_collapse_v36d.py:398 SOLVE_TOL = {"rtol": 1e-12, "atol": 1e-16}, :3326 set from --field-rtol/--field-atol). Run args: "field_rtol": 1e-12, "field_atol": 1e-16, "support_rel": 1e-14 (<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json, "args").
- Self-consistency stop rule as executed: converged when |dE/E| < stop_E and max|N_out - N_in| < stop_fields and max|v_out - v_in| < stop_fields, at most maxit iterations, the smallest-residual iterate kept (engine_collapse_v36d.py:2240-2252; engine_collapse_v19.py:1257). Values in every reference-line run json: stop_E 1e-07, stop_fields 1e-06, maxit 20, mix "plain" (P64_s4, c19, B2, W3, W5); mix "newton" for P64_s2, RS32_s2, M3c_s4_nk, the 128 Newton runs, ISO_128, the r_s/a 256 members and Computation C (Newton args in <internal>/pod_launch_C_v11.sh:100: --nk-maxiter 30 --nk-krylov 40 --nk-restarts 0 --nk-line-search halving --nk-halvings 6 --source-continuation two-stage --end-above 1e-3). P64_s4 args also nk_maxiter 8, nk_inner 15, mix_alpha 0.5, delta_rtol_every 10, commutator_every 1, c_LZ 1.0, window_max 4, weyl_refine_max 10 (same json).
- Paper vs records (reported, not resolved): the paper certifies a record at a one-iteration change of at most 10^-3 in N and v (05_star.tex, "Method and controls"); the engines' own stop rule is 1e-6 in N, v and 1e-7 in |dE/E|. The 1e-3 level is the readout's certification rule, version 1.3 (definitions block of <internal>/ladders_readout_v1_3.json, "FableAPI_cert_rule_v1": kept or confirming residual above 1e-3 ends the certified prefix; records above 1e-4 "banded").
- Initial-data fill tolerances (reported for completeness): prepare_dust(passes=200, tol=1e-10), fill_lowest degeneracy tol 1e-9 (engine_collapse_v5.py:146, 164); head of P64_s4 json: "passes": 15, "last_change" 9.06e-11.

### 1b. Light-field propagation (principal light run v6o_B2, held runs, conformal control)

- Propagator: Chebyshev series of exp(-i tau h) on [-w, w], w the Gershgorin bound of the banded h, order K the smallest K >= |z| + 4 with Bessel tail 2 sum_{j>K} |J_j(z)| <= cheb_tol: LIGHTCODE/page_history_v6o.py:565-592 (cheb_apply, propagate), <internal>/page_ops_v6.py:341-352 (cheb_order; md5 31a85c000cecb4425f5af66828392ab6). No dense eigendecomposition in the propagation (page_history_v6o.py:235-236 docstring). cheb_tol 1e-16 in v6o_B2, the W3 light runs, B2_hold_v6s and the conformal run (run settings, <internal>/v6o_B2_k1_d13.json "cheb_tol": 1e-16); 1e-14 in the held-at-15/30 runs v6kA_c15/c30 (SOLVE/results/page/v6bA_runs/v6kA_c15/v6kA_c15_summary.json; default of v6o is 1e-14, page_history_v6o.py:758).
- Held-field propagation: "frozen_prop": "eig" (dense eigendecomposition of the held h, one per control) in v6o_B2, v6kA_c15, B2_hold_v6s settings; eigh_csr = torch.linalg.eigh on the densified Hermitian CSR, eigenvalues float64, vectors complex128 (page_ops_v6.py:407-412).
- Held vacuum (comparator projector): "sector (eigh of h_+ = W_+^dagger h W_+, n x n; the - sector's occupied vectors conj(V_+[:, e_+ > 0]) at -e_+)" (v6o_B2_k1_d13.json, clocks.00000.0000.per_channel.1.vacuum_route); trace-norm route "gram" = eigvalsh of 1 - M^dagger M (same json, trace_norm.route); vac_cluster_tol 1e-09, cut_sigma_tol 1e-10, cut_bound_tol 1e-06 (settings.args).
- Time sub-steps: interp "linear", interp_sub 8 (v6o_B2 settings; record_substeps at <internal>/page_w1_sector_v1d.py:698-716, fields at each sub-step's midpoint u = (i + 0.5)/n). Marker 8 owns the interpolation detail.
- Precision: complex128 columns ("torch.save {"Phi": cpu complex128 ...}", page_history_v6o.py:60; page_ops_v6.eigh_csr returns complex128).

### 1c. Software versions and hardware (lane records only; no run json or log records versions)

- P64_s4 (v36d, from T = 0 on pod 2 to T/r_s 18, continued on pod 7 from the T0018.00 slice md5 5e44bd04675d35c666bbe6e61606247b; <internal>/status.txt; SOLVE/STATUS.md:1598): venv_v36d = Python 3.11.10, torch 2.4.1+cu124, numpy 2.4.6, scipy 1.17.1 (SOLVE/STATUS.md:1300, "the cards' ladder_env/venv_v36d"; SOLVE/RUN_RECORD.md:364 "/workspace/venv_v36d (numpy 2.4.6, scipy 1.17.1: pod 2's versions; pod 4's system has 1.26.3 and 1.14.1)"). GPU: RunPod H100 (pod names; card type "H100 80GB HBM3" recorded for qi2_h100_1 at SOLVE/STATUS.md:778).
- B2 and the v19 pods: "numpy 1.26.3 and scipy 1.14.1 installed to match pod 1, torch 2.4.1+cu124 present; pod 1's engine code deployed, md5-equal to the box's for engine_collapse_v19" (SOLVE/RUN_RECORD.md:233). Python version for these pods: NOT FOUND.
- Light runs (v6o, v6s) and Computation C: venv_np1263 = numpy 1.26.3, scipy 1.14.1, torch 2.4.1+cu124 (SOLVE/STATUS.md:1300, 1570). v6o_B2 ran on pod 7 (kappa 1 to 6+) and RunPod R1 (<internal>/v6o_status.txt; BUSI/STATUS.md:1400).
- Reopen/256 kit: "py 3.11.10, numpy 1.26.3, scipy 1.14.1, torch 2.4.1+cu124, numba 0.61.2" (SOLVE/STATUS.md:710, 718).
- Box runs (readouts, some massless controls): torch 2.8.0+cu128, numpy 2.2.6, scipy 1.17.0 (SOLVE/RUN_RECORD.md:38, 64). The record notes torch 2.4.1 failing the massless-shell fill where torch 2.8.0 passes (RUN_RECORD.md:64); no principal massive run is recorded under torch 2.8.
- CUDA driver for the principal runs: NOT FOUND (only qi2_h100_1's driver 580.126.09, SOLVE/STATUS.md:778).

### 1d. Randomization

- Matter engines: no random-number use. grep for random/seed/randn/default_rng returns 0 hits in engine_collapse_v5, v19, v17, engine_fields_v14, v16, engine_mixing_v1, engine_ops_v2, v3, v34_cert, engine_vacflow_v1, engine_stress_v2; the 21 hits in engine_collapse_v36d.py are the word "seeds" of the deterministic chiral orbitals of the --massless-ball option (engine_collapse_v36d.py:1671-1706), which is off in the principal runs ("massless_ball": false in the P64_s4 json). Statement: the matter runs are deterministic up to floating-point reduction order on the GPU; no seed exists.
- Light readout: one seeded random routine, sigma_max_estimate (torch.Generator().manual_seed(0), complex Gaussian block), LIGHTCODE/page_d13_readout_v1d.py:464-469; it gives a lower estimate on a non-default trace-norm route; the principal records use the "gram" route ("route_short": "gram" in v6o_B2_k1_d13.json), so it is not exercised there. No other random use found in page_history_v6o.py, page_w1_sector_v1d.py, page_ops_v6.py, page_core_v6.py.

### 1e. Residual definitions as implemented

- Charge: E = W(r_collar)/(2g), W solved by DOP853 from the summed sources (engine_fields_v16.py:383-400; engine_fields_v14.py:294-303; charge() = engine_fields_v12/v13 "E = W(r_collar)/(2g)", engine_fields_v13.py:796-800, v16 imports it at engine_fields_v16.py:55). Self-consistency charge residual: dE = |E - E_prev|/|E| per iteration (engine_collapse_v36d.py:2242-2243). Printed per record: E/E0 and dE/dT (log lines, e.g. B2r50 log). Charge drift between records is read from the stored E/E0 series; no separate drift statistic is defined in the engine.
- Field residual ("res a b" in the log): a = kept residual max(max|N_out - N_app|, max|v_out - v_app|) of the kept iterate, b = the confirming plain step's residual (engine_collapse_v36d.py:2246-2248, 2852-2863).
- Reference/commutator diagnostic: ||[h_m, R2]||_F (engine_collapse_v36d.py:1339), printed "[h,R2] max" per record.
- Purity of the matter state: no per-record ||C^2 - C|| or ||Phi^dagger Phi - 1|| check found in the matter engines (searched engine_collapse_v36d.py, v19, v5 for purity/idempotency/orthonormality/Gram checks; only the preparation's gram_cond_max and discarded_norm_max, engine_collapse_v5.py:180-190, v36d:502-513). NOT FOUND as a measured residual for the heavy field.
- Light-field impurity ("propagation floor"): T_meas = ||Phi^dagger Phi - 1||_1, with the a priori Chebyshev allowance T_cheb (construction + truncation + rounding 16 K_max eps per sub-step per column) and premise T_meas <= T_cheb (<internal>/page_w1_sector_v1d.py:278-310); recorded per clock in "floor" (v6o_B2_k1_d13.json: T_meas 3.63e-11, T_cheb 2.33e-10 at T = 0); sector checks "pairing_defect_state", "purity_gap" per cut (same json).

Status: PARTIAL. Precision, algorithms, tolerances, randomness and the residual definitions are FOUND in code with lines; software versions are FOUND only in lane STATUS/RUN_RECORD lines, not in per-run metadata (run json "args"/"head" and logs carry none); Python version for the v19 pods, CUDA driver/library versions and the eigh backend are NOT FOUND; a heavy-field purity residual is NOT FOUND (looked in SOLVE/code/engine engine_collapse_v5/v19/v36d, run jsons under <internal>/{v36d_from0,B2_final}, run logs in their logs/ directories).

## Marker 2

> [VALUE NEEDED: actual radial chain lengths for the reference collapse, each step/width/channel-count control, both long interior completions, principal light propagation, held continuations, conformal control, cubic transfer and feedback control; read launch arguments and matrix/column shapes from the primary payloads referenced by the lineage and stored_identity.json records.]

Conventions as executed: chain sites r_n = n a, n = 1..nsites (Chan.R = arange(nsites) + 1, engine_collapse_v5.py:101), two components per site, so the per-channel matrix is (2 nsites) x (2 nsites); channels kappa = 1..kappa_sea for sea and dust together (SOLVE/specs/LADDERS_spec_v1.md:13). Chain end in r_s = nsites/(r_s/a). "ell" is the averaging width sigma/a (ell/sigma identity: SOLVE/results/page_v4.1/PAGE_solved_history_v4.1.md:480 "sigma = ell = 4 a"). md5 below is of the run json on the box unless stated.

### 2a. The reference collapse (three records carry the name in the lineage)

| record | engine | nsites (chain end) | kappa_sea | tau | ell / prep_ell | T/r_s span | json, md5 |
|---|---|---|---|---|---|---|---|
| c19 production (Fig. 1 panels a, b) | engine_collapse_v17 to T/r_s 11.0 (resume_from "C3prod/c17_rs64_A_R2_tau8_ell4_avg", resume_T_over_rs 11.0), engine_collapse_v19 from 11.125 | 2048 (32 r_s) | 10 | 8 | 4 / none | 0-40, 321 records | <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json, e27e457d37615cff8d5216f05a494f89 |
| P64_s4 "rung 4 on v36d from T = 0" (ladder/attribution reference) | engine_collapse_v36d (json "script" field reads engine_collapse_v34.py) | 2048 (32 r_s) | 10 | 8 | 4 / 4 | 0-40, 321 records; 0-18 on pod 2, 18-40 from slice T0018.00 on pod 7 | <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json, d7d69adcea7a43f8375ab4c7c5a80efc |
| B2 (background of the principal light run) | engine_collapse_v19 from T = 0, resumed at 50 after a crash | 4096 (64 r_s) | 10 | 8 | 4 / none | 0-100, 801 records | <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json, 417a4018cf6497f9b0020408783f44f5; launch line SOLVE/code/engine/launch_B2_pod4_v1.sh:13 (--nsites 4096 --kappa-sea 10 --tau 8 --T-over-rs 100); restart launch_B2r50_pod4_v1.sh |

Common args of all three: m 0.2, g 0.25, rs 64, R0_over_rs 4, reading A, reference R2, site_fields averaged, collar_policy flag. Paper vs records: 05_star.tex gives no chain length; 07_page_curve.tex states the completion comparison "changes ... the chain length", consistent with B2 4096 against the completions' 2048. The figure producer reads Fig. 1(a,b) from c19 and Fig. 1(c) from P64_s4/P64_s2/P64_s8 (BUSI/code/figures/figures_data_v1.py:25-32).

### 2b. Step controls (all m 0.2, rs 64, kappa_sea 10, ell 4; from ladders_readout_v1_3.json members and SOLVE/results/attrib/attrib_readout_v1.json "histories")

- P64_s4_tau4: engine_collapse_v19, nsites 2048, tau 4, to 40, 641 records; <internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg.json, f2e1d8a7483447426855e3211535a367.
- P64_s4_tau2: engine_collapse_v25, nsites 2048, tau 2, to 21.5, 689 records; <internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg.json, 9ba6f6ff302d8a903925f20583c96656.
- Conserving step W2_dg (paper: "certified to 12.5"): nsites 2048, tau 8, to 40, v36f then v36l rule; <internal>/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f.json, 1d4b69ef8ebed850f04f66a94b40b680.
- "three settings of the reference's crossing convention": not extracted (not in this marker's list; no record identified).

### 2c. Width (averaging-scale) controls

- P64_s2: nsites 2048, ell 2, prep_ell 4, tau 8, mix newton, composite history (v36d relaunch from 17.625 across seams); <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json, 0a005ac5f66bef02a0104e4278867765.
- P64_s8: nsites 2048, ell 8, prep_ell 4, tau 8, plain, v36d from T = 0; <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json, d1ca037a77c198aa83b33055495eceff.
- The light run's "sigma/a 8 history" is a different record: c27 prep4 ell8 (engine_collapse_v27; SOLVE/code/page/launch_page_history_v6_v6.sh:46-47 collapse_c27_rs64_A_R2_tau8_ell8_prep4_avg.json; md5 ef0ae520 per PAGE_solved_history_v4.1.md:5).

### 2d. Channel-count control (W5: kappa_sea 10, 14, 20 at tau 8 and 4, common preparation, to T/r_s 14; SOLVE/results/attrib/W5_W4_formation_attribution_v1.md section 3)

All nsites 2048, ell 4, prep_ell 4, plain, engine v36d (json "script" engine_collapse_v34.py), to 14 (113 records at tau 8, 225 at tau 4):
- W5_k10_tau8 6d04fb0698c2a87d6f5f739456f26b3c, W5_k14_tau8 b0ab8831a9f48ca269298dcc0c971c65, W5_k20_tau8 e231f0e4b35de310699fa2791425b1be (<internal>/lad_W5_k{10,14,20}_tau8_m0p2_rs64_n2048_R04_k{10,14,20}_tau8_ell4_prep4_A_R2_avg_plain_v36d);
- W5_k10_tau4 dd11691d63ef9995f6010b789793a846, W5_k14_tau4 d6a3e67886a7731037c32ca99bccea84, W5_k20_tau4 0d4af63403bf64b4dd5c56cedd48aca2 (<internal>/lad_W5_k{..}_tau4_...).
Source: SOLVE/results/attrib/attrib_readout_v1.json "histories".*.meta (path, md5, args). Per-channel matrix 4096 x 4096; channels 10, 14 or 20.

### 2e. Interior completions from 11.125 (W3; engine v36g; all nsites 2048, kappa_sea 10, tau 8, ell 4, prep_ell 4, plain, T_over_rs 40)

All three restart from P64_s4's slice: resume_slice "/workspace/w3/orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_T0011.00.npz"; they arm at the first record failing the hydrodynamic gate at or after T/r_s 11.125 (SOLVE/results/page_v4.1/PAGE_solved_history_v4.1.md:479-482; SOLVE/results/attrib/SOURCES_attrib_v1.json w3_notes "The completions start at T/r_s 11.125 ..., resumed from rung 4's T0011.00 slice").
- held sources, charge projected (W3fs, "frozen-sources"): <internal>/collapse_w3_frozen-sources_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g.json, 61513d1597b4ee5645b9ae035afac342; 321 records to 40; projection tolerance 1e-09 (PAGE_solved_history_v4.1.md:479).
- widened to sigma/a 8 inside (W3pa, "paired-avg"): .../w3_paired-avg/collapse_w3_paired-avg_..._v36g.json, 9888b8a106d8e55ee8747a2c0910e01b; 321 records to 40; shell mixing width sqrt(8^2 - 4^2) a before the ell 4 solve (PAGE_solved_history_v4.1.md:480).
- point-sampled core (ends at 12.5): .../w3_point-core/collapse_w3_point-core_..._v36g.json, 003b5b768112986236af06760b5978f7; 101 records, last 12.5, run ended rc 1 (SOURCES_attrib_v1.json w3_notes).
The two long completions' light runs: v6o_W3fs_k1to1/k2to2 and v6o_W3pa_k1to1/k2to2, nsites 8192, T_end_rs 40, cheb_tol 1e-16 (<internal>/*_d13.json, settings.args).
Paper vs records: paper "continuing the reference collapse from T/r_s=11.125"; records restart from P64_s4's T/r_s 11.0 slice, arming at 11.125; the light comparison is against B2 (v19, 4096 sites), not P64_s4.

### 2f. Principal light propagation (v6o_B2; Fig. 2(a), (b))

page_history_v6o (md5 4c793ad7110ec0e4dcf3881b84d2115b) on record B2n4096 (fields_key "applied"): nsites 8192 (128 r_s; light chain end) against the engine's 4096 (64 r_s), the vacuum exterior appended beyond the engine domain, join_rs "32,64"; channels kappa 1 to 18, one run per channel group (v6o_B2 k1-k6, v6o_B2_k7to9, k10to12, k13to16, k17to18), T 0 to 100 r_s, collect_every_rs 2, interp linear, interp_sub 8, cheb_tol 1e-16, cuts 2,16,24,30,48 r_s, R_collar 16, R_cut 24, eta 0.9192, sector halving on (settings.args of <internal>/v6o_B2_k1_d13.json; composed links <internal>/v6o_B2_k<K>_d13.{json,npz}). Shapes at T = 0, kappa 1: vacuum_n_neg 8192, + sector n_plus 4096, cut 2 r_s n_modes_X 16128 / n_modes_Xc 256, n_sites_X 8064 / n_sites_Xc 128 (same json). Paper statement S1 "8192 sites" (PAGE_solved_history_v4.1.md:24) agrees.

### 2g. Held continuations

- Held at T/r_s 15 and 30 for 300 r_s (paper 07_page_curve.tex, "With the fields held at the records T/r_s=15 and 30"): v6kA_c15 / v6kA_c30, page_history_v6k, record c19 r11 (2048-site engine record), nsites 8192, channels 1,2, freeze_at_rs 15 (30), freeze_duration_rs 300, init_state v6bA_s4_state_at_015.000.pt (030), join_rs "16,32", cheb_tol 1e-14, frozen_prop eig (SOLVE/results/page/v6bA_runs/v6kA_c15/v6kA_c15_summary.json; args list SOLVE/results/page/V6K/hold_args_c15.txt; launcher lineage SOLVE/code/page/launch_page_history_v6_v6.sh:46-62). The plateau numbers 5.4 to 6.3 and -0.017 to 0.076 are cited from version 3b holds (SOLVE/results/page_v4.1/PAGE_v41_part3.md:12, "copied from version 3b, md5 561e18ab").
- Held from 100 to 500 (Fig. 2(c)): v6s_B2hold, page_history_v6s, record B2n4096, nsites 32768 (512 r_s), channel kappa 1 only, T 0 to 500, hold_fields_from 100, hold_step_rs 1, interp linear, interp_sub 8, cheb_tol 1e-16, frozen_prop eig; summary <internal>/v6s_B2hold_k1_summary.json (md5 817c33756ce68561c81ce17070ec78a1 per BUSI/FIGURES_SOURCES_v1.md); ran on R1 2026-10-20 11:30-16:03 UTC (v6o_status.txt). Longer grafts (G2L_c15/c30, --append 69992/139980/265000, page_graft_pullback_v2.py; SOLVE/code/page/pod4_longchains_v1.txt) exist but are not identified as paper inputs.

### 2h. Conformal control (07_page_curve.tex Eq. page-conformal)

v6s_B2_kz: kappa_zero true, record B2n4096, nsites 8192, channel 1 line, T_end 100, interp linear, interp_sub 8, cheb_tol 1e-16, join_rs 32,... (<internal>/v6s_B2_kz_k1_d13.json, md5 8db1de1196f40b617d4b3c26816e9725). Record discrepancy: the json's settings "code" field reads "page_history_v6p.py", while SOLVE/results/page_verify/NOTES_D1_conformal_control_v1.2.md (v1 part, section header) names "page_history_v6s --kappa-zero ... on B2's 8192-site chain". Paper ring circumference L_c = 265 r_s (not a chain length; the ray-circle label).

### 2i. Cubic transfer

- Source record: set17_rs32_A_R2_tau8_ell4_avg, engine_collapse_v17, nsites 1024 (32 r_s nominal; record's own 2GE 29.2504 a), kappa_sea 10, tau 8, to 40; read at slice 72 (T = 18 x 32 a) (<internal>/collapse_set17_rs32_A_R2_tau8_ell4_avg.json, b2df7c8a07130374edcf9e82cd087807; BUSI/notes/GAP10_T7_final_v1.3.md:27, 35). Slice fields file record_slice.npz md5 334ecb67f5ed85811159e35db35866fe (BUSI/notes/O5_witness_v1.1.md:52).
- Cubic operator boxes (four-component, open box, Chebyshev solver, fields held, point sampling): T7 fixed runs H 103 (L 207, faces 3.55 r_s), 20 sweeps of tau = r_s = 29 a, B 7, --rs 29 --ma 0.2 --ell 4 --Rb_frac 3.86 --profile file --sampling point (BUSI/notes/GAP10_T7_final_v1.2.md:21, 25); O5 comparison boxes H 103 (L 207, 3 sweeps) and H 232 (L 465, faces 8 r_s, 10 sweeps) (BUSI/notes/O5_witness_v1.1.md:76). Chain side of O5: page_ops_v6 chain on the same record (O5_witness_v1.1.md:54); its chain length is not stated there: NOT FOUND.

### 2j. Feedback control (Computation C)

- r_s/a 32 member: C32_s4_ell4_tau2_twofield_v3, engine_twofield_v8, nsites 2048 (64 r_s), ell 4, prep_ell 4, kappa_sea 10, tau 2, mix newton, light-kappas 1,2, guard kappa 3, light coupling on, T 0 to 80, 1281 records (BUSI/results/compC/C_readout_s4_v3c.md:3-7; json md5 2516db2052b0a5bf27a6419ba2be6ae7 to 40 and ecf7ad93fc23fa0960850f817ea11084 continued to 80; COMMON argument line <internal>/pod_launch_C_v11.sh:100).
- Coupling-off control: C32_s4_ell4_tau2_twofield_v3_off, same args with light_coupling off; C_readout_s4_v3c.md lists it to T/r_s 50 (801 records; md5 22283cccc6a0dfd1f8b81db68fbf5887 and 45c705072201db7acd6d272413487a09); a later continuation to 80 exists: <internal>/collapse_C32_s4_ell4_tau2_twofield_v3_off.json (md5 b9e675aaa2dbb94bc6d12ba6cac7fa9a; nsites 2048, tau 2, T_over_rs 80, resume_slice ..._off_T0050.00.npz). Paper vs records: paper reads the comparison "to T/r_s=80"; the v3c readout's matched window ends at the control's 50.
- r_s/a 64 pairs (paper: "reach formation only with the light field's occupied set reassigned"): C64_s4_ell4_tau2_twofield_v3 (nsites 2048 = 32 r_s, tau 2, to 56, Newton, light kappas 1,2) <internal>/segment_c64s4_lagged_20261018T203005Z; C64_s4_ell4_tau1_twofield_v4L (tau 1, to 16) <internal>/.

### 2k. Ladder members (Patch 3; from ladders_readout_v1_3.json members.*.meta, scratch copy <internal>/ladder_meta.txt; table SOLVE/results/ladders/SCALING_ladder_v1.3.md:68-86)

All R0_over_rs 4 unless stated, reference R2, prep_ell 4 unless "none".

| member | (zeta, r_s/a) | nsites (chain end) | kappa_sea | tau | ell | engine / mix | T span | json md5 |
|---|---|---|---|---|---|---|---|---|
| S4_16_s4 (E0/Emax 1.14) | (0.2, 16) | 512 (32 r_s) | 10 | 2 | 4 | v36d plain | 0-40 | b87134f9742e57c16b0cb3bfd1b2fd9b |
| S4_16_s2 (1.14) | (0.2, 16) | 512 | 10 | 2 | 2 | v36d plain | 0-40 | 52f02084c6fcb0ca9690af82b9bb5680 |
| S4_16_s4_tau1 (1.14) | (0.2, 16) | 512 | 10 | 1 | 4 | v36d plain | 0-16 | 9689881076f6c34eab7f691327be555b |
| M3a_s4 (1.14) | (0.1, 64) | 2048 (32 r_s) | 10 | 8 | 4 | v36d plain | 0-40 | 778dc5cef53012b71461a2e1d2de1044 |
| M3a_s2 | (0.1, 64) | 2048 | 10 | 8 | 2 | v36d plain | 0-40 | d8aa172c70456055b6867f0489e3cf96 |
| M3a_s4_tau4 | (0.1, 64) | 2048 | 10 | 4 | 4 | v36d plain | 0-16 | 66382686ad07f2ca581659153c90767a |
| RS32_s4 | (0.2, 32) | 1024 (32 r_s) | 10 | 2 | 4 | engine_collapse_v31 plain | 0-40, 641 rec | 898fd8ebe0dd9424de6f20074d04f568 |
| RS32_s2 | (0.2, 32) | 1024 | 10 | 2 | 2 | v32 newton | 0-40 | d35bc0fd5f1b7765e495668f4300b90c |
| RS32_s4_tau1 | (0.2, 32) | 1024 | 10 | 1 | 4 (prep none) | v25 plain | 0-40 | 6e8b8973c771c6a4a01db8926c30a46a |
| S4_128_s4 | (0.2, 128) | 4096 (32 r_s) | 10 | 8 | 4 | v36d plain | 0-19.0625 (306 rec) | 5fe643bb6c3ef225f05e194c38908a6a |
| S4_128_s4_nk_v36m / _v36q | (0.2, 128) | 4096 | 10 | 8 | 4 | newton | to 2.94 / 2.81 | 2889024ae017804f75cec4e152e6c6e2 / ea458c364755815db5584b98cbb4b1aa |
| S4_128_s8 | (0.2, 128) | 4096 | 10 | 8 | 8 | v36d plain | 0-40, 641 rec | ff475918f57758118221aa59fda52a86 |
| S4_128_s16 | (0.2, 128) | 4096 | 10 | 8 | 16 | v36d plain | 0-40, 641 rec | c2aee55e56201f6a216aa9844dd6bcdb |
| ISO_128_s4 (isotropic) | (0.2, 128) | 4096 | 10 | 8 | 4 | v36nq newton, dust_set isotropic, resumed from T0003.00 | to 40 | 74e11d8113dbb82ff350377b29814da9 (<internal>/run_h2) |
| M3b_s4 | (0.4, 64) | 2048 | 10 | 8 | 4 | v36d plain | 0-40 | 6209f14cf64e8a4f614babccb28eb04c |
| M3b_s8 | (0.4, 64) | 2048 | 10 | 8 | 8 | v36d plain | 0-40 | 0b834bc19856a2ae6c9f46ce3bd21445 |
| M3c_s4_nk | (0.1, 96) | 3072 (32 r_s) | 10 | 8 | 4 | v36j newton | 0-22.17 | 60f92fbfae76f621867f543eaad49029 |
| M3c_s4_plain | (0.1, 96) | 3072 | 10 | 8 | 4 | v36d plain | 0-22.08 | 6489517c34dccd5c536e99a7a03d3bb3 |
| M3d_s4 | (0.05, 326) | 3912 (12 r_s) | 12 | 8 | 4 | v36e plain | 0-0.491 (21 rec), T_over_rs arg 30 | 726cb35fb005e3ef2ea05c3e3adfa38f |
| S4_R2_m0p2 (release 2 r_s) | (0.2, 64) | 2048 | 10 | 8 | 4 | v36d plain | 0-40 | aae1a09ee3ebdeac0ad5efc726ce5df3 |
| S4_R2_m0p1 (release 2 r_s) | (0.1, 64) | 2048 | 10 | 8 | 4 | v36d plain | 0-40 | d9b428d05430ac7e989fd2f62f92c5a1 |
| ISO_64_s4 (isotropic, composed isoB) | (0.2, 64) | 2048 | 10 | 8 | 4 | v36n plain / Newton episode 13.625-14.5 | 0-40 | 227cd66d25ffc650ffee69eb0da6270c (<internal>/collapse_isoB.json) |
| ISO_64_s4_plain | (0.2, 64) | 2048 | 10 | 8 | 4 | v36n plain | 0-40 | fde06ed72e8ec1d22759f576186af5aa |

Paths: <internal>/collapse_lad_<...>.json as listed in each member's meta.path (ladder_meta.txt); RS32_s4 <internal>/collapse_c31_rs32_A_R2_tau2_ell4_prep4_avg.json; RS32_s2 and RS32_s4_tau1 under <internal>/.

E0/Emax = 1.14 members: the "E0/E_max (charge ratio, c_E 0.135858)" column reads 1.14 for S4_16_s4, S4_16_s2, S4_16_s4_tau1 and M3a_s4 (SOLVE/results/ladders/SCALING_ladder_v1.3.md:66-86); S4_R2_m0p1 reads 1.13.

r_s/a 256 (not in ladder v1.3): the "flat" tapered-box preparation members, engine_collapse_v34 line v36nq_rd13mfb, nsites 4096 (16 r_s), kappa_sea 15, tau 16, ell 8 (F256_s8) or 4 (F256_s4), prep_ell 4, Newton, R0_over_rs 4; flatbox label "the lowest levels of the free chain with U = E_k ((R - R_in)/W)^2 beyond R_in = 3.5 r_s (W = 0.5 r_s ..." (<internal>/collapse_lad_F256_s8_fbt05_m0p2_rs256_n4096_R04_k15_tau16_ell8_prep4_A_R2_avg_newton_v36nq_rd13mfb.json, md5 1b78aa940acc9b75c1f925a903db968e, T_over_rs 13, resumed from T0004.50; halved-step control tau 8: <internal>/..._tau8_..._half450.json, 7d325d2c7b1ff3b15ffec36e9a3c0188). A well-prepared W256_s4 exists (n4096, kappa_sea 14, tau 16, ell 4; <internal>/w256rd1d_dry). Plan vs record: SOLVE/results/NOTES_reopen2_card_plan_v1.md:55 planned "--nsites 8192 (32 r_s/a ...) --ell 16 --tau 16"; the stored 256 members carry nsites 4096 and kappa_sea 14 or 15. Which 256 run carries the paper's "formation inside the gate ... at sigma/a=8" was not pinned to a single file here.

### 2l. stored_identity.json

SOLVE_B second-route identity (data/solve_b/results/PAGE_secondroute_v1/stored_identity.json in the archive; local <internal>/PAGE_secondroute_v1) was not used for chain lengths in this section; its shapes belong to the second-route readout (markers 8, 9).

Status: PARTIAL. FOUND with record paths and md5: the three reference records, both step controls and the conserving step, the width controls, the six channel-count rungs, the three completions, the principal light run, both held continuations, the conformal control, the cubic-transfer source record and boxes, the feedback member and control, and every Patch 3 ladder member at r_s/a 16, 32, 64, 96, 128, 326 and the 2 r_s releases and E0/Emax 1.14 members. NOT FOUND: the chain length of the O5 chain side of the cubic transfer (looked in BUSI/notes/O5_witness_v1.1.md, GAP10_T7_final_v1.2/1.3.md); the single 256 run behind the paper's sigma/a 8 formation sentence (candidates listed); the crossing-convention step controls were not extracted.

## Marker 3

> [VALUE NEEDED: actual reference path, overlap/assignment rule, degeneracy and zero-crossing convention, subdivision limits and per-step subdivision choices for every principal run and control; extract the executed continuation functions and logged assignment/subdivision records associated with the same lineage.]

### 3.1 Executed continuation for P64_s4 (and every v36d/v25+ member): engine_ops_v3.continue_sea_v3

Selected by `continue_ref_v34` (`engine_collapse_v36d.py:1353-1358`): `OPS3.continue_sea_v3(ch, window, f_cur, e, U, n_sea, tau, c_LZ, eig_at, clocked_host_sparse)` since weyl_certified is null. File `<internal>/engine_ops_v3.py` (md5 460664cd52b8babd42f9a9c227499662 = archive original_md5).

- Reference path: from the window's last entry only (the previous accepted record's R2 and applied fields; window trimmed to one entry after each record, `engine_collapse_v36d.py:2369-2373`), along the straight line of the fields f(s) = f_prev + s (f_cur - f_prev) (`engine_ops_v3.py:124-125,141-142`). Starting R2 at T = 0: negative-energy eigenvectors at the prepared fields (`engine_collapse_v36d.py:2069-2072`).
- Rate and sub-steps: dH = h_m[dN, dv] (host sparse); ||dH||_G = largest absolute row sum (Gershgorin, `engine_ops_v2.py:102-104`); s_T = ||dH||_G / tau; tau_LZ = 1/(c_LZ sqrt(s_T)) (`engine_ops_v3.py:125-128`).
  - if tau_LZ <= tau: nsub = max(1, round(c_LZ sqrt(tau ||dH||_G))) sub-steps, eigendecomposition at each intermediate configuration (eig_at), assignment at every sub-step, the last sub-step on the trial's own eigendecomposition; mode "LZ sub-steps (nsub n)" (`:132-154`).
  - else: one-step assignment at the endpoint; mode "one step", nsub 1 (`:155-161`).
- Subdivision limit: none; nsub is not capped and the continuation never rejects (`:133`; docstring `:2-5`). c_LZ = 1.0 (args).
- Weyl test (label only): verified iff 2 ||dH||_G/nsub < gap at each sub-step start, gap = previous "weyl_gap" (0 if a block is partly occupied, else the occupied-to-empty gap between blocks); failure only sets verified False and the sticky label "LZ convention (c_LZ = 1) from T/rs ..." (`engine_ops_v3.py:130-160`; label `engine_collapse_v36d.py:2322-2326`).
- Overlap/assignment rule (`assign_max_overlap_v3`, `engine_ops_v3.py:53-119`): A = R2_prev^dag U; weights w_k = sum |A_.k|^2; the n_sea largest weights occupied, order by weight descending, ties by energy ascending (`:64`); count exact.
- Degeneracy convention: a block = eigenvalues whose whole width e[j-1] - e[i] < tol_block x max|e| (no transitive joining), tol_block 1e-8; if intermediate weights (0.1 < w < 0.9) remain, retried at 1e-6 and kept if fewer (`:23-34,37-50,58-63`); multi-member blocks rotated to the eigenbasis of their weight matrix A_b^dag A_b, weights replaced by its eigenvalues (`:44-48,88-91`).
- Tie at the cut (|w - w_cut| < 1e-12 straddling n_sea): broken by the predecessor's mask when it decides the tie, else by energy and flagged ambiguous_tie (`:65-81`).
- Zero-crossing convention: a rotated block whose eigenvalues straddle 0 gives its vectors no energy sign (excluded from pocket counts; "straddling_blocks", "sign_free_vectors") (`:94-99`); pockets = occupied with e > 0 plus empty with e < 0 among signed vectors; inversions counted by block order (`:100-104`).
- Fields at which the sub-steps are evaluated within a record: the trial applied fields of each self-consistency iteration, each time from the same previous-record start (`engine_collapse_v36d.py:2225-2231`).

### 3.2 Logged per-step choices (P64_s4)

- Per record and channel the run json stores `continuation[kappa]` = {mode, nsub, m, m_eff, verified, margin_min, s_T, tau_LZ, norm_dH_G, window_consistency, n_intermediate, min_occupied_weight, max_empty_weight, inversions, n_pocket, pockets_occupied_positive, holes_empty_negative, tol_block, rotated_blocks, gap_between_blocks, partly_occupied_blocks, mask_equals_negative_set} (`engine_collapse_v36d.py:2296-2301`); "nsub max" per record in the log line (`:2592`). ambiguous_tie, tie_by_predecessor, straddling_blocks and sign_free_vectors are computed but not copied into the record (not in that key list; grep of the json finds none).
- Tallies over the 321 records x 10 channels of the P64_s4 json (box, grep): mode "one step" 3031, "LZ sub-steps (nsub 1)" 143, "LZ sub-steps (nsub 2)" 36; no nsub above 2. LZ sub-step mode first at T/r_s 10.75, on 27 records. nsub 2 occurs at T/r_s 11.75 (1 channel), 11.875 (2), 12.0 (3), 12.125 (4), 12.25 (4), 12.375 (5), 12.5 (5), 12.625 (4), 12.75 (4), 12.875 (2), 13.0 (1), 13.125 (1). "verified": false on 454 channel-records (96 records), first T/r_s 11.125, last 36.25; log label "LZ convention (c_LZ = 1) from T/rs 11.125 (channels 1..10)" (log lines 18.125 to 40, all "nsub max 1").
- Full per-record lists: run json above (records[].continuation); log `runs_pod7_ell4/logs/..._to40.log` covers 18.125 to 40 only; the 0 to 18.125 segment's log is not in runs_pod7_ell4/ (not found on box under v36d_from0/).

### 3.3 Controls and comparators

- sigma/a 8 (P64_s8, plain v36d, `<internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json`, md5 d1ca037a77c198aa83b33055495eceff): "one step" 3099, "LZ sub-steps (nsub 1)" 111; ops v3.
- sigma/a 2 (P64_s2, composite Newton history v36cc/v36d, two-stage source continuation, end_above 1e-3, nk_set resolve; `<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json`, md5 0a005ac5f66bef02a0104e4278867765): "one step" 2477, nsub 1: 287, nsub 2: 298, nsub 3: 142, nsub 4: 6; ops v3. v36d's second-round set rule (CROSS-LOWER trial, fewer flips, ties by larger smallest gap) applies only under --nk-set resolve (`engine_collapse_v36d.py:2-11`; `cross_lower_v36d` `:2867`).
- Formation second route (Newton v36nq, 11.5) uses the same c_LZ 1.0, window_max 4 arguments (`p64_newton_h100_1/args_v1.txt`); its continuation module identity inside v36nq was not opened.
- c19 production (Fig. 1) records 11.125 to 40: engine_ops_v2.continue_sea_v2 (`engine_ops_v2.py:1-20,107-...`; md5 cdabf12b56be61d1d26879b57f073650): same tau_LZ/nsub rule; else m = round(tau_LZ/tau), at most 4 and at most the records held, assignment in one step from the R2 m records back (sliding window); blocks by successive near-equalities |e[j] - e[j-1]| < 1e-8 scale (transitive, unlike v3's whole width; `engine_ops_v2.py:22-32`); ties by energy only. Logged modes (box c19 json): "window (m 1 of 1)" 192, "(m 2 of 2)" 1613, "(m 3 of 3)" 310, "(m 4 of 4)" 32, "LZ sub-steps (nsub 1)" 137, "(nsub 2)" 36. Records 0 to 11 (89) are the v17 parent's: Weyl test at nsub 1, else subdivisions 2, 4, ..., 64 with the test per sub-segment, finest kept and flagged, an unresolved split rejects (docstring `engine_collapse_v36d.py:242-249`); 89 entries "halvings": 0 in the c19 json.
- Step controls: P64_s4_tau4 = `collapse_c19_rs64_A_R2_tau4_ell4_avg.json` (engine v19, md5 f2e1d8a7483447426855e3211535a367); P64_s4_tau2 = `C3prod25/collapse_c25_rs64_A_R2_tau2_ell4_avg.json` (engine v25 i.e. ops v3, to 21.5, md5 9ba6f6ff302d8a903925f20583c96656) (ladder readout meta). Their per-step mode tallies were not extracted.
- Crossing-convention controls ("three settings", 05_star.tex:85): c_LZ 0.5 = c25_rs64_A_R2_tau8_ell4_avg_cLZ0.5_from14, c_LZ 1 = c19_rs64_A_R2_tau8_ell4_avg_r11, c_LZ 2 = c25_rs64_A_R2_tau8_ell4_avg_cLZ2_from14 (`<internal>/clz_ladder_v2.md:3`); identical to the printed digits through 13.25, first differences at 13.375 (same file, table). These are on the c19/v25 lineage, not on P64_s4's v36d run.

### 3.4 Paper vs code

- Paper (05_star.tex:57-58 at 24434791d9f): "continued spectrally over the step's endpoints or its subdivisions; the per-step choice is not read." Code logs the per-step choice per record and channel (3.2). Disagreement in wording reported, not resolved.
- Historical "halves the interval at most three times" (declared_local_area.tex, quoted in the correction text) does not match any executed continuation found: ops v3/v2 have no cap; v17 (c19 records 0-11) allows up to 6 halvings (64 sub-steps).

Status: FOUND for P64_s4 (rule, degeneracy/zero-crossing convention, no limit, logged choices) and for c19, P64_s8, P64_s2 and the c_LZ controls. PARTIAL for the step controls (tau 4, tau 2) and the long interior completions, held continuations, conformal, cubic-transfer and feedback controls: their per-step tallies were not extracted (records exist in the run jsons named in the ladder readout meta); v36nq's internal continuation not opened. Looked in: engine_collapse_v36d.py, engine_ops_v3.py, engine_ops_v2.py, box run jsons and logs listed above, solve results/engine/C3lz25, ladders_readout_v1_3.json.

## Marker 4

> [VALUE NEEDED: assertion-to-artifact map for all principal figures and numerical result families, including exact producer functions, executed source hashes, launch arguments, dataset hashes, payload selectors, analysis functions, independent-route dependencies and final PDF hashes; obtain from the figure scripts, their input bindings, execution records and current paper build.]

Abbreviations: WT-busi = <internal>/busi; WT-solve = <internal>/solve; WT-solveB = <internal>/solve; DEL = <internal>/deliveries; ARCH = <internal>/entanglement-stars, branch origin/main (f690b45; local checkout is on branch sync-corrected-observable 3a1f962, which does not contain fermion_new_codes/, data/ or proofs/); box = 192.168.1.93. "md5" = md5 of the file bytes as recorded in the cited record; sha256 computed here only where stated.

### 4.1 The three figures of ms_fermion_new.tex at 24434791d9f

Included figures (PAPER_flat_24434791.tex:1510, 2192, 2598): item5_collapse_v3.1.pdf, page_v5.pdf, page_curve_amps.pdf. None of the three PDFs, and none of the figure-data .npz files, is in ARCH origin/main (git ls-tree search for *.pdf and the names: no match outside the old fermion/ tree).

| Figure | PDF at 24434791d9f: bytes, md5, sha256 | commit that added it |
|---|---|---|
| Fig. 1 fermion_new/figures/item5_collapse_v3.1.pdf | 37040; 643fd822b0cae873cbba8c84546ad866; 97f0d657cb4393a10a3306b5ba8fa25914834f33a6a8761f2570f5615610eafb | e804b9c5fc8 (20 Oct 2026) |
| Fig. 2 fermion_new/figures/page_v5.pdf | 35071; 83eacd3a3533da5424da89269ca6f61b; 9fb426fc4697eed3cb3dd32862f4545dc606f018f1d81555147a6430190b1718 | c6785dff871 (20 Oct 2026) |
| Page-curve figure fermion_new/figures/page_curve_amps.pdf | 244128; 1b053d204f4194719f405c226473dc60; 2404d02d8127a4a0a15aec3bda072cf1b872f1a33d0927067e543a3b6307f476 | a55713ff364 (26 Oct 2026) |

Source: `git -C <internal>/.wt-fermion-minimal show 24434791d9f:<path> | sha256sum / md5sum / wc -c`; `git log -- <path>`. The delivered copies are byte-identical: DEL/claude_busi/FIGURES_v1/item5_collapse_v3.1.pdf and page_v5.pdf (same sha256), DEL/codex_solve/page_curve_amps_R1.pdf (same sha256; also stated in DEL/codex_solve/PAGE_FIGURE_R1_v1.md:110).

#### Fig. 1 (item5_collapse_v3.1.pdf)

- Plot script: WT-busi/code/figures/figure_item5_collapse_v3_1.py, md5 8eab4a5f77ce56e64158cbb5495bd745, sha256 ad5d992d73faea243d303a82a2657c4d02ba790bc288e9615dfee3d214cd98fd (box copy <internal>/code). Function main() (lines 28-97 of the file): loads only fig_item5_collapse_v3_data.npz; panel (a) keys a_r_over_rs, a_N_T0, a_absv_T0, a_N_T15, a_absv_T15, a_zeros_T15_over_rs, a_Rt_T15_over_rs, a_circle_N; (b) contourf of b_N_minus_absv at level 0, b_Rt_over_rs, b_2gE_over_rs, b_Rc_over_rs, ev_formation/ev_closes/ev_reopens; (c) c_T_s{2,4,8}, c_minNv_s{2,4,8}. Usage line: `python figure_item5_collapse_v3.py DATA_NPZ OUT_STEM` (docstring, line 8). No recorded launch command found; the README states it was rendered on the box with matplotlib 3.10.8, qi2 env, nice 10 (DEL/claude_busi/FIGURES_v1/README.md:3).
- Data script: WT-busi/code/figures/figures_data_v1.py, md5 e2469b43bc4547c0e695f2e7a1eb6fb8, sha256 4d7186f1c761bee65a346d38088f9d687e33dff4b00018a14ce3cfdeb63159a4. Usage `python figures_data_v1.py OUT_DIR PAGE_FIG_NPZ READOUT_TSV` (line 15). Executed arguments, reconstructed from its own log (DEL/claude_busi/FIGURES_v1/figures_data_v1.log, md5 26d63b440e8a35afe910af8034dc71a1): OUT_DIR = out (box <internal>/out), PAGE_FIG_NPZ = <internal>/PAGE_fig_v4.1_data.npz, READOUT_TSV = in/PAGE_v4.1_readout_export.tsv. No shell script recording the command line exists in <internal>/figures_v1 (ls: code/, in/, out/ only).
- Output: fig_item5_collapse_v3_data.npz, 104762 bytes, md5 941fa62180cd9799e0fed0c18059e62f, sha256 0c753bfc3e1691d895b194347351956ad3b40552a9cbbeeb83b89e9856a0d350 (DEL/claude_busi/FIGURES_v1/).
- Inputs and selectors (figures_data_v1.py lines 25-32, 52-94; md5 from the log):
  - c19 production fields <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz, md5 876dd5c6dee34b1c290b33eaeb4d25b1; keys T, r_nodes, N_nodes, v_nodes (node grid, 513 nodes at 4a, 321 records; README "Series and sources"); records selected by argmin |T/64 - 0| and |T/64 - 15|; radii kept r/64 <= 2.
  - c19 run json .../collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json, md5 e27e457d37615cff8d5216f05a494f89; selector records[i15]["trapping_zeros"] (0.11012479, 0.80875633 r_s, log).
  - P64_s4 fields <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40_fields.npz, md5 01444a399b9466047a5b0f118af61f77: check only, not drawn (lines 108-115; log: max|dN| 5.363e-10, max|dv| 1.103e-08, 0 sign-mask cells differ).
  - Ladder readout <internal>/ladders_readout_v1_3.npz (md5 f172431d6a99447f27bdd046a6482af3) and .json (md5 63d46f940870542526e597984668964d); keys P64_s4__T, __Rt_rs, __EE0, __Rc, __minNv, __certified; json members.P64_s4.counts.two_gE0 (59.0689), .formation.T (12.125), .episodes[0].closes_at (17.375), [1].first (18.375); panel (c) members P64_s4, P64_s2, P64_s8 cut at cert_end_index.
  - Conversions applied in the data script: T/r_s = T/64, r/r_s = r/64, 2GE/r_s = (E/E_0)*59.0689/64 (notes string, lines 95-104).
- Producer of the ladder readout used by Fig. 1: box <internal>/ISO64_bridge_ladder_v1.py (md5 edfd5a4df0d423b13ab6779a7d67da6e, box MANIFEST_v1.md5), which reruns the v1.3 readout with ISO_64_s4 added and an 8-record Newton patch for ISO_64_s4 only (ladder/newton_patch_v1.json, md5 89b5102203cfa04082ae9974890d25ef; run.log last line "newton records patched: ... ISO_64_s4: 8", all others 0). This file (63d46f94...) differs from the solve lane's own v1.3 readout <internal>/ladders_readout_v1_3.json (md5 930e75da87c1559f0318ed7a8a5206ed, MANIFEST_v1_3.md5), produced by WT-solve/code/ladders/RUN_ladders_v1_3.sh (md5 2be90e40d1d7ae58e5c84d96c5719b6e) calling ladders_readout_v1_3.py (md5 20048af9566537dc7540b4dad33b4e8a) with --catalog catalog_v1.json, catalog_v1_A.json, catalog_v1_B.json, --certext certext_production_v1.json, --followup certext_followup_v1.json.
- Run records behind Fig. 1 (from ladders_readout_v1_3.json members.*.meta): P64_s4 json md5 d7d69adcea7a43f8375ab4c7c5a80efc; P64_s2 <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json md5 0a005ac5f66bef02a0104e4278867765; P64_s8 <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json md5 d1ca037a77c198aa83b33055495eceff; P64_s4_c19 md5 e27e457d37615cff8d5216f05a494f89 (engine entries: "None" 0-11.0, 89 records; engine_collapse_v19 11.125-40, 232 records).
- Note: panels (a) and (b) are drawn from the c19 production record (engine v19), not from P64_s4 (v36d from T=0); the paper caption does not name the record. The data script records the choice of node grid over site grid (README "Choices": site grid's sign disagrees with the readout at 9 records).

#### Fig. 2 (page_v5.pdf)

- Plot script: WT-busi/code/figures/figure_page_v5.py, md5 6a8e6283fa35b6dd3ff23c6959e0190a, sha256 058db684dacaa0aa7fb6a1a07449d7fd5b37605cbd9b0f9ec88757f52c44b62c; main() lines 20-77: loads only fig_page_v5_data.npz; (a) a_S_cut{2,16,30} and a_light_cut* ticks, budget line constants T_FORM 12.125, BUDGET_100 10.4 (line 19, "the plan's printed endpoints", not computed); (b) b_ratio_signed, b_ratio_mean_pm5, finite and t >= 10; (c) c_excess16_cut*, HOLD = 100.0; the text "3.9x10^5 r_s" and "1.7 to 2.7 parts in 10^4" are literals in the script (lines 48, 77).
- Data: same figures_data_v1.py run (above); output fig_page_v5_data.npz, 8453 bytes, md5 0cc932f3376506c341584c74c50b8e65, sha256 960f63df63070fd2a66f5374d36a0ce96e0b9164f648da41278e3210e064f0ec. Selectors (figures_data_v1.py lines 123-152): P["clocks"], P["SUM_B2_cut"+c], P["light_band_cut"+c] for c in 2,16,30; P["ratio12_TH"] and its mean over |t - t_i| <= 5 (all finite stored checkpoints); hold json rows[].per_channel["1"].interior.cuts[c].excess times 16; hold["T_hold"]/64.
- Inputs: PAGE_fig_v4.1_data.npz (box <internal>/out_v41, md5 5119ffa229c003ebb7eb69e13b96962c); PAGE_v4.1_readout_export.tsv (md5 8636ce15063259bcf8502c90aed028ca; copy check only, 93 pairs, max |diff| 0); <internal>/v6s_B2hold_k1_summary.json (md5 817c33756ce68561c81ce17070ec78a1) and _interior.md (md5 45ab942d16fb35879019e6e2bc171905) (log).
- Producer of PAGE_fig_v4.1_data.npz: WT-solve/code/page_v4/assemble_page_v41_v1.sh (copies and runs run_page_v4_box_v1b.sh on the box) -> page_v4_data_v1b.py (md5 26f909ef3cecee6506b32a6ffcd9ce45) -> page_v4_data_v1.json (0b36f0bf5f27cc240bc9911b7817cd6e) and page_v4_data_v1.npz (0ddef18391770b4eaa2e6a3fbb6405e4) -> page_v4_fig_v1b.py (md5 54753621a327c49f3fa556c0a839361e) -> PAGE_fig_v4.1_data.npz; page_v4_tables_v1b.py (13fd1cc05ed0caaacb846bed0470cfa5) -> page_v4_readout_export_v1b.tsv (= PAGE_v4.1_readout_export.tsv, 8636ce15...). Output md5s: WT-solve/results/page_v4.1/OUTPUTS.md5; input md5s (128 lines): WT-solve/results/page_v4.1/MANIFEST.md5 (B2 production kappa 1-18 summary and d13 json under <internal>/v6o_B2*, post-pass <internal>/W1_B2_composed_v3.json md5 32a4752f37ccddadc8d4983d6089e1dd, probes kappa 24/32/48, s8 production). Busi summary of this chain: WT-busi/FIGURES_SOURCES_v1.md:18-32.
- Light-channel runs behind the sums: code page_history_v6o.py; summary <internal>/v6o_B2_k1_summary.json "settings": record /workspace/engine/B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json, fields ..._B2n4096_fields.npz key "applied", nsites 8192, T 0 to 100 r_s, collect every 2 r_s, R_collar 16 r_s, join_rs "32,64", cheb_tol 1e-16, interp linear, interp_sub 8, kappas "1-16", columns at 12, 20, 40; "history": 801 records, tau 8, ell 4, g 0.25, engine_sites 4096. So the Fig. 2 light field runs on the B2 history (c19 r11 continued on 4096 sites), not on the P64_s4 file. Second route: WT-solveB results PAGE_secondroute_v1 (archived as data/solve_b/results/PAGE_secondroute_v1/, 46 files); per DEL/codex_solve/SECONDROUTE_PAGE_excess_v1.md:5 and :479-487 it is "independent arithmetic on those exports, not a regeneration of the light occupied columns, covariance, or restriction eigenvalues", reading the same <internal>/v6o_B2_k<K>_d13.{npz,json}, K = 1-18 (shared inputs). Two-route comparison: data/hreduce/results/page_balance_compare_v1/comparison_result.json (manifest md5 5ba7705db7242b9cc4896ecc738619b8).
- Panel (c) hold: producer page_history_v6s.py (busi w1 launchers, e.g. WT-busi/code/w1/launch_v6s_B2_card_v2.sh); reading DEL/lead_gaps/FableAPI_reading_B2_hold_drain_v1.md. No second route located.

#### Page-curve figure (page_curve_amps.pdf)

- Route 1 (the installed PDF): DEL/codex_solve/page_curve_amps_R1.py, 323121 bytes, sha256 259d449f96def51a949edcca2f1ce685c84427e1e59cc1b8b607e078c99b9478, md5 737b822b025c2dc2d5c69b8ef20c49f0; a single runner embedding eleven hash-verified bodies (e.g. code/producer_final_embedded_v1.py sha256 0f443c8be5c1f73bba05bbe84545246ae02a413ccbc8203a13c8e1772fb158f7, script header), stages `--stage unpack|pairs|spectra|anchors|original-inventory|recovery|admit-recovery|primary1920|exact-controls|exact-crosschecks|chain-inventory|spectral-evidence|inventory-evidence|checks|render|plot-check` (PAGE_FIGURE_R1_v1.md:39); base <internal>/page_fig_r1, PYTHON <internal>/python (script lines 14-15). CSV page_curve_amps_R1.csv sha256 2d9d2a2c6e40c3a6c04b06f697d54f95696a87a0c5de13ccae0eb6d118b11557 (md5 94f31b217f809301e7d958fda5a34544); PDF sha256 2404d02d... (PAGE_FIGURE_R1_v1.md:108-110). Box evidence: <internal>/final_plot_artifact_check.json, figure_geometry.json, raw/chain_inventory_1920_v1/validation.json (PAGE_FIGURE_R1_v1.md:116). Quadrature: 960/1920 nodes per channel primary, 400/800 scalar; channels k = 1..8 multiplicity 64k; q in [0.001, 3] (PAGE_FIGURE_R1_v1.md, "Comparator and scattering evidence").
- Route 2: DEL/claude_alt_solve/page_curve_amps_R2.py (sha256 4c31d159ec0f3708ffb96d80a11a6058899c581d24acb50c91ab042caaa6f140), page_curve_amps_R2_data.py (58a7816478c230e50d459a8e935553c971a000977169fa87fc5a6b32c48648ac), page_curve_amps_R2.csv (77274167a173838efef57e69783da3ce9e83aa6c39eb1060e559acecbf72d77d), page_curve_amps_R2_report.json (60b420cca0b8fa6b169d07196308787bd743fa712b197156bff155235b02b400), page_curve_amps_R2.pdf (73b10141a1cff58d5586b867a25e8b3d410ea3b225e0d48376d51a039f29da1f, not installed); box <internal>/page_fig_r2. Route comparison: DEL/lead_gaps/PAGE_FIGURE_COMPARE_v1.md (C_0, D_0, calR, T_W, r_0, z_x, tau_x agree to all printed digits; the routes are stated blind to each other; shared input is the signed spec WRITE_PAGE_SIGNED_v1.md sec. 3 A-J).
- Neither route's code nor output is in ARCH (archive date 23 Oct; figure added 26 Oct).
- The paper worktree also holds fermion_new/figures/page_conditional_{data.py,fig.py,npz,pdf} (sha256 of the .py/.npz at 24434791d9f: data.py 061a0816..., fig.py e6492cff..., npz 6a334167...); page_conditional.pdf is not \included at 24434791d9f.

### 4.2 Result families (pfix1 sec. 5 list plus Patch 5)

"R1" = first route, "R2" = second route. "Shared" = inputs the two routes both read. Where only a directory or report is named, the exact producer function and launch line were not pinned (PARTIAL).

| Family | R1 producer / record | R2 producer / record | Shared inputs; archive status |
|---|---|---|---|
| Formation and width/step/channel controls | Engine runs, records and args per member in ladders_readout_v1_3.json members.*.meta (see marker 2 section); readout producers above. | P64_s4 by Newton from T=0: <internal>/collapse_lad_P64_s4_newton_v36nq_h100_1.json (md5 8f31394bc66d929e3d53aad9e43c1d21, registry_v1.json key P64_s4_newton, "FORMATION_newton_route_v1"); report WT-solve/results/FORMATION_newton_route_v1.1.md; reading code WT-busi/code/readings/formation_newton_v1/ (fn_*.py, run_box.sh) | Same preparation and engine family (v36 lineage); R2 is a different solver (Newton), same chain. Run records not in ARCH (box only). |
| Size and parameter ladders | Same readout (ladders_readout_v1_3, members M3a/M3b/M3c/M3d, RS32, S4_128, S4_16, S4_R2, ISO_64; run.log at <internal>/run.log lists records, certified extents and formation per member) | WT-solveB code run_LADDERS_v1p3_arithmetic_v1.py and compute_LADDERS_v1p3_*.py; results data/solve_b/results/LADDERS_v1p3_arithmetic_v1/ (145 files in ARCH; e.g. scalar_records/P64_s4.json original_md5 90f6ad73da2d997472f73d643d4d1fde) | Arithmetic on the same stored run jsons (not a re-solve). |
| Redshift | DEL/route2/REDSHIFT_received_v4.1.md with REDSHIFT_received_v4.1_scripts/ (rz_v4_head_v3.py md5 d11191e596c9641fa89645406ba34449, rz_v4_variants_v4.py ab7af70a3dd306ccf5354877b482d15e, rz_v3_law_v2.py 24c74d4ebbbd2e4fd29374d55d6a4a5e, rz_v4_a2fix_v3.py, rz_v4_dispev_tables_v3.py; MD5SUMS.txt); ray passes rz_rays_v4n / rz_rays_v4nc and re-integration rz_v4_reint_v1.py (REDSHIFT_received_v4.1.md:96, :1037); history registry WT-busi/code/redshift/registry_v1.json (md5 efcfad78bb6e2c942f21cd4ef9028270; reader read_history_v1.py) | "independent integrator" rows (Table R1v4, REDSHIFT_received_v4.1.md:39, :1037) are a second integrator on the same events; solve_b WEB40I_ray_window_v1 (ARCH data/solve_b/results/WEB40I_ray_window_v1/) | Same stored fields (registry). The rz_* scripts are NOT in ARCH (git ls-tree: no match). |
| Core density and spectra | WT-solve/results/core_well_v1, core_hf; reading DEL/lead_gaps/FableAPI_reading_CORE_HF_v1.md | hreduce second route: ARCH data/hreduce/results/second_route/core_well/ (current_v1/core_well_current_v1.json md5 2bdbfd481d272abb892e40e9c7d7586f; profiles_a/b tables); WT-busi/code/readings/route2_core_b29_v1/ | Same stored slices. |
| Exchange, variance, branch witness | WT-busi/code/hf/core_hf_v1.py, qh_branch_witness_v1b.py (RUN_ORDER_QH_v1b.md); engine_massfluct_v3 (named in route2_table.md) | WT-busi/code/route2/ (route2_exchange.py, route2_variance.py, route2_dressing_gmres.py, run_route2_v1.sh); table WT-busi/results/route2_table.md (P-R2-2 rows "ENTERS (two routes agree)", box <internal>/route2) | Same star family record star_m0.1_g0.25_ell8.json (route2_table.md:3); branch witness on one route (paper l.1811). |
| Curvature | WT-busi/code/readings/curvature_v1/ (cv_fields.py, cv_rays.py, cv_compare.py, cv_riemann_sympy.py); reading DEL/lead_gaps/FableAPI_reading_CURVATURE_readout_v1.md (+ _scripts) | WT-solveB code run_CURVATURE_secondroute_v1.py, curvature_core_v1.py; ARCH data/solve_b/results/CURVATURE_secondroute_v1/ (summaries.json md5 a5288231ae42ae3ae58fed923a8880d9; per-member P64_s2, P64_s4, P64_s4_tau2, P64_s4_tau4, P64_s8, prod_r11, B2) | Same stored field records. |
| Packet continuation | WT-busi/code/readings/packet_v1/ (packet_v1_propagate.py, run_packet_v1.sh) | WT-solveB code run_PACKET_secondroute_v1.py, packet_core_v1.py; ARCH data/solve_b/results/PACKET_secondroute_v1/ (result_bindings.json md5 b46b8240684fa7ea7f4e415cf833aecf) | Same stored back-averaged fields 12.125-40. |
| First-law / evolution residuals | WT-solve/code/firstlaw, results firstlaw, firstlaw_v2 | ARCH data/hreduce/results/firstlaw_records_v2, _v2p1 (tex_build.json only); WT-busi/code/route2/route2_firstlaw_eps.py | Same records. Producer function not pinned. |
| Light excess and collar flux | PAGE v4.1 chain (4.1, Fig. 2) | PAGE_secondroute_v1 (4.1, Fig. 2); comparison page_balance_compare_v1 | Same v6o_B2 exports; R2 does not regenerate covariance (SECONDROUTE_PAGE_excess_v1.md:5). Paper l.2153 states that sum is on one route. |
| Held drains | page_history_v6s.py, <internal>/v6s_B2hold (summary md5 817c3375...); DEL/lead_gaps/FableAPI_reading_B2_hold_drain_v1.md | NOT FOUND | - |
| Hawking comparator rates (luminosity eq., conformal companion; Patch 5) | C_0 = 0.00523731547127, D_0 = 0.107876529936 from page_curve_amps_R1 (PAGE_FIGURE_R1_v1.md table); paper Eq. sc-luminosity l.2093 "L = 0.005237, on two routes", "S_marg dot = 0.1079" | page_curve_amps_R2 (C_0 0.00523731547128, D_0 0.107876529936; PAGE_FIGURE_COMPARE_v1.md) | Shared signed spec only. The producer of the paper's n-dot = 0.01554 was not located; conformal-companion rate producer NOT FOUND. |
| Feedback | busi compC two-field engine (WT-busi/code/compC/engine_twofield_v*.py, build_engine_twofield_v9c*.py, C_readout_v3c.py); report DEL/claude_busi/C_sigma4_coupling_effect_v1.2.md | Paper l.2280 says "on two routes". DEL/claude_busi/C_sigma4_coupling_effect_v1.2.md:47 says of the other lane's reading: "icloud's derivation ... reproduces no number (its own statement), so nothing below is a second numerical route". Second numerical route NOT FOUND; paper and record disagree (reported, not resolved). | - |
| Cross-root transfer (Eq. B-transfer certified gap) | DEL/lead_gaps/FableAPI_reading_COMP2_W7_v1.md (+ _RECOMPUTE) | ARCH data/hreduce/results/second_route/W7_certificate_v1/w7_certificate_result.json (md5 0128b4a6d5fe9db83266ce23faefd7a6; inputs_early.npz, inputs_late.npz); data/solve_b/results/W7_bound_secondroute_v1/ | Same stored columns (column_operands.json). Mapping of "cross-root transfer" to W7 is by name only (not verified). |
| Evolved trapping cut (exterior entropy 109-311 nats) | WT-busi/code/readings/evolvedcut_v1/ (ec_recompute.py, ec_spectra.py, run_box.sh) | WT-solveB run_EVOLVEDCUT_secondroute_v1.py; ARCH data/solve_b/results/EVOLVEDCUT_secondroute_v1/ (624 files), EVOLVEDCUT_purity_bound_v1; comparison DEL/lead_gaps/FableAPI_reading_EVOLVEDCUT_route_compare_v1.md | Same stored slices. |
| Conversion fits (held-geometry Fermi fits, peeling rates) | NOT FOUND as a producer; the numbers 1.00 to 1.03 / 0.89 to 0.92 appear in staged section drafts (DEL/claude_z/ITEM7_page_curve_*.tex) and DEL/claude_z/FABLE_integration_reading_step2_v1.md:1916, which quotes different values for "Fermi fits on two held records" (T_fit/T_H = 1.26 to 1.30 and 0.85 to 0.88). Candidate producer family: J3P raymap (DEL/claude_alt_solve/J3P_raymap_v5_c19_rs64_*.md), not verified. | NOT FOUND (paper l.2339 says one route) | - |
| Flat-area coefficients and interval enclosures | hreduce: ARCH data/hreduce/results/sphere_directional/eta111_certificate.json (original_md5 eb0cd966f9fe3c302d2111b31be8ea67), massive_coefficient_summary.json (a46ea5c392dca1f78fb6cc0cf7e6b014); eta_massive_scalar_v1/p111_result.json (f1c50aef7cb737646f7eb992e6ea4bcd); code ARCH fermion_new_codes/hreduce/results/sphere_directional, eta_massive_scalar_v1 | Second numerical route "the exact [111] integral and the convex bounds" was moved to the busi lane (WT-solve/RUN_RECORD.md, inbox #2 line); its output not located | Four eta_massive_scalar_v1 files over 10 MB are omitted (SKIPPED_LARGE.tsv rows 2-5). |
| Continuum maximum-mass constants (c_E, c_R) = (0.135858, 1.18689) | J1 continuum family: WT-solve/code/strand5/strand5_star_v3.py (md5 b19b7304e75578102ceab033ee210939), record WT-solve/results/box_records/s5_star/J1run2/star_v3.json (md5 c90214324f1b2cd9c7fec46b1a736ca8; meta g 16, G = m = 1, 60 members z_c in [0.05, 3], grids 2000,4000,8000); stated maximum "z_c 0.833854, M 0.13585788, R_s 1.1868907, 2GM/R_s 0.228931" (WT-solve/results/RESULT_strand5_family.md:98); refined estimate 0.135858102 (WT-solve/results/strand5/J4items2/RESULT_J4_items12_v1.md:51, producer code/strand5/strand5_J4_items12_v1.py md5 76e9822ff134381ae2e117038317bdaa). A v4 rerun exists (launch_strand5_J1_v3.sh -> strand5_star_v4.py md5 e76638b9084972a1ddeec8a9fd7e8bec, J1run3). | Paper's check: "two-state Oppenheimer-Volkoff values over sqrt8" (05_star.tex); no executed producer of that check located | J1 record not in ARCH data. |
| Conditional-figure arithmetic | page_curve_amps R1 (above) | page_curve_amps R2 (above) | Signed spec. Not in ARCH. |

### 4.3 Confirmation of the original_md5 strings quoted in pfix1 sec. 5 (Patch 2)

All seven match ARCH origin/main:data/MANIFEST.tsv exactly (size, original_md5, neutralized):
busi/redshift/registry_v1.json 27828 efcfad78bb6e2c942f21cd4ef9028270 1; busi/perf_host/LINEAGES_h2default_v1.tsv 1564 fc7e1573f63762ff228b4d1b67721824 0; solve_b/results/LADDERS_v1p3_arithmetic_v1/scalar_records/P64_s4.json 718405 90f6ad73da2d997472f73d643d4d1fde 0; solve_b/results/PAGE_secondroute_v1/stored_identity.json 1716 44190ee612906414e5e642f83da954a7 1; solve_b/results/PAGE_secondroute_v1/spectral_entropy.json 8882205 d9b6056712e38f0131660626b7b53b5d 1; hreduce/results/sphere_directional/eta111_certificate.json 385264 eb0cd966f9fe3c302d2111b31be8ea67 0; hreduce/results/eta_massive_scalar_v1/p111_result.json 27665 f1c50aef7cb737646f7eb992e6ea4bcd 1. The local registry_v1.json (WT-busi/code/redshift/registry_v1.json) has md5 efcfad78... (the original, un-neutralized bytes).

Archived code differs from the executed code by neutralization: e.g. ARCH fermion_new_codes/busi/figures/figures_data_v1.py md5 8acffbdaf1d6f0b30f1a015d89fc21f3 vs executed e2469b43...; fermion_new_codes/MANIFEST.tsv records original_md5 e2469b43bc4547c0e695f2e7a1eb6fb8, neutralized 1 (diff: input paths replaced by "<internal>/<name>", plan name replaced). Same for figure_item5_collapse_v3_1.py (ARCH md5 563b07ce..., original 8eab4a5f...) and figure_page_v5.py (ARCH 6716f705..., original 6a8e6283...).

### Status: PARTIAL

Found: all three figure PDF hashes; full producer chain, scripts with md5/sha256, input files with md5 and payload selectors for Figs. 1 and 2 and for the page-curve figure (both routes); the seven quoted manifest md5s. Gaps: (1) no recorded command lines for figures_data_v1.py and the two plot scripts (arguments reconstructed from the data script's log and the usage strings); (2) per-family producer functions and launch lines are named at script/directory level only, not at function level, for first-law, core, curvature, packet, evolved cut, transfer; (3) NOT FOUND: second route for held drains; second numerical route for feedback (paper says two routes, the busi report says none); producers for conversion fits, for n-dot = 0.01554 and the conformal-companion rate; the executed sqrt8 OV check; (4) ARCH lacks the three PDFs, the figure npz files, the page_curve_amps R1/R2 code, the rz_* redshift scripts and all solve_a run records. Looked in: <internal>/.wt-fermion-minimal (git), WT-busi (STATUS.md, FIGURES_SOURCES_v1.md, code/, results/), WT-solve (code/, results/, RUN_RECORD.md, STATUS.md), WT-solveB (code/, results/), DEL (claude_busi, claude_alt_solve, codex_solve, route2, claude_z, lead_gaps), ARCH origin/main (data/MANIFEST.tsv, fermion_new_codes/MANIFEST.tsv, ls-tree), box <internal>/figures_v1, <internal>/ISO64_bridge_v1, <internal>/page_v4, <internal>/v1_3, <internal>/page_fig_r1, <internal>/v6o_B2.

## Marker 5

> [VALUE NEEDED: complete event selection and receiver-frequency normalization for the headline redshifts, with integration tolerances and certified-window masks; obtain from data/busi/redshift/registry_v1.json, its referenced ray records and the executed raymap producer.]

Sources:
- Report: <internal>/REDSHIFT_received_v3.1.md. The paper's mapping to it is in <internal>/SOURCES.md.
- Producers are in …/REDSHIFT_received_v3_scripts/code/:
  - rz_rays_v4.py (94a08d9f…) and rz_rays_v4m.py (199a43a0…) for the ray passes.
  - rz_v3_head_v1.py (418a5b8f…) for the headline tables.
  - rz_v3_reint_v3.py (ba008023…) for re-integration.
  - rz_v3_dispev_v1.py (000ae638…) for the dispersive rays.
- Headline table file: out/v3_head_v1/head_tables.md, md5 d26628e8f4f0a5864500de476b412d66 (REDSHIFT_received_v3.1.md:91).
- Registry: box <internal>/registry_v1.json, md5 efcfad78bb6e2c942f21cd4ef9028270. This equals the archive's MANIFEST original_md5 for data/busi/redshift/registry_v1.json.
  - The archived copy is neutralized: its sha256 is 9492a01f… against 4a5f9c6d… for the box/local copy, and the diff shows paths replaced by `<internal>/`.

Event selection (rz_rays_v4.py):
- **A1 clocks** are comoving test clocks released at T = 0 from r_0 = 0.5, 1, 1.5, 2, 3, 4 r_s (:481-482). dr/dt = v and dτ/dt = N, integrated by RK4 on the h grid (:153-197).
  - Emission is every τ/4 (stride = round(0.25 τ/h)) up to T_birth + 4 r_s and every τ thereafter, plus every h within 1 r_s of a crossing of the outer root (:497-506).
- **A2** clocks are released from 1 r_s at T_rel = 14, 18, …, 160 r_s (:484-486). The paper does not print them as results (SOURCES.md, conventions line).
- **B launches** are made at every record from T_birth - 1 r_s (every 4th record past 40 r_s) at r_e = R_out(T_e) + δ (:512-529).
  - δ takes 24 log-spaced values from 1e-3 a to 2 r_s outward and -0.5, -1, -2, -4, -8, -16 a inward.
  - Before birth the minimum of f is used. No launch is made at the last record.
- **C emitters** are the dust R_50 and R_90 (:530-541). They are not in the paper's headline.
- The headline is the largest L_T over each class's arrivals at R_D = 4 r_s inside the certified interval, with L_D of the same event (rz_v3_head_v1.py:8-9, Table H1 caption at REDSHIFT_received_v3.1.md:93).
- Masks (rz_v3_head_v1.py:10-12):
  - (e) r_e ≥ 1 a.
  - (c) the clock has not entered the first site (|r| < 1 a) before emission.
  - (r) the ray's smallest |r| ≥ 1 a (rz_rays_v4m).
  - "resolved" means all masks that apply.
- Headline events (Table H2, REDSHIFT_received_v3.1.md:116 onward):
  - P64_s4, A1 r_e ≥ 1a: clock 5 (r_0 = 4 r_s); T_e = 10.5356 r_s; τ_e = 9.5445 r_s; r_e = 19.057 a; T_D = 40.0000 r_s; I = 9.910; ln A_D = -0.1172; post-gate share 0.959; pre-birth yes. L_T/L_D = 10.456/10.338.
  - P64_s2, A1: T_e = 10.4793; r_e = 18.742 a; T_D = 40.0000; L_T/L_D = 15.676/15.569.
  - B maxima (L_D): P64_s8 3.582; P64_s4_tau4 6.469; S4_128_s8 7.601; S4_128_s16 3.527. On the completions (A1, A2 and B population): 3.681 (held-source) and 4.111 (widened) (Table H1; SOURCES.md).

Receiver normalization:
- L_T = ln(1/G_D), where G_D = dτ_e/dT_D is the coordinate-clock ratio.
- L_D = ln(1/g_D) = L_T + ln A_D, with A_D = sqrt(N_D² - v_D²) at the static receiver at R_D = 4 r_s (rz_v3_head_v1.py:8; Table H1 caption).
- Fronts: dr/dt = f = N+v and dψ/dt = -∂_r f, so Z = f(r_c, t) e^{-∫∂_r f dt} (rz_rays_v4.py docstring :3-8). The emitter Doppler factor is D_e = 1 for normal-comoving emitters.
- The receiver set is 2, 4, 8, 12, 16 r_s plus r_c = clip(max r_vac over the certified window, 4, 16) r_s (:472-477). The headline is read at 4 r_s.

Integration tolerances:
- Production: fixed-step RK4, h = 0.5 a, steps on record boundaries (:438, :454-455).
- Ladder: h/2 and h/4. A1 differences are ≤ 9.06e-09 and B ≤ 9.07e-09; A2 is first order (the early-release artifact) (REDSHIFT_received_v3.1.md:20).
- Independent integrator: DOP853, rtol 1e-11, atol 1e-12, restarted at every record (rz_v3_reint_v3.py:13, :72-73). Against h/4 it agrees to 5.96e-08 (A1) and 1.01e-07 (B), with 106 events and 0 capped (REDSHIFT_received_v3.1.md:20).
- Dispersive rays: DOP853 at the same rtol and atol (rz_v3_dispev_v1.py:61). The continuum re-integration from the headline events reproduces the stored L_T to 2.81e-05 (REDSHIFT_received_v3.1.md, response A3-7.2).
- Field-error envelope: eps_c = 2·max(kept residual, 1e-12) per record (rz_rays_v4.py:479).

Certified-window masks:
- The arrival must be at or before the member's certified record: "inside the solver-certified record interval" (rz_v3_head_v1.py:14; T_cert at rz_rays_v4.py:467-470).
- Certified extents come from members_v2.py MEMBERS, citing LADDER_certification_list_v1.2:
  - P64_s4, P64_s2, P64_s8, P64_s4_tau4, S4_128_s8, S4_128_s16: 40.0.
  - P64_s4_tau2: 12.75. P64_s4_newton: 12.5. S4_128_s4: 3.5625. ISO_128_s4: 12.0. ISO_128_s4_plain: 3.25. M3c_s4_96: 22.0833.
  - The completions P64_s4_heldsrc and P64_s4_widened use "rule": the first record whose kept residual max|applied - stored| > 1e-3 ends the extent (members_v3.py; rz_rays_v4.py:76-80).
- The one-term gate is ell·max_r|∂_r v_bar| > 0.1. Its first record sets T_gate, which is used for the post-gate shares (rz_rays_v4.py:464-466).

### Paper against code (marker 5)

- The paper defines the received fraction "at infinity", g = ω_∞/ω_em (05_star.tex, received-map paragraph), and places the receiver at 4 r_s. The printed numbers are the static-receiver L_D at 4 r_s, which includes ln A_D ≈ -0.117 at the reference headline. No infinity transfer is applied: the exterior is not appended in the ray code.
- The paper's one-term gate compares σ|H| "out to the dust-energy centroid" (05_star.tex method paragraph). The ray code's gate (used only for post-gate shares) takes the maximum over the whole chain (rz_rays_v4.py:464).
- The paper's "3x10^4 … emitted at 0.30 r_s before the first root formed" corresponds to r_e = 19.057 a ≈ 0.298 r_s, emitted pre-birth at T_e = 10.5356 against the first root at 12.125 (Table H2; SOURCES.md row 4). Here 0.30 r_s is a radius, not a time interval.
- In the archive, registry_v1.json is present but neutralized. None of the producing rz_* scripts, members_v2/v3.py or the ray-pass outputs are archived.

Status: FOUND in the redshift lane's scripts and report. The producing scripts are not in the companion archive, and the archived registry_v1.json has its paths replaced by `<internal>/…`.

## Marker 6

> [VALUE NEEDED: complete initial occupied-column construction for the principal collapse and its five-column and isotropic preparations, including radial envelopes, phases, projections, orthogonalization, sea occupations, copy multiplicities and seeds or an explicit deterministic statement; extract the executed preparation function and launch arguments through the lineage of engine_collapse_v36d.py and its actual successors, and the initial covariance/column payloads named by that lineage.]

Extended by FableAPI_pfix2_v1 Patch 3 to the ladder members r_s/a = 32, 128, 256, 326; zeta = 0.4, 0.1, 0.05; the counts released from 2r_s; and the E_0/E_max = 1.14 members.

Path abbreviations used below:
- SA = <internal>/solve (the solve_a working tree)
- BU = <internal>/busi (the busi working tree)
- AR = the companion archive, <internal>/entanglement-stars at origin/main f690b45 (the local working tree is checked out on branch sync-corrected-observable 3a1f962, which has no fermion_new_codes/; archive files were read with `git show origin/main:<path>`)
- BOX = 192.168.1.93

### 6.1 Principal run identity and lineage (P64_s4, the reference collapse)

- Two-segment history, both executed with engine_collapse_v36d.py:
  - Segment 0 to 18.125: tag c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0, fresh preparation (resume_slice null, resume_from null). Run json BOX <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json (md5 9ad6b3fedd9343438f3b1f33758ea483 in BOX <internal>/MANIFEST_pod2_engine.md5); 146 records, last T/r_s 18.125. Log md5 4430123e4e493306bea9afccbbcd9b62 (same manifest). Launched 2026-10-18T06:01:06Z on pod 2, PID 1619004 (BOX .../pod_sync/pod2_engine/v36d_from0/status.txt; an earlier 05:58:11Z launch ended at the argument check, log ...v1_argerror.log).
  - Segment 18 to 40: tag ..._v36d_from0_to40, resumed by --resume-slice from orbitals_..._T0018.00.npz (md5 5e44bd04675d35c666bbe6e61606247b) on pod 7 (BOX <internal>/status.txt). Run json md5 d7d69adcea7a43f8375ab4c7c5a80efc (BOX <internal>/MANIFEST_runs_pod7_ell4.md5); 321 records 0 to 40 (records 0 to 18 replayed from segment 0).
- Launch scripts: SA/code/engine/launch_v36d_from0_rungs_pod2_v2.sh:15-21 (segment 0); SA/code/engine/launch_v36d_from0_continue_pod7_v2.sh:13-14, 41 (segment 18 to 40).
- Executed engine: engine_collapse_v36d.py md5 0315e0826ee872ee194474876ef523f9 (SA/code/engine/engine_collapse_v36d.py; same md5 on BOX <internal>/engine_collapse_v36d.py; the AR file fermion_new_codes/solve_a/engine/engine_collapse_v36d.py is the neutralized copy, git-blob md5 b4f191a189f8c99adbeebc3544fd8787, listed in AR fermion_new_codes/MANIFEST.tsv with original_md5 0315e0826ee872ee194474876ef523f9, neutralized 1; neutralization changed docstring words only, line numbering is identical, 3342 lines). STATUS records the pod-2 code tree as "md5-equal to pod 4's" (SA/STATUS.md:1598).
- Code/record disagreement: the run json's "script" field reads "engine_collapse_v34.py" for both segments, but the launched file is engine_collapse_v36d.py. The string is hard-coded in v36d at SA/code/engine/engine_collapse_v36d.py:2622, 2644, 2666.
- Segment-0 launch arguments (from the run json "args", BOX path above; and launch_v36d_from0_rungs_pod2_v2.sh:16): m 0.2, g 0.25, rs 64, R0_over_rs 4.0, reading A, shell null, nsites 2048, ell 4, prep_ell 4, kappa_sea 10, tau 8.0, T_over_rs 18.125, reference R2, half_offset 0, site_fields averaged, field_rtol 1e-12, field_atol 1e-16, stop_E 1e-07, stop_fields 1e-06, maxit 20, mix plain, c_LZ 1.0, window_max 4, massless_ball false, fields_v17 false, field_kernel_cache true, v36_fixes true, gpu 0, slices_c128 "1,...,18".
- Initial-data head (identical in both segment jsons): N_B 640.0, passes 15, last_change 9.05658326111336e-11, E0 118.13786225596981 (2gE0 = 59.069 a; paper 05_star.tex:73 "2GE_0=59.07a"), f_kin 0.10627830322099649, f_kin_bulk_frame 0.015889930183391382, dust_momentum_total 55.776952492218626, projection {discarded_norm_max 8.720139503815005e-07, gram_cond_max 1.0000011908931403, stationarity_max 0.0003705970712787891}, residual_W 7.19e-11, residual_invN 7.43e-11, interp_floor_N 4.25e-09, interp_floor_v 3.27e-09, R0 256.0 (a), prep_ell 4, evolution_ell 4, kernel "the Gaussian of standard deviation ell (per-axis variance sigma^2 = ell^2, plus a^2/12 where the cell average enters); prepared at ell 4, evolved at ell 4", preparation 79.65 s (BOX .../runs_pod7_ell4/collapse_..._to40.json "head"; same line printed in the segment-0 log, line 4).

### 6.2 Executed preparation function (principal, step and width controls, all well-filled ladder members)

Call path: run() at SA/code/engine/engine_collapse_v36d.py:1997; chains built at :2014 as `chs = [Chan(k, a.m, a.nsites, bool(a.half_offset)) for k in range(1, a.kappa_sea + 1)]`; with no resume and no --massless-ball the preparation is `prepare_dust(chs, a.g, R, prep_ell, NB, a.R0_over_rs * rs, a.m, a.reading, shell)` at :2042, NB = rs/(2 g m) at :2013, prep_ell = --prep-ell or --ell at :2034. prepare_dust is defined at :475-544 (same lines in AR fermion_new_codes/solve_a/engine/engine_collapse_v36d.py). Helpers are imported from engine_collapse_v5.py (v36d:385-394): T_, eigT, dens, Chan, logistic, fill_lowest (definitions at SA/code/engine/engine_collapse_v5.py:76, 92, 96, 115, 120, 131; same lines in AR). The chain operator is engine_vacflow_v1.chain_sparse (SA/code/engine/engine_vacflow_v1.py:53-73).

As implemented, step by step:
1. Chains: channel kappa = 1..kappa_sea (10 for the principal); sites R_n = n, n = 1..nsites (half_offset 0) (engine_collapse_v5.py:99-102); two components per site, dense 2*nsites matrices in complex128 (engine_collapse_v5.py:92-93, 104-106). h_0: on-site m sigma_z + (kappa/R) sigma_y, bond -(i/2) sigma_x to the next site, open ends; Q the two-step operator with -(i/4) (engine_vacflow_v1.py:54-72). Clocked operator h_m[N,v] = (1/2){N,h_0} + (1/2){v,Q} (engine_collapse_v5.py:109-113).
2. Copy multiplicity: each chain carries weight w = 16 kappa (engine_collapse_v5.py:101); every dust column's occupation f applies to all 16 kappa copies.
3. Start fields N = 1, v = 0 (v36d:479). Self-consistency loop up to passes = 200, tol 1e-10 on max |change| of N or v (v36d:475, 481, 533-534), Anderson mixing engine_mixing_v1.Anderson() with defaults depth 6, beta 0.5, alpha_linear 0.3 and a reset rule when the residual exceeds 2x the best (SA/code/engine/engine_mixing_v1.py:11-32; v36d:480, 531).
4. Radial envelope (the "levitating well"): U0 = m/2 (v36d:481); reading A: N_s = sqrt(max(N^2 - v^2, 1e-12)) (v36d:483); U_w(r) = U0 s(r) + m (1 - N_s(r)) (1 - s(r)), s = 1/(1 + exp(-(R - R0)/4)), a logistic wall at R0 = R0_over_rs * r_s of width 4 a (v36d:487-489; engine_collapse_v5.py:76-77). R0 = 256 a for the principal.
5. Per channel: hw = h_m[N, v] + U_w (reading A keeps v) and its dense eigendecomposition with torch.linalg.eigh; separately the eigendecomposition of h_m[N, v]; the positive-eigenvalue eigenvectors of h_m[N, v] form P+ (v36d:492-498). Level list = the first 400 positive eigenvalues of hw per channel, each tagged with the chain weight w (v36d:499).
6. Filling: fill_lowest(levels, N_B) sorts all levels of all channels, groups levels within 1e-9, fills groups in order with f = min(1, left/w_group) so the last group is fractional and uniform over its copies, until the weighted count reaches N_B (engine_collapse_v5.py:131-145; v36d:500).
7. Projection and orthogonalization: for the kept columns of each channel raw = P+ P+^dagger Uwv[:, keep] (projection on the positive band of h_m[N, v]); Loewdin orthonormalization Phi = raw S^{-1/2}, S = raw^dagger raw, via torch.linalg.eigh(S) (v36d:504-517). Diagnostics recorded: discarded norm, Gram condition number, stationarity residual of Phi against hw (v36d:510-516).
8. Sources: eps += w dens(h, Phi, f), J += w dens(Q, Phi, f); fields solved by solve() (engine_fields_v16.solve_constraints_ivp with rtol/atol from --field-rtol/--field-atol, then kernel averaging when site_fields = averaged) at ell = prep_ell (v36d:401-412, 518, 521; OPT and SOLVE_TOL set from the arguments at v36d:3325-3326). After the loop one more solve from the last sources (v36d:535-537). E0 = FLD.charge(info, g) (v36d:545).
9. Phases: none fixed in the code. Column phases are those returned by torch.linalg.eigh (v36d:494; engine_collapse_v5.py:115-117), then mixed by the Loewdin transform.
10. Sea occupation and reference at T = 0: after the preparation, per channel the clocked h_m[N, v] at the final prepared fields is diagonalized and every negative-eigenvalue eigenvector is occupied (f = 1): st["sea"] = U[:, neg]; the R2 reference starts equal to that sea, ref0 = U[:, neg] (v36d:2063-2073). The dust columns Phi with occupations f are stored beside it (st["dust"], st["f"]). No explicit orthogonalization of the dust against the sea beyond the positive-band projection of step 7 appears in the code. n_sea = 2048 per channel (= nsites) at the principal's T 18 resume check (to40 json head.resume_slice_v34.channels, "n_sea": 2048 in every channel).
11. Seeds and randomness: no random-number call (np.random, torch.rand*, default_rng, manual_seed, random.) occurs anywhere in the import closure of engine_collapse_v36d.py (engine_collapse_v36d, v17, v5, engine_fields_v3, v4, v12, v16, v17, engine_fields_kcache_v1, engine_mixing_v1, engine_ops_v1, v2, v3, engine_ops_v34_cert, engine_pairsolve_v1, engine_stress_v2, engine_vacflow_v1), checked by text search of SA/code/engine/. The preparation is therefore deterministic in its code; no statement on GPU-library run-to-run bitwise reproducibility was found. The only "seed" in v36d is the unused --massless-ball chiral Slater seeds (v36d:1671-1853), off for every run here (massless_ball false in the args).

Dust columns of the principal (from the slice check at T 18, head.resume_slice_v34.channels[k].n_ball of the to40 json; the column count is not changed by the propagation): kappa 1: 5, 2: 4, 3: 3, 4: 2, 5: 2, 6 to 10: 0. Weighted count 16 x (1x5 + 2x4 + 3x3 + 4x2 + 5x2) = 640 = N_B, which with f <= 1 forces f = 1 on every dust column (arithmetic from the stored counts; the f arrays themselves are in the slices, not read). Paper 05_star.tex:73-77 "the dust filling k=1 to 5" agrees. The ladder tables define the "own onset" as 4 a times the largest chain column count, "20 a for the production's set" (SA/results/ladders/SCALING_ladder_v1.3.md:64), i.e. the five columns of kappa = 1; this is the "five-column" set of the paper (05_star.tex:210; flat paper line 1085).

Identity of the preparation across engine versions: the body of prepare_dust is byte-identical (apart from the one head label naming the fields module) in engine_collapse_v17.py:239, v19:272, v25:288, v31:326, v32:358, v34:413, v36d:475 (SA/code/engine/), and in the executed copies v36e:491 and v36j:483 (BOX <internal>/code_base, md5 71b92f61648681191c9aae9e7ec7a71b and f16010800331a8e866e03a45db547e77). The only difference v17 to v36d is the label "engine_fields_v12" vs "engine_fields_v16" in the head (diff of the two functions).

### 6.3 The c19 production record (Fig. 1 a,b fields) and its records 0 to 11

- c19: BOX <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json (md5 e27e457d37615cff8d5216f05a494f89 per the ladder readout meta); script engine_collapse_v19.py; args m 0.2, g 0.25, rs 64, R0_over_rs 4, reading A, nsites 2048, ell 4, kappa_sea 10, tau 8, reference R2, site_fields averaged, mix plain, maxit 20, stop_E 1e-07, stop_fields 1e-06, c_LZ 1.0, window_max 4, resume_from "C3prod/c17_rs64_A_R2_tau8_ell4_avg", resume_T_over_rs 11.0 (no prep_ell argument: prepared at ell 4).
- Records 0 to 11.0 (89 records) are replayed from c17: head.resume = {from C3prod/c17_rs64_A_R2_tau8_ell4_avg, script engine_collapse_v17.py, T_over_rs 11.0, records_replayed 89, max_eps_rel 3.70e-14, max_J_rel 5.46e-14, cinit dN 1.15e-14, dv 2.55e-15}. The replay re-prepares the initial data with engine_collapse_v17.prepare_dust (resume_state, SA/code/engine/engine_collapse_v36d.py:1467-1482 shows the same mechanism in the later engine).
- c17: BOX <internal>/collapse_c17_rs64_A_R2_tau8_ell4_avg.json (md5 5e4abc36e766241f72e4dc53209cb155; fields npz md5 19473f40da5e058915ea881f31fac77c); script engine_collapse_v17.py (local md5 8fdca1df50817c9f885e64c17cbddc50 = AR MANIFEST original_md5), fresh preparation (head.resume null), args as c19 but maxit 30, verify_every 10, no c_LZ/window_max keys.
- c17 and c19 head: N_B 640.0, passes 15, E0 118.13786225597009, projection {8.720139499374113e-07, 1.0000011908931377, 0.00037059707127871044}. The v36d preparation's E0 118.13786225596981 differs in the 14th significant digit.

### 6.4 Isotropic preparation (ISO_64_s4, isoB; ISO_128_s4)

- Engine: engine_collapse_v36n.py, built by BU/code/iso/build_engine_collapse_v36n.py (AR fermion_new_codes/busi/iso/build_engine_collapse_v36n.py) as marked patches over engine_collapse_v36j.py; executed copy BOX <internal>/engine_collapse_v36n.py md5 0c84a4466cd50a57621526faade1b2d3 (prepare_dust at :775-850). Neither engine_collapse_v36j.py nor engine_collapse_v36n.py is in AR (only the builder and engine_collapse_v36n_selftest.py).
- Option: --dust-set isotropic [--dust-kappa-max K_d] [--dust-multiplicity 1|2] (builder docstring, BU/code/iso/build_engine_collapse_v36n.py:30-55). As executed (v36n:474-535): once per preparation, per channel, the positive levels at or above m of the free chain h_kappa (N = 1, v = 0, no well) on the cavity sites R <= R_0 (hard lattice wall) are the ordering keys (iso_begin :474-498); in fill_lowest the well's level (kappa, j) is ordered by the cavity's j-th level (iso_key :500-506); weights multiplicity x 16 kappa, f in [0,1] (iso_levels :514-530); the orbitals remain the well's eigenvectors Uwv[:, j], projected on the positive band of h_m[N, v] and Loewdin-orthonormalized as in 6.2; everything after the preparation unchanged.
- ISO_64_s4_plain (original plain run): BOX <internal>/collapse_lad_ISO_64_s4_..._plain_v36n.json, md5 fde06ed72e8ec1d22759f576186af5aa. args: nsites 2048, ell 4, prep_ell 4, kappa_sea 10, tau 8, mix plain, dust_set isotropic, dust_kappa_max null, dust_multiplicity 1, resume_slice null. Head: E0 118.21171894186747, N_B 640.0, passes 15, projection {8.504105033502185e-07, 1.0000009374957721, 0.0003692490817428166}; catalog: multiplicity 1, g_per_momentum 8, weights "16 kappa x f, f <= 1", cavity_sites 256, R0 256.0, top_channel 6, E_F 0.20265736477149743, k_F_R0_catalog 8.374, kappa_max_rule 10.419; columns per channel kappa 1: 4 (j 0-3), 2: 4, 3: 3, 4: 2, 5: 1, 6: 1, all f = 1.0 (particles 64, 128, 144, 128, 80, 96; sum 640).
- ISO_64_s4 (the composed isoB history used as the comparator): BOX <internal>/collapse_isoB.json md5 227cd66d25ffc650ffee69eb0da6270c; the same preparation head (E0 118.21171894186747); resumed by resume_slice .../ISO_64_s4_nk_ep/states_J2/orbitals_..._newton_v36n_T0014.50.npz; label in the ladder readout "plain v36n to 13.5, Newton episode 13.625 to 14.5, plain to 40".
- ISO_128_s4 (r_s/a 128, sigma/a 4): last segment launched 2026-10-21T13:23:08Z with engine_collapse_v36nq_rd2.py md5 186a4ccd580365dc97f437672ad63096, engine_fields_v16.py md5 7a69faadfad4a1e824a41107f474c69c, args file md5 5b0f81437fea84a5243ac7eb30704c61 (BOX <internal>/LAUNCH.txt, args_v1.txt): rs 128, nsites 4096, ell 4, prep_ell 4, tau 8, kappa_sea 10, dust_set isotropic, mix newton, resume from the h2 run's T0005.00 slice. The h2 run (BOX <internal>/collapse_lad_ISO_128_s4_nkqh2_..._newton_v36nq.json) resumed from .../iso128_nk/states/orbitals_lad_ISO_128_s4_nkq_..._T0003.00.npz; its head: N_B 1280.0, E0 235.17137214931623, top_channel 8, columns_kappa_1 6. The T = 0 run of this lineage (ISO_128_s4_nkq) was not located on the box; PARTIAL.

### 6.5 Ladder members (Patch 3 extension)

Preparation rule stated by the ladder tables: "Every member and control shares one preparation rule: the dust is the levitating well's zero-temperature filling within the sea set kappa = 1 to kappa_sea (engine prepare_dust with fill_lowest ...), released from coordinate rest at R_0 = 4 r_s (2 r_s for the 2 preparation members), prepared at ell 4 and evolved at the member's ell. No member runs the isotropic catalog" (SA/results/ladders/SCALING_ladder_v1.3.md:64; MASS_ladder_v1.3.md:52). The prepare_dust body in every engine named below is the one in 6.2 (identity checked for v19, v25, v31, v34, v36d, v36e, v36j; see 6.2). All are deterministic in the sense of 6.2 item 11 (v36e, v36j, v36n copies also contain no RNG call).

Launch arguments and preparation heads (args from BOX <internal>/ladders_readout_v1_3.json members[*].meta.args, which copies each run json's args; E0, N_B, E0/E_max and own-onset from SCALING_ladder_v1.3.md:68-86 and MASS_ladder_v1.3.md:56-69):

| member | (zeta, r_s/a) | run json md5 | engine (json "script" / tag) | nsites | kappa_sea | R0/r_s | prep_ell / ell | tau | N_B | E0 | E0/E_max | own onset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RS32_s4 | (0.2, 32) | 898fd8ebe0dd9424de6f20074d04f568 | engine_collapse_v31.py (c31) | 1024 | 10 | 4 | 4 / 4 | 2 | 320 | 59.7890 | 2.20 | 16 a |
| RS32_s2 | (0.2, 32) | d35bc0fd5f1b7765e495668f4300b90c | engine_collapse_v32.py, Newton | 1024 | 10 | 4 | 4 / 2 | 2 | 320 | 59.7890 | 2.20 | 16 a |
| RS32_s4_tau1 | (0.2, 32) | 6e8b8973c771c6a4a01db8926c30a46a | engine_collapse_v25.py | 1024 | 10 | 4 | none (= ell 4) | 1 | 320 | 59.7890 | 2.20 | 16 a |
| S4_128_s4 | (0.2, 128) | 5fe643bb6c3ef225f05e194c38908a6a | v36d (tag ..._plain_v36d) | 4096 | 10 | 4 | 4 / 4 | 8 | 1280 | 234.8342 | 8.64 | 24 a |
| S4_128_s8 | (0.2, 128) | ff475918f57758118221aa59fda52a86 | v36d | 4096 | 10 | 4 | 4 / 8 | 8 | 1280 | 234.8342 | 8.64 | 24 a |
| S4_128_s16 | (0.2, 128) | c2aee55e56201f6a216aa9844dd6bcdb | v36d | 4096 | 10 | 4 | 4 / 16 | 8 | 1280 | 234.8342 | 8.64 | 24 a |
| S4_128_s4_nk_v36m / _v36q | (0.2, 128) | 2889024ae017804f75cec4e152e6c6e2 / ea458c364755815db5584b98cbb4b1aa | v36m / v36q, Newton | 4096 | 10 | 4 | 4 / 4 | 8 | 1280 | 234.8342 | 8.64 | 24 a |
| M3b_s4 | (0.4, 64) | 6209f14cf64e8a4f614babccb28eb04c | v36d | 2048 | 10 | 4 | 4 / 4 | 8 | 320 | 113.6966 | 16.74 | 20 a |
| M3b_s8 | (0.4, 64) | 0b834bc19856a2ae6c9f46ce3bd21445 | v36d | 2048 | 10 | 4 | 4 / 8 | 8 | 320 | 113.6966 | 16.74 | 20 a |
| M3a_s4 | (0.1, 64) | 778dc5cef53012b71461a2e1d2de1044 | v36d | 2048 | 10 | 4 | 4 / 4 | 8 | 1280 | 123.4741 | 1.14 | 24 a |
| M3a_s2 | (0.1, 64) | d8aa172c70456055b6867f0489e3cf96 | v36d | 2048 | 10 | 4 | 4 / 2 | 8 | 1280 | 123.4741 | 1.14 | 24 a |
| M3a_s4_tau4 | (0.1, 64) | 66382686ad07f2ca581659153c90767a | v36d | 2048 | 10 | 4 | 4 / 4 | 4 | 1280 | 123.4741 | 1.14 | 24 a |
| M3c_s4_nk | (0.1, 96) | 60f92fbfae76f621867f543eaad49029 | v36j, Newton | 3072 | 10 | 4 | 4 / 4 | 8 | 1920 | 181.8183 | 1.67 | 28 a |
| M3c_s4_plain | (0.1, 96) | 6489517c34dccd5c536e99a7a03d3bb3 | v36d | 3072 | 10 | 4 | 4 / 4 | 8 | 1920 | 181.8183 | 1.67 | 28 a |
| M3d_s4 | (0.05, 326) | 726cb35fb005e3ef2ea05c3e3adfa38f | v36e | 3912 | 12 | 4 | 4 / 4 | 8 | 13040 | 620.6128 | 1.43 | not available |
| S4_16_s4 | (0.2, 16) | b87134f9742e57c16b0cb3bfd1b2fd9b | v36d | 512 | 10 | 4 | 4 / 4 | 2 | 160 | 30.9668 | 1.14 | 12 a |
| S4_16_s2 | (0.2, 16) | 52f02084c6fcb0ca9690af82b9bb5680 | v36d | 512 | 10 | 4 | 4 / 2 | 2 | 160 | 30.9668 | 1.14 | 12 a |
| S4_16_s4_tau1 | (0.2, 16) | 9689881076f6c34eab7f691327be555b | v36d | 512 | 10 | 4 | 4 / 4 | 1 | 160 | 30.9668 | 1.14 | 12 a |
| S4_R2_m0p2 | (0.2, 64), released at 2 r_s | aae1a09ee3ebdeac0ad5efc726ce5df3 | v36d | 2048 | 10 | 2 | 4 / 4 | 8 | 640 | 109.8332 | 4.04 | 20 a |
| S4_R2_m0p1 | (0.1, 64), released at 2 r_s | d9b428d05430ac7e989fd2f62f92c5a1 | v36d | 2048 | 10 | 2 | 4 / 4 | 8 | 1280 | 123.0561 | 1.13 | 24 a |
| P64_s4_tau4 | (0.2, 64) step control | f2e1d8a7483447426855e3211535a367 | engine_collapse_v19.py (c19 tau4, from T 0) | 2048 | 10 | 4 | none (= ell 4) | 4 | 640 | 118.1379 | 4.35 | 20 a |
| P64_s4_tau2 | (0.2, 64) step control | 9ba6f6ff302d8a903925f20583c96656 | engine_collapse_v25.py (c25) | 2048 | 10 | 4 | none (= ell 4) | 2 | 640 | 118.1379 | 4.35 | 20 a |
| P64_s8 | (0.2, 64) width control | d1ca037a77c198aa83b33055495eceff | v36d (from0, same launch script with --ell 8) | 2048 | 10 | 4 | 4 / 8 | 8 | 640 | 118.1379 | 4.35 | 20 a |
| P64_s2 | (0.2, 64) width control | 0a005ac5f66bef02a0104e4278867765 | composite, Newton (v32 member + v36d from 17.625) | 2048 | 10 | 4 | 4 / 2 | 8 | 640 | 118.1379 | 4.35 | 20 a |

Common to all rows above: m = zeta, g 0.25, reading A, reference R2, site_fields averaged, maxit 20, c_LZ 1.0 (ladder readout args).

- The E_0/E_max = 1.14 members are M3a_s4 (0.1, 64) and S4_16_s4 (0.2, 16), with their controls M3a_s2, M3a_s4_tau4, S4_16_s2, S4_16_s4_tau1 (column "E0/E_max (charge ratio, c_E 0.135858)" = 1.14 at SCALING_ladder_v1.3.md:68, 72, 73, 86 and MASS_ladder_v1.3.md:58, 67, 68). S4_R2_m0p1 reads 1.13.
- The counts released from 2 r_s are S4_R2_m0p2 and S4_R2_m0p1 (R0_over_rs 2.0 in their args; role "preparation:P64_s4" and "preparation:M3a_s4").
- r_s/a = 256 (F256_s8, the member the paper reports at sigma/a = 8): a different preparation, --prep flatbox. Executed args BOX <internal>/args.txt: m 0.2, g 0.25, R0_over_rs 4, reading A, kappa_sea 15, reference R2, site_fields averaged, rs 256, nsites 4096, ell 8, prep_ell 4, tau 16, dust_set well, prep flatbox, prep_edge taper, prep_taper 0.5, mix newton (with the reopen2 Newton block), fields collocation, eigh banded, jvp eigenbasis, slices_c128 from 0.0000; this file is the continuation from seg0's T0000.06 slice (md5 19afad608c6813f316aa609fdf6c9042; seg0 run json md5 93e680925e3653fa6a40324e77ddf52c) (run/RUN_MD5.txt). Executed engine engine_collapse_v36nq_rd13mfb.py md5 02a1729ab808948d443eed9144b75b03 with engine_fields_v16.py md5 7a69faadfad4a1e824a41107f474c69c and engine_banded_eigh_v1.py md5 73ee46facc09b8cb77f095b2803a5c8e (run/RUN_MD5.txt; box copy BOX <internal>/engine_collapse_v36nq_rd13mfb.py has the same md5). Builder: BU/code/gatecomp/flat/build_engine_collapse_v36nq_rd13mfb.py (AR fermion_new_codes/busi/gatecomp/flat/build_engine_collapse_v36nq_rd13mfb.py); the engine itself is not in AR. Construction as executed (engine flat_begin, :750-786): per channel the free chain (N = 1, v = 0) on sites R <= R_wall, plus for the taper U(R) = E_k ((R - R_in)/W)^2 beyond R_in = R_b - W, R_wall = R_b + W, E_k the hard box's Weyl Fermi kinetic energy at R_b on the degeneracy-8 chains; eigenvectors at or above m (at most 400) embedded with zeros beyond R_wall are the orbitals; filled by fill_lowest with weights 16 kappa to N_B; projected each pass on the positive band of h_m[N, v] and Loewdin-orthonormalized; the well's eigendecomposition is not made (builder docstring :12-27). Head (BOX .../F256_s8_end_T5.00/run/collapse_lad_F256_s8_fbt05_..._v36nq_rd13mfb.json): E0 512.7168365781232 (2gE0 256.358 a), N_B 2560.0, passes 5, projection {1.9253085243864376e-06, 1.0000044693863723, 0.0007460032740357177}; flatbox: edge taper, R_b 1024 a, W 128 a, R_in 896 a, R_wall 1152 a, E_k_taper 0.0006510914610996232, box_sites 1152, top_channel 10, sea_top 15, E_F 0.20046839901286362; columns per channel kappa 1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 4 (last f = 0.6667), 7: 2, 8: 2, 9: 2, 10: 1, all other f = 1.0. The builder docstring states that under flatbox "the dust moves with the river ..., not at coordinate rest" (builder :24-25). Other 256 members: F256_s4 retired (SA/STATUS.md:736), W256_s4 run on qi2_h100_1 (SA/STATUS.md:656, 740); their preparations were not extracted.

### 6.6 Initial covariance / column payloads

- Principal: no T = 0 slice was written (slices_c128 "1,...,18"). The first stored state is orbitals_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_T0001.00.npz, md5 6691e45132b127c89a9bba24229536ab (BOX <internal>/MANIFEST_slices.md5); T 18 slice md5 5e44bd04675d35c666bbe6e61606247b. The slice files are not in the box directory listing (only hf_done.txt, md5_done.txt); the mirror log names the destination dataset 'solve/v36d_from0_slices/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0' (BOX .../v36d_from0/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0/mirror_v2_v36d_from0_ell4.log), and SA/STATUS.md:722 and 712 record the v36d_from0 tree's offload to Google Drive and the restore of run records under 1 GB only. The slice writer stores per channel sea_k, dust_k, f_k and R2_k (SA/code/engine/engine_collapse_v36d.py:1975-1979). The T = 0 fields (node values) are in record 0 of the run fields npz (to40 fields npz md5 01444a399b9466047a5b0f118af61f77; segment-0 fields npz md5 d44ceb7dbd401866588823047067133d).
- c17/c19: no slices before T 12 (c19 slices_c128 "12,16,20,24,28", c17 none).
- F256_s8: T0000.00 slice requested in its slice list; seg0 states not checked on the box.
- AR holds none of these payloads: no orbitals/slice file, no principal run json, in data/MANIFEST.tsv or SKIPPED_LARGE.tsv (search for orbitals, T0000, v36d_from0, c19, c17, ISO, F256); AR data holds derived records only (e.g. data/solve_b/results/LADDERS_v1p3_arithmetic_v1/payload_member_bindings/P64_s4.json).

### 6.7 Paper and code/records disagreements (reported, not resolved)

1. Rest condition. Paper 05_star.tex:71-72: "released at rest relative to the lattice at 4r_s". Ladder tables: "released from coordinate rest at R_0 = 4 r_s" (SCALING_ladder_v1.3.md:64). Code: the dust columns are eigenvectors of h_m[N, v] + U_w at the self-consistent fields with v kept (reading A), projected on the positive band of h_m[N, v] (v36d:492-505); the recorded head has dust_momentum_total 55.776952492218626, f_kin 0.1063, f_kin_bulk_frame 0.0159. No explicit rest criterion is implemented in prepare_dust.
2. "five-column" at r_s/a = 128. Paper 05_star.tex:210: "the five-column sigma/a=4 solve failing there" (at 128). The ladder table's own onset for S4_128_s4 is 24 a, i.e. a largest chain column count of 6 (SCALING_ladder_v1.3.md:64 definition; S4_128 rows), while the 64 production's set is 20 a (5 columns).
3. Engine name in the records: run json "script": "engine_collapse_v34.py" for runs executed with engine_collapse_v36d.py (hard-coded, v36d:2622, 2644, 2666); likewise for the v36e/v36j/v36n members (ladder readout meta "script").
4. r_s/a = 256: paper 05_star.tex:224-226 reports formation "at one width, sigma/a=8" without naming its preparation; the executed member F256_s8 uses the tapered flat box (--prep flatbox --prep-edge taper --prep-taper 0.5), kappa_sea 15, tau 16 a, Newton fields with collocation, not the well filling, kappa_sea 10 and tau 8 a of the reference.
5. Multiplicity wording. Paper 05_star.tex:10 gives the continuum family "16 states per momentum" and the chain weight 16k (05_star.tex:47-49); the isotropic dust set as executed is the "degeneracy-8 ball" (catalog g_per_momentum 8, multiplicity 1, f <= 1), with the degeneracy-16 ball available only by --dust-multiplicity 2, which no listed run used (builder docstring BU/code/iso/build_engine_collapse_v36n.py:30-45; ISO run args dust_multiplicity 1).

Note on method: the box json heads in 6.1, 6.4 and 6.5 were read with short read-only python json parsing on the box and with grep; nothing was written there.

Status: PARTIAL. Found: the executed preparation function and every construction step (envelope, filling, multiplicity, projection, Loewdin, sea, reference, no RNG) with line numbers; launch arguments and heads for the principal, c17/c19, ISO_64, F256_s8 and every ladder member listed. Not found: (a) a stored T = 0 covariance or column payload for the principal (no T 0 slice; first slice T 1 offloaded off the box, md5 only); (b) the per-column dust occupations f at T = 0 as stored numbers (inferred f = 1 from counts); (c) the T = 0 run of the ISO_128_s4 lineage and the preparations of W256_s4 and F256_s4; (d) the executed engine files v36e, v36j, v36n, v36nq_rd2, v36nq_rd13mfb in the companion archive. Looked in: SA/code/engine, SA/results/ladders, SA/STATUS.md, SA/RUN_RECORD.md, BU/code/iso, BU/code/gatecomp/flat, AR origin/main (fermion_new_codes, data/MANIFEST.tsv, SKIPPED_LARGE.tsv), BOX <internal>/{v36d_from0, pod_sync, ladders, isoB, iso128_h2, reopen2, compositions/ISO64_bridge_v1}, BOX <internal>/flat, BOX <internal>/{C3prod, C3prod19}.

## Marker 7

> [VALUE NEEDED: entropy and flux payload selectors, cut-site inclusions, held-vacuum construction, eigenvalue endpoint handling, propagation-floor computation, channel weighting and tail/common-mode statistic for each displayed sum; obtain from the actual figure and spectral-analysis functions and their saved input bindings.]

Figure chain (Fig. 2, page_v5.pdf):
- page_v5.pdf has md5 83eacd3a3533da5424da89269ca6f61b, both in the worktree and at commit 24434791d9f.
- It is drawn by busi/code/figures/figure_page_v5.py (md5 6a8e6283fa35b6dd3ff23c6959e0190a) from fig_page_v5_data.npz (md5 0cc932f3376506c341584c74c50b8e65).
- That npz was written by figures_data_v1.py (md5 e2469b43bc4547c0e695f2e7a1eb6fb8) (<internal>/README.md).
- figures_data_v1.py reads:
  - <internal>/PAGE_fig_v4.1_data.npz (md5 5119ffa229c003ebb7eb69e13b96962c, box md5sum).
  - <internal>/v6s_B2hold_k1_summary.json (md5 817c33756ce68561c81ce17070ec78a1).
- PAGE_fig_v4.1_data.npz was written by solve/code/page_v4/page_v4_fig_v1b.py (md5 54753621…) from page_v4_data_v1.{json,npz}.
- Those were produced by page_v4_data_v1b.py (md5 26f909ef3cecee6506b32a6ffcd9ce45) over the B2 channel files κ = 1-18 (<internal>/FIGURES_SOURCES_v1.md, Fig. 2 block).

Payload selectors:
- **Fig. 2(a).**
  - `a_S_cut{2,16,30} = P["SUM_B2_cut"+c]` (figures_data_v1.py:126-128).
  - `SUM_B2_cut*` = column j of `SUM_B2` (page_v4_fig_v1b.py:26, :45).
  - `SUM_B2 = KB2["SUM"]` with SUM = Σ_κ W_κ x_κ over κ = 1…18, set to NaN if any channel is missing (page_v4_data_v1b.py:147-153, ks = range(1,19) at :214).
  - x_κ is the d13 `excess` = S_X(C) - S_X(P_-(h)) of one channel (channel_arrays :109-121).
  - The clocks are the intersection of all 18 channels' row clocks (:216). The plot keeps t ≤ 100 (figure_page_v5.py:317).
  - Tick marks `light_band_cut*` = arrival record clocks of the formation's light from 2 and 1.5 × 2GE_f (page_v4_fig_v1b.py:48; light_travel_a page_v4_data_v1b.py:171-177, :409-416). This is outgoing PG travel at the formation record's charge, F(x) = x²/2 + x + ln(x-1).
  - The "thermal budget" line is hard-coded as (12.125, 0) to (100, 10.4) (figure_page_v5.py:19, `T_FORM, BUDGET_100`; :322). It is not computed by the script.
- **Fig. 2(b).**
  - `b_ratio_signed = P["ratio12_TH"]` (figures_data_v1.py:129, :136), and ratio12_TH = (L_κ1 + L_κ2)/(L_H,κ1 + L_H,κ2) at T_H (page_v4_data_v1b.py:349, :356).
  - L is the per-channel `L_interval` = (Σ of 16κ-weighted per-record collar flows at 16 r_s since the previous checkpoint)/Δt (page_history_v6o.py:2189-2192).
  - L_H,TH = continuum_comparator(channels, 2gE_k, T_H): Σ_κ (64κ/2π) ∫ x Γ_κ(x) f_FD(x/(r_s T)) dx / r_s², with x on 48 Gauss-Legendre nodes in [0, 1.5] and Γ_κ from page_scatter_v5.gamma_table (page_history_v6o.py:856-871, :2186-2187).
  - T_H = 1/(4π·2gE_k) (:1086), i.e. 1/(8πGE) at the record's E.
  - The line is the mean of the signed ratio over all finite checkpoints with |t - t_i| ≤ 5 r_s (figures_data_v1.py:129-134). Points are drawn only for t ≥ 10 (figure_page_v5.py:333).
- **Fig. 2(c).** c_excess16_cut{2,16,30} = 16 × rows[].per_channel["1"].interior.cuts[c].excess of the held run, with held flags from rows[].held and hold_from = T_hold/64 (figures_data_v1.py:138-144).

Cut-site inclusions:
- The cut radii are CUT_R_ABS = {2: 128, 16: 1024, 24: 1536, 30: 1920, 48: 3072} lattice units (cut × 64; page_v4_data_v1b.py:70-71).
- In exterior_entropies_sector (page_w1_sector_v1d.py:415-476), X = sites with R > R_abs (mask at :439) and Xc = R ≤ R_abs.
- S_X = 2 S_Xc(C_+): the + sector's complement block, doubled for the - sector by the pairing (:440-442).
- X includes the appended PG exterior out to site 8192.
- The second-route identity record uses "X consists of stored R < 2 r_s; both components", 127 selected sites, on the heavy covariance (<internal>/stored_identity.json, keys cut_convention and selected_site_count). The production's Xc at 2 r_s has 128 sites (R ≤ 128).

Held-vacuum construction:
- killing_vacuum_sector(h_csr) (page_w1_sector_v1d.py:352-369):
  - Dense eigendecomposition of the + sector block h_+.
  - The occupied set is V_+[:, e_+ < 0] together with conj(V_+[:, e_+ > 0]) for the - sector.
  - Exact zeros are excluded from both, and their count is reported as n_zero_1e-12.
- h is the full-chain (8192-site) clocked operator at the record's own fields H.fields(k) (checkpoint_lean, page_history_v6o.py:1673; vacuum at :1715 via the readout's cr.vacuum). The cut restriction is applied afterwards: the Gram of V_+ rows with R ≤ R_abs (exterior_entropies_sector :455-461).
- The vacuum route in the run is "sector": d13.cfg.vac_route = "sector" in the summary, while the CLI default was "dense". w1_set_vac_route overrides it (page_history_v6o.py:1377-1385).
- The ruling-2 runA note says the state at t_k last saw the ramp field at t_k - Δ_sub/2. The held vacuum is at the record field (box <internal>/runA_v1.py, header lines 1-5; md5 a720f4f44b76859d89b39e073b89f263).

Eigenvalue endpoint handling:
- Production: Gram spectra by torch eigvalsh of ½(G+G†), using the smaller of X†X and XX†, with a CPU eigvalsh fallback if cusolver fails (gram_eigs, page_w1_sector_v1d.py:394-408).
- Entropy: binary_entropy_t clamps x to [0,1] and uses xlogy (page_ops_v6.py:433-435).
- Second route: np.clip(values, 0, 1). A spectrum is invalid if any value is < -1e-8 or > 1+1e-8 (<internal>/run_PAGE_spectral_secondroute_v1.py:34-50, md5 6273d7c51eed23747f4303e1db92659d).
- Recorded totals: eigenvalue_extrema [-2.31e-12, 2697.27] and 14113923 eigenvalues clipped within tolerance (spectral_entropy.json, top keys).

Propagation-floor computation:
- Per channel, propagation_floor (page_w1_sector_v1d.py:278-311):
  - T_cheb = 4 n_+^{3/2} ε + 2 n_+ Σ_s tail_s + n_+ n_sub · 16 · K_max · ε, where ε = float64 eps.
  - T_meas = ||Φ†Φ - 1||_1.
- excess_floors (:313-349):
  - low = 2 m h2(T_meas/m) + round; high = 2 m h2(T_cheb/m) + round; round = 2 m h2(8ε).
  - m = min(n_Xc, max(n_+, q)).
- The floor attached to each displayed sum is F_meas = Σ_κ 16κ f_meas,κ (page_v4_data_v1b.py:152, `FMW`).
  - f_meas = 2 m h2((T_trunc + T_meas)/m) + 2 m h2(8ε) (w1_postpass_v1f.f_meas, <internal>/w1_postpass_v1f.py:153-163, md5 f64669038e16b94e9cdf7a8bcc5a1205).
  - A term is "resolved" if |x| > f_meas (page_v4_data_v1b.py:159).

Channel weighting: W_κ = 16κ for κ = 1…18 (page_v4_data_v1b.py:143; ChannelOps self.w = 16κ, page_history_v6o.py:541).

Tail and common-mode statistic:
- c = max over κ = 12…18 of |16κ x_κ| at each (clock, cut). share = K·c = 18c (page_v4_data_v1b.py:155-157; w1_postpass_v1f.py:225, :232-233).
- Verdict order (w1_postpass_v1f.py:241-248):
  - "not formed" if |S| < share.
  - "within the floor" if tol·|S| < F_meas.
  - "converged" if the last three terms fall (q_x < 1 here and at the neighbouring clocks), are resolved, and the geometric tail ≤ tol|S| and share ≤ tol|S|.
  - Otherwise "formed, tail open".
- The spread statistic is max/min of |16κ x_κ| over κ = 12…18 (page_v4_data_v1b.py:289-291).
- Probes at κ = 24, 32 and 48 (ruling2 runs) are used for the conditional geometric majorant |288 x_18| q/(1-q), which is never applied (:257-288).
- Fig. 2(a) marker fill uses the verdict ∈ {formed, tail open; converged} (page_v4_fig_v1b.py:39-45). figure_page_v5.py does not use the share band or the verdict fill: it draws "-o" for all clocks.

### Paper against code (marker 7)

- The paper says the share is "18 times its largest weighted term over k=12 to 18 provided no lower channel's weighted term exceeds it" (07_page_curve.tex:186-189). The code takes c over κ = 12…18 unconditionally. No check against lower channels was found in readout_v1f (w1_postpass_v1f.py:225-233) or sums_K (page_v4_data_v1b.py:155-157).
- The paper places L_col "at the collar's retarded clock in each record's stationary chart" (07_page_curve.tex, flux paragraph after the figure). Fig. 2(b) plots against the record clock T/r_s (figures_data_v1.py:135, `b_t_over_rs = t = P["clocks"]`).
  - t_inf_collar is computed (page_history_v6o.py:2206) but is not the abscissa.
  - The offset t_inf - T is 0.37 to 0.51 r_s over 14-100 (page_v4_fig_v1b.py, a1 xlabel near line 110).
- The paper defines P_- as "the Hamiltonian held at the same cut and record" (07_page_curve.tex:80-81). The code builds P_- of the full 8192-site chain operator at record k and then restricts to the cut. No cut-local Hamiltonian is built.
- The figure's thermal-budget endpoint 10.4 is a constant (figure_page_v5.py:19), the "plan's printed number". It is not recomputed.

Status: PARTIAL. Everything is found for the production route and the figure chain, except that page_d13_readout_v1d.py (the ChannelReadout that calls the entropy functions) is read by reference, not line by line.

## Marker 8

> [VALUE NEEDED: exact spatial and temporal interpolation, averaging order, exterior join radius, charge interpolation/holding choice, taper or matching rule, origin/far-boundary treatment and time substeps for principal light and ray generators; obtain from the executed page_history/page_ops/raymap functions selected by stored_identity.json, registry_v1.json and their primary input bindings.]

### A. Principal light generator (Page/light channel, Fig. 2(a)(b), Sec. "The light field on the computed collapse")

Producer and run:
- Executed code is page_history_v6o.py. The run summary records `"code": "page_history_v6o.py"` and no hash (box <internal>/v6o_B2_k1_summary.json, key `code`).
- The box copy is <internal>/page_history_v6o.py, md5 4c793ad7110ec0e4dcf3881b84d2115b, 224825 bytes. It is recorded "untouched" with that md5 at <internal>/STATUS.md:1160.
- It was built by busi/code/w1/build_page_history_v6o_v1d.py from page_history_v6n.py (box md5 87daa125…) and page_d13_readout_v1b.py, per <internal>/RUN_W1_v1d.md:13-23.
- Imported modules (box code_pod7):
  - page_w1_sector_v1d.py, md5 0eaed28a4caaa809ea03cc92e2baf4f2. This equals the local file <internal>/page_w1_sector_v1d.py.
  - page_d13_readout_v1d.py, md5 70383bb15fa5c642a1ac7d570e2f923e.
  - page_ops_v6.py, md5 31a85c000cecb4425f5af66828392ab6. This equals the local file <internal>/page_ops_v6.py.
  - page_scatter_v5.py, md5 82568abf7fa2de038bba7cd84abf951c.
- Launch line: RUN_W1_v1d.md:49. Exact arguments:
  `--record $D/collapse_$T.json --fields $D/collapse_${T}_fields.npz --fields-key applied --nsites 8192 --kappas 1-16 --kappa-hard-max 16 --interp linear --interp-sub 8 --cheb-tol 1e-16 --sector auto --sector-propagate on --sector-check-every-rs 10 --audit on --cuts-rs 2,16,24,30,48 --R-cut-rs 24 --R-collar-rs 16 --join-rs 32,64 --collect-every-rs 2 --extra-ck-rs 15,30 --ext-every-rs 10 --vac-every-rs 10 --T-end-rs 100 --save-every 4 --columns-at-rs 12,20,40 --columns-every-rs 20`
- The stored settings (summary `settings`) confirm the following: record /workspace/engine/B2/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json, fields_key applied, nsites 8192, interp linear, interp_sub 8, cheb_tol 1e-16, T_end_rs 100, join_rs "32,64", channels per file 1, eta 0.9192, device cuda:0, threads 4.
- Channels 1-9 are in runs/v6o_B2. Channels 10-12 are in runs/v6o_B2_k10to12, 13-16 in runs/v6o_B2_k13to16 and 17-18 in runs/v6o_B2_k17to18. All are composed by symlink in <internal>/prod (box `ls`).
- Input history ("B2"): <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_B2n4096.json, md5 417a4018cf6497f9b0020408783f44f5 (box md5sum). Its fields npz is beside it.
  - Header: script engine_collapse_v19.py; nsites 4096; ell 4; tau 8; T_over_rs 100; reference R2; mix plain; stop_E 1e-07; stop_fields 1e-06; maxit 20; resume_from /root/B2crash/…, resume_T_over_rs 50.0.
  - Site fields are "averaged (engine_fields_v14, solve_constraints_ivp)".
  - Registry entry P64_s4_B2n4096 (registry_v1.json): "c19 at r_s/a 64 on 4096 sites (doubled box), 0 to 100", npz md5 cb26fcfaf3d6e80a17f0e73bac687301.
  - Summary `history`: 801 records, engine_sites 4096, T_end 6400.

Values, each with its source:
- **Fields used (averaging).** These are the engine's applied (accepted-iterate, back-averaged) site fields `N_applied`, `v_applied`, taken without further averaging (History.__init__, page_history_v6o.py:318-345; `kN, kv = ("N_applied","v_applied")` at :331). The engine does the back-averaging (kernel width ell = 4) before storage; the light code does none.
- **Spatial interpolation.** There is none. The light chain uses the engine's site grid R = 1 … n_e (unit spacing) directly. The code checks that the engine sites are R = 1…n_e (:337-341). The chain has R = 1…nsites (:336).
- **Exterior and join radius.**
  - `History.fields(k)` (:510-514): `N = 1`, `v = -sqrt(2 g E_k / R)` on every chain site, then overwritten by the record's applied fields on R ≤ n_e.
  - The join is therefore at R = n_e = 4096 a = 64 r_s, and the appended Painlevé–Gullstrand vacuum exterior runs from site 4097 to 8192 (128 r_s).
  - `--join-rs 32,64` is only a check window: `join_check` (:516-522) records the largest |N-1| and the relative |v - v_PG| over 32-64 r_s per record (row keys join_max_dN, join_max_rel_dv).
- **Taper or matching rule.** There is none. The switch at n_e is abrupt (:510-514); no blending code exists in fields().
- **Charge.** E_k is the record json's `E` (the collar charge) for the record at each stored clock (:332-334, `self.E`). It is held per record in the exterior.
  - Under the production linear interpolation, record_substeps blends the whole chain arrays fields(k) and fields(k+1), exterior included (page_w1_sector_v1d.py:712-717). The exterior v is therefore linear in time between -sqrt(2gE_k/R) and -sqrt(2gE_{k+1}/R); this is not E interpolated inside the square root.
  - The last record is held: if k+1 ≥ len(H) the record's fields are used piecewise-constant (:705-709).
- **Temporal interpolation and substeps (production).** `record_substeps` (page_w1_sector_v1d.py:698-717), mode 'linear':
  - n_sub = 8 substeps per record interval of tau = 8, i.e. 1 lattice unit each.
  - The field in substep i is (N0 + u (N1-N0), v0 + u (v1-v0)) with u = (i+0.5)/8, the midpoint.
  - The label is "the fields interpolated linearly between records in 8 sub-steps of tau/8, each at its midpoint" (summary `w1.propagation`; propagation_label :719-725).
- **Controls.**
  - `--interp none`: piecewise-constant over [T_k, T_k+tau), the v6n production, run as v6o_ctrl_B2 (RUN_W1_v1d.md:127-129, κ = 1-8 pass).
  - `--cubic-window` / mode 'cubic': Catmull-Rom in time, 1-unit substeps (History.cubic_fields, page_history_v6o.py:524-533; record_substeps :701-711).
  - The paper's "cubic spline" control is a different code. It is a scipy natural C2 CubicSpline in time through every record, run by page_history_v6s (md5 611d190f) for κ = 12, 16, 24, 48, with Catmull-Rom for κ = 12 and 48 (<internal>/NOTCOMP_2.4_v3.md:7, :37).
  - 16-substep control: run (c) of κ = 18, <internal>/v6o_B2r2c_sub16 (page_v4_data_v1b.py:55). It was launched by job_v6o_r2.sh (box md5 74c340d6…) with `--interp $INTERP --interp-sub $SUB`.
- **Operator assembly (order).** For each substep: fields(k) assembled (applied interior plus PG exterior), then linear time blend, then `clocked_csr`: h[N,v] = ½{N,h0} + ½{v,Q} (page_ops_v6.py:92-97).
  - h0 and Q come from `chain_sparse(R, m=0, kappa)` (page_ops_v6.py:72-89). The diagonal is ±m with -iκ/R and +iκ/R off-diagonal spin entries per site. Nearest-neighbour hops are only between sites i and i+1 for i = 1…n-1. Q couples sites i and i+2 (entries ∓0.25i).
  - So there is no hop across the origin (the chain starts at R = 1) and the last site is open (no hop beyond R = nsites); this is origin and far-boundary treatment by construction (:84-88).
- **Propagator.**
  - Chebyshev expansion of exp(-i h τ_sub) on the Gershgorin interval (cheb_apply, page_history_v6o.py:565-583).
  - The order is the smallest K ≥ |z|+4 with Bessel tail 2 Σ_{j>K}|J_j(z)| ≤ cheb_tol = 1e-16 (page_ops_v6.cheb_order :341-353).
  - Propagation is in the massless + sector (`sector_propagate on`, `sector_used: true` in summary `w1`).
  - The initial state is the ground state of h(T_0) per channel, from a dense eigendecomposition (docstring page_history_v6o.py:233-236).
- **Held-geometry continuation (Fig. 2(c)).**
  - Code page_history_v6s.py (box md5 611d190f…), mode "lean interior". Summary: <internal>/v6s_B2hold_k1_summary.json, md5 817c33756ce68561c81ce17070ec78a1.
  - Hold: k_hold 800, T_hold 6400 (100 r_s), E_hold 106.0533, records_added 3200 to T 32000 (500 r_s).
  - Label: "the fields (N, v) and E of the record at T_hold held for every later record (copies at the spacing tau); the propagation across each held record is exact in the held h".
  - nsites 32768 (not 8192), interp linear, sub 8.
  - The initial state was by scipy zheevr MRRR plus a Löwdin step (summary `init`).

### B. Principal ray generator (received redshift; Sec. IV received map)

Producer:
- rz_rays_v4.py (md5 94a08d9f278cbf696360b3be67031b81). The ray-radius pass rz_rays_v4m.py (md5 199a43a00f1073abaeac76c97aa7f80b) was made from it by mk_rays_v4m.py.
- Library rz_lib_v2.py (md5 e1d77a04e2f5150c198cb6107daf192a). Members module members_v2.py / members_v3.py (md5 664f6797… / 115f589b…).
- All are in <internal>/. The provenance lines are at <internal>/REDSHIFT_received_v3.1.md:91.
- Inputs are resolved through the registry <internal>/registry_v1.json, md5 efcfad78bb6e2c942f21cd4ef9028270. It is identical to <internal>/registry_v1.json (members_v2.py: REGISTRY, base_of).
- Reference member P64_s4 → registry key P64_s4_prod:
  - json <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json (md5 d7d69adcea7a43f8375ab4c7c5a80efc).
  - npz …_to40_fields.npz (md5 01444a399b9466047a5b0f118af61f77).

Values:
- **Production variant `bar_lin_cubic`** (rz_rays_v4.py:424-431, default at :437): fields "barred", time "lin", space "cubic".
  - "Barred" means the stored site fields `N`, `v` = "the kernel back-averaged (barred, 'coupled') fields" (rz_lib_v2.py:3-6; rz_rays_v4.py:73).
  - The other variants: app_hold_cubic (N_applied/v_applied held over [T_k, T_k+1), the "production replay"); bar_pchipt_cubic; bar_lin_pchip; bar_lin_linear; dense_lin_quintic (node fields N_nodes/v_nodes on r_i = i·ell through the engine's quintic parity spline).
- **Spatial interpolation.**
  - scipy CubicSpline, not-a-knot, through the sites R_j = j a (rz_lib_v2.py:70-74).
  - Data are parity-extended with 6 ghost sites at -1…-6 and no knot at 0: N even, v odd, f = N+v with f(-r) = N(r) - v(r) (rz_lib_v2.py:20, :54-67).
- **Temporal interpolation.** Linear between records of the per-record spatial polynomials (Field.at, rz_lib_v2.py:132-155).
- **Averaging order.** The engine back-averages first. The ray code then fits the cubic in r per record, then blends linearly in t (Field.at). There is no further averaging.
- **Exterior, join, taper.** No exterior is appended. The domain is the engine chain only (2048 sites = 32 r_s for P64_s4).
  - Receivers are limited to R[-1] - 1 r_s (`rmax_chain`, rz_rays_v4.py:473-476).
  - Outward integration stops at max(receivers) + 1 a (:235).
- **Charge.** Not used in the ray equations. The charge enters only diagnostics: 2gE_k per record (root_table) and r_vac with VAC_TOL 1e-3 (:31-38).
- **Origin.** The backward route is continued through r ≤ 0 using |r| and sign(r) (docstring :1; emitters_comoving rhs :164-168) on the parity-extended interpolant.
- **Time stepping.**
  - Fixed-grid classical RK4 on t_n = n h, h = 0.5 a (default `--h 0.5`, :438).
  - h must divide the record spacing (assert :454-455), so steps never straddle a record.
  - The receiver crossing is located by cubic Hermite with 8 Newton steps (:200-207).
  - Ladder controls at h/2 and h/4. Independent integrator: scipy DOP853, rtol 1e-11, atol 1e-12, restarted at every record time (rz_v3_reint_v3.py:13, :72-73, :96-97, :117-118).
- **Dispersive rays** (rz_v3_dispev_v1.py, md5 000ae638e415b36c52daff7145027cfd):
  - Ω = v q(p) + N ε(p) with q = sin(2p)/2 and ε = |sin p|, massless radial channel.
  - Integrated as (r, ln p) by piecewise DOP853, rtol 1e-11, atol 1e-12 (:6-12, :61).
  - ap_e = 0.05, 0.1, 0.2, 0.3.
- **Long-wave ray map (κ_u readout).** busi/code/w9/raymap_solved_flow_v3.py (md5 317dfadb…) imports raymap_solved_flow_v2b and page_d13_readout_v1's per-record Chebyshev propagation (header lines 5-9). Its interpolation choices were not traced further.

### Paper against code (marker 8)

- The paper says the light is propagated "on a history with … the vacuum exterior appended beyond the computed domain" (07_page_curve.tex:197-199). The code agrees, and the join is at the engine's last site: 64 r_s on B2, not at 32 r_s.
- The light history is B2: the c19 lineage on 4096 engine sites, engine_collapse_v19 and engine_fields_v14, resumed at T/r_s 50. It is not the v36d P64_s4 run.
  - The registry records P64_s4_B2 = P64_s4 "to 5.8e-6 through 40" (members_v2.py, MEMBERS["P64_s4_B2"]).
  - The paper's reference collapse is stated as one history (05_star.tex method paragraph).
- The paper's "cubic spline" control (07_page_curve.tex:185-186) is the natural spline of page_history_v6s. page_history_v6o's own cubic option is Catmull-Rom.
- The Fig. 2(c) held run uses 32768 chain sites. The production uses 8192.

### Archive status (marker 8 producers)

`git -C <internal>/entanglement-stars ls-tree -r origin/main` was checked for each file.

- **Not in the archive:**
  - page_history_v6o.py, page_history_v6n.py, page_history_v6s.py, page_d13_readout_v1d.py and page_d13_readout_v1b.py.
  - runA_v1.py and job_v6o_r2.sh, RUN_W1_v1d.md and the launch_v6o_B2_pod7_v1d.sh body.
  - All rz_* redshift scripts and members_v2/v3.py.
  - The v6o channel files, the B2 history, PAGE_fig_v4.1_data.npz and fig_page_v5_data.npz.
- **In the archive (neutralized: archive bytes differ from executed bytes, and the MANIFEST original_md5 equals the executed md5):** page_w1_sector_v1d.py, page_ops_v6.py, page_scatter_v5.py, w1_postpass_v1f.py, page_interior_v6s.py, page_v4_data_v1b.py, build_page_history_v6o_v1d.py and busi/w9/raymap_solved_flow_v3.py.

Status: PARTIAL. The light generator and the ray generator are both identified, and every listed choice is read from code. The executed light producer (page_history_v6o.py) and the executed ray producer (the rz_* scripts) are not in the companion archive (see the end of this section). No hash of the executed light code is stored in the run outputs themselves.

## Marker 9 (final page count): skipped, the lead fills it

> [VALUE NEEDED: final rendered page count, equation/table fit and unresolved-reference/overfull-box report after all accepted edits; obtain by building ms_fermion_new.tex with its committed class, bibliography and figure files.]

Not extracted, per the instruction. For context only: the subject line of paper commit 24434791d9f ends "; 30 pages". The document class is `\documentclass[prd,reprint,superscriptaddress,longbibliography,floatfix]{revtex4-2}` (<internal>/ms_fermion_new.tex:1). In the paper worktree, ms_fermion_new.bbl and ms_fermion_new.blg show as modified and uncommitted.

Status: SKIPPED (the lead fills it).

## Marker 10 (paper and archive commits): skipped, the lead fills it; the git HEADs seen are recorded below

> [VALUE NEEDED: immutable current-paper commit and companion-archive commit, including the exact bibliography target for McKinneyCode2026; obtain from the two Git commit records and the bibliography file used by ms_fermion_new.tex.]

Git HEADs and refs seen on 27 Oct 2026. They are recorded here, not chosen.
- Paper worktree <internal>/.wt-fermion-minimal, branch PR3: HEAD is fc938764ee48046920d6cad9271866cdc33ba525 (an hourly backup commit of lane notes). Its parent chain is 7c5600559b2, then 24434791d9f0868b509e3bebd6e8f943ef375602 (the "Page subsection ... 30 pages" commit). `git diff --stat 24434791d9f HEAD -- fermion_new ms_fermion_new.tex` is empty, so the paper source is unchanged since 24434791d9f. The working tree has uncommitted changes outside the paper source, including ms_fermion_new.bbl, .blg, ledger/ and .taskq/.
- Companion archive https://github.com/pseudotensor/entanglement-stars: main is f690b45eceee4a2dcddcc4f1bb456300881e87c5 (merge of PR 1). Branch fermion-new-proofs-codes-data is 248429d0e40ffa2a215634166e4bbf0b6aec465e. Branch sync-corrected-observable is 3a1f9620f2712c1456bd291ec85cd489df138384. These were confirmed by `git ls-remote origin`. The local clone <internal>/entanglement-stars has sync-corrected-observable checked out, and its working tree has no fermion material.
- qi2 work trees (`git worktree list`): <internal>/qi2 4d51cc6a98a [fix-reaudit-kinginputs]; .wt-builder-solve dd5ca3b53f8 [builder-solve]; .wt-builder 44cccedc04b [builder-numerics]; .wt-route2-solve cc69bd5c0d8 [route2-solve]; .wt-route2-hreduce 98c0c40711c [route2-hreduce]; .wt-route2-busi 1a9eaf894ac [route2-busi-2]; .wt-integrate-20261021 93dbab526a0; .wt-autointegrate c4d30509ed5 (detached).
- The bibliography entry is the same at 24434791d9f and in the worktree (ms_fermion_new.bib:748-754). It reads `@misc{McKinneyCode2026, author = {McKinney, Jonathan C.}, title = {Numerical codes for ``{Entanglement Stars}''}, year = {2026}, howpublished = {\url{https://github.com/pseudotensor/entanglement-stars}}, note = {Includes self-consistent solvers, figure-generation scripts, and pre-computed data}}`. It targets the repository URL only, with no commit, tag or DOI pin. The citation is made at ms_fermion_new.tex via the flat paper line 104 ("Codes and data are available at~\cite{McKinneyCode2026}.").

Status: SKIPPED (the lead fills it); the HEADs are recorded above.

## Marker 11

> [VALUE NEEDED: literal equations or pseudocode for S_j, R_j and T_j, including which covariance and reference enter each iteration, relaxation, row quadrature, boundary conditions, stopping/failure rules and the fields used in the exponential; extract the executed step and field-solve functions selected by the primary run lineage, not an unpinned engine_fields version.]

### 11.1 Executed lineage and code identity (reference collapse P64_s4)

- Run: `collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40` (321 records, T/r_s 0 to 40). Box copy: `192.168.1.93:<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json`, md5 d7d69adcea7a43f8375ab4c7c5a80efc (`.../MANIFEST_runs_pod7_ell4.md5`; same md5 in the ladder readout meta `<internal>/ladders_readout_v1_3.json`, members.P64_s4.meta). Fields npz md5 01444a399b9466047a5b0f118af61f77; spectra 3b7f723edd596ef337a4c22b7c94b070; handoff bfca6c549da5fd0665c4926f13e04300; log `logs/c32_..._to40.log` a8ce4358bed925038ca8c809c2376f58 (same manifest).
- Two segments, one engine file:
  - T/r_s 0 to 18.125 on pod 2: `launch_v36d_from0_rungs_pod2_v2.sh` (`<internal>/launch_v36d_from0_rungs_pod2_v2.sh:15-16,21`): `E=/workspace/engine/code_v36d/engine_collapse_v36d.py`, `--m 0.2 --g 0.25 --rs 64 --R0-over-rs 4 --reading A --nsites 2048 --kappa-sea 10 --reference R2 --site-fields averaged --T-over-rs 18.125 --prep-ell 4 --tau 8 --gpu 0 --slices-c128 1..18 --mix plain --field-kernel-cache --v36-fixes --ell 4`.
  - T/r_s 18.125 to 40 on pod 7: `launch_v36d_from0_continue_pod7_v2.sh:13-14,41` (`continue 4`), same arguments plus `--T-over-rs 40 --resume-slice states/.../orbitals_..._T0018.00.npz --resume-run $W/$TAG`; slice md5 5e44bd04675d35c666bbe6e61606247b (box `runs_pod7_ell4/status.txt`). Resume gate 18 to 18.125 "146 records; all within twice the floor: True" (box `runs_pod7_ell4/gate18_ell4/status.txt`). Segment placement: `<internal>/STATUS.md:1727,1827,1833`.
- Executed engine: `engine_collapse_v36d.py` from `/workspace/engine/code_v36d/` (path in the run log's first line, box log above). Working copy `<internal>/engine_collapse_v36d.py` md5 0315e0826ee872ee194474876ef523f9 = archive original_md5 (`entanglement-stars` origin/main `fermion_new_codes/MANIFEST.tsv`, row solve_a/engine/engine_collapse_v36d.py) = the md5 recorded for code_v36d on pod8b (`solve/results/ops/NOTES_reorient6_copies_v1.md:49`). The pod 2/pod 7 code_v36d tree is recorded only as "md5 of the code tree equal" to pod 4's (`launch_v36d_from0_rungs_pod2_v2.sh:12`; busi STATUS.md:1833); no per-file md5 of the pod 7 copy was found.
- Naming in the run json is not the executed version: v36d writes `"script": "engine_collapse_v34.py"` literally (`engine_collapse_v36d.py:2622,2644,2666`) and `solver_meta["engine"] = "engine_collapse_v25"` literally (`:2108`); the json head label says "v31 ... ops v3" (`:2503`). The executed file is v36d.
- Imports actually executed (`engine_collapse_v36d.py:377-383`), working-copy md5 = archive MANIFEST original_md5 in every case: engine_collapse_v5 (fc6faf26a353f6d904a5242874e25586), engine_fields_v16 (1ffcf5e88c7953d7bfee025d57f4fb86; also `busi/code/perf_host/LINEAGES_h2default_v1.tsv` "before" column), engine_mixing_v1 (d142154a1e454fed344f1b0ddacf71e3; unused on the plain path), engine_ops_v3 (460664cd52b8babd42f9a9c227499662), engine_ops_v2 (cdabf12b56be61d1d26879b57f073650; gershgorin only), engine_stress_v2, engine_ops_v34_cert (not called: weyl_certified null). engine_fields_v16 imports engine_fields_v12 (d25e02532ee3faea4a886494ac6f31cb) for SmoothedSources5, site_fields_kernel_average, charge (`engine_fields_v16.py:49-64`); v12 imports engine_fields_v4 (6ab508c2f078c5659dfa3d02c23c3e1d) for kernels and coarse_grid (`engine_fields_v12.py:76-78`). Chain operator from engine_vacflow_v1.chain_sparse (5225e959b9e0bc730062041d60d800c8; `engine_collapse_v5.py:81,102`). `--field-kernel-cache` installs engine_fields_kcache_v1 (010f220b067e37e6e39b94579a2a4c80), stated bitwise (`engine_collapse_v36d.py:3307-3310`, docstring :41-43).
- Archive copies are text-neutralized (docstring words changed; 52 differing lines for v36d) so their byte md5 differs from original_md5 (e.g. archive v36d blob md5 b4f191a189f8c99adbeebc3544fd8787); the original_md5 column matches the working copies.
- Options inert on this path: `--v36-fixes` acts only at `:1060,1263` (v15/v17 continuations, not called) and `:3034,3038` (Newton loop); `--mix-alpha 0.5` is read only when `--mix damped` (`:2220-2221`; solver_meta mix_alpha None for plain, `:2109`); `--window-max 4` is not read by ops v3 (window trimmed to its last entry, `:2373`); `--weyl-refine-max 10` only with `--weyl-certified` (null in args).

### 11.2 The record step as implemented (plain path, `engine_collapse_v36d.py` run(), lines 2153-2664)

Notation: record j at clock T_j = j tau, tau = 8 a; channels kappa = 1..10, weight w_kappa = 16 kappa (`engine_collapse_v5.py:101`); per channel the state is st = {sea columns, dust columns with occupations f, reference R2 columns, n_sea, window (last accepted record's R2, applied fields, gaps, mask)}.

S_j (sources), `:2160-2174`:
- base densities from the state at the start of the step (the propagated state at T_j, before any iteration): `base[kappa] = (dens(h,sea), dens(Q,sea), dens(h,dust,f), dens(Q,dust,f))` (`:2163`). dens(A, Phi, f) = per site sum over columns f Re(Phi^dag P_n A Phi), summed over the 2 spinor components (`engine_collapse_v5.py:120-128`); h is the unclocked chain h0, Q the two-step current.
- for a trial reference ref_kappa: eps = sum_kappa w_kappa [dens(h,sea) - dens(h,ref) + dens(h,dust,f)], J = sum_kappa w_kappa [dens(Q,sea) - dens(Q,ref) + dens(Q,dust,f)] (`:2170-2173`).
- So the covariance entering every iteration of record j is C(T_j) (start of step); no midpoint covariance on this path. A midpoint covariance C_prev + (C_new - C_prev)/2 exists only under `--source-continuation two-stage` (stage 1; docstring `:135-145`, `sources_with` `:2176-2187`), which requires `--mix newton`; P64_s4 ran with it null. It was used by the sigma/a 2 control P64_s2 (`source_continuation: "two-stage"` in box `<internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json` args).

R_j (field solve), `solve()` `:401-412` -> `engine_fields_v16.solve_constraints_ivp` (`engine_fields_v16.py:265-471`), then kernel averaging:
- Smoothed sources: rho_ell(r) = sum_n K_0(r,R_n) eps_n, J_ell(r) = sum_n K_1(r,R_n) J_n, tabulated on spacing ell/16 and quintic-splined on the parity-extended grid (`engine_fields_v12.py:83-121`); K_l Gaussian shell kernels of standard deviation ell (`engine_fields_v4.py:37-...`); sites R_n = n, n = 1..2048 (`engine_collapse_v5.py:102`).
- Rows integrated outward once (no iteration inside the solve, maxit=1): U' = 8 pi g r^2 (rho_ell + q J_ell), (ln N)' = -4 pi g r J_ell / q, q = -sqrt(U/r) (`engine_fields_v16.py:145-151`), scipy solve_ivp DOP853, rtol 1e-12, atol 1e-16 (`:173,217,229,249`; tolerances from `--field-rtol/--field-atol`, `engine_collapse_v36d.py:3236,3326`). Downward zero of U handled by the s = sqrt(U) stretch from U = 1e-6 U_scale; U <= 0 intervals continued as flat patch (L' = 0) and labelled (`engine_fields_v16.py:2-16,131-262`).
- Boundary conditions: centre regular series to r^5 below r_0 = ell/64 with analytic kernel coefficients (`engine_fields_v16.py:19-24,304-306`); collar node r_collar = i ell, i = ceil((R_last + 8 ell)/ell), or the chain's last node under collar_policy "flag" (`:276-292`); N normalized to 1 at r_collar, N = exp(ln N - ln N(r_collar)) (`:364-368`); W from dW/dr = 8 pi g r^2 N^2 rho_ell, W(0) = 0, DOP853 (`:383-384`); E = W(r_collar)/(2g) (`:399`; charge() `engine_fields_v12.py:756-760`); v = -sqrt(W/r) (`:411-417`); sites beyond the collar N = 1, v = -sqrt(2gE/R) (`:452-454`).
- Site fields entering h_m ("averaged"): N_n = int 4 pi r^2 K_0(r,R_n) N(r) dr, v_n = int 4 pi r^2 K_1(r,R_n) v(r) dr, trapezoid on spacing ell/16, band |r - R_n| <= 8 ell (`engine_fields_v12.py:817-846`; called at `engine_collapse_v36d.py:408`).
- Failure of R_j: any failed integration returns info["failed"] (`engine_fields_v16.py:303-306,386-388`); the record is written with the failure and the run stops ("the re-solve is REJECTED", `engine_collapse_v36d.py:2253-2256`).

Self-consistency iteration (relaxation), `:2198-2251`:
- k = 0 (lagged): applied fields = previous record's accepted applied fields f_ref; reference = st.ref (previous record's accepted R2); no eigendecomposition; one solve gives (N_1, v_1) (`:2198-2205`).
- k = 1..maxit (20): plain substitution N_app, v_app = previous solve's output (`:2222`; no damping, alpha unused); eigendecomposition of h_m(N_app, v_app) per channel (`:2223`, torch.linalg.eigh `engine_collapse_v5.py:115-117`); R2 continued by T_j from the window's last entry (the previous record's accepted R2 and applied fields, not the previous iterate) to (N_app, v_app) (`:2225-2231`); S_j; R_j.
- Stop: |E_k - E_{k-1}|/|E_k| < stop_E (1e-7) and max_n |N_out - N_app| < stop_fields (1e-6) and max_n |v_out - v_app| < 1e-6 (`:2243-2251`; defaults `:3249-3251`, args in run json).
- Not converged at maxit: the smallest-residual iterate (residual max(dN, dv)) is kept and the record flagged "unconverged", the run continues (`:2248-2249,2262-2269`). No eigendecomposition at the accepted fields: record failed, run stops (`:2257-2260`).
- Accepted configuration: (N_app, v_app) of the stopping iterate, its eigendecomposition and its continued R2; the record stores the solve output N, v and separately N_applied, v_applied (fields npz, `:2132-2140`).
- Executed counts on P64_s4: all 321 records `"converged": true`, `"unconverged": null`; `"iterations"` field: 2 (1 record), 3 (12), 4 (224), 5 (80), 6, 7, 18, 21 (1 each) (box run json, grep counts).

The exponential (fields used), `:2650-2664`:
- h_m = (1/2){N, h0} + (1/2){v, Q}, entrywise (N_i + N_j)/2 h0_ij + (v_i + v_j)/2 Q_ij on the 2 x 2048 complex128 matrix (`clocked_gpu` `:929-944`; host form `:1108-1112`).
- with the accepted (N_app, v_app): h_m = U diag(e) U^dag; sea and dust columns propagated X <- U exp(-i e tau) U^dag X (`:2654-2659`); the reference R2 is not propagated (kept as continued); T <- T + tau (`:2664`). Fields held fixed over the step.

Chain operator boundary conditions: h0 on-site m sigma_z + (kappa/R_n) sigma_y, bond -(i/2) sigma_x between n and n+1 only (no hop across the origin, last site open); Q = -(i/4) between n and n+2 (`engine_vacflow_v1.py:53-73`). Initial reference R2 at T = 0: the negative-energy eigenvectors of h_m at the prepared fields (`engine_collapse_v36d.py:2064-2072`).

### 11.3 Certification rule (readout, not engine)

- The engine stop rule is 1e-7 / 1e-6 (above). The 1e-3 rule is applied afterwards by the ladder readout v1.3: "the certified prefix (before the first non-counting record: kept or confirming residual above 1e-3) ... records counting with either residual above 1e-4 are banded" (`ladders_readout_v1_3.json` "definitions", box `<internal>/ladder`).

### 11.4 Fig. 1 production c19 (records 11.125 to 40)

- `<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json` (box), md5 e27e457d37615cff8d5216f05a494f89 (ladder readout meta). Args: script engine_collapse_v19.py, nsites 2048, ell 4, kappa_sea 10, tau 8, reference R2, site_fields averaged, stop_E 1e-7, stop_fields 1e-6, maxit 20, mix plain, c_LZ 1.0, window_max 4, resume_from C3prod/c17_rs64_A_R2_tau8_ell4_avg, resume_T_over_rs 11.0 (json head).
- Records 0 to 11 are the engine_collapse_v17 parent's (89 records, label "R2 (the continued sea, degeneracy aware); engine_fields_v12"); records 11.125 to 40 (232) are v19 (label "Landau-Zener matched maximum overlap (c_LZ = 1, window 4); engine_fields_v14") (json label tallies; ladder meta engine_entries).
- v19 imports engine_fields_v14 (dc864dfeafbf233f5669b961a51cb7fc) and engine_ops_v2 (`engine_collapse_v19.py:181-185`); md5 v19 9e4e6abf4f3a23d97ac8ba79dda85785 (= archive original_md5). Same plain loop structure (v17's iteration, docstring `engine_collapse_v36d.py:234-247`); differs from v36d in field module (v14 vs v16) and continuation (ops v2 with a window, below).

### 11.5 Second route for the formation clock

- The second solver is the same discretization by Newton, not a solve_b code: `lad_P64_s4_newton_v36nq_h100_1`, `engine_collapse_v36nq.py` md5 8549c5429e0fcd2608c92434a3c26967 with `engine_fields_v16.py` md5 d7250a370c0f083f5a8c740c108d49e7 (= engine_fields_v16b, not 1ffcf5e8), env H2_PW_ONE=1 H2_LOCAL_SPLINE=1, `--mix newton --nk-jvp exact --nk-line-search halving --nk-halvings 6 --nk-krylov 40 --nk-restarts 2 --nk-maxiter 30 --nk-set resolve --end-above 0.001 --pair-workers 2`, T/r_s 0 to 13 (box `<internal>/LAUNCH.txt`, `args_v1.txt`; result `<internal>/FORMATION_newton_route_v1.1.md` sec. 1-3; solve STATUS.md:1034). v36nq source on box `<internal>/engine_collapse_v36nq.py` (md5 8549c542, verified); not in the archive (only its builders under fermion_new_codes/busi/). LINEAGES_h2default_v1.tsv records v36nq's fixture gate vs v16 as BITWISE.
- solve_b (`<internal>/solve`): no collapse step function found. `FORMATION_step_second_route_v1.tex` is a specification of three fixed-source solves and one stored step on c19 records 96 to 97 and states "No scientific code has been written or executed here" (lines 11-31); `C_fixture_second_route_v1.tex` is a 256-site prescribed-field light fixture; `WEB40I_formation_scalar_results_v1.tex` is an analytic comparator QC ("not ... a production-history calculation").

### 11.6 Paper vs code

- Paper (05_star.tex at 24434791d9f, lines 54-58): "The Gauss rows use the midpoint sources minus the reference, continued spectrally over the step's endpoints or its subdivisions; the per-step choice is not read." Code (P64_s4, plain): the sources are those of the covariance at the step's start, C(T_j), minus the continued reference (`:2163,2166-2174`); no midpoint covariance on this path. A midpoint covariance appears only in the two-stage source continuation (Newton members, e.g. P64_s2). The per-step continuation choice is logged per record and channel (11.2/marker 3). Disagreement reported, not resolved.
- Paper (lines 60-61): certification at a one-iteration change of at most 1e-3. Code: engine stop rule 1e-7 (|dE/E|) and 1e-6 (fields); 1e-3 is the readout's certification rule (11.3).
- Paper "fields solved self-consistently from the state at the start of each step and held over the step" agrees with `:2163` and `:2654-2659`.

Status: FOUND (P64_s4 lineage, S_j, R_j, iteration, stop/failure rules, exponential). PARTIAL only in that the per-file md5 of the code tree actually on pods 2 and 7 is not recorded (only "tree md5-equal"); looked in solve STATUS.md, RUN_RECORD.md, results/ops/NOTES_reorient6_copies_v1.md, busi STATUS.md, box runs_pod7_ell4/ and v36d_from0/ manifests.

## Marker 12 (SHA256 of distributed bytes; omitted arrays)

> [VALUE NEEDED: SHA256 of distributed source, data, figure and manifest bytes, and resolvable locations for required omitted arrays; obtain from the archived files, data/MANIFEST.tsv, fermion_new_codes/MANIFEST.tsv and SKIPPED_LARGE.tsv.]

Archive read: companion repository https://github.com/pseudotensor/entanglement-stars, branch main at commit f690b45eceee4a2dcddcc4f1bb456300881e87c5 (merge of PR 1, branch fermion-new-proofs-codes-data at 248429d0e40ffa2a215634166e4bbf0b6aec465e, "Fermion paper: add proofs, source codes and result data", 23 Oct 2026). `git ls-remote origin` on 27 Oct 2026 gives the same main head, so the local origin/main ref is current. The local clone <internal>/entanglement-stars has branch sync-corrected-observable (3a1f962) checked out, and that working tree has no fermion_new_codes/, data/ or proofs/ directories. Every value below was read from git objects at f690b45 (`git show origin/main:<path>`), not from the working tree.

**Full per-file SHA256 list.** See the companion file PFIX_VALUES_v1_sha256.tsv, in the same folder as this file. It has one row for each of the 5780 blobs at f690b45, with the columns path, size_bytes, sha256 and git_blob_sha1. The hashes were computed over the archived blob bytes with `git cat-file --batch` and Python hashlib. The file counts per tree are: data/ 1680 files (1678 listed in its manifest, plus MANIFEST.tsv and README.md), 775,508,741 bytes; fermion_new_codes/ 3805 files (3803 plus the 2), 54,994,974 bytes; proofs/ 68 files (66 plus the 2), 1,960,915 bytes.

**Manifest and readme bytes at f690b45:**

| path | bytes | SHA256 |
|---|---|---|
| data/MANIFEST.tsv | 189931 | fc43e4ecbf540a4e1f9213e1ad0d0c4d0403f4dd4ac99e9bf37dadbb0d5cac0b |
| fermion_new_codes/MANIFEST.tsv | 306774 | 726c8caa39f31cc16f031d46585d14167b97c11626e43a11f8dcfda88e3896d9 |
| proofs/MANIFEST.tsv | 4551 | 3f257fad489d99cb4d1c8bc562ace682ce3fdb12c259190410c9f99fb37ae7a3 |
| SKIPPED_LARGE.tsv | 17520 | 93aac7dfb9c7ac571bb94dbb7d9774d08eaf2fbbc55207485dbe2db7bc650ae8 |
| README.md (top) | 16069 | 251a9b36a92b428de396b55572c780de97e5759bdc2b099904bd9b4b19142b42 |
| data/README.md | 1029 | 05f9ba50f292d3ab8691d1df96a598fcebc5778c9fb076c657784b0b52948ed1 |
| fermion_new_codes/README.md | 2372 | cbf80d53f43202a3a2305e7ccbe9a78b0084c8e099f1229f6cd0d048aef30020 |
| proofs/README.md | 8159 | 296bc0ee2f7a0d3e9bbe76e03f5e0819aea3627790f8f26b2056d44b9a0e75bb |

**What the manifests record.** They do not record SHA256.
- data/MANIFEST.tsv header: `file size_bytes original_md5 neutralized`. fermion_new_codes/MANIFEST.tsv uses the same header. proofs/MANIFEST.tsv header: `file original_md5`, with no neutralized column. SKIPPED_LARGE.tsv header: `intended_repo_path size_bytes size_MB md5 reason where_it_lives`.
- data/README.md (f690b45) says text data files had provenance paths and lane names neutralized, and that binary .npz and .npy files are byte-identical. fermion_new_codes/README.md says the same for code. proofs/README.md says that "Two mechanical edits were made to every file".

**Manifest consistency check.** Each manifest row was compared with the tree, and each archived blob's md5 with original_md5 wherever the row claims a byte copy.
- data: 1678 rows, the same as the 1678 tree files, with no row missing either way. 789 rows have neutralized=1. Of the 889 rows with neutralized=0, 878 have archived bytes whose md5 equals original_md5. In **11 rows flagged neutralized=0 the archived bytes differ** from original_md5, and every one of them is smaller than the size_bytes the manifest records. In each case the archived md5 is the right-hand value below:
  - hreduce/results/eta_massive_scalar_v1/environment_continuation/separate_images_v1/saved_t9_child_rows_v1.tsv (manifest 620152 B / archived 618379 B; 7a3a6725d59d9db324665715ab6e4c0e vs 24bb80ba20d9ed2bbbca9b3c28f6b27a)
  - .../separate_images_v1/saved_t9_geometry_rows_v1.tsv (129038 / 127265; eac855f4... vs 2f5340c6...)
  - .../separate_images_v1/saved_t9_parent_rows_v1.tsv (259374 / 258930; 9b825da2... vs e42601f9...)
  - hreduce/results/second_route/core_well/profiles_a_tables_v1/core_profiles_A_radial_v1.tsv (5014545 / 5008400)
  - .../profiles_a_tables_v1/core_profiles_A_shell_v1.tsv (3936 / 3887)
  - .../profiles_b_tables_v1/core_profiles_B_radial_v1.tsv (2265356 / 2228491)
  - .../profiles_b_tables_v1/core_profiles_B_shell_v1.tsv (20985 / 20696)
  - hreduce/results/second_route/residual_audit/v1/roots/readout_v1/root_records_v1.tsv (5496100 / 5462001)
  - solve_b/results/PACKET_secondroute_v1/review/exponential_metadata_v1.tsv (3021818 / 2999963)
  - solve_b/results/PACKET_secondroute_v1/review/sensitivities_v1.tsv (309487 / 307022)
  - solve_b/results/PACKET_secondroute_v1/review/trajectory_summary_v1.tsv (119820 / 119691)
  
  For these 11 rows the neutralized flag is therefore inconsistent with the bytes.
- fermion_new_codes: 3803 rows, the same as the 3803 tree files. 3204 rows have neutralized=1. All 599 rows with neutralized=0 match original_md5 byte for byte.
- proofs: 66 rows, the same as the 66 tree files. There is no neutralized column. 4 files match original_md5, and 62 differ, which is consistent with the README's statement that every file was edited mechanically. For those 62 rows, original_md5 is not a hash of the distributed bytes.
- So for every neutralized file (789 data, 3204 code, 62 proofs, plus the 11 data rows above), the only hash of the distributed bytes is the SHA256 in PFIX_VALUES_v1_sha256.tsv. original_md5 identifies the pre-neutralization original.

**Key executed source files: original bytes against archived bytes.** Originals are read from the work trees, and their md5 equals the manifest's original_md5.

| file | original md5 (= manifest original_md5) | original SHA256 | archived (neutralized) SHA256 |
|---|---|---|---|
| solve_a/engine/engine_collapse_v36d.py | 0315e0826ee872ee194474876ef523f9 | 3ebb07b5ee7dba34d940ea174f5bcb57a7c20706ad6e37894daaf790e2192666 | a92b11717d83883d9a6c4231ca433715b82ee5749e3467980089c44b03718d43 |
| solve_a/engine/engine_collapse_v34.py | 5f3fa577e6174c2f637f6a2a97db71e1 | f5afd7b69765e1db8f2a1231ec039561058e1311a88dbb01a0caf94ce7f6eb42 | 12088c6e3a66b6392a3411cbaad22adae7888a5eed64c4bac7c45e8540e120f8 |
| solve_a/engine/engine_collapse_v25.py | 5afaae43b7f54e027af12d5f813df685 | f157fea1bca942bf85ea0a5a5b97a68eaa61e7dd5c1cfbc9a1e8832e48435597 | b5682fa5b3f6ee958ddd38b3a1b7f9607507312362ced2276978fc0602a4a4ff |
| solve_a/engine/engine_collapse_v19.py | 9e4e6abf4f3a23d97ac8ba79dda85785 | e4968cbb0b15a1dc4686dc8a08c28bef13d7ca9bb4b4acf2c6379dfddc4e1bb7 | 5572010056b1dc40cb57a9938cd7cb6c3daa03098e78803b0dc329bf3df2abe2 |
| solve_a/engine/engine_fields_v16.py | 1ffcf5e88c7953d7bfee025d57f4fb86 | ff556d2c4672bc5dbf5989154a62a95486052920a1e9f8308960e88912d0f528 | 5f4a0ad9f082d61535c41e2228c6019f6e5bda9ba5995ab0011d258e47cc4f56 |
| busi/figures/figures_data_v1.py | e2469b43bc4547c0e695f2e7a1eb6fb8 | 4d7186f1c761bee65a346d38088f9d687e33dff4b00018a14ce3cfdeb63159a4 | 603cc03219c32ad1ae886b3dde755e9645530c9bd376fb6298b2c9ff2b7ed8ad |
| busi/figures/figure_item5_collapse_v3_1.py | 8eab4a5f77ce56e64158cbb5495bd745 | ad5d992d73faea243d303a82a2657c4d02ba790bc288e9615dfee3d214cd98fd | 41371ea752ba71e948338636ed16e3fcbc87d477d8057cfdb27c1ab4e458cb54 |
| busi/figures/figure_page_v5.py | 6a8e6283fa35b6dd3ff23c6959e0190a | 058db684dacaa0aa7fb6a1a07449d7fd5b37605cbd9b0f9ec88757f52c44b62c | b225f4106c23062427f2197239d30331f540b6565b1edf60b3f95eaddf0aa6d7 |
| busi/figures/figure_item5_collapse_v3.py | 398546faa2bf93693791fb589117fd7c | c89df0affd9115fd528aa78a18933938cfbe27e83ed248e5c2c0403d867c80ff | d971b7d100ffe20072be020bee839b07f85597ab02dd279fc4db9ae7bf5b888b |

Original paths: <internal>/engine and <internal>/. The executed-engine md5 0315e082 for v36d is the one recorded in <internal>/STATUS.md:1290 and :1450 ("v36d 0315e082"). The engine_fields_v16 md5 1ffcf5e8 is the one recorded in data/busi/perf_host/LINEAGES_h2default_v1.tsv (archive).

**Figure bytes.** f690b45 contains none of the paper's three included figure PDFs (item5_collapse_v3.1.pdf, page_v5.pdf, page_curve_amps.pdf). It also does not contain the figure-data files fig_item5_collapse_v3_data.npz and fig_page_v5_data.npz, or the Page-figure input PAGE_fig_v4.1_data.npz. This was checked by `git ls-tree -r --name-only origin/main | grep -iE 'item5|page_v5|page_curve_amps|fig_page|fig_item5|PAGE_fig'`, which finds only scripts. The archive's only figure files are the older fermion/numerical/figures/*.pdf|png. The PDF hashes at the paper commit are in Marker 4.

**Omitted arrays: what SKIPPED_LARGE.tsv covers.** SKIPPED_LARGE.tsv has 55 rows. Every one has where_it_lives = "hreduce working tree on the author's workstation" (13 rows) or "solve_b working tree on the author's workstation" (42 rows), followed by a relative path. These are md5 plus a private location, not a public resolvable location. data/README.md says "a copy is available on request". The reasons given are: 10-20 MB (22 rows), 2-10 MB not referenced by a delivery (24 rows), over 20 MB (9 rows).

**Principal payloads that are neither in the archive nor listed in SKIPPED_LARGE.tsv.** Only the box location and the box md5 were found for these:

| payload (figure/readout input) | box location | md5 (source) |
|---|---|---|
| reference collapse P64_s4 run json | 192.168.1.93:<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json | d7d69adcea7a43f8375ab4c7c5a80efc (MANIFEST_runs_pod7_ell4.md5 in that dir). Byte copy listed in SKIPPED_LARGE as data/solve_b<internal>/input_records.json (same md5), location "solve_b working tree on the author's workstation" |
| P64_s4 fields npz (read by figures_data_v1.py:27) | same dir, ..._to40_fields.npz, 34,227,750 B | 01444a399b9466047a5b0f118af61f77 (same manifest); not in archive, not in SKIPPED_LARGE |
| P64_s4 spectra npz | same dir, ..._to40_spectra.npz, 106,836,448 B | 3b7f723edd596ef337a4c22b7c94b070; not in archive, not in SKIPPED_LARGE |
| P64_s4 handoff npz | same dir, ..._to40_handoff.npz, 51,787,439 B | bfca6c549da5fd0665c4926f13e04300; not in archive, not in SKIPPED_LARGE |
| P64_s4 run log | same dir, logs/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.log | a8ce4358bed925038ca8c809c2376f58; not in archive |
| c19 production run json (figures_data_v1.py:26) | 192.168.1.93:<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json | e27e457d37615cff8d5216f05a494f89 (md5sum on the box). Byte copy listed in SKIPPED_LARGE as data/solve_b<internal>/input_records.json |
| c19 fields npz (figures_data_v1.py:25; Fig. 1(a),(b) fields) | same dir, collapse_c19_rs64_A_R2_tau8_ell4_avg_r11_fields.npz, 34,263,267 B | 876dd5c6dee34b1c290b33eaeb4d25b1 (as logged by figures_data_v1.py, <internal>/figures_data_v1.log; see Marker 4); not in archive, not in SKIPPED_LARGE |
| ladder readout (figures_data_v1.py:28-29) | 192.168.1.93:<internal>/ladders_readout_v1_3.json and .npz | json 63d46f940870542526e597984668964d, npz f172431d6a99447f27bdd046a6482af3 (MANIFEST_ladders_readout_v1_3.md5); neither in data/MANIFEST.tsv nor in SKIPPED_LARGE |
| held drain summary (figures_data_v1.py:30-31) | 192.168.1.93:<internal>/v6s_B2hold_k1_summary.json (749,544 B) and v6s_B2hold_k1_interior.md | 817c33756ce68561c81ce17070ec78a1 and 45ab942d16fb35879019e6e2bc171905 (md5sum on the box); not found in data/MANIFEST.tsv by md5 |

The md5 search over data/MANIFEST.tsv and SKIPPED_LARGE.tsv was done by grepping for each md5 above. The held-drain and c19-fields md5s were not grepped against SKIPPED_LARGE.tsv, but its 55 rows all lie in the hreduce and solve_b trees. The archived figure-data and Page-figure inputs (fig_*_data.npz, PAGE_fig_v4.1_data.npz md5 5119ffa229c003ebb7eb69e13b96962c) are also absent from both; see Marker 4. Marker 4 lists further figure inputs, such as PAGE_fig_v4.1_data.npz and the readout TSV.

Status: **PARTIAL**.
- FOUND: SHA256 of every distributed source, data and manifest byte at f690b45 (the companion TSV), and the manifest consistency results.
- NOT FOUND: there are no figure bytes in the archive. No public resolvable location exists for any omitted array: SKIPPED_LARGE gives only "author's workstation" relative paths, and the principal run payloads (P64_s4 fields, spectra and handoff npz; c19 fields npz; ladder readout; held-drain summary) are absent from both the archive and SKIPPED_LARGE.tsv, and exist only on the compute box at the paths above.
- Where I looked: the git tree at f690b45, the three MANIFEST.tsv files, SKIPPED_LARGE.tsv, the three README.md files, the box run directories and their MANIFEST_*.md5 files, and the work-tree originals.
