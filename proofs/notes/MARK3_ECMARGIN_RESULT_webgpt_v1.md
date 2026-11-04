# MARK3 route 2 (route2): energy-current margin, #583 RESULT NOTE (closes #583), 2026-11-01T01:09Z (true UTC). mark3_r2.

All labels are DIAGNOSTIC, not a certificate. Implemented law: light W_2 weight 16 (paper law 32; reviewer B reflaw). C_M1nl23 is L0. Same blindness throughout: route-1 data was used only as the raw inputs granted under #547/#584 and stayed on the box. Every output listed is on the box under singqA_r1/ecmargin_v1.

- Spec: MEASURE_EnergyCurrentMargin_spec_v1.md (bb813120).
- Addenda, each frozen before its numbers:
  - v1 63132779
  - v1.1 5780115a
  - v1.2 1aca7be4
  - v1.2.1 9eb719c8 (the weights_only allowlist of 6 numpy entries, lead-approved; 4 dated amendments)
  - v1.3 ecd2c664
  - v1.4 94202ba7

## Part A: local inequality |J| <= nu3 rho, 573 records (twins ON, twins OFF, C_M1nl23)
- **Outputs:** ecmA_summary_v1.json 24c6c908, ecmA2_bulk_summary_v1.json 438509a7, ecmA3_summary_v1.json ea1f5da2.
- **Code:** singq_ecmA_v1/v2/v3 = 7b1b6027 / cc7758ce / 7694ae8f.
- **Where it fails:** the inequality fails at every record, but only at y >= 352. The size is <= 1.8e-14 for the twins and 7.9e-13 for C_M1nl23.
- **Interior hypothesis:** it holds at every node y < 352, both untrapped and trapped.
- **Envelope share:** the envelope part of the ln N drop is <= 0.18% for the twins and 1.05% for C_M1nl23.
- **Split-bound floor:** finite at all 573 records. The worst floors are 0.0311 for ON and for OFF (actual min N 0.110), and 0.0200 for C_M1nl23 (actual 0.0716).

## (H') add-on, lead #585 (Theorem 3', mean enclosed density), 573 records
- **Outputs:** ecmC_summary_v1.json dcaff30e. Code: singq_ecmC_v1 ae90ad65.
- **Identity (1.1):** every slice is admitted, i.e. |w| = sqrt(2GM/r) exactly. So (1.1) is the rows' own law; its finite-difference residual is <= 4.0e-4 relative, with the sign as written.
- **c = max |J_l| / rho_bar** on the untrapped current support: 0.979 (ON), 0.978 (OFF), 1.291 (C_M1nl23). The maximum always sits in the core, at r 2.75 to 17.
- **(H) has no useful floor:** the current's support reaches the grid edge, where rho_bar falls to 6e-11.
- **The bound holds in 573/573 records**, against the actual untrapped drop (max 1.64). Three readings of r:
  - Literal B1 (r = 0.25): max 50 (twins) / 66 (C_M1nl23).
  - Integral form B2: max 6.1 / 6.0 / 7.8, giving floors exp(-B2) >= 4.3e-4.
- **Caveat:** the trapped stretch, which makes up part of the total drop L (max 2.21 / 2.64), is outside Theorem 3'.

## Part B, twins: sector split at 65 stored FULL states
- **Groups:** ON eqC 20; OFF 17 at matching t; pod_archive v1, 14 ON (t 0-781) and 14 OFF (t 0-809).
- **Rows:** ecmB_v3.jsonl 905ac652 (tranche 1) and ecmB_v3_t2.jsonl 4dfdb1be (tranche 2).
- **Summaries:** ecmB3_summary_t1_v1.json b2c230ec, ecmB3_summary_t2_v1.json ca97f3f7, ecmB3_summary_all_v1.json e6d58bf7.
- **Code:** singq_ecmB_v3 4fb08ba3, singq_ecmB_sum_v2 adb5a293.
- **Checks:** the closure (occ + coh - t against the stored sources) is <= 2.3e-11 at t >= 864 and <= 4.2e-8 at t 0-809. The P_star band check is <= 1.8e-14.
- **BULK y < 352, untrapped:** the total margin is POSITIVE at every node of all 65 states, carried by the occupation sector.
  - Total min: +6.2e-10 (t >= 864); +2.2e-14 at t = 0 (C(0) is close to P_star), rising to about 3e-9 by t 300-800.
  - coh and pol are each negative at many nodes (down to -1e-4 / -4.5e-4) but cancel. No bulk node fails the total.
- **ENVELOPE y >= 352, history ON (t >= 367):**
  - The total fails at -6e-15 to -1.8e-14 absolute. That is small but ABOVE Part A's stated evaluation error (eps_rho <= 4.5e-16, eps_J <= 8.3e-17).
  - It is driven by the band-coherent sector: coherent magnitudes are 6e-15 to 2e-14 at the failing nodes.
  - It lies in the near-vacuum tail (y about 500-900 out to the grid edge, |W0 rho| <= 4e-14, against a bulk peak of about 3e-6 at t < 800 and 3.5e-3 to 6e-2 at t >= 864). There, at the worst node, |J| is about 2.8 rho.
- **ENVELOPE y >= 352, history OFF (all t except t = 73.5):**
  - The total failure is <= 2.2e-17. That is below eps, i.e. within evaluation error.
  - So the above-eps envelope failure is ON-only, which ties it to the light-chain coherence.
- **Early-state exceptions:**
  - t = 73.5, in BOTH histories (-sign): total -4.7e-15 (ON) / -7.4e-15 (OFF). This failure is OCCUPATION-driven: occ margin -6.3e-15 / -6.4e-15, at about 1.6-2% of the local scale (|W0 rho| up to 9e-13), at the inner envelope edge (y >= 363).
  - t = 0: roundoff only (-4e-20).
  - ON t 147-294: the envelope total is >= -7e-17, i.e. within or near eps. The ON coherent failure above eps sets in by t = 367 and persists to 1452.
- **Occupation sector, everywhere except t = 73.5:** its own negative margin (<= about 4e-19) is below eps, i.e. roundoff.

## Part B, C_M1nl23 (granted by #584): 12 steps, t 864-1984
- **Outputs:** ecmB_r1_v2.jsonl 50b9d25f, ecmB_r1_summary_v1.json af31f0a8. Code: singq_ecmBr1_v2 253dd94f.
- **P_star:** the sea projector of h(u*), with u* = Fields(x*). Its band equals the file's Pb to 4.1e-14. The file's Z is the run's step-0 state, not P_star.
- **Closure:** occ + coh - t reproduces x to <= 4.0e-11. With +t the error is 0.09 and with no t it is 0.045, so the polarization term is required, with the subtracted sign.
- **BULK:** the total is positive at every step (min 1.3e-10), carried by the occupation sector (min 1.7e-10). coh and pol are each negative to about -1e-4 / -4e-4 but cancel.
- **ENVELOPE:** the total fails at every step, at -1.2e-14 (t 864) shrinking to -2.4e-15 (t 1984). That is small but ABOVE eps.
  - It is driven by the band-coherent sector: |coh| is 1.8e-15 to 1.6e-14 at the failing nodes.
  - It sits in the near-vacuum tail (|W0 rho| <= 4e-14 at the failing nodes). At late steps it concentrates in a shell, y about 1280-1740.
  - Envelope characterization: ecmB_envfail_v1.jsonl 64db2ea2.

## One-line result
On every stored state of both route-2 histories and C_M1nl23:
- **Bulk:** the averaged two-sided margin is positive in the bulk, and the occupation sector alone carries it.
- **Envelope:** the only above-eps failures are in the near-vacuum envelope at 1e-15 to 1e-14 absolute. They come from the band-coherent current in the ON / C_M1nl23 histories, plus one occupation-driven transient at t = 73.5 in both twins.
- **Clock bound:** with the (H') constant c <= 1.29, the mean-density clock bound is finite and holds at all 573 records.
