# Vertex reference contractions (VREF), route 2 (solve), v1

Spec: lead_gaps/AstraAPI_vref_v1 (SIGNED). This route is the tangent/adjoint rows and spectral response. It ran blind to route2's route 1.

Files (lane `results/vref_r2/`):
- NOTES_vref_r2.md (md5 e55213c854c46a0b2c264ca0cc8811a6)
- vertex_control.json (md5 b6f5a8866f098cbe1a18d7e8a69383f3)

Box work: <internal>/vref_r2. Single route.

## Slices
- **11.0:** no stored slice exists. It was rebuilt exactly by replaying 4 recorded steps backward from the stored 11.5 slice (W5 record 88). Sources match the record to 2.5e-11.
- **15.0:** no stored c19 B2n4096 slice exists. It uses the release route-2 exact backward rebuild from the stored T = 100 slice (record 120, source-checked to 7.6e-11).
  - These are 4096-site chains, so the spec's 2048-site identity is not met. This is labelled.
  - It was preferred to W5 at 14.0, which has 9 crossed occupations and so is excluded by the gapped-ground-reference condition.

## Gates: PASS on both slices
- Weighted Tr(C-bar - P) = 640.0 exactly.
- The reference is the negative projector with no crossings ([h, P] ~ 1e-12).
- The Hartree state is pure (Slater).

## Reference energy and dressing

| | 11.0 | 15.0 |
|---|---|---|
| F/E | -8.06e-4 | -4.21e-3 |
| F single pair | -0.0946 | -0.4576 |
| F double pair (direct) | -3.9e-4 | -1.1e-4 |
| exchange (bounded only) | <= 5.1e-7 | <= 1.3e-7 |
| Z_2 (dressing norm) | **0.136** | **1.095** |
| min single denominator | 0.271 | 0.019 |

**Verdict: the reference-correlation test FAILS at both slices.**
- F/E is small (below 1 %), but the first-order dressing norm Z_2 fails the 1 % gate.
- Nearly all of Z_2 is the single-pair term: the Hartree field of C-bar - P acting between the reference's occupied and empty levels.
- The mixed rho-J (sqrt G) block dominates the dressed vertex, so a G^2-only truncation would be wrong.

## Branch audit
- **<B>:** 0.0044 at 11.0 (only the R50 mass exists there). At 15.0 it is **0.0817**: the branch is non-empty (weight >= 0.918), but it marginally fails to reproduce the paper's 0.080 bound. The inner root at r = 7.0 contributes 0.074.
- **Mass bias:** beta/M <= 3.6e-4 (PASS).
- **Hole identity:** holds to 2.5e-7 (H = 0.0074 at 11.0, 0.51 at 15.0).
- **Caveat:** the mass Hessians omit one higher-order reference-response term, which was not estimated.

## Unresolved
- **Leakage:** the global certificate at 11.0 is ~2.7e9, useless as a bound. The 15.0 version was stopped.
- **Signed exchange:** bounded only.
- **Source derivatives:** at 11.0 only two probe directions exist, implying a field shift of ~5e-5 relative (indicative only). Nothing was computed at 15.0.

## Worker slips
- One box-side python -c one-liner.
- Concurrent processes at times exceeded the 6-thread cap.
- One GPU out-of-memory crash, recovered.
