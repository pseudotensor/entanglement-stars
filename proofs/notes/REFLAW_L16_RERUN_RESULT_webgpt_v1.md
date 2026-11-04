# REFLAW L16 RERUN RESULT (route2, vref_r1), v1: lead #561

**Label.** VERIFIED INVERSE, PAPER-LAW LIGHT WEIGHT 32, AT THE STORED IMPLEMENTED-LAW (WEIGHT-16) STATES.

This certifies invertibility of the weight-32 operator at these same 159 stored ON states. It does NOT certify corrected (weight-32) trajectories.

**Frozen before the run.** REFLAW_L16_RERUN_ADDENDUM_v1.md, md5 3a088a36b9573d5d8806af3499318c83.

## Setup

| Item | Value |
|---|---|
| Verifier | mark3_r2's singq_vinv_v5 (4eff3368), run as a box copy, singq_vinv_v5_l32box.py 941ccba9 (only the GPU lock path and sys.path differ) |
| Spec | spec_on_l32_v1.txt ea7ccf73 |
| Records | the 159 ON records of singqA_r1/singq_vinv4 |
| Light weight | 32 (spec column 6); the light weight enters only T, since B is weight-free (mark3) |
| Compute | box CPU, 12 workers, nice 19; fp64 GPU step on box GPU 2 |
| Wall | 14:13:34Z to 17:08:41Z, 31 Oct 2026; DONE, 159 rows, all 12 workers rc 0 |

**Spec identity vs mark3's original.** mark3's vinv_specs_all.txt (c6ff9815) has the same 159 labels and t strings; column 6 differs only 16 -> 32.
- All 159 rec files are byte-identical (md5 = pod3_stage/vinv1.md5).
- 5 boundary records resolve to the neighbouring segment's copy of the same step file, with the same md5.

## Result

**Table** (compare_l32_vs_l16_v1.json, md5 995a5ac78e17d182bdc700edcd0140f0)

| | weight 16 (singq_vinv4) | weight 32 (this rerun) |
|---|---|---|
| accepted (e < 1) | 159/159 | 159/159 |
| max beta_R (verified bound on \|\|Ft^-1\|\|_b) | 2.8964 (t = 0) | 3.0805 (t = 0) |
| max e | 5.95e-8 | 6.93e-8 |
| ratio beta32/beta16 over the records | - | 1.0023 to 1.0636 |
| tangent validation | valid | valid at all 159 |

\|\|X\|\|_b = sqrt(\|\|X\|\|_1 \|\|X\|\|_inf) on the raw site fields (Nb, vb). Ft = I + B T_W, with T_W now carrying the light polarization block at weight 32.

**Worst records** (largest beta_R, ratio and e, all the same records):

| Record | beta_R w16 | beta_R w32 | ratio | e (w32) |
|---|---|---|---|---|
| on_0.0000 | 2.8964 | 3.0805 | 1.0636 | 6.93e-8 |
| on_9.1929 | 2.8962 | 3.0803 | 1.0636 | 6.93e-8 |
| on_18.3857 | 2.8956 | 3.0797 | 1.0636 | 6.93e-8 |
| on_27.5786 | 2.8947 | 3.0786 | 1.0635 | 6.92e-8 |
| on_36.7714 | 2.8934 | 3.0772 | 1.0635 | 6.91e-8 |

The release-time end carries the largest bound and the largest change. As reviewer B noted, t_L16(u_star) = 0 but T_L16(u_star) != 0.

## Scope

- **Not affected:** OFF (light weight 0) and C_M1nl23 (no light chain in T) are unaffected by the normalization, so not rerun.
- **What is not certified:** this verifies the weight-32 Jacobian at the stored implemented-law states. Corrected weight-32 histories are not certified. Their departure from the stored ones is quantified at first order in REFLAW_L16_AUDIT_webgpt_v1.md (b1f54b4d): it exceeds the step-control spread by 1.5 to 4.4 orders of magnitude.

## Files

Box <internal>/l16:
- vinv32/vinv32_w0..w11.jsonl (per-record rows);
- vinv32/compare_l32_vs_l16_v1.json;
- scripts (l32_run_v1.sh ee06898d, l32_status_v1.sh fcf5b187, l32_compare_v1.py, l32_specdiff_v1.py).
