# VREF route 1 (route2), v1 addendum: the R_50 mass term at T/r_s 15.0 under both R_50 definitions (lead #328, #330)

**Scope.**
- Reference collapse on the doubled domain B2n4096, record 120 (T/r_s 15.0).
- VREF route 1's own replayed state (s13), ten channels, with complete second-order masses.
- This addendum changes nothing in VREF_R1_v1.md. It adds the R_50 term at a second radius.

**Finding: the routes agree on the state.**
- The entire <B> difference between the routes (0.0798 vs 0.0817) is the choice of R_50.
- On the same state, with the same machinery, route 1 reproduces the other route's radius (27.0a), mass (76.83), term (0.00675) and total (0.0817).
- No state, mass-operator or norm difference remains at the shown digits.

## 1. The two R_50 definitions (same replayed state; s32_b2_r50_v1.py, r50_s32_v1.json)

Both definitions:
- use the dust columns only, Z_k[:, n_sea:] of the replayed T15 panels, with copy weight 16k, k = 1..10;
- use no sea and no pairs;
- take a per-site weight on shells R_n = n a, accumulate it cumulatively, and find half the total by linear interpolation between shells;
- represent the result at the nearest 0.1a grid row.

The two differ only in the weight:

| | weight per site n | total | half radius | represented row |
|---|---|---|---|---|
| **Number-weighted** (route 1, VREF_R1_v1) | sum_k 16k sum_dust \|psi_c(n)\|^2 (both spinor components) | 640.000 | **28.8545a** | 28.9a |
| **Energy-weighted** (bare dust energy) | sum_k 16k Re sum_dust psi_c^dag Pi_n h0_k psi_c | 367.233 | **27.0072a** | 27.0a |

**Paper text, PAPER_flat_3f3f40a2fc3.tex:**
- Lines 1866-1872: "The compression is read on the dust's half-mass radius \(R_{50}\): ... on the \(\sigma/a=4\) history, which stores no slice at 13 to 14, its compression minimum unsampled, it is 31.4, 28.9 and \(31.4a\) at 12, 15 and 16."
  - The printed 28.9a at 15 equals the number-weighted value.
  - The phrase "half-mass" does not by itself fix the weight: rest mass (number times m a) or bare energy.
- Lines 1981-1982: "the represented masses at the nearest \(0.1a\) grid rows (Sec.~\ref{sec:model-operators})". This is the rounding rule both radii follow.

## 2. Branch moments at both radii (complete K_j; reordered about C-bar)

| | R_50 = 27.0a (energy-weighted) | R_50 = 28.9a (number-weighted) |
|---|---|---|
| M | 76.82817 | 80.77506 |
| beta/M | 0.0003631 | 0.0003725 |
| sqrt(S_1)/M | 0.04104 | 0.03477 |
| sqrt(S_2)/M | [0.001469, 0.001501] | [0.001289, 0.001318] |
| 4(beta^2 + S_1 + S_2)/M^2 | [0.006746, 0.006746] | [0.004843, 0.004843] |
| + r_in term (7.0a, unchanged) | [0.074210, 0.074211] | [0.074210, 0.074211] |
| + R_t term (51.8a, unchanged) | [0.0007496, 0.0007497] | [0.0007496, 0.0007497] |
| **<B> total** | **[0.081705, 0.081706]** | **[0.079802, 0.079803]** |

**Implication for the 0.080 bound.**
- With the number-weighted R_50, <B> stays just below 0.080, as in VREF_R1_v1.
- With the energy-weighted R_50, <B> = 0.0817, which EXCEEDS 0.080 at T/r_s 15.0.
- So the paper's claim <B> <= 0.080 at this slice holds or fails depending on the R_50 definition.

## 3. Construction and checks

**The 27.0a mass, built exactly as h14 + h15 built the 28.9a mass (p60, H100).**
- M is from the discrete admitted solve. It equals the engine's Misner-Sharp mass to 5e-11 relative.
- The window identity holds to 5.0e-13, and S' reproduces h11 to 1.1e-15.
- G = S^T M' (S = I - chi V), with |G - M'|/|M'| = 0.0222.
- K0 = S^T Hess(M - lambda . u) S is built on the GPU.
- Xi comes from Zolotarev poles per channel (tol 1e-13). The relative size of V Xi V against K0 is 0.039 (max) and 0.057 (Frobenius).
- K was transferred as an upper triangle. The full matrix was rebuilt on the box and its sha256 matched the pod's.

**The box CPU pass (s61) used s40c's formulas unchanged.**
- beta uses s40c's D-kernel.
- S_1 is exact, with L = X[G] - X_K(D).
- S_2 is bracketed: direct with Pi^C, exchange bounded by the unweighted channel sum Pi_u.
- Re-evaluated with K_R50 at 28.9a, the pass reproduces s40c's beta and S_2 bracket exactly.

**Independent G-only pass (s59).**
- It gives the same M at both radii and reproduces h14's G at 28.9 to 7e-16.
- The X_K(D) part of L changes S_1 by only 0.11% at 28.9 (complete S_1 = 7.887199, matching s40c to 8e-14).
- At 27.0, S_1 is 9.9305 from G alone and 9.9406 complete.

**Unchanged.** The r_in and R_t terms are those of s40c; their radii do not depend on R_50.

## 4. Record

| File | md5 |
|---|---|
| box out/b2_t15/branch_r27_s61_v1.json | 739ed83cde2cae656f439d5f95990751 |
| box out/b2_t15/r50_twodefs_s59_v1.json | 92e9926e903d07f1ae135ac5c9dc3fb5 |
| box out/b2_t15/r50_s32_v1.json (both definitions) | 7e730e0b1cb81fb2fdae4706c5f5b349 |
| box out/b2_t15/branch_s40c_v1.json (28.9a, r_in, R_t) | 4ecc2933530a8930ef293af3e205db41 |
| box out/b2_t15/from_h100_b/K27_p60_v1.json (pod out/b2_t15/; K full sha256 inside) | e6d089507e8e71f3f0868c32b6a4ff5d |
| pod out/b2_t15/K27_triu_p60_v1.npy (sha256 0ac1b460...2496, matched on the box) | - |
| scripts p60_b2_K27_v1.py / s61_b2_branch_r27_v1.py / s59_b2_r50_twodefs_v1.py | 8de2f7482103b2c01f7af5d6a88b1961 / bd4e8084b06c426e320c591689eabe92 / 269171b228f705d71de35811d78edb6c |

**Self red-team.**
- **Could the 27.0 numbers be the 28.9 K or G mislabelled?**
  - M differs by 4 units.
  - K27's sha256 differs from K_R50's, and each matched on transfer.
  - G at 27.0 agrees between s59, s61 and p60 (|G - M'|/|M'| equal to 1e-14).
- **Could the S_2 bound kind differ between the radii?** It is the same unweighted channel bound at both radii.
- **Could the energy weight be computed differently from the other route's?** Its half radius (27.0072a) rounds to their 27.0a row. Their M (76.83) and term (0.00675) match this route's 76.828 and 0.006746.
- **Not checked:** which definition the paper intends. Its printed 28.9 at 15 matches the number-weighted reading.
