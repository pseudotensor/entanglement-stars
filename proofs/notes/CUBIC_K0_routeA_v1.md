# CUBIC_K0, route A, v1: the MASSLESS limit of the cubic strain tensor K (lead #684)

**Frozen:** ADDENDUM_A10_vrc620_cubicK0_v1.md (7328a66f), frozen before any number.

**Blindness:** blind to the other route.

**Script and run:** vrc_k0_v1.py (0b6877cc), on box CPUs 12-19 with 8 threads (lead #686), nice 10, 560 s, no Traceback.

**Output:** vrc620/r8/k0_v1.json (545d48ba) and k0_v1.out (405bab58).

**Object and map.** kappa[a, i, b, j] = (1/N^3) sum_p q_i q_j c_a c_b (delta_ab - s_a s_b/eps^2)/(4 eps^3), with s = sin p, c = cos p, q = sin(2p)/2 and eps^2 = |s|^2 + m^2. This is vrc_kmid_v1's formula, evaluated in slabs. The symmetric map is:
- c11 = kappa_11,11;
- c12 = kappa_11,22;
- c44(sym) = (kappa_12,12 + kappa_12,21)/2, with K(S, S)/|S|^2 = 2 c44(sym) for S = (e1 e2 + e2 e1)/2.

**Self-check.** At m = 0.2 and N = 128 the slab code reproduces kmid_v1's formula_128 to <= 3e-18, for c11 0.00983418224815878, c12, kappa_12,12 and c44.

**Grid.** The midpoint grid p_i = -pi + (2i + 1) pi/N: odd multiples of pi/N, never on the doubler nodes {0, pi}.

## (A) m = 0 directly

| | N = 128 | N = 256 | N = 512 | N = 1024 |
|---|---|---|---|---|
| c11 | 0.010527683045775268 | 0.010519860927110788 | 0.010517906222327287 | 0.010517417597355022 |
| c12 = kappa_12,21 | -0.003912633007540469 | -0.0039087219662522375 | -0.003907744614141171 | -0.003907500301659429 |
| kappa_12,12 | 0.01629589227657693 | 0.016303694381736848 | 0.01630564524213124 | 0.016306132978029814 |
| c44(sym) | 0.006191629634518231 | 0.006197486207742305 | 0.006198950313995034 | 0.006199316338185193 |

**Fitted exponent of the algebraic convergence:**

| | p from (128, 256, 512) | p' from (256, 512, 1024) |
|---|---|---|
| c11 | 2.00061 | 2.00015 |
| c12 | 2.00060 | 2.00015 |
| kappa_12,12 | 1.99975 | 1.99994 |
| c44 | 2.00004 | 2.00001 |

So the error is 1/N^2, as expected from the 1/|p| integrand at the doubler nodes. The enclosure is not widened, since |p - p'| < 0.3.

**Richardson from (512, 1024) with p', and the frozen enclosure** K_inf +- (|K_inf - K_1024| + |K_inf - K_inf(p)|):

| | K_inf (m = 0) | ENCLOSURE (A) |
|---|---|---|
| **c11** | **0.010517254745127136** | [0.010517091824069635, 0.010517417666184636] |
| **c12** | **-0.00390741887551772** | [-0.003907500335634016, -0.0039073374154014245] |
| kappa_12,12 | 0.01630629556590763 | [0.016306132950141997, 0.016306458181673266] |
| kappa_12,21 | = c12 | = c12 |
| **c44(sym)** | **0.006199438345193912** | [0.006199316335122994, 0.00619956035526483] |

These are 16 pi G K_L at G/a^2 = 1/4 (MASSLESS): uniaxial 4 pi c11 = 0.13216 and pure shear 8 pi c44(sym) = 0.15580.

## (B) Cross-check, m a -> 0 (N = 512 and 1024; every m grid-converged to 1e-3)

| m | c11 (N 1024) | c12 | kappa_12,12 | c44(sym) |
|---|---|---|---|---|
| 0.1 | 0.010331114859869957 | -0.0035828590468026597 | 0.015542631336818604 | 0.005979886145007972 |
| 0.05 | 0.010469286909553191 | -0.003797612539265702 | 0.01605746052273604 | 0.0061299239917351695 |
| 0.025 | 0.01050514277115763 | -0.0038726974790236402 | 0.016229500263347272 | 0.006178401392161816 |

**Frozen m -> 0 extrapolations and their verdict:**

| | linear in m (0.05, 0.025) | quadratic in m (0.1, 0.05, 0.025) | inside (A)? |
|---|---|---|---|
| c11 | 0.010540998632762068 | 0.010518845190603945 | NO |
| c12 | -0.003947782418781578 | -0.00392625454779919 | NO |
| kappa_12,12 | 0.016401540003958503 | 0.01634462343572685 | NO |
| c44(sym) | 0.006226878792588463 | 0.00620918444396383 | NO |

**FROZEN FINAL ENCLOSURE** (A10: widened to cover the m-extrapolations because they are inconsistent):

| | final enclosure (frozen rule) |
|---|---|
| c11 | [0.010517091824069635, 0.010540998632762068] |
| c12 | [-0.003947782418781578, -0.0039073374154014245] |
| kappa_12,12 | [0.016306132950141997, 0.016401540003958503] |
| c44(sym) | [0.006199316335122994, 0.006226878792588463] |

## Post-hoc diagnosis (NOT frozen; labelled): why (B) disagrees with (A)

The m-sequence approaches (A)'s m = 0 value monotonically. The deviation is non-analytic, about m^2 ln(1/m), so polynomial-in-m extrapolations are the wrong model.

**Measured.** Successive ratios of [K(m) - K_inf(A)], for m 0.1/0.05 and then 0.05/0.025:

| | ratio 0.1/0.05 | ratio 0.05/0.025 |
|---|---|---|
| c11 | 3.88 | 3.96 |
| c12 | 2.96 | 3.16 |
| kappa_12,12 | 3.07 | 3.24 |
| c44 | 3.16 | 3.30 |

**Model.** A pure m^2 law gives ratio 4. For m^2 ln(1/m) the ratio is 4 ln(0.05)/ln(0.025) = 3.25 at the second step, and it rises toward 4. That matches c12, kappa_12,12 and c44.

**Check.** A pure-m^2 extrapolation from (0.05, 0.025) lands 1.6e-7 (c11) to 1.9e-5 (kappa_12,12) from (A).

**Reading.** The m = 0 grid calculation (A) is the direct and well-converged object: exponent 2.000 on both triplets, enclosure +-1.6e-7 (about 1.5e-5 relative), and the m-sequence converges into it. The widening in the frozen final enclosure comes from A10's choice of a polynomial-in-m cross-check model, not from any inconsistency of the m = 0 values.
- **I recommend quoting (A)'s enclosure**, with the frozen widened one printed beside it. The choice is the lead's.

## SELF RED-TEAM

1. (A) is the lattice formula at exactly m = 0 on a node-avoiding grid. Its error model (pure 1/N^2) is verified to an exponent of 2.000. A log(N)/N^2 component would show as drift in p between the triplets; the measured drift is <= 5e-4.
2. The m = 0 object is the massless naive cubic Dirac parent with its doublers. It is not the radial chain, and not a continuum value.
3. kappa_12,21 = c12 exactly, as at m = 0.2: a structural identity.
4. The post-hoc m^2 ln m diagnosis was not frozen. It explains, but does not replace, the frozen widening.
