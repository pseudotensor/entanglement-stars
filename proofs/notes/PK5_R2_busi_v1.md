# PK5 route 2 (busi, inbox #720): accessible-sector diagnostics, retained capacity (area vs volume)

Spec: deliveries/lead_gaps/AstraAPI_pk25_v1.md, Package 5 (P5.1-P5.10). Paper anchors are from PAPER_flat_3f3f40a2fc3.tex. I checked the formation-ladder sentence ("The 128 values use sigma/a=8,16; its well-filled width-4 solve fails") in PAPER_flat_6547f6066cd.tex: it is unchanged there.

Blindness: I opened none of route2's package 5 files, STATUS lines, code or box directories. I saw the directory name <internal>/webgpt_pk5_r1 in a listing of <internal>/CONS and did not enter it.

Compute: box only, CPU at 8 threads (nlwp 15; the env has two OpenBLAS pools). No GPU was used, so nothing competed with RELEASE M1 or REFDEF for a card.

Code: <internal>/pk5_r2, byte-identical with box <internal>/code. Every table below is emitted by the named script and was pasted from its output file, not retyped.

## Verdict

**ACCESSIBLE_SECTOR_NOT_IDENTIFIED.** The supplied records contain no admitted input sector S_0, no branch projector Pi and no E2-hat. The exact charge H_+ (spec charge A) is therefore BLOCKED. Under P5.1 and P5.10 the full accessible-capacity calculation is unresolved, and adding an area-sized store would still be a declaration.

The computable diagnostics give:

1. **Kinematic counts.**
   - On the represented radial chains (10 channels, 16k copies each), the root-interior Fock dimension R0 = M_A ln 2 grows linearly in n_in, the number of interior sites. This is sub-area, and it is a property of the fixed 10-channel truncation, not of the model.
   - On the cubic lattice the same interior holds 4 N_ball modes, so R0 and R1 are exactly volume order (kind 4, kinematic).
   - The tail of channels k > 10 on the chains is not zero. Its honest bound is the cubic count, which is volume order.
2. **Charge-resolved counts** (fixed N_A and a held-Killing-energy shell around the test-state charge).
   - Centred on the field test state, these hold a few percent of R0 and grow at most about linearly in x = r_Q/a over 32-64-128: every p_eff interval lies within [0.18, 1.05].
   - Centred on the saved states (clocks not formation-matched), p_eff spans [0.26, 2.17] across preparations. The width-4 failed branch alone is area-like. That spread is not a law.
   - They are finite-size diagnostics of a diagnostic charge (spec C), not a leading coefficient.
3. **State diagnostics** (saved slices).
   - The interior's Gaussian reduced state has an extensive number variance.
   - So P(N_A = n*) is small, and the pinching bound of P5.5B is dominated by p_omit log D_omit. A fixed-number sector carries almost none of the actual state.
4. **Evolution-uniform errors.**
   - The spatial store is not an operator-invariant subspace: the cross coupling lambda_A = ||(I-A) hbar A|| is 0.465-0.600 a^-1 on every counted record (sup 0.5778 over B2n4096).
   - The trace-distance budget therefore saturates at 1 within about 2 a of time after formation.
   - Uniformity of an area-sized count over the stored history is UNDECIDED: no bound below 1 exists.

So no area-sized accessible count is derived, which agrees with S5. The one controlled scaling statement is the volume-order kinematic count of the cubic interior. The decision row "volume-order omitted retained entropy excludes an area ceiling" is met only as a bound (S_O <= M_O ln 2 with M_O volume order), not as a computed entropy. Area scaling remains unsupported, and it is not excluded for an admitted sector, because none is supplied.

## P5.1 Objects (define_retained_problem)

- **A (candidate store):** all represented chain modes on sites R_n < R_t, both spinor rows, channels k = 1..10, each with g_k = 16k exact copies.
  - R_t is the outer geometric root of N - |v| on the record's node (ODE) fields, recomputed by linear interpolation.
  - The applied (back-averaged) field root is reported beside it. It is absent at 32, at 128 s16 and in the ISO control: the back-averaged fields hold no site root there.
- **O (other retained):** exterior chain modes (R_n >= R_t to the open end), channels k > 10, the light species (not represented on these records), the filled valleys (uncounted by the chain bookkeeping), and slow, bound or fountain content (no admitted classification).
- **E_out:** empty. No Package 2 input is used.
- **S_0:** not supplied, so ACCESSIBLE_SECTOR_NOT_IDENTIFIED. **Branch Pi:** not supplied. **Dynamics:** the held Hartree chain evolution over the stored records (exp(-i tau h_k[N_app, v_app]) per record).
- **Charge used:** spec C, the held Killing energy Q_C = c^dag X hbar X c, with hbar = h_k[N_applied, v_applied] as stored (never re-averaged) and q0 = 0 at the empty A modes.
  - Charge B (Q^(1) restricted as an enclosed mass) needs the continued-reference response. Charge A (H_+) needs E2-hat and Pi. Both are BLOCKED.
  - The difference between C and A is not measurable here: BLOCKED.
- **Number:** N_A. [Q_C, N_A] = 0 exactly, so the product traces are sector dimensions.
- **Sector rules** (declared before any size comparison; window centres fixed by the test state or saved state, never by an area fit):
  - R0: Fock(A).
  - R1: N_A = n*.
  - R2a: N_A = n* and |Q_C - q*| <= dq/2 with dq = 0.01 E.
  - R2b: as R2a with dq = 1e-4 E.
  - Centres (n*, q*): field rows (kind 1) use the held chain's negative-energy projector P_-, a test state (mask_equals_negative_set holds on the records). State rows (kind 2) use the saved covariance C = sea sea^+ + dust f dust^+.

## P5.3 Counting algorithm and the small parent

- **Upper bound (rigorous):** log D <= inf_{alpha,beta} [log Z + beta q + |beta| dq/2 - alpha n]. Log Z is evaluated at the worst case over per-level Kahan residual enclosures, and any (alpha, beta) gives a valid bound.
- **Lower bound (rigorous):** one occupation pattern (n_l) with sum n_l = n and the whole charge enclosure [sum n_l(eps_l - delta_l), sum n_l(eps_l + delta_l)] inside the window. The permutations among the g_l exact copies of each level keep (N, Q), so D >= prod binom(g_l, n_l).
- **Nested windows:** the R2a lower bound is raised to max(lb_R2a, lb_R2b), because the R2b window lies inside the R2a window.
- **Exact enclosure:** meet in the middle over occupation patterns, with no charge mesh (the per-configuration charge error is at most n max delta, never treated as a one-mode error). Brute-force Fock enumeration checks it.
- **Small parent** (size 32, j66, the smallest rooted member; size 16 has no root through T/r_s = 40 at sigma 2 or 4; script pk5_small_v1.py, out_small_v1/small_parent_v1.json):

Small-parent log, verbatim (pk5_small_v1.py stdout). Brute force equals the exact enclosure in every case, and the bounds contain the exact counts.

```
record j=66 T/rs=16.5000 E=58.785733 R_t(json)=21.8768 R_t(nodes, recomputed)=20.6508 n_in=20
store: M_A=35200 levels=400 n*=17600 (17600.0000) q*=-12923.202248 eps in [-3.8400, 3.8400] delta_max 5.88e-13  (21.6 s)
null: R0 = M log2 = 24398.780756; R1 log binom = 24393.320557; logsum_n binom - M log2 = 3.27e-11; Chernoff UB at beta-free 24398.780756
cluster16 dq=0.05 brute 272  mitm [272, 272]  bounds log [4.1589, 10.6311] = [64.0, 41403.6]
cluster16 dq=0.50 brute 2444  mitm [2444, 2444]  bounds log [4.1589, 10.8950] = [64.0, 53904.7]
cluster16 dq=2.00 brute 8651  mitm [8651, 8651]  bounds log [5.5452, 11.0904] = [256.0, 65536.0]
cluster160 R2a dq=0.5879 exact log D in [17.146795788588626, 17.146795788588626]  bounds [15.120160930043658, 21.5955]  (4 s)
cluster160 R2b dq=0.005879 exact log D in [None, None]  bounds [None, 17.7104]  (9 s)
cluster160 wide dq=5.879 exact log D in [42.10570587117496, 42.10570587117496]  bounds [35.678521934664914, 47.2993]  (13 s)
store16 R2a dq=0.5879 log D in [1306.7756132853071, 1571.4309]  (R0 24398.78, R1 24393.32)  alpha,beta=[7.88661548e-08 1.21665224e+01]  1.1 s
store16 R2b dq=0.005879 log D in [1303.9330988497857, 1567.8888]  (R0 24398.78, R1 24393.32)  alpha,beta=[7.20818041e-08 1.21785325e+01]  0.0 s
done
W5_64_s4 j=97 T/rs=12.1250 E=116.604413 rQ=58.3022 R_t nodes 38.4513 (json 38.55622097786145, applied 34.50910137378203) n_in=38
```

## P5.9 Optimization and cost (owner rules #698 and #715)

Levers applied:
- **Symmetry blocks:** per channel. The 16k copies are used analytically, as multiplicities.
- **Windowed spectrum:** the dense eigh is replaced by the interior rows of P_- = (1 - sign h)/2. These come from a sech-trapezoid in log t of banded Cholesky solves of h^2 + t^2 (pk5_lib3_v1.py).
- **Gap and count:** one eigenvalue-only banded call gives the gap and the negative count.

Real arithmetic is not available. The flow term v Q is odd under every antiunitary that keeps sigma1 p_r and sigma2 k/r, so complex128 is required.

Not applied:
- GPU: not needed. The counts themselves take about 1 s, and the rest is banded or I/O bound.
- Warm start and batching: not relevant, since each record is independent and small.

One-step A/B against the dense eigh (pk5_ab_proj_v1.py, then pk5_ab_proj_v2.py on the production setting):

```
ab1.log:W5_64_s4 j=97 n_in=38 dim=4096
ab1.log:k= 1 gap 0.07303 (eigh 0.07303) nneg 2048 (eigh 2048) nodes 285  max|dP| 2.05e-15  dTr 1.54e-14  dTr(HP) 1.97e-15  eigh 24.8 s  banded 10.7 s  speedup 2.3x
ab1.log:k=10 gap 0.14448 (eigh 0.14448) nneg 2048 (eigh 2048) nodes 287  max|dP| 1.22e-15  dTr 7.22e-16  dTr(HP) 8.45e-15  eigh 24.7 s  banded 10.9 s  speedup 2.3x
ab1.log:S4_128_s8 j=179 n_in=72 dim=8192
ab1.log:k= 1 gap 0.07161 (eigh 0.07161) nneg 4096 (eigh 4096) nodes 285  max|dP| 3.11e-15  dTr 2.66e-14  dTr(HP) 4.07e-14  eigh 199.6 s  banded 93.8 s  speedup 2.1x
ab2.out:W5_64_s4 j=97 n_in=38 dim=4096
ab2.out:k= 1 gap 0.07303 (eigh 0.07303) nneg 2048 (eigh 2048) nodes 181  max|dP| 4.71e-13  dTr 1.55e-14  dTr(HP) 4.18e-12  eigh 25.3 s  banded 3.4 s  speedup 7.4x
ab2.out:k=10 gap 0.14448 (eigh 0.14448) nneg 2048 (eigh 2048) nodes 183  max|dP| 1.07e-12  dTr 8.33e-16  dTr(HP) 1.46e-13  eigh 25.1 s  banded 3.4 s  speedup 7.3x
```

Per-member cost on the loaded box (load about 15 on 72 cores; 8 BLAS threads, but the banded LAPACK solves run on one core), pk5_counts_v3 stdout:

```
done
done S4_32_s4_b066 in 11 s
done S4_32_s4_j066 in 13 s
done W5_64_s4_b097 in 33 s
done W5_64_s4_j097 in 39 s
done S4_128_s8_b179 in 366 s
done S4_128_s8_j179 in 479 s
done S4_128_s16_b185 in 418 s
done S4_128_s16_j185 in 517 s
done S4_128_s4_b177 in 342 s
done S4_128_s4_j177 in 401 s
done ISO_128_s4_b184 in 517 s
done ISO_128_s4_j184 in 527 s
done W5_64_s4_j100_state in 84 s
done W5_64_s4_j104_state in 92 s
done W5_64_s4_j112_state in 94 s
done S4_128_s8_j184_state in 817 s
done S4_128_s8_j192_state in 906 s
done S4_128_s8_j208_state in 1036 s
done S4_128_s8_j224_state in 943 s
done ISO_128_s4_j184_state in 697 s
done S4_128_s4_j184_state in 812 s
done: 801 records, 47 store jumps, saturation {'t_from_formation': 1.7306559271442687, 'lambda_sup': 0.5778156040814456, 'T_over_rs_formation': 12.125}, 27 s
done: 113 records, 8 store jumps, saturation {'t_from_formation': 1.7306585640437855, 'lambda_sup': 0.5778147236988451, 'T_over_rs_formation': 12.125}, 0 s
```

Total: about 2.6 CPU-hours. The bound, not the arithmetic, is the limiting resource. The R2 enclosure width comes from the single-pattern lower bound, not from the solver.

## P5.7 Formation selection

All rows come from inputs_manifest.json and manifest_records.tsv (pk5_manifest_v2.py).

**Last root-free and first rooted records:**

| member | last root-free j | first rooted j | T/r_s |
|---|---|---|---|
| 32 | 65 | 66 | 16.5 |
| 64 W5 | 96 | 97 | 12.125 |
| 128 s8 | 178 | 179 | 11.1875 |
| 128 s16 | 184 | 185 | 11.5625 |
| width-4 well-filled branch | 176 | 177 | 11.0625 |
| ISO width-4 control | 183 | 184 | 11.5 |
| B2n4096 | 96 | 97 | 12.125 |

**Interpolated birth.** Fields are linearly interpolated between the two records, and the birth is the bisected zero of min(N - |v|) on the node grid. The store radius at birth is the double root R* at node-grid resolution (about 2 a): R* = 20, 32, 56, 64 and 80 for 32, 64, s8, s16 and ISO. That gives a plus-or-minus 1 site sensitivity in n_in.

**Families are never pooled.** The 128 s8 and s16 well-filled family, the failed width-4 well-filled branch (S7) and the isotropic width-4 control (a different, degeneracy-8 preparation) appear as separate rows and separate pairs.

**Size 16** (sigma 2 and 4) has no geometric root through T/r_s = 40, so it gives no store.

**Applied versus geometric roots.** The applied (coupled, back-averaged) fields hold no site root at 32, 128 s16 or the ISO control. Where they do, their outer root lies 0.57-9.08 a inside the geometric root (largest at the first rooted records).

**Covariates** (width, tau, channel count, preparation tag, certification extent) are carried in every row of p5_sector_counts.

## Sector counts, scaling and state diagnostics (script output, pasted)

The table headers below say "pk5_counts_v2.py"; that label is printed by pk5_tables_v1.py. The JSON rows themselves were produced by pk5_counts_v3.py, which is v2 run on pk5_lib3_v1 after the A/B. In the "sector counts" table the "R_t/a" column for birth rows is R*, and the "S(C_O) ext chain" column is the exterior of the root on the chains.

Table: sector counts at formation (script pk5_tables_v1.py from pk5_counts_v2.py outputs)

| member | kind | x = r_Q/a | R_t/a | n_in | rule | log D lower | log D upper |
|---|---|---|---|---|---|---|---|
| ISO_128_s4 birth b184 | field-only (interpolated birth field) | 116.891 | 80.000 | 79 | R0 Fock(A) [field:P_-] | 96375.1840 | 96375.1840 |
| ISO_128_s4 birth b184 | field-only (interpolated birth field) | 116.891 | 80.000 | 79 | R1 N_A=n* [field:P_-] | 96369.0369 | 96369.0369 |
| ISO_128_s4 birth b184 | field-only (interpolated birth field) | 116.891 | 80.000 | 79 | R2a dq=0.01E [field:P_-] | 3201.1320 | 3753.6524 |
| ISO_128_s4 birth b184 | field-only (interpolated birth field) | 116.891 | 80.000 | 79 | R2b dq=1e-4E [field:P_-] | 3182.1729 | 3727.6434 |
| ISO_128_s4 birth b184 | cubic background (kinematic) | 116.891 | 80.000 | 79 | R0 cubic site-centred | 5943351.6835 | 5943351.6835 |
| ISO_128_s4 birth b184 | cubic background (kinematic) | 116.891 | 80.000 | 79 | R1 cubic half filling site-centred | 5943343.4755 | 5943343.4755 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R0 Fock(A) [field:P_-] | 98815.0621 | 98815.0621 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R1 N_A=n* [field:P_-] | 98808.9025 | 98808.9025 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R2a dq=0.01E [field:P_-] | 3308.0898 | 3840.6567 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R2b dq=1e-4E [field:P_-] | 3262.3466 | 3814.1719 |
| ISO_128_s4 j184 | cubic background (kinematic) | 116.887 | 81.427 | 81 | R0 cubic site-centred | 6271692.7303 | 6271692.7303 |
| ISO_128_s4 j184 | cubic background (kinematic) | 116.887 | 81.427 | 81 | R1 cubic half filling site-centred | 6271684.4955 | 6271684.4955 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R0 Fock(A) [field:P_-] | 98815.0621 | 98815.0621 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R1 N_A=n* [field:P_-] | 98808.9025 | 98808.9025 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R2a dq=0.01E [field:P_-] | 3308.0898 | 3840.6567 |
| ISO_128_s4 j184 | field-only (P_- test-state centre) | 116.887 | 81.427 | 81 | R2b dq=1e-4E [field:P_-] | 3262.3466 | 3814.1719 |
| ISO_128_s4 j184 | saved state | 116.887 | 81.427 | 81 | R0 Fock(A) [state:C] | 98815.0621 | 98815.0621 |
| ISO_128_s4 j184 | saved state | 116.887 | 81.427 | 81 | R1 N_A=n* [state:C] | 98798.3324 | 98798.3324 |
| ISO_128_s4 j184 | saved state | 116.887 | 81.427 | 81 | R2a dq=0.01E [state:C] | 5688.4195 | 6405.1745 |
| ISO_128_s4 j184 | saved state | 116.887 | 81.427 | 81 | R2b dq=1e-4E [state:C] | 5683.0603 | 6384.2486 |
| ISO_128_s4 j184 | cubic background (kinematic) | 116.887 | 81.427 | 81 | R0 cubic site-centred | 6271692.7303 | 6271692.7303 |
| ISO_128_s4 j184 | cubic background (kinematic) | 116.887 | 81.427 | 81 | R1 cubic half filling site-centred | 6271684.4955 | 6271684.4955 |
| S4_128_s16 birth b185 | field-only (interpolated birth field) | 116.988 | 64.000 | 63 | R0 Fock(A) [field:P_-] | 76856.1594 | 76856.1594 |
| S4_128_s16 birth b185 | field-only (interpolated birth field) | 116.988 | 64.000 | 63 | R1 N_A=n* [field:P_-] | 76850.1255 | 76850.1255 |
| S4_128_s16 birth b185 | field-only (interpolated birth field) | 116.988 | 64.000 | 63 | R2a dq=0.01E [field:P_-] | 2488.3833 | 2920.2431 |
| S4_128_s16 birth b185 | field-only (interpolated birth field) | 116.988 | 64.000 | 63 | R2b dq=1e-4E [field:P_-] | 2465.0367 | 2898.3785 |
| S4_128_s16 birth b185 | cubic background (kinematic) | 116.988 | 64.000 | 63 | R0 cubic site-centred | 3044055.6566 | 3044055.6566 |
| S4_128_s16 birth b185 | cubic background (kinematic) | 116.988 | 64.000 | 63 | R1 cubic half filling site-centred | 3044047.7832 | 3044047.7832 |
| S4_128_s16 j185 | field-only (P_- test-state centre) | 116.937 | 79.931 | 79 | R0 Fock(A) [field:P_-] | 96375.1840 | 96375.1840 |
| S4_128_s16 j185 | field-only (P_- test-state centre) | 116.937 | 79.931 | 79 | R1 N_A=n* [field:P_-] | 96369.0369 | 96369.0369 |
| S4_128_s16 j185 | field-only (P_- test-state centre) | 116.937 | 79.931 | 79 | R2a dq=0.01E [field:P_-] | 3057.1554 | 3570.6206 |
| S4_128_s16 j185 | field-only (P_- test-state centre) | 116.937 | 79.931 | 79 | R2b dq=1e-4E [field:P_-] | 3023.1743 | 3545.2279 |
| S4_128_s16 j185 | cubic background (kinematic) | 116.937 | 79.931 | 79 | R0 cubic site-centred | 5931174.4738 | 5931174.4738 |
| S4_128_s16 j185 | cubic background (kinematic) | 116.937 | 79.931 | 79 | R1 cubic half filling site-centred | 5931166.2669 | 5931166.2669 |
| S4_128_s4 birth b177 | field-only (interpolated birth field) | 116.662 | 52.000 | 51 | R0 Fock(A) [field:P_-] | 62216.8909 | 62216.8909 |
| S4_128_s4 birth b177 | field-only (interpolated birth field) | 116.662 | 52.000 | 51 | R1 N_A=n* [field:P_-] | 62210.9627 | 62210.9627 |
| S4_128_s4 birth b177 | field-only (interpolated birth field) | 116.662 | 52.000 | 51 | R2a dq=0.01E [field:P_-] | 2577.3253 | 2975.1573 |
| S4_128_s4 birth b177 | field-only (interpolated birth field) | 116.662 | 52.000 | 51 | R2b dq=1e-4E [field:P_-] | 2560.2624 | 2953.6260 |
| S4_128_s4 birth b177 | cubic background (kinematic) | 116.662 | 52.000 | 51 | R0 cubic site-centred | 1632331.1117 | 1632331.1117 |
| S4_128_s4 birth b177 | cubic background (kinematic) | 116.662 | 52.000 | 51 | R1 cubic half filling site-centred | 1632323.5499 | 1632323.5499 |
| S4_128_s4 j177 | field-only (P_- test-state centre) | 116.626 | 61.705 | 61 | R0 Fock(A) [field:P_-] | 74416.2813 | 74416.2813 |
| S4_128_s4 j177 | field-only (P_- test-state centre) | 116.626 | 61.705 | 61 | R1 N_A=n* [field:P_-] | 74410.2635 | 74410.2635 |
| S4_128_s4 j177 | field-only (P_- test-state centre) | 116.626 | 61.705 | 61 | R2a dq=0.01E [field:P_-] | 3077.0756 | 3498.6520 |
| S4_128_s4 j177 | field-only (P_- test-state centre) | 116.626 | 61.705 | 61 | R2b dq=1e-4E [field:P_-] | 3043.6182 | 3474.2267 |
| S4_128_s4 j177 | cubic background (kinematic) | 116.626 | 61.705 | 61 | R0 cubic site-centred | 2728324.3433 | 2728324.3433 |
| S4_128_s4 j177 | cubic background (kinematic) | 116.626 | 61.705 | 61 | R1 cubic half filling site-centred | 2728316.5246 | 2728316.5246 |
| S4_128_s4 j184 | field-only (P_- test-state centre) | 110.651 | 90.153 | 90 | R0 Fock(A) [field:P_-] | 109794.5134 | 109794.5134 |
| S4_128_s4 j184 | field-only (P_- test-state centre) | 110.651 | 90.153 | 90 | R1 N_A=n* [field:P_-] | 109788.3012 | 109788.3012 |
| S4_128_s4 j184 | field-only (P_- test-state centre) | 110.651 | 90.153 | 90 | R2a dq=0.01E [field:P_-] | 9109.6224 | 9868.7209 |
| S4_128_s4 j184 | field-only (P_- test-state centre) | 110.651 | 90.153 | 90 | R2b dq=1e-4E [field:P_-] | 9077.2794 | 9825.8906 |
| S4_128_s4 j184 | saved state | 110.651 | 90.153 | 90 | R0 Fock(A) [state:C] | 109794.5134 | 109794.5134 |
| S4_128_s4 j184 | saved state | 110.651 | 90.153 | 90 | R1 N_A=n* [state:C] | 109774.4863 | 109774.4863 |
| S4_128_s4 j184 | saved state | 110.651 | 90.153 | 90 | R2a dq=0.01E [state:C] | 16606.2941 | 17770.1179 |
| S4_128_s4 j184 | saved state | 110.651 | 90.153 | 90 | R2b dq=1e-4E [state:C] | 16576.7112 | 17748.6727 |
| S4_128_s4 j184 | cubic background (kinematic) | 110.651 | 90.153 | 90 | R0 cubic site-centred | 8510829.8372 | 8510829.8372 |
| S4_128_s4 j184 | cubic background (kinematic) | 110.651 | 90.153 | 90 | R1 cubic half filling site-centred | 8510821.4497 | 8510821.4497 |
| S4_128_s8 birth b179 | field-only (interpolated birth field) | 116.762 | 56.000 | 55 | R0 Fock(A) [field:P_-] | 67096.6471 | 67096.6471 |
| S4_128_s8 birth b179 | field-only (interpolated birth field) | 116.762 | 56.000 | 55 | R1 N_A=n* [field:P_-] | 67090.6811 | 67090.6811 |
| S4_128_s8 birth b179 | field-only (interpolated birth field) | 116.762 | 56.000 | 55 | R2a dq=0.01E [field:P_-] | 2593.4236 | 2998.5479 |
| S4_128_s8 birth b179 | field-only (interpolated birth field) | 116.762 | 56.000 | 55 | R2b dq=1e-4E [field:P_-] | 2577.0006 | 2976.7398 |
| S4_128_s8 birth b179 | cubic background (kinematic) | 116.762 | 56.000 | 55 | R0 cubic site-centred | 2038581.9017 | 2038581.9017 |
| S4_128_s8 birth b179 | cubic background (kinematic) | 116.762 | 56.000 | 55 | R1 cubic half filling site-centred | 2038574.2287 | 2038574.2287 |
| S4_128_s8 j179 | field-only (P_- test-state centre) | 116.637 | 72.912 | 72 | R0 Fock(A) [field:P_-] | 87835.6107 | 87835.6107 |
| S4_128_s8 j179 | field-only (P_- test-state centre) | 116.637 | 72.912 | 72 | R1 N_A=n* [field:P_-] | 87829.5101 | 87829.5101 |
| S4_128_s8 j179 | field-only (P_- test-state centre) | 116.637 | 72.912 | 72 | R2a dq=0.01E [field:P_-] | 3401.5304 | 3891.2546 |
| S4_128_s8 j179 | field-only (P_- test-state centre) | 116.637 | 72.912 | 72 | R2b dq=1e-4E [field:P_-] | 3384.2805 | 3864.4900 |
| S4_128_s8 j179 | cubic background (kinematic) | 116.637 | 72.912 | 72 | R0 cubic site-centred | 4501411.4667 | 4501411.4667 |
| S4_128_s8 j179 | cubic background (kinematic) | 116.637 | 72.912 | 72 | R1 cubic half filling site-centred | 4501403.3977 | 4501403.3977 |
| S4_128_s8 j184 | field-only (P_- test-state centre) | 114.950 | 93.504 | 93 | R0 Fock(A) [field:P_-] | 113454.3305 | 113454.3305 |
| S4_128_s8 j184 | field-only (P_- test-state centre) | 114.950 | 93.504 | 93 | R1 N_A=n* [field:P_-] | 113448.1019 | 113448.1019 |
| S4_128_s8 j184 | field-only (P_- test-state centre) | 114.950 | 93.504 | 93 | R2a dq=0.01E [field:P_-] | 6557.1752 | 7236.3448 |
| S4_128_s8 j184 | field-only (P_- test-state centre) | 114.950 | 93.504 | 93 | R2b dq=1e-4E [field:P_-] | 6528.6631 | 7196.7050 |
| S4_128_s8 j184 | saved state | 114.950 | 93.504 | 93 | R0 Fock(A) [state:C] | 113454.3305 | 113454.3305 |
| S4_128_s8 j184 | saved state | 114.950 | 93.504 | 93 | R1 N_A=n* [state:C] | 113435.0124 | 113435.0124 |
| S4_128_s8 j184 | saved state | 114.950 | 93.504 | 93 | R2a dq=0.01E [state:C] | 10044.8171 | 10927.6278 |
| S4_128_s8 j184 | saved state | 114.950 | 93.504 | 93 | R2b dq=1e-4E [state:C] | 10020.3992 | 10899.1042 |
| S4_128_s8 j184 | cubic background (kinematic) | 114.950 | 93.504 | 93 | R0 cubic site-centred | 9495559.0833 | 9495559.0833 |
| S4_128_s8 j184 | cubic background (kinematic) | 114.950 | 93.504 | 93 | R1 cubic half filling site-centred | 9495550.6411 | 9495550.6411 |
| S4_128_s8 j192 | field-only (P_- test-state centre) | 111.558 | 100.655 | 100 | R0 Fock(A) [field:P_-] | 121993.9038 | 121993.9038 |
| S4_128_s8 j192 | field-only (P_- test-state centre) | 111.558 | 100.655 | 100 | R1 N_A=n* [field:P_-] | 121987.6389 | 121987.6389 |
| S4_128_s8 j192 | field-only (P_- test-state centre) | 111.558 | 100.655 | 100 | R2a dq=0.01E [field:P_-] | 10835.4731 | 11642.4973 |
| S4_128_s8 j192 | field-only (P_- test-state centre) | 111.558 | 100.655 | 100 | R2b dq=1e-4E [field:P_-] | 10793.7882 | 11593.7160 |
| S4_128_s8 j192 | saved state | 111.558 | 100.655 | 100 | R0 Fock(A) [state:C] | 121993.9038 | 121993.9038 |
| S4_128_s8 j192 | saved state | 111.558 | 100.655 | 100 | R1 N_A=n* [state:C] | 121972.4791 | 121972.4791 |
| S4_128_s8 j192 | saved state | 111.558 | 100.655 | 100 | R2a dq=0.01E [state:C] | 18416.4286 | 19722.1831 |
| S4_128_s8 j192 | saved state | 111.558 | 100.655 | 100 | R2b dq=1e-4E [state:C] | 18396.8134 | 19696.8469 |
| S4_128_s8 j192 | cubic background (kinematic) | 111.558 | 100.655 | 100 | R0 cubic site-centred | 11844357.6194 | 11844357.6194 |
| S4_128_s8 j192 | cubic background (kinematic) | 111.558 | 100.655 | 100 | R1 cubic half filling site-centred | 11844349.0667 | 11844349.0667 |
| S4_128_s8 j208 | field-only (P_- test-state centre) | 110.600 | 106.479 | 106 | R0 Fock(A) [field:P_-] | 129313.5380 | 129313.5380 |
| S4_128_s8 j208 | field-only (P_- test-state centre) | 110.600 | 106.479 | 106 | R1 N_A=n* [field:P_-] | 129307.2440 | 129307.2440 |
| S4_128_s8 j208 | field-only (P_- test-state centre) | 110.600 | 106.479 | 106 | R2a dq=0.01E [field:P_-] | 14056.9997 | 15106.2670 |
| S4_128_s8 j208 | field-only (P_- test-state centre) | 110.600 | 106.479 | 106 | R2b dq=1e-4E [field:P_-] | 14004.1414 | 15050.6970 |
| S4_128_s8 j208 | saved state | 110.600 | 106.479 | 106 | R0 Fock(A) [state:C] | 129313.5380 | 129313.5380 |
| S4_128_s8 j208 | saved state | 110.600 | 106.479 | 106 | R1 N_A=n* [state:C] | 129290.9454 | 129290.9454 |
| S4_128_s8 j208 | saved state | 110.600 | 106.479 | 106 | R2a dq=0.01E [state:C] | 17889.1114 | 19210.0926 |
| S4_128_s8 j208 | saved state | 110.600 | 106.479 | 106 | R2b dq=1e-4E [state:C] | 17866.0653 | 19169.6165 |
| S4_128_s8 j208 | cubic background (kinematic) | 110.600 | 106.479 | 106 | R0 cubic site-centred | 14020318.5197 | 14020318.5197 |
| S4_128_s8 j208 | cubic background (kinematic) | 110.600 | 106.479 | 106 | R1 cubic half filling site-centred | 14020309.8826 | 14020309.8826 |
| S4_128_s8 j224 | field-only (P_- test-state centre) | 110.233 | 108.187 | 108 | R0 Fock(A) [field:P_-] | 131753.4161 | 131753.4161 |
| S4_128_s8 j224 | field-only (P_- test-state centre) | 110.233 | 108.187 | 108 | R1 N_A=n* [field:P_-] | 131747.1127 | 131747.1127 |
| S4_128_s8 j224 | field-only (P_- test-state centre) | 110.233 | 108.187 | 108 | R2a dq=0.01E [field:P_-] | 9393.9108 | 10402.4522 |
| S4_128_s8 j224 | field-only (P_- test-state centre) | 110.233 | 108.187 | 108 | R2b dq=1e-4E [field:P_-] | 9347.0792 | 10345.6984 |
| S4_128_s8 j224 | saved state | 110.233 | 108.187 | 108 | R0 Fock(A) [state:C] | 131753.4161 | 131753.4161 |
| S4_128_s8 j224 | saved state | 110.233 | 108.187 | 108 | R1 N_A=n* [state:C] | 131730.4872 | 131730.4872 |
| S4_128_s8 j224 | saved state | 110.233 | 108.187 | 108 | R2a dq=0.01E [state:C] | 12619.4829 | 13820.8494 |
| S4_128_s8 j224 | saved state | 110.233 | 108.187 | 108 | R2b dq=1e-4E [state:C] | 12584.2923 | 13774.7519 |
| S4_128_s8 j224 | cubic background (kinematic) | 110.233 | 108.187 | 108 | R0 cubic site-centred | 14703661.8265 | 14703661.8265 |
| S4_128_s8 j224 | cubic background (kinematic) | 110.233 | 108.187 | 108 | R1 cubic half filling site-centred | 14703653.1656 | 14703653.1656 |
| S4_32_s4 birth b66 | field-only (interpolated birth field) | 29.397 | 20.000 | 19 | R0 Fock(A) [field:P_-] | 23178.8417 | 23178.8417 |
| S4_32_s4 birth b66 | field-only (interpolated birth field) | 29.397 | 20.000 | 19 | R1 N_A=n* [field:P_-] | 23173.4072 | 23173.4072 |
| S4_32_s4 birth b66 | field-only (interpolated birth field) | 29.397 | 20.000 | 19 | R2a dq=0.01E [field:P_-] | 1217.0544 | 1479.6470 |
| S4_32_s4 birth b66 | field-only (interpolated birth field) | 29.397 | 20.000 | 19 | R2b dq=1e-4E [field:P_-] | 1216.3613 | 1476.1552 |
| S4_32_s4 birth b66 | cubic background (kinematic) | 29.397 | 20.000 | 19 | R0 cubic site-centred | 92524.0582 | 92524.0582 |
| S4_32_s4 birth b66 | cubic background (kinematic) | 29.397 | 20.000 | 19 | R1 cubic half filling site-centred | 92517.9316 | 92517.9316 |
| S4_32_s4 j66 | field-only (P_- test-state centre) | 29.393 | 20.651 | 20 | R0 Fock(A) [field:P_-] | 24398.7808 | 24398.7808 |
| S4_32_s4 j66 | field-only (P_- test-state centre) | 29.393 | 20.651 | 20 | R1 N_A=n* [field:P_-] | 24393.3206 | 24393.3206 |
| S4_32_s4 j66 | field-only (P_- test-state centre) | 29.393 | 20.651 | 20 | R2a dq=0.01E [field:P_-] | 1306.7756 | 1571.4309 |
| S4_32_s4 j66 | field-only (P_- test-state centre) | 29.393 | 20.651 | 20 | R2b dq=1e-4E [field:P_-] | 1303.9331 | 1567.8888 |
| S4_32_s4 j66 | cubic background (kinematic) | 29.393 | 20.651 | 20 | R0 cubic site-centred | 102788.1817 | 102788.1817 |
| S4_32_s4 j66 | cubic background (kinematic) | 29.393 | 20.651 | 20 | R1 cubic half filling site-centred | 102782.0024 | 102782.0024 |
| W5_64_s4 birth b97 | field-only (interpolated birth field) | 58.391 | 32.000 | 31 | R0 Fock(A) [field:P_-] | 37818.1102 | 37818.1102 |
| W5_64_s4 birth b97 | field-only (interpolated birth field) | 58.391 | 32.000 | 31 | R1 N_A=n* [field:P_-] | 37812.4308 | 37812.4308 |
| W5_64_s4 birth b97 | field-only (interpolated birth field) | 58.391 | 32.000 | 31 | R2a dq=0.01E [field:P_-] | 1861.7414 | 2162.1125 |
| W5_64_s4 birth b97 | field-only (interpolated birth field) | 58.391 | 32.000 | 31 | R2b dq=1e-4E [field:P_-] | 1861.7414 | 2153.8360 |
| W5_64_s4 birth b97 | cubic background (kinematic) | 58.391 | 32.000 | 31 | R0 cubic site-centred | 380008.2377 | 380008.2377 |
| W5_64_s4 birth b97 | cubic background (kinematic) | 58.391 | 32.000 | 31 | R1 cubic half filling site-centred | 380001.4047 | 380001.4047 |
| W5_64_s4 j97 | field-only (P_- test-state centre) | 58.302 | 38.451 | 38 | R0 Fock(A) [field:P_-] | 46357.6834 | 46357.6834 |
| W5_64_s4 j97 | field-only (P_- test-state centre) | 58.302 | 38.451 | 38 | R1 N_A=n* [field:P_-] | 46351.9023 | 46351.9023 |
| W5_64_s4 j97 | field-only (P_- test-state centre) | 58.302 | 38.451 | 38 | R2a dq=0.01E [field:P_-] | 2309.3705 | 2678.7509 |
| W5_64_s4 j97 | field-only (P_- test-state centre) | 58.302 | 38.451 | 38 | R2b dq=1e-4E [field:P_-] | 2305.0124 | 2669.3762 |
| W5_64_s4 j97 | cubic background (kinematic) | 58.302 | 38.451 | 38 | R0 cubic site-centred | 660788.2976 | 660788.2976 |
| W5_64_s4 j97 | cubic background (kinematic) | 58.302 | 38.451 | 38 | R1 cubic half filling site-centred | 660781.1879 | 660781.1879 |
| W5_64_s4 j100 | field-only (P_- test-state centre) | 57.365 | 45.524 | 45 | R0 Fock(A) [field:P_-] | 54897.2567 | 54897.2567 |
| W5_64_s4 j100 | field-only (P_- test-state centre) | 57.365 | 45.524 | 45 | R1 N_A=n* [field:P_-] | 54891.3910 | 54891.3910 |
| W5_64_s4 j100 | field-only (P_- test-state centre) | 57.365 | 45.524 | 45 | R2a dq=0.01E [field:P_-] | 3345.6845 | 3772.5005 |
| W5_64_s4 j100 | field-only (P_- test-state centre) | 57.365 | 45.524 | 45 | R2b dq=1e-4E [field:P_-] | 3321.1960 | 3760.9780 |
| W5_64_s4 j100 | saved state | 57.365 | 45.524 | 45 | R0 Fock(A) [state:C] | 54897.2567 | 54897.2567 |
| W5_64_s4 j100 | saved state | 57.365 | 45.524 | 45 | R1 N_A=n* [state:C] | 54884.9764 | 54884.9764 |
| W5_64_s4 j100 | saved state | 57.365 | 45.524 | 45 | R2a dq=0.01E [state:C] | 4260.2199 | 4743.0237 |
| W5_64_s4 j100 | saved state | 57.365 | 45.524 | 45 | R2b dq=1e-4E [state:C] | 4256.4455 | 4732.9845 |
| W5_64_s4 j100 | cubic background (kinematic) | 57.365 | 45.524 | 45 | R0 cubic site-centred | 1096345.3503 | 1096345.3503 |
| W5_64_s4 j100 | cubic background (kinematic) | 57.365 | 45.524 | 45 | R1 cubic half filling site-centred | 1096337.9875 | 1096337.9875 |
| W5_64_s4 j104 | field-only (P_- test-state centre) | 55.300 | 47.606 | 47 | R0 Fock(A) [field:P_-] | 57337.1348 | 57337.1348 |
| W5_64_s4 j104 | field-only (P_- test-state centre) | 55.300 | 47.606 | 47 | R1 N_A=n* [field:P_-] | 57331.2474 | 57331.2474 |
| W5_64_s4 j104 | field-only (P_- test-state centre) | 55.300 | 47.606 | 47 | R2a dq=0.01E [field:P_-] | 4555.9847 | 5038.1746 |
| W5_64_s4 j104 | field-only (P_- test-state centre) | 55.300 | 47.606 | 47 | R2b dq=1e-4E [field:P_-] | 4536.4922 | 5025.5817 |
| W5_64_s4 j104 | saved state | 55.300 | 47.606 | 47 | R0 Fock(A) [state:C] | 57337.1348 | 57337.1348 |
| W5_64_s4 j104 | saved state | 55.300 | 47.606 | 47 | R1 N_A=n* [state:C] | 57323.8801 | 57323.8801 |
| W5_64_s4 j104 | saved state | 55.300 | 47.606 | 47 | R2a dq=0.01E [state:C] | 5379.9216 | 5906.9416 |
| W5_64_s4 j104 | saved state | 55.300 | 47.606 | 47 | R2b dq=1e-4E [state:C] | 5371.3938 | 5895.8423 |
| W5_64_s4 j104 | cubic background (kinematic) | 55.300 | 47.606 | 47 | R0 cubic site-centred | 1254471.6303 | 1254471.6303 |
| W5_64_s4 j104 | cubic background (kinematic) | 55.300 | 47.606 | 47 | R1 cubic half filling site-centred | 1254464.2002 | 1254464.2002 |
| W5_64_s4 j112 | field-only (P_- test-state centre) | 54.640 | 50.626 | 50 | R0 Fock(A) [field:P_-] | 60996.9519 | 60996.9519 |
| W5_64_s4 j112 | field-only (P_- test-state centre) | 54.640 | 50.626 | 50 | R1 N_A=n* [field:P_-] | 60991.0335 | 60991.0335 |
| W5_64_s4 j112 | field-only (P_- test-state centre) | 54.640 | 50.626 | 50 | R2a dq=0.01E [field:P_-] | 5831.3695 | 6442.9052 |
| W5_64_s4 j112 | field-only (P_- test-state centre) | 54.640 | 50.626 | 50 | R2b dq=1e-4E [field:P_-] | 5831.3695 | 6427.1260 |
| W5_64_s4 j112 | saved state | 54.640 | 50.626 | 50 | R0 Fock(A) [state:C] | 60996.9519 | 60996.9519 |
| W5_64_s4 j112 | saved state | 54.640 | 50.626 | 50 | R1 N_A=n* [state:C] | 60982.9874 | 60982.9874 |
| W5_64_s4 j112 | saved state | 54.640 | 50.626 | 50 | R2a dq=0.01E [state:C] | 6528.6704 | 7171.4742 |
| W5_64_s4 j112 | saved state | 54.640 | 50.626 | 50 | R2b dq=1e-4E [state:C] | 6509.2109 | 7156.9646 |
| W5_64_s4 j112 | cubic background (kinematic) | 54.640 | 50.626 | 50 | R0 cubic site-centred | 1507159.8213 | 1507159.8213 |
| W5_64_s4 j112 | cubic background (kinematic) | 54.640 | 50.626 | 50 | R1 cubic half filling site-centred | 1507152.2994 | 1507152.2994 |

Table: pairwise p_eff intervals (script pk5_tables_v1.py)

| selection | pair | rule | p_eff interval |
|---|---|---|---|
| birth | S4_32_s4->W5_64_s4 | R0 | [0.713, 0.713] |
| birth | S4_32_s4->W5_64_s4 | R1 | [0.713, 0.713] |
| birth | S4_32_s4->W5_64_s4 | R0_cubic | [2.059, 2.059] |
| birth | S4_32_s4->W5_64_s4 | R2a | [0.335, 0.837] |
| birth | S4_32_s4->W5_64_s4 | R2b | [0.338, 0.833] |
| birth | W5_64_s4->S4_128_s8 | R0 | [0.827, 0.827] |
| birth | W5_64_s4->S4_128_s8 | R1 | [0.827, 0.827] |
| birth | W5_64_s4->S4_128_s8 | R0_cubic | [2.424, 2.424] |
| birth | W5_64_s4->S4_128_s8 | R2a | [0.262, 0.688] |
| birth | W5_64_s4->S4_128_s8 | R2b | [0.259, 0.677] |
| birth | W5_64_s4->S4_128_s16 | R0 | [1.020, 1.020] |
| birth | W5_64_s4->S4_128_s16 | R1 | [1.021, 1.021] |
| birth | W5_64_s4->S4_128_s16 | R0_cubic | [2.994, 2.994] |
| birth | W5_64_s4->S4_128_s16 | R2a | [0.202, 0.648] |
| birth | W5_64_s4->S4_128_s16 | R2b | [0.194, 0.637] |
| birth | S4_32_s4->S4_128_s8 | R0 | [0.771, 0.771] |
| birth | S4_32_s4->S4_128_s8 | R1 | [0.771, 0.771] |
| birth | S4_32_s4->S4_128_s8 | R0_cubic | [2.242, 2.242] |
| birth | S4_32_s4->S4_128_s8 | R2a | [0.407, 0.654] |
| birth | S4_32_s4->S4_128_s8 | R2b | [0.404, 0.649] |
| birth | S4_32_s4->S4_128_s16 | R0 | [0.868, 0.868] |
| birth | S4_32_s4->S4_128_s16 | R1 | [0.868, 0.868] |
| birth | S4_32_s4->S4_128_s16 | R0_cubic | [2.529, 2.529] |
| birth | S4_32_s4->S4_128_s16 | R2a | [0.376, 0.634] |
| birth | S4_32_s4->S4_128_s16 | R2b | [0.371, 0.629] |
| birth | W5_64_s4->S4_128_s4 | R0 | [0.719, 0.719] |
| birth | W5_64_s4->S4_128_s4 | R1 | [0.719, 0.719] |
| birth | W5_64_s4->S4_128_s4 | R0_cubic | [2.106, 2.106] |
| birth | W5_64_s4->S4_128_s4 | R2a | [0.254, 0.677] |
| birth | W5_64_s4->S4_128_s4 | R2b | [0.250, 0.667] |
| birth | W5_64_s4->ISO_128_s4 | R0 | [1.348, 1.348] |
| birth | W5_64_s4->ISO_128_s4 | R1 | [1.348, 1.348] |
| birth | W5_64_s4->ISO_128_s4 | R0_cubic | [3.962, 3.962] |
| birth | W5_64_s4->ISO_128_s4 | R2a | [0.565, 1.010] |
| birth | W5_64_s4->ISO_128_s4 | R2b | [0.562, 1.000] |
| first_rooted | S4_32_s4->W5_64_s4 | R0 | [0.937, 0.937] |
| first_rooted | S4_32_s4->W5_64_s4 | R1 | [0.937, 0.937] |
| first_rooted | S4_32_s4->W5_64_s4 | R0_cubic | [2.717, 2.717] |
| first_rooted | S4_32_s4->W5_64_s4 | R2a | [0.562, 1.048] |
| first_rooted | S4_32_s4->W5_64_s4 | R2b | [0.563, 1.046] |
| first_rooted | W5_64_s4->S4_128_s8 | R0 | [0.922, 0.922] |
| first_rooted | W5_64_s4->S4_128_s8 | R1 | [0.922, 0.922] |
| first_rooted | W5_64_s4->S4_128_s8 | R0_cubic | [2.767, 2.767] |
| first_rooted | W5_64_s4->S4_128_s8 | R2a | [0.344, 0.752] |
| first_rooted | W5_64_s4->S4_128_s8 | R2b | [0.342, 0.745] |
| first_rooted | W5_64_s4->S4_128_s16 | R0 | [1.052, 1.052] |
| first_rooted | W5_64_s4->S4_128_s16 | R1 | [1.052, 1.052] |
| first_rooted | W5_64_s4->S4_128_s16 | R0_cubic | [3.153, 3.153] |
| first_rooted | W5_64_s4->S4_128_s16 | R2a | [0.190, 0.626] |
| first_rooted | W5_64_s4->S4_128_s16 | R2b | [0.179, 0.619] |
| first_rooted | S4_32_s4->S4_128_s8 | R0 | [0.929, 0.929] |
| first_rooted | S4_32_s4->S4_128_s8 | R1 | [0.929, 0.929] |
| first_rooted | S4_32_s4->S4_128_s8 | R0_cubic | [2.742, 2.742] |
| first_rooted | S4_32_s4->S4_128_s8 | R2a | [0.560, 0.792] |
| first_rooted | S4_32_s4->S4_128_s8 | R2b | [0.558, 0.788] |
| first_rooted | S4_32_s4->S4_128_s16 | R0 | [0.995, 0.995] |
| first_rooted | S4_32_s4->S4_128_s16 | R1 | [0.995, 0.995] |
| first_rooted | S4_32_s4->S4_128_s16 | R0_cubic | [2.937, 2.937] |
| first_rooted | S4_32_s4->S4_128_s16 | R2a | [0.482, 0.728] |
| first_rooted | S4_32_s4->S4_128_s16 | R2b | [0.475, 0.724] |
| first_rooted | W5_64_s4->S4_128_s4 | R0 | [0.683, 0.683] |
| first_rooted | W5_64_s4->S4_128_s4 | R1 | [0.683, 0.683] |
| first_rooted | W5_64_s4->S4_128_s4 | R0_cubic | [2.045, 2.045] |
| first_rooted | W5_64_s4->S4_128_s4 | R2a | [0.200, 0.599] |
| first_rooted | W5_64_s4->S4_128_s4 | R2b | [0.189, 0.592] |
| first_rooted | W5_64_s4->ISO_128_s4 | R0 | [1.088, 1.088] |
| first_rooted | W5_64_s4->ISO_128_s4 | R1 | [1.088, 1.088] |
| first_rooted | W5_64_s4->ISO_128_s4 | R0_cubic | [3.235, 3.235] |
| first_rooted | W5_64_s4->ISO_128_s4 | R2a | [0.303, 0.731] |
| first_rooted | W5_64_s4->ISO_128_s4 | R2b | [0.288, 0.724] |

Table: omitted entropy and state diagnostics (script pk5_tables_v1.py from pk5_counts_v2.py state rows)

| member | x | T/r_s | S(C_A) | S(ref_A) | S(C_O) ext chain | S_O k>10 bound | var N_A | P(N_A=n*) | pinching RHS R1 | approx-support log dim | orth defect |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ISO_128_s4 j184 | 116.887 | 11.5000 | 1434.3151 | 830.3349 | 1431.4886 | 6271692.7 | 448.91 | 1.883e-02 | 98814.8 | 6210.6 | 1.3e-12 |
| S4_128_s4 j184 | 110.651 | 11.5000 | 1292.4233 | 837.2852 | 1292.4233 | 8510829.8 | 397.25 | 2.002e-02 | 109794.2 | 5789.2 | 1.3e-12 |
| S4_128_s8 j184 | 114.950 | 11.5000 | 1301.6504 | 836.9967 | 1301.6504 | 9495559.1 | 401.15 | 1.992e-02 | 113454.0 | 5755.9 | 1.3e-12 |
| S4_128_s8 j192 | 111.558 | 12.0000 | 1169.1305 | 839.8949 | 1169.1305 | 11844357.6 | 345.75 | 2.146e-02 | 121993.5 | 5722.6 | 1.3e-12 |
| S4_128_s8 j208 | 110.600 | 13.0000 | 1015.1699 | 841.9662 | 1015.1699 | 14020318.5 | 289.71 | 2.344e-02 | 129313.1 | 17278.8 | 1.3e-12 |
| S4_128_s8 j224 | 110.233 | 14.0000 | 946.1699 | 842.5714 | 946.1699 | 14703661.8 | 268.81 | 2.433e-02 | 131753.0 | 23744.4 | 1.4e-12 |
| W5_64_s4 j100 | 57.365 | 12.5000 | 1050.7181 | 781.3385 | 1050.7181 | 1096345.4 | 315.85 | 2.245e-02 | 54897.1 | 7752.2 | 5.8e-13 |
| W5_64_s4 j104 | 55.300 | 13.0000 | 1002.8762 | 786.4579 | 1002.8762 | 1254471.6 | 295.01 | 2.322e-02 | 57336.9 | 7097.8 | 5.8e-13 |
| W5_64_s4 j112 | 54.640 | 14.0000 | 938.2699 | 793.2599 | 938.2699 | 1507159.8 | 269.46 | 2.430e-02 | 60996.7 | 13042.3 | 6.0e-13 |

Uniform errors B2n4096 (script pk5_uniform_v1.py): 801 records, 47 store jumps, saturation {"t_from_formation": 1.7306559271442687, "lambda_sup": 0.5778156040814456, "T_over_rs_formation": 12.125}

Uniform errors W5_64_s4 (script pk5_uniform_v1.py): 113 records, 8 store jumps, saturation {"t_from_formation": 1.7306585640437855, "lambda_sup": 0.5778147236988451, "T_over_rs_formation": 12.125}

Table: saved-state rows, earliest post-formation slice (script pk5_tables_state_v1.py; clocks not formation-matched)

| member | T/r_s | x | R_t | n_in | R0 | R2a state centre | R2a/x^2 | R2a/x^3 | beta | S(C_A) | S(C_A)-S(ref_A) | dS/x^2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W5_64_s4 j100 | 12.5000 | 57.365 | 45.524 | 45 | 54897.26 | [4260.22, 4743.02] | [1.2946, 1.4413] | [0.022568, 0.025126] | 17.66 | 1050.72 | 269.38 | 0.08186 |
| S4_128_s8 j184 | 11.5000 | 114.950 | 93.504 | 93 | 113454.33 | [10044.82, 10927.63] | [0.7602, 0.8270] | [0.006613, 0.007194] | 25.04 | 1301.65 | 464.65 | 0.03516 |
| S4_128_s4 j184 | 11.5000 | 110.651 | 90.153 | 90 | 109794.51 | [16606.29, 17770.12] | [1.3563, 1.4514] | [0.012258, 0.013117] | 19.56 | 1292.42 | 455.14 | 0.03717 |
| ISO_128_s4 j184 | 11.5000 | 116.887 | 81.427 | 81 | 98815.06 | [5688.42, 6405.17] | [0.4164, 0.4688] | [0.003562, 0.004011] | 18.07 | 1434.32 | 603.98 | 0.04421 |

| pair | R2a state-centre p_eff | S(C_A) p_eff | S(C_A)-S(ref_A) p_eff | R0 p_eff |
|---|---|---|---|---|
| W5_64_s4->S4_128_s8 | [1.080, 1.355] | 0.308 | 0.784 | 1.044 |
| W5_64_s4->S4_128_s4 | [1.907, 2.174] | 0.315 | 0.798 | 1.055 |
| W5_64_s4->ISO_128_s4 | [0.255, 0.573] | 0.437 | 1.134 | 0.826 |

## P5.5 Omitted-sector entropy

**A. Mode algebra** (Gaussian, saved states). S(C_O) on the exterior chain equals S(C_A) to all printed digits wherever the dust occupations are 0 or 1, as a pure state requires. The ISO control is the exception (1431.49 against 1434.32), because its preparation has fractional dust weights.

Resolution:
- Inside/outside the root: done.
- Computed channels: done.
- Channels k > 10: no state exists. The honest bound is S_O <= (all cubic interior modes) ln 2, which is volume order (column "S_O k>10 bound"). An uncomputed channel tail is not zero.
- Light species: not represented on these records, so UNBOUNDED/BLOCKED.
- Bound/slow/propagating split: UNDECIDED (no admitted classification).

Total: S_retained <= S_A + S_O, with S_O volume-order bounded and not computed. So the store interpretation is unsupported (P5.10 row 4).

**B. Discarded many-body sectors.** P(N_A = n*) is about 0.02 for every saved state; this is the exact characteristic-function inversion of the Gaussian reduced state. So p_omit is about 0.98 for R1, and the pinching right-hand side h2(p) + (1-p) log D_acc + p log D_omit is essentially M_A ln 2 (column "pinching RHS R1"). A fixed-number sector is not a description of the actual state.

**Approximate support.** The diagnostic counts modes with c in (1e-6, 1 - 1e-6), with discarded weight at most 4e-3 by a union bound. It is a state diagnostic only (P5.10 row 3).

## P5.6 Evolution-uniform errors (pk5_uniform_v1.py)

B2n4096 has 801 records through T/r_s = 100; W5 has 113. Results:
- For fixed A, the cross block of hbar is affine in t under the declared linear field interpolation. Its norm is therefore convex, and the sup over each interval is the maximum of the endpoint values (an exact between-record enclosure).
- lambda_A sup = 0.5778 a^-1. Integrated from formation at T/r_s 12.125, the trace-distance budget reaches 1 after 1.73 a of time, inside the first record interval (tau = 8 a).
- The store changes discontinuously 47 times over B2n4096 (8 over W5) as R_t crosses sites. Each change is an explicit jump with no geometric-term bound.
- Discarded weight: only on slices (state rows).
- Charge-operator error ||H_+ - Q_C||: BLOCKED.
- Branch leakage: BLOCKED (no Pi).
- Return/output error: not applicable (E_out is empty).
- Covariance numerical error: orthonormality defect 6e-13 to 1.4e-12. Kahan enclosures of the A-block eigenvalues: 1.7e-13 to 4.5e-12.
- Verdict: **UNDECIDED.** No evolution-uniform error below 1 exists for the spatial store over any stored interval longer than about 2 a. An operator-norm bound cannot see the trapping, which is a property of the flow acting on the states.

## P5.4 Branch and interaction calculation

**BLOCKED** on the operands E2-hat, the represented masses M_j, Pi and the continued-reference response. Radial fields alone do not supply them, and there is no admitted cubic covariance (P5.4, last paragraph). The quadratic and kinematic diagnostics above continue without them.

## P5.8 Radial chains versus cubic lattice

- **Chain counts** are counts of the declared finite chain model. Its copies are replicated exactly (g_k = 16k), and its R0 is linear in n_in at a fixed channel set.
- **Cubic kinematic counts** (one four-component field; valleys inside the zone, not multiplied again) are 4 N_ball ln 2, exactly volume order. Site-centred and cell-centred origins are both in the JSON.
  - The R0_cubic p_eff values in the table (2.0-4.0) are not exponents of a law. They follow R_t^3 at the varying R_t/r_Q of each member, since the x used is r_Q.
- **Cubic R2:** BLOCKED (no cubic state). A degeneracy-weighted radial count cannot certify the cubic retained capacity.

## P5.10 Decisions (only what the tables license)

| Outcome row | Status here |
|---|---|
| Controlled c3 > 0 for the admissible sector | Not available: S0 not identified. The cubic interior's kinematic count is exactly volume order, but it is not an accessible sector. |
| c3 = 0, finite c2, S_O = o(x^2), errors o(x^2) | Not established. S_O is bounded only at volume order, and the uniform errors saturate. |
| Area scaling for one prepared state only | Saved-state entropies grow sublinearly in x at the earliest post-formation slices: S(C_A) p_eff 0.31-0.44; S(C_A) - S(ref_A) p_eff 0.78-1.13. These are state diagnostics, not ceilings. |
| Area-sized store, other sectors unbounded | Applies to any store proposal: the omitted retained algebra is volume-order bounded and not computed. |
| Non-area/non-volume scaling | The field-centre R2 counts give p_eff intervals within [0.18, 1.05] over every pair. The state-centre R2 counts (clocks not matched) give [1.08, 1.36] (64 to s8), [1.91, 2.17] (64 to the width-4 branch) and [0.26, 0.57] (64 to ISO). The spread across preparations means there is no law. |
| Only 32/64/128 trends | Leading coefficient UNRESOLVED. |
| Missing sector, branch, charge-error control | **Full accessible-capacity calculation UNRESOLVED (ACCESSIBLE_SECTOR_NOT_IDENTIFIED).** Adding an area-sized store would be a declaration. |

No Page mark closes from this package. Release, saturation and heavy-information recovery remain separate (S8).

## FABLE NEEDED (rulings, not staged)

1. Is the held-Killing-energy charge Q_C an admissible stand-in for any accessible-sector rule, or must every count wait for H_+ on an admitted branch?
2. Which admitted input sector S0 (preparations and controls) should define accessibility? Without it, P5 cannot pass its first gate.
3. Should the uncomputed channels k > 10 be bounded by the cubic interior count (as here), or by a replayed vacuum-sea state on the held fields (a test-state construction, kind 3)?

## Self red team

1. **Charge.** Every count uses spec charge C, not the boundary charge, and C does not commute with the dynamics. The counts are therefore kinematic or diagnostic only, and that is how they are labelled.
2. **Window centres.**
   - The field centre is the P_- test state, a near-vacuum interior (beta 12-52 over the rows). It is cold by construction, which is why R2 is a few percent of R0.
   - The state centre comes from the saved covariance, at clocks not matched to formation (12.5 at 64; 11.5 at 128, which is 0.31 r_s after the s8 formation).
   - No count was centred to fit a size law. The rules were fixed before the first size comparison.
3. **Lower bound.** It comes from a single pattern, so it can be loose: a relative gap (ub - lb)/ub of 7-18 percent at the field centre and 7-11 percent at the state centre. In the 160-mode exact check the true count sat 2.0 nats above the lower bound and 4.4 below the upper. The p_eff intervals carry the whole gap. Summing over many patterns would tighten them; that was not done.
4. **Root definition.**
   - The store uses the geometric (node) root. The applied-field root is smaller, or absent (32, s16, ISO). Counting inside the applied root would shrink every chain count, and at 32, s16 and ISO it would leave no store.
   - The birth R* is quantized to the node grid.
5. **The 128 slices are not at formation.** No replay to the formation record (kind 3) was run. That would be one chain step per record from the 11.0 slice, and it is listed as not done.
6. **Size 32 has no slices on the box,** so it has no state row.
7. **Blindness held.** Nothing from route 1 was read. I saw only the directory name <internal>/webgpt_pk5_r1 in a listing.
8. **Rule slips, disclosed.**
   - One empty local `python3 -` ran by mistake (no input, no output).
   - One box launch ran at nlwp 31 because the env has two OpenBLAS pools. I killed it by pidfile with a ps check; everything since ran at 8 threads, nlwp 15.
   - pk5_small_v1, pk5_counts_v1/v2 and pk5_uniform_v1 were edited in place before any successful run (an allow_pickle fix and a node-root change). Every changed engine after a successful run got a new name (counts_v3, lib3, ab_v2).
9. **Numbers.** All numbers in the tables were pasted from script output files, not retyped. Values quoted in prose are copied from those tables.

## Files

The files are in PK5_R2_busi_v1_files/ beside this report:
- inputs_manifest.json
- inputs_md5_all.tsv: every read input with its md5, including the B2n4096 inputs. Both B2n4096 md5s match their .md5 files.
- small_parent_v1.json
- tables/: p5_sector_counts, p5_omitted_entropy and p5_uniform_errors as .parquet and .tsv; p5_scaling_bounds.json; p5_state_scaling_v1.json; tables_v1.md; tables_state_v1.md
- logs/: A/B, profile, MD5SUMS_code.txt, MD5SUMS_outputs.txt

The Package 2 files of the spec are not part of this route. The decisions above are the content of decisions.md.

Box: <internal>/pk5_r2 (RESUME_NOTES.txt, MD5SUMS_code.txt, MD5SUMS_outputs.txt). The worktree code matches the box copy: md5sum -c reports 0 mismatches.
