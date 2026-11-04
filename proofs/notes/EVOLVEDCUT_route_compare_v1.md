# EVOLVEDCUT route comparison v1: first route (paper Sec. V F "109 to 311 nats") against the second route

Inbox #248. Agreement or disagreement only; no physics judgment. Comparison computed on the box CPU (nice 10, one thread) from the two routes' stored per-channel jsons; no entropy was recomputed. All inputs were read only.

## 1. Sources

**Paper sentence.** <internal>/07_page_curve.tex (md5 2c40c0eff22be639a83c09f354fe386f, last commit 6916a27d057), subsection "The entanglement across the evolved trapping cut", line 469: "chain sums on the channels kappa <= 10 with their multiplicities put the heavy field's exterior entropy above the held Killing vacuum's by 109 to 311 and 116 to 305 nats ... the evolved cut has these chain sums on one route". The first range is the sigma/a 4 production; the second is the sigma/a 8 member.

**First route (route 1).** EVOLVEDCUT_slices_v1, the solve lane's readout (module evolvedcut_slices_v1.py, built by the busi lane as solve builder in commit 323feec38ac on 18 Oct, run on the box on 18 Oct; cold-read by reviewer A as "route 1" in <internal>/FableAPI_cold_reading_EVOLVEDCUT_slices_v1.md, DELIVERIES.md line 677).
- Code: <internal>/evolvedcut_slices_v1.py (md5 62f292d44b94fbe08e6580faf2feeae8); run notes <internal>/RUN_evolvedcut_slices_v1.md.
- Stored per-channel numbers: box 192.168.1.93:<internal>/evolvedcut_prod_T00{12,15,16,20,24,25,28,30,35,40}.00.json (md5s cef412c4..., dce9606c..., 37d20215..., 7b71b832..., ff4a44d2..., 89b3222f..., 951924c9..., 2920fec6..., c5914ec3..., 241f26f9...; full list in compare_v1.json "inputs").
- Tables: same directory, EVOLVEDCUT_slices_v1_prod.md (md5 7801cbbe42f9a2cabc59dc4a77f00492, identical to the cold-reading input copy), EVOLVEDCUT_slices_v1_s8.md (md5 95bf415f3d941062332412952e871bba), EVOLVEDCUT_slices_v1_all.json (md5 9625973cd6a697443925ad958ff22c4c).
- The prod row range 109.36 (T15) to 310.89 (T40) is the paper's "109 to 311"; the s8 range 116.10 (T16) to 305.41 (T20) is the "116 to 305".

**Second route (route 2).** EVOLVEDCUT_secondroute_v1, produced by the codex_solve lane (worktree <internal>/.wt-route2-solve; called icloud's route in the task).
- Deliveries: <internal>/SECONDROUTE_EVOLVEDCUT_primary_channels_v1.md (md5 c443415e8f89fd60062746d2b1acfdbe) and SECONDROUTE_EVOLVEDCUT_v1.md (md5 57caa56168990bc8f1dc6b16aa57d6f3).
- Frozen data: <internal>/results.json (md5 8a3c7730c495cd1e8904a0f1ec7427d1) and tables.json (md5 a78c9bfb159b87599affe16d0422d1a4), commit b858ff7f554. Spec <internal>/EVOLVEDCUT_secondroute_spec_v1.md (md5 4049b2633e76b6e83e84c813a2197730); producer code run_EVOLVEDCUT_secondroute_v1.py (md5 a4a74114495255d03c6b4d4fce25b344).

**This comparison.** Scripts <internal>/evolvedcut_route_compare_v1.py (md5 74e36ffa60ee08312518b4a3046c9061) and evolvedcut_route_compare_table_v1.py (md5 01cb4d0dd86505b4332abf87244869c2), run on the box in <internal>/evolvedcut_route_compare_v1 (route 2's two jsons copied there, md5s unchanged). Outputs copied to <internal>/compare_v1.json (md5 209eea50d2d95d9cac3be7abac4e0ba7) and compare_v1_tables.md (md5 49189dc93bc16e2f99c846f37632133e).

## 2. Definitions side by side

| Item | Route 1 (EVOLVEDCUT_slices_v1, prod) | Route 2 (EVOLVEDCUT_secondroute_v1) | Same? |
|---|---|---|---|
| History, parameters | c19_rs64_A_R2_tau8_ell4_avg_r11; ma 0.2, G/a^2 0.25, sigma/a 4, tau/a 8, 2048 sites | same tag, same parameters | yes |
| Slice files | <internal>/orbitals_..._T*.npz at run time (18 Oct; that path no longer exists on the box) | <internal>/orbitals_..._T*.npz | same basenames and byte sizes; the route-1 path is gone, so byte identity is not verified by md5 (the files have link count 2; the driver lists the pod_sync directory as the second copy of the same slices) |
| Clocks | T/r_s 12, 15, 16, 20, 24, 25, 28, 30, 35, 40 | same 10 | yes |
| Applied fields | the slice's N_applied, v_applied | the slice's N_applied, v_applied | yes |
| Channels, weights | kappa 1 to 10, weight 16 kappa, sum 880 | kappa 1 to 10, weight 16 kappa | yes |
| Partition | X = sites n > n_in, both spinor components together | same, R_n/a = n | yes |
| Cut position | the record's outermost trapping zero (record root); 2gE of the record at T12 (no root) | the outermost negative-to-positive crossing of N - abs v of the applied fields, linearly interpolated (applied root); charge fallback 2gE at T12; plus partitions shifted -1 and +1 site | **no**, except T12. Route 1's record root lies 1 to 3 sites outside route 2's applied root at T15 to T40 (route 2's spec, line 33, records this choice) |
| Interior sites | 58, 51, 50, 38, 38, 37, 37, 37, 37, 36 | primary 58, 50, 48, 37, 36, 34, 36, 35, 36, 35; saved partitions primary plus/minus 1 | equal partitions exist at T12 (shift 0), T15, T20, T28, T35, T40 (shift +1); none at T16, T24, T25, T30 |
| State C | pure Gaussian from the Loewdin-orthonormalized occupied columns (sea and dust) | (a) raw C = V V^dag + W F W^dag (gated; unavailable at the complex64 clocks 15, 25, 30, 35, 40); (b) orthonormal (polar) control Q = [V,W]([V,W]^dag[V,W])^(-1/2), F = 1 | route 1 equals route 2's polar control (Loewdin = polar; all occupations 1). Compared like with like against the polar control; raw shown beside |
| Held vacuum | 1(h_m < 0) at the applied fields (route 1 code) | P_- = 1(h_kappa < 0), h_kappa = (1/2){N, h_0} + (1/2){v, Q_r}, open chain, independently built | same definition, independent constructions |
| Flat reference | 1(h_0 < 0) at N = 1, v = 0 | P_0 = 1(h_0 < 0) | yes |
| Excess | S_X(C) - S_X(P_-), weighted, summed | same (polar excess, raw excess) | yes |
| Entropy side | S_X on the exterior (Gram 2053 x 2053) and S_Xc on the interior block; both saved | exterior entropy (dimension 2053) and interior entropy (dimension 2 n_in) for C; held and flat saved as one entropy of the interior block (dimension 2 n_in) | the exterior and complement sides are compared separately below |
| Area normalization | eta A with eta A/a^2 = 47312 at R = 64 a, A at the record root | eta_* A_lat with eta_* 0.9192 at the applied root | **different**; not compared (the task compares entropies) |

## 3. Stated errors used

- Route 1: per channel, pure-cut identity defect and purity gap at most 2.1e-10 nats (its table; 1.85e-7 nats on the 880-weight sum); the complex64 slices carry a source floor of 6.2e-6 to 8.7e-6 relative on the sources, which route 1 states is not propagated to the entropy; the kappa = 10 truncation is untailed.
- Route 2: no entropy error is stated ("digits display saved arithmetic, not physical uncertainty"); its spectral rounding tolerance tau_eig = 64 d eps max(1, max abs lambda) is 2.9e-11 per eigenvalue at d = 2053, and its cut shifts are declared placement sensitivities, not errors.
- The larger of the two stated entropy errors is therefore route 1's 2.1e-10 per channel and 1.85e-7 per weighted sum. Every test below uses that number.

## 4. Comparison

The tables below are copied unchanged from compare_v1_tables.md. "Complement side" means route 1's S_Xc (interior block) against route 2's interior-block entropy at the same partition, and for the excess, the weighted (S_Xc(C) - S_Xc(P_-)) of each route.

### Clock table (weighted, 16 kappa, kappa <= 10)

| T/r_s | dtype | route 1 cut: R/a (source); interior sites | route 2 primary cut: R/a (label); interior sites | route 2 partitions saved | like-for-like partition | route 1 excess (paper range) | route 2 polar excess at that partition | difference (1 - 2) | within 880 x 2.1e-10 = 1.85e-7? | complement-side difference (1 - 2) | within 1.85e-7? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | complex128 | 58.4668 (2gE, no root); 58 | 58.4668 (charge_fallback); 58 | 57,58,59 | shift +0 (58 sites) | 285.101126742 | 285.101129312 | -2.570e-06 | NO | -1.953e-09 | yes |
| 15 | complex64 | 51.7604 (record outer trapping zero); 51 | 50.8591 (applied_root); 50 | 49,50,51 | shift +1 (51 sites) | 109.355860741 | 109.355863275 | -2.534e-06 | NO | +4.751e-09 | yes |
| 16 | complex128 | 50.8371 (record outer trapping zero); 50 | 48.5364 (applied_root); 48 | 47,48,49 | none (route 1 at 50 sites) | 117.972265370 | not computed by route 2 at 50 sites | - | - | - | - |
| 20 | complex128 | 38.4322 (record outer trapping zero); 38 | 37.2108 (applied_root); 37 | 36,37,38 | shift +1 (38 sites) | 296.510131979 | 296.510134632 | -2.652e-06 | NO | +8.066e-10 | yes |
| 24 | complex128 | 38.9166 (record outer trapping zero); 38 | 36.8778 (applied_root); 36 | 35,36,37 | none (route 1 at 38 sites) | 286.185106656 | not computed by route 2 at 38 sites | - | - | - | - |
| 25 | complex64 | 37.2883 (record outer trapping zero); 37 | 34.6995 (applied_root); 34 | 33,34,35 | none (route 1 at 37 sites) | 307.360006268 | not computed by route 2 at 37 sites | - | - | - | - |
| 28 | complex128 | 37.5535 (record outer trapping zero); 37 | 36.3898 (applied_root); 36 | 35,36,37 | shift +1 (37 sites) | 297.715847296 | 297.715850023 | -2.727e-06 | NO | -7.592e-10 | yes |
| 30 | complex64 | 37.5110 (record outer trapping zero); 37 | 35.9333 (applied_root); 35 | 34,35,36 | none (route 1 at 37 sites) | 295.700534302 | not computed by route 2 at 37 sites | - | - | - | - |
| 35 | complex64 | 37.2193 (record outer trapping zero); 37 | 36.0165 (applied_root); 36 | 35,36,37 | shift +1 (37 sites) | 299.443538883 | 299.443541371 | -2.488e-06 | NO | +1.286e-09 | yes |
| 40 | complex64 | 36.3100 (record outer trapping zero); 36 | 35.0420 (applied_root); 35 | 34,35,36 | shift +1 (36 sites) | 310.888400433 | 310.888402919 | -2.486e-06 | NO | +1.012e-09 | yes |

### Route 2 primary (applied-root) excess beside route 1 (record-root) excess, different cuts, not a comparison

| T/r_s | route 1 excess (record root) | route 2 polar excess (applied root, shift 0) | route 2 raw excess (shift 0) |
|---|---|---|---|
| 12 | 285.101127 | 285.101129 | 285.101131 |
| 15 | 109.355861 | 111.252537 | unavailable |
| 16 | 117.972265 | 149.757879 | 149.757881 |
| 20 | 296.510132 | 297.688767 | 297.688769 |
| 24 | 286.185107 | 292.055167 | 292.055168 |
| 25 | 307.360006 | 323.755759 | unavailable |
| 28 | 297.715847 | 298.001796 | 298.001798 |
| 30 | 295.700534 | 298.960750 | unavailable |
| 35 | 299.443539 | 299.772206 | unavailable |
| 40 | 310.888400 | 311.270006 | unavailable |

### Weighted component sums at the matched partitions (route 1, route 2, difference)

| T/r_s | S_X(C) r1 | S_X(C) r2 polar | diff | S_X(P_-) r1 | S_X(P_-) r2 | diff | S_X(P_0) r1 | S_X(P_0) r2 | diff | S_X(C) r2 raw | diff r1 - raw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 1092.484971812 | 1092.484974218 | -2.41e-06 | 807.383845069 | 807.383844906 | +1.63e-07 | 807.273777318 | 807.273777187 | +1.31e-07 | 1092.484975429 | -3.62e-06 |
| 15 | 904.627645966 | 904.627648339 | -2.37e-06 | 795.271785226 | 795.271785064 | +1.62e-07 | 795.027763579 | 795.027763434 | +1.45e-07 | unavailable (gate) | - |
| 20 | 1055.499807263 | 1055.499809751 | -2.49e-06 | 758.989675284 | 758.989675119 | +1.65e-07 | 758.591456230 | 758.591456077 | +1.54e-07 | 1055.499811761 | -4.50e-06 |
| 28 | 1052.834099719 | 1052.834102280 | -2.56e-06 | 755.118252422 | 755.118252256 | +1.66e-07 | 754.668026073 | 754.668025916 | +1.56e-07 | 1052.834104870 | -5.15e-06 |
| 35 | 1054.557281066 | 1054.557283387 | -2.32e-06 | 755.113742183 | 755.113742016 | +1.67e-07 | 754.668026073 | 754.668025916 | +1.56e-07 | unavailable (gate) | - |
| 40 | 1061.862144526 | 1061.862146846 | -2.32e-06 | 750.973744093 | 750.973743928 | +1.66e-07 | 750.525786245 | 750.525786091 | +1.54e-07 | unavailable (gate) | - |

### Per channel at the matched partitions (unweighted, nats; differences route 1 minus route 2)

| T/r_s | kappa | S_X(C) r1 | S_X(C) r2 polar | d ext | d complement | d held (ext; comp) | d flat (ext; comp) | d excess (ext) | d C vs r2 raw | r1 ext-int | r2 polar ext-int | ext within 2.1e-10 | comp within 2.1e-10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 1 | 2.3138386763 | 2.3138386792 | -2.88e-09 | -5.18e-12 | +1.90e-10; -1.97e-12 | +1.21e-10; -7.02e-11 | -3.07e-09 | -4.03e-09 | +1.48e-10 | +3.03e-09 | NO | yes |
| 12 | 2 | 2.2872580415 | 2.2872580444 | -2.91e-09 | +4.57e-12 | +1.87e-10; -2.35e-12 | +1.14e-10; -7.81e-11 | -3.10e-09 | -4.05e-09 | +1.50e-10 | +3.07e-09 | NO | yes |
| 12 | 3 | 2.1305395753 | 2.1305395778 | -2.48e-09 | -3.28e-12 | +1.83e-10; -1.12e-12 | +1.20e-10; -7.19e-11 | -2.66e-09 | -4.21e-09 | +1.45e-10 | +2.62e-09 | NO | yes |
| 12 | 4 | 2.0001967105 | 2.0001967140 | -3.45e-09 | -3.37e-12 | +1.85e-10; -4.00e-12 | +1.50e-10; -5.35e-11 | -3.64e-09 | -4.13e-09 | +1.59e-10 | +3.61e-09 | NO | yes |
| 12 | 5 | 2.1836166867 | 2.1836166890 | -2.28e-09 | -4.37e-12 | +1.84e-10; -3.42e-12 | +1.61e-10; -4.52e-11 | -2.46e-09 | -4.18e-09 | +1.55e-10 | +2.43e-09 | NO | yes |
| 12 | 6 | 0.9344229731 | 0.9344229756 | -2.47e-09 | +5.96e-12 | +1.74e-10; -4.45e-12 | +1.49e-10; -4.59e-11 | -2.65e-09 | -4.16e-09 | +1.57e-10 | +2.64e-09 | NO | yes |
| 12 | 7 | 0.9209594435 | 0.9209594469 | -3.35e-09 | -1.96e-12 | +1.85e-10; -2.30e-12 | +1.43e-10; -4.68e-11 | -3.53e-09 | -4.18e-09 | +1.52e-10 | +3.50e-09 | NO | yes |
| 12 | 8 | 0.9064847051 | 0.9064847081 | -3.05e-09 | -2.39e-11 | +1.84e-10; -2.47e-12 | +1.47e-10; -4.49e-11 | -3.23e-09 | -4.10e-09 | +1.54e-10 | +3.18e-09 | NO | yes |
| 12 | 9 | 0.8912622710 | 0.8912622735 | -2.46e-09 | -3.61e-12 | +1.82e-10; -3.18e-12 | +1.55e-10; -4.48e-11 | -2.64e-09 | -4.01e-09 | +1.59e-10 | +2.62e-09 | NO | yes |
| 12 | 10 | 0.8754974954 | 0.8754974978 | -2.43e-09 | -3.90e-12 | +1.98e-10; -2.52e-12 | +1.58e-10; -4.22e-11 | -2.63e-09 | -4.07e-09 | +1.58e-10 | +2.58e-09 | NO | yes |
| 15 | 1 | 1.5554426087 | 1.5554426119 | -3.14e-09 | +5.25e-13 | +1.76e-10; -2.11e-12 | +1.31e-10; -5.99e-11 | -3.32e-09 | unavailable | +2.06e-10 | +3.35e-09 | NO | yes |
| 15 | 2 | 1.4983292317 | 1.4983292337 | -1.96e-09 | -4.09e-13 | +1.86e-10; -8.91e-13 | +1.43e-10; -6.79e-11 | -2.15e-09 | unavailable | +2.05e-10 | +2.17e-09 | NO | yes |
| 15 | 3 | 1.4265711862 | 1.4265711887 | -2.56e-09 | -8.54e-13 | +1.89e-10; -1.04e-12 | +1.42e-10; -6.02e-11 | -2.75e-09 | unavailable | +2.00e-10 | +2.76e-09 | NO | yes |
| 15 | 4 | 1.2796388923 | 1.2796388946 | -2.29e-09 | -2.73e-13 | +1.79e-10; -2.53e-12 | +1.52e-10; -4.51e-11 | -2.47e-09 | unavailable | +2.00e-10 | +2.49e-09 | NO | yes |
| 15 | 5 | 1.4121686132 | 1.4121686153 | -2.13e-09 | -2.63e-13 | +1.80e-10; -9.17e-13 | +1.71e-10; -3.74e-11 | -2.31e-09 | unavailable | +1.99e-10 | +2.33e-09 | NO | yes |
| 15 | 6 | 0.9283868383 | 0.9283868413 | -2.99e-09 | -2.97e-12 | +1.74e-10; -2.98e-12 | +1.61e-10; -3.81e-11 | -3.16e-09 | unavailable | +2.06e-10 | +3.19e-09 | NO | yes |
| 15 | 7 | 0.9113521978 | 0.9113522005 | -2.67e-09 | -1.95e-12 | +1.81e-10; -2.64e-12 | +1.67e-10; -3.85e-11 | -2.85e-09 | unavailable | +2.03e-10 | +2.87e-09 | NO | yes |
| 15 | 8 | 0.8936707681 | 0.8936707707 | -2.63e-09 | -1.25e-12 | +1.89e-10; -1.95e-12 | +1.67e-10; -3.61e-11 | -2.82e-09 | unavailable | +2.03e-10 | +2.83e-09 | NO | yes |
| 15 | 9 | 0.8748884962 | 0.8748884988 | -2.63e-09 | +1.27e-12 | +1.89e-10; -1.14e-13 | +1.71e-10; -3.78e-11 | -2.82e-09 | unavailable | +2.06e-10 | +2.84e-09 | NO | yes |
| 15 | 10 | 0.8554865582 | 0.8554865614 | -3.24e-09 | +2.45e-11 | +1.87e-10; -1.26e-12 | +1.72e-10; -3.49e-11 | -3.42e-09 | unavailable | +1.99e-10 | +3.46e-09 | NO | yes |
| 20 | 1 | 2.3942688141 | 2.3942688165 | -2.45e-09 | +2.15e-13 | +1.76e-10; -1.63e-12 | +1.64e-10; -4.20e-11 | -2.62e-09 | -5.02e-09 | +1.57e-10 | +2.61e-09 | NO | yes |
| 20 | 2 | 2.3949819237 | 2.3949819260 | -2.35e-09 | +4.26e-13 | +1.83e-10; -7.28e-13 | +1.57e-10; -4.84e-11 | -2.53e-09 | -5.10e-09 | +1.54e-10 | +2.50e-09 | NO | yes |
| 20 | 3 | 2.2092623994 | 2.2092624015 | -2.14e-09 | +1.47e-14 | +1.84e-10; -1.19e-12 | +1.53e-10; -3.99e-11 | -2.32e-09 | -5.23e-09 | +1.56e-10 | +2.29e-09 | NO | yes |
| 20 | 4 | 2.0160953082 | 2.0160953107 | -2.55e-09 | +3.44e-12 | +1.83e-10; -1.40e-12 | +1.63e-10; -3.09e-11 | -2.74e-09 | -5.15e-09 | +1.51e-10 | +2.71e-09 | NO | yes |
| 20 | 5 | 2.1310534340 | 2.1310534361 | -2.16e-09 | +2.77e-13 | +1.79e-10; -1.01e-12 | +1.84e-10; -2.48e-11 | -2.34e-09 | -5.16e-09 | +1.55e-10 | +2.32e-09 | NO | yes |
| 20 | 6 | 0.8920895332 | 0.8920895362 | -2.93e-09 | -4.44e-13 | +1.89e-10; -1.34e-12 | +1.72e-10; -2.50e-11 | -3.12e-09 | -5.19e-09 | +1.51e-10 | +3.08e-09 | NO | yes |
| 20 | 7 | 0.8674684896 | 0.8674684922 | -2.59e-09 | +1.73e-13 | +1.91e-10; -2.55e-12 | +1.70e-10; -2.36e-11 | -2.78e-09 | -5.16e-09 | +1.55e-10 | +2.75e-09 | NO | yes |
| 20 | 8 | 0.8424884968 | 0.8424885001 | -3.30e-09 | -8.94e-13 | +1.86e-10; -1.92e-12 | +1.77e-10; -2.11e-11 | -3.48e-09 | -5.11e-09 | +1.56e-10 | +3.45e-09 | NO | yes |
| 20 | 9 | 0.8171324179 | 0.8171324207 | -2.80e-09 | +1.18e-13 | +1.86e-10; -4.43e-13 | +1.78e-10; -2.12e-11 | -2.98e-09 | -5.00e-09 | +1.52e-10 | +2.95e-09 | NO | yes |
| 20 | 10 | 0.7918153330 | 0.7918153363 | -3.36e-09 | -3.25e-12 | +1.96e-10; -1.16e-12 | +1.85e-10; -2.09e-11 | -3.56e-09 | -5.06e-09 | +1.56e-10 | +3.51e-09 | NO | yes |
| 28 | 1 | 2.5350866430 | 2.5350866451 | -2.02e-09 | -6.39e-14 | +1.91e-10; -3.94e-13 | +1.52e-10; -4.08e-11 | -2.21e-09 | -5.77e-09 | +1.61e-10 | +2.18e-09 | NO | yes |
| 28 | 2 | 2.4036282952 | 2.4036282974 | -2.18e-09 | -2.03e-13 | +1.83e-10; -1.28e-12 | +1.50e-10; -4.70e-11 | -2.36e-09 | -5.85e-09 | +1.60e-10 | +2.34e-09 | NO | yes |
| 28 | 3 | 2.2075788076 | 2.2075788106 | -3.03e-09 | +1.42e-12 | +1.88e-10; -1.32e-12 | +1.61e-10; -3.83e-11 | -3.22e-09 | -6.00e-09 | +1.59e-10 | +3.19e-09 | NO | yes |
| 28 | 4 | 1.9812701782 | 1.9812701806 | -2.38e-09 | +8.98e-13 | +1.83e-10; -1.03e-12 | +1.78e-10; -2.95e-11 | -2.56e-09 | -5.92e-09 | +1.53e-10 | +2.53e-09 | NO | yes |
| 28 | 5 | 2.1342896710 | 2.1342896737 | -2.65e-09 | -4.20e-13 | +1.96e-10; -1.13e-13 | +1.79e-10; -2.41e-11 | -2.84e-09 | -5.88e-09 | +1.57e-10 | +2.80e-09 | NO | yes |
| 28 | 6 | 0.8889836748 | 0.8889836771 | -2.36e-09 | -2.74e-12 | +1.86e-10; -1.43e-12 | +1.72e-10; -2.42e-11 | -2.55e-09 | -5.91e-09 | +1.51e-10 | +2.51e-09 | NO | yes |
| 28 | 7 | 0.8636345104 | 0.8636345139 | -3.50e-09 | -4.94e-13 | +1.88e-10; -9.63e-13 | +1.81e-10; -2.28e-11 | -3.68e-09 | -5.89e-09 | +1.53e-10 | +3.65e-09 | NO | yes |
| 28 | 8 | 0.8375874791 | 0.8375874825 | -3.31e-09 | -8.60e-12 | +1.90e-10; -1.64e-12 | +1.81e-10; -2.04e-11 | -3.50e-09 | -5.86e-09 | +1.59e-10 | +3.47e-09 | NO | yes |
| 28 | 9 | 0.8113927553 | 0.8113927582 | -2.93e-09 | -2.23e-12 | +1.88e-10; -1.08e-12 | +1.83e-10; -2.02e-11 | -3.12e-09 | -5.73e-09 | +1.58e-10 | +3.09e-09 | NO | yes |
| 28 | 10 | 0.7857944256 | 0.7857944286 | -3.03e-09 | -4.21e-13 | +1.90e-10; -1.03e-12 | +1.83e-10; -1.96e-11 | -3.22e-09 | -5.83e-09 | +1.57e-10 | +3.19e-09 | NO | yes |
| 35 | 1 | 2.6691634261 | 2.6691634288 | -2.74e-09 | +1.82e-13 | +1.80e-10; -6.93e-13 | +1.52e-10; -4.08e-11 | -2.92e-09 | unavailable | +2.08e-10 | +2.95e-09 | NO | yes |
| 35 | 2 | 2.4532191149 | 2.4532191180 | -3.17e-09 | -1.03e-12 | +1.94e-10; -1.65e-12 | +1.50e-10; -4.70e-11 | -3.37e-09 | unavailable | +2.06e-10 | +3.38e-09 | NO | yes |
| 35 | 3 | 2.2188123146 | 2.2188123170 | -2.37e-09 | -7.24e-13 | +1.87e-10; -1.74e-12 | +1.61e-10; -3.83e-11 | -2.56e-09 | unavailable | +2.11e-10 | +2.58e-09 | NO | yes |
| 35 | 4 | 1.9738829273 | 1.9738829294 | -2.11e-09 | -3.81e-13 | +1.99e-10; -1.08e-12 | +1.78e-10; -2.95e-11 | -2.30e-09 | unavailable | +2.06e-10 | +2.31e-09 | NO | yes |
| 35 | 5 | 2.1065251553 | 2.1065251580 | -2.66e-09 | +4.16e-13 | +1.87e-10; -6.97e-13 | +1.79e-10; -2.41e-11 | -2.85e-09 | unavailable | +2.08e-10 | +2.87e-09 | NO | yes |
| 35 | 6 | 0.8894777882 | 0.8894777904 | -2.19e-09 | -3.55e-13 | +1.87e-10; -1.90e-12 | +1.72e-10; -2.42e-11 | -2.37e-09 | unavailable | +2.10e-10 | +2.40e-09 | NO | yes |
| 35 | 7 | 0.8639991967 | 0.8639991994 | -2.66e-09 | -1.97e-12 | +1.83e-10; -1.04e-12 | +1.81e-10; -2.28e-11 | -2.84e-09 | unavailable | +2.03e-10 | +2.86e-09 | NO | yes |
| 35 | 8 | 0.8378003380 | 0.8378003402 | -2.24e-09 | +1.62e-12 | +1.95e-10; -5.61e-13 | +1.81e-10; -2.04e-11 | -2.44e-09 | unavailable | +2.05e-10 | +2.45e-09 | NO | yes |
| 35 | 9 | 0.8114893498 | 0.8114893531 | -3.33e-09 | -3.06e-12 | +1.90e-10; -1.80e-12 | +1.83e-10; -2.02e-11 | -3.52e-09 | unavailable | +2.04e-10 | +3.53e-09 | NO | yes |
| 35 | 10 | 0.7858966023 | 0.7858966050 | -2.75e-09 | +4.19e-12 | +1.90e-10; -1.60e-12 | +1.83e-10; -1.96e-11 | -2.94e-09 | unavailable | +2.07e-10 | +2.97e-09 | NO | yes |
| 40 | 1 | 2.9586910009 | 2.9586910034 | -2.47e-09 | -6.15e-13 | +1.86e-10; -1.73e-12 | +1.65e-10; -3.96e-11 | -2.66e-09 | unavailable | +2.08e-10 | +2.68e-09 | NO | yes |
| 40 | 2 | 2.5224248514 | 2.5224248544 | -2.99e-09 | -6.31e-14 | +1.90e-10; -4.11e-13 | +1.57e-10; -4.54e-11 | -3.18e-09 | unavailable | +2.07e-10 | +3.19e-09 | NO | yes |
| 40 | 3 | 2.2187076620 | 2.2187076646 | -2.60e-09 | +3.93e-13 | +1.93e-10; -8.17e-13 | +1.63e-10; -3.69e-11 | -2.80e-09 | unavailable | +2.06e-10 | +2.81e-09 | NO | yes |
| 40 | 4 | 2.0135420528 | 2.0135420554 | -2.66e-09 | +8.16e-13 | +1.91e-10; -7.74e-13 | +1.72e-10; -2.84e-11 | -2.85e-09 | unavailable | +2.01e-10 | +2.86e-09 | NO | yes |
| 40 | 5 | 2.1265191849 | 2.1265191881 | -3.19e-09 | -1.63e-13 | +1.87e-10; -7.85e-13 | +1.78e-10; -2.31e-11 | -3.38e-09 | unavailable | +2.04e-10 | +3.40e-09 | NO | yes |
| 40 | 6 | 0.8851737592 | 0.8851737617 | -2.55e-09 | -4.91e-13 | +1.90e-10; -6.29e-13 | +1.70e-10; -2.30e-11 | -2.74e-09 | unavailable | +2.07e-10 | +2.75e-09 | NO | yes |
| 40 | 7 | 0.8589088914 | 0.8589088936 | -2.21e-09 | -6.99e-14 | +1.91e-10; -8.77e-13 | +1.72e-10; -2.15e-11 | -2.40e-09 | unavailable | +2.03e-10 | +2.41e-09 | NO | yes |
| 40 | 8 | 0.8320304192 | 0.8320304217 | -2.53e-09 | +6.75e-13 | +1.78e-10; -1.50e-12 | +1.82e-10; -1.92e-11 | -2.71e-09 | unavailable | +2.05e-10 | +2.73e-09 | NO | yes |
| 40 | 9 | 0.8055675769 | 0.8055675799 | -2.93e-09 | +1.63e-12 | +1.93e-10; -6.57e-13 | +1.81e-10; -1.87e-11 | -3.12e-09 | unavailable | +2.09e-10 | +3.14e-09 | NO | yes |
| 40 | 10 | 0.7790199867 | 0.7790199892 | -2.48e-09 | -3.18e-13 | +1.87e-10; -7.02e-13 | +1.77e-10; -1.85e-11 | -2.67e-09 | unavailable | +2.07e-10 | +2.69e-09 | NO | yes |

### Maxima over matched clocks

- max |d C| = 3.497e-09
- max |d C_complement| = 2.450e-11
- max |d held| = 1.990e-10
- max |d held_complement| = 4.451e-12
- max |d flat| = 1.853e-10
- max |d flat_complement| = 7.810e-11
- max |d excess_polar| = 3.685e-09
- max |r1 ext-int| = 2.110e-10; max |r2 polar ext-int| = 3.650e-09
- max |d C vs r2 raw| = 6.003e-09

### T12 charge-fallback control, as route 2 presents it

Route 2 labels T12 a charge-fallback control (no applied trapping root; cut at R_ch/a = 2g(Ea) = 58.4668, 58 interior sites, complex128, raw and polar both accepted) and excludes it from its applied-root tables. Route 1 cuts T12 at the record's 2gE = 58.4668 a, also 58 interior sites, and lists it among its rows (the paper's range 109 to 311 is set by T15 and T40, not T12). This is the one clock where both routes' primary cuts coincide. Weighted excess: route 1 285.101126742, route 2 polar 285.101129312 (difference -2.57e-6), route 2 raw 285.101130522 (difference -3.78e-6); held sums differ by +1.63e-7 and flat sums by +1.31e-7; on the complement side the excess differs by -1.95e-9. Per channel (table above): the exterior S_X(C) differs by -2.3e-9 to -3.5e-9, the complement side by at most 2.4e-11, held by +1.7e-10 to +2.0e-10 (exterior) and at most 4.5e-12 (complement).

### Not comparable (different cuts)

At T16, T24, T25 and T30 route 1's partition (50, 38, 37, 37 interior sites) is 2 or 3 sites outside route 2's applied root, and route 2 saved no partition there. The two routes' numbers at these clocks belong to different cuts and are listed side by side only (second table above): route 1 117.97, 286.19, 307.36, 295.70 against route 2's applied-root polar 149.76, 292.06, 323.76, 298.96. No agreement or disagreement is stated for them.

The sigma/a 8 range (116 to 305 nats in the paper) has no second-route counterpart: route 2 covers only the sigma/a 4 production.

## 5. Statement

At the six clocks where both routes saved the same partition (T12 at the common 2gE cut; T15, T20, T28, T35, T40 at route 1's record-root partition, which is route 2's +1-site partition), they agree at about 1e-8 relative. The weighted excess differs by -2.49e-6 to -2.73e-6 nats, against values of 109 to 311 nats. That is outside the larger stated error (route 1's 2.1e-10 per channel, 1.85e-7 on the weighted sum) by about a factor 14. So against the stated errors, the exterior-side numbers formally disagree. The difference is the same sign at all 60 channel rows, -2.0e-9 to -3.5e-9 per channel in S_X(C), and it sits entirely in route 2's exterior-block entropy of C. Route 2's own exterior minus interior entropy for that pure state is +2.2e-9 to +3.7e-9 per channel, while route 1's is at most 2.1e-10. On the complement side the two routes agree within the stated error at every channel and clock: S_Xc(C) to at most 2.5e-11, held to 4.5e-12, flat to 7.8e-11, and the weighted complement-side excess to at most 4.8e-9 (limit 1.85e-7). The held and flat exterior entropies agree within 2.1e-10 at every channel (at most 1.99e-10). Route 2's raw (non-orthonormalized) C, available only at T12, T20 and T28, is a different object from route 1's Loewdin C and differs from it by 4.0e-9 to 6.0e-9 per channel. The paper's endpoints are both matched clocks: 109.355861 (route 1, T15) against 109.355863 (route 2, +1 site), and 310.888400 (route 1, T40) against 310.888403 (route 2, +1 site). At T16, T24, T25 and T30 the routes use different cuts and are not compared. The paper's sentence names the record root as its cut ("the record's outer dense root or its 2GE"), and route 2's primary applied-root cut is a different cut, 1 to 3 sites further in.

## 6. Scope

Compared: route 1's prod (sigma/a 4) per-channel stored entropies S_X(C), S_Xc(C), S_X(P_-), S_Xc(P_-), S_X(P_0), S_Xc(P_0), each channel's excess and the 16-kappa weighted sums for kappa 1 to 10. They are set against route 2's saved per-channel polar-control exterior and interior entropies, raw exterior entropy where gated valid, held and flat entropies, at equal interior site counts, on the same clocks, fields, weights and channel set. Not compared: (i) T16, T24, T25, T30 (no common partition); (ii) the sigma/a 8 member and the ladder (no route 2); (iii) area normalizations and ratios (eta A 47312 at R = 64 a against eta_* A_lat at the applied root, different definitions); (iv) route 1's t_X, nu, rank and premise columns and route 2's r_held_10 and gate diagnostics (no counterpart, or different quantities); (v) the slice files' byte identity (route 1's path was removed after its run; the T12 agreement on the complement side to 1e-11 is consistent with the same input, not a proof of it); (vi) any omitted-channel tail, continued reference, r_ref or physical reading. Nothing was recomputed from the slices; both routes' numbers are their stored values.
