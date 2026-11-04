# W5_W4_formation_attribution_v1: the formation offset's attribution on the (0.2, 64) member: the heavy sea's angular truncation (W5), the pair sources (W4 (b)), the step (tau 4 and tau 2 against tau 8; the conserving step), beside the dust comparator (W4 (a)) and the records' production numbers

Composed by attrib_readout_v1.py (the producer, box CPU, stored files only) and attrib_compose_v1.py (this text, from the producer's json). The builder composes; no sentence here reads the physics. Formation is the first record holding a trapping zero (the ladder reader's rule, MASS_ladder_v1's definitions block), R_t the outer dense root, R_t/(2 g E) the record's own ratio at its charge E; the dust comparator's root is the zero of N - |v| on its dense fields, its fold clock by the integrator's event, its radius over its own 2 g E0 (the comparator conserves E to 1e-9). The engine records sit on a grid of tau/r_s (0.125 at tau 8, 0.0625 at tau 4, 0.03125 at tau 2): a formation clock is read to that grid, and two clocks on the same grid that agree are equal only to its spacing.

## 1 The offset: the production histories against the dust comparator (W4 (a))

| history | quoted T_f / ratio | T_f (records) | R_t/(2gE) | dust comparator run | fold T/r_s | fold R/(2gE0) | first comparator record with a root T/r_s | its outer root/(2gE0) | offset T_f - fold | offset T_f - first record | offset ratio - fold | offset ratio - first record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s2 | 12.000 / 0.65 | 12.0000 | 0.6545 | s2_density_continuum_rest_lapse-on_adaptive | 10.7279 | 0.7699 | 10.7500 | 0.8193 | +1.2721 | +1.2500 | -0.1154 | -0.1648 |
| P64_s4 | 12.125 / 0.66 | 12.1250 | 0.6613 | s4_density_continuum_rest_lapse-on_adaptive | 10.8157 | 0.7490 | 10.8750 | 0.8519 | +1.3093 | +1.2500 | -0.0877 | -0.1906 |
| P64_s8 | 12.750 / 0.76 | 12.7500 | 0.7620 | s8_density_continuum_rest_lapse-on_adaptive | 10.9879 | 0.5847 | 11.0000 | 0.6867 | +1.7621 | +1.7500 | +0.1773 | +0.0754 |

The quoted production numbers (12.000/0.65, 12.125/0.66, 12.750/0.76) are reproduced by the stored records to their printed digits. The comparator's variants (its own step, dispersion, kernel, release and lapse) are in section 4.

## 2 The attribution table: each switch against its reference at formation (switch minus reference)

| switch | history | reference | T_f switch / reference | dT_f (r_s) | dR_t (a) | d(R_t/(2gE)) | record grid of the switch (r_s) | both certified |
|---|---|---|---|---|---|---|---|---|
| angular truncation (W5) | W5_k14_tau8 | W5_k10_tau8 | 12.1250 / 12.1250 | +0.0000 | +0.005 | +0.0002 | 0.125 | yes |
| angular truncation (W5) | W5_k20_tau8 | W5_k10_tau8 | 12.1250 / 12.1250 | +0.0000 | -0.086 | -0.0012 | 0.125 | yes |
| angular truncation (W5) | W5_k14_tau4 | W5_k10_tau4 | 12.0000 / 12.0000 | +0.0000 | -0.016 | -0.0002 | 0.0625 | yes |
| angular truncation (W5) | W5_k20_tau4 | W5_k10_tau4 | 12.0000 / 12.0000 | +0.0000 | -0.030 | -0.0003 | 0.0625 | yes |
| W5's reference against the production (the same arguments, the W5 launcher) | W5_k10_tau8 | P64_s4 | 12.1250 / 12.1250 | +0.0000 | +0.000 | +0.0000 | 0.125 | yes |
| pair sources off (W4 (b)) | W4b_pairsoff | P64_s4 | 12.1250 / 12.1250 | +0.0000 | -0.003 | -0.0002 | 0.125 | yes |
| step tau 4 and tau 2 against tau 8 | W5_k10_tau4 | W5_k10_tau8 | 12.0000 / 12.1250 | -0.1250 | -1.233 | -0.0253 | 0.0625 | yes |
| step tau 4 and tau 2 against tau 8 | W5_k14_tau4 | W5_k14_tau8 | 12.0000 / 12.1250 | -0.1250 | -1.254 | -0.0257 | 0.0625 | yes |
| step tau 4 and tau 2 against tau 8 | W5_k20_tau4 | W5_k20_tau8 | 12.0000 / 12.1250 | -0.1250 | -1.177 | -0.0244 | 0.0625 | yes |
| step tau 4 and tau 2 against tau 8 | P64_s4_tau4 | P64_s4 | 12.0000 / 12.1250 | -0.1250 | -1.233 | -0.0253 | 0.0625 | yes |
| step tau 4 and tau 2 against tau 8 | P64_s4_tau2 | P64_s4 | 11.9062 / 12.1250 | -0.2188 | -3.887 | -0.0726 | 0.03125 | yes |
| the conserving step (W2's dg history) | W2_dg | P64_s4 | 11.8750 / 12.1250 | -0.2500 | -2.429 | -0.0497 | 0.125 | yes |
| averaging scale (the stored spread) | P64_s2 | P64_s4 | 12.0000 / 12.1250 | -0.1250 | -0.458 | -0.0069 | 0.125 | yes |
| averaging scale (the stored spread) | P64_s8 | P64_s4 | 12.7500 / 12.1250 | +0.6250 | +5.971 | +0.1007 | 0.125 | yes |

Against the sigma/a 4 offset T_f - fold = +1.3093 r_s (T_f - first comparator record +1.2500 r_s; ratio -0.0877 against the fold): the switches above move the engine's formation clock by the amounts in the dT_f column, each read on its record grid.

## 3 W5: the heavy sea's angular truncation (kappa_sea 10, 14, 20 at tau 8 and tau 4, from the common preparation, to T/r_s 14)

| rung | kappa_sea | tau/a | formation T/r_s | R_t/a | R_t/(2gE) | E/E0 | gate sigma/R_t | residual | certification | last record | last converged | excluded records | E0 (head) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W5_k10_tau8 | 10 | 8.0 | 12.1250 | 38.556 | 0.6613 | 0.98702 | 0.104 | 6.41e-07 | certified (the record counts and lies inside the 30.3 prefix) | 14.0000 | 14.0000 | none | 118.137862256 |
| W5_k14_tau8 | 14 | 8.0 | 12.1250 | 38.561 | 0.6615 | 0.98686 | 0.104 | 2.53e-08 | certified (the record counts and lies inside the 30.3 prefix) | 14.0000 | 14.0000 | none | 118.137862256 |
| W5_k20_tau8 | 20 | 8.0 | 12.1250 | 38.470 | 0.6601 | 0.98661 | 0.104 | 1.79e-07 | certified (the record counts and lies inside the 30.3 prefix) | 14.0000 | 13.8750 | 13.5000, 14.0000 | 118.137862256 |
| W5_k10_tau4 | 10 | 4.0 | 12.0000 | 37.323 | 0.6360 | 0.99341 | 0.107 | 4.00e-07 | certified (the record counts and lies inside the 30.3 prefix) | 14.0000 | 14.0000 | 13.4375 | 118.137862256 |
| W5_k14_tau4 | 14 | 4.0 | 12.0000 | 37.307 | 0.6359 | 0.99329 | 0.107 | 1.83e-08 | certified (the record counts and lies inside the 30.3 prefix) | 14.0000 | 14.0000 | none | 118.137862256 |
| W5_k20_tau4 | 20 | 4.0 | 12.0000 | 37.293 | 0.6357 | 0.99310 | 0.107 | 1.23e-07 | certified (the record counts and lies inside the 30.3 prefix) | 14.0000 | 12.9375 | 13.0000, 13.1250 | 118.137862256 |

- W5_k20_tau8: the record at 14 does not count (last converged record 13.8750; capped records 13.5000, 14.0000); its slice readout at 14 below is that record's, labelled.
- W5_k20_tau4: the record at 14 does not count (last converged record 12.9375; first above 1e-3 at 13.0000); its slice readout at 14 below is that record's, labelled.

### 3a The pairs at T/r_s 14 (per channel kappa: the pair count w (n_sea - ||R2^dag S||^2) and the bare energy w (Tr h P_sea - Tr h P_R2); the slice readout of w5_readout_v1)

| rung | pair count per channel | bare pair energy per channel | sum pair count | sum bare pair energy | sum Killing pair energy |
|---|---|---|---|---|---|
| W5_k10_tau4 | 1:5.10e-02 2:1.62e-02 3:3.87e-03 4:1.96e-03 5:2.19e-03 6:3.07e-03 7:4.21e-03 8:5.94e-03 9:9.16e-03 10:1.33e-02 | 1:6.34e-02 2:2.42e-02 3:9.19e-03 4:1.02e-02 5:1.05e-02 6:9.61e-03 7:1.47e-02 8:1.79e-02 9:2.70e-02 10:3.42e-02 | 1.11e-01 | 2.21e-01 | 5.88e-02 |
| W5_k10_tau8 | 1:5.23e-02 2:5.41e-02 3:2.74e-02 4:3.54e-02 5:4.38e-02 6:5.21e-02 7:5.58e-02 8:6.21e-02 9:6.13e-02 10:5.46e-02 | 1:7.17e-02 2:2.45e-02 3:4.98e-02 4:6.92e-02 5:7.02e-02 6:1.03e-01 7:9.18e-02 8:1.22e-01 9:1.34e-01 10:7.79e-02 | 4.99e-01 | 8.14e-01 | 2.53e-01 |
| W5_k14_tau4 | 1:5.07e-02 2:1.61e-02 3:3.86e-03 4:1.95e-03 5:2.17e-03 6:3.04e-03 7:4.21e-03 8:5.95e-03 9:9.13e-03 10:1.32e-02 11:1.61e-02 12:1.72e-02 13:1.68e-02 14:1.63e-02 | 1:6.30e-02 2:2.41e-02 3:9.25e-03 4:1.04e-02 5:1.02e-02 6:1.01e-02 7:1.48e-02 8:1.79e-02 9:2.71e-02 10:3.34e-02 11:3.82e-02 12:4.05e-02 13:3.76e-02 14:3.77e-02 | 1.77e-01 | 3.74e-01 | 1.53e-01 |
| W5_k14_tau8 | 1:5.18e-02 2:1.07e-01 3:2.71e-02 4:3.50e-02 5:4.33e-02 6:5.15e-02 7:5.51e-02 8:6.12e-02 9:6.05e-02 10:5.41e-02 11:4.82e-02 12:4.11e-02 13:3.58e-02 14:3.17e-02 | 1:7.17e-02 2:-2.91e-02 3:4.97e-02 4:6.93e-02 5:6.97e-02 6:1.02e-01 7:9.09e-02 8:1.21e-01 9:1.31e-01 10:7.65e-02 11:8.94e-02 12:9.75e-02 13:9.22e-02 14:1.02e-01 | 7.03e-01 | 1.13e+00 | 4.01e-01 |
| W5_k20_tau4 | 1:5.03e-02 2:1.60e-02 3:3.91e-03 4:2.09e-03 5:2.32e-03 6:3.24e-03 7:4.45e-03 8:6.22e-03 9:9.36e-03 10:1.33e-02 11:1.63e-02 12:1.73e-02 13:1.69e-02 14:1.65e-02 15:1.63e-02 16:1.70e-02 17:1.91e-02 18:2.01e-02 19:2.03e-02 20:1.98e-02 | 1:6.24e-02 2:2.38e-02 3:9.41e-03 4:1.12e-02 5:9.15e-03 6:1.16e-02 7:1.42e-02 8:1.91e-02 9:2.88e-02 10:3.15e-02 11:4.04e-02 12:3.95e-02 13:3.74e-02 14:3.83e-02 15:4.06e-02 16:3.31e-02 17:5.39e-02 18:3.67e-02 19:6.23e-02 20:3.98e-02 | 2.91e-01 | 6.43e-01 | 2.91e-01 |
| W5_k20_tau8 | 1:5.11e-02 2:9.78e-01 3:2.69e-02 4:3.49e-02 5:4.30e-02 6:5.12e-02 7:5.47e-02 8:6.03e-02 9:5.97e-02 10:5.38e-02 11:4.75e-02 12:4.07e-02 13:3.53e-02 14:3.15e-02 15:2.88e-02 16:2.71e-02 17:2.60e-02 18:2.65e-02 19:2.58e-02 20:2.49e-02 | 1:7.20e-02 2:-9.09e-01 3:5.17e-02 4:7.10e-02 5:6.93e-02 6:1.04e-01 7:9.25e-02 8:1.21e-01 9:1.29e-01 10:7.98e-02 11:9.12e-02 12:9.83e-02 13:9.18e-02 14:1.03e-01 15:1.11e-01 16:1.06e-01 17:8.20e-02 18:8.82e-02 19:7.34e-02 20:6.03e-02 | 1.73e+00 | 7.86e-01 | 5.82e-01 |

### 3b The inner-shell share above pi/4 and the 16 kappa E_kappa tail at T/r_s 14

| rung | R_50 (a) | inner-shell dust count | inner-shell share above pi/4 | whole dust share above pi/4 | sum 16 kappa E_kappa (pairs) | sum 16 kappa E_kappa (whole heavy) | top channel's share of the pairs' tail | share of channels above 10 | 16 kappa E_kappa (pairs) per channel | eigenvalues against the slice's stored (max) | pockets (occupied positive) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W5_k10_tau4 | 22.0 | 256.22 | 0.5126 | 0.2348 | 5.90e-02 | 3.96e+01 | 0.3327 | 0.0000 | 1:3.59e-03 2:1.48e-03 3:9.62e-04 4:1.24e-03 5:2.08e-03 6:3.46e-03 7:5.33e-03 8:8.06e-03 9:1.32e-02 10:1.96e-02 | 2.44e-15 | 6 |
| W5_k10_tau8 | 19.0 | 252.22 | 0.5286 | 0.2549 | 2.54e-01 | 3.73e+01 | 0.1511 | 0.0000 | 1:5.95e-03 2:9.76e-03 3:1.47e-02 4:2.06e-02 5:2.61e-02 6:3.09e-02 7:3.34e-02 8:3.68e-02 9:3.74e-02 10:3.84e-02 | 2.89e-15 | 6 |
| W5_k14_tau4 | 22.0 | 258.47 | 0.5108 | 0.2346 | 1.53e-01 | 3.96e+01 | 0.1438 | 0.6174 | 1:3.55e-03 2:1.47e-03 3:9.54e-04 4:1.22e-03 5:2.07e-03 6:3.45e-03 7:5.32e-03 8:8.06e-03 9:1.31e-02 10:1.95e-02 11:2.38e-02 12:2.51e-02 13:2.37e-02 14:2.21e-02 | 2.22e-15 | 6 |
| W5_k14_tau8 | 19.0 | 256.97 | 0.5269 | 0.2555 | 4.03e-01 | 3.73e+01 | 0.0845 | 0.3725 | 1:5.86e-03 2:1.05e-02 3:1.46e-02 4:2.05e-02 5:2.59e-02 6:3.07e-02 7:3.31e-02 8:3.64e-02 9:3.69e-02 10:3.81e-02 11:4.02e-02 12:3.87e-02 13:3.70e-02 14:3.40e-02 | 2.66e-15 | 6 |
| W5_k20_tau4 | 22.0 | 260.87 | 0.5080 | 0.2339 | 2.91e-01 | 3.96e+01 | 0.0803 | 0.7969 | 1:3.50e-03 2:1.47e-03 3:9.78e-04 4:1.28e-03 5:2.12e-03 6:3.52e-03 7:5.41e-03 8:8.19e-03 9:1.32e-02 10:1.95e-02 11:2.39e-02 12:2.52e-02 13:2.39e-02 14:2.22e-02 15:2.11e-02 16:2.12e-02 17:2.33e-02 18:2.40e-02 19:2.40e-02 20:2.34e-02 | 3.00e-15 | 6 |
| W5_k20_tau8 | 19.0 | 262.91 | 0.5256 | 0.2563 | 5.99e-01 | 3.74e+01 | 0.0511 | 0.5552 | 1:5.78e-03 2:2.50e-02 3:1.46e-02 4:2.05e-02 5:2.59e-02 6:3.07e-02 7:3.30e-02 8:3.61e-02 9:3.66e-02 10:3.82e-02 11:3.98e-02 12:3.86e-02 13:3.67e-02 14:3.39e-02 15:3.17e-02 16:3.01e-02 17:2.98e-02 18:3.09e-02 19:3.02e-02 20:3.06e-02 | 2.89e-15 | 6 |

Definitions (w5_readout_v1's): R_50 the radius enclosing half of the fields row's dust energy at 14; the share the number-weighted fraction of the inner dust on lattice momenta with |sin a k| > sin(pi/4); E_kappa^pairs = Tr[|h_m| (P_sea - P_R2)^2], E_kappa^heavy = Tr[|h_m| (C - P_-(h_m))^2] on the record's clocked operator h_m, each weighted by 16 kappa.

## 4 W4 (a): the dust comparator's runs (dust_comparator_v1c on the production's stored T = 0 profile)

| run (sigma, kernel, dispersion, release, lapse, step) | fold T/r_s | fold R (a) | fold R/(2gE0) | first record with a root T/r_s | its outer root (a) | /(2gE0) | central caustic T/r_s | E drift net (max) | end |
|---|---|---|---|---|---|---|---|---|---|
| s2_density_continuum_rest_lapse-on_adaptive | 10.7279 | 45.011 | 0.7699 | 10.7500 | 47.900 | 0.8193 | 11.1469 | 1.65e-10 | completed to T/r_s 16.0000 |
| s4_density_continuum_rest_lapse-on_adaptive | 10.8157 | 43.794 | 0.7490 | 10.8750 | 49.814 | 0.8519 | 11.1471 | 2.88e-10 | completed to T/r_s 16.0000 |
| s8_density_continuum_rest_lapse-on_adaptive | 10.9879 | 34.200 | 0.5847 | 11.0000 | 40.165 | 0.6867 | 11.2170 | 3.40e-10 | completed to T/r_s 16.0000 |
| s4_density_continuum_rest_lapse-on_frozen-tau8 | 10.9529 | 44.269 | 0.7571 | 11.0000 | 51.122 | 0.8743 | 11.5010 | 5.33e-03 | completed to T/r_s 16.0000 on frozen-field steps of tau 8 a |
| s4_density_continuum_rest_lapse-on_frozen-tau4 | 10.8850 | 42.315 | 0.7237 | 10.9375 | 50.540 | 0.8644 | 11.3138 | 2.45e-03 | completed to T/r_s 16.0000 on frozen-field steps of tau 4 a |
| s0.25_density_continuum_rest_lapse-on_adaptive | 10.6820 | 43.992 | 0.7525 | 10.7500 | 49.216 | 0.8418 | 10.9417 | 4.97e-10 | completed to T/r_s 16.0000 |
| s2_density_sine_rest_lapse-on_adaptive | 10.6781 | 45.408 | 0.7760 | 10.7500 | 50.191 | 0.8577 | 11.1227 | 1.74e-10 | completed to T/r_s 16.0000 |
| s4_density_sine_rest_lapse-on_adaptive | 10.7677 | 44.345 | 0.7577 | 10.8750 | 51.567 | 0.8811 | 11.1141 | 2.77e-10 | completed to T/r_s 16.0000 |
| s8_density_sine_rest_lapse-on_adaptive | 10.9477 | 35.420 | 0.6050 | 11.0000 | 46.307 | 0.7910 | 11.1789 | 3.41e-10 | completed to T/r_s 16.0000 |
| s4_radial_continuum_rest_lapse-on_adaptive | 10.7967 | 43.540 | 0.7447 | 10.8750 | 50.363 | 0.8614 | 10.9937 | 9.79e-11 | completed to T/r_s 16.0000 |
| s4_vector_continuum_rest_lapse-on_adaptive | 10.8391 | 44.109 | 0.7543 | 10.8750 | 48.915 | 0.8365 | - | 1.86e-10 | a shell's momentum reached k_max 20 at T/r_s 12.1754 (the central caustic blueshifts a shell's momentum withou |
| s4_density_continuum_read_lapse-on_adaptive | 10.8154 | 43.729 | 0.7479 | 10.8750 | 49.774 | 0.8513 | 11.1453 | 2.76e-10 | completed to T/r_s 16.0000 |
| s2_density_continuum_rest_lapse-off_adaptive | - | - | - | - | - | - | - | 3.76e+00 | a shell's momentum reached k_max 20 at T/r_s 7.7020 (the central caustic blueshifts a shell's momentum without |
| s4_density_continuum_rest_lapse-off_adaptive | - | - | - | - | - | - | - | 4.79e+00 | a shell's momentum reached k_max 20 at T/r_s 9.6137 (the central caustic blueshifts a shell's momentum without |
| s8_density_continuum_rest_lapse-off_adaptive | - | - | - | - | - | - | - | 4.16e+00 | ESCAPE: a shell reached the grid guard 416.5 a at T/r_s 11.6658; the run stopped |

A second copy of the sigma/a 4 runs (busi's fundmarks w4a_pairsoff, default_s4) prints the same fold and first-record numbers to every digit of its log; it is listed in SOURCES and not repeated.

## 5 W4 (b) and the conserving step

| history | label | script field | engine entries | formation T/r_s | R_t/a | R_t/(2gE) | E/E0 | residual | first episode closes | last record | last converged | excluded records |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | production sigma/a 4 (rung 4, v36d from T = 0) | engine_collapse_v34.py | engine_collapse_v25 (0.000 to 40.000) | 12.1250 | 38.556 | 0.6613 | 0.98702 | 6.41e-07 | 17.3750 | 40.0000 | 40.0000 | none |
| W4b_pairsoff | W4 (b): pair sources off (engine v36g), sigma/a 4, tau 8, to 14 | engine_collapse_v34.py | engine_collapse_v25 (0.000 to 14.000) | 12.1250 | 38.553 | 0.6611 | 0.98727 | 0.00e+00 | open | 14.0000 | 14.0000 | none |
| W2_dg | W2: the conserving step (dg, v36f to 12.375; the v36l overlap rule from 12.5) | engine_collapse_v34.py | engine_collapse_v25 (0.000 to 40.000) | 11.8750 | 36.127 | 0.6116 | 1.00000 | 8.85e-07 | 15.8750 | 40.0000 | 40.0000 | 12.6250, 13.6250 |
| P64_s4_tau4 | production sigma/a 4, tau 4 (engine v19) | engine_collapse_v19.py | engine_collapse_v19 (0.000 to 40.000) | 12.0000 | 37.323 | 0.6360 | 0.99341 | 4.00e-07 | 16.3750 | 40.0000 | 40.0000 | 13.4375 |
| P64_s4_tau2 | production sigma/a 4, tau 2 (engine v25) | engine_collapse_v25.py | engine_collapse_v25 (0.000 to 21.500) | 11.9062 | 34.669 | 0.5888 | 0.99689 | 2.20e-07 | 16.0938 | 21.5000 | 12.7500 | 12.7812, 12.8125, 13.4062 |

- W4 (b) ran on engine v36g with --pair-sources off (its launch line in <internal>/STATUS); 113 records to 14; its formation record's residual prints as stored (0.0).
- W2's history carries the conserving (dg) step on v36f through 12.375, a restart at 12.5 and the v36l overlap rule from 12.625 (W2_v36l_readout_v1); its formation 11.8750 lies in the dg part; its non-counting records: 12.6250, 13.6250.
- The sigma/a 2 production point is the composite history (Newton across its seams); the plain rung 2 is not on the box.
- The tau 4 step control appears twice: the production's tau 4 record (engine v19) and W5's kappa_sea 10 tau 4 rung (v36d); both formation rows print in section 2 (12.0000 at 37.3229 a and 12.0000 at 37.3229 a).
- W4 (a) has run (busi's dust_comparator_v1c, 15 runs on the stored T = 0 profile of the W2 member, whose preparation's E0 equals the production's); the lapse-off runs end before any root (their end column).

## 6 SOURCES

| history | run json | bytes | md5 |
|---|---|---|---|
| P64_s2 | <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json | 7014125 | 0a005ac5f66bef02a0104e4278867765 |
| P64_s4 | <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json | 4857142 | d7d69adcea7a43f8375ab4c7c5a80efc |
| P64_s8 | <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json | 4650292 | d1ca037a77c198aa83b33055495eceff |
| P64_s4_tau4 | <internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg.json | 9046968 | f2e1d8a7483447426855e3211535a367 |
| P64_s4_tau2 | <internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg.json | 9405229 | 9ba6f6ff302d8a903925f20583c96656 |
| W5_k10_tau8 | <internal>/collapse_lad_W5_k10_tau8_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 1583334 | 6d04fb0698c2a87d6f5f739456f26b3c |
| W5_k14_tau8 | <internal>/collapse_lad_W5_k14_tau8_m0p2_rs64_n2048_R04_k14_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 1869878 | b0ab8831a9f48ca269298dcc0c971c65 |
| W5_k20_tau8 | <internal>/collapse_lad_W5_k20_tau8_m0p2_rs64_n2048_R04_k20_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 2307225 | e231f0e4b35de310699fa2791425b1be |
| W5_k10_tau4 | <internal>/collapse_lad_W5_k10_tau4_m0p2_rs64_n2048_R04_k10_tau4_ell4_prep4_A_R2_avg_plain_v36d.json | 3137608 | dd11691d63ef9995f6010b789793a846 |
| W5_k14_tau4 | <internal>/collapse_lad_W5_k14_tau4_m0p2_rs64_n2048_R04_k14_tau4_ell4_prep4_A_R2_avg_plain_v36d.json | 3707716 | d6a3e67886a7731037c32ca99bccea84 |
| W5_k20_tau4 | <internal>/collapse_lad_W5_k20_tau4_m0p2_rs64_n2048_R04_k20_tau4_ell4_prep4_A_R2_avg_plain_v36d.json | 4570206 | 0d4af63403bf64b4dd5c56cedd48aca2 |
| W4b_pairsoff | <internal>/collapse_w4b_pairsoff_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g.json | 1595401 | 51cd019bf750e6f33ef7f65342b6f5fd |
| W2_dg | <internal>/collapse_lad_W2_dg_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36f.json | 13132001 | 1d4b69ef8ebed850f04f66a94b40b680 |

- W5 slice readout: busi's w5_readout_v1.py (md5 fc214952b6383b52267ce8bfc81b5870, copied unchanged to <internal>/attrib) run on the box CPU over the 6 rungs' T0014.00 slices; its outputs <internal>/w5_readout_v1.json (md5 6098f449ceff9cd4640160b9ed72c0d8) and w5_readout_v1.md beside it; engine_vacflow_v1.py md5 5225e959b9e0bc730062041d60d800c8
- W4 (a) outputs: <internal>/default (DUST_comparator_v1c.md md5 9ce4747c5559415584fbfcc305801e12; the run jsons' md5 in attrib_readout_v1.json; manifest <internal>/MANIFEST_w4a.md5); the copy <internal>/default_s4
- producer <internal>/attrib_readout_v1.py (md5 fdcda76e38362a9ac6f41935732aa57d) with the ladder reader <internal>/ladders_readout_v1.py; composer attrib_compose_v1.py (md5 948b8d6f1a1849a000f7bd34b84b76ea); figures attrib_figs_v1.py (md5 3d87996333a22f7e349f1553faf4b011); this list attrib_sources_v1.py (md5 3df003b38fbd07ec9bec14ae65bf6e61)
- outputs <internal>/attrib (attrib_readout_v1.json, attrib_readout_v1.npz, the figure W5_W4_formation_attribution_v1_fig.png), box mirror <internal>/attrib with MANIFEST_attrib_readout_v1.md5
