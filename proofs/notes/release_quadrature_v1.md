## The released ball's outer layer: the flat-slice and proper times to r_g (engine_release_quadrature_v1.py; engine spec 37.3 (3), WEBGPT40F P20 secs. 2, 2.1; record-only)

- read from collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json: g 0.25, the nominal r_s 64 a, the head's release radius R0 256.000000 a and charge E0 118.13786225597 (2 g E0 = 59.068931 a), the T = 0 record's R_sh 238.000 a.
- t = int_R^R0 gamma/z dx, tau = int_R^R0 dx/z, gamma = (e^2 + b^2)/(e + b z), e = sqrt(1 - r_g/R0), b = sqrt(r_g/x), z = sqrt(r_g/x - r_g/R0) (P20 eqs. 6, 10), to R = r_g; the canonical-equation check from rest beside (P20's own check: t 846.3911794961956 and tau 794.7905151009893 lattice units at r_g 59.05).

| case | R_0 (a) | r_g = 2gE (a) | t (a) | tau (a) | t/64 a (nominal T/rs) | tau/64 a | t/(2 g E0) | tau/(2 g E0) | t/r_g | tau/r_g | canonical t (a) | canonical tau (a) | quad error estimates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| the production's release (head R0, head E0) | 256.000000 | 59.068931 | 846.2534331892 | 794.6415607185 | 13.2227098936 | 12.4162743862 | 14.3265404846 | 13.4527838162 | 14.3265404846 | 13.4527838162 | 846.2534331893 | 794.6415607186 | 4.8e-11, 5.0e-11 |
| the production's T = 0 dust radius R_sh (99 percent of the dust energy inside), head E0 | 238.000000 | 59.068931 | 758.0860662073 | 707.6446851799 | 11.8450947845 | 11.0569482059 | 12.8339221945 | 11.9799812129 | 12.8339221945 | 11.9799812129 | 758.0860662074 | 707.6446851800 | 1.6e-11, 1.6e-11 |
| P20's control r_g 59.05 at R_0 256 (its table: 13.2248621796 flat-slice, 12.4186017985 proper, in units of 64 a) | 256.000000 | 59.050000 | 846.3911794962 | 794.7905151010 | 13.2248621796 | 12.4186017985 | 14.3288724433 | 13.4553055206 | 14.3334662065 | 13.4596192227 | 846.3911794963 | 794.7905151011 | 4.8e-11, 5.0e-11 |
| P20's control r_g 64 at R_0 256 (its table: 12.6937264016 flat-slice, 11.8416820247 proper, in units of 64 a) | 256.000000 | 64.000000 | 812.3984897038 | 757.8676495815 | 12.6937264016 | 11.8416820247 | 13.7533975000 | 12.8302245378 | 12.6937264016 | 11.8416820247 | 812.3984897039 | 757.8676495816 | 1.6e-11, 1.6e-11 |

### Beside: the production's recorded formation and fold clocks (T/rs, the nominal r_s 64 a)

| history | tau | first record with a zero (T/rs) | the record before it | m = min(N - abs v) at the two | fold clock (linear) | (quadratic) | in units of 2 g E0: first record, fold clock | the release case's t/64 a minus the first record, minus the fold clock (T/rs) |
|---|---|---|---|---|---|---|---|---|
| tau 8 (the production) | 8 | 12.12500 | 12.00000 | +0.0222, -0.0279 | 12.05530 | 12.05709 | 13.137194, 13.061676 | +1.09771, +1.16741 |
| tau 4 | 4 | 12.00000 | 11.93750 | +0.0001, -0.0276 | 11.93779 | 11.93780 | 13.001759, 12.934352 | +1.22271, +1.28492 |
| tau 2 | 2 | 11.90625 | 11.87500 | +0.0010, -0.0131 | 11.87718 | 11.87722 | 12.900183, 12.868684 | +1.31646, +1.34553 |

- every number above is printed by the script (the quadrature, the canonical check, and the jsons' records); the unit statements are the columns' definitions, nothing is read or ruled.
