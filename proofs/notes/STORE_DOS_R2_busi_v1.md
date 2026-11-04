# STORE DOS SCREEN, route 2 (busi, inbox #722): pocket DOS from the stored cubic symbols

Box: <internal>/store_dos_r2. Code: <internal>/store_dos_r2, byte-identical with box code/ (md5-compared). Spec: deliveries/lead_gaps/STORE_DOS_SCREEN_SPEC_v1.md. Brief: busi/specs/STORE_DOS_R2_brief_v1.md.

Blindness: I did not open any route2 file, STATUS line, code or box directory on this item.

## Verdict
**REJECT** under the preregistered pass bands, judged on the light field only. The bandwidth spread fails at both sizes, at an enclosed 0.55 to 0.60 against a 0.10 band. Light rho at a Delta = 0.01 is also below 0.10 at both sizes. The 64/32 ratio itself is inside its band at every Delta.

The secondary match (a sensitivity check) also rejects. There the ratio is out of band, because the size-32 family never develops an open pocket.

What this rejects is this numerical availability estimate on these stored records. It is a screen of a conditional thermodynamic availability, not a capacity. It neither establishes nor refutes an accessible area store.

Scaling interpretation (the spec's correction): rho_2/rho_1 = sqrt(g_2/g_1) (R_1/R_2)^{3/2}. A constant ratio would test volume-like DOS under the inserted linear budget. It would not independently test memory-area scaling. Both factors are printed beside the ratio table.

Where the count comes from: the primary records lie just past the symbol's root, with s_max = |v|/N about 1.005. There the light count is dominated by strongly tipped valley cones, not by an open Fermi pocket. So N(Delta) is strongly superlinear and g_Delta(0) is not a constant. Only the size-64 secondary record (deep pocket) gives a Delta-flat g. The extra size-128 member (width 8) is in the same near-threshold regime as the primaries.
## What was counted
The symbol is the paper's native cubic symbol frozen at each site (PAPER_flat_3f3f40a2fc3.tex, App. B.5, and Eqs. dm-fermion, dm-current, dm-hamiltonian). It is built on the stored back-averaged applied fields N_applied and v_applied. They are never averaged again. The flow is radial, v(x) = v(|x|) times the unit vector of x, and the fields are interpolated linearly in radius from the native radial sites.

At site x the symbol eigenvalues are omega = s N eps(k) + v.q(k), with s = +1 or -1. Here eps(k)^2 = m^2 + sum_i sin^2 k_i and q_i(k) = sin(2 k_i)/2. Each eigenvalue has multiplicity 2, so there are 4 states per site in all.

"Multiplicities once" is read this way: the Brillouin-zone integral already contains all eight valleys and both spinor doublets. No valley factor 16 and no channel weight 16k is applied. A Delta-independent factor in this convention would rescale rho by its square root; ratios and spreads would not change.

The count is N(|omega| < Delta) = sum over sites x with |x| < R_t of c(x; Delta), where c(x; Delta) = sum over bands of the integral over the zone d^3k/(2 pi)^3 of 1[|omega| < Delta].
- Light field: m = 0, the paper's massless four-component Dirac field with the same operator, clock and flow (Sec. "The radiating species and the light field").
- Heavy field: m a = 0.2, the record's mass.
- Combined: light plus heavy.

Reductions are all exact:
- f = N eps - u.q is pi-periodic in each k_i, which reduces the zone to one valley cell [-pi/2, pi/2)^3 times 8.
- k -> -k maps the s = -1 band onto s = +1, and the same map takes v to -v.
- c is O_h-invariant in the flow direction, so sites are summed as O_h orbits. This was checked numerically to 1e-16 (sd_inspect_v1).

## Quadrature (k-grid)
The cell is integrated in spherical coordinates about the local flow direction, from the valley out to the cell boundary (sd_lib_v2.c_numeric_sph):
- Polar angle: Gauss-Legendre panels on geometric breakpoints at the scale sqrt|1 - s^2|, s = |v|/N, plus a breakpoint at the critical angle arccos(1/s).
- Azimuth: periodic trapezoid.
- Along each ray: f is linear on each segment, and the t^2 weight is integrated exactly over the part of the segment where |f| < Delta. f(0) = 0 exactly, so the tipped cone at the valley is resolved by construction.

Two levels are used: S1 = (6 GL per panel, 32 azimuths, 512 segments) and S2 = (10, 48, 1024). The convergence ladder against S3 = (16, 96, 2048) and the earlier cube-face quadrature is in sd_conv_v1.
- Light, s from 0.5 to 1.05: S1 and S2 agree with S3 to about 1e-5 relative.
- Light at s = 1.5, and heavy: they agree to about 3e-3.
- Total states per site: 3.9998, against an exact 4.

Error charged per numeric orbit:
- Light: 2 |S2 - S1|.
- Heavy: 2 max(|S2 - S1|, |S2 - cube-face 48x48x576|). For heavy, |S2 - S1| alone underestimated |S3 - S2| in the convergence test, so the independent cube-face estimate is added.

Analytic leading cone (light, pocket-free, long semi-axis A(0.04) < 0.05): the error charged is A^2 times the value. This held against S3 at A = 0.03 to 0.11 in sd_conv_v1/sd_inspect_v1.

Heavy exclusion: by the Cauchy bound of App. B.5, f >= N g_s(zeta). Orbits where this bound, on the linear and the cubic fields, exceeds 0.04 contribute exactly 0. The minimum over the S-grid includes the Hoelder allowance (1+s) sqrt(h).

## Records, roots and the matching rule
The inputs are the PK5 R2 manifest's records, reused read-only. Each json and fields npz was md5-checked against <internal>/inputs_md5.tsv, and all six families matched (sd_inspect_v1, out_inspect_v1/inspect.json). Nothing was written into the PK5 directory, and the running PK5 job was not touched.

The rule was declared in code/MATCHING_RULE_v1.txt at 2026-10-29T14:50:19Z, before any count.

The ball is the digital ball |x| < R_t, with R_t the outer root of N - |v| on the dense site fields (the "dense root").

Primary match:
- Rule: the first record in each family at which the dense fields carry a root.
- Size 32: S4_32_s4 j70 (T/r_s 17.5).
- Size 64: W5_64_s4 j97 (T/r_s 12.125).
- The PK5 manifest's earlier size-32 "first rooted" record, j66, is rooted only on the node (geometric) fields. Its dense and applied fields are root-free at j66 to j69, so the symbol there has no pocket and no dense root.

Secondary match:
- Rule: the primary plus 1.25 T/r_s, the largest offset available at size 32.
- Size 32: j75. This is the last dense-rooted size-32 record; the dense root disappears at j76.
- Size 64: j107.
- A primary PASS that fails here would have been reported as NO PASS.

Size 128 (extra; not in the spec's ratio; width 8, not 4): S4_128_s8 j179. The width-16 member was not run, on cost.

## Error sources (each a column in the tables)
- quad: the k-grid / quadrature error, as charged above. In the FULL records (primaries) it is computed on every orbit. In the SUB records (secondaries and the 128 extra), S2 and the cubic-field run are done on every tenth orbit, scaled to all orbits by the ratio of the S1 sums, times 2.
- digit: ball digitization, |site-centred minus cell-centred origin|, both at S1.
- interp: radial field interpolation, |cubic-spline minus linear fields|, S1.
- dN root: the change in count over a ball radius of R_t +- dR, with dR = max(|linear - cubic-spline root|, |dense - applied root|). The enclosure carries this through the ball and through the R_t in the formula together: rho_lo = min over R in {R_t - dR, R_t, R_t + dR} of rho(N(R) - E, R), where E = quad + digit + interp; rho_hi likewise with N(R) + E.
- Bandwidth edge: the edges omega = +-Delta are located by the piecewise-linear ray interpolation, so their error is inside quad. The float error of omega is about 1e-15, and no separate term is charged.
- Not enclosed (model level, see FABLE NEEDED): the frozen-symbol (Weyl) approximation itself, that is, symbol counts in place of eigenvalues of the cubic operator on the ball. The node-root ball is a definitional alternative, reported in its own column ("rho at node root", primaries only) and kept out of the enclosure.


## Tables (verbatim output of sd_tables_v3.py, box tables_v3.log; counts by sd_count_v1.py / sd_count_v2.py as each header says)

Tables emitted by sd_tables_v3.py from <internal>/out_count_v1. G/a^2 = 1/4, a = 1.

### S4_32_s4 j70 (primary), T/r_s 17.5000, R_t 21.05520 (cubic 21.06655, applied 21.05521, dR 1.13e-02), node root 25.7902, sites 39127, mode FULL, counts by sd_count_v1.py
| species | a Delta | N(abs omega < Delta) | g_Delta(0) | rho_Delta | enclosure | quad | digit | interp | dN root | rho at node root |
|---|---|---|---|---|---|---|---|---|---|---|
| light | 0.01 | 13.2366 | 661.831 | 0.076868 | [0.076539, 0.077134] | 0.0011 | 0.015 | 0.075 | 0.042 | 0.068964 |
| light | 0.02 | 46.6119 | 1165.3 | 0.102 | [0.10168, 0.10223] | 0.0022 | 0.052 | 0.16 | 0.14 | 0.092936 |
| light | 0.04 | 169.327 | 2116.59 | 0.13746 | [0.13713, 0.13768] | 0.0046 | 0.18 | 0.35 | 0.44 | 0.12558 |
| heavy | 0.01 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 |
| heavy | 0.02 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 |
| heavy | 0.04 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 |
| combined | 0.01 | 13.2366 | 661.831 | 0.076868 | [0.076539, 0.077134] | 0.0011 | 0.015 | 0.075 | 0.042 | 0.068964 |
| combined | 0.02 | 46.6119 | 1165.3 | 0.102 | [0.10168, 0.10223] | 0.0022 | 0.052 | 0.16 | 0.14 | 0.092936 |
| combined | 0.04 | 169.327 | 2116.59 | 0.13746 | [0.13713, 0.13768] | 0.0046 | 0.18 | 0.35 | 0.44 | 0.12558 |
counts per site in ball (light, Delta 0.01/0.02/0.04): 0.0003383 / 0.001191 / 0.004328; N/(4 pi R_t^2): 0.002376 / 0.008367 / 0.03039

### W5_64_s4 j97 (primary), T/r_s 12.1250, R_t 34.50914 (cubic 34.53910, applied 34.50910, dR 3.00e-02), node root 38.4513, sites 172317, mode FULL, counts by sd_count_v1.py
| species | a Delta | N(abs omega < Delta) | g_Delta(0) | rho_Delta | enclosure | quad | digit | interp | dN root | rho at node root |
|---|---|---|---|---|---|---|---|---|---|---|
| light | 0.01 | 41.8433 | 2092.17 | 0.065134 | [0.064933, 0.065247] | 0.0039 | 0.03 | 0.11 | 0.22 | 0.06499 |
| light | 0.02 | 142.631 | 3565.79 | 0.085033 | [0.08483, 0.085134] | 0.0079 | 0.096 | 0.23 | 0.71 | 0.085223 |
| light | 0.04 | 501.419 | 6267.73 | 0.11274 | [0.11252, 0.11283] | 0.016 | 0.32 | 0.49 | 2.3 | 0.11267 |
| heavy | 0.01 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 |
| heavy | 0.02 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 |
| heavy | 0.04 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | 0 |
| combined | 0.01 | 41.8433 | 2092.17 | 0.065134 | [0.064933, 0.065247] | 0.0039 | 0.03 | 0.11 | 0.22 | 0.06499 |
| combined | 0.02 | 142.631 | 3565.79 | 0.085033 | [0.08483, 0.085134] | 0.0079 | 0.096 | 0.23 | 0.71 | 0.085223 |
| combined | 0.04 | 501.419 | 6267.73 | 0.11274 | [0.11252, 0.11283] | 0.016 | 0.32 | 0.49 | 2.3 | 0.11267 |
counts per site in ball (light, Delta 0.01/0.02/0.04): 0.0002428 / 0.0008277 / 0.00291; N/(4 pi R_t^2): 0.002796 / 0.009531 / 0.03351

### S4_32_s4 j75 (secondary), T/r_s 18.7500, R_t 21.27236 (cubic 21.32801, applied 21.27239, dR 5.57e-02), node root 26.3299, sites 40339, mode SUB, counts by sd_count_v2.py
| species | a Delta | N(abs omega < Delta) | g_Delta(0) | rho_Delta | enclosure | quad | digit | interp | dN root | rho at node root |
|---|---|---|---|---|---|---|---|---|---|---|
| light | 0.01 | 14.4522 | 722.608 | 0.079093 | [0.078368, 0.079896] | 0.0022 | 0.12 | 0.14 | 0.11 | - |
| light | 0.02 | 53.0796 | 1326.99 | 0.10718 | [0.10643, 0.10804] | 0.0047 | 0.38 | 0.32 | 0.37 | - |
| light | 0.04 | 200.008 | 2500.1 | 0.14712 | [0.14625, 0.1481] | 0.01 | 1.3 | 0.74 | 1.2 | - |
| heavy | 0.01 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | - |
| heavy | 0.02 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | - |
| heavy | 0.04 | 0.0135982 | 0.169978 | 0.0012131 | [0, 0.0017424] | 0.0033 | 0.00014 | 0.011 | 0 | - |
| combined | 0.01 | 14.4522 | 722.608 | 0.079093 | [0.078368, 0.079896] | 0.0022 | 0.12 | 0.14 | 0.11 | - |
| combined | 0.02 | 53.0796 | 1326.99 | 0.10718 | [0.10643, 0.10804] | 0.0047 | 0.38 | 0.32 | 0.37 | - |
| combined | 0.04 | 200.021 | 2500.27 | 0.14712 | [0.14625, 0.14811] | 0.013 | 1.3 | 0.75 | 1.2 | - |
counts per site in ball (light, Delta 0.01/0.02/0.04): 0.0003583 / 0.001316 / 0.004958; N/(4 pi R_t^2): 0.002542 / 0.009334 / 0.03517

### W5_64_s4 j107 (secondary), T/r_s 13.3750, R_t 48.30651 (cubic 48.30350, applied 48.30651, dR 3.01e-03), node root 49.1693, sites 472273, mode SUB, counts by sd_count_v2.py
| species | a Delta | N(abs omega < Delta) | g_Delta(0) | rho_Delta | enclosure | quad | digit | interp | dN root | rho at node root |
|---|---|---|---|---|---|---|---|---|---|---|
| light | 0.01 | 2035.19 | 101760 | 0.27428 | [0.2725, 0.27604] | 24 | 0.082 | 2.2 | 0 | - |
| light | 0.02 | 4056.74 | 101419 | 0.27382 | [0.27283, 0.2748] | 24 | 0.16 | 4.4 | 0 | - |
| light | 0.04 | 8130.68 | 101633 | 0.27411 | [0.27361, 0.27461] | 19 | 0.64 | 8.7 | 0 | - |
| heavy | 0.01 | 1222.4 | 61119.8 | 0.21257 | [0.20965, 0.21544] | 31 | 0.0024 | 1.9 | 0 | - |
| heavy | 0.02 | 2462.56 | 61564 | 0.21334 | [0.21171, 0.21495] | 33 | 0.064 | 3.8 | 0 | - |
| heavy | 0.04 | 5073.38 | 63417.2 | 0.21652 | [0.21562, 0.21742] | 33 | 0.0047 | 7.9 | 0 | - |
| combined | 0.01 | 3257.59 | 162880 | 0.347 | [0.34382, 0.35016] | 55 | 0.085 | 4 | 0 | - |
| combined | 0.02 | 6519.3 | 162983 | 0.34711 | [0.34534, 0.34888] | 57 | 0.23 | 8.3 | 0 | - |
| combined | 0.04 | 13204.1 | 165051 | 0.34931 | [0.34836, 0.35026] | 52 | 0.65 | 17 | 0 | - |
counts per site in ball (light, Delta 0.01/0.02/0.04): 0.004309 / 0.00859 / 0.01722; N/(4 pi R_t^2): 0.0694 / 0.1383 / 0.2773

### S4_128_s8 j179 (primary-extra), T/r_s 11.1875, R_t 63.83072 (cubic 63.83717, applied 63.83072, dR 6.45e-03), node root 72.9116, sites 1089895, mode SUB, counts by sd_count_v2.py
| species | a Delta | N(abs omega < Delta) | g_Delta(0) | rho_Delta | enclosure | quad | digit | interp | dN root | rho at node root |
|---|---|---|---|---|---|---|---|---|---|---|
| light | 0.01 | 264.826 | 13241.3 | 0.065137 | [0.065029, 0.065236] | 0.051 | 0.39 | 0.37 | 0.15 | - |
| light | 0.02 | 874.844 | 21871.1 | 0.083714 | [0.083602, 0.083816] | 0.1 | 1.2 | 0.78 | 0.5 | - |
| light | 0.04 | 2976.58 | 37207.3 | 0.10919 | [0.10907, 0.1093] | 0.2 | 4 | 1.7 | 1.6 | - |
| heavy | 0.01 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | - |
| heavy | 0.02 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | - |
| heavy | 0.04 | 0 | 0 | 0 | [0, 0] | 0 | 0 | 0 | 0 | - |
| combined | 0.01 | 264.826 | 13241.3 | 0.065137 | [0.065029, 0.065236] | 0.051 | 0.39 | 0.37 | 0.15 | - |
| combined | 0.02 | 874.844 | 21871.1 | 0.083714 | [0.083602, 0.083816] | 0.1 | 1.2 | 0.78 | 0.5 | - |
| combined | 0.04 | 2976.58 | 37207.3 | 0.10919 | [0.10907, 0.1093] | 0.2 | 4 | 1.7 | 1.6 | - |
counts per site in ball (light, Delta 0.01/0.02/0.04): 0.000243 / 0.0008027 / 0.002731; N/(4 pi R_t^2): 0.005172 / 0.01709 / 0.05814

(S4_128_s16 j188: not computed)

### primary pair S4_32_s4 j70 / W5_64_s4 j97: 64/32 ratio, bandwidth spread, verdict
R_32/R_64 = 0.610134; (R_32/R_64)^{3/2} = 0.476582 (the spec's scaling identity rho2/rho1 = sqrt(g2/g1) (R1/R2)^{3/2})
| species | a Delta | rho(64)/rho(32) | enclosure | band [0.8,1.25] | sqrt(g64/g32) |
|---|---|---|---|---|---|
| light | 0.01 | 0.84735 | [0.84182, 0.85247] | in | 1.778 |
| light | 0.02 | 0.83368 | [0.82978, 0.83728] | in | 1.7493 |
| light | 0.04 | 0.82012 | [0.81725, 0.82276] | in | 1.7208 |
| heavy | 0.01 | nan | [nan, inf] | straddle | nan |
| heavy | 0.02 | nan | [nan, inf] | straddle | nan |
| heavy | 0.04 | nan | [nan, inf] | straddle | nan |
| combined | 0.01 | 0.84735 | [0.84182, 0.85247] | in | 1.778 |
| combined | 0.02 | 0.83368 | [0.82978, 0.83728] | in | 1.7493 |
| combined | 0.04 | 0.82012 | [0.81725, 0.82276] | in | 1.7208 |
| species | spread 32 (central, enclosure) | spread 64 (central, enclosure) | screen outcome (preregistered rule) |
|---|---|---|---|
| light | 0.5941 [0.5869, 0.6013] | 0.5598 [0.5553, 0.5646] | REJECT |
| heavy | nan [nan, inf] | nan [nan, inf] | REJECT (not decisive) |
| combined | 0.5941 [0.5869, 0.6013] | 0.5598 [0.5553, 0.5646] | REJECT (not decisive) |

### secondary pair S4_32_s4 j75 / W5_64_s4 j107: 64/32 ratio, bandwidth spread, verdict
R_32/R_64 = 0.440362; (R_32/R_64)^{3/2} = 0.292223 (the spec's scaling identity rho2/rho1 = sqrt(g2/g1) (R1/R2)^{3/2})
| species | a Delta | rho(64)/rho(32) | enclosure | band [0.8,1.25] | sqrt(g64/g32) |
|---|---|---|---|---|---|
| light | 0.01 | 3.4678 | [3.4108, 3.5224] | out | 11.867 |
| light | 0.02 | 2.5547 | [2.5252, 2.5821] | out | 8.7423 |
| light | 0.04 | 1.8632 | [1.8474, 1.8776] | out | 6.3759 |
| heavy | 0.01 | nan | [nan, inf] | straddle | nan |
| heavy | 0.02 | nan | [nan, inf] | straddle | nan |
| heavy | 0.04 | 178.49 | [123.75, inf] | out | 610.81 |
| combined | 0.01 | 4.3873 | [4.3034, 4.4682] | out | 15.013 |
| combined | 0.02 | 3.2386 | [3.1964, 3.2782] | out | 11.082 |
| combined | 0.04 | 2.3743 | [2.3519, 2.3949] | out | 8.1249 |
| species | spread 32 (central, enclosure) | spread 64 (central, enclosure) | screen outcome (preregistered rule) |
|---|---|---|---|
| light | 0.6347 [0.6142, 0.6553] | 0.001681 [0, 0.01296] | REJECT |
| heavy | nan [nan, inf] | 0.01855 [0.003133, 0.0367] | REJECT (not decisive) |
| combined | 0.6347 [0.6142, 0.6554] | 0.006641 [0, 0.01864] | REJECT (not decisive) |

FINAL (light field decides; primary pair, secondary as declared sensitivity): primary REJECT, secondary REJECT -> REJECT

## Self red team
1. **Is the REJECT an artefact of the matching rule?** Partly, in what it tests. The verdict rests on the primary pair, and both records sit just past the symbol's root. There s_max = |v|/N is 1.0052 (size 32) and 1.0053 (size 64).
   - In that regime the light count comes mostly from the strongly tipped valley cones near the root, not from an open Fermi pocket. N grows roughly as Delta^1.8, so g_Delta(0) is not flat and the spread is 0.56 to 0.59 against a 0.10 band.
   - The spread misses by a factor of about 6. Every quantified error is below 1e-2 of rho, so no listed error source can reverse it.
   - The secondary size-64 record (j107, s up to 1.52) has a real pocket. There g is flat in Delta (spread 0.0017 [0, 0.013]) and rho is about 0.274.
   - Size 32 never reaches that regime: its dense root vanishes at j76. So the 64/32 ratio is out of band on the secondary too (1.9 to 3.5).
   - Any rule that pairs records by elapsed time after the first root fails the ratio for this reason. A rule that pairs by pocket depth has no size-32 partner.
   - The REJECT is therefore of this numerical availability estimate on these stored records. The size-32 family is the limiting factor.
2. **Delta = 0.01 is below the band on the primaries** (light rho 0.077 at size 32, 0.065 at size 64). That alone also rejects.
3. **Counting convention.** "Multiplicities once" was read as the native zone integral with Kramers doublets counted as 2 states, and no 16 or 16k factor. A constant factor c rescales rho by sqrt(c) and leaves the ratio and the spread unchanged. The spread fails for any such factor. FABLE NEEDED if a different convention was meant.
4. **Quadrature validation was on synthetic symbols, not on record orbits.** S3 was run only on synthetic (N, s) points spanning the record range (sd_conv_v1). The per-orbit error is 2|S2 - S1|; for heavy, the cube-face estimate is added because |S2 - S1| underestimated |S3 - S2|. The total quad errors are below 1e-4 of N on the primaries, against a verdict margin of a factor of 6.
5. **Subsampled records** (secondaries, 128) scale every-tenth-orbit errors to the whole set. This is an estimate, not a bound, and is marked SUB in the tables. The verdict does not depend on these records.
6. **The heavy field is exactly zero on the primaries.** The exact Cauchy frozen-gap bound excludes every orbit: with m a = 0.2 the pocket threshold exceeds 1 and s_max is 1.005. Heavy counts are therefore no information on the primaries, and their ratios are undefined (nan in the tables). The combined values equal the light values there. They were never used for the verdict.
7. **Weyl / frozen symbol.** Counting frozen-symbol phase space is not counting the cubic operator's eigenvalues. Near the root the fields change on the scale ell, with sigma of about 4a, while the near-threshold cones reach out to |k| of about 0.3. The O(1/(ell k)) corrections are not enclosed, and this is stated under FABLE NEEDED. It cannot make g flat in Delta at the primaries without changing the regime.
8. **Process.** No local compute was used: every python ran on the box at nice 10, ulimit -v 16000000, with 8 threads. The first inspection launch used BLAS caps of 8 and showed nlwp 29; every later launch had the BLAS pools at 1 and nlwp 15. I killed two of my own processes, the stale chain v1 bash (pidfile) and the 128 s8 child of chain v2. The child's pid was read via ppid of my own pidfile'd chain, and I checked its exact command line in the same command before killing it. That 128 record was then rerun from scratch by chain v3. No pattern kills, no remote pgrep, no writes to the PK5 directory, no API stages, no subagents.

## FABLE NEEDED
- Whether "count native eigenvalue multiplicities once" means the convention used here (4 states per site, valleys inside the zone integral). This does not affect the verdict.
- Whether a frozen-symbol count is the intended "native symbol eigenvalue" count, or whether diagonalising the cubic operator on the ball was meant. The verdict margin makes this moot for the spread failure.
- Whether a matching rule by pocket depth (s_max) was intended. If so, size 32 has no partner record for the size-64 secondary.

## Box MD5SUMS (<internal>/MD5SUMS)
```
ce81efbac4e477aa9e2e3ebca95fcf3d  code/chain_count_v1.sh
63c37bc46f598a0881cc64f280b00ebe  code/chain_count_v2.sh
6a60438e60a92e1cf055a7ac6dea0f75  code/chain_count_v3.sh
947c6c8fe0e9b7687e7b1e12d19f7961  code/launch_sh_v2.sh
caed089c01103eeca78a551e63a94161  code/launch_v1.sh
ae8d8f053ef495e2a283319bb8f4ee21  code/launch_v2.sh
57f26e3f86682d255f6f7eb36596e1ba  code/MATCHING_RULE_v1.txt
af89db134309fd3055d05e320a39b8a6  code/sd_conv_v1.py
97e3fdf3f3056009b53acde47271a55c  code/sd_count_v1.py
12787a8e60e055b57a8d4c85c9361169  code/sd_count_v2.py
9394d9fd332cb1dfdb2af05a2743e83a  code/sd_inspect_v1.py
0eeb0788b6e2b85b6db5913632e65599  code/sd_lib_v1.py
4ecca0233a0383c6e489049b2175d4e9  code/sd_lib_v2.py
e749b4d8c17a8398a1bbcfe320cb83a4  code/sd_tables_v1.py
0cd7f0022301d6cddde0384d3d2569ae  code/sd_tables_v2.py
91f47293faee82bac159c1efe1fab22e  code/sd_tables_v3.py
7e8bb7d494432640e0711e0d4d94e825  out_count_v1/count_S4_128_s8_j179.json
2d576eb43f308a36a64c375ae9d2b7e1  out_count_v1/count_S4_32_s4_j70.json
fd5a15c9aa3bec7bc09ca0df784d2291  out_count_v1/count_S4_32_s4_j75.json
448ccd8fc57eb04c89d8c02b43c4f241  out_count_v1/count_W5_64_s4_j107.json
77d84429e505d1e30783b2bf58d5f890  out_count_v1/count_W5_64_s4_j97.json
86b31bc0551243d669c49df86f27cb9d  out_count_v1/reps_S4_128_s8_j179_cell.npz
4010b752671c03769daf5c8b17dcd0a9  out_count_v1/reps_S4_128_s8_j179_site.npz
6a017d320facab58de017458e2a3e3f1  out_count_v1/reps_S4_32_s4_j70_cell.npz
3628fe256d055c4826b1dcb3bbc5d4af  out_count_v1/reps_S4_32_s4_j70_site.npz
cb16dada0af0d28469b9833aff192e8d  out_count_v1/reps_S4_32_s4_j75_cell.npz
9e8703f35d3323d2cbe7a10fd9770dde  out_count_v1/reps_S4_32_s4_j75_site.npz
8482528396138cb6feb157ce6eeb1968  out_count_v1/reps_W5_64_s4_j107_cell.npz
269cd2159b2479de80106a3c88792092  out_count_v1/reps_W5_64_s4_j107_site.npz
ddfb49c0de44b5ef03807349bad30dfa  out_count_v1/reps_W5_64_s4_j97_cell.npz
e2f958e75fa9b6f06660c5ddcf8b4853  out_count_v1/reps_W5_64_s4_j97_site.npz
a1f4a11a7b2a40af51f8a9495292f8c6  out_count_v1/tables.json
dbb46e1fd44d2656f24d1ea509c6d53a  out_count_v1/tables.md
4387f7eb5c4b9c27773c6388b4a34770  out_inspect_v1/inspect.json
b2a38c95bdde18b73102068af986db05  out_conv_v1/conv.json
dbb46e1fd44d2656f24d1ea509c6d53a  tables_v3.log
```

## STATUS
STORE DOS route 2 (busi, #722) delivered: REJECT. On the stored back-averaged cubic symbols at the first dense-rooted records (size 32 j70, size 64 j97, rule declared before counting), the light-field pocket rho is printed per bandwidth in the tables, with tight enclosures.
- The 64/32 ratio is in band.
- The bandwidth spread fails at both sizes, about six times over the 0.10 band, and light rho at a Delta = 0.01 is below the 0.10 band edge.
- The failure is physical, not numerical: just after formation the count comes from tipped valley cones, so g_Delta(0) is not flat in Delta.
- The later size-64 record has a flat, pocket-like g. Size 32 never forms an open pocket, so the secondary ratio is out of band too.
- The heavy field is exactly zero on the primaries (frozen-gap exclusion), and the combined values equal the light ones there; neither substitutes for the light-only verdict.
- This is a screen of a conditional availability, not a capacity. No box job is left running.
