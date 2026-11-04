# CAPACITY UPPER BOUND, route I-new (frozen-symbol local Gibbs sum), route2, v1

**Leads and manifests.**
- Leads #364, #365, #370, #371.
- Manifests: v1 (642f1616), amendment 1 (2f73f6eb), amendment 2 (3919c35e).
- Blind to the other route.

**Status: UNDECIDED.** This holds per reviewer B (capst2a and capst4a) and reviewer A (capst3f). Hartree insufficiency is not a model no-go.

**What the numbers say.**

| Bound | Result |
|---|---|
| In sector U (unrefilled pockets) | The exact Gibbs ceiling at E_0 is 0.08-0.10 B on the stored slices and 0.28-0.31 B on the vacuum proxy. It is below 0.9 B at every size |
| Unrestricted (E_0 + E_refill) | Volume-sized on the proxy: 3.4, 6.8 and 13.6 B at L = 64, 128 and 256, with exponent 2.99 |

The U figure is not a complete upper bound, so it cannot give INSUFFICIENCY:
- the budget premise E_X <= E_0 is not established (reviewer B);
- the cubic-parent clauses (sigma_X, cross-cut energy, block mixing, leakage) are not computed on this route;
- the local-DOS (semiclassical) remainder is estimated, not enclosed.

## 1. Corrections made during this job

**The v1 route II tables (s77) are withdrawn.**
- They used root finding of delta(w n.q - eps) along k_1. They were not grid-converged: refinement changed them by 10-100 percent, and they were 8 percent off the 8K closed form on the axis.
- They are replaced by the linear tetrahedron method (s82).
- s82's first version, s81, had a Bloechl-formula error in the middle interval. It was found by the 8K check, fixed in s82, and s81 is withdrawn.

**s79's proxy pass is withdrawn.** It put the origin site into a cell with w of about 2.5e5. s82 redoes it with the origin excluded and w clipped at 4, a clipped volume of 2.4e-4.

**Kept from s79:** its slice pass. That pass had no origin issue.

## 2. Conventions (amendment 2)

**Symbol.** h_X = sum(e_x + w_x.p_x), w = v/N, so omega = w.q(k) pm eps(k) and N drops out. The paper's cubic operator: q_i = sin(2k_i)/2, eps = (m^2 + sum sin^2 k)^(1/2). Both fields are included: light (m = 0) and heavy (ma = 0.2).

**Sector U.**
- Free modes only: lower band at omega < 0, cost |omega|; upper band at omega > 0.
- D_free(0) = nu_0, with no sqrt 2.
- S+_U = (pi/sqrt 3) sqrt(nu_0 E_0), and the exact F_K^U(E_0) on the full free-mode cost DOS.

**Unrestricted.** All modes, F_K(E_0 + E_refill), with E_refill the local pocket-cost sum.

**Constants.** f = 4 eta g = 0.887 (eta(0.2) = 0.887, g = 1/4), so B = f pi L^2/g = 11.1 L^2. E_0 = L/(2g), with L = r_s/a = 2gE.

## 3. The transverse measure: the 8K check (capst3f, capst4a (2)) and convergence

Axial direction, m = 0. Site DOS: reviewer A's closed form (|w|/pi^3) int 8K(k) ds, with k^2 = t(2 - t), against the tetrahedron 4J.

| w | 8K closed form | tetra Nk = 48 | 64 | 96 | 128 | rel. err. at 128 |
|---|---|---|---|---|---|---|
| 1.2 | 0.150297 | 0.135603 | 0.141005 | 0.145772 | 0.147879 | 1.6e-2 |
| 1.5 | 0.359177 | 0.350766 | 0.353723 | 0.356543 | 0.357807 | 3.8e-3 |
| 2.0 | 0.771847 | 0.762634 | 0.764703 | 0.768944 | 0.770015 | 2.4e-3 |

- The convergence is O(h^2): the Richardson (4 J_128 - J_64)/3 at w = 1.2 is 0.15017, within 0.1 percent of the closed form.
- The 4 pi form gives 0.1486, 0.3377 and 0.6079: low by 1-21 percent, growing with w.
- **The tables used (Nk = 64) are biased low:** by 6 percent at w = 1.2, 1.5 percent at 1.5 and 0.9 percent at 2, compared with the closed form. On the proxy, weighted by its w distribution, this is a low bias of about 2-3 percent in nu_0 (about 1-1.5 percent in S+_U).
- Diagonal directions (111, 110) and ma = 0.2 converge similarly. At Nk 96 against 128 they agree to 0.0-0.8 percent (json grid_convergence).
- **Direction against the cubic axes (capst4a (f)):** 20 sector directions with solid-angle weights. The direction-averaged site DOS is well above the axial one; at m = 0 it is 0.240, 0.491 and 0.813 at w = 1.2, 1.5 and 2. Per-site nearest-direction resolution changes nu_0 by -1.2 percent on proxy64 and -1.9 percent on the slices, against the average.

## 4. Results

### 4.1 Vacuum proxy (N = 1, w^2 = R/r, X = {r < R}); the exponents are read here

| L | c_nu | light share | S+_U (Sommerfeld) | rho_U Somm | F_K^U(E_0) exact | rho_FU | E_refill | E_refill/E_0 | F_K(E_0 + E_refill) | rho unrestricted |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 1.423 | 0.605 | 12533 | 0.275 | 13955 | 0.306 | 7294 | 57.0 | 1.563e5 | 3.42 |
| 128 | 1.423 | 0.605 | 50134 | 0.275 | 53371 | 0.292 | 58375 | 228 | 1.241e6 | 6.80 |
| 256 | 1.423 | 0.605 | 200534 | 0.275 | 207990 | 0.285 | 466997 | 912 | 9.911e6 | 13.57 |

**Size exponents (proxy):**
- S+_U: 2.000.
- F_K^U: 1.949, inside [1.9, 2.1].
- Unrestricted F_K: 2.993.
- E_refill: 3.000.

**Against reviewer A's predictions:**
- c_nu: reviewer A 0.63 massless axial, about 0.4 at ma = 0.2. Here 1.42, both fields and direction-averaged. The direction average and the heavy field (40 percent of nu_0) account for the difference.
- rho_U: reviewer A 0.10-0.22. Here 0.27-0.31.
- E_refill/E_0: reviewer A about 15 and 60 at 64 and 128. Here 57 and 228, about four times larger, both fields.
- The unrestricted exponent goes to 3, as predicted.

### 4.2 Stored slices (first rooted record, held fields)

| | slice64 (W5, record 97, T/r_s 12.125) | slice128 (S4_128_s8, record 179, T/r_s 11.1875) |
|---|---|---|
| r_s, R_t | 58.30a, 38.56a | 116.64a, 72.94a |
| max w inside X | 1.0053 | 1.0056 |
| c_nu (light only; heavy pockets closed) | 0.00169 | 0.00163 |
| S+_U Sommerfeld, rho_U | 358, 0.0094 | 1411, 0.0093 |
| F_K^U(E_0) exact (full free DOS), rho_FU | 3970, **0.105** | 11807, **0.078** |
| F_K^U at E_max = M(R) | 2926 | 8413 |
| S_loc (energy multiplier only; OLD-BACKGROUND budgets), rho_loc | 1872, 0.049 | 6226, 0.041 |
| E_refill; pocket modes | 0; 0 | 0; 0 |
| F_K unrestricted = F_K^U (no refill) | 0.105 B | 0.078 B |
| M(R) = R_t w(R_t)^2/(2g), M(R)/E_0 | 74.75, 0.641 | 141.52, 0.607 |
| Size exponent of F_K^U (64 -> 128) | 1.57, outside [1.9, 2.1] | |

**Reading.**
- At the first rooted record the pockets have only just opened (w <= 1.006). The local zero-surface DOS is small, so the Sommerfeld estimate S+_U understates.
- The exact Gibbs infimum on the full free DOS (near-tilted cones at small cost) is what binds: 0.08-0.10 B.
- Slice minus proxy (the preparation dependence at this epoch): rho_FU -0.20 at 64 and -0.21 at 128.

### 4.3 Energy terms normalized to the slice charge M(R) (capst4a)

| | slice64 | slice128 |
|---|---|---|
| E_0/M(R) | 1.560 | 1.648 |
| E_refill/M(R) | 0 | 0 |
| sum of the cell budgets (OLD-BACKGROUND)/M(R) | 1.040 | 1.037 |
| sigma_X/M(R), cross-cut energy/M(R), interband Tr[(C - P_b)h_bu]/M(R) | NOT COMPUTED (see sec. 5) | NOT COMPUTED |

## 5. reviewer B's capst4a (a)-(g) and reviewer A's clauses on this route

| Item | Status on route I-new |
|---|---|
| (a) sigma_X = Tr(P_b h_X) - sum (r_N + w.r_v), eps_X | NOT COMPUTED: needs the finite-parent P_b. Route II-new (the digital ball) is the vehicle |
| (b) compressed Pi_X h Pi_X and the cross-cut energy | NOT COMPUTED: the local symbol has no cut bonds. The local sum is an interior quantity by construction |
| (c) exact Tr[(P_b - P_g)h_X] against the band-block pocket sum | In the frozen symbol h_bu = 0 pointwise, so the local E_refill is the band-block sum identically. The exact finite-parent value and the mixing difference are NOT COMPUTED |
| (d) trace-norm bound on Tr[(C - P_b)h_bu] | NOT COMPUTED: needs a cubic-parent covariance |
| (e) U leakage, including pocket-projector motion and intraband transport | NOT COMPUTED: needs a cubic-parent covariance |
| (f) full-BZ integrals, radial directions against the cubic axes, ma = 0 and 0.2 | DONE (secs. 3 and 4). c_nu and rho_U are estimates, not enclosures (Nk = 64 tables biased low by 1-6 percent, sec. 3) |
| (g) stored sources from a level-following projector | S_loc's cell budgets (eps_tot + w J_tot from the stored records) are flagged OLD-BACKGROUND |
| Work clause (E_rec - E_prep) | -1.533 (64), -1.596 (128): the charge falls slightly from preparation |
| N_B ln 2 | not readable from these records' continuation entries (0 found): NOT COMPUTED |
| E_+, comparator drift, U-leakage weights Tr(C Pi^u), Tr((1 - C)Pi^b) | NOT COMPUTED (no light-field state or cubic covariance on this route) |
| Semiclassical remainder (h_X spectrum against the local DOS) | NOT ENCLOSED. The amendment 1 small-ball estimate was not run: the s78 pass stopped on the 256 slice, which has no rooted record. Route II-new carries this |

## 6. Self red-team

| Risk | Check or status |
|---|---|
| A wrong DOS quadrature | Two earlier failures were found and corrected (secs. 1 and 3). The current tetrahedron tables converge as O(h^2) to the 8K closed form; the residual low bias of 1-6 percent is quantified |
| Proxy origin artefact | Origin excluded and w clipped at 4. The clip affects 2.4e-4 of the volume |
| The Sommerfeld/exact mismatch on the slices | Expected: the pockets barely open. The exact Gibbs value is reported as the bound |
| M(R)/E_0 = 0.61-0.64 against reviewer A's 0.95 | M(R) is read at the dense root R_t from w = v/N, where w(R_t) = 1 by definition. reviewer A's figure may refer to a different radius or charge. Printed as measured |
| Wrong matched family | 64 and 128 use sigma/r_s matched (s4/l4 at 64, s8/l8 at 128). The epochs are T/r_s 12.125 and 11.19, not identical. The 256 slice has no rooted record in its json, is out of scope under amendment 2, and is labelled "unestablished link" |

## 7. Record (box <internal>/capacity)

| File | md5 |
|---|---|
| tetra_s82_v1.json (tables, 8K check, c_nu, S+_U, proxy F_K, E_refill, exponents) | bd2639ce40d0d550c6138f7288c9b953 |
| tetra_tables_s82_v1.npz | 1ad1aa4118f958915db3ace3f406dce9 |
| amend2_s79_v1.json (slice F_K^U, S_loc, M(R); its proxy rows superseded) | 7da515053eb051ef41b0a58a7ab844e5 |
| routeII_s77_v1.json (withdrawn tables; kept for the record) | b9ae689656aeb4b718282fe37358fda9 |
| scripts s82_capacity_tetra_fix_v1.py, s79_capacity_amend2_v1.py | b68c92d5, 68185cbf |
