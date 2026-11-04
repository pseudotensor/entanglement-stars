# SOURCE_MARK14 route 2 (route2 lane): box record

Box 192.168.1.93, `<internal>/python -I`, `nice -n 10`, OMP/OPENBLAS/MKL threads 16 (reviewer B scripts force 1 internally). Work dir `<internal>/{astra,fable,indep}/`; copies of every log, JSON and script in `scratchpad/sm14_api/box/results/`. Scripts were extracted from `astra_result.md` and `fable_result.md` with awk, unedited. Route 1 was not consulted: no busi SOURCE_MARK14 delivery or busi box output was opened. This record covers runs and reproduction only; it makes no physics judgement.

Script hashes (sha256, first 12): sm14_parent.py 98cf3bd6c518; sm14_source_bounds.py 77a4e0d5a1dc; s1_sectors.py f813b3ea3c83; s2_leading.py 8923352189a8; s3_parent.py a900643025f5.

Independent script: `indep/indep_sm14.py` (own code, imports neither route's scripts). Part A gives sector ranks: dense eigh of h0, then direct (Pi, Z) projectors, then the Gamma split solved from the identity, C4 and C3 class traces with its own spin-lifted rotations. Part B gives reviewer B's endpoint arithmetic plus finite-lattice checks of its local bounds. Part C gives the reviewer A leading coefficients from eq. (8) by an independent route: mu(V0) by root-bracketing on a grid with no unimodality assumption, g(rho) = int_0^1 (1-c/rho) mu(rho/c) dc tabulated in rho = e/|q|, and an octant-reduced BZ midpoint rule. It also gives the global min of e/|q| by Nelder-Mead plus a 161^3 grid. A refinement variant, `indep_sm14_fine.py`, uses a 5x finer mu table, 3.6x more rho nodes and N = 192, 256. Run times: smoke 98.5 s, full 110.3 s, fine 184.3 s. All exited 0.

---

## Part 1 — reviewer B (`astra_result.md`)

### 1.1 Script runs

| Script | Mode | Exit | Time | PASS / FAIL lines | JSON |
|---|---|---|---|---|---|
| sm14_parent.py | `--smoke` (n = 1, 3) | 0 | 0.26 s | 49 / 0 | sm14_parent_smoke.json |
| sm14_parent.py | full (n = 1, 3, 7, 9; zeta = 2/10, 1/10) | 0 | 0.49 s | 99 / 0 | sm14_parent.json |
| sm14_source_bounds.py | single mode (digits 16, 32, 64) | 0 | 1.04 s | 48 / 0 | sm14_source_bounds.json |

No fixes were needed. The only stderr output is two scipy FutureWarnings (int64 diags cast to float64), which are harmless. In bounds.log, `grep FAIL` matches two lines, but they are negative-control lines that print "... FAILS" with a PASS prefix ("PASS negative control: doubled kernel normalization FAILS" and "PASS negative control: missing endpoint current weight FAILS"). Neither is a FAIL line.

PASS lines in sm14_parent (per n, zeta):
- Z_commutes_H0, Z_commutes_range_two_control, beta_anticommutator_kinetic, mass_gap_square_identity, H0_second_moment, range_two_bond_moment;
- rank_sigma_+-1, dimension_sigma_+-1, coverage, occupied_total;
- at n = 7, negative_control_342_for_Zplus_rejected;
- "known-answer n=1 mass-only control".

PASS lines in sm14_source_bounds:
- exact K normalization and radial moments (1, 1/5, 1/17), cube r^4 moment 19/240, negative control (doubled normalization);
- symbolic source-gradient residual zero (3-site control), whole-region zero-flow source = H0, negative control (missing endpoint current weight);
- outward intervals at 16/32/64 digits (identical to 12 decimals);
- 36 per-point outer-bound rows.

### 1.2 Verdict-block numbers

| reviewer B row | reviewer B value | Box script output | Independent | Status |
|---|---|---|---|---|
| Integer block coverage | F_+ + F_- = I, d_sigma = 2n^3, total 4n^3 | dims [686, 686] n=7, [1458, 1458] n=9 | (Pi,Z) dims sum per Z: 344+342 = 686 (n=7), 730+728 = 1458 (n=9) | REPRODUCED |
| Inherited ranks r_sigma = n^3 - sigma(-1)^M | n=7: (r_-, r_+) = (342, 344); n=9: (730, 728) | script counts beta = -1 basis vectors at the mass-only end of the gapped homotopy (n = 1, 3, 7, 9) | direct eigh of h0: n=7 Z=+1 344, Z=-1 342; n=9 728 / 730 (also n=3: 28/26, n=5: 124/126) | REPRODUCED |
| Rank residuals | 0 | coverage, occupied_total PASS | 0 | REPRODUCED |
| a Delta_initial = 2 zeta | 2/5, 1/5 | exact string (from the mass-square identity) | min abs eigenvalue of h0 = 0.2000 at zeta = 0.2, n = 3..9 (S1 and indep), so the gap is 2 zeta | REPRODUCED |
| Kernel moments (eq. 7 input) | 1, 1/5, 1/17; 19/240 | sympy exact, PASS | — | REPRODUCED (script) |
| Literal prefix remainder 11 B 2^-p | factor 11/2^p | JSON prints 11/2^64 and 11/2^512 only | not scripted (inherited R2 (8) constant) | NOT SCRIPTED |
| B_zeta (eq. 5) | t/3 + 3 sqrt2/20 | 0.625787822556 (zeta 1/5), 0.621738891938 (1/10) | same, to 1e-12 | REPRODUCED |
| I_band, I_tr, I_1 | B, T, Q intervals | zeta 1/5: [-0.519722, 0.150931], [-0.150931, 0.519722], [-0.625788, 0.625788]; zeta 1/10: [-0.515673, 0.157958], mirror, +-0.621739 | identical | REPRODUCED |
| Baseline (eq. 6) a chi^3 Tr(h_R P0)/16pi | [-t/6, -(e+zeta)/12] | [-0.206828, -0.161963] (1/5); [-0.204803, -0.152911] (1/10) | identical. reviewer A's Itilde gives -Itilde/6 = -0.20196 / -0.19972, inside both. Interior E_xx on the n=9 lattice lies in [1.20889, 1.22771], inside the claimed [0.97178, 1.24097] (boundary sites go down to 0.8518, but they carry no weight) | REPRODUCED |
| Local nuclear norms ‖e_x‖_1 <= 4t, ‖p_xi‖_1 <= sqrt2 | bounds | not scripted | n=9 sampled interior sites: 4.963869458396346 vs 4t = 4.963869458396343; 1.4142135623730954 vs sqrt2. Saturated to 3e-15 (tight, holds) | REPRODUCED (independent only) |
| sum abs(w_xi) <= 12 pi/5 chi^-3 (int x^2 u <= 2/5) | 2/5 | not scripted | int_0^1 x^2 sqrt(f1/x) dx = 0.396523 <= 0.4 | REPRODUCED (independent only) |
| sum_x w_x = (4 pi/3) chi^-3 (eq. 3) | identity | not scripted | reviewer A S3 toy (n=7, chi0 = 0.8): sum w = 8.1699 vs 8.1700 | REPRODUCED (toy, via S3) |
| q_band, q_tr, q_1 per point: chi^2 q in K_j x {B, T, Q}, K_j = 32 pi / j | 36 rows | e.g. CF-lo 1/5: [-15.999551, 4.646374], [-4.646374, 15.999551], +-19.264776; CF-hi 1/5: [-12.802150, 3.717828], +-15.414842; control 1/5: [-13.062034, 3.793300], +-15.727763; CF-lo 1/10: [-15.874906, 4.862717], +-19.140131; CF-hi 1/10: [-12.702414, 3.890936], +-15.315106; control 1/10: [-12.960273, 3.969922], +-15.626003. c = 1 and c = 4 identical | all 36 rows within 1.0e-12 | REPRODUCED |
| Sign | UNDECIDED; names A_sigma = F_sigma h_R F_sigma, its source-dominant descendants and their transport contractions, plus L_bd, L_np, L_tr | every q_1 interval is symmetric about 0; every q_band and q_tr interval contains 0 | confirmed: no complete interval excludes zero, and a specific source matrix is named | REPRODUCED (status) |
| Finer sector tables, character remainder, late a Delta_b, fine reduction ratios, boundary, non-power, leading-coefficient remainder, P2, other s | UNDECIDED (named) | — | — | NOT SCRIPTED (declared UNDECIDED) |
| Atom-to-Weyl [0,0], omitted-block [0,0], coarse reduction error [0,0] | by construction | — | — | NOT SCRIPTED (definitional) |
| Normalized leading interval Q_zeta = [-B, B] | +-0.625788 / +-0.621739 (in c2 = chi^2 q/g units: +-62.911 / +-62.504) | from JSON | same | REPRODUCED |

reviewer B's finite-n tables are identity checks at n <= 9. They are not a chi-extrapolation, as reviewer B itself states.

---

## Part 2 — reviewer A (`fable_result.md`)

### 2.1 Script runs

| Script | Mode | Exit | Time | PASS / FAIL lines | JSON |
|---|---|---|---|---|---|
| s1_sectors.py | smoke (n = 3, 5) | 0 | 2.83 s | 20 / 0 | s1_sectors_smoke.json |
| s1_sectors.py | full (n = 3, 5, 7, 9) | 0 | 41.21 s | 38 / 0 | s1_sectors_full.json (= s1_sectors.json) |
| s2_leading.py | smoke (N <= 64) | 0 | 1.52 s | 11 / **1** | s2_leading_smoke.json |
| s2_leading.py | full (N = 32..512) | 0 | 22.48 s | 12 / **1** | s2_leading.json |
| s3_parent.py (verbatim) | smoke | **1** | — | — | none: `ModuleNotFoundError: No module named 's1_sectors'` |
| s3_parent_fix.py | smoke (n = 5) | 0 | 13.92 s | 29 / **1** | s3_parent_smoke.json |
| s3_parent_fix.py | full (n = 7) | 0 | 172.23 s | 29 / **1** | s3_parent.json |

The `big` mode of S1 (n = 11) was not run, because the stated full sizes cover the verdict block.

**Mechanical fix (s3_parent_fix.py).** `python -I` drops the script directory from sys.path, so the `from s1_sectors import ...` line fails. The fix is one inserted line, with nothing else changed:
```
6a7
> import sys as _s, os as _o; _s.path.insert(0,_o.path.dirname(_o.path.abspath(__file__)))  # box fix: -I drops script dir from sys.path
```

**FAIL lines, verbatim:**
1. S2 full: `FAIL Phi tabulation vs direct nested quadrature (max rel dev 1.6e-05); Vmax=1.97911011 at x_m=0.23888081` (smoke: 2.3e-04). The script's own threshold is 1e-6. The tabulated Phi used for 𝒦 is therefore accurate only to about 1e-5 relative, which is larger than the quoted 𝒦 half-width (rel 5e-7). See 2.3.
2. S3 smoke and full: `FAIL NEGATIVE CONTROL spinor-less C4 rejected for V (|[D,V]|=5.55e-17)` (smoke 4.16e-17). V = ½{v̄_i, Q_i} ⊗ I4 is spin-scalar and v̄ is O_h-symmetrised, so a spinor-less rotation commutes with V exactly. This control cannot fail by construction: it is a defect of the control, not a mismatch. The h0 version of the same control in S1 does PASS (|[D,h0]| = 0.707).

Remark on S1: the line "all 192 elements commute with h0" is evaluated only for n <= 7. At n = 9 it prints 0.0e+00 without computing anything (`comm<1e-9 or n>7`). The trace identity and the ranks are computed at n = 9.

### 2.2 S3 toy outputs (n = 7, chi0 = 0.80037, zeta = 0.2, c = 1, g = 1/4, u0 = 1; reviewer A rows 9 and 25, a control, not a leading coefficient)

| Quantity | Box value |
|---|---|
| a E_R | 2.498857 |
| Tr[h_R P0] | -19.791154 (Tr[J-part P0] = -1.05e-17, exact 0) |
| q_band (toy) | **+0.032603** (smoke n = 5: -0.005603) |
| q_tr (toy) | -0.039280 (half-steps -0.039279, step error 1.0e-6) |
| q_1 (toy) | -0.006677 |
| Ky Fan unconstrained q_band dual | [-16.08529, +0.24512] |
| leakage ‖C_h - P0‖, coherence ‖(1-P0) C_h P0‖ | 0.05821, 0.05811 |
| 12 blocks: d_b, r_b | E12: (64,32), (62,30), (62,32), (64,32); E52: (56,28) x4; G32: (224,112) x4 — r_b = formulas |
| certified Fermi gaps a Delta_F (min over tau, grid minus Lipschitz) | E12 0.79128 / 0.71009 / 0.71009 / 0.79128; E52 1.32799 / 1.08707 / 1.08707 / 1.32799; G32 0.79128 / 1.04923 / 1.04923 / 0.79128; all at tau = 1; all positive |
| width bound R2 (27)-(29) | holds for every block (widths 11.3 to 131.8 against contributions of at most 0.0137) |
| rank P_lit minus number of negative eigenvalues of h(1) | 0 |

The toy values were produced by S3 only. They were not independently recomputed: rebuilding the averaged toy h_R is a full second implementation. The toy q_band is positive at n = 7 and negative at n = 5. reviewer A's row 12 makes its non-positivity claim at symbol level (chi -> 0), not at the toy parent. This is recorded without judgement.

### 2.3 Verdict-block numbers

| reviewer A row | reviewer A value | Box script output | Independent | Status |
|---|---|---|---|---|
| 1 Sector totals | n=7 344/342, n=9 728/730 | S1: Z=+1 344.000000, Z=-1 342.000000 (n=7); 728 / 730 (n=9) | identical (direct eigh) | REPRODUCED |
| 2 Isotype table (G192) | n=7 Z=+1: E½ 32, 32; E5/2 28, 28; G3/2 112, 112. Z=-1: 30, 32, 28, 28, 112, 112. n=9 Z=+1: 64, 64, 60, 60, 240, 240; Z=-1: 64, 66, ... | S1 and S3 identical (E½ pi=+1, Z=-1 = 30 at n=7; E½ pi=-1, Z=-1 = 66 at n=9) | own (Pi,Z) projectors plus the C4/C3 class-trace split give exactly these at n = 3, 5, 7, 9 | REPRODUCED |
| 3 All-parent character remainder 0; closed rank formulas | quasi-polynomials | S1: closed formulas match all 12 isotypes at n = 3, 5, 7, 9 (dev <= 3.1e-13); trace identity (2) for all 192 elements (dev <= 3.2e-14) | own split matches the formulas at n = 3..9 | REPRODUCED at n = 3..9; the all-odd-n statement is analytic (§2.1) and NOT SCRIPTED beyond n = 9 |
| 4 Integer block coverage, 12 blocks, dims | dim(E½) = (n^3+5n)/6 + pi sigma, etc.; sum 4n^3 | S1: dims sum 4n^3; S3: SVD block ranks = formulas, bases orthonormal and complete, h0 block-diagonal (1.5e-15) | own dims: n=7 E½ 64/62/62/64, E5/2 56, G3/2 224; n=9 130/128/128/130, 120, 480 | REPRODUCED |
| 5 Rank residuals 0 | 0 | S3: sum of r_b = 686, residual 0 | 0 | REPRODUCED |
| 6 h_R explicit, Hermitian, G192-invariant, Tr[J P0] = 0 | eq. (1) | S3 (toy): Hermitian, commutes with all 192 (5.6e-16), Tr[J-part P0] = -1.05e-17 | — | REPRODUCED (toy, via script) |
| 7 V_c(axis), V_c(diag), pocket x bound | 1.21980 / 1.10499; 1.12211 / 1.05940; 0.7942 / 0.8910 | S2: 1.2198039027 / 1.1049875621; 1.122115 / 1.059400 | V_c(axis) 1.2198039027 / 1.1049875621. Global min of e/abs(q) over the BZ = 1.1221146452 / 1.0594003070, equal to the diagonal formula, so the diagonal is the global minimiser (an assumption the pocket bound needs). x_max = 0.794192 / 0.891004 | REPRODUCED |
| 7 aDelta_sym >= 2 zeta sqrt(1-V^2) | closed form | S2 axis margins: at V = 0.5, 0.34715 vs bound 0.34641 (zeta .2); 0 at V_c (2e-9) | — | REPRODUCED (script; axis only) |
| 8 exact pencil gaps at physical size | UNDECIDED (pocket-encounter Fermi gaps) | — | — | NOT SCRIPTED (declared UNDECIDED) |
| 9 toy Fermi gaps | PENDING | values in 2.2 | — | PRODUCED (script only) |
| 10 reduction errors, ratios | UNDECIDED R_W | — | — | NOT SCRIPTED |
| 11 source-dual endpoints: floor -(32pi/3) Itilde, sorted -4𝒦 | PENDING | S2: Itilde = 1.2117396506 (zeta .2), 1.1983337559 (.1); c2_floor = -40.605785 / -40.156550; 𝒦 = 7.412317e-3 +- 3.9e-9 / 8.070189e-3 +- 2.3e-9; c2_sorted = -0.0296493 / -0.0322808 | Itilde 1.2117396506 / 1.1983337559 (to 1e-12); c2_floor -40.605785 / -40.156550. 𝒦: 7.4123541e-3 / 8.0702232e-3 (fine run, N = 256; the coarse run gave 7.4123477e-3 / 8.0702171e-3, and the tabulation refinement moved 6e-9). The reviewer A 𝒦 lies 3.7e-8 / 3.4e-8 below these, about 9-15x reviewer A's quoted half-width, with relative difference 5e-6 / 4e-6 | Itilde, floor REPRODUCED. 𝒦 value REPRODUCED to 5 significant figures (c2_sorted -0.029649 / -0.032281). 𝒦 half-width **NOT REPRODUCED**: it is not outward, since it sits below S2's own Phi tabulation error (the FAIL line) and below the box difference |
| 12 q_band leading interval [-(32pi/3) g Itilde, 0] chi^-2; sorted -4 g 𝒦 chi^-2 (HYP-A) | sign UNDECIDED, named pocket-encounter family / phi_av | — | chi^2 q floors: CF-lo -12.43440, CF-hi -9.94947, control -10.15145 (zeta .2); -12.29684, -9.83940, -10.03914 (.1). Sorted: -0.0090793, -0.0072649, -0.0074124 (.2); -0.0098851, -0.0079097, -0.0080702 (.1) | REPRODUCED (arithmetic from Itilde, 𝒦). The upper endpoint is 0, so the interval does not exclude zero and the sign is not claimed; a specific block (the pocket-encounter family) is named |
| 13 q_tr: X_-2 = X_-1 = 0 | PASS (theorem T3) | toy only: q_tr = -0.03928 at chi0 = 0.8 | — | NOT SCRIPTED at leading order (analytic T3); toy value only |
| 14 q_1 = row 12 at leading orders | sign UNDECIDED | — | — | follows rows 12 and 13 |
| 15 c2 in [-(32pi/3) Itilde, 0]; generic -4𝒦 | VALUE | as row 11 | as row 11 | REPRODUCED (numbers); remainder radii NOT SCRIPTED |
| 16 boundary mu(zeta) = ln(1+zeta/6) | 0.03279 / 0.01653 | — | 0.0327898 / 0.0165293 | REPRODUCED (arithmetic); the Combes–Thomas bound itself is NOT SCRIPTED |
| 17, 18, 19, 23 non-power, atom-to-Weyl, omitted-block, complete remainder | UNDECIDED (named) | — | — | NOT SCRIPTED |
| 20 c-independence | structure | — | pocket x_max 0.794 / 0.891 < 1, consistent with the stated margin | NOT SCRIPTED (c-term coefficient) |
| 21 E-level values: chi_c = 1.0947e-38, q1_lead = -16𝒦/(j^2 chi_c^2) phi_av | PENDING | S2 (zeta .2): CF-lo -9.28037e73, CF-hi -5.94176e73, control -6.18545e73; (.1) -1.01040e74, -6.46912e73, -6.73443e73; floors -1.27098e77, -8.13747e76, -8.47120e76 / -1.25692e77, -8.04744e76, -8.37748e76 | chi_c = 1.0946907e-38; q1_lead (zeta .2) -9.28040e73, -5.94179e73, -6.18547e73; (.1) -1.01041e74, -6.46914e73, -6.73445e73; floors identical | REPRODUCED (agreement 4e-6 relative, from 𝒦) |
| S2 controls | — | mean abs(q_z) = 0.3183179 vs 1/pi; Itilde(10) = 10.074674999 vs series 10.074675039; sign lemma min(e - abs(q)) = 0.057718 / 0.021717 | — | REPRODUCED (script) |
| 22 P2 | BLOCKED-input | — | — | NOT SCRIPTED |
| 24 other s | UNDECIDED | — | — | NOT SCRIPTED |

Field data used by both S2 and indep: V_max = 1.97911011 at x_m = 0.2388808. x_in(V=1) = 0.0905021. V^2 = f1/x is unimodal on (0, 0.3): exactly 1 sign change of the grid derivative.

### 2.4 Earlier reviewer A attempt (`fable_incomplete.md`; noted, not run)

Where its numbers or claims differ from the final reviewer A answer:
- **Group.** It uses an order-768 group with 20 isotypes, against G192 with 12.
- **Character remainder.** It allows a nonzero O(n^2) remainder bound, against "0 exactly".
- **h_R.** It uses the local reading u0(r) = sqrt(f1/x) inside the current weight, with h_R = ½{A, H(1)} + Delta_bd. The final answer uses constant u0 = 1.
- **q_band.** It puts the leading interval at Pi [I_Nomega - K_W, I_Nomega] with an energy-ordered upper endpoint.
- **q_tr.** It gives q_tr = 2g C_tr at order chi^-1, nonzero. The final answer has X_-2 = X_-1 = 0.
- **Omitted-block error.** It gives 0 (VALUE), against the final UNDECIDED.
- **Boundary.** It uses sinh mu = zeta/2, against mu = ln(1 + zeta/6).
- **chi_0^ctrl.** It quotes 1.0945602e-38, against the final 1.0947e-38 derived from E_R = 1.827e38/a.
- **P2.** It gave hand estimates (u_max ≈ 1.97, S(1) ≈ 0.60), whereas the final answer says BLOCKED-input. The u_max = 1.97 agrees with V_max = 1.97911 above.

---

## Part 3 — Row-by-row comparison (reviewer B vs reviewer A, with box results)

| Required output | reviewer B | reviewer A | Verdict | Box note |
|---|---|---|---|---|
| h_R on the finite parent | current weight W_ell^dag(1_B v_i), v = -u(r) r̂ with u = sqrt(f1/x) | -u0 W_R r̂_i with u0 = 1 (product of the averaged indicator and the site r̂) | **DIFFER** | different operators in the current term. Box numbers are not like-for-like for any quantity that depends on the current weight. The current part of Tr(h_R P0) is 0 for both (toy 1e-17) |
| Sector totals r_sigma | 342/344; 730/728 | same | AGREE | reproduced by three codes |
| Finer sector tables / intertwiners | UNDECIDED (F_alpha native not built) | G192 12-isotype table | **DIFFER** (status) | reviewer A's table reproduced at n = 3..9 |
| All-parent character remainder | UNDECIDED R_char | 0 exactly (closed quasi-polynomials) | **DIFFER** (status) | formulas hold at n = 3, 5, 7, 9 (two codes) |
| Integer block coverage | coarse F_+-, exact | 12 isotype blocks, exact | AGREE (both exact; reviewer A finer) | reproduced |
| Rank residuals | 0 | 0 | AGREE | reproduced |
| a Delta_b | initial 2 zeta; late UNDECIDED | symbol-level closed form; exact UNDECIDED (pocket encounters); toy gaps | AGREE on the UNDECIDED status at physical size; values differ in scope | V_c values reproduced; toy gaps 0.71 to 1.33 |
| Reduction errors / valid ratios | coarse [0,0]; fine UNDECIDED | UNDECIDED R_W | AGREE (UNDECIDED) | — |
| Source-dual endpoints | formulas (10)-(12), A_+- spectra UNDECIDED; no numbers | Ky Fan floor c2 = -40.606 / -40.157; sorted -0.02965 / -0.03228 | **DIFFER** (reviewer A numeric, reviewer B not) | reviewer A floor lies inside reviewer B's band interval (c2 units [-52.248, 15.173] / [-51.841, 15.880]) |
| q_band | broad interval containing 0 (e.g. CF-lo 1/5 chi^2 q in [-16.00, 4.65]) | [floor, 0] at chi^-2 (e.g. CF-lo 1/5 [-12.434, 0]), sign UNDECIDED, conditional NEGATIVE on HYP-A | AGREE on sign UNDECIDED; intervals compatible (reviewer A inside reviewer B) | — |
| q_tr | broad interval containing 0 (CF-lo 1/5 [-4.65, 16.00]) | X_-2 = X_-1 = 0 (theorem) | **DIFFER** (reviewer A sharper; compatible, since 0 is inside reviewer B's) | toy q_tr = -0.0393 only |
| q_1 | +-B, sign UNDECIDED | = q_band at leading orders, UNDECIDED | AGREE (UNDECIDED) | — |
| Named overlap block / source matrix | A_sigma = F_sigma h_R F_sigma and descendants | pocket-encounter family (sorting fraction phi_av, operand A) | **DIFFER** (different named blocker) | both name a specific object, not a count |
| Boundary error | UNDECIDED L_bd | VALUE bound, mu = ln(1+zeta/6) | **DIFFER** | mu arithmetic reproduced |
| Non-power error | UNDECIDED | UNDECIDED (Gevrey tail) | AGREE | — |
| Atom-to-Weyl error | [0,0] (no Weyl substitution) | UNDECIDED R_W (Weyl used) | **DIFFER** (method-dependent) | — |
| Omitted-block error | [0,0] (coarse cover omits nothing) | UNDECIDED (pocket family) | **DIFFER** | — |
| Leading-coefficient interval | c2 in +-62.911 / +-62.504 (limiting envelope) | c2 in [-40.606, 0] / [-40.157, 0], non-positive at symbol level | **DIFFER** (reviewer A sharper; compatible) | both reproduced |
| Leading remainder | UNDECIDED R_lead | structure only (R_W, R_tr, R_ell, R_np, R_bd), coefficients open | AGREE (not closed) | — |
| c-control | bounds uniform on c in [1,4] | leading term c-independent, O(c^2 chi^1.5) correction | AGREE in effect | — |
| P2 global enclosures | UNDECIDED (maximand not supplied) | BLOCKED-input | AGREE | — |
| Other s | UNDECIDED | UNDECIDED | AGREE | — |
| Complete source remainder | UNDECIDED L_src | partly UNDECIDED | AGREE | — |
| Normalization check vs (B) Q_w = 32 pi g chi^-2 I | I_x = chi^2 q/(32 pi g) | Q_w = -(16pi/3) g Itilde chi^-2, "iff I = Itilde/6" | not compared | -(16pi/3) Itilde = 32 pi I needs I = -Itilde/6; the sign convention of I is not reconciled in either text |

**Sign status (both).** Neither route claims a decided sign. Every complete interval quoted contains zero: reviewer B's q_1 = +-K_j B_zeta, and reviewer A's leading q_band/q_1 interval has upper endpoint 0. Each route names a specific block or source matrix that keeps the overlap. The box finds no number that contradicts either sign statement.

---

## Part 4 — Extension of the box checks to odd n = 11 to 21 (lead #137)

Script: `indep/indep_sm14_bign.py` (sha256 first 12 195bff8bed0d, md5 69f077e9cb00a69da487c64c4b4be5ee). This is the box implementer's own code. It imports neither route's scripts and does not reuse Part A of `indep_sm14.py`.
- **Run settings.** GPU 2 (`CUDA_VISIBLE_DEVICES=2`), torch 2.8.0+cu128, complex128/float64, `nice -n 10`, 16 threads, one job at a time.
- **Modes and timings.**
  - smoke (n = 5, 7): 7.4 s.
  - full (n = 11, 13, 15, 17): 187.7 s. Per n: 5.6, 15.8, 43.7, 110.9 s.
  - ext (n = 19, 21): 968.6 s. Per n: 270.5, 686.3 s.
- **Outcome.** All three exited 0. Every PASS/FAIL line in the three logs is PASS: smoke 45, full 73, ext 45, with 0 FAIL. Copies are in `box/results/indep_bign/`. JSON md5s: full e854ac95d5e1fed940e76c6446a50a4a, ext 34b8338210dbb7d358a27e80258d5cbf, smoke 59f0957efa9d01e907fc4f08c521a9bc.

### 4.0 Method

**Group.** 2O is generated by closure from su2(π/2, z) and su2(2π/3, (1,1,1)), giving order 48. Its SO(3) image comes from the adjoint map, which the script checks is a homomorphism on all 48² products.
- Characters are built by the box's own construction rather than copied from reviewer A's:
  - E½: tr s;
  - E5/2: tr s times the sign of the axis permutation (that is, A2);
  - G3/2: tr s times (number of axes fixed by the permutation − 1) (that is, E).
- These characters are orthonormal over 2O, with deviation 8.8e-17.
- Elements of G192 are (s, p, z). The site action is x → (−1)^p R(s) x, the spin factor is β^p kron(I2, s), and the factor Z^z is applied separately.

**Spectra.** h0 is built as a sparse matrix. Z = (−1)^{Σx} ⊗ β is diagonal in the site-spin basis, and h0 has zero Z=+1/Z=−1 coupling at every n (checked, max 0.0). So h0 splits into two coordinate blocks, each of dimension 2n³ (n = 21: 18522), and these are diagonalised separately:
- ζ = 0.2: eigenvectors by `torch.linalg.eigh` on cuda.
- ζ = 0.1: eigenvalues only.

**Traces.** For each of the 96 pairs (s, p), Tr_σ[Π^p C(s) P0] is computed on each block from the occupied eigenvectors, by permutation-plus-4×4 action in column chunks. The full trace is Tr[Z^z Π^p C(s) P0] = Σ_σ σ^z Tr_σ[...]. This uses the facts that Π and C(s) preserve both blocks and that Z = σ on block σ.

**Isotype ranks.**
- Ranks are rank(Γ, π, σ) = (d_Γ/192) Σ_g χ_Γ(s) π^p σ^z Tr[D(g) P0].
- Dimensions follow the same formula with Tr D(g) counted directly on the fixed sites.

**Commutation.** All 192 elements are checked against h0 exactly in sparse form, ‖[D, h0]‖_max ≤ 5.2e-16 at every n. A spinor-less C4 negative control gives 0.707 at every n.

### 4.1 (a) Isotype table, closed rank formulas and sector totals

The exact ranks are listed below. The order within each Z row is E½(π=+), E½(π=−), E5/2(π=+), E5/2(π=−), G3/2(π=+), G3/2(π=−). Every entry equals reviewer A's closed formula: E½: (n³+5n)/12 + (πσ − π − κσ)/2, E5/2: (n³−n)/12, G3/2: (n³−n)/3. Every rank is a multiple of its isotype dimension.

| n | κ | Z = +1 ranks (sum) | Z = −1 ranks (sum) | r_σ = n³ − σκ | max abs(rank − formula) | min abs(E) (ζ = .2 / .1) |
|---|---|---|---|---|---|---|
| 11 | −1 | 116, 116, 110, 110, 440, 440 (1332) | 114, 116, 110, 110, 440, 440 (1330) | 1332 / 1330 | 8.5e-13 | 0.2000 / 0.1000 |
| 13 | +1 | 188, 188, 182, 182, 728, 728 (2196) | 188, 190, 182, 182, 728, 728 (2198) | 2196 / 2198 | 1.7e-12 | 0.2000 / 0.1000 |
| 15 | −1 | 288, 288, 280, 280, 1120, 1120 (3376) | 286, 288, 280, 280, 1120, 1120 (3374) | 3376 / 3374 | 2.5e-12 | 0.2000 / 0.1000 |
| 17 | +1 | 416, 416, 408, 408, 1632, 1632 (4912) | 416, 418, 408, 408, 1632, 1632 (4914) | 4912 / 4914 | 1.8e-12 | 0.2000 / 0.1000 |
| 19 | −1 | 580, 580, 570, 570, 2280, 2280 (6860) | 578, 580, 570, 570, 2280, 2280 (6858) | 6860 / 6858 | 3.2e-12 | 0.2000 / 0.1000 |
| 21 | +1 | 780, 780, 770, 770, 3080, 3080 (9260) | 780, 782, 770, 770, 3080, 3080 (9262) | 9260 / 9262 | 4.5e-12 | 0.2000 / 0.1000 |

Further results at every n:
- **Sector totals at both ζ.** The Z-block negative-eigenvalue counts equal n³ − κ and n³ + κ at both ζ = 0.2 and ζ = 0.1. reviewer A's ζ-independence statement therefore holds at these n.
- **Occupied count.** There are 2n³ occupied states and no zero modes.
- **Dimensions.** The isotype dimensions equal reviewer A's dim formulas (deviation ≤ 6.4e-12) and sum to 4n³.
- **Negative control.** Replacing κ by −κ in the formulas mismatches by 1.00 at every n.
- **Smoke run.** At n = 5 and 7 the smoke run reproduces the Part 2 values: n = 7 gives Z=+1 32, 32, 28, 28, 112, 112 and Z=−1 30, 32, 28, 28, 112, 112.

### 4.2 (b) All-parent character remainder

| n | max abs(exact rank − closed formula) over the 12 isotypes | integrality deviation | trace-level identity (2), max dev over 192 elements |
|---|---|---|---|
| 11 | 8.5e-13 | 8.5e-13 | 1.3e-13 |
| 13 | 1.7e-12 | 1.7e-12 | 4.5e-13 |
| 15 | 2.5e-12 | 2.5e-12 | 1.6e-13 |
| 17 | 1.8e-12 | 1.8e-12 | 6.7e-13 |
| 19 | 3.2e-12 | 3.2e-12 | 9.5e-13 |
| 21 | 4.5e-12 | 4.5e-12 | 1.1e-12 |

- **Identity (2).** Identity (2) is Tr[D P0] = tr(s)·{N_fix for (p,z)=(0,0); 1 for (1,1); −1 for (1,0); −κ for (0,1)}, with N_fix counted directly from the site permutation. It holds on all 192 elements at every n.
- **Trace sums.** reviewer A's three trace sums hold exactly at every n:
  - Σ_s abs(tr s)² N_fix = 8n³ + 40n;
  - Σ_s sgn(s) abs(tr s)² N_fix = 8n³ − 8n;
  - Σ_s χ_E(s) abs(tr s)² N_fix = 16n³ − 16n.
  For example, n = 21 gives 74928, 73920 and 147840.
- **Remainder.** The remainder is zero to rounding (≤ 4.5e-12) at n = 3 to 21. The box therefore extends the Part 2 range n ≤ 9 to n ≤ 21. The all-odd-n statement remains reviewer A's analytic argument (§2.1). The box checks it at these n and does not prove it.

### 4.3 (c) Ky Fan floor and sorted endpoint at finite n

reviewer A defines c2_floor = −(32π/3) Ĩ and c2_sorted = −4𝒦 as BZ integrals. Neither text defines a finite-n version, so the box reports two finite-n analogues, both chosen by the box:
- **(i) Open cube.** Ĩ_open(n) = Tr abs(h0)/(4n³) from the exact open-cube spectrum. Its eigenvalues come in ±ẽ pairs on the torus, so Tr abs(h)/(4N) is the mean of ẽ.
- **(ii) BZ grid.** The n³ midpoint grid u = −π + 2π(k + ½)/n gives Ĩ_BZ(n) and 𝒦_BZ(n). The 𝒦 integrand is the same as in `indep_sm14.py` part C, 4π ẽ g(ẽ/abs(q̃)). The script asserts that every grid ρ lies inside the tabulated range, which is safe because every grid value is at least V_c(diag) by 4.4.

The sorted endpoint is defined only as a BZ integral, so it has no open-cube analogue here. The limits are from Part 2: −40.605785 / −40.156550 and −0.0296494 / −0.0322809.

| n | c2_floor open cube (ζ = .2 / .1) | n·(Ĩ − Ĩ_open) (.2 / .1) | c2_floor BZ grid (.2 / .1) | c2_sorted BZ grid (.2 / .1) |
|---|---|---|---|---|
| 11 | −38.70187 / −38.22751 | 0.62498 / 0.63322 | −40.605694 / −40.156093 | −0.0296610 / −0.0322631 |
| 13 | −38.99841 / −38.52820 | 0.62356 / 0.63170 | −40.605761 / −40.156376 | −0.0296505 / −0.0322807 |
| 15 | −39.21502 / −38.74775 | 0.62254 / 0.63061 | −40.605778 / −40.156477 | −0.0296675 / −0.0322868 |
| 17 | −39.38018 / −38.91511 | 0.62176 / 0.62979 | −40.605783 / −40.156517 | −0.0296581 / −0.0322933 |
| 19 | −39.51027 / −39.04693 | 0.62114 / 0.62915 | −40.605785 / −40.156534 | −0.0296612 / −0.0322937 |
| 21 | −39.61540 / −39.15343 | 0.62065 / 0.62863 | −40.605785 / −40.156542 | −0.0296414 / −0.0322768 |
| limit (Part 2) | −40.605785 / −40.156550 | — | −40.605785 / −40.156550 | −0.0296494 / −0.0322809 |

Trend:
- **Open-cube floor.** The open-cube floor approaches the limit from above with a 1/n deficit, n·(Ĩ − Ĩ_open) ≈ 0.62 / 0.63, falling slowly. This is a surface term of the open cube.
  - The two-point 1/n extrapolation from (n−2, n) gives Ĩ = 1.212445 at n = 13, falling to 1.211988 at n = 21 (ζ = .2), and 1.199093 falling to 1.198593 (ζ = .1). These approach Ĩ = 1.2117397 / 1.1983338 from above.
  - At n = 21 the residual is 2.5e-4, roughly O(1/n²).
- **BZ-grid floor.** The BZ-grid floor converges fast. At n = 21 it agrees with the limit to 2.5e-7 (ζ = .2) and 7.5e-6 (ζ = .1).
- **BZ-grid sorted endpoint.** The sorted endpoint on the BZ grid is not monotone in n. It scatters around the limit with deviation at most 1.8e-5 (ζ = .2) and 1.3e-5 (ζ = .1) for n = 13 to 21 (n = 11: 1.2e-5 / 1.8e-5). This is consistent with a non-smooth integrand: the pocket boundary rho = V_max and the threshold support near V_c. It agrees in size with the coarse-grid behaviour seen in Part 2.

### 4.4 (d) Certified global minimum of e/abs(q) over the BZ

**Reduction.** ẽ² and abs(q̃)² depend on u only through s_i = sin² u_i, and they are invariant under u_i → −u_i and u_i → π − u_i. So the torus reduces exactly to the octant [0, π/2]³.

**Function certified.** The script certifies G_V(u) = ẽ² − V² abs(q̃)² = ζ² + Σ_i [sin² u_i − V² sin² u_i cos² u_i] > 0 for given V. Where abs(q̃) > 0 this is equivalent to ẽ/abs(q̃) > V. Where abs(q̃) = 0, G_V = ẽ² ≥ ζ² > 0, so no division occurs.

**Covering.**
- The covering is adaptive and starts from 64³ cubes of half-width r = π/256.
- On each cube the bound is G ≥ G(c) − abs(∇G(c))·√3 r − ½·H_b·3r² − 1e-12, using:
  - the exact gradient ∂_i G = sin 2u_i − V² sin(4u_i)/2;
  - the diagonal Hessian bound H_b = 2 + 2V² ≥ abs(∂²_i G) = abs(2 cos 2u_i − 2V² cos 4u_i).
- Cubes that fail are split into 8 and retested.
- Evaluation is in float64 with a fixed slack of 1e-12. This is not interval arithmetic.

| ζ | V_c(diag) | u* (s* = (V²−1)/(2V²)) | e/abs(q) at (u*, u*, u*) | covering at V_c(diag) − 1e-6 | covering at V_c(diag) − 1e-8 | set where e/abs(q) < V_c(diag) + 1e-3 |
|---|---|---|---|---|---|---|
| 0.2 | 1.122114645207 | 0.3265597441 | = V_c(diag) (dev 0) | CERTIFIED, depth 6, 266120 cubes | CERTIFIED, depth 9, 268563 cubes, final r 2.4e-5 | inside [0.289, 0.363]³ |
| 0.1 | 1.059400307007 | 0.2356218734 | = V_c(diag) (dev 2e-16) | CERTIFIED, depth 6, 270005 cubes | CERTIFIED, depth 10, 274660 cubes, final r 1.2e-5 | inside [0.1965, 0.2729]³ |

Results:
- **Bracket.** min over the BZ of e/abs(q) lies in [V_c(diag) − 1e-8, V_c(diag)]. The upper end is attained at (u*, u*, u*) and its sign images. Every point with ratio below V_c(diag) + 1e-3 lies in one small cube around u* in each octant image. That cube contains no axis or face point.
- **Negative controls.**
  - V_c(diag) + 1e-6 cannot be certified, and the open cubes contain a point with G < 0 (−6.2e-7 / −3.3e-7).
  - V_c(axis) − 1e-6 is not a global lower bound.
- **Closed-form cross-check by the box.** With s_i = sin² u_i ∈ [0, 1], G_V = ζ² + Σ_i [(1 − V²) s_i + V² s_i²] is separable and convex in s.
  - For V > 1 its minimum is ζ² − 3(V² − 1)²/(4V²), attained at s_i = s* for all i.
  - This is ≥ 0 if and only if V² − (2ζ/√3)V − 1 ≤ 0, that is, V ≤ ζ/√3 + √(1 + ζ²/3) = V_c(diag).
  - So inf e/abs(q) = V_c(diag) exactly, attained only at s_1 = s_2 = s_3 = s*, which is the diagonal.
  - The covering is consistent with this to 1e-8. The covering is the requested certified sweep, and the separable identity is an algebraic statement that the covering checks.
  - The Part 2 Nelder-Mead and 161³-grid value is superseded by this bracket.

### 4.5 Status changes against Part 3

- **Finer sector tables.** "reviewer A's table reproduced at n = 3..9" becomes reproduced at n = 3 to 21. The all-odd-n claim is still analytic beyond n = 21.
- **All-parent character remainder.** It is zero to ≤ 4.5e-12 at n = 3 to 21, with identity (2) holding on all 192 elements at each n. The quasi-polynomial statement for all odd n is analytic.
- **V_c.** The global minimum of e/abs(q) is certified to [V_c(diag) − 1e-8, V_c(diag)] at both ζ by the covering. The separable reduction makes it exact, attained on the diagonal only.
- **Finite-n endpoints.** The open-cube Ky Fan analogue converges to −40.606 / −40.157 with a 1/n surface deficit (n·ΔĨ ≈ 0.62 / 0.63). The BZ-grid analogues converge as in Part 2. The finite-n analogues are the box's own definitions and are not taken from either text.
- **Unchanged.** Nothing in Parts 1 to 3 is altered. Route 1 (busi) was not opened.

---

## Part 5 — reviewer A step 1 (`fable_step1.md`, task `TASK_step1.md`): box runs and checks of the replacement cells

**Setup.**
- Work dir: `<internal>/fable_step1`. Copies are in `box/results/fable_step1/`.
- Runs: `python -I`, `nice -n 10`, 16 threads, one job at a time as a serial chain.
- Everything ran on CPU, GPU 2 unused. The largest dense object is the 1372-dimensional n = 7 toy, small enough for CPU, and the scripts are scipy-based, so a GPU port would mean editing them.
- Route 1 (busi) was not opened.

**Script assembly.** reviewer A's document has four python blocks: block 1 is the complete `s2b_leading.py`, and blocks 2-4 are edit instructions for S3. I applied them to the box-fixed `s3_parent_fix.py` exactly as reviewer A describes them, producing `s3_parent_step1.py`. The diff against `s3_parent_fix.py` is recorded in `s3_parent_step1.diff`.
- **(i) Current weight.** The `wqv/wv/ok` lines are inserted after the `wR=…; w2=…` line, and `wv=sum(wv[perm_of(R)]@R …)/48` is appended to the symmetrisation line.
- **(ii) Current term.** In the `W4=…; Epart=…; Jpart=…; hR=Epart-u0*Jpart` line, only the `Jpart…hR` part is replaced by reviewer A's line. I read "replace the `Jpart=…; hR=Epart-u0*Jpart` line" that way; W4 and Epart are kept. The `'u0'` record is changed as instructed.
- **(iii) Negative controls.** The two old lines (pC, and the spinor-less-C4-on-V control) are replaced by reviewer A's four lines.

**Other edits.**
- The sys.path line from Part 2 is kept. `s1_sectors.py` is copied unchanged beside S3 for its import.
- `s1_sectors_step1.py` is s1 with the `or n>7` short-circuit dropped, as reviewer A instructs (two-line diff).
- No other edits. No mechanical fixes were needed: every script exited 0 on its first run.

Script hashes (sha256, first 12): s2b_leading.py 1fc18e2af3e1, s3_parent_step1.py 907509d9c561, s1_sectors_step1.py 9d0bbf0f7aef.

### 5.1 Runs

| Script | Mode | Exit | Time | PASS / FAIL | JSON (md5, first 12) |
|---|---|---|---|---|---|
| s2b_leading.py | smoke (octant N = 16, 32, 64) | 0 | 1.97 s | 8 / 0 | s2b_leading_smoke.json 208682657192 |
| s2b_leading.py | full (octant N = 48, 96, 192) | 0 | 4.76 s | 8 / 0 | s2b_leading.json 1f76181a7065 |
| s3_parent_step1.py | smoke (n = 5) | 0 | 16.23 s | 32 / 0 | s3_parent_smoke.json ab0703318a6c |
| s3_parent_step1.py | full (n = 7) | 0 | 169.00 s | 32 / 0 | s3_parent.json e18734282faa |
| s1_sectors_step1.py | full (n = 3, 5, 7, 9) | 0 | 207.59 s | 38 / 0 | s1_sectors.json 5706b617fcfe |

The three box findings from Part 2 are cleared by these runs:
- **Φ tabulation control.** The control is rebuilt against direct nested quadrature with exact μ and ν, so it can fail. It now passes: `PASS Phi, Phi-prime tabulations vs direct nested quadrature (max rel dev 1.6e-11)`. The smoke run gives 4.2e-9; the threshold is 1e-6.
- **s3 negative controls.** The new controls can fail, and both do so as designed:
  - `PASS NEGATIVE CONTROL spinor-less C4 rejected for h_R (|[D,h_R]|=4.95e-01)`. This control can fail because h_R carries α.
  - `PASS NEGATIVE CONTROL O_h-breaking V_bad rejected by spin-lifted C4 (|[D,V_bad]|=5.00e-03)`.
- **S1 commutator at n = 9.** It is now computed rather than short-circuited: `PASS n=9: all 192 elements commute with h0 (max |[D,h0]|=5.6e-16)`.

Other s2b PASS lines:
- the closed forms of the two angular integrals;
- mean abs(q_z) = 1/π, and Ĩ(10) against its series;
- the 𝒦′ lines: 𝒦′ = 3.075758e-03 ± 1.6e-12 in (0, r_max 𝒦] with r_max = 1.2654, and 3.755740e-03 ± 2.1e-12 with r_max = 1.5762;
- the Ĩ lines.

**S3 toy with the adopted current weight** (n = 7, χ0 = 0.80037, ζ = 0.2, c = 1, g = 1/4):

| Quantity | Value |
|---|---|
| Current-weight convergence | 3e-4 (smoke) |
| Tr[h_R P0] | −19.791154. Unchanged from Part 2, since the current part is 0 (6.9e-20 smoke). |
| Toy q_band | **+0.031567** (Part 2: +0.032603) |
| Toy q_tr | −0.038490 (half-step error 1.0e-6; Part 2: −0.039280) |
| Toy q_1 | −0.006922 (Part 2: −0.006677) |
| Smoke (n = 5) | q_band = −0.005603 |
| Ky Fan dual for q_band | [−16.07847, 0.23831] |
| Spectrum of h_R | ±20.386647, symmetric |
| Leakage and coherence norms | 0.05821 and 0.05811 (unchanged; they do not depend on h_R) |
| 12 Fermi gaps | Unchanged |
| R2 width bound | Holds per block (widths 11.3 to 131.7, contributions ≤ 0.0132) |

reviewer A's exact identity floor + ceiling = 2·baseline holds on the toy to all printed digits: −16.07847 + 0.23831 = −15.84016, and 2·(−19.791154)/2.498857 = −15.84016.

### 5.2 Independent checks (`indep/indep_step1.py`, own code, sha256 0592434ec47a; smoke 6.7 s, full 25.8 s, exit 0)

**Method.**
- **Integrand.** 𝒦 and 𝒦′ come from the x-integral form, with the angular integral done in closed form and no Φ/Φ′ tabulation: 𝒦 = 4π·BZmean[abs(q̃)·ψ_K(ρ)] and 𝒦′ = 4π·BZmean[abs(q̃)·ψ_K′(ρ)], where
  - ψ_K(ρ) = ∫ x²[ρ − ½ − ρ²/V + ρ²/(2V²)] 1{V > ρ} dx,
  - ψ_K′(ρ) = ∫ x² (V − ρ)²/(2V) 1{V > ρ} dx.
- **Discretisation.** ψ is tabulated in s = √(V_max − ρ) at two densities (1201 and 2401 nodes), and the BZ uses an octant midpoint rule with N = 64, 128, 256, 384.
- **Control.** The closed form is checked against a direct 2D quadrature at three (ẽ, abs(q̃)) points, with relative deviation 4.2e-10.

**Further checks.**
- **Ratio bound.** The pointwise ratio bound is scanned on ρ ∈ [V_c(diag), V_max), V ∈ (ρ, V_max].
- **γ_ζ.** γ_ζ = min_BZ(ẽ − abs(q̃)) is computed in s_i = sin² u_i space, with a 201³ grid plus an L-BFGS-B polish.
- **S identity.** It is checked on an n = 5 open cube with random real weights.
- **Block-rank identity.** It is checked for the closed formulas at n = 3 to 21.

| Quantity | reviewer A step 1 / s2b | Box independent | Status |
|---|---|---|---|
| 𝒦 (constant-u₀ variant), outward cell | [7.4121e-3, 7.4125e-3] / [8.0700e-3, 8.0704e-3]; s2b 7.412355827e-3 ± 1.35e-10 / 8.070224781e-3 ± 8.1e-11 | 7.41235577e-3 / 8.07022486e-3 (N = 384; tabulation density changes the value by < 3e-14; N steps 256 → 384 move it +4.3e-11 / +2.5e-11) | **REPRODUCED.** The printed interval covers every value on record: S2 7.412317e-3, box Part 2 coarse 7.4123477e-3 and fine 7.4123541e-3, s2b, and this run. The s2b ± agrees with this run to 5e-11 / 8e-11. The Part 2 "fine" value 7.4123541e-3 was 1.7e-9 low, because of my own linear μ interpolation; the x-integral route here supersedes it, and the Part 2 five-figure statement is unchanged. |
| 𝒦′ (adopted h_R) | 3.0757577838e-3 ± 1.6e-12 / 3.7557396834e-3 ± 2.1e-12 | 3.0757577836e-3 / 3.7557396832e-3 | **REPRODUCED** (agreement 2e-13) |
| A-priori bound 0 < 𝒦′ ≤ r_max 𝒦, r_max = V_max(V_max − V_c)/((2V_c − 1)V_max − V_c) | 1.2654 / 1.5762; r_max 𝒦 = 9.38e-3 / 1.272e-2 | r_max = 1.2654050827 / 1.5761697166. A grid scan of the pointwise ratio V(V − ρ)/((2ρ − 1)V − ρ) peaks at (ρ, V) = (V_c(diag), V_max), equal to r_max. The algebraic factorisation denominator = (V − ρ)(2Vρ − V − ρ) is checked by hand. r_max·𝒦_up = 9.3798e-3 / 1.27203e-2 | **REPRODUCED**. The actual ratio 𝒦′/𝒦 is 0.415 / 0.465, well inside the bound. |
| Ĩ | 1.2117396506(1) / 1.1983337559(1) | 1.2117396506285 / 1.1983337559234 | **REPRODUCED** |
| Baseline c₂ = −(16π/3)Ĩ | −20.302893 / −20.078275 | −20.3028927 / −20.0782748 | **REPRODUCED** |
| Ceiling (sorted) c₂ = +4𝒦′ | printed as a bound (0, 0.0376] / (0, 0.0509], "PENDING-NUMERICS" | 4𝒦′ = 0.012303031 / 0.015022959 (s2b and box). The bound 4 r_max 𝒦_up = 0.037519 / 0.050881 is correctly rounded outward to 0.0376 / 0.0509 | bound **REPRODUCED**; value now computed |
| Floor c₂ = −(32π/3)Ĩ − 4𝒦′ | [−40.644, −40.6057] / [−40.208, −40.1565] | value −40.618088 / −40.171573. Bound endpoint −40.643305 / −40.207431 | **REPRODUCED** (the bound is rounded outward, and the value lies inside) |
| floor + ceiling = 2·baseline (S identity) | exact | s2b: −40.618088 + 0.012303 = −40.605785 = 2·(−20.302893); toy (5.1) exact to 6 digits | **REPRODUCED** |
| S = βα₂K anticommutes with h0, Q_i, h_R (real weights), Z, Π; spec(h_R) symmetric | theorem | Every anticommutator is exactly 0.0 (n = 5 cube), and the spectrum is symmetric to 1.1e-14. Negative control: adding an S-even term 0.05·w⊗I₄ breaks it (0.0996). | **REPRODUCED** |
| tr₄(P0)_{yx} = 2δ_{yx} on the open cube | theorem | max deviation 8.9e-16 (n = 5) | **REPRODUCED** |
| r(Γ,π,σ) + r(Γ,−π,−σ) = dim(Γ,π,σ) | holds for the formulas | 0 violations at n = 3 to 21. The Part 4 exact ranks equal the formulas, so it also holds for the exact ranks. | **REPRODUCED** |
| Pocket bound x < 1/V_c(diag)² | 0.7942 / 0.8910 | 0.794192 / 0.891004 (Part 4 certified V_c(diag)) | **REPRODUCED** |
| Boundary μ = ln(1 + ζ/6) | 3.2790e-2 / 1.6529e-2 | 3.27898e-2 / 1.65293e-2 | **REPRODUCED** |
| γ_ζ = min_BZ(ẽ − abs(q̃)) | 5.7718e-2 / 2.1717e-2 (S2's 64³ grid minimum, quoted as the minimum) | global minimum **5.76965e-2 / 2.07354e-2**, on the diagonal at s_i = 0.061063 / 0.031893. The 201³ s-grid gives 5.77023e-2 / 2.07620e-2. | **NOT REPRODUCED**. reviewer A's values are grid values that lie above the true minimum: +2.2e-5 (0.04 %) and +9.8e-4 (4.7 %). |
| μ_ext = ½ ln(1 + γ_ζ/9) | 3.196e-3 / 1.205e-3 | 3.19513e-3 / **1.15064e-3** | ζ = 1/5 **REPRODUCED** at the printed precision (3.195 vs 3.196, last digit). ζ = 1/10 **NOT REPRODUCED** (1.151e-3, not 1.205e-3). The rate is still positive and doubly-exponential, so "< exp(−10^150)" is unaffected. |
| Combes–Thomas lemma for P_lit, Lieb–Robinson for C_h, C_bd | stated | — | NOT SCRIPTED |
| Normalisation I := 𝓘_P₀ = −Ĩ/6 | −0.201957 / −0.199722, inside reviewer B (6) [−0.206828, −0.161963] / [−0.204803, −0.152911] | −0.2019566 / −0.1997223; reviewer B intervals as in Part 1 | **REPRODUCED** |
| E-level, sorted upper bounds 16 r_max 𝒦_up/(j²χ_c²) | ≤ 1.175e74 / 7.52e73 / 7.83e73 (ζ = 1/5); ≤ 1.593e74 / 1.020e74 / 1.062e74 (ζ = 1/10) | 1.17437e74 / 7.51892e73 / 7.82729e73; 1.59261e74 / 1.01967e74 / 1.06149e74 | **REPRODUCED** (all rounded outward) |
| E-level, floor lower bounds 4c₂_floor,lo/(j²χ_c²) | ≥ −1.2722e77 / −8.145e76 / −8.479e76 (1/5); ≥ −1.2585e77 / −8.057e76 / −8.388e76 (1/10) | −1.272155e77 / −8.144986e76 / −8.479029e76; −1.258512e77 / −8.057636e76 / −8.388097e76 | **NOT REPRODUCED as outward bounds** for four of six entries: control 1/5 (−8.479029 < −8.479), CF-lo 1/10, CF-hi 1/10 and control 1/10. These are rounded inward by < 1e-4 relative. Outward versions are −8.4791e76, −1.2586e77, −8.058e76 and −8.389e76. The other two (CF-lo 1/5, CF-hi 1/5) are outward. |
| E-level values with the computed 𝒦′ (s2b) | not printed | sorted q₁: 3.8509e73 / 2.4655e73 / 2.5667e73 (1/5); 4.7023e73 / 3.0106e73 / 3.1341e73 (1/10). Floors: −1.27137e77 / −8.13993e76 / −8.47377e76; −1.25739e77 / −8.05045e76 / −8.38062e76 | s2b script values. The box recomputes the same arithmetic from its own 𝒦′, agreeing to 1e-10. |

### 5.3 Check of reviewer A's statements of the box Part 4 results

- **Row 2 cell.** The cell reads "reproduced by three independent codes at n = 3, 5, 7, 9 and by the box's own code at n = 11–21".
  - The n = 11–21 part is correct (Part 4.1).
  - "three independent codes" is overstated. reviewer A's S3 imports the group, characters and `apply_D` from reviewer A's S1, so S1 and S3 are one code base. The independent reproductions at n = 3–9 are reviewer A S1 (with S3 sharing its helpers) and the box (`indep_sm14.py` part A).
  - The n = 7 and n = 9 tables in the cell match Parts 2 and 4 entry by entry.
  - The repair statement is consistent with the tables: the two Z=+1 units at n = 7 sit in (E½, π = +), 32 against 30.
- **Row 3 cell.** Both figures in the cell are correct: "all 192 elements at n = 3–21, dev ≤ 1.1e-12" (Part 4.2 maximum 1.1e-12 at n = 21; the n ≤ 9 maxima are ≤ 3.2e-14) and "dev ≤ 4.5e-12 at n = 3–21" (Part 4.2). The closed formulas quoted in the cell are those that the box checks.
- **V_c(diag) closed form.** reviewer A's Part 4 paragraph says: "G = ζ² + Σ[(1−V²)s_i + V²s_i²] ≥ 0 iff V ≤ ζ/√3 + √(1+ζ²/3)". This matches box 4.4 exactly, including the separable form and the diagonal-only attainment. reviewer A also states that this certifies the tabulation range ρ ≥ V_c(diag): the box's `indep_step1.py` asserts it on every BZ grid point used.
- **Finite-n endpoint definition.** reviewer A states that its Ĩ, 𝒦 and 𝒦′ are BZ integrals, that the box's (ii) BZ grid is their discretisation, and that the open-cube 1/n deficit is a cube-surface term which does not enter the regional quantity. This is consistent with Part 4.3.
- **BZ-grid scatter.** "1.8e-5 at n ≤ 21" matches Part 4.3 (n = 13–21; n = 11 gives 1.8e-5 at ζ = 1/10). The stated cause, the kink of Φ ∼ (V_max − ρ)^{3/2}, is NOT SCRIPTED. reviewer A's s2b uses octant N ≥ 48, and the printed 𝒦 interval covers the scatter.

### 5.4 Per-row printability of the replacement cells

| Row | reviewer A step 1 | Printable as it stands? |
|---|---|---|
| 1 h_R current term | AGREE WITH ASTRA (adopts reviewer B (1)–(2)); toy regenerated (5.1) | Yes. The adoption is reflected in the regenerated toy and in s2b. The current-part zero for the adopted weight is confirmed (Tr[J-part P0] = 6.9e-20 smoke). |
| 2 Finer sector tables | REPLACEMENT TEXT | Yes, with one wording edit: "three independent codes" should read "two independent codes (reviewer A S1/S3, box)". All numbers match. |
| 3 Character remainder | REPLACEMENT TEXT | Yes. All numbers match Parts 2 and 4. |
| 4 Source-dual endpoints | REPLACEMENT TEXT | The bounds are correct and outward. The cell still says 𝒦′ is "PENDING-NUMERICS", but s2b has now produced it and the box reproduced it: 𝒦′ = 3.0757578e-3 / 3.7557397e-3. Printing as it stands is correct but stale. Suggested update: ceiling c₂ = 0.0123030 / 0.0150230, floor −40.618088 / −40.171573, with the a-priori bounds kept as the enclosure. The 𝒦 interval [7.4121e-3, 7.4125e-3] / [8.0700e-3, 8.0704e-3] is outward. |
| 5 q_tr | REPLACEMENT TEXT | Yes. reviewer B's six q_tr rows match Part 1 exactly. Edit: "to be regenerated" is now done (toy q_tr = −0.038490). |
| 6 Named blocker | REPLACEMENT TEXT | Yes. Its only number is x < 1/V_c(diag)² = 0.7942 / 0.8910, which is reproduced. |
| 7 Boundary error | REPLACEMENT TEXT | **Not as it stands.** γ_ζ = 5.7718e-2 / 2.1717e-2 are grid values, not the minimum; the minimum is 5.76965e-2 / 2.07354e-2. μ_ext = 3.196e-3 / 1.205e-3 should read 3.195e-3 / 1.151e-3. The conclusion "< exp(−10^150)" stands. The Combes–Thomas lemma is stated, not scripted. |
| 8 Atom-to-Weyl | REPLACEMENT TEXT | Yes (no numbers). |
| 9 Omitted-block | AGREE WITH ASTRA ([0,0] for both covers) | Yes. Box Part 4 dims sum 4n³ at every n. |
| 10 Leading-coefficient interval | REPLACEMENT TEXT | **Not as it stands**, for a rounding reason only. Parts (a) to (c) are correct; (a) matches Part 1. Part (c) is stale in the same way as row 4: the sorted value is now computed, c₂ = +0.0123030 / +0.0150230. In the E-level floors, four of six lower bounds are rounded inward: "−8.479e76" (control 1/5), "−1.2585e77", "−8.057e76" and "−8.388e76" (1/10). Outward versions are −8.4791e76, −1.2586e77, −8.058e76 and −8.389e76; alternatively print the computed floors (5.2 last row). All the sorted upper bounds are outward. |
| Box-findings text (𝒦, Φ, s3, toy, sign) | replacement text | Yes. The 𝒦 interval is outward. The Φ control now passes and can fail. The s3 controls can fail and pass. The toy statement is consistent with the regenerated toy, which still has positive q_band at n = 7 and negative at n = 5. The normalisation sign I := −Ĩ/6 is reproduced. |

Nothing in Parts 1 to 4 is changed, apart from the note in 5.2 on the Part 2 "fine" 𝒦 value.

---

## Part 6 — reviewer B step 2 (`astra_step2.md`, task `TASK_step2.md`): box check of the step-2 cells

reviewer B's step 2 contains no scripts. The box check is a single script, `step2/indep/indep_step2.py` (sha256 first 12 b7fc4164fbf3; own code). It ran at `<internal>/indep` on CPU (mpmath, sympy, numpy; GPU 2 not needed), with nice 10 and 16 threads. Run time 7.6 s, exit 0: 13 PASS lines and 1 FAIL. The FAIL is the row-10 outward check (6.2), a finding against the printed cell, not a script defect. Copies are in `box/results/indep/` (step2.log, indep_step2.json). Route 1 (busi) was not opened.

Step-2 verdicts:
- Row 1: ACCEPT FABLE'S TEXT.
- Rows 2, 3, 4, 5, 8, 9 and 10: ACCEPT WITH EDITS.
- Rows 6 and 7: REJECT, each with its own cell.

### 6.1 The Part-5 corrections, as printed in step 2

| Part-5 correction | Printed in step 2 | Status |
|---|---|---|
| Row 7: γ_ζ minima 5.76965e-2 / 2.07354e-2 | "γ_ζ ≃ 0.0576965/0.0207354" (row 7 cell and body) | **REPRODUCED.** Root of reviewer B's (7.1) 3t²(3−4t) = ζ²(1−2t)², unique in (0, 1/2): t = 0.0610634709950 / 0.0318926177147. Then γ = F(t) = 0.0576964710255 / 0.0207353874047, which equals the scan minimum over t ∈ [0, 1] (2·10⁴ points) and the 3D s-grid minimum (0.0576964916 / 0.0207354698, 401³ on [0, 0.2]³). The 1D reduction is confirmed. |
| Row 7: μ_ext = 1.151e-3 at ζ = 1/10 | "μ_ext = ½ log(1 + γ_ζ/9) ≃ 0.00319513/0.00115064" | **REPRODUCED** (0.0031951288724 / 0.0011506409761) |
| Row 10: four outward floor bounds | "−8.4791×10⁷⁶" (control, 1/5); "−1.2586×10⁷⁷, −8.058×10⁷⁶, −8.389×10⁷⁶" (1/10); the other two "−1.2722×10⁷⁷, −8.145×10⁷⁶" unchanged | **REPRODUCED.** All six are outward against −1.272155e77, −8.144986e76, −8.479029e76 / −1.258512e77, −8.057636e76, −8.388097e76 (Part 5.2). |
| Row 10: sorted upper bounds | 1.175e74 / 7.52e73 / 7.83e73; 1.593e74 / 1.020e74 / 1.062e74 | **REPRODUCED** (outward, Part 5.2) |
| Rows 4 and 10: 𝒦′ computed, not pending | Ĩ 1.2117396506 / 1.1983337559; 𝒦′ 0.0030757577838 / 0.0037557396834; b −20.302893 / −20.078275; u +0.012303031 / +0.015022959; l −40.618088 / −40.171573 | **REPRODUCED** (Part 5.2; agreement 2e-13 for 𝒦′) |
| Row 10: computed physical-scale table | sorted 3.8509e73, 2.4655e73, 2.5667e73 / 4.7023e73, 3.0106e73, 3.1341e73; floors −1.27137e77, −8.13993e76, −8.47377e76 / −1.25739e77, −8.05045e76, −8.38062e76 | **REPRODUCED** (Part 5.2, s2b values; box arithmetic agrees to 1e-10) |
| Row 2: two independent codes, not three | "two independent implementations at n = 3, 5, 7, 9 — reviewer A S1/S3 and the box — plus the box's separately constructed Part-4 implementation at odd n = 11, …, 21" | **REPRODUCED** (matches Part 5.3) |
| Row 5: toy regenerated | q_tr = −0.038490, half-step difference 1e-6; toy table q_band +0.031567, q_tr −0.038490, q_1 −0.006922, dual [−16.07847, 0.23831], n = 5 q_band −0.005603 | **REPRODUCED** (Part 5.1) |

### 6.2 New numbers in the step-2 cells

| Cell | Number or statement | Box | Status |
|---|---|---|---|
| Row 2 | n = 7 and n = 9 tables; rank and dimension formulas | Parts 2, 4 | **REPRODUCED** |
| Row 2 | SVD frames versus DEF §3 projected lexicographic Gram–Schmidt | frame-gauge statement | NOT SCRIPTED (no number) |
| Row 3 | Non-central remainders after the leading d_Γ² n³/48: 5n/12 + (πσ − π − κσ)/2, −n/12, −n/3 | The leading term is (d/192)[χ(1)·2n³ + χ(−1)·(−2n³)] = d²n³/48, since χ(−1) = −d on the spinorial isotypes. rank − d²n³/48 equals the three expressions with 0 violations at odd n = 3..21 (all 12 isotypes) | **REPRODUCED** |
| Row 3 | trace identity dev 1.1e-12, ranks dev 4.5e-12, n = 3..21 | Part 4.2 | **REPRODUCED** |
| Row 3 | gapped-homotopy justification | argument | NOT SCRIPTED (it is reviewer B's own Part-1 rank argument; the box checked it at n ≤ 21 through the ranks) |
| Row 4 | ∫₀^{ρ⁻²} x²(x^{−1/2} − ρ)²/(2x^{−1/2}) dx = 1/(105ρ⁵) | sympy exact: 1/(105ρ⁵) | **REPRODUCED** |
| Row 4 | ψ_K′(ρ) ≤ 1/(105ρ⁵) on the actual field (uses V(x) ≤ x^{−1/2}) | spot check at 40 values of ρ ∈ [1.06, 1.97]: max of 105ρ⁵ψ_K′ is 0.828 ≤ 1 | **REPRODUCED** (spot check; the inequality itself is analytic) |
| Row 4 | ⟨abs(q̃)⟩_BZ ≤ √(3/8) | ⟨abs(q̃)⟩ = 0.596901 ≤ 0.612372 | **REPRODUCED** |
| Row 4 / 10 | U_ζ = 4π√(3/8)/(105 V_c⁵); u ∈ (0, 4U_ζ]; l ∈ [−(32π/3)t_ζ − 4U_ζ, −(16π/3)(√(ζ²+3) + ζ)] | reviewer B prints formulas only. Box decimals: U_ζ = 0.0411955 / 0.0549207, so 4U = 0.164782 / 0.219683; l ∈ [−41.7500, −32.5647] / [−41.3979, −30.7446]. The computed 𝒦′ is 0.075 / 0.068 of U_ζ, and the computed l lies inside | **REPRODUCED** (formulas valid; decimals are box-computed, not printed) |
| Row 4 | "Strict positivity follows on an open phase-space set already in the region f₁ = 1" | argument | NOT SCRIPTED (consistent with 𝒦′ > 0 computed) |
| Row 4 | Richardson and `10*dev` are not validated quadrature remainders | the quoted code lines are present verbatim in `s2b_leading.py` | **CONFIRMED** (`return Ki,2*abs(Ki-c),p` and `Kpw=Kpw+10*dev*Kp+1e-12`) |
| Row 5 | per-point rows χ² q_tr ∈ (32π/j) 𝒯_ζ (six intervals) | Part 1 | **REPRODUCED** |
| Row 5 | X₀ formula | formula | NOT SCRIPTED |
| Row 6 | V_c⁻² ≃ 0.794192 / 0.891004, outward upper bounds 0.7942 / 0.8911 | Part 4.4 and 5.2: 0.7941920563 / 0.8910043228 | **REPRODUCED** (outward) |
| Row 7 | μ₀ = log(1 + ζ/6) ≃ 0.0327898 / 0.0165293 | 0.032789822823 / 0.016529301951 | **REPRODUCED** |
| Row 7 | enclosure ζ²/(2√(ζ²+3)) ≤ γ_ζ ≤ ζ | 0.0114708 ≤ 0.0576965 ≤ 0.2; 0.0028820 ≤ 0.0207354 ≤ 0.1 | **REPRODUCED** |
| Row 7 | χ_c⁻¹ = 9.135×10³⁷; 2μ_ext χ_c⁻⁴ ≃ 4.45×10¹⁴⁹ / 1.60×10¹⁴⁹, both below 10¹⁵⁰ | χ_c⁻¹ = 9.135e37. Exponents 2μ_ext χ₀⁻⁴(1 − 2χ₀³): control 4.44992e149 / 1.60252e149; CF-lo 6.67647e149 / 2.40435e149; CF-hi 4.27461e149 / 1.53939e149. All are below 10¹⁵⁰ | **REPRODUCED.** The claim "< exp(−10^150)" does not follow from μ_ext at any admitted point. With the P0 rate μ₀ alone the exponents would be 6.85e150 / 3.45e150 (CF-lo), above 10¹⁵⁰; that is the rate on which reviewer A's original row 16 rested |
| Row 10 | F₁(χ) ∈ [−32πB_ζ, 32πB_ζ] ⊂ [−62.911, 62.911] / [−62.504, 62.504] | 32πB_ζ = **62.9110536** / **62.5040107** | **NOT REPRODUCED as printed.** The printed bounds 62.911 / 62.504 lie inside 32πB_ζ by 5.4e-5 / 1.1e-5, so the stated "⊂" is false at the last digit. Outward: ±62.912 / ±62.505. The same inward rounding is in reviewer B's first answer (c₂ units), reviewer A's step-1 row 10 and box Part 3 (box erratum 2 below). |
| Row 10 | analytic comparator enclosure via U_ζ | as row 4 | **REPRODUCED** |
| Box findings | 𝒦 half-width 2e-7 means sorted half-width 8e-7; −0.0296675 (n = 15) lies outside −4·[0.0074121, 0.0074125] = [−0.0296500, −0.0296484] | arithmetic | **REPRODUCED** |
| Box findings | updated 𝒦 values 0.00741235577 / 0.00807022486; r_max 1.2654050827 / 1.5761697166 | Part 5.2 | **REPRODUCED** |
| S identity / normalisation | I := −Ĩ/6 ≃ −0.2019566 / −0.1997223; Q_w⁽⁰⁾ = −(16π/3) g Ĩ χ⁻² | Part 5.2 | **REPRODUCED** |
| Rows 8, 9 | no new numbers | — | — |

### 6.3 The rejected rows against the box evidence

**Row 6 (REJECT; held cell: UNDECIDED, source-dominant literal mark/source contractions).** The box has no computation that decides whether scalar sorting fractions exhaust the block contractions, and nothing in Parts 1-5 contradicts reviewer B's reason. Its supporting facts are checked:
- **The reviewer A quote.** reviewer B quotes reviewer A's row-10 sentence "fills with marks stranded above the Fermi level after exact crossings are bounded only by the floor". It is present verbatim in `fable_step1.md`.
- **Pocket bound.** The cell's pocket bound x < V_c⁻² = 0.794192 / 0.891004 is reproduced, and its outward bounds 0.7942 / 0.8911 are correct.
- **Toy evidence.** The toy (Parts 2.2 and 5.1) has all twelve block Fermi gaps certified positive along τ ∈ [0, 1], and no pocket at its scale. It therefore exercises neither the exact-crossing nor the stranded-mark case, and supports neither text on this point.
- **Verdict.** The rejection holds against the box evidence, which is silent on the deciding question.

**Row 7 (REJECT; held cell: exact-parent truncation error [0,0], reduced boundary certificate UNDECIDED L_bd).** Every number in the held cell is reproduced (6.1, 6.2). reviewer B's arithmetic objection is confirmed: with μ_bd = μ_ext the exponent 2μ_ext χ₀⁻⁴ is below 10¹⁵⁰ at all six (point, ζ) combinations, by factors 1.5 to 6.5, so "< exp(−10^150)" is not implied. The operator-level Combes–Thomas and Lieb–Robinson lemmas, which the held cell names as missing, were NOT SCRIPTED in Part 5 and are not scripted here. The rejection holds against the box evidence.

### 6.4 Box errata (the box's own statements)

1. **Part 5.2, row "μ_ext"; Part 5.4, row 7; the Part-5 summary to the lead.** The box wrote that the rate correction left "< exp(−10^150)" unaffected ("the conclusion … stands"). That was wrong: the box never computed the exponent. With μ_ext the exponent is 1.5e149 to 6.7e149, below 10¹⁵⁰ (6.2). reviewer B's step 2 identified this ("notwithstanding the box's prose endorsement"). Both rate values are positive, so the bound remains doubly exponentially small; only the stated threshold fails.
2. **Part 3, row "Leading-coefficient interval".** The box printed reviewer B's envelope as "c2 in +-62.911 / +-62.504", rounded inward. Outward: ±62.912 / ±62.505 (6.2).
3. **Part 5.2, row 𝒦.** As noted there, the Part-2 "fine" value 7.4123541e-3 was 1.7e-9 low. This is unchanged.

Nothing else in Parts 1 to 5 is altered.

### 6.5 Addendum: six-decimal rounding of the per-point rows (row 5 cell)

Step 2's row-5 cell prints the six exact-parent χ² q_tr rows to six decimals as "outward per-point rows", while adding "the authoritative endpoints are the exact algebraic 𝒯_ζ endpoints". Against the 12-decimal directed values from `sm14_source_bounds.py` (Part 1), 7 of the 12 printed endpoints are rounded inward, each by at most 4.4e-7:

| Point | Printed | Exact (12 dp, directed) | Outward at 6 dp |
|---|---|---|---|
| CF-lo 1/5 | [−4.646374, 15.999551] | [−4.646374380252, 15.999551256100] | [−4.646375, 15.999552] (both printed ends inward) |
| CF-hi 1/5 | [−3.717828, 12.802150] | [−3.717828132939, 12.802150000470] | [−3.717829, 12.802151] (both inward) |
| control 1/5 | [−3.793300, 13.062034] | [−3.793300044038, 13.062033645480] | [−3.793301, 13.062034] (lower inward) |
| CF-lo 1/10 | [−4.862717, 15.874906] | [−4.862716970616, 15.874905585497] | as printed (outward) |
| CF-hi 1/10 | [−3.890936, 12.702414] | [−3.890936131344, 12.702413917475] | [−3.890937, 12.702414] (lower inward) |
| control 1/10 | [−3.969922, 12.960273] | [−3.969922134811, 12.960272919999] | [−3.969923, 12.960273] (lower inward) |

**Status: NOT REPRODUCED as outward decimals.** The exact algebraic endpoints, which the cell declares authoritative, are REPRODUCED. The same six-decimal figures appear in reviewer A's step-1 row 5, and the box's Part 5.4 called them matching. That is box erratum 4: they match only to rounding.
