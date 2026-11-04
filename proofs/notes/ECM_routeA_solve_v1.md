# ECM route A (solve), v1: full-source energy-current margins on the reference collapse (lead #747, #750). FROZEN 1 Nov 2026 ~00:15Z. NUMERICAL DIAGNOSTIC. Route 1, BLIND to route 2.

**Frozen beforehand:** ADDENDUM_ECM1_v1 (b8272c7a) and the (H') add-on ADDENDUM_ECM1_Hprime_v1 (88bc6607).

**Code:**
- ecm_margins_v2 8c36486a (v2 splits matter and tail at R_s);
- ecm_sectors_v1 fbcd6b9b;
- ecm_hprime_v2 f42a7fd4;
- ecm_aggregate_v1 ed74b908.

**Outputs, on the box:**
- <internal>/ecm_v2 (per-record ecm_NNNN.json, ECM_aggregate_v1_final.json md5 ba0354dd, hprime_v2_all.json md5 29f47a8f);
- <internal>/rec (sec_NNNN.json).

## Law and sets

**Law.** The history c19_rs64_A_R2_tau8_ell4_avg_r11 uses the HELD (level-following) reference P_* = R2, the stored-mask eigenvectors.
- There is NO polarization functional, so **rho_pol = J_pol = 0 STRUCTURALLY**: the slot is empty, not measured small.
- The full sources are the engine's own: sea - R2 + dust per shell, w_k = 16 kappa, the engine kernels, on the dense grid with step ell/64.

**Sets:**
- untrapped: U = 2GM < r;
- matter region: r <= R_s = (last |rho| > 1e-6 max) + 8 ell;
- tail: r > R_s;
- c = nu3 = 0.891175 (ma = 0.2).

## (A) Full-source margins, all 321 records

**Untrapped matter region, records 22-320.** |J| <= nu3 rho holds at EVERY point: there is no negative margin and no rho <= 0 point. Max |J|/rho by band:

| band | max \|J\|/rho | where |
|---|---|---|
| 41-95 | 0.821 | record 41, at the support edge r = R_s = 436 |
| 96-110 (formation) | 0.500 | record 96 |
| 111-200 | 0.811 | record 153, r = 111 |
| 201-320 | 0.798 | record 201, r = 261 |

The margin to nu3 is >= 8 % throughout. The conservative pass |J| + eps_J <= nu3 (rho - eps_rho) holds on records 35-320.

**Untrapped matter region, records 0-21** (the pre-collapse slab).
- Negative margins appear only at the outer support edge, where rho is at its numerical floor (rho_max ~4e-6, edge rho ~1e-14).
- The MINIMUM margin is m_+ = -1.66e-14, at record 6, r ~ 296 (interval 294-319, R_s = 318.6); m_- reaches -7.8e-17 at record 12.
- rho <= 0 at a few edge points on records 5-16 (down to -6.5e-16). |J|/rho there reaches 1853 (record 9, r = 311).
- The clock contribution over the negative intervals is <= 2.3e-10, against L ~0.22.
- The conservative pass fails on records 0-34, at these edge points only, which are within the accuracy floor.

**Vacuum tail.** |J|/rho reaches 4.9e6 with rho at ~1e-12 relative. Its clock contribution over negative-margin points is <= 8.4e-6 (record 102).

**Trapped.** 216 records have trapped intervals (e.g. record 104: r in [3.25, 47.6]). The full-current cap is max |J| = 0.01227 (record 105).

## (B) Sector decomposition (42 records)

Records: 96-110, 112, 120, and every 8th from 128 to 320. Sectors use P = P_* = R2 per channel:
- occupation sector: Delta_diag = PCP + QCQ - P;
- coherence sector: Delta_coh = PCQ + QCP.

**Checks.** The sector blocks sum to the full totals to <= 7.2e-12. The full totals equal the stored eps_tot and J_tot to <= 2.6e-12.

**Untrapped matter region:**
- min total margin: m_+ = 9.0e-10 and m_- = 7.7e-11, both positive, at the support edge.
- min occupation-alone margin: 9.0e-10 and 7.7e-11, also positive. There is NO record where the occupation sector alone is negative.
- coherence ("res" = coherence; pol = 0) min: -1.69e-4 (+) and -1.62e-4 (-).
- the coherence sector is at most ~1 % of the occupation sector: max |rho_coh|/|rho_occ| = 0.0101 and max |J_coh|/|J_occ| = 0.0112.
- The test -(c rho_res +- J_res) <= c rho_occ +- J_occ fails at ZERO points on all 42 records.

## (H') Mean-enclosed-density criterion (reviewer A Theorem 3', lead #750), all 321 records

**Identity.** (ln N)' = J_l sqrt(6 pi G / rho_bar), with rho_bar = 3M/(4 pi r^3). It is verified to a max relative residual of 1.9e-5, limited by the finite difference on [ell, R_s].

**On the untrapped support of the current** (U < r, |J_l| > 1e-6 max):
- c' = max |J_l|/rho_bar <= 1.183 (record 134, r = 29.4). For example, record 104 gives 0.165 and record 174 gives 0.551. A finite c' holds on every record.
- min rho_bar = 6.3e-8 > 0: there is no hollow.

**Theorem 3' bound against the actual drop.**
- The integral form c' int_S sqrt(6 pi G rho_bar) is >= the actual clock drop over the untrapped support on ALL records (max actual/bound = 0.545).
- The closed form 3c' sqrt(2 G M_sup / r_in) is >= even the total drop on all records, but it is loose.
- Most of the clock drop lies on TRAPPED intervals: at record 104, 1.96 of 2.10; at record 174 (max L), 2.31 of 2.68. That part needs the full-current cap (max |J| = 0.0123).

## Reading (diagnostic)

**Untrapped matter (records 22-320).** On the untrapped matter region of every record from 22 on, the full averaged sources satisfy |J| <= nu3 rho with a >= 8 % margin, and the occupation sector alone satisfies it. The coherence sector is at the 1 % level. (H') holds with c' <= 1.18 everywhere.

**Failures.** The only failures of the strict pointwise criterion are:
- numerical-floor edge points in the early slab (clock weight <= 2.3e-10);
- the vacuum tail (clock weight <= 8.4e-6).

**Trapped intervals.** Most of the clock drop in the collapse happens on trapped intervals, where only the current cap applies.

**Not claimed:** no enclosure; no preservation by the dynamics; the held-reference law only (T_W is not available).
