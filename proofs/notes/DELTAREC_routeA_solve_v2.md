# DELTAREC route A (solve), v2: lower enclosures of delta on every radial cell, records 0..320 (owner priority #762 step 1). Route 1, BLIND to route 2.

## Scope (it applies to every number below)

The enclosures are rigorous FOR THE ANCHORED M1 MAP / STORED INTERPOLANT:
- the fp64 PPoly sources of each record (SmoothedSources5's splines), taken as data;
- the fp64 series start at r0 = ell/64;
- the fixed collar.

The arithmetic assumptions are IEEE binary64 round-to-nearest, and mpmath's exp, sinh, erfc and coth accurate to 1e-25 relative at 50 digits. The exp(-C) values additionally assume libm exp is within 1 ulp; this enters only through a factor (1 - 1e-15).

**Errors that are NOT enclosed:**

1. The source/state error between the stored shell totals and the exact discrete state. The replay source checks give about 1e-13, but that is not an enclosure here. A certified eps_C exists only at records 104, 107 and 200, from covenc, and is not used here. NOT ENCLOSED.
2. The representation error of the engine's spline interpolant against the exact kernel sums. The engine reports about 1e-12 relative (SmoothedSources5.spline_error), but it is NOT ENCLOSED. delta is computed for the interpolant.
3. The engine's own truncation and solve error. It does NOT enter, because the Picard interval march replaces DOP853. The engine's DOP853 U is used only as a containment check, and it lies inside the enclosure at every node.

The time slabs between records are not covered.

## Timestamps (from `date -u` and the logs)

- Brief received and started at 06:13:06Z.
- Method note DELTAREC_METHOD_routeA_v2.md frozen at 06:17:41Z (md5 9836f363).
- First number: the test at record 104 with delta_rec_v3, at 06:25:43Z.
- ADDENDUM1 frozen at 06:26:24Z (md5 43355e74), before production.
- Production with delta_rec_v4: 06:26:49Z to 06:35:55Z, DONE for 321 records, 0 failures.
- Report at 06:36:28Z.

## Method (DELTAREC_METHOD_routeA_v2 + ADDENDUM1)

**Cells.** delta(r) = q_eff^2 S_kappa M/(m J_+^2) on cells of width ell/64, up to r_collar, with q_eff^2 = 220 and m = 0.2.

**S_kappa.** Lower and upper bounds per cell:
- K0 is unimodal in r. This is proved via the Mittag-Leffler expansion: (x coth x - 1)/x^2 is decreasing. So the minimum of K0 on a cell is at an endpoint.
- Phi is increasing, and is evaluated with a cancellation-free erfc closed form.
- Windows of 12 ell, plus an analytic far-shell tail bound of 1e-26.
- The values are computed in mpmath at 50 digits and rounded outward.
- The centre cell is factored as (S r^3).

**M.** An interval march with Picard containment on adaptive sub-steps, min(ell/256, 1e-4 r), from the fp64 start, with outward rounding everywhere. On [0, r0], the series M/r^3 is used.

**rho and J.** Interval Horner on the PPoly pieces.

**C.** C_num with I_S upper (ADDENDUM1: the sharper peak-cell bound K0 <= pref min(1, ell^2/(2 r R))), and exp(-C) taken downward.

**q convention.** q_eff^2 = W q_*^2 = 880/4 = 220. It is stated once; delta and d both scale as q^2, and C is invariant. v1's "plain q_*^2 gives 0.138" is retracted as convention-mixed.

## Results (321 of 321 records enclosed; for the anchored M1 map)

**History-wide lower enclosure: inf delta >= 117.52.**
- It is attained at record 287, cell [16.875, 16.9375]. The cell is certified TRAPPED (U_lo > r).
- The |J| variant gives the same value: 117.52, record 287.
- The certified-UNTRAPPED-cell inf is 215.75, at record 189.
- For comparison, v1's nominal grid values were 121.16 and 217.9. The rigorous lower bounds sit about 3 % and 1 % below them. That is the M width plus the cell min/max of S and J.

| band | min delta_lo | record | cell [r_k, r_k+1] | flag | min over untrapped cells (record) |
|---|---|---|---|---|---|
| 0-40 | 4.566e4 | 0 | [187.125, 187.1875] | untrapped | 4.566e4 (0) |
| 41-95 | 3318.3 | 95 | [21.6875, 21.75] | untrapped | 3318.3 (95) |
| 96-110 | 132.14 | 107 | [13.6875, 13.75] | trapped | 992.6 (99) |
| 111-200 | 124.10 | 183 | [18.8125, 18.875] | trapped | 215.75 (189) |
| 201-320 | 117.52 | 287 | [16.875, 16.9375] | trapped | 278.39 (240) |

**Flags at the per-record inf.** 204 records trapped and 117 untrapped. The largest number of ambiguous cells in any one record is 47 (cells where U_lo <= r <= U_up). The inf was never attained on an ambiguous cell.

**Centre.**
- The J spline's constant coefficient at r = 0 is NOT exactly zero in 311 of 321 records (max |c0| = 3.39e-20). It is exactly 0 in 10 records.
- So J(0) != 0 in the model. delta(0) is then FINITE and positive, because S M stays finite.
- The rigorous centre-cell lower bound is >= 8.11e6 over all records. The centre is never binding, and there is no floor.

**Tail.** The cell bounds run up to the collar. J -> 0 makes delta large. The tail is never binding.

**Clock constant** (for the anchored M1 map, upward):
- C_num per record, with that record's own rigorous inf, ranges from 1.371 to 47.414.
- So N >= exp(-C) >= 2.56e-21 on every record.
- The history-wide C (with delta = 117.52) has max 47.414 (record 287).

**M enclosure quality.** The maximum relative width over all records and nodes is 0.92 %. The engine's DOP853 U lies outside the enclosure at 0 nodes, over all records.

## Files

**Box <internal>/** (171 MB):
- S_cells.npz (fa36357f);
- per record rec/drec3_NNNN.json, plus rec/drec3_NNNN.npz with the SLABGAP H_d anchor inputs per cell: M_lo (float32, rounded down), j_up = J_+ upper (float32, rounded up), delta_lo, and the trapped flag, plus Mr3_lo for the centre cell;
- DELTAREC_v3_summary.json (ff935f3c; the file name comes from the v3 agg code; produced by v4);
- DELTAREC_v4_report.json (020ffccd);
- run.log (DONE).

**Box <internal>/deltarec_v3:** the S node files (mpmath values, reused as data by v4), and the v3 test-104 outputs (superseded).

**Deliveries (this directory):**
- DELTAREC_routeA_solve_v2.md (this file);
- DELTAREC_v2_summary.json and DELTAREC_v2_report.json (copies of the two box JSONs above);
- DELTAREC_METHOD_routeA_v2.md (9836f363) and DELTAREC_METHOD_routeA_v2_ADDENDUM1.md (43355e74).

**Code** (local singcert dir and box code dir):

| file | md5 | notes |
|---|---|---|
| delta_rec_v3.py | 8d60dc26 | S nodes and the test |
| delta_rec_v4.py | a488ea54 | production |
| deltarec_report_v1.py | 8756abb5 | |
| run_deltarec_v3_A.sh | 6eef01d6 | |
| run_deltarec_v4.sh | 922f6abd | |
| run_deltarec_v3_B.sh | | written, never run |

## Not claimed

- Errors 1 and 2 of the scope section are not enclosed.
- The time slabs between records are not covered (SLABGAP).
- Only the held-reference law is covered.
- "Rigorous" here always means: for the anchored M1 map, under the stated arithmetic assumptions.
