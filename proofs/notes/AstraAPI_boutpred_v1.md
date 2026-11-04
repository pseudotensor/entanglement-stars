
## 1. The predicate is present

**The earlier “predicate absent” objection is cleared.** `05_LIGHTEND_ADDENDUM3_v1.txt`, §3, states verbatim:

> Status "CERTIFIED (rigorous rounding)" iff: H <= budget; DC2B K1, K2 (v2), K3 PASS; window vs global within 2 (rb_glob + rb_win);  
> L_out tail non-growing (|c(1024) - c(512)| <= |c(512) - c(256)|); two-end ENCLOSED (|c(1024) - c(256)| <= budget);  
> and in final mode: inner parent check PASS (every coefficient |c_in(native, evd, L = 1024) - c_DC2B(base, 1024)| <= its budget),  
> native-chain gap gates PASS (both drivers), and every hrecon family whose b_out monomial is nonzero at eps = 1/128 is CONTROLLED.

Thus the floorless comparison is **non-growing**, not strictly shrinking:
\[
|Q_{1024}-Q_{512}|\le |Q_{512}-Q_{256}|.
\]

**Error and zero handling:**
- The same section specifies `"L_term = 2 |c(1024) - c(512)|"` alongside variants, pavg, rigorous rounding and fp_sum. `06_LIGHTEND_BULKSUB_AUDIT_v1.txt`, §(iii), attributes to `lightend_deliver_v1.py`, `heavy_out`, ll.170/205: `"H rounded up (h_up)"` and `"budget = max(1e-3 |c|, 1e-6)"`.
- Addendum 3, §2: `"half_width is null iff a component is null, and then status is never CERTIFIED"`.
- **No numerical-error allowance or near-zero exemption is stated for the tail comparison.** Exact equality, including \(0\le0\), passes; a zero earlier increment with a nonzero later increment fails. The budget floor does not enter this comparison.
- Addendum 3, §3, requires K6 `"ratios R(1/32)/R(1/64), R(1/64)/R(1/128) in [5.657, 11.314]"` and
  > |R(1/128)| > the MEASURED enclosure 2 |exact - exact_evr| + |bulk(8192) - bulk(4096)| + sum H_in |mon_in| / N_in + sum H_out |mon_out| / N_out.
- The imported DC2B gate definitions, detailed rounding conventions and block **ZERO** classification are **not reproduced** in these attachments. That does not prevent evaluating—and failing—the explicit necessary tail condition. The original three-split test remains unexhibited: audit §(iii) says `"NOT implemented anywhere"`.

## 2. Apply it: all 40 deposited candidates fail

Use the common-subtraction construction in `00_AstraAPI_boutcmp2_v1.md`, §2:
\[
Q=X_{2048}-B_{\rm loc}^{[1]},\qquad \Delta Q=\Delta E+\Delta W.
\]

Its signed-increment table, combined with `09_BOUT_B2_EXTRACT_v1_part2.md`, `"terms/W"`, gives **\(|\Delta_2Q|>|\Delta_1Q|\) for every record**. Examples, rounded:

| Record | \(|\Delta_1Q|\) | \(|\Delta_2Q|\) |
|---|---:|---:|
| \(k=1,\ e00\) | \(9.31\,10^{-17}\) | \(1.43\,10^{-15}\) |
| \(k=5,\ h00\) | \(1.84\,10^{-8}\) | \(2.68\,10^{-7}\) |

| Coefficients | \(k=1\) | \(k=2\) | \(k=3\) | \(k=4\) | \(k=5\) |
|---|---|---|---|---|---|
| **Each of** e00, e01, e11, e02, f00, f01, h00, g00 | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED | NOT CERTIFIED |

**These are failures of the literal rule on deposited centers**, not a claim that exact tail growth is resolved outside numerical errors in every case. Passing the \(10^{-6}\) diagnostics cannot waive this separate condition. Missing runtime/gate evidence cannot rescue a failed necessary condition.

## 3. Minimal remaining runtime comparison

`01_BOUT_DECIDER_SPEC_v1.md`, amendment 3(a), requires `"One index set and one truncation"` and `"B_impl zero at pairs where it subtracted nothing"`.

A must export, **from the actual v5 `B_loc`/subtraction runtime**, per parent, cutoff and summand family:

- ordered row/pair indices; truncation and allocation weights;
- subtraction-applicability masks, retaining zero terms;
- polynomial coordinates/anchors; wall-straddling and added-site conventions;
- fixed finite-part offsets.

Compare these against **B-v2**, reporting **mismatch counts and maximum offset differences**, with code/data hashes. B’s export described in `00_AstraAPI_boutcmp2_v1.md`, §1, names `bout_b_run_v1.B_impl`; therefore its **v1→v2 convention equivalence must also be exhibited**. Matching formulas alone is not that runtime check.