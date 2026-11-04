
## Decision

**Reasons (2) and (3) are cleared; every numerical increment test and the amendment-3(d) label comparison passes. Reason (1) is substantially reduced, but the supplied B-v1 export does not establish the complete A-v5/B-v2 runtime comparison. Reason (4) remains: the original acceptance predicate is still absent. No new endpoint is CERTIFIED.**

### 1. Recheck the previous objections

| Objection | Checkable evidence | Finding |
|---|---|---|
| **(1) Common indices/masks/anchors** | `02_BOUT_EXPORT_SUMMARY_v1.md`: `"X(L) = sum_{z < L} summand (sharp, weight 1, math.fsum)"`; every parent has `"anchor_x" = 2047`, `"z = 2047 - x (never re-anchored)"`, and `"r = x + 1"`. Thus zero-based \(x=2047\) correctly means \(n=2048\), **not an anchor mismatch**. The export identifies all 11 summand families, polynomial continuation, zero terms and the two wall-straddling rows. | **B’s exported convention is exhibited. Full amendment (a) remains unverified:** this export names `BOUT_B_REPORT_v1.json` and `bout_b_run_v1.B_impl`; it contains no A-v5 `B_loc` runtime comparison or explicit v1→v2 index/mask equivalence check. |
| **(2) Solver 2 invalid** | `13_BOUT_B2_EXTRACT_v1_part4.md`, `"Q1_form2dd_vs_dense"`: all 70 row-family checks and 330 sum checks pass. `10_BOUT_B2_EXTRACT_v1.md`: `"s_8192_status" = 'complete (form II dd validated by Q1 at 2048 and 4096)'`; `"s on every parent = 2\|X(f1dd) - X(f2dd)\| + quad + tail + fp_dd"`. | **Cleared on the deposited numerical evidence.** The 8192 bounds are now parent-specific, not inherited from 2048. |
| **(3) A’s excessive half-widths** | `04_BOUT_A5_EXTRACT_v1_part2.md`: `"units/max_halfwidth = 2.4733002494238255e-09"`. `09_BOUT_A5_EXTRACT_v1_part7.md`, `deposit_N0_1`, largest direct `increment_R.hw_1024-512`: \(1.7319269643092062\,10^{-9}\), \(k=5,h00\). | **Cleared for all eight coefficients and all five \(k\).** |
| Reproduction | `14_BOUT_B2_EXTRACT_v1_part5.md`: all 120 `"ok"` entries are `True`; largest absolute `"diff"` is \(1.459058915354028\,10^{-10}\), `k=1,f01_1024`, against `"tau_r" = 1e-06`. | **PASS.** |
| A’s admission controls | `04_BOUT_A5_EXTRACT_v1_part2.md`: `"KC4_R_even_in_k": {"PASS": true}` and `"KC5_e00_edge_only": {"PASS": true}`, also passing for `N0_1_gates`. | **PASS.** The remaining `"N_gradient/PASS_ratio_ge_6 = False"` is explicitly labelled a **diagnostic** in `05_BOUT_A5_EXTRACT_v1_part3.md`, not an additional stipulated return gate. |

**Exact remaining amendment-(a) check:** compare the actual A-v5 and B-v2 ordered indices, truncation/allocation weights, subtraction-applicability masks, polynomial evaluation coordinates, added-site/wall conventions and fixed offsets. Report mismatch counts and maximum offset differences, retaining zero subtraction where applicable. A’s statement in `03_BOUT_A5_EXTRACT_v1.md`—`"the code value lightend_v1.moments_gen on zero rows"`—identifies its subtraction, but does not exhibit the corresponding `B_loc` runtime objects.

### 2. Formation and increment tests

Use A’s literal-\(N_0=1\) branch, without replacing B’s implemented subtraction. Write
\[
b_A=B_{{\rm impl},A}^{[1]},\quad b_B=B_{{\rm impl},B},\quad
\delta=b_A-b_B,
\]
\[
R=R_A^{[1]}+\delta,\qquad R^{(0)}=R_A^{(0),[1]}+\delta,
\]
\[
E=X_{8192}-B_{\rm loc}^{[1]},\qquad
C=W+R+E=X_{2048}-b_B,\qquad Q=C-R=W+E.
\]

Sources: `08_BOUT_A5_EXTRACT_v1_part6.md`, `"deposit_N0_1"`; `11_BOUT_B2_EXTRACT_v1_part2.md`, `"terms/X_8192"` and `"terms/W"`; `12_BOUT_B2_EXTRACT_v1_part3.md`, `"terms/B_impl"`.

**The formed 120 values of \(C\), and their 80 signed increments, are the `"terms/C"` table in `12_BOUT_B2_EXTRACT_v1_part3.md`.** For example, \(k=1,e11,L=256\):
\[
0.009054034165059809
-0.000002306909337983916
+0.010327359489062186
=0.01937908674478401.
\]

Below, each cell gives the newly formed signed pair
\((\Delta_{256\to512}E,\Delta_{512\to1024}E)\), rounded. For every record,
\[
\Delta Q=\Delta E+\Delta W,\qquad
\Delta C=\Delta R+\Delta Q.
\]
The separate \(\Delta R^{(0)},\Delta(R-R^{(0)})\) and direct bounds are supplied in `09_BOUT_A5_EXTRACT_v1_part7.md`; apply \(\Delta\delta\) to \(R,R^{(0)}\).

| Coefficient | \(k=1\) | \(k=2\) | \(k=3\) | \(k=4\) | \(k=5\) |
|---|---|---|---|---|---|
| e00 | −9.31e−17, −1.43e−15 | −4.57e−16, −5.89e−15 | −1.06e−15, −1.33e−14 | −1.91e−15, −2.37e−14 | −3.00e−15, −3.71e−14 |
| e01 | −9.59e−14, −1.75e−12 | −4.16e−13, −7.14e−12 | −9.49e−13, −1.61e−11 | −1.70e−12, −2.87e−11 | −2.65e−12, −4.48e−11 |
| e11 | −6.22e−11, −1.97e−9 | −2.62e−10, −7.98e−9 | −5.95e−10, −1.80e−8 | −1.06e−9, −3.20e−8 | −1.66e−9, −5.00e−8 |
| e02 | −6.53e−11, −1.20e−9 | −2.68e−10, −4.87e−9 | −6.05e−10, −1.10e−8 | −1.08e−9, −1.95e−8 | −1.68e−9, −3.05e−8 |
| f00 | −6.38e−13, −4.89e−12 | −2.62e−12, −1.98e−11 | −5.93e−12, −4.47e−11 | −1.06e−11, −7.95e−11 | −1.65e−11, −1.24e−10 |
| f01 | −2.40e−10, −3.74e−9 | −9.88e−10, −1.52e−8 | −2.23e−9, −3.42e−8 | −3.98e−9, −6.09e−8 | −6.22e−9, −9.51e−8 |
| h00 | +7.34e−10, +1.07e−8 | +2.94e−9, +4.29e−8 | +6.62e−9, +9.65e−8 | +1.18e−8, +1.71e−7 | +1.84e−8, +2.68e−7 |
| g00 | −8.81e−10, −8.64e−9 | −3.54e−9, −3.47e−8 | −7.96e−9, −7.80e−8 | −1.42e−8, −1.39e−7 | −2.21e−8, −2.17e−7 |

With \(a_\Delta\) A’s direct increment half-width, propagate
\[
s_{\Delta E}=s_{\Delta X_{8192}}+a_\Delta,\qquad
s_{\Delta Q}=s_{\Delta X_{2048}}+a_\Delta.
\]
The shared subtraction cancels; **8192 is unnecessary for the direct \(Q\) bound**.

| Test, all 40 records × both increments | Maximum absolute center | Upper bound on propagated \(s\) | Result |
|---|---:|---:|---|
| \(\Delta W\) | \(2.665\,10^{-15}\) | \(7.349\,10^{-10}\) | **80/80 PASS** |
| \(\Delta E\) | \(2.678\,10^{-7}\) | \(2.100\,10^{-9}\) | **80/80 PASS** |
| \(\Delta C-\Delta R\) | \(2.678\,10^{-7}\) | \(2.100\,10^{-9}\) | **80/80 PASS** |
| 4096–8192 control | Zero reported centers at all cutoffs and increments | Cutoffs \(<6.533\,10^{-10}\); increments \(<7.349\,10^{-10}\) | **PASS** |

Sources for the last table: the preceding arithmetic and `11_BOUT_B2_EXTRACT_v1_part2.md`, `"terms/W"` and `"terms/control_4096_8192"`. Every threshold is therefore **\(\max(10s,10^{-6})=10^{-6}\)**. Even \(|\text{center}|+s\) is below it. Formed \(E,Q\) cutoff half-widths are below \(2.801\,10^{-9}\).

### 3. Amendment 3(d): **fixed-\(r\) label comparison PASSES**

The common-subtraction conversion leaves \(G=R-R^{(0)}\) unchanged. From `09_BOUT_A5_EXTRACT_v1_part7.md`, every direct increment satisfies
\[
|\Delta G|+s_{\Delta G}<1.732\,10^{-9}
<10^{-6}\le\max(\tau_r,0.05|\Delta R|).
\]

Thus **all 80 comparisons support the stipulated fixed-\(r\) mechanism label**, rather than the omitted-gradient alternative. This is a diagnosis of the deposited quantities, **not certification**, and does not replace the remaining runtime consistency check.

### 4. Original-rule verdict and printable conclusion

| Records—each coefficient at each listed \(k\) | Original-rule verdict |
|---|---|
| e00, e01, e11, e02, f00, f01, h00, g00; each at \(k=1,2,3,4,5\) | **UNDECIDED—all 40; none CERTIFIED** |

The missing statement is the **verbatim necessary-and-sufficient original acceptance predicate for the new \(Q=C-R\) triplet**: its exact floorless shrinking comparison, numerical-error/zero handling, and unchanged audited gates with the required new-run outputs. `00_BOUT_DECIDER_SPEC_v1.md` only says `"retain the audited gates unchanged"`, rejects `"below-floor tails count equal"`, and calls the three-split test `"NOT implemented anywhere"`. Those statements do not specify that predicate. Existing publication status consequently remains **NOT CERTIFIED**.

The disclosure should also accompany any report: v5 is a **post-exposure numerical refinement of the sealed v4c prediction**, not a freshly blind prediction. It does not itself demonstrate a fitted repair; verifying the claimed bounds-only change requires the sealed v4c→v5 method/code diff.

**Strongest printable sentence:**

> For the deposited mapped sums, every prescribed cutoff-increment diagnostic passes at \(10^{-6}\), with propagated numerical half-widths below \(10^{-8}\), and every amendment-3(d) comparison supports the fixed-\(r\) subtraction diagnosis. No heavy-outer endpoint value is newly certified: the cross-group runtime consistency comparison and the original acceptance predicate remain to be established.