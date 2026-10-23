# Evolved-cut entropy from the pure-state complement

Scope [R, N, S]: request 125, sixty channels at the six authorized matched partitions of the \(\sigma/a=4\) run. **All six comparisons satisfy the requested inequality under the fixed floating-point operation model and the supplied first-route allowance. An unconditional error certificate is not established.** The own weighted upper bounds are \(0.00522\)–\(0.00675\) nats, far broader than the observed gaps; this is not a precision claim at \(10^{-7}\) nats.

The ideal state is \(P_T=T(T^\dagger T)^{-1}T^\dagger\), the range projector of the represented source sea/dust columns, whose selected occupations are all one. Purity gives \(S_X(P_T)=S_Y(P_T)\). The numerical reading therefore uses the **saved complementary-block entropy**, of dimension \(72\)–\(116\), and subtracts the saved small-block held-vacuum entropy. The held target is the negative projector of the exact finite radial operator on the represented applied fields. This is a separately defined polar control; it neither repairs a rejected raw covariance nor identifies the pre-storage physical history. T12 is a charge-fallback control. T15 and T40 are the matched paper endpoints, using their specified one-site outward partitions.

The [frozen derivation](<internal>/EVOLVEDCUT_purity_bound_spec_v1.md) bounds source-column range changes in the computed polar factor, covariance formation, measured-residual evaluation, eigenvector nonorthogonality, the separate `eigvalsh`/`eigh` spectrum difference, and held-projector/operator errors. For each state or held block it propagates the resulting Frobenius errors through
\[
 U=d\,h\!\left(\min\left\{\frac{f+\beta_F}{\sqrt d}+\xi,\frac12\right\}\right)+\zeta_d,
 \qquad U_\Delta(T)=\sum_{\kappa=1}^{10}16\kappa\,[U_{C,\kappa}+U_{H,\kappa}],
\]
where \(h\) is binary entropy, \(f\) bounds matrix formation and the bridge to the ideal projector, \(\beta_F\) bounds the small eigensystem residual after orthogonality/rounding inflation, \(\xi\) bounds the two eigensolvers' spectrum difference, and \(\zeta_d\) covers entropy evaluation under the stated function-error assumption. The weights total 880. No empirical purity-asymmetry floor, cancellation or fitted allowance enters this bound.

Let \(\widehat\Delta\) be the exact weighted arithmetic of the saved complement-minus-held readings and \(p\) the first route's printed excess. The reported gap upper includes its separate decimal-display uncertainty: \(G=|\widehat\Delta-p|+0.5\times10^{-9}\). Each row satisfies \(G\le1.85\times10^{-7}+U_\Delta\), with the first term supplied by the source [S]. All entries are nats; upper columns are rounded upward, while displayed readings are rounded for presentation.

| \(T/r_0\) | Shift / interior sites | Own \(\widehat\Delta\) [N] | Printed first excess [S] | Gap upper \(G\) [N, S] | Conditional own \(U_\Delta\) [R, N] |
|---:|---:|---:|---:|---:|---:|
| 12, control | 0 / 58 | 285.10112677308 | 285.101126742 | \(3.157690810\times10^{-8}\) | 0.006743435819 |
| 15 | +1 / 51 | 109.35586072133 | 109.355860741 | \(2.016616839\times10^{-8}\) | 0.006297179483 |
| 20 | +1 / 38 | 296.51013200938 | 296.510131979 | \(3.087574155\times10^{-8}\) | 0.005376246603 |
| 28 | +1 / 37 | 297.71584732641 | 297.715847296 | \(3.091492189\times10^{-8}\) | 0.005300936657 |
| 35 | +1 / 37 | 299.44353886812 | 299.443538883 | \(1.537948210\times10^{-8}\) | 0.005300725707 |
| 40 | +1 / 36 | 310.88840041701 | 310.888400433 | \(1.649115047\times10^{-8}\) | 0.005223084271 |

The [complete companion](<internal>/EVOLVEDCUT_purity_bound_channels_v1.md) lists every per-channel state bound \(U_C\), held-reference bound \(U_H\), their sum and weighted term, and all six separate weighted bounds. The weighted state-entropy uppers range approximately from \(0.00284\) to \(0.00367\) nats; held-reference uppers range from \(0.00238\) to \(0.00308\) nats. Their size is dominated by conservative formation/isometry envelopes, not the observed route discrepancy.

All sixty rows satisfy the fixed positivity, dimension, occupation and held-gap conditions. The scalar evaluator passed fourteen manufactured controls and the sixty frozen source bindings. It used exact rational representations of saved binary64 inputs and eighty-digit outward interval arithmetic on one CPU; it reran no eigensolver, covariance or entropy trace. The derivation and its sole independent review were frozen before evaluating the bound or comparison verdict. The new evaluator's arithmetic enclosure does **not** establish that the original BLAS/norm/scaling, overflow/underflow, square-root, `log`/`log1p`, and operator-entry assumptions hold for the old execution. Those implementation hypotheses remain explicit, and every `certified_from_saved_record_upper` and unconditional comparison verdict remains null. The source's \(1.85\times10^{-7}\) allowance also remains supplied rather than independently certified here.

The earlier direct-exterior readings still fail their original diagnostic criterion; this complement evaluation does not rewrite that result. The comparison does not newly verify byte identity of the two routes' column/orthonormalization inputs. No conclusion is supplied for unmatched T16/24/25/30 partitions, width eight, the excitation-channel clause, raw-state/history identification, continued `R2` reference, omitted angular tail, genuine three-dimensional reference remainder, massive sphere coefficient or an entropy-match sentence. The limited incidental source exposure is recorded precisely in the frozen specification and is not used.

SOURCES:

- `<internal>/icloud.md`, request 125; `<internal>/ReviewA_evcut_routes_v1.md`, section (1), within the disclosed exposure limit [S].
- [Pinned first-route table](<internal>/EVOLVEDCUT_route_compare_v1.md), only the six authorized matched printed excesses and supplied allowance [S].
- [Frozen bound specification](<internal>/EVOLVEDCUT_purity_bound_spec_v1.md), equations (1)–(11), targets, arithmetic hypotheses and exact source paths [R].
- [Complete inputs](<internal>/matched_input_metadata.json), [controls](<internal>/known_answers.json), and [complete results](<internal>/results.json), with exact source/code/spec hashes and all intermediate intervals [N]. Results SHA256: `2d895fa0ca5a9326dc6339bb7847ce2216f95087e4f7095eb73527a8e7ff8431`.
- [Bound evaluator](<internal>/run_EVOLVEDCUT_purity_bound_v1.py), [outward arithmetic](<internal>/decimal_interval_v1.py), and [original producer](<internal>/run_EVOLVEDCUT_secondroute_v1.py) [N].
- [Existing sole review](<internal>/redteam_EVOLVEDCUT_secondroute_v1.md), request-125 independent derivation, recipe, implementation and actual-output dispositions [R, N].

Tags: [R] derived conditional bound; [N] own saved-data arithmetic; [S] supplied input or allowance.
