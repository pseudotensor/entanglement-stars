# Evolved-cut exterior entropy error statement

Scope [N, R, S]: the six matched partitions only—T12 at the charge fallback, and T15/T20/T28/T35/T40 at the saved \(+1\)-site partitions—with \(\kappa=1,\ldots,10\) and multiplicity \(16\kappa\). The comparison is between the first route's declared Löwdin pure state [S] and our explicit polar control [N]. Our primary applied-root results remain unchanged.

**Our exterior entropy has a measured empirical resolution scale, but no certified upper error bound.** Define from the frozen polar fields
\[
\delta_\kappa=\widehat S_X(\widehat C_\kappa)
 -\widehat S_{X^c}(\widehat C_\kappa),\qquad
E_2^{\rm emp}=\sum_{\kappa=1}^{10}16\kappa|\delta_\kappa|.
\]
All 60 defects are positive, ranging from \(2.1699\times10^{-9}\) to \(3.6495\times10^{-9}\) nats per channel. This is the measured pure-state consistency floor that the exterior precision statement must include. For a common ideal pure target, \(e_X-e_{X^c}=\delta_\kappa\), hence \(\max(|e_X|,|e_{X^c}|)\ge|\delta_\kappa|/2\) [R]. It does **not** prove \(|e_X|\le|\delta_\kappa|\): an exterior upper bound inferred from this identity needs a certified complement error; a separate direct error analysis could instead bound the exterior. The analogous weighted lower bound is \(E_2^{\rm emp}/2\) here because all signs agree.

The table gives held-subtracted weighted excesses. Route-one values and gaps are supplied [S]; route-two values and \(E_2^{\rm emp}\) are extracted from frozen own fields [N]. Gaps are shown at rounded source precision. All entries are nats.

| \(T/r_0\); shift; interior sites | Route-one excess | Route-two polar excess | Gap, route one minus two | \(E_2^{\rm emp}\) |
|---|---:|---:|---:|---:|
| 12; 0; 58, charge fallback | 285.101126742 | 285.101129312 | \(-2.57\times10^{-6}\) | \(2.53911\times10^{-6}\) |
| 15; +1; 51 | 109.355860741 | 109.355863275 | \(-2.53\times10^{-6}\) | \(2.55370\times10^{-6}\) |
| 20; +1; 38 | 296.510131979 | 296.510134632 | \(-2.65\times10^{-6}\) | \(2.62237\times10^{-6}\) |
| 28; +1; 37 | 297.715847296 | 297.715850023 | \(-2.73\times10^{-6}\) | \(2.69705\times10^{-6}\) |
| 35; +1; 37 | 299.443538883 | 299.443541371 | \(-2.49\times10^{-6}\) | \(2.50281\times10^{-6}\) |
| 40; +1; 36 | 310.888400433 | 310.888402919 | \(-2.49\times10^{-6}\) | \(2.50157\times10^{-6}\) |

The **original stated diagnostic criterion still fails**: the exterior discrepancies exceed the first route's quoted \(2.1\times10^{-10}\) per channel and the weighted excess gaps exceed \(1.85\times10^{-7}\), by about a factor 14 in the weighted test. Our original handoff supplied no entropy upper error; its eigenvalue tolerance and residual controls cannot be substituted for one.

Descriptively, all matched exterior-channel and weighted-excess discrepancies are no larger than \(|\delta_\kappa|+2.1\times10^{-10}\) and \(E_2^{\rm emp}+1.85\times10^{-7}\), respectively. This includes both paper endpoints, T15 and T40. The comparison source also reports complement-side channel differences at most \(2.450\times10^{-11}\), held/flat exterior differences at most \(1.990\times10^{-10}\), and weighted complement-side excess differences at most \(4.751\times10^{-9}\) [S]. The discrepancy is predominantly associated with our exterior asymmetry; the other differences are nonzero. The discrepancies lie on the observed asymmetry scale. Neither this scale nor the source's purity diagnostic is an independently calibrated error allowance, so these inequalities establish no within-error coverage and do not alter the failed original criterion. **Agreement within rigorous upper error bounds remains unproved.**

The eigenvalue admission policy \(\tau_{\rm eig}=64d\epsilon_{64}\max(1,\max|\lambda|)\), measured residuals and representative checks remain unchanged. Cut shifts measure placement sensitivity. The raw covariance is a separate object, admitted only at T12/T20/T28 among these six clocks; polar asymmetry does not bound its entropy error or restore rejected raw rows. Input-byte identity and physical history remain unverified, and continued-reference, three-dimensional, massive-normalization and omitted-channel limits are unchanged. No unmatched partition or area ratio is compared. Only subtraction and summation of saved scalar fields were performed; no entropy or eigensolver was rerun.

## SOURCES

- <internal>/icloud.md — entries #121--#122.
- <internal>/EVOLVEDCUT_route_compare_v1.md — pin of <internal>/EVOLVEDCUT_route_compare_v1.md, section five and the authorized matched definitions/tables/errors [S].
- [Frozen own fields](<internal>/results.json) and [all 60 matched diagnostics](<internal>/matched_purity_diagnostics.json) [N].
- <internal>/extract_EVOLVEDCUT_error_fields_v1.py — data-only extraction, with channel/weight/site and known-value checks.
- <internal>/EVOLVEDCUT_secondroute_spec_v1.md — unchanged numerical policy and the precise empirical-error definition [R].
