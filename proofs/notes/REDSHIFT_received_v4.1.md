# REDSHIFT_received_v4.1: v4 with the final checks' local points fixed

route2 lane builder (analyst), for inbox #59. The final checks of REDSHIFT_received_v4.md (md5 f526e514f21187706c64b03cde01f824, frozen, unchanged) are deliveries/lead_gaps/FableAPI_rzchk5_v1.md (not ready; three local items, no headline number affected) and deliveries/lead_gaps/AstraAPI_rzchk5_v1.md (eight points). v4.1 is v4 with exactly those points fixed: sentences narrowed or relabelled, row labels added, Table I1e completed, and the A2 event-level export supplied; nothing else changes. **No number changed, a headline or otherwise: verified** (the five revised scripts' json equal v4's, the A2 table's apart from the export's entry). It rules on no physics. The text is v4's with counted replacements, made by rz_v41_compose_v1.py. Composed 2026-10-22 07:55:59 UTC.

The lead (inbox #59): "Lead, 07:52 UTC. reviewer A item 1: 'a fixed late arrival changes by at most 0.047' is broader than Table V1v4 (the dense fields move the P64_s4 A1 headline arrival by 1.440 and P64_s2's by 0.237), so narrow it to the table. Deliver REDSHIFT_received_v4.1.md; no further full check is planned unless a fix changes a headline."

## v4.1 response (one row per point of the two final checks)

| id | the point (quoted) | what was done | where |
|---|---|---|---|
| F5-1 | §1 item 4, last sentence | The sentence is narrowed to its table: at the tested A1 and A2 headline events, under the variants other than the dense fields, a fixed late arrival changes by at most 0.047; over all matched arrivals the hold changes L_T by up to 3.782 (W2), over those with L_T ≥ 5 by -0.058 (P64_s2), and the dense fields change the A1 headline events by 1.440 (P64_s4) and 0.237 (P64_s2). §0's bullet (1) now excepts Tables V1v4 and V1b, which compare stored (early-release) passes. | §1 item 4; §0 |
| F5-2 | Table I1c and Table L1v3 | P64_s4_ext200's resolved rows carry '(e, c only: ray mask unavailable)' in Table I1c (and in I1e and I1's cross-member rows), and every one of its rows in Tables L1v3 and L2v3 does (rz_v4_head_v3, rz_v3_law_v2); response row A4-3.3's 'in every row' now holds. | §3 Tables I1, I1c, I1e; §6; §2 row A4-3.3 |
| F5-3 | Table I1e | Table I1e exports the (b) endpoints per clock on every member with comoving clocks: 157 rows over 18 members (v4: three members), W2, P64_s8, S4_128_s8, S4_128_s16 and both completions included; the citations of I1e now hold as written. | §3 Table I1e |
| A5-1 | False distinct-emission grouping. | Relabelled, as reviewer B allows without a new experiment: the groups are ARRIVAL CLUSTERS (the A rows of one class and clock with consecutive arrivals 1 a apart, whose emission times differ, by up to 0.39 a, S4_128_s16 A2.8) and the statistics over them cluster-based; the grouping and every number are unchanged (rz_v4_dispev_tables_v3). | §1 item 6; §0 (5); §9.0 Tables 9-H1v4 to 9-H3v4; §2 row F4-6.2 |
| A5-2 | Missing ray-mask qualifications remain. | As F5-2. | §3; §6 |
| A5-3 | Tail endpoints are only partially exported. | As F5-3; I1's caption sends the reader to I1e for the endpoints per clock, now on every member. | §3 Tables I1, I1e |
| A5-4 | Resolved-domain integration limits are inconsistent. | H3's caption now states the ends the code uses: Δτ = τ(t_end) − τ(T_rel) on the clock's stored worldline samples (step 0.5 a); resolved t_end = the last stored sample strictly before the sample at which the clock is first inside the first site, or the certified end if earlier (P64_s4 A1.0: domain end 11.2031 against first entry 11.2109, one sample earlier); unmasked t_end = the last stored sample at or before min(the clock's last finite sample, the certified end). The computation is unchanged. | §3 Table H3 caption |
| A5-5 | Fixed-emission censoring remains stronger than the demonstrated classification. | The statuses are named for what the comparison reads, each variant's sampled map (consecutive arrivals 1 a apart inside its certified interval): bracketed / label past its last sampled received emission of the clock / before its first / no adjacent bracket, with 'non-receipt not verified' (no direct event location, §11). | §1 item 4; §4 Table A2Xv2; §5 Tables V1v4, V2v4; §2 row A4-2.3; §11 |
| A5-6 | The summary re-broadens the 0.047 statistic. | As F5-1. | §1 item 4 |
| A5-7 | Blueshift robustness still exceeds the tested population. | The hold result is restricted to the 4 members with variant passes (P64_s8, S4_128_s16, S4_128_s8, W2) and the listed events; P64_s4_heldsrc and P64_s4_widened are named untested under the variants. | §1 item 9 |
| A5-8 | A2 repair/export claims overstate their coverage. | §0's bullet (1) excepts Tables V1v4 and V1b (stored, early-release passes, compared like with like); Table A2Xv2 is called a per-class aggregate of the event-matched comparison (§0, §2 rows A4-1.2 and A4-2.1, its caption), and the event level is supplied: out/v41_a2fix_v3/a2_events.csv.gz, 39331 stored A2 events with their fixed-arrival values in both passes and their fixed-emission status and changes. | §0; §2; §4 Table A2Xv2 |

## 0. Status

- **What is new in v4.** (1) The later-released comoving clocks (A2) are released exactly at T_rel (rz_rays_v4n; v3's passes released them one step early): every ray pass is rerun (17 histories; P64_s4_ext200's records are not locatable), every table built on the comoving clocks except Tables V1v4 and V1b is recomputed from the exact-release passes (the propagation variants exist only as stored passes, which release A2 one step early, so V1v4 and V1b compare stored main and stored variants, like with like), and Table A2Xv2 prints, per member and class, the largest signed event-matched changes against the stored implementation (an aggregate; every stored A2 event's changes are exported in REDSHIFT_received_v4.1_scripts/out/v41_a2fix_v3/a2_events.csv.gz). (2) The ray's smallest |r| includes the emission endpoint and each step's cubic-Hermite minimum, so it never exceeds r_e. (3) H3's transfer diagnostics and received shares and H4's '> 1' counts are printed on the resolved domain beside the unmasked one; the (b) tails' endpoints are exported per clock on every member with comoving clocks (Table I1e). (4) Table HXv4 reconciles the heavy channel's crossings with Table 4.4-8's received counts. (5) The dispersive rays at the headline events are rerun from the exact-release events and counted per arrival cluster (consecutive arrival samples of one clock, whose emission times differ), 'earlier' and 'less redshifted' apart, every exception listed (Tables 9-H1v4 to 9-H3v4). (6) The variants tables use one sign convention (variant minus main), print every fixed-emission event's sampled-map status and the blueshift maxima's own change (Tables V1v4, V1b, V2v4). (7) The summary (sec. 1) states what both checks agree v3 establishes and narrows every other statement to its table.
- **Carried unchanged from REDSHIFT_received_v3.1** (pasted by marker lines; each carried caption line names its folder): Tables H9 and T1 (sec. 3; T1 is v1's tick check on v1's passes, whose A2 clocks are released one step early), S1c and S2c (sec. 7), M1, Y1, Y1h, Y2 and C1 (sec. 8), v2's forward-population dispersive pass with its heavy addendum (sec. 9.1, archival, its 'cap' sentences rewritten as AstraAPI_rzchk4 sec. 4.5 asks), and the production timing (sec. 10).
- **P64_s4_ext200**: records not locatable (registered path <internal>/out; the ext300 kit holds dangling symlinks to them), so its rows use its stored v1 pass only: its A2 clocks are released one step early and its ray mask (r) is unavailable, so its 'resolved' entries apply the emission and clock filters only (labelled so in every row); its records equal P64_s4's through T/r_s 40 (to 1.1e-8 over 0 to 40, deliveries/claude_busi/REDSHIFT_inputs_inventory_v1.md); its whole-window value 117.420 (Table H6, arriving at T/r_s 200.000) is unreproducible. Nothing in this file rests on it beyond that identity.
- **P64_s4_B2**: its whole-window value 49.833 (Table H6) arrives at T/r_s 100.000, past its admitted interval (to 40): uncertified, not a headline. Its Table H1 values inside the interval agree with P64_s4's to 9.4e-06 (largest difference over the classes).
- **'Inside the solver-certified record interval'** means the solve's own stop rule held on the records; it certifies neither the background's physical continuation nor interpolation support, field accuracy or geometric-optics validity (AstraAPI_rzchk4 sec. 1.4).
- **Caption lines**: 'Producing script: code/… Table file: out/…' without a folder note refers to deliveries/route2/REDSHIFT_received_v4.1_scripts/ (the five tables whose scripts v4.1 revised for labels); lines carried from v4 or earlier name their folder in brackets. The tables of sec. 3 to 9 are pasted verbatim from the table files named.
- The completion line is the last line of this file.

## 1. Summary

**What v3 establishes, as both checks agree (inbox #58), and v4 keeps.** Large received redshifts at a static receiver, within the solver-certified window, on the long-wave (continuum) map: ln(1/g_D) = L_D = 10.338 (e^10.3) on the reference collapse P64_s4 and 15.569 (e^15.6) at half its averaging width, P64_s2 (Table H1; comoving clocks with r_e ≥ 1 a); dependent at order one on the averaging width, the step, the propagation background and the interior completion (item 2); and rays of finite lattice momentum launched at the same events stop near ln(2/(ak)²) (item 6). The items below narrow every further number to its table.

1. **The headline per member** (Table H1; L_T = ln(1/G_D) / L_D = ln(1/g_D) of the same event, class in brackets; inside the solver-certified record interval; maxima over the sampled arrivals, 1 a apart, of the stored populations): P64_s4 10.456 / 10.338 (A1 r_e≥1a); P64_s2 15.676 / 15.569 (A1 r_e≥1a); P64_s8 3.708 / 3.582 (B); P64_s4_tau4 7.184 / 7.059 (A2); W2 4.877 / 4.741 (B); S4_128_s8 7.754 / 7.632 (A2); S4_128_s16 3.654 / 3.527 (B); P64_s4_heldsrc 3.803 / 3.681 (B); P64_s4_widened 4.239 / 4.111 (B). The A2 values are those of clocks released exactly at T_rel (Table A2Xv2 prints the change from v3's). The masks (emission at r_e ≥ 1 a, the clock outside the first site at emission, the ray's smallest |r| ≥ 1 a with the emission endpoint included) remove none of these 9 events. 4 of the seven long-window headlines (P64_s4, P64_s2, P64_s4_tau4, S4_128_s8) arrive at T_D/r_s = 40.0000, the window's last sample: window-censored maxima of fronts still hovering, not asymptotic values. The two largest (P64_s4 and P64_s2) are emitted before the first root forms (Table H2, 'pre-birth'). P64_s4_B2 and P64_s4_ext200 are not headlines (sec. 0).
2. **Order-one dependence on width, step, background and completion** (Tables H1, V1v4, V2v4). Width: the headline rises from 3.708 (B) to 10.456 to 15.676 as σ/a falls 8 → 4 → 2 at r_s/a 64; at r_s/a 128 it is 7.754 at σ/a 8 and 3.654 at σ/a 16. Step: it falls from 10.456 to 7.184 (A2) when the step is halved (τ 8 → 4); the finest step (τ 2) is certified only to T/r_s 12.75, inside which its largest L_T is 0.745. The conserving step (W2) gives 4.877 (B). Background: the dense fields of the same solve (a different propagation background, not an interpolation) change L_T at the A1 / A2 headline events by 1.440 / 1.433 on P64_s4 and 2.322 / 1.071 on P64_s4_tau4 (dense minus main, Table V1v4 on the stored passes). Completion: the held-source and widened completions agree with the reference on the 2785 matched flights ending before their branch at T/r_s 11.125 to the printed precision (largest change 0.000; not a statement about later flights) and give 3.460 / 3.708 for A1 with r_e ≥ 1 a, 3.460 / 4.050 for the A1 and A2 clocks and 3.803 / 4.239 over A1, A2 and B, against the reference's 10.456 (Table V2v4); the completions are untested under the propagation variants.
3. **Integration** (Tables R1v4, A2Xv2). On the exact-release clocks the step ladder's h-against-h/2 differences at fixed arrival are at most 9.06e-09 on A1 with r_e ≥ 1 a (observed order 2.23 to 5.48), 9.45e-10 on A2 (1.70 to 3.55) and 9.07e-09 on B (2.40 to 3.98); the independent integrator (DOP853, restarted at every record, clocks released at T_rel) agrees with h/4 to 5.96e-08 (A1), 1.54e-08 (A2) and 1.01e-07 (B) on 106 events over 16 histories (48 A1, 30 A2, 28 B), 0 capped; P64_s4_B2 and P64_s4_ext200 are not separately integrated. On P64_s2's A1 headline the difference is 5.96e-08. These are same-reconstruction cross-checks, not bounds on the background or on the physical signal. Against v3's stored passes, which released the A2 clocks one step early, the exact release changes A2's L_T at fixed arrival by at most 0.000334 to 0.00749 per member and its emission times by 0.194 to 0.717 a; at a fixed emission label the largest changes of a stored A2 event are ΔT_D -512.421 a and ΔL_T -6.554 (P64_s2; Table A2Xv2). v3's 'agrees with h/4 to 0.001' on A2 is withdrawn: it described the early-release clocks.
4. **Propagation variants, at fixed arrival and at fixed emission label** (Table V1v4, on the stored passes, main and variants alike releasing A2 one step early; every change signed, variant minus main). At fixed arrival the largest change at the tested A1 and A2 headline events, under the variants other than the dense fields (item 2), is -0.047 (P64_s2, app_hold_cubic, A1 r_e>=1a); B is compared by launch label, not at fixed arrival: under the production hold the largest change of a matched B launch, per member, is P64_s2 -5.093 (3355 matched; 13 not received); P64_s4 -2.002 (2916 matched; 5 not received); P64_s4_tau4 -0.692 (6783 matched; 4 not received); P64_s8 0.171 (3832 matched; 1 not received); S4_128_s16 0.113 (10016 matched; 3 not received); S4_128_s8 -1.267 (7235 matched; 0 not received); W2 0.352 (4299 matched; 7 not received). At a fixed emission label (clock, τ_e) the largest change at a headline event is -12.371 (P64_s2, app_hold_cubic, A1 r_e>=1a), its arrival moving by -992.640 a, and on that variant's sampled map the main run's events are 16277 bracketed / 1101 past its last sampled received emission of the clock / 5 before its first / 0 without an adjacent bracket (sampled-map statuses; non-receipt is not verified). At the tested A1 and A2 headline events, under the variants other than the dense fields, a fixed late arrival changes by at most 0.047 while a fixed emission label's arrival moves by hundreds of a; over all matched arrivals the production hold changes L_T by up to 3.782 (W2), over those with L_T ≥ 5 by -0.058 (P64_s2), and the dense fields change the A1 headline events by 1.440 (P64_s4) and 0.237 (P64_s2) (item 2): an empirical statement about these inverse maps, not a conditioning bound.
5. **Field sensitivity** (Table R2v4; NOT an error bound: the field error is UNBOUNDED, no bound on the background fields is established). Per 10⁻³ of the three field models, L_T at the A headline events changes by 0.002 to 0.393 (largest model per member and class; no perturbed A event lost); on the B events by up to 3.653 with the launch event held fixed (P64_s2, 37 of 120 perturbed launches lost there) and 5.476 with the launch relabelled to the perturbed fields' own root (P64_s2).
6. **Dispersive rays at the headline events** (Tables 9-H1v4 to 9-H3v4; massless radial channel; rays only: no packet is run, so no finite-band received spectrum or transmitted weight is claimed; selected ray diagnostics at the headline events, not extrema over all launches). Launched at the exact-release headline events, the dispersive rays arrive earlier than the stored continuum arrival on every received (event, ap_e) of the nine members. On P64_s4, P64_s2, P64_s4_tau4, S4_128_s8 they arrive less redshifted on every event, with median L_F/L_T over arrival clusters (cluster-weighted) 0.306 to 0.907 (a ratio of logarithms, not a frequency-reduction factor); their largest L_F is 6.722, 5.513, 4.663 and 3.847 at ap_e 0.05, 0.1, 0.2 and 0.3, against reviewer B's crossover estimate ln(2/ap_e²) = 6.68, 5.30, 3.91 and 3.10, which 56 (row, ap_e) pairs exceed: 44 (arrival cluster, ap_e) pairs over 16 arrival clusters (an A cluster is the five arrival samples, 1 a apart, of one clock, whose emission times differ by up to 0.39 a; each B launch is its own cluster). On the other five members (W2, P64_s8, S4_128_s16, P64_s4_heldsrc, P64_s4_widened) the median L_F/L_T is 0.578 to 0.999, and on P64_s4_heldsrc, P64_s8, S4_128_s16, W2 the dispersive ray of the same event can be more redshifted, by up to P64_s4_heldsrc 0.155; P64_s8 0.069; S4_128_s16 0.005; W2 0.053 (Table 9-H3v4, all B launches); on P64_s4_heldsrc the dispersive maximum at ap_e ≤ 0.1 (3.810) exceeds the member's continuum headline (3.803). The static receiver's L_gF = L_F + ln A_D and L_gK = L_K + ln A_D are printed apart (two emitter conventions).
7. **Selections, not bounds** (Tables I1, I1c, I1e). In selection (a) some member delivers a blueshifted received event (smallest L_T over the members: A1 -4.305 on W2, A2 -0.586 on S4_128_s16), so no uniform redshift bound holds across the members. The (b) tails are outcome-relative: each ends at its own population's last eligible received emission (Table I1e), and the masks move that endpoint, so a (b) value describes the selection, not the history, and none is quoted here. Of the B launches within 1 a outside a standing root on the three named, post-hoc-selected 64-site histories P64_s4, P64_s2, P64_s4_tau4, 68 percent, 54 percent, 63 percent were not received inside the interval (received after its end / not received by the last stored record: P64_s4 0 / 1949 of 2884; P64_s2 0 / 1687 of 3136; P64_s4_tau4 0 / 3026 of 4816); the received ones have L_T ≥ 2.241 (Table I1c).
8. **The fits, per member** (Tables L1v3, L2v3 recomputed on the exact-release passes; L_T and L_D give the same out-of-fit winners). K_H + c on P64_s4 (7 of 10 clocks), P64_s4_tau4 (8 of 10 clocks), W2 (9 of 9 clocks), S4_128_s8 (7 of 10 clocks), P64_s4_widened (7 of 9 clocks); Q + c on P64_s2 (9 of 10 clocks); the constant on P64_s8 (7 of 7 clocks), S4_128_s16 (6 of 7 clocks), P64_s4_heldsrc (6 of 7 clocks). No receiver-time law is reported (the zone exit is read on the step grid).
9. **Blueshifts** (Tables H4, V1b). Blueshifted received events of the normal-comoving classes (A1, A2, B) occur on P64_s8, W2, S4_128_s8, S4_128_s16, P64_s4_heldsrc, P64_s4_widened; the dust emitters (C) add P64_s4, P64_s2, P64_s4_tau4, M3c_s4_96, P64_s4_B2, P64_s4_ext200. On the 4 members with variant passes (P64_s8, S4_128_s16, S4_128_s8, W2), the most blueshifted A1, A2 and B event changes under the production hold by: P64_s8 A1 -1.819 by 0.257; P64_s8 B -0.748 by -0.020; S4_128_s16 A1 -2.541 by 0.255; S4_128_s16 A2 -0.586 by 0.00553; S4_128_s16 B -0.629 by -0.00514; S4_128_s8 A1 -3.946 by 2.273; S4_128_s8 B -1.832 by 0.011; W2 A1 -4.305 by 3.739; W2 B -3.548 by -0.023 (signed, hold minus main); each of these listed events stays blueshifted at its arrival. Under the dense fields the A1 maxima change by 2.915 to 6.175, and on P64_s8, S4_128_s16, S4_128_s8, W2 they are no longer blueshifted. On these 4 members the listed events' existence under the hold is the robust statement; their size is not. P64_s4_heldsrc and P64_s4_widened, which also have blueshifted normal-comoving events, have no variant passes and are untested under the variants. No cause (caustic or otherwise) is assigned; the backward map shows no monotonicity reversal on any member.
10. **The limit, with its premises in the same sentence.** Arbitrarily large redshift at increasing size follows from the continuum construction only if (premises, none measured here) the near-root optical region persists over a delay of order r_s ln(r_s/a), its root moves by less than the launch offset δ, the averaging, evolution, reconstruction and field errors are small on that offset, and the signal regime and the exterior transfer stay controlled; it is stated at fixed q = ω r_s, which is not a fixed physical frequency as r_s grows. At the computed sizes the completion spread (item 2) shows the records do not determine persistence. (v3.1 sec. 1 item 9, its cross-reference renumbered to this list.)
11. **The heavy channel's counts reconciled** (Table HXv4). Table HX counted a launch when its first crossing of 4 r_s lies inside the interval; Table 4.4-8 counts it when, in addition, ln(1/Z_F) is finite. Over the 56 (member, ap_e) rows, 1563 crossings differ, all of them dust (C) launches whose D_e is not positive (A 0, B 0, C 1563); HX's totals and 4.4-8's are both reproduced on every row.

## 2. Response table (every point of both v3 checks)

Status: fixed, recomputed (new numbers), relabelled, declined (reason). reviewer B's points are its section headings (A4-x.y, quoted); reviewer A's are its sec. 6 items (F4-6.n, titles quoted). v3's response rows A3-4.1 and A3-11.4 are corrected in rows F4-6.5 and F4-6.11.

| id | the point (quoted title) | status | where | what was done |
|---|---|---|---|---|
| A4-1.1 | The continuum observable is now correctly distinguished | no change asked | §3 Table H1 | L_T = ln(1/G_D) and L_D = ln(1/g_D) stay named as v3 named them. |
| A4-1.2 | The headline table is internally consistent—with an A2 qualification | fixed and recomputed | §0; §1 item 1; §3 Table H1; §4 Table A2Xv2 | The A2 clocks are released at T_rel exactly (rz_rays_v4n) and every comoving-clock table is recomputed; H1's A2 headlines are now P64_s4_tau4 7.184 / 7.059 and S4_128_s8 7.754 / 7.632; Table A2Xv2 prints the per-class aggregate of the signed event-matched changes (every event: a2_events.csv.gz, v4.1). The two largest A1 events (P64_s4 and P64_s2) are emitted before the first root forms, said in §1 item 1. |
| A4-1.3 | Completion populations are finally separated correctly | no change asked | §1 items 1, 2; §5 Table V2v4 | The completions' L_D maxima (3.681, 4.111) are printed beside their L_T maxima. |
| A4-1.4 | Certification and censoring must remain narrow labels | relabelled | §0; §3 Table H6 | Whole-window values past the admitted interval carry † with their arrival: P64_s4_B2 49.833 (T/r_s 100.000), P64_s4_ext200 117.420 (T/r_s 200.000, unreproducible); §0 says what 'inside the solver-certified record interval' does not certify; four of the seven long-window headlines are called window-censored. |
| A4-2.1 | The A2 release defect is explained, not fixed | fixed and recomputed | §0; §1 item 3; §4 Tables A2Xv2, R1v4; every comoving-clock table | rz_rays_v4n releases each A2 clock exactly at T_rel (the emitter loop also resets a clock at its release time); all ray passes rerun; H1–H8, I1, I1c, I1e, L1, L2, F2, R1–R3, V2 and 9-H1 to 9-H3 are recomputed from them. Table A2Xv2 prints, per member and class, the largest signed event-matched changes (exact minus stored; an aggregate, the event level in a2_events.csv.gz, v4.1) of T_e, τ_e, r_e, L_T and L_D at fixed arrival and of T_D, L_T and L_D at fixed emission label: ∣ΔL_T∣ at fixed arrival 0.000334 to 0.00749 per member, ∣ΔT_e∣ 0.194 to 0.717 a. On the exact-release clocks the A2 ladder's observed order is 1.70 to 3.55 (v3: 0.99 to 1.18) and DOP853 agrees with h/4 to 1.54e-08 (v3: 1.30e-03). |
| A4-2.2 | The new B and `P64_s2` integration coverage is genuine reported progress | relabelled | §1 item 3; §4 Table R1v4 caption | '106 events over 16 histories, 0 capped; P64_s4_B2 and P64_s4_ext200 are not separately integrated' (P64_s4_B2's Table H1 values agree with P64_s4's to 9.4e-06, P64_s4_ext200's records with P64_s4's to 1.1e-8 over T/r_s 0 to 40; nothing is counted as an independent rerun of them). |
| A4-2.3 | Fixed-emission coverage is still incomplete at crucial events | recomputed and relabelled; the direct location not run | §1 item 4; §4 Tables R1v4, A2Xv2; §5 Tables V1v4, V2v4 | On the exact-release clocks the five-arrival bracket holds for the A2 headline events: P64_s4 20 of 20 at h/2, 20 at h/4; P64_s2 20 of 20 at h/2, 20 at h/4; P64_s4_tau4 19 of 20 at h/2, 19 at h/4; S4_128_s8 20 of 20 at h/2, 20 at h/4. Every unbracketed event is counted by reason (label outside the window / map not monotone / too few arrivals). Every fixed-emission comparison prints the main events by sampled-map status (bracketed / label past the variant's last sampled received emission of the clock / before its first / no adjacent bracket; non-receipt not verified, v4.1), none merged into another; one sign convention (variant minus main) throughout. 'Well-conditioned' is replaced by an empirical statement about the tested maps. Not run: locating the common (clock, τ_e) event on each background and integrating from it (a new experiment; inbox #58). |
| A4-2.4 | The 0.047 number remains an A-event statistic | relabelled | §1 item 4 | -0.047 is the largest change at fixed arrival at the tested A1 and A2 headline events; B is compared by launch label and its changes are printed apart. |
| A4-2.5 | Field-error control remains absent | unchanged (declared) | §1 item 5; §11 | UNBOUNDED; R2v4 is a sensitivity on the surviving events and prints the events lost beside the maxima. |
| A4-3.1 | H3 does not contain the promised resolved counterpart | recomputed | §3 Tables H3, H4 | H3 prints both domains: unmasked (as v3) and resolved (masked events only, consecutive resolved arrivals 1 a apart, the domain ending at the clock's first entry into the first site, printed); H4's '> 1' counts on both; the received shares on each population's own domain. |
| A4-3.2 | The reported “ray’s smallest radius” is not always the full-ray minimum | recomputed; stencil audit not run | §3 Table H2; §11 | rz_rays_v4n records the minimum over the ray including the emission endpoint and each step's cubic-Hermite minimum; H2's smallest ∣r∣ never exceeds r_e (P64_s8 A1 sub-site: r_e 3.57e-04 a, smallest 3.57e-04 a). A stencil-by-stencil interpolation-support audit is not run (§11). |
| A4-3.3 | “Resolved” is still printed where a required mask is unavailable | relabelled | §0; §3 Tables H1, H2, H4, H6, I1, I1c | P64_s4_ext200's 'resolved' entries read 'e, c only: ray mask unavailable' in every row (Tables I1c, I1e, L1v3 and L2v3 included in v4.1); its H1 row carries the full note. |
| A4-3.4 | The resolved tail selection has changed its endpoint | relabelled and exported | §1 item 7; §3 Tables I1, I1e | The rule is stated in I1's caption (for each population separately, the tail ends at that population's last eligible received emission); Table I1e exports the endpoints per clock; the (a) shares are computed on each population's own domain. |
| A4-3.5 | The near-root censoring discrepancy is successfully explained | relabelled | §1 item 7; §3 Table I1c | 'never' reads 'not received by the last stored record'; 'the three deepest' reads 'the three named, post-hoc-selected 64-site histories'. |
| A4-3.6 | The new heavy cross-tabulation does not reproduce the old received counts | recomputed | §1 item 11; §9.2 Table HXv4 | Both predicates stated and evaluated on common launch ids: the 1563 crossings 4.4-8 does not count are all dust launches with D_e not positive; HX's and 4.4-8's totals are reproduced on every row. |
| A4-4.1 | The major population objection is repaired | recomputed | §9.0 | The headline events are now the exact-release implementation's (rz_rays_v4n), and 'exact emission events' is said of that implementation. |
| A4-4.2 | The new dispersive headline arithmetic checks out | recomputed | §9.0 Table 9-H2v4 | The arithmetic is reprinted on the exact-release events, per row and per arrival cluster. |
| A4-4.3 | “Earlier and less redshifted” is not universally true over the new nine-history population | fixed | §1 item 6; §9.0 Tables 9-H2v4, 9-H3v4 | 'Earlier' (on every received ray) and 'less redshifted' are counted apart; the exceptions (P64_s4_heldsrc 0.155; P64_s8 0.069; S4_128_s16 0.005; W2 0.053) are listed; L_F/L_T is labelled a ratio of logarithms. |
| A4-4.4 | Receiver normalization is improved; emitter convention still matters | fixed and relabelled | §9.0 Tables 9-H1v4, 9-H2v4; §9.1 | L_gK = L_K + ln A_D is printed beside L_gF; the archival C-pooled tables and '46 of 47' are labelled archival and convention-dependent, with AstraAPI_rzchk3 sec. 7.3's C-free count quoted beside them (37 of 37 A/B classwise reductions; a reduction count, not a cap). |
| A4-4.5 | Neither a universal cap nor packet validation has been established | fixed (text) | §9.1; §11 | The archival item 4's 'The deepest certified redshifts are capped' and 'show the same cap per class' are rewritten without 'cap'; packets are not computed (§11). |
| A4-5 | Coverage against both hostile checks | answered | this table | Each row of reviewer B's coverage table is answered by the A4 and F4 rows here. |
| A4-6 | What must be measured to establish the owner’s result | not computed (no new scope) | §11 | The predeclared experiment, field control, packets and scaling tests are not run (inbox #58: no new routes, packets or experiments). |
| A4-7, A4-8 | Exact replacement sentences the paper may print; Thirty-page requirement | declined (the lead's) | - | The paper's text and page count are the lead's. |
| F4-6.1 | §1 item 5. | fixed | §1 item 6 | Earlier on every received event; less redshifted on P64_s4, P64_s2, P64_s4_tau4, S4_128_s8 on every event; the exceptions with their sizes (P64_s4_heldsrc 0.155; P64_s8 0.069; S4_128_s16 0.005; W2 0.053) and P64_s4_heldsrc's dispersive maximum above its continuum headline are stated. |
| F4-6.2 | §1 item 5 and Table 9-H2. | fixed | §1 item 6; §9.0 Tables 9-H1v4, 9-H2v4 | Rows are grouped into arrival clusters (the A rows of one class and clock whose arrivals form one consecutive run on the 1 a grid, their emission times differing; each B row a launch): 56 (row, ap_e) pairs exceed the crossover, 44 (arrival cluster, ap_e) pairs over 16 arrival clusters; the median L_F/L_T over clusters is 0.306 to 0.907 on the four members, 0.578 to 0.999 on the other five. |
| F4-6.3 | §1 items 1 and 2. | fixed | §1 item 2 | The trends are stated in words: width 3.708 → 10.456 → 15.676 (σ/a 8 → 4 → 2), step 10.456 → 7.184 (τ 8 → 4), the finest step certified only to T/r_s 12.75. |
| F4-6.4 | §1 item 3 and Table R1v3. | recomputed (exact release) | §1 item 3; §4 Tables A2Xv2, R1v4 | The stored values are no longer bounded by the ladder: the clocks are released exactly and the change is measured (A4-2.1). '0.001' is withdrawn; on the exact-release clocks DOP853 agrees with h/4 to 1.54e-08 on A2. |
| F4-6.5 | Table H3, Table H4v3 last column, response row A3-4.1. | recomputed | §3 Tables H3, H4; this table | H3's two transfer diagnostics and shares and H4's '> 1' counts are printed on the resolved domain beside the unmasked one. v3's row A3-4.1 overstated: H3 carried no mask in v3; it does in v4. |
| F4-6.6 | §0, Table H1, H2, H6, L2v3, response rows. | relabelled | §0; §3 Tables H1, H2, H6; §6 note; rows A4-1.4, A4-3.3 | P64_s4_ext200: records not locatable; v1 pass only; identical to P64_s4 through T/r_s 40 (records to 1.1e-8); whole-window value 117.420 unreproducible. P64_s4_B2's 49.833 is marked uncertified past 40 where it is printed. |
| F4-6.7 | Table V1v3. | relabelled (caption chosen; no new runs) | §1 item 2; §3 note; §5 Table V2v4 | The completion headlines are captioned 'untested under the propagation variants'. |
| F4-6.8 | Table HX. | recomputed | §9.2 Table HXv4 | See A4-3.6. |
| F4-6.9 | §1 item 8. | recomputed | §1 item 9; §5 Table V1b | The blueshift maxima's own change under the hold is printed per member and class (A1, A2, B), and under the dense fields. |
| F4-6.10 | Table I1 (b). | relabelled | §1 item 7; §3 Table I1 caption | I1's caption says the masks move the endpoint (P64_s2's resolved A1 row ends earlier on its clocks), so a (b) value describes the selection; the summary quotes no (b) value. |
| F4-6.11 | §2 status column. | fixed | this table | A3-4.1 now reads: masks applied, H3 and H4 included (v4). A3-11.4 now reads: masks applied; no alternative centre treatment; no stencil-support audit. |

## 3. Headline and events (rz_v4_head_v3.py; out/v41_head_v3/head_tables.md, verbatim)

Producing script: code/rz_v4_head_v3.py (md5 d11191e596c9641fa89645406ba34449). Table file: out/v41_head_v3/head_tables.md (md5 5d94c9cfc5df0a8dc38ddeb4974eec5e). Launcher code/v41tables_run1.sh (md5 f6a2feea9246225a26c442330fc256bc). Made from v4's code/rz_v4_head_v2.py (REDSHIFT_received_v4_scripts, md5 b9933be6103dd0d5ab3ba04c8d7721bc) by code/mk_head_v4_v3_patch.py (md5 6803a16347eec1cbcdfc9c997e71cf21): the ray-mask label on Table I1c's, I1e's and I1's cross-member resolved rows, Table I1e on every member, H3's caption; its json equals v4's out/v4_head_v2/head_tables.json: yes. Data as v4 (the exact-release passes; P64_s4_ext200: its v1 pass).

Notes: the completions' rows (P64_s4_heldsrc, P64_s4_widened) are untested under the propagation variants (sec. 5); P64_s4_ext200 as sec. 0 says (records not locatable; v1 pass only; identical to P64_s4 through T/r_s 40, to 1.1e-8; whole-window value unreproducible); in Table I1 a (b) value describes its selection, whose endpoint the masks move, and the summary quotes none.

**Table H1. The headline, renamed and split** (rz_v4_head_v3.py on the exact-release passes rz_rays_v4n and rz_rays_v4nc: v3's columns, with the later-released clocks (A2) released at T_rel exactly, Table A2Xv2 printing the change; P64_s4_ext200 is the exception named in its row). At R_D = 4 r_s, over arrivals inside the solver-certified record interval. In each 'a / b' cell, a is the largest L_T = ln(1/G_D) of the class (G_D = dτ_e/dT_D, the coordinate-clock ratio) and b is L_D = ln(1/g_D) = L_T + ln A_D of THE SAME EVENT (g_D the static receiver's frequency ratio, A_D = √(N_D² − v_D²) at the arrival); the 'max L_D' columns are L_D maximised independently over the same population. 'resolved' applies the masks (e) r_e ≥ 1 a, (c) the clock has not entered the first site by the emission, (r) the ray's smallest |r| ≥ 1 a.

| member | inside the solver-certified record interval to T/r_s | A1, r_e ≥ 1 a: L_T / L_D same event; max L_D | A1 resolved: L_T / L_D; max L_D | A1 sub-site: L_T / L_D | A2: L_T / L_D; max L_D | A2 resolved: L_T / L_D; max L_D | B: L_T / L_D; max L_D | B resolved: L_T / L_D; max L_D | C: L_T / L_D; max L_D |
|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | 40.0000 | 10.456 / 10.338; 10.338 | 10.456 / 10.338; 10.338 | - | 9.452 / 9.335; 9.335 | 9.452 / 9.335; 9.335 | 9.128 / 9.010; 9.010 | 9.128 / 9.010; 9.010 | 4.283 / 4.149; 4.149 |
| P64_s2 | 40.0000 | 15.676 / 15.569; 15.569 | 15.676 / 15.569; 15.569 | 15.385 / 15.278 | 14.395 / 14.288; 14.288 | 14.395 / 14.288; 14.288 | 10.033 / 9.919; 9.919 | 10.033 / 9.919; 9.919 | 3.132 / 3.025; 3.025 |
| P64_s8 | 40.0000 | 3.241 / 3.115; 3.115 | 3.241 / 3.115; 3.115 | 2.288 / 2.161 | 3.172 / 3.045; 3.045 | 3.172 / 3.045; 3.045 | 3.708 / 3.582; 3.582 | 3.708 / 3.582; 3.582 | 3.099 / 2.973; 2.973 |
| P64_s4_tau4 | 40.0000 | 6.107 / 5.982; 5.982 | 6.107 / 5.982; 5.982 | - | 7.184 / 7.059; 7.059 | 7.184 / 7.059; 7.059 | 6.596 / 6.469; 6.469 | 6.596 / 6.469; 6.469 | 4.835 / 4.707; 4.707 |
| P64_s4_tau2 | 12.7500 | 0.745 / 0.617; 0.617 | 0.745 / 0.617; 0.617 | - | - | - | - | - | 0.622 / 0.493; 0.493 |
| P64_s4_newton | 12.5000 | 0.724 / 0.598; 0.598 | 0.724 / 0.598; 0.598 | - | - | - | - | - | 0.590 / 0.462; 0.462 |
| W2 | 40.0000 | 3.517 / 3.361; 3.361 | 3.517 / 3.361; 3.361 | 4.081 / 3.944 | 4.543 / 4.406; 4.406 | 4.543 / 4.406; 4.406 | 4.877 / 4.741; 4.741 | 4.877 / 4.741; 4.741 | 4.461 / 4.305; 4.305 |
| ISO_64_s4 | 13.5000 | 0.720 / 0.600; 0.600 | 0.720 / 0.600; 0.600 | - | - | - | 0.766 / 0.646; 0.646 | 0.766 / 0.646; 0.646 | 0.629 / 0.509; 0.509 |
| S4_128_s8 | 40.0000 | 3.434 / 3.290; 3.290 | 3.434 / 3.290; 3.290 | 6.699 / 6.577 | 7.754 / 7.632; 7.632 | 7.754 / 7.632; 7.632 | 7.723 / 7.601; 7.601 | 7.723 / 7.601; 7.601 | 4.685 / 4.548; 4.548 |
| S4_128_s16 | 40.0000 | 3.285 / 3.158; 3.158 | 3.285 / 3.158; 3.158 | 2.292 / 2.165 | 2.782 / 2.654; 2.654 | 2.782 / 2.654; 2.654 | 3.654 / 3.527; 3.527 | 3.654 / 3.527; 3.527 | 3.622 / 3.495; 3.495 |
| S4_128_s4 | 3.5625 | 0.701 / 0.571; 0.571 | 0.701 / 0.571; 0.571 | - | - | - | - | - | 0.240 / 0.110; 0.110 |
| ISO_128_s4 | 12.0000 | 0.703 / 0.572; 0.572 | 0.703 / 0.572; 0.572 | - | - | - | - | - | 0.593 / 0.467; 0.467 |
| ISO_128_s4_plain | 3.2500 | 0.703 / 0.572; 0.572 | 0.703 / 0.572; 0.572 | - | - | - | - | - | 0.234 / 0.104; 0.104 |
| M3c_s4_96 | 22.0833 | 0.716 / 0.587; 0.587 | 0.716 / 0.587; 0.587 | - | 0.922 / 0.792; 0.793 | 0.922 / 0.792; 0.793 | - | - | 0.721 / 0.591; 0.591 |
| P64_s4_B2 | 40.0000 | 10.456 / 10.338; 10.338 | 10.456 / 10.338; 10.338 | - | 9.452 / 9.335; 9.335 | 9.452 / 9.335; 9.335 | 9.128 / 9.010; 9.010 | 9.128 / 9.010; 9.010 | 4.283 / 4.149; 4.149 |
| P64_s4_ext200 (records not locatable: its v1 pass, whose A2 clocks are released one step early; ray mask (r) unavailable, so its 'resolved' columns apply the emission and clock filters only; its records equal P64_s4's through T/r_s 40; its whole-window value is unreproducible) | 40.0000 | 10.456 / 10.338; 10.338 | 10.456 / 10.338; 10.338 | - | 9.457 / 9.340; 9.340 | 9.457 / 9.340; 9.340 | 9.128 / 9.010; 9.010 | 9.128 / 9.010; 9.010 | 4.283 / 4.149; 4.149 |
| P64_s4_heldsrc | 40.0000 | 3.460 / 3.340; 3.340 | 3.460 / 3.340; 3.340 | 0.455 / 0.311 | 2.787 / 2.668; 2.668 | 2.787 / 2.668; 2.668 | 3.803 / 3.681; 3.681 | 3.803 / 3.681; 3.681 | 3.477 / 3.339; 3.339 |
| P64_s4_widened | 40.0000 | 3.708 / 3.583; 3.583 | 3.708 / 3.583; 3.583 | 3.092 / 2.964 | 4.050 / 3.922; 3.922 | 4.050 / 3.922; 3.922 | 4.239 / 4.111; 4.111 | 4.239 / 4.111; 4.111 | 3.925 / 3.800; 3.800 |

**Table H2. The headline events** (as v2, with the masks): per member and class, the event that sets the class's largest L_T in Table H1 (unmasked classes of v2 and the resolved ones): clock (A: index; B: offset δ in a; C: quantile), emission T_e, proper time τ_e (A only), radius r_e, arrival T_D, I = ∫∂_r c dT along the ray, c_D, ln A_D, D_e, the ray's smallest |r| (a), and the masks: post-gate share of the flight (one-term gate), pre-birth emission, sub-site emission (e), clock inside the first site before the emission (c).

| member | class | clock | T_e/r_s | τ_e/r_s | r_e (a) | T_D/r_s | I | c_D | ln A_D | D_e | ray's smallest abs r (a) | post-gate share (one-term gate) | pre-birth | sub-site emission | clock inside the first site before it |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | A1 r_e≥1a | 5 | 10.5356 | 9.5445 | 19.057 | 40.0000 | 9.910 | 0.5793 | -0.1172 | 1.0000 | 18.902 | 0.959 | yes | no | no |
| P64_s4 | A1 resolved | 5 | 10.5356 | 9.5445 | 19.057 | 40.0000 | 9.910 | 0.5793 | -0.1172 | 1.0000 | 18.902 | 0.959 | yes | no | no |
| P64_s4 | A2 | 6 | 14.3532 | 0.3491 | 42.236 | 40.0000 | 8.906 | 0.5793 | -0.1172 | 1.0000 | 34.698 | 1.000 | no | no | no |
| P64_s4 | A2 resolved | 6 | 14.3532 | 0.3491 | 42.236 | 40.0000 | 8.906 | 0.5793 | -0.1172 | 1.0000 | 34.698 | 1.000 | no | no | no |
| P64_s4 | B | δ 0.100 | 17.6250 | - | 35.350 | 39.7464 | 8.581 | 0.5790 | -0.1175 | 1.0000 | 35.350 | 1.000 | no | no | no |
| P64_s4 | B resolved | δ 0.100 | 17.6250 | - | 35.350 | 39.7464 | 8.581 | 0.5790 | -0.1175 | 1.0000 | 35.350 | 1.000 | no | no | no |
| P64_s4 | C | R_50 | 11.5000 | - | 42.859 | 26.6804 | 3.983 | 0.5556 | -0.1341 | 0.7500 | 42.859 | 0.984 | yes | no | no |
| P64_s4 | C resolved | R_50 | 11.5000 | - | 42.859 | 26.6804 | 3.983 | 0.5556 | -0.1341 | 0.7500 | 42.859 | 0.984 | yes | no | no |
| P64_s2 | A1 r_e≥1a | 5 | 10.4793 | 9.4950 | 18.742 | 40.0000 | 15.161 | 0.5973 | -0.1072 | 1.0000 | 18.646 | 0.940 | yes | no | no |
| P64_s2 | A1 resolved | 5 | 10.4793 | 9.4950 | 18.742 | 40.0000 | 15.161 | 0.5973 | -0.1072 | 1.0000 | 18.646 | 0.940 | yes | no | no |
| P64_s2 | A1 sub-site | 0 | 10.0425 | 8.2974 | 0.463 | 40.0000 | 14.870 | 0.5973 | -0.1072 | 1.0000 | 0.289 | 0.926 | yes | yes | yes |
| P64_s2 | A2 | 6 | 14.4128 | 0.4060 | 39.469 | 40.0000 | 13.880 | 0.5973 | -0.1072 | 1.0000 | 31.286 | 1.000 | no | no | no |
| P64_s2 | A2 resolved | 6 | 14.4128 | 0.4060 | 39.469 | 40.0000 | 13.880 | 0.5973 | -0.1072 | 1.0000 | 31.286 | 1.000 | no | no | no |
| P64_s2 | B | δ -1.000 | 11.5000 | - | 40.000 | 32.6345 | 9.505 | 0.5896 | -0.1143 | 1.0000 | 32.261 | 0.965 | yes | no | no |
| P64_s2 | B resolved | δ -1.000 | 11.5000 | - | 40.000 | 32.6345 | 9.505 | 0.5896 | -0.1143 | 1.0000 | 32.261 | 0.965 | yes | no | no |
| P64_s2 | C | R_50 | 11.3750 | - | 43.668 | 22.8122 | 2.818 | 0.5639 | -0.1069 | 0.7723 | 43.668 | 0.923 | yes | no | no |
| P64_s2 | C resolved | R_50 | 11.3750 | - | 43.668 | 22.8122 | 2.818 | 0.5639 | -0.1069 | 0.7723 | 43.668 | 0.923 | yes | no | no |
| P64_s8 | A1 r_e≥1a | 5 | 10.7669 | 9.7412 | 17.632 | 25.5781 | 2.603 | 0.5286 | -0.1259 | 1.0000 | 17.586 | 0.993 | yes | no | no |
| P64_s8 | A1 resolved | 5 | 10.7669 | 9.7412 | 17.632 | 25.5781 | 2.603 | 0.5286 | -0.1259 | 1.0000 | 17.586 | 0.993 | yes | no | no |
| P64_s8 | A1 sub-site | 5 | 23.6076 | 14.8356 | 3.57e-04 | 38.7188 | 1.696 | 0.5533 | -0.1264 | 1.0000 | 3.57e-04 | 1.000 | no | yes | yes |
| P64_s8 | A2 | 9 | 26.4830 | 0.4303 | 40.083 | 38.8750 | 2.580 | 0.5534 | -0.1262 | 1.0000 | 40.055 | 1.000 | no | no | no |
| P64_s8 | A2 resolved | 9 | 26.4830 | 0.4303 | 40.083 | 38.8750 | 2.580 | 0.5534 | -0.1262 | 1.0000 | 40.055 | 1.000 | no | no | no |
| P64_s8 | B | δ 2.142 | 12.2500 | - | 48.142 | 25.5764 | 3.070 | 0.5286 | -0.1259 | 1.0000 | 48.142 | 1.000 | yes | no | no |
| P64_s8 | B resolved | δ 2.142 | 12.2500 | - | 48.142 | 25.5764 | 3.070 | 0.5286 | -0.1259 | 1.0000 | 48.142 | 1.000 | yes | no | no |
| P64_s8 | C | R_90 | 13.0000 | - | 56.232 | 23.8990 | 2.396 | 0.5283 | -0.1260 | 1.0672 | 56.232 | 1.000 | no | no | no |
| P64_s8 | C resolved | R_90 | 13.0000 | - | 56.232 | 23.8990 | 2.396 | 0.5283 | -0.1260 | 1.0672 | 56.232 | 1.000 | no | no | no |
| P64_s4_tau4 | A1 r_e≥1a | 5 | 10.6971 | 9.6684 | 15.911 | 40.0000 | 5.558 | 0.5773 | -0.1248 | 1.0000 | 12.826 | 0.968 | yes | no | no |
| P64_s4_tau4 | A1 resolved | 5 | 10.6971 | 9.6684 | 15.911 | 40.0000 | 5.558 | 0.5773 | -0.1248 | 1.0000 | 12.826 | 0.968 | yes | no | no |
| P64_s4_tau4 | A2 | 7 | 18.8215 | 0.5895 | 31.102 | 40.0000 | 6.634 | 0.5773 | -0.1248 | 1.0000 | 31.096 | 1.000 | no | no | no |
| P64_s4_tau4 | A2 resolved | 7 | 18.8215 | 0.5895 | 31.102 | 40.0000 | 6.634 | 0.5773 | -0.1248 | 1.0000 | 31.096 | 1.000 | no | no | no |
| P64_s4_tau4 | B | δ 0.462 | 19.5000 | - | 34.212 | 38.7799 | 6.043 | 0.5756 | -0.1264 | 1.0000 | 33.003 | 1.000 | no | no | no |
| P64_s4_tau4 | B resolved | δ 0.462 | 19.5000 | - | 34.212 | 38.7799 | 6.043 | 0.5756 | -0.1264 | 1.0000 | 33.003 | 1.000 | no | no | no |
| P64_s4_tau4 | C | R_50 | 19.5000 | - | 34.454 | 36.9077 | 5.355 | 0.5722 | -0.1283 | 0.3400 | 34.454 | 1.000 | no | no | no |
| P64_s4_tau4 | C resolved | R_50 | 19.5000 | - | 34.454 | 36.9077 | 5.355 | 0.5722 | -0.1283 | 0.3400 | 34.454 | 1.000 | no | no | no |
| P64_s4_tau2 | A1 r_e≥1a | 5 | 6.9157 | 6.4324 | 82.749 | 12.7500 | 0.100 | 0.5245 | -0.1285 | 1.0000 | 82.685 | 0.209 | yes | no | no |
| P64_s4_tau2 | A1 resolved | 5 | 6.9157 | 6.4324 | 82.749 | 12.7500 | 0.100 | 0.5245 | -0.1285 | 1.0000 | 82.685 | 0.209 | yes | no | no |
| P64_s4_tau2 | C | R_50 | 7.9375 | - | 118.126 | 12.6997 | 0.249 | 0.5244 | -0.1286 | 0.7614 | 118.126 | 0.245 | yes | no | no |
| P64_s4_tau2 | C resolved | R_50 | 7.9375 | - | 118.126 | 12.6997 | 0.249 | 0.5244 | -0.1286 | 0.7614 | 118.126 | 0.245 | yes | no | no |
| P64_s4_newton | A1 r_e≥1a | 5 | 6.7894 | 6.3212 | 85.194 | 12.5000 | 0.085 | 0.5276 | -0.1266 | 1.0000 | 85.182 | 0.131 | yes | no | no |
| P64_s4_newton | A1 resolved | 5 | 6.7894 | 6.3212 | 85.194 | 12.5000 | 0.085 | 0.5276 | -0.1266 | 1.0000 | 85.182 | 0.131 | yes | no | no |
| P64_s4_newton | C | R_50 | 7.7500 | - | 121.388 | 12.3593 | 0.226 | 0.5258 | -0.1277 | 0.7568 | 121.388 | 0.132 | yes | no | no |
| P64_s4_newton | C resolved | R_50 | 7.7500 | - | 121.388 | 12.3593 | 0.226 | 0.5258 | -0.1277 | 0.7568 | 121.388 | 0.132 | yes | no | no |
| W2 | A1 r_e≥1a | 5 | 10.2405 | 9.3043 | 22.487 | 26.0000 | 2.894 | 0.5360 | -0.1558 | 1.0000 | 22.273 | 0.920 | yes | no | no |
| W2 | A1 resolved | 5 | 10.2405 | 9.3043 | 22.487 | 26.0000 | 2.894 | 0.5360 | -0.1558 | 1.0000 | 22.273 | 0.920 | yes | no | no |
| W2 | A1 sub-site | 5 | 18.7654 | 12.0434 | 4.44e-05 | 36.6719 | 3.510 | 0.5651 | -0.1367 | 1.0000 | 4.44e-05 | 1.000 | no | yes | yes |
| W2 | A2 | 7 | 19.8342 | 1.0085 | 18.811 | 36.6562 | 3.972 | 0.5651 | -0.1367 | 1.0000 | 18.723 | 1.000 | no | no | no |
| W2 | A2 resolved | 7 | 19.8342 | 1.0085 | 18.811 | 36.6562 | 3.972 | 0.5651 | -0.1367 | 1.0000 | 18.723 | 1.000 | no | no | no |
| W2 | B | δ 0.770 | 21.3750 | - | 34.020 | 36.5812 | 4.306 | 0.5649 | -0.1366 | 1.0000 | 34.020 | 1.000 | no | no | no |
| W2 | B resolved | δ 0.770 | 21.3750 | - | 34.020 | 36.5812 | 4.306 | 0.5649 | -0.1366 | 1.0000 | 34.020 | 1.000 | no | no | no |
| W2 | C | R_90 | 12.3750 | - | 52.468 | 26.1779 | 2.897 | 0.5369 | -0.1560 | 2.5640 | 52.468 | 1.000 | no | no | no |
| W2 | C resolved | R_90 | 12.3750 | - | 52.468 | 26.1779 | 2.897 | 0.5369 | -0.1560 | 2.5640 | 52.468 | 1.000 | no | no | no |
| ISO_64_s4 | A1 r_e≥1a | 5 | 7.4079 | 6.8589 | 76.530 | 13.5000 | 0.101 | 0.5386 | -0.1200 | 1.0000 | 76.465 | 0.226 | yes | no | no |
| ISO_64_s4 | A1 resolved | 5 | 7.4079 | 6.8589 | 76.530 | 13.5000 | 0.101 | 0.5386 | -0.1200 | 1.0000 | 76.465 | 0.226 | yes | no | no |
| ISO_64_s4 | B | δ 128.000 | 11.3750 | - | 191.000 | 13.4395 | 0.147 | 0.5384 | -0.1201 | 1.0000 | 191.000 | 0.637 | yes | no | no |
| ISO_64_s4 | B resolved | δ 128.000 | 11.3750 | - | 191.000 | 13.4395 | 0.147 | 0.5384 | -0.1201 | 1.0000 | 191.000 | 0.637 | yes | no | no |
| ISO_64_s4 | C | R_50 | 8.5000 | - | 112.952 | 13.4994 | 0.285 | 0.5386 | -0.1200 | 0.7597 | 112.952 | 0.275 | yes | no | no |
| ISO_64_s4 | C resolved | R_50 | 8.5000 | - | 112.952 | 13.4994 | 0.285 | 0.5386 | -0.1200 | 0.7597 | 112.952 | 0.275 | yes | no | no |
| S4_128_s8 | A1 r_e≥1a | 5 | 9.9368 | 9.1214 | 54.409 | 25.2109 | 2.849 | 0.5569 | -0.1438 | 1.0000 | 54.166 | 0.947 | yes | no | no |
| S4_128_s8 | A1 resolved | 5 | 9.9368 | 9.1214 | 54.409 | 25.2109 | 2.849 | 0.5569 | -0.1438 | 1.0000 | 54.166 | 0.947 | yes | no | no |
| S4_128_s8 | A1 sub-site | 5 | 16.0681 | 11.2471 | 2.68e-04 | 40.0000 | 6.176 | 0.5925 | -0.1220 | 1.0000 | 2.68e-04 | 1.000 | no | yes | yes |
| S4_128_s8 | A2 | 7 | 18.7951 | 0.6086 | 62.388 | 40.0000 | 7.231 | 0.5925 | -0.1220 | 1.0000 | 54.159 | 1.000 | no | no | no |
| S4_128_s8 | A2 resolved | 7 | 18.7951 | 0.6086 | 62.388 | 40.0000 | 7.231 | 0.5925 | -0.1220 | 1.0000 | 54.159 | 1.000 | no | no | no |
| S4_128_s8 | B | δ -2.000 | 18.7500 | - | 62.000 | 39.9378 | 7.199 | 0.5924 | -0.1221 | 1.0000 | 54.181 | 1.000 | no | no | no |
| S4_128_s8 | B resolved | δ -2.000 | 18.7500 | - | 62.000 | 39.9378 | 7.199 | 0.5924 | -0.1221 | 1.0000 | 54.181 | 1.000 | no | no | no |
| S4_128_s8 | C | R_90 | 11.6875 | - | 73.243 | 28.1218 | 1.352 | 0.5683 | -0.1374 | 15.9349 | 31.538 | 1.000 | no | no | no |
| S4_128_s8 | C resolved | R_90 | 11.6875 | - | 73.243 | 28.1218 | 1.352 | 0.5683 | -0.1374 | 15.9349 | 31.538 | 1.000 | no | no | no |
| S4_128_s16 | A1 r_e≥1a | 5 | 10.0708 | 9.2421 | 51.039 | 24.2500 | 2.646 | 0.5276 | -0.1272 | 1.0000 | 51.003 | 0.987 | yes | no | no |
| S4_128_s16 | A1 resolved | 5 | 10.0708 | 9.2421 | 51.039 | 24.2500 | 2.646 | 0.5276 | -0.1272 | 1.0000 | 51.003 | 0.987 | yes | no | no |
| S4_128_s16 | A1 sub-site | 5 | 22.8150 | 14.7382 | 5.63e-04 | 37.0156 | 1.706 | 0.5564 | -0.1277 | 1.0000 | 5.63e-04 | 1.000 | no | yes | yes |
| S4_128_s16 | A2 | 8 | 23.5312 | 1.1054 | 40.154 | 36.9453 | 2.196 | 0.5563 | -0.1278 | 1.0000 | 39.995 | 1.000 | no | no | no |
| S4_128_s16 | A2 resolved | 8 | 23.5312 | 1.1054 | 40.154 | 36.9453 | 2.196 | 0.5563 | -0.1278 | 1.0000 | 39.995 | 1.000 | no | no | no |
| S4_128_s16 | B | δ -2.000 | 11.1250 | - | 98.750 | 24.2524 | 3.015 | 0.5276 | -0.1272 | 1.0000 | 98.750 | 1.000 | yes | no | no |
| S4_128_s16 | B resolved | δ -2.000 | 11.1250 | - | 98.750 | 24.2524 | 3.015 | 0.5276 | -0.1272 | 1.0000 | 98.750 | 1.000 | yes | no | no |
| S4_128_s16 | C | R_90 | 11.8750 | - | 110.979 | 23.7142 | 2.764 | 0.5272 | -0.1266 | 1.2426 | 110.979 | 1.000 | no | no | no |
| S4_128_s16 | C resolved | R_90 | 11.8750 | - | 110.979 | 23.7142 | 2.764 | 0.5272 | -0.1266 | 1.2426 | 110.979 | 1.000 | no | no | no |
| S4_128_s4 | A1 r_e≥1a | 5 | 1.3479 | 1.3414 | 427.473 | 2.6562 | 0.050 | 0.5214 | -0.1303 | 1.0000 | 427.456 | 0.000 | yes | no | no |
| S4_128_s4 | A1 resolved | 5 | 1.3479 | 1.3414 | 427.473 | 2.6562 | 0.050 | 0.5214 | -0.1303 | 1.0000 | 427.456 | 0.000 | yes | no | no |
| S4_128_s4 | C | R_90 | 2.1875 | - | 427.057 | 3.5046 | 0.057 | 0.5215 | -0.1303 | 0.6260 | 427.057 | 0.000 | yes | no | no |
| S4_128_s4 | C resolved | R_90 | 2.1875 | - | 427.057 | 3.5046 | 0.057 | 0.5215 | -0.1303 | 0.6260 | 427.057 | 0.000 | yes | no | no |
| ISO_128_s4 | A1 r_e≥1a | 5 | 1.3422 | 1.3360 | 427.700 | 2.6484 | 0.051 | 0.5211 | -0.1305 | 1.0000 | 427.551 | 0.000 | yes | no | no |
| ISO_128_s4 | A1 resolved | 5 | 1.3422 | 1.3360 | 427.700 | 2.6484 | 0.051 | 0.5211 | -0.1305 | 1.0000 | 427.551 | 0.000 | yes | no | no |
| ISO_128_s4 | C | R_50 | 7.4375 | - | 254.916 | 11.8603 | 0.208 | 0.5279 | -0.1261 | 0.7759 | 254.916 | 0.039 | yes | no | no |
| ISO_128_s4 | C resolved | R_50 | 7.4375 | - | 254.916 | 11.8603 | 0.208 | 0.5279 | -0.1261 | 0.7759 | 254.916 | 0.039 | yes | no | no |
| ISO_128_s4_plain | A1 r_e≥1a | 5 | 1.3422 | 1.3360 | 427.700 | 2.6484 | 0.051 | 0.5211 | -0.1305 | 1.0000 | 427.550 | 0.000 | yes | no | no |
| ISO_128_s4_plain | A1 resolved | 5 | 1.3422 | 1.3360 | 427.700 | 2.6484 | 0.051 | 0.5211 | -0.1305 | 1.0000 | 427.550 | 0.000 | yes | no | no |
| ISO_128_s4_plain | C | R_90 | 1.9375 | - | 428.084 | 3.2392 | 0.056 | 0.5211 | -0.1305 | 0.6227 | 428.084 | 0.000 | yes | no | no |
| ISO_128_s4_plain | C resolved | R_90 | 1.9375 | - | 428.084 | 3.2392 | 0.056 | 0.5211 | -0.1305 | 0.6227 | 428.084 | 0.000 | yes | no | no |
| M3c_s4_96 | A1 r_e≥1a | 5 | 13.6536 | 11.7540 | 28.517 | 22.0833 | 0.073 | 0.5257 | -0.1288 | 1.0000 | 28.353 | 0.059 | yes | no | no |
| M3c_s4_96 | A1 resolved | 5 | 13.6536 | 11.7540 | 28.517 | 22.0833 | 0.073 | 0.5257 | -0.1288 | 1.0000 | 28.353 | 0.059 | yes | no | no |
| M3c_s4_96 | A2 | 6 | 14.4945 | 0.3946 | 81.928 | 21.9167 | 0.274 | 0.5231 | -0.1305 | 1.0000 | 81.691 | 0.045 | yes | no | no |
| M3c_s4_96 | A2 resolved | 6 | 14.4945 | 0.3946 | 81.928 | 21.9167 | 0.274 | 0.5231 | -0.1305 | 1.0000 | 81.691 | 0.045 | yes | no | no |
| M3c_s4_96 | C | R_90 | 18.0833 | - | 214.766 | 21.9716 | 0.323 | 0.5238 | -0.1300 | 0.7801 | 214.766 | 0.100 | yes | no | no |
| M3c_s4_96 | C resolved | R_90 | 18.0833 | - | 214.766 | 21.9716 | 0.323 | 0.5238 | -0.1300 | 0.7801 | 214.766 | 0.100 | yes | no | no |
| P64_s4_B2 | A1 r_e≥1a | 5 | 10.5356 | 9.5445 | 19.057 | 40.0000 | 9.910 | 0.5793 | -0.1172 | 1.0000 | 18.902 | 0.959 | yes | no | no |
| P64_s4_B2 | A1 resolved | 5 | 10.5356 | 9.5445 | 19.057 | 40.0000 | 9.910 | 0.5793 | -0.1172 | 1.0000 | 18.902 | 0.959 | yes | no | no |
| P64_s4_B2 | A2 | 6 | 14.3532 | 0.3491 | 42.236 | 40.0000 | 8.906 | 0.5793 | -0.1172 | 1.0000 | 34.698 | 1.000 | no | no | no |
| P64_s4_B2 | A2 resolved | 6 | 14.3532 | 0.3491 | 42.236 | 40.0000 | 8.906 | 0.5793 | -0.1172 | 1.0000 | 34.698 | 1.000 | no | no | no |
| P64_s4_B2 | B | δ 0.100 | 17.6250 | - | 35.350 | 39.7464 | 8.581 | 0.5790 | -0.1175 | 1.0000 | 35.350 | 1.000 | no | no | no |
| P64_s4_B2 | B resolved | δ 0.100 | 17.6250 | - | 35.350 | 39.7464 | 8.581 | 0.5790 | -0.1175 | 1.0000 | 35.350 | 1.000 | no | no | no |
| P64_s4_B2 | C | R_50 | 11.5000 | - | 42.859 | 26.6804 | 3.983 | 0.5556 | -0.1341 | 0.7500 | 42.859 | 0.984 | yes | no | no |
| P64_s4_B2 | C resolved | R_50 | 11.5000 | - | 42.859 | 26.6804 | 3.983 | 0.5556 | -0.1341 | 0.7500 | 42.859 | 0.984 | yes | no | no |
| P64_s4_ext200 | A1 r_e≥1a | 5 | 10.5356 | 9.5445 | 19.057 | 40.0000 | 9.910 | 0.5793 | -0.1172 | 1.0000 | - | 0.959 | yes | no | no |
| P64_s4_ext200 | A1 resolved (e, c only: ray mask unavailable) | 5 | 10.5356 | 9.5445 | 19.057 | 40.0000 | 9.910 | 0.5793 | -0.1172 | 1.0000 | - | 0.959 | yes | no | no |
| P64_s4_ext200 | A2 | 6 | 14.3447 | 0.3485 | 42.277 | 40.0000 | 8.911 | 0.5793 | -0.1172 | 1.0000 | - | 1.000 | no | no | no |
| P64_s4_ext200 | A2 resolved (e, c only: ray mask unavailable) | 6 | 14.3447 | 0.3485 | 42.277 | 40.0000 | 8.911 | 0.5793 | -0.1172 | 1.0000 | - | 1.000 | no | no | no |
| P64_s4_ext200 | B | δ 0.100 | 17.6250 | - | 35.350 | 39.7464 | 8.581 | 0.5790 | -0.1175 | 1.0000 | - | 1.000 | no | no | no |
| P64_s4_ext200 | B resolved (e, c only: ray mask unavailable) | δ 0.100 | 17.6250 | - | 35.350 | 39.7464 | 8.581 | 0.5790 | -0.1175 | 1.0000 | - | 1.000 | no | no | no |
| P64_s4_ext200 | C | R_50 | 11.5000 | - | 42.859 | 26.6804 | 3.983 | 0.5556 | -0.1341 | 0.7500 | - | 0.984 | yes | no | no |
| P64_s4_ext200 | C resolved (e, c only: ray mask unavailable) | R_50 | 11.5000 | - | 42.859 | 26.6804 | 3.983 | 0.5556 | -0.1341 | 0.7500 | - | 0.984 | yes | no | no |
| P64_s4_heldsrc | A1 r_e≥1a | 5 | 10.4133 | 9.4468 | 20.966 | 24.4062 | 2.852 | 0.5446 | -0.1200 | 1.0000 | 20.714 | 0.896 | yes | no | no |
| P64_s4_heldsrc | A1 resolved | 5 | 10.4133 | 9.4468 | 20.966 | 24.4062 | 2.852 | 0.5446 | -0.1200 | 1.0000 | 20.714 | 0.896 | yes | no | no |
| P64_s4_heldsrc | A1 sub-site | 5 | 19.7522 | 13.0814 | 0.014 | 29.2656 | -0.096 | 0.5763 | -0.1440 | 1.0000 | 0.014 | 1.000 | no | yes | yes |
| P64_s4_heldsrc | A2 | 6 | 14.1206 | 0.1193 | 57.029 | 24.3594 | 2.179 | 0.5443 | -0.1194 | 1.0000 | 57.016 | 1.000 | no | no | no |
| P64_s4_heldsrc | A2 resolved | 6 | 14.1206 | 0.1193 | 57.029 | 24.3594 | 2.179 | 0.5443 | -0.1194 | 1.0000 | 57.016 | 1.000 | no | no | no |
| P64_s4_heldsrc | B | δ -2.000 | 11.5000 | - | 45.000 | 24.5203 | 3.197 | 0.5454 | -0.1219 | 1.0000 | 45.000 | 0.971 | yes | no | no |
| P64_s4_heldsrc | B resolved | δ -2.000 | 11.5000 | - | 45.000 | 24.5203 | 3.197 | 0.5454 | -0.1219 | 1.0000 | 45.000 | 0.971 | yes | no | no |
| P64_s4_heldsrc | C | R_90 | 12.6250 | - | 49.291 | 25.4888 | 2.376 | 0.5545 | -0.1381 | 1.6677 | 49.291 | 1.000 | no | no | no |
| P64_s4_heldsrc | C resolved | R_90 | 12.6250 | - | 49.291 | 25.4888 | 2.376 | 0.5545 | -0.1381 | 1.6677 | 49.291 | 1.000 | no | no | no |
| P64_s4_widened | A1 r_e≥1a | 5 | 10.3554 | 9.4001 | 21.893 | 24.9688 | 3.073 | 0.5302 | -0.1249 | 1.0000 | 21.756 | 0.905 | yes | no | no |
| P64_s4_widened | A1 resolved | 5 | 10.3554 | 9.4001 | 21.893 | 24.9688 | 3.073 | 0.5302 | -0.1249 | 1.0000 | 21.756 | 0.905 | yes | no | no |
| P64_s4_widened | A1 sub-site | 5 | 18.1211 | 12.2860 | 0.007 | 36.1094 | 2.506 | 0.5563 | -0.1281 | 1.0000 | 0.007 | 1.000 | no | yes | yes |
| P64_s4_widened | A2 | 8 | 22.5758 | 0.4854 | 37.423 | 36.2500 | 3.464 | 0.5564 | -0.1280 | 1.0000 | 37.409 | 1.000 | no | no | no |
| P64_s4_widened | A2 resolved | 8 | 22.5758 | 0.4854 | 37.423 | 36.2500 | 3.464 | 0.5564 | -0.1280 | 1.0000 | 37.409 | 1.000 | no | no | no |
| P64_s4_widened | B | δ 0.462 | 21.2500 | - | 32.962 | 36.0515 | 3.652 | 0.5562 | -0.1282 | 1.0000 | 32.962 | 1.000 | no | no | no |
| P64_s4_widened | B resolved | δ 0.462 | 21.2500 | - | 32.962 | 36.0515 | 3.652 | 0.5562 | -0.1282 | 1.0000 | 32.962 | 1.000 | no | no | no |
| P64_s4_widened | C | R_90 | 12.6250 | - | 53.895 | 25.2985 | 3.028 | 0.5304 | -0.1252 | 1.3013 | 53.895 | 1.000 | no | no | no |
| P64_s4_widened | C resolved | R_90 | 12.6250 | - | 53.895 | 25.2985 | 3.028 | 0.5304 | -0.1252 | 1.3013 | 53.895 | 1.000 | no | no | no |

**Table H3. Clock domains** (as v2; reviewer B rzchk3 secs. 4.5 and 5): per comoving clock, from its own worldline and the backward map: release; first time inside the first site (r < 1 a, † past the certified end); first time inside the matter root (†); the last emission received inside the solver-certified record interval on the backward map (T_e, τ_e, r_e there); the last forward launch received; the received share of the clock's proper time; two energy-transfer DIAGNOSTICS on the backward map: ∫G_D dτ_e/Δτ (weighted by the coordinate-clock ratio; v2's) and ∫g_D dτ_e/Δτ (weighted by the static receiver's ratio). Neither is a transmission probability (no transmission treatment; a time-dependent background can supply energy). Two domains, each with Δτ = τ(t_end) − τ(T_rel) read on the clock's stored worldline samples (step 0.5 a): UNMASKED (v2's and v3's: every received event, sub-site emissions included; t_end the last stored sample at or before min(the clock's last finite sample, the certified end)) and RESOLVED (v4: only events passing the masks (e), (c), (r), summed over consecutive arrivals both resolved and 1 a apart; t_end the last stored sample strictly before the sample at which the clock is first inside the first site, one sample (0.5 a) before the printed first-entry time, or at or before the certified end if that is earlier; printed as the domain end).

| member | clock | release T/r_s | r_0/r_s | inside the first site from T/r_s | inside the matter root from T/r_s | last received emission (backward): T_e/r_s, τ_e/r_s, r_e (a) | last received forward launch T_e/r_s | unmasked: received proper-time share / ∫G_D dτ_e/Δτ / ∫g_D dτ_e/Δτ | resolved: domain end T/r_s; share / ∫G_D dτ_e/Δτ / ∫g_D dτ_e/Δτ |
|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | A1.0 | 0.00 | 0.50 | 11.2109 | 12.1250 | 10.1436, 8.4027, 2.592 | 10.1250 | 0.582 / 0.326 / 0.370 | 11.2031; 0.913 / 0.512 / 0.581 |
| P64_s4 | A1.1 | 0.00 | 1.00 | 11.7812 | 12.1250 | 10.2006, 8.4901, 5.327 | 10.1875 | 0.587 / 0.317 / 0.359 | 11.7734; 0.883 / 0.477 / 0.541 |
| P64_s4 | A1.2 | 0.00 | 1.50 | 12.0469 | 12.1250 | 10.2615, 8.6071, 8.113 | 10.2500 | 0.592 / 0.309 / 0.351 | 12.0391; 0.876 / 0.457 / 0.519 |
| P64_s4 | A1.3 | 0.00 | 2.00 | 12.2109 | 12.1250 | 10.3213, 8.7502, 10.719 | 10.3125 | 0.598 / 0.302 / 0.343 | 12.2031; 0.875 / 0.442 / 0.502 |
| P64_s4 | A1.4 | 0.00 | 3.00 | 12.3984 | 12.1250 | 10.4308, 9.1063, 15.169 | 10.4062 | 0.610 / 0.290 / 0.329 | 12.3906; 0.879 / 0.417 / 0.474 |
| P64_s4 | A1.5 | 0.00 | 4.00 | 12.5156 | 12.1250 | 10.5356, 9.5445, 19.057 | 10.5312 | 0.591 / 0.265 / 0.301 | 12.5078; 0.839 / 0.376 / 0.427 |
| P64_s4 | A2.6 | 14.00 | 1.00 | 17.6953 | 14.2344 | 14.3532, 0.3491, 42.236 | 14.3516 | 0.073 / 0.004 / 0.004 | 17.6875; 0.275 / 0.014 / 0.016 |
| P64_s4 | A2.7 | 18.00 | 1.00 | 20.7656 | 18.8750 | 18.5888, 0.4771, 36.529 | 18.5000 | 0.118 / 0.013 / 0.015 | 20.7578; 0.404 / 0.046 / 0.052 |
| P64_s4 | A2.8 | 22.00 | 1.00 | 25.8359 | 22.5312 | 22.5435, 0.4826, 36.774 | 22.5391 | 0.141 / 0.016 / 0.018 | 25.8281; 0.368 / 0.041 / 0.047 |
| P64_s4 | A2.9 | 26.00 | 1.00 | 28.7031 | 26.5625 | 26.5417, 0.4880, 36.719 | 26.5391 | 0.176 / 0.022 / 0.025 | 28.6953; 0.442 / 0.056 / 0.063 |
| P64_s4 | A2.10 | 30.00 | 1.00 | 33.1562 | 30.5547 | 30.4259, 0.3942, 42.772 | 30.4219 | 0.175 / 0.026 / 0.029 | 33.1484; 0.314 / 0.047 / 0.052 |
| P64_s4 | A2.11 | 34.00 | 1.00 | 37.4688 | 34.5469 | -, -, - | - | - / - / - | 37.4609; - / - / - |
| P64_s2 | A1.0 | 0.00 | 0.50 | 9.2188 | 12.0000 | 10.0425, 8.2974, 0.463 | 9.1875 | 0.660 / 0.372 / 0.420 | 9.2109; 0.966 / 0.589 / 0.666 |
| P64_s2 | A1.1 | 0.00 | 1.00 | 11.1484 | 12.0000 | 10.1010, 8.3974, 3.318 | 10.0938 | 0.666 / 0.363 / 0.411 | 11.1406; 0.916 / 0.500 / 0.565 |
| P64_s2 | A1.2 | 0.00 | 1.50 | 11.6953 | 12.0000 | 10.1777, 8.5308, 6.844 | 10.1562 | 0.672 / 0.355 / 0.402 | 11.6875; 0.890 / 0.470 / 0.532 |
| P64_s2 | A1.3 | 0.00 | 2.00 | 11.9375 | 12.0000 | 10.2483, 8.6845, 9.892 | 10.2188 | 0.678 / 0.347 / 0.393 | 11.9297; 0.885 / 0.453 / 0.512 |
| P64_s2 | A1.4 | 0.00 | 3.00 | 12.1719 | 12.0000 | 10.3695, 9.0518, 14.749 | 10.3438 | 0.692 / 0.333 / 0.377 | 12.1641; 0.889 / 0.427 / 0.484 |
| P64_s2 | A1.5 | 0.00 | 4.00 | 12.2969 | 12.0000 | 10.4793, 9.4950, 18.742 | 10.4688 | 0.668 / 0.303 / 0.343 | 12.2891; 0.848 / 0.385 / 0.436 |
| P64_s2 | A2.6 | 14.00 | 1.00 | 16.6172 | 14.3203 | 14.4128, 0.4060, 39.469 | 14.4062 | 0.127 / 0.011 / 0.012 | 16.6094; 0.401 / 0.034 / 0.038 |
| P64_s2 | A2.7 | 18.00 | 1.00 | 20.9844 | 18.7266 | 18.7257, 0.5914, 32.096 | 18.7188 | 0.207 / 0.032 / 0.036 | 20.9766; 0.561 / 0.087 / 0.098 |
| P64_s2 | A2.8 | 22.00 | 1.00 | 24.4766 | 22.7031 | 22.6965, 0.6120, 31.287 | 22.6953 | 0.238 / 0.037 / 0.042 | 24.4688; 0.596 / 0.093 / 0.104 |
| P64_s2 | A2.9 | 26.00 | 1.00 | 28.2500 | 26.6797 | 26.6713, 0.6086, 31.528 | 26.6641 | 0.269 / 0.042 / 0.046 | 28.2422; 0.598 / 0.092 / 0.103 |
| P64_s2 | A2.10 | 30.00 | 1.00 | 32.2266 | 30.6719 | 30.5722, 0.5261, 36.519 | 30.5703 | 0.284 / 0.050 / 0.055 | 32.2188; 0.502 / 0.088 / 0.098 |
| P64_s2 | A2.11 | 34.00 | 1.00 | 36.5859 | 34.7188 | -, -, - | - | - / - / - | 36.5781; - / - / - |
| P64_s8 | A1.0 | 0.00 | 0.50 | 11.5078 | 13.0000 | 23.9021, 14.1273, 3.12e-05 | 11.5000 | 0.708 / 0.684 / 0.783 | 11.5000; 0.981 / 0.557 / 0.634 |
| P64_s8 | A1.1 | 0.00 | 1.00 | 12.1484 | 13.0000 | 23.9021, 14.1697, 6.51e-05 | 12.1250 | 0.709 / 0.665 / 0.762 | 12.1406; 0.991 / 0.688 / 0.786 |
| P64_s8 | A1.2 | 0.00 | 1.50 | 12.4609 | 13.0000 | 23.9021, 14.2401, 1.01e-04 | 12.4375 | 0.710 / 0.648 / 0.742 | 12.4531; 0.991 / 0.743 / 0.849 |
| P64_s8 | A1.3 | 0.00 | 2.00 | 12.6562 | 13.0000 | 23.9021, 14.3385, 1.37e-04 | 12.6250 | 0.711 / 0.631 / 0.722 | 12.6484; 0.989 / 0.759 / 0.867 |
| P64_s8 | A1.4 | 0.00 | 3.00 | 12.8984 | 13.0000 | 23.9021, 14.6155, 2.08e-04 | 12.8750 | 0.715 / 0.601 / 0.688 | 12.8906; 0.987 / 0.747 / 0.854 |
| P64_s8 | A1.5 | 0.00 | 4.00 | 13.0703 | 13.0000 | 23.9021, 14.9805, 2.86e-04 | 13.0625 | 0.696 / 0.560 / 0.641 | 13.0625; 0.951 / 0.751 / 0.859 |
| P64_s8 | A2.6 | 14.00 | 1.00 | 19.6094 | 14.2422 | 23.9053, 3.9080, 0.100 | 19.5000 | 0.402 / 0.148 / 0.169 | 19.6016; 0.947 / 0.447 / 0.514 |
| P64_s8 | A2.7 | 18.00 | 1.00 | 25.0391 | - | 23.9825, 2.9263, 2.445 | 23.8750 | 0.336 / 0.086 / 0.098 | 25.0312; 0.853 / 0.218 / 0.249 |
| P64_s8 | A2.8 | 22.00 | 1.00 | 26.7656 | - | 24.3689, 1.4602, 12.378 | 24.2500 | 0.206 / 0.034 / 0.039 | 26.7578; 0.575 / 0.095 / 0.109 |
| P64_s8 | A2.9 | 26.00 | 1.00 | 29.7734 | - | 26.5462, 0.4819, 37.046 | 26.5000 | 0.091 / 0.009 / 0.010 | 29.7656; 0.296 / 0.029 / 0.033 |
| P64_s8 | A2.10 | 30.00 | 1.00 | 35.7812 | - | 30.3045, 0.2794, 48.480 | 30.2500 | 0.066 / 0.009 / 0.010 | 35.7734; 0.109 / 0.014 / 0.016 |
| P64_s8 | A2.11 | 34.00 | 1.00 | 38.6484 | - | -, -, - | - | - / - / - | 38.6406; - / - / - |
| P64_s4_tau4 | A1.0 | 0.00 | 0.50 | 11.1328 | 12.0000 | 10.3606, 8.5688, 2.130 | 10.3594 | 0.573 / 0.308 / 0.351 | 11.1250; 0.938 / 0.505 / 0.575 |
| P64_s4_tau4 | A1.1 | 0.00 | 1.00 | 11.6797 | 12.0000 | 10.4089, 8.6492, 4.388 | 10.4062 | 0.576 / 0.299 / 0.341 | 11.6719; 0.908 / 0.472 / 0.537 |
| P64_s4_tau4 | A1.2 | 0.00 | 1.50 | 11.9297 | 12.0000 | 10.4607, 8.7589, 6.701 | 10.4531 | 0.581 / 0.293 / 0.333 | 11.9219; 0.900 / 0.453 / 0.515 |
| P64_s4_tau4 | A1.3 | 0.00 | 2.00 | 12.0859 | 12.0000 | 10.5118, 8.8951, 8.876 | 10.5000 | 0.586 / 0.286 / 0.326 | 12.0781; 0.898 / 0.438 / 0.499 |
| P64_s4_tau4 | A1.4 | 0.00 | 3.00 | 12.2578 | 12.0000 | 10.6060, 9.2396, 12.615 | 10.5938 | 0.598 / 0.275 / 0.313 | 12.2500; 0.900 / 0.415 / 0.472 |
| P64_s4_tau4 | A1.5 | 0.00 | 4.00 | 12.3750 | 12.0000 | 10.6971, 9.6684, 15.911 | 10.6875 | 0.579 / 0.252 / 0.287 | 12.3672; 0.858 / 0.374 / 0.425 |
| P64_s4_tau4 | A2.6 | 14.00 | 1.00 | 18.0078 | 14.1953 | 15.3487, 0.7469, 12.962 | 15.3438 | 0.139 / 0.011 / 0.012 | 18.0000; 0.544 / 0.042 / 0.048 |
| P64_s4_tau4 | A2.7 | 18.00 | 1.00 | 21.3203 | 20.3125 | 18.8215, 0.5895, 31.102 | 18.8125 | 0.131 / 0.015 / 0.017 | 21.3125; 0.414 / 0.046 / 0.054 |
| P64_s4_tau4 | A2.8 | 22.00 | 1.00 | 26.1172 | 22.5391 | 22.5711, 0.4992, 35.756 | 22.5703 | 0.136 / 0.015 / 0.017 | 26.1094; 0.371 / 0.040 / 0.046 |
| P64_s4_tau4 | A2.9 | 26.00 | 1.00 | 28.8906 | 27.3750 | 26.6131, 0.5313, 34.145 | 26.5625 | 0.178 / 0.022 / 0.025 | 28.8828; 0.421 / 0.052 / 0.059 |
| P64_s4_tau4 | A2.10 | 30.00 | 1.00 | 33.7344 | 30.5781 | 30.4490, 0.4064, 42.307 | 30.4453 | 0.174 / 0.027 / 0.030 | 33.7266; 0.300 / 0.046 / 0.052 |
| P64_s4_tau4 | A2.11 | 34.00 | 1.00 | 36.6172 | 34.6250 | -, -, - | - | - / - / - | 36.6094; - / - / - |
| P64_s4_tau2 | A1.0 | 0.00 | 0.50 | 11.0938 | 11.9375 | 5.3612, 4.3756, 13.508 | 5.3594 | 0.442 / 0.296 / 0.338 | 11.0859; 0.481 / 0.322 / 0.367 |
| P64_s4_tau2 | A1.1 | 0.00 | 1.00 | 11.6250 | 11.9375 | 5.6266, 4.6338, 26.721 | 5.6250 | 0.466 / 0.297 / 0.339 | 11.6172; 0.488 / 0.311 / 0.355 |
| P64_s4_tau2 | A1.2 | 0.00 | 1.50 | 11.8750 | 11.9375 | 5.8837, 4.9083, 38.953 | 5.8828 | 0.491 / 0.299 / 0.341 | 11.8672; 0.507 / 0.309 / 0.352 |
| P64_s4_tau2 | A1.3 | 0.00 | 2.00 | 12.0234 | 11.9375 | 6.1171, 5.1885, 49.614 | 6.1094 | 0.514 / 0.299 / 0.341 | 12.0156; 0.526 / 0.307 / 0.350 |
| P64_s4_tau2 | A1.4 | 0.00 | 3.00 | 12.1875 | 11.9375 | 6.5313, 5.7818, 67.510 | 6.5312 | 0.557 / 0.299 / 0.341 | 12.1797; 0.566 / 0.304 / 0.347 |
| P64_s4_tau2 | A1.5 | 0.00 | 4.00 | 12.2969 | 11.9375 | 6.9157, 6.4324, 82.749 | 6.9141 | 0.551 / 0.278 / 0.316 | 12.2891; 0.557 / 0.281 / 0.320 |
| P64_s4_tau2 | A2.6 | 14.00 | 1.00 | 18.2031† | 14.1719† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_tau2 | A2.7 | 18.00 | 1.00 | - | 21.1875† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_newton | A1.0 | 0.00 | 0.50 | 11.2109 | 12.1250 | 5.1880, 4.2317, 13.977 | 5.1875 | 0.427 / 0.288 / 0.328 | 11.2031; 0.460 / 0.310 / 0.353 |
| P64_s4_newton | A1.1 | 0.00 | 1.00 | 11.7812 | 12.1250 | 5.4622, 4.4963, 27.594 | 5.4375 | 0.451 / 0.289 / 0.329 | 11.7734; 0.468 / 0.299 / 0.341 |
| P64_s4_newton | A1.2 | 0.00 | 1.50 | 12.0469 | 12.1250 | 5.7269, 4.7764, 40.171 | 5.7188 | 0.477 / 0.291 / 0.332 | 12.0391; 0.486 / 0.297 / 0.339 |
| P64_s4_newton | A1.3 | 0.00 | 2.00 | 12.2109 | 12.1250 | 5.9669, 5.0612, 51.113 | 5.9375 | 0.500 / 0.292 / 0.333 | 12.2031; 0.506 / 0.296 / 0.337 |
| P64_s4_newton | A1.4 | 0.00 | 3.00 | 12.3984 | 12.1250 | 6.3931, 5.6628, 69.497 | 6.3750 | 0.544 / 0.293 / 0.334 | 12.3906; 0.546 / 0.295 / 0.336 |
| P64_s4_newton | A1.5 | 0.00 | 4.00 | - | 12.1250 | 6.7894, 6.3212, 85.194 | 6.7812 | 0.540 / 0.273 / 0.311 | 12.5000; 0.540 / 0.273 / 0.311 |
| W2 | A1.0 | 0.00 | 0.50 | 11.0625 | 11.8750 | 19.0896, 11.2831, 3.53e-06 | 11.0312 | 0.705 / 3.345 / 3.887 | 11.0547; 0.966 / 0.609 / 0.697 |
| W2 | A1.1 | 0.00 | 1.00 | 11.5781 | 11.8750 | 19.0896, 11.3275, 7.63e-06 | 11.5625 | 0.706 / 3.175 / 3.690 | 11.5703; 0.978 / 2.293 / 2.658 |
| W2 | A1.2 | 0.00 | 1.50 | 11.8125 | 11.8750 | 19.0896, 11.3998, 1.22e-05 | 11.7812 | 0.708 / 3.001 / 3.488 | 11.8047; 0.968 / 2.120 / 2.458 |
| W2 | A1.3 | 0.00 | 2.00 | 11.9531 | 11.8750 | 19.0896, 11.5002, 1.69e-05 | 11.9375 | 0.709 / 2.835 / 3.294 | 11.9453; 0.963 / 1.971 / 2.285 |
| W2 | A1.4 | 0.00 | 3.00 | 12.1172 | 11.8750 | 19.0896, 11.7816, 2.63e-05 | 12.0938 | 0.714 / 2.540 / 2.951 | 12.1094; 0.960 / 1.724 / 1.997 |
| W2 | A1.5 | 0.00 | 4.00 | 12.2266 | 11.8750 | 19.0896, 12.1527, 3.61e-05 | 12.2188 | 0.690 / 2.261 / 2.627 | 12.2188; 0.912 / 1.491 / 1.726 |
| W2 | A2.6 | 14.00 | 1.00 | 18.3828 | 14.1484 | 19.1180, 1.7429, 0.622 | 18.3750 | 0.270 / 0.053 / 0.062 | 18.3750; 0.971 / 0.219 / 0.255 |
| W2 | A2.7 | 18.00 | 1.00 | 22.3594 | 22.1250 | 19.9985, 1.0788, 16.764 | 19.8750 | 0.196 / 0.027 / 0.031 | 22.3516; 0.553 / 0.075 / 0.088 |
| W2 | A2.8 | 22.00 | 1.00 | 24.4219 | 22.6094 | 22.6526, 0.5514, 33.037 | 22.6484 | 0.129 / 0.014 / 0.016 | 24.4141; 0.514 / 0.055 / 0.064 |
| W2 | A2.9 | 26.00 | 1.00 | 30.1016 | 30.2500 | 26.9153, 0.6884, 24.720 | 26.8750 | 0.191 / 0.022 / 0.025 | 30.0938; 0.413 / 0.047 / 0.054 |
| W2 | A2.10 | 30.00 | 1.00 | 32.5312 | 30.6562 | 30.4881, 0.4252, 41.558 | 30.4844 | 0.162 / 0.026 / 0.030 | 32.5234; 0.376 / 0.061 / 0.070 |
| W2 | A2.11 | 34.00 | 1.00 | 37.8594 | 37.7500 | -, -, - | - | - / - / - | 37.8516; - / - / - |
| ISO_64_s4 | A1.0 | 0.00 | 0.50 | 12.2422 | 12.3750 | 6.1094, 5.0239, 16.600 | 6.0938 | 0.483 / 0.331 / 0.377 | 12.2344; 0.503 / 0.345 / 0.392 |
| ISO_64_s4 | A1.1 | 0.00 | 1.00 | 12.5234 | 12.3750 | 6.3680, 5.2697, 29.817 | 6.3438 | 0.505 / 0.330 / 0.376 | 12.5156; 0.519 / 0.340 / 0.387 |
| ISO_64_s4 | A1.2 | 0.00 | 1.50 | 12.6406 | 12.3750 | 6.5739, 5.4975, 39.945 | 6.5625 | 0.523 / 0.326 / 0.371 | 12.6328; 0.535 / 0.333 / 0.380 |
| ISO_64_s4 | A1.3 | 0.00 | 2.00 | 12.7109 | 12.3750 | 6.7546, 5.7328, 48.526 | 6.7500 | 0.541 / 0.321 / 0.365 | 12.7031; 0.552 / 0.327 / 0.373 |
| ISO_64_s4 | A1.4 | 0.00 | 3.00 | 12.7969 | 12.3750 | 7.0870, 6.2584, 63.423 | 7.0625 | 0.577 / 0.314 / 0.358 | 12.7891; 0.586 / 0.319 / 0.363 |
| ISO_64_s4 | A1.5 | 0.00 | 4.00 | 12.8672 | 12.3750 | 7.4079, 6.8589, 76.530 | 7.4062 | 0.567 / 0.288 / 0.328 | 12.8594; 0.574 / 0.292 / 0.333 |
| ISO_64_s4 | A2.6 | 14.00 | 1.00 | 17.6641† | 14.2422† | -, -, - | - | - / - / - | -; - / - / - |
| ISO_64_s4 | A2.7 | 18.00 | 1.00 | 21.0469† | 19.5000† | -, -, - | - | - / - / - | -; - / - / - |
| ISO_64_s4 | A2.8 | 22.00 | 1.00 | 26.3281† | 22.4922† | -, -, - | - | - / - / - | -; - / - / - |
| ISO_64_s4 | A2.9 | 26.00 | 1.00 | 28.6484† | 26.5469† | -, -, - | - | - / - / - | -; - / - / - |
| ISO_64_s4 | A2.10 | 30.00 | 1.00 | 33.6016† | 30.5156† | -, -, - | - | - / - / - | -; - / - / - |
| ISO_64_s4 | A2.11 | 34.00 | 1.00 | 37.1484† | 34.5312† | -, -, - | - | - / - / - | -; - / - / - |
| S4_128_s8 | A1.0 | 0.00 | 0.50 | 11.3086 | 11.1875 | 16.0681, 10.4054, 4.70e-05 | 11.2969 | 0.687 / 2.991 / 3.423 | 11.3047; 0.951 / 1.343 / 1.533 |
| S4_128_s8 | A1.1 | 0.00 | 1.00 | 11.4609 | 11.1875 | 16.0681, 10.4448, 8.56e-05 | 11.4531 | 0.688 / 2.758 / 3.156 | 11.4570; 0.946 / 1.228 / 1.402 |
| S4_128_s8 | A1.2 | 0.00 | 1.50 | 11.5312 | 11.1875 | 16.0681, 10.5127, 1.19e-04 | 11.5156 | 0.690 / 2.573 / 2.945 | 11.5273; 0.999 / 3.519 / 4.026 |
| S4_128_s8 | A1.3 | 0.00 | 2.00 | 11.5781 | 11.1875 | 16.0681, 10.6098, 1.49e-04 | 11.5625 | 0.691 / 2.417 / 2.765 | 11.5742; 0.997 / 3.269 / 3.741 |
| S4_128_s8 | A1.4 | 0.00 | 3.00 | 11.6445 | 11.1875 | 16.0681, 10.8862, 2.05e-04 | 11.6406 | 0.697 / 2.150 / 2.460 | 11.6406; 0.995 / 2.847 / 3.257 |
| S4_128_s8 | A1.5 | 0.00 | 4.00 | 11.6953 | 11.1875 | 16.0681, 11.2471, 2.68e-04 | 11.6875 | 0.672 / 1.889 / 2.162 | 11.6914; 0.944 / 2.441 / 2.793 |
| S4_128_s8 | A2.6 | 14.00 | 1.00 | 19.4492 | 14.1797 | 16.6610, 1.0792, 16.176 | 16.6250 | 0.190 / 0.025 / 0.028 | 19.4453; 0.516 / 0.067 / 0.077 |
| S4_128_s8 | A2.7 | 18.00 | 1.00 | 20.5156 | 19.3125 | 18.7951, 0.6086, 62.388 | 18.7500 | 0.133 / 0.018 / 0.020 | 20.5117; 0.479 / 0.063 / 0.073 |
| S4_128_s8 | A2.8 | 22.00 | 1.00 | 25.8359 | 22.5898 | 22.7306, 0.6354, 55.464 | 22.7305 | 0.166 / 0.019 / 0.022 | 25.8320; 0.429 / 0.049 / 0.056 |
| S4_128_s8 | A2.9 | 26.00 | 1.00 | 29.6797 | 26.7422 | 26.7246, 0.6171, 62.105 | 26.7227 | 0.203 / 0.032 / 0.036 | 29.6758; 0.480 / 0.075 / 0.085 |
| S4_128_s8 | A2.10 | 30.00 | 1.00 | 32.8477 | 30.7500 | 30.5949, 0.5373, 71.629 | 30.5938 | 0.225 / 0.040 / 0.045 | 32.8438; 0.425 / 0.075 / 0.085 |
| S4_128_s8 | A2.11 | 34.00 | 1.00 | 37.3477 | 34.7734 | -, -, - | - | - / - / - | 37.3438; - / - / - |
| S4_128_s16 | A1.0 | 0.00 | 0.50 | 11.5547 | 11.7500 | 25.4293, 15.2844, 2.94e-06 | 11.5469 | 0.750 / 1.062 / 1.219 | 11.5508; 0.987 / 1.157 / 1.327 |
| S4_128_s16 | A1.1 | 0.00 | 1.00 | 11.8164 | 11.7500 | 25.4293, 15.3254, 5.60e-06 | 11.8125 | 0.750 / 1.021 / 1.172 | 11.8125; 0.989 / 1.312 / 1.506 |
| S4_128_s16 | A1.2 | 0.00 | 1.50 | 11.9492 | 11.7500 | 25.4293, 15.3945, 8.05e-06 | 11.9375 | 0.751 / 0.985 / 1.130 | 11.9453; 0.991 / 1.334 / 1.531 |
| S4_128_s16 | A1.3 | 0.00 | 2.00 | 12.0430 | 11.7500 | 25.4293, 15.4925, 1.04e-05 | 12.0312 | 0.752 / 0.952 / 1.093 | 12.0391; 0.994 / 1.340 / 1.539 |
| S4_128_s16 | A1.4 | 0.00 | 3.00 | 12.1719 | 11.7500 | 25.4293, 15.7698, 1.50e-05 | 12.1562 | 0.756 / 0.893 / 1.025 | 12.1680; 0.995 / 1.266 / 1.454 |
| S4_128_s16 | A1.5 | 0.00 | 4.00 | 12.2773 | 11.7500 | 25.4293, 16.1293, 2.03e-05 | 12.2656 | 0.736 / 0.824 / 0.946 | 12.2734; 0.949 / 1.159 / 1.331 |
| S4_128_s16 | A2.6 | 14.00 | 1.00 | 22.9688 | 14.2500 | 25.4300, 5.2812, 0.041 | 22.9375 | 0.509 / 0.283 / 0.323 | 22.9648; 0.968 / 0.369 / 0.424 |
| S4_128_s16 | A2.7 | 18.00 | 1.00 | 24.8672 | - | 25.4360, 4.0855, 0.376 | 24.8125 | 0.445 / 0.237 / 0.270 | 24.8633; 0.993 / 0.437 / 0.499 |
| S4_128_s16 | A2.8 | 22.00 | 1.00 | 26.0117 | - | 25.4759, 2.1560, 2.523 | 25.4375 | 0.298 / 0.160 / 0.182 | 26.0078; 0.909 / 0.489 / 0.555 |
| S4_128_s16 | A2.9 | 26.00 | 1.00 | 31.6914 | - | 27.0551, 0.8150, 34.313 | 27.0000 | 0.150 / 0.025 / 0.029 | 31.6875; 0.356 / 0.059 / 0.068 |
| S4_128_s16 | A2.10 | 30.00 | 1.00 | 34.9414 | - | 30.5351, 0.4416, 81.825 | 30.5000 | 0.105 / 0.019 / 0.022 | 34.9375; 0.173 / 0.032 / 0.036 |
| S4_128_s16 | A2.11 | 34.00 | 1.00 | 38.1562 | - | -, -, - | - | - / - / - | 38.1523; - / - / - |
| S4_128_s4 | A1.0 | 0.00 | 0.50 | 11.2617† | 11.0625† | -, -, - | - | - / - / - | 3.5625; - / - / - |
| S4_128_s4 | A1.1 | 0.00 | 1.00 | 11.3750† | 11.0625† | -, -, - | - | - / - / - | 3.5625; - / - / - |
| S4_128_s4 | A1.2 | 0.00 | 1.50 | 11.4219† | 11.0625† | -, -, - | - | - / - / - | 3.5625; - / - / - |
| S4_128_s4 | A1.3 | 0.00 | 2.00 | 11.4531† | 11.0625† | -, -, - | - | - / - / - | 3.5625; - / - / - |
| S4_128_s4 | A1.4 | 0.00 | 3.00 | 11.4922† | 11.0625† | 0.8792, 0.8192, 337.804 | 0.8750 | 0.254 / 0.132 / 0.150 | 3.5625; 0.254 / 0.132 / 0.150 |
| S4_128_s4 | A1.5 | 0.00 | 4.00 | 11.5234† | 11.0625† | 1.8097, 1.7917, 399.342 | 1.7969 | 0.371 / 0.185 / 0.211 | 3.5625; 0.371 / 0.185 / 0.211 |
| S4_128_s4 | A2.6 | 14.00 | 1.00 | 17.0469† | 14.4805† | -, -, - | - | - / - / - | -; - / - / - |
| ISO_128_s4 | A1.0 | 0.00 | 0.50 | 11.8320 | 11.5625 | 4.8957, 4.0145, 33.956 | 4.8906 | 0.404 / 0.277 / 0.316 | 11.8281; 0.407 / 0.279 / 0.318 |
| ISO_128_s4 | A1.1 | 0.00 | 1.00 | 11.9648 | 11.5625 | 5.1850, 4.2880, 63.026 | 5.1719 | 0.430 / 0.280 / 0.319 | 11.9609; 0.431 / 0.281 / 0.320 |
| ISO_128_s4 | A1.2 | 0.00 | 1.50 | 12.0234† | 11.5625 | 5.4618, 4.5738, 89.681 | 5.4531 | 0.456 / 0.283 / 0.323 | 12.0000; 0.456 / 0.283 / 0.323 |
| ISO_128_s4 | A1.3 | 0.00 | 2.00 | 12.0586† | 11.5625 | 5.7232, 4.8704, 113.835 | 5.7188 | 0.481 / 0.286 / 0.326 | 12.0000; 0.481 / 0.286 / 0.326 |
| ISO_128_s4 | A1.4 | 0.00 | 3.00 | - | 11.5625 | 6.1699, 5.4784, 152.971 | 6.1562 | 0.528 / 0.289 / 0.329 | 12.0000; 0.528 / 0.289 / 0.329 |
| ISO_128_s4 | A1.5 | 0.00 | 4.00 | - | 11.5625 | 6.5672, 6.1338, 185.362 | 6.5625 | 0.524 / 0.268 / 0.305 | 12.0000; 0.524 / 0.268 / 0.305 |
| ISO_128_s4_plain | A1.0 | 0.00 | 0.50 | 11.8398† | 11.5625† | -, -, - | - | - / - / - | 3.2500; - / - / - |
| ISO_128_s4_plain | A1.1 | 0.00 | 1.00 | 11.9688† | 11.5625† | -, -, - | - | - / - / - | 3.2500; - / - / - |
| ISO_128_s4_plain | A1.2 | 0.00 | 1.50 | 12.0234† | 11.5625† | -, -, - | - | - / - / - | 3.2500; - / - / - |
| ISO_128_s4_plain | A1.3 | 0.00 | 2.00 | 12.0586† | 11.5625† | -, -, - | - | - / - / - | 3.2500; - / - / - |
| ISO_128_s4_plain | A1.4 | 0.00 | 3.00 | 12.0977† | 11.5625† | 0.7087, 0.6600, 347.015 | 0.7031 | 0.225 / 0.117 / 0.134 | 3.2500; 0.225 / 0.117 / 0.134 |
| ISO_128_s4_plain | A1.5 | 0.00 | 4.00 | 12.1211† | 11.5625† | 1.6470, 1.6341, 408.946 | 1.6406 | 0.356 / 0.177 / 0.202 | 3.2500; 0.356 / 0.177 / 0.202 |
| M3c_s4_96 | A1.0 | 0.00 | 0.50 | 17.1719 | 21.1667 | 13.2821, 10.7042, 3.935 | 13.2708 | 0.679 / 0.454 / 0.519 | 17.1667; 0.796 / 0.532 / 0.608 |
| M3c_s4_96 | A1.1 | 0.00 | 1.00 | 18.6927 | 21.1667 | 13.3436, 10.7866, 8.195 | 13.3333 | 0.682 / 0.440 / 0.503 | 18.6875; 0.747 / 0.481 / 0.550 |
| M3c_s4_96 | A1.2 | 0.00 | 1.50 | 19.3958 | 21.1667 | 13.4050, 10.8926, 12.369 | 13.3958 | 0.686 / 0.427 / 0.489 | 19.3906; 0.731 / 0.455 / 0.521 |
| M3c_s4_96 | A1.3 | 0.00 | 2.00 | 19.8125 | 21.1667 | 13.4623, 11.0194, 16.199 | 13.4583 | 0.691 / 0.416 / 0.475 | 19.8073; 0.725 / 0.436 / 0.499 |
| M3c_s4_96 | A1.4 | 0.00 | 3.00 | 20.2917 | 21.1667 | 13.5632, 11.3406, 22.777 | 13.5625 | 0.700 / 0.395 / 0.452 | 20.2865; 0.724 / 0.408 / 0.467 |
| M3c_s4_96 | A1.5 | 0.00 | 4.00 | 20.5885 | 21.1667 | 13.6536, 11.7540, 28.517 | 13.6458 | 0.679 / 0.363 / 0.415 | 20.5833; 0.697 / 0.372 / 0.426 |
| M3c_s4_96 | A2.6 | 14.00 | 1.00 | 21.8490 | 21.1667 | 14.5790, 0.4609, 79.687 | 14.5625 | 0.098 / 0.039 / 0.044 | 21.8438; 0.098 / 0.039 / 0.045 |
| M3c_s4_96 | A2.7 | 18.00 | 1.00 | - | 21.1667 | -, -, - | - | - / - / - | 22.0833; - / - / - |
| P64_s4_B2 | A1.0 | 0.00 | 0.50 | 11.2109 | 12.1250 | 10.1436, 8.4027, 2.592 | 10.1250 | 0.582 / 0.326 / 0.370 | 11.2031; 0.913 / 0.512 / 0.581 |
| P64_s4_B2 | A1.1 | 0.00 | 1.00 | 11.7812 | 12.1250 | 10.2006, 8.4901, 5.327 | 10.1875 | 0.587 / 0.317 / 0.359 | 11.7734; 0.883 / 0.477 / 0.541 |
| P64_s4_B2 | A1.2 | 0.00 | 1.50 | 12.0469 | 12.1250 | 10.2615, 8.6071, 8.113 | 10.2500 | 0.592 / 0.309 / 0.351 | 12.0391; 0.876 / 0.457 / 0.519 |
| P64_s4_B2 | A1.3 | 0.00 | 2.00 | 12.2109 | 12.1250 | 10.3213, 8.7502, 10.719 | 10.3125 | 0.598 / 0.302 / 0.343 | 12.2031; 0.875 / 0.442 / 0.502 |
| P64_s4_B2 | A1.4 | 0.00 | 3.00 | 12.3984 | 12.1250 | 10.4308, 9.1063, 15.169 | 10.4062 | 0.610 / 0.290 / 0.329 | 12.3906; 0.879 / 0.417 / 0.474 |
| P64_s4_B2 | A1.5 | 0.00 | 4.00 | 12.5156 | 12.1250 | 10.5356, 9.5445, 19.057 | 10.5312 | 0.591 / 0.265 / 0.301 | 12.5078; 0.839 / 0.376 / 0.427 |
| P64_s4_B2 | A2.6 | 14.00 | 1.00 | 17.6953 | 14.2344 | 14.3532, 0.3491, 42.236 | 14.3516 | 0.073 / 0.004 / 0.004 | 17.6875; 0.275 / 0.014 / 0.016 |
| P64_s4_B2 | A2.7 | 18.00 | 1.00 | 20.7656 | 18.8750 | 18.5888, 0.4771, 36.529 | 18.5000 | 0.118 / 0.013 / 0.015 | 20.7578; 0.404 / 0.045 / 0.052 |
| P64_s4_B2 | A2.8 | 22.00 | 1.00 | 25.8359 | 22.5312 | 22.5435, 0.4826, 36.774 | 22.5391 | 0.141 / 0.016 / 0.018 | 25.8281; 0.368 / 0.041 / 0.047 |
| P64_s4_B2 | A2.9 | 26.00 | 1.00 | 28.7031 | 26.5625 | 26.5417, 0.4880, 36.719 | 26.5391 | 0.176 / 0.022 / 0.025 | 28.6953; 0.442 / 0.056 / 0.063 |
| P64_s4_B2 | A2.10 | 30.00 | 1.00 | 33.1562 | 30.5547 | 30.4259, 0.3942, 42.772 | 30.4219 | 0.175 / 0.026 / 0.029 | 33.1484; 0.314 / 0.047 / 0.052 |
| P64_s4_B2 | A2.11 | 34.00 | 1.00 | 37.4688 | 34.5469 | -, -, - | - | - / - / - | 37.4609; - / - / - |
| P64_s4_B2 | A2.12 | 50.00 | 1.00 | 52.8984† | 50.5703† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_B2 | A2.13 | 70.00 | 1.00 | 73.0625† | 70.5547† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_B2 | A2.14 | 90.00 | 1.00 | 92.9688† | 90.5625† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.0 | 0.00 | 0.50 | 11.2109 | 12.1250 | 10.1436, 8.4027, 2.592 | 10.1250 | 0.582 / 0.326 / 0.370 | 11.2031; 0.913 / 0.512 / 0.581 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.1 | 0.00 | 1.00 | 11.7812 | 12.1250 | 10.2006, 8.4901, 5.327 | 10.1875 | 0.587 / 0.317 / 0.359 | 11.7734; 0.883 / 0.477 / 0.541 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.2 | 0.00 | 1.50 | 12.0469 | 12.1250 | 10.2615, 8.6071, 8.113 | 10.2500 | 0.592 / 0.309 / 0.351 | 12.0391; 0.876 / 0.457 / 0.519 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.3 | 0.00 | 2.00 | 12.2109 | 12.1250 | 10.3213, 8.7502, 10.719 | 10.3125 | 0.598 / 0.302 / 0.343 | 12.2031; 0.875 / 0.442 / 0.502 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.4 | 0.00 | 3.00 | 12.3984 | 12.1250 | 10.4308, 9.1063, 15.169 | 10.4062 | 0.610 / 0.290 / 0.329 | 12.3906; 0.879 / 0.417 / 0.474 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.5 | 0.00 | 4.00 | 12.5156 | 12.1250 | 10.5356, 9.5445, 19.057 | 10.5312 | 0.591 / 0.265 / 0.301 | 12.5078; 0.839 / 0.376 / 0.427 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.6 | 14.00 | 1.00 | 17.6797 | 14.2266 | 14.3447, 0.3485, 42.277 | 14.3438 | 0.071 / 0.003 / 0.004 | 17.6719; 0.270 / 0.013 / 0.015 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.7 | 18.00 | 1.00 | 20.7656 | 18.8750 | 18.5811, 0.4772, 36.520 | 18.5000 | 0.117 / 0.013 / 0.015 | 20.7578; 0.402 / 0.045 / 0.051 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.8 | 22.00 | 1.00 | 25.8281 | 22.5234 | 22.5357, 0.4825, 36.779 | 22.5312 | 0.139 / 0.015 / 0.017 | 25.8203; 0.364 / 0.040 / 0.045 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.9 | 26.00 | 1.00 | 28.6953 | 26.5547 | 26.5339, 0.4880, 36.715 | 26.5312 | 0.174 / 0.022 / 0.025 | 28.6875; 0.438 / 0.055 / 0.062 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.10 | 30.00 | 1.00 | 33.1484 | 30.5469 | 30.4187, 0.3947, 42.743 | 30.4141 | 0.174 / 0.026 / 0.029 | 33.1406; 0.313 / 0.046 / 0.052 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.11 | 34.00 | 1.00 | 37.4531 | 34.5391 | -, -, - | - | - / - / - | 37.4453; - / - / - |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.12 | 50.00 | 1.00 | 52.8906† | 50.5625† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.13 | 70.00 | 1.00 | 73.0547† | 70.5469† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.14 | 90.00 | 1.00 | 92.9297† | 90.5469† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.15 | 120.00 | 1.00 | 123.0156† | 120.5312† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.16 | 160.00 | 1.00 | 163.1016† | 160.5469† | -, -, - | - | - / - / - | -; - / - / - |
| P64_s4_heldsrc | A1.0 | 0.00 | 0.50 | 11.2109 | 12.1250 | 32.9477, 19.9683, 0.001 | 11.1875 | 0.792 / 1.313 / 1.509 | 11.2031; 0.979 / 0.648 / 0.738 |
| P64_s4_heldsrc | A1.1 | 0.00 | 1.00 | 11.7734 | 12.1250 | 32.9478, 20.0117, 0.003 | 11.7500 | 0.792 / 1.274 / 1.464 | 11.7656; 0.989 / 1.313 / 1.506 |
| P64_s4_heldsrc | A1.2 | 0.00 | 1.50 | 12.0469 | 12.1250 | 32.9478, 20.0830, 0.005 | 12.0312 | 0.793 / 1.233 / 1.417 | 12.0391; 0.974 / 1.225 / 1.405 |
| P64_s4_heldsrc | A1.3 | 0.00 | 2.00 | 12.2031 | 12.1250 | 32.9478, 20.1823, 0.007 | 12.1875 | 0.794 / 1.194 / 1.372 | 12.1953; 0.967 / 1.151 / 1.320 |
| P64_s4_heldsrc | A1.4 | 0.00 | 3.00 | 12.3906 | 12.1250 | 32.9479, 20.4607, 0.010 | 12.3750 | 0.796 / 1.123 / 1.290 | 12.3828; 0.989 / 1.578 / 1.814 |
| P64_s4_heldsrc | A1.5 | 0.00 | 4.00 | 12.5156 | 12.1250 | 32.9480, 20.8276, 0.014 | 12.5000 | 0.779 / 1.047 / 1.203 | 12.5078; 0.937 / 1.388 / 1.596 |
| P64_s4_heldsrc | A2.6 | 14.00 | 1.00 | - | 14.2812 | 33.1288, 10.0500, 8.248 | 33.1250 | 0.662 / 0.461 / 0.527 | 40.0000; 0.662 / 0.461 / 0.527 |
| P64_s4_heldsrc | A2.7 | 18.00 | 1.00 | - | - | 33.5409, 9.1491, 27.122 | 33.5000 | 0.655 / 0.446 / 0.509 | 40.0000; 0.655 / 0.446 / 0.509 |
| P64_s4_heldsrc | A2.8 | 22.00 | 1.00 | - | - | 33.7325, 7.4642, 35.947 | 33.6250 | 0.614 / 0.427 / 0.486 | 40.0000; 0.614 / 0.427 / 0.486 |
| P64_s4_heldsrc | A2.9 | 26.00 | 1.00 | - | - | 33.9556, 5.3428, 46.255 | 33.8750 | 0.541 / 0.388 / 0.440 | 40.0000; 0.541 / 0.388 / 0.440 |
| P64_s4_heldsrc | A2.10 | 30.00 | 1.00 | - | - | 34.2131, 2.9630, 58.202 | 34.1250 | 0.405 / 0.303 / 0.342 | 40.0000; 0.405 / 0.303 / 0.342 |
| P64_s4_heldsrc | A2.11 | 34.00 | 1.00 | - | - | 34.3335, 0.2426, 63.789 | 34.2500 | 0.054 / 0.041 / 0.046 | 40.0000; 0.054 / 0.041 / 0.046 |
| P64_s4_widened | A1.0 | 0.00 | 0.50 | 11.2109 | 12.2500 | 19.0489, 11.7096, 4.36e-04 | 11.1875 | 0.657 / 1.062 / 1.216 | 11.2031; 0.970 / 0.666 / 0.759 |
| P64_s4_widened | A1.1 | 0.00 | 1.00 | 11.8672 | 12.2500 | 19.0490, 11.7531, 9.34e-04 | 11.8438 | 0.658 / 1.016 / 1.163 | 11.8594; 0.985 / 1.460 / 1.671 |
| P64_s4_widened | A1.2 | 0.00 | 1.50 | 12.1953 | 12.2500 | 19.0490, 11.8245, 0.001 | 12.1875 | 0.659 / 0.970 / 1.111 | 12.1875; 0.989 / 1.519 / 1.739 |
| P64_s4_widened | A1.3 | 0.00 | 2.00 | 12.3906 | 12.2500 | 19.0490, 11.9239, 0.002 | 12.3750 | 0.661 / 0.927 / 1.062 | 12.3828; 0.992 / 1.477 / 1.690 |
| P64_s4_widened | A1.4 | 0.00 | 3.00 | 12.6250 | 12.2500 | 19.0491, 12.2030, 0.003 | 12.5938 | 0.666 / 0.851 / 0.975 | 12.6172; 0.993 / 1.356 / 1.553 |
| P64_s4_widened | A1.5 | 0.00 | 4.00 | 12.7891 | 12.2500 | 19.0491, 12.5707, 0.004 | 12.7812 | 0.646 / 0.771 / 0.883 | 12.7812; 0.950 / 1.222 / 1.399 |
| P64_s4_widened | A2.6 | 14.00 | 1.00 | 19.8125 | 14.1875 | 19.1216, 1.7802, 1.489 | 19.0000 | 0.226 / 0.037 / 0.042 | 19.8047; 0.885 / 0.144 / 0.165 |
| P64_s4_widened | A2.7 | 18.00 | 1.00 | 22.9922 | 24.0000 | 19.9062, 1.0334, 14.995 | 19.8750 | 0.150 / 0.016 / 0.019 | 22.9844; 0.491 / 0.053 / 0.061 |
| P64_s4_widened | A2.8 | 22.00 | 1.00 | 26.4922 | 24.0000 | 22.6953, 0.5751, 31.931 | 22.6250 | 0.102 / 0.010 / 0.012 | 26.4844; 0.328 / 0.033 / 0.037 |
| P64_s4_widened | A2.9 | 26.00 | 1.00 | 30.6797 | - | 26.6297, 0.5366, 33.063 | 26.6250 | 0.118 / 0.012 / 0.013 | 30.6719; 0.281 / 0.028 / 0.031 |
| P64_s4_widened | A2.10 | 30.00 | 1.00 | 34.8438 | - | 30.3903, 0.3431, 45.419 | 30.3750 | 0.101 / 0.015 / 0.017 | 34.8359; 0.184 / 0.027 / 0.031 |
| P64_s4_widened | A2.11 | 34.00 | 1.00 | 38.8047 | - | -, -, - | - | - / - / - | 38.7969; - / - / - |

**Table H4. Blueshifted arrivals and tick-map flags** (reviewer A rzchk 2(d), 7.4; reviewer B rzchk3 sec. 5): per member, over arrivals inside the solver-certified record interval at 4 r_s, per class: arrivals blueshifted in the coordinate-clock sense (L_T < 0, G_D > 1) and for the static receiver (L_D < 0, g_D > 1), with the largest −L_T. Tick-map flags on the backward map (A1, A2): monotonicity reversals (consecutive arrivals 1 a apart whose τ_e decreases); bunched pairs (consecutive one-sided slopes dτ_e/dT_D differing by more than a factor 2, the larger above 1). These are flags; no cause (caustic or otherwise) is assigned. Clocks with either H3 diagnostic above 1, on the unmasked and on the resolved domain.

| member | A1: blueshifted L_T<0 / L_D<0, largest −L_T | A2: same | B: same | C: same | backward: monotonicity reversals / bunched pairs | clocks with ∫G_D dτ_e/Δτ > 1 / ∫g_D dτ_e/Δτ > 1: unmasked; resolved |
|---|---|---|---|---|---|---|
| P64_s4 | 0 / 0, -0.357 | 0 / 0, -1.534 | 0 / 0, -0.720 | 72 / 77, 3.181 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s2 | 0 / 0, -0.360 | 0 / 0, -1.338 | 0 / 0, -0.691 | 66 / 89, 2.404 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s8 | 277 / 332, 1.819 | 0 / 0, -0.353 | 8 / 10, 0.748 | 80 / 87, 2.725 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s4_tau4 | 0 / 0, -0.358 | 0 / 0, -1.508 | 0 / 0, -0.722 | 94 / 96, 2.605 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s4_tau2 | 0 / 0, -0.358 | 0 / 0, - | 0 / 0, - | 0 / 0, -0.156 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s4_newton | 0 / 0, -0.357 | 0 / 0, - | 0 / 0, - | 0 / 0, -0.158 | 0 / 0 | 0 / 0; 0 / 0 |
| W2 | 93 / 115, 4.305 | 0 / 0, -1.130 | 50 / 52, 3.548 | 58 / 60, 4.364 | 0 / 12 | 6 / 6; 5 / 5 |
| ISO_64_s4 | 0 / 0, -0.324 | 0 / 0, - | 0 / 0, -0.766 | 0 / 0, -0.158 | 0 / 0 | 0 / 0; 0 / 0 |
| S4_128_s8 | 155 / 176, 3.946 | 0 / 0, -1.352 | 7 / 11, 1.832 | 109 / 113, 4.654 | 0 / 24 | 6 / 6; 6 / 6 |
| S4_128_s16 | 871 / 1020, 2.541 | 175 / 195, 0.586 | 7 / 11, 0.629 | 158 / 186, 3.194 | 0 / 0 | 2 / 5; 6 / 6 |
| S4_128_s4 | 0 / 0, -0.644 | 0 / 0, - | 0 / 0, - | 0 / 0, -0.157 | 0 / 0 | 0 / 0; 0 / 0 |
| ISO_128_s4 | 0 / 0, -0.339 | 0 / 0, - | 0 / 0, - | 0 / 0, -0.158 | 0 / 0 | 0 / 0; 0 / 0 |
| ISO_128_s4_plain | 0 / 0, -0.635 | 0 / 0, - | 0 / 0, - | 0 / 0, -0.158 | 0 / 0 | 0 / 0; 0 / 0 |
| M3c_s4_96 | 0 / 0, -0.313 | 0 / 0, -0.910 | 0 / 0, - | 0 / 63, -0.099 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s4_B2 | 0 / 0, -0.357 | 0 / 0, -1.534 | 0 / 0, -0.720 | 72 / 77, 3.180 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | 0 / 0, -0.357 | 0 / 0, -1.540 | 0 / 0, -0.720 | 72 / 77, 3.181 | 0 / 0 | 0 / 0; 0 / 0 |
| P64_s4_heldsrc | 246 / 278, 3.149 | 7 / 22, 0.020 | 16 / 19, 2.171 | 43 / 53, 3.274 | 0 / 0 | 6 / 6; 5 / 5 |
| P64_s4_widened | 179 / 196, 2.905 | 0 / 0, -0.987 | 37 / 40, 2.546 | 83 / 89, 3.455 | 0 / 6 | 2 / 4; 5 / 5 |

**Table H5. Post-gate shares on the flights inside the solver-certified record interval** (as v2): the share of each flight after this pass's ONE-TERM gate T_gate (σ max|∂_r v̄| > 0.1 on the back-averaged site fields); the six-term and ‖K‖_op gates are not applied here.

| member | T_b/r_s | T_gate/r_s (one-term) | one-term gate before the first root? | A1 median / max | A2 median / max | B median / max | C median / max |
|---|---|---|---|---|---|---|---|
| P64_s4 | 12.1250 | 11.7500 | yes | 0.852 / 0.959 | 1.000 / 1.000 | 1.000 / 1.000 | 0.580 / 1.000 |
| P64_s2 | 12.0000 | 12.2500 | no | 0.811 / 0.940 | 1.000 / 1.000 | 1.000 / 1.000 | 0.684 / 1.000 |
| P64_s8 | 13.0000 | 10.8750 | yes | 0.943 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| P64_s4_tau4 | 12.0000 | 11.6250 | yes | 0.853 / 0.968 | 1.000 / 1.000 | 1.000 / 1.000 | 0.166 / 1.000 |
| P64_s4_tau2 | 11.9375 | 11.5312 | yes | 0.000 / 0.209 | - / - | - / - | 0.000 / 0.367 |
| P64_s4_newton | 12.1250 | 11.7500 | yes | 0.000 / 0.131 | - / - | - / - | 0.000 / 0.212 |
| W2 | 11.8750 | 11.5000 | yes | 0.854 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.526 / 1.000 |
| ISO_64_s4 | 12.3750 | 12.1250 | yes | 0.000 / 0.226 | - / - | 0.637 / 0.637 | 0.000 / 0.366 |
| S4_128_s8 | 11.1875 | 10.7500 | yes | 0.899 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.430 / 1.000 |
| S4_128_s16 | 11.7500 | 10.2500 | yes | 0.951 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| S4_128_s4 | 11.0625 | 11.1250 | no | 0.000 / 0.000 | - / - | - / - | 0.000 / 0.000 |
| ISO_128_s4 | 11.5625 | 11.6875 | no | 0.000 / 0.058 | - / - | - / - | 0.000 / 0.080 |
| ISO_128_s4_plain | 11.5625 | 11.6875 | no | 0.000 / 0.000 | - / - | - / - | 0.000 / 0.000 |
| M3c_s4_96 | 21.1667 | 21.5833 | no | 0.000 / 0.059 | 0.000 / 0.067 | - / - | 0.000 / 0.100 |
| P64_s4_B2 | 12.1250 | 11.7500 | yes | 0.852 / 0.959 | 1.000 / 1.000 | 1.000 / 1.000 | 0.580 / 1.000 |
| P64_s4_ext200 | 12.1250 | 11.7500 | yes | 0.852 / 0.959 | 1.000 / 1.000 | 1.000 / 1.000 | 0.580 / 1.000 |
| P64_s4_heldsrc | 12.1250 | 11.8750 | yes | 0.842 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.996 / 1.000 |
| P64_s4_widened | 12.2500 | 11.7500 | yes | 0.846 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.765 / 1.000 |

**Table H6. Largest received L_T over all classes, with the class named** (as v2; reviewer B rzchk3 sec. 2.1): inside the solver-certified record interval, unmasked and resolved, with the L_D of the same event and the independent L_D maximum; and over the whole stored window.

| member | inside: max L_T (class) / L_D same event | inside, resolved: max L_T (class) / L_D same event | inside: max L_D (class) | inside, resolved: max L_D (class) | whole window: max L_T (class) |
|---|---|---|---|---|---|
| P64_s4 | 10.456 (A1) / 10.338 | 10.456 (A1) / 10.338 | 10.338 (A1) | 10.338 (A1) | 10.456 (A1) |
| P64_s2 | 15.676 (A1) / 15.569 | 15.676 (A1) / 15.569 | 15.569 (A1) | 15.569 (A1) | 15.676 (A1) |
| P64_s8 | 3.708 (B) / 3.582 | 3.708 (B) / 3.582 | 3.582 (B) | 3.582 (B) | 3.708 (B) |
| P64_s4_tau4 | 7.184 (A2) / 7.059 | 7.184 (A2) / 7.059 | 7.059 (A2) | 7.059 (A2) | 7.184 (A2) |
| P64_s4_tau2 | 0.745 (A1) / 0.617 | 0.745 (A1) / 0.617 | 0.617 (A1) | 0.617 (A1) | 2.792 (C) †, arriving at T/r_s 21.165 past the admitted interval |
| P64_s4_newton | 0.724 (A1) / 0.598 | 0.724 (A1) / 0.598 | 0.598 (A1) | 0.598 (A1) | 0.724 (A1) |
| W2 | 4.877 (B) / 4.741 | 4.877 (B) / 4.741 | 4.741 (B) | 4.741 (B) | 4.877 (B) |
| ISO_64_s4 | 0.766 (B) / 0.646 | 0.766 (B) / 0.646 | 0.646 (B) | 0.646 (B) | 9.987 (A1) †, arriving at T/r_s 40.000 past the admitted interval |
| S4_128_s8 | 7.754 (A2) / 7.632 | 7.754 (A2) / 7.632 | 7.632 (A2) | 7.632 (A2) | 7.754 (A2) |
| S4_128_s16 | 3.654 (B) / 3.527 | 3.654 (B) / 3.527 | 3.527 (B) | 3.527 (B) | 3.654 (B) |
| S4_128_s4 | 0.701 (A1) / 0.571 | 0.701 (A1) / 0.571 | 0.571 (A1) | 0.571 (A1) | 2.096 (C) †, arriving at T/r_s 18.757 past the admitted interval |
| ISO_128_s4 | 0.703 (A1) / 0.572 | 0.703 (A1) / 0.572 | 0.572 (A1) | 0.572 (A1) | 0.719 (B) †, arriving at T/r_s 12.018 past the admitted interval |
| ISO_128_s4_plain | 0.703 (A1) / 0.572 | 0.703 (A1) / 0.572 | 0.572 (A1) | 0.572 (A1) | 0.898 (B) †, arriving at T/r_s 14.295 past the admitted interval |
| M3c_s4_96 | 0.922 (A2) / 0.792 | 0.922 (A2) / 0.792 | 0.793 (A2) | 0.793 (A2) | 0.922 (A2) |
| P64_s4_B2 | 10.456 (A1) / 10.338 | 10.456 (A1) / 10.338 | 10.338 (A1) | 10.338 (A1) | 49.833 (A1) †, arriving at T/r_s 100.000 past the admitted interval |
| P64_s4_ext200 (e, c only: ray mask unavailable) | 10.456 (A1) / 10.338 | 10.456 (A1) / 10.338 | 10.338 (A1) | 10.338 (A1) | 117.420 (A1) †, arriving at T/r_s 200.000 past the admitted interval; unreproducible: records not locatable |
| P64_s4_heldsrc | 3.803 (B) / 3.681 | 3.803 (B) / 3.681 | 3.681 (B) | 3.681 (B) | 3.803 (B) |
| P64_s4_widened | 4.239 (B) / 4.111 | 4.239 (B) / 4.111 | 4.111 (B) | 4.111 (B) | 4.239 (B) |

**Table H7. The dust emitters' validity** (as v2).

| member | quantile | launches | timelike | rejected | median abs Δ ln D_e | max abs Δ ln D_e |
|---|---|---|---|---|---|---|
| P64_s4 | R_50 | 320 | 167 | 153 | 0.005 | 0.658 |
| P64_s4 | R_90 | 320 | 241 | 79 | 0.016 | 2.261 |
| P64_s2 | R_50 | 320 | 101 | 219 | 0.003 | 0.528 |
| P64_s2 | R_90 | 320 | 281 | 39 | 0.021 | 1.021 |
| P64_s8 | R_50 | 320 | 272 | 48 | 0.006 | 0.408 |
| P64_s8 | R_90 | 320 | 243 | 77 | 0.011 | 1.100 |
| P64_s4_tau4 | R_50 | 640 | 372 | 268 | 0.004 | 0.695 |
| P64_s4_tau4 | R_90 | 640 | 419 | 221 | 0.006 | 1.799 |
| P64_s4_tau2 | R_50 | 688 | 553 | 135 | 8.66e-04 | 2.676 |
| P64_s4_tau2 | R_90 | 688 | 477 | 211 | 9.11e-04 | 0.484 |
| P64_s4_newton | R_50 | 100 | 100 | 0 | 0.002 | 0.034 |
| P64_s4_newton | R_90 | 100 | 100 | 0 | 0.002 | 0.077 |
| W2 | R_50 | 320 | 210 | 110 | 0.011 | 0.941 |
| W2 | R_90 | 320 | 244 | 76 | 0.008 | 1.599 |
| ISO_64_s4 | R_50 | 320 | 170 | 150 | 0.004 | 0.818 |
| ISO_64_s4 | R_90 | 320 | 209 | 111 | 0.017 | 1.415 |
| S4_128_s8 | R_50 | 640 | 354 | 286 | 0.007 | 1.875 |
| S4_128_s8 | R_90 | 640 | 452 | 188 | 0.013 | 1.407 |
| S4_128_s16 | R_50 | 640 | 538 | 102 | 0.008 | 0.246 |
| S4_128_s16 | R_90 | 640 | 433 | 207 | 0.009 | 0.745 |
| S4_128_s4 | R_50 | 305 | 209 | 96 | 0.002 | 2.636 |
| S4_128_s4 | R_90 | 305 | 228 | 77 | 0.002 | 0.299 |
| ISO_128_s4 | R_50 | 193 | 193 | 0 | 0.001 | 0.203 |
| ISO_128_s4 | R_90 | 193 | 193 | 0 | 0.001 | 1.148 |
| ISO_128_s4_plain | R_50 | 230 | 194 | 36 | 0.001 | 0.200 |
| ISO_128_s4_plain | R_90 | 230 | 195 | 35 | 0.001 | 1.193 |
| M3c_s4_96 | R_50 | 265 | 265 | 0 | 6.15e-04 | 0.009 |
| M3c_s4_96 | R_90 | 265 | 265 | 0 | 9.61e-04 | 0.022 |
| P64_s4_B2 | R_50 | 800 | 167 | 633 | 0.005 | 0.658 |
| P64_s4_B2 | R_90 | 800 | 677 | 123 | 0.024 | 2.261 |
| P64_s4_ext200 | R_50 | 1600 | 167 | 1433 | 0.005 | 0.658 |
| P64_s4_ext200 | R_90 | 1600 | 1478 | 122 | 0.020 | 2.261 |
| P64_s4_heldsrc | R_50 | 320 | 294 | 26 | 0.002 | 1.241 |
| P64_s4_heldsrc | R_90 | 320 | 242 | 78 | 0.002 | 0.179 |
| P64_s4_widened | R_50 | 320 | 266 | 54 | 0.017 | 0.879 |
| P64_s4_widened | R_90 | 320 | 267 | 53 | 0.009 | 1.327 |

**Table H8. ln A_D and the transfer beyond 4 r_s** (as v2, with the breakdown behind reviewer A rzchk3 item 9): ln A_D median over arrivals inside the solver-certified record interval, backward (A) / forward (B, C); then, per receiver beyond 4 r_s and per launch class, on forward launches received at both 4 r_s and that receiver inside the interval: count and largest abs(ln G_D(r) − ln G_D(4 r_s)). v2's column is the outermost receiver over B and C; v1's exterior check took every forward front (A, B, C) reaching every receiver ≥ 4 r_s.

| member | ln A_D median: backward A / forward B, C | receiver r/r_s | A (forward clock launches): n, max abs | B: n, max abs | C: n, max abs | B and C (v2's column at the outermost): n, max abs |
|---|---|---|---|---|---|---|
| P64_s4 | -0.1242 / -0.1233 | 8.000 | 2108, 0.040 | 1844, 0.041 | 289, 0.040 | 2133, 0.041 |
| P64_s4 | -0.1242 / -0.1233 | 12.000 | 2012, 0.042 | 1153, 0.042 | 257, 0.042 | 1410, 0.042 |
| P64_s4 | -0.1242 / -0.1233 | 16.000 (r_c) | 1934, 0.034 | 296, 0.009 | 192, 0.034 | 488, 0.034 |
| P64_s2 | -0.1140 / -0.1133 | 8.000 | 2210, 0.064 | 2338, 0.029 | 275, 0.064 | 2613, 0.064 |
| P64_s2 | -0.1140 / -0.1133 | 12.000 | 2070, 0.065 | 1455, 0.032 | 240, 0.065 | 1695, 0.065 |
| P64_s2 | -0.1140 / -0.1133 | 16.000 (r_c) | 1934, 0.065 | 354, 0.013 | 192, 0.065 | 546, 0.065 |
| P64_s8 | -0.1301 / -0.1360 | 8.000 | 2672, 0.036 | 3116, 0.036 | 353, 0.036 | 3469, 0.036 |
| P64_s8 | -0.1301 / -0.1360 | 11.891 (r_c) | 2124, 0.038 | 1613, 0.038 | 239, 0.038 | 1852, 0.038 |
| P64_s8 | -0.1301 / -0.1360 | 12.000 | 2112, 0.038 | 1409, 0.038 | 237, 0.038 | 1646, 0.038 |
| P64_s8 | -0.1301 / -0.1360 | 16.000 | 1973, 0.012 | 174, 0.004 | 197, 0.012 | 371, 0.012 |
| P64_s4_tau4 | -0.1304 / -0.1326 | 8.000 | 4237, 0.055 | 4646, 0.055 | 538, 0.055 | 5184, 0.055 |
| P64_s4_tau4 | -0.1304 / -0.1326 | 12.000 | 3947, 0.056 | 2375, 0.056 | 479, 0.056 | 2854, 0.056 |
| P64_s4_tau4 | -0.1304 / -0.1326 | 12.234 (r_c) | 3941, 0.056 | 2221, 0.056 | 479, 0.056 | 2700, 0.056 |
| P64_s4_tau4 | -0.1304 / -0.1326 | 16.000 | 3806, 0.019 | 547, 0.006 | 379, 0.019 | 926, 0.019 |
| P64_s4_tau2 | -0.1310 / -0.1311 | 5.734 (r_c) | 3066, 0.006 | 0, - | 427, 0.006 | 427, 0.006 |
| P64_s4_tau2 | -0.1310 / -0.1311 | 8.000 | 1104, 0.005 | 0, - | 247, 0.004 | 247, 0.004 |
| P64_s4_tau2 | -0.1310 / -0.1311 | 12.000 | 20, 0.003 | 0, - | 0, - | 0, - |
| P64_s4_tau2 | -0.1310 / -0.1311 | 16.000 | 0, - | 0, - | 0, - | 0, - |
| P64_s4_newton | -0.1311 / -0.1311 | 5.750 (r_c) | 731, 0.010 | 0, - | 105, 0.009 | 105, 0.009 |
| P64_s4_newton | -0.1311 / -0.1311 | 8.000 | 250, 0.008 | 0, - | 60, 0.008 | 60, 0.008 |
| P64_s4_newton | -0.1311 / -0.1311 | 12.000 | 1, 0.006 | 0, - | 0, - | 0, - |
| P64_s4_newton | -0.1311 / -0.1311 | 16.000 | 0, - | 0, - | 0, - | 0, - |
| W2 | -0.1312 / -0.1425 | 8.000 | 2536, 0.071 | 2768, 0.071 | 284, 0.071 | 3052, 0.071 |
| W2 | -0.1312 / -0.1425 | 12.000 | 1988, 0.072 | 1638, 0.072 | 244, 0.072 | 1882, 0.072 |
| W2 | -0.1312 / -0.1425 | 16.000 (r_c) | 1874, 0.001 | 258, 0.001 | 188, 0.001 | 446, 0.001 |
| ISO_64_s4 | -0.1311 / -0.1311 | 5.844 (r_c) | 866, 0.026 | 0, - | 115, 0.026 | 115, 0.026 |
| ISO_64_s4 | -0.1311 / -0.1311 | 8.000 | 388, 0.021 | 0, - | 73, 0.020 | 73, 0.020 |
| ISO_64_s4 | -0.1311 / -0.1311 | 12.000 | 17, 0.015 | 0, - | 0, - | 0, - |
| ISO_64_s4 | -0.1311 / -0.1311 | 16.000 | 0, - | 0, - | 0, - | 0, - |
| S4_128_s8 | -0.1276 / -0.1302 | 8.000 | 4974, 0.068 | 5121, 0.068 | 528, 0.063 | 5649, 0.068 |
| S4_128_s8 | -0.1276 / -0.1302 | 12.000 | 4155, 0.068 | 3990, 0.068 | 498, 0.063 | 4488, 0.068 |
| S4_128_s8 | -0.1276 / -0.1302 | 13.188 (r_c) | 3873, 0.068 | 2951, 0.068 | 453, 0.063 | 3404, 0.068 |
| S4_128_s8 | -0.1276 / -0.1302 | 16.000 | 3669, 0.029 | 718, 0.015 | 355, 0.029 | 1073, 0.029 |
| S4_128_s16 | -0.1300 / -0.1355 | 8.000 | 5134, 0.046 | 6625, 0.046 | 617, 0.046 | 7242, 0.046 |
| S4_128_s16 | -0.1300 / -0.1355 | 12.000 | 4982, 0.050 | 4900, 0.050 | 542, 0.050 | 5442, 0.050 |
| S4_128_s16 | -0.1300 / -0.1355 | 13.211 (r_c) | 3920, 0.050 | 3106, 0.050 | 426, 0.050 | 3532, 0.050 |
| S4_128_s16 | -0.1300 / -0.1355 | 16.000 | 3726, 0.009 | 583, 0.003 | 363, 0.009 | 946, 0.009 |
| S4_128_s4 | -0.1303 / -0.1303 | 4.234 (r_c) | 142, 6.13e-05 | 0, - | 39, 6.12e-05 | 39, 6.12e-05 |
| S4_128_s4 | -0.1303 / -0.1303 | 8.000 | 0, - | 0, - | 0, - | 0, - |
| S4_128_s4 | -0.1303 / -0.1303 | 12.000 | 0, - | 0, - | 0, - | 0, - |
| S4_128_s4 | -0.1303 / -0.1303 | 16.000 | 0, - | 0, - | 0, - | 0, - |
| ISO_128_s4 | -0.1304 / -0.1304 | 4.336 (r_c) | 2016, 0.026 | 0, - | 245, 0.023 | 245, 0.023 |
| ISO_128_s4 | -0.1304 / -0.1304 | 8.000 | 406, 0.017 | 0, - | 103, 0.014 | 103, 0.014 |
| ISO_128_s4 | -0.1304 / -0.1304 | 12.000 | 0, - | 0, - | 0, - | 0, - |
| ISO_128_s4 | -0.1304 / -0.1304 | 16.000 | 0, - | 0, - | 0, - | 0, - |
| ISO_128_s4_plain | -0.1305 / -0.1305 | 4.195 (r_c) | 125, 5.15e-05 | 0, - | 33, 5.15e-05 | 33, 5.15e-05 |
| ISO_128_s4_plain | -0.1305 / -0.1305 | 8.000 | 0, - | 0, - | 0, - | 0, - |
| ISO_128_s4_plain | -0.1305 / -0.1305 | 12.000 | 0, - | 0, - | 0, - | 0, - |
| ISO_128_s4_plain | -0.1305 / -0.1305 | 16.000 | 0, - | 0, - | 0, - | 0, - |
| M3c_s4_96 | -0.1340 / -0.1343 | 8.000 | 2488, 0.008 | 0, - | 303, 0.008 | 303, 0.008 |
| M3c_s4_96 | -0.1340 / -0.1343 | 9.771 (r_c) | 1936, 0.008 | 0, - | 254, 0.008 | 254, 0.008 |
| M3c_s4_96 | -0.1340 / -0.1343 | 12.000 | 1246, 0.008 | 0, - | 190, 0.007 | 190, 0.007 |
| M3c_s4_96 | -0.1340 / -0.1343 | 16.000 | 186, 0.008 | 0, - | 68, 0.008 | 68, 0.008 |
| P64_s4_B2 | -0.1242 / -0.1233 | 8.000 | 2108, 0.040 | 1844, 0.041 | 289, 0.040 | 2133, 0.041 |
| P64_s4_B2 | -0.1242 / -0.1233 | 12.000 | 2012, 0.042 | 1153, 0.042 | 257, 0.042 | 1410, 0.042 |
| P64_s4_B2 | -0.1242 / -0.1233 | 16.000 (r_c) | 1934, 0.034 | 296, 0.009 | 192, 0.034 | 488, 0.034 |
| P64_s4_ext200 | -0.1242 / -0.1233 | 8.000 | 2105, 0.041 | 1844, 0.041 | 289, 0.040 | 2133, 0.041 |
| P64_s4_ext200 | -0.1242 / -0.1233 | 12.000 | 2011, 0.042 | 1153, 0.042 | 257, 0.042 | 1410, 0.042 |
| P64_s4_ext200 | -0.1242 / -0.1233 | 16.000 (r_c) | 1934, 0.034 | 296, 0.009 | 192, 0.034 | 488, 0.034 |
| P64_s4_heldsrc | -0.1305 / -0.1374 | 8.000 | 2698, 0.076 | 4292, 0.076 | 322, 0.076 | 4614, 0.076 |
| P64_s4_heldsrc | -0.1305 / -0.1374 | 12.000 | 2529, 0.083 | 2491, 0.083 | 263, 0.082 | 2754, 0.083 |
| P64_s4_heldsrc | -0.1305 / -0.1374 | 16.000 (r_c) | 1934, 0.034 | 305, 0.009 | 192, 0.034 | 497, 0.034 |
| P64_s4_widened | -0.1299 / -0.1321 | 8.000 | 2553, 0.038 | 2563, 0.038 | 327, 0.038 | 2890, 0.038 |
| P64_s4_widened | -0.1299 / -0.1321 | 12.000 | 2005, 0.039 | 1134, 0.039 | 245, 0.039 | 1379, 0.039 |
| P64_s4_widened | -0.1299 / -0.1321 | 14.547 (r_c) | 1944, 0.015 | 338, 0.004 | 193, 0.015 | 531, 0.015 |
| P64_s4_widened | -0.1299 / -0.1321 | 16.000 | 1911, 0.015 | 243, 0.004 | 192, 0.015 | 435, 0.015 |

**Table I1. Received-sample minima over declared selections** (reviewer B rzchk p. 12; rzchk3 secs. 4.2-4.4; reviewer A rzchk3 item 10). For each selection, the SMALLEST L_T and L_D over the received events in it inside the solver-certified record interval, and counts. These are sample minima over received events, not bounds over all signals, and no signal band is specified. (a) a clock's whole received worldline; (b) a clock's last Δτ of proper time before its last eligible received emission, the endpoint taken FOR EACH POPULATION SEPARATELY (an outcome-relative selection with no censored count; the masks move the endpoint, so the masked and unmasked (b) rows are different selections, not one selection with fewer events: P64_s2's resolved A1 row ends earlier on its clocks; the endpoints per clock are in Table I1e); (c) root-neighbourhood launches B within δ ≤ 1, 4, 16 a outside a standing outer root, emitted inside the interval, with the not-received launches split: received after the interval's end (inside the stored window) / not received by the last stored record. A1 and A2 apart, unmasked and resolved; for (a) the received share of the clocks' proper time over the population's own domain (unmasked: Table H3's unmasked share; resolved: its resolved share).

| member | population | (a) whole worldline: min L_T / min L_D (n; received proper-time share over the population's domain) | (b) last 0.01 r_s | (b) last 0.1 r_s | (b) last 1 r_s |
|---|---|---|---|---|---|
| P64_s4 | A1 | 0.357 / 0.226 (13879; 0.582 to 0.610) | 5.249 / 5.121 (3557) | 3.283 / 3.164 (5834) | 1.106 / 0.987 (8334) |
| P64_s4 | A1 resolved | 0.357 / 0.226 (13879; 0.839 to 0.913) | 5.249 / 5.121 (3557) | 3.283 / 3.164 (5834) | 1.106 / 0.987 (8334) |
| P64_s4 | A2 | 1.534 / 1.414 (3301; 0.073 to 0.176) | 2.658 / 2.541 (1231) | 2.191 / 2.073 (2317) | 1.534 / 1.414 (3301) |
| P64_s4 | A2 resolved | 1.534 / 1.414 (3301; 0.275 to 0.442) | 2.658 / 2.541 (1231) | 2.191 / 2.073 (2317) | 1.534 / 1.414 (3301) |
| P64_s4 | A1+A2 (v2) | 0.357 / 0.226 (17180; 0.073 to 0.610) | 2.658 / 2.541 (4788) | 2.191 / 2.073 (8151) | 1.106 / 0.987 (11635) |
| P64_s2 | A1 | 0.360 / 0.228 (13876; 0.660 to 0.692) | 4.846 / 4.725 (5084) | 2.930 / 2.822 (6386) | 1.019 / 0.912 (8456) |
| P64_s2 | A1 resolved | 0.360 / 0.228 (12518; 0.848 to 0.966) | 1.233 / 1.126 (4249) | 1.158 / 1.051 (5348) | 0.741 / 0.632 (7206) |
| P64_s2 | A2 | 1.338 / 1.224 (3515; 0.127 to 0.284) | 2.668 / 2.560 (1750) | 2.150 / 2.042 (2564) | 1.338 / 1.224 (3515) |
| P64_s2 | A2 resolved | 1.338 / 1.224 (3515; 0.401 to 0.598) | 2.668 / 2.560 (1750) | 2.150 / 2.042 (2564) | 1.338 / 1.224 (3515) |
| P64_s2 | A1+A2 (v2) | 0.360 / 0.228 (17391; 0.127 to 0.692) | 2.668 / 2.560 (6834) | 2.150 / 2.042 (8950) | 1.019 / 0.912 (11971) |
| P64_s8 | A1 | -1.819 / -1.958 (13883; 0.696 to 0.715) | 1.956 / 1.831 (30) | 1.956 / 1.831 (330) | 1.255 / 1.123 (2394) |
| P64_s8 | A1 resolved | -1.777 / -1.916 (9581; 0.951 to 0.991) | -1.777 / -1.916 (6) | -1.777 / -1.916 (11) | -1.777 / -1.916 (605) |
| P64_s8 | A2 | 0.353 / 0.214 (3088; 0.066 to 0.402) | 1.959 / 1.834 (36) | 1.959 / 1.834 (388) | 1.258 / 1.126 (1870) |
| P64_s8 | A2 resolved | 0.353 / 0.214 (2503; 0.109 to 0.947) | 0.843 / 0.706 (32) | 0.806 / 0.669 (342) | 0.355 / 0.216 (1582) |
| P64_s8 | A1+A2 (v2) | -1.819 / -1.958 (16971; 0.066 to 0.715) | 1.956 / 1.831 (66) | 1.956 / 1.831 (718) | 1.255 / 1.123 (4264) |
| P64_s4_tau4 | A1 | 0.358 / 0.227 (13879; 0.573 to 0.598) | 4.959 / 4.832 (858) | 3.348 / 3.212 (3214) | 1.372 / 1.247 (8027) |
| P64_s4_tau4 | A1 resolved | 0.358 / 0.227 (13879; 0.858 to 0.938) | 4.959 / 4.832 (858) | 3.348 / 3.212 (3214) | 1.372 / 1.247 (8027) |
| P64_s4_tau4 | A2 | 1.508 / 1.380 (3243; 0.131 to 0.178) | 2.643 / 2.518 (645) | 2.154 / 2.027 (1871) | 1.508 / 1.380 (3243) |
| P64_s4_tau4 | A2 resolved | 1.508 / 1.380 (3243; 0.300 to 0.544) | 2.643 / 2.518 (645) | 2.154 / 2.027 (1871) | 1.508 / 1.380 (3243) |
| P64_s4_tau4 | A1+A2 (v2) | 0.358 / 0.227 (17122; 0.131 to 0.598) | 2.643 / 2.518 (1503) | 2.154 / 2.027 (5085) | 1.372 / 1.247 (11270) |
| P64_s4_tau2 | A1 | 0.358 / 0.227 (3415; 0.442 to 0.557) | 0.486 / 0.357 (12) | 0.480 / 0.352 (73) | 0.433 / 0.302 (689) |
| P64_s4_tau2 | A1 resolved | 0.358 / 0.227 (3415; 0.481 to 0.566) | 0.486 / 0.357 (12) | 0.480 / 0.352 (73) | 0.433 / 0.302 (689) |
| P64_s4_tau2 | A2 | - | - | - | - |
| P64_s4_tau2 | A2 resolved | - | - | - | - |
| P64_s4_tau2 | A1+A2 (v2) | 0.358 / 0.227 (3415; 0.442 to 0.557) | 0.486 / 0.357 (12) | 0.480 / 0.352 (73) | 0.433 / 0.302 (689) |
| P64_s4_newton | A1 | 0.357 / 0.226 (3319; 0.427 to 0.544) | 0.465 / 0.338 (12) | 0.462 / 0.334 (73) | 0.423 / 0.293 (680) |
| P64_s4_newton | A1 resolved | 0.357 / 0.226 (3319; 0.460 to 0.546) | 0.465 / 0.338 (12) | 0.462 / 0.334 (73) | 0.423 / 0.293 (680) |
| P64_s4_newton | A2 | - | - | - | - |
| P64_s4_newton | A2 resolved | - | - | - | - |
| P64_s4_newton | A1+A2 (v2) | 0.357 / 0.226 (3319; 0.427 to 0.544) | 0.465 / 0.338 (12) | 0.462 / 0.334 (73) | 0.423 / 0.293 (680) |
| W2 | A1 | -4.305 / -4.457 (13879; 0.690 to 0.714) | 2.799 / 2.664 (66) | 2.799 / 2.664 (1074) | 0.374 / 0.223 (4002) |
| W2 | A1 resolved | -4.276 / -4.428 (9774; 0.912 to 0.978) | -4.276 / -4.428 (6) | -4.276 / -4.428 (7) | -4.276 / -4.428 (1998) |
| W2 | A2 | 1.130 / 0.979 (3174; 0.129 to 0.270) | 2.544 / 2.408 (66) | 2.095 / 1.958 (1054) | 1.130 / 0.979 (2829) |
| W2 | A2 resolved | 1.130 / 0.979 (2686; 0.376 to 0.971) | 2.347 / 2.203 (57) | 1.941 / 1.795 (912) | 1.130 / 0.979 (2398) |
| W2 | A1+A2 (v2) | -4.305 / -4.457 (17053; 0.129 to 0.714) | 2.544 / 2.408 (132) | 2.095 / 1.958 (2128) | 0.374 / 0.223 (6831) |
| ISO_64_s4 | A1 | 0.324 / 0.193 (3711; 0.483 to 0.577) | 0.465 / 0.345 (12) | 0.459 / 0.338 (72) | 0.429 / 0.299 (677) |
| ISO_64_s4 | A1 resolved | 0.324 / 0.193 (3711; 0.503 to 0.586) | 0.465 / 0.345 (12) | 0.459 / 0.338 (72) | 0.429 / 0.299 (677) |
| ISO_64_s4 | A2 | - | - | - | - |
| ISO_64_s4 | A2 resolved | - | - | - | - |
| ISO_64_s4 | A1+A2 (v2) | 0.324 / 0.193 (3711; 0.483 to 0.577) | 0.465 / 0.345 (12) | 0.459 / 0.338 (72) | 0.429 / 0.299 (677) |
| S4_128_s8 | A1 | -3.946 / -4.081 (27750; 0.672 to 0.697) | 5.007 / 4.881 (2538) | 3.297 / 3.168 (6384) | -3.946 / -4.081 (8670) |
| S4_128_s8 | A1 resolved | -3.891 / -4.027 (19084; 0.944 to 0.999) | -3.891 / -4.027 (6) | -3.891 / -4.027 (6) | -3.891 / -4.027 (1218) |
| S4_128_s8 | A2 | 1.352 / 1.225 (6736; 0.133 to 0.225) | 2.437 / 2.315 (1561) | 2.042 / 1.920 (3903) | 1.352 / 1.225 (6659) |
| S4_128_s8 | A2 resolved | 1.352 / 1.225 (6736; 0.425 to 0.516) | 2.437 / 2.315 (1561) | 2.042 / 1.920 (3903) | 1.352 / 1.225 (6659) |
| S4_128_s8 | A1+A2 (v2) | -3.946 / -4.081 (34486; 0.133 to 0.697) | 2.437 / 2.315 (4099) | 2.042 / 1.920 (10287) | -3.946 / -4.081 (15329) |
| S4_128_s16 | A1 | -2.541 / -2.683 (27759; 0.736 to 0.756) | -0.485 / -0.611 (6) | -0.550 / -0.675 (48) | -0.589 / -0.714 (636) |
| S4_128_s16 | A1 resolved | -2.533 / -2.676 (17825; 0.949 to 0.995) | -2.533 / -2.676 (6) | -2.533 / -2.676 (12) | -2.533 / -2.676 (130) |
| S4_128_s16 | A2 | -0.586 / -0.711 (6338; 0.105 to 0.509) | -0.484 / -0.609 (14) | -0.548 / -0.673 (128) | -0.586 / -0.711 (1641) |
| S4_128_s16 | A2 resolved | -0.554 / -0.679 (5918; 0.173 to 0.993) | -0.554 / -0.679 (14) | -0.554 / -0.679 (156) | -0.554 / -0.679 (2366) |
| S4_128_s16 | A1+A2 (v2) | -2.541 / -2.683 (34097; 0.105 to 0.756) | -0.485 / -0.611 (20) | -0.550 / -0.675 (176) | -0.589 / -0.714 (2277) |
| S4_128_s4 | A1 | 0.644 / 0.514 (532; 0.254 to 0.371) | 0.644 / 0.514 (6) | 0.644 / 0.514 (51) | 0.644 / 0.514 (461) |
| S4_128_s4 | A1 resolved | 0.644 / 0.514 (532; 0.254 to 0.371) | 0.644 / 0.514 (6) | 0.644 / 0.514 (51) | 0.644 / 0.514 (461) |
| S4_128_s4 | A2 | - | - | - | - |
| S4_128_s4 | A2 resolved | - | - | - | - |
| S4_128_s4 | A1+A2 (v2) | 0.644 / 0.514 (532; 0.254 to 0.371) | 0.644 / 0.514 (6) | 0.644 / 0.514 (51) | 0.644 / 0.514 (461) |
| ISO_128_s4 | A1 | 0.339 / 0.209 (6279; 0.404 to 0.528) | 0.426 / 0.305 (17) | 0.426 / 0.305 (136) | 0.404 / 0.273 (1325) |
| ISO_128_s4 | A1 resolved | 0.339 / 0.209 (6279; 0.407 to 0.528) | 0.426 / 0.305 (17) | 0.426 / 0.305 (136) | 0.404 / 0.273 (1325) |
| ISO_128_s4 | A2 | - | - | - | - |
| ISO_128_s4 | A2 resolved | - | - | - | - |
| ISO_128_s4 | A1+A2 (v2) | 0.339 / 0.209 (6279; 0.404 to 0.528) | 0.426 / 0.305 (17) | 0.426 / 0.305 (136) | 0.404 / 0.273 (1325) |
| ISO_128_s4_plain | A1 | 0.635 / 0.504 (452; 0.225 to 0.356) | 0.635 / 0.504 (6) | 0.635 / 0.504 (51) | 0.635 / 0.504 (421) |
| ISO_128_s4_plain | A1 resolved | 0.635 / 0.504 (452; 0.225 to 0.356) | 0.635 / 0.504 (6) | 0.635 / 0.504 (51) | 0.635 / 0.504 (421) |
| ISO_128_s4_plain | A2 | - | - | - | - |
| ISO_128_s4_plain | A2 resolved | - | - | - | - |
| ISO_128_s4_plain | A1+A2 (v2) | 0.635 / 0.504 (452; 0.225 to 0.356) | 0.635 / 0.504 (6) | 0.635 / 0.504 (51) | 0.635 / 0.504 (421) |
| M3c_s4_96 | A1 | 0.313 / 0.177 (10579; 0.679 to 0.700) | 0.614 / 0.486 (12) | 0.611 / 0.480 (115) | 0.549 / 0.415 (1090) |
| M3c_s4_96 | A1 resolved | 0.313 / 0.177 (10579; 0.697 to 0.796) | 0.614 / 0.486 (12) | 0.611 / 0.480 (115) | 0.549 / 0.415 (1090) |
| M3c_s4_96 | A2 | 0.910 / 0.777 (111; 0.098 to 0.098) | 0.921 / 0.793 (3) | 0.921 / 0.791 (25) | 0.910 / 0.777 (111) |
| M3c_s4_96 | A2 resolved | 0.910 / 0.777 (111; 0.098 to 0.098) | 0.921 / 0.793 (3) | 0.921 / 0.791 (25) | 0.910 / 0.777 (111) |
| M3c_s4_96 | A1+A2 (v2) | 0.313 / 0.177 (10690; 0.098 to 0.700) | 0.614 / 0.486 (15) | 0.611 / 0.480 (140) | 0.549 / 0.415 (1201) |
| P64_s4_B2 | A1 | 0.357 / 0.226 (13879; 0.582 to 0.610) | 5.249 / 5.121 (3557) | 3.283 / 3.164 (5834) | 1.106 / 0.987 (8334) |
| P64_s4_B2 | A1 resolved | 0.357 / 0.226 (13879; 0.839 to 0.913) | 5.249 / 5.121 (3557) | 3.283 / 3.164 (5834) | 1.106 / 0.987 (8334) |
| P64_s4_B2 | A2 | 1.534 / 1.414 (3301; 0.073 to 0.176) | 2.658 / 2.541 (1231) | 2.191 / 2.073 (2317) | 1.534 / 1.414 (3301) |
| P64_s4_B2 | A2 resolved | 1.534 / 1.414 (3301; 0.275 to 0.442) | 2.658 / 2.541 (1231) | 2.191 / 2.073 (2317) | 1.534 / 1.414 (3301) |
| P64_s4_B2 | A1+A2 (v2) | 0.357 / 0.226 (17180; 0.073 to 0.610) | 2.658 / 2.541 (4788) | 2.191 / 2.073 (8151) | 1.106 / 0.987 (11635) |
| P64_s4_ext200 | A1 | 0.357 / 0.226 (13879; 0.582 to 0.610) | 5.249 / 5.121 (3557) | 3.283 / 3.164 (5834) | 1.106 / 0.987 (8334) |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1 resolved | 0.357 / 0.226 (13879; 0.839 to 0.913) | 5.249 / 5.121 (3557) | 3.283 / 3.164 (5834) | 1.106 / 0.987 (8334) |
| P64_s4_ext200 | A2 | 1.540 / 1.420 (3290; 0.071 to 0.174) | 2.663 / 2.545 (1233) | 2.195 / 2.077 (2319) | 1.540 / 1.420 (3290) |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2 resolved | 1.540 / 1.420 (3290; 0.270 to 0.438) | 2.663 / 2.545 (1233) | 2.195 / 2.077 (2319) | 1.540 / 1.420 (3290) |
| P64_s4_ext200 | A1+A2 (v2) | 0.357 / 0.226 (17169; 0.071 to 0.610) | 2.663 / 2.545 (4790) | 2.195 / 2.077 (8153) | 1.106 / 0.987 (11624) |
| P64_s4_heldsrc | A1 | -3.149 / -3.293 (13879; 0.779 to 0.796) | 0.249 / 0.129 (6) | 0.249 / 0.129 (54) | 0.249 / 0.129 (498) |
| P64_s4_heldsrc | A1 resolved | -3.126 / -3.269 (8905; 0.937 to 0.989) | -3.126 / -3.269 (6) | -3.126 / -3.269 (7) | -3.126 / -3.269 (602) |
| P64_s4_heldsrc | A2 | -0.020 / -0.165 (3488; 0.054 to 0.662) | 0.249 / 0.129 (6) | 0.249 / 0.129 (54) | 0.249 / 0.129 (436) |
| P64_s4_heldsrc | A2 resolved | -0.020 / -0.165 (3488; 0.054 to 0.662) | 0.249 / 0.129 (6) | 0.249 / 0.129 (54) | 0.249 / 0.129 (436) |
| P64_s4_heldsrc | A1+A2 (v2) | -3.149 / -3.293 (17367; 0.054 to 0.796) | 0.249 / 0.129 (12) | 0.249 / 0.129 (108) | 0.249 / 0.129 (934) |
| P64_s4_widened | A1 | -2.905 / -3.042 (13879; 0.646 to 0.666) | 1.692 / 1.568 (24) | 1.692 / 1.568 (294) | 0.866 / 0.730 (3570) |
| P64_s4_widened | A1 resolved | -2.876 / -3.014 (10021; 0.950 to 0.993) | -2.664 / -2.802 (6) | -2.664 / -2.802 (11) | -2.876 / -3.014 (634) |
| P64_s4_widened | A2 | 0.987 / 0.850 (3146; 0.101 to 0.226) | 1.730 / 1.606 (46) | 1.730 / 1.606 (603) | 1.184 / 1.047 (2710) |
| P64_s4_widened | A2 resolved | 0.987 / 0.850 (3146; 0.184 to 0.885) | 1.730 / 1.606 (46) | 1.730 / 1.606 (603) | 1.184 / 1.047 (2710) |
| P64_s4_widened | A1+A2 (v2) | -2.905 / -3.042 (17025; 0.101 to 0.666) | 1.692 / 1.568 (70) | 1.692 / 1.568 (897) | 0.866 / 0.730 (6280) |

**Across members**, per selection and population: the smallest of the members' minima (member), over the members printed above.

| selection | population | smallest L_T (member) | smallest L_D (member) |
|---|---|---|---|
| (a) whole received worldline | A1 | -4.305 (W2) | -4.457 (W2) |
| (a) whole received worldline | A1 resolved | -4.276 (W2) | -4.428 (W2) |
| (a) whole received worldline | A2 | -0.586 (S4_128_s16) | -0.711 (S4_128_s16) |
| (a) whole received worldline | A2 resolved | -0.554 (S4_128_s16) | -0.679 (S4_128_s16) |
| (a) whole received worldline | A1+A2 (v2) | -4.305 (W2) | -4.457 (W2) |
| (b) last 0.01 r_s | A1 | -0.485 (S4_128_s16) | -0.611 (S4_128_s16) |
| (b) last 0.01 r_s | A1 resolved | -4.276 (W2) | -4.428 (W2) |
| (b) last 0.01 r_s | A2 | -0.484 (S4_128_s16) | -0.609 (S4_128_s16) |
| (b) last 0.01 r_s | A2 resolved | -0.554 (S4_128_s16) | -0.679 (S4_128_s16) |
| (b) last 0.01 r_s | A1+A2 (v2) | -0.485 (S4_128_s16) | -0.611 (S4_128_s16) |
| (b) last 0.1 r_s | A1 | -0.550 (S4_128_s16) | -0.675 (S4_128_s16) |
| (b) last 0.1 r_s | A1 resolved | -4.276 (W2) | -4.428 (W2) |
| (b) last 0.1 r_s | A2 | -0.548 (S4_128_s16) | -0.673 (S4_128_s16) |
| (b) last 0.1 r_s | A2 resolved | -0.554 (S4_128_s16) | -0.679 (S4_128_s16) |
| (b) last 0.1 r_s | A1+A2 (v2) | -0.550 (S4_128_s16) | -0.675 (S4_128_s16) |
| (b) last 1 r_s | A1 | -3.946 (S4_128_s8) | -4.081 (S4_128_s8) |
| (b) last 1 r_s | A1 resolved | -4.276 (W2) | -4.428 (W2) |
| (b) last 1 r_s | A2 | -0.586 (S4_128_s16) | -0.711 (S4_128_s16) |
| (b) last 1 r_s | A2 resolved | -0.554 (S4_128_s16) | -0.679 (S4_128_s16) |
| (b) last 1 r_s | A1+A2 (v2) | -3.946 (S4_128_s8) | -4.081 (S4_128_s8) |
| (c) δ ≤ 1 a | B | -0.363 (W2) | -0.516 (W2) |
| (c) δ ≤ 1 a | B resolved | -0.363 (W2) | -0.516 (W2) |
| (c) δ ≤ 4 a | B | -0.363 (W2) | -0.516 (W2) |
| (c) δ ≤ 4 a | B resolved | -0.363 (W2) | -0.516 (W2) |
| (c) δ ≤ 16 a | B | -0.363 (W2) | -0.516 (W2) |
| (c) δ ≤ 16 a | B resolved | -0.363 (W2) | -0.516 (W2) |

**Table I1c. B launches near a standing root** (selection (c)): min L_T / min L_D over the received; counts received inside the interval / received after its end / not received by the last stored record; and the launches emitted at the certified end's record itself (present only where the stored records run past it, since no B launch is made at a history's last record).

| member | population | δ ≤ 1 a: min L_T / min L_D (received inside, received after the end, not received by the last stored record) | δ ≤ 4 a | δ ≤ 16 a | launches at the certified end's record (δ ≤ 1, 4, 16 a) |
|---|---|---|---|---|---|
| P64_s4 | B | 2.453 / 2.319 (935, 0, 1949) | 2.278 / 2.149 (1260, 0, 2242) | 1.947 / 1.827 (1520, 0, 2394) | 0, 0, 0 |
| P64_s4 | B resolved | 2.453 / 2.319 (935, 0, 1949) | 2.278 / 2.149 (1260, 0, 2242) | 1.947 / 1.827 (1520, 0, 2394) | 0, 0, 0 |
| P64_s2 | B | 2.349 / 2.236 (1449, 0, 1687) | 2.107 / 1.997 (1864, 0, 1944) | 1.695 / 1.585 (2171, 0, 2085) | 0, 0, 0 |
| P64_s2 | B resolved | 2.349 / 2.236 (1449, 0, 1687) | 2.107 / 1.997 (1864, 0, 1944) | 1.695 / 1.585 (2171, 0, 2085) | 0, 0, 0 |
| P64_s8 | B | 1.361 / 1.222 (308, 0, 0) | 1.361 / 1.222 (374, 0, 0) | 1.361 / 1.222 (418, 0, 0) | 0, 0, 0 |
| P64_s8 | B resolved | 1.361 / 1.222 (308, 0, 0) | 1.361 / 1.222 (374, 0, 0) | 1.361 / 1.222 (418, 0, 0) | 0, 0, 0 |
| P64_s4_tau4 | B | 2.241 / 2.097 (1790, 0, 3026) | 2.142 / 2.000 (2225, 0, 3623) | 1.892 / 1.765 (2612, 0, 3924) | 0, 0, 0 |
| P64_s4_tau4 | B resolved | 2.241 / 2.097 (1790, 0, 3026) | 2.142 / 2.000 (2225, 0, 3623) | 1.892 / 1.765 (2612, 0, 3924) | 0, 0, 0 |
| P64_s4_tau2 | B | - / - (0, 0, 378) | - / - (0, 0, 459) | - / - (0, 0, 513) | 14, 17, 19 |
| P64_s4_tau2 | B resolved | - / - (0, 0, 378) | - / - (0, 0, 459) | - / - (0, 0, 513) | 14, 17, 19 |
| P64_s4_newton | B | - / - (0, 0, 42) | - / - (0, 0, 51) | - / - (0, 0, 57) | 0, 0, 0 |
| P64_s4_newton | B resolved | - / - (0, 0, 42) | - / - (0, 0, 51) | - / - (0, 0, 57) | 0, 0, 0 |
| W2 | B | -0.363 / -0.516 (840, 0, 714) | -0.363 / -0.516 (1023, 0, 864) | -0.363 / -0.516 (1151, 0, 958) | 0, 0, 0 |
| W2 | B resolved | -0.363 / -0.516 (840, 0, 714) | -0.363 / -0.516 (1023, 0, 864) | -0.363 / -0.516 (1151, 0, 958) | 0, 0, 0 |
| ISO_64_s4 | B | - / - (0, 86, 54) | - / - (0, 109, 61) | - / - (0, 128, 62) | 14, 17, 19 |
| ISO_64_s4 | B resolved | - / - (0, 86, 54) | - / - (0, 109, 61) | - / - (0, 128, 62) | 14, 17, 19 |
| S4_128_s8 | B | 0.677 / 0.541 (1471, 0, 3196) | 0.677 / 0.541 (1899, 0, 3845) | 0.677 / 0.541 (2279, 0, 4183) | 0, 0, 0 |
| S4_128_s8 | B resolved | 0.677 / 0.541 (1471, 0, 3196) | 0.677 / 0.541 (1899, 0, 3845) | 0.677 / 0.541 (2279, 0, 4183) | 0, 0, 0 |
| S4_128_s16 | B | 0.616 / 0.474 (598, 0, 0) | 0.616 / 0.474 (736, 0, 0) | 0.616 / 0.474 (828, 0, 0) | 0, 0, 0 |
| S4_128_s16 | B resolved | 0.616 / 0.474 (598, 0, 0) | 0.616 / 0.474 (736, 0, 0) | 0.616 / 0.474 (828, 0, 0) | 0, 0, 0 |
| S4_128_s4 | B | - / - (0, 0, 0) | - / - (0, 0, 0) | - / - (0, 0, 0) | 0, 0, 0 |
| S4_128_s4 | B resolved | - / - (0, 0, 0) | - / - (0, 0, 0) | - / - (0, 0, 0) | 0, 0, 0 |
| ISO_128_s4 | B | - / - (0, 0, 104) | - / - (0, 0, 128) | - / - (0, 0, 144) | 13, 16, 18 |
| ISO_128_s4 | B resolved | - / - (0, 0, 104) | - / - (0, 0, 128) | - / - (0, 0, 144) | 13, 16, 18 |
| ISO_128_s4_plain | B | - / - (0, 0, 0) | - / - (0, 0, 0) | - / - (0, 0, 0) | 0, 0, 0 |
| ISO_128_s4_plain | B resolved | - / - (0, 0, 0) | - / - (0, 0, 0) | - / - (0, 0, 0) | 0, 0, 0 |
| M3c_s4_96 | B | - / - (0, 0, 154) | - / - (0, 0, 176) | - / - (0, 0, 209) | 0, 0, 0 |
| M3c_s4_96 | B resolved | - / - (0, 0, 154) | - / - (0, 0, 176) | - / - (0, 0, 209) | 0, 0, 0 |
| P64_s4_B2 | B | 2.453 / 2.319 (935, 948, 1015) | 2.278 / 2.149 (1260, 1226, 1033) | 1.947 / 1.827 (1520, 1378, 1035) | 14, 17, 19 |
| P64_s4_B2 | B resolved | 2.453 / 2.319 (935, 948, 1015) | 2.278 / 2.149 (1260, 1226, 1033) | 1.947 / 1.827 (1520, 1378, 1035) | 14, 17, 19 |
| P64_s4_ext200 | B | 2.453 / 2.319 (935, 948, 1015) | 2.278 / 2.149 (1260, 1226, 1033) | 1.947 / 1.827 (1520, 1378, 1035) | 14, 17, 19 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | B resolved | 2.453 / 2.319 (935, 948, 1015) | 2.278 / 2.149 (1260, 1226, 1033) | 1.947 / 1.827 (1520, 1378, 1035) | 14, 17, 19 |
| P64_s4_heldsrc | B | 0.445 / 0.301 (294, 0, 0) | 0.445 / 0.301 (357, 0, 0) | 0.445 / 0.301 (399, 0, 0) | 0, 0, 0 |
| P64_s4_heldsrc | B resolved | 0.445 / 0.301 (294, 0, 0) | 0.445 / 0.301 (357, 0, 0) | 0.445 / 0.301 (399, 0, 0) | 0, 0, 0 |
| P64_s4_widened | B | 1.762 / 1.624 (644, 0, 0) | 1.762 / 1.624 (782, 0, 0) | 1.762 / 1.624 (874, 0, 0) | 0, 0, 0 |
| P64_s4_widened | B resolved | 1.762 / 1.624 (644, 0, 0) | 1.762 / 1.624 (782, 0, 0) | 1.762 / 1.624 (874, 0, 0) | 0, 0, 0 |

**Table I1e. The (b) selections' endpoints per clock** (AstraAPI_rzchk4 sec. 3.4, AstraAPI_rzchk5): on every member with comoving clocks, each clock's last eligible received emission inside the solver-certified record interval, per population (τ_e/r_s, T_e/r_s, r_e in a); the masked and unmasked selections end at different emissions wherever these differ.

| member | clock | A1 or A2: τ_e/r_s, T_e/r_s, r_e (a) | resolved: τ_e/r_s, T_e/r_s, r_e (a) |
|---|---|---|---|
| P64_s4 | A1.0 | 8.4027, 10.1436, 2.592 | 8.4027, 10.1436, 2.592 |
| P64_s4 | A1.1 | 8.4901, 10.2006, 5.327 | 8.4901, 10.2006, 5.327 |
| P64_s4 | A1.2 | 8.6071, 10.2615, 8.113 | 8.6071, 10.2615, 8.113 |
| P64_s4 | A1.3 | 8.7502, 10.3213, 10.719 | 8.7502, 10.3213, 10.719 |
| P64_s4 | A1.4 | 9.1063, 10.4308, 15.169 | 9.1063, 10.4308, 15.169 |
| P64_s4 | A1.5 | 9.5445, 10.5356, 19.057 | 9.5445, 10.5356, 19.057 |
| P64_s4 | A2.6 | 0.3491, 14.3532, 42.236 | 0.3491, 14.3532, 42.236 |
| P64_s4 | A2.7 | 0.4771, 18.5888, 36.529 | 0.4771, 18.5888, 36.529 |
| P64_s4 | A2.8 | 0.4826, 22.5435, 36.774 | 0.4826, 22.5435, 36.774 |
| P64_s4 | A2.9 | 0.4880, 26.5417, 36.719 | 0.4880, 26.5417, 36.719 |
| P64_s4 | A2.10 | 0.3942, 30.4259, 42.772 | 0.3942, 30.4259, 42.772 |
| P64_s2 | A1.0 | 8.2974, 10.0425, 0.463 | 7.5844, 9.1722, 1.042 |
| P64_s2 | A1.1 | 8.3974, 10.1010, 3.318 | 8.3974, 10.1010, 3.318 |
| P64_s2 | A1.2 | 8.5308, 10.1777, 6.844 | 8.5308, 10.1777, 6.844 |
| P64_s2 | A1.3 | 8.6845, 10.2483, 9.892 | 8.6845, 10.2483, 9.892 |
| P64_s2 | A1.4 | 9.0518, 10.3695, 14.749 | 9.0518, 10.3695, 14.749 |
| P64_s2 | A1.5 | 9.4950, 10.4793, 18.742 | 9.4950, 10.4793, 18.742 |
| P64_s2 | A2.6 | 0.4060, 14.4128, 39.469 | 0.4060, 14.4128, 39.469 |
| P64_s2 | A2.7 | 0.5914, 18.7257, 32.096 | 0.5914, 18.7257, 32.096 |
| P64_s2 | A2.8 | 0.6120, 22.6965, 31.287 | 0.6120, 22.6965, 31.287 |
| P64_s2 | A2.9 | 0.6086, 26.6713, 31.528 | 0.6086, 26.6713, 31.528 |
| P64_s2 | A2.10 | 0.5261, 30.5722, 36.519 | 0.5261, 30.5722, 36.519 |
| P64_s8 | A1.0 | 14.1273, 23.9021, 3.12e-05 | 9.3803, 11.3997, 1.113 |
| P64_s8 | A1.1 | 14.1697, 23.9021, 6.51e-05 | 9.8397, 12.0079, 1.192 |
| P64_s8 | A1.2 | 14.2401, 23.9021, 1.01e-04 | 10.0907, 12.2950, 1.267 |
| P64_s8 | A1.3 | 14.3385, 23.9021, 1.37e-04 | 10.2870, 12.4596, 1.353 |
| P64_s8 | A1.4 | 14.6155, 23.9021, 2.08e-04 | 10.6649, 12.6375, 1.555 |
| P64_s8 | A1.5 | 14.9805, 23.9021, 2.86e-04 | 11.2090, 12.9800, 1.174 |
| P64_s8 | A2.6 | 3.9080, 23.9053, 0.100 | 1.9130, 19.5247, 1.031 |
| P64_s8 | A2.7 | 2.9263, 23.9825, 2.445 | 2.9263, 23.9825, 2.445 |
| P64_s8 | A2.8 | 1.4602, 24.3689, 12.378 | 1.4602, 24.3689, 12.378 |
| P64_s8 | A2.9 | 0.4819, 26.5462, 37.046 | 0.4819, 26.5462, 37.046 |
| P64_s8 | A2.10 | 0.2794, 30.3045, 48.480 | 0.2794, 30.3045, 48.480 |
| P64_s4_tau4 | A1.0 | 8.5688, 10.3606, 2.130 | 8.5688, 10.3606, 2.130 |
| P64_s4_tau4 | A1.1 | 8.6492, 10.4089, 4.388 | 8.6492, 10.4089, 4.388 |
| P64_s4_tau4 | A1.2 | 8.7589, 10.4607, 6.701 | 8.7589, 10.4607, 6.701 |
| P64_s4_tau4 | A1.3 | 8.8951, 10.5118, 8.876 | 8.8951, 10.5118, 8.876 |
| P64_s4_tau4 | A1.4 | 9.2396, 10.6060, 12.615 | 9.2396, 10.6060, 12.615 |
| P64_s4_tau4 | A1.5 | 9.6684, 10.6971, 15.911 | 9.6684, 10.6971, 15.911 |
| P64_s4_tau4 | A2.6 | 0.7469, 15.3487, 12.962 | 0.7469, 15.3487, 12.962 |
| P64_s4_tau4 | A2.7 | 0.5895, 18.8215, 31.102 | 0.5895, 18.8215, 31.102 |
| P64_s4_tau4 | A2.8 | 0.4992, 22.5711, 35.756 | 0.4992, 22.5711, 35.756 |
| P64_s4_tau4 | A2.9 | 0.5313, 26.6131, 34.145 | 0.5313, 26.6131, 34.145 |
| P64_s4_tau4 | A2.10 | 0.4064, 30.4490, 42.307 | 0.4064, 30.4490, 42.307 |
| P64_s4_tau2 | A1.0 | 4.3756, 5.3612, 13.508 | 4.3756, 5.3612, 13.508 |
| P64_s4_tau2 | A1.1 | 4.6338, 5.6266, 26.721 | 4.6338, 5.6266, 26.721 |
| P64_s4_tau2 | A1.2 | 4.9083, 5.8837, 38.953 | 4.9083, 5.8837, 38.953 |
| P64_s4_tau2 | A1.3 | 5.1885, 6.1171, 49.614 | 5.1885, 6.1171, 49.614 |
| P64_s4_tau2 | A1.4 | 5.7818, 6.5313, 67.510 | 5.7818, 6.5313, 67.510 |
| P64_s4_tau2 | A1.5 | 6.4324, 6.9157, 82.749 | 6.4324, 6.9157, 82.749 |
| P64_s4_newton | A1.0 | 4.2317, 5.1880, 13.977 | 4.2317, 5.1880, 13.977 |
| P64_s4_newton | A1.1 | 4.4963, 5.4622, 27.594 | 4.4963, 5.4622, 27.594 |
| P64_s4_newton | A1.2 | 4.7764, 5.7269, 40.171 | 4.7764, 5.7269, 40.171 |
| P64_s4_newton | A1.3 | 5.0612, 5.9669, 51.113 | 5.0612, 5.9669, 51.113 |
| P64_s4_newton | A1.4 | 5.6628, 6.3931, 69.497 | 5.6628, 6.3931, 69.497 |
| P64_s4_newton | A1.5 | 6.3212, 6.7894, 85.194 | 6.3212, 6.7894, 85.194 |
| W2 | A1.0 | 11.2831, 19.0896, 3.53e-06 | 8.7658, 10.6271, 1.604 |
| W2 | A1.1 | 11.3275, 19.0896, 7.63e-06 | 9.2407, 11.2435, 1.675 |
| W2 | A1.2 | 11.3998, 19.0896, 1.22e-05 | 9.3277, 11.2673, 2.587 |
| W2 | A1.3 | 11.5002, 19.0896, 1.69e-05 | 9.4426, 11.2912, 3.466 |
| W2 | A1.4 | 11.7816, 19.0896, 2.63e-05 | 9.7504, 11.3363, 5.029 |
| W2 | A1.5 | 12.1527, 19.0896, 3.61e-05 | 10.1469, 11.3816, 6.472 |
| W2 | A2.6 | 1.7429, 19.1180, 0.622 | 1.4856, 18.3297, 1.034 |
| W2 | A2.7 | 1.0788, 19.9985, 16.764 | 1.0788, 19.9985, 16.764 |
| W2 | A2.8 | 0.5514, 22.6526, 33.037 | 0.5514, 22.6526, 33.037 |
| W2 | A2.9 | 0.6884, 26.9153, 24.720 | 0.6884, 26.9153, 24.720 |
| W2 | A2.10 | 0.4252, 30.4881, 41.558 | 0.4252, 30.4881, 41.558 |
| ISO_64_s4 | A1.0 | 5.0239, 6.1094, 16.600 | 5.0239, 6.1094, 16.600 |
| ISO_64_s4 | A1.1 | 5.2697, 6.3680, 29.817 | 5.2697, 6.3680, 29.817 |
| ISO_64_s4 | A1.2 | 5.4975, 6.5739, 39.945 | 5.4975, 6.5739, 39.945 |
| ISO_64_s4 | A1.3 | 5.7328, 6.7546, 48.526 | 5.7328, 6.7546, 48.526 |
| ISO_64_s4 | A1.4 | 6.2584, 7.0870, 63.423 | 6.2584, 7.0870, 63.423 |
| ISO_64_s4 | A1.5 | 6.8589, 7.4079, 76.530 | 6.8589, 7.4079, 76.530 |
| S4_128_s8 | A1.0 | 10.4054, 16.0681, 4.70e-05 | 8.9084, 10.6551, 4.781 |
| S4_128_s8 | A1.1 | 10.4448, 16.0681, 8.56e-05 | 8.9763, 10.6923, 8.224 |
| S4_128_s8 | A1.2 | 10.5127, 16.0681, 1.19e-04 | 9.5718, 11.5044, 1.131 |
| S4_128_s8 | A1.3 | 10.6098, 16.0681, 1.49e-04 | 9.6710, 11.5094, 1.385 |
| S4_128_s8 | A1.4 | 10.8862, 16.0681, 2.05e-04 | 9.9514, 11.5185, 1.832 |
| S4_128_s8 | A1.5 | 11.2471, 16.0681, 2.68e-04 | 10.3164, 11.5283, 2.287 |
| S4_128_s8 | A2.6 | 1.0792, 16.6610, 16.176 | 1.0792, 16.6610, 16.176 |
| S4_128_s8 | A2.7 | 0.6086, 18.7951, 62.388 | 0.6086, 18.7951, 62.388 |
| S4_128_s8 | A2.8 | 0.6354, 22.7306, 55.464 | 0.6354, 22.7306, 55.464 |
| S4_128_s8 | A2.9 | 0.6171, 26.7246, 62.105 | 0.6171, 26.7246, 62.105 |
| S4_128_s8 | A2.10 | 0.5373, 30.5949, 71.629 | 0.5373, 30.5949, 71.629 |
| S4_128_s16 | A1.0 | 15.2844, 25.4293, 2.94e-06 | 9.4745, 11.3715, 1.482 |
| S4_128_s16 | A1.1 | 15.3254, 25.4293, 5.60e-06 | 9.7177, 11.6606, 1.480 |
| S4_128_s16 | A1.2 | 15.3945, 25.4293, 8.05e-06 | 9.8801, 11.8045, 1.478 |
| S4_128_s16 | A1.3 | 15.4925, 25.4293, 1.04e-05 | 10.0604, 11.9393, 1.324 |
| S4_128_s16 | A1.4 | 15.7698, 25.4293, 1.50e-05 | 10.4106, 12.0669, 1.335 |
| S4_128_s16 | A1.5 | 16.1293, 25.4293, 2.03e-05 | 10.8326, 12.1839, 1.299 |
| S4_128_s16 | A2.6 | 5.2812, 25.4300, 0.041 | 3.9721, 22.9610, 1.006 |
| S4_128_s16 | A2.7 | 4.0855, 25.4360, 0.376 | 3.7900, 24.8164, 1.080 |
| S4_128_s16 | A2.8 | 2.1560, 25.4759, 2.523 | 2.1560, 25.4759, 2.523 |
| S4_128_s16 | A2.9 | 0.8150, 27.0551, 34.313 | 0.8150, 27.0551, 34.313 |
| S4_128_s16 | A2.10 | 0.4416, 30.5351, 81.825 | 0.4416, 30.5351, 81.825 |
| S4_128_s4 | A1.4 | 0.8192, 0.8792, 337.804 | 0.8192, 0.8792, 337.804 |
| S4_128_s4 | A1.5 | 1.7917, 1.8097, 399.342 | 1.7917, 1.8097, 399.342 |
| ISO_128_s4 | A1.0 | 4.0145, 4.8957, 33.956 | 4.0145, 4.8957, 33.956 |
| ISO_128_s4 | A1.1 | 4.2880, 5.1850, 63.026 | 4.2880, 5.1850, 63.026 |
| ISO_128_s4 | A1.2 | 4.5738, 5.4618, 89.681 | 4.5738, 5.4618, 89.681 |
| ISO_128_s4 | A1.3 | 4.8704, 5.7232, 113.835 | 4.8704, 5.7232, 113.835 |
| ISO_128_s4 | A1.4 | 5.4784, 6.1699, 152.971 | 5.4784, 6.1699, 152.971 |
| ISO_128_s4 | A1.5 | 6.1338, 6.5672, 185.362 | 6.1338, 6.5672, 185.362 |
| ISO_128_s4_plain | A1.4 | 0.6600, 0.7087, 347.015 | 0.6600, 0.7087, 347.015 |
| ISO_128_s4_plain | A1.5 | 1.6341, 1.6470, 408.946 | 1.6341, 1.6470, 408.946 |
| M3c_s4_96 | A1.0 | 10.7042, 13.2821, 3.935 | 10.7042, 13.2821, 3.935 |
| M3c_s4_96 | A1.1 | 10.7866, 13.3436, 8.195 | 10.7866, 13.3436, 8.195 |
| M3c_s4_96 | A1.2 | 10.8926, 13.4050, 12.369 | 10.8926, 13.4050, 12.369 |
| M3c_s4_96 | A1.3 | 11.0194, 13.4623, 16.199 | 11.0194, 13.4623, 16.199 |
| M3c_s4_96 | A1.4 | 11.3406, 13.5632, 22.777 | 11.3406, 13.5632, 22.777 |
| M3c_s4_96 | A1.5 | 11.7540, 13.6536, 28.517 | 11.7540, 13.6536, 28.517 |
| M3c_s4_96 | A2.6 | 0.4609, 14.5790, 79.687 | 0.4609, 14.5790, 79.687 |
| P64_s4_B2 | A1.0 | 8.4027, 10.1436, 2.592 | 8.4027, 10.1436, 2.592 |
| P64_s4_B2 | A1.1 | 8.4901, 10.2006, 5.327 | 8.4901, 10.2006, 5.327 |
| P64_s4_B2 | A1.2 | 8.6071, 10.2615, 8.113 | 8.6071, 10.2615, 8.113 |
| P64_s4_B2 | A1.3 | 8.7502, 10.3213, 10.719 | 8.7502, 10.3213, 10.719 |
| P64_s4_B2 | A1.4 | 9.1063, 10.4308, 15.169 | 9.1063, 10.4308, 15.169 |
| P64_s4_B2 | A1.5 | 9.5445, 10.5356, 19.057 | 9.5445, 10.5356, 19.057 |
| P64_s4_B2 | A2.6 | 0.3491, 14.3532, 42.236 | 0.3491, 14.3532, 42.236 |
| P64_s4_B2 | A2.7 | 0.4771, 18.5888, 36.529 | 0.4771, 18.5888, 36.529 |
| P64_s4_B2 | A2.8 | 0.4826, 22.5435, 36.774 | 0.4826, 22.5435, 36.774 |
| P64_s4_B2 | A2.9 | 0.4880, 26.5417, 36.719 | 0.4880, 26.5417, 36.719 |
| P64_s4_B2 | A2.10 | 0.3942, 30.4259, 42.772 | 0.3942, 30.4259, 42.772 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.0 | 8.4027, 10.1436, 2.592 | 8.4027, 10.1436, 2.592 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.1 | 8.4901, 10.2006, 5.327 | 8.4901, 10.2006, 5.327 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.2 | 8.6071, 10.2615, 8.113 | 8.6071, 10.2615, 8.113 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.3 | 8.7502, 10.3213, 10.719 | 8.7502, 10.3213, 10.719 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.4 | 9.1063, 10.4308, 15.169 | 9.1063, 10.4308, 15.169 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.5 | 9.5445, 10.5356, 19.057 | 9.5445, 10.5356, 19.057 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.6 | 0.3485, 14.3447, 42.277 | 0.3485, 14.3447, 42.277 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.7 | 0.4772, 18.5811, 36.520 | 0.4772, 18.5811, 36.520 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.8 | 0.4825, 22.5357, 36.779 | 0.4825, 22.5357, 36.779 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.9 | 0.4880, 26.5339, 36.715 | 0.4880, 26.5339, 36.715 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.10 | 0.3947, 30.4187, 42.743 | 0.3947, 30.4187, 42.743 |
| P64_s4_heldsrc | A1.0 | 19.9683, 32.9477, 0.001 | 9.0034, 10.9307, 1.339 |
| P64_s4_heldsrc | A1.1 | 20.0117, 32.9478, 0.003 | 9.5144, 11.6153, 1.267 |
| P64_s4_heldsrc | A1.2 | 20.0830, 32.9478, 0.005 | 9.5971, 11.6334, 1.970 |
| P64_s4_heldsrc | A1.3 | 20.1823, 32.9478, 0.007 | 9.7077, 11.6516, 2.655 |
| P64_s4_heldsrc | A1.4 | 20.4607, 32.9479, 0.010 | 10.3157, 12.1944, 1.570 |
| P64_s4_heldsrc | A1.5 | 20.8276, 32.9480, 0.014 | 10.6913, 12.2097, 2.082 |
| P64_s4_heldsrc | A2.6 | 10.0500, 33.1288, 8.248 | 10.0500, 33.1288, 8.248 |
| P64_s4_heldsrc | A2.7 | 9.1491, 33.5409, 27.122 | 9.1491, 33.5409, 27.122 |
| P64_s4_heldsrc | A2.8 | 7.4642, 33.7325, 35.947 | 7.4642, 33.7325, 35.947 |
| P64_s4_heldsrc | A2.9 | 5.3428, 33.9556, 46.255 | 5.3428, 33.9556, 46.255 |
| P64_s4_heldsrc | A2.10 | 2.9630, 34.2131, 58.202 | 2.9630, 34.2131, 58.202 |
| P64_s4_heldsrc | A2.11 | 0.2426, 34.3335, 63.789 | 0.2426, 34.3335, 63.789 |
| P64_s4_widened | A1.0 | 11.7096, 19.0489, 4.36e-04 | 8.9273, 10.8265, 1.481 |
| P64_s4_widened | A1.1 | 11.7531, 19.0490, 9.34e-04 | 9.5402, 11.6440, 1.323 |
| P64_s4_widened | A1.2 | 11.8245, 19.0490, 0.001 | 9.8241, 11.9919, 1.343 |
| P64_s4_widened | A1.3 | 11.9239, 19.0490, 0.002 | 10.0570, 12.2336, 1.290 |
| P64_s4_widened | A1.4 | 12.2030, 19.0491, 0.003 | 10.4558, 12.4738, 1.319 |
| P64_s4_widened | A1.5 | 12.5707, 19.0491, 0.004 | 10.9239, 12.7001, 1.181 |
| P64_s4_widened | A2.6 | 1.7802, 19.1216, 1.489 | 1.7802, 19.1216, 1.489 |
| P64_s4_widened | A2.7 | 1.0334, 19.9062, 14.995 | 1.0334, 19.9062, 14.995 |
| P64_s4_widened | A2.8 | 0.5751, 22.6953, 31.931 | 0.5751, 22.6953, 31.931 |
| P64_s4_widened | A2.9 | 0.5366, 26.6297, 33.063 | 0.5366, 26.6297, 33.063 |
| P64_s4_widened | A2.10 | 0.3431, 30.3903, 45.419 | 0.3431, 30.3903, 45.419 |

(v2, in deliveries/route2/REDSHIFT_received_v2_scripts/) Producing script: code/rz_v2_clockfloor_v1.py (md5 b3067d81572fc00035f29164efdc8404). Table file: out/v2_clockfloor_v1/clockfloor_tables.md (md5 bec0e1a09a8ef61ae8d367d3d48071b4).

**Table H9. The A1 clocks near the centre** (rz_v2_clockfloor_v1.py; reviewer B rzchk p. 6): the six A1 clocks' stored radii at T/r_s 14, 20, 30, 39: smallest / largest |r| (a), distinct |r| values (to 1e-12 relative), and negative radii (sign flips of the parity continuation).

| member | T/r_s 14 | 20 | 30 | 39 |
|---|---|---|---|---|
| P64_s4 | 1.867e-03 / 1.862e-02, 6 distinct, 0 negative | 1.103e-06 / 1.083e-05, 6 distinct, 0 negative | 6.056e-10 / 6.057e-10, 6 distinct, 3 negative | 4.688e-09 / 4.688e-09, 6 distinct, 3 negative |
| P64_s2 | 9.399e-06 / 8.135e-04, 6 distinct, 0 negative | 2.693e-09 / 2.693e-09, 6 distinct, 5 negative | 1.775e-07 / 1.775e-07, 4 distinct, 5 negative | 4.821e-07 / 4.821e-07, 1 distinct, 5 negative |
| P64_s8 | 1.796e-02 / 1.647e-01, 6 distinct, 0 negative | 2.680e-04 / 2.458e-03, 6 distinct, 0 negative | 7.327e-08 / 5.607e-07, 6 distinct, 0 negative | 1.672e-08 / 1.722e-08, 6 distinct, 0 negative |
| P64_s4_tau4 | 1.190e-03 / 1.195e-02, 6 distinct, 0 negative | 1.540e-06 / 1.584e-05, 6 distinct, 0 negative | 2.907e-07 / 2.907e-07, 6 distinct, 3 negative | 2.624e-10 / 2.624e-10, 6 distinct, 3 negative |
| W2 | 7.499e-04 / 7.580e-03, 6 distinct, 0 negative | 1.897e-06 / 1.959e-05, 6 distinct, 0 negative | 3.526e-08 / 3.526e-08, 6 distinct, 2 negative | 1.525e-07 / 1.525e-07, 6 distinct, 2 negative |
| S4_128_s8 | 4.531e-04 / 2.581e-03, 6 distinct, 0 negative | 4.011e-07 / 2.194e-06, 6 distinct, 0 negative | 1.678e-11 / 1.678e-11, 6 distinct, 4 negative | 7.140e-09 / 7.140e-09, 5 distinct, 4 negative |
| S4_128_s16 | 7.844e-03 / 5.427e-02, 6 distinct, 0 negative | 4.291e-04 / 2.968e-03, 6 distinct, 0 negative | 2.370e-08 / 1.674e-07, 6 distinct, 0 negative | 8.535e-10 / 8.699e-10, 6 distinct, 0 negative |
| ISO_64_s4 | 9.954e-03 / 4.100e-02, 6 distinct, 0 negative | 1.114e-05 / 4.603e-05, 6 distinct, 0 negative | 6.996e-08 / 6.996e-08, 6 distinct, 4 negative | 1.400e-10 / 1.400e-10, 6 distinct, 2 negative |
| M3c_s4_96 | 3.167e+00 / 2.565e+01, 6 distinct, 0 negative | 1.999e-01 / 1.618e+00, 6 distinct, 0 negative | - | - |

**Table T1. The tick check per member inside the certified window** (v1 sec. 7.1, reviewer A rzchk 2(d)): pasted from REDSHIFT_received_v1_scripts/out/tick_audit_v1/tick_audit_v1.md (md5 3b9d801d7595853c4600ea37252ebd6f; producing script rz_tick_audit_v1.py in that folder's code/).

| member | certified to T/r_s | pairs (1) | (1) all pairs, all arrivals: max | worst pair of (1): t_rec/r_s, ln(1/Z), certified?, below precision floor?, steep? | (2) certified arrivals only: max | worst pair of (2): t_rec/r_s, ln(1/Z), below floor?, steep? | (3) certified, precision floor applied | (4) certified, smoothness rule applied | (5) certified, both (the v7 rule) | pairs dropped in (5): floor / steep |
|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | 40.000 | 17147 | 7.66e-05 | 24.969, 2.753, yes, no, no | 7.66e-05 | 24.969, 2.753, no, no | 7.66e-05 | 7.66e-05 | 7.66e-05 | 0 / 0 |
| P64_s2 | 40.000 | 17361 | 2.30e-04 | 24.312, 2.247, yes, no, no | 2.30e-04 | 24.312, 2.247, no, no | 2.30e-04 | 2.30e-04 | 2.30e-04 | 0 / 0 |
| P64_s8 | 40.000 | 16936 | 0.006 | 28.844, -1.755, yes, no, no | 0.006 | 28.844, -1.755, no, no | 0.006 | 0.006 | 0.006 | 0 / 668 |
| P64_s4_tau4 | 40.000 | 17087 | 9.34e-05 | 24.797, 2.561, yes, no, no | 9.34e-05 | 24.797, 2.561, no, no | 9.34e-05 | 9.34e-05 | 9.34e-05 | 0 / 0 |
| P64_s4_tau2 | 12.750 | 6763 | 7.56e-06 | 12.688, 0.674, yes, no, no | 7.56e-06 | 12.688, 0.674, no, no | 7.56e-06 | 7.56e-06 | 7.56e-06 | 0 / 0 |
| W2 | 40.000 | 17018 | 1.183 | 29.328, -2.056, yes, no, yes | 1.183 | 29.328, -2.056, no, yes | 1.183 | 2.71e-04 | 2.71e-04 | 0 / 360 |
| ISO_64_s4 | 13.500 | 17147 | 6.57e-05 | 25.188, 2.959, no, no, no | 5.73e-05 | 13.250, 0.456, no, no | 5.73e-05 | 5.73e-05 | 5.73e-05 | 0 / 0 |
| S4_128_s8 | 40.000 | 34453 | 0.437 | 28.727, -2.191, yes, no, yes | 0.437 | 28.727, -2.191, no, yes | 0.437 | 1.70e-04 | 1.70e-04 | 0 / 480 |
| S4_128_s16 | 40.000 | 34063 | 0.008 | 27.047, -2.541, yes, no, no | 0.008 | 27.047, -2.541, no, no | 0.008 | 0.008 | 0.008 | 0 / 574 |
| ISO_128_s4 | 12.000 | 6315 | 1.01e-04 | 12.000, 0.672, no, no, no | 9.23e-05 | 11.938, 0.432, no, no | 9.23e-05 | 9.23e-05 | 9.23e-05 | 0 / 0 |
| S4_128_s4 | 3.562 | 11657 | 2.30e-04 | 11.750, 0.424, no, no, no | 1.75e-07 | 1.977, 0.698, no, no | 1.75e-07 | 1.75e-07 | 1.75e-07 | 0 / 0 |
| ISO_128_s4_plain | 3.250 | 8091 | 0.001 | 12.125, 0.488, no, no, no | 1.81e-07 | 2.102, 0.700, no, no | 1.81e-07 | 1.81e-07 | 1.81e-07 | 0 / 0 |
| M3c_s4_96 | 22.083 | 10676 | 2.31e-05 | 22.000, 0.631, yes, no, no | 2.31e-05 | 22.000, 0.631, no, no | 2.31e-05 | 2.31e-05 | 2.31e-05 | 0 / 0 |
| P64_s4_newton | 12.500 | 3307 | 2.52e-05 | 12.375, 0.463, yes, no, no | 2.52e-05 | 12.375, 0.463, no, no | 2.52e-05 | 2.52e-05 | 2.52e-05 | 0 / 0 |
| P64_s4_B2 | 40.000 | 67614 | - | 71.172, 30.124, no, yes, yes | 7.66e-05 | 24.969, 2.753, no, no | 7.66e-05 | 7.66e-05 | 7.66e-05 | 0 / 0 |
| P64_s4_ext200 | 40.000 | 170425 | - | 70.797, 29.878, no, yes, yes | 7.66e-05 | 24.969, 2.753, no, no | 7.66e-05 | 7.66e-05 | 7.66e-05 | 0 / 0 |

## 4. The A2 release, integration, field sensitivity and regime on the headline events (rz_v4_a2fix_v3.py; rz_v4_reint_tables_v1.py)

Producing script: code/rz_v4_a2fix_v3.py (md5 c1975992e5222cb04aa3f0f6bf876e3a). Table file: out/v41_a2fix_v3/a2fix_tables.md (md5 0cb295a4a3cbc3d3ad1a66f1a6d48468). Launcher code/v41tables_run1.sh. Made from v4's code/rz_v4_a2fix_v2.py (md5 b9b86394054b915437ed7af7d961f7a2) by code/mk_a2fix_v4_v3_patch.py (md5 152c74ed79c74c63608565fff995b556): the caption, the status names, and the event-level export out/v41_a2fix_v3/a2_events.csv.gz (md5 69a2efed000abcf4e722d59cc9214eee; 39331 stored A2 events; the plain csv stays on the box); its table rows equal v4's: yes; its json equals v4's apart from the export's entry: yes.

**Table A2Xv2. The A2 release, corrected** (rz_v4_a2fix_v3.py): a per-class AGGREGATE of the event-matched comparison, whose event level is exported beside it (a2_events.csv: every stored A2 event inside the interval). Per member and class, on the backward map at 4 r_s over arrivals inside the solver-certified record interval, the signed changes EXACT RELEASE MINUS STORED (rz_rays_v4n against rz_rays_v4m, which released the later-released clocks one step early), each the change of largest magnitude with its sign, and at the stored pass's class headline event. FIXED ARRIVAL (clock, T_D), over the arrivals received in both: T_e, τ_e, r_e (a), L_T, L_D. FIXED EMISSION LABEL (clock, τ_e): the stored events by status on the exact pass's SAMPLED map, its consecutive arrivals 1 a apart (bracketed / label past its last sampled received emission of that clock / before its first / no adjacent bracket; sampled-map statuses, non-receipt not verified), and over the bracketed the changes of T_D (a), L_T and L_D. Last, the class's largest L_T in each pass. A1 (released at T = 0) is the control: 0 means identical.

| member | class | fixed arrival: matched | ΔT_e (a): largest; at headline | Δτ_e (a) | Δr_e (a) | ΔL_T | ΔL_D | fixed emission, sampled-map status: bracketed / past the last sampled / before the first sampled / no adjacent bracket | ΔT_D (a): largest; at headline | ΔL_T | ΔL_D | largest L_T: stored → exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | A1 | 13879 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13879 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 10.4556 → 10.4556 |
| P64_s4 | A2 | 3290 | 0.539; 0.539 | -0.105; 0.0355 | 0.0813; -0.0405 | -0.00506; -0.005 | -0.00506; -0.005 | 3254 / 36 / 0 / 0 | -173; -173 | -1.71; -1.71 | -1.71; -1.71 | 9.4573 → 9.4523 |
| P64_s2 | A1 | 13876 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13876 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 15.6758 → 15.6758 |
| P64_s2 | A2 | 3507 | 0.533; 0.533 | -0.125; 0.0298 | 0.0915; -0.0345 | -0.00626; -0.00522 | -0.00626; -0.00522 | 3505 / 2 / 0 / 0 | -512; -512 | -6.55; -6.55 | -6.56; -6.56 | 14.4003 → 14.3951 |
| P64_s8 | A1 | 13883 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13883 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 3.2406 → 3.2406 |
| P64_s8 | A2 | 3075 | 0.591; 0.441 | -0.185; -0.0418 | 0.0786; 0.0303 | 0.00415; -0.000648 | 0.00415; -0.000648 | 3068 / 7 / 0 / 0 | 2.48; 0.997 | -0.0163; -0.000715 | -0.0163; -0.000699 | 3.1722 → 3.1716 |
| P64_s4_tau4 | A1 | 13879 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13879 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 6.1072 → 6.1072 |
| P64_s4_tau4 | A2 | 3230 | 0.717; 0.408 | -0.106; -0.037 | 0.0814; 0.034 | -0.00537; 0.00127 | -0.00537; 0.00127 | 3186 / 44 / 0 / 0 | 40.3; status 1 | 0.356; - | 0.357; - | 7.1824 → 7.1837 |
| P64_s4_tau2 | A1 | 3415 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 3415 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7454 → 0.7454 |
| P64_s4_newton | A1 | 3319 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 3319 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7242 → 0.7242 |
| W2 | A1 | 13879 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13879 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 4.0809 → 4.0809 |
| W2 | A2 | 3161 | 0.699; 0.191 | -0.106; -0.101 | 0.0804; 0.0438 | -0.00463; 0.00102 | -0.00463; 0.00102 | 3154 / 7 / 0 / 0 | 9.45; 9.44 | 0.051; -0.00421 | 0.0512; -0.00437 | 4.5416 → 4.5427 |
| ISO_64_s4 | A1 | 3711 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 3711 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7197 → 0.7197 |
| S4_128_s8 | A1 | 27750 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 27750 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 6.6994 → 6.6994 |
| S4_128_s8 | A2 | 6725 | 0.575; 0.415 | -0.123; -0.0374 | 0.0897; 0.0273 | -0.0027; 2.53e-05 | -0.0027; 2.53e-05 | 6617 / 108 / 0 / 0 | 75.2; status 1 | 0.285; - | 0.285; - | 7.7539 → 7.7540 |
| S4_128_s16 | A1 | 27759 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 27759 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 3.2852 → 3.2852 |
| S4_128_s16 | A2 | 6326 | 0.569; 0.174 | -0.26; -0.185 | 0.0878; 0.0557 | 0.00152; 0.000759 | 0.00152; 0.000759 | 6321 / 5 / 0 / 0 | 2.99; 2.99 | -0.0128; 0.000683 | -0.0128; 0.000707 | 2.7815 → 2.7822 |
| S4_128_s4 | A1 | 532 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 532 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7011 → 0.7011 |
| ISO_128_s4 | A1 | 6279 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 6279 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7029 → 0.7029 |
| ISO_128_s4_plain | A1 | 452 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 452 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7029 → 0.7029 |
| M3c_s4_96 | A1 | 10579 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 10579 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 0.7157 → 0.7157 |
| M3c_s4_96 | A2 | 111 | 0.194; 0.17 | -0.263; -0.261 | 0.0961; 0.0863 | 0.000334; 0.000331 | 0.000334; 0.000331 | 110 / 1 / 0 / 0 | 0.66; 0.656 | 0.000432; 0.00032 | 0.000443; 0.000378 | 0.9220 → 0.9223 |
| P64_s4_B2 | A1 | 13879 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13879 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 10.4556 → 10.4556 |
| P64_s4_B2 | A2 | 3290 | 0.539; 0.539 | -0.105; 0.0355 | 0.0813; -0.0405 | -0.00506; -0.005 | -0.00506; -0.005 | 3254 / 36 / 0 / 0 | -173; -173 | -1.71; -1.71 | -1.71; -1.71 | 9.4573 → 9.4523 |
| P64_s4_heldsrc | A1 | 13879 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13879 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 3.4596 → 3.4596 |
| P64_s4_heldsrc | A2 | 3482 | 0.564; 0.469 | -0.358; -0.0311 | 0.0933; 0.0287 | -0.00335; -0.00329 | -0.00335; -0.00329 | 3476 / 6 / 0 / 0 | 0.546; 0.505 | -0.00609; -0.00332 | -0.00625; -0.00339 | 2.7904 → 2.7871 |
| P64_s4_widened | A1 | 13879 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 0; 0 | 13879 / 0 / 0 / 0 | 0; 0 | 0; 0 | 0; 0 | 3.7076 → 3.7076 |
| P64_s4_widened | A2 | 3134 | 0.641; 0.455 | -0.107; -0.0264 | 0.0739; 0.0189 | 0.00749; -0.00176 | 0.00749; -0.00176 | 3129 / 5 / 0 / 0 | 3.21; 1.51 | -0.0159; -0.00185 | -0.0159; -0.00183 | 4.0520 → 4.0503 |

Over the members with A2 clocks (M3c_s4_96, P64_s2, P64_s4, P64_s4_B2, P64_s4_heldsrc, P64_s4_tau4, P64_s4_widened, P64_s8, S4_128_s16, S4_128_s8, W2): the largest |ΔL_T| at fixed arrival per member ranges 0.000334 to 0.00749 and the largest |ΔT_e| 0.194 to 0.717 a; the A1 clocks are identical in both passes on every member: yes.

(v4, in deliveries/route2/REDSHIFT_received_v4_scripts/) Producing script: code/rz_v4_reint_tables_v1.py (md5 f831fd37616bd7405c0d923bca24ac9a). Table file: out/v4_reint_tables_v1/reint4_tables.md (md5 4104a441b402cf8c094c231688ed6beb). Launcher code/v4tables_run1.sh (md5 14b6bacbe79a20253dc7abc5574c399c). Data: code/rz_v4_reint_v1.py (md5 eb398be7fbebb98b7ffbe54c3f043dc9; launchers run_v4reint_one_v1.sh, md5 dcb615da12bc0c8ae614fa06efe53d59, and queue_v4reint_v1.sh, md5 85d3924d4345cfa2b30493a0cb842227), its json in out/v4_reint_v1_json/ (npz on the box); the B rows from v3's rz_v3_reint_v1.py and rz_v3_reint_v3.py outputs (deliveries/route2/REDSHIFT_received_v3_scripts/out/v3_reint_v1_json/ and v3_reint_v3_json/).

**Table R1v4. Step ladder and the independent integrator on the headline events, every class** (rz_v4_reint_v1.py re-integrates the A events on the exact-release clocks; B rows from rz_v3_reint_v1.py, launch event held fixed; this table: rz_v4_reint_tables_v1.py). Per member and class (A1 r_e ≥ 1 a and A2: the backward comoving-clock events of the exact-release pass rz_rays_v4n; B: the forward root-neighbourhood launches): the events (the class's largest L_T arrivals at 4 r_s inside the solver-certified record interval, up to 20); the h = 0.5 a re-integration against the stored pass; max abs Δ L_T between h and h/2 and between h/2 and h/4 at FIXED ARRIVAL (A) or fixed launch (B), unrounded; the observed order log2 of their ratio (median over the events); for A, the FIXED EMISSION LABEL reading (clock, τ_e: each step's local map over the arrivals T_D + k a, k = −2..2, read at the h = 0.5 a run's τ_e, linearly between the bracketing pair and by a cubic through the five), over the events it brackets, with the events it does not bracket counted by reason (at h/2 and at h/4: label outside the five-arrival window / map not monotone / too few finite arrivals); and the independent integrator (DOP853, rtol 1e-11, restarted at every record time, clocks released at T_rel exactly) against h/4 on the class's top events: max abs Δ L_T and the shift of the emission time (A) or arrival time (B) against the stored pass, in a. 16 histories; P64_s4_B2 and P64_s4_ext200 are not separately integrated.

| member | class | events | h 0.5 rerun vs stored | h vs h/2 | h/2 vs h/4 | observed order | fixed emission: h vs h/2 / h/2 vs h/4 (cubic) over the bracketed | fixed emission, not bracketed at h/2; at h/4: outside the window / not monotone / too few | DOP853 vs h/4: max abs Δ L_T (events) | DOP853 − stored: max abs ΔT_e or ΔT_D (a) |
|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | A1 r_e>=1a | 20 | 0.000e+00 | 4.676e-10 | 4.541e-11 | 5.48 | 8.712e-08 / 5.414e-09 (cubic 8.677e-08 / 5.392e-09) over 18 / 18 | 2 / 0 / 0; 2 / 0 / 0 | 6.609e-09 (3) | 1.214e-09 |
| P64_s4 | A2 | 20 | 0.000e+00 | 3.732e-10 | 5.130e-11 | 3.14 | 1.123e-06 / 1.623e-07 (cubic 1.117e-06 / 1.616e-07) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 3.698e-10 (3) | 1.631e-07 |
| P64_s4 | B | 20 | 0.000e+00 | 6.266e-10 | 4.585e-11 | 3.51 | - | - | 6.267e-09 (3) | 1.314e-08 |
| P64_s2 | A1 r_e>=1a | 20 | 0.000e+00 | 9.060e-09 | 4.806e-10 | 3.23 | 5.762e-05 / 1.185e-05 (cubic 3.705e-05 / 2.734e-06) over 18 / 18 | 2 / 0 / 0; 2 / 0 / 0 | 5.959e-08 (3) | 8.772e-09 |
| P64_s2 | A2 | 20 | 0.000e+00 | 4.523e-10 | 1.269e-10 | 1.80 | 2.758e-04 / 4.894e-05 (cubic 2.743e-04 / 4.866e-05) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 2.039e-09 (3) | 1.390e-07 |
| P64_s2 | B | 20 | 0.000e+00 | 9.069e-09 | 5.574e-10 | 3.72 | - | - | 6.622e-09 (3) | 7.696e-07 |
| S4_128_s8 | A1 r_e>=1a | 20 | 0.000e+00 | 3.557e-11 | 4.217e-12 | 2.40 | 3.587e-11 / 4.239e-12 (cubic 3.592e-11 / 4.237e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 4.069e-09 (3) | 3.417e-10 |
| S4_128_s8 | A2 | 20 | 0.000e+00 | 2.946e-11 | 3.523e-12 | 3.09 | 2.347e-10 / 2.394e-11 (cubic 2.350e-10 / 2.397e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 2.433e-09 (3) | 3.898e-09 |
| S4_128_s8 | B | 20 | 0.000e+00 | 5.086e-11 | 3.078e-12 | 3.98 | - | - | 5.506e-09 (3) | 3.816e-09 |
| P64_s4_tau4 | A1 r_e>=1a | 20 | 0.000e+00 | 3.381e-09 | 6.298e-10 | 2.23 | 8.978e-09 / 9.726e-10 (cubic 8.949e-09 / 9.709e-10) over 18 / 18 | 2 / 0 / 0; 2 / 0 / 0 | 4.502e-09 (3) | 2.275e-09 |
| P64_s4_tau4 | A2 | 20 | 0.000e+00 | 3.479e-11 | 7.492e-12 | 2.54 | 9.081e-09 / 7.808e-10 (cubic 9.037e-09 / 7.771e-10) over 19 / 19 | 1 / 0 / 0; 1 / 0 / 0 | 9.703e-10 (3) | 3.847e-08 |
| P64_s4_tau4 | B | 20 | 0.000e+00 | 6.196e-11 | 8.638e-12 | 3.73 | - | - | 1.414e-09 (3) | 6.585e-09 |
| W2 | A1 r_e>=1a | 20 | 0.000e+00 | 1.715e-09 | 1.459e-10 | 3.04 | 1.712e-09 / 1.461e-10 (cubic 1.711e-09 / 1.461e-10) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 3.479e-08 (3) | 8.902e-10 |
| W2 | A2 | 20 | 0.000e+00 | 2.621e-10 | 3.558e-11 | 2.14 | 2.565e-10 / 3.727e-11 (cubic 2.544e-10 / 3.712e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.007e-08 (3) | 2.699e-08 |
| W2 | B | 20 | 0.000e+00 | 2.131e-10 | 3.588e-11 | 3.19 | - | - | 8.955e-09 (3) | 5.867e-09 |
| P64_s8 | A1 r_e>=1a | 20 | 0.000e+00 | 1.966e-10 | 2.371e-11 | 3.09 | 1.959e-10 / 2.376e-11 (cubic 1.962e-10 / 2.379e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 6.167e-09 (3) | 5.390e-10 |
| P64_s8 | A2 | 20 | 0.000e+00 | 5.126e-11 | 8.390e-12 | 2.05 | 5.337e-11 / 7.591e-12 (cubic 5.349e-11 / 7.544e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.534e-09 (3) | 4.248e-08 |
| P64_s8 | B | 20 | 0.000e+00 | 9.343e-11 | 9.484e-12 | 3.36 | - | - | 1.171e-08 (3) | 1.904e-09 |
| S4_128_s16 | A1 r_e>=1a | 20 | 0.000e+00 | 2.014e-11 | 2.288e-12 | 2.88 | 2.005e-11 / 2.256e-12 (cubic 2.003e-11 / 2.265e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 2.099e-09 (3) | 1.780e-10 |
| S4_128_s16 | A2 | 20 | 0.000e+00 | 1.473e-11 | 1.526e-12 | 3.14 | 1.474e-11 / 1.517e-12 (cubic 1.474e-11 / 1.517e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 9.888e-10 (3) | 1.687e-10 |
| S4_128_s16 | B | 20 | 0.000e+00 | 1.978e-11 | 2.451e-12 | 2.47 | - | - | 3.069e-09 (3) | 1.612e-09 |
| M3c_s4_96 | A1 r_e>=1a | 20 | 0.000e+00 | 1.821e-11 | 2.081e-12 | 3.05 | 1.821e-11 / 2.082e-12 (cubic 1.821e-11 / 2.082e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 3.060e-10 (3) | 2.169e-10 |
| M3c_s4_96 | A2 | 20 | 0.000e+00 | 1.613e-11 | 2.454e-12 | 2.61 | 1.613e-11 / 2.454e-12 (cubic 1.613e-11 / 2.454e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.066e-09 (3) | 1.633e-10 |
| P64_s4_tau2 | A1 r_e>=1a | 20 | 0.000e+00 | 4.605e-11 | 1.143e-11 | 2.41 | 4.606e-11 / 1.143e-11 (cubic 4.606e-11 / 1.143e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 8.648e-11 (3) | 2.365e-11 |
| P64_s4_newton | A1 r_e>=1a | 20 | 0.000e+00 | 5.452e-11 | 9.184e-12 | 2.76 | 5.452e-11 / 9.184e-12 (cubic 5.452e-11 / 9.184e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.540e-09 (3) | 9.657e-10 |
| ISO_64_s4 | A1 r_e>=1a | 20 | 0.000e+00 | 7.909e-11 | 8.263e-12 | 3.74 | 7.909e-11 / 8.264e-12 (cubic 7.909e-11 / 8.264e-12) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 3.665e-09 (3) | 2.756e-09 |
| ISO_64_s4 | B | 1 | 0.000e+00 | 1.063e-12 | 7.838e-14 | - | - | - | 2.253e-11 (1) | 1.603e-11 |
| S4_128_s4 | A1 r_e>=1a | 20 | 0.000e+00 | 8.596e-12 | 4.208e-13 | 3.31 | 8.596e-12 / 4.208e-13 (cubic 8.596e-12 / 4.209e-13) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 4.052e-10 (3) | 1.426e-10 |
| ISO_128_s4 | A1 r_e>=1a | 20 | 0.000e+00 | 1.279e-11 | 5.874e-13 | 4.83 | 1.279e-11 / 5.874e-13 (cubic 1.279e-11 / 5.874e-13) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 4.700e-10 (3) | 3.647e-10 |
| ISO_128_s4_plain | A1 r_e>=1a | 20 | 0.000e+00 | 1.277e-11 | 5.915e-13 | 4.80 | 1.277e-11 / 5.915e-13 (cubic 1.277e-11 / 5.914e-13) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 4.707e-10 (3) | 5.971e-10 |
| P64_s4_heldsrc | A1 r_e>=1a | 20 | 0.000e+00 | 1.073e-09 | 9.143e-11 | 3.45 | 1.069e-09 / 9.111e-11 (cubic 1.070e-09 / 9.104e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 4.172e-08 (3) | 4.467e-10 |
| P64_s4_heldsrc | A2 | 20 | 0.000e+00 | 9.454e-10 | 8.120e-11 | 3.55 | 9.509e-10 / 8.865e-11 (cubic 9.632e-10 / 8.935e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.544e-08 (3) | 1.296e-08 |
| P64_s4_heldsrc | B | 20 | 0.000e+00 | 1.916e-09 | 2.587e-10 | 2.40 | - | - | 1.013e-07 (3) | 1.195e-08 |
| P64_s4_widened | A1 r_e>=1a | 20 | 0.000e+00 | 4.138e-10 | 6.082e-11 | 2.46 | 4.285e-10 / 6.060e-11 (cubic 4.274e-10 / 6.065e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.193e-08 (3) | 7.414e-10 |
| P64_s4_widened | A2 | 20 | 0.000e+00 | 1.215e-10 | 3.286e-11 | 1.70 | 2.433e-10 / 2.807e-11 (cubic 2.260e-10 / 2.708e-11) over 20 / 20 | 0 / 0 / 0; 0 / 0 / 0 | 1.037e-08 (3) | 4.692e-08 |
| P64_s4_widened | B | 20 | 0.000e+00 | 2.457e-10 | 2.553e-11 | 3.05 | - | - | 3.679e-09 (3) | 9.804e-10 |

Members without B rows (no B launch received at 4 r_s inside the solver-certified record interval; rz_v3_reint_v1 mode B stops on an empty event set there): M3c_s4_96, P64_s4_tau2, P64_s4_newton, S4_128_s4, ISO_128_s4, ISO_128_s4_plain. DOP853 events: A1 48, A2 30, B 28 (106 in all), 0 capped. The B events of v3's runs equal the exact-release pass at the same launch labels (record, δ: L_T, T_e, r_e) on 10 of 10 members with B rows.

**Table R2v4. Empirical field sensitivity on the headline events, every class** (NOT an error bound; no bound on the background field error is established): L_T re-integrated with the fields perturbed by (P1) the record's own iterate difference rescaled to max ε, (P2) δN = ε sin(r/σ), (P3) δN = ε, each sign. A rows: rz_v4_reint_v1.py on the exact-release events (clocks, released at T_rel exactly, and the receiver re-evaluated on the perturbed fields); B rows from v3: the launch event held fixed ('B', rz_v3_reint_v1.py) and the launch relabelled by the B launch rule on the perturbed fields ('B relabelled', rz_v3_reint_v3.py). Per member and class: max abs Δ L_T over the events (at the top event) per model, and the events lost (the perturbed ray no longer meets the clock, or the perturbed B front is not received by the last record), of events × 6.

| member | class | ε | P1: max (top) | P2: max (top) | P3: max (top) | events lost (of events × 6) |
|---|---|---|---|---|---|---|
| P64_s4 | A1 r_e>=1a | 0.001 | 0.010 (0.010) | 0.236 (0.236) | 0.039 (0.039) | 0 of 120 |
| P64_s4 | A1 r_e>=1a | 0.0001 | 0.001 (9.81e-04) | 0.024 (0.024) | 0.004 (0.004) | 0 of 120 |
| P64_s4 | A2 | 0.001 | 0.011 (0.011) | 0.242 (0.242) | 0.046 (0.046) | 0 of 120 |
| P64_s4 | A2 | 0.0001 | 0.001 (0.001) | 0.024 (0.024) | 0.004 (0.004) | 0 of 120 |
| P64_s4 | B | 0.001 | 1.542 (1.542) | 1.623 (1.623) | 2.752 (2.752) | 33 of 120 |
| P64_s4 | B | 0.0001 | 0.297 (0.297) | 0.325 (0.325) | 0.978 (0.978) | 9 of 120 |
| P64_s4 | B relabelled | 0.001 | 1.542 (1.542) | 3.147 (3.147) | 2.752 (2.752) | 28 of 120 |
| P64_s4 | B relabelled | 0.0001 | 0.297 (0.297) | 0.325 (0.325) | 0.978 (0.978) | 7 of 120 |
| P64_s2 | A1 r_e>=1a | 0.001 | 0.004 (0.004) | 0.393 (0.391) | 0.043 (0.043) | 0 of 120 |
| P64_s2 | A1 r_e>=1a | 0.0001 | 3.87e-04 (3.87e-04) | 0.039 (0.039) | 0.004 (0.004) | 0 of 120 |
| P64_s2 | A2 | 0.001 | 0.004 (0.004) | 0.352 (0.352) | 0.039 (0.039) | 0 of 120 |
| P64_s2 | A2 | 0.0001 | 3.56e-04 (3.56e-04) | 0.035 (0.035) | 0.004 (0.004) | 0 of 120 |
| P64_s2 | B | 0.001 | 2.107 (0.393) | 2.977 (2.977) | 3.653 (3.653) | 37 of 120 |
| P64_s2 | B | 0.0001 | 0.090 (0.033) | 2.116 (1.095) | 2.407 (1.531) | 3 of 120 |
| P64_s2 | B relabelled | 0.001 | 0.764 (0.393) | 5.476 (5.476) | 3.653 (3.653) | 4 of 120 |
| P64_s2 | B relabelled | 0.0001 | 0.056 (0.033) | 1.095 (1.095) | 1.531 (1.531) | 2 of 120 |
| S4_128_s8 | A1 r_e>=1a | 0.001 | 3.87e-04 (3.42e-04) | 0.074 (0.073) | 0.011 (0.009) | 0 of 120 |
| S4_128_s8 | A1 r_e>=1a | 0.0001 | 3.86e-05 (3.40e-05) | 0.007 (0.007) | 0.001 (8.54e-04) | 0 of 120 |
| S4_128_s8 | A2 | 0.001 | 0.022 (0.022) | 0.033 (0.032) | 0.040 (0.040) | 0 of 120 |
| S4_128_s8 | A2 | 0.0001 | 0.002 (0.002) | 0.003 (0.003) | 0.004 (0.004) | 0 of 120 |
| S4_128_s8 | B | 0.001 | 0.429 (0.429) | 1.283 (1.283) | 1.383 (1.383) | 37 of 120 |
| S4_128_s8 | B | 0.0001 | 0.057 (0.057) | 0.221 (0.221) | 0.245 (0.245) | 3 of 120 |
| S4_128_s8 | B relabelled | 0.001 | 0.429 (0.429) | 1.283 (1.283) | 1.383 (1.383) | 15 of 120 |
| S4_128_s8 | B relabelled | 0.0001 | 0.057 (0.057) | 0.221 (0.221) | 0.245 (0.245) | 3 of 120 |
| P64_s4_tau4 | A1 r_e>=1a | 0.001 | 0.036 (0.035) | 0.233 (0.232) | 0.324 (0.324) | 0 of 120 |
| P64_s4_tau4 | A1 r_e>=1a | 0.0001 | 0.003 (0.003) | 0.023 (0.023) | 0.028 (0.028) | 0 of 120 |
| P64_s4_tau4 | A2 | 0.001 | 0.012 (0.011) | 0.145 (0.145) | 0.031 (0.031) | 0 of 120 |
| P64_s4_tau4 | A2 | 0.0001 | 0.001 (0.001) | 0.014 (0.014) | 0.003 (0.003) | 0 of 120 |
| P64_s4_tau4 | B | 0.001 | 0.345 (0.113) | 0.714 (0.692) | 0.901 (0.549) | 29 of 120 |
| P64_s4_tau4 | B | 0.0001 | 0.028 (0.010) | 0.133 (0.049) | 0.237 (0.097) | 1 of 120 |
| P64_s4_tau4 | B relabelled | 0.001 | 0.806 (0.113) | 1.235 (0.844) | 0.901 (0.549) | 11 of 120 |
| P64_s4_tau4 | B relabelled | 0.0001 | 0.740 (0.010) | 0.769 (0.049) | 0.237 (0.097) | 0 of 120 |
| W2 | A1 r_e>=1a | 0.001 | 0.001 (0.001) | 0.049 (0.036) | 0.009 (0.006) | 0 of 120 |
| W2 | A1 r_e>=1a | 0.0001 | 1.35e-04 (1.33e-04) | 0.005 (0.003) | 8.39e-04 (5.82e-04) | 0 of 120 |
| W2 | A2 | 0.001 | 0.002 (0.002) | 0.062 (0.058) | 0.017 (0.014) | 0 of 120 |
| W2 | A2 | 0.0001 | 1.79e-04 (1.52e-04) | 0.006 (0.006) | 0.002 (0.001) | 0 of 120 |
| W2 | B | 0.001 | 0.007 (2.57e-04) | 0.068 (0.066) | 0.096 (0.038) | 0 of 120 |
| W2 | B | 0.0001 | 7.26e-04 (1.56e-05) | 0.007 (0.007) | 0.010 (0.002) | 0 of 120 |
| W2 | B relabelled | 0.001 | 0.176 (2.57e-04) | 0.524 (0.172) | 0.096 (0.038) | 0 of 120 |
| W2 | B relabelled | 0.0001 | 0.168 (1.56e-05) | 0.170 (0.007) | 0.010 (0.002) | 0 of 120 |
| P64_s8 | A1 r_e>=1a | 0.001 | 0.005 (0.004) | 0.012 (0.011) | 0.008 (0.006) | 0 of 120 |
| P64_s8 | A1 r_e>=1a | 0.0001 | 4.59e-04 (4.48e-04) | 0.001 (0.001) | 8.13e-04 (6.16e-04) | 0 of 120 |
| P64_s8 | A2 | 0.001 | 0.008 (0.008) | 0.020 (0.020) | 0.015 (0.013) | 0 of 120 |
| P64_s8 | A2 | 0.0001 | 8.25e-04 (7.85e-04) | 0.002 (0.002) | 0.001 (0.001) | 0 of 120 |
| P64_s8 | B | 0.001 | 0.004 (0.004) | 0.032 (0.011) | 0.035 (0.010) | 0 of 120 |
| P64_s8 | B | 0.0001 | 4.17e-04 (3.60e-04) | 0.003 (0.001) | 0.003 (7.38e-04) | 0 of 120 |
| P64_s8 | B relabelled | 0.001 | 0.063 (0.007) | 0.084 (0.034) | 0.035 (0.010) | 0 of 120 |
| P64_s8 | B relabelled | 0.0001 | 4.17e-04 (3.60e-04) | 0.043 (0.001) | 0.003 (7.38e-04) | 0 of 120 |
| S4_128_s16 | A1 r_e>=1a | 0.001 | 0.001 (0.001) | 0.008 (0.008) | 0.004 (0.003) | 0 of 120 |
| S4_128_s16 | A1 r_e>=1a | 0.0001 | 1.34e-04 (1.32e-04) | 8.18e-04 (7.64e-04) | 3.67e-04 (2.72e-04) | 0 of 120 |
| S4_128_s16 | A2 | 0.001 | 0.004 (0.004) | 0.011 (0.011) | 0.016 (0.015) | 0 of 120 |
| S4_128_s16 | A2 | 0.0001 | 4.00e-04 (3.79e-04) | 0.001 (0.001) | 0.002 (0.002) | 0 of 120 |
| S4_128_s16 | B | 0.001 | 0.002 (0.002) | 0.029 (0.008) | 0.034 (0.007) | 0 of 120 |
| S4_128_s16 | B | 0.0001 | 2.26e-04 (1.96e-04) | 0.003 (6.38e-04) | 0.003 (3.22e-04) | 0 of 120 |
| S4_128_s16 | B relabelled | 0.001 | 0.030 (0.002) | 0.115 (0.041) | 0.034 (0.007) | 0 of 120 |
| S4_128_s16 | B relabelled | 0.0001 | 2.26e-04 (1.96e-04) | 0.022 (0.003) | 0.003 (3.22e-04) | 0 of 120 |
| M3c_s4_96 | A1 r_e>=1a | 0.001 | 1.19e-05 (7.42e-06) | 0.001 (0.001) | 0.002 (0.002) | 0 of 120 |
| M3c_s4_96 | A1 r_e>=1a | 0.0001 | 1.19e-06 (7.42e-07) | 1.39e-04 (1.15e-04) | 2.21e-04 (2.19e-04) | 0 of 120 |
| M3c_s4_96 | A2 | 0.001 | 2.38e-07 (9.94e-09) | 0.002 (0.002) | 0.003 (0.003) | 0 of 120 |
| M3c_s4_96 | A2 | 0.0001 | 2.38e-08 (9.94e-10) | 1.97e-04 (1.96e-04) | 2.76e-04 (2.76e-04) | 0 of 120 |
| P64_s4_tau2 | A1 r_e>=1a | 0.001 | 2.42e-06 (3.20e-07) | 0.002 (0.002) | 0.002 (0.002) | 0 of 120 |
| P64_s4_tau2 | A1 r_e>=1a | 0.0001 | 2.42e-07 (3.20e-08) | 1.63e-04 (1.63e-04) | 2.05e-04 (2.04e-04) | 0 of 120 |
| P64_s4_newton | A1 r_e>=1a | 0.001 | 4.84e-04 (4.84e-04) | 0.001 (0.001) | 0.002 (0.002) | 0 of 120 |
| P64_s4_newton | A1 r_e>=1a | 0.0001 | 4.84e-05 (4.84e-05) | 1.07e-04 (1.07e-04) | 2.06e-04 (2.04e-04) | 0 of 120 |
| ISO_64_s4 | A1 r_e>=1a | 0.001 | 3.81e-08 (3.81e-08) | 0.001 (4.89e-04) | 0.002 (0.002) | 0 of 120 |
| ISO_64_s4 | A1 r_e>=1a | 0.0001 | 3.81e-09 (3.81e-09) | 1.41e-04 (4.87e-05) | 2.01e-04 (2.00e-04) | 0 of 120 |
| ISO_64_s4 | B | 0.001 | 3.23e-08 (3.23e-08) | 0.001 (0.001) | 0.002 (0.002) | 0 of 6 |
| ISO_64_s4 | B | 0.0001 | 3.23e-09 (3.23e-09) | 1.30e-04 (1.30e-04) | 2.12e-04 (2.12e-04) | 0 of 6 |
| ISO_64_s4 | B relabelled | 0.001 | 3.23e-08 (3.23e-08) | 0.004 (0.004) | 0.002 (0.002) | 0 of 6 |
| ISO_64_s4 | B relabelled | 0.0001 | 3.23e-09 (3.23e-09) | 5.55e-04 (5.55e-04) | 2.12e-04 (2.12e-04) | 0 of 6 |
| S4_128_s4 | A1 r_e>=1a | 0.001 | 5.30e-09 (2.51e-09) | 0.001 (1.14e-04) | 0.002 (0.002) | 0 of 120 |
| S4_128_s4 | A1 r_e>=1a | 0.0001 | 5.30e-10 (2.51e-10) | 1.25e-04 (1.11e-05) | 2.02e-04 (2.02e-04) | 0 of 120 |
| ISO_128_s4 | A1 r_e>=1a | 0.001 | 1.83e-06 (1.45e-07) | 0.001 (2.28e-04) | 0.002 (0.002) | 0 of 120 |
| ISO_128_s4 | A1 r_e>=1a | 0.0001 | 1.83e-07 (1.45e-08) | 1.34e-04 (2.25e-05) | 2.03e-04 (2.02e-04) | 0 of 120 |
| ISO_128_s4_plain | A1 r_e>=1a | 0.001 | 2.87e-09 (1.92e-09) | 0.001 (2.28e-04) | 0.002 (0.002) | 0 of 120 |
| ISO_128_s4_plain | A1 r_e>=1a | 0.0001 | 2.87e-10 (1.92e-10) | 1.34e-04 (2.25e-05) | 2.03e-04 (2.02e-04) | 0 of 120 |
| P64_s4_heldsrc | A1 r_e>=1a | 0.001 | 0.003 (0.002) | 0.014 (0.012) | 0.003 (0.002) | 0 of 120 |
| P64_s4_heldsrc | A1 r_e>=1a | 0.0001 | 2.88e-04 (2.26e-04) | 0.001 (0.001) | 3.36e-04 (1.51e-04) | 0 of 120 |
| P64_s4_heldsrc | A2 | 0.001 | 0.004 (0.003) | 0.014 (0.014) | 0.003 (9.14e-04) | 0 of 120 |
| P64_s4_heldsrc | A2 | 0.0001 | 4.09e-04 (3.50e-04) | 0.001 (0.001) | 2.97e-04 (8.11e-05) | 0 of 120 |
| P64_s4_heldsrc | B | 0.001 | 0.006 (0.004) | 0.040 (0.019) | 0.061 (0.031) | 0 of 120 |
| P64_s4_heldsrc | B | 0.0001 | 6.17e-04 (4.11e-04) | 0.004 (0.002) | 0.006 (0.002) | 0 of 120 |
| P64_s4_heldsrc | B relabelled | 0.001 | 0.115 (0.008) | 0.399 (0.151) | 0.079 (0.031) | 0 of 120 |
| P64_s4_heldsrc | B relabelled | 0.0001 | 0.059 (4.11e-04) | 0.116 (0.006) | 0.007 (0.002) | 0 of 120 |
| P64_s4_widened | A1 r_e>=1a | 0.001 | 0.005 (0.004) | 0.005 (0.005) | 0.004 (0.001) | 0 of 120 |
| P64_s4_widened | A1 r_e>=1a | 0.0001 | 4.60e-04 (4.27e-04) | 5.15e-04 (4.93e-04) | 3.48e-04 (1.11e-04) | 0 of 120 |
| P64_s4_widened | A2 | 0.001 | 0.004 (0.003) | 0.013 (0.011) | 0.019 (0.017) | 0 of 120 |
| P64_s4_widened | A2 | 0.0001 | 3.59e-04 (3.00e-04) | 0.001 (0.001) | 0.002 (0.002) | 0 of 120 |
| P64_s4_widened | B | 0.001 | 0.009 (0.009) | 0.030 (0.023) | 0.059 (0.037) | 0 of 120 |
| P64_s4_widened | B | 0.0001 | 9.32e-04 (9.32e-04) | 0.003 (0.002) | 0.006 (0.003) | 0 of 120 |
| P64_s4_widened | B relabelled | 0.001 | 0.026 (0.009) | 0.063 (0.038) | 0.059 (0.037) | 0 of 120 |
| P64_s4_widened | B relabelled | 0.0001 | 9.32e-04 (9.32e-04) | 0.034 (0.002) | 0.006 (0.003) | 0 of 120 |

**Table R3v4. The regime of the headline events, every class**: for each class's top event (A: the exact-release pass; B: as v3): L_T, I, c_D and ln A_D; the emitted lattice momentum a k_e a quantum would need for a received frequency ω r_s = 0.1 and 1, (i) with ω the RECEIVED COORDINATE FREQUENCY (a k_e = (ω r_s)(a/r_s) e^I/c_D) and (ii) with ω the static receiver's LOCAL frequency (a k_e = (ω r_s)(a/r_s) A_D e^I/c_D); the largest a k along the path (coordinate ω r_s = 1); and L_T − ln(2/(ak_e)²) at ak_e 0.1 / 0.2, the event's distance above reviewer B's crossover estimate of the continuum approximation (a breakdown scale, not a cap on received redshift).

| member | class | L_T | I | c_D | ln A_D | a k_e, coordinate ω r_s = 0.1 / 1 | a k_e, local ω r_s = 0.1 / 1 | largest a k along the path (coordinate, ω r_s = 1) | L_T − ln(2/(ak_e)²) at ak_e 0.1 / 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | A1 r_e>=1a | 10.456 | 9.910 | 0.5793 | -0.1172 | 54.279 / 542.786 | 48.274 / 482.738 | 796.771 | 5.157 / 6.544 |
| P64_s4 | A2 | 9.452 | 8.906 | 0.5793 | -0.1172 | 19.902 / 199.023 | 17.701 / 177.005 | 199.212 | 4.154 / 5.540 |
| P64_s4 | B (δ 0.100 a) | 9.128 | 8.581 | 0.5790 | -0.1175 | 14.386 / 143.857 | 12.791 / 127.914 | 143.857 | 3.829 / 5.216 |
| P64_s2 | A1 r_e>=1a | 15.676 | 15.161 | 0.5973 | -0.1072 | 10040.466 / 1.00e+05 | 9019.902 / 90199.015 | 1.44e+05 | 10.378 / 11.764 |
| P64_s2 | A2 | 14.395 | 13.880 | 0.5973 | -0.1072 | 2789.492 / 27894.921 | 2505.954 / 25059.537 | 28009.340 | 9.097 / 10.483 |
| P64_s2 | B (δ -1.000 a) | 10.033 | 9.505 | 0.5896 | -0.1143 | 35.583 / 355.826 | 31.739 / 317.395 | 355.913 | 4.735 / 6.121 |
| S4_128_s8 | A1 r_e>=1a | 3.434 | 2.849 | 0.5569 | -0.1438 | 0.024 / 0.242 | 0.021 / 0.210 | 0.342 | -1.864 / -0.478 |
| S4_128_s8 | A2 | 7.754 | 7.231 | 0.5925 | -0.1220 | 1.821 / 18.209 | 1.612 / 16.118 | 18.210 | 2.456 / 3.842 |
| S4_128_s8 | B (δ -2.000 a) | 7.723 | 7.199 | 0.5924 | -0.1221 | 1.765 / 17.652 | 1.562 / 15.624 | 17.672 | 2.425 / 3.811 |
| P64_s4_tau4 | A1 r_e>=1a | 6.107 | 5.558 | 0.5773 | -0.1248 | 0.702 / 7.017 | 0.619 / 6.194 | 22.082 | 0.809 / 2.195 |
| P64_s4_tau4 | A2 | 7.184 | 6.634 | 0.5773 | -0.1248 | 2.059 / 20.590 | 1.817 / 18.173 | 22.082 | 1.885 / 3.272 |
| P64_s4_tau4 | B (δ 0.462 a) | 6.596 | 6.043 | 0.5756 | -0.1264 | 1.144 / 11.438 | 1.008 / 10.081 | 11.438 | 1.298 / 2.684 |
| W2 | A1 r_e>=1a | 3.517 | 2.894 | 0.5360 | -0.1558 | 0.053 / 0.526 | 0.045 / 0.450 | 0.712 | -1.781 / -0.395 |
| W2 | A2 | 4.543 | 3.972 | 0.5651 | -0.1367 | 0.147 / 1.468 | 0.128 / 1.280 | 2.056 | -0.756 / 0.631 |
| W2 | B (δ 0.770 a) | 4.877 | 4.306 | 0.5649 | -0.1366 | 0.205 / 2.051 | 0.179 / 1.789 | 2.051 | -0.421 / 0.965 |
| P64_s8 | A1 r_e>=1a | 3.241 | 2.603 | 0.5286 | -0.1259 | 0.040 / 0.399 | 0.035 / 0.352 | 0.638 | -2.058 / -0.671 |
| P64_s8 | A2 | 3.172 | 2.580 | 0.5534 | -0.1262 | 0.037 / 0.373 | 0.033 / 0.328 | 0.373 | -2.127 / -0.740 |
| P64_s8 | B (δ 2.142 a) | 3.708 | 3.070 | 0.5286 | -0.1259 | 0.064 / 0.637 | 0.056 / 0.562 | 0.637 | -1.590 / -0.204 |
| S4_128_s16 | A1 r_e>=1a | 3.285 | 2.646 | 0.5276 | -0.1272 | 0.021 / 0.209 | 0.018 / 0.184 | 0.302 | -2.013 / -0.627 |
| S4_128_s16 | A2 | 2.782 | 2.196 | 0.5563 | -0.1278 | 0.013 / 0.126 | 0.011 / 0.111 | 0.182 | -2.516 / -1.130 |
| S4_128_s16 | B (δ -2.000 a) | 3.654 | 3.015 | 0.5276 | -0.1272 | 0.030 / 0.302 | 0.027 / 0.266 | 0.302 | -1.644 / -0.258 |
| M3c_s4_96 | A1 r_e>=1a | 0.716 | 0.073 | 0.5257 | -0.1288 | 0.002 / 0.021 | 0.002 / 0.019 | 0.032 | -4.583 / -3.196 |
| M3c_s4_96 | A2 | 0.922 | 0.274 | 0.5231 | -0.1305 | 0.003 / 0.026 | 0.002 / 0.023 | 0.032 | -4.376 / -2.990 |
| P64_s4_tau2 | A1 r_e>=1a | 0.745 | 0.100 | 0.5245 | -0.1285 | 0.003 / 0.033 | 0.003 / 0.029 | 0.040 | -4.553 / -3.167 |
| P64_s4_newton | A1 r_e>=1a | 0.724 | 0.085 | 0.5276 | -0.1266 | 0.003 / 0.032 | 0.003 / 0.028 | 0.039 | -4.574 / -3.188 |
| ISO_64_s4 | A1 r_e>=1a | 0.720 | 0.101 | 0.5386 | -0.1200 | 0.003 / 0.032 | 0.003 / 0.028 | 0.040 | -4.579 / -3.192 |
| ISO_64_s4 | B (δ 128.000 a) | 0.766 | 0.147 | 0.5384 | -0.1201 | 0.003 / 0.034 | 0.003 / 0.030 | 0.034 | -4.532 / -3.146 |
| S4_128_s4 | A1 r_e>=1a | 0.701 | 0.050 | 0.5214 | -0.1303 | 0.002 / 0.016 | 0.001 / 0.014 | 0.016 | -4.597 / -3.211 |
| ISO_128_s4 | A1 r_e>=1a | 0.703 | 0.051 | 0.5211 | -0.1305 | 0.002 / 0.016 | 0.001 / 0.014 | 0.016 | -4.595 / -3.209 |
| ISO_128_s4_plain | A1 r_e>=1a | 0.703 | 0.051 | 0.5211 | -0.1305 | 0.002 / 0.016 | 0.001 / 0.014 | 0.016 | -4.595 / -3.209 |
| P64_s4_heldsrc | A1 r_e>=1a | 3.460 | 2.852 | 0.5446 | -0.1200 | 0.050 / 0.497 | 0.044 / 0.441 | 0.708 | -1.839 / -0.452 |
| P64_s4_heldsrc | A2 | 2.787 | 2.179 | 0.5443 | -0.1194 | 0.025 / 0.254 | 0.023 / 0.225 | 0.254 | -2.511 / -1.125 |
| P64_s4_heldsrc | B (δ -2.000 a) | 3.803 | 3.197 | 0.5454 | -0.1219 | 0.070 / 0.701 | 0.062 / 0.620 | 0.701 | -1.495 / -0.109 |
| P64_s4_widened | A1 r_e>=1a | 3.708 | 3.073 | 0.5302 | -0.1249 | 0.064 / 0.637 | 0.056 / 0.562 | 0.896 | -1.591 / -0.204 |
| P64_s4_widened | A2 | 4.050 | 3.464 | 0.5564 | -0.1280 | 0.090 / 0.897 | 0.079 / 0.789 | 0.898 | -1.248 / 0.138 |
| P64_s4_widened | B (δ 0.462 a) | 4.239 | 3.652 | 0.5562 | -0.1282 | 0.108 / 1.083 | 0.095 / 0.953 | 1.083 | -1.059 / 0.327 |

**Table R4v4. The linear zone's half-width Δ over root records** (as v2, read from the exact-release passes, whose root tables are the records'): quartiles of Δ (in a) over the records with an outer root inside the solver-certified record interval, and 1/(k_e Δ) at the median Δ for ak_e = 0.1 and 0.2 (geometric optics needs it ≪ 1).

| member | root records | Δ (a): q10 / median / q90 | 1/(k_e Δ) at the median: ak_e 0.1 / 0.2 |
|---|---|---|---|
| P64_s4 | 207 | 2.25 / 6.00 / 9.25 | 1.67 / 0.83 |
| P64_s2 | 225 | 3.75 / 5.25 / 7.90 | 1.90 / 0.95 |
| S4_128_s8 | 360 | 2.50 / 8.25 / 15.53 | 1.21 / 0.61 |
| P64_s4_tau4 | 345 | 0.50 / 5.50 / 8.75 | 1.82 / 0.91 |
| W2 | 112 | 1.28 / 6.75 / 9.25 | 1.48 / 0.74 |
| P64_s8 | 22 | 1.80 / 6.25 / 10.00 | 1.60 / 0.80 |
| S4_128_s16 | 46 | 3.62 / 18.00 / 28.25 | 0.56 / 0.28 |
| M3c_s4_96 | 12 | 3.78 / 12.12 / 19.73 | 0.82 / 0.41 |
| P64_s4_tau2 | 27 | 4.55 / 6.50 / 9.65 | 1.54 / 0.77 |
| P64_s4_newton | 4 | 2.48 / 6.88 / 9.00 | 1.45 / 0.73 |
| ISO_64_s4 | 10 | 5.70 / 6.25 / 9.00 | 1.60 / 0.80 |
| ISO_128_s4 | 8 | 12.10 / 17.88 / 27.90 | 0.56 / 0.28 |
| P64_s4_heldsrc | 21 | 1.25 / 6.25 / 9.75 | 1.60 / 0.80 |
| P64_s4_widened | 46 | 0.50 / 4.75 / 9.75 | 2.11 / 1.05 |

## 5. Propagation variants and interior completions (rz_v4_variants_v4.py; out/v41_variants_v4/variants_tables.md, verbatim)

Producing script: code/rz_v4_variants_v4.py (md5 ab7af70a3dd306ccf5354877b482d15e). Table file: out/v41_variants_v4/variants_tables.md (md5 4a072dc22808d1b12a23830b6c711e35). Launcher code/v41tables_run1.sh. Made from v4's code/rz_v4_variants_v3.py (md5 79c44a143659991cd9035460fce16d37) by code/mk_variants_v4_v4_patch.py (md5 5ee46a9b63dc2bb4036d9d30d242eda6): the fixed-emission statuses named as sampled-map statuses; its json equals v4's out/v4_variants_v3/variants_tables.json: yes. Data as v4.

**Table V1v4. Propagation variants on the headline population, at fixed arrival and at fixed emission label** (rz_v4_variants_v4.py; the stored passes, rz_rays_v4, main and variants alike releasing the later clocks one step early, so the A2 columns compare like implementations). Signed changes are VARIANT MINUS MAIN: dL_T = L_T(variant) − L_T(main), dT_D = T_D(variant) − T_D(main); each 'largest' is the signed change of largest magnitude. Fixed arrival (clock, T_D): received in both / main only / variant only; the largest dL_T over all matched and over L_T ≥ 5 in the main run; dL_T at the main run's headline events of A1 r_e ≥ 1 a / A1 sub-site / A2 (the TESTED A headline events; B is compared by label in the last column, not at fixed arrival). Fixed emission label (clock, τ_e): the main's received events by their status on the variant's SAMPLED map, its consecutive arrivals 1 a apart inside its certified interval (bracketed / label past the variant's last sampled received emission of that clock / before its first / no adjacent bracket; sampled-map statuses: non-receipt is not verified, since no event is located directly), the largest dL_T among those received in the variant, over L_T ≥ 5, the largest dT_D, and dL_T (dT_D) at the A1 r_e ≥ 1 a and A2 headline events or their status. B by label (record, δ from each pass's own root): the largest dL_T (matched; not received in the variant).

| member | variant | fixed arrival: both / main only / variant only | fixed arrival: largest dL_T all / L_T ≥ 5 | fixed arrival dL_T at the headline events: A1 r_e≥1a / A1 sub-site / A2 | fixed emission, sampled-map status: bracketed / past the last sampled / before the first sampled / no adjacent bracket | fixed emission: largest dL_T all / L_T ≥ 5; largest dT_D (a) | fixed emission at the headline events: A1 r_e≥1a / A2 | B by label: largest dL_T (matched; not received) |
|---|---|---|---|---|---|---|---|---|
| ISO_128_s4 | app_hold_cubic (the production hold) | 6279 / 0 / 0 | 0.007 / - | -2.30e-04 / - / - | 6274 / 0 / 5 / 0 | 0.007 / -; -1.21 | -2.30e-04 (-0.02) / - | - (0; 0) |
| ISO_128_s4 | dense_lin_quintic (the dense fields) | 6279 / 0 / 0 | 2.89e-04 / - | 2.53e-04 / - / - | 6273 / 6 / 0 / 0 | 2.95e-04 / -; 0.24 | 2.53e-04 (0.07) / - | - (0; 0) |
| ISO_64_s4 | dense_lin_quintic (the dense fields) | 3710 / 1 / 0 | 0.002 / - | 0.002 / - / - | 3704 / 6 / 1 / 0 | 0.002 / -; 0.60 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / - | 0.003 (1; 0) |
| P64_s2 | app_hold_cubic (the production hold) | 17383 / 0 / 1 | -0.076 / -0.058 | -0.047 / -0.046 / -0.002 | 16277 / 1101 / 5 / 0 | -12.371 / -12.371; -992.64 | -12.371 (-992.64) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | -5.093 (3355; 13) |
| P64_s2 | dense_lin_quintic (the dense fields) | 17382 / 1 / 0 | 0.241 / 0.241 | 0.237 / 0.237 / 0.217 | 10441 / 6941 / 1 / 0 | 5.328 / 4.430; 415.08 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 2.577 (3347; 21) |
| P64_s2 | bar_lin_cubic h025 (step h/2) | 17383 / 0 / 4 | -0.003 / -0.003 | 1.44e-10 / 2.48e-08 / -0.003 | 17376 / 5 / 2 / 0 | -5.841 / -5.841; -459.31 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / -5.841 (-459.31) | 9.07e-09 (3368; 0) |
| P64_s2 | bar_pchipt_cubic (PCHIP in time) | 17383 / 0 / 0 | 0.013 / 0.002 | 1.54e-04 / -1.30e-04 / 7.45e-05 | 15054 / 2322 / 7 / 0 | 6.006 / 6.006; 467.17 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 2.349 (3366; 2) |
| P64_s2 | bar_lin_pchip (PCHIP in r) | 17383 / 0 / 0 | -8.77e-04 / -8.77e-04 | -3.94e-04 / -4.24e-04 / -4.62e-04 | 16772 / 605 / 6 / 0 | 4.623 / 4.623; 362.28 | -0.028 (-2.39) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | -0.378 (3368; 0) |
| P64_s2 | bar_lin_linear (linear in r) | 17383 / 0 / 0 | -0.036 / -0.036 | -0.027 / -0.027 / -0.030 | 16975 / 402 / 6 / 0 | -6.040 / -6.040; -474.12 | -4.703 (-369.61) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 0.505 (3367; 1) |
| P64_s4 | app_hold_cubic (the production hold) | 17167 / 2 / 1 | -0.043 / -0.043 | -0.038 / - / 0.004 | 16744 / 419 / 6 / 0 | -6.294 / -6.294; -857.50 | -6.294 (-857.50) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | -2.002 (2916; 5) |
| P64_s4 | dense_lin_quintic (the dense fields) | 17160 / 9 / 0 | 1.441 / 1.441 | 1.440 / - / 1.433 | 10074 / 7094 / 1 / 0 | 6.509 / -; 766.00 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 3.534 (2720; 201) |
| P64_s4 | bar_lin_cubic h025 (step h/2) | 17169 / 0 / 7 | -0.003 / -0.003 | -3.63e-10 / - / -0.003 | 17138 / 27 / 4 / 0 | -1.180 / -1.180; -120.24 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / -1.180 (-120.24) | 1.44e-09 (2921; 0) |
| P64_s4 | bar_pchipt_cubic (PCHIP in time) | 17169 / 0 / 0 | -0.002 / -0.002 | 2.28e-05 / - / -0.001 | 17140 / 21 / 8 / 0 | -0.591 / -0.591; -62.61 | -0.406 (-43.78) / -0.268 (-29.06) | -0.153 (2921; 0) |
| P64_s4 | bar_lin_pchip (PCHIP in r) | 17169 / 0 / 0 | -0.008 / -0.008 | 0.006 / - / 0.006 | 17140 / 23 / 6 / 0 | 0.028 / 0.028; 2.45 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 1.901 (2907; 14) |
| P64_s4 | bar_lin_linear (linear in r) | 17169 / 0 / 0 | 0.024 / 0.024 | -0.017 / - / -0.021 | 17161 / 2 / 6 / 0 | -1.192 / -1.192; -121.59 | -1.166 (-119.07) / -0.796 (-83.46) | 1.800 (2905; 16) |
| P64_s4_tau2 | app_hold_cubic (the production hold) | 3415 / 0 / 0 | -0.001 / - | -0.001 / - / - | 3410 / 0 / 5 / 0 | -0.001 / -; -0.40 | -0.001 (-0.40) / - | - (0; 0) |
| P64_s4_tau4 | app_hold_cubic (the production hold) | 17108 / 1 / 0 | 0.036 / -0.033 | -0.019 / - / -0.023 | 17042 / 60 / 7 / 0 | -1.358 / -1.358; -278.95 | -1.358 (-278.95) / -0.921 (-114.04) | -0.692 (6783; 4) |
| P64_s4_tau4 | dense_lin_quintic (the dense fields) | 17099 / 10 / 0 | 2.322 / 2.322 | 2.322 / - / 1.071 | 11819 / 5289 / 1 / 0 | 5.079 / -; 603.62 | label past the variant's last sampled received emission of the clock (non-receipt not verified) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 5.089 (6414; 373) |
| P64_s8 | app_hold_cubic (the production hold) | 16956 / 2 / 0 | -0.513 / - | -0.003 / -1.73e-04 / 0.005 | 16948 / 2 / 8 / 0 | 0.499 / -; -72.60 | -0.247 (-67.98) / 0.003 (-4.13) | 0.171 (3832; 1) |
| P64_s8 | dense_lin_quintic (the dense fields) | 16894 / 64 / 0 | 3.040 / - | 0.555 / 0.935 / 0.821 | 13373 / 3579 / 6 / 0 | 2.230 / -; 408.74 | -1.695 (201.34) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 1.938 (3656; 177) |
| S4_128_s16 | app_hold_cubic (the production hold) | 34084 / 1 / 1 | -0.582 / - | -0.002 / -1.68e-05 / -0.006 | 34043 / 35 / 7 / 0 | 0.342 / -; -76.59 | -0.115 (-74.27) / -0.012 (-19.76) | 0.113 (10016; 3) |
| S4_128_s16 | dense_lin_quintic (the dense fields) | 33965 / 120 / 0 | 3.525 / - | 0.438 / 0.992 / 1.039 | 26166 / 7916 / 3 / 0 | 2.405 / -; 939.39 | -1.533 (292.22) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 1.984 (9374; 645) |
| S4_128_s8 | app_hold_cubic (the production hold) | 34473 / 2 / 0 | -2.982 / -0.026 | -0.002 / -0.015 / -0.016 | 34418 / 51 / 6 / 0 | -2.385 / -2.385; -616.83 | -0.099 (-87.62) / -1.359 (-348.13) | -1.267 (7235; 0) |
| S4_128_s8 | dense_lin_quintic (the dense fields) | 34458 / 17 / 0 | 6.285 / 1.624 | 0.191 / 1.293 / 1.558 | 22826 / 11646 / 3 / 0 | 5.699 / -; 1152.18 | -0.084 (150.73) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 3.784 (6967; 268) |
| W2 | app_hold_cubic (the production hold) | 17039 / 1 / 0 | 3.782 / - | -0.004 / -7.49e-04 / -0.009 | 17029 / 6 / 5 / 0 | 0.868 / -; -169.49 | -0.379 (-85.19) / -0.151 (-66.66) | 0.352 (4299; 7) |
| W2 | dense_lin_quintic (the dense fields) | 17029 / 11 / 0 | 5.998 / - | 0.159 / 0.415 / 0.446 | 12736 / 4303 / 1 / 0 | 2.906 / -; 547.81 | -0.145 (80.13) / label past the variant's last sampled received emission of the clock (non-receipt not verified) | 1.591 (4184; 122) |

**Table V1b. The blueshift maxima under the variants** (FableAPI_rzchk4 item 9): per member, variant and class, the main run's most blueshifted received event inside its certified interval (its L_T < 0 and arrival T_D/r_s) and the variant's dL_T at the same (clock, T_D) ('not received' where the variant has no arrival there); B: the main run's most blueshifted received launch and the variant's change at the same launch label (record, δ).

| member | variant | A1: main L_T (T_D/r_s); dL_T | A2: main L_T (T_D/r_s); dL_T | B by label: main L_T (T_D/r_s); dL_T |
|---|---|---|---|---|
| P64_s8 | app_hold_cubic | -1.819 (28.828); 0.257 | - | -0.748 (28.778); -0.020 |
| P64_s8 | dense_lin_quintic | -1.819 (28.828); 2.915 | - | -0.748 (28.778); 0.192 |
| S4_128_s16 | app_hold_cubic | -2.541 (27.047); 0.255 | -0.586 (39.867); 0.006 | -0.629 (26.969); -0.005 |
| S4_128_s16 | dense_lin_quintic | -2.541 (27.047); 3.416 | -0.586 (39.867); 2.573 | -0.629 (26.969); 0.844 |
| S4_128_s8 | app_hold_cubic | -3.946 (28.719); 2.273 | - | -1.832 (28.702); 0.011 |
| S4_128_s8 | dense_lin_quintic | -3.946 (28.719); 6.175 | - | -1.832 (28.702); -1.170 |
| W2 | app_hold_cubic | -4.305 (29.312); 3.739 | - | -3.548 (29.312); -0.023 |
| W2 | dense_lin_quintic | -4.305 (29.312); 5.943 | - | -3.548 (29.312); -0.335 |

**Table V2v4. The interior completions of the reference collapse** (rz_v4_variants_v4.py on the exact-release passes: rz_rays_v4n for the reference, rz_rays_v4nc for the held-source and widened completions; the completions have no propagation-variant passes and are untested under them). Split at the completions' branch T/r_s 11.125 by the ARRIVAL time; signed changes COMPLETION MINUS REFERENCE. Per completion: flights ending before the branch (matched, largest dL_T); after it (matched, largest dL_T, median dL_T); at fixed emission label after the branch, the reference's received events by their status on the completion's sampled map (bracketed / label past its last sampled received emission of the clock / before its first / no adjacent bracket; non-receipt not verified) and the largest dL_T and dT_D among the received; and the largest L_T inside the solver-certified record interval for A1 with r_e ≥ 1 a, for the A1 and A2 clocks, and for A1, A2 and B (reference in brackets).

| completion | before 11.125: matched, largest dL_T | after: matched, largest dL_T, median dL_T | after, fixed emission, sampled-map status: bracketed / past the last sampled / before the first sampled / no adjacent bracket; largest dL_T; largest dT_D (a) | max L_T, A1 r_e ≥ 1 a (reference) | max L_T, A1 and A2 clocks (reference) | max L_T, A1, A2 and B (reference) |
|---|---|---|---|---|---|---|
| P64_s4_heldsrc | 2785, 0.000 | 14395, -10.207, -3.044 | 14390 / 0 / 5 / 0; -8.420; -886.51 | 3.460 (10.456) | 3.460 (10.456) | 3.803 (10.456) |
| P64_s4_widened | 2785, 0.000 | 14240, -8.763, -0.759 | 14350 / 43 / 2 / 0; -7.928; -736.20 | 3.708 (10.456) | 4.050 (10.456) | 4.239 (10.456) |

## 6. The fits (rz_v3_law_v2.py on the exact-release passes; out/v41_law_v2/law_tables.md, verbatim)

Producing script: code/rz_v3_law_v2.py (md5 24c74d4ebbbd2e4fd29374d55d6a4a5e). Table file: out/v41_law_v2/law_tables.md (md5 8e69f71756d5f5a50a37de266fffbd46). Launcher code/v41tables_run1.sh. Made from rz_v3_law_v1.py (v3, v4; md5 d53cb372a876220371ff969319925cdf) by code/mk_law_v3_v2_patch.py (md5 b99d321b4d3f83f22639693f3713be8e): the ray-mask label on every row of P64_s4_ext200; its json equals v4's out/v4_law_v1/law_tables.json: yes; its captions keep their v3 names (L1v3, L2v3). Data as v4.

Note: P64_s4_ext200 (labelled '(e, c only: ray mask unavailable)' in every row of Tables L1v3 and L2v3): records not locatable; v1 pass only, its A2 clocks released one step early; identical to P64_s4 through T/r_s 40; whole-window value unreproducible (sec. 0).

**Table L1v3. The fits per comoving clock, on the resolved population** (rz_v3_law_v2.py): backward events at 4 r_s with a hover in the outer root's linear zone, inside the solver-certified record interval, masked (r_e ≥ 1 a, clock outside the first site at emission, ray's smallest |r| ≥ 1 a). For L_T = ln(1/G_D): (K) slope / intercept / rms on the ray's own hover integral K_H; (Q) slope / rms on Q(u) = ∫κ_R(T_x(u')) du'; (E) κ_fit/κ_H; out-of-fit rms constant / exponential / K_H + c / Q + c and the winner. For L_D = ln(1/g_D): (K) slope and the out-of-fit winner. dT_x/du median: T_x is read on the h = 0.5 a step grid and the arrivals are 1 a apart, so this median takes multiples of 0.25 only (a label, not a measurement).

| member | clock | arrivals (of v2's) | u/r_s range | L_T: (K) slope / intercept / rms | L_T: (Q) slope / rms | L_T: (E) κ_fit/κ_H | L_T out-of-fit rms: const / exp / K_H / Q | L_T winner | L_D: (K) slope | L_D winner | dT_x/du median (grid-quantized) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | A1.0 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.466 / 0.439 | 0.869 / 0.113 | 0.728 | 3.854 / 1.233 / 0.482 / 0.595 | K_H | 1.028 | K_H | 0.500 |
| P64_s4 | A1.1 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.506 / 0.439 | 0.869 / 0.113 | 0.728 | 3.853 / 1.234 / 0.482 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4 | A1.2 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.548 / 0.439 | 0.869 / 0.113 | 0.728 | 3.853 / 1.235 / 0.482 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4 | A1.3 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.587 / 0.439 | 0.869 / 0.113 | 0.728 | 3.852 / 1.235 / 0.481 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4 | A1.4 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.656 / 0.438 | 0.869 / 0.114 | 0.728 | 3.852 / 1.236 / 0.481 / 0.597 | K_H | 1.027 | K_H | 0.500 |
| P64_s4 | A1.5 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.719 / 0.438 | 0.869 / 0.114 | 0.728 | 3.852 / 1.236 / 0.481 / 0.597 | K_H | 1.027 | K_H | 0.500 |
| P64_s4 | A2.6 | 1031 (1031) | 23.91..40.00 | 1.037 / 2.900 / 0.426 | 0.840 / 0.143 | 0.706 | 3.683 / 1.423 / 0.501 / 0.703 | K_H | 1.039 | K_H | 0.500 |
| P64_s4 | A2.7 | 688 (688) | 29.27..40.00 | 0.972 / 3.153 / 0.325 | 0.976 / 0.028 | 0.940 | 3.124 / 0.202 / 0.413 / 0.084 | Q | 0.974 | Q | 0.500 |
| P64_s4 | A2.8 | 511 (511) | 32.03..40.00 | 0.850 / 3.224 / 0.265 | 0.929 / 0.043 | 0.890 | 2.253 / 0.392 / 0.492 / 0.157 | Q | 0.852 | Q | 0.500 |
| P64_s4 | A2.9 | 255 (255) | 36.03..40.00 | 1.250 / 2.150 / 0.161 | 0.987 / 0.019 | 1.079 | 1.355 / 0.051 / 0.377 / 0.015 | Q | 1.252 | Q | 0.500 |
| P64_s2 | A1.0 | 0 (1138) | - | fewer than 20 resolved arrivals | - | - | - | - | - | - | - |
| P64_s2 | A1.1 | 1138 (1138) | 22.23..40.00 | 1.044 / 2.261 / 0.425 | 1.072 / 0.160 | 1.176 | 7.210 / 1.087 / 0.565 / 0.496 | Q | 1.044 | Q | 0.500 |
| P64_s2 | A1.2 | 1138 (1138) | 22.23..40.00 | 1.044 / 2.317 / 0.425 | 1.072 / 0.161 | 1.177 | 7.211 / 1.083 / 0.566 / 0.497 | Q | 1.044 | Q | 0.500 |
| P64_s2 | A1.3 | 1138 (1138) | 22.23..40.00 | 1.044 / 2.361 / 0.425 | 1.072 / 0.161 | 1.177 | 7.211 / 1.081 / 0.567 / 0.498 | Q | 1.044 | Q | 0.500 |
| P64_s2 | A1.4 | 1138 (1138) | 22.23..40.00 | 1.044 / 2.432 / 0.425 | 1.072 / 0.161 | 1.177 | 7.211 / 1.082 / 0.567 / 0.498 | Q | 1.044 | Q | 0.500 |
| P64_s2 | A1.5 | 1138 (1138) | 22.23..40.00 | 1.044 / 2.496 / 0.425 | 1.072 / 0.161 | 1.177 | 7.211 / 1.082 / 0.567 / 0.498 | Q | 1.044 | Q | 0.500 |
| P64_s2 | A2.6 | 1040 (1040) | 23.77..40.00 | 1.056 / 2.657 / 0.395 | 1.068 / 0.104 | 1.203 | 6.701 / 0.654 / 0.527 / 0.414 | Q | 1.057 | Q | 0.500 |
| P64_s2 | A2.7 | 735 (735) | 28.53..40.00 | 0.980 / 2.636 / 0.291 | 1.025 / 0.026 | 1.229 | 4.816 / 0.207 / 0.307 / 0.115 | Q | 0.981 | Q | 0.750 |
| P64_s2 | A2.8 | 483 (483) | 32.47..40.00 | 0.905 / 2.878 / 0.242 | 1.000 / 0.046 | 1.221 | 3.143 / 0.129 / 0.370 / 0.041 | Q | 0.906 | Q | 0.750 |
| P64_s2 | A2.9 | 263 (263) | 35.91..40.00 | 0.731 / 3.002 / 0.190 | 0.957 / 0.049 | 1.159 | 1.633 / 0.158 / 0.644 / 0.076 | Q | 0.732 | Q | 0.500 |
| P64_s2 | A2.10 | 58 (58) | 39.11..40.00 | 4.143 / 2.213 / 0.017 | 0.724 / 5.19e-04 | 0.825 | 0.254 / 0.005 / 0.195 / 0.097 | exponential | 4.148 | exponential | 0.250 |
| P64_s8 | A1.0 | 259 (259) | 24.36..28.39 | -0.319 / 2.645 / 0.661 | -1.221 / 0.408 | -0.850 | 1.147 / 1.290 / 1.333 / 1.875 | constant | -0.331 | constant | 0.500 |
| P64_s8 | A1.1 | 259 (259) | 24.36..28.39 | -0.320 / 2.680 / 0.662 | -1.223 / 0.409 | -0.852 | 1.149 / 1.291 / 1.335 / 1.877 | constant | -0.332 | constant | 0.500 |
| P64_s8 | A1.2 | 259 (259) | 24.36..28.39 | -0.322 / 2.715 / 0.663 | -1.226 / 0.409 | -0.853 | 1.152 / 1.292 / 1.337 / 1.879 | constant | -0.334 | constant | 0.500 |
| P64_s8 | A1.3 | 259 (259) | 24.36..28.39 | -0.323 / 2.748 / 0.664 | -1.228 / 0.409 | -0.855 | 1.153 / 1.292 / 1.339 / 1.881 | constant | -0.335 | constant | 0.500 |
| P64_s8 | A1.4 | 259 (259) | 24.36..28.39 | -0.325 / 2.808 / 0.666 | -1.232 / 0.410 | -0.857 | 1.156 / 1.294 / 1.342 / 1.884 | constant | -0.337 | constant | 0.500 |
| P64_s8 | A1.5 | 259 (259) | 24.36..28.39 | -0.326 / 2.867 / 0.667 | -1.234 / 0.411 | -0.859 | 1.159 / 1.295 / 1.344 / 1.886 | constant | -0.338 | constant | 0.500 |
| P64_s8 | A2.6 | 228 (228) | 24.48..28.03 | -0.777 / 2.844 / 0.436 | -0.933 / 0.274 | -0.662 | 0.786 / 0.911 / 0.920 / 1.438 | constant | -0.795 | constant | 0.500 |
| P64_s4_tau4 | A1.0 | 1028 (1028) | 23.95..40.00 | 0.925 / 2.275 / 0.242 | 0.315 / 0.394 | 0.268 | 1.491 / 1.892 / 0.168 / 2.803 | K_H | 0.929 | K_H | 0.250 |
| P64_s4_tau4 | A1.1 | 1028 (1028) | 23.95..40.00 | 0.924 / 2.316 / 0.242 | 0.315 / 0.395 | 0.267 | 1.489 / 1.893 / 0.167 / 2.806 | K_H | 0.928 | K_H | 0.250 |
| P64_s4_tau4 | A1.2 | 1028 (1028) | 23.95..40.00 | 0.923 / 2.359 / 0.241 | 0.314 / 0.395 | 0.267 | 1.487 / 1.894 / 0.166 / 2.808 | K_H | 0.927 | K_H | 0.250 |
| P64_s4_tau4 | A1.3 | 1028 (1028) | 23.95..40.00 | 0.922 / 2.399 / 0.240 | 0.314 / 0.395 | 0.266 | 1.485 / 1.894 / 0.165 / 2.810 | K_H | 0.926 | K_H | 0.250 |
| P64_s4_tau4 | A1.4 | 1028 (1028) | 23.95..40.00 | 0.922 / 2.470 / 0.240 | 0.313 / 0.395 | 0.266 | 1.484 / 1.894 / 0.165 / 2.812 | K_H | 0.925 | K_H | 0.250 |
| P64_s4_tau4 | A1.5 | 1028 (1028) | 23.95..40.00 | 0.921 / 2.535 / 0.240 | 0.313 / 0.395 | 0.266 | 1.483 / 1.894 / 0.165 / 2.812 | K_H | 0.925 | K_H | 0.250 |
| P64_s4_tau4 | A2.6 | 913 (913) | 23.95..40.00 | 1.150 / 2.234 / 0.206 | 0.467 / 0.407 | 0.397 | 2.238 / 1.855 / 0.399 / 2.098 | K_H | 1.155 | K_H | 0.500 |
| P64_s4_tau4 | A2.7 | 610 (610) | 30.48..40.00 | 1.195 / 2.911 / 0.248 | 0.794 / 0.072 | 0.696 | 1.903 / 0.710 / 0.511 / 0.548 | K_H | 1.199 | K_H | 0.500 |
| P64_s4_tau4 | A2.8 | 508 (508) | 32.08..40.00 | 1.356 / 2.757 / 0.256 | 0.712 / 0.044 | 0.593 | 1.366 / 0.276 / 0.517 / 0.564 | exponential | 1.360 | exponential | 0.500 |
| P64_s4_tau4 | A2.9 | 149 (149) | 37.69..40.00 | 0.731 / 3.308 / 0.068 | 0.900 / 0.015 | 1.074 | 0.752 / 0.045 / 0.311 / 0.082 | exponential | 0.732 | exponential | 0.500 |
| W2 | A1.0 | 341 (838) | 23.95..29.27 | 1.235 / 1.386 / 0.460 | -1.052 / 0.442 | -0.733 | 1.311 / 1.716 / 0.738 / 2.347 | K_H | 1.230 | K_H | 0.000 |
| W2 | A1.1 | 341 (838) | 23.95..29.27 | 1.236 / 1.428 / 0.461 | -1.054 / 0.443 | -0.734 | 1.313 / 1.717 / 0.740 / 2.349 | K_H | 1.231 | K_H | 0.000 |
| W2 | A1.2 | 341 (838) | 23.95..29.27 | 1.237 / 1.471 / 0.461 | -1.055 / 0.443 | -0.735 | 1.315 / 1.718 / 0.741 / 2.351 | K_H | 1.233 | K_H | 0.000 |
| W2 | A1.3 | 341 (838) | 23.95..29.27 | 1.238 / 1.512 / 0.462 | -1.056 / 0.443 | -0.735 | 1.316 / 1.719 / 0.742 / 2.352 | K_H | 1.233 | K_H | 0.000 |
| W2 | A1.4 | 341 (838) | 23.95..29.27 | 1.239 / 1.583 / 0.462 | -1.057 / 0.444 | -0.736 | 1.317 / 1.720 / 0.743 / 2.353 | K_H | 1.234 | K_H | 0.000 |
| W2 | A1.5 | 341 (838) | 23.95..29.27 | 1.239 / 1.648 / 0.462 | -1.057 / 0.444 | -0.736 | 1.317 / 1.720 / 0.743 / 2.353 | K_H | 1.234 | K_H | 0.000 |
| W2 | A2.6 | 227 (715) | 24.12..32.56 | 1.173 / 1.976 / 0.112 | -0.096 / 0.199 | -0.106 | 0.312 / 0.599 / 0.163 / 1.323 | K_H | 1.165 | K_H | 0.250 |
| W2 | A2.7 | 497 (497) | 32.25..40.00 | 0.850 / 2.793 / 0.204 | 0.164 / 0.499 | 0.129 | 0.543 / 1.933 / 0.318 / 1.933 | K_H | 0.852 | K_H | 0.250 |
| W2 | A2.8 | 497 (497) | 32.25..40.00 | 0.933 / 2.750 / 0.177 | 0.187 / 0.497 | 0.151 | 0.554 / 1.944 / 0.205 / 1.887 | K_H | 0.936 | K_H | 0.250 |
| S4_128_s8 | A1.0 | 865 (2265) | 21.94..28.69 | 0.725 / 1.899 / 0.549 | -0.632 / 0.497 | -0.442 | 1.169 / 1.654 / 0.616 / 2.570 | K_H | 0.721 | K_H | 0.000 |
| S4_128_s8 | A1.1 | 865 (2265) | 21.94..28.69 | 0.724 / 1.942 / 0.548 | -0.630 / 0.496 | -0.441 | 1.166 / 1.652 / 0.613 / 2.568 | K_H | 0.720 | K_H | 0.000 |
| S4_128_s8 | A1.2 | 865 (2265) | 21.94..28.69 | 0.723 / 1.980 / 0.547 | -0.629 / 0.496 | -0.440 | 1.164 / 1.650 / 0.611 / 2.565 | K_H | 0.719 | K_H | 0.000 |
| S4_128_s8 | A1.3 | 865 (2265) | 21.94..28.69 | 0.723 / 2.017 / 0.546 | -0.628 / 0.495 | -0.439 | 1.162 / 1.650 / 0.610 / 2.564 | K_H | 0.719 | K_H | 0.000 |
| S4_128_s8 | A1.4 | 865 (2265) | 21.94..28.69 | 0.722 / 2.087 / 0.545 | -0.626 / 0.495 | -0.438 | 1.161 / 1.649 / 0.608 / 2.562 | K_H | 0.718 | K_H | 0.000 |
| S4_128_s8 | A1.5 | 865 (2265) | 21.94..28.69 | 0.722 / 2.162 / 0.545 | -0.626 / 0.495 | -0.438 | 1.160 / 1.650 / 0.608 / 2.561 | K_H | 0.718 | K_H | 0.000 |
| S4_128_s8 | A2.6 | 1643 (1643) | 23.71..40.00 | 1.285 / 2.044 / 0.249 | 0.670 / 0.191 | 0.543 | 2.651 / 0.840 / 0.699 / 1.238 | K_H | 1.289 | K_H | 0.500 |
| S4_128_s8 | A2.7 | 1400 (1400) | 29.07..40.00 | 1.289 / 2.718 / 0.263 | 0.755 / 0.094 | 0.700 | 2.293 / 0.412 / 0.709 / 0.778 | exponential | 1.292 | exponential | 0.500 |
| S4_128_s8 | A2.8 | 827 (827) | 33.29..40.00 | 1.278 / 2.958 / 0.224 | 0.868 / 0.050 | 0.820 | 1.795 / 0.571 / 0.436 / 0.270 | Q | 1.281 | Q | 0.500 |
| S4_128_s8 | A2.9 | 491 (491) | 36.17..40.00 | 1.938 / 2.120 / 0.164 | 0.882 / 0.041 | 0.970 | 1.140 / 0.235 / 0.625 / 0.170 | Q | 1.942 | Q | 0.500 |
| S4_128_s16 | A1.0 | 627 (627) | 21.97..26.86 | 0.735 / 1.870 / 0.752 | -0.770 / 0.642 | -0.617 | 1.190 / 1.958 / 1.277 / 2.059 | constant | 0.724 | constant | 0.250 |
| S4_128_s16 | A1.1 | 627 (627) | 21.97..26.86 | 0.735 / 1.911 / 0.751 | -0.769 / 0.642 | -0.617 | 1.189 / 1.958 / 1.276 / 2.058 | constant | 0.724 | constant | 0.250 |
| S4_128_s16 | A1.2 | 627 (627) | 21.97..26.86 | 0.736 / 1.949 / 0.751 | -0.768 / 0.642 | -0.616 | 1.188 / 1.958 / 1.275 / 2.058 | constant | 0.724 | constant | 0.250 |
| S4_128_s16 | A1.3 | 627 (627) | 21.97..26.86 | 0.736 / 1.984 / 0.751 | -0.768 / 0.642 | -0.616 | 1.188 / 1.958 / 1.274 / 2.057 | constant | 0.724 | constant | 0.250 |
| S4_128_s16 | A1.4 | 627 (627) | 21.97..26.86 | 0.736 / 2.050 / 0.751 | -0.767 / 0.642 | -0.615 | 1.187 / 1.958 / 1.274 / 2.057 | constant | 0.725 | constant | 0.250 |
| S4_128_s16 | A1.5 | 627 (627) | 21.97..26.86 | 0.737 / 2.120 / 0.750 | -0.767 / 0.642 | -0.615 | 1.187 / 1.958 / 1.274 / 2.057 | constant | 0.725 | constant | 0.250 |
| S4_128_s16 | A2.6 | 71 (71) | 25.67..26.22 | 3.440 / 1.295 / 0.201 | -8.000 / 0.038 | -2.297 | 0.373 / 0.007 / 0.366 / 0.416 | exponential | 3.443 | exponential | 0.750 |
| P64_s4_B2 | A1.0 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.466 / 0.439 | 0.869 / 0.113 | 0.728 | 3.854 / 1.233 / 0.482 / 0.595 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_B2 | A1.1 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.506 / 0.439 | 0.869 / 0.113 | 0.728 | 3.853 / 1.234 / 0.482 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_B2 | A1.2 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.548 / 0.439 | 0.869 / 0.113 | 0.728 | 3.853 / 1.235 / 0.482 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_B2 | A1.3 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.587 / 0.439 | 0.869 / 0.113 | 0.728 | 3.852 / 1.235 / 0.481 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_B2 | A1.4 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.656 / 0.438 | 0.869 / 0.114 | 0.728 | 3.852 / 1.236 / 0.481 / 0.597 | K_H | 1.027 | K_H | 0.500 |
| P64_s4_B2 | A1.5 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.719 / 0.438 | 0.869 / 0.114 | 0.728 | 3.852 / 1.236 / 0.481 / 0.597 | K_H | 1.027 | K_H | 0.500 |
| P64_s4_B2 | A2.6 | 1031 (1031) | 23.91..40.00 | 1.037 / 2.900 / 0.426 | 0.840 / 0.143 | 0.706 | 3.683 / 1.423 / 0.501 / 0.703 | K_H | 1.039 | K_H | 0.500 |
| P64_s4_B2 | A2.7 | 688 (688) | 29.27..40.00 | 0.972 / 3.153 / 0.325 | 0.976 / 0.028 | 0.940 | 3.124 / 0.202 / 0.413 / 0.084 | Q | 0.974 | Q | 0.500 |
| P64_s4_B2 | A2.8 | 511 (511) | 32.03..40.00 | 0.850 / 3.224 / 0.265 | 0.929 / 0.043 | 0.890 | 2.253 / 0.392 / 0.492 / 0.157 | Q | 0.852 | Q | 0.500 |
| P64_s4_B2 | A2.9 | 255 (255) | 36.03..40.00 | 1.250 / 2.150 / 0.161 | 0.987 / 0.019 | 1.079 | 1.355 / 0.051 / 0.377 / 0.015 | Q | 1.252 | Q | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.0 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.466 / 0.439 | 0.869 / 0.113 | 0.728 | 3.854 / 1.233 / 0.482 / 0.595 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.1 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.506 / 0.439 | 0.869 / 0.113 | 0.728 | 3.853 / 1.234 / 0.482 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.2 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.548 / 0.439 | 0.869 / 0.113 | 0.728 | 3.853 / 1.235 / 0.482 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.3 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.587 / 0.439 | 0.869 / 0.113 | 0.728 | 3.852 / 1.235 / 0.481 / 0.596 | K_H | 1.028 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.4 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.656 / 0.438 | 0.869 / 0.114 | 0.728 | 3.852 / 1.236 / 0.481 / 0.597 | K_H | 1.027 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A1.5 | 1053 (1053) | 23.56..40.00 | 1.026 / 2.719 / 0.438 | 0.869 / 0.114 | 0.728 | 3.852 / 1.236 / 0.481 / 0.597 | K_H | 1.027 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.6 | 1031 (1031) | 23.91..40.00 | 1.037 / 2.900 / 0.426 | 0.840 / 0.142 | 0.706 | 3.683 / 1.421 / 0.501 / 0.703 | K_H | 1.039 | K_H | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.7 | 688 (688) | 29.27..40.00 | 0.972 / 3.155 / 0.325 | 0.976 / 0.028 | 0.940 | 3.123 / 0.202 / 0.413 / 0.084 | Q | 0.974 | Q | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.8 | 512 (512) | 32.02..40.00 | 0.851 / 3.222 / 0.266 | 0.929 / 0.043 | 0.890 | 2.257 / 0.391 / 0.491 / 0.158 | Q | 0.853 | Q | 0.500 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | A2.9 | 255 (255) | 36.03..40.00 | 1.249 / 2.151 / 0.161 | 0.987 / 0.019 | 1.080 | 1.355 / 0.050 / 0.377 / 0.015 | Q | 1.251 | Q | 0.500 |
| P64_s4_heldsrc | A1.0 | 249 (249) | 23.05..26.92 | 0.241 / 2.319 / 0.829 | -1.230 / 0.551 | -0.980 | 1.403 / 1.864 / 1.591 / 2.258 | constant | 0.223 | constant | 0.250 |
| P64_s4_heldsrc | A1.1 | 249 (249) | 23.05..26.92 | 0.240 / 2.359 / 0.830 | -1.232 / 0.552 | -0.982 | 1.405 / 1.865 / 1.593 / 2.260 | constant | 0.222 | constant | 0.250 |
| P64_s4_heldsrc | A1.2 | 249 (249) | 23.05..26.92 | 0.240 / 2.400 / 0.831 | -1.234 / 0.552 | -0.983 | 1.407 / 1.866 / 1.594 / 2.262 | constant | 0.222 | constant | 0.250 |
| P64_s4_heldsrc | A1.3 | 249 (249) | 23.05..26.92 | 0.240 / 2.440 / 0.832 | -1.235 / 0.553 | -0.984 | 1.408 / 1.867 / 1.595 / 2.263 | constant | 0.222 | constant | 0.250 |
| P64_s4_heldsrc | A1.4 | 249 (249) | 23.05..26.92 | 0.241 / 2.508 / 0.833 | -1.236 / 0.553 | -0.985 | 1.410 / 1.868 / 1.597 / 2.265 | constant | 0.223 | constant | 0.250 |
| P64_s4_heldsrc | A1.5 | 249 (249) | 23.05..26.92 | 0.241 / 2.571 / 0.833 | -1.237 / 0.553 | -0.985 | 1.410 / 1.868 / 1.597 / 2.265 | constant | 0.223 | constant | 0.250 |
| P64_s4_heldsrc | A2.6 | 30 (30) | 25.66..26.11 | -4.049 / 1.827 / 0.145 | -5.045 / 0.025 | -2.214 | 0.323 / 0.007 / 0.352 / 0.385 | exponential | -4.075 | exponential | 1.000 |
| P64_s4_widened | A1.0 | 381 (479) | 23.88..29.81 | 1.296 / 1.348 / 0.417 | -0.726 / 0.429 | -0.539 | 1.111 / 0.968 / 0.568 / 2.351 | K_H | 1.299 | K_H | 0.250 |
| P64_s4_widened | A1.1 | 381 (479) | 23.88..29.81 | 1.298 / 1.386 / 0.418 | -0.727 / 0.430 | -0.540 | 1.113 / 0.968 / 0.569 / 2.352 | K_H | 1.301 | K_H | 0.250 |
| P64_s4_widened | A1.2 | 381 (479) | 23.88..29.81 | 1.299 / 1.426 / 0.418 | -0.728 / 0.430 | -0.541 | 1.114 / 0.969 / 0.571 / 2.354 | K_H | 1.302 | K_H | 0.250 |
| P64_s4_widened | A1.3 | 381 (479) | 23.88..29.81 | 1.300 / 1.465 / 0.419 | -0.729 / 0.430 | -0.542 | 1.116 / 0.970 / 0.572 / 2.355 | K_H | 1.303 | K_H | 0.250 |
| P64_s4_widened | A1.4 | 381 (479) | 23.88..29.81 | 1.301 / 1.532 / 0.419 | -0.730 / 0.431 | -0.542 | 1.117 / 0.970 / 0.573 / 2.356 | K_H | 1.304 | K_H | 0.250 |
| P64_s4_widened | A1.5 | 381 (479) | 23.88..29.81 | 1.302 / 1.595 / 0.419 | -0.730 / 0.431 | -0.543 | 1.117 / 0.971 / 0.573 / 2.357 | K_H | 1.305 | K_H | 0.250 |
| P64_s4_widened | A2.6 | 391 (391) | 23.88..39.98 | 0.703 / 2.359 / 0.250 | -0.079 / 0.228 | -0.056 | 0.472 / 0.940 / 0.215 / 4.265 | K_H | 0.697 | K_H | 0.250 |
| P64_s4_widened | A2.7 | 98 (98) | 38.47..39.98 | 2.969 / 2.387 / 0.197 | -4.421 / 0.061 | -1.277 | 0.588 / 0.156 / 0.442 / 0.713 | exponential | 2.966 | exponential | 0.250 |
| P64_s4_widened | A2.8 | 98 (98) | 38.47..39.98 | 2.253 / 3.219 / 0.148 | -3.333 / 0.048 | -0.963 | 0.445 / 0.128 / 0.299 / 0.569 | exponential | 2.250 | exponential | 0.250 |

**Table L2v3. Per member** (from Table L1v3, resolved population): clocks fitted; ranges over the clocks of the (K) slope, the (Q) slope and the (E) κ_fit/κ_H for L_T; the out-of-fit winners counted (constant / exponential / K_H / Q) for L_T and for L_D; and the members' statement read off the L_T winners.

| member | clocks fitted (dropped by the masks) | L_T (K) slope range | L_T (Q) slope range | L_T (E) κ_fit/κ_H range | L_T winners | L_D winners | L_D (K) slope range |
|---|---|---|---|---|---|---|---|
| P64_s4 | 10 (0) | 0.850 to 1.250 | 0.840 to 0.987 | 0.706 to 1.079 | 0 / 0 / 7 / 3 | 0 / 0 / 7 / 3 | 0.852 to 1.252 |
| P64_s2 | 10 (1) | 0.731 to 4.143 | 0.724 to 1.072 | 0.825 to 1.229 | 0 / 1 / 0 / 9 | 0 / 1 / 0 / 9 | 0.732 to 4.148 |
| P64_s8 | 7 (0) | -0.777 to -0.319 | -1.234 to -0.933 | -0.859 to -0.662 | 7 / 0 / 0 / 0 | 7 / 0 / 0 / 0 | -0.795 to -0.331 |
| P64_s4_tau4 | 10 (0) | 0.731 to 1.356 | 0.313 to 0.900 | 0.266 to 1.074 | 0 / 2 / 8 / 0 | 0 / 2 / 8 / 0 | 0.732 to 1.360 |
| W2 | 9 (0) | 0.850 to 1.239 | -1.057 to 0.187 | -0.736 to 0.151 | 0 / 0 / 9 / 0 | 0 / 0 / 9 / 0 | 0.852 to 1.234 |
| S4_128_s8 | 10 (0) | 0.722 to 1.938 | -0.632 to 0.882 | -0.442 to 0.970 | 0 / 1 / 7 / 2 | 0 / 1 / 7 / 2 | 0.718 to 1.942 |
| S4_128_s16 | 7 (0) | 0.735 to 3.440 | -8.000 to -0.767 | -2.297 to -0.615 | 6 / 1 / 0 / 0 | 6 / 1 / 0 / 0 | 0.724 to 3.443 |
| P64_s4_B2 | 10 (0) | 0.850 to 1.250 | 0.840 to 0.987 | 0.706 to 1.079 | 0 / 0 / 7 / 3 | 0 / 0 / 7 / 3 | 0.852 to 1.252 |
| P64_s4_ext200 (e, c only: ray mask unavailable) | 10 (0) | 0.851 to 1.249 | 0.840 to 0.987 | 0.706 to 1.080 | 0 / 0 / 7 / 3 | 0 / 0 / 7 / 3 | 0.853 to 1.251 |
| P64_s4_heldsrc | 7 (0) | -4.049 to 0.241 | -5.045 to -1.230 | -2.214 to -0.980 | 6 / 1 / 0 / 0 | 6 / 1 / 0 / 0 | -4.075 to 0.223 |
| P64_s4_widened | 9 (0) | 0.703 to 2.969 | -4.421 to -0.079 | -1.277 to -0.056 | 0 / 2 / 7 / 0 | 0 / 2 / 7 / 0 | 0.697 to 2.966 |

## 7. Birth and the two surfaces, with the certification marked (carried from REDSHIFT_received_v3.1 sec. 7)

(v3, in deliveries/route2/REDSHIFT_received_v3_scripts/) Producing script: code/rz_v3_misc_v1.py (md5 9823adfa626be5f4546911ec3a36e97d). Table file: out/v3_misc_v1/misc3_tables.md (md5 92af02813bb948933ed5cb9bbd5d95e5). It prints Tables S1c, S2c (here), Y1h (sec. 8) and HX (sec. 9). Data: v2's out/v2_birthsurf_v1/birthsurf_tables.json (md5 ede6e47998a4421328dfe8a3a2877de1, REDSHIFT_received_v2_scripts), the ray maps and the heavy pass on the box.

**Table S1c. Birth from the records, with the certification marked** (v2's Table S1, rz_v2_birthsurf_v1's json; AstraAPI_rzchk3 sec. 4.5): † = past the member's certified end (the first root record T_b, or a birth-law fit window ending past it).

| member | certified to T/r_s | T_b/r_s (first root record) | inside the certified interval? | zero of f_min: linear / quadratic T/r_s | birth-law t_b/r_s, window 0.25 / 0.5 / 1 r_s (records) | fit windows inside the interval |
|---|---|---|---|---|---|---|
| P64_s4 | 40.0000 | 12.1250 | yes | 12.1119 / 12.1125 | 12.0768 (3) / 11.9638 (5) / 11.5779 (9) | yes / yes / yes |
| P64_s2 | 40.0000 | 12.0000 | yes | 11.9091 / 11.9109 | 11.6791 (3) / 11.4473 (5) / 11.1896 (9) | yes / yes / yes |
| P64_s8 | 40.0000 | 13.0000 | yes | 12.9444 / 12.9435 | 12.8861 (3) / 12.7841 (5) / 12.3859 (9) | yes / yes / yes |
| P64_s4_tau4 | 40.0000 | 12.0000 | yes | 11.9917 / 11.9919 | 11.9453 (5) / 11.7990 (9) / 11.2963 (17) | yes / yes / yes |
| P64_s4_tau2 | 12.7500 | 11.9375 | yes | 11.9307 / 11.9308 | 11.8742 (9) / 11.7018 (17) / 11.1130 (33) | yes / yes / no † |
| P64_s4_newton | 12.5000 | 12.1250 | yes | 12.1119 / 12.1125 | 12.0768 (3) / 12.0303 (4) / 12.0303 (4) | yes / no † / no † |
| W2 | 40.0000 | 11.8750 | yes | 11.8685 / 11.8689 | 11.8364 (3) / 11.7109 (5) / 11.2442 (9) | yes / yes / yes |
| ISO_64_s4 | 13.5000 | 12.3750 | yes | 12.3635 / 12.3641 | 12.3340 (3) / 12.2383 (5) / 11.8974 (9) | yes / yes / yes |
| S4_128_s8 | 40.0000 | 11.1875 | yes | 11.1782 / 11.1786 | 11.1220 (5) / 11.0176 (9) / 10.6799 (17) | yes / yes / yes |
| S4_128_s16 | 40.0000 | 11.7500 | yes | 11.7373 / 11.7372 | 11.6974 (5) / 11.5995 (9) / 11.1831 (17) | yes / yes / yes |
| S4_128_s4 | 3.5625 | 11.0625 | no † | 11.0594 / 11.0596 | 11.0150 (5) / 10.9413 (9) / 10.7273 (17) | no † / no † / no † |
| ISO_128_s4 | 12.0000 | 11.5625 | yes | 11.5043 / 11.5045 | 11.3699 (5) / 11.2494 (9) / 11.2494 (9) | yes / no † / no † |
| ISO_128_s4_plain | 3.2500 | 11.5625 | no † | 11.5046 / 11.5049 | 11.3708 (5) / 11.2497 (9) / 10.8012 (17) | no † / no † / no † |
| M3c_s4_96 | 22.0833 | 21.1667 | yes | 21.1595 / 21.1597 | 21.1493 (4) / 21.1091 (7) / 20.9636 (12) | yes / yes / no † |

**Table S2c. Energy-zero against group-velocity turning surfaces, with the certification marked** (v2's Table S2 rows; † = the record lies past the member's certified end).

| member | T/r_s | inside? | R (a) | ak 0.1: R − r_Ω / R − r_g | ak 0.2 | ak 0.3 |
|---|---|---|---|---|---|---|
| P64_s4 | 12.125 | yes | 34.54 | 2.771 / - | - / - | - / - |
| P64_s4 | 12.250 | yes | 40.59 | 0.726 / 2.225 | 2.989 / - | 7.460 / - |
| P64_s4 | 12.375 | yes | 43.00 | 0.623 / 1.870 | 2.482 / 7.617 | 5.589 / - |
| P64_s4 | 12.500 | yes | 44.42 | 0.591 / 1.765 | 2.336 / 6.910 | 5.161 / 16.015 |
| P64_s4 | 12.625 | yes | 45.28 | 0.577 / 1.722 | 2.277 / 6.674 | 5.007 / 14.517 |
| P64_s4 | 13.125 | yes | 47.27 | 0.544 / 1.626 | 2.153 / 6.349 | 4.754 / 13.716 |
| P64_s4 | 14.125 | yes | 50.00 | 0.545 / 1.632 | 2.164 / 6.490 | 4.823 / 14.868 |
| P64_s4 | 15.125 | yes | 50.91 | 0.589 / 1.784 | 2.380 / 7.694 | 5.518 / - |
| P64_s4 | 16.125 | yes | 47.00 | 1.160 / 3.587 | 4.855 / - | - / - |
| P64_s4 | 19.125 | yes | 34.52 | 0.820 / 2.683 | 3.788 / - | - / - |
| P64_s4 | 20.125 | yes | 37.35 | 0.445 / 1.335 | 1.772 / 5.375 | 3.974 / 13.211 |
| P64_s4 | 21.125 | yes | 37.81 | 0.419 / 1.255 | 1.665 / 5.000 | 3.714 / 11.491 |
| P64_s4 | 22.125 | yes | 37.96 | 0.449 / 1.354 | 1.803 / 5.612 | 4.105 / 14.782 |
| P64_s4 | 23.125 | yes | 37.62 | 0.513 / 1.568 | 2.102 / 7.122 | 4.996 / - |
| P64_s4 | 24.125 | yes | 36.70 | 0.663 / 2.096 | 2.869 / - | - / - |
| P64_s4 | 25.125 | yes | 34.53 | 0.751 / 2.357 | 3.222 / - | - / - |
| P64_s4 | 26.125 | yes | 35.43 | 0.529 / 1.618 | 2.169 / 7.563 | 5.206 / - |
| P64_s4 | 27.125 | yes | 36.15 | 0.435 / 1.310 | 1.742 / 5.378 | 3.947 / 13.943 |
| P64_s4 | 28.125 | yes | 36.39 | 0.416 / 1.249 | 1.659 / 5.051 | 3.731 / 12.129 |
| P64_s4 | 29.125 | yes | 36.27 | 0.440 / 1.331 | 1.774 / 5.561 | 4.056 / 15.676 |
| P64_s4 | 30.125 | yes | 35.90 | 0.498 / 1.523 | 2.042 / 7.023 | 4.886 / - |
| P64_s4 | 31.125 | yes | 35.04 | 0.605 / 1.888 | 2.561 / - | 6.920 / - |
| P64_s4 | 32.125 | yes | 34.57 | 0.583 / 1.805 | 2.439 / - | 6.281 / - |
| P64_s4 | 33.125 | yes | 35.20 | 0.485 / 1.472 | 1.965 / 6.298 | 4.547 / - |
| P64_s4 | 34.125 | yes | 35.94 | 0.417 / 1.252 | 1.662 / 5.057 | 3.735 / 12.277 |
| P64_s4 | 35.125 | yes | 36.01 | 0.419 / 1.261 | 1.676 / 5.156 | 3.792 / 12.802 |
| P64_s4 | 36.125 | yes | 35.79 | 0.448 / 1.358 | 1.813 / 5.867 | 4.212 / - |
| P64_s4 | 37.125 | yes | 34.59 | 0.570 / 1.778 | 2.413 / - | 6.511 / - |
| P64_s4 | 38.125 | yes | 33.41 | 0.644 / 2.036 | 2.787 / - | - / - |
| P64_s4 | 39.125 | yes | 34.24 | 0.489 / 1.489 | 1.992 / 6.687 | 4.703 / - |
| P64_s2 | 12.000 | yes | 37.53 | 0.793 / 2.457 | 3.326 / - | 9.588 / - |
| P64_s2 | 12.125 | yes | 40.97 | 0.633 / 1.897 | 2.516 / 7.664 | 5.645 / - |
| P64_s2 | 12.250 | yes | 42.79 | 0.591 / 1.763 | 2.332 / 6.846 | 5.130 / 15.144 |
| P64_s2 | 12.375 | yes | 43.70 | 0.571 / 1.701 | 2.248 / 6.552 | 4.927 / 13.935 |
| P64_s2 | 12.500 | yes | 43.79 | 0.552 / 1.644 | 2.174 / 6.336 | 4.766 / 13.389 |
| P64_s2 | 13.000 | yes | 44.08 | 0.496 / 1.482 | 1.963 / 5.782 | 4.332 / 12.414 |
| P64_s2 | 14.000 | yes | 45.16 | 0.490 / 1.466 | 1.942 / 5.760 | 4.301 / 12.871 |
| P64_s2 | 15.000 | yes | 45.30 | 0.705 / 2.205 | 2.975 / 8.624 | 6.696 / 16.650 |
| P64_s2 | 16.000 | yes | 39.12 | 0.576 / 1.709 | 2.249 / 6.219 | 4.766 / 13.708 |
| P64_s2 | 17.000 | yes | 34.45 | 0.406 / 1.208 | 1.597 / 4.679 | 3.510 / 10.104 |
| P64_s2 | 18.000 | yes | 32.44 | 0.356 / 1.060 | 1.403 / 4.120 | 3.088 / 8.873 |
| P64_s2 | 19.000 | yes | 32.11 | 0.342 / 1.023 | 1.355 / 4.034 | 3.009 / 8.983 |
| P64_s2 | 20.000 | yes | 32.18 | 0.359 / 1.077 | 1.428 / 4.262 | 3.176 / 9.440 |
| P64_s2 | 21.000 | yes | 32.42 | 0.374 / 1.126 | 1.496 / 4.535 | 3.362 / 10.028 |
| P64_s2 | 22.000 | yes | 31.48 | 0.348 / 1.038 | 1.373 / 4.013 | 3.013 / 8.664 |
| P64_s2 | 23.000 | yes | 31.13 | 0.336 / 1.004 | 1.330 / 3.925 | 2.937 / 8.556 |
| P64_s2 | 24.000 | yes | 31.29 | 0.338 / 1.011 | 1.339 / 3.974 | 2.968 / 8.751 |
| P64_s2 | 25.000 | yes | 31.60 | 0.343 / 1.027 | 1.361 / 4.070 | 3.029 / 9.172 |
| P64_s2 | 26.000 | yes | 31.59 | 0.353 / 1.056 | 1.398 / 4.142 | 3.096 / 9.109 |
| P64_s2 | 27.000 | yes | 31.26 | 0.348 / 1.038 | 1.373 / 4.024 | 3.018 / 8.751 |
| P64_s2 | 28.000 | yes | 31.20 | 0.336 / 1.002 | 1.327 / 3.923 | 2.933 / 8.630 |
| P64_s2 | 29.000 | yes | 31.48 | 0.341 / 1.019 | 1.350 / 4.020 | 2.996 / 9.065 |
| P64_s2 | 30.000 | yes | 31.28 | 0.389 / 1.162 | 1.538 / 4.513 | 3.387 / 9.847 |
| P64_s2 | 31.000 | yes | 31.16 | 0.358 / 1.063 | 1.403 / 4.071 | 3.063 / 8.807 |
| P64_s2 | 32.000 | yes | 30.86 | 0.349 / 1.045 | 1.386 / 4.127 | 3.078 / 9.095 |
| P64_s2 | 33.000 | yes | 30.55 | 0.332 / 0.993 | 1.316 / 3.905 | 2.915 / 8.669 |
| P64_s2 | 34.000 | yes | 30.16 | 0.330 / 0.988 | 1.309 / 3.900 | 2.907 / 8.734 |
| P64_s2 | 35.000 | yes | 29.87 | 0.345 / 1.033 | 1.368 / 4.054 | 3.029 / 8.980 |
| P64_s2 | 36.000 | yes | 29.61 | 0.352 / 1.053 | 1.395 / 4.166 | 3.101 / 9.460 |
| P64_s2 | 37.000 | yes | 30.06 | 0.322 / 0.960 | 1.271 / 3.752 | 2.806 / 8.311 |
| P64_s2 | 38.000 | yes | 30.19 | 0.330 / 0.986 | 1.304 / 3.838 | 2.875 / 8.388 |
| P64_s2 | 39.000 | yes | 30.35 | 0.350 / 1.045 | 1.384 / 4.110 | 3.068 / 9.111 |
| P64_s2 | 40.000 | yes | 31.09 | 0.342 / 1.024 | 1.358 / 4.054 | 3.019 / 9.176 |
| P64_s8 | 13.000 | yes | 40.34 | 1.599 / - | - / - | - / - |
| P64_s8 | 13.125 | yes | 43.37 | 0.945 / 3.044 | 4.237 / - | - / - |
| P64_s8 | 13.250 | yes | 44.98 | 0.804 / 2.493 | 3.373 / - | 8.995 / - |
| P64_s8 | 13.375 | yes | 46.07 | 0.744 / 2.279 | 3.060 / 11.228 | 7.456 / - |
| P64_s8 | 13.500 | yes | 46.90 | 0.714 / 2.174 | 2.910 / 9.868 | 6.895 / - |
| P64_s8 | 14.000 | yes | 49.08 | 0.679 / 2.059 | 2.749 / 8.907 | 6.389 / - |
| P64_s8 | 15.000 | yes | 49.66 | 0.852 / 2.650 | 3.588 / - | 9.332 / - |
| P64_s4_tau4 | 12.000 | yes | 32.83 | - / - | - / - | - / - |
| P64_s4_tau4 | 12.062 | yes | 37.54 | 0.899 / 2.879 | 3.998 / - | - / - |
| P64_s4_tau4 | 12.125 | yes | 39.82 | 0.730 / 2.229 | 2.990 / 11.994 | 7.328 / - |
| P64_s4_tau4 | 12.188 | yes | 41.42 | 0.664 / 2.001 | 2.663 / 8.470 | 6.101 / - |
| P64_s4_tau4 | 12.250 | yes | 42.65 | 0.631 / 1.891 | 2.509 / 7.617 | 5.618 / - |
| P64_s4_tau4 | 13.000 | yes | 49.03 | 0.564 / 1.686 | 2.232 / 6.580 | 4.927 / 14.203 |
| P64_s4_tau4 | 14.000 | yes | 52.16 | 0.567 / 1.697 | 2.250 / 6.740 | 5.011 / 15.428 |
| P64_s4_tau4 | 15.000 | yes | 53.22 | 0.619 / 1.880 | 2.511 / 8.297 | 5.883 / - |
| P64_s4_tau4 | 16.000 | yes | 44.52 | 2.284 / - | - / - | - / - |
| P64_s4_tau4 | 21.000 | yes | 36.44 | 0.459 / 1.384 | 1.841 / 5.718 | 4.181 / - |
| P64_s4_tau4 | 22.000 | yes | 37.73 | 0.415 / 1.242 | 1.646 / 4.916 | 3.660 / 11.122 |
| P64_s4_tau4 | 23.000 | yes | 37.98 | 0.450 / 1.357 | 1.806 / 5.618 | 4.110 / 14.779 |
| P64_s4_tau4 | 24.000 | yes | 37.38 | 0.558 / 1.724 | 2.326 / 9.743 | 5.888 / - |
| P64_s4_tau4 | 25.000 | yes | 32.75 | - / - | - / - | - / - |
| P64_s4_tau4 | 28.000 | yes | 33.66 | 0.541 / 1.663 | 2.237 / 8.286 | 5.495 / - |
| P64_s4_tau4 | 29.000 | yes | 35.29 | 0.401 / 1.203 | 1.598 / 4.851 | 3.586 / 11.690 |
| P64_s4_tau4 | 30.000 | yes | 35.34 | 0.415 / 1.248 | 1.658 / 5.087 | 3.745 / 12.624 |
| P64_s4_tau4 | 31.000 | yes | 35.26 | 0.473 / 1.442 | 1.930 / 6.451 | 4.555 / - |
| P64_s4_tau4 | 32.000 | yes | 34.15 | 0.700 / 2.314 | 3.289 / - | - / - |
| P64_s4_tau4 | 33.000 | yes | 30.03 | - / - | - / - | - / - |
| P64_s4_tau4 | 34.000 | yes | 30.13 | 1.183 / - | - / - | - / - |
| P64_s4_tau4 | 35.000 | yes | 34.00 | 0.474 / 1.444 | 1.932 / 6.449 | 4.555 / - |
| P64_s4_tau4 | 36.000 | yes | 34.78 | 0.396 / 1.189 | 1.580 / 4.817 | 3.555 / 11.662 |
| P64_s4_tau4 | 37.000 | yes | 34.70 | 0.408 / 1.229 | 1.636 / 5.091 | 3.723 / 14.011 |
| P64_s4_tau4 | 38.000 | yes | 33.79 | 0.513 / 1.583 | 2.133 / 8.305 | 5.322 / - |
| P64_s4_tau4 | 39.000 | yes | 31.54 | 0.976 / - | - / - | - / - |
| P64_s4_tau4 | 40.000 | yes | 30.63 | 1.242 / - | - / - | - / - |
| P64_s4_tau2 | 11.938 | yes | 32.02 | - / - | - / - | - / - |
| P64_s4_tau2 | 11.969 | yes | 35.24 | 1.148 / 4.247 | - / - | - / - |
| P64_s4_tau2 | 12.000 | yes | 37.04 | 0.897 / 2.863 | 3.964 / - | - / - |
| P64_s4_tau2 | 12.031 | yes | 38.38 | 0.790 / 2.442 | 3.301 / - | 9.091 / - |
| P64_s4_tau2 | 12.062 | yes | 39.46 | 0.730 / 2.225 | 2.980 / 11.242 | 7.227 / - |
| P64_s4_tau2 | 12.938 | no † | 49.96 | 0.575 / 1.717 | 2.273 / 6.700 | 5.018 / 14.460 |
| P64_s4_tau2 | 13.938 | no † | 53.32 | 0.578 / 1.732 | 2.296 / 6.878 | 5.114 / 15.748 |
| P64_s4_tau2 | 14.938 | no † | 54.46 | 0.636 / 1.935 | 2.588 / 8.703 | 6.113 / - |
| P64_s4_newton | 12.125 | yes | 34.54 | 2.771 / - | - / - | - / - |
| P64_s4_newton | 12.250 | yes | 40.59 | 0.726 / 2.225 | 2.989 / - | 7.460 / - |
| P64_s4_newton | 12.375 | yes | 43.00 | 0.623 / 1.870 | 2.482 / 7.617 | 5.589 / - |
| P64_s4_newton | 12.500 | yes | 44.42 | 0.591 / 1.765 | 2.336 / 6.910 | 5.161 / 16.015 |
| W2 | 11.875 | yes | 31.30 | - / - | - / - | - / - |
| W2 | 12.000 | yes | 39.11 | 0.727 / 2.213 | 2.961 / 10.696 | 7.104 / - |
| W2 | 12.125 | yes | 42.38 | 0.638 / 1.907 | 2.527 / 7.588 | 5.626 / - |
| W2 | 12.250 | yes | 44.66 | 0.615 / 1.834 | 2.426 / 7.116 | 5.336 / 15.661 |
| W2 | 12.375 | yes | 46.42 | 0.607 / 1.810 | 2.394 / 7.003 | 5.260 / 14.974 |
| W2 | 12.875 | yes | 50.85 | 0.586 / 1.749 | 2.316 / 6.826 | 5.112 / 14.731 |
| W2 | 13.875 | yes | 54.44 | 0.590 / 1.767 | 2.342 / 7.013 | 5.216 / 16.051 |
| W2 | 14.875 | yes | 55.64 | 0.653 / 1.991 | 2.666 / 9.150 | 6.354 / - |
| W2 | 22.875 | yes | 36.51 | 0.425 / 1.273 | 1.689 / 5.081 | 3.770 / 11.984 |
| W2 | 23.875 | yes | 37.84 | 0.421 / 1.262 | 1.674 / 5.043 | 3.741 / 11.681 |
| W2 | 24.875 | yes | 37.69 | 0.500 / 1.522 | 2.037 / 6.759 | 4.794 / - |
| W2 | 25.875 | yes | 34.00 | - / - | - / - | - / - |
| W2 | 30.875 | yes | 33.93 | 0.437 / 1.324 | 1.766 / 5.625 | 4.068 / - |
| W2 | 31.875 | yes | 34.68 | 0.394 / 1.183 | 1.570 / 4.736 | 3.511 / 11.112 |
| W2 | 32.875 | yes | 34.29 | 0.453 / 1.379 | 1.844 / 6.055 | 4.317 / - |
| W2 | 33.875 | yes | 31.67 | 1.004 / - | - / - | - / - |
| W2 | 37.875 | yes | 29.60 | 0.783 / 2.660 | 3.926 / - | - / - |
| W2 | 38.875 | yes | 33.62 | 0.407 / 1.226 | 1.630 / 5.023 | 3.686 / 13.760 |
| W2 | 39.875 | yes | 34.03 | 0.408 / 1.230 | 1.635 / 5.053 | 3.708 / 13.147 |
| ISO_64_s4 | 12.375 | yes | 36.25 | - / - | - / - | - / - |
| ISO_64_s4 | 12.500 | yes | 42.39 | 0.758 / 2.331 | 3.138 / - | 7.973 / - |
| ISO_64_s4 | 12.625 | yes | 44.68 | 0.640 / 1.926 | 2.559 / 7.936 | 5.798 / - |
| ISO_64_s4 | 12.750 | yes | 45.94 | 0.600 / 1.794 | 2.377 / 7.085 | 5.275 / 16.744 |
| ISO_64_s4 | 12.875 | yes | 46.63 | 0.581 / 1.734 | 2.295 / 6.762 | 5.062 / 14.891 |
| ISO_64_s4 | 13.375 | yes | 47.82 | 0.544 / 1.625 | 2.151 / 6.344 | 4.750 / 13.702 |
| ISO_64_s4 | 14.375 | no † | 50.12 | 0.541 / 1.622 | 2.150 / 6.439 | 4.789 / 14.691 |
| ISO_64_s4 | 15.375 | no † | 50.81 | 0.582 / 1.762 | 2.349 / 7.496 | 5.412 / - |
| ISO_64_s4 | 16.375 | no † | 48.70 | 1.056 / 3.493 | 4.904 / - | - / - |
| ISO_64_s4 | 20.375 | no † | 38.82 | 0.485 / 1.463 | 1.947 / 6.082 | 4.438 / - |
| ISO_64_s4 | 21.375 | no † | 39.21 | 0.426 / 1.275 | 1.690 / 5.031 | 3.751 / 11.272 |
| ISO_64_s4 | 22.375 | no † | 39.41 | 0.440 / 1.322 | 1.756 / 5.345 | 3.948 / 12.812 |
| ISO_64_s4 | 23.375 | no † | 38.97 | 0.497 / 1.511 | 2.020 / 6.592 | 4.715 / - |
| ISO_64_s4 | 24.375 | no † | 38.20 | 0.612 / 1.918 | 2.611 / - | 7.653 / - |
| ISO_64_s4 | 25.375 | no † | 35.55 | 1.040 / 3.757 | - / - | - / - |
| ISO_64_s4 | 26.375 | no † | 35.56 | 0.667 / 2.101 | 2.873 / - | - / - |
| ISO_64_s4 | 27.375 | no † | 37.10 | 0.459 / 1.387 | 1.846 / 5.815 | 4.226 / - |
| ISO_64_s4 | 28.375 | no † | 37.60 | 0.417 / 1.253 | 1.662 / 5.027 | 3.723 / 11.851 |
| ISO_64_s4 | 29.375 | no † | 37.70 | 0.426 / 1.281 | 1.702 / 5.202 | 3.837 / 12.560 |
| ISO_64_s4 | 30.375 | no † | 37.41 | 0.465 / 1.411 | 1.884 / 6.078 | 4.372 / - |
| ISO_64_s4 | 31.375 | no † | 36.81 | 0.543 / 1.676 | 2.259 / 9.008 | 5.666 / - |
| ISO_64_s4 | 32.375 | no † | 35.92 | 0.640 / 2.017 | 2.754 / - | 8.924 / - |
| ISO_64_s4 | 33.375 | no † | 35.78 | 0.525 / 1.606 | 2.153 / 7.406 | 5.151 / - |
| ISO_64_s4 | 34.375 | no † | 36.34 | 0.446 / 1.339 | 1.779 / 5.420 | 4.000 / 13.377 |
| ISO_64_s4 | 35.375 | no † | 37.05 | 0.413 / 1.241 | 1.648 / 5.010 | 3.703 / 12.002 |
| ISO_64_s4 | 36.375 | no † | 37.05 | 0.428 / 1.290 | 1.716 / 5.336 | 3.904 / 14.254 |
| ISO_64_s4 | 37.375 | no † | 36.85 | 0.467 / 1.417 | 1.892 / 6.124 | 4.397 / - |
| ISO_64_s4 | 38.375 | no † | 36.43 | 0.527 / 1.626 | 2.191 / 8.451 | 5.469 / - |
| ISO_64_s4 | 39.375 | no † | 35.08 | 0.544 / 1.662 | 2.226 / 7.570 | 5.293 / - |
| S4_128_s8 | 11.188 | yes | 63.84 | 5.521 / - | - / - | - / - |
| S4_128_s8 | 11.250 | yes | 76.02 | 1.907 / 5.871 | 7.920 / - | 22.594 / - |
| S4_128_s8 | 11.312 | yes | 82.03 | 1.603 / 4.832 | 6.428 / 20.235 | 14.653 / - |
| S4_128_s8 | 11.375 | yes | 86.07 | 1.465 / 4.393 | 5.827 / 17.530 | 13.007 / - |
| S4_128_s8 | 11.438 | yes | 88.98 | 1.382 / 4.136 | 5.479 / 16.279 | 12.150 / 36.720 |
| S4_128_s8 | 12.188 | yes | 100.73 | 1.133 / 3.385 | 4.482 / 13.224 | 9.900 / 28.609 |
| S4_128_s8 | 13.188 | yes | 105.42 | 1.116 / 3.343 | 4.431 / 13.275 | 9.867 / 30.785 |
| S4_128_s8 | 14.188 | yes | 106.22 | 1.263 / 3.880 | 5.220 / 19.838 | 12.944 / - |
| S4_128_s8 | 20.188 | yes | 67.70 | 0.796 / 2.383 | 3.159 / 9.415 | 7.016 / 21.257 |
| S4_128_s8 | 21.188 | yes | 70.19 | 0.810 / 2.438 | 3.241 / 9.967 | 7.329 / 24.434 |
| S4_128_s8 | 22.188 | yes | 70.48 | 0.947 / 2.899 | 3.889 / 13.428 | 9.327 / - |
| S4_128_s8 | 23.188 | yes | 67.20 | 1.864 / 8.024 | - / - | - / - |
| S4_128_s8 | 26.188 | yes | 59.51 | 0.776 / 2.339 | 3.111 / 9.708 | 7.083 / - |
| S4_128_s8 | 27.188 | yes | 61.22 | 0.737 / 2.216 | 2.943 / 8.999 | 6.631 / 22.442 |
| S4_128_s8 | 28.188 | yes | 61.53 | 0.853 / 2.606 | 3.491 / 11.791 | 8.290 / - |
| S4_128_s8 | 29.188 | yes | 60.53 | 1.217 / 3.963 | 5.547 / - | - / - |
| S4_128_s8 | 30.188 | yes | 57.95 | 1.626 / 5.993 | - / - | - / - |
| S4_128_s8 | 31.188 | yes | 55.91 | 1.078 / 3.385 | 4.615 / - | 13.863 / - |
| S4_128_s8 | 32.188 | yes | 56.72 | 0.725 / 2.188 | 2.914 / 9.134 | 6.654 / - |
| S4_128_s8 | 33.188 | yes | 57.07 | 0.756 / 2.294 | 3.062 / 9.867 | 7.102 / - |
| S4_128_s8 | 34.188 | yes | 56.70 | 0.897 / 2.781 | 3.759 / 16.029 | 9.609 / - |
| S4_128_s8 | 35.188 | yes | 55.83 | 1.098 / 3.526 | 4.885 / - | - / - |
| S4_128_s8 | 36.188 | yes | 54.62 | 1.159 / 3.702 | 5.105 / - | - / - |
| S4_128_s8 | 37.188 | yes | 55.37 | 0.822 / 2.508 | 3.359 / 11.429 | 7.984 / - |
| S4_128_s8 | 38.188 | yes | 56.78 | 0.761 / 2.311 | 3.085 / 9.966 | 7.165 / - |
| S4_128_s8 | 39.188 | yes | 56.82 | 0.793 / 2.422 | 3.244 / 10.931 | 7.694 / - |
| S4_128_s16 | 11.750 | yes | 74.30 | - / - | - / - | - / - |
| S4_128_s16 | 11.812 | yes | 82.36 | 2.307 / 7.908 | 11.998 / - | - / - |
| S4_128_s16 | 11.875 | yes | 86.54 | 1.838 / 5.785 | 7.915 / - | - / - |
| S4_128_s16 | 11.938 | yes | 89.44 | 1.637 / 5.042 | 6.795 / - | 17.387 / - |
| S4_128_s16 | 12.000 | yes | 91.65 | 1.524 / 4.650 | 6.229 / 22.107 | 14.955 / - |
| S4_128_s16 | 12.750 | yes | 102.19 | 1.292 / 3.891 | 5.174 / 15.975 | 11.721 / 44.349 |
| S4_128_s16 | 13.750 | yes | 103.44 | 1.604 / 4.936 | 6.642 / 23.869 | 16.233 / - |
| S4_128_s4 | 11.062 | no † | 57.11 | - / - | - / - | - / - |
| S4_128_s4 | 11.125 | no † | 72.89 | 1.924 / 5.858 | 7.850 / - | 19.620 / - |
| S4_128_s4 | 11.188 | no † | 79.80 | 1.630 / 4.891 | 6.487 / 19.573 | 14.484 / - |
| S4_128_s4 | 11.250 | no † | 84.25 | 1.479 / 4.430 | 5.871 / 17.396 | 13.012 / 38.487 |
| S4_128_s4 | 11.312 | no † | 87.14 | 1.383 / 4.136 | 5.477 / 16.176 | 12.111 / 34.666 |
| S4_128_s4 | 12.062 | no † | 93.18 | 1.046 / 3.123 | 4.133 / 12.155 | 9.112 / 26.082 |
| S4_128_s4 | 13.062 | no † | 97.55 | 1.013 / 3.031 | 4.015 / 11.949 | 8.903 / 28.913 |
| S4_128_s4 | 14.062 | no † | 85.77 | 1.372 / 4.153 | 5.555 / 16.704 | 12.601 / - |
| S4_128_s4 | 15.062 | no † | 67.68 | 1.097 / 3.314 | 4.426 / - | 10.850 / - |
| S4_128_s4 | 16.062 | no † | 59.17 | 0.952 / 2.909 | 3.900 / 13.408 | 9.318 / - |
| S4_128_s4 | 17.062 | no † | 56.74 | 0.656 / 1.959 | 2.593 / 7.643 | 5.723 / 16.581 |
| S4_128_s4 | 18.062 | no † | 57.35 | 0.613 / 1.833 | 2.428 / 7.200 | 5.377 / 15.914 |
| S4_128_s4 | 19.062 | no † | 58.66 | 0.637 / 1.913 | 2.540 / 7.707 | 5.699 / 18.318 |
| ISO_128_s4 | 11.562 | yes | 93.06 | 2.032 / 6.502 | 9.010 / - | - / - |
| ISO_128_s4 | 11.625 | yes | 97.56 | 1.580 / 4.818 | 6.451 / 22.354 | 15.376 / - |
| ISO_128_s4 | 11.688 | yes | 100.11 | 1.411 / 4.256 | 5.663 / 17.627 | 12.884 / - |
| ISO_128_s4 | 11.750 | yes | 101.64 | 1.322 / 3.969 | 5.269 / 15.946 | 11.808 / 37.962 |
| ISO_128_s4 | 11.812 | yes | 102.42 | 1.265 / 3.790 | 5.026 / 15.029 | 11.188 / 33.769 |
| ISO_128_s4_plain | 11.562 | no † | 93.03 | 2.036 / 6.519 | 9.039 / - | - / - |
| ISO_128_s4_plain | 11.625 | no † | 97.55 | 1.581 / 4.821 | 6.456 / 22.404 | 15.396 / - |
| ISO_128_s4_plain | 11.688 | no † | 100.10 | 1.412 / 4.257 | 5.665 / 17.638 | 12.890 / - |
| ISO_128_s4_plain | 11.750 | no † | 101.63 | 1.322 / 3.970 | 5.270 / 15.950 | 11.811 / 37.997 |
| ISO_128_s4_plain | 11.812 | no † | 102.42 | 1.265 / 3.791 | 5.026 / 15.032 | 11.189 / 33.780 |
| ISO_128_s4_plain | 12.562 | no † | 99.64 | 1.043 / 3.113 | 4.119 / 12.119 | 9.081 / 26.248 |
| ISO_128_s4_plain | 13.562 | no † | 98.42 | 1.010 / 3.021 | 4.003 / 12.130 | 8.933 / - |
| M3c_s4_96 | 21.167 | yes | 61.39 | - / - | - / - | - / - |
| M3c_s4_96 | 21.250 | yes | 68.82 | 1.746 / 5.940 | 8.776 / - | - / - |
| M3c_s4_96 | 21.333 | yes | 71.87 | 1.318 / 4.119 | 5.599 / - | 16.028 / - |
| M3c_s4_96 | 21.417 | yes | 73.82 | 1.141 / 3.492 | 4.684 / 16.566 | 11.290 / - |
| M3c_s4_96 | 21.500 | yes | 75.19 | 1.043 / 3.162 | 4.219 / 13.581 | 9.766 / - |

## 8. R7, the ray map, capture, the slope fill (M1, Y1, Y1h, Y2, C1 carried from REDSHIFT_received_v3.1 sec. 8; F1v3 and F2v3 recomputed)

(v3, in deliveries/route2/REDSHIFT_received_v3_scripts/) Table Y1h: code/rz_v3_misc_v1.py (md5 9823adfa626be5f4546911ec3a36e97d), table file out/v3_misc_v1/misc3_tables.md (md5 92af02813bb948933ed5cb9bbd5d95e5).

(v2, in deliveries/route2/REDSHIFT_received_v2_scripts/) Producing script: code/rz_v2_misc_v1.py (md5 b3daab2e7c2264596ce413c0e8397454). Table file: out/v2_misc_v1/misc_tables.md (md5 796fefb7f50ae611891e4ee2470a8513). The hold ray map: code/rz_raymap_v2.py (md5 4a8a25fc700579c367c31fd55ebf880e).

**Table M1. reviewer A R7 on the records** (rz_v2_misc_v1.py, from the stored Z_k map of rz_rays_v4): the radial chain's bound ν = max q/ε at the run's m a, its inverse, the largest max(−v/N) over the certified root records, and the certified root records on which the heavy species' stationary Killing ratio N + v(√(1 + m_c²) − m_c) is negative somewhere (κ_ch = 0), of the certified root records.

| member | m a | ν (chain) | 1/ν | largest max(−v/N), certified root records | records with a negative heavy minimum / certified root records |
|---|---|---|---|---|---|
| P64_s4 | 0.20 | 0.8198 | 1.2198 | 1.531 | 38 / 207 |
| P64_s2 | 0.20 | 0.8198 | 1.2198 | 2.277 | 215 / 225 |
| P64_s8 | 0.20 | 0.8198 | 1.2198 | 1.114 | 0 / 22 |
| P64_s4_tau4 | 0.20 | 0.8198 | 1.2198 | 1.543 | 77 / 345 |
| P64_s4_tau2 | 0.20 | 0.8198 | 1.2198 | 1.512 | 15 / 27 |
| P64_s4_newton | 0.20 | 0.8198 | 1.2198 | 1.197 | 0 / 4 |
| W2 | 0.20 | 0.8198 | 1.2198 | 1.546 | 34 / 112 |
| ISO_64_s4 | 0.20 | 0.8198 | 1.2198 | 1.548 | 6 / 10 |
| S4_128_s8 | 0.20 | 0.8198 | 1.2198 | 1.561 | 57 / 360 |
| S4_128_s16 | 0.20 | 0.8198 | 1.2198 | 1.163 | 0 / 46 |
| S4_128_s4 | 0.20 | 0.8198 | 1.2198 | - | 0 / 0 |
| ISO_128_s4 | 0.20 | 0.8198 | 1.2198 | 1.656 | 4 / 8 |
| ISO_128_s4_plain | 0.20 | 0.8198 | 1.2198 | - | 0 / 0 |
| M3c_s4_96 | 0.10 | 0.9050 | 1.1050 | 1.560 | 8 / 12 |
| P64_s4_B2 | 0.20 | 0.8198 | 1.2198 | 1.531 | 38 / 207 |
| P64_s4_ext200 | 0.20 | 0.8198 | 1.2198 | 1.531 | 38 / 207 |

**Table Y1. The ray map with every reading's certification status** (rz_raymap_v1 outputs, main variant, entry and exit at 4 r_s): arrivals entered; late κ_u/κ_H (smoothed over 0.25 r_s, κ_H at the arrival's charge): median over the last 5 r_s inside the certified window, and over the last 5 r_s of the stored window when that lies past it (†); the three longest segments passing reviewer B 4.1's tolerances (u/r_s range, † where any part lies past the certified end).

| member | certified to T/r_s | entered / arrivals | late κ_u/κ_H, certified (u/r_s range) | late κ_u/κ_H, stored window past the certified end † | longest tolerance-passing segments (u/r_s) |
|---|---|---|---|---|---|
| P64_s4 | 40.0000 | 1869 / 2305 | 0.998 (35.00..40.00) | - | 0.656 r_s (22.81..23.47) |
| P64_s2 | 40.0000 | 1867 / 2305 | 1.240 (35.00..40.00) | - | 0.297 r_s (22.52..22.81); 0.000 r_s (11.30..11.30) |
| P64_s8 | 40.0000 | 1871 / 2305 | 0.391 (35.00..40.00) | - | none |
| P64_s4_tau4 | 40.0000 | 1869 / 2305 | 0.407 (35.00..40.00) | - | 0.562 r_s (22.69..23.25) |
| P64_s4_tau2 | 12.7500 | 684 / 1121 | 0.068 (10.83..12.75) | 0.420 † | none |
| W2 | 40.0000 | 1868 / 2305 | -0.478 (35.00..40.00) | - | 2.094 r_s (31.09..33.19); 0.438 r_s (22.59..23.03); 0.062 r_s (30.94..31.00) |
| ISO_64_s4 | 13.5000 | 1872 / 2305 | 0.058 (10.77..13.50) | 0.939 † | 0.672 r_s (23.06..23.73) † |
| S4_128_s8 | 40.0000 | 3732 / 4609 | 0.822 (35.00..40.00) | - | 0.367 r_s (21.76..22.12) |
| S4_128_s16 | 40.0000 | 3737 / 4609 | -0.282 (35.00..40.00) | - | none |
| S4_128_s4 | 3.5625 | 1051 / 1929 | - (-..-) | 0.244 † | none |
| ISO_128_s4 | 12.0000 | 167 / 1033 | 0.056 (10.77..12.00) | -0.193 † | none |
| P64_s4_B2 | 40.0000 | 5709 / 6145 | 0.998 (35.00..40.00) | 1.117 † | 0.656 r_s (22.81..23.47) |
| P64_s4_ext200 | 40.0000 | 12109 / 12545 | 0.998 (35.00..40.00) | 1.130 † | 0.656 r_s (22.81..23.47) |

**Table Y1h. The late peeling exponent under the production hold** (P64_s4, ray map entry and exit at 4 r_s; FableAPI_rzchk3 item 8): κ_u/κ_H smoothed over 0.25 r_s (κ_H at the arrival's charge), median over the last 5 r_s inside the solver-certified record interval, on the main map and on rz_raymap_v2's hold (record k's applied fields held over [T_k, T_k + τ]).

| map | arrivals entered | late κ_u/κ_H median (u/r_s range) | q25 / q75 there |
|---|---|---|---|
| main (rz_raymap_v1, back-averaged fields, linear in time) | 1869 | 0.998 (35.00..40.00) | 0.912 / 1.062 |
| the production hold (rz_raymap_v2) | 1869 | 0.998 (35.00..40.00) | 0.904 / 1.062 |

**Table Y2. The production hold on the ray map** (P64_s4, entry and exit at 4 r_s): rz_raymap_v1's "app_hold_cubic" map was the applied fields linear in time (its tracer ignored the variant's time mode); rz_raymap_v2 honours the hold. Each against the main map (rz_raymap_v1 and v2 give the main map identically): over the certified window, max |Δp| (a), max |Δ ln p'|, max |Δ κ_u/κ_H| (smoothed as Y1).

| map | arrivals matched | max abs Δp (a) | max abs Δ ln p' | max abs Δ κ_u/κ_H |
|---|---|---|---|---|
| v1 "app_hold_cubic" (applied fields, linear in time) | 1869 | 9.69e-06 | 1.82e-06 | 5.09e-06 |
| v2 app_hold_cubic (the hold) | 1869 | 3.993 | 0.041 | 0.185 |
| v2 bar_lin_cubic (the main map, rerun) | 1869 | 0.000 | 0.000 | 0.000 |

**Table C1. Capture as finite-time non-return on the held fields** (rows parsed from REDSHIFT_received_v1's out/capture_v3/tables_capture14.md): per ω/m on the solved slice: the band means of the channel non-return A at 50 r_s,h, smallest and largest over the six κ/κ_c bands; the strict sum σ/(πb_c²) at 150 and 300 r_s,h; the adjusted sum (outgoing weight beyond the counted region counted as returned, outer absorber, version 11) at 150 and 300; its relative change from 150 to 300.

| ω/m | band means of A at 50: smallest / largest | strict at 150 / 300 | adjusted at 150 / 300 | adjusted 150 → 300, relative change |
|---|---|---|---|---|
| 1.1 | 0.0077 / 0.0132 | - / - | - / - | - |
| 1.25 | 0.0033 / 0.0084 | 0.1118 / 0.0773 | 0.0825 / 0.0766 | -0.072 |
| 1.5 | 7.201e-04 / 0.0077 | 0.0878 / 0.0853 | 0.0842 / 0.0850 | 0.010 |
| 2 | 1.557e-04 / 0.0026 | 0.0163 / 0.0163 | 0.0159 / 0.0163 | 0.025 |

(v4, in deliveries/route2/REDSHIFT_received_v4_scripts/) Producing script: code/rz_v3_fill_v3.py (md5 481c09f23dbfe8f0ff0b844ac6e2e132). Table file: out/v4_fill_v1/fill_tables.md (md5 0bbd74aa2846ba69a061f642e319d09b). Unchanged from v3 (md5 equal to v3's: yes); run on the exact-release passes by code/v4tables_run1.sh; its captions keep their v3 names (F1v3, F2v3). Its table file equals v3's out/v3_fill_v3/fill_tables.md byte for byte: yes (F1 reads the records; F2 reads the A1 clocks, which the exact release leaves unchanged).

**Table F1v3. The slope fill per member, with the certification marked** (rz_v3_fill_v3.py): over the root records inside the solver-certified record interval, the median κ_R/κ_H (back-averaged fields) and the median of the engine's dense 4gE κ_T; the first matter-root episode: its span, its records and how many lie inside the interval (an episode with none is uncertified as a whole), its duration and ∫κ_R dT over it, and over its certified part.

| member | certified to T/r_s | root records inside | median κ_R/κ_H | median 4gE κ_T (dense) | first episode T/r_s | its records (inside the interval) | duration / r_s | ∫κ_R dT over it (certified part) |
|---|---|---|---|---|---|---|---|---|
| P64_s4 | 40.0000 | 207 | 0.956 | 1.064 | 12.125–16.625 | 37 (37) | 4.500 | 2.173 (2.173) |
| P64_s2 | 40.0000 | 225 | 1.208 | 1.229 | 12.000–40.000 | 225 (225) | 28.000 | 20.939 (20.939) |
| P64_s8 | 40.0000 | 22 | 0.744 | 0.977 | 13.000–15.625 | 22 (22) | 2.625 | 1.007 (1.007) |
| P64_s4_tau4 | 40.0000 | 345 | 0.961 | 1.068 | 12.000–16.062 | 66 (66) | 4.062 | 1.953 (1.953) |
| P64_s4_tau2 | 12.7500 | 27 | 0.934 | 0.956 | 11.938–15.844 | 126 (27) | 3.906 | 1.871 (0.389) |
| W2 | 40.0000 | 112 | 1.000 | 1.181 | 11.875–15.625 | 31 (31) | 3.750 | 1.793 (1.793) |
| S4_128_s8 | 40.0000 | 360 | 0.986 | 1.244 | 11.188–14.750 | 58 (58) | 3.562 | 1.793 (1.793) |
| S4_128_s16 | 40.0000 | 46 | 0.783 | 0.962 | 11.750–14.562 | 46 (46) | 2.812 | 1.102 (1.102) |
| S4_128_s4 | 3.5625 | 0 | - | - | 11.062–19.062 | 129 (0: uncertified as a whole) | 8.000 | 4.521 (-) |
| ISO_128_s4 | 12.0000 | 8 | 0.880 | 0.887 | 11.562–12.062 | 9 (8) | 0.500 | 0.242 (0.207) |
| ISO_64_s4 | 13.5000 | 10 | 0.960 | 0.979 | 12.375–16.875 | 37 (10) | 4.500 | 2.217 (0.572) |
| M3c_s4_96 | 22.0833 | 12 | 0.904 | 0.922 | 21.167–22.083 | 12 (12) | 0.917 | 0.404 (0.404) |
| P64_s4_newton | 12.5000 | 4 | 0.833 | 0.903 | 12.125–12.500 | 4 (4) | 0.375 | 0.158 (0.158) |
| ISO_128_s4_plain | 3.2500 | 0 | - | - | 11.562–14.250 | 44 (0: uncertified as a whole) | 2.688 | 1.453 (-) |

**Table F2v3. At matched arrival clock** (reviewer A rzchk 7.1; reviewer B 4.7, rzchk3 sec. 4.1): L_T at 4 r_s of each comoving clock A1.k (r_0 = 0.5, 1, 1.5, 2, 3, 4 r_s) at T_D/r_s = 20, 30, 40, linear in the arrival grid; ‡ = not resolved (emission r_e < 1 a, the clock inside the first site by the emission, or the ray's smallest |r| < 1 a), † = past the member's certified end. v2's column was A1.0.

| set | member | clock | T_D 20 | T_D 30 | T_D 40 |
|---|---|---|---|---|---|
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s2 | A1.0 (r_0 0.50 r_s) | 1.462 ‡ | 7.292 ‡ | 15.385 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s2 | A1.1 (r_0 1.00 r_s) | 1.536 | 7.345 | 15.438 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s2 | A1.2 (r_0 1.50 r_s) | 1.586 | 7.403 | 15.496 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s2 | A1.3 (r_0 2.00 r_s) | 1.628 | 7.448 | 15.541 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s2 | A1.4 (r_0 3.00 r_s) | 1.698 | 7.519 | 15.613 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s2 | A1.5 (r_0 4.00 r_s) | 1.761 | 7.583 | 15.676 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s4 | A1.0 (r_0 0.50 r_s) | 1.452 | 4.827 | 10.206 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s4 | A1.1 (r_0 1.00 r_s) | 1.495 | 4.866 | 10.245 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s4 | A1.2 (r_0 1.50 r_s) | 1.540 | 4.907 | 10.286 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s4 | A1.3 (r_0 2.00 r_s) | 1.581 | 4.946 | 10.325 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s4 | A1.4 (r_0 3.00 r_s) | 1.651 | 5.015 | 10.393 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s4 | A1.5 (r_0 4.00 r_s) | 1.713 | 5.078 | 10.456 |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s8 | A1.0 (r_0 0.50 r_s) | 1.296 | 0.579 ‡ | 1.956 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s8 | A1.1 (r_0 1.00 r_s) | 1.337 | 0.579 ‡ | 1.956 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s8 | A1.2 (r_0 1.50 r_s) | 1.378 | 0.579 ‡ | 1.956 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s8 | A1.3 (r_0 2.00 r_s) | 1.416 | 0.579 ‡ | 1.956 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s8 | A1.4 (r_0 3.00 r_s) | 1.483 | 0.579 ‡ | 1.956 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | P64_s8 | A1.5 (r_0 4.00 r_s) | 1.546 | 0.580 ‡ | 1.956 ‡ |
| widths at r_s/a 64 (σ/a 2, 4, 8) | envelope, resolved and certified | A1.0 | 1.296..1.452 (2) | 4.827..4.827 (1) | 10.206..10.206 (1) |
| widths at r_s/a 64 (σ/a 2, 4, 8) | envelope, resolved and certified | A1.1 | 1.337..1.536 (3) | 4.866..7.345 (2) | 10.245..15.438 (2) |
| widths at r_s/a 64 (σ/a 2, 4, 8) | envelope, resolved and certified | A1.2 | 1.378..1.586 (3) | 4.907..7.403 (2) | 10.286..15.496 (2) |
| widths at r_s/a 64 (σ/a 2, 4, 8) | envelope, resolved and certified | A1.3 | 1.416..1.628 (3) | 4.946..7.448 (2) | 10.325..15.541 (2) |
| widths at r_s/a 64 (σ/a 2, 4, 8) | envelope, resolved and certified | A1.4 | 1.483..1.698 (3) | 5.015..7.519 (2) | 10.393..15.613 (2) |
| widths at r_s/a 64 (σ/a 2, 4, 8) | envelope, resolved and certified | A1.5 | 1.546..1.761 (3) | 5.078..7.583 (2) | 10.456..15.676 (2) |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4 | A1.0 (r_0 0.50 r_s) | 1.452 | 4.827 | 10.206 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4 | A1.1 (r_0 1.00 r_s) | 1.495 | 4.866 | 10.245 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4 | A1.2 (r_0 1.50 r_s) | 1.540 | 4.907 | 10.286 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4 | A1.3 (r_0 2.00 r_s) | 1.581 | 4.946 | 10.325 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4 | A1.4 (r_0 3.00 r_s) | 1.651 | 5.015 | 10.393 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4 | A1.5 (r_0 4.00 r_s) | 1.713 | 5.078 | 10.456 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau4 | A1.0 (r_0 0.50 r_s) | 1.496 | 2.783 | 5.860 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau4 | A1.1 (r_0 1.00 r_s) | 1.541 | 2.823 | 5.897 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau4 | A1.2 (r_0 1.50 r_s) | 1.586 | 2.865 | 5.937 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau4 | A1.3 (r_0 2.00 r_s) | 1.628 | 2.905 | 5.976 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau4 | A1.4 (r_0 3.00 r_s) | 1.699 | 2.975 | 6.044 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau4 | A1.5 (r_0 4.00 r_s) | 1.762 | 3.039 | 6.107 |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau2 | A1.0 (r_0 0.50 r_s) | 1.518 † | - ‡ † | - ‡ † |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau2 | A1.1 (r_0 1.00 r_s) | 1.563 † | - ‡ † | - ‡ † |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau2 | A1.2 (r_0 1.50 r_s) | 1.609 † | - ‡ † | - ‡ † |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau2 | A1.3 (r_0 2.00 r_s) | 1.650 † | - ‡ † | - ‡ † |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau2 | A1.4 (r_0 3.00 r_s) | 1.721 † | - ‡ † | - ‡ † |
| steps at σ/a 4 (τ 8, 4, 2) | P64_s4_tau2 | A1.5 (r_0 4.00 r_s) | 1.785 † | - ‡ † | - ‡ † |
| steps at σ/a 4 (τ 8, 4, 2) | envelope, resolved and certified | A1.0 | 1.452..1.496 (2) | 2.783..4.827 (2) | 5.860..10.206 (2) |
| steps at σ/a 4 (τ 8, 4, 2) | envelope, resolved and certified | A1.1 | 1.495..1.541 (2) | 2.823..4.866 (2) | 5.897..10.245 (2) |
| steps at σ/a 4 (τ 8, 4, 2) | envelope, resolved and certified | A1.2 | 1.540..1.586 (2) | 2.865..4.907 (2) | 5.937..10.286 (2) |
| steps at σ/a 4 (τ 8, 4, 2) | envelope, resolved and certified | A1.3 | 1.581..1.628 (2) | 2.905..4.946 (2) | 5.976..10.325 (2) |
| steps at σ/a 4 (τ 8, 4, 2) | envelope, resolved and certified | A1.4 | 1.651..1.699 (2) | 2.975..5.015 (2) | 6.044..10.393 (2) |
| steps at σ/a 4 (τ 8, 4, 2) | envelope, resolved and certified | A1.5 | 1.713..1.762 (2) | 3.039..5.078 (2) | 6.107..10.456 (2) |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s8 | A1.0 (r_0 0.50 r_s) | 1.720 | 2.010 ‡ | 6.699 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s8 | A1.1 (r_0 1.00 r_s) | 1.760 | 2.010 ‡ | 6.699 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s8 | A1.2 (r_0 1.50 r_s) | 1.796 | 2.010 ‡ | 6.699 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s8 | A1.3 (r_0 2.00 r_s) | 1.831 | 2.010 ‡ | 6.699 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s8 | A1.4 (r_0 3.00 r_s) | 1.898 | 2.011 ‡ | 6.699 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s8 | A1.5 (r_0 4.00 r_s) | 1.970 | 2.011 ‡ | 6.699 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s16 | A1.0 (r_0 0.50 r_s) | 1.566 | 0.807 ‡ | -0.485 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s16 | A1.1 (r_0 1.00 r_s) | 1.606 | 0.807 ‡ | -0.485 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s16 | A1.2 (r_0 1.50 r_s) | 1.643 | 0.807 ‡ | -0.485 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s16 | A1.3 (r_0 2.00 r_s) | 1.677 | 0.807 ‡ | -0.485 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s16 | A1.4 (r_0 3.00 r_s) | 1.743 | 0.807 ‡ | -0.485 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | S4_128_s16 | A1.5 (r_0 4.00 r_s) | 1.812 | 0.807 ‡ | -0.485 ‡ |
| widths at r_s/a 128 (σ/a 8, 16) | envelope, resolved and certified | A1.0 | 1.566..1.720 (2) | - | - |
| widths at r_s/a 128 (σ/a 8, 16) | envelope, resolved and certified | A1.1 | 1.606..1.760 (2) | - | - |
| widths at r_s/a 128 (σ/a 8, 16) | envelope, resolved and certified | A1.2 | 1.643..1.796 (2) | - | - |
| widths at r_s/a 128 (σ/a 8, 16) | envelope, resolved and certified | A1.3 | 1.677..1.831 (2) | - | - |
| widths at r_s/a 128 (σ/a 8, 16) | envelope, resolved and certified | A1.4 | 1.743..1.898 (2) | - | - |
| widths at r_s/a 128 (σ/a 8, 16) | envelope, resolved and certified | A1.5 | 1.812..1.970 (2) | - | - |

## 9. The dispersive pass

### 9.0 v4: the rays at the headline events (rz_v3_dispev_v1.py on the exact-release passes; tables rz_v4_dispev_tables_v3.py, out/v41_dispev_tables_v3/dispev4_tables.md, verbatim)

Producing script: code/rz_v4_dispev_tables_v3.py (md5 de7d5069fe389f6474f5174289174c46). Table file: out/v41_dispev_tables_v3/dispev4_tables.md (md5 d079deaba258889972649730d7989a3b). Launcher code/v41tables_run1.sh. Made from v4's code/rz_v4_dispev_tables_v2.py (md5 c049cccb94e8131d7497803f6a6fce75) by code/mk_dispev_tables_v4_v3_patch.py (md5 d38e223dbb53ee268239b9bb0688886f): its groups called arrival clusters and its statistics cluster-based; its json equals v4's out/v4_dispev_tables_v2/dispev4_tables.json: yes. Data: v4's out/v4_dispev_v1/dispev_<member>.json (rz_v3_dispev_v1.py on the exact-release passes).

**Table 9-H1v4. Dispersive rays at the headline events themselves** (rz_v3_dispev_v1.py on the exact-release passes rz_rays_v4n / rz_rays_v4nc; this table: rz_v4_dispev_tables_v3.py; massless radial channel; the class's top 5 rows by L_T at 4 r_s inside the solver-certified record interval; A1 r_e ≥ 1 a and A2 at the exact emission events of the exact-release clock implementation from the backward map, B at its launch event; C not used). Per row: the ARRIVAL CLUSTER it belongs to (K#: the A rows of one class and one clock whose arrivals form one consecutive run on the 1 a grid; their emission times differ, by the spread printed after Table 9-H2v4, so a cluster is not one emission event; every B row is its own launch, a cluster of one); the stored L_T and its arrival T_D/r_s; the continuum front re-integrated forward from the event (L_T, a reproduction check; '-' where the stored arrival is the last record); per emitted lattice momentum ap_e: the dispersive ray's arrival T_D/r_s and L_F = ln(ε(p_e)/Ω_D) (reviewer A's convention F; coordinate frequency at the receiver), L_K = ln(ω_K/Ω_D) (reviewer B's phase rate along the normal-comoving emitter), and the static receiver's L_gF = L_F + ln A_D and L_gK = L_K + ln A_D (two conventions, kept distinct); nr = not received by the last record; ‡ = the ray's ap passed π/2.

| member | cluster | class | stored L_T (T_D/r_s) | continuum from the event: L_T | ap_e 0.05: T_D/r_s, L_F / L_K / L_gF / L_gK | ap_e 0.1 | ap_e 0.2 | ap_e 0.3 |
|---|---|---|---|---|---|---|---|---|
| P64_s4 | K1 | A1 r_e>=1a | 10.456 (40.000) | - | 29.75, 4.985 / 4.985 / 4.854 / 4.855 | 26.09, 3.988 / 3.990 / 3.857 / 3.859 | 22.17, 2.518 / 2.526 / 2.400 / 2.408 | 20.27, 1.767 / 1.785 / 1.649 / 1.667 |
| P64_s4 | K1 | A1 r_e>=1a | 10.447 (39.984) | 10.447 | 29.75, 4.985 / 4.985 / 4.854 / 4.855 | 26.09, 3.988 / 3.990 / 3.857 / 3.859 | 22.17, 2.518 / 2.526 / 2.400 / 2.408 | 20.27, 1.767 / 1.785 / 1.649 / 1.667 |
| P64_s4 | K1 | A1 r_e>=1a | 10.437 (39.969) | 10.437 | 29.75, 4.985 / 4.985 / 4.854 / 4.855 | 26.09, 3.988 / 3.990 / 3.857 / 3.859 | 22.17, 2.518 / 2.526 / 2.400 / 2.408 | 20.27, 1.767 / 1.785 / 1.649 / 1.667 |
| P64_s4 | K1 | A1 r_e>=1a | 10.428 (39.953) | 10.428 | 29.75, 4.984 / 4.985 / 4.854 / 4.854 | 26.09, 3.988 / 3.990 / 3.857 / 3.859 | 22.17, 2.518 / 2.526 / 2.400 / 2.408 | 20.27, 1.767 / 1.785 / 1.649 / 1.667 |
| P64_s4 | K1 | A1 r_e>=1a | 10.419 (39.938) | 10.419 | 29.75, 4.984 / 4.985 / 4.854 / 4.854 | 26.09, 3.988 / 3.990 / 3.857 / 3.859 | 22.17, 2.518 / 2.526 / 2.400 / 2.408 | 20.27, 1.767 / 1.785 / 1.649 / 1.667 |
| P64_s4 | K2 | A2 | 9.452 (40.000) | - | 33.24, 5.692 / 5.694 / 5.568 / 5.570 | 30.87, 4.335 / 4.342 / 4.207 / 4.214 | 28.77, 3.543 / 3.572 / 3.411 / 3.439 | 27.27, 3.286 / 3.349 / 3.153 / 3.216 |
| P64_s4 | K2 | A2 | 9.443 (39.984) | 9.443 | 33.24, 5.692 / 5.694 / 5.568 / 5.569 | 30.87, 4.335 / 4.342 / 4.207 / 4.214 | 28.76, 3.543 / 3.572 / 3.411 / 3.439 | 27.27, 3.286 / 3.349 / 3.153 / 3.216 |
| P64_s4 | K2 | A2 | 9.434 (39.969) | 9.434 | 33.24, 5.692 / 5.693 / 5.567 / 5.569 | 30.87, 4.335 / 4.342 / 4.207 / 4.214 | 28.76, 3.543 / 3.572 / 3.411 / 3.439 | 27.27, 3.286 / 3.349 / 3.153 / 3.216 |
| P64_s4 | K2 | A2 | 9.425 (39.953) | 9.425 | 33.24, 5.692 / 5.693 / 5.567 / 5.569 | 30.87, 4.335 / 4.342 / 4.207 / 4.214 | 28.76, 3.543 / 3.572 / 3.411 / 3.439 | 27.27, 3.286 / 3.349 / 3.153 / 3.216 |
| P64_s4 | K2 | A2 | 9.416 (39.938) | 9.416 | 33.24, 5.691 / 5.693 / 5.567 / 5.569 | 30.87, 4.335 / 4.342 / 4.207 / 4.214 | 28.76, 3.543 / 3.572 / 3.411 / 3.439 | 27.27, 3.286 / 3.349 / 3.153 / 3.216 |
| P64_s4 | K3 | B | 9.128 (39.746) | 9.128 | 34.58, 6.086 / 6.087 / 5.963 / 5.964 | 32.03, 4.752 / 4.758 / 4.626 / 4.632 | 29.81, 3.371 / 3.397 / 3.241 / 3.266 | 28.54, 2.726 / 2.783 / 2.594 / 2.650 |
| P64_s4 | K4 | B | 8.344 (39.932) | 8.344 | 37.25, 6.647 / 6.649 / 6.527 / 6.529 | 35.22, 5.513 / 5.520 / 5.391 / 5.398 | 32.66, 4.251 / 4.277 / 4.126 / 4.152 | 31.15, 3.294 / 3.352 / 3.167 / 3.225 |
| P64_s4 | K5 | B | 8.319 (39.888) | 8.319 | 37.24, 6.642 / 6.644 / 6.522 / 6.524 | 35.22, 5.512 / 5.519 / 5.390 / 5.397 | 32.66, 4.250 / 4.277 / 4.125 / 4.152 | 31.15, 3.294 / 3.352 / 3.167 / 3.225 |
| P64_s4 | K6 | B | 8.278 (39.818) | 8.278 | 37.23, 6.634 / 6.635 / 6.514 / 6.516 | 35.22, 5.510 / 5.516 / 5.388 / 5.394 | 32.66, 4.250 / 4.276 / 4.125 / 4.151 | 31.15, 3.294 / 3.352 / 3.166 / 3.224 |
| P64_s4 | K7 | B | 8.213 (39.708) | 8.213 | 37.21, 6.620 / 6.622 / 6.500 / 6.502 | 35.21, 5.506 / 5.513 / 5.384 / 5.391 | 32.65, 4.248 / 4.275 / 4.123 / 4.150 | 31.15, 3.293 / 3.351 / 3.166 / 3.224 |
| P64_s2 | K1 | A1 r_e>=1a | 15.676 (40.000) | - | 26.29, 4.625 / 4.625 / 4.504 / 4.505 | 24.40, 3.278 / 3.280 / 3.169 / 3.171 | 21.60, 2.430 / 2.438 / 2.323 / 2.331 | 19.96, 1.733 / 1.752 / 1.626 / 1.645 |
| P64_s2 | K1 | A1 r_e>=1a | 15.664 (39.984) | 15.665 | 26.29, 4.625 / 4.625 / 4.504 / 4.505 | 24.40, 3.278 / 3.280 / 3.169 / 3.171 | 21.60, 2.430 / 2.438 / 2.323 / 2.331 | 19.96, 1.733 / 1.752 / 1.626 / 1.645 |
| P64_s2 | K1 | A1 r_e>=1a | 15.653 (39.969) | 15.653 | 26.29, 4.625 / 4.625 / 4.504 / 4.505 | 24.40, 3.278 / 3.280 / 3.169 / 3.171 | 21.60, 2.430 / 2.438 / 2.323 / 2.331 | 19.96, 1.733 / 1.752 / 1.626 / 1.645 |
| P64_s2 | K1 | A1 r_e>=1a | 15.642 (39.953) | 15.642 | 26.29, 4.625 / 4.625 / 4.504 / 4.505 | 24.40, 3.278 / 3.280 / 3.169 / 3.171 | 21.60, 2.430 / 2.438 / 2.323 / 2.331 | 19.96, 1.733 / 1.752 / 1.626 / 1.645 |
| P64_s2 | K1 | A1 r_e>=1a | 15.630 (39.938) | 15.630 | 26.29, 4.625 / 4.625 / 4.504 / 4.505 | 24.40, 3.278 / 3.280 / 3.169 / 3.171 | 21.60, 2.430 / 2.438 / 2.323 / 2.331 | 19.96, 1.733 / 1.752 / 1.626 / 1.645 |
| P64_s2 | K2 | A2 | 14.395 (40.000) | - | 29.95, 6.259 / 6.261 / 6.142 / 6.144 | 28.21, 4.841 / 4.848 / 4.722 / 4.729 | 26.46, 3.497 / 3.525 / 3.376 / 3.404 | 25.34, 2.780 / 2.842 / 2.665 / 2.727 |
| P64_s2 | K2 | A2 | 14.384 (39.984) | 14.384 | 29.95, 6.259 / 6.261 / 6.142 / 6.144 | 28.21, 4.841 / 4.848 / 4.722 / 4.729 | 26.46, 3.497 / 3.525 / 3.376 / 3.404 | 25.34, 2.780 / 2.842 / 2.665 / 2.727 |
| P64_s2 | K2 | A2 | 14.372 (39.969) | 14.372 | 29.95, 6.259 / 6.261 / 6.142 / 6.144 | 28.21, 4.841 / 4.848 / 4.722 / 4.729 | 26.46, 3.497 / 3.525 / 3.376 / 3.404 | 25.34, 2.780 / 2.842 / 2.665 / 2.727 |
| P64_s2 | K2 | A2 | 14.361 (39.953) | 14.361 | 29.95, 6.259 / 6.261 / 6.142 / 6.144 | 28.21, 4.841 / 4.848 / 4.722 / 4.729 | 26.46, 3.497 / 3.525 / 3.376 / 3.404 | 25.34, 2.780 / 2.842 / 2.665 / 2.727 |
| P64_s2 | K2 | A2 | 14.350 (39.938) | 14.350 | 29.95, 6.259 / 6.261 / 6.142 / 6.144 | 28.21, 4.841 / 4.848 / 4.722 / 4.729 | 26.46, 3.497 / 3.525 / 3.376 / 3.404 | 25.34, 2.780 / 2.842 / 2.665 / 2.727 |
| P64_s2 | K3 | B | 10.033 (32.635) | 10.033 | 27.61, 5.988 / 5.989 / 5.868 / 5.869 | 25.85, 4.698 / 4.704 / 4.580 / 4.585 | 23.85, 3.711 / 3.733 / 3.604 / 3.626 | 21.95, 2.922 / 2.971 / 2.815 / 2.864 |
| P64_s2 | K4 | B | 9.974 (33.246) | 9.974 | 28.90, 6.467 / 6.469 / 6.349 / 6.350 | 27.20, 5.110 / 5.117 / 4.989 / 4.996 | 25.36, 3.802 / 3.829 / 3.687 / 3.713 | 24.15, 3.057 / 3.115 / 2.948 / 3.007 |
| P64_s2 | K5 | B | 9.041 (36.612) | 9.041 | 33.75, 6.722 / 6.724 / 6.609 / 6.611 | 32.14, 5.370 / 5.376 / 5.255 / 5.261 | 30.43, 4.080 / 4.106 / 3.963 / 3.990 | 29.30, 3.257 / 3.315 / 3.140 / 3.198 |
| P64_s2 | K6 | B | 8.973 (36.527) | 8.973 | 33.74, 6.715 / 6.717 / 6.602 / 6.604 | 32.14, 5.368 / 5.374 / 5.253 / 5.259 | 30.43, 4.079 / 4.106 / 3.963 / 3.989 | 29.30, 3.257 / 3.315 / 3.139 / 3.197 |
| P64_s2 | K7 | B | 8.910 (36.331) | 8.910 | 33.62, 6.707 / 6.709 / 6.594 / 6.596 | 32.03, 5.365 / 5.372 / 5.250 / 5.257 | 30.31, 4.082 / 4.108 / 3.965 / 3.991 | 29.19, 3.256 / 3.315 / 3.139 / 3.197 |
| S4_128_s8 | K1 | A1 r_e>=1a | 3.434 (25.211) | 3.434 | 24.83, 3.402 / 3.403 / 3.260 / 3.260 | 23.83, 3.155 / 3.157 / 3.033 / 3.035 | 21.16, 2.465 / 2.474 / 2.346 / 2.355 | 19.36, 1.765 / 1.784 / 1.644 / 1.664 |
| S4_128_s8 | K1 | A1 r_e>=1a | 3.434 (25.219) | 3.434 | 24.83, 3.403 / 3.404 / 3.261 / 3.261 | 23.84, 3.156 / 3.159 / 3.034 / 3.036 | 21.16, 2.467 / 2.476 / 2.348 / 2.357 | 19.36, 1.765 / 1.784 / 1.645 / 1.664 |
| S4_128_s8 | K1 | A1 r_e>=1a | 3.434 (25.203) | 3.434 | 24.82, 3.401 / 3.402 / 3.258 / 3.259 | 23.83, 3.153 / 3.156 / 3.031 / 3.034 | 21.16, 2.464 / 2.473 / 2.345 / 2.354 | 19.35, 1.764 / 1.783 / 1.644 / 1.663 |
| S4_128_s8 | K1 | A1 r_e>=1a | 3.434 (25.227) | 3.434 | 24.84, 3.404 / 3.405 / 3.262 / 3.262 | 23.85, 3.158 / 3.160 / 3.036 / 3.038 | 21.17, 2.468 / 2.477 / 2.349 / 2.358 | 19.36, 1.766 / 1.785 / 1.645 / 1.665 |
| S4_128_s8 | K1 | A1 r_e>=1a | 3.434 (25.195) | 3.434 | 24.81, 3.400 / 3.400 / 3.257 / 3.258 | 23.82, 3.152 / 3.154 / 3.030 / 3.032 | 21.16, 2.463 / 2.471 / 2.344 / 2.352 | 19.35, 1.763 / 1.783 / 1.643 / 1.663 |
| S4_128_s8 | K2 | A2 | 7.754 (40.000) | 7.754 | 37.16, 6.339 / 6.340 / 6.215 / 6.216 | 34.89, 5.308 / 5.314 / 5.180 / 5.186 | 31.27, 4.136 / 4.160 / 4.007 / 4.031 | 29.60, 3.034 / 3.086 / 2.900 / 2.953 |
| S4_128_s8 | K2 | A2 | 7.750 (39.992) | 7.750 | 37.16, 6.338 / 6.339 / 6.213 / 6.215 | 34.89, 5.308 / 5.314 / 5.180 / 5.186 | 31.27, 4.136 / 4.160 / 4.007 / 4.030 | 29.60, 3.034 / 3.086 / 2.900 / 2.953 |
| S4_128_s8 | K2 | A2 | 7.746 (39.984) | 7.746 | 37.16, 6.337 / 6.338 / 6.212 / 6.214 | 34.89, 5.307 / 5.313 / 5.180 / 5.186 | 31.27, 4.136 / 4.160 / 4.007 / 4.030 | 29.60, 3.034 / 3.086 / 2.900 / 2.953 |
| S4_128_s8 | K2 | A2 | 7.743 (39.977) | 7.743 | 37.16, 6.335 / 6.337 / 6.211 / 6.213 | 34.89, 5.307 / 5.313 / 5.179 / 5.185 | 31.27, 4.136 / 4.160 / 4.006 / 4.030 | 29.60, 3.034 / 3.086 / 2.900 / 2.953 |
| S4_128_s8 | K2 | A2 | 7.739 (39.969) | 7.739 | 37.16, 6.334 / 6.336 / 6.210 / 6.211 | 34.89, 5.307 / 5.313 / 5.179 / 5.185 | 31.27, 4.136 / 4.160 / 4.006 / 4.030 | 29.60, 3.034 / 3.086 / 2.900 / 2.953 |
| S4_128_s8 | K3 | B | 7.723 (39.938) | 7.723 | 37.09, 6.294 / 6.295 / 6.169 / 6.171 | 34.81, 5.284 / 5.290 / 5.156 / 5.162 | 31.17, 4.073 / 4.097 / 3.943 / 3.967 | 29.54, 2.998 / 3.050 / 2.864 / 2.916 |
| S4_128_s8 | K4 | B | 7.242 (39.428) | 7.242 | 37.79, 6.491 / 6.492 / 6.367 / 6.369 | 35.96, 5.449 / 5.456 / 5.323 / 5.329 | 33.04, 4.663 / 4.689 / 4.536 / 4.562 | 30.77, 3.558 / 3.616 / 3.428 / 3.485 |
| S4_128_s8 | K5 | B | 7.178 (38.643) | 7.178 | 36.78, 6.110 / 6.111 / 5.984 / 5.986 | 34.77, 5.274 / 5.280 / 5.146 / 5.152 | 31.25, 4.127 / 4.151 / 3.997 / 4.021 | 29.61, 3.038 / 3.091 / 2.905 / 2.957 |
| S4_128_s8 | K6 | B | 7.021 (39.506) | 7.021 | 38.14, 6.433 / 6.435 / 6.310 / 6.312 | 36.43, 5.416 / 5.422 / 5.290 / 5.297 | 33.90, 4.593 / 4.620 / 4.466 / 4.492 | 31.53, 3.847 / 3.906 / 3.718 / 3.777 |
| S4_128_s8 | K7 | B | 6.910 (38.732) | 6.910 | 37.53, 6.264 / 6.265 / 6.140 / 6.141 | 35.96, 5.383 / 5.389 / 5.256 / 5.263 | 33.23, 4.644 / 4.670 / 4.517 / 4.543 | 30.94, 3.619 / 3.676 / 3.489 / 3.546 |
| P64_s4_tau4 | K1 | A1 r_e>=1a | 6.107 (40.000) | - | 32.15, 3.055 / 3.055 / 2.921 / 2.922 | 30.82, 2.791 / 2.793 / 2.654 / 2.656 | 28.61, 3.217 / 3.225 / 3.075 / 3.083 | 22.38, 2.638 / 2.656 / 2.515 / 2.533 |
| P64_s4_tau4 | K1 | A1 r_e>=1a | 6.099 (39.984) | 6.099 | 32.15, 3.056 / 3.056 / 2.922 / 2.922 | 30.82, 2.792 / 2.794 / 2.654 / 2.656 | 28.61, 3.217 / 3.225 / 3.075 / 3.083 | 22.38, 2.638 / 2.656 / 2.515 / 2.533 |
| P64_s4_tau4 | K1 | A1 r_e>=1a | 6.090 (39.969) | 6.090 | 32.16, 3.057 / 3.057 / 2.923 / 2.923 | 30.82, 2.792 / 2.794 / 2.655 / 2.657 | 28.61, 3.218 / 3.226 / 3.076 / 3.084 | 22.38, 2.638 / 2.656 / 2.515 / 2.532 |
| P64_s4_tau4 | K1 | A1 r_e>=1a | 6.082 (39.953) | 6.082 | 32.16, 3.058 / 3.058 / 2.924 / 2.924 | 30.82, 2.792 / 2.794 / 2.655 / 2.657 | 28.61, 3.218 / 3.226 / 3.076 / 3.084 | 22.38, 2.638 / 2.656 / 2.514 / 2.532 |
| P64_s4_tau4 | K1 | A1 r_e>=1a | 6.073 (39.938) | 6.073 | 32.16, 3.059 / 3.059 / 2.925 / 2.925 | 30.82, 2.793 / 2.795 / 2.656 / 2.658 | 28.61, 3.219 / 3.227 / 3.077 / 3.085 | 22.38, 2.638 / 2.655 / 2.514 / 2.532 |
| P64_s4_tau4 | K2 | A2 | 7.184 (40.000) | - | 36.96, 5.855 / 5.857 / 5.727 / 5.729 | 33.76, 4.858 / 4.864 / 4.727 / 4.733 | 31.00, 3.206 / 3.229 / 3.070 / 3.092 | 29.66, 2.541 / 2.590 / 2.401 / 2.450 |
| P64_s4_tau4 | K2 | A2 | 7.175 (39.984) | 7.175 | 36.95, 5.854 / 5.856 / 5.726 / 5.727 | 33.75, 4.857 / 4.863 / 4.727 / 4.732 | 31.00, 3.206 / 3.229 / 3.070 / 3.092 | 29.66, 2.541 / 2.590 / 2.401 / 2.450 |
| P64_s4_tau4 | K2 | A2 | 7.166 (39.969) | 7.166 | 36.95, 5.853 / 5.854 / 5.725 / 5.726 | 33.75, 4.857 / 4.862 / 4.726 / 4.732 | 31.00, 3.206 / 3.229 / 3.069 / 3.092 | 29.66, 2.541 / 2.590 / 2.401 / 2.450 |
| P64_s4_tau4 | K2 | A2 | 7.156 (39.953) | 7.156 | 36.94, 5.852 / 5.853 / 5.724 / 5.725 | 33.75, 4.856 / 4.862 / 4.725 / 4.731 | 31.00, 3.206 / 3.228 / 3.069 / 3.092 | 29.66, 2.541 / 2.590 / 2.401 / 2.450 |
| P64_s4_tau4 | K2 | A2 | 7.147 (39.938) | 7.147 | 36.94, 5.851 / 5.852 / 5.723 / 5.724 | 33.75, 4.855 / 4.861 / 4.724 / 4.730 | 31.00, 3.206 / 3.228 / 3.069 / 3.092 | 29.66, 2.541 / 2.590 / 2.401 / 2.450 |
| P64_s4_tau4 | K3 | B | 6.596 (38.780) | 6.596 | 37.19, 5.990 / 5.991 / 5.862 / 5.863 | 34.70, 5.248 / 5.254 / 5.119 / 5.125 | 31.73, 3.669 / 3.693 / 3.534 / 3.558 | 30.33, 2.861 / 2.914 / 2.722 / 2.775 |
| P64_s4_tau4 | K4 | B | 6.553 (38.690) | 6.553 | 37.03, 5.945 / 5.946 / 5.816 / 5.818 | 34.49, 5.184 / 5.190 / 5.054 / 5.060 | 31.58, 3.584 / 3.607 / 3.448 / 3.472 | 30.20, 2.805 / 2.857 / 2.666 / 2.719 |
| P64_s4_tau4 | K5 | B | 6.514 (39.630) | 6.514 | 38.75, 6.051 / 6.053 / 5.924 / 5.926 | 37.22, 5.467 / 5.473 / 5.339 / 5.345 | 34.20, 4.566 / 4.592 / 4.436 / 4.462 | 32.23, 3.462 / 3.520 / 3.328 / 3.386 |
| P64_s4_tau4 | K6 | B | 6.505 (39.272) | 6.505 | 35.33, 5.138 / 5.139 / 5.009 / 5.010 | 32.21, 3.681 / 3.686 / 3.548 / 3.552 | 29.94, 2.474 / 2.493 / 2.335 / 2.353 | 28.69, 2.035 / 2.076 / 1.893 / 1.935 |
| P64_s4_tau4 | K7 | B | 6.494 (39.335) | 6.494 | 38.44, 6.053 / 6.054 / 5.926 / 5.928 | 36.83, 5.499 / 5.505 / 5.370 / 5.377 | 33.72, 4.504 / 4.530 / 4.373 / 4.400 | 31.89, 3.376 / 3.434 / 3.242 / 3.300 |
| W2 | K1 | A1 r_e>=1a | 3.517 (26.000) | 3.517 | 25.61, 3.473 / 3.474 / 3.320 / 3.320 | 24.63, 3.141 / 3.144 / 3.008 / 3.010 | 21.99, 2.520 / 2.529 / 2.389 / 2.398 | 20.17, 1.852 / 1.872 / 1.721 / 1.741 |
| W2 | K1 | A1 r_e>=1a | 3.517 (25.984) | 3.517 | 25.59, 3.471 / 3.471 / 3.317 / 3.318 | 24.62, 3.140 / 3.143 / 3.007 / 3.009 | 21.98, 2.518 / 2.527 / 2.387 / 2.396 | 20.16, 1.851 / 1.871 / 1.720 / 1.740 |
| W2 | K1 | A1 r_e>=1a | 3.517 (26.016) | 3.517 | 25.62, 3.476 / 3.476 / 3.322 / 3.323 | 24.64, 3.143 / 3.145 / 3.009 / 3.011 | 21.99, 2.523 / 2.531 / 2.392 / 2.401 | 20.17, 1.853 / 1.873 / 1.722 / 1.742 |
| W2 | K1 | A1 r_e>=1a | 3.517 (25.969) | 3.517 | 25.58, 3.468 / 3.469 / 3.315 / 3.316 | 24.61, 3.140 / 3.142 / 3.006 / 3.008 | 21.98, 2.515 / 2.524 / 2.385 / 2.393 | 20.16, 1.850 / 1.870 / 1.719 / 1.739 |
| W2 | K1 | A1 r_e>=1a | 3.517 (26.031) | 3.517 | 25.64, 3.478 / 3.479 / 3.324 / 3.325 | 24.66, 3.145 / 3.147 / 3.011 / 3.013 | 22.00, 2.525 / 2.534 / 2.394 / 2.403 | 20.17, 1.854 / 1.874 / 1.723 / 1.743 |
| W2 | K2 | A2 | 4.543 (36.656) | 4.543 | 35.59, 4.401 / 4.402 / 4.264 / 4.265 | 33.73, 3.519 / 3.522 / 3.379 / 3.382 | 31.67, 2.404 / 2.416 / 2.259 / 2.271 | 30.41, 1.917 / 1.945 / 1.768 / 1.796 |
| W2 | K2 | A2 | 4.543 (36.641) | 4.543 | 35.58, 4.399 / 4.400 / 4.262 / 4.263 | 33.73, 3.516 / 3.519 / 3.376 / 3.379 | 31.67, 2.403 / 2.416 / 2.258 / 2.270 | 30.41, 1.917 / 1.945 / 1.768 / 1.795 |
| W2 | K2 | A2 | 4.543 (36.672) | 4.543 | 35.61, 4.403 / 4.404 / 4.266 / 4.267 | 33.74, 3.523 / 3.526 / 3.382 / 3.385 | 31.67, 2.405 / 2.417 / 2.259 / 2.272 | 30.41, 1.918 / 1.945 / 1.768 / 1.796 |
| W2 | K2 | A2 | 4.542 (36.625) | 4.542 | 35.57, 4.397 / 4.398 / 4.260 / 4.261 | 33.72, 3.513 / 3.516 / 3.372 / 3.376 | 31.67, 2.403 / 2.415 / 2.257 / 2.270 | 30.41, 1.917 / 1.944 / 1.767 / 1.795 |
| W2 | K2 | A2 | 4.542 (36.688) | 4.542 | 35.62, 4.405 / 4.406 / 4.268 / 4.269 | 33.74, 3.526 / 3.529 / 3.386 / 3.389 | 31.67, 2.406 / 2.418 / 2.260 / 2.273 | 30.41, 1.918 / 1.945 / 1.769 / 1.796 |
| W2 | K3 | B | 4.877 (36.581) | 4.877 | 36.22, 4.832 / 4.834 / 4.696 / 4.697 | 35.25, 4.656 / 4.662 / 4.519 / 4.524 | 33.28, 3.545 / 3.568 / 3.404 / 3.427 | 32.01, 2.817 / 2.868 / 2.673 / 2.724 |
| W2 | K4 | B | 4.816 (37.203) | 4.816 | 36.84, 4.869 / 4.871 / 4.732 / 4.734 | 35.78, 4.752 / 4.758 / 4.615 / 4.621 | 33.45, 3.646 / 3.669 / 3.505 / 3.528 | 32.09, 2.856 / 2.907 / 2.712 / 2.763 |
| W2 | K5 | B | 4.754 (35.720) | 4.754 | 35.38, 4.696 / 4.697 / 4.559 / 4.560 | 34.53, 4.329 / 4.335 / 4.190 / 4.196 | 32.95, 3.355 / 3.378 / 3.213 / 3.236 | 31.78, 2.711 / 2.761 / 2.565 / 2.616 |
| W2 | K6 | B | 4.736 (35.595) | 4.736 | 35.29, 4.674 / 4.675 / 4.536 / 4.538 | 34.53, 4.326 / 4.332 / 4.187 / 4.193 | 33.03, 3.399 / 3.422 / 3.257 / 3.280 | 31.89, 2.756 / 2.807 / 2.611 / 2.662 |
| W2 | K7 | B | 4.734 (36.267) | 4.734 | 36.05, 4.693 / 4.694 / 4.556 / 4.558 | 35.43, 4.591 / 4.597 / 4.453 / 4.460 | 33.81, 3.759 / 3.784 / 3.619 / 3.644 | 32.60, 3.019 / 3.074 / 2.876 / 2.931 |
| P64_s8 | K1 | A1 r_e>=1a | 3.241 (25.578) | 3.241 | 25.03, 3.157 / 3.157 / 3.031 / 3.032 | 23.82, 2.737 / 2.739 / 2.611 / 2.613 | 21.65, 1.926 / 1.933 / 1.800 / 1.807 | 20.06, 1.465 / 1.481 / 1.339 / 1.355 |
| P64_s8 | K1 | A1 r_e>=1a | 3.241 (25.594) | 3.241 | 25.04, 3.160 / 3.161 / 3.034 / 3.035 | 23.83, 2.740 / 2.742 / 2.614 / 2.616 | 21.65, 1.927 / 1.934 / 1.801 / 1.808 | 20.07, 1.466 / 1.481 / 1.340 / 1.355 |
| P64_s8 | K1 | A1 r_e>=1a | 3.240 (25.562) | 3.240 | 25.01, 3.154 / 3.154 / 3.028 / 3.028 | 23.81, 2.734 / 2.735 / 2.608 / 2.609 | 21.65, 1.925 / 1.932 / 1.799 / 1.806 | 20.06, 1.465 / 1.480 / 1.339 / 1.354 |
| P64_s8 | K1 | A1 r_e>=1a | 3.240 (25.609) | 3.240 | 25.06, 3.164 / 3.164 / 3.038 / 3.038 | 23.84, 2.744 / 2.745 / 2.618 / 2.619 | 21.66, 1.928 / 1.935 / 1.802 / 1.809 | 20.07, 1.466 / 1.482 / 1.340 / 1.356 |
| P64_s8 | K1 | A1 r_e>=1a | 3.240 (25.547) | 3.240 | 25.00, 3.150 / 3.151 / 3.024 / 3.025 | 23.80, 2.730 / 2.732 / 2.604 / 2.606 | 21.64, 1.924 / 1.931 / 1.798 / 1.805 | 20.06, 1.464 / 1.480 / 1.338 / 1.354 |
| P64_s8 | K2 | A2 | 3.172 (38.875) | 3.172 | 38.78, 3.164 / 3.165 / 3.037 / 3.039 | 38.52, 3.129 / 3.135 / 3.002 / 3.008 | 37.65, 2.915 / 2.940 / 2.788 / 2.812 | 36.66, 2.573 / 2.626 / 2.445 / 2.498 |
| P64_s8 | K2 | A2 | 3.172 (38.859) | 3.172 | 38.77, 3.163 / 3.165 / 3.037 / 3.038 | 38.51, 3.127 / 3.133 / 3.001 / 3.007 | 37.64, 2.913 / 2.937 / 2.785 / 2.810 | 36.65, 2.571 / 2.624 / 2.442 / 2.496 |
| P64_s8 | K2 | A2 | 3.172 (38.891) | 3.172 | 38.80, 3.164 / 3.166 / 3.038 / 3.039 | 38.53, 3.130 / 3.136 / 3.003 / 3.009 | 37.66, 2.918 / 2.942 / 2.791 / 2.815 | 36.66, 2.575 / 2.629 / 2.447 / 2.500 |
| P64_s8 | K2 | A2 | 3.171 (38.844) | 3.171 | 38.75, 3.163 / 3.164 / 3.036 / 3.038 | 38.49, 3.126 / 3.132 / 2.999 / 3.005 | 37.63, 2.910 / 2.934 / 2.783 / 2.807 | 36.64, 2.569 / 2.622 / 2.440 / 2.493 |
| P64_s8 | K2 | A2 | 3.171 (38.906) | 3.171 | 38.81, 3.164 / 3.166 / 3.038 / 3.040 | 38.55, 3.131 / 3.137 / 3.005 / 3.011 | 37.67, 2.921 / 2.945 / 2.793 / 2.817 | 36.67, 2.577 / 2.631 / 2.449 / 2.502 |
| P64_s8 | K3 | B | 3.708 (25.576) | 3.708 | 25.43, 3.697 / 3.698 / 3.571 / 3.572 | 25.01, 3.611 / 3.617 / 3.486 / 3.491 | 23.74, 3.143 / 3.165 / 3.017 / 3.039 | 22.46, 2.613 / 2.663 / 2.487 / 2.537 |
| P64_s8 | K4 | B | 3.683 (25.894) | 3.683 | 25.73, 3.702 / 3.703 / 3.576 / 3.577 | 25.25, 3.665 / 3.671 / 3.539 / 3.545 | 23.80, 3.166 / 3.188 / 3.040 / 3.062 | 22.41, 2.597 / 2.646 / 2.471 / 2.520 |
| P64_s8 | K5 | B | 3.677 (25.262) | 3.677 | 25.11, 3.644 / 3.646 / 3.518 / 3.520 | 24.69, 3.519 / 3.525 / 3.393 / 3.399 | 23.46, 3.035 / 3.057 / 2.909 / 2.931 | 22.22, 2.527 / 2.576 / 2.401 / 2.450 |
| P64_s8 | K6 | B | 3.629 (26.110) | 3.629 | 25.97, 3.666 / 3.667 / 3.539 / 3.541 | 25.53, 3.697 / 3.703 / 3.571 / 3.577 | 24.06, 3.269 / 3.291 / 3.143 / 3.165 | 22.64, 2.682 / 2.732 / 2.556 / 2.606 |
| P64_s8 | K7 | B | 3.621 (25.064) | 3.621 | 24.94, 3.587 / 3.588 / 3.461 / 3.462 | 24.60, 3.475 / 3.481 / 3.349 / 3.355 | 23.56, 3.057 / 3.080 / 2.931 / 2.954 | 22.42, 2.584 / 2.636 / 2.458 / 2.510 |
| S4_128_s16 | K1 | A1 r_e>=1a | 3.285 (24.250) | 3.285 | 23.84, 3.227 / 3.227 / 3.100 / 3.101 | 22.87, 2.884 / 2.886 / 2.757 / 2.759 | 20.88, 2.095 / 2.103 / 1.968 / 1.977 | 19.29, 1.587 / 1.605 / 1.460 / 1.479 |
| S4_128_s16 | K1 | A1 r_e>=1a | 3.285 (24.242) | 3.285 | 23.84, 3.225 / 3.226 / 3.098 / 3.099 | 22.87, 2.882 / 2.884 / 2.755 / 2.757 | 20.88, 2.094 / 2.103 / 1.968 / 1.976 | 19.29, 1.586 / 1.605 / 1.459 / 1.478 |
| S4_128_s16 | K1 | A1 r_e>=1a | 3.285 (24.258) | 3.285 | 23.85, 3.229 / 3.229 / 3.102 / 3.103 | 22.88, 2.886 / 2.888 / 2.759 / 2.761 | 20.88, 2.096 / 2.104 / 1.969 / 1.978 | 19.29, 1.587 / 1.606 / 1.460 / 1.479 |
| S4_128_s16 | K1 | A1 r_e>=1a | 3.285 (24.234) | 3.285 | 23.83, 3.223 / 3.224 / 3.097 / 3.097 | 22.86, 2.880 / 2.882 / 2.753 / 2.755 | 20.87, 2.093 / 2.102 / 1.967 / 1.975 | 19.29, 1.586 / 1.604 / 1.459 / 1.478 |
| S4_128_s16 | K1 | A1 r_e>=1a | 3.285 (24.266) | 3.285 | 23.86, 3.230 / 3.231 / 3.104 / 3.104 | 22.88, 2.888 / 2.890 / 2.761 / 2.764 | 20.88, 2.097 / 2.105 / 1.970 / 1.978 | 19.30, 1.587 / 1.606 / 1.461 / 1.479 |
| S4_128_s16 | K2 | A2 | 2.782 (36.945) | 2.782 | 36.69, 2.751 / 2.751 / 2.623 / 2.623 | 36.05, 2.609 / 2.612 / 2.480 / 2.484 | 34.57, 2.140 / 2.153 / 2.009 / 2.022 | 33.36, 1.769 / 1.798 / 1.637 / 1.666 |
| S4_128_s16 | K2 | A2 | 2.782 (36.953) | 2.782 | 36.70, 2.751 / 2.752 / 2.623 / 2.624 | 36.05, 2.610 / 2.613 / 2.481 / 2.485 | 34.57, 2.140 / 2.153 / 2.010 / 2.023 | 33.37, 1.769 / 1.798 / 1.637 / 1.666 |
| S4_128_s16 | K2 | A2 | 2.782 (36.938) | 2.782 | 36.68, 2.750 / 2.751 / 2.622 / 2.623 | 36.04, 2.608 / 2.611 / 2.479 / 2.483 | 34.57, 2.139 / 2.152 / 2.008 / 2.022 | 33.36, 1.769 / 1.798 / 1.636 / 1.666 |
| S4_128_s16 | K2 | A2 | 2.782 (36.961) | 2.782 | 36.70, 2.752 / 2.752 / 2.624 / 2.624 | 36.06, 2.611 / 2.614 / 2.482 / 2.486 | 34.58, 2.141 / 2.154 / 2.010 / 2.023 | 33.37, 1.769 / 1.799 / 1.637 / 1.666 |
| S4_128_s16 | K2 | A2 | 2.782 (36.930) | 2.782 | 36.67, 2.750 / 2.750 / 2.622 / 2.622 | 36.03, 2.607 / 2.610 / 2.478 / 2.482 | 34.56, 2.138 / 2.152 / 2.008 / 2.021 | 33.36, 1.769 / 1.798 / 1.636 / 1.666 |
| S4_128_s16 | K3 | B | 3.654 (24.252) | 3.654 | 24.11, 3.644 / 3.646 / 3.518 / 3.519 | 23.69, 3.549 / 3.554 / 3.422 / 3.428 | 22.45, 3.061 / 3.083 / 2.935 / 2.956 | 21.18, 2.535 / 2.583 / 2.409 / 2.457 |
| S4_128_s16 | K4 | B | 3.651 (24.201) | 3.651 | 24.05, 3.635 / 3.636 / 3.508 / 3.509 | 23.62, 3.525 / 3.530 / 3.398 / 3.403 | 22.36, 3.023 / 3.045 / 2.897 / 2.918 | 21.09, 2.498 / 2.545 / 2.372 / 2.419 |
| S4_128_s16 | K5 | B | 3.647 (24.383) | 3.647 | 24.24, 3.653 / 3.654 / 3.526 / 3.527 | 23.84, 3.589 / 3.594 / 3.462 / 3.468 | 22.58, 3.116 / 3.138 / 2.989 / 3.011 | 21.30, 2.581 / 2.630 / 2.454 / 2.503 |
| S4_128_s16 | K6 | B | 3.633 (24.042) | 3.633 | 23.91, 3.607 / 3.608 / 3.480 / 3.482 | 23.55, 3.499 / 3.505 / 3.372 / 3.378 | 22.44, 3.054 / 3.076 / 2.927 / 2.950 | 21.25, 2.559 / 2.609 / 2.433 / 2.482 |
| S4_128_s16 | K7 | B | 3.621 (23.954) | 3.621 | 23.82, 3.587 / 3.589 / 3.461 / 3.462 | 23.43, 3.462 / 3.468 / 3.336 / 3.341 | 22.29, 2.995 / 3.016 / 2.868 / 2.890 | 21.08, 2.499 / 2.547 / 2.372 / 2.420 |
| P64_s4_heldsrc | K1 | A1 r_e>=1a | 3.460 (24.406) | 3.460 | 23.98, 3.355 / 3.356 / 3.238 / 3.238 | 23.04, 2.926 / 2.928 / 2.809 / 2.811 | 21.20, 2.116 / 2.124 / 1.998 / 2.006 | 19.75, 1.611 / 1.629 / 1.493 / 1.511 |
| P64_s4_heldsrc | K1 | A1 r_e>=1a | 3.459 (24.391) | 3.459 | 23.97, 3.350 / 3.350 / 3.232 / 3.233 | 23.04, 2.922 / 2.924 / 2.804 / 2.806 | 21.20, 2.114 / 2.122 / 1.997 / 2.005 | 19.74, 1.611 / 1.629 / 1.492 / 1.511 |
| P64_s4_heldsrc | K1 | A1 r_e>=1a | 3.459 (24.422) | 3.459 | 24.00, 3.361 / 3.361 / 3.243 / 3.244 | 23.05, 2.930 / 2.932 / 2.813 / 2.815 | 21.21, 2.117 / 2.125 / 2.000 / 2.008 | 19.75, 1.612 / 1.630 / 1.494 / 1.512 |
| P64_s4_heldsrc | K1 | A1 r_e>=1a | 3.459 (24.375) | 3.459 | 23.95, 3.344 / 3.345 / 3.227 / 3.227 | 23.03, 2.917 / 2.920 / 2.800 / 2.802 | 21.20, 2.112 / 2.121 / 1.995 / 2.003 | 19.74, 1.610 / 1.628 / 1.492 / 1.510 |
| P64_s4_heldsrc | K1 | A1 r_e>=1a | 3.459 (24.438) | 3.459 | 24.01, 3.366 / 3.367 / 3.249 / 3.249 | 23.06, 2.934 / 2.936 / 2.817 / 2.819 | 21.21, 2.119 / 2.127 / 2.001 / 2.009 | 19.75, 1.613 / 1.631 / 1.494 / 1.513 |
| P64_s4_heldsrc | K2 | A2 | 2.787 (24.359) | 2.787 | 24.32, 2.784 / 2.785 / 2.665 / 2.666 | 24.19, 2.761 / 2.767 / 2.643 / 2.649 | 23.71, 2.593 / 2.618 / 2.476 / 2.500 | 23.06, 2.332 / 2.386 / 2.214 / 2.269 |
| P64_s4_heldsrc | K2 | A2 | 2.787 (24.375) | 2.787 | 24.33, 2.785 / 2.786 / 2.665 / 2.667 | 24.20, 2.764 / 2.770 / 2.646 / 2.652 | 23.73, 2.598 / 2.622 / 2.480 / 2.505 | 23.07, 2.336 / 2.390 / 2.218 / 2.273 |
| P64_s4_heldsrc | K2 | A2 | 2.787 (24.344) | 2.787 | 24.30, 2.783 / 2.784 / 2.664 / 2.665 | 24.17, 2.757 / 2.764 / 2.639 / 2.646 | 23.70, 2.588 / 2.613 / 2.471 / 2.495 | 23.05, 2.328 / 2.382 / 2.210 / 2.265 |
| P64_s4_heldsrc | K2 | A2 | 2.787 (24.391) | 2.787 | 24.35, 2.785 / 2.787 / 2.666 / 2.667 | 24.22, 2.766 / 2.773 / 2.648 / 2.655 | 23.74, 2.603 / 2.627 / 2.485 / 2.510 | 23.08, 2.339 / 2.394 / 2.222 / 2.277 |
| P64_s4_heldsrc | K2 | A2 | 2.786 (24.328) | 2.786 | 24.28, 2.781 / 2.783 / 2.662 / 2.664 | 24.16, 2.754 / 2.760 / 2.636 / 2.642 | 23.69, 2.583 / 2.608 / 2.466 / 2.491 | 23.04, 2.324 / 2.379 / 2.207 / 2.261 |
| P64_s4_heldsrc | K3 | B | 3.803 (24.520) | 3.803 | 24.36, 3.810 / 3.812 / 3.691 / 3.692 | 23.90, 3.672 / 3.677 / 3.554 / 3.560 | 22.65, 3.078 / 3.099 / 2.961 / 2.982 | 21.44, 2.523 / 2.570 / 2.406 / 2.453 |
| P64_s4_heldsrc | K4 | B | 3.713 (24.004) | 3.713 | 23.88, 3.659 / 3.660 / 3.541 / 3.543 | 23.53, 3.497 / 3.502 / 3.379 / 3.385 | 22.51, 3.008 / 3.030 / 2.891 / 2.913 | 21.43, 2.515 / 2.564 / 2.397 / 2.446 |
| P64_s4_heldsrc | K5 | B | 3.698 (24.813) | 3.698 | 24.65, 3.759 / 3.760 / 3.634 / 3.635 | 24.12, 3.746 / 3.751 / 3.628 / 3.633 | 22.67, 3.078 / 3.098 / 2.961 / 2.981 | 21.37, 2.485 / 2.530 / 2.368 / 2.412 |
| P64_s4_heldsrc | K6 | B | 3.660 (24.395) | 3.660 | 24.30, 3.651 / 3.653 / 3.532 / 3.534 | 24.01, 3.567 / 3.573 / 3.449 / 3.455 | 23.11, 3.154 / 3.178 / 3.036 / 3.061 | 22.10, 2.682 / 2.736 / 2.564 / 2.619 |
| P64_s4_heldsrc | K7 | B | 3.654 (24.963) | 3.654 | 24.84, 3.699 / 3.700 / 3.569 / 3.570 | 24.41, 3.809 / 3.815 / 3.689 / 3.695 | 23.03, 3.254 / 3.276 / 3.137 / 3.159 | 21.73, 2.648 / 2.697 / 2.531 / 2.579 |
| P64_s4_widened | K1 | A1 r_e>=1a | 3.708 (24.969) | 3.708 | 24.44, 3.569 / 3.569 / 3.444 / 3.444 | 23.35, 3.047 / 3.049 / 2.922 / 2.924 | 21.37, 2.179 / 2.187 / 2.054 / 2.062 | 19.83, 1.656 / 1.675 / 1.531 / 1.550 |
| P64_s4_widened | K1 | A1 r_e>=1a | 3.708 (24.984) | 3.708 | 24.46, 3.574 / 3.575 / 3.449 / 3.450 | 23.36, 3.051 / 3.053 / 2.926 / 2.928 | 21.37, 2.180 / 2.188 / 2.055 / 2.063 | 19.83, 1.657 / 1.675 / 1.532 / 1.550 |
| P64_s4_widened | K1 | A1 r_e>=1a | 3.707 (24.953) | 3.707 | 24.43, 3.563 / 3.563 / 3.438 / 3.439 | 23.35, 3.043 / 3.046 / 2.919 / 2.921 | 21.36, 2.178 / 2.186 / 2.052 / 2.061 | 19.83, 1.656 / 1.674 / 1.531 / 1.549 |
| P64_s4_widened | K1 | A1 r_e>=1a | 3.707 (25.000) | 3.707 | 24.47, 3.580 / 3.581 / 3.455 / 3.456 | 23.37, 3.055 / 3.057 / 2.930 / 2.932 | 21.37, 2.181 / 2.190 / 2.056 / 2.065 | 19.83, 1.658 / 1.676 / 1.532 / 1.551 |
| P64_s4_widened | K1 | A1 r_e>=1a | 3.707 (24.938) | 3.707 | 24.42, 3.557 / 3.558 / 3.432 / 3.433 | 23.34, 3.040 / 3.042 / 2.915 / 2.917 | 21.36, 2.176 / 2.185 / 2.051 / 2.059 | 19.83, 1.655 / 1.674 / 1.530 / 1.549 |
| P64_s4_widened | K2 | A2 | 4.050 (36.250) | 4.050 | 36.09, 4.038 / 4.040 / 3.910 / 3.912 | 35.65, 3.955 / 3.961 / 3.826 / 3.833 | 34.39, 3.474 / 3.499 / 3.344 / 3.368 | 33.24, 2.937 / 2.992 / 2.804 / 2.859 |
| P64_s4_widened | K2 | A2 | 4.050 (36.266) | 4.050 | 36.11, 4.039 / 4.041 / 3.911 / 3.913 | 35.66, 3.958 / 3.964 / 3.829 / 3.836 | 34.40, 3.477 / 3.502 / 3.347 / 3.372 | 33.24, 2.938 / 2.994 / 2.806 / 2.861 |
| P64_s4_widened | K2 | A2 | 4.050 (36.234) | 4.050 | 36.08, 4.038 / 4.039 / 3.909 / 3.911 | 35.63, 3.952 / 3.958 / 3.823 / 3.830 | 34.39, 3.471 / 3.496 / 3.340 / 3.365 | 33.23, 2.935 / 2.990 / 2.802 / 2.857 |
| P64_s4_widened | K2 | A2 | 4.050 (36.281) | 4.050 | 36.12, 4.040 / 4.042 / 3.912 / 3.914 | 35.67, 3.961 / 3.967 / 3.832 / 3.839 | 34.41, 3.480 / 3.505 / 3.350 / 3.375 | 33.25, 2.940 / 2.996 / 2.808 / 2.863 |
| P64_s4_widened | K2 | A2 | 4.050 (36.219) | 4.050 | 36.06, 4.036 / 4.038 / 3.908 / 3.910 | 35.62, 3.949 / 3.955 / 3.820 / 3.827 | 34.38, 3.468 / 3.493 / 3.337 / 3.362 | 33.23, 2.933 / 2.988 / 2.800 / 2.856 |
| P64_s4_widened | K3 | B | 4.239 (36.051) | 4.239 | 35.77, 4.203 / 4.205 / 4.074 / 4.076 | 35.06, 4.006 / 4.012 / 3.877 / 3.883 | 33.47, 3.366 / 3.389 / 3.234 / 3.257 | 32.21, 2.815 / 2.867 / 2.681 / 2.734 |
| P64_s4_widened | K4 | B | 4.236 (36.252) | 4.236 | 35.96, 4.221 / 4.222 / 4.093 / 4.094 | 35.21, 4.047 / 4.053 / 3.918 / 3.924 | 33.53, 3.389 / 3.412 / 3.257 / 3.281 | 32.25, 2.827 / 2.879 / 2.693 / 2.746 |
| P64_s4_widened | K5 | B | 4.235 (36.240) | 4.235 | 35.93, 4.218 / 4.220 / 4.090 / 4.091 | 35.15, 4.030 / 4.036 / 3.900 / 3.906 | 33.44, 3.359 / 3.382 / 3.227 / 3.250 | 32.14, 2.802 / 2.854 / 2.668 / 2.720 |
| P64_s4_widened | K6 | B | 4.231 (36.279) | 4.231 | 36.00, 4.220 / 4.221 / 4.091 / 4.093 | 35.28, 4.062 / 4.068 / 3.933 / 3.939 | 33.63, 3.415 / 3.439 / 3.283 / 3.307 | 32.35, 2.849 / 2.902 / 2.715 / 2.768 |
| P64_s4_widened | K7 | B | 4.227 (36.371) | 4.227 | 36.08, 4.225 / 4.226 / 4.097 / 4.098 | 35.30, 4.071 / 4.077 / 3.942 / 3.948 | 33.57, 3.402 / 3.426 / 3.271 / 3.294 | 32.27, 2.834 / 2.887 / 2.700 / 2.753 |

**Table 9-H2v4. Per member and ap_e, over the headline events of Table 9-H1v4, counted per row and per ARRIVAL CLUSTER** (rows / clusters; cluster-based counts, not counts of distinct emission events): received by the last record; the largest L_F, L_gF and L_gK; those whose L_F exceeds reviewer B's crossover estimate ln(2/ap_e²) (6.68, 5.30, 3.91, 3.10 at ap_e 0.05, 0.1, 0.2, 0.3; a breakdown scale of the continuum approximation, not a cap); arriving earlier than the stored arrival; arriving less redshifted (L_F < L_T); then over the clusters (each read on its largest-L_T row; cluster-weighted): the median of L_F/L_T (a RATIO OF LOGARITHMS, not a frequency-reduction factor) and of the arrival shift T_D(dispersive) − T_D(stored) in r_s; the largest L_F − L_T (positive = the dispersive ray more redshifted); and the largest |L_K − L_F|.

| member | ap_e | rows / clusters | received | largest L_F / L_gF / L_gK | L_F > ln(2/ap_e²) | earlier | less redshifted (L_F < L_T) | median L_F/L_T (clusters) | median ΔT_D (r_s, clusters) | largest L_F − L_T | largest abs(L_K − L_F) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | 0.05 | 15 / 7 | 15 / 7 | 6.647 / 6.527 / 6.529 | 0 / 0 | 15 / 7 | 15 / 7 | 0.797 | -2.68 | -1.593 | 0.002 |
| P64_s4 | 0.1 | 15 / 7 | 15 / 7 | 5.513 / 5.391 / 5.398 | 4 / 4 | 15 / 7 | 15 / 7 | 0.661 | -4.71 | -2.707 | 0.007 |
| P64_s4 | 0.2 | 15 / 7 | 15 / 7 | 4.251 / 4.126 / 4.152 | 4 / 4 | 15 / 7 | 15 / 7 | 0.509 | -7.27 | -3.965 | 0.028 |
| P64_s4 | 0.3 | 15 / 7 | 15 / 7 | 3.294 / 3.167 / 3.225 | 9 / 5 | 15 / 7 | 15 / 7 | 0.395 | -8.78 | -4.920 | 0.063 |
| P64_s2 | 0.05 | 15 / 7 | 15 / 7 | 6.722 / 6.609 / 6.611 | 3 / 3 | 15 / 7 | 15 / 7 | 0.648 | -4.35 | -2.203 | 0.002 |
| P64_s2 | 0.1 | 15 / 7 | 15 / 7 | 5.370 / 5.255 / 5.261 | 3 / 3 | 15 / 7 | 15 / 7 | 0.512 | -6.05 | -3.544 | 0.007 |
| P64_s2 | 0.2 | 15 / 7 | 15 / 7 | 4.082 / 3.965 / 3.991 | 3 / 3 | 15 / 7 | 15 / 7 | 0.381 | -7.89 | -4.828 | 0.028 |
| P64_s2 | 0.3 | 15 / 7 | 15 / 7 | 3.257 / 3.140 / 3.198 | 3 / 3 | 15 / 7 | 15 / 7 | 0.306 | -9.10 | -5.653 | 0.062 |
| S4_128_s8 | 0.05 | 15 / 7 | 15 / 7 | 6.491 / 6.367 / 6.369 | 0 / 0 | 15 / 7 | 15 / 7 | 0.896 | -1.64 | -0.030 | 0.002 |
| S4_128_s8 | 0.1 | 15 / 7 | 15 / 7 | 5.449 / 5.323 / 5.329 | 8 / 4 | 15 / 7 | 15 / 7 | 0.752 | -3.47 | -0.276 | 0.007 |
| S4_128_s8 | 0.2 | 15 / 7 | 15 / 7 | 4.663 / 4.536 / 4.562 | 10 / 6 | 15 / 7 | 15 / 7 | 0.644 | -6.39 | -0.966 | 0.026 |
| S4_128_s8 | 0.3 | 15 / 7 | 15 / 7 | 3.847 / 3.718 / 3.777 | 3 / 3 | 15 / 7 | 15 / 7 | 0.491 | -8.66 | -1.668 | 0.058 |
| P64_s4_tau4 | 0.05 | 15 / 7 | 15 / 7 | 6.053 / 5.926 / 5.928 | 0 / 0 | 15 / 7 | 15 / 7 | 0.907 | -1.66 | -0.441 | 0.002 |
| P64_s4_tau4 | 0.1 | 15 / 7 | 15 / 7 | 5.499 / 5.370 / 5.377 | 2 / 2 | 15 / 7 | 15 / 7 | 0.791 | -4.20 | -0.995 | 0.007 |
| P64_s4_tau4 | 0.2 | 15 / 7 | 15 / 7 | 4.566 / 4.436 / 4.462 | 2 / 2 | 15 / 7 | 15 / 7 | 0.547 | -7.11 | -1.948 | 0.026 |
| P64_s4_tau4 | 0.3 | 15 / 7 | 15 / 7 | 3.462 / 3.328 / 3.386 | 2 / 2 | 15 / 7 | 15 / 7 | 0.432 | -8.49 | -3.053 | 0.058 |
| W2 | 0.05 | 15 / 7 | 15 / 7 | 4.869 / 4.732 / 4.734 | 0 / 0 | 15 / 7 | 14 / 6 | 0.988 | -0.36 | 0.053 | 0.002 |
| W2 | 0.1 | 15 / 7 | 15 / 7 | 4.752 / 4.615 / 4.621 | 0 / 0 | 15 / 7 | 15 / 7 | 0.913 | -1.33 | -0.064 | 0.006 |
| W2 | 0.2 | 15 / 7 | 15 / 7 | 3.759 / 3.619 / 3.644 | 0 / 0 | 15 / 7 | 15 / 7 | 0.718 | -3.30 | -0.975 | 0.025 |
| W2 | 0.3 | 15 / 7 | 15 / 7 | 3.019 / 2.876 / 2.931 | 0 / 0 | 15 / 7 | 15 / 7 | 0.578 | -4.57 | -1.663 | 0.055 |
| P64_s8 | 0.05 | 15 / 7 | 15 / 7 | 3.702 / 3.576 / 3.577 | 0 / 0 | 15 / 7 | 13 / 5 | 0.997 | -0.15 | 0.037 | 0.002 |
| P64_s8 | 0.1 | 15 / 7 | 15 / 7 | 3.697 / 3.571 / 3.577 | 0 / 0 | 15 / 7 | 14 / 6 | 0.974 | -0.57 | 0.069 | 0.006 |
| P64_s8 | 0.2 | 15 / 7 | 15 / 7 | 3.269 / 3.143 / 3.165 | 0 / 0 | 15 / 7 | 15 / 7 | 0.848 | -1.83 | -0.251 | 0.024 |
| P64_s8 | 0.3 | 15 / 7 | 15 / 7 | 2.682 / 2.556 / 2.606 | 0 / 0 | 15 / 7 | 15 / 7 | 0.705 | -3.11 | -0.594 | 0.053 |
| S4_128_s16 | 0.05 | 15 / 7 | 15 / 7 | 3.653 / 3.526 / 3.527 | 0 / 0 | 15 / 7 | 14 / 6 | 0.993 | -0.15 | 0.005 | 0.001 |
| S4_128_s16 | 0.1 | 15 / 7 | 15 / 7 | 3.589 / 3.462 / 3.468 | 0 / 0 | 15 / 7 | 15 / 7 | 0.963 | -0.56 | -0.059 | 0.006 |
| S4_128_s16 | 0.2 | 15 / 7 | 15 / 7 | 3.116 / 2.989 / 3.011 | 0 / 0 | 15 / 7 | 15 / 7 | 0.828 | -1.80 | -0.532 | 0.022 |
| S4_128_s16 | 0.3 | 15 / 7 | 15 / 7 | 2.581 / 2.454 / 2.503 | 0 / 0 | 15 / 7 | 15 / 7 | 0.690 | -3.08 | -1.013 | 0.050 |
| P64_s4_heldsrc | 0.05 | 15 / 7 | 15 / 7 | 3.810 / 3.691 / 3.692 | 0 / 0 | 15 / 7 | 12 / 4 | 0.999 | -0.13 | 0.061 | 0.002 |
| P64_s4_heldsrc | 0.1 | 15 / 7 | 15 / 7 | 3.809 / 3.689 / 3.695 | 0 / 0 | 15 / 7 | 13 / 5 | 0.974 | -0.55 | 0.155 | 0.006 |
| P64_s4_heldsrc | 0.2 | 15 / 7 | 15 / 7 | 3.254 / 3.137 / 3.159 | 0 / 0 | 15 / 7 | 15 / 7 | 0.833 | -1.87 | -0.184 | 0.025 |
| P64_s4_heldsrc | 0.3 | 15 / 7 | 15 / 7 | 2.682 / 2.564 / 2.619 | 0 / 0 | 15 / 7 | 15 / 7 | 0.677 | -3.08 | -0.447 | 0.055 |
| P64_s4_widened | 0.05 | 15 / 7 | 15 / 7 | 4.225 / 4.097 / 4.098 | 0 / 0 | 15 / 7 | 15 / 7 | 0.996 | -0.29 | -0.002 | 0.002 |
| P64_s4_widened | 0.1 | 15 / 7 | 15 / 7 | 4.071 / 3.942 / 3.948 | 0 / 0 | 15 / 7 | 15 / 7 | 0.956 | -1.04 | -0.089 | 0.006 |
| P64_s4_widened | 0.2 | 15 / 7 | 15 / 7 | 3.480 / 3.350 / 3.375 | 0 / 0 | 15 / 7 | 15 / 7 | 0.800 | -2.72 | -0.570 | 0.025 |
| P64_s4_widened | 0.3 | 15 / 7 | 15 / 7 | 2.940 / 2.808 / 2.863 | 0 / 0 | 15 / 7 | 15 / 7 | 0.667 | -4.01 | -1.110 | 0.055 |

Scope, computed: median L_F/L_T over arrival clusters, per member and ap_e: 0.306 to 0.907 on P64_s4, P64_s2, P64_s4_tau4, S4_128_s8; 0.578 to 0.999 on W2, P64_s8, S4_128_s16, P64_s4_heldsrc, P64_s4_widened. Crossover exceedances on those four members: 56 (row, ap_e) pairs, 44 (arrival cluster, ap_e) pairs, 16 arrival clusters (over all members: 56 pairs of rows, 16 clusters). Every received dispersive ray arrives earlier than the stored arrival: yes. On the four members every received dispersive ray is less redshifted: yes. Members with a received ray not less redshifted or not earlier (Table 9-H3v4): P64_s4_heldsrc (largest L_F − L_T 0.155), P64_s8 (largest L_F − L_T 0.069), S4_128_s16 (largest L_F − L_T 0.005), W2 (largest L_F − L_T 0.053).

Emission-time spread of each A arrival cluster's rows (a; the rows' emission times differ by this much): P64_s4 A1 r_e>=1a.5 1.48e-04, A2.6 3.33e-04; P64_s2 A1 r_e>=1a.5 8.07e-07, A2.6 2.40e-06; S4_128_s8 A1 r_e>=1a.5 1.48e-01, A2.7 2.57e-03; P64_s4_tau4 A1 r_e>=1a.5 1.18e-02, A2.7 5.69e-03; W2 A1 r_e>=1a.5 1.46e-01, A2.7 9.91e-02; P64_s8 A1 r_e>=1a.5 1.97e-01, A2.9 2.02e-01; S4_128_s16 A1 r_e>=1a.5 1.72e-01, A2.8 3.90e-01; P64_s4_heldsrc A1 r_e>=1a.5 1.56e-01, A2.6 2.51e-01; P64_s4_widened A1 r_e>=1a.5 1.21e-01, A2.8 8.78e-02.

Continuum reproduction from the emission side (forward DOP853 from the event against the stored L_T), max abs difference (rows checked): P64_s4 6.90e-08 (13); P64_s2 2.81e-05 (13); S4_128_s8 5.49e-09 (15); P64_s4_tau4 1.13e-08 (13); W2 6.70e-08 (15); P64_s8 2.69e-08 (15); S4_128_s16 3.07e-09 (15); P64_s4_heldsrc 1.02e-07 (15); P64_s4_widened 1.84e-08 (15).

**Table 9-H3v4. The exceptions** (rz_v4_dispev_tables_v3.py): every arrival cluster and ap_e whose dispersive ray is received but NOT less redshifted than the stored continuum value (L_F ≥ L_T), and every one NOT earlier (T_D(dispersive) ≥ T_D(stored)); then per member the largest dispersive L_F at ap_e ≤ 0.1 against the largest stored L_T over the member's tested events (A1 r_e ≥ 1 a, A2, B).

| member | cluster | class | stored L_T (T_D/r_s) | ap_e | L_F | L_F − L_T | dispersive T_D/r_s | which |
|---|---|---|---|---|---|---|---|---|
| W2 | K4 | B | 4.816 (37.203) | 0.05 | 4.869 | 0.053 | 36.840 | not less redshifted |
| P64_s8 | K4 | B | 3.683 (25.894) | 0.05 | 3.702 | 0.018 | 25.733 | not less redshifted |
| P64_s8 | K6 | B | 3.629 (26.110) | 0.05 | 3.666 | 0.037 | 25.968 | not less redshifted |
| P64_s8 | K6 | B | 3.629 (26.110) | 0.1 | 3.697 | 0.069 | 25.526 | not less redshifted |
| S4_128_s16 | K5 | B | 3.647 (24.383) | 0.05 | 3.653 | 0.005 | 24.244 | not less redshifted |
| P64_s4_heldsrc | K3 | B | 3.803 (24.520) | 0.05 | 3.810 | 0.007 | 24.360 | not less redshifted |
| P64_s4_heldsrc | K5 | B | 3.698 (24.813) | 0.05 | 3.759 | 0.061 | 24.647 | not less redshifted |
| P64_s4_heldsrc | K7 | B | 3.654 (24.963) | 0.05 | 3.699 | 0.044 | 24.835 | not less redshifted |
| P64_s4_heldsrc | K5 | B | 3.698 (24.813) | 0.1 | 3.746 | 0.048 | 24.122 | not less redshifted |
| P64_s4_heldsrc | K7 | B | 3.654 (24.963) | 0.1 | 3.809 | 0.155 | 24.414 | not less redshifted |

| member | largest stored L_T over the tested events (class) | largest dispersive L_F at ap_e 0.05 / 0.1 (class) | dispersive above continuum? |
|---|---|---|---|
| P64_s4 | 10.456 (A1 r_e>=1a) | 6.647 (B) / 5.513 (B) | no |
| P64_s2 | 15.676 (A1 r_e>=1a) | 6.722 (B) / 5.370 (B) | no |
| S4_128_s8 | 7.754 (A2) | 6.491 (B) / 5.449 (B) | no |
| P64_s4_tau4 | 7.184 (A2) | 6.053 (B) / 5.499 (B) | no |
| W2 | 4.877 (B) | 4.869 (B) / 4.752 (B) | no |
| P64_s8 | 3.708 (B) | 3.702 (B) / 3.697 (B) | no |
| S4_128_s16 | 3.654 (B) | 3.653 (B) / 3.589 (B) | no |
| P64_s4_heldsrc | 3.803 (B) | 3.810 (B) / 3.809 (B) | yes |
| P64_s4_widened | 4.239 (B) | 4.225 (B) / 4.071 (B) | no |

### 9.1 v2's pass on the forward population (archival: pasted from REDSHIFT_received_v3.1 sec. 9.1, which pasted REDSHIFT_received_v2 sec. 9; superseded sentences marked [v3: …] and [v4: …]; it uses the forward launch population on v1's passes, whose A2 clocks are released one step early, and keeps 'certified' for 'inside the solver-certified record interval')

Producing scripts (deliveries/route2/REDSHIFT_received_v1_scripts/, where they ran): code/rz_disp_tables_v4.py (md5 3b82e6182e6fd7f0cfe5105b11bb4d86), table file out/disp_tables_v1/disp_tables_v4.md (md5 5cfc72c685ddfb72f624453c435d54f4); code/rz_disp_ctrl_tables_v1.py (md5 5ac01d2676a1ea2453d7273bc8c2411f), out/disp_tables_v1/disp_ctrl_v1.md (md5 e0af37c54b43e22f5483efe6f9b43993); code/rz_disp_heavy_tables_v3.py (md5 f6c4c21ed691448fb254810a996f5b40), out/disp_tables_v1/disp_heavy_v1.md (md5 0dbde59606e22ad9b3a9a2256f32bffb); the rays: code/rz_disp_v1.py (md5 b93e7fc2f7939548d9a3a20c666c62a3) and code/rz_disp_v2.py (md5 c9ff1f2d8ddc9f9c86b34462cc03f7d2). In this pasted text 'certified' (as in 'largest certified ln(1/Z)', 'received by T_cert') means received inside the member's solver-certified record interval, as everywhere in v3.

Moved verbatim from v1 (where it was added at 03:23:00 and 03:52:24 UTC, after the review copy): REDSHIFT_received_v1_scripts/parts_box/p15_disp44_v2.md (md5 1a378163b2bdc6304362e691b9ee3a15) and p16_disp44_heavy.md (md5 a992f9b356aee092b44db50a0e4bd79a); their tables are those folders' out/disp_tables_v1/disp_tables_v4.md, disp_ctrl_v1.md and disp_heavy_v1.md, by rz_disp_tables_v4.py, rz_disp_ctrl_tables_v1.py and rz_disp_heavy_tables_v3.py. Heading levels are shifted down by one; nothing else is changed.

### 4.4 reviewer B's dispersive pass (AstraAPI_redshift_v1 sec. 4.4), added for inbox #52

**What was run.** Rays of the local positive-band chain symbol of reviewer B 4.4, Ω(T, r, p) = v q(p) + N ε_j(p) with q = sin(2ap)/(2a) and ε_j = √(m² + sin²(ap)/a² + j²/r²), integrated with reviewer B's Hamilton equations (dr/dT = v cos(2ap) + N q/ε_j, dp/dT = −v′q − N′ε_j + N j²/(ε_j r³)) for the massless radial channel (m = 0, j = 0). Fields: each member's back-averaged site fields, linear in time and cubic in r (the continuum pass's main variant, sec. 2.1). Launch events: the continuum pass's own (rays_v4: comoving clocks A1 and A2, root-neighbourhood launches B, dust R_50 and R_90 as C), each at the emitted lattice momenta ap_e = 0.05, 0.1, 0.2, 0.3. Integration: RK4 in (r, ln ap) on the record-aligned grid, h = 0.5 a, arrivals located by the continuum pass's Hermite crossing. At each receiver the pass records the coordinate frequency Ω_D = Ω(t_D, r_D, p_D) (reviewer A's Killing frequency once the ray is in the stationary exterior), the largest ap along the path, and whether ap passed π/2.

Emitted frequency, in the two conventions reviewer B's caution separates (4.4: "do not equate the bare band energy ε_j with the frequency measured along every moving emitter"): (F) ε_j(p_e), reviewer A's comoving energy; (K) ω_K = (Ω_e − v_e p_e)/N_e, the phase rate of exp(i(pr − ΩT)) along the normal-comoving worldline per unit proper time. Z_F = Ω_D/ε_e and Z_K = Ω_D/ω_K both tend to the continuum Z as ap_e → 0. Material emitters (C) carry the continuum pass's Doppler factor 1/D_e in both. [v3: not the finite-dispersion phase-frequency normalization of a moving emitter (reviewer B rzchk3 sec. 7.5); C is a convention-specific diagnostic here and is dropped from v3's finite-frequency comparisons.] Every comparison with the continuum is launch by launch: the same launch event, the 4 r_s receiver, and the member's certified extent (a launch counts as received when its arrival is inside it).

Members (14, every member of the continuum pass's main list except the two continuations): P64_s4_B2 and P64_s4_ext200 were not run (they equal P64_s4 through 40 r_s, and past 40 they are uncertified). The heavy channel (m a 0.2, j = 1) was not run: see the self-tests. [v3: superseded; the addendum below ran it with the (r, p) integrator.]

| member | launches (all four classes) | certified to T/r_s | records to T/r_s | run seconds | continuum npz md5 (first 8) | script md5 (first 8) |
|---|---|---|---|---|---|---|
| P64_s4 | 10946 | 40.0000 | 40.0000 | 205 | 77df1e78 | b93e7fc2 |
| P64_s2 | 11102 | 40.0000 | 40.0000 | 217 | a18a9efd | b93e7fc2 |
| P64_s8 | 10201 | 40.0000 | 40.0000 | 203 | cc6a3570 | b93e7fc2 |
| P64_s4_tau4 | 20807 | 40.0000 | 40.0000 | 367 | 8392285b | b93e7fc2 |
| W2 | 10614 | 40.0000 | 40.0000 | 221 | 28ff45c6 | b93e7fc2 |
| S4_128_s8 | 22294 | 40.0000 | 40.0000 | 843 | 4c6c1270 | b93e7fc2 |
| S4_128_s16 | 20797 | 40.0000 | 40.0000 | 798 | b37e1e40 | b93e7fc2 |
| P64_s4_tau2 | 21049 | 12.7500 | 21.5000 | 229 | 27651bd0 | b93e7fc2 |
| ISO_64_s4 | 11003 | 13.5000 | 40.0000 | 212 | ae80b067 | b93e7fc2 |
| ISO_128_s4 | 5716 | 12.0000 | 12.0625 | 88 | 97b50869 | b93e7fc2 |
| S4_128_s4 | 9719 | 3.5625 | 19.0625 | 241 | a4fdcf0d | b93e7fc2 |
| ISO_128_s4_plain | 6907 | 3.2500 | 14.3750 | 145 | 16c89215 | b93e7fc2 |
| M3c_s4_96 | 7361 | 22.0833 | 22.0833 | 141 | b77f18de | b93e7fc2 |
| P64_s4_newton | 2840 | 12.5000 | 12.5000 | 26 | a2e35319 | b93e7fc2 |

**Self-tests** (rz_selftest_disp_v1.py on box CPU; the log's lines pasted from logs/selftest_disp_v1.log):

```
start 2026-10-22T02:51:24Z
PG 0.005 {'max_rel_dOmega': 1.3183293567919918e-08, 'max_abs_dlnZ_vs_stationary_map': 1.318329045929545e-08, 'max_abs_dlnZ_vs_continuum': 0.0005060578357167778}
PG 0.1 {'max_rel_dOmega': 1.0918175252783158e-08, 'max_abs_dlnZ_vs_stationary_map': 1.0918175252783158e-08, 'max_abs_dlnZ_vs_continuum': 0.18423709799647714}
PG 0.3 {'max_rel_dOmega': 4.503144301892803e-09, 'max_abs_dlnZ_vs_stationary_map': 4.503143635758988e-09, 'max_abs_dlnZ_vs_continuum': 1.032686075910426}
PG heavy {'max_rel_dOmega': nan, 'received': 0}
drift 0.1 {'max_abs_dtD': 6.17658986357128e-06, 'max_rel_dpD': 4.113835450070269e-08}
drift 0.3 {'max_abs_dtD': 4.109865813006763e-06, 'max_rel_dpD': 2.3302860308049844e-08}
end rc=0 2026-10-22T02:51:35Z
```

Reading of the log. (1) On the stationary Painlevé-Gullstrand exterior (N = 1, v = −√(r_s/r), r_s/a 64), Ω is conserved along every ray to 1.32e-08 (relative, worst of ap_e 0.005, 0.1, 0.3), and the received Z_F equals reviewer A's stationary map N_e + v_e cos(ap_e) to the same accuracy. At ap_e = 0.005 it is within 5.06e-04 in ln Z of the continuum 1 − √(r_s/r_e) (emitters at 1.05 to 3 r_s); at ap_e 0.1 and 0.3 the stationary map itself differs from the continuum by up to 0.184 and 1.033 in ln Z over the same emitters, which is the dispersion being measured, not an error. (2) On a drifting-charge exterior the arrival time at 4 r_s agrees with scipy's RK45 (rtol 1e-11) to 6.18e-06 a and the arrival momentum to 4.11e-08 (relative). (3) The heavy channel FAILED its test: on the stationary exterior no ray was received and the integrator overflowed. Heavy rays are bound where Ω < m; they turn at p = 0, which the variable ln ap cannot follow. [v3: superseded by the addendum's (r, p) integrator; 'turn at p = 0' reads 'change the sign of p', a momentum sign change, not a located radial turning point (reviewer B rzchk3 sec. 8.2).] The heavy dispersive pass needs an integrator in p itself; it is listed with sec. 8's open items.

#### 4.4 tables (rz_disp_tables_v4.py; out/disp_tables_v1/disp_tables_v4.md)

**Table 4.4-1. Largest certified ln(1/Z) at 4 r_s, dispersive against continuum, launch by launch.** Per member and emitted lattice momentum ap_e: launches; received by T_cert in the dispersive pass / in the continuum pass / both / dispersive only / continuum only; then per class the largest certified ln(1/Z): the dispersive pass (convention F, Z_F = Omega_D/eps(p_e)) over all its received launches; the dispersive pass over the launches received in BOTH passes; the continuum over the same launches (like for like); last column the largest |ln Z_K − ln Z_F| over the dispersive pass's received launches (reviewer B's emitter convention K against reviewer A's F). A = comoving clocks (A1 and A2 together; forward launches, emitted every tau/4, so the continuum A column here can sit below the backward-route headline of sec. 2, which samples every a of arrival); B = root-neighbourhood launches; C = dust R_50 and R_90.

| member | ap_e | launches | received cert: disp / cont / both / disp only / cont only | A: F all / F both / cont both | B: F all / F both / cont both | C: F all / F both / cont both | max abs(ln Z_K − ln Z_F) |
|---|---|---|---|---|---|---|---|
| P64_s4 | 0.05 | 10946 | 6064 / 5460 / 5460 / 604 / 0 | 7.356 / 5.582 / 7.311 | 8.712 / 6.647 / 9.128 | 4.176 / 4.176 / 4.283 | 0.002 |
| P64_s4 | 0.1 | 10946 | 6103 / 5460 / 5460 / 643 / 0 | 6.521 / 4.787 / 7.311 | 9.018 / 5.513 / 9.128 | 4.080 / 4.080 / 4.283 | 0.009 |
| P64_s4 | 0.2 | 10946 | 6993 / 5460 / 5460 / 1533 / 0 | 7.343 / 3.895 / 7.311 | 8.590 / 4.418 / 9.128 | 5.575 / 3.991 / 4.283 | 0.037 |
| P64_s4 | 0.3 | 10946 | 7668 / 5460 / 5460 / 2208 / 0 | 6.017 / 3.315 / 7.311 | 10.078 / 3.686 / 9.128 | 3.465 / 3.426 / 4.283 | 0.081 |
| P64_s2 | 0.05 | 11102 | 6691 / 6037 / 6037 / 654 / 0 | 7.934 / 6.196 / 6.933 | 9.003 / 6.722 / 10.033 | 3.080 / 3.080 / 3.132 | 0.003 |
| P64_s2 | 0.1 | 11102 | 6788 / 6037 / 6037 / 751 / 0 | 7.438 / 5.238 / 6.933 | 9.287 / 5.382 / 10.033 | 5.014 / 2.935 / 3.132 | 0.011 |
| P64_s2 | 0.2 | 11102 | 7151 / 6037 / 6037 / 1114 / 0 | 7.169 / 4.021 / 6.933 | 7.432 / 4.082 / 10.033 | 3.315 / 2.784 / 3.132 | 0.042 |
| P64_s2 | 0.3 | 11102 | 7506 / 6037 / 6037 / 1469 / 0 | 10.467 / 3.215 / 6.933 | 8.263 / 3.397 / 10.033 | 4.060 / 2.628 / 3.132 | 0.093 |
| P64_s8 | 0.05 | 10201 | 6921 / 6902 / 6902 / 19 / 0 | 3.209 / 3.209 / 3.216 | 3.702 / 3.702 / 3.708 | 3.075 / 3.075 / 3.099 | 0.002 |
| P64_s8 | 0.1 | 10201 | 6977 / 6902 / 6902 / 75 / 0 | 3.192 / 3.192 / 3.216 | 3.697 / 3.697 / 3.708 | 3.147 / 3.147 / 3.099 | 0.007 |
| P64_s8 | 0.2 | 10201 | 7182 / 6902 / 6902 / 280 / 0 | 3.032 / 3.032 / 3.216 | 3.663 / 3.663 / 3.708 | 3.337 / 3.337 / 3.099 | 0.029 |
| P64_s8 | 0.3 | 10201 | 7816 / 6902 / 6902 / 914 / 0 | 2.969 / 2.969 / 3.216 | 3.607 / 3.607 / 3.708 | 3.310 / 3.310 / 3.099 | 0.064 |
| P64_s4_tau4 | 0.05 | 20807 | 12621 / 11681 / 11681 / 940 / 0 | 5.976 / 5.728 / 6.490 | 6.775 / 6.053 / 6.596 | 4.572 / 4.572 / 4.835 | 0.002 |
| P64_s4_tau4 | 0.1 | 20807 | 12858 / 11681 / 11681 / 1177 / 0 | 5.525 / 4.866 / 6.490 | 6.858 / 5.499 / 6.596 | 5.900 / 4.238 / 4.835 | 0.009 |
| P64_s4_tau4 | 0.2 | 20807 | 13637 / 11681 / 11681 / 1956 / 0 | 5.250 / 4.244 / 6.490 | 6.870 / 4.566 / 6.596 | 4.565 / 4.333 / 4.835 | 0.037 |
| P64_s4_tau4 | 0.3 | 20807 | 14702 / 11681 / 11681 / 3021 / 0 | 4.928 / 3.719 / 6.490 | 7.033 / 4.194 / 6.596 | 5.446 / 4.716 / 4.835 | 0.081 |
| W2 | 0.05 | 10614 | 7450 / 7230 / 7230 / 220 / 0 | 4.520 / 4.520 / 4.521 | 4.869 / 4.869 / 4.877 | 4.466 / 4.466 / 4.461 | 0.002 |
| W2 | 0.1 | 10614 | 7589 / 7230 / 7230 / 359 / 0 | 4.517 / 4.517 / 4.521 | 4.869 / 4.869 / 4.877 | 4.460 / 4.460 / 4.461 | 0.009 |
| W2 | 0.2 | 10614 | 7685 / 7230 / 7230 / 455 / 0 | 4.504 / 4.504 / 4.521 | 4.816 / 4.816 / 4.877 | 4.211 / 4.211 / 4.461 | 0.035 |
| W2 | 0.3 | 10614 | 7774 / 7230 / 7230 / 544 / 0 | 4.480 / 4.480 / 4.521 | 4.775 / 4.580 / 4.877 | 4.158 / 4.158 / 4.461 | 0.077 |
| S4_128_s8 | 0.05 | 22294 | 13225 / 13060 / 13060 / 165 / 0 | 6.031 / 4.851 / 5.483 | 7.452 / 6.491 / 7.723 | 4.659 / 4.659 / 4.685 | 0.002 |
| S4_128_s8 | 0.1 | 22294 | 13423 / 13060 / 13060 / 363 / 0 | 5.710 / 4.477 / 5.483 | 7.610 / 5.449 / 7.723 | 5.221 / 4.620 / 4.685 | 0.010 |
| S4_128_s8 | 0.2 | 22294 | 14758 / 13060 / 13060 / 1698 / 0 | 5.431 / 3.892 / 5.483 | 7.272 / 4.663 / 7.723 | 5.065 / 4.643 / 4.685 | 0.039 |
| S4_128_s8 | 0.3 | 22294 | 15915 / 13060 / 13060 / 2855 / 0 | 4.714 / 3.436 / 5.483 | 7.413 / 3.968 / 7.723 | 4.920 / 4.920 / 4.685 | 0.085 |
| S4_128_s16 | 0.05 | 20797 | 15984 / 15961 / 15961 / 23 / 0 | 3.277 / 3.277 / 3.273 | 3.653 / 3.653 / 3.654 | 3.595 / 3.595 / 3.622 | 0.002 |
| S4_128_s16 | 0.1 | 20797 | 16043 / 15961 / 15961 / 82 / 0 | 3.274 / 3.274 / 3.273 | 3.649 / 3.649 / 3.654 | 3.513 / 3.513 / 3.622 | 0.007 |
| S4_128_s16 | 0.2 | 20797 | 16204 / 15961 / 15961 / 243 / 0 | 3.224 / 3.224 / 3.273 | 3.634 / 3.634 / 3.654 | 3.619 / 3.619 / 3.622 | 0.030 |
| S4_128_s16 | 0.3 | 20797 | 16404 / 15961 / 15961 / 443 / 0 | 3.036 / 3.036 / 3.273 | 3.604 / 3.604 / 3.654 | 3.427 / 3.427 / 3.622 | 0.065 |
| P64_s4_tau2 | 0.05 | 21049 | 5239 / 5225 / 5225 / 14 / 0 | 0.744 / 0.743 / 0.745 | - / - / - | 0.623 / 0.620 / 0.622 | 9.70e-04 |
| P64_s4_tau2 | 0.1 | 21049 | 5280 / 5225 / 5225 / 55 / 0 | 0.740 / 0.737 / 0.745 | - / - / - | 0.620 / 0.614 / 0.622 | 0.004 |
| P64_s4_tau2 | 0.2 | 21049 | 5437 / 5225 / 5225 / 212 / 0 | 0.726 / 0.715 / 0.745 | - / - / - | 0.604 / 0.592 / 0.622 | 0.015 |
| P64_s4_tau2 | 0.3 | 21049 | 5653 / 5225 / 5225 / 428 / 0 | 0.702 / 0.683 / 0.745 | - / - / - | 0.584 / 0.557 / 0.622 | 0.035 |
| ISO_64_s4 | 0.05 | 11003 | 1446 / 1442 / 1442 / 4 / 0 | 0.717 / 0.717 / 0.720 | 0.764 / 0.764 / 0.766 | 0.627 / 0.627 / 0.629 | 9.90e-04 |
| ISO_64_s4 | 0.1 | 11003 | 1456 / 1442 / 1442 / 14 / 0 | 0.713 / 0.711 / 0.720 | 0.760 / 0.760 / 0.766 | 0.621 / 0.621 / 0.629 | 0.004 |
| ISO_64_s4 | 0.2 | 11003 | 1502 / 1442 / 1442 / 60 / 0 | 0.699 / 0.691 / 0.720 | 0.744 / 0.744 / 0.766 | 0.611 / 0.598 / 0.629 | 0.016 |
| ISO_64_s4 | 0.3 | 11003 | 1561 / 1442 / 1442 / 119 / 0 | 0.676 / 0.666 / 0.720 | 0.717 / 0.717 / 0.766 | 0.586 / 0.567 / 0.629 | 0.035 |
| ISO_128_s4 | 0.05 | 5716 | 2442 / 2435 / 2435 / 7 / 0 | 0.702 / 0.702 / 0.703 | - / - / - | 0.592 / 0.592 / 0.593 | 9.86e-04 |
| ISO_128_s4 | 0.1 | 5716 | 2463 / 2435 / 2435 / 28 / 0 | 0.698 / 0.698 / 0.703 | - / - / - | 0.589 / 0.589 / 0.593 | 0.004 |
| ISO_128_s4 | 0.2 | 5716 | 2542 / 2435 / 2435 / 107 / 0 | 0.683 / 0.683 / 0.703 | 0.706 / - / - | 0.576 / 0.575 / 0.593 | 0.016 |
| ISO_128_s4 | 0.3 | 5716 | 2651 / 2435 / 2435 / 216 / 0 | 0.660 / 0.660 / 0.703 | 0.687 / - / - | 0.555 / 0.543 / 0.593 | 0.035 |
| S4_128_s4 | 0.05 | 9719 | 224 / 224 / 224 / 0 / 0 | 0.700 / 0.700 / 0.701 | - / - / - | 0.239 / 0.239 / 0.240 | 8.32e-04 |
| S4_128_s4 | 0.1 | 9719 | 226 / 224 / 224 / 2 / 0 | 0.696 / 0.696 / 0.701 | - / - / - | 0.235 / 0.235 / 0.240 | 0.003 |
| S4_128_s4 | 0.2 | 9719 | 231 / 224 / 224 / 7 / 0 | 0.682 / 0.682 / 0.701 | - / - / - | 0.222 / 0.220 / 0.240 | 0.013 |
| S4_128_s4 | 0.3 | 9719 | 241 / 224 / 224 / 17 / 0 | 0.658 / 0.658 / 0.701 | - / - / - | 0.200 / 0.196 / 0.240 | 0.029 |
| ISO_128_s4_plain | 0.05 | 6907 | 196 / 196 / 196 / 0 / 0 | 0.702 / 0.702 / 0.703 | - / - / - | 0.233 / 0.233 / 0.234 | 8.31e-04 |
| ISO_128_s4_plain | 0.1 | 6907 | 197 / 196 / 196 / 1 / 0 | 0.698 / 0.698 / 0.703 | - / - / - | 0.229 / 0.229 / 0.234 | 0.003 |
| ISO_128_s4_plain | 0.2 | 6907 | 202 / 196 / 196 / 6 / 0 | 0.683 / 0.683 / 0.703 | - / - / - | 0.214 / 0.214 / 0.234 | 0.013 |
| ISO_128_s4_plain | 0.3 | 6907 | 211 / 196 / 196 / 15 / 0 | 0.660 / 0.660 / 0.703 | - / - / - | 0.192 / 0.190 / 0.234 | 0.029 |
| M3c_s4_96 | 0.05 | 7361 | 4325 / 4315 / 4315 / 10 / 0 | 0.920 / 0.920 / 0.922 | - / - / - | 0.719 / 0.719 / 0.721 | 0.001 |
| M3c_s4_96 | 0.1 | 7361 | 4359 / 4315 / 4315 / 44 / 0 | 0.913 / 0.913 / 0.922 | - / - / - | 0.714 / 0.714 / 0.721 | 0.004 |
| M3c_s4_96 | 0.2 | 7361 | 4475 / 4315 / 4315 / 160 / 0 | 0.889 / 0.887 / 0.922 | - / - / - | 0.699 / 0.692 / 0.721 | 0.016 |
| M3c_s4_96 | 0.3 | 7361 | 4630 / 4315 / 4315 / 315 / 0 | 0.854 / 0.846 / 0.922 | - / - / - | 0.669 / 0.655 / 0.721 | 0.036 |
| P64_s4_newton | 0.05 | 2840 | 1280 / 1277 / 1277 / 3 / 0 | 0.723 / 0.723 / 0.724 | - / - / - | 0.588 / 0.588 / 0.590 | 9.59e-04 |
| P64_s4_newton | 0.1 | 2840 | 1291 / 1277 / 1277 / 14 / 0 | 0.719 / 0.718 / 0.724 | - / - / - | 0.583 / 0.583 / 0.590 | 0.004 |
| P64_s4_newton | 0.2 | 2840 | 1329 / 1277 / 1277 / 52 / 0 | 0.704 / 0.700 / 0.724 | - / - / - | 0.572 / 0.563 / 0.590 | 0.015 |
| P64_s4_newton | 0.3 | 2840 | 1382 / 1277 / 1277 / 105 / 0 | 0.681 / 0.669 / 0.724 | - / - / - | 0.549 / 0.530 / 0.590 | 0.034 |

**Table 4.4-2. ln Z_disp − ln Z_cont at 4 r_s by the continuum's ln(1/Z) band, launches received in both passes by T_cert, pooled over the members of Table 4.4-1.** Per band and ap_e: count; median of Δ_F = ln Z_F − ln Z_cont (positive = the dispersive ray arrives less redshifted); median |Δ_F|; 95th percentile and maximum of |Δ_F|; median of Δ_K = ln Z_K − ln Z_cont. reviewer B's crossover estimate Z ≈ ½(2aΩ_D)^{2/3} with Ω_D ≈ Z·ap_e solves to Z_c = (ap_e)²/2, i.e. ln(1/Z_c) = ln(2/ap_e²) = 6.68, 5.30, 3.91, 3.10 for ap_e = 0.05, 0.1, 0.2, 0.3.

| ap_e | continuum ln(1/Z) band | both received | median Δ_F | median abs Δ_F | 95th pct abs Δ_F | max abs Δ_F | median Δ_K |
|---|---|---|---|---|---|---|---|
| 0.05 | 0–1 | 37926 | 0.001 | 0.001 | 0.007 | 0.590 | 7.49e-04 |
| 0.05 | 1–2 | 19935 | 0.003 | 0.005 | 0.036 | 0.441 | 0.002 |
| 0.05 | 2–3 | 12062 | 0.011 | 0.013 | 0.077 | 0.415 | 0.009 |
| 0.05 | 3–4 | 5610 | 0.024 | 0.027 | 0.127 | 1.031 | 0.022 |
| 0.05 | 4–5 | 2219 | 0.083 | 0.084 | 0.293 | 2.254 | 0.081 |
| 0.05 | 5–6 | 841 | 0.224 | 0.224 | 0.820 | 3.082 | 0.223 |
| 0.05 | 6–8 | 436 | 0.548 | 0.548 | 1.181 | 2.097 | 0.546 |
| 0.05 | 8–12 | 37 | 1.677 | 1.677 | 3.135 | 4.045 | 1.675 |
| 0.05 | 12–∞ | 0 | - | - | - | - | - |
| 0.1 | 0–1 | 37926 | 0.005 | 0.005 | 0.028 | 1.491 | 0.003 |
| 0.1 | 1–2 | 19935 | 0.012 | 0.019 | 0.140 | 1.394 | 0.007 |
| 0.1 | 2–3 | 12062 | 0.042 | 0.051 | 0.264 | 1.125 | 0.036 |
| 0.1 | 3–4 | 5610 | 0.095 | 0.103 | 0.405 | 1.842 | 0.088 |
| 0.1 | 4–5 | 2219 | 0.293 | 0.296 | 0.773 | 3.058 | 0.286 |
| 0.1 | 5–6 | 841 | 0.696 | 0.696 | 1.782 | 3.663 | 0.689 |
| 0.1 | 6–8 | 436 | 1.376 | 1.376 | 2.279 | 3.268 | 1.369 |
| 0.1 | 8–12 | 37 | 2.882 | 2.882 | 4.473 | 5.335 | 2.876 |
| 0.1 | 12–∞ | 0 | - | - | - | - | - |
| 0.2 | 0–1 | 37926 | 0.020 | 0.020 | 0.097 | 2.971 | 0.011 |
| 0.2 | 1–2 | 19935 | 0.047 | 0.072 | 0.475 | 2.214 | 0.028 |
| 0.2 | 2–3 | 12062 | 0.153 | 0.182 | 0.694 | 1.499 | 0.129 |
| 0.2 | 3–4 | 5610 | 0.333 | 0.352 | 1.033 | 2.379 | 0.308 |
| 0.2 | 4–5 | 2219 | 0.839 | 0.839 | 1.496 | 3.314 | 0.814 |
| 0.2 | 5–6 | 841 | 1.575 | 1.575 | 2.450 | 4.276 | 1.548 |
| 0.2 | 6–8 | 436 | 2.496 | 2.496 | 3.484 | 4.031 | 2.470 |
| 0.2 | 8–12 | 37 | 4.147 | 4.147 | 5.840 | 6.322 | 4.121 |
| 0.2 | 12–∞ | 0 | - | - | - | - | - |
| 0.3 | 0–1 | 37926 | 0.042 | 0.043 | 0.177 | 3.139 | 0.023 |
| 0.3 | 1–2 | 19935 | 0.100 | 0.150 | 0.767 | 2.167 | 0.058 |
| 0.3 | 2–3 | 12062 | 0.315 | 0.357 | 1.105 | 1.911 | 0.262 |
| 0.3 | 3–4 | 5610 | 0.635 | 0.649 | 1.556 | 2.753 | 0.578 |
| 0.3 | 4–5 | 2219 | 1.357 | 1.357 | 2.056 | 3.694 | 1.299 |
| 0.3 | 5–6 | 841 | 2.277 | 2.277 | 3.023 | 4.377 | 2.220 |
| 0.3 | 6–8 | 436 | 3.277 | 3.277 | 4.285 | 4.725 | 3.219 |
| 0.3 | 8–12 | 37 | 5.025 | 5.025 | 6.505 | 7.112 | 4.967 |
| 0.3 | 12–∞ | 0 | - | - | - | - | - |

**Table 4.4-3. The crossover, per member and ap_e.** Launches received in both passes by T_cert at 4 r_s, sorted by the continuum's x = ln(1/Z_cont). (i) the first band of width 0.5 in x (with at least 5 launches) whose median |Δ_F| exceeds 0.1; (ii) the smallest x >= 0 of any single launch with |Δ_F| > 0.1, and (ii') the number of blueshifted launches (x < 0, Z > 1) with |Δ_F| > 0.1; (iii) reviewer B's estimate ln(1/g_d), g_d = ½(2aΩ_D)^{2/3} with the dispersive Ω_D, median over the launches of band (i); (iv) ln(2/ap_e²), the same estimate solved with Ω_D = Z·ap_e; (v) the largest x received in both (the reach of the comparison); (vi) the largest dispersive ln(1/Z_F) over the same launches (all classes), to set beside (iv). 'none' = no band crosses 0.1 below the reach.

| member | ap_e | (i) first band with median abs Δ_F > 0.1 | (ii) smallest x >= 0 with abs Δ_F > 0.1 (ii') | (iii) ln(1/g_d) there | (iv) ln(2/ap_e²) | (v) largest x received in both | (vi) largest ln(1/Z_F) over the same launches |
|---|---|---|---|---|---|---|---|
| P64_s4 | 0.05 | 4.5–5.0 | 2.187 (0) | 5.267 | 6.68 | 9.128 | 6.647 |
| P64_s4 | 0.1 | 3.0–3.5 | 1.299 (0) | 3.876 | 5.30 | 9.128 | 5.513 |
| P64_s4 | 0.2 | 2.0–2.5 | 0.735 (0) | 2.730 | 3.91 | 9.128 | 4.418 |
| P64_s4 | 0.3 | 1.0–1.5 | 0.559 (0) | 1.756 | 3.10 | 9.128 | 3.686 |
| P64_s2 | 0.05 | 4.5–5.0 | 2.239 (0) | 5.297 | 6.68 | 10.033 | 6.722 |
| P64_s2 | 0.1 | 3.0–3.5 | 1.221 (0) | 3.816 | 5.30 | 10.033 | 5.382 |
| P64_s2 | 0.2 | 2.0–2.5 | 0.738 (0) | 2.707 | 3.91 | 10.033 | 4.082 |
| P64_s2 | 0.3 | 1.0–1.5 | 0.237 (0) | 1.763 | 3.10 | 10.033 | 3.397 |
| P64_s8 | 0.05 | none | 0.014 (267) | - | 6.68 | 3.708 | 3.702 |
| P64_s8 | 0.1 | 3.5–4.0 | 0.014 (270) | 4.007 | 5.30 | 3.708 | 3.697 |
| P64_s8 | 0.2 | 2.0–2.5 | 0.014 (270) | 2.832 | 3.91 | 3.708 | 3.663 |
| P64_s8 | 0.3 | 1.0–1.5 | 0.014 (270) | 1.929 | 3.10 | 3.708 | 3.607 |
| P64_s4_tau4 | 0.05 | 5.0–5.5 | 2.025 (0) | 5.575 | 6.68 | 6.596 | 6.053 |
| P64_s4_tau4 | 0.1 | 3.0–3.5 | 1.323 (0) | 3.912 | 5.30 | 6.596 | 5.499 |
| P64_s4_tau4 | 0.2 | 2.0–2.5 | 0.017 (2) | 2.752 | 3.91 | 6.596 | 4.566 |
| P64_s4_tau4 | 0.3 | 1.0–1.5 | 0.017 (2) | 1.757 | 3.10 | 6.596 | 4.716 |
| W2 | 0.05 | 4.5–5.0 | 0.052 (272) | 5.347 | 6.68 | 4.877 | 4.869 |
| W2 | 0.1 | 4.0–4.5 | 0.028 (309) | 4.528 | 5.30 | 4.877 | 4.869 |
| W2 | 0.2 | 2.0–2.5 | 0.028 (309) | 2.801 | 3.91 | 4.877 | 4.816 |
| W2 | 0.3 | 1.0–1.5 | 0.028 (313) | 1.815 | 3.10 | 4.877 | 4.580 |
| S4_128_s8 | 0.05 | 5.0–5.5 | 0.004 (510) | 5.569 | 6.68 | 7.723 | 6.491 |
| S4_128_s8 | 0.1 | 4.0–4.5 | 0.004 (526) | 4.464 | 5.30 | 7.723 | 5.449 |
| S4_128_s8 | 0.2 | 2.0–2.5 | 0.004 (530) | 2.642 | 3.91 | 7.723 | 4.663 |
| S4_128_s8 | 0.3 | 1.0–1.5 | 0.004 (532) | 1.786 | 3.10 | 7.723 | 4.920 |
| S4_128_s16 | 0.05 | none | 0.012 (685) | - | 6.68 | 3.654 | 3.653 |
| S4_128_s16 | 0.1 | 2.0–2.5 | 0.012 (708) | 3.268 | 5.30 | 3.654 | 3.649 |
| S4_128_s16 | 0.2 | 1.5–2.0 | 0.012 (708) | 2.498 | 3.91 | 3.654 | 3.634 |
| S4_128_s16 | 0.3 | 1.5–2.0 | 0.012 (714) | 2.219 | 3.10 | 3.654 | 3.604 |
| P64_s4_tau2 | 0.05 | none | - (0) | - | 6.68 | 0.745 | 0.743 |
| P64_s4_tau2 | 0.1 | none | - (0) | - | 5.30 | 0.745 | 0.737 |
| P64_s4_tau2 | 0.2 | none | - (0) | - | 3.91 | 0.745 | 0.715 |
| P64_s4_tau2 | 0.3 | none | - (0) | - | 3.10 | 0.745 | 0.683 |
| ISO_64_s4 | 0.05 | none | - (0) | - | 6.68 | 0.766 | 0.764 |
| ISO_64_s4 | 0.1 | none | - (0) | - | 5.30 | 0.766 | 0.760 |
| ISO_64_s4 | 0.2 | none | - (0) | - | 3.91 | 0.766 | 0.744 |
| ISO_64_s4 | 0.3 | none | - (0) | - | 3.10 | 0.766 | 0.717 |
| ISO_128_s4 | 0.05 | none | - (0) | - | 6.68 | 0.703 | 0.702 |
| ISO_128_s4 | 0.1 | none | - (0) | - | 5.30 | 0.703 | 0.698 |
| ISO_128_s4 | 0.2 | none | - (0) | - | 3.91 | 0.703 | 0.683 |
| ISO_128_s4 | 0.3 | none | - (0) | - | 3.10 | 0.703 | 0.660 |
| S4_128_s4 | 0.05 | none | - (0) | - | 6.68 | 0.701 | 0.700 |
| S4_128_s4 | 0.1 | none | - (0) | - | 5.30 | 0.701 | 0.696 |
| S4_128_s4 | 0.2 | none | - (0) | - | 3.91 | 0.701 | 0.682 |
| S4_128_s4 | 0.3 | none | - (0) | - | 3.10 | 0.701 | 0.658 |
| ISO_128_s4_plain | 0.05 | none | - (0) | - | 6.68 | 0.703 | 0.702 |
| ISO_128_s4_plain | 0.1 | none | - (0) | - | 5.30 | 0.703 | 0.698 |
| ISO_128_s4_plain | 0.2 | none | - (0) | - | 3.91 | 0.703 | 0.683 |
| ISO_128_s4_plain | 0.3 | none | - (0) | - | 3.10 | 0.703 | 0.660 |
| M3c_s4_96 | 0.05 | none | - (0) | - | 6.68 | 0.922 | 0.920 |
| M3c_s4_96 | 0.1 | none | - (0) | - | 5.30 | 0.922 | 0.913 |
| M3c_s4_96 | 0.2 | none | - (0) | - | 3.91 | 0.922 | 0.887 |
| M3c_s4_96 | 0.3 | none | - (0) | - | 3.10 | 0.922 | 0.846 |
| P64_s4_newton | 0.05 | none | - (0) | - | 6.68 | 0.724 | 0.723 |
| P64_s4_newton | 0.1 | none | - (0) | - | 5.30 | 0.724 | 0.718 |
| P64_s4_newton | 0.2 | none | - (0) | - | 3.91 | 0.724 | 0.700 |
| P64_s4_newton | 0.3 | none | - (0) | - | 3.10 | 0.724 | 0.669 |

**Table 4.4-4. reviewer A R7 (the lattice leak) on B launches at or inside the long-wave root (delta <= 0) of a record with a root.** Ceiling: Z_R7 = N_R (1 − cos ap_e), N_R the record's clock at its outer root (R7: content emitted at or inside the long-wave root reaches infinity at that fraction of its comoving frequency at most, on stationary fields). Per member and ap_e: such launches; those emitted with Ω_e <= 0 (inside the k-dependent surface |v| cos(ap_e) = N, where the stationary map delivers nothing); then the launches received at 4 r_s by T_cert in the dispersive pass, split by route: (rel) also received in the continuum pass, i.e. released by the root's motion; (leak) received in the dispersive pass only. Per route: count, how many exceed the ceiling, and ln(Z_F/Z_R7) median / max; last, for the leak route, the median ln(Ω_D/Ω_e) over launches with Ω_e > 0 (reviewer B's dΩ/dT diagnostic, 0 on stationary fields) and the median / largest certified ln(1/Z_F), the largest ap reached along the path (median / max) and the flight time T_D − T_e in r_s (median / max).

| member | ap_e | B launches delta <= 0 at a root record (Ω_e <= 0) | rel: received / above ceiling / ln(Z_F/Z_R7) median, max | leak: received / above ceiling / ln(Z_F/Z_R7) median, max | leak: median ln(Ω_D/Ω_e), Ω_e > 0 (count) | leak: ln(1/Z_F) median / max | leak: max ap along path, median / max | leak: flight / r_s, median / max |
|---|---|---|---|---|---|---|---|---|
| P64_s4 | 0.05 | 1236 (1218) | 165 / 165 / 3.262, 4.241 | 42 / 40 / 5.755, 7.573 | 3.571 (6) | 0.983 / 7.161 | 0.472 / 0.761 | 17.369 / 26.448 |
| P64_s4 | 0.1 | 1236 (1091) | 165 / 165 / 1.993, 2.902 | 79 / 58 / 1.125, 5.423 | 1.911 (13) | 4.254 / 7.442 | 0.150 / 0.723 | 16.086 / 25.833 |
| P64_s4 | 0.2 | 1236 (661) | 165 / 152 / 0.871, 1.687 | 207 / 92 / -0.180, 3.261 | 0.087 (121) | 4.268 / 8.072 | 0.200 / 0.657 | 13.899 / 27.135 |
| P64_s4 | 0.3 | 1236 (375) | 165 / 129 / 0.393, 1.107 | 373 / 87 / -0.306, 1.902 | -0.004 (279) | 3.578 / 9.415 | 0.300 / 0.586 | 11.830 / 25.904 |
| P64_s2 | 0.05 | 1344 (1344) | 136 / 136 / 3.333, 4.297 | 168 / 164 / 5.820, 7.464 | - (0) | 1.033 / 9.003 | 0.831 / 1.008 | 17.468 / 27.020 |
| P64_s2 | 0.1 | 1344 (1322) | 136 / 136 / 2.086, 2.962 | 161 / 133 / 3.210, 5.311 | 3.732 (3) | 2.188 / 9.287 | 0.800 / 0.980 | 17.052 / 26.727 |
| P64_s2 | 0.2 | 1344 (875) | 136 / 136 / 1.073, 1.730 | 280 / 87 / -0.298, 3.151 | 0.219 (164) | 4.387 / 7.432 | 0.200 / 0.917 | 12.805 / 26.864 |
| P64_s2 | 0.3 | 1344 (633) | 136 / 132 / 0.711, 1.301 | 380 / 48 / -0.300, 1.806 | 0.084 (317) | 3.554 / 8.263 | 0.300 / 0.856 | 10.943 / 27.876 |
| P64_s8 | 0.05 | 132 (130) | 132 / 132 / 5.036, 6.941 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s8 | 0.1 | 132 (102) | 132 / 132 / 3.577, 5.167 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s8 | 0.2 | 132 (53) | 132 / 132 / 1.982, 3.012 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s8 | 0.3 | 132 (28) | 132 / 127 / 0.791, 1.965 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_tau4 | 0.05 | 2064 (1885) | 530 / 530 / 3.631, 4.844 | 60 / 60 / 6.196, 7.721 | 4.070 (10) | 0.658 / 6.180 | 0.368 / 0.790 | 15.843 / 20.677 |
| P64_s4_tau4 | 0.1 | 2064 (1601) | 530 / 530 / 2.267, 3.550 | 105 / 97 / 3.778, 5.637 | 4.059 (17) | 1.617 / 6.173 | 0.346 / 0.749 | 15.373 / 18.520 |
| P64_s4_tau4 | 0.2 | 2064 (908) | 530 / 504 / 1.059, 2.416 | 240 / 175 / 0.879, 3.480 | 0.702 (74) | 3.222 / 6.555 | 0.269 / 0.683 | 13.891 / 19.221 |
| P64_s4_tau4 | 0.3 | 2064 (537) | 530 / 387 / 0.399, 1.685 | 344 / 149 / -0.227, 2.190 | -0.069 (191) | 3.480 / 6.968 | 0.300 / 0.615 | 12.638 / 19.430 |
| W2 | 0.05 | 666 (639) | 318 / 318 / 4.387, 7.875 | 35 / 35 / 5.217, 7.535 | 2.889 (6) | 1.670 / 3.292 | 0.174 / 0.540 | 15.430 / 17.722 |
| W2 | 0.1 | 666 (563) | 318 / 318 / 2.981, 5.754 | 53 / 53 / 3.796, 5.327 | 3.107 (10) | 1.687 / 3.302 | 0.222 / 0.590 | 14.513 / 17.733 |
| W2 | 0.2 | 666 (337) | 318 / 307 / 1.547, 3.642 | 54 / 54 / 2.393, 3.222 | 2.324 (17) | 1.723 / 3.531 | 0.309 / 0.528 | 13.953 / 16.966 |
| W2 | 0.3 | 666 (210) | 318 / 242 / 0.607, 2.363 | 54 / 44 / 1.409, 1.859 | 1.241 (22) | 1.842 / 4.569 | 0.361 / 0.475 | 14.153 / 16.574 |
| S4_128_s8 | 0.05 | 2154 (2127) | 577 / 577 / 3.921, 7.874 | 8 / 8 / 0.810, 6.988 | 4.247 (1) | 6.018 / 6.761 | 0.050 / 0.488 | 18.311 / 18.862 |
| S4_128_s8 | 0.1 | 2154 (1607) | 577 / 577 / 2.545, 5.766 | 44 / 30 / 0.672, 5.265 | 2.672 (5) | 4.796 / 7.234 | 0.109 / 0.538 | 15.823 / 19.890 |
| S4_128_s8 | 0.2 | 2154 (847) | 577 / 547 / 1.247, 3.636 | 190 / 94 / -0.009, 3.134 | 0.188 (116) | 4.090 / 6.849 | 0.200 / 0.509 | 13.182 / 19.229 |
| S4_128_s8 | 0.3 | 2154 (399) | 577 / 435 / 0.476, 2.400 | 414 / 132 / -0.295, 1.673 | -0.052 (338) | 3.592 / 6.678 | 0.300 / 0.454 | 11.246 / 20.101 |
| S4_128_s16 | 0.05 | 276 (259) | 276 / 276 / 5.148, 6.974 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s16 | 0.1 | 276 (167) | 276 / 276 / 3.686, 5.318 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s16 | 0.2 | 276 (76) | 276 / 276 / 2.009, 3.342 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s16 | 0.3 | 276 (28) | 276 / 259 / 0.702, 2.214 | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_tau2 | 0.05 | 816 (790) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_tau2 | 0.1 | 816 (630) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_tau2 | 0.2 | 816 (356) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_tau2 | 0.3 | 816 (213) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_64_s4 | 0.05 | 1206 (1182) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_64_s4 | 0.1 | 1206 (1061) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_64_s4 | 0.2 | 1206 (636) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_64_s4 | 0.3 | 1206 (376) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4 | 0.05 | 48 (48) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4 | 0.1 | 48 (31) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4 | 0.2 | 48 (15) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4 | 0.3 | 48 (7) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s4 | 0.05 | 768 (762) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s4 | 0.1 | 768 (556) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s4 | 0.2 | 768 (301) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| S4_128_s4 | 0.3 | 768 (166) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4_plain | 0.05 | 264 (253) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4_plain | 0.1 | 264 (167) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4_plain | 0.2 | 264 (83) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| ISO_128_s4_plain | 0.3 | 264 (37) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| M3c_s4_96 | 0.05 | 66 (62) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| M3c_s4_96 | 0.1 | 66 (46) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| M3c_s4_96 | 0.2 | 66 (25) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| M3c_s4_96 | 0.3 | 66 (11) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_newton | 0.05 | 18 (16) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_newton | 0.1 | 18 (11) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_newton | 0.2 | 18 (6) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |
| P64_s4_newton | 0.3 | 18 (4) | 0 / 0 / -, - | 0 / 0 / -, - | - (0) | - / - | - / - | - / - |

**Table 4.4-5. Flags.** Per member and ap_e: launches that ever passed ap = π/2 (the long-wave outgoing branch's edge) and how many of them were received by T_cert; the largest ap along the path over received launches (median / max); launches still inside the chain at the end of the stored records (status 2: trapped, or still in flight, at the last record; most sit at the inner root, where outgoing rays from the centre and ingoing rays from the trapped zone meet); received launches emitted with Ω_e <= 0; the median and the largest |ln(Ω_D/Ω_e)| over launches received by T_cert with Ω_e > 0 (reviewer B's diagnostic of a time-dependent interior; 0 on stationary fields).

| member | ap_e | passed π/2 (received cert) | max ap along path, received: median / max | status 2 at the end of the records | received with Ω_e <= 0 | abs ln(Ω_D/Ω_e), received, Ω_e > 0: median / max |
|---|---|---|---|---|---|---|
| P64_s4 | 0.05 | 0 (0) | 0.050 / 0.868 | 8521 | 299 | 0.218 / 6.729 |
| P64_s4 | 0.1 | 0 (0) | 0.100 / 0.859 | 8499 | 223 | 0.193 / 5.799 |
| P64_s4 | 0.2 | 0 (0) | 0.200 / 0.845 | 8407 | 265 | 0.178 / 6.252 |
| P64_s4 | 0.3 | 0 (0) | 0.300 / 0.860 | 8268 | 277 | 0.106 / 7.943 |
| P64_s2 | 0.05 | 0 (0) | 0.050 / 1.087 | 8621 | 406 | 0.193 / 6.116 |
| P64_s2 | 0.1 | 0 (0) | 0.100 / 1.019 | 8601 | 320 | 0.177 / 7.012 |
| P64_s2 | 0.2 | 0 (0) | 0.200 / 0.958 | 8533 | 269 | 0.119 / 6.107 |
| P64_s2 | 0.3 | 0 (0) | 0.300 / 0.922 | 8403 | 237 | 0.088 / 9.982 |
| P64_s8 | 0.05 | 0 (0) | 0.050 / 0.444 | 7847 | 211 | 0.249 / 7.416 |
| P64_s8 | 0.1 | 0 (0) | 0.100 / 0.538 | 7826 | 181 | 0.222 / 8.135 |
| P64_s8 | 0.2 | 0 (0) | 0.200 / 0.633 | 7718 | 124 | 0.231 / 6.840 |
| P64_s8 | 0.3 | 0 (0) | 0.300 / 0.712 | 7529 | 84 | 0.204 / 7.277 |
| P64_s4_tau4 | 0.05 | 0 (0) | 0.050 / 0.887 | 16064 | 739 | 0.308 / 9.150 |
| P64_s4_tau4 | 0.1 | 0 (0) | 0.100 / 0.878 | 16026 | 676 | 0.299 / 7.259 |
| P64_s4_tau4 | 0.2 | 0 (0) | 0.200 / 0.863 | 15879 | 591 | 0.253 / 6.927 |
| P64_s4_tau4 | 0.3 | 0 (0) | 0.300 / 0.876 | 15635 | 463 | 0.192 / 6.851 |
| W2 | 0.05 | 0 (0) | 0.050 / 0.898 | 8281 | 559 | 0.346 / 8.857 |
| W2 | 0.1 | 0 (0) | 0.100 / 0.893 | 8259 | 531 | 0.349 / 8.915 |
| W2 | 0.2 | 0 (0) | 0.200 / 0.878 | 8191 | 395 | 0.274 / 7.901 |
| W2 | 0.3 | 0 (0) | 0.300 / 0.891 | 8078 | 303 | 0.231 / 7.402 |
| S4_128_s8 | 0.05 | 0 (0) | 0.050 / 0.948 | 17419 | 741 | 0.300 / 10.124 |
| S4_128_s8 | 0.1 | 0 (0) | 0.100 / 0.965 | 17458 | 596 | 0.293 / 9.098 |
| S4_128_s8 | 0.2 | 0 (0) | 0.200 / 0.967 | 17351 | 467 | 0.266 / 11.327 |
| S4_128_s8 | 0.3 | 0 (0) | 0.300 / 0.978 | 17125 | 260 | 0.218 / 7.600 |
| S4_128_s16 | 0.05 | 0 (0) | 0.050 / 0.528 | 16103 | 334 | 0.357 / 9.147 |
| S4_128_s16 | 0.1 | 0 (0) | 0.100 / 0.605 | 16060 | 231 | 0.313 / 8.741 |
| S4_128_s16 | 0.2 | 0 (0) | 0.200 / 0.702 | 15893 | 81 | 0.308 / 5.428 |
| S4_128_s16 | 0.3 | 0 (0) | 0.300 / 0.781 | 15302 | 32 | 0.224 / 5.748 |
| P64_s4_tau2 | 0.05 | 0 (0) | 0.064 / 0.078 | 20573 | 0 | 0.078 / 0.257 |
| P64_s4_tau2 | 0.1 | 0 (0) | 0.127 / 0.155 | 20564 | 0 | 0.077 / 0.255 |
| P64_s4_tau2 | 0.2 | 0 (0) | 0.252 / 0.304 | 20521 | 0 | 0.073 / 0.246 |
| P64_s4_tau2 | 0.3 | 0 (0) | 0.372 / 0.444 | 20445 | 0 | 0.067 / 0.232 |
| ISO_64_s4 | 0.05 | 0 (0) | 0.066 / 0.080 | 8497 | 0 | 0.085 / 0.258 |
| ISO_64_s4 | 0.1 | 0 (0) | 0.131 / 0.159 | 8481 | 0 | 0.084 / 0.254 |
| ISO_64_s4 | 0.2 | 0 (0) | 0.259 / 0.312 | 8399 | 0 | 0.079 / 0.247 |
| ISO_64_s4 | 0.3 | 0 (0) | 0.383 / 0.454 | 8267 | 0 | 0.073 / 0.233 |
| ISO_128_s4 | 0.05 | 0 (0) | 0.065 / 0.081 | 5716 | 0 | 0.076 / 0.215 |
| ISO_128_s4 | 0.1 | 0 (0) | 0.130 / 0.161 | 5716 | 0 | 0.075 / 0.212 |
| ISO_128_s4 | 0.2 | 0 (0) | 0.258 / 0.315 | 5716 | 0 | 0.071 / 0.203 |
| ISO_128_s4 | 0.3 | 0 (0) | 0.381 / 0.459 | 5716 | 0 | 0.065 / 0.189 |
| S4_128_s4 | 0.05 | 0 (0) | 0.050 / 0.053 | 9666 | 0 | 0.002 / 0.012 |
| S4_128_s4 | 0.1 | 0 (0) | 0.100 / 0.107 | 9665 | 0 | 0.002 / 0.012 |
| S4_128_s4 | 0.2 | 0 (0) | 0.200 / 0.212 | 9660 | 0 | 0.002 / 0.011 |
| S4_128_s4 | 0.3 | 0 (0) | 0.300 / 0.317 | 9652 | 0 | 0.002 / 0.011 |
| ISO_128_s4_plain | 0.05 | 0 (0) | 0.050 / 0.054 | 6907 | 0 | 0.002 / 0.010 |
| ISO_128_s4_plain | 0.1 | 0 (0) | 0.100 / 0.107 | 6907 | 0 | 0.002 / 0.010 |
| ISO_128_s4_plain | 0.2 | 0 (0) | 0.200 / 0.214 | 6907 | 0 | 0.001 / 0.009 |
| ISO_128_s4_plain | 0.3 | 0 (0) | 0.300 / 0.320 | 6907 | 0 | 0.001 / 0.009 |
| M3c_s4_96 | 0.05 | 0 (0) | 0.067 / 0.083 | 7105 | 0 | 0.113 / 0.296 |
| M3c_s4_96 | 0.1 | 0 (0) | 0.133 / 0.164 | 7099 | 0 | 0.112 / 0.291 |
| M3c_s4_96 | 0.2 | 0 (0) | 0.264 / 0.319 | 7079 | 0 | 0.109 / 0.271 |
| M3c_s4_96 | 0.3 | 0 (0) | 0.389 / 0.460 | 7035 | 0 | 0.104 / 0.246 |
| P64_s4_newton | 0.05 | 0 (0) | 0.064 / 0.078 | 2840 | 0 | 0.074 / 0.234 |
| P64_s4_newton | 0.1 | 0 (0) | 0.127 / 0.155 | 2840 | 0 | 0.074 / 0.232 |
| P64_s4_newton | 0.2 | 0 (0) | 0.251 / 0.304 | 2840 | 0 | 0.070 / 0.223 |
| P64_s4_newton | 0.3 | 0 (0) | 0.372 / 0.444 | 2840 | 0 | 0.064 / 0.210 |

**Headline (script-printed).**

- ap_e 0.05 (14 members): like for like (launches received in both passes by T_cert at 4 r_s), the largest ln(1/Z) dispersive (F) minus continuum: clocks A -1.730 to 0.004, root launches B -3.311 to -0.001; the dispersive pass's own largest (all its received launches) minus its like-for-like value: A 0.000 to 1.774, B 0.000 to 2.281; crossover band (median |Δ_F| > 0.1) on 5 of 14 members at x = 4.500 to 5.000 (reviewer B's ln(2/ap_e²) = 6.68); R7 at or inside the root: released route 2134 received, 2134 above the ceiling; leak route 313 received, 307 above the ceiling; max |ln Z_K − ln Z_F| 8.31e-04 to 0.003.
- ap_e 0.1 (14 members): like for like (launches received in both passes by T_cert at 4 r_s), the largest ln(1/Z) dispersive (F) minus continuum: clocks A -2.525 to 0.001, root launches B -4.651 to -0.005; the dispersive pass's own largest (all its received launches) minus its like-for-like value: A 0.000 to 2.199, B 0.000 to 3.905; crossover band (median |Δ_F| > 0.1) on 7 of 14 members at x = 2.000 to 4.000 (reviewer B's ln(2/ap_e²) = 5.30); R7 at or inside the root: released route 2134 received, 2134 above the ceiling; leak route 442 received, 371 above the ceiling; max |ln Z_K − ln Z_F| 0.003 to 0.011.
- ap_e 0.2 (14 members): like for like (launches received in both passes by T_cert at 4 r_s), the largest ln(1/Z) dispersive (F) minus continuum: clocks A -3.416 to -0.017, root launches B -5.952 to -0.021; the dispersive pass's own largest (all its received launches) minus its like-for-like value: A 0.000 to 3.448, B 0.000 to 4.172; crossover band (median |Δ_F| > 0.1) on 7 of 14 members at x = 1.500 to 2.000 (reviewer B's ln(2/ap_e²) = 3.91); R7 at or inside the root: released route 2134 received, 2054 above the ceiling; leak route 971 received, 502 above the ceiling; max |ln Z_K − ln Z_F| 0.013 to 0.042.
- ap_e 0.3 (14 members): like for like (launches received in both passes by T_cert at 4 r_s), the largest ln(1/Z) dispersive (F) minus continuum: clocks A -3.997 to -0.041, root launches B -6.636 to -0.049; the dispersive pass's own largest (all its received launches) minus its like-for-like value: A 0.000 to 7.252, B 0.000 to 6.392; crossover band (median |Δ_F| > 0.1) on 7 of 14 members at x = 1.000 to 1.500 (reviewer B's ln(2/ap_e²) = 3.10); R7 at or inside the root: released route 2134 received, 1711 above the ceiling; leak route 1565 received, 460 above the ceiling; max |ln Z_K − ln Z_F| 0.029 to 0.093.

#### 4.4 controls on P64_s4 (rz_disp_ctrl_tables_v1.py; out/disp_tables_v1/disp_ctrl_v1.md)

**Table 4.4-6. Step control on P64_s4 (reviewer B 4.6 A): h = 0.5 a against h = 0.25 a**, the same 10946 launches, at 4 r_s, received by T_cert.

| ap_e | received in both / h 0.5 only / h 0.25 only | abs Δ ln Z_F: median / max | abs Δ t_D (a): median / max | largest ln(1/Z_F): h 0.5 / h 0.25 |
|---|---|---|---|---|
| 0.05 | 6064 / 0 / 0 | 5.06e-11 / 1.86e-07 | 2.35e-10 / 2.44e-05 | 8.712 / 8.712 |
| 0.1 | 6103 / 0 / 0 | 4.96e-11 / 1.40e-07 | 3.41e-10 / 2.05e-05 | 9.018 / 9.018 |
| 0.2 | 6993 / 0 / 0 | 5.15e-11 / 1.72e-06 | 9.70e-10 / 1.77e-04 | 8.590 / 8.590 |
| 0.3 | 7668 / 0 / 0 | 7.33e-11 / 4.94e-08 | 2.94e-09 / 8.49e-06 | 10.078 / 10.078 |

**Table 4.4-7. The dense fields on P64_s4** (quintic node spline, 11129 launches from the dense continuum pass, h = 0.5 a), at 4 r_s, received by T_cert: the main variant's Tables 4.4-1 and 4.4-3 read on the dense fields.

| ap_e | received: disp / cont / both | A: F all / F both / cont both | B: F all / F both / cont both | first band with median abs Δ_F > 0.1 | largest ln(1/Z_F) like for like |
|---|---|---|---|---|---|
| 0.05 | 6200 / 5395 / 5395 | 8.422 / 6.047 / 7.502 | 9.364 / 6.615 / 8.588 | 4.5–5.0 | 6.615 |
| 0.1 | 6411 / 5395 / 5395 | 6.828 / 5.033 / 7.502 | 8.865 / 5.465 / 8.588 | 3.0–3.5 | 5.465 |
| 0.2 | 7108 / 5395 / 5395 | 6.574 / 3.912 / 7.502 | 8.640 / 4.253 / 8.588 | 2.0–2.5 | 4.253 |
| 0.3 | 7691 / 5395 / 5395 | 8.469 / 3.397 / 7.502 | 7.673 / 3.552 / 8.588 | 1.0–1.5 | 3.552 |

#### What the 4.4 tables show (numbers as printed above; no ruling)

1. **Small redshift.** Launch by launch, at continuum ln(1/Z) below 1, ln Z_F − ln Z_cont has median 0.001, 0.005, 0.020, 0.042 for ap_e = 0.05, 0.1, 0.2, 0.3 (Table 4.4-2, pooled over the 14 members). Divided by (ap_e)² these are 0.51, 0.51, 0.49, 0.47, so the leading difference scales as (ap_e)².
2. **Sign.** In 32 of the 32 (band, ap_e) cells of Table 4.4-2 the median Δ_F is positive: the dispersive ray arrives less redshifted than the continuum ray from the same launch.
3. **Crossover.** The first 0.5-wide band of continuum ln(1/Z) in which the median |Δ_F| exceeds 0.1 (Table 4.4-3, column (i)) exists on 5 / 7 / 7 / 7 of 14 members for ap_e = 0.05 / 0.1 / 0.2 / 0.3. Its lower edge lies at x = 4.5 to 5.0; 2.0 to 4.0; 1.5 to 2.0; 1.0 to 1.5 respectively. That edge is 1.68 to 2.18; 1.30 to 3.30; 1.91 to 2.41; 1.60 to 2.10 below reviewer B's ln(2/ap_e²) (the crossover ½(2aΩ_D)^{2/3} solved with Ω_D = Z ap_e), and 0.57 to 0.85; 0.46 to 1.27; 0.64 to 1.00; 0.72 to 0.93 below reviewer B's ln(1/g_d) evaluated with the band's own Ω_D (column (iii)). So reviewer B's scale marks where the correction is of order one, not where it reaches 0.1. The other 7 members (P64_s4_tau2, ISO_64_s4, ISO_128_s4, S4_128_s4, ISO_128_s4_plain, M3c_s4_96, P64_s4_newton) do not reach a certified ln(1/Z) above 0.922 at 4 r_s (column (v)), so the comparison has no crossover on them.
4. **The deepest certified redshifts, like for like** [v4: rewritten without 'cap' (AstraAPI_rzchk4 sec. 4.5); archival, C pooled]. On the members whose continuum comparison reaches past ln(2/ap_e²) (column (v) > (iv)), the largest dispersive ln(1/Z_F) over the same launches (column (vi)) minus ln(2/ap_e²) is: ap_e 0.05: P64_s4 -0.04, P64_s2 0.04, S4_128_s8 -0.19; ap_e 0.1: P64_s4 0.21, P64_s2 0.08, P64_s4_tau4 0.20, S4_128_s8 0.15; ap_e 0.2: P64_s4 0.51, P64_s2 0.17, P64_s4_tau4 0.65, W2 0.90, S4_128_s8 0.75; ap_e 0.3: P64_s4 0.58, P64_s2 0.30, P64_s8 0.51, P64_s4_tau4 1.62, W2 1.48, S4_128_s8 1.82, S4_128_s16 0.50. In Table 4.4-1's like-for-like columns, per class, the dispersive largest over launches received in both passes sits below the continuum's largest over those launches wherever the continuum goes deeper than ln(2/ap_e²): in 46 of the 47 (member, ap_e, class) cells where the continuum's like-for-like largest exceeds ln(2/ap_e²), the dispersive one is smaller. [v4: an archival count over cells that pool the dust class C, whose Doppler normalization is convention-dependent, so it is not a C-free finite-frequency comparison; AstraAPI_rzchk3 sec. 7.3: "Removing C leaves **37 of 37 A/B classwise comparisons** with the stated reduction. That is a narrower arithmetic summary of the printed table—not a replacement physical cap theorem." A reduction count, not a cap.]
5. **Arrivals the continuum does not have.** Launches received by T_cert in the dispersive pass only, summed over members: 2663 / 3608 / 7883 / 12669 for ap_e = 0.05 / 0.1 / 0.2 / 0.3 (against 81445 / 81445 / 81445 / 81445 received in both). Launches received in the continuum only: 0 / 0 / 0 / 0. The at-or-inside-root part of the dispersive-only arrivals is Table 4.4-4's leak route.
6. **reviewer A R7 on the histories.** B launches at or inside the long-wave root (δ <= 0) of a root record, received by T_cert at 4 r_s, summed over members, as (received, above the ceiling N_R(1 − cos ap_e)): released route (also received in the continuum) ap_e 0.05: (2134, 2134); ap_e 0.1: (2134, 2134); ap_e 0.2: (2134, 2054); ap_e 0.3: (2134, 1711); leak route (dispersive only) ap_e 0.05: (313, 307); ap_e 0.1: (442, 371); ap_e 0.2: (971, 502); ap_e 0.3: (1565, 460). R7's ceiling is a statement on stationary fields; on these records the fraction of received at-or-inside launches above it is 1.00 / 0.97 / 0.82 / 0.59 for ap_e = 0.05 / 0.1 / 0.2 / 0.3. The leak-route launches occur on P64_s4, P64_s2, P64_s4_tau4, W2, S4_128_s8; their median flight per (member, ap_e) is 10.9 to 18.3 r_s; the median (over members) of their median largest ap along the path is 0.368 / 0.222 / 0.200 / 0.300 for ap_e = 0.05 / 0.1 / 0.2 / 0.3, and the largest ap reached on any leak path is 1.008. Inference, not tested here: the low-ap_e leak rays gained lattice momentum while hovering and left once the lattice group velocity v cos(2ap) + N cos(ap) turned outward. The ray description at ap near 1, on fields that vary over σ = 2 to 8 a, is outside the regime this pass certifies; reviewer B 4.5's packets are the test, not run here.
7. **Emitter convention.** The largest |ln Z_K − ln Z_F| over received launches, over all members, is 0.003 / 0.011 / 0.042 / 0.093 for ap_e = 0.05 / 0.1 / 0.2 / 0.3 (Table 4.4-1, last column): reviewer B's caution is a correction of that size here.
8. **Branch.** 0 (launch, ap_e) pairs, summed over members, passed ap = π/2 (Table 4.4-5); the largest ap reached along any received path is 1.087.
9. **Ω along the path** (reviewer B's dΩ/dT diagnostic). The median |ln(Ω_D/Ω_e)| over launches received with Ω_e > 0 ranges 0.001 to 0.357 across members and ap_e. 9895 received (launch, ap_e) pairs, summed over members and the four ap_e, were emitted with Ω_e <= 0, inside the k-dependent surface, where the stationary map delivers nothing.
10. **Controls on P64_s4.** Step (Table 4.4-6): h = 0.5 a against 0.25 a, the largest |Δ ln Z_F| over launches received in both is 1.86e-07 / 1.40e-07 / 1.72e-06 / 4.94e-08 for ap_e = 0.05 / 0.1 / 0.2 / 0.3 (medians 5.06e-11 / 4.96e-11 / 5.15e-11 / 7.33e-11); launches received at one step only (h 0.5 only + h 0.25 only): 0+0 / 0+0 / 0+0 / 0+0. Dense fields (Table 4.4-7): the crossover band (i) is 4.5–5.0 / 3.0–3.5 / 2.0–2.5 / 1.0–1.5 against the main variant's 4.5–5.0 / 3.0–3.5 / 2.0–2.5 / 1.0–1.5, and the like-for-like largest ln(1/Z_F) is 6.615 / 5.465 / 4.253 / 3.552 against 6.647 / 5.513 / 4.418 / 3.686.

**Not covered by this pass.** The heavy channel (self-test 3) [v3: superseded, run in the addendum]; reviewer B 4.5's packets and reviewer A route 2 (the wave test of the leak route and of the ray regime); the uncertified late windows of P64_s4_B2 and P64_s4_ext200; the dispersive pass on variants other than the main one and the dense control.
#### 4.4 addendum: the heavy channel (added 2026-10-22 03:52:24 UTC)

**What was run.** rz_disp_v2.py integrates reviewer B's ray equations in (r, p) itself; v1's variable ln ap cannot pass p = 0, where a bound heavy ray turns (v1's self-test 3) [v3: a momentum sign change; a radial turning point is ṙ = 0]. Channel: each member's own m a, angular channel j = 1. Launch events: the continuum pass's, emitted outgoing at ap_e = 0.2, 0.3, 0.6, 1. The massless pass's 0.05 and 0.1 are not run here: at m a = 0.2 the band energy √(m² + sin²(ap_e)) is 0.2062 and 0.2235 there, within 12 percent of m before the j²/r² term, and already at ap_e = 0.2 the launches bound at infinity (Ω_e < m) number 130197 of 171356 summed over members (Table 4.4-8). Members: the same 14. Receivers and certified extents are as before. For m = j = 0 rz_disp_v2 calls v1's integrator unchanged.

| member | m a | launches | integrator | run seconds | script md5 (first 8) |
|---|---|---|---|---|---|
| P64_s4 | 0.20 | 10946 | v2 (r, p) | 179 | c9ff1f2d |
| P64_s2 | 0.20 | 11102 | v2 (r, p) | 201 | c9ff1f2d |
| P64_s8 | 0.20 | 10201 | v2 (r, p) | 192 | c9ff1f2d |
| P64_s4_tau4 | 0.20 | 20807 | v2 (r, p) | 344 | c9ff1f2d |
| W2 | 0.20 | 10614 | v2 (r, p) | 210 | c9ff1f2d |
| S4_128_s8 | 0.20 | 22294 | v2 (r, p) | 740 | c9ff1f2d |
| S4_128_s16 | 0.20 | 20797 | v2 (r, p) | 706 | c9ff1f2d |
| P64_s4_tau2 | 0.20 | 21049 | v2 (r, p) | 195 | c9ff1f2d |
| ISO_64_s4 | 0.20 | 11003 | v2 (r, p) | 211 | c9ff1f2d |
| ISO_128_s4 | 0.20 | 5716 | v2 (r, p) | 72 | c9ff1f2d |
| S4_128_s4 | 0.20 | 9719 | v2 (r, p) | 199 | c9ff1f2d |
| ISO_128_s4_plain | 0.20 | 6907 | v2 (r, p) | 100 | c9ff1f2d |
| M3c_s4_96 | 0.10 | 7361 | v2 (r, p) | 129 | c9ff1f2d |
| P64_s4_newton | 0.20 | 2840 | v2 (r, p) | 26 | c9ff1f2d |

**Self-tests** (rz_selftest_disp_v2.py; logs/selftest_disp_v2.log, lines pasted):

```
start 2026-10-22T03:31:59Z
PG heavy 0.3 {'received': 1, 'bound_at_infinity': 5, 'turned': 0, 'max_rel_dOmega_received': 2.7644553313166398e-14, 'max_abs_dZ_vs_stationary_map': 1.509903313490213e-14, 'status': [3, 3, 3, 3, 1]}
PG heavy 0.6 {'received': 4, 'bound_at_infinity': 2, 'turned': 0, 'max_rel_dOmega_received': 5.325344609730109e-10, 'max_abs_dZ_vs_stationary_map': 1.5442314094116227e-10, 'status': [3, 1, 1, 1, 1]}
PG heavy 1.0 {'received': 5, 'bound_at_infinity': 0, 'turned': 0, 'max_rel_dOmega_received': 3.735309839214551e-10, 'max_abs_dZ_vs_stationary_map': 1.8194157291873125e-10, 'status': [1, 1, 1, 1, 1]}
massless lin vs log 0.1 {'both': 5, 'max_abs_dlnOmD': 1.896371948364028e-12, 'max_abs_dtD': 1.6484591469634324e-11}
massless lin vs log 0.3 {'both': 5, 'max_abs_dlnOmD': 5.131450819816157e-13, 'max_abs_dtD': 1.5711520973127335e-10}
drift heavy {'both': 2, 'max_abs_dtD': 1.550867978039605e-06, 'max_rel_dpD': 3.9262730844313865e-09}
end rc=0 2026-10-22T03:32:15Z
```

Reading of the log. (1) On the stationary Painlevé-Gullstrand exterior (r_s/a 64), heavy chain m a 0.2, j = 1, emitters at 1.05 to 3 r_s: Ω is conserved on every received heavy ray to 5.33e-10 (relative) and the received Z_F equals the stationary map Ω_e/ε_e to 1.82e-10. Rays bound at infinity (Ω_e < m) are counted in the log; on this exterior an outgoing heavy quantum emitted close to the root is dragged inward and falls (status 3), it does not turn in p. (2) On the massless channel the (r, p) and the (r, ln ap) integrators agree to 1.90e-12 in ln Ω_D and 1.57e-10 a in arrival time. (3) On the drifting-charge exterior, heavy, ap_e 0.6, the arrival time at 4 r_s agrees with scipy's RK45 (rtol 1e-11) to 1.55e-06 a and the arrival momentum to 3.93e-09 (relative), on the 2 rays that arrive in both.

##### Heavy-channel tables (rz_disp_heavy_tables_v3.py; out/disp_tables_v1/disp_heavy_v1.md)

**Table 4.4-8. The heavy channel** (rz_disp_v2, the (r, p) integrator; the member's own m a, angular channel j = 1; the continuum pass's launch events, emitted outgoing at ap_e), at 4 r_s. Fates of the launches: received by T_cert / received at all; fell to r <= 0.5 a; left the chain; still inside at the end of the records; p changed sign along the path. Bound at infinity: Ω_e < m at emission. [v3: an instantaneous stationary escape-threshold count, not a fate on a time-dependent history; the 'fates' overlap (a ray can cross 4 r_s and stay inside the chain): Table HX cross-tabulates them.]

| member | m a | ap_e | launches | received cert / all | fell to the centre | left the chain | inside at the end | p changed sign | bound at infinity (Ω_e < m) | largest certified ln(1/Z_F): A / B / C |
|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | 0.20 | 0.2 | 10946 | 1731 / 1731 | 0 | 31 | 10915 | 8521 | 9401 | 1.623 / 0.465 / 3.046 |
| P64_s4 | 0.20 | 0.3 | 10946 | 2476 / 2476 | 0 | 1140 | 9806 | 7290 | 8250 | 1.605 / 0.699 / 3.124 |
| P64_s4 | 0.20 | 0.6 | 10946 | 3909 / 3909 | 0 | 2347 | 8599 | 5399 | 7192 | 1.755 / 1.228 / 3.116 |
| P64_s4 | 0.20 | 1 | 10946 | 8845 / 8845 | 0 | 3770 | 7176 | 470 | 860 | 1.839 / 1.571 / 2.946 |
| P64_s2 | 0.20 | 0.2 | 11102 | 1741 / 1741 | 0 | 72 | 11030 | 8425 | 9689 | 1.494 / 0.454 / 0.264 |
| P64_s2 | 0.20 | 0.3 | 11102 | 2400 / 2400 | 0 | 1275 | 9827 | 7217 | 8565 | 1.599 / 0.688 / 1.066 |
| P64_s2 | 0.20 | 0.6 | 11102 | 3981 / 3981 | 0 | 2296 | 8806 | 5268 | 7333 | 1.774 / 1.208 / 1.941 |
| P64_s2 | 0.20 | 1 | 11102 | 8717 / 8717 | 0 | 3585 | 7517 | 765 | 1057 | 1.986 / 1.574 / 2.193 |
| P64_s8 | 0.20 | 0.2 | 10201 | 2070 / 2070 | 0 | 60 | 10141 | 7632 | 8466 | 1.269 / 0.471 / 0.845 |
| P64_s8 | 0.20 | 0.3 | 10201 | 2600 / 2600 | 0 | 1133 | 9068 | 5825 | 7407 | 1.291 / 0.711 / 0.973 |
| P64_s8 | 0.20 | 0.6 | 10201 | 5037 / 5037 | 0 | 2482 | 7719 | 2560 | 5777 | 1.748 / 1.235 / 1.267 |
| P64_s8 | 0.20 | 1 | 10201 | 8644 / 8644 | 0 | 3587 | 6614 | 67 | 142 | 1.770 / 1.589 / 0.791 |
| P64_s4_tau4 | 0.20 | 0.2 | 20807 | 4019 / 4019 | 0 | 53 | 20754 | 16353 | 17653 | 1.638 / 0.473 / 1.500 |
| P64_s4_tau4 | 0.20 | 0.3 | 20807 | 4902 / 4902 | 0 | 2139 | 18668 | 13666 | 15511 | 1.649 / 0.708 / 1.001 |
| P64_s4_tau4 | 0.20 | 0.6 | 20807 | 7662 / 7662 | 0 | 4616 | 16191 | 10486 | 13473 | 1.779 / 1.238 / 1.760 |
| P64_s4_tau4 | 0.20 | 1 | 20807 | 16633 / 16633 | 0 | 7408 | 13399 | 439 | 1058 | 1.912 / 1.576 / 2.057 |
| W2 | 0.20 | 0.2 | 10614 | 2421 / 2421 | 0 | 21 | 10593 | 8495 | 9022 | 1.664 / 0.484 / 1.484 |
| W2 | 0.20 | 0.3 | 10614 | 2710 / 2710 | 0 | 986 | 9628 | 6971 | 8001 | 1.654 / 0.716 / 1.638 |
| W2 | 0.20 | 0.6 | 10614 | 4315 / 4315 | 0 | 2266 | 8348 | 4665 | 6938 | 1.847 / 1.252 / 1.543 |
| W2 | 0.20 | 1 | 10614 | 8591 / 8591 | 0 | 3700 | 6914 | 246 | 514 | 1.947 / 1.574 / 1.994 |
| S4_128_s8 | 0.20 | 0.2 | 22294 | 4545 / 4545 | 0 | 3 | 22291 | 17962 | 19137 | 1.618 / 0.470 / 3.179 |
| S4_128_s8 | 0.20 | 0.3 | 22294 | 5022 / 5022 | 0 | 2054 | 20240 | 15622 | 17090 | 1.735 / 0.705 / 1.119 |
| S4_128_s8 | 0.20 | 0.6 | 22294 | 8054 / 8054 | 0 | 4429 | 17865 | 11831 | 14784 | 1.854 / 1.235 / 1.659 |
| S4_128_s8 | 0.20 | 1 | 22294 | 17958 / 17958 | 0 | 7321 | 14973 | 458 | 1346 | 1.872 / 1.517 / 3.692 |
| S4_128_s16 | 0.20 | 0.2 | 20797 | 4473 / 4473 | 0 | 41 | 20756 | 15849 | 17341 | 1.585 / 0.484 / 0.638 |
| S4_128_s16 | 0.20 | 0.3 | 20797 | 4982 / 4982 | 0 | 2018 | 18779 | 13090 | 15326 | 1.629 / 0.714 / 0.882 |
| S4_128_s16 | 0.20 | 0.6 | 20797 | 10336 / 10336 | 0 | 4722 | 16075 | 4725 | 11021 | 1.692 / 1.236 / 1.339 |
| S4_128_s16 | 0.20 | 1 | 20797 | 17555 / 17555 | 0 | 7297 | 13500 | 19 | 235 | 1.829 / 1.461 / 1.334 |
| P64_s4_tau2 | 0.20 | 0.2 | 21049 | 2922 / 4808 | 0 | 0 | 21049 | 9305 | 15029 | 0.457 / - / 0.130 |
| P64_s4_tau2 | 0.20 | 0.3 | 21049 | 4668 / 7635 | 0 | 0 | 21049 | 6476 | 11428 | 0.548 / - / 0.309 |
| P64_s4_tau2 | 0.20 | 0.6 | 21049 | 6104 / 10264 | 0 | 273 | 20776 | 2416 | 9446 | 0.551 / 0.557 / 0.390 |
| P64_s4_tau2 | 0.20 | 1 | 21049 | 6572 / 14864 | 0 | 745 | 20304 | 56 | 391 | 0.388 / 0.350 / 0.190 |
| ISO_64_s4 | 0.20 | 0.2 | 11003 | 879 / 1848 | 0 | 3 | 11000 | 8525 | 9344 | 0.453 / - / 0.127 |
| ISO_64_s4 | 0.20 | 0.3 | 11003 | 1316 / 2479 | 0 | 1203 | 9800 | 7300 | 8211 | 0.532 / - / 0.302 |
| ISO_64_s4 | 0.20 | 0.6 | 11003 | 1690 / 4125 | 0 | 2382 | 8621 | 5355 | 7149 | 0.530 / 0.546 / 0.386 |
| ISO_64_s4 | 0.20 | 1 | 11003 | 1813 / 8896 | 0 | 3873 | 7130 | 484 | 847 | 0.376 / 0.343 / 0.187 |
| ISO_128_s4 | 0.20 | 0.2 | 5716 | 1376 / 1391 | 0 | 0 | 5716 | 0 | 2392 | 0.451 / - / 0.115 |
| ISO_128_s4 | 0.20 | 0.3 | 5716 | 2191 / 2207 | 0 | 0 | 5716 | 0 | 964 | 0.522 / - / 0.305 |
| ISO_128_s4 | 0.20 | 0.6 | 5716 | 2859 / 2882 | 0 | 0 | 5716 | 0 | 443 | 0.506 / 0.524 / 0.374 |
| ISO_128_s4 | 0.20 | 1 | 5716 | 3037 / 3061 | 0 | 0 | 5716 | 0 | 5 | 0.337 / 0.327 / 0.180 |
| S4_128_s4 | 0.20 | 0.2 | 9719 | 81 / 2243 | 0 | 0 | 9719 | 4980 | 6851 | 0.431 / - / -0.086 |
| S4_128_s4 | 0.20 | 0.3 | 9719 | 162 / 3491 | 0 | 0 | 9719 | 4020 | 5146 | 0.516 / - / 0.053 |
| S4_128_s4 | 0.20 | 0.6 | 9719 | 266 / 4787 | 0 | 10 | 9709 | 1129 | 4149 | 0.505 / - / 0.074 |
| S4_128_s4 | 0.20 | 1 | 9719 | 318 / 6621 | 0 | 68 | 9651 | 84 | 331 | 0.336 / - / -0.070 |
| ISO_128_s4_plain | 0.20 | 0.2 | 6907 | 70 / 1891 | 0 | 0 | 6907 | 1428 | 3583 | 0.430 / - / -0.095 |
| ISO_128_s4_plain | 0.20 | 0.3 | 6907 | 136 / 2834 | 0 | 0 | 6907 | 841 | 2118 | 0.516 / - / 0.047 |
| ISO_128_s4_plain | 0.20 | 0.6 | 6907 | 225 / 3690 | 0 | 0 | 6907 | 3 | 1486 | 0.506 / - / 0.066 |
| ISO_128_s4_plain | 0.20 | 1 | 6907 | 272 / 4066 | 0 | 0 | 6907 | 0 | 56 | 0.332 / - / -0.075 |
| M3c_s4_96 | 0.10 | 0.2 | 7361 | 4259 / 4259 | 0 | 0 | 7361 | 0 | 984 | 0.624 / - / 0.462 |
| M3c_s4_96 | 0.10 | 0.3 | 7361 | 4576 / 4576 | 0 | 64 | 7297 | 0 | 797 | 0.806 / - / 0.561 |
| M3c_s4_96 | 0.10 | 0.6 | 7361 | 5038 / 5038 | 0 | 403 | 6958 | 0 | 143 | 0.720 / 0.604 / 0.488 |
| M3c_s4_96 | 0.10 | 1 | 7361 | 5180 / 5180 | 0 | 441 | 6920 | 0 | 1 | 0.589 / 0.362 / 0.175 |
| P64_s4_newton | 0.20 | 0.2 | 2840 | 712 / 712 | 0 | 0 | 2840 | 0 | 1305 | 0.455 / - / 0.125 |
| P64_s4_newton | 0.20 | 0.3 | 2840 | 1139 / 1139 | 0 | 0 | 2840 | 0 | 402 | 0.539 / - / 0.297 |
| P64_s4_newton | 0.20 | 0.6 | 2840 | 1489 / 1489 | 0 | 0 | 2840 | 0 | 188 | 0.535 / - / 0.368 |
| P64_s4_newton | 0.20 | 1 | 2840 | 1595 / 1595 | 0 | 0 | 2840 | 0 | 0 | 0.375 / - / 0.167 |

**Table 4.4-9. reviewer A R7's heavy bound on the histories.** ν(m) = max_p q/ε of the radial chain at j = 0 (computed); ν_3 = 0.891 as reviewer A states it for m a 0.2. For the normal-comoving launches (A clocks and B root launches, D_e = 1) received at 4 r_s by T_cert, split by where they were emitted: outside the long-wave root (N_e + v_e > 0; R7 states the bound at or outside the root) and at or inside it. Per group: received, how many have Z_F below N_e(1 − ν) (N_e the clock at the emission event) with the smallest Z_F/(N_e(1 − ν)), and the same against ν_3. Then the material emitters (dust, Z_F carries 1/D_e): received, below N_e(1 − ν), and their largest D_e. Last, the median |ln(Ω_D/Ω_e)| over received launches with Ω_e > 0.

| member | ap_e | ν(m) | A, B outside: received / below ν (smallest ratio) / below ν_3 (smallest ratio) | A, B at or inside: received / below ν (smallest ratio) / below ν_3 (smallest ratio) | dust: received / below ν / largest D_e among those | median abs ln(Ω_D/Ω_e) |
|---|---|---|---|---|---|---|
| P64_s4 | 0.2 | 0.8198 | 1533 / 0 (2.443) / 0 (4.038) | 1 / 0 (4.012) / 0 (6.632) | 197 / 1 / 13.696 | 0.074 |
| P64_s4 | 0.3 | 0.8198 | 2213 / 0 (2.529) / 0 (4.181) | 0 / 0 (-) / 0 (-) | 263 / 2 / 13.696 | 0.081 |
| P64_s4 | 0.6 | 0.8198 | 3566 / 0 (1.640) / 0 (2.712) | 31 / 0 (1.714) / 0 (2.833) | 312 / 2 / 13.696 | 0.054 |
| P64_s4 | 1 | 0.8198 | 7391 / 0 (1.262) / 0 (2.087) | 1088 / 0 (1.902) / 0 (3.144) | 366 / 1 / 13.696 | 0.021 |
| P64_s2 | 0.2 | 0.8198 | 1599 / 0 (2.471) / 0 (4.085) | 6 / 0 (3.776) / 0 (6.242) | 136 / 0 / - | 0.071 |
| P64_s2 | 0.3 | 0.8198 | 2170 / 0 (2.512) / 0 (4.152) | 0 / 0 (-) / 0 (-) | 230 / 0 / - | 0.079 |
| P64_s2 | 0.6 | 0.8198 | 3623 / 0 (1.163) / 0 (1.922) | 18 / 0 (1.705) / 0 (2.819) | 340 / 1 / 2.217 | 0.050 |
| P64_s2 | 1 | 0.8198 | 7254 / 2 (0.911) / 0 (1.506) | 1108 / 0 (1.810) / 0 (2.991) | 355 / 1 / 3.200 | 0.016 |
| P64_s8 | 0.2 | 0.8198 | 1888 / 0 (2.869) / 0 (4.742) | 0 / 0 (-) / 0 (-) | 182 / 0 / - | 0.096 |
| P64_s8 | 0.3 | 0.8198 | 2349 / 0 (2.727) / 0 (4.508) | 0 / 0 (-) / 0 (-) | 251 / 0 / - | 0.097 |
| P64_s8 | 0.6 | 0.8198 | 4710 / 0 (1.425) / 0 (2.356) | 0 / 0 (-) / 0 (-) | 327 / 0 / - | 0.058 |
| P64_s8 | 1 | 0.8198 | 8030 / 0 (1.595) / 0 (2.638) | 193 / 0 (2.194) / 0 (3.627) | 421 / 0 / - | 0.013 |
| P64_s4_tau4 | 0.2 | 0.8198 | 3635 / 0 (2.355) / 0 (3.893) | 56 / 0 (3.514) / 0 (5.809) | 328 / 0 / - | 0.099 |
| P64_s4_tau4 | 0.3 | 0.8198 | 4463 / 0 (2.395) / 0 (3.959) | 2 / 0 (3.290) / 0 (5.439) | 437 / 0 / - | 0.094 |
| P64_s4_tau4 | 0.6 | 0.8198 | 7077 / 0 (1.596) / 0 (2.638) | 91 / 0 (1.713) / 0 (2.831) | 494 / 1 / 1.700 | 0.065 |
| P64_s4_tau4 | 1 | 0.8198 | 14591 / 0 (1.259) / 0 (2.081) | 1413 / 0 (1.928) / 0 (3.188) | 629 / 2 / 2.800 | 0.026 |
| W2 | 0.2 | 0.8198 | 2185 / 0 (2.392) / 0 (3.954) | 67 / 0 (3.503) / 0 (5.791) | 169 / 0 / - | 0.125 |
| W2 | 0.3 | 0.8198 | 2464 / 0 (2.318) / 0 (3.831) | 30 / 0 (2.822) / 0 (4.666) | 216 / 0 / - | 0.116 |
| W2 | 0.6 | 0.8198 | 4009 / 0 (1.611) / 0 (2.664) | 49 / 0 (1.793) / 0 (2.964) | 257 / 0 / - | 0.089 |
| W2 | 1 | 0.8198 | 7624 / 0 (1.140) / 0 (1.884) | 643 / 0 (1.937) / 0 (3.202) | 324 / 1 / 2.898 | 0.029 |
| S4_128_s8 | 0.2 | 0.8198 | 4106 / 0 (2.298) / 0 (3.799) | 100 / 0 (3.506) / 0 (5.797) | 339 / 3 / 15.935 | 0.112 |
| S4_128_s8 | 0.3 | 0.8198 | 4590 / 0 (2.161) / 0 (3.572) | 16 / 0 (2.834) / 0 (4.684) | 416 / 0 / - | 0.098 |
| S4_128_s8 | 0.6 | 0.8198 | 7388 / 0 (1.629) / 0 (2.693) | 209 / 0 (1.669) / 0 (2.758) | 457 / 0 / - | 0.069 |
| S4_128_s8 | 1 | 0.8198 | 15487 / 0 (1.795) / 0 (2.967) | 1874 / 0 (2.124) / 0 (3.512) | 597 / 4 / 15.935 | 0.031 |
| S4_128_s16 | 0.2 | 0.8198 | 4021 / 0 (2.341) / 0 (3.870) | 79 / 0 (3.529) / 0 (5.834) | 373 / 0 / - | 0.109 |
| S4_128_s16 | 0.3 | 0.8198 | 4554 / 0 (2.241) / 0 (3.705) | 0 / 0 (-) / 0 (-) | 428 / 0 / - | 0.098 |
| S4_128_s16 | 0.6 | 0.8198 | 9759 / 0 (1.620) / 0 (2.679) | 0 / 0 (-) / 0 (-) | 577 / 0 / - | 0.067 |
| S4_128_s16 | 1 | 0.8198 | 16463 / 0 (1.438) / 0 (2.378) | 348 / 0 (2.380) / 0 (3.935) | 744 / 0 / - | 0.016 |
| P64_s4_tau2 | 0.2 | 0.8198 | 2596 / 0 (3.634) / 0 (6.008) | 0 / 0 (-) / 0 (-) | 326 / 0 / - | 0.033 |
| P64_s4_tau2 | 0.3 | 0.8198 | 4180 / 0 (3.355) / 0 (5.546) | 0 / 0 (-) / 0 (-) | 488 / 0 / - | 0.048 |
| P64_s4_tau2 | 0.6 | 0.8198 | 5518 / 0 (3.187) / 0 (5.269) | 0 / 0 (-) / 0 (-) | 586 / 0 / - | 0.038 |
| P64_s4_tau2 | 1 | 0.8198 | 5942 / 0 (3.921) / 0 (6.482) | 0 / 0 (-) / 0 (-) | 630 / 0 / - | 0.013 |
| ISO_64_s4 | 0.2 | 0.8198 | 792 / 0 (3.634) / 0 (6.007) | 0 / 0 (-) / 0 (-) | 87 / 0 / - | 0.039 |
| ISO_64_s4 | 0.3 | 0.8198 | 1186 / 0 (3.355) / 0 (5.546) | 0 / 0 (-) / 0 (-) | 130 / 0 / - | 0.053 |
| ISO_64_s4 | 0.6 | 0.8198 | 1534 / 0 (3.222) / 0 (5.327) | 0 / 0 (-) / 0 (-) | 156 / 0 / - | 0.042 |
| ISO_64_s4 | 1 | 0.8198 | 1646 / 0 (3.947) / 0 (6.525) | 0 / 0 (-) / 0 (-) | 167 / 0 / - | 0.014 |
| ISO_128_s4 | 0.2 | 0.8198 | 1228 / 0 (3.637) / 0 (6.012) | 0 / 0 (-) / 0 (-) | 148 / 0 / - | 0.033 |
| ISO_128_s4 | 0.3 | 0.8198 | 1967 / 0 (3.359) / 0 (5.553) | 0 / 0 (-) / 0 (-) | 224 / 0 / - | 0.047 |
| ISO_128_s4 | 0.6 | 0.8198 | 2588 / 0 (3.288) / 0 (5.436) | 0 / 0 (-) / 0 (-) | 271 / 0 / - | 0.035 |
| ISO_128_s4 | 1 | 0.8198 | 2744 / 0 (4.004) / 0 (6.619) | 0 / 0 (-) / 0 (-) | 293 / 0 / - | 0.011 |
| S4_128_s4 | 0.2 | 0.8198 | 69 / 0 (3.640) / 0 (6.018) | 0 / 0 (-) / 0 (-) | 12 / 0 / - | 7.18e-05 |
| S4_128_s4 | 0.3 | 0.8198 | 128 / 0 (3.363) / 0 (5.559) | 0 / 0 (-) / 0 (-) | 34 / 0 / - | 5.18e-04 |
| S4_128_s4 | 0.6 | 0.8198 | 206 / 0 (3.400) / 0 (5.620) | 0 / 0 (-) / 0 (-) | 60 / 0 / - | 7.00e-04 |
| S4_128_s4 | 1 | 0.8198 | 249 / 0 (4.097) / 0 (6.773) | 0 / 0 (-) / 0 (-) | 69 / 0 / - | 2.54e-04 |
| ISO_128_s4_plain | 0.2 | 0.8198 | 63 / 0 (3.637) / 0 (6.013) | 0 / 0 (-) / 0 (-) | 7 / 0 / - | 5.77e-05 |
| ISO_128_s4_plain | 0.3 | 0.8198 | 110 / 0 (3.359) / 0 (5.553) | 0 / 0 (-) / 0 (-) | 26 / 0 / - | 2.77e-04 |
| ISO_128_s4_plain | 0.6 | 0.8198 | 173 / 0 (3.396) / 0 (5.614) | 0 / 0 (-) / 0 (-) | 52 / 0 / - | 4.53e-04 |
| ISO_128_s4_plain | 1 | 0.8198 | 210 / 0 (4.095) / 0 (6.769) | 0 / 0 (-) / 0 (-) | 62 / 0 / - | 1.10e-04 |
| M3c_s4_96 | 0.2 | 0.9050 | 3877 / 0 (5.983) / 0 (5.215) | 0 / 0 (-) / 0 (-) | 382 / 0 / - | 0.100 |
| M3c_s4_96 | 0.3 | 0.9050 | 4172 / 0 (5.829) / 0 (5.081) | 0 / 0 (-) / 0 (-) | 404 / 0 / - | 0.100 |
| M3c_s4_96 | 0.6 | 0.9050 | 4609 / 0 (5.781) / 0 (5.039) | 0 / 0 (-) / 0 (-) | 429 / 0 / - | 0.076 |
| M3c_s4_96 | 1 | 0.9050 | 4731 / 0 (7.363) / 0 (6.418) | 0 / 0 (-) / 0 (-) | 449 / 0 / - | 0.038 |
| P64_s4_newton | 0.2 | 0.8198 | 630 / 0 (3.634) / 0 (6.008) | 0 / 0 (-) / 0 (-) | 82 / 0 / - | 0.031 |
| P64_s4_newton | 0.3 | 0.8198 | 1018 / 0 (3.355) / 0 (5.546) | 0 / 0 (-) / 0 (-) | 121 / 0 / - | 0.046 |
| P64_s4_newton | 0.6 | 0.8198 | 1343 / 0 (3.391) / 0 (5.606) | 0 / 0 (-) / 0 (-) | 146 / 0 / - | 0.036 |
| P64_s4_newton | 1 | 0.8198 | 1439 / 0 (4.090) / 0 (6.762) | 0 / 0 (-) / 0 (-) | 156 / 0 / - | 0.012 |

##### What the heavy tables show (numbers as printed above; no ruling)

1. **Fates, summed over the members.** Launches per ap_e = 0.2 / 0.3 / 0.6 / 1: 171356 / 171356 / 171356 / 171356. Received at 4 r_s by T_cert: 31299 / 39280 / 60965 / 105730. Fell to the centre: 0 / 0 / 0 / 0. Bound at infinity at emission (Ω_e < m): 130197 / 109216 / 89522 / 6843. p changed sign along the path: 107475 / 88318 / 53837 / 3088.
2. **The deepest certified heavy redshifts.** The largest certified ln(1/Z_F) at 4 r_s, ranging over members: comoving clocks 0.430 to 1.664; 0.516 to 1.735; 0.505 to 1.854; 0.332 to 1.986; root launches 0.454 to 0.484; 0.688 to 0.716; 0.524 to 1.252; 0.327 to 1.589 (per ap_e = 0.2 / 0.3 / 0.6 / 1).
3. **reviewer A R7's heavy bound, Z ≥ N(1 − ν).** Normal-comoving emitters (A clocks and B root launches, D_e = 1), received by T_cert, summed over members: 28531 / 35612 / 56501 / 100468 per ap_e = 0.2 / 0.3 / 0.6 / 1. With the chain's own ν(m) (0.8198, 0.9050 for the members' masses), 0 / 0 / 0 / 2 of them lie below N_e(1 − ν), and the smallest Z_F/(N_e(1 − ν)) ranges 2.298 to 5.983; 2.161 to 5.829; 1.163 to 5.781; 0.911 to 7.363. With reviewer A's ν_3 = 0.891: 0 / 0 / 0 / 0 below; smallest ratio 3.799 to 6.632; 3.572 to 5.559; 1.922 to 5.620; 1.506 to 6.773. The material emitters (dust, Z_F carrying 1/D_e) are outside R7's frame: of 2768 / 3668 / 4464 / 5262 received, 4 / 2 / 4 / 9 lie below N_e(1 − ν), and the largest D_e among those ranges 1.700 to 15.935 over the (member, ap_e) cells (Table 4.4-9).
4. **Ω along the heavy paths.** The median |ln(Ω_D/Ω_e)| over certified received launches with Ω_e > 0 ranges 5.77e-05 to 0.125 across members and ap_e.

**Still not covered.** reviewer B 4.5's packets and reviewer A route 2; heavy channels with j > 1; the heavy pass on variants other than the main one.

### 9.2 The heavy channel's crossings reconciled (rz_v4_heavyx_v1.py; out/v4_heavyx_v1/heavyx_tables.md, verbatim)

(v4, in deliveries/route2/REDSHIFT_received_v4_scripts/) Producing script: code/rz_v4_heavyx_v1.py (md5 6c6cd325f2a3b39dd173ebf3ed3a8ede). Table file: out/v4_heavyx_v1/heavyx_tables.md (md5 17e6ce76dea9d9ad31dac2f002846458). Launcher code/v4tables_run1.sh. Data: the rz_disp_v2 outputs on the box (out/disp_v2/), Table 4.4-8's json (v1's out/disp_tables_v1/disp_heavy_v1.json) and v3's out/v3_misc_v1/misc3_tables.json. v3's Table HX (REDSHIFT_received_v3.1 sec. 9.1) is superseded by Table HXv4, whose 'crossed inside' column reproduces it.

**Table HXv4. The heavy channel's crossings reconciled with Table 4.4-8** (rz_v4_heavyx_v1.py on the rz_disp_v2 outputs; AstraAPI_rzchk4 sec. 3.6, FableAPI_rzchk4 item 8). Per member and ap_e, at 4 r_s, on common launch ids. Predicates: CROSSED INSIDE (v3's Table HX) = the first crossing time of 4 r_s is finite and inside the solver-certified record interval; RECEIVED CERT (Table 4.4-8) = crossed inside AND ln(1/Z_F) finite, i.e. Ω_D/ε(p_e) > 0 at the crossing and D_e > 0 (D_e = 1 for the A and B launches). The crossings 4.4-8 does not count are split by the first condition they fail: Ω_D not finite / Ω_D ≤ 0 at the crossing / ε(p_e) ≤ 0 or not finite / D_e not positive (dust). Both totals are checked against v3's HX json and the 4.4-8 json ('=' where they agree). Then the received-cert launches by final status (left the chain / still inside at the end of the records / fell to r ≤ 0.5 a) and by class (A / B / C).

| member | ap_e | crossed inside (v3 HX) | received cert (4.4-8) | not counted by 4.4-8: Ω_D not finite / Ω_D ≤ 0 / ε(p_e) ≤ 0 / D_e not positive | of those, by class A / B / C | received cert by final status: left / inside / fell | received cert by class: A / B / C |
|---|---|---|---|---|---|---|---|
| ISO_128_s4 | 0.2 | 1376 (=) | 1376 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 1376 / 0 | 1228 / 0 / 148 |
| ISO_128_s4 | 0.3 | 2191 (=) | 2191 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2191 / 0 | 1967 / 0 / 224 |
| ISO_128_s4 | 0.6 | 2859 (=) | 2859 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2859 / 0 | 2586 / 2 / 271 |
| ISO_128_s4 | 1 | 3037 (=) | 3037 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 3037 / 0 | 2740 / 4 / 293 |
| ISO_128_s4_plain | 0.2 | 70 (=) | 70 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 70 / 0 | 63 / 0 / 7 |
| ISO_128_s4_plain | 0.3 | 136 (=) | 136 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 136 / 0 | 110 / 0 / 26 |
| ISO_128_s4_plain | 0.6 | 225 (=) | 225 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 225 / 0 | 173 / 0 / 52 |
| ISO_128_s4_plain | 1 | 272 (=) | 272 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 272 / 0 | 210 / 0 / 62 |
| ISO_64_s4 | 0.2 | 879 (=) | 879 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 879 / 0 | 792 / 0 / 87 |
| ISO_64_s4 | 0.3 | 1316 (=) | 1316 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 1142 / 174 / 0 | 1186 / 0 / 130 |
| ISO_64_s4 | 0.6 | 1690 (=) | 1690 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 1690 / 0 / 0 | 1532 / 2 / 156 |
| ISO_64_s4 | 1 | 1813 (=) | 1813 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 1813 / 0 / 0 | 1642 / 4 / 167 |
| M3c_s4_96 | 0.2 | 4259 (=) | 4259 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 4259 / 0 | 3877 / 0 / 382 |
| M3c_s4_96 | 0.3 | 4576 (=) | 4576 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 63 / 4513 / 0 | 4172 / 0 / 404 |
| M3c_s4_96 | 0.6 | 5038 (=) | 5038 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 402 / 4636 / 0 | 4608 / 1 / 429 |
| M3c_s4_96 | 1 | 5180 (=) | 5180 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 440 / 4740 / 0 | 4727 / 4 / 449 |
| P64_s2 | 0.2 | 1741 (=) | 1741 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 72 / 1669 / 0 | 1434 / 171 / 136 |
| P64_s2 | 0.3 | 2401 (=) | 2400 (=) | 0 / 0 / 0 / 1 | 0 / 0 / 1 | 1274 / 1126 / 0 | 1824 / 346 / 230 |
| P64_s2 | 0.6 | 3994 (=) | 3981 (=) | 0 / 0 / 0 / 13 | 0 / 0 / 13 | 2295 / 1686 / 0 | 2369 / 1272 / 340 |
| P64_s2 | 1 | 8763 (=) | 8717 (=) | 0 / 0 / 0 / 46 | 0 / 0 / 46 | 3565 / 5152 / 0 | 2831 / 5531 / 355 |
| P64_s4 | 0.2 | 1754 (=) | 1731 (=) | 0 / 0 / 0 / 23 | 0 / 0 / 23 | 31 / 1700 / 0 | 1404 / 130 / 197 |
| P64_s4 | 0.3 | 2510 (=) | 2476 (=) | 0 / 0 / 0 / 34 | 0 / 0 / 34 | 1139 / 1337 / 0 | 1851 / 362 / 263 |
| P64_s4 | 0.6 | 3958 (=) | 3909 (=) | 0 / 0 / 0 / 49 | 0 / 0 / 49 | 2346 / 1563 / 0 | 2391 / 1206 / 312 |
| P64_s4 | 1 | 8943 (=) | 8845 (=) | 0 / 0 / 0 / 98 | 0 / 0 / 98 | 3754 / 5091 / 0 | 2842 / 5637 / 366 |
| P64_s4_newton | 0.2 | 712 (=) | 712 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 712 / 0 | 630 / 0 / 82 |
| P64_s4_newton | 0.3 | 1139 (=) | 1139 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 1139 / 0 | 1018 / 0 / 121 |
| P64_s4_newton | 0.6 | 1489 (=) | 1489 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 1489 / 0 | 1343 / 0 / 146 |
| P64_s4_newton | 1 | 1595 (=) | 1595 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 1595 / 0 | 1439 / 0 / 156 |
| P64_s4_tau2 | 0.2 | 2922 (=) | 2922 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2922 / 0 | 2596 / 0 / 326 |
| P64_s4_tau2 | 0.3 | 4668 (=) | 4668 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 4668 / 0 | 4180 / 0 / 488 |
| P64_s4_tau2 | 0.6 | 6104 (=) | 6104 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 272 / 5832 / 0 | 5517 / 1 / 586 |
| P64_s4_tau2 | 1 | 6572 (=) | 6572 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 744 / 5828 / 0 | 5935 / 7 / 630 |
| P64_s4_tau4 | 0.2 | 4052 (=) | 4019 (=) | 0 / 0 / 0 / 33 | 0 / 0 / 33 | 53 / 3966 / 0 | 3031 / 660 / 328 |
| P64_s4_tau4 | 0.3 | 4946 (=) | 4902 (=) | 0 / 0 / 0 / 44 | 0 / 0 / 44 | 2138 / 2764 / 0 | 3631 / 834 / 437 |
| P64_s4_tau4 | 0.6 | 7732 (=) | 7662 (=) | 0 / 0 / 0 / 70 | 0 / 0 / 70 | 4615 / 3047 / 0 | 4606 / 2562 / 494 |
| P64_s4_tau4 | 1 | 16800 (=) | 16633 (=) | 0 / 0 / 0 / 167 | 0 / 0 / 167 | 7362 / 9271 / 0 | 5057 / 10947 / 629 |
| P64_s8 | 0.2 | 2079 (=) | 2070 (=) | 0 / 0 / 0 / 9 | 0 / 0 / 9 | 60 / 2010 / 0 | 1753 / 135 / 182 |
| P64_s8 | 0.3 | 2610 (=) | 2600 (=) | 0 / 0 / 0 / 10 | 0 / 0 / 10 | 1132 / 1468 / 0 | 2056 / 293 / 251 |
| P64_s8 | 0.6 | 5077 (=) | 5037 (=) | 0 / 0 / 0 / 40 | 0 / 0 / 40 | 2481 / 2556 / 0 | 2549 / 2161 / 327 |
| P64_s8 | 1 | 8759 (=) | 8644 (=) | 0 / 0 / 0 / 115 | 0 / 0 / 115 | 3586 / 5058 / 0 | 2646 / 5577 / 421 |
| S4_128_s16 | 0.2 | 4480 (=) | 4473 (=) | 0 / 0 / 0 / 7 | 0 / 0 / 7 | 41 / 4432 / 0 | 3077 / 1023 / 373 |
| S4_128_s16 | 0.3 | 5010 (=) | 4982 (=) | 0 / 0 / 0 / 28 | 0 / 0 / 28 | 2017 / 2965 / 0 | 3769 / 785 / 428 |
| S4_128_s16 | 0.6 | 10422 (=) | 10336 (=) | 0 / 0 / 0 / 86 | 0 / 0 / 86 | 4721 / 5615 / 0 | 4987 / 4772 / 577 |
| S4_128_s16 | 1 | 17796 (=) | 17555 (=) | 0 / 0 / 0 / 241 | 0 / 0 / 241 | 7296 / 10259 / 0 | 5306 / 11505 / 744 |
| S4_128_s4 | 0.2 | 81 (=) | 81 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 81 / 0 | 69 / 0 / 12 |
| S4_128_s4 | 0.3 | 162 (=) | 162 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 0 / 162 / 0 | 128 / 0 / 34 |
| S4_128_s4 | 0.6 | 266 (=) | 266 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 9 / 257 / 0 | 206 / 0 / 60 |
| S4_128_s4 | 1 | 318 (=) | 318 (=) | 0 / 0 / 0 / 0 | 0 / 0 / 0 | 67 / 251 / 0 | 249 / 0 / 69 |
| S4_128_s8 | 0.2 | 4571 (=) | 4545 (=) | 0 / 0 / 0 / 26 | 0 / 0 / 26 | 3 / 4542 / 0 | 2965 / 1241 / 339 |
| S4_128_s8 | 0.3 | 5052 (=) | 5022 (=) | 0 / 0 / 0 / 30 | 0 / 0 / 30 | 2053 / 2969 / 0 | 3466 / 1140 / 416 |
| S4_128_s8 | 0.6 | 8119 (=) | 8054 (=) | 0 / 0 / 0 / 65 | 0 / 0 / 65 | 4428 / 3626 / 0 | 4758 / 2839 / 457 |
| S4_128_s8 | 1 | 18130 (=) | 17958 (=) | 0 / 0 / 0 / 172 | 0 / 0 / 172 | 7262 / 10696 / 0 | 5775 / 11586 / 597 |
| W2 | 0.2 | 2438 (=) | 2421 (=) | 0 / 0 / 0 / 17 | 0 / 0 / 17 | 21 / 2400 / 0 | 1691 / 561 / 169 |
| W2 | 0.3 | 2727 (=) | 2710 (=) | 0 / 0 / 0 / 17 | 0 / 0 / 17 | 985 / 1725 / 0 | 1857 / 637 / 216 |
| W2 | 0.6 | 4350 (=) | 4315 (=) | 0 / 0 / 0 / 35 | 0 / 0 / 35 | 2265 / 2050 / 0 | 2330 / 1728 / 257 |
| W2 | 1 | 8678 (=) | 8591 (=) | 0 / 0 / 0 / 87 | 0 / 0 / 87 | 3671 / 4920 / 0 | 2698 / 5569 / 324 |

Over the 56 (member, ap_e) rows: 1563 crossings inside the interval are not counted by Table 4.4-8 (Ω_D not finite 0; Ω_D ≤ 0 at the crossing 0; ε(p_e) ≤ 0 or not finite 0; D_e not positive 1563; by class A 0, B 0, C 1563). Crossed-inside totals equal v3's HX on every row: yes; received-cert totals equal Table 4.4-8 on every row: yes.

## 10. Production timing (reviewer B rzchk p. 7, exact change 9)

Read from <internal>/engine_collapse_v36d.py (md5 0315e0826ee872ee194474876ef523f9), by rz_v2_timing_v1.py; line numbers as in that file.

- **The step loop** (line 2152): `for step in range(step0, nsteps + 1):`
- **The record's source densities are those of the current state st, the state at the record's clock T** (lines 2160 to 2163):
```
 2160          base = {}
 2161          for ch in chs:
 2162              s_ = st[ch.kappa]
 2163              base[ch.kappa] = (dens(ch.h, s_["sea"]), dens(ch.Q, s_["sea"]), dens(ch.h, s_["dust"], s_["f"]), dens(ch.Q, s_["dust"], s_["f"]))
```
- **The solve at the record starts from the previous applied fields and the CURRENT state's sources** (line 2198): `N_app, v_app = f_ref[0].copy(), f_ref[1].copy()`; the sources are the densities of the state at the record's clock T (the `sources(cache_)` function above it), and the accepted iterate is the record's N_applied, v_applied.
- **The record writes those applied fields** (line 2583): `fields["N_applied"].append(N_app.copy()); fields["v_applied"].append(v_app.copy())`
- **Then the state is propagated over one step with them** (lines 2653 to 2664):
```
 2652          for ch in chs:
 2653              _hm, e, U, ref, _ci = cache[ch.kappa]
 2654              ph = torch.exp(-1j * e.to(torch.complex128) * a.tau)
 2655              s_ = st[ch.kappa]
 2656              prop = lambda X: U @ (ph[:, None] * (U.conj().T @ X))
 2657              s_["sea"] = prop(s_["sea"])
 2658              if s_["dust"].shape[1]:
 2659                  s_["dust"] = prop(s_["dust"])
 2660              cache[ch.kappa] = (None, None, None, ref, _ci)
 2661              del e, U
 2662          sync_dev()
 2663          last_prop_s = time.time() - t_p
 2664          T += a.tau
```
- **The only midpoint in the solve** is the source continuation's stage 1, a fallback for a record whose direct solve fails its stop rule (docstring, line 136): `rule, the fixed point is solved first (stage 1) with the sources of the midpoint covariance C_prev + (C_new - C_prev)/2 from the previous` The record keeps the iterate of its own map (stage 2 or the direct solve).

**Reading.** The fields applied over [T_k, T_k + τ] are solved from the state at T_k (the start of the step), written as record k's N_applied, v_applied, and held over the step; the engine does not solve at the step's midpoint. The manuscript's 'fields solved self-consistently at each step's midpoint and held over the step' does not describe this engine (reviewer B p. 7: 'If Input 1's interpretation is right, correct the manuscript'). The replay variant app_hold_cubic follows the engine.

## 11. Declined, not computed, and premises

- Packets (reviewer B 4.5, reviewer A route 2): not computed; no finite-band spectrum, arrival distribution or transmitted weight is claimed (A3-7.7, A4-4.5).
- A bound on the background field error: not established; the stored records do not carry the solve's conditioning, and tighter or independent solves are not available to this lane; Table R2v4 is a sensitivity (A3-3.6, A4-2.5).
- Fixed-emission comparisons by direct location: locating the common (clock, τ_e) event on each background and integrating from it is not run; the comparisons interpolate each map between consecutive arrivals and print every event's sampled-map status; non-receipt is not verified (A4-2.3; AstraAPI_rzchk5).
- Alternative centre treatments and a stencil-by-stencil interpolation-support audit: not run; the masks (e), (c), (r) are applied, (r) now including the emission endpoint (A3-11.4, A4-3.2).
- The propagation variants on the two completions: not run; their headlines are captioned untested under the variants (reviewer A rzchk4 item 7).
- A predeclared uniform blackness experiment (emitter interval, band, accepted histories, censoring rules fixed before looking): not run (A3-11.7, A4-6).
- Persistence and optical-error control over delays growing as r_s ln(r_s/a): premises of the limit (§1 item 10); REDSHIFT_256 is the vehicle (A3-11.8, A4-6).
- The paper's sentences (reviewer A rzchk4 sec. 8, reviewer B rzchk4 sec. 7) and the manuscript's page count (reviewer B rzchk4 sec. 8): the lead's.

## 12. Files and MD5SUMS

deliveries/route2/REDSHIFT_received_v4.1_scripts/ (code/, logs/, out/; MD5SUMS.txt lists every file, md5 of MD5SUMS.txt 3d7719d35908e8b56390e97e39a1ab13): the five label-revised tables scripts (rz_v4_head_v3.py, rz_v4_a2fix_v3.py, rz_v4_variants_v4.py, rz_v3_law_v2.py, rz_v4_dispev_tables_v3.py), their generators (mk_*_patch.py, each made from the v4 or v3 script it names by counted replacements), the launcher v41tables_run1.sh, copyback_v41_v1.sh, this composer, the run's log and the five table outputs with the A2 event export. Everything else is v4's: deliveries/route2/REDSHIFT_received_v4_scripts/ (md5 of its MD5SUMS.txt ecae35d0b74084c466f408c184939f57), unchanged. The box working directory is <internal>/redshift_received_20261022; the npz files and the plain a2_events.csv stay there.

```
7862d86522a9d3ae2b2f503403a8788f  ./code/copyback_v41_v1.sh
152c74ed79c74c63608565fff995b556  ./code/mk_a2fix_v4_v3_patch.py
d38e223dbb53ee268239b9bb0688886f  ./code/mk_dispev_tables_v4_v3_patch.py
6803a16347eec1cbcdfc9c997e71cf21  ./code/mk_head_v4_v3_patch.py
b99d321b4d3f83f22639693f3713be8e  ./code/mk_law_v3_v2_patch.py
5ee46a9b63dc2bb4036d9d30d242eda6  ./code/mk_variants_v4_v4_patch.py
24c74d4ebbbd2e4fd29374d55d6a4a5e  ./code/rz_v3_law_v2.py
d9151321400c94054535ec28abb38042  ./code/rz_v41_compose_v1.py
c1975992e5222cb04aa3f0f6bf876e3a  ./code/rz_v4_a2fix_v3.py
de7d5069fe389f6474f5174289174c46  ./code/rz_v4_dispev_tables_v3.py
d11191e596c9641fa89645406ba34449  ./code/rz_v4_head_v3.py
ab7af70a3dd306ccf5354877b482d15e  ./code/rz_v4_variants_v4.py
f6a2feea9246225a26c442330fc256bc  ./code/v41tables_run1.sh
931c34b8df5db39cc5cdfc3b2df81017  ./logs/v41tables_run1.log
69a2efed000abcf4e722d59cc9214eee  ./out/v41_a2fix_v3/a2_events.csv.gz
df6d4b2c2072417a3a5cf15512c52a27  ./out/v41_a2fix_v3/a2fix_tables.json
0cb295a4a3cbc3d3ad1a66f1a6d48468  ./out/v41_a2fix_v3/a2fix_tables.md
04e95f57c64fd65f118317105a8ee7fd  ./out/v41_dispev_tables_v3/dispev4_tables.json
d079deaba258889972649730d7989a3b  ./out/v41_dispev_tables_v3/dispev4_tables.md
5a8861d706c8a8b354af947e02173784  ./out/v41_head_v3/head_tables.json
5d94c9cfc5df0a8dc38ddeb4974eec5e  ./out/v41_head_v3/head_tables.md
f6f01d48fc336fcb6f36cdfa2c5a003f  ./out/v41_law_v2/law_tables.json
8e69f71756d5f5a50a37de266fffbd46  ./out/v41_law_v2/law_tables.md
09b945dc4ff78e37ca50ed568357a3e4  ./out/v41_variants_v4/variants_tables.json
4a072dc22808d1b12a23830b6c711e35  ./out/v41_variants_v4/variants_tables.md
```


SECTION V4.1 COMPLETE 2026-10-22 07:55:59 UTC
