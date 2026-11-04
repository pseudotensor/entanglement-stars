# COMP2_W7_five_items_v1.2: under reviewer A's gate ruling 1a and rulings 5 and 6

Solve lane, 20 Oct 2026. Supersedes v1.1 (which stays as issued). Producer's reading; NOT YET RED-TEAMED (its red-team list is in the notes below).

**Prints (ruling 1a; the slices the route of record, the propagated set the control; no sentence number differs between the routes at a printed digit):** the certified bound at record 15, slice-only (15 and 56 slices), WEIGHTED 3.0 of 611 (3.0054591; 0.076 of 15.34 per copy in the record); the far count beyond 16 r_s at 56 50.719793 (8 digits on both routes), 0.99669 of it in (m, b] with mean 1.204 m on the production's own held operator; the whole dust's current through 16 r_s summed over channels outward at the five sampled clocks 48 to 56 (8 digits on both routes; 65 of 65 propagated records outward); the half-arrival 39.45 r_s after formation weighted, referred to 56 and still rising (39.62 per copy in the record). Capacity record: the minimum over rooted records 26.34 at m e_min at T/r_s 15.5 (23.10 at m), no verdict. To the record only (they differ at a printed digit): the weight beyond 24 r_s at 56 (0.015945 slices, 0.015955 propagated) and the small Killing bins (to 2.5e-4).

**Correction to the ruling's conditions sentence (FOR FABLE):** the slice-vs-propagated difference after 40 is NOT the propagator drifting from the production engine's own step: the propagator's input changes at 40 only in its files (r11 -> ext100 json and applied fields, the same engine v19, tau, record grid, ell 4, 2048 engine sites; their fields agree over the 97 shared records 28 to 40 to 2.2e-10). The difference is the production CHAIN'S LENGTH: it starts at the production chain's wall at 32 r_s (at 48: 1.44e-4 weighted, peaked at 31.8 r_s, below 4e-8 inside 24 r_s) and travels inward as reflected content (at 56: 9.1e-3, peaked at 19.7 r_s, 1.2e-7 inside 16 r_s). Test: the stored 48 slice propagated to 50 on the production's own 2048-site chain reproduces the stored 50 slice to 1e-13 for every kappa; on the lane's 8192-site chain it misses by about 2e-7. It begins after 40 because the far dust reaches 32 r_s then (beyond 24 r_s: 3.5e-4 at 40, 6.8e-3 at 48). So the sentence's 'drifting from the production engine's own step after 40' should read 'differing by the production chain's reflection from its 32 r_s wall after 40'. Red-team note: the slices (the route of record) carry that wall; by 56 its reflection is inside the printed region (no printed digit of the far count crossed yet; the 24 r_s weight's third digit moved), a reason the record stops at 56.

Box <internal>/W7_v1_2 (W7_record_items_v1_2.json); scripts in the solve worktree solve/code/compositions/.

---

This version applies <internal>/FableAPI_w7_gate_v1.md, together with rulings 5 and 6 of FableAPI_w7_rulings_v1.md.

- **What it recomputes.** No number in v1.1 was recomputed. The v1.1 outputs (<internal>/W7_compose_v1_1.json) carry both routes; v1.2 assigns them to their roles.
- **What is new.** The two record items in section 3, computed on the box CPU from stored data (W7_v1_2/W7_record_items_v1_2.json, md5 65798d28972569c6f9c83a89835108db).
- **Units.** Counts are weighted (16 kappa, 640 dust fermions), with per copy in the record.

## Changes from v1.1

1. **Ruling 1a, printing.** The 48 to 56 numbers computed on the stored slices print. The propagated set is their control. The v1.1 gate (overlap at the slices' storage floor) is withdrawn, not restated.
2. **Ruling 1a, route of record.** At 48 to 56 the slices are the route of record for:
   - the far count beyond 16 r_s;
   - its Killing split on the production's own held operator and chain (2048 sites, to 32 r_s);
   - the current through 16 r_s;
   - p_late in the bound.

   So the bound is slice-only (the 15 and 56 slices), with the propagated route as its check. The propagated set is the route of record only for what no slice holds: the inside-origin piece, the record cadence between sampled clocks (65 of 65, the ledger after 40, the cancellation check, the half-arrival clock 51.575).
3. **Ruling 1a, the sentence condition.** Where the routes differ at a printed digit, both go to the record and neither to the sentence. None of the sentence's numbers differs (section 2).
4. **Ruling 1a, record items.**
   - What changes in the propagator's input at 40 is the record and field files only (section 3.1).
   - The difference sits at the production chain's end and is reflected inward from it. A direct test attributes it to the chain, not the step (sections 3.2 and 3.3).
5. **Ruling 5.** The capacity record carries the minimum over rooted records, 26.34 at m e_min at T/r_s 15.5 (23.10 at m), in place of C(15) = 26.5. No verdict.
6. **Ruling 6.**
   - The sentence's clock is the weighted 39.45 r_s; the record keeps 39.62 per copy.
   - The bound parenthesis is the weighted 3.0 of 611; the record keeps 0.076 of 15.34 per copy.

## 1. Numbers for the paragraph (slices; the propagated control beside)

| T/r_s | beyond 16 r_s, slice | propagated | current J16 summed, slice | propagated |
|---|---|---|---|---|
| 48 | 1.0456843 | 1.0456843 | +0.027211223 | +0.027211223 |
| 50 | 10.720464 | 10.720464 | +0.12684557 | +0.12684557 |
| 52 | 29.370936 | 29.370936 | +0.14404705 | +0.14404705 |
| 54 | 44.019229 | 44.019229 | +0.074376391 | +0.074376391 |
| 56 | 50.719793 | 50.719793 | +0.029526457 | +0.029526457 |

The current is outward at every sampled clock on the slices. The record keeps 65 of 65 propagated records from 48 to 56. Per-channel signs, also in the record: every kappa is outward at every sampled clock, the smallest being kappa 5 at 56 (+4.8e-6 per copy).

At 56, on the full held operator of the production's chain:

| Quantity | Slice | Propagated (8192-site chain) |
|---|---|---|
| (m, b] share of the far count | 0.996692 | 0.996687 |
| Mean energy above m | 1.20395 m | 1.20395 m |
| Above b | 3.6e-5 of the far count | |

The block control differs by 1.4e-4 per copy. The threshold term is 0.058 weighted on the slice chain and 0.010 on the propagated chain.

**Certified bound at record 15** (dense root 51.760 a). The slice route gives 3.0054591 weighted, 0.49 percent of the 610.82 inside (per copy 0.0756118 of 15.342). The propagated check gives 3.0054591. The applied root gives 2.8013162 against 2.8013163.

**Half-arrival of the whole dust** (record cadence, propagated): weighted T/r_s 51.575, 39.45 r_s after formation, referred to 56 and still rising (per copy 39.62 in the record). The slices bracket it in (50, 52); linear interpolation between the 50 and 52 slices gives 51.570.

**Capacity record:** min over rooted records 26.34 (m e_min) at T/r_s 15.5, 23.10 at m. The weight outside the dense root at 15 is 29.18 weighted. No verdict.

## 2. Where the two routes differ at a printed digit

None of the sentence's numbers differs. The far count agrees to 8 digits, the current to 8, the bound to 7, the (m, b] share to 5 (0.99669) and the mean to 4 (1.204 m).

These differ and go to the record only, slice value first:

- **Weight beyond 24 r_s at 56:** 0.015945 against 0.015955. At 50 to 54 the differences are 2e-7 to 2.7e-6.
- **Small Killing bins at 56:**
  - (-inf, -b]: 0.0011452 against 0.0011423
  - (-m, m]: 0.12143 against 0.12168
  - (b, inf): 0.0018195 against 0.0018149
  - (m, b]: 50.55201 against 50.55177

## 3. Record items (not gates)

### 3.1 What changes in the propagator's input at T/r_s 40

engine_single_object_B_v2.CombinedHistory switches its record json and applied-field npz at 40. Up to 40 it reads r11 (C3prod19, collapse_c19_rs64_A_R2_tau8_ell4_avg_r11); after 40 it reads ext100 (C3ext19, from <internal>/run_snapshot).

Nothing else changes:

- **Grid and engine.** Both files have the same engine (engine_collapse_v19.py), tau 8, record spacing 0.125 r_s, ell 4, 2048 engine sites and the same field keys.
- **Run arguments.** They differ only in the run's bookkeeping: out, resume point, slice clocks and dtype, T end.
- **Solver.** Records 32 to 40 (r11) and 40 to 56 (ext100) both converge in 3 to 4 iterations.
- **Fields.** ext100 restarted from the r11 T28 slice; its applied fields over the 97 shared records 28 to 40 equal r11's to at most 2.2e-10 (1.8e-12 at 40).
- **Propagator.** The chain (8192 sites, the vacuum fields at the record's E beyond the engine's 2048 sites), the propagator (DUST v2b's frozen-field record step, Chebyshev 1e-14) and the record grid are the same on both sides.

So the switch at 40 introduces no input difference. The drift starts after 40 for another reason (3.3).

### 3.2 Where along the chain the difference sits (weighted L1 of the site weights)

**At 48.** The total is 1.44e-4. Almost all of it sits at 24 to 32 r_s:

| Radius | Difference |
|---|---|
| 24 to 28 r_s | 3.9e-5 |
| 28 to 30 r_s | 4.8e-5 |
| 30 to 31 r_s | 2.8e-5 |
| 31 to 32 r_s | 2.9e-5 |
| inside 24 r_s | below 4e-8 |

The peak is at 31.78 r_s and the median radius at 29.4 r_s. The last 64 sites (31 to 32 r_s) hold 20 percent of it.

**At 56.** The total is 9.1e-3. It has moved inward:

| Radius | Difference |
|---|---|
| 16 to 20 r_s | 4.7e-3 |
| 20 to 24 r_s | 2.5e-3 |
| 24 to 28 r_s | 7.6e-4 |
| 28 to 32 r_s | 1.1e-3 |
| inside 16 r_s | 1.2e-7 |

The peak is at 19.66 r_s and the median at 19.9 r_s.

At 48 the propagated set holds 2.9e-5 weighted beyond 32 r_s, where the production chain has no sites; at 56 it holds 4.7e-4. The difference appears first at the production chain's end (32 r_s) and travels inward with the reflected content. It does not originate in the interior.

### 3.3 Attribution test, 48 to 50

The stored 48 slice's dust columns (Loewdin, as B.5) were advanced 16 records on the stored ext100 fields by DUST v2b's propagator, twice:

- **On the production's own chain (2048 sites, ending at 32 r_s):** against the stored 50 slice, 1 - overlap is at most 1.1e-13 and the site-weight L1 at most 1.3e-13, every kappa.
- **On B's chain (8192 sites):** 1 - overlap is 1.5e-7 to 2.4e-7 and the L1 1.7e-7 to 2.7e-7, with 0.75e-7 to 1.2e-7 per column carried beyond 32 r_s.

So the propagator's step is the production engine's step to the storage floor. The slice-vs-propagated difference is the chain: the production's chain ends at 32 r_s, and the propagated control's runs to 128 r_s. The drift starts after 40 because the far dust reaches the 32 r_s end only then (the weight beyond 24 r_s is 3.5e-4 weighted at 40 and 6.8e-3 at 48).

This replaces v1.1's attribution, "B's frozen-field record step against the production engine's own step", which was by elimination and is wrong. It also qualifies the conditions sentence's phrase "drifting from the production engine's own step after 40": the drift is from the production engine's chain end, not its step.

## For the red team

1. **The slices' route includes a wall at 32 r_s.** The production's own chain ends there. From 48 on, the far dust reaching it is reflected, and by 56 the reflected part sits at 16 to 24 r_s, inside the region whose count prints. The propagated control has no wall before 128 r_s. The two give the same far count to 8 digits at 56, so the reflected weight crosses no printed digit yet. But the 24 r_s weight already differs at the third digit, and the record should stop at 56, as the ruling says.
2. **The conditions sentence names the wrong cause.** It reads "drifting from the production engine's own step after 40". Section 3.3 shows the step agrees to 1e-13 on the same chain. The cause is the chain length (2048 against 8192 sites). The sentence needs that word changed.
3. **The (m, b] share depends on which chain the held operator lives on.** It is 0.99669 at the printed digits on both chains, but the threshold term is six times larger on the short chain (0.058 against 0.010 weighted). The Killing split is on the production's chain by ruling.
4. **The half-arrival clock comes from the propagated cadence.** The slices bracket it (51.570 by interpolation against 51.575). It is referred to a still-rising count.
5. **The bound uses the dense root at 15** (51.760 a, the record's trapping zero). The applied root gives 2.80 instead of 3.01.
6. **The inside-origin piece (37.57) has only the propagated route.** It carries the chain difference as its error, which grows from 0 at 40 to about 1e-2 weighted in L1 at 56.
7. **One analysis code serves both routes.** The Loewdin construction, the held operator and the bins are shared. hreduce's rederivation of the bound from the 15 and 56 slices is the outstanding second route.
8. **The capacity minimum is at 15.5, not 15.** It is a record item with no verdict.

## Files and md5

- **Script** (copy on the box in W7_v1_2/code/): <internal>/W7_record_items_v1_2.py, md5 dd643e42b52c0238202f2def98dcd262. It uses W7_common_v1.py, md5 821809e1dd0d5f119e405074fbad0efd.
- **Output:** <internal>/W7_record_items_v1_2.json, md5 65798d28972569c6f9c83a89835108db, with its log W7_record_items_v1_2.log.
- **Numbers re-used from v1.1:** W7_v1_1/W7_compose_v1_1.json, md5 6c961bb675d2a9ddbc07c7208e78d486. Its inputs and md5s are listed in NOTES_W7_five_items_v1.1.md.
