# MASS_ladder_v1.3: the mass ladder at sigma/a 4, tau 8 (m a 0.4, 0.2 and 0.1 at r_s/a 64; 0.1 at r_s/a 96; 0.05 at r_s/a 326), composed from the stored members, with the corrections of the correctness reading

Version 1.3 of MASS_ladder_v1 (versions 1, 1.1 and 1.2 stay as issued). Composed by ladders_readout_v1_3.py (the producer, box CPU, stored run files only) and ladders_compose_v1_3.py (this text, from the producer's json). The builder composes; no sentence here reads the physics.

## Changes from v1.2

- The certification rule is the production rule (deliveries/lead_gaps/FableAPI_cert_rule_v1.md): a record is certified when its kept residual and its confirming residual are both at most 1e-3, and a history is certified to its last record before the first that fails; records counting with either residual between 1e-4 and 1e-3 are banded and carry a dagger (†) in every table. Only the certification inputs changed (the producer's counting test, ladders_readout_v1_3.py); every record, clock, radius and fit input is read as in version 1.2. The converged flag no longer gates. A Newton record's confirming residual is solve_v32's (version 1.2 held it to stop_fields 1e-6). A plain record kept unconverged is confirmed by its iteration_history entry after the kept iterate, which the ruling asks for before the tables recompose (the plain mixer applies the kept fields, so that entry is the map's change out of them); a plain record whose kept iterate is its last entry, and every converged record, rest on the kept residual.
- That test shortens extents of LADDER_certification_list_v1.1 (certext_production_v1.json, which tested the kept residual alone on plain members): M3a_s2 1.375 to none (record 0: kept 9.57e-05, following entry 1.02e-03); P64_s4_tau2 12.78125 to 12.75 (record 12.78125: kept 6.96e-05, following entry 3.08e-03). It shortens no other member (certext_followup_v1.py); the producer's extents equal the list's on the other 23 of 27 members.
- Certified extents that move (T/r_s): M3c_s4_nk 21.916667 to 22.083333; M3d_s4 0.196319 to 0.490798; P64_s2 none to 40; P64_s4_tau4 13.375 to 40. Unchanged and short of the last record: P64_s4_tau2 (12.75), M3a_s2 (none).
- Formations that become certified: P64_s2 at 12 (v1.2: 30.3 only). No formation loses its certification; every formation is certified.
- Banded records inside the certified extents, marked † wherever a table prints them: M3c_s4_nk 22 (kept 2.21e-04, confirming 3.89e-04); P64_s4_tau4 13.4375 (kept 6.18e-05, confirming 8.81e-04); none is a formation record. Counting banded records past an extent carry (unc.) † (9 in this document's members, listed in 6.1).
- Values that change, per member (every other member value is unchanged):
  - M3c_s4_nk: counting records 265 to 266; non-counting records 2 to 1; first non-counting record 22.166667 (kept residual above 1e-3; confirming residual above 1e-3); min R_c over the certified records 39.21 a at 21.916667 to 32.01 a at 22.083333 (0.333 r_s); episode 1 depth over certified records -0.2653 to -0.3699 (at 22.083333); (unc.) removed at matched clocks 22; (n.c.) removed at 22
  - M3d_s4: counting records 16 to 21; non-counting records 5 to 0
  - P64_s2: counting records 313 to 321; non-counting records 8 to 0; min R_c over the certified records none (no certified record) to 10.85 a at 12.875 (0.169 r_s); episode 1 depth over certified records none to -0.6618 (at 12.75); (unc.) removed at matched clocks 0 to 40; (n.c.) removed at 0, 6; (unc.) removed at T_f + 0, 0.5, 1, 2, 4
  - P64_s4_tau4: counting records 640 to 641; non-counting records 1 to 0; episode 2 depth over certified records none to -0.2408 (at 21.75); episode 3 depth over certified records none to -0.1952 (at 29.25); (unc.) removed at matched clocks 14 to 40; (unc.) removed at T_f + 2, 4
  - P64_s4_tau2: first non-counting record 12.78125 (confirming residual above 1e-3)
  - M3a_s2: counting records 277 to 292; non-counting records 44 to 29; first non-counting record 0 (confirming residual above 1e-3); (n.c.) becomes (unc.) at 1
- Control spreads that change (6.3, control minus member, over clocks counting and certified in both):
  - M3a_s2 against M3a_s4: raw common clocks 277 to 292
  - M3c_s4_plain against M3c_s4_nk: common certified clocks 264 to 266; last common clock 21.916667 to 22.083333; max |d E/E0| 2.168e-06 to 1.376e-05; max |d min(N - |v|)| 4.736e-05 to 0.0001253; common zero records 10 to 12; raw common clocks 265 to 266
  - P64_s2 against P64_s4: common certified clocks 0 to 321; last common clock none to 40; max |d E/E0| none to 0.08159; max |d min(N - |v|)| none to 0.3114; max |d R_c|/r_s none to 0.3994; common zero records 0 to 216; max |d R_t| (a) none to 11.3; max |d R_t|/R_t none to 0.2223; at none to 16; classes certified / 30.3 only to certified / certified; raw common clocks 313 to 321
  - P64_s4_tau4 against P64_s4: common certified clocks 108 to 321; last common clock 13.375 to 40; max |d E/E0| 0.03249 to 0.04485; max |d min(N - |v|)| 0.106 to 0.1668; max |d R_c|/r_s 0.06955 to 0.5809; common zero records 11 to 183; max |d R_t| (a) 2.949 to 7.871; max |d R_t|/R_t 0.07649 to 0.2044; at 12.125 to 20.125
- Fits: none of the mass diagnostics (5.1 to 5.3) changes.
- The sigma/a 2 control at 64 is now certified: P64_s2 against P64_s4 (certified / certified): |dT_f| 0.1250 r_s, |dR_t|/R_t at formation 1.2%, max |dR_t|/R_t over 216 common certified zero records 22.2% at 16.0000; within both criteria at formation, the radius difference over the common records exceeding 10 percent.
- Wording: the definitions paragraph, the end-state notes of section 8, the convergence table's columns (confirming residual above 1e-3 for both solvers, the band column) and the sentences that quoted the strict extents now read the producer's extents. The change lists of versions 1.1 and 1.2 stay in those versions. Every changed value with its old and new entry (1906 paths, 970 of them new per-record band flags) is in ladders_diff_v1_3.json.

## 0 The members' end states

- M3b_s4 (0.4, 64) s4 t8 (mass member (0.4, 64); 2048 sites, tau 8, N_B 320): first saved zero 5.3750 at R_t 21.039 a (0.3287 r_s, root fraction 0.3727, gate 0.190), certified (inside the certified prefix); last converged record 40.0000.
- P64_s4 (0.2, 64) s4 t8 (mass and size member (0.2, 64), the production's history (a reproduction of the v19 record); 2048 sites, tau 8, N_B 640): first saved zero 12.1250 at R_t 38.556 a (0.6024 r_s, root fraction 0.6613, gate 0.104), certified (inside the certified prefix); last converged record 40.0000.
- M3a_s4 (0.1, 64) s4 t8 (mass member (0.1, 64), near-maximal; 2048 sites, tau 8, N_B 1280): no trapping zero in its records to 40.0000; last converged record 40.0000.
- M3c_s4_nk (0.1, 96) s4 t8 (mass member (0.1, 96); 3072 sites, tau 8, N_B 1920): first saved zero 21.1667 at R_t 65.617 a (0.6835 r_s, root fraction 0.7270, gate 0.061), certified (inside the certified prefix); last converged record 22.0833, first non-counting record 22.1667 (kept residual above 1e-3; confirming residual above 1e-3).
- M3d_s4 (0.05, 326) s4 t8 (mass member (0.05, 326) (kappa_sea 12, plain, v36e); 3912 sites, tau 8, N_B 13040): no trapping zero in its records to 0.4908; last converged record 0.4908.

Controls and comparators (not ladder members):

- M3b_s8 (0.4, 64) s8 t8 (sigma/a 8 control of M3b_s4 (its formation grazing); 2048 sites, tau 8, N_B 320): first saved zero 6.7500 at R_t 26.230 a (0.4098 r_s, root fraction 0.4625, gate 0.305), certified (inside the certified prefix); last converged record 40.0000.
- P64_s2 (0.2, 64) s2 t8 (sigma/a 2 point at 64: the composite history (Newton across its seams); 2048 sites, tau 8, N_B 640): first saved zero 12.0000 at R_t 38.098 a (0.5953 r_s, root fraction 0.6545, gate 0.052), certified (inside the certified prefix); last converged record 40.0000.
- P64_s8 (0.2, 64) s8 t8 (sigma/a 8 point at 64; 2048 sites, tau 8, N_B 640): first saved zero 12.7500 at R_t 44.527 a (0.6957 r_s, root fraction 0.7620, gate 0.180), certified (inside the certified prefix); last converged record 40.0000.
- P64_s4_tau4 (0.2, 64) s4 t4 (tau 4 control of P64_s4 (engine v19); 2048 sites, tau 4, N_B 640): first saved zero 12.0000 at R_t 37.323 a (0.5832 r_s, root fraction 0.6360, gate 0.107), certified (inside the certified prefix); last converged record 40.0000.
- P64_s4_tau2 (0.2, 64) s4 t2 (tau 2 control of P64_s4 (engine v25, to 21.5); 2048 sites, tau 2, N_B 640): first saved zero 11.9062 at R_t 34.669 a (0.5417 r_s, root fraction 0.5888, gate 0.115), certified (inside the certified prefix); last converged record 12.7500, first non-counting record 12.7812 (confirming residual above 1e-3).
- P64_s4_c19 (0.2, 64) s4 t8 (the production record (engine v19); P64_s4 reproduces it; 2048 sites, tau 8, N_B 640): first saved zero 12.1250 at R_t 38.556 a (0.6024 r_s, root fraction 0.6613, gate 0.104), certified (inside the certified prefix); last converged record 40.0000.
- M3a_s2 (0.1, 64) s2 t8 (sigma/a 2 control of M3a_s4; 2048 sites, tau 8, N_B 1280): no trapping zero in its records to 40.0000; last converged record none (its first record does not count), first non-counting record 0.0000 (confirming residual above 1e-3).
- M3a_s4_tau4 (0.1, 64) s4 t4 (tau 4 control of M3a_s4 (to 16); 2048 sites, tau 4, N_B 1280): no trapping zero in its records to 16.0000; last converged record 16.0000.
- M3c_s4_plain (0.1, 96) s4 t8 (plain-iteration comparator of M3c_s4_nk; 3072 sites, tau 8, N_B 1920): first saved zero 21.1667 at R_t 65.607 a (0.6834 r_s, root fraction 0.7269, gate 0.061), certified (inside the certified prefix); last converged record 22.0833.

Definitions (version 1.3; deliveries/lead_gaps/FableAPI_cert_rule_v1.md). A record is certified when the self-consistent solve of its fields has converged to a residual of at most 1e-3, the residual being the largest change of N or v over one iteration of the solve at its final fields; a residual epsilon shifts an exterior root of N - |v| by 2 r_s epsilon, about a tenth of a spacing at the threshold here, and a history is certified to its last record before the first that fails. Residuals between 1e-4 and 1e-3 are marked † in the tables, and a reading past a history's certified extent says so where it is made. As applied: both the kept residual (engine_ladder_compare_v5.rec_residual) and the confirming residual (a Newton record's solve_v32 confirming residual; a plain record kept unconverged, the iteration_history entry after the kept iterate) must be at most 1e-3; a record failing either does not count and is named. Two extents print for every member: the 30.3 prefix, ending before the first record whose kept residual exceeds 1e-3, and the certified extent (the certified prefix), ending before the first non-counting record of any kind. The last converged record is the last record of the certified extent. A formation is certified when its record lies inside the certified extent, and only such formations enter a fit. Formation is the first saved record holding a trapping zero (trapping_zeros non-empty), the birth lying in the interval (T_f minus the record step, T_f], whose width prints with each member; R_t at formation is the outer zero of the dense N + v on that first saved record and not the birth radius; the 2 zeros of every formation record print with no recorded tangency and a positive slope at R_t (the inner zero's simplicity is not tested). R_t/(2 g E) is the record's own stored ratio at its charge E (g 1/4): the root fraction. The root gate is sigma/R_t with sigma = ell. E/E0 is against the head's E0 of the ell 4 preparation, so it is not 1 at T = 0 for the re-averaged controls. N_B (m a)^3/1.087 is the nominal count ratio (to the displayed approximation the rest-energy ratio N_B m/E_max), not the record's charge ratio; the charge ratio E0/E_max uses the stored E0 and E_max = c_E/(g^(3/2) m^2) with c_E 0.135858. Labels: (n.c.) the record does not count; (unc.) the record lies after the member's certified extent; † the record counts and is banded (kept or confirming residual between 1e-4 and 1e-3).

## 1 The member table

Every member and control shares one preparation rule: the dust is the levitating well's zero-temperature filling within the sea set kappa = 1 to kappa_sea (engine prepare_dust with fill_lowest: the well of depth set by the release radius R_0, the lowest levels of all channels filled to the count N_B = r_s/(2 g m a), g = 1/4, the projection onto the clocked positive band), released from coordinate rest at R_0 = 4 r_s (2 r_s for the 2 preparation members), prepared at ell 4 and evolved at the member's ell. No member runs the isotropic catalog; each member's own onset prints as 4 a times the largest chain column count of its catalog at T = 0 (20 a for the production's set), an upper bound on the per-valley onset 4 a n_rad,max that the T = 0 valley census fixes (12 a to 20 a for the production's set by the flat-well control's split), beside the isotropic form 1.222 N_B^(1/3) a; the isotropic onset and the exclusion floor 0.49 N_B^(1/3) a are comparators, not demonstrated bounds on the chain's dust.

| member | role | m a | r_s/a | sigma/a | tau/a (tau/r_s) | sites | N_B | N_B (m a)^3/1.087 (nominal count ratio) | E0 (head) | 2 g E0 (a) | E0/(N_B m) | E0/E_max (charge ratio, c_E 0.135858) | f_kin (head) | p_F/m isotropic release form | (p_F/m)^2/5 isotropic (a release comparator) | own onset, chain columns (4 a times columns; upper bound) (/r_s) | isotropic 1.222 N_B^(1/3) a (/r_s) | exclusion floor 0.49 N_B^(1/3) a | records (first to last T/r_s; dropped without E) | last converged record (certified prefix) | 30.3 prefix's last counting record | first non-counting record (its gate) | end state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b_s4 | mass member (0.4, 64) | 0.4 | 64 | 4 | 8 (0.12500) | 2048 | 320 | 18.84 | 113.6966 | 56.848 | 0.8883 | 16.74 | 0.1209 | 0.064 | 0.000822 | 20 a (0.312) | 8.4 a (0.131) | 3.4 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s4 | mass and size member (0.2, 64), the production's history (a reproduction of the v19 record) | 0.2 | 64 | 4 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| M3a_s4 | mass member (0.1, 64), near-maximal | 0.1 | 64 | 4 | 8 (0.12500) | 2048 | 1280 | 1.18 | 123.4741 | 61.737 | 0.9646 | 1.14 | 0.1600 | 0.407 | 0.033127 | 24 a (0.375) | 13.3 a (0.207) | 5.3 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40; no trapping zero in any record |
| M3c_s4_nk | mass member (0.1, 96) | 0.1 | 96 | 4 | 8 (0.08333) | 3072 | 1920 | 1.77 | 181.8183 | 90.909 | 0.9470 | 1.67 | 0.1294 | 0.311 | 0.019293 | 28 a (0.292) | 15.2 a (0.158) | 6.1 a | 267 (0.000 to 22.1667; 0) | 22.0833 | 22.0833 | 22.1667 (kept residual above 1e-3; confirming residual above 1e-3) | stopped at 22.1667 of its requested 40; certified to 22.0833 |
| M3d_s4 | mass member (0.05, 326) (kappa_sea 12, plain, v36e) | 0.05 | 326 | 4 | 8 (0.02454) | 3912 | 13040 | 1.50 | 620.6128 | 310.306 | 0.9519 | 1.43 | 0.1496 | 0.346 | 0.024000 | not available | 28.8 a (0.088) | 11.6 a | 21 (0.000 to 0.4908; 0) | 0.4908 | 0.4908 | none | stopped at 0.4908 of its requested 30; no trapping zero in any record |
| M3b_s8 | sigma/a 8 control of M3b_s4 (its formation grazing) | 0.4 | 64 | 8 | 8 (0.12500) | 2048 | 320 | 18.84 | 113.6966 | 56.848 | 0.8883 | 16.74 | 0.1209 | 0.064 | 0.000822 | 20 a (0.312) | 8.4 a (0.131) | 3.4 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s2 | sigma/a 2 point at 64: the composite history (Newton across its seams) | 0.2 | 64 | 2 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s8 | sigma/a 8 point at 64 | 0.2 | 64 | 8 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s4_tau4 | tau 4 control of P64_s4 (engine v19) | 0.2 | 64 | 4 | 4 (0.06250) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 641 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s4_tau2 | tau 2 control of P64_s4 (engine v25, to 21.5) | 0.2 | 64 | 4 | 2 (0.03125) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 689 (0.000 to 21.5000; 0) | 12.7500 | 12.7500 | 12.7812 (confirming residual above 1e-3) | records to 21.5; certified to 12.7500 |
| P64_s4_c19 | the production record (engine v19); P64_s4 reproduces it | 0.2 | 64 | 4 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| M3a_s2 | sigma/a 2 control of M3a_s4 | 0.1 | 64 | 2 | 8 (0.12500) | 2048 | 1280 | 1.18 | 123.4741 | 61.737 | 0.9646 | 1.14 | 0.1600 | 0.407 | 0.033127 | 24 a (0.375) | 13.3 a (0.207) | 5.3 a | 321 (0.000 to 40.0000; 0) | none (its first record does not count) | 1.0000 | 0.0000 (confirming residual above 1e-3) | records to 40; certified to no record; no trapping zero in any record |
| M3a_s4_tau4 | tau 4 control of M3a_s4 (to 16) | 0.1 | 64 | 4 | 4 (0.06250) | 2048 | 1280 | 1.18 | 123.4741 | 61.737 | 0.9646 | 1.14 | 0.1600 | 0.407 | 0.033127 | 24 a (0.375) | 13.3 a (0.207) | 5.3 a | 257 (0.000 to 16.0000; 0) | 16.0000 | 16.0000 | none | records to 16; no trapping zero in any record |
| M3c_s4_plain | plain-iteration comparator of M3c_s4_nk | 0.1 | 96 | 4 | 8 (0.08333) | 3072 | 1920 | 1.77 | 181.8183 | 90.909 | 0.9470 | 1.67 | 0.1294 | 0.311 | 0.019293 | 28 a (0.292) | 15.2 a (0.158) | 6.1 a | 266 (0.000 to 22.0833; 0) | 22.0833 | 22.0833 | none | stopped at 22.0833 of its requested 40 |

The p_F/m of the isotropic release form and its (p_F/m)^2/5 are release comparators, not measured pressure corrections of the anisotropic well fillings; the members' own f_kin (head) gives (p_F/m)^2 = f_kin/0.3 by the composer's cold-gas conversion, 0.35 to 0.53 on every member and non-monotone in m a. P64_s4 (v36d from T = 0) agrees with the v19 production record P64_s4_c19 to 1.2e-8 in E/E0 on all 321 clocks (6.3): a reproduction of the same arithmetic, not an independent route.

## 2 The first saved trapping record per member, with the bracket, the root fraction and the gate

| member | first saved record with a zero T_f/r_s | bracket (previous record T/r_s, min(N - |v|) there; width r_s) | zero of the stored min(N - |v|) series in the bracket: linear; quadratic (a diagnostic, not an event location) | R_t/a (first record) | R_t/r_s | R_t/(2 g E) (root fraction) | E/E0 | zeros (a) | no tangency; slope at R_t | root gate sigma/R_t | residual (kept); confirming | certification († banded) | first episode closes at | min(N - |v|) in the first episode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b_s4 | 5.3750 | 5.2500, 0.0007; 0.12500 | 5.2519; 5.2518 | 21.039 | 0.3287 | 0.3727 | 0.99294 | 11.080, 21.039 | yes; 1.34e-02 | 0.190 | 4.68e-07; - | certified (inside the certified prefix) | 7.0000 | -0.1037 (5.6250) |
| P64_s4 | 12.1250 | 12.0000, 0.0222; 0.12500 | 12.0553; 12.0571 | 38.556 | 0.6024 | 0.6613 | 0.98702 | 23.239, 38.556 | yes; 5.46e-03 | 0.104 | 6.41e-07; - | certified (inside the certified prefix) | 17.3750 | -0.4410 (13.0000) |
| M3a_s4 | none to 40.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| M3c_s4_nk | 21.1667 | 21.0833, 0.0102; 0.08333 | 21.1321; 21.1328 | 65.617 | 0.6835 | 0.7270 | 0.99284 | 49.772, 65.617 | yes; 1.73e-03 | 0.061 | 1.68e-09; 1.00e-10 | certified (inside the certified prefix) | open to 22.1667 | -0.4357 (22.1667, unc.); over certified records -0.3699 (22.0833); 1 of 13 records do not count, 1 uncertified |
| M3d_s4 | none to 0.4908 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| M3b_s8 | 6.7500 | 6.6250, 0.0064; 0.12500 | 6.7445; 6.7420 | 26.230 | 0.4098 | 0.4625 | 0.99763 | 24.913, 26.230 | yes; 8.75e-04 | 0.305 | 8.82e-07; - | certified (inside the certified prefix) | 7.2500 | -0.0031 (6.8750) |
| P64_s2 | 12.0000 | 11.8750, 0.0103; 0.12500 | 11.8965; 11.8979 | 38.098 | 0.5953 | 0.6545 | 0.98550 | 18.096, 38.098 | yes; 6.33e-03 | 0.052 | 3.10e-08; 3.32e-10 | certified (inside the certified prefix) | open to 40.0000 | -0.6618 (12.7500) |
| P64_s8 | 12.7500 | 12.6250, 0.0010; 0.12500 | 12.6291; 12.6291 | 44.527 | 0.6957 | 0.7620 | 0.98920 | 27.562, 44.527 | yes; 5.71e-03 | 0.180 | 2.59e-08; - | certified (inside the certified prefix) | 16.3750 | -0.1908 (13.7500) |
| P64_s4_tau4 | 12.0000 | 11.9375, 0.0001; 0.06250 | 11.9378; 11.9378 | 37.323 | 0.5832 | 0.6360 | 0.99341 | 22.029, 37.323 | yes; 5.33e-03 | 0.107 | 4.00e-07; - | certified (inside the certified prefix) | 16.3750 | -0.4651 (12.8750) |
| P64_s4_tau2 | 11.9062 | 11.8750, 0.0010; 0.03125 | 11.8772; 11.8772 | 34.669 | 0.5417 | 0.5888 | 0.99689 | 23.924, 34.669 | yes; 4.01e-03 | 0.115 | 2.20e-07; - | certified (inside the certified prefix) | 16.0938 (unc.) | -0.4787 (12.8125, unc.); over certified records -0.4739 (12.7500); 3 of 134 records do not count, 106 uncertified |
| P64_s4_c19 | 12.1250 | 12.0000, 0.0222; 0.12500 | 12.0553; 12.0571 | 38.556 | 0.6024 | 0.6613 | 0.98702 | 23.239, 38.556 | yes; 5.46e-03 | 0.104 | 6.41e-07; - | certified (inside the certified prefix) | 17.3750 | -0.4410 (13.0000) |
| M3a_s2 | none to 40.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| M3a_s4_tau4 | none to 16.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| M3c_s4_plain | 21.1667 | 21.0833, 0.0102; 0.08333 | 21.1322; 21.1330 | 65.607 | 0.6834 | 0.7269 | 0.99284 | 49.787, 65.607 | yes; 1.73e-03 | 0.061 | 3.79e-07; - | certified (inside the certified prefix) | open to 22.0833 | -0.3698 (22.0833) |

Over the 0.5 r_s after the first saved record the root fraction grows by 0.095 to 0.210 of 2GE on the certified formed members (M3b_s4 0.3727 to 0.4679; P64_s4 0.6613 to 0.8140; M3c_s4_nk 0.7270 to 0.8649; P64_s2 0.6545 to 0.7950; P64_s8 0.7620 to 0.8623; P64_s4_tau4 0.6360 to 0.8076; P64_s4_tau2 0.5888 to 0.7989; P64_s4_c19 0.6613 to 0.8140; M3c_s4_plain 0.7269 to 0.8649), so a first-record radius carries a sampling scatter of several hundredths of 2GE depending on where in the step the birth fell; no fit of R_t at formation and no intercept of T_f in any variable is a law before the crossing of N - |v| is located between records on every member (busi B7).

M3b_s8's formation is grazing: min(N - |v|) -0.0003 at the record, -0.0031 over its 4-record episode, zeros 1.32 a apart; the spread it sets beside M3b_s4 is that of a marginal episode. The mass direction compares a root at 0.3727 of the charge (0.4, 64), gate 0.190, with roots at 0.6613 (0.2, 64) and 0.7270 (0.1, 96): an interior core root against outer roots.

The trapped episodes per member (labels as in section 0):

- M3b_s4: 1: 5.3750 to 6.8750 (13 records; R_t 21.039 to 30.743 a; min(N - |v|) -0.1037 at 5.6250; closes at 7.0000); 2: 8.1250 to 10.8750 (23 records; R_t 26.232 to 24.905 a; min(N - |v|) -0.1123 at 9.5000; closes at 11.0000); 3: 29.0000 to 30.2500 (11 records; R_t 22.912 to 23.044 a; min(N - |v|) -0.0232 at 29.3750; closes at 30.3750)
- P64_s4: 1: 12.1250 to 17.2500 (42 records; R_t 38.556 to 37.932 a; min(N - |v|) -0.4410 at 13.0000; closes at 17.3750); 2: 18.3750 to 40.0000 (174 records; R_t 34.350 to 36.310 a; min(N - |v|) -0.2211 at 20.7500; open at the last record)
- M3c_s4_nk: 1: 21.1667 to 22.1667 (13 records; R_t 65.617 to 77.918 a; min(N - |v|) -0.4357 at 22.1667 (unc.); open at the last record; 1 of 13 records do not count, 1 uncertified)
- M3b_s8: 1: 6.7500 to 7.1250 (4 records; R_t 26.230 to 28.548 a; min(N - |v|) -0.0031 at 6.8750; closes at 7.2500)
- P64_s2: 1: 12.0000 to 40.0000 (225 records; R_t 38.098 to 31.423 a; min(N - |v|) -0.6618 at 12.7500; open at the last record)
- P64_s8: 1: 12.7500 to 16.2500 (29 records; R_t 44.527 to 50.517 a; min(N - |v|) -0.1908 at 13.7500; closes at 16.3750); 2: 26.6250 to 29.6250 (25 records; R_t 35.899 to 39.774 a; min(N - |v|) -0.0561 at 28.0000; closes at 29.7500); 3: 38.8750 to 40.0000 (10 records; R_t 37.951 to 40.963 a; min(N - |v|) -0.0269 at 39.7500; open at the last record)
- P64_s4_tau4: 1: 12.0000 to 16.3125 (70 records; R_t 37.323 to 42.893 a; min(N - |v|) -0.4651 at 12.8750; closes at 16.3750); 2: 20.1250 to 25.5000 (87 records; R_t 30.642 to 32.572 a; min(N - |v|) -0.2408 at 21.7500; closes at 25.5625); 3: 26.9375 to 40.0000 (210 records; R_t 30.546 to 34.464 a; min(N - |v|) -0.1952 at 29.2500; open at the last record)
- P64_s4_tau2: 1: 11.9062 to 16.0625 (134 records; R_t 34.669 to 43.373 a; min(N - |v|) -0.4787 at 12.8125 (unc.); closes at 16.0938 (unc.); 3 of 134 records do not count, 106 uncertified); 2: 21.0312 to 21.5000 (16 records; R_t 30.009 to 36.375 a; min(N - |v|) -0.0799 at 21.5000 (unc.); open at the last record; 16 uncertified)
- P64_s4_c19: 1: 12.1250 to 17.2500 (42 records; R_t 38.556 to 37.932 a; min(N - |v|) -0.4410 at 13.0000; closes at 17.3750); 2: 18.3750 to 40.0000 (174 records; R_t 34.350 to 36.310 a; min(N - |v|) -0.2211 at 20.7500; open at the last record)
- M3c_s4_plain: 1: 21.1667 to 22.0833 (12 records; R_t 65.607 to 78.207 a; min(N - |v|) -0.3698 at 22.0833; open at the last record)

## 3 The dust-energy centroid R_c against the onsets (the centroid's minimum over the certified records; not the core's compression radius, which the requirement reads on the half-mass radius with the dust's bare energy per particle there and which is not composed here)

| member | min R_c (dust-energy centroid) over the certified records (a) | at T/r_s | min R_c/r_s | own onset, chain columns (upper bound) | isotropic 1.222 N_B^(1/3) a | exclusion floor | minimum at the last certified record |
|---|---|---|---|---|---|---|---|
| M3b_s4 | 34.75 | 9.2500 | 0.543 | 20 a | 8.4 a | 3.4 a | no |
| P64_s4 | 14.52 | 13.2500 | 0.227 | 20 a | 10.5 a | 4.2 a | no |
| M3a_s4 | 159.59 | 40.0000 | 2.494 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |
| M3c_s4_nk | 32.01 | 22.0833 | 0.333 | 28 a | 15.2 a | 6.1 a | yes (still falling at the end of the certified records) |
| M3d_s4 | 874.08 | 0.0000 | 2.681 | not available | 28.8 a | 11.6 a | no |
| M3b_s8 | 37.88 | 8.7500 | 0.592 | 20 a | 8.4 a | 3.4 a | no |
| P64_s2 | 10.85 | 12.8750 | 0.169 | 20 a | 10.5 a | 4.2 a | no |
| P64_s8 | 22.40 | 14.0000 | 0.350 | 20 a | 10.5 a | 4.2 a | no |
| P64_s4_tau4 | 14.83 | 13.0000 | 0.232 | 20 a | 10.5 a | 4.2 a | no |
| P64_s4_tau2 | 15.92 | 12.7500 | 0.249 | 20 a | 10.5 a | 4.2 a | yes (still falling at the end of the certified records) |
| P64_s4_c19 | 14.52 | 13.2500 | 0.227 | 20 a | 10.5 a | 4.2 a | no |
| M3a_s2 | - | - | - | 24 a | 13.3 a | 5.3 a | no |
| M3a_s4_tau4 | 168.92 | 16.0000 | 2.639 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |
| M3c_s4_plain | 32.02 | 22.0833 | 0.333 | 28 a | 15.2 a | 6.1 a | yes (still falling at the end of the certified records) |

M3d_s4's centroid minimum is at T = 0: its centroid moves outward over its 21 certified records, unlike every other released member; not attributed here.

## 4 The matched-clock readouts (every 2 r_s; (n.c.) the record does not count, (unc.) it lies after the certified prefix)

### 4a min(N - |v|)

| T/r_s | M3b_s4 | P64_s4 | M3a_s4 | M3c_s4_nk | M3d_s4 |
|---|---|---|---|---|---|
| 0 | 0.4380 | 0.4945 | 0.4971 | 0.5018 | 0.5072 |
| 2 | 0.4319 | 0.4910 | 0.4991 | 0.5027 | - |
| 4 | 0.3474 | 0.4830 | 0.5047 | 0.5050 | - |
| 6 | -0.0795 | 0.4661 | 0.5085 | 0.5039 | - |
| 8 | 0.0031 | 0.4322 | 0.5095 | 0.4978 | - |
| 10 | -0.0772 | 0.3552 | 0.5076 | 0.4846 | - |
| 12 | 0.0791 | 0.0222 | 0.5030 | 0.4604 | - |
| 14 | 0.0754 | -0.2161 | 0.4987 | 0.4214 | - |
| 16 | 0.0694 | -0.0424 | 0.4960 | 0.3697 | - |
| 18 | 0.1011 | 0.0047 | 0.4941 | 0.2933 | - |
| 20 | 0.0270 | -0.1463 | 0.4923 | 0.1593 | - |
| 22 | 0.0504 | -0.1351 | 0.4905 | -0.3142 † | - |
| 24 | 0.0363 | -0.0550 | 0.4885 | - | - |
| 26 | 0.0761 | -0.0747 | 0.4865 | - | - |
| 28 | 0.0102 | -0.1758 | 0.4843 | - | - |
| 30 | -0.0098 | -0.0910 | 0.4819 | - | - |
| 32 | 0.0520 | -0.0620 | 0.4794 | - | - |
| 34 | 0.0628 | -0.1712 | 0.4768 | - | - |
| 36 | 0.0376 | -0.1173 | 0.4743 | - | - |
| 38 | 0.0518 | -0.0540 | 0.4721 | - | - |
| 40 | 0.0084 | -0.1603 | 0.4703 | - | - |

### 4b E/E0

| T/r_s | M3b_s4 | P64_s4 | M3a_s4 | M3c_s4_nk | M3d_s4 |
|---|---|---|---|---|---|
| 0 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| 2 | 1.00011 | 1.00003 | 1.00002 | 1.00001 | - |
| 4 | 1.00018 | 1.00007 | 1.00002 | 1.00002 | - |
| 6 | 0.98953 | 1.00010 | 0.99998 | 1.00001 | - |
| 8 | 0.98566 | 1.00003 | 0.99993 | 0.99996 | - |
| 10 | 0.98478 | 0.99935 | 0.99991 | 0.99990 | - |
| 12 | 0.98390 | 0.98981 | 0.99990 | 0.99982 | - |
| 14 | 0.98323 | 0.92502 | 0.99990 | 0.99973 | - |
| 16 | 0.98249 | 0.91882 | 0.99989 | 0.99956 | - |
| 18 | 0.98197 | 0.91692 | 0.99989 | 0.99913 | - |
| 20 | 0.98150 | 0.91186 | 0.99989 | 0.99729 | - |
| 22 | 0.98135 | 0.90844 | 0.99989 | 0.97337 † | - |
| 24 | 0.98128 | 0.90646 | 0.99989 | - | - |
| 26 | 0.98117 | 0.90580 | 0.99989 | - | - |
| 28 | 0.98104 | 0.90366 | 0.99988 | - | - |
| 30 | 0.98098 | 0.90297 | 0.99988 | - | - |
| 32 | 0.98081 | 0.90242 | 0.99988 | - | - |
| 34 | 0.98067 | 0.90112 | 0.99988 | - | - |
| 36 | 0.98060 | 0.90061 | 0.99988 | - | - |
| 38 | 0.98055 | 0.89982 | 0.99987 | - | - |
| 40 | 0.98051 | 0.89900 | 0.99987 | - | - |

### 4c R_c/r_s (the dust-energy centroid)

| T/r_s | M3b_s4 | P64_s4 | M3a_s4 | M3c_s4_nk | M3d_s4 |
|---|---|---|---|---|---|
| 0 | 1.8362 | 2.5875 | 2.7811 | 2.8145 | 2.6812 |
| 2 | 1.7498 | 2.5478 | 2.7831 | 2.7996 | - |
| 4 | 1.4447 | 2.4271 | 2.7616 | 2.7409 | - |
| 6 | 0.7588 | 2.2133 | 2.7239 | 2.6385 | - |
| 8 | 0.5955 | 1.8851 | 2.6860 | 2.4957 | - |
| 10 | 0.5792 | 1.4031 | 2.6613 | 2.3226 | - |
| 12 | 0.8450 | 0.6776 | 2.6495 | 2.1269 | - |
| 14 | 1.0473 | 0.3207 | 2.6450 | 1.9068 | - |
| 16 | 1.1308 | 0.5039 | 2.6425 | 1.6543 | - |
| 18 | 1.2429 | 0.6217 | 2.6382 | 1.3558 | - |
| 20 | 1.1874 | 0.5405 | 2.6299 | 0.9788 | - |
| 22 | 1.2071 | 0.5197 | 2.6166 | 0.3715 † | - |
| 24 | 1.2176 | 0.7295 | 2.5987 | - | - |
| 26 | 1.2333 | 0.7969 | 2.5777 | - | - |
| 28 | 1.1376 | 0.6774 | 2.5560 | - | - |
| 30 | 1.0779 | 0.8516 | 2.5368 | - | - |
| 32 | 1.1942 | 1.0181 | 2.5223 | - | - |
| 34 | 1.2960 | 0.9010 | 2.5130 | - | - |
| 36 | 1.3005 | 0.9933 | 2.5070 | - | - |
| 38 | 1.3156 | 1.2350 | 2.5013 | - | - |
| 40 | 1.2982 | 1.1415 | 2.4936 | - | - |

### 4d R_t/r_s ("none": no zero in the record)

| T/r_s | M3b_s4 | P64_s4 | M3a_s4 | M3c_s4_nk | M3d_s4 |
|---|---|---|---|---|---|
| 0 | none | none | none | none | none |
| 2 | none | none | none | none | - |
| 4 | none | none | none | none | - |
| 6 | 0.4258 | none | none | none | - |
| 8 | none | none | none | none | - |
| 10 | 0.4799 | none | none | none | - |
| 12 | none | none | none | none | - |
| 14 | none | 0.7902 | none | none | - |
| 16 | none | 0.7943 | none | none | - |
| 18 | none | none | none | none | - |
| 20 | none | 0.6005 | none | none | - |
| 22 | none | 0.6119 | none | 0.8172 † | - |
| 24 | none | 0.6081 | none | - | - |
| 26 | none | 0.5799 | none | - | - |
| 28 | none | 0.5868 | none | - | - |
| 30 | 0.3673 | 0.5861 | none | - | - |
| 32 | none | 0.5730 | none | - | - |
| 34 | none | 0.5801 | none | - | - |
| 36 | none | 0.5813 | none | - | - |
| 38 | none | 0.5595 | none | - | - |
| 40 | none | 0.5673 | none | - | - |

### 4e At the first saved trapping record and after it (T_f + 0, 0.5, 1, 2, 4 r_s)

| member | T_f + 0 | T_f + 0.5 | T_f + 1 | T_f + 2 | T_f + 4 |
|---|---|---|---|---|---|
| M3b_s4 | R_t/r_s 0.3287, R_t/(2gE) 0.3727, E/E0 0.99294, min(N-|v|) -0.0491, gate 0.190 | R_t/r_s 0.4114, R_t/(2gE) 0.4679, E/E0 0.98984, min(N-|v|) -0.0903, gate 0.152 | R_t/r_s 0.4632, R_t/(2gE) 0.5276, E/E0 0.98825, min(N-|v|) -0.0377, gate 0.135 | R_t/r_s none, R_t/(2gE) -, E/E0 0.98615, min(N-|v|) 0.0213, gate - | R_t/r_s 0.5103, R_t/(2gE) 0.5835, E/E0 0.98458, min(N-|v|) -0.1071, gate 0.122 |
| P64_s4 | R_t/r_s 0.6024, R_t/(2gE) 0.6613, E/E0 0.98702, min(N-|v|) -0.0279, gate 0.104 | R_t/r_s 0.7230, R_t/(2gE) 0.8140, E/E0 0.96234, min(N-|v|) -0.2880, gate 0.086 | R_t/r_s 0.7520, R_t/(2gE) 0.8731, E/E0 0.93323, min(N-|v|) -0.4367, gate 0.083 | R_t/r_s 0.7936, R_t/(2gE) 0.9302, E/E0 0.92439, min(N-|v|) -0.1960, gate 0.079 | R_t/r_s 0.7717, R_t/(2gE) 0.9102, E/E0 0.91861, min(N-|v|) -0.0381, gate 0.081 |
| M3c_s4_nk | R_t/r_s 0.6835, R_t/(2gE) 0.7270, E/E0 0.99284, min(N-|v|) -0.0072, gate 0.061 | R_t/r_s 0.8077, R_t/(2gE) 0.8649, E/E0 0.98623, min(N-|v|) -0.1508, gate 0.052 | R_t/r_s 0.8116, R_t/(2gE) 0.8938, E/E0 0.95889, min(N-|v|) -0.4357, gate 0.051 (n.c.) | - | - |
| M3b_s8 | R_t/r_s 0.4098, R_t/(2gE) 0.4625, E/E0 0.99763, min(N-|v|) -0.0003, gate 0.305 | R_t/r_s none, R_t/(2gE) -, E/E0 0.99727, min(N-|v|) 0.0033, gate - | R_t/r_s none, R_t/(2gE) -, E/E0 0.99688, min(N-|v|) 0.0234, gate - | R_t/r_s none, R_t/(2gE) -, E/E0 0.99590, min(N-|v|) 0.0444, gate - | R_t/r_s none, R_t/(2gE) -, E/E0 0.99533, min(N-|v|) 0.1071, gate - |
| P64_s2 | R_t/r_s 0.5953, R_t/(2gE) 0.6545, E/E0 0.98550, min(N-|v|) -0.0496, gate 0.052 | R_t/r_s 0.6883, R_t/(2gE) 0.7950, E/E0 0.93803, min(N-|v|) -0.4197, gate 0.045 | R_t/r_s 0.6924, R_t/(2gE) 0.8573, E/E0 0.87513, min(N-|v|) -0.5464, gate 0.045 | R_t/r_s 0.7091, R_t/(2gE) 0.9026, E/E0 0.85116, min(N-|v|) -0.1777, gate 0.044 | R_t/r_s 0.6177, R_t/(2gE) 0.7898, E/E0 0.84743, min(N-|v|) -0.1138, gate 0.051 |
| P64_s8 | R_t/r_s 0.6957, R_t/(2gE) 0.7620, E/E0 0.98920, min(N-|v|) -0.0307, gate 0.180 | R_t/r_s 0.7807, R_t/(2gE) 0.8623, E/E0 0.98096, min(N-|v|) -0.1453, gate 0.160 | R_t/r_s 0.8149, R_t/(2gE) 0.9051, E/E0 0.97544, min(N-|v|) -0.1908, gate 0.153 | R_t/r_s 0.8494, R_t/(2gE) 0.9436, E/E0 0.97533, min(N-|v|) -0.1288, gate 0.147 | R_t/r_s none, R_t/(2gE) -, E/E0 0.97357, min(N-|v|) 0.0237, gate - |
| P64_s4_tau4 | R_t/r_s 0.5832, R_t/(2gE) 0.6360, E/E0 0.99341, min(N-|v|) -0.0276, gate 0.107 | R_t/r_s 0.7302, R_t/(2gE) 0.8076, E/E0 0.97972, min(N-|v|) -0.3178, gate 0.086 | R_t/r_s 0.7792, R_t/(2gE) 0.8739, E/E0 0.96600, min(N-|v|) -0.4465, gate 0.080 | R_t/r_s 0.8268, R_t/(2gE) 0.9328, E/E0 0.96037, min(N-|v|) -0.1917, gate 0.076 | R_t/r_s 0.7428, R_t/(2gE) 0.8411, E/E0 0.95688, min(N-|v|) -0.0172, gate 0.084 |
| P64_s4_tau2 | R_t/r_s 0.5417, R_t/(2gE) 0.5888, E/E0 0.99689, min(N-|v|) -0.0131, gate 0.115 | R_t/r_s 0.7301, R_t/(2gE) 0.7989, E/E0 0.99014, min(N-|v|) -0.3136, gate 0.086 | R_t/r_s 0.7907, R_t/(2gE) 0.8717, E/E0 0.98277, min(N-|v|) -0.4530, gate 0.079 (unc.) | R_t/r_s 0.8438, R_t/(2gE) 0.9332, E/E0 0.97970, min(N-|v|) -0.1939, gate 0.074 (unc.) | R_t/r_s 0.7250, R_t/(2gE) 0.8034, E/E0 0.97778, min(N-|v|) -0.0089, gate 0.086 (unc.) |
| M3c_s4_plain | R_t/r_s 0.6834, R_t/(2gE) 0.7269, E/E0 0.99284, min(N-|v|) -0.0072, gate 0.061 | R_t/r_s 0.8077, R_t/(2gE) 0.8649, E/E0 0.98623, min(N-|v|) -0.1507, gate 0.052 | - | - | - |

## 5 The 2-coefficient diagnostics (no exponent, intercept or law is printed)

Members with a certified formation: M3b_s4, P64_s4, M3c_s4_nk. Without one: M3a_s4 (none to 40.0000), M3d_s4 (none to 0.4908).

### 5.1 Y = c0 + c1 x on the 3 formed members, x = (p_F/m)^2 of the isotropic release form

Under the assumption that Y depends on m a and r_s/a only through x = (p_F/m)^2 of the isotropic release form (x scales as (m a)^(-8/3) at fixed r_s/a and as (a/r_s)^(4/3) at fixed m a), the 3 formed members lie on one variable; both coefficients float, so this is a 2-coefficient line with 1 residual degree of freedom and not a test of the asymptotic form, which fixes the cold-limit value at the comparator's clock and the slope at 1/5 of it (the reading gives the cold-limit value as 13.22 r_s for release at 4 r_s with the production's charge; the stored W4 (a) comparator on the production's own profile folds at 10.8157 r_s, first root record 10.8750): the fitted c0 and c1/c0 below are not those values. The residuals are of the size of the stored averaging-scale spreads. The (0.4, 64) member's formation lies 1.375 r_s and 24.7% in radius from its sigma/a 8 control (itself grazing), outside the acceptance criterion, and is printed in this fit labelled so and in no law.

- model: 2-coefficient diagnostic Y = c0 + c1 x, x = (p_F/m)^2 of the isotropic release form
- members: M3b_s4, P64_s4, M3c_s4_nk (3 members, 2 coefficients, residual degrees of freedom 1)
- x per member: M3b_s4 0.00411, P64_s4 0.02609, M3c_s4_nk 0.09646

| Y | coefficients | member: Y, fit, residual | rms residual over members; per residual degree of freedom | max |residual| |
|---|---|---|---|---|
| T_f | c0 6.1077, c1 160.617 (c0 from first-record clocks; cadence error up to the record step per member; not an intercept) | M3b_s4: 5.3750, 6.7676, -1.3926; P64_s4: 12.1250, 10.2976, +1.8274; M3c_s4_nk: 21.1667, 21.6015, -0.4349 | 1.35e+00; 2.34e+00 | 1.83e+00 |
| R_t_over_rs | c0 0.40374, c1 3.18544 (first-record radii, at the sampling scatter of the first record; not a radius law) | M3b_s4: 0.3287, 0.4168, -0.0881; P64_s4: 0.6024, 0.4868, +0.1156; M3c_s4_nk: 0.6835, 0.7110, -0.0275 | 8.54e-02; 1.48e-01 | 1.16e-01 |
| R_t_over_2gE | c0 0.454948, c1 3.12795 (first-record radii, at the sampling scatter of the first record; not a radius law) | M3b_s4: 0.3727, 0.4678, -0.0951; P64_s4: 0.6613, 0.5365, +0.1248; M3c_s4_nk: 0.7270, 0.7567, -0.0297 | 9.22e-02; 1.60e-01 | 1.25e-01 |

| member | residual T_f | residual R_t/r_s | residual R_t/(2gE) | the member's stored control spreads at formation (control's class) |
|---|---|---|---|---|
| M3b_s4 | -1.3926 | -0.0881 | -0.0951 | sigma M3b_s8: dT_f +1.3750, d(R_t/r_s) +0.0811 (certified / certified) |
| P64_s4 | +1.8274 | +0.1156 | +0.1248 | sigma P64_s2: dT_f -0.1250, d(R_t/r_s) -0.0072 (certified / certified); sigma P64_s8: dT_f +0.6250, d(R_t/r_s) +0.0933 (certified / certified); step P64_s4_tau4: dT_f -0.1250, d(R_t/r_s) -0.0193 (certified / certified); step P64_s4_tau2: dT_f -0.2188, d(R_t/r_s) -0.0607 (certified / certified) |
| M3c_s4_nk | -0.4349 | -0.0275 | -0.0297 | solver M3c_s4_plain: dT_f +0.0000, d(R_t/r_s) -0.0001 (certified / certified) |

The diagnostic's value at the members it does not contain (each is below or beyond that member's records):

- M3a_s4: the fit's value T_f 32.7117, R_t/r_s 0.9314, R_t/(2gE) 0.9731; the member holds no trapping zero in its records to 40.0000
- M3d_s4: the fit's value T_f 25.3819, R_t/r_s 0.7860, R_t/(2gE) 0.8303; the member holds no trapping zero in its records to 0.4908

- members: M3b_s4, P64_s4, M3c_s4_nk; the exact form Y = Y_0 + A x^p through the 3 points (no residual; not a law):
  - T_f: p 0.412, Y_0 -0.5365, A 56.89
  - R_t_over_rs: p -0.539, Y_0 0.7627, A -0.02247
  - R_t_over_2gE: p -0.697, Y_0 0.7711, A -0.008634

### 5.2 The same line with the members' own coldness x = f_kin/0.3 (the composer's cold-gas conversion; a diagnostic)

- model: 2-coefficient diagnostic Y = c0 + c1 x, x = f_kin/0.3 (an equivalent cold-isotropic kinetic parameter of each stored preparation, the composer's cold-gas conversion)
- members: M3b_s4, P64_s4, M3c_s4_nk (3 members, 2 coefficients, residual degrees of freedom 1)
- x per member: M3b_s4 0.40305, P64_s4 0.35426, M3c_s4_nk 0.43149

| Y | coefficients | member: Y, fit, residual | rms residual over members; per residual degree of freedom | max |residual| |
|---|---|---|---|---|
| T_f | c0 -22.5324, c1 89.3877 (c0 from first-record clocks; cadence error up to the record step per member; not an intercept) | M3b_s4: 5.3750, 13.4948, -8.1198; P64_s4: 12.1250, 9.1341, +2.9909; M3c_s4_nk: 21.1667, 16.0377, +5.1290 | 5.81e+00; 1.01e+01 | 8.12e+00 |
| R_t_over_rs | c0 0.40831, c1 0.327854 (first-record radii, at the sampling scatter of the first record; not a radius law) | M3b_s4: 0.3287, 0.5404, -0.2117; P64_s4: 0.6024, 0.5245, +0.0780; M3c_s4_nk: 0.6835, 0.5498, +0.1337 | 1.51e-01; 2.62e-01 | 2.12e-01 |
| R_t_over_2gE | c0 0.540623, c1 0.117059 (first-record radii, at the sampling scatter of the first record; not a radius law) | M3b_s4: 0.3727, 0.5878, -0.2151; P64_s4: 0.6613, 0.5821, +0.0792; M3c_s4_nk: 0.7270, 0.5911, +0.1359 | 1.54e-01; 2.66e-01 | 2.15e-01 |

- M3a_s4: the fit's value T_f 25.1497, R_t/r_s 0.5832, R_t/(2gE) 0.6031; the member holds no trapping zero in its records to 40.0000
- M3d_s4: the fit's value T_f 22.0497, R_t/r_s 0.5718, R_t/(2gE) 0.5990; the member holds no trapping zero in its records to 0.4908

### 5.3 The dependence at r_s/a 64 (2 formed members)

| member | m a | p_F/m isotropic | p_F/m own (cold-gas conversion) | first saved zero T_f/r_s | record step | R_t/a | R_t/r_s | root fraction | gate | class |
|---|---|---|---|---|---|---|---|---|---|---|
| M3b_s4 | 0.4 | 0.064 | 0.635 | 5.3750 | 0.125 | 21.039 | 0.3287 | 0.3727 | 0.190 | certified |
| P64_s4 | 0.2 | 0.162 | 0.595 | 12.1250 | 0.125 | 38.556 | 0.6024 | 0.6613 | 0.104 | certified |
| M3a_s4 | 0.1 | 0.407 | 0.730 | none to 40.0000 | - | - | - | - | - | - |

## 6 The convergence tables

### 6.1 Per member

| member | script; solver | engine entries in the records | records (dropped without E) | converged True as stored | counting records | converged False as stored (not a gate) | kept residual above 1e-3 | confirming residual above 1e-3: count; clocks; max (at) (Newton: solve_v32's; plain: the entry after the kept iterate, on the records kept unconverged) | residual max (at) | max over the records up to the last converged record | banded records in the certified extent † (outside it) | 30.3 prefix's last counting record | last converged record (certified prefix) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 1.00e-06 (27.1250) | 1.00e-06 | none (none) | 40.0000 | 40.0000 |
| P64_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.97e-07 (15.8750) | 9.97e-07 | none (none) | 40.0000 | 40.0000 |
| M3a_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.99e-07 (21.2500) | 9.99e-07 | none (none) | 40.0000 | 40.0000 |
| M3c_s4_nk | engine_collapse_v34.py; newton (at most 30 steps, 40 Krylov vectors, set resolve, ends above 0.001) | engine_collapse_v25 (0.000 to 22.167, 267) | 267 (0) | 265 | 266 | 2 | 1 | 1; 22.1667; 3.34e-02 (22.1667) | 7.90e-03 (22.1667) | 2.21e-04 | 22.0000 † (none) | 22.0833 | 22.0833 |
| M3d_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 0.491, 21) | 21 (0) | 16 | 21 | 5 | 0 | 0; none; 1.82e-05 (of 5 records kept unconverged, 5 with a following entry) | 1.69e-06 (0.4663) | 1.69e-06 | none (none) | 0.4908 | 0.4908 |
| M3b_s8 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.88e-07 (6.2500) | 9.88e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s2 | engine_collapse_v34.py; newton (at most 30 steps, 40 Krylov vectors, set resolve, ends above 0.001) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; 2.99e-06 (0.0000) | 9.89e-07 (7.8750) | 9.89e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s8 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.93e-07 (5.0000) | 9.93e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s4_tau4 | engine_collapse_v19.py; plain (at most 20 iterations) | engine_collapse_v19 (0.000 to 40.000, 641) | 641 (0) | 640 | 641 | 1 | 0 | 0; none; 8.81e-04 (of 1 records kept unconverged, 1 with a following entry) | 6.18e-05 (13.4375) | 6.18e-05 | 13.4375 † (none) | 40.0000 | 40.0000 |
| P64_s4_tau2 | engine_collapse_v25.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 21.500, 689) | 689 (0) | 686 | 686 | 3 | 1 | 3; 12.7812, 12.8125, 13.4062; 7.08e-03 (of 3 records kept unconverged, 3 with a following entry) | 3.91e-03 (12.8125) | 9.98e-07 | none (none) | 12.7500 | 12.7500 |
| P64_s4_c19 | engine_collapse_v19.py; plain (at most 20 iterations) | None (0.000 to 11.000, 89); engine_collapse_v19 (11.125 to 40.000, 232) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.97e-07 (15.8750) | 9.97e-07 | none (none) | 40.0000 | 40.0000 |
| M3a_s2 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 277 | 292 | 44 | 23 | 26; 0.0000, 0.7500, 0.8750, 1.1250, 1.2500, 1.3750, 1.5000, 1.6250 (and 18 more); 2.54e-01 (of 44 records kept unconverged, 33 with a following entry) | 6.39e-02 (3.5000) | - | none (0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 1.0000 (and 3 more)) | 1.0000 | none |
| M3a_s4_tau4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 16.000, 257) | 257 (0) | 257 | 257 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.99e-07 (1.1250) | 9.99e-07 | none (none) | 16.0000 | 16.0000 |
| M3c_s4_plain | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 22.083, 266) | 266 (0) | 266 | 266 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.88e-07 (18.5000) | 9.88e-07 | none (none) | 22.0833 | 22.0833 |

### 6.2 The records that do not count, with their gates (complete)

- M3c_s4_nk: 1 records do not count: 22.1667 (kept residual above 1e-3; confirming residual above 1e-3; residual 7.90e-03, confirming 3.34e-02)
- P64_s4_tau2: 3 records do not count: 12.7812 (confirming residual above 1e-3; residual 6.96e-05, confirming 3.08e-03); 12.8125 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.91e-03, confirming 7.08e-03); 13.4062 (confirming residual above 1e-3; residual 3.63e-04, confirming 2.59e-03)
- M3a_s2: 29 records do not count: 0.0000 (confirming residual above 1e-3; residual 9.57e-05, confirming 1.02e-03); 0.7500 (confirming residual above 1e-3; residual 3.16e-04, confirming 3.38e-03); 0.8750 (confirming residual above 1e-3; residual 6.44e-04, confirming 2.42e-03); 1.1250 (confirming residual above 1e-3; residual 6.92e-04, confirming 2.38e-03); 1.2500 (confirming residual above 1e-3; residual 1.52e-04, confirming 1.47e-03); 1.3750 (confirming residual above 1e-3; residual 7.05e-04, confirming 5.93e-03); 1.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.15e-03, confirming 1.34e-02); 1.6250 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.88e-03, confirming 1.67e-02); 1.7500 (kept residual above 1e-3; confirming residual above 1e-3; residual 2.35e-03, confirming 1.03e-02); 1.8750 (kept residual above 1e-3; confirming residual above 1e-3; residual 7.68e-03, confirming 7.93e-03); 2.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 6.84e-03, confirming 1.10e-01); 2.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.40e-02, confirming 4.23e-02); 2.3750 (kept residual above 1e-3; residual 5.94e-03); 2.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 5.60e-02, confirming 1.53e-01); 2.6250 (kept residual above 1e-3; residual 1.92e-03); 2.7500 (kept residual above 1e-3; confirming residual above 1e-3; residual 7.47e-03, confirming 2.70e-02); 3.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 4.67e-03, confirming 6.87e-03); 3.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 2.02e-02, confirming 2.96e-02); 3.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 6.39e-02, confirming 1.09e-01); 3.7500 (kept residual above 1e-3; confirming residual above 1e-3; residual 4.33e-02, confirming 5.54e-02); 4.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.61e-02, confirming 1.41e-01); 4.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 2.61e-02, confirming 2.04e-01); 4.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 5.54e-02, confirming 2.54e-01); 4.7500 (kept residual above 1e-3; confirming residual above 1e-3; residual 4.49e-02, confirming 7.43e-02); 5.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.40e-02, confirming 6.60e-02); 5.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.32e-02, confirming 4.07e-02); 5.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.94e-02, confirming 2.64e-02); 5.7500 (kept residual above 1e-3; residual 1.02e-03); 6.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 9.24e-03, confirming 1.83e-02)

### 6.3 The stored controls (control minus member)

| member | control | kind | formation classes (member / control) | dT_f (control - member) | dR_t (a) | dR_t/R_t | d(R_t/r_s) | d(R_t/(2gE)) | clocks counting and certified in both (from to) | max |d E/E0| | max |d min(N - |v|)| | max |d R_c|/r_s | max |d R_t| (a); max |d R_t|/R_t (at) where both hold a zero (count) | counting records regardless of certification (labelled diagnostic): common clocks; max |d R_t| (a); max |d R_t|/R_t (at); at 16 (count) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3b_s4 | M3b_s8 | control_sigma | certified / certified | 1.3750 | 5.191 | 0.2467 | 0.0811 | 0.0898 | 321 (0.0000 to 40.0000) | 1.24e-02 | 2.94e-01 | 6.29e-01 | 4.625; 0.1499 (6.7500) (2) | 321; 4.625; 0.1499 (6.7500); - (2) |
| M3a_s4 | M3a_s2 | control_sigma | - | - | - | - | - | - | 0 (- to -) | - | - | - | -; - (-) (0) | 292; -; - (-); - (0) |
| M3a_s4 | M3a_s4_tau4 | control_step | - | - | - | - | - | - | 129 (0.0000 to 16.0000) | 5.31e-05 | 3.96e-04 | 3.12e-03 | -; - (-) (0) | 129; -; - (-); - (0) |
| M3c_s4_nk | M3c_s4_plain | control_solver | certified / certified | 0.0000 | -0.010 | -0.0002 | -0.0001 | -0.0001 | 266 (0.0000 to 22.0833) | 1.38e-05 | 1.25e-04 | 4.58e-05 | 0.010; 0.0002 (21.1667) (12) | 266; 0.010; 0.0002 (21.1667); - (12) |
| P64_s4 | P64_s2 | control_sigma | certified / certified | -0.1250 | -0.458 | -0.0119 | -0.0072 | -0.0069 | 321 (0.0000 to 40.0000) | 8.16e-02 | 3.11e-01 | 3.99e-01 | 11.303; 0.2223 (16.0000) (216) | 321; 11.303; 0.2223 (16.0000); 0.2223 (216) |
| P64_s4 | P64_s8 | control_sigma | certified / certified | 0.6250 | 5.971 | 0.1549 | 0.0933 | 0.1007 | 321 (0.0000 to 40.0000) | 6.16e-02 | 4.05e-01 | 5.74e-01 | 6.770; 0.1803 (28.3750) (64) | 321; 6.770; 0.1803 (28.3750); 0.0408 (64) |
| P64_s4 | P64_s4_tau4 | control_step | certified / certified | -0.1250 | -1.233 | -0.0320 | -0.0193 | -0.0253 | 321 (0.0000 to 40.0000) | 4.49e-02 | 1.67e-01 | 5.81e-01 | 7.871; 0.2044 (20.1250) (183) | 321; 7.871; 0.2044 (20.1250); 0.0649 (183) |
| P64_s4 | P64_s4_tau2 | control_step | certified / certified | -0.2188 | -3.887 | -0.1008 | -0.0607 | -0.0726 | 103 (0.0000 to 12.7500) | 3.16e-02 | 1.61e-01 | 1.06e-01 | 4.022; 0.1043 (12.1250) (6) | 173; 6.045; 0.1556 (21.1250); 0.1174 (36) |
| P64_s4 | P64_s4_c19 | comparator | certified / certified | 0.0000 | 0.000 | 0.0000 | 0.0000 | -0.0000 | 321 (0.0000 to 40.0000) | 1.24e-08 | 3.81e-11 | 1.46e-11 | 0.000; 0.0000 (17.2500) (216) | 321; 0.000; 0.0000 (17.2500); 0.0000 (216) |

The (0.4, 64) and (0.1, 96) members have no stored step control; the (0.1, 96) member has no stored averaging-scale control (its plain comparator agrees with it over the common certified clocks); the (0.1, 64) member's controls hold no trapping zero; the (0.2, 64) step controls are certified to 40 (tau 4) and 12.75 (tau 2), and the sigma/a 2 composite to 40 (6.1). P64_s2 against P64_s4 (certified / certified): |dT_f| 0.1250 r_s, |dR_t|/R_t at formation 1.2%, max |dR_t|/R_t over 216 common certified zero records 22.2% at 16.0000; within both criteria at formation, the radius difference over the common records exceeding 10 percent.

## 7 The catalogs behind the chain-column onsets

| member | catalog read from | chain columns per channel (kappa: columns) | top channel | largest column count | 4 a times it (upper bound) | head comparison of the rerun preparation (N_B, E0, f_kin relative differences) |
|---|---|---|---|---|---|---|
| M3b_s4 | status file t0 line of M3b_s4 | 1:5 2:4 3:2 4:1 5:0 6:0 7:0 8:0 9:0 10:0 | 4 | 5 | 20 a | - (status file) |
| P64_s4 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| M3a_s4 | status file t0 line of M3a_s4_tau4 (same preparation arguments) | 1:6 2:5 3:4 4:4 5:3 6:2 7:2 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| M3c_s4_nk | status file t0 line of M3c_s4_nk | 1:7 2:6 3:5 4:4 5:4 6:3 7:2 8:2 9:1 10:0 | 9 | 7 | 28 a | - (status file) |
| M3d_s4 | not available (not available) | - | - | - | - | - |
| M3b_s8 | status file t0 line of M3b_s8 | 1:5 2:4 3:2 4:1 5:0 6:0 7:0 8:0 9:0 10:0 | 4 | 5 | 20 a | - (status file) |
| P64_s2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s8 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s4_tau4 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s4_tau2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s4_c19 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| M3a_s2 | status file t0 line of M3a_s4_tau4 (same preparation arguments) | 1:6 2:5 3:4 4:4 5:3 6:2 7:2 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| M3a_s4_tau4 | status file t0 line of M3a_s4_tau4 | 1:6 2:5 3:4 4:4 5:3 6:2 7:2 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| M3c_s4_plain | status file t0 line of M3c_s4 (same preparation arguments) | 1:7 2:6 3:5 4:4 5:4 6:3 7:2 8:2 9:1 10:0 | 9 | 7 | 28 a | - (status file) |

## 8 Not composed, truncated or open

- The (0.1, 64) member M3a_s4 holds no trapping zero in its records to 40.000 (its requested end), nor do its sigma/a 2 control (its first record does not count; first above 1e-3 at 1.5) and its tau 4 control (to 16.000): the mass ladder's formation readouts at r_s/a 64 are 2 members.
- The (0.1, 96) member M3c_s4_nk (Newton, v36j) is certified to 22.0833: the record 22.0000 counts banded (kept 2.21e-04, confirming 3.89e-04); its end record 22.1667 is kept at 7.90e-03 and does not count. The busi member line's "to 22.167" names that end record. Its formation 21.1667 lies inside the certified extent.
- The (0.05, 326) member M3d_s4 has not reached formation: records to 0.4908, certified to 0.4908 (its 5 records kept unconverged have kept residuals at most 1.69e-06 and following entries at most 1.82e-05); it is the kappa_sea 12 plain run (v36e), not the isotropic set the requirement names; its catalog is not composed (no t0 line; a 3912-site preparation was not rerun).
- No step control exists for the (0.4, 64) and (0.1, 96) members, and no averaging-scale control for the (0.1, 96) member.
- The tau 2 control of the (0.2, 64) member is certified to 12.75 (the record 12.78125: confirming residual above 1e-3; kept 6.96e-05, confirming 3.08e-03), the tau 4 control through its last record (its record 13.4375 kept unconverged counts banded: kept 6.18e-05, following entry 8.81e-04); the tau 2 control's later closings, second roots and radii are uncertified.
- Not composed (the reading's routing): the birth located between records (busi B7), the T = 0 valley census for the per-valley onsets, the (32, 4, 8) step member, the core's compression radius and bare energy per particle, nu, chi^(1/3), epsilon_c and c_b.

## 9 Figure data and figures

- figure: <internal>/MASS_ladder_v1.3_fig.png (and .pdf), drawn by <internal>/ladders_figs_v1_3.py from the arrays below
- arrays: <internal>/ladders_readout_v1_3.npz (per member per record: T/r_s, E/E0, R_t, R_t/r_s, R_t/(2gE), min(N - |v|), R_c, residual, confirming residual, converged as stored, counts, certified, conf_fail, capped, above, the number of zeros, tangencies, pockets, inversions, slope at R_t; keys <member>__<column>), md5 in SOURCES
- numbers: <internal>/ladders_readout_v1_3.json (formation, episodes, matched clocks, fits with coefficients and residuals, control spreads, catalogs)

## 10 SOURCES

| member | run json (as read) | bytes | md5 | script field |
|---|---|---|---|---|
| M3b_s4 | <internal>/collapse_lad_M3b_s4_m0p4_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 4510403 | 6209f14cf64e8a4f614babccb28eb04c | engine_collapse_v34.py |
| P64_s4 | <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json | 4857142 | d7d69adcea7a43f8375ab4c7c5a80efc | engine_collapse_v34.py |
| M3a_s4 | <internal>/collapse_lad_M3a_s4_m0p1_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 4425449 | 778dc5cef53012b71461a2e1d2de1044 | engine_collapse_v34.py |
| M3c_s4_nk | <internal>/collapse_lad_M3c_s4_nk_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36j.json | 4783721 | 60f92fbfae76f621867f543eaad49029 | engine_collapse_v34.py |
| M3d_s4 | <internal>/collapse_lad_M3d_s4_m0p05_rs326_n3912_R04_k12_tau8_ell4_prep4_A_R2_avg_plain_v36e.json | 381605 | 726cb35fb005e3ef2ea05c3e3adfa38f | engine_collapse_v34.py |
| M3b_s8 | <internal>/collapse_lad_M3b_s8_m0p4_rs64_n2048_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json | 4375774 | 0b834bc19856a2ae6c9f46ce3bd21445 | engine_collapse_v34.py |
| P64_s2 | <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json | 7014125 | 0a005ac5f66bef02a0104e4278867765 | engine_collapse_v34.py |
| P64_s8 | <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json | 4650292 | d1ca037a77c198aa83b33055495eceff | engine_collapse_v34.py |
| P64_s4_tau4 | <internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg.json | 9046968 | f2e1d8a7483447426855e3211535a367 | engine_collapse_v19.py |
| P64_s4_tau2 | <internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg.json | 9405229 | 9ba6f6ff302d8a903925f20583c96656 | engine_collapse_v25.py |
| P64_s4_c19 | <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json | 4083175 | e27e457d37615cff8d5216f05a494f89 | engine_collapse_v19.py |
| M3a_s2 | <internal>/collapse_lad_M3a_s2_m0p1_rs64_n2048_R04_k10_tau8_ell2_prep4_A_R2_avg_plain_v36d.json | 4699087 | d8aa172c70456055b6867f0489e3cf96 | engine_collapse_v34.py |
| M3a_s4_tau4 | <internal>/collapse_lad_M3a_s4_tau4_m0p1_rs64_n2048_R04_k10_tau4_ell4_prep4_A_R2_avg_plain_v36d.json | 3534693 | 66382686ad07f2ca581659153c90767a | engine_collapse_v34.py |
| M3c_s4_plain | <internal>/collapse_lad_M3c_s4_m0p1_rs96_n3072_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 3695488 | 6489517c34dccd5c536e99a7a03d3bb3 | engine_collapse_v34.py |

- catalog (chain columns): <internal>/catalog_v1.json (md5 007fec3e0073e36cc402522ba1a36795), written by <internal>/ladders_catalog_v1.py
- catalog (chain columns): <internal>/catalog_v1_A.json (md5 50084b438ea991ecfd4edc02c9f7cfd1), written by <internal>/ladders_catalog_v1.py
- catalog (chain columns): <internal>/catalog_v1_B.json (md5 37a7ec570643c2a8cd7cc4bf0a4c2a72), written by <internal>/ladders_catalog_v1.py
- producer: <internal>/ladders_readout_v1_3.py (md5 20048af9566537dc7540b4dad33b4e8a), version 1.2's producer with the production certification rule, rerun on the same stored files; helper engine_ladder_compare_v5.py (md5 94ac400330b2ebf032cdb1312c520d53)
- certification inputs checked by the producer: <internal>/certext_production_v1.json (md5 64c2dfd1e5a4bccaf61e1ad8be2ee717, LADDER_certification_list_v1.1) and <internal>/certext_followup_v1.json (md5 0bf27951ca23c9b6012aadde84038989, written by <internal>/certext_followup_v1.py, the following-entry test)
- composer: <internal>/ladders_compose_v1_3.py and <internal>/compose_text_v1_3.py; the readout json <internal>/ladders_readout_v1_3.json (md5 930e75da87c1559f0318ed7a8a5206ed); the arrays <internal>/ladders_readout_v1_3.npz (md5 b4c8b2f6a285dfd929f9895a9d81981d), mirrored on the box at <internal>/v1_3 with its md5 manifest
- engine of the catalog's rerun preparations: <internal>/engine_collapse_v36d.py (md5 0315e0826ee872ee194474876ef523f9)
- the reading applied: deliveries/lead_gaps/FableAPI_reading_LADDERS_v1.md (reviewer A Part A, reviewer B Part B, the lane's cross-check); the certification rule applied: deliveries/lead_gaps/FableAPI_cert_rule_v1.md
- W4 (a) comparator numbers quoted in 5.1: <internal>/s4_density_continuum_rest_lapse-on_adaptive.json (through deliveries/claude_alt_solve/W5_W4_formation_attribution_v1.md section 4)
