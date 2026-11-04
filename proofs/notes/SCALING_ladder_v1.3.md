# SCALING_ladder_v1.3: the size ladder at m a 0.2, sigma/a 4 (r_s/a 16, 32, 64, 128), the width cross at r_s/a 64 and 128, and the 2 r_s releases, composed from the stored members, with the corrections of the correctness reading

Version 1.3 of SCALING_ladder_v1 (versions 1, 1.1 and 1.2 stay as issued). Composed by ladders_readout_v1_3.py and ladders_compose_v1_3.py from stored files only. The builder composes; no sentence here reads the physics. The size member at r_s/a 32 is the 1-field production c31_rs32_A_R2_tau2_ell4_prep4_avg (tau 2, 1024 sites).

## Changes from v1.2

- The certification rule is the production rule (deliveries/lead_gaps/FableAPI_cert_rule_v1.md): a record is certified when its kept residual and its confirming residual are both at most 1e-3, and a history is certified to its last record before the first that fails; records counting with either residual between 1e-4 and 1e-3 are banded and carry a dagger (†) in every table. Only the certification inputs changed (the producer's counting test, ladders_readout_v1_3.py); every record, clock, radius and fit input is read as in version 1.2. The converged flag no longer gates. A Newton record's confirming residual is solve_v32's (version 1.2 held it to stop_fields 1e-6). A plain record kept unconverged is confirmed by its iteration_history entry after the kept iterate, which the ruling asks for before the tables recompose (the plain mixer applies the kept fields, so that entry is the map's change out of them); a plain record whose kept iterate is its last entry, and every converged record, rest on the kept residual.
- That test shortens extents of LADDER_certification_list_v1.1 (certext_production_v1.json, which tested the kept residual alone on plain members): S4_128_s4 3.8125 to 3.5625 (record 3.625: kept 5.30e-04, following entry 2.26e-03); P64_s4_tau2 12.78125 to 12.75 (record 12.78125: kept 6.96e-05, following entry 3.08e-03); S4_R2_m0p2 40 to 6 (record 6.125: kept 8.42e-04, following entry 3.17e-02). It shortens no other member (certext_followup_v1.py); the producer's extents equal the list's on the other 23 of 27 members.
- Certified extents that move (T/r_s): S4_128_s4 3.4375 to 3.5625; RS32_s2 none to 40; P64_s2 none to 40; P64_s4_tau4 13.375 to 40; S4_128_s4_nk_v36m 0.5 to 2.875; S4_128_s4_nk_v36q 0.5 to 2.75; S4_R2_m0p1 31.875 to 32.125. Unchanged and short of the last record: P64_s4_tau2 (12.75), S4_R2_m0p2 (6).
- Formations that become certified: RS32_s2 at 14.125 (v1.2: 30.3 only); P64_s2 at 12 (v1.2: 30.3 only). No formation loses its certification; S4_128_s4's at 11.0625 stays uncertified.
- Banded records inside the certified extents, marked † wherever a table prints them: RS32_s2 0 (kept 1.86e-04, confirming 1.23e-04); P64_s4_tau4 13.4375 (kept 6.18e-05, confirming 8.81e-04); S4_R2_m0p1 32 (kept 1.07e-04, confirming 8.66e-04); S4_R2_m0p1 32.125 (kept 2.88e-04, confirming 3.17e-04); none is a formation record. Counting banded records past an extent carry (unc.) † (7 in this document's members, listed in 6.1).
- Values that change, per member (every other member value is unchanged):
  - S4_128_s4: counting records 274 to 285; non-counting records 32 to 21; first non-counting record 3.625 (confirming residual above 1e-3); min R_c over the certified records 312.11 a at 3.4375 to 310.81 a at 3.5625 (2.428 r_s); (n.c.) becomes (unc.) at 5
  - RS32_s2: counting records 637 to 641; non-counting records 4 to 0; min R_c over the certified records none (no certified record) to 10.09 a at 15.6875 (0.315 r_s); episode 1 depth over certified records none to -0.3617 (at 15.5625); episode 2 depth over certified records none to -0.2235 (at 26.5625); episode 3 depth over certified records none to -0.1978 (at 37.0625); (unc.) removed at matched clocks 0 to 40; (n.c.) removed at 0; (unc.) removed at T_f + 0, 0.5, 1, 2, 4
  - P64_s2: counting records 313 to 321; non-counting records 8 to 0; min R_c over the certified records none (no certified record) to 10.85 a at 12.875 (0.169 r_s); episode 1 depth over certified records none to -0.6618 (at 12.75); (unc.) removed at matched clocks 0 to 40; (n.c.) removed at 0, 6; (unc.) removed at T_f + 0, 0.5, 1, 2, 4
  - P64_s4_tau4: counting records 640 to 641; non-counting records 1 to 0; episode 2 depth over certified records none to -0.2408 (at 21.75); episode 3 depth over certified records none to -0.1952 (at 29.25); (unc.) removed at matched clocks 14 to 40; (unc.) removed at T_f + 2, 4
  - P64_s4_tau2: first non-counting record 12.78125 (confirming residual above 1e-3)
  - S4_128_s4_nk_v36m: counting records 44 to 47; non-counting records 4 to 1; first non-counting record 2.9375 (kept residual above 1e-3); min R_c over the certified records 329.02 a at 0.5 to 317.33 a at 2.875 (2.479 r_s); (unc.) removed at matched clocks 1 to 2
  - S4_128_s4_nk_v36q: counting records 42 to 45; non-counting records 4 to 1; first non-counting record 2.8125 (kept residual above 1e-3; confirming residual above 1e-3); min R_c over the certified records 329.02 a at 0.5 to 318.36 a at 2.75 (2.487 r_s); (unc.) removed at matched clocks 1 to 2
  - S4_R2_m0p2: counting records 318 to 319; non-counting records 3 to 2; first non-counting record 6.125 (confirming residual above 1e-3)
  - S4_R2_m0p1: counting records 301 to 307; non-counting records 20 to 14; first non-counting record 32.25 (kept residual above 1e-3; confirming residual above 1e-3); min R_c over the certified records 21.32 a at 31.875 to 16.94 a at 32.125 (0.265 r_s); episode 1 depth over certified records -0.2810 to -0.3778 (at 32.125); (unc.) removed at matched clocks 32; (n.c.) removed at 32; (n.c.) becomes (unc.) at 36
- Control spreads that change (6.3, control minus member, over clocks counting and certified in both):
  - P64_s2 against P64_s4: common certified clocks 0 to 321; last common clock none to 40; max |d E/E0| none to 0.08159; max |d min(N - |v|)| none to 0.3114; max |d R_c|/r_s none to 0.3994; common zero records 0 to 216; max |d R_t| (a) none to 11.3; max |d R_t|/R_t none to 0.2223; at none to 16; classes certified / 30.3 only to certified / certified; raw common clocks 313 to 321
  - P64_s4_tau4 against P64_s4: common certified clocks 108 to 321; last common clock 13.375 to 40; max |d E/E0| 0.03249 to 0.04485; max |d min(N - |v|)| 0.106 to 0.1668; max |d R_c|/r_s 0.06955 to 0.5809; common zero records 11 to 183; max |d R_t| (a) 2.949 to 7.871; max |d R_t|/R_t 0.07649 to 0.2044; at 12.125 to 20.125
  - RS32_s2 against RS32_s4: common certified clocks 0 to 641; last common clock none to 40; max |d E/E0| none to 0.01695; max |d min(N - |v|)| none to 0.4538; max |d R_c|/r_s none to 0.6722; common zero records 0 to 50; max |d R_t| (a) none to 4.448; max |d R_t|/R_t none to 0.2061; at none to 15.8125; classes certified / 30.3 only to certified / certified; raw common clocks 637 to 641
  - S4_128_s4_nk_v36m against S4_128_s4: common certified clocks 9 to 47; last common clock 0.5 to 2.875; max |d E/E0| 3.609e-11 to 5.828e-06; max |d min(N - |v|)| 1.658e-12 to 1.625e-06; max |d R_c|/r_s 2.989e-12 to 1.349e-06; raw common clocks 44 to 47
  - S4_128_s4_nk_v36q against S4_128_s4: common certified clocks 9 to 45; last common clock 0.5 to 2.75; max |d E/E0| 3.609e-11 to 4.663e-06; max |d min(N - |v|)| 1.658e-12 to 1.291e-06; max |d R_c|/r_s 2.989e-12 to 8.265e-07; raw common clocks 42 to 45
  - S4_128_s8 against S4_128_s4: common certified clocks 56 to 58; last common clock 3.4375 to 3.5625; max |d min(N - |v|)| 0.0006728 to 0.0006888; max |d R_c|/r_s 0.001582 to 0.001671; raw common clocks 274 to 285
  - S4_128_s16 against S4_128_s4: common certified clocks 56 to 58; last common clock 3.4375 to 3.5625; max |d min(N - |v|)| 0.003126 to 0.003199; max |d R_c|/r_s 0.007866 to 0.008308; raw common clocks 274 to 285
  - S4_R2_m0p2 against P64_s4: raw common clocks 318 to 319
  - S4_R2_m0p1 against M3a_s4: common certified clocks 256 to 258; last common clock 31.875 to 32.125; max |d E/E0| 0.02466 to 0.03676; max |d min(N - |v|)| 0.7606 to 0.8571; max |d R_c|/r_s 2.19 to 2.257; raw common clocks 301 to 307
- Fits (5.3): the certified width cross gains P64_s2: 4 points to 5 (P64_s2, P64_s4, P64_s8, S4_128_s8, S4_128_s16), residual degrees of freedom 0 to 1; T_f coefficients 15.75, 1.172, 3.066, 0.5203 to 20.42, 2.134, 0.541, -3.987e-15, residuals +0.0833, -0.1667, +0.0833, +0.0000, -0.0000; it equals version 1.2's labelled set with the sigma/a 2 composite, which therefore no longer prints separately; P64_s2 leaves the uncertified list of the every-record fit; the sigma/a 2 to 4 step at 64 is now certified in the per-doubling table; the design rank's certified rows go from 8 to 10 (adding RS32_s2 and P64_s2), rank 8 to 8, rows beyond the rank 0 to 2. No size-ladder fit (5.1, 5.2) and no release pair (5.4) changes.
- The factor-2 averaging control at r_s/a 32 is now certified on both sides and fails: RS32_s2 against RS32_s4 (certified / certified): |dT_f| 1.6875 r_s, |dR_t|/R_t at formation 6.9%, max |dR_t|/R_t over 50 common certified zero records 20.6% at 15.8125; the clock spread exceeds 0.5 r_s; the radius difference over the common records exceeds 10 percent: the control fails.
- Wording: the definitions paragraph, the end-state notes of section 8, the convergence table's columns (confirming residual above 1e-3 for both solvers, the band column) and the sentences that quoted the strict extents now read the producer's extents. The change lists of versions 1.1 and 1.2 stay in those versions. Every changed value with its old and new entry (1906 paths, 970 of them new per-record band flags) is in ladders_diff_v1_3.json.

## 0 The members' end states

- S4_16_s4 (0.2, 16) s4 t2 (size member (0.2, 16), near-maximal and outside the hydrodynamic hierarchy on both ends; 512 sites, tau 2, N_B 160): no trapping zero in its records to 40.0000; last converged record 40.0000.
- RS32_s4 (0.2, 32) s4 t2 (size member (0.2, 32), the 1-field production c31_rs32_A_R2_tau2_ell4_prep4_avg; 1024 sites, tau 2, N_B 320): first saved zero 15.8125 at R_t 21.586 a (0.6746 r_s, root fraction 0.7251, gate 0.185), certified (inside the certified prefix); last converged record 40.0000.
- P64_s4 (0.2, 64) s4 t8 (mass and size member (0.2, 64), the production's history (a reproduction of the v19 record); 2048 sites, tau 8, N_B 640): first saved zero 12.1250 at R_t 38.556 a (0.6024 r_s, root fraction 0.6613, gate 0.104), certified (inside the certified prefix); last converged record 40.0000.
- S4_128_s4 (0.2, 128) s4 t8 (size member (0.2, 128); 4096 sites, tau 8, N_B 1280): first saved zero 11.0625 at R_t 61.815 a (0.4829 r_s, root fraction 0.5300, gate 0.065), uncertified: after the member's first record above 1e-3 (3.8750); last converged record 3.5625, first non-counting record 3.6250 (confirming residual above 1e-3).

Width points, preparation members, controls and comparators:

- S4_128_s8 (0.2, 128) s8 t8 (width point (128, sigma/a 8), finished to 40; 4096 sites, tau 8, N_B 1280): first saved zero 11.1875 at R_t 72.936 a (0.5698 r_s, root fraction 0.6253, gate 0.110), certified (inside the certified prefix); last converged record 40.0000.
- S4_128_s16 (0.2, 128) s16 t8 (width point (128, sigma/a 16), at the gate's edge; 4096 sites, tau 8, N_B 1280): first saved zero 11.5625 at R_t 79.962 a (0.6247 r_s, root fraction 0.6838, gate 0.200), certified (inside the certified prefix); last converged record 40.0000.
- S4_R2_m0p2 (0.2, 64) s4 t8 (preparation member, release at 2 r_s, (0.2, 64); 2048 sites, tau 8, N_B 640): first saved zero 5.0000 at R_t 40.393 a (0.6311 r_s, root fraction 0.7431, gate 0.099), certified (inside the certified prefix); last converged record 6.0000, first non-counting record 6.1250 (confirming residual above 1e-3).
- S4_R2_m0p1 (0.1, 64) s4 t8 (preparation member, release at 2 r_s, (0.1, 64); 2048 sites, tau 8, N_B 1280): first saved zero 30.6250 at R_t 47.972 a (0.7496 r_s, root fraction 0.7842, gate 0.083), certified (inside the certified prefix); last converged record 32.1250, first non-counting record 32.2500 (kept residual above 1e-3; confirming residual above 1e-3).
- S4_16_s2 (0.2, 16) s2 t2 (sigma/a 2 control of S4_16_s4; 512 sites, tau 2, N_B 160): no trapping zero in its records to 40.0000; last converged record 40.0000.
- S4_16_s4_tau1 (0.2, 16) s4 t1 (tau 1 control of S4_16_s4 (to 16); 512 sites, tau 1, N_B 160): no trapping zero in its records to 16.0000; last converged record 16.0000.
- RS32_s2 (0.2, 32) s2 t2 (sigma/a 2 control of RS32_s4 (Newton, engine v32); 1024 sites, tau 2, N_B 320): first saved zero 14.1250 at R_t 20.095 a (0.6280 r_s, root fraction 0.6731, gate 0.100), certified (inside the certified prefix); last converged record 40.0000.
- RS32_s4_tau1 (0.2, 32) s4 t1 (tau 1 control of RS32_s4 (engine v25); 1024 sites, tau 1, N_B 320): first saved zero 15.7188 at R_t 21.871 a (0.6835 r_s, root fraction 0.7331, gate 0.183), certified (inside the certified prefix); last converged record 40.0000.
- P64_s2 (0.2, 64) s2 t8 (sigma/a 2 point at 64: the composite history (Newton across its seams); 2048 sites, tau 8, N_B 640): first saved zero 12.0000 at R_t 38.098 a (0.5953 r_s, root fraction 0.6545, gate 0.052), certified (inside the certified prefix); last converged record 40.0000.
- P64_s8 (0.2, 64) s8 t8 (sigma/a 8 point at 64; 2048 sites, tau 8, N_B 640): first saved zero 12.7500 at R_t 44.527 a (0.6957 r_s, root fraction 0.7620, gate 0.180), certified (inside the certified prefix); last converged record 40.0000.
- P64_s4_tau4 (0.2, 64) s4 t4 (tau 4 control of P64_s4 (engine v19); 2048 sites, tau 4, N_B 640): first saved zero 12.0000 at R_t 37.323 a (0.5832 r_s, root fraction 0.6360, gate 0.107), certified (inside the certified prefix); last converged record 40.0000.
- P64_s4_tau2 (0.2, 64) s4 t2 (tau 2 control of P64_s4 (engine v25, to 21.5); 2048 sites, tau 2, N_B 640): first saved zero 11.9062 at R_t 34.669 a (0.5417 r_s, root fraction 0.5888, gate 0.115), certified (inside the certified prefix); last converged record 12.7500, first non-counting record 12.7812 (confirming residual above 1e-3).
- S4_128_s4_nk_v36m (0.2, 128) s4 t8 (Newton run of S4_128_s4 (v36m), stopped; 4096 sites, tau 8, N_B 1280): no trapping zero in its records to 2.9375; last converged record 2.8750, first non-counting record 2.9375 (kept residual above 1e-3).
- S4_128_s4_nk_v36q (0.2, 128) s4 t8 (Newton run of S4_128_s4 (v36q), stopped; 4096 sites, tau 8, N_B 1280): no trapping zero in its records to 2.8125; last converged record 2.7500, first non-counting record 2.8125 (kept residual above 1e-3; confirming residual above 1e-3).

Definitions (version 1.3; deliveries/lead_gaps/FableAPI_cert_rule_v1.md). A record is certified when the self-consistent solve of its fields has converged to a residual of at most 1e-3, the residual being the largest change of N or v over one iteration of the solve at its final fields; a residual epsilon shifts an exterior root of N - |v| by 2 r_s epsilon, about a tenth of a spacing at the threshold here, and a history is certified to its last record before the first that fails. Residuals between 1e-4 and 1e-3 are marked † in the tables, and a reading past a history's certified extent says so where it is made. As applied: both the kept residual (engine_ladder_compare_v5.rec_residual) and the confirming residual (a Newton record's solve_v32 confirming residual; a plain record kept unconverged, the iteration_history entry after the kept iterate) must be at most 1e-3; a record failing either does not count and is named. Two extents print for every member: the 30.3 prefix, ending before the first record whose kept residual exceeds 1e-3, and the certified extent (the certified prefix), ending before the first non-counting record of any kind. The last converged record is the last record of the certified extent. A formation is certified when its record lies inside the certified extent, and only such formations enter a fit. Formation is the first saved record holding a trapping zero (trapping_zeros non-empty), the birth lying in the interval (T_f minus the record step, T_f], whose width prints with each member; R_t at formation is the outer zero of the dense N + v on that first saved record and not the birth radius; the 2 zeros of every formation record print with no recorded tangency and a positive slope at R_t (the inner zero's simplicity is not tested). R_t/(2 g E) is the record's own stored ratio at its charge E (g 1/4): the root fraction. The root gate is sigma/R_t with sigma = ell. E/E0 is against the head's E0 of the ell 4 preparation, so it is not 1 at T = 0 for the re-averaged controls. N_B (m a)^3/1.087 is the nominal count ratio (to the displayed approximation the rest-energy ratio N_B m/E_max), not the record's charge ratio; the charge ratio E0/E_max uses the stored E0 and E_max = c_E/(g^(3/2) m^2) with c_E 0.135858. Labels: (n.c.) the record does not count; (unc.) the record lies after the member's certified extent; † the record counts and is banded (kept or confirming residual between 1e-4 and 1e-3).

## 1 The member table

Every member and control shares one preparation rule: the dust is the levitating well's zero-temperature filling within the sea set kappa = 1 to kappa_sea (engine prepare_dust with fill_lowest: the well of depth set by the release radius R_0, the lowest levels of all channels filled to the count N_B = r_s/(2 g m a), g = 1/4, the projection onto the clocked positive band), released from coordinate rest at R_0 = 4 r_s (2 r_s for the 2 preparation members), prepared at ell 4 and evolved at the member's ell. No member runs the isotropic catalog; each member's own onset prints as 4 a times the largest chain column count of its catalog at T = 0 (20 a for the production's set), an upper bound on the per-valley onset 4 a n_rad,max that the T = 0 valley census fixes (12 a to 20 a for the production's set by the flat-well control's split), beside the isotropic form 1.222 N_B^(1/3) a; the isotropic onset and the exclusion floor 0.49 N_B^(1/3) a are comparators, not demonstrated bounds on the chain's dust.

| member | role | m a | r_s/a | sigma/a | tau/a (tau/r_s) | sites | N_B | N_B (m a)^3/1.087 (nominal count ratio) | E0 (head) | 2 g E0 (a) | E0/(N_B m) | E0/E_max (charge ratio, c_E 0.135858) | f_kin (head) | p_F/m isotropic release form | (p_F/m)^2/5 isotropic (a release comparator) | own onset, chain columns (4 a times columns; upper bound) (/r_s) | isotropic 1.222 N_B^(1/3) a (/r_s) | exclusion floor 0.49 N_B^(1/3) a | records (first to last T/r_s; dropped without E) | last converged record (certified prefix) | 30.3 prefix's last counting record | first non-counting record (its gate) | end state |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S4_16_s4 | size member (0.2, 16), near-maximal and outside the hydrodynamic hierarchy on both ends | 0.2 | 16 | 4 | 2 (0.12500) | 512 | 160 | 1.18 | 30.9668 | 15.483 | 0.9677 | 1.14 | 0.1802 | 0.407 | 0.033127 | 12 a (0.750) | 6.6 a (0.415) | 2.7 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40; no trapping zero in any record |
| RS32_s4 | size member (0.2, 32), the 1-field production c31_rs32_A_R2_tau2_ell4_prep4_avg | 0.2 | 32 | 4 | 2 (0.06250) | 1024 | 320 | 2.36 | 59.7890 | 29.894 | 0.9342 | 2.20 | 0.1303 | 0.256 | 0.013147 | 16 a (0.500) | 8.4 a (0.261) | 3.4 a | 641 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s4 | mass and size member (0.2, 64), the production's history (a reproduction of the v19 record) | 0.2 | 64 | 4 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| S4_128_s4 | size member (0.2, 128) | 0.2 | 128 | 4 | 8 (0.06250) | 4096 | 1280 | 9.42 | 234.8342 | 117.417 | 0.9173 | 8.64 | 0.0985 | 0.102 | 0.002070 | 24 a (0.188) | 13.3 a (0.104) | 5.3 a | 306 (0.000 to 19.0625; 0) | 3.5625 | 3.8125 | 3.6250 (confirming residual above 1e-3) | stopped at 19.0625 of its requested 40; certified to 3.5625 |
| S4_16_s2 | sigma/a 2 control of S4_16_s4 | 0.2 | 16 | 2 | 2 (0.12500) | 512 | 160 | 1.18 | 30.9668 | 15.483 | 0.9677 | 1.14 | 0.1802 | 0.407 | 0.033127 | 12 a (0.750) | 6.6 a (0.415) | 2.7 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40; no trapping zero in any record |
| S4_16_s4_tau1 | tau 1 control of S4_16_s4 (to 16) | 0.2 | 16 | 4 | 1 (0.06250) | 512 | 160 | 1.18 | 30.9668 | 15.483 | 0.9677 | 1.14 | 0.1802 | 0.407 | 0.033127 | 12 a (0.750) | 6.6 a (0.415) | 2.7 a | 257 (0.000 to 16.0000; 0) | 16.0000 | 16.0000 | none | records to 16; no trapping zero in any record |
| RS32_s2 | sigma/a 2 control of RS32_s4 (Newton, engine v32) | 0.2 | 32 | 2 | 2 (0.06250) | 1024 | 320 | 2.36 | 59.7890 | 29.894 | 0.9342 | 2.20 | 0.1303 | 0.256 | 0.013147 | 16 a (0.500) | 8.4 a (0.261) | 3.4 a | 641 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| RS32_s4_tau1 | tau 1 control of RS32_s4 (engine v25) | 0.2 | 32 | 4 | 1 (0.03125) | 1024 | 320 | 2.36 | 59.7890 | 29.894 | 0.9342 | 2.20 | 0.1303 | 0.256 | 0.013147 | 16 a (0.500) | 8.4 a (0.261) | 3.4 a | 1281 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s2 | sigma/a 2 point at 64: the composite history (Newton across its seams) | 0.2 | 64 | 2 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s8 | sigma/a 8 point at 64 | 0.2 | 64 | 8 | 8 (0.12500) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s4_tau4 | tau 4 control of P64_s4 (engine v19) | 0.2 | 64 | 4 | 4 (0.06250) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 641 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| P64_s4_tau2 | tau 2 control of P64_s4 (engine v25, to 21.5) | 0.2 | 64 | 4 | 2 (0.03125) | 2048 | 640 | 4.71 | 118.1379 | 59.069 | 0.9230 | 4.35 | 0.1063 | 0.162 | 0.005217 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 689 (0.000 to 21.5000; 0) | 12.7500 | 12.7500 | 12.7812 (confirming residual above 1e-3) | records to 21.5; certified to 12.7500 |
| S4_128_s4_nk_v36m | Newton run of S4_128_s4 (v36m), stopped | 0.2 | 128 | 4 | 8 (0.06250) | 4096 | 1280 | 9.42 | 234.8342 | 117.417 | 0.9173 | 8.64 | 0.0985 | 0.102 | 0.002070 | 24 a (0.188) | 13.3 a (0.104) | 5.3 a | 48 (0.000 to 2.9375; 0) | 2.8750 | 2.8750 | 2.9375 (kept residual above 1e-3) | stopped at 2.9375 of its requested 40; certified to 2.8750; no trapping zero in any record |
| S4_128_s4_nk_v36q | Newton run of S4_128_s4 (v36q), stopped | 0.2 | 128 | 4 | 8 (0.06250) | 4096 | 1280 | 9.42 | 234.8342 | 117.417 | 0.9173 | 8.64 | 0.0985 | 0.102 | 0.002070 | 24 a (0.188) | 13.3 a (0.104) | 5.3 a | 46 (0.000 to 2.8125; 0) | 2.7500 | 2.7500 | 2.8125 (kept residual above 1e-3; confirming residual above 1e-3) | stopped at 2.8125 of its requested 40; certified to 2.7500; no trapping zero in any record |
| S4_128_s8 | width point (128, sigma/a 8), finished to 40 | 0.2 | 128 | 8 | 8 (0.06250) | 4096 | 1280 | 9.42 | 234.8342 | 117.417 | 0.9173 | 8.64 | 0.0985 | 0.102 | 0.002070 | 24 a (0.188) | 13.3 a (0.104) | 5.3 a | 641 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| S4_128_s16 | width point (128, sigma/a 16), at the gate's edge | 0.2 | 128 | 16 | 8 (0.06250) | 4096 | 1280 | 9.42 | 234.8342 | 117.417 | 0.9173 | 8.64 | 0.0985 | 0.102 | 0.002070 | 24 a (0.188) | 13.3 a (0.104) | 5.3 a | 641 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40 |
| S4_R2_m0p2 | preparation member, release at 2 r_s, (0.2, 64) | 0.2 | 64 | 4 | 8 (0.12500) | 2048 | 640 | 4.71 | 109.8332 | 54.917 | 0.8581 | 4.04 | 0.2539 | 0.323 | 0.020869 | 20 a (0.312) | 10.5 a (0.165) | 4.2 a | 321 (0.000 to 40.0000; 0) | 6.0000 | 40.0000 | 6.1250 (confirming residual above 1e-3) | records to 40; certified to 6.0000 |
| S4_R2_m0p1 | preparation member, release at 2 r_s, (0.1, 64) | 0.1 | 64 | 4 | 8 (0.12500) | 2048 | 1280 | 1.18 | 123.0561 | 61.528 | 0.9614 | 1.13 | 0.4546 | 0.814 | 0.132509 | 24 a (0.375) | 13.3 a (0.207) | 5.3 a | 321 (0.000 to 40.0000; 0) | 32.1250 | 32.1250 | 32.2500 (kept residual above 1e-3; confirming residual above 1e-3) | records to 40; certified to 32.1250 |
| M3a_s4 | mass member (0.1, 64), near-maximal | 0.1 | 64 | 4 | 8 (0.12500) | 2048 | 1280 | 1.18 | 123.4741 | 61.737 | 0.9646 | 1.14 | 0.1600 | 0.407 | 0.033127 | 24 a (0.375) | 13.3 a (0.207) | 5.3 a | 321 (0.000 to 40.0000; 0) | 40.0000 | 40.0000 | none | records to 40; no trapping zero in any record |

The step in units of r_s differs along the size ladder (tau/r_s 0.125 at 16 and 64, 0.0625 at 32 and 128), and sigma/r_s halves at each doubling at fixed sigma/a 4 (0.25, 0.125, 0.0625, 0.03125): the size direction at fixed sigma/a carries both, which the width cross does not separate (its model has no step term) and the step controls bound at r_s/a 64 only (and at 32 between tau 2 and tau 1); the (32, 4, 8) member is the step pair at a second size.

## 2 The first saved trapping record per member, with the bracket, the root fraction and the gate

| member | first saved record with a zero T_f/r_s | bracket (previous record T/r_s, min(N - |v|) there; width r_s) | zero of the stored min(N - |v|) series in the bracket: linear; quadratic (a diagnostic, not an event location) | R_t/a (first record) | R_t/r_s | R_t/(2 g E) (root fraction) | E/E0 | zeros (a) | no tangency; slope at R_t | root gate sigma/R_t | residual (kept); confirming | certification († banded) | first episode closes at | min(N - |v|) in the first episode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S4_16_s4 | none to 40.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| RS32_s4 | 15.8125 | 15.7500, 0.0046; 0.06250 | 15.7930; 15.7929 | 21.586 | 0.6746 | 0.7251 | 0.99582 | 18.786, 21.586 | yes; 2.94e-03 | 0.185 | 8.76e-08; - | certified (inside the certified prefix) | 19.3125 | -0.0899 (17.2500) |
| P64_s4 | 12.1250 | 12.0000, 0.0222; 0.12500 | 12.0553; 12.0571 | 38.556 | 0.6024 | 0.6613 | 0.98702 | 23.239, 38.556 | yes; 5.46e-03 | 0.104 | 6.41e-07; - | certified (inside the certified prefix) | 17.3750 | -0.4410 (13.0000) |
| S4_128_s4 | 11.0625 | 11.0000, 0.0330; 0.06250 | 11.0490; 11.0498 | 61.815 | 0.4829 | 0.5300 | 0.99326 | 42.683, 61.815 | yes; 1.46e-03 | 0.065 | 1.62e-08; - | uncertified: after the member's first record above 1e-3 (3.8750) | open to 19.0625 | -0.8021 (11.6875, unc.); over certified records - (-); 16 of 129 records do not count, 129 uncertified |
| S4_16_s2 | none to 40.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| S4_16_s4_tau1 | none to 16.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| RS32_s2 | 14.1250 | 14.0625, 0.0034; 0.06250 | 14.0780; 14.0782 | 20.095 | 0.6280 | 0.6731 | 0.99861 | 14.179, 20.095 | yes; 6.33e-03 | 0.100 | 1.27e-07; 3.05e-09 | certified (inside the certified prefix) | 18.9375 | -0.3617 (15.5625) |
| RS32_s4_tau1 | 15.7188 | 15.6875, 0.0001; 0.03125 | 15.6885; 15.6885 | 21.871 | 0.6835 | 0.7331 | 0.99790 | 18.335, 21.871 | yes; 3.71e-03 | 0.183 | 4.52e-08; - | certified (inside the certified prefix) | 19.2188 | -0.0939 (17.1250) |
| P64_s2 | 12.0000 | 11.8750, 0.0103; 0.12500 | 11.8965; 11.8979 | 38.098 | 0.5953 | 0.6545 | 0.98550 | 18.096, 38.098 | yes; 6.33e-03 | 0.052 | 3.10e-08; 3.32e-10 | certified (inside the certified prefix) | open to 40.0000 | -0.6618 (12.7500) |
| P64_s8 | 12.7500 | 12.6250, 0.0010; 0.12500 | 12.6291; 12.6291 | 44.527 | 0.6957 | 0.7620 | 0.98920 | 27.562, 44.527 | yes; 5.71e-03 | 0.180 | 2.59e-08; - | certified (inside the certified prefix) | 16.3750 | -0.1908 (13.7500) |
| P64_s4_tau4 | 12.0000 | 11.9375, 0.0001; 0.06250 | 11.9378; 11.9378 | 37.323 | 0.5832 | 0.6360 | 0.99341 | 22.029, 37.323 | yes; 5.33e-03 | 0.107 | 4.00e-07; - | certified (inside the certified prefix) | 16.3750 | -0.4651 (12.8750) |
| P64_s4_tau2 | 11.9062 | 11.8750, 0.0010; 0.03125 | 11.8772; 11.8772 | 34.669 | 0.5417 | 0.5888 | 0.99689 | 23.924, 34.669 | yes; 4.01e-03 | 0.115 | 2.20e-07; - | certified (inside the certified prefix) | 16.0938 (unc.) | -0.4787 (12.8125, unc.); over certified records -0.4739 (12.7500); 3 of 134 records do not count, 106 uncertified |
| S4_128_s4_nk_v36m | none to 2.9375 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| S4_128_s4_nk_v36q | none to 2.8125 | - | - | - | - | - | - | - | - | - | - | - | - | - |
| S4_128_s8 | 11.1875 | 11.1250, 0.0078; 0.06250 | 11.1369; 11.1375 | 72.936 | 0.5698 | 0.6253 | 0.99336 | 38.062, 72.936 | yes; 2.31e-03 | 0.110 | 4.64e-08; - | certified (inside the certified prefix) | 14.9375 | -0.5382 (11.8750) |
| S4_128_s16 | 11.5625 | 11.5000, 0.0075; 0.06250 | 11.5181; 11.5184 | 79.962 | 0.6247 | 0.6838 | 0.99591 | 53.071, 79.962 | yes; 2.08e-03 | 0.200 | 9.84e-08; - | certified (inside the certified prefix) | 15.1875 | -0.2735 (12.4375) |
| S4_R2_m0p2 | 5.0000 | 4.8750, 0.0258; 0.12500 | 4.9566; 4.9585 | 40.393 | 0.6311 | 0.7431 | 0.98987 | 27.723, 40.393 | yes; 3.47e-03 | 0.099 | 3.71e-07; - | certified (inside the certified prefix) | 10.3750 (unc.) | -0.5021 (6.1250, unc.); over certified records -0.4809 (6.0000); 2 of 43 records do not count, 34 uncertified |
| S4_R2_m0p1 | 30.6250 | 30.5000, 0.0052; 0.12500 | 30.5430; 30.5441 | 47.972 | 0.7496 | 0.7842 | 0.99419 | 33.941, 47.972 | yes; 2.66e-03 | 0.083 | 2.88e-07; - | certified (inside the certified prefix) | open to 40.0000 | -0.6015 (32.6250, unc.); over certified records -0.3778 (32.1250); 14 of 76 records do not count, 63 uncertified |
| M3a_s4 | none to 40.0000 | - | - | - | - | - | - | - | - | - | - | - | - | - |

Over the 0.5 r_s after the first saved record the root fraction grows by 0.100 to 0.230 of 2GE on the certified formed members (RS32_s4 0.7251 to 0.8557; P64_s4 0.6613 to 0.8140; RS32_s2 0.6731 to 0.8212; RS32_s4_tau1 0.7331 to 0.8562; P64_s2 0.6545 to 0.7950; P64_s8 0.7620 to 0.8623; P64_s4_tau4 0.6360 to 0.8076; P64_s4_tau2 0.5888 to 0.7989; S4_128_s8 0.6253 to 0.8554; S4_128_s16 0.6838 to 0.8788; S4_R2_m0p2 0.7431 to 0.9212; S4_R2_m0p1 0.7842 to 0.9017), so a first-record radius carries a sampling scatter of several hundredths of 2GE depending on where in the step the birth fell; no fit of R_t at formation and no intercept of T_f in any variable is a law before the crossing of N - |v| is located between records on every member (busi B7).

S4_128_s4's uncertified formation record (11.0625) sits amid a failing solve: 21 of its records lie above 1e-3, the residual reaching 2.66e-02 at 11.8125; it prints beside the width cross labelled so and enters no law.

The trapped episodes per member:

- RS32_s4: 1: 15.8125 to 19.2500 (56 records; R_t 21.586 to 23.634 a; min(N - |v|) -0.0899 at 17.2500; closes at 19.3125)
- P64_s4: 1: 12.1250 to 17.2500 (42 records; R_t 38.556 to 37.932 a; min(N - |v|) -0.4410 at 13.0000; closes at 17.3750); 2: 18.3750 to 40.0000 (174 records; R_t 34.350 to 36.310 a; min(N - |v|) -0.2211 at 20.7500; open at the last record)
- S4_128_s4: 1: 11.0625 to 19.0625 (129 records; R_t 61.815 to 59.366 a; min(N - |v|) -0.8021 at 11.6875 (unc.); open at the last record; 16 of 129 records do not count, 129 uncertified)
- RS32_s2: 1: 14.1250 to 18.8750 (77 records; R_t 20.095 to 22.250 a; min(N - |v|) -0.3617 at 15.5625; closes at 18.9375); 2: 24.1250 to 30.1875 (98 records; R_t 18.235 to 19.407 a; min(N - |v|) -0.2235 at 26.5625; closes at 30.2500); 3: 34.3125 to 40.0000 (92 records; R_t 17.095 to 21.533 a; min(N - |v|) -0.1978 at 37.0625; open at the last record)
- RS32_s4_tau1: 1: 15.7188 to 19.1875 (112 records; R_t 21.871 to 22.859 a; min(N - |v|) -0.0939 at 17.1250; closes at 19.2188)
- P64_s2: 1: 12.0000 to 40.0000 (225 records; R_t 38.098 to 31.423 a; min(N - |v|) -0.6618 at 12.7500; open at the last record)
- P64_s8: 1: 12.7500 to 16.2500 (29 records; R_t 44.527 to 50.517 a; min(N - |v|) -0.1908 at 13.7500; closes at 16.3750); 2: 26.6250 to 29.6250 (25 records; R_t 35.899 to 39.774 a; min(N - |v|) -0.0561 at 28.0000; closes at 29.7500); 3: 38.8750 to 40.0000 (10 records; R_t 37.951 to 40.963 a; min(N - |v|) -0.0269 at 39.7500; open at the last record)
- P64_s4_tau4: 1: 12.0000 to 16.3125 (70 records; R_t 37.323 to 42.893 a; min(N - |v|) -0.4651 at 12.8750; closes at 16.3750); 2: 20.1250 to 25.5000 (87 records; R_t 30.642 to 32.572 a; min(N - |v|) -0.2408 at 21.7500; closes at 25.5625); 3: 26.9375 to 40.0000 (210 records; R_t 30.546 to 34.464 a; min(N - |v|) -0.1952 at 29.2500; open at the last record)
- P64_s4_tau2: 1: 11.9062 to 16.0625 (134 records; R_t 34.669 to 43.373 a; min(N - |v|) -0.4787 at 12.8125 (unc.); closes at 16.0938 (unc.); 3 of 134 records do not count, 106 uncertified); 2: 21.0312 to 21.5000 (16 records; R_t 30.009 to 36.375 a; min(N - |v|) -0.0799 at 21.5000 (unc.); open at the last record; 16 uncertified)
- S4_128_s8: 1: 11.1875 to 14.8750 (60 records; R_t 72.936 to 85.441 a; min(N - |v|) -0.5382 at 11.8750; closes at 14.9375); 2: 19.1875 to 23.7500 (74 records; R_t 56.079 to 59.944 a; min(N - |v|) -0.2708 at 20.3125; closes at 23.8125); 3: 24.8750 to 40.0000 (243 records; R_t 52.285 to 60.667 a; min(N - |v|) -0.1918 at 26.7500; open at the last record)
- S4_128_s16: 1: 11.5625 to 15.1250 (58 records; R_t 79.962 to 98.699 a; min(N - |v|) -0.2735 at 12.4375; closes at 15.1875); 2: 25.0625 to 27.8125 (45 records; R_t 62.334 to 71.656 a; min(N - |v|) -0.0684 at 26.1875; closes at 27.8750); 3: 35.5000 to 36.7500 (21 records; R_t 62.076 to 63.999 a; min(N - |v|) -0.0105 at 36.1250; closes at 36.8125)
- S4_R2_m0p2: 1: 5.0000 to 10.2500 (43 records; R_t 40.393 to 39.453 a; min(N - |v|) -0.5021 at 6.1250 (unc.); closes at 10.3750 (unc.); 2 of 43 records do not count, 34 uncertified); 2: 11.2500 to 40.0000 (231 records; R_t 35.824 to 38.795 a; min(N - |v|) -0.2824 at 13.5000 (unc.); open at the last record; 231 uncertified)
- S4_R2_m0p1: 1: 30.6250 to 40.0000 (76 records; R_t 47.972 to 46.766 a; min(N - |v|) -0.6015 at 32.6250 (unc.); open at the last record; 14 of 76 records do not count, 63 uncertified)

## 3 The dust-energy centroid R_c against the onsets (not the core's compression radius; not composed here)

| member | min R_c (dust-energy centroid) over the certified records (a) | at T/r_s | min R_c/r_s | own onset, chain columns (upper bound) | isotropic 1.222 N_B^(1/3) a | exclusion floor | minimum at the last certified record |
|---|---|---|---|---|---|---|---|
| S4_16_s4 | 39.31 | 0.0000 | 2.457 | 12 a | 6.6 a | 2.7 a | no |
| RS32_s4 | 16.19 | 17.4375 | 0.506 | 16 a | 8.4 a | 3.4 a | no |
| P64_s4 | 14.52 | 13.2500 | 0.227 | 20 a | 10.5 a | 4.2 a | no |
| S4_128_s4 | 310.81 | 3.5625 | 2.428 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |
| S4_16_s2 | 38.25 | 7.7500 | 2.391 | 12 a | 6.6 a | 2.7 a | no |
| S4_16_s4_tau1 | 39.31 | 0.0000 | 2.457 | 12 a | 6.6 a | 2.7 a | no |
| RS32_s2 | 10.09 | 15.6875 | 0.315 | 16 a | 8.4 a | 3.4 a | no |
| RS32_s4_tau1 | 16.11 | 17.3125 | 0.504 | 16 a | 8.4 a | 3.4 a | no |
| P64_s2 | 10.85 | 12.8750 | 0.169 | 20 a | 10.5 a | 4.2 a | no |
| P64_s8 | 22.40 | 14.0000 | 0.350 | 20 a | 10.5 a | 4.2 a | no |
| P64_s4_tau4 | 14.83 | 13.0000 | 0.232 | 20 a | 10.5 a | 4.2 a | no |
| P64_s4_tau2 | 15.92 | 12.7500 | 0.249 | 20 a | 10.5 a | 4.2 a | yes (still falling at the end of the certified records) |
| S4_128_s4_nk_v36m | 317.33 | 2.8750 | 2.479 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |
| S4_128_s4_nk_v36q | 318.36 | 2.7500 | 2.487 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |
| S4_128_s8 | 23.95 | 11.9375 | 0.187 | 24 a | 13.3 a | 5.3 a | no |
| S4_128_s16 | 35.10 | 12.6250 | 0.274 | 24 a | 13.3 a | 5.3 a | no |
| S4_R2_m0p2 | 11.33 | 6.0000 | 0.177 | 20 a | 10.5 a | 4.2 a | yes (still falling at the end of the certified records) |
| S4_R2_m0p1 | 16.94 | 32.1250 | 0.265 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |
| M3a_s4 | 159.59 | 40.0000 | 2.494 | 24 a | 13.3 a | 5.3 a | yes (still falling at the end of the certified records) |

## 4 The matched-clock readouts (every 2 r_s; (n.c.) the record does not count, (unc.) it lies after the certified prefix)

### 4a min(N - |v|)

| T/r_s | S4_16_s4 | RS32_s4 | P64_s4 | S4_128_s4 | S4_128_s8 | S4_128_s16 | P64_s2 | P64_s8 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.4799 | 0.4875 | 0.4945 | 0.4980 | 0.4983 | 0.4994 | 0.4943 | 0.4956 |
| 2 | 0.4791 | 0.4850 | 0.4910 | 0.4943 | 0.4948 | 0.4966 | 0.4905 | 0.4929 |
| 4 | 0.4829 | 0.4814 | 0.4830 | 0.4827 (n.c.) | 0.4834 | 0.4861 | 0.4821 | 0.4855 |
| 6 | 0.4866 | 0.4717 | 0.4661 | 0.4596 (unc.) | 0.4608 | 0.4650 | 0.4647 | 0.4698 |
| 8 | 0.4893 | 0.4501 | 0.4322 | 0.4133 (unc.) | 0.4155 | 0.4232 | 0.4300 | 0.4385 |
| 10 | 0.4917 | 0.4038 | 0.3552 | 0.2915 (unc.) | 0.2982 | 0.3218 | 0.3499 | 0.3700 |
| 12 | 0.4966 | 0.3198 | 0.0222 | -0.4458 (n.c.) | -0.5040 | -0.1972 | -0.0496 | 0.1449 |
| 14 | 0.5036 | 0.1860 | -0.2161 | -0.0746 (unc.) | -0.0924 | -0.0876 | -0.1777 | -0.1869 |
| 16 | 0.5114 | -0.0216 | -0.0424 | -0.0446 (unc.) | 0.0555 | 0.0470 | -0.1138 | -0.0233 |
| 18 | 0.5192 | -0.0709 | 0.0047 | -0.2047 (n.c.) | 0.0904 | 0.1530 | -0.2794 | 0.0870 |
| 20 | 0.5266 | 0.0373 | -0.1463 | - | -0.2280 | 0.2355 | -0.1599 | 0.1609 |
| 22 | 0.5335 | 0.1103 | -0.1351 | - | -0.0936 | 0.2486 | -0.2301 | 0.2042 |
| 24 | 0.5399 | 0.1738 | -0.0550 | - | 0.0053 | 0.1335 | -0.2097 | 0.1755 |
| 26 | 0.5457 | 0.2210 | -0.0747 | - | -0.1173 | -0.0673 | -0.1901 | 0.0511 |
| 28 | 0.5509 | 0.2433 | -0.1758 | - | -0.1097 | 0.0095 | -0.2181 | -0.0561 |
| 30 | 0.5556 | 0.2479 | -0.0910 | - | -0.0436 | 0.0988 | -0.1567 | 0.0154 |
| 32 | 0.5599 | 0.2293 | -0.0620 | - | -0.1422 | 0.1398 | -0.2009 | 0.1017 |
| 34 | 0.5637 | 0.1872 | -0.1712 | - | -0.0881 | 0.0813 | -0.1824 | 0.1456 |
| 36 | 0.5671 | 0.1055 | -0.1173 | - | -0.0585 | -0.0102 | -0.1531 | 0.1336 |
| 38 | 0.5700 | 0.0240 | -0.0540 | - | -0.1350 | 0.0373 | -0.2225 | 0.0459 |
| 40 | 0.5726 | 0.0097 | -0.1603 | - | -0.0883 | 0.1000 | -0.1762 | -0.0258 |

### 4b E/E0

| T/r_s | S4_16_s4 | RS32_s4 | P64_s4 | S4_128_s4 | S4_128_s8 | S4_128_s16 | P64_s2 | P64_s8 |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.00000 | 1.00000 | 1.00000 | 1.00000 | 1.00016 | 1.00078 | 0.99984 | 1.00064 |
| 2 | 1.00002 | 1.00001 | 1.00003 | 1.00001 | 1.00017 | 1.00079 | 1.00011 | 1.00066 |
| 4 | 1.00001 | 1.00003 | 1.00007 | 1.00005 (n.c.) | 1.00020 | 1.00082 | 1.00066 | 1.00071 |
| 6 | 0.99997 | 1.00001 | 1.00010 | 1.00009 (unc.) | 1.00024 | 1.00085 | 1.00082 | 1.00074 |
| 8 | 0.99996 | 0.99992 | 1.00003 | 1.00011 (unc.) | 1.00025 | 1.00087 | 1.00075 | 1.00068 |
| 10 | 0.99996 | 0.99976 | 0.99935 | 0.99976 (unc.) | 0.99995 | 1.00068 | 1.00004 | 1.00017 |
| 12 | 0.99996 | 0.99947 | 0.98981 | 0.88991 (n.c.) | 0.95010 | 0.98932 | 0.98550 | 0.99615 |
| 14 | 0.99995 | 0.99858 | 0.92502 | 0.87055 (unc.) | 0.93881 | 0.98313 | 0.85116 | 0.97490 |
| 16 | 0.99993 | 0.99543 | 0.91882 | 0.86876 (unc.) | 0.93599 | 0.98139 | 0.84743 | 0.97462 |
| 18 | 0.99992 | 0.99406 | 0.91692 | 0.85739 (n.c.) | 0.93522 | 0.97885 | 0.83547 | 0.97153 |
| 20 | 0.99991 | 0.99371 | 0.91186 | - | 0.92660 | 0.97772 | 0.83431 | 0.96900 |
| 22 | 0.99990 | 0.99302 | 0.90844 | - | 0.92375 | 0.97748 | 0.83318 | 0.96837 |
| 24 | 0.99989 | 0.99241 | 0.90646 | - | 0.92201 | 0.97703 | 0.83208 | 0.96803 |
| 26 | 0.99988 | 0.99213 | 0.90580 | - | 0.92068 | 0.97391 | 0.83219 | 0.96630 |
| 28 | 0.99988 | 0.99205 | 0.90366 | - | 0.91879 | 0.97391 | 0.83139 | 0.96306 |
| 30 | 0.99987 | 0.99204 | 0.90297 | - | 0.91809 | 0.97298 | 0.83052 | 0.96299 |
| 32 | 0.99987 | 0.99203 | 0.90242 | - | 0.91701 | 0.97259 | 0.83011 | 0.96149 |
| 34 | 0.99987 | 0.99192 | 0.90112 | - | 0.91671 | 0.97240 | 0.83002 | 0.96083 |
| 36 | 0.99987 | 0.99145 | 0.90061 | - | 0.91641 | 0.97156 | 0.83004 | 0.96082 |
| 38 | 0.99986 | 0.99070 | 0.89982 | - | 0.91596 | 0.97160 | 0.82996 | 0.96012 |
| 40 | 0.99986 | 0.99060 | 0.89900 | - | 0.91589 | 0.97115 | 0.83012 | 0.95873 |

### 4c R_c/r_s (the dust-energy centroid)

| T/r_s | S4_16_s4 | RS32_s4 | P64_s4 | S4_128_s4 | S4_128_s8 | S4_128_s16 | P64_s2 | P64_s8 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.4567 | 2.4998 | 2.5875 | 2.5742 | 2.5742 | 2.5742 | 2.5875 | 2.5875 |
| 2 | 2.4677 | 2.4742 | 2.5478 | 2.5275 | 2.5283 | 2.5312 | 2.5469 | 2.5507 |
| 4 | 2.4664 | 2.3841 | 2.4271 | 2.3895 (n.c.) | 2.3915 | 2.3994 | 2.4243 | 2.4350 |
| 6 | 2.4697 | 2.2244 | 2.2133 | 2.1429 (unc.) | 2.1470 | 2.1633 | 2.2070 | 2.2291 |
| 8 | 2.4993 | 1.9972 | 1.8851 | 1.7547 (unc.) | 1.7627 | 1.7939 | 1.8737 | 1.9143 |
| 10 | 2.5549 | 1.7158 | 1.4031 | 1.1424 (unc.) | 1.1599 | 1.2266 | 1.3817 | 1.4595 |
| 12 | 2.6279 | 1.3856 | 0.6776 | 0.2024 (n.c.) | 0.1872 | 0.3665 | 0.6088 | 0.8323 |
| 14 | 2.7105 | 1.0052 | 0.3207 | 0.4335 (unc.) | 0.4603 | 0.4274 | 0.3281 | 0.3501 |
| 16 | 2.7975 | 0.6208 | 0.5039 | 0.5704 (unc.) | 0.6820 | 0.6662 | 0.4139 | 0.5317 |
| 18 | 2.8856 | 0.5201 | 0.6217 | 0.4020 (n.c.) | 0.8626 | 0.9076 | 0.3149 | 0.7292 |
| 20 | 2.9726 | 0.6641 | 0.5405 | - | 0.6396 | 1.1433 | 0.4651 | 0.9189 |
| 22 | 3.0574 | 0.8244 | 0.5197 | - | 0.6843 | 1.2752 | 0.4703 | 1.0584 |
| 24 | 3.1389 | 0.9828 | 0.7295 | - | 0.9937 | 1.2194 | 0.4976 | 1.0806 |
| 26 | 3.2169 | 1.1217 | 0.7969 | - | 1.0411 | 0.9035 | 0.5845 | 0.9522 |
| 28 | 3.2909 | 1.2249 | 0.6774 | - | 0.9464 | 0.9922 | 0.5624 | 0.7838 |
| 30 | 3.3607 | 1.2874 | 0.8516 | - | 1.2977 | 1.2801 | 0.7010 | 0.9226 |
| 32 | 3.4264 | 1.2931 | 1.0181 | - | 1.3047 | 1.5431 | 0.6910 | 1.1773 |
| 34 | 3.4877 | 1.2371 | 0.9010 | - | 1.3501 | 1.6248 | 0.7759 | 1.3888 |
| 36 | 3.5447 | 1.1151 | 0.9933 | - | 1.6972 | 1.4751 | 0.8628 | 1.4785 |
| 38 | 3.5970 | 0.9781 | 1.2350 | - | 1.6073 | 1.5844 | 0.8381 | 1.3968 |
| 40 | 3.6447 | 0.9678 | 1.1415 | - | 1.7632 | 1.9058 | 0.9378 | 1.2434 |

### 4d R_t/r_s ("none": no zero in the record)

| T/r_s | S4_16_s4 | RS32_s4 | P64_s4 | S4_128_s4 | S4_128_s8 | S4_128_s16 | P64_s2 | P64_s8 |
|---|---|---|---|---|---|---|---|---|
| 0 | none | none | none | none | none | none | none | none |
| 2 | none | none | none | none | none | none | none | none |
| 4 | none | none | none | none (n.c.) | none | none | none | none |
| 6 | none | none | none | none (unc.) | none | none | none | none |
| 8 | none | none | none | none (unc.) | none | none | none | none |
| 10 | none | none | none | none (unc.) | none | none | none | none |
| 12 | none | none | none | 0.7289 (n.c.) | 0.7855 | 0.7877 | 0.5953 | none |
| 14 | none | none | 0.7902 | 0.6812 (unc.) | 0.8445 | 0.8771 | 0.7091 | 0.8267 |
| 16 | none | 0.7515 | 0.7943 | 0.4745 (unc.) | none | none | 0.6177 | 0.8268 |
| 18 | none | 0.8420 | none | 0.4527 (n.c.) | none | none | 0.5113 | none |
| 20 | none | none | 0.6005 | - | 0.5418 | none | 0.5082 | none |
| 22 | none | none | 0.6119 | - | 0.5739 | none | 0.4967 | none |
| 24 | none | none | 0.6081 | - | none | none | 0.4940 | none |
| 26 | none | none | 0.5799 | - | 0.4868 | 0.6181 | 0.4989 | none |
| 28 | none | none | 0.5868 | - | 0.5070 | none | 0.4923 | 0.6900 |
| 30 | none | none | 0.5861 | - | 0.5086 | none | 0.4949 | none |
| 32 | none | none | 0.5730 | - | 0.4682 | none | 0.4877 | none |
| 34 | none | none | 0.5801 | - | 0.4760 | none | 0.4765 | none |
| 36 | none | none | 0.5813 | - | 0.4743 | 0.5262 | 0.4686 | none |
| 38 | none | none | 0.5595 | - | 0.4687 | none | 0.4767 | none |
| 40 | none | none | 0.5673 | - | 0.4740 | none | 0.4910 | 0.6400 |

### 4e At the first saved trapping record and after it

| member | T_f + 0 | T_f + 0.5 | T_f + 1 | T_f + 2 | T_f + 4 |
|---|---|---|---|---|---|
| RS32_s4 | R_t/r_s 0.6746, R_t/(2gE) 0.7251, E/E0 0.99582, min(N-|v|) -0.0021, gate 0.185 | R_t/r_s 0.7952, R_t/(2gE) 0.8557, E/E0 0.99483, min(N-|v|) -0.0505, gate 0.157 | R_t/r_s 0.8231, R_t/(2gE) 0.8862, E/E0 0.99418, min(N-|v|) -0.0816, gate 0.152 | R_t/r_s 0.8417, R_t/(2gE) 0.9064, E/E0 0.99404, min(N-|v|) -0.0784, gate 0.149 | R_t/r_s none, R_t/(2gE) -, E/E0 0.99377, min(N-|v|) 0.0279, gate - |
| P64_s4 | R_t/r_s 0.6024, R_t/(2gE) 0.6613, E/E0 0.98702, min(N-|v|) -0.0279, gate 0.104 | R_t/r_s 0.7230, R_t/(2gE) 0.8140, E/E0 0.96234, min(N-|v|) -0.2880, gate 0.086 | R_t/r_s 0.7520, R_t/(2gE) 0.8731, E/E0 0.93323, min(N-|v|) -0.4367, gate 0.083 | R_t/r_s 0.7936, R_t/(2gE) 0.9302, E/E0 0.92439, min(N-|v|) -0.1960, gate 0.079 | R_t/r_s 0.7717, R_t/(2gE) 0.9102, E/E0 0.91861, min(N-|v|) -0.0381, gate 0.081 |
| S4_128_s4 | R_t/r_s 0.4829, R_t/(2gE) 0.5300, E/E0 0.99326, min(N-|v|) -0.0091, gate 0.065 (unc.) | R_t/r_s 0.7019, R_t/(2gE) 0.8254, E/E0 0.92707, min(N-|v|) -0.7237, gate 0.045 (n.c.) | R_t/r_s 0.7314, R_t/(2gE) 0.8993, E/E0 0.88654, min(N-|v|) -0.4076, gate 0.043 (n.c.) | R_t/r_s 0.7651, R_t/(2gE) 0.9564, E/E0 0.87208, min(N-|v|) -0.1415, gate 0.041 (unc.) | R_t/r_s 0.5361, R_t/(2gE) 0.6720, E/E0 0.86957, min(N-|v|) -0.0389, gate 0.058 (unc.) |
| S4_128_s8 | R_t/r_s 0.5698, R_t/(2gE) 0.6253, E/E0 0.99336, min(N-|v|) -0.0333, gate 0.110 | R_t/r_s 0.7561, R_t/(2gE) 0.8554, E/E0 0.96355, min(N-|v|) -0.4827, gate 0.083 | R_t/r_s 0.7993, R_t/(2gE) 0.9199, E/E0 0.94728, min(N-|v|) -0.4145, gate 0.078 | R_t/r_s 0.8348, R_t/(2gE) 0.9670, E/E0 0.94118, min(N-|v|) -0.1790, gate 0.075 | R_t/r_s none, R_t/(2gE) -, E/E0 0.93687, min(N-|v|) 0.0172, gate - |
| S4_128_s16 | R_t/r_s 0.6247, R_t/(2gE) 0.6838, E/E0 0.99591, min(N-|v|) -0.0184, gate 0.200 | R_t/r_s 0.7966, R_t/(2gE) 0.8788, E/E0 0.98821, min(N-|v|) -0.2170, gate 0.157 | R_t/r_s 0.8403, R_t/(2gE) 0.9321, E/E0 0.98274, min(N-|v|) -0.2707, gate 0.149 | R_t/r_s 0.8748, R_t/(2gE) 0.9700, E/E0 0.98312, min(N-|v|) -0.1380, gate 0.143 | R_t/r_s none, R_t/(2gE) -, E/E0 0.98191, min(N-|v|) 0.0246, gate - |
| P64_s2 | R_t/r_s 0.5953, R_t/(2gE) 0.6545, E/E0 0.98550, min(N-|v|) -0.0496, gate 0.052 | R_t/r_s 0.6883, R_t/(2gE) 0.7950, E/E0 0.93803, min(N-|v|) -0.4197, gate 0.045 | R_t/r_s 0.6924, R_t/(2gE) 0.8573, E/E0 0.87513, min(N-|v|) -0.5464, gate 0.045 | R_t/r_s 0.7091, R_t/(2gE) 0.9026, E/E0 0.85116, min(N-|v|) -0.1777, gate 0.044 | R_t/r_s 0.6177, R_t/(2gE) 0.7898, E/E0 0.84743, min(N-|v|) -0.1138, gate 0.051 |
| P64_s8 | R_t/r_s 0.6957, R_t/(2gE) 0.7620, E/E0 0.98920, min(N-|v|) -0.0307, gate 0.180 | R_t/r_s 0.7807, R_t/(2gE) 0.8623, E/E0 0.98096, min(N-|v|) -0.1453, gate 0.160 | R_t/r_s 0.8149, R_t/(2gE) 0.9051, E/E0 0.97544, min(N-|v|) -0.1908, gate 0.153 | R_t/r_s 0.8494, R_t/(2gE) 0.9436, E/E0 0.97533, min(N-|v|) -0.1288, gate 0.147 | R_t/r_s none, R_t/(2gE) -, E/E0 0.97357, min(N-|v|) 0.0237, gate - |
| RS32_s2 | R_t/r_s 0.6280, R_t/(2gE) 0.6731, E/E0 0.99861, min(N-|v|) -0.0104, gate 0.100 | R_t/r_s 0.7631, R_t/(2gE) 0.8212, E/E0 0.99464, min(N-|v|) -0.1420, gate 0.082 | R_t/r_s 0.7936, R_t/(2gE) 0.8591, E/E0 0.98882, min(N-|v|) -0.2966, gate 0.079 | R_t/r_s 0.8210, R_t/(2gE) 0.8915, E/E0 0.98571, min(N-|v|) -0.2597, gate 0.076 | R_t/r_s 0.8093, R_t/(2gE) 0.8809, E/E0 0.98343, min(N-|v|) -0.0424, gate 0.077 |
| RS32_s4_tau1 | R_t/r_s 0.6835, R_t/(2gE) 0.7331, E/E0 0.99790, min(N-|v|) -0.0034, gate 0.183 | R_t/r_s 0.7978, R_t/(2gE) 0.8562, E/E0 0.99738, min(N-|v|) -0.0536, gate 0.157 | R_t/r_s 0.8255, R_t/(2gE) 0.8863, E/E0 0.99704, min(N-|v|) -0.0858, gate 0.151 | R_t/r_s 0.8439, R_t/(2gE) 0.9061, E/E0 0.99697, min(N-|v|) -0.0808, gate 0.148 | R_t/r_s none, R_t/(2gE) -, E/E0 0.99682, min(N-|v|) 0.0290, gate - |
| P64_s4_tau4 | R_t/r_s 0.5832, R_t/(2gE) 0.6360, E/E0 0.99341, min(N-|v|) -0.0276, gate 0.107 | R_t/r_s 0.7302, R_t/(2gE) 0.8076, E/E0 0.97972, min(N-|v|) -0.3178, gate 0.086 | R_t/r_s 0.7792, R_t/(2gE) 0.8739, E/E0 0.96600, min(N-|v|) -0.4465, gate 0.080 | R_t/r_s 0.8268, R_t/(2gE) 0.9328, E/E0 0.96037, min(N-|v|) -0.1917, gate 0.076 | R_t/r_s 0.7428, R_t/(2gE) 0.8411, E/E0 0.95688, min(N-|v|) -0.0172, gate 0.084 |
| P64_s4_tau2 | R_t/r_s 0.5417, R_t/(2gE) 0.5888, E/E0 0.99689, min(N-|v|) -0.0131, gate 0.115 | R_t/r_s 0.7301, R_t/(2gE) 0.7989, E/E0 0.99014, min(N-|v|) -0.3136, gate 0.086 | R_t/r_s 0.7907, R_t/(2gE) 0.8717, E/E0 0.98277, min(N-|v|) -0.4530, gate 0.079 (unc.) | R_t/r_s 0.8438, R_t/(2gE) 0.9332, E/E0 0.97970, min(N-|v|) -0.1939, gate 0.074 (unc.) | R_t/r_s 0.7250, R_t/(2gE) 0.8034, E/E0 0.97778, min(N-|v|) -0.0089, gate 0.086 (unc.) |
| S4_R2_m0p2 | R_t/r_s 0.6311, R_t/(2gE) 0.7431, E/E0 0.98987, min(N-|v|) -0.0137, gate 0.099 | R_t/r_s 0.7680, R_t/(2gE) 0.9212, E/E0 0.97158, min(N-|v|) -0.2361, gate 0.081 | R_t/r_s 0.7661, R_t/(2gE) 0.9574, E/E0 0.93255, min(N-|v|) -0.4809, gate 0.082 | R_t/r_s 0.7784, R_t/(2gE) 0.9827, E/E0 0.92321, min(N-|v|) -0.2152, gate 0.080 (unc.) | R_t/r_s 0.7723, R_t/(2gE) 0.9818, E/E0 0.91675, min(N-|v|) -0.0442, gate 0.081 (unc.) |
| S4_R2_m0p1 | R_t/r_s 0.7496, R_t/(2gE) 0.7842, E/E0 0.99419, min(N-|v|) -0.0100, gate 0.083 | R_t/r_s 0.8587, R_t/(2gE) 0.9017, E/E0 0.99060, min(N-|v|) -0.0863, gate 0.073 | R_t/r_s 0.8857, R_t/(2gE) 0.9375, E/E0 0.98266, min(N-|v|) -0.2009, gate 0.071 | R_t/r_s 0.9179, R_t/(2gE) 0.9931, E/E0 0.96144, min(N-|v|) -0.6015, gate 0.068 (n.c.) | R_t/r_s 0.8906, R_t/(2gE) 0.9748, E/E0 0.95031, min(N-|v|) -0.1783, gate 0.070 (unc.) |

## 5 The dependences and 2-coefficient diagnostics (no exponent, intercept or law is printed)

### 5.1 The size ladder at sigma/a 4: the dependence

The fixed-width size ladder supplies 2 certified first-record formation points (RS32_s4, P64_s4), insufficient to test even a fixed-exponent intercept-and-amplitude model; the free exponent on 4 members is not evaluable.

| member | r_s/a | tau/a (tau/r_s) | sigma/r_s | first saved zero T_f/r_s (class) | record step | R_t/a | R_t/r_s | root fraction | gate | last converged record |
|---|---|---|---|---|---|---|---|---|---|---|
| S4_16_s4 | 16 | 2 (0.12500) | 0.25000 | none to 40.0000 | - | - | - | - | - | 40.0000 |
| RS32_s4 | 32 | 2 (0.06250) | 0.12500 | 15.8125 (certified) | 0.0625 | 21.586 | 0.6746 | 0.7251 | 0.185 | 40.0000 |
| P64_s4 | 64 | 8 (0.12500) | 0.06250 | 12.1250 (certified) | 0.1250 | 38.556 | 0.6024 | 0.6613 | 0.104 | 40.0000 |
| S4_128_s4 | 128 | 8 (0.06250) | 0.03125 | 11.0625 (uncertified) | 0.0625 | 61.815 | 0.4829 | 0.5300 | 0.065 | 3.5625 |

### 5.2 A 2-coefficient fit in (a/r_s)^(1/3) (the composer's choice of variable; the paper's chi^(1/3) is the exponent of the shed energy share f_E, not of the formation clock, whose asymptotic form at fixed m a is the comparator's clock with a pressure correction of order chi^(4/3), nor of the root fraction, whose asymptotic form is an approach to 1): Y = c0 + c1 (a/r_s)^(1/3)

- model: 2-coefficient diagnostic Y = c0 + c1 (a/r_s)^(1/3) on the certified size members
- members: RS32_s4, P64_s4 (2 members, 2 coefficients, residual degrees of freedom 0)
- x per member: RS32_s4 0.31498, P64_s4 0.25000

| Y | coefficients | member: Y, fit, residual | rms residual over members; per residual degree of freedom | max |residual| |
|---|---|---|---|---|
| T_f | not printed (dof 0: a dependence) | RS32_s4: 15.8125, 15.8125, -0.0000; P64_s4: 12.1250, 12.1250, -0.0000 | 0 (dof 0) | 0 (dof 0) |
| R_t_over_rs | not printed (dof 0: a dependence) | RS32_s4: 0.6746, 0.6746, -0.0000; P64_s4: 0.6024, 0.6024, -0.0000 | 0 (dof 0) | 0 (dof 0) |
| R_t_over_2gE | not printed (dof 0: a dependence) | RS32_s4: 0.7251, 0.7251, -0.0000; P64_s4: 0.6613, 0.6613, -0.0000 | 0 (dof 0) | 0 (dof 0) |

With the uncertified formation record of the (0.2, 128) sigma/a 4 member inside (labelled; not a law):

- model: 2-coefficient diagnostic Y = c0 + c1 (a/r_s)^(1/3) including uncertified formation records (labelled; no law)
- members: RS32_s4, P64_s4, S4_128_s4 (3 members, 2 coefficients, residual degrees of freedom 1)
- x per member: RS32_s4 0.31498, P64_s4 0.25000, S4_128_s4 0.19843
- formation records inside this fit that are not certified (labelled; no law): S4_128_s4 (uncertified)

| Y | coefficients | member: Y, fit, residual | rms residual over members; per residual degree of freedom | max |residual| |
|---|---|---|---|---|
| T_f | c0 2.45636, c1 41.434 (c0 from first-record clocks; cadence error up to the record step per member; not an intercept) | RS32_s4: 15.8125, 15.5072, +0.3053; P64_s4: 12.1250, 12.8149, -0.6899; S4_128_s4: 11.0625, 10.6779, +0.3846 | 4.89e-01; 8.47e-01 | 6.90e-01 |
| R_t_over_rs | c0 0.174034, c1 1.62148 (first-record radii, at the sampling scatter of the first record; not a radius law) | RS32_s4: 0.6746, 0.6848, -0.0102; P64_s4: 0.6024, 0.5794, +0.0230; S4_128_s4: 0.4829, 0.4958, -0.0128 | 1.63e-02; 2.83e-02 | 2.30e-02 |
| R_t_over_2gE | c0 0.220396, c1 1.64431 (first-record radii, at the sampling scatter of the first record; not a radius law) | RS32_s4: 0.7251, 0.7383, -0.0132; P64_s4: 0.6613, 0.6315, +0.0298; S4_128_s4: 0.5300, 0.5467, -0.0166 | 2.11e-02; 3.66e-02 | 2.98e-02 |

- S4_16_s4: the fit's value T_f 18.8994, R_t/r_s 0.8175, R_t/(2gE) 0.8729; the member holds no trapping zero in its records to 40.0000

- members: RS32_s4, P64_s4, S4_128_s4; the exact form Y = Y_0 + A (a/r_s)^p through the 3 points (no residual; not a law):
  - T_f: p 1.795, Y_0 10.6324, A 2608
  - R_t_over_rs: p -0.728, Y_0 0.7844, A -0.008798
  - R_t_over_2gE: p -1.041, Y_0 0.7854, A -0.001634

### 5.3 The width cross: Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a) (a pair of straight lines in ln(sigma/a); c0 is not a continuum-size intercept)

The 6 points: r_s/a 64 at sigma/a 2, 4, 8 and r_s/a 128 at sigma/a 4, 8, 16. Certified: (64, 2), (64, 4), (64, 8), (128, 8), (128, 16); not certified: (128, 4) (formation 11.0625, certified extent 3.5625, first kept residual above 1e-3 at 3.875). The z term carries one halving of tau/r_s between the sizes (0.125 at 64, 0.0625 at 128), whose share by the 64 step controls is -0.125 r_s and -0.019 in R_t/r_s per halving, measured at 1 size only. The (128, 8) point is read from its final run json (finished to T/r_s 40).

The dependence per doubling of sigma/a (the records' numbers):

| r_s/a | from | to | dT_f (r_s) | d(R_t/r_s) | d(root fraction) | both certified |
|---|---|---|---|---|---|---|
| 64 | P64_s2 | P64_s4 | +0.1250 | +0.0072 | +0.0069 | yes |
| 64 | P64_s4 | P64_s8 | +0.6250 | +0.0933 | +0.1007 | yes |
| 128 | S4_128_s4 | S4_128_s8 | +0.1250 | +0.0869 | +0.0953 | no |
| 128 | S4_128_s8 | S4_128_s16 | +0.3750 | +0.0549 | +0.0585 | yes |

On the 5 certified points (4 coefficients, 1 residual degree of freedom):

- model: Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a), on the certified points
- members: P64_s2, P64_s4, P64_s8, S4_128_s8, S4_128_s16 (5 members, 4 coefficients, residual degrees of freedom 1)
- (z, h) per member: P64_s2 (-4.1589, 0.6931), P64_s4 (-4.1589, 1.3863), P64_s8 (-4.1589, 2.0794), S4_128_s8 (-4.8520, 2.0794), S4_128_s16 (-4.8520, 2.7726)

| Y | coefficients | member: Y, fit, residual | rms residual over members; per residual degree of freedom | max |residual| |
|---|---|---|---|---|
| T_f | c0 20.4167, c1 2.13399, c2 0.541011, c3 -3.9868e-15 (c0 from first-record clocks; cadence error up to the record step per member; not an intercept) | P64_s2: 12.0000, 11.9167, +0.0833; P64_s4: 12.1250, 12.2917, -0.1667; P64_s8: 12.7500, 12.6667, +0.0833; S4_128_s8: 11.1875, 11.1875, +0.0000; S4_128_s16: 11.5625, 11.5625, -0.0000 | 9.13e-02; 2.04e-01 | 1.67e-01 |
| R_t_over_rs | c0 1.28404, c1 0.181139, c2 0.0320966, c3 -0.00970564 (first-record radii, at the sampling scatter of the first record; not a radius law) | P64_s2: 0.5953, 0.5809, +0.0144; P64_s4: 0.6024, 0.6312, -0.0287; P64_s8: 0.6957, 0.6814, +0.0144; S4_128_s8: 0.5698, 0.5698, +0.0000; S4_128_s16: 0.6247, 0.6247, -0.0000 | 1.57e-02; 3.52e-02 | 2.87e-02 |
| R_t_over_2gE | c0 1.39585, c1 0.19496, c2 0.0370334, c3 -0.00975495 (first-record radii, at the sampling scatter of the first record; not a radius law) | P64_s2: 0.6545, 0.6388, +0.0156; P64_s4: 0.6613, 0.6926, -0.0313; P64_s8: 0.7620, 0.7464, +0.0156; S4_128_s8: 0.6253, 0.6253, +0.0000; S4_128_s16: 0.6838, 0.6838, -0.0000 | 1.71e-02; 3.83e-02 | 3.13e-02 |

- S4_128_s4: the fit's value T_f 10.8125, R_t/r_s 0.5149, R_t/(2gE) 0.5668; the member's record: T_f 11.0625, R_t/r_s 0.4829, R_t/(2gE) 0.5300 (uncertified)

Every formation record of the 6 points (labelled; not a law):

- model: Y = c0 + c1 z + c2 h + c3 z h, z = ln(a/r_s), h = ln(sigma/a), on every formation record (labelled)
- members: P64_s2, P64_s4, P64_s8, S4_128_s4, S4_128_s8, S4_128_s16 (6 members, 4 coefficients, residual degrees of freedom 2)
- (z, h) per member: P64_s2 (-4.1589, 0.6931), P64_s4 (-4.1589, 1.3863), P64_s8 (-4.1589, 2.0794), S4_128_s4 (-4.8520, 1.3863), S4_128_s8 (-4.8520, 2.0794), S4_128_s16 (-4.8520, 2.7726)
- formation records inside this fit that are not certified (labelled; no law): S4_128_s4 (uncertified)

| Y | coefficients | member: Y, fit, residual | rms residual over members; per residual degree of freedom | max |residual| |
|---|---|---|---|---|
| T_f | c0 17.6667, c1 1.47275, c2 1.62303, c3 0.260171 (c0 from first-record clocks; cadence error up to the record step per member; not an intercept) | P64_s2: 12.0000, 11.9167, +0.0833; P64_s4: 12.1250, 12.2917, -0.1667; P64_s8: 12.7500, 12.6667, +0.0833; S4_128_s4: 11.0625, 11.0208, +0.0417; S4_128_s8: 11.1875, 11.2708, -0.0833; S4_128_s16: 11.5625, 11.5208, +0.0417 | 9.32e-02; 1.61e-01 | 1.67e-01 |
| R_t_over_rs | c0 1.63595, c1 0.265757, c2 -0.10637, c3 -0.0429998 (first-record radii, at the sampling scatter of the first record; not a radius law) | P64_s2: 0.5953, 0.5809, +0.0144; P64_s4: 0.6024, 0.6312, -0.0287; P64_s8: 0.6957, 0.6814, +0.0144; S4_128_s4: 0.4829, 0.4883, -0.0053; S4_128_s8: 0.5698, 0.5591, +0.0107; S4_128_s16: 0.6247, 0.6300, -0.0053 | 1.53e-02; 2.65e-02 | 2.87e-02 |
| R_t_over_2gE | c0 1.80085, c1 0.292343, c2 -0.12232, c3 -0.0480713 (first-record radii, at the sampling scatter of the first record; not a radius law) | P64_s2: 0.6545, 0.6388, +0.0156; P64_s4: 0.6613, 0.6926, -0.0313; P64_s8: 0.7620, 0.7464, +0.0156; S4_128_s4: 0.5300, 0.5362, -0.0061; S4_128_s8: 0.6253, 0.6131, +0.0123; S4_128_s16: 0.6838, 0.6899, -0.0061 | 1.68e-02; 2.91e-02 | 3.13e-02 |

The design rank (the requirements' edit 5): the model matrix [1, z, x^2, h, zh, j, j^2, zj] with x = 64 a/r_s, z = x^(4/3), h = (16 sigma/r_s)^2, j = 8 tau/r_s (the rank does not depend on the columns' normalization):

| row set | rows (r_s/a, sigma/a, tau/a) | rows | rank (of 8) | rows minus rank | singular values |
|---|---|---|---|---|---|
| certified | RS32_s4 (32, 4, 2); RS32_s4_tau1 (32, 4, 1); RS32_s2 (32, 2, 2); P64_s4 (64, 4, 8); P64_s2 (64, 2, 8); P64_s8 (64, 8, 8); P64_s4_tau4 (64, 4, 4); P64_s4_tau2 (64, 4, 2); S4_128_s8 (128, 8, 8); S4_128_s16 (128, 16, 8) | 10 | 8 | 2 | 18.99, 4.474, 4.276, 1.66, 0.9206, 0.2618, 0.1742, 0.0445 |
| every_formation_record | RS32_s4 (32, 4, 2); RS32_s4_tau1 (32, 4, 1); RS32_s2 (32, 2, 2); P64_s4 (64, 4, 8); P64_s2 (64, 2, 8); P64_s8 (64, 8, 8); P64_s4_tau4 (64, 4, 4); P64_s4_tau2 (64, 4, 2); S4_128_s4 (128, 4, 8); S4_128_s8 (128, 8, 8); S4_128_s16 (128, 16, 8) | 11 | 8 | 3 | 18.99, 4.558, 4.294, 1.676, 1.098, 0.2632, 0.2043, 0.0447 |

The review's rank-7 null (z - 1)(2/3 - 7h/24 + j) holds on its own 8-row set; the certified rows here have rank 8 of 8 on 10 rows, 2 rows beyond the rank (full rank alone is not a calibration).

### 5.4 The release radius (the 2 r_s preparation members beside the 4 r_s members)

| member | m a | R_0/r_s | first saved zero T_f/r_s (class) | R_t/a | R_t/r_s | root fraction | gate | last converged record |
|---|---|---|---|---|---|---|---|---|
| P64_s4 | 0.2 | 4 | 12.1250 (certified) | 38.556 | 0.6024 | 0.6613 | 0.104 | 40.0000 |
| S4_R2_m0p2 | 0.2 | 2 | 5.0000 (certified) | 40.393 | 0.6311 | 0.7431 | 0.099 | 6.0000 |
| M3a_s4 | 0.1 | 4 | none to 40.0000 | - | - | - | - | 40.0000 |
| S4_R2_m0p1 | 0.1 | 2 | 30.6250 (certified) | 47.972 | 0.7496 | 0.7842 | 0.083 | 32.1250 |

## 6 The convergence tables

### 6.1 Per member

| member | script; solver | engine entries in the records | records (dropped without E) | converged True as stored | counting records | converged False as stored (not a gate) | kept residual above 1e-3 | confirming residual above 1e-3: count; clocks; max (at) (Newton: solve_v32's; plain: the entry after the kept iterate, on the records kept unconverged) | residual max (at) | max over the records up to the last converged record | banded records in the certified extent † (outside it) | 30.3 prefix's last counting record | last converged record (certified prefix) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S4_16_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.98e-07 (26.2500) | 9.98e-07 | none (none) | 40.0000 | 40.0000 |
| RS32_s4 | engine_collapse_v31.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 641) | 641 (0) | 641 | 641 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.97e-07 (3.8125) | 9.97e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.97e-07 (15.8750) | 9.97e-07 | none (none) | 40.0000 | 40.0000 |
| S4_128_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 19.062, 306) | 306 (0) | 274 | 285 | 32 | 12 | 21; 3.6250, 3.7500, 3.8750, 4.0000, 4.1250, 11.5000, 11.5625, 11.6250 (and 13 more); 1.98e-01 (of 32 records kept unconverged, 25 with a following entry) | 2.66e-02 (11.8125) | 6.03e-05 | none (4.2500, 5.0000, 5.1250, 12.4375, 18.5625) | 3.8125 | 3.5625 |
| S4_16_s2 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.98e-07 (4.8750) | 9.98e-07 | none (none) | 40.0000 | 40.0000 |
| S4_16_s4_tau1 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 16.000, 257) | 257 (0) | 257 | 257 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.76e-07 (7.6875) | 9.76e-07 | none (none) | 16.0000 | 16.0000 |
| RS32_s2 | engine_collapse_v32.py; newton (at most 30 steps, 40 Krylov vectors, set None, ends above 0.001) | engine_collapse_v25 (0.000 to 40.000, 641) | 641 (0) | 640 | 641 | 1 | 0 | 0; none; 1.23e-04 (0.0000) | 1.86e-04 (0.0000) | 1.86e-04 | 0.0000 † (none) | 40.0000 | 40.0000 |
| RS32_s4_tau1 | engine_collapse_v25.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 1281) | 1281 (0) | 1281 | 1281 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.99e-07 (33.3438) | 9.99e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s2 | engine_collapse_v34.py; newton (at most 30 steps, 40 Krylov vectors, set resolve, ends above 0.001) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; 2.99e-06 (0.0000) | 9.89e-07 (7.8750) | 9.89e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s8 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.93e-07 (5.0000) | 9.93e-07 | none (none) | 40.0000 | 40.0000 |
| P64_s4_tau4 | engine_collapse_v19.py; plain (at most 20 iterations) | engine_collapse_v19 (0.000 to 40.000, 641) | 641 (0) | 640 | 641 | 1 | 0 | 0; none; 8.81e-04 (of 1 records kept unconverged, 1 with a following entry) | 6.18e-05 (13.4375) | 6.18e-05 | 13.4375 † (none) | 40.0000 | 40.0000 |
| P64_s4_tau2 | engine_collapse_v25.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 21.500, 689) | 689 (0) | 686 | 686 | 3 | 1 | 3; 12.7812, 12.8125, 13.4062; 7.08e-03 (of 3 records kept unconverged, 3 with a following entry) | 3.91e-03 (12.8125) | 9.98e-07 | none (none) | 12.7500 | 12.7500 |
| S4_128_s4_nk_v36m | engine_collapse_v34.py; newton (at most 30 steps, 40 Krylov vectors, set resolve, ends above 0.001) | engine_collapse_v25 (0.000 to 2.938, 48) | 48 (0) | 46 | 47 | 2 | 1 | 0; none; 7.51e-04 (2.9375) | 1.84e-03 (2.9375) | 8.20e-05 | none (none) | 2.8750 | 2.8750 |
| S4_128_s4_nk_v36q | engine_collapse_v34.py; newton (at most 30 steps, 40 Krylov vectors, set resolve, ends above 0.001) | engine_collapse_v25 (0.000 to 2.812, 46) | 46 (0) | 45 | 45 | 1 | 1 | 1; 2.8125; 5.48e-03 (2.8125) | 1.37e-03 (2.8125) | 9.91e-07 | none (none) | 2.7500 | 2.7500 |
| S4_128_s8 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 641) | 641 (0) | 641 | 641 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.99e-07 (18.0000) | 9.99e-07 | none (none) | 40.0000 | 40.0000 |
| S4_128_s16 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 641) | 641 (0) | 641 | 641 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 3.71e-07 (9.8750) | 3.71e-07 | none (none) | 40.0000 | 40.0000 |
| S4_R2_m0p2 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 318 | 319 | 3 | 0 | 2; 6.1250, 6.5000; 3.17e-02 (of 3 records kept unconverged, 2 with a following entry) | 8.42e-04 (6.1250) | 9.03e-07 | none (none) | 40.0000 | 6.0000 |
| S4_R2_m0p1 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 301 | 307 | 20 | 12 | 14; 32.2500, 32.3750, 32.5000, 32.6250, 32.7500, 32.8750, 33.0000, 33.1250 (and 6 more); 1.84e-01 (of 20 records kept unconverged, 18 with a following entry) | 1.27e-02 (32.7500) | 2.88e-04 | 32.0000 †, 32.1250 † (35.1250, 36.0000) | 32.1250 | 32.1250 |
| M3a_s4 | engine_collapse_v34.py; plain (at most 20 iterations) | engine_collapse_v25 (0.000 to 40.000, 321) | 321 (0) | 321 | 321 | 0 | 0 | 0; none; - (of 0 records kept unconverged, 0 with a following entry) | 9.99e-07 (21.2500) | 9.99e-07 | none (none) | 40.0000 | 40.0000 |

### 6.2 The records that do not count, with their gates (complete)

- S4_128_s4: 21 records do not count: 3.6250 (confirming residual above 1e-3; residual 5.30e-04, confirming 2.26e-03); 3.7500 (confirming residual above 1e-3; residual 6.49e-04, confirming 3.62e-03); 3.8750 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.32e-03, confirming 4.14e-03); 4.0000 (confirming residual above 1e-3; residual 9.84e-04, confirming 3.87e-03); 4.1250 (confirming residual above 1e-3; residual 6.85e-04, confirming 2.72e-03); 11.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.20e-03, confirming 3.49e-02); 11.5625 (confirming residual above 1e-3; residual 3.96e-04, confirming 8.66e-03); 11.6250 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.85e-03, confirming 8.04e-03); 11.6875 (kept residual above 1e-3; confirming residual above 1e-3; residual 9.36e-03, confirming 8.76e-02); 11.7500 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.61e-02, confirming 1.98e-01); 11.8125 (kept residual above 1e-3; confirming residual above 1e-3; residual 2.66e-02, confirming 2.78e-02); 11.8750 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.18e-03, confirming 4.83e-03); 11.9375 (kept residual above 1e-3; confirming residual above 1e-3; residual 5.88e-03, confirming 2.33e-02); 12.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 2.44e-03, confirming 3.39e-02); 12.0625 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.14e-03, confirming 1.87e-03); 12.1250 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.89e-03, confirming 2.51e-02); 12.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 8.16e-03, confirming 8.19e-03); 12.3125 (confirming residual above 1e-3; residual 4.24e-04, confirming 7.35e-03); 16.9375 (confirming residual above 1e-3; residual 3.84e-05, confirming 6.54e-03); 17.6875 (confirming residual above 1e-3; residual 7.01e-04, confirming 2.70e-02); 18.0000 (confirming residual above 1e-3; residual 2.76e-04, confirming 3.03e-03)
- P64_s4_tau2: 3 records do not count: 12.7812 (confirming residual above 1e-3; residual 6.96e-05, confirming 3.08e-03); 12.8125 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.91e-03, confirming 7.08e-03); 13.4062 (confirming residual above 1e-3; residual 3.63e-04, confirming 2.59e-03)
- S4_128_s4_nk_v36m: 1 records do not count: 2.9375 (kept residual above 1e-3; residual 1.84e-03, confirming 7.51e-04)
- S4_128_s4_nk_v36q: 1 records do not count: 2.8125 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.37e-03, confirming 5.48e-03)
- S4_R2_m0p2: 2 records do not count: 6.1250 (confirming residual above 1e-3; residual 8.42e-04, confirming 3.17e-02); 6.5000 (confirming residual above 1e-3; residual 4.94e-04, confirming 6.28e-03)
- S4_R2_m0p1: 14 records do not count: 32.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 2.47e-03, confirming 1.87e-02); 32.3750 (kept residual above 1e-3; confirming residual above 1e-3; residual 8.80e-03, confirming 1.84e-01); 32.5000 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.02e-02, confirming 6.51e-02); 32.6250 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.25e-02, confirming 6.10e-02); 32.7500 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.27e-02, confirming 1.49e-01); 32.8750 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.97e-03, confirming 4.72e-03); 33.0000 (kept residual above 1e-3; confirming residual above 1e-3; residual 8.25e-03, confirming 6.49e-02); 33.1250 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.07e-02, confirming 7.77e-02); 33.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.82e-03, confirming 1.02e-02); 33.3750 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.74e-03, confirming 1.18e-02); 33.6250 (kept residual above 1e-3; confirming residual above 1e-3; residual 1.35e-03, confirming 1.63e-02); 34.0000 (confirming residual above 1e-3; residual 9.80e-04, confirming 3.38e-03); 34.2500 (kept residual above 1e-3; confirming residual above 1e-3; residual 3.66e-03, confirming 4.02e-03); 35.3750 (confirming residual above 1e-3; residual 2.12e-04, confirming 1.05e-02)

### 6.3 The stored controls and the width and release pairs (second minus first)

| member | control | kind | formation classes (member / control) | dT_f (control - member) | dR_t (a) | dR_t/R_t | d(R_t/r_s) | d(R_t/(2gE)) | clocks counting and certified in both (from to) | max |d E/E0| | max |d min(N - |v|)| | max |d R_c|/r_s | max |d R_t| (a); max |d R_t|/R_t (at) where both hold a zero (count) | counting records regardless of certification (labelled diagnostic): common clocks; max |d R_t| (a); max |d R_t|/R_t (at); at 16 (count) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P64_s4 | P64_s2 | control_sigma | certified / certified | -0.1250 | -0.458 | -0.0119 | -0.0072 | -0.0069 | 321 (0.0000 to 40.0000) | 8.16e-02 | 3.11e-01 | 3.99e-01 | 11.303; 0.2223 (16.0000) (216) | 321; 11.303; 0.2223 (16.0000); 0.2223 (216) |
| P64_s4 | P64_s8 | control_sigma | certified / certified | 0.6250 | 5.971 | 0.1549 | 0.0933 | 0.1007 | 321 (0.0000 to 40.0000) | 6.16e-02 | 4.05e-01 | 5.74e-01 | 6.770; 0.1803 (28.3750) (64) | 321; 6.770; 0.1803 (28.3750); 0.0408 (64) |
| P64_s4 | P64_s4_tau4 | control_step | certified / certified | -0.1250 | -1.233 | -0.0320 | -0.0193 | -0.0253 | 321 (0.0000 to 40.0000) | 4.49e-02 | 1.67e-01 | 5.81e-01 | 7.871; 0.2044 (20.1250) (183) | 321; 7.871; 0.2044 (20.1250); 0.0649 (183) |
| P64_s4 | P64_s4_tau2 | control_step | certified / certified | -0.2188 | -3.887 | -0.1008 | -0.0607 | -0.0726 | 103 (0.0000 to 12.7500) | 3.16e-02 | 1.61e-01 | 1.06e-01 | 4.022; 0.1043 (12.1250) (6) | 173; 6.045; 0.1556 (21.1250); 0.1174 (36) |
| S4_16_s4 | S4_16_s2 | control_sigma | - | - | - | - | - | - | 321 (0.0000 to 40.0000) | 3.02e-03 | 5.81e-02 | 7.39e-01 | -; - (-) (0) | 321; -; - (-); - (0) |
| S4_16_s4 | S4_16_s4_tau1 | control_step | - | - | - | - | - | - | 129 (0.0000 to 16.0000) | 3.38e-05 | 1.06e-04 | 7.85e-04 | -; - (-) (0) | 129; -; - (-); - (0) |
| RS32_s4 | RS32_s2 | control_sigma | certified / certified | -1.6875 | -1.492 | -0.0691 | -0.0466 | -0.0520 | 641 (0.0000 to 40.0000) | 1.69e-02 | 4.54e-01 | 6.72e-01 | 4.448; 0.2061 (15.8125) (50) | 641; 4.448; 0.2061 (15.8125); 0.0888 (50) |
| RS32_s4 | RS32_s4_tau1 | control_step | certified / certified | -0.0938 | 0.284 | 0.0132 | 0.0089 | 0.0080 | 641 (0.0000 to 40.0000) | 4.55e-03 | 3.71e-02 | 1.11e-01 | 1.756; 0.0814 (15.8125) (55) | 641; 1.756; 0.0814 (15.8125); 0.0270 (55) |
| S4_128_s4 | S4_128_s4_nk_v36m | control_solver | - | - | - | - | - | - | 47 (0.0000 to 2.8750) | 5.83e-06 | 1.63e-06 | 1.35e-06 | -; - (-) (0) | 47; -; - (-); - (0) |
| S4_128_s4 | S4_128_s4_nk_v36q | control_solver | - | - | - | - | - | - | 45 (0.0000 to 2.7500) | 4.66e-06 | 1.29e-06 | 8.27e-07 | -; - (-) (0) | 45; -; - (-); - (0) |
| S4_128_s4 | S4_128_s8 | control_sigma | uncertified / certified | 0.1250 | 11.121 | 0.1799 | 0.0869 | 0.0953 | 58 (0.0000 to 3.5625) | 1.57e-04 | 6.89e-04 | 1.67e-03 | -; - (-) (0) | 285; 30.370; 0.4018 (14.6250); - (47) |
| S4_128_s4 | S4_128_s16 | control_sigma | uncertified / certified | 0.5000 | 18.147 | 0.2936 | 0.1418 | 0.1538 | 58 (0.0000 to 3.5625) | 7.79e-04 | 3.20e-03 | 8.31e-03 | -; - (-) (0) | 285; 36.533; 0.5193 (15.0000); - (46) |
| P64_s4 | S4_R2_m0p2 | preparation | certified / certified | -7.1250 | 1.836 | 0.0476 | 0.0287 | 0.0817 | 49 (0.0000 to 6.0000) | 6.76e-02 | 9.47e-01 | 2.04e+00 | -; - (-) (0) | 319; 9.878; 0.1899 (15.5000); 0.1700 (216) |
| M3a_s4 | S4_R2_m0p1 | preparation | - | - | - | - | - | - | 258 (0.0000 to 32.1250) | 3.68e-02 | 8.57e-01 | 2.26e+00 | -; - (-) (0) | 307; -; - (-); - (0) |
| S4_128_s8 | S4_128_s16 | control_sigma (the certified factor-2 pair at 128) | certified / certified | 0.3750 | 7.026 | 0.0963 | 0.0549 | 0.0585 | 641 (0.0000 to 40.0000) | 5.59e-02 | 5.13e-01 | 6.79e-01 | 22.398; 0.2844 (26.5000) (120) | 641; 22.398; 0.2844 (26.5000); - (120) |

The factor-2 averaging controls against the acceptance criteria (0.5 r_s in T_f, 10 percent in R_t): RS32_s2 against RS32_s4 (certified / certified): |dT_f| 1.6875 r_s, |dR_t|/R_t at formation 6.9%, max |dR_t|/R_t over 50 common certified zero records 20.6% at 15.8125; the clock spread exceeds 0.5 r_s; the radius difference over the common records exceeds 10 percent: the control fails; P64_s2 against P64_s4 (certified / certified): |dT_f| 0.1250 r_s, |dR_t|/R_t at formation 1.2%, max |dR_t|/R_t over 216 common certified zero records 22.2% at 16.0000; within both criteria at formation, the radius difference over the common records exceeding 10 percent; P64_s8 against P64_s4 (certified / certified): |dT_f| 0.6250 r_s, |dR_t|/R_t at formation 15.5%, max |dR_t|/R_t over 64 common certified zero records 18.0% at 28.3750; the clock spread exceeds 0.5 r_s; the radius spread at formation exceeds 10 percent; the radius difference over the common records exceeds 10 percent: the control fails; S4_128_s16 against S4_128_s8 (certified / certified): |dT_f| 0.3750 r_s, |dR_t|/R_t at formation 9.6%, max |dR_t|/R_t over 120 common certified zero records 28.4% at 26.5000; within both criteria at formation, the radius difference over the common records exceeding 10 percent.

## 7 The catalogs behind the chain-column onsets, and the r_s/a 32 member's preparation

| member | catalog read from | chain columns per channel (kappa: columns) | top channel | largest column count | 4 a times it (upper bound) | head comparison of the rerun preparation (N_B, E0, f_kin relative differences) |
|---|---|---|---|---|---|---|
| S4_16_s4 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of S4_16_s4 | 1:3 2:2 3:1 4:0 5:0 6:0 7:0 8:0 9:0 10:0 | 3 | 3 | 12 a | N_B 0.00e+00, E0 1.61e-15, f_kin 1.08e-15 |
| RS32_s4 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of RS32_s4 | 1:4 2:3 3:2 4:1 5:0 6:0 7:0 8:0 9:0 10:0 | 4 | 4 | 16 a | N_B 0.00e+00, E0 3.57e-16, f_kin 3.83e-15 |
| P64_s4 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| S4_128_s4 | status file t0 line of S4_128_s4_nk (same preparation arguments) | 1:6 2:6 3:4 4:4 5:3 6:2 7:1 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| S4_16_s2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of S4_16_s4 (same preparation arguments) | 1:3 2:2 3:1 4:0 5:0 6:0 7:0 8:0 9:0 10:0 | 3 | 3 | 12 a | N_B 0.00e+00, E0 1.61e-15, f_kin 1.08e-15 |
| S4_16_s4_tau1 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of S4_16_s4 (same preparation arguments) | 1:3 2:2 3:1 4:0 5:0 6:0 7:0 8:0 9:0 10:0 | 3 | 3 | 12 a | N_B 0.00e+00, E0 1.61e-15, f_kin 1.08e-15 |
| RS32_s2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of RS32_s4 (same preparation arguments) | 1:4 2:3 3:2 4:1 5:0 6:0 7:0 8:0 9:0 10:0 | 4 | 4 | 16 a | N_B 0.00e+00, E0 3.57e-16, f_kin 3.83e-15 |
| RS32_s4_tau1 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of RS32_s4 (same preparation arguments) | 1:4 2:3 3:2 4:1 5:0 6:0 7:0 8:0 9:0 10:0 | 4 | 4 | 16 a | N_B 0.00e+00, E0 3.57e-16, f_kin 3.83e-15 |
| P64_s2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s8 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s4_tau4 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| P64_s4_tau2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of P64_s4 (same preparation arguments) | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 2.41e-16, f_kin 7.70e-15 |
| S4_128_s4_nk_v36m | status file t0 line of S4_128_s4_nk (same preparation arguments) | 1:6 2:6 3:4 4:4 5:3 6:2 7:1 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| S4_128_s4_nk_v36q | status file t0 line of S4_128_s4_nk (same preparation arguments) | 1:6 2:6 3:4 4:4 5:3 6:2 7:1 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| S4_128_s8 | status file t0 line of S4_128_s8 | 1:6 2:6 3:4 4:4 5:3 6:2 7:1 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| S4_128_s16 | status file t0 line of S4_128_s4_nk (same preparation arguments) | 1:6 2:6 3:4 4:4 5:3 6:2 7:1 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |
| S4_R2_m0p2 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of S4_R2_m0p2 | 1:5 2:4 3:3 4:2 5:2 6:0 7:0 8:0 9:0 10:0 | 5 | 5 | 20 a | N_B 0.00e+00, E0 3.88e-16, f_kin 4.37e-15 |
| S4_R2_m0p1 | engine_collapse_v36d.prepare_dust on the box CPU with the run json's arguments of S4_R2_m0p1 | 1:6 2:6 3:4 4:4 5:3 6:2 7:1 8:0 9:0 10:0 | 7 | 6 | 24 a | N_B 0.00e+00, E0 3.46e-16, f_kin 1.47e-15 |
| M3a_s4 | status file t0 line of M3a_s4_tau4 (same preparation arguments) | 1:6 2:5 3:4 4:4 5:3 6:2 7:2 8:0 9:0 10:0 | 7 | 6 | 24 a | - (status file) |

The r_s/a 32 member's preparation rule: the engine_collapse_v36d preparation (the well's filling within kappa_sea 10, prepare_dust with fill_lowest) run on the box CPU with that member's stored arguments reproduces its stored head: N_B relative difference 0.0e+00, E0 3.6e-16, f_kin 3.8e-15; it is the same rule as every other member's.

## 8 Not composed, truncated or open

- The (0.2, 16) member S4_16_s4 holds no trapping zero to 40.000; neither do its sigma/a 2 control (to 40.000) and its tau 1 control (to 16.000).
- The (0.2, 128) sigma/a 4 member S4_128_s4 (plain) is certified to 3.5625 (the record 3.625: confirming residual above 1e-3; kept 5.30e-04, confirming 2.26e-03); its first kept residual above 1e-3 is at 3.8750; its records continue to 19.0625 with a first zero at 11.0625 (R_t 61.815 a), uncertified. Its Newton runs stopped early: v36m at 2.9375, certified to 2.875 (the record 2.9375: kept residual above 1e-3; kept 1.84e-03, confirming 7.51e-04); v36q at 2.8125, certified to 2.75 (the record 2.8125: kept residual above 1e-3; confirming residual above 1e-3; kept 1.37e-03, confirming 5.48e-03); no Newton run of this member has landed.
- The sigma/a 2 point at r_s/a 64 is the composite history (Newton across its seams), certified through its last record; the plain rung 2 json (md5 a405b4923eab8ff22ddccab78e59af6d) is not on the box.
- RS32_s2 is certified through its last record, its T = 0 record banded (kept 1.86e-04, confirming 1.23e-04); the r_s/a 32 averaging-scale comparison is certified on both sides (6.3).
- The (0.2, 64) release member at 2 r_s, S4_R2_m0p2, is certified to 6.0 (the record 6.125: confirming residual above 1e-3; kept 8.42e-04, confirming 3.17e-02); its formation at 5.0 lies inside.
- S4_128_s8 finished on R1 (queue one_S4_128_s8: finished 2026-10-19T23:14:48Z, resumed from its T/r_s 30 slice at 21:22:33Z); read from its final run json mirrored to the box (records to 40.0000).
- Not composed (the reading's routing): the birth located between records (busi B7), the (32, 4, 8) step member, the T = 0 valley census, chi^(1/3) on the shed energy share, epsilon_c and c_b.

## 9 Figure data and figures

- figure: <internal>/SCALING_ladder_v1.3_fig.png (and .pdf), drawn by <internal>/ladders_figs_v1_3.py from the arrays below
- arrays: <internal>/ladders_readout_v1_3.npz (per member per record: T/r_s, E/E0, R_t, R_t/r_s, R_t/(2gE), min(N - |v|), R_c, residual, confirming residual, converged as stored, counts, certified, conf_fail, capped, above, the number of zeros, tangencies, pockets, inversions, slope at R_t; keys <member>__<column>), md5 in SOURCES
- numbers: <internal>/ladders_readout_v1_3.json (formation, episodes, matched clocks, fits with coefficients and residuals, control spreads, catalogs)

## 10 SOURCES

| member | run json (as read) | bytes | md5 | script field |
|---|---|---|---|---|
| S4_16_s4 | <internal>/collapse_lad_S4_16_s4_m0p2_rs16_n512_R04_k10_tau2_ell4_prep4_A_R2_avg_plain_v36d.json | 4389648 | b87134f9742e57c16b0cb3bfd1b2fd9b | engine_collapse_v34.py |
| RS32_s4 | <internal>/collapse_c31_rs32_A_R2_tau2_ell4_prep4_avg.json | 8699390 | 898fd8ebe0dd9424de6f20074d04f568 | engine_collapse_v31.py |
| P64_s4 | <internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json | 4857142 | d7d69adcea7a43f8375ab4c7c5a80efc | engine_collapse_v34.py |
| S4_128_s4 | <internal>/collapse_lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 4580941 | 5fe643bb6c3ef225f05e194c38908a6a | engine_collapse_v34.py |
| S4_16_s2 | <internal>/collapse_lad_S4_16_s2_m0p2_rs16_n512_R04_k10_tau2_ell2_prep4_A_R2_avg_plain_v36d.json | 4453652 | 52f02084c6fcb0ca9690af82b9bb5680 | engine_collapse_v34.py |
| S4_16_s4_tau1 | <internal>/collapse_lad_S4_16_s4_tau1_m0p2_rs16_n512_R04_k10_tau1_ell4_prep4_A_R2_avg_plain_v36d.json | 3502488 | 9689881076f6c34eab7f691327be555b | engine_collapse_v34.py |
| RS32_s2 | <internal>/collapse_c32_rs32_A_R2_tau2_ell2_prep4_avg_nk40r0b.json | 10821396 | d35bc0fd5f1b7765e495668f4300b90c | engine_collapse_v32.py |
| RS32_s4_tau1 | <internal>/collapse_c25_rs32_A_R2_tau1_ell4_avg.json | 17071585 | 6e8b8973c771c6a4a01db8926c30a46a | engine_collapse_v25.py |
| P64_s2 | <internal>/collapse_c32_rs64_A_R2_tau8_ell2_prep4_avg_nk40r2b_v36cc_v36d.json | 7014125 | 0a005ac5f66bef02a0104e4278867765 | engine_collapse_v34.py |
| P64_s8 | <internal>/collapse_c32_rs64_A_R2_tau8_ell8_prep4_avg_plain_v36d_from0_to40.json | 4650292 | d1ca037a77c198aa83b33055495eceff | engine_collapse_v34.py |
| P64_s4_tau4 | <internal>/collapse_c19_rs64_A_R2_tau4_ell4_avg.json | 9046968 | f2e1d8a7483447426855e3211535a367 | engine_collapse_v19.py |
| P64_s4_tau2 | <internal>/collapse_c25_rs64_A_R2_tau2_ell4_avg.json | 9405229 | 9ba6f6ff302d8a903925f20583c96656 | engine_collapse_v25.py |
| S4_128_s4_nk_v36m | <internal>/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m.json | 882492 | 2889024ae017804f75cec4e152e6c6e2 | engine_collapse_v34.py |
| S4_128_s4_nk_v36q | <internal>/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q.json | 854928 | ea458c364755815db5584b98cbb4b1aa | engine_collapse_v34.py |
| S4_128_s8 | <internal>/collapse_lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json | 9473926 | ff475918f57758118221aa59fda52a86 | engine_collapse_v34.py |
| S4_128_s16 | <internal>/collapse_lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d.json | 8985158 | c2aee55e56201f6a216aa9844dd6bcdb | engine_collapse_v34.py |
| S4_R2_m0p2 | <internal>/collapse_lad_S4_R2_m0p2_m0p2_rs64_n2048_R02_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 4812694 | aae1a09ee3ebdeac0ad5efc726ce5df3 | engine_collapse_v34.py |
| S4_R2_m0p1 | <internal>/collapse_lad_S4_R2_m0p1_m0p1_rs64_n2048_R02_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 4559199 | d9b428d05430ac7e989fd2f62f92c5a1 | engine_collapse_v34.py |
| M3a_s4 | <internal>/collapse_lad_M3a_s4_m0p1_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json | 4425449 | 778dc5cef53012b71461a2e1d2de1044 | engine_collapse_v34.py |

- catalog (chain columns): <internal>/catalog_v1.json (md5 007fec3e0073e36cc402522ba1a36795), written by <internal>/ladders_catalog_v1.py
- catalog (chain columns): <internal>/catalog_v1_A.json (md5 50084b438ea991ecfd4edc02c9f7cfd1), written by <internal>/ladders_catalog_v1.py
- catalog (chain columns): <internal>/catalog_v1_B.json (md5 37a7ec570643c2a8cd7cc4bf0a4c2a72), written by <internal>/ladders_catalog_v1.py
- producer: <internal>/ladders_readout_v1_3.py (md5 20048af9566537dc7540b4dad33b4e8a), version 1.2's producer with the production certification rule, rerun on the same stored files; helper engine_ladder_compare_v5.py (md5 94ac400330b2ebf032cdb1312c520d53)
- certification inputs checked by the producer: <internal>/certext_production_v1.json (md5 64c2dfd1e5a4bccaf61e1ad8be2ee717, LADDER_certification_list_v1.1) and <internal>/certext_followup_v1.json (md5 0bf27951ca23c9b6012aadde84038989, written by <internal>/certext_followup_v1.py, the following-entry test)
- composer: <internal>/ladders_compose_v1_3.py and <internal>/compose_text_v1_3.py; the readout json <internal>/ladders_readout_v1_3.json (md5 930e75da87c1559f0318ed7a8a5206ed); the arrays <internal>/ladders_readout_v1_3.npz (md5 b4c8b2f6a285dfd929f9895a9d81981d), mirrored on the box at <internal>/v1_3 with its md5 manifest
- engine of the catalog's rerun preparations: <internal>/engine_collapse_v36d.py (md5 0315e0826ee872ee194474876ef523f9)
- the reading applied: deliveries/lead_gaps/FableAPI_reading_LADDERS_v1.md (reviewer A Part A, reviewer B Part B, the lane's cross-check); the certification rule applied: deliveries/lead_gaps/FableAPI_cert_rule_v1.md
- W4 (a) comparator numbers quoted in 5.1: <internal>/s4_density_continuum_rest_lapse-on_adaptive.json (through deliveries/claude_alt_solve/W5_W4_formation_attribution_v1.md section 4)
