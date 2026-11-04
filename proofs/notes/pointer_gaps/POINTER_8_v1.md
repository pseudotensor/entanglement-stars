# POINTER_8_v1: the K = 8 interior bound and the assumed capacity at the computed trapping radius (06_scaling:216)

Numbers emitted by jq from pointer_notes/pointer8_numbers_v1.json (md5 9b24b6f09960f447737604b33131886a), written by pointer8_numbers_v1.py (eebe575add6ff1c50063c302131054ac) on box CPU from the stored reference collapse's record json <internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json (md5 e27e457d37615cff8d5216f05a494f89).

- Computed trapping radius: R_t = 49.48070699859734 a, the stored record at T/r_s = 13.5 (1 record at 49.48a).
- K = 8 interior bound 32 K (K+1) (R_t/a) ln 2 = 79021.23851103091 nats.
- Assumed capacity 4 pi f (R_t/a)^2 (f A/(4G) at G/a^2 = 0.25), with the paper's f = 0.887: 27290.109394503263 nats.

The same two numbers, checked by hand by both reviewers, are in brk9_release/BRK9_CLOSED_SENTENCE_v1.md (b90e699c), line 13: "4 pi (0.887)(49.48^2) = 27 289.5 (00 prints 27 290); 2304 ln2 x 49.48 = 7.90e4"; the bound is derived in brk9_release/round4/fable_fresh_theorem_result.md (bf7fdd48) sec. (2).

In this repository the script is `code/solve_a/pointer_notes/pointer8_numbers_v1.py` and its output is `data/solve_a/pointer_notes/pointer8_numbers_v1.json`.
