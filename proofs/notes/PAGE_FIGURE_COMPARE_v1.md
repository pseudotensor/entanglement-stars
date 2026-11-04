# PAGE FIGURE: two-route comparison (lead, mechanical; no rulings)

Route 1: icloud, deliveries/codex_solve/PAGE_FIGURE_R1_v1.md + page_curve_amps_R1.{pdf,py,csv} (box <internal>/page_fig_r1).
Route 2: solve, deliveries/claude_alt_solve/PAGE_FIGURE_R2_v1.md + page_curve_amps_R2.{pdf,py,csv} (box <internal>/page_fig_r2).
Both blind to each other. Governing: WRITE_PAGE_SIGNED_v1.md, FIGURE SPEC sections A-J.

## REQUIRED numbers (as printed by each route)

| Quantity | Route 1 | Route 2 | Agreement |
|---|---|---|---|
| C_0 | 0.00523731547127 (rel. diff from spec +5.2e-11) | 0.00523731547128 | ~1e-12 relative |
| D_0 | 0.107876529936 | 0.107876529936 | all printed digits |
| calR = D_0/(4 pi C_0) | 1.63911101758 | 1.63911101758 | all printed digits |
| T_W | 0.00263960826448 | 0.00263960826448 | all printed digits |
| r_0 (post-Page uniform share) | 0.805043401354722 | 0.805043401355 | all printed digits |
| z_x | 0.625927907892850 | 0.625927907893 | all printed digits |
| tau_x | 0.514018924681352 | 0.514018924681 | all printed digits |
| Page peak f = 1 (z_P, tau_P) | 0.621084526820049 at 0.510530015901316 | (spec reproduction <= 5.3e-11) | consistent |
| Gamma_k(q) anchors (k = 1,2,4,8; q = .01,.1,.3,...) | 16-digit table | 13-digit table, two forms | agree to ~1e-11 relative (largest at k = 8, q = .01: 2.343988157166e-54 vs 2.343988157163/7284e-54) |
| LP normalized gap | 0 exact (exact-chain certificates); 960/1920 envelope change 7.08e-6 < 1e-5 | 6.0e-14 | both pass I.6 |

## Acceptance differences (J and I)

- J item 13 (dot separated from the Page guide): Route 1 marks J PASS, "the dot/guide separation is resolved in the inset without moving either coordinate"; "the main-panel marker footprint is wider than that small separation, so no distinct main-panel gap is claimed". Route 2 marks item 13 FAIL in the main panel only (tau_x - tau_P = 0.68 pt at full width; about 13 pt apart in the inset); no change made; needs a spec decision. The two routes describe the same geometry and differ only in how they read item 13.
- Caption: Route 2 notes the caption should state the inset share range [0.35, 0.65] (inset ticks unlabelled to keep 8 pt lettering).
- I.2 outer scattering endpoint: Route 1 constructs an explicit residual-to-solution bound at both endpoints (exterior: exp(theta_b) B rho_b^-N / [N(1 - 1/rho_b)]). Route 2's inner endpoint has an explicit bound; its outer endpoint uses series smallest term plus x10 extension, which it labels diagnostic, not rigorous.
- Both: physical identification of the comparator response is UNDECIDED; the admissible shading is not an error band; the formal endpoint is not a completion claim.
