# LAMBDA_I4_SPEC_v1: the fermion share I_4^iso of the fourth-order transverse polarization (two blind routes)

Source of the integrand: api/r7/A_brk_lambda.md (reviewer B, md5 8695b547): eqs. (6)-(8), (15), (24)-(30). Validated line by line by api/r7/F_brk_lambda_val.md (reviewer A, md5 f0df0d48), whose section 3 is the binding recipe.

Integrand: mu = 0.2; k in [-pi/2, pi/2]^3; the four (n, t) of A eq. (26); s_+-, c_+- as degree-4 truncated series in z from (28); e_+- = e sqrt(1 + y_+-) by (29); R, S, Cbar by (6)-(8); T by (15). f_4 = [z^4] of T/(e_- + e_+), by series division. f_iso = (2/7) f_100 + (8/35)(f_110par + f_110perp) + (9/35) f_111.
Per-k independent check of the series arithmetic: [z^4] g = (1/16) sum_{j=0..15} g(r e^{2 pi i j/16}) r^-4 e^{-pi i j/2}, with r = 0.04. Report the maximum per-k mismatch.

Route A (trapezoid): Trap_288 = 288^-3 sum_j f_iso(-pi/2 + pi j/288). Proved bound |I - Trap_288| < 5e-11 (A eq. 35).
Route B (Gauss-Legendre): GL_512, normalized weights, nodes (pi/2) x_j. Proved bound |I - GL_512| < 5e-12 (A eq. 38).

Each route reports its value with its proved bound plus float-rounding error (outward), and the sign of I_4^iso. Acceptance: the two intervals intersect (A eq. 40). The routes stay blind to each other until both have frozen their notes: LAMBDA_I4_ROUTE_A_v1.md and LAMBDA_I4_ROUTE_B_v1.md.
Compute: box cores 10-13 or pod CPU only, never box GPU 1. Outputs go under <internal>/. Script files only, no python -c, no deletes, kill only own pids.
