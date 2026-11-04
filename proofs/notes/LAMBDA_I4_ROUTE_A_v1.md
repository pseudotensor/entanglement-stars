# LAMBDA_I4_ROUTE_A_v1 -- I_4^iso by the trapezoid sum Trap_288 (route A, blind)

Frozen 2026-11-04 08:47Z by cubic-a. Spec: LAMBDA_I4_SPEC_v1.md (955a986a). Integrand: api/r7/A_brk_lambda.md (8695b547), eqs. (6)-(8), (15) and (24)-(30), with the recipe of api/r7/F_brk_lambda_val.md (f0df0d48) section 3. Route B's work was not read.

## Result

I_4^iso = Trap_288 = -0.003816008786938482

Outward interval: I_4^iso in [-0.0038160088335999695, -0.0038160087402769943]

Sign: **negative**. The whole interval lies below zero.

Response coefficient (eq. 27): Pi_4,T^iso = -2 I_4^iso in [0.007632017480553989, 0.007632017667199939].

The interval's half-width, 4.67e-11, is almost all the proved quadrature bound of eq. (35). The float-rounding bound is 7.6e-16.

## Method

- Integrand: mu = 0.2 and the four (n, t) of eq. (26). The series s_+-, c_+- are degree-4 truncated z-series from eq. (28). Then e_+- = e sqrt(1 + y_+-) by eq. (29), with y_+- of order z, so its powers truncate cleanly. R, S and Cbar come from eqs. (6)-(8) and T from eq. (15). f_4 = [z^4] T/(e_- + e_+) is taken by series division.
- Isotropic combination: f_iso = (2/7) f_100 + (8/35)(f_110par + f_110perp) + (9/35) f_111, per eq. (25).
- Sum: Trap_288 = 288^-3 times the sum of f_iso over k_i = -pi/2 + pi j_i/288, all 288^3 = 23,887,872 points. The sum uses math.fsum over all point values, which is correctly rounded.
- Float-rounding bound: every series coefficient carries a first-order running error bound. It covers products and sums (u|result| plus (2n + 2)u times the sum of |a_i b_{n-i}| per coefficient), inexact constants, 2u per libm sin and cos at the base point plus the argument rounding, and the square-root and reciprocal recurrences. The bound is the summed point bounds divided by N^3, plus u|fsum|/N^3 and 2u|Trap|. Each part's full error sum enters, which over-counts and so is conservative.
- Quadrature bound: eq. (35) with M = 8e8, delta = 0.08 and xi = exp(-46.08), evaluated as M expm1(3(log1p(xi) - log1p(-xi))) to avoid cancellation. It gives 4.666e-11, below the 5e-11 of eq. (36).

## Checks

- **Cauchy check at every k.** [z^4] g = (1/16) sum_j g(r e^{2 pi i j/16}) r^-4 e^{-pi i j/2}, with r = 0.04 and g evaluated from the full complex functions (bilinear products, principal square root). The maximum mismatch against the series value over all 288^3 points, per direction (100, 110par, 110perp, 111), is 1.67e-11, 2.66e-11, 1.59e-11 and 2.82e-11. That is consistent with the Cauchy sum's own rounding of order r^-4 eps |g|.
- **Running bound against extended precision.** Every eighth plane, 36 planes in all, was recomputed in numpy longdouble. The largest |f_64 - f_ld| is 1.17e-15, and the largest ratio of deviation to running bound is 0.040, so the bound covers the actual float error.
- **Straggler cross-check.** The original worker for planes 72-143 (pid 52525) logged per-plane fsums. For all 32 planes it completed, its fsum is bit-identical to the fsum of the same plane from the split workers.

## Provenance (box <internal>/routeA)

The scripts are lam_i4_trapA_v1.py (caad82c0), which computes the planes, and lam_i4_combineA_v3.py (7e2fb0a0), which combines them. v2 (74ce0ca8) printed the eq. (35) bound as 0 through floating cancellation; v1 is 88d88a54. The combined json is out/trapA_N288_combined_v3.json (d36e78d9).

The parts are each a range of j1 planes:

| part | planes | md5 |
|---|---|---|
| p0 | 0-71 | a42ce11c |
| q72 | 72-83 | 635084d3 |
| q84 | 84-95 | aa8e61f8 |
| q96 | 96-107 | e1b1606b |
| q108 | 108-119 | 38ed4a16 |
| q120 | 120-131 | e3ca6ad8 |
| q132 | 132-143 | 9849feb9 |
| p2 | 144-215 | 0807419e |
| p3 | 216-287 | 7ff801f6 |

The q parts were the straggler split. The compute used box CPU only, cores 10-13 plus idle cores 25, 28 and 32. The summed CPU time was 7,060 s.
