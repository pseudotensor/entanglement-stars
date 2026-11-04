# LAMBDA_I4_ROUTE_B_v1 (lane alt, route B, blind): I_4^iso by the tensor Gauss-Legendre rule GL_512 with normalized weights

Status: frozen (444). Spec: LAMBDA_I4_SPEC_v1.md (955a986a); integrand A_brk_lambda.md (8695b547) eqs. (6)-(8), (15), (24)-(30); binding recipe
F_brk_lambda_val.md (f0df0d48) sec. 3; proved bound A eqs. (38)-(39). Blind: route A's work not read or asked for. CPU only (box core 10 and three
m1pole-A CPU threads), no GPU. Files in sec. 5.

## 1. Result

  I_4^iso in [-0.0038160087913770616, -0.0038160087824998293]      GL_512 = -0.0038160087869384455, total outward radius 4.439e-12, sign: negative
  Pi_4,T^iso = -2 I_4^iso in [0.007632017564999658, 0.007632017582754124];  c_4,band^iso = -I_4^iso in [0.0038160087824998293, 0.0038160087913770616]   (A eq. (27)).

Radius budget: proved quadrature bound 12 M rho^-(2n-1)/(rho - 1), delta = 0.08, M = 8e8, rho = 1.052226, n = 512: 4.436e-12; floating-point evaluation of
f_iso at every node including the node and weight uncertainties (running forward error analysis, sec. 3): 2.553e-15; summation (math.fsum per chunk and
over the 512 chunk sums, exactly rounded, together with the products w_i w_j w_k f, counted at 8u of the absolute sum 0.0039): 3.465e-18; the three
added with a 4u margin and the endpoints stepped outward by one double.

## 2. What was computed (the binding recipe, F sec. 3)

mu = 0.2; k in [-pi/2, pi/2]^3 (A eq. (30)); the four (n, t) of A (26) with weights 2/7, 8/35, 8/35, 9/35 (A (25)); s_+-, c_+- = sin, cos(k_i +- z n_i/2)
as degree-4 truncated series in z (A (28)); e_+- = e sqrt(1 + y_+-), y_+- = (e_+-^2 - e^2)/e^2, by (29) (exact to z^4 since y = O(z)); R = sum_j t_j s_j c_j
cos(z n_j) (6); S_r = (z/4)(n x t)_r Cbar_r, Cbar_r = [c_i(k_+) c_j(k_+) + c_i(k_-) c_j(k_-)]/2, (i, j, r) cyclic (7)-(8); T by (15); f_4 = [z^4] T/(e_- + e_+)
by truncated series division (reciprocal recurrence); f_iso = (2/7) f_100 + (8/35)(f_110par + f_110perp) + (9/35) f_111.
GL_512 = sum_{i,j,k} w^_i w^_j w^_k f_iso((pi/2) x_i, (pi/2) x_j, (pi/2) x_k), w^_j = 1/((1 - x_j^2) P_512'(x_j)^2), sum w^ = 1 (A (37)); the sum is
taken chunk by chunk in k_1 (512 chunks of 512^2 nodes), each chunk by math.fsum, the 512 chunk sums by math.fsum.

Independent check of the series arithmetic at each k (F sec. 3): [z^4] g = (1/16) sum_{j<16} g(r e^{2 pi i j/16}) r^-4 e^{-pi i j/2}, r = 0.04, with g
evaluated from the exact complex expressions (no series), at EVERY node and for each of the four (n, t): maximum mismatch 2.69e-11 absolute
(4.41e-11 relative to the largest |f_4| = 0.6099); per direction (100, 110par, 110perp, 111): 1.71e-11, 2.62e-11, 1.61e-11, 2.69e-11. The check's own resolution is
set by double rounding of g = O(1) divided by r^4 = 2.6e-6, about 4e-11, and the 16-point rule's aliasing (0.04/0.23)^16 ~ 7e-13 relative; it is a
check of the series code, not part of the certified radius.
f_iso at the nodes ranges over [-0.514004, 0.000385].

## 3. Floating-point certification (what the radius covers)

- Nodes: numpy leggauss(512) nodes x_j certified by a sign change of P_512 at x_j +- 5e-16 in 40-digit decimal interval arithmetic (three-term
  recurrence; no node needed refinement): max node error 6.110e-16; weights w^_j = 1/((1 - x^2) P'(x)^2) as decimal intervals over the root bracket,
  max weight error 4.080e-15; sum of the double weights - 1 = 0.0e+00. The nodes and weights are stored once (out/nodes_n512_v2.npz, md5 164f4a69) and
  loaded by every worker, so all hosts used identical doubles.
- Node coordinates k_j = fl(fl(pi)/2 x_j): eps_k = (pi/2)(max node error + 2u) = 1.309e-15 is injected as an absolute error on sin k_i, cos k_i (Lipschitz 1);
  sin, cos of the 512 node values by libm (math.sin, math.cos) with a 1-ulp (2u) allowance.
- f_iso at every node: every series coefficient carries an absolute error bound (u|x| per rounding; |a| db + |b| da + da db per product; m u sum|terms|
  per m-term sum; the reciprocal recurrence with its division bounds, its constant term asserted bounded away from 0); the rounded direction
  components 1/sqrt2, 1/sqrt3 (relative 3u), the cross products n x t, mu^2 = fl(0.2)^2 (4u) and the weights 2/7, 8/35, 9/35 (u) carry their own terms;
  the exact-zero constant term of y_+- (computed residue 0.0e+00) is kept inside its error bound. The weight errors enter as sum |f| (dw_i w_j w_k + ...).
- Summation and products: 8u of the absolute sum (two exactly rounded fsum levels and the three weight products).
- Arithmetic is elementwise IEEE double (numpy) in blocks of 4096 nodes; the values do not depend on the block size (checked: chunk 0 identical at
  blocks 8192 and 32768, chunk 1 at 4096 and 2048).

Cross-checks of the worker against the serial first version (lambda_i4_routeB_v1.py e90940ef, same series code without the completed error terms):
n = 16: GL bitwise equal (-0.0038550560906085393); n = 512, chunk 0: 1.192592092756148e-10 (v1 printed 1.192592092756e-10).

## 4. Workers and core-hours

| part file | chunks | host | block | seconds | s per chunk | md5 |
|---|---|---|---|---|---|---|
| part_v2_1_box10_0_128_partial.json | [0, 128) | gpu | 4096 | 1619 | 22.8 | 3e346dab |
| part_v2_1_pod_128_256.json | [128, 256) | 5fd61034b136 | 4096 | 1562 | 12.2 | f2ddb852 |
| part_v2_1_pod_256_384.json | [256, 384) | 5fd61034b136 | 4096 | 1553 | 12.1 | 8edf16b9 |
| part_v2_1_pod_384_512.json | [384, 512) | 5fd61034b136 | 4096 | 1561 | 12.2 | 7847ddcd |
| part_v2_1_pod_71_90.json | [71, 90) | 5fd61034b136 | 4096 | 237 | 12.5 | 14d9ce81 |
| part_v2_1_pod_90_109.json | [90, 109) | 5fd61034b136 | 4096 | 232 | 12.2 | a5cef0c6 |
| part_v2_1_pod_109_128.json | [109, 128) | 5fd61034b136 | 4096 | 235 | 12.4 | 5fc341ed |
Worker seconds in total 6999 (1.94 core-hours); hosts: 5fd61034b136/numpy 1.26.3/python 3.11.10; gpu/numpy 2.2.6/python 3.12.12.

## 5. Files

lambda_i4_nodes_v2.py 56964b6f -> out/nodes_n512_v2.npz 164f4a69, out/nodes_n512_v2.json 1eb8fda8; lambda_i4_routeB_v2_1.py 16984915
(worker; v2 eac0af8e plus incremental part files); lambda_i4_routeB_v2_combine.py d480e111 -> out/lambda_i4_routeB_GL512_v2.json 74ee6c8e;
part files and logs under out/ (table above); lambda_i4_routeB_v1.py e90940ef (first version, serial; out/test_n16.json, killed at chunk 1 of 512).
Box <internal>/routeB; pod m1pole-A /workspace/alt_btable_v2/lambda_i4/. Slips: none in this item.
