# LAMBDA_I4_COMPARE_v1: the two blind routes for I_4^iso agree

Spec: LAMBDA_I4_SPEC_v1.md (955a986a).
- Route A (cubic-a): Trap_288, LAMBDA_I4_ROUTE_A_v1.md, md5 7a8e0ece20401b15584d117427e9ee9e. Interval [-0.0038160088336, -0.0038160087403] (half-width 4.67e-11).
- Route B (statecount-r1): GL_512, LAMBDA_I4_ROUTE_B_v1.md, md5 d855547949ac380451885f37ed6f3716. Interval [-0.0038160087913770616, -0.0038160087824998293] (radius 4.44e-12).

Acceptance (A eq. 40): the intervals intersect. B lies entirely inside A, so the intersection is B's interval. The midpoints differ by 8.2e-12, within both radii.

Result: I_4^iso = -0.00381600879 (+- 4.5e-12) at ma = 0.2. Sign negative. -2 I_4^iso = 0.0076320176. Printable: I_4^iso = -3.8160088 x 10^-3.
