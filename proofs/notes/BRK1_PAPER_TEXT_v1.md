# BRK-1 sentence: paper text for paperapi (v1)

From busi. The text is reviewer B astra_busi_brk1cadenceA (2fa1fa83) sec. 4, VERBATIM, extracted by awk.
- Basis: rdebitb2 rAii_v3 rung 1 at step 13253 (history M1dt): ||L_T||+ = 0.47531 (w = g) and 0.56918 (w = 1). G0-G2 passed. The values come from the run logs; the json rerun is due about 12:10Z.
- reviewer B ruled that the radius test needs mu, delta, e_0 and beta in ONE norm, and it evaluated the test exactly:
  - w = 1 (delta = 1.913e-4) opens dt, 1a and 2a at both budgets;
  - w = g opens dt only at beta = 1e-3, and dt/1a/2a at 1e-2;
  - NEITHER opens 8a.
- Single-step bounds do not give the uniform-in-time sup required, so continuation at any positive cadence is NOT established today. The sentence says exactly that.
- An update may follow if rAii_v4 tightens the bounds (about 13:00Z). The uniform clause stays either way.
- Owner word rule: checked clean ("establish" is the permitted word).

----- BEGIN TEXT -----
At step 13253 we establish \(\mu_T\le0.47531/a\) in the \(g\)-weighted norm and \(\mu_T\le0.56918/a\) in the unweighted norm, making \(2a\) an unweighted-norm target at both stated budgets; to establish continuation by two independent calculations, we still need seed, total-defect and growth bounds valid throughout each interval.
----- END TEXT -----
