# NONCONS_indep_G2G4_v1: independent recomputation of check C1c (G2, G4) for NONCONS_universal_v1

Lane: busi, inbox #486 item (f); the test is reviewer A (3)(i) in deliveries/lead_gaps/ReviewA_noncons_v1.md. Author: an independent checker agent (analyst), not the author of NONCONS_universal_v1 or its scripts. Independence: nothing in busi/code/noncons/ or <internal>/noncons was opened, copied, imported or run, and none of the original outputs were read. The only inputs were the written definitions: NONCONS_universal_v1 sections 1 and 4 (model, conventions, the G2/G4 coordinates in row C1c, hypothesis (A')) and VERTEX_stress_v1 eq. (1) (the on-site converter W_x and the pair operators Phi). The NONCONS v1 values in row C1c were used only for the final comparison.

**Verdict: PASS on both criteria.** |A|_F^2(G2) = 16 exactly. The four-energy Jacobian on the momentum-conserving manifold has exact rank 4 at G2 and at G4.

## 1. How I read the definitions

- Model. a = 1 and m = 3/5. h_{0,A}(p) = m_A beta + sum_i alpha_i sin p_i, with alpha_i = tau_x (x) sigma_i, beta = tau_z (x) I_2, and component order (upper up, upper down, lower up, lower down). The bands are E_H = (m^2 + |sin p|^2)^{1/2} and E_L = |sin p|.
- Converter (VERTEX_stress_v1 (1)). W_x = sum_mu eta_mu Phi_{L,x}^{mu dag} Phi_{H,x}^mu + h.c., with Phi_A^mu = c_A^T M^mu c_A (no factor 1/2), M = (tau_z (x) sigma_y, tau_y (x) sigma_z, i tau_y (x) I_2, -tau_y (x) sigma_x) and eta = diag(1,-1,-1,-1). The script checks that all four M^mu are antisymmetric.
- Stripped amplitude. A[t1 t2; s1 s2] = V_s <L k1 t1; L k2 t2| V |H p1 s1; H p2 s2>, with every leg in the positive band and |pair> = c^dag_1 c^dag_2 |0>. The antisymmetry of M gives <0|Phi^mu|12> = -2 phi_1^T M phi_2. This yields the pair formula A = 4 sum_mu eta_mu conj(v1^T M^mu v2)(u1^T M^mu u2), which is the same as NONCONS v1 section 4. The h.c. term does not contribute to HH -> LL.
- |A|_F^2 is the sum of |entries|^2 over the 16 spin entries. It does not depend on the spin basis chosen within each positive band, because a unitary change of basis on any leg leaves it unchanged. Route R2 below uses no spinor basis at all.
- Shell, from (A'). E_H(p1) + E_H(p2) = E_L(k1) + E_L(k2) with k1 + k2 = p1 + p2. All four legs must lie in the valley-0 cube |p_j| < pi/2 with energies in I = (m, 1/a) = (3/5, 1).
- The tested Jacobian. The momentum-conserving manifold is R^9 with coordinates (p1, p2, k1), where k2 = p1 + p2 - k1. The map is x -> (E_H(p1), E_H(p2), E_L(k1), E_L(k2)), so the Jacobian is 4x9. Rank 4 of this map is equivalent to two things together: dG != 0 for G = E_L1 + E_L2 - E_H1 - E_H2, and surjectivity of the 3-energy map (E_H1, E_H2, E_L1) restricted to the shell, which is the form used in Theorem 2. Both forms were computed. A third reading would drop momentum conservation (12 free variables). The rank there is trivially 4, so it is not informative and was not used.
- Points, from row C1c. G2: p1 = (t,0,0), p2 = (0,0,t), k1 = (t,t,0), k2 = (0,-t,t), with t = asin 3/5. G4: the same, except the first component of p1 and k1 is asin 5/13.

## 2. Method

The calculation is exact in sympy 1.14. Each momentum component is carried as an exact (sin, cos) pair: asin 3/5 -> (3/5, 4/5) and asin 5/13 -> (5/13, 12/13). Momenta are added with the angle-addition formulas, so momentum conservation is checked exactly (the difference is sin 0, cos 1 in every component). No floating-point value enters any result.

The amplitude is computed by three routes:

- R1, explicit spinors. u_s = (E + h) e_s / (2E(E+m))^{1/2}. The script asserts exactly that each is an eigenvector with eigenvalue +E and that the pair is orthonormal. A is then evaluated with the pair formula.
- R2, basis-free. |A|_F^2 = 16 sum_{mu,nu} eta_mu eta_nu conj(T_L[mu,nu]) T_H[mu,nu], with T[mu,nu] = tr(M^mu^T P_1 conj(M^nu) P_2^T) and band projectors P = (1 + h/E)/2.
- R3, Fock space from first principles. An explicit 8-mode fermionic Fock space (4 heavy and 4 light components on one site) with exact CAR signs. W_0 is built directly from the definition, and A = <0| b_L(v2) b_L(v1) W_0 b_H(u1)^dag b_H(u2)^dag |0>. This checks the factor 4, the antisymmetrisation and the sign of the pair formula entry by entry.

For the Jacobian, dE/dp_j = sin p_j cos p_j / E. Multiplying each row by its E_i > 0 does not change the rank, and it makes every entry rational, so the rank is computed exactly over Q. A nonzero 4x4 minor is printed as a certificate. Separately, the rank of the 3-energy map on ker dG (an 8-dimensional space) is computed with the unscaled exact rows.

Run: box compute-host, conda qi2 python, nice -n 10, OMP_NUM_THREADS=8. It took 12.8 s of user CPU.

## 3. Exact results

| | G2 | G4 |
|---|---|---|
| E_H(p1), E_H(p2) | 3 sqrt2/5, 3 sqrt2/5 | sqrt(2146)/65 = 0.712691373655, 3 sqrt2/5 = 0.848528137424 |
| E_L(k1), E_L(k2) | 3 sqrt2/5, 3 sqrt2/5 | sqrt(2146)/65, 3 sqrt2/5 |
| energy balance, all energies in (3/5, 1), valley-0 cube | exact 0; yes; yes | exact 0; yes; yes |
| \|A\|_F^2, R1 = R2 exactly | **16** | **32 - 624 sqrt(1073)/1073 = 12.950449660772896945** (minimal polynomial 1073 x^2 - 68672 x + 709376) |
| R3 Fock-space tensor = R1 tensor entrywise | exact | exact |
| scaled Jacobian (rows E_i dE_i, cols p1, p2, k1) | rows (12/25 e_1), (12/25 e_6), (12/25 e_7 + 12/25 e_8), (0,-12/25,12/25, 0,-12/25,12/25, 0,12/25,-12/25) | the same with 12/25 -> 60/169 in entries (1,1) and (3,7) |
| exact rank, 4-energy map on momentum-conserving manifold | **4** (minor on cols (0,1,5,6) = -20736/390625) | **4** (minor on cols (0,1,5,6) = -20736/714025) |
| rank of (E_H1, E_H2, E_L1) on the shell tangent space; dG != 0 | 3; yes | 3; yes |

Comparison with NONCONS_universal_v1, row C1c:

- The energies agree: all four are 3 sqrt2/5 at G2, and at G4 they are sqrt(2146)/65 = 0.7127 and 3 sqrt2/5 on each side.
- |A|_F^2(G2) = 16 agrees.
- |A|_F^2(G4) = 12.95 agrees. v1 gives no closed form in the note; the exact value is 32 - 624/sqrt(1073).
- Exact rank 4 at both points agrees.

At G2 the two light-heavy energy pairs coincide (E_H1 = E_L1 = E_H2 = E_L2). The rank is nevertheless 4 because the four momentum gradients are independent. G4 breaks that degeneracy and still has rank 4.

## 4. PASS/FAIL per criterion (reviewer A (3)(i))

1. |A|_F^2(G2) = 16 exactly: **PASS**. Three routes agree exactly, one of them basis-free and one built from first principles in Fock space.
2. Four-energy Jacobian of rank 4 on the momentum-conserving manifold at G2: **PASS** (exact over Q, with a nonzero-minor certificate).
3. The same at G4: **PASS**.

Overall (3)(i): **PASS**. Hypothesis (A') of Theorem 2 is independently confirmed at G2 and G4 for the on-site converter. The filtered converter is not part of this item; it is item (e)/(3)(ii).

## 5. Files

- Script, local: <internal>/indep_G2G4_v1.py, md5 d3021eef27c76048ecd91d4c56ee7de8 (identical on the box).
- Box, <internal>/: indep_G2G4_v1.py (d3021eef27c76048ecd91d4c56ee7de8), indep_G2G4_v1.out (bf13055d3561739a78e062e5759eaebc), indep_G2G4_v1.err (4c94b11628ef2dbaadb465674955b0a1, /usr/bin/time report only), and MD5SUMS.

Nothing was committed. No other lane's files and no manuscript were touched.
