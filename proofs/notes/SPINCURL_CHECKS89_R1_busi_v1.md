# SPINCURL_CHECKS89_R1_busi_v1: spin-curl checks 8 and 9, route 1 (busi)

busi implementer (builder analyst), inbox #743 (and #745), 30 Oct 2026.
- Spec: FableAPI_dec3n_v1.md (md5 bc35140c), section 6, items 8 and 9, with sections 1, 2 and 7. Paper: Eq. (dm-hamiltonian) at commit 7eca3f52603 (PAPER_flat_7eca3f52603.tex), Appendices B and D.
- Reused, md5-checked:
  - busi's checks 1-7 route-1 library (sc_lib_v1.py 6a0d64c3);
  - busi's items34 Appendix D construction: items34_lib 3bfa9c88, n3_background_v1 3e076017, s1 npz e4a6d55e, s2 d2 npz 59690a0b;
  - the engine ops engine_ops_v1 301c45a2 and engine_fields_v4 6ab508c2.
- Blind to route 2: no other lane's SPINCURL file, code or box directory was opened.
- Box: jon@192.168.1.93, <internal>/spincurl89_r1.
  - CPU jobs ran at 4 threads per pool (nlwp 7-17), ulimit 16 GB.
  - Route B ran on GPU 0 (ulimit 96 GB under #741). It was admitted with 48.6 GB free and peaked at 1.9 GB RSS.
  - GPU 1 and GPU 2 were not used. No API stages, no commits.
- Code: busi/code/spincurl89_r1/. All 12 files are byte-identical to the box's code/ directory (md5 lists compared).
- Supporting files: SPINCURL_CHECKS89_R1_busi_v1_files/ holds the output JSONs, the route A/B and item 8 v1 logs, and box_MD5SUMS.txt.

## Per-item table

| item | output | threshold | measured | verdict |
|---|---|---|---|---|
| 8a | max \|W^dag curl v\| on every history entering the entropy budget, per history | = 0 | 0.0 exactly on all 6 histories (51 distinct slices; per-history table below) | PASS |
| 8b | B.6's clipped comparison flow printed as a potential flow wherever used, or the clipping redone through the potential | curl v_cmp = 0 | Redone through the potential on the only numerical B.6 construction (the NOTCOMP 3.2b kit, record c19 r11, T/r_s 15). grad Phi_cmp - v_cmp = 0.0, and max \|W^dag curl v_cmp\| = 0.0 at 1479 sites | PASS (redone) |
| 8c | measured \|\|h_spin\|\| / ((3/2) C_1 eps) on random fields of Lipschitz constant C_1 eps | <= 1 | max 0.234 (Lanczos); max 0.468 for the rigorous row-sum upper bound. 34 fields | PASS |
| 8d | rotational-path estimates | none | nothing printed | - |
| 9a | max \|curl b\| / \|grad b\| over the stored flow bumps | printed | 1.000 pointwise and as a ratio of sups (L2 ratio sqrt(2/3) = 0.8165). Back-averaged at the sites: 0.908-1.000 over all 771 flow bumps | printed |
| 9b | anchor/path enclosure, two routes | gap > 0 along the path | path gap >= 0.12887 (A), >= 0.15393 (B); paper's 0.12 holds on both | PASS x2 |
| 9c | gap enclosure, two routes | > 0 | g0 >= 0.199216605072 (A), >= 0.199208 (B) | PASS x2 |
| 9d | Gram lambda_min(Gamma) > 0, two routes | > 0 | >= 3.4212e-11 (A), >= 3.8013e-11 (B); float 3.80139e-11 | PASS x2 |
| 9e | strict-mixedness radius against 1.1e-7, two routes | >= 1.1e-7 | eta_max >= 1.88e-7 (A), >= 2.41e-7 (B). Margin at 1.1e-7: 2.36e-5 (A), 3.07e-5 (B) | PASS x2 |

## Two-route enclosure table (item 9; spin vertex in the test vertices and in P(u); b = 0, d = 2, 1028 tests)

| enclosure | route A | route B | items34, no spin (for reference) | both pass |
|---|---|---|---|---|
| background spin site field (W^dag curl v_0) | 0.0 exactly (analytic Jacobian of the radial flow) | 6.0e-9, curl on the kernel (a quadrature value); its operator bound 4.5e-9 was subtracted from B's gap | - | yes |
| gap of h[u_0] | >= 0.19921660507 (Kahan/Parlett on a full eigh) | >= 0.199208 (Cholesky of h^2 - (g^2 + c) I) | >= 0.19921660507 | yes |
| \|\|h[u_0] - h_0\|\| | <= 0.14148 (row sum) | <= 0.091342 (Lanczos plus Cholesky certificate) | <= 0.1415 | - |
| path gap from the flat sea, (m + g0 - \|\|dH\|\|)/2 | >= 0.12887 | >= 0.15393 | >= 0.1289 | yes (> 0.12) |
| Gram lambda_min, energy block (257) | >= 7.4347e-10 | >= 8.2607e-10 | 7.43e-10 | yes |
| Gram lambda_min, momentum block (771, spin-coupled) | >= 3.4212e-11 | >= 3.8013e-11 | 4.93e-12 | yes |
| anchor: \|\|Z\|\|, margin of C_0 = eps - \|\|Z\|\| | <= 4.36155e-5, margin >= 5.6384e-5 | <= 4.36207e-5, margin >= 5.6379e-5 | 4.36159e-5, 5.638e-5 | yes |
| Lambda_lin (all / clock / flow) | 298.28 / 21.38 / 297.51 | 233.00 / 16.44 / 232.43 | 500.9 / 28.1 / 500.2 | - |
| Lambda_h (all) | 0.67213 | 0.60399 | 0.67211 | - |
| eta_max, all directions | 1.88e-7 | 2.41e-7 | 1.12e-7 | yes |
| eta_max, clock only / flow only | 2.63e-6 / 1.89e-7 | 3.40e-6 / 2.41e-7 | 2.00e-6 / 1.12e-7 | yes |
| margin at the paper's 1.1e-7 (all) | 2.357e-5 | 3.074e-5 | not computed | yes |

**Plainly: all four applicable enclosures (anchor/path, gap, Gram, strict mixedness) pass on both routes with the spin vertex.** With the spin vertex the paper's numbers hold:
- the gap stays above 0.12 along the straight path;
- the Gram matrix is positive definite;
- C(u) stays strictly mixed for bump coefficients up to 1.1e-7.

The 1.1e-7 radius now has more margin than before: 1.88e-7 against 1.1255e-7 without spin. The spin part adds a positive semidefinite term to the momentum Gram (cross traces vanish), which raises lambda_min 7x and lowers Lambda_lin from 501 to 233-298. Per the brief, I do not remove the mark; the lead does.

Route A agrees with route B:
- the Gram matrices agree to 3.5e-18 (max entry);
- A_d agrees to 1.4e-17;
- the bump fields F and grad F agree to 2.5e-16 and 5.6e-17;
- the h0 and Q operators are identical to the engine's (0.0);
- the seven operator types have disjoint supports (overlap count 0, checked in B).

Route A (FLOAT) finite-amplitude rows along its computed worst direction:
- at 1 x eta_max, the evaluated margin is 7.65e-5;
- at 4 x eta_max, it is 4.32e-5;
- the gap of h[u] is 0.199216 with 4218 negative levels, so no level crosses.

The bound is therefore conservative.

## Constructions

Spin vertex on Appendix D's ball:
- S_k = Sigma^lat_k is the restriction to the R_b = 8 ball of the open cube's (c_i c_j) (x) Sigma_k, with c_i = (T_i + T_i^dag)/2. This uses the same restriction convention as the engine's h0 and Q, reproduced to 0.0.
- Test vertices (momentum): A_(q,i) = 1/2{G(y_q - x), Q_i} + 1/4 sum_jk eps_ijk 1/2{d_{y_j} G(y_q - x), S_k}. This is J_l = W J + 1/4 curl W Z with the candidate's Gaussian kernel.
- In P(u): h[phi_j] gains 1/8 sum_k {(W^dag curl phi_j)_k, S_k} for flow bumps, with W^dag curl(e_a psi) = grad(G * psi) x e_a.
- The radial background has zero spin site field, so h[u_0], P(u_0), the gap and the path are unchanged.
- The anchor still changes, because the eps-mixture's spin moments enter L: the spin density of P(u_0) is 6.0e-7 (max) and the spin moment 6.5e-10. ||Z|| moves by about 1e-10.

Route A (s89_routeA_v1 + s89_libA_v2, CPU):
- Uses the engine's ball operators plus S_k, and field-form operators.
- The Gram is built by the exact structured formula in long double and certified by long-double Cholesky with Higham's bound.
- Inequality chain and bounds are items34's: row-sum Lambda_lin and Lambda_h, Kahan/Parlett ||Z||, mpmath interval-Hessian remainder, Davis-Kahan dP.
- The orbital-only Gram reproduces the stored items34 Gram exactly (0.0), and its certificate (4.9317e-12) and the gap reproduce to all digits.

Route B (s89_routeB_v1, GPU 0 for the dense steps), independent code. It does not use route A's code, the engine operators, the engine kernels, sympy or mpmath:
- own ball lattice, Dirac matrices and hop lists;
- own bump integrals (a direct double Gauss-Legendre integral of G and dG/ds);
- own numpy laws, with A_d by complex step;
- Gram by bond factorisation (Gamma = sum_t B_t^T B_t) in long double, certified by double eigh plus the Kahan/Parlett residual theorem evaluated in long double;
- solves by eigendecomposition with long-double refinement;
- Lambda_lin^2 = ||sum_j X_j^2|| and Lambda_h^2 in kernel form (no loop over j);
- every operator norm certified as a Lanczos estimate plus a dense Cholesky of (sI - A);
- gap by a Cholesky of h^2 - g^2 I;
- remainder by its own analytic Hessian majorant.

Shared between the routes:
- the stored DATA (u_0 site fields, background jets, rho/J site densities in A);
- the inequality chain (Weyl + Davis-Kahan + Cauchy-Schwarz).

## Item 8 details

**8a per history** (route i: curl from the analytic Jacobian, then K_ell * B_a back-averaging with the history's own ell, the operator's construction; route ii: curl moved onto the kernel, a quadrature check). Every slice of P02 route 1 (the 44 slices, driver_ef2.sh) and TRAPPING_CUT_EF R2 (the 20 records, tcef_run6.py) is included, at 36 lattice sites per slice.

| history | slices (R1 / R2) | ell | max \|W^dag curl v\| route i | route ii / max flow site field |
|---|---|---|---|---|
| ISO_64 ladder (rs64) | 17 (17 / 0) | 4 | 0.0 | 1.2e-9 |
| ISO_128 ladder (rs128) | 5 (5 / 0) | 4 | 0.0 | 2.9e-10 |
| c19 rs24 B4n1536 | 5 (5 / 0) | 4 | 0.0 | 1.3e-10 |
| c32 rs32 nk40r0b | 8 (8 / 8) | 2 | 0.0 | 3.3e-9 |
| c19 rs64 r11 | 10 (5 / 10) | 4 | 0.0 | 1.7e-9 |
| W256 newton rd13mf (rs256) | 6 (4 / 2) | 4 | 0.0 | 1.2e-10 |

All are radial channel-chain histories. The 44 + 20 slice count is reproduced: 51 distinct slices, 13 shared.

**8b.** The kit clipped the SITE field at s0 = 1, s_c = 0.95 w_c = 1.0668 (w_c = 1.1230), so its comparison was not a smooth flow. The clipping was redone through the potential instead:
- The smooth radial profile is clipped with the same tanh map, giving v_cmp(r) r_hat = grad Phi_cmp with Phi_cmp(r) = int_0^r v_cmp; grad Phi_cmp - v_cmp = 0.0.
- W^dag curl v_cmp = 0.0 by route i. Route ii gives 4.7e-7, quadrature error on a C1 profile.
- After back-averaging, |v_cmp|/N reaches at most 1.0648 < s_c, so the frozen-gap premise holds (the kernel is positive).
- The redone comparison differs from the kit's site clip by at most 4.7e-3 in the site field, and by 2.8e-3 where the kit leaves the field unclipped. That spread is the kernel's support, which B.6's "within two sites of U" must then be read to include (red team R4).
- The other B.6 use, SOURCE_REMAINDER24 D1's reviewer A C7 certificate, is symbol-level and has the same radial setting.
- The kit was never run in production, and no printed paper number uses B.6.

**8c.** C_1 eps is taken per component, as max_a sup |grad v_a|. It is sampled on a 0.25 grid (a lower bound of the sup, so the ratio is not understated). The parent is periodic, L = 16, ell = 2.
- 24 compact fields (radius 2.5-4.5; potential, curlA and turn bumps): max ratio 0.059 (row-sum 0.175). Three pure-potential fields give exactly 0.
- 10 slowly varying periodic fields (item8c_v3):
  - ABC (Beltrami) flows: 0.196-0.234 (row-sum <= 0.468); the A = B = C ABC flow at k = 2 pi/16 gives the max, 0.234.
  - shear: 0.131.
  - random 5-mode fields: 0.074-0.095.
- The rule's kernel weights sum to 1.0000032.
- Analytically, the ratio is <= 1 because ||S_k|| <= 1 and the kernel is a positive average.

## Self red-team

- **R1, model scope.** Item 9 recertifies Appendix D's PM1 candidate as printed: Gaussian kernel ell = 2, d = 2, b = 0, the bump family w = 4. b = +-1 and d = 3, 4 were not rerun.
- **R2, shared inputs.** Both routes take the stored background (u_0 site fields, jets) as DATA. So does items34 (jets checked there at 4e-10). Route A also uses the stored rho/J site densities, which it recomputes to 0.0 from its own eigh.
- **R3, shared mathematics.** Route independence is in the code and the numerical certificates, not in the inequality chain. Both routes use the same Weyl/Davis-Kahan/Cauchy-Schwarz bound, and the same Higham/Kahan theorems with assumed IEEE rounding (x87 long double u = 2^-64). Route B's complex Cholesky uses a conservative gamma_{8(n+1)} for the blocked GPU algorithm; I did not derive that constant.
- **R4, 8b reading.**
  - Clipping the smooth flow, then back-averaging, changes the site field slightly beyond the kit's clipped set (up to 2.8e-3 where the kit is unclipped). U must therefore be taken as the smooth pocket enlarged by the kernel support.
  - For non-radial potential flows, clipping through the potential is not generally possible. Every history here is radial, so this does not arise.
- **R5, 8a coverage.**
  - The zero is structural: a radial field's Jacobian is symmetric bitwise. That is the paper's construction, and route ii confirms it only to quadrature accuracy.
  - "Entropy budget" was read as the trapping-cut direct-state budget (K <= 7.07 over 44 histories; K <= 7.49 over 20 records). Other entropy readings (Page curve records) were not enumerated.
- **R6, 8c.** Lanczos ran at tol 1e-6 (v2 stalled at 1e-10 on the near-degenerate top of the smooth spectra and was stopped). The row-sum column is the rigorous upper bound. Only L = 16 was run, so the smallest eps is 2 pi/16 against ell = 2.
- **R7, Z spin density.** The background's spin density is nonzero (6e-7), which I did not expect from parity. It is included in the anchor; its effect on ||Z|| is 1e-10.
- **R8, process.**
  - item8_v1 crashed in part c (ARPACK, a zero h_spin); parts a and b in its log equal v2's.
  - item8c_v2 was stopped twice: once (pidfile plus ps) to put route B on GPU 0 first (#745), once for the ARPACK stall. v3 replaced it.
  - libA_v1 (dense long-double products) was replaced by v2 (sparse; same Gram, reproduced to 0.0 vs stored).
  - Slips: a local `perl -0pi` and local `sed -i 's/...'` text substitutions on my own code files, no compute; logged by the lead (#745). On the box, `python3 -c 1` was typed in an ssh line once (box, harmless).
  - One own job at a time throughout; one chain script sequenced them.

## Cost (timed)
- probe 554 s (dense long-double Gram).
- route A 463 s to the bound, 673 s with the finite rows (CPU).
- route B 289 s (GPU 0).
- item 8 v2 7.8 min; item8c v3 91 s.
- No run approached 2 box-hours.

## Box MD5SUMS
<internal>/MD5SUMS (md5 of the file 35239be2d1d453cb5f919b26fb11e331), copied to the _files folder. Main entries:
```
fdf58b5eb942675ad71c0214b3e83807  code/s89_routeB_v1.py
c99e67b1d0d091a907d6a9261de494a8  code/s89_routeA_v1.py
3726b5c1f27521167f6669c88537a792  code/s89_libA_v2.py
8dd105e212640ed61c80913711d2d572  code/s89_item8_v2.py
016c81049dd6df51f4a821732f6632bc  code/s89_item8c_v3.py
7527f8f0c97c1ccda4195beb95787723  out/routeA_v1.json
6f959afe8d943da3d4ec1ee06db685bf  out/routeB_v1.json
123df11b6a7c4a3fec1b9accc2883bd6  out/item8_v2.json
b4835fd84cd3a38a228eab48c66ea8f9  out/item8c_v3.json
9f156cc94c23ddb00ac3e5fa5805e4fe  out/probe_v1.json
```
Superseded (not used): s89_item8_v1 part c, s89_item8c_v2, s89_libA_v1.

No box job of mine is running: every pidfile was checked, the chain has exited, and GPU 0 was released.
