# POINTER_14_v1: the thirteen second-gradient terms and the explicit momentum law (gauss_law_premises:39, App. C)

Paper text: "cubic symmetry, inversion, and time reversal leave the energy row six second-gradient terms and the momentum row seven (listed in Ref.), thirteen coefficients ... (explicit form in Ref.)".

## What the code has, plainly

The engine code (engine_collapse_v36d 0315e082, engine_fields_v16 1ffcf5e8) and the W_2 module (dmres_w2chain_v1 b9c3716c) contain no list of the thirteen terms.
- The engine implements only the spherical rows at the paper's coefficients: engine_fields_v16.integrate_UL, rhs_pos, U' = 8 pi g r^2 (rho_ell + q J_ell) and (ln N)' = -4 pi g r J_ell / q.
- The W_2 module tabulates the second-gradient reference.
- No file vaclaw_w2_v3 exists under the solve lane's box directories.

So the list below comes from the symmetry count, checked by a script, not from the engine. The count agrees with the paper: 6 + 7 = 13. No overclaim to correct.

## The count (script ptr14_count_v1.py 072472bf, box CPU; output ptr14_count_v1.json)

**Setup.**
- One-derivative, degree-zero blocks: D = grad v / N (9 components; polar x polar; odd under time reversal, since the flow is odd) and g = grad N / N (3; polar; even).
- Second-derivative blocks: d d v / N (V (x) Sym^2 V, 18; odd) and d d N / N (Sym^2 V, 6; even).
- The flow enters only through its derivatives. O_h is the 48 signed permutations, inversion included.

**Multiplicities by characters** (1/48 sum chi_W chi_target):
- **Energy row** (scalars, even under time reversal): Sym^2(D) gives 4, Sym^2(g) gives 1, and d d N gives 1, for a total of 6. The odd blocks D (x) g and d d v contribute 0 and 0: inversion already removes them.
- **Momentum row** (polar vectors, odd under time reversal): d d v gives 3 and D (x) g gives 4, for a total of 7. The even blocks contribute 0, 0 and 0.
- Total 13.

**The energy row's six terms** (K = (grad v + grad v^T)/(2N), Omega = (grad v - grad v^T)/(2N)):
1. (tr K)^2
2. tr(K^2)
3. sum_i K_ii^2 (cubic)
4. tr(Omega^T Omega)
5. |grad N/N|^2
6. lap N/N (equivalently div(grad N/N), modulo term 5)

The three strain terms are the A1g, E_g and T_2g squares of K; K x Omega has no invariant, since K is in A1g + E_g + T_2g and Omega in T_1g.

**The momentum row's seven terms:**
1. grad(div v)/N
2. lap v/N
3. e_i d_i^2 v_i / N (cubic)
4. K g
5. (tr K) g
6. e_i K_ii g_i (cubic)
7. Omega g

Terms 1-3 are the three O_h-independent vectors linear in the second derivatives of v. div K, grad tr K and div Omega are combinations of them plus terms 4-7, since d_j(N Omega_ij) = d_i(N tr K) - d_j(N K_ij).

## Reciprocity and the explicit laws

Reciprocal response (the laws are the variations of one density) reduces the 13 coefficients to the five of Eq. (dm-density): A = (N/16 pi G)[a_1 (tr K)^2 + a_2 tr(K^2) + a_3 sum_i K_ii^2 + b tr(Omega^T Omega) + c |grad N/N|^2].

**Momentum law.** With dA/d(d_i v_j) = (1/8 pi G)[a_1 (tr K) delta_ij + a_2 K_ij + a_3 delta_ij K_jj + b Omega_ij], in which N cancels:

  J_ell,j = dA/dv_j = -(1/8 pi G) [ a_1 d_j tr K + a_2 d_i K_ij + a_3 d_j K_jj + b d_i Omega_ij ].

At a_1 = -1, a_2 = 1, a_3 = b = c = 0 this is J_ell = -div[K - (tr K) I]/(8 pi G), Eq. (dm-h2).

**Energy law.** dA/dN = -A/N - (c/8 pi G) div(grad N/N), the paper's form.

**Spherical reduction.** K = diag(K_r, K_t, K_t) with K_r = v'/N and K_t = v/(rN) gives the rows of eq:dr-radial, which the engine integrates (above).
