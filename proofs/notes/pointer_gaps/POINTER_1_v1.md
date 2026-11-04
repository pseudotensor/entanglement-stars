# POINTER_1_v1: the closed-form lapse behind 05_star:228 (Sec. VI C, after eq:sc-clock-exact)

Paper text: "whose integral from r to r_1 gives N(r) in closed form (Ref.)". This note gives the closed form, its derivation from the paper's own rows, and a script check on stored records.

## Closed form

For M > 0 on [r, r_1] (w nonzero) on one connected solution, r < r_1:

  N(r) = N(r_1) (M(r_1)/M(r))^{1/2} exp( - int_r^{r_1} 2 pi s^2 rho_ell(s) / M(s) ds ),

or equivalently N(r) = N(r_1) exp( - int_r^{r_1} ((lambda - 1)/2) d ln M ), with lambda = 4 pi r^2 rho_ell / M'. On an annulus of constant lambda this is N proportional to M^{(lambda-1)/2}, the paper's next clause.

## Derivation from the paper's rows (eq:dr-charge, eq:dr-radial)

1. The rows give w = v/N, M = r w^2/(2G), M' = 4 pi r^2 (rho_ell + w J_ell r), and v N'/(4 pi G r N^2) = -J_ell r.
2. The second row gives (ln N)' = N'/N = -4 pi G r J_ell r N / v = -4 pi G r J_ell r / w.
3. The charge row gives w J_ell r = (M' - 4 pi r^2 rho_ell)/(4 pi r^2), so J_ell r = (M' - 4 pi r^2 rho_ell)/(4 pi r^2 w).
4. Substituting, (ln N)' = -G (M' - 4 pi r^2 rho_ell)/(r w^2) = -(M' - 4 pi r^2 rho_ell)/(2M), since r w^2 = 2GM. That is 2 pi r^2 rho_ell/M - M'/(2M), Eq. (sc-clock-exact).
5. Integrate from r to r_1: ln N(r_1) - ln N(r) = int_r^{r_1} 2 pi s^2 rho_ell/M ds - (1/2) ln(M(r_1)/M(r)). This gives the closed form.

Nothing else enters: no quasi-static or same-source premise.

## Script check on stored records

- **Script:** ptr1_clock_v1.py (bbee77fc), run on box CPU core 29. Output ptr1_clock_v1.json; files in pointers_v1/ (MD5_pointers_v1.txt).
- **History:** the stored r_s/a = 32 history lad_S4_32_s4_tau8_m0p2_rs32_n1024_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.
  - run json md5 33576e218228a120038113a7eac900da, fields md5 9a0715acc8572c85028e4540f1d7ff52;
  - engine_collapse_v36d 0315e0826ee872ee194474876ef523f9, engine_fields_v16 1ffcf5e88c7953d7bfee025d57f4fb86.
- **Engine variables.** In the engine, U = r w^2 = 2GM; U' = 8 pi g r^2 (rho_ell + q J_ell) and (ln N)' = -4 pi g r J_ell / q, q = w (engine_fields_v16.integrate_UL, rhs_pos). So the closed form reads N(r) = N(r_1) (U(r_1)/U(r))^{1/2} exp(-int_r^{r_1} 4 pi g s^2 rho_ell/U ds).
- **Re-solve.** Each record's own sources are re-solved by the engine's call, engine_fields_v16.solve_constraints_ivp, with the run's options. The re-solved node lapse reproduces the stored N_nodes (column "re-solve").
- **Evaluation.** The closed form is evaluated by Simpson's rule (h = ell/512) with r_1 = the collar radius. Two versions of U are used:
  - A: the solve's U integration;
  - B: U = W/N^2 from the independent W integration.
  - The window starts at r = ell; below it the two log terms cancel near the regular centre.

| T/r_s | R_t (a) | window (a) | N at window start | re-solve vs stored N_nodes, max abs | closed form A vs engine N, max rel | closed form B vs engine N, max rel | closed form B vs stored N_nodes, max rel |
|---|---|---|---|---|---|---|---|
| 0 | none | [4, 236] | 0.7742093000154513 | 2.220446049250313e-16 | 4.625189120588402e-12 | 4.617195514811101e-12 | 8.057998712729386e-13 |
| 10 | none | [4, 1024] | 0.7348825133314022 | 7.205347429817266e-14 | 1.5556289589824246e-10 | 1.123967585670016e-11 | 1.0019207685729725e-11 |
| 18 | 26.28743420353739 | [4, 1024] | 0.2223080184387228 | 1.8565149417781868e-12 | 5.428169025378793e-11 | 5.4539706084710815e-11 | 5.0002890716882575e-11 |
| 30 | none | [4, 1024] | 0.4018140341095929 | 5.151434834260726e-13 | 9.475087381360936e-12 | 4.9817927560980024e-12 | 7.414069358446795e-13 |
| 40 | none | [4, 1024] | 0.26381262931744315 | 6.261657858885883e-14 | 4.920552854059679e-11 | 6.77613520849718e-11 | 6.305334032674637e-11 |

The largest deviation over all records and both forms is 1.5556289589824246e-10, at the integrator's tolerance (rtol 1e-12, with Simpson's rule at h = ell/512). The check includes a trapped record (T/r_s = 18) and a late one (T/r_s = 40).
