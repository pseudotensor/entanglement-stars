# POINTER_9_v1: the windowed charge identity (06_scaling:260)

Assembled from existing results; no new analysis. Paper text cited from the flattened tex `marks_m18_23_35/todo35b/P3_current_nc.tex` (md5 d4451765f830fc5c818e2e4505e8232f).

## Statement

1. **Linearized windowed mass** (P3 line 504): N(r) δM(r) = Tr(h̄_r δC) − ⟨ū_r, δr⟩. Here ū_r are the fields cut off outside r before back-averaging, and h̄_r = h[ū_r].
2. **Per-increment split** (P3 lines 1504-1515). Between consecutive solved configurations j, j+1:
   N_{r,j} δ_j M(r) = Σ_a g_a Tr(h_{a,r,j} δ_j C_a) − ⟨ū_{<r,j}, δ_j r_tot⟩ + e_{r,j},
   with δ_j X = X_{j+1} − X_j, g_a the copy weight, h_{a,r,j} the windowed generator of species a, r_tot the copy-weighted reference, and e_{r,j} the remainder (increment remainder, source and solve errors). In the differentiable limit the first term is the **state term** Ξ_r and the second the **reference term** Ξ_r^ref. At a simple fine-grid trapping radius, N_R β_R Ė_R = Ξ_R − Ξ_R^ref, with E_R = R/(2G) and β_R = 1 − 2G M'_R > 0.

## Derivation (in reviewer results)

- `marks_m18_23_35/todo35b/fable_t35bF_result.md` (md5 19c1277bad1d8171fe60a646858b6cf9):
  - sec. 1.1: Euler's identity for the degree-one reference makes the windowed reference rate an exterior pairing. Only Ξ_r − Ξ_r^ref = N_r ∂_T M(r) ("the windowed charge identity, exact on a differentiable regular inward branch") can be bounded.
  - sec. 1.4: the remainder.
  - sec. 1.8: the state and reference terms are additive by species.
- Independent re-derivation: `marks_m18_23_35/todo35b/round3/fable_t35red_result.md` (md5 ee5fcf3286762d7a3a4abccc37efd57c), part (1) "Derivation". It re-derives the fixed-r identity N_r ∂_T M(r, T) = Ξ_r − Ξ_r^ref, with Ξ_r = Σ_a g_a Tr(h[ū_{<r}] Ċ_a) and Ξ_r^ref = ⟨ū_{<r}, ṙ_tot⟩, from its base ∂_r(N δM) = 4π r² (N δρ_ℓ + v δJ_ℓr). It reports that this base holds exactly on solutions of both rows: the N' δM term cancels by the momentum row. The task is `round3/TASK_fable_redteam_package.md` (md5 f730772dab814be574e4c708824d84cd), item (1).
