# APPD_SPINCURL_RA_busi_v1: Appendix D spin-curl correction, Route A (busi)

busi implementer (builder analyst), inbox #786, 30 Oct 2026.
- Inputs: FableAPI_appdf_v1.md sections 1 and 3; AstraAPI_appda_v1.md (00a66c23); my APPD_RECERT_W_R1_busi_v1 (5ceec68a) and its box dir, with code reused and md5-checked:
  - appd_wA_v3.py 30e45df0;
  - kappaA_v1.npz e1e361cc;
  - s89_libA_v2 3726b5c1;
  - w2cubic_v2 d5e18c09, read-only.
- Frozen manifest: APPD_SPINCURL_RA_manifest_v1.md, md5 fc155673 (sent to main before the runs).
- Blind to route2's Route B: none of its files was opened.
- Box <internal>/appd_recert_w_r1.
  - srS_v1 ran on GPU 0 in 2 s. Its guard sampled once at exec, so the logged 2.9 MB peak RSS is not meaningful.
  - srA_v1 ran on CPU at 3 threads per pool, nlwp 14, peak RSS 5.5 GB, 847 s.
  - GPU 1 was never touched. No API stages, no commits.
- Code: busi/code/appd_recert_w_r1/, byte-identical with the box copy (md5s compared).

## Result table (Route A; same PM1 candidate and ball: b = 0, d = 2, 1028 tests, Gaussian ell = 2, bumps w = 4; reading P)

| item | result |
|---|---|
| (0) check: eq. 5 on v = omega x x, N = 1 (W density) | -0.0344089974131 = -S\|omega\|^2/4 exactly (ratio 1.000000000000001, for omega = e_1, e_2, e_3, (1,1,1)/sqrt3; radius 8e-12) => **the correction IS needed** |
| eq. 5's antisymmetric-gradient sector | three equal eigenvalues 0.0172044987065 = S/8; cross block with the symmetric sector 2.5e-17 (cubic T_1 vs A_1+E+T_2) |
| (1) S, enclosed | **S = 0.137635989652 +- 3.2e-11** (equal for k = 1, 2, 3 to 8e-17); midpoint Np 512 + polystrip bound (sigma 0.1, M 2.09e3) + rounding |
| (2) Route A, fixed-p 2nd order: chi_kl = <tr[P_0 Sig^lat_k Q_0 Sig^lat_l]/(2 eps)> from explicit 4x4 Dirac matrices | chi = S I within the radii (diagonal - S = 0.0; off-diagonal <= 1.1e-19; imaginary part 0). The interband part (b - curl v/4).P_0 Sig^lat Q_0 over the denominator 2 N eps gives -(S/N)\|b - curl v/4\|^2 |
| corrected W_2 used | K_full = K - (S/16) C, i.e. eq. 5 + (S/16)\|curl_D v\|^2/N. It equals sym(K) to 6e-17 (eq. 5 with its curl sector deleted); kappa radius 1.96e-12 -> 3.97e-12 |
| discrete curl of the sampled u_0 (enclosed) | max \|curl_D v_0\| = 6.45e-4 (2.1 percent of max \|D v_0\| = 0.0308); included exactly through K_full's linearization at u_0 |
| A_ref change, \|\|A_ref(full) - A_ref(eq. 5)\|\|_F | 0.012201, against \|\|A_ref(eq. 5)\|\|_F 0.022415 and \|\|A_d\|\|_F 3.301 (flow-momentum block: 0.37 percent of A_d's) |
| ... versus reviewer A's delta r_v = +(S/8) curl(curl dv/N_0) | agrees to 2.75e-7 (Frobenius). The difference is the u_0 curl-residual and r_N terms; energy rows and clock columns change by <= 2.1e-7 |
| anchor \|\|Z\|\|, margin d_0 (unchanged) | 4.36155e-5, >= 5.6384e-5 |
| Gram lambda_min energy / momentum (unchanged) | >= 7.4347e-10 / 3.4212e-11 |
| gap / path gap (unchanged) | >= 0.199216605 / >= 0.12887 |

## Strict mixedness with delta r_v (certificates; beside eq. 5 alone, APPD_RECERT_W_R1 route A)

| quantity | eq. 5 + spin-curl correction (full W_2) | eq. 5 alone | shift |
|---|---|---|---|
| Lambda_ref | 0.010307 | 0.011969 | -0.00166 |
| Lambda_tot (all / clock / flow) | 298.27966 / 21.37629 / 297.51412 | 298.27952 / 21.37629 / 297.51398 | +1.4e-4 (allowances) |
| **sup at \|\|c\|\| = 1.1e-7** = \|\|Z\|\| + 1.1e-7 Lambda_tot + rem | **7.643270e-5** (4.36155e-5 + 3.281076e-5 + 6.40e-9) | 7.643277e-5 (4.36155e-5 + 3.281075e-5 + 6.49e-9) | -7.6e-11 |
| **margin eps - sup at 1.1e-7** | **2.356730e-5 > 0** | 2.356723e-5 | +7.6e-11 |
| **certified radius eta_max (all)** | **>= 1.889689e-7** (1.88e-7) | >= 1.889681e-7 | +8e-13 |
| eta_max clock only / flow only | >= 2.6330e-6 / 1.8952e-7 | 2.6329e-6 / 1.8952e-7 | ~0 |
| reference gap over the ball at 1.1e-7 | >= 0.1992165 | same | - |
| FLOAT: exact C(u) at 1x / 4x eta_max (worst direction, module grads with K_full) | margin 7.649e-5 / 4.321e-5 | 7.649e-5 / 4.321e-5 | - |

**Plainly:**
- The paper's full W_2 differs from eq. 5 on curl-carrying fields, so the correction is real.
- On Appendix D's printed candidate it moves the strict-mixedness sup at 1.1e-7 by -7.6e-11, against a margin of 2.36e-5. reviewer A estimated up to 5e-6; the recomputed shift is about 6e4 times smaller.
- The certified radius stays 1.89e-7, above the printed 1.1e-7.
- The printed Appendix D claims (gap above 0.12, positive Gram, exact matching, strict mixedness to 1.1e-7) hold with the correction on Route A.
- The correction is small because the reference response is a tiny part of the R-weighted variation. Lambda is set by the laws' source A_d (298), and the corrected reference response contributes Lambda_ref = 0.010. reviewer A's per-site ratio of one fifth does not survive the Gram weighting.

## Self red-team
- **R1, scope.** This is Route A only: busi's adapted 8-9 route A on the PM1 candidate, reading P (u_star = u_0, P_star = P(u_0)). This run did not redo route B of APPD_RECERT_W_R1. route2's Route B (the rigid-turn energies and a compact curl bump) is the independent check.
- **R2, what is certified about the closed form.** For a CONSTANT interband perturbation, chi = S I and the -(S/N)|b - curl v/4|^2 coefficient are verified by explicit matrices. Check (0) verifies that eq. 5's curl sector is exactly S/16 |curl v|^2. That eq. 5's curl sector plus the spin block make the full W_2's rotation sector (reviewer A's identity [h_0, L_Q + Sig^lat/2] = 0) is taken from reviewer A. With check (0) it fixes K_full = sym(K); I did not compute the cross term b.(curl v) independently.
- **R3, discretization.** The correction is discretized as (S/16)|curl_D v|^2/N with the module's D_a (centred; one-sided at the surface) and differentiated exactly. reviewer A's continuum delta r_v agrees to 2.8e-7 in Frobenius norm. Surface terms of eq. 5 remain uncertified (reviewer B), as before.
- **R4, reading F.** With the correction it still fails (||Z|| = 8.09e-3 and 2.81e-4 against eps = 1e-4), as in APPD_RECERT_W_R1. The anchor reading remains the lead's call.
- **R5, assumptions.** IEEE rounding, and libm within 1 ulp inside the S, chi and kappa quadratures.
- **R6, process.**
  - One own job at a time.
  - Busi job count logged under the fill exceptions (#781/#783/#784).
  - Owner rule #789 (no shell globs) followed; no glob was used.
  - srS's guard logged a 2.9 MB peak because the job finished in 2 s, before the first resident sample.

## Cost
srS 2 s (GPU 0). srA 847 s (CPU, 3 threads per pool), including about 31 s of finite-amplitude rows.

## Box MD5SUMS
<internal>/MD5SUMS_SR, md5 of the file 2a134e52b612c9ced2ae5720c3597684, copied to the _files folder as box_MD5SUMS_SR.txt. Main entries:
```
fc155673335b0a99f6dad81b72dca2be  code/APPD_SPINCURL_RA_manifest_v1.md
9761d1379aaf3b150f4158c570cabf58  code/srS_v1.py
5bb40c12877f87c2d9c43545efdcc71b  code/srA_v1.py
799af1ec52ea435f7099236c1370a69f  code/chain_sr_v1.sh
5fd6a6677c60f4eb8d9112de597981ac  code/run_stage_v2.sh
7dd684c63d42566cb920711965b7b4c1  out/srS_v1.json
a5894f33c97b84f16196506f401707f9  out/srS_v1.npz
f7bfe06a249376cea6f5faa6953bf381  out/srA_v1.json
100fb5a05b368a67b5ab2fabee274282  out/srA_v1.npz
```
No box job of mine is running (pidfiles checked). None of my processes is on GPU 0; at 17:30Z it held two other processes (2230384, 971744), not mine.
