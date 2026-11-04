# MARK3 route 2 (route2): eq. (2) analogue for the radial chain kernel (lead #592 (b)). METHOD NOTE v1, frozen 2026-11-01T05:05Z (true UTC), BEFORE any number. mark3_r2.

CONTEXT: reviewer B singsqrt (2), with the scan definitions of MARK3_DELTAREC_METHOD_webgpt_v1.md (23ecffc2). Under (H) with margin delta, the clock obeys |ln N(0)| <= C, where
  C = 4 pi G (2 G m delta)^(-1/2) sum_n sqrt(Q2_n) (1/4 pi) int_0^inf y^(-3/2) sqrt(Phi_n(y)) dy.
- Here b_n = W1[:,n]/V_n is the implemented vector kernel (a K1(y, r_n; sigma)/V_n), Phi_n(y) = int_0^y 4 pi s^2 b_n, and Q2_n = sum over chains covering n of ff_c q*_c^2.

THE ANALOGUE: use Phi_n(y) <= min(Phi_n(inf), A_n y^3), with A_n = (4 pi/3) kappa*_n and kappa*_n = sup_y b_n(y). Exactly:
  int_0^inf y^(-3/2) sqrt(min(Phi_inf, A y^3)) dy = 3 A^(1/6) Phi_inf^(1/3).
Hence
  C_eq2 = 4 pi G (2 G m delta)^(-1/2) (3/(4 pi)) sum_n sqrt(Q2_n) A_n^(1/6) Phi_n(inf)^(1/3) >= C_site.

CLOSED FORM, for r_n = n a >> sigma:
- sup_y K1(y, r_n) -> 1/(sqrt(2 pi) sigma), so kappa*_n ~ 1/(4 pi sqrt(2 pi) sigma r_n^2), giving A_n ~ 1/(3 sqrt(2 pi) sigma r_n^2).
- Phi_n(inf) -> 1.
- sum_{n<=K} n^(-1/3) ~ (3/2) K^(2/3).
- With Q2_H = sum_heavy ff_c q*_c^2 on n < ns_h and Q2_L = ff_L q*_L^2 on all n < ns_l:
  C_eq2 ~ (3 G/sqrt(2 G m delta)) (3 sqrt(2 pi) sigma/a)^(-1/6) (1/a) (3/2) [ sqrt(Q2_H+Q2_L) ns_h^(2/3) + sqrt(Q2_L) (ns_l^(2/3) - ns_h^(2/3)) ] (a^(1/3) factors collected).
- The script verifies this asymptotic form against the exact site sum, which uses the true kappa*_n and Phi_n(inf) of the implemented kernel, including small n, where r_n is not much larger than sigma.

NUMBERS TO REPORT:
- Per kernel/weight set (twins ON, twins OFF, C_M1nl23): kappa*_n and Phi_n(inf) ranges.
- The exact C_eq2 coefficient K_eq2 = C_eq2 sqrt(m delta), the closed-form value and its relative error.
- C_eq2 at each group's worst Lambda = m delta_lo from deltarec2 (ON 9.0703, OFF 8.5158, C_M1nl23 8.8592), with exp(-C_eq2).
- Compared with C_site and C_direct.
- Box CPU, seconds. Script singq_deltarec_eq2_v1.py; md5 with the results. DIAGNOSTIC.
