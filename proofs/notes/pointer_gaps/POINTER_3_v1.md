# POINTER 3 (07_page_curve:29): the two-field spacing 2.71 l_P (2.55 to 2.86), 2.69 l_P at ma = 0.2, and the temperature ratio one half

Formula (printed, eq. page-share): two-field spacing a = 2 sqrt(eta_h + eta_l) l_P, one-field spacing a = 2 sqrt(eta) l_P.
Inputs, from two independent interval calculations that agree (R7E4_COMPARE_v1.md, md5 6c4e055d; R7E4_ROUTE1_v1.md 769ca57d; R7E4_ROUTE2_v1.md 8b1dde32; App. A.2): eta(0) in [0.8164, 1.0203], eta(0.2) in [0.7883, 0.9831].
- Physical mass (ma -> 0, both fields at eta(0)): a = 2 sqrt(2 eta(0)) l_P. The interval endpoints, rounded outward, are 2 sqrt(2 x 0.8164) = 2.5555, printed as 2.55 (2.56 is not supported, since 2.56^2/8 = 0.8192 > 0.8164), and 2 sqrt(2 x 1.0203) = 2.8567, printed as 2.86. The midpoint value 2 sqrt(2 x 0.91835) = 2.7106 is printed as 2.71.
- Computed mass ma = 0.2 (heavy field at eta(0.2), light field at eta(0)), midpoints: 2 sqrt(0.8857 + 0.91835) = 2.686, printed as 2.69.
- Temperature ratio: at the one-field spacing a = 2 sqrt(eta) l_P, the two-field statistical temperature relative to its matched value is eta/(eta_h + eta_l), which is exactly one half when both fields carry eta(0) (massless limit).
Rounding rules: R7E4_A2_TEXT_v1_1.md (eaea9c5e); agreed by reviewer B (A_r7e4_a2text 18791399, accept d441cb3e) and reviewer A (F_r7e4_a2text_val 8b3b7f31).
