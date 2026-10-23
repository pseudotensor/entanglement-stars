"""Second reader: exact kernel overlaps from the #141 definition and the #145/Table F1 static combinations. Read-only, no inputs."""
import json, sys
import sympy as sp
t, s = sp.symbols("t s", real=True)
q = sp.Rational(3003, 2048) * (1 - s**2) ** 6
# w = q * 1_[-1/2,1/2] (convolution): w(t) = int_{t-1/2}^{t+1/2} q(s) 1_{|s|<=1} ds
def w_piece(lo, hi):  # valid for t where integration limits clip as given
    return sp.integrate(q, (s, lo, hi))
# pieces of w on [-3/2,3/2]
W = sp.Piecewise((w_piece(-1, t + sp.Rational(1, 2)), (t >= -sp.Rational(3, 2)) & (t < -sp.Rational(1, 2))),
                 (w_piece(t - sp.Rational(1, 2), t + sp.Rational(1, 2)), (t >= -sp.Rational(1, 2)) & (t <= sp.Rational(1, 2))),
                 (w_piece(t - sp.Rational(1, 2), 1), (t > sp.Rational(1, 2)) & (t <= sp.Rational(3, 2))), (0, True))
brk = [sp.Rational(k, 2) for k in range(-3, 4)]
def G(r):
    tot = 0
    for a, b in zip(brk[:-1], brk[1:]):
        f = W.subs(t, (a + b) / 2)  # identify piece
        tot += sp.integrate(sp.piecewise_fold(W * W.subs(t, t - r)).rewrite(sp.Piecewise), (t, a, b))
    return sp.nsimplify(sp.simplify(tot))
G0, G1, G2, G3 = [G(r) for r in range(4)]
given = [sp.Rational(388627261303, 549873254400), sp.Rational(456290907571, 3115948441600), sp.Rational(180865117, 983983718400)]
res = {"G": [str(G0), str(G1), str(G2), str(G3)], "equal_given": [G0 == given[0], G1 == given[1], G2 == given[2]],
       "sum": str(G0 + 2 * G1 + 2 * G2)}
G0, G1, G2 = given
combos = {"site0_endpoint_copy": (19 * G0 + 20 * G1 + 3 * G2) / 64, "flat": (9 * G0 + 8 * G1 + G2) / 16,
          "Nm1_eq_1": (5 * G0 + 6 * G1 + G2) / 16, "halo_removed": (9 * G0 + 6 * G1) / 32,
          "dH_dNm1_entry": (G0 + 4 * G1 + G2) / 32, "Schur_n2": (2 * G0 + 9 * G1 + 6 * G2) / 32, "Schur_n3_n7": (2 * G0 + 9 * G1 + 9 * G2) / 32}
res["combos"] = {k: [str(v), float(v)] for k, v in combos.items()}
json.dump(res, open(sys.argv[1], "w"), indent=1); print(json.dumps(res, indent=1))
