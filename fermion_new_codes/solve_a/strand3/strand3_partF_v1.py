#!/usr/bin/env python
"""Strand 3 addendum part F (spec solve/specs/SPEC_strand3_addendum.md, part F and the lead's 'Part F: the momentum-space kernel with a
momentum-dependent mass'; builder analyst): the vacuum kernel k_a(q) of the four-packet inventory when a packet's polar operator carries
the Wilson-type mass M_kappa(k) = m_s + kappa^-1 sum_i (1 - cos k_i) in place of m_s.

Route (the spec's general route: strand2c_kernel_v1's perturbation theory with the mass matrix elements replaced, carried as explicit
2 x 2 spin blocks): psi = 1 + eps c, c = cos(q x_1), hat c the momentum-shift operator <k'|hat c|k> = (1/2)(delta_{k',k+q} +
delta_{k',k-q}) (one shift of weight 1 when 2q = 0 mod 2 pi); A = A_0 + eps A_1 + eps^2 A_2 with, for the declared placement M psi^-1
(the mass operator to the left of psi^-1),
  <k|A_0|k> = M(k) + i sigma.S(k),   <k'|A_1|k> = -c_k'k [M(k') + 3 i sigma.S(k)],   <k|A_2|k> = (c^2)_kk [M(k) + 6 i sigma.S(k)],
and for the variant psi^-1/2 M psi^-1/2 (the symmetric placement) <k'|A_1|k> = -c_k'k [(M(k) + M(k'))/2 + 3 i sigma.S(k)] with the mass
part of <k|A_2|k> equal to (3/4)(c^2)_kk M(k) + (1/4) sum_j c_kj^2 M(j), the exact second-order diagonal element of
psi^-1/2 M psi^-1/2 = M - (eps/2)(hat c M + M hat c) + eps^2 [(3/8)(hat c^2 M + M hat c^2) + (1/4) hat c M hat c] (at first order this is
the spec's (M(k) + M(k'))/2 rule).  X = A^dag A, X~ = A A^dag: X_0 = (M^2 + S^2) 1 per k, X'_k'k = A_0(k')^dag A_1,k'k + A_1,kk'^dag A_0(k),
X''_kk = 2 [A_0^dag A_2 + A_2^dag A_0 + sum_j A_1,jk^dag A_1,jk], and X~'_k'k = A_0(k') A_1,kk'^dag + A_1,k'k A_0(k)^dag, X~''_kk =
2 [A_0 A_2^dag + A_2 A_0^dag + sum_j A_1,kj A_1,kj^dag]; with F = sqrt, F1(a, b) = 1/(sqrt a + sqrt b), F'(a) = 1/(2 sqrt a),
F'1(a, b) = -1/(2 sqrt a sqrt b (sqrt a + sqrt b)) (the Daleckii-Krein second-order formula of strand2c_kernel_v1):
  T'' = 2 sum_kk' F1 c_k'k tr X'_k'k + sum_k F' tr X''_kk + sum_kk' F'1 ||X'_k'k||^2,   U'' the same with X~,
  W^(2) = (1/2) sum_s eta_s (U_s'' - T_s''),   W^(11) = sum_s eta_s sum_kk' F1 c_k'k tr (X~' - X')_k'k,   k_a(q) = -2 W^(2)/(q_lat^2 L^3),
per packet k_s(q) = -(U_s'' - T_s'')/(q_lat^2 L^3) and mixed_s = -W_s^(11)/(q_lat^2 L^3).  The 2 x 2 blocks are carried as Pauli
components (b_0, b_x, b_y, b_z).  The transverse spin part sigma_2 S_2 + sigma_3 S_3 is rotated about axis 1 to sigma_2 |S_perp| (one
constant unitary per transverse class; spin traces invariant), so the sums run over k_1 and the classes of (sin^2 k_2 + sin^2 k_3,
cos k_2 + cos k_3) with their counts.
Controls (--control): (1) kappa = infinity (M = m) against strand2c_kernel_v1.packet_terms, per packet and q; (2) the dense real-space
W_0 of one packet on the periodic L^3 box (L = 8, 12; twist 0; m = 0.1; q = 2 pi/L), A built as a matrix from M = m + (1/(2 kappa))
(-Delta_lattice) with the same placement, eps in {2e-2, 1e-2, 5e-3} with both signs, D(eps) = (W(eps) + W(-eps) - 2 W(0))/eps^2 extrapolated
twice (R1(e) = (4 D(e/2) - D(e))/3, R2 = (16 R1(e/2) - R1(e))/15; the part A control's procedure on the symmetric difference), against
W^(2) of this route; kappa in {infinity, 0.3, 0.01, 0.002}, both placements.
Settings (main run): the equal rectangles aH = 1 and 0.5 (m_light = 0.1, twist 0) and the physical row of the 2c addendum's part A (v)
(strand 3's lambda = 1 rectangle at a/l_P = 0.0213: light packet massless, heavy 0.0130903, 0.0130903, 0.0185125, twist 0.371 on every
axis); kappa in {0.003, 0.0023, 0.002} and infinity; L = 128; both placements; per packet, so the report forms variant 1 (the light
packet only) and variant 2 (all four packets); q = 2 pi n/L, n = 1..L/2, one pool task per n.  Writes <out>/partF.json (main run) or
<out>/control.json (--control)."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("PARTF_THREADS", "1"))

import argparse
import json
import socket
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "strand2"), HERE):
    sys.path.insert(0, _p)
import strand2c_kernel_v1 as KC  # noqa: E402

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
EPS = (2e-2, 1e-2, 5e-3)
KAPPAS = (0.003, 0.0023, 0.002)
SETTINGS = {
    "equal aH=1": {"theta": 0.0, "masses": KC.masses_eq(0.1, 1.0)[0], "aH": 1.0},
    "equal aH=0.5": {"theta": 0.0, "masses": KC.masses_eq(0.1, 0.5)[0], "aH": 0.5},
    "physical a/l_P 0.0213": {"theta": 0.371, "masses": [0.0, 0.013090347820517666, 0.013090347820517666, 0.01851254742395717],
                              "aH": 0.013090347820517666},
}
ETAS = (1, -1, -1, 1)


# ---- 2 x 2 blocks as Pauli components (b0, bx, by, bz): b0 1 + b . sigma
def pmul(a, b):
    a0, ax, ay, az = a
    b0, bx, by, bz = b
    return (a0 * b0 + ax * bx + ay * by + az * bz,
            a0 * bx + b0 * ax + 1j * (ay * bz - az * by),
            a0 * by + b0 * ay + 1j * (az * bx - ax * bz),
            a0 * bz + b0 * az + 1j * (ax * by - ay * bx))


def pdag(a):
    return tuple(np.conj(x) for x in a)


def padd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def ptr(a):
    return 2 * a[0]


def pfrob(a):
    return 2 * (np.abs(a[0]) ** 2 + np.abs(a[1]) ** 2 + np.abs(a[2]) ** 2 + np.abs(a[3]) ** 2)


def classes(L, theta, wilson=True):
    """Transverse classes of (k_2, k_3) on the L x L grid with twist theta on both axes: (sin^2 k_2 + sin^2 k_3, cos k_2 + cos k_3)."""
    k = KC.grid(L, theta)
    s2 = np.sin(k) ** 2
    c = np.cos(k)
    sp2 = (s2[:, None] + s2[None, :]).ravel()
    cp = (c[:, None] + c[None, :]).ravel()
    keys = np.stack([np.round(sp2, 12), np.round(cp, 12) if wilson else np.zeros_like(cp)], 1)
    _, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = np.asarray(inv).ravel()
    return np.bincount(inv, weights=sp2) / cnt, np.bincount(inv, weights=cp) / cnt, cnt.astype(float)


def packet_terms_M(m, kappa, L, n_q, theta1, cls, placement="left"):
    """(T'', U'', W11 contribution) for one packet with the mass operator M_kappa (kappa = inf: M = m), shift n_q along axis 1."""
    sp2, cp, cnt = cls
    k1 = KC.grid(L, theta1)
    zero = np.zeros((L, sp2.size))
    s1 = np.sin(k1)[:, None] + zero
    c1 = np.cos(k1)[:, None] + zero
    sp = np.sqrt(sp2)[None, :] + zero
    if np.isfinite(kappa):
        M = m + ((1 - c1) + (2 - cp[None, :])) / kappa
    else:
        M = m + zero
    z = zero + 0j
    shifts = [n_q % L, (-n_q) % L]
    nb = [(shifts[0], 1.0)] if shifts[0] == shifts[1] else [(shifts[0], 0.5), (shifts[1], 0.5)]
    c2d = sum(w * w for _, w in nb)                       # (c^2)_kk

    def roll(a, sh):                                       # roll(a, sh)[i] = a[i + sh]
        return np.roll(a, -sh, axis=0)

    A0 = (M + 0j, 1j * s1, 1j * sp, z)
    x = M * M + s1 * s1 + sp * sp
    sq = np.sqrt(x)
    if placement == "left":
        m2 = c2d * M
    elif placement == "sym":
        m2 = 0.75 * c2d * M + 0.25 * sum(w * w * roll(M, sh) for sh, w in nb)
    else:
        raise ValueError(placement)
    A2 = (m2 + 0j, 6j * c2d * s1, 6j * c2d * sp, z)
    trXpp = ptr(padd(pmul(pdag(A0), A2), pmul(pdag(A2), A0))).real
    trXtpp = ptr(padd(pmul(A0, pdag(A2)), pmul(A2, pdag(A0)))).real
    accT = np.zeros_like(x)
    accU = np.zeros_like(x)
    accW = np.zeros_like(x)
    for sh, w in nb:
        Mp, s1p, sqp = roll(M, sh), roll(s1, sh), roll(sq, sh)
        A0p = (Mp + 0j, 1j * s1p, 1j * sp, z)
        if placement == "left":
            mf, mb = Mp, M                                   # <k'|A_1|k> carries M(k'), <k|A_1|k'> carries M(k)
        else:
            mf = mb = 0.5 * (M + Mp)
        Fw = (-w * mf + 0j, -3j * w * s1, -3j * w * sp, z)   # <k'|A_1|k>, k' = k + shift
        Bw = (-w * mb + 0j, -3j * w * s1p, -3j * w * sp, z)  # <k|A_1|k'>
        Xp = padd(pmul(pdag(A0p), Fw), pmul(pdag(Bw), A0))   # X'_k'k
        Xtp = padd(pmul(A0p, pdag(Bw)), pmul(Fw, pdag(A0)))  # X~'_k'k
        F1 = 1.0 / (sq + sqp)
        Fp1 = -1.0 / (2 * sq * sqp * (sq + sqp))
        trXp = ptr(Xp).real
        trXtp = ptr(Xtp).real
        accT += 2 * F1 * w * trXp + Fp1 * pfrob(Xp)
        accU += 2 * F1 * w * trXtp + Fp1 * pfrob(Xtp)
        accW += F1 * w * (trXtp - trXp)
        trXpp = trXpp + ptr(pmul(pdag(Fw), Fw)).real        # (A_1^dag A_1)_kk
        trXtpp = trXtpp + ptr(pmul(Bw, pdag(Bw))).real      # (A_1 A_1^dag)_kk
    Fp = 0.5 / sq
    accT += Fp * 2 * trXpp
    accU += Fp * 2 * trXtpp
    W = cnt[None, :]
    return float(np.sum(W * accT)), float(np.sum(W * accU)), float(np.sum(W * accW))


def kernel_terms(m, kappa, L, n_q, theta1, cls, placement):
    t2, u2, w11 = packet_terms_M(m, kappa, L, n_q, theta1, cls, placement)
    q = 2 * np.pi * n_q / L
    ql2 = 2 * (1 - np.cos(q))
    return {"T2": t2, "U2": u2, "W11": w11, "k": -(u2 - t2) / (ql2 * L ** 3), "mixed": -w11 / (ql2 * L ** 3)}


# ---- dense real-space control
def dense_W2(L, m, kappa, n, placement):
    q = 2 * np.pi * n / L
    N3 = L ** 3
    ar = np.arange(N3)
    pos = np.stack([ar % L, (ar // L) % L, ar // (L * L)], 1)
    Dw = np.zeros((2 * N3, 2 * N3), complex)
    Lap = np.zeros((N3, N3))
    for j, S in enumerate((SX, SY, SZ)):
        for sg in (1, -1):
            p = pos.copy()
            p[:, j] = (p[:, j] + sg) % L
            y = p[:, 0] + L * p[:, 1] + L * L * p[:, 2]
            for a in range(2):
                for b in range(2):
                    Dw[2 * ar + a, 2 * y + b] += sg * S[a, b] / 2j     # symbol sigma_j sin k_j
            Lap[ar, y] += -0.5
    Lap[ar, ar] += 3.0                                                  # symbol sum_j (1 - cos k_j) = (1/2)(-Delta)
    Mop = m * np.eye(N3) + (Lap / kappa if np.isfinite(kappa) else 0.0)
    I2 = np.eye(2)
    W = {}
    for e in [0.0] + [s * e_ for e_ in EPS for s in (1, -1)]:
        psi = 1 + e * np.cos(q * pos[:, 0])
        if placement == "left":
            Mpart = Mop * (1 / psi)[None, :]
        else:
            r = psi ** -0.5
            Mpart = r[:, None] * Mop * r[None, :]
        A = np.kron(Mpart, I2) + 1j * (np.repeat(psi ** -3.0, 2)[:, None] * Dw)
        Lv = np.repeat(psi, 2)
        tr = []
        for X in (A.conj().T @ A, A @ A.conj().T):
            w, V = np.linalg.eigh(X)
            tr.append(float(np.sum(Lv[:, None] * np.abs(V) ** 2 * np.sqrt(np.clip(w, 0, None))[None, :])))
        W[e] = tr[1] - tr[0]
    D = {e_: (W[e_] + W[-e_] - 2 * W[0.0]) / (e_ * e_) for e_ in EPS}
    R1 = [(4 * D[EPS[1]] - D[EPS[0]]) / 3, (4 * D[EPS[2]] - D[EPS[1]]) / 3]
    R2 = (16 * R1[1] - R1[0]) / 15
    return {"W2_R2": R2 / 2, "W2_R1": R1[1] / 2, "W2_D_smallest_eps": D[EPS[2]] / 2, "W_at_0": W[0.0]}


def control(out):
    res = {"packet_terms": [], "dense": []}
    t0 = time.time()
    # (1) kappa = infinity against packet_terms on the settings' masses at L = 128, n = 1, 2, 5, 17, 64
    for sname, S in SETTINGS.items():
        L = 128
        cls = classes(L, S["theta"], True)
        sp, cnt = KC.perp_classes(L, S["theta"], S["theta"])
        for n in (1, 2, 5, 17, 64):
            for m in sorted(set(S["masses"])):
                t2, u2, w11 = KC.packet_terms(m, L, n, (S["theta"],) * 3, sp, cnt)
                for pl in ("left", "sym"):
                    r = packet_terms_M(m, np.inf, L, n, S["theta"], cls, pl)
                    res["packet_terms"].append({"setting": sname, "n": n, "m": m, "placement": pl, "T2_ref": t2, "U2_ref": u2, "W11_ref": w11,
                                                "T2": r[0], "U2": r[1], "W11": r[2],
                                                "rel_W2": abs((r[1] - r[0]) - (u2 - t2)) / abs(u2 - t2),
                                                "rel_W11": abs(r[2] - w11) / max(abs(w11), 1e-300)})
        print("control (1) %s: max rel W2 %.2e, W11 %.2e (%.0f s)" % (
            sname, max(x["rel_W2"] for x in res["packet_terms"] if x["setting"] == sname),
            max(x["rel_W11"] for x in res["packet_terms"] if x["setting"] == sname), time.time() - t0), flush=True)
    # (2) dense real-space W_0 at L = 8, 12
    for L in (8, 12):
        for kappa in (np.inf, 0.3, 0.01, 0.002):
            for pl in (("left",) if not np.isfinite(kappa) else ("left", "sym")):
                d = dense_W2(L, 0.1, kappa, 1, pl)
                cls = classes(L, 0.0, True)
                r = packet_terms_M(0.1, kappa, L, 1, 0.0, cls, pl)
                W2 = 0.5 * (r[1] - r[0])
                row = {"L": L, "m": 0.1, "kappa": kappa if np.isfinite(kappa) else "inf", "placement": pl, "n": 1, "W2_route": W2}
                row.update(d)
                row["rel_R2"] = abs(W2 - d["W2_R2"]) / abs(d["W2_R2"])
                row["rel_R1"] = abs(W2 - d["W2_R1"]) / abs(d["W2_R1"])
                if not np.isfinite(kappa):
                    sp, cnt = KC.perp_classes(L, 0.0, 0.0)
                    t2, u2, _ = KC.packet_terms(0.1, L, 1, (0.0, 0.0, 0.0), sp, cnt)
                    row["W2_packet_terms"] = 0.5 * (u2 - t2)
                res["dense"].append(row)
                print("control (2) L %d kappa %s %s: route %.12e dense R2 %.12e rel %.2e (R1 rel %.2e) (%.0f s)" % (
                    L, row["kappa"], pl, W2, d["W2_R2"], row["rel_R2"], row["rel_R1"], time.time() - t0), flush=True)
                with open(os.path.join(out, "control.json"), "w") as fh:
                    json.dump(res, fh, indent=1)
    res["seconds"] = time.time() - t0
    with open(os.path.join(out, "control.json"), "w") as fh:
        json.dump(res, fh, indent=1)


_G = {}


def winit(L):
    _G["L"] = L
    _G["cls"] = {s: classes(L, S["theta"], True) for s, S in SETTINGS.items()}


def wtask(args):
    sname, n = args
    L = _G["L"]
    S = SETTINGS[sname]
    cls = _G["cls"][sname]
    rows = []
    for m in sorted(set(S["masses"])):
        r = kernel_terms(m, np.inf, L, n, S["theta"], cls, "left")
        r.update({"setting": sname, "n": n, "m": m, "kappa": "inf", "placement": "any"})
        rows.append(r)
        for kappa in KAPPAS:
            for pl in ("left", "sym"):
                r = kernel_terms(m, kappa, L, n, S["theta"], cls, pl)
                r.update({"setting": sname, "n": n, "m": m, "kappa": kappa, "placement": pl})
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.control:
        control(a.out)
        return
    t0 = time.time()
    L = a.L
    res = {"meta": {"host": socket.gethostname(), "L": L, "kappas": list(KAPPAS),
                    "settings": {s: {"theta": S["theta"], "masses": list(S["masses"]), "aH": S["aH"], "etas": list(ETAS)} for s, S in SETTINGS.items()},
                    "start_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, "rows": []}
    tasks = [(s, n) for s in SETTINGS for n in range(1, L // 2 + 1)]
    with Pool(a.workers, initializer=winit, initargs=(L,)) as pool:
        for i, rows in enumerate(pool.imap(wtask, tasks)):
            res["rows"] += rows
            if (i + 1) % 16 == 0 or i + 1 == len(tasks):
                print("%d of %d tasks (%.0f s)" % (i + 1, len(tasks), time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "partF.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("part F done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
