#!/usr/bin/env python
"""Strand 2d sec. 8(2): the dressed fixed-vacuum cost J^prep,vac = 2 <Z_1^dag Omega| H_exc|_(1,1) |Z_1^dag Omega> with the completion, on
C8 (validation against the gate's q-block route, strand2d_prep31_dressed_v2) and B4 (16384-dimensional (1,1) block; builder analyst).

(1,1) states |p, h> = q_p^dag q_(nW+h)^dag |Omega> (particle p, hole h; the q-block tuple (p, nW + h)), index p nW + h.  The completion on
(1,1): S[(p',h'), (p,h)] = v_IJ/(eps_p' + eps_h' + eps_p + eps_h), v_IJ = 4 W_A[p', nW + h', p, nW + h] of strand2a_lib.two_two restricted to
the four index-type combinations whose creation pair and annihilation pair are each one particle and one hole; the needed blocks of the
transformed seed T'_{ijkl} = sum_t w_t m1_t[i, k] m2_t[j, l] (m_t = Q^dag M_t Q, the seed grouped as strand2d_lib_v1.site_terms) are built
by one matrix product over the term index each, and two_two's sign and ordering rules are applied combination by combination.
H_exc|_(1,1) at the flat point is e^S H_0 e^S with H_0 = eps_p + eps_h (the dressed annihilators map to the one-excitation blocks, where S
vanishes); u = e^S w by its Taylor series (S Hermitian and small), w = z (z_ph = <u_p| dP_+/dlam |v_h>, central differences h = 1e-4, as
stage 1); E11 = sum (eps_p + eps_h) |u_ph|^2, free with u = w.  Writes <out>/vac11_<lattice>_gM<gM>_gB<gB>.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse
import itertools
import json
import socket
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "frozen"))
import strand2a_lib as S2  # noqa: E402
import strand2d_lib_v1 as L2D  # noqa: E402

CT = torch.complex128


def v_ph(lat, light, psi, gM, gB, Lam, dev):
    """v[(p', h'), (p, h)] (nW^2 x nW^2) for the ph class."""
    nW = light.nW
    Q = torch.as_tensor(light.Q, dtype=CT, device=dev)
    terms = L2D.site_terms(lat, psi, gM, gB, Lam)
    m1s, m2s, ws = [], [], []
    for w, M1, M2 in terms:
        m1s.append(Q.conj().T @ torch.as_tensor(M1, dtype=CT, device=dev) @ Q)
        m2s.append(Q.conj().T @ torch.as_tensor(M2, dtype=CT, device=dev) @ Q)
        ws.append(w)
    rng = {0: slice(0, nW), 1: slice(nW, 2 * nW)}
    v = torch.zeros((nW, nW, nW, nW), dtype=CT, device=dev)       # [p', h', p, h]
    for ti, tj, tk, tl in itertools.product((0, 1), repeat=4):
        types = ["C" if ti == 0 else "A", "C" if tj == 0 else "A", "A" if tl == 0 else "C", "A" if tk == 0 else "C"]
        if types.count("C") != 2:
            continue
        labels = ["i", "j", "l", "k"]
        tmap = {"i": ti, "j": tj, "k": tk, "l": tl}
        cpos = [p for p in range(4) if types[p] == "C"]
        apos = [p for p in range(4) if types[p] == "A"]
        c1, c2 = labels[cpos[0]], labels[cpos[1]]
        a1p, a2p = labels[apos[0]], labels[apos[1]]
        tgt = (tmap[c1], tmap[c2], tmap[a2p], tmap[a1p])
        if sorted(tgt[:2]) != [0, 1] or sorted(tgt[2:]) != [0, 1]:
            continue
        permn = cpos + apos
        inv = sum(1 for x in range(4) for y in range(x + 1, 4) if permn[x] > permn[y])
        sign = -1.0 if inv % 2 else 1.0
        # T'_sub[i, j, k, l] over the type ranges (ti, tj, tk, tl)
        A = torch.stack([(w * m1[rng[ti], rng[tk]]).reshape(-1) for w, m1 in zip(ws, m1s)])
        Bm = torch.stack([m2[rng[tj], rng[tl]].reshape(-1) for m2 in m2s])
        K = A.T @ Bm
        Tsub = K.reshape(nW, nW, nW, nW).permute(0, 2, 1, 3)        # [i, j, k, l]
        blk = torch.einsum("ijkl->" + c1 + c2 + a2p + a1p, Tsub) * sign   # W[c1, c2, a2p, a1p] block
        del A, Bm, K, Tsub
        # v = W[p,h,p',h'] - W[h,p,p',h'] - W[p,h,h',p'] + W[h,p,h',p'] on the creation pair (indices 0, 1) and annihilation pair (2, 3)
        if tgt == (0, 1, 0, 1):
            v += blk
        elif tgt == (1, 0, 0, 1):
            v -= blk.permute(1, 0, 2, 3)
        elif tgt == (0, 1, 1, 0):
            v -= blk.permute(0, 1, 3, 2)
        elif tgt == (1, 0, 1, 0):
            v += blk.permute(1, 0, 3, 2)
        del blk
    return v.reshape(nW * nW, nW * nW)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lattice", required=True, choices=["C8", "B4"])
    ap.add_argument("--m", type=float, default=0.1)
    ap.add_argument("--gM", type=float, required=True)
    ap.add_argument("--gB", type=float, required=True)
    ap.add_argument("--Lam", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    dev = torch.device(a.device)
    lat = L2D.lattice(a.lattice)
    n = lat["nsite"]
    nW = n * lat["ncomp"]
    one = np.ones(n)
    light = S2.Light(lat, a.m, one, one)
    eps = torch.as_tensor(light.eps, dtype=torch.float64, device=dev)
    eph = (eps[:, None] + eps[None, :]).reshape(-1)                  # eps_p + eps_h, index p nW + h
    v = v_ph(lat, light, one, a.gM, a.gB, a.Lam, dev) if (a.gM or a.gB) else None
    res_herm = None
    if v is not None:
        S = v / (eph[:, None] + eph[None, :]).to(CT)
        res_herm = float(torch.max(torch.abs(S - S.conj().T)))
        del v
    print("S built (%.0f s)" % (time.time() - t0), flush=True)
    qs = [0.0, np.pi] if a.lattice == "C8" else [0.0, np.pi / 2, np.pi]
    res = {"meta": {"host": socket.gethostname(), "lattice": a.lattice, "gM": a.gM, "gB": a.gB, "Lambda": a.Lam, "m": a.m, "device": a.device,
                    "S_hermiticity": res_herm}, "rows": []}
    x1 = lat["coord"][:, 0].astype(float)
    for q in qs:
        c = one if q == 0 else np.cos(q * x1)
        hh = 1e-4
        Pp = S2.Light(lat, a.m, one + hh * c, one).vp
        Pm = S2.Light(lat, a.m, one - hh * c, one).vp
        dP = (Pp @ Pp.conj().T - Pm @ Pm.conj().T) / (2 * hh)
        z = light.vp.conj().T @ dP @ light.vm
        w = torch.as_tensor(z.reshape(-1), dtype=CT, device=dev)
        E11_free = float(torch.sum(eph * torch.abs(w) ** 2))
        row = {"q": q, "E11_free": E11_free, "J_prep_vac_free": 2 * E11_free}
        if a.gM or a.gB:
            u = w.clone()
            term = w.clone()
            for nn in range(1, 60):
                term = (S @ term) / nn
                u = u + term
                if float(torch.linalg.norm(term)) < 1e-16 * float(torch.linalg.norm(u)):
                    break
            E11 = float(torch.sum(eph * torch.abs(u) ** 2))
            row.update({"E11_dressed": E11, "J_prep_vac_dressed": 2 * E11, "ratio_dressed_over_free": E11 / E11_free, "taylor_terms": nn})
        res["rows"].append(row)
        print(json.dumps(row), "(%.0f s)" % (time.time() - t0), flush=True)
    res["meta"]["seconds"] = time.time() - t0
    with open(os.path.join(a.out, "vac11_%s_gM%g_gB%g.json" % (a.lattice, a.gM, a.gB)), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
