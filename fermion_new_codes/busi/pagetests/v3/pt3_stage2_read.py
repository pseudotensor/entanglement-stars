#!/usr/bin/env python
"""pt3_stage2_read.py (busi, PAGE v3.1 stage 2 reader): the held audit's output algebra from the stored G = X^dag X (the output modes'
anticommutator matrix at T = 0) and Cf = X^dag f_ws(h(0)) X at the ws ladder.

Nested modes in coefficient space: the bins of one family, ordered by arrival at the collar (u ascending; with both bands, the two bands' bins
at equal u enter together as one bin pair), are orthonormalized in that order by the Cholesky factor of G (sequential Gram-Schmidt: a later
bin never changes an earlier mode); near-null directions (pivot below 1e-10 of the column norm) are dropped and counted. C_out = L^-1 Cf L^-dag.
eps (the entropy intervals' eigenvalue error): ||C_out(ws/4) - C_out(ws/2)||_2 (the ladder's geometric majorant: the step error at ws/4 is at
most the ws/2 -> ws/4 change when the ladder converges geometrically with ratio <= 1/2; the ratio is printed) plus the Gram's norm defect.
Printed per family and band set: S(R_n), S(b_n), I(b_n:R_{n-1}), S(b_n|R_{n-1}) with certified intervals; the purification reader
(pt3_readers.purification); the start-(a) tie (the same at ws); the inventory columns' occupations and entropy at the last readout clock."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pt3_core as K  # noqa: E402
import pt3_readers as RD  # noqa: E402


def nested_from_gram(G, Cs, groups, drop=1e-10):
    """groups: list of index lists in arrival order. Returns the orthonormal coefficient matrix A (columns over the kept modes, X A orthonormal),
    the group sizes, and C_out for each Cf in Cs."""
    idx = [i for g in groups for i in g]
    Gs = G[np.ix_(idx, idx)]
    Gs = 0.5 * (Gs + Gs.conj().T)
    n = len(idx)
    # pivoted-free Cholesky in the given order with dropping: sequential Gram-Schmidt on the Gram
    A = np.zeros((n, 0), complex)
    sizes = []
    pos = 0
    dropped = 0
    for g in groups:
        kept = 0
        for _ in g:
            e = np.zeros(n, complex); e[pos] = 1.0
            v = e - A @ (A.conj().T @ (Gs @ e))
            v = v - A @ (A.conj().T @ (Gs @ v))
            nr2 = float(np.real(np.vdot(v, Gs @ v)))
            if nr2 > drop * float(np.real(Gs[pos, pos])):
                A = np.concatenate([A, (v / np.sqrt(nr2))[:, None]], axis=1)
                kept += 1
            else:
                dropped += 1
            pos += 1
        sizes.append(kept)
    ortho = float(np.max(np.abs(A.conj().T @ Gs @ A - np.eye(A.shape[1]))))
    outs = []
    for C in Cs:
        Cc = C[np.ix_(idx, idx)]
        outs.append(A.conj().T @ Cc @ A)
    return A, sizes, outs, {"dropped": dropped, "orthonormality_defect": ortho}


def entropies(Cout, sizes, eps):
    edges = np.cumsum([0] + sizes)
    rows, prev = [], {"S": 0.0, "S_lo": 0.0, "S_hi": 0.0}
    for k in range(len(sizes)):
        if sizes[k] == 0:
            continue
        R = K.cov_entropy(Cout[: edges[k + 1], : edges[k + 1]], eps)
        b = K.cov_entropy(Cout[edges[k]: edges[k + 1], edges[k]: edges[k + 1]], eps)
        rows.append({"n": k + 1, "S_R": R["S"], "S_R_lo": R["S_lo"], "S_R_hi": R["S_hi"], "S_b": b["S"], "I_b_R": b["S"] + prev["S"] - R["S"],
                     "S_b_given_R": R["S"] - prev["S"], "S_b_given_R_lo": R["S_lo"] - prev["S_hi"], "S_b_given_R_hi": R["S_hi"] - prev["S_lo"],
                     "valid": bool(R["valid"] and b["valid"]), "excursion": max(R["excursion"], b["excursion"]),
                     "occ_b": float(np.real(np.trace(Cout[edges[k]: edges[k + 1], edges[k]: edges[k + 1]])))})
        prev = R
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    a = ap.parse_args()
    J = json.load(open(a.stem + ".json"))
    Z = np.load(a.stem + "_blocks.npz")
    G, C1, C2, C4 = Z["G"], Z["Cf_ws"], Z["Cf_ws2"], Z["Cf_ws4"]
    cols = J["columns"]
    out = {"tag": J["tag"], "norm_defect_max": J["norm_defect_max"], "certificate": J["certificate"], "monitor": J["monitor"], "families": {}}
    fams = sorted({c["family"] for c in cols if c["kind"] == "bin"})
    for fam in fams:
        for bandset in ((1,), (-1,), (1, -1)):
            sel = [i for i, c in enumerate(cols) if c["kind"] == "bin" and c["family"] == fam and c["band"] in bandset]
            us = sorted({cols[i]["u_rs"] for i in sel})
            groups = [[i for i in sel if cols[i]["u_rs"] == u] for u in us]
            A, sizes, (o1, o2, o4), info = nested_from_gram(G, (C1, C2, C4), groups)
            d42 = float(np.linalg.norm(o4 - o2, 2)); d21 = float(np.linalg.norm(o2 - o1, 2))
            eps = d42 + float(J["norm_defect_max"])
            rows = entropies(o4, sizes, eps)
            rows_a = entropies(o1, sizes, eps + d21)
            floor = max((r["S_R_hi"] - r["S_R_lo"]) for r in rows) / 2 if rows else 0.0
            verdict = RD.purification(rows, floor=floor)
            key = "family%d_band%s" % (fam, "".join("+" if b > 0 else "-" for b in bandset))
            out["families"][key] = {"u_rs": us, "sizes": sizes, "nested": info, "ladder": {"ws2_to_ws4": d42, "ws_to_ws2": d21,
                                    "ratio": (d42 / d21) if d21 > 0 else None}, "eps": eps, "rows_b": rows, "rows_a_tie": [{"n": r["n"], "S_R": r["S_R"], "occ_b": r["occ_b"]} for r in rows_a],
                                    "S_R_peak": max(r["S_R"] for r in rows) if rows else None, "S_R_last": rows[-1]["S_R"] if rows else None,
                                    "floor_half_width": floor, "purification": verdict}
            K.log("%s: %d bins, eps %.2e (ladder ratio %s), S(R) peak %.4f last %.4f; %s: %s" % (
                key, len(rows), eps, out["families"][key]["ladder"]["ratio"], out["families"][key]["S_R_peak"], out["families"][key]["S_R_last"],
                verdict["verdict"], verdict["why"]))
    inv = [i for i, c in enumerate(cols) if c["kind"] == "inventory"]
    if inv:
        A, sizes, (o4,), info = nested_from_gram(G, (C4,), [inv])
        ent = K.cov_entropy(o4, float(np.linalg.norm(o4 - nested_from_gram(G, (C2,), [inv])[2][0], 2)))
        out["inventory"] = {"e_over_TH": [cols[i]["e_over_TH"] for i in inv], "occupations": [float(np.real(C4[i, i] / G[i, i])) for i in inv],
                            "entropy": ent, "nested": info}
        K.log("inventory at the last readout: e/T_H %s occupations %s entropy %.3e [%.3e, %.3e]" % (
            np.round(out["inventory"]["e_over_TH"], 3).tolist(), np.round(out["inventory"]["occupations"], 6).tolist(), ent["S"], ent["S_lo"], ent["S_hi"]))
    K.dump_json(out, a.stem + "_read.json")
    K.log("wrote %s_read.json" % a.stem)


if __name__ == "__main__":
    main()
