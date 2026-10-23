#!/usr/bin/env python
"""Strand 3 A.5 assembly (spec sec. 3; builder analyst): per scan row (a/l_P, lambda), the four q_s, eta_100(q_s) from the
A.1 runs at the exact q_s (both ladders: (16,12) and (32,24)), eta_tot^100 = sum_s eta_100(q_s), eta_req = (a/l_P)^2/4,
R^100 = eta_tot^100/eta_req, the per-packet shares, R^sphere where A.3 covers all four q_s, the allowed flags, and the
light-packet-only R_1 at a/l_P = 1.909; the second column R^static = R/16 (spec sec. 3, cut counted at the
static-chart proper spacing a_H = psi_H^2 a = 4a).  Writes <out>/scan.json.
"""
import argparse
import glob
import json
import os


def key(q):
    return "%.12g" % q


def load_planar(out):
    d = {}
    for f in glob.glob(os.path.join(out, "planar", "planar_*.json")):
        r = json.load(open(f))
        p = tuple(r["normal"])
        d[(p, int(r["depth"]), int(r["nk"]), key(r["mass"]))] = r
    return d


def load_sphere(out):
    d = {}
    for f in glob.glob(os.path.join(out, "sphere", "sphere_q*.json")):
        r = json.load(open(f))
        d[key(r["mass"])] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masses", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    M = json.load(open(args.masses))
    P = load_planar(args.out)
    S = load_sphere(args.out)
    rows = []
    for r in M["rows"]:
        row = dict(r)
        for lad, (d, nk) in (("16_12", (16, 12)), ("32_24", (32, 24))):
            etas = []
            for q in r["q"]:
                rec = P.get(((1, 0, 0), d, nk, key(q)))
                etas.append(None if rec is None else rec["eta"])
            row["eta100_" + lad] = etas
            if all(e is not None for e in etas):
                tot = sum(etas)
                row["eta_tot100_" + lad] = tot
                row["R100_" + lad] = tot / r["eta_req"]
                row["R100_static_" + lad] = tot / r["eta_req"] / 16.0  # spec sec. 3 second column: a_H = 4a, R^static = R/16
                row["shares_" + lad] = [e / tot for e in etas]
        sph = [S.get(key(q)) for q in r["q"]]
        if all(s is not None for s in sph):
            tot = sum(s["eta_sphere"] for s in sph)
            row["eta_sphere"] = [s["eta_sphere"] for s in sph]
            row["eta_tot_sphere"] = tot
            row["R_sphere"] = tot / r["eta_req"]
            row["R_sphere_static"] = tot / r["eta_req"] / 16.0
        rows.append(row)
    req = 1.909 ** 2 / 4
    light = {"eta_req_1.909": req}
    s0 = S.get(key(0.0))
    if s0:
        light["R1_sphere"] = s0["eta_sphere"] / req
    for lad, (d, nk) in (("16_12", (16, 12)), ("32_24", (32, 24))):
        rec = P.get(((1, 0, 0), d, nk, key(0.0)))
        if rec:
            light["R1_100_" + lad] = rec["eta"] / req
    with open(os.path.join(args.out, "scan.json"), "w") as fh:
        json.dump({"rows": rows, "light_packet_only": light}, fh, indent=1)
    print("scan rows %d; with R100_32_24: %d; with R_sphere: %d" % (
        len(rows), sum(1 for r in rows if "R100_32_24" in r), sum(1 for r in rows if "R_sphere" in r)))


if __name__ == "__main__":
    main()
