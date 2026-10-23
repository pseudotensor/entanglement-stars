#!/usr/bin/env python3
"""obs_ball_cut_estimate_v1.py (builder analyst; S6 = J4): memory and time estimates per (R, box) for the large radii of the ball-cut
ladder, from obs_ball_cut_v1.py's exact counts (counts mode) and its measured box-CPU timings (run records, bench).  CPU only, seconds.

Memory (exact block sizes): host = all ten irrep blocks of the largest offset (FP64) + the kernel tables (six (2 rmax + 1)^3 FP64 arrays per
box) + the chunk buffers (four 2.5e7-element FP64 arrays); GPU = the largest block K (m_e m_o) + its smaller Gram matrix (min(m)^2) + one
more min(m)^2 FP64 array for the eigensolver's workspace and copies (an allowance, not measured).
Time: per block t = a m^3 (m = min(m_e, m_o)), a fitted on the measured box-CPU blocks with m >= --mfit for each route (svd = gesdd values,
gram = Gram + eigvalsh); per run the assembly t = b n_pairs (n_pairs = even orbits x odd orbits) fitted on the measured runs with R >= 24;
GPU times scale the CPU block times by the ratio of a complex128 eigh at n = 4096 (vectors) on that GPU (--gpu-c4096, the lane's logged
timings, inputs) to the same eigh on the box CPU (bench) - an indirect calibration (different routine), to be replaced by one timed block.
Args: --counts --runs --bench --gpu-c4096 NAME:SECONDS ... --radii --box-ratios --out-json --md."""
import argparse
import json
import math

import numpy as np

IRREPS = ("A1+", "A2+", "E+", "T1+", "T2+", "A1-", "A2-", "E-", "T1-", "T2-")
DIM = {"A1+": 1, "A2+": 1, "E+": 2, "T1+": 3, "T2+": 3, "A1-": 1, "A2-": 1, "E-": 2, "T1-": 3, "T2-": 3}


def load_jsonl(files):
    out = []
    for f in files:
        with open(f) as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--gpu-c4096", nargs="+", default=[])
    ap.add_argument("--radii", type=float, nargs="+", default=[40, 48, 56, 62, 64])
    ap.add_argument("--box-ratios", type=float, nargs="+", default=[4, 6, 8])
    ap.add_argument("--mfit", type=int, default=1500)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    counts = json.load(open(a.counts))["rows"]
    runs = load_jsonl(a.runs)
    bench = json.load(open(a.bench))
    # per-block cubic coefficients
    pts = {"svd": [], "gram": []}
    for r in runs:
        for b in r["blocks"].values():
            m = min(b["m_e"], b["m_o"])
            if m < a.mfit:
                continue
            route = "gram" if b["route"].startswith("gram") else "svd"
            pts[route].append((m, b["t_solve"]))
            if "t_gram" in b:
                pts["gram"].append((m, b["t_gram"]))
    coef = {}
    for k, v in pts.items():
        if v:
            m = np.array([p[0] for p in v], float)
            t = np.array([p[1] for p in v], float)
            coef[k] = dict(a=float(np.sum(t * m ** 3) / np.sum(m ** 6)), n=len(v), m_range=[int(m.min()), int(m.max())],
                           rel_resid_max=float(np.abs(t / (np.sum(t * m ** 3) / np.sum(m ** 6) * m ** 3) - 1).max()))
    # assembly per orbit pair
    ap_pts = []
    nmap = {(float(c["R"]), int(c["delta16"])): c for c in counts}
    for r in runs:
        if float(r["R"]) < 24:
            continue
        c = nmap.get((float(r["R"]), int(r["delta16"])))
        if c is None:
            continue
        ap_pts.append((c["n_orbits"][0] * c["n_orbits"][1], r["t_assemble"] + r["t_frob"]))
    P = np.array([p[0] for p in ap_pts], float)
    T = np.array([p[1] for p in ap_pts], float)
    basm = float(np.sum(T * P) / np.sum(P * P)) if len(P) else float("nan")
    cpu_c4096 = bench.get("complex128_eigh_n4096_vectors")
    gpus = {}
    for g in a.gpu_c4096:
        nm, sec = g.split(":")
        gpus[nm] = dict(c4096=float(sec), ratio=float(sec) / cpu_c4096 if cpu_c4096 else float("nan"))
    route = "gram" if "gram" in coef else "svd"
    out = dict(script="obs_ball_cut_estimate_v1.py", block_coef=coef, assembly_per_pair=basm, assembly_points=len(ap_pts),
               cpu_complex128_eigh_n4096=cpu_c4096, gpus=gpus, route_for_gpu=route, rows=[])
    md = ["## S6 (J4) large-radius resources (obs_ball_cut_estimate_v1.py)", "",
          "Block solve model t = a m^3 (m = min(m_e, m_o)), fitted on the box-CPU blocks (12 threads) with m >= %d: %s." % (
              a.mfit, "; ".join("%s a = %.3e s (n %d, m %d to %d, largest relative residual %.2f)" % (k, v["a"], v["n"], v["m_range"][0], v["m_range"][1],
                                                                                                  v["rel_resid_max"]) for k, v in coef.items())),
          "Assembly (numpy on the host CPU, single gather stream + 12-thread matmul): %.3e s per even-odd orbit pair (%d runs with R >= 24)." % (
              basm, len(ap_pts)),
          "GPU scaling: complex128 eigh (vectors) n 4096 on the box CPU %.2f s (bench); %s; the GPU block times below are the CPU times "
          "of the %s route times these ratios." % (cpu_c4096 if cpu_c4096 else float("nan"),
                                                     "; ".join("%s %.2f s (ratio %.3f, the lane's logged timing, an input)" % (k, v["c4096"], v["ratio"]) for k, v in gpus.items()),
                                                     route), "",
          "| R | box L | n (offsets) | largest block | host RAM GB | GPU GB | assembly, box CPU, 8 offsets (min) | solve, box CPU 12 thr (min) | %s |" % (
              " | ".join("solve, %s (min)" % k for k in gpus)), "|---|---|---|---|---|---|---|---|%s" % ("---|" * len(gpus))]
    for R in a.radii:
        rows = [c for c in counts if float(c["R"]) == float(R)]
        if len(rows) < 8:
            md.append("| %g | (counts missing) |" % R)
            continue
        big = max(rows, key=lambda c: c["n"])
        rmax = 2 * int(math.ceil(R + 7 / 16.0)) + 2
        W = 2 * rmax + 1
        lg = big["largest"]
        mmin = min(lg[1], lg[2])
        host = big["all_blocks_GB"] + 6 * W ** 3 * 8 / 1e9 * len(a.box_ratios) + 4 * 2.5e7 * 8 / 1e9
        gpu = (lg[1] * lg[2] + 2 * mmin ** 2) * 8 / 1e9
        asm = sum(basm * c["n_orbits"][0] * c["n_orbits"][1] for c in rows) / 60
        cpu_solve = sum(sum(coef[route]["a"] * min(c["m"][nm]) ** 3 for nm in IRREPS) for c in rows) / 60
        for cr in a.box_ratios:
            L = int(round(cr * R))
            L += L % 2
            row = dict(R=R, L=L, n=[min(c["n"] for c in rows), max(c["n"] for c in rows)], largest=lg, host_GB=host, gpu_GB=gpu,
                       assembly_min=asm, solve_cpu_min=cpu_solve, solve_gpu_min={k: cpu_solve * v["ratio"] for k, v in gpus.items()})
            out["rows"].append(row)
            md.append("| %g | %d | %d to %d | %s %dx%d | %.1f | %.1f | %.1f | %.1f | %s |" % (
                R, L, row["n"][0], row["n"][1], lg[0], lg[1], lg[2], host, gpu, asm, cpu_solve,
                " | ".join("%.1f" % row["solve_gpu_min"][k] for k in gpus)))
    md += ["", "Per (R, box) is eight offsets; the kernel table per box takes seconds; every box repeats the assembly and the solve.  Host RAM "
           "assumes all ten blocks of one (offset, box) are assembled before solving (obs_ball_cut_v1.py's run mode); obs_ball_cut_v2.py "
           "assembles one block at a time on the device (host RAM a few GB, device peak about the GPU column), its device assembly not timed on a GPU."]
    # the exact unit-window design (obs_ball_cut_window_v1.py): every distinct ball in [R - 1/2, R + 1/2], one box
    md += ["", "### Exact unit-window design (one box): distinct balls per radius and the solve time at the per-ball cost of the offsets above", "",
           "| R | balls in [R - 1/2, R + 1/2] | solve, box CPU 12 thr (min) | %s |" % " | ".join("solve, %s (min)" % k for k in gpus),
           "|---|---|---|%s" % ("---|" * len(gpus))]
    out["window_rows"] = []
    for R in a.radii:
        rows = [c for c in counts if float(c["R"]) == float(R)]
        if len(rows) < 8:
            continue
        b = int(math.floor(R + 0.5))
        rr = np.arange(-b, b + 1)
        k2 = (rr[:, None, None] ** 2 + rr[None, :, None] ** 2 + rr[None, None, :] ** 2).ravel()
        lo2, hi2 = (R - 0.5) ** 2, (R + 0.5) ** 2
        nb = 1 + int(np.unique(k2[(k2 > lo2) & (k2 <= hi2)]).size)
        per = np.mean([sum(coef[route]["a"] * min(c["m"][nm]) ** 3 for nm in IRREPS) for c in rows]) / 60
        wr = dict(R=R, balls=nb, solve_cpu_min=nb * per, solve_gpu_min={k: nb * per * v["ratio"] for k, v in gpus.items()})
        out["window_rows"].append(wr)
        md.append("| %g | %d | %.1f | %s |" % (R, nb, wr["solve_cpu_min"], " | ".join("%.1f" % wr["solve_gpu_min"][k] for k in gpus)))
    with open(a.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    with open(a.md, "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
