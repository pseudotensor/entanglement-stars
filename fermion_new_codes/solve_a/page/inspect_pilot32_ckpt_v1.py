#!/usr/bin/env python
"""inspect_pilot32_ckpt_v1.py (builder analyst, 2026-10-16; SPEC_final4b_engine.md 36.1 (5)): prints what a page_recursion_v5 checkpoint
holds for the archive direct-route check (read-only): the run's position, each channel's stage record (its initial memory, the stored
coefficients and the collected modes' chunk files) and the FIFO. Usage: python inspect_pilot32_ckpt_v1.py CKPT"""
import sys

import torch

p = sys.argv[1]
s = torch.load(p, map_location="cpu", weights_only=False)
print("checkpoint", p)
print("position: k %s, n_block %s, t %.1f (t/r_s1 %.1f), E %.6f, t_stage0 %.1f; stages %d, transitions %d, curve rows %d" % (
    s["k"], s["n_block"], s["t"], s["t"] / s["cfg"]["rs"], s["E"], s["t_stage0"], len(s["stages"]), len(s["transitions"]), len(s["curve"])))
print("top-level keys:", sorted(s.keys()))
for kap, c in s["channels"].items():
    r = c["rec"]
    fifo_cols = sum(int(x[0].shape[1]) for x in c["fifo"])
    print("kappa %s: n_transitions %s, t_ref %.1f, memory D %s, FIFO %d entries, %d columns (site basis), oldest at t %s" % (
        kap, c.get("n_transitions"), c["t_ref"], tuple(c["D"].shape), len(c["fifo"]), fifo_cols, ("%.1f" % c["fifo"][0][2]) if c["fifo"] else "-"))
    if r is None:
        print("  stage record: None")
        continue
    print("  stage record: t_init %.1f (t/r_s1 %.1f), D_init %s, MD_init %s, coefficient blocks %d (columns %d), tQ %d (first %s, last %s), n_coll %s, n_coll_all %s, cQ_from %s" % (
        r["t_init"], r["t_init"] / s["cfg"]["rs"], tuple(r["D_init"].shape), tuple(r["MD_init"].shape), len(r["a"]), sum(int(x.shape[1]) for x in r["a"]),
        len(r["tQ"]), ("%.1f" % (r["tQ"][0] / s["cfg"]["rs"])) if r["tQ"] else "-", ("%.1f" % (r["tQ"][-1] / s["cfg"]["rs"])) if r["tQ"] else "-",
        r["n_coll"], r.get("n_coll_all"), r.get("cQ_from")))
    print("  chunk files listed by the record (path, blocks): %s" % [(f.split("/")[-1], n) for f, n in r.get("cQ_files", [])])
