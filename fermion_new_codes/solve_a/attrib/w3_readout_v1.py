#!/usr/bin/env python
"""w3_readout_v1.py (composition builder, the solve lane): the producer of W3_completions_v1 (W3 of final4/WEB40I_TASKS/busi.md, the
completion control). Reads stored files only: the record's own history (rung 4, the (0.2, 64) sigma/a 4 tau 8 member on v36d from T = 0
to 40) and the busi lane's completions run on pod 4 from its T0011.00 slice (completion from T/r_s 11.125: frozen-sources, paired-avg,
point-core), their records json and fields npz (the completions' files recovered from the Drive mirror with the manifest's md5).
Per history per record: T/r_s, E/E0, R_t (the outer dense root) and the zeros, min(N - |v|), the residual and the counting flag (the
ladder reader's rule), and the far dust energy: the bare dust energy of the fields row's shells (eps_tot - eps_pairs) beyond 16 r_s and
beyond 24 r_s (sites R > 16 r_s, R > 24 r_s), and the pairs' energy beyond the same cuts. The spread at each matched clock is
max - min over the histories holding a counting record there (labelled with the count of histories).
Usage (box CPU): python w3_readout_v1.py --ladders-code DIR --w3-dir DIR --out-dir DIR"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

OWN = ("own", "<internal>/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40",
       "the record's own history (rung 4, v36d from T = 0)")
COMP = [("frozen-sources", "the sources frozen inside the gate radius, the charge matched to the record's E"),
        ("paired-avg", "the paired averaging widened to sigma/a 8 inside the gate radius, sigma/a 4 outside"),
        ("point-core", "the core coupling point-sampled inside the gate radius")]
CUTS = (16.0, 24.0)


def md5sum(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders-code", required=True)
    ap.add_argument("--w3-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.ladders_code)
    import ladders_readout_v1 as LR
    os.makedirs(a.out_dir, exist_ok=True)
    hist = [(OWN[0], OWN[1], OWN[2])]
    for nm, lab in COMP:
        tg = "w3_%s_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36g" % nm
        hist.append((nm, os.path.join(a.w3_dir, "runs", "w3_" + nm, "collapse_" + tg), lab))
    H, arrays = {}, {}
    for nm, base, lab in hist:
        jp, fp = base + ".json", base + "_fields.npz"
        meta, cols, ints, zeros_all = LR.load_member(nm, jp, lab, "w3", None)
        s = LR.summarize(meta, cols, ints, zeros_all)
        J = json.load(open(jp))
        A = J["args"]
        Z = np.load(fp, allow_pickle=True)
        R = np.asarray(Z["R"], float); Tf = np.asarray(Z["T"], float) / float(A["rs"])
        eps_d = np.asarray(Z["eps_tot"], float) - np.asarray(Z["eps_pairs"], float); eps_p = np.asarray(Z["eps_pairs"], float)
        far = {}
        for c in CUTS:
            m = R > c * float(A["rs"])
            far["dust_%g" % c] = eps_d[:, m].sum(axis=1); far["pairs_%g" % c] = eps_p[:, m].sum(axis=1)
        # align the fields rows to the records
        T = cols["T"]
        idx = [int(np.argmin(np.abs(Tf - t))) for t in T]
        ok = np.array([abs(Tf[i] - t) < 1e-6 for i, t in zip(idx, T)])
        for k, v in far.items():
            arrays["%s__%s" % (nm, k)] = np.where(ok, v[idx], np.nan)
        for k in ("T", "EE0", "Rt", "minNv", "res"):
            arrays["%s__%s" % (nm, k)] = cols[k]
        arrays["%s__counts" % nm] = ints["counts"]
        comp = {k: A.get(k) for k in A if k.startswith("completion")}
        H[nm] = {"label": lab, "json": jp, "json_md5": md5sum(jp), "fields": fp, "fields_md5": md5sum(fp), "script": J.get("script"),
                 "engine_entries": meta["engine_entries"], "completion_args": comp, "resume_slice": A.get("resume_slice"),
                 "convergence": s["convergence"], "episodes": s["episodes"], "formation": s["formation"], "records": meta["records"],
                 "fields_rows_matched": int(ok.sum())}
        print(nm, meta["records"], s["convergence"]["last"], s["convergence"]["last_converged"], flush=True)
    names = [h[0] for h in hist]
    clocks = [11.125] + [float(t) for t in range(12, 41)]
    rows = []
    for t in clocks:
        row = {"T": t, "per": {}}
        for nm in names:
            T = arrays["%s__T" % nm]
            j = np.where(np.abs(T - t) < 1e-6)[0]
            if not len(j):
                continue
            j = int(j[0])
            row["per"][nm] = {"EE0": float(arrays["%s__EE0" % nm][j]), "R_t": (None if not np.isfinite(arrays["%s__Rt" % nm][j]) else float(arrays["%s__Rt" % nm][j])),
                              "minNv": float(arrays["%s__minNv" % nm][j]), "counts": bool(arrays["%s__counts" % nm][j]),
                              **{k: (None if not np.isfinite(arrays["%s__%s" % (nm, k)][j]) else float(arrays["%s__%s" % (nm, k)][j]))
                                 for k in ("dust_16", "dust_24", "pairs_16", "pairs_24")}}
        sp = {}
        for k in ("EE0", "R_t", "dust_16", "dust_24", "minNv"):
            v = [p[k] for p in row["per"].values() if p["counts"] and p.get(k) is not None]
            sp[k] = {"n": len(v), "spread": (max(v) - min(v)) if len(v) >= 2 else None, "min": min(v) if v else None, "max": max(v) if v else None}
        nz = [nm for nm, p in row["per"].items() if p["counts"]]
        sp["R_t_holders"] = [nm for nm, p in row["per"].items() if p["R_t"] is not None]
        sp["counting"] = nz
        row["spread"] = sp
        rows.append(row)
    out = {"histories": H, "order": names, "matched": rows, "cuts_over_rs": CUTS}
    jp = os.path.join(a.out_dir, "w3_readout_v1.json")
    json.dump(out, open(jp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    npz = os.path.join(a.out_dir, "w3_readout_v1.npz")
    np.savez_compressed(npz, **arrays)
    with open(os.path.join(a.out_dir, "MANIFEST_w3_readout_v1.md5"), "w") as f:
        for p in (jp, npz):
            f.write("%s  %s\n" % (md5sum(p), os.path.basename(p)))
    print("wrote", jp)


if __name__ == "__main__":
    main()
