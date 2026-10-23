#!/usr/bin/env python
"""inject_extract_v1.py (busi, analyst; CVCONFIRM, ReviewA_cvcause_v1 sec. 3): one record's stored row of an engine run, as the small npz
that engine_collapse_v36nq_rd13mfi.py's --inject-fields reads (keys R, T, N_applied, v_applied, N, v as one-row arrays, the arrays copied
bitwise from the fields npz) with its provenance in inject_source (json: the run's dir and tag, the fields npz's and json's md5, the row,
the record's clock, converged, iterations, the unconverged entry, solve_v32's kept residual, kept_from and kept evaluation, N(0) from the
stored nodes). Read only on the source run. Refuses to overwrite.
    python inject_extract_v1.py RUN_DIR TAG T_OVER_RS OUT.npz"""
import hashlib
import json
import os
import sys

import numpy as np


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    d, tag, t, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    if os.path.exists(out):
        raise SystemExit("%s exists; nothing written" % out)
    pf = os.path.join(d, "collapse_%s_fields.npz" % tag); pj = os.path.join(d, "collapse_%s.json" % tag)
    J = json.load(open(pj)); rs = float(J["args"]["rs"])
    Z = np.load(pf, allow_pickle=True)
    T = np.asarray(Z["T"], float)
    hit = np.where(np.abs(T / rs - t) < 1e-6)[0]
    if len(hit) != 1:
        raise SystemExit("the fields npz holds %d rows at T/rs %g" % (len(hit), t))
    i = int(hit[0])
    rec = [r for r in J["records"] if r.get("T") is not None and abs(r["T"] / rs - t) < 1e-6]
    if len(rec) != 1:
        raise SystemExit("the json holds %d records at T/rs %g" % (len(rec), t))
    r = rec[0]; s = r.get("solve_v32") or {}
    src = {"run_dir": os.path.abspath(d), "tag": tag, "fields_npz_md5": md5f(pf), "json_md5": md5f(pj), "row": i, "T": float(T[i]), "T_over_rs": float(T[i] / rs),
           "nsites": int(J["args"]["nsites"]), "rs": rs, "tau": float(J["args"]["tau"]), "script": J.get("script"),
           "converged": r.get("converged"), "iterations": r.get("iterations"), "unconverged": r.get("unconverged"),
           "kept_residual": s.get("kept_residual"), "kept_from": s.get("kept_from"), "kept_evaluation": s.get("kept_evaluation"),
           "N0_stored_nodes": float(np.asarray(Z["N_nodes"][i], float)[0]), "r_nodes0": float(np.asarray(Z["r_nodes"][i], float)[0]),
           "extract": "inject_extract_v1.py"}
    arrs = {k: np.asarray(Z[k][i])[None, :] for k in ("N_applied", "v_applied", "N", "v")}
    np.savez(out, R=np.asarray(Z["R"]), T=np.array([T[i]]), inject_source=json.dumps(src), **arrs)
    Y = np.load(out, allow_pickle=True)
    ok = all(np.array_equal(Y[k][0], Z[k][i]) for k in arrs) and np.array_equal(Y["R"], Z["R"])
    print("wrote %s (md5 %s): row %d at T/rs %.6f of %s; arrays equal to the source's row bitwise: %s; converged %s, kept residual %s (%s, evaluation %s); N(0) %.9f" % (
        out, md5f(out), i, T[i] / rs, tag, ok, r.get("converged"), s.get("kept_residual"), s.get("kept_from"), s.get("kept_evaluation"), src["N0_stored_nodes"]))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
