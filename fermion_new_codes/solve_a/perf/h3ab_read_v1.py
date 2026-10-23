"""h3ab_read_v1.py (solve 8a7b5274, PERF H3 A/B on h100-f): read the arms against H3_0 per busi ready_jobs_v3 sec. 35 (light; runs on h100-f).
1) every npz member of the four run npz files (fields, handoff, s11, spectra) bitwise against H3_0 (dtype, shape, bytes), and the json records
   equal key by key except the wall keys (seconds, timing, *_s, slice_saved), as a second check beside cmp_v9c3_runs.py --args-equal;
2) the record walls (json 'seconds') after T/rs 13: whole segment, 13.344 and 13.625 separately; the process wall (wall.txt);
3) the 'v9c4 perf (the run)' line; 4) telemetry: GPU util mean, share < 10 %, share > 90 %, memory.used max, power mean.
Usage: python h3ab_read_v1.py OUT.json H3_0_DIR [ARM_DIR ...]"""
import csv
import json
import os
import re
import sys

import numpy as np

TAG = "C64_s8_ell8_tau2_twofield_v4L"
WALLK = {"seconds", "timing", "slice_saved"}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in WALLK and not k.endswith("_s")}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def recs(d):
    return json.load(open(os.path.join(d, "collapse_%s.json" % TAG)))


def walls(d):
    R = [r for r in recs(d)["records"] if r.get("E") is not None and r["T_over_rs"] > 13.0 + 1e-9]
    w = {"n": len(R), "first": R[0]["T_over_rs"] if R else None, "last": R[-1]["T_over_rs"] if R else None,
         "sum_s": sum(float(r.get("seconds", 0.0)) for r in R)}
    for t in (13.34375, 13.625):
        m = [float(r.get("seconds", 0.0)) for r in R if abs(r["T_over_rs"] - t) < 1e-6]
        w["s_%g" % t] = m[0] if m else None
    w["per_record"] = [[r["T_over_rs"], float(r.get("seconds", 0.0))] for r in R]
    try:
        txt = open(os.path.join(d, "wall.txt")).read()
        w["process_wall_s"] = int(re.search(r"wall_s (\d+)", txt).group(1))
        w["rc"] = int(re.search(r" rc (\d+)", txt).group(1))
    except Exception:
        pass
    return w


def npz_cmp(a, b):
    out = {}
    for suf in ("fields", "handoff", "s11", "spectra"):
        fa, fb = [os.path.join(x, "collapse_%s_%s.npz" % (TAG, suf)) for x in (a, b)]
        if not (os.path.exists(fa) and os.path.exists(fb)):
            out[suf] = "missing %s %s" % (os.path.exists(fa), os.path.exists(fb)); continue
        Za, Zb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
        ka, kb = set(Za.files), set(Zb.files)
        bad = sorted(ka ^ kb)
        for k in sorted(ka & kb):
            x, y = Za[k], Zb[k]
            if x.dtype != y.dtype or x.shape != y.shape:
                bad.append(k); continue
            if x.dtype == object:
                if repr(x.tolist()) != repr(y.tolist()):
                    bad.append(k)
            elif np.ascontiguousarray(x).tobytes() != np.ascontiguousarray(y).tobytes():
                bad.append(k)
        out[suf] = {"members": len(ka & kb), "differ": bad}
    return out


def rec_cmp(a, b):
    A, B = recs(a)["records"], recs(b)["records"]
    n = min(len(A), len(B))
    diff = [A[i].get("T_over_rs") for i in range(n) if strip(A[i]) != strip(B[i])]
    return {"len": [len(A), len(B)], "records_differing_but_walls": diff}


def perf_line(d):
    L = [l.strip() for l in open(os.path.join(d, "segment_run.log"), errors="replace") if "v9c4 perf" in l]
    return L


def telemetry(d):
    p = os.path.join(d, "telemetry", "smi.csv")
    u, m, pw = [], [], []
    try:
        for row in csv.reader(open(p)):
            try:
                u.append(float(row[1].split()[0])); m.append(float(row[3].split()[0])); pw.append(float(row[4].split()[0]))
            except Exception:
                pass
    except Exception:
        return None
    if not u:
        return None
    return {"samples": len(u), "util_mean": round(sum(u) / len(u), 1), "share_lt10": round(sum(x < 10 for x in u) / len(u), 3),
            "share_gt90": round(sum(x > 90 for x in u) / len(u), 3), "mem_max_MiB": max(m), "power_mean_W": round(sum(pw) / len(pw), 1)}


def main():
    out, ref, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
    res = {"H3_0": {"dir": ref, "walls": walls(ref), "telemetry": telemetry(ref), "perf": perf_line(ref)}}
    for d in arms:
        name = os.path.basename(d.rstrip("/")).split("_20")[0].replace("H3AB_", "")
        w = walls(d)
        res[name] = {"dir": d, "walls": w, "telemetry": telemetry(d), "perf": perf_line(d), "npz_vs_H3_0": npz_cmp(ref, d), "records_vs_H3_0": rec_cmp(ref, d)}
        w0 = res["H3_0"]["walls"]
        res[name]["gain_vs_H3_0_pct"] = {k: (round(100.0 * (w0[k] - w[k]) / w0[k], 2) if w0.get(k) and w.get(k) else None)
                                          for k in ("sum_s", "s_13.3438", "s_13.625", "process_wall_s")}
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        w = v["walls"]
        print(k, "records", w["n"], w["first"], "->", w["last"], "sum_s %.1f" % w["sum_s"], "13.344 %s" % w.get("s_13.3438"), "13.625 %s" % w.get("s_13.625"),
              "process_wall_s", w.get("process_wall_s"), "rc", w.get("rc"))
        print("  telemetry", v["telemetry"]); print("  perf", v["perf"])
        if "npz_vs_H3_0" in v:
            print("  npz", v["npz_vs_H3_0"]); print("  records", v["records_vs_H3_0"]); print("  gain %", v["gain_vs_H3_0_pct"])


if __name__ == "__main__":
    main()
