"""s16A_end_lib_v1.py (busi): shared helpers for the B' s = 16 stage A endpoint reads (2c, 2b, the 1e2 L clock).
Loading with an md5 check against each run's MD5SUMS, spectra, block comparisons, log parsing. No numbers are produced here;
the emitters (s16A_read_2c_v1.py, s16A_read_2b_v1.py and later versions) print every number. Never edit after first use: new file."""
import hashlib
import json
import math
import os
import re

import numpy as np

L = 16.0
T1 = math.log(2) * L / 0.1     # 6.93 L clock, in a
T2 = math.log(2) * L / 0.01    # 69.3 L clock, in a
T3 = 100.0 * L                 # 1e2 L clock, in a


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_run(d, tag, need_md5sums=True):
    """Load tag.json and tag.npz from d; verify both against d/tag.MD5SUMS (or d/MD5SUMS). Returns a dict."""
    pj, pz = os.path.join(d, tag + ".json"), os.path.join(d, tag + ".npz")
    have = {os.path.basename(pj): md5(pj), os.path.basename(pz): md5(pz)}
    listed = {}
    for cand in (os.path.join(d, tag + ".MD5SUMS"), os.path.join(d, "MD5SUMS")):
        if os.path.exists(cand):
            for line in open(cand):
                parts = line.split()
                if len(parts) == 2:
                    listed[os.path.basename(parts[1].lstrip("*"))] = parts[0]
    ok = all(listed.get(k) == v for k, v in have.items())
    if need_md5sums and not ok:
        raise SystemExit("MD5 CHECK FAILED for %s/%s: have %s listed %s" % (d, tag, have, {k: listed.get(k) for k in have}))
    z = np.load(pz)
    return {"dir": d, "tag": tag, "j": json.load(open(pj)), "z": z, "md5_npz": have[os.path.basename(pz)], "md5_json": have[os.path.basename(pj)],
            "md5_ok": ok, "T": np.asarray(z["T"]), "ns": int(z["n_supply"]), "K": len(z["e"])}


def cidx(Ts, T, rel=1e-6):
    Ts = np.asarray(Ts)
    i = int(np.argmin(np.abs(Ts - T)))
    return i if abs(Ts[i] - T) < rel * T else None


def G_at(R, T):
    i = cidx(R["T"], T)
    return None if i is None else np.asarray(R["z"]["G"][i])


def herm(G):
    return 0.5 * (G + G.conj().T)


def eigp(G):
    return np.sort(np.linalg.eigvalsh(herm(G)))


def blocks(G, ns):
    return {"ss": G[:ns, :ns], "ee": G[ns:, ns:], "se": G[:ns, ns:]}


def cmp_full(Ga, Gb, ns):
    """Ga, Gb: full (supply + exit) Grams in the same basis. Element and ordered-spectrum differences, per block and joint."""
    d = Ga - Gb
    out = {"max_dG": float(np.max(np.abs(d))), "norm2_dG": float(np.linalg.norm(d, 2))}
    for k, sl in (("ss", (slice(0, ns), slice(0, ns))), ("ee", (slice(ns, None), slice(ns, None))), ("se", (slice(0, ns), slice(ns, None)))):
        out["max_dG_" + k] = float(np.max(np.abs(d[sl])))
    pa, pb = eigp(Ga), eigp(Gb)
    out["dp_joint"] = pa - pb
    out["p_joint_a"], out["p_joint_b"] = pa, pb
    for k, sl in (("ss", slice(0, ns)), ("ee", slice(ns, None))):
        qa, qb = eigp(Ga[sl, sl]), eigp(Gb[sl, sl])
        out["dp_" + k] = qa - qb
        out["p_%s_a" % k], out["p_%s_b" % k] = qa, qb
    out["norm_se_a"] = float(np.linalg.norm(Ga[:ns, ns:], 2))
    out["norm_se_b"] = float(np.linalg.norm(Gb[:ns, ns:], 2))
    return out


def cmp_block(Ga, Gb):
    d = Ga - Gb
    pa, pb = eigp(Ga), eigp(Gb)
    return {"max_dG": float(np.max(np.abs(d))), "norm2_dG": float(np.linalg.norm(d, 2)), "max_ddiag": float(np.max(np.abs(np.diag(d).real))),
            "dp": pa - pb, "pa": pa, "pb": pb}


def joint_summary(G, ns):
    pj = eigp(G)
    pu = np.sort(np.concatenate([eigp(G[:ns, :ns]), eigp(G[ns:, ns:])]))
    ps, pe = eigp(G[:ns, :ns]), eigp(G[ns:, ns:])
    Gse = G[:ns, ns:]
    sv = np.linalg.svd(Gse, compute_uv=False)
    return {"pj": pj, "pu": pu, "ps": ps, "pe": pe, "norm_se": float(sv[0]), "max_se": float(np.max(np.abs(Gse))), "sv_se": sv,
            "dark": int((pj < 1e-2).sum()), "max_pj_minus_pu": float(np.max(np.abs(pj - pu))), "trace": float(np.trace(G).real),
            "mirror_dp": float(np.max(np.abs(ps - pe)))}


def parse_log(path):
    """Engine log: per-clock completion seconds, batch lines, the 'wrote' line."""
    out = {"clock_done": [], "batches": [], "wrote_s": None, "lines": 0}
    if not os.path.exists(path):
        return out
    for line in open(path):
        out["lines"] += 1
        m = re.search(r"clock (\d+)/(\d+) T/L ([0-9.e+]+) done \((\d+) s\)", line)
        if m:
            out["clock_done"].append((float(m.group(3)), float(m.group(4))))
        m = re.search(r"batch (\d+)/(\d+) at T/L ([0-9.e+]+) \((\d+) s; step ([0-9.]+) s median\)", line)
        if m:
            out["batches"].append((int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
        m = re.search(r"wrote (\S+) \((\d+) s\)", line)
        if m:
            out["wrote_s"] = float(m.group(2))
    return out


def manifest_row(R, log):
    j = R["j"]; tm = j["timing"]; a = j["args"]
    K = R["K"]; Tmax_a = float(max(R["T"]))
    wall = log["wrote_s"] if log["wrote_s"] else tm["seconds_total_this_session"]
    rate_step = tm["step_median_s"] / (min(tm["col_batch"], K) * tm["dt"])
    rate_full = wall / (K * Tmax_a)
    pm = j.get("prep_manifest", {})
    return {"tag": R["tag"], "dt": tm["dt"], "onset": pm.get("onset_L", a["onset"]), "box_L": pm.get("box_L", a["box_L"]), "eta": a["eta_max"], "K": K, "ns": R["ns"],
            "clocks": [float(x) / L for x in R["T"]], "steps": int(round(Tmax_a / tm["dt"])), "prep": os.path.basename(a.get("prep", "")), "step_median": tm["step_median_s"], "wall_h": wall / 3600.0,
            "session_s": tm["seconds_total_this_session"], "resumed_from": tm.get("resumed_from", []), "peak_GB": tm.get("peak_mem_GB") or 0.0,
            "device": tm.get("device_name", ""), "col_batch": tm["col_batch"], "rate_step": rate_step, "rate_full": rate_full,
            "col_a_per_s": K * Tmax_a / wall, "md5_npz": R["md5_npz"], "md5_json": R["md5_json"], "md5_ok": R["md5_ok"]}


def residual_vs_bar(p, bar):
    """Residual 1 - p per ordered direction against a numerical bar (array or scalar)."""
    r = 1.0 - np.asarray(p)
    bar = np.broadcast_to(np.asarray(bar, float), r.shape)
    return {"max_res": float(r.max()), "median_res": float(np.median(r)), "n_res_gt_1e3": int((r > 1e-3).sum()), "n_res_gt_1e4": int((r > 1e-4).sum()),
            "n_resolved_10x": int((r > 10 * bar).sum()), "n_resolved_1x": int((r > bar).sum()), "n": len(r),
            "min_ratio": float(np.min(r / np.maximum(bar, 1e-300))), "max_bar": float(bar.max())}


def fmt(x, d=3):
    return ("%%.%dg" % d) % x
