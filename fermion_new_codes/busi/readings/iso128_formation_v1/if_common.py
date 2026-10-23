#!/usr/bin/env python
"""if_common.py (busi lane, correctness reading of ISO128_formation_readout_v1): my own reader of the ladder run jsons and of the
H2 engine log. Nothing here imports the producer's modules (ladders_readout_v1_3, engine_ladder_compare_v5/v6) or the delivery's
scripts; the rule is implemented from its text (ReviewA_cert_rule_v1, REVIEWA_rc_floor_v1 sec. 7: a record counts when its kept
residual is at most 1e-3 and, where a confirming residual exists, that one too; it prints in a band when either is above 1e-4; a
history is certified to the record before its first record that does not count; a plain record stored unconverged is confirmed by
the iteration-history entry after its kept iterate)."""
import hashlib
import json
import re

import numpy as np

CAP = 1e-3
BAND = 1e-4
B = "<internal>/solve_a_bulk"
OUT = "<internal>/iso128_formation_v1"
TAG = "lad_ISO_128_s4_nkqh2_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36nq"
H2_JSON = B + "/iso128_h2/run_h2/collapse_%s.json" % TAG
H2_FIELDS = B + "/iso128_h2/run_h2/collapse_%s_fields.npz" % TAG
H2_LOG = B + "/iso128_h2/logs/%s.log" % TAG
SLICE_11p5 = B + "/iso128_h2/states_h2/orbitals_%s_T0011.50.npz" % TAG
SLICE_5 = B + "/iso128_h2/states_h2/orbitals_%s_T0005.00.npz" % TAG
MIRROR_5_JSON = B + "/resume_mirror/pod6/%s/T0005.00/collapse_%s.json" % (TAG, TAG)
RESTART_LOG = B + "/iso128_h2<internal>/restart.log"
STOP_FILE = B + "/iso128_h2/STOP_AT_FORMATION_h2.txt"
MD5_BOX = B + "/iso128_h2/MD5_box.txt"
MD5_CARD = B + "/iso128_h2/MD5_card.txt"
PULL_LOG = B + "/iso128_h2/pull_v1.log"
L = B + "/ladders"
MEMBERS = {
    "ISO_128_s4_plain": L + "/p6/runs/lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n/collapse_lad_ISO_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n.json",
    "S4_128_s4": L + "/r2/runs/lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s4_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36d.json",
    "S4_128_s4_nk_v36m": L + "/p8/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36m.json",
    "S4_128_s4_nk_v36q": L + "/r3/runs/lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q/collapse_lad_S4_128_s4_nk_m0p2_rs128_n4096_R04_k10_tau8_ell4_prep4_A_R2_avg_newton_v36q.json",
    "S4_128_s8": L + "/r1/runs/lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s8_m0p2_rs128_n4096_R04_k10_tau8_ell8_prep4_A_R2_avg_plain_v36d.json",
    "S4_128_s16": L + "/p8/runs/lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d/collapse_lad_S4_128_s16_m0p2_rs128_n4096_R04_k10_tau8_ell16_prep4_A_R2_avg_plain_v36d.json",
    "ISO_64_s4": B + "/isoB/collapse_isoB.json",
    # the certified r_s/a 64 members read 3 compared across formation (ISO_128_nkq_build_v2 sec. 3), and the ISO_64 plain original
    "production_c19": "<internal>/collapse_c19_rs64_A_R2_tau8_ell4_avg_r11.json",
    "P64_s4": B + "/v36d_from0/runs_pod7_ell4/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json",
    "ISO_64_s4_plain": L + "/r1/runs/lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n/collapse_lad_ISO_64_s4_m0p2_rs64_n2048_R04_k10_tau8_ell4_prep4_A_R2_avg_plain_v36n.json",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def canon_md5(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def load(path):
    return json.load(open(path))


def records(J):
    return [r for r in J["records"] if isinstance(r, dict) and "T_over_rs" in r and r.get("E") is not None]


def is_newton(J):
    return (J.get("args") or {}).get("mix") == "newton"


def fl(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def ih_res(h):
    return max(float(h[1]), float(h[2]))


def residuals(r, newton):
    """My reading of a record's residuals: dict(kept, conf, kept_alt, conf_alt, how).
    Newton record with a solve_v32 block: kept = its kept_residual, conf = its confirming_residual (the kept round's).
    Newton record without one (the T = 0 row the resume keeps): the interior entries of the iteration history are the evaluations
    (first entry the start, last the confirming step); kept = the smallest of them when converged, else the stored unconverged
    residual; conf = the last entry. With two entries the interior is empty: kept None ("missing"); kept_alt/conf_alt read the
    row as the plain T = 0 record it is (the last entry, kept = conf).
    Plain record: kept = the last entry when converged, else the stored unconverged residual; conf = the entry after the kept
    iterate when the record is stored unconverged and such an entry exists (the production rule's following-entry test), else
    None (no confirming test)."""
    ih = r.get("iteration_history") or []
    conv = r.get("converged")
    sv = r.get("solve_v32") if isinstance(r.get("solve_v32"), dict) else None
    out = {"kept": None, "conf": None, "kept_alt": None, "conf_alt": None, "how": ""}
    if newton and sv is not None and sv.get("kept_residual") is not None:
        out["kept"] = fl(sv["kept_residual"])
        out["conf"] = fl(sv.get("confirming_residual"))
        out["how"] = "solve_v32"
        return out
    if not ih:
        out["how"] = "no history"
        return out
    if newton:
        if conv is True or conv is None:
            inner = [ih_res(h) for h in ih[1:-1]]
            out["kept"] = min(inner) if inner else None
        else:
            out["kept"] = fl((r.get("unconverged") or {}).get("residual"))
        out["conf"] = ih_res(ih[-1])
        out["kept_alt"] = ih_res(ih[-1])
        out["conf_alt"] = ih_res(ih[-1])
        out["how"] = "history (newton, no solve_v32)"
        return out
    if conv is True:
        out["kept"] = ih_res(ih[-1])
        out["how"] = "history (plain, converged: last entry)"
        return out
    u = r.get("unconverged") or {}
    out["kept"] = fl(u.get("residual"))
    k = u.get("iteration")
    if k is not None and k + 1 < len(ih) and out["kept"] is not None and abs(ih_res(ih[k]) - out["kept"]) <= 1e-15 * max(1.0, abs(out["kept"])):
        out["conf"] = ih_res(ih[k + 1])
        out["how"] = "history (plain, unconverged: stored residual; following entry)"
    else:
        out["how"] = "history (plain, unconverged: stored residual; no following entry)"
    return out


def counts_of(kept, conf, newton):
    """The production rule on one record: (counts, banded, why)."""
    if kept is None:
        return False, False, "kept residual missing"
    if newton and conf is None:
        return False, False, "confirming residual missing"
    why = []
    if kept > CAP:
        why.append("kept above 1e-3")
    if conf is not None and conf > CAP:
        why.append("confirming above 1e-3")
    if why:
        return False, False, "; ".join(why)
    band = kept > BAND or (conf is not None and conf > BAND)
    return True, band, ""


def certify(R, newton, t0_as_plain=False):
    """Per-record rows and the extent. t0_as_plain: a Newton record without a solve block and with an empty interior (the T = 0
    row) is read by its last entry (kept = conf = that entry) instead of as missing."""
    rows = []
    for i, r in enumerate(R):
        q = residuals(r, newton)
        kept, conf = q["kept"], q["conf"]
        if newton and kept is None and t0_as_plain and q["kept_alt"] is not None:
            kept, conf = q["kept_alt"], q["conf_alt"]
        c, b, why = counts_of(kept, conf, newton)
        rows.append({"i": i, "T": float(r["T_over_rs"]), "kept": kept, "conf": conf, "counts": c, "band": b, "why": why,
                     "converged": r.get("converged"), "how": q["how"]})
    k = next((x["i"] for x in rows if not x["counts"]), None)
    for x in rows:
        x["certified"] = (k is None) or (x["i"] < k)
    ext = {"records": len(rows), "first_noncounting_T": (rows[k]["T"] if k is not None else None),
           "first_noncounting_why": (rows[k]["why"] if k is not None else None),
           "certified_through_T": (rows[k - 1]["T"] if (k is not None and k > 0) else (None if k == 0 else rows[-1]["T"])),
           "n_certified": (k if k is not None else len(rows)),
           "band_T": [x["T"] for x in rows if x["band"]],
           "band_T_certified": [x["T"] for x in rows if x["band"] and x["certified"]],
           "noncounting_T": [x["T"] for x in rows if not x["counts"]],
           "max_kept": max((x["kept"] for x in rows if x["kept"] is not None), default=None),
           "max_kept_T": max((x for x in rows if x["kept"] is not None), key=lambda x: x["kept"], default={"T": None})["T"],
           "max_conf": max((x["conf"] for x in rows if x["conf"] is not None), default=None),
           "max_conf_T": max((x for x in rows if x["conf"] is not None), key=lambda x: x["conf"], default={"T": None})["T"]}
    return ext, rows


def certify_kept_only(R, newton):
    """Rule 30.3 as engine_ladder_compare_v5 read it (the kept residual alone), for reference."""
    rows = []
    for i, r in enumerate(R):
        q = residuals(r, newton)
        kept = q["kept"] if q["kept"] is not None else q["kept_alt"]
        ok = kept is not None and kept <= CAP
        rows.append({"i": i, "T": float(r["T_over_rs"]), "kept": kept, "counts": ok, "band": ok and kept > BAND})
    k = next((x["i"] for x in rows if not x["counts"]), None)
    return {"first_noncounting_T": (rows[k]["T"] if k is not None else None),
            "certified_through_T": (rows[k - 1]["T"] if (k is not None and k > 0) else (None if k == 0 else rows[-1]["T"])),
            "band_T": [x["T"] for x in rows if x["band"]]}


def pockets_of(r):
    p = r.get("n_pocket") or {}
    return int(sum(int(v) for v in p.values())) if isinstance(p, dict) and p else 0


def zeros_of(r):
    z = r.get("trapping_zeros") or []
    return [float(x) for x in z] if isinstance(z, (list, tuple)) else []


def row_of(r, J):
    """The record-phase quantities of one record: E/E0, min(N - |v|), R_t (a, /r_s, /2gE), r_in (a, /r_s), pockets, inversions."""
    A = J["args"]
    rs, g = float(A["rs"]), float(A["g"])
    z = zeros_of(r)
    E = fl(r.get("E"))
    out = {"T": float(r["T_over_rs"]), "EE0": fl(r.get("E_over_E0")), "E": E, "minNv": fl(r.get("min_N_minus_absv")),
           "r_minNv": fl(r.get("r_min_N_minus_absv")), "nzeros": len(z), "zeros": z, "Rc": fl(r.get("R_c")),
           "pockets": pockets_of(r), "inversions": int(r.get("inversions_total") or 0), "converged": r.get("converged")}
    if z:
        Rt = fl(r.get("R_t"))
        Rt = Rt if Rt is not None else max(z)
        rin = fl(r.get("r_in"))
        rin = rin if rin is not None else min(z)
        out.update({"Rt": Rt, "Rt_rs": Rt / rs, "Rt_2gE_json": fl(r.get("R_t_over_2gE")), "Rt_2gE": (Rt / (2 * g * E) if E else None),
                    "rin": rin, "rin_rs": rin / rs, "Rt_is_max_zero": abs(Rt - max(z)) < 1e-9, "rin_is_min_zero": abs(rin - min(z)) < 1e-9})
    else:
        out.update({"Rt": None, "Rt_rs": None, "Rt_2gE_json": None, "Rt_2gE": None, "rin": None, "rin_rs": None})
    return out


def at_clock(R, t, tol=1e-6):
    for i, r in enumerate(R):
        if abs(float(r["T_over_rs"]) - t) < tol:
            return i
    return None


# ---- the H2 engine log ----
NEWTON_RE = re.compile(r"^T/rs (?P<T>[0-9.]+) newton \((?P<kind>[a-z]+)\): (?P<status>.*?); (?P<ev>\d+) evaluations in (?P<sec>\d+) s; "
                       r"kept residual (?P<kept>[0-9.e+-]+) \(evaluation (?P<kev>\d+)\), the confirming plain step's residual (?P<conf>[0-9.e+-]+) "
                       r"and abs dE/E (?P<dE>[0-9.e+-]+); (?P<conv>converged|not converged); line search steps \[(?P<ls>[^\]]*)\]")
EE0_RE = re.compile(r"^T/rs (?P<T>[0-9.]+) E/E0 (?P<EE0>[0-9.]+) dust (?P<dust>[0-9.e+-]+) pairs (?P<pairs>[0-9.e+-]+) R_t (?P<Rt>None|[0-9.]+) "
                    r"\(2gE (?P<twogE>[0-9.]+); d1\.\.d4 of N\+v over exact (?P<d4>[^;]*); r_in (?P<rin>None|[0-9.]+)\) min\(N-\|v\|\) (?P<minNv>[0-9.e+-]+) "
                    r"R_sh (?P<Rsh>[0-9.]+) R_c (?P<Rc>[0-9.]+) pockets (?P<pockets>\d+) inversions (?P<inv>\d+)")


def parse_log(path):
    """Every record of the engine log, keyed by the clock as printed (3 decimals): the Newton line and the E/E0 line."""
    newton, ee0, other = {}, {}, []
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = NEWTON_RE.match(line)
        if m:
            d = m.groupdict()
            newton[d["T"]] = {"kind": d["kind"], "status": d["status"], "evaluations": int(d["ev"]), "seconds": int(d["sec"]),
                              "kept": float(d["kept"]), "kept_evaluation": int(d["kev"]), "conf": float(d["conf"]), "dE": float(d["dE"]),
                              "converged": d["conv"] == "converged", "line_search": d["ls"], "line": line}
            continue
        m = EE0_RE.match(line)
        if m:
            d = m.groupdict()
            ee0[d["T"]] = {"EE0": float(d["EE0"]), "dust": float(d["dust"]), "pairs": float(d["pairs"]),
                           "Rt": (None if d["Rt"] == "None" else float(d["Rt"])), "twogE": float(d["twogE"]),
                           "rin": (None if d["rin"] == "None" else float(d["rin"])), "minNv": float(d["minNv"]), "Rsh": float(d["Rsh"]),
                           "Rc": float(d["Rc"]), "pockets": int(d["pockets"]), "inversions": int(d["inv"]), "line": line, "d4": d["d4"]}
            continue
        if line.startswith("T/rs"):
            other.append(line[:200])
    return newton, ee0, other


def log_key(t):
    """The clock as the engine prints it (%.3f, round half to even on the exact binary sixteenths: 3.0625 -> 3.062, 11.4375 -> 11.438)."""
    return "%.3f" % t


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda x: (float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))
