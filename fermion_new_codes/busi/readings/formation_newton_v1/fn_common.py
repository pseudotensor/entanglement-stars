"""busi reading formation_newton_v1 (reviewer A): common module. Own reader for the correctness reading of
deliveries/solve_a/FORMATION_newton_route_v1.md (P64_s4 by Newton from T = 0 against the production P64_s4).

Nothing here imports the delivery's scripts, the producer's rc_audit_v1.py or engine_ladder_compare_v5; the rule is implemented from its
text (ReviewA_cert_rule_v1; ReviewA_reading_ISO128_formation_readout_v1 sec. 0): a record counts when its kept residual is at most 1e-3
and, where it has a confirming residual, that one too; it prints in a band when either is above 1e-4; a history is certified to the
record before its first record that does not count; a Newton record's two residuals are its solve_v32 block's (the kept round's); a plain
record's kept residual is its last iteration-history entry when converged, else the stored unconverged residual with the following entry
as its confirming value. Box only (compute-host), CPU, read-only on every input."""
import hashlib, json, math, os, re

BOX = "<internal>/solve_a_bulk"
ND = BOX + "/p64_newton_h100_1"
NJ = ND + "/final/run/collapse_lad_P64_s4_newton_v36nq_h100_1.json"          # the card's final json (8f31394b)
NJ_READOUT = ND + "/readout/newton_run.json"                                 # the copy the delivery's readout used
NJ_EARLY = ND + "/run/collapse_lad_P64_s4_newton_v36nq_h100_1.json"          # the mirror's 11.875 flush (f0127de5)
NF = ND + "/final/run/collapse_lad_P64_s4_newton_v36nq_h100_1_fields.npz"    # e50744ed
NLOG = ND + "/final/logs/lad_P64_s4_newton_v36nq_h100_1.log"                 # 3102f05c (to 12.75)
NLOG_EARLY = ND + "/logs/lad_P64_s4_newton_v36nq_h100_1.log"
PD = BOX + "/v36d_from0/runs_pod7_ell4"
PJ = PD + "/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.json"       # the production P64_s4 (T18 resume, 0 to 40)
PF = PD + "/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40_fields.npz"
PLOG = PD + "/logs/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_to40.log"
P0D = BOX + "/pod_sync/pod2_engine/v36d_from0"
P0J = P0D + "/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.json"          # the from-0 member itself (0 to 18.125)
P0F = P0D + "/collapse_c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0_fields.npz"
P0LOG = P0D + "/logs/c32_rs64_A_R2_tau8_ell4_prep4_avg_plain_v36d_from0.log"
OUT = "<internal>/formation_newton_v1"

RS = 64.0
G = 0.25
CAP = 1e-3      # the production rule's threshold
BAND = 1e-4     # the band mark
FLOOR = 1e-6    # the engine's stop target for the fields


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(path):
    with open(path) as f:
        return json.load(f)


def records(J):
    return [r for r in J["records"] if isinstance(r, dict) and "T_over_rs" in r and r.get("E") is not None]


def ckey(T):
    """clock key: the record cadence is 0.125 r_s, so round to 1/1024 and print short."""
    return round(float(T) * 1024.0) / 1024.0


def fl(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def entry_res(e):
    """an iteration-history entry is [abs dE/E, change of N, change of v]; its residual is the larger field change."""
    if not isinstance(e, (list, tuple)) or len(e) < 3:
        return None
    a, b = fl(e[1]), fl(e[2])
    if a is None and b is None:
        return None
    return max(x for x in (a, b) if x is not None)


def newton_residuals(r):
    """A Newton record's residuals from its solve_v32 block: the last solve's kept_residual (the kept evaluation) and
    confirming_residual (the plain step after it); beside them, the same two read from the iteration history at the kept evaluation
    and the entry after it, so the block can be checked against the history."""
    out = {"kept": None, "conf": None, "kept_evaluation": None, "evaluations": None, "meets_stop_rule": None, "failed_solve": None,
           "status": None, "n_solves": 0, "dE": None, "kept_hist": None, "conf_hist": None, "dE_hist": None, "how": "no solve_v32"}
    sv = r.get("solve_v32")
    if not isinstance(sv, dict) or not sv.get("solves"):
        return out
    solves = sv["solves"]
    s = solves[-1]
    out.update({"kept": fl(s.get("kept_residual")), "conf": fl(s.get("confirming_residual")), "kept_evaluation": s.get("kept_evaluation"),
                "evaluations": s.get("evaluations"), "meets_stop_rule": s.get("meets_stop_rule"), "failed_solve": s.get("failed_solve"),
                "status": s.get("status"), "n_solves": len(solves), "dE": fl(s.get("confirming_dE")), "how": "solve_v32 (last solve)"})
    ih = r.get("iteration_history") or []
    k = out["kept_evaluation"]
    if isinstance(k, int) and 0 <= k < len(ih):
        out["kept_hist"] = entry_res(ih[k])
        if k + 1 < len(ih):
            out["conf_hist"] = entry_res(ih[k + 1])
            out["dE_hist"] = fl(ih[k + 1][0])
    return out


def plain_residuals(r):
    """A plain record's residuals: converged -> the last history entry (kept), no confirming value; stored unconverged -> the stored
    residual (kept) and the history entry after the kept iterate (confirming), when there is one."""
    ih = r.get("iteration_history") or []
    out = {"kept": None, "conf": None, "how": "", "converged": bool(r.get("converged")), "iterations": r.get("iterations")}
    u = r.get("unconverged")
    if r.get("converged") and not u:
        out["kept"] = entry_res(ih[-1]) if ih else None
        out["how"] = "history (plain, converged: last entry)"
        return out
    ures = None
    if isinstance(u, dict):
        for key in ("residual", "kept_residual", "best_residual"):
            if fl(u.get(key)) is not None:
                ures = fl(u.get(key))
                break
    if ures is None and ih:
        ures = min((entry_res(e) for e in ih[1:] if entry_res(e) is not None), default=None)
        out["how"] = "history (plain, unconverged: smallest entry; "
    else:
        out["how"] = "history (plain, unconverged: stored residual; "
    out["kept"] = ures
    k = None
    for i, e in enumerate(ih):
        v = entry_res(e)
        if v is not None and ures is not None and abs(v - ures) <= 1e-15 * max(1.0, abs(ures)):
            k = i
    if k is not None and k + 1 < len(ih):
        out["conf"] = entry_res(ih[k + 1])
        out["how"] += "following entry)"
    else:
        out["how"] += "no following entry)"
    return out


def counts_of(kept, conf):
    """the production rule on one record: (counts, band, why)."""
    if kept is None:
        return False, False, "kept residual missing"
    why = []
    if kept > CAP:
        why.append("kept above 1e-3")
    if conf is not None and conf > CAP:
        why.append("confirming above 1e-3")
    band = kept > BAND or (conf is not None and conf > BAND)
    return (not why), band, "; ".join(why)


def certify(rows):
    """rows: list of dicts with T, kept, conf. Returns the extent under the production rule and the band list."""
    k = None
    for i, x in enumerate(rows):
        c, b, why = counts_of(x["kept"], x["conf"])
        x["counts"], x["band"], x["why"] = c, b, why
        if not c and k is None:
            k = i
    ext = {"records": len(rows), "first_noncounting_T": (rows[k]["T"] if k is not None else None),
           "certified_through_T": (rows[k - 1]["T"] if k is not None and k > 0 else (rows[-1]["T"] if k is None else None)),
           "records_certified": (k if k is not None else len(rows)),
           "bands": [x["T"] for x in rows[: (k if k is not None else len(rows))] if x["band"]],
           "max_kept": max((x["kept"] for x in rows if x["kept"] is not None), default=None),
           "max_kept_T": max((x for x in rows if x["kept"] is not None), key=lambda x: x["kept"], default={"T": None})["T"],
           "max_conf": max((x["conf"] for x in rows if x["conf"] is not None), default=None),
           "max_conf_T": max((x for x in rows if x["conf"] is not None), key=lambda x: x["conf"], default={"T": None})["T"],
           "n_kept_above_floor": sum(1 for x in rows if x["kept"] is not None and x["kept"] > FLOOR),
           "n_conf_above_floor": sum(1 for x in rows if x["conf"] is not None and x["conf"] > FLOOR),
           "n_conf_missing": sum(1 for x in rows if x["conf"] is None)}
    return ext


def quantities(r):
    E = fl(r.get("E"))
    z = [fl(x) for x in (r.get("trapping_zeros") or [])]
    Rt, rin = fl(r.get("R_t")), fl(r.get("r_in"))
    np_ = r.get("n_pocket") or {}
    return {"T": float(r["T_over_rs"]), "T_abs": fl(r.get("T")), "EE0": fl(r.get("E_over_E0")), "E": E, "minNv": fl(r.get("min_N_minus_absv")),
            "r_minNv": fl(r.get("r_min_N_minus_absv")), "Rt": Rt, "rin": rin, "zeros": z, "nzeros": len(z),
            "Rt_2gE_json": fl(r.get("R_t_over_2gE")), "Rt_2gE": (Rt / (2 * G * E) if (Rt is not None and E) else None),
            "twogE": (2 * G * E if E else None), "Rt_rs": (Rt / RS if Rt is not None else None), "rin_rs": (rin / RS if rin is not None else None),
            "Rc": fl(r.get("R_c")), "Rsh": fl(r.get("R_sh")), "Nmin": fl(r.get("N_min")), "pockets": sum(int(v) for v in np_.values()) if np_ else None,
            "inversions": r.get("inversions_total"), "converged": r.get("converged"), "iterations": r.get("iterations"),
            "eigh": r.get("eigendecompositions"), "seconds": fl(r.get("seconds")), "tangencies": len(r.get("tangencies") or []),
            "mask_neg": r.get("mask_equals_negative_set"), "label": r.get("continuation_label")}


NEWTON_RE = re.compile(r"^T/rs (?P<T>[0-9.]+) newton \((?P<kind>[^)]*)\): (?P<status>[^;]*); (?P<nev>\d+) evaluations in (?P<sec>\d+) s; "
                       r"kept residual (?P<kept>[0-9.e+-]+) \(evaluation (?P<kev>\d+)\), the confirming plain step's residual (?P<conf>[0-9.e+-]+) "
                       r"and abs dE/E (?P<dE>[0-9.e+-]+); (?P<conv>[^;]*); line search steps (?P<ls>.*)$")
EE0_RE = re.compile(r"^T/rs (?P<T>[0-9.]+) E/E0 (?P<EE0>[0-9.]+) dust (?P<dust>[0-9.e+-]+) pairs (?P<pairs>[0-9.e+-]+) R_t (?P<Rt>None|[0-9.]+) "
                    r"\(2gE (?P<twogE>[0-9.]+); d1\.\.d4 of N\+v over exact (?P<d4>[^;]*); r_in (?P<rin>None|[0-9.]+)\) min\(N-\|v\|\) (?P<minNv>[0-9.e+-]+) "
                    r"R_sh (?P<Rsh>[0-9.]+) R_c (?P<Rc>[0-9.]+) pockets (?P<pockets>\d+) inversions (?P<inv>\d+) intermediate (?P<inter>\d+) "
                    r"mask=neg (?P<mask>\w+) nsub max (?P<nsub>\d+) it (?P<it>\d+) \(eigh (?P<eigh>\d+)\)")


def parse_log(path):
    """both record line types, keyed by the printed clock (3 decimals as the log prints it)."""
    newton, ee0, n_lines, other = {}, {}, 0, []
    with open(path, errors="replace") as f:
        for line in f:
            n_lines += 1
            if not line.startswith("T/rs "):
                continue
            m = NEWTON_RE.match(line.rstrip("\n"))
            if m:
                d = m.groupdict()
                newton[ckey(d["T"])] = {"T": float(d["T"]), "kind": d["kind"], "status": d["status"], "evaluations": int(d["nev"]), "seconds": int(d["sec"]),
                                        "kept": float(d["kept"]), "kept_evaluation": int(d["kev"]), "conf": float(d["conf"]), "dE": float(d["dE"]),
                                        "conv": d["conv"], "line_search": d["ls"]}
                continue
            m = EE0_RE.match(line)
            if m:
                d = m.groupdict()
                ee0[ckey(d["T"])] = {"T": float(d["T"]), "EE0": float(d["EE0"]), "dust": float(d["dust"]), "pairs": float(d["pairs"]),
                                     "Rt": (None if d["Rt"] == "None" else float(d["Rt"])), "twogE": float(d["twogE"]), "d4": d["d4"],
                                     "rin": (None if d["rin"] == "None" else float(d["rin"])), "minNv": float(d["minNv"]), "Rsh": float(d["Rsh"]),
                                     "Rc": float(d["Rc"]), "pockets": int(d["pockets"]), "inversions": int(d["inv"]), "intermediate": int(d["inter"]),
                                     "mask": d["mask"], "nsub": int(d["nsub"]), "it": int(d["it"]), "eigh": int(d["eigh"]), "line": line[:400]}
                continue
            other.append(line[:160])
    return {"newton": newton, "ee0": ee0, "n_lines": n_lines, "unparsed_T_lines": other}


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    return md5(path)


def fmt(x, s="%.6g"):
    return "-" if x is None else (s % x)
