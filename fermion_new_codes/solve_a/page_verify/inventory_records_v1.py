#!/usr/bin/env python3
"""inventory_records_v1.py (solve, page_verify Task 1): read-only inventory of heavy-engine record jsons and light-channel
state/summary files named on the command line. For each record json: script, rs, nsites, tau, ell, sigma (args.ell), number of
records, first/last T/r_s, the trapping clock range, resume/slice args. For each summary json: code, settings (record, nsites,
channels/kappas, cuts_rs, freeze_at_rs, init_state, T_end), rows' clock range and whether cut entropies are stored.
Writes one json (argv[1]) and prints a table. No compute beyond json parsing and md5.
"""
import hashlib, json, os, sys


def md5(p, block=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rec_info(p):
    J = json.load(open(p))
    a = J.get("args", {})
    R = J.get("records", [])
    T = [r.get("T_over_rs") for r in R if isinstance(r, dict)]
    trapped = [r.get("T_over_rs") for r in R if isinstance(r, dict) and r.get("R_t") not in (None, 0, 0.0)]
    return {"kind": "record", "script": J.get("script"), "rs": a.get("rs"), "nsites": a.get("nsites"), "tau": a.get("tau"),
            "ell": a.get("ell"), "m": a.get("m"), "g": a.get("g"), "T_over_rs_arg": a.get("T_over_rs"),
            "n_records": len(R), "T_first": T[0] if T else None, "T_last": T[-1] if T else None,
            "trapped_first": trapped[0] if trapped else None, "trapped_last": trapped[-1] if trapped else None,
            "resume_from": a.get("resume_from"), "resume_T_over_rs": a.get("resume_T_over_rs"),
            "slice_dir": a.get("slice_dir"), "tag": a.get("tag"), "runner": J.get("runner")}


def sum_info(p):
    J = json.load(open(p))
    s = J.get("settings", {})
    rows = J.get("rows", [])
    t = [r.get("t_over_rs") for r in rows if isinstance(r, dict)]
    has_cut = False
    for r in rows[:3]:
        for ch in (r.get("per_channel") or {}).values():
            if isinstance(ch, dict) and any("cut" in k for k in ch.keys()):
                has_cut = True
    keep = {k: s.get(k) for k in ("record", "fields", "nsites", "channels", "kappas", "cuts_rs", "freeze_at_rs",
                                  "freeze_duration_rs", "init_state", "T_end_rs", "interp", "interp_sub", "lean", "save_at_rs")}
    return {"kind": "summary", "code": J.get("code"), "settings": keep, "n_rows": len(rows),
            "t_first": t[0] if t else None, "t_last": t[-1] if t else None, "per_channel_cut_keys": has_cut,
            "end_reason": J.get("end_reason"), "final": J.get("final")}


def main():
    out = sys.argv[1]
    res = []
    for p in sys.argv[2:]:
        e = {"path": p, "exists": os.path.exists(p)}
        if e["exists"]:
            e["size"] = os.path.getsize(p)
            e["md5"] = md5(p)
            try:
                if p.endswith(".json") and "summary" in os.path.basename(p):
                    e.update(sum_info(p))
                elif p.endswith(".json"):
                    e.update(rec_info(p))
            except Exception as ex:  # noqa
                e["error"] = repr(ex)
        res.append(e)
        print(json.dumps({k: v for k, v in e.items() if k != "runner"}, default=str)[:900])
    json.dump(res, open(out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
