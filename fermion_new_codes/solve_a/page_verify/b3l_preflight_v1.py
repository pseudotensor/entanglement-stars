#!/usr/bin/env python3
"""b3l_preflight_v1.py (solve, page_verify B3 light kit): the dry run of a page_history_v6q.py command line (v6q's --memory-figure-only
stops after its parse but prints the 8192-site figure only). Without propagating anything it
 (1) parses the argument list with v6q's own parse() (every flag checked against v6q's argparse; the D13 group included);
 (2) loads the record with v6q's own History (the chain of --nsites sites, the engine's sites R = 1..n_e, the vacuum exterior beyond);
 (3) checks: T_end within the record; the join window inside the engine's domain (v6q reads it on the engine's sites only); every cut and
     R_cut, R_collar inside the chain; the chain's light-crossing length against T_end (nsites a >= T_end, B2's ratio 1.28 printed beside);
     the kappa list; and reads the join departure at the first record, 100 r_s and the last;
 (4) prints the D13 memory figure at this --nsites (estimated from the shapes; B2's measured peaks were 18 to 24 GiB at 8192).
Usage: PYTHONPATH=<code_v6>:<code_d13> b3l_preflight_v1.py -- <the v6q argument list>   (json on stdout; exit 1 on a failed check)"""
import json, sys


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    import page_history_v6q as P
    import page_d13_readout_v1d as D13
    a = P.parse(argv)
    rep = {"argv": argv, "parsed": {k: v for k, v in vars(a).items() if isinstance(v, (int, float, str, bool, type(None)))}, "checks": {}}
    H = P.History(a.record, a.fields, a.nsites, g=a.g, fields_key=a.fields_key)
    rs = float(a.rs_nominal or H.rs_nom)
    T_last = float(H.T[-1]) / rs
    c = rep["checks"]
    c["record_loaded"] = {"ok": True, "n_records": len(H.T), "T_first_rs": float(H.T[0]) / rs, "T_last_rs": T_last, "engine_sites": H.ne, "chain_sites": H.nsites, "tau": H.tau, "g": H.g, "rs": rs}
    c["T_end_within_record"] = {"ok": a.T_end_rs is None or a.T_end_rs <= T_last + 1e-9, "T_end_rs": a.T_end_rs}
    lo, hi = [float(x) for x in a.join_rs.split(",")]
    c["join_window_inside_engine"] = {"ok": hi * rs <= H.ne + 1e-9 and lo < hi, "join_rs": a.join_rs, "engine_end_rs": H.ne / rs}
    cuts = [float(x) for x in a.cuts_rs.split(",")] + [a.R_cut_rs, a.R_collar_rs]
    c["cuts_inside_chain"] = {"ok": max(cuts) * rs < a.nsites, "max_cut_rs": max(cuts), "chain_rs": a.nsites / rs,
                              "cuts_beyond_engine_rs": [x for x in cuts if x * rs > H.ne]}
    Tend = a.T_end_rs if a.T_end_rs is not None else T_last
    c["chain_light_crossing"] = {"ok": a.nsites >= Tend * rs, "chain_over_T_end": a.nsites / (Tend * rs), "B2_ratio": 8192 / (100 * 64.0)}
    ks = P.parse_kappas(a.kappas)
    c["kappas"] = {"ok": len(ks) > 0 and min(ks) >= 1, "kappas": ks}
    jl = {}
    for t in (float(H.T[0]) / rs, 100.0, T_last):
        k = min(range(len(H.T)), key=lambda i: abs(float(H.T[i]) / rs - t))
        jN, jv = H.join_check(k, lo * rs, hi * rs)
        jl["%.3f" % (float(H.T[k]) / rs)] = {"max_dN": jN, "max_rel_dv": jv}
    c["join_departure"] = {"ok": True, "at": jl}
    fig = D13.memory_figure(a.nsites, None, a.chunk, a.R_cut_rs, rs, getattr(a, "source_max", 128))
    rep["memory_figure_GiB"] = {k: round(v, 2) for k, v in fig.items() if k.endswith("_GiB")}
    rep["ok"] = all(v.get("ok") for v in c.values())
    print(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
