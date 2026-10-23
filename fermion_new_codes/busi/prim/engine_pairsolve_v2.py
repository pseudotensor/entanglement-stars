"""engine_pairsolve_v2.py (busi, v36u perf; W2-P2 of final4/PERF/PERF_TRACKER.md, REVIEWA_perf_w2_v1 sec. 2 P2): engine_pairsolve_v1's persistent worker,
answering the dg step's Gauss-node field solves for engine_collapse_v36u's --dg-node-workers 3. Each worker is a spawned process that imports the
engine module named in cfg["engine"] (the running engine file's own module name), binds the same field module, the same kernel cache
(engine_fields_kcache_v1 over the same modules), OPT and SOLVE_TOL as the parent, and answers (eps, J, g, R, ell) with the parent's own
solve(eps, J, g, R, ell, None): the same code on the same arrays, so N, v and the collar charge are expected bit for bit equal to the parent's
serial solve (the pair-workers gate's precedent, bench1 B3; the gate checks it). Beyond v1 it returns the node's charge E = FLD.charge(info, g),
computed in the worker by the same function the parent's dgk_node_v36k calls. Reply: ("ok", N, v, failed, nfev, E or None, kcache hits, kcache misses)
or ("error", text)."""
import importlib
import os


def worker_main(conn, cfg):
    E = importlib.import_module(cfg["engine"])
    E.FLD = importlib.import_module(cfg["fld"])
    if cfg["kcache"]:
        import engine_fields_kcache_v1 as KC
        import engine_fields_v4 as _F4, engine_fields_v12 as _F12, engine_fields_v14 as _F14, engine_fields_v16 as _F16
        KC.install([_F4, _F12, _F14, _F16, E.FLD])
        stats = KC.STATS
    else:
        stats = {"hits": 0, "misses": 0}
    E.OPT.update(cfg["opt"]); E.SOLVE_TOL.update(cfg["tol"])
    conn.send(("ready", os.getpid()))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        eps, J, g, R, ell = msg
        h0, m0 = stats["hits"], stats["misses"]
        try:
            Nn, vn, Wn, info = E.solve(eps, J, g, R, ell, None)
            failed = info.get("failed")
            nfev = int(info["ivp"]["nfev"]) if info.get("ivp") else None
            En = None if failed else float(E.FLD.charge(info, g))
            conn.send(("ok", Nn, vn, failed, nfev, En, stats["hits"] - h0, stats["misses"] - m0))
        except Exception as ex:  # reported; the parent then solves this node itself
            conn.send(("error", "%s: %s" % (type(ex).__name__, str(ex)[:200])))
