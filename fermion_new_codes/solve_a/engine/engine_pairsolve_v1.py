"""engine_pairsolve_v1.py (solve builder 5; REVIEWA_gpu_economics_v1 sec. 5, line 4): the persistent worker processes behind
engine_collapse_v36b's --pair-workers option. Each worker is a spawned process (not a thread: the GIL; not a per-product fork: start-up
about the cost of a solve) that imports engine_collapse_v36b as a module, sets the same field module (engine_fields_v16, or engine_fields_v17
under --fields-v17), the same kernel cache (engine_fields_kcache_v1 over the same modules), OPT and SOLVE_TOL as the parent, and then
answers requests (eps, J, g, R, ell) with the parent's own solve(eps, J, g, R, ell, None): the same code on the same inputs, so the result is
expected bit for bit equal to the parent's serial solve (the gate checks it). It returns (N, v, failed, nfev, cache hits, cache misses)."""
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
            conn.send(("ok", Nn, vn, failed, nfev, stats["hits"] - h0, stats["misses"] - m0))
        except Exception as ex:  # reported; the parent then computes this pair serially
            conn.send(("error", "%s: %s" % (type(ex).__name__, str(ex)[:200])))
