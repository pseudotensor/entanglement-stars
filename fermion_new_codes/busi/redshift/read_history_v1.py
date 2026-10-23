"""busi REDSHIFT inputs (inbox #416): reader for the stored collapse histories in the inventory
(deliveries/busi/REDSHIFT_inputs_inventory_v1.md). Loads N(r, T) and v(r, T) of any listed history into ONE common format and
checks the md5s of the json and the fields npz against the registry. Read-only; it computes nothing beyond indexing.

Common format (a dict; --out writes the same keys to an npz):
  label, kind, rs (a), T (n_T,) in units of r_s, T_a (n_T,) in a, r (n_r,) in a, N (n_T, n_r), v (n_T, n_r),
  converged (n_T,) bool from the json record of the same clock, min_m_rec (n_T,) the record's min(N - |v|) (on the dense field),
  r_min_rec (n_T,), R_t (n_T,) the record's outer trapping root (nan where none), R_c (n_T,), E_over_E0 (n_T,),
  zeros (list of n_T lists: the record's trapping_zeros), site_fields ('averaged' or 'dense'), npz, json, npz_md5, json_md5, args.
kind:
  'site'    the npz rows N, v at the sites R = 1..nsites (a): the solve from the record's sources, KERNEL-AVERAGED over the sites
            when site_fields = averaged (every history listed), so not the point values the trapping zeros are defined on;
  'applied' N_applied, v_applied at the sites: the fields of the last self-consistency iteration's h_m (differ from 'site' at the
            convergence level);
  'nodes'   N_nodes, v_nodes at the field solver's nodes r_nodes (0, ell, 2 ell, ...; fixed across records): POINT values of the
            dense solution (engine_fields_v16: N_dense(r_nodes), v from W at the nodes); the closest stored thing to the dense fields.
  The dense fields themselves (any r) are not stored; they are re-solved from eps_tot, J_tot with the field module's
  solve_constraints_ivp (GATECOMP_iii_read_v1 sec. 2: reproduces the records' R_t to 4.8e-10 a). --sources adds eps_tot, J_tot,
  g, ell to the dict for that.
Usage (box: <internal>/python, nice -n 10):
  python read_history_v1.py --list
  python read_history_v1.py LABEL [--kind site|applied|nodes] [--out f.npz] [--no-md5] [--sources]
  python read_history_v1.py --smoke [--no-md5]       (every history, all three kinds; one line each)
  python read_history_v1.py --smoke-dense             (one record per history re-solved dense, against the record's R_t, min)
  in Python: from read_history_v1 import load_history; H = load_history("ISO_128_s4", kind="nodes")"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATHS = ("<internal>/registry_v1.json", os.path.join(HERE, "registry_v1.json"))


def registry(path=None):
    for p in ([path] if path else REGISTRY_PATHS):
        if p and os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError("registry_v1.json not found in %s" % (REGISTRY_PATHS,))


def md5(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def load_history(label, kind="site", check_md5=True, sources=False, reg=None):
    reg = reg or registry()
    if label not in reg:
        raise KeyError("unknown history %r; --list shows the labels" % label)
    e = reg[label]
    out = {"label": label, "kind": kind, "npz": e["npz"], "json": e["json"]}
    if check_md5:
        for k in ("npz", "json"):
            got = md5(e[k])
            if got != e[k + "_md5"]:
                raise ValueError("%s: %s md5 %s differs from the registry's %s" % (label, k, got, e[k + "_md5"]))
        out["md5_checked"] = True
    out["npz_md5"], out["json_md5"] = e["npz_md5"], e["json_md5"]
    J = json.load(open(e["json"]))
    args = J["args"]; rs = float(args["rs"])
    Z = np.load(e["npz"], allow_pickle=True)
    T_a = np.asarray(Z["T"], float)
    if kind == "site":
        r = np.asarray(Z["R"], float); N = np.asarray(Z["N"], float); v = np.asarray(Z["v"], float)
    elif kind == "applied":
        r = np.asarray(Z["R"], float); N = np.asarray(Z["N_applied"], float); v = np.asarray(Z["v_applied"], float)
    elif kind == "nodes":
        rn = Z["r_nodes"]; r = np.asarray(rn[0], float)
        for k, x in enumerate(rn):
            if x is None or len(x) != len(r) or not np.allclose(np.asarray(x, float), r):
                raise ValueError("%s: r_nodes of row %d differ from row 0 (not a fixed node grid)" % (label, k))
        N = np.vstack([np.asarray(x, float) for x in Z["N_nodes"]]); v = np.vstack([np.asarray(x, float) for x in Z["v_nodes"]])
    else:
        raise ValueError("kind must be site, applied or nodes")
    if N.shape != (len(T_a), len(r)) or v.shape != N.shape:
        raise ValueError("%s: shape N %s v %s against T %d r %d" % (label, N.shape, v.shape, len(T_a), len(r)))
    recs = {round(float(x["T"]), 9): x for x in J["records"] if isinstance(x, dict) and x.get("T") is not None}
    nan = float("nan")
    rows = [recs.get(round(float(t), 9), {}) for t in T_a]
    g = lambda key: np.array([(float(x[key]) if x.get(key) is not None else nan) for x in rows])
    out.update({"rs": rs, "T": T_a / rs, "T_a": T_a, "r": r, "N": N, "v": v,
                "converged": np.array([bool(x.get("converged", False)) for x in rows]),
                "has_record": np.array([bool(x) for x in rows]),
                "min_m_rec": g("min_N_minus_absv"), "r_min_rec": g("r_min_N_minus_absv"), "R_t": g("R_t"), "R_c": g("R_c"),
                "E_over_E0": g("E_over_E0"), "zeros": [list(x.get("trapping_zeros") or []) for x in rows],
                "site_fields": args.get("site_fields"), "args": {k: args.get(k) for k in ("rs", "nsites", "tau", "ell", "prep_ell",
                "kappa_sea", "m", "g", "mix", "dust_set", "site_fields", "reading", "reference") if k in args}})
    if sources:
        out.update({"eps_tot": np.asarray(Z["eps_tot"], float), "J_tot": np.asarray(Z["J_tot"], float), "R_sites": np.asarray(Z["R"], float),
                    "g": float(Z["g"]), "ell": float(Z["ell"])})
    return out


FM_DIR = "<internal>/code"   # engine_fields_v16.py 7a69faad (GATECOMP_iii's field module)
FM_MD5 = "7a69faadfad4a1e824a41107f474c69c"
_FM = {}


def field_module(fm_dir=FM_DIR, name="engine_fields_v16"):
    """The field solver of record, imported read-only (no bytecode written into the other lane's directory)."""
    if name not in _FM:
        p = os.path.join(fm_dir, name + ".py")
        if name == "engine_fields_v16" and md5(p) != FM_MD5:
            raise ValueError("%s md5 differs from %s" % (p, FM_MD5))
        sys.dont_write_bytecode = True
        sys.path.insert(0, fm_dir)
        _FM[name] = __import__(name)
    return _FM[name]


def dense_record(H, k, fm_dir=FM_DIR):
    """Re-solve record k's DENSE fields from its stored shell totals (needs load_history(..., sources=True)), on the route
    GATECOMP_iii_read_v1 used (solve_constraints_ivp, collar flag, support 1e-14). Returns callables N(r), v(r) for r >= 0 and the
    trapping zeros / min(N - |v|) of the dense field. This is a field solve, not a redshift."""
    FM = field_module(fm_dir)
    _, _, _, info = FM.solve_constraints_ivp(H["eps_tot"][k], H["J_tot"][k], H["g"], H["R_sites"], H["ell"], collar_policy="flag",
                                             support_rel=1e-14)
    if info.get("failed"):
        raise RuntimeError("ivp failed at T/rs %.4f: %s" % (H["T"][k], info.get("failed")))
    tz = FM.trapping_radius(None, None, H["R_sites"], info)
    return {"N": info["N_dense"], "v": info["v_dense"], "zeros": [float(z) for z in tz["zeros"]], "min": tz.get("min"),
            "r_min": tz.get("r_min"), "r_nodes": np.asarray(info["r_nodes"], float)}


def save(H, path):
    keep = {k: H[k] for k in ("T", "T_a", "r", "N", "v", "converged", "has_record", "min_m_rec", "r_min_rec", "R_t", "R_c", "E_over_E0")}
    keep.update({"label": H["label"], "kind": H["kind"], "rs": H["rs"], "npz_md5": H["npz_md5"], "json_md5": H["json_md5"],
                 "site_fields": str(H["site_fields"]), "args": json.dumps(H["args"]),
                 "zeros": np.array([json.dumps(z) for z in H["zeros"]])})
    np.savez_compressed(path, **keep)


def summary(H):
    m_stored = np.min(H["N"] - np.abs(H["v"]), axis=1)
    ok = np.isfinite(H["min_m_rec"])
    dm = float(np.nanmax(np.abs(m_stored[ok] - H["min_m_rec"][ok]))) if ok.any() else nan
    first = np.flatnonzero(np.isfinite(H["R_t"]))
    root = ("%.4f (R_t %.2f)" % (H["T"][first[0]], H["R_t"][first[0]])) if len(first) else "none"
    return ("%-16s %-7s T %.4f..%.4f n_T %4d n_r %5d r %.1f..%.1f | finite %s | records %d/%d, unconverged %d | first root %s | "
            "max|min_r(N-|v|) stored - record| %.2e" % (H["label"], H["kind"], H["T"][0], H["T"][-1], len(H["T"]), len(H["r"]), H["r"][0],
            H["r"][-1], bool(np.isfinite(H["N"]).all() and np.isfinite(H["v"]).all()), int(H["has_record"].sum()), len(H["T"]),
            int((~H["converged"] & H["has_record"]).sum()), root, dm))


nan = float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", nargs="?"); ap.add_argument("--kind", default="site", choices=("site", "applied", "nodes"))
    ap.add_argument("--out"); ap.add_argument("--no-md5", action="store_true"); ap.add_argument("--sources", action="store_true")
    ap.add_argument("--list", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--registry")
    ap.add_argument("--smoke-dense", action="store_true")
    a = ap.parse_args()
    reg = registry(a.registry)
    if a.list:
        for k, e in reg.items():
            print("%-16s %s\n%16s %s" % (k, e["desc"], "", e["npz"]))
        return
    if a.smoke:
        bad = 0
        for k in reg:
            for i, kind in enumerate(("site", "applied", "nodes")):
                try:
                    H = load_history(k, kind=kind, check_md5=(not a.no_md5 and i == 0), reg=reg)
                    print(("md5 OK  " if H.get("md5_checked") else "        ") + summary(H), flush=True)
                except Exception as ex:  # noqa
                    bad += 1; print("FAIL    %-16s %-7s %r" % (k, kind, ex), flush=True)
        print("smoke: %d histories x 3 kinds, %d failures" % (len(reg), bad))
        sys.exit(1 if bad else 0)
    if a.smoke_dense:
        # one record per history: the first-root record (else the last), re-solved dense, against the record's own R_t and min
        for k in reg:
            try:
                H = load_history(k, kind="site", check_md5=False, sources=True, reg=reg)
                fr = np.flatnonzero(np.isfinite(H["R_t"])); i = int(fr[0]) if len(fr) else len(H["T"]) - 1
                Dn = dense_record(H, i)
                Rt = Dn["zeros"][-1] if Dn["zeros"] else float("nan")
                print("dense   %-16s T %.4f  R_t dense %.6f record %.6f (diff %.1e)  min dense %.6f record %.6f (diff %.1e)" % (
                    k, H["T"][i], Rt, H["R_t"][i], abs(Rt - H["R_t"][i]) if np.isfinite(H["R_t"][i]) else float("nan"),
                    Dn["min"], H["min_m_rec"][i], abs(Dn["min"] - H["min_m_rec"][i])), flush=True)
            except Exception as ex:  # noqa
                print("FAIL    dense %-16s %r" % (k, ex), flush=True)
        return
    H = load_history(a.label, kind=a.kind, check_md5=not a.no_md5, sources=a.sources, reg=reg)
    print(summary(H))
    if a.out:
        save(H, a.out); print("wrote", a.out)


if __name__ == "__main__":
    main()
