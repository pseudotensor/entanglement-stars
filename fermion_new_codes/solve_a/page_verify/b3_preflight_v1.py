#!/usr/bin/env python3
"""b3_preflight_v1.py (solve, page_verify B3 kit): the dry run of a run_collapse_v19_from_slice_v1.py command line, which has no dry-run
mode of its own. Without propagating anything it
 (1) pops the runner's own flags (--slice, --min-free-gb) exactly as the runner does;
 (2) parses the rest with engine_collapse_v19's own argparse (its main() is called with run() and the GPU backend replaced by stubs);
 (3) repeats the runner's static gate checks: the stored run's args equal to this run's on the runner's key list, the slice's T/rs equal to
     --resume-T-over-rs, the slice dtype complex128, the stored fields holding a record at T_res, the slice's applied fields equal (0.0) to the
     stored ones at T_res, and the slice's channel set equal to the engine's; (the sources gate, 1e-8 on eps and J, needs the channel
     operators and runs at the engine's start; it is not repeated here);
 (4) reports the output paths the engine will write, the slice clocks and the disk space for them, and the md5 of every input.
Usage: b3_preflight_v1.py CODE_DIR -- <the runner's argument list>   (prints a json report; exit 1 on any failed check)"""
import hashlib, json, os, shutil, sys
import numpy as np


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    code = sys.argv[1]
    argv = sys.argv[sys.argv.index("--") + 1:]
    sys.path.insert(0, code)
    rep = {"checks": {}, "argv": list(argv)}

    def pop(key, default=None):
        if key in argv:
            i = argv.index(key); v = argv[i + 1]; del argv[i:i + 2]; return v
        return default
    SLICE = pop("--slice"); MINFREE = float(pop("--min-free-gb", "60"))
    import engine_collapse_v19 as E
    got = {}
    E.run = lambda a: got.setdefault("a", a)
    E.C5.backend = lambda gpu: None
    sys.argv = ["engine_collapse_v19.py"] + argv
    E.main()
    a = got["a"]
    rep["parsed_args"] = vars(a)
    ok = True

    def chk(name, cond, detail):
        nonlocal ok
        rep["checks"][name] = {"pass": bool(cond), "detail": detail}
        ok = ok and bool(cond)
    d, tag = os.path.split(a.resume_from)
    pj = os.path.join(d, "collapse_%s.json" % tag); pf = os.path.join(d, "collapse_%s_fields.npz" % tag)
    J = json.load(open(pj)); ra = J["args"]
    keys = ("m", "g", "rs", "R0_over_rs", "reading", "nsites", "ell", "kappa_sea", "tau", "half_offset", "site_fields", "shell", "reference",
            "c_LZ", "window_max", "stop_E", "stop_fields", "maxit", "mix", "collar_policy", "support_rel")
    diffs = {k: (ra.get(k), getattr(a, k)) for k in keys if not E._num_equal(ra.get(k), getattr(a, k))}
    chk("stored_args_equal", not diffs, diffs or "all %d keys equal" % len(keys))
    S = np.load(SLICE, allow_pickle=True)
    rs = a.rs; T_res = float(S["T"])
    chk("slice_clock", abs(T_res / rs - a.resume_T_over_rs) < 1e-9, "slice T/rs %.6f, requested %g" % (T_res / rs, a.resume_T_over_rs))
    chk("slice_dtype", str(S["dtype"]) == "complex128", str(S["dtype"]))
    Z = np.load(pf, allow_pickle=True)
    Tz = np.asarray(Z["T"], float); hits = np.where(np.abs(Tz - T_res) < 1e-9)[0]
    chk("stored_record_at_T_res", len(hits) > 0, "k_res %s of %d stored fields records" % (int(hits[0]) if len(hits) else None, len(Tz)))
    if len(hits):
        k = int(hits[0])
        dN = float(np.max(np.abs(np.asarray(S["N_applied"], float) - np.asarray(Z["N_applied"][k], float))))
        dv = float(np.max(np.abs(np.asarray(S["v_applied"], float) - np.asarray(Z["v_applied"][k], float))))
        chk("applied_fields_equal", dN == 0.0 and dv == 0.0, "max |dN| %.3e, |dv| %.3e (the runner requires 0.0)" % (dN, dv))
    kap = [int(x) for x in np.asarray(S["kappas"])]
    chk("slice_channels", all(("sea_%d" % k_ in S.files and "R2_%d" % k_ in S.files and "e_%d" % k_ in S.files) for k_ in kap), "kappas %s" % kap)
    rec = [r for r in J["records"] if r.get("E") is not None]
    chk("stored_records_reach_T_res", abs(rec[-1]["T"] - T_res) < 1e-9 or any(abs(r["T"] - T_res) < 1e-9 for r in rec),
        "stored records %g to %g (T/rs)" % (rec[0]["T_over_rs"], rec[-1]["T_over_rs"]))
    sl = [float(x) for x in a.slices_c128.split(",") if x.strip()]
    nrec = int(round((a.T_over_rs - a.resume_T_over_rs) * rs / a.tau))
    slice_bytes = os.path.getsize(SLICE)
    os.makedirs(a.out, exist_ok=True)
    if a.slice_dir:
        os.makedirs(a.slice_dir, exist_ok=True)
    fr_out = shutil.disk_usage(a.out).free / 1e9; fr_sl = shutil.disk_usage(a.slice_dir).free / 1e9 if a.slice_dir else None
    need = len(sl) * slice_bytes / 1e9
    chk("slice_disk", fr_sl is None or fr_sl - need > MINFREE, "slices %d x %.2f GB = %.1f GB; free %.1f GB; runner floor %.0f GB" % (len(sl), slice_bytes / 1e9, need, fr_sl or -1, MINFREE))
    chk("out_not_existing", not os.path.exists(os.path.join(a.out, "collapse_%s.json" % a.tag)), os.path.join(a.out, "collapse_%s.json" % a.tag))
    rep["plan"] = {"T_res_over_rs": T_res / rs, "T_end_over_rs": a.T_over_rs, "new_records": nrec, "slices_c128": sl,
                   "outputs": [os.path.join(a.out, "collapse_%s%s" % (a.tag, s)) for s in (".json", "_fields.npz", "_handoff.npz", "_spectra.npz")],
                   "slice_files": [os.path.join(a.slice_dir, "orbitals_%s_T%07.2f.npz" % (a.tag, x)) for x in sl] if a.slice_dir else []}
    rep["inputs_md5"] = {p: {"bytes": os.path.getsize(p), "md5": md5(p)} for p in (SLICE, pj, pf, os.path.join(d, "collapse_%s_handoff.npz" % tag))
                         if os.path.exists(p)}
    rep["code_md5"] = {f: md5(os.path.join(code, f)) for f in sorted(os.listdir(code)) if f in (
        "run_collapse_v19_from_slice_v1.py", "engine_collapse_v19.py", "engine_collapse_v17.py", "engine_collapse_v5.py", "engine_fields_v14.py",
        "engine_fields_v12.py", "engine_fields_v4.py", "engine_fields_v3.py", "engine_mixing_v1.py", "engine_ops_v1.py", "engine_ops_v2.py",
        "engine_stress_v1.py", "engine_vacflow_v1.py")}
    rep["all_pass"] = ok
    print(json.dumps(rep, indent=1, default=str))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
