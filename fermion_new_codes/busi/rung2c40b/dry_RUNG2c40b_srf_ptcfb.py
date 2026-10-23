# dry check (busi #339): the continuation line's argparse and the resume's checks, read-only, no GPU, nothing written.
# Usage on R1: cd <internal>/code_v36d && venv python - < this file  (QLINE embedded below)
import sys, os, json, shlex, hashlib
import numpy as np
QLINE = open(os.environ["QLINE_FILE"]).read() if os.environ.get("QLINE_FILE") else None
QLINE = QLINE or os.environ["QLINE"]
L = "<internal>/ladders"; RT = "RUNG2c40b_srf"; PT = "RUNG2c40b_srf_ptcfb"
cmd = QLINE.split("engine_collapse_v36z.py ", 1)[1].split(" >", 1)[0]
cmd = cmd.replace("${RT}", RT).replace("$RT", RT).replace("$PT", PT).replace("$L", L)
argv = shlex.split(cmd)
fails = []
def ok(c, msg):
    print(("OK   " if c else "FAIL ") + msg, flush=True)
    if not c: fails.append(msg)
sys.path.insert(0, "<internal>/code_v36d")
import importlib
E = importlib.import_module("engine_collapse_v36z")
ok(hashlib.md5(open(E.__file__, "rb").read()).hexdigest() == "924eefed596c9a32fe029294a2dae498", "engine md5 924eefed (%s)" % E.__file__)
E.C5.backend = lambda g: None
seen = {}
def dry_run(a):
    seen["a"] = a
E.run = dry_run
sys.argv = ["engine_collapse_v36z.py"] + argv
E.main()
a = seen["a"]
ok(True, "argparse and main()'s checks passed")
ok(a.nk_ptc == "fallback" and E.PTC.get("fallback") and E.PTC.get("delta0") == 1.0 and not E.PTC.get("test"), "PTC fallback on, delta0 1: %s" % {k: E.PTC[k] for k in ("on", "fallback", "test", "delta0", "one_cache")})
ok(E.SRF.get("on") and E.SRF.get("max_halvings") == 3 and E.SRF.get("above") == 1e-3, "SRF on, 3 halvings, above %s, recover %s" % (E.SRF.get("above"), E.SRF.get("recover")))
ok(a.end_above == 1e-3 and a.nk_plain_fallback == "on" and a.nk_host_release == "on", "end-above 0.001, plain fallback on, host release on")
# the resume's checks (resume_slice_v34 without the GPU eigendecomposition)
rdir, rtag = os.path.split(a.resume_run)
jp = os.path.join(rdir, "collapse_%s.json" % rtag)
ok(hashlib.md5(open(jp, "rb").read()).hexdigest() == "bb19aba7f5f4e8e3f32424d12218497d", "resume json md5 bb19aba7 (%s)" % jp)
J = json.load(open(jp))
for k_ in ("m", "g", "rs", "nsites", "ell", "kappa_sea", "tau", "reference", "site_fields", "c_LZ", "mix", "massless_ball", "fields_v17"):
    ok(J["args"].get(k_) == getattr(a, k_), "resume arg %s: run %r, line %r" % (k_, J["args"].get(k_), getattr(a, k_)))
diff = sorted(k for k in set(J["args"]) | set(vars(a)) if J["args"].get(k, "<absent>") != vars(a).get(k, "<absent>"))
print("     args differing from the run's stored args (informational): %s" % ", ".join("%s: %r -> %r" % (k, J["args"].get(k, "<absent>"), vars(a).get(k, "<absent>")) for k in diff))
Z = np.load(a.resume_slice)
ok(str(Z["dtype"]) == "complex128", "slice dtype %s" % Z["dtype"])
Ts = float(Z["T"]); rs = a.rs
ok(abs(Ts / rs - 12.0) < 1e-12, "slice clock T/rs %.4f" % (Ts / rs))
recs = [r for r in J["records"] if r.get("T") is not None and r["T"] <= Ts + 1e-9]
ok(bool(recs) and abs(recs[-1]["T"] - Ts) <= 1e-9 and recs[-1].get("E") is not None, "accepted record at the slice clock (record %d of %d in the json; later records dropped by the resume: %s)" % (len(recs) - 1, len(J["records"]), ", ".join("%.4f" % (r["T"] / rs) for r in J["records"] if r["T"] > Ts + 1e-9)))
ok(not any(r.get("unconverged") for r in recs), "no unconverged record at or before the slice")
Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
Tz = np.asarray(Zf["T"], float); ks = int(np.argmin(np.abs(Tz - Ts)))
ok(abs(Tz[ks] - Ts) <= 1e-9 and ks == len(recs) - 1, "fields npz row %d at the slice clock (rows %d)" % (ks, len(Tz)))
fchk = max(float(np.max(np.abs(np.asarray(Z["N_applied"], float) - Zf["N_applied"][ks]))), float(np.max(np.abs(np.asarray(Z["v_applied"], float) - Zf["v_applied"][ks]))))
ok(fchk == 0.0, "slice applied fields against the stored row %.1e" % fchk)
for k in range(1, a.kappa_sea + 1):
    miss = [p % k for p in ("e_%d", "sea_%d", "dust_%d", "f_%d", "R2_%d", "mask_%d") if (p % k) not in Z.files]
    if miss: ok(False, "slice keys missing %s" % miss)
ok(True, "slice holds e/sea/dust/f/R2/mask for channels 1..%d" % a.kappa_sea)
for fn in ("handoff", "spectra", "s11"):
    p_ = os.path.join(rdir, "collapse_%s_%s.npz" % (rtag, fn))
    if os.path.exists(p_):
        Tx = np.asarray(np.load(p_, allow_pickle=True)["T"], float)
        ok(int(np.sum(Tx <= Ts + 1e-9)) == ks + 1, "%s npz rows kept %d of %d" % (fn, int(np.sum(Tx <= Ts + 1e-9)), len(Tx)))
    else:
        print("     %s npz absent (the resume skips it)" % fn)
ok("N_cinit" in Zf.files and "v_cinit" in Zf.files and "E0" in J["head"], "fields npz N_cinit/v_cinit and head E0 present")
ok(not os.path.exists(os.path.join(a.out, "collapse_%s.json" % a.tag)), "no json yet at %s (never overwritten)" % a.out)
ok(not os.path.exists(a.slice_dir) or not os.listdir(a.slice_dir), "slice dir %s absent or empty" % a.slice_dir)
ok(os.path.abspath(a.out) != os.path.abspath(rdir), "out dir differs from the resumed run's dir")
print("DRY_OK" if not fails else "DRY_FAIL (%d)" % len(fails))
