"""cmp_v36w2_selftest.py (busi SYM-W1): selftest_v36w2_box.sh's default run (engine_collapse_v36w2 under the harness's name) against v36v's stored
prim_vii self-test output (<internal>/prim_vii), bitwise by cmp_v36v_busi.compare (records, head, npz arrays, slices, logs
with the walls masked); and the log run (--hbar-prop-log on) against the same with its "v36w2 cheb" lines and the hbar_prop_v36w2 entry keys removed."""
import importlib.util, json, os, sys, shutil
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("C", "<internal>/cmp_v36v_busi.py")
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
ST = "<internal>/selftest"; REF = "<internal>/prim_vii"
C.ROOTS = tuple(C.ROOTS) + (ST,)
ok1 = C.compare("(defaults) v36w2 def vs v36v prim_vii", os.path.join(ST, "def"), "fz", REF, "fz", os.path.join(ST, "def", "prim.log"), os.path.join(REF, "prim.log"))
# the log run: strip its additions into a copy, then the same bitwise compare
src = os.path.join(ST, "log"); dst = os.path.join(ST, "log_stripped")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
J = json.load(open(os.path.join(dst, "collapse_fz.json")))
n = 0
for r in J["records"]:
    for s_ in ((r.get("dg_v36f") or {}).get("substeps") or []):
        if "hbar_prop_v36w2" in s_:
            del s_["hbar_prop_v36w2"]; n += 1
json.dump(J, open(os.path.join(dst, "collapse_fz.json"), "w"))
L = [l_ for l_ in open(os.path.join(dst, "prim.log"), errors="replace") if "v36w2 cheb" not in l_]
m = sum(1 for l_ in open(os.path.join(dst, "prim.log"), errors="replace") if "v36w2 cheb" in l_)
open(os.path.join(dst, "prim.log"), "w").writelines(L)
print("log run: %d hbar_prop_v36w2 entries and %d 'v36w2 cheb' lines removed" % (n, m))
ok2 = C.compare("(--hbar-prop-log on, its additions removed) v36w2 log vs v36v prim_vii", dst, "fz", REF, "fz", os.path.join(dst, "prim.log"), os.path.join(REF, "prim.log"))
print("SUMMARY:", "PASS" if (ok1 and ok2) else "FAIL")
