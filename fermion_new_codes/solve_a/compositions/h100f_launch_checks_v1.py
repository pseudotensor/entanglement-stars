#!/usr/bin/env python
"""h100f_launch_checks_v1.py (solve 8a7b5274, COMP1 h100-f checks). Adapted from box <internal>/launch_engine_test_v1.py:
the same argument list rebuilt from the control continuation json (same skip set), plus --gpu 0 (the control's own value; the box test ran on CPU),
14 threads, paths under <internal>/comp1. Writes three run scripts:
  check1_v3.sh, check1_v8.sh : engine_twofield_v3 / v8 (implicit default) from the v8-written control slice T/r_s 60, segment end 60.125, no slices.
  check2_v3.sh               : engine_twofield_v3 from the control's v3-written T/r_s 50 slice (pod 1), segment end 60, orbitals slices at 55 and 60.
and run_all.sh running them one after the other.
Attempt 2 (21:15 UTC): every engine runs through h100f_npz_cache_wrapper_v1.py (attempt 1 stopped at the restart before running out of memory)."""
import json, os

W = "<internal>/comp1"
TAG = "C32_s4_ell4_tau2_twofield_v3_off"
CTLJ = W + "/in/R2run/collapse_%s.json" % TAG
RUN = W + "/in/R2run/" + TAG
S60 = W + "/in/slices/orbitals_%s_T0060.00.npz" % TAG
S50 = W + "/in/slices/orbitals_%s_T0050.00.npz" % TAG
PY = "<internal>/python"
ENG = W + "/code/compC"
EDIR = W + "/code/compC_engine_v36"
WRAP = W + "/h100f_npz_cache_wrapper_v1.py"   # attempt 2: I/O-only cache of the *_fields.npz members (see that file)

a = dict(json.load(open(CTLJ))["args"])
skip = {"gpu", "out", "tag", "resume_slice", "resume_run", "segment_end_T_over_rs", "slices_c128", "slices_c64", "slices_light_c128", "slice_dir",
        "reference_solve", "resume_from", "resume_T_over_rs", "resume_grandparent", "light_follow", "benchmark"}
flags = []
for k, v in a.items():
    if k in skip or v is None:
        continue
    f = "--" + k.replace("_", "-")
    if isinstance(v, bool):
        if v:
            flags.append(f)
    else:
        flags.append("%s %s" % (f, str(v)))
ENV = ("export OMP_NUM_THREADS=14 OPENBLAS_NUM_THREADS=14 MKL_NUM_THREADS=14 NUMEXPR_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 "
       "BUSI_COMPC_DIR=%s BUSI_ENGINE_DIR=%s BUSI_PAGE_DIR=%s" % (ENG, EDIR, EDIR))
jobs = [("check1_v3", "v3", S60, 60.125, "''"), ("check1_v8", "v8", S60, 60.125, "''"), ("check2_v3", "v3", S50, 60.0, "55,60")]
for name, ver, sl, seg, sc in jobs:
    od = os.path.join(W, "out", name)
    os.makedirs(od, exist_ok=True)
    cmd = ("cd %s && %s && %s -I -B %s %s/engine_twofield_%s.py %s --gpu 0 --resume-slice %s --resume-run %s --segment-end-T-over-rs %g "
           "--slices-c128 %s --slices-c64 '' --slices-light-c128 '' --slice-dir %s --out %s --tag %s > %s/run.log 2>&1; echo EXIT $? >> %s/run.log"
           % (od, ENV, PY, WRAP, ENG, ver, " ".join(flags), sl, RUN, seg, sc, od, od, TAG, od, od))
    open(os.path.join(W, "%s.sh" % name), "w").write("#!/bin/bash\n" + cmd + "\n")
open(os.path.join(W, "run_all.sh"), "w").write("#!/bin/bash\ncd %s\nfor j in check1_v3 check1_v8 check2_v3; do echo START $j $(date -u) >> run_all.log; bash $j.sh; echo END $j $(date -u) $(tail -1 out/$j/run.log) >> run_all.log; done\necho ALLDONE >> run_all.log\n" % W)
print("flags:", " ".join(flags))
