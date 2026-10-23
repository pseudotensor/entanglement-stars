#!/usr/bin/env python
"""m325_dry_v1.py (busi f52b1152, 2026-10-20; inbox #325): the dry check of one m325 launch, no solve and no GPU. Imports the engine named on the
command line, replaces its backend call and run() so that main() parses the launch's own argument list (the engine's argparse and its validation
lines, the v36g completion checks included) and then runs the file-level part of version 34 (a)'s resume (resume_slice_v34, read from the engine):
the physics-argument check against the resume run's json (with v36g's relaxed rule when the engine has it), the slice's dtype, a json record at
the slice's clock with its E, the fields npz row at that clock, the slice's applied fields against that row, and every channel's arrays present
in the slice. For --completion frozen-sources --completion-charge matched it reads the target run's E(T) as v36g does and counts the targets at
the clocks the run will solve (missing targets would take the gated FALLBACK). Prints PASS or FAIL lines and "DRY: n checks, m failed".
Usage: python m325_dry_v1.py ENGINE.py [the engine's arguments ...]"""
import importlib.util
import json
import os
import sys

import numpy as np

FAILS = []


def chk(ok, msg):
    print(("PASS: " if ok else "FAIL: ") + msg, flush=True)
    if not ok:
        FAILS.append(msg)


def dry(E, a):
    rs = a.rs
    print("parsed: tag %s, out %s, T/rs end %g, tau %g, mix %s, slices_c128 %s" % (a.tag, a.out, a.T_over_rs, a.tau, a.mix, a.slices_c128), flush=True)
    comp = getattr(a, "completion", None)
    if comp is not None:
        print("parsed completion: %s from %g, radius %s r_s, ell_in %g, charge %s, completion_run %s" % (
            comp, a.completion_from_T_over_rs, a.completion_radius_over_rs, a.completion_ell_in, a.completion_charge, a.completion_run), flush=True)
    chk(not os.path.exists(a.out), "the output directory %s does not exist yet" % a.out)
    rdir, rtag = os.path.split(a.resume_run)
    jp = os.path.join(rdir, "collapse_%s.json" % rtag)
    J = json.load(open(jp))
    bad = []
    for k_ in ("m", "g", "rs", "nsites", "ell", "kappa_sea", "tau", "reference", "site_fields", "c_LZ", "mix", "massless_ball", "fields_v17"):
        if J["args"].get(k_) != getattr(a, k_):
            if hasattr(E, "resume_args_relaxed_v36g") and E.resume_args_relaxed_v36g(a, J["args"], k_):
                continue
            bad.append((k_, J["args"].get(k_), getattr(a, k_)))
    chk(not bad, "version 34 (a) argument check against %s: %s" % (os.path.basename(jp), bad or "every key equal"))
    Z = np.load(a.resume_slice)
    dt = E.slice_dtype_v36g(a, Z) if hasattr(E, "slice_dtype_v36g") else str(Z["dtype"])
    chk(dt == "complex128", "the slice's dtype %s" % dt)
    Ts = float(Z["T"])
    recs = [r for r in J["records"] if r.get("T") is not None and r["T"] <= Ts + 1e-9]
    chk(bool(recs) and abs(recs[-1]["T"] - Ts) < 1e-9 and recs[-1].get("E") is not None,
        "the run's json holds an accepted record at the slice's clock T/rs %.4f (%d records to it, %d in all)" % (Ts / rs, len(recs), len(J["records"])))
    Zf = np.load(os.path.join(rdir, "collapse_%s_fields.npz" % rtag), allow_pickle=True)
    Tz = np.asarray(Zf["T"], float); ks = int(np.argmin(np.abs(Tz - Ts)))
    chk(abs(Tz[ks] - Ts) < 1e-9, "the fields npz holds a row at the slice's clock (row %d of %d)" % (ks, len(Tz)))
    fchk = max(float(np.max(np.abs(np.asarray(Z["N_applied"], float) - Zf["N_applied"][ks]))), float(np.max(np.abs(np.asarray(Z["v_applied"], float) - Zf["v_applied"][ks]))))
    chk(fchk == 0.0, "the slice's applied fields against the stored row: max |diff| %.1e" % fchk)
    miss = [k for k in range(1, a.kappa_sea + 1) for p in ("e_", "sea_", "dust_", "f_", "R2_", "mask_") if "%s%d" % (p, k) not in Z.files]
    chk(not miss, "every channel's e, sea, dust, f, R2 and mask arrays are in the slice (%d channels)%s" % (a.kappa_sea, "" if not miss else ": missing %s" % miss))
    for nm in ("handoff", "spectra"):
        p_ = os.path.join(rdir, "collapse_%s_%s.npz" % (rtag, nm))
        print("note: %s npz %s" % (nm, "present" if os.path.exists(p_) else "ABSENT (the engine skips it)"), flush=True)
    print("the run continues at T/rs %.4f and ends at %g: %d records" % ((Ts + a.tau) / rs, a.T_over_rs, int(round((a.T_over_rs * rs - Ts) / a.tau))), flush=True)
    if comp == "frozen-sources" and a.completion_charge == "matched":
        cr = a.completion_run or a.resume_run
        cj = os.path.join(os.path.dirname(cr), "collapse_%s.json" % os.path.basename(cr))
        if os.path.exists(cj):
            St = E.load_stored_E_v36g(cr, rs)
            need = [round((Ts + i * a.tau) / rs, 6) for i in range(1, int(round((a.T_over_rs * rs - Ts) / a.tau)) + 1)]
            have = sum(1 for t in need if t in St)
            chk(have == len(need), "the matched targets in %s: %d of the %d clocks the run solves" % (os.path.basename(cj), have, len(need)))
        else:
            print("note: the matched target run %s is not there yet (own56 produces it); with --completion-run given the check runs once it exists" % cj, flush=True)
    if comp is not None:
        print("note: the completion arms at the first ungated record at or after T/rs %g; the arming record is solved as the record's own, then re-solved" % a.completion_from_T_over_rs, flush=True)


def main():
    eng = sys.argv[1]
    sys.path.insert(0, os.path.dirname(os.path.abspath(eng)))
    spec = importlib.util.spec_from_file_location("m325_engine", eng)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    E.C5.backend = lambda gpu: print("dry: backend(%s) not called" % gpu, flush=True)
    E.run = lambda a: dry(E, a)
    sys.argv = [eng] + sys.argv[2:]
    E.main()
    print("DRY: %d failed%s" % (len(FAILS), "" if not FAILS else ": " + "; ".join(FAILS)), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
