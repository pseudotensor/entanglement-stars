#!/usr/bin/env python
"""resume_view_v1.py (busi lane, GATECOMP kit (ii) controls, inbox #400): a restart VIEW of a parent run for a control that changes one
evolution setting at a stored slice (ReviewA_gatecomp_v1.md sec. 2 (ii): "tau/r_s = 1/32 restarted from the 8a member's T = 9 slice; the
plain mix on the same stretch"; the point-sampled control; ReviewB_gatecomp_v1.md's local re-integration).

Why: the engine's version-34 restart (engine_collapse_v36nq_rd13mf.py resume_slice_v34) refuses a call whose m, g, rs, nsites, ell,
kappa_sea, tau, reference, site_fields, c_LZ, mix, massless_ball or fields_v17 differ from the parent json's args. A control that changes
tau, mix or site_fields at a restart is therefore refused by design. This script writes NEW_DIR/collapse_TAG.json = the parent's json with
only the named args replaced and one head entry added, head["resume_view_v1"] = {parent json path and md5, the changed keys (old, new),
the reason, this script's md5}; the parent's _fields, _handoff, _spectra, _s11 npz are copied unchanged (md5 printed; a hard link with
--link only for a parent that has ENDED, refused in code otherwise, since a running parent rewrites its npz in place). --at T refuses unless the parent's json holds a
record and its fields npz a row at T/r_s = T (the engine's restart needs both; the fields npz is written every 5 records, so make the view
once the parent is past T + 5 tau).
The engine is NOT changed. It copies the parent head into the control's own json (head = dict(J["head"])), so the control's run carries
this entry. Only tau, mix and site_fields may be changed (each is a control's declared change); anything else is refused.

Checked on the engine source (busi/notes/GATECOMP_kits_v1.md sec. 4): the record at T is computed from the state at T (sources, the
self-consistent fields, the continued reference), and the state is then propagated by exp(-i h(T) tau); the restart's first propagation is
the same operation with the call's tau, so a tau change at a restart is what a fresh run at the new tau does from that state. The loop's
step index continues from the parent's record index, so the end clock must be set by --segment-end-T-over-rs (the kit's lines do). The M1
Simpson diagnostic's steps (--m1-simpson-steps auto) are step indices and land where they land (a diagnostic).

THIS BYPASSES A SAFETY CHECK ON PURPOSE. It is staged for the lead's and reviewer A's word; the fallback is the control fresh from T = 0.

  resume_view_v1.py PARENT_DIR/PARENT_TAG NEW_DIR --set tau=8 [--set mix=plain] [--set site_fields=dense] --reason TEXT [--at T] [--link] [--check-only]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

CHECKED = ("m", "g", "rs", "nsites", "ell", "kappa_sea", "tau", "reference", "site_fields", "c_LZ", "mix", "massless_ball", "fields_v17")
ALLOWED = {"tau": float, "mix": str, "site_fields": str}
CHOICES = {"mix": ("plain", "anderson", "damped", "newton"), "site_fields": ("averaged", "dense")}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent"); ap.add_argument("new_dir")
    ap.add_argument("--set", action="append", default=[], help="KEY=VALUE, KEY in tau, mix, site_fields")
    ap.add_argument("--reason", required=True); ap.add_argument("--link", action="store_true")
    ap.add_argument("--at", type=float, default=None, help="the slice clock T/r_s: the json record and the fields row must exist")
    ap.add_argument("--check-only", dest="check_only", action="store_true", help="print the view's changes, write nothing")
    a = ap.parse_args()
    pdir, ptag = os.path.split(a.parent.rstrip("/"))
    pj = os.path.join(pdir, "collapse_%s.json" % ptag)
    J = json.load(open(pj))
    changed = {}
    for s in a.set:
        k, v = s.split("=", 1)
        if k not in ALLOWED:
            sys.exit("refused: %s is not a declared control change (allowed: %s)" % (k, ", ".join(ALLOWED)))
        val = ALLOWED[k](v)
        if k in CHOICES and val not in CHOICES[k]:
            sys.exit("refused: %s=%s not in %s" % (k, val, CHOICES[k]))
        if J["args"].get(k) == val:
            sys.exit("refused: %s is already %r in the parent" % (k, val))
        changed[k] = [J["args"].get(k), val]
    if not changed:
        sys.exit("refused: nothing to change")
    if a.at is not None:
        import numpy as np
        rs = float(J["args"]["rs"])
        hasrec = any(abs(r["T"] / rs - a.at) < 1e-9 and r.get("E") is not None for r in J["records"])
        Zf = np.load(os.path.join(pdir, "collapse_%s_fields.npz" % ptag), allow_pickle=True)
        hasrow = bool(np.any(np.abs(np.asarray(Zf["T"], float) / rs - a.at) < 1e-9))
        print("at T/rs %.4f: json record %s, fields row %s" % (a.at, hasrec, hasrow))
        if not (hasrec and hasrow):
            sys.exit("refused: the parent does not yet hold the record and the fields row at T/rs %.4f" % a.at)
    jm = md5(pj)
    entry = {"parent_json": pj, "parent_json_md5": jm, "changed": changed, "reason": a.reason,
             "script": os.path.abspath(__file__), "script_md5": md5(os.path.abspath(__file__)),
             "note": "the engine's v34 restart checks %s against the call; this view declares the control's change; records, rows and slice are the parent's" % ", ".join(CHECKED)}
    print("parent %s (md5 %s): %d records to T/rs %.4f; changing %s" % (pj, jm, len(J["records"]), J["records"][-1]["T"] / float(J["args"]["rs"]), json.dumps(changed)))
    if a.check_only:
        print("CHECK ONLY: nothing written"); return
    if os.path.exists(a.new_dir) and os.listdir(a.new_dir):
        sys.exit("refused: %s exists and is not empty" % a.new_dir)
    if a.link:   # ReviewA_resview_v1 (3): a hard link only to a parent that has ended (a running parent rewrites its npz in place)
        rs_ = float(J["args"]["rs"]); last = J["records"][-1]["T"] / rs_ if J["records"] else -1.0
        seg = J["args"].get("segment_end_T_over_rs"); tend = float(seg) if seg is not None else float(J["args"]["T_over_rs"])
        ended = any(k.startswith("end_") for k in J) or last >= tend - 1e-9
        if not ended:
            sys.exit("refused: --link needs a parent that has ended (last record T/rs %.4f, its end %.4f, no end_* entry); copy instead" % (last, tend))
    os.makedirs(a.new_dir, exist_ok=True)
    for k, (old, new) in changed.items():
        J["args"][k] = new
    J["head"]["resume_view_v1"] = entry
    out = os.path.join(a.new_dir, "collapse_%s.json" % ptag)
    with open(out, "w") as f:
        json.dump(J, f)
    for suf in ("fields", "handoff", "spectra", "s11"):
        src = os.path.join(pdir, "collapse_%s_%s.npz" % (ptag, suf))
        if os.path.exists(src):
            dst = os.path.join(a.new_dir, os.path.basename(src))
            if a.link:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
            entry.setdefault("npz_md5", {})[os.path.basename(src)] = md5(dst)
    entry["view_json_md5"] = md5(out)
    with open(os.path.join(a.new_dir, "RESUME_VIEW.json"), "w") as f:
        json.dump(entry, f, indent=1)
    print("written %s (md5 %s) and the npz (%s); RESUME_VIEW.json beside: %s" % (out, entry["view_json_md5"], "hard links" if a.link else "copies", json.dumps(entry.get("npz_md5"))))


if __name__ == "__main__":
    main()
