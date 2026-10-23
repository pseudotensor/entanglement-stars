#!/usr/bin/env python
"""resume_check_v1.py (busi lane, inbox #293, 2026-10-19): on the BOX, check that a mirrored resume state holds every file the engine's
resume route needs, together and consistent, by repeating the engine's own resume checks (read-only; never writes).

  slice <dir> <tag> <slice.npz>   ladder engines v36d..v36q/p and twofield v8/v9b/v9c: --resume-slice <dir>/<slice> --resume-run <dir>/<tag>
                                  (resume_slice_v34: complex128 slice; json with an accepted record at the slice clock; fields npz with a row
                                  there and the slice's applied fields; handoff and spectra npz present; twofield: the light state sea_L<k> in
                                  the same slice for every light kappa)
  v6o <dir> <prefix> [inputs..]   page_history_v6o --resume: <prefix>_state.pt loadable with k, rows, channels{Phi,F}; the record/fields inputs exist
  qh <dir> <member>               qh_branch_witness_v1b --stage channels: qh_member<m>_channels.npz (kappa list), qh_member<m>.json, _arrays.npz
Prints KEY value lines: RESUMABLE yes|no, WHY ..., FINAL yes|no, LAST <clock>.
"""
import json
import os
import sys

import numpy as np


def out(k, v):
    print("%s %s" % (k, v))


def check_slice(d, tag, sl):
    why = []
    p = os.path.join(d, sl)
    try:
        Z = np.load(p)
        dt = str(Z["dtype"]); Ts = float(Z["T"])
        N_s = np.asarray(Z["N_applied"], float); v_s = np.asarray(Z["v_applied"], float)
        keys = set(Z.files)
    except Exception as e:
        out("RESUMABLE", "no"); out("WHY", "slice unreadable: %r" % (e,)); out("FINAL", "no"); return
    if dt != "complex128":
        why.append("slice dtype %s, not complex128" % dt)
    jp = os.path.join(d, "collapse_%s.json" % tag)
    try:
        J = json.load(open(jp))
    except Exception as e:
        out("RESUMABLE", "no"); out("WHY", "run json unreadable: %r" % (e,)); out("FINAL", "no"); return
    A = J.get("args", {})
    rs = float(A.get("rs") or 1.0)
    recs = [r for r in J.get("records", []) if r.get("T") is not None and r["T"] <= Ts + 1e-9]
    if not recs or abs(recs[-1]["T"] - Ts) > 1e-9 or recs[-1].get("E") is None:
        why.append("json has no accepted record at the slice clock T/rs %.4f" % (Ts / rs))
    try:
        Zf = np.load(os.path.join(d, "collapse_%s_fields.npz" % tag), allow_pickle=True)
        Tz = np.asarray(Zf["T"], float); ks = int(np.argmin(np.abs(Tz - Ts)))
        if abs(Tz[ks] - Ts) > 1e-9:
            why.append("fields npz has no row at the slice clock")
        else:
            fchk = max(float(np.max(np.abs(N_s - Zf["N_applied"][ks]))), float(np.max(np.abs(v_s - Zf["v_applied"][ks]))))
            out("FIELDS_VS_SLICE", "%.3e" % fchk)
            if fchk > 1e-8:
                why.append("fields npz applied fields differ from the slice's by %.2e" % fchk)
    except Exception as e:
        why.append("fields npz unreadable: %r" % (e,))
    for fn in ("handoff", "spectra"):
        q = os.path.join(d, "collapse_%s_%s.npz" % (tag, fn))
        if not os.path.exists(q):
            why.append("%s npz missing" % fn)
        else:
            try:
                Zx = np.load(q, allow_pickle=True); Tx = np.asarray(Zx["T"], float)
                if not np.any(Tx <= Ts + 1e-9):
                    why.append("%s npz has no row at or below the slice clock" % fn)
            except Exception as e:
                why.append("%s npz unreadable: %r" % (fn, e))
    lk = str(A.get("light_kappas") or "").strip()
    if lk and lk not in ("None", "none", "off", ""):
        for k in [int(x) for x in lk.split(",") if x.strip()]:
            if "sea_L%d" % k not in keys:
                why.append("slice lacks the light state sea_L%d (twofield resume refuses)" % k)
    # the engine's ladder-kappa keys
    nk = [k for k in keys if k.startswith("sea_") and not k.startswith("sea_L")]
    if not nk:
        why.append("slice holds no sea_<kappa> arrays")
    allrecs = [r for r in J.get("records", []) if r.get("T") is not None]
    lastT = allrecs[-1]["T"] / rs if allrecs else float("nan")
    tend = A.get("T_over_rs")
    final = (tend is not None and allrecs and lastT >= float(tend) - 1e-9)
    out("SCRIPT", J.get("script", "?"))
    out("SLICE_T", "%.4f" % (Ts / rs))
    out("LAST", "%.4f" % lastT)
    out("FINAL", "yes" if final else "no")
    out("RESUMABLE", "no" if why else "yes")
    out("WHY", "; ".join(why) if why else "-")


def check_v6o(d, prefix, inputs):
    why = []
    import torch
    sp = os.path.join(d, prefix + "_state.pt")
    try:
        st = torch.load(sp, map_location="cpu", weights_only=False)
        k = st["k"]; ch = st["channels"]
        for kk, s in ch.items():
            if s.get("Phi") is None or s.get("F") is None:
                why.append("channel %s lacks Phi/F" % kk)
        out("STATE_RECORD", k); out("CHANNELS", ",".join(sorted(ch.keys())))
    except Exception as e:
        why.append("state unreadable: %r" % (e,))
    for q in inputs:
        if not os.path.exists(q):
            why.append("input missing: %s" % q)
    sj = os.path.join(d, prefix + "_summary.json")
    final = False
    if os.path.exists(sj):
        try:
            final = bool(json.load(open(sj)).get("final"))
        except Exception:
            pass
    out("FINAL", "yes" if final else "no")
    out("RESUMABLE", "no" if why else "yes")
    out("WHY", "; ".join(why) if why else "-")


def check_qh(d, mem):
    why = []
    ck = os.path.join(d, "qh_member%s_channels.npz" % mem)
    try:
        Z = np.load(ck, allow_pickle=False)
        ks = list(Z["kappa"].astype(int))
        out("LAST", ks[-1] if ks else "none")
        if not ks:
            why.append("checkpoint holds no channel")
    except Exception as e:
        why.append("checkpoint unreadable: %r" % (e,))
    for fn in ("qh_member%s.json" % mem, "qh_member%s_arrays.npz" % mem):
        if not os.path.exists(os.path.join(d, fn)):
            why.append("%s missing" % fn)
    try:
        np.load(os.path.join(d, "qh_member%s_arrays.npz" % mem)).files
    except Exception as e:
        why.append("arrays npz unreadable: %r" % (e,))
    out("FINAL", "yes" if os.path.exists(os.path.join(d, "qh_member%s_channels.json" % mem)) else "no")
    out("RESUMABLE", "no" if why else "yes")
    out("WHY", "; ".join(why) if why else "-")


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "slice":
        check_slice(sys.argv[2], sys.argv[3], sys.argv[4])
    elif kind == "v6o":
        check_v6o(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif kind == "qh":
        check_qh(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("unknown kind %s" % kind)
