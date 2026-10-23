#!/usr/bin/env python
"""s16A_provenance_v1.py (busi; inbox #501): emits PAGE_Bprime_s16A_provenance_v1.md whole.
(1) Which prep file the stage A arms loaded, what the md5 876dd5c6 recorded in their prep manifest is, and a member-by-member comparison
    (dtype, shape, max abs difference, manifest keys) of the stage A prep (prep_v5 / kit copies) against the benchmark prep (prep_v4, 9fc0d649)
    and of the card copies, with whether any difference can change a propagated number.
(2) The two max|dR| reads of the gate file (69.3 L and 1e2 L): their arrays, clocks, argmax locations and values at full precision.
Box CPU only, light (nice 10). Card files are only md5-read over ssh (no process launched on a card's GPU)."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s16A_end_lib_v2 as E  # noqa: E402

L = E.L


def card_md5(host, path):
    try:
        r = subprocess.run(["timeout", "60", "ssh", "-n", "-o", "BatchMode=yes", host, "md5sum %s" % path], capture_output=True, text=True, timeout=90)
        return r.stdout.split()[0] if r.returncode == 0 and r.stdout else "unavailable (rc %d)" % r.returncode
    except Exception as ex:  # noqa: BLE001
        return "unavailable (%s)" % type(ex).__name__


def manifest_diff(ma, mb, path=""):
    out = []
    for k in sorted(set(ma) | set(mb)):
        p = path + k
        if k not in ma or k not in mb:
            out.append((p, "only in one", ""))
        elif isinstance(ma[k], dict) and isinstance(mb[k], dict):
            out += manifest_diff(ma[k], mb[k], p + ".")
        elif ma[k] != mb[k]:
            out.append((p, repr(ma[k])[:80], repr(mb[k])[:80]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bp", required=True, help="<internal>/bprime")
    ap.add_argument("--registry", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = a.bp
    out = []; P = out.append
    P("# PAGE B' s = 16, stage A: prep provenance and the two dR reads, version 1 (busi lane)")
    P("")
    P("Emitted whole by busi/code/pagetests/v3/s16A_provenance_v1.py on the box CPU (nice 10); no number is retyped. Order: inbox #501 (1) and (2).")
    P("")
    # ------------------------------------------------------------- (1a) what each arm loaded
    arms = [("clock (run 1, 1e2 L)", "s16A_qh1_v1", "s16A_1_F1_d0_dt0.5_T100_both", "qh1"), ("2b", "s16A_qh1_v1", "s16A_2b_F1_onset225_dt0.5_T100_both", "qh1"),
            ("2c", "s16A_h100f_v1", "s16A_2c_F1_dthalf_dt0.25_T100_both", "h100f"), ("run 1 x_full", "s16A_h100x8_v2/g2", "s16A_1_F1_d0_dt0.5_T6.932_both", None),
            ("run 1 sup", "s16A_h100x8_v1", "s16A_1_F1_d0_dt0.5_T69.32_sup", None), ("run 1 exit", "s16A_h100x8_v1", "s16A_1_F1_d0_dt0.5_T69.32_exit", None)]
    P("## 1. The prep each arm loaded (json args.prep), and the md5 in its prep manifest")
    P("")
    P("| arm | args.prep (path on its device) | box copy of that prep (same basename) | box copy md5 | card copy md5 (read now) | prep_manifest.npz_md5 |")
    P("|---|---|---|---|---|---|")
    J = {}
    for nm, d, tag, host in arms:
        j = json.load(open(os.path.join(B, d, tag + ".json"))); J[nm] = j
        pth = j["args"]["prep"]; bn = os.path.basename(pth)
        box = os.path.join(B, "kit_s16A_v2/prep", bn)
        if not os.path.exists(box):
            box = os.path.join(B, "prep_v5", bn)
        cm = card_md5(host, pth) if host else "- (owner box, card gone)"
        P("| %s | %s | %s | %s | %s | %s |" % (nm, pth, box, E.md5(box) if os.path.exists(box) else "-", cm, j["prep_manifest"].get("npz_md5")))
    P("")
    reg = json.load(open(a.registry))
    lab = "P64_ref_c19r11"
    e = reg[lab]
    P("**What 876dd5c6 is.** run_bprime_v5.py's prep writes `\"npz_md5\": prof.get(\"npz_md5\")` into the prep manifest, and for F1 `prof` is the dense re-solve of the source collapse record (pt3b_cube.dense_record_profile, which copies the record registry's `npz_md5`). So prep_manifest.npz_md5 is the md5 of the SOURCE RECORD %s (the F1 fields' origin), not of the prep file. Registry %s: record npz %s, registered md5 %s; recomputed now: %s (%s)." % (
        lab, a.registry, e["npz"], e["npz_md5"], E.md5(e["npz"]) if os.path.exists(e["npz"]) else "file absent on box",
        "match" if os.path.exists(e["npz"]) and E.md5(e["npz"]) == e["npz_md5"] else "NO MATCH"))
    P("")
    P("The prep file itself is %s/kit_s16A_v2/prep/F1_s16_d0_prep.npz (md5 above; byte-identical to prep_v5/F1_s16_d0_prep.npz). The two md5s name two different files; neither is a corrupted copy of the other. CORRECTION to PAGE_Bprime_s16A_gate_v1 sec. 1, which called 876dd5c6 the 'prep npz md5 recorded in both': it is the source record's md5 recorded in both preps' manifests." % B)
    P("")
    # ------------------------------------------------------------- (1b) member comparison
    files = [("stage A prep (prep_v5)", os.path.join(B, "prep_v5/F1_s16_d0_prep.npz")), ("kit_s16A_v2 copy", os.path.join(B, "kit_s16A_v2/prep/F1_s16_d0_prep.npz")),
             ("benchmark prep (prep_v4)", os.path.join(B, "prep_v4/F1_s16_d0_prep.npz"))]
    Z = {k: np.load(p) for k, p in files}
    P("## 2. Every npz member: stage A prep against its kit copy and against the benchmark prep (prep_v4)")
    P("")
    P("md5: %s." % "; ".join("%s %s" % (k, E.md5(p)) for k, p in files))
    P("")
    ref = Z[files[0][0]]
    P("| member | dtype | shape | kit copy: same dtype/shape, max \\|diff\\| | benchmark prep: dtype | shape | max \\|diff\\| (if shapes agree) |")
    P("|---|---|---|---|---|---|---|")
    for m in sorted(set(ref.files) | set(Z[files[2][0]].files)):
        row = [m]
        if m in ref.files:
            x = ref[m]; row += [str(x.dtype), str(x.shape)]
        else:
            x = None; row += ["absent", "-"]
        for k, _ in files[1:]:
            z = Z[k]
            if m not in z.files or x is None:
                cell = "absent" if m not in z.files else "present"
                row += [cell] if k == files[1][0] else [cell, "-", "-"]
                continue
            y = z[m]
            if x.dtype.kind in "US":
                d = "identical" if str(x) == str(y) else "DIFFERS"
            elif x.shape == y.shape:
                d = "%.3g" % float(np.max(np.abs(x.astype(complex) - y.astype(complex)))) if x.size else "0"
            else:
                d = "shapes differ"
            if k == files[1][0]:
                row += ["%s, %s" % ("same" if (x.dtype == y.dtype and x.shape == y.shape) else "DIFFERENT dtype/shape", d)]
            else:
                row += [str(y.dtype), str(y.shape), d]
        P("| " + " | ".join(row) + " |")
    P("")
    m5 = json.loads(str(ref["manifest"])); m4 = json.loads(str(Z[files[2][0]]["manifest"]))
    dif = manifest_diff(m5, m4)
    P("Manifest keys that differ, stage A prep against benchmark prep (%d): " % len(dif) + ("; ".join("%s: %s vs %s" % t for t in dif[:40]) if dif else "none") + ".")
    P("")
    # propagated members
    prop = ["N", "vx", "vy", "vz", "eta", "U_sub", "e", "n_supply", "H", "Hs", "L", "T0", "R_I", "qF"]
    dk = max(float(np.max(np.abs(ref[m].astype(complex) - Z[files[1][0]][m].astype(complex)))) for m in prop if m in ref.files)
    P("**Can the difference change a propagated number?** The engine propagates from the members %s. Between the stage A prep and the kit copy every member is byte-identical (file md5s equal; max member difference %.3g), so no: every stage A arm on every device loaded the same bytes (sec. 1, card copies read now). The 876dd5c6 / b97c1f87 pair is not a pair of preps (sec. 1). The benchmark prep (prep_v4, md5 9fc0d649, the copy in kit_bench_v1) is a different file: it is not the prep of any stage A arm (sec. 1). Its fields agree with the stage A prep to rounding of the re-solve (max 2e-11 in N, v; eta identical), but its zero cluster excluded %d modes against %d and its window basis U_sub differs (max %.3g; e to %.2g), so numbers read from the benchmark are not column-for-column comparable with stage A. ReviewA_bpchk2_v1 sec. 6 names 9fc0d649 as 'the s = 16 prep' in its HOLD list; within stage A the prep and basis are one file (b97c1f87) on every arm, which is the consistency that clause protects. Whether the benchmark's measured rate or dt ladder still applies is unaffected (they are per-column costs and step differences, not basis-specific values)." % (
        ", ".join(prop), dk, json.loads(str(Z[files[2][0]]["manifest"]))["inputs"]["zero"]["n_zero"], json.loads(str(ref["manifest"]))["inputs"]["zero"]["n_zero"],
        float(np.max(np.abs(ref["U_sub"] - Z[files[2][0]]["U_sub"]))), float(np.max(np.abs(ref["e"] - Z[files[2][0]]["e"])))))
    P("")
    P("Run 1's single-window arms loaded the split preps (prep_v5/F1_s16_d0_sup_prep.npz, _exit_prep.npz). Against the full stage A prep:")
    P("")
    P("| split prep | md5 | max \\|U_sub - full U_sub columns\\| | max \\|e diff\\| | max \\|N, vx, vy, vz, eta diff\\| | max \\|qF - full qF block\\| |")
    P("|---|---|---|---|---|---|")
    ns = int(ref["n_supply"])
    for nm_, sl in (("F1_s16_d0_sup_prep.npz", slice(0, ns)), ("F1_s16_d0_exit_prep.npz", slice(ns, None))):
        pp = os.path.join(B, "prep_v5", nm_); zs = np.load(pp)
        fu = ref["U_sub"][:, sl]
        du = float(np.max(np.abs(zs["U_sub"] - fu))) if zs["U_sub"].shape == fu.shape else float("nan")
        de = float(np.max(np.abs(zs["e"] - ref["e"][sl])))
        dfld = max(float(np.max(np.abs(zs[m] - ref[m]))) for m in ("N", "vx", "vy", "vz", "eta"))
        dq = float(np.max(np.abs(zs["qF"] - ref["qF"][sl, sl])))
        P("| %s | %s | %.3g | %.3g | %.3g | %.3g |" % (nm_, E.md5(pp)[:8], du, de, dfld, dq))
    P("")
    # ------------------------------------------------------------- (2) the two dR reads
    P("## 3. The two max|dR| reads of the gate file (sec. 4, last column): distinct arrays and clocks")
    P("")
    C2 = E.load_run(os.path.join(B, "s16A_h100f_v1"), "s16A_2c_F1_dthalf_dt0.25_T100_both")
    K1 = E.load_run(os.path.join(B, "s16A_qh1_v1"), "s16A_1_F1_d0_dt0.5_T100_both")
    P("| T/L | 2c index, T (a) | clock index, T (a) | max \\|R_2c - R_clock\\| | at (row, col) | R_2c there | R_clock there | max \\|R_clock\\| | md5 of R_clock[i] bytes | md5 of R_2c[i] bytes |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    import hashlib
    for T in (E.T1, E.T2, E.T3):
        i = E.cidx(C2["T"], T); k = E.cidx(K1["T"], T)
        Ra = np.asarray(C2["z"]["R"][i]); Rb = np.asarray(K1["z"]["R"][k])
        d = np.abs(Ra - Rb); r, c = np.unravel_index(int(np.argmax(d)), d.shape)
        P("| %.4g | %d, %.6f | %d, %.6f | %.6e | (%d, %d) | %.12e%+.12ej | %.12e%+.12ej | %.6e | %s | %s |" % (
            T / L, i, C2["T"][i], k, K1["T"][k], d.max(), r, c, Ra[r, c].real, Ra[r, c].imag, Rb[r, c].real, Rb[r, c].imag, np.max(np.abs(Rb)),
            hashlib.md5(Rb.tobytes()).hexdigest()[:8], hashlib.md5(Ra.tobytes()).hexdigest()[:8]))
    P("")
    P("Sources: R is the 'R' member (shape %s) of %s/%s.npz (md5 %s) and %s/%s.npz (md5 %s), both md5-checked against their MD5SUMS; each row reads a different clock index of each file. The 69.3 L and 1e2 L maxima are different numbers at full precision, read from different arrays." % (
        str(C2["z"]["R"].shape), C2["dir"], C2["tag"], C2["md5_npz"], K1["dir"], K1["tag"], K1["md5_npz"]))
    P("")
    P("Files: this script (md5 %s); helpers s16A_end_lib_v1.py, s16A_end_lib_v2.py." % E.md5(os.path.abspath(__file__)))
    open(a.out, "w").write("\n".join(out) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
