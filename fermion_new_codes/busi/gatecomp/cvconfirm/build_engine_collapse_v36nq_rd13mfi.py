#!/usr/bin/env python
"""build_engine_collapse_v36nq_rd13mfi.py (busi, analyst; CVCONFIRM, the confirmation ReviewA_cvcause_v1 sec. 3 ordered (inbox #430);
busi/notes/CVCONFIRM_build_v1.md): engine_collapse_v36nq_rd13mfi.py from CV2048's engine engine_collapse_v36nq_rd13mf.py (md5 248f25f3...,
code16d) by marked replacement blocks, in the form of build_engine_collapse_v36nq_rd13mfb.py (busi/code/gatecomp/flat).
Rules (checked): the source's md5 begins with 248f25f3 (--expect-src-md5); every OLD anchor is copied verbatim from the source and occurs
exactly once; every line a NEW block adds or changes carries "field inject"; the reverse pass (NEW -> OLD, each NEW once) restores the source
byte for byte (in memory before writing and again on the written file; available as --reverse); the output compiles; nothing is overwritten.
    python build_engine_collapse_v36nq_rd13mfi.py --src .../engine_collapse_v36nq_rd13mf.py [--dst .../engine_collapse_v36nq_rd13mfi.py] [--check-only]
    python build_engine_collapse_v36nq_rd13mfi.py --reverse --src .../engine_collapse_v36nq_rd13mfi.py --dst .../restored.py --expect .../engine_collapse_v36nq_rd13mf.py

The option (absent by default: nothing of the patch runs but one dictionary read per call site, so the records, fields, spectra, handoff
and slices are 248f25f3's bitwise; the json's args gain the keys inject_fields and inject_T_over_rs, both None):
--inject-fields NPZ --inject-T-over-rs T: at the record whose clock is T (|T/r_s - T_inject| < 1e-6), after the record's own solve has run
    and been accepted as usual (the parent's path, bitwise, up to that point), the record's accepted APPLIED fields (N_applied, v_applied:
    the fields that build h_m[N, v], continue the reference and propagate the state) are replaced by the stored row of another run at that
    clock, read from NPZ (an engine fields npz, or its one-row extract by inject_extract_v1.py) and mapped onto this run's sites by the
    shared radii: the source chain's first nsites sites must carry this chain's radii exactly (this chain is the inner part of the
    source's); the values are copied as stored, with no renormalisation. The record is then evaluated at the injected fields exactly as a
    Newton evaluation is (nk_stage_v32's evalG): every channel's eigendecomposition at them, the reference continued from the record's
    window, the sources from this run's own state, one field solve; that evaluation is the record's accepted one (its masks, diagnostics,
    window, f_ref for the next record and the propagation exp(-i h tau)), and the run continues as usual. The record's solved fields N, v are
    that solve's (how the source's own stored N, v were made at its kept iterate); the source's stored N, v are not copied into the record
    (they would not match the record's sources, nodes or charge, and they do not enter the propagation): their difference is read and
    written instead. The record gains a "field_inject" entry (the file and its md5, the source's provenance, the row, the mapped sites,
    the injected fields against the record's own accepted ones, the injected evaluation's own residual max|N - N_applied|, max|v - v_applied|,
    the solve against the source's stored N, v, N(0) and E both ways); the head gains "field_inject" at that record; the label gains
    "field inject: ..."; its own solve's entries (solve_v32, converged, iterations) stay the record's own solve's. The run refuses at start
    when the file lacks R, T, N_applied or v_applied, holds no or several rows at T, T is off the record clock (T r_s / tau not an
    integer), the source chain is shorter or its first nsites radii differ, or (with --resume-slice) T is not after the slice's clock or is
    beyond the segment's end; and it raises at the record if the injected field solve fails. The injection acts once.
Blocks:
  A  the module docstring's lineage lines;
  B  before iter_cache_drop: INJ, inj_configure, inj_label, inj_begin, inj_record, inj_attach;
  C  run(): inj_begin after R = chs[0].R;
  D  run(): inj_record after the record's accepted evaluation is chosen (after best = None);
  E  run(): inj_attach after the record's solve_v32 entry;
  F  run(): inj_label after iso_label;
  G  the options (after --nk-plain-fallback);
  H  main(): inj_configure after iso_configure.
"""
import argparse
import hashlib
import os
import py_compile

MARK = "field inject"
SRC_MD5_PREFIX = "248f25f3"
SRC_NAME = "engine_collapse_v36nq_rd13mf.py"
OUT_NAME = "engine_collapse_v36nq_rd13mfi.py"


def blk(s):
    return s[1:] if s.startswith("\n") else s


EDITS = []

EDITS.append(("A lineage", blk(r'''
#!/usr/bin/env python
"""engine_collapse_v36nq_rd13mf.py (busi, analyst; MERGE-1: RD-2's --fields wiring'''), blk(r'''
#!/usr/bin/env python
"""engine_collapse_v36nq_rd13mfi.py (busi, analyst; CVCONFIRM, the field injection; built by build_engine_collapse_v36nq_rd13mfi.py over v36nq_rd13mf, md5 248f25f3) [field inject]
version 36nq_rd13mfi = version 36nq_rd13mf with one option, absent by default, so that the defaults are 248f25f3 bitwise (the json's args gain 2 keys): [field inject]
--inject-fields NPZ --inject-T-over-rs T: at the record at clock T, after its own solve, the accepted applied fields (N_applied, v_applied) are replaced [field inject]
    by another run's stored row at T (an engine fields npz or its extract), mapped onto this chain's sites by the shared radii (this chain the inner [field inject]
    part of the source's), copied as stored; the record is evaluated there as a Newton evaluation is (eigendecompositions, the reference continued from [field inject]
    the window, this run's sources, one field solve) and that evaluation is the record's accepted one; the run propagates and continues as usual [field inject]
    (ReviewA_cvcause_v1 sec. 3: "the injection of P's stored 3.3125 applied and kept fields on X's sites"). Version 36nq_rd13mf's docstring follows. [field inject]
engine_collapse_v36nq_rd13mf.py (busi, analyst; MERGE-1: RD-2's --fields wiring''')))

EDITS.append(("B helpers", blk(r'''


def iter_cache_drop(a, cache):
'''), blk(r'''


# ------------------------------------------------------------------------------------------------------- field inject (busi, analyst; CVCONFIRM)   [field inject]
INJ = {"on": False, "path": None, "md5": None, "T_over_rs": None, "row": None, "i": None, "src": None, "R_src": None, "done": False, "rec": None, "desc": ""}   # field inject: the option's state (by default nothing of field inject runs)


def inj_configure(a, ap):   # field inject: main(), after iso_configure; returns at once when the option is absent
    p_ = getattr(a, "inject_fields", None); t_ = getattr(a, "inject_T_over_rs", None)   # field inject
    if p_ is None and t_ is None:   # field inject
        return   # field inject
    if p_ is None or t_ is None:   # field inject
        ap.error("field inject: --inject-fields and --inject-T-over-rs go together")   # field inject
    if not os.path.isfile(p_):   # field inject
        ap.error("field inject: no file %s" % p_)   # field inject
    import hashlib as _hl_inj   # field inject
    with open(p_, "rb") as fh_:   # field inject
        md5_ = _hl_inj.md5(fh_.read()).hexdigest()   # field inject
    Z = np.load(p_, allow_pickle=True)   # field inject
    miss = [k_ for k_ in ("R", "T", "N_applied", "v_applied") if k_ not in Z.files]   # field inject
    if miss:   # field inject
        ap.error("field inject: %s lacks %s" % (p_, ", ".join(miss)))   # field inject
    Tz = np.atleast_1d(np.asarray(Z["T"], float))   # field inject
    hit = np.where(np.abs(Tz / a.rs - t_) < 1e-6)[0]   # field inject
    if len(hit) != 1:   # field inject
        ap.error("field inject: %s holds %d rows at T/rs %.6f (must be exactly 1)" % (p_, len(hit), t_))   # field inject
    k_ = t_ * a.rs / a.tau   # field inject
    if abs(k_ - round(k_)) > 1e-9 or t_ > a.T_over_rs + 1e-9:   # field inject
        ap.error("field inject: T/rs %.6f is not on the record clock (T rs / tau = %.9f) or is beyond --T-over-rs" % (t_, k_))   # field inject
    i_ = int(hit[0])   # field inject
    row = {q: np.array(np.atleast_2d(np.asarray(Z[q], float))[i_], dtype=float) for q in ("N_applied", "v_applied", "N", "v") if q in Z.files}   # field inject
    src = None   # field inject
    if "inject_source" in Z.files:   # field inject: the extract's provenance (inject_extract_v1.py)
        src = json.loads(str(Z["inject_source"]))   # field inject
    INJ.update(on=True, path=os.path.abspath(p_), md5=md5_, T_over_rs=float(t_), row=row, i=i_, src=src, R_src=np.asarray(Z["R"], float).copy(), T_src=float(Tz[i_]))   # field inject
    INJ["desc"] = ("the accepted applied fields of the record at T/rs %.6f replaced by row %d of %s (md5 %s), mapped onto this chain's sites by the shared radii, "   # field inject
                   "copied as stored; the record evaluated there as a Newton evaluation is, and propagated as usual (ReviewA_cvcause_v1 sec. 3)" % (t_, i_, os.path.basename(p_), md5_))   # field inject
    print("field inject: %s" % INJ["desc"], flush=True)   # field inject


def inj_label(label):   # field inject: the run label (the same object by default)
    return label if not INJ["on"] else label + "; field inject: " + INJ["desc"]   # field inject


def inj_begin(a, R, rs):   # field inject: run(), once R is known: the source's radii and the clock against this run's (nothing by default)
    if not INJ["on"]:   # field inject
        return   # field inject
    Rr = np.asarray(R, float); n_ = len(Rr); Rs_ = INJ["R_src"]   # field inject
    if len(Rs_) < n_ or not np.array_equal(Rs_[:n_], Rr):   # field inject
        raise SystemExit("field inject: the source's chain (%d sites) does not carry this chain's %d radii on its first sites" % (len(Rs_), n_))   # field inject
    for q in INJ["row"].values():   # field inject
        if q.shape != Rs_.shape:   # field inject
            raise SystemExit("field inject: a source row has %d values against the source's %d radii" % (q.size, Rs_.size))   # field inject
    Ti = INJ["T_over_rs"] * rs   # field inject
    if getattr(a, "resume_slice", None):   # field inject
        Ts = float(np.load(a.resume_slice)["T"])   # field inject: the slice's clock (one member of the npz read)
        if not Ti > Ts + 1e-9:   # field inject
            raise SystemExit("field inject: T/rs %.6f is not after the slice's clock T/rs %.6f" % (INJ["T_over_rs"], Ts / rs))   # field inject
    if getattr(a, "segment_end_T_over_rs", None) is not None and INJ["T_over_rs"] > a.segment_end_T_over_rs + 1e-9:   # field inject
        raise SystemExit("field inject: T/rs %.6f is beyond the segment's end %.6f" % (INJ["T_over_rs"], a.segment_end_T_over_rs))   # field inject
    print("field inject: the source's first %d of %d sites carry this chain's radii exactly; the injection waits for the record at T/rs %.6f" % (n_, len(Rs_), INJ["T_over_rs"]), flush=True)   # field inject


def inj_record(a, chs, st, cache, srcs, sources, do_solve, T, rs, N_app, v_app, Nn, vn, Wn, info, head):   # field inject: the record's accepted evaluation (the same objects unless this is the record)
    if not INJ["on"] or INJ["done"] or abs(T / rs - INJ["T_over_rs"]) > 1e-6:   # field inject
        return cache, srcs, Nn, vn, Wn, info, N_app, v_app   # field inject
    t0_ = time.time(); n_ = len(N_app); row = INJ["row"]   # field inject
    Ni = row["N_applied"][:n_].copy(); vi = row["v_applied"][:n_].copy()   # field inject: mapped by the shared radii (inj_begin checked them), copied as stored
    mx = lambda x: float(np.max(np.abs(x)))   # field inject
    nd0 = lambda inf_: float(np.asarray(inf_["N_nodes"], float)[0]) if inf_.get("N_nodes") is not None else None   # field inject
    own = {"N0": nd0(info), "E": float(FLD.charge(info, a.g)), "dNapp": mx(Ni - N_app), "dvapp": mx(vi - v_app), "resN": mx(Nn - N_app), "resv": mx(vn - v_app)}   # field inject
    for k_ in list(cache):   # field inject: the record's own accepted evaluation released before the injected one is made
        cache[k_] = None   # field inject
    eig = eigh_all(chs, Ni, vi, bool(a.streams))   # field inject: as nk_stage_v32's evalG does at an iterate
    new = {}   # field inject
    for ch in chs:   # field inject
        e, U = eig.pop(ch.kappa)   # field inject
        if a.reference == "R2":   # field inject
            t_c = time.time(); s_ = st[ch.kappa]   # field inject
            ref, cinfo, mask = continue_ref_v34(a, ch, s_["window"], (Ni, vi), e, U, s_["n_sea"])   # field inject
            cinfo["_mask"] = mask   # field inject
            TM["continuation_s"] += time.time() - t_c   # field inject
            TM["fast" if cinfo.get("nsub", 1) == 1 else "midpoint"] += 1   # field inject
        else:   # field inject
            ref = st[ch.kappa]["sea"]; cinfo = {"mode": "R1", "_mask": (e < 0).cpu().numpy()}   # field inject
        new[ch.kappa] = (None, e, U, ref, cinfo)   # field inject
    del eig   # field inject
    srcs_i = sources(new)   # field inject: this run's own state's sources at the injected fields
    Nn_i, vn_i, Wn_i, info_i = do_solve(srcs_i)   # field inject
    if info_i.get("failed"):   # field inject
        raise RuntimeError("field inject: the field solve at the injected fields failed (%s); nothing is propagated" % info_i["failed"])   # field inject
    ent = {"label": "field inject: " + INJ["desc"], "file": INJ["path"], "file_md5": INJ["md5"], "source": INJ["src"], "source_row": INJ["i"],   # field inject
           "source_T": INJ["T_src"], "T_over_rs": T / rs, "sites_mapped": n_, "source_sites": int(len(INJ["R_src"])),   # field inject
           "source_N_applied_last_mapped_minus_1": float(Ni[-1] - 1.0), "source_N_applied_last_minus_1": float(row["N_applied"][-1] - 1.0),   # field inject
           "injected_minus_own_N_applied": own["dNapp"], "injected_minus_own_v_applied": own["dvapp"],   # field inject
           "own_residual_N": own["resN"], "own_residual_v": own["resv"],   # field inject: the record's own accepted evaluation, max|N - N_applied|, max|v - v_applied|
           "injected_residual_N": mx(Nn_i - Ni), "injected_residual_v": mx(vn_i - vi),   # field inject: the injected evaluation's own distance from its fixed point
           "solve_minus_source_N": (mx(Nn_i - row["N"][:n_]) if "N" in row else None), "solve_minus_source_v": (mx(vn_i - row["v"][:n_]) if "v" in row else None),   # field inject
           "solve_minus_source_N_centre": (float(Nn_i[0] - row["N"][0]) if "N" in row else None),   # field inject: at site 1
           "N0_injected": nd0(info_i), "N0_own": own["N0"], "E_injected": float(FLD.charge(info_i, a.g)), "E_own": own["E"],   # field inject
           "nsub_max_injected": max([int(new[ch.kappa][4].get("nsub") or 1) for ch in chs]), "seconds": time.time() - t0_}   # field inject
    INJ["rec"] = ent; INJ["done"] = True   # field inject
    head["field_inject"] = {k_: ent[k_] for k_ in ("label", "file", "file_md5", "source", "source_row", "source_T", "T_over_rs", "sites_mapped", "source_sites")}   # field inject
    print("field inject at T/rs %.4f: applied fields replaced by row %d of %s (%d of %d sites); injected minus own: N_applied %.3e, v_applied %.3e; residual at the injected fields N %.3e v %.3e (own %.3e %.3e); solve minus source N %s v %s; N(0) %.9f (own %.9f); E %.9f (own %.9f) (%.0f s)" % (   # field inject
        T / rs, INJ["i"], os.path.basename(INJ["path"]), n_, ent["source_sites"], own["dNapp"], own["dvapp"], ent["injected_residual_N"], ent["injected_residual_v"], own["resN"], own["resv"],   # field inject
        ("%.3e" % ent["solve_minus_source_N"]) if ent["solve_minus_source_N"] is not None else "-", ("%.3e" % ent["solve_minus_source_v"]) if ent["solve_minus_source_v"] is not None else "-",   # field inject
        ent["N0_injected"] if ent["N0_injected"] is not None else float("nan"), own["N0"] if own["N0"] is not None else float("nan"), ent["E_injected"], own["E"], ent["seconds"]), flush=True)   # field inject
    return new, srcs_i, Nn_i, vn_i, Wn_i, info_i, Ni, vi   # field inject


def inj_attach(rec):   # field inject: the injected record's entry (nothing on any other record)
    if INJ["rec"] is not None and abs(rec["T_over_rs"] - INJ["rec"]["T_over_rs"]) < 1e-9:   # field inject
        rec["field_inject"] = INJ["rec"]   # field inject


def iter_cache_drop(a, cache):
''')))

EDITS.append(("C begin", blk(r'''
    R = chs[0].R
'''), blk(r'''
    R = chs[0].R
    inj_begin(a, R, rs)   # field inject (nothing by default)
''')))

EDITS.append(("D record", blk(r'''
        best = None
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
'''), blk(r'''
        best = None
        cache, srcs, Nn, vn, Wn, info, N_app, v_app = inj_record(a, chs, st, cache, srcs, sources, do_solve, T, rs, N_app, v_app, Nn, vn, Wn, info, head)   # field inject (the same objects by default)
        eps, J, eps_pairs, eps_dust, J_pairs = srcs
''')))

EDITS.append(("E attach", blk(r'''
        if solve32 is not None:
            rec["solve_v32"] = solve32
'''), blk(r'''
        if solve32 is not None:
            rec["solve_v32"] = solve32
        inj_attach(rec)   # field inject (nothing by default)
''')))

EDITS.append(("F label", blk(r'''
    label = iso_label(label, a)                                                     # version 36n (the same label by default)
'''), blk(r'''
    label = iso_label(label, a)                                                     # version 36n (the same label by default)
    label = inj_label(label)   # field inject (the same label by default)
''')))

EDITS.append(("G options", blk(r'''
    a = ap.parse_args()
    if a.fields != "ivp" or a.fields_warm_start != "off":   # rd2 fields
'''), blk(r'''
    ap.add_argument("--inject-fields", dest="inject_fields", default=None, help="(field inject) an engine fields npz (or its one-row extract) whose stored applied fields replace the accepted ones at --inject-T-over-rs")   # field inject
    ap.add_argument("--inject-T-over-rs", dest="inject_T_over_rs", type=float, default=None, help="(field inject) the record's clock in units of r_s")   # field inject
    a = ap.parse_args()
    if a.fields != "ivp" or a.fields_warm_start != "off":   # rd2 fields
''')))

EDITS.append(("H configure", blk(r'''
    iso_configure(a, ap)                                                            # version 36n
'''), blk(r'''
    iso_configure(a, ap)                                                            # version 36n
    inj_configure(a, ap)   # field inject
''')))


def md5(b):
    return hashlib.md5(b).hexdigest()


def forward(text):
    for name, old, new in EDITS:
        c = text.count(old)
        if c != 1:
            raise SystemExit("edit %s: the OLD anchor occurs %d times (must be 1); it begins: %r" % (name, c, old.splitlines()[0][:100]))
        text = text.replace(old, new)
    return text


def reverse(text):
    for name, old, new in reversed(EDITS):
        c = text.count(new)
        if c != 1:
            raise SystemExit("reverse %s: the NEW block occurs %d times (must be 1)" % (name, c))
        text = text.replace(new, old)
    return text


def marker_check():
    """every line a NEW block adds or changes carries MARK; a line of OLD is not new (nor is one moved out from behind the opening triple
    quote); blank lines are neutral."""
    bad, n_marked = [], 0
    for name, old, new in EDITS:
        olds = set(old.splitlines())
        olds |= {l[3:] for l in old.splitlines() if l.startswith('"""')}
        for ln in new.splitlines():
            if ln in olds or ln.strip() == "":
                continue
            if MARK in ln:
                n_marked += 1
            else:
                bad.append((name, ln))
    return bad, n_marked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_NAME)
    ap.add_argument("--dst", default=None, help="default: %s beside --src" % OUT_NAME)
    ap.add_argument("--reverse", action="store_true", help="src is %s: restore %s into dst" % (OUT_NAME, SRC_NAME))
    ap.add_argument("--expect", default=None, help="(reverse) the file the restored text must equal byte for byte")
    ap.add_argument("--expect-src-md5", default=SRC_MD5_PREFIX, help="(build) the prefix the source's md5 must start with (default %s; '' to skip)" % SRC_MD5_PREFIX)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if a.dst is None:
        a.dst = os.path.join(os.path.dirname(os.path.abspath(a.src)), OUT_NAME if not a.reverse else "engine_collapse_v36nq_rd13mf_restored.py")
    src = open(a.src, "rb").read()
    text = src.decode("utf-8")
    if a.reverse:
        out = reverse(text).encode("utf-8")
        if not a.check_only:
            if os.path.exists(a.dst):
                raise SystemExit("%s exists; nothing written (never overwritten)" % a.dst)
            open(a.dst, "wb").write(out)
        print("reverse: %s %s (md5 %s)" % ("would write" if a.check_only else "wrote", a.dst, md5(out)))
        if a.expect:
            ok = out == open(a.expect, "rb").read()
            print("restored equals %s byte for byte: %s" % (a.expect, ok))
            raise SystemExit(0 if ok else 1)
        return
    print("source %s: md5 %s, %d bytes" % (a.src, md5(src), len(src)))
    if a.expect_src_md5 and not md5(src).startswith(a.expect_src_md5):
        raise SystemExit("the source's md5 does not start with %s: not %s; nothing written" % (a.expect_src_md5, SRC_NAME))
    if MARK in text:
        raise SystemExit("the source already carries %r; nothing written" % MARK)
    for name, old, new in EDITS:
        print("  anchor %-14s occurs %d time(s) in the input (%d -> %d lines)" % (name, text.count(old), len(old.splitlines()), len(new.splitlines())))
    bad, n_marked = marker_check()
    if bad:
        for name, ln in bad:
            print("  UNMARKED new line in %s: %r" % (name, ln))
        raise SystemExit("new lines without the marker")
    print("every new or changed line carries %r (%d marked lines)" % (MARK, n_marked))
    built = forward(text)
    out = built.encode("utf-8")
    back = reverse(built).encode("utf-8")
    print("reversible (NEW -> OLD restores the input byte for byte): %s" % (back == src))
    if back != src:
        raise SystemExit("the reverse pass does not restore the input; nothing written")
    try:
        compile(built, a.dst, "exec")
    except SyntaxError as e:
        raise SystemExit("the result does not compile (%s); nothing written" % e)
    if a.check_only:
        print("check only: %s would be %d bytes, md5 %s" % (a.dst, len(out), md5(out)))
        return
    if os.path.exists(a.dst):
        if open(a.dst, "rb").read() == out:
            print("%s exists and equals the build byte for byte (md5 %s); not rewritten" % (a.dst, md5(out)))
            return
        raise SystemExit("%s exists and differs from the build; nothing written (never overwritten)" % a.dst)
    open(a.dst, "wb").write(out)
    written = open(a.dst, "rb").read()
    if written != out or reverse(written.decode("utf-8")).encode("utf-8") != src:
        raise SystemExit("the written file does not reverse to the input")
    py_compile.compile(a.dst, doraise=True)
    n_lines = sum(1 for ln in built.splitlines() if MARK in ln)
    print("wrote %s: %d bytes, md5 %s; compiles; the reverse of the written file equals the source byte for byte" % (a.dst, len(out), md5(out)))
    print("check: grep -c '%s' %s  ->  %d" % (MARK, a.dst, n_lines))


if __name__ == "__main__":
    main()
