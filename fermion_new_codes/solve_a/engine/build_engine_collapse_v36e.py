#!/usr/bin/env python
"""build_engine_collapse_v36e.py (reviewer A for the solve lane's builder; LADDERS_spec_v1 section 3): writes engine_collapse_v36e.py from
engine_collapse_v36d.py by 11 exact string replacements. Each OLD is copied exactly from engine_collapse_v36d.py and must occur exactly once;
each NEW contains its OLD plus inserted text, so that replacing every NEW by its OLD restores the source byte for byte (checked here). The new
options are off by default: with --first-return-collar-over-rs absent nothing of the inserted code runs and the json's "args" dict alone gains
the 3 new keys, so v36e with the option off is v36d bitwise in every record number, printed line and stored array (the runtime gate:
engine_collapse_v36e_selftest.py --gate on 3 records of the r_s/a 16 member on both engines, the launcher's gate36e mode).
Usage: python build_engine_collapse_v36e.py [engine_collapse_v36d.py path] [engine_collapse_v36e.py path]   (defaults: beside this script)"""
import hashlib
import os
import py_compile
import sys

REPLACEMENTS = [
("docstring",
r'''"""engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change''',
r'''"""engine_collapse_v36e.py (reviewer A for the solve lane's builder; LADDERS_spec_v1 item 3 (d): the (0.05, 326) member "to T/r_s 30 or the chain
end's first return into the collar, the first-return estimate printed"): version 36e = version 36d with one option, off by default, so that the
defaults are version 36d bitwise in every record number, every printed line and every stored array (the json's "args" dict gains the 3 keys below):
--first-return-collar-over-rs X [--first-return-launch-over-rs Y] [--first-return-end-rule end|print]: the first-return estimate. At the formation
    record (the first record holding a zero of N - |v|) an outgoing null ray is launched from r = Y r_s, or, when Y is absent, from that record's
    R_sh (the radius enclosing 99 percent of the dust energy), or from its outer zero plus the averaging scale where R_sh lies inside that (a ray
    at a zero never moves); it is advanced record by record on each record's dense fields, frozen over the record's step as the state's own step
    is (dr/dT = N - |v| outward; at the chain's end, the last site or the collar node, whichever is inner, it reflects and continues inward with
    dr/dT = -(N + |v|)), by midpoint steps of 0.5 a of clock on the fields linearly interpolated from a grid of 4 points per lattice unit; at
    every record its remaining path is integrated on that record's fields alone and the projected clocks of its arrival at the chain's end and
    of its return into the collar's inner radius X r_s print in the line "v36e first return" after the record line and go in the record's
    first_return_v36e entry (the ray's position, phase and speed; TRAPPED where N - |v| <= 0 at the ray; the projection capped at 4 run lengths);
    with the end rule "end" the run ends after the first record whose clock is at or past the ray's return (the record written, nothing
    propagated from it, the json's end_v36e entry), with "print" nothing ends. At a slice restart the ray's state is taken from the last stored
    record's entry (every entry stores the position advanced to the next record). With the option off none of this runs and nothing is added.
Version 36d's docstring follows.
engine_collapse_v36d.py (the solve lane's analyst builder 11; engine spec 53.1 (1), (2)): version 36d = version 36c with one change'''),
("module state",
r'''NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other''',
r'''NSR = {"on": False, "prev_sig": None, "rec": None, "prev_m": None, "other": None}   # version 36c (--nk-set resolve); 36d: prev_m, other
FR = {"on": False, "launched": False, "launch_T": None, "launch_r": None, "launch_rule": None, "r": None, "phase": None, "reflected_T": None, "returned_T": None, "cap_runs": 4.0}   # version 36e (--first-return-collar-over-rs)'''),
("functions before run",
r'''def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)''',
r'''def first_return_desc_v36e(a):
    """version 36e: the option in words (the run label; the record's engine_v36e entry)."""
    launch = ("r = %g r_s" % a.first_return_launch_over_rs) if a.first_return_launch_over_rs is not None else (
        "the formation record's R_sh (the radius enclosing 99 percent of the dust energy), or its outer zero of N - |v| plus the averaging scale where R_sh lies inside that")
    return ("the first-return estimate: the outgoing null ray (dr/dT = N - |v|) launched at the formation record (the first record holding a zero of N - |v|) "
            "from %s, advanced record by record on each record's dense fields frozen over the record's step, reflected at the chain's end (then dr/dT = -(N + |v|)), "
            "its projected arrival at the chain's end and return into the collar's inner radius %g r_s printed per record on the record's own fields; end rule: %s") % (
        launch, a.first_return_collar_over_rs,
        "the run ends after the first record at or past the return" if a.first_return_end_rule == "end" else "the estimate printed, nothing ends")


def first_return_init_v36e(a, recs):
    """version 36e: at a slice restart the ray's state from the last stored record carrying an entry (its position advanced to the next record)."""
    FR.update({"launched": False, "launch_T": None, "launch_r": None, "launch_rule": None, "r": None, "phase": None, "reflected_T": None, "returned_T": None})
    for r_ in reversed(recs or []):
        e_ = r_.get("first_return_v36e")
        if e_ is None:
            continue
        if e_.get("launched"):
            FR.update({"launched": True, "launch_T": e_["launch_T"], "launch_r": e_["launch_r"], "launch_rule": e_.get("launch_rule"), "r": e_["r_next"],
                       "phase": e_["phase_next"], "reflected_T": e_.get("reflected_T"), "returned_T": e_.get("returned_T")})
            print("version 36e: the ray's state restored from the stored record at T/rs %.4f: at r %.3f a (%s) for the next record" % (r_["T"] / a.rs, FR["r"], FR["phase"]), flush=True)
        else:
            print("version 36e: the stored records to the slice hold no launched ray (no zero of N - |v| yet): the launch waits for the formation record", flush=True)
        break


def first_return_integrate_v36e(r, phase, t, t_stop, grid, s_out, s_in, r_end, stop_in=None, h=0.5):
    """version 36e: the ray (r, phase) from t to t_stop on frozen fields: dr/dT = s_out (phase out) or s_in (phase in), both linearly interpolated on grid,
    midpoint steps of h; the reflection at r_end turns the phase; stop_in: the radius whose inward crossing ends the integration. The events carry the
    reflection's and the return's clocks (linear interpolation inside the step). Returns (r, phase, t, events)."""
    ev = {}
    while t < t_stop - 1e-12:
        dt = min(h, t_stop - t)
        sp = s_out if phase == "out" else s_in
        f1 = float(np.interp(r, grid, sp))
        rm = min(max(r + 0.5 * dt * f1, 0.0), r_end)
        f2 = float(np.interp(rm, grid, sp))
        rn = r + dt * f2
        if phase == "out" and rn >= r_end and rn > r:
            frac = (r_end - r) / (rn - r)
            t = t + frac * dt; r = r_end; phase = "in"; ev["reflected_T"] = t
            continue
        if phase == "in" and stop_in is not None and rn <= stop_in and rn < r:
            frac = (r - stop_in) / (r - rn)
            t = t + frac * dt; ev["returned_T"] = t
            return stop_in, phase, t, ev
        r = min(max(rn, 0.0), r_end); t += dt
    return r, phase, t, ev


def first_return_record_v36e(a, info, R, rs, T, Rsh, zeros):
    """version 36e, per record (the docstring): the launch at the formation record; the projection on this record's dense fields from the ray's position at
    this clock (to the chain's end and back to the collar's inner radius); the return test (the position at this clock at or inside the collar's inner
    radius in the inward phase); the advance over the record's step on these fields. Returns (the record's entry, returned)."""
    rc = float(info["r_collar"]); r_end = float(min(float(np.asarray(R, float)[-1]), rc))
    R_c = float(a.first_return_collar_over_rs) * rs
    grid = np.linspace(0.5, r_end, int(round(4.0 * (r_end - 0.5))) + 1)
    Ng = np.asarray(info["N_dense"](grid), float).ravel(); vg = np.asarray(info["v_dense"](grid), float).ravel()
    s_out = Ng - np.abs(vg); s_in = -(Ng + np.abs(vg))
    ent = {"collar_over_rs": float(a.first_return_collar_over_rs), "collar_r": R_c, "chain_end_r": r_end, "T_over_rs": T / rs, "launched": bool(FR["launched"])}
    if not FR["launched"]:
        if not zeros:
            ent["note"] = "not launched: no zero of N - |v| on this record"
            return ent, False
        z_out = float(max(zeros))
        if a.first_return_launch_over_rs is not None:
            r0 = float(a.first_return_launch_over_rs) * rs; rule = "the option's %g r_s" % a.first_return_launch_over_rs
        elif Rsh is not None and float(Rsh) > z_out + float(a.ell):
            r0 = float(Rsh); rule = "the formation record's R_sh"
        else:
            r0 = z_out + float(a.ell); rule = "the formation record's outer zero plus the averaging scale (R_sh %s)" % (("%.2f inside it" % float(Rsh)) if Rsh is not None else "undefined")
        r0 = min(max(r0, 0.5), r_end)
        FR.update({"launched": True, "launch_T": T, "launch_r": r0, "launch_rule": rule, "r": r0, "phase": "out", "reflected_T": None, "returned_T": None})
        ent["launched"] = True
    r_now, ph = float(FR["r"]), FR["phase"]
    T_cap = T + FR["cap_runs"] * float(a.T_over_rs) * rs
    _, _, _, evp = first_return_integrate_v36e(r_now, ph, T, T_cap, grid, s_out, s_in, r_end, stop_in=R_c)
    returned = bool(ph == "in" and r_now <= R_c + 1e-9)
    r_next, ph_next, _, eva = first_return_integrate_v36e(r_now, ph, T, T + a.tau, grid, s_out, s_in, r_end, stop_in=None)
    if FR["reflected_T"] is None and "reflected_T" in eva:
        FR["reflected_T"] = eva["reflected_T"]
    if FR["returned_T"] is None and "returned_T" in evp and evp["returned_T"] <= T + a.tau + 1e-9:
        FR["returned_T"] = evp["returned_T"]
    if returned and FR["returned_T"] is None:
        FR["returned_T"] = T
    FR["r"], FR["phase"] = float(r_next), ph_next
    sp_now = float(np.interp(r_now, grid, s_out if ph == "out" else s_in))
    end_T = FR["reflected_T"] if FR["reflected_T"] is not None else evp.get("reflected_T")
    ret_T = FR["returned_T"] if FR["returned_T"] is not None else evp.get("returned_T")
    ent.update({"launch_T": FR["launch_T"], "launch_T_over_rs": FR["launch_T"] / rs, "launch_r": FR["launch_r"], "launch_r_over_rs": FR["launch_r"] / rs,
                "launch_rule": FR["launch_rule"], "r": r_now, "r_over_rs": r_now / rs, "phase": ph, "speed_at_r": sp_now,
                "trapped": bool(ph == "out" and sp_now <= 0.0), "reflected_T": FR["reflected_T"],
                "end_arrival_T_over_rs": (end_T / rs if end_T is not None else None), "end_arrival_is_projected": bool(FR["reflected_T"] is None),
                "return_T_over_rs": (ret_T / rs if ret_T is not None else None), "return_is_projected": bool(FR["returned_T"] is None),
                "projection_cap_T_over_rs": T_cap / rs, "returned": returned, "r_next": float(r_next), "phase_next": ph_next})
    return ent, returned


def first_return_line_v36e(fr):
    """version 36e: the printed first-return line of a record."""
    if not fr.get("launched"):
        return "v36e first return: %s" % fr.get("note", "not launched")
    f_ = lambda x: ("%.3f" % x) if x is not None else "beyond the projection's cap"
    return ("v36e first return: ray launched at T/rs %.4f from r %.2f a (%.3f r_s; %s); now r %.2f a (%.3f r_s, %s, dr/dT %+.4f%s); chain end %.0f a: arrival T/rs %s%s; "
            "collar %g r_s: return T/rs %s%s") % (
        fr["launch_T_over_rs"], fr["launch_r"], fr["launch_r_over_rs"], fr["launch_rule"], fr["r"], fr["r_over_rs"], fr["phase"], fr["speed_at_r"], " TRAPPED" if fr["trapped"] else "",
        fr["chain_end_r"], f_(fr["end_arrival_T_over_rs"]), " (projected)" if fr["end_arrival_is_projected"] else " (reflected)",
        fr["collar_over_rs"], f_(fr["return_T_over_rs"]), (" (projected)" if fr["return_is_projected"] else (" RETURNED" if fr["returned"] else " (crossed within this record's step)")))


def run(a):
    torch = BK["torch"]
    os.makedirs(a.out, exist_ok=True)'''),
("run label",
r'''    if v34_on(a):
        label += "; v34: " + v34_desc(a)
    rs = a.rs''',
r'''    if v34_on(a):
        label += "; v34: " + v34_desc(a)
    if FR["on"]:
        label += "; v36e: " + first_return_desc_v36e(a)
    rs = a.rs'''),
("run state",
r'''    s11_store = {}
    end33 = None
    end34 = None''',
r'''    s11_store = {}
    end33 = None
    end34 = None
    end36e = None
    if FR["on"]:
        first_return_init_v36e(a, recs if a.resume_slice else [])'''),
("record entry",
r'''        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy()); fields["eps_tot"].append(eps.copy()); fields["J_tot"].append(J.copy())''',
r'''        fr36e = None; returned36e = False
        if FR["on"]:
            fr36e, returned36e = first_return_record_v36e(a, info, R, rs, T, Rsh, tz["zeros"])
            rec["first_return_v36e"] = fr36e
            rec["engine_v36e"] = first_return_desc_v36e(a)
        recs.append(rec)
        fields["T"].append(T); fields["N"].append(N.copy()); fields["v"].append(v.copy()); fields["eps_tot"].append(eps.copy()); fields["J_tot"].append(J.copy())'''),
("record line",
r'''        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)''',
r'''        print(_line + (v32_record_segment(solve32) if solve32 is not None else ""), flush=True)
        if fr36e is not None:
            print("T/rs %.4f %s" % (T / rs, first_return_line_v36e(fr36e)), flush=True)'''),
("end rule",
r'''        if a.segment_end_T_over_rs is not None and T / rs >= a.segment_end_T_over_rs - 1e-9:
            end34 = {"T": T, "T_over_rs": T / rs, "segment_end_T_over_rs": a.segment_end_T_over_rs, "rule": "version 34 (b): the segment ends after this record"}''',
r'''        if FR["on"] and returned36e and a.first_return_end_rule == "end":
            end36e = {"T": T, "T_over_rs": T / rs, "first_return": fr36e, "collar_over_rs": a.first_return_collar_over_rs,
                      "rule": "version 36e: the chain end's first return into the collar at this record (the outgoing null ray launched at the formation record, reflected at the chain's end, back at the collar's inner radius); the record is written, nothing propagated from it"}
            print("T/rs %.4f: version 36e: the chain end's first return into the collar (%g r_s): the run ends after this record" % (T / rs, a.first_return_collar_over_rs), flush=True)
            break
        if a.segment_end_T_over_rs is not None and T / rs >= a.segment_end_T_over_rs - 1e-9:
            end34 = {"T": T, "T_over_rs": T / rs, "segment_end_T_over_rs": a.segment_end_T_over_rs, "rule": "version 34 (b): the segment ends after this record"}'''),
("final dump",
r'''        json.dump(dict({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {})), fh, default=float)''',
r'''        json.dump(dict({"script": "engine_collapse_v34.py", "args": vars(a), "head": head, "records": recs}, **({"end_v32": end32} if end32 is not None else {}),
                       **({"end_s11": end33} if end33 is not None else {}), **({"end_v34": end34} if end34 is not None else {}),
                       **({"end_v36e": end36e} if end36e is not None else {})), fh, default=float)'''),
("arguments",
r'''    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")
    a = ap.parse_args()''',
r'''    ap.add_argument("--nk-set", dest="nk_set", default=None, choices=["resolve"], help="(v36c) the set-resolving Newton solve")
    ap.add_argument("--first-return-collar-over-rs", dest="first_return_collar_over_rs", type=float, default=None, help="(v36e) the collar's inner radius in r_s: the first-return estimate per record and, under --first-return-end-rule end, the end at the chain end's first return into it")
    ap.add_argument("--first-return-launch-over-rs", dest="first_return_launch_over_rs", type=float, default=None, help="(v36e) the ray's launch radius in r_s at the formation record (default: that record's R_sh, or its outer zero plus the averaging scale)")
    ap.add_argument("--first-return-end-rule", dest="first_return_end_rule", default="end", choices=["end", "print"], help="(v36e) end: the run ends after the first record at or past the return; print: the estimate alone")
    a = ap.parse_args()'''),
("validation",
r'''    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True
    C5.backend(a.gpu)
    run(a)''',
r'''    if a.nk_set == "resolve":
        if a.nk_jvp != "exact" or a.nk_line_search != "halving":
            ap.error("version 36c: --nk-set resolve needs --nk-jvp exact and --nk-line-search halving")
        NSR["on"] = True
    FR["on"] = a.first_return_collar_over_rs is not None
    if a.first_return_launch_over_rs is not None and not FR["on"]:
        ap.error("version 36e: --first-return-launch-over-rs acts with --first-return-collar-over-rs")
    if FR["on"]:
        if not (0.0 < a.first_return_collar_over_rs * a.rs < a.nsites):
            ap.error("version 36e: the collar's inner radius must lie inside the chain (0 < X r_s < nsites a)")
        if a.first_return_launch_over_rs is not None and not (0.0 < a.first_return_launch_over_rs * a.rs < a.nsites):
            ap.error("version 36e: the launch radius must lie inside the chain (0 < Y r_s < nsites a)")
    C5.backend(a.gpu)
    run(a)'''),
]


def build(text):
    """the forward replacements on the source text; every OLD must occur exactly once. Returns the built text."""
    for name, old, new in REPLACEMENTS:
        n = text.count(old)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36e: the OLD string of '%s' occurs %d times in the source (1 required): the source is not input 02's engine_collapse_v36d.py" % (name, n))
        text = text.replace(old, new)
    return text


def reverse(text):
    """the reverse replacements (NEW -> OLD, in reverse order); every NEW must occur exactly once."""
    for name, old, new in reversed(REPLACEMENTS):
        n = text.count(new)
        if n != 1:
            raise SystemExit("build_engine_collapse_v36e: the NEW text of '%s' occurs %d times in the built file (1 required)" % (name, n))
        text = text.replace(new, old)
    return text


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "engine_collapse_v36d.py")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(src)), "engine_collapse_v36e.py")
    raw = open(src, "rb").read()
    text = raw.decode("utf-8")
    out = build(text)
    back = reverse(out)
    if back != text:
        raise SystemExit("build_engine_collapse_v36e: the reverse replacements do not restore the source (a NEW text overlaps another)")
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))
    py_compile.compile(dst, doraise=True)
    print("built %s from %s: %d replacements; source md5 %s; built md5 %s; the reverse replacements restore the source byte for byte; compiled" % (
        dst, src, len(REPLACEMENTS), md5(raw), md5(out.encode("utf-8"))))


if __name__ == "__main__":
    main()
