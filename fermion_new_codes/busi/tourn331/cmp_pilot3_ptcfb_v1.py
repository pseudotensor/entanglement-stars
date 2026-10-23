#!/usr/bin/env python
"""cmp_pilot3_ptcfb_v1.py (busi, analyst; #331): the PASS reading of the L4 pilot (pilot3_ptcfb_RUNG2c: RUNG2c40_v36q's line on
engine_collapse_v36z.py with --nk-ptc fallback --nk-ptc-delta0 1, from the member's T0002.00 slice to 4.25), TOURNAMENT_reading_v1 sec. 4.4:
  P1  the records 2.125 to 3.75 are RUNG2c40_v36q's bitwise: every record's json equal but walls and the options' "ptc" / "ptc_fallback" / "srf"
      entries (E/E0 and the kept residuals to every digit); no ladder fired there (no "ptc_fallback" entry in them);
  P2  3.875 converged at the 8 a step (the record follows 3.75 by tau = 0.125 r_s, no sub-record) and its solve carries a "ptc_fallback" entry
      (the ladder ran from the failed Newton step) and the ladder's steps ("ptc_fallback_from_newton_step");
  P3  4.0, 4.125 and 4.25 converged at the 8 a step;
  READ  E/E0 at 3.875 to 4.25 against the PTC pilot's (pilot2_ptc_RUNG2c, if --ptc json given) and against the member's tau drift.
Reads two or three run jsons; no engine, no device. Prints one line per check and PASS or FAIL.
    python cmp_pilot3_ptcfb_v1.py --pilot .../collapse_pilot3_ptcfb_RUNG2c.json --member .../collapse_RUNG2c40_v36q.json [--ptc .../collapse_pilot2_ptc_RUNG2c.json]
"""
import argparse
import json

DROP = ("ptc", "ptc_fallback", "srf")


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in DROP and k != "slice_saved" and not k.endswith("_s") and not any(w in k for w in ("second", "timing", "wall"))}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def find(x, key):
    """every value under `key` anywhere in x."""
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            if k == key:
                out.append(v)
            out += find(v, key)
    elif isinstance(x, list):
        for v in x:
            out += find(v, key)
    return out


def recs(J):
    return {round(r["T_over_rs"], 6): r for r in J["records"] if r.get("T_over_rs") is not None}


def kept(r):
    u = r.get("unconverged")
    return (u or {}).get("residual") if u else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--ptc", default=None)
    a = ap.parse_args()
    P, M = recs(json.load(open(a.pilot))), recs(json.load(open(a.member)))
    Q = recs(json.load(open(a.ptc))) if a.ptc else {}
    fails = []
    rep = [t for t in sorted(P) if 2.0 < t <= 3.75 + 1e-9]
    want = [round(2.0 + 0.125 * k, 6) for k in range(1, 15)]
    if rep != want:
        fails.append("P1: the pilot's records in (2.0, 3.75] are %s, not the grid %s" % (rep, want))
    bad = [t for t in want if t in P and t in M and json.dumps(strip(P[t]), sort_keys=True, default=float) != json.dumps(strip(M[t]), sort_keys=True, default=float)]
    lad = [t for t in want if t in P and any(isinstance(v, dict) for v in find(P[t], "ptc_fallback"))]   # the options entry (a string) is not a hand-over
    print("P1 2.125 to 3.75 bitwise RUNG2c40_v36q (records but walls and the ptc/srf entries): %s; ladder fired there: %s" % ("yes" if not bad else "NO at %s" % bad, lad or "no"))
    if bad or lad:
        fails.append("P1: records differ at %s; ladder at %s" % (bad, lad))
    t1 = 3.875
    r = P.get(t1)
    if r is None:
        fails.append("P2: no 3.875 record")
    else:
        pf = [v for v in find(r, "ptc_fallback") if isinstance(v, dict)]
        nl = len(find(r, "ptc_fallback_from_newton_step"))
        ok = bool(r.get("converged")) and bool(pf) and nl > 0 and (t1 - 0.125) in P and not find(r, "srf")
        print("P2 3.875: converged %s (kept %s), at 8 a (follows 3.75, no sub-record): %s; hand-overs %s; ladder steps %d" % (
            r.get("converged"), kept(r), ((t1 - 0.125) in P), [(q.get("from_newton_step"), q.get("why"), q.get("status"), q.get("ladder_steps")) for q in pf], nl))
        if not ok:
            fails.append("P2: 3.875 not converged at 8 a by the ladder")
    for t in (4.0, 4.125, 4.25):
        r = P.get(t)
        ok = r is not None and bool(r.get("converged"))
        print("P3 %.3f: %s" % (t, ("converged" if ok else ("missing" if r is None else "NOT converged (kept %s)" % kept(r)))))
        if not ok:
            fails.append("P3: %.3f" % t)
    sub = [t for t in P if t > 3.75 and abs((t / 0.125) - round(t / 0.125)) > 1e-6]
    if sub:
        fails.append("off-grid records %s (a sub-stepped stretch: not this pilot's line)" % sub)
    for t in (3.875, 4.0, 4.125, 4.25):
        if t in P:
            print("READ %.3f: E/E0 %.10f%s; member %s" % (t, P[t].get("E_over_E0"), ("; PTC pilot %.10f (diff %+.2e)" % (Q[t]["E_over_E0"], P[t]["E_over_E0"] - Q[t]["E_over_E0"])) if t in Q else "",
                                                     ("%.10f" % M[t]["E_over_E0"]) if t in M else "none"))
    for f_ in fails:
        print("FAIL: " + f_)
    print("PILOT3 %s (%d checks failed)" % ("PASS" if not fails else "FAIL", len(fails)))
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
