#!/usr/bin/env python
"""star_ladder_v1.py (builder analyst; engine spec 26.4 (4): "Sfam8 and Sfam7 continue as ladders of members beyond their N_B,max with the
ell control at every such member, stopping at the first member with min(N - |v|) below zero or an ell spread above 10 percent, whichever
first (BUILDER: new launcher files; the current member finishes)"): runs engine_star_v8 one member at a time on the given GPU over --NB-list
(ascending, each beyond the family's continuum N_B,max), each invocation with --NB-list that one value and --ell-control 4,8,16 (so the
control is at that member), and after each member reads its json: the member's min(N - |v|) and the ell controls' spread (largest minus
smallest converged E over E at the family's ell); stops at the first member with min(N - |v|) <= 0 or a spread above --spread-stop (0.1),
or at a member that fails to converge (recorded; the ladder cannot read past it); a status line per member to --status. Args: --m --g
--ell --nsites --NB-list --out --tag-prefix --status --gpu."""
import argparse
import glob
import json
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, required=True); ap.add_argument("--g", type=float, required=True)
    ap.add_argument("--ell", type=float, default=8.0); ap.add_argument("--nsites", type=int, default=3072)
    ap.add_argument("--NB-list", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag-prefix", required=True)
    ap.add_argument("--status", required=True); ap.add_argument("--spread-stop", type=float, default=0.1)
    ap.add_argument("--eos", default="EOS/eos_v4_tables.npz"); ap.add_argument("--ovpg", default="OVPG/ovpg_families.json")
    a = ap.parse_args()
    py = sys.executable
    os.makedirs(a.out, exist_ok=True)
    say = lambda txt: open(a.status, "a").write("star ladder %s %s %s UTC\n" % (a.tag_prefix, txt, time.strftime("%H:%M:%S", time.gmtime())))
    for nb in sorted(float(x) for x in a.NB_list.split(",")):
        tag = "%s_NB%.0f" % (a.tag_prefix, nb)
        say("member N_B %.3f start" % nb)
        log = open(os.path.join(a.out, "log_%s.log" % tag), "w")
        rc = subprocess.call([py, "code/engine_star_v8.py", "--m", repr(a.m), "--g", repr(a.g), "--ell", repr(a.ell), "--nsites", str(a.nsites), "--eos", a.eos,
                              "--ovpg", a.ovpg, "--NB-list", "%.6f" % nb, "--ell-control", "4,8,16", "--gpu", "0", "--out", a.out, "--tag", tag], stdout=log, stderr=subprocess.STDOUT)
        log.close()
        js = sorted(glob.glob(os.path.join(a.out, "*%s*.json" % tag)), key=os.path.getmtime)
        if rc != 0 or not js:
            say("member N_B %.3f exit %d (no record): the ladder stops" % (nb, rc)); return
        d = json.load(open(js[-1]))
        mem = [q for q in d.get("members", []) if abs(q.get("NB_target", q.get("N_B", -1)) - nb) < 1e-3 or abs(q.get("N_B", -1) - nb) < 1e-3]
        if not mem or mem[-1].get("failed"):
            say("member N_B %.3f did not converge (%s): the ladder stops" % (nb, (mem[-1].get("flags") if mem else "no member"))); return
        q = mem[-1]
        spread = None
        for c in d.get("ell_control") or []:
            rows = c.get("rows", {})
            Es = [r["E"] for r in rows.values() if not r.get("failed")]
            own = rows.get("%g" % a.ell, {}).get("E")
            if len(Es) >= 2 and own:
                spread = (max(Es) - min(Es)) / abs(own)
        say("member N_B %.3f converged: E %.6f, min(N - |v|) %+.4f, ell spread %s" % (nb, q["E"], q["min_N_minus_absv"], ("%.4f" % spread) if spread is not None else "-"))
        if q["min_N_minus_absv"] <= 0.0:
            say("stop: min(N - |v|) <= 0 at N_B %.3f" % nb); return
        if spread is not None and spread > a.spread_stop:
            say("stop: the ell spread %.4f above %g at N_B %.3f" % (spread, a.spread_stop, nb)); return
    say("the list ended")


if __name__ == "__main__":
    main()
