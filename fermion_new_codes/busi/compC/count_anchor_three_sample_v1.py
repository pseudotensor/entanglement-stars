#!/usr/bin/env python
"""count_anchor_three_sample_v1.py (busi lane builder, 20 Oct; REVIEWA_C64_toy_readout_v1 sec. 1, the reporting requirement on every G4 run, not a
gate): over a run's json, every primitive entry (source "primitive" or "walk+primitive") of the passage blocks' crossings, light and heavy: the
decided events (kind passage or incomplete, decided True) and among them those with exactly 3 samples one of which lies at the event's s_a.

The entries carry "samples" (since v9c1b) and "piece" = [s_p, s_c] but not s_a (the primitive's event has it; the engine's entry does not copy it).
So the count printed is an upper bound: an entry with exactly 3 samples one of which lies strictly before s_p (the anchor's sample sits at s_a < s_p;
a window sample can too). Zero is therefore exact; a nonzero bound is reported with the entries, and comes back to the desk.

Usage. python count_anchor_three_sample_v1.py RUN.json [RUN.json ...]
"""
import json
import sys


def entries(rec):
    for sp, chans in (("light", (rec.get("light_v36") or {}).get("channels") or {}), ("heavy", rec.get("continuation") or {})):
        for ch, c in (chans.items() if isinstance(chans, dict) else []):
            sec = (c or {}).get("sector") or {}
            for e in sec.get("crossings") or []:
                if "primitive" in str(e.get("source")):
                    yield sp, ch, e


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tot_dec = tot_hit = tot_ent = 0
    for path in argv:
        J = json.load(open(path)); n_ent = n_dec = 0; hits = []
        for rec in J.get("records") or []:
            for sp, ch, e in entries(rec):
                n_ent += 1
                if e.get("kind") in ("passage", "incomplete") and e.get("decided"):
                    n_dec += 1
                    smp = e.get("samples") or []; sp_ = (e.get("piece") or [None])[0]
                    if len(smp) == 3 and sp_ is not None and any(float(s) < float(sp_) for s, _d in smp):
                        hits.append((rec.get("T_over_rs"), sp, ch, e.get("s_star"), smp))
        tot_dec += n_dec; tot_hit += len(hits); tot_ent += n_ent
        print("%s: primitive entries %d; decided events (passage or incomplete) %d; with exactly 3 samples one before s_p (bound on 'one at s_a') %d%s" % (
            path, n_ent, n_dec, len(hits), (" %s" % hits[:5]) if hits else ""))
    print("total: %d of %d decided events (primitive entries %d)" % (tot_hit, tot_dec, tot_ent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
