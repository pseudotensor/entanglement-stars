#!/usr/bin/env python3
"""j4production_scan.py -- the record scanner shared by the version-2 and prefix drivers (the function of
j4_production_v2.scan_done, importable without the version-2 driver's argument parsing)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import j4window_v2 as jw  # noqa: E402


def scan_done(dirs):
    """(canonical ball, L) -> path of an existing record with an entropy."""
    done = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.json'):
                continue
            p = os.path.join(d, f)
            try:
                with open(p) as fh:
                    r = json.load(fh)
            except Exception:
                continue
            if isinstance(r, dict) and all(k in r for k in ('S', 'n2max', 'L')) and r['S'] == r['S']:
                done[(jw.canonical_ball(int(r['n2max'])), int(r['L']))] = p
    return done
