#!/usr/bin/env python
"""apply_lead_result_v4.py (builder analyst; a new file after versions 1 to 3, written without the earlier ones at hand): applies a lead turn's answer file mechanically, as the lead's blocks name them:
"=== APPEND <path> ===" ... "=== END APPEND ===" appended to <path>; "=== INSERT <path> BEFORE <marker> ===" ... "=== END INSERT ===" inserted
before the first line equal to <marker>; "=== FILE <path> ===" ... "=== END FILE ===" written verbatim to <path>, refused when <path> exists
(the lane never overwrites a delivery; the builder then picks the next version by hand and records it); "=== HANDBACK ===" ... "=== END
HANDBACK ===" written to --handback. Paths beginning "solve/" resolve against --worktree. Prints what it did; --dry-run prints only.
Args: --result --worktree --handback [--dry-run]."""
import argparse
import os
import re


def resolve(p, wt):
    return os.path.join(wt, p[len("solve/"):]) if p.startswith("solve/") else p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True); ap.add_argument("--worktree", required=True); ap.add_argument("--handback", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    txt = open(a.result).read()
    blocks = re.findall(r"^=== (APPEND|INSERT|FILE) (.+?) ===\n(.*?)\n=== END \1 ===$", txt, re.S | re.M)
    hb = re.search(r"^=== HANDBACK ===\n(.*?)\n=== END HANDBACK ===$", txt, re.S | re.M)
    for kind, head, body in blocks:
        if kind == "APPEND":
            p = resolve(head.strip(), a.worktree)
            assert os.path.exists(p), p
            print("APPEND %d lines to %s" % (body.count("\n") + 1, p))
            if not a.dry_run:
                cur = open(p).read()
                open(p, "a").write(("" if cur.endswith("\n\n") else ("\n" if cur.endswith("\n") else "\n\n")) + body.rstrip("\n") + "\n")
        elif kind == "INSERT":
            m = re.match(r"(.+?) BEFORE (.+)$", head.strip())
            assert m, head
            p, marker = resolve(m.group(1).strip(), a.worktree), m.group(2).strip()
            L = open(p).read().split("\n")
            idx = [i for i, l in enumerate(L) if l.strip() == marker]
            assert len(idx) >= 1, "marker %r not in %s" % (marker, p)
            print("INSERT %d lines into %s before line %d (%s)" % (body.count("\n") + 1, p, idx[0] + 1, marker))
            if not a.dry_run:
                L = L[:idx[0]] + body.rstrip("\n").split("\n") + L[idx[0]:]
                open(p, "w").write("\n".join(L))
        else:
            p = resolve(head.strip(), a.worktree)
            if os.path.exists(p):
                print("REFUSED FILE %s: it exists (pick the next version by hand)" % p)
                continue
            print("FILE %s (%d lines)" % (p, body.count("\n") + 1))
            if not a.dry_run:
                open(p, "w").write(body.rstrip("\n") + "\n")
    if hb:
        print("HANDBACK to %s (%d lines)" % (a.handback, hb.group(1).count("\n") + 1))
        if not a.dry_run:
            open(a.handback, "w").write(hb.group(1).rstrip("\n") + "\n")


if __name__ == "__main__":
    main()
