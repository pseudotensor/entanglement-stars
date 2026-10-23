#!/usr/bin/env python
"""filter_md_rows_v1.py (builder analyst): a shortened copy of a builder table for a red team's input budget (requests under 50,000
characters): every line kept except the data rows of markdown tables whose first cell (T/rs) is not a multiple of --every; a note naming the
full table is inserted after the first heading. Nothing is recomputed. Args: --in --out --every --full-name."""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=float, default=0.5); ap.add_argument("--full-name", required=True)
    a = ap.parse_args()
    out = []; noted = False; dropped = 0
    for l in open(a.inp).read().split("\n"):
        if l.startswith("| ") and not l.startswith("|---"):
            c = l.split("|")[1].strip()
            try:
                t = float(c)
            except ValueError:
                out.append(l); continue
            q = t / a.every
            if abs(q - round(q)) > 1e-9:
                dropped += 1; continue
        out.append(l)
        if not noted and l.startswith("#"):
            out.append(""); out.append("(A shortened copy for the red team's input budget: the table rows every %g r_s only; the full table, every record, is %s; nothing recomputed.)" % (a.every, a.full_name)); noted = True
    open(a.out, "w").write("\n".join(out))
    print("kept %d lines, dropped %d rows" % (len(out), dropped))


if __name__ == "__main__":
    main()
