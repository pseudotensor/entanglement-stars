#!/usr/bin/env python
"""compose_handoff_head_v1.py (builder analyst; engine spec 30.6 (6): "FILE 3 appended to STAR_family_statement_v3.1 as version 3.2 and
carried at handoff version 5's head"): inserts a statement file (its header line and paragraphs, verbatim) into a printed handoff's head
after the last statement block and before the head's "---" separator, nothing else changed; asserts the printed head ends with the
expected statement (--after: the header prefix of the block it must follow). Args: --handoff --statement --after --out."""
import argparse


def main():
    ap = argparse.ArgumentParser()
    for k in ("handoff", "statement", "after", "out"):
        ap.add_argument("--" + k, required=True)
    a = ap.parse_args()
    L = open(a.handoff).read().split("\n")
    sep = L.index("---")
    heads = [i for i in range(sep) if L[i].startswith("# ")]
    assert heads and L[heads[-1]].startswith(a.after), "the head's last statement is not %r: %r" % (a.after, L[heads[-1]][:80] if heads else None)
    st = open(a.statement).read().strip("\n").split("\n")
    assert st[0].startswith("# "), st[0][:80]
    j = sep
    while j > 0 and not L[j - 1].strip():
        j -= 1
    out = L[:j] + [""] + st + [""] + L[sep:]
    open(a.out, "w").write("\n".join(out))
    print("inserted %d lines before the separator at line %d" % (len(st), sep + 1))


if __name__ == "__main__":
    main()
