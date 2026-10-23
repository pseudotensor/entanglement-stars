#!/usr/bin/env python
"""rd1d_trace_read_v1.py (busi, PERF RD-1d (c)): the memtrace files of the arms, per phase path: count, max entry, max exit, max peak (GiB
allocated); and per arm the peak of each stage of the continuation record. Prints a table.
    python rd1d_trace_read_v1.py ARM_DIR [ARM_DIR ...]"""
import collections
import re
import os
import sys


RX = re.compile(r"MEMTRACE (\S+) (.+?) T/rs (\S+) entry (\S+) exit (\S+) peak (\S+)")


def read(p):
    rows = collections.OrderedDict()
    for ln in open(p):
        if not ln.startswith("MEMTRACE ") or " entry " not in ln:
            continue
        m = RX.match(ln)
        if m is None:
            continue
        path, T = m.group(2), m.group(3)
        en, ex, pk = float(m.group(4)), float(m.group(5)), float(m.group(6))
        k = (T, path)
        r = rows.setdefault(k, [0, 0.0, 0.0, 0.0])
        r[0] += 1; r[1] = max(r[1], en); r[2] = max(r[2], ex); r[3] = max(r[3], pk)
    return rows


def main():
    for d in sys.argv[1:]:
        p = os.path.join(d, "memtrace.txt")
        print("== %s" % d)
        if not os.path.exists(p):
            print("   (no trace)"); continue
        for (T, path), (n, en, ex, pk) in read(p).items():
            print("   T/rs %s  %-70s n %4d  entry max %7.3f  exit max %7.3f  peak %7.3f GiB" % (T, path, n, en, ex, pk))


if __name__ == "__main__":
    main()
