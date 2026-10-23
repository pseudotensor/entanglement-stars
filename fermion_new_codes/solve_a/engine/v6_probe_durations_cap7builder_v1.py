import re, sys
import numpy as np
for path in sys.argv[1:]:
    print("==", path)
    prev_key, prev_t, start_t = None, 0.0, 0.0
    rows = []
    for line in open(path):
        m = re.match(r"^(.{12}) \| (.*?) omega ([0-9.]+): \|kappa\| (\d+): .*\((\d+) s\)\s*$", line)
        if not m:
            continue
        key = (m.group(1), m.group(2)); ka = int(m.group(4)); t = float(m.group(5))
        if key != prev_key:
            if prev_key is not None:
                rows.append((prev_key, last_ka, start_t, prev_t))
            start_t = prev_t; prev_key = key
        last_ka = ka; prev_t = t
    if prev_key is not None:
        rows.append((prev_key, last_ka, start_t, prev_t))
    for (bg, pl), ka, t0, t1 in rows:
        print("%s | %s: channels to |kappa| %d, %.0f s (log seconds %.0f to %.0f)" % (bg, pl, ka, t1 - t0, t0, t1))
