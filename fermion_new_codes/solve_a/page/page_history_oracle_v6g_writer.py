#!/usr/bin/env python
"""page_history_oracle_v6g_writer.py (builder analyst, 2026-10-17): the v6g oracle's stand-in for a running continuation. It copies a fixture
history (a fields npz and a record json in the engine's layout) into OUT_JSON and OUT_NPZ record by record, rewriting both files whole the way
engine_collapse_v19 does (the json first, then the fields npz, neither atomically): each file's bytes are written in two halves with a pause
between them, so a reader that polls meets partial files. It starts with the first --start-records records, adds one every --interval-s
seconds, and when every record is written it creates --marker (if given). Usage: python page_history_oracle_v6g_writer.py FIX_NPZ FIX_JSON
OUT_JSON OUT_NPZ --start-records N --interval-s 0.4 [--pause-s 0.05] [--marker PATH]"""
import argparse
import io
import json
import time

import numpy as np


def write_slow(path, data, pause):
    h = len(data) // 2
    with open(path, "wb") as f:
        f.write(data[:h]); f.flush()
        time.sleep(pause)
        f.write(data[h:]); f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fix_npz"); ap.add_argument("fix_json"); ap.add_argument("out_json"); ap.add_argument("out_npz")
    ap.add_argument("--start-records", type=int, required=True); ap.add_argument("--interval-s", type=float, default=0.4)
    ap.add_argument("--pause-s", type=float, default=0.05); ap.add_argument("--marker", default=None)
    a = ap.parse_args()
    Z = dict(np.load(a.fix_npz, allow_pickle=True))
    J = json.load(open(a.fix_json))
    n_all = len(Z["T"])
    per_record = [k for k, v in Z.items() if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_all and k != "R"]
    n = a.start_records
    while True:
        recs = [r for r in J["records"] if r["T"] <= float(Z["T"][n - 1]) + 1e-9]
        write_slow(a.out_json, json.dumps({"args": J["args"], "records": recs}).encode("utf-8"), a.pause_s)
        sub = {k: (v[:n] if k in per_record else v) for k, v in Z.items()}
        buf = io.BytesIO()
        np.savez_compressed(buf, **sub)
        write_slow(a.out_npz, buf.getvalue(), a.pause_s)
        print("writer: %d of %d records written at %.3f" % (n, n_all, time.time()), flush=True)
        if n >= n_all:
            break
        n += 1
        time.sleep(a.interval_s)
    if a.marker:
        open(a.marker, "w").write("done: %d records\n" % n_all)
        print("writer: the marker %s written" % a.marker, flush=True)


if __name__ == "__main__":
    main()
