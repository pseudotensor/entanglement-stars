#!/usr/bin/env python
"""sync_ext_check_v1.py (builder analyst, 2026-10-17): the validation of one staged copy of a running engine's record json and fields npz
(sync_ext_to_pod1_v1.sh): the json parses, the npz loads whole (every array read, so a truncated zip fails), and the records present in both
(T in the npz with a json record carrying E, consecutive from the first) are counted. Prints one line: "OK <records> <last T/rs> <md5 json>
<md5 npz>" or "BAD <reason>"; exit 0 either way. Usage: python -I sync_ext_check_v1.py JSON NPZ"""
import hashlib
import io
import json
import sys

import numpy as np


def main():
    pj, pf = sys.argv[1:3]
    try:
        bj = open(pj, "rb").read(); bf = open(pf, "rb").read()
        D = json.loads(bj.decode("utf-8"))
        Z = np.load(io.BytesIO(bf), allow_pickle=True)
        arrs = {k: Z[k] for k in Z.files}
        T = np.asarray(arrs["T"], float)
        if np.asarray(arrs["N_applied"]).shape[0] != len(T) or np.asarray(arrs["v_applied"]).shape[0] != len(T):
            print("BAD the applied fields and T differ in length"); return
        rs = float(D["args"]["rs"])
        byT = {round(float(r["T"]), 6) for r in D["records"] if isinstance(r, dict) and r.get("E") is not None}
        n = 0
        for t in T:
            if round(float(t), 6) not in byT:
                break
            n += 1
        if n == 0:
            print("BAD no record in both files"); return
        print("OK %d %.6f %s %s" % (n, float(T[n - 1]) / rs, hashlib.md5(bj).hexdigest(), hashlib.md5(bf).hexdigest()))
    except Exception as ex:                                               # noqa: BLE001 (a partial copy: retried at the next poll)
        print("BAD %s: %s" % (type(ex).__name__, str(ex).replace("\n", " ")[:200]))


if __name__ == "__main__":
    main()
