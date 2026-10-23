#!/usr/bin/env python
"""split_prep_v1.py (busi; inbox #463: the owner-box arms must finish and copy off small outputs by 20:45, and the measured copy rate from
the owner box is about 4.3 MB/s, so 8.8 GB checkpoints cannot be moved): writes, for each given prep, a supply-only and an exit-only prep
(<tag>_sup_prep.npz: the 128 supply columns; <tag>_exit_prep.npz: the 128 exit columns, n_supply 0), same fields, same basis vectors
(a column subset of the orthonormal prep_v5 basis, unchanged). The supply-exit cross block of G is then not measured by these arms; it is
measured by the full-span arm on h100-f."""
import json
import sys

import numpy as np

for p in sys.argv[1:]:
    Z = np.load(p)
    d = {k: Z[k] for k in Z.files}
    ns = int(Z["n_supply"]); K = len(Z["e"])
    for lab, idx in (("sup", np.arange(ns)), ("exit", np.arange(ns, K))):
        o = dict(d)
        o["U_sub"] = Z["U_sub"][:, idx]; o["e"] = Z["e"][idx]; o["qF"] = Z["qF"][np.ix_(idx, idx)]
        o["n_supply"] = ns if lab == "sup" else 0
        m = json.loads(str(Z["manifest"])); m["column_subset"] = {"window": lab, "of": p.split("/")[-1], "columns": [int(idx[0]), int(idx[-1])]}
        o["manifest"] = json.dumps(m)
        q = p.replace("_prep.npz", "_%s_prep.npz" % lab)
        np.savez_compressed(q, **o)
        json.dump(m, open(q.replace(".npz", ".json"), "w"), indent=1, default=str)
        print(q, o["U_sub"].shape, o["n_supply"])
