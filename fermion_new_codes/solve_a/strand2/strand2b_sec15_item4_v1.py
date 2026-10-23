#!/usr/bin/env python
"""Strand 2b sec. 15 item 4 (builder analyst): from jacgap2/jacgap_v2.json, D - E O_free with no fitted scale, D the recorded Jacobian
difference (FD minus analytic) in the adjoint rows' psi columns at the cube (2,0) solution of the k = 1 timing case, O_free the undressed
norm-defect companion 2 Re <w_y|w_z>; E = E_(2,0) = the dressed (2,0) state's energy and E = E_free = the free (2,0) expectation, both as
recorded there.  Reports max |D - E O_free|/max |D| for each E, and beside it the same with the opposite sign (D + E O_free).
Writes <out>/item4.json."""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jacgap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    J = json.load(open(a.jacgap))
    D = np.array(J["D_adjoint_psi"], float)
    O = np.array(J["O_free"], float)
    mD = float(np.max(np.abs(D)))
    res = {"max_abs_D": mD, "E20_dressed": J["E20_dressed"], "E20_free_expectation": J["E20_free_expectation"], "rows": []}
    for name, E in (("E_(2,0) dressed", J["E20_dressed"]), ("E_free", J["E20_free_expectation"])):
        for sg, lab in ((-1, "D - E O_free"), (+1, "D + E O_free")):
            r = float(np.max(np.abs(D + sg * E * O)) / mD)
            res["rows"].append({"E": name, "E_value": E, "combination": lab, "rel": r, "below_1e-2": bool(r < 1e-2)})
    print(json.dumps(res, indent=1))
    with open(os.path.join(a.out, "item4.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
