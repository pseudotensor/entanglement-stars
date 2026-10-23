#!/usr/bin/env python
"""compare_core_slices_v3_v2_v1.py (builder analyst; engine spec 37.7): engine_core_slices_v3's docstring says its version 2 columns are
version 2's computation unchanged; per slice, every version 2 number (N_core, the dust count, the normal-ordered number, R_pack, the packing
ratio, B, H_core, H at the median node, D, D/V_core, the version 1 column) of a version 3 json against the version 2 json's: the count of
slices, the largest relative difference per column, and the verdict (bitwise, or the largest difference). Args: --v2 --v3."""
import argparse
import json

COLS = ("R_c", "N_core", "N_dust_core", "N_normal_ordered_core", "R_pack", "packing_ratio", "B", "H_core_mean", "H_median", "defect", "defect_per_volume", "v1_median_rho_dot_minus_rhs")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--v2", required=True); ap.add_argument("--v3", required=True)
    a = ap.parse_args()
    r2 = {r["slice"]: r for r in json.load(open(a.v2))["rows"]}; r3 = {r["slice"]: r for r in json.load(open(a.v3))["rows"]}
    common = sorted(set(r2) & set(r3))
    print("slices: version 2 %d, version 3 %d, common %d" % (len(r2), len(r3), len(common)))
    worst = {}; nb = 0; n = 0
    for s in common:
        for c in COLS:
            x, y = r2[s].get(c), r3[s].get(c)
            if x is None and y is None:
                continue
            n += 1
            if x == y:
                nb += 1
            d = abs(y - x) / max(abs(x), 1e-300) if (x is not None and y is not None) else float("inf")
            worst[c] = max(worst.get(c, 0.0), d)
    for c in COLS:
        if c in worst:
            print("  %s: largest relative difference %.3e" % (c, worst[c]))
    w = max(worst.values()) if worst else 0.0
    print("values compared %d, equal to the bit %d; the largest relative difference %.3e: %s" % (n, nb, w, "bitwise" if nb == n else ("within 1e-12" if w <= 1e-12 else "DIFFERS")))


if __name__ == "__main__":
    main()
