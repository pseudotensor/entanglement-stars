#!/usr/bin/env python
"""test_w10_defect6_setrule.py (busi W10): defect 6 of REVIEWA_REVIEW_web40i section 5. The occupied-set rule of engine_collapse_v36d
(--nk-set resolve, rounds_v36c): the fewer-sorted-slot-flips criterion can prefer refilling (the masks index energy-sorted eigenvectors);
m_.size == prev_m.size admits broadcasting of shapes (2, 1) and (2,); len(meet) > 1 counts rounds, not distinct sets. Gate: the
two-level counterexample and the 8-dimensional overlap example as unit tests.
(1) two levels, one occupied, the levels cross between the previous record and the kept iterate: the continued set has 2 sorted-slot
    flips and the refilled set 0, so the v36d rule keeps the refilled set; the subspace classifier (D = n - ||R_prev^dag R||_F^2) gives
    D 0 for the continued set and 1 for the refilled one and keeps the continued set;
(2) 8 dimensions, n_sea 4, orbitals 3, 4 rotated into 5, 6 by 70 degrees: the set with 0 sorted-slot flips has D = 2 sin^2(70) = 1.77,
    the set with 4 flips has D = 2 cos^2(70) = 0.23; the classifier keeps the latter, as the engine's own maximum-overlap continuation
    (engine_ops_v3.assign_max_overlap_v3, when importable) does;
(3) the broadcast: np.sum(a != b) with shapes (2, 1) and (2,) counts a (2, 2) array (2) although .size agrees; the shape-checked count is None;
(4) the same-set case: rounds 1 and 3 meeting on one set are one distinct set (v36d labelled them "two self-consistent sets").
Run: qi2 python test_w10_defect6_setrule.py"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def rule_v36d(cands, prev_mask):
    """v36d: fewer sorted-slot flips from the previous record's mask, ties by the larger smallest gap."""
    key = lambda c: (int(np.sum(c["mask"] != prev_mask)), -c["gap"])
    return min(cands, key=key)


def rule_v36e(cands, R_prev):
    """v36e: the smaller subspace distance D, ties within 1e-6 by the larger smallest gap."""
    for c in cands:
        c["D"] = W10.subspace_flips({1: c["R"]}, {1: R_prev})
    Ds = sorted(c["D"] for c in cands)
    if Ds[1] - Ds[0] <= 1e-6 * max(1.0, Ds[1]):
        return max(cands, key=lambda c: c["gap"])
    return min(cands, key=lambda c: (c["D"], -c["gap"]))


def main():
    # (1) the two-level counterexample
    prev_mask = np.array([True, False])
    R_prev = np.array([[1.0], [0.0]])                      # the previous record's occupied orbital: the lower level e_1
    U_new = np.array([[0.0, 1.0], [1.0, 0.0]])             # after the crossing the old upper state is slot 0 and the old occupied state slot 1
    A = {"name": "A (continued: the old occupied state, now the upper slot)", "mask": np.array([False, True]), "R": U_new[:, [1]], "gap": 0.3}
    B = {"name": "B (refilled: the lowest slot)", "mask": np.array([True, False]), "R": U_new[:, [0]], "gap": 0.3}
    fA = W10.sorted_slot_flips(A["mask"], prev_mask); fB = W10.sorted_slot_flips(B["mask"], prev_mask)
    kept_old = rule_v36d([A, B], prev_mask); kept_new = rule_v36e([A, B], R_prev)
    check(fA == 2 and fB == 0 and kept_old is B, "two-level counterexample: sorted-slot flips A %d, B %d: the v36d rule keeps %s (refilling)" % (fA, fB, kept_old["name"]))
    check(abs(A["D"]) < 1e-15 and abs(B["D"] - 1.0) < 1e-15 and kept_new is A,
          "gate (the two-level counterexample): subspace distance D_A %.3f, D_B %.3f: the classifier keeps %s" % (A["D"], B["D"], kept_new["name"]))
    # (2) the 8-dimensional overlap example
    th = math.radians(70.0); c, s = math.cos(th), math.sin(th)
    E8 = np.eye(8)
    U8 = np.zeros((8, 8))
    U8[:, 0] = E8[:, 0]; U8[:, 1] = E8[:, 1]
    U8[:, 2] = c * E8[:, 2] + s * E8[:, 4]; U8[:, 3] = c * E8[:, 3] + s * E8[:, 5]
    U8[:, 4] = -s * E8[:, 2] + c * E8[:, 4]; U8[:, 5] = -s * E8[:, 3] + c * E8[:, 5]
    U8[:, 6] = E8[:, 6]; U8[:, 7] = E8[:, 7]
    prev8 = np.array([True, True, True, True, False, False, False, False])
    R8_prev = E8[:, :4]
    mA = prev8.copy(); mB = np.array([True, True, False, False, True, True, False, False])
    A8 = {"name": "A (slots 1-4)", "mask": mA, "R": U8[:, mA], "gap": 0.2}
    B8 = {"name": "B (slots 1, 2, 5, 6)", "mask": mB, "R": U8[:, mB], "gap": 0.2}
    k_old = rule_v36d([A8, B8], prev8); k_new = rule_v36e([A8, B8], R8_prev)
    check(W10.sorted_slot_flips(mA, prev8) == 0 and W10.sorted_slot_flips(mB, prev8) == 4 and k_old is A8,
          "8-dimensional overlap example: sorted-slot flips A 0, B 4: the v36d rule keeps A")
    check(abs(A8["D"] - 2 * s * s) < 1e-12 and abs(B8["D"] - 2 * c * c) < 1e-12 and k_new is B8,
          "gate (the 8-dimensional overlap example): D_A %.4f = 2 sin^2(70), D_B %.4f = 2 cos^2(70): the classifier keeps B, whose occupied subspace is the previous record's up to 0.23 of an orbital" % (A8["D"], B8["D"]))
    try:
        for d in (os.environ.get("W10_ENGINE_DIR"),):
            if d and d not in sys.path:
                sys.path.insert(0, d)
        import torch
        import engine_ops_v3 as EO
        e = torch.arange(1, 9, dtype=torch.float64)
        ref, info, mask = EO.assign_max_overlap_v3(torch.as_tensor(R8_prev, dtype=torch.complex128), e, torch.as_tensor(U8, dtype=torch.complex128), 4)
        check(np.array_equal(np.asarray(mask, bool), mB), "engine_ops_v3.assign_max_overlap_v3 (the engine's own continuation) picks set B on the 8-dimensional example: mask %s" % np.asarray(mask, int))
    except ImportError:
        print("INFO engine_ops_v3/torch not importable (set W10_ENGINE_DIR): the engine's continuation not run on the example")
    # (3) the broadcast
    a21 = np.array([[True], [False]]); b2 = np.array([True, False])
    broadcast_count = int(np.sum(a21 != b2))
    check(a21.size == b2.size and broadcast_count == 2 and W10.sorted_slot_flips(a21, b2) is None,
          "the broadcast: shapes (2, 1) and (2,) pass the .size test and np.sum(a != b) counts %d over a (2, 2) comparison; sorted_slot_flips refuses (None)" % broadcast_count)
    check(W10.sorted_slot_flips(np.array([1, 0, 1], bool), np.array([1, 1, 1], bool)) == 1, "the shape-checked count on equal shapes: 1 flip")
    # (4) distinct sets
    meet = [{"held_sig": "a", "round": 1}, {"held_sig": "a", "round": 3}]
    check(len(meet) > 1 and len(W10.distinct_meeting_sets(meet)) == 1, "rounds 1 and 3 meeting on the same set: v36d's len(meet) > 1 labelled two sets; the distinct count is %d (no label)" % len(W10.distinct_meeting_sets(meet)))
    check(len(W10.distinct_meeting_sets([{"held_sig": "a"}, {"held_sig": "b"}])) == 2, "two rounds on different sets: 2 distinct sets (the label)")
    print("test_w10_defect6_setrule: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
