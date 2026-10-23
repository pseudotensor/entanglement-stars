#!/usr/bin/env python
"""w10_ball_head_check_v1.py (busi W10, reviewer A; W10_reading_v2.md section 1.5, RUN_W10_v3.md step C): the massless-ball head
pair's gate in the form that holds for any head.

Block D5.1 (defect 5) changes prepare_massless_ball_v33's two momentum-space head diagnostics by exactly half the signed weight of
the bins that sit on the thresholds: weight_ingoing_spinor_k_negative by (w_0 - w_-pi)/2 (the k = 0 bin, which the strict mask
dropped, and the k = -pi bin, which it counted) and weight_above_quarter_zone by (dropped - counted)/2 over the four bins at
|k| = pi/4 and 3 pi/4 (which of them the strict mask counted is this box's libm's rounding of |sin k| against math.sin(math.pi/4)).
RUN_W10_v2 step 4 gated the ball pair with test 5's INFO numbers for the chiral seeds at k0 0.1 on 2048 sites (8.4e-5 and 1.5e-3);
they do not transfer to a head at k0 0.0215 on 1024 or 512 sites (a bin's share scales as 1/N; the k = 0 bin's share grows when the
head's origin tail stops oscillating, 1/k0 > x0; the j = 5 seed's lobe edge (2j+1)/sigma + k0 sits at the quarter-zone threshold):
W10_reading_v2 section 1.2. The checks here:
  (1) --a A.json --b B.json (A the base engine's, B the W10 engine's, the same arguments): the head rows paired by j; every common
      key other than the two moved ones bitwise equal; B's _strict_v33 keys bitwise A's values; B's ingoing + outgoing by velocity
      1 to 1e-12; the signed changes printed per orbital with their reading (the sign names the heavier bin); a negative k-negative
      change needs the k = -pi bin, which weight_outgoing_second_valley_spinor_minus contains;
  (2) --slice FILE: the stored head orbitals (the T = 0 orbitals npz; --orbital-key names the array, default "dust"; a column's
      component layout, interleaved (2 n + a, the lib's and test 8's convention) or stacked, is chosen by matching B's strict values
      when --layout auto): the readouts recomputed, the threshold bins' shares printed (w_0, w_-pi, the four quarter-zone boundary
      bins with their ulps from the threshold and the strict mask's verdict), each change checked against half the threshold bins'
      weight and against the jsons' change to 1e-10;
  (3) --seeds --nsites N --k0 K0 [--x0 32 --sigma 16]: the same shares for the chiral seed family j = 0..5: at (2048, 0.1) test 5's
      INFO numbers, at (1024, 0.0215) and (512, 0.0215) the yardstick for the pathfinder's head (the seed family's k = 0 share is an
      upper estimate of the head's: the fixed point moves weight away from the origin).
Run: qi2 python w10_ball_head_check_v1.py --a collapse_v36d_ball.json --b collapse_v36i_ball.json [--slice orbitals_..._T0.npz]
     qi2 python w10_ball_head_check_v1.py --seeds --nsites 1024 --k0 0.0215
Exit 0 when every check passes, 1 otherwise."""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

MOVED = ("weight_ingoing_spinor_k_negative", "weight_above_quarter_zone")
STRICT = ("weight_ingoing_spinor_k_negative_strict_v33", "weight_above_quarter_zone_strict_v33")
NEW_B = STRICT + ("weight_ingoing_by_velocity", "weight_outgoing_by_velocity", "weight_outgoing_second_valley_spinor_minus", "convention_w10")
THR = math.sin(math.pi / 4)
FAIL = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)
    return cond


def find_rows(obj, acc):
    """the massless-ball head rows of a collapse json: the dicts that hold both "j" and the k-negative key, wherever they sit."""
    if isinstance(obj, dict):
        if "j" in obj and MOVED[0] in obj:
            acc.append(obj)
        else:
            for v in obj.values():
                find_rows(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_rows(v, acc)


def readouts(psi0, psi1):
    """prepare_massless_ball_v33's per-orbital momentum-space readouts with block D5.1 (the strict v33 values and the half-bin ones,
    the velocity-resolved ones of defect 9) and the shares of the bins on the thresholds: w_0 (k = 0), w_-pi (k = -pi), and the
    quarter-zone boundary bins (|sin k| within 1e-12 of sin(pi/4)) with their distance from the threshold in ulps and the strict
    mask's verdict on this box's libm."""
    psi0 = np.asarray(psi0, complex); psi1 = np.asarray(psi1, complex)
    N = len(psi0)
    k = 2.0 * math.pi * np.fft.fftfreq(N)
    c0 = np.fft.fft(psi0); c1 = np.fft.fft(psi1)
    inw = 0.5 * np.abs(c0 - c1) ** 2
    outw = 0.5 * np.abs(c0 + c1) ** 2
    spin_tot = np.abs(c0) ** 2 + np.abs(c1) ** 2
    tot = max(float(spin_tot.sum()), 1e-300)
    s = np.sin(k); a = np.abs(s)
    w_kneg = W10.half_bin_mask(s, 0.0, "below")
    w_quarter = W10.half_bin_mask(a, THR, "above")
    w_in_plus, w_in_minus = W10.ingoing_weights(k)
    w_out_plus, w_out_minus = W10.outgoing_weights(k)
    r = {"N": N,
         "kneg_strict": float(inw[s < 0].sum()) / tot, "kneg_half": float((inw * w_kneg).sum()) / tot,
         "quarter_strict": float(spin_tot[a > THR].sum()) / tot, "quarter_half": float((spin_tot * w_quarter).sum()) / tot,
         "ingoing_by_velocity": float((outw * w_in_plus + inw * w_in_minus).sum()) / tot,
         "outgoing_by_velocity": float((outw * w_out_plus + inw * w_out_minus).sum()) / tot,
         "second_valley_minus": float((inw * w_out_minus).sum()) / tot,
         "w_0": float(inw[0]) / tot,
         "w_-pi": (float(inw[N // 2]) / tot if N % 2 == 0 else 0.0)}
    onb = np.where(np.abs(a - THR) <= 1e-12)[0]
    r["boundary"] = [{"n": (int(n) if n < N // 2 else int(n) - N), "k": float(k[n]), "share": float(spin_tot[n]) / tot,
                      "counted": bool(a[n] > THR), "ulps": int(round((float(a[n]) - THR) / np.spacing(THR)))} for n in onb]
    return r


def deltas(r):
    return r["kneg_half"] - r["kneg_strict"], r["quarter_half"] - r["quarter_strict"]


def describe(r, label):
    d_in, d_q = deltas(r)
    Wb = sum(b["share"] for b in r["boundary"])
    bins = ", ".join("n %+d share %.3e %s (%+d ulp)" % (b["n"], b["share"], "counted" if b["counted"] else "dropped", b["ulps"]) for b in r["boundary"])
    print("  %s: w_0 %.4e, w_-pi %.4e -> k-negative change %+.4e (half the heavier bin: %.4e); boundary bins %s (total %.4e) -> quarter-zone change %+.4e (half the total: %.4e); by velocity ingoing %.7f outgoing %.7f, second valley on (1, -1) %.4e" % (
        label, r["w_0"], r["w_-pi"], d_in, 0.5 * max(r["w_0"], r["w_-pi"]), bins, Wb, d_q, 0.5 * Wb,
        r["ingoing_by_velocity"], r["outgoing_by_velocity"], r["second_valley_minus"]), flush=True)
    return d_in, d_q, Wb


def sign_reading(d_in, d_q):
    if d_in >= 0.0:
        s_in = "the k = 0 bin outweighs the k = -pi bin by %.4e" % (2.0 * d_in)
    else:
        s_in = "the k = -pi bin (the second valley's zone boundary) outweighs the k = 0 bin by %.4e" % (-2.0 * d_in)
    if d_q >= 0.0:
        s_q = "the pair the strict mask dropped outweighs the pair it counted by %.4e" % (2.0 * d_q)
    else:
        s_q = "the pair the strict mask counted outweighs the pair it dropped by %.4e" % (-2.0 * d_q)
    return s_in, s_q


def compare_jsons(pa, pb):
    """the head rows of two collapse jsons (A the base engine's, B the W10 engine's), paired by j; returns {j: (rowA, rowB, d_in, d_q)}."""
    A = []; B = []
    find_rows(json.load(open(pa)), A); find_rows(json.load(open(pb)), B)
    check(len(A) > 0 and len(A) == len(B), "head rows found: %d in A (%s), %d in B (%s)" % (len(A), pa, len(B), pb))
    Bj = {r["j"]: r for r in B}
    pairs = {}
    for ra in A:
        j = ra["j"]
        rb = Bj.get(j)
        if rb is None:
            check(False, "orbital %s: no row in B" % j)
            continue
        common = [key for key in ra if key in rb and key not in MOVED]
        diff = [key for key in common if ra[key] != rb[key]]
        check(not diff, "orbital %s: the %d common keys other than the two moved ones are bitwise equal%s" % (
            j, len(common), "" if not diff else "; differing: %s" % diff))
        for m, s_ in zip(MOVED, STRICT):
            check(s_ in rb and rb[s_] == ra[m], "orbital %s: B's %s equals A's %s bitwise (%r)" % (j, s_, m, ra[m]))
        missing = [key for key in NEW_B if key not in rb]
        check(not missing, "orbital %s: B carries the six new keys of D5.1%s" % (j, "" if not missing else "; missing %s" % missing))
        d_in = float(rb[MOVED[0]]) - float(ra[MOVED[0]]); d_q = float(rb[MOVED[1]]) - float(ra[MOVED[1]])
        v_in = float(rb.get("weight_ingoing_by_velocity", float("nan"))); v_out = float(rb.get("weight_outgoing_by_velocity", float("nan")))
        check(abs(v_in + v_out - 1.0) <= 1e-12, "orbital %s: ingoing + outgoing by velocity = %.15f (1 to 1e-12)" % (j, v_in + v_out))
        sv = float(rb.get("weight_outgoing_second_valley_spinor_minus", 0.0))
        check(d_in >= -0.5 * sv - 1e-15, "orbital %s: the k-negative change %+.4e is at least -(1/2) weight_outgoing_second_valley_spinor_minus = %+.4e (a negative change needs the k = -pi bin, which that key contains)" % (j, d_in, -0.5 * sv))
        s_in, s_q = sign_reading(d_in, d_q)
        print("  orbital %s: k-negative strict %.7f half-bin %.7f (change %+.4e: %s); quarter zone strict %.7f half-bin %.7f (change %+.4e: %s); by velocity ingoing %.7f outgoing %.7f, second valley on (1, -1) %.4e" % (
            j, float(ra[MOVED[0]]), float(rb[MOVED[0]]), d_in, s_in, float(ra[MOVED[1]]), float(rb[MOVED[1]]), d_q, s_q, v_in, v_out, sv), flush=True)
        pairs[j] = (ra, rb, d_in, d_q)
    return pairs


def split(col, layout):
    N = col.shape[0] // 2
    if layout == "interleaved":
        return col[0::2], col[1::2]
    return col[:N], col[N:]


def slice_check(path, key_prefix, layout, pairs):
    Z = np.load(path, allow_pickle=False)
    names = sorted(nm for nm in Z.files if nm.startswith(key_prefix))
    if not names:
        check(False, "--slice %s: no array named %s*; the keys are %s (name the head orbitals' array with --orbital-key)" % (path, key_prefix, sorted(Z.files)))
        return
    print("INFO quarter-zone threshold math.sin(math.pi / 4) = %.17g; the boundary bins are the n with |sin k| within 1e-12 of it, their ulps from it decide the strict mask's verdict" % THR)
    for nm in names:
        arr = np.asarray(Z[nm])
        if arr.ndim != 2:
            print("INFO %s: shape %s is not a column array; skipped" % (nm, arr.shape)); continue
        if arr.shape[0] % 2 != 0 or (arr.shape[1] % 2 == 0 and arr.shape[0] < arr.shape[1]):
            arr = arr.T                                                   # the orbitals were the rows
        if arr.shape[0] % 2 != 0:
            print("INFO %s: shape %s has no even site dimension; skipped" % (nm, arr.shape)); continue
        n2, ncol = arr.shape
        print("INFO %s: %d columns on %d sites" % (nm, ncol, n2 // 2))
        layouts = [layout] if layout != "auto" else ["interleaved", "stacked"]
        chosen = None
        for lay in layouts:
            rs = [readouts(*split(arr[:, j], lay)) for j in range(ncol)]
            if pairs:
                js = [j for j in range(ncol) if j in pairs]
                ok = bool(js) and all(abs(rs[j]["kneg_strict"] - float(pairs[j][1][STRICT[0]])) <= 1e-9
                                      and abs(rs[j]["quarter_strict"] - float(pairs[j][1][STRICT[1]])) <= 1e-9 for j in js)
                detail = ""
                if not ok and js:
                    detail = " (column %d: k-negative strict %.7f against B's %.7f)" % (js[0], rs[js[0]]["kneg_strict"], float(pairs[js[0]][1][STRICT[0]]))
                print("INFO layout %s: %s B's strict values%s" % (lay, "matches" if ok else "does not match", detail))
                if ok:
                    chosen = (lay, rs); break
            else:
                chosen = (lay, rs); break
        if chosen is None:
            check(False, "%s: no layout reproduces B's strict values (the array is not the head, or its component layout is neither interleaved nor stacked)" % nm)
            continue
        lay, rs = chosen
        for j, r in enumerate(rs):
            d_in, d_q, Wb = describe(r, "%s column %d (%s)" % (nm, j, lay))
            check(abs(d_in) <= 0.5 * max(r["w_0"], r["w_-pi"]) + 1e-15 and abs(d_q) <= 0.5 * Wb + 1e-15,
                  "%s column %d: both changes are within half the threshold bins' weight" % (nm, j))
            if j in pairs:
                check(abs(d_in - pairs[j][2]) <= 1e-10 and abs(d_q - pairs[j][3]) <= 1e-10,
                      "%s column %d: the recomputed changes equal the jsons' (%+.4e vs %+.4e; %+.4e vs %+.4e)" % (nm, j, d_in, pairs[j][2], d_q, pairs[j][3]))


def seed(N, x0, sigma, k0, j):
    """the S11 chiral seed j on the (1, -1)/sqrt 2 spinor (the formula of test_w10_defect5_halfbin: sqrt(sigma/pi) (sigma - i x)^j /
    (sigma + i x)^(j+1) e^(-i k0 x), x = R - x0, R = 1..N)."""
    xx = np.arange(N) + 1.0 - x0
    psi = math.sqrt(sigma / math.pi) * (sigma - 1j * xx) ** j / (sigma + 1j * xx) ** (j + 1) * np.exp(-1j * k0 * xx)
    return psi / math.sqrt(2.0), -psi / math.sqrt(2.0)


def seeds_report(N, x0, sigma, k0):
    print("INFO the chiral seed family j = 0..5 on %d sites (x0 %g, sigma %g, k0 %g): the threshold bins' shares and the changes D5.1 makes; the centre wavenumber (2j+1)/sigma + k0 against the quarter-zone threshold pi/4 = %.4f" % (N, x0, sigma, k0, math.pi / 4))
    mx_in = 0.0; mx_q = 0.0
    for j in range(6):
        r = readouts(*seed(N, x0, sigma, k0, j))
        d_in, d_q, Wb = describe(r, "seed j %d (centre wavenumber %.4f)" % (j, (2 * j + 1) / sigma + k0))
        mx_in = max(mx_in, abs(d_in)); mx_q = max(mx_q, abs(d_q))
    print("INFO seed family on %d sites at k0 %g: largest |change| %.3e (k-negative) and %.3e (quarter zone); at (2048, 0.1) these are test 5's INFO numbers 8.39e-05 and 1.53e-03; a bin's share scales as 1/N" % (N, k0, mx_in, mx_q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=None, help="the base engine's collapse json (v36d or v36f) of the ball run")
    ap.add_argument("--b", default=None, help="the W10 engine's collapse json (v36i or v36h) of the same run")
    ap.add_argument("--slice", default=None, help="the T = 0 orbitals npz of the W10 engine's ball run (the head is the dust at T = 0)")
    ap.add_argument("--orbital-key", default="dust", help="the name (prefix) of the head orbitals' array in the npz")
    ap.add_argument("--layout", default="auto", choices=["auto", "interleaved", "stacked"], help="a column's component layout: 2 n + a (interleaved) or the two components stacked")
    ap.add_argument("--seeds", action="store_true", help="the chiral seed family's shares at --nsites, --k0, --x0, --sigma")
    ap.add_argument("--nsites", type=int, default=1024)
    ap.add_argument("--k0", type=float, default=0.0215)
    ap.add_argument("--x0", type=float, default=32.0)
    ap.add_argument("--sigma", type=float, default=16.0)
    a = ap.parse_args()
    if not (a.a or a.b or a.slice or a.seeds):
        ap.print_help()
        sys.exit(2)
    pairs = {}
    if a.a and a.b:
        pairs = compare_jsons(a.a, a.b)
    elif a.a or a.b:
        check(False, "--a and --b go together")
    if a.slice:
        slice_check(a.slice, a.orbital_key, a.layout, pairs)
    if a.seeds:
        seeds_report(a.nsites, a.x0, a.sigma, a.k0)
    print("w10_ball_head_check: %s" % ("ALL PASS" if not FAIL else "%d FAIL" % len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
