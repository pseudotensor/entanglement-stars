#!/bin/bash
# launch_ladder_v36d_v1c.sh: version 1b with LADDERS_spec_v1_rulings.md applied (the builder): ruling 6, the 2 r_s releases' half-r_s slices 3 to 10 (SLX);
# ruling 3, M3d_s4's extra clock 28 (SLX) and the policy read per member at each start or resume from $ROOT/policy_<member> when that file exists,
# else SLICE_POLICY, recorded in the status file; rulings 1, 2, 4 and 5 change no argument. Also: a card's queue from $ROOT/queue_<card> when that
# file exists (inbox #172: pod 6, idle, takes R3's first members while R3 runs the tau 2 rerun; R3's file then lists the rest). Version 1b follows.
# launch_ladder_v36d_v1b.sh: version 1 with one syntax fix in mode_queue (the builder, a mechanical fix: the queue list is read into Q before the loop); nothing else changed. Version 1 follows.
# launch_ladder_v36d_v1.sh (reviewer A for the solve lane, applied by the busi lane's analyst builder; its spec is solve/specs/LADDERS_spec_v1.md):
# one launcher for every member of MASS_ladder_v1 (#162 item 3) and SCALING_ladder_v1 (#162 item 4) on the rented H100 cards R1, R2, R3 (#163),
# on engine_collapse_v36d unchanged for every member but M3d_s4, which runs on engine_collapse_v36e (build_engine_collapse_v36e.py from v36d;
# --first-return-collar-over-rs 8 gives the plan's "to T/r_s 30 or the chain end's first return into the collar, the first-return estimate
# printed"). The rungs' launchers (launch_v36d_from0_rungs_pod2_v2.sh, launch_v36d_from0_rung2_pod3_v1.sh) are its template: the production's
# base arguments, nice 10, 8 threads, expandable segments, a detached setsid nohup run with pid and status files, the disk guard by SIGSTOP/SIGCONT.
# Usage: bash launch_ladder_v36d_v1.sh MEMBER MODE
#   MEMBER  M3a_s4 M3a_s2 M3a_s4_tau4 M3b_s4 M3b_s8 M3c_s4 M3d_s4 S4_16_s4 S4_16_s2 S4_16_s4_tau1 S4_128_s4 S4_128_s8 S4_128_s16 S4_R2_m0p2 S4_R2_m0p1
#   MODE    check    the preflight (engine md5, environment, CUDA, free device memory and disk against the member's derived needs); prints the command
#           t0       the preparation and the T = 0 record alone (--benchmark) into t0check/<tag>; the T = 0 check into the status file
#           start    the detached fresh run; refused when its json exists or its pid is alive; the guard and the watcher (inline T = 0 check,
#                    progress every 30 min, hours and cost at the end)
#           resume   from the newest usable complex128 slice: the 2-record gate against the stored records (the launcher's comparison, and
#                    compare_member_floor_v2 --criterion on each of the 2 records when present), the stored files backed up, the detached continuation
#           status   the member's state
#           gate36e  3 records (T/rs 0, 0.125, 0.25) of S4_16_s4 on v36d and on v36e with its option off, compared bitwise (the v36e gate; M3d_s4's precondition)
#   MEMBER = R1 | R2 | R3 with MODE queue: the card's members in the plan's order, each started (or resumed once) when the card is idle; writes
#           "<card> queue empty" into STATUS when done (#163). Run it detached:
#           setsid nohup bash launch_ladder_v36d_v1.sh R3 queue > ${QI2_ROOT}/queue_R3.out 2>&1 < /dev/null &
# Environment (defaults): LADDER_ROOT=${QI2_ROOT}/ladders (runs and states on /workspace, #163)  CODE_V36D=${QI2_ROOT}/code_v36d
#   PY_V36D=${QI2_ROOT}/python  GPU=0  SLICE_POLICY=full|lean|bounce (full: the spec's slice rule)  RATE_USD_PER_H (the card's price;
#   the cost lines say "rate unset" otherwise)  FORCE_CHECK=1 FORCE_RESUME=1 (overrides, recorded)  CMP_MEMBER_FLOOR (compare_member_floor_v2.py's path,
#   default $CODE_V36D/compare_member_floor_v2.py)  CMP_TEMPLATE (its call with the placeholders {PY} {CMP} {GATE} {STORED} {T}; the default is a guess at
#   its interface, to be set from its --help)
set -u
SELF=$(readlink -f "$0")
MEMBER=${1:?usage: launch_ladder_v36d_v1.sh MEMBER MODE}; MODE=${2:?usage: launch_ladder_v36d_v1.sh MEMBER MODE}
ROOT=${LADDER_ROOT:-${QI2_ROOT}/ladders}
CODE=${CODE_V36D:-${QI2_ROOT}/code_v36d}
PY=${PY_V36D:-${QI2_ROOT}/python}
GPU=${GPU:-0}
POLICY=${SLICE_POLICY:-full}
RATE=${RATE_USD_PER_H:-}
HELPER=$ROOT/ladder_helper_v1.py
STF=$ROOT/STATUS
mkdir -p $ROOT/runs $ROOT/states $ROOT/logs $ROOT/t0check $ROOT/gates $ROOT/pids $ROOT/failed || exit 1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

write_helper() { cat > $HELPER <<'PYEOF'
"""ladder_helper_v1.py: the python side of launch_ladder_v36d_v1.sh, rewritten at each call (numpy, json and engine_collapse_v36d's file formats only)."""
import glob, hashlib, json, math, os, platform, re, sys
import numpy as np


def nsteps_engine(tend, rs, tau):
    return int(round(float(tend) * float(rs) / float(tau)))      # engine_collapse_v36d.run: nsteps = int(round(a.T_over_rs * rs / a.tau))


def slice_lists(rs, tau, tend, policy, extra):
    """the slice clocks: every target x (0, 1 to 10, 11 to 16 by halves, 17 to 40, plus the member's extra halves) mapped to the record nearest it
    (k = floor(x rs/tau + 1/2), ties to the later record, capped at the run's last record) and listed as k tau/rs; full: all complex128; lean:
    complex128 at 0, the halves 11 to 16, 20, 24, 30, 35, 40 and the extras, complex64 elsewhere; bounce: the lean complex128 set alone."""
    rs, tau, tend = float(rs), float(tau), float(tend)
    nst = nsteps_engine(tend, rs, tau)
    ex = [float(x) for x in extra.split(",") if x.strip()] if extra and extra != "none" else []
    base = [0.0] + [float(x) for x in range(1, 11)] + [11.0 + 0.5 * i for i in range(11)] + [float(x) for x in range(17, 41)]
    lean = set([0.0, 20.0, 24.0, 30.0, 35.0, 40.0] + [11.0 + 0.5 * i for i in range(11)] + ex)
    c128, c64, seen = [], [], set()
    for x in sorted(set(base + ex)):
        if x > tend + 1e-9:
            continue
        k = min(int(math.floor(x * rs / tau + 0.5)), nst)
        if k in seen:
            continue
        seen.add(k)
        v = k * tau / rs
        if policy == "full" or x in lean:
            c128.append(v)
        elif policy == "lean":
            c64.append(v)
    return c128, c64


def cmd_slices(rs, tau, tend, policy, extra):
    c128, c64 = slice_lists(rs, tau, tend, policy, extra)
    print("C128=%s" % ",".join("%.10f" % v for v in c128))
    print("C64=%s" % ",".join("%.10f" % v for v in c64))


def cmd_estimate(n, K, rs, tau, tend, policy, extra):
    """derived (LADDERS_spec_v1 section 1): device memory n^2 (256 K + 150) bytes; a complex128 slice 64 n^2 K bytes plus 2 percent; the base pace 24 s per
    record at 2048 sites and 10 channels scaled by K/10 and by (n/2048)^2 (lower) to (n/2048)^3 (upper), floor 5 s; the guard at 2 and 4 slices' room."""
    n, K = int(n), int(K)
    c128, c64 = slice_lists(rs, tau, tend, policy, extra)
    mem_mib = n * n * (256 * K + 150) / 2.0 ** 20
    sg = 64.0 * n * n * K * 1.02 / 1e9
    recs = nsteps_engine(tend, rs, tau) + 1
    base = 24.0 * (K / 10.0)
    t_lo = max(5.0, base * (n / 2048.0) ** 2); t_hi = max(5.0, base * (n / 2048.0) ** 3)
    total = len(c128) * sg + len(c64) * sg / 2.0 + 2.0
    for k_, v_ in (("MEM_MIB", "%d" % math.ceil(mem_mib)), ("SLICE_GB", "%.2f" % sg), ("SLICE64_GB", "%.2f" % (sg / 2.0)), ("N_C128", "%d" % len(c128)),
                   ("N_C64", "%d" % len(c64)), ("TOTAL_GB", "%.0f" % total), ("RECORDS", "%d" % recs), ("HOURS_LO", "%.1f" % (recs * t_lo / 3600.0)),
                   ("HOURS_HI", "%.1f" % (recs * t_hi / 3600.0)), ("PAUSE_GB", "%d" % max(2, math.ceil(2.0 * sg))), ("RESUME_GB", "%d" % max(4, math.ceil(4.0 * sg)))):
        print("%s=%s" % (k_, v_))


def cmd_clocks(rs, nsites, R0, coll, rsh=None):
    """the vacuum Painleve-Gullstrand estimates in r_s (derived; LADDERS_spec_v1 section 5): N = 1, |v| = sqrt(r_s/r); the outgoing null ray's clock
    from r_1 to r_2 is r_s [u^2 + 2u + 2 ln(u - 1)], the ingoing one's r_s [u^2 - 2u + 2 ln(u + 1)], u = sqrt(r/r_s)."""
    rs, R0, coll = float(rs), float(R0), float(coll); xe = float(nsites) / rs
    F = lambda u: u * u + 2.0 * u + 2.0 * math.log(u - 1.0)
    G = lambda u: u * u - 2.0 * u + 2.0 * math.log(u + 1.0)
    Tff = 0.5 * math.pi * R0 ** 1.5
    xl = (float(rsh) / rs) if (rsh is not None and float(rsh) / rs > 1.05) else R0
    t_end1 = F(math.sqrt(xe)) - F(math.sqrt(xl)); t_back = G(math.sqrt(xe)) - G(math.sqrt(coll))
    t_end2 = Tff + F(math.sqrt(xe)) - F(math.sqrt(1.9))
    print("vacuum estimates in r_s (derived): free-fall clock from rest at %g r_s = (pi/2)(R0/r_s)^1.5 = %.2f (the plan's 12.6 at 4 r_s); release front from %.2f r_s at T = 0: "
          "at the chain end (%g r_s) %.1f, back into %g r_s at %.1f; formation ray from 1.9 r_s at the free-fall clock (the launch that puts the production's formation light at "
          "16 r_s at 35.6): at the chain end %.1f, back into %g r_s at %.1f" % (R0, Tff, xl, xe, t_end1, coll, t_end1 + t_back, t_end2, coll, t_end2 + t_back))


def cmd_t0(jpath, spath, m, g, rs, K):
    """the T = 0 check (LADDERS_spec_v1 section 5) on the json's head and first record and on the T = 0 slice; exit 0 PASS, 1 FAIL."""
    m, g, rs, K = float(m), float(g), float(rs), int(K)
    J = json.load(open(jpath)); h = J["head"]
    recs = [r for r in J["records"] if r.get("E") is not None]
    if not recs:
        print("FAIL: no accepted T = 0 record in %s" % jpath); return 1
    r0 = recs[0]; fails, notes = [], []
    NBx = rs / (2.0 * g * m)
    NB = float(h.get("N_B") if h.get("N_B") is not None else float("nan")); E0 = float(h.get("E0") if h.get("E0") is not None else float("nan"))
    pj = h.get("projection") or {}
    print("preparation: N_B %.6f against r_s/(2 g m) = %.6f (difference %.2e); E0 %.10f; E0/(N_B m) %.6f; E0/m %.2f; passes %s; last change %s; f_kin %s; dust momentum total %s; Gram cond %s; stationarity %s; prep ell %s, evolution ell %s" % (
        NB, NBx, NB - NBx, E0, E0 / (NB * m) if NB > 0 else float("nan"), E0 / m, h.get("passes"), h.get("last_change"), h.get("f_kin"), h.get("dust_momentum_total"),
        pj.get("gram_cond_max"), pj.get("stationarity_max"), h.get("prep_ell"), h.get("evolution_ell")))
    if not math.isfinite(E0):
        fails.append("E0 not finite")
    if not (abs(NB - NBx) <= 1e-6 * NBx):
        fails.append("N_B %.6f differs from r_s/(2 g m) %.6f (the 400-level cap or the filling)" % (NB, NBx))
    kid = h.get("kernel_identities")
    if kid:
        print("kernel identities: K0 on 1: max |. - 1| %.3e; K1 on c r: max abs %.3e (relative %.3e); K0 weight sums %.15f to %.15f" % (
            kid["K0_constant_max_abs"], kid["K1_linear_max_abs"], kid["K1_linear_max_rel"], kid["K0_weight_sums"][0], kid["K0_weight_sums"][1]))
        if not all(math.isfinite(float(x)) for x in (kid["K0_constant_max_abs"], kid["K1_linear_max_abs"])):
            fails.append("kernel identities not finite")
    else:
        notes.append("no kernel identities in the head (site fields not averaged?)")
    sol = r0.get("solver") or {}; ind = r0.get("indicator") or {}
    rsh = r0.get("R_sh")
    print("T = 0 record: E %.10f (E/E0 %.10f); min(N - |v|) %s at r %s; zeros of N - |v| %s; R_sh %s a (%s r_s); R_c %s a; r_collar %s; residuals W %s 1/N %s; collar %s; B_estimate %s; continuation %s; mask = negative set %s" % (
        r0["E"], r0.get("E_over_E0", float("nan")), r0.get("min_N_minus_absv"), r0.get("r_min_N_minus_absv"), r0.get("trapping_zeros"), rsh, ("%.3f" % (rsh / rs)) if rsh else "-",
        r0.get("R_c"), r0.get("r_collar"), sol.get("residual_W"), sol.get("residual_invN"), ind.get("collar_label"), ind.get("B_estimate"), r0.get("continuation_label"), r0.get("mask_equals_negative_set")))
    mn = r0.get("min_N_minus_absv")
    if mn is None or not (float(mn) > 0.0):
        fails.append("min(N - |v|) not positive at T = 0 (a root at the release)")
    if r0.get("trapping_zeros"):
        fails.append("zeros of N - |v| at T = 0: %s" % r0.get("trapping_zeros"))
    if spath and os.path.exists(spath):
        try:
            Z = np.load(spath)
            kap = [int(k) for k in Z["kappas"]]; w = [float(x) for x in Z["weights"]]
            rows, tot, top, fmax = [], 0.0, 0, 0
            for k_, w_ in zip(kap, w):
                f = np.asarray(Z["f_%d" % k_], float)
                nd = int(Z["dust_%d" % k_].shape[1]) if ("dust_%d" % k_) in Z.files else len(f)
                cnt = w_ * float(f.sum()); tot += cnt
                if nd:
                    top = max(top, k_)
                fmax = max(fmax, len(f))
                rows.append("kappa %d (w %g): %d columns, %.3f particles, lowest occupation %s" % (k_, w_, nd, cnt, ("%.4f" % f.min()) if len(f) else "-"))
            rule = (NB / 8.0) ** (1.0 / 3.0)
            print("dust per channel at T = 0: %s; weighted total %.6f; highest channel holding dust %d against the plan's (N_B/8)^(1/3) rule %.2f (%d) and the set's top %d; the largest per-channel level count %d against the engine's cap 400" % (
                "; ".join(rows), tot, top, rule, int(math.ceil(rule)), K, fmax))
            if fmax >= 400:
                fails.append("the preparation's 400-level cap per channel binds")
            if top == K:
                notes.append("the set's top channel %d holds dust; whether channel %d would hold dust is not tested" % (K, K + 1))
            if abs(tot - NB) > 1e-6 * max(NB, 1.0):
                notes.append("the slice's weighted dust count differs from N_B by %.3e" % (tot - NB))
        except Exception as ex:
            notes.append("the T = 0 slice could not be read (%s: %s)" % (type(ex).__name__, str(ex)[:120]))
    else:
        notes.append("no T = 0 slice read")
    for n_ in notes:
        print("note: %s" % n_)
    if fails:
        print("FAIL: %s" % "; ".join(fails)); return 1
    print("PASS"); return 0


def cmd_newest(sdir, jpath, fpath):
    """the newest usable complex128 slice: dtype complex128, mtime older than 120 s, an accepted record at its clock in the json, a row at it in the fields npz."""
    fs = glob.glob(os.path.join(sdir, "orbitals_*_T*.npz"))
    def tkey(f):
        mm = re.search(r"_T([0-9]+\.[0-9]+)\.npz$", f); return float(mm.group(1)) if mm else -1.0
    try:
        J = json.load(open(jpath)); rs = float(J["args"]["rs"])
        Ts = [float(r["T"]) for r in J["records"] if r.get("E") is not None]
        Tf = np.asarray(np.load(fpath, allow_pickle=True)["T"], float)
    except Exception:
        return 1
    now = os.path.getmtime
    import time
    for f in sorted(fs, key=tkey, reverse=True):
        try:
            if time.time() - now(f) < 120:
                continue
            Z = np.load(f)
            if str(Z["dtype"]) != "complex128":
                continue
            T = float(Z["T"])
            if not any(abs(t - T) < 1e-9 for t in Ts) or not np.any(np.abs(Tf - T) < 1e-9):
                continue
            print("%s %.10f" % (f, T / rs)); return 0
        except Exception:
            continue
    return 1


def cmd_after(jpath, ts):
    J = json.load(open(jpath)); rs = float(J["args"]["rs"])
    aft = sorted(r["T"] / rs for r in J["records"] if r.get("E") is not None and r["T"] / rs > float(ts) + 1e-6)
    print("N_AFTER=%d" % len(aft)); print("T1=%s" % (("%.10f" % aft[0]) if aft else "")); print("T2=%s" % (("%.10f" % aft[1]) if len(aft) > 1 else ""))


def cmd_finished(jpath, tend, tau, rs):
    if not os.path.exists(jpath):
        return 2
    J = json.load(open(jpath))
    for k in ("end_v36e", "end_v32", "end_v34", "end_s11"):
        if k in J:
            print("finished: ended by rule %s at T/rs %s" % (k, (J[k] or {}).get("T_over_rs"))); return 0
    recs = [r for r in J["records"] if r.get("E") is not None]
    if not recs:
        return 1
    last = recs[-1]["T"] / float(rs)
    tlast = nsteps_engine(tend, rs, tau) * float(tau) / float(rs)
    if last >= tlast - 1e-6:
        print("finished: last record T/rs %.6f (the run's last clock %.6f)" % (last, tlast)); return 0
    print("not finished: last record T/rs %.6f of %.6f" % (last, tlast)); return 1


def cmd_last(jpath):
    J = json.load(open(jpath)); rs = float(J["args"]["rs"])
    recs = [r for r in J["records"] if r.get("E") is not None]
    if not recs:
        print("no accepted record yet"); return 1
    print("T_over_rs=%.6f records=%d E_over_E0=%.7f" % (recs[-1]["T"] / rs, len(recs), recs[-1].get("E_over_E0", float("nan")))); return 0


def cmd_gatecmp(sdir, stag, gdir, gtag, t1, t2):
    """the 2-record gate: the gate run's records at t1, t2 against the stored run's (LADDERS_spec_v1 section 7's tolerances)."""
    JS = json.load(open(os.path.join(sdir, "collapse_%s.json" % stag))); JG = json.load(open(os.path.join(gdir, "collapse_%s.json" % gtag)))
    rs = float(JS["args"]["rs"])
    ZS = np.load(os.path.join(sdir, "collapse_%s_fields.npz" % stag), allow_pickle=True); ZG = np.load(os.path.join(gdir, "collapse_%s_fields.npz" % gtag), allow_pickle=True)
    SS = np.load(os.path.join(sdir, "collapse_%s_spectra.npz" % stag), allow_pickle=True); SG = np.load(os.path.join(gdir, "collapse_%s_spectra.npz" % gtag), allow_pickle=True)
    ok = True
    def rec_at(J, t):
        c = [r for r in J["records"] if r.get("E") is not None and abs(r["T"] / rs - t) < 1e-7]; return c[0] if c else None
    def row_at(Z, t):
        T = np.asarray(Z["T"], float); i = np.where(np.abs(T / rs - t) < 1e-7)[0]; return int(i[0]) if len(i) else None
    for t in (float(t1), float(t2)):
        a_, b_ = rec_at(JS, t), rec_at(JG, t)
        if a_ is None or b_ is None:
            print("T/rs %.4f: record missing (stored %s, gate %s)" % (t, a_ is not None, b_ is not None)); ok = False; continue
        E = abs(float(a_["E"]))
        for key, tol, scale in (("E", 1e-9, E), ("E_dust_shells", 1e-8, E), ("E_pairs_shells", 1e-8, E), ("H_in", 1e-8, E), ("min_N_minus_absv", 1e-8, 1.0), ("R_t", 1e-6, 1.0)):
            x, y = a_.get(key), b_.get(key)
            d = 0.0 if (x is None and y is None) else (float("inf") if (x is None or y is None) else abs(float(x) - float(y)))
            good = d <= tol * scale; ok = ok and good
            print("T/rs %.4f %s: stored %s gate %s |diff| %.3e (tolerance %.0e x %s) %s" % (t, key, x, y, d, tol, "E" if scale != 1.0 else "1", "ok" if good else "DIFFERS"))
        ia, ib = row_at(ZS, t), row_at(ZG, t)
        if ia is None or ib is None:
            print("T/rs %.4f: fields row missing (stored %s, gate %s)" % (t, ia is not None, ib is not None)); ok = False
        else:
            for key in ("N", "v", "N_applied", "v_applied", "eps_tot", "J_tot"):
                d = float(np.max(np.abs(np.asarray(ZS[key][ia], float) - np.asarray(ZG[key][ib], float))))
                good = d <= 1e-8; ok = ok and good
                print("T/rs %.4f fields %s: max |diff| %.3e (tolerance 1e-08) %s" % (t, key, d, "ok" if good else "DIFFERS"))
        ja = np.where(np.abs(np.asarray(SS["T"], float) / rs - t) < 1e-7)[0]; jb = np.where(np.abs(np.asarray(SG["T"], float) / rs - t) < 1e-7)[0]
        if len(ja) and len(jb):
            neq = [k for k in SS.files if k.startswith("mask_") and not np.array_equal(SS[k][ja[0]], SG[k][jb[0]])]
            good = not neq; ok = ok and good
            print("T/rs %.4f masks: %s" % (t, "equal in every channel" if good else "DIFFER in %s" % neq))
        else:
            print("T/rs %.4f masks: spectra row missing" % t); ok = False
    print("GATE PASS" if ok else "GATE FAIL"); return 0 if ok else 1


def cmd_env():
    print("python %s" % platform.python_version())
    for mod in ("numpy", "scipy", "torch"):
        try:
            m_ = __import__(mod); print("%s %s" % (mod, m_.__version__))
        except Exception as ex:
            print("%s not importable (%s)" % (mod, type(ex).__name__))
    try:
        import torch
        print("cuda available %s; device %s; torch cuda %s" % (torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", torch.version.cuda))
    except Exception as ex:
        print("torch cuda check failed: %s" % type(ex).__name__)


def cmd_chainmd5(d):
    """the code tree's md5: every .py under the tree, sorted by relative path, concatenated."""
    paths = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(".py"):
                paths.append(os.path.relpath(os.path.join(root, f), d))
    h = hashlib.md5()
    for p in sorted(paths):
        h.update(open(os.path.join(d, p), "rb").read())
    print(h.hexdigest())


if __name__ == "__main__":
    c = sys.argv[1]; args = sys.argv[2:]
    fn = {"slices": cmd_slices, "estimate": cmd_estimate, "clocks": cmd_clocks, "t0": cmd_t0, "newest": cmd_newest, "after": cmd_after, "finished": cmd_finished,
          "last": cmd_last, "gatecmp": cmd_gatecmp, "env": cmd_env, "chainmd5": cmd_chainmd5}[c]
    rc = fn(*args)
    sys.exit(rc or 0)
PYEOF
}
write_helper

now() { date -u +%FT%TZ; }
st() { echo "$(now) $*" | tee -a "$STF"; }
free_gb() { df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc 0-9; }
alive() { local p; p=$(cat "$PIDF" 2>/dev/null); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
gpu_free_mib() { nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9; }
gpu_busy() { [ -n "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

member() {   # the member table (LADDERS_spec_v1 section 2); sets the globals the modes use
  M=0.2; RS=64; N=2048; R0=4; K=10; TAU=8; ELL=4; PREP=4; TEND=40; COLL=16; ENGINE=v36d; EXTRA=""; SLX=""
  case $1 in
    M3a_s4)        M=0.1;  CARD=R3; ORDER=1;;
    M3a_s2)        M=0.1;  ELL=2; CARD=R3; ORDER=2;;
    M3a_s4_tau4)   M=0.1;  TAU=4; TEND=16; CARD=R3; ORDER=3;;
    M3b_s4)        M=0.4;  CARD=R3; ORDER=4;;
    M3b_s8)        M=0.4;  ELL=8; CARD=R3; ORDER=5;;
    S4_16_s4)      RS=16; N=512; TAU=2; CARD=R3; ORDER=6;;
    S4_16_s2)      RS=16; N=512; TAU=2; ELL=2; CARD=R3; ORDER=7;;
    S4_16_s4_tau1) RS=16; N=512; TAU=1; TEND=16; CARD=R3; ORDER=8;;
    M3c_s4)        M=0.1;  RS=96; N=3072; CARD=R3; ORDER=9;;
    M3d_s4)        M=0.05; RS=326; N=3912; K=12; TEND=30; COLL=8; CARD=R1; ORDER=1; ENGINE=v36e; EXTRA="--first-return-collar-over-rs 8 --first-return-end-rule end"; SLX="28";;   # v1c: ruling 3, the clock 28
    S4_128_s4)     RS=128; N=4096; CARD=R2; ORDER=1;;
    S4_128_s8)     RS=128; N=4096; ELL=8; CARD=R2; ORDER=2;;
    S4_128_s16)    RS=128; N=4096; ELL=16; CARD=R2; ORDER=3;;
    S4_R2_m0p2)    R0=2; SLX="3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10"; CARD=R2; ORDER=4;;          # v1c: ruling 6, every 0.5 r_s from 3 to 10
    S4_R2_m0p1)    M=0.1; R0=2; SLX="3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10"; CARD=R2; ORDER=5;;
    *) echo "unknown member $1" >&2; return 1;;
  esac
  MT=$(echo $M | tr . p)
  TAG=lad_${1}_m${MT}_rs${RS}_n${N}_R0${R0}_k${K}_tau${TAU}_ell${ELL}_prep${PREP}_A_R2_avg_plain_${ENGINE}
  E=$CODE/engine_collapse_${ENGINE}.py
  RUN=$ROOT/runs/$TAG; STD=$ROOT/states/$TAG; LOG=$ROOT/logs/$TAG.log; STF=$ROOT/status_$1.txt; PIDF=$ROOT/pids/$1.pid
  JSON=$RUN/collapse_$TAG.json
  POLICY=${SLICE_POLICY:-full}; POLSRC="SLICE_POLICY"   # v1c: ruling 3, the per-member policy file read at each start or resume
  if [ -f $ROOT/policy_$1 ]; then POLICY=$(tr -dc a-z < $ROOT/policy_$1); POLSRC="$ROOT/policy_$1"; fi
  case $POLICY in full|lean|bounce) ;; *) echo "member $1: policy '$POLICY' from $POLSRC is not full, lean or bounce" >&2; return 1;; esac
}
queue_of() { [ -f $ROOT/queue_$1 ] && { tr '\n' ' ' < $ROOT/queue_$1; return 0; }   # v1c: a card's queue file overrides the table (#172: pod 6 takes R3's first members)
  case $1 in R1) echo "M3d_s4";; R2) echo "S4_128_s4 S4_128_s8 S4_128_s16 S4_R2_m0p2 S4_R2_m0p1";;
  R3) echo "M3a_s4 M3a_s2 M3a_s4_tau4 M3b_s4 M3b_s8 S4_16_s4 S4_16_s2 S4_16_s4_tau1 M3c_s4";; *) return 1;; esac; }
lists() { eval "$($PY $HELPER slices $RS $TAU $TEND $POLICY "${SLX:-none}")"; }
estimates() { eval "$($PY $HELPER estimate $N $K $RS $TAU $TEND $POLICY "${SLX:-none}")"; }
base_args() { echo "--m $M --g 0.25 --rs $RS --R0-over-rs $R0 --reading A --nsites $N --kappa-sea $K --reference R2 --site-fields averaged --prep-ell $PREP --ell $ELL --tau $TAU --T-over-rs $TEND --gpu $GPU --mix plain --field-kernel-cache --v36-fixes"; }
engine_args() {   # $1 out dir, $2 tag, $3 slice dir
  local A; A="$(base_args) --slices-c128 $C128"; [ -n "$C64" ] && A="$A --slices-c64 $C64"
  echo "$A $EXTRA --slice-dir $3 --out $1 --tag $2"
}
hours_since() { awk "BEGIN{printf \"%.2f\", ($(date -u +%s) - $1)/3600}"; }
cost_of() { [ -n "$RATE" ] && awk "BEGIN{printf \"%.2f USD at $RATE USD/h\", $1 * $RATE}" || echo "rate unset"; }

mode_check() {
  st "=== check $MEMBER tag $TAG card $CARD order $ORDER engine $ENGINE policy $POLICY (from $POLSRC) ==="
  local rc=0
  [ -e $E ] || { st "FAIL: engine $E missing"; return 1; }
  st "engine $E md5 $(md5sum $E | cut -c1-32); code tree md5 $($PY $HELPER chainmd5 $CODE 2>/dev/null) by the helper's method (every .py sorted by path, concatenated; the facts give 74e9948d2d5eb7f3463a3df8f8cb769f for code_v36d by its own method)"
  if [ $ENGINE = v36e ]; then
    st "v36e build check: $($PY $CODE/engine_collapse_v36e_selftest.py --build-check $CODE/engine_collapse_v36d.py $CODE/engine_collapse_v36e.py 2>&1 | tail -1)"
    st "v36e runtime gate: $(grep -h 'GATE' $ROOT/status_M3d_s4.txt 2>/dev/null | tail -1 | cut -c1-120)"
  fi
  st "environment: $($PY $HELPER env 2>&1 | tr '\n' ';') expected (facts): python 3.11.10 numpy 2.4.6 scipy 1.17.1 torch 2.4.1+cu124"
  st "cuda: $(nvidia-smi -i $GPU --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader 2>&1 | head -1); compute apps now: $(nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ' ')"
  estimates
  st "estimate (derived): device memory $MEM_MIB MiB; slice $SLICE_GB GB complex128 ($N_C128 of them, $N_C64 complex64 of $SLICE64_GB GB); about $TOTAL_GB GB on /data; $RECORDS records; $HOURS_LO to $HOURS_HI h at the base pace; guard pause below $PAUSE_GB GB free on /workspace, resume above $RESUME_GB GB"
  local fm; fm=$(gpu_free_mib); local need=$(( MEM_MIB * 13 / 10 ))
  if [ -n "$fm" ] && [ "$fm" -ge "$need" ]; then st "device memory: free $fm MiB >= 1.3 x estimate $need MiB"; else st "FAIL: device memory free ${fm:-unknown} MiB < 1.3 x estimate $need MiB"; rc=1; fi
  local ws; ws=$(free_gb $ROOT)
  st "disk: $ROOT free ${ws:-?} GB on $(df -T $ROOT 2>/dev/null | tail -1 | awk '{print $1" ("$2")"}'); container disk / free $(free_gb /) GB; need at least $RESUME_GB GB free on /workspace"
  [ -n "$ws" ] && [ "$ws" -ge "$RESUME_GB" ] || { st "FAIL: /workspace free below $RESUME_GB GB"; rc=1; }
  lists
  st "slices: complex128 at T/rs $C128; complex64 at ${C64:-none}"
  st "command: cd $RUN && nice -n 10 $PY $E $(engine_args $RUN $TAG $STD)"
  st "$($PY $HELPER clocks $RS $N $R0 $COLL)"
  [ -e $JSON ] && st "note: $JSON exists (start would refuse; resume applies)"
  return $rc
}

t0_lines() {   # $1 json, $2 slice; the T = 0 check into the status file; returns the helper's verdict
  local out rc
  out=$($PY $HELPER t0 "$1" "$2" $M 0.25 $RS $K 2>&1); rc=$?
  while IFS= read -r l; do st "t0: $l"; done <<< "$out"
  st "t0: $($PY $HELPER clocks $RS $N $R0 $COLL $(echo "$out" | grep -o 'R_sh [0-9.]* a' | head -1 | awk '{print $2}'))"
  return $rc
}

mode_t0() {
  st "=== t0 $MEMBER (the preparation and the T = 0 record by --benchmark) ==="
  local TD=$ROOT/t0check/$TAG; mkdir -p $TD/states
  if [ ! -e $TD/collapse_${TAG}_t0.json ]; then
    ( cd $TD && nice -n 10 $PY $E $(base_args) --benchmark --slices-c128 0 $EXTRA --slice-dir $TD/states --out $TD --tag ${TAG}_t0 > $TD/t0.log 2>&1 )
    st "t0 run exit $? (log $TD/t0.log)"
  fi
  if t0_lines $TD/collapse_${TAG}_t0.json $TD/states/orbitals_${TAG}_t0_T0000.00.npz; then st "t0 PASS (separate run)"; rm -f $TD/states/orbitals_${TAG}_t0_T0000.00.npz; return 0
  else st "t0 FAIL (separate run); the T = 0 slice kept at $TD/states"; return 1; fi
}

launch() {   # $1 fresh|resume, $2 extra arguments (the resume's)
  mkdir -p $RUN $STD
  local ARGS; ARGS="$(engine_args $RUN $TAG $STD) $2"
  setsid nohup bash -c "cd $RUN; nice -n 10 $PY $E $ARGS > $LOG 2>&1 < /dev/null & echo \$! > $PIDF; wait \$!; echo \$? > $PIDF.rc" > /dev/null 2>&1 < /dev/null &
  sleep 3; rm -f $PIDF.rc
  local P; P=$(cat $PIDF 2>/dev/null)
  date -u +%s > $PIDF.t0
  st "launched ($1) pid ${P:-?} at $(now); engine $ENGINE; policy $POLICY (from $POLSRC); log $LOG; command: $PY $E $ARGS"
  setsid nohup bash $SELF $MEMBER _guard $P > /dev/null 2>&1 < /dev/null &
  setsid nohup bash $SELF $MEMBER _watch $P $1 > /dev/null 2>&1 < /dev/null &
}

mode_start() {
  st "=== start $MEMBER ==="
  [ -e $JSON ] && { st "REFUSED: $JSON exists; never a fresh launch over an existing run (use resume)"; return 1; }
  alive && { st "REFUSED: pid $(cat $PIDF) alive"; return 1; }
  ls $ROOT/failed/${TAG}_* > /dev/null 2>&1 && { st "REFUSED: a quarantined run of $TAG sits in $ROOT/failed (the lead decides)"; return 1; }
  if [ $ENGINE = v36e ] && ! grep -q 'GATE PASS' $STF 2>/dev/null; then st "REFUSED: no GATE PASS of the v36e gate in $STF (run gate36e first)"; return 1; fi
  if ! mode_check; then [ "${FORCE_CHECK:-0}" = 1 ] && st "check failed; FORCE_CHECK=1: proceeding" || { st "start refused: the check failed (FORCE_CHECK=1 overrides)"; return 1; }; fi
  lists; launch fresh ""
}

mode_guard() {   # $1 pid: pause below 2 slices' room on /workspace, resume above 4 slices' room
  local P=$1 paused=0 f; estimates
  while kill -0 $P 2>/dev/null; do
    f=$(free_gb $ROOT)
    if [ $paused = 0 ] && [ -n "$f" ] && [ "$f" -lt "$PAUSE_GB" ]; then kill -STOP $P && paused=1 && st "guard: paused pid $P at $f GB free (below 2 slices' room, $PAUSE_GB GB)"; fi
    if [ $paused = 1 ] && [ -n "$f" ] && [ "$f" -ge "$RESUME_GB" ]; then kill -CONT $P && paused=0 && st "guard: resumed pid $P at $f GB free (above 4 slices' room, $RESUME_GB GB)"; fi
    sleep 30
  done
  st "guard: pid $P gone"
}

mode_watch() {   # $1 pid, $2 fresh|resume: the inline T = 0 check (fresh), progress every 30 min, hours and cost at the end
  local P=$1 kind=$2 checked=0 t0 tick=0
  t0=$(cat $PIDF.t0 2>/dev/null || date -u +%s)
  [ $kind = resume ] && checked=1
  while kill -0 $P 2>/dev/null; do
    if [ $checked = 0 ] && [ -e $JSON ] && $PY $HELPER last $JSON > /dev/null 2>&1; then
      local S0=$STD/orbitals_${TAG}_T0000.00.npz
      if [ -e $S0 ] && [ $(( $(date +%s) - $(stat -c %Y $S0) )) -gt 120 ]; then
        if t0_lines $JSON $S0; then st "t0 PASS (inline, on the run's own first record)"; checked=1
        else
          st "t0 FAIL: the run is stopped and quarantined under $ROOT/failed"
          kill -TERM $P; sleep 10; kill -KILL $P 2>/dev/null; sleep 2
          local stamp; stamp=$(date -u +%Y%m%dT%H%M%SZ); mv $RUN $ROOT/failed/${TAG}_$stamp; mv $STD $ROOT/failed/${TAG}_${stamp}_states
          echo "$(now) $MEMBER t0 FAIL quarantined ${TAG}_$stamp" >> $ROOT/STATUS; break
        fi
      fi
    fi
    tick=$((tick + 1))
    if [ $((tick % 30)) = 0 ]; then st "progress: $(grep -o '^T/rs [0-9.]*' $LOG 2>/dev/null | tail -1); $(grep 'projection at' $LOG 2>/dev/null | tail -1 | cut -c1-140); slices on the pod $(ls $STD 2>/dev/null | wc -l); /workspace free $(free_gb $ROOT) GB; $(hours_since $t0) h so far"; fi
    sleep 60
  done
  sleep 5
  local H; H=$(hours_since $t0)
  local fin; fin=$($PY $HELPER finished $JSON $TEND $TAU $RS 2>&1) && fin="yes ($fin)" || fin="no ($fin)"
  st "exit: pid $P ended at $(now) rc $(cat $PIDF.rc 2>/dev/null); wall $H h this launch ($kind); cost $(cost_of $H); last record $($PY $HELPER last $JSON 2>/dev/null); finished: $fin; log tail: $(tail -n 1 $LOG 2>/dev/null | cut -c1-160)"
  echo "$(now) $MEMBER ($kind) $H h, $(cost_of $H); finished: $fin" >> $ROOT/STATUS
}

mode_resume() {
  st "=== resume $MEMBER ==="
  [ -e $JSON ] || { st "REFUSED: no stored run $JSON (use start)"; return 1; }
  alive && { st "REFUSED: pid $(cat $PIDF) alive"; return 1; }
  local S TS; read -r S TS < <($PY $HELPER newest $STD $JSON $RUN/collapse_${TAG}_fields.npz)
  [ -n "${S:-}" ] || { st "REFUSED: no usable complex128 slice in $STD (needs an accepted record and a fields row at its clock)"; return 1; }
  st "newest usable complex128 slice $S at T/rs $TS"
  eval "$($PY $HELPER after $JSON $TS)"
  lists
  if [ "$N_AFTER" -ge 2 ]; then
    local GD=$ROOT/gates/${TAG}_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $GD/states
    st "gate: the engine from the slice to T/rs $T2 (2 records, $T1 and $T2) into $GD"
    ( cd $GD && nice -n 10 $PY $E $(engine_args $GD ${TAG}_gate $GD/states) --resume-slice $S --resume-run $RUN/$TAG --segment-end-T-over-rs $T2 > $GD/gate.log 2>&1 )
    st "gate run exit $? (log $GD/gate.log)"
    local out rc; out=$($PY $HELPER gatecmp $RUN $TAG $GD ${TAG}_gate $T1 $T2 2>&1); rc=$?
    while IFS= read -r l; do st "gate: $l"; done <<< "$out"
    local CMP=${CMP_MEMBER_FLOOR:-$CODE/compare_member_floor_v2.py}
    if [ -e $CMP ]; then
      local TPL=${CMP_TEMPLATE:-"{PY} {CMP} --criterion {GATE} {STORED} --T-over-rs {T}"}
      for T in $T1 $T2; do
        local cmd; cmd=$(echo "$TPL" | sed "s#{PY}#$PY#; s#{CMP}#$CMP#; s#{GATE}#$GD/${TAG}_gate#; s#{STORED}#$RUN/$TAG#; s#{T}#$T#")
        st "compare_member_floor_v2 --criterion at T/rs $T: $cmd"
        st "compare_member_floor_v2 output: $(eval $cmd 2>&1 | tail -n 3 | tr '\n' ' ') exit ${PIPESTATUS[0]}"
      done
    else st "compare_member_floor_v2 not found at $CMP; the launcher's comparison decides"; fi
    rm -rf $GD/states
    if [ $rc != 0 ]; then [ "${FORCE_RESUME:-0}" = 1 ] && st "gate failed; FORCE_RESUME=1: proceeding" || { st "resume refused: the 2-record gate failed (FORCE_RESUME=1 overrides)"; return 1; }; fi
  else st "gate: $N_AFTER stored record(s) past the slice, fewer than 2: no gate; the resume proceeds"; fi
  local BK=$RUN/pre_resume_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $BK; cp $RUN/collapse_${TAG}*.json $RUN/collapse_${TAG}*.npz $BK/ 2>/dev/null
  st "stored json and npz backed up to $BK (the resumed run rewrites the records from the slice onward)"
  launch resume "--resume-slice $S --resume-run $RUN/$TAG"
}

mode_status() {
  echo "member $MEMBER tag $TAG card $CARD order $ORDER engine $ENGINE policy $POLICY"
  echo "pid: $(cat $PIDF 2>/dev/null) alive: $(alive && echo yes || echo no); rc: $(cat $PIDF.rc 2>/dev/null); launched: $(date -u -d @$(cat $PIDF.t0 2>/dev/null || echo 0) +%FT%TZ 2>/dev/null)"
  echo "json: $( [ -e $JSON ] && $PY $HELPER last $JSON 2>&1 || echo none )"
  echo "finished: $($PY $HELPER finished $JSON $TEND $TAU $RS 2>&1 && echo yes || echo no)"
  echo "newest usable slice: $($PY $HELPER newest $STD $JSON $RUN/collapse_${TAG}_fields.npz 2>/dev/null); slices on the pod: $(ls $STD 2>/dev/null | wc -l)"
  echo "log: $(grep -o '^T/rs [0-9.]*' $LOG 2>/dev/null | tail -1); $(grep 'projection at' $LOG 2>/dev/null | tail -1 | cut -c1-160)"
  echo "disk: $(df -h $ROOT 2>/dev/null | tail -1); device: $(nvidia-smi -i $GPU --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null)"
  echo "status file tail ($STF):"; tail -n 15 $STF 2>/dev/null
}

mode_gate36e() {   # the v36e gate: 3 records of S4_16_s4 on v36d and on v36e with the option off, compared bitwise
  local STF0=$STF; member S4_16_s4; STF=$STF0
  local G=$ROOT/gates/v36e_off_gate_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $G/d/states $G/e/states
  st "=== gate36e: 3 records (T/rs 0, 0.125, 0.25) of S4_16_s4 on engine_collapse_v36d and on engine_collapse_v36e with its option off, into $G ==="
  ( cd $G/d && nice -n 10 $PY $CODE/engine_collapse_v36d.py $(base_args) --slices-c128 0 --segment-end-T-over-rs 0.25 --slice-dir $G/d/states --out $G/d --tag g > $G/d/run.log 2>&1 ); st "v36d segment exit $?"
  ( cd $G/e && nice -n 10 $PY $CODE/engine_collapse_v36e.py $(base_args) --slices-c128 0 --segment-end-T-over-rs 0.25 --slice-dir $G/e/states --out $G/e --tag g > $G/e/run.log 2>&1 ); st "v36e segment exit $?"
  local out; out=$($PY $CODE/engine_collapse_v36e_selftest.py --gate $G/d/g $G/e/g 2>&1)
  while IFS= read -r l; do st "gate36e: $l"; done <<< "$out"
}

mode_queue() {   # MEMBER is the card: the plan's order, each member started or resumed once when the card is idle
  local card=$MEMBER m attempts Q
  Q=$(queue_of $card) || exit 1   # v1b: the builder's syntax fix (v1 wrote "for m in $(queue_of $card) || exit 1; do", which bash refuses)
  for m in $Q; do
    member $m; attempts=0
    while true; do
      if $PY $HELPER finished $JSON $TEND $TAU $RS > /dev/null 2>&1; then st "queue $card: $m finished; next"; break; fi
      if alive; then sleep 300; continue; fi
      if ls $ROOT/failed/${TAG}_* > /dev/null 2>&1; then st "queue $card: $m quarantined; blocked"; echo "$(now) $card queue blocked at $m (quarantined)" >> $ROOT/STATUS; exit 1; fi
      if [ -e $JSON ]; then
        [ $attempts -ge 1 ] && { st "queue $card: $m not finished after one resume; blocked"; echo "$(now) $card queue blocked at $m" >> $ROOT/STATUS; exit 1; }
        attempts=$((attempts + 1))
        while gpu_busy; do st "queue $card: the card is busy with another process; waiting"; sleep 300; done
        mode_resume || { echo "$(now) $card queue blocked at $m (resume refused)" >> $ROOT/STATUS; exit 1; }
      else
        while gpu_busy; do st "queue $card: the card is busy with another process (R3's rerun); waiting"; sleep 300; done
        mode_start || { echo "$(now) $card queue blocked at $m (start refused)" >> $ROOT/STATUS; exit 1; }
      fi
      sleep 120
    done
  done
  echo "$(now) $card queue empty" >> $ROOT/STATUS; st "$card queue empty"
}

case $MODE in
  queue) STF=$ROOT/STATUS; mode_queue;;
  *) member $MEMBER || exit 1
     case $MODE in
       check) mode_check;; t0) mode_t0;; start) mode_start;; resume) mode_resume;; status) mode_status;; gate36e) mode_gate36e;;
       _guard) mode_guard ${3:?pid};; _watch) mode_watch ${3:?pid} ${4:-fresh};;
       *) echo "unknown mode $MODE" >&2; exit 1;;
     esac;;
esac
