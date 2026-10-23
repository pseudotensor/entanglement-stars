#!/usr/bin/env python
"""test_w10_defect14_audit_v2.py (busi W10): defect 14 of REVIEWA_REVIEW_web40i section 5, the quoted-code audit items (sources fixed at the
start state during the field iteration; dE comparing inner iterations, not records; a best iterate saved but not restored; a failed solve
reaching the propagation), conditional on unquoted guards. The builder confirms or repairs each with a test; gate: the test.
Static confirmations on the engine source of the guards named in W10_fix_blocks_v2.md defect 14, unit tests of w10_lib_v1.e_prev_rule
(item (b)), and the (d) guard executed on a failed info.
Version 2 (W10_reading_v1.md section 3, after the box's first round, which audited v36d's text under a stale v36e name):
  (i) the audited file is labelled by its content (the line "import w10_lib_v1 as W10" present: repaired, v36e or v36g; absent: unrepaired,
      v36d or W2's v36f), not by its name; a file named engine_collapse_v36e.py or _v36g.py with unrepaired content is reported stale and
      FAILs (the hazard: a builder basing on it);
  (ii) the two lines that need the applied blocks ((b)'s "uses e_prev_rule", (d)'s "repaired: guarded") are EXPECTED-FAIL on unrepaired
      content and FAIL on repaired content (exit 2 for the pre-apply state, 1 for a FAIL, 0 for all PASS);
  (iii) item (c) prints every return statement of newton_record and newton_record_v32 with its line, and the statements binding the saved
      best; its rule: each function binds the saved best (b = nk["best"], b = final["best"], or stores nk["best"] = ...), and every return
      at or after that binding hands it back (the dict, its container or names unpacked from its "sol"/"cache"/"app"/"srcs"); returns before
      the binding, returns of None/constants and returns whose text carries "failed" or "stopped" are the failure exits run() breaks on. v1
      demanded the unpacked form at both sites and failed at its count, which the log could not separate from a real defect;
  (iv) item (d)'s guard is exercised, not only grepped: the statements "N, v, Wn, info = solve(...)" and the guard that follows it are
      extracted from ball_fixed_point_v33 by ast into a probe function whose solve is patched to return a failed info; the probe must raise
      RuntimeError naming the massless ball and the injected failure (the whole preparation cannot be called here: it needs the S11 state);
      a NameError in the probe is a guard that names something out of scope (k0 is also checked to be bound in the function).
--engine FILE audits that file (v36d, v36e, W2's v36f, v36g); the default is engine_collapse_v36e.py in $W10_ENGINE_DIR, else v36d there.
--import-engine also imports the module (torch needed; seconds) and reports whether ball_fixed_point_v33, solve and the lib resolve in it.
--results GLOB lists the delivered records' flags (which runs the findings concern).
Run: qi2 python test_w10_defect14_audit_v2.py [--engine FILE] [--import-engine] [--results 'DIR/**/collapse_*.json']"""
import argparse
import ast
import glob
import importlib
import json
import os
import sys
import textwrap
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w10_lib_v1 as W10  # noqa: E402

FAIL = []
EXPECTED = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def check_needs_apply(cond, msg, repaired):
    """a line that needs the applied blocks: FAIL on repaired content, EXPECTED-FAIL on unrepaired content."""
    if cond:
        print("PASS " + msg, flush=True)
    elif repaired:
        print("FAIL " + msg, flush=True); FAIL.append(msg)
    else:
        print("EXPECTED-FAIL (before the apply) " + msg, flush=True); EXPECTED.append(msg)


def find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def own_nodes(f):
    """every node of f's body outside nested function and class definitions (their returns are theirs, not f's)."""
    out = []
    stack = list(f.body)
    while stack:
        node = stack.pop()
        out.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            stack.append(child)
    return out


def seg(src, node):
    s = ast.get_source_segment(src, node)
    return " ".join(s.split()) if s else "<source segment not available>"


def sub_key(sl):
    """the string key of a subscript's slice (x["best"] -> "best"), None otherwise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        idx = getattr(ast, "Index", None)
    if idx is not None and isinstance(sl, idx):                      # python 3.8
        sl = sl.value
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value
    return None


def fn_args(f):
    return [x.arg for x in f.args.posonlyargs + f.args.args + f.args.kwonlyargs]


def audit_returns(src, tree, fname):
    """item (c) for one function: (ok, printed lines)."""
    f = find_func(tree, fname)
    if f is None:
        return False, ["%s: not found in the source" % fname]
    nodes = own_nodes(f)
    assigns = sorted((n for n in nodes if isinstance(n, ast.Assign)), key=lambda n: n.lineno)
    best_names, containers, unpacked, lines = set(), set(), set(), []
    first_bind = None
    for n in assigns:
        v = n.value
        if isinstance(v, ast.Subscript) and sub_key(v.slice) == "best" and isinstance(v.value, ast.Name):
            containers.add(v.value.id)
            for t in n.targets:
                if isinstance(t, ast.Name):
                    best_names.add(t.id)
            first_bind = n.lineno if first_bind is None else min(first_bind, n.lineno)
            lines.append("%s line %d: %s   (the saved best bound to a name)" % (fname, n.lineno, seg(src, n)))
        for t in n.targets:
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and sub_key(t.slice) == "best":
                containers.add(t.value.id)
    for n in assigns:
        v = n.value
        if isinstance(v, ast.Subscript) and isinstance(v.value, ast.Name) and v.value.id in best_names and sub_key(v.slice) in ("sol", "cache", "app", "srcs"):
            for t in n.targets:
                for e in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if isinstance(e, ast.Name):
                        unpacked.add(e.id)
            lines.append("%s line %d: %s   (unpacked from the saved best)" % (fname, n.lineno, seg(src, n)))
    rets = sorted((n for n in nodes if isinstance(n, ast.Return)), key=lambda n: n.lineno)
    ok = bool(best_names or containers) and bool(rets)
    if not (best_names or containers):
        lines.append("%s: no statement binds or stores a saved best (nothing like b = nk[\"best\"] or nk[\"best\"] = ...)" % fname)
    hands = 0
    for r in rets:
        text = seg(src, r)
        refs = {n.id for n in ast.walk(r) if isinstance(n, ast.Name)} if r.value is not None else set()
        if refs & (best_names | unpacked):
            kind = "hands back the saved best (the dict or its unpacked fields)"; hands += 1
        elif refs & containers:
            kind = "hands back the container that holds the saved best (the caller unpacks it)"; hands += 1
        elif first_bind is not None and r.lineno < first_bind:
            kind = "an exit before the best is bound (a failure or stopped exit)"
        elif r.value is None or isinstance(r.value, ast.Constant) or any(w in text for w in ("failed", "stopped")):
            kind = "a failure exit"
        else:
            kind = "DOES NOT hand back the saved best: a defect 14 (c) unless it is a failure exit under another name (read it; block D14.3 if it returns another iterate)"
            ok = False
        lines.append("%s line %d: %s   -> %s" % (fname, r.lineno, text, kind))
    if hands == 0:
        ok = False
        lines.append("%s: no return hands back the saved best" % fname)
    return ok, lines


def is_solve_assign(st):
    if not (isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name) and st.value.func.id == "solve"):
        return False
    t = st.targets[0]
    names = [e.id for e in t.elts if isinstance(e, ast.Name)] if isinstance(t, ast.Tuple) else ([t.id] if isinstance(t, ast.Name) else [])
    return "info" in names


def ball_guard(src, tree):
    """(f, the solve statement before project_at, the guard If that follows it or None, a reason when nothing is found)."""
    f = find_func(tree, "ball_fixed_point_v33")
    if f is None:
        return None, None, None, "ball_fixed_point_v33 not found in the source"
    body = f.body
    idx_pa = next((i for i, st in enumerate(body) if isinstance(st, ast.FunctionDef) and st.name == "project_at"), None)
    cands = [i for i, st in enumerate(body) if is_solve_assign(st) and (idx_pa is None or i < idx_pa)]
    if not cands:
        return f, None, None, "no 'N, v, Wn, info = solve(...)' at the top level of ball_fixed_point_v33's body before project_at"
    i = cands[-1]
    nxt = body[i + 1] if i + 1 < len(body) else None
    guard = nxt if isinstance(nxt, ast.If) and 'info.get("failed")' in (ast.get_source_segment(src, nxt.test) or "") else None
    return f, body[i], guard, None


def run_guard_probe(src, f, solve_stmt, guard):
    """the solve statement and its guard as a function of f's arguments (all None, k0 0.1234) with solve patched to return a failed info:
    (the RuntimeError's message or None, an error text or None)."""
    block = textwrap.dedent("\n".join(src.split("\n")[solve_stmt.lineno - 1: guard.end_lineno]))
    args = fn_args(f)
    params = ", ".join("%s=%r" % (x, 0.1234 if x == "k0" else None) for x in args)
    probe = "def w10_probe(%s):\n%s\n" % (params, textwrap.indent(block, "    "))
    t = solve_stmt.targets[0]
    names = [e.id if isinstance(e, ast.Name) else None for e in t.elts] if isinstance(t, ast.Tuple) else [t.id]
    vals = [None] * len(names)
    vals[names.index("info")] = {"failed": "injected failure"}
    loaded = {n.id for n in ast.walk(solve_stmt.value) if isinstance(n, ast.Name)} | {n.id for n in ast.walk(guard) if isinstance(n, ast.Name)}
    ns = {n: None for n in loaded if n not in args and n != "solve"}
    if "k0" in ns:
        ns["k0"] = 0.1234
    ns["solve"] = (lambda *a_, **k_: tuple(vals)) if len(names) > 1 else (lambda *a_, **k_: vals[0])
    try:
        exec(compile(probe, "<w10 guard probe>", "exec"), ns)
        ns["w10_probe"]()
    except RuntimeError as ex:
        return str(ex), None
    except Exception as ex:  # noqa: BLE001
        return None, "%s: %s" % (type(ex).__name__, ex)
    return None, "the probe returned without raising"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None, help="the engine source to audit (default: engine_collapse_v36e.py in $W10_ENGINE_DIR, else v36d)")
    ap.add_argument("--import-engine", action="store_true", help="also import the module and report what resolves in it (torch needed)")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    d = os.environ.get("W10_ENGINE_DIR") or HERE
    path = a.engine
    if path is None:
        for nm in ("engine_collapse_v36e.py", "engine_collapse_v36d.py"):
            if os.path.exists(os.path.join(d, nm)):
                path = os.path.join(d, nm)
                break
    if path is None or not os.path.exists(path):
        check(False, "no engine source found in %s (set W10_ENGINE_DIR or --engine)" % d)
        sys.exit(1)
    src = open(path).read()
    repaired = "import w10_lib_v1 as W10" in src
    base = os.path.basename(path)
    print("INFO auditing %s (%s by content: the line 'import w10_lib_v1 as W10' is %s)" % (path, "repaired" if repaired else "unrepaired", "present" if repaired else "absent"), flush=True)
    if not repaired and base in ("engine_collapse_v36e.py", "engine_collapse_v36g.py"):
        check(False, "stale %s: a file under a W10 output's name with unrepaired content (rm it before any builder takes it; RUN_W10_v2 step 0)" % base)
    tree = ast.parse(src)

    def has(s, count=None):
        n = src.count(s)
        return (n >= 1) if count is None else (n == count)
    # (a) sources fixed at the start state during the field iteration: by design (version 17 (c), the frozen step)
    base_line = 'base[ch.kappa] = (dens(ch.h, s_["sea"]), dens(ch.Q, s_["sea"]), dens(ch.h, s_["dust"], s_["f"]), dens(ch.Q, s_["dust"], s_["f"]))'
    i_base = src.find(base_line); i_src = src.find("def sources(cache_):"); i_rec = src.find("recs.append(rec)"); i_prop = src.rfind('s_["sea"] = prop(s_["sea"])')
    check(i_base > 0 and i_src > i_base and has('hs_, Qs_, dd, Jd = base[ch.kappa]') and i_rec > 0 and i_prop > i_rec,
          "(a) confirmed by design: the base densities are computed once per record from the propagated state before the iteration, sources() subtracts the continued reference's densities at each trial, and the propagation follows the record (the frozen exponential step of defect 1; the W2 dg step is where the sources move within a record)")
    # (b) dE: inner iterations by design; the converged flag by the confirming step; the record rate separate; the E_prev rule tested
    check(has('conv = bool(b["resid"] < a.stop_fields and dE_c is not None and dE_c < a.stop_E)') and has('rec["dE_dT"] = (E - recs[-1]["E"]) / dT'),
          "(b) confirmed: the record's converged flag uses the confirming plain step's |dE/E| (dE_c) and the record-to-record rate is rec['dE_dT']; the history's dE column is the iteration's own measure")
    check(W10.e_prev_rule(False, True, True) and W10.e_prev_rule(False, False, False) and W10.e_prev_rule(True, True, True)
          and not W10.e_prev_rule(True, False, True) and not W10.e_prev_rule(True, True, False),
          "(b) e_prev_rule: without --v36-fixes E_prev moves at every evaluation (trials and finite differences included); with it only at an evaluation on the held set that becomes the best (version 36 (b))")
    check_needs_apply(has("W10.e_prev_rule(V36[\"fixes\"], held_ok, nk[\"best\"] is None or max(dN_, dv_) < nk[\"best\"][\"resid\"])"),
                      "(b) the Newton evaluation uses e_prev_rule (block D14.1)", repaired)
    # (c) best restored
    check(has('cache, srcs = best["cache"], best["srcs"]') and has('Nn, vn, Wn, info = best["sol"]') and has('N_app, v_app = best["app"]'),
          "(c) confirmed: the plain loop restores the smallest-residual iterate's cache, sources, solve and applied fields when unconverged")
    check(has('nk["best"] = chosen["best"]; nk["failed"] = chosen["failed"]') and has('ctr["E_prev"] = rounds[0]["E_prev_after"]'),
          "(c) confirmed: rounds_v36c keeps the chosen round's best and restores E_prev when round 1 is kept after a second round")
    allok = True
    for fname in ("newton_record", "newton_record_v32"):
        ok, lines = audit_returns(src, tree, fname)
        for ln in lines:
            print("  " + ln, flush=True)
        allok = allok and ok
    check(allok, "(c) newton_record (v31) and newton_record_v32: each binds the saved best and every return at or after that binding hands it back (the dict, its container or the values unpacked from its sol/cache/app); the returns before it are the failure exits (every return site printed above; v1's pattern demanded the unpacked form at both sites and failed at its count)")
    # (d) a failed solve never reaches the propagation
    check(has('recs.append({"T": T, "failed": info["failed"], "label": label, "iteration_history": it_hist})') and has('"failed": "no eigendecomposition at the accepted fields"'),
          "(d) confirmed: run() writes a failed record and breaks before any propagation (a failed solve, or no eigendecomposition at the accepted fields)")
    check(has('return np.full(2 * n, np.inf if V36["fixes"] else 1.0)') and has('raise RuntimeError("version 35: a field solve failed inside the exact J.v")'),
          "(d) confirmed: a failed solve inside the Newton loop is never saved as the best iterate (inf residual with --v36-fixes, version 32's fabricated 1.0 without it, which corrupts the direction and not the kept iterate); a failed pair solve in the exact J.v raises")
    f_ball, solve_stmt, guard, reason = ball_guard(src, tree)
    if reason:
        check(False, "(d) ball_fixed_point_v33: %s" % reason)
    elif guard is None:
        check_needs_apply(False, "(d) repaired: ball_fixed_point_v33's solve of the converged sources (line %d) is guarded before project_at (unguarded here: block D14.2 adds the guard)" % solve_stmt.lineno, repaired)
    else:
        msg, err = run_guard_probe(src, f_ball, solve_stmt, guard)
        k0_ok = "k0" in fn_args(f_ball) or any(isinstance(n, ast.Assign) and n.lineno < guard.lineno and any(isinstance(t, ast.Name) and t.id == "k0" for t in n.targets) for n in own_nodes(f_ball))
        check(msg is not None and "massless ball" in msg and "injected failure" in msg and k0_ok,
              "(d) repaired: ball_fixed_point_v33's solve of the converged sources (line %d) is guarded before project_at (line %d); the guard, extracted into a probe whose solve returns a failed info, raised RuntimeError(%r)%s%s" % (
                  solve_stmt.lineno, guard.lineno, msg, "" if err is None else "; probe error: " + err, "" if k0_ok else "; k0 is not bound in ball_fixed_point_v33: edit D14.2's message"))
    if a.import_engine:
        try:
            edir = os.path.dirname(os.path.abspath(path))
            if edir not in sys.path:
                sys.path.insert(0, edir)
            mod = importlib.import_module(os.path.splitext(base)[0])
            print("INFO the engine module imports: ball_fixed_point_v33 %s, solve %s (a patched module-level solve reaches the guarded call), the lib W10 %s" % (
                "callable" if callable(getattr(mod, "ball_fixed_point_v33", None)) else "MISSING",
                "callable" if callable(getattr(mod, "solve", None)) else "MISSING",
                "imported" if getattr(mod, "W10", None) is not None else "absent (unrepaired content)"), flush=True)
        except Exception as ex:  # noqa: BLE001
            print("INFO the engine module does not import here (%s: %s): the static audit and the probe stand" % (type(ex).__name__, str(ex)[:160]), flush=True)
    if a.results:
        files = sorted(glob.glob(a.results, recursive=True))
        for p in files:
            try:
                J = json.load(open(p))
            except Exception as ex:  # noqa: BLE001
                print("  %s: unreadable (%s)" % (p, ex)); continue
            args = J.get("args", {}); recs = J.get("records", [])
            print("  %s: streams %s, v36_fixes %s, nk_set %s, two-set records %d, massless_ball %s, records %d, stopped records %d" % (
                p, args.get("streams"), args.get("v36_fixes"), args.get("nk_set"), sum(1 for r in recs if "v36d" in r), args.get("massless_ball"), len(recs),
                sum(1 for r in recs if str(r.get("status", "")).startswith("stopped"))))
    if FAIL:
        print("test_w10_defect14_audit_v2: %d FAIL%s" % (len(FAIL), "" if not EXPECTED else " (and %d EXPECTED-FAIL: unrepaired content)" % len(EXPECTED)))
        sys.exit(1)
    if EXPECTED:
        print("test_w10_defect14_audit_v2: ALL PASS except %d EXPECTED-FAIL (unrepaired content: run w10_apply_fixes_v2.py, then this test again)" % len(EXPECTED))
        sys.exit(2)
    print("test_w10_defect14_audit_v2: ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
