#!/usr/bin/env python3
"""b3_import_closure_v1.py (solve, page_verify B3 kit): the transitive import closure of a script within its code directory, by static
parsing (ast) of every import statement, including those inside functions (marked 'lazy'); third-party modules listed separately.
No module is executed. Usage: b3_import_closure_v1.py CODE_DIR ENTRY.py [ENTRY2.py ...] > out.json"""
import ast, hashlib, json, os, sys


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def imports_of(path):
    tree = ast.parse(open(path).read(), path)
    out = []
    for node in ast.walk(tree):
        lazy = False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # find whether inside a function: walk parents is costly; check col_offset > 0 as a proxy for a non-top-level import
            lazy = node.col_offset > 0
            if isinstance(node, ast.Import):
                for a in node.names:
                    out.append((a.name.split(".")[0], lazy, node.lineno))
            elif node.module and node.level == 0:
                out.append((node.module.split(".")[0], lazy, node.lineno))
    return out


def main():
    code = sys.argv[1]
    todo = list(sys.argv[2:]); seen = {}; third = {}
    while todo:
        f = todo.pop()
        name = os.path.splitext(os.path.basename(f))[0]
        if name in seen:
            continue
        p = os.path.join(code, os.path.basename(f))
        seen[name] = {"path": p, "bytes": os.path.getsize(p), "md5": md5(p), "imports": []}
        for mod, lazy, ln in imports_of(p):
            local = os.path.join(code, mod + ".py")
            seen[name]["imports"].append({"module": mod, "lazy": lazy, "line": ln, "local": os.path.exists(local)})
            if os.path.exists(local):
                todo.append(local)
            else:
                third.setdefault(mod, set()).add(name)
    print(json.dumps({"code_dir": code, "local": seen, "third_party": {k: sorted(v) for k, v in third.items()}}, indent=1))


if __name__ == "__main__":
    main()
