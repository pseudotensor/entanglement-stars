"""Extract the files of a reviewer A answer written as '=== FILE: <path> ===' ... '=== END FILE ===' blocks.
Usage: python extract_files.py <answer.md> <root dir>. Prints each path and its size; refuses paths outside busi/."""
import os, re, sys
text = open(sys.argv[1], encoding="utf-8").read()
root = sys.argv[2]
pat = re.compile(r"^=== FILE: (.+?) ===\n(.*?)^=== END FILE ===", re.S | re.M)
n = 0
for m in pat.finditer(text):
    rel = m.group(1).strip()
    if not rel.startswith("busi/") or ".." in rel:
        print("REFUSED", rel); continue
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(m.group(2))
    print(f"{rel}  {len(m.group(2))} chars"); n += 1
print(n, "files")
tail = text[text.rfind("=== END FILE ===") + len("=== END FILE ==="):]
print("---- text after the last file ----"); print(tail.strip()[:4000])
