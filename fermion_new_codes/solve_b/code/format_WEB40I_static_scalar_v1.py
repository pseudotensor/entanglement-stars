#!/usr/bin/env python3
"""Format saved X7 files only; no stellar solver, quadrature or root calculation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 ** 2), b""):
            h.update(part)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    run = args.run
    data = json.loads((run / "summary.json").read_text())
    receipt = json.loads((run / "execution.json").read_text())
    if receipt.get("status") != "terminal":
        raise SystemExit("terminal receipt required")
    if receipt.get("summary_sha256") != sha(run / "summary.json"):
        raise SystemExit("receipt/summary identity mismatch")
    rows = ["# X7 fixed scalar attempt", "", "Saved data only; empirical controls, not interval enclosures.",
            "", f"Status: {data['status']}; child exit {receipt['returncode']}.",
            "", "| Level/curve | Hc | M | B | R | Local criteria |", "|---|---:|---:|---:|---:|---|"]
    for key, p in data.get("peaks", {}).items():
        if p["status"] == "complete":
            rows.append(f"| {key} | {p['Hc']:.12g} | {p['M']:.12g} | {p['B']:.12g} | {p['R']:.12g} | {p['all_local_criteria']} |")
        else:
            rows.append(f"| {key} | incomplete | | | | {p.get('reason', '')} |")
    rows += ["", "| Level/response | M_eta | B_eta | Fixed-count envelope | Envelope residual |",
             "|---|---:|---:|---:|---:|"]
    for level, r in data.get("responses", {}).items():
        for mode, p in r.get("values", {}).items():
            rows.append(f"| {level}/{mode} | {p['M_eta']:.12g} | {p['B_eta']:.12g} | {p['fixed_count_envelope']:.12g} | {p['envelope_residual']:.5g} |")
    rows += ["", "## Individual saved criteria", "", "```json",
             json.dumps({"criteria": data.get("criteria", {}), "claim_criteria": data.get("claim_criteria", {}),
                         "comparisons": data.get("comparisons", {})}, indent=2),
             "```", "", "Located peaks only. No finite-native 149, nonlinear merger 3.9, orbital maximum or production-history clearance.", ""]
    (run / "scalar.md").write_text("\n".join(rows))
    manifest = ["path\tsize_bytes\tsha256\tbox_location"]
    for p in sorted(run.rglob("*")):
        if p.is_file() and ("raw" in p.relative_to(run).parts or p.name == "process.log"):
            manifest.append(f"{p.relative_to(run)}\t{p.stat().st_size}\t{sha(p)}\tcompute-host:{p}")
    (run / "LARGE_FILES.tsv").write_text("\n".join(manifest) + "\n")
    print(json.dumps({"formatted": "scalar.md", "raw_files_left_on_box": len(manifest) - 1}))


if __name__ == "__main__":
    main()
