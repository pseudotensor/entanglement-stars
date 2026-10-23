"""Losslessly render stored certificate witnesses; no numerical calculation."""

import json
from pathlib import Path


def main():
    directory = Path(__file__).resolve().parent
    data = json.loads((directory / "results.json").read_text())
    lines = ["Complete stored certificate witnesses", "",
             "Every printed endpoint/value is copied without discarding digits.",
             "One witness list is printed only when the two stored lists are exactly equal.",
             "All aggregate records at both arithmetic precisions follow the lists.", "",
             "Settings: " + json.dumps(data["settings"], sort_keys=True),
             "Runtime: " + json.dumps(data["runtime"], sort_keys=True),
             "Analytic tail: " + json.dumps(data["analytic_tail"], sort_keys=True)]
    equality = {}
    for phase, key in [("A", "accepted_cells"), ("B", "midpoint_terms")]:
        same = data[phase]["70"][key] == data[phase]["100"][key]
        equality[phase] = same
        lines.extend(["", f"Phase {phase}: serialized 70/100 witness lists exactly identical = {same}"])
        for precision in (["70"] if same else ["70", "100"]):
            lines.append(f"Complete phase {phase} witness list ({precision} arithmetic digits):")
            for row in data[phase][precision][key]:
                lines.append(json.dumps(row, separators=(",", ":"), sort_keys=True))
        for precision in ["70", "100"]:
            aggregate = {k: v for k, v in data[phase][precision].items() if k != key}
            lines.extend([f"Phase {phase} aggregate ({precision} digits):",
                          json.dumps(aggregate, sort_keys=True)])
    lines.extend(["", "Complete C records: " + json.dumps(data["C"], sort_keys=True),
                  "C precision differences: " + json.dumps(data.get("differences", {}), sort_keys=True),
                  "Combined certificate passed: " + str(data["certificate_passed"])])
    target = directory / "CERTIFICATE_WITNESSES.txt"
    target.write_text("\n".join(lines) + "\n")
    print(json.dumps({"serialized_lists_identical": equality,
                      "rendering": str(target), "lines": len(lines)}))


if __name__ == "__main__":
    main()
