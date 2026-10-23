# Data for the fermion paper

Small result files (`.npz`, `.npy`, `.json`, `.jsonl`, `.csv`, `.tsv`) behind printed numbers and figures in the fermion paper. They are grouped by working group to match `../fermion_new_codes/` and `../proofs/`, and each group's run-directory layout is kept.

Selection rules:
- Every result file of 2 MB or less is included.
- Files of 2 to 10 MB are included only when one of the group's issued derivations or reports references them.
- Files over 10 MB were left out to stay under the repository budget. Every omitted file is listed with its size and md5 in `../SKIPPED_LARGE.tsv`, and a copy is available on request.

Text data files (`.json`, `.tsv`, `.csv`, `.jsonl`) had internal absolute paths in provenance fields replaced by `<internal>/<name>`, and internal reviewer or lane names replaced by neutral labels. Numerical values are unchanged. Binary arrays (`.npz`, `.npy`) are byte-identical to the originals. `MANIFEST.tsv` gives each file's original md5 and whether it was neutralized.
