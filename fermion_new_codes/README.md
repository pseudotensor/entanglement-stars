# Codes for the fermion paper

Scientific source code (Python and shell drivers) that produced the numbers and figures of the fermion paper. It is grouped by the working group that wrote and ran it, and each group's original directory layout is kept below its folder.

| Folder | What it covers |
|--------|----------------|
| `busi/` | Collapse engine runs and their readouts; the non-conservation checks (`noncons/`, `noncons_indep/`, backing `../proofs/busi/NONCONS_*`); the mock-recovery / clock-well audits (`clockwell*/`, backing `../proofs/busi/MOCK_recovery_audit_v1.2.md`); foundations checks; the Page-curve tests; and the paper figure scripts (`figures/`, for example `figure_page_v5.py` and `figures_data_v1.py`). |
| `solve_a/` | The collapse engine (`engine/`), Page-curve recursion and readouts (`page/`, `page_v4/`, `page_verify/`), first-law and flow computations (`firstlaw/`, `flow/`), the ladders, and the strand computations (`strand0/` to `strand7/`, `strandS/`) behind the radiation, entropy and collapse sections. |
| `hreduce/` | The derivation-check package: short checks for the `declared_*` derivations, and the vertex gate, amplitude, force and stress programs (`vertex_*`) behind `../proofs/hreduce/VERTEX_*`. |
| `solve_b/` | Independent second-route computations: the ladder arithmetic, curvature, evolved-cut and purification checks, and the Page balance readouts behind `../proofs/solve_b/`. |
| `second_route/` | Independent second-route checks of the B' channel and gate computations. |

Notes:
- Absolute paths in the scripts pointed to the group's own machines and bulk storage. In Python they have been replaced by `<internal>/<name>` placeholders, and in shell scripts by `${QI2_ROOT}/<name>`. Set them to your own locations before running anything.
- Internal reviewer or lane names in comments and docstrings were replaced by neutral labels. No logic was changed.
- Left out: operations scripts (copying and mirroring to remote compute pods), scripts that call external language-model APIs, credential tests, logs, checkpoints and caches.
- Most scripts need `numpy` and `scipy`. Some need `jax`, `cupy` or `torch` for GPU runs, and a few need `clarabel` or `python-flint`.
- `MANIFEST.tsv` lists every file with its size, the md5 of the original source file, and whether the text was neutralized (1) or copied byte-for-byte (0).
