#!/usr/bin/env python
"""slices144_io_v1.py (reviewer A for the solve builder, the busi lane; inbox busi #144 items 1, 2 and 6; SLICE_INVENTORY_r64_v1.md).
One reader for every stored slice of the 2 r_s/a 64 histories (A: the production c19 r11, its continuation B with Brerun98's T100 slice,
the doubled chain B2) and of the sigma/a 8 member c27, and the driver that fetches each slice once and runs the 3 readouts on it
(interior_continuation_v1, core_kinematics_v1, page_own_object_v1).

The slice (every orbitals_*.npz, written by engine_collapse_v19 and v25 with np.savez, uncompressed): T, rs, R (n,), N_applied, v_applied
(n,), kappas, weights (16 kappa), dtype, and per channel kappa sea_k (2n x n), dust_k (2n x n_dust_k), f_k, R2_k (2n x n), e_k (2n,),
mask_k (packed 2n bits). Rows interleaved, 2 components per site: rows 2j and 2j + 1 belong to site j at R[j]. Every zip member is stored
(ZIP_STORED), so each array is memory-mapped at its byte offset inside the zip (zip_member_offsets reads the local file header and the
npy header) instead of loaded: a 10.7 GB B2 slice costs no RAM until a channel is asked for. The occupied columns of channel kappa are
Phi_k = [sea_k | dust_k] (f all 1), returned in complex128 (Loewdin-orthonormalized through engine_massive_channel_D_v2.loewdin by default,
the stored and the orthonormalized defects printed) with the multiplicity w = 16 kappa and the slice's N_applied, v_applied (the record's
accepted site fields). Complex64 slices (the production's 15, 25, 30, 35, 40 and every sigma/a 8 slice) are flagged and carry the floor
measured in results/engine/C3ext19/slice_restart_check_v1.md: the sources formed from the columns differ from the stored shell totals by
6.2e-6 to 8.7e-6 relative and the columns' orthonormality defect is about 6.5e-8 (a complex128 slice: 3.9e-14); every quantity read from a
complex64 slice is printed with that floor beside it.

Locations (the inventory's; box paths first): D = <internal>/solve_a_bulk, E = <internal>/engine. The dataset-only
slices (B's 40 to 56, B2's 12 to 98) come from the owner's dataset pseudotensor/qi2-fermion-bulk, downloaded one at a time into a scratch
folder on /data (the free space is checked first against the slice's size plus --min-free-gb), their sha256 compared with the dataset's
LFS metadata and with the inventory's 12-character prefix, used, and deleted after use (--keep-downloads keeps them). The token is read
inside this process from the environment after `source <internal>/gr_exports.sh` (--token-env names the variable, HF_TOKEN by default;
HUGGING_FACE_HUB_TOKEN and HF_API_TOKEN are tried next); it is never printed, logged or written.

Imports by name, never copies: page_ops_v6 (chain_sparse and clocked_csr: the heavy channel's h_m(N, v) = (1/2){N, h_0} + (1/2){v, Q} on
the stored chain; number_current_csr, current_out_csr), engine_core_slices_v3 (frac_radius), engine_massive_channel_D_v2 (loewdin,
region_rows, bin_spectrum), engine_rows_v2 (deriv), page_d13_readout_v1b (exterior_entropies, killing_vacuum, inventories, source_modes,
first_return), the histories' field modules (engine_fields_v14 for A's segments, engine_fields_v16 for the sigma/a 8 member). setup() puts
--engine-code and --page-code on sys.path and imports them; a missing optional module is recorded in SETUP and the local fallback is named
in every output that used it.

Driver: python slices144_io_v1.py --history prod,s8 --clocks all --readouts interior,core,page --out DIR [--scratch DIR]; --check lists
every slice of the inventory with its location and size (nothing computed). Resumable per slice and per readout (a slice's json present =
done; --force recomputes). Every threshold is a named argument; the raw arrays of every slice go to an npz beside its json; no plotting;
no mutual information; no dates in any output (elapsed seconds only)."""
import os
import sys


def _pre_threads():
    """the BLAS and torch thread count from --threads (16 where not given), set before numpy is imported."""
    n = None
    argv = sys.argv[1:]
    for i, x in enumerate(argv):
        if x == "--threads" and i + 1 < len(argv):
            n = argv[i + 1]
        elif x.startswith("--threads="):
            n = x.split("=", 1)[1]
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if n is not None:
            os.environ[var] = str(int(n))
        else:
            os.environ.setdefault(var, "16")


_pre_threads()
import argparse  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
from numpy.lib import format as npf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = "slices144_io_v1.py"
BOX_D = "<internal>/solve_a_bulk"
BOX_E = "<internal>/engine"
ENGINE_CODE = "<internal>/engine"
PAGE_CODE = "<internal>/v6"
HF_REPO = "pseudotensor/qi2-fermion-bulk"
SIZES = {(2048, "c128"): 2685802106, (2048, "c64"): 1343100290, (4096, "c128"): 10740296746}
C64_FLOOR = {"source_rel_range": [6.2e-6, 8.7e-6], "orthonormality_defect": 6.5e-8, "complex128_reference": 3.9e-14,
             "source": "results/engine/C3ext19/slice_restart_check_v1.md: the sources formed from the complex64 T40 columns against the stored shell totals (relative) and the columns' orthonormality defect; every quantity read from a complex64 slice carries this floor"}
READOUT_MODULES = {"interior": "interior_continuation_v1", "core": "core_kinematics_v1", "page": "page_own_object_v1"}
T0 = time.time()

OPS = None
D13 = None
CS3 = None
RW = None
DM = None
SA = None
torch = None
SETUP = {}

B_SHA = {40: "24c357b0a7ba", 42: "1b0ffa5b3a65", 44: "b6ff14d57f8a", 46: "5b9a20814b0b", 48: "7c374e7deffb", 50: "7908df087da8", 52: "a003c7635bab",
         54: "cf96189fec06", 56: "d495b84f65b5"}
B2_SHA = {12: "28384caf8d7d", 16: "bb5f1dd93bce", 20: "0c011b736195", 24: "1bab0ca1cb40", 28: "0beb513e9cb4", 36: "a43c46d76c09", 40: "38f38047bb30",
          42: "a627c1ef9c5e", 44: "d5af76f1897b", 46: "33daef815753", 48: "eafce41cc2d3", 50: "309c85b1014e", 52: "8baef0a660f8", 54: "4b5e36ba67f8",
          56: "aea9e5006153", 58: "7d01f333efea", 60: "8629619768ec", 62: "0988fd57650e", 64: "13285ef62151", 66: "fc4b0802125d", 68: "f81b9db6de44",
          70: "9b1733117d39", 72: "7fa808dd13b1", 74: "470e180cf82a", 76: "de6c0571cae9", 78: "b7344df14d2e", 80: "62a340100c53", 82: "7ac46bde1e7f",
          84: "c538a5a9b39d", 86: "710e4d2d998c", 88: "1bbe07797b76", 90: "eb146c14e9d7", 92: "52ebcde77c9e", 94: "b423195891c8", 96: "4c8eb67fe0b4",
          98: "af16b95d41a8", 100: "7bc1244564bb"}
PROD_SHA_LOCAL = {12: "c125002d1ca9", 15: "f8ccddfa853b", 16: "fff1aafbf1fc", 20: "ef852664954b", 24: "2c7a2c4deffb", 25: "72b3f04c01e4",
                  28: "d4a43165b564", 30: "9d247ead4887", 35: "eccbc7bedad7", 40: "b18b83d67130"}


class Logger:
    """prints with the elapsed seconds and appends the same line to a log file (no dates anywhere)."""

    def __init__(self, path=None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def __call__(self, msg):
        line = "[%9.1f s] %s" % (time.time() - T0, msg)
        print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")


def jdefault(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch is not None and isinstance(o, torch.Tensor):
        return o.detach().cpu().numpy().tolist()
    return str(o)


def fx(x, fmt):
    """a formatted number or '-' for None."""
    if x is None:
        return "-"
    try:
        return fmt % x
    except (TypeError, ValueError):
        return str(x)


# -------------------------------------------------------------------------------------------------------------- the inventory
def slice_name(tag, t):
    return "orbitals_%s_T%07.2f.npz" % (tag, float(t))


def _entry(H, t, dtype, box_paths, hf_path=None, sha=None, sha_source=None, note=None, records_json=None, fields_npz=None):
    e = {"history": H["key"], "history_label": H["label"], "tag": H["tag"], "T_over_rs": float(t), "T": float(t) * H["rs"], "dtype": dtype,
         "nsites": H["nsites"], "expected_size": SIZES[(H["nsites"], dtype)], "box_paths": list(box_paths), "hf_path": hf_path, "sha256_prefix": sha,
         "sha_source": sha_source, "records_json": records_json or H["records_json"], "fields_npz": fields_npz or H["fields_npz"], "engine": H["engine"],
         "fields_module": H["fields_module"], "note": note}
    e["name"] = os.path.basename(box_paths[0]) if box_paths else os.path.basename(hf_path)
    return e


def inventory(D=BOX_D, E=BOX_E):
    """every stored slice of the inventory with its box paths (first the preferred copy), its dataset path and sha256 prefix, and the
    records json and fields npz of its segment."""
    H = {}
    # A, the production (10 slices on /data; the second copy under pod_sync; E/H100b/states is a symlink to D/engine/H100b/states)
    tag = "c19_rs64_A_R2_tau8_ell4_avg_r11"
    base = os.path.join(E, "H100b", "C3prod19")
    h = {"key": "prod", "label": "A, production (r_s/a 64, tau 8, sigma/a 4; 2048 sites, chain end 32 r_s)", "tag": tag, "nsites": 2048, "rs": 64.0,
         "sigma_over_a": 4.0,
         "engine": "engine_collapse_v19 (records 11.125 to 40, engine_fields_v14; every slice native v19); records 0 to 11 the replayed parent c17's (engine_collapse_v17, engine_fields_v12)",
         "fields_module": "engine_fields_v14", "records_json": os.path.join(base, "collapse_%s.json" % tag),
         "fields_npz": os.path.join(base, "collapse_%s_fields.npz" % tag), "data_npz": os.path.join(E, "COLL11", "COLLAPSE_records_v1.1_c19_rs64_data.npz"),
         "slices": []}
    dirs = [os.path.join(D, "engine", "H100b", "states", tag), os.path.join(D, "pod_sync", "pod1_engine", "states", tag), os.path.join(E, "H100b", "states", tag)]
    for t, dt in ((12, "c128"), (15, "c64"), (16, "c128"), (20, "c128"), (24, "c128"), (25, "c64"), (28, "c128"), (30, "c64"), (35, "c64"), (40, "c64")):
        h["slices"].append(_entry(h, t, dt, [os.path.join(d, slice_name(tag, t)) for d in dirs], sha=PROD_SHA_LOCAL[t],
                                  sha_source="the LOCAL copy's sha256 (results/LARGE_FILES.tsv); --verify-sha hashes the box copy against it"))
    H["prod"] = h
    # B, the continuation ext100 (40 to 56 dataset-only; 58 to 94 on /data; 96, 98 on /; 100 Brerun98's)
    tag = "c19_rs64_A_R2_tau8_ell4_avg_ext100"
    h = {"key": "B", "label": "A, continuation B ext100 (2048 sites, chain end 32 r_s; 28.125 to 100 by run_collapse_v19_from_slice_v1 on the box GPU 2)",
         "tag": tag, "nsites": 2048, "rs": 64.0, "sigma_over_a": 4.0,
         "engine": "engine_collapse_v19 (engine_fields_v14) through run_collapse_v19_from_slice_v1 from the production's complex128 T28 slice; records 0 to 28 the production's",
         "fields_module": "engine_fields_v14", "records_json": os.path.join(E, "C3ext19", "collapse_%s.json" % tag),
         "fields_npz": os.path.join(E, "C3ext19", "collapse_%s_fields.npz" % tag), "data_npz": None, "slices": []}
    dD = os.path.join(D, "engine", "states", tag)
    dE = os.path.join(E, "states", tag)
    r98 = "c19_rs64_A_R2_tau8_ell4_avg_ext100_r98"
    for t in range(40, 101, 2):
        if t <= 56:
            h["slices"].append(_entry(h, t, "c128", [os.path.join(dD, slice_name(tag, t)), os.path.join(dE, slice_name(tag, t))],
                                      hf_path="solve/B_ext100_slices/" + slice_name(tag, t), sha=B_SHA[t], sha_source="the dataset's LFS sha256",
                                      note="dataset-only (the box copy was deleted after the upload's size match)"))
        elif t <= 94:
            h["slices"].append(_entry(h, t, "c128", [os.path.join(dD, slice_name(tag, t)), os.path.join(dE, slice_name(tag, t))], note="a single copy, on /data"))
        elif t <= 98:
            h["slices"].append(_entry(h, t, "c128", [os.path.join(dE, slice_name(tag, t)), os.path.join(dD, slice_name(tag, t))],
                                      note="a single copy, a regular file on the box's / (96 percent full)"))
        else:
            h["slices"].append(_entry(h, t, "c128", [os.path.join(D, "Brerun98", "states", slice_name(r98, 100)), os.path.join(dE, slice_name(tag, 100))],
                                      note="Brerun98's slice (B wrote none at 100: its 60 GB disk guard); its records json holds B's records to 98 and the rerun's 98.125 to 100 (within 4.7e-10 in E/E0 of B's)",
                                      records_json=os.path.join(D, "Brerun98", "out", "collapse_%s.json" % r98),
                                      fields_npz=os.path.join(D, "Brerun98", "out", "collapse_%s_fields.npz" % r98)))
    H["B"] = h
    # B2, the doubled chain (37 slices; all in the dataset; T100 also on the box)
    tag = "c19_rs64_A_R2_tau8_ell4_avg_B2n4096"
    h = {"key": "B2", "label": "A, B2 (the doubled chain: 4096 sites, chain end 64 r_s; 0 to 100 on pod 4)", "tag": tag, "nsites": 4096, "rs": 64.0, "sigma_over_a": 4.0,
         "engine": "engine_collapse_v19 (engine_fields_v14) from T = 0 on pod 4; restarted from its own T50 slice by run_collapse_v19_from_slice_v1 after the 51.875 crash (slices 52 to 100 the restart's)",
         "fields_module": "engine_fields_v14", "records_json": os.path.join(D, "B2_final", "collapse_%s.json" % tag),
         "fields_npz": os.path.join(D, "B2_final", "collapse_%s_fields.npz" % tag), "data_npz": None, "slices": []}
    for t in [12, 16, 20, 24, 28, 36] + list(range(40, 101, 2)):
        h["slices"].append(_entry(h, t, "c128", [os.path.join(D, "B2_final", "states", slice_name(tag, t))], hf_path="solve/B2_slices/" + slice_name(tag, t),
                                  sha=B2_SHA[t], sha_source="the dataset's LFS sha256", note="on the box (D/B2_final/states) and in the dataset" if t == 100 else "dataset-only (10.7 GB)"))
    H["B2"] = h
    # the sigma/a 8 member (8 complex64 slices, the only copy, on /data)
    tag = "c27_rs64_A_R2_tau8_ell8_prep4_avg"
    base = os.path.join(E, "H100b", "C3lad27")
    h = {"key": "s8", "label": "the sigma/a 8 member (lad_s8_now.json; r_s/a 64, tau 8, ell 8, prep_ell 4; 2048 sites, chain end 32 r_s)", "tag": tag, "nsites": 2048,
         "rs": 64.0, "sigma_over_a": 8.0,
         "engine": "script engine_collapse_v27.py, records labelled engine_collapse_v25 (engine_fields_v16, solve_constraints_ivp); every slice complex64",
         "fields_module": "engine_fields_v16", "records_json": os.path.join(base, "collapse_%s.json" % tag),
         "fields_npz": os.path.join(base, "collapse_%s_fields.npz" % tag), "data_npz": os.path.join(E, "COLL", "COLLAPSE_records_v1_%s_data.npz" % tag), "slices": []}
    dirs = [os.path.join(D, "engine", "H100b", "states", tag), os.path.join(E, "H100b", "states", tag)]
    for t in (12, 13, 14, 16, 20, 25, 30, 40):
        h["slices"].append(_entry(h, t, "c64", [os.path.join(d, slice_name(tag, t)) for d in dirs], note="the only copy (pod 1's removed after an md5 match)"))
    H["s8"] = h
    return H


def select_entries(H, clocks):
    if str(clocks).strip().lower() == "all":
        return list(H["slices"])
    want = [float(x) for x in str(clocks).split(",") if x.strip()]
    return [e for e in H["slices"] if any(abs(e["T_over_rs"] - w) < 1e-6 for w in want)]


def locate(entry):
    """the first existing box copy (symlinks followed) or None."""
    for p in entry["box_paths"]:
        if os.path.isfile(p):
            return p
    return None


def verify_size(path, entry):
    sz = os.path.getsize(path)
    if sz != entry["expected_size"]:
        raise RuntimeError("%s: %d bytes, the inventory expects %d" % (path, sz, entry["expected_size"]))
    return sz


def sha256_file(path, chunk=1 << 26):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ the dataset
def _token(token_env):
    for v in (token_env, "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_API_TOKEN"):
        if v:
            t = os.environ.get(v)
            if t:
                return t
    return None


def download_slice(entry, scratch, sub, token_env, min_free_gb, log):
    """the dataset-only slice into sub (under the scratch folder on /data): the free space checked, the sha256 against the dataset's LFS
    metadata and the inventory's prefix, the size against the inventory. The token stays inside this function."""
    os.makedirs(sub, exist_ok=True)
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(scratch, "hub_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import HfApi, hf_hub_download
    token = _token(token_env)
    log("dataset %s: fetching %s (token %s)" % (HF_REPO, entry["hf_path"], "present in the environment" if token else "ABSENT: the download is tried without one"))
    expected = None
    size = int(entry["expected_size"])
    try:
        infos = HfApi(token=token).get_paths_info(HF_REPO, [entry["hf_path"]], repo_type="dataset")
        for inf in infos:
            lfs = getattr(inf, "lfs", None)
            if lfs:
                expected = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                sz = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
                if sz:
                    size = int(sz)
    except Exception as ex:  # noqa: BLE001
        log("the dataset's metadata could not be read (%s): the inventory's sha256 prefix is the only hash check" % type(ex).__name__)
    free = shutil.disk_usage(scratch).free
    if free < size + min_free_gb * 1e9:
        raise RuntimeError("free space on %s is %.1f GB; the slice needs %.1f GB plus the %.0f GB guard: stopped" % (scratch, free / 1e9, size / 1e9, min_free_gb))
    t0 = time.time()
    local = hf_hub_download(repo_id=HF_REPO, filename=entry["hf_path"], repo_type="dataset", local_dir=sub, token=token)
    dt = time.time() - t0
    sz = os.path.getsize(local)
    log("downloaded %s: %.2f GB in %.0f s (%.1f MB/s)" % (os.path.basename(local), sz / 1e9, dt, sz / 1e6 / max(dt, 1e-9)))
    if sz != entry["expected_size"]:
        shutil.rmtree(sub, ignore_errors=True)
        raise RuntimeError("downloaded size %d differs from the inventory's %d: deleted" % (sz, entry["expected_size"]))
    t1 = time.time()
    sha = sha256_file(local)
    ok_full = expected is None or sha == expected
    ok_prefix = (not entry.get("sha256_prefix")) or sha.startswith(entry["sha256_prefix"])
    log("sha256 %s (%.0f s): %s" % (sha, time.time() - t1, "matches the dataset's LFS hash" if (expected is not None and ok_full) else "no LFS hash to compare") +
        ("; matches the inventory's prefix" if ok_prefix else "; DOES NOT match the inventory's prefix %s" % entry.get("sha256_prefix")))
    if not (ok_full and ok_prefix):
        shutil.rmtree(sub, ignore_errors=True)
        raise RuntimeError("sha256 mismatch for %s (computed %s, LFS %s, inventory prefix %s): deleted" % (entry["hf_path"], sha, expected, entry.get("sha256_prefix")))
    return local


class fetch_slice:
    """with fetch_slice(entry, a, log) as path: a box copy where one exists (size checked; --verify-sha hashes it), else the dataset's
    download into the scratch folder, deleted on exit unless --keep-downloads."""

    def __init__(self, entry, a, log):
        self.entry = entry
        self.a = a
        self.log = log
        self.downloaded = False
        self.path = None
        self.sub = None

    def __enter__(self):
        p = locate(self.entry)
        if p is not None:
            verify_size(p, self.entry)
            if getattr(self.a, "verify_sha", False) and self.entry.get("sha256_prefix"):
                sha = sha256_file(p)
                if not sha.startswith(self.entry["sha256_prefix"]):
                    raise RuntimeError("%s: sha256 %s does not start with the inventory's %s (%s)" % (p, sha, self.entry["sha256_prefix"], self.entry.get("sha_source")))
                self.log("sha256 of %s verified against the inventory's prefix" % p)
            self.log("slice on the box: %s (%d bytes, size ok)" % (p, self.entry["expected_size"]))
            self.path = p
            return p
        if not self.entry.get("hf_path"):
            raise FileNotFoundError("no box copy of %s (looked at %s) and the slice is not in the dataset" % (self.entry["name"], self.entry["box_paths"]))
        self.sub = os.path.join(self.a.scratch, "dl_" + self.entry["name"][:-4])
        self.path = download_slice(self.entry, self.a.scratch, self.sub, self.a.token_env, self.a.min_free_gb, self.log)
        self.downloaded = True
        return self.path

    def __exit__(self, et, ev, tb):
        if self.downloaded and not getattr(self.a, "keep_downloads", False):
            gc.collect()
            shutil.rmtree(self.sub, ignore_errors=True)
            self.log("deleted the downloaded slice %s (dataset-only; --keep-downloads keeps it)" % self.entry["name"])
        return False


# ------------------------------------------------------------------------------------------------------------- the reader
_LFH = struct.Struct("<4s2B4HL2L2H")


def zip_member_offsets(path):
    """per member of the uncompressed npz: the byte offset of the array data inside the zip, the shape, dtype and order from the npy
    header (zip64 members included: the sizes are not needed, the npy header carries the shape)."""
    out = {}
    with zipfile.ZipFile(path) as zf, open(path, "rb") as f:
        for info in zf.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("%s: member %s is compressed (type %d); the slices are stored uncompressed" % (path, info.filename, info.compress_type))
            f.seek(info.header_offset)
            hdr = _LFH.unpack(f.read(_LFH.size))
            if hdr[0] != b"PK\x03\x04":
                raise ValueError("%s: no local file header for %s at offset %d" % (path, info.filename, info.header_offset))
            f.seek(info.header_offset + _LFH.size + hdr[10] + hdr[11])
            version = npf.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = npf.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = npf.read_array_header_2_0(f)
            else:
                shape, fortran, dtype = npf._read_array_header(f, version)
            key = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            out[key] = {"offset": int(f.tell()), "shape": tuple(int(s) for s in shape), "dtype": np.dtype(dtype), "fortran": bool(fortran), "bytes": int(info.file_size)}
    return out


class SliceReader:
    """one stored slice: the scalars and site arrays loaded, every column array memory-mapped on demand; per channel Phi_k with its
    multiplicity, the stored dust columns, R2, e_k and the unpacked mask."""

    def __init__(self, path, entry=None, log=None):
        self.path = os.path.abspath(path)
        self.entry = dict(entry or {})
        self.log = log or (lambda s: print(s, flush=True))
        self.members = zip_member_offsets(self.path)
        self._cache = {}
        self.T = float(self.small("T"))
        self.rs = float(self.small("rs"))
        self.T_over_rs = self.T / self.rs
        self.R = np.asarray(self.small("R"), float).reshape(-1)
        self.nsites = len(self.R)
        self.n2 = 2 * self.nsites
        kap = np.atleast_1d(self.small("kappas"))
        self.kappas = [int(k) for k in kap]
        self.weights = {int(k): float(w) for k, w in zip(kap, np.atleast_1d(self.small("weights")))}
        self.has_applied = "N_applied" in self.members and "v_applied" in self.members
        self.N_applied = np.asarray(self.small("N_applied"), float).reshape(-1) if self.has_applied else None
        self.v_applied = np.asarray(self.small("v_applied"), float).reshape(-1) if self.has_applied else None
        self.column_dtype = self.members["sea_%d" % self.kappas[0]]["dtype"]
        self.dtype_str = str(self.small("dtype")) if "dtype" in self.members else str(self.column_dtype)
        self.complex64 = self.column_dtype == np.dtype(np.complex64)
        self.n_sea = {k: int(self.members["sea_%d" % k]["shape"][1]) for k in self.kappas}
        self.n_dust = {k: int(self.members["dust_%d" % k]["shape"][1]) if ("dust_%d" % k) in self.members else 0 for k in self.kappas}
        self.floor = dict(C64_FLOOR, applies=bool(self.complex64))
        if self.complex64:
            self.log("complex64 slice %s: the source error %.1e to %.1e relative and the orthonormality defect %.1e are the floor of every quantity read from its columns (a complex128 slice: %.1e)" % (
                os.path.basename(self.path), C64_FLOOR["source_rel_range"][0], C64_FLOOR["source_rel_range"][1], C64_FLOOR["orthonormality_defect"], C64_FLOOR["complex128_reference"]))

    def small(self, key):
        if key not in self._cache:
            with np.load(self.path, allow_pickle=False) as Z:
                self._cache[key] = Z[key]
        return self._cache[key]

    def mmap(self, key):
        mb = self.members[key]
        return np.memmap(self.path, dtype=mb["dtype"], mode="r", offset=mb["offset"], shape=mb["shape"], order="F" if mb["fortran"] else "C")

    def array(self, key):
        """any member: memory-mapped where it is a numeric array of rank >= 1, loaded otherwise."""
        mb = self.members[key]
        if len(mb["shape"]) >= 1 and mb["dtype"].kind in "cfiub" and int(np.prod(mb["shape"])) > 0:
            return self.mmap(key)
        return self.small(key)

    def e(self, kappa):
        return np.asarray(self.small("e_%d" % kappa), float).reshape(-1) if ("e_%d" % kappa) in self.members else None

    def mask(self, kappa):
        if ("mask_%d" % kappa) not in self.members:
            return None
        return np.unpackbits(np.asarray(self.small("mask_%d" % kappa), np.uint8))[: self.n2].astype(bool)

    def dust(self, kappa):
        """the stored (tagged) dust columns in complex128 and their occupations f."""
        f = np.asarray(self.small("f_%d" % kappa), float).reshape(-1)
        if self.n_dust[kappa] == 0:
            return np.zeros((self.n2, 0), np.complex128), f
        return np.asarray(self.mmap("dust_%d" % kappa), dtype=np.complex128), f

    def sea(self, kappa):
        return np.asarray(self.mmap("sea_%d" % kappa), dtype=np.complex128)

    def R2(self, kappa):
        return np.asarray(self.mmap("R2_%d" % kappa), dtype=np.complex128)

    def channel(self, kappa, orthonormalize=True):
        """Phi_k = [sea_k | dust_k] in complex128 with f, the multiplicity w = 16 kappa, e_k, the mask, the stored orthonormality defect
        and the Loewdin-orthonormalized columns (engine_massive_channel_D_v2.loewdin) when orthonormalize is set."""
        n_s, n_d = self.n_sea[kappa], self.n_dust[kappa]
        Phi = np.empty((self.n2, n_s + n_d), dtype=np.complex128)
        Phi[:, :n_s] = self.mmap("sea_%d" % kappa)
        f_d = np.asarray(self.small("f_%d" % kappa), float).reshape(-1)
        if n_d:
            Phi[:, n_s:] = self.mmap("dust_%d" % kappa)
        f = np.concatenate([np.ones(n_s), f_d])
        out = {"kappa": int(kappa), "weight": self.weights[int(kappa)], "n_sea": n_s, "n_dust": n_d, "f": f, "f_dust": f_d, "complex64": self.complex64,
               "e": self.e(kappa), "mask": self.mask(kappa),
               "dust_sea_overlap_max": float(np.max(np.abs(Phi[:, :n_s].conj().T @ Phi[:, n_s:]))) if n_d else 0.0}
        if orthonormalize:
            Phi_o, e0, e1, route = orthonormalize_columns(Phi)
            del Phi
            out.update({"Phi": Phi_o, "stored_orthonormality": e0, "loewdin_orthonormality": e1, "orthonormalization": route})
        else:
            G = Phi.conj().T @ Phi
            out.update({"Phi": Phi, "stored_orthonormality": float(np.max(np.abs(G - np.eye(G.shape[0])))), "loewdin_orthonormality": None, "orthonormalization": "none (the stored columns)"})
        return out

    def close(self):
        self._cache.clear()
        gc.collect()


def orthonormalize_columns(X):
    """X G^(-1/2), G = X^dagger X: engine_massive_channel_D_v2.loewdin (torch on the CPU) where importable, else the same step in numpy;
    returns (X_o, the stored defect max|G - I|, the defect after, the route's name)."""
    if DM is not None and SA is not None:
        Xo, G, e0, e1 = DM.loewdin(np.asarray(X, np.complex128))
        return np.asarray(Xo, np.complex128), float(e0), float(e1), "engine_massive_channel_D_v2.loewdin (X G^(-1/2), torch on the CPU)"
    X = np.asarray(X, np.complex128)
    G = X.conj().T @ X
    G = 0.5 * (G + G.conj().T)
    w, V = eigh(G)
    Xo = X @ ((V * (1.0 / np.sqrt(np.clip(w, 1e-300, None)))[None, :]) @ V.conj().T)
    e0 = float(np.max(np.abs(G - np.eye(G.shape[0])))) if G.shape[0] else 0.0
    G1 = Xo.conj().T @ Xo
    e1 = float(np.max(np.abs(G1 - np.eye(G1.shape[0])))) if G1.shape[0] else 0.0
    return Xo, e0, e1, "local Loewdin step X (X^dag X)^(-1/2) in numpy (engine_massive_channel_D_v2 not importable)"


# ------------------------------------------------------------------------------------------------------------ the history
class HistoryContext:
    """the history's records json (per segment: B's T100 slice reads Brerun98's), its fields npz and its field module."""

    def __init__(self, H, a, log):
        self.H = H
        self.a = a
        self.log = log
        self._J = {}
        self._Z = {}
        self._F = {}
        J = self.records({"records_json": H["records_json"]})
        self.args = J["args"]
        self.m = float(self.args["m"])
        self.g = float(self.args["g"])
        self.rs = float(self.args["rs"])
        self.ell = float(self.args["ell"])
        self.nsites = int(self.args["nsites"])
        self.tau = float(self.args.get("tau", 8))
        self.E0 = float(J.get("E0", J.get("head", {}).get("E0", 0.0)) or 0.0) if isinstance(J.get("E0", 0.0), (int, float)) else 0.0

    def records(self, entry):
        p = entry.get("records_json") or self.H["records_json"]
        if p not in self._J:
            with open(p) as f:
                self._J[p] = json.load(f)
            self.log("records json %s: %d records" % (p, len(self._J[p].get("records", []))))
        return self._J[p]

    def record_at(self, entry, T):
        for r in self.records(entry)["records"]:
            if r.get("E") is not None and abs(float(r["T"]) - float(T)) < 1e-6:
                return r
        return {}

    def record_prev_next(self, entry, T):
        rs_ = [r for r in self.records(entry)["records"] if r.get("E") is not None]
        rs_.sort(key=lambda r: float(r["T"]))
        prev = [r for r in rs_ if float(r["T"]) < float(T) - 1e-6]
        nxt = [r for r in rs_ if float(r["T"]) > float(T) + 1e-6]
        return (prev[-1] if prev else None), (nxt[0] if nxt else None)

    def fields_file(self, entry):
        p = entry.get("fields_npz") or self.H["fields_npz"]
        if p not in self._Z:
            self._Z[p] = np.load(p, allow_pickle=True)
            self.log("fields npz %s: keys %s" % (p, sorted(self._Z[p].files)))
        return p, self._Z[p]

    def fields_key(self, entry, key):
        p, Z = self.fields_file(entry)
        if (p, key) not in self._F:
            self._F[(p, key)] = np.asarray(Z[key], float) if key in Z.files else None
        return self._F[(p, key)]

    def fields_index(self, entry, T):
        Tz = self.fields_key(entry, "T")
        k = int(np.argmin(np.abs(Tz - float(T))))
        if abs(float(Tz[k]) - float(T)) > 1e-6:
            self.log("WARNING: the fields npz has no record at T %.3f (nearest %.3f)" % (T, Tz[k]))
        return k

    def field_module(self):
        return importlib.import_module(self.H["fields_module"])

    def formation(self, entry):
        """the first record with a trapping zero: its clock and outer root (the formation's light for the first-return estimate)."""
        for r in sorted(self.records(entry)["records"], key=lambda r: float(r["T"])):
            if r.get("E") is None:
                continue
            z = r.get("trapping_zeros")
            if z:
                return {"T": float(r["T"]), "T_over_rs": float(r["T"]) / self.rs, "r_root": float(max(z))}
        return None

    def engine_label(self, entry, rec):
        own = rec.get("engine") or rec.get("record_label")
        return "%s; the record's own label: %s" % (entry["engine"], own if own else "absent")


# ----------------------------------------------------------------------------------------------------------- the imports
def setup(engine_code=None, page_code=None, threads=None, log=None):
    """sys.path and the imports by name; SETUP records what was found."""
    global OPS, D13, CS3, RW, DM, SA, torch, SETUP
    log = log or (lambda s: print(s, flush=True))
    dirs = [HERE]
    for d in (engine_code, page_code, os.environ.get("SLICES144_ENGINE_CODE"), os.environ.get("SLICES144_PAGE_CODE"), os.environ.get("PAGE_CODE_V6")):
        if d:
            dirs += [x for x in str(d).split(os.pathsep) if x]
    for d in dirs:
        d = os.path.abspath(d)
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    import torch as _torch
    torch = _torch
    if threads:
        torch.set_num_threads(int(threads))
    import page_ops_v6 as _ops
    OPS = _ops
    SETUP = {"page_ops_v6": getattr(_ops, "__file__", None), "torch_threads": torch.get_num_threads(), "omp_threads": os.environ.get("OMP_NUM_THREADS")}
    for names, key in ((("page_d13_readout_v1b", "page_d13_readout_v1"), "D13"), (("engine_core_slices_v3",), "CS3"), (("engine_rows_v2",), "RW"),
                       (("engine_massive_channel_D_v2",), "DM")):
        got = None
        err = []
        for name in names:
            try:
                got = importlib.import_module(name)
                SETUP[key] = "%s (%s)" % (name, getattr(got, "__file__", None))
                break
            except Exception as ex:  # noqa: BLE001
                err.append("%s: %s: %s" % (name, type(ex).__name__, ex))
        globals()[key] = got
        if got is None:
            SETUP[key] = "not importable (%s)" % "; ".join(err)
    if DM is not None:
        try:
            import engine_single_object_A_v2 as _sa
            _sa.set_device(-1)
            SA = _sa
            SETUP["engine_single_object_A_v2"] = "device set to the CPU"
        except Exception as ex:  # noqa: BLE001
            SA = None
            SETUP["engine_single_object_A_v2"] = "not usable (%s: %s): local fallbacks for the Loewdin step and the regions" % (type(ex).__name__, ex)
    return SETUP


# ---------------------------------------------------------------------------------------------------- shared arithmetic
def operators(R, m, kappa, N, v):
    """(h_0, Q, h_m) as host CSR matrices by page_ops_v6: h_m = (1/2){N, h_0} + (1/2){v, Q} on the stored chain (R the slice's radii)."""
    h0, Q = OPS.chain_sparse(np.asarray(R, float), float(m), int(kappa))
    H = OPS.clocked_csr(h0, Q, np.asarray(N, float), np.asarray(v, float))
    return h0.tocsr(), Q.tocsr(), H.tocsr()


def eigh(A):
    """the Hermitian eigendecomposition (torch.linalg.eigh on the CPU; numpy where torch is absent); (w ascending, V)."""
    A = np.asarray(A, dtype=np.complex128)
    A = 0.5 * (A + A.conj().T)
    if torch is not None:
        w, V = torch.linalg.eigh(torch.as_tensor(A))
        return np.asarray(w.numpy(), float), np.asarray(V.numpy(), np.complex128)
    w, V = np.linalg.eigh(A)
    return np.asarray(w, float), np.asarray(V, np.complex128)


def eigvalsh(A):
    A = np.asarray(A, dtype=np.complex128)
    A = 0.5 * (A + A.conj().T)
    if torch is not None:
        return np.asarray(torch.linalg.eigvalsh(torch.as_tensor(A)).numpy(), float)
    return np.asarray(np.linalg.eigvalsh(A), float)


def frac_radius(R, cum, q):
    """engine_core_slices_v3.frac_radius: the radius at which the cumulative fraction first reaches q, linear between sites."""
    R = np.asarray(R, float)
    cum = np.asarray(cum, float)
    if CS3 is not None:
        return float(CS3.frac_radius(R, cum, q))
    i = int(np.searchsorted(cum, q))
    if i <= 0:
        return float(R[0])
    if i >= len(R):
        return float(R[-1])
    c0, c1 = cum[i - 1], cum[i]
    return float(R[i - 1] + (q - c0) / (c1 - c0) * (R[i] - R[i - 1])) if c1 > c0 else float(R[i])


def call_dense(fun, r):
    """a dense field callable (the field module's N_dense, v_dense) at a scalar or an array of radii."""
    r = np.asarray(r, float)
    scalar = r.ndim == 0
    rr = np.atleast_1d(r)
    try:
        out = np.asarray(fun(rr), float).reshape(-1)
        if out.shape[0] != rr.shape[0]:
            raise ValueError("shape")
    except Exception:  # noqa: BLE001
        out = np.array([float(np.asarray(fun(float(x)), float).reshape(-1)[0]) for x in rr])
    return float(out[0]) if scalar else out


def richardson_d1(f, x, h):
    """f'(x) by the central difference at h and h/2, Richardson-combined."""
    d = lambda hh: (f(x + hh) - f(x - hh)) / (2.0 * hh)  # noqa: E731
    return float((4.0 * d(0.5 * h) - d(h)) / 3.0)


def cuts_from_arg(s):
    return [("%g" % float(x), float(x)) for x in str(s).split(",") if str(x).strip()]


def interp_zeros(R, f):
    """the zeros of the piecewise-linear interpolant of f on the sites (sign changes and exact zeros) with the local slope."""
    R = np.asarray(R, float)
    f = np.asarray(f, float)
    out = []
    for i in range(len(R) - 1):
        a_, b_ = f[i], f[i + 1]
        if a_ == 0.0 or a_ * b_ < 0.0:
            r = R[i] if a_ == 0.0 else R[i] + a_ / (a_ - b_) * (R[i + 1] - R[i])
            out.append({"R": float(r), "slope": float((b_ - a_) / (R[i + 1] - R[i]))})
    return out


def out_paths(a, readout, entry):
    d = os.path.join(a.out, readout)
    os.makedirs(d, exist_ok=True)
    base = os.path.join(d, "%s_%s_T%07.2f" % (readout, entry["history"], entry["T_over_rs"]))
    return base + ".json", base + ".npz"


def save_outputs(pj, pz, row, arrays):
    tmp = pj + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f, default=jdefault, indent=1)
    os.replace(tmp, pj)
    if arrays:
        tmpz = pz + ".tmp.npz"
        np.savez(tmpz, **{k: np.asarray(v) for k, v in arrays.items()})
        os.replace(tmpz, pz)


def load_rows(a, readout, hists):
    d = os.path.join(a.out, readout)
    rows = []
    if not os.path.isdir(d):
        return rows
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json") and fn.startswith(readout + "_"):
            try:
                with open(os.path.join(d, fn)) as f:
                    r = json.load(f)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(r, dict) and r.get("history") in hists and "T_over_rs" in r:
                rows.append(r)
    rows.sort(key=lambda r: (r["history"], r["T_over_rs"]))
    return rows


def slice_header(rd, ctx, entry, rec, readout, title, code):
    return {"readout": readout, "title": title, "code": code, "history": entry["history"], "history_label": ctx.H["label"], "tag": entry["tag"],
            "slice": os.path.basename(rd.path), "T": rd.T, "T_over_rs": rd.T_over_rs, "rs": rd.rs, "nsites": rd.nsites, "chain_end_over_rs": float(rd.R[-1] / rd.rs),
            "dtype": rd.dtype_str, "complex64": bool(rd.complex64), "floor": rd.floor, "engine": ctx.engine_label(entry, rec), "record_found": bool(rec),
            "record_converged": rec.get("converged"), "sigma_over_a": ctx.ell, "m": ctx.m, "g": ctx.g, "tau": ctx.tau, "note": entry.get("note"), "setup": SETUP}


def wait_for_load(max_load, log):
    """waits while the 1-minute load average exceeds max_load percent of the CPUs (the box runs GPU jobs beside this)."""
    if max_load is None:
        return
    ncpu = os.cpu_count() or 1
    while True:
        la = os.getloadavg()[0]
        pct = 100.0 * la / ncpu
        if pct < max_load:
            return
        log("the box load %.1f on %d CPUs is %.0f percent (guard %.0f): waiting 60 s" % (la, ncpu, pct, max_load))
        time.sleep(60)


def _load_arg(s):
    return None if str(s).strip().lower() in ("none", "off", "") else float(s)


def add_common_arguments(ap):
    g = ap.add_argument_group("slices144 (shared by the 3 readouts)")
    g.add_argument("--box-D", dest="box_D", default=BOX_D)
    g.add_argument("--box-E", dest="box_E", default=BOX_E)
    g.add_argument("--engine-code", dest="engine_code", default=ENGINE_CODE, help="the directory of engine_core_slices_v3, engine_massive_channel_D_v2, engine_rows_v2 and the field modules")
    g.add_argument("--page-code", dest="page_code", default=PAGE_CODE, help="the directory of page_ops_v6.py and page_d13_readout_v1b.py")
    g.add_argument("--threads", type=int, default=None, help="BLAS and torch threads (16 where not given)")
    g.add_argument("--history", default="prod", help="comma list of prod, B, B2, s8 (or all)")
    g.add_argument("--clocks", default="all", help="all or a comma list of T/r_s")
    g.add_argument("--readouts", default="interior,core,page")
    g.add_argument("--out", default=os.path.join(BOX_D, "slices144", "out"))
    g.add_argument("--scratch", default=os.path.join(BOX_D, "slices144", "scratch"), help="the download folder for dataset-only slices (on /data)")
    g.add_argument("--token-env", dest="token_env", default="HF_TOKEN")
    g.add_argument("--min-free-gb", dest="min_free_gb", type=float, default=60.0, help="free space the scratch disk must keep beyond the slice's size")
    g.add_argument("--keep-downloads", dest="keep_downloads", action="store_true")
    g.add_argument("--verify-sha", dest="verify_sha", action="store_true", help="hash the box copies against the inventory's sha256 prefixes (slow)")
    g.add_argument("--force", action="store_true", help="recompute a slice whose json exists")
    g.add_argument("--max-load", dest="max_load", type=_load_arg, default=90.0, help="wait while the 1-minute load average exceeds this percent of the CPUs (none: no wait)")
    g.add_argument("--stop-on-missing", dest="stop_on_missing", action="store_true")
    g.add_argument("--compose-only", dest="compose_only", action="store_true", help="only the tables from the jsons present")
    g.add_argument("--cuts-rs", dest="cuts_rs", default="16,24", help="the exterior cuts in r_s")
    g.add_argument("--loewdin", default="on", choices=["on", "off"], help="Loewdin-orthonormalize the occupied columns before the Gram and interior readouts")
    g.add_argument("--m-split-at", dest="m_split_at", type=float, default=1.0, help="the Killing split threshold in units of m")
    g.add_argument("--hfrac", type=float, default=32.0, help="the dense derivative step ell/hfrac (engine_core_slices_v3's)")
    g.add_argument("--collar-policy", dest="collar_policy", default="flag")
    g.add_argument("--support-rel", dest="support_rel", type=float, default=1e-14)
    g.add_argument("--check-eigs-kappa", dest="check_eigs_kappa", type=int, default=1, help="the channel whose h_m eigenvalues are checked against the stored e_k where a dense spectrum is not computed anyway (0: none)")


def check(a):
    INV = inventory(a.box_D, a.box_E)
    keys = list(INV) if a.history.strip().lower() == "all" else [h.strip() for h in a.history.split(",") if h.strip()]
    for hk in keys:
        H = INV[hk]
        print("## %s: %s" % (hk, H["label"]))
        print("   engine: %s" % H["engine"])
        for lab, p in (("records json", H["records_json"]), ("fields npz", H["fields_npz"]), ("records data npz", H["data_npz"])):
            print("   %s: %s (%s)" % (lab, p, "present" if (p and os.path.isfile(p)) else ("none listed" if not p else "ABSENT")))
        for e in H["slices"]:
            p = locate(e)
            if p:
                sz = os.path.getsize(p)
                st = "box %s (%s)" % (p, "size ok" if sz == e["expected_size"] else "SIZE %d differs from %d" % (sz, e["expected_size"]))
            elif e.get("hf_path"):
                st = "dataset-only: %s (sha256 prefix %s)" % (e["hf_path"], e["sha256_prefix"])
            else:
                st = "MISSING: no box copy at %s and not in the dataset" % e["box_paths"]
            extra = "" if e.get("records_json") == H["records_json"] else " [records: %s]" % e["records_json"]
            print("   T/r_s %6.2f %-5s %s%s" % (e["T_over_rs"], e["dtype"], st, extra))
        print()


def run(a):
    log = Logger(os.path.join(a.out, "slices144_driver.log"))
    setup(a.engine_code, a.page_code, a.threads, log)
    log("%s: setup %s" % (CODE, json.dumps(SETUP, default=str)))
    INV = inventory(a.box_D, a.box_E)
    hists = list(INV) if a.history.strip().lower() == "all" else [h.strip() for h in a.history.split(",") if h.strip()]
    mods = {}
    for r in [x.strip() for x in a.readouts.split(",") if x.strip()]:
        mods[r] = importlib.import_module(READOUT_MODULES[r])
    if not a.compose_only:
        for hk in hists:
            H = INV[hk]
            ctx = HistoryContext(H, a, log)
            entries = select_entries(H, a.clocks)
            log("history %s (%s): %d slices selected, readouts %s" % (hk, H["label"], len(entries), list(mods)))
            for e in entries:
                todo = [m for m in mods.values() if a.force or not m.is_done(e, a)]
                if not todo:
                    log("T/r_s %.3f: every readout's json is present (--force recomputes)" % e["T_over_rs"])
                    continue
                wait_for_load(a.max_load, log)
                try:
                    with fetch_slice(e, a, log) as path:
                        rd = SliceReader(path, e, log)
                        log("slice %s: T/r_s %.3f, %d sites, %s, kappas %s, dust columns %s" % (os.path.basename(path), rd.T_over_rs, rd.nsites, rd.dtype_str, rd.kappas, rd.n_dust))
                        for m in todo:
                            t1 = time.time()
                            m.process_slice(rd, ctx, e, a)
                            log("  %s done in %.0f s" % (m.READOUT, time.time() - t1))
                        rd.close()
                        del rd
                except FileNotFoundError as ex:
                    log("MISSING slice at T/r_s %.3f: %s" % (e["T_over_rs"], ex))
                    if a.stop_on_missing:
                        raise
    for m in mods.values():
        m.compose(a, hists)
        log("%s: tables composed for %s" % (m.READOUT, hists))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(ap)
    ap.add_argument("--check", action="store_true", help="list every slice of the inventory with its location; nothing computed")
    for name in READOUT_MODULES.values():
        try:
            sys.path.insert(0, HERE)
            importlib.import_module(name).add_arguments(ap)
        except Exception as ex:  # noqa: BLE001
            print("the readout module %s could not add its arguments (%s: %s)" % (name, type(ex).__name__, ex), flush=True)
    a = ap.parse_args(argv)
    if a.check:
        return check(a)
    run(a)


if __name__ == "__main__":
    main()
