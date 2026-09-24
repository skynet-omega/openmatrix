"""Standalone NumPy verifier for the portable owner-overlay capsule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def check_layout(a: dict[str, np.ndarray], meta: dict) -> None:
    n = len(a["caps"])
    rows, ptr, src = a["active_rows"], a["row_ptr"], a["sources"]
    e = len(src)
    need(meta["schema"] == "owner_overlay_portable_capsule_v1"
         and meta["selected_edges"] == e and meta["active_receiver_rows"] == len(rows),
         "Metadata/layout disagreement")
    need(rows.dtype == np.int32 and ptr.dtype == np.int64 and src.dtype == np.int32
         and rows.ndim == ptr.ndim == src.ndim == 1
         and ptr.shape == (len(rows) + 1,) and ptr[0] == 0 and ptr[-1] == e
         and np.all(np.diff(ptr) > 0) and np.all(np.diff(rows) > 0)
         and np.all((rows >= 0) & (rows < n)) and np.all((src >= 0) & (src < n)),
         "Invalid reduced CSR")
    need(a["visual_u8"].shape == (n,) and a["visual_u8"].dtype == np.uint8
         and np.all(a["visual_u8"] <= 1) and a["caps"].dtype == np.float64
         and np.isfinite(a["caps"]).all(), "Invalid cell metadata")
    for key in ("weight_before", "weight_after"):
        need(a[key].shape == (e,) and a[key].dtype == np.float64
             and np.isfinite(a[key]).all(), "Invalid " + key)
    for state in ("A", "B"):
        for suffix in ("before", "after"):
            key = "release_" + state + "_" + suffix
            need(a[key].shape == (n,) and a[key].dtype == np.float64
                 and np.isfinite(a[key]).all() and np.all(a[key] >= 0.),
                 "Invalid " + key)
        key = "expected_" + state
        need(a[key].shape == (3, len(rows)) and a[key].dtype == np.float64
             and np.isfinite(a[key]).all(), "Invalid " + key)
    need(a["scale"].shape == () and np.isfinite(a["scale"])
         and float(a["scale"]) > 0 and a["connected"].shape == (),
         "Invalid scalar configuration")


def reconstruct(a: dict[str, np.ndarray], state: str) -> np.ndarray:
    """Independent per-receiver reduction; never imports the builder."""
    rows, ptr, src = a["active_rows"], a["row_ptr"], a["sources"]
    before, after = a["release_" + state + "_before"], a["release_" + state + "_after"]
    w0, w1 = a["weight_before"], a["weight_after"]
    caps, visual = a["caps"], a["visual_u8"]
    scale, connected = float(a["scale"]), bool(a["connected"])
    out = np.zeros((3, len(rows)), dtype=np.float64)
    for k, row in enumerate(rows):
        v = bool(visual[row])
        for edge in range(int(ptr[k]), int(ptr[k + 1])):
            pre = int(src[edge])
            old = float(w0[edge] * before[pre])
            new = float(w1[edge] * after[pre])
            if v:
                out[1, k] += max(new * scale, 0.) - max(old * scale, 0.)
                out[2, k] += max(-new * scale, 0.) - max(-old * scale, 0.)
            elif connected or not bool(visual[pre]):
                out[0, k] += (new - old) * caps[pre]
    return out


def compare(observed: np.ndarray, expected: np.ndarray, tolerance: float) -> float:
    need(observed.shape == expected.shape and np.isfinite(observed).all()
         and np.isfinite(expected).all(), "Nonfinite/mismatched delta")
    error = float(np.max(np.abs(observed - expected)))
    need(error <= tolerance, "Sparse delta identity failed")
    return error


def verify(folder: Path, mutation_checks: bool = False) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / "MANIFEST.json").read_text())
    need(manifest["schema"] == "owner_overlay_capsule_manifest_v1", "Wrong manifest")
    for name, entry in manifest["files"].items():
        path = folder / name
        need(path.is_file() and sha(path) == entry["sha256"]
             and path.stat().st_size == entry["bytes"], "File hash/size changed: " + name)
    meta = json.loads((folder / "META.json").read_text())
    with np.load(folder / "capsule.npz", allow_pickle=False) as archive:
        a = {name: archive[name] for name in archive.files}
    check_layout(a, meta)
    tolerance = 1e-8
    output = {state: reconstruct(a, state) for state in ("A", "B")}
    errors = {state: compare(output[state], a["expected_" + state], tolerance)
              for state in ("A", "B")}
    rejected = 0
    if mutation_checks:
        changed = a["expected_A"].copy()
        changed[0, 0] += 1e-4
        try:
            compare(output["A"], changed, tolerance)
        except ValueError:
            rejected += 1
        bad = dict(a)
        bad["row_ptr"] = a["row_ptr"].copy()
        bad["row_ptr"][0] = 1
        try:
            check_layout(bad, meta)
        except ValueError:
            rejected += 1
        need(rejected == 2, "Mutation guards failed")
    return {
        "schema": "owner_overlay_capsule_verify_v1",
        "status": "PASS",
        "manifest_sha256": sha(folder / "MANIFEST.json"),
        "errors_by_state": errors,
        "mutation_rejections": rejected,
        "full_organism_reproduced": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = verify(args.capsule, mutation_checks=True)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out is not None:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.write_text(encoded)
    print(encoded, end="")
