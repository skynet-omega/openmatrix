"""Read-only edge-incidence gate for a conservative specialized-row fallback.

Counts the frozen base CSR, not the extra edges inside owner-specific kernels.
The result cannot establish runtime speed or full coefficient equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROUND = ROOT / "motor_nuevo/round_20260924_fast5s_02"
CAPTURE = ROOT / "motor_nuevo/multirate_real_20260924_01/capture_01"
sys.path.insert(0, str(ROUND))
import a_waveform_cpu as old  # noqa: E402


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def add_rows(mask: np.ndarray, rows: np.ndarray, n: int, key: str) -> np.ndarray:
    require(rows.ndim == 1 and np.issubdtype(rows.dtype, np.integer), key + " dtype")
    require(np.all((rows >= 0) & (rows < n)), key + " range")
    out = np.zeros(n, dtype=np.bool_)
    out[rows] = True
    mask |= out
    return out


def recipient_rows(ptr: np.ndarray, indices: np.ndarray, sources: np.ndarray) -> np.ndarray:
    n = len(ptr) - 1
    require(sources.ndim == 1 and np.issubdtype(sources.dtype, np.integer)
            and np.all((sources >= 0) & (sources < n)), "Invalid substituted sources")
    out = np.zeros(n, dtype=np.bool_)
    for lo in range(0, len(indices), 2_000_000):
        hi = min(lo + 2_000_000, len(indices))
        local = np.flatnonzero(np.isin(indices[lo:hi], sources))
        if len(local):
            out[np.searchsorted(ptr, local.astype(np.int64) + lo, side="right") - 1] = True
    return out


def run(out: Path) -> dict:
    started = time.monotonic()
    out = out.resolve()
    require(out != CAPTURE and not out.is_relative_to(CAPTURE), "Output inside capture")
    require(not out.exists(), "Output must be unique")
    plan_path = HERE / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    require(plan["schema"] == "owner_fallback_edge_gate_v1", "Wrong plan")
    require(plan["budget"]["executions"] == 1 and plan["budget"]["gpu_seconds"] == 0,
            "Unexpected budget")
    expected = plan["input"]
    paths = {
        "effective_gpu_arrays.npz": CAPTURE / "effective_gpu_arrays.npz",
        "EFFECTIVE_ARRAY_PATHS.json": CAPTURE / "EFFECTIVE_ARRAY_PATHS.json",
        "executed_gpu_coefficient_layout.py": CAPTURE / "executed_sources/legacy_sources__gpu_coefficient_layout.py",
        "a_waveform_cpu.py": ROUND / "a_waveform_cpu.py",
    }
    hashes = {key: sha256(value) for key, value in paths.items()}
    require(hashes == expected, "Input hash mismatch")
    mapping = json.loads(paths["EFFECTIVE_ARRAY_PATHS.json"].read_text())
    keys = ("cuda/indptr", "cuda/indices", "cuda/visual", *old.OVERRIDE_ROW_KEYS,
            *old.PATCH_POSITION_KEYS, *old.SUBSTITUTE_SOURCE_KEYS)
    require(all(key in mapping for key in keys), "Missing effective array")
    with np.load(paths["effective_gpu_arrays.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[mapping[key]] for key in set(keys)}
    ptr, indices, visual = (arrays[key] for key in ("cuda/indptr", "cuda/indices", "cuda/visual"))
    n = len(ptr) - 1
    require(ptr.dtype == np.int64 and indices.dtype == np.int32 and visual.dtype == np.bool_
            and n == 166700 and visual.shape == (n,) and ptr[0] == 0
            and np.all(np.diff(ptr) >= 0) and int(ptr[-1]) == len(indices)
            and np.all((indices >= 0) & (indices < n)), "Frozen CSR invalid")
    edges = np.diff(ptr)
    full_edges = int(ptr[-1])
    require(full_edges == plan["thresholds_frozen"]["base_edges_expected"], "Wrong base edge count")
    union = np.zeros(n, dtype=np.bool_)
    details: dict[str, dict[str, int]] = {}
    for key in old.OVERRIDE_ROW_KEYS:
        family = add_rows(union, arrays[key], n, key)
        details[key] = {"rows": int(family.sum()), "base_csr_edges": int(edges[family].sum())}
    for key in old.PATCH_POSITION_KEYS:
        positions = arrays[key]
        require(positions.ndim == 1 and np.issubdtype(positions.dtype, np.integer)
                and np.all((positions >= 0) & (positions < full_edges)), "Weight patch position invalid: " + key)
        rows = np.searchsorted(ptr, positions, side="right") - 1
        family = add_rows(union, rows, n, key)
        details[key] = {"rows": int(family.sum()), "base_csr_edges": int(edges[family].sum())}
    for key in old.SUBSTITUTE_SOURCE_KEYS:
        family = recipient_rows(ptr, indices, arrays[key])
        union |= family
        details[key] = {"rows": int(family.sum()), "base_csr_edges": int(edges[family].sum())}
    special_rows = int(union.sum())
    special_edges = int(edges[union].sum())
    require(special_rows == plan["thresholds_frozen"]["special_rows_expected"],
            "Owner union differs from independently frozen screen")
    require(special_edges + int(edges[~union].sum()) == full_edges,
            "Edge partition does not conserve CSR count")
    threshold_fraction = float(plan["thresholds_frozen"]["residual_full_sweep_fraction"])
    threshold_edges = threshold_fraction * full_edges
    result = {
        "schema": "owner_fallback_edge_gate_result_v1",
        "status": "COMPLETE",
        "plan_sha256": sha256(plan_path),
        "script_sha256": sha256(Path(__file__)),
        "input_sha256": hashes,
        "neurons": n,
        "base_csr_edges": full_edges,
        "special_receiver_rows": special_rows,
        "special_receiver_fraction": special_rows / n,
        "special_base_csr_edges_one_pass": special_edges,
        "special_base_csr_edge_fraction_one_pass": special_edges / full_edges,
        "other_base_csr_edges_one_pass": full_edges - special_edges,
        "nominal_residual_edges_after_5_727_sweeps": threshold_edges,
        "special_fits_nominal_residual_once": special_edges <= threshold_edges,
        "decision": ("NOT_RULED_OUT_BY_THIS_EDGE_GATE" if special_edges <= threshold_edges
                     else "FULL_ROW_FALLBACK_EXCEEDS_A_SIX_SWEEP_GATE"),
        "family_incidence_overlaps_do_not_sum": details,
        "interpretation_limit": "Counts base CSR row incidence once. Does not count specialized kernel reads, dynamic weight changes, build/recapture, local RHS calls, or wall time. A custom owner kernel may avoid a full-row fallback.",
        "stage_admission": False,
        "wall_s": time.monotonic() - started,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    require(result["wall_s"] <= plan["budget"]["wall_seconds_max"], "Wall budget exceeded")
    require(result["maxrss_kib"] <= plan["budget"]["ram_gib_max"] * 1024**2,
            "RAM budget exceeded")
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    require(len(payload.encode()) <= plan["budget"]["output_bytes_max"], "Output budget exceeded")
    out.mkdir(parents=True, exist_ok=False)
    (out / "RESULT.json").write_text(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("90-second CPU budget")))
    signal.alarm(90)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    print(json.dumps(run(args.out), ensure_ascii=False, allow_nan=False), flush=True)
