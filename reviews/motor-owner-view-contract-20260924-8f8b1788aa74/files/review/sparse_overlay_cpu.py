"""Exact sparse-delta algebra on the captured whole-brain CSR (CPU preflight).

The PN factor is observed; APL factors and PNKC release changes are explicitly
synthetic. This tests a reusable operator representation, not model parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAPTURE = ROOT / "motor_nuevo/multirate_real_20260924_01/capture_01"
MAX_CHUNK = 2_000_000


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def connected_flag(path: Path) -> bool:
    match = re.compile(r'^\s*"visual_output_connected"\s*:\s*(true|false)\s*,?\s*$')
    flags = []
    with path.open() as stream:
        for line in stream:
            found = match.fullmatch(line.rstrip("\n"))
            if found:
                flags.append(found.group(1) == "true")
    need(len(flags) == 1, "Ambiguous visual connection flag")
    return flags[0]


def edge_channels(row: np.ndarray, source: np.ndarray, weight: np.ndarray,
                  release: np.ndarray, caps: np.ndarray, visual: np.ndarray,
                  scale: float, connected: bool, n: int) -> np.ndarray:
    """Return ordinary signed, visual positive and visual negative sums."""
    need(row.shape == source.shape == weight.shape, "Edge layout mismatch")
    present = visual[row]
    value = weight * release[source]
    ordinary = np.where(~present & (connected | ~visual[source]), value * caps[source], 0.)
    visual_value = value * scale
    positive = np.where(present, np.maximum(visual_value, 0.), 0.)
    negative = np.where(present, np.maximum(-visual_value, 0.), 0.)
    return np.vstack(tuple(np.bincount(row, weights=channel, minlength=n)
                           for channel in (ordinary, positive, negative)))


def full_channels(ptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                  release: np.ndarray, caps: np.ndarray, visual: np.ndarray,
                  scale: float, connected: bool) -> np.ndarray:
    n = len(ptr) - 1
    out = np.zeros((3, n), dtype=np.float64)
    for lo in range(0, len(indices), MAX_CHUNK):
        hi = min(lo + MAX_CHUNK, len(indices))
        row = np.searchsorted(ptr, np.arange(lo, hi, dtype=np.int64), side="right") - 1
        out += edge_channels(row, indices[lo:hi], weights[lo:hi], release,
                             caps, visual, scale, connected, n)
    return out


class VersionGuard:
    def __init__(self, weight: str, source: str, event: str):
        self.expected = (weight, source, event)

    def query(self, weight: str, source: str, event: str) -> None:
        need((weight, source, event) == self.expected,
             "Cached operator version invalidated")


def run(out_dir: Path) -> dict:
    started = time.monotonic()
    out_dir = out_dir.resolve()
    need(not out_dir.exists() and not out_dir.is_relative_to(CAPTURE), "Output not unique/outside capture")
    plan_path = HERE / "PLAN_02.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "real_csr_sparse_overlay_gate_v1", "Wrong plan")
    paths = {name: CAPTURE / name for name in plan["inputs_sha256"]}
    actual_hashes = {name: sha(path) for name, path in paths.items()}
    need(actual_hashes == plan["inputs_sha256"], "Frozen input hash changed")
    mapping = json.loads(paths["EFFECTIVE_ARRAY_PATHS.json"].read_text())
    keys = ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps",
            "cuda/visual", "cuda/pi", "_general_positions", "_apl_gpu_positions",
            "_orn_pn_cuda/pn_rows", "_apl_gpu_rows", "_pnkc_cuda/source_rows")
    need(all(key in mapping for key in keys), "Missing effective array path")
    with np.load(paths["effective_gpu_arrays.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[mapping[key]] for key in keys}
    ptr, indices, weights, caps, visual = (arrays[key] for key in
        ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps", "cuda/visual"))
    n, m = len(ptr) - 1, len(arrays["cuda/pi"])
    need(n == 166700 and ptr.dtype == np.int64 and indices.dtype == np.int32
         and weights.dtype == caps.dtype == np.float64 and visual.dtype == np.bool_
         and ptr.shape == (n + 1,) and ptr[0] == 0 and ptr[-1] == len(indices) == len(weights)
         and caps.shape == visual.shape == (n,) and np.all(np.diff(ptr) >= 0)
         and np.all((indices >= 0) & (indices < n)) and np.isfinite(weights).all()
         and np.isfinite(caps).all(), "Frozen CSR invalid")
    inputs = json.loads(paths["block_inputs.json"].read_text())
    with np.load(paths["block_inputs.npz"], allow_pickle=False) as archive:
        pn_factor = archive[inputs["general_transmission"]["__array__"]].copy()
    need(pn_factor.shape == arrays["_general_positions"].shape and np.isfinite(pn_factor).all(),
         "PN factor layout")
    with np.load(paths["trace_state.npz"], allow_pickle=False) as archive:
        states = archive["values"][list(plan["fixture"]["captured_state_indices"])].copy()
    need(states.shape == (2, 359373) and np.isfinite(states).all(), "Captured state layout")
    scale = float(inputs["parameters"]["conductance_per_stored_weight"])
    need(np.isfinite(scale) and scale > 0., "Invalid stored-weight scale")
    connected = connected_flag(paths["final_state/session.json"])
    with np.load(paths["block_events.npz"], allow_pickle=False) as archive:
        event_arrays = tuple(archive[key].copy() for key in archive.files)
        port_sources = archive["array_0"].astype(np.int32, copy=True)
    need(len(port_sources) == 4062, "Wrong port count")
    event_version = digest_arrays(*event_arrays)
    pn_positions = arrays["_general_positions"]
    apl_positions = arrays["_apl_gpu_positions"]
    patch = np.unique(np.r_[pn_positions, apl_positions]).astype(np.int64)
    need(np.all((patch >= 0) & (patch < len(indices))), "Patch position outside CSR")
    sources = np.unique(np.r_[arrays["_orn_pn_cuda/pn_rows"], arrays["_apl_gpu_rows"],
                               arrays["_pnkc_cuda/source_rows"]]).astype(np.int32)
    need(np.all((sources >= 0) & (sources < n)), "Substitution source outside CSR")
    source_edge_positions = np.flatnonzero(np.isin(indices, sources)).astype(np.int64)
    overlay_positions = np.union1d(patch, source_edge_positions)
    port_edges = int(np.count_nonzero(np.isin(indices, port_sources)))
    overlay_fraction = len(overlay_positions) / len(indices)

    effective_weights = weights.copy()
    effective_weights[apl_positions] = weights[apl_positions] * (
        0.7 + 0.2 * np.sin(np.arange(len(apl_positions), dtype=np.float64)))
    effective_weights[pn_positions] = weights[pn_positions] * pn_factor
    weight_token = digest_arrays(patch, effective_weights[patch])
    overlay_rows = np.searchsorted(ptr, overlay_positions, side="right") - 1
    errors = []
    tokens = []
    for state in states:
        release = state[n + 2 * m:n + 2 * m + n].copy()
        need(release.shape == (n,) and np.isfinite(release).all() and np.all(release >= 0.),
             "Actual saved release outside physical domain")
        effective_release = release.copy()
        effective_release[arrays["_pnkc_cuda/source_rows"]] = (
            0.5 * release[arrays["_pnkc_cuda/source_rows"]] + 0.01)
        effective_release[arrays["_orn_pn_cuda/pn_rows"]] = 1.
        effective_release[arrays["_apl_gpu_rows"]] = 1.
        source_token = digest_arrays(sources, effective_release[sources])
        guard = VersionGuard(weight_token, source_token, event_version)
        guard.query(weight_token, source_token, event_version)
        tokens.append(source_token)
        baseline = full_channels(ptr, indices, weights, release, caps, visual, scale, connected)
        full_effective = full_channels(ptr, indices, effective_weights, effective_release,
                                       caps, visual, scale, connected)
        old_subset = edge_channels(overlay_rows, indices[overlay_positions],
                                   weights[overlay_positions], release, caps, visual,
                                   scale, connected, n)
        new_subset = edge_channels(overlay_rows, indices[overlay_positions],
                                   effective_weights[overlay_positions], effective_release,
                                   caps, visual, scale, connected, n)
        reconstructed = baseline + new_subset - old_subset
        errors.append({"ordinary_signed": float(np.max(np.abs(full_effective[0] - reconstructed[0]))),
                       "visual_positive": float(np.max(np.abs(full_effective[1] - reconstructed[1]))),
                       "visual_negative": float(np.max(np.abs(full_effective[2] - reconstructed[2])))})
    need(tokens[0] != tokens[1], "Two actual states did not exercise version change")
    a = VersionGuard(weight_token, tokens[0], event_version)
    rejected = 0
    for bad in (("x" + weight_token[1:], tokens[0], event_version),
                (weight_token, tokens[1], event_version),
                (weight_token, tokens[0], "x" + event_version[1:])):
        try:
            a.query(*bad)
        except ValueError:
            rejected += 1
    a.query(weight_token, tokens[0], event_version)
    max_error = max(max(row.values()) for row in errors)
    gates = plan["gates_frozen"]
    pass_identity = max_error <= gates["identity_max_abs"]
    pass_sparse = overlay_fraction <= gates["delta_edge_union_fraction_max"]
    pass_version = rejected >= gates["stale_token_rejections_min"]
    result = {
        "schema": "real_csr_sparse_overlay_gate_result_v1",
        "status": "COMPLETE",
        "plan_sha256": sha(plan_path),
        "script_sha256": sha(Path(__file__)),
        "input_sha256": actual_hashes,
        "captured_state_indices": plan["fixture"]["captured_state_indices"],
        "base_csr_edges": len(indices),
        "port_source_count": len(port_sources),
        "port_csr_edges": port_edges,
        "patch_positions_pn": len(pn_positions),
        "patch_positions_apl": len(apl_positions),
        "patch_positions_union": len(patch),
        "substituted_sources_union": len(sources),
        "substituted_source_edges": len(source_edge_positions),
        "delta_edge_union": len(overlay_positions),
        "delta_edge_union_fraction": overlay_fraction,
        "identity_max_abs_by_state_and_channel": errors,
        "identity_max_abs_all": max_error,
        "stale_token_rejections": rejected,
        "restored_A_token_valid": True,
        "gates": {"identity": pass_identity, "delta_fraction": pass_sparse,
                  "versions": pass_version},
        "classification": (gates["status_if_pass"] if all((pass_identity, pass_sparse, pass_version))
                           else "REJECT_SPARSE_DELTA_PRIMITIVE"),
        "scope": "Base CSR with actual graph and captured release; actual PN factors, synthetic APL factors and PNKC release. Visual E/I handled separately. Does not reproduce full owner target/rate, CNS state, or organism time.",
        "stage_admission": False,
        "wall_s": time.monotonic() - started,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    need(result["wall_s"] <= plan["budget"]["cpu_wall_seconds_max"], "Wall budget exceeded")
    need(result["maxrss_kib"] <= plan["budget"]["ram_gib_max"] * 1024**2,
         "RAM budget exceeded")
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    need(len(payload.encode()) <= plan["budget"]["output_bytes_max"], "Output budget exceeded")
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "RESULT.json").write_text(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second CPU budget")))
    signal.alarm(120)
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    print(json.dumps(run(args.out), ensure_ascii=False, allow_nan=False), flush=True)
