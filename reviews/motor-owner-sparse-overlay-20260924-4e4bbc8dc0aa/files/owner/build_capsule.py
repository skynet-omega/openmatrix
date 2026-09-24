"""Build a compact real-graph, synthetic-owner-factor review capsule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import signal
import time

import numpy as np

from sparse_overlay_cpu import (CAPTURE, HERE, connected_flag, edge_channels,
                                need, sha)


def build(out: Path) -> dict:
    start = time.monotonic()
    out = out.resolve()
    need(not out.exists() and not out.is_relative_to(CAPTURE), "Output must be unique/outside capture")
    plan_path = HERE / "CAPSULE_PLAN_05.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "owner_overlay_portable_capsule_plan_v1", "Wrong capsule plan")
    frozen = {name: sha(HERE / name) for name in plan["frozen_predecessors"]}
    need(frozen == plan["frozen_predecessors"], "Predecessor changed")
    plan02 = json.loads((HERE / "PLAN_02.json").read_text())
    paths = {name: CAPTURE / name for name in plan02["inputs_sha256"]}
    need({name: sha(path) for name, path in paths.items()} == plan02["inputs_sha256"],
         "Capture changed")
    mapping = json.loads(paths["EFFECTIVE_ARRAY_PATHS.json"].read_text())
    keys = ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps",
            "cuda/visual", "cuda/pi", "_general_positions", "_apl_gpu_positions",
            "_orn_pn_cuda/pn_rows", "_apl_gpu_rows", "_pnkc_cuda/source_rows")
    with np.load(paths["effective_gpu_arrays.npz"], allow_pickle=False) as archive:
        arr = {key: archive[mapping[key]] for key in keys}
    ptr, indices, weights, caps, visual = (arr[key] for key in
        ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps", "cuda/visual"))
    n, m = len(ptr) - 1, len(arr["cuda/pi"])
    need(n == 166700 and int(ptr[-1]) == len(indices) == len(weights), "Wrong graph")
    inputs = json.loads(paths["block_inputs.json"].read_text())
    with np.load(paths["block_inputs.npz"], allow_pickle=False) as archive:
        pn_factor = archive[inputs["general_transmission"]["__array__"]].copy()
    with np.load(paths["trace_state.npz"], allow_pickle=False) as archive:
        states = archive["values"][list(plan02["fixture"]["captured_state_indices"])].copy()
    need(states.shape == (2, 359373), "Wrong captured states")
    scale = float(inputs["parameters"]["conductance_per_stored_weight"])
    connected = connected_flag(paths["final_state/session.json"])
    pn_positions, apl_positions = arr["_general_positions"], arr["_apl_gpu_positions"]
    patch_positions = np.unique(np.r_[pn_positions, apl_positions])
    substituted = np.unique(np.r_[arr["_orn_pn_cuda/pn_rows"], arr["_apl_gpu_rows"],
                                   arr["_pnkc_cuda/source_rows"]]).astype(np.int32)
    positions = np.union1d(patch_positions, np.flatnonzero(np.isin(indices, substituted)))
    row = np.searchsorted(ptr, positions, side="right") - 1
    active, local_row, counts = np.unique(row, return_inverse=True, return_counts=True)
    active = active.astype(np.int32)
    row_ptr = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    gates = plan["gates_frozen"]
    need(len(positions) == gates["expected_edges"] and len(active) == gates["expected_active_receivers"],
         "Unexpected sparse overlay layout")
    effective_weights = weights.copy()
    effective_weights[apl_positions] = weights[apl_positions] * (
        0.7 + 0.2 * np.sin(np.arange(len(apl_positions), dtype=np.float64)))
    effective_weights[pn_positions] = weights[pn_positions] * pn_factor
    capsule = {
        "active_rows": active,
        "row_ptr": row_ptr,
        "sources": indices[positions].copy(),
        "weight_before": weights[positions].copy(),
        "weight_after": effective_weights[positions].copy(),
        "caps": caps.copy(),
        "visual_u8": visual.astype(np.uint8),
        "scale": np.array(scale, dtype=np.float64),
        "connected": np.array(connected, dtype=np.bool_),
    }
    for name, state in zip(("A", "B"), states):
        release = state[n + 2*m:n + 2*m + n].copy()
        need(np.isfinite(release).all() and np.all(release >= 0.), "Invalid saved release")
        changed = release.copy()
        changed[arr["_pnkc_cuda/source_rows"]] = (
            0.5 * release[arr["_pnkc_cuda/source_rows"]] + 0.01)
        changed[arr["_orn_pn_cuda/pn_rows"]] = 1.
        changed[arr["_apl_gpu_rows"]] = 1.
        old = edge_channels(row, capsule["sources"], capsule["weight_before"],
                            release, caps, visual, scale, connected, n)
        new = edge_channels(row, capsule["sources"], capsule["weight_after"],
                            changed, caps, visual, scale, connected, n)
        capsule["release_" + name + "_before"] = release
        capsule["release_" + name + "_after"] = changed
        capsule["expected_" + name] = (new - old)[:, active]
    need(np.isfinite(capsule["expected_A"]).all() and np.isfinite(capsule["expected_B"]).all(),
         "Nonfinite capsule expected")
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "capsule.npz", **capsule)
    meta = {
        "schema": "owner_overlay_portable_capsule_v1",
        "label": "Real whole-brain CSR subset; synthetic APL/PNKC factors",
        "base_csr_edges": int(ptr[-1]),
        "selected_edges": len(positions),
        "active_receiver_rows": len(active),
        "state_query_indices": plan02["fixture"]["captured_state_indices"],
        "actual_PN_factor": True,
        "synthetic_APL_and_PNKC_factors": True,
        "channels": ["ordinary_signed", "visual_positive", "visual_negative"],
        "upstream_input_sha256": plan02["inputs_sha256"],
        "full_organism_or_full_coefficient_included": False,
        "stage_admission": False,
    }
    (out / "META.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "owner_overlay_capsule_manifest_v1",
        "plan_sha256": sha(plan_path),
        "builder_sha256": sha(Path(__file__)),
        "files": {name: {"sha256": sha(out / name), "bytes": (out / name).stat().st_size}
                  for name in ("capsule.npz", "META.json")},
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    from verify_capsule import verify
    verified = verify(out, mutation_checks=True)
    elapsed = time.monotonic() - start
    total_bytes = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    budget = plan["budget"]
    need(elapsed <= budget["wall_seconds_max"] and total_bytes <= budget["output_bytes_max"]
         and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
             <= budget["ram_gib_max"] * 1024**2, "Capsule budget exceeded")
    result = {
        "schema": "owner_overlay_capsule_build_result_v1",
        "status": "COMPLETE",
        "plan_sha256": sha(plan_path),
        "manifest_sha256": sha(out / "MANIFEST.json"),
        "verified": verified,
        "files_bytes": total_bytes,
        "wall_s": elapsed,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "classification": "PORTABLE_PRIMITIVE_ONLY",
        "stage_admission": False,
    }
    (out / "RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60-second capsule budget")))
    signal.alarm(60)
    print(json.dumps(build(args.out), ensure_ascii=False), flush=True)
