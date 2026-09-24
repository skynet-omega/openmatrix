"""One bounded GPU check of the generic sparse owner-edge overlay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np

from sparse_overlay_cpu import (CAPTURE, HERE, VersionGuard, connected_flag,
                                digest_arrays, edge_channels, need, sha)


def run(out: Path) -> dict:
    started = time.monotonic()
    out = out.resolve()
    need(not out.exists() and not out.is_relative_to(CAPTURE), "Output must be unique/outside capture")
    plan_path = HERE / "PLAN_03.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "real_csr_sparse_overlay_cuda_gate_v1", "Wrong plan")
    predecessor_paths = {name: HERE / name for name in plan["frozen_predecessors"]}
    need({name: sha(path) for name, path in predecessor_paths.items()}
         == plan["frozen_predecessors"], "Predecessor changed")
    predecessor = json.loads((HERE / "PLAN_02.json").read_text())
    paths = {name: CAPTURE / name for name in predecessor["inputs_sha256"]}
    need({name: sha(path) for name, path in paths.items()}
         == predecessor["inputs_sha256"], "Frozen graph/input changed")
    mapping = json.loads(paths["EFFECTIVE_ARRAY_PATHS.json"].read_text())
    keys = ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps",
            "cuda/visual", "cuda/pi", "_general_positions", "_apl_gpu_positions",
            "_orn_pn_cuda/pn_rows", "_apl_gpu_rows", "_pnkc_cuda/source_rows")
    with np.load(paths["effective_gpu_arrays.npz"], allow_pickle=False) as archive:
        arrays = {key: archive[mapping[key]] for key in keys}
    ptr, indices, weights, caps, visual = (arrays[key] for key in
        ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps", "cuda/visual"))
    n, m = len(ptr) - 1, len(arrays["cuda/pi"])
    need(n == 166700 and int(ptr[-1]) == len(indices) == len(weights)
         == plan["gates_frozen"]["expected_base_edges"], "Wrong frozen CSR")
    inputs = json.loads(paths["block_inputs.json"].read_text())
    with np.load(paths["block_inputs.npz"], allow_pickle=False) as archive:
        pn_factor = archive[inputs["general_transmission"]["__array__"]].copy()
    with np.load(paths["trace_state.npz"], allow_pickle=False) as archive:
        states = archive["values"][list(predecessor["fixture"]["captured_state_indices"])].copy()
    need(states.shape == (2, 359373) and np.isfinite(states).all(), "Wrong saved states")
    with np.load(paths["block_events.npz"], allow_pickle=False) as archive:
        event_token = digest_arrays(*(archive[key] for key in archive.files))
    scale = float(inputs["parameters"]["conductance_per_stored_weight"])
    connected = connected_flag(paths["final_state/session.json"])
    pn_positions, apl_positions = (arrays[key] for key in
        ("_general_positions", "_apl_gpu_positions"))
    patch = np.unique(np.r_[pn_positions, apl_positions]).astype(np.int64)
    sources = np.unique(np.r_[arrays["_orn_pn_cuda/pn_rows"], arrays["_apl_gpu_rows"],
                               arrays["_pnkc_cuda/source_rows"]]).astype(np.int32)
    source_positions = np.flatnonzero(np.isin(indices, sources)).astype(np.int64)
    positions = np.union1d(patch, source_positions)
    need(len(positions) == plan["gates_frozen"]["expected_delta_edges"],
         "Wrong overlay edge count")
    effective_weights = weights.copy()
    effective_weights[apl_positions] = weights[apl_positions] * (
        0.7 + 0.2 * np.sin(np.arange(len(apl_positions), dtype=np.float64)))
    effective_weights[pn_positions] = weights[pn_positions] * pn_factor
    weight_token = digest_arrays(patch, effective_weights[patch])
    rows = np.searchsorted(ptr, positions, side="right") - 1
    active = np.unique(rows).astype(np.int32)
    begin = np.searchsorted(positions, ptr[active], side="left").astype(np.int64)
    end = np.searchsorted(positions, ptr[active + 1], side="left").astype(np.int64)
    need(np.all(end > begin) and int(np.sum(end - begin)) == len(positions),
         "Reduced row pointers do not cover every overlay edge")

    out.mkdir(parents=True, exist_ok=False)
    os.environ["CUPY_CACHE_DIR"] = str(out / "cupy_cache")
    import cupy as cp

    cp.cuda.Device(0).use()
    free_before, _ = cp.cuda.runtime.memGetInfo()
    module = cp.RawModule(code=(HERE / "owner_overlay.cu").read_text(),
                          options=("--std=c++11", "--fmad=false", "--prec-div=true",
                                   "--prec-sqrt=true"),
                          name_expressions=["owner_overlay_delta"])
    kernel = module.get_function("owner_overlay_delta")
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        d_active = cp.asarray(active)
        d_begin, d_end = cp.asarray(begin), cp.asarray(end)
        d_sources = cp.asarray(indices[positions])
        d_old_weight = cp.asarray(weights[positions])
        d_new_weight = cp.asarray(effective_weights[positions])
        d_caps = cp.asarray(caps)
        d_visual = cp.asarray(visual.view(np.uint8))
        d_output = cp.empty((3, len(active)), dtype=cp.float64)
    stream.synchronize()
    free_after_setup, _ = cp.cuda.runtime.memGetInfo()

    snapshots = []
    errors = []
    source_tokens = []
    for index in (0, 1, 0):
        state = states[index]
        release = state[n + 2*m:n + 2*m + n].copy()
        need(release.shape == (n,) and np.isfinite(release).all()
             and np.all(release >= 0.), "Invalid saved source release")
        effective_release = release.copy()
        effective_release[arrays["_pnkc_cuda/source_rows"]] = (
            0.5 * release[arrays["_pnkc_cuda/source_rows"]] + 0.01)
        effective_release[arrays["_orn_pn_cuda/pn_rows"]] = 1.
        effective_release[arrays["_apl_gpu_rows"]] = 1.
        source_token = digest_arrays(sources, effective_release[sources])
        source_tokens.append(source_token)
        guard = VersionGuard(weight_token, source_token, event_token)
        guard.query(weight_token, source_token, event_token)
        expected_new = edge_channels(rows, indices[positions], effective_weights[positions],
                                     effective_release, caps, visual, scale, connected, n)
        expected_old = edge_channels(rows, indices[positions], weights[positions], release,
                                     caps, visual, scale, connected, n)
        expected = (expected_new - expected_old)[:, active]
        with stream:
            d_release_before = cp.asarray(release)
            d_release_after = cp.asarray(effective_release)
            kernel(((len(active) + 127) // 128,), (128,),
                   (np.int32(len(active)), d_active, d_begin, d_end, d_sources,
                    d_old_weight, d_new_weight, d_release_before, d_release_after,
                    d_caps, d_visual, np.float64(scale), np.bool_(connected), d_output),
                   stream=stream)
            observed = cp.asnumpy(d_output)
        stream.synchronize()
        need(np.isfinite(observed).all(), "Nonfinite CUDA overlay")
        snapshots.append(observed)
        errors.append(float(np.max(np.abs(observed - expected))))
        del d_release_before, d_release_after
    need(source_tokens[0] != source_tokens[1], "B version did not change")
    restored = bool(np.array_equal(snapshots[0], snapshots[2]))
    guard_a = VersionGuard(weight_token, source_tokens[0], event_token)
    rejected = 0
    for bad in (("x" + weight_token[1:], source_tokens[0], event_token),
                (weight_token, source_tokens[1], event_token),
                (weight_token, source_tokens[0], "x" + event_token[1:])):
        try:
            guard_a.query(*bad)
        except ValueError:
            rejected += 1
    guard_a.query(weight_token, source_tokens[0], event_token)
    free_after_queries, _ = cp.cuda.runtime.memGetInfo()
    additional_vram = max(0, free_before - min(free_after_setup, free_after_queries))
    max_error = max(errors)
    result = {
        "schema": "real_csr_sparse_overlay_cuda_result_v1",
        "status": "COMPLETE",
        "plan_sha256": sha(plan_path),
        "runner_sha256": sha(Path(__file__)),
        "kernel_sha256": sha(HERE / "owner_overlay.cu"),
        "frozen_predecessor_sha256": plan["frozen_predecessors"],
        "base_csr_edges": len(indices),
        "overlay_edges": len(positions),
        "active_receiver_rows": len(active),
        "cuda_max_abs_error_by_query_A_B_A": errors,
        "cuda_max_abs_error": max_error,
        "A_restored_bitwise": restored,
        "stale_token_rejections": rejected,
        "additional_vram_bytes_observed": additional_vram,
        "classification": ("CUDA_SPARSE_DELTA_PRIMITIVE_ONLY"
                           if max_error <= plan["gates_frozen"]["max_absolute_channel_error"]
                           and restored and rejected >= plan["gates_frozen"]["min_stale_token_rejections"]
                           else "REJECT_CUDA_OVERLAY"),
        "scope": "Only overlay current from PN/APL weight patches and source substitutions on real graph. APL/PNKC factors are synthetic. No full owner target/rate, candidate CNS integration, body, or engine speed claim.",
        "stage_admission": False,
        "wall_s": time.monotonic() - started,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    budget = plan["budget"]
    need(result["wall_s"] <= budget["wall_seconds_max"], "Wall budget exceeded")
    need(result["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2, "RAM budget exceeded")
    need(additional_vram <= budget["vram_gib_additional_max"] * 1024**3,
         "VRAM budget exceeded")
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    need(len(payload.encode()) <= budget["output_bytes_max"], "Output budget exceeded")
    (out / "RESULT.json").write_text(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("90-second GPU budget")))
    signal.alarm(90)
    print(json.dumps(run(args.out), ensure_ascii=False, allow_nan=False), flush=True)
