"""Bounded real-graph test of event/owner current composition, not a CNS engine."""
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
CAPTURE = ROOT / "motor_nuevo/multirate_real_20260924_01/capture_01"
EVENT_SOURCE = ROOT / "motor_nuevo/multirate_real_20260924_01/event_sparse"
sys.path.insert(0, str(EVENT_SOURCE))
from event_sparse import convolution, decode_npz, prepare_ports, project_cpu  # noqa: E402
from sparse_overlay_cpu import connected_flag, edge_channels, need, sha  # noqa: E402


def channels(positions, ptr, indices, weights, release, caps, visual, scale, connected, n):
    rows = np.searchsorted(ptr, positions, side="right") - 1
    return edge_channels(rows, indices[positions], weights[positions], release,
                         caps, visual, scale, connected, n)


def run(out: Path):
    started = time.monotonic()
    out = out.resolve()
    need(not out.exists() and not out.is_relative_to(CAPTURE), "Output must be unique")
    plan_path = HERE / "OWNER_EVENT_OVERLAP_PLAN_10.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "owner_event_overlap_gate_plan_v1", "Wrong frozen plan")
    actual = {name: sha(CAPTURE / name) for name in plan["inputs_sha256"]}
    need(actual == plan["inputs_sha256"], "Capture input hash mismatch")
    dependencies = {
        "event_sparse.py": sha(EVENT_SOURCE / "event_sparse.py"),
        "sparse_overlay_cpu.py": sha(HERE / "sparse_overlay_cpu.py"),
    }
    need(dependencies == plan["source_dependencies_sha256"], "Source dependency changed")
    mapping = json.loads((CAPTURE / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    keys = ("cuda/indptr", "cuda/indices", "cuda/weights", "cuda/caps", "cuda/visual",
            "cuda/pi", "_general_positions", "_apl_gpu_positions",
            "_orn_pn_cuda/pn_rows", "_apl_gpu_rows", "_pnkc_cuda/source_rows")
    need(all(k in mapping for k in keys), "Missing captured array")
    with np.load(CAPTURE / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        a = {k: z[mapping[k]] for k in keys}
    ptr, indices, weights, caps, visual = (
        a[k] for k in ("cuda/indptr", "cuda/indices", "cuda/weights",
                      "cuda/caps", "cuda/visual"))
    n, m = len(ptr) - 1, len(a["cuda/pi"])
    need(n == 166700 and len(indices) == 25582938 and ptr[-1] == len(indices),
         "Unexpected whole graph")
    ports = prepare_ports(decode_npz(CAPTURE / "block_events"), n)
    with np.load(CAPTURE / "trace_clock.npz", allow_pickle=False) as z:
        times = z["query_s"][[2, 59]].copy()
    with np.load(CAPTURE / "trace_state.npz", allow_pickle=False) as z:
        states = z["values"][[2, 59]].copy()
    need(states.shape == (2, 359373), "Captured state shape")
    meta = json.loads((CAPTURE / "block_inputs.json").read_text())
    scale = float(meta["parameters"]["conductance_per_stored_weight"])
    connected = connected_flag(CAPTURE / "final_state/session.json")
    with np.load(CAPTURE / "block_inputs.npz", allow_pickle=False) as z:
        factor = z[meta["general_transmission"]["__array__"]].copy()
    pn_pos = a["_general_positions"]
    apl_pos = a["_apl_gpu_positions"]
    need(len(factor) == len(pn_pos), "PN factor layout")
    source_rows = np.unique(np.r_[a["_orn_pn_cuda/pn_rows"], a["_apl_gpu_rows"],
                                  a["_pnkc_cuda/source_rows"]]).astype(np.int32)
    patch_pos = np.unique(np.r_[pn_pos, apl_pos]).astype(np.int64)
    owner_source_pos = np.flatnonzero(np.isin(indices, source_rows)).astype(np.int64)
    owner_pos = np.union1d(patch_pos, owner_source_pos)
    port_pos = np.flatnonzero(np.isin(indices, ports.rows)).astype(np.int64)
    union_pos = np.union1d(owner_pos, port_pos)
    overlap_pos = np.intersect1d(owner_pos, port_pos, assume_unique=True)
    patch_port_pos = np.intersect1d(patch_pos, port_pos, assume_unique=True)
    source_port = np.intersect1d(source_rows, ports.rows, assume_unique=True)
    need(len(union_pos) == len(port_pos) + len(owner_pos) - len(overlap_pos),
         "Union accounting")
    owner_weights = weights.copy()
    owner_weights[apl_pos] = weights[apl_pos] * (
        0.7 + 0.2 * np.sin(np.arange(len(apl_pos), dtype=np.float64)))
    owner_weights[pn_pos] = weights[pn_pos] * factor
    per_query = []
    for state_index, t, state in zip((2, 59), times, states):
        t = float(t)
        q, s_port = project_cpu(ports, t)
        q_actual = state[ports.rows]
        release_event = state[n + 2 * m:n + 2 * m + n].copy()
        need(release_event.shape == (n,), "Captured release shape")
        error_q = float(np.max(np.abs(q - q_actual)))
        error_s = float(np.max(np.abs(s_port - release_event[ports.rows])))
        need(max(error_q, error_s) <= plan["gates_frozen"]["port_projection_max_abs"],
             "Actual port state differs from event projection")
        s_no_event = (ports.s * np.exp(-t / ports.ts)
                      + ports.q * convolution(t, ports.tau, ports.ts))
        release_pre = release_event.copy()
        release_pre[ports.rows] = s_no_event

        def substitute(release):
            result = release.copy()
            pnkc = a["_pnkc_cuda/source_rows"]
            result[pnkc] = 0.5 * release[pnkc] + 0.01
            result[a["_orn_pn_cuda/pn_rows"]] = 1.
            result[a["_apl_gpu_rows"]] = 1.
            return result

        owner_release = substitute(release_event)
        owner_pre_release = substitute(release_pre)
        c0 = channels(union_pos, ptr, indices, weights, release_pre,
                      caps, visual, scale, connected, n)
        ce = channels(union_pos, ptr, indices, weights, release_event,
                      caps, visual, scale, connected, n)
        co = channels(union_pos, ptr, indices, owner_weights, owner_release,
                      caps, visual, scale, connected, n)
        composed = c0 + (ce - c0) + (co - ce)
        owner_from_pre = channels(union_pos, ptr, indices, owner_weights,
                                  owner_pre_release, caps, visual, scale, connected, n)
        wrong = ce + (owner_from_pre - c0)
        error = float(np.max(np.abs(composed - co)))
        wrong_error = float(np.max(np.abs(wrong - co)))
        need(np.isfinite(composed).all() and np.isfinite(wrong).all()
             and error <= plan["gates_frozen"]["composed_vs_direct_max_abs"],
             "Event/owner composition error")
        per_query.append({
            "state_index": state_index, "t_s": t,
            "q_vs_trace_max_abs": error_q,
            "s_vs_trace_max_abs": error_s,
            "event_delta_max_abs": float(np.max(np.abs(ce - c0))),
            "owner_delta_max_abs": float(np.max(np.abs(co - ce))),
            "composed_vs_direct_max_abs": error,
            "wrong_pre_event_owner_baseline_max_abs": wrong_error,
        })
    result = {
        "schema": "owner_event_overlap_gate_result_v1",
        "classification": "STRUCTURAL_COMPOSITION_ONLY",
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "input_sha256": actual, "dependency_sha256": dependencies,
        "base_edges": len(indices), "port_sources": len(ports.rows),
        "port_edges": len(port_pos), "owner_edges": len(owner_pos),
        "owner_port_overlap_edges": len(overlap_pos),
        "owner_weight_patch_port_edges": len(patch_port_pos),
        "owner_source_port_sources": len(source_port),
        "union_edges": len(union_pos),
        "queries": per_query,
        "actual_factors": "PN general only",
        "synthetic_factors": "APL weights and PNKC release",
        "full_target_rate_validated": False,
        "organism_executed": False,
        "stage_admission": False,
        "wall_s": time.monotonic() - started,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    budget = plan["budget"]
    need(result["wall_s"] <= budget["cpu_wall_seconds_max"]
         and result["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2,
         "Budget exceeded")
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    need(len(payload.encode()) <= budget["output_bytes_max"], "Output budget")
    out.mkdir(parents=True, exist_ok=False)
    (out / "RESULT.json").write_text(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60s budget")))
    signal.alarm(60)
    print(json.dumps(run(arguments.out), ensure_ascii=False, allow_nan=False))
