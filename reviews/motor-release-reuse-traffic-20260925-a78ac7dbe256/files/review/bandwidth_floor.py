"""Conditional traffic bound for the current full-CSR evaluation schedule.

This is not a kernel benchmark or an impossibility result for other algorithms.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cupy as cp
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(plan_path, out_path):
    plan = json.loads(plan_path.read_text())
    capture = Path(plan["capture"])
    for name, expected in plan["capture_sha256"].items():
        if digest(capture / name) != expected:
            raise ValueError(f"Changed capture: {name}")
    for name, expected in plan["source_sha256"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed source: {name}")
    mapping = json.loads((capture / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    with np.load(capture / "effective_gpu_arrays.npz", allow_pickle=False) as arrays:
        indices = arrays[mapping["cuda/indices"]]
        weights = arrays[mapping["cuda/weights"]]
        n_edges = int(len(indices))
        bytes_per_pass = int(indices.nbytes + weights.nbytes)
    with np.load(capture / "trace_clock.npz", allow_pickle=False) as clock:
        captured_queries = int(len(clock["query_s"]))
    whole = json.loads((capture / "RESULT.json").read_text())
    trials = int(whole["runtime"]["CNS"]["accepted"] + whole["runtime"]["CNS"]["rejected"])
    if n_edges != 25582938 or captured_queries != 60 or whole["requested_trial_ms"] != 1:
        raise ValueError("Unexpected frozen workload")
    if indices.dtype != np.int32 or weights.dtype != np.float64:
        raise ValueError("Unexpected CSR layout")
    device = cp.cuda.runtime.getDeviceProperties(0)
    bus_bits = int(device["memoryBusWidth"])
    clock_khz = int(device["memoryClockRate"])
    l2_bytes = int(device["l2CacheSize"])
    # Local CUDA reports 10.501 GHz DRAM clock; GDDR transfers twice per cycle.
    peak_Bps = 2 * clock_khz * 1000 * (bus_bits / 8)
    per_block_budget_s = (600 / 5000) * (125 / 1000)
    per_ms_budget_s = 600 / 5000
    calls_per_trial = 6  # graph_core.trial: 1 full midpoint + 2 fine midpoints, 2 calls each
    full_1ms = trials * calls_per_trial
    result = {
        "status": "CONDITIONAL_TRAFFIC_BOUND_ONLY",
        "capture_sha256": plan["capture_sha256"],
        "source_sha256": plan["source_sha256"],
        "gpu": {
            "name": device["name"].decode(),
            "memory_bus_width_bits": bus_bits,
            "memory_clock_khz": clock_khz,
            "l2_bytes": l2_bytes,
            "calculated_theoretical_peak_Bps": peak_Bps,
        },
        "graph": {"edges": n_edges, "index_dtype": str(indices.dtype), "weight_dtype": str(weights.dtype),
                  "index_and_weight_bytes_per_full_pass": bytes_per_pass,
                  "graph_bytes_to_l2_ratio": bytes_per_pass / l2_bytes},
        "captured_125us_block": {
            "full_coefficient_queries": captured_queries,
            "target_wall_s_for_125us": per_block_budget_s,
            "index_weight_bytes_if_read_each_query": captured_queries * bytes_per_pass,
            "ideal_peak_DRAM_seconds_for_those_bytes": captured_queries * bytes_per_pass / peak_Bps,
            "maximum_full_passes_at_peak_before_other_work": math.floor(per_block_budget_s * peak_Bps / bytes_per_pass),
        },
        "whole_captured_1ms": {
            "accepted_trials": int(whole["runtime"]["CNS"]["accepted"]),
            "rejected_trials": int(whole["runtime"]["CNS"]["rejected"]),
            "coefficient_calls_per_trial_from_frozen_graph_core": calls_per_trial,
            "calculated_full_queries": full_1ms,
            "target_wall_s_for_1ms": per_ms_budget_s,
            "index_weight_bytes_if_read_each_query": full_1ms * bytes_per_pass,
            "ideal_peak_DRAM_seconds_for_those_bytes": full_1ms * bytes_per_pass / peak_Bps,
            "maximum_full_passes_at_peak_before_other_work": math.floor(per_ms_budget_s * peak_Bps / bytes_per_pass),
        },
        "assumption": "Current full CSR kernel reads every source index and FP64 weight on every coefficient query, without inter-query DRAM reuse. This is plausible because the two arrays exceed L2, but it is a conditional traffic estimate, not a measured bandwidth or universal lower bound.",
        "not_counted": "CSR pointers, release/state, output stores, owner kernels, launches, sync, body, CPU. Peak is theoretical; sustained bandwidth is lower.",
    }
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.out), indent=2, allow_nan=False))
