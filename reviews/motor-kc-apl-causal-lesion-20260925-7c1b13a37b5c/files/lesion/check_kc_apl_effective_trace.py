"""Read only: test effective KC/APL writes in 60 already captured CNS queries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from sparse_overlay_cpu import need, sha

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent / "multirate_real_20260924_01/capture_01"


def run(out: Path):
    start = time.monotonic()
    out = out.resolve()
    need(not out.exists(), "Unique output required")
    plan_path = HERE / "KC_APL_EFFECTIVE_TRACE_PLAN_14.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "kc_apl_effective_write_trace_plan_v1", "Wrong plan")
    actual = {k: sha(CAPTURE / k) for k in plan["frozen_inputs_sha256"]}
    need(actual == plan["frozen_inputs_sha256"], "Captured bytes changed")
    checks = json.loads((CAPTURE / "ORACLE_CHECK.json").read_text())
    reference_exact = sum(
        c["projected_state_exact"] and c["target_exact"] and c["rate_exact"]
        for c in checks)
    mapping = json.loads((CAPTURE / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    keys = ("cuda/indptr", "cuda/indices", "cuda/pi", "_dynamic_gpu_rows",
            "_apl_gpu_rows")
    need(all(k in mapping for k in keys), "Missing owner rows")
    with np.load(CAPTURE / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        a = {k: z[mapping[k]] for k in keys}
    n = len(a["cuda/indptr"]) - 1
    m = len(a["cuda/pi"])
    transmission_start = n + 2 * m
    dynamic = a["_dynamic_gpu_rows"].astype(np.int64, copy=False)
    apl = transmission_start + a["_apl_gpu_rows"].astype(np.int64, copy=False)
    with np.load(CAPTURE / "trace_state.npz", allow_pickle=False) as z:
        state = z["values"]
    with np.load(CAPTURE / "trace_target.npz", allow_pickle=False) as z:
        target = z["values"]
    with np.load(CAPTURE / "trace_rate.npz", allow_pickle=False) as z:
        rate = z["values"]
    expected = plan["expected"]
    need(state.shape == target.shape == rate.shape
         and state.shape[0] == expected["queries"]
         and len(dynamic) == expected["rows_dynamic"]
         and len(apl) == expected["rows_apl_transmission"]
         and len(np.unique(dynamic)) == len(dynamic)
         and len(np.unique(apl)) == len(apl)
         and np.all((dynamic >= 0) & (dynamic < state.shape[1]))
         and np.all((apl >= 0) & (apl < state.shape[1])),
         "Trace or row layout changed")
    need(np.isfinite(state[:, dynamic]).all()
         and np.isfinite(target[:, dynamic]).all()
         and np.isfinite(rate[:, dynamic]).all()
         and np.isfinite(state[:, apl]).all()
         and np.isfinite(target[:, apl]).all()
         and np.isfinite(rate[:, apl]).all(), "Nonfinite selected values")
    exact_dynamic_target = bool(np.array_equal(target[:, dynamic], state[:, dynamic]))
    exact_dynamic_rate = bool(np.count_nonzero(rate[:, dynamic]) == 0)
    exact_apl_target = bool(np.array_equal(target[:, apl], state[:, apl]))
    exact_apl_rate = bool(np.count_nonzero(rate[:, apl]) == 0)
    ptr, indices = a["cuda/indptr"], a["cuda/indices"]
    base_edges_dynamic = int(np.diff(ptr)[dynamic].sum())
    with np.load(CAPTURE / "block_events.npz", allow_pickle=False) as z:
        port_sources = z["array_0"].astype(np.int32, copy=True)
    port_mask = np.isin(indices, port_sources)
    counts = np.r_[0, np.cumsum(port_mask, dtype=np.int64)]
    port_edges_dynamic = int((counts[ptr[dynamic + 1]] - counts[ptr[dynamic]]).sum())
    total_port_edges = int(port_mask.sum())
    result = {
        "schema": "kc_apl_effective_write_trace_result_v1",
        "classification": "CONSUMED_TRACE_IDENTITY_ONLY",
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "input_sha256": actual, "queries": state.shape[0],
        "oracle_reference_queries_exact": reference_exact,
        "dynamic_rows": len(dynamic), "apl_transmission_rows": len(apl),
        "dynamic_target_equals_state_exact": exact_dynamic_target,
        "dynamic_rate_zero_exact": exact_dynamic_rate,
        "apl_transmission_target_equals_state_exact": exact_apl_target,
        "apl_transmission_rate_zero_exact": exact_apl_rate,
        "dynamic_target_max_abs": float(np.max(np.abs(target[:, dynamic] - state[:, dynamic]))),
        "dynamic_rate_max_abs": float(np.max(np.abs(rate[:, dynamic]))),
        "apl_transmission_target_max_abs": float(np.max(np.abs(target[:, apl] - state[:, apl]))),
        "apl_transmission_rate_max_abs": float(np.max(np.abs(rate[:, apl]))),
        "base_csr_edges_into_dynamic_rows": base_edges_dynamic,
        "port_edges_into_dynamic_rows": port_edges_dynamic,
        "total_port_edges": total_port_edges,
        "port_edge_fraction_into_dynamic_rows": port_edges_dynamic / total_port_edges,
        "actual_coefficient_consumed": True,
        "full_owner_candidate_parity": False,
        "safe_pruning_proven": False,
        "organism_executed": False,
        "stage_admission": False,
        "wall_s": time.monotonic() - start,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    gates = {
        "queries": result["queries"],
        "rows_dynamic": result["dynamic_rows"],
        "rows_apl_transmission": result["apl_transmission_rows"],
        "target_dynamic_equals_state_exact": exact_dynamic_target,
        "rate_dynamic_all_zero_exact": exact_dynamic_rate,
        "target_apl_transmission_equals_state_exact": exact_apl_target,
        "rate_apl_transmission_zero_exact": exact_apl_rate,
        "oracle_reference_queries_exact": reference_exact,
    }
    need(gates == expected, "Expected effective write did not survive the full wrapper")
    budget = plan["budget"]
    need(result["wall_s"] <= budget["cpu_wall_seconds_max"]
         and result["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2, "Budget exceeded")
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    need(len(payload.encode()) <= budget["output_bytes_max"], "Output budget")
    out.mkdir(parents=True, exist_ok=False)
    (out / "RESULT.json").write_text(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out), allow_nan=False))
