"""Count structural receiver overlap of real event-port edges and owner rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from owner_precedence_screen import WRITERS
from sparse_overlay_cpu import need, sha

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent / "multirate_real_20260924_01/capture_01"


def run(out: Path):
    start = time.monotonic()
    out = out.resolve()
    need(not out.exists(), "Unique output required")
    plan_path = HERE / "PORT_OWNER_RECEIVER_PLAN_12.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "port_owner_receiver_incidence_plan_v1", "Wrong plan")
    actual = {k: sha(CAPTURE / k) for k in plan["inputs_sha256"]}
    need(actual == plan["inputs_sha256"], "Capture hash changed")
    need({k: sha(HERE / k) for k in plan["source_sha256"]}
         == plan["source_sha256"], "Owner source changed")
    mapping = json.loads((CAPTURE / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    keys = {"cuda/indptr", "cuda/indices"} | {entry[2] for entry in WRITERS}
    need(all(k in mapping for k in keys), "Missing owner arrays")
    with np.load(CAPTURE / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[mapping[k]] for k in keys}
    ptr, indices = arrays["cuda/indptr"], arrays["cuda/indices"]
    n = len(ptr) - 1
    with np.load(CAPTURE / "block_events.npz", allow_pickle=False) as z:
        port_sources = z["array_0"].astype(np.int32, copy=True)
    need(len(port_sources) == len(np.unique(port_sources))
         and np.all((port_sources >= 0) & (port_sources < n)), "Invalid port rows")
    positions = np.flatnonzero(np.isin(indices, port_sources))
    receiver = np.searchsorted(ptr, positions, side="right") - 1
    active = np.unique(receiver)
    target_touches = np.zeros(n, dtype=np.uint8)
    rate_touches = np.zeros(n, dtype=np.uint8)
    records = []
    for label, _, key, target_op, rate_op in WRITERS:
        row_mask = np.zeros(n, dtype=np.bool_)
        row_mask[arrays[key]] = True
        overlap = row_mask[active]
        records.append({
            "owner": label, "target_op": target_op, "rate_op": rate_op,
            "owner_rows": int(row_mask.sum()),
            "port_active_receiver_rows_in_owner": int(overlap.sum()),
            "port_edges_into_owner_rows": int(row_mask[receiver].sum()),
        })
        if target_op is not None:
            target_touches[row_mask] += 1
        if rate_op is not None:
            rate_touches[row_mask] += 1
    records.sort(key=lambda x: (-x["port_edges_into_owner_rows"], x["owner"]))
    gate = plan["gates_frozen"]
    need(len(indices) == gate["base_edges"]
         and len(port_sources) == gate["port_sources"]
         and len(positions) == gate["port_edges"]
         and len(WRITERS) == gate["owner_families"]
         and len(active) == gate["port_active_receiver_rows"], "Frozen topology mismatch")
    result = {
        "schema": "port_owner_receiver_incidence_result_v1",
        "classification": "STATIC_POTENTIAL_ONLY",
        "plan_sha256": sha(plan_path), "script_sha256": sha(Path(__file__)),
        "input_sha256": actual, "source_sha256": plan["source_sha256"],
        "base_edges": len(indices), "port_sources": len(port_sources),
        "port_edges": len(positions), "port_active_receiver_rows": len(active),
        "port_active_rows_with_any_special_target": int(np.count_nonzero(target_touches[active])),
        "port_active_rows_with_multiple_target_touches": int(np.count_nonzero(target_touches[active] >= 2)),
        "port_active_rows_with_any_special_rate": int(np.count_nonzero(rate_touches[active])),
        "port_active_rows_with_multiple_rate_touches": int(np.count_nonzero(rate_touches[active] >= 2)),
        "owner_family_incidence": records,
        "effective_owner_activation_checked": False,
        "full_target_rate_validated": False,
        "organism_executed": False,
        "stage_admission": False,
        "wall_s": time.monotonic() - start,
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
    args = parser.parse_args()
    print(json.dumps(run(args.out), ensure_ascii=False, allow_nan=False))
