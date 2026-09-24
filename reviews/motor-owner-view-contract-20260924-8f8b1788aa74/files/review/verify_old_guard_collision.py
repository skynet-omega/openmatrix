"""Reproduce the published v1 guard's missing-before-view collision."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from sparse_overlay_cpu import VersionGuard, digest_arrays, need, sha
from chatgpt_original.overlay_vistas import Vista, aplicar, corriente, delta

HERE = Path(__file__).resolve().parent


def run():
    started = time.monotonic()
    plan_path = HERE / "LEGACY_GUARD_COLLISION_PLAN_11.json"
    plan = json.loads(plan_path.read_text())
    need(plan["schema"] == "legacy_guard_collision_replay_plan_v1", "Wrong plan")
    sources = {name: sha(HERE / name) for name in plan["frozen_source_sha256"]}
    need(sources == plan["frozen_source_sha256"], "Input source changed")
    rows = np.array([0], dtype=np.int32)
    ptr = np.array([0, 1], dtype=np.int64)
    src = np.array([0], dtype=np.int32)
    w0 = np.array([2.], dtype=np.float64)
    w1 = np.array([3.], dtype=np.float64)
    sA = np.array([.25], dtype=np.float64)
    sB = np.array([.5], dtype=np.float64)
    s1 = np.array([1.], dtype=np.float64)
    weight = digest_arrays(np.array([0]), w1)
    source = digest_arrays(np.array([0]), s1)
    event = hashlib.sha256(b"same-event-history").hexdigest()
    legacy = VersionGuard(weight, source, event)
    legacy.query(weight, source, event)
    legacy.query(weight, source, event)
    ctx = dict(consumer="ordinary", epoch_ns=0, t_s_hex=float(.125).hex(),
               phase="accepted", operator="w1", owners="owner1", events=event,
               candidate="same")

    def view(w, s):
        return Vista(rows, ptr, src, w, s, np.array([1.]),
                     np.array([0], dtype=np.uint8), 1., True, ctx)

    before_a, before_b, after = view(w0, sA), view(w0, sB), view(w1, s1)
    delta_a, delta_b = delta(before_a, after), delta(before_b, after)
    got_a, got_b = float(delta_a.values[0, 0]), float(delta_b.values[0, 0])
    rejected = False
    try:
        aplicar(corriente(before_b), delta_a)
    except ValueError:
        rejected = True
    result = {
        "schema": "legacy_guard_collision_replay_result_v1",
        "legacy_guard_accepts_both": True,
        "delta_A": got_a, "delta_B": got_b,
        "stale_reuse_error": abs(got_a - got_b),
        "new_view_keys_distinct": before_a.key != before_b.key,
        "new_guard_rejects_stale": rejected,
        "source_sha256": sources,
        "plan_sha256": sha(plan_path),
        "wall_s": time.monotonic() - started,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "One-edge synthetic diagnostic. No published runner reused a delta.",
    }
    for key, expected in plan["gates_frozen"].items():
        need(result[key] == expected, "Frozen gate failed: " + key)
    budget = plan["budget"]
    need(result["wall_s"] <= budget["cpu_wall_seconds_max"]
         and result["maxrss_kib"] <= budget["ram_gib_max"] * 1024**2,
         "Budget exceeded")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    path = args.out.resolve()
    need(not path.exists(), "Unique output required")
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    plan = json.loads((HERE / "LEGACY_GUARD_COLLISION_PLAN_11.json").read_text())
    need(len(payload.encode()) <= plan["budget"]["output_bytes_max"], "Output budget")
    path.mkdir(parents=True, exist_ok=False)
    (path / "RESULT.json").write_text(payload)
    print(json.dumps(result, allow_nan=False))
