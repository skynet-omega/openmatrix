"""Offline diagnostic of exact source-release reuse in one captured CNS block.

This does not integrate an organism or certify a QSS/MRI candidate. It counts
the base CSR edges whose source release differs between actual oracle queries.
"""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fractions(delta, outdegree, thresholds):
    return {
        str(level): {
            "source_count": int(np.count_nonzero(delta > level)),
            "edge_count": int(outdegree[delta > level].sum(dtype=np.int64)),
        }
        for level in thresholds
    }


def run(plan_file, output_file):
    plan = json.loads(plan_file.read_text())
    capture = Path(plan["capture"])
    start = time.monotonic()
    for name, expected in plan["sha256"].items():
        require(sha256(capture / name) == expected, f"Captured input changed: {name}")
    paths = json.loads((capture / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    with np.load(capture / "effective_gpu_arrays.npz", allow_pickle=False) as arrays:
        n = len(arrays[paths["cuda/indptr"]]) - 1
        sources = arrays[paths["cuda/indices"]]
        photo_rows = arrays[paths["cuda/pi"]]
    require(sources.ndim == 1 and sources.dtype.kind in "iu", "CSR columns")
    require(sources.size == 25582938 and n == 166700, "Unexpected graph size")
    require(np.all(sources < n), "CSR source out of range")
    outdegree = np.bincount(sources, minlength=n).astype(np.int64)
    require(int(outdegree.sum()) == len(sources), "Bad edge accounting")
    del sources
    with np.load(capture / "block_events.npz", allow_pickle=False) as events:
        event_rows = events["array_0"].astype(np.int64)
    require(np.all(event_rows < n), "Event row out of range")
    with np.load(capture / "trace_state.npz", allow_pickle=False) as states:
        state = states["values"]
    with np.load(capture / "trace_clock.npz", allow_pickle=False) as clock:
        times = clock["query_s"]
        require(str(clock["phase"]) == "accepted", "Wrong phase")
        require(np.unique(clock["epoch"]).size == 1, "Multiple epochs")
    ts = n + 2 * len(photo_rows)
    require(state.shape == (60, 359373) and ts + n <= state.shape[1], "State layout")
    release = np.ascontiguousarray(state[:, ts : ts + n])
    require(np.isfinite(release).all(), "Nonfinite release")
    levels = plan["descriptive_abs_thresholds"]
    span = np.ptp(release, axis=0)
    pair = np.abs(np.diff(release, axis=0))
    adjacent = [fractions(pair[i], outdegree, levels) for i in range(len(pair))]
    same_time = []
    for t in np.unique(times):
        positions = np.flatnonzero(times == t)
        if len(positions) > 1:
            d = np.ptp(release[positions], axis=0)
            same_time.append({
                "time_hex": float(t).hex(),
                "query_indices": positions.tolist(),
                "release_difference": fractions(d, outdegree, levels),
            })
    event_mask = np.zeros(n, dtype=bool)
    event_mask[event_rows] = True
    result = {
        "status": "COMPLETE_DESCRIPTIVE_ONLY",
        "capture_sha256": plan["sha256"],
        "n_sources": n,
        "n_edges": int(outdegree.sum()),
        "n_queries": len(release),
        "unique_query_times": int(len(np.unique(times))),
        "time_reversals_in_query_order": int(np.count_nonzero(np.diff(times) < 0)),
        "transmission_start": ts,
        "thresholds_are_descriptive_not_error_contract": levels,
        "span_all_queries": fractions(span, outdegree, levels),
        "span_event_sources": {
            str(level): {
                "source_count": int(np.count_nonzero(event_mask & (span > level))),
                "edge_count": int(outdegree[event_mask & (span > level)].sum()),
            }
            for level in levels
        },
        "adjacent_query_edge_count": {
            str(level): {
                "minimum": int(min(x[str(level)]["edge_count"] for x in adjacent)),
                "median": float(np.median([x[str(level)]["edge_count"] for x in adjacent])),
                "maximum": int(max(x[str(level)]["edge_count"] for x in adjacent)),
                "total": int(sum(x[str(level)]["edge_count"] for x in adjacent)),
            }
            for level in levels
        },
        "same_time_repeated_queries": same_time,
        "wall_s": time.monotonic() - start,
        "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "interpretation_limit": "Observed RK oracle-call order only; not a lower bound for a different integrator or certified QSS scheduling.",
    }
    output_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.out), indent=2, allow_nan=False))
