"""Cold verifier for the portable release-reuse capsule and selected outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(directory, release_result, same_time_result, out):
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    capsule = directory / "release_capsule.npz"
    require(sha(capsule) == manifest["capsule_sha256"], "Capsule hash")
    with np.load(capsule, allow_pickle=False) as z:
        release, outdegree, times = (z[k] for k in ("release", "outdegree", "query_s"))
        sample_ids, sample_state, sample_target, sample_rate = (
            z[k] for k in ("sample_query_ids", "sample_state", "sample_target", "sample_rate"))
    require(release.shape == (60, 166700) and release.dtype == np.float64, "Release shape")
    require(outdegree.shape == (166700,) and outdegree.dtype == np.int32, "Outdegree shape")
    require(sample_state.shape == sample_target.shape == sample_rate.shape == (4, 359373), "Sample shape")
    require(sample_ids.tolist() == [0, 2, 1, 4], "Sample IDs")
    require(np.isfinite(release).all() and np.isfinite(sample_target).all() and np.isfinite(sample_rate).all(), "Nonfinites")
    require(int(outdegree.sum()) == manifest["n_edges"] == 25582938, "Edge sum")
    older = json.loads(release_result.read_text())
    pair = np.abs(np.diff(release, axis=0))
    span = np.max(release, axis=0) - np.min(release, axis=0)
    checks = {}
    for threshold in (0., 1e-12, 1e-9, 1e-6, 1e-4):
        key = str(threshold)
        count = int(np.count_nonzero(span > threshold))
        edges = int(outdegree[span > threshold].sum(dtype=np.int64))
        observed = older["span_all_queries"][key]
        require(count == observed["source_count"] and edges == observed["edge_count"], "Span mismatch " + key)
        per_pair = [int(outdegree[p > threshold].sum(dtype=np.int64)) for p in pair]
        old_pair = older["adjacent_query_edge_count"][key]
        require(min(per_pair) == old_pair["minimum"] and max(per_pair) == old_pair["maximum"]
                and float(np.median(per_pair)) == old_pair["median"] and sum(per_pair) == old_pair["total"],
                "Adjacent mismatch " + key)
        checks[key] = {"span_sources": count, "span_edges": edges, "adjacent_median_edges": float(np.median(per_pair))}
    pairs = json.loads(same_time_result.read_text())["pairs"]
    for left, right in ((0, 2), (1, 4)):
        where = {int(k): i for i, k in enumerate(sample_ids)}
        old = next(p for p in pairs if p["query_indices"] == [left, right])
        require(times[left] == times[right] and float(times[left]).hex() == old["time_hex"], "Time mismatch")
        for name, values in (("state", sample_state), ("target", sample_target), ("rate", sample_rate)):
            a, b = values[where[left]], values[where[right]]
            difference = np.abs(a - b)
            expected = old[name]
            require(int(np.count_nonzero(a != b)) == expected["changed_coordinates"], "Count mismatch")
            require(float(np.max(difference)) == expected["max_abs"], "Maximum mismatch")
            require(int(np.argmax(difference)) == expected["max_abs_index"], "Position mismatch")
    verdict = {"status": "PASS_PORTABLE_SUBSET", "capsule_sha256": sha(capsule),
               "release_reuse_checks": checks,
               "same_time_pairs_verified": [[0, 2], [1, 4]],
               "limits": manifest["limits"]}
    out.write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capsule", type=Path, required=True)
    p.add_argument("--release-result", type=Path, required=True)
    p.add_argument("--same-time-result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.capsule, a.release_result, a.same_time_result, a.out), indent=2))
