"""Cold verifier of portable XOR release vectors and two sampled full outputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def run(directory, release_result, same_time_result, out):
    manifest = json.loads((directory / "MANIFEST.json").read_text())
    capsule = directory / "release_capsule_xor.npz"
    need(sha(capsule) == manifest["capsule_sha256"], "Capsule hash")
    with np.load(capsule, allow_pickle=False) as z:
        first, xor, degree, times = (z[k] for k in ("release_first_bits", "release_adjacent_xor", "outdegree", "query_s"))
        sample_ids, sample_state, sample_target, sample_rate = (
            z[k] for k in ("sample_query_ids", "sample_state", "sample_target", "sample_rate"))
    need(first.shape == (166700,) and xor.shape == (59, 166700) and first.dtype == xor.dtype == np.uint64, "XOR layout")
    bits = np.empty((60, 166700), dtype=np.uint64)
    bits[0] = first
    bits[1:] = np.bitwise_xor.accumulate(xor, axis=0) ^ first
    release = bits.view(np.float64)
    need(hashlib.sha256(release.tobytes()).hexdigest() == manifest["release_uncompressed_sha256"], "Decoded release hash")
    need(degree.shape == (166700,) and degree.dtype == np.int32, "Outdegree shape")
    need(int(degree.sum()) == manifest["n_edges"] == 25582938, "Edge count")
    need(sample_ids.tolist() == [0, 2, 1, 4] and sample_state.shape == sample_target.shape == sample_rate.shape == (4, 359373), "Sample layout")
    need(np.isfinite(release).all() and np.isfinite(sample_target).all() and np.isfinite(sample_rate).all(), "Nonfinites")
    old = json.loads(release_result.read_text())
    span = np.max(release, axis=0) - np.min(release, axis=0)
    adjacent = np.abs(np.diff(release, axis=0))
    checks = {}
    for level in (0., 1e-12, 1e-9, 1e-6, 1e-4):
        key = str(level)
        count = int(np.count_nonzero(span > level))
        edges = int(degree[span > level].sum(dtype=np.int64))
        expected = old["span_all_queries"][key]
        need((count, edges) == (expected["source_count"], expected["edge_count"]), "Span " + key)
        visits = [int(degree[row > level].sum(dtype=np.int64)) for row in adjacent]
        expected = old["adjacent_query_edge_count"][key]
        need((min(visits), float(np.median(visits)), max(visits), sum(visits)) ==
             (expected["minimum"], expected["median"], expected["maximum"], expected["total"]), "Adjacency " + key)
        checks[key] = {"span_sources": count, "span_edges": edges, "median_adjacent_edges": float(np.median(visits))}
    old_pairs = json.loads(same_time_result.read_text())["pairs"]
    lookup = {int(k): i for i, k in enumerate(sample_ids)}
    for left, right in ((0, 2), (1, 4)):
        expected = next(p for p in old_pairs if p["query_indices"] == [left, right])
        need(times[left] == times[right] and float(times[left]).hex() == expected["time_hex"], "Sample time")
        for name, values in (("state", sample_state), ("target", sample_target), ("rate", sample_rate)):
            a, b = values[lookup[left]], values[lookup[right]]
            difference = np.abs(a - b)
            stats = expected[name]
            need((int(np.count_nonzero(a != b)), float(np.max(difference)), int(np.argmax(difference))) ==
                 (stats["changed_coordinates"], stats["max_abs"], stats["max_abs_index"]), "Sample " + name)
    receipt = {"status": "PASS_PORTABLE_SUBSET", "capsule_sha256": sha(capsule),
               "release_reuse_checks": checks, "same_time_pairs_verified": [[0, 2], [1, 4]],
               "limits": manifest["limits"]}
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--capsule", type=Path, required=True)
    p.add_argument("--release-result", type=Path, required=True)
    p.add_argument("--same-time-result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.capsule, a.release_result, a.same_time_result, a.out), indent=2))
