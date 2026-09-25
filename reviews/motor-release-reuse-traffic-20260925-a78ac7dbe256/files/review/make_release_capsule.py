"""Package a bounded real-data subset for independent release-reuse checks."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(plan_path, out):
    start = time.monotonic()
    plan = json.loads(plan_path.read_text())
    cap = Path(plan["capture"])
    for name, expected in plan["capture_sha256"].items():
        if sha(cap / name) != expected:
            raise ValueError("Input changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    mapping = json.loads((cap / "EFFECTIVE_ARRAY_PATHS.json").read_text())
    with np.load(cap / "effective_gpu_arrays.npz", allow_pickle=False) as z:
        indices = z[mapping["cuda/indices"]]
        n = len(z[mapping["cuda/indptr"]]) - 1
        photo_count = len(z[mapping["cuda/pi"]])
    if n != 166700 or len(indices) != 25582938:
        raise ValueError("Graph identity")
    outdegree = np.bincount(indices, minlength=n).astype(np.int32)
    source_index_sha = hashlib.sha256(indices.tobytes()).hexdigest()
    del indices
    with np.load(cap / "trace_state.npz", allow_pickle=False) as z:
        full_state = z["values"]
    release = np.ascontiguousarray(full_state[:, n + 2 * photo_count : n + 2 * photo_count + n])
    selected = np.array(plan["sample_query_ids"], dtype=np.int32)
    sample_state = full_state[selected].copy()
    del full_state
    with np.load(cap / "trace_target.npz", allow_pickle=False) as z:
        sample_target = z["values"][selected].copy()
    with np.load(cap / "trace_rate.npz", allow_pickle=False) as z:
        sample_rate = z["values"][selected].copy()
    with np.load(cap / "trace_clock.npz", allow_pickle=False) as z:
        times = z["query_s"].copy()
    path = out / "release_capsule.npz"
    np.savez_compressed(path, release=release, outdegree=outdegree,
                        query_s=times, sample_query_ids=selected,
                        sample_state=sample_state, sample_target=sample_target,
                        sample_rate=sample_rate)
    stat = path.stat()
    if stat.st_size > plan["budget"]["capsule_bytes_max"]:
        raise ValueError("Capsule exceeds preregistered size")
    manifest = {"schema": "release_reuse_capsule_v1", "status": "COMPLETE",
                "capture_sha256": plan["capture_sha256"],
                "source_index_array_sha256": source_index_sha,
                "capsule_sha256": sha(path), "capsule_bytes": stat.st_size,
                "n_edges": int(outdegree.sum()),
                "sample_query_ids": selected.tolist(),
                "wall_s": time.monotonic() - start,
                "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "limits": "Outdegree derives from the local full CSR. Capsule reproduces release incidence and sampled same-time outputs; it does not include all graph columns or all target/rate queries."}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.plan, a.out), indent=2))
