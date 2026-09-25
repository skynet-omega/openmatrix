"""Check whether captured coefficient outputs can be keyed by query time alone."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return z["values"]


def diff(a, b):
    d = np.abs(a - b)
    return {
        "equal_bytes": bool(a.tobytes() == b.tobytes()),
        "changed_coordinates": int(np.count_nonzero(a != b)),
        "max_abs": float(np.max(d)),
        "max_abs_index": int(np.argmax(d)),
    }


def run(plan_file, output_file):
    start = time.monotonic()
    plan = json.loads(plan_file.read_text())
    capture = Path(plan["capture"])
    for name, expected in plan["sha256"].items():
        if digest(capture / name) != expected:
            raise ValueError(f"Capture changed: {name}")
    with np.load(capture / "trace_clock.npz", allow_pickle=False) as z:
        times = z["query_s"]
        if str(z["phase"]) != "accepted" or len(np.unique(z["epoch"])) != 1:
            raise ValueError("Wrong epoch or phase")
    arrays = {name: load(capture / f"trace_{name}.npz") for name in ("state", "target", "rate")}
    if any(a.shape != (60, 359373) or a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Unexpected captured arrays")
    pairs = []
    for t in np.unique(times):
        ids = np.flatnonzero(times == t)
        if len(ids) != 2:
            if len(ids) > 2:
                raise ValueError("Unexpected time multiplicity")
            continue
        i, j = map(int, ids)
        pairs.append({"query_indices": [i, j], "time_hex": float(t).hex(),
                      **{name: diff(a[i], a[j]) for name, a in arrays.items()}})
    result = {"status": "COMPLETE_DESCRIPTIVE_ONLY", "capture_sha256": plan["sha256"],
              "same_time_pairs": len(pairs),
              "pairs_with_changed_target": sum(not p["target"]["equal_bytes"] for p in pairs),
              "pairs_with_changed_rate": sum(not p["rate"]["equal_bytes"] for p in pairs),
              "pairs": pairs, "wall_s": time.monotonic()-start,
              "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "interpretation_limit": "A time-only cache is falsified for this captured RK call sequence; no candidate integrator or speed is evaluated."}
    output_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.out), indent=2, allow_nan=False))
