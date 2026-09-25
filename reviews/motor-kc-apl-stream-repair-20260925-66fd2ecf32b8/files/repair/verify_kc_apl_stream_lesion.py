"""Recompute the stream-repaired KC/APL lesion from its archived arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def need(ok, message):
    if not ok:
        raise ValueError(message)


def run(directory, out):
    report = json.loads((directory / "LESION_RESULT.json").read_text())
    path = directory / "ABA_ARRAYS.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    need(digest == report["aba_arrays_sha256"], "A/B/A array hash")
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    ids = arrays["query_indices"]
    n = 359373
    need(ids.tolist() == [2, report["current_capture_queries"] - 1], "Query identity")
    need(ids.tolist() == report["chosen_query_indices"], "Receipt query identity")
    for name in ("consumed_state", "consumed_target", "consumed_rate", "baseline_z",
                 "baseline_target", "baseline_rate", "lesion_target", "lesion_rate",
                 "restored_target", "restored_rate"):
        a = arrays[name]
        need(a.shape == (2, n) and a.dtype == np.float64 and np.isfinite(a).all(), "Array layout: " + name)
    for lhs, rhs in (("consumed_state", "baseline_z"),
                     ("consumed_target", "baseline_target"),
                     ("consumed_rate", "baseline_rate"),
                     ("baseline_target", "lesion_target"),
                     ("baseline_rate", "lesion_rate"),
                     ("baseline_target", "restored_target"),
                     ("baseline_rate", "restored_rate")):
        need(np.array_equal(arrays[lhs], arrays[rhs]), "Output mismatch: " + lhs + "/" + rhs)
    before, lesion, restored = (arrays[k] for k in ("weights_before", "weights_lesion", "weights_restored"))
    pos = arrays["selected_positions"]
    need(pos.ndim == 1 and len(pos) == 836605 and len(np.unique(pos)) == len(pos), "Position layout")
    need(before.shape == lesion.shape == restored.shape == pos.shape, "Weight layout")
    need(np.isfinite(before).all() and np.isfinite(restored).all(), "Weight nonfinites")
    need(int(np.count_nonzero(before)) == report["selected_nonzero_weights"] == 698768,
         "Nonzero weight count")
    need(int(np.count_nonzero(lesion)) == 0 and np.array_equal(before, restored),
         "Weight lesion or restoration")
    need(report["status"] == "PASS_LIMITED" and report["runner_exit_code"] == 0
         and report["budget_ok"] and report["frozen_gates_ok"], "Runner/contract result")
    result = {"status": "PASS_ARRAY_RECOMPUTE_LIMITED", "aba_arrays_sha256": digest,
              "query_indices": ids.tolist(), "selected_edges": len(pos),
              "zeroed_nonzero_weights": int(np.count_nonzero(before)),
              "coefficient_coordinates_compared_per_arm": 2 * 2 * n,
              "all_compared_values_equal": True,
              "scope": "Two in-run query states in one neutral 125-us accepted CNS block; no pruned-kernel speed or long behavior."}
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.result, a.out), indent=2, allow_nan=False))
