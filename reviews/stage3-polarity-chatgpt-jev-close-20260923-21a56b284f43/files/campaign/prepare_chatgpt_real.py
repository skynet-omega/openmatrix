"""Adapt the lossless four-arm subset to ChatGPT's unmodified input contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("odor_left", "odor_right", "uniform", "sham")
FOCUS = {
    "ORN_DM1": ("ORN", "rootSide"),
    "DM1_lPN": ("PN", "somaSide"),
    "MBON32": ("MBON", "somaSide"),
    "LAL170": ("LAL", "somaSide"),
    "LAL171": ("LAL", "somaSide"),
    "DNa02": ("DNa02", "instance"),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a unique output directory")
    args.out.mkdir(parents=True)
    with np.load(args.subset / "Q_SUBSET.npz", allow_pickle=False) as z:
        ids, times = z["ids"].copy(), z["times_ms"].copy()
        if len(ids) != 9483 or not np.array_equal(times, [0, 1, 20, 40, 60, 100, 160, 220, 320, 335]):
            raise ValueError("Unexpected portable subset")
        arms = {}
        for arm in ARMS:
            path = args.out / f"{arm}.npz"
            np.savez_compressed(path, ids=ids, times_ms=times, q=z["q_" + arm].copy())
            arms[arm] = {"file": path.name, "sha256": sha(path)}
    mapping = {typ: {"stage": stage,
                     "source": "MaleCNS v1.0 type/side annotation, selected prospectively from stimulus and published DNa02 pathway; exploratory readback"}
               for typ, (stage, _) in FOCUS.items()}
    plan = {
        "dataset": "male_v10_166700_archived_PFG_q_subset_9483",
        "pipeline": "historical_PFG_failed_branch_20260919_not_current_protected_runtime",
        "quantity": "q_snapshot_normalized_release_not_spikes_or_signed_flux",
        "unit": "1",
        "time_unit": "ms",
        "keys": {"ids": "ids", "time": "times_ms", "value": "q"},
        "arms": arms,
        "expected_times_ms": times.tolist(),
        "window_ms": [220, 220],
        "side_field": "rootSide",
        "side_by_type": {typ: side for typ, (_, side) in FOCUS.items()},
        "stage_by_type": mapping,
        "max_array_mib": 256,
        "source_subset_sha256": sha(args.subset / "Q_SUBSET.npz"),
        "source_metadata_sha256": sha(args.subset / "NODES_SUBSET.csv"),
        "code_sha256": sha(Path(__file__).with_name("chatgpt_contrastes_original.py")),
    }
    (args.out / "PLAN_CHATGPT.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"plan_sha256": sha(args.out / "PLAN_CHATGPT.json"), "arms": arms}, indent=2))


if __name__ == "__main__":
    main()
