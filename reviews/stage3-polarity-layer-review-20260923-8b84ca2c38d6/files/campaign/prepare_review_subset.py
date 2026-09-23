"""Create a portable, read-only subset of four full-brain q snapshots for review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_layers import ARMS, NODES, SOURCE, TIMES, anatomical_groups, check, sha

FOCUS = ("ORN_DM1", "DM1_lPN", "MBON32", "LAL170", "LAL171", "DNa02")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--nodes", type=Path, default=NODES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(not args.out.exists(), "Use a unique output directory")
    all_ids = np.load(args.nodes.parent / "node_ids.npy", allow_pickle=False)
    meta = pd.read_parquet(args.nodes).set_index("bodyId").loc[all_ids].reset_index()
    masks = anatomical_groups(meta)
    selected = np.logical_or.reduce(list(masks.values()))
    indices = np.flatnonzero(selected)
    subset_ids = all_ids[indices]
    cols = ["bodyId", "type", "class", "instance", "rootSide", "somaSide", "nt_consensus_nt"]
    subset_meta = meta.loc[indices, cols].copy()
    check(np.array_equal(subset_meta["bodyId"].to_numpy(), subset_ids), "Subset metadata mismatch")

    arrays = {"ids": subset_ids, "times_ms": TIMES}
    source_hashes = {"nodes": sha(args.nodes), "node_ids": sha(args.nodes.parent / "node_ids.npy")}
    for arm in ARMS:
        path = args.source / f"{arm}_network_trace.npz"
        with np.load(path, allow_pickle=False) as data:
            check(np.array_equal(data["ids"], all_ids), f"{arm}: source IDs mismatch")
            check(np.array_equal(data["times_ms"], TIMES), f"{arm}: source times mismatch")
            arrays["q_" + arm] = data["q"][:, indices].copy()
        source_hashes[arm] = sha(path)

    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "Q_SUBSET.npz", **arrays)
    subset_meta.to_csv(args.out / "NODES_SUBSET.csv", index=False)
    rows = []
    for cell_type in FOCUS:
        mask = subset_meta["type"].astype("string").eq(cell_type).fillna(False).to_numpy()
        cell_indices = np.flatnonzero(mask)
        check(len(cell_indices) > 0, f"Missing focus type {cell_type}")
        for side in ("L", "R"):
            x = subset_meta.iloc[cell_indices]
            side_col = "rootSide" if cell_type.startswith("ORN_") else "somaSide"
            chosen = cell_indices[(x[side_col] == side).fillna(False).to_numpy()]
            check(len(chosen) > 0, f"Missing {cell_type} {side}")
            for t_index, t in enumerate(TIMES):
                values = {arm: float(arrays["q_" + arm][t_index, chosen].mean()) for arm in ARMS}
                rows.append({"type": cell_type, "side": side, "time_ms": int(t), "n": len(chosen),
                             **values, "anti_odor": (values["odor_left"] - values["odor_right"]) / 2,
                             "common_vs_sham": (values["odor_left"] + values["odor_right"]) / 2 - values["sham"]})
    pd.DataFrame(rows).to_csv(args.out / "FOCUS.csv", index=False, float_format="%.17g")
    provenance = {"schema": "stage3_review_subset_v1", "source_sha256": source_hashes,
                  "subset_cells": int(len(indices)), "focus": FOCUS,
                  "scope": "Lossless subset of stored q samples; no edge transmission, current-pipeline state or causal simulation.",
                  "post_hoc_status": "Focus readback chosen after broad group analysis; do not use as confirmatory selection."}
    (args.out / "SUBSET_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"subset_cells": len(indices), "focus_rows": len(rows),
                      "files_sha256": {p.name: sha(p) for p in args.out.iterdir()}}, indent=2))


if __name__ == "__main__":
    main()
