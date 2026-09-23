"""Recompute published snapshot contrasts from the portable anatomical subset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_layers import ARMS, TIMES, anatomical_groups, laterality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True,
                        help="Directory with Q_SUBSET.npz, NODES_SUBSET.csv and CONTRASTS.csv")
    args = parser.parse_args()
    p = args.package
    nodes = pd.read_csv(p / "NODES_SUBSET.csv", low_memory=False)
    with np.load(p / "Q_SUBSET.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    if not np.array_equal(nodes["bodyId"].to_numpy(), arrays["ids"]):
        raise ValueError("Portable metadata/array ID mismatch")
    if not np.array_equal(arrays["times_ms"], TIMES):
        raise ValueError("Portable time mismatch")
    side, _ = laterality(nodes)
    groups = anatomical_groups(nodes)
    expected = pd.read_csv(p / "CONTRASTS.csv")
    if len(expected) != len(groups) * len(TIMES):
        raise ValueError("Wrong output count")
    worst = 0.0
    for _, row in expected.iterrows():
        name, time_ms = str(row["group"]), int(row["time_ms"])
        i = int(np.flatnonzero(TIMES == time_ms)[0])
        mask = groups[name]
        left = mask & (side == "L")
        right = mask & (side == "R")
        lr = {arm: float(arrays["q_" + arm][i, left].mean() - arrays["q_" + arm][i, right].mean())
              for arm in ARMS}
        actual = {
            "L_minus_R_sham": lr["sham"],
            "L_minus_R_odor_left": lr["odor_left"],
            "L_minus_R_odor_right": lr["odor_right"],
            "L_minus_R_uniform": lr["uniform"],
            "anti_odor_LminusR": (lr["odor_left"] - lr["odor_right"]) / 2,
            "common_odor_LminusR_vs_sham": (lr["odor_left"] + lr["odor_right"]) / 2 - lr["sham"],
            "uniform_LminusR_vs_sham": lr["uniform"] - lr["sham"],
        }
        if (int(row["n_L"]), int(row["n_R"]), int(row["n_unknown"])) != (
                int(left.sum()), int(right.sum()), int(mask.sum() - left.sum() - right.sum())):
            raise ValueError(f"{name}@{time_ms}: anatomy count mismatch")
        for field, value in actual.items():
            difference = abs(float(row[field]) - value)
            worst = max(worst, difference)
            if difference > 1e-12:
                raise ValueError(f"{name}@{time_ms} {field}: mismatch {difference}")
    print(f"Portable verification: {len(expected)} group-time rows; max absolute difference {worst:.3g}")


if __name__ == "__main__":
    main()
