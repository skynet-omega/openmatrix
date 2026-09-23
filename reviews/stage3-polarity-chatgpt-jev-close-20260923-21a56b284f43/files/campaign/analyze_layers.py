"""Read-only anatomical q contrasts in four archived full-network traces.

Cell-body side is only a laterality proxy beyond ORNs; this is not a flux or
causal path analysis. The PFG intervention in the source campaign failed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("odor_left", "odor_right", "uniform", "sham")
TIMES = np.array([0, 1, 20, 40, 60, 100, 160, 220, 320, 335], dtype=np.int32)
SOURCE = Path("/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live")
NODES = Path("/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def anatomical_groups(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    typ = nodes["type"].astype("string")
    cls = nodes["class"].astype("string")
    masks = {
        "ORN": typ.str.startswith("ORN_", na=False),
        "ALPN": cls.eq("ALPN").fillna(False),
        "ALLN": cls.eq("ALLN").fillna(False),
        "LH_type_prefix": typ.str.startswith("LH", na=False),
        "MBON": cls.eq("MBON").fillna(False),
        "CX": cls.eq("CX").fillna(False),
        "LAL_type_prefix": typ.str.startswith("LAL", na=False),
        "DNa02": typ.eq("DNa02").fillna(False),
    }
    return {name: mask.to_numpy(dtype=bool) for name, mask in masks.items()}


def laterality(nodes: pd.DataFrame) -> tuple[np.ndarray, dict]:
    root = nodes["rootSide"].astype("string")
    soma = nodes["somaSide"].astype("string")
    suffix = nodes["instance"].astype("string").str.extract(r"_([LR])$")[0]
    conflict = soma.isin(["L", "R"]) & suffix.isin(["L", "R"]) & (soma != suffix)
    check(not bool(conflict.fillna(False).any()), "Soma/instance side conflict")
    root_conflict = (root.isin(["L", "R"]) & suffix.isin(["L", "R"]) & (root != suffix)).fillna(False)
    # The ORN sensory-root side is preferred. For other cells soma side is a
    # documented proxy, not proof of an ipsilateral functional projection.
    side = root.where(root.isin(["L", "R"]), soma.where(soma.isin(["L", "R"]), "unknown"))
    side = side.mask(root_conflict, "unknown")
    return side.to_numpy(dtype=str), {
        "root_side_known": int(root.isin(["L", "R"]).sum()),
        "soma_fallback_known": int((~root.isin(["L", "R"]) & soma.isin(["L", "R"])).sum()),
        "soma_instance_conflicts": int(conflict.fillna(False).sum()),
        "root_instance_conflicts_excluded": int(root_conflict.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--nodes", type=Path, default=NODES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check(not args.out.exists(), "Output directory exists; use a unique run directory")

    graph_ids_path = args.nodes.parent / "node_ids.npy"
    graph_ids = np.load(graph_ids_path, allow_pickle=False)
    nodes = pd.read_parquet(args.nodes)
    check(len(nodes) == 166700 and nodes["bodyId"].is_unique, "Invalid node metadata")
    nodes = nodes.set_index("bodyId").loc[graph_ids].reset_index()
    check(np.array_equal(nodes["bodyId"].to_numpy(), graph_ids), "Node ID order mismatch")
    groups = anatomical_groups(nodes)
    side, side_provenance = laterality(nodes)

    traces: dict[str, np.ndarray] = {}
    input_hashes = {"nodes": sha(args.nodes), "node_ids": sha(graph_ids_path)}
    observed_max_diff = {}
    for arm in ARMS:
        network = args.source / f"{arm}_network_trace.npz"
        trace = args.source / f"{arm}_trace.npz"
        with np.load(network, allow_pickle=False) as data:
            check(set(data.files) == {"ids", "times_ms", "q"}, f"{arm}: unexpected fields")
            check(np.array_equal(data["ids"], graph_ids), f"{arm}: ID order mismatch")
            check(np.array_equal(data["times_ms"], TIMES), f"{arm}: time mismatch")
            q = data["q"].copy()
        check(q.shape == (10, 166700) and bool(np.isfinite(q).all()), f"{arm}: invalid q")
        with np.load(trace, allow_pickle=False) as data:
            ids = data["monitored_ids"]
            ms = data["ms"]
            check(np.array_equal(ms, np.arange(336)), f"{arm}: incomplete observer timeline")
            ix = pd.Index(graph_ids).get_indexer(ids)
            check(bool((ix >= 0).all()), f"{arm}: monitored ID absent")
            differences = np.abs(q[:, ix] - data["monitored_q"][TIMES])
            observed_max_diff[arm] = float(np.max(differences))
            check(observed_max_diff[arm] <= 1e-12, f"{arm}: snapshot/observer mismatch")
        traces[arm] = q
        input_hashes[f"{arm}_network"] = sha(network)
        input_hashes[f"{arm}_trace"] = sha(trace)

    records = []
    group_metadata = {}
    for group, mask in groups.items():
        left = np.flatnonzero(mask & (side == "L"))
        right = np.flatnonzero(mask & (side == "R"))
        unknown = int(np.sum(mask & ~np.isin(side, ["L", "R"])))
        group_metadata[group] = {"total": int(mask.sum()), "left": len(left),
                                 "right": len(right), "unknown": unknown,
                                 "side_semantics": "sensory root" if group == "ORN" else "cell-body proxy"}
        check(len(left) > 0 and len(right) > 0, f"{group}: missing side")
        for t_index, t in enumerate(TIMES):
            lr = {}
            values = {}
            for arm in ARMS:
                q = traces[arm][t_index]
                values[arm] = {"L": float(q[left].mean()), "R": float(q[right].mean())}
                lr[arm] = values[arm]["L"] - values[arm]["R"]
            records.append({
                "group": group, "time_ms": int(t),
                "n_L": len(left), "n_R": len(right), "n_unknown": unknown,
                "L_minus_R_sham": lr["sham"],
                "L_minus_R_odor_left": lr["odor_left"],
                "L_minus_R_odor_right": lr["odor_right"],
                "L_minus_R_uniform": lr["uniform"],
                "anti_odor_LminusR": (lr["odor_left"] - lr["odor_right"]) / 2,
                "common_odor_LminusR_vs_sham": (lr["odor_left"] + lr["odor_right"]) / 2 - lr["sham"],
                "uniform_LminusR_vs_sham": lr["uniform"] - lr["sham"],
                "left_side_anti_odor": (values["odor_left"]["L"] - values["odor_right"]["L"]) / 2,
                "right_side_anti_odor": (values["odor_left"]["R"] - values["odor_right"]["R"]) / 2,
            })
    output = {
        "schema": "stage3_anatomical_snapshot_contrast_v1",
        "precommitted_primary_snapshot_ms": 220,
        "source_limit": "Archived failed PFG branch, one preparation, sparse q snapshots. No causal localization, direct signed synaptic flow, or live-fly match.",
        "anatomical_limit": "LH/LAL type-prefix groups are candidate sets; CX overlaps some LAL. Soma side is a proxy beyond ORN and bilateral axons may cross.",
        "group_metadata": group_metadata,
        "side_provenance": side_provenance,
        "snapshot_observer_max_abs_diff": observed_max_diff,
        "input_sha256": input_hashes,
    }
    args.out.mkdir(parents=True)
    pd.DataFrame.from_records(records).to_csv(args.out / "CONTRASTS.csv", index=False, float_format="%.17g")
    (args.out / "RESULT.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"primary_220ms": [r for r in records if r["time_ms"] == 220],
                      "group_metadata": group_metadata}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
