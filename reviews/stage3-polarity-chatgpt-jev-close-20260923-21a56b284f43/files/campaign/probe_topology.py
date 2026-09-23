"""Exploratory unsigned anatomical path counts; not neural flux or causal evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hashlib import sha256

DEFAULT = Path("/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10")


def digest(path):
    h = sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("Use a unique output file")
    node_file = a.data / "nodes.parquet"
    id_file = a.data / "node_ids.npy"
    graph_file = a.data / "counts_pre_post.npz"
    nodes = pd.read_parquet(node_file, columns=["bodyId", "type", "instance", "rootSide", "somaSide"])
    ids = np.load(id_file, allow_pickle=False)
    if len(ids) != 166700 or not np.array_equal(nodes.bodyId.to_numpy(), ids):
        raise ValueError("Node ID order mismatch")
    with np.load(graph_file, allow_pickle=False) as z:
        ptr, cols, weights = z["indptr"], z["indices"], z["data"]
    if ptr.shape != (166701,) or len(cols) != len(weights) or int(ptr[-1]) != len(cols):
        raise ValueError("Anatomical graph layout mismatch")

    cell_type = nodes["type"].astype("string")
    group = {name: np.flatnonzero(cell_type.eq(name).fillna(False).to_numpy())
             for name in ("ORN_DM1", "DM1_lPN", "MBON32", "LAL170", "LAL171", "DNa02")}
    group["LAL170_171"] = np.r_[group["LAL170"], group["LAL171"]]
    for name, selected in group.items():
        if len(selected) == 0:
            raise ValueError(f"Missing {name}")

    def row(i):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        c = cols[lo:hi]
        if len(c) > 1 and np.any(c[1:] <= c[:-1]):
            raise ValueError(f"Unsorted/duplicate anatomical row {i}")
        return c, weights[lo:hi]

    def edges(sources, targets):
        found = []
        for i in sources:
            neighbors, counts = row(i)
            for j in targets:
                pos = int(np.searchsorted(neighbors, j))
                if pos < len(neighbors) and int(neighbors[pos]) == int(j):
                    found.append({"pre_id": int(ids[i]), "post_id": int(ids[j]),
                                  "count": int(counts[pos])})
        return found

    pairs = {}
    for src, dst in (("ORN_DM1", "DM1_lPN"), ("DM1_lPN", "MBON32"),
                     ("DM1_lPN", "DNa02"), ("MBON32", "LAL170_171"),
                     ("MBON32", "DNa02"), ("LAL170_171", "DNa02")):
        match = edges(group[src], group[dst])
        pairs[src + "__to__" + dst] = {"edges": len(match),
                                      "unsigned_synapse_count": sum(x["count"] for x in match),
                                      "details": match}
    sides = {}
    for s in ("L", "R"):
        source = [i for i in group["ORN_DM1"] if nodes.at[i, "rootSide"] == s]
        for t in ("L", "R"):
            target = [i for i in group["DM1_lPN"] if nodes.at[i, "somaSide"] == t]
            match = edges(source, target)
            sides[s + "_to_" + t] = {"ORN_count": len(source),
                                     "connections": len(match),
                                     "unsigned_synapse_count": sum(x["count"] for x in match)}
    bridges = {}
    for i in group["DM1_lPN"]:
        found = []
        neighbors, counts = row(i)
        for j, first_count in zip(neighbors, counts):
            for last in edges([j], group["DNa02"]):
                found.append({"middle_id": int(ids[j]), "middle_type": str(nodes.at[j, "type"]),
                              "first_count": int(first_count), "second_count": last["count"],
                              "target_id": last["post_id"]})
        bridges[str(ids[i])] = found
    result = {"schema": "stage3_unsigned_anatomical_probe_v1",
              "status": "post_hoc_exploratory",
              "limit": "Raw count graph is unsigned and may differ from the effective simulated operator. Two-hop existence is neither active flow nor proof of behavioral causation.",
              "sources_sha256": {name: digest(path) for name, path in
                                 (("nodes", node_file), ("node_ids", id_file), ("counts_pre_post", graph_file))},
              "group_sizes": {name: len(v) for name, v in group.items()},
              "ORN_DM1_to_DM1_lPN_by_side": sides,
              "selected_pairs": pairs,
              "DM1_lPN_two_hop_to_DNa02": bridges}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"by_side": sides, "pair_totals": {k: {f: v[f] for f in ("edges", "unsigned_synapse_count")}
                 for k, v in pairs.items()}, "two_hop_counts": {k: len(v) for k, v in bridges.items()}}, indent=2))


if __name__ == "__main__":
    main()
