"""Audited unsigned preparations on canonical MaleCNS v1.0 neuron IDs.

Selection is anatomical, before dynamics or outcome inspection. A VNC view
retains descending/ascending interfaces and every explicit VNC superclass,
including tentative annotations. Its exact canonical boundary is saved, not
silently interpreted as a physiological zero-input condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse


SCHEMA = "matrix_anatomical_preparation_v1"
INTERFACE_CLASSES = frozenset({
    "ascending_neuron", "descending_neuron", "sensory_ascending",
    "sensory_descending", "efferent_ascending", "efferent_descending",
})
CANONICAL_FILES = ("nodes.parquet", "node_ids.npy", "counts_pre_post.npz",
                   "node_connectivity.parquet")
RAW_BOUNDARY_COLUMNS = (
    "input_from_annotated_without_superclass", "input_from_unannotated_ids",
    "output_to_annotated_without_superclass", "output_to_unannotated_ids",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n")


def selection_mask(nodes, scope="vnc_context"):
    if scope == "all":
        return np.ones(len(nodes), dtype=bool)
    if scope != "vnc_context":
        raise ValueError("scope must be 'vnc_context' or 'all'")
    labels = nodes["superclass"].fillna("").astype(str)
    certain_label = labels.str.removesuffix("_tbc")
    return (labels.str.startswith("vnc_") |
            certain_label.isin(INTERFACE_CLASSES)).to_numpy(dtype=bool)


def _count_sum(graph):
    if graph.nnz and int(graph.data.max()) * graph.nnz > np.iinfo(np.int64).max:
        total = sum(map(int, graph.data))
        if total > np.iinfo(np.int64).max:
            raise ValueError("Synapse sum exceeds exact int64 storage")
        return total
    return int(graph.sum())


def _check_counts(graph, shape):
    if not sparse.isspmatrix_csr(graph) or graph.shape != shape:
        raise ValueError("Unexpected CSR shape or format")
    graph.check_format(full_check=True)
    if graph.dtype != np.dtype("int64") or not graph.has_canonical_format:
        raise ValueError("Counts must be canonical int64 CSR")
    if np.any(graph.data < 0):
        raise ValueError("Anatomical counts cannot be negative")
    _count_sum(graph)


def _check_ids(ids, nodes, canonical_indices=None):
    if ids.dtype != np.dtype("int64") or ids.ndim != 1 or not len(ids):
        raise ValueError("Need nonempty one-dimensional int64 IDs")
    if np.any(ids < 0) or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("IDs must be unique and numerically ascending")
    if not np.array_equal(ids, nodes["bodyId"].to_numpy()):
        raise ValueError("Metadata and graph ID order differ")
    if canonical_indices is not None:
        if not np.array_equal(nodes["node_index"].to_numpy(), canonical_indices):
            raise ValueError("Canonical node indices differ from metadata")


@dataclass(frozen=True)
class AnatomicalPreparation:
    node_ids: np.ndarray
    canonical_indices: np.ndarray
    counts_pre_post: sparse.csr_matrix
    nodes: pd.DataFrame
    node_connectivity: pd.DataFrame
    external_node_ids: np.ndarray
    incoming_pre_post: sparse.csr_matrix
    outgoing_pre_post: sparse.csr_matrix
    manifest: dict


def prepare_anatomy(source_dir, output_dir, scope="vnc_context"):
    """Create a new anatomical artifact; never modifies canonical source files."""
    started = time.monotonic()
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be disjoint from the canonical source")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserving existing/partial preparation: {output}")
    provenance_path = source / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("schema") != "matrix_malecns_v10_anatomy_v1":
        raise ValueError("Require the canonical MaleCNS v1.0 import")
    expected = {a["filename"]: a for a in provenance["artifacts"]}
    sources = []
    for name in CANONICAL_FILES:
        path, record = source / name, expected[name]
        actual = sha256(path)
        if actual != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Canonical artifact changed: {name}")
        sources.append({"filename": name, "path": str(path),
                        "sha256": actual, "bytes": path.stat().st_size})
    nodes = pd.read_parquet(source / "nodes.parquet")
    ids = np.load(source / "node_ids.npy", allow_pickle=False)
    graph = sparse.load_npz(source / "counts_pre_post.npz")
    connectivity = pd.read_parquet(source / "node_connectivity.parquet")
    _check_ids(ids, nodes, np.arange(len(ids), dtype=np.int64))
    _check_ids(ids, connectivity, np.arange(len(ids), dtype=np.int64))
    _check_counts(graph, (len(ids), len(ids)))
    graph_total = _count_sum(graph)
    if graph.nnz != provenance["csr"]["stored_pairs"] or graph_total != provenance["csr"]["count_sum"]:
        raise ValueError("Canonical CSR differs from recorded totals")
    for axis, field in [(0, "internal_input_synapses"), (1, "internal_output_synapses")]:
        if not np.array_equal(np.asarray(graph.sum(axis=axis)).ravel(), connectivity[field]):
            raise ValueError(f"Canonical per-node conservation failed: {field}")
    mask = selection_mask(nodes, scope)
    selected, external = np.flatnonzero(mask), np.flatnonzero(~mask)
    if not len(selected):
        raise ValueError("Selected preparation is empty")
    internal = graph[selected, :][:, selected].tocsr()
    incoming = graph[external, :][:, selected].tocsr()
    outgoing = graph[selected, :][:, external].tocsr()
    selected_nodes = nodes.iloc[selected].copy().reset_index(drop=True)
    selected_nodes.insert(0, "preparation_index", np.arange(len(selected), dtype=np.int64))
    con = connectivity.iloc[selected].copy().reset_index(drop=True)
    con.insert(0, "preparation_index", np.arange(len(selected), dtype=np.int64))
    con["preparation_input_synapses"] = np.asarray(internal.sum(axis=0)).ravel()
    con["preparation_output_synapses"] = np.asarray(internal.sum(axis=1)).ravel()
    con["boundary_input_synapses"] = np.asarray(incoming.sum(axis=0)).ravel()
    con["boundary_output_synapses"] = np.asarray(outgoing.sum(axis=1)).ravel()
    if not np.array_equal(con.preparation_input_synapses + con.boundary_input_synapses, con.internal_input_synapses):
        raise ValueError("Preparation input boundary does not conserve counts")
    if not np.array_equal(con.preparation_output_synapses + con.boundary_output_synapses, con.internal_output_synapses):
        raise ValueError("Preparation output boundary does not conserve counts")
    partition = {name: {"pairs": int(g.nnz), "synapses": _count_sum(g)}
                 for name, g in [("internal", internal), ("incoming", incoming), ("outgoing", outgoing)]}
    partition["external_to_external"] = {
        "pairs": int(graph.nnz) - sum(p["pairs"] for p in partition.values()),
        "synapses": graph_total - sum(p["synapses"] for p in partition.values()),
    }
    if any(v < 0 for p in partition.values() for v in p.values()):
        raise ValueError("Invalid graph partition")
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "node_ids.npy", ids[selected])
    np.save(output / "canonical_indices.npy", selected.astype(np.int64))
    np.save(output / "external_node_ids.npy", ids[external])
    selected_nodes.to_parquet(output / "nodes.parquet", index=False)
    con.to_parquet(output / "node_connectivity.parquet", index=False)
    sparse.save_npz(output / "counts_pre_post.npz", internal)
    sparse.save_npz(output / "incoming_pre_post.npz", incoming)
    sparse.save_npz(output / "outgoing_pre_post.npz", outgoing)
    manifest = {
        "schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope, "scientific_status": "ANATOMICAL_PREPARATION_ONLY",
        "selection_rule": "All canonical nodes" if scope == "all" else
            "Every superclass beginning vnc_, plus ascending_neuron, descending_neuron, sensory_ascending, sensory_descending, efferent_ascending, efferent_descending and their _tbc variants. No threshold, outcome or graph-degree selection.",
        "orientation": "row=pre, column=post; activity propagates with C.T @ rates",
        "nodes": len(selected), "canonical_nodes": len(ids), "external_nodes": len(external),
        "superclass_counts": {str(k): int(v) for k, v in selected_nodes.superclass.value_counts(dropna=False).items()},
        "canonical_partition": partition,
        "original_noncanonical_boundary_totals": {c: int(con[c].sum()) for c in RAW_BOUNDARY_COLUMNS},
        "sources": sources, "source_provenance_sha256": sha256(provenance_path),
        "source_release": provenance["release"], "source_module_sha256": sha256(Path(__file__)),
        "conservation": {"canonical_counts": True, "per_node_inputs": True, "per_node_outputs": True},
        "limitations": [
            "Superclass-based VNC context is not a neuropil-ROI-specific reproduction of the Pugliese paper.",
            "Boundary matrices contain canonical partners only; noncanonical original endpoints remain per-node aggregate counts.",
            "The external-to-external graph is accounted for, not included in a VNC artifact. Use scope=all for the complete canonical graph.",
            "No neurotransmitter sign, physiological parameter, size proxy, boundary activity, stimulus, motor-to-muscle map, or body controller is assigned here.",
            "Missing boundary activity is unknown, not physiological rest; a runtime must declare its boundary condition.",
        ],
        "elapsed_seconds": time.monotonic() - started,
    }
    manifest["artifacts"] = [{"filename": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
                             for p in sorted(output.iterdir()) if p.is_file()]
    _json(output / "manifest.json", manifest)
    return manifest


def load_preparation(path, verify_hashes=True):
    """Load a self-contained anatomical artifact without the original dataset."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported anatomical preparation schema")
    required = {"nodes.parquet", "node_connectivity.parquet", "node_ids.npy",
                "canonical_indices.npy", "external_node_ids.npy", "counts_pre_post.npz",
                "incoming_pre_post.npz", "outgoing_pre_post.npz"}
    if {a["filename"] for a in manifest["artifacts"]} != required:
        raise ValueError("Incomplete preparation artifact manifest")
    if verify_hashes:
        for artifact in manifest["artifacts"]:
            p = path / artifact["filename"]
            if p.stat().st_size != artifact["bytes"] or sha256(p) != artifact["sha256"]:
                raise ValueError(f"Preparation artifact changed: {p.name}")
    ids = np.load(path / "node_ids.npy", allow_pickle=False)
    canonical = np.load(path / "canonical_indices.npy", allow_pickle=False)
    ext = np.load(path / "external_node_ids.npy", allow_pickle=False)
    nodes = pd.read_parquet(path / "nodes.parquet")
    con = pd.read_parquet(path / "node_connectivity.parquet")
    _check_ids(ids, nodes, canonical)
    _check_ids(ids, con, canonical)
    if not np.array_equal(nodes.preparation_index, np.arange(len(ids))):
        raise ValueError("Preparation indices do not match rows")
    if canonical.dtype != np.dtype("int64") or canonical.shape != ids.shape or np.any(canonical[1:] <= canonical[:-1]):
        raise ValueError("Invalid canonical index mapping")
    if ext.dtype != np.dtype("int64") or ext.ndim != 1 or np.any(ext[1:] <= ext[:-1]) or np.intersect1d(ext, ids).size:
        raise ValueError("Invalid external ID mapping")
    graphs = [sparse.load_npz(path / name) for name in
              ["counts_pre_post.npz", "incoming_pre_post.npz", "outgoing_pre_post.npz"]]
    for graph, shape in zip(graphs, [(len(ids), len(ids)), (len(ext), len(ids)), (len(ids), len(ext))]):
        _check_counts(graph, shape)
    for graph, key in zip(graphs, ["internal", "incoming", "outgoing"]):
        if graph.nnz != manifest["canonical_partition"][key]["pairs"] or _count_sum(graph) != manifest["canonical_partition"][key]["synapses"]:
            raise ValueError(f"Preparation totals differ: {key}")
    internal, incoming, outgoing = graphs
    if not np.array_equal(np.asarray(internal.sum(axis=0) + incoming.sum(axis=0)).ravel(), con.internal_input_synapses):
        raise ValueError("Loaded input boundary fails conservation")
    if not np.array_equal(np.asarray(internal.sum(axis=1) + outgoing.sum(axis=1)).ravel(), con.internal_output_synapses):
        raise ValueError("Loaded output boundary fails conservation")
    return AnatomicalPreparation(ids, canonical, internal, nodes, con, ext, incoming, outgoing, manifest)


def prepare_functional_ports(preparation_dir, output_dir):
    """Expose literal anatomical groups and existing candidate plastic edges.

    These groups do not define sensory physiology, a steering law, dopamine
    valence, or a synaptic update rule. Compartment labels identify candidate
    populations; they do not locate receptors on individual contact sites.
    """
    preparation_dir, output = Path(preparation_dir).resolve(), Path(output_dir).resolve()
    if preparation_dir == output or preparation_dir in output.parents or output in preparation_dir.parents:
        raise ValueError("Ports output must be disjoint from the preparation")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Preserving existing/partial port artifact: {output}")
    p = load_preparation(preparation_dir)
    d = p.nodes
    ports, rules = {}, {}
    for side in ["L", "R"]:
        orn = d.type.eq("ORN_DM1") & d.superclass.eq("cb_sensory") & d.rootSide.eq(side)
        if not d.loc[orn, "instance"].eq(f"ORN_DM1_{side}").all():
            raise ValueError("ORN_DM1 instance/rootSide annotations conflict")
        name = f"ORN_DM1_{side}"
        ports[name] = d.loc[orn, "bodyId"].astype(int).tolist()
        rules[name] = f"type=ORN_DM1, superclass=cb_sensory, rootSide={side}, instance cross-checked"
        for cell_type in ["DNa02", "DNg100"]:
            name = f"{cell_type}_{side}"
            ports[name] = d.loc[d.type.eq(cell_type) & d.instance.eq(name), "bodyId"].astype(int).tolist()
            rules[name] = "Exact type and complete instance annotation; output-side physiology not inferred from soma"
            if len(ports[name]) != 1:
                raise ValueError(f"Expected exactly one explicitly annotated {name}")
        pam = d.type.fillna("").str.match(r"^PAM08(?:_|$)") & d["class"].eq("DAN") & d.instance.fillna("").str.endswith(f"(y4)_{side}")
        name = f"PAM08_{side}"
        ports[name] = d.loc[pam, "bodyId"].astype(int).tolist()
        rules[name] = f"type=PAM08 or PAM08_ subtype, class=DAN, instance suffix=(y4)_{side}"
        name = f"MBON05_{side}"
        ports[name] = d.loc[d.type.eq("MBON05") & d.instance.eq(f"MBON05(y4>y1y2)_{side}"), "bodyId"].astype(int).tolist()
        rules[name] = f"Exact type=MBON05 and instance=MBON05(y4>y1y2)_{side}"
        if not ports[f"ORN_DM1_{side}"] or not ports[f"PAM08_{side}"] or len(ports[name]) != 1:
            raise ValueError(f"Missing or ambiguous functional population on side {side}")
    kc_mask = d["class"].eq("Kenyon_Cell") & d.type.fillna("").str.startswith("KCg")
    kc_rows = np.flatnonzero(kc_mask.to_numpy())
    ports["KC_gamma"] = d.loc[kc_mask, "bodyId"].astype(int).tolist()
    rules["KC_gamma"] = "class=Kenyon_Cell and type begins KCg; all annotated gamma subclasses retained"
    ports["vnc_motor_all"] = d.loc[d.superclass.eq("vnc_motor"), "bodyId"].astype(int).tolist()
    rules["vnc_motor_all"] = "Exact superclass=vnc_motor; no motor-to-muscle physiology inferred"
    for segment in ["T1", "T2", "T3"]:
        for side in ["L", "R"]:
            name = f"vnc_motor_{segment}_{side}"
            mask = d.superclass.eq("vnc_motor") & d.somaNeuromere.eq(segment) & d.somaSide.eq(side)
            ports[name] = d.loc[mask, "bodyId"].astype(int).tolist()
            rules[name] = "Soma location grouping for observation only; includes all thoracic motor annotations, not exclusively leg muscles"
    index = {int(identifier): i for i, identifier in enumerate(p.node_ids)}
    def edges(rows, cols):
        coo = p.counts_pre_post[rows, :][:, cols].tocoo()
        pre, post = rows[coo.row], cols[coo.col]
        return pd.DataFrame({"pre_id": p.node_ids[pre], "post_id": p.node_ids[post],
                             "preparation_pre_index": pre, "preparation_post_index": post,
                             "canonical_pre_index": p.canonical_indices[pre], "canonical_post_index": p.canonical_indices[post],
                             "synapse_count": coo.data}).sort_values(["pre_id", "post_id"]).reset_index(drop=True)
    plastic_edges, modulation_edges = [], []
    for side in ["L", "R"]:
        mb_rows = np.array([index[i] for i in ports[f"MBON05_{side}"]], dtype=np.int64)
        pam_rows = np.array([index[i] for i in ports[f"PAM08_{side}"]], dtype=np.int64)
        targets = edges(kc_rows, mb_rows)
        targets["modulator_port"] = f"PAM08_{side}"
        targets["compartment_label"] = "gamma4"
        plastic_edges.append(targets)
        for target, rows in [("KC_gamma", kc_rows), ("MBON05", mb_rows)]:
            rows_df = edges(pam_rows, rows)
            rows_df["target_population"] = target
            rows_df["modulator_port"] = f"PAM08_{side}"
            modulation_edges.append(rows_df)
    plastic_edges = pd.concat(plastic_edges, ignore_index=True)
    modulation_edges = pd.concat(modulation_edges, ignore_index=True)
    if plastic_edges.empty:
        raise ValueError("No anatomical KC_gamma-to-MBON05 candidate synapses")
    output.mkdir(parents=True, exist_ok=True)
    plastic_edges.to_csv(output / "plasticity_candidate_edges.csv", index=False)
    modulation_edges.to_csv(output / "modulator_anatomical_edges.csv", index=False)
    selected_ids = sorted(set(i for values in ports.values() for i in values))
    d.loc[d.bodyId.isin(selected_ids)].to_parquet(output / "port_nodes.parquet", index=False)
    manifest = {
        "schema": "matrix_anatomical_ports_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_manifest_sha256": sha256(preparation_dir / "manifest.json"),
        "selection_rules": rules, "ports": ports,
        "population_counts": {name: len(ids) for name, ids in ports.items()},
        "plasticity_candidate_pairs": len(plastic_edges),
        "plasticity_candidate_contacts": int(plastic_edges.synapse_count.sum()),
        "modulator_anatomical_pairs": len(modulation_edges),
        "modulator_anatomical_contacts": int(modulation_edges.synapse_count.sum()),
        "source_module_sha256": sha256(Path(__file__)),
        "limitations": [
            "Named compartment selection is anatomical metadata, not a demonstrated learning rule or measured receptor-level dopamine action.",
            "No foreign neuron IDs, weights, model activity, sensory transduction, reward or steering signal is imported.",
            "ORN side uses rootSide confirmed by instance; DNa02/DNg100/MBON05/PAM08 use explicit instance labels, not guessed output side from soma.",
            "No cut, shuffle, synaptic scaling or sign has been applied to canonical counts.",
        ],
    }
    manifest["artifacts"] = [{"filename": f.name, "bytes": f.stat().st_size, "sha256": sha256(f)}
                             for f in sorted(output.iterdir()) if f.is_file()]
    _json(output / "ports.json", manifest)
    return manifest
