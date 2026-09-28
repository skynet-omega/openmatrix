"""Candidate gamma4 plasticity on existing MaleCNS KC-to-MBON05 edges.

This is an explicit engineering hypothesis motivated by compartmental dopamine
plasticity, not a measured equation, a reproduction of Cohn/Hige, or validated
learning. Eligibility has a fixed 1 s time constant. Contact-weighted PAM08
activity depresses existing KC->MBON05 weights at candidate rate 0.1/s.
There is no inferred reward valence, target direction, weight floor or rescue.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


SCHEMA = "matrix_candidate_gamma_plasticity_v1"
ELIGIBILITY_TAU_S = 1.0
DEPRESSION_RATE_PER_S = 0.1
STATE_FIELDS = frozenset({
    "schema", "enabled", "time_ns", "update_count", "sources", "topology_sha256",
    "pre_ids", "post_ids", "pre_indices", "post_indices", "positions", "base_weights",
    "factors", "eligibility", "dan_ids", "dan_indices", "dan_weight_indptr",
    "dan_weight_indices", "dan_weight_data", "dan_weight_shape",
})


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _matrix(brain):
    matrix = getattr(brain, "W", None)
    if matrix is None:
        matrix = getattr(brain, "Wcsr", None)
    if not sparse.isspmatrix_csr(matrix) or matrix.dtype != np.dtype("float32") or not matrix.has_canonical_format:
        raise ValueError("Brain requires canonical float32 CSR weights, post rows/pre columns")
    return matrix


def _topology_hash(brain, matrix):
    h = hashlib.sha256()
    for a in [brain.node_ids, matrix.indptr, matrix.indices]:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    h.update(str(matrix.shape).encode())
    return h.hexdigest()


def _positions(matrix, pres, posts):
    result = np.empty(len(pres), dtype=np.int64)
    for k, (pre, post) in enumerate(zip(pres, posts)):
        start, end = int(matrix.indptr[post]), int(matrix.indptr[post + 1])
        j = start + int(np.searchsorted(matrix.indices[start:end], pre))
        if j >= end or int(matrix.indices[j]) != int(pre):
            raise ValueError("Candidate/modulatory edge absent from the anatomical topology")
        result[k] = j
    return result


def _ids_to_indices(ids, query):
    query = np.asarray(query)
    if query.dtype.kind not in "iu" or query.ndim != 1 or np.any(query < 0):
        raise ValueError("Anatomical IDs must be exact nonnegative integer arrays")
    indices = np.searchsorted(ids, query)
    if np.any(indices >= len(ids)) or not np.array_equal(ids[indices], query):
        raise ValueError("Port ID absent from the brain")
    return indices.astype(np.int64)


class CandidateGammaPlasticity:
    """Only listed anatomical synapses can change; state is self-contained.

    The caller advances the brain and then calls ``step`` with the same
    interval. Current KC/PAM08 rates are held over that interval. Disabling
    learning freezes factors/weights but keeps causal eligibility history.
    Restoring never silently writes over the brain's effective saved weights.
    """

    def __init__(self, brain, ports_directory, enabled=True):
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be boolean")
        folder = Path(ports_directory)
        port_path = folder / "ports.json"
        manifest = json.loads(port_path.read_text())
        if manifest.get("schema") != "matrix_anatomical_ports_v1":
            raise ValueError("Unsupported anatomical ports schema")
        required = {"plasticity_candidate_edges.csv", "modulator_anatomical_edges.csv", "port_nodes.parquet"}
        records = {a["filename"]: a for a in manifest["artifacts"]}
        if set(records) != required:
            raise ValueError("Incomplete anatomical ports manifest")
        for name, record in records.items():
            path = folder / name
            if path.stat().st_size != record["bytes"] or _sha(path) != record["sha256"]:
                raise ValueError(f"Anatomical port artifact changed: {name}")
        id_dtypes = {"pre_id": "int64", "post_id": "int64", "synapse_count": "int64"}
        candidates = pd.read_csv(folder / "plasticity_candidate_edges.csv", dtype=id_dtypes)
        modulators = pd.read_csv(folder / "modulator_anatomical_edges.csv", dtype=id_dtypes)
        if candidates.empty or candidates.duplicated(["pre_id", "post_id"]).any() or np.any(candidates.synapse_count <= 0):
            raise ValueError("Candidate edges must be unique and contain positive anatomical counts")
        if modulators.duplicated(["pre_id", "post_id", "modulator_port"]).any() or np.any(modulators.synapse_count <= 0):
            raise ValueError("Modulatory edges must be unique positive anatomical counts")
        ports = manifest["ports"]
        gamma_ids = set(ports["KC_gamma"])
        for row in candidates.itertuples(index=False):
            if row.modulator_port not in {"PAM08_L", "PAM08_R"} or row.compartment_label != "gamma4":
                raise ValueError("Candidate edge has an unrecognized dopamine compartment")
            side = row.modulator_port[-1]
            if int(row.pre_id) not in gamma_ids or int(row.post_id) not in ports[f"MBON05_{side}"]:
                raise ValueError("Candidate edge contradicts gamma4 population/side annotations")
        self.brain, self._W = brain, _matrix(brain)
        ids = np.asarray(brain.node_ids)
        if ids.dtype != np.dtype("int64") or ids.ndim != 1 or np.any(ids[1:] <= ids[:-1]) or self._W.shape != (len(ids), len(ids)):
            raise ValueError("Invalid canonical brain ID mapping")
        self.pre_ids = candidates.pre_id.to_numpy(np.int64)
        self.post_ids = candidates.post_id.to_numpy(np.int64)
        self.pre_indices = _ids_to_indices(ids, self.pre_ids)
        self.post_indices = _ids_to_indices(ids, self.post_ids)
        self.positions = _positions(self._W, self.pre_indices, self.post_indices)
        self.base_weights = self._W.data[self.positions].copy()
        if not np.isfinite(self.base_weights).all() or np.any(self.base_weights <= 0):
            raise ValueError("Candidate KC edges must have positive finite baseline efficacy")
        self.dan_ids = np.asarray(sorted(set(ports["PAM08_L"] + ports["PAM08_R"])), dtype=np.int64)
        self.dan_indices = _ids_to_indices(ids, self.dan_ids)
        dan_column = {int(identifier): i for i, identifier in enumerate(self.dan_ids)}
        contact_lookup = {}
        for row in modulators.itertuples(index=False):
            if row.modulator_port not in {"PAM08_L", "PAM08_R"} or int(row.pre_id) not in ports[row.modulator_port]:
                raise ValueError("Modulator edge contradicts PAM08 annotations")
            pre_idx = _ids_to_indices(ids, np.array([row.pre_id], dtype=np.int64))
            post_idx = _ids_to_indices(ids, np.array([row.post_id], dtype=np.int64))
            _positions(self._W, pre_idx, post_idx)
            if row.target_population == "KC_gamma":
                if int(row.post_id) not in gamma_ids:
                    raise ValueError("Modulator target is not an annotated gamma KC")
                contact_lookup.setdefault((row.modulator_port, int(row.post_id)), []).append(
                    (dan_column[int(row.pre_id)], int(row.synapse_count)))
            elif row.target_population != "MBON05" or int(row.post_id) not in ports[f"MBON05_{row.modulator_port[-1]}"]:
                raise ValueError("Unrecognized modulatory target population")
        rows, cols, values = [], [], []
        for k, row in enumerate(candidates.itertuples(index=False)):
            contacts = contact_lookup.get((row.modulator_port, int(row.pre_id)), [])
            total = sum(count for _, count in contacts)
            for column, count in contacts:
                rows.append(k); cols.append(column); values.append(count / total)
        self.dan_weights = sparse.csr_matrix((np.asarray(values, dtype=np.float64), (rows, cols)),
                                             shape=(len(candidates), len(self.dan_ids)))
        self.factors = np.ones(len(candidates), dtype=np.float64)
        self.eligibility = np.zeros(len(candidates), dtype=np.float64)
        self.enabled, self.time_ns, self.update_count = enabled, 0, 0
        self.topology_sha256 = _topology_hash(brain, self._W)
        self.sources = {"ports_json_sha256": _sha(port_path),
                        "plasticity_candidate_edges_sha256": records["plasticity_candidate_edges.csv"]["sha256"],
                        "modulator_anatomical_edges_sha256": records["modulator_anatomical_edges.csv"]["sha256"],
                        "module_sha256": _sha(Path(__file__))}

    def _guard(self):
        matrix = _matrix(self.brain)
        if matrix is not self._W or not np.array_equal(self.brain.node_ids[self.pre_indices], self.pre_ids) or not np.array_equal(self.brain.node_ids[self.post_indices], self.post_ids):
            raise ValueError("Brain topology or ID mapping changed after plasticity attachment")
        if (np.any(self.positions >= matrix.nnz) or
                not np.array_equal(matrix.indices[self.positions], self.pre_indices) or
                np.any(self.positions < matrix.indptr[self.post_indices]) or
                np.any(self.positions >= matrix.indptr[self.post_indices + 1])):
            raise ValueError("Plasticity CSR positions no longer identify the candidate edges")
        expected = (self.base_weights.astype(np.float64) * self.factors).astype(np.float32)
        if not np.array_equal(matrix.data[self.positions], expected):
            raise ValueError("Brain weights disagree with the plasticity state")
        if not np.array_equal(self.brain.node_ids[self.dan_indices], self.dan_ids):
            raise ValueError("Dopamine ID mapping changed")

    def step(self, dt_s):
        if isinstance(dt_s, bool) or not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Plasticity duration must be finite and positive")
        dt_ns = int(round(float(dt_s) * 1_000_000_000))
        if dt_ns <= 0 or not math.isclose(dt_ns, dt_s * 1_000_000_000, rel_tol=0, abs_tol=1e-4):
            raise ValueError("Plasticity duration must be representable in integer nanoseconds")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must remain boolean")
        self._guard()
        indexes = np.concatenate([self.pre_indices, self.dan_indices])
        rates = np.asarray(self.brain.rates)[indexes].astype(np.float64)
        caps = np.asarray(self.brain.r_max)[indexes].astype(np.float64)
        if not np.isfinite(rates).all() or not np.isfinite(caps).all() or np.any(caps <= 0) or np.any(rates < 0) or np.any(rates > caps):
            raise ValueError("Plasticity requires finite neural rates within [0,r_max]")
        n = len(self.pre_indices)
        kc = rates[:n] / caps[:n]
        dan = rates[n:] / caps[n:]
        duration = dt_ns / 1_000_000_000
        decay = math.exp(-duration / ELIGIBILITY_TAU_S)
        eligibility = self.eligibility * decay + kc * (1.0 - decay)
        factors = self.factors
        if self.enabled:
            local_dan = np.asarray(self.dan_weights @ dan).ravel()
            factors = factors * np.exp(-DEPRESSION_RATE_PER_S * duration * local_dan * eligibility)
        if not np.isfinite(eligibility).all() or not np.isfinite(factors).all():
            raise FloatingPointError("Nonfinite plasticity update; no clipping or rescue applied")
        if self.enabled:
            self._W.data[self.positions] = (self.base_weights.astype(np.float64) * factors).astype(np.float32)
        self.eligibility, self.factors = eligibility, factors
        self.time_ns += dt_ns
        self.update_count += 1

    def state_dict(self):
        self._guard()
        result = {"schema": SCHEMA, "enabled": self.enabled, "time_ns": self.time_ns,
                  "update_count": self.update_count, "sources": self.sources.copy(),
                  "topology_sha256": self.topology_sha256,
                  "dan_weight_indptr": self.dan_weights.indptr.copy(),
                  "dan_weight_indices": self.dan_weights.indices.copy(),
                  "dan_weight_data": self.dan_weights.data.copy(),
                  "dan_weight_shape": list(self.dan_weights.shape)}
        for name in ["pre_ids", "post_ids", "pre_indices", "post_indices", "positions", "base_weights",
                     "factors", "eligibility", "dan_ids", "dan_indices"]:
            result[name] = getattr(self, name).copy()
        return result

    @classmethod
    def from_state(cls, brain, state):
        if set(state) != STATE_FIELDS or state.get("schema") != SCHEMA:
            raise ValueError("Unsupported or incomplete plasticity state")
        if not isinstance(state["enabled"], bool):
            raise ValueError("Invalid saved learning flag")
        for key in ["time_ns", "update_count"]:
            if isinstance(state[key], bool) or not isinstance(state[key], (int, np.integer)) or state[key] < 0:
                raise ValueError("Invalid saved plasticity clock")
        matrix = _matrix(brain)
        if state["topology_sha256"] != _topology_hash(brain, matrix):
            raise ValueError("Saved plasticity topology differs from the brain")
        if set(state["sources"]) != {"ports_json_sha256", "plasticity_candidate_edges_sha256", "modulator_anatomical_edges_sha256", "module_sha256"}:
            raise ValueError("Incomplete plasticity provenance")
        if state["sources"]["module_sha256"] != _sha(Path(__file__)):
            raise ValueError("Plasticity implementation changed since the checkpoint")
        obj = cls.__new__(cls)
        obj.brain, obj._W = brain, matrix
        for name in ["pre_ids", "post_ids", "pre_indices", "post_indices", "positions", "dan_ids", "dan_indices"]:
            a = np.asarray(state[name])
            if a.dtype != np.dtype("int64") or a.ndim != 1 or np.any(a < 0):
                raise ValueError(f"Invalid saved integer array: {name}")
            setattr(obj, name, a.copy())
        n = len(obj.pre_ids)
        if not n or any(len(getattr(obj, name)) != n for name in ["post_ids", "pre_indices", "post_indices", "positions"]) or len(set(zip(obj.pre_ids, obj.post_ids))) != n:
            raise ValueError("Invalid saved plastic-edge mapping")
        if len(obj.dan_ids) != len(obj.dan_indices) or len(np.unique(obj.dan_ids)) != len(obj.dan_ids):
            raise ValueError("Invalid saved dopamine mapping")
        if np.any(obj.pre_indices >= len(brain.node_ids)) or np.any(obj.post_indices >= len(brain.node_ids)) or np.any(obj.dan_indices >= len(brain.node_ids)):
            raise ValueError("Saved node index out of bounds")
        for name, dtype in [("base_weights", np.float32), ("factors", np.float64), ("eligibility", np.float64)]:
            a = np.asarray(state[name])
            if a.dtype != np.dtype(dtype) or a.shape != (n,) or not np.isfinite(a).all() or np.any(a < 0):
                raise ValueError(f"Invalid saved plasticity array: {name}")
            setattr(obj, name, a.copy())
        if np.any(obj.base_weights <= 0) or np.any(obj.factors > 1) or np.any(obj.eligibility > 1):
            raise ValueError("Saved plasticity state outside its candidate domain")
        shape = tuple(state["dan_weight_shape"])
        if shape != (n, len(obj.dan_ids)):
            raise ValueError("Saved dopamine projection shape differs")
        data = np.asarray(state["dan_weight_data"])
        if data.dtype != np.dtype("float64") or not np.isfinite(data).all() or np.any(data <= 0):
            raise ValueError("Invalid saved dopamine contact weights")
        obj.dan_weights = sparse.csr_matrix((data.copy(), np.asarray(state["dan_weight_indices"]).copy(),
                                            np.asarray(state["dan_weight_indptr"]).copy()), shape=shape)
        obj.dan_weights.check_format(full_check=True)
        if not obj.dan_weights.has_canonical_format:
            raise ValueError("Saved dopamine projection has duplicate entries")
        sums = np.asarray(obj.dan_weights.sum(axis=1)).ravel()
        if not np.all((sums == 0) | np.isclose(sums, 1, rtol=0, atol=1e-12)):
            raise ValueError("Saved dopamine contact weights are not normalized")
        obj.enabled = state["enabled"]
        obj.time_ns, obj.update_count = int(state["time_ns"]), int(state["update_count"])
        obj.sources, obj.topology_sha256 = state["sources"].copy(), state["topology_sha256"]
        obj._guard()
        return obj
