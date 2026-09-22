"""An independently implemented rate-model candidate on canonical neuron IDs.

Equation 1 of Pugliese et al., bioRxiv 10.1101/2025.09.12.675944:
  tau_i dr_i/dt = max(0, rmax_i*tanh(a_i/rmax_i*(I_i + (W r)_i-theta_i))) - r_i
Source: https://faculty.washington.edu/tuthill/docs/Pugliese_cpg_2025.pdf

This is NOT a reproduction of their MANC/FANC experiment: it extends that
equation to MaleCNS, omits their morphological normalization of gain/threshold,
uses candidate synaptic scale 0.03, NT consensus signs instead of hemilineages,
and float32 explicit Euler instead of adaptive Dopri5. Rates
are model Hz; external drive and threshold use arbitrary model units, not mV
or measured current. No cell's state is claimed to reproduce its physiology.
Histamine inhibition is an additional global hypothesis, without receptor
specificity; unknown/modulatory outputs are zero but their edges are retained.
External boundary activity is assumed zero. No morphology, synaptic delays,
plasticity, modulation, intrinsic bursting or sensory/motor encoding is added.

Euler substeps never exceed 1 ms or the smallest tau, so each update is a
convex combination of a bounded activation and the previous rate. This proves
bounds preservation, NOT dynamical accuracy or convergence for a whole graph.
Convergence must be assessed for each experimental protocol. W.data can be
changed by an explicitly separate plasticity mechanism; its effective values
are saved. This module itself never updates weights through learning.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import scipy
from scipy import sparse


SCHEMA = "matrix_anatomical_rate_brain_v1"
MAX_DT_NS = 1_000_000
SYNAPTIC_SCALE = 0.03
SIGN_HYPOTHESIS = {
    "acetylcholine": 1, "ach": 1,
    "gaba": -1, "glutamate": -1, "glu": -1,
    "histamine": -1, "his": -1,
}
ARRAY_FIELDS = ("node_ids", "nt_labels", "nt_sign", "tau_s", "gain", "theta", "r_max", "rates")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _array_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()


def _runtime():
    return {"python": list(sys.version_info[:2]), "numpy": np.__version__, "scipy": scipy.__version__}


def _positive_normal(rng, mean, std, n):
    """Rejection sampling gives a normal truncated at zero, not clipping."""
    out = rng.normal(mean, std, n).astype(np.float32)
    while np.any(out <= 0):
        mask = out <= 0
        out[mask] = rng.normal(mean, std, int(mask.sum())).astype(np.float32)
    return out


class AnatomicalRateBrain:
    """Full prepared graph, with deterministic bounded rate integration.

    ``step(dt_s, input_by_id)`` holds drive constant over that interval. A
    mapping uses canonical numeric IDs; a dense vector follows ``node_ids``.
    Unknown IDs, fractional nanosecond durations and nonfinite values fail
    before state changes. Call boundaries splitting a substep can change
    Euler discretization; equal substep sequences resume bit exactly on the
    same recorded runtime. Checkpoints never overwrite an existing path.
    """

    def __init__(self, preparation, seed=0):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        ids = np.asarray(preparation.node_ids)
        if (ids.ndim != 1 or ids.dtype != np.int64 or not ids.size
                or np.any(ids < 0) or np.any(ids[1:] <= ids[:-1])):
            raise ValueError("Need unique ascending int64 canonical node IDs")
        graph = preparation.counts_pre_post
        if not sparse.isspmatrix_csr(graph) or graph.shape != (ids.size, ids.size):
            raise ValueError("counts_pre_post must be a square CSR on the ordered IDs")
        graph.check_format(full_check=True)
        if (graph.dtype.kind not in "iu" or not graph.has_canonical_format
                or np.any(graph.data < 0)):
            raise ValueError("Need canonical nonnegative integer synapse counts")
        if not np.array_equal(ids, preparation.nodes["bodyId"].to_numpy()):
            raise ValueError("Neuron metadata order does not match graph IDs")
        self.node_ids = ids.copy()
        labels = preparation.nodes["nt_consensus_nt"].fillna("unknown").astype(str).str.lower().str.strip()
        self.nt_labels = np.asarray(labels.to_list(), dtype="U")
        self.nt_sign = np.array([SIGN_HYPOTHESIS.get(label, 0) for label in self.nt_labels], dtype=np.int8)
        self.W = graph.transpose().tocsr().astype(np.float32)
        # POST rows, PRE column indices: signs belong to the emitting neuron.
        self.W.data *= np.float32(SYNAPTIC_SCALE) * self.nt_sign[self.W.indices]
        # Explicit zero entries are intentionally preserved for every edge.
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        n = ids.size
        self.tau_s = _positive_normal(self.rng, 0.02, 0.002, n)
        self.gain = _positive_normal(self.rng, 1.0, 0.1, n)
        self.theta = _positive_normal(self.rng, 7.5, 0.6, n)
        self.r_max = _positive_normal(self.rng, 200.0, 10.0, n)
        self.rates = np.zeros(n, dtype=np.float32)
        self.time_ns = 0
        self.max_neural_dt_ns = MAX_DT_NS
        self.learning_enabled = False
        self.source_anatomy_identity = {
            "preparation_manifest": json.loads(json.dumps(preparation.manifest)),
            "ordered_ids_sha256": _array_sha(self.node_ids),
            "counts_pre_post_sha256": _array_sha(graph.indptr, graph.indices, graph.data),
            "nt_labels_sha256": _array_sha(self.nt_labels),
        }
        self.model_metadata = {
            "equation_source": "https://doi.org/10.1101/2025.09.12.675944",
            "equation": "tau*dr/dt=max(0,rmax*tanh((gain/rmax)*(input+W@r-theta)))-r",
            "synaptic_scale_initial": SYNAPTIC_SCALE,
            "weight_orientation": "post rows, pre columns; sign by presynaptic consensus NT",
            "sign_hypothesis": SIGN_HYPOTHESIS.copy(),
            "other_nt_sign": 0,
            "nt_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
            "zero_sign_neurons": int(np.count_nonzero(self.nt_sign == 0)),
            "zero_sign_edges": int(np.count_nonzero(self.W.data == 0)),
            "stored_anatomical_edges": int(self.W.nnz),
            "precision": "float32 rates, parameters, weights and sparse multiplication",
            "integrator": "explicit Euler, dt<=1ms and dt<=minimum tau",
            "priors": "independent normal(mean,SD), rejection truncated at zero; tau=(.02,.002)s, gain=(1,.1), theta=(7.5,.6), rmax=(200,10)Hz",
            "morphological_normalization": "absent: no valid morphological scale available",
            "boundary_condition": "zero external activity; this is an assumption, not measured rest",
            "learning": "off in this module; external modifications to W.data are checkpointed",
            "scientific_status": "UNVALIDATED_FULL_GRAPH_RATE_MODEL_CANDIDATE",
        }
        self._source_sha256 = _sha(__file__)
        self._topology_sha256 = self._topology_hash()
        self._freeze_identity()
        self._validate(full=True)

    @property
    def neuron_ids(self):
        return self.node_ids

    @property
    def n_neurons(self):
        return self.node_ids.size

    def _freeze_identity(self):
        for a in (self.node_ids, self.nt_labels, self.nt_sign, self.W.indptr, self.W.indices):
            a.flags.writeable = False

    def _topology_hash(self):
        return _array_sha(self.node_ids, self.nt_labels, self.nt_sign, self.W.indptr, self.W.indices)

    def _validate(self, full=False):
        if type(self.time_ns) is not int or self.time_ns < 0:
            raise ValueError("time_ns must be a nonnegative integer")
        if type(self.max_neural_dt_ns) is not int or not 0 < self.max_neural_dt_ns <= MAX_DT_NS:
            raise ValueError("max_neural_dt_ns must be an integer in [1, 1000000]")
        if self.learning_enabled is not False:
            raise ValueError("Internal learning is unsupported; use an explicit external mechanism")
        n = self.node_ids.size
        for name in ("tau_s", "gain", "theta", "r_max", "rates"):
            a = getattr(self, name)
            if not isinstance(a, np.ndarray) or a.shape != (n,) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError(f"Invalid finite float32 vector: {name}")
            if np.any(a < 0) or (name != "rates" and np.any(a == 0)):
                raise ValueError(f"Invalid nonnegative/positive parameter: {name}")
        if np.min(self.tau_s) < 1e-9 or np.any(self.rates > self.r_max):
            raise ValueError("Rate outside bounds or tau below one nanosecond")
        if full:
            if (self.node_ids.dtype != np.int64 or self.node_ids.ndim != 1 or not n
                    or np.any(self.node_ids < 0) or np.any(self.node_ids[1:] <= self.node_ids[:-1])):
                raise ValueError("Invalid ordered canonical IDs")
            if (self.nt_labels.shape != (n,) or self.nt_labels.dtype.kind != "U"
                    or self.nt_sign.shape != (n,) or self.nt_sign.dtype != np.int8
                    or not np.array_equal(self.nt_sign, [SIGN_HYPOTHESIS.get(s, 0) for s in self.nt_labels])):
                raise ValueError("NT labels/signs differ from recorded hypothesis")
            if not sparse.isspmatrix_csr(self.W) or self.W.shape != (n, n) or self.W.dtype != np.float32:
                raise ValueError("Invalid float32 POST-PRE CSR weights")
            self.W.check_format(full_check=True)
            if not self.W.has_canonical_format or not np.isfinite(self.W.data).all():
                raise ValueError("Nonfinite or noncanonical weights")
            if self._topology_hash() != self._topology_sha256:
                raise ValueError("Canonical IDs, NT identity or graph topology changed")
            if _array_sha(self.node_ids) != self.source_anatomy_identity["ordered_ids_sha256"]:
                raise ValueError("Neuron IDs differ from source anatomy identity")
            if _array_sha(self.nt_labels) != self.source_anatomy_identity["nt_labels_sha256"]:
                raise ValueError("Neurotransmitters differ from source anatomy identity")

    def step(self, dt_s, input_by_id=None):
        """Advance and return rates; external drive is in arbitrary model units."""
        if isinstance(dt_s, bool) or not np.isscalar(dt_s):
            raise ValueError("dt_s must be a finite nonnegative duration")
        value = float(dt_s)
        if not math.isfinite(value) or value < 0:
            raise ValueError("dt_s must be a finite nonnegative duration")
        duration_ns = round(value * 1e9)
        if not math.isclose(value * 1e9, duration_ns, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Duration must contain an integer number of nanoseconds")
        self._validate()
        drive = np.zeros(self.n_neurons, dtype=np.float32)
        if isinstance(input_by_id, Mapping):
            for node_id, amplitude in input_by_id.items():
                if isinstance(node_id, bool) or not isinstance(node_id, (int, np.integer)):
                    raise ValueError("Input IDs must be canonical integers")
                index = np.searchsorted(self.node_ids, node_id)
                if index == self.n_neurons or self.node_ids[index] != node_id:
                    raise ValueError(f"Unknown canonical neuron ID: {node_id}")
                drive[index] = amplitude
        elif input_by_id is not None:
            drive = np.asarray(input_by_id, dtype=np.float32)
        if drive.shape != (self.n_neurons,) or not np.isfinite(drive).all():
            raise ValueError("Input must be a finite vector on ordered neuron IDs")
        remaining = duration_ns
        step_limit = min(self.max_neural_dt_ns, int(float(np.min(self.tau_s)) * 1e9))
        rates = self.rates.copy()
        gain_over_max = self.gain / self.r_max
        while remaining:
            h_ns = min(remaining, step_limit)
            current = self.W @ rates + drive - self.theta
            if not np.isfinite(current).all():
                raise ValueError("Nonfinite recurrent input; weights/state cannot be advanced")
            target = self.r_max * np.maximum(np.tanh(gain_over_max * current), np.float32(0))
            alpha = np.float32(h_ns * 1e-9) / self.tau_s
            rates += alpha * (target - rates)
            # Correct at most float32 roundoff at the analytical bounds.
            np.clip(rates, 0, self.r_max, out=rates)
            remaining -= h_ns
        self.rates[:] = rates
        self.time_ns += duration_ns
        return self.rates

    def save_checkpoint(self, path):
        """Write a new self-contained directory; never overwrite a checkpoint."""
        self._validate(full=True)
        if _sha(__file__) != self._source_sha256:
            raise ValueError("Runtime source changed since this brain was created")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"Preserving existing checkpoint: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
        try:
            np.savez_compressed(staging / "state.npz", **{name: getattr(self, name) for name in ARRAY_FIELDS})
            sparse.save_npz(staging / "weights_post_pre.npz", self.W)
            shutil.copyfile(__file__, staging / "anatomical_rate_brain.py")
            metadata = {
                "schema": SCHEMA, "time_ns": self.time_ns, "seed": self.seed,
                "max_neural_dt_ns": self.max_neural_dt_ns, "learning_enabled": False,
                "rng_state": self.rng.bit_generator.state, "runtime": _runtime(),
                "source_sha256": self._source_sha256, "topology_sha256": self._topology_sha256,
                "source_anatomy_identity": self.source_anatomy_identity,
                "model_metadata": self.model_metadata,
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha(p)} for p in sorted(staging.iterdir())},
            }
            (staging / "manifest.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
            staging.rename(path)
        except BaseException:
            shutil.rmtree(staging)
            raise
        return path

    @classmethod
    def load_checkpoint(cls, path):
        """Restore exact own weights/state; require the recorded code/runtime."""
        path = Path(path)
        meta = json.loads((path / "manifest.json").read_text())
        if meta.get("schema") != SCHEMA:
            raise ValueError("Unsupported anatomical rate brain checkpoint")
        if meta["source_sha256"] != _sha(__file__) or meta["runtime"] != _runtime():
            raise ValueError("Checkpoint requires its recorded source and numerical runtime")
        if set(meta["files"]) != {"state.npz", "weights_post_pre.npz", "anatomical_rate_brain.py"}:
            raise ValueError("Incomplete checkpoint file manifest")
        for name, info in meta["files"].items():
            p = path / name
            if p.stat().st_size != info["bytes"] or _sha(p) != info["sha256"]:
                raise ValueError(f"Checkpoint integrity failed: {name}")
        if meta["files"]["anatomical_rate_brain.py"]["sha256"] != meta["source_sha256"]:
            raise ValueError("Archived source differs from executed source")
        brain = cls.__new__(cls)
        with np.load(path / "state.npz", allow_pickle=False) as state:
            if set(state.files) != set(ARRAY_FIELDS):
                raise ValueError("Unexpected checkpoint arrays")
            for name in ARRAY_FIELDS:
                setattr(brain, name, state[name].copy())
        brain.W = sparse.load_npz(path / "weights_post_pre.npz")
        for name in ("time_ns", "seed", "max_neural_dt_ns", "learning_enabled", "source_anatomy_identity", "model_metadata"):
            setattr(brain, name, meta[name])
        if type(brain.seed) is not int or brain.seed < 0:
            raise ValueError("Invalid saved seed")
        brain.rng = np.random.default_rng()
        brain.rng.bit_generator.state = meta["rng_state"]
        brain._source_sha256 = meta["source_sha256"]
        brain._topology_sha256 = meta["topology_sha256"]
        brain._validate(full=True)
        brain._freeze_identity()
        return brain
