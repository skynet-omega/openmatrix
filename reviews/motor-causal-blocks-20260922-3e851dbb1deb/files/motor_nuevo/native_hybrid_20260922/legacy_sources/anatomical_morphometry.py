"""Explicit morphology intervention on the continuing canonical rate brain.

Pugliese et al. scale gain by inverse relative cell size and threshold by
relative size. Here the denominator is the median volume of this entire
canonical CNS, not their T1-selected network. This is an unvalidated transfer
of a published modelling hypothesis, not an inferred physiological fit.

Volumes come from the same MaleCNS v1.0 segmentation. They are not membrane
surface, capacitance, or input resistance; neither size scaling nor the
connectome specifies those quantities. No missing size is silently imputed.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from anatomical_rate_brain import _array_sha, _sha


SCHEMA = "matrix_anatomical_morphometry_v1"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "data/anatomical_morphometry/v0.1"


def load_morphometry(manifest_or_directory=None):
    """Load hashed volumes; missing values require an explicit preservation policy."""
    path = Path(manifest_or_directory or DEFAULT_DIRECTORY)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported anatomical morphometry schema")
    artifact = manifest["artifact"]
    if artifact["filename"] != "canonical_morphometry.parquet":
        raise ValueError("Unexpected morphometry artifact")
    table_path = manifest_path.parent / artifact["filename"]
    if table_path.stat().st_size != artifact["bytes"] or _sha(table_path) != artifact["sha256"]:
        raise ValueError("Morphometry artifact integrity failed")
    table = pd.read_parquet(table_path)
    if list(table.columns) != ["bodyId", "size_source_voxels"]:
        raise ValueError("Unexpected morphometry columns")
    ids, sizes = table.bodyId.to_numpy(), table.size_source_voxels.to_numpy()
    if (ids.dtype != np.int64 or not len(ids) or np.any(ids[1:] <= ids[:-1])
            or sizes.dtype != np.float64 or np.any(np.isinf(sizes)) or np.any(sizes <= 0)):
        raise ValueError("Need ascending unique int64 IDs and positive float64 observed sizes")
    observed = np.isfinite(sizes)
    if not observed.any():
        raise ValueError("No observed morphology")
    missing_ids = ids[~observed].tolist()
    if missing_ids and (manifest.get("missing_size_policy") != "preserve_original_parameters_without_size_imputation"
                        or manifest.get("missing_size_ids") != missing_ids):
        raise ValueError("Missing sizes require an explicit exact-ID preservation policy")
    if (manifest["canonical_ids"] != len(ids)
            or manifest["matched_ids"] != len(ids)
            or manifest["positive_finite_sizes"] != int(observed.sum())
            or manifest["ordered_ids_sha256"] != _array_sha(ids)
            or manifest["sizes_sha256"] != _array_sha(sizes)):
        raise ValueError("Incomplete or inconsistent canonical morphometry coverage")
    median = float(np.median(sizes[observed]))
    if manifest["normalization_median_source_voxels"] != median:
        raise ValueError("Morphometry normalization differs from recorded whole-CNS median")
    return ids, sizes, median, manifest, manifest_path


def apply_morphometry(brain, manifest_or_directory=None):
    """Apply once, preserving current rates, weights, clock and RNG exactly.

This is an explicit parameter intervention on an existing life. It does not
reset that life or pretend these new parameters were in effect in its past.
The returned receipt is also saved in brain.model_metadata, so the ordinary
brain checkpoint carries it without altering the frozen v1 runtime module.
"""
    if "morphometry_intervention" in brain.model_metadata:
        raise ValueError("Morphometry was already applied; refusing cumulative scaling")
    brain._validate(full=True)
    ids, sizes, median, manifest, path = load_morphometry(manifest_or_directory)
    if not np.array_equal(ids, brain.node_ids):
        raise ValueError("Morphometry IDs must equal every ordered current brain ID")
    observed = np.isfinite(sizes)
    # The NaNs remain missing in the source artifact. For those exact IDs the
    # original parameters are copied unchanged, not assigned an estimated size.
    scale = np.ones(len(sizes), dtype=np.float64)
    scale[observed] = sizes[observed] / median
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        gain = (brain.gain.astype(np.float64) / scale).astype(np.float32)
        theta = (brain.theta.astype(np.float64) * scale).astype(np.float32)
    if (not np.isfinite(gain).all() or not np.isfinite(theta).all()
            or np.any(gain <= 0) or np.any(theta <= 0)):
        raise ValueError("Morphometry produces invalid float32 parameters")
    # Everything is validated before either parameter is changed.
    receipt = {
        "schema": "matrix_morphometry_intervention_v1",
        "at_time_ns": brain.time_ns,
        "formula": "observed s_i=size_i/median(observed_size_all_canonical); gain_i <- gain_i/s_i; theta_i <- theta_i*s_i; missing-size parameters preserved exactly",
        "equation_source": "https://doi.org/10.1101/2025.09.12.675944",
        "source_implementation": "https://github.com/smpuglie/Pugliese_2026/blob/0452ec8359c1ffa4833fc578482e21525c669a0d/src/utils/sim_utils.py",
        "module_sha256": _sha(__file__),
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha(path),
        "dataset": deepcopy(manifest),
        "normalization_scope": "all observed positive volumes on ordered canonical neurons of this continuing CNS",
        "scaled_neurons": int(observed.sum()),
        "unscaled_missing_size_neurons": int((~observed).sum()),
        "unscaled_missing_size_ids": ids[~observed].tolist(),
        "preserved_missing_gain_theta_sha256": _array_sha(brain.gain[~observed], brain.theta[~observed]),
        "size_scale_min_median_max": [float(scale.min()), float(np.median(scale)), float(scale.max())],
        "before_gain_theta_sha256": _array_sha(brain.gain, brain.theta),
        "after_gain_theta_sha256": _array_sha(gain, theta),
        "preserved_weights_post_pre_sha256": _array_sha(brain.W.indptr, brain.W.indices, brain.W.data),
        "preserved_rates_sha256": _array_sha(brain.rates),
        "preserved_other_parameters_sha256": _array_sha(brain.tau_s, brain.r_max),
        "preserved_rng_state": deepcopy(brain.rng.bit_generator.state),
        "scientific_status": "UNVALIDATED_WHOLE_CNS_VOLUME_SCALING_HYPOTHESIS",
        "limitations": [
            "Segmentation volume is not measured membrane surface, capacitance or input resistance.",
            "Whole-CNS median differs from the source paper's selected-network median.",
            "Uniform synaptic efficacy, neurotransmitter signs and external drive remain candidate hypotheses.",
            "No target firing rate, action, reward, weight reset, feedback controller or behaviour fit is introduced.",
        ],
    }
    brain.gain, brain.theta = gain, theta
    brain.model_metadata["morphological_normalization"] = "explicit same-release volume intervention; see morphometry_intervention"
    brain.model_metadata["morphometry_intervention"] = deepcopy(receipt)
    return receipt
