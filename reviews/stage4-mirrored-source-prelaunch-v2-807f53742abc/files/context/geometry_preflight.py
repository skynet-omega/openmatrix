"""Read-only CPU discriminator for prospective mirrored finite sources.

Samples candidate fields on old trajectories. These are counterfactual input
samples, never odor consumed by those organisms or evidence of navigation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = Path(__file__).resolve().parent
GEOMETRY = ROOT / "campanas/etapa4_diseno_20260923_17/GEOMETRY.json"
CLOSE = ROOT / "campanas/etapa4_gaussian_power_recovery_20260924_24/CLOSE.json"
PLAN = CAMPAIGN / "PLAN.json"
PREVIOUS = ROOT / "campanas/etapa4_gaussian_power_recovery_20260924_24"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample(points: np.ndarray, source: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-np.sum((points[..., :2] - source) ** 2, axis=-1) / (2 * sigma * sigma))


def source_pair(geometry: dict, old_specs: dict) -> tuple[dict, dict]:
    antennae = np.asarray(geometry["prepared_antennae_mm"], dtype=np.float64)
    q = np.asarray(geometry["prepared_qpos_root"], dtype=np.float64)
    require(antennae.shape == (2, 3) and q.shape == (7,), "Invalid prepared geometry")
    require(np.isfinite(antennae).all() and np.isfinite(q).all(), "Nonfinite geometry")
    midpoint = antennae[:, :2].mean(axis=0)
    baseline = float(np.linalg.norm(antennae[0, :2] - antennae[1, :2]))
    require(baseline > 0, "Zero antenna baseline")
    left = (antennae[0, :2] - antennae[1, :2]) / baseline
    forward = np.array([left[1], -left[0]])
    w, x, y, z = q[3:]
    heading = np.array([1 - 2 * (y * y + z * z), 2 * (w * z + x * y)])
    if float(forward @ heading) < 0:
        forward = -forward

    old_plus = old_specs["plus"]
    old_minus = old_specs["minus"]
    old_c = [sample(antennae, np.asarray(old_specs[key]["source_mm"]),
                    float(old_specs[key]["sigma_mm"])) for key in ("plus", "minus")]
    require(np.max(np.abs(old_c[0] - old_c[1])) <= 1e-12,
            "Old pair does not share initial concentration")
    old_common = float(np.mean(old_c[0]))

    # Fixed dimensionless geometry: a=1.5, sigma=1.5 antenna baselines.
    # The single forward distance is solved to match the already registered
    # initial common concentration; no motor result enters this equation.
    lateral_b = sigma_b = 1.5
    near_base = math.exp(-((lateral_b - 0.5) ** 2) / (2 * sigma_b ** 2))
    far_base = math.exp(-((lateral_b + 0.5) ** 2) / (2 * sigma_b ** 2))
    base_common = 0.5 * (near_base + far_base)
    require(0 < old_common <= base_common < 1, "No real matched-common solution")
    forward_b = math.sqrt(-2 * sigma_b ** 2 * math.log(old_common / base_common))
    sigma = sigma_b * baseline
    pair = {
        "left": {"source_mm": (midpoint + baseline * (forward_b * forward + lateral_b * left)).tolist(),
                 "sigma_mm": sigma},
        "right": {"source_mm": (midpoint + baseline * (forward_b * forward - lateral_b * left)).tolist(),
                  "sigma_mm": sigma},
    }
    initial = {key: sample(antennae, np.asarray(spec["source_mm"]), sigma)
               for key, spec in pair.items()}
    common_error = abs(float(initial["left"].mean() - old_common))
    pair_common_error = abs(float(initial["left"].mean() - initial["right"].mean()))
    contrast_sum = float((initial["left"][0] - initial["left"][1]) +
                         (initial["right"][0] - initial["right"][1]))
    require(max(common_error, pair_common_error, abs(contrast_sum)) < 1e-13,
            "Mirrored sources do not meet exact initial control")
    require(initial["left"][0] > initial["left"][1] and
            initial["right"][0] < initial["right"][1], "No signed reversal")
    calibration = {
        "antenna_baseline_mm": baseline,
        "old_initial_common": old_common,
        "new_initial_common": float(initial["left"].mean()),
        "common_match_abs_error": common_error,
        "left_initial_L_R": initial["left"].tolist(),
        "right_initial_L_R": initial["right"].tolist(),
        "left_minus_right_contrast_at_initial": float(initial["left"][0] - initial["left"][1]),
        "forward_offset_baselines": forward_b,
        "lateral_offset_baselines": lateral_b,
        "sigma_baselines": sigma_b,
        "matched_property": "initial bilateral mean only; no claim of matched future history",
        "old_specs_identity": {key: old_specs[key] for key in ("plus", "minus")},
    }
    return pair, calibration


def trace_readback(path: Path, old_spec: dict, pair: dict) -> dict:
    with np.load(path, allow_pickle=False) as z:
        points = z["antenas_mm"].copy()
        actual = z["concentracion_campo"].copy()
        phase = z["fase"].copy()
        steps = z["paso"].copy()
    require(points.shape == (440, 2, 3) and actual.shape == (440, 2),
            "Unexpected full-organism trace shape")
    require(np.array_equal(phase[:40], np.full(40, "preparacion")) and
            np.array_equal(phase[40:], np.full(400, "ensayo")) and
            np.array_equal(steps[40:], np.arange(1, 401)), "Trace phase/clock mismatch")
    require(np.isfinite(points).all() and np.isfinite(actual).all(), "Nonfinite trace")
    old_pred = sample(points[40:], np.asarray(old_spec["source_mm"]),
                      float(old_spec["sigma_mm"]))
    old_error = float(np.max(np.abs(old_pred - actual[40:])))
    require(old_error < 1e-12, "Exact Gaussian formula disagrees with consumed field")
    fields = {key: sample(points[40:], np.asarray(spec["source_mm"]),
                          float(spec["sigma_mm"])) for key, spec in pair.items()}
    trial_late = slice(299, 400)
    left_lr = fields["left"][:, 0] - fields["left"][:, 1]
    right_lr = fields["right"][:, 0] - fields["right"][:, 1]
    return {
        "trace_sha256": sha(path),
        "old_consumed_field_formula_max_abs_error": old_error,
        "candidate_scope": "counterfactual field on an old pose; not consumed by its CNS",
        "late_mean_signed_LR_left_source": float(np.mean(left_lr[trial_late])),
        "late_mean_signed_LR_right_source": float(np.mean(right_lr[trial_late])),
        "late_mean_abs_source_pair_field_difference": float(np.mean(np.abs(fields["left"][trial_late] - fields["right"][trial_late]))),
        "last_left_field_L_R": fields["left"][-1].tolist(),
        "last_right_field_L_R": fields["right"][-1].tolist(),
        "last_signed_LR_by_source": [float(left_lr[-1]), float(right_lr[-1])],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    require(not args.out.exists(), "Output directory already exists")
    plan = json.loads(PLAN.read_text())
    require(plan["status"] == "OPTIONS_ONLY_NO_NEW_ORGANISM" and
            plan["budget"]["new_organism_runs"] == 0, "Planning budget changed")
    require(sha(CLOSE) == plan["basis"]["close_sha256"], "Scientific close changed")
    geometry = json.loads(GEOMETRY.read_text())
    require(sha(GEOMETRY) == "3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3",
            "Prepared geometry changed")
    plus = PREVIOUS / "native_plus_01"
    minus = PREVIOUS / "native_minus_02"
    old_specs = json.loads((plus / "GAUSSIAN_SPEC.json").read_text())
    require(old_specs == json.loads((minus / "GAUSSIAN_SPEC.json").read_text()),
            "Old source definitions changed across arms")
    pair, calibration = source_pair(geometry, old_specs)
    arms = {key: trace_readback(folder / "traces.npz", old_specs[key], pair)
            for key, folder in (("plus", plus), ("minus", minus))}
    elapsed_wall, elapsed_cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
    require(elapsed_wall <= plan["budget"]["CPU_wall_s_max"], "CPU preflight over budget")
    result = {
        "schema": "stage4_mirrored_geometry_cpu_preflight_v1",
        "status": "CPU_COMPLETE_DESCRIPTIVE_ONLY",
        "selected_hypothesis": "A",
        "sources": pair,
        "calibration": calibration,
        "old_trace_counterfactuals": arms,
        "wall_s": elapsed_wall,
        "cpu_s": elapsed_cpu,
        "input_sha256": {"plan": sha(PLAN), "geometry": sha(GEOMETRY), "close": sha(CLOSE),
                         "old_plus_spec": sha(plus / "GAUSSIAN_SPEC.json"),
                         "old_minus_spec": sha(minus / "GAUSSIAN_SPEC.json")},
        "code_sha256": sha(Path(__file__)),
        "organism_executed": False,
        "neural_effect_tested": False,
        "stage4_admission": False,
    }
    args.out.mkdir(parents=True)
    (args.out / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "wall_s": elapsed_wall,
                      "initial_contrast": calibration["left_minus_right_contrast_at_initial"],
                      "arms": arms}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
