"""Rebuild the exploratory one-second trajectory screen from raw arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'etapa4_mirrored_source_20260924_26'


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q)[..., 3:7].T
    return np.rad2deg(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def bearing(q: np.ndarray, source: np.ndarray) -> np.ndarray:
    direction = np.arctan2(source[1] - 10 * q[:, 1], source[0] - 10 * q[:, 0])
    delta = direction - np.deg2rad(yaw(q))
    return np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta)))


def run(path: Path, output: Path) -> dict:
    plan = json.loads((HERE / 'PLAN.json').read_text())
    result = json.loads((path / 'RESULT.json').read_text())
    frozen = json.loads((path / 'FROZEN.json').read_text())
    need(result['status'] == 'COMPLETE' and result['cleanup_errors'] == [], 'Incomplete organism run')
    need(result['field'] == plan['chosen_arm'] and result['engine'] == 'causal_cuda', 'Wrong arm or engine')
    need(result['completed_preparation_ms'] == 40 and result['completed_trial_ms'] == 1000,
         'Incomplete horizon')
    need(any(name.endswith('__run_long.py') and sha == digest(HERE / 'run_long.py')
             for name, sha in frozen.items()), 'Runner not frozen with this code')
    trace_path = path / 'traces.npz'
    baseline_path = OLD / f"native_{plan['chosen_arm']}_01" / 'traces.npz'
    with np.load(trace_path, allow_pickle=False) as actual_file, np.load(baseline_path, allow_pickle=False) as old_file:
        z = {key: actual_file[key] for key in actual_file.files}
        baseline = {key: old_file[key] for key in old_file.files}
    need(z['fase'].tolist() == ['preparacion'] * 40 + ['ensayo'] * 1000, 'Phase clock')
    need(z['paso'].tolist() == list(range(1, 41)) + list(range(1, 1001)), 'Step clock')
    for key, value in z.items():
        if value.dtype.kind in 'fiu':
            need(np.isfinite(value).all(), 'Nonfinite ' + key)
    need(z['qpos'].shape[0] == 1040 and z['antenas_mm'].shape == (1040, 2, 3), 'Trace shape')
    need(np.max(abs(np.sum(z['qpos'][:, 3:7] ** 2, axis=1) - 1)) < 1e-8,
         'Quaternion norm')
    for key in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns'):
        need(np.array_equal(z[key], z['CNS_time_ns']), 'Clock mismatch ' + key)
        need(np.all(np.diff(z[key]) == 1_000_000), 'Clock increment ' + key)
    source = np.asarray(json.loads((HERE / 'CAMPOS.json').read_text())[plan['chosen_arm']]['source_mm'])
    sigma = float(json.loads((HERE / 'CAMPOS.json').read_text())[plan['chosen_arm']]['sigma_mm'])
    concentration = np.exp(-np.sum((z['antenas_mm'][40:, :, :2] - source) ** 2, axis=2) /
                           (2 * sigma * sigma))
    need(np.max(abs(concentration - z['concentracion_campo'][40:])) <= 1e-12,
         'Gaussian field differs from antenna pose')
    need(np.max(abs(z['sensores_pendientes'][40:, :2] - concentration)) <= 1e-12,
         'Pending sensor differs from field')
    need(np.array_equal(z['sensores_usados'][41:], z['sensores_pendientes'][40:-1]),
         'Committed sensor lag differs from one millisecond')
    q = z['DN_q_usada'][40:]
    b = z['DN_baseline'][40:]
    expected_command = np.tanh(250 * ((q[:, 2] - b[:, 2]) - (q[:, 3] - b[:, 3]))) * np.deg2rad(5)
    need(float(np.max(abs(expected_command - z['command_yaw_rate_rad_s'][40:]))) <= 1e-12,
         'Motor reader differs from locked equation')
    prefix = {}
    for key in ('qpos', 'position_mm', 'yaw_delta_deg', 'command_yaw_rate_rad_s',
                'concentracion_campo', 'sensores_usados'):
        need(z[key][:440].shape == baseline[key].shape, 'Historical prefix shape ' + key)
        prefix[key] = float(np.max(abs(z[key][:440] - baseline[key])))
    limits = plan['screen']['prefix_max_abs']
    prefix_ok = all(prefix[k] <= limits[k] for k in prefix)
    signed = bearing(z['qpos'], source)
    error_start = float(abs(signed[39]))
    error_400 = float(abs(signed[439]))
    error_1000 = float(abs(signed[-1]))
    distance = np.linalg.norm(z['position_mm'][:, :2] - source, axis=1)
    upright_min = float(z['upright'][40:].min())
    contacts_min = int(z['contact_active'][40:].sum(axis=1).min())
    screen = plan['screen']
    support_ok = upright_min >= screen['upright_min'] and contacts_min >= screen['contacts_min']
    improvement = error_start - error_1000
    late_improvement = error_400 - error_1000
    positive = (prefix_ok and support_ok and improvement >= screen['bearing_improvement_deg_min']
                and late_improvement >= screen['late_bearing_improvement_deg_min'])
    classification = ('INCOMPARABLE' if not prefix_ok or not support_ok else
                      'SCREEN_POSITIVE_NEEDS_MATCHED_CONTROL' if positive else
                      'SCREEN_NEGATIVE_FOR_THIS_HORIZON')
    receipt = {
        'schema': 'stage4_one_second_raw_screen_v1',
        'classification': classification,
        'stage4_navigation_admitted': False,
        'stage5_feedback_admitted': False,
        'field': plan['chosen_arm'],
        'run_result_sha256': digest(path / 'RESULT.json'),
        'trace_sha256': digest(trace_path),
        'historical_trace_sha256': digest(baseline_path),
        'plan_sha256': digest(HERE / 'PLAN.json'),
        'verifier_sha256': digest(Path(__file__)),
        'historical_prefix_max_abs': prefix,
        'historical_prefix_within_frozen_limits': prefix_ok,
        'bearing_error_deg': {'prepared': error_start, 'trial_400ms': error_400,
                              'trial_1000ms': error_1000},
        'bearing_improvement_deg': improvement,
        'late_bearing_improvement_deg': late_improvement,
        'body_source_distance_mm': {'prepared': float(distance[39]),
                                    'trial_400ms': float(distance[439]),
                                    'trial_1000ms': float(distance[-1])},
        'upright_min': upright_min,
        'contacts_min': contacts_min,
        'max_abs_reader_error_rad_s': float(np.max(abs(expected_command - z['command_yaw_rate_rad_s'][40:]))),
        'interpretation': 'One fixed-source life is an exploratory trajectory screen. Absolute approach can arise from common forward drive; neither feedback nor navigation is proven without a matched control.'
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.out), ensure_ascii=False, allow_nan=False))
