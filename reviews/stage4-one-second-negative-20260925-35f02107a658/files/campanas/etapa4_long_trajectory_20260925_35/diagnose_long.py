"""Post-run descriptive decomposition of bearing, body yaw and neural command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run(raw: Path, output: Path) -> None:
    verified = HERE / 'RAW_VERIFIED_01.json'
    optimized = HERE / 'RAW_VERIFIED_02.json'
    need(verified.read_bytes() == optimized.read_bytes(), 'Normal/-O verification differs')
    result = json.loads(verified.read_text())
    trace = raw / 'traces.npz'
    need(result['classification'] == 'SCREEN_NEGATIVE_FOR_THIS_HORIZON'
         and result['trace_sha256'] == digest(trace), 'Trace does not match verified negative')
    spec = json.loads((HERE / 'CAMPOS.json').read_text())[result['field']]
    source = np.asarray(spec['source_mm'], dtype=float)
    with np.load(trace, allow_pickle=False) as z:
        q = z['qpos']
        w, x, y, zz = q[:, 3:7].T
        yaw = np.rad2deg(np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz)))
        source_direction = np.rad2deg(np.arctan2(source[1] - 10 * q[:, 1],
                                                source[0] - 10 * q[:, 0]))
        cmd_deg_s = np.rad2deg(z['command_yaw_rate_rad_s'][40:])
        lr = z['sensores_usados'][40:, 0] - z['sensores_usados'][40:, 1]
        forward = z['command_forward_mm_s'][40:]
        need(np.isfinite(yaw).all() and np.isfinite(source_direction).all()
             and np.isfinite(cmd_deg_s).all() and np.isfinite(lr).all(), 'Nonfinite input')
        need(len(cmd_deg_s) == 1000 and np.array_equal(np.unique(forward), [0.2]),
             'Unexpected duration or forward command')
        body_xy_change = (z['position_mm'][-1, :2] - z['position_mm'][39, :2]).tolist()
    receipt = {
        'schema': 'stage4_one_second_posthoc_decomposition_v1',
        'classification': 'POSTHOC_DESCRIPTIVE_NOT_AN_ADMISSION_GATE',
        'trace_sha256': digest(trace),
        'raw_verifier_sha256': digest(verified),
        'source_sha256': digest(HERE / 'CAMPOS.json'),
        'analysis_code_sha256': digest(Path(__file__)),
        'source_direction_change_deg': float(source_direction[-1] - source_direction[39]),
        'body_yaw_change_deg': float(yaw[-1] - yaw[39]),
        'neural_command_integral_deg': float(cmd_deg_s.sum() / 1000),
        'neural_command_integral_first400_deg': float(cmd_deg_s[:400].sum() / 1000),
        'neural_command_integral_last600_deg': float(cmd_deg_s[400:].sum() / 1000),
        'neural_command_rate_deg_s': {
            'mean': float(cmd_deg_s.mean()), 'min': float(cmd_deg_s.min()),
            'max': float(cmd_deg_s.max()),
            'positive_fraction': float(np.mean(cmd_deg_s > 0)),
            'saturated_fraction': float(np.mean(abs(cmd_deg_s) >= 4.99))
        },
        'consumed_l_minus_r': {
            'first100_mean': float(lr[:100].mean()),
            'last100_mean': float(lr[-100:].mean())
        },
        'forward_command_mm_s': 0.2,
        'body_xy_change_mm': body_xy_change,
        'interpretation': 'On this trajectory the line of sight moved far more than body yaw; the neural command did not approach the ±5 deg/s ceiling. The source distance decrease is compatible with tonic forward motion. This does not identify the upstream neural mechanism or prove a body-independent bottleneck.'
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.out)
