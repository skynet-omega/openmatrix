"""Fixed-window readback of the saved 335-ms whole-organism experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live')
ARMS = ('odor_left', 'odor_right', 'uniform', 'sham')
ONSET = 11
ENDS = (111, 211, 311)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw(qpos: np.ndarray) -> np.ndarray:
    w, x, y, z = qpos[:, 3:7].T
    return np.rad2deg(np.unwrap(np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--out', type=Path, default=HERE / 'TIME_WINDOWS.json')
    args = parser.parse_args()
    values = {}
    hashes = {}
    for arm in ARMS:
        tp = args.source / f'{arm}_trace.npz'
        npz = args.source / f'{arm}_native_transmission.npz'
        with np.load(tp, allow_pickle=False) as z:
            if not np.array_equal(z['ms'], np.arange(336)):
                raise ValueError(f'{arm}: incomplete trace')
            ids = z['monitored_ids']
            left = np.flatnonzero(ids == 523769)
            right = np.flatnonzero(ids == 10360)
            if len(left) != 1 or len(right) != 1:
                raise ValueError(f'{arm}: DNa02 identification')
            a = yaw(z['qpos'])
            d = z['monitored_q'][:, left[0]] - z['monitored_q'][:, right[0]]
        with np.load(npz, allow_pickle=False) as z:
            if not np.array_equal(z['times_ms'], np.arange(336)) or not np.array_equal(z['target_ids'], [523769, 10360]):
                raise ValueError(f'{arm}: native input alignment')
            n = z['net'][:, 0] - z['net'][:, 1]
        if not np.isfinite(a).all() or not np.isfinite(d).all() or not np.isfinite(n).all():
            raise ValueError(f'{arm}: nonfinite trajectory')
        values[arm] = {
            str(end-ONSET): {
                'end_ms': end,
                'yaw_delta_from_onset_deg': float(a[end] - a[ONSET]),
                'DNa02_L_minus_R_at_end': float(d[end]),
                'native_net_L_minus_R_at_end': float(n[end]),
            }
            for end in ENDS
        }
        hashes[arm] = {'trace': sha(tp), 'native': sha(npz)}
    for arm in ARMS:
        for elapsed in (str(end-ONSET) for end in ENDS):
            values[arm][elapsed]['yaw_delta_from_onset_minus_sham_deg'] = (
                values[arm][elapsed]['yaw_delta_from_onset_deg'] -
                values['sham'][elapsed]['yaw_delta_from_onset_deg']
            )
    result = {
        'schema': 'stage3_saved_time_windows_v1',
        'onset_ms': ONSET,
        'source': str(args.source),
        'source_hashes': hashes,
        'arms': values,
        'interpretation': 'Exploratory readback of the same 335-ms PFG experiment, not an independent replicate or a new motor run.',
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
