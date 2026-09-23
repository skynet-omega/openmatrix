"""Descriptive fixed-grid readback of the saved PFG whole-organism traces.

This historical branch is not the protected current pipeline. The grid was
fixed before reading these intermediate samples; no threshold is fitted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live')
HERE = Path(__file__).resolve().parent
ARMS = ('odor_left', 'odor_right', 'uniform', 'sham')
ONSET = 11
ELAPSED = (250, 275, 300, 320)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw(qpos: np.ndarray) -> np.ndarray:
    w, x, y, z = qpos[:, 3:7].T
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def main() -> None:
    arms = {}
    hashes = {}
    for arm in ARMS:
        path = SOURCE / f'{arm}_trace.npz'
        with np.load(path, allow_pickle=False) as z:
            ms = z['ms']
            if not np.array_equal(ms, np.arange(336)):
                raise ValueError(f'{arm}: incomplete trace')
            angle = yaw(z['qpos'])
            ids = z['monitored_ids']
            left = np.flatnonzero(ids == 523769)
            right = np.flatnonzero(ids == 10360)
            if len(left) != 1 or len(right) != 1:
                raise ValueError(f'{arm}: DNa02 IDs')
            dna = z['monitored_q'][:, left[0]] - z['monitored_q'][:, right[0]]
        arms[arm] = {
            str(t): {
                'sample_ms': ONSET+t,
                'yaw_delta_from_onset_deg': float(angle[ONSET+t]-angle[ONSET]),
                'DNa02_q_L_minus_R_at_end': float(dna[ONSET+t]),
            }
            for t in ELAPSED
        }
        hashes[arm] = sha256(path)
    for arm in ARMS:
        for t in ELAPSED:
            key = str(t)
            arms[arm][key]['yaw_minus_sham_deg'] = (
                arms[arm][key]['yaw_delta_from_onset_deg'] -
                arms['sham'][key]['yaw_delta_from_onset_deg']
            )
    for t in ELAPSED:
        key = str(t)
        arms['odor_left'][key]['left_minus_right_yaw_deg'] = (
            arms['odor_left'][key]['yaw_delta_from_onset_deg']-
            arms['odor_right'][key]['yaw_delta_from_onset_deg']
        )
    result = {
        'schema':'historical_pfg_late_fixed_grid_v1',
        'onset_sample_ms':ONSET,
        'elapsed_grid_ms':ELAPSED,
        'source':str(SOURCE),
        'source_sha256':hashes,
        'arms':arms,
        'interpretation':'Descriptive readback of the same PFG 335 ms traces; not an independent replicate, response-onset estimate, or current protected organism.',
    }
    (HERE/'HISTORICAL_LATE_READBACK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
