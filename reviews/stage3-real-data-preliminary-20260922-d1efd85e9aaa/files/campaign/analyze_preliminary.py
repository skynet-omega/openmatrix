"""Read-only stage-3 trace audit and one published live-fly raw file.

No simulator run or parameter fitting. The live file lacks trial-side labels, so
this script deliberately does not score directional agreement from that file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live')
ARMS = ('odor_left', 'odor_right', 'uniform', 'sham')
WINDOW = (161, 311)  # inherited source onset 11 ms + prespecified 150–300 ms
RAW_FILE = HERE / 'real_fly_11634643.mat'
NODES = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet')
COUNTS = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/counts_pre_post.npz')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def window_mean(x: np.ndarray, ms: np.ndarray) -> float:
    lo, hi = WINDOW
    ix = (ms >= lo) & (ms <= hi)
    return float(np.trapz(x[ix], ms[ix]) / (hi - lo))


def yaw_deg(qpos: np.ndarray) -> np.ndarray:
    w, x, y, z = qpos[:, 3:7].T
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))


def simulation_audit() -> dict:
    traces = {a: read_npz(SOURCE / f'{a}_trace.npz') for a in ARMS}
    ref = traces['sham']
    nodes = (pd.DataFrame(json.loads(NODES.read_text())) if NODES.suffix == '.json'
             else pd.read_parquet(NODES)).set_index('bodyId')
    if nodes.loc[523769, 'instance'] != 'DNa02_L' or nodes.loc[10360, 'instance'] != 'DNa02_R':
        raise ValueError('DNa02 anatomical side mismatch')
    orn_sides = nodes.loc[ref['ORN_ids'], 'rootSide'].to_numpy()
    if set(orn_sides) != {'L', 'R'}:
        raise ValueError('ORN side metadata missing or unexpected')
    output = {}
    for arm, t in traces.items():
        ms = t['ms']
        if not np.array_equal(ms, np.arange(336)):
            raise ValueError(f'{arm}: incomplete time axis')
        if not np.array_equal(t['monitored_ids'], ref['monitored_ids']):
            raise ValueError(f'{arm}: nonmatched observer ids')
        if not np.isfinite(t['monitored_q']).all() or not np.isfinite(t['qpos']).all():
            raise ValueError(f'{arm}: nonfinite')
        ids = t['monitored_ids']
        li = int(np.flatnonzero(ids == 523769)[0])
        ri = int(np.flatnonzero(ids == 10360)[0])
        pair = t['monitored_q'][:, li] - t['monitored_q'][:, ri]
        ref_pair = ref['monitored_q'][:, li] - ref['monitored_q'][:, ri]
        native = read_npz(SOURCE / f'{arm}_native_transmission.npz')
        if not np.array_equal(native['target_ids'], np.array([523769, 10360])):
            raise ValueError(f'{arm}: native input targets mismatch')
        angle = yaw_deg(t['qpos'])
        output[arm] = {
            'yaw_change_deg_161_311': float(angle[311]-angle[161]),
            'DNa02_L_minus_R_mean_q_161_311': window_mean(pair, ms),
            'DNa02_L_minus_R_change_from_sham': window_mean(pair-ref_pair, ms),
            'DNa02_L_q_mean': window_mean(t['monitored_q'][:, li], ms),
            'DNa02_R_q_mean': window_mean(t['monitored_q'][:, ri], ms),
            'steering_mean_161_311': window_mean(t['neural_steering_applied'], ms),
            'DNa02_native_net_mean_L_R': [window_mean(native['net'][:, i], ms) for i in (0, 1)],
            'DNa02_native_target_mean_L_R': [window_mean(native['target'][:, i], ms) for i in (0, 1)],
            'ORN_q_L2_from_sham_mean': window_mean(np.linalg.norm(t['ORN_q']-ref['ORN_q'], axis=1), ms),
            'ORN_left_mean_q': window_mean(t['ORN_q'][:, orn_sides == 'L'].mean(axis=1), ms),
            'ORN_right_mean_q': window_mean(t['ORN_q'][:, orn_sides == 'R'].mean(axis=1), ms),
            'input_left_right_mean': [window_mean(t['sensors_used'][:, i], ms[:-1]) for i in (0, 1)],
            'trace_sha256': sha(SOURCE / f'{arm}_trace.npz'),
            'native_transmission_sha256': sha(SOURCE / f'{arm}_native_transmission.npz'),
        }
    q = read_npz(SOURCE / 'odor_left_network_trace.npz')
    if q['q'].shape != (10, 166700):
        raise ValueError('Unexpected full-network snapshot shape')
    # Occupancy describes q at stored snapshots, not firing or active-set size.
    occupancy = [{'time_ms': int(t), 'fraction_q_gt_0p01': float(np.mean(row > 0.01)),
                  'fraction_q_gt_0p1': float(np.mean(row > 0.1))}
                 for t, row in zip(q['times_ms'], q['q'])]
    if COUNTS.suffix == '.npy':
        outgoing_edges = np.load(COUNTS, allow_pickle=False)
    else:
        with np.load(COUNTS, allow_pickle=False) as graph:
            outgoing_edges = np.diff(graph['indptr']).astype(np.int64)
    if outgoing_edges.shape != (166700,) or int(outgoing_edges.sum()) != 25582938:
        raise ValueError('Unexpected connectome graph')
    edge_occupancy = [{'time_ms': int(q['times_ms'][i]),
                       'fraction_edges_from_q_gt_0p01': float(outgoing_edges[q['q'][i] > 0.01].sum() / outgoing_edges.sum())}
                      for i in (0, 6, 9)]
    return {
        'model_source': str(SOURCE),
        'model_species_preparation': 'male MCNS reconstructed CNS + modeled sensory input + FlyBody MuJoCo',
        'DNa02_ids': {'left': 523769, 'right': 10360},
        'ORN_counts_by_side': {'left': int(np.sum(orn_sides == 'L')), 'right': int(np.sum(orn_sides == 'R'))},
        'anatomical_metadata_sha256': sha(NODES),
        'ON_window_ms': list(WINDOW),
        'arms': output,
        'cross_arm': {
            'ORN_q_L2_left_vs_right_mean': window_mean(np.linalg.norm(
                traces['odor_left']['ORN_q']-traces['odor_right']['ORN_q'], axis=1), ref['ms']),
            'DNa02_L_minus_R_left_minus_right': output['odor_left']['DNa02_L_minus_R_mean_q_161_311']-
                                                output['odor_right']['DNa02_L_minus_R_mean_q_161_311'],
        },
        'stored_q_occupancy': occupancy,
        'stored_outgoing_edge_occupancy': edge_occupancy,
        'network_snapshot_sha256': sha(SOURCE / 'odor_left_network_trace.npz'),
        'connectivity_sha256': sha(COUNTS),
        'warning': 'Occupancy of q or outgoing edges is not firing fraction or proof that any cell can skip fine integration.'
    }


def living_fly_audit() -> dict:
    d = loadmat(RAW_FILE, squeeze_me=True)
    fs = float(d['ephys_SR'])
    ball_fs = float(d['ball_SR'])
    stim = np.asarray(d['stim']).reshape(-1)
    ephys_t = np.asarray(d['t_ephys']).reshape(-1)
    yaw = np.asarray(d['yaw']).reshape(-1)
    ball_t = np.asarray(d['t_ball']).reshape(-1)
    if fs != 4000 or ball_fs != 100 or len(stim) != len(ephys_t) or len(yaw) != len(ball_t):
        raise ValueError('Unexpected raw sample rates or lengths')
    if not all(np.isfinite(a).all() for a in (stim, ephys_t, yaw, ball_t)):
        raise ValueError('Nonfinite raw live data')
    active = stim > 0
    edges = np.diff(active.astype(np.int8))
    on = np.flatnonzero(edges == 1) + 1
    off = np.flatnonzero(edges == -1) + 1
    if len(on) != len(off) or not np.all(off > on):
        raise ValueError('Unpaired stimulus edges')
    return {
        'source_doi': '10.7910/DVN/0NCLP1',
        'file_id': 11634643,
        'file_sha256': sha(RAW_FILE),
        'file_size_bytes': RAW_FILE.stat().st_size,
        'recording': 'DNa02 single-cell ephys + ball yaw velocity, Orco-LexA/CsChrimson fictive odor',
        'ephys_sample_rate_Hz': fs,
        'ball_sample_rate_Hz': ball_fs,
        'continuous_duration_s': float(ephys_t[-1]),
        'stimulus_on_count': int(len(on)),
        'stimulus_duration_s_median': float(np.median((off-on)/fs)),
        'stimulus_onsets_s_first_five': ephys_t[on[:5]].tolist(),
        'yaw_variable_min_max': [float(np.min(yaw)), float(np.max(yaw))],
        'directional_labels_present': False,
        'directional_comparison_valid': False,
        'reason': 'Archive file contains a single 0/5 stimulus channel and yaw, but no per-trial left/right/both label. Original MATLAB grouped trial types by separate filenames; those labels are absent from this file.'
    }


def main():
    global SOURCE, NODES, COUNTS, RAW_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--nodes-side', type=Path, default=NODES)
    parser.add_argument('--outdegree', type=Path, default=COUNTS)
    parser.add_argument('--live-raw', type=Path, default=RAW_FILE)
    parser.add_argument('--skip-live', action='store_true', help='Review simulation without downloading the 63 MB Dataverse file')
    parser.add_argument('--out', type=Path, default=HERE / 'RESULT.json')
    args = parser.parse_args()
    SOURCE, NODES, COUNTS, RAW_FILE = args.source, args.nodes_side, args.outdegree, args.live_raw
    out = {'schema': 'stage3_preliminary_readback_v1',
           'simulation': simulation_audit(), 'living_fly': None if args.skip_live else living_fly_audit(),
           'biological_validation': False}
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
