"""Read-only spatial proxy on the first 1 ms of a saved full-organism trace.

This measures changes in the model's normalized q, not spikes, synaptic input,
or a safe set of states that can be advanced analytically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live/odor_left_network_trace.npz')
IDS = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/node_ids.npy')
GRAPH = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/counts_pre_post.npz')
THRESHOLDS = (1e-6, 1e-4, 1e-3, 1e-2)
TOP_FRACTIONS = (.02, .03, .05, .10)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        while block := src.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--ids', type=Path, default=IDS)
    parser.add_argument('--degree', type=Path, default=GRAPH)
    parser.add_argument('--out', type=Path, default=HERE / 'SPATIAL_PROXY.json')
    args = parser.parse_args()
    with np.load(args.source, allow_pickle=False) as z:
        times = z['times_ms']
        ids = z['ids']
        q0, q1 = z['q'][:2]
    node_ids = np.load(args.ids, allow_pickle=False)
    if q0.shape != (166700,) or not np.array_equal(times[:2], [0, 1]):
        raise ValueError('Expected 166700 full-network states at 0 and 1 ms')
    if not np.array_equal(ids, node_ids):
        raise ValueError('Anatomical node order differs from trace')
    if args.degree.suffix == '.npy':
        degree = np.load(args.degree, allow_pickle=False).astype(np.int64)
    else:
        with np.load(args.degree, allow_pickle=False) as z:
            degree = np.diff(z['indptr']).astype(np.int64)
    if degree.shape != (166700,) or int(degree.sum()) != 25582938:
        raise ValueError('Unexpected graph or edge count')
    if not np.isfinite(q0).all() or not np.isfinite(q1).all():
        raise ValueError('Nonfinite full-network state')

    change = np.abs(q1 - q0)
    proxy = degree * change
    total_proxy = float(np.sum(proxy, dtype=np.float64))
    if not total_proxy > 0:
        raise ValueError('No measured variation')
    n, total_edges = len(change), int(degree.sum())
    threshold_rows = []
    for threshold in THRESHOLDS:
        active = change > threshold
        threshold_rows.append({
            'threshold_abs_delta_q': threshold,
            'neuron_fraction_above': float(np.count_nonzero(active) / n),
            'outgoing_edge_fraction_from_above': float(degree[active].sum() / total_edges),
            'outdegree_weighted_delta_q_fraction_above': float(proxy[active].sum() / total_proxy),
        })
    rank = np.argsort(-change, kind='stable')
    top_rows = []
    for fraction in TOP_FRACTIONS:
        k = int(round(fraction * n))
        selected = rank[:k]
        top_rows.append({
            'selected_neuron_fraction': float(k / n),
            'outgoing_edge_fraction': float(degree[selected].sum() / total_edges),
            'outdegree_weighted_delta_q_fraction': float(proxy[selected].sum() / total_proxy),
            'min_selected_abs_delta_q': float(change[selected[-1]]),
        })
    result = {
        'schema': 'spatial_q_change_proxy_v1',
        'times_ms': [0, 1],
        'neuron_count': n,
        'outgoing_edge_count': total_edges,
        'source_hashes': {str(p): digest(p) for p in (args.source, args.ids, args.degree)},
        'ids_exactly_equal_node_ids': True,
        'thresholds': threshold_rows,
        'rank_by_abs_delta_q': top_rows,
        'min_abs_delta_q': float(change.min()),
        'median_abs_delta_q': float(np.median(change)),
        'max_abs_delta_q': float(change.max()),
        'total_outdegree_weighted_delta_q': total_proxy,
        'limitation': ('q is a continuous modeled source coordinate, not a spike, received current, '
                       'or an integration eligibility certificate. Degree ignores weights, delays, '
                       'chemistry, and cancellation. This is one 1-ms interval of one organism.'),
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
