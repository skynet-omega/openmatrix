"""Descriptive native DNa02 afferent balance; no new simulation or fitting."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work/stage3_upstream_pfg_test_20260919/live')
DEFAULT_NODES = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/nodes.parquet')
ARMS = ('odor_left', 'odor_right', 'uniform', 'sham')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--nodes', type=Path, default=DEFAULT_NODES)
    p.add_argument('--out', type=Path, default=HERE / 'AFFERENT_RANKING.json')
    a = p.parse_args()
    source = {arm: load(a.source/f'{arm}_native_transmission.npz') for arm in ARMS}
    first = source['sham']
    ids = first['afferent_ids']
    weights = first['signed_weights']
    caps = first['caps']
    if weights.shape != (2, 2208) or ids.shape != (2208,) or not np.array_equal(first['target_ids'], [523769, 10360]):
        raise ValueError('Unexpected native afferent layout')
    for arm, data in source.items():
        for key in ('afferent_ids', 'signed_weights', 'caps', 'times_ms'):
            if not np.array_equal(data[key], first[key]):
                raise ValueError(f'{arm}: altered {key}')
        if not np.array_equal(data['times_ms'], np.arange(336)):
            raise ValueError(f'{arm}: wrong time axis')
        if not all(np.isfinite(data[k]).all() for k in ('transmission', 'net')):
            raise ValueError(f'{arm}: nonfinite inputs')
    nodes = (pd.DataFrame(json.loads(a.nodes.read_text())) if a.nodes.suffix == '.json'
             else pd.read_parquet(a.nodes)).set_index('bodyId')
    missing = set(map(int, ids)) - set(nodes.index)
    if missing:
        raise ValueError(f'Missing {len(missing)} afferent metadata IDs')
    annotations = nodes.loc[ids]
    cell_type = annotations['type'].fillna('unknown').astype(str).to_numpy()
    side = annotations['rootSide'].fillna(annotations['somaSide'] if 'somaSide' in annotations else 'unknown').astype(str).to_numpy()
    lo, hi = 161, 311
    sel = slice(lo, hi+1)
    rates = {arm: data['transmission'][sel]*caps[None, :] for arm, data in source.items()}
    validation = {}
    for arm in ARMS:
        predicted = rates[arm] @ weights.T
        observed = source[arm]['net'][sel]
        validation[arm] = float(np.max(np.abs(predicted-observed)))
        if validation[arm] > 1e-8:
            raise ValueError(f'{arm}: native readback reconstruction failed')

    bilateral_weight = weights[0]-weights[1]
    sham_mean = rates['sham'].mean(axis=0)

    def describe(contrast, arm):
        contribution = contrast*bilateral_weight
        groups = {}
        for i in range(len(ids)):
            label = f'{cell_type[i]}|{side[i]}'
            groups[label] = groups.get(label, 0.)+float(contribution[i])
        top_groups = sorted(groups.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        ranking = np.argsort(-np.abs(contribution))[:20]
        top_cells = [dict(id=int(ids[i]), type=cell_type[i], side=side[i],
                          contribution_L_minus_R=float(contribution[i]),
                          weight_L=float(weights[0,i]), weight_R=float(weights[1,i]),
                          mean_transmission_Hz_change=float(contrast[i])) for i in ranking]
        target = source[arm]['net'][sel].mean(axis=0)-source['sham']['net'][sel].mean(axis=0) if arm!='sham' else source['sham']['net'][sel].mean(axis=0)
        delta = float(target[0]-target[1])
        if abs(float(contribution.sum())-delta) > 1e-8:
            raise ValueError(f'{arm}: contrast sum mismatch')
        return {'DNa02_native_net_L_minus_R': delta,
                'top_groups_by_abs_contribution': [{'group': k, 'contribution_L_minus_R': v} for k,v in top_groups],
                'top_cells_by_abs_contribution': top_cells}

    result = {'schema': 'native_DNa02_afferent_rank_v1',
              'scope': 'Descriptive weighted-input accounting at committed 1-ms endpoints; ranking is not causal attribution.',
              'window_ms': [lo, hi],
              'afferents': len(ids),
              'reconstruction_max_abs': validation,
              'arms_vs_sham': {arm: describe(rates[arm].mean(axis=0)-sham_mean, arm) for arm in ARMS if arm!='sham'},
              'sham_baseline': describe(sham_mean, 'sham')}
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'reconstruction_max_abs':validation,
                      'signed_net_contrasts':{k:v['DNa02_native_net_L_minus_R'] for k,v in result['arms_vs_sham'].items()},
                      'sham_signed_net':result['sham_baseline']['DNa02_native_net_L_minus_R']},indent=2))

if __name__ == '__main__':
    main()
