"""A prespecified cell-identity control preserving per-antenna input histograms.

This is a synthetic control, not a physiological odor or a neural simulation.
"""
from collections import Counter
import csv
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 'post47-composition-20260927-first-control'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def calculate():
    anatomy_path = HERE / 'ANATOMIA_SELECCIONADA.json'
    anatomy = json.loads(anatomy_path.read_text())
    preflight = json.loads((HERE / 'PREFLIGHT_RESULT.json').read_text())
    template = {(r['side'], 'ORN_' + r['glomerulus']): r for r in preflight['profiles']
                if r['pattern'] == 'empirical_shape_matched'}
    records, summary = [], {}
    for side in ('L', 'R'):
        cells = sorted((r for r in anatomy['records'] if r['side'] == side), key=lambda r: r['id'])
        increments = [Q(template[side, r['type']]['increment_rational']) for r in cells]
        order = sorted(range(len(cells)), key=lambda i: hashlib.sha256(
            f'{SEED}|{side}|{cells[i]["id"]}'.encode()).hexdigest())
        shuffled = [increments[i] for i in order]
        need(Counter(increments) == Counter(shuffled), 'Changed input histogram')
        need(sum(increments) == sum(shuffled), 'Changed input sum')
        changed = sum(a != b for a, b in zip(increments, shuffled))
        need(changed > 0, 'Identity permutation has no effect on assignments')
        for cell, original, permuted in zip(cells, increments, shuffled):
            records.append(dict(id=cell['id'], type=cell['type'], side=side,
                baseline_Hz=template[side, cell['type']]['common_baseline_Hz'],
                profile_increment_rational=str(original),
                permuted_increment_rational=str(permuted),
                profile_increment_Hz=float(original), permuted_increment_Hz=float(permuted)))
        summary[side] = dict(cells=len(cells), changed_assignments=changed,
            identical_increment_histogram=True, exact_increment_sum_rational=str(sum(increments)),
            increment_sum=float(sum(increments)))
    return dict(seed=SEED, ordering='SHA256(seed|side|bodyId), first prespecified control only',
        anatomy_sha256=hashlib.sha256(anatomy_path.read_bytes()).hexdigest(), sides=summary,
        preserves='Nominal incremental-rate histogram and sum per antenna; same cell-specific basal input and waveform.',
        limit='Not matching synaptic current or activity after terminal modulation. Mixed increments within a receptor type are a synthetic control. No biological concentration assigned.',
        new_neural_steps=0, records=records)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    data = calculate()
    path = HERE / 'PERMUTATION_RESULT.json'
    if args.verify:
        need(data == json.loads(path.read_text()), 'Different permutation control')
    else:
        with path.open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({k: v for k, v in data.items() if k != 'records'}, indent=2))
