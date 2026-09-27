"""Exact arithmetic for proposed sensory controls; no organism or fitted law.

The boundary is a nominal peripheral rate increment, BEFORE terminal modulation.
Matching this sum does not imply matching effective synaptic input downstream.
"""
import argparse
import csv
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def calculate(folder):
    started = time.process_time()
    folder = Path(folder)
    sources = json.loads((folder / 'INPUT_HASHES.json').read_text())
    for name, digest in sources.items():
        need(hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest,
             'Changed input: ' + name)
    with (folder / 'PERFIL_1_HEXANOL.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    contract = json.loads((folder / 'CONTRACT.json').read_text())
    names = [r['receptor'] for r in rows]
    need(len(rows) == 15 and len(set(names)) == 15, 'Wrong profile identities')
    inc = {r['receptor']: Q(r['increment_Hz']) for r in rows}
    basal = {r['receptor']: Q(r['baseline_Hz']) for r in rows}
    count = {r['receptor']: int(r['model_cells']) for r in rows}
    sides = {k: contract['anatomy'][k]['sides'] for k in names}
    for k in names:
        need(set(sides[k]) <= {'L', 'R', 'unknown'}, 'Unrecognized antenna label')
        need(sum(sides[k].values()) == count[k], 'Population count mismatch')
        need(inc[k] >= 0 and basal[k] >= 0, 'This design assumes nonnegative source means')
    total = sum(count[k] * inc[k] for k in names)
    legacy = count['Or42b'] * inc['Or42b'] / 2
    out = dict(scope='Engineering input preflight only; not a chemical concentration or neural result',
               source_hashes=sources, rows=len(rows),
               mapped_neurons=sum(count.values()),
               unknown_side_neurons=sum(sides[k].get('unknown', 0) for k in names),
               full_profile_weighted_increment=float(total),
               selective45_weighted_increment=float(legacy),
               unnormalized_increment_ratio=float(total / legacy),
               unknown_side_increment_at_full_profile=float(sum(
                   sides[k].get('unknown', 0) * inc[k] for k in names)),
               sides={}, profiles=[])
    for side in ('L', 'R'):
        n = {k: sides[k].get(side, 0) for k in names}
        n_total = sum(n.values())
        target = n['Or42b'] * inc['Or42b'] / 2
        full = sum(n[k] * inc[k] for k in names)
        scale = target / full
        patterns = {
            'selective_DM1_matched': {k: inc[k] / 2 if k == 'Or42b' else Q(0) for k in names},
            'empirical_shape_matched': {k: scale * inc[k] for k in names},
            'flat_population_matched': {k: target / n_total for k in names},
            'sham_common_baseline': {k: Q(0) for k in names},
        }
        for label, pattern in patterns.items():
            need(sum(n[k] * pattern[k] for k in names) ==
                 (0 if label == 'sham_common_baseline' else target), 'Unequal nominal increment')
            for k in names:
                out['profiles'].append(dict(pattern=label, side=side, receptor=k,
                    glomerulus=contract['anatomy'][k]['type'][4:], cells=n[k],
                    common_baseline_Hz=float(basal[k]), increment_Hz=float(pattern[k]),
                    increment_rational=str(pattern[k]), total_nominal_Hz=float(basal[k] + pattern[k])))
        out['sides'][side] = dict(known_cells=n_total,
            common_baseline_weighted_sum=float(sum(n[k] * basal[k] for k in names)),
            target_weighted_increment=float(target), full_profile_weighted_increment=float(full),
            empirical_scale=float(scale), empirical_scale_rational=str(scale),
            flat_increment_per_cell=float(target / n_total), exact_rational_matching=True)
    out['unknown_side_policy'] = ('No invented antenna assignment. Keep their inherited input/background '
        'policy and recurrent connections unchanged across arms; recurrent activity remains free to differ. '
        'No new direct peripheral stimulus. This is a partial profile, '
        'not absence of biological responses in these neurons.')
    out['not_controlled_by_matching'] = ['terminal modulation', 'synaptic weighting and connectivity',
        'receptor-specific dynamics', 'population synchrony', 'postsynaptic gain', 'chemical concentration']
    out['new_neural_steps'] = 0
    out['CPU_s'] = time.process_time() - started
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    value = calculate(HERE)
    output = HERE / 'PREFLIGHT_RESULT.json'
    if args.verify:
        old = json.loads(output.read_text())
        old.pop('CPU_s'); check = dict(value); check.pop('CPU_s')
        need(old == check, 'Recomputed input design differs')
    else:
        with output.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        with (HERE / 'MATCHED_PROFILES.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(value['profiles'][0]))
            writer.writeheader(); writer.writerows(value['profiles'])
    print(json.dumps({k: v for k, v in value.items() if k != 'profiles'}, indent=2))


if __name__ == '__main__':
    main()
