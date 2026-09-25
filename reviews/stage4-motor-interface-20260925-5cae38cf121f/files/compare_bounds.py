"""Execute ChatGPT's unmodified function and compare to integer-duration code."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from chatgpt_bounds_original import cota_monotona
from reader_bounds import monotone_bound, need


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root/'PLAN.json').read_text())
    samples = []
    results = {}
    for name in ('long_identity', 'long_equal'):
        row = plan['inputs'][name]
        path = Path(row['path'])
        need(hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], 'Input changed')
        with np.load(path, allow_pickle=False) as z:
            q = z['DN_q_usada'][40:]
            b = z['DN_baseline'][40:]
            x = (q[:,2]-b[:,2])-(q[:,3]-b[:,3])
            ticks = np.diff(z['CNS_time_ns'][39:])
            samples.append((x, ticks))
    cases = dict(zip(('identity', 'equal'), samples))
    cases['shared_equal_minus_identity'] = (np.r_[samples[1][0], samples[0][0]],
                                           np.r_[samples[1][1], -samples[0][1]])
    for name, (x, ticks) in cases.items():
        ours, _ = monotone_bound(x, ticks)
        donor = cota_monotona(x, ticks.astype(float)*1e-9)
        err = max(abs(ours['lower_deg']-donor['min_grados_mando']),
                  abs(ours['upper_deg']-donor['max_grados_mando']))
        need(err <= 1e-12, 'Independent implementations disagree')
        results[name] = dict(integer_duration=ours, chatgpt_float_duration=donor,
                             maximum_absolute_difference_deg=err)
    out = dict(results=results, donor_message_id='af1741c4-7b88-43d1-bcf1-ae1cd6096b17',
               donor_sha256=hashlib.sha256((root/'chatgpt_bounds_original.py').read_bytes()).hexdigest(),
               execution='local; ChatGPT did not execute these real arrays',
               stage_admission=False, full_organism_runs=0)
    need(not args.out.exists(), 'Preserve previous result')
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(out, allow_nan=False))


if __name__ == '__main__':
    main()
