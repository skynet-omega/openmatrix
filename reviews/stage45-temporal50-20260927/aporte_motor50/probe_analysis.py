"""Evaluator fixtures only: never loads an arm or executes a neuron/body/GPU."""
from pathlib import Path
import hashlib
import json
import resource
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze50 as a


def fixture():
    n = 2240
    slots = np.tile(np.arange(16), 140)
    duration = np.where(slots % 2, 125000, 62500).astype(np.int64)
    r = np.zeros((n, 140), np.float64)
    f = r[:, :128].reshape(n, 4, 2, 16)
    f[..., 5] = 1.0  # theta; zero net/drive => negative margin
    f[..., 6] = 1.0  # gain
    f[..., 8] = 1.0  # base rate
    f[..., 9] = 1.0  # tau
    f[..., 10] = -1.0
    f[..., 12] = 1.0  # final rate
    f[..., 15] = np.array([0., .5, .75, 0.])[None, :, None]
    r[:, 129] = duration * 1e-9
    r[:, 130] = duration * 1e-9
    f[..., 14] = f[..., 15] * r[:, 129, None, None]
    f[:, 3, :, 14] = np.nextafter(r[:, 130], r[:, 128])[:, None]
    r[:, 138] = 1.0
    names = ('state net positive_aux negative_aux drive theta gain base_target '
             'base_rate tau margin final_target final_rate derivative '
             'evaluation_time_s stage_fraction').split()
    d = dict(ids=a.DN_IDS[:2].copy(), fields=np.array(names), records=r,
             trials=np.ones(n, np.int64), offsets=np.arange(n+1),
             ms=np.repeat(np.arange(3001, 3141), 16),
             committed=np.tile([False, True], 1120),
             accepted=np.ones(n, np.int64), rejected=np.zeros(n, np.int64),
             start_ns=47486000000 + np.repeat(np.arange(140), 16)*1000000
                      + (slots//2)*125000,
             duration_ns=duration, epoch=np.arange(n))
    return d, {'DN_q_actual': np.zeros((140, 4))}


def main():
    start = time.process_time()
    protocol = json.loads((HERE.parent/'ANALYSIS_PROTOCOL.json').read_text())
    hashes = {p: hashlib.sha256((HERE.parent/p).read_bytes()).hexdigest() == h
              for p, h in protocol['sources'].items()}
    a.need(all(hashes.values()), 'Analysis source changed during review')
    x = {arm: np.zeros(140) for arm in a.ARMS}
    x['p00'][10:130] = 6.
    x['m00'][10:130] = 2.
    x['p00'][:10] = 1e6
    x['p00'][130:] = -2e6
    a.need(a.contrast(x)['mean_J'] == 2., 'J sign, factor, or window mismatch')
    a.need(a.gate(-1.6e-5, -.02)['screen_pass'], 'Negative inclusive gates')
    d, t = fixture()
    base = a.check_dng(d, t)
    cases = {}
    for name in ('negative_tau', 'illegal_positive_target', 'RHS_clock_shift'):
        z = {k: v.copy() for k, v in d.items()}
        f = z['records'][:, :128].reshape(2240, 4, 2, 16)
        if name == 'negative_tau':
            f[0, 0, 0, 9] = -1.
        elif name == 'illegal_positive_target':
            # Negative margin and positive gain require target exactly zero.
            f[0, 0, 0, [7, 11]] = .5
            f[0, 0, 0, 13] = .5
        else:
            f[0, 0, 0, 14] += 1.
        try:
            result = a.check_dng(z, t)
        except (ValueError, KeyError) as exc:
            cases[name] = dict(rejected=True, error=str(exc))
        else:
            cases[name] = dict(rejected=False,
                               reported_target_max=result['target_max'])
    out = dict(scope='Synthetic record fixtures only; no partial50 data read',
               frozen_hashes_match=hashes, window_J_and_negative_gates=True,
               synthetic_baseline=base, cases=cases,
               CPU_s=time.process_time()-start,
               peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               neural_ms=0, body_steps=0, GPU_calls=0)
    destination = HERE/'ANALYSIS_PROBE.json'
    a.need(not destination.exists(), 'Preserve earlier probe')
    destination.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(out, allow_nan=False))


if __name__ == '__main__':
    main()
