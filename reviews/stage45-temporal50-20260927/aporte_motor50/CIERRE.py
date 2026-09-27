"""Independent final contrast from complete recorded arms; no model execution."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import resource
resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    start = time.process_time()
    arms = ['p00', 'p01', 'p10', 'p11', 'm00', 'm01', 'm10', 'm11']
    coeff = [1, -1, -1, 1, -1, 1, 1, -1]
    queue = json.loads((ROOT/'QUEUE_RESULT.json').read_text())
    need(queue['status'] == 'COMPLETE', 'Wait for all arms')
    need([a['arm'] for a in queue['arms']] == arms, 'Eight complete arms required')
    need(all(a['status'] == 'COMPLETE' and a['committed_ms'] == 140
             for a in queue['arms']), 'Incomplete arm')
    neural, yaw = [], []
    bodykeys = ('qpos', 'qvel', 'position_mm', 'yaw_delta_deg', 'contact_active',
                'contact_force_N', 'generalized_force_native', 'energy_motor_J')
    body = {k: True for k in bodykeys}
    first_body = None
    prop_equal = True
    reference_prefix = npz(ROOT/'reference/prefix49.npz')
    prefix_equal = True
    targets = {}
    forward_max = {}
    hashes = {}
    for arm in arms:
        p = ROOT/arm
        t = npz(p/'traces.npz')
        with np.load(p/'input_and_observers.npz', allow_pickle=False) as o:
            need(np.array_equal(o['DN_ids'], [10045,10056,10118,10065,523769,10360]),
                 'DN identities')
            q, prop = o['DN_q'].copy(), o['proprioception'].copy()
        need(q.shape == (140, 6), 'DN shape')
        need(np.array_equal(q[:, :4], t['DN_q_actual']), 'DN capture mismatch')
        need(np.array_equal(t['paso'], np.arange(3001, 3141)), 'Window ownership')
        neural.append([float(row[2])-float(row[3]) for row in q])
        yaw.append([float(v)*180.0/math.pi for v in t['neural_yaw_unapplied_rad_s']])
        prefix_equal &= all(np.array_equal(t[k][:10], v)
                            for k, v in reference_prefix.items())
        if first_body is None:
            first_body = {k: t[k].copy() for k in bodykeys}
            first_prop = prop.copy()
        for k in bodykeys:
            body[k] &= np.array_equal(t[k], first_body[k])
        prop_equal &= np.array_equal(prop, first_prop)
        forward_max[arm] = float(np.max(t['command_forward_mm_s']))
        need(np.all(t['command_yaw_rate_rad_s'] == 0), 'Unexpected applied yaw')
        d = npz(p/'dng100_observed.npz')
        need(np.array_equal(d['ids'], [10045,10056]), 'DNg identities')
        f = d['records'][:, :128].reshape(-1, 4, 2, 16)
        need(np.isfinite(f).all(), 'Nonfinite DNg')
        targets[arm] = dict(minimum=float(f[..., 11].min()), maximum=float(f[..., 11].max()))
        if arm == 'p00':
            dng_for_corruption = d
        hashes[arm] = {name: hashlib.sha256((p/name).read_bytes()).hexdigest()
                       for name in ('traces.npz', 'input_and_observers.npz', 'dng100_observed.npz')}
    # Independent coefficient sum: no author contrast/window/gate function used.
    def independent(values):
        return math.fsum(c*values[a][s] for s in range(10, 130)
                         for a, c in enumerate(coeff))/240.0
    j_neural, j_yaw = independent(neural), independent(yaw)
    gate = abs(j_neural) >= 0.000016 and abs(j_yaw) >= 0.02
    author = json.loads((ROOT/'RESULTADOS.json').read_text())
    author_neural = author['contrasts']['DNb05']['mean_J']
    author_yaw = author['contrasts']['raw_yaw_deg_s']['mean_J']
    need(gate == author['decision']['screen_pass'], 'Different scientific decision')
    # One additional real-record corruption, using only the supplemental checker.
    sys.path.insert(0, str(ROOT))
    from verify_dng_law50 import check_law
    reference = npz(ROOT/'reference/frozen_selection49.npz')
    unaltered = check_law(dng_for_corruption, reference)
    changed = {k: v.copy() for k, v in dng_for_corruption.items()}
    changed['records'][0, 6] *= 2.0
    try:
        check_law(changed, reference)
    except ValueError as exc:
        corruption = dict(rejected=True, error=str(exc),
                          change='p00 first RHS left DNg positive gain doubled in memory only')
    else:
        raise ValueError('Changed frozen positive gain escaped verification')
    out = dict(scope='All8 recorded arms; independent signed coefficient sum, no CNS replay',
               window_samples=[11,130], J_neural=j_neural, J_yaw_deg_s=j_yaw,
               thresholds=dict(neural=0.000016, yaw_deg_s=0.02),
               fraction_of_threshold=dict(neural=abs(j_neural)/0.000016, yaw=abs(j_yaw)/0.02),
               difference_from_author=dict(neural=j_neural-author_neural, yaw_deg_s=j_yaw-author_yaw),
               same_scientific_decision=True, screen_pass=gate,
               classification='PROMETEDOR_NO_CONFIRMADO' if gate else 'DESCARTADO',
               DNg_targets=targets, forward_max_mm_s=forward_max,
               body_exact=body, proprioception_exact=prop_equal,
               all33_fields_common_prefix_exact=prefix_equal,
               real_record_supplement_check=unaltered, additional_corruption=corruption,
               scientific_ms=sum(a['committed_ms'] for a in queue['arms']),
               simulation_CPU_s=math.fsum(a['CPU_s'] for a in queue['arms']),
               simulation_wall_s=queue['wall_s'], source_hashes=hashes,
               CPU_s=time.process_time()-start,
               process_CPU_total_s=time.process_time(),
               peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
               new_neural_ms=0, GPU_calls=0, stage4_pass=False, stage5_pass=False)
    dest = HERE/'CIERRE.json'
    need(not dest.exists(), 'Preserve prior review result')
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('source_hashes','DNg_targets')}, allow_nan=False))


if __name__ == '__main__':
    main()
