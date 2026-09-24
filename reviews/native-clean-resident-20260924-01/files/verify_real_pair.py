"""Reconstruct full-state equality from both independently saved organism runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LEFT = HERE/'real_reference_01'
RIGHT = HERE/'real_resident_02'
NAMES = ('session','prosthesis','published','effective_operator')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b, za, zb, path, changes, limit=50):
    if len(changes) > limit:
        return
    if isinstance(a, dict) and set(a) == {'__array__'}:
        if not isinstance(b, dict) or set(b) != {'__array__'}:
            changes.append({'path':path,'reason':'array marker mismatch'})
            return
        x = za[a['__array__']]
        y = zb[b['__array__']]
        if x.dtype != y.dtype or x.shape != y.shape:
            changes.append({'path':path,'reason':'dtype or shape mismatch',
                            'left_shape':x.shape,'right_shape':y.shape,
                            'left_dtype':str(x.dtype),'right_dtype':str(y.dtype)})
            return
        same = np.array_equal(x,y,equal_nan=x.dtype.kind in 'fc')
        if not same:
            entry = {'path':path,'reason':'array values differ','shape':x.shape,
                     'dtype':str(x.dtype)}
            if x.dtype.kind in 'fc':
                finite = np.isfinite(x) & np.isfinite(y)
                delta = np.abs(x[finite]-y[finite])
                entry['finite_max_abs'] = float(np.max(delta)) if delta.size else None
                entry['nonfinite_mismatch_count'] = int(np.count_nonzero(
                    ~finite & ~(np.isnan(x)&np.isnan(y))))
            changes.append(entry)
        return
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            changes.append({'path':path,'reason':'dict keys differ'})
            return
        for key in a:
            compare(a[key],b[key],za,zb,path+'/'+str(key),changes,limit)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            changes.append({'path':path,'reason':'list length differs'})
            return
        for index,(x,y) in enumerate(zip(a,b)):
            compare(x,y,za,zb,path+'/'+str(index),changes,limit)
    elif type(a) is not type(b) or a != b:
        changes.append({'path':path,'reason':'scalar differs',
                        'left':repr(a)[:100],'right':repr(b)[:100]})


def main() -> int:
    left_result = json.loads((LEFT/'RESULT.json').read_text())
    right_result = json.loads((RIGHT/'RESULT.json').read_text())
    out = {'schema':'native_clean_full_state_pair_verify_v1',
           'reference_complete':left_result['status']=='COMPLETE',
           'candidate_complete':right_result['status']=='COMPLETE',
           'files':{},'scientific_state_exact':False,
           'speed_result_admissible':False,
           'user_target_5s_in_600s_demonstrated':False}
    for name in NAMES:
        paths = [LEFT/'state'/f'{name}.{suffix}' for suffix in ('json','npz')]
        other = [RIGHT/'state'/f'{name}.{suffix}' for suffix in ('json','npz')]
        a,b = json.loads(paths[0].read_text()),json.loads(other[0].read_text())
        with np.load(paths[1],allow_pickle=False) as za, np.load(other[1],allow_pickle=False) as zb:
            changes = []
            compare(a,b,za,zb,name,changes)
        out['files'][name] = {'exact_semantic':not changes,'changes':changes,
            'json_sha256_equal':digest(paths[0])==digest(other[0]),
            'npz_sha256_equal':digest(paths[1])==digest(other[1])}
    left_boundary = LEFT/'state/boundary.json'
    right_boundary = RIGHT/'state/boundary.json'
    out['files']['boundary'] = {'exact_bytes':digest(left_boundary)==digest(right_boundary)}
    out['clock_equal'] = left_result.get('time_ns') == right_result.get('time_ns')
    a = left_result.get('runtime',{}).get('CNS',{})
    b = right_result.get('runtime',{}).get('CNS',{})
    out['trial_counts_equal'] = all(a.get(k)==b.get(k) for k in
                                    ('epochs','accepted','rejected','mandatory_event_boundaries'))
    out['scientific_state_exact'] = (out['reference_complete'] and out['candidate_complete']
        and out['clock_equal'] and out['trial_counts_equal']
        and all(v.get('exact_semantic',v.get('exact_bytes')) for v in out['files'].values()))
    # One corruption check of each data representation, without touching disk.
    scalar_changes = []
    compare({'x':1},{'x':2},{},{},'corruption',scalar_changes)
    array_changes = []
    compare({'__array__':'a'},{'__array__':'b'},
            {'a':np.array([1.,2.])},{'b':np.array([1.,3.])},
            'corruption',array_changes)
    out['corruption_checks_pass'] = bool(scalar_changes and array_changes)
    out['step_wall_reference_s'] = left_result.get('step_wall_s')
    out['step_wall_candidate_s'] = right_result.get('step_wall_s')
    out['candidate_over_reference_step_wall_ratio'] = (
        right_result['step_wall_s']/left_result['step_wall_s'])
    out['speed_result_admissible'] = out['scientific_state_exact'] and out['corruption_checks_pass']
    (HERE/'REAL_PAIR_VERIFY_01.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'scientific_state_exact':out['scientific_state_exact'],
        'candidate_over_reference_step_wall_ratio':out['candidate_over_reference_step_wall_ratio'],
        'session_changes':out['files']['session']['changes'][:10]},allow_nan=False))
    return 0 if out['corruption_checks_pass'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
