"""Execute bounded CPU interface tests, never import the organism."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import resource
import time
from pathlib import Path
import numpy as np
from air_interface import AirInterface, Parameters, require

resource.setrlimit(resource.RLIMIT_CPU, (10, 12))
start = time.process_time()
h = Path(__file__).resolve().parent
require(not (h / 'AIR_INTERFACE.json').exists(), 'Preserve prior result')
m = AirInterface()
passed = []
def check(name, condition):
    require(bool(condition), name)
    passed.append(name)
def e(w, v=(0., 0., 0.), r=np.eye(3)):
    return m.encode(air_velocity_world_mm_s=w, body_velocity_world_mm_s=v, body_to_world=r)
def rejected(name, call):
    try:
        call()
    except (ValueError, TypeError):
        passed.append(name)
    else:
        raise RuntimeError(name)

check('zero relative air gives zero extra input', np.count_nonzero(e([0, 0, 0])) == 0)
check('co-moving air gives zero extra input', np.count_nonzero(e([20, -8, 3], [20, -8, 3])) == 0)
w, v, shift = np.array([40., 20., 0.]), np.array([2., -1., 0.]), np.array([100., -70., 5.])
check('Galilean invariance', np.array_equal(e(w, v), e(w + shift, v + shift)))
r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
check('world rotation covariance', np.array_equal(e(w, v), e(r @ w, r @ v, r)))
left, right = e([0, 100, 0]), e([0, -100, 0])
for side in ('L', 'R'):
    c = (m.sides == side) & m.is_c
    ee = (m.sides == side) & ~m.is_c
    check('reversing lateral air exchanges C/E on ' + side,
          np.all(left[c] == right[ee][0]) and np.all(left[ee] == right[c][0]))
    other = 'R' if side == 'L' else 'L'
    for family in (True, False):
        a = left[(m.sides == side) & (m.is_c == family)]
        b = right[(m.sides == other) & (m.is_c == family)]
        check('mirror symmetry per receptor ' + side + str(family), np.all(a == b[0]))
cases = np.array([e(wind) for wind in ([0,0,0],[0,100,0],[0,-100,0],[100,0,0],[-100,0,0],[0,0,100])])
check('bounded finite outputs', np.isfinite(cases).all() and cases.min() >= 0 and cases.max() <= 80)
check('declared planar proxy ignores vertical air', np.count_nonzero(cases[-1]) == 0)
base = np.full(166700, 0.25)
out = m.add_to_external_drive(base, air_velocity_world_mm_s=w, body_velocity_world_mm_s=v, body_to_world=np.eye(3))
outside = np.ones(len(base), bool); outside[m.rows] = False
check('additive update preserves other ports and base', np.array_equal(out[outside], base[outside])
      and np.all(base == 0.25) and np.array_equal(out[m.rows], base[m.rows] + e(w,v)))
rejected('reflection rejected', lambda: e(w, r=np.diag([1,1,-1])))
rejected('nonfinite velocity rejected', lambda: e([np.nan,0,0]))
rejected('wrong dimensions rejected', lambda: e([1,2]))
rejected('invalid parameter rejected', lambda: AirInterface(parameters=Parameters(0,80)))
rejected('goal position absent from API', lambda: m.encode(source_position=[1,2,3]))
np.savez_compressed(h/'air_interface_cases.npz', source_ids=m.ids, source_rows=m.rows,
                    wind_world_mm_s=np.array([[0,0,0],[0,100,0],[0,-100,0],[100,0,0],[-100,0,0],[0,0,100]]),
                    extra_drive=cases)
result = dict(status='EXECUTED_CPU_ENGINEERING_INTERFACE_NOT_ORGANISM_VALIDATION',
    parameters=dict(half_speed_mm_s=100., max_drive_model_units=80., projection=m.projection.tolist()),
    family_convention='C positive pull, E negative push; assumed mechanics projection',
    total_counts_by_side={s:int(np.sum(m.sides==s)) for s in ('L','R')},
    normalization='Same response per receptor; no rescaling for unequal anatomical populations',
    state='Stateless; caller owns sampling clock and checkpoint', passed=passed,
    tested_cases=len(cases), CPU_s=time.process_time()-start, process_CPU_total_s=time.process_time(),
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    CNS_steps=0, GPU_calls=0, download_bytes=0)
(h/'AIR_INTERFACE.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result, indent=2))
