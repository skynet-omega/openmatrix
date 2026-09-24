"""Four substantive CPU fixtures; checks remain active under python -O."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from event_sparse import (EpochGuard, convolution, current_cpu, need,
                          partition_csr, prepare_ports, project_cpu)


def expect_failure(fn, text):
    try:
        fn()
    except ValueError as e:
        need(text in str(e), 'Unexpected failure: ' + str(e))
        return str(e)
    raise ValueError('Missing required rejection: ' + text)


def fixture_inputs():
    data = dict(rows=np.array([1, 4, 6]), q=np.array([.2, .4, .1]),
                s=np.array([.03, .12, .2]), tau=np.array([.005, .005*(1.+1e-12), .009]),
                ts=.005, times=np.array([.0007, .0007, .0007, .0009]),
                event_rows=np.array([0, 0, 0, 2]),
                jumps=np.array([0., 0., .1, .05]),
                setop=np.array([True, True, False, False]),
                posts=np.array([.8, .3, 0., 0.]), duration_ns=1000000)
    ptr = np.array([0, 3, 6, 9, 10, 10, 10, 10], dtype=np.int64)
    idx = np.array([0, 1, 4, 1, 4, 6, 1, 4, 6, 0], dtype=np.int32)
    weights = np.array([2., -3., 4., 1., -2., 3., -1., 2., -4., 1.], dtype=np.float64)
    caps = np.array([1., 2., 3., 4., 5., 6., 7.])
    visual = np.array([False, True, False, False, True, False, False])
    return data, ptr, idx, weights, caps, visual


def run_fixtures():
    data, ptr, idx, weights, caps, visual = fixture_inputs()
    ports = prepare_ports(data, len(caps))
    part = partition_csr(ptr, idx, weights, ports, len(caps))
    q, s = project_cpu(ports, .0008)
    release = np.zeros(len(caps)); release[ports.rows] = s
    records = []
    # 1. Partition, exact/near-equal taus, empty receptor, caps and signed visual channels.
    need(np.array_equal(np.sort(np.r_[part.port_positions, part.continuous_positions]), np.arange(len(idx))), 'Partition cover')
    errors = []
    for connected in (False, True):
        full = current_cpu(ptr, idx, weights, release, caps, visual, 1./30., connected)
        reduced = current_cpu(part.port_ptr, part.port_sources, part.port_weights, release, caps, visual, 1./30., connected)
        errors.append(float(np.max(np.abs(full-reduced))))
        need(np.array_equal(full, reduced), 'Synthetic currents differ')
        need(full[1, 0] > 0. and full[1, 1] > 0. and full[2, 0] < 0., 'Lost signed currents')
    t=.001
    need(abs(float(convolution(t,.005,.005))-(t/.005)*np.exp(-t/.005))<1e-16, 'Equal tau limit')
    records.append(dict(name='partition_caps_visual_and_heterogeneous_taus', passed=True, max_abs=max(errors)))
    # 2. Visibility, repeated SET ordinal, ADD after SET, and speculative rollback.
    before = project_cpu(ports, np.nextafter(.0007, 0.))[0]
    exact = project_cpu(ports, .0007)[0]
    need(abs(exact[0]-.4)<1e-15 and before[0]<.2, 'SET order or event visibility')
    project_cpu(ports, .001)
    q2,s2 = project_cpu(ports, .0008)
    need(np.array_equal(q,q2) and np.array_equal(s,s2), 'Speculative time mutated projection')
    records.append(dict(name='late_within_trial_repeated_SET_ADD_and_rollback', passed=True, q_at_event=float(exact[0])))
    # 3. Arrival after an already answered query must invalidate, never silently patch history.
    guard = EpochGuard(part.weight_version,ports.version)
    guard.query(.0008,part.weight_version,ports.version)
    message = expect_failure(lambda:guard.insert_event(.0007),'Late event')
    expect_failure(lambda:guard.query(.0006,part.weight_version,ports.version),'invalidated')
    records.append(dict(name='late_inserted_event_invalidates_epoch',passed=True,message=message))
    # 4. An epoch/version mismatch is explicit and cannot disappear under -O.
    guard = EpochGuard(part.weight_version,ports.version)
    message = expect_failure(lambda:guard.query(.0008,'wrong-version',ports.version),'Wrong effective W')
    expect_failure(lambda:guard.query(.0008,part.weight_version,'wrong-events'),'Wrong event version')
    records.append(dict(name='wrong_W_or_event_version_rejected',passed=True,message=message))
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();records=run_fixtures()
    result=dict(schema='event_sparse_cpu_fixtures_v1',fixtures=records,optimized=not __debug__,wall_s=time.perf_counter()-start)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
