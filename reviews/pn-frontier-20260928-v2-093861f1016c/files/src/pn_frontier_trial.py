"""Bounded, registered intervention at a declared generic PN emission boundary.

Reuses the qualified historical loader explicitly. The writer touches only the
FP32 CSR scratch vector; specialized PN routes and neuronal states remain live.
Held signals are evaluator interventions, never an exact RHS replay or a policy.
"""
from pathlib import Path
import json
import sys
import time
import resource
import numpy as np
from session_io import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def bounded_total_control(mean, target, caps):
    """Match aggregate caps*q by proportional headroom, without clipping.

    This is a scalar-dose control, not a direction-pure or biological stimulus.
    The deterministic redistribution itself is part of the declared intervention.
    """
    m, t, caps = [np.asarray(x, dtype=float) for x in (mean, target, caps)]
    require(m.ndim == t.ndim == 2 and m.shape == t.shape and caps.shape == (m.shape[1],), 'dose shape')
    require(all(np.isfinite(x).all() for x in (m, t, caps)) and np.all(caps > 0), 'dose finite/caps')
    require(np.all((m >= 0) & (m <= 1)) and np.all((t >= 0) & (t <= 1)), 'dose domain')
    current, wanted = m @ caps, t @ caps
    result = m.copy()
    for j, (a, b) in enumerate(zip(current, wanted)):
        if b > a:
            headroom = (1 - m[j]) @ caps
            require(headroom > 0, 'no positive dose headroom')
            result[j] += (b - a) / headroom * (1 - m[j])
        elif b < a:
            require(a > 0, 'no removable dose')
            result[j] *= b / a
    require(np.all((result >= 0) & (result <= 1)), 'dose operation escaped domain')
    return result.astype(np.float32)


def baseline_comparison(trace, reference, contract):
    w = slice(contract['window_ms'][0] - 1, contract['window_ms'][1])
    q = lambda t: (t['DN_q_usada'] - t['DN_baseline'])[:, 2:4]
    # Bound the pointwise error, not only cancellation in a window mean.
    qerr = float(np.max(np.abs(np.diff(q(trace)[w], axis=1) - np.diff(q(reference)[w], axis=1))))
    yerr = float(np.max(np.abs(np.rad2deg(trace['command_yaw_rate_rad_s'][w] - reference['command_yaw_rate_rad_s'][w]))))
    forward_equal = np.array_equal(trace['command_forward_mm_s'], reference['command_forward_mm_s'][:len(trace['command_forward_mm_s'])])
    return {'maximum_window_DNb_consumed_error': qerr, 'maximum_window_yaw_error_deg_s': yerr,
            'forward_identical': bool(forward_equal),
            'passed': qerr <= contract['baseline_max_DNb_error_q'] and
                      yerr <= contract['baseline_max_yaw_error_deg_s'] and bool(forward_equal)}


def make_installer(rows, enabled):
    def installer(brain, stimulus):
        import cupy as cp
        import fp32_operator
        import observe57
        from hold_probe56 import CUDA
        ob, undo_parent = observe57.installer(brain, stimulus)
        original_class = fp32_operator.FastCSR
        indices = cp.asarray(rows, dtype=cp.int64)
        values = cp.zeros(len(rows), dtype=cp.float32)
        bad = cp.zeros(1, dtype=cp.uint64)
        # Reuse the qualified writer56 kernel; only the explicit held vector
        # schedule changes. Keep its pre-write witnesses as well as PN55's
        # downstream consumed values. Historical source is never modified.
        mode = cp.asarray([1 if enabled else 2], dtype=cp.int32)
        before = {k:cp.zeros(686, cp.float32) for k in ('first','last','lo','hi')}
        before['counts'] = cp.zeros(686, cp.uint64)
        kernel = cp.RawKernel(CUDA, 'terminal56', options=('--fmad=false',))
        class Written(original_class):
            def __init__(self, owner, **kwargs):
                super().__init__(owner, **kwargs)
                previous = self.kernel
                def write(grid, block, args):
                    kernel((6,), (128,), (indices, args[4], values, mode,
                           before['first'],before['last'],before['lo'],before['hi'],before['counts'],bad))
                    return previous(grid, block, args)
                self.kernel = write
        fp32_operator.FastCSR = Written
        ob.frontier_writer = {'held': values, 'bad': bad, 'before':before}
        def undo():
            fp32_operator.FastCSR = original_class
            undo_parent()
        return ob, undo
    return installer


def evaluate(inputs, parameters, output):
    wall, cpu = time.monotonic(), time.process_time()
    out = Path(output)
    c = json.loads(Path(inputs['contract']).read_text())
    require(c['schema'] == 'matrix_pn_frontier_trial_v1', 'trial schema')
    require(set(parameters) == {'arm'} and parameters['arm'] in c['arms'], 'trial arm')
    arm = parameters['arm']; spec = c['arms'][arm]
    for key, expected in c['bindings'].items():
        require(sha256(inputs[key]) == expected, 'trial input changed: ' + key)
    # A failed fidelity check is a recorded scientific stop, not a retry request.
    for key in inputs:
        if key.startswith('gate_'):
            gate = json.loads(Path(inputs[key]).read_text())
            if not gate.get('gate_pass', False):
                result = {'status': 'NOT_RUN_AFTER_GATE', 'arm': arm, 'gate_pass': False,
                          'blocking_gate': key, 'new_CNS_ms': 0, 'stage_admission': None}
                save(out/'assessment.json', result)
                (out/'REPORT.md').write_text('# Intervención PN\n\nNo ejecutada: falló una cualificación previa.\n')
                return {'metrics': {'new_CNS_ms': 0, 'gate_pass': 0}, 'assessment': result}
    require(spec['duration_ms'] <= c['maximum_arm_ms'], 'trial duration')
    resource.setrlimit(resource.RLIMIT_CPU, (c['per_arm_CPU_s'], c['per_arm_CPU_s']+5))
    root = Path(inputs['loader']).resolve().parents[3]
    require(root.name == 'matrix', 'canonical loader location')
    paths = [Path(inputs['observer']).parent, Path(inputs['loader']).parent,
             Path(inputs['spatial_owner']).parent, Path(inputs['air_owner']).parent,
             Path(inputs['PN_observer']).parent, Path(inputs['writer_source']).parent]
    sys.path[:0] = [str(p.resolve()) for p in paths]
    from resume49 import build
    from continuation54 import SpatialOwner, install_motor, Intervals, step
    from pilot52_owners import AirOwner
    from observer52 import Panel
    import observe57
    import pn_probe55
    with np.load(inputs['support'], allow_pickle=False) as z:
        rows, ids = z['rows'].copy(), z['ids'].copy()
    require(len(rows) == len(ids) == 686 and len(np.unique(rows)) == 686, 'PN writer support')
    tapes = {}
    for side in ('L', 'R'):
        with np.load(inputs['tape_'+side], allow_pickle=False) as z:
            tapes[side] = z['first'][:spec['duration_ms']].copy()
    with np.load(inputs['reference_'+spec['side']], allow_pickle=False) as z:
        reference = {k: z[k][:spec['duration_ms']] for k in z.files}
    run = air = spatial = undo_motor = None
    traces, consumed, orn_consumed, pn_states, before_writer = [], [], [], [], []
    status = {'status': 'STARTED', 'arm': arm, 'new_CNS_ms': 0, 'attempted_CNS_ms': 0,
              'contract_sha256': sha256(inputs['contract']), 'stage_admission': None}
    try:
        run = build(root/c['checkpoint'], out/'capture', observer_installer=make_installer(rows, spec['mode'] != 'live'))
        h = run.obj.core.hybrid
        require(np.array_equal(h.brain.node_ids[rows], ids), 'runtime PN identity mismatch')
        require(run.session.adapter.core is None, 'writer installed too late')
        air = AirOwner(run, {'mode':'parent', 'air':0, 'odor':True}, c['prefix_ms'])
        spatial = SpatialOwner(run, spec['side'], c['prefix_ms'])
        undo_motor = install_motor(run, True); run.auditor = Intervals(run, True)
        caps = h.brain.r_max[rows].astype(float)
        mean = ((tapes['L'].astype(float)+tapes['R'])/2).astype(np.float32)
        hold = (bounded_total_control(mean, tapes[spec['side']], caps) if spec['mode']=='dose'
                else mean if spec['mode']=='common' else tapes[spec['side']])
        require(np.isfinite(hold).all() and np.all((hold >= 0) & (hold <= 1)), 'held signal domain')
        panel = Panel(h.brain.node_ids, True, run.stimulus.spec['ids'])
        initial_q = h.release().copy()
        from source_inventory import imported, verify
        executed = imported(); save(out/'EXECUTED_SOURCES.json', executed)
        np.savez_compressed(out/'stimulus.npz', PN_ids=ids, PN_rows=rows, caps=caps,
                            held=hold, native_first=tapes[spec['side']], initial_q=initial_q)
        expected_body = float(run.obj.body.data.time); initial_steps = int(run.obj.body.steps)
        print(json.dumps({'arm': arm, 'status':'RESTORED', 'mode':spec['mode']}), flush=True)
        for j in range(spec['duration_ms']):
            status['attempted_CNS_ms'] += 1
            save(out/'STATUS.json', status)
            observe57.reset(run)
            run.observer.frontier_writer['before']['counts'].fill(0)
            run.observer.frontier_writer['bad'].fill(0)
            run.observer.frontier_writer['held'].set(np.ascontiguousarray(hold[j]))
            row = step(run, spatial); air.check()
            require(int(run.observer.frontier_writer['bad'].get()[0]) == 0, 'invalid value consumed')
            for _ in range(40): expected_body += 2.5e-5
            require(float(run.obj.body.data.time) == expected_body and int(run.obj.body.steps) == initial_steps+40*(j+1), 'body clock recurrence')
            pn = pn_probe55.sample(run)
            pre = {k:v.get() for k,v in run.observer.frontier_writer['before'].items()}
            require(np.array_equal(pre['counts'], pn['counts']), 'writer/consumer call coverage')
            if spec['mode'] != 'live':
                for key in ('first','last','lo','hi'):
                    require(np.array_equal(pn[key], hold[j]), 'writer did not reach every PN CSR consumer call')
            else:
                for key in pn: require(np.array_equal(pre[key],pn[key]), 'identity writer changes its operand')
                for key in reference:
                    require(np.array_equal(row[key], reference[key][j]), 'live identity differs '+key)
                with np.load(inputs['tape_'+spec['side']], allow_pickle=False) as z:
                    for key in pn: require(np.array_equal(pn[key], z[key][j]), 'live PN witness differs '+key)
            traces.append(row); consumed.append(pn); orn_consumed.append(observe57.sample(run))
            before_writer.append(pre)
            q = h.release(); panel.record(q, row['CNS_time_ns']); pn_states.append(q[rows].copy())
            status['new_CNS_ms'] += 1
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < c['RAM_bytes'], 'RAM limit')
            import cupy as cp
            free, total = cp.cuda.runtime.memGetInfo()
            require(total-free < c['VRAM_bytes'], 'VRAM limit')
            if (j+1)%16 == 0 or j+1 == spec['duration_ms']:
                print(json.dumps({'arm':arm, 'ms':j+1, 'wall_s':time.monotonic()-wall}), flush=True)
        panel.flush(out/'wide_observation.npz')
        save(out/'EVENTS.json', run.session.events.audit)
        save(out/'SPATIAL_OWNER_FINAL.json', spatial.state())
        trace = {k: np.asarray([r[k] for r in traces]) for k in traces[0]}
        if spec['mode'] == 'self':
            baseline = baseline_comparison(trace, reference, c)
            status.update(baseline_fidelity=baseline, gate_pass=baseline['passed'])
        else: status['gate_pass'] = True
        np.savez_compressed(out/'neural_endpoints.npz', PN_ids=ids, PN_rows=rows, PN_q=pn_states, final_q=h.release())
        status.update(status='COMPLETE', initial_exact=run.initial['exact'], observer=run.observer.report())
        verify(executed)
    except BaseException as error:
        status.update(status='FAILED', gate_pass=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for name, values in [('traces',traces), ('PN_consumed',consumed), ('ORN_consumed',orn_consumed), ('PN_before_writer',before_writer)]:
            if values: np.savez_compressed(out/(name+'.npz'), **{k:np.asarray([r[k] for r in values]) for k in values[0]})
        if run is not None and run.observer.epochs: run.observer.flush(out/'dng100_observed.npz')
        cleanup_errors = []
        for fn in (None if spatial is None else spatial.close, undo_motor,
                   None if air is None else air.close, None if run is None else run.close):
            if fn is not None:
                try: fn()
                except BaseException as error: cleanup_errors.append(type(error).__name__+': '+str(error))
        if cleanup_errors: status.update(status='FAILED',gate_pass=False,cleanup_errors=cleanup_errors)
        status.update(CPU_s=time.process_time()-cpu, wall_s=time.monotonic()-wall)
        save(out/'assessment.json', status)
    require(status['status']=='COMPLETE', 'cleanup failed; inspect assessment')
    (out/'REPORT.md').write_text('# Intervención de frontera PN\n\n'+arm+': '+status['status']+
        '. Cualificación: '+str(status['gate_pass'])+'. No admite etapas.\n')
    return {'metrics': {'new_CNS_ms':status['new_CNS_ms'], 'gate_pass':int(status['gate_pass'])}, 'assessment':status}


def compare(inputs, parameters, output):
    require(not parameters, 'comparison parameters belong in contract')
    c = json.loads(Path(inputs['contract']).read_text())
    results = {name:json.loads(Path(inputs[name]).read_text()) for name in c['arms']}
    valid = all(r.get('gate_pass', False) for r in results.values())
    result = {'schema':'matrix_pn_frontier_comparison_v1', 'instrument_valid':valid,
              'new_CNS_ms':sum(r['new_CNS_ms'] for r in results.values()), 'arms':results,
              'stage_admission':None, 'mediation_assessed':False, 'material_partial_effect':False,
              'scope':c['scope']}
    if valid:
        w = slice(c['window_ms'][0]-1, c['window_ms'][1]); means = {}
        for name in c['arms']:
            if name.startswith('live_'): continue
            with np.load(Path(inputs[name]).parent/'traces.npz', allow_pickle=False) as z:
                means[name] = {'yaw_deg_s':float(np.rad2deg(z['command_yaw_rate_rad_s'][w]).mean()),
                               'forward_mm_s':float(z['command_forward_mm_s'][w].mean())}
        shifts={mode:{side:means[mode+'_'+side]['yaw_deg_s']-means['self_'+side]['yaw_deg_s']
                      for side in ('L','R')} for mode in ('common','dose')}
        mediated = {m:(x['R']-x['L'])/2 for m,x in shifts.items()}
        material = all(x['L'] <= -c['minimum_each_shift_deg_s'] and x['R'] >= c['minimum_each_shift_deg_s']
                       and mediated[m] >= c['minimum_mediated_half_contrast_deg_s'] for m,x in shifts.items())
        result.update(means=means, shifts=shifts, mediated_half_contrast_deg_s=mediated,
                      material_partial_effect=material, mediation_assessed=True)
    save(Path(output)/'assessment.json', result)
    text = '# Contraste causal de frontera PN\n\n'+c['scope']+'\n\n'
    text += f"CNS nuevo: {result['new_CNS_ms']} ms. Instrumento válido: {valid}.\n\n"
    text += ('Efecto material parcial: '+str(result['material_partial_effect']) if valid else
             'La cualificación temporal falló. Los brazos causales posteriores no se ejecutan; causalidad no evaluada.')
    (Path(output)/'REPORT.md').write_text(text+'\n')
    return {'metrics':{'instrument_valid':int(valid), 'new_CNS_ms':result['new_CNS_ms'],
                       'material_partial_effect':int(result['material_partial_effect'])}, 'assessment':result}
