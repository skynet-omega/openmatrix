"""Explicit restoration of campaign48 owners and an OFF-tail continuation.

No historical source is changed. Not a navigation trial or biological rescue.
One active runtime per process, following the inherited runtime contract.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
C48 = ROOT / 'campanas/etapa45_composicion_20260927_48'
BASE = ROOT / 'motor_nuevo/full_pipeline_review_20260925_12'
LONG = ROOT / 'motor_nuevo/dynamics_12s_20260926_13'
sys.path[:0] = [str(HERE), str(C48), str(BASE), str(LONG)]
import run_trial as reference
from observations import need, sha, save
from state_compare import compare_trees
from protocol import UniformOdor, NeuralPropulsion, CausalIntervals
from sensory48 import TerminalInput

MOTOR_KEYS = ('substeps', 'body_calls', 'trial_step', 'raw_forward', 'forward', 'raw_yaw')


def verify_checkpoint(folder):
    folder = Path(folder)
    m = json.loads((folder/'MANIFEST.json').read_text())
    need(m['schema'] == 'composition48_scientific_state_v1', 'Wrong checkpoint schema')
    need({p.name for p in folder.iterdir() if p.is_file()} == set(m['files']) | {'MANIFEST.json'},
         'Checkpoint file set differs')
    for name, expected in m['files'].items():
        need(sha(folder/name) == expected, 'Checkpoint digest differs: ' + name)
    return m


def interval_state(auditor):
    return {k: getattr(auditor, k) for k in ('origin_ns', 'pending', 'previous_dn', 'baseline')}


def motor_state(motor):
    return {k: getattr(motor, k) for k in MOTOR_KEYS}


def boundary_state(run):
    return dict(schema='zero_legacy_odor_prescribed_ORN_v1',
                legacy_origin_ns=run.obj.core.world.boundary.origin_ns,
                plan=run.obj.core.world.boundary.plan, arm=run.stimulus.arm)


def qualify(run, folder):
    from session_io import read_state
    from operator_state import OperatorState, LEGACY_BINDINGS
    obj = run.obj
    getters = dict(session=obj.core.state_dict, prosthesis=obj.state,
                   published=lambda: dict(rates=obj.core.brain.rates,
                       time_ns=obj.core.brain.time_ns, rng=obj.core.brain.rng.bit_generator.state),
                   stored_operator=lambda: dict(weights=obj.core.brain.W.data),
                   effective_operator=lambda: OperatorState(obj.core.hybrid, LEGACY_BINDINGS).state_dict(),
                   input_owner=run.stimulus.state,
                   interval_owner=lambda: interval_state(run.auditor),
                   motor_owner=lambda: motor_state(run.motor))
    trees = {key: compare_trees(read_state(Path(folder)/key), getter())
             for key, getter in getters.items()}
    boundary = json.loads((Path(folder)/'boundary.json').read_text())
    trees['boundary'] = compare_trees(boundary, boundary_state(run))
    return dict(exact=all(v['exact'] for v in trees.values()), trees=trees,
                scope='All serialized scientific owners before any continuation; no tolerance.')


class Continued:
    def __init__(self):
        self.obj = self.session = self.stimulus = None
        self.undo = self.undo_observer = None
        self.undo_clock = None

    def step(self):
        import cupy as cp
        k = self.stimulus.k + 1
        used = self.auditor.before(self.obj, k)
        self.stimulus.consume(k)
        self.observer.current_ms = k
        self.obj.step()
        cp.cuda.runtime.deviceSynchronize()
        row = self.d.captura(self.obj, self.ports, 'cola_OFF', k, used, self.yaw0, cp)
        self.stimulus.observed()
        row.update(forward_unclipped_mm_s=self.motor.raw_forward,
                   neural_yaw_unapplied_rad_s=self.motor.raw_yaw)
        self.auditor.after(self.obj, row, k, used)
        return row

    def save(self, folder):
        from checkpoint48 import save_checkpoint
        save_checkpoint(Path(folder), self.obj, self.motor, self.stimulus, self.auditor)

    def close(self):
        errors = []
        for fn in (None if self.session is None else self.session.close,
                   self.undo, self.undo_observer,
                   None if self.stimulus is None else self.stimulus.close,
                   None if self.obj is None else self.obj.close,
                   self.undo_clock):
            if fn is not None:
                try:
                    fn()
                except BaseException as exc:
                    errors.append(repr(exc))
        if errors:
            raise RuntimeError('Cleanup failed: ' + repr(errors))


def build(source, output, *, tail_until_ms=3200, observer_installer=None):
    """Restore a final48 state or a49 split; extend only the duration limit.

    observer_installer(brain, stimulus) is an optional compatible instrumentation
    factory returning (observer, undo). Equivalence of new observers is separate.
    """
    import cupy as cp
    from motor_runtime import load
    from restore48 import restore
    from session_io import read_state
    from pn_cns_ports import PnCnsPorts
    from runtime_session import RuntimeSession
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run = Continued()
    try:
        from clock_guard49 import install as install_clock
        run.clock_overlay, run.undo_clock = install_clock()
        save(output/'CLOCK_OVERLAY.json', run.clock_overlay)
        run.obj, run.d, *_ = load(output/'static_inputs')
        run.restoration = restore(run.obj, source)
        h = run.obj.core.hybrid
        need(not run.obj.core.plasticity.enabled, 'Plasticity changed')
        need(not h.pn_online_manifest['general_outputs']['enabled'], 'PN interface changed')
        boundary = json.loads((source/'boundary.json').read_text())
        plan = copy.deepcopy(boundary['plan'])
        saved_input = read_state(source/'input_owner')
        need(saved_input['consumed_ms'] >= plan['odor_off_ms'], 'Only post-odor tail is allowed')
        need(plan['odor_off_ms'] == 3000, 'Unexpected original odor offset')
        run.ports = PnCnsPorts(h, 10208)
        # Preserve the same event/PN/FP32 owners as48, with no numerical change.
        for name in ('event_waveform', 'event_ports', 'event_coupling', 'native_cell', 'device_cell'):
            need(name not in sys.modules, 'Preloaded event owner: ' + name)
        sys.path.insert(0, str(reference.EVENTS))
        import event_waveform, event_ports, event_coupling, native_cell, device_cell
        for mod in (event_waveform, event_ports, event_coupling, native_cell, device_cell):
            need(Path(mod.__file__).resolve().parent == reference.EVENTS, 'Wrong event implementation')
        sys.path.insert(0, str(BASE/'pn'))
        from install_pn import install as install_pn
        run.pn_overlay = install_pn()
        sys.path.insert(0, str(BASE/'engine'))
        from real_model import install
        if observer_installer is None:
            from observer import install as observer_installer
        run.stimulus = TerminalInput(h, plan, boundary['arm'])
        run.stimulus.k = int(saved_input['consumed_ms'])
        run.stimulus.current = saved_input['nominal_Hz'].copy()
        run.stimulus.device_rates.set(np.ascontiguousarray(run.stimulus.current))
        run.observer, run.undo_observer = observer_installer(h, run.stimulus)
        run.undo = install('persistent_fp32')
        run.session = RuntimeSession(h, 'causal_cuda')
        run.observer.bind_context()
        field = run.obj.core.world.boundary
        run.motor = NeuralPropulsion(run.obj)
        for key, value in read_state(source/'motor_owner').items():
            need(key in MOTOR_KEYS, 'Unexpected motor field')
            setattr(run.motor, key, copy.deepcopy(value))
        # Constructor expresses a new experiment; a resumed owner instead keeps
        # its historical origin, pending inputs, baseline and previous DN read.
        run.auditor = CausalIntervals.__new__(CausalIntervals)
        run.auditor.field, run.auditor.motor, run.auditor.plan = field, run.motor, plan
        for key, value in read_state(source/'interval_owner').items():
            need(key in ('origin_ns', 'pending', 'previous_dn', 'baseline'), 'Unexpected interval field')
            setattr(run.auditor, key, copy.deepcopy(value))
        # A trace reference must not reset at every cold split. It never enters
        # the motor; retain48's recorded common initial pose for this diagnostic.
        with np.load(C48/boundary['arm']/'initial_observation.npz', allow_pickle=False) as initial:
            run.yaw0 = run.d.yaw_grados(initial['qpos'])
        run.initial = qualify(run, source)
        save(output/'INITIAL.json', run.initial)
        need(run.initial['exact'], 'Serialized scientific owners differ before first step')
        need(run.stimulus.k <= tail_until_ms <= 3200, 'Tail outside qualified clock horizon')
        # Sole protocol change: permit inspecting the OFF tail beyond the old
        # duration guard. Keep odor_on/off, all nominal targets and owners intact.
        old_duration = plan['duration_ms']
        plan['duration_ms'] = tail_until_ms
        field.plan = plan
        run.appendix = dict(source=str(source), consumed_ms=run.stimulus.k,
                            original_duration_ms=old_duration, duration_ms=tail_until_ms,
                            odor_off_ms=plan['odor_off_ms'], policy='original_OFF_tail',
                            neural_state_clamped=False, navigation_experiment=False)
        save(output/'APPENDIX.json', run.appendix)
        cp.cuda.get_current_stream().synchronize()
        return run
    except BaseException as exc:
        try:
            run.close()
        except BaseException as cleanup:
            # Preserve the actual restoration failure when a partially built
            # historical owner also rejects its normal teardown entrypoint.
            exc.add_note('Secondary cleanup failure: ' + repr(cleanup))
        raise
