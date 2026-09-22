"""Execution-only snapshot specialization of the admitted CxHP8 CNS205.

Uses the same parent advance, PN solve, coupling schedule and validators.
The cold error path retains the strict historical loader and route rebinding.
No membrane, receptor, current, precision or anatomy changes.
"""
import copy
import numpy as np
from cxhp8_position_brain import GpuCxHP8PositionBrain
from cxhp8_position_field import AMPLITUDE
from projection_parallel_brain import GpuProjectionParallelBrain
from execution_parent_snapshot import ParentSnapshotFrames
from kc_apl_dynamics import cascade
from kcgamma_regional_brain import _record_hash

POLICY = 'Private_parent_snapshot_once_per_advance_v1'


class GpuSnapshotExecutionBrain(GpuCxHP8PositionBrain):
    SCHEMA = 'matrix_snapshot_execution_brain_v1'

    @classmethod
    def adopt(cls, parent):
        if type(parent) is not GpuCxHP8PositionBrain:
            raise ValueError('Exact admitted CxHP8 parent required')
        before = parent.state_dict()
        obj = cls.__new__(cls)
        obj.__dict__.update(parent.__dict__)
        obj.execution_snapshot_manifest = dict(
            policy=POLICY, origin_ns=parent.time_ns,
            reference_parent_schema=parent.SCHEMA,
            parent_state_sha256=_record_hash(before),
            changes_equations=False, changes_precision=False,
            changes_timestep=False, changes_anatomy=False,
            recovery='Unchanged strict parent loader and route rebinding',
            scope='Execution allocation only; no new biological function')
        obj.execution_snapshot_manifest['record_sha256'] = _record_hash(obj.execution_snapshot_manifest)
        obj._bind_online_routes()
        obj._rebind_continuing_inputs()
        after = obj.state_dict()
        after.pop('execution_snapshot_manifest')
        after['schema'] = before['schema']
        if _record_hash(after) != _record_hash(before):
            raise ValueError('Execution adoption changed inherited state')
        return obj

    def validate_execution(self):
        m = self.execution_snapshot_manifest
        if m['record_sha256'] != _record_hash(m) or m['policy'] != POLICY:
            raise ValueError('Execution snapshot policy changed')
        if any(m[k] is not False for k in ('changes_equations', 'changes_precision', 'changes_timestep', 'changes_anatomy')):
            raise ValueError('Unexpected model change in execution policy')

    def advance(self, dt_ns, drive, light):
        self.validate_execution()
        self.validate_cxhp8()
        if self.cxhp8_pending['sample_time_ns'] != self.time_ns:
            raise ValueError('Stale CxHP8 input')
        drive = np.asarray(drive, dtype=float).copy()
        if drive.shape != (self.brain.n_neurons,):
            raise ValueError('Invalid drive shape')
        drive[self._cxhp8_rows] += AMPLITUDE * self.cxhp8_pending['applied_index']
        self._validated_inputs(dt_ns, drive, light)
        self.validate_online()
        frames = ParentSnapshotFrames(self)
        left = dt_ns
        p = self.kc_electrical_scales_manifest['source']['PN_KCgamma']
        m = self.pn_online_manifest
        while left:
            ns = min(left, m['coupling_ns'])
            before, published = frames.capture(self)
            try:
                source_before = self._online_source.state_dict()
                first = self._online_ports.observe()
                old = self.kc_electrical_scales_state['filters'][:, self._online_slot].copy()
                try:
                    GpuProjectionParallelBrain.advance(self, ns, drive, light)
                    x, y = cascade(old[0], old[1], 0., p['cascade_fast_tau_s'], p['cascade_slow_tau_s'], ns * 1e-9)
                    self.kc_electrical_scales_state['filters'][:, self._online_slot] = [x, y]
                    second = self._online_ports.observe()
                    report = self._online_source.advance(
                        ns, first, second, connected=m['orn_connected'],
                        rtol=1e-9, atol=5e-6, maxiter=220, max_newton=8,
                        gate_atol=1e-12, stage_predictor='linear')
                    if not report['accepted']:
                        raise RuntimeError('FinePN step rejected: ' + str(report.get('reason')))
                    self.validate_online()
                    self.validate_scales()
                except BaseException:
                    self._restore_joint(before, source_before, published)
                    raise
            finally:
                frames.release(before)
            left -= ns

    def _restore_joint(self, parent_state, source_state, published):
        manifest = copy.deepcopy(self.execution_snapshot_manifest)
        super()._restore_joint(parent_state, source_state, published)
        self.execution_snapshot_manifest = manifest
        self.validate_execution()

    def state_dict(self):
        self.validate_execution()
        out = super().state_dict()
        out.update(schema=self.SCHEMA, execution_snapshot_manifest=copy.deepcopy(self.execution_snapshot_manifest))
        return out

    @classmethod
    def from_state(cls, brain, saved):
        if saved.get('schema') != cls.SCHEMA:
            raise ValueError('Wrong execution snapshot schema')
        parent = {k: v for k, v in saved.items() if k != 'execution_snapshot_manifest'}
        parent['schema'] = GpuCxHP8PositionBrain.SCHEMA
        base = GpuCxHP8PositionBrain.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.execution_snapshot_manifest = copy.deepcopy(saved['execution_snapshot_manifest'])
        obj._bind_online_routes()
        obj._rebind_continuing_inputs()
        obj.validate_execution()
        return obj

    @staticmethod
    def backend_identity():
        out = GpuCxHP8PositionBrain.backend_identity()
        out['execution_snapshot'] = POLICY
        return out
