"""Versioned allocation-only specialization of the RH219 neural runtime."""
import copy
from snapshot_execution_brain import GpuSnapshotExecutionBrain
from gpu_coefficient_buffers import CoefficientBuffers
from gpu_coefficient_layout import assemble
from kcgamma_regional_brain import _record_hash

POLICY = 'FP64_segment_assembly_private_frames_owned_outputs_v1'


class GpuCoefficientBufferBrain(GpuSnapshotExecutionBrain):
    SCHEMA = 'matrix_coefficient_buffer_brain_v1'

    @classmethod
    def adopt(cls, parent):
        if type(parent) is not GpuSnapshotExecutionBrain:
            raise ValueError('Exact snapshot execution parent required')
        before = parent.state_dict()
        obj = cls.__new__(cls)
        obj.__dict__.update(parent.__dict__)
        obj.coefficient_buffer_manifest = dict(policy=POLICY, origin_ns=parent.time_ns,
            reference_parent_schema=parent.SCHEMA, parent_state_sha256=_record_hash(before),
            changes_equations=False, changes_precision=False, changes_timestep=False,
            changes_anatomy=False, public_outputs='Independent owned target/rate arrays',
            scratch='Private reusable frames; not scientific state; recreated after reload or rollback')
        obj.coefficient_buffer_manifest['record_sha256'] = _record_hash(obj.coefficient_buffer_manifest)
        obj._bind_online_routes()
        obj._rebind_continuing_inputs()
        after = obj.state_dict()
        after.pop('coefficient_buffer_manifest')
        after['schema'] = parent.SCHEMA
        if _record_hash(before) != _record_hash(after):
            raise ValueError('Coefficient allocation adoption changed inherited state')
        return obj

    def validate_coefficients(self):
        m = self.coefficient_buffer_manifest
        if m['policy'] != POLICY or m['record_sha256'] != _record_hash(m):
            raise ValueError('Coefficient allocation policy changed')

    def coefficients_gpu(self, state, drive, light):
        if not hasattr(self, '_coefficient_buffers'):
            self._coefficient_buffers = CoefficientBuffers(len(self.state))
        with self._coefficient_buffers.acquire(state) as frame:
            target, rate = assemble(self, state, drive, light, frame=frame)
            if frame.used != {'target': len(state), 'rate': len(state)}:
                raise RuntimeError('Incomplete coefficient assembly')
            # A caller may retain one stage while evaluating another. Never
            # expose reusable storage through the inherited public interface.
            return target.copy(), rate.copy()

    def advance(self, dt_ns, drive, light):
        self.validate_coefficients()
        return super().advance(dt_ns, drive, light)

    def _restore_joint(self, parent_state, source_state, published):
        manifest = copy.deepcopy(self.coefficient_buffer_manifest)
        super()._restore_joint(parent_state, source_state, published)
        self.coefficient_buffer_manifest = manifest
        self.validate_coefficients()

    def state_dict(self):
        self.validate_coefficients()
        out = super().state_dict()
        out.update(schema=self.SCHEMA,
                   coefficient_buffer_manifest=copy.deepcopy(self.coefficient_buffer_manifest))
        return out

    @classmethod
    def from_state(cls, brain, saved):
        if saved.get('schema') != cls.SCHEMA:
            raise ValueError('Wrong coefficient buffer schema')
        parent = {k: v for k, v in saved.items() if k != 'coefficient_buffer_manifest'}
        parent['schema'] = GpuSnapshotExecutionBrain.SCHEMA
        base = GpuSnapshotExecutionBrain.from_state(brain, parent)
        obj = cls.__new__(cls)
        obj.__dict__.update(base.__dict__)
        obj.coefficient_buffer_manifest = copy.deepcopy(saved['coefficient_buffer_manifest'])
        obj._bind_online_routes()
        obj._rebind_continuing_inputs()
        obj.validate_coefficients()
        return obj

    @staticmethod
    def backend_identity():
        out = GpuSnapshotExecutionBrain.backend_identity()
        out['coefficient_buffers'] = POLICY
        return out
