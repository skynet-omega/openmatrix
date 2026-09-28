"""Quarantine one disputed motor assignment without erasing its CNS neuron.

The existing effective muscle state and committed interval survive adoption.
Future output from820896 is omitted from its old levator slot, with divisor3.
No alternative muscle, force, synapse, or neuronal parameter is invented.
"""
from pathlib import Path
import copy
import numpy as np
from serial_ctr_session import SerialCTrSession, SOURCES as PARENT_SOURCES
from serial_ctr_storage import load_serial_ctr_session
from motor_release_port import selective_mean
from kc_session_storage import save_session
from kc_audited_session import ROOT, _fingerprints, sha256, require_covered_dependencies

SOURCES = tuple(PARENT_SOURCES) + ('rf_tarsal_identity_session.py',)
ENTRYPOINT = 'rf_tarsal_identity_session.py'
POLICY = 'quarantine_disputed_820896_levator_slot_keep_divisor3_v1'


class RFTarsalIdentitySession(SerialCTrSession):
    SCHEMA = 'matrix_rf_tarsal_identity_session_v1'

    @classmethod
    def from_checkpoint(cls, path, *, withdraw=True):
        if type(withdraw) is not bool:
            raise ValueError('Explicit withdrawal boolean required')
        parent = SerialCTrSession.load(path)
        obj = cls(); obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema', 'config', 'source_identity', 'intervention'}
            before = _fingerprints(parent, excluded)
            obj.config = copy.deepcopy(parent.config)
            obj.config['rf_tarsal_identity_policy'] = dict(policy=POLICY, withdraw=withdraw,
                removed_id=820896, role='levator', position=2, denominator=3,
                start_ns=obj.time_ns, inherited_pending_interval_preserved=True,
                muscle_reassigned=False)
            obj.config.update(candidate=cls.SCHEMA,
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,
                    static_local_imports=require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES)))
            after = _fingerprints(obj, excluded)
            checks = {k: v == after[k] for k, v in before.items()}
            if not all(checks.values()):
                raise ValueError('Identity quarantine changed inherited history')
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation=POLICY, parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/'manifest.json'), time_ns=obj.time_ns,
                preserved_state_checks=checks, neural_parameters_changed=False,
                canonical_neurons_removed=0, muscle_reassigned=False,
                biological_action_identified=False)
            obj.source_identity = {n: sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate(); obj._validate_tarsal_pending()
            return obj
        except BaseException:
            obj.close(); raise

    def tarsal_command(self):
        p = self.config['rf_tarsal_identity_policy']
        if not p['withdraw'] or not self.config['rf_tarsal_input_connected']:
            return super().tarsal_command()
        q = self.hybrid.release()
        return np.array([selective_mean(q[ix], 2 if role == 1 else None)
                         for role, ix in enumerate(self.tarsal_indices)])

    def _validate_tarsal_pending(self):
        p = self.config['rf_tarsal_identity_policy']
        expected = super().tarsal_command() if self.time_ns == p['start_ns'] else self.tarsal_command()
        if not np.array_equal(self.body.pending_tarsal, expected):
            raise ValueError('Tarsal pending differs from preserved interval/identity policy')

    def _validate(self):
        super()._validate()
        p = self.config.get('rf_tarsal_identity_policy', {})
        fixed = dict(policy=POLICY, removed_id=820896, role='levator', position=2,
                     denominator=3, inherited_pending_interval_preserved=True, muscle_reassigned=False)
        if (set(p) != set(fixed) | {'withdraw', 'start_ns'}
                or any(p.get(k) != v or type(p.get(k)) is not type(v) for k, v in fixed.items())
                or type(p['withdraw']) is not bool or type(p['start_ns']) is not int
                or not 0 <= p['start_ns'] <= self.time_ns
                or (self.time_ns-p['start_ns']) % self.CONTROL_NS
                or self.body.tarsal_ids != [[825721], [817210, 820110, 820896]]):
            raise ValueError('Invalid disputed motor assignment policy')

    def _manifest_fields(self):
        out = super()._manifest_fields()
        p = self.config['rf_tarsal_identity_policy']
        out.update(schema=self.SCHEMA, runtime_source_files=len(SOURCES),
            rf_tarsal_identity_policy=copy.deepcopy(p), rf_tarsal_registered_MN=4,
            rf_tarsal_future_MN_with_output=3 if p['withdraw'] else 4,
            motor_ids_with_future_output=109 if p['withdraw'] else 110)
        return out

    def save(self, path):
        self._validate(); self._validate_tarsal_pending(); self._validate_serial_pending(); self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES) != self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Identity policy source contract changed')
        return save_session(self, path, SOURCES)

    @classmethod
    def load(cls, path):
        require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES)
        return load_serial_ctr_session(path, cls, cls.SCHEMA, SOURCES, cls.BRAIN)
