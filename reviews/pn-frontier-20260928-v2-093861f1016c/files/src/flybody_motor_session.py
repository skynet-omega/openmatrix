"""Versioned selective MN input boundary in the continuing full CNS/FlyBody.

The pending command at adoption is preserved for its existing interval. Only
newly computed commands are masked. Aggregate activation and all neural/body
history are retained; no independent historical MN muscle state is invented.
"""
from pathlib import Path
import copy
import numpy as np
from flybody_cns_session import FlyBodyCNSSession, SOURCES as PARENT_SOURCES
from flybody_cns_storage import load_flybody_session
from kc_session_storage import save_session
from kc_audited_session import ROOT, _fingerprints, sha256, require_covered_dependencies
from native_motor_body import LEGS
from motor_release_port import POLICY, selective_mean

SOURCES = tuple(PARENT_SOURCES)+('motor_release_port.py','flybody_motor_session.py')
ENTRYPOINT = 'flybody_motor_session.py'


class FlyBodyMotorSession(FlyBodyCNSSession):
    SCHEMA = 'matrix_flybody_motor_session_v1'

    @classmethod
    def from_checkpoint(cls, path, *, removed_id=None):
        if removed_id is not None and (type(removed_id) is not int or removed_id <= 0):
            raise ValueError('Explicit canonical integer MN ID or None required')
        path = Path(path).resolve()
        parent = FlyBodyCNSSession.load(path)
        obj = cls(); obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema','config','source_identity','intervention'}
            before = _fingerprints(parent, excluded)
            found = [(leg, role, ids.index(removed_id))
                     for leg, roles in obj.motor_ids.items() for role, ids in roles.items()
                     if removed_id in ids]
            if removed_id is not None and len(found) != 1:
                raise ValueError('MN must occupy exactly one existing motor input slot')
            leg, role, position = found[0] if found else (None, None, None)
            obj.config = copy.deepcopy(obj.config)
            obj.config['motor_input_intervention'] = dict(policy=POLICY, removed_id=removed_id,
                leg=leg, role=role, position=position, start_ns=obj.time_ns,
                denominator=len(obj.motor_ids[leg][role]) if found else None,
                inherited_pending_interval_preserved=True)
            closure = require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
            obj.config.update(candidate=cls.SCHEMA,
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=closure))
            after = _fingerprints(obj,excluded)
            checks = {k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):
                raise ValueError('Motor boundary changed inherited state')
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Preserve pending interval; optionally remove one future normalized MN input',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                time_ns=obj.time_ns,preserved_state_checks=checks,
                new_canonical_neurons=0,neural_equations_changed=False,
                motor_policy=copy.deepcopy(obj.config['motor_input_intervention']),
                independent_motor_unit_identity_or_force=False)
            obj.source_identity = {n:sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate(); obj._validate_pending(); obj._validate_afferent_pending()
            return obj
        except BaseException:
            obj.close(); raise

    def motor_excitation(self):
        policy = self.config['motor_input_intervention']
        if policy['removed_id'] is None:
            return super().motor_excitation()
        release = self.hybrid.release()
        return np.array([[selective_mean(release[self.motor_indices[leg][role]],
            policy['position'] if (leg,role)==(policy['leg'],policy['role']) else None)
            for role in ('flexor','extensor')] for leg in LEGS])

    def _validate(self):
        super()._validate()
        p = self.config.get('motor_input_intervention',{})
        if (set(p) != {'policy','removed_id','leg','role','position','start_ns','denominator','inherited_pending_interval_preserved'}
                or p['policy'] != POLICY or type(p['start_ns']) is not int
                or p['start_ns'] > self.time_ns or p['start_ns'] < 0
                or (self.time_ns-p['start_ns']) % self.CONTROL_NS
                or p['inherited_pending_interval_preserved'] is not True):
            raise ValueError('Invalid selective motor policy')
        if p['removed_id'] is None:
            if any(p[k] is not None for k in ('leg','role','position','denominator')):
                raise ValueError('Reference mode has a hidden motor selection')
        else:
            if (type(p['removed_id']) is not int or p['leg'] not in self.motor_ids
                    or p['role'] not in ('flexor','extensor') or type(p['position']) is not int):
                raise ValueError('Unknown motor selection')
            ids = self.motor_ids[p['leg']][p['role']]
            if (not 0 <= p['position'] < len(ids) or ids[p['position']] != p['removed_id']
                    or type(p['denominator']) is not int or p['denominator'] != len(ids)):
                raise ValueError('Motor slot or denominator changed')

    def _validate_pending(self):
        super()._validate_pending()
        if self.time_ns == self.config['motor_input_intervention']['start_ns']:
            expected = super().motor_excitation()
        else:
            expected = self.motor_excitation()
        if not np.array_equal(self.pending_excitation,expected):
            raise ValueError('Pending motor input disagrees with its preserved interval/policy')

    def state_dict(self):
        out = super().state_dict(); out['schema'] = self.SCHEMA; return out

    def _manifest_fields(self):
        out = super()._manifest_fields()
        out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),
            motor_input_intervention=copy.deepcopy(self.config['motor_input_intervention']))
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES) != self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Motor session source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_flybody_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)
