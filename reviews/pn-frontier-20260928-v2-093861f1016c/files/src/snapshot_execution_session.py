"""Versioned execution-only CNS205 continuation, retaining archival format."""
from pathlib import Path
import copy
from cxhp8_position_session import CxHP8PositionSession, SOURCES as PARENT_SOURCES
from snapshot_execution_brain import GpuSnapshotExecutionBrain, POLICY
from rf_tarsal_storage import load_rf_tarsal_session
from kc_session_storage import save_session
from kc_audited_session import ROOT, _fingerprints, sha256, require_covered_dependencies

SOURCES = tuple(PARENT_SOURCES) + ('execution_parent_snapshot.py', 'snapshot_execution_brain.py', 'snapshot_execution_session.py')
ENTRYPOINT = 'snapshot_execution_session.py'


class SnapshotExecutionSession(CxHP8PositionSession):
    SCHEMA = 'matrix_snapshot_execution_session_v1'
    BRAIN = GpuSnapshotExecutionBrain

    @classmethod
    def from_checkpoint(cls, path):
        parent = CxHP8PositionSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema', 'config', 'source_identity', 'intervention', 'hybrid'}
            before = _fingerprints(parent, excluded)
            obj.hybrid = cls.BRAIN.adopt(parent.hybrid)
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA, electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint=ENTRYPOINT, static_local_imports=require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES)))
            after = _fingerprints(obj, excluded)
            checks = {k: v == after[k] for k, v in before.items()}
            if not all(checks.values()):
                raise ValueError('Execution adoption changed inherited session state')
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention), operation=POLICY,
                parent_checkpoint=str(Path(path).resolve()), parent_manifest_sha256=sha256(Path(path)/'manifest.json'),
                time_ns=obj.time_ns, preserved_state_checks=checks, changes_equations=False,
                changes_timestep=False, new_canonical_neurons=0, new_biological_function=False)
            obj.source_identity = {n: sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate()
            obj._validate_pending()
            obj._validate_cxhp8_pending()
            return obj
        except BaseException:
            obj.close()
            raise

    def _manifest_fields(self):
        out = super()._manifest_fields()
        out.update(schema=self.SCHEMA, runtime_source_files=len(SOURCES), execution_snapshot_policy=POLICY)
        return out

    def save(self, path):
        self.hybrid.validate_execution()
        self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES) != self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Execution snapshot source contract changed')
        return save_session(self, path, SOURCES)

    @classmethod
    def load(cls, path):
        require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES)
        obj = load_rf_tarsal_session(path, cls, cls.SCHEMA, SOURCES, cls.BRAIN)
        try:
            obj._validate_cxhp8_pending()
            return obj
        except BaseException:
            obj.close()
            raise
