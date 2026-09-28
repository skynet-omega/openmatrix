"""RH motor/canonical state with versioned neural coefficient allocation."""
from pathlib import Path
import copy
from rh_tarsal_session import RHTarsalSession, SOURCES as PARENT_SOURCES
from rh_tarsal_storage import load_rh_tarsal_session
from coefficient_buffer_brain import GpuCoefficientBufferBrain, POLICY
from kc_session_storage import save_session
from kc_audited_session import ROOT, _fingerprints, sha256, require_covered_dependencies

SOURCES = tuple(PARENT_SOURCES) + ('gpu_coefficient_layout.py', 'gpu_coefficient_buffers.py',
                                 'coefficient_buffer_brain.py', 'coefficient_buffer_session.py')
ENTRYPOINT = 'coefficient_buffer_session.py'


class CoefficientBufferSession(RHTarsalSession):
    SCHEMA = 'matrix_coefficient_buffer_session_v1'
    BRAIN = GpuCoefficientBufferBrain

    @classmethod
    def from_checkpoint(cls, path):
        parent = RHTarsalSession.load(path)
        obj = cls()
        obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema', 'config', 'source_identity', 'intervention', 'hybrid'}
            before = _fingerprints(parent, excluded)
            obj.hybrid = cls.BRAIN.adopt(parent.hybrid)
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA, electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,
                    static_local_imports=require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES)))
            after = _fingerprints(obj, excluded)
            checks = {k: v == after[k] for k, v in before.items()}
            if not all(checks.values()):
                raise ValueError('Coefficient execution adoption changed body or pending inputs')
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation=POLICY, parent_checkpoint=str(Path(path).resolve()),
                parent_manifest_sha256=sha256(Path(path)/'manifest.json'), time_ns=obj.time_ns,
                preserved_state_checks=checks, changes_equations=False,
                changes_timestep=False, changes_precision=False, new_biological_function=False)
            obj.source_identity = {n: sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate()
            return obj
        except BaseException:
            obj.close()
            raise

    def _manifest_fields(self):
        out = super()._manifest_fields()
        out.update(schema=self.SCHEMA, runtime_source_files=len(SOURCES), coefficient_buffer_policy=POLICY)
        return out

    def save(self, path):
        self.hybrid.validate_coefficients()
        self._validate_rh_pending()
        self._validate_tarsal_pending()
        self._validate_serial_pending()
        self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES) != self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Coefficient source contract changed')
        return save_session(self, path, SOURCES)

    @classmethod
    def load(cls, path):
        require_covered_dependencies(ROOT/'src', ENTRYPOINT, SOURCES)
        return load_rh_tarsal_session(path, cls, cls.SCHEMA, SOURCES, cls.BRAIN)
