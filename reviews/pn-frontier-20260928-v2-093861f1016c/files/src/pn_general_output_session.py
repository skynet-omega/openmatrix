"""Persist all PN output interfaces without modifying the refined parent."""
from pathlib import Path
import copy
from pn_electrical_refined_session import PnElectricalRefinedSession, SOURCES as PARENT_SOURCES
from pn_general_output_brain import GpuPnGeneralOutputBrain
from kc_audited_session import ROOT, _fingerprints, sha256, require_covered_dependencies
from kc_session_storage import save_session, load_session
from kcgamma_regional_brain import _record_hash

SCHEMA = 'matrix_pn_general_output_session_v1'
SOURCES = tuple(PARENT_SOURCES) + ('pn_general_output_port.py', 'pn_general_output_source.py',
                                 'pn_general_output_brain.py', 'pn_general_output_session.py')


class PnGeneralOutputSession(PnElectricalRefinedSession):
    @classmethod
    def from_checkpoint(cls, path, spec, *, enabled=True, coupling_ns=None):
        path = Path(path).resolve(); parent = PnElectricalRefinedSession.load(path)
        obj = cls(); obj.__dict__.update(parent.__dict__)
        try:
            excluded = {'schema', 'config', 'source_identity', 'intervention', 'hybrid'}
            before = _fingerprints(parent, excluded)
            oldsource = parent.hybrid._online_source.state_dict()
            obj.hybrid = GpuPnGeneralOutputBrain.adopt(parent.hybrid, spec, enabled=enabled, coupling_ns=coupling_ns)
            if _record_hash(oldsource) != _record_hash(obj.hybrid._online_source.state_dict()['base']):
                raise ValueError('General output adoption altered previous histories')
            closure = require_covered_dependencies(ROOT/'src', 'pn_general_output_session.py', SOURCES)
            obj.config = copy.deepcopy(parent.config)
            obj.config.update(candidate='PN_GENERAL_OUTPUT_v1',
                electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_general_output_session.py', static_local_imports=closure),
                PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after = _fingerprints(obj, excluded)
            checks = {k: v == after[k] for k, v in before.items()}
            if not all(checks.values()):
                raise ValueError('General output adoption changed body or pending input')
            obj.intervention = dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Replace629general future PN contributions with local release',
                parent_checkpoint=str(path), parent_manifest_sha256=sha256(path/'manifest.json'),
                time_ns=obj.time_ns, preserved_state_checks=checks, new_canonical_neurons=0,
                full_PN_replacement=False, biological_validation=False)
            obj.source_identity = {n: sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate(); return obj
        except BaseException:
            obj.close(); raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid, GpuPnGeneralOutputBrain):
            raise ValueError('Wrong general-output backend')

    def state_dict(self):
        out = super().state_dict(); out['schema'] = SCHEMA; return out

    def _manifest_fields(self):
        out = super()._manifest_fields()
        connected = self.hybrid.pn_online_manifest['general_outputs']['enabled']
        out.update(schema=SCHEMA, runtime_source_files=len(SOURCES),
                   PN_output_pairs_replaced=1095 if connected else 466,
                   PN_output_pairs_legacy=0 if connected else 629,
                   PN_general_outputs_enabled=connected)
        return out

    def save(self, path):
        if require_covered_dependencies(ROOT/'src', 'pn_general_output_session.py', SOURCES) != self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('General output source contract changed')
        return save_session(self, path, SOURCES)

    @classmethod
    def load(cls, path):
        require_covered_dependencies(ROOT/'src', 'pn_general_output_session.py', SOURCES)
        return load_session(path, cls, SCHEMA, SOURCES, GpuPnGeneralOutputBrain)
