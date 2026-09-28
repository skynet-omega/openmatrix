"""Continue the complete spatial CNS with axonal-event-driven gamma output."""
from pathlib import Path
import copy
from kc_spatial_session import KcSpatialSession,SOURCES as PARENT_SOURCES,ROOT,_fingerprints,_digest,sha256
from kc_axonal_brain import GpuKcAxonalBrain
from kc_session_storage import save_session,load_session

SCHEMA='matrix_kc_axonal_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('kc_axonal_release.py','kc_axonal_brain.py','kc_session_storage.py','kc_axonal_session.py')


class KcAxonalSession(KcSpatialSession):
    @classmethod
    def from_checkpoint(cls,path,*,enabled=True):
        path=Path(path).resolve();parent=KcSpatialSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded)
            obj.hybrid=GpuKcAxonalBrain.adopt(parent.hybrid,enabled=enabled)
            obj.config=copy.deepcopy(obj.config)
            obj.config.update(candidate='KC_AXONAL_RELEASE_v1',kc_axonal_enabled=enabled,
                electrical_backend_identity=obj.hybrid.backend_identity(),biological_validation=False,learning_demonstrated=False)
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Axonal integration altered inherited organism fields')
            obj.intervention=dict(previous_intervention=copy.deepcopy(obj.intervention),
                operation='Connect sampled WT9 axonal port peaks to canonical gamma-lobe output fractions',
                time_ns=obj.time_ns,parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled,preserved_state_checks=checks,new_neurons=0,new_anatomical_pairs=0,
                calcium_and_vesicles_identified=False,biological_validation=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_afferent_pending();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuKcAxonalBrain):raise ValueError('Missing axonal component')
        self.hybrid.validate_axonal()
        if self.config['kc_axonal_enabled']!=self.hybrid.kc_axonal_manifest['enabled']:raise ValueError('Axonal configuration mismatch')

    def state_dict(self):
        result=super().state_dict();result['schema']=SCHEMA;return result

    def _manifest_fields(self):
        result=super()._manifest_fields();m=self.hybrid.kc_axonal_manifest
        result.update(schema=SCHEMA,kc_axonal_enabled=m['enabled'],axonal_bands=12,
            axonal_gamma_cells=len(m['rows']),axonal_output_calcium_identified=False,
            axonal_output_status=m['routing']);return result

    def save(self,path):return save_session(self,path,SOURCES)
    @classmethod
    def load(cls,path):return load_session(path,cls,SCHEMA,SOURCES,GpuKcAxonalBrain)
