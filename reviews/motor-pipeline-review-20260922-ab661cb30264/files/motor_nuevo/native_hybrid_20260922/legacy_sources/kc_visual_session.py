"""Saved full-CNS continuation with conservative visual calyx input placement."""
from pathlib import Path
import copy
from kc_axonal_session import KcAxonalSession,SOURCES as PARENT_SOURCES,ROOT,_fingerprints,sha256
from kc_visual_brain import GpuKcVisualBrain
from kc_session_storage import save_session,load_session

SCHEMA='matrix_kc_visual_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('kc_visual_ports.py','kc_visual_brain.py','kc_visual_session.py')


class KcVisualSession(KcAxonalSession):
    @classmethod
    def from_checkpoint(cls,path,*,enabled=True):
        path=Path(path).resolve();parent=KcAxonalSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.hybrid=GpuKcVisualBrain.adopt(parent.hybrid,enabled=enabled)
            obj.config=copy.deepcopy(obj.config);obj.config.update(candidate='KC_VISUAL_DENDRITIC_PORT_v1',kc_visual_enabled=enabled,
                electrical_backend_identity=obj.hybrid.backend_identity(),biological_validation=False)
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Visual integration changed inherited organism state')
            obj.intervention=dict(previous_intervention=copy.deepcopy(obj.intervention),operation='Route canonical gamma-d visual inputs to dendritic port',
                time_ns=obj.time_ns,parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                enabled=enabled,preserved_state_checks=checks,new_neurons=0,new_anatomical_pairs=0,biological_validation=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_afferent_pending();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuKcVisualBrain):raise ValueError('Missing visual-port component')
        self.hybrid.validate_visual_ports()
        if self.config['kc_visual_enabled']!=self.hybrid.kc_visual_manifest['enabled']:raise ValueError('Visual configuration mismatch')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();m=self.hybrid.kc_visual_manifest
        out.update(schema=SCHEMA,kc_visual_enabled=m['enabled'],kc_visual_dendritic_targets=len(m['rows']),
            kc_visual_dendritic_pairs=m['pairs'],kc_visual_conductance_totals_preserved=True);return out

    def save(self,path):return save_session(self,path,SOURCES)
    @classmethod
    def load(cls,path):return load_session(path,cls,SCHEMA,SOURCES,GpuKcVisualBrain)
