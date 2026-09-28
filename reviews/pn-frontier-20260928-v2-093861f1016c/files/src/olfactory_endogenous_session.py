"""Persisted whole-CNS olfactory source replacement with live KC/APL/body."""
from pathlib import Path
import copy
from kc_audited_session import (KcAuditedSession,SOURCES as PARENT_SOURCES,
    ROOT,_fingerprints,sha256,require_covered_dependencies)
from kc_session_storage import save_session,load_session
from olfactory_endogenous_brain import GpuOlfactoryEndogenousBrain

SCHEMA='matrix_olfactory_endogenous_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('olfactory_endogenous_brain.py','olfactory_endogenous_session.py')


class OlfactoryEndogenousSession(KcAuditedSession):
    @classmethod
    def from_checkpoint(cls,path,*,enabled=True):
        path=Path(path).resolve();parent=KcAuditedSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded)
            obj.hybrid=GpuOlfactoryEndogenousBrain.adopt(parent.hybrid,enabled=enabled)
            closure=require_covered_dependencies(ROOT/'src','olfactory_endogenous_session.py',SOURCES)
            obj.config=copy.deepcopy(obj.config)
            obj.config.update(candidate='OLFACTORY_ENDOGENOUS_CNS_v1',
                olfactory_endogenous_enabled=enabled,
                electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='olfactory_endogenous_session.py',static_local_imports=closure),
                biological_validation=False)
            after=_fingerprints(obj,excluded)
            checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Source migration changed inherited organism state')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Retire held external ORN->PN source in favor of continuing canonical ORN activity',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                time_ns=obj.time_ns,enabled=enabled,preserved_state_checks=checks,
                inherited_synaptic_tails_preserved=True,native_PN_geometry_replaced=False,
                biological_validation=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_afferent_pending();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuOlfactoryEndogenousBrain):
            raise ValueError('Missing endogenous olfactory component')
        self.hybrid.validate_endogenous()
        if self.config['olfactory_endogenous_enabled']!=self.hybrid.olfactory_endogenous_manifest['enabled']:
            raise ValueError('Olfactory source configuration differs')

    def present_olfactory_pulse(self,**kwargs):
        if self.config['olfactory_endogenous_enabled']:
            raise ValueError('The held-rate port is inactive; stimuli must enter the canonical ORN input, not the historical device')
        return super().present_olfactory_pulse(**kwargs)

    def step(self):
        row=super().step()
        enabled=self.config['olfactory_endogenous_enabled']
        if not enabled:
            return row
        row['olfactory_endogenous_source_active']=enabled
        row['prosthetic_afferent_rate_active']=not enabled
        if enabled:
            # These parent diagnostic fields described a held value as used.
            row.pop('prosthetic_afferent_rate_used_hz',None)
            row.pop('prosthetic_afferent_rate_pending_hz',None)
        return row

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();m=self.hybrid.olfactory_endogenous_manifest
        out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),neural_equations_changed=m['enabled'],
            olfactory_endogenous_enabled=m['enabled'],olfactory_source_cells=len(m['source_ids']),
            olfactory_source_pairs=m['selected_pairs'],legacy_held_afferent_rate_active=not m['enabled'],
            native_olfactory_transduction_claimed=False,native_PN_geometry_replaced=False)
        return out

    def save(self,path):
        closure=require_covered_dependencies(ROOT/'src','olfactory_endogenous_session.py',SOURCES)
        if closure!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Source import contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','olfactory_endogenous_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuOlfactoryEndogenousBrain)
