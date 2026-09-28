"""Add canonical LF coxal output to the continuing189 neural/body history."""
from pathlib import Path
import copy
import numpy as np
from flybody_motor_session import FlyBodyMotorSession,SOURCES as PARENT_SOURCES
from flybody_cns_body import CNSFlyBodyMuscles
from flybody_cns_sensors import FlyBodyEye,FlyBodyProprioception
from pn_graph_solver_session import PnGraphSolverSession
from coxal_body import CoxalBody,PRIOR_SHA256
from coxal_storage import load_coxal_session
from kc_session_storage import save_session
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies

SOURCES=tuple(PARENT_SOURCES)+('coxal_body.py','coxal_storage.py','coxal_session.py')
ENTRYPOINT='coxal_session.py'


class CoxalSession(FlyBodyMotorSession):
    SCHEMA='matrix_flybody_coxal_session_v1'

    @classmethod
    def from_checkpoint(cls,path,*,connected=True):
        if type(connected) is not bool:raise ValueError('Explicit coxal connection required')
        parent=FlyBodyMotorSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            # This first extension uses the unablated189 preparation only.
            if parent.config['motor_input_intervention']['removed_id'] is not None:
                raise ValueError('Coxal adoption requires the reference tibial preparation')
            excluded={'schema','config','source_identity','intervention','body'}
            before=_fingerprints(parent,excluded)
            old_body=parent.body
            obj.body=CoxalBody.from_parent(old_body,np.zeros(7));obj._index_coxa()
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(coxal_input_connected=connected,coxal_prior_sha256=PRIOR_SHA256,
                coxal_origin_ns=obj.time_ns,candidate=cls.SCHEMA)
            obj.body.pending_coxal=obj.coxal_command()
            obj.muscles.body=obj.body;obj.proprioception.body=obj.body;obj.eyes.body=obj.body
            closure=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
            obj.config['runtime_source_contract']=dict(entrypoint=ENTRYPOINT,static_local_imports=closure)
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Coxal adoption changed inherited history: '+str(checks))
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Add seven provisional LF active muscle paths to eleven canonical MN; no prior proximal active contribution existed',
                parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),
                time_ns=obj.time_ns,preserved_state_checks=checks,physical_integration_state_preserved=True,
                new_activation_zero=True,body_pose_reset=False,neural_equations_changed=False,
                existing_joint_passivity_retained=True,additional_muscle_passive_bias=False,
                individual_MN_force_calibrated=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_coxal_pending();obj._validate_afferent_pending()
            old_body.close();return obj
        except BaseException:obj.close();raise

    def _index_coxa(self):
        self.coxal_indices=[np.searchsorted(self.brain.node_ids,ids) for ids in self.body.coxal_ids]
        for idx,ids in zip(self.coxal_indices,self.body.coxal_ids):
            if np.any(idx>=len(self.brain.node_ids)) or not np.array_equal(self.brain.node_ids[idx],ids):
                raise ValueError('Missing canonical coxal MN')

    def coxal_command(self):
        if not self.config['coxal_input_connected']:return np.zeros(7)
        q=self.hybrid.release();return np.array([q[idx].mean() for idx in self.coxal_indices])

    def _validate(self):
        # Preserve all ancestral neural/physical checks; update only187's exact
        # body factory and the explicitly reference-only tibial selection.
        PnGraphSolverSession._validate(self)
        if (type(self.body) is not CoxalBody or type(self.muscles) is not CNSFlyBodyMuscles
                or type(self.eyes) is not FlyBodyEye or type(self.proprioception) is not FlyBodyProprioception
                or self.output_connected or self.pending_cyborg_command!=0.
                or self.config['flybody_clock_origin_ns']!=self.body.origin_ns
                or type(self.config['coxal_input_connected']) is not bool
                or self.config['coxal_prior_sha256']!=PRIOR_SHA256
                or self.config['coxal_origin_ns']!=self.body.coxal_origin_ns):
            raise ValueError('Invalid coxal family/body policy')
        p=self.config['motor_input_intervention']
        if (p['removed_id'] is not None or any(p[k] is not None for k in ['leg','role','position','denominator'])
                or p['policy']!='zero_one_future_release_keep_pool_denominator_v1'
                or p['inherited_pending_interval_preserved'] is not True):
            raise ValueError('This coxal family requires reference tibial inputs')
        self.body.validate_coxa()
        if self.body.coxal_time_ns!=self.time_ns:raise ValueError('Coxal/session clocks differ')

    def _validate_coxal_pending(self):
        if not np.array_equal(self.body.pending_coxal,self.coxal_command()):
            raise ValueError('Coxal pending input differs from current canonical release')

    def step(self):
        self._validate_coxal_pending()
        used=self.body.pending_coxal.copy()
        row=super().step()
        # Inherited1ms loop uses held old command; new command starts next loop.
        self.body.pending_coxal=self.coxal_command();self._validate_coxal_pending()
        row.update(coxal_command_used=used.tolist(),coxal_command_pending=self.body.pending_coxal.tolist(),
            coxal_activation=self.body.activation.tolist(),coxal_force_N=self.body.last_tension_N.tolist(),
            coxal_prior_sha256=PRIOR_SHA256,coxal_neural_input_connected=self.config['coxal_input_connected'])
        return row

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),
            physical_body_schema=CoxalBody.SCHEMA,coxal_prior_sha256=PRIOR_SHA256,
            coxal_muscles=7,coxal_canonical_MN=11,coxal_input_connected=self.config['coxal_input_connected'],
            coxal_initialization='Zero new activation, existing physical integration and CNS state preserved',
            coxal_force_biologically_calibrated=False)
        return out

    def save(self,path):
        self._validate_coxal_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Coxal source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_coxal_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)
