"""Canonical CNS/PN with FlyBody motor and sensory feedback.

New physical preparation, unchanged neural equations/history. Old optical
servo is disconnected and retained only as a historical state variable; no
torque or optical ray in the new preparation depends on its position.
"""
from pathlib import Path
import copy
import numpy as np
from pn_graph_solver_session import PnGraphSolverSession, SOURCES as PARENT_SOURCES
from flybody_cns_body import CNSFlyBody, CNSFlyBodyMuscles
from flybody_cns_sensors import FlyBodyProprioception, FlyBodyEye
from flybody_torque_port import FlyBodyEffectiveTibia
from flybody_cns_storage import load_flybody_session
from kc_session_storage import save_session
from kc_audited_session import ROOT, sha256, require_covered_dependencies
from kcgamma_regional_brain import _record_hash
from anatomical_rate_brain import ARRAY_FIELDS

SOURCES=tuple(dict.fromkeys(tuple(PARENT_SOURCES)+('flybody_torque_port.py','flybody_cns_body.py','flybody_cns_sensors.py','flybody_cns_storage.py','flybody_cns_session.py')))
ENTRYPOINT='flybody_cns_session.py'


def neural_fingerprints(s):
    hybrid=s.hybrid.state_dict();hybrid.pop('held_boundary_light')
    return dict(hybrid_except_pending_image=_record_hash(hybrid),
        neural_arrays=_record_hash({n:getattr(s.brain,n) for n in ARRAY_FIELDS}),
        connectivity=_record_hash((s.brain.W.shape,s.brain.W.indptr,s.brain.W.indices,s.brain.W.data)),
        RNG=_record_hash(s.brain.rng.bit_generator.state), plasticity=_record_hash(s.plasticity.state_dict()),
        history=_record_hash((s.history,s.used_light)), pending_MN=_record_hash(s.pending_excitation),
        canonical_IDs=_record_hash(s.brain.node_ids), probe=_record_hash(s.probe))


class FlyBodyCNSSession(PnGraphSolverSession):
    SCHEMA='matrix_flybody_cns_session_v1'

    @classmethod
    def from_checkpoint(cls,path):
        path=Path(path).resolve();parent=PnGraphSolverSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        old_body=parent.body
        try:
            before=neural_fingerprints(parent)
            old_physical=dict(body=parent.body.state_dict(), muscles=parent.muscles.state_dict(),
                pending_sensors=parent.pending_sensors.copy(), pending_proprioception=copy.deepcopy(parent.pending_proprioception),
                pending_light=parent.pending_light.copy(), held_boundary_light=parent.hybrid.held_boundary_light.copy(),
                initial_camera=copy.deepcopy(parent.initial_camera), pending_cyborg_command=parent.pending_cyborg_command)
            obj.body=CNSFlyBody.at_clock(obj.time_ns)
            sensor=parent.proprioception
            obj.proprioception=FlyBodyProprioception.__new__(FlyBodyProprioception)
            obj.proprioception._initialize(obj.brain,obj.body,sensor.polarity,sensor.manifest,sensor.manifest_origin)
            obj.eyes=FlyBodyEye.from_state(obj.brain,obj.body,parent.eyes.state_dict(),obj.rotor)
            obj.muscles=CNSFlyBodyMuscles(obj.body)
            # Same normalized activation history, now acting through explicitly
            # new force-length geometry. Last forces/torques start uncomputed.
            obj.muscles.activation[:]=parent.muscles.activation
            obj.output_connected=False
            obj.pending_cyborg_command=0.
            obj.initial_camera=obj.eyes.pose()
            obj.pending_sensors=obj.world.sense(obj.body.observe())
            obj.pending_proprioception=obj.proprioception.sample()
            obj.pending_light=obj.eyes.sample(obj.light_world,obj.time_ns)
            obj.hybrid.set_boundary_light(obj.boundary_light())
            obj.config=copy.deepcopy(parent.config)
            closure=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
            obj.config.update(candidate=cls.SCHEMA,cyborg_output_connected=False,
                looming_camera_fixed=False,flybody_clock_origin_ns=obj.time_ns,
                flybody_neural_closed_loop=True,flybody_biological_validation=False,
                retired_servo_optical_contribution=False,
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=closure))
            after=neural_fingerprints(obj);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('FlyBody migration changed neural history: '+str(checks))
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Replace physical preparation and return sensory samples from FlyBody; retire detached rotor optical contribution once',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,
                preserved_neural_checks=checks,preserved_neural_sha256=before,
                retired_physical_state_sha256=_record_hash(old_physical),retired_physical_history_in_parent=True,
                retained_muscle_activation=True,new_body_zero_velocity=True,
                new_body_source_qpos0=True,no_claim_of_physical_energy_or_pose_continuity=True,
                physical_clock_origin_ns=obj.time_ns,neural_equations_changed=False,
                neural_gains_changed=False,biological_validation=False,learned_controller=False,
                servo_retained_as_disconnected_historical_device=True)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_afferent_pending()
            old_body.close()
            return obj
        except BaseException:
            old_body.close()
            obj.close()
            raise

    def _validate(self):
        super()._validate()
        if (type(self.body) is not CNSFlyBody or type(self.eyes) is not FlyBodyEye
                or type(self.proprioception) is not FlyBodyProprioception
                or type(self.muscles) is not CNSFlyBodyMuscles
                or self.output_connected or self.pending_cyborg_command!=0.
                or self.config['flybody_clock_origin_ns']!=self.body.origin_ns):
            raise ValueError('Wrong physical body, sensory return or retired optical connection')

    def step(self):
        row=super().step()
        row['retinal_physical_head_motion_rms']=row.pop('retinal_cyborg_motion_rms')
        row['retired_servo_diagnostic']=row.pop('cyborg')
        row['flybody_age_ns']=self.time_ns-self.body.origin_ns
        return row

    def _manifest_fields(self):
        out=super()._manifest_fields()
        out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),
            physical_body_schema='matrix_flybody_cns_body_v1',physical_clock_origin_ns=self.body.origin_ns,
            optical_motion_source='FlyBody physical head',prosthesis_kind='uncalibrated_effective_muscles_and_sensory_transduction',
            bounded_cervical_enabled=False,retired_bounded_cervical_state_retained=True,
            motor_cut_scope='Physical assay must explicitly cut six muscle torques; legacy rotor connection is already zero',
            flybody_neural_closed_loop=True,learned_controller=False,animal_ability_demonstrated=False)
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('FlyBody runtime source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_flybody_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)
