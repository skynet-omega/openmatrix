"""Whole CNS carrier for an explicitly synthetic contact-traction prosthesis.

The inner carrier has a distinct schema and cannot be loaded as CNS223.
Only this outer runtime restores the prosthetic contact-force owner.
"""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import mujoco as mj
R=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'work/stage2_prosthesis_cns_20260915'))
from runtime import Runtime as Parent, CoefficientBufferSession, read_state, write_state, sha
CP=R/'work/stage2_contact_prosthesis_20260915/controller.py'

class Carrier(CoefficientBufferSession):
    SCHEMA='CNS223_carrier_requires_synthetic_contact_runtime_v1'

class ContactRuntime(Parent):
    SCHEMA='CNS223_synthetic_contact_prosthesis_v1'

    def set_command(self,speed=0.,steering=0.,mode='device'):
        if not np.isfinite([speed,steering]).all() or not 0<=speed<=.5 or abs(steering)>1:
            raise ValueError('Contact device accepts 0..0.5 mm/s and steering -1..1')
        super().set_command(speed,steering,mode)

    def _module(self):
        spec=importlib.util.spec_from_file_location('contact_prosthesis_controller',CP)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def _withdraw(self):
        super()._withdraw()
        self.controller=self._module().Controller(self.body.model,self.body.data)

    def body_advance(self,torque_native,nsteps=1):
        b=self.body
        if type(nsteps) is not int or nsteps<1:raise ValueError('Invalid physical steps')
        for _ in range(nsteps):
            if b.steps*round(b.dt*1e9)<self.origin_ns+1000000:
                self.native_advance(torque_native,1);continue
            if not self.active:self._withdraw()
            b.validate_coxa()
            if np.any(b.data.xfrc_applied):raise ValueError('Undeclared body wrench')
            self.last_ignored_tibia=np.asarray(torque_native).copy();self._shadow_step()
            speed,steering=self.requested
            if self.command_mode=='hold':speed=steering=0.
            elif self.command_mode=='neural':
                dq=self.last_dn-self.dn_baseline
                speed=float(np.clip(.2+np.mean(dq[:2]),0.,.5))
                steering=float(np.tanh(10.*(dq[2]-dq[3])))
            # Production commissioning is 400 ms recovery + 200 ms at rest.
            # An explicitly marked early wiring probe bypasses only this wait.
            if not self.controller.activation_is_probe and self.controller.elapsed_ns<600000000:
                speed=steering=0.
            self.controller.set_command(forward_mm_s=float(speed),yaw_rate_rad_s=float(steering*np.deg2rad(5.)))
            force=self.controller.torque(b.data,b.dt)
            if force.shape!=(b.model.nv,) or not np.isfinite(force).all():raise ValueError('Invalid prosthetic force')
            # A force applied at a foot contact correctly has free-root entries
            # through J.T F. They are not direct root actuation and are logged.
            self.last_force=force.copy();self.last_work_J=float(force@b.data.qvel*b.dt*1e-7)
            b.data.ctrl[:]=0.;b.data.qfrc_applied[:]=force
            mj.mj_step(b.model,b.data);b.steps+=1;self.synthetic_steps+=1
            b.data.qfrc_applied[:]=0.;b.data.ctrl[:]=0.
            if np.any(b.data.warning.number) or not np.isfinite(b.data.qpos).all() or not np.isfinite(b.data.qvel).all():raise FloatingPointError('Contact prosthesis integration failed')
            b.validate_coxa()

    def step(self):
        self.last_dn=self.core.hybrid.release()[self.dn_ix].copy()
        row=self.core.step()
        row['synthetic_contact_prosthesis']=dict(active=self.active,mode=self.command_mode,withdrawals=self.withdrawals,
            requested=self.requested.tolist(),dn_ids=self.dn_ids.tolist(),dn_release_used=self.last_dn.tolist(),
            last_generalized_force_native=self.last_force.tolist(),root_entries_are_foot_contact_reactions=True,
            six_leg_walking_claimed=False,biological_validation=False)
        return row

    def state(self):
        s=super().state();s['schema']=self.SCHEMA
        sources=[Path(__file__),CP,R/'work/stage2_prosthesis_cns_20260915/runtime.py',
                 R/'work/stage2_synthetic_prosthesis_20260915/controller.py',R/'work/stage2_synthetic_prosthesis_20260915/prototype.py']
        s['sources']={str(p.relative_to(R)):sha(p) for p in sources}
        return s

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        original=self.core.__class__
        try:
            self.core.__class__=Carrier
            self.core.save(path/'core_carrier')
        finally:self.core.__class__=original
        write_state(path/'prosthesis',self.state())
        manifest=dict(schema=self.SCHEMA,entrypoint=str(Path(__file__).resolve()),neuron_count=self.core.brain.n_neurons,
            core_runtime_sources=len(self.core.source_identity),synthetic_function=True,biological_validation=False,
            core_carrier_schema=Carrier.SCHEMA,time_ns=self.core.time_ns,withdrawals=self.withdrawals,
            files={str(p.relative_to(path)):sha(p) for p in [path/'core_carrier/manifest.json',path/'prosthesis.json',path/'prosthesis.npz']})
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path);m=json.loads((path/'manifest.json').read_text());s=read_state(path/'prosthesis')
        if m['schema']!=cls.SCHEMA or s['schema']!=cls.SCHEMA:raise ValueError('Wrong contact prosthesis schema')
        for n,h in m['files'].items():
            if sha(path/n)!=h:raise ValueError('Changed archive '+n)
        for n,h in s['sources'].items():
            if sha(R/n)!=h:raise ValueError('Changed strategy '+n)
        core=Carrier.load(path/'core_carrier');core.__class__=CoefficientBufferSession
        obj=cls(core)
        for name in ['origin_ns','active','withdrawals','command_mode','requested','last_force','last_ignored_tibia','last_work_J',
                     'synthetic_steps','dn_baseline','last_dn','parent','parent_manifest']:setattr(obj,name,s[name])
        if s['controller'] is not None:obj.controller=obj._module().Controller.from_state(obj.body.model,obj.body.data,s['controller'])
        if obj.withdrawals!=int(obj.active):raise ValueError('Force owner mismatch')
        return obj
