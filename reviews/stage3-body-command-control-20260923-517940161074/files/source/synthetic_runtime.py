"""CNS223 with an explicit, replaceable synthetic leg force strategy.

Load the OUTER archive with Runtime.load. The inner core_archive stores the
unchanged neural runtime and all state/sensors. Its native motor-force strategy
is retired by this wrapper; it is not a standalone continuation of this branch.
"""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
import mujoco as mj
R=Path(__file__).resolve().parents[2]; D=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'src'))
from coefficient_buffer_session import CoefficientBufferSession
from session_io import read_state, write_state
CP=R/'work/stage2_synthetic_prosthesis_20260915/controller.py'
spec=importlib.util.spec_from_file_location('prosthesis_controller',CP)
ctrl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctrl)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def rk4(a,u,parameters,dt):
    def f(x):return np.array([mj.mju_muscleDynamics(float(v),float(w),p) for v,w,p in zip(u,x,parameters)])
    k1=f(a);k2=f(a+.5*dt*k1);k3=f(a+.5*dt*k2);k4=f(a+dt*k3)
    out=a+dt*(k1+2*k2+2*k3+k4)/6
    if not np.isfinite(out).all() or np.any((out<0)|(out>1)):raise ValueError('Shadow activation left bounds')
    return out

class Runtime:
    SCHEMA='CNS223_synthetic_leg_strategy_v1'
    def __init__(self,core):
        self.core=core;self.body=core.body
        self.origin_ns=core.time_ns;self.active=False;self.withdrawals=0
        self.controller=None;self.native_advance=self.body.advance
        self.command_mode='hold';self.requested=np.zeros(2);self.last_force=np.zeros(self.body.model.nv)
        self.last_ignored_tibia=np.zeros(6);self.last_work_J=0.;self.synthetic_steps=0
        self.dn_ids=np.array([10045,10056,523769,10360]);self.dn_ix=np.searchsorted(core.brain.node_ids,self.dn_ids)
        np.testing.assert_array_equal(core.brain.node_ids[self.dn_ix],self.dn_ids)
        self.dn_baseline=core.hybrid.release()[self.dn_ix].copy();self.last_dn=self.dn_baseline.copy()
        # Instance-owned strategy, never a global MuJoCo callback or core edit.
        self.body.advance=self.body_advance

    @classmethod
    def from_core(cls,path):
        obj=cls(CoefficientBufferSession.load(path))
        obj.parent=str(Path(path).resolve());obj.parent_manifest=sha(Path(path)/'manifest.json')
        return obj

    def set_command(self,speed=0.,steering=0.,mode='device'):
        if mode not in ('hold','device','neural'):raise ValueError('Undeclared command policy')
        if not np.isfinite([speed,steering]).all() or not 0<=speed<=1 or abs(steering)>1:raise ValueError('Command bounds')
        self.command_mode=mode;self.requested=np.array([speed,steering],float)

    def _withdraw(self):
        if self.active or self.withdrawals:raise ValueError('Duplicate force withdrawal')
        b=self.body
        # Preserve normalized activation and pending inputs as shadow states.
        # Only their physical force authority is removed.
        for name in ['last_tension_N','last_coxal_torque_Nm','last_tr_tension_N','last_tr_torque_Nm',
                     'last_rf_tension_N','last_rf_torque_Nm','last_serial_force_N','last_serial_generalized_SI',
                     'ft_last_force_N','ft_last_generalized_SI']:
            getattr(b,name).fill(0.)
        b.last_tarsal_torque_Nm=0.;b.rh_last_torque_Nm=0.
        b.data.qfrc_applied[:]=0.;b.data.ctrl[:]=0.
        self.controller=ctrl.Controller(b.model,b.data)
        self.active=True;self.withdrawals=1

    def _shadow_step(self):
        b=self.body;dt=b.dt;ns=round(dt*1e9)
        for a,u,p in [('activation','pending_coxal','dynamics'),('tr_activation','pending_tr','tr_dynamics'),('rf_activation','pending_rf','rf_dynamics')]:
            setattr(b,a,rk4(getattr(b,a),getattr(b,u),getattr(b,p),dt))
        z=b.serial_bank;z.activation=rk4(z.activation,b.pending_serial,z.dynamics,dt);z.time_ns+=ns
        for name,pending,prior in [('tarsal_activation','pending_tarsal','tarsal_prior'),('rh_activation','rh_pending','rh_prior')]:
            a=getattr(b,name);u=getattr(b,pending);p=getattr(b,prior)
            tau=np.where(u>a,p['activation_tau_s'],p['deactivation_tau_s'])
            out=a+-np.expm1(-dt/tau)*(u-a)
            setattr(b,name,float(out) if name=='rh_activation' else out)
        for clock in ['coxal_time_ns','tr_time_ns','rf_time_ns','tarsal_time_ns','rh_time_ns']:
            setattr(b,clock,getattr(b,clock)+ns)

    def body_advance(self,torque_native,nsteps=1):
        b=self.body
        if type(nsteps) is not int or nsteps<1:raise ValueError('Invalid physical steps')
        for _ in range(nsteps):
            if b.steps*round(b.dt*1e9)<self.origin_ns+1000000:
                self.native_advance(torque_native,1);continue
            if not self.active:self._withdraw()
            b.validate_coxa()
            if np.any(b.data.xfrc_applied):raise ValueError('Undeclared body force')
            self.last_ignored_tibia=np.asarray(torque_native).copy()
            self._shadow_step()
            speed,steering=self.requested
            if self.command_mode=='hold':speed=steering=0.
            elif self.command_mode=='neural':
                # Explicit device decoder: basal stepping is supplied. Only
                # canonical DN release enters it, never odor/source coordinates.
                dq=self.last_dn-self.dn_baseline
                speed=float(np.clip(.25+np.mean(dq[:2]),0.,1.))
                steering=float(np.tanh(10.*(dq[2]-dq[3])))
            self.controller.set_command(float(speed),float(steering))
            force=self.controller.torque(b.data,b.dt)
            if np.any(force[:6]) or not np.isfinite(force).all():raise ValueError('Root force/nonfinite force')
            self.last_force=force.copy();self.last_work_J=float(force@b.data.qvel*b.dt*1e-7)
            b.data.ctrl[:]=0.;b.data.qfrc_applied[:]=force
            mj.mj_step(b.model,b.data);b.steps+=1;self.synthetic_steps+=1
            # External commands are consumed each step, not accumulated.
            # Last applied synthetic force is persisted in the outer sidecar.
            b.data.qfrc_applied[:]=0.;b.data.ctrl[:]=0.
            if np.any(b.data.warning.number) or not np.isfinite(b.data.qpos).all() or not np.isfinite(b.data.qvel).all():raise FloatingPointError('Synthetic physical integration failed')
            b.validate_coxa()

    def step(self):
        self.last_dn=self.core.hybrid.release()[self.dn_ix].copy()
        row=self.core.step()
        row['synthetic_locomotion']=dict(active=self.active,mode=self.command_mode,withdrawals=self.withdrawals,
            dn_ids=self.dn_ids.tolist(),dn_release_used=self.last_dn.tolist(),requested=self.requested.tolist(),
            command=None if self.controller is None else [self.controller.speed,self.controller.steering],
            leg_force_Nm=(self.last_force[6:]*1e-7).tolist(),force_scale_Nm=1e-7,
            native_force_authority='retired; activation/pending remain shadow states' if self.active else 'committed first millisecond',
            biological_locomotion=False)
        return row

    def state(self):
        return dict(schema=self.SCHEMA,origin_ns=self.origin_ns,active=self.active,withdrawals=self.withdrawals,
            controller=None if self.controller is None else self.controller.state_dict(),command_mode=self.command_mode,
            requested=self.requested.copy(),last_force=self.last_force.copy(),last_ignored_tibia=self.last_ignored_tibia.copy(),
            last_work_J=self.last_work_J,synthetic_steps=self.synthetic_steps,dn_baseline=self.dn_baseline.copy(),last_dn=self.last_dn.copy(),
            parent=self.parent,parent_manifest=self.parent_manifest,
            sources={str(p.relative_to(R)):sha(p) for p in [Path(__file__),CP,CP.with_name('prototype.py')]})

    def save(self,path):
        path=Path(path);path.mkdir(parents=True,exist_ok=False)
        self.core.save(path/'core_archive')
        write_state(path/'prosthesis',self.state())
        manifest=dict(schema=self.SCHEMA,entrypoint=str(Path(__file__).resolve()),neuron_count=self.core.brain.n_neurons,
            core_runtime_sources=len(self.core.source_identity),synthetic_function=True,biological_validation=False,
            core_archive_is_not_standalone=True,time_ns=self.core.time_ns,withdrawals=self.withdrawals,
            files={str(p.relative_to(path)):sha(p) for p in [path/'core_archive/manifest.json',path/'prosthesis.json',path/'prosthesis.npz']})
        (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    @classmethod
    def load(cls,path):
        path=Path(path);m=json.loads((path/'manifest.json').read_text());s=read_state(path/'prosthesis')
        if m['schema']!=cls.SCHEMA or s['schema']!=cls.SCHEMA:raise ValueError('Wrong outer schema')
        for n,h in m['files'].items():
            if sha(path/n)!=h:raise ValueError('Changed archive '+n)
        for n,h in s['sources'].items():
            if sha(R/n)!=h:raise ValueError('Changed strategy source '+n)
        obj=cls(CoefficientBufferSession.load(path/'core_archive'))
        for name in ['origin_ns','active','withdrawals','command_mode','requested','last_force','last_ignored_tibia','last_work_J',
                     'synthetic_steps','dn_baseline','last_dn','parent','parent_manifest']:setattr(obj,name,s[name])
        if s['controller'] is not None:obj.controller=ctrl.Controller.from_state(obj.body.model,obj.body.data,s['controller'])
        if obj.withdrawals!=int(obj.active):raise ValueError('Force ownership mismatch')
        return obj

    def close(self):self.core.close()
