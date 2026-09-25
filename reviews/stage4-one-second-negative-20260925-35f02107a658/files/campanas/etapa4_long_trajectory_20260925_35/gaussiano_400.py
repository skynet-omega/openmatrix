"""Frontera y auditor CPU para piloto S+/S-. No runner neuronal ni CUDA.
No modifica el Gaussian ni las fuentes de etapa3. Su esquema no implica reanudación.
--poses evalúa olor hipotético sobre poses BINARIAS históricas, nunca olor consumido.
"""
import argparse, copy, csv, hashlib, json, time, traceback
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
GEOM_SHA='3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3'
POSE_SHA='fa0c4c905fc2a172e3403510ac675534f9ff7d54393d8e0af7d0cc08dfc5c15b'
DT=1_000_000
PLAN={'schema':'gaussian_intensity_pilot_v1','fields':['plus','minus'],
      'preparation_ms':40,'trial_ms':1000,'source_offset_baselines':[.5,2.],
      'sigma_baselines':2.,'purpose':'intensidad/historia hasta mando; no navegación',
      'stage3_admission':False,'stage4_admission':False}
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def concentration(xy,source,sigma):
    return np.exp(-np.sum((np.asarray(xy)[...,:2]-source)**2,axis=-1)/(2.*sigma**2))
def geometry(path):
    need(sha(path)==GEOM_SHA,'Geometría distinta: no adaptar fuentes al resultado')
    g=json.loads(Path(path).read_text());a=np.array(g['prepared_antennae_mm'])
    q=np.array(g['prepared_qpos_root']);b=float(np.linalg.norm(a[0,:2]-a[1,:2]))
    left=(a[0,:2]-a[1,:2])/b;forward=np.array([left[1],-left[0]])
    w,x,y,z=q[3:];heading=np.array([1-2*(y*y+z*z),2*(w*z+x*y)])
    if forward@heading<0: forward=-forward
    m=a[:,:2].mean(0)
    return g,{s:{'source_mm':(m+sign*.5*b*forward+2*b*left).tolist(),
                 'sigma_mm':2*b,'geometry_sha256':GEOM_SHA}
              for s,sign in (('plus',1),('minus',-1))}
class FixedGaussian:
    """Delega la fórmula/pose al AntennalBoundary original; ignora su fuente histórica."""
    def __init__(self,base,world,arm,spec):
        need(arm in PLAN['fields'],'Campo desconocido')
        self.base,self.world,self.arm=base,world,arm
        self.spec=copy.deepcopy(spec);self.origin_ns=int(world.time_ns)
        self.source=np.asarray(spec['source_mm'],dtype=float).copy()
        self.source.flags.writeable=False;self.sigma=float(spec['sigma_mm'])
        need(self.source.shape==(2,) and np.isfinite(self.source).all() and np.isfinite(self.sigma) and self.sigma>0,'Fuente')
    def sample(self,data,source_mm,sigma_mm):
        elapsed=self.world.time_ns-self.origin_ns
        need(0<=elapsed<=1000*DT and elapsed%DT==0,'Reloj fuera del piloto')
        need(self.world.next_event==len(self.world.events),'Fuente tiene eventos pendientes')
        r=self.base.sample(data,self.source,self.sigma)
        need(np.shape(r['antennae_mm'])==(2,3),'Lado/forma antenal')
        c=np.asarray(r['concentration'])
        need(c.shape==(2,) and np.isfinite(c).all() and ((c>=0)&(c<=1)).all(),'Concentración')
        need(r['contact_reinforcement']==0,'Refuerzo de contacto no permitido')
        return r  # Sin volver a calcular o escalar la señal.
    def metadata(self):
        return dict(schema='fixed_gaussian_intensity_v1',arm=self.arm,installed_ns=self.origin_ns,
                    source_mm=self.source.tolist(),sigma_mm=self.sigma,
                    geometry_sha256=self.spec['geometry_sha256'],sampling_order=['L','R'],
                    source_authority='boundary.json; no world.source_mm histórico',
                    canonical_scale_declared=80.,biological_transduction_calibrated=False)
def clock(obj):
    c,b=obj.core,obj.body;t=int(c.world.time_ns)
    need(c.time_ns==t==b.steps*round(b.dt*1e9),'Relojes cuerpo/mundo/core diferentes')
    need(c.world.body is b,'Propietario corporal distinto')
    return t
POSE_ATOL_NATIVE=1e-10
ANTENNA_ATOL_MM=1e-10

def prepared_pose_guard(obj,base,prepared_row,geometry_path):
    # Frozen deterministic preparation identity, not physiological precision.
    reference_bytes=Path(geometry_path).read_bytes()
    need(hashlib.sha256(reference_bytes).hexdigest()==GEOM_SHA,'Geometría de referencia alterada')
    g=json.loads(reference_bytes)
    expected_q=np.asarray(g['prepared_qpos_root'],dtype=float)
    expected_a=np.asarray(g['prepared_antennae_mm'],dtype=float)
    live_q=np.asarray(obj.body.data.qpos,dtype=float)
    row_q=np.asarray(prepared_row['qpos'],dtype=float)
    row_a=np.asarray(prepared_row['antenas_mm'],dtype=float)
    w=obj.core.world
    live_a=np.asarray(base.sample(obj.body.data,w.source_mm,w.sigma_mm)['antennae_mm'],dtype=float)
    need(expected_q.shape==(7,) and expected_a.shape==(2,3),'Forma de referencia preparada')
    need(live_q.ndim==row_q.ndim==1 and len(live_q)>=7 and len(row_q)>=7,'Forma de qpos preparada')
    need(live_a.shape==row_a.shape==(2,3),'Forma antenal preparada')
    arrays=(expected_q,expected_a,live_q[:7],row_q[:7],live_a,row_a)
    need(all(np.isfinite(x).all() for x in arrays),'Pose preparada no finita')
    for label,actual,expected,tol in (
        ('cuerpo',live_q[:7],expected_q,POSE_ATOL_NATIVE),
        ('antenas vivas',live_a,expected_a,ANTENNA_ATOL_MM),
        ('fila qpos',row_q[:7],live_q[:7],POSE_ATOL_NATIVE),
        ('fila antenas',row_a,live_a,ANTENNA_ATOL_MM)):
        need(np.allclose(actual,expected,rtol=0.,atol=tol,equal_nan=False),'Pose preparada distinta: '+label)
    return dict(geometry_sha256=GEOM_SHA,qpos_root_max_abs_native=float(np.max(abs(live_q[:7]-expected_q))),
                antennae_max_abs_mm=float(np.max(abs(live_a-expected_a))),
                qpos_atol_native=POSE_ATOL_NATIVE,antennae_atol_mm=ANTENNA_ATOL_MM,rtol=0.,
                scope='Root pose and live antenna proxies only; not full prepared-state equivalence')

def install(obj,base,arm,spec,prepared_row,*,geometry_path):
    """Sólo al ON, DESPUÉS de guardar/comparar el estado preparado completo."""
    c,w=obj.core,obj.core.world;t=clock(obj)
    need(prepared_row['fase']=='preparacion' and prepared_row['paso']==40,'Preparación no verificada')
    need(prepared_row['CNS_time_ns']==t and np.all(c.pending_sensors==0),'Preparación/reloj')
    need(t>w.installed_ns,'No sobrescribir primer intervalo histórico')
    need(type(base).__name__=='AntennalBoundary','Pasar frontera Gaussian base, no wrapper lateral')
    ids=[obj.body.model.geom('antenna_'+s+'_collision').id for s in ('left','right')]
    need(base.model is obj.body.model and np.array_equal(base.geoms,ids),'Propietario/orden L/R de geometrías')
    need(w.next_event==len(w.events),'No mover la fuente durante el piloto')
    pose_guard=prepared_pose_guard(obj,base,prepared_row,geometry_path)
    f=FixedGaussian(base,w,arm,spec);old=w.boundary;pending=c.pending_sensors.copy()
    try:
        w.boundary=f;c.pending_sensors=w.sense(obj.body.observe())
        need(c.pending_sensors.shape==(3,) and c.pending_sensors[2]==0,'Entrada sensorial')
        obj._validate_adapter()
        auditor=Intervals(obj,f,prepared_row);auditor.prepared_pose_guard=pose_guard
        return auditor
    except BaseException:
        w.boundary=old;c.pending_sensors=pending
        raise
class Intervals:
    def __init__(self,obj,field,prepared):
        self.field=field;self.origin=clock(obj);self.n=0;self.armed=False;self.invalid=False
        self.expected=obj.core.pending_sensors.copy();self.q=np.array(prepared['DN_q_actual'],copy=True)
        self.baseline=np.array(prepared['DN_baseline'],copy=True);self.records=[]
    def before(self,obj):
        need(not self.invalid and not self.armed and self.n<1000,'Auditor inválido/reentrante')
        need(obj.core.world.boundary is self.field,'Frontera reemplazada')
        need(clock(obj)==self.origin+self.n*DT,'Inicio de intervalo incorrecto')
        need(np.array_equal(obj.core.pending_sensors,self.expected),'Muestra pendiente alterada')
        self.used=self.expected.copy();self.armed=True
        return self.used.copy()
    def after(self,obj,row):
        try:
            need(self.armed and not self.invalid,'Sin inicio de intervalo')
            end=self.origin+(self.n+1)*DT;need(clock(obj)==end,'Paso físico no es 1 ms')
            need(row['fase']=='ensayo' and row['paso']==self.n+1,'Paso/fase')
            need(all(row[k]==end for k in ('CNS_time_ns','PN_time_ns','body_time_ns')),'Reloj de traza')
            need(np.array_equal(row['sensores_usados'],self.used),'La muestra usada no fue la comprometida')
            geom=self.field.sample(obj.body.data,obj.core.world.source_mm,obj.core.world.sigma_mm)
            need(np.array_equal(row['antenas_mm'],geom['antennae_mm']),'Pose/lado de antenas')
            need(np.array_equal(row['concentracion_campo'],geom['concentration']),'Campo/traza')
            pending=np.r_[geom['concentration'],0.]
            need(np.array_equal(row['sensores_pendientes'],pending) and np.array_equal(obj.core.pending_sensors,pending),'Pending vs pose final')
            need(np.array_equal(row['DN_q_usada'],self.q),'Lag neural no es un intervalo')
            need(np.array_equal(row['DN_baseline'],self.baseline),'Baseline cambiado')
            dq=self.q-self.baseline
            command=np.tanh(250*(dq[2]-dq[3]))*np.deg2rad(5.)
            need(command==row['command_yaw_rate_rad_s'],'Lector angular alterado')
            r={'sample_used_ns':end-DT,'sample_pending_ns':end,'used':self.used.tolist(),
               'pending':pending.tolist(),'expected_80c_used_NOT_consumer_witness':(80*self.used[:2]).tolist(),
               'PN_available_NOT_consumer_witness':np.asarray(row['PN_general_transmission']).tolist(),
               'command_yaw_rate_rad_s':float(command)}
            self.records.append(r);self.expected=pending.copy();self.q=np.array(row['DN_q_actual'],copy=True)
            self.n+=1;self.armed=False;return r
        except BaseException:
            self.invalid=True
            raise
    def abort(self): self.invalid=True
    def export(self,path):
        save(path,{'field':self.field.metadata(),'prepared_pose_guard':self.prepared_pose_guard,'intervals':self.records,
                   'complete':self.n==1000 and not self.invalid and not self.armed,
                   'input_contract_only':True,'stage4_admission':False})
def pose_readback(path,specs,out):
    need(sha(path)==POSE_SHA,'CSV histórico distinto')
    with Path(path).open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    need(len(rows)==1604,'Faltan poses históricas')
    vals=[]
    for arm in ('sham','odor_left','odor_right','uniform'):
        subset=[r for r in rows if r['arm']==arm]
        need([r['phase'] for r in subset]==['preparacion']+['ensayo']*400,'Fases CSV')
        need([int(r['step']) for r in subset]==[40]+list(range(1,401)),'Tiempos CSV')
        for r in subset:
            xy=np.array([[float(r['antenna_'+s+'_'+axis+'_mm']) for axis in ('x','y')] for s in ('left','right')])
            cs=[concentration(xy,specs[s]['source_mm'],specs[s]['sigma_mm']) for s in PLAN['fields']]
            for s,c in zip(PLAN['fields'],cs):
                vals.append([arm,r['phase'],int(r['step']),s,*c,c.mean(),(c[0]-c[1])/c.sum()])
    with (out/'GAUSSIANA_HIPOTETICA.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['arm_binary','phase','step','hypothetical_source','cL','cR','common','normalized_LR']);w.writerows(vals)
    return {'poses':len(rows),'sha256':POSE_SHA,'scope':'Poses binarias; ninguna entrada gaussiana consumida'}
def selftest(g,specs,geometry_path):
    """Dobles CPU de API, no MuJoCo ni AntennalWorld ejecutado."""
    points=np.array(g['prepared_antennae_mm']);baseline=np.zeros(4);q=np.array([0.,0.,.2,.1])
    class AntennalBoundary:
        geoms=np.array([0,1])
        def sample(self,data,source,sigma):
            c=concentration(data.points,source,sigma)
            return dict(antennae_mm=data.points.copy(),concentration=c,canonical_ORN_drive=80*c,contact_reinforcement=0.)
    class World:
        def sense(self,unused):
            if self.time_ns==self.installed_ns:return self.committed_sensors.copy()
            return np.r_[self.boundary.sample(self.body.data,self.source_mm,self.sigma_mm)['concentration'],0.]
    def setup():
        b=NS(dt=.000025,steps=1600,data=NS(points=points.copy(),qpos=np.array(g['prepared_qpos_root'],copy=True)),observe=lambda:None,
             model=NS(geom=lambda name:NS(id=0 if name=='antenna_left_collision' else 1)))
        w=World();w.body=b;w.time_ns=40*DT;w.installed_ns=0;w.events=[];w.next_event=0
        w.source_mm=np.zeros(2);w.sigma_mm=1.;w.boundary=AntennalBoundary();w.boundary.model=b.model;w.committed_sensors=np.zeros(3)
        c=NS(world=w,time_ns=w.time_ns,pending_sensors=np.zeros(3));o=NS(core=c,body=b,_validate_adapter=lambda:None)
        prep=dict(fase='preparacion',paso=40,CNS_time_ns=w.time_ns,DN_q_actual=q,DN_baseline=baseline,qpos=b.data.qpos.copy(),antenas_mm=points.copy())
        return o,prep
    o,p=setup();aud=install(o,o.core.world.boundary,'plus',specs['plus'],p,geometry_path=geometry_path);used=aud.before(o)
    o.body.steps+=40;o.core.time_ns+=DT;o.core.world.time_ns+=DT
    o.body.data.points[:,:2]+=[.0002,0.]
    geom=aud.field.sample(o.body.data,None,None);o.core.pending_sensors=o.core.world.sense(None)
    row=dict(fase='ensayo',paso=1,CNS_time_ns=41*DT,PN_time_ns=41*DT,body_time_ns=41*DT,
        sensores_usados=used,sensores_pendientes=o.core.pending_sensors.copy(),antenas_mm=geom['antennae_mm'],
        concentracion_campo=geom['concentration'],DN_q_usada=q,DN_q_actual=q+.001,DN_baseline=baseline,
        command_yaw_rate_rad_s=np.tanh(250*(q[2]-q[3]))*np.deg2rad(5.),PN_general_transmission=np.array([.2]))
    # Clonar solo el estado diagnóstico; todas las copias siguen el mismo objeto fixture.
    rejected=[]
    for field in ('clock','used','side'):
        trial=copy.copy(aud);trial.records=[];bad=copy.deepcopy(row)
        if field=='clock':bad['PN_time_ns']+=1
        elif field=='used':bad['sensores_usados'][0]+=.01
        else:bad['antenas_mm']=bad['antenas_mm'][::-1]
        try:trial.after(o,bad)
        except ValueError:rejected.append(field)
        else:raise AssertionError('Corrupción aceptada '+field)
    aud.after(o,row)
    c=[concentration(points,specs[s]['source_mm'],specs[s]['sigma_mm']) for s in PLAN['fields']]
    need(np.max(abs(c[0]-c[1]))<1e-12 and all(x[0]>x[1] for x in c),'Igualdad/lado inicial')
    return dict(scope='CPU fixture API con geometría real; no organismo',rejected=rejected,
                one_interval_checked=True,initial_L_R=c[0].tolist(),initial_pair_error=float(np.max(abs(c[0]-c[1]))))
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--poses',type=Path);ap.add_argument('--selftest',action='store_true');ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False);save(a.out/'PLAN.json',PLAN);r={};t=time.perf_counter()
    try:
        g,specs=geometry(a.geometry);save(a.out/'CAMPOS.json',specs)
        if a.selftest:r['test']=selftest(g,specs,a.geometry)
        if a.poses:r['readback']=pose_readback(a.poses,specs,a.out)
        r['status']='CPU_COMPLETE'
    except Exception:r.update(status='FAILED_RETAINED',error=traceback.format_exc())
    r.update(wall_s=time.perf_counter()-t,code_sha256=sha(Path(__file__)),organism_executed=False,CUDA_executed=False)
    save(a.out/'RESULTADO.json',r);print(json.dumps(r,indent=2,ensure_ascii=False));return 0 if r['status']=='CPU_COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
