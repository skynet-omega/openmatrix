"""Contrato externo PROSPECTIVO; sólo lectura. No certifica el organismo.
Pares: causal_cuda/reference_cuda de la MISMA intervención y 40+400 ms.
No interpola, ajusta lag, cambia baseline ni modifica el gate histórico.
"""
import argparse, hashlib, json, traceback, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
EPS, MATERIAL, DT = .002, .02, .001  # grados; segundos
ARMS = ('sham','odor_left','odor_right','uniform')
IDS = (10176,10208,10360,523769,10065,10118)
PLAN = dict(schema='orientation_functional_v1', yaw_sup_deg=EPS,
    command_L1_deg=EPS, material_contrast_deg=MATERIAL,
    sign_reserve_deg=EPS, contrast_reserve_deg=2*EPS,
    sample_ms=1, preparation_ms=40, trial_ms=400,
    historical_hidden_state_gate='FAIL_PRESERVADO', stage3_admission=False)
def need(ok,msg):
    if not ok: raise ValueError(msg)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def mx(v): return float(np.max(np.abs(v)))
def reader(q,b):
    d=q-b
    return np.tanh(250*(d[:,2]-d[:,3]))*np.deg2rad(5)
def load(folder,arm,engine,hashes):
    def file(name):
        p=folder/name;hashes[str(p)]=digest(p);return p
    def js(name): return json.loads(file(name).read_text(encoding='utf-8'))
    def arrays(name):
        p=file(name)
        with zipfile.ZipFile(p) as z: need(sum(a.file_size for a in z.infolist())<256*1024**2,'NPZ demasiado grande')
        with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    r,c=js('RESULT.json'),js('RUN_CONTRACT.json'); intervention=js('INTERVENTION.json');frozen=js('FROZEN.json')
    need(r['status']=='COMPLETE' and r['error'] is None and not r['cleanup_errors'],'Ejecución fallida')
    need(r['odor']==c['odor']==arm and r['engine']==c['engine']==engine,'Brazo/perfil incorrecto')
    need(r['completed_preparation_ms']==c['preparation_ms']==40 and r['completed_trial_ms']==c['trial_ms']==400,'Horizonte incompleto')
    need(c['event_boundaries'] is True and intervention['dynamic_466_enabled'] is True,'Fronteras/rutas cambiadas')
    need(intervention['anatomy_changed'] is False and intervention['motor_decoder_changed'] is False,'Otra intervención')
    sig=({k:c[k] for k in ('checkpoint_manifest_sha256','interface_intervention','event_representation')},intervention,frozen)
    t,f=arrays('traces.npz'),arrays('flow/FLOW.npz');flow_receipt=js('flow/RESULT.json')
    need(flow_receipt['samples']==440 and flow_receipt['max_target_error']<=1e-9 and flow_receipt['max_rate_error']<=1e-12,'Tap no reconstruye')
    for data in (t,f):
        for k,v in data.items():
            if v.dtype.kind in 'fiu': need(np.isfinite(v).all(),'No finito: '+k)
    phase=np.array(['preparacion']*40+['ensayo']*400)
    need(np.array_equal(t['fase'],phase) and np.array_equal(t['paso'],np.r_[np.arange(1,41),np.arange(1,401)]),'Fases/pasos')
    for v in t.values(): need(v.ndim>0 and len(v)==440,'Traza incompleta')
    clock=t['CNS_time_ns'];need(clock.dtype.kind in 'iu' and clock.shape==(440,) and np.all(np.diff(clock)==1000000),'Reloj CNS')
    for k in ('PN_time_ns','body_time_ns'): need(np.array_equal(clock,t[k]),'Relojes físicos distintos')
    exposure={'sham':[0,0,0],'odor_left':[1,0,0],'odor_right':[0,1,0],'uniform':[1,1,0]}[arm]
    need(t['sensores_usados'].shape==(440,3) and np.all(t['sensores_usados'][:40]==0),'Preparación no limpia')
    need(np.array_equal(t['sensores_usados'][40:],np.tile(exposure,(400,1))),'Exposición diferente')
    need(np.array_equal(f['ids'],IDS) and np.array_equal(f['time_ns'],clock) and np.array_equal(f['phase'],phase),'Identidad/tiempo flujo')
    need(f['target'].shape==f['raw_signed'].shape==(440,6),'Forma flujo')
    q,u,b=[t[k] for k in ('DN_q_actual','DN_q_usada','DN_baseline')]
    need(q.shape==u.shape==b.shape==(440,4),'Lector declarado DNb05: columnas 2/3')
    need(np.array_equal(u[1:],q[:-1]) and np.array_equal(b,np.tile(b[0],(440,1))),'Lag/baseline')
    omega=t['command_yaw_rate_rad_s'];need(omega.shape==(440,) and np.array_equal(reader(u,b),omega),'Decoder cambiado')
    pos=t['qpos'];need(pos.ndim==2 and pos.shape[1]>=7,'Forma corporal')
    w,x,y,z=pos[:,3:7].T;need(mx(w*w+x*x+y*y+z*z-1)<1e-8,'Quaternion inválido')
    psi=np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
    theta=psi[40:]-psi[39];need(mx(theta-t['yaw_delta_deg'][40:])<1e-10,'Yaw no corresponde a pose/ON')
    active=t['contact_active'];normal=t['normal_force_N'];up=t['upright']
    need(active.ndim==2 and normal.shape==active.shape and up.shape==(440,),'Forma apoyo')
    support=bool(np.all(up>0) and np.all(active.astype(bool).sum(1)>=1) and np.all(normal.sum(1)>0))
    log=js('EVENT_AUDIT.json') if (folder/'EVENT_AUDIT.json').is_file() else None
    return dict(t=t,f=f,sig=sig,theta=theta,omega=omega,support=support,events=log)
def event_report(a,b):
    if a is None or b is None: return {'status':'NO_PUBLICADO; no se supone igualdad'}
    def collect(log):
        out=defaultdict(list);pred=0
        for block in log['blocks']:
            d=block['duration_ns'];need(d in (62500,125000),'Fase temporal desconocida')
            if d==62500: pred+=len(block['events']);continue
            for e in block['events']:
                ts=float(e['time_s']);post=e['post_q'];need(np.isfinite(ts) and 0<=ts<=d*1e-9 and (post is None or np.isfinite(post)),'Evento inválido')
                out[(e['producer'],e['row'],e['neuron_id'])].append((block['start_elapsed_ns']+ts*1e9,post))
        for v in out.values():v.sort(key=lambda x:x[0])
        return out,pred
    aa,pa=collect(a);bb,pb=collect(b);changed=[];dt=[];dq=[]
    for k in sorted(set(aa)|set(bb)):
        x,y=aa.get(k,[]),bb.get(k,[])
        if len(x)!=len(y):changed.append({'key':k,'counts':[len(x),len(y)]});continue
        for (tx,qx),(ty,qy) in zip(x,y):
            dt.append(abs(tx-ty))
            if qx is not None and qy is not None:dq.append(abs(qx-qy))
    return dict(status='DESCRIPTIVO; no nuevo gate discreto',predictor_counts=[pa,pb],
        accepted_counts=[sum(map(len,aa.values())),sum(map(len,bb.values()))],count_differences=changed,
        max_ordinal_time_ns=max(dt) if dt else None,max_post_q=max(dq) if dq else None)
def compare(a,b):
    need(a['sig']==b['sig'],'Operador/intervención/fuentes no equivalentes')
    x,y=a['t'],b['t'];need(set(x)==set(y),'Esquema de trazas distinto')
    need(np.array_equal(x['CNS_time_ns'],y['CNS_time_ns']) and np.array_equal(x['DN_baseline'],y['DN_baseline']),'Inicio/baseline distintos')
    yaw=max(mx(a['theta']-b['theta']),mx(x['yaw_delta_deg'][:40]-y['yaw_delta_deg'][:40]))
    cmd=float(np.rad2deg(np.sum(np.abs(a['omega']-b['omega']))*DT)) # incluye preparación
    discrete={k:int(np.count_nonzero(x[k]!=y[k])) for k in x if x[k].dtype.kind in 'biu' and not np.array_equal(x[k],y[k])}
    floats={k:mx(x[k]-y[k]) for k in x if x[k].dtype.kind in 'fc'}
    targets={str(i):mx(a['f']['target'][:,j]-b['f']['target'][:,j]) for j,i in enumerate(IDS)}
    return dict(observation_pair_ok=bool(yaw<=EPS and cmd<=EPS and a['support'] and b['support']),
        yaw_sup_deg=yaw,command_L1_deg=cmd,errors_by_trace_native_units=floats,
        targets_by_id=targets,raw_by_id={str(i):mx(a['f']['raw_signed'][:,j]-b['f']['raw_signed'][:,j]) for j,i in enumerate(IDS)},
        discrete_trace_changes=discrete,events=event_report(a['events'],b['events']))
def orientation(runs):
    if set(runs)!=set(ARMS):return {'status':'PENDIENTE_CUATRO_BRAZOS','complete':False}
    values=[]
    for p in (0,1):
        base=runs['sham'][p]
        for arm in ARMS:
            d=runs[arm][p];need(d['sig']==base['sig'],'Fuentes diferentes entre olores')
            need(set(d['t'])==set(base['t']),'Esquema distinto entre olores')
            need(all(np.array_equal(d['t'][k][:40],base['t'][k][:40]) for k in base['t']),'Preparación no idéntica dentro del perfil')
        v={arm:float(runs[arm][p]['theta'][-1]) for arm in ARMS}
        margins={f'L-{c}':v['odor_left']-v[c] for c in ('sham','uniform')}
        margins.update({f'{c}-R':v[c]-v['odor_right'] for c in ('sham','uniform')})
        ok=v['odor_left']>EPS and v['odor_right']<-EPS and min(margins.values())>MATERIAL+2*EPS
        values.append(dict(yaw_400_deg=v,margins_deg=margins,odd_deg=(v['odor_left']-v['odor_right'])/2,directional_observation_ok=bool(ok)))
    return {'complete':True,'profiles':values,'both_directional':all(v['directional_observation_ok'] for v in values)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8'));need(plan['schema']==PLAN['schema'] and plan.get('registered_utc'),'Plan sin registro')
    need(plan['pairs'] and set(plan['pairs'])<=set(ARMS),'Brazos inválidos')
    a.out.mkdir(exist_ok=False);save(a.out/'CONTRATO_EFECTIVO.json',dict(criteria=PLAN,input_plan=plan))
    report={'stage3_admission':False,'historical_gate':'FAIL_PRESERVADO','input_hashes':{},'pairs':{}}
    try:
        runs={}
        for arm,paths in plan['pairs'].items():
            need(set(paths)=={'causal','reference'},'Par incompleto')
            dirs=[(a.plan.parent/paths[k]).resolve() for k in ('causal','reference')];need(dirs[0]!=dirs[1],'Misma ejecución usada dos veces')
            pair=[load(d,arm,e,report['input_hashes']) for d,e in zip(dirs,('causal_cuda','reference_cuda'))]
            report['pairs'][arm]=compare(*pair);runs[arm]=pair
        report['orientation']=orientation(runs)
        report['observations_concordant']=all(x['observation_pair_ok'] for x in report['pairs'].values())
        report['remaining']='Retirada del lector en organismo acoplado y continuación propia no verificadas por este analizador. Sin certificado entre muestras ni de error respecto de la solución exacta.'
        report['status']='ANALISIS_COMPLETO'
    except Exception:report.update(status='BLOQUEADO',error=traceback.format_exc())
    report['code_sha256']=digest(Path(__file__));save(a.out/'RESULTADO.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False));return 0 if report['status']=='ANALISIS_COMPLETO' and report['observations_concordant'] else 2
if __name__=='__main__':raise SystemExit(main())

