"""Frozen-window reconstruction from all eight arms; no CNS integration."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = tuple(s+a+b for s in ('p','m') for a,b in ('00','01','10','11'))
PLAN_HASH = '0518e0c37d35e8a3d55467b40eea87620b53ff0f60bc412931c93b54ea16129c'
DN_IDS = np.array([10045,10056,10118,10065,523769,10360],np.int64)
SL = slice(10,130)

def need(ok,message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def read_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def expected_schedule(arm):
    """Independent indexing implementation, not imported from stimulus owner."""
    need(arm in ARMS,'Unknown arm')
    bits = (1,0,1,1,0,0,0,1,1,0,1,0)
    lag = 40 if arm[0]=='p' else -40
    x = np.zeros((140,2),np.float64)
    for t in range(120):
        x[t+10,0] = bits[t//10] ^ int(arm[1])
        x[t+10,1] = bits[((t-lag)%120)//10] ^ int(arm[2])
    return x

def contrast(values):
    plus = values['p00']+values['p11']-values['p01']-values['p10']
    minus = values['m00']+values['m11']-values['m01']-values['m10']
    j = (plus-minus)*.5
    return dict(I_plus=plus,I_minus=minus,J=j,mean_J=float(np.mean(j[SL])))

def gate(neural,yaw):
    need(np.isfinite(neural) and np.isfinite(yaw),'Nonfinite contrast')
    a,b = abs(neural)>=1.6e-5,abs(yaw)>=.02
    return dict(neural_material=bool(a),command_material=bool(b),screen_pass=bool(a and b),
                classification='PROMETEDOR_NO_CONFIRMADO' if a and b else 'DESCARTADO',
                stage4_pass=False,stage5_pass=False)

def check_trace(arm,t,o,owner,initial,prefix,spec):
    need(set(t)==set(prefix),'Trace field set differs')
    need(all(v.shape[0]==140 for v in t.values()),'Incomplete trace')
    for source in (t,o):
        for k,v in source.items():
            if v.dtype.kind in 'fc':
                need(np.isfinite(v).all(),'Nonfinite '+k)
    need(np.array_equal(t['paso'],np.arange(3001,3141)),'Interval counter')
    clocks=47486000000+np.arange(1,141,dtype=np.int64)*1000000
    for k in ('CNS_time_ns','PN_time_ns','body_time_ns'):
        need(np.array_equal(t[k],clocks),'Clock '+k)
    for k,v in prefix.items():
        need(t[k].dtype==v.dtype and np.array_equal(t[k][:10],v),'Shared49 prefix '+k)
    need(np.all(t['fase']=='cola_OFF'),'Legacy phase label changed')
    for k in ('sensores_usados','sensores_pendientes','concentracion_campo'):
        need(np.all(t[k]==0),'Legacy odor applied '+k)
    need(np.array_equal(o['DN_ids'],DN_IDS),'DN identities')
    need(o['DN_q'].shape==(140,6),'DN shape')
    need(np.array_equal(o['DN_q'][:,:4],t['DN_q_actual']),'DN observer alias')
    need(np.array_equal(o['ORN_ids'],spec['ids']),'ORN identities')
    need(np.array_equal(o['ORN_sides'],spec['sides']),'ORN sides')
    need(o['proprioception'].shape==(140,151),'Proprioceptor shape')
    sched=expected_schedule(arm)
    side=np.where(spec['sides']=='L',0,1)
    expected=spec['baseline'][None,:]+spec['delta'][None,:]*sched[:,side]
    need(np.array_equal(o['nominal_Hz'],expected),'Input target schedule')
    need(np.array_equal(sched.sum(axis=0),[60.,60.]),'Marginal dose')
    need(owner['schema']=='temporal50_external_owner_v1' and owner['arm']==arm,'Temporal owner')
    need(owner['origin_consumed_ms']==3000,'Temporal origin')
    need(owner['instantaneous_state_clamped'] is False,'Instantaneous clamp flag')
    need(owner['incoming_ORN_target_modulation_bypassed'] is True,'Intervention disclosure flag')
    need(np.array_equal(np.array(owner['nominal_schedule']),sched),'Owner schedule')
    previous=np.vstack([initial['previous_dn'],t['DN_q_actual'][:-1]])
    need(np.array_equal(t['DN_q_usada'],previous),'Motor latency')
    need(np.array_equal(t['DN_baseline'],np.broadcast_to(initial['baseline'],(140,4))),'Baseline recentered')
    delta=previous-initial['baseline']
    raw=np.array([float(np.mean(x[:2])) for x in delta])
    yaw=np.array([float(np.tanh(250.*(x[2]-x[3]))*math.radians(5.)) for x in delta])
    need(np.array_equal(t['forward_unclipped_mm_s'],raw),'Raw forward decoder')
    need(np.array_equal(t['command_forward_mm_s'],np.clip(raw,0.,.5)),'Forward decoder')
    need(np.array_equal(t['neural_yaw_unapplied_rad_s'],yaw),'Raw yaw decoder')
    need(np.all(t['command_yaw_rate_rad_s']==0),'Unexpected applied yaw')

def check_dng(d,trace=None):
    need(np.array_equal(d['ids'],DN_IDS[:2]),'DNg IDs')
    names=list(d['fields'])
    need(names==['state','net','positive_aux','negative_aux','drive','theta','gain','base_target','base_rate','tau','margin','final_target','final_rate','derivative','evaluation_time_s','stage_fraction'],'DNg field semantics')
    r=d['records']; n=len(d['trials'])
    need(r.shape==(int(d['trials'].sum()),140) and np.isfinite(r).all(),'DNg records')
    need(n==2240 and np.array_equal(d['offsets'],np.r_[0,np.cumsum(d['trials'])]),'DNg epoch ownership')
    need(np.array_equal(d['ms'],np.repeat(np.arange(3001,3141),16)),'DNg interval')
    need(np.array_equal(d['committed'],np.tile([False,True],1120)),'DNg committed flags')
    need(np.array_equal(d['accepted']+d['rejected'],d['trials']),'DNg trial counts')
    slots=np.tile(np.arange(16),140)
    expected=47486000000+np.repeat(np.arange(140),16)*1000000+(slots//2)*125000
    need(np.array_equal(d['start_ns'],expected),'DNg absolute clock')
    need(np.array_equal(d['duration_ns'],np.where(slots%2,125000,62500)),'DNg predictor duration')
    # All stored stages; target statistics below are not a physiology claim.
    f=r[:,:128].reshape(-1,4,2,16)
    margin=(f[...,1].astype(np.float32)+f[...,4].astype(np.float32))-f[...,5].astype(np.float32)
    need(np.array_equal(f[...,10],margin.astype(np.float64)),'DNg net-plus-drive-minus-threshold')
    need(np.all(f[...,11]>=0) and np.all(f[...,11]<=1),'DNg target range')
    need(np.array_equal(f[...,7],f[...,11]),'DNg target overridden')
    need(np.array_equal(f[...,8],f[...,12]),'DNg rate overridden')
    need(np.array_equal(f[...,13],f[...,12]*(f[...,11]-f[...,0])),'DNg RHS derivative')
    need(np.all(r[:,132:134]==0),'DNg scheduler warning flags')
    need(np.array_equal(r[:,138],(r[:,131]<=1).astype(np.float64)),'DNg acceptance decision')
    need(np.array_equal(d['epoch'],np.arange(n)),'DNg epoch sequence')
    for e in range(n):
        x=r[d['offsets'][e]:d['offsets'][e+1]]
        need(len(x)>0 and np.array_equal(x[:,139],np.arange(len(x))),'DNg trial sequence')
        ok=x[:,138].astype(bool)
        need(int(ok.sum())==int(d['accepted'][e]),'DNg accepted count')
        previous=0.; state=x[0,134:136]
        for row,accepted in zip(x,ok):
            need(row[128]==previous and np.array_equal(row[134:136],state),'DNg commit/reject continuity')
            if accepted:
                previous=row[130];state=row[136:138]
        need(previous==d['duration_ns'][e]*1e-9,'DNg epoch end')
        if trace is not None and e%16==15:
            need(np.array_equal(state,trace['DN_q_actual'][e//16,:2]),'DNg committed observation')
    return dict(target_min=float(f[...,11].min()),target_max=float(f[...,11].max()),
                margin_min=f[...,10].min(axis=(0,1)).tolist(),margin_max=f[...,10].max(axis=(0,1)).tolist(),
                records=len(r),scope='All captured DNg RHS stages, predictor and committed')

def compute(folder=HERE):
    started=time.process_time(); folder=Path(folder)
    need(sha(folder/'PLAN.json')==PLAN_HASH,'Changed frozen criteria')
    plan=read_json(folder/'PLAN.json')
    receipt=read_json(folder/'reference/RECEIPT.json')
    for name,digest in receipt['files'].items():
        need(sha(folder/'reference'/name)==digest,'Changed reference '+name)
    need(receipt['source_manifest_sha256']==plan['source_manifest_sha256'],'Initial state source')
    queue=read_json(folder/'QUEUE_RESULT.json')
    need(queue['status']=='COMPLETE' and queue['total_neural_ms']==1120,'Incomplete factorial experiment')
    need([r['arm'] for r in queue['arms']]==list(ARMS),'Queue arm set/order')
    spec=read_npz(folder/'reference/input_spec.npz')
    initial=read_npz(folder/'reference/initial.npz'); prefix=read_npz(folder/'reference/prefix49.npz')
    data={}; summaries={}; sources=None
    for arm in ARMS:
        p=folder/arm; r=read_json(p/'RESULT.json')
        need(r['arm']==arm and r['status']=='COMPLETE' and r['committed_ms']==r['attempted_ms']==140,'Incomplete arm '+arm)
        need(r['plan_sha256']==PLAN_HASH,'Arm contract '+arm)
        current=read_json(p/'EXECUTED_SOURCES.json')
        if sources is None:
            sources=current
        need(current==sources,'Unequal execution source sets')
        t=read_npz(p/'traces.npz'); o=read_npz(p/'input_and_observers.npz')
        check_trace(arm,t,o,read_json(p/'TEMPORAL_OWNER.json'),initial,prefix,spec)
        dn=check_dng(read_npz(p/'dng100_observed.npz'),t)
        data[arm]=(t,o)
        summaries[arm]=dict(DNb05_L_minus_R_mean=float((o['DN_q'][SL,2]-o['DN_q'][SL,3]).mean()),
            raw_yaw_mean_deg_s=float(np.rad2deg(t['neural_yaw_unapplied_rad_s'][SL]).mean()),
            forward_max_mm_s=float(t['command_forward_mm_s'].max()),DNg100=dn,
            CPU_s=r['CPU_s'],wall_s=r['wall_s'])
    series={}
    choices={'DNb05':lambda t,o:o['DN_q'][:,2]-o['DN_q'][:,3],
             'raw_yaw_deg_s':lambda t,o:np.rad2deg(t['neural_yaw_unapplied_rad_s']),
             'DNa02':lambda t,o:o['DN_q'][:,4]-o['DN_q'][:,5],
             'PN_10208':lambda t,o:t['PN_q_legacy'][:,0],
             'PN_10176':lambda t,o:t['PN_q_legacy'][:,1]}
    for name,fn in choices.items():
        c=contrast({a:fn(*data[a]) for a in ARMS})
        series[name]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()}
    bodykeys=('qpos','qvel','position_mm','yaw_delta_deg','contact_active','contact_force_N','generalized_force_native','energy_motor_J')
    body={k:all(np.array_equal(data[a][0][k],data['p00'][0][k]) for a in ARMS) for k in bodykeys}
    prop=all(np.array_equal(data[a][1]['proprioception'],data['p00'][1]['proprioception']) for a in ARMS)
    decision=gate(series['DNb05']['mean_J'],series['raw_yaw_deg_s']['mean_J'])
    out=dict(schema='temporal50_analysis_v1',plan_sha256=PLAN_HASH,arms=summaries,contrasts=series,
             decision=decision,body_equal=body,consumed_proprioception_equal=prop,
             shared_prefix_fields=len(prefix),shared_prefix_ms=10,CPU_simulation_s=sum(summaries[a]['CPU_s'] for a in ARMS),
             neural_ms=1120,analysis_CPU_s=time.process_time()-started,
             PN_legacy_ids=[10208,10176],PN_scope='Normalized legacy proxy q of two IDs, not all PNs or calcium',
             interpretation='Lag-sensitive bilateral interaction in this prepared state/epoch; not an identified motion detector',
             caveat='A time-varying instantaneous joint map can also produce J; one exposed preparation, no seed inference; target consumption checked online by frozen runtime, not independently replayed for every ORN RHS from this capsule')
    return out

def render(v):
    d=v['decision']; c=v['contrasts']; yes=lambda x:'sí' if x else 'no'
    lines=['# Campaña 50: contraste temporal bilateral','',
        f"**{d['classification']}** para la criba temporal fijada. Etapas 4/5 abiertas.",'',
        f"Completados ocho brazos de 140 ms: {v['neural_ms']} ms nuevos de CNS; {v['CPU_simulation_s']:.3f} s CPU de simulación.",'',
        '| Observable | J medio, muestras 11–130 | Umbral previo | Material |',
        '|---|---:|---:|---|',
        f"| DNb05 L−R, antes del lector | {c['DNb05']['mean_J']:.12g} | 0,000016 | {yes(d['neural_material'])} |",
        f"| Giro crudo sin aplicar, °/s | {c['raw_yaw_deg_s']['mean_J']:.12g} | 0,02 | {yes(d['command_material'])} |",'',
        'J=(I(+40 ms)−I(−40 ms))/2; I=X00+X11−X01−X10. Cada antena recibe 60 ms de nivel alto y 60 ms basal dentro de la ventana. No se seleccionaron máximos ni otra ventana para el veredicto.','',
        '| Brazo | DNb05 L−R medio | Giro crudo medio, °/s | Avance máximo, mm/s | Objetivo DNg100 máximo |',
        '|---|---:|---:|---:|---:|']
    for a,r in v['arms'].items():
        lines.append(f"| {a} | {r['DNb05_L_minus_R_mean']:.10g} | {r['raw_yaw_mean_deg_s']:.10g} | {r['forward_max_mm_s']:.10g} | {r['DNg100']['target_max']:.10g} |")
    lines += ['',f"Prefijo común: {v['shared_prefix_fields']} campos idénticos durante {v['shared_prefix_ms']} ms contra49. Cuerpo idéntico en todos los campos examinados: {yes(all(v['body_equal'].values()))}. Propiocepción consumida idéntica: {yes(v['consumed_proprioception_equal'])}.",'',
        'Secundarios conservados: DNa02 y q de PN10208/PN10176. Esta pareja no representa la población PN completa, la salida multicompartmental de PN10208 ni una medición de calcio.','',
        'El contraste anula respuestas separables de las antenas y, en esta ventana cíclica, un mapa conjunto instantáneo estacionario. Un mapa conjunto instantáneo que varíe con el tiempo también puede dar J distinto de cero. No identifica il3LN6 ni demuestra un detector biológico de movimiento.','',
        'El lector de giro se registró sin aplicarlo y no hubo viento. Ningún resultado de esta criba aprueba navegación, recuperación ni equivalencia neurobiológica. Una preparación expuesta, sin inferencia de variación entre semillas.','',
        'La cápsula reconstruye el análisis de las ocho trazas y todos los registros DNg100. La comprobación de consumo ORN se ejecutó en cada RHS mediante el código congelado; no se conserva aquí el tensor completo de cada RHS de las 694 ORN. Los checkpoints íntegros se conservan localmente; la reejecución portable de todo el CNS es un alcance distinto.','']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=HERE);p.add_argument('--write',action='store_true');a=p.parse_args()
    v=compute(a.folder)
    if a.write:
        need(not (a.folder/'RESULTADOS.json').exists(),'Preserve original analysis')
        save(a.folder/'RESULTADOS.json',v);(a.folder/'RESULTADOS.md').write_text(render(v))
    print(json.dumps(dict(decision=v['decision'],J_neural=v['contrasts']['DNb05']['mean_J'],J_yaw_deg_s=v['contrasts']['raw_yaw_deg_s']['mean_J'],analysis_CPU_s=v['analysis_CPU_s']),allow_nan=False))

if __name__=='__main__':
    main()
