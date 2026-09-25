"""Criba de trazas 40+1000 ms: control L/R exacto vs media bilateral.
No valida checkpoints, todos los estados, error de discretizacion ni feedback.
El ON instala usados[40]; el retardo ordinario se verifica desde usados[41].
"""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
N, P = 1040, 40

def exigir(ok, msg):
    if not ok: raise ValueError(msg)

def exacto(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()

def cargar(path):
    with np.load(path, allow_pickle=False) as z: d = {k:z[k].copy() for k in z.files}
    exigir(d['fase'].tolist() == ['preparacion']*P+['ensayo']*1000, 'Fases')
    exigir(d['paso'].dtype.kind in 'iu' and d['paso'].tolist() == list(range(1,41))+list(range(1,1001)), 'Pasos')
    for k, a in d.items():
        if a.dtype.kind in 'fiu': exigir(np.isfinite(a).all(), 'No finito: '+k)
    for k in ('CNS_time_ns','PN_time_ns','body_time_ns'):
        a=d[k]; exigir(a.shape==(N,) and a.dtype.kind in 'iu', 'Reloj '+k)
        exigir(np.array_equal(a,d['CNS_time_ns']) and np.all(np.diff(a)==1000000), 'Cronologia '+k)
    for k in ('sensores_usados','sensores_pendientes','position_mm'):
        exigir(d[k].shape==(N,3) and d[k].dtype==np.float64, 'Layout '+k)
    exigir(np.all(d['sensores_usados'][:P]==0), 'Olor durante preparacion')
    exigir(exacto(d['sensores_usados'][P+1:],d['sensores_pendientes'][P:-1]), 'Retardo de consumo')
    q=d['qpos']; exigir(q.dtype==np.float64 and q.ndim==2 and q.shape[0]==N and q.shape[1]>=7, 'qpos')
    exigir(np.max(abs(np.sum(q[:,3:7]**2,axis=1)-1))<1e-8, 'Cuaternion')
    for k in ('command_forward_mm_s','command_yaw_rate_rad_s','yaw_delta_deg','upright'):
        exigir(d[k].shape==(N,), 'Layout '+k)
    exigir(d['contact_active'].ndim==2 and d['contact_active'].shape[0]==N, 'Contactos')
    exigir(np.all(d['command_forward_mm_s'][P:]==.2), 'Avance modificado')
    q,b=d['DN_q_usada'],d['DN_baseline']
    exigir(q.shape==b.shape and q.ndim==2 and q.shape[0]==N and q.shape[1]>=4, 'DN')
    esperado=np.tanh(250*((q[P:,2]-b[P:,2])-(q[P:,3]-b[P:,3])))*np.deg2rad(5)
    exigir(np.max(abs(esperado-d['command_yaw_rate_rad_s'][P:]))<=1e-12, 'Lector modificado')
    return d

def sin_contraste(a):
    b=a.copy(); media=(a[P:,0]+a[P:,1])*.5
    b[P:,0]=media; b[P:,1]=media
    return b

def metricas(d, fuente):
    q=d['qpos']; w,x,y,z=q[:,3:7].T
    psi=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    alfa=np.arctan2(fuente[1]-10*q[:,1],fuente[0]-10*q[:,0])
    error=abs(np.rad2deg(np.arctan2(np.sin(alfa-psi),np.cos(alfa-psi))))
    distancia=np.linalg.norm(d['position_mm'][:,:2]-fuente,axis=1)
    u=np.rad2deg(d['command_yaw_rate_rad_s'][P:]); l1=float(abs(u).sum()*.001)
    dn=d['DN_q_usada'][P:]-d['DN_baseline'][P:]; l,r=dn[:,2],dn[:,3]
    denominador=float((abs(l)+abs(r)).sum())
    return dict(bearing_deg=[float(error[k]) for k in (39,439,1039)],
        bearing_integral_deg_s=float(.0005*np.sum(error[39:-1]+error[40:])),
        progreso_mm=float(distancia[39]-distancia[-1]),
        command_integral_deg=float(u.sum()*.001),command_L1_deg=l1,
        cancelacion_temporal=None if l1==0 else float(1-abs(u.sum()*.001)/l1),
        cancelacion_entrada_DN=None if denominador==0 else float(1-abs(l-r).sum()/denominador),
        mejora_historica_0p5=bool(error[39]-error[-1]>=.5 and error[439]-error[-1]>=.5),
        apoyo=bool(d['upright'][P:].min()>=.95 and d['contact_active'][P:].sum(axis=1).min()>=4))

def verificar(donante, control, sin_d, campos):
    d,c,a=[cargar(p) for p in (donante,control,sin_d)]
    for k in ('CNS_time_ns','PN_time_ns','body_time_ns'):
        exigir(exacto(d[k],c[k]) and exacto(d[k],a[k]), 'Origen temporal '+k)
    claves=('qpos','position_mm','yaw_delta_deg','command_yaw_rate_rad_s',
            'command_forward_mm_s','DN_q_usada','DN_baseline','sensores_usados',
            'sensores_pendientes','upright','contact_active')
    for k in claves:
        exigir(exacto(d[k],c[k]), 'Replay identidad fallo: '+k)
        exigir(exacto(d[k][:P],a[k][:P]), 'Preparacion en traza difiere: '+k)
    exigir(exacto(d['DN_baseline'],a['DN_baseline']), 'Baseline DN distinto')
    for k in ('sensores_usados','sensores_pendientes'):
        exigir(exacto(sin_contraste(d[k]),a[k]), 'Cinta sinD incorrecta: '+k)
    fuente=np.asarray(json.loads(Path(campos).read_text())['minus']['source_mm'],float)
    exigir(fuente.shape==(2,) and np.isfinite(fuente).all(), 'Fuente derecha')
    mc,ma=metricas(c,fuente),metricas(a,fuente)
    return dict(status='COMPARACION_DESCRIPTIVA_NO_ADMISION',control=mc,sinD=ma,
        efecto_D_bearing_deg=ma['bearing_deg'][-1]-mc['bearing_deg'][-1],
        efecto_D_integral_deg_s=ma['bearing_integral_deg_s']-mc['bearing_integral_deg_s'],
        efecto_D_progreso_mm=mc['progreso_mm']-ma['progreso_mm'],
        estado_preparado_completo_verificado=False,error_numerico_1s_verificado=False,
        stage4_admitted=False,stage5_admitted=False,
        sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (donante,control,sin_d,campos)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('donante','control','sinD','campos','out'): p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); exigir(not a.out.exists(), 'Salida existente')
    try: r=verificar(a.donante,a.control,a.sinD,a.campos)
    except Exception as e: r=dict(status='FALLO_RETENIDO',error=repr(e))
    r['codigo_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x',encoding='utf-8') as f: json.dump(r,f,indent=2,allow_nan=False); f.write('\n')
    print(json.dumps(r,indent=2,allow_nan=False))
    raise SystemExit(2 if r['status']=='FALLO_RETENIDO' else 0)
