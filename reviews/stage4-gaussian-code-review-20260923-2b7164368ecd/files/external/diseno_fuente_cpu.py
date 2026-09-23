"""Diseño geométrico usando dos JSON reales. No ejecuta cuerpo, CNS o campos vivos.
La gaussiana coincide con AntennalBoundary.sample; sólo se evalúa su fórmula.
El replay propuesto usa un DONANTE REAL futuro, nunca las curvas proxy de este script.
"""
import argparse, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
HASHES = {
    'geometry':'3214fd7600d6ad2ec9daca0484eb2fa3d895b9afc8e1321f9d379ec9521276f3',
    'context':'486d939c2f3bc55489763d9f1c5062cd31ac8c2a52a1524e547d235fcb8ccb39'
}
PLAN = {
    'scope':'Diseño propuesto; no prerregistro de ejecución del organismo',
    'source_longitudinal_in_baselines':0.5,
    'source_lateral_in_baselines':2.0,
    'sigma_in_baselines':2.0,
    'arrival_radius_in_baselines':0.5,
    'material_distance_in_baselines':0.05,
    'proposed_distance_error_fraction':0.1,
    'pilot_horizon_s':4.0,
    'readouts_s':[0.4,4.0,8.0],
    'cost_basis_min_per_0_4s':{'native':23.0,'reference':32.0},
    'organism_runs_executed':0
}
def need(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def read(p,expected):
    need(sha(p)==expected,'SHA distinto de la entrada publicada: '+str(p))
    return json.loads(p.read_text(encoding='utf-8'))
def concentrations(points,source,sigma):
    # Misma fórmula y unidades mm del AntennalBoundary publicado.
    return np.exp(-np.sum((points[...,:2]-source)**2,axis=-1)/(2.*sigma**2))
def run(geometry,context):
    g=read(geometry,HASHES['geometry']);ctx=read(context,HASHES['context'])
    a=np.asarray(g['prepared_antennae_mm'],dtype=float)
    q=np.asarray(g['prepared_qpos_root'],dtype=float)
    need(a.shape==(2,3) and q.shape==(7,) and np.isfinite(a).all() and np.isfinite(q).all(),'Geometría inválida')
    scale=float(g['body_native_units_per_mm']);v=float(g['forward_command_mm_s'])
    need(scale>0 and v>0,'Escala/velocidad inválida')
    body=q[:2]/scale;mid=a[:,:2].mean(axis=0)
    baseline=float(np.linalg.norm(a[0,:2]-a[1,:2]));need(baseline>0,'Antenas coincidentes')
    lateral=(a[0,:2]-a[1,:2])/baseline
    forward=np.array([lateral[1],-lateral[0]])
    w,x,y,z=q[3:];need(abs(w*w+x*x+y*y+z*z-1)<1e-8,'Quaternion inválido')
    heading=np.array([1-2*(y*y+z*z),2*(w*z+x*y)])
    need(np.linalg.norm(heading)>0,'Rumbo horizontal degenerado')
    if forward@heading<0:forward=-forward
    sigma=PLAN['sigma_in_baselines']*baseline
    radius=PLAN['arrival_radius_in_baselines']*baseline
    effect=PLAN['material_distance_in_baselines']*baseline
    error=effect*PLAN['proposed_distance_error_fraction']
    offset=PLAN['source_longitudinal_in_baselines']*baseline
    side=PLAN['source_lateral_in_baselines']*baseline
    sources={'donor':mid+offset*forward+side*lateral,
             'transfer':mid-offset*forward+side*lateral}
    initial={k:concentrations(a,s,sigma) for k,s in sources.items()}
    initial_diff=float(np.max(abs(initial['donor']-initial['transfer'])))
    need(initial_diff<1e-12,'La transferencia no conserva ambos estímulos iniciales')
    specs={};rows=[]
    for name,source in sources.items():
        delta=source-body;along=float(delta@forward)
        lateral_offset=abs(float(delta@lateral))
        d0=float(np.linalg.norm(delta))
        raydistance=lateral_offset if along>=0 else d0
        need(d0>radius and raydistance>radius,'Fuente en la región inicial o recta tónica')
        c0=initial[name]
        specs[name]={'source_mm':source.tolist(),'sigma_mm':sigma,
            'initial_concentration_L_R':c0.tolist(),'initial_drive_80c_L_R':(80*c0).tolist(),
            'body_initial_distance_mm':d0,'distance_to_forward_ray_mm':raydistance,
            'arrival_radius_mm':radius,
            'conditional_arrival_lower_bound_s':(d0-radius)/v}
        for t in PLAN['readouts_s']:
            # Traslación geométrica hipotética: NO trayectoria predicha del cuerpo.
            da=v*t*forward;pts=a.copy();pts[:,:2]+=da
            straight_distance=float(np.linalg.norm(source-(body+da)))
            rows.append({'source':name,'seconds':t,'hypothetical_straight_distance_mm':straight_distance,
                'hypothetical_straight_progress_mm':d0-straight_distance,
                'hypothetical_concentration_L_R':concentrations(pts,source,sigma).tolist()})
    trials=ctx['derived_from_existing_native_trials']
    need(set(trials)=={'sham','odor_left','odor_right','uniform'},'Faltan brazos descriptivos')
    displacement={k:np.asarray(r['translation_mm'],dtype=float)[:2] for k,r in trials.items()}
    old_bounds={k:float(np.linalg.norm(d-displacement['sham'])) for k,d in displacement.items()}
    costs={str(t):{k:minutes*t/.4 for k,minutes in PLAN['cost_basis_min_per_0_4s'].items()} for t in PLAN['readouts_s']}
    T=PLAN['pilot_horizon_s'];donor=sources['donor'];transfer=sources['transfer']
    # Demuestra que la transferencia discrimina el campo sin cambiar el primer olor.
    points=a.copy();points[:,:2]+=v*.4*forward
    cd=concentrations(points,donor,sigma);ct=concentrations(points,transfer,sigma)
    need(np.max(abs(cd-ct))>0,'Transferencia degenerada')
    return {
      'status':'CPU_GEOMETRY_COMPLETE','plan':PLAN,'inputs_sha256':HASHES,
      'baseline_mm':baseline,'forward_axis':forward.tolist(),'left_axis':lateral.tolist(),
      'antennal_midpoint_mm':mid.tolist(),'prepared_body_mm':body.tolist(),
      'sources':specs,'equal_initial_exposure_max_abs':initial_diff,
      'uniform_concentration':float(initial['donor'].mean()),'sham_concentration':0.,
      'material_advantage_mm':effect,'proposed_pair_distance_error_mm':error,
      'proposed_advantage_with_two_error_reserves_mm':effect+2*error,
      'old_400ms_distance_contrast_upper_bounds_mm':old_bounds,
      'old_400ms_bound_scope':'Desigualdad triangular sobre endpoints de ensayos laterales antiguos; no predicción gaussiana.',
      'geometric_straight_proxies':rows,
      'transfer_exposure_gap_after_hypothetical_0_4s':(cd-ct).tolist(),
      'budget_projection_min_per_arm':costs,
      'five_arm_native_projection_hours':5*costs[str(T)]['native']/60,
      'primary_metric':'P=d_body(0)-d_body(T). Donor: ventaja sobre sham/uniform. Transfer: ventaja sobre sham/uniform/replay-donor.',
      'reuse_controls':'Sólo si fuente no tiene salida distinta del olor; sham/uniform tienen misma señal en ambas geometrías.',
      'future_replay':'Secuencia sensorial usada por el donante real, aplicada en geometría transfer; nunca estas curvas proxy.',
      'stage3_admission':False,'stage4_admission':False,
      'limits':'Sin simulación; cota de llegada condicional a rapidez planar<=0.2mm/s, no una cota del cuerpo. Coste extrapolado, no medido. Sin viento, recompensa, reescalado neuronal ni calibración fisiológica.'
    }
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);save(a.out/'PLAN.json',PLAN)
    wall=time.perf_counter();cpu=time.process_time()
    try:result=run(a.geometry,a.context)
    except Exception:result={'status':'FAILED_RETAINED','error':traceback.format_exc()}
    result.update(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,code_sha256=sha(Path(__file__)))
    save(a.out/'RESULTADO.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))
    return 0 if result['status']=='CPU_GEOMETRY_COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
