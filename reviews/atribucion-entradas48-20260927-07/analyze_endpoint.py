"""Portable direct-input accounting; no CNS, body or runtime imports."""
import hashlib,json,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ARMS=['sham','dm1','profile','permuted']
IDS=[10045,10056,10118,10065,523769,10360]
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def warp_sum(x):
    x=np.asarray(x,dtype=np.float32);lanes=np.zeros(32,np.float32)
    for i in range(0,len(x),32):lanes[:len(x[i:i+32])]+=x[i:i+32]
    for d in [16,8,4,2,1]:
        old=lanes.copy();lanes[:32-d]=old[:32-d]+old[d:]
    return float(lanes[0])
def groups(values,labels):
    out=[dict(type=str(t),value=float(values[labels==t].sum()),edges=int((labels==t).sum())) for t in np.unique(labels)]
    return sorted(out,key=lambda x:(-abs(x['value']),x['type']))
def calculate():
    plan=json.loads((HERE/'PLAN.json').read_text());e=json.loads((HERE/'EXTRACTION.json').read_text())
    need(sha(HERE/'PLAN.json')==e['plan_sha256'],'Changed context/plan')
    need(sha(HERE/'endpoint_inputs.npz')==e['inputs_sha256'],'Changed inputs')
    need(plan['arms']==ARMS and [i for t in plan['targets'] for i in t['ids']]==IDS,'Selection changed')
    need(plan['new_neural_steps_max']==0 and plan['GPU_runs_max']==0,'Budget/scope changed')
    out={}; contrasts={}
    with np.load(HERE/'endpoint_inputs.npz',allow_pickle=False) as z:
        need(z['target_ids'].tolist()==IDS,'Input identities changed')
        need(z['ptr'].shape==(7,) and z['ptr'][0]==0 and np.all(np.diff(z['ptr'])>0),'Row partition')
        caps=z['caps'];w0=z['sham_cuda_weights'];s0=z['sham_transmission']
        need(np.all(caps>0),'Caps positive')
        for key in z.files:
            a=z[key]
            if a.dtype.kind in 'fc':need(np.isfinite(a).all(),'Nonfinite '+key)
        for arm in ARMS:
            m=e['metadata'][arm]
            need(m['time_ns']==47486000000 and m['state_layout']['transmission_start']==177758,'Boundary/context changed')
            need(m['visual_output_connected'] and not m['general_PN_enabled'],'Mask/PN route changed')
            need(m['csr_verified_against_checkpoint'] and m['publication_equal_release_times_caps'],'Extraction witness absent')
            need(m['APL_edges_per_target']==[0]*6 and not z[arm+'_APL_presynaptic'].any(),'APL routing needs explicit handling')
            need(all(not v.get('target_rows',[]) and not v.get('rows',[]) for v in m['specialized_target_membership'].values()),'Specialized row override')
            for p in ['weights','tau','theta','gain']:
                need(np.array_equal(z[arm+'_'+p],z[arm+'_cuda_'+p]),'Host/device operands differ '+p)
            w=z[arm+'_cuda_weights'];s=z[arm+'_transmission'];x=s*caps;x0=s0*caps
            term=w*x
            # Symmetric bilinear decomposition includes a separate weight term,
            # including when that term is exactly zero for all saved arms.
            state_component=(w+w0)*.5*(x-x0)
            weight_component=(x+x0)*.5*(w-w0)
            residue=(term-w0*x0)-(state_component+weight_component)
            need(np.max(np.abs(residue))<1e-9,'Bilinear identity')
            rows=[];diffs=[]
            for k,ident in enumerate(IDS):
                sl=slice(*z['ptr'][k:k+2]);t=term[sl];ty=z['pre_types'][sl]
                t32=w[sl].astype(np.float32)*(s[sl].astype(np.float32)*caps[sl].astype(np.float32))
                net32=warp_sum(t32)
                instant=w[sl]*(z[arm+'_release'][sl]*caps[sl])
                orn=z[arm+'_prescribed_ORN'][sl]
                row=dict(id=ident,type=str(z['target_types'][k]),side=str(z['target_sides'][k]),edges=len(t),
                    positive=float(t[t>0].sum()),negative=float(t[t<0].sum()),net=float(t.sum()),net_fp32=net32,
                    normalized_q=float(z[arm+'_target_q'][k]),theta=float(z[arm+'_cuda_theta'][k]),
                    margin_if_drive_zero=float(t.sum()-z[arm+'_cuda_theta'][k]),
                    gain=float(z[arm+'_cuda_gain'][k]),tau_s=float(z[arm+'_cuda_tau'][k]),
                    absolute_flow=float(np.abs(t).sum()),net_using_instantaneous_release=float(instant.sum()),
                    instantaneous_minus_filtered_net=float(instant.sum()-t.sum()),
                    prescribed_ORN_direct_edges=int(orn.sum()),prescribed_ORN_direct_input=float(t[orn].sum()),
                    groups=groups(t,ty))
                if k<2:
                    rhs=z[arm+'_last_RHS'][k]
                    row['last_recorded_RHS_net']=float(rhs[1])
                    row['endpoint_minus_last_recorded_RHS']=net32-float(rhs[1])
                    row['last_recorded_drive']=float(rhs[4])
                    row['last_recorded_target']=float(rhs[7])
                    names=e['last_RHS_witness'][arm]['fields']
                    need(names[10]=='margin' and names[13]=='derivative','RHS field semantics')
                    row['last_recorded_margin']=float(rhs[names.index('margin')])
                    row['last_recorded_derivative']=float(rhs[names.index('derivative')])
                    margin32=(np.float32(rhs[1])+np.float32(rhs[4]))-np.float32(rhs[5])
                    need(float(margin32)==row['last_recorded_margin'],'Recorded margin identity')
                rows.append(row)
                diffs.append(dict(id=ident,total_net_change=float(t.sum()-(w0*x0)[sl].sum()),
                    signal_term=float(state_component[sl].sum()),weight_term=float(weight_component[sl].sum()),
                    signal_term_prescribed_ORN=float(state_component[sl][orn].sum()),
                    signal_term_remaining_sources=float(state_component[sl][~orn].sum()),
                    theta_change=float(z[arm+'_theta'][k]-z['sham_theta'][k]),
                    signal_groups=groups(state_component[sl],ty),operator_groups=groups(weight_component[sl],ty)))
            out[arm]=rows;contrasts[arm]=diffs
        shared=all(np.array_equal(z[a+'_cuda_weights'],w0) for a in ARMS)
        need(shared,'Different weights require another bound')
        for a in ARMS:
            need(np.array_equal(z[a+'_theta'],z['sham_theta']),'Different theta')
        signals=np.stack([z[a+'_transmission']*caps for a in ARMS])
        lo,hi=signals.min(axis=0),signals.max(axis=0)
        best=w0*np.where(w0>=0,hi,lo);worst=w0*np.where(w0>=0,lo,hi)
        bounds=[]
        for k,ident in enumerate(IDS):
            sl=slice(*z['ptr'][k:k+2]);upper=float(best[sl].sum());lower=float(worst[sl].sum())
            theta=float(z['sham_theta'][k])
            need(all(lower-1e-9<=out[a][k]['net']<=upper+1e-9 for a in ARMS),'Bound containment')
            bounds.append(dict(id=ident,lower_net=lower,upper_net=upper,theta=theta,
                upper_margin_if_drive_zero=upper-theta,
                entire_box_subthreshold_if_drive_zero=bool(upper-theta<0),
                physiological_range=False,dynamic_reachability_proven=False))
    return dict(classification='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA',
        scope='Arithmetic of six direct input rows at four final checkpoints. Not a recurrent causal attribution or historical whole-RHS replay.',
        rows=out,contrasts_to_sham=contrasts,weights_equal_across_arms=shared,observed_endpoint_box=bounds,
        stage4_pass=False,stage5_pass=False,neural_steps=0,causal_mechanism_identified=False,
        bilinear_identity_tolerance_absolute=1e-9,
        limitation='No held midpoint/physical event trajectory; no biological units calibration. Numerical proximity to a last recorded row does not recover its missing operands.')
def report(r):
    lines=['# Atribución de entradas — campaña 48','',r['scope'],'',
        'Cálculo CPU desde cuatro estados finales de 3000 ms. No se avanzó el cerebro. Las etapas 4/5 continúan abiertas.',
        '', '| Condición | DNg100 L: entrada + | Entrada − | Saldo | DNg100 R: saldo |','|---|---:|---:|---:|---:|']
    for a in ARMS:
        l,rr=r['rows'][a][:2];lines.append(f"| {a} | {l['positive']:.3f} | {l['negative']:.3f} | {l['net']:.3f} | {rr['net']:.3f} |")
    lines+=['','Unidades internas del modelo. La suma no es una medición de corriente biológica. Los pesos guardados de estas filas son idénticos entre brazos; sus diferencias provienen algebraicamente de la transmisión presináptica. Esto no identifica qué mecanismo generó ese estado.',
        '', '| Destino | q izquierdo sham → profile | q derecho sham → profile |','|---|---:|---:|']
    for k in [0,2,4]:
        s=r['rows']['sham'];p=r['rows']['profile'];lines.append(f"| {s[k]['type']} | {s[k]['normalized_q']:.6g} → {p[k]['normalized_q']:.6g} | {s[k+1]['normalized_q']:.6g} → {p[k+1]['normalized_q']:.6g} |")
    lines+=['','Hay actividad de DNb05, sin evidencia de orientación útil en este contraste. DNa02 sigue siendo un observador y no sustituye el lector.',
        '', 'Las seis filas suman '+str(sum(x['edges'] for x in r['rows']['sham']))+' aristas. Aristas entrantes desde ORN prescritas, en orden DNg100 L/R, DNb05 L/R y DNa02 L/R: '+str([x['prescribed_ORN_direct_edges'] for x in r['rows']['sham']])+'. Ninguna recibe APL directamente; se conservaron los controles de rutas especializadas y salida PN general desactivada.',
        '', 'Los mayores aportes inhibidores por tipo hacia DNg100 incluyen GNG127 y GNG031. Ya aparecían en el preparado del 26-09; no son un mecanismo nuevo descubierto aquí. Su tamaño no autoriza a retirarlos: una lesión por ranking confundiría aporte actual con causalidad y podría compensar una calibración incorrecta.',
        '', 'Control algebraico adicional: se permite combinar independientemente los mínimos/máximos de cada entrada observados en los cuatro estados finales, conservando W/caps. Es una caja artificial, más permisiva que elegir un brazo real; no es una trayectoria ni un rango fisiológico. Para DNg100, los máximos márgenes con drive=0 son '+str([round(x['upper_margin_if_drive_zero'],6) for x in r['observed_endpoint_box'][:2]])+'. Resultado por fila: '+str([x['entire_box_subthreshold_if_drive_zero'] for x in r['observed_endpoint_box'][:2]])+' (True significa que toda esta caja queda bajo umbral). No excluye estados fuera de estos rangos ni descarta el proyecto.',
        '', 'En RESULTADOS.json están todas las agrupaciones, diferencias, sumas FP32 y comparación separada con el último RHS registrado. No se declara reproducción del RHS completo ni continuación cualificada del checkpoint.',
        '', 'Verificación: `python3 verify_capsule.py` y `python3 -O verify_capsule.py`. Recalcular: `python3 analyze_endpoint.py`. La extracción completa usa `extract_endpoint.py` y requiere los originales fijados por hashes en EXTRACTION.json; esos estados completos no forman parte de esta cápsula.']
    return '\n'.join(lines)+'\n'
if __name__=='__main__':
    t=time.process_time();r=calculate();write(HERE/'RESULTADOS.json',r)
    (HERE/'RESULTADOS.md').write_text(report(r))
    print(json.dumps({'scope':r['scope'],'same_weights':r['weights_equal_across_arms'],'CPU_s':time.process_time()-t,'DNg100_FP32_vs_last_RHS':{a:[x['endpoint_minus_last_recorded_RHS'] for x in r['rows'][a][:2]] for a in ARMS}}))
