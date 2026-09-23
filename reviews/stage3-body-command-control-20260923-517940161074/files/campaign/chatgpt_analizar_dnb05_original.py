"""Descomposición offline de cuatro brazos del snapshot 50b9fd3e.
No simula cuerpo/CNS. Rebasar el lector abajo es SOLO cálculo contrafactual.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse, csv, hashlib, json, time, traceback, zipfile
from pathlib import Path
import numpy as np

ARMS=('sham','odor_left','odor_right','uniform')
DN_IDS=[10118,10065]
FLOW_IDS=[10176,10208,10360,523769]  # PN R,L; DNa02 R,L.
VENTANAS=(250,275,300,320,350,400)
def exigir(ok,mensaje):
    if not ok: raise ValueError(mensaje)
def guardar(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def maxabs(x): return float(np.max(np.abs(x),initial=0.))
def decodificar(q,base):
    d=q-base
    return np.tanh(250*(d[:,2]-d[:,3]))*np.deg2rad(5)
def yaw(qpos):
    w,x,y,z=qpos[:,3:7].T
    exigir(maxabs(w*w+x*x+y*y+z*z-1)<1e-8,'Quaternion no unitario')
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
def csvout(p,filas):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]));w.writeheader();w.writerows(filas)

def analizar(root,out,manifest=None):
    out.mkdir(exist_ok=False);tic=time.perf_counter();hashes={};data={};checks={};common=None
    expected={} if manifest is None else {r['path']:r for r in json.loads(manifest.read_text())['files']}
    def entrada(rel):
        p=root/rel; exigir(p.is_file(),'Falta '+str(p));digest=hashfile(p);hashes[rel]=digest
        if manifest is not None:
            record=expected.get('campaign/'+rel)
            exigir(record is not None and record['sha256']==digest and record['bytes']==p.stat().st_size,'Hash/bytes distintos: '+rel)
        return p
    def js(rel): return json.loads(entrada(rel).read_text(encoding='utf-8'))
    def arrays(rel):
        p=entrada(rel)
        with zipfile.ZipFile(p) as z:
            exigir(sum(x.file_size for x in z.infolist())<=256*1024**2,'NPZ mayor de 256 MiB')
        with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
    plan={'scope':'Cuatro brazos 40+400 ms; no adaptación ni simulación','windows_ms':VENTANAS,
          'decoder':'tanh(250*((qL-bL)-(qR-bR)))*deg2rad(5)','lag_expected_ms':1,
          'counterfactual':'Rebasado al ON SOLO offline; no predice otro cuerpo',
          'yaw_readback_atol_deg':1e-10,'limits':'No certifica error biológico ni numérico del organismo'}
    guardar(out/'PLAN_ANALISIS.json',plan)
    try:
        for arm in ARMS:
            folder='full_'+arm+'_01/'
            receipt=js(folder+'RESULT.json');contract=js(folder+'RUN_CONTRACT.json')
            inter=js(folder+'preparation_inputs/INTERVENCIONES.json');frozen=js(folder+'FROZEN.json')
            exigir(receipt['status']=='COMPLETE' and receipt['error'] is None and not receipt['cleanup_errors'],'Corrida incompleta')
            exigir(receipt['odor']==contract['odor']==arm,'Brazo incorrecto')
            exigir(receipt['completed_preparation_ms']==40 and receipt['completed_trial_ms']==400,'Horizonte incompleto')
            exigir(contract['preparation_ms']==40 and contract['trial_ms']==400 and contract['event_boundaries'] is True,'Contrato distinto')
            exigir(inter['dn_ids_lector'][2:]==DN_IDS and len(inter['dn_ids_lector'])==4,'IDs del lector')
            identity={k:contract[k] for k in ('engine','checkpoint_manifest_sha256','plan_sha256','event_representation')}
            firma=(identity,frozen,inter)
            if common is None: common=firma
            else: exigir(common==firma,'Operador/preparación declarados distintos entre brazos')
            t=arrays(folder+'traces.npz');flow=arrays(folder+'flow/FLOW.npz')
            phase=np.array(['preparacion']*40+['ensayo']*400)
            exigir(np.array_equal(t['fase'],phase) and np.array_equal(t['paso'],np.r_[np.arange(1,41),np.arange(1,401)]),'Fases/pasos')
            for k,v in t.items():
                exigir(v.ndim>=1 and len(v)==440,'Forma de traza: '+k)
                if v.dtype.kind in 'fiu': exigir(np.isfinite(v).all(),'No finito: '+k)
            clock=t['CNS_time_ns']
            exigir(clock.shape==(440,) and clock.dtype.kind in 'iu' and np.all(clock[1:]-clock[:-1]==1000000),'Reloj CNS')
            exigir(np.array_equal(clock,t['PN_time_ns']) and np.array_equal(clock,t['body_time_ns']),'Relojes distintos')
            exigir(t['sensores_usados'].shape==(440,3) and np.all(t['sensores_usados'][:40]==0),'Preparación no limpia')
            odor={'sham':[0,0,0],'odor_left':[1,0,0],'odor_right':[0,1,0],'uniform':[1,1,0]}[arm]
            exigir(np.array_equal(t['sensores_usados'][40:],np.tile(odor,(400,1))),'Exposición consumida')
            q,used,base=(t[k] for k in ('DN_q_actual','DN_q_usada','DN_baseline'))
            exigir(q.shape==used.shape==base.shape==(440,4),'Ejes DN')
            exigir(np.array_equal(used[1:],q[:-1]),'No conserva desfase declarado 1 ms')
            exigir(np.array_equal(base,np.tile(inter['baseline_inicial'],(440,1))),'Baseline cambió')
            command=decodificar(used,base)
            exigir(np.array_equal(command,t['command_yaw_rate_rad_s']),'Fórmula DNb05 no exacta')
            if arm!='sham':
                s=data['sham']['trace'];exigir(set(t)==set(s),'Campos distintos')
                exigir(all(np.array_equal(t[k][:40],s[k][:40]) for k in t),'Preparación registrada no idéntica')
                exigir(np.array_equal(clock,s['CNS_time_ns']),'Reloj absoluto distinto')
            exigir(np.array_equal(flow['ids'],FLOW_IDS) and np.array_equal(flow['time_ns'],clock)
                    and np.array_equal(flow['phase'],phase),'IDs/reloj de flujo')
            exigir(flow['target'].shape==flow['raw_signed'].shape==(440,4),'Forma de flujo')
            exigir(np.isfinite(flow['target']).all() and np.isfinite(flow['raw_signed']).all(),'Flujo no finito')
            psi=yaw(t['qpos']);theta=psi[40:]-psi[39]
            erryaw=maxabs(theta-t['yaw_delta_deg'][40:])
            exigir(erryaw<=1e-10,'Yaw guardado no coincide con qpos/origen')
            dt=(clock[40:]-clock[39:-1]).astype(float)*1e-9
            integral=np.rad2deg(np.cumsum(command[40:]*dt))
            nuevo_base=np.broadcast_to(q[39],base.shape)
            offline=np.rad2deg(np.cumsum(decodificar(used,nuevo_base)[40:]*dt))
            diff=used[:,2]-used[:,3]-(base[:,2]-base[:,3])
            checks[arm]={'lag1_exact':True,'decoder_exact':True,'yaw_readback_error_deg':erryaw,
                'lag0_command_error_rad_s':maxabs(decodificar(q,base)-command),
                'lag2_command_error_rad_s':maxabs(decodificar(q[:-2],base[2:])-command[2:]),
                'DNa02_nonzero_targets_sampled':int(np.count_nonzero(flow['target'][:,2:])),
                'DNb05_reader_ids_L_R':DN_IDS,'baseline_L_R':base[0,2:].tolist(),
                'q_used_ON_L_R':used[40,2:].tolist(),
                'qvel_max_abs_native_units':maxabs(t['qvel'])}
            data[arm]={'trace':t,'theta':theta,'command':integral,'residual':theta-integral,
                       'offline_rebased_command':offline,'d':diff[40:],'flow':flow['target'][40:]}
        rows=[]
        for ms in VENTANAS:
            i=ms-1
            for arm in ARMS:
                d=data[arm];r={'arm':arm,'ms':ms,'DNb05_delta_used':float(d['d'][i]),
                  'PN_target_L_R':float(d['flow'][i,1]-d['flow'][i,0]),
                  'OFFLINE_rebased_command_integral_deg':float(d['offline_rebased_command'][i])}
                for key in ('theta','command','residual'):
                    r[key+'_deg']=float(d[key][i])
                    for control in ('sham','uniform'):
                        r[key+'_minus_'+control+'_deg']=float(d[key][i]-data[control][key][i])
                rows.append(r)
        csvout(out/'DESCOMPOSICION.csv',rows)
        contrasts=[]
        for ms in VENTANAS:
            i=ms-1;r={'ms':ms}
            for key in ('theta','command','residual'):
                L,R,U,S=[data[a][key][i] for a in ('odor_left','odor_right','uniform','sham')]
                r[key+'_odd_deg']=float((L-R)/2);r[key+'_common_deg']=float((L+R)/2-S)
                r[key+'_uniform_minus_sham_deg']=float(U-S)
            contrasts.append(r)
        csvout(out/'CONTRASTES.csv',contrasts)
        np.savez_compressed(out/'CURVAS.npz',ms=np.arange(1,401),arms=np.array(ARMS),
            **{key:np.stack([data[a][key] for a in ARMS]) for key in ('theta','command','residual','offline_rebased_command')})
        result={'status':'OFFLINE_COMPLETE','checks':checks,'contrasts':contrasts,'input_sha256':hashes,
                'manifest_checked':manifest is not None,'code_sha256':hashfile(Path(__file__)),
                'wall_s':time.perf_counter()-tic,'stage3_admission':False,
                'limits':'Residual yaw-command es contabilidad, no torque causal. Rebasado offline no es intervención. Target DNa02 cero se refiere SOLO a muestras.'}
        guardar(out/'RESULTADO.json',result);return result
    except Exception:
        guardar(out/'FALLO.json',{'error':traceback.format_exc(),'input_sha256':hashes});raise
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path);a=p.parse_args()
    r=analizar(a.campaign.resolve(),a.out,a.manifest)
    print(json.dumps({'status':r['status'],'contraste_400':r['contrasts'][-1],'checks':r['checks']},indent=2,ensure_ascii=False))
