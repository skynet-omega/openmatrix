"""Lectura offline: una interfaz bilateral, cuatro brazos, muestras confirmadas.
No simula, no modifica ecuaciones, no estima causalidad ni error entre muestras.
"""
import argparse, csv, hashlib, json, tempfile, traceback, zipfile
from pathlib import Path
import numpy as np
BRAZOS = ('odor_left', 'odor_right', 'uniform', 'sham')
IDENTIDAD = ('pipeline_sha256', 'operator_sha256', 'routes_sha256', 'initial_sha256')
def exigir(ok, mensaje):
    if not ok: raise ValueError(mensaje)
def guardar(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def huella(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def escalar(z, k):
    exigir(k in z and z[k].shape == (), 'Falta metadato escalar: '+k)
    return str(z[k].item())
def cargar(p, plan, brazo):
    with zipfile.ZipFile(p) as f:
        exigir(sum(x.file_size for x in f.infolist()) <= 128*1024**2, 'NPZ supera 128 MiB descomprimido')
    with np.load(p, allow_pickle=False) as f: z = {k: f[k].copy() for k in f.files}
    for k, esperado in [('arm', brazo), ('interface', plan['interface']), ('unit', plan['unit'])]:
        exigir(escalar(z, k) == esperado, 'Metadato distinto: '+k)
    for k in IDENTIDAD:
        exigir(escalar(z, k) == plan[k], 'Identidad distinta: '+k)
    t, ids, grupos = z['t_ns'], z['target_ids'], z['groups']
    exigir(t.ndim == 1 and len(t)>1 and t.dtype.kind in 'iu', 'Reloj entero requerido')
    exigir((t>=0).all() and (t[1:]>t[:-1]).all(), 'Reloj no creciente')
    exigir(ids.shape == (2,) and ids.dtype.kind in 'iu' and len(np.unique(ids))==2, 'Dos destinos distintos requeridos')
    exigir(np.array_equal(ids, plan['target_ids_L_R']), 'Orden/identidad de destinos L/R')
    exigir(grupos.ndim == 1 and len(grupos)>0 and grupos.dtype.kind in 'US', 'Grupos textuales requeridos')
    exigir(len(np.unique(grupos))==len(grupos) and all(str(g) and not str(g).startswith('__') for g in grupos), 'Grupos duplicados/reservados')
    exigir(z['committed'].shape == t.shape and z['committed'].dtype.kind == 'b' and z['committed'].all(), 'Muestras no confirmadas')
    forma = (len(t), 2, len(grupos))
    if plan['mode'] == 'conductance_inward':
        exigir(plan['unit'] == 'pA', 'nS*mV debe expresarse en pA')
        g, v, e = [z[k] for k in ('g_nS', 'V_mV', 'E_mV')]
        exigir(all(a.shape == forma and a.dtype.kind in 'fiu' and np.isfinite(a).all() for a in (g,v,e)), 'Conductancia/voltajes invalidos')
        exigir((g>=0).all(), 'Conductancia negativa')
        valores = g.astype(float)*(e.astype(float)-v.astype(float))
    else:
        valores = z['value'].astype(float)
        exigir(valores.shape == forma, 'Forma value incorrecta')
    otro, usado = z['other'], z['consumed']
    exigir(otro.shape == usado.shape == (len(t),2), 'Forma de other/consumed incorrecta')
    for a in (valores, otro, usado):
        exigir(a.dtype.kind in 'fiu' and np.isfinite(a).all(), 'Valor no finito/no numerico')
    error = float(np.max(np.abs(valores.sum(axis=2)+otro-usado)))
    exigir(error <= plan['reconstruction_atol'], 'No reconstruye la entrada consumida: '+str(error))
    return t, grupos, valores, otro, usado, error

def analizar(plan_path, out):
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    exigir(plan['mode'] in ('conductance_inward','effective_input'), 'Modo no admitido')
    ids_plan = plan['target_ids_L_R']
    exigir(len(ids_plan)==2 and all(type(i) is int and i>=0 for i in ids_plan) and ids_plan[0]!=ids_plan[1], 'IDs L/R enteros y distintos')
    exigir(set(plan['arms']) == set(BRAZOS) and plan['sample_kind']=='committed_endpoint', 'Brazos/muestras incorrectos')
    exigir(isinstance(plan['interface'],str) and plan['interface'] and isinstance(plan['unit'],str) and plan['unit'], 'Interfaz/unidad')
    for k in IDENTIDAD:
        s = plan[k]; exigir(isinstance(s,str) and len(s)==64 and all(c in '0123456789abcdef' for c in s), 'SHA256 invalido: '+k)
    exigir(type(plan['onset_ns']) is int and plan['onset_ns']>=0, 'Onset entero no negativo')
    ventana = plan['window_after_onset_ns']
    exigir(len(ventana)==2 and all(type(i) is int for i in ventana) and 0<=ventana[0]<ventana[1], 'Ventana temporal')
    tolerancia = plan['reconstruction_atol']
    exigir(np.isfinite(tolerancia) and tolerancia>0, 'Tolerancia algebraica invalida')
    out.mkdir(exist_ok=False); guardar(out/'CONTRATO.json', plan)
    datos=[]; recibos={}; rutas=set()
    try:
        for brazo in BRAZOS:
            item = plan['arms'][brazo]; p = (plan_path.parent/item['file']).resolve()
            exigir(p not in rutas, 'Mismo archivo en dos condiciones'); rutas.add(p)
            digest=huella(p)
            if item.get('sha256'): exigir(digest==item['sha256'], 'Archivo modificado')
            t, grupos, valor, otro, usado, error = cargar(p,plan,brazo)
            if datos:
                exigir(np.array_equal(t,t0) and np.array_equal(grupos,g0), 'Tiempos/grupos distintos; no se alinean')
            else: t0,g0=t,grupos
            datos.append(np.concatenate((valor,otro[:,:,None],usado[:,:,None]),axis=2))
            recibos[brazo]={'sha256':digest,'reconstruction_max_abs':error}
        a,b=[plan['onset_ns']+i for i in ventana]
        ia,ib=np.flatnonzero(t0==a),np.flatnonzero(t0==b)
        exigir(len(ia)==len(ib)==1, 'Ventana incompleta: se requieren ambos extremos exactos')
        i,j=int(ia[0]),int(ib[0]); dt=np.diff(t0[i:j+1]).astype(float)
        promedios=[]
        for x in datos:
            y=x[i:j+1]; promedios.append(np.sum((y[:-1]+y[1:])*.5*dt[:,None,None],axis=0)/float(b-a))
        medias=np.array(promedios); bilateral=medias[:,0,:]-medias[:,1,:]
        nombres=list(map(str,g0))+['__OTHER__','__TOTAL_CONSUMED__']; filas=[]
        for k,nombre in enumerate(nombres):
            L,R,U,S=bilateral[:,k]
            fila={'group':nombre,'unit':plan['unit'],'odd_L_R':float((L-R)/2),
                  'common_vs_sham':float((L+R)/2-S),'L_sham':float(L-S),
                  'R_sham':float(R-S),'U_sham':float(U-S),'R_uniform':float(R-U)}
            for h,brazo in enumerate(BRAZOS):
                fila['input_L_'+brazo]=float(medias[h,0,k]); fila['input_R_'+brazo]=float(medias[h,1,k])
            filas.append(fila)
        with (out/'CONTRASTES.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
        resultado={'status':'SUMA_Y_CONTRASTES_RECONSTRUIDOS','inputs':recibos,'total':filas[-1],
            'window_absolute_ns':[a,b],'samples_used':j-i+1,'source_sha256':huella(Path(__file__)),
            'limits':'Media trapezoidal entre muestras; no cota continua, causalidad o autenticidad de la captura. other no se resta del total. Una interfaz por ejecucion.',
            'stage3_admission':False}
        guardar(out/'RESULTADO.json',resultado); return resultado
    except Exception:
        guardar(out/'FALLO.json',{'error':traceback.format_exc(),'inputs':recibos}); raise

def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp); firma=hashlib.sha256(b'fixture_no_organismo').hexdigest()
        plan={'mode':'conductance_inward','interface':'fixture','unit':'pA','sample_kind':'committed_endpoint',
              'target_ids_L_R':[10,20],'onset_ns':0,'window_after_onset_ns':[0,335000000],
              'reconstruction_atol':1e-10,'arms':{},**{k:firma for k in IDENTIDAD}}
        for brazo in BRAZOS:
            t=np.array([0,100000000,200000000,300000000,335000000],dtype=np.int64)
            g=np.ones((5,2,2));v=np.full_like(g,-60.);e=np.zeros_like(g);e[:,:,1]=-70.
            if brazo in ('odor_left','uniform'):g[:,0,0]=2.
            if brazo in ('odor_right','uniform'):g[:,1,0]=2.
            valor=g*(e-v);z=dict(t_ns=t,target_ids=np.array([10,20]),groups=np.array(['exc','inh']),
                committed=np.ones(5,dtype=bool),g_nS=g,V_mV=v,E_mV=e,other=np.zeros((5,2)),consumed=valor.sum(2),
                arm=np.array(brazo),interface=np.array('fixture'),unit=np.array('pA'),**{k:np.array(firma) for k in IDENTIDAD})
            np.savez(r/(brazo+'.npz'),**z);plan['arms'][brazo]={'file':brazo+'.npz'}
        p=r/'plan.json';guardar(p,plan);res=analizar(p,r/'ok')
        exigir(abs(res['total']['odd_L_R']-60.)<1e-12,'Signo/unidad incorrectos')
        objetivo=r/'odor_right.npz'; original=objetivo.read_bytes();rechazos=[]
        for caso in ('consumed','ids','nonfinite','uncommitted','outward_sign','window400'):
            with np.load(objetivo,allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
            if caso=='consumed':z['consumed'][0,0]+=.01
            elif caso=='ids':z['target_ids']=z['target_ids'][::-1]
            elif caso=='nonfinite':z['g_nS'][0,0,0]=np.nan
            elif caso=='uncommitted':z['committed'][-1]=False
            elif caso=='outward_sign':z['consumed']*=-1
            else:plan['window_after_onset_ns']=[250000000,400000000]
            np.savez(objetivo,**z);guardar(p,plan)
            try:analizar(p,r/caso)
            except ValueError:rechazos.append(caso)
            else:raise AssertionError('Corrupcion aceptada: '+caso)
            objetivo.write_bytes(original);plan['window_after_onset_ns']=[0,335000000]
        return {'scope':'SOLO aritmetica CPU sintetica; no OpenMatrix','odd_pA':60.,'rejected':rechazos}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selftest',action='store_true')
    p.add_argument('--plan',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.selftest: print(json.dumps(selftest(),indent=2))
    else:
        exigir(a.plan is not None and a.out is not None,'Se requieren --plan y --out')
        print(json.dumps(analizar(a.plan.resolve(),a.out),indent=2,ensure_ascii=False))
