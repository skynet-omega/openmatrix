"""Contrastes descriptivos de cuatro NPZ. No simulacion ni inferencia causal.
CSV/JSON necesitan NumPy; Parquet/Feather necesitan ademas pandas y su lector.
IDs y tiempos deben estar en cada NPZ. No se supone el orden de una tabla externa.
"""
import argparse, csv, hashlib, json, re, sys, tempfile, time, traceback, zipfile
from pathlib import Path
import numpy as np
BRAZOS = ('odor_left', 'odor_right', 'uniform', 'sham')
ETAPAS = ('ORN', 'LN', 'PN', 'LHN', 'LAL', 'DNa02')
def exigir(ok, mensaje):
    if not ok: raise ValueError(mensaje)
def leer_json(p): return json.loads(p.read_text(encoding='utf-8'))
def guardar(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def texto(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return ''
    return str(x).strip()
def identificador(x):
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)): y = int(x)
    elif isinstance(x, str) and re.fullmatch(r'[0-9]+', x.strip()): y = int(x)
    else: raise ValueError('bodyId no entero exacto: '+repr(x))
    exigir(0 <= y <= np.iinfo(np.int64).max, 'bodyId fuera de int64'); return y
def metadatos(p):
    if p.suffix.lower()=='.json': filas = leer_json(p)
    elif p.suffix.lower()=='.csv':
        with p.open(encoding='utf-8', newline='') as f: filas = list(csv.DictReader(f))
    else:
        import pandas as pd
        exigir(p.suffix.lower() in ('.parquet','.feather'), 'Formato anatomico no admitido')
        tabla = pd.read_parquet(p) if p.suffix.lower()=='.parquet' else pd.read_feather(p)
        filas = tabla.astype(object).where(pd.notna(tabla), None).to_dict('records')
    exigir(isinstance(filas, list), 'Anatomia debe contener filas')
    resultado = {}
    for r in filas:
        exigir(isinstance(r, dict) and 'bodyId' in r, 'Falta bodyId anatomico')
        i = identificador(r['bodyId']); exigir(i not in resultado, 'bodyId anatomico duplicado')
        resultado[i] = r
    return resultado
def clasificar(ids, nodos, plan):
    reglas = plan.get('stage_by_type', {}); overrides = plan.get('side_by_type', {})
    for tipo, regla in reglas.items():
        exigir(set(regla)=={'stage','source'} and texto(regla['stage']) and texto(regla['source']), 'Regla sin etapa/procedencia')
    filas = []
    for j, i in enumerate(ids):
        r = nodos.get(int(i), {}); tipo = texto(r.get('type')) or 'SIN_TIPO'
        campo = overrides.get(tipo, plan.get('side_field', 'rootSide'))
        exigir(campo in ('rootSide','somaSide','instance'), 'Campo de lado no admitido')
        bruto = texto(r.get(campo))
        match = re.fullmatch(r'.+_(L|R)', bruto) if campo=='instance' else None
        lado = match.group(1) if match else (bruto if campo!='instance' and bruto in ('L','R') else 'SIN_LADO')
        regla = reglas.get(tipo, {})
        filas.append(dict(columna=j, bodyId=int(i), type=tipo, stage=regla.get('stage','SIN_ETAPA'),
            stage_source=regla.get('source',''), lado=lado, campo_lado=campo, valor_lado=bruto,
            rootSide=texto(r.get('rootSide')), somaSide=texto(r.get('somaSide')), instance=texto(r.get('instance')),
            anotacion_presente=bool(r)))
    return filas
def cargar(p, plan, brazo):
    claves = plan['keys']; requeridas = [claves[k] for k in ('ids','time','value')]
    with zipfile.ZipFile(p) as z:
        total = sum(i.file_size for i in z.infolist() if i.filename in [k+'.npy' for k in requeridas])
        exigir(total <= plan.get('max_array_mib',256)*1024**2, 'Arrays exceden limite de lectura')
    with np.load(p, allow_pickle=False) as z:
        exigir(all(k in z for k in requeridas), f'{brazo}: faltan {requeridas}; presentes {z.files}')
        ids, t, x = [z[k].copy() for k in requeridas]
        if 'arm' in z:
            etiqueta = z['arm']; exigir(etiqueta.shape==() and str(etiqueta.item())==brazo, 'Etiqueta de brazo incompatible')
        if 'dataset' in z: exigir(str(z['dataset'].item())==plan['dataset'], 'Dataset incompatible')
        etiqueta_verificada = 'arm' in z
    exigir(ids.ndim==1 and ids.dtype.kind in 'iu' and len(ids)>0, 'Eje de IDs entero requerido')
    exigir(np.max(ids)<=np.iinfo(np.int64).max and np.min(ids)>=0 and len(np.unique(ids))==len(ids), 'IDs invalidos/duplicados')
    ids = ids.astype(np.int64)
    exigir(t.ndim==1 and len(t)>0 and t.dtype.kind in 'fiu' and np.isfinite(t).all(), 'Tiempo invalido')
    exigir(t[0]>=0 and np.all(np.diff(t.astype(float))>0), 'Tiempo no estrictamente creciente')
    exigir(x.ndim==2 and x.shape==(len(t),len(ids)) and x.dtype.kind in 'fiu', 'Se requiere valor[tiempo,ID]')
    exigir(np.isfinite(x).all(), 'Datos no finitos; no se imputan')
    if 'expected_times_ms' in plan: exigir(np.array_equal(t,plan['expected_times_ms']), 'Tiempos no previstos')
    return ids, t, x.astype(np.float64), etiqueta_verificada

def contrastes(v):
    L,R,U,S = v
    return {'L_sham':L-S, 'R_sham':R-S, 'U_sham':U-S, 'impar':(L-R)/2,
            'comun':(L+R)/2-S, 'R_uniform':R-U, 'L_uniform':L-U}
def csv_escribir(p, filas):
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
def analizar(plan_path, nodes_path, out):
    plan = leer_json(plan_path)
    exigir(set(plan['arms'])==set(BRAZOS) and plan.get('time_unit')=='ms', 'Cuatro brazos y tiempo ms requeridos')
    exigir(plan.get('dataset') and plan.get('pipeline') and plan.get('quantity') and plan.get('unit'), 'Falta procedencia/magnitud')
    exigir(set(plan['keys'])=={'ids','time','value'}, 'Claves NPZ invalidas')
    out.mkdir(exist_ok=False); guardar(out/'PLAN_EJECUTADO.json',plan); inicio=time.perf_counter()
    recibos = {}; datos=[]; etiquetas={}
    try:
        nodos = metadatos(nodes_path); recibos['anatomia_sha256']=sha(nodes_path)
        vistos=set()
        for brazo in BRAZOS:
            entrada=plan['arms'][brazo]; p=(plan_path.parent/entrada['file']).resolve()
            exigir(p not in vistos, 'Mismo archivo asignado a dos brazos'); vistos.add(p)
            digest=sha(p)
            if entrada.get('sha256'): exigir(digest==entrada['sha256'], 'Huella de entrada distinta')
            ids,t,x,etiqueta=cargar(p,plan,brazo)
            if datos:
                exigir(np.array_equal(ids,ids0), 'IDs/orden no coinciden entre brazos; no se reordenan')
                exigir(np.array_equal(t,t0), 'Tiempos distintos; no se interpola ni ajusta lag')
            else: ids0,t0=ids,t
            datos.append(x); etiquetas[brazo]=etiqueta
            recibos[brazo]={'path':str(p),'sha256':digest,'sha256_esperado_verificado':bool(entrada.get('sha256'))}
        tabla=clasificar(ids0,nodos,plan); grupos={}
        for r in tabla:
            grupos.setdefault(('tipo',r['type']),[]).append(r['columna'])
            if r['stage']!='SIN_ETAPA': grupos.setdefault(('etapa',r['stage']),[]).append(r['columna'])
        ordenar=sorted(grupos); lados=np.array([r['lado'] for r in tabla])
        # NaN de salida significa lado ausente; nunca sustituye datos de entrada.
        curvas=np.full((len(ordenar),4,3,len(t0)),np.nan); conteos=np.zeros((len(ordenar),3),dtype=int)
        for g,key in enumerate(ordenar):
            ix=np.array(grupos[key])
            for k,lado in enumerate(('L','R','SIN_LADO')):
                jj=ix[lados[ix]==lado]; conteos[g,k]=len(jj)
                if len(jj):
                    for b,x in enumerate(datos): curvas[g,b,k]=x[:,jj].mean(axis=1)
        window=plan.get('window_ms',[float(t0[0]),float(t0[-1])]); lo,hi=window
        exigir(np.isfinite(window).all() and lo<=hi, 'Ventana invalida')
        sel=(t0>=lo)&(t0<=hi); exigir(sel.any(), 'Sin observaciones en la ventana')
        for r in tabla:
            j=r['columna']; v=np.array([x[sel,j].mean() for x in datos])
            r.update({b:float(v[k]) for k,b in enumerate(BRAZOS)})
            r.update({k:float(vv) for k,vv in contrastes(v).items()})
        csv_escribir(out/'NODOS.csv',tabla); resumen=[]
        for g,(nivel,nombre) in enumerate(ordenar):
            bilateral=bool(conteos[g,0] and conteos[g,1])
            par=curvas[g,:,0]-curvas[g,:,1]
            row={'nivel':nivel,'grupo':nombre,'n_L':int(conteos[g,0]),'n_R':int(conteos[g,1]),
                 'n_sin_lado':int(conteos[g,2]),'bilateral_disponible':bilateral}
            row.update({b:float(par[k,sel].mean()) if bilateral else None for k,b in enumerate(BRAZOS)})
            row.update({k:float(v[sel].mean()) if bilateral else None for k,v in contrastes(par).items()})
            resumen.append(row)
        csv_escribir(out/'BILATERAL.csv',resumen)
        np.savez_compressed(out/'CURVAS.npz',times_ms=t0,niveles=np.array([x[0] for x in ordenar]),
            grupos=np.array([x[1] for x in ordenar]),brazos=np.array(BRAZOS),lados=np.array(['L','R','SIN_LADO']),
            medias=curvas,conteos=conteos,ids=ids0)
        result={'estado':'ANALISIS_DESCRIPTIVO_COMPLETO','n':len(ids0),'tiempos':len(t0),
            'ventana_solicitada_ms':window,'instantes_usados_ms':t0[sel].tolist(),
            'media':'Aritmetica de instantes guardados; no integral temporal ni p-valor',
            'sin_anotacion':sum(not r['anotacion_presente'] for r in tabla),
            'sin_tipo':sum(r['type']=='SIN_TIPO' for r in tabla),'sin_lado':sum(r['lado']=='SIN_LADO' for r in tabla),
            'sin_etapa':sum(r['stage']=='SIN_ETAPA' for r in tabla),
            'etapas_sin_mapeo_en_eje':[g for g in ETAPAS if not any(r['stage']==g for r in tabla)],
            'diferencia_primera_muestra_vs_sham':{b:float(np.max(abs(datos[k][0]-datos[3][0]))) for k,b in enumerate(BRAZOS)},
            'etiqueta_brazo_contrastada_dentro_NPZ':etiquetas,'entradas':recibos,
            'codigo_sha256':sha(Path(__file__)),'wall_s':time.perf_counter()-inicio,
            'limites':'Procedencia declarada; no valida circuito, corrientes, picos, causalidad, motor ni conducta. NaN en CURVAS solo denota lado ausente.',
            'stage3_admission':False}
        guardar(out/'RESULTADO.json',result); return result
    except Exception:
        guardar(out/'FALLO.json',{'error':traceback.format_exc(),'entradas_leidas':recibos}); raise

def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp); ns=r/'nodes.json'; p=r/'plan.json'
        guardar(ns,[{'bodyId':10,'type':'DNa02','instance':'DNa02_L'},
                    {'bodyId':20,'type':'DNa02','instance':'DNa02_R'}, {'bodyId':30}])
        plan={'dataset':'fixture','pipeline':'fixture','quantity':'q','unit':'1','time_unit':'ms',
              'keys':{'ids':'ids','time':'times_ms','value':'q'},'side_by_type':{'DNa02':'instance'},
              'stage_by_type':{'DNa02':{'stage':'DNa02','source':'fixture explicitamente anotado'}},'arms':{}}
        for b,vals in zip(BRAZOS,([.2,.5,.3],[.5,.2,.3],[.4,.4,.3],[.1,.1,.3])):
            f=r/(b+'.npz'); np.savez(f,ids=np.array([10,20,30]),times_ms=np.array([0,1]),q=np.array([[.1,.1,.3],vals]),arm=np.array(b))
            plan['arms'][b]={'file':f.name}
        guardar(p,plan);res=analizar(p,ns,r/'ok')
        exigir(res['sin_tipo']==1 and res['sin_lado']==1, 'Se invento una etiqueta')
        with (r/'ok/BILATERAL.csv').open() as f: rows=list(csv.DictReader(f))
        dn=next(x for x in rows if x['nivel']=='tipo' and x['grupo']=='DNa02')
        exigir(abs(float(dn['impar'])+.15)<1e-14,'Signo incorrecto')
        f=r/'odor_right.npz'; original=f.read_bytes(); fallos=[]
        for clave in ('ids','times_ms','q','arm'):
            with np.load(f,allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
            if clave=='ids': d[clave]=d[clave][::-1]
            elif clave=='times_ms': d[clave]=np.array([0,2])
            elif clave=='q': d[clave][0,0]=np.nan
            else: d[clave]=np.array('sham')
            np.savez(f,**d)
            try: analizar(p,ns,r/clave)
            except ValueError: fallos.append(clave)
            else: raise AssertionError('Corrupcion aceptada: '+clave)
            f.write_bytes(original)
        return {'scope':'SOLO fixture CPU, no NPZ del organismo','signos':True,
                'etiquetas_ausentes_preservadas':True,'corrupciones_rechazadas':fallos}
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--selftest',action='store_true')
    ap.add_argument('--plan',type=Path); ap.add_argument('--nodes',type=Path); ap.add_argument('--out',type=Path); args=ap.parse_args()
    if args.selftest: print(json.dumps(selftest(),indent=2))
    else:
        exigir(args.plan and args.nodes and args.out,'Se requieren --plan, --nodes, --out')
        print(json.dumps(analizar(args.plan.resolve(),args.nodes.resolve(),args.out),indent=2))
