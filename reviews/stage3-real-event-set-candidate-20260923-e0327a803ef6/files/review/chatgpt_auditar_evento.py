"""Replay CPU del puerto, no simulador. No cambia límites ni usa clipping.
--waveform: fuente publicada para ejecutar lif_record y contrastar metadatos.
--case: JSON forense; números decimales o float.hex(), no un checkpoint.
"""
import argparse, ast, hashlib, importlib.util, json, math, time, traceback
from decimal import Decimal as D, localcontext
from pathlib import Path
import numpy as np
from scipy.linalg import expm

BLOB = '22ea8ddf2b50092390967a126e53e810358ff032'
def exigir(ok, mensaje):
    if not ok: raise ValueError(mensaje)
def numero(x):
    return float.fromhex(x) if isinstance(x,str) and ('0x' in x.lower()) else float(x)
def guardar(p, x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def conv(t,tq,ts):
    z=t*(1/ts-1/tq)
    if z==0: return (t/ts)*math.exp(-t/ts)
    if abs(z)<.5: return (t/ts)*math.exp(-t/ts)*math.expm1(z)/z
    den=1-ts/tq
    return (math.exp(-t/tq)*(-math.expm1(-z)) if z>0 else math.exp(-t/ts)*math.expm1(z))/den

def validar(caso):
    c={k:numero(caso[k]) for k in ('q0','s0','tau_q','tau_s','duration_s','query_s')}
    exigir(np.isfinite(list(c.values())).all(),'Operandos no finitos')
    exigir(c['tau_q']>0 and c['tau_s']>0 and 0<=c['query_s']<=c['duration_s'],'Tiempo/tau')
    c['events']=[]; previo=-math.inf
    for e in caso['events']:
        t=numero(e['time']);j=numero(e['jump']);op=e.get('op','ADD')
        exigir(np.isfinite([t,j]).all() and previo<=t<=c['duration_s'] and t>=0,'Orden/evento inválido')
        exigir(op in ('ADD','SET'),'Operación no admitida'); previo=t
        post=numero(e['post']) if op=='SET' else None
        exigir(post is None or np.isfinite(post),'SET necesita post finito')
        c['events'].append(dict(time=t,jump=j,op=op,post=post))
    return c

def replay(c, referencia=False):
    """Evolución por segmentos; SET es el resultado físico del emisor, no un clamp."""
    q,s=c['q0'],c['s0'];previo=0.;tq,ts=c['tau_q'],c['tau_s']
    A=np.array([[-1/tq,0],[1/ts,-1/ts]])
    def flujo(dt,q,s):
        if referencia: return tuple(expm(A*dt)@np.array([q,s]))
        return q*math.exp(-dt/tq),s*math.exp(-dt/ts)+q*conv(dt,tq,ts)
    for e in c['events']:
        if e['time']>c['query_s']: break
        q,s=flujo(e['time']-previo,q,s);previo=e['time']
        q=e['post'] if e['op']=='SET' else q+e['jump']
    return np.array(flujo(c['query_s']-previo,q,s))

def aditivo(c, precision):
    """Suma del registro ADD; no convierte diferencias a transiciones SET."""
    with localcontext() as ctx:
        ctx.prec=precision;dd=lambda x:D.from_float(float(x))
        t,tq=dd(c['query_s']),dd(c['tau_q'])
        q=dd(c['q0'])*(-t/tq).exp()
        for e in c['events']:
            if e['time']<=c['query_s']:
                q+=dd(e['jump'])*(-(t-dd(e['time']))/tq).exp()
        return q

def informe_caso(caso):
    c=validar(caso);t=c['query_s'];q=c['q0']*math.exp(-t/c['tau_q'])
    for e in c['events']:
        if e['time']<=t: q+=e['jump']*math.exp(-(t-e['time'])/c['tau_q'])
    alta=aditivo(c,90);alta2=aditivo(c,110)
    correcto=replay(c);ref=replay(c,True)
    with localcontext() as ctx:
        ctx.prec=110;exceso=str(alta2-D(1));acuerdo=str(abs(alta2-alta))
    return {'add_q_fp64':q,'add_q_hex':q.hex(),'add_q_90':str(alta),
            'add_q_110':str(alta2),'add_exceso_sobre_1':exceso,'diferencia_90_110':acuerdo,
            'transicion_declarada_qs':correcto.tolist(),'referencia_expm_qs':ref.tolist(),
            'max_error_expm':float(np.max(abs(correcto-ref))),
            'limite':'Decimal no es intervalo riguroso; 1 es frontera del ejemplo, no dominio universal.'}

def cargar_fuente(p):
    b=p.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
    exigir(blob==BLOB,'La fuente no corresponde al snapshot revisado')
    spec=importlib.util.spec_from_file_location('wf_original',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    s=ast.get_source_segment(b.decode(),next(n for n in ast.parse(b).body if isinstance(n,ast.FunctionDef) and n.name=='lif_record'))
    cambios=[('jumps=np.zeros_like(times);clipped=0','jumps=np.zeros_like(times);clipped=0;posts=np.full_like(times,np.nan);resets=np.zeros_like(times,dtype=np.bool_)'),
      ('            if event_value>1.:clipped+=1;event_value=1.','            reset_event=event_value>1.\n            if event_value>1.:clipped+=1;event_value=1.'),
      ('times[k,n]=t;jumps[k,n]=event_value-before;n+=1','times[k,n]=t;jumps[k,n]=event_value-before;posts[k,n]=event_value;resets[k,n]=reset_event;n+=1'),
      ('return clipped,times,jumps','return clipped,times,jumps,posts,resets')]
    for antes,despues in cambios:
        exigir(s.count(antes)==1,'No se encontró ancla única');s=s.replace(antes,despues)
    espacio={'np':np,'math':math};exec(compile(s,'<lif_con_metadatos>','exec'),espacio)
    from numba import njit
    m.lif_record=njit(cache=False,fastmath=False)(m.lif_record.py_func)  # Evitar cache de importacion dinamica.
    observado=njit(cache=False,fastmath=False)(espacio['lif_record'])
    return m,observado,s,hashlib.sha256(b).hexdigest()

def pruebas(fuente,out):
    m,observado,s,sha=cargar_fuente(fuente)
    (out/'lif_metadatos_propuesto.py').write_text('import numpy as np\nimport math\n'+s+'\n',encoding='utf-8')
    dt=.000125;tq=.024;te=.00003125;q0=.7
    base=[np.array([-1.]),np.array([0.]),np.array([0],dtype=np.int64),np.array([q0]),
          np.array([1.]),np.array([math.log(2)/te]),np.array([-1.]),np.array([0.]),
          np.array([1/(.5*tq)]),np.array([tq])]
    a=[x.copy() for x in base];b=[x.copy() for x in base]
    viejo=m.lif_record(*a,dt);nuevo=observado(*b,dt)
    exigir(all(x.tobytes()==y.tobytes() for x,y in zip(a,b)),'Metadatos alteraron estado físico')
    exigir(viejo[0]==nuevo[0] and all(x.tobytes()==y.tobytes() for x,y in zip(viejo[1:],nuevo[1:3])),'Alteración de eventos')
    t=float(viejo[1][0,0]);j=float(viejo[2][0,0]);post=float(nuevo[3][0,0])
    exigir(viejo[0]==1 and bool(nuevo[4][0,0]) and post==1.,'Fixture no produjo transición saturada')
    caso=dict(q0=q0,s0=.2,tau_q=tq,tau_s=.005,duration_s=dt,query_s=t,
              events=[dict(time=t,jump=j,op='SET',post=post)])
    guardar(out/'CASO_SINTETICO.json',caso);c=validar(caso);r=informe_caso(caso)
    w=m.Waveform([q0],[.2],[tq],.005,dt);w.add([t],np.array([0]),[j]);q,sval=w.at(t)
    exigir(q[0]>1 and aditivo(c,90)>D(1),'No se reprodujo el exceso codificado')
    exigir(replay(c)[0]==1.,'SET no conserva el valor físico exacto')
    maximo=0.
    for consulta in sorted(set([0.,t,dt,np.nextafter(t,0),np.nextafter(t,dt)])):
        d=dict(c,query_s=float(consulta));maximo=max(maximo,float(np.max(abs(replay(d)-replay(d,True)))))
    exigir(maximo<1e-12,'Discrepancia frente a expm')
    continuo=dict(c,query_s=dt);exigir(replay(continuo)[0]==a[3][0],'Cambio del endpoint q físico')
    # ADD puro: comparación con Waveform original; no promover equivalencia bit a bit global.
    add=dict(c,events=[dict(time=t,jump=.1,op='ADD',post=None)],query_s=dt)
    wa=m.Waveform([q0],[.2],[tq],.005,dt);wa.add([t],np.array([0]),[.1])
    erradd=float(np.max(abs(replay(add)-np.array([x[0] for x in wa.at(dt)]))))
    exigir(erradd<1e-12,'ADD puro no concuerda')
    # Orden en la misma marca, independiente del orden de ejecución de hilos.
    e1=dict(time=t,jump=0.,op='SET',post=.7);e2=dict(time=t,jump=.1,op='ADD',post=None)
    exigir(replay(dict(c,events=[e1,e2]))[0]==.7+.1,'Orden SET/ADD')
    exigir(replay(dict(c,events=[e2,e1]))[0]==.7,'Orden ADD/SET')
    for post in (-2.,3.):
        e=dict(time=t,jump=0.,op='SET',post=post)
        exigir(replay(dict(c,events=[e]))[0]==post,'Se introdujo dominio universal')
    np.savez(out/'EVIDENCIA_CPU.npz',legacy_at_event=np.r_[q,sval],set_at_event=replay(c),
             original_final_q=a[3],record_times=viejo[1],record_jumps=viejo[2],posts=nuevo[3])
    return {'source_sha256':sha,'source_blob':BLOB,'caso':'SINTETICO; no ID41645',
      'estado_fisico_con_y_sin_metadatos_exacto':True,'eventos_originales_exactos':True,
      'legacy_Waveform_q':float(q[0]),'replay':r,'max_error_prefijos_expm':maximo,
      'add_only_error':erradd,'orden_ADD_SET':True,'dominios_heterogeneos':True}

def dump_antes_rollback(g,puertos,raiz,meta):
    """Llamar dentro de code!=0, ANTES de cp.copyto(x,backup).
    Solo diagnostica. No publica checkpoint ni evalúa coeficientes.
    """
    g.stream.synchronize();p=Path(raiz);p.mkdir(parents=True,exist_ok=False)
    arrays={k:getattr(g,k).get(stream=g.stream) for k in ('x','backup','full','fine','clock','status')}
    for k in ('qr','sr','q','s','tq','ts','times','jumps','counts'):
        arrays['port_'+k]=getattr(puertos,k).get(stream=g.stream)
    arrays['lower']=g.lower_host.copy();arrays['upper']=g.upper_host.copy()
    np.savez(p/'FALLO_NO_REANUDABLE.npz',**arrays)
    malos=np.flatnonzero(~np.isfinite(arrays['fine'])|(arrays['fine']<arrays['lower'])|(arrays['fine']>arrays['upper']))
    for fila in malos[:16]:
        slots=np.flatnonzero(arrays['port_qr']==fila)
        if len(slots)!=1: continue
        k=int(slots[0]);n=int(arrays['port_counts'][k]);reloj=arrays['clock']
        eventos=[dict(time=float(arrays['port_times'][k,j]).hex(),jump=float(arrays['port_jumps'][k,j]).hex(),op='ADD') for j in range(n)]
        consulta=float(reloj[0]+reloj[1])
        caso=dict(q0=float(arrays['port_q'][k]).hex(),s0=float(arrays['port_s'][k]).hex(),
          tau_q=float(arrays['port_tq'][k]).hex(),tau_s=float(arrays['port_ts'][0]).hex(),
          query_s=consulta.hex(),duration_s=float(meta['duration_s']).hex(),events=eventos,
          fila=int(fila),slot=k,observed=float(arrays['fine'][fila]).hex(),post_fisico_ausente=True)
        guardar(p/f'PUERTO_{fila}.json',caso)
    guardar(p/'STATUS.json',dict(meta=meta,failed=True,resumable=False,bad_rows=malos.tolist(),
         limit='x=último subpaso aceptado, backup=entrada de advance; no son checkpoint conjunto'))

def main():
    import sys
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--waveform',type=Path)
    ap.add_argument('--case',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    exigir(bool(a.waveform)^bool(a.case),'Elegir --waveform o --case');a.out.mkdir(exist_ok=False)
    tic=time.perf_counter();cpu=time.process_time();r={'status':'INCOMPLETE'}
    try:
        r['result']=pruebas(a.waveform,a.out) if a.waveform else informe_caso(json.loads(a.case.read_text()))
        r['status']='COMPLETE'
    except Exception:r['error']=traceback.format_exc()
    r.update(wall_s=time.perf_counter()-tic,cpu_s=time.process_time()-cpu,cuda_executed=False,
      organism_executed=False,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    guardar(a.out/'RESULTADO.json',r);print(json.dumps(r,indent=2,ensure_ascii=False));return 0 if r['status']=='COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
