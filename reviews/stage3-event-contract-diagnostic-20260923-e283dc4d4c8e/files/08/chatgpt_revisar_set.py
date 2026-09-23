"""Pruebas CPU sobre c17d10f. No modifica fuente, tolerancias ni organismo.
El prefijo usa un caso sintético; los operandos de REAL provienen del evento 41645.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse, hashlib, importlib.util, json, math, sys, time, traceback
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scipy.linalg import expm
SHA='2549fb0ca90afa90020d1dcd533e59a32fc59f0eee12605919832730e33f3feb'
SHA_PARENT='49986d70a5a72fe33267c7261b904d5ef68d0bde4e6344ff050e38dfeec75fae'
REAL=(.8061807853958571,.8830133842861765,.02426011860370636,.005,4.486788730489399e-6,.19396829995917275,.000125)
TOL=1e-12

def exigir(ok,msg):
    if not ok: raise AssertionError(msg)
def cargar(p,nombre,esperado):
    b=p.read_bytes(); exigir(hashlib.sha256(b).hexdigest()==esperado,'SHA de fuente distinto')
    spec=importlib.util.spec_from_file_location(nombre,p)
    m=importlib.util.module_from_spec(spec);sys.modules[nombre]=m;spec.loader.exec_module(m)
    return m

def oracle(w,t,incluir=True):
    """Referencia independiente: exponencial 2x2 y saltos ordenados por fila."""
    out=np.column_stack((w.q,w.s)).copy()
    for r in range(len(w.q)):
        A=np.array([[-1/w.tau[r],0.],[1/w.ts,-1/w.ts]])
        prev=0.; x=out[r].copy()
        orden=sorted((k for k,rr in enumerate(w.rows) if rr==r),key=lambda k:(w.times[k],k))
        for k in orden:
            e=w.times[k]
            if e>t or (not incluir and e==t): break
            x=expm(A*(e-prev))@x;prev=e
            if np.isfinite(w.posts[k]): x[0]=w.posts[k]
            else: x[0]+=w.jumps[k]
        out[r]=expm(A*(t-prev))@x
    return out.T

def prefix(W):
    w=W([.7],[.2],[.024],.005,.000125)
    w.add([.00003125],np.array([0]),[.1]);t=.00008
    a=np.array(w.at(t));w.add([.0001],np.array([0]),[0.],posts=[.7]);b=np.array(w.at(t))
    return {'tiempo_s':t,'SET_futuro_s':.0001,'antes':a.ravel().tolist(),
            'despues':b.ravel().tolist(),'delta':(b-a).ravel().tolist(),
            'bits_iguales':a.tobytes()==b.tobytes(),
            'hex_antes':[float(v).hex() for v in a.ravel()],
            'hex_despues':[float(v).hex() for v in b.ravel()]}

def positivos(m):
    W=m.Waveform;peor=0.;continuidad=0.;consultas=0
    for tq,ts in ((.024,.005),(.005,.005),(.005,.024)):
        for orden in (0,1):
            w=W([.4],[.2],[tq],ts,.000125)
            te=40e-6
            saltos=[0.,.1] if orden==0 else [.1,0.]
            posts=[.7,np.nan] if orden==0 else [np.nan,.7]
            w.add([te,te],np.array([0,0]),saltos,posts=posts)
            w.add([90e-6,70e-6],np.array([0,0]),[0.,.05],posts=[.6,np.nan])
            exigir(w.at(te)[0][0]==(.7+.1 if orden==0 else .7),'Orden simultáneo')
            for t in (0.,np.nextafter(te,0),te,np.nextafter(te,1),70e-6,90e-6,.000125):
                err=float(np.max(abs(np.array(w.at(t))-oracle(w,t))))
                peor=max(peor,err);consultas+=1
            continuidad=max(continuidad,float(abs(w.at(te)[1][0]-oracle(w,te,False)[1,0])))
    exigir(peor<TOL and continuidad<TOL,'Referencia/continuidad')
    w=W([.2],[.3],[.01],.005,.000125); antes=(list(w.times),list(w.posts))
    try:w.add([30e-6,30e-6],np.array([0,0]),[0.,0.],posts=[.7,np.inf])
    except ValueError:pass
    else:raise AssertionError('Publicación inválida aceptada')
    exigir((w.times,w.posts)==antes,'Publicación parcial')
    return {'consultas_expm':consultas,'max_error_expm':peor,
            'max_error_s_continua':continuidad,'rechazo_sin_eventos_parciales':True}

def caso_real(m):
    q,s,tq,ts,te,j,dur=REAL
    w=m.Waveform([q],[s],[tq],ts,dur);w.add([te],np.array([0]),[j])
    add=float(w.at(te)[0][0]);compensado=math.fsum([q*math.exp(-te/tq),j])
    w=m.Waveform([q],[s],[tq],ts,dur);w.add([te],np.array([0]),[j],posts=[1.])
    setq=float(w.at(te)[0][0]);expm_error=float(np.max(abs(np.array(w.at(dur))-oracle(w,dur))))
    with localcontext() as c:
        c.prec=100;D=Decimal.from_float
        alto=D(q)*(-D(te)/D(tq)).exp()+D(j)
        exceso=str(alto-Decimal(1))
    exigir(setq==1. and expm_error<TOL,'Replay real SET')
    return {'ADD_fp64':add,'ADD_fsum':compensado,'ADD_exceso_100digitos':exceso,
            'SET_post_declarado':setq,'error_expm_al_final':expm_error,
            'alcance':'Replay de operandos publicados; no simulación de la célula real.'}

def padre(m,p):
    dt=.01;tq=.024
    base=[np.array([-1.]),np.array([0.]),np.array([0],dtype=np.int64),np.array([.2]),
          np.array([1.]),np.array([math.log(2)/.00003125]),np.array([-1.]),np.array([0.]),
          np.array([120.]),np.array([tq])]
    a=[x.copy() for x in base];b=[x.copy() for x in base]
    old=p.lif_record(*a,dt);new=m.lif_record(*b,dt)
    exacto=all(x.tobytes()==y.tobytes() for x,y in zip(a,b)) and old[0]==new[0]
    exacto=exacto and all(x.tobytes()==y.tobytes() for x,y in zip(old[1:],new[1:3]))
    exigir(exacto,'Cambió el productor')
    mask=np.isfinite(new[1][0]);w=m.Waveform([.2],[.3],[tq],.005,dt)
    w.add(new[1][0,mask],np.zeros(mask.sum(),dtype=np.int64),new[2][0,mask],posts=new[3][0,mask])
    qend=float(w.at(dt)[0][0]);exigir(qend==b[3][0],'No recupera estado final de la fuente')
    return {'eventos':int(mask.sum()),'productor_y_eventos_anteriores_bitabit':True,
            'q_final_reconstruida':qend,'q_final_fuente':float(b[3][0])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--parent',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(exist_ok=False);inicio=time.perf_counter();cpu=time.process_time()
    r={'CUDA_ejecutado':False,'organismo_ejecutado':False,'criterio_referencia':TOL,'fuente_sha256':SHA}
    try:
        m=cargar(a.source.resolve(),'wf_c17d10_review',SHA)
        r['positivos']=positivos(m);r['evento_real']=caso_real(m);r['prefix_original']=prefix(m.Waveform)
        if a.parent:r['comparacion_productor']=padre(m,cargar(a.parent.resolve(),'wf_parent_review',SHA_PARENT))
        # Candidata local: tres guardias dependen sólo de SET ya ocurridos.
        texto=a.source.read_text()
        guardia='any(np.isfinite(self.posts))'
        exigir(texto.count(guardia)==2,'Anclas distintas')
        texto=texto.replace(guardia,'any(np.isfinite(p) and te<=time for p,te in zip(self.posts,self.times))')
        antes='if np.isfinite(post)}';exigir(texto.count(antes)==1,'Ancla filas distinta')
        texto=texto.replace(antes,'if np.isfinite(post) and self.times[i]<=time}')
        f=a.out/'event_waveform_prefix_candidate.py';f.write_text(texto)
        h=hashlib.sha256(f.read_bytes()).hexdigest();c=cargar(f.resolve(),'wf_prefix_candidate_review',h)
        r['prefix_candidata']=prefix(c.Waveform);r['positivos_candidata']=positivos(c)
        exigir(r['prefix_candidata']['bits_iguales'],'La candidata no restaura el prefijo probado')
        r['parche_solo_CPU_sha256']=h
        r['estado']='COMPLETO_REVISION_CPU'
    except Exception:r.update(estado='FALLO_CONSERVADO',error=traceback.format_exc())
    r.update(pared_s=time.perf_counter()-inicio,cpu_s=time.process_time()-cpu,
             codigo_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'RESULTADO.json').write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(r,indent=2,ensure_ascii=False));return 0 if r['estado']=='COMPLETO_REVISION_CPU' else 1
if __name__=='__main__':sys.exit(main())
