"""CPU: eventos prescritos, masa constante no singular y dos tratamientos temporales.
A=Radau con cortes. B=Taylor afin con cota de truncacion en aritmetica exacta.
No CUDA, DAE, deteccion endogena, retardos, ni conversion automatica de unidades.
"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json, math, hashlib, traceback
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import scipy
from scipy.linalg import lu_factor, lu_solve, expm
from scipy.integrate import solve_ivp
PLAN = dict(rtol=1e-8, atol=1e-10, error_externo=1e-4,
            presupuesto_truncacion=1e-7, norma_paso=.5, grado_max=64,
            consultas=51, cpu_max_s=60, ram_max_bytes=2*1024**3)
def exigir(ok, texto):
    if not ok: raise ValueError(texto)
@dataclass(frozen=True)
class Evento:
    t: float
    fila: int
    op: str
    valor: float
class Modelo:
    def __init__(self, nombre, M, x0, escalas, limites, unidades, K=None, b=None, F=None, puerto=None):
        self.nombre=nombre; self.x0=np.array(x0,dtype=float,copy=True); n=len(self.x0)
        self.M=np.array(M,dtype=float,copy=True); self.escalas=np.array(escalas,dtype=float)
        self.lo=np.broadcast_to(limites[0],(n,)).copy(); self.hi=np.broadcast_to(limites[1],(n,)).copy()
        exigir(self.M.shape==(n,n) and self.x0.shape==(n,), 'Forma masa/estado')
        exigir(np.isfinite(self.M).all() and np.linalg.cond(self.M)<1e12, 'Masa singular/mal condicionada')
        exigir(self.escalas.shape==(n,) and np.isfinite(self.escalas).all() and (self.escalas>0).all(), 'Escalas')
        exigir(len(unidades)==n and not np.isnan(self.lo).any() and not np.isnan(self.hi).any() and (self.lo<=self.hi).all(), 'Dominio/unidades')
        for i,j in zip(*np.nonzero(self.M)):
            exigir(unidades[i]==unidades[j], 'Masa mezcla dimensiones de estado')
        self.unidades=tuple(unidades); self.lu=lu_factor(self.M); self.validar(self.x0)
        self.K=None if K is None else np.array(K,dtype=float,copy=True)
        self.b=np.zeros(n) if b is None else np.array(b,dtype=float,copy=True)
        exigir((self.K is None)!=(F is None), 'Declarar K o F, no ambos')
        if self.K is not None:
            exigir(self.K.shape==(n,n) and self.b.shape==(n,) and np.isfinite(self.K).all() and np.isfinite(self.b).all(), 'Operador afin')
        self.F=F; self.puerto=puerto or (lambda t: np.zeros(0))
    def validar(self,x):
        exigir(np.isfinite(x).all() and (x>=self.lo).all() and (x<=self.hi).all(), 'Estado fuera de dominio')
    def raw(self,t,x):
        y=self.K@x+self.b if self.K is not None else np.asarray(self.F(t,x,self.puerto(t)),dtype=float)
        exigir(y.shape==x.shape and np.isfinite(y).all(), 'RHS no finito/forma incorrecta')
        return y
    def aumentado(self):
        exigir(self.K is not None, 'B solo admite ecuaciones afines CONSTANTES')
        L=np.linalg.solve(self.M,np.column_stack((self.K,self.b)))
        H=np.zeros((len(self.x0)+1,)*2); H[:-1]=L
        return H
class Sesion:
    def __init__(self, modelo, eventos=()):
        self.m=modelo; self.eventos=tuple(eventos); n=len(modelo.x0)
        for e in self.eventos:
            exigir(type(e.fila) is int and 0<=e.fila<n and e.op in ('ADD','SET'), 'Evento')
            exigir(np.isfinite([e.t,e.valor]).all() and e.t>0, 'Eventos prescritos t>0')
        x=modelo.x0.copy(); x.flags.writeable=False
        self.publicado=(0.,x,()); self.cota=0.
    def avanzar(self,fin,modo='A'):
        m=self.m; inicio,x0,anteriores=self.publicado
        exigir(np.isfinite(fin) and fin>inicio and modo in ('A','B','ref'), 'Reloj/modo')
        trabajo=dict(rhs=0,solve_masa=0,solve_masa_matriz=0,matvec=0,expm=0,pasos=0,factorizaciones=0)
        cpu=time.process_time(); pared=time.perf_counter(); x=x0.copy(); E=self.cota
        ev=sorted([(j,e) for j,e in enumerate(self.eventos) if inicio<e.t<=fin],key=lambda z:(z[1].t,z[0]))
        cortes=sorted({inicio,fin}|{e.t for _,e in ev}); malla=np.linspace(inicio,fin,PLAN['consultas'])
        ts=[inicio]; ys=[x.copy()]; cotas=[E if modo=='B' else 0.]; publicados=list(anteriores)
        H=m.aumentado() if modo=='B' or (modo=='ref' and m.K is not None) else None
        if H is not None: trabajo['solve_masa_matriz']=1
        if modo=='B':
            D=np.r_[m.escalas,1.]; H=H*D[None,:]/D[:,None]; norma=float(np.linalg.norm(H,np.inf))
            exigir(norma*(fin-inicio)<500, 'Cota demasiado grande: alcance rechazado')
        for a,z in zip(cortes[:-1],cortes[1:]):
            consultas=np.unique(np.r_[malla[(malla>a)&(malla<z)],z])
            if modo in ('A','ref') and not (modo=='ref' and m.K is not None):
                def fun(t,y):
                    trabajo['rhs']+=1; trabajo['solve_masa']+=1
                    return lu_solve(m.lu,m.raw(t,y)) if modo=='A' else np.linalg.solve(m.M,m.raw(t,y))
                sol=solve_ivp(fun,(a,z),x,method='Radau' if modo=='A' else 'DOP853',
                    rtol=PLAN['rtol'] if modo=='A' else 1e-12,
                    atol=m.escalas*(PLAN['atol'] if modo=='A' else 1e-14),dense_output=True)
                exigir(sol.success,sol.message); trabajo['pasos']+=len(sol.t)-1
                trabajo['factorizaciones']+=getattr(sol,'nlu',0)
                vals=sol.sol(consultas).T; x=vals[-1].copy(); bs=np.zeros(len(vals))
            elif modo=='ref':
                vals=[]; v=np.r_[x,1.]
                for t in consultas:
                    vals.append((expm(H*(t-a))@v)[:-1]); trabajo['expm']+=1
                vals=np.array(vals); x=vals[-1].copy(); bs=np.zeros(len(vals))
            else:
                piezas=[]; t=a; v=np.r_[x/m.escalas,1.]
                while t<z:
                    dt=min(z-t,PLAN['norma_paso']/norma if norma else z-t)
                    final=z if dt==z-t else t+dt
                    exigir(final>t, 'Paso sin progreso'); dt=final-t; alpha=norma*dt
                    presupuesto=PLAN['presupuesto_truncacion']*(dt/(fin-inicio))*math.exp(-norma*(fin-final))
                    term=v.copy(); suma=v.copy(); coef=[v.copy()]
                    for k in range(1,PLAN['grado_max']+1):
                        siguiente=(dt/k)*(H@term); trabajo['matvec']+=1
                        cola=float(np.linalg.norm(siguiente,np.inf))/(1-alpha/(k+1))
                        if cola<=presupuesto: break
                        suma+=siguiente; coef.append(siguiente.copy()); term=siguiente
                    else: raise RuntimeError('Rechazo: presupuesto Taylor no satisfecho')
                    E=math.exp(alpha)*E+cola
                    piezas.append((t,final,coef,E,suma.copy())); v=suma; t=final; trabajo['pasos']+=1
                vals=[]; bs=[]
                for t in consultas:
                    iz,de,co,bo,ultimo=next(p for p in piezas if p[0]<=t<=p[1])
                    theta=(t-iz)/(de-iz); p=co[-1].copy()
                    for c in co[-2::-1]: p=p*theta+c
                    if t==de:p=ultimo
                    vals.append(p[:-1]*m.escalas); bs.append(bo)
                vals=np.array(vals); x=v[:-1]*m.escalas
            for t,y,be in zip(consultas,vals,bs):
                m.validar(y); ts.append(float(t)); ys.append(y.copy()); cotas.append(float(be))
            for j,e in ev:
                if e.t!=z: continue
                x[e.fila]=e.valor if e.op=='SET' else x[e.fila]+e.valor
                m.validar(x); publicados.append(j); ts.append(z); ys.append(x.copy()); cotas.append(E if modo=='B' else 0.)
        resultado=dict(t=np.array(ts),x=np.array(ys),cota=np.array(cotas),trabajo=trabajo,
                       pared_s=time.perf_counter()-pared,cpu_s=time.process_time()-cpu)
        x.flags.writeable=False
        self.publicado=(fin,x,tuple(publicados)); self.cota=E if modo=='B' else 0.
        return resultado

def modelos():
    tau=125e-6; K=np.array([[-1.,0,0],[1,-1,0],[0,1,-1]])/tau
    tarde=Modelo('tardio',np.eye(3),[0,0,0],[1]*3,(0.,1.),['1']*3,K=K)
    M=np.array([[2,.2,0,0],[.2,1,0,0],[0,0,1,.1],[0,0,.1,1.4]])
    K=np.array([[-2,.3,.1,0],[-.2,-1.5,0,.1],[.001,0,-1,.4],[0,.001,.2,-2]])
    b=np.array([-100.,-80.,.3,.8]); x=[-60.,-65.,2.,.5]
    datos=(M,x,[100.,100.,3.,3.],([-100,-100,0,0],[50,50,10,10]),['mV','mV','mM','mM'])
    afin=Modelo('masa_recurrente',*datos,K=K,b=b)
    nonlinear=Modelo('no_lineal',*datos,F=lambda t,x,u:K@x+b+np.array([3*np.tanh(x[2]),0,-.02*x[2]**2,u[0]]),puerto=lambda t:np.array([.03*np.sin(2*t)]))
    ev=(Evento(.21,0,'ADD',5.),Evento(.6,2,'SET',3.),Evento(.6,2,'ADD',.2),Evento(.87,1,'SET',-50.))
    return [(tarde,(Evento(.9*tau,0,'ADD',.5),),tau),(afin,ev,1.),(nonlinear,ev,1.)]

def control_sin_corte():
    h=125e-6; evento=.9*h; F=np.array([[-1.,0],[1,-1]])/h
    def proy(y,t):
        z=y.copy(); z[:2]=0 if t<evento else expm(F*(t-evento))@np.array([.5,0.]); return z
    def paso(y,t,dt):
        z=proy(y,t); medio=z.copy(); medio[2]+=(-np.expm1(-dt/(2*h)))*(z[1]-z[2])
        medio=proy(medio,t+dt/2); out=z.copy(); out[2]+=(-np.expm1(-dt/h))*(medio[1]-z[2])
        return proy(out,t+dt)
    entero=paso(np.zeros(3),0.,h); mitad=paso(paso(np.zeros(3),0.,h/2),h/2,h/2)
    return dict(estimador=float(np.max(abs(entero-mitad))),z=float(mitad[2]),error_real=float(.5*np.exp(-.1)*.1**2/2))

def pruebas():
    m=Modelo('orden',[[1.]],[.8],[1.],(0.,1.),['1'],K=[[0.]])
    ev=[Evento(.05,0,'SET',.7),Evento(.05,0,'ADD',.1)]
    for modo in ('A','B'):
        for eventos,objetivo in ((ev,.8),(ev[::-1],.7)):
            s=Sesion(m,eventos); s.avanzar(.1,modo); exigir(abs(s.publicado[1][0]-objetivo)<1e-14,'Orden')
        s=Sesion(m,[Evento(.03,0,'SET',.4),Evento(.08,0,'SET',2.)]); antes=s.publicado
        try:s.avanzar(.1,modo)
        except ValueError:pass
        else:raise RuntimeError('Dominio no rechazado')
        exigir(s.publicado is antes,'Publicacion parcial')
    s=Sesion(modelos()[2][0]); antes=s.publicado
    try:s.avanzar(.1,'B')
    except ValueError:pass
    else:raise RuntimeError('B acepto no linealidad')
    exigir(s.publicado is antes,'Mutacion tras rechazo')
    return {'orden_ADD_SET':True,'rollback_dominio':True,'B_rechaza_no_lineal':True}

def main():
    out=Path(sys.argv[1] if len(sys.argv)>1 else 'resultado_temporal'); out.mkdir(exist_ok=False)
    (out/'PLAN.json').write_text(json.dumps(PLAN,indent=2),encoding='utf-8')
    res={'estado':'EN_PROCESO','plan':PLAN,'modelos':[],'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    inicio=time.perf_counter(); cpu=time.process_time()
    try:
        res['pruebas']=pruebas(); res['control_sin_corte']=control_sin_corte()
        exigir(res['control_sin_corte']['estimador']==0 and res['control_sin_corte']['error_real']>1e-4,'Control negativo')
        for m,ev,fin in modelos():
            ref=Sesion(m,ev).avanzar(fin,'ref'); np.savez(out/(m.nombre+'_ref.npz'),t=ref['t'],x=ref['x'])
            fila={'modelo':m.nombre,'segundos':fin,'referencia':'expm' if m.K is not None else 'DOP853 refinado','resultados':{}}
            for modo in ('A','B') if m.K is not None else ('A',):
                r=Sesion(m,ev).avanzar(fin,modo); exigir(np.array_equal(r['t'],ref['t']),'Tiempos distintos')
                err=np.max(np.abs(r['x']-ref['x'])/m.escalas,axis=1)
                fila['resultados'][modo]={'error_max':float(err.max()),'cota_max':float(r['cota'].max()) if modo=='B' else None,
                    'cota_cubre_muestras_observadas':bool(np.all(err<=r['cota'])) if modo=='B' else None,
                    'trabajo':r['trabajo'],'pared_s':r['pared_s'],'cpu_s':r['cpu_s']}
                np.savez(out/(m.nombre+'_'+modo+'.npz'),t=r['t'],x=r['x'],cota=r['cota'])
                (out/'PARCIAL.json').write_text(json.dumps(fila,indent=2),encoding='utf-8')
                exigir(err.max()<=PLAN['error_externo'],'Criterio externo incumplido')
                if modo=='B':exigir(r['cota'].max()<=PLAN['presupuesto_truncacion'],'Presupuesto de truncacion')
            res['modelos'].append(fila)
        res['estado']='COMPLETO'
    except Exception:res.update(estado='FALLO_CONSERVADO',error=traceback.format_exc())
    res.update(pared_s=time.perf_counter()-inicio,cpu_s=time.process_time()-cpu,numpy=np.__version__,scipy=scipy.__version__,
        alcance='CPU; eventos prescritos; cota B de truncacion, NO del redondeo/solve de masa; A sin cota global rigurosa; no organismo')
    (out/'RESULTADO.json').write_text(json.dumps(res,indent=2,allow_nan=False),encoding='utf-8'); print(json.dumps(res,indent=2))
    return 0 if res['estado']=='COMPLETO' else 1
if __name__=='__main__':
    if sys.platform.startswith('linux'):
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(PLAN['cpu_max_s'],PLAN['cpu_max_s']))
        resource.setrlimit(resource.RLIMIT_AS,(PLAN['ram_max_bytes'],PLAN['ram_max_bytes']))
    sys.exit(main())
