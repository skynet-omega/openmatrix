"""Criba Arnoldi CPU, no integrador. Interfaces full(t,z)->(F,aristas), version().
No genera eventos: fija tiempo/lado/propietarios del estado publicado.
Usa el RHS ODE EFECTIVO, incluida masa; no sustituye masas por identidad.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
H=125e-6
E=25582938
SHA='dabe36b9a35f5577fc90e315fa198097aa1dd49c975eba55d57a659e356625b9'
def need(ok,msg):
    if not ok:raise ValueError(msg)
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def datos(d):
    z,v,mask,f=(d[k] for k in ('z','v','mask','full_f'))
    n=len(z)
    need(n>0 and n<=400000 and z.shape==v.shape==mask.shape==(n,) and f.shape==(6,n),'Layout')
    need(mask.dtype==bool and all(x.dtype==np.float64 and np.isfinite(x).all() for x in (z,v,f)),'Tipos/finitud')
    need(np.all(v[mask]==0) and np.all(f[:,mask]==0),'Prescritos')
    need(bits(f[0],f[5]),'Repeticion no exacta')
    D=1e-7+1e-5*abs(z);u=v/D
    need(abs(max(abs(u))-1)<1e-12,'Direccion de una tolerancia')
    need(np.isfinite(d['magnitude']) and d['magnitude']>0,'Escala fisica')
    return z,v,mask,f,D,np.linalg.norm(u)
def ortogonal(w,Q):
    r=w.copy();h=np.zeros(len(Q))
    for _ in range(2):
        for i,q in enumerate(Q):
            a=float(q@r);h[i]+=a;r-=a*q
    return h,r
def indicadores(f,D,mask):
    odd=H*abs((f[1]-f[2])/2-(f[3]-f[4]))/D
    even=H*abs((f[1]-f[0])+(f[2]-f[0]))/(2*D)
    return {'central_consistency':float(max(odd[~mask])),
            'even_remainder':float(max(even[~mask]))}
def cargar(root):
    root=Path(root);path=root/'data/JVP_REAL_ARRAYS.npz';p=root/'receipts/JVP_PREFLIGHT.json'
    need(hashlib.sha256(path.read_bytes()).hexdigest()==SHA,'Hash de arrays')
    need(hashlib.sha256(p.read_bytes()).hexdigest()=='266d5d47fd5aaaf8dc08181460b06bd88529d510935fffd5d1102083a0877047','Hash de escala')
    with np.load(path,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    d['magnitude']=json.loads(p.read_text())['raw_residual_normalized_max'];return d
def archivo(root,out):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);d=cargar(root)
    z,v,mask,f,D,beta=datos(d);q=v/D/beta
    w=H*(f[3]-f[4])/(D*beta);h,r=ortogonal(w,[q]);bn=np.linalg.norm(r)
    q2=np.zeros_like(q) if bn==0 else r/bn
    np.savez_compressed(out/'ARNOLDI_SEED.npz',q1=q,q2=q2,h11=h[0],h21=bn)
    result=indicadores(f,D,mask)
    result.update(h11=float(h[0]),h21=float(bn),
        physical_scale=H*d['magnitude'],
        first_action_residual_at_tau1=float(H*d['magnitude']*beta*max(abs(r))*abs(phi(np.array([[h[0]]]),1.)[0])),
        scope='Una columna aproximada. No dimension suficiente ni error integrado.',CUDA=False)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');return result

def phi(K,s):
    A=np.zeros((len(K)+1,len(K)+1));A[:-1,:-1]=K;A[0,-1]=1
    return expm(s*A)[:-1,-1]

def vivo(d,full,version,cost_full,setup,lower,upper,out):
    """Hasta dos columnas nuevas. NO reutiliza Jv del archivo como consulta gratis.
    cost_full: cota auditada por consulta, incluidos propietarios; setup: trabajo nuevo.
    version devuelve los cuatro hashes de contenido, no etiquetas constantes.
    Contexto exclusivo; full copia/sincroniza sus buffers antes de retornarlos.
    """
    out=Path(out);out.mkdir(exist_ok=False,parents=True);start=time.monotonic()
    z,v,mask,f,D,beta=datos(d);t=float(np.asarray(d['time']).reshape(-1)[0])
    need(np.isfinite(t),'Tiempo');n=len(z)
    lo=np.broadcast_to(np.asarray(lower,float),z.shape);hi=np.broadcast_to(np.asarray(upper,float),z.shape)
    need(not np.isnan(lo).any() and not np.isnan(hi).any() and np.all(lo<=hi),'Dominio')
    need(type(cost_full)is int and cost_full>=E and type(setup)is int and setup>=0,'Coste desconocido')
    work=setup;calls=[];inputs=[];Q=[];hist=[]
    initial=tuple(version());need(len(initial)==4 and all(isinstance(x,str) and len(x)==64 for x in initial),'Versiones')
    result={'status':'STARTED','full_calls':0,'setup_edges':setup,'backend':'external_callback'}
    def oracle(x):
        nonlocal work
        need(time.monotonic()-start<120 and work+cost_full<=6*E,'BUDGET_STOP')
        need(tuple(version())==initial,'Version modificada')
        need(np.isfinite(x).all() and np.all((lo<=x)&(x<=hi)),'DOMAIN_STOP')
        a=x.copy();before=a.tobytes();inputs.append(a.copy())
        # Cargo el limite ANTES: un fallo no borra trabajo intentado.
        work+=cost_full;val,used=full(t,a);val=np.array(val,copy=True)
        need(type(used)is int and E<=used<=cost_full,'Coste del callback supera su cota')
        need(val.dtype==np.float64 and val.shape==(n,) and np.isfinite(val).all(),'RHS')
        need(np.all(val[mask]==0) and a.tobytes()==before and tuple(version())==initial,'Mutacion/prescrito/version')
        need(time.monotonic()-start<120,'TIME_STOP');calls.append(val);return val
    try:
        # Dos llamadas reservadas: referencia y repeticion; cada columna cuesta dos.
        mmax=min(2,(6*E-setup-2*cost_full)//(2*cost_full))
        need(mmax>=1,'BUDGET_STOP_BEFORE_GPU')
        base=oracle(z);need(bits(base,f[0]),'Oraculo no coincide con contexto archivado')
        Q=[v/D/beta];K=np.zeros((mmax+1,mmax))
        for j in range(mmax):
            q=Q[j];amplitude=1/max(abs(q));step=amplitude*D*q;step[mask]=0
            plus=oracle(z+step);minus=oracle(z-step)
            even=float(max((H*abs((plus-base)+(minus-base))/(2*D))[~mask]))
            w=H*(plus-minus)/(2*amplitude*D);h,r=ortogonal(w,Q)
            K[:j+1,j]=h;bn=np.linalg.norm(r);K[j+1,j]=bn
            # Residuo de u'=Ku+H*D^-1*slow_f, u(0)=0; NO cota continua.
            vals=[float(H*d['magnitude']*beta*max(abs(r))*abs(phi(K[:j+1,:j+1],s)[-1])) for s in (0.,.25,.5,.75,1.)]
            need(np.isfinite(vals).all(),'Residuo no finito')
            hist.append({'dimension':j+1,'even_remainder':even,'sampled_linear_residual':max(vals)})
            if even>.1:result['status']='NONLINEAR_LOCAL_STOP';break
            if max(vals)<=.1:result['status']='SMALL_SAMPLED_DEFECT_NOT_CERTIFIED';break
            if bn==0:result['status']='NUMERICAL_BREAKDOWN';break
            Q.append(r/bn)
        else:result['status']='DIMENSION_COST_STOP'
        repeat=oracle(z);need(bits(base,repeat),'Repeticion final difiere')
    except Exception as exc:result.update(status='FAILED_RETAINED',error=repr(exc))
    result.update(full_calls=len(inputs),charged_edges=work,equivalents=work/E,
        history=hist,wall_s=time.monotonic()-start,scope='Operador congelado: sin correccion no lineal, eventos o trayectoria.',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(out/'AUDIT.npz',inputs=np.array(inputs),outputs=np.array(calls),basis=np.array(Q))
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--capsule',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(archivo(a.capsule,a.out),indent=2))
