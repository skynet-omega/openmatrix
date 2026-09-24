"""Delta de propietario ligado a su vista base y al consumidor exacto.
Sólo CPU/auditoría. No integra ni sustituye PN/APL/KC. No elimina fallback.
Las vistas son copias privadas; los hashes no son un plan de caché GPU rápido.
"""
import argparse, hashlib, json
from pathlib import Path
from dataclasses import dataclass
import numpy as np


def exigir(ok, texto):
    if not ok: raise ValueError(texto)


def huella(*objetos):
    h=hashlib.sha256()
    for x in objetos:
        if isinstance(x,np.ndarray):
            x=np.ascontiguousarray(x); b=x.dtype.str.encode()+str(x.shape).encode()+x.tobytes()
        else: b=str(x).encode()
        h.update(len(b).to_bytes(8,'little'));h.update(b)
    return h.hexdigest()


def privado(a, tipo):
    exigir(isinstance(a,np.ndarray) and a.dtype==np.dtype(tipo),'Tipo de array incorrecto')
    a=np.array(a,copy=True,order='C');exigir(np.isfinite(a).all(),'No finito')
    a.flags.writeable=False;return a


class Vista:
    def __init__(self, rows, ptr, src, w, s, caps, visual, scale, connected, contexto):
        requeridos={'consumer','epoch_ns','t_s_hex','phase','operator','owners','events','candidate'}
        exigir(set(contexto)==requeridos,'Contexto incompleto')
        exigir(type(contexto['epoch_ns']) is int and contexto['epoch_ns']>=0,'Época')
        exigir(all(isinstance(contexto[k],str) and contexto[k] for k in requeridos-{'epoch_ns'}),'Identidades')
        t=float.fromhex(contexto['t_s_hex']);exigir(np.isfinite(t) and t>=0,'Tiempo')
        self.ctx=json.dumps(contexto,sort_keys=True,separators=(',',':'))
        self.rows=privado(rows,'int32');self.ptr=privado(ptr,'int64');self.src=privado(src,'int32')
        self.w=privado(w,'float64');self.s=privado(s,'float64');self.caps=privado(caps,'float64')
        self.visual=privado(visual,'uint8');n=len(self.s);e=len(self.src);m=len(self.rows)
        exigir(m>0 and all(x.ndim==1 for x in (self.rows,self.ptr,self.src,self.w,self.s,self.caps,self.visual)),'Dimensiones')
        exigir(self.ptr.shape==(m+1,) and self.ptr[0]==0 and self.ptr[-1]==e and np.all(np.diff(self.ptr)>0),'CSR')
        exigir(self.w.shape==(e,) and self.caps.shape==self.visual.shape==(n,),'Formas')
        exigir(np.all(np.diff(self.rows)>0) and np.all((self.rows>=0)&(self.rows<n)) and np.all((self.src>=0)&(self.src<n)),'Índices')
        exigir(np.all(self.visual<=1) and type(connected) is bool and np.isfinite(scale) and scale>0,'Flags/escala')
        self.scale=float(scale);self.connected=connected
        self.structure=huella(self.rows,self.ptr,self.src)
        self.rules=huella(self.caps,self.visual,self.scale,self.connected)
        self.key=huella(self.ctx,self.structure,self.rules,self.w,self.s)
        self.receivers=np.repeat(np.arange(m),np.diff(self.ptr));self.receivers.flags.writeable=False

    def reduce(self,x):
        out=np.vstack([np.bincount(self.receivers,weights=c,minlength=len(self.rows)) for c in x])
        exigir(np.isfinite(out).all(),'Reducción no finita');return out

    def terms(self):
        v=self.w*self.s[self.src];vis=self.visual[self.rows][self.receivers].astype(bool)
        allow=self.connected | (self.visual[self.src]==0);u=v*self.scale
        return np.vstack((np.where(~vis & allow,v*self.caps[self.src],0.),
                          np.where(vis,np.maximum(u,0.),0.),np.where(vis,np.maximum(-u,0.),0.)))


@dataclass(frozen=True)
class Corriente:
    view: str
    ctx: str
    values: np.ndarray


@dataclass(frozen=True)
class Delta:
    before: str
    after: str
    ctx: str
    values: np.ndarray


def corriente(v):
    return Corriente(v.key,v.ctx,privado(v.reduce(v.terms()),'float64'))


def delta(a,b):
    exigir(a.ctx==b.ctx and a.structure==b.structure and a.rules==b.rules,'Otra consulta, estructura o ley; reconstruir')
    # Mismas operaciones por arista que owner_overlay.cu, sin omitir términos cruzados.
    old=a.w*a.s[a.src];new=b.w*b.s[b.src]
    vis=a.visual[a.rows][a.receivers].astype(bool);allow=a.connected | (a.visual[a.src]==0)
    x=np.vstack((np.where(~vis & allow,(new-old)*a.caps[a.src],0.),
        np.where(vis,np.maximum(new*a.scale,0.)-np.maximum(old*a.scale,0.),0.),
        np.where(vis,np.maximum(-new*a.scale,0.)-np.maximum(-old*a.scale,0.),0.)))
    return Delta(a.key,b.key,a.ctx,privado(a.reduce(x),'float64'))


def aplicar(base,d):
    exigir(type(base) is Corriente and type(d) is Delta and base.view==d.before and base.ctx==d.ctx,'Delta sobre baseline equivocado/duplicado')
    exigir(base.values.shape==d.values.shape,'Forma del delta')
    return Corriente(d.after,base.ctx,privado(base.values+d.values,'float64'))


def sobrescribir(target,rate,patches,orden,ctx):
    """SET parciales en orden EFECTIVO de retorno de los wrappers.
    patches={owner:{ctx,rows,target? ,rate?}}. Ausente != cero. Sin clipping.
    El adaptador debe proporcionar el orden real; no se infiere por anatomía.
    """
    a=privado(target,'float64').copy();r=privado(rate,'float64').copy()
    exigir(a.ndim==1 and a.shape==r.shape and len(set(orden))==len(orden) and set(orden)==set(patches),'Plan incompleto')
    for owner in orden:
        p=patches[owner];exigir(p['ctx']==ctx,'Patch de otra consulta')
        ix=privado(p['rows'],'int64');exigir(ix.ndim==1 and len(np.unique(ix))==len(ix) and np.all((ix>=0)&(ix<len(a))),'Filas patch')
        exigir(set(p)<= {'ctx','rows','target','rate'} and bool({'target','rate'}&set(p)),'Patch vacío/desconocido')
        for name,out in (('target',a),('rate',r)):
            if name in p:
                v=privado(p[name],'float64');exigir(v.shape==ix.shape,'Forma patch');out[ix]=v
    return privado(a,'float64'),privado(r,'float64')


def selftest():
    c=dict(consumer='base',epoch_ns=0,t_s_hex=float(.125).hex(),phase='accepted',operator='W1',owners='O1',events='E1',candidate='X1')
    rows=np.array([0,1],np.int32);ptr=np.array([0,1,2],np.int64);src=np.array([0,0],np.int32)
    def v(s,w=(2.,2.),context=c):
        return Vista(rows,ptr,src,np.array(w),np.array([s,0.]),np.array([1.,1.]),np.array([0,1],np.uint8),1.,True,context)
    initial,event,owner=v(.25),v(.5),v(1.,(3.,3.))
    de=delta(initial,event);do=delta(event,owner)
    final=aplicar(aplicar(corriente(initial),de),do)
    exigir(np.array_equal(final.values,corriente(owner).values),'Composición correcta falla')
    wrong=corriente(initial).values+de.values+delta(initial,owner).values
    exigir(wrong[0,0]==3.5 and final.values[0,0]==3.,'Falsador de solapamiento')
    stale=delta(initial,owner);correct=delta(event,owner)
    changed=v(1.,(4.,4.));exigir(changed.key!=owner.key,'Peso nuevo sin invalidación')
    topology=Vista(rows,ptr,np.array([1,0],np.int32),owner.w,owner.s,owner.caps,owner.visual,1.,True,c)
    exigir(stale.values[0,0]==2.5 and correct.values[0,0]==2.,'Cambio de baseline')
    blocked=0
    for fn in (lambda:aplicar(corriente(event),stale),lambda:aplicar(final,do),
               lambda:delta(event,v(1.,context=dict(c,t_s_hex=float(.25).hex()))),
               lambda:delta(event,v(1.,context=dict(c,events='E2'))),
               lambda:delta(event,v(1.,context=dict(c,consumer='another_consumer'))),
               lambda:delta(event,topology)):
        try:fn()
        except ValueError:blocked+=1
        else:raise ValueError('Guard no rechazó corrupción')
    exigir(blocked==6,'Faltan rechazos')
    rate=np.array([7.,4.]);target=np.array([.7,.8]);ix=np.array([0],np.int64)
    patches={'inner':dict(ctx=event.ctx,rows=ix,target=np.array([.2]),rate=np.array([9.])),
             'outer':dict(ctx=event.ctx,rows=ix,target=np.array([-.4]))}
    a,r=sobrescribir(target,rate,patches,['inner','outer'],event.ctx)
    exigir(a[0]==-.4 and r[0]==9. and target[0]==.7,'Prioridad/máscara/propiedad')
    a2,_=sobrescribir(target,rate,patches,['outer','inner'],event.ctx)
    exigir(a2[0]!=a[0],'La prioridad no fue ejercitada')
    return dict(stale_delta=2.5,current_delta=2.,wrong_overlap=3.5,correct_overlap=3.,guards_rejected=blocked,
                overwrite_target=float(a[0]),preserved_rate=float(r[0]),weight_refresh=True,CUDA_executed=False,organism_executed=False)


def verificar_capsula(folder):
    """Mismo gate portable 1e-8. No hay eventos/propietarios completos en esta cápsula."""
    esperado='36e5a64f9830b24344dfac03c90a50ced78be04d205354136548750fcdb845d4'
    fichero=Path(folder)/'capsule.npz'
    exigir(hashlib.sha256(fichero.read_bytes()).hexdigest()==esperado,'Cápsula distinta de 34d2d5c')
    with np.load(fichero,allow_pickle=False) as pack: a={k:pack[k] for k in pack.files}
    errores={}
    for estado in ('A','B'):
        c=dict(consumer='capsule/edge_channels',epoch_ns=0,t_s_hex=float(0).hex(),phase='capsule-fixture',
               operator='captured-fixture',owners='PN-real_APL-PNKC-synthetic',events='NOT_INCLUDED',
               candidate=huella(a['release_'+estado+'_before']))
        def vista(lado):
            return Vista(a['active_rows'],a['row_ptr'],a['sources'],a['weight_'+lado],
                         a['release_'+estado+'_'+lado],a['caps'],a['visual_u8'],
                         float(a['scale']),bool(a['connected']),c)
        old,new=vista('before'),vista('after');d=delta(old,new)
        exp=a['expected_'+estado]
        exigir(d.values.shape==exp.shape and np.isfinite(exp).all(),'Expected inválido')
        error=float(np.max(abs(d.values-exp)));exigir(error<=1e-8,'Gate portable falló; no reajustar')
        errores[estado]=error
    return dict(mode='REAL_CAPSULE_DELTA_ONLY',errors=errores,events_validated=False,
                full_target_rate_validated=False,CUDA_executed=False,organism_executed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--selftest',action='store_true');m.add_argument('--capsule',type=Path)
    args=ap.parse_args();result=selftest() if args.selftest else verificar_capsula(args.capsule)
    result['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result,indent=2,allow_nan=False))
