"""Lectura offline de dos paquetes publicados; no simula ni modifica el organismo.
Contrastes de cuatro condiciones y falsadores retrospectivos de mantener salidas.
No certifica active-set, causalidad, ni error entre muestras de 1 ms.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse,csv,hashlib,json,sys,time,traceback,tempfile
from pathlib import Path
import numpy as np
ARMS=('odor_left','odor_right','uniform','sham')
PLAN={'ventana_ms':[161,311],'onset_endpoint_ms':11,'reconstruccion_atol':1e-8,
      'limite_hold_s':1e-4,'salida_baja':.01,'horizontes_ms':[1,5,10,25],
      'onset_numerico_atol':1e-8,'persistencia_muestras':3,
      'naturaleza':'Analisis exploratorio informado por resultados publicados; sin ajuste de parametros.'}
def exigir(ok,msg):
    if not ok: raise ValueError(msg)
def leerj(p): return json.loads(p.read_text(encoding='utf-8'))
def guardar(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def localizar(root,rel):
    candidatos=[root/rel,root/'campaign'/rel,root/'files/campaign'/rel]
    found=[p for p in candidatos if p.is_file()]
    exigir(len(found)==1,f'Ruta ausente/ambigua: {root} / {rel}')
    return found[0]
def npz(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k].copy() for k in z.files}
def meta(p):
    rows=leerj(p); exigir(isinstance(rows,list),'Metadatos deben ser lista')
    ids=[r['bodyId'] for r in rows]
    exigir(all(type(i) is int for i in ids) and len(set(ids))==len(ids),'IDs invalidos/duplicados')
    return dict(zip(ids,rows))
def lado(r): return r.get('rootSide') or r.get('somaSide') or 'unknown'
def csvfile(p,rows):
    if not rows: return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def contrastes(x):
    L,R,U,S=x
    return {'L_sham':L-S,'R_sham':R-S,'U_sham':U-S,
            'impar_L_R':(L-R)/2,'comun_LR_sham':(L+R)/2-S,
            'R_uniform':R-U,'L_uniform':L-U,'residuo_cuatro':U-L-R+S}
def primero(v,ms):
    ok=(ms>=11)&(v < -PLAN['onset_numerico_atol'])
    n=PLAN['persistencia_muestras']
    for i in range(len(ok)-n+1):
        if ok[i:i+n].all(): return int(ms[i])
    return None
def yaw(q):
    exigir(q.ndim==2 and q.shape[1]>=7,'Forma corporal qpos')
    w,x,y,z=q[:,3:7].T
    exigir(np.max(abs(w*w+x*x+y*y+z*z-1))<1e-5,'Quaternion no unitario')
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))
def analizar(principal,afroot,out):
    out.mkdir(exist_ok=False);guardar(out/'PLAN.json',PLAN); tic=time.perf_counter(); archivos={}
    def archivo(root,rel):
        p=localizar(root,rel);archivos[str(p.resolve())]=sha(p);return p
    try:
        protocol=leerj(archivo(principal,'traces/protocol.json'))
        md=meta(archivo(afroot,'afferent_metadata.json')); lados=meta(archivo(principal,'side_metadata.json'))
        exigir(lados[523769]['instance']=='DNa02_L' and lados[10360]['instance']=='DNa02_R','Identidad DNa02')
        nat={a:npz(archivo(principal,f'traces/{a}_native_transmission.npz')) for a in ARMS}
        tr={a:npz(archivo(principal,f'traces/{a}_trace.npz')) for a in ARMS}
        ref=nat['sham']; ids=ref['afferent_ids']; w=ref['signed_weights']; caps=ref['caps']; ms=ref['times_ms']
        exigir(ids.ndim==1 and ids.dtype.kind in 'iu' and len(set(ids.tolist()))==len(ids),'IDs aferentes')
        exigir(w.shape==(2,len(ids)) and caps.shape==(len(ids),),'Pesos/caps')
        exigir(np.isfinite(w).all() and np.isfinite(caps).all() and (caps>0).all(),'Pesos/caps invalidos')
        exigir(np.array_equal(ms,np.arange(336)),'Se requieren endpoints 0..335 ms')
        exigir(set(map(int,ids))<=set(md),'Metadatos aferentes incompletos')
        ornids=tr['sham']['ORN_ids']; mids=tr['sham']['monitored_ids']
        oside=np.array([lado(lados[int(i)]) for i in ornids])
        exigir(set(oside)=={'L','R'},'Lateralidad ORN incompleta')
        di=[int(np.flatnonzero(mids==i)[0]) for i in (523769,10360)]
        errors={}; initial={}; first_input={}; balances=[]; sn=[]; stages=[]
        for a in ARMS:
            d=nat[a];t=tr[a]
            for k in ('afferent_ids','signed_weights','caps','times_ms'):
                exigir(np.array_equal(d[k],ref[k]),f'{a}: cambio de {k}')
            exigir(np.array_equal(d['target_ids'],[523769,10360]),f'{a}: destinos invertidos')
            exigir(d['transmission'].shape==(336,len(ids)) and d['net'].shape==(336,2) and d['target'].shape==(336,2),'Forma de lectura nativa')
            for k in ('transmission','net','target'): exigir(np.isfinite(d[k]).all(),f'{a}: no finito {k}')
            for k in ('ms','ORN_ids','monitored_ids'): exigir(np.array_equal(t[k],tr['sham'][k]),f'{a}: eje {k}')
            exigir(np.array_equal(t['ms'],ms),'Relojes traza/entrada')
            exigir(t['ORN_q'].shape==(336,len(ornids)) and t['monitored_q'].shape==(336,len(mids)),'Eje neural')
            exigir(t['sensors_used'].shape==(335,3),'Se requieren 335 entradas con tres canales por intervalo')
            for k in ('ORN_q','monitored_q','qpos','sensors_used','neural_steering_applied'):
                exigir(np.isfinite(t[k]).all(),f'{a}: traza no finita {k}')
            s=d['transmission']; pred=(s*caps)@w.T
            errors[a]=float(np.max(abs(pred-d['net'])))
            exigir(errors[a]<=PLAN['reconstruccion_atol'],f'{a}: suma nativa no reconstruida')
            initial[a]=float(np.max(abs(s[:11]-ref['transmission'][:11])))
            ix=np.flatnonzero(np.any(t['sensors_used']!=tr['sham']['sensors_used'],axis=1))
            first_input[a]=None if not len(ix) else [int(ms[ix[0]]),int(ms[ix[0]+1])]
            sn.append(s);balances.append(d['net'][:,0]-d['net'][:,1])
            oq=t['ORN_q'][:,oside=='L'].mean(1)-t['ORN_q'][:,oside=='R'].mean(1)
            stages.append([oq,balances[-1],d['target'][:,0]-d['target'][:,1],
                           t['monitored_q'][:,di[0]]-t['monitored_q'][:,di[1]],
                           t['neural_steering_applied'].reshape(336),yaw(t['qpos'])])
        guardar(out/'ENTRADAS_SHA256.json',archivos)
        lo,hi=PLAN['ventana_ms']; sel=(ms>=lo)&(ms<=hi); sn=np.array(sn); balances=np.array(balances)
        bw=caps*(w[0]-w[1]); labels=np.array([str(md[int(i)].get('type') or 'unknown')+'|'+lado(md[int(i)]) for i in ids])
        groups=np.unique(labels); indexes=[np.flatnonzero(labels==g) for g in groups]
        bygroup=np.stack([np.column_stack([(s[:,ix]*bw[ix]).sum(1) for ix in indexes]) for s in sn])
        exigir(np.max(abs(bygroup.sum(2)-balances))<1e-8,'Balance de grupos inconsistente')
        con=contrastes(bygroup); rows=[]
        for j,g in enumerate(groups):
            r={'grupo':str(g),'ids':';'.join(map(str,ids[indexes[j]])), 'n':len(indexes[j]),
               'baseline_sham':float(bygroup[3,sel,j].mean())}
            r.update({k:float(v[sel,j].mean()) for k,v in con.items()})
            r['primer_impar_negativo_3muestras_ms']=primero(con['impar_L_R'][:,j],ms)
            ix=indexes[j]; ds=(sn[1,sel][:,ix]-sn[3,sel][:,ix]).mean(0)*caps[ix]
            r['R_sham_entrada_L']=float(ds@w[0,ix]);r['R_sham_entrada_R']=float(ds@w[1,ix])
            rows.append(r)
        rows.sort(key=lambda r:(r['impar_L_R'],r['grupo']));csvfile(out/'GRUPOS.csv',rows)
        np.savez_compressed(out/'CONTRASTES.npz',ms=ms,grupos=groups,**con)
        interface=[]; names=['ORN_q_L_R','entrada_DNa02_L_R','target_DNa02_L_R','DNa02_q_L_R','mando','yaw_grados']
        for j,name in enumerate(names):
            x=np.array(stages)[:,j,:]; r={'interfaz':name,**{a:float(x[i,sel].mean()) for i,a in enumerate(ARMS)}}
            r.update({k:float(v[sel].mean()) for k,v in contrastes(x).items()})
            r['primer_impar_negativo_3muestras_ms']=primero(contrastes(x)['impar_L_R'],ms);interface.append(r)
        csvfile(out/'INTERFACES.csv',interface)
        mons=[]
        for j,i in enumerate(mids):
            row=md.get(int(i),lados.get(int(i),{}));x=np.array([tr[a]['monitored_q'][:,j] for a in ARMS])
            r={'id':int(i),'tipo':row.get('type','unknown'),'lado':lado(row)}
            r.update({k:float(v[sel].mean()) for k,v in contrastes(x).items()});mons.append(r)
        csvfile(out/'MONITORIZADOS.csv',mons)
        qdata=npz(archivo(principal,'traces/odor_left_network_trace.npz'))
        deg=np.load(archivo(principal,'outdegree.npy'),allow_pickle=False);q=qdata['q'];tq=qdata['times_ms']
        exigir(q.ndim==2 and q.shape==(len(tq),len(deg)) and len(tq)>1 and np.all(np.diff(tq)>0),'Captura espacial')
        exigir(np.isfinite(q).all() and deg.dtype.kind in 'iu' and (deg>=0).all() and deg.sum()>0,'Estado/grados')
        occ=[]
        for t,v in zip(tq,q):
            active=v>.01;occ.append({'ms':int(t),'q_gt_001':float(active.mean()),'aristas_desde_q_gt_001':float(deg[active].sum()/deg.sum())})
        changes=[]
        for k in range(len(tq)-1):
            change=abs(q[k+1]-q[k])>PLAN['limite_hold_s']
            changes.append({'desde_ms':int(tq[k]),'hasta_ms':int(tq[k+1]),'fraccion_q_no_constante':float(change.mean())})
        hold=[]; edge=(w!=0).sum(0);eps=PLAN['limite_hold_s']
        for arm,s in zip(ARMS,sn):
            for h in PLAN['horizontes_ms']:
                for policy in ('s_baja','historia_casi_constante'):
                    elegidas=fallidas=0;shares=[];maxs=0.;maxinput=0.;maxbal=0.;maxbound=0.
                    for t in range(lo,hi-h+1,h):
                        mask=s[t]<=.01 if policy=='s_baja' else np.ptp(s[t-h:t+1],axis=0)<=eps
                        delta=s[t+1:t+h+1]-s[t];diff=delta[:,mask]
                        bad=np.max(abs(delta),axis=0)>eps
                        elegidas+=int(mask.sum());fallidas+=int((mask&bad).sum());shares.append(float(edge[mask].sum()/edge.sum()))
                        maxs=max(maxs,float(np.max(abs(diff),initial=0)))
                        err=(diff*caps[mask])@w[:,mask].T
                        bd=abs(w[:,mask]*caps[mask])@np.max(abs(diff),axis=0)
                        maxbound=max(maxbound,float(np.max(bd,initial=0)))
                        maxinput=max(maxinput,float(np.max(abs(err),initial=0)))
                        maxbal=max(maxbal,float(np.max(abs(err[:,0]-err[:,1]),initial=0)))
                    hold.append({'brazo':arm,'h_ms':h,'politica':policy,'decisiones_fuente_ventana':elegidas,
                                 'decisiones_fuente_fallidas':fallidas,'fraccion_aristas_DN_candidatas_media':float(np.mean(shares)),
                                 'max_error_s_muestreado':maxs,'max_error_entrada_DN':maxinput,'max_error_balance_DN':maxbal,'cota_entrada_muestreada':maxbound})
        csvfile(out/'HOLD_DIAGNOSTICO.csv',hold)
        total={k:float(v[sel].mean()) for k,v in contrastes(balances).items()}
        result={'estado':'COMPLETO_ANALISIS_OFFLINE','reconstruccion_max':errors,'contrastes_entrada_DN':total,
                'diferencia_aferente_preON_max':initial,'primer_intervalo_input_distinto_sham':first_input,
                'tres_grupos_impar_mas_negativo':rows[:3],'interfaces':interface,'ocupacion':occ,'cambios_q':changes,
                'procedencia':{k:protocol.get(k) for k in ('source','schema','probe_schema','new_operation','preparation_history_clarification')},
                'active_set_certificado':False,'causa_identificada':False,'stage3_admission':False,
                'limites':'Hold usa pasado para elegir; evalua solo salidas grabadas, no integra recurrencia ni acota entre muestras. Uniforme no garantiza dosis igual. No p-valores ni replicas.',
                'wall_s':time.perf_counter()-tic}
        guardar(out/'ENTRADAS_SHA256.json',archivos);guardar(out/'RESULTADO.json',result);return result
    except Exception:
        guardar(out/'FALLO.json',{'error':traceback.format_exc(),'archivos':archivos});raise

def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        r=Path(tmp);p=r/'p/campaign';a=r/'a/campaign';(p/'traces').mkdir(parents=True);a.mkdir(parents=True)
        ids=np.array([101,102,103]);ms=np.arange(336);w=np.array([[2.,.5,-1.],[0.,1.,0.]])
        sides=[{'bodyId':900,'rootSide':'L'},{'bodyId':901,'rootSide':'R'},
               {'bodyId':523769,'instance':'DNa02_L'},{'bodyId':10360,'instance':'DNa02_R'}]
        guardar(p/'side_metadata.json',sides);guardar(a/'afferent_metadata.json',[{'bodyId':int(i),'type':'G'+str(j),'rootSide':'L'} for j,i in enumerate(ids)])
        guardar(p/'traces/protocol.json',{'schema':'FIXTURE_SINTETICO_NO_ORGANISMO','duration_ms':335})
        ramp=np.where(ms>=11,1-np.exp(-(ms-10)/30),0.)
        coeff=[[.012,.04,.004],[.07,.02,.003],[.08,.035,.0035],[0.,0.,0.]]
        for arm,c in zip(ARMS,coeff):
            s=np.array([.05,.04,.003])[None,:]+ramp[:,None]*c
            s[:,2]+=.001*(1+np.tanh((ms-200)/2))  # Cambio continuo bajo umbral de amplitud.
            net=(s*10)@w.T;target=np.tanh(net)
            native=dict(afferent_ids=ids,signed_weights=w,caps=np.ones(3)*10,times_ms=ms,target_ids=np.array([523769,10360]),transmission=s,net=net,target=target)
            np.savez(p/f'traces/{arm}_native_transmission.npz',**native)
            orn=np.full((336,2),.1);orn+=ramp[:,None]*({'odor_left':[.4,0],'odor_right':[0,.4],'uniform':[.4,.4],'sham':[0,0]}[arm])
            angle=ms*.0001;qpos=np.zeros((336,7));qpos[:,3]=np.cos(angle/2);qpos[:,6]=np.sin(angle/2)
            trace=dict(ms=ms,ORN_ids=np.array([900,901]),ORN_q=orn,monitored_ids=np.array([523769,10360]),monitored_q=.2+.01*net,
                       qpos=qpos,neural_steering_applied=np.ones(336)*.1,sensors_used=np.column_stack((orn[1:]-.1,np.zeros(335))))
            np.savez(p/f'traces/{arm}_trace.npz',**trace)
        tq=np.array([0,1,20,40,60,100,160,220,320,335]);q=np.outer(tq/335,np.linspace(0,.8,8))
        np.savez(p/'traces/odor_left_network_trace.npz',times_ms=tq,q=q);np.save(p/'outdegree.npy',np.arange(1,9,dtype=np.int64))
        result=analizar(r/'p',r/'a',r/'out');exigir(result['contrastes_entrada_DN']['impar_L_R']<0,'Signo perdido')
        exigir(result['tres_grupos_impar_mas_negativo'][0]['grupo']=='G0|L','Grupo erroneo')
        with (r/'out/HOLD_DIAGNOSTICO.csv').open() as f:
            exigir(any(int(x['decisiones_fuente_fallidas'])>0 for x in csv.DictReader(f)),'Hold no detectado')
        path=p/'traces/odor_right_native_transmission.npz';original=path.read_bytes();caught=[]
        for key in ('times_ms','target_ids','transmission'):
            d=npz(path)
            if key=='times_ms':d[key][50]+=1
            elif key=='target_ids':d[key]=d[key][::-1]
            else:d[key][50,0]+=.2
            np.savez(path,**d)
            try:analizar(r/'p',r/'a',r/('mal_'+key))
            except ValueError:caught.append(key)
            else:raise AssertionError('Corrupcion aceptada: '+key)
            path.write_bytes(original)
        return {'fixture':'SINTETICO; sin arrays de OpenMatrix','contrastes_y_suma':True,'hold_refutado_en_fixture':True,'corrupciones_detectadas':caught}
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--principal',type=Path);parser.add_argument('--aferentes',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.selftest:print(json.dumps(selftest(),indent=2,ensure_ascii=False))
    else:
        exigir(all((args.principal,args.aferentes,args.out)),'Se requieren --principal --aferentes --out')
        res=analizar(args.principal,args.aferentes,args.out)
        print(json.dumps({k:res[k] for k in ('estado','reconstruccion_max','contrastes_entrada_DN','active_set_certificado','wall_s')},indent=2))
