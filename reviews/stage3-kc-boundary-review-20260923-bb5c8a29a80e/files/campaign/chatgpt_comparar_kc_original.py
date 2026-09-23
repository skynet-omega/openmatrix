"""Revisión CPU del snapshot a1ab767. No integra neuronas ni ejecuta CUDA.
--campaign lee GATE, COMPACT_NUMERIC y ambos EVENT_AUDIT sin modificar fuentes.
--physical compila el consumidor CUDA publicado como función escalar CPU.
"""
import argparse, ctypes, hashlib, json, os, subprocess, sys, time, traceback, zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
FULL=125000
CASES=('native_sham_20_01','reference_sham_20_01')
SOURCE_SHA='f0ed6544052c6e2c872b7ef97ef74b33fcbc69b29d3aaf936c6eba078342351f'
GATE_SHA='ebff3939f1d6b4fd10c0a123a0396c08c0c2c920d621e0d429443044e3823cd5'
PLAN={'continuous_abs_limit':1e-4,'derivative_fixture_atol':1e-7,
      'event_matching':'ordinal por productor/fila/ID; no búsqueda de lag',
      'event_time_gate':'ningún umbral nuevo; reportar diferencia cruda y redondeada',
      'scope':'endpoint KC final y eventos somáticos; no historia axonal completa'}
def require(ok,msg):
    if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def audit(path):
    d=js(path); blocks=d['blocks']; require(blocks and len(blocks)%2==0,'Pares predictor/aceptado incompletos')
    seq={'predictor':defaultdict(list),'accepted':defaultdict(list)}; starts=[]; mapping={}
    for i in range(0,len(blocks),2):
        p,b=blocks[i:i+2]; start=b['start_elapsed_ns']
        require(p['duration_ns']==FULL//2 and b['duration_ns']==FULL,'Duración no declarada')
        require(type(start) is int and p['start_elapsed_ns']==start,'Inicio predictor distinto')
        if starts: require(start==starts[-1]+FULL,'Hueco/repetición de bloques')
        starts.append(start)
        for kind,block in (('predictor',p),('accepted',b)):
            for e in block['events']:
                r,n=e['row'],e['neuron_id']; require(type(r) is int and type(n) is int and min(r,n)>=0,'Identidad inválida')
                require(r not in mapping or mapping[r]==n,'Fila reasignada a otro ID'); mapping[r]=n
                t=float(e['time_s']);jump=float(e['jump']);post=e['post_q']
                require(np.isfinite([t,jump]).all() and 0<=t<=block['duration_ns']*1e-9,'Evento fuera de bloque')
                require(post is None or np.isfinite(post),'Post no finito')
                key=(e['producer'],r,n); rel=(start-blocks[0]['start_elapsed_ns'])+t*1e9
                seq[kind][key].append({'elapsed_ns':rel,'rounded_ns':start+round(t*1e9),
                    'post':post,'jump':jump,'block_start_ns':start})
    for group in seq.values():
        for events in group.values(): events.sort(key=lambda e:e['elapsed_ns'])
    return seq,starts

def pair_events(a,b):
    missing=[]; rows=[]
    for key in sorted(set(a)|set(b)):
        aa,bb=a.get(key,[]),b.get(key,[])
        if len(aa)!=len(bb): missing.append({'key':list(key),'n_native':len(aa),'n_reference':len(bb)});continue
        for ordinal,(x,y) in enumerate(zip(aa,bb)):
            rows.append({'producer':key[0],'row':key[1],'id':key[2],'ordinal':ordinal,
                'native_elapsed_ns':x['elapsed_ns'],'reference_elapsed_ns':y['elapsed_ns'],
                'time_abs_ns':abs(x['elapsed_ns']-y['elapsed_ns']),
                'rounded_abs_ns':abs(x['rounded_ns']-y['rounded_ns']),
                'post_abs':None if x['post'] is None or y['post'] is None else abs(x['post']-y['post']),
                'jump_abs':abs(x['jump']-y['jump'])})
    changed=[r for r in rows if r['time_abs_ns'] or r['jump_abs'] or (r['post_abs'] or 0)]
    first=min(changed,key=lambda r:min(r['native_elapsed_ns'],r['reference_elapsed_ns'])) if changed else None
    return {'paired':len(rows),'count_mismatch':missing,'first_recorded_difference':first,
            'max_time_abs_ns':max((r['time_abs_ns'] for r in rows),default=None),
            'max_rounded_abs_ns':max((r['rounded_abs_ns'] for r in rows),default=None),
            'max_post_abs':max((r['post_abs'] for r in rows if r['post_abs'] is not None),default=None)},rows

def compare(folder,out):
    gate=js(folder/'GATE.json');require(sha(folder/'GATE.json')==GATE_SHA,'GATE no corresponde al snapshot')
    p=folder/'COMPACT_NUMERIC.npz';require(sha(p)==gate['compact_sha256'],'Compacto modificado')
    with zipfile.ZipFile(p) as z: require(sum(x.file_size for x in z.infolist())<128*1024**2,'Compacto excede límite')
    with np.load(p,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
    require(np.array_equal(arrays['ids'],[10176,10208,10360,523769,10065,10118]),'IDs focales')
    clock=arrays['time_ns'];require(clock.shape==(20,) and clock.dtype.kind in 'iu','Reloj focal')
    require(np.all(clock[1:]-clock[:-1]==1000000),'Muestreo focal distinto')
    kc=[]
    for path,expected in gate['kc_errors'].items():
        key=path.replace('/','__');x,y=(arrays[key+'__'+suffix] for suffix in ('native','reference'))
        require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'KC inválido')
        d=abs(x-y);idx=np.unravel_index(np.argmax(d),d.shape); maximum=float(d[idx])
        require(list(map(int,idx))==expected['index'] and maximum==expected['max_abs'],'Máximo KC distinto')
        kc.append({'path':path,'shape':list(x.shape),'local_index':list(map(int,idx)),
            'max_abs':maximum,'values':[float(x[idx]),float(y[idx])],
            'above_frozen_limit':int((d>PLAN['continuous_abs_limit']).sum()),
            'worst_value_ratio':None if y[idx]==0 else float(x[idx]/y[idx]),
            'neuron_id':None,'first_time':None})
    focal=[]
    for i,n in enumerate(arrays['ids']):
        x,y=arrays['native_target'][:,i],arrays['reference_target'][:,i]
        require(x.shape==y.shape==(20,) and np.isfinite(x).all() and np.isfinite(y).all(),'Target inválido')
        d=abs(x-y);require(float(d.max())==gate['per_id'][str(n)]['target_max_abs_between_engines'],'Target distinto')
        first=np.flatnonzero(d>0)
        focal.append({'id':int(n),'max_abs':float(d.max()),'first_nonidentical_endpoint_ns':int(clock[first[0]]) if len(first) else None})
    logs=[];starts=[];inputs={'compact':sha(p),'gate':sha(folder/'GATE.json')}
    for name in CASES:
        p=folder/name/'EVENT_AUDIT.json';digest=sha(p)
        require(digest==gate['source_sha256'][name]['EVENT_AUDIT.json'],'Auditoría alterada')
        log,st=audit(p);logs.append(log);starts.append(st);inputs[name]=digest
    require(starts[0]==starts[1] and len(starts[0])*FULL==20000000,'Horizonte de eventos diferente')
    compared={}
    for kind in ('predictor','accepted'):
        summary,rows=pair_events(logs[0][kind],logs[1][kind]);compared[kind]=summary
        summary['total_native']=sum(map(len,logs[0][kind].values()));summary['total_reference']=sum(map(len,logs[1][kind].values()))
        save(out/(kind+'_pairs.json'),rows)
    frozen=gate['accepted_events'];ac=compared['accepted']
    require(ac['max_rounded_abs_ns']==frozen['maximum_event_time_abs_ns'],'Desfase publicado no reconstruido')
    require(ac['total_native']==frozen['committed_events_causal'] and ac['total_reference']==frozen['committed_events_reference'],'Conteos distintos')
    result={'scope':'Arrays finales y eventos REALES; no reejecución del organismo','inputs':inputs,'kc':kc,'focal':focal,'events':compared,
        'event_clock_origin_ns':starts[0][0],'first_KC_state_divergence':'NO_IDENTIFICABLE: faltan historias KC y mapa fila-local/ID',
        'frozen_gate':gate['numerical_gate'],'stage3_admission':False}
    save(out/'COMPARACION.json',result);return result

HEADER='''#include <cmath>
using std::isfinite;
#define __global__
struct Dim { int x; }; static Dim blockIdx{0},blockDim{1},threadIdx{0};
inline int atomicOr(int*p,int v){int old=*p;*p|=v;return old;}
'''
WRAPPER='''
extern "C" void replay(const long long*t,const double*v,int n,double*out){
 int coord=0,flag=0;double q=.2,s=.1,last=v[0],slope=0,trough=v[0];
 double cap=100,tau=.024,gain=1;long long count=0,clip=0;
 for(int k=0;k<n;k++){
  if(k){long long clock[2]={t[k-1],2*(t[k]-t[k-1])};
   commit_axon(1,1,1,2,clock,&coord,v+k,0,.005,&cap,&tau,&gain,-40.,20.,
    &q,&s,&last,&slope,&trough,&count,&clip,&flag);}
  double row[8]={q,s,last,slope,trough,double(count),double(clip),double(flag)};
  for(int j=0;j<8;j++)out[8*k+j]=row[j];
 }
}
'''
def fixture(source,out):
    require(sha(source)==SOURCE_SHA,'Fuente CUDA distinta de la publicada')
    cpp=out/'host_consumer.cpp';cpp.write_text(HEADER+source.read_text()+WRAPPER,encoding='utf-8')
    lib=out/'host_consumer.so';cmd=['g++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',str(cpp),'-o',str(lib)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=15);(out/'COMPILACION.txt').write_text(r.stdout+r.stderr)
    require(r.returncode==0,'Falló compilación CPU');dll=ctypes.CDLL(str(lib))
    dll.replay.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    def run(name,t,f):
        t=np.ascontiguousarray(t,dtype=np.int64);v=np.ascontiguousarray(f(t),dtype=np.float64);a=np.empty((len(t),8))
        dll.replay(t.ctypes.data,v.ctypes.data,len(t),a.ctypes.data);require(np.isfinite(a).all() and not a[:,7].any(),'Fixture no finito')
        np.savez(out/(name+'.npz'),time_ns=t,voltage=v,output=a);return a
    coarse=np.arange(0,10001,2000);fine=np.arange(0,10001,500);fun=lambda t:-65.+t*1e-5
    x,y=run('linear_coarse',coarse,fun),run('linear_fine',fine,fun)
    derivative=abs(x[-1,3]/2e-6-y[-1,3]/.5e-6)
    require(x[-1,2]==y[-1,2] and derivative<PLAN['derivative_fixture_atol'],'Control lineal')
    fun=lambda t:-66.-.04*((t-5000.)/1000.)**2
    p=run('peak_coarse',[0,3000,5500],fun);q=run('peak_fine',[0,1000,2000,3000,4000,5000,5500],fun)
    require(p[-1,2]==q[-1,2] and p[-1,5]==q[-1,5]==0,'Control de pico sin espiga')
    require(abs(p[-1,4]-q[-1,4])>PLAN['continuous_abs_limit'],'No ejercitó diferencia de trough')
    result={'source_sha256':SOURCE_SHA,'compiler':cmd,'linear_raw_slopes':[float(x[-1,3]),float(y[-1,3])],
        'linear_normalized_derivative_diff':derivative,'peak_final_voltage':[float(p[-1,2]),float(q[-1,2])],
        'peak_trough':[float(p[-1,4]),float(q[-1,4])],'peak_counts':[int(p[-1,5]),int(q[-1,5])],
        'scope':'Fuente de commit_axon recompilada CPU escalar; voltajes sintéticos exactos, NO CUDA ni integrador',
        'CUDA_executed':False,'organism_executed':False}
    save(out/'FIXTURE.json',result);return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--campaign',type=Path);ap.add_argument('--physical',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    require(bool(a.campaign)^bool(a.physical),'Elegir --campaign o --physical');a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'PLAN.json',PLAN);wall=time.perf_counter();cpu=time.process_time();result={}
    try:
        result['data']=compare(a.campaign,a.out) if a.campaign else fixture(a.physical,a.out)
        result['execution']='COMPLETO'
    except Exception:result.update(execution='FALLO_CONSERVADO',error=traceback.format_exc())
    result.update(wall_s=time.perf_counter()-wall,cpu_s=time.process_time()-cpu,code_sha256=sha(Path(__file__)))
    save(a.out/'RESULTADO.json',result);print(json.dumps(result,indent=2,ensure_ascii=False));return 0 if result['execution']=='COMPLETO' else 1
if __name__=='__main__':sys.exit(main())
