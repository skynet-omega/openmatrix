"""Bounded author-exposed engineering pilots, full-array external error tests."""
from pathlib import Path
import sys,time,json,hashlib,argparse,resource
import numpy as np
from scipy.integrate import solve_ivp
from model import Model,require
from development import state,scalar,hh
from extensions import fixture

def manufactured(n=32):
    w=np.linspace(1,4,n);ph=np.linspace(.1,1,n);rr=[];cc=[];vv=[]
    for i in range(n):
        for j in range(max(0,i-1),min(n,i+2)):rr.append(i);cc.append(j);vv.append(1 if i==j else .2)
    p={'id':'manufactured','count':n,'states':{'x':state('1',np.sin(ph).tolist(),1,'velocity-k*(x-z)-kc*(x-z)**3')},
       'parameters':{'omega':scalar('1/s',w.tolist()),'phase':scalar('1',ph.tolist()),'k':scalar('1/s',np.geomspace(1,1e6,n).tolist()),'kc':scalar('1/s',1)},
       'derived':{'z':{'unit':'1','expr':'sin(omega*t+phase)'}},'inputs':{'velocity':scalar('1/s',0)},'outputs':{'zdot':{'unit':'1/s','expr':'omega*cos(omega*t+phase)'}}}
    return {'version':1,'populations':[p],'mass':{'row':rr,'col':cc,'values':vv},'connections':[{'source':['manufactured','zdot'],'target':['manufactured','velocity'],'pattern':'coo','row':rr,'col':cc,'weights':vv}]}
def case(name):
    if name=='clamp':
        return {'version':1,'populations':[{'id':'p','count':2,'states':{'x':state('1',[0,0],1,'r')},'parameters':{'r':scalar('1/s',[0,1])}}], 'mass':{'row':[0,0,1,1],'col':[0,1,0,1],'values':[2,1,1,2]},'clamps':[{'population':'p','state':'x','cells':[0],'value':0}]},1.
    if name=='stiff':return manufactured(),1.
    if name=='hh':return hh(3),.05
    if name=='mixed':return fixture(),.2
    raise ValueError(name)
def exact(name,t):
    if name=='clamp':return np.array([0*t,.5*t])
    if name=='stiff':return np.sin(np.linspace(1,4,32)[:,None]*t+np.linspace(.1,1,32)[:,None])
    return None
def main():
    p=argparse.ArgumentParser();p.add_argument('out');p.add_argument('--case',required=True);p.add_argument('--profile',choices=['fast','precise']);p.add_argument('--reference',action='store_true');p.add_argument('--reference-dir');p.add_argument('--implementation',default='native',choices=['native','python']);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();spec,end=case(a.case);m=Model(spec);times=np.linspace(0,end,21)
    (out/'model.json').write_text(json.dumps(spec,indent=2)+'\n')
    ctx={'case':a.case,'implementation':a.implementation,'model_identity':m.identity,'profile':a.profile,'end':end,'states':m.n,'connections':m.connection.nnz,'mass_nnz':m.mass.nnz,'times':times.tolist()}
    (out/'CONTEXT.json').write_text(json.dumps(ctx,indent=2)+'\n')
    if a.reference:
        rows=[]
        for rtol,atol in [(1e-10,1e-12),(1e-12,1e-14)]:
            r=solve_ivp(m.rhs,(0,end),m.initial,method='Radau',t_eval=times,rtol=rtol,atol=atol*m.scale)
            require(r.success,'reference failed');rows.append(r.y)
        change=float(np.max(abs(rows[1]-rows[0])/(m.scale[:,None]+abs(rows[1]))));require(change<1e-7,'reference unresolved')
        np.savez_compressed(out/'reference.npz',times=times,loose=rows[0],tight=rows[1],scale=m.scale)
        result=ctx|{'refinement_error':change,'wall_s':time.perf_counter()-begin}
    else:
        from implicit import Implicit
        e=Implicit(spec,a.profile,implementation=a.implementation);ys=[e.read()];start=time.perf_counter();failure=None
        try:
            for t in times[1:]:e.advance(float(t),wall_limit_s=max(.01,180-(time.perf_counter()-begin)));ys.append(e.read())
        except Exception as exc:failure=repr(exc)
        elapsed=time.perf_counter()-start;stats=e.stats();e.close();y=np.array(ys).T
        ref=exact(a.case,times);refsha=None
        if ref is None:
            rp=Path(a.reference_dir)/'reference.npz';refsha=hashlib.sha256(rp.read_bytes()).hexdigest()
            with np.load(rp,allow_pickle=False) as z:
                require(np.array_equal(z['times'],times),'reference grid');ref=z['tight']
        err=float(np.max(abs(y-ref[:,:y.shape[1]])/(m.scale[:,None]+abs(ref[:,:y.shape[1]]))))
        eligible=(failure is None and y.shape==ref.shape and np.isfinite(y).all() and err<=({'fast':.01,'precise':1e-5}[a.profile]))
        np.savez_compressed(out/'trajectory.npz',times=times[:y.shape[1]],candidate=y,reference=ref[:,:y.shape[1]],scale=m.scale)
        result=ctx|{'setup_s':e.setup_s,'advance_scan_s':elapsed,'max_scaled_error':err,'eligible':bool(eligible),'failure':failure,'stats':stats,'reference_sha256':refsha,'measured_window_s':time.perf_counter()-begin,'host_peak_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
