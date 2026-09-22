"""Targeted tests prompted by external review, not independent confirmation."""
from pathlib import Path
import sys,json,copy,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from model import Model,require
from runtime import Engine
from development import base,scalar,state
from extension_test import freeze_check

def main():
    freeze_check();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    # ChatGPT's two-state mass/clamp falsifier. The clamped row changes BEFORE solving M.
    spec={'version':1,'populations':[{'id':'test','count':2,'states':{'x':state('1',[0,0],1,'drive')},'parameters':{'drive':scalar('1/s',[0,1])}}],
          'mass':{'row':[0,0,1,1],'col':[0,1,0,1],'values':[2,1,1,2]},
          'clamps':[{'population':'test','state':'x','cells':[0],'value':0}]}
    e=Engine(spec,backend='cpu');slope=e.model.rhs(0,e.read());require(np.max(abs(slope-[0,.5]))<1e-14,'non-diagonal clamp incorrect')
    e.advance(.1);require(np.max(abs(e.read()-[0,.05]))<1e-12,'clamp trajectory incorrect')
    # Manufactured stiff 32-state system, M not diagonal. z(t) is analytically known.
    n=32;omega=np.linspace(1,4,n);phase=np.linspace(.1,1,n);initial=np.sin(phase)
    rows=[];cols=[];vals=[]
    for i in range(n):
        for j in range(max(0,i-1),min(n,i+2)):rows.append(i);cols.append(j);vals.append(1 if i==j else .2)
    p={'id':'manufactured','count':n,'states':{'x':state('1',initial.tolist(),1,'velocity-k*(x-z)-kc*(x-z)**3')},
       'parameters':{'omega':scalar('1/s',omega.tolist()),'phase':scalar('1',phase.tolist()),'k':scalar('1/s',np.geomspace(1,1e6,n).tolist()),'kc':scalar('1/s',1)},
       'derived':{'z':{'unit':'1','expr':'sin(omega*t+phase)'}},'inputs':{'velocity':scalar('1/s',0)},
       'outputs':{'zdot':{'unit':'1/s','expr':'omega*cos(omega*t+phase)'}}}
    spec={'version':1,'populations':[p],'mass':{'row':rows,'col':cols,'values':vals},'connections':[{'source':['manufactured','zdot'],'target':['manufactured','velocity'],'pattern':'coo','row':rows,'col':cols,'weights':vals}]}
    m=Model(spec);e=Engine(spec,backend='cpu');e.cpu_method='Radau';times=np.linspace(0,.1,11);ys=[e.read()]
    for t in times[1:]:e.advance(float(t));ys.append(e.read())
    y=np.array(ys).T;exact=np.sin(omega[:,None]*times+phase[:,None]);err=float(np.max(abs(y-exact)/(1+abs(exact))))
    require(err<=1e-5,'manufactured stiff trajectory fails')
    np.savez_compressed(out/'manufactured.npz',times=times,candidate=y,reference=exact)
    # Scanning must not affect controller/state; captured buffers have persistent ownership.
    spec=base(32);a=Engine(spec);b=Engine(spec)
    for t in np.linspace(.001,.02,20):
        a.advance(float(t));b.advance(float(t))
        for _ in range(3):a.scan('cells','v',[0,7,23])
    require(np.array_equal(a.read(),b.read()) and a.next_h==b.next_h,'scanner changes simulation')
    # Active model replaced vs independently restarting the exact declared map.
    a=Engine(spec);a.advance(.01);old=a.read();changed=copy.deepcopy(spec);changed['connections']=[]
    b=Engine(changed);b.gpu.write(old);b.t=.01;b.next_h=1e-4;a.replace(changed)
    a.advance(.03);b.advance(.03);require(np.array_equal(a.read(),b.read()),'live edit differs from separate segments')
    result={'source':'External advice from ChatGPT, locally executed author test','mass_clamp_derivative':slope.tolist(),'manufactured32_stiff_scaled_error':err,'scanner_no_effect_bit_exact':True,'edit_equal_separate_segments_bit_exact':True,'core_matches_registered_repair':True,'wall_s':time.perf_counter()-start}
    freeze_check();(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
