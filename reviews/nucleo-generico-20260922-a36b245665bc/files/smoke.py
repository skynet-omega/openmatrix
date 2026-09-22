from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from model import Model
from runtime import Engine
from development import base,hh
for name,spec in [('base',base()),('hh',hh())]:
    start=time.perf_counter();e=Engine(spec);ref=Model(spec)
    a=ref.rhs(0,ref.initial);b=e.gpu.rhs_read(0,ref.initial)
    print(name,'RHS',np.max(np.abs(a-b)/(1+np.abs(a))), 'setup',e.setup_s,flush=True)
    e.advance(.01);print(name,'t',e.t,'finite',np.isfinite(e.read()).all(),'accepted',e.gpu.accepted,'rejected',e.gpu.rejected,'wall',time.perf_counter()-start,flush=True)
