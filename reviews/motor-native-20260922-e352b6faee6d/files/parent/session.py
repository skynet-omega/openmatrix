"""Epoch operations for the implicit prototype.
A saved physical state restarts adaptation; it is explicitly NOT an exact solver checkpoint.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from model import Model,require,digest,integer_array
from implicit import Implicit
from runtime import PROFILES
class Session:
    def __init__(self,spec,profile='precise'):
        self.effective_tolerances=tuple(PROFILES[profile]);self.solver=Implicit(spec,profile);self.edits=[]
    @property
    def t(self):return self.solver.t
    @property
    def model(self):return self.solver.model
    def advance(self,end,wall_limit_s=180):return self.solver.advance(end,wall_limit_s)
    def read(self):return self.solver.read()
    def scan(self,population,state,cells=None):
        p=self.model.by_id[population];sl=p['states'][state];ids=p['ids'];idx=np.arange(sl.start,sl.stop)
        if cells is not None:
            lookup={int(v):i for i,v in enumerate(ids)};positions=[lookup[int(v)] for v in integer_array(cells,'scan cells')];ids=ids[positions];idx=idx[positions]
        with self.solver.stream:values=self.solver.x[idx].get(stream=self.solver.stream)
        return {'time':self.t,'population':population,'state':state,'cell_ids':ids.tolist(),'values':values.tolist(),'model_identity':self.model.identity}
    def replace(self,spec):
        # Old integrator/callback ownership stays live until the replacement fully initializes.
        m=Model(spec,prepare_mass_solver=False);old=self.read();new=m.initial.copy()
        for name,p in m.by_id.items():
            if name not in self.model.by_id:continue
            op=self.model.by_id[name];_,oi,ni=np.intersect1d(op['ids'],p['ids'],return_indices=True)
            for key,sl in p['states'].items():
                if key in op['states']:
                    osl=op['states'][key];require(m.units[sl.start]==self.model.units[osl.start],'state unit changed')
                    new[sl.start+ni]=old[osl.start+oi]
        new[m.clamp_mask]=m.clamp_values[m.clamp_mask];m.raw_rhs(self.t,new)
        require(tuple(PROFILES[self.solver.profile])==self.effective_tolerances,"profile changed during live session; explicit migration required")
        fresh=Implicit(spec,self.solver.profile,self.t,new)
        receipt={'time':self.t,'before':self.model.identity,'after':fresh.model.identity,'kind':'new integration epoch; adaptation restarted'}
        previous=self.solver;self.solver=fresh;previous.close();self.edits.append(receipt);return receipt
    def contract(self):
        import cupy as cp
        h=Path(__file__).resolve().parent
        return {'version':1,'mode':'physical_restart_new_adaptation','profile':self.solver.profile,'rtol':self.effective_tolerances[0],'atol_scale':self.effective_tolerances[1],
                'sources':{f:hashlib.sha256((h/f).read_bytes()).hexdigest() for f in ['model.py','runtime.py','autodiff.py','coupled.py','implicit.py','bridge.cpp','session.py','DEPENDENCY.json']},
                'numpy':np.__version__,'cupy':cp.__version__,'CUDA_runtime':cp.cuda.runtime.runtimeGetVersion(),'library_sha256':hashlib.sha256((h/'libopenmatrix_ark.so').read_bytes()).hexdigest()}
    def save_restart(self,path):
        p=Path(path);p.mkdir(parents=True,exist_ok=False);np.save(p/'state.npy',self.read())
        meta={'contract':self.contract(),'spec':self.model.spec,'identity':self.model.identity,'time':self.t,'edits':self.edits,'state_sha256':hashlib.sha256((p/'state.npy').read_bytes()).hexdigest(),'unsupported':'No serialization of SUNDIALS adaptive/Newton/Krylov history. Restart is a declared new epoch.'}
        raw=(json.dumps(meta,indent=2,allow_nan=False)+'\n').encode();(p/'restart.json').write_bytes(raw);(p/'restart.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
    @classmethod
    def restore_restart(cls,path):
        p=Path(path);raw=(p/'restart.json').read_bytes();require(hashlib.sha256(raw).hexdigest()==(p/'restart.sha256').read_text().strip(),'restart metadata corruption');meta=json.loads(raw)
        require(meta['contract']['mode']=='physical_restart_new_adaptation','restart semantics mismatch')
        require(hashlib.sha256((p/'state.npy').read_bytes()).hexdigest()==meta['state_sha256'],'restart state corruption')
        state=np.load(p/'state.npy',allow_pickle=False);require(state.dtype==np.float64,'restart dtype')
        obj=cls.__new__(cls);obj.effective_tolerances=tuple(PROFILES[meta['contract']['profile']]);obj.solver=Implicit(meta['spec'],meta['contract']['profile'],meta['time'],state);obj.edits=meta['edits']
        try:require(obj.contract()==meta['contract'] and obj.model.identity==meta['identity'],'restart executable contract changed; explicit migration required')
        except Exception:obj.close();raise
        return obj
    def close(self):self.solver.close()
