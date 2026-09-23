from pathlib import Path
import sys,time,json,os,hashlib
H=Path(__file__).resolve().parent;ROOT=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
os.environ['NUMBA_CACHE_DIR']=str(H/'numba_cache')
sys.path[:0]=[str(ROOT/'work/motor13_20260922'),str(ROOT/'src')]
import motor_runtime
from threadpoolctl import threadpool_limits

def main():
    out=H/'inspection_01';start=time.perf_counter()
    with threadpool_limits(limits=1,user_api='blas'):
        obj,*_=motor_runtime.load(out)
        try:
            import cupy as cp
            h=obj.core.hybrid;pn=h._online_source.pn
            chain=[];p=pn
            for _ in range(10):
                chain.append({'type':type(p).__name__,'keys':list(p.__dict__)})
                if not hasattr(p,'base'):break
                p=p.base
            r={'load_s':time.perf_counter()-start,'CNS_state':len(h.state),'CNS_time_ns':h.time_ns,'class_mro':[c.__name__ for c in type(h).__mro__],'PN_chain':chain,'coupling_ns':h.pn_online_manifest['coupling_ns'],'kc_schedule':{k:h.kc_spatial_manifest.get(k) for k in ['inner_step_ns','coupling_step_ns']},'params':h.parameters,'source_type':type(h._online_source).__name__,'source_keys':list(h._online_source.__dict__),'pn_base_type':type(p).__name__,'body_dt':obj.body.dt,'gpu_pool_bytes':cp.get_default_memory_pool().total_bytes()}
            motor_runtime.dump(out/'INSPECTION.json',r);print(json.dumps(r,indent=2),flush=True)
        finally:obj.close()
if __name__=='__main__':main()
