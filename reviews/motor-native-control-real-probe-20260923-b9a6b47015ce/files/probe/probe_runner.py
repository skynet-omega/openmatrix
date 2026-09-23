"""Local harness for the unchanged external C++ probe; never modifies biology."""
from pathlib import Path
import ctypes as ct,hashlib,json,sys,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'campanas/etapa3_motor_nuevo_20260922'))
import graph_core

def main():
    mode=int(sys.argv[1]);out=Path(sys.argv[2]).resolve()
    if mode not in (-1,0,1):raise ValueError('Unknown probe mode')
    library=HERE/'generated'/('libgraph_control_parent.so' if mode==-1 else 'libgraph_control_probe.so')
    original_init=graph_core.NativeGraph.__init__;original_close=graph_core.NativeGraph.close
    setup=[]
    def initialize(self,*args,**kwargs):
        kwargs['native_library']=library;original_init(self,*args,**kwargs)
        self._diagnostic_probe_armed=False
        if mode!=-1:
            t=time.perf_counter();lib=self.lib
            lib.engine_probe_arm.argtypes=[ct.c_void_p,ct.c_int,ct.c_int];lib.engine_probe_arm.restype=ct.c_int
            lib.engine_probe_save.argtypes=[ct.c_void_p,ct.c_char_p];lib.engine_probe_save.restype=ct.c_int
            lib.engine_probe_clear.argtypes=[ct.c_void_p];lib.engine_probe_clear.restype=ct.c_int
            if lib.engine_probe_arm(self.handle,512,mode)!=0:raise RuntimeError(lib.engine_error().decode())
            self._diagnostic_probe_armed=True;setup.append({'pool_setup_s':time.perf_counter()-t})
    def close(self):
        try:
            if getattr(self,'_diagnostic_probe_armed',False):
                t=time.perf_counter()
                try:
                    if self.lib.engine_probe_save(self.handle,str(out/'SONDA.json').encode())!=0:raise RuntimeError('Native probe export failed')
                finally:
                    code=self.lib.engine_probe_clear(self.handle);self._diagnostic_probe_armed=False
                    if code!=0:raise RuntimeError('Native probe clear failed')
                setup[-1]['export_clear_s']=time.perf_counter()-t
        finally:original_close(self)
    graph_core.NativeGraph.__init__=initialize;graph_core.NativeGraph.close=close
    sys.path.insert(0,str(HERE))
    import run_set
    sys.argv=[str(HERE/'run_set.py'),'--out',str(out),'--odor','sham','--engine','causal_cuda','--ms','1','--observe','off']
    try:code=run_set.main()
    finally:
        if out.is_dir():
            (out/'PROBE_SETUP.json').write_text(json.dumps({'mode':mode,'setup':setup,
               'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
               'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
        graph_core.NativeGraph.__init__=original_init;graph_core.NativeGraph.close=original_close
    raise SystemExit(code)
if __name__=='__main__':main()
