"""Candidate B: preserve a nominal proposal after a low-error forced cut."""
from pathlib import Path
import sys,subprocess,hashlib,json
T=Path(__file__).resolve().parent
sys.path.insert(0,str(T.parents[1]/'campanas/etapa3_motor_nuevo_20260922'))
from graph_core import NativeGraph

def build():
 parent=T.parent/'pipeline_review_20260922/graph_control_trace.cpp';code=parent.read_text()
 old='*next=std::min(maxstep,std::max(minstep,(long)std::floor(h*1e9*(e<.1?2:1))));'
 new='''long grown=(long)std::floor(h*1e9*(e<.1?2:1));
    // A forcing boundary is not an accuracy-driven reduction of the proposal.
    // The next full attempt still executes the unchanged error/domain checks.
    if(h<requested && e<.1)grown=std::max(grown,std::min(*next,maxstep));
    *next=std::min(maxstep,std::max(minstep,grown));'''
 if code.count(old)!=1:raise ValueError('Controller source changed')
 text=code.replace(old,new);source=T/'graph_control_nominal.cpp';library=T/'libgraph_control_nominal.so';source.write_text(text)
 subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(source),'-o',str(library),'-I/usr/local/cuda/include','-L/usr/local/cuda/lib64','-lcudart'],check=True)
 (T/'NOMINAL_SOURCE.json').write_text(json.dumps({'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'Retain nominal proposal only after accepted forced cut with e<0.1; all future attempts checked.','changes_numerical_sequence':True},indent=2)+'\n')
 return library

class NominalGraph(NativeGraph):
 def __init__(self,*args,**kw):
  library=T/'libgraph_control_nominal.so'
  if not library.exists():raise RuntimeError('Build and verify nominal controller before using it')
  kw['native_library']=library;super().__init__(*args,**kw)

if __name__=='__main__':print(build())
