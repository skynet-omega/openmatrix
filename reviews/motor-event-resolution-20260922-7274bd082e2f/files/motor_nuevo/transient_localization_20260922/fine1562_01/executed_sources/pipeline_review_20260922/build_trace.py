from pathlib import Path
import subprocess,hashlib,json
H=Path(__file__).resolve().parent;parent=H.parent/'native_hybrid_20260922/graph_control_v2.cpp'
s=parent.read_text().replace('#include <atomic>','#include <atomic>\n#include <vector>')
s=s.replace('struct Runner{','struct Runner{\n std::vector<double> trace;')
s=s.replace('error.clear();double end=duration*1e-9,used=0.;','error.clear();r->trace.clear();double end=duration*1e-9,used=0.;')
needle='   if(e<=1.){\n    if(r->hs[2]!=0)'
insert='''   int reason=0;
   if(h==available&&ix<ne)reason|=1;
   if(h==available&&stop==end)reason|=2;
   if(h==requested&&*next<=maxstep)reason|=4;
   if(h==requested&&maxstep<=*next)reason|=8;
   r->trace.insert(r->trace.end(),{used,h,requested,double(*next),double(maxstep),available,stop,double(reason),e,e<=1.?1.:0.});
   if(e<=1.){
    if(r->hs[2]!=0)'''
if s.count(needle)!=1:raise RuntimeError('Trace insertion site changed')
s=s.replace(needle,insert)
s+='''\nextern "C" long engine_trace_count(void*p){return ((Runner*)p)->trace.size()/10;}
extern "C" int engine_trace_copy(void*p,double*out,long capacity){auto&r=((Runner*)p)->trace;if(capacity<(long)r.size())return -1;std::copy(r.begin(),r.end(),out);return 0;}
'''
(H/'graph_control_trace.cpp').write_text(s)
subprocess.run(['g++','-O3','-std=c++17','-fPIC','-shared',str(H/'graph_control_trace.cpp'),'-lcudart','-o',str(H/'libgraph_control_trace.so')],check=True)
(H/'TRACE_SOURCE.json').write_text(json.dumps({'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'instrumented_sha256':hashlib.sha256(s.encode()).hexdigest(),'change':'record decision causes after same trial, no numerical/controller rule changes','trace_columns':['start_s','h_s','requested_s','next_ns','max_ns','available_s','stop_s','reason_bits','error','accepted']},indent=2)+'\n')
