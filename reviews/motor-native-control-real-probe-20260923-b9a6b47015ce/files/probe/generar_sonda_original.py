"""Genera controlador instrumentado desde la fuente verificada; no lo instala.
Requiere el graph_control_v2.cpp publicado. Ningún ID ni ecuación neuronal.
"""
import argparse, hashlib, json
from pathlib import Path
SHA='5aa0935103ad2e29d7bf05683ad3e291949f04c7abcd6537f945cc90daa62642'
TRIAL='ck(cudaMemcpyAsync(r->clock,r->hc,2*sizeof(double),cudaMemcpyHostToDevice,r->stream));ck(cudaGraphLaunch(r->graph,r->stream));ck(cudaMemcpyAsync(r->hs,r->status,3*sizeof(double),cudaMemcpyDeviceToHost,r->stream));ck(cudaStreamSynchronize(r->stream));'
COMMIT='ck(cudaMemcpyAsync(r->x,r->fine,r->n*sizeof(double),cudaMemcpyDeviceToDevice,r->stream));'
CODE=r'''
#include <vector>
#include <array>
#include <cstdio>
#include <memory>
static_assert(sizeof(long)==8,"ABI Linux/WSL64 required");
namespace probe {
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Sample {
 cudaEvent_t ev[6]={}; bool recorded[6]={},done=false,commit=false;
 int epoch=-1; double clock[2]={},status[3]={},api[5]={};
};
struct Epoch {long duration=0,ne=0,next_in=0,next_out=0,c[3]={};double wall=0,err=0;int code=0;};
struct Probe {
 Runner* owner; int mode; size_t used=0,epochs=0; bool overflow=false,invalid=false,in_call=false;
 int runtime=0,driver=0; std::vector<Sample> rows; std::array<Epoch,64> ep{};Sample* current=nullptr;
 Probe(Runner*r,int n,int m):owner(r),mode(m),rows(n){}
 ~Probe(){for(auto&s:rows)for(auto e:s.ev)if(e)cudaEventDestroy(e);}
};
static thread_local Probe* P=nullptr;
static void mark(Sample*s,int k,Runner*r){
 if(s&&P->mode){ck(cudaEventRecord(s->ev[k],r->stream));s->recorded[k]=true;}
}
static Sample* start(Runner*r){
 if(!P||P->owner!=r||!P->in_call)return nullptr;
 if(P->used==P->rows.size()){P->overflow=true;P->current=nullptr;return nullptr;}
 Sample*s=&P->rows[P->used++];s->epoch=int(P->epochs-1);P->current=s;
 s->clock[0]=r->hc[0];s->clock[1]=r->hc[1];return s;
}
static void trial(Runner*r){
 Sample*s=start(r);
 if(!s){
  ck(cudaMemcpyAsync(r->clock,r->hc,2*sizeof(double),cudaMemcpyHostToDevice,r->stream));
  ck(cudaGraphLaunch(r->graph,r->stream));
  ck(cudaMemcpyAsync(r->hs,r->status,3*sizeof(double),cudaMemcpyDeviceToHost,r->stream));
  ck(cudaStreamSynchronize(r->stream));return;
 }
 mark(s,0,r);auto t=Clock::now();
 ck(cudaMemcpyAsync(r->clock,r->hc,2*sizeof(double),cudaMemcpyHostToDevice,r->stream));
 if(s)s->api[0]=ms(t);mark(s,1,r);t=Clock::now();
 ck(cudaGraphLaunch(r->graph,r->stream));
 if(s)s->api[1]=ms(t);mark(s,2,r);t=Clock::now();
 ck(cudaMemcpyAsync(r->hs,r->status,3*sizeof(double),cudaMemcpyDeviceToHost,r->stream));
 if(s)s->api[2]=ms(t);mark(s,3,r);t=Clock::now();
 ck(cudaStreamSynchronize(r->stream));
 if(s){s->api[3]=ms(t);std::copy(r->hs,r->hs+3,s->status);s->done=true;}
}
static void commit(Runner*r){
 Sample*s=P&&P->owner==r&&P->in_call?P->current:nullptr;mark(s,4,r);auto t=Clock::now();
 ck(cudaMemcpyAsync(r->x,r->fine,r->n*sizeof(double),cudaMemcpyDeviceToDevice,r->stream));
 if(s){s->api[4]=ms(t);s->commit=true;}mark(s,5,r);
}
struct Scope {
 Probe*p=nullptr;Epoch*e=nullptr;Clock::time_point begin;
 Scope(void*r,long d,long ne,long next){
  if(P&&P->owner==r&&!P->in_call){
   p=P;p->current=nullptr;
   if(p->epochs==p->ep.size()){p->overflow=true;p=nullptr;return;}
   e=&p->ep[p->epochs++];e->duration=d;e->ne=ne;e->next_in=next;p->in_call=true;begin=Clock::now();
  }
 }
 void finish(int code,long next,const long*c,double err){
  if(e){e->wall=ms(begin);e->code=code;e->next_out=next;
   if(code==0){std::copy(c,c+3,e->c);e->err=err;}}
 }
 ~Scope(){if(p){p->in_call=false;p->current=nullptr;}}
};
static double elapsed(Sample&a,int i,Sample&b,int j){
 if(!P->mode||!a.recorded[i]||!b.recorded[j])return -1.;float value=0;
 if(cudaEventQuery(a.ev[i])!=cudaSuccess||cudaEventQuery(b.ev[j])!=cudaSuccess||
    cudaEventElapsedTime(&value,a.ev[i],b.ev[j])!=cudaSuccess){P->invalid=true;return -1.;}
 return value;
}
}
extern "C" int engine_probe_arm(void*p,int capacity,int mode){
 try{
  if(!p||probe::P||capacity<1||capacity>10000||(mode!=0&&mode!=1))throw std::runtime_error("probe contract");
  auto*r=(Runner*)p;if(r->busy.test_and_set())throw std::runtime_error("runner busy");
  struct Unlock{Runner*r;~Unlock(){r->busy.clear();}} unlock{r};
  cudaStreamCaptureStatus status;ck(cudaStreamIsCapturing(r->stream,&status));
  if(status!=cudaStreamCaptureStatusNone)throw std::runtime_error("arm outside capture only");
  std::unique_ptr<probe::Probe> q(new probe::Probe(r,capacity,mode));
  ck(cudaRuntimeGetVersion(&q->runtime));ck(cudaDriverGetVersion(&q->driver));
  if(mode)for(auto&s:q->rows)for(auto&e:s.ev)ck(cudaEventCreate(&e));
  probe::P=q.release();return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
extern "C" int engine_probe_clear(void*p){
 if(!probe::P||probe::P->owner!=p||probe::P->in_call)return -1;
 delete probe::P;probe::P=nullptr;return 0;
}
extern "C" int engine_probe_save(void*p,const char*path){
 auto*q=probe::P;if(!q||q->owner!=p||q->in_call||!path)return -1;
 FILE*f=std::fopen(path,"wx");if(!f)return -1;
 std::fprintf(f,"{\"mode\":%d,\"cuda_runtime\":%d,\"cuda_driver\":%d,\"rows\":[",q->mode,q->runtime,q->driver);
 for(size_t n=0;n<q->used;n++){
  auto&s=q->rows[n];double v[6];
  v[0]=probe::elapsed(s,0,s,1);v[1]=probe::elapsed(s,1,s,2);v[2]=probe::elapsed(s,2,s,3);
  v[3]=s.commit?probe::elapsed(s,3,s,4):-1.;v[4]=s.commit?probe::elapsed(s,4,s,5):-1.;v[5]=-1.;
  if(n&&q->rows[n-1].epoch==s.epoch){auto&prev=q->rows[n-1];v[5]=probe::elapsed(prev,prev.commit?5:3,s,0);}
  std::fprintf(f,"%s{\"epoch\":%d,\"t_hex\":\"%a\",\"h_hex\":\"%a\",\"status_hex\":[\"%a\",\"%a\",\"%a\"],\"done\":%s,\"committed\":%s,\"stream_ms\":[",
   n?",":"",s.epoch,s.clock[0],s.clock[1],s.status[0],s.status[1],s.status[2],s.done?"true":"false",s.commit?"true":"false");
  for(int i=0;i<6;i++)std::fprintf(f,"%s%.9g",i?",":"",v[i]);std::fprintf(f,"],\"host_api_ms\":[");
  for(int i=0;i<5;i++)std::fprintf(f,"%s%.17g",i?",":"",s.api[i]);std::fprintf(f,"]}");
 }
 std::fprintf(f,"],\"epochs\":[");
 for(size_t k=0;k<q->epochs;k++){
  auto&e=q->ep[k];std::fprintf(f,"%s{\"duration_ns\":%ld,\"event_boundaries\":%ld,\"next_in\":%ld,\"next_out\":%ld,\"return_code\":%d,\"wall_ms\":%.17g,\"counts_if_success\":[%ld,%ld,%ld],\"maxerr_hex_if_success\":\"%a\"}",
   k?",":"",e.duration,e.ne,e.next_in,e.next_out,e.code,e.wall,e.c[0],e.c[1],e.c[2],e.err);
 }
 std::fprintf(f,"],\"overflow\":%s,\"timing_query_failed\":%s,\"stream_columns\":[\"H2D\",\"graph_interval\",\"D2H\",\"decision_gap\",\"commit_D2D\",\"between_trials\"],\"host_columns\":[\"H2D_API\",\"launch_API\",\"D2H_API\",\"synchronize_API\",\"commit_API\"],\"absent_marker\":-1,\"scope\":\"Stream intervals, not exclusive kernel time. Host and device intervals overlap. No numerical admission.\"}\n",q->overflow?"true":"false",q->invalid?"true":"false");
 bool bad=std::ferror(f);bad=std::fclose(f)!=0||bad;return bad?-1:0;
}
'''
WRAPPERS=r'''
extern "C" int engine_advance(void*p,long d,long*n,long lo,long hi,double budget,long*c,double*e){
 probe::Scope s(p,d,-1,*n);int code=base_engine_advance(p,d,n,lo,hi,budget,c,e);s.finish(code,*n,c,code==0?*e:0.);return code;
}
extern "C" int engine_advance_events(void*p,long d,long*n,long lo,long hi,double budget,const double*t,long nt,long*c,double*e){
 probe::Scope s(p,d,nt,*n);int code=base_engine_advance_events(p,d,n,lo,hi,budget,t,nt,c,e);s.finish(code,*n,c,code==0?*e:0.);return code;
}
'''
def generate(source):
    b=source.read_bytes()
    if hashlib.sha256(b).hexdigest()!=SHA: raise ValueError('Fuente no coincide con cd642f2')
    text=b.decode()
    if text.count(TRIAL)!=2 or text.count(COMMIT)!=2: raise ValueError('Anclas distintas')
    for name in ('engine_advance_events','engine_advance'):
        if text.count('int '+name+'(')!=1: raise ValueError('Firma distinta')
        text=text.replace('int '+name+'(','int base_'+name+'(')
    text=text.replace(TRIAL,'probe::trial(r);').replace(COMMIT,'probe::commit(r);')
    return text.replace('extern "C" {',CODE+'\nextern "C" {',1)+WRAPPERS
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    result=generate(a.source);a.out.mkdir(exist_ok=False)
    p=a.out/'graph_control_probe.cpp';p.write_text(result,encoding='utf-8')
    (a.out/'graph_control_parent.cpp').write_bytes(a.source.read_bytes())
    report={'parent_sha256':SHA,'probe_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'changes':'Solo llamadas de ejecución envueltas; aritmética/control original conservados',
            'requires':'CUDA Runtime + compilador C++17; misma thread OS y mismo owner para arm/advance/save/clear'}
    (a.out/'MANIFEST.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
