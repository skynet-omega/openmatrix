// Generic native controller for an externally captured FP64 target/rate trial.
// The owner keeps graph and device buffers alive; every operation uses one stream.
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <atomic>
static thread_local std::string error;
static void ck(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
struct Runner{
 cudaGraphExec_t graph;cudaStream_t stream;double *clock,*status,*x,*fine,*hc=nullptr,*hs=nullptr;long n;std::atomic_flag busy=ATOMIC_FLAG_INIT;
 Runner(void*g,void*s,double*c,double*r,double*y,double*f,long N):graph((cudaGraphExec_t)g),stream((cudaStream_t)s),clock(c),status(r),x(y),fine(f),n(N){ck(cudaMallocHost((void**)&hc,2*sizeof(double)));ck(cudaMallocHost((void**)&hs,3*sizeof(double)));}
 ~Runner(){cudaStreamSynchronize(stream);if(hc)cudaFreeHost(hc);if(hs)cudaFreeHost(hs);}
};
extern "C" {
const char* engine_error(){return error.c_str();}
void* engine_create(void*g,void*s,double*c,double*r,double*y,double*f,long n){try{if(!g||!c||!r||!y||!f||n<=0)throw std::runtime_error("invalid graph contract");return new Runner(g,s,c,r,y,f,n);}catch(const std::exception&e){error=e.what();return nullptr;}}
void engine_destroy(void*p){delete (Runner*)p;}
int engine_advance(void*p,long duration,long*next,long minstep,long maxstep,double budget,long*counts,double*maxerr){
 auto*r=(Runner*)p;if(r->busy.test_and_set()){error="nonreentrant graph session";return -1;}
 struct Unlock{Runner*r;~Unlock(){r->busy.clear();}} unlock{r};
 try{error.clear();if(duration<=0||minstep<=0||maxstep<minstep||*next<=0||!std::isfinite(budget)||budget<=0)throw std::runtime_error("invalid epoch controls");
  long left=duration,used=0,attempts=0;counts[0]=counts[1]=0;counts[2]=maxstep;*maxerr=0.;auto start=std::chrono::steady_clock::now();
  while(left){if(++attempts>10000)throw std::runtime_error("trial budget");if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>budget)throw std::runtime_error("native epoch time budget");long h=std::min(left,std::min(*next,maxstep));r->hc[0]=used*1e-9;r->hc[1]=h*1e-9;
   ck(cudaMemcpyAsync(r->clock,r->hc,2*sizeof(double),cudaMemcpyHostToDevice,r->stream));ck(cudaGraphLaunch(r->graph,r->stream));ck(cudaMemcpyAsync(r->hs,r->status,3*sizeof(double),cudaMemcpyDeviceToHost,r->stream));ck(cudaStreamSynchronize(r->stream));
   double e=r->hs[0];if(!std::isfinite(e)||!std::isfinite(r->hs[1])||r->hs[1]!=0)throw std::runtime_error("nonfinite native trial");
   if(e<=1.){if(r->hs[2]!=0)throw std::runtime_error("accepted state outside declared domain");ck(cudaMemcpyAsync(r->x,r->fine,r->n*sizeof(double),cudaMemcpyDeviceToDevice,r->stream));used+=h;left-=h;counts[0]++;counts[2]=std::min(counts[2],h);*maxerr=std::max(*maxerr,e);*next=std::min(maxstep,e<.1?h*2:h);}
   else{counts[1]++;if(h/2<minstep)throw std::runtime_error("accuracy limit");*next=h/2;}
  }ck(cudaStreamSynchronize(r->stream));return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
}
// Known forcing discontinuities are mandatory internal boundaries. Fractional
// nanoseconds stay in double seconds; the outer physical clock remains integer.
extern "C" int engine_advance_events(void*p,long duration,long*next,long minstep,long maxstep,double budget,const double*events,long ne,long*counts,double*maxerr){
 auto*r=(Runner*)p;if(r->busy.test_and_set()){error="nonreentrant graph session";return -1;}
 struct Unlock{Runner*r;~Unlock(){r->busy.clear();}} unlock{r};
 try{
  error.clear();double end=duration*1e-9,used=0.;
  if(duration<=0||minstep<=0||maxstep<minstep||*next<=0||ne<0||!std::isfinite(budget)||budget<=0)throw std::runtime_error("invalid event epoch controls");
  for(long k=0;k<ne;k++)if(!std::isfinite(events[k])||events[k]<0||events[k]>end||(k&&events[k]<events[k-1]))throw std::runtime_error("invalid event boundary");
  long ix=0,attempts=0;counts[0]=counts[1]=0;counts[2]=maxstep;*maxerr=0.;auto start=std::chrono::steady_clock::now();
  while(used<end){
   if(++attempts>10000)throw std::runtime_error("event trial budget");
   if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>budget)throw std::runtime_error("event epoch time budget");
   while(ix<ne&&events[ix]<=used)ix++;
   double stop=ix<ne?events[ix]:end,available=stop-used;
   long nominal=std::min(*next,maxstep);
   double requested=nominal*1e-9,h=std::min(available,requested);
   const bool forced_cut=h<requested;
   if(!(h>0)||used+h==used)throw std::runtime_error("event time resolution exhausted");
   r->hc[0]=used;r->hc[1]=h;
   ck(cudaMemcpyAsync(r->clock,r->hc,2*sizeof(double),cudaMemcpyHostToDevice,r->stream));ck(cudaGraphLaunch(r->graph,r->stream));ck(cudaMemcpyAsync(r->hs,r->status,3*sizeof(double),cudaMemcpyDeviceToHost,r->stream));ck(cudaStreamSynchronize(r->stream));
   double e=r->hs[0];if(!std::isfinite(e)||!std::isfinite(r->hs[1])||r->hs[1]!=0)throw std::runtime_error("nonfinite event trial");
   if(e<=1.){
    if(r->hs[2]!=0)throw std::runtime_error("accepted event state outside domain");
    ck(cudaMemcpyAsync(r->x,r->fine,r->n*sizeof(double),cudaMemcpyDeviceToDevice,r->stream));
    used=h==available?stop:used+h;counts[0]++;counts[2]=std::min(counts[2],(long)std::ceil(h*1e9));*maxerr=std::max(*maxerr,e);
    // An event tail shorter than minstep is legal if accepted. It must not
    // permanently shrink the next proposal, nor bypass a rejected error test.
    if(forced_cut && e<.1) {
     // Recover a proposal, never accept an untested step or enlarge from a tail.
     *next=std::max(minstep,nominal);
    } else {
     *next=std::min(maxstep,std::max(minstep,(long)std::floor(h*1e9*(e<.1?2:1))));
    }
   }else{counts[1]++;long smaller=(long)std::floor(h*1e9*.5);if(smaller<minstep)throw std::runtime_error("event accuracy limit");*next=smaller;}
  }
  ck(cudaStreamSynchronize(r->stream));return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
