// Native graph transaction controller. Model graphs own trial and commit rules.
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>
static thread_local std::string error;
static void ck(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
struct Session {
 cudaGraphExec_t trial,commit;cudaStream_t stream;long long*clock;double*result;int*flag;
 long long*hc=nullptr;double*he=nullptr;int*hf=nullptr;
 Session(void*t,void*c,void*s,void*k,void*r,void*f):trial((cudaGraphExec_t)t),commit((cudaGraphExec_t)c),stream((cudaStream_t)s),clock((long long*)k),result((double*)r),flag((int*)f){
  try{ck(cudaMallocHost((void**)&hc,2*sizeof(long long)));ck(cudaMallocHost((void**)&he,sizeof(double)));ck(cudaMallocHost((void**)&hf,sizeof(int)));}
  catch(...){if(hc)cudaFreeHost(hc);if(he)cudaFreeHost(he);if(hf)cudaFreeHost(hf);throw;}
 }
 ~Session(){cudaStreamSynchronize(stream);cudaFreeHost(hc);cudaFreeHost(he);cudaFreeHost(hf);}
};
extern "C" {
const char* cell_error(){return error.c_str();}
void* cell_create(void*t,void*c,void*s,void*k,void*r,void*f){try{return new Session(t,c,s,k,r,f);}catch(const std::exception&e){error=e.what();return nullptr;}}
void cell_destroy(void*p){delete (Session*)p;}
int cell_advance(void*p,long long duration,long long maximum,long long*stats,double*maximum_error){
 auto*x=(Session*)p;
 try{
  if(duration<=0||maximum<=0||maximum>25000)throw std::runtime_error("invalid membrane epoch");
  long long left=duration,used=0,h=std::min(maximum,left),attempts=0;stats[0]=stats[1]=0;stats[2]=25000;*maximum_error=0.;
  auto start=std::chrono::steady_clock::now();
  while(left){
   h=std::min(h,left);if(left-h>0&&left-h<200)h=left;
   if(h/2<1||++attempts>10000)throw std::runtime_error("membrane work bound");
   if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>60)throw std::runtime_error("membrane epoch time bound");
   x->hc[0]=used;x->hc[1]=h;ck(cudaMemcpyAsync(x->clock,x->hc,2*sizeof(long long),cudaMemcpyHostToDevice,x->stream));
   ck(cudaGraphLaunch(x->trial,x->stream));ck(cudaMemcpyAsync(x->he,x->result,sizeof(double),cudaMemcpyDeviceToHost,x->stream));ck(cudaStreamSynchronize(x->stream));
   const double e=*x->he;if(!std::isfinite(e))throw std::runtime_error("nonfinite membrane estimator");
   if(e<=1.){
    ck(cudaGraphLaunch(x->commit,x->stream));ck(cudaMemcpyAsync(x->hf,x->flag,sizeof(int),cudaMemcpyDeviceToHost,x->stream));ck(cudaStreamSynchronize(x->stream));
    if(*x->hf)throw std::runtime_error("physical event capacity/state failed");
    used+=h;left-=h;stats[0]++;stats[2]=std::min(stats[2],h/2);*maximum_error=std::max(*maximum_error,e);
    h=std::min(maximum,e<.1?h*2:h);
   }else{stats[1]++;if(h/2<200)throw std::runtime_error("membrane accuracy needs less than200ns");h/=2;}
  }
  return 0;
 }catch(const std::exception&e){error=e.what();return -1;}
}
}
