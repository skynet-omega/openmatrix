// Included by bridge.cpp. Large vectors never cross the host boundary in a callback.
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <chrono>
#include <atomic>
#include <cmath>
static void cuda_ok(cudaError_t x,const char* where){if(x!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(x));}
static void driver_ok(CUresult x,const char* where){if(x!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(x,&s);throw std::runtime_error(std::string(where)+": "+(s?s:"driver error"));}}
struct Dynamic {const double* x;const double* v;double* dest;double t;double gamma;};
static_assert(sizeof(Dynamic)==40,"CUDA dynamic ABI");
struct Native {
 cudaStream_t stream;CUmodule module=nullptr;std::unordered_map<std::string,CUfunction> functions;
 cudaGraphExec_t graphs[5]={};Dynamic* device=nullptr;Dynamic* host=nullptr;int* flag=nullptr;int* hostflag=nullptr;
 std::vector<int> counts;std::vector<void*> p;int n,ni;std::string error;long calls[5]={},fallback_calls=0;
 std::atomic_flag busy=ATOMIC_FLAG_INIT;double seconds_limit=1e100;std::chrono::steady_clock::time_point began;
 Native(cudaStream_t s,int states,int inputs):stream(s),n(states),ni(inputs){began=std::chrono::steady_clock::now();}
 ~Native(){cudaStreamSynchronize(stream);for(auto g:graphs)if(g)cudaGraphExecDestroy(g);if(module)cuModuleUnload(module);if(device)cudaFree(device);if(flag)cudaFree(flag);if(host)cudaFreeHost(host);if(hostflag)cudaFreeHost(hostflag);}
 template<typename... Args>void launch(const std::string& name,int blocks,int threads,Args... values){
   CUfunction f;auto it=functions.find(name);if(it==functions.end()){driver_ok(cuModuleGetFunction(&f,module,name.c_str()),"kernel lookup");functions[name]=f;}else f=it->second;
   void* argv[]={static_cast<void*>(&values)...};driver_ok(cuLaunchKernel(f,blocks,1,1,threads,1,1,0,reinterpret_cast<CUstream>(stream),argv,nullptr),name.c_str());
 }
 void population(const char* phase){for(size_t j=0;j<counts.size();++j)launch(std::string(phase)+"_"+std::to_string(j),(counts[j]+255)/256,256,device,p[0],p[9],p[10],p[7],p[8],p[1],p[11],flag);}
 void project(bool tangent){if(ni)launch("project",(ni+7)/8,256,p[4],p[5],p[6],p[tangent?8:7],p[tangent?3:2],p[tangent?10:9],ni,flag);}
 void operation(int kind){
   if(kind==2){launch("mass",(n+7)/8,256,device,p[12],p[13],p[14],n,flag);return;}
   if(kind==4){launch("massprec",(n+255)/256,256,device,p[11],n,flag);return;}
   population(kind==1?"output_jvp":"output");project(false);if(kind==1)project(true);
   population(kind==0?"rhs":kind==1?"jvp":"prec");
 }
 void initialize(const char* source,const unsigned long long* pointers,const int* population_counts,int npops){
   for(int i=0;i<16;i++)p.push_back(reinterpret_cast<void*>(pointers[i]));counts.assign(population_counts,population_counts+npops);
   nvrtcProgram program=nullptr;if(nvrtcCreateProgram(&program,source,"model.cu",0,nullptr,nullptr)!=NVRTC_SUCCESS)throw std::runtime_error("NVRTC create");
   const char* options[]={"--gpu-architecture=compute_89","--std=c++14","--include-path=/usr/include"};
   auto status=nvrtcCompileProgram(program,3,options);size_t len=0;
   if(status!=NVRTC_SUCCESS){nvrtcGetProgramLogSize(program,&len);std::string log(len,'\0');nvrtcGetProgramLog(program,log.data());nvrtcDestroyProgram(&program);throw std::runtime_error("NVRTC compile: "+log);}
   nvrtcGetPTXSize(program,&len);std::vector<char> ptx(len);nvrtcGetPTX(program,ptx.data());nvrtcDestroyProgram(&program);driver_ok(cuModuleLoadData(&module,ptx.data()),"load generated program");
   cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device),sizeof(Dynamic)),"device params");cuda_ok(cudaMalloc(reinterpret_cast<void**>(&flag),sizeof(int)),"error flag");
   cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&host),sizeof(Dynamic),cudaHostAllocDefault),"pinned params");cuda_ok(cudaHostAlloc(reinterpret_cast<void**>(&hostflag),sizeof(int),cudaHostAllocDefault),"pinned flag");
   for(int kind=0;kind<5;kind++){
     cudaGraph_t graph=nullptr;cuda_ok(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"begin graph");
     try{operation(kind);cuda_ok(cudaStreamEndCapture(stream,&graph),"end graph");cuda_ok(cudaGraphInstantiate(&graphs[kind],graph,nullptr,nullptr,0),"instantiate graph");cuda_ok(cudaGraphDestroy(graph),"destroy graph template");}
     catch(...){cudaStreamEndCapture(stream,&graph);if(graph)cudaGraphDestroy(graph);throw;}
   }
 }
 int call(int kind,double t,double* x,double* v,double* dest,double gamma){
   if(busy.test_and_set()){error="Concurrent native callback rejected";return -1;}
   struct Unlock{std::atomic_flag& b;~Unlock(){b.clear();}}unlock{busy};
   try{
     error.clear();if(kind<0||kind>4||!std::isfinite(t)||!std::isfinite(gamma)||!dest||((kind<=1||kind==3)&&!x)||(kind!=0&&!v))throw std::runtime_error("invalid callback parameters");
     if(std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count()>seconds_limit)throw std::runtime_error("native pilot wall budget exhausted");
     calls[kind]++;*host=Dynamic{x,v,dest,t,gamma};*hostflag=0;
     cuda_ok(cudaMemcpyAsync(device,host,sizeof(Dynamic),cudaMemcpyHostToDevice,stream),"update dynamic parameters");cuda_ok(cudaMemsetAsync(flag,0,sizeof(int),stream),"clear flag");
     cuda_ok(cudaGraphLaunch(graphs[kind],stream),"launch graph");cuda_ok(cudaMemcpyAsync(hostflag,flag,sizeof(int),cudaMemcpyDeviceToHost,stream),"read aggregate flag");cuda_ok(cudaStreamSynchronize(stream),"callback completion");
     if(*hostflag&2)fallback_calls++;if(*hostflag&1)throw std::runtime_error("nonfinite native equation or output");return 0;
   }catch(const std::exception& e){error=e.what();return -1;}
 }
};
static thread_local std::string native_create_error;
extern "C" {
void* om_native_create(const char* source,const long* sizes,const unsigned long long* ptrs,const int* counts,void* stream){Native* p=nullptr;try{p=new Native(reinterpret_cast<cudaStream_t>(stream),sizes[0],sizes[1]);p->initialize(source,ptrs,counts,sizes[2]);return p;}catch(const std::exception& e){native_create_error=e.what();delete p;return nullptr;}}
const char* om_native_error(void* p){return p?static_cast<Native*>(p)->error.c_str():native_create_error.c_str();}
int om_native_call(void* p,int kind,double t,double* x,double* v,double* out,double gamma){return static_cast<Native*>(p)->call(kind,t,x,v,out,gamma);}
void om_native_limit(void* p,double seconds){auto* n=static_cast<Native*>(p);n->began=std::chrono::steady_clock::now();n->seconds_limit=seconds;}
void om_native_stats(void* p,long* out){auto* n=static_cast<Native*>(p);for(int i=0;i<5;i++)out[i]=n->calls[i];out[5]=n->fallback_calls;}
void om_native_destroy(void* p){delete static_cast<Native*>(p);}
}
