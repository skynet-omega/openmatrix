// Minimal project-local C ABI for official SUNDIALS CUDA vectors and ARKStep.
#include <arkode/arkode.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_ls.h>
#include <nvector/nvector_cuda.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_cuda_policies.hpp>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdio>
#include "native_support.cpp"
using Callback=int(*)(int,double,double*,double*,double*,double);
static thread_local std::string last_error;
struct Bridge {
 SUNContext ctx=nullptr;N_Vector y=nullptr,atol=nullptr,ratol=nullptr;SUNLinearSolver ls=nullptr,mls=nullptr;void* mem=nullptr;
 std::vector<double> host,host_atol,host_ratol;Callback cb;Native* native=nullptr;cudaStream_t stream;double time=0;
 sundials::cuda::ThreadDirectExecPolicy exec;
 sundials::cuda::BlockReduceExecPolicy reduce;
 Bridge(long n,Callback callback,cudaStream_t st):host(n),host_atol(n),host_ratol(n),cb(callback),stream(st),exec(256,st),reduce(256,0,st){}
 ~Bridge(){if(mem)ARKodeFree(&mem);if(ls)SUNLinSolFree(ls);if(mls)SUNLinSolFree(mls);if(ratol)N_VDestroy(ratol);if(atol)N_VDestroy(atol);if(y)N_VDestroy(y);if(ctx)SUNContext_Free(&ctx);}
};
static void ck(int status,const char* where){if(status<0)throw std::runtime_error(std::string(where)+": "+std::to_string(status));}
static double* ptr(N_Vector v){return v?N_VGetDeviceArrayPointer_Cuda(v):nullptr;}
static int dispatch(void* d,int k,double t,double* y,double* v,double* out,double gamma){auto* b=static_cast<Bridge*>(d);return b->native?b->native->call(k,t,y,v,out,gamma):b->cb(k,t,y,v,out,gamma);}
static int rhs(double t,N_Vector y,N_Vector f,void* d){return dispatch(d,0,t,ptr(y),nullptr,ptr(f),0);}
static int jvp(N_Vector v,N_Vector j,double t,N_Vector y,N_Vector,void* d,N_Vector){return dispatch(d,1,t,ptr(y),ptr(v),ptr(j),0);}
static int mass(N_Vector v,N_Vector m,double t,void* d){return dispatch(d,2,t,nullptr,ptr(v),ptr(m),0);}
static int psetup(double t,N_Vector y,N_Vector,sunbooleantype,sunbooleantype* jcur,double gamma,void* d){*jcur=SUNTRUE;return dispatch(d,5,t,ptr(y),nullptr,nullptr,gamma);}
static int prec(double t,N_Vector y,N_Vector,N_Vector r,N_Vector z,double gamma,double,int,void* d){return dispatch(d,3,t,ptr(y),ptr(r),ptr(z),gamma);}
static int mprec(double t,N_Vector r,N_Vector z,double,int,void* d){return dispatch(d,4,t,nullptr,ptr(r),ptr(z),0);}
extern "C" {
const char* om_error(){return last_error.c_str();}
void* om_create_internal(long n,double* state,double* atol,double* ratol,double rtol,double t0,void* stream,Callback cb,Native* native){
 Bridge* b=nullptr;
 try{
  b=new Bridge(n,cb,reinterpret_cast<cudaStream_t>(stream));b->time=t0;b->native=native;
  ck(SUNContext_Create(SUN_COMM_NULL,&b->ctx),"SUNContext_Create");
  b->y=N_VMake_Cuda(n,b->host.data(),state,b->ctx);b->atol=N_VMake_Cuda(n,b->host_atol.data(),atol,b->ctx);
  b->ratol=N_VMake_Cuda(n,b->host_ratol.data(),ratol,b->ctx);
  if(!b->y||!b->atol||!b->ratol)throw std::runtime_error("CUDA vector allocation");
  ck(N_VSetKernelExecPolicy_Cuda(b->y,&b->exec,&b->reduce),"CUDA vector stream");
  ck(N_VSetKernelExecPolicy_Cuda(b->atol,&b->exec,&b->reduce),"CUDA tolerance stream");
  ck(N_VSetKernelExecPolicy_Cuda(b->ratol,&b->exec,&b->reduce),"CUDA residual tolerance stream");
  b->mem=ARKStepCreate(nullptr,rhs,t0,b->y,b->ctx);if(!b->mem)throw std::runtime_error("ARKStepCreate");
  ck(ARKodeSetUserData(b->mem,b),"user data");
  ck(ARKStepSetTableNum(b->mem,ARKODE_ESDIRK325L2SA_5_2_3,ARKODE_ERK_NONE),"ESDIRK table");
  ck(ARKodeSVtolerances(b->mem,rtol,b->atol),"tolerances");
  b->ls=SUNLinSol_SPGMR(b->y,SUN_PREC_LEFT,30,b->ctx);b->mls=SUNLinSol_SPGMR(b->y,SUN_PREC_LEFT,30,b->ctx);
  if(!b->ls||!b->mls)throw std::runtime_error("SPGMR allocation");
  ck(ARKodeSetLinearSolver(b->mem,b->ls,nullptr),"Newton solver");
  ck(ARKodeSetJacTimes(b->mem,nullptr,jvp),"global JVP");
  ck(ARKodeSetPreconditioner(b->mem,psetup,prec),"Newton preconditioner");
  ck(ARKodeSetMassLinearSolver(b->mem,b->mls,nullptr,SUNFALSE),"mass solver");
  ck(ARKodeSetMassTimes(b->mem,nullptr,mass,b),"mass product");
  ck(ARKodeResVtolerance(b->mem,b->ratol),"residual tolerances");
  ck(ARKodeSetEpsLin(b->mem,.05),"linear tolerance factor");
  ck(ARKodeSetMassEpsLin(b->mem,.05),"mass tolerance factor");
  ck(ARKodeSetMassPreconditioner(b->mem,nullptr,mprec),"mass preconditioner");
  ck(ARKodeSetMaxNumSteps(b->mem,100000),"step budget");
  ck(ARKodeSetInitStep(b->mem,1e-6),"initial step");
  return b;
 }catch(const std::exception& e){last_error=e.what();delete b;return nullptr;}
}
void* om_create(long n,double* state,double* atol,double* ratol,double rtol,double t0,void* stream,Callback cb){return om_create_internal(n,state,atol,ratol,rtol,t0,stream,cb,nullptr);}
void* om_create_native(long n,double* state,double* atol,double* ratol,double rtol,double t0,void* stream,void* native){return om_create_internal(n,state,atol,ratol,rtol,t0,stream,nullptr,static_cast<Native*>(native));}
int om_advance(void* handle,double end,double* actual){
 auto* b=static_cast<Bridge*>(handle);
 try{ck(ARKodeSetStopTime(b->mem,end),"stop time");ck(ARKodeEvolve(b->mem,end,b->y,&b->time,ARK_NORMAL),"ARKodeEvolve");*actual=b->time;return 0;}
 catch(const std::exception& e){last_error=e.what();*actual=b->time;return -1;}
}
int om_stats(void* handle,long* out){
 auto* b=static_cast<Bridge*>(handle);
 ARKodeGetNumSteps(b->mem,out);ARKodeGetNumRhsEvals(b->mem,1,out+1);ARKodeGetNumNonlinSolvIters(b->mem,out+2);ARKodeGetNumLinIters(b->mem,out+3);ARKodeGetNumErrTestFails(b->mem,out+4);return 0;
}
void om_destroy(void* handle){delete static_cast<Bridge*>(handle);}
}
