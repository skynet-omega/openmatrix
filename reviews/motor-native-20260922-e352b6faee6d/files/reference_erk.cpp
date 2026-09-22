// Existing SUNDIALS ERK8 as an independent high-accuracy reference, identity mass only.
#include "bridge.cpp"
#include <arkode/arkode_erkstep.h>
extern "C" {
void* om_erk_create(long n,double* state,double* atol,double rtol,double t0,void* stream,void* native){
 Bridge* b=nullptr;try{
  b=new Bridge(n,nullptr,reinterpret_cast<cudaStream_t>(stream));b->native=static_cast<Native*>(native);b->time=t0;
  ck(SUNContext_Create(SUN_COMM_NULL,&b->ctx),"reference context");
  b->y=N_VMake_Cuda(n,b->host.data(),state,b->ctx);b->atol=N_VMake_Cuda(n,b->host_atol.data(),atol,b->ctx);
  if(!b->y||!b->atol)throw std::runtime_error("reference vectors");
  ck(N_VSetKernelExecPolicy_Cuda(b->y,&b->exec,&b->reduce),"reference stream");ck(N_VSetKernelExecPolicy_Cuda(b->atol,&b->exec,&b->reduce),"reference tolerances stream");
  b->mem=ERKStepCreate(rhs,t0,b->y,b->ctx);if(!b->mem)throw std::runtime_error("ERKStepCreate");
  ck(ARKodeSetUserData(b->mem,b),"reference userdata");ck(ERKStepSetTableNum(b->mem,ARKODE_FEHLBERG_13_7_8),"reference tableau");ck(ARKodeSVtolerances(b->mem,rtol,b->atol),"reference tolerances");ck(ARKodeSetInitStep(b->mem,1e-6),"reference initial step");ck(ARKodeSetMaxNumSteps(b->mem,100000),"reference step budget");return b;
 }catch(const std::exception& e){last_error=e.what();delete b;return nullptr;}
}
void om_erk_stats(void* handle,long* out){auto* b=static_cast<Bridge*>(handle);ARKodeGetNumSteps(b->mem,out);ARKodeGetNumRhsEvals(b->mem,0,out+1);ARKodeGetNumErrTestFails(b->mem,out+2);}
}
