// Generic device-owned adaptive transaction for independent state blocks.
// Model::trial does not mutate committed state; commit publishes inside private
// epoch storage. The host publishes the epoch only if ALL blocks succeed.
// All participating lanes must receive identical trial error and commit status.
template<class Model>
__device__ int advance_independent_block(Model& model, long long duration,
 long long maximum, long long minimum, long long attempts_limit,
 long long* statistics, double* max_error) {
 long long used=0,h=maximum,accepted=0,rejected=0,min_half=maximum,attempts=0;
 double peak=0.;
 while(used<duration){
  h=h<duration-used?h:duration-used;
  if(duration-used-h>0&&duration-used-h<minimum)h=duration-used;
  if(h/2<1||++attempts>attempts_limit)return 1;
  double e=model.trial(h);
  if(!isfinite(e))return 2;
  if(e<=1.){
   if(model.commit(used,h))return 3;
   used+=h;accepted++;min_half=min_half<h/2?min_half:h/2;
   peak=peak>e?peak:e;
   h=e<.1?h*2:h;h=h<maximum?h:maximum;
  }else{
   rejected++;if(h/2<minimum)return 4;h/=2;
  }
 }
 if(threadIdx.x==0){statistics[0]=accepted;statistics[1]=rejected;
  statistics[2]=min_half;*max_error=peak;}
 return 0;
}
