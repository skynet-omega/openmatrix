
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#define __global__
#define __device__
struct Index { int x=0; } threadIdx, blockIdx;
struct Dimension { int x=1; } blockDim;
using std::min; using std::max; using std::isfinite;
enum Failure : int {
  ok = 0, no_progress = 1, invalid_trial = 2, invalid_domain = 3,
  accuracy_limit = 4, trial_limit = 5, device_launch_failure = 6
};
struct Control {
  double* clock;
  const double* status;
  double* state;
  const double* fine;
  const double* events;
  long n;
  long event_count;
  long next_ns;
  long min_ns;
  long max_ns;
  long min_accepted_ns;
  long accepted;
  long rejected;
  long attempts;
  long max_attempts;
  long event_index;
  double end_s;
  double used_s;
  double attempted_h_s;
  double attempted_stop_s;
  double max_error;
  int accept;
  int failure;
  int launch_status;
};
__global__ void prepare(Control* c) {
  if (blockIdx.x || threadIdx.x) return;
  if (c->attempts >= c->max_attempts) {
    c->failure = trial_limit;
    return;
  }
  ++c->attempts;
  while (c->event_index < c->event_count &&
         c->events[c->event_index] <= c->used_s) ++c->event_index;
  const double stop = c->event_index < c->event_count
      ? c->events[c->event_index] : c->end_s;
  const double available = stop - c->used_s;
  const long proposed_ns = min(c->next_ns, c->max_ns);
  const double h = fmin(available, static_cast<double>(proposed_ns) * 1e-9);
  if (!(h > 0.0) || c->used_s + h == c->used_s) {
    c->failure = no_progress;
    return;
  }
  c->attempted_h_s = h;
  c->attempted_stop_s = stop;
  c->clock[0] = c->used_s;
  c->clock[1] = h;
  // A clipped endpoint is the scheduled floating-point timestamp itself.
  // Reconstructing it as start + (stop - start) can change its event side.
  c->clock[2] = h == available ? stop : c->used_s + h;
}
__global__ void decide(Control* c) {
  if (blockIdx.x || threadIdx.x) return;
  if (c->failure) return;
  const double error = c->status[0];
  if (!isfinite(error) || !isfinite(c->status[1]) || c->status[1] != 0.0) {
    c->failure = invalid_trial;
    return;
  }
  const double h = c->attempted_h_s;
  if (error <= 1.0) {
    if (c->status[2] != 0.0) {
      c->failure = invalid_domain;
      return;
    }
    // An interior event can clip a valid proposal to an arbitrarily short
    // tail. Preserve its pre-cut size only under the existing small-error
    // rule; the next trial still clips at events and checks its own error.
    // Compute this before advancing used_s. Epoch ends never qualify.
    const long previous_ns = c->next_ns;
    const double requested_s = static_cast<double>(
        min(previous_ns, c->max_ns)) * 1e-9;
    const double available_s = c->attempted_stop_s - c->used_s;
    const bool recover_proposal = c->event_index < c->event_count &&
        c->attempted_stop_s < c->end_s && h == available_s &&
        h < requested_s && error < .1;
    c->accept = 1;
    c->used_s = c->clock[2];
    ++c->accepted;
    c->min_accepted_ns = min(c->min_accepted_ns,
                              static_cast<long>(ceil(h * 1e9)));
    c->max_error = fmax(c->max_error, error);
    const long proposal = static_cast<long>(floor(h * 1e9 *
                                                   (error < .1 ? 2.0 : 1.0)));
    c->next_ns = min(c->max_ns, max(c->min_ns, proposal));
    if (recover_proposal)
      c->next_ns = min(c->max_ns, max(c->next_ns, previous_ns));
  } else {
    c->accept = 0;
    ++c->rejected;
    const long smaller = static_cast<long>(floor(h * 1e9 * .5));
    if (smaller < c->min_ns) c->failure = accuracy_limit;
    else c->next_ns = smaller;
  }
}
extern "C" __global__ void endpoint_clocks(const double*c,double*left,double*right) {
 if(threadIdx.x==0 && blockIdx.x==0) {
  left[0]=nextafter(c[2],c[0]);left[1]=0.;
  right[0]=c[2];right[1]=0.;
 }
}
__device__ double conv(double t,double tq,double ts){
 double z=t*(1/ts-1/tq),den=1-ts/tq;
 if(fabs(z)<1e-5)return t/ts*exp(-t/ts)*(1+z/2+z*z/6+z*z*z/24);
 return (z>=0?exp(-t/tq)*(-expm1(-fmax(z,0.))):exp(-t/ts)*expm1(fmin(z,0.)))/(den==0?1.:den);
}
extern "C" __global__ void port(double*y,const long long*qr,const long long*sr,const double*q,const double*s,const double*tq,const double*ts,const double*et,const double*ej,const bool*setop,const double*post,const int*counts,int n,int width,const double*clock,double fraction){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 double t=clock[0]+fraction*clock[1];
 double a=q[i]*exp(-t/tq[i]),b=s[i]*exp(-t/ts[0])+q[i]*conv(t,tq[i],ts[0]);
 bool has_set=false;
 for(int k=0;k<counts[i];k++){int p=i*width+k;if(et[p]<=t){double u=t-et[p];a+=ej[p]*exp(-u/tq[i]);b+=ej[p]*conv(u,tq[i],ts[0]);}}
 for(int k=0;k<counts[i];k++){int p=i*width+k;if(et[p]<=t&&setop[p])has_set=true;}
 if(has_set){
   double qc=q[i],sc=s[i],prev=0.;
   for(int k=0;k<counts[i];k++){
     int p=i*width+k;double mark=et[p];if(mark>t)break;
     double dt=mark-prev;
     sc=sc*exp(-dt/ts[0])+qc*conv(dt,tq[i],ts[0]);
     qc=qc*exp(-dt/tq[i]);
     qc=setop[p]?post[p]:qc+ej[p];
     prev=mark;
   }
   double dt=t-prev;
   a=qc*exp(-dt/tq[i]);
   b=sc*exp(-dt/ts[0])+qc*conv(dt,tq[i],ts[0]);
 }
 y[qr[i]]=a;y[sr[i]]=b;
}
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double projected_q(double t,double stop){
 double y[2]={0.,0.},q[1]={0.},s[1]={0.},tq[1]={.02},ts[1]={.005};
 double et[1]={stop},ej[1]={1.},post[1]={1.},clock[2]={t,0.};
 long long qr[1]={0},sr[1]={1};bool sets[1]={true};int counts[1]={1};
 port(y,qr,sr,q,s,tq,ts,et,ej,sets,post,counts,1,1,clock,0.);
 return y[0];
}
Control setup(double* clock,double* status,const double* events,long count,double start,double end){
 Control c={};c.clock=clock;c.status=status;c.events=events;c.event_count=count;
 c.used_s=start;c.end_s=end;c.next_ns=100000;c.min_ns=100;c.max_ns=1000000;
 c.min_accepted_ns=1000000;c.max_attempts=10000;return c;
}
void clipped(double start,double stop,bool demand_legacy_failure){
 double clock[3]={},status[3]={},events[2]={start,stop},left[2]={},right[2]={};
 auto c=setup(clock,status,events,2,start,125000*1e-9);prepare(&c);
 need(c.failure==0 && clock[2]==stop,"prepare did not preserve the canonical boundary");
 endpoint_clocks(clock,left,right);
 need(left[0]<stop && right[0]==stop,"endpoint side differs");
 need(projected_q(left[0],stop)==0. && projected_q(right[0],stop)==1.,"ADD/SET port side differs");
 if(demand_legacy_failure)need(start+(stop-start)!=stop,"missing historical cancellation example");
 status[0]=fabs(clock[1]*(-1./8.)*projected_q(left[0],stop))/1e-7;
 decide(&c);
 need(c.failure==0 && c.accept==1 && c.used_s==stop && c.next_ns>=100000,"interior recovery/commit changed");
}
int main(){try{
 need(sizeof(long)==8 && std::numeric_limits<double>::is_iec559,"CPU numerical ABI mismatch");
 clipped(1.1461103097022343e-05,2.8348089914503676e-05,true);
 clipped(5.581565907498342e-06,5.825387811154045e-05,true);
 clipped(1.1461103097022343e-05/128.,2.8348089914503676e-05/128.,true);
 clipped(1e-5,std::nextafter(1e-5,INFINITY),false);
 double clock[3]={},status[3]={},left[2]={},right[2]={};
 double e[1]={125000*1e-9};
 auto c=setup(clock,status,e,1,100000*1e-9,e[0]);prepare(&c);endpoint_clocks(clock,left,right);decide(&c);
 need(c.used_s==e[0] && c.next_ns<100000,"terminal event unexpectedly recovered proposal");
 c=setup(clock,status,nullptr,0,0.,125000*1e-9);c.next_ns=10000;prepare(&c);
 need(clock[2]==clock[0]+clock[1],"free endpoint arithmetic changed");
 decide(&c);need(c.used_s==clock[2] && c.next_ns==20000,"ordinary controller changed");
 double two[2]={1.1461103097022343e-05,2.8348089914503676e-05};
 status[0]=2.;c=setup(clock,status,two,2,two[0],125000*1e-9);prepare(&c);
 long expected=long(floor(clock[1]*1e9*.5));decide(&c);
 need(!c.accept && c.used_s==two[0] && c.next_ns==expected && c.rejected==1,"rejection bypassed");
 status[0]=1e300;c=setup(clock,status,two,2,two[0],125000*1e-9);prepare(&c);decide(&c);
 need(!c.accept && c.used_s==two[0],"domain rejection sentinel bypassed");
 std::cout<<"{\"status\":\"PASS_EXTRACTED_CPU_FUNCTIONS\",\"checks\":8}"<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
