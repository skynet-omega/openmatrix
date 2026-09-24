// Physical event/filter update compiled separately from numerical trials.
// The adapter declares observation vectors, axonal coordinates and thresholds.
extern "C" __global__ void commit_event(int n,int dim,int width,int split,const long long*clock,
 const double*v,double rest,const double*observe,double threshold,double prominence,
 const double*caps,const double*tau,double*q,double*last,double*slope0,double*trough,
 long long*counts,long long*clipped,int*event_count,double*event_time,double*event_jump,double*event_post,int*flag){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 long long ns=split==1?clock[1]/2:clock[1]-clock[1]/2;
 double dt=ns*1e-9,voltage=0.;for(int j=0;j<dim;j++)voltage+=v[i*dim+j]*observe[j];voltage+=rest;
 double slope=voltage-last[i];bool peak=slope0[i]>0&&slope<=0;
 bool event=peak&&last[i]>threshold&&last[i]-trough[i]>=prominence;
 double before=q[i]*exp(-dt/tau[i]);double after=before+(event?1./(caps[i]*tau[i]):0.);
 clipped[i]+=(after>1.);q[i]=fmin(after,1.);counts[i]+=event;
 double jump=q[i]-before;
 if(event){
  int k=event_count[i];if(k>=width){atomicOr(flag,1);}else{
   event_time[i*width+k]=(clock[0]+(split==1?clock[1]/2:clock[1]))*1e-9;
   event_jump[i*width+k]=jump;event_post[i*width+k]=q[i];event_count[i]=k+1;
  }
 }
 if(!isfinite(q[i])||!isfinite(voltage))atomicOr(flag,2);
 trough[i]=peak?voltage:fmin(trough[i],voltage);last[i]=voltage;slope0[i]=slope;
}
extern "C" __global__ void commit_axon(int n,int dim,int axons,int split,const long long*clock,
 const int*coords,const double*v,double rest,double ts,const double*caps,const double*tau,const double*gain,
 double threshold,double prominence,double*q,double*s,double*last,double*slope0,double*trough,long long*counts,long long*clipped,int*flag){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n*axons)return;int cell=j/axons,port=j%axons;
 double dt=(split==1?clock[1]/2:clock[1]-clock[1]/2)*1e-9;
 double tq=tau[cell],eq=exp(-dt/tq),es=exp(-dt/ts),factor;
 if(fabs(tq-ts)<=1e-15)factor=dt/ts*es;else factor=tq*(eq-es)/(tq-ts);
 double voltage=v[cell*dim+coords[port]]+rest,slope=voltage-last[j];bool peak=slope0[j]>0.&&slope<=0.;
 bool event=peak&&last[j]>threshold&&last[j]-trough[j]>=prominence;
 double snext=s[j]*es+gain[cell]*q[j]*factor,qn=q[j]*eq+(event?1./(caps[cell]*tq):0.);
 clipped[j]+=(qn>1.);q[j]=fmin(qn,1.);s[j]=snext;counts[j]+=event;
 if(!isfinite(q[j])||!isfinite(s[j])||!isfinite(voltage))atomicOr(flag,4);
 trough[j]=peak?voltage:fmin(trough[j],voltage);last[j]=voltage;slope0[j]=slope;
}
