#include <math_constants.h>
__device__ double sigmoid(double z){double e=exp(-fabs(z));return z>=0?1/(1+e):e/(1+e);}
extern "C" __global__ void channels(int N,const double*v,const double*base,double q,const double*gbar,const double*E,double*x,double*ionic,double*jac,double*ge,int*flag){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;double w=v[i];double steady[4]={sigmoid(.1121*(w+29.13)),sigmoid(-.2*(w+47)),sigmoid(.2717*(w+48.77)),sigmoid(.0502*(w+12.85))};
 double tau[4]={(.127+3.434*sigmoid(-(w+45.35)/5.98))/1000,(.36+exp((w+20.65)/-10.47))/1000,.001,(2.03+1.96*sigmoid(-(w-30.83)/3.12))/1000};
 double sm=(tau[0]*1000-.127)/3.434,sn=(tau[3]*1000-2.03)/1.96;double dtau[4]={-3.434/5.98*sm*(1-sm)/1000,-(tau[1]*1000-.36)/10.47/1000,0.,-1.96/3.12*sn*(1-sn)/1000};double slope[4]={.1121,-.2,.2717,.0502},dx[4],xx[4];ge[i]=0.;
 for(int k=0;k<4;k++){xx[k]=(tau[k]*base[4*i+k]+q*steady[k])/(tau[k]+q);x[4*i+k]=xx[k];dx[k]=(q*steady[k]*(1-steady[k])*slope[k]+dtau[k]*(base[4*i+k]-xx[k]))/(tau[k]+q);double error=fabs(xx[k]-base[4*i+k]-q*(steady[k]-xx[k])/tau[k]);if(error>ge[i])ge[i]=error;if(!isfinite(xx[k])||xx[k]<0||xx[k]>1||!isfinite(tau[k])||tau[k]<=0)atomicOr(flag,1);}
 double m=xx[0],h=xx[1],p=xx[2],n=xx[3];double g[3]={gbar[3*i]*m*m*m*h,gbar[3*i+1]*p,gbar[3*i+2]*n*n*n*n};double dg[3]={gbar[3*i]*(3*m*m*h*dx[0]+m*m*m*dx[1]),gbar[3*i+1]*dx[2],gbar[3*i+2]*4*n*n*n*dx[3]};jac[i]=0.;
 for(int k=0;k<3;k++){ionic[3*i+k]=g[k]*(w-E[k]);jac[i]+=g[k]+dg[k]*(w-E[k]);if(!isfinite(ionic[3*i+k]))atomicOr(flag,2);}if(!isfinite(jac[i]))atomicOr(flag,2);
}
extern "C" __global__ void calcium(int N,int K,const double*v,const double*base,double q,const double*half,const double*slope,const double*tau,const double*power,const double*gbar,double reversal,int enabled,double*x,double*cur,double*jac,double*frozen,double*ge,int*flag){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;double dx[4],factors[4],product=1.;ge[i]=0.;
 for(int k=0;k<K;k++){int at=i*K+k;double steady=sigmoid((v[i]-half[at])/slope[at]);x[at]=(tau[at]*base[at]+q*steady)/(tau[at]+q);dx[k]=q*steady*(1-steady)/(slope[at]*(tau[at]+q));factors[k]=pow(x[at],power[k]);product*=factors[k];double error=fabs(x[at]-base[at]-q*(steady-x[at])/tau[at]);if(error>ge[i])ge[i]=error;if(!isfinite(x[at])||x[at]<0||x[at]>1)atomicOr(flag,1);}
 double derivative=0.;for(int k=0;k<K;k++){double other=1.;for(int j=0;j<K;j++)if(j!=k)other*=factors[j];derivative+=power[k]*pow(x[i*K+k],power[k]-1)*dx[k]*other;}
 double g=enabled?gbar[i]*product:0.,dg=enabled?gbar[i]*derivative:0.;frozen[i]=g;cur[i]=g*(v[i]-reversal);jac[i]=g+dg*(v[i]-reversal);if(!isfinite(cur[i]+jac[i]))atomicOr(flag,2);
}
extern "C" __global__ void difference(int N,const int*ptr,const int*idx,const double*data,const double*sums,const double*v,double*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;double total=sums[i]*v[i],correction=0.;for(int k=ptr[i];k<ptr[i+1];k++){int j=idx[k];if(j==i)continue;double term=data[k]*(v[j]-v[i]),value=total+term;correction+=fabs(total)>=fabs(term)?(total-value)+term:(term-value)+total;total=value;}out[i]=total+correction;
}
extern "C" __global__ void csr(int N,const int*ptr,const int*idx,const double*data,const double*v,int minimum,double*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;double total=0.,correction=0.;for(int k=ptr[i];k<ptr[i+1];k++){double term=data[k]*v[idx[k]],value=total+term;correction+=fabs(total)>=fabs(term)?(total-value)+term:(term-value)+total;total=value;}out[i]=ptr[i+1]-ptr[i]>=minimum?total+correction:total;
}
extern "C" __global__ void scatter_current(int N,const long long*nodes,const double*values,int stride,double*out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;double s=0;for(int j=0;j<stride;j++)s+=values[i*stride+j];out[nodes[i]]+=s;}
extern "C" __global__ void synapse_current(int N,const long long*nodes,const double*g,const double*E,double leak,const double*v,double*r){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;long long node=nodes[i];r[node]+=g[i]*(v[node]+leak-E[i]);}
extern "C" __global__ void reduce_values(int N,const double*x,int mode,double*out){
 __shared__ double s[256];int t=threadIdx.x;double v=mode==1?CUDART_INF:(mode==2?-CUDART_INF:0.);
 for(int i=blockIdx.x*256+t;i<N;i+=gridDim.x*256){double a=x[i];if(mode==0)a*=a;if(isnan(a)||isnan(v))v=CUDART_NAN;else v=mode==1?fmin(v,a):(mode==2?fmax(v,a):v+a);}
 s[t]=v;__syncthreads();for(int off=128;off;off/=2){if(t<off){double a=s[t],b=s[t+off];s[t]=(isnan(a)||isnan(b))?CUDART_NAN:(mode==1?fmin(a,b):(mode==2?fmax(a,b):a+b));}__syncthreads();}if(t==0)out[blockIdx.x]=s[0];
}
extern "C" __global__ void reduce_finish(int N,const double*x,int mode,double*out){
 __shared__ double s[256];int t=threadIdx.x;double v=t<N?x[t]:(mode==1?CUDART_INF:(mode==2?-CUDART_INF:0.));s[t]=v;__syncthreads();for(int off=128;off;off/=2){if(t<off){double a=s[t],b=s[t+off];s[t]=(isnan(a)||isnan(b))?CUDART_NAN:(mode==1?fmin(a,b):(mode==2?fmax(a,b):a+b));}__syncthreads();}if(t==0)out[0]=mode==0?sqrt(s[0]):s[0];
}
extern "C" __global__ void controls(const double*ref,const double*n0,const double*n1,const double*n2,const double*ge,const double*ce,const double*l0,const double*l1,const double*s0,const double*s1,const int*flag,double*out){if(threadIdx.x||blockIdx.x)return;out[0]=ref[0];out[1]=n0[0];out[2]=n1[0];out[3]=n2[0];out[4]=ge[0];out[5]=ce[0];out[6]=l0[0];out[7]=l1[0];out[8]=s0[0];out[9]=s1[0];out[10]=(double)flag[0];}
