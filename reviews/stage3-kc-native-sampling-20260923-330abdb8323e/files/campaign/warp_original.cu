
__device__ void rates(double v,double*s,double*t){
 s[0]=1./(1.+exp(-.1121*(v+29.13)));s[1]=1./(1.+exp(.2*(v+47.)));
 s[2]=1./(1.+exp(-.2717*(v+48.77)));s[3]=1./(1.+exp(-.0502*(v+12.85)));
 t[0]=(.1270+3.434/(1.+exp((v+45.35)/5.98)))/1000.;
 t[1]=(.36+exp((v+20.65)/-10.47))/1000.;t[2]=.001;
 t[3]=(2.03+1.96/(1.+exp((v-30.83)/3.12)))/1000.;
}
__device__ bool solve(double*A,double*b,double*x,double*error){
 double original[289],right[17];
 for(int k=0;k<289;k++)original[k]=A[k];for(int i=0;i<17;i++)right[i]=b[i];
 for(int j=0;j<17;j++){
  double p=A[j*17+j];if(!(p>0.)||!isfinite(p))return false;
  for(int i=j+1;i<17;i++){
   double factor=A[i*17+j]/p;
   for(int k=j+1;k<17;k++)A[i*17+k]-=factor*A[j*17+k];
   b[i]-=factor*b[j];
  }
 }
 for(int i=16;i>=0;i--){double v=b[i];for(int j=i+1;j<17;j++)v-=A[i*17+j]*x[j];x[i]=v/A[i*17+i];}
 double amax=0.,bmax=0.,xmax=0.,rmax=0.;
 for(int i=0;i<17;i++){
  double r=-right[i],a=0.;for(int j=0;j<17;j++){r+=original[i*17+j]*x[j];a+=fabs(original[i*17+j]);}
  rmax=fmax(rmax,fabs(r));amax=fmax(amax,a);bmax=fmax(bmax,fabs(right[i]));xmax=fmax(xmax,fabs(x[i]));
 }
 *error=rmax/fmax(amax*xmax+bmax,1e-300);
 return isfinite(*error)&&*error<=1e-12;
}

extern "C" __global__ void warp_midpoint(int count,double dt,double rest,const double*v,const double*g,
 const double*ge,const double*gi,const double*cur,const double*C,const double*G,
 const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
 double*vo,double*go,double*errors){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=17)return;
 const unsigned mask=0x1ffff;double A[17],original[17],b,br,x=0.,pred=0.,s[4],tau[4],gm[4];
 double f0,f1,f2,ratio=2./dt,v0=v[n*17+i];
 rates(v0+rest,s,tau);for(int q=0;q<4;q++)gm[q]=s[q]+(g[n*68+i*4+q]-s[q])*exp(-.5*dt/tau[q]);
 f0=gm[0]*gm[0]*gm[0]*gm[1];f1=gm[2];f2=gm[3]*gm[3]*gm[3]*gm[3];
 for(int stage=0;stage<2;stage++){
  double rhs=cur[n*17+i];
  for(int z=0;z<4;z++)rhs+=(-rest*ge[n*4+z]+(-68.-rest)*gi[n*4+z])*shuntb[z*17+i];
  for(int j=0;j<17;j++){
   int ij=i*17+j;double k=G[ij];for(int z=0;z<4;z++)k+=(ge[n*4+z]+gi[n*4+z])*shuntG[z*289+ij];
   A[j]=k;
  }
  for(int port=0;port<17;port++){
   double ff[3]={__shfl_sync(mask,f0,port),__shfl_sync(mask,f1,port),__shfl_sync(mask,f2,port)};
   for(int q=0;q<3;q++){
    int z=port*3+q;rhs+=ff[q]*ena[z]*chanb[z*17+i];
    for(int j=0;j<17;j++)A[j]+=ff[q]*chanG[z*289+i*17+j];
   }
  }
  if(stage==1)rhs*=2.;
  for(int j=0;j<17;j++){
   double vj=__shfl_sync(mask,v0,j),k=A[j];
   rhs+=(ratio*C[i*17+j]-(stage==1?k:0.))*vj;A[j]=k+ratio*C[i*17+j];original[j]=A[j];
  }
  b=rhs;br=rhs;
  for(int j=0;j<17;j++){
   double pivot=__shfl_sync(mask,A[j],j),bj=__shfl_sync(mask,b,j);
   if(!(pivot>0.)||!isfinite(pivot)){if(i==0)errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
   double factor=i>j?A[j]/pivot:0.;
   for(int k=j+1;k<17;k++){double ajk=__shfl_sync(mask,A[k],j);if(i>j)A[k]-=factor*ajk;}
   if(i>j)b-=factor*bj;
  }
  for(int j=16;j>=0;j--){
   if(i==j)x=b/A[j];double xj=__shfl_sync(mask,x,j);if(i<j)b-=A[j]*xj;
  }
  double residual=-br,row=0.;for(int j=0;j<17;j++){double xj=__shfl_sync(mask,x,j);residual+=original[j]*xj;row+=fabs(original[j]);}
  double rmax=fabs(residual),amax=row,bmax=fabs(br),xmax=fabs(x);
  for(int o=16;o>0;o/=2){
   double rr=__shfl_down_sync(mask,rmax,o),aa=__shfl_down_sync(mask,amax,o),bb=__shfl_down_sync(mask,bmax,o),xx=__shfl_down_sync(mask,xmax,o);
   if(i+o<17){rmax=fmax(rmax,rr);amax=fmax(amax,aa);bmax=fmax(bmax,bb);xmax=fmax(xmax,xx);}
  }
  double err=__shfl_sync(mask,rmax/fmax(amax*xmax+bmax,1e-300),0);
  if(i==0)errors[n*2+stage]=err;
  if(!isfinite(err)||err>1e-12){if(i==0)errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
  if(stage==0){
   pred=x;rates(pred+rest,s,tau);for(int q=0;q<4;q++){
    double old=g[n*68+i*4+q];gm[q]=s[q]+(old-s[q])*exp(-.5*dt/tau[q]);go[n*68+i*4+q]=s[q]+(old-s[q])*exp(-dt/tau[q]);
   }
   bool bad=false;for(int q=0;q<4;q++){double val=go[n*68+i*4+q];bad|=!isfinite(val)||val<0.||val>1.;}
   if(__any_sync(mask,bad)){if(i==0)errors[n*2+stage]=__longlong_as_double(0x7ff0000000000000LL);return;}
   f0=gm[0]*gm[0]*gm[0]*gm[1];f1=gm[2];f2=gm[3]*gm[3]*gm[3]*gm[3];
  }
 }
 if(__any_sync(mask,!isfinite(x))){if(i==0)errors[n*2+1]=__longlong_as_double(0x7ff0000000000000LL);return;}
 vo[n*17+i]=x;
}
