extern "C" __global__ void ros_trial(int count,const long long* clock,double rest,const double*v,const double*g,
const double*ge,const double*gi,const double*cur,const double*C,const double*G,
const double*chanG,const double*chanb,const double*shuntG,const double*shuntb,const double*ena,
double*kv,double*kg,double*vh,double*gh,double*vo,double*go,double*error){
 int n=blockIdx.x,i=threadIdx.x;if(n>=count||i>=17)return;const unsigned mask=0x1ffff;
 double h=clock[1]*1e-9,shift=1./(.435866521508459*h),fraction=double(clock[1]/2)/double(clock[1]);
 double v0=v[n*17+i],g0[4],s[4],tau[4],dgate[4],A[17],original[17],stagev=v0,stageg[4];
 rates(v0+rest,s,tau);
 for(int q=0;q<4;q++){g0[q]=g[n*68+i*4+q];dgate[q]=shift+1./tau[q];}
 // Frozen block-diagonal W: exact partial voltage Jacobian at initial gates,
 // exact gate self-derivative. Cross blocks are deliberately approximate zero.
 double f0=g0[0]*g0[0]*g0[0]*g0[1],f1=g0[2],f2=g0[3]*g0[3]*g0[3]*g0[3];
 #pragma unroll
 for(int j=0;j<17;j++){
  A[j]=G[i*17+j]+shift*C[i*17+j];
  for(int z=0;z<4;z++)A[j]+=(ge[n*4+z]+gi[n*4+z])*shuntG[z*289+i*17+j];
 }
 for(int p=0;p<17;p++){
  double ff[3]={__shfl_sync(mask,f0,p),__shfl_sync(mask,f1,p),__shfl_sync(mask,f2,p)};
  for(int q=0;q<3;q++){
   #pragma unroll
   for(int j=0;j<17;j++)A[j]+=ff[q]*chanG[(p*3+q)*289+i*17+j];
  }
 }
 #pragma unroll
 for(int j=0;j<17;j++)original[j]=A[j];
 // Positive-pivot LU for this passive-voltage W block; residual checked at
 // every stage. No assumption of positivity about the full nonlinear RHS.
 #pragma unroll
 for(int j=0;j<17;j++){
  double pivot=__shfl_sync(mask,A[j],j);
  if(!(pivot>0.)||!isfinite(pivot)){if(i==0)error[n]=__longlong_as_double(0x7ff0000000000000LL);return;}
  double factor=i>j?A[j]/pivot:0.;
  #pragma unroll
  for(int k=j+1;k<17;k++){double value=__shfl_sync(mask,A[k],j);if(i>j)A[k]-=factor*value;}
  if(i>j)A[j]=factor;
 }
 for(int stage=0;stage<4;stage++){
  stagev=v0;double cv=0.;
  for(int q=0;q<4;q++)stageg[q]=g0[q];
  for(int prev=0;prev<stage;prev++){
   double k=kv[(n*4+prev)*17+i];stagev+=AT[stage*4+prev]*k;cv+=GI[stage*4+prev]*k/h;
   for(int q=0;q<4;q++)stageg[q]+=AT[stage*4+prev]*kg[((n*4+prev)*17+i)*4+q];
  }
  rates(stagev+rest,s,tau);
  for(int q=0;q<4;q++){
   double rhs=(s[q]-stageg[q])/tau[q];
   for(int prev=0;prev<stage;prev++)rhs-=GI[stage*4+prev]*kg[((n*4+prev)*17+i)*4+q]/h;
   kg[((n*4+stage)*17+i)*4+q]=rhs/dgate[q];
  }
  f0=stageg[0]*stageg[0]*stageg[0]*stageg[1];f1=stageg[2];f2=stageg[3]*stageg[3]*stageg[3]*stageg[3];
  double rhs=cur[n*17+i];
  for(int z=0;z<4;z++)rhs+=(-rest*ge[n*4+z]+(-68.-rest)*gi[n*4+z])*shuntb[z*17+i];
  #pragma unroll
  for(int j=0;j<17;j++){
   double passive=G[i*17+j];for(int z=0;z<4;z++)passive+=(ge[n*4+z]+gi[n*4+z])*shuntG[z*289+i*17+j];
   rhs-=passive*__shfl_sync(mask,stagev,j)+C[i*17+j]*__shfl_sync(mask,cv,j);
  }
  for(int p=0;p<17;p++){
   double ff[3]={__shfl_sync(mask,f0,p),__shfl_sync(mask,f1,p),__shfl_sync(mask,f2,p)};
   for(int q=0;q<3;q++){
    int z=p*3+q;double flux=ena[z]*chanb[z*17+i];
    #pragma unroll
    for(int j=0;j<17;j++)flux-=chanG[z*289+i*17+j]*__shfl_sync(mask,stagev,j);
    rhs+=ff[q]*flux;
   }
  }
  double b=rhs,x=0.;
  #pragma unroll
  for(int j=0;j<17;j++){double bj=__shfl_sync(mask,b,j);if(i>j)b-=A[j]*bj;}
  #pragma unroll
  for(int j=16;j>=0;j--){if(i==j)x=b/A[j];double xx=__shfl_sync(mask,x,j);if(i<j)b-=A[j]*xx;}
  double residual=-rhs,row=0.;
  #pragma unroll
  for(int j=0;j<17;j++){residual+=original[j]*__shfl_sync(mask,x,j);row+=fabs(original[j]);}
  double rm=fabs(residual),am=row,bm=fabs(rhs),xm=fabs(x);
  for(int o=16;o>0;o/=2){double rr=__shfl_down_sync(mask,rm,o),aa=__shfl_down_sync(mask,am,o),bb=__shfl_down_sync(mask,bm,o),xx=__shfl_down_sync(mask,xm,o);if(i+o<17){rm=fmax(rm,rr);am=fmax(am,aa);bm=fmax(bm,bb);xm=fmax(xm,xx);}}
  double relative=__shfl_sync(mask,rm/fmax(am*xm+bm,1e-300),0);
  if(!isfinite(relative)||relative>1e-12){if(i==0)error[n]=__longlong_as_double(0x7ff0000000000000LL);return;}
  kv[(n*4+stage)*17+i]=x;
  __syncwarp(mask);
 }
 double out=v0,half=v0,ev=0.,outg[4],halfg[4],eg[4]={0,0,0,0};
 for(int q=0;q<4;q++){outg[q]=g0[q];halfg[q]=g0[q];}
 for(int stage=0;stage<4;stage++){
  double weight=fraction*(BIT[stage*3]+fraction*(BIT[stage*3+1]+fraction*BIT[stage*3+2]));
  double k=kv[(n*4+stage)*17+i];out+=BT[stage]*k;half+=weight*k;ev+=(BT[stage]-BET[stage])*k;
  for(int q=0;q<4;q++){double kgq=kg[((n*4+stage)*17+i)*4+q];outg[q]+=BT[stage]*kgq;halfg[q]+=weight*kgq;eg[q]+=(BT[stage]-BET[stage])*kgq;}
 }
 double e=fabs(ev)/2e-5;bool bad=!isfinite(out)||!isfinite(half);
 for(int q=0;q<4;q++){e=fmax(e,fabs(eg[q])/2e-7);bad|=!isfinite(outg[q])||!isfinite(halfg[q]);if(outg[q]<0||outg[q]>1||halfg[q]<0||halfg[q]>1)e=fmax(e,2.);go[n*68+i*4+q]=outg[q];gh[n*68+i*4+q]=halfg[q];}
 if(__any_sync(mask,bad)){if(i==0)error[n]=__longlong_as_double(0x7ff0000000000000LL);return;}
 vo[n*17+i]=out;vh[n*17+i]=half;
 for(int o=16;o>0;o/=2){double ee=__shfl_down_sync(mask,e,o);if(i+o<17)e=fmax(e,ee);}
 if(i==0)error[n]=e;
}
