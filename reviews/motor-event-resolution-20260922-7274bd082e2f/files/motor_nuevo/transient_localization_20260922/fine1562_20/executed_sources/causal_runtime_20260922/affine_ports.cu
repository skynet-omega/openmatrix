// Declared affine state blocks (constant coordinate included by model adapter).
// Timestamped jumps and coefficient switches are applied in chronological order.
// exp(dt L)y by bounded Taylor action, ||dt L||_infinity <= 0.5 per substep.
// This integrates the RECEIVER response, not only the area of a port waveform.
extern "C" __global__ void affine_prefix(int blocks,int dim,int segments,
 const double*matrix,const double*times,const double*jumps,const double*initial,
 double until,double*out,int*status){
 int n=blockIdx.x,i=threadIdx.x;if(n>=blocks||i>=dim)return;
 unsigned mask=dim==32?0xffffffffu:((1u<<dim)-1u);
 double y=initial[n*dim+i];int work=0;
 for(int segment=0;segment<segments;segment++){
  double start=times[segment];if(start>until)break;
  y+=jumps[(n*segments+segment)*dim+i];
  if(__any_sync(mask,!isfinite(y))){if(i==0)status[n]=3;return;}
  if(start==until)break;
  double end=fmin(times[segment+1],until),duration=end-start;
  const double*A=matrix+((n*segments+segment)*dim+i)*dim;
  double row=0.;for(int j=0;j<dim;j++)row+=fabs(A[j]);
  double norm=row;
  for(int j=0;j<dim;j++)norm=fmax(norm,__shfl_sync(mask,row,j));
  double need=ceil(duration*norm/.5);
  if(!isfinite(need)||need>4096){if(i==0)status[n]=1;return;}
  int steps=need>1?(int)need:1;work+=steps;
  if(work>16384){if(i==0)status[n]=2;return;}
  double dt=duration/steps;
  for(int step=0;step<steps;step++){
   double sum=y,term=y;
   for(int k=1;k<=20;k++){
    double next=0.;for(int j=0;j<dim;j++)next+=A[j]*__shfl_sync(mask,term,j);
    term=next*(dt/k);sum+=term;
   }
   if(__any_sync(mask,!isfinite(sum))){if(i==0)status[n]=3;return;}
   y=sum;
  }
  if(end==until&&until<times[segment+1])break;
 }
 out[n*dim+i]=y;
}
