"""Portable reconstruction from same-call operands; no CuPy or CNS needed."""
from pathlib import Path
import argparse,json,time,hashlib,math
HERE=Path(__file__).resolve().parent
REFERENCE_SHA256='9414489fb9f8a238d7ac2ec9a6ee318597be6612e467227a415bd3d0625158d6'
import numpy as np
IDS=np.array([10045,10056,10118,10065,523769,10360],np.int64)
FIELDS=('state','net','positive_aux','negative_aux','drive','theta','gain','base_target','base_rate','tau','margin','final_target','final_rate','derivative','evaluation_time_s','stage_fraction')
def need(x,m):
 if not x:raise ValueError(m)
def same(a,b,m):need(np.array_equal(a,b),m)
def warp(terms):
 lanes=np.zeros((*terms.shape[:-1],32),np.float32)
 for start in range(0,terms.shape[-1],32):
  part=terms[...,start:start+32];lanes[...,:part.shape[-1]]+=part
 for d in (16,8,4,2,1):lanes[...,:32-d]+=lanes[...,d:].copy()
 return lanes[...,0]
def verify(z):
 for name in z:
  a=z[name]
  if a.dtype.kind in 'fc':need(np.isfinite(a).all(),'Nonfinite '+name)
 ref=HERE/'frozen_selection.npz'
 need(hashlib.sha256(ref.read_bytes()).hexdigest()==REFERENCE_SHA256,'Frozen selection hash')
 with np.load(ref,allow_pickle=False) as original:reference={k:original[k] for k in original.files}
 for current,old in [('ids','target_ids'),('rows','target_rows'),('ptr','ptr'),('positions','positions'),('pre_rows','pre_rows'),('pre_ids','pre_ids')]:same(z[current],reference[old],'Frozen anatomy '+current)
 same(z['ids'],IDS,'Identity');same(z['fields'],FIELDS,'RHS fields')
 same(z['operand_fields'],['weight_consumed','transmission_consumed','cap_consumed','included'],'Operand fields')
 r=z['records'];o=z['operands'];p=z['ptr'];N=len(r)
 need(r.dtype==np.float64 and r.shape==(N,404),'Scalar schema')
 need(o.dtype==np.float32 and o.shape==(N,4,7227,4),'Operand schema')
 need(p.dtype.kind in 'iu' and p.shape==(7,) and p[0]==0 and p[-1]==7227 and np.all(np.diff(p)>0),'CSR ptr')
 need(np.all((o[...,3]==0)|(o[...,3]==1)),'Inclusion flags')
 rhs=r[:,:384].reshape(N,4,6,16)
 # Generic cells retain this operator's final target/rate; no direct overrides.
 same(rhs[...,7],rhs[...,11],'Target overridden');same(rhs[...,8],rhs[...,12],'Rate overridden')
 same(rhs[...,12]*(rhs[...,11]-rhs[...,0]),rhs[...,13],'Derivative mismatch')
 need(np.all(rhs[...,9]>0),'Nonpositive tau')
 same((np.float32(1)/rhs[...,9].astype(np.float32)).astype(np.float64),rhs[...,8],'Rate/tau mismatch')
 same(rhs[...,15],np.broadcast_to(np.array([0.,.5,.75,0.])[None,:,None],(N,4,6)),'RK stage order')
 et=r[:,384,None,None]+rhs[...,15]*r[:,385,None,None]
 et[:,3,:]=np.nextafter(r[:,386],r[:,384])[:,None]
 same(et,rhs[...,14],'RHS clock mismatch')
 for name,index in [('sham_cuda_theta',5),('sham_cuda_gain',6),('sham_cuda_tau',9)]:same(rhs[...,index],np.broadcast_to(reference[name].astype(np.float32),(N,4,6)),'Frozen parameter '+name)
 need(np.all(rhs[...,4]==0),'Undeclared destination drive')
 same(o[...,0],np.broadcast_to(reference['sham_cuda_weights'].astype(np.float32),(N,4,7227)),'Frozen effective weights')
 same(o[...,2],np.broadcast_to(reference['caps'].astype(np.float32),(N,4,7227)),'Frozen caps')
 margins=[];max_target_ulp=0
 for cell in range(6):
  a=o[:,:,p[cell]:p[cell+1]]
  t=np.where(a[...,3]!=0,a[...,0]*(a[...,1]*a[...,2]),np.float32(0))
  net=warp(t)
  same(net,rhs[:,:,cell,1],'Consumed FP32 net '+str(IDS[cell]))
  same(warp(np.where(t>=0,t,np.float32(0))),rhs[:,:,cell,2],'Positive auxiliary')
  same(warp(np.where(t<0,t,np.float32(0))),rhs[:,:,cell,3],'Negative auxiliary')
  margin=np.float32(net+rhs[:,:,cell,4].astype(np.float32))-rhs[:,:,cell,5].astype(np.float32)
  same(margin,rhs[:,:,cell,10],'FP32 margin')
  zero=(rhs[:,:,cell,6]*margin)<=0
  need(np.all(rhs[:,:,cell,7][zero]==0),'Negative argument did not rectify')
  need(np.all((rhs[:,:,cell,7]>=0)&(rhs[:,:,cell,7]<=1)),'Target range')
  target=rhs[:,:,cell,7].astype(np.float32)
 same(target.astype(np.float64),rhs[:,:,cell,7],'Target FP32 representation')
 argument=rhs[:,:,cell,6].astype(np.float32)*margin
 desired=np.array([max(0.,math.tanh(float(v))) for v in argument.flat],np.float32).reshape(argument.shape)
 ulp=np.abs(target.view(np.uint32).astype(np.int64)-desired.view(np.uint32).astype(np.int64))
 need(np.all(ulp<=2),'Positive tanhf beyond documented2ULP')
 max_target_ulp=max(max_target_ulp,int(ulp.max()))
 margins.append([float(margin.min()),float(margin.max())])
 offsets=z['offsets'];K=len(offsets)-1
 need(offsets.dtype.kind in 'iu' and offsets[0]==0 and offsets[-1]==N and np.all(np.diff(offsets)>0),'Offsets')
 need(K==16,'One complete recorded macrostep required')
 ms=z['ms'];need(np.all(ms==ms[0]) and 3001<=ms[0]<=3200,'Macrostep identity')
 slots=np.arange(16);expected_committed=slots%2==1
 need(z['committed'].dtype==np.bool_,'Committed dtype')
 same(z['committed'],expected_committed,'Predictor/committed ownership')
 same(z['duration_ns'],np.where(expected_committed,125000,62500),'Epoch durations')
 same(z['start_ns'],47486000000+(ms[0]-3001)*1000000+(slots//2)*125000,'Absolute epoch clocks')
 same(z['epoch'],(ms[0]-3001)*16+slots,'Global epoch sequence')
 for k in range(K):
  q=r[offsets[k]:offsets[k+1]];need(len(q)==z['trials'][k],'Trial count')
  same(q[:,403],np.arange(len(q)),'Trial sequence')
  need(np.all((q[:,402]==0)|(q[:,402]==1)),'Acceptance flag not binary')
  accept=(q[:,387]<=1)&(q[:,388]==0)&(q[:,389]==0)
  same(q[:,402],accept.astype(np.float64),'Acceptance corruption')
  need(int(accept.sum())==z['accepted'][k] and int((~accept).sum())==z['rejected'][k],'Accepted/rejected counts')
  last=0.;state=q[0,390:396].copy()
  for row,ok in zip(q,accept):
   need(row[384]==last,'Trial start clock');same(row[390:396],state,'Before/reject state')
   if ok:last=row[386];state=row[396:402].copy()
  need(last==z['duration_ns'][k]*1e-9,'Epoch duration')
 return dict(files=1,epochs=K,trials=N,RHS_evaluations=N*4,exact_sums=N*4*6,margin_ranges=margins,maximum_positive_target_ULP=max_target_ulp,stage4_pass=False,stage5_pass=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('files',nargs='+',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
 start=time.process_time();out=[]
 for path in args.files:
  with np.load(path,allow_pickle=False) as z:r=verify(z)
  out.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**r))
 report=dict(schema='operand49_verified_v1',results=out,cpu_s=time.process_time()-start)
 if args.output:
  need(not args.output.exists(),'Preserve previous verification');args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))
if __name__=='__main__':main()
