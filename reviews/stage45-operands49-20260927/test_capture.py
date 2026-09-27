"""CUDA fixture: coefficient parity, six-row operands and owned trial copies."""
from pathlib import Path
import hashlib,json,time,resource
import numpy as np
import cupy as cp
H=Path(__file__).resolve().parent
P=H.parent/'etapa45_composicion_20260927_48/coefficient_observed.cu'
OPT=('--std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true')
def need(x,m):
 if not x: raise ValueError(m)
def same(a,b,m):need(np.array_equal(a,b),m)
def warp(x):
 lane=np.zeros(32,np.float32)
 for start in range(0,len(x),32):lane[:len(x[start:start+32])]+=x[start:start+32]
 for d in (16,8,4,2,1):lane[:32-d]+=lane[d:].copy()
 return lane[0]
def main():
 wall=time.monotonic();cpu=time.process_time();rng=np.random.default_rng(4901)
 n=8;lengths=np.array([67,39,1,0,48,70,33,65]);ptr=np.r_[0,np.cumsum(lengths)].astype(np.int64)
 idx=rng.integers(0,n,size=ptr[-1],dtype=np.int32);w=rng.uniform(-2,2,len(idx)).astype(np.float32)
 release=np.linspace(.1,.8,n,dtype=np.float32);caps=np.linspace(1,2,n,dtype=np.float32)
 visual=np.zeros(n,bool);visual[7]=True
 tau=np.linspace(.01,.03,n,dtype=np.float32);gain=np.full(n,.1,np.float32)
 theta=np.array([-100,100,0,0,2,-2,1,1],np.float32);drive=np.zeros(n,np.float32);photo=np.zeros(n,np.float32)
 rows=np.arange(6,dtype=np.int32);optr=ptr[:7].copy();E=int(optr[-1])
 base=[np.int32(n),*[cp.asarray(x) for x in (ptr,idx,w,release,caps,visual,tau,gain,theta,drive,photo)],np.float32(1)]
 old=cp.RawKernel(P.read_text(),'coefficient_fast',options=OPT);new=cp.RawKernel((H/'coefficient_capture.cu').read_text(),'coefficient_fast',options=OPT)
 dst=[cp.empty(n,cp.float64) for _ in range(4)];csr2=cp.empty((2,10),cp.float64);csr6=cp.empty((6,10),cp.float64);ops=cp.empty((E,4),cp.float32)
 rg=cp.asarray(rows);opg=cp.asarray(optr);checks=[]
 for connected in (False,True):
  old((1,),(256,),(*base,np.bool_(connected),dst[0],dst[1],rg,csr2))
  new((1,),(256,),(*base,np.bool_(connected),dst[2],dst[3],rg,opg,csr6,ops))
  same(dst[0].get(),dst[2].get(),'target arithmetic altered');same(dst[1].get(),dst[3].get(),'rate altered')
  same(csr2.get(),csr6.get()[:2],'two-row witness altered')
  expected=np.column_stack((w[:E],release[idx[:E]],caps[idx[:E]],np.logical_or(connected,~visual[idx[:E]]))).astype(np.float32)
  same(ops.get(),expected,'actual operands/mask')
  obs=csr6.get()
  for j in rows:
   e=expected[optr[j]:optr[j+1]];terms=np.where(e[:,3]!=0,e[:,0]*(e[:,1]*e[:,2]),np.float32(0))
   same(warp(terms),obs[j,0],'FP32 warp net')
   margin=np.float32(np.float32(obs[j,0])+drive[j])-theta[j]
   same(margin,obs[j,9],'FP32 margin')
  need(dst[2].get()[0]>0 and dst[2].get()[1]==0,'positive and negative controls')
  checks.append('coefficient_exact_connected_'+str(connected))
 mod=cp.RawModule(code=(H/'capture.cu').read_text(),options=OPT)
 krhs,kstage,kcopy,ktrial=[mod.get_function(s) for s in ('capture_rhs','copy_stage','copy_trial_operands','capture_trial')]
 stages=cp.empty((4,E,4),cp.float32);stored=cp.full((2,4,E,4),np.nan,cp.float32)
 scratch=cp.empty((4,6,16),cp.float64);records=cp.full((2,404),np.nan,cp.float64);count=cp.zeros(1,cp.uint64)
 clock=cp.asarray([0.,.001,.001],dtype=cp.float64);status=cp.zeros(3,cp.float64)
 y=cp.linspace(.01,.08,8,dtype=cp.float64);after=y+.001;rhs=dst[3]*(dst[2]-y)
 witnesses=[];switness=[]
 for trial in range(3):
  stagecopy=[]
  for st,f in enumerate((0.,.5,.75,1.)):
   ops.fill(trial*10+st);stagecopy.append(ops.get())
   krhs((1,),(6,),(rg,y,dst[2],dst[3],rhs,clock,np.float64(f),np.int32(st),csr6,scratch))
   kstage(((E*4+255)//256,),(256,),(ops,stages,np.int32(E*4),np.int32(st)))
  status[0]=.5 if trial==0 else 2.
  kcopy(((E*16+255)//256,),(256,),(stages,count,stored,np.int32(E*16),np.int32(2)))
  ktrial((1,),(256,),(rg,scratch,clock,status,y,after,count,records,np.int32(2)))
  if trial<2:witnesses.append(np.array(stagecopy));switness.append(scratch.get())
 cp.cuda.get_current_stream().synchronize()
 same(stored.get(),np.array(witnesses),'owned operand records/overflow');rr=records.get()
 same(rr[:,:384],np.array(switness).reshape(2,384),'owned scalar records');same(rr[:,402],[1.,0.],'acceptance flags');same(rr[:,403],[0.,1.],'trial sequence')
 need(int(count.get()[0])==3,'Overflow must remain detectable')
 same(rr[:,390:396],np.tile(y.get()[:6],(2,1)),'before state');same(rr[:,396:402],np.tile(after.get()[:6],(2,1)),'after state')
 checks+=['four_RHS_stages','accepted_and_rejected_trial','record_ownership','overflow_detectable','both_side_state_copy']
 # Deliberately changing a saved operand must fail the exact reconstruction.
 corrupt=witnesses[0].copy();corrupt[0,0,0]+=1
 need(not np.array_equal(corrupt,stored.get()[0]),'corruption control')
 report=dict(schema='capture49_fixture_v1',passed=True,checks=checks,neural_ms=0,body_steps=0,cpu_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,device=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P,H/'coefficient_capture.cu',H/'capture.cu']})
 out=H/('FIXTURE_OPTIMIZED.json' if not __debug__ else 'FIXTURE.json');need(not out.exists(),'Preserve fixture output');out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
