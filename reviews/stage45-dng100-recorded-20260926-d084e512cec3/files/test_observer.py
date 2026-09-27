"""Software checks only; no organism, training or scientific parameter search."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time
import numpy as np
import cupy as cp
HERE=Path(__file__).resolve().parent
ENGINE=HERE.parents[1]/'motor_nuevo/full_pipeline_review_20260925_12/engine'
sys.path.insert(0,str(ENGINE))
from observer import Observer,WIDTH

def need(ok,message):
    if not ok:raise RuntimeError(message)

def main():
    start=time.monotonic();n=64;rng=np.random.default_rng(47)
    ptr=cp.asarray(np.arange(n+1,dtype=np.int64)*47)
    idx=cp.asarray(rng.integers(0,n,n*47,dtype=np.int32))
    w=cp.asarray(rng.normal(size=n*47).astype(np.float32))
    release=cp.asarray(rng.uniform(size=n).astype(np.float32))
    caps=cp.asarray(rng.uniform(1,200,size=n).astype(np.float32))
    visual=cp.zeros(n,dtype=cp.bool_);visual[4:8]=True
    tau=cp.full(n,.01,dtype=cp.float32);gain=cp.full(n,.001,dtype=cp.float32)
    theta=cp.full(n,1,dtype=cp.float32);drive=cp.zeros(n,dtype=cp.float32)
    photo=cp.ones(n,dtype=cp.float32);rows=cp.asarray([36,46],dtype=cp.int32)
    a=cp.empty(n);r=cp.empty(n);b=cp.empty(n);s=cp.empty(n);diag=cp.full((2,10),np.nan)
    options=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true')
    original=cp.RawKernel((ENGINE/'fast_fp32_coefficient.cu').read_text(),'coefficient_fast',options=options)
    observed=cp.RawKernel((HERE/'coefficient_observed.cu').read_text(),'coefficient_fast',options=options)
    common=(np.int32(n),ptr,idx,w,release,caps,visual,tau,gain,theta,drive,photo,np.float32(1),np.bool_(True))
    grid=((n*32+255)//256,)
    original(grid,(256,),(*common,a,r));observed(grid,(256,),(*common,b,s,rows,diag))
    need(np.array_equal(a.get(),b.get()) and np.array_equal(r.get(),s.get()),'Diagnostic modified neural operator')
    values=diag.get();need(np.isfinite(values).all(),'Diagnostic missing')
    need(np.array_equal(values[:,6],b.get()[[36,46]]),'Base target capture')
    module=cp.RawModule(code=(HERE/'observe.cu').read_text(),options=('--std=c++17','--fmad=false'))
    rhs=module.get_function('capture_rhs');trial=module.get_function('capture_trial')
    scratch=cp.empty((4,2,16));count=cp.zeros(1,dtype=cp.uint64)
    records=cp.full((2,WIDTH),np.nan);state=cp.full(n,.2);fine=cp.full(n,.25)
    clock=cp.asarray([0.,.001,.001]);status=cp.zeros(3)
    # A downstream owner override must be recorded separately from CSR output.
    b[36]=.7;derivative=s*(b-state)
    for i,f in enumerate([0.,.5,.75,1.]):
        rhs((1,),(2,),(rows,state,b,s,derivative,clock,np.float64(f),np.int32(i),diag,scratch))
    trial((1,),(256,),(rows,scratch,clock,status,state,fine,count,records,np.int32(2)))
    status[0]=2.
    trial((1,),(256,),(rows,scratch,clock,status,state,fine,count,records,np.int32(2)))
    host=records.get();need(int(count.get()[0])==2,'Append count')
    need(np.array_equal(host[:,138],[1,0]),'Accepted/rejected separation')
    points=host[0,:128].reshape(4,2,16)
    need(np.all(points[:,0,11]==.7),'Final consumed target override missing')
    need(np.all(points[:,0,7]==values[0,6]),'Base target overwritten')
    need(np.array_equal(points[:,0,14],[0.,.0005,.00075,.001]),'Stage clock')
    # Overflow is observable and must leave existing data intact.
    trial((1,),(256,),(rows,scratch,clock,status,state,fine,count,records,np.int32(2)))
    need(int(count.get()[0])==3 and np.array_equal(host,records.get()),'Overflow guard')
    out=dict(status='PASS',operator_outputs_exact=True,owner_override_distinguished=True,
             acceptance_flags_distinguished=True,overflow_visible_and_non_destructive=True,
             organism_ms=0,wall_s=time.monotonic()-start)
    (HERE/'SOFTWARE_TEST.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
