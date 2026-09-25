"""Model-independent algebra, full mass dynamics and transactional failure."""
from pathlib import Path
import json
import numpy as np
import cupy as cp
from scipy.linalg import expm
from block_executor import BlockExecutor, need

HERE=Path(__file__).resolve().parent


def dense_source():
    source='#include "dense_warp.cuh"\n'
    for n in (1,3,17,32):
        source+=f'''
extern "C" __global__ void solve{n}(const double*A,const double*b,double tol,double*x,double*err,int*ok){{
 int i=threadIdx.x;if(i>={n})return;
 double row[{n}];for(int j=0;j<{n};j++)row[j]=A[i*{n}+j];
 auto r=neurocore::dense_warp_solve<{n}>(row,b[i],tol);
 x[i]=r.x;err[i]=r.error;ok[i]=r.ok;
}}
'''
    return source


def check():
    module=cp.RawModule(code=dense_source(),options=('--std=c++17','--fmad=false','--prec-div=true','-I'+str(HERE)))
    def solve(a,b,tol=1e-12):
        n=len(b);da=cp.asarray(a);db=cp.asarray(b)
        out=cp.empty(n);err=cp.empty(n);ok=cp.empty(n,dtype=cp.int32)
        module.get_function('solve'+str(n))((1,),(32,),(da,db,np.float64(tol),out,err,ok))
        values=out.get();errors=err.get();good=ok.get()
        need(np.array_equal(da.get(),a,equal_nan=True) and np.array_equal(db.get(),b,equal_nan=True),'original system modified')
        return values,errors,good
    rng=np.random.default_rng(937)
    cases=[]
    for n in (1,3,17,32):
        r=rng.normal(size=(n,n));a=r@r.T+np.eye(n)*n;b=rng.normal(size=n)
        x,e,ok=solve(a,b)
        expected=np.linalg.solve(a,b)
        maximum=float(np.max(abs(x-expected)))
        residual=float(np.max(abs(a@x-b))/(np.max(np.sum(abs(a),axis=1))*np.max(abs(x))+np.max(abs(b))))
        need(ok.all() and maximum<1e-11 and residual<1e-12,'dense solve mismatch')
        need(np.array_equal(e,np.full(n,e[0])),'nonuniform residual')
        cases.append({'dimension':n,'max_solution_error':maximum,'original_matrix_residual':residual})
        if n==3:
            fixed_a,fixed_b=a,b
            need(e[0]>0,'strict residual rejection fixture lost its positive error')
            need(not solve(a,b,0.)[2].any(),'residual threshold was ignored')
    for kind in ('singular','negative_pivot','nan','inf','overflow'):
        a=np.eye(3);b=np.ones(3)
        if kind=='singular':a[1,1]=0.
        if kind=='negative_pivot':a[0,0]=-1.
        if kind=='nan':a[0,1]=np.nan
        if kind=='inf':b[2]=np.inf
        if kind=='overflow':a=np.full((3,3),7e307);np.fill_diagonal(a,1e308)
        need(not solve(a,b)[2].any(),'invalid dense system accepted: '+kind)

    # Three coupled RC coordinates with a full nonidentity mass matrix.
    mass=np.array([[2.,.2,.1],[.2,1.5,.15],[.1,.15,1.]])
    stiffness=np.array([[900.,-100.,-50.],[-100.,1200.,-80.],[-50.,-80.,700.]])
    forcing=np.array([[120.,40.,10.],[-50.,25.,90.]])
    initial=np.array([[.1,.3,.2],[.8,.2,.4]])
    state={'y':cp.asarray(initial)}
    engine=BlockExecutor((HERE/'linear_mass_model.cu').read_text(),'linear_mass3',2,state)
    dm,dk,df=cp.asarray(mass),cp.asarray(stiffness),cp.asarray(forcing)
    def bind(s,counts,error,status):
        return (np.int32(2),np.int64(2000000),np.int64(100000),s['y'],dm,dk,df,
                np.float64(1e-10),np.float64(1e-7),counts,error,status)
    try:
        result,counts,error=engine.run(state,bind)
        actual=result['y'].get()
        equilibrium=np.linalg.solve(stiffness,forcing.T).T
        propagator=expm(-np.linalg.solve(mass,stiffness)*.002)
        expected=equilibrium+(propagator@(initial-equilibrium).T).T
        mass_error=float(np.max(abs(actual-expected)))
        need(mass_error<2e-6 and counts[:,1].sum()>0,'nonidentity mass dynamics failed')
        need(np.array_equal(state['y'].get(),initial),'input state mutated during transaction')
    finally:engine.close()
    # One failing block after private commits must not publish the successful
    # block or contaminate a subsequent epoch. This fixture is test-only.
    fixture='''
#include <math_constants.h>
#include "independent_blocks.cuh"
struct FaultModel {
 double*state;int n,bad;long long used;
 __device__ double trial(long long){return n==bad&&used>=4?CUDART_INF:0.;}
 __device__ int commit(long long u,long long h){state[n]+=1.;used=u+h;return 0;}
};
extern "C" __global__ void fault(double*state,int bad,long long*counts,double*err,int*status){
 if(threadIdx.x)return;int n=blockIdx.x;FaultModel m{state,n,bad,0};
 status[n]=advance_independent_block(m,12,4,2,100,counts+n*3,err+n);
}
'''
    state={'y':cp.asarray([[1.],[2.]])};original=state['y'].get()
    engine=BlockExecutor(fixture,'fault',2,state)
    def fault_bind(bad):
        return lambda s,c,e,status:(s['y'],np.int32(bad),c,e,status)
    try:
        try:engine.run(state,fault_bind(1))
        except RuntimeError as exc:need('before publication' in str(exc),'wrong failure cause')
        else:raise ValueError('partial failing epoch published')
        need(np.array_equal(state['y'].get(),original),'failed transaction changed caller state')
        good,_,_=engine.run(state,fault_bind(-1))
        need(np.array_equal(good['y'].get(),original+3),'retry used contaminated private state')
        engine._gate.acquire()
        try:
            try:engine.run(state,fault_bind(-1))
            except RuntimeError as exc:need('already in use' in str(exc),'wrong overlap error')
            else:raise ValueError('overlapping epoch accepted')
        finally:engine._gate.release()
    finally:engine.close()
    return {'status':'PASS_BLOCK_GUARDS','dense_cases':cases,'nonidentity_mass_max_error':mass_error,
        'mass_accepted':counts[:,0].tolist(),'mass_rejected':counts[:,1].tolist(),
        'invalid_systems_rejected':True,'strict_residual_rejected':True,'rollback_all_blocks':True,
        'retry_after_failure':True,'python_optimized':not __debug__,
        'scope':'Generic algebra and declared mass model; organism validated separately'}


if __name__=='__main__':print(json.dumps(check(),allow_nan=False))
