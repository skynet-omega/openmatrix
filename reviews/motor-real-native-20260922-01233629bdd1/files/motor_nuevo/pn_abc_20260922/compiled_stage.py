"""CPU nopython electrical stages: full coordinates, original SDIRK/kinetics.

Outer validation, presampling, observation and calcium chemistry remain original.
No fast-math, clipping, neural-output caching or mass approximation.
"""
import numpy as np
from numba import njit
from pn_graph_elimination_backend import eliminate,fill_core,back
from pn_mass_backend import _difference_action

@njit(cache=True)
def sigmoid(z):
    e=np.exp(-abs(z));return 1/(1+e) if z>=0 else e/(1+e)

@njit(cache=True)
def channels(v,base,q,gbar,reversal):
    n=len(v);x=np.empty((n,4));ionic=np.empty((n,3));jac=np.empty(n);frozen=np.empty(n);error=0.
    for i in range(n):
        w=v[i];steady=np.array([sigmoid(.1121*(w+29.13)),sigmoid(-.2*(w+47)),sigmoid(.2717*(w+48.77)),sigmoid(.0502*(w+12.85))])
        tau=np.array([.127+3.434*sigmoid(-(w+45.35)/5.98),.36+np.exp((w+20.65)/-10.47),1.,2.03+1.96*sigmoid(-(w-30.83)/3.12)])/1000
        sm=(tau[0]*1000-.127)/3.434;sn=(tau[3]*1000-2.03)/1.96
        dtau=np.array([-3.434/5.98*sm*(1-sm),-(tau[1]*1000-.36)/10.47,0.,-1.96/3.12*sn*(1-sn)])/1000
        slopes=np.array([.1121,-.2,.2717,.0502]);dx=np.empty(4)
        for k in range(4):
            x[i,k]=(tau[k]*base[i,k]+q*steady[k])/(tau[k]+q)
            if not np.isfinite(tau[k]) or tau[k]<=0 or not np.isfinite(x[i,k]) or x[i,k]<0 or x[i,k]>1:raise FloatingPointError('Invalid compiled gate')
            dx[k]=(q*steady[k]*(1-steady[k])*slopes[k]+dtau[k]*(base[i,k]-x[i,k]))/(tau[k]+q)
            error=max(error,abs(x[i,k]-base[i,k]-q*(steady[k]-x[i,k])/tau[k]))
        m,h,p,z=x[i];dm,dh,dp,dz=dx
        g=gbar[i]*np.array([m**3*h,p,z**4]);dg=gbar[i]*np.array([3*m*m*h*dm+m**3*dh,dp,4*z**3*dz])
        jac[i]=0.;frozen[i]=0.
        for k in range(3):
            ionic[i,k]=g[k]*(w-reversal[k]);jac[i]+=g[k]+dg[k]*(w-reversal[k]);frozen[i]+=g[k]
        if not np.isfinite(jac[i]) or not np.isfinite(ionic[i]).all():raise FloatingPointError('Invalid compiled current')
    return x,ionic,jac,frozen,error

@njit(cache=True)
def calcium(v,base,q,half,slope,tau,power,gbar,reversal,enabled):
    x=np.empty_like(base);cur=np.empty(len(v));jac=cur.copy();frozen=cur.copy();error=0.
    for i in range(len(v)):
        dx=np.empty(len(power));factors=np.empty(len(power));product=1.
        for k in range(len(power)):
            steady=sigmoid((v[i]-half[i,k])/slope[i,k]);x[i,k]=(tau[i,k]*base[i,k]+q*steady)/(tau[i,k]+q)
            if not np.isfinite(x[i,k]) or x[i,k]<0 or x[i,k]>1:raise FloatingPointError('Invalid compiled Ca gate')
            dx[k]=q*steady*(1-steady)/(slope[i,k]*(tau[i,k]+q));factors[k]=x[i,k]**power[k];product*=factors[k]
            error=max(error,abs(x[i,k]-base[i,k]-q*(steady-x[i,k])/tau[i,k]))
        derivative=0.
        for k in range(len(power)):
            other=1.
            for j in range(len(power)):
                if j!=k:other*=factors[j]
            derivative+=power[k]*x[i,k]**(power[k]-1)*dx[k]*other
        g=gbar[i]*product if enabled else 0.;dg=gbar[i]*derivative if enabled else 0.
        frozen[i]=g;cur[i]=g*(v[i]-reversal);jac[i]=g+dg*(v[i]-reversal)
        if not np.isfinite(cur[i]+jac[i]):raise FloatingPointError('Invalid compiled Ca current')
    return x,cur,jac,frozen,error

@njit(cache=True)
def matvec(indptr,indices,data,v,minimum):
    out=np.empty_like(v)
    for i in range(len(v)):
        total=0.;correction=0.
        for k in range(indptr[i],indptr[i+1]):
            term=data[k]*v[indices[k]];value=total+term
            if abs(total)>=abs(term):correction+=(total-value)+term
            else:correction+=(term-value)+total
            total=value
        out[i]=total+correction if indptr[i+1]-indptr[i]>=minimum else total
    return out

@njit(cache=True)
def evaluate(v,vbase,xbase,q,shift,current,G,M,gsum,active,gbar,reversal,leak,sn,sg,se,cn,cb,ca_params,ca_reversal,enabled):
    x,ionic,jac,frozen,ge=channels(v[active]+leak,xbase,q,gbar,reversal)
    cx,ci,cj,cf,ce=calcium(v[cn]+leak,cb,q,*ca_params,ca_reversal,enabled)
    r=matvec(*M,(v-vbase)*shift,16)+_difference_action(*G,gsum,v)-current
    for i,node in enumerate(active):r[node]+=ionic[i].sum()
    for i,node in enumerate(sn):r[node]+=sg[i]*(v[node]+leak-se[i])
    for i,node in enumerate(cn):r[node]+=ci[i]
    return r,np.linalg.norm(r),x,ionic,jac,frozen,max(ge,ce),cx,ci,cj,cf

@njit(cache=True)
def graph_solve(rhs,shift,diagonal,plan):
    steps,pairs,dg,dm,base,mass,core,edges=plan
    d=dg+shift*dm+diagonal;values=base+shift*mass;b=rhs.copy()
    if not eliminate(steps,pairs,d,values,b):raise FloatingPointError('Compiled nonpositive pivot')
    matrix=np.diag(d[core]);fill_core(edges,values,matrix);x=np.zeros_like(b)
    if len(core):x[core]=np.linalg.solve(matrix,b[core])
    back(steps,pairs,d,values,b,x);return x

@njit(cache=True)
def run_stage(vbase,xbase,guess,reference,q,shift,current,G,M,gsum,C,active,gbar,reversal,leak,sn,sg,se,cn,cb,ca_params,ca_reversal,enabled,plan,rtol,atol,max_newton,gate_atol):
    v=guess.copy();data=evaluate(v,vbase,xbase,q,shift,current,G,M,gsum,active,gbar,reversal,leak,sn,sg,se,cn,cb,ca_params,ca_reversal,enabled)
    rr=evaluate(reference,vbase,xbase,q,shift,current,G,M,gsum,active,gbar,reversal,leak,sn,sg,se,cn,cb,ca_params,ca_reversal,enabled)
    threshold=max(atol,rtol*rr[1]);history=np.full((max_newton+1,5),np.nan);total=0
    for iteration in range(max_newton+1):
        r,norm,x,ionic,jac,frozen,ge,cx,ci,cj,cf=data;history[iteration,0]=norm;history[iteration,1]=ge
        if np.isfinite(norm) and norm<=threshold and ge<=gate_atol:return v,x,ionic,cx,ci,cj,cf,ge,history[:iteration+1],threshold,total,0
        if not np.isfinite(norm) or iteration==max_newton:break
        diagonal=np.zeros(len(v))
        for i,node in enumerate(active):diagonal[node]+=jac[i]
        for i,node in enumerate(sn):diagonal[node]+=sg[i]
        for i,node in enumerate(cn):diagonal[node]+=cj[i]
        signed=np.all(C*shift+diagonal>=0);history[iteration,4]=1. if signed else 0.
        if not signed:
            diagonal.fill(0.)
            for i,node in enumerate(active):diagonal[node]+=frozen[i]
            for i,node in enumerate(sn):diagonal[node]+=sg[i]
            for i,node in enumerate(cn):diagonal[node]+=cf[i]
        correction=graph_solve(-r,shift,diagonal,plan);total+=1
        # Original full linear equation, up to3 refinements, never reduced acceptance.
        limit=.25*threshold
        for refinement in range(3):
            linear=matvec(*G,correction,1000000000)+shift*matvec(*M,correction,1000000000)+diagonal*correction+r
            linear_norm=np.linalg.norm(linear);history[iteration,2]=linear_norm
            if np.isfinite(linear_norm) and linear_norm<=limit:break
            if refinement<2:correction+=graph_solve(-linear,shift,diagonal,plan);total+=1
        if not np.isfinite(linear_norm) or linear_norm>limit: return v,x,ionic,cx,ci,cj,cf,ge,history[:iteration+1],threshold,total,2
        success=False
        for backtrack in range(9):
            trial=v+2.**(-backtrack)*correction
            candidate=evaluate(trial,vbase,xbase,q,shift,current,G,M,gsum,active,gbar,reversal,leak,sn,sg,se,cn,cb,ca_params,ca_reversal,enabled)
            if np.isfinite(candidate[1]) and (candidate[1]<norm or candidate[1]<=threshold):
                v=trial;data=candidate;history[iteration,3]=2.**(-backtrack);success=True;break
        if not success:return v,x,ionic,cx,ci,cj,cf,ge,history[:iteration+1],threshold,total,3
    return v,x,ionic,cx,ci,cj,cf,ge,history[:iteration+1],threshold,total,1

def compiled_stage(self,vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess,*,shift,q,current,ca,rtol,atol,max_newton,gate_atol,progress):
    if ca is None:raise ValueError('This bounded prototype requires captured Ca port')
    sn,sg,se=synapse[:3] if synapse is not None else (np.empty(0,dtype=np.int64),np.empty(0),np.empty(0))
    se=np.broadcast_to(se,sn.shape).copy();p=self.backend._graph_plan;b=self.backend
    plan=tuple(getattr(p,k) for k in ('steps','pairs','dg','dm','base','mass','core','core_edges'))
    try:
        values=run_stage(vbase,xbase,guess,guess if threshold_guess is None else threshold_guess,q,shift,current,
          (b.G.indptr,b.G.indices,b.G.data),(b.M.indptr,b.M.indices,b.M.data),b._G_sum,b.C,self.active_nodes,self.gbar_nS,self.reversal_mV,self.leak_reversal_mV,
          sn,sg,se,ca.nodes,ca_base,(ca.half,ca.slope,ca.tau,ca.power,ca.gbar),ca.reversal,ca.enabled,plan,rtol,atol,max_newton,gate_atol)
    except (FloatingPointError,np.linalg.LinAlgError) as exc:return None,dict(accepted=False,reason='compiled_numerical_failure',detail=str(exc),history=[],total_iterations=0)
    v,x,ionic,cx,ci,cj,cf,ge,rows,threshold,total,code=values
    history=[]
    for i,row in enumerate(rows):
        record={'iteration':i,'residual_l2_pA':float(row[0]),'gate_residual':float(row[1])}
        for k,value in zip(('linear_residual_l2_pA','step_scale','exact_local_jacobian'),row[2:]):
            if np.isfinite(value):record[k]=float(value)
        history.append(record)
        if progress:progress(record)
    report=dict(accepted=code==0,history=history,threshold_pA=threshold,total_iterations=total)
    if code:report['reason']={1:'stage_residual_failed',2:'linear_residual_failed',3:'no_nonlinear_decrease'}[code];return None,report
    return (v,x,ionic,dict(gates=cx,current=ci,jacobian=cj,frozen_conductance=cf,gate_error=ge)),report
