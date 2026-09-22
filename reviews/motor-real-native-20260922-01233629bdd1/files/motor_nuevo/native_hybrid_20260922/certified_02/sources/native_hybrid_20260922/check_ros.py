from pathlib import Path
import sys,json
import numpy as np,cupy as cp
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent;OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work')
sys.path[:0]=[str(OLD/'motor13_20260922'),str(HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922')]
from verify_transport import read_state
from ros_native import kernel
from ros_tableau import AT,GI,BT,BET,BIT,GAMMA
s=read_state(HERE/'baseline_01/brain_final');c=s['kc_spatial_manifest']['cell'];st=s['kc_spatial_state'];n=4
G=c['channel_G_nS'].reshape(51,17,17);b=c['channel_b_nS'].reshape(51,17);ena=np.tile(np.array([60.,60.,-80.])-c['rest_mV'],17)
rng=np.random.default_rng(20260923);v=st['delta'][[0,500,1000,1556]].copy();g=st['gates'][[0,500,1000,1556]].copy();ge=rng.uniform(0,.1,(n,4));gi=rng.uniform(0,.1,(n,4));current=np.zeros((n,17))
base=[cp.asarray(x) for x in (v,g,ge,gi,current,c['C_nF'],c['G_nS'],G,b,c['shunt_G'],c['shunt_b'],ena)]
kernel=kernel();Cinv=np.linalg.inv(c['C_nF'])
def rates(v):
 s=np.array([1/(1+np.exp(-.1121*(v+29.13))),1/(1+np.exp(.2*(v+47))),1/(1+np.exp(-.2717*(v+48.77))),1/(1+np.exp(-.0502*(v+12.85)))]).T
 t=np.array([(.1270+3.434/(1+np.exp((v+45.35)/5.98)))/1000,(.36+np.exp((v+20.65)/-10.47))/1000,np.full_like(v,.001),(2.03+1.96/(1+np.exp((v-30.83)/3.12)))/1000]).T
 return s,t

def rhs(index,y):
 vv=y[:17];gg=y[17:].reshape(17,4);m,h,p,nn=gg.T;f=np.array([m**3*h,p,nn**4]).T.reshape(51)
 K=c['G_nS']+np.einsum('k,kij->ij',ge[index]+gi[index],c['shunt_G'])+np.einsum('k,kij->ij',f,G)
 B=current[index]+(-c['rest_mV']*ge[index]+(-68-c['rest_mV'])*gi[index])@c['shunt_b']+(f*ena)@b
 ss,tt=rates(vv+c['rest_mV']);return np.r_[Cinv@(B-K@vv),((ss-gg)/tt).ravel()]

def cpu_ros(index,y,h):
 vv=y[:17];gg=y[17:].reshape(17,4);m,hh,p,nn=gg.T;f=np.array([m**3*hh,p,nn**4]).T.reshape(51);_,tau=rates(vv+c['rest_mV'])
 K=c['G_nS']+np.einsum('k,kij->ij',ge[index]+gi[index],c['shunt_G'])+np.einsum('k,kij->ij',f,G)
 M=np.zeros((85,85));M[:17,:17]=c['C_nF'];M[17:,17:]=np.eye(68)
 W=np.zeros_like(M);W[:17,:17]=-K;W[17:,17:]=np.diag(-1/tau.ravel());mat=M/(GAMMA*h)-W
 from scipy.linalg import lu_factor,lu_solve
 lu=lu_factor(mat);stages=[]
 for k in range(4):
  yy=y.copy();corr=np.zeros_like(y)
  for j in range(k):yy+=AT[k,j]*stages[j];corr+=GI[k,j]*stages[j]/h
  stages.append(lu_solve(lu,M@rhs(index,yy)-M@corr))
 stages=np.stack(stages);high=y+BT@stages;mid=y+np.array([.5,.25,.125])@BIT.T@stages
 return high,mid
checks=[]
for ns in (25000,12500,6250,3125):
 clock=cp.asarray([0,ns],dtype=cp.int64);kv=cp.empty((n,4,17));kg=cp.empty((n,4,17,4));vh=cp.empty_like(base[0]);gh=cp.empty_like(base[1]);vo=cp.empty_like(vh);go=cp.empty_like(gh);err=cp.zeros(n)
 kernel((n,),(32,),(np.int32(n),clock,np.float64(c['rest_mV']),*base,kv,kg,vh,gh,vo,go,err));cp.cuda.get_current_stream().synchronize()
 actual=np.concatenate((vo.get(),go.get().reshape(n,68)),axis=1);middle=np.concatenate((vh.get(),gh.get().reshape(n,68)),axis=1)
 if not np.isfinite(actual).all() or not np.isfinite(err.get()).all():raise RuntimeError('Bad Rosenbrock state')
 ev=eg=emv=emg=ecpu=0.
 for i in range(n):
  y=np.r_[v[i],g[i].ravel()];h=ns*1e-9
  sol=solve_ivp(lambda t,y:rhs(i,y),(0,h),y,method='Radau',rtol=1e-10,atol=1e-12,t_eval=[(ns//2)*1e-9,h])
  if not sol.success:raise RuntimeError('Independent reference failed')
  ch,cm=cpu_ros(i,y,h);ecpu=max(ecpu,float(np.max(abs(ch-actual[i]))))
  if not np.allclose(ch,actual[i],rtol=1e-11,atol=1e-10):raise RuntimeError('CPU/GPU tableau disagree')
  # Dense interpolation uses the actual integer half, including odd ns.
  if ns%2==0 and not np.allclose(cm,middle[i],rtol=1e-11,atol=1e-10):raise RuntimeError('Dense tableau mismatch')
  ev=max(ev,float(np.max(abs(sol.y[:17,1]-actual[i,:17]))));eg=max(eg,float(np.max(abs(sol.y[17:,1]-actual[i,17:]))))
  emv=max(emv,float(np.max(abs(sol.y[:17,0]-middle[i,:17]))));emg=max(emg,float(np.max(abs(sol.y[17:,0]-middle[i,17:]))))
 checks.append(dict(ns=ns,voltage=ev,gates=eg,mid_voltage=emv,mid_gates=emg,cpu_gpu_max=ecpu,embedded_max=float(err.max().get())))
 print(json.dumps(checks[-1]),flush=True)
report=dict(checks=checks,kernel=kernel.attributes)
(HERE/'ROS_OPERATOR_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
if checks[-1]['voltage']>2e-5 or checks[-1]['gates']>2e-7:raise RuntimeError('Small-step error exceeds fixed budget')
