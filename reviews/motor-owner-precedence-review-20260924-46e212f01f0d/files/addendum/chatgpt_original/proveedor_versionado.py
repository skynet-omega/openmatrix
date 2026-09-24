"""Proveedor CPU/C++ sombra: continuous/event/owner; fallback completo obligatorio.
No integrador, no speedup. Oracle y taps deben ser del MISMO candidato proyectado.
"""
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"
import argparse, ctypes as ct, hashlib, json
import subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

CPP=r"""
#include <cmath>
#include <cstdint>
extern "C" void split_coefficient(
 int nr,const int64_t*p,const int32_t*j,const uint8_t*kind,
 const double*w,const double*s,const double*cap,
 const uint8_t*vr,const uint8_t*vs,const double*tau,const double*gain,
 const double*theta,const double*drive,const double*photo,double scale,int linked,
 double*total,double*parts,double*target,double*rate){
 for(int i=0;i<nr;i++){
  double v[4][2][32]={};
  for(int l=0;l<32;l++)for(int64_t e=p[i]+l;e<p[i+1];e+=32){
   int col=j[e],k=kind[e]+1;
   if(vr[i]){
    double x=(w[e]*scale)*s[col];
    if(x>=0){v[0][0][l]+=x;v[k][0][l]+=x;}
    else{v[0][1][l]-=x;v[k][1][l]-=x;}
   }else if(linked||!vs[col]){
    double x=w[e]*(s[col]*cap[col]);
    v[0][0][l]+=x;v[k][0][l]+=x;
   }
  }
  for(int d=16;d;d/=2)for(int l=0;l<d;l++)
   for(int k=0;k<4;k++)for(int c=0;c<2;c++)v[k][c][l]+=v[k][c][l+d];
  for(int c=0;c<2;c++){
   total[2*i+c]=v[0][c][0];
   for(int k=0;k<3;k++)parts[6*i+2*k+c]=v[k+1][c][0];
  }
  if(vr[i]){
   double a=v[0][0][0];a+=photo[i];double den=1.+a+v[0][1][0];
   target[i]=(.25+a)/den;rate[i]=den/tau[i];
  }else{
   target[i]=fmax(0.,tanh(gain[i]*(v[0][0][0]+drive[i]-theta[i])));
   rate[i]=1./tau[i];
  }
 }
}
"""
def need(ok,msg):
    if not ok:raise ValueError(msg)
def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(a.dtype.str.encode());h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def own(a,dtype,shape,name):
    need(isinstance(a,np.ndarray) and a.dtype==dtype and a.shape==shape,"Layout/type "+name)
    need(np.isfinite(a).all(),"Nonfinite "+name)
    return np.array(a,copy=True,order="C")
def ptr(a):return a.ctypes.data_as(ct.c_void_p)
def build(out):
    src=out/"provider.cpp";src.write_text(CPP);so=out/"provider.so"
    cmd=["g++","-std=c++17","-O2","-fno-fast-math","-ffp-contract=off",
         "-fPIC","-shared",str(src),"-o",str(so)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
    (out/"BUILD.txt").write_text(" ".join(cmd)+"\n"+r.stdout+r.stderr)
    need(r.returncode==0,"C++ compilation failed")
    lib=ct.CDLL(str(so.resolve()))
    lib.split_coefficient.argtypes=[ct.c_int]+[ct.c_void_p]*13+[ct.c_double,ct.c_int]+[ct.c_void_p]*4
    lib.split_coefficient.restype=None
    return lib

class Provider:
    """Roles: 0 continua, 1 evento prescrito, 2 sustitucion de propietario.
    safe declara filas base demostradas por dependencias; defecto: ninguna.
    La base ocupa target/rate[:nr]; los tails siempre usan el oraculo completo.
    """
    def __init__(self,lib,p,j,kind,ns,row_ids,source_ids,safe=None):
        self.nr=len(p)-1;self.ns=ns
        self.p=own(p,np.dtype("int64"),(self.nr+1,),"ptr")
        E=len(j);self.j=own(j,np.dtype("int32"),(E,),"indices")
        self.kind=own(kind,np.dtype("uint8"),(E,),"edge role")
        self.rows=own(row_ids,np.dtype("int64"),(self.nr,),"row ids")
        self.sources=own(source_ids,np.dtype("int64"),(ns,),"source ids")
        self.safe=own(np.zeros(self.nr,bool) if safe is None else safe,np.dtype("bool"),(self.nr,),"certified base rows")
        need(self.nr>0 and ns>0 and E>0 and p[0]==0 and p[-1]==E and np.all(np.diff(p)>=0),"CSR offsets")
        need(np.all((j>=0)&(j<ns)) and np.all(kind<=2),"CSR domain")
        need(len(np.unique(row_ids))==self.nr and len(np.unique(source_ids))==ns and np.all(row_ids>=0) and np.all(source_ids>=0),"Stable identities")
        self.structural=digest(self.p,self.j,self.kind,self.rows,self.sources,self.safe)
        for a in (self.p,self.j,self.kind,self.rows,self.sources,self.safe):a.flags.writeable=False
        self.lib=lib;self.token=None;self.busy=False;self.calls=0
        self.edge_roles=np.bincount(kind,minlength=3).tolist()
    def bind(self,token):
        need(not self.busy and isinstance(token,tuple) and len(token)==5 and all(isinstance(x,str) and x for x in token),"Version token")
        self.token=token
    def check_structure(self,p,j,kind,rows,sources,safe):
        need(digest(p,j,kind,rows,sources,safe)==self.structural,"Structural change: rebuild provider")
    def base(self,f):
        E=len(self.j);n=self.nr;ns=self.ns
        specs={"weights":(E,),"release":(ns,),"caps":(ns,),
               "tau":(n,),"gain":(n,),"theta":(n,),"drive":(n,),"photo":(n,)}
        a={k:own(f[k],np.dtype("float64"),s,k) for k,s in specs.items()}
        vr=own(f["receiver_visual"],np.dtype("bool"),(n,),"receiver_visual").astype(np.uint8)
        vs=own(f["source_visual"],np.dtype("bool"),(ns,),"source_visual").astype(np.uint8)
        need(np.all(a["tau"]>0) and type(f["connected"]) is bool and np.isfinite(f["scale"]),"Parameters")
        total=np.empty((n,2));parts=np.empty((n,3,2));target=np.empty(n);rate=np.empty(n)
        args=[self.p,self.j,self.kind,a["weights"],a["release"],a["caps"],vr,vs,
              a["tau"],a["gain"],a["theta"],a["drive"],a["photo"]]
        self.lib.split_coefficient(n,*[ptr(x) for x in args],float(f["scale"]),int(f["connected"]),
                                  ptr(total),ptr(parts),ptr(target),ptr(rate))
        need(all(np.isfinite(x).all() for x in (total,parts,target,rate)),"Nonfinite native result")
        return dict(total=total,parts=parts,target=target,rate=rate,
                    weight_hash=digest(a["weights"]),
                    split_roundoff=float(np.max(abs(total-parts.sum(axis=1)))))
    def query(self,token,t,y,oracle):
        """Oracle: version() y __call__(token,t,x), respuesta topology/z/target/rate,
        base opcional, edge_visits, work_complete. Sin mutar x; NumPy FP64 tras
        sincronizar. edge_visits incluye base y TODOS los reemplazos especializados.
        """
        need(not self.busy and token==self.token and oracle.version()==token,"Epoch/event/phase stale")
        need(isinstance(y,np.ndarray) and y.dtype==np.float64 and y.ndim==1 and np.isfinite(y).all() and np.isfinite(t),"Candidate")
        self.busy=True;start=time.perf_counter()
        try:
            x=y.copy();before=x.tobytes();r=oracle(token,float(t),x)
            need(x.tobytes()==before and token==self.token==oracle.version(),"Oracle mutated candidate or version")
            need(r["topology"]==self.structural,"Effective topology/layout mismatch")
            z=own(r["z"],np.dtype("float64"),y.shape,"projected candidate")
            a=own(r["target"],np.dtype("float64"),y.shape,"full target")
            b=own(r["rate"],np.dtype("float64"),y.shape,"full rate")
            need(len(y)>=self.nr,"Base larger than state")
            v=r["edge_visits"];need(type(v) is int and v>=len(self.j),"Fallback cost must include full base traversal")
            complete=r["work_complete"];need(type(complete) is bool,"Work attribution required")
            report={"strict_oracle_calls":1,"oracle_edge_visits":v,"work_complete":complete,
                    "fallback_rows":len(y),"structural_setup_edges":len(self.j) if self.calls==0 else 0,
                    "partitioned_pass_edges":0,"all_outputs_returned":True}
            diag=None
            if r.get("base") is not None:
                diag=self.base(r["base"])
                eq=(diag["target"].view(np.uint64)==a[:self.nr].view(np.uint64)) & \
                   (diag["rate"].view(np.uint64)==b[:self.nr].view(np.uint64))
                good=self.safe&eq;bad=self.safe&~eq
                idx=np.flatnonzero(good);a[idx]=diag["target"][idx];b[idx]=diag["rate"][idx]
                report.update(partitioned_pass_edges=len(self.j),edge_roles=self.edge_roles,
                    safe_rows=int(self.safe.sum()),certified_rows_mismatch=int(bad.sum()),
                    fallback_rows=len(y)-int(good.sum()),split_roundoff=diag["split_roundoff"],
                    mismatch_rows=np.flatnonzero(bad).tolist(),
                    base_target_max_abs=float(np.max(abs(diag["target"]-r["target"][:self.nr]))),
                    base_rate_max_abs=float(np.max(abs(diag["rate"]-r["rate"][:self.nr]))),
                    effective_weights_hash=diag["weight_hash"])
            report["counted_edges"]=v+report["partitioned_pass_edges"]+report["structural_setup_edges"]
            report["wall_s"]=time.perf_counter()-start
            report["classification"]="IDENTITY_BY_STRICT_FALLBACK_NO_SPEED_CLAIM"
            need(token==self.token==oracle.version(),"Version changed during native calculation")
            self.calls+=1
            return {"z":z,"target":a,"rate":b,"currents":diag,"report":report}
        finally:self.busy=False

def fixture(lib):
    p=np.array([0,3,6,9],np.int64);j=np.array([0,1,2]*3,np.int32)
    roles=np.array([0,1,2]*3,np.uint8)
    pro=Provider(lib,p,j,roles,3,np.arange(3,dtype=np.int64),np.arange(3,dtype=np.int64),np.array([1,1,0],bool))
    tok=("operator-1","ports-1","owners-1","boundary-1","accepted")
    pro.bind(tok)
    frame=dict(weights=np.array([.5,-.25,1.]*3),release=np.array([.2,.3,.4]),
        caps=np.ones(3),tau=np.array([.02,.03,.04]),gain=np.array([.1,.2,.3]),
        theta=np.zeros(3),drive=np.zeros(3),photo=np.array([0.,.2,0.]),scale=.7,connected=True,
        receiver_visual=np.array([0,1,0],bool),source_visual=np.array([0,1,0],bool))
    class O:
        token=tok;mode=0;calls=0
        def version(self):return self.token
        def __call__(self,key,t,x):
            self.calls+=1
            u=max(0.,t-.9);q=0. if t<.9 else .7*np.exp(-u/.1)
            s=0. if t<.9 else .7*(u/.1)*np.exp(-u/.1)
            z=x.copy();z[-2:]=q,s
            f={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in frame.items()}
            f["release"]=np.array([x[0],s,x[1]])
            f["weights"]*=1.+.1*x[1]+.01*self.mode  # effective weights vary with candidate
            A=[];R=[]
            for row in range(3):
                net=neg=0.
                for e in range(p[row],p[row+1]):
                    c=j[e]
                    if f["receiver_visual"][row]:
                        v=(f["weights"][e]*f["scale"])*f["release"][c]
                        if v>=0:net+=v
                        else:neg-=v
                    elif f["connected"] or not f["source_visual"][c]:
                        net+=f["weights"][e]*(f["release"][c]*f["caps"][c])
                if f["receiver_visual"][row]:
                    net+=f["photo"][row];den=1.+net+neg;A.append((.25+net)/den);R.append(den/f["tau"][row])
                else:A.append(max(0.,np.tanh(f["gain"][row]*(net+f["drive"][row]-f["theta"][row]))));R.append(1./f["tau"][row])
            A=np.r_[A,q,s];R=np.r_[R,0.,0.]
            A[2]=-2.-x[2];R[2]=7.+x[2]  # unknown owner: negative target, not [0,1]
            if self.mode==2:A[0]+=.001  # undeclared specialized replacement
            self.last=(z.copy(),A.copy(),R.copy())
            return dict(topology=pro.structural,z=z,target=A,rate=R,base=f,edge_visits=12,work_complete=True)
    oracle=O();x=np.array([.2,.4,.6,0.,0.]);before=x.copy()
    def query(key,t,state):
        result=pro.query(key,t,state,oracle)
        need(all(np.array_equal(result[k].view(np.uint64),v.view(np.uint64))
                 for k,v in zip(("z","target","rate"),oracle.last)),"Full identity failed")
        return result
    r1=query(tok,.8,x);r2=query(tok,1.,x);r3=query(tok,.8,x)
    need(np.array_equal(r1["target"],r3["target"]) and r2["z"][-1]>0 and np.array_equal(x,before),"Replay/event/input")
    need(r1["target"][2]==-2.6 and r1["rate"][2]==7.6 and np.all(r2["rate"][-2:]==0),"Owners/prescribed")
    saved=r1["target"].copy();oracle.mode=1;r4=query(tok,.8,x)
    need(r4["report"]["effective_weights_hash"]!=r1["report"]["effective_weights_hash"] and np.array_equal(saved,r1["target"]),"Weights or returned alias")
    oracle.mode=2;r5=query(tok,.8,x)
    need(r5["report"]["certified_rows_mismatch"]>=1,"Unknown replacement hidden")
    nt=("operator-1","ports-2","owners-1","boundary-1","accepted");oracle.token=nt
    stale=False
    try:pro.query(tok,.8,x,oracle)
    except ValueError:stale=True
    need(stale,"New event version not blocked");pro.bind(nt);query(nt,.8,x)
    jj=j.copy();jj[0]=1;structural=False
    try:pro.check_structure(p,jj,roles,pro.rows,pro.sources,pro.safe)
    except ValueError:structural=True
    need(structural,"Structural mutation accepted")
    need(np.array_equal(r1["target"],saved),"Output lost ownership")
    return dict(identity_with_owner_fallback=True,negative_owner_target=True,visual_EI=True,
                retrospective_queries=True,late_SET_visible=True,weight_refresh=True,
                event_version_rejected=stale,structure_rejected=structural,
                unmodelled_output_falls_back=True,calls=oracle.calls,
                reports=[r["report"] for r in (r1,r2,r3,r4,r5)],
                CUDA_executed=False,organism_executed=False)
if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--selftest",action="store_true",required=True)
    ap.add_argument("--out",type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    save(args.out/"PLAN.json",dict(mode="shadow_strict_fallback",CPU_budget_s=120,no_tolerance_changes=True))
    try:
        result=dict(status="COMPLETE_CPU_FIXTURE",data=fixture(build(args.out)),
                    CUDA_executed=False,organism_executed=False)
    except Exception:result=dict(status="FAILED_RETAINED",error=traceback.format_exc())
    result.update(wall_s=time.perf_counter()-start,code_sha256=file_sha(Path(__file__)))
    save(args.out/"RESULT.json",result);print(json.dumps(result,indent=2,allow_nan=False))
    raise SystemExit(0 if result["status"]=="COMPLETE_CPU_FIXTURE" else 2)

