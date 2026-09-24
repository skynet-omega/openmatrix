"""Candidata aislada: fusionar sólo las actualizaciones exponenciales del padre.
No cambia las seis evaluaciones, el estimador, eventos, dominio o controlador.
--selftest compila el kernel como bucle CPU; no ejecuta CUDA ni el organismo.
"""
import argparse, ast, hashlib, json, subprocess, time, traceback
from pathlib import Path
SHA = '898dbf27525887da8f127ffe8114df734a8dcc8a99072f667e263ed82838a497'
KERNEL = r'''
extern "C" __global__ void relax_fp64(const double* z,const double* a,
 const double* b,const double* clock,double scale,int n,double* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){
  double t=__dmul_rn(scale,clock[1]);
  double arg=__dmul_rn(t,b[i]);
  double e=-expm1(arg);
  double d=__dsub_rn(a[i],z[i]);
  out[i]=__dadd_rn(z[i],__dmul_rn(e,d));
 }
}
'''
METHOD = ''' def relax_update(self,z,a,b,scale):
  for value in (z,a,b):
   if value.shape!=(self.n,) or value.dtype!=cp.float64 or not value.flags.c_contiguous:
    raise ValueError('Fusion requires contiguous one-dimensional FP64 state/coefficients')
  out=cp.empty_like(z)
  self.relax_kernel(self.grid,(256,),(z,a,b,self.clock,np.float64(scale),np.int32(self.n),out))
  return out
'''
PLAN = {'operation':'Fuse elementwise exponential updates only',
 'coefficients_per_attempt':6,'control':'Original C++ and norm; no nominal recovery or B2',
 'precision':'FP64; explicit round-to-nearest add/sub/mul, no FMA in the new update',
 'scope':'Candidate for a new isolated graph; not promoted',
 'CPU_test':'Scalar C++ operation sequence, not CuPy/libdevice parity or performance'}
def need(ok,msg):
 if not ok: raise ValueError(msg)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def once(s,a,b):
 need(s.count(a)==1,'Source anchor absent or ambiguous: '+a[:75]);return s.replace(a,b,1)
def generate(parent,out):
 need(sha(parent)==SHA,'Parent SHA256 mismatch')
 source=parent.read_text(encoding='utf-8');tree=ast.parse(source)
 assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KERNEL' for t in n.targets))
 original=ast.literal_eval(assign.value)
 source=once(source,'KERNEL=r\'\'\'','KERNEL=r\'\'\'\n'+KERNEL)
 source=once(source,"self.summary=mod.get_function('status');","self.summary=mod.get_function('status');self.relax_kernel=mod.get_function('relax_fp64');")
 source=once(source,' def midpoint(self,y,frac,start):',METHOD+' def midpoint(self,y,frac,start):')
 source=once(source,'middle=z+(-cp.expm1(-.5*frac*self.clock[1]*b))*(a-z)',
                     'middle=self.relax_update(z,a,b,-.5*frac)')
 source=once(source,'out=z+(-cp.expm1(-frac*self.clock[1]*b))*(a-z)',
                     'out=self.relax_update(z,a,b,-frac)')
 # Never accidentally load a different default native library from the output folder.
 source=once(source,'  start=time.perf_counter();self.stream=',
                    "  if native_library is None:raise ValueError('Explicit original native_library required')\n  start=time.perf_counter();self.stream=")
 compile(source,'graph_core_fused.py','exec')
 (out/'graph_core_fused.py').write_text(source,encoding='utf-8')
 (out/'graph_core_parent.py').write_bytes(parent.read_bytes())
 (out/'relax_fp64.cu').write_text(KERNEL,encoding='utf-8')
 # The original norm/check/status CUDA source remains an unmodified suffix.
 parsed=ast.parse(source)
 modified=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='KERNEL' for t in n.targets))
 need(modified.endswith(original),'Original CUDA checks changed')
 return {'parent_sha256':SHA,'candidate_sha256':sha(out/'graph_core_fused.py'),
         'kernel_sha256':sha(out/'relax_fp64.cu'),'original_checks_unchanged':True}
HEAD = r'''
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <cfenv>
#define __global__
struct Dim{int x;}; static Dim blockIdx{0},blockDim{1},threadIdx{0};
inline double __dmul_rn(double a,double b){volatile double r=a*b;return r;}
inline double __dsub_rn(double a,double b){volatile double r=a-b;return r;}
inline double __dadd_rn(double a,double b){volatile double r=a+b;return r;}
'''
BODY = r'''
using V=std::vector<double>;
bool bits(double a,double b){return std::memcmp(&a,&b,8)==0;}
V parent(const V&z,const V&a,const V&b,double h,double scale){
 int n=z.size();V u(n),v(n),w(n),d(n),p(n),o(n);
 volatile double t=scale*h;
 for(int i=0;i<n;i++)u[i]=t*b[i];
 for(int i=0;i<n;i++)v[i]=expm1(u[i]);
 for(int i=0;i<n;i++)w[i]=-v[i];
 for(int i=0;i<n;i++)d[i]=a[i]-z[i];
 for(int i=0;i<n;i++)p[i]=w[i]*d[i];
 for(int i=0;i<n;i++)o[i]=z[i]+p[i];
 return o;
}
V candidate(const V&z,const V&a,const V&b,double h,double scale){
 V o(z.size());double c[2]={0.,h};
 for(int i=0;i<(int)z.size();i++){threadIdx.x=i;
  relax_fp64(z.data(),a.data(),b.data(),c,scale,z.size(),o.data());}
 return o;
}
int main(){try{
 if(std::fesetround(FE_TONEAREST))throw std::runtime_error("rounding mode");
 uint64_t checked=0;int cases=0;const double scales[]={-.5,-1.,-.25,-.5,-.25,-.5};
 for(int n:{1,17,256,257,131071}){
  V z(n),a(n),b(n);
  for(int i=0;i<n;i++){z[i]=((i%83)-41)*.03125;a[i]=((i%61)-30)*.0625;b[i]=(i%19)*127.5;}
  for(double h:{0.,1e-7,125e-6,.01})for(double scale:scales){
   V z0=z,a0=a,b0=b;auto x=parent(z,a,b,h,scale),y=candidate(z,a,b,h,scale);
   for(int i=0;i<n;i++){if(!bits(x[i],y[i]))throw std::runtime_error("finite bit mismatch");checked++;}
   if(z!=z0||a!=a0||b!=b0)throw std::runtime_error("input mutation");cases++;
  }
 }
 V z={0.,-0.,.2,.2,.2,1.},a={-0.,0.,-2.,3.,.8,.7},b={0.,0.,1e9,1e9,-1e9,INFINITY};
 auto x=parent(z,a,b,.001,-1),y=candidate(z,a,b,.001,-1);
 for(int i=0;i<(int)z.size();i++)if(!(bits(x[i],y[i])||(std::isnan(x[i])&&std::isnan(y[i]))))
  throw std::runtime_error("IEEE edge mismatch");
 if(y[2]!=-2.||y[3]!=3.)throw std::runtime_error("unrequested clipping");
 V u={.2},v={.8},r={100.};auto good=candidate(u,v,r,.001,-1),bad=candidate(u,v,r,.001,1);
 if(bits(good[0],bad[0]))throw std::runtime_error("negative control not detected");
 std::cout<<"{\"finite_cases\":"<<cases<<",\"finite_elements_bitwise_equal\":"<<checked
          <<",\"input_immutable\":true,\"heterogeneous_domain_no_clipping\":true,"
          <<"\"sign_corruption_detected\":true,\"CUDA_executed\":false,\"organism_executed\":false}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 2;}}
'''
def test(out):
 cpp=out/'fixture.cpp';cpp.write_text(HEAD+KERNEL+BODY,encoding='utf-8');exe=out/'fixture'
 cmd=['g++','-std=c++17','-O3','-fno-fast-math','-ffp-contract=off','-frounding-math',str(cpp),'-o',str(exe)]
 build=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
 (out/'BUILD.txt').write_text(' '.join(cmd)+'\n'+build.stdout+build.stderr,encoding='utf-8')
 need(build.returncode==0,'CPU compilation failed')
 run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=20)
 (out/'CPU_STDOUT.json').write_text(run.stdout,encoding='utf-8')
 (out/'CPU_STDERR.txt').write_text(run.stderr,encoding='utf-8')
 need(run.returncode==0,'CPU fixture failed');return json.loads(run.stdout)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('parent',type=Path);p.add_argument('out',type=Path)
 p.add_argument('--selftest',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 save(a.out/'PLAN.json',PLAN);t=time.perf_counter();result={}
 try:
  result['generation']=generate(a.parent,a.out)
  if a.selftest:result['CPU']=test(a.out)
  result['status']='COMPLETE'
 except Exception:result.update(status='FAILED_RETAINED',error=traceback.format_exc())
 result.update(wall_s=time.perf_counter()-t,generator_sha256=sha(Path(__file__)))
 save(a.out/'RESULTADO.json',result);print(json.dumps(result,indent=2,ensure_ascii=False))
 return 0 if result['status']=='COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
