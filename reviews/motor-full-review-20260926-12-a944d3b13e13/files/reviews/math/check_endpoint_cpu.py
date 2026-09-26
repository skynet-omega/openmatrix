"""Compile and exercise the real endpoint/control/port expressions on CPU only.

CUDA qualifiers and launch indices are replaced for a one-thread host fixture;
the copied function bodies are extracted from sources, not reimplemented.
No CuPy import, CUDA compilation or neural simulation occurs.
"""
from pathlib import Path
import ast
import hashlib
import json
import math
import subprocess

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[1]
ROOT = HERE.parents[3]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def braced(text, marker):
    start = text.index(marker)
    left = text.index("{", start)
    depth = 0
    for i in range(left, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i+1]
    raise ValueError("Unclosed source block")


def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Missing literal " + name)


def main():
    report_path = HERE / "ENDPOINT_CPU_CHECK.json"
    need(not report_path.exists(), "Use a new result location; this CPU check is already recorded")
    controller_path = REVIEW / "engine/resident_controller.cu"
    runtime_path = REVIEW / "engine/graph_runtime.py"
    port_path = ROOT / "campanas/etapa3_pn629_intervention_20260923_15/event_ports.py"
    controller = controller_path.read_text()
    kernel = literal(runtime_path, "KERNEL")
    port = literal(port_path, "CODE")
    parts = [braced(controller, "enum Failure")+";",
             braced(controller, "struct Control")+";",
             braced(controller, "__global__ void prepare"),
             braced(controller, "__global__ void decide"),
             braced(kernel, 'extern "C" __global__ void endpoint_clocks'),
             braced(port, "__device__ double conv"),
             braced(port, 'extern "C" __global__ void port')]
    prefix = r'''
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#define __global__
#define __device__
struct Index { int x=0; } threadIdx, blockIdx;
struct Dimension { int x=1; } blockDim;
using std::min; using std::max; using std::isfinite;
'''
    main_code = r'''
void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
double projected_q(double t,double stop){
 double y[2]={0.,0.},q[1]={0.},s[1]={0.},tq[1]={.02},ts[1]={.005};
 double et[1]={stop},ej[1]={1.},post[1]={1.},clock[2]={t,0.};
 long long qr[1]={0},sr[1]={1};bool sets[1]={true};int counts[1]={1};
 port(y,qr,sr,q,s,tq,ts,et,ej,sets,post,counts,1,1,clock,0.);
 return y[0];
}
Control setup(double* clock,double* status,const double* events,long count,double start,double end){
 Control c={};c.clock=clock;c.status=status;c.events=events;c.event_count=count;
 c.used_s=start;c.end_s=end;c.next_ns=100000;c.min_ns=100;c.max_ns=1000000;
 c.min_accepted_ns=1000000;c.max_attempts=10000;return c;
}
void clipped(double start,double stop,bool demand_legacy_failure){
 double clock[3]={},status[3]={},events[2]={start,stop},left[2]={},right[2]={};
 auto c=setup(clock,status,events,2,start,125000*1e-9);prepare(&c);
 need(c.failure==0 && clock[2]==stop,"prepare did not preserve the canonical boundary");
 endpoint_clocks(clock,left,right);
 need(left[0]<stop && right[0]==stop,"endpoint side differs");
 need(projected_q(left[0],stop)==0. && projected_q(right[0],stop)==1.,"ADD/SET port side differs");
 if(demand_legacy_failure)need(start+(stop-start)!=stop,"missing historical cancellation example");
 status[0]=fabs(clock[1]*(-1./8.)*projected_q(left[0],stop))/1e-7;
 decide(&c);
 need(c.failure==0 && c.accept==1 && c.used_s==stop && c.next_ns>=100000,"interior recovery/commit changed");
}
int main(){try{
 need(sizeof(long)==8 && std::numeric_limits<double>::is_iec559,"CPU numerical ABI mismatch");
 clipped(1.1461103097022343e-05,2.8348089914503676e-05,true);
 clipped(5.581565907498342e-06,5.825387811154045e-05,true);
 clipped(1.1461103097022343e-05/128.,2.8348089914503676e-05/128.,true);
 clipped(1e-5,std::nextafter(1e-5,INFINITY),false);
 double clock[3]={},status[3]={},left[2]={},right[2]={};
 double e[1]={125000*1e-9};
 auto c=setup(clock,status,e,1,100000*1e-9,e[0]);prepare(&c);endpoint_clocks(clock,left,right);decide(&c);
 need(c.used_s==e[0] && c.next_ns<100000,"terminal event unexpectedly recovered proposal");
 c=setup(clock,status,nullptr,0,0.,125000*1e-9);c.next_ns=10000;prepare(&c);
 need(clock[2]==clock[0]+clock[1],"free endpoint arithmetic changed");
 decide(&c);need(c.used_s==clock[2] && c.next_ns==20000,"ordinary controller changed");
 double two[2]={1.1461103097022343e-05,2.8348089914503676e-05};
 status[0]=2.;c=setup(clock,status,two,2,two[0],125000*1e-9);prepare(&c);
 long expected=long(floor(clock[1]*1e9*.5));decide(&c);
 need(!c.accept && c.used_s==two[0] && c.next_ns==expected && c.rejected==1,"rejection bypassed");
 status[0]=1e300;c=setup(clock,status,two,2,two[0],125000*1e-9);prepare(&c);decide(&c);
 need(!c.accept && c.used_s==two[0],"domain rejection sentinel bypassed");
 std::cout<<"{\"status\":\"PASS_EXTRACTED_CPU_FUNCTIONS\",\"checks\":8}"<<std::endl;return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
'''
    source = HERE / "endpoint_contract.cpp"
    executable = HERE / "endpoint_contract_cpu"
    source.write_text(prefix+"\n".join(parts)+main_code)
    command = ["g++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off",
               str(source), "-o", str(executable)]
    compile_result = subprocess.run(command, text=True, capture_output=True, timeout=30)
    need(compile_result.returncode == 0, compile_result.stderr)
    checked = subprocess.run([str(executable)], text=True, capture_output=True, timeout=10)
    need(checked.returncode == 0, checked.stderr)
    result = json.loads(checked.stdout)
    origin_ns = 1523250000
    origin = origin_ns*1e-9
    boundary = 2.8348089914503676e-05
    absolute = origin+boundary
    adjacent = math.nextafter(boundary, math.inf)
    result.update(gpu_executed=False, neural_simulations_executed=0,
                  cpu_compile_command=command,
                  source_sha256={str(p): sha(p) for p in (controller_path,runtime_path,port_path,source)},
                  absolute_clock_check=dict(origin_ns=origin_ns, relative_boundary_s=boundary,
                      absolute_boundary_s=absolute, recovered_relative_s=absolute-origin,
                      recovered_relative_difference_s=absolute-origin-boundary,
                      adjacent_relative_boundaries_collapse=(origin+adjacent == absolute),
                      current_runtime_scope="CNS uses epoch-relative events; no origin+boundary absolute float is used for decisions"))
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
