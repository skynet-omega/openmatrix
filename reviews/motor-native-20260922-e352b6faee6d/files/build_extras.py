"""Build optional B preconditioner and established ERK reference against the local pinned dependency."""
from pathlib import Path
import subprocess,uuid
H=Path(__file__).resolve().parent
def main():
    logdir=H/'build_runs'/uuid.uuid4().hex[:12];logdir.mkdir(parents=True)
    for src,dst in [('bridge_blocks.cpp','libopenmatrix_blocks.so'),('reference_erk.cpp','libreference_erk.so')]:
        args=['/usr/bin/g++-12','-std=c++17','-O3','-shared','-fPIC',src,'-I'+str(H/'vendor/install/include'),'-L'+str(H/'vendor/install/lib'),'-Wl,-rpath,$ORIGIN/vendor/install/lib','-lsundials_arkode','-lsundials_nveccuda','-lsundials_sunlinsolspgmr','-lsundials_core','-lcudart','-lcuda','-lnvrtc','-o',dst]
        with (logdir/(src+'.log')).open('w') as f:subprocess.run(args,cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(str(logdir))
if __name__=='__main__':main()
