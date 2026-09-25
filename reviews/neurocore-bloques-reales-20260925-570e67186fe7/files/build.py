"""Build the generic scheduler from source, without changing an existing binary."""
from pathlib import Path
import argparse
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    # Keep the local compiler paired with the 12.1 headers/device runtime below.
    # PATH also has a 12.0 nvcc; mixing it with libcudadevrt 12.1 cannot link.
    local_nvcc=Path('/home/daroch/miniconda3/pkgs/cuda-nvcc-12.1.66-0/bin/nvcc')
    p.add_argument('--nvcc',default=str(local_nvcc) if local_nvcc.is_file() else shutil.which('nvcc'))
    p.add_argument('--cxx',default=shutil.which('g++-12') or shutil.which('g++'))
    p.add_argument('--include',default='/home/daroch/miniconda3/pkgs/cuda-cudart-dev-12.1.55-0/include')
    p.add_argument('--libdir',default='/home/daroch/miniconda3/pkgs/cuda-cudart-static-12.1.55-0/lib')
    p.add_argument('--arch',default='sm_89')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise ValueError('Build output must be new')
    if not a.nvcc or not a.cxx:raise ValueError('Specify a compatible CUDA compiler and C++ compiler')
    command=[a.nvcc,'-std=c++17','-O3','--shared','-rdc=true','-arch='+a.arch,
             '-ccbin='+a.cxx,'-Xcompiler=-fPIC','-I'+a.include,'-L'+a.libdir,
             str(Path(__file__).with_name('resident_controller.cu')),'-o',str(a.out.resolve())]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
