"""Rebuild the pinned local dependency; no global installation or credentials."""
from pathlib import Path
import subprocess,os,json,hashlib,tarfile,uuid
H=Path(__file__).resolve().parent
SHA='ddf5daba8397ea89287a0fec6f1b3bc3fe6c548b'
LOGDIR=H/'build_runs'/uuid.uuid4().hex[:12]
def run(args,log):
    LOGDIR.mkdir(parents=True,exist_ok=True)
    with (LOGDIR/log).open('w') as f:subprocess.run(args,cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
    src=H/'vendor/sundials';archive=H/'vendor/sundials-7.6.0-build.tar.gz'
    expected=json.loads((H/'DEPENDENCY.json').read_text())['archive_sha256']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected:raise ValueError('upstream archive hash mismatch')
    with tarfile.open(archive) as t:
        for m in t.getmembers():
            if m.name.startswith('/') or '..' in Path(m.name).parts or not(m.isfile() or m.isdir()):raise ValueError('unsafe upstream member')
        if not src.exists():
            src.mkdir(parents=True);t.extractall(src)
        for m in t.getmembers():
            if m.isfile() and (src/m.name).read_bytes()!=t.extractfile(m).read():raise ValueError('expanded upstream source changed: '+m.name)
    install=H/'vendor/install';build=H/'vendor/build_gcc12'
    run(['cmake','-S',str(src),'-B',str(build),'-DCMAKE_C_COMPILER=/usr/bin/gcc-12','-DCMAKE_CXX_COMPILER=/usr/bin/g++-12','-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12','-DCMAKE_BUILD_TYPE=Release',f'-DCMAKE_INSTALL_PREFIX={install}','-DENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=89','-DSUNDIALS_PRECISION=double','-DSUNDIALS_INDEX_SIZE=64','-DBUILD_ARKODE=ON','-DBUILD_CVODE=OFF','-DBUILD_CVODES=OFF','-DBUILD_IDA=OFF','-DBUILD_IDAS=OFF','-DBUILD_KINSOL=OFF','-DBUILD_STATIC_LIBS=OFF','-DEXAMPLES_ENABLE_C=OFF','-DEXAMPLES_ENABLE_CXX=OFF','-DEXAMPLES_ENABLE_CUDA=OFF'],'build_configure.log')
    run(['cmake','--build',str(build),'--parallel','4'],'build_compile.log')
    run(['cmake','--install',str(build)],'build_install.log')
    run(['/usr/bin/g++-12','-std=c++17','-O3','-shared','-fPIC','bridge.cpp','-I'+str(install/'include'),'-L'+str(install/'lib'),'-Wl,-rpath,$ORIGIN/vendor/install/lib','-lsundials_arkode','-lsundials_nveccuda','-lsundials_sunlinsolspgmr','-lsundials_core','-lcudart','-o','libopenmatrix_ark.so'],'build_bridge.log')
    print('Pinned SUNDIALS CUDA and adapter built locally; logs: '+str(LOGDIR))
if __name__=='__main__':main()
