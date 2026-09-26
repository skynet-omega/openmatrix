"""Small resource snapshots outside measured neural/body advancement."""
from pathlib import Path
import os,subprocess,resource

def sample():
    processes=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=path.read_bytes().decode(errors='replace').split('\0')
            if args and 'python' in args[0]:
                scripts=[x for x in args[1:] if x.endswith('.py')]
                if scripts:processes.append({'pid':int(path.parent.name),'script':scripts[0]})
        except (OSError,ValueError):pass
    r=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu',
                      '--format=csv,noheader'],capture_output=True,text=True,timeout=5)
    return {'pid':os.getpid(),'maxrss_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'python_processes':processes,'gpu_sample':r.stdout.strip() if r.returncode==0 else 'UNAVAILABLE'}
