"""Portable raw-data reconstruction plus corruption rejection, no CNS imports."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import json
import shutil
import tempfile
import time
import resource
import numpy as np
from verify53 import compute, arrays, sha, need

H=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=H);parser.add_argument('--corruptions',action='store_true')
    args=parser.parse_args();root=args.root.resolve();start=time.process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(60,65))
    expected=json.loads((root/'RESULTADOS.json').read_text())
    result,_=compute(root)
    need(result==expected,'Reconstructed result differs from saved result')
    if (root/'MANIFEST.json').exists():
        manifest=json.loads((root/'MANIFEST.json').read_text())
        for name,item in manifest['files'].items():
            need(sha(root/name)==item['sha256'],'Delivery member changed: '+name)
    rejected=[]
    if args.corruptions:
        with tempfile.TemporaryDirectory(prefix='jo53-check-') as td:
            copy=Path(td)/'data';copy.mkdir()
            top=['PILOT_PLAN.json','PILOT_FREEZE.json','PANEL.npz','JO_anatomy_arrays.npz']
            for name in top:shutil.copy2(root/name,copy/name)
            for name in ['reference52','qual_identity_input_observer']:
                shutil.copytree(root/name,copy/name)
            for arm in result['arms']:
                (copy/arm).mkdir()
                for name in ['RESULT.json','traces.npz','neural_and_inputs.npz','wide_observation.npz']:
                    shutil.copy2(root/arm/name,copy/arm/name)
            arm='airL_odor0'
            # These alterations target physical meaning, observation and decisions.
            cases=[('wrong_kernel_input','neural_and_inputs.npz','JO_consumed_FP32',(30,0),1.),
                   ('wrong_physical_units','neural_and_inputs.npz','air_kinematics',(30,3),1.),
                   ('changed_source_identity','neural_and_inputs.npz','JO_ids',(0,),1),
                   ('missing_kernel_evaluation','neural_and_inputs.npz','JO_kernel_audit',(30,0),-1),
                   ('changed_neural_output','neural_and_inputs.npz','q',(30,2),.01),
                   ('applied_yaw','traces.npz','command_yaw_rate_rad_s',(30,),.01)]
            for label,name,key,index,value in cases:
                path=copy/arm/name;values=arrays(path)
                if label=='missing_kernel_evaluation':values[key][index]=0
                else:values[key][index]+=value
                np.savez_compressed(path,**values)
                try:compute(copy)
                except ValueError as exc:rejected.append(dict(case=label,error=str(exc)))
                else:raise ValueError('Corruption accepted: '+label)
                shutil.copy2(root/arm/name,path)
            path=copy/'PILOT_PLAN.json';plan=json.loads(path.read_text());plan['criteria']['raw_yaw_deg_s_min']=0.
            path.write_text(json.dumps(plan))
            try:compute(copy)
            except ValueError as exc:rejected.append(dict(case='changed_materiality',error=str(exc)))
            else:raise ValueError('Changed criterion accepted')
    out=dict(status='PASS',exact_result_reconstruction=True,corruptions_rejected=rejected,
             CPU_s=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             new_CNS_ms=0,scope='Recorded-data reproducibility; not a new neural simulation or portable GPU resume')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
