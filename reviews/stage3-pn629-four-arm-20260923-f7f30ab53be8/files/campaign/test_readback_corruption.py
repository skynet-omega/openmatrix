"""Deliberately corrupt copies of real completed data; never mutate the evidence."""
from pathlib import Path
import json,shutil,tempfile
from verify_readback import HERE,PARENT,verify

def main():
    verify()
    with tempfile.TemporaryDirectory(prefix='readback-corruption-',dir=HERE) as td:
        root=Path(td)
        original=json.loads((HERE/'CLOSE.json').read_text())
        for name in ('PLAN.json','CLOSE.json'):
            shutil.copyfile(HERE/name,root/name)
        for name in original['runs']:
            folder=root/('full_'+name+'_01');folder.mkdir()
            shutil.copyfile(HERE/folder.name/'traces.npz',folder/'traces.npz')
        shutil.copytree(HERE/'body_factorial_01',root/'body_factorial_01')
        verify(root,PARENT)
        mutations={
            'promotion_flag':lambda d:d.update(stage3_admission=True),
            'stop_flag':lambda d:d.update(stop_after_sham_right=not d['stop_after_sham_right']),
            'reported_yaw':lambda d:d['runs']['odor_right']['windows']['400'].update(yaw_delta_deg=0.0),
            'missing_arm':lambda d:d['runs'].pop('uniform'),
        }
        detected=[]
        for name,mutate in mutations.items():
            value=json.loads(json.dumps(original));mutate(value)
            (root/'CLOSE.json').write_text(json.dumps(value))
            try:verify(root,PARENT)
            except ValueError:detected.append(name)
            else:raise RuntimeError('Corruption undetected: '+name)
        shutil.copyfile(HERE/'CLOSE.json',root/'CLOSE.json')
        plan=json.loads((root/'PLAN.json').read_text());plan['material_effect_deg']=0.0
        (root/'PLAN.json').write_text(json.dumps(plan))
        try:verify(root,PARENT)
        except ValueError:detected.append('criterion')
        else:raise RuntimeError('Modified criterion undetected')
    print(json.dumps({'real_data_verified':True,'detected_corruptions':detected}))

if __name__=='__main__':main()
