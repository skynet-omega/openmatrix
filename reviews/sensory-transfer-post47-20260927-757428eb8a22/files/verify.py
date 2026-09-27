"""Recompute evidence and reject source corruption and an altered sensory protocol."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import numpy as np
from compare import calculate, need

HERE = Path(__file__).resolve().parent

def main():
    result, _, _ = calculate(HERE)
    need(result == json.loads((HERE/'RESULTADOS.json').read_text()), 'Recalculation differs')
    with tempfile.TemporaryDirectory(prefix='sensory-verify-') as td:
        target=Path(td)
        for name in ('OBSERVATIONS.npz','PROVENANCE.json','CONTRACT.json','DOOR_EXTRACTION.json'):
            shutil.copy2(HERE/name,target/name)
        shutil.copytree(HERE/'primary_door',target/'primary_door')
        path=target/'primary_door/Or42b.csv'
        original=path.read_bytes();path.write_bytes(original.replace(b'83.667',b'84.667',1))
        try:
            calculate(target)
        except ValueError as e:
            need(str(e)=='Biological source changed','Unexpected rejection')
        else:
            raise ValueError('Changed biological source accepted')
        path.write_bytes(original)
        with np.load(target/'OBSERVATIONS.npz',allow_pickle=False) as z:
            data={k:z[k].copy() for k in z.files}
        data['odor__sensores_usados'][1000,0]=.51
        np.savez_compressed(target/'OBSERVATIONS.npz',**data)
        provenance=json.loads((target/'PROVENANCE.json').read_text())
        provenance['projection_sha256']=hashlib.sha256((target/'OBSERVATIONS.npz').read_bytes()).hexdigest()
        (target/'PROVENANCE.json').write_text(json.dumps(provenance))
        try:
            calculate(target)
        except ValueError as e:
            need(str(e)=='Consumed protocol differs','Unexpected semantic rejection')
        else:
            raise ValueError('Changed stimulus accepted after rehashing')
    print(json.dumps(dict(recalculation=True,source_corruption_rejected=True,
                          rehashed_wrong_stimulus_rejected=True,new_neural_steps=0)))

if __name__=='__main__':
    main()
