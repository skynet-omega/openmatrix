"""Posthoc distance description; deliberately outside the frozen Stage4 gate."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    fields=json.loads((HERE/'CAMPOS.json').read_text())
    out={'schema':'stage4_mirrored_trajectory_descriptive_v1',
         'scope':'Single 400-ms pair; absolute distance reduction can result from common forward motion. No navigation/feedback admission.',
         'fields_sha256':sha(HERE/'CAMPOS.json'),'code_sha256':sha(__file__),'arms':{}}
    for arm in ('plus','minus'):
        path=HERE/f'native_{arm}_01/traces.npz'
        with np.load(path,allow_pickle=False) as z:
            if z['fase'].tolist()!=['preparacion']*40+['ensayo']*400:
                raise ValueError('Only complete 40+400-ms trace')
            source=np.asarray(fields[arm]['source_mm'],dtype=float)
            body=z['position_mm'][:,:2]
            antennas=z['antenas_mm'][:,:,:2].mean(axis=1)
            db=np.linalg.norm(body-source,axis=1)
            da=np.linalg.norm(antennas-source,axis=1)
            out['arms'][arm]={'trace_sha256':sha(path),
                              'body_start_mm':float(db[39]),'body_end_mm':float(db[-1]),
                              'body_change_mm':float(db[-1]-db[39]),
                              'antenna_midpoint_start_mm':float(da[39]),
                              'antenna_midpoint_end_mm':float(da[-1]),
                              'antenna_midpoint_change_mm':float(da[-1]-da[39]),
                              'body_xy_change_mm':(body[-1]-body[39]).tolist()}
    out['body_distance_change_plus_minus_mm']=out['arms']['plus']['body_change_mm']-out['arms']['minus']['body_change_mm']
    out['antenna_distance_change_plus_minus_mm']=out['arms']['plus']['antenna_midpoint_change_mm']-out['arms']['minus']['antenna_midpoint_change_mm']
    target=HERE/'TRAJECTORY_DESCRIPTIVE_01.json'
    with target.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
