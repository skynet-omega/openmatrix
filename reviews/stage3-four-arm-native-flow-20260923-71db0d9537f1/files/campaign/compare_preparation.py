"""Compare science fields before the odor switch, ignoring ZIP container bytes."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from compare_smoke import compare,FIELDS

H=Path(__file__).resolve().parent
BASE=H/'full_sham_01/prepared_state'

def main():
    a=json.loads((BASE/'session.json').read_text())
    results={}
    for label,folder in [('odor_left',H/'full_odor_left_01/prepared_state'),
                         ('odor_right',H/'full_odor_right_01/prepared_state'),
                         ('uniform',H/'full_uniform_01/prepared_state')]:
        if not folder.is_dir():continue
        b=json.loads((folder/'session.json').read_text())
        differs=[]
        with np.load(BASE/'session.npz',allow_pickle=False) as za,np.load(folder/'session.npz',allow_pickle=False) as zb:
            for field in FIELDS:compare(a[field],b[field],za,zb,field,differs)
        results[label]={'scientific_state_exact':not differs,
                        'different_paths':differs[:50],'different_path_count':len(differs)}
    output={'schema':'stage3_four_arm_preodor_semantic_comparison_v1','arms':results,
            'scope':'Selected complete scientific trees from the serialized prepared state; binary ZIP equality is not required.'}
    (H/'PREPARATION_COMPARE.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(output))
    if any(not x['scientific_state_exact'] for x in results.values()):raise SystemExit(2)

if __name__=='__main__':main()
