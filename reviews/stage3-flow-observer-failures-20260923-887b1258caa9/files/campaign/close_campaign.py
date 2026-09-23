"""Mechanical receipt for the two bounded instrumentation campaigns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'etapa3_flujo_20260923_01'


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = {name:read(PREV/name/'RESULT.json') for name in ('smoke_off_01','smoke_on_01')}
    new = {name:read(HERE/name/'RESULT.json') for name in ('smoke_on_01','smoke_on_02')}
    expect = (
        old['smoke_off_01']['status'] == 'COMPLETE' and
        old['smoke_off_01']['completed_trial_ms'] == 1 and
        old['smoke_on_01']['completed_trial_ms'] == 0 and
        'Unreconstructed effective coefficient' in old['smoke_on_01']['error']['message'] and
        new['smoke_on_01']['completed_trial_ms'] == 0 and
        'No final accepted CNS coefficient stage' in new['smoke_on_01']['error']['message'] and
        new['smoke_on_02']['completed_trial_ms'] == 0 and
        'Unreconstructed effective coefficient' in new['smoke_on_02']['error']['message']
    )
    if not expect:
        raise ValueError('Unexpected source campaign result; inspect before closing')
    if list(HERE.glob('full_*')) or list(PREV.glob('full_*')):
        raise ValueError('Unexpected full run found')
    fixture = read(HERE/'GRAPH_CAPTURE_FIXTURE.json')
    if fixture['callbacks'] != 18 or fixture['accepted'] <= 0:
        raise ValueError('CUDA graph fixture did not verify capture/replay')
    refs = {
        'historical_readback':PREV/'HISTORICAL_LATE_READBACK.json',
        'chatgpt_original_python':PREV/'chatgpt_analizar_flujo.py',
        'first_plan':PREV/'PLAN.json',
        'second_plan':HERE/'PLAN.json',
        'graph_fixture':HERE/'GRAPH_CAPTURE_FIXTURE.json',
        'current_observer':HERE/'flow_observer.py',
    }
    result = {
        'schema':'stage3_flow_instrumentation_closure_v1',
        'classification':'BLOQUEADO_POR_INSTRUMENTACION',
        'stage3_pass':None,
        'previous_budget':{'zero_loads':1,'smokes':2,'full_400ms':0},
        'current_budget':{'smokes':2,'full_400ms':0,'graph_fixture':1},
        'smokes':{
            'previous':{name:{'status':v['status'],'ms':v['completed_trial_ms'],
                              'wall_s':v['wall_total_s'],'error':None if v['error'] is None else v['error']['message']}
                        for name,v in old.items()},
            'current':{name:{'status':v['status'],'ms':v['completed_trial_ms'],
                             'wall_s':v['wall_total_s'],'error':v['error']['message']}
                       for name,v in new.items()},
        },
        'sha256':{name:sha(path) for name,path in refs.items()},
        'external_analysis':'ChatGPT original code passed synthetic selftest locally, not real flow; its declared SHA is verified in this receipt.',
        'scientific_scope':'No four-arm protected-organism readout or long-horizon numerical reference. No causal localization or stage-3 admission.',
    }
    if result['sha256']['chatgpt_original_python'] != 'db74b46d18770475df8eb4d77de7d05e731d37aea1abffd4208360fa93915884':
        raise ValueError('ChatGPT code digest changed')
    (HERE/'CLOSE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
