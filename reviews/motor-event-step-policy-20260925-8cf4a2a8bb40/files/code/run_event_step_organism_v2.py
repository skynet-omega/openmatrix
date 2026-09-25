"""Run the same one-ms organism with either compiled event-step controller."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PLAN = HERE/'EVENT_STEP_PLAN_42.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def run(mode, out):
    start = time.monotonic()
    plan = json.loads(PLAN.read_text())
    need(plan['schema'] == 'event_step_organism_plan_v1', 'Plan schema')
    need(mode in ('baseline','candidate'), 'Mode')
    need(not out.exists(), 'Unique output required')
    for rel, expected in plan['frozen_sha256'].items():
        need(sha(ROOT/rel) == expected, 'Frozen source changed: '+rel)
    lib = HERE/'event_step_binary_v2'/('lib'+mode+'.so')
    need(sha(lib) == plan['binary_sha256'][mode], 'Binary changed')
    sys.path[:0] = [str(ROOT/'motor_nuevo/epoch_cost_20260923'),
                    str(ROOT/'motor_nuevo/pipeline_review_20260922')]
    import run_set
    import runtime_session
    selected = {}
    original_init = runtime_session.RuntimeSession.__init__
    def install_selected(session, *args, **kwargs):
        # run_set verifies that event owners were not imported before its local
        # source selection. The adapter is therefore patched only afterward.
        import organism_adapter
        parent = organism_adapter.NativeGraph
        class SelectedNativeGraph(parent):
            def __init__(self, *a, **kw):
                need(kw.get('native_library') is not None, 'Event native library required')
                kw['native_library'] = lib
                super().__init__(*a, **kw)
        organism_adapter.NativeGraph = SelectedNativeGraph
        selected.update(module=organism_adapter,parent=parent)
        return original_init(session,*args,**kwargs)
    runtime_session.RuntimeSession.__init__ = install_selected
    original_argv = sys.argv
    sys.argv = ['run_set.py','--out',str(out),'--odor','sham',
                '--engine','causal_cuda','--ms','1','--observe','off',
                '--cuda-profile','off','--profile','off','--kc-capture','off']
    try:
        code = run_set.main()
    finally:
        sys.argv = original_argv
        runtime_session.RuntimeSession.__init__ = original_init
        if selected:
            selected['module'].NativeGraph = selected['parent']
    result = json.loads((out/'RESULT.json').read_text())
    receipt = {'schema':'event_step_mode_receipt_v1','mode':mode,
               'plan_sha256':sha(PLAN),'native_source_sha256':plan['native_source_sha256'][mode],
               'binary_sha256':sha(lib),'run_result_sha256':sha(out/'RESULT.json'),
               'preparation_weights_sha256':sha(out/'preparation_inputs/intervenciones_W.npz'),
               'status':result['status'],'runner_exit_code':code,
               'wall_s':time.monotonic()-start,
               'step_wall_s':sum(json.loads(line)['step_wall_s'] for line in
                                 (out/'PROGRESS.jsonl').read_text().splitlines())
                             if (out/'PROGRESS.jsonl').exists() else 0.,
               'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'disk_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}
    budget = plan['budget_per_run']
    receipt['budget_ok'] = (receipt['wall_s'] <= budget['wall_s_max'] and
                            receipt['maxrss_kib'] <= budget['ram_gib_max']*1024**2 and
                            receipt['disk_bytes'] <= budget['disk_bytes_max'])
    (out/'EVENT_STEP_RECEIPT.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    need(code == 0 and result['status'] == 'COMPLETE' and receipt['budget_ok'],
         'Organism or budget failed')
    print(json.dumps({k:receipt[k] for k in ('mode','status','step_wall_s','wall_s',
                                            'maxrss_kib','budget_ok')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('baseline','candidate'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    run(args.mode,args.out)
