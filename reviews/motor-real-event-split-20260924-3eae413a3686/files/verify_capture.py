"""Compare the instrumented 1-ms organism to the frozen full-state parent."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/'architecture_round_20260924_01'


def main():
    output = HERE/('NEUTRALITY_01.json' if __debug__ else 'NEUTRALITY_01_OPTIMIZED.json')
    if output.exists():
        raise FileExistsError(output)
    spec = importlib.util.spec_from_file_location('pair_verifier', PREV/'verify_fusion_pair.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.BASE = PREV/'fusion_base_1ms_01'
    verifier.FUSED = HERE/'capture_01'
    states = {name:verifier.compare_encoded(name) for name in ('session','prosthesis','published')}
    exact = {name:verifier.sha(verifier.BASE/name)==verifier.sha(verifier.FUSED/name)
             for name in ('final_state/boundary.json','EVENT_AUDIT.json','traces.npz')}
    runs = [json.loads((p/'RESULT.json').read_text()) for p in (verifier.BASE,verifier.FUSED)]
    fields = ('epochs','accepted','rejected','event_capacity_per_cell_per_epoch','mandatory_event_boundaries')
    exact['CNS_scientific_counts'] = all(runs[0]['runtime']['CNS'][k]==runs[1]['runtime']['CNS'][k] for k in fields)
    exact['events'] = runs[0]['runtime']['events']==runs[1]['runtime']['events']
    complete = all(r['status']=='COMPLETE' and r['completed_trial_ms']==1 and not r['cleanup_errors'] for r in runs)
    # Exercise the same semantic comparator on a changed scientific scalar.
    detected=[]
    verifier.differ({'time_ns':1000},{'time_ns':1001},{},{},'owner',detected)
    mutation_detected=detected==['owner/time_ns']
    passed=complete and mutation_detected and all(exact.values()) and all(s['equal'] and s['nan_count']==0 for s in states.values())
    result={'schema':'effective_capture_neutrality_v1','neutral':passed,
            'complete_1ms_pair':complete,'semantic_states':states,'exact_other':exact,
            'deliberate_clock_mutation_detected':mutation_detected,
            'reference':str(verifier.BASE),'capture':str(verifier.FUSED),
            'comparator_sha256':hashlib.sha256((PREV/'verify_fusion_pair.py').read_bytes()).hexdigest(),
            'scope':'All serialized scientific owners, RNG, events, output and counts in this one sham step; not universal correctness.',
            'stage_admission':False}
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0 if passed else 2


if __name__=='__main__':
    sys.exit(main())
