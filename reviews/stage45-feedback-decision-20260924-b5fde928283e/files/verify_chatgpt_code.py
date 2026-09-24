"""Bounded local verification of the unedited external geometry calculator."""
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = ROOT/'campanas/etapa4_reference_budget_20260924_27/prepared/reviews/stage4-reference-recovery-20260924-323cd648010a/files'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    with (HERE/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    start = time.monotonic()
    code = HERE/'criba_feedback_chatgpt_original.py'
    inputs = [FILES/p for p in (
        'runs/plus/executed_sources/16__etapa4_diseno_20260923_17__GEOMETRY.json',
        'recovery/CAMPOS.json', 'recovery/CLOSE_01.json', 'prior/CLOSE_01.json')]
    lock = {str(p.relative_to(ROOT)):sha(p) for p in [code,Path(__file__),HERE/'EXTERNAL_CODE_PLAN.md',*inputs]}
    save('EXTERNAL_CODE_LOCK.json', lock)
    spec = importlib.util.spec_from_file_location('external_feedback_unchanged', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proposed = module.run(FILES, 2, 1.5, 30)
    revised = module.run(FILES, 2, .5, 1)
    need(proposed['maximum_command_integral_after_perturbation_deg']==2.5, '30deg command integral')
    need(proposed['earliest_command_only_compensation_from_on_s']==7.5, '30deg minimum command time')
    need(math.isclose(revised['earliest_command_only_compensation_from_on_s'], .7, rel_tol=0, abs_tol=1e-15), '1deg time')
    need(math.isclose(revised['ideal_equal_speed_position_separation_upper_mm'], .005235921299024, rel_tol=0, abs_tol=1e-14), 'ideal XY geometry')
    root = json.loads((HERE/'RAW_FEASIBILITY_02.json').read_text())
    for side in ('plus', 'minus'):
        a = root['runs']['reference_'+side]
        b = revised['geometry'][side]
        need(abs(b['bearing_initial_deg']-a['heading_error_start_deg']) < 1e-7, 'independent initial heading')
        need(abs(b['distance_initial_mm']-a['distance_start_mm']) < 1e-8, 'independent initial distance')
    negative = []
    for values in ((2,2,1),(2,.5,0),(float('nan'),.5,1)):
        try:
            module.run(FILES,*values)
        except ValueError as e:
            negative.append(str(e))
        else:
            raise ValueError('invalid input accepted')
    need(len(negative)==3, 'negative count')
    need(all(sha(ROOT/p)==v for p,v in lock.items()), 'source changed during test')
    save('EXTERNAL_CODE_RESULT.json', {
        'status':'CPU_GEOMETRY_RECOMPUTED_NO_ORGANISM',
        'response_message_id':'306c27aa-6e90-43c6-ae95-57f1963ed299',
        'original_code_unmodified': True, 'source_lock_sha256': sha(HERE/'EXTERNAL_CODE_LOCK.json'),
        'gemini_case':proposed, 'chatgpt_case':revised, 'invalid_inputs_rejected':negative,
        'wall_s':time.monotonic()-start,
        'limitation':'Geometry at initial pose, not pulse time; cost includes scaled preparation; no CNS/MuJoCo response.'})
    print(json.dumps({'geometry_checked':revised['geometry'], 'cost_h':revised['cost_linear_h'],
                      'negatives':len(negative),'wall_s':time.monotonic()-start},indent=2))


if __name__=='__main__':
    main()
