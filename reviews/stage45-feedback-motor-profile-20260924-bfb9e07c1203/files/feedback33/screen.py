"""Geometry-only readback of an already exposed mirrored-source native pair."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAM=ROOT/'campanas/etapa4_mirrored_source_20260924_26'
FIELDS=CAM/'CAMPOS.json'
PLUS=CAM/'native_plus_01/traces.npz'
MINUS=CAM/'native_minus_01/traces.npz'
CLOSE=ROOT/'campanas/etapa4_reference_budget_20260924_27/CLOSE_01.json'

def need(ok,msg):
    if not ok:raise ValueError(msg)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def read_trace(path):
    with np.load(path,allow_pickle=False) as z:
        trial=np.flatnonzero(z['fase']=='ensayo')
        need(len(trial)==400 and np.array_equal(z['paso'][trial],np.arange(1,401)), 'Trial clock '+str(path))
        return {'prepared':z['qpos'][trial[0]-1].copy(),
                'antenna':z['antenas_mm'][trial].copy(),
                'field':z['concentracion_campo'][trial].copy(),
                'qpos':z['qpos'][trial].copy(),
                'yaw':z['yaw_delta_deg'][trial].copy()}

def calc(points,spec):
    src=np.asarray(spec['source_mm'],dtype=float)
    sigma=float(spec['sigma_mm'])
    need(src.shape==(2,) and np.isfinite(src).all() and np.isfinite(sigma) and sigma>0,'Field')
    return np.exp(-np.sum((points[...,:2]-src)**2,axis=-1)/(2*sigma*sigma))

def main():
    start=time.monotonic()
    plan=json.loads((HERE/'PLAN.json').read_text())
    need(plan['schema']=='stage45_exposed_pair_spatial_feedback_screen_v1', 'Plan')
    need(not (HERE/'RESULT.json').exists(), 'Already screened')
    original=json.loads(CLOSE.read_text())
    need(original['stage4_admission'] is False, 'Wrong prior classification')
    fields=json.loads(FIELDS.read_text())
    plus=read_trace(PLUS);minus=read_trace(MINUS)
    need(np.array_equal(plus['prepared'],minus['prepared']),'Prepared bodies differ')
    need(all(np.isfinite(array).all() for t in (plus,minus) for array in t.values()),'Nonfinite trace')
    original_plus=calc(plus['antenna'],fields['plus'])
    original_minus=calc(minus['antenna'],fields['minus'])
    e_plus=float(np.max(abs(original_plus-plus['field'])))
    e_minus=float(np.max(abs(original_minus-minus['field'])))
    need(e_plus<=plan['criteria']['plus_field_reconstruction_max_abs'] and
         e_minus<=plan['criteria']['original_minus_field_reconstruction_max_abs'], 'Original field mismatch')
    minus_on_plus=calc(plus['antenna'],fields['minus'])
    lr_live=original_minus[:,0]-original_minus[:,1]
    lr_yoked=minus_on_plus[:,0]-minus_on_plus[:,1]
    gap=np.abs(lr_live-lr_yoked)
    late=gap[300:400]
    mean_late=float(np.mean(late))
    budget=plan['budget']
    elapsed=time.monotonic()-start
    need(elapsed<=budget['wall_s_max'] and
         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<budget['RAM_GiB_max'],
         'CPU/RAM budget')
    result={'schema':'stage45_existing_feedback_screen_result_v1',
            'classification':('A_MATERIAL_GEOMETRIC_DIFFERENCE' if mean_late>=plan['criteria']['late_301_400_mean_abs_signed_LR_difference_min']
                              else 'B_SMALL_GEOMETRIC_DIFFERENCE'),
            'source_sha256':{str(path):sha(path) for path in (PLUS,MINUS,FIELDS,CLOSE)},
            'plan_sha256':sha(HERE/'PLAN.json'),'code_sha256':sha(Path(__file__)),
            'plus_recompute_max_abs':e_plus,'minus_recompute_max_abs':e_minus,
            'late_301_400_mean_abs_signed_LR_difference':mean_late,
            'full_1_400_max_abs_signed_LR_difference':float(np.max(gap)),
            'final_signed_LR_live_minus_source':float(lr_live[-1]),
            'final_signed_LR_on_plus_donor':float(lr_yoked[-1]),
            'final_native_yaw_gap_deg':float(abs(plus['yaw'][-1]-minus['yaw'][-1])),
            'first100_mean_abs_gap':float(np.mean(gap[:100])),
            'late_mean_abs_xy_antenna_separation_mm':float(np.mean(np.linalg.norm(minus['antenna'][300:,:,:2]-plus['antenna'][300:,:,:2],axis=-1))),
            'organism_runs':0,'body_runs':0,'stage4_admission':False,'stage5_admission':False,
            'scope':'Same minus field on already-exposed different live poses; not a source-switch organism or formal feedback bound.',
            'cpu_wall_s':elapsed}
    with (HERE/'RESULT.json').open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
