"""Counterfactual odor tape on recorded donor antenna positions only."""
from pathlib import Path
import hashlib,json,time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DONOR=ROOT/'campanas/etapa4_mirrored_source_20260924_26/native_plus_01/traces.npz'
FIELDS=ROOT/'campanas/etapa4_mirrored_source_20260924_26/CAMPOS.json'

def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.monotonic()
    plan=json.loads((HERE/'PLAN.json').read_text());f=json.loads(FIELDS.read_text())
    need(plan['schema']=='stage45_source_switch_geometric_screen_v1' and
         f['plus']['sigma_mm']==f['minus']['sigma_mm'],'Source/plan identity')
    with np.load(DONOR,allow_pickle=False) as z:
        trial=np.flatnonzero(z['fase']=='ensayo')
        need(len(trial)==400 and np.array_equal(z['paso'][trial],np.arange(1,401)),'Trial clock')
        antenna=z['antenas_mm'][trial,:,:2].copy()
        recorded=z['concentracion_campo'][trial].copy()
        used=z['sensores_usados'][trial,:2].copy()
        pending=z['sensores_pendientes'][trial,:2].copy()
        need(np.array_equal(recorded,pending),'Recorded pending/field mismatch')
        need(np.array_equal(used[1:],pending[:-1]),'One-ms committed sensory lag missing')
    def concentration(which):
        src=np.asarray(f[which]['source_mm'],dtype=float)
        sigma=float(f[which]['sigma_mm'])
        need(src.shape==(2,) and np.isfinite(src).all() and sigma>0,'Source domain')
        return np.exp(-np.sum((antenna-src)**2,axis=-1)/(2*sigma*sigma))
    plus=concentration('plus');minus=concentration('minus')
    err=float(np.max(abs(plus-recorded)))
    tape=plus.copy();tape[100:]=minus[100:]
    need(np.isfinite(tape).all() and np.min(tape)>=0 and np.max(tape)<=1,'Tape domain')
    lr_plus=plus[:,0]-plus[:,1];lr_tape=tape[:,0]-tape[:,1]
    late=slice(200,400)
    late_delta=float(np.mean(abs(lr_tape[late]-lr_plus[late])))
    c=plan['criteria']
    checks={
       'prefix_exact':err<=c['prefix_plus_field_max_abs'] and np.array_equal(tape[:100],plus[:100]),
       'late_material':late_delta>c['post_switch_mean_abs_signed_contrast_change_min'],
       'opposite_sign':float(np.mean(lr_plus[late]))>0 and float(np.mean(lr_tape[late]))<0,
       'budget':time.monotonic()-start<=plan['budget']['wall_s_max']}
    with (HERE/'donor_new_source_tape.npz').open('xb') as out:
        np.savez_compressed(out,concentration_pending=tape,source_plus_concentration=plus,
                            source_minus_concentration=minus,source_change_after_ms=np.array(100,dtype=np.int64))
    result={'schema':'stage45_source_switch_geometry_v1',
            'classification':'GEOMETRY_ONLY_ELIGIBLE' if all(checks.values()) else 'GEOMETRY_SCREEN_FAIL',
            'stage4_admission':False,'stage5_admission':False,'organism_runs':0,
            'donor_sha256':sha(DONOR),'fields_sha256':sha(FIELDS),
            'plan_sha256':sha(HERE/'PLAN.json'),'code_sha256':sha(Path(__file__)),
            'tape_sha256':sha(HERE/'donor_new_source_tape.npz'),
            'original_plus_recompute_max_abs':err,
            'new_first_pending_at_trial_ms':101,
            'first_new_consumed_at_trial_ms':102,
            'mean_signed_LR_plus_201_400':float(np.mean(lr_plus[late])),
            'mean_signed_LR_new_201_400':float(np.mean(lr_tape[late])),
            'mean_abs_change_signed_LR_201_400':late_delta,
            'checks':checks,'cpu_wall_s':time.monotonic()-start,
            'interpretation':'Counterfactual source on saved poses; no living online branch or motor response.'}
    with (HERE/'RESULT.json').open('x') as out:
        json.dump(result,out,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
