"""Raw close after postprocessing NameError; never reruns the physical body."""
from pathlib import Path
import hashlib, json, math, time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'campanas/etapa4_reference_budget_20260924_27/reference_plus_01'


def need(ok, msg):
    if not ok:raise ValueError(msg)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, x):
    with p.open('x',encoding='utf-8') as f:
        json.dump(x,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n')


def main():
    start=time.monotonic()
    plan=json.loads((HERE/'PLAN.json').read_text())
    repair=json.loads((HERE/'CLOSE_REPAIR_PLAN_02.json').read_text())
    need(repair['budget']['new_body_arms_max']==0 and not (HERE/'RESULT.json').exists(), 'Not a fresh close')
    need(json.loads((HERE/'FAILURE.json').read_text())=={
        'type':'NameError','message':"name 'refine_y' is not defined",'stage4_admission':False,'stage5_admission':False},'Original failure changed')
    lock=json.loads((HERE/'SOURCE_LOCK.json').read_text())
    for rel, old_hash in lock.items():
        path=Path(rel) if rel.startswith('/') else ROOT/rel
        need(sha(path)==old_hash, 'Frozen source changed: '+rel)
    arms={}; traces={}; hashes={}
    for name in plan['arms']:
        path=HERE/name/'RESULT.json';trace=HERE/name/'trace.npz'
        arms[name]=json.loads(path.read_text())
        need(arms[name]['status']=='COMPLETE' and arms[name]['body_ms']==200 and
             arms[name]['name']==name, 'Incomplete body arm')
        with np.load(trace,allow_pickle=False) as z:
            need(set(z.files)=={'qpos','qvel','yaw_deg','contacts','normal_N','upright','pulse_native'},'Trace schema')
            traces[name]={k:z[k].copy() for k in z.files}
        need(all(a.shape[0]==200 and np.isfinite(a).all() for a in traces[name].values()),'Trace domain')
        hashes[name]={'result':sha(path),'trace':sha(trace)}
    z=traces['zero_25us'];p=traces['pulse_25us'];fine=traces['pulse_12p5us']
    need(all(r['initial_integration_sha256']==arms['zero_25us']['initial_integration_sha256'] for r in arms.values()), 'Starts differ')
    pulse=json.loads((HERE/'PULSE_SPEC.json').read_text())
    expected=np.zeros(200)
    expected[plan['model']['pulse_start_ms']:plan['model']['pulse_end_ms_exclusive']]=pulse['torque_native']
    need(np.array_equal(z['pulse_native'],np.zeros(200)) and np.array_equal(p['pulse_native'],expected)
         and np.array_equal(fine['pulse_native'],expected),'Pulse window/content')
    need(arms['pulse_25us']['pulse_substeps']==800 and arms['pulse_12p5us']['pulse_substeps']==1600,'Pulse physical step count')
    with np.load(SOURCE/'traces.npz',allow_pickle=False) as ref:
        trial=np.flatnonzero(ref['fase']=='ensayo')[:200]
        errors={k:float(np.max(abs(z[out]-ref[src][trial]))) for out,src,k in
                (('qpos','qpos','qpos'),('qvel','qvel','qvel'),('yaw_deg','yaw_delta_deg','yaw_deg'))}
    c=plan['checks']
    effect=float(p['yaw_deg'][-1]-z['yaw_deg'][-1])
    refine_yaw=float(np.max(abs(p['yaw_deg']-fine['yaw_deg'])))
    refine_xy=float(np.max(abs(p['qpos'][:,:2]-fine['qpos'][:,:2]))*10)
    normal_ratio=float(np.min(np.minimum(p['normal_N'],fine['normal_N'])/z['normal_N']))
    elapsed=float(json.loads((HERE/'PROGRESS_pulse_12p5us.json').read_text())['elapsed_s'])
    checks={
        'historical_zero':errors['qpos']<=c['zero_qpos_max_abs'] and errors['qvel']<=c['zero_qvel_max_abs'] and errors['yaw_deg']<=c['zero_yaw_deg_max_abs'],
        'effect_sign':effect*plan['model']['torque_sign']>0,
        'effect_size':c['pulse_final_yaw_difference_deg_min']<=abs(effect)<=c['pulse_final_yaw_difference_deg_max'],
        'refined_yaw':refine_yaw<=c['pulse_refinement_yaw_deg_max_abs'],
        'refined_xy':refine_xy<=c['pulse_refinement_xy_mm_max_abs'],
        'contacts':min(int(np.min(p['contacts'])),int(np.min(fine['contacts'])))>=c['contact_count_each_ms_min'],
        'support':normal_ratio>=c['normal_force_ratio_vs_zero_min'],
        'upright':min(float(np.min(p['upright'])),float(np.min(fine['upright'])))>=c['upright_each_ms_min'],
        'body_budget':elapsed<=plan['budget']['CPU_wall_total_s_max']}
    need(elapsed>0 and time.monotonic()-start<=repair['budget']['CPU_wall_s_max'],'Budgets')
    out={'schema':'stage45_body_pulse_postprocess_repair_v1',
         'classification':'PROMETEDOR_NO_CONFIRMADO' if all(checks.values()) else 'DESCARTADO_EN_ESTE_CONTRATO',
         'body_only':True,'organism_runs':0,'stage4_admission':False,'stage5_admission':False,
         'original_runner_failure_sha256':sha(HERE/'FAILURE.json'),
         'original_runner_source_sha256':lock[str((HERE/'run_preflight.py').relative_to(ROOT))],
         'plan_sha256':lock[str((HERE/'PLAN.json').relative_to(ROOT))],
         'source_lock_sha256':sha(HERE/'SOURCE_LOCK.json'),
         'repair_plan_sha256':sha(HERE/'CLOSE_REPAIR_PLAN_02.json'),
         'repair_code_sha256':sha(Path(__file__)),
         'arm_hashes':hashes,'baseline_errors':errors,'pulse_spec':pulse,
         'effect_final_deg':effect,'refinement_yaw_sup_deg':refine_yaw,
         'refinement_xy_sup_mm':refine_xy,'normal_force_ratio_min':normal_ratio,
         'checks':checks,'body_wall_total_s':elapsed,'close_cpu_s':time.monotonic()-start,
         'interpretation':'No new body arms; failed fixed effect minimum is not retuned. Next rival may use a changed source rather than increasing the pulse.'}
    write_new(HERE/'CLOSE_02.json',out)
    print(json.dumps({k:out[k] for k in ('classification','effect_final_deg','refinement_yaw_sup_deg','refinement_xy_sup_mm','normal_force_ratio_min','baseline_errors','checks','body_wall_total_s','close_cpu_s')},indent=2))


if __name__=='__main__': main()
