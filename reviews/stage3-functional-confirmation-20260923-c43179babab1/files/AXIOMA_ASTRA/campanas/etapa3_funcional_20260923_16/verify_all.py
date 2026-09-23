"""Recompute the prospective functional result; execution booleans are insufficient."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import check_remaining as remaining
from checkpoint_compare import sha
from check_support import check as support

PLAN_SHA='04129eb49efc057b142230ce2b9470b446dfdf2ad68d417e2e462868bccfaa41'
EXTERNAL_SHA='51ae660168bdfd4ba9595cf7362c6492354a708b93e3835d4f6c7f93c6e3ff54'
ARMS=('sham','odor_left','odor_right','uniform')
def need(ok,msg):
    if not ok:raise ValueError(msg)
def js(p):return json.loads(p.read_text())
def raw_flow(folder,operator_folder):
    """Rebuild consumed targets/rates from native samples, not their PASS receipt."""
    folder,operator_folder=Path(folder),Path(operator_folder)
    descriptor=js(operator_folder/'effective_operator.json');meta=js(folder/'flow/METADATA.json')
    f=remaining.arrays(folder/'flow/FLOW.npz');ids=list(map(str,f['ids']))
    rows=np.asarray([meta['rows'][k]['row'] for k in ids],dtype=np.int64)
    params={}
    with np.load(operator_folder/'effective_operator.npz',allow_pickle=False) as z:
        for name in ('gain','theta','tau'):
            value=z[descriptor['values'][name]['__array__']]
            need(np.isfinite(value).all(),'Nonfinite effective operator')
            params[name]=value[rows]
    records=[json.loads(line) for line in (folder/'flow/FLOW.jsonl').read_text().splitlines()]
    need(len(records)==len(f['time_ns']),'Native flow samples missing')
    raw=np.asarray([[r['rows'][k]['raw_signed'] for k in ids] for r in records]);target=np.asarray([[r['rows'][k]['target'] for k in ids] for r in records])
    drive=np.asarray([[r['rows'][k]['drive'] for k in ids] for r in records]);theta=np.asarray([[r['rows'][k]['theta'] for k in ids] for r in records])
    rate=np.asarray([[r['rows'][k]['rate'] for k in ids] for r in records])
    need(all(np.isfinite(v).all() for v in (raw,target,drive,theta,rate)),'Nonfinite flow record')
    need(np.array_equal(raw,f['raw_signed']) and np.array_equal(target,f['target']),'Native flow archive/log differ')
    need(np.array_equal(np.asarray([r['time_ns'] for r in records]),f['time_ns']),'Native flow clocks differ')
    need(np.array_equal(theta,np.tile(params['theta'],(len(records),1))),'Flow operator differs from saved effective operator')
    expected=np.maximum(0.,np.tanh(params['gain']*(raw+drive-params['theta'])))
    target_error=float(np.max(np.abs(expected-target)));rate_error=float(np.max(np.abs(1./params['tau']-rate)))
    need(target_error<=1e-9 and rate_error<=1e-12,'Raw native flow fails target/rate reconstruction')
    return {'samples':len(records),'max_target_error':target_error,'max_rate_error':rate_error,'recomputed_from_raw_log':True}
def verify_sources(folder):
    frozen=js(folder/'FROZEN.json')
    for name,digest in frozen.items():need(sha(folder/'executed_sources'/name)==digest,'Changed executed source '+name)
    extra=js(folder/'EXPERIMENT_CONTRACT.json')['sources']
    for i,(name,digest) in enumerate(extra.items()):
        p=folder/'experiment_sources'/(str(i)+'__'+Path(name).name)
        need(sha(p)==digest,'Changed experimental source '+name)
def native_support(root,native,criteria):
    out={}
    for arm in ARMS:
        folder=root/'body_support_01'/arm
        a=remaining.arrays(folder/'trace.npz');b=remaining.arrays(native/('full_'+arm+'_01')/'traces.npz')
        for key in ('qpos','qvel','yaw_delta_deg','command_forward_mm_s','command_yaw_rate_rad_s'):
            need(np.array_equal(a[key],b[key][40:]),'Mechanical replay differs from CNS trajectory '+arm+'/'+key)
        s=remaining.arrays(folder/'support.npz');dt=s['duration_s'];imp=s['vertical_impulse_Ns'];mg=float(s['mg_N'])
        need(dt.shape==(401,) and imp.shape==(401,3) and mg>0,'Incomplete native support')
        need(np.allclose(dt,np.arange(401)*.001,rtol=0,atol=1e-9),'Native mechanical clock')
        ratios=(imp[10:]-imp[:-10])/(.010*mg);up=float(np.min(1-2*(a['qpos'][:,4]**2+a['qpos'][:,5]**2)))
        legs=float(ratios[:,1].min());abd=float(ratios[:,0].max())
        need(legs>=criteria['min_leg_10ms_weight_fraction'] and abd<=criteria['max_abdominal_10ms_weight_fraction'] and up>=criteria['upright_min'],'Native support gate '+arm)
        out[arm]={'min_leg_10ms_weight_fraction':legs,'max_abdominal_10ms_weight_fraction':abd,'upright_min':up,
                  'support_sha256':sha(folder/'support.npz'),'physical_replay_exact':True}
    return out
def verify(root,native):
    root,native=Path(root).resolve(),Path(native).resolve()
    need(sha(root/'PLAN.json')==PLAN_SHA,'Changed prospective contract')
    plan=js(root/'PLAN.json');criteria=plan['criteria'];source=native/'chatgpt_verificador_original.py'
    need(sha(source)==EXTERNAL_SHA==plan['external_code_sha256'],'Changed external observable comparator')
    spec=importlib.util.spec_from_file_location('preserved_external',source);ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
    need(ext.EPS==criteria['yaw_sup_deg_max']==criteria['command_L1_deg_max'] and ext.MATERIAL==criteria['material_deg'],'External criteria differ')
    hashes={};pairs={};observations={};reference_support={};flows={};elapsed=0.
    for arm in ARMS:
        a=native/('full_'+arm+'_01');b=root/('reference_'+arm+'_01');verify_sources(b)
        # load/compare recomputes yaw from pose, time/exposure, reader law,
        # absolute command differences and descriptive accepted physical events.
        pair=[ext.load(a,arm,'causal_cuda',hashes),ext.load(b,arm,'reference_cuda',hashes)]
        observation=ext.compare(*pair);need(observation['observation_pair_ok'],'Observable mismatch '+arm)
        observations[arm]=observation;pairs[arm]=pair
        flows[arm]={'native':raw_flow(a,b/'prepared_state'),'reference':raw_flow(b,b/'prepared_state')}
        r=js(b/'RESULT.json');need(0<r['wall_total_s']<=plan['budget']['wall_each_s_max'],'Reference wall budget')
        elapsed+=r['wall_total_s'];reference_support[arm]=support(b,criteria);need(reference_support[arm]['passed'],'Reference support '+arm)
    orientation=ext.orientation(pairs);need(orientation.get('complete') and orientation['both_directional'],'Bilateral functional orientation failed')
    mechanical=native_support(root,native,criteria)
    need(js(root/'body_support_01/RESULT.json')['wall_s']<=plan['budget']['mechanical_wall_total_s_max'],'Replay wall budget')
    # Reuse the deterministic scientific checks without overwriting historical receipts.
    prior=(remaining.HERE,remaining.NATIVE,remaining.save)
    remaining.HERE,remaining.NATIVE,remaining.save=root,native,lambda *_:None
    try:
        withdrawals=remaining.withdrawal_pair(root/'withdrawal_odor_left_01',root/'withdrawal_odor_right_01')
        continuation=remaining.continuation(root/'continuation_right_01')
    finally:remaining.HERE,remaining.NATIVE,remaining.save=prior
    for name in ('withdrawal_odor_left_01','withdrawal_odor_right_01'):
        verify_sources(root/name);elapsed+=js(root/name/'RESULT.json')['wall_total_s']
        flows[name]=raw_flow(root/name,root/name/'prepared_state')
    resume=root/'continuation_right_01';elapsed+=js(resume/'RESULT.json')['wall_total_s']
    executed=js(resume/'RUN_CONTRACT.json')['sources']
    flows['continuation']=raw_flow(resume,resume/'state_100ms')
    for i,(name,digest) in enumerate(executed.items()):need(sha(resume/'executed_sources'/(str(i)+'__'+Path(name).name))==digest,'Continuation source changed')
    need(elapsed<=plan['budget']['organism_wall_total_s_max'],'Aggregate organism wall budget')
    return {'schema':'verified_functional_stage3_scope_v1','classification':'CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA',
        'functional_stage3_pass':True,'historical_hidden_state_gate':'FAIL_PRESERVADO',
        'scope':plan['scope'],'biological_equivalence':False,'source_navigation_demonstrated':False,'stage4_completed':False,
        'observable_pairs':observations,'native_flow_reconstruction':flows,'orientation':orientation,'reference_support':reference_support,
        'native_support':mechanical,'coupled_withdrawals':withdrawals,'continuation':continuation,
        'organism_runs':7,'organism_wall_s':elapsed,'budget':plan['budget'],'input_hashes':hashes,
        'contract_sha256':PLAN_SHA,'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--native',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    need(not a.out.exists(),'Preserve previous verification receipts')
    try:result=verify(a.root,a.native or a.root.parent/'etapa3_pn629_intervention_20260923_15');code=0
    except Exception as e:result={'classification':'BLOQUEADO','functional_stage3_pass':False,'error':{'type':type(e).__name__,'message':str(e)}};code=2
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in result if k not in ('observable_pairs','input_hashes','native_support','reference_support','coupled_withdrawals','continuation')},indent=2));return code
if __name__=='__main__':raise SystemExit(main())
