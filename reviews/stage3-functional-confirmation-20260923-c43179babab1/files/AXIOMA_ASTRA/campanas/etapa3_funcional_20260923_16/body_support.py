"""Measure historical support on exact physical replays of the four native trajectories."""
from pathlib import Path
import hashlib,importlib.util,json,time,resource
import numpy as np

HERE=Path(__file__).resolve().parent;NATIVE=HERE.parent/'etapa3_pn629_intervention_20260923_15'
SOURCE=HERE.parent/'etapa3_causal_controls_20260923_11/body_replay.py'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
replay=module('unchanged_body_replay',SOURCE)
mechanics=module('preserved_contact_observer',HERE/'mechanical_observer_original.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if not __debug__:raise RuntimeError('Historical body code requires normal Python')
    plan=json.loads((HERE/'PLAN.json').read_text());out=HERE/'body_support_01';out.mkdir(exist_ok=False)
    start=time.monotonic();prep=NATIVE/'full_sham_01/prepared_state'
    manifest=json.loads((prep/'MANIFEST.json').read_text())
    for n in ('session.json','session.npz','prosthesis.json','prosthesis.npz'):
        if sha(prep/n)!=manifest['files'][n]['sha256']:raise ValueError('Prepared source changed')
    state=replay.read_state(prep/'session');body_state=state['body'];del state
    prosthesis=replay.read_state(prep/'prosthesis');controller_state=prosthesis['controller'];del prosthesis
    source={'prepared_manifest_sha256':sha(prep/'MANIFEST.json'),'body_replay_sha256':sha(SOURCE),
            'observer_sha256':sha(HERE/'mechanical_observer_original.py'),'harness_sha256':sha(Path(__file__)),
            'plan_sha256':sha(HERE/'PLAN.json')}
    original=replay.RHTarsalBody.from_state;descriptor=replay.RHTarsalBody.__dict__.get('from_state')
    holder={};cases={}
    def restore(cls,s):
        body=original(s);observer=mechanics.MechanicalObserver(body);samples=[];read_sample=observer._sample
        initial=observer.read();close=body.close
        def sample(dt):
            read_sample(dt)
            if observer.sample_steps%40==0:samples.append(observer.read())
        observer._sample=sample;observer.start()
        def close_observed():
            try:observer.close()
            finally:close()
        body.close=close_observed;holder.update(samples=samples,initial=initial)
        return body
    replay.RHTarsalBody.from_state=classmethod(restore)
    try:
        for arm in ('sham','odor_right','odor_left','uniform'):
            if time.monotonic()-start>plan['budget']['mechanical_wall_total_s_max']-30:raise TimeoutError('Mechanical replay budget')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>plan['budget']['RAM_GiB_max']:raise MemoryError('Mechanical RAM budget')
            path=NATIVE/('full_'+arm+'_01')/'traces.npz'
            with np.load(path,allow_pickle=False) as z:
                trial=z['fase']=='ensayo'
                commands=np.stack((z['command_forward_mm_s'][trial],z['command_yaw_rate_rad_s'][trial]),axis=1)
                reference={k:z[k][trial].copy() for k in ('qpos','qvel','yaw_delta_deg')}
            result=replay.run_case(arm,body_state,controller_state,commands,out/arm,dict(source,trace_sha256=sha(path)),reference)
            samples=holder['samples'];mg=holder['initial']['mgN']
            duration=np.asarray([0.]+[r['duration_s'] for r in samples]);impulse=np.asarray([[0.,0.,0.]]+[r['vertical_impulse_Ns'] for r in samples])
            np.savez_compressed(out/arm/'support.npz',duration_s=duration,vertical_impulse_Ns=impulse,mg_N=mg)
            if max(result['reference_errors'].values())!=0.:raise ValueError('Physical replay was not exact: '+arm)
            if len(samples)!=400 or not np.allclose(duration,np.arange(401)*.001,rtol=0,atol=1e-9):raise ValueError('Mechanical sample clock')
            support=(impulse[10:]-impulse[:-10])/(.010*mg);pose=reference['qpos'];up=1-2*(pose[:,4]**2+pose[:,5]**2)
            row={'reference_errors':result['reference_errors'],'min_leg_10ms_weight_fraction':float(support[:,1].min()),
                 'max_abdominal_10ms_weight_fraction':float(support[:,0].max()),'upright_min':float(up.min()),'wall_s':result['wall_s']}
            c=plan['criteria'];row['passed']=(row['min_leg_10ms_weight_fraction']>=c['min_leg_10ms_weight_fraction'] and row['max_abdominal_10ms_weight_fraction']<=c['max_abdominal_10ms_weight_fraction'] and row['upright_min']>=c['upright_min'])
            cases[arm]=row;(out/'RESULT.json').write_text(json.dumps({'cases':cases,'source':source,'wall_s':time.monotonic()-start,'complete':len(cases)==4,'scope':'Exact replay of the physical trajectory only; not a neural withdrawal or full organism continuation'},indent=2)+'\n')
            if not row['passed']:raise ValueError('Native support failed: '+arm)
        if time.monotonic()-start>plan['budget']['mechanical_wall_total_s_max']:raise TimeoutError('Aggregate mechanical budget')
        print(json.dumps({'cases':cases,'wall_s':time.monotonic()-start}))
    finally:
        if descriptor is None:delattr(replay.RHTarsalBody,'from_state')
        else:replay.RHTarsalBody.from_state=descriptor

if __name__=='__main__':
    try:main()
    except BaseException as e:
        (HERE/'BODY_SUPPORT_FAILURE.json').write_text(json.dumps({'type':type(e).__name__,'message':str(e),'stage3_admission':False},indent=2)+'\n')
        raise
