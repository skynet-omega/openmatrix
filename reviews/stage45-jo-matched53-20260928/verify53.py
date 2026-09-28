"""Reconstruct scientific conclusions from raw input/output, without CNS imports."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def arrays(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def contract(root):
    freeze = json.loads((root/'PILOT_FREEZE.json').read_text())
    need(sha(root/'PILOT_PLAN.json') == freeze['plan_sha256'], 'Contract changed')
    return json.loads((root/'PILOT_PLAN.json').read_text())


def qualify(root):
    contract(root)
    new = root/'qual_identity_input_observer'
    ref = root/'reference52/qual_parent_wide'
    r = json.loads((new/'RESULT.json').read_text())
    need(r['status'] == 'COMPLETE' and r['initial_exact'] and r['committed_ms'] == r['attempted_ms'] == 2,
         'Qualification incomplete')
    counts = {}
    for name in ['traces.npz', 'neural_and_inputs.npz', 'wide_observation.npz', 'dng100_observed.npz']:
        a, b = arrays(new/name), arrays(ref/name)
        need(set(b) <= set(a), 'Missing legacy fields')
        for k in b:
            need(np.array_equal(a[k], b[k]), 'Qualification differs: '+name+':'+k)
        counts[name] = len(b)
    for name in ['OWNERS_FINAL.json', 'EVENTS.json']:
        need(json.loads((new/name).read_text()) == json.loads((ref/name).read_text()),
             'Qualification scientific owner/event mismatch: '+name)
    a = arrays(new/'neural_and_inputs.npz')
    need(np.array_equal(a['JO_consumed_FP32'], a['JO_drive'].astype(np.float32)), 'FP32 observer mismatch')
    need(np.all(a['JO_kernel_audit'][:, 0] > 0) and np.all(a['JO_kernel_audit'][:, 1] == 0), 'Unobserved or invalid RHS')
    return dict(status='PASS', CNS_ms=2, legacy_fields=counts, scientific_owners_and_events_exact=True,
                FP32_input_observed=True, scope='New read-only probe, normalization off, versus prior52 qualification; not biological equivalence')


def compute(root):
    p = contract(root)
    need(p['duration_ms'] == 90 and p['prefix_ms'] == 10 and p['analysis_window_ms'] == [51,90], 'Temporal contract')
    need(p['criteria']['DNb05_directional_q_min'] == 1.6e-5 and p['criteria']['raw_yaw_deg_s_min'] == .02, 'Materiality changed')
    qualify(root)
    anatomy = arrays(root/'JO_anatomy_arrays.npz')
    panel = arrays(root/'PANEL.npz')
    data, reports, hashes = {}, {}, {}
    initial_q = None
    common_total = None
    window = slice(50,90)
    for arm, spec in p['arms'].items():
        d = root/arm
        r = json.loads((d/'RESULT.json').read_text())
        need(r['status'] == 'COMPLETE' and r['attempted_ms'] == r['committed_ms'] == 90 and r['initial_exact'], 'Incomplete '+arm)
        need(r['setting'] == spec and r['plan_sha256'] == sha(root/'PILOT_PLAN.json'), 'Wrong condition/plan')
        a, t, wide = arrays(d/'neural_and_inputs.npz'), arrays(d/'traces.npz'), arrays(d/'wide_observation.npz')
        need(all(np.isfinite(v).all() for v in a.values()), 'Nonfinite '+arm)
        need(a['q'].shape == (90,16) and a['JO_drive'].shape == (90,335), 'Shape')
        need(np.array_equal(a['JO_ids'],anatomy['source_ids']) and np.array_equal(a['JO_rows'],anatomy['source_rows']), 'JO identity')
        need(np.array_equal(a['ids'],[10045,10056,10118,10065,10442,10760,523769,10360,10888,11067,11074,512006,11960,11702,523640,10371]), 'Reader identity')
        need(np.array_equal(wide['ids'],panel['ids']) and np.array_equal(wide['rows'],panel['canonical_rows']), 'Panel identity')
        need(wide['q'].shape == (90,2757) and np.isfinite(wide['q']).all(), 'Wide observations')
        need(np.array_equal(wide['time_ns'],t['CNS_time_ns']), 'Panel clock')
        need(np.array_equal(t['CNS_time_ns'],47486000000+np.arange(1,91,dtype=np.int64)*1000000), 'Clock')
        need(np.array_equal(t['paso'],np.arange(3001,3091)), 'Interval order')
        need(np.array_equal(a['q'][:,:4],t['DN_q_actual']), 'Reader projection')
        need(np.array_equal(t['DN_q_usada'][1:],t['DN_q_actual'][:-1]), 'Motor latency')
        need(np.array_equal(t['command_yaw_rate_rad_s'],np.zeros(90)), 'Yaw applied unexpectedly')
        kin=a['air_kinematics'];need(kin.shape==(90,15),'Kinematics')
        need(np.array_equal(kin[1:,3:6],10*t['qvel'][:-1,:3]), 'Body velocity latency/units')
        need(np.array_equal(kin[0,3:6],10*a['initial_native_velocity']), 'Initial velocity units')
        rot0=kin[0,6:].reshape(3,3)
        field=rot0@np.array([0.,100.*spec['air'],0.])
        need(np.array_equal(kin[:,:3],np.broadcast_to(field,(90,3))), 'World field')
        side=(anatomy['source_side']=='R').astype(int)
        sign=np.where(np.char.startswith(anatomy['source_type'],'JO-C'),1.,-1.)
        axes=np.array([[1.,1.,0.],[1.,-1.,0.]])/np.sqrt(2.)

        def encode(field,velocity,rotation):
            speed=np.maximum((axes@(rotation.T@(field-velocity)))[side]*sign,0.)
            return 80.*(speed/(speed+100.))

        reference_totals=np.array([encode(rot0@np.array([0.,100.*s,0.]),kin[0,3:6],rot0).sum() for s in [1,-1]])
        need(np.array_equal(a['JO_reference_totals'],reference_totals), 'Total selected from wrong state')
        total=float(reference_totals.min())
        need(float(a['JO_target_total'])==total,'Fixed target total')
        if common_total is None:common_total=total
        need(total==common_total,'Different totals between conditions')
        raw=np.zeros((90,335))
        for i,x in enumerate(kin[10:],10):raw[i]=encode(x[:3],x[3:6],x[6:].reshape(3,3))
        need(np.array_equal(a['JO_raw'],raw),'Independent receptor reconstruction')
        expected=np.zeros((90,335),np.float32)
        expected[10:]=(raw[10:]*(total/raw[10:].sum(axis=1))[:,None]).astype(np.float32)
        need(np.array_equal(a['JO_drive'],expected.astype(np.float64)), 'Matched input transform')
        need(np.array_equal(a['JO_consumed_FP32'],expected), 'Actual kernel input')
        need(np.array_equal(raw>0,expected>0), 'Support changed')
        need(np.all(a['JO_kernel_audit'][:,0]>0) and np.all(a['JO_kernel_audit'][:,1]==0), 'Kernel witness incomplete')
        totals=expected[10:].astype(np.float64).sum(axis=1)
        need(np.all(np.abs(totals-total)<=1e-7*total),'Postcast dose mismatch')
        nominal=a['nominal_Hz'];need(nominal.shape==(90,694),'ORN shape')
        need(np.array_equal(nominal[:10],np.broadcast_to(nominal[0],(10,694))),'Odor prefix')
        if spec['odor']:need(np.max(nominal[10:]-nominal[0])>0,'Missing odor')
        else:need(np.array_equal(nominal,np.broadcast_to(nominal[0],nominal.shape)),'No-odor contaminated')
        if initial_q is None:initial_q=a['initial_q']
        need(np.array_equal(initial_q,a['initial_q']),'Initial neuronal state')
        old=arrays(root/'reference52'/arm/'neural_and_inputs.npz')
        need(np.array_equal(a['q'][:10],old['q'][:10]),'Common prefix changed')
        need(np.array_equal(nominal,old['nominal_Hz']),'Odor intervention changed')
        delta=t['DN_q_usada']-t['DN_baseline']
        yaw=np.tanh(250*(delta[:,2]-delta[:,3]))*np.deg2rad(5.)
        need(np.array_equal(yaw,t['neural_yaw_unapplied_rad_s']),'Decoder changed')
        direction=a['q'][:,2]-a['q'][:,3]
        data[arm]=dict(a=a,t=t,wide=wide,direction=direction,yaw=yaw*180/np.pi,totals=totals)
        reports[arm]=dict(mean_DNb_q=float(direction[window].mean()),mean_yaw_deg_s=float((yaw*180/np.pi)[window].mean()),
            total_mean=float(totals.mean()),max_total_error=float(np.abs(totals-total).max()),
            active_cells_min=int((expected[10:]>0).sum(axis=1).min()),
            L2_mean=float(np.linalg.norm(expected[10:].astype(np.float64),axis=1).mean()),
            receptor_scale_min=float((total/raw[10:].sum(axis=1)).min()),receptor_scale_max=float((total/raw[10:].sum(axis=1)).max()),
            mean_forward_mm_s=float(t['forward_unclipped_mm_s'][window].mean()),kernel_evaluations=int(a['JO_kernel_audit'][:,0].sum()))
        hashes[arm]={name:sha(d/name) for name in ['RESULT.json','traces.npz','neural_and_inputs.npz','wide_observation.npz']}
    contrasts={}
    for odor in [0,1]:
        left,right=data[f'airL_odor{odor}'],data[f'airR_odor{odor}']
        qseries=(left['direction']-right['direction'])/2
        yseries=(left['yaw']-right['yaw'])/2
        q,y=float(qseries[window].mean()),float(yseries[window].mean())
        dose=float(np.max(np.abs(left['totals']-right['totals'])))
        need(dose<=2e-7*common_total,'Pair dose bound')
        groups={}
        dif=left['wide']['q']-right['wide']['q']
        for name,mask in zip(panel['group_names'],panel['group_masks']):
            v=dif[:,mask]
            groups[str(name)]=dict(population=int(mask.sum()),ever_above_threshold=int(np.any(np.abs(v)>1e-6,axis=0).sum()),
                endpoint_above_threshold=int((np.abs(v[-1])>1e-6).sum()),window_RMS=float(np.sqrt(np.mean(v[window]**2))))
        contrasts[str(odor)]=dict(half_L_minus_R_DNb_q=q,half_L_minus_R_yaw_deg_s=y,
            both_material=bool(abs(q)>=1.6e-5 and abs(y)>=.02),max_pair_total_difference=dose,
            positive_samples=int((yseries[window]>0).sum()),negative_samples=int((yseries[window]<0).sum()),groups=groups)
    result=dict(schema='jo_matched53_verified_v1',qualification=qualify(root),common_total=common_total,
        arms=reports,contrasts=contrasts,odor_interaction=dict(q=contrasts['1']['half_L_minus_R_DNb_q']-contrasts['0']['half_L_minus_R_DNb_q'],
            yaw_deg_s=contrasts['1']['half_L_minus_R_yaw_deg_s']-contrasts['0']['half_L_minus_R_yaw_deg_s']),
        source_hashes=hashes,stage4_pass=False,stage5_pass=False,
        limits='One exposed preparation; diagnostic external transformation; no physical yaw or perturbation; L2/support/anatomy not matched.')
    return result,data


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qualification',action='store_true');ap.add_argument('--root',type=Path,default=H)
    args=ap.parse_args()
    result=qualify(args.root) if args.qualification else compute(args.root)[0]
    filename='QUALIFICATION.json' if args.qualification else 'RESULTADOS.json'
    (args.root/filename).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result if args.qualification else {k:result[k] for k in ['contrasts','odor_interaction','stage4_pass','stage5_pass']}))


if __name__=='__main__':
    main()
