"""Portable reconstruction from recordings, including deliberate corruption tests."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import copy
import hashlib
import json
import tempfile
import time
import numpy as np
from analyze_wide52 import analyze
from verify_qualification52 import verify as qualify, arrays, need
from verify_epochs52 import verify as epochs

H = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def check(root, manifest=True):
    if manifest:
        m = json.loads((root/'MANIFEST.json').read_text())
        for name, entry in m['files'].items():
            path = root/name
            need(path.is_file() and path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], 'Manifest mismatch: '+name)
    q = qualify(root)
    need(q == json.loads((root/'QUALIFICATION.json').read_text()), 'Qualification recomputation')
    populations, result = analyze(root)
    need(result == json.loads((root/'PILOT_RESULTS.json').read_text()), 'Original metrics recomputation')
    expected = json.loads((root/'POPULATIONS.json').read_text())
    expected.pop('CPU_s', None)
    need(populations == expected, 'Population summary or correction metrics changed')
    plan = json.loads((root/'PILOT_PLAN.json').read_text())
    trial_reports = {}
    for arm in plan['arms']:
        n = arrays(root/arm/'neural_and_inputs.npz')
        o = arrays(root/arm/'dng100_observed.npz')
        trial_reports[arm] = epochs(o, n['q'], n['initial_q'][o['rows']])
    receipt = json.loads((root/'QUEUE_RESULT.json').read_text())
    actual = [json.loads((root/('qual_'+name)/'RESULT.json').read_text()) for name in plan['qualifications']]
    actual += [json.loads((root/name/'RESULT.json').read_text()) for name in plan['arms']]
    need(all(x['status']=='COMPLETE' for x in actual), 'Incomplete run')
    for key in ['attempted_ms','committed_ms','CPU_s']:
        need(sum(r[key] for r in actual) == receipt[key], 'Budget receipt does not match arms: '+key)
    need(receipt['attempted_ms'] == receipt['committed_ms'] == 916, 'Exposure budget')
    need(receipt['CPU_s'] <= plan['budgets']['CPU_s'] and receipt['queue_wall_s'] <= plan['budgets']['wall_s'], 'Execution budget exceeded')
    need(result['stage4_pass'] is False and result['stage5_pass'] is False, 'Behavioral stage scope')
    verdict=json.loads((root/'VERDICT.json').read_text())
    air_material=all(result['air'][str(o)]['both_material'] for o in [0,1])
    expected_verdict=dict(classification='PROMETEDOR_NO_CONFIRMADO' if air_material or result['conductance']['promising_screen'] else 'DESCARTADO',
        repair_classification='CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA',
        air_original_joint_gate_both_odors=air_material,
        conductance_specific_gate=result['conductance']['promising_screen'],
        stage4_pass=False,stage5_pass=False,physical_air_units_reconstructed=True,
        wide_observer_qualified=True,attempted_neural_ms=receipt['attempted_ms'],
        next_quantity_control=populations['input_quantity_matching_next'])
    need(verdict==expected_verdict,'Verdict not reconstructed from data')
    old=arrays(root/'parent51_projection.npz');anatomy=arrays(root/'donors/JO_anatomy_arrays.npz')
    L=old['airL_odor0__JO_drive'][10:];R=old['airR_odor0__JO_drive'][10:]
    perm=np.arange(335)
    for typ in np.unique(anatomy['source_type']):
        for side in ['L','R']:
            rows=np.flatnonzero((anatomy['source_type']==typ)&(anatomy['source_side']==side))
            perm[rows]=rows[::-1]
    screen=json.loads((root/'INPUT_CONTROL_SCREEN.json').read_text())
    changed=int(sum(np.count_nonzero(v[:,perm]!=v) for v in [L,R]))
    need(changed==screen['C']['changed_input_values'] and int((perm!=np.arange(335)).sum())==screen['C']['permuted_id_positions'],'Permutation control recomputation')
    common=min(L[0].sum(),R[0].sum());a=[v*(common/v.sum(axis=1))[:,None] for v in [L,R]]
    need(float(common)==screen['A']['common_L1'],'L1 construction')
    need(float(max(np.abs(v.sum(axis=1)-common).max() for v in a))==screen['A']['L1_max_residual'],'L1 residual')
    need(float(np.mean(np.linalg.norm(a[1],axis=1)/np.linalg.norm(a[0],axis=1)))==screen['A']['L2_ratio_R_over_L_mean'],'Remaining L2 confound')
    from analyze_fp32_boundary import compute as fp32_compute
    fp32_expected=json.loads((root/'FP32_BOUNDARY.json').read_text());fp32_expected.pop('CPU_s',None)
    need(fp32_compute(root)==fp32_expected,'FP32 boundary reconstruction')
    return dict(status='PASS', scientific_arms=10, recorded_CNS_ms=916,
                original_metrics_exact=True, population_metrics_exact=True,
                corrected_units_reconstructed=True, live_observer_pairs_exact=True,
                solver_epochs=trial_reports,
                scope='CPU reconstruction of records. No new CNS, no52 GPU resume or behavioral admission.')


def corruption_tests(root):
    """Exercise scientific checks, not merely the outer file hash guard."""
    rejected=[]
    def reject(label, fn):
        try:
            fn()
        except (ValueError, KeyError) as exc:
            rejected.append(dict(case=label, reason=str(exc)))
        else:
            raise ValueError('Corruption escaped: '+label)
    d=root/'qual_parent_wide'
    o=arrays(d/'dng100_observed.npz'); n=arrays(d/'neural_and_inputs.npz')
    for key in ['committed','ms','duration_ns']:
        bad={k:v.copy() for k,v in o.items()}
        bad[key][0]=True if key=='committed' else bad[key][0]+1
        reject('solver_'+key, lambda bad=bad: epochs(bad,n['q'],n['initial_q'][o['rows']]))
    import verify_qualification52 as qualmod
    # A temporary view shares immutable input files; only specified corruptions
    # become private files. Nothing in the campaign is edited.
    with tempfile.TemporaryDirectory(prefix='corruption52_',dir=root.parent) as tmp:
        temp=Path(tmp)/root.name;temp.mkdir()
        relevant=['PILOT_PLAN.json','PILOT_FREEZE.json','PANEL.npz','native_initial_q.npy','identity49_reference.npz']
        for f in relevant:(temp/f).symlink_to((root/f).resolve())
        (temp/'donors').symlink_to((root/'donors').resolve(),target_is_directory=True)
        plan=json.loads((root/'PILOT_PLAN.json').read_text())
        for name in plan['qualifications']:
            source=root/('qual_'+name);target=temp/source.name;target.mkdir()
            for f in source.iterdir():
                if f.is_file():(target/f.name).symlink_to(f.resolve())
        def replace(relative, value):
            f=temp/relative
            f.unlink()
            if isinstance(value,dict) and f.suffix=='.npz':np.savez_compressed(f,**value)
            else:f.write_text(json.dumps(value,indent=2)+'\n')
        def restore(relative):
            f=temp/relative;f.unlink();f.symlink_to((root/relative).resolve())
        for field in ['ORN_ids','ORN_q','q']:
            relative='qual_parent_wide/wide_observation.npz';bad=arrays(root/relative)
            if field=='ORN_ids':bad[field][0]+=1
            else:bad[field][0,0]=np.nan
            replace(relative,bad);reject('wide_'+field,lambda:qualify(temp));restore(relative)
        for mode in ['parent']:
            rels=['qual_'+mode+s+'/OWNERS_FINAL.json' for s in ['_narrow','_wide']]
            for rel in rels:replace(rel,{})
            reject('both_owner_maps_empty',lambda:qualify(temp))
            for rel in rels:restore(rel)
        relative='qual_parent_wide/neural_and_inputs.npz';bad=arrays(root/relative)
        bad['air_kinematics'][:,3:6]/=10
        replace(relative,bad);reject('old_velocity_unit_bug',lambda:qualify(temp));restore(relative)
        relative='qual_parent_wide/RESULT.json';bad=json.loads((root/relative).read_text());bad['setting']['odor']=False
        replace(relative,bad);reject('condition_flag',lambda:qualify(temp));restore(relative)
        relative='PILOT_PLAN.json';bad=copy.deepcopy(plan);bad['criteria']['raw_yaw_deg_s_min']=.002
        replace(relative,bad);reject('criterion_change',lambda:qualify(temp));restore(relative)
    qualmod.H=root
    need(len(rejected)==10,'Expected ten corruption cases')
    return rejected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=H)
    p.add_argument('--without-manifest',action='store_true');p.add_argument('--no-corruptions',action='store_true')
    a=p.parse_args();cpu=time.process_time()
    r=check(a.root,manifest=not a.without_manifest)
    if not a.no_corruptions:r['corruptions_rejected']=corruption_tests(a.root)
    r['CPU_s']=time.process_time()-cpu
    print(json.dumps(r))
