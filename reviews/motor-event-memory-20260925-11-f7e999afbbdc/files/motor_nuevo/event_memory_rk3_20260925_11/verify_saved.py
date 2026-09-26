"""Recompute saved comparisons and exercise two material rejection cases."""
from pathlib import Path
import argparse,copy,json,tempfile
import numpy as np
from evaluate import evaluate
from compare_runs import need
from compare_stable import compare as stable_compare
from check_integrity import audit, check_trace, final_trees

HERE=Path(__file__).resolve().parent

def scientific(report):
    r=copy.deepcopy(report)
    # Absolute file locations change on extraction; payloads are hashed by the
    # package manifest. Numerical fields, timings and decisions must agree.
    r.pop('sources',None);r.pop('raw_hashes',None)
    return r

def verify(long=False):
    control=HERE/'control_100ms_01';candidate=HERE/'candidate_100ms_01'
    checks=[]
    r=evaluate(control,candidate,100,True)
    need(scientific(r)==scientific(json.loads((HERE/'PAIR100.json').read_text())),
         'Saved short-pair comparison differs')
    checks.append('short_pair_recomputed')
    r=stable_compare(candidate)
    need(scientific(r)==scientific(json.loads((HERE/'STABLE100.json').read_text())),
         'Saved stable100 comparison differs')
    checks.append('stable100_recomputed')
    with tempfile.TemporaryDirectory(prefix='event-memory-verify-') as directory:
        fake=Path(directory)
        for p in candidate.iterdir():
            if p.name not in ('RESULT.json','traces.npz'):(fake/p.name).symlink_to(p,target_is_directory=p.is_dir())
        result=json.loads((candidate/'RESULT.json').read_text())
        result['status']='INCOMPLETE'
        (fake/'RESULT.json').write_text(json.dumps(result))
        (fake/'traces.npz').symlink_to(candidate/'traces.npz')
        try:evaluate(control,fake,100,True)
        except ValueError as e:need('incomplete' in str(e).lower(),'Wrong incomplete-run failure')
        else:raise ValueError('Incomplete run accepted')
        checks.append('incomplete_run_rejected')
        result['status']='COMPLETE';(fake/'RESULT.json').write_text(json.dumps(result))
        (fake/'traces.npz').unlink()
        with np.load(candidate/'traces.npz') as z:arrays={k:z[k].copy() for k in z.files}
        arrays['command_forward_mm_s'][0]+=.01
        np.savez_compressed(fake/'traces.npz',**arrays)
        r=evaluate(control,fake,100,True)
        need(r['status']=='FAIL' and not r['gates']['same_forward_commands'],'Changed forward command accepted')
        checks.append('changed_forward_command_rejected')
        origin=int(arrays['CNS_time_ns'][0])-1_000_000
        for defect in ('truncated_column', 'stopped_PN_clock'):
            damaged={k:v.copy() for k,v in arrays.items()}
            if defect=='truncated_column':damaged['position_mm']=damaged['position_mm'][:-1]
            else:damaged['PN_time_ns'][:]=damaged['PN_time_ns'][0]
            np.savez_compressed(fake/'traces.npz',**damaged)
            try:check_trace(fake,100,origin)
            except ValueError:pass
            else:raise ValueError('Integrity audit accepted '+defect)
            checks.append(defect+'_rejected')
        try:final_trees(fake/'missing_final',100)
        except ValueError as e:need('Missing final' in str(e),'Wrong missing-final failure')
        else:raise ValueError('Missing final trees accepted')
        checks.append('missing_final_PN_rejected')
        stale=fake/'stale_final';(stale/'state_100ms').mkdir(parents=True)
        for name,stage in [('pn_state',100),('published',0)]:
            for suffix in ('.json','.npz'):
                (stale/'state_100ms'/f'{name}{suffix}').symlink_to(candidate/f'state_{stage}ms'/f'{name}{suffix}')
        try:final_trees(stale,100)
        except ValueError as e:need('final clock' in str(e),'Wrong stale-clock failure')
        else:raise ValueError('Initial snapshot passed as final state')
        checks.append('stale_final_publication_rejected')
    report=audit(long)
    expected='INTEGRITY1000.json' if long else 'INTEGRITY100.json'
    need(report==json.loads((HERE/expected).read_text()),'Saved integrity audit differs')
    checks.append('external_integrity_recomputed')
    if long:
        candidate=HERE/'candidate_1000ms_01'
        r=evaluate(HERE.parent/'equivalence_1s_20260925_09/optimized_1000ms_01',candidate,1000)
        need(scientific(r)==scientific(json.loads((HERE/'PARENT1000.json').read_text())),
             'Saved parent1000 comparison differs')
        r=stable_compare(candidate)
        need(scientific(r)==scientific(json.loads((HERE/'STABLE1000.json').read_text())),
             'Saved stable1000 comparison differs')
        checks.extend(['parent1000_recomputed','stable1000_recomputed'])
    return {'status':'PASS_SAVED_EVIDENCE','checks':checks,'python_optimized':not __debug__,
            'scope':'CPU reconstruction of recorded results; no new neuronal/body advancement.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--long',action='store_true');p.add_argument('--out',type=Path)
    a=p.parse_args();r=verify(a.long)
    if a.out:
        need(not a.out.exists(),'New verification output required')
        a.out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
