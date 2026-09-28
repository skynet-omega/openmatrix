"""Reconstruct frontier-trial checks from explicitly supplied arrays.

The acquisition workflow's summary is advisory here: every raw array is a
declared input. This readback has no simulator imports or implicit sibling data.
"""
from pathlib import Path
import json
import math
import hashlib
import os
# Acquisition and workbench readback fix both libraries to one thread. Preserve
# that reduction order in the portable CLI before NumPy loads its BLAS runtime.
if __name__ == '__main__':
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np


def require(ok, message):
    if not ok: raise ValueError(message)


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {k:z[k] for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_contract(path, freeze):
    receipt=json.loads(Path(freeze).read_text())
    require(sha(path)==receipt['contract_sha256'], 'frozen scientific criteria changed')
    return json.loads(Path(path).read_text())


def command_check(trace):
    current, used, baseline = [trace[k] for k in ('DN_q_actual','DN_q_usada','DN_baseline')]
    require(current.shape == used.shape == baseline.shape and used.shape[1] == 4, 'DN shape')
    require(np.array_equal(used[1:], current[:-1]), 'causal DN phase changed')
    delta = used-baseline
    yaw = np.tanh(250*(delta[:,2]-delta[:,3]))*math.radians(5)
    forward = np.clip(delta[:,:2].mean(1), 0, .5)
    require(np.array_equal(yaw, trace['command_yaw_rate_rad_s']), 'command/consumed DN disagreement')
    require(np.array_equal(forward, trace['command_forward_mm_s']), 'forward reader disagreement')


def writer_check(held, before, after, mode):
    shape=held.shape
    require(shape[1]==686 and np.all((held>=0)&(held<=1)) and np.isfinite(held).all(), 'held domain/support')
    require(set(before)==set(after)=={'first','last','lo','hi','counts'}, 'writer witness fields')
    require(all(v.shape==shape for s in (before,after) for v in s.values()), 'writer shape')
    require(np.array_equal(before['counts'],after['counts']) and np.all(after['counts']>0), 'writer coverage')
    for s in (before,after):
        require(s['counts'].dtype.kind in 'iu' and np.array_equal(s['counts'],np.broadcast_to(s['counts'][:,:1],shape)), 'unequal PN counts')
        for k in ('first','last','lo','hi'): require(np.isfinite(s[k]).all(), 'nonfinite writer')
        require(np.all(s['lo']<=s['first']) and np.all(s['first']<=s['hi']) and
                np.all(s['lo']<=s['last']) and np.all(s['last']<=s['hi']), 'witness extrema')
    if mode=='live':
        require(all(np.array_equal(before[k],after[k]) for k in before), 'identity changed values')
    else:
        require(all(np.array_equal(after[k],held) for k in ('first','last','lo','hi')), 'held value not consumed')


def evaluate(inputs, parameters, output):
    require(not parameters, 'use the declared contract')
    c=frozen_contract(inputs['contract'],inputs['freeze'])
    require(c['schema']=='matrix_pn_frontier_trial_v1', 'contract schema')
    expected={'source','contract','freeze','support','reference_L','reference_R','anchor','tape_L','tape_R'}
    expected.update(a+'_'+k for a in c['arms'] for k in ('trace','stimulus','before','after','panel','receipt'))
    require(set(inputs)==expected, 'all raw inputs must be explicit')
    refs={s:arrays(inputs['reference_'+s]) for s in ('L','R')}
    witnesses={s:arrays(inputs['tape_'+s]) for s in ('L','R')}
    tapes={s:witnesses[s]['first'] for s in ('L','R')}
    for key in ('support','reference_L','reference_R','tape_L','tape_R'):
        require(sha(inputs[key])==c['bindings'][key], 'frozen reference changed: '+key)
    support=arrays(inputs['support']);anchor=arrays(inputs['anchor'])
    w=slice(c['window_ms'][0]-1,c['window_ms'][1]);results={};curves={};initial=None;caps_reference=None
    dn_ids=[10045,10056,10118,10065]
    for arm,spec in c['arms'].items():
        n=spec['duration_ms'];side=spec['side'];mode=spec['mode']
        t=arrays(inputs[arm+'_trace']);s=arrays(inputs[arm+'_stimulus'])
        before=arrays(inputs[arm+'_before']);after=arrays(inputs[arm+'_after']);panel=arrays(inputs[arm+'_panel'])
        receipt=json.loads(Path(inputs[arm+'_receipt']).read_text())
        require(receipt['status']=='COMPLETE' and receipt['arm']==arm and receipt['new_CNS_ms']==n, 'incomplete acquisition')
        require(receipt['contract_sha256']==sha(inputs['contract']), 'acquisition contract mismatch')
        require(np.array_equal(s['PN_ids'],support['ids']) and np.array_equal(s['PN_rows'],support['rows']), 'PN support mismatch')
        require(s['caps'].shape==(686,) and np.isfinite(s['caps']).all() and np.all(s['caps']>0), 'caps domain')
        if caps_reference is None: caps_reference=s['caps'].copy()
        else: require(np.array_equal(s['caps'],caps_reference), 'caps changed between arms')
        require(s['initial_q'].shape==(166700,) and np.isfinite(s['initial_q']).all(), 'initial state shape/domain')
        require(len(np.unique(panel['ids']))==len(panel['ids']) and all(x in panel['ids'] for x in dn_ids), 'ambiguous/missing DN IDs')
        ix=[int(np.flatnonzero(panel['ids']==x)[0]) for x in dn_ids]
        require(np.array_equal(panel['q'][:,ix],t['DN_q_actual']), 'panel/trace DN order differs')
        for key in ('CNS_time_ns','PN_time_ns','body_time_ns'):
            require(np.array_equal(t[key],refs[side][key][:n]), 'absolute clock mismatch')
        require(np.array_equal(panel['time_ns'],t['CNS_time_ns']), 'panel clock')
        require(np.array_equal(t['DN_baseline'],np.broadcast_to(anchor['DN_baseline'],(n,4))), 'reader baseline changed')
        for key,value in t.items():
            if value.dtype.kind in 'fciu': require(np.isfinite(value).all(), 'nonfinite trace '+key)
        if initial is None: initial=s['initial_q'].copy()
        else: require(np.array_equal(initial,s['initial_q']), 'initial neural state changed')
        native=tapes[side][:n]
        require(np.array_equal(s['native_first'],native), 'native template altered')
        prescribed=native.copy()
        if mode in ('common','dose'):
            prescribed=((tapes['L'][:n].astype(float)+tapes['R'][:n])/2).astype(np.float32)
        if mode=='dose':
            m=prescribed.astype(float);target=native.astype(float);caps=s['caps']
            require(caps.shape==(686,) and np.isfinite(caps).all() and np.all(caps>0), 'caps identity/domain')
            current=m@caps;want=target@caps
            for j in range(n):
                if want[j]>current[j]: m[j]+=(want[j]-current[j])/((1-m[j])@caps)*(1-m[j])
                elif want[j]<current[j]: m[j]*=want[j]/current[j]
            prescribed=m.astype(np.float32)
        require(np.array_equal(s['held'],prescribed), 'prescribed stimulus disagrees with frozen extraction')
        command_check(t);writer_check(s['held'],before,after,mode)
        y=np.rad2deg(t['command_yaw_rate_rad_s']);native=np.rad2deg(refs[side]['command_yaw_rate_rad_s'][:n])
        d=t['DN_q_usada']-t['DN_baseline'];q=d[:,2]-d[:,3]
        rd=refs[side]['DN_q_usada'][:n]-refs[side]['DN_baseline'][:n];rq=rd[:,2]-rd[:,3]
        if mode=='live':
            require(all(np.array_equal(t[k],refs[side][k][:n]) for k in refs[side]), 'live reference differs')
            require(all(np.array_equal(after[k],witnesses[side][k][:n]) for k in after), 'live PN witness differs')
            qual=True
        else:
            qual=True
            if mode=='self':
                # Use the same frozen quantitative rules, recomputed from raw data.
                qual=(float(np.max(abs(q[w]-rq[w])))<=c['baseline_max_DNb_error_q'] and
                      float(np.max(abs(np.rad2deg(t['command_yaw_rate_rad_s'][w]-refs[side]['command_yaw_rate_rad_s'][w]))))<=c['baseline_max_yaw_error_deg_s'] and
                      np.array_equal(t['command_forward_mm_s'],refs[side]['command_forward_mm_s'][:n]))
        require(qual==receipt['gate_pass'], 'receipt fidelity flag disagrees with arrays')
        result={'qualified':bool(qual),'CNS_ms':n,'CPU_s':receipt['CPU_s'],'wall_s':receipt['wall_s']}
        if mode!='live':
            result.update(mean_yaw_deg_s=float(y[w].mean()),mean_consumed_DNb_delta_q=float(q[w].mean()),
                          mean_forward_mm_s=float(t['command_forward_mm_s'][w].mean()),
                          maximum_forward_mm_s=float(t['command_forward_mm_s'].max()))
            if mode=='self':
                result.update(maximum_window_yaw_error_deg_s=float(np.max(abs(np.rad2deg(t['command_yaw_rate_rad_s'][w]-refs[side]['command_yaw_rate_rad_s'][w])))),
                              maximum_window_DNb_error_q=float(np.max(abs(q[w]-rq[w]))))
            # Scalar quantity control is model caps*q, never physiological current.
            wanted=s['native_first'].astype(float)@s['caps'];delivered=s['held'].astype(float)@s['caps']
            scale=np.maximum(np.abs(wanted),np.finfo(float).tiny)
            result['max_relative_total_caps_q_error']=float(np.max(abs(delivered-wanted)/scale))
            if mode=='dose':require(result['max_relative_total_caps_q_error']<=2e-7, 'dose control not matched after FP32')
            curves[arm]=y;curves[arm+'_minus_native']=y-native
        results[arm]=result
    shifts={m:{s:results[m+'_'+s]['mean_yaw_deg_s']-results['self_'+s]['mean_yaw_deg_s']
               for s in ('L','R')} for m in ('common','dose')}
    halves={m:(results[m+'_L']['mean_yaw_deg_s']-results[m+'_R']['mean_yaw_deg_s'])/2
            for m in ('self','common','dose')}
    mediated={m:halves['self']-halves[m] for m in ('common','dose')}
    material=all(v['L']<=-c['minimum_each_shift_deg_s'] and v['R']>=c['minimum_each_shift_deg_s'] and
                 mediated[m]>=c['minimum_mediated_half_contrast_deg_s'] for m,v in shifts.items())
    all_valid=all(x['qualified'] for x in results.values())
    result={'schema':'matrix_pn_frontier_raw_readback_v1','raw_verified':True,'instrument_valid':all_valid,
            'arms':results,'shifts_deg_s':shifts,'half_L_minus_R_yaw_deg_s':halves,
            'mediated_half_contrast_deg_s':mediated,'material_partial_effect':bool(material and all_valid),
            'new_CNS_ms':0,'original_CNS_ms':sum(x['CNS_ms'] for x in results.values()),
            'original_CPU_s':sum(x['CPU_s'] for x in results.values()),
            'stage4_admission':False,'stage5_admission':False,'scope':c['scope'],
            'uncertainty':'Single exposed deterministic preparation. No sampling confidence interval or independent-organism claim.'}
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    (out/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'curves.npz',**curves)
    lines=['# PN: verificación desde arrays explícitos','',''+c['scope'],'',
           '| Brazo | Giro medio °/s | Avance medio mm/s |','|---|---:|---:|']
    for arm,r in results.items():
        if 'mean_yaw_deg_s' in r:lines.append(f"| {arm} | {r['mean_yaw_deg_s']:.9g} | {r['mean_forward_mm_s']:.9g} |")
    lines+=['',f'Instrumento cualificado: {all_valid}. Efecto parcial material según contrato: {material and all_valid}.',
            'CNS nuevo para esta lectura: 0 ms. La adquisición consumió '+str(result['original_CNS_ms'])+' ms.',
            'Etapas4/5 abiertas; no hay generalización ni validación fisiológica.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return {'metrics':{'raw_verified':1,'material_partial_effect':int(material and all_valid),'new_CNS_ms':0},'assessment':result}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Read declared PN arrays; never starts a simulation.')
    parser.add_argument('manifest',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();base=args.manifest.resolve().parent
    manifest=json.loads(args.manifest.read_text())
    require(manifest['schema']=='matrix_pn_frontier_readback_inputs_v1','input manifest schema')
    inputs={k:str((base/v).resolve()) for k,v in manifest['inputs'].items()}
    require(all(Path(p).is_relative_to(base) for p in inputs.values()), 'standalone inputs must stay within extracted package')
    require(not args.output.exists(),'preserve previous readback output')
    print(json.dumps(evaluate(inputs,{},args.output)['assessment'],indent=2,allow_nan=False))
