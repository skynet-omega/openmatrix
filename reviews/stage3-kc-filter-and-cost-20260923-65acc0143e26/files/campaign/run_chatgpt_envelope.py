"""Run ChatGPT's unchanged function on actual accepted SET event histories.

Only the source-filter function is called: captures lack full trial/delta/gate
records required by the complete comparator. Never fabricate those records.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import chatgpt_localizar_original as external
from filter_error_real import events,PARENT

HERE=Path(__file__).resolve().parent
def main():
    out=HERE/'chatgpt_envelope_01';out.mkdir(exist_ok=False)
    plan={'cases':4,'mode':'unchanged external port_envelope function',
          'scope':'unit-gain source filter, common initial s set0 because it cancels',
          'bound_comparison_fp64_slack':1e-11,'not_executed':'full capture/rollback/proposal comparator; missing per-trial records'}
    (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    z=np.load(HERE/'native_on_01/kc_native_0/input_00.npz',allow_pickle=False)
    ids=z['selected_ids'];aa=events(PARENT/'native_sham_20_01/EVENT_AUDIT.json',ids)
    bb=events(PARENT/'reference_sham_20_01/EVENT_AUDIT.json',ids)
    exact=json.loads((HERE/'filter_error_01/RESULT.json').read_text())['rows']
    rows=[];inputs=[]
    for k,body in enumerate(ids):
        def capture(ev):
            samples=[]
            for t,q in ev:
                tns=t*1e9;it=int(round(tns))
                if abs(tns-it)>1e-5:raise ValueError('Cannot coerce fractional-ns real event')
                samples.append({'t_ns':it,'event':True,'memory':{'q':q}})
            return {'meta':{'duration_ns':20000000,'neuron_id':int(body)},
                    'parameters':{'tau_q_s':float(z['tau'][k]),'tau_s_s':float(z['ts']),'gain':1.},
                    'initial':{'q':float(z['state_q'][k]),'s':0.},'samples':samples}
        a,b=capture(aa[int(body)]),capture(bb[int(body)])
        result=external.port_envelope(a,b);inputs.append({'a':a,'b':b})
        comparisons={name:result[field]-exact[k][name] for name,field in
          [('max_q_error','sup_abs_q'),('max_s_error','upper_sup_abs_s'),
           ('l1_q_s','upper_integral_abs_q_s'),('l1_s_s','upper_integral_abs_s_s')]}
        if min(comparisons.values()) < -plan['bound_comparison_fp64_slack']:
            raise ValueError('External envelope does not enclose independent analytic result')
        rows.append({'neuron_id':int(body),'external':result,'bound_minus_independent_exact':comparisons})
    (out/'INPUT.json').write_text(json.dumps(inputs,indent=2)+'\n')
    result={'plan':plan,'rows':rows,'all_bounds_cover_independent_CPP':True,
            'external_source_sha256':hashlib.sha256((HERE/'chatgpt_localizar_original.py').read_bytes()).hexdigest(),
            'stage3_admission':False}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'all_bounds_cover_independent_CPP':True,'cases':4,'rows':rows}))
if __name__=='__main__':main()
