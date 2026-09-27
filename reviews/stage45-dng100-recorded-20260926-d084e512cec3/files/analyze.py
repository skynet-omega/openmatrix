"""Recalculate committed DNg100 observations; never integrate an organism."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,time
import numpy as np
HERE=Path(__file__).resolve().parent

def need(ok,message):
    if not ok:raise ValueError(message)

def summarize_arm(folder,allow_partial=False):
    totals={};covered=[];latest=None
    for block in sorted((folder/'blocks').glob('*ms')):
        with np.load(block/'dng100_observed.npz') as z:
            need(np.array_equal(z['ids'],[10045,10056]),'Observer identities')
            records=z['records'];off=z['offsets'];n=np.diff(off)
            need(off[0]==0 and off[-1]==len(records) and np.array_equal(n,z['trials']),'Epoch coverage')
            need(np.isfinite(records).all(),'Nonfinite observed record')
            epochs=np.repeat(np.arange(len(n)),n)
            ms=z['ms'][epochs];committed=z['committed'][epochs]
            accepted=records[:,138].astype(bool)
            need(np.array_equal(accepted,records[:,131]<=1) and np.all(records[:,132:134]==0),
                 'Decision reconstruction')
            for i in range(len(n)):
                sel=slice(off[i],off[i+1]);a=accepted[sel]
                need(int(a.sum())==int(z['accepted'][i]) and int((~a).sum())==int(z['rejected'][i]),'Epoch counts')
                r=records[sel];last=0.;state=r[0,134:136].copy()
                for trial,ok in zip(r,a):
                    need(trial[128]==last and np.array_equal(trial[134:136],state),'Trial continuity')
                    if ok:last=trial[130];state=trial[136:138].copy()
                need(last==int(z['duration_ns'][i])*1e-9,'Epoch time coverage')
            # A biological millisecond has eight discarded midpoint predictors
            # and eight retained125us epochs. Require the actual stored schedule.
            unique=np.unique(z['ms']);covered.extend(unique.tolist());latest=int(unique[-1])
            for m in unique:
                q=z['ms']==m
                need(q.sum()==16 and z['committed'][q].sum()==8,'Midpoint disposition coverage')
                need(np.all(z['duration_ns'][q & z['committed']]==125000),'Committed epoch duration')
                need(np.all(z['duration_ns'][q & ~z['committed']]==62500),'Predictor epoch duration')
            fields=records[:,:128].reshape(-1,4,2,16)
            for phase,mask in [('baseline',ms<=1000),('stimulus',ms>1000)]:
                use=mask & committed & accepted
                if not use.any():continue
                # Stages1-3 drive high; stage4 is separately recorded for error.
                f=fields[use,:3];h=records[use,129]
                weight=h[:,None]*np.array([2/9,1/3,4/9])[None,:]
                rec=totals.setdefault(phase,dict(duration_s=0.,accepted_trials=0,
                    min=np.full((2,14),np.inf),max=np.full((2,14),-np.inf),integral=np.zeros((2,14)),
                    positive_target_evaluations=np.zeros(2,np.int64),
                    positive_margin_evaluations=np.zeros(2,np.int64),
                    base_final_target_mismatches=0,positive_k4_evaluations=np.zeros(2,np.int64)))
                rec['duration_s']+=float(h.sum());rec['accepted_trials']+=int(use.sum())
                rec['min']=np.minimum(rec['min'],f[:,:,:,:14].min(axis=(0,1)))
                rec['max']=np.maximum(rec['max'],f[:,:,:,:14].max(axis=(0,1)))
                rec['integral']+=(f[:,:,:,:14]*weight[:,:,None,None]).sum(axis=(0,1))
                rec['positive_target_evaluations']+=(f[:,:,:,11]>0).sum(axis=(0,1))
                rec['positive_margin_evaluations']+=(f[:,:,:,10]>0).sum(axis=(0,1))
                rec['positive_k4_evaluations']+=(fields[use,3,:,11]>0).sum(axis=0)
                rec['base_final_target_mismatches']+=int(np.count_nonzero(f[:,:,:,7]!=f[:,:,:,11]))
    need(covered==list(range(1,(latest or 0)+1)),'Missing/duplicated observed milliseconds')
    if not allow_partial:need(latest==3000,'Incomplete3s acquisition')
    fields=('state','net','positive_aux','negative_aux','drive','theta','gain',
            'base_target','base_rate','tau','margin','final_target','final_rate','derivative')
    out={}
    for phase,r in totals.items():
        out[phase]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in r.items()
                    if k not in ('min','max','integral')}
        out[phase]['cells']=[dict(id=ident,**{name:dict(min=float(r['min'][i,j]),
            max=float(r['max'][i,j]),time_weighted_mean=float(r['integral'][i,j]/r['duration_s']))
            for j,name in enumerate(fields)}) for i,ident in enumerate([10045,10056])]
    return dict(completed_ms=latest,phases=out)

def main():
    started=time.monotonic();result={}
    for arm in ('sham','odor'):
        record=json.loads((HERE/arm/'RESULT.json').read_text())
        need(record['status']=='COMPLETE' and record['parent_observations_exact'],'Parent/control acquisition failed')
        result[arm]=summarize_arm(HERE/arm)
    zero=all(all(v==0 for v in d['positive_target_evaluations']) for a in result.values() for d in a['phases'].values())
    output=dict(status='COMPLETE',classification='COHERENT_ZERO_TARGET_IN_RECORDED_COMMITTED_EVOLUTION' if zero else 'POSITIVE_TARGET_REQUIRES_LOCALIZATION',
        arms=result,stage4_admission=False,stage5_admission=False,
        scope='Original model/context, retained successful steps. Does not establish physiological law or justify retuning.',
        wall_s=time.monotonic()-started)
    (HERE/'RESULTADOS.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'],classification=output['classification'],wall_s=output['wall_s'])),flush=True)

if __name__=='__main__':main()
