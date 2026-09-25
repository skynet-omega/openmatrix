"""Descriptive check of saved effective coefficients, not a new model or gain search."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();start=time.perf_counter()
    need(not args.out.exists(),'Unique output');args.out.mkdir(parents=True)
    root=Path(__file__).resolve().parent
    historical=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
    saved=root.parent/'etapa4_contrast_ablation_20260925_36/identity_01/final_state'
    names=dict(populations=root/'population_01/RESULT.json', effective_metadata=saved/'effective_operator.json',
               effective_arrays=saved/'effective_operator.npz',checkpoint_manifest=saved/'MANIFEST.json',
               volumes=historical/'data/anatomical_morphometry/v0.1/canonical_morphometry.parquet',
               volumes_manifest=historical/'data/anatomical_morphometry/v0.1/manifest.json',
               prior_audit=historical/'work/stage3_progression_20260916/motor_size_audit.md',
               withdrawal=historical/'config/motor_size_withdrawal_v1.json',
               formula=historical/'src/anatomical_morphometry.py')
    provenance={k:dict(path=str(v),sha256=hashlib.sha256(v.read_bytes()).hexdigest()) for k,v in names.items()}
    meta=json.loads(names['effective_metadata'].read_text());manifest=json.loads(names['checkpoint_manifest'].read_text())
    need(provenance['effective_arrays']['sha256']==manifest['files']['effective_operator.npz']['sha256'],'Checkpoint coefficient identity')
    vm=json.loads(names['volumes_manifest'].read_text())
    need(provenance['volumes']['sha256']==vm['artifact']['sha256'],'Volume source identity')
    table=pd.read_parquet(names['volumes'])
    vol=table.set_index('bodyId')['size_source_voxels']
    median=float(np.median(vol[np.isfinite(vol)]))
    need(median==vm['normalization_median_source_voxels'],'Reference changed')
    panel=json.loads(names['populations'].read_text())['rows']
    arrays={}
    with np.load(names['effective_arrays'],allow_pickle=False) as z:
        for key in ('theta','gain'):
            arrays[key]=z[meta['values'][key]['__array__']]
            cuda=z[meta['values']['cuda_'+key]['__array__']]
            need(np.array_equal(arrays[key],cuda),'Host/CUDA coefficient mismatch')
    rows=[]
    for row in panel:
        ix=row['row_index'];nid=row['bodyId']
        rows.append(dict(bodyId=nid,type=row['type'],instance=row['instance'],
                         volume_source_voxels=float(vol.loc[nid]),relative_to_whole_cns_median=float(vol.loc[nid]/median),
                         effective_theta=float(arrays['theta'][ix]),effective_gain_q=float(arrays['gain'][ix]),
                         prepared_model_hz=row['prepared_model_hz'],final_identity_model_hz=row['identity_final_model_hz']))
    d={x['bodyId']:x for x in rows}
    need(d[10360]['effective_theta']==641.8660278320312 and d[523769]['effective_theta']==547.9972534179688,'DNa02 values differ from historical finding')
    out=dict(scope='Descriptive endpoint coefficients; volume is not capacitance and saved coefficients are not fitted physiology',
             budget=dict(wall_s=60,max_rss_gib=2,new_organism_runs=0),sources=provenance,
             global_volume_median=median,front_cohort_median=998603428.5,
             front_to_global_reference_ratio=998603428.5/median,host_cuda_coefficients_exact=True,rows=rows,
             conclusion='DNa02 effective thresholds match the already documented global volume intervention. Reapplying the withdrawn frontal normalization is not a repair.',
             no_parameter_change=True,no_stage_admission=True)
    (args.out/'RESULT.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    with (args.out/'PARAMETERS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    runtime=dict(wall_s=time.perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    need(runtime['wall_s']<60 and runtime['peak_rss_gib']<2,'Budget exceeded')
    (args.out/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(json.dumps(dict(median=median,ratio=out['front_to_global_reference_ratio'],DNa02=[d[10360],d[523769]],runtime=runtime)))


if __name__=='__main__':
    main()
