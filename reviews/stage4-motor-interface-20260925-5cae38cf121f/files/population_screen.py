"""Read preselected anatomical populations from fixed published checkpoint arrays."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import pandas as pd


def need(ok, text):
    if not ok:
        raise ValueError(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); start = time.perf_counter()
    root = Path(__file__).resolve().parent
    plan = json.loads((root/'POPULATION_PLAN.json').read_text())
    need(not args.out.exists(), 'Unique output required')
    args.out.mkdir(parents=True)
    paths = {}
    for name, spec in plan['inputs'].items():
        path = Path(spec['path'])
        need(hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256'], 'Source identity: '+name)
        paths[name] = path
    table = pd.read_parquet(paths['nodes'], columns=['bodyId','type','instance','somaSide','rootSide','node_index','matrix_node_index'])
    ids = table.bodyId.to_numpy()
    need(len(ids) == 166700 and np.all(np.diff(ids) > 0), 'Canonical sorted ID mismatch')
    rates = {}; clocks = {}
    for arm in ('identity_01','no_contrast_01'):
        for state in ('prepared_state','final_state'):
            key = arm+'/'+state
            meta = json.loads(paths[key+'/published.json'].read_text())
            manifest = json.loads(paths[key+'/MANIFEST.json'].read_text())
            need(manifest['files']['published.npz']['sha256'] == plan['inputs'][key+'/published.npz']['sha256'], 'Checkpoint manifest mismatch')
            with np.load(paths[key+'/published.npz'], allow_pickle=False) as z:
                value = z[meta['rates']['__array__']]
                need(value.shape == (len(ids),) and value.dtype == np.float32 and np.isfinite(value).all(), 'Invalid published rate vector')
                rates[key] = value.copy()
            clocks[key] = meta['time_ns']
    need(np.array_equal(rates['identity_01/prepared_state'],rates['no_contrast_01/prepared_state']), 'Unmatched published preparation')
    need(clocks['identity_01/final_state'] == clocks['no_contrast_01/final_state'], 'Unmatched final clocks')
    need(clocks['identity_01/final_state']-clocks['identity_01/prepared_state'] == 1000000000, 'Not one second')
    selected = table['type'].fillna('').isin(plan['types']).to_numpy()
    rows=[]; missing=[]
    for typ in plan['types']:
        idx=np.flatnonzero(table['type'].fillna('').to_numpy() == typ)
        if not len(idx):
            missing.append(typ)
        for ix in idx:
            original=float(rates['identity_01/final_state'][ix]); equal=float(rates['no_contrast_01/final_state'][ix])
            row={k:(None if pd.isna(v) else v) for k,v in table.iloc[ix].to_dict().items()}
            row.update(row_index=int(ix), prepared_model_hz=float(rates['identity_01/prepared_state'][ix]),
                       identity_final_model_hz=original, equal_final_model_hz=equal,
                       identity_minus_equal_model_hz=original-equal,
                       used_by_actuator=int(ids[ix]) in (10045,10056,10118,10065))
            rows.append(row)
    result=dict(rows=rows,missing_types=missing,selected_cells=len(rows),
                prepared_published_arrays_exact=True,clocks_ns=clocks,
                units='Published float32 model Hz = release q * model r_max; not physiological recordings',
                scope='Prepared/final endpoints only; no inference of within-trial silence or causality',
                source_plan_sha256=hashlib.sha256((root/'POPULATION_PLAN.json').read_bytes()).hexdigest(),
                stage4_admitted=False,stage5_admitted=False,new_organism_runs=0)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (args.out/'POPULATIONS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(args.out/'published_population_subset.npz',body_ids=ids[selected],
                        **{k.replace('/','__'):v[selected] for k,v in rates.items()})
    runtime=dict(wall_s=time.perf_counter()-start,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    need(runtime['wall_s'] < plan['budget']['cpu_wall_s'] and runtime['peak_rss_gib'] < plan['budget']['max_rss_gib'], 'Resource budget exceeded')
    (args.out/'RUNTIME.json').write_text(json.dumps(runtime,indent=2)+'\n')
    print(json.dumps(dict(rows=rows,missing=missing,runtime=runtime),allow_nan=False))


if __name__ == '__main__':
    main()
