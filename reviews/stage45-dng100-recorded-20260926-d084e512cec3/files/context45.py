"""Read the pre-identified Sapkal walk/stop populations from archived45 outputs.

No population is selected by its response, and this is not a motor decoder.
Only saved100ms publications are available; never infer unrecorded transients.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
PARENT = HERE.parent / 'iniciacion_olfativa_20260926_45'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.process_time()
    source = OLD / 'work/walking_state_sources_20260914/identities.json'
    crosswalk = json.loads(source.read_text())
    nodes_path = OLD / 'data/male_v10/nodes.parquet'
    nodes = pd.read_parquet(nodes_path).sort_values('node_index')
    require(np.array_equal(nodes.node_index, np.arange(166700)), 'Dataset row order')
    ids = np.asarray(crosswalk['selected_ids'], np.int64)
    rows = np.searchsorted(nodes.bodyId.to_numpy(np.int64), ids)
    require(np.array_equal(nodes.bodyId.to_numpy(np.int64)[rows], ids), 'Missing identity')
    labels = {int(c['bodyId']): g['paper_label'] for g in crosswalk['groups']
              for c in g['canonical_rows']}
    inputs = {str(source): sha(source), str(nodes_path): sha(nodes_path)}
    arrays = {}
    records = []
    for arm in ('sham', 'odor'):
        folder = PARENT / arm
        result = json.loads((folder / 'RESULT.json').read_text())
        require(result['status'] == 'COMPLETE' and result['completed_ms'] == 4000,
                'Incomplete original45')
        initial = folder / 'initial_observation.npz'
        inputs[str(initial)] = sha(initial)
        with np.load(initial) as z:
            origin = int(z['time_ns'])
            series = [z['published_output'][rows].copy()]
        times = [0]
        for block in sorted((folder / 'blocks').glob('*ms')):
            path = block / 'published.npz'
            digest = sha(path)
            manifest = json.loads((block / 'MANIFEST.json').read_text())
            require(digest == manifest['hashes'][path.name], 'Changed publication')
            inputs[str(path)] = digest
            with np.load(path) as z:
                require(z['output'].shape == (166700,), 'Publication shape')
                times.append((int(z['time_ns']) - origin) // 1000000)
                series.append(z['output'][rows].copy())
        require(times == list(range(0, 4001, 100)), 'Publication coverage')
        values = np.stack(series)
        require(np.isfinite(values).all(), 'Nonfinite publication')
        arrays[arm] = values
        for column, identity in enumerate(ids):
            n = nodes.iloc[int(rows[column])]
            records.append(dict(arm=arm, id=int(identity), paper_label=labels[int(identity)],
                type=n['type'], side=n['somaSide'], initial=float(values[0, column]),
                sampled_min=float(values[:, column].min()),
                sampled_max=float(values[:, column].max()),
                at_2s=float(values[20, column]), at_3s=float(values[30, column])))
    require(np.array_equal(arrays['odor'][:11], arrays['sham'][:11]), 'Pre-odor difference')
    np.savez_compressed(HERE / 'CONTEXT45.npz', ms=np.asarray(times), ids=ids, **arrays)
    with (HERE / 'CONTEXT45.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    receipt = dict(status='COMPLETE', source_campaign=45, observations=41,
        time_spacing_ms=100, CNS_steps=0, body_steps=0,
        cells_selected_from='Sapkal2024 crosswalk prepared14September, before45 outcomes',
        units='Published cap*q, nominal model output. Not measured Hz; not q itself.',
        limitation='Endpoint publications cannot exclude intervening transients, identify received currents, '
                   'or prove causal mediation. No new reader or physiological parameter inferred.',
        inputs=inputs, script_sha256=sha(Path(__file__)),
        CPU_s=time.process_time()-started)
    (HERE / 'CONTEXT45.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k != 'inputs'}))


if __name__ == '__main__':
    main()
