"""Read the two DM1 PN partner counts without loading the organism."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
SOURCE = Path('/home/daroch/AXIOMA_FLYWIRE/matrix/data/male_v10/node_connectivity.parquet')
IDS = (10176, 10208)


def main() -> None:
    table = pq.read_table(SOURCE, filters=[('bodyId','in',list(IDS))])
    rows = {int(row['bodyId']):row for row in table.to_pylist()}
    if set(rows) != set(IDS):
        raise ValueError('PN pair not unique in metadata')
    result = {
        'schema':'dm1_pn_pair_anatomical_counts_v1',
        'source':str(SOURCE),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'rows':{str(i):rows[i] for i in IDS},
        'scope':'Partner and synapse counts in saved connectome metadata, not effective functional input/output or evidence that adapters should be identical.',
    }
    (HERE/'PN_PAIR_STRUCTURE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
