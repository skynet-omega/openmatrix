"""Freeze code plus scientific inputs before the paired trials."""
from pathlib import Path
import json
from source_inventory import sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')


def main():
    paths = set()
    for record in (HERE/'diagnostic_20ms_01/EXECUTED_SOURCES_SETUP.json',
                   HERE/'diagnostic_20ms_01/EXECUTED_SOURCES_FINAL.json',
                   ROOT/'campanas/etapa45_navigation_wind_20260925_40/SOURCE_LOCK.json'):
        paths.update(Path(p) for p in json.loads(record.read_text()))
    for folder in (HERE, HERE/'engine', HERE/'pn'):
        paths.update(p for p in folder.iterdir() if p.suffix in ('.py','.cu','.so'))
    stable = ROOT/'campanas/etapa45_navigation_wind_20260925_40'
    prefix = stable/'navigation_minus_filtered_wind_03'
    paths.update((HERE/'PLAN.json', prefix/'GAUSSIAN_SPEC.json', prefix/'traces.npz',
                  prefix/'prepared_state/MANIFEST.json',
                  OLD/'config/matrix_diagnostico_olfativo_v1.json',
                  OLD/'data/male_v10/nodes.parquet', OLD/'data/male_v10/provenance.json'))
    for folder in (prefix/'prepared_state',
                   OLD/'work/stage234_settling_extension_20260915/settled_700ms/core_carrier/brain'):
        paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.json','.npz','.py'))
    mapping = {str(p.resolve()):sha(p) for p in sorted(paths)}
    with (HERE/'SOURCES.json').open('x') as f:
        json.dump(mapping, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(files=len(mapping), source_lock_sha256=sha(HERE/'SOURCES.json'))))


if __name__ == '__main__':
    main()
