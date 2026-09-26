"""CPU replay from an extracted capsule; relocate paths, never numerical data.

The capsule root contains AXIOMA_ASTRA/ and AXIOMA_FLYWIRE/. Frozen payloads
remain byte-identical. Locator strings are mapped only in parsed JSON, allowing
the original frozen verifiers to run without access to either original tree.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    original = ('/home/daroch/AXIOMA_ASTRA', '/home/daroch/AXIOMA_FLYWIRE')
    destinations = (str(root/'AXIOMA_ASTRA'), str(root/'AXIOMA_FLYWIRE'))
    def forbid_original(event, values):
        if event == 'open' and values and isinstance(values[0], (str, bytes)):
            name = values[0].decode() if isinstance(values[0], bytes) else values[0]
            if any(name == prefix or name.startswith(prefix+'/') for prefix in original):
                raise PermissionError('Capsule verification attempted original-tree access: '+name)
    sys.addaudithook(forbid_original)
    manifest = json.loads((root/'CAPSULE.json').read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe capsule member')
        path = root/relative
        if not path.is_file() or path.is_symlink() or digest(path) != expected:
            raise ValueError('Capsule payload differs: '+name)
    def relocate(value):
        if isinstance(value, str):
            for prefix, target in zip(original, destinations):
                value = value.replace(prefix, target)
            return value
        if isinstance(value, list):
            return [relocate(x) for x in value]
        if isinstance(value, dict):
            out = {relocate(k):relocate(v) for k,v in value.items()}
            if len(out) != len(value):
                raise ValueError('Relocation collapsed distinct keys')
            return out
        return value
    here = root/'AXIOMA_ASTRA/motor_nuevo/full_pipeline_review_20260925_12'
    sys.path.insert(0, str(here))
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        reader = result.read_json
        result.read_json = lambda path:relocate(reader(path))
        return result
    verifier = module('capsule_frozen_verifier12', here/'verify_runs.py')
    runtime = module('capsule_runtime_supplement12', here/'reviews/math/check_runtime.py')
    runtime.FLYWIRE = root/'AXIOMA_FLYWIRE/matrix'
    checks = []
    verdicts = {}
    for ms in (100, 2000):
        stable = here/f'stable_{ms}ms_01'
        reviewed = here/f'reviewed_{ms}ms_01'
        actual = verifier.compare(stable, reviewed, ms)
        expected = relocate(json.loads((here/f'PAIR{ms}.json').read_text()))
        if actual != expected:
            keys = [k for k in set(actual)|set(expected) if actual.get(k) != expected.get(k)]
            raise ValueError('Recomputed comparison differs: '+str((ms,keys)))
        verdicts[str(ms)] = actual['status']
        actual_runtime = runtime.compare(stable, reviewed, ms)
        expected_runtime = relocate(json.loads((here/f'RUNTIME_PAIR{ms}.json').read_text()))
        if actual_runtime != expected_runtime:
            keys = [k for k in set(actual_runtime)|set(expected_runtime)
                    if actual_runtime.get(k) != expected_runtime.get(k)]
            raise ValueError('Recomputed runtime contract differs: '+str((ms,keys)))
        checks.extend([f'pair_{ms}ms_recomputed', f'runtime_{ms}ms_recomputed'])
    report = dict(status='PASS_EXTRACTED_SAVED_EVIDENCE', files_verified=len(manifest['files']),
                  checks=checks, functional_verdicts_reproduced=verdicts,
                  original_tree_access='blocked_by_audit_hook',
                  python_optimized=not __debug__,
                  scope='Saved data and frozen source verification only; no new neuronal/body execution')
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
