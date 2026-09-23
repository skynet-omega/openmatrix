"""Small, pickle-free state codec for MATRIX organism API boundaries."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_state(path, state):
    path = Path(path)
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object arrays cannot be persisted')
            key = f'array_{len(arrays)}'
            arrays[key] = value
            return {'__array__': key}
        if isinstance(value, np.generic):
            return encode(value.item())
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise ValueError('State dictionaries require string keys')
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if value is None or type(value) in (int, float, str, bool):
            return value
        raise TypeError(f'Unrecognized state type: {type(value)}')

    descriptor = encode(state)
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    path.with_suffix('.json').write_text(json.dumps(descriptor, indent=2, allow_nan=False))


def read_state(path):
    path = Path(path)
    descriptor = json.loads(path.with_suffix('.json').read_text())
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'__array__'}:
                    return arrays[value['__array__']].copy()
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        return decode(descriptor)
