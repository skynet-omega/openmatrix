"""The conserved organism declares its effective-weight writers."""
import cupy as cp
import numpy as np


def writer_layout(brain):
    arrays = []
    for name in ('_general_positions', '_apl_gpu_positions'):
        value = getattr(brain, name, None)
        arrays.append(None if value is None else (name, value.data.ptr, value.shape, str(value.dtype)))
    return (type(brain), tuple(arrays), getattr(brain, '_parallel_position', None))


def declared_positions(brain):
    groups = {}
    for name in ('_general_positions', '_apl_gpu_positions'):
        if hasattr(brain, name):
            groups[name] = cp.asnumpy(getattr(brain, name)).astype(np.int64, copy=False).reshape(-1)
    if hasattr(brain, '_parallel_position'):
        groups['_parallel_position'] = np.asarray([brain._parallel_position], dtype=np.int64)
    if not groups:
        raise ValueError('This compatibility model must declare its dynamic weight owners')
    positions = np.unique(np.concatenate(list(groups.values())))
    return positions, {name: int(value.size) for name,value in groups.items()}
