"""Legacy organism adapter for the versioned external-driver checkpoint.

The generic envelope stays stimulus-agnostic. The scoped bind hook is needed
only because the archived AntennalContactRuntime loader has no driver callback.
It is removed on both success and failure; future model loaders can pass an
ordinary restoration callback to the same envelope.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np

import external_checkpoint

OLD = Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0] = [str(OLD / 'src'),
                str(OLD / 'work/stage4_antennal_contact_adapter_20260915'),
                str(OLD / 'work/stage3_static_lateral_field_20260916')]
from antennal_runtime import AntennalContactRuntime
from antennal_world import AntennalWorld
from static_field import StaticLateralField

SCHEMA = 'axioma_static_lateral_field_v1'
FIELD_SOURCE = Path(sys.modules[StaticLateralField.__module__].__file__).resolve()
_BIND_LOCK = threading.Lock()
_METADATA_KEYS = {'arm', 'installed_ns', 'center_mm', 'odor_axis',
                  'first_ON_after_install_ms', 'live_geometry',
                  'stationary_world_field', 'calibrated_optogenetic_drive'}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _export(obj):
    world = obj.core.world
    field = world.boundary
    if type(field) is not StaticLateralField or field.world is not world:
        raise ValueError('Expected one bound StaticLateralField')
    return {'schema': SCHEMA, 'state': {
        'world_time_ns': int(world.time_ns),
        'pending_sensors_sha256': _sha(np.ascontiguousarray(obj.core.pending_sensors).tobytes()),
        'field_source_sha256': _sha(FIELD_SOURCE.read_bytes()),
        'field_metadata': field.metadata(),
    }}


def _validate(driver):
    if set(driver) != {'schema', 'state'} or driver['schema'] != SCHEMA:
        raise ValueError('Wrong static field checkpoint schema')
    state = driver['state']
    if set(state) != {'world_time_ns', 'pending_sensors_sha256',
                      'field_source_sha256', 'field_metadata'}:
        raise ValueError('Incomplete static field state')
    if type(state['world_time_ns']) is not int or state['world_time_ns'] < 0:
        raise ValueError('Invalid field clock')
    if state['field_source_sha256'] != _sha(FIELD_SOURCE.read_bytes()):
        raise ValueError('Static field implementation changed')
    meta = state['field_metadata']
    if not isinstance(meta, dict) or set(meta) != _METADATA_KEYS:
        raise ValueError('Incomplete field metadata')
    if meta['arm'] not in ('odor_left', 'odor_right', 'uniform', 'sham'):
        raise ValueError('Unknown lateral field arm')
    if (type(meta['installed_ns']) is not int or
            not 0 <= meta['installed_ns'] <= state['world_time_ns']):
        raise ValueError('Invalid field installation clock')
    center = np.asarray(meta['center_mm'], dtype=np.float64)
    axis = np.asarray(meta['odor_axis'], dtype=np.float64)
    onset = meta['first_ON_after_install_ms']
    if (center.shape != (2,) or axis.shape != (2,) or
            not np.isfinite(center).all() or not np.isfinite(axis).all() or
            abs(np.linalg.norm(axis) - 1.) > 1e-12 or
            not isinstance(onset, (int, float)) or not np.isfinite(onset)):
        raise ValueError('Invalid field geometry or onset')
    if (meta['live_geometry'] is not True or
            meta['stationary_world_field'] is not True or
            meta['calibrated_optogenetic_drive'] is not False):
        raise ValueError('Unexpected lateral field semantics')
    for key in ('pending_sensors_sha256', 'field_source_sha256'):
        value = state[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid driver digest')
    return state, meta, center, axis


@contextmanager
def _restore_before_validation(driver):
    state, meta, center, axis = _validate(driver)
    with _BIND_LOCK:
        original = AntennalWorld.bind
        owner_thread = threading.get_ident()
        calls = 0

        def bind_with_driver(world, body):
            nonlocal calls
            if threading.get_ident() != owner_thread:
                return original(world, body)
            original(world, body)
            if int(world.time_ns) != state['world_time_ns']:
                raise ValueError('External driver/world clock mismatch')
            field = StaticLateralField.__new__(StaticLateralField)
            field.base, field.world, field.arm = world.boundary, world, meta['arm']
            field.center, field.axis = center.copy(), axis.copy()
            field.origin_ns = meta['installed_ns']
            field.onset_ms = float(meta['first_ON_after_install_ms'])
            if field.metadata() != meta:
                raise ValueError('External field differs after reconstruction')
            world.boundary = field
            calls += 1

        AntennalWorld.bind = bind_with_driver
        try:
            yield
            if calls != 1:
                raise ValueError('Expected one external driver bind')
        finally:
            AntennalWorld.bind = original


def save(obj, path):
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    return external_checkpoint.save(obj, path, _export)


def load(path):
    obj = external_checkpoint.load(path, AntennalContactRuntime.load,
                                   _restore_before_validation)
    try:
        state, meta, _, _ = _validate(json.loads((Path(path) / 'driver.json').read_text()))
        if (_sha(np.ascontiguousarray(obj.core.pending_sensors).tobytes()) !=
                state['pending_sensors_sha256'] or
                obj.core.world.boundary.metadata() != meta):
            raise ValueError('Restored external driver disagrees with pending state')
        return obj
    except BaseException:
        obj.close()
        raise
