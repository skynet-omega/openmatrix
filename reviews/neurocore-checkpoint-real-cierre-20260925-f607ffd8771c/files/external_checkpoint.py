"""Versioned envelope for a model checkpoint and its external driver state.

The engine owns integrity and ordering. The model supplies its own state codec,
loader, and pre-validation restoration hook; no stimulus type enters this core.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable

SCHEMA = 'axioma_external_checkpoint_v1'


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def _finish(stage: Path, target: Path, driver: dict[str, Any]) -> None:
    if set(driver) != {'schema', 'state'} or not isinstance(driver['schema'], str):
        raise ValueError('External driver requires a named version and state')
    organism_manifest = stage / 'organism/manifest.json'
    if not organism_manifest.is_file():
        raise ValueError('Model checkpoint has no manifest')
    _write_json(stage / 'driver.json', driver)
    _write_json(stage / 'manifest.json', {
        'schema': SCHEMA,
        'model_manifest_sha256': _digest(organism_manifest),
        'driver_sha256': _digest(stage / 'driver.json'),
    })
    stage.rename(target)


def save(model: Any, target: Path, export_driver: Callable[[Any], dict[str, Any]]) -> Path:
    """Commit model and external state together, without modifying model files."""
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '-', dir=target.parent))
    try:
        driver = export_driver(model)
        model.save(stage / 'organism')
        if driver != export_driver(model):
            raise ValueError('Driver changed while model checkpoint was saved')
        _finish(stage, target, driver)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return target


def load(target: Path, load_model: Callable[[Path], Any],
         restore_driver: Callable[[dict[str, Any]], AbstractContextManager]) -> Any:
    """Restore driver during model binding, before its pending-input checks."""
    target = Path(target).resolve()
    manifest = json.loads((target / 'manifest.json').read_text())
    if set(manifest) != {'schema', 'model_manifest_sha256', 'driver_sha256'} or manifest['schema'] != SCHEMA:
        raise ValueError('Wrong external checkpoint envelope')
    if set(p.name for p in target.iterdir()) != {'manifest.json', 'driver.json', 'organism'}:
        raise ValueError('Incomplete external checkpoint envelope')
    driver_bytes = (target / 'driver.json').read_bytes()
    if hashlib.sha256(driver_bytes).hexdigest() != manifest['driver_sha256']:
        raise ValueError('Changed external driver state')
    if _digest(target / 'organism/manifest.json') != manifest['model_manifest_sha256']:
        raise ValueError('Changed model checkpoint manifest')
    driver = json.loads(driver_bytes.decode('utf-8'))
    if set(driver) != {'schema', 'state'}:
        raise ValueError('Incomplete external driver state')
    with restore_driver(driver):
        return load_model(target / 'organism')
