"""Fast corruption checks for the real versioned checkpoint envelope."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import external_checkpoint
import static_lateral_checkpoint

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'run_10_versioned_driver/checkpoint_v1'


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    outcomes = {}
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch) / 'checkpoint'
        (folder / 'organism').mkdir(parents=True)
        for name in ('manifest.json', 'driver.json', 'organism/manifest.json'):
            shutil.copyfile(SOURCE / name, folder / name)
        manifest = json.loads((folder / 'manifest.json').read_text())
        driver = json.loads((folder / 'driver.json').read_text())
        model_manifest = (folder / 'organism/manifest.json').read_bytes()

        def denied(_):
            raise RuntimeError('Invalid envelope reached the model loader')

        def reject(name, expected):
            try:
                external_checkpoint.load(folder, denied,
                                         static_lateral_checkpoint._restore_before_validation)
            except ValueError as exc:
                if str(exc) != expected:
                    raise RuntimeError(name + ': wrong rejection ' + str(exc)) from exc
                outcomes[name] = str(exc)
            else:
                raise RuntimeError(name + ': corrupted checkpoint was accepted')

        (folder / 'driver.json').unlink()
        reject('missing_driver', 'Incomplete external checkpoint envelope')
        dump(folder / 'driver.json', driver)
        (folder / 'driver.json').write_bytes((folder / 'driver.json').read_bytes() + b' ')
        reject('changed_driver', 'Changed external driver state')
        dump(folder / 'driver.json', driver)
        (folder / 'organism/manifest.json').write_bytes(model_manifest + b' ')
        reject('changed_model_manifest', 'Changed model checkpoint manifest')
        (folder / 'organism/manifest.json').write_bytes(model_manifest)

        def signed_driver(value):
            dump(folder / 'driver.json', value)
            updated = dict(manifest)
            updated['driver_sha256'] = hashlib.sha256((folder / 'driver.json').read_bytes()).hexdigest()
            dump(folder / 'manifest.json', updated)

        altered = json.loads(json.dumps(driver))
        altered['state']['field_source_sha256'] = '0' * 64
        signed_driver(altered)
        reject('wrong_field_source', 'Static field implementation changed')
        altered = json.loads(json.dumps(driver))
        del altered['state']['field_metadata']['odor_axis']
        signed_driver(altered)
        reject('missing_field_parameter', 'Incomplete field metadata')
        altered = json.loads(json.dumps(driver))
        altered['schema'] = 'unknown_driver_v1'
        signed_driver(altered)
        reject('unknown_driver', 'Wrong static field checkpoint schema')

    class FakeModel:
        closes = 0

        def close(self):
            self.closes += 1

    fake_model = FakeModel()
    model_class = static_lateral_checkpoint.AntennalContactRuntime
    original_model_loader = model_class.__dict__['load']
    original_envelope_loader = external_checkpoint.load

    def fail_after_model_loaded(_, load_model, restore_driver):
        load_model(Path('/unused'))
        raise ValueError('Expected one external driver bind')

    try:
        model_class.load = classmethod(lambda cls, path: fake_model)
        external_checkpoint.load = fail_after_model_loaded
        try:
            static_lateral_checkpoint.load(Path('/unused'))
        except ValueError as exc:
            if str(exc) != 'Expected one external driver bind':
                raise RuntimeError('Wrong post-load failure') from exc
        else:
            raise RuntimeError('Post-load failure was accepted')
        if fake_model.closes != 1:
            raise RuntimeError('Loaded model was not closed exactly once')
        outcomes['postload_failure_closes_model'] = 'closed once'
    finally:
        model_class.load = original_model_loader
        external_checkpoint.load = original_envelope_loader
    (HERE / 'CHECKPOINT_GUARDS.json').write_text(json.dumps(outcomes, indent=2) + '\n')
    print(json.dumps({'passed': len(outcomes), 'guards': outcomes}))


if __name__ == '__main__':
    main()
