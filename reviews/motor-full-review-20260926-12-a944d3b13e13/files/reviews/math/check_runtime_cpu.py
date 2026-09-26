"""Finite CPU corruption checks using the already recorded 20ms diagnostic.

No simulation, engine import, CUDA call, shared-library load or source mutation.
The diagnostic is not retroactively admitted as a 100/2000ms comparison.
"""
from pathlib import Path
import copy
import json
import unittest

import check_runtime as verifier


class RuntimeContractFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = verifier.HERE / 'diagnostic_20ms_01'
        cls.result = verifier.read_json(cls.folder / 'RESULT.json')
        cls.events = verifier.read_json(cls.folder / 'EVENTS.json')
        cls.lock, cls.checked = verifier.frozen_sources()

    def check(self, result, events=None, engine='reviewed'):
        return verifier.validate_runtime(result, self.events if events is None else events,
                                         engine, 20, self.lock)

    def test_recorded_diagnostic_runtime_contract(self):
        report = self.check(self.result)
        self.assertEqual(report['epochs'], 320)
        self.assertEqual(report['event_records_including_discarded_predictors'],
                         sum(len(block['events']) for block in self.events))

    def test_method_and_implementation_corruptions(self):
        wrong_method = copy.deepcopy(self.result)
        wrong_method['runtime']['CNS']['method'] = 'exponential_midpoint'
        with self.assertRaisesRegex(ValueError, 'Wrong reviewed CNS method'):
            self.check(wrong_method)
        wrong_owner = copy.deepcopy(self.result)
        module, path = verifier.implementations('stable')['CNS']
        wrong_owner['runtime']['installed_implementations']['CNS'] = {
            'module': module, 'source': str(path), 'sha256': self.lock[str(path)]}
        with self.assertRaisesRegex(ValueError, 'Wrong implementation source: CNS'):
            self.check(wrong_owner)

    def test_runtime_flags_coupling_and_count_corruptions(self):
        changes = [
            (('event_boundaries',), False, 'Event boundaries disabled'),
            (('coupling_ns',), 62500, 'Runtime coupling changed'),
            (('partition', 'predictor_restore_exact'), False, 'Predictor restoration not exact'),
            (('CNS', 'rhs_evaluations'), 0, 'Wrong RK3'),
            (('events', 'events'), 0, 'Event runtime total differs'),
        ]
        for path, value, message in changes:
            with self.subTest(path=path):
                bad = copy.deepcopy(self.result)
                target = bad['runtime']
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.check(bad)

    def test_binary_path_and_overlay_corruptions(self):
        for key, value, message in (
            ('engine_path', str(verifier.HERE.parent / 'event_memory_rk3_20260925_11/engine'), 'Wrong reviewed engine_path'),
            ('engine_library_sha256', '0' * 64, 'Wrong reviewed library hash'),
        ):
            with self.subTest(key=key):
                bad = copy.deepcopy(self.result)
                bad[key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.check(bad)
        bad = copy.deepcopy(self.result)
        bad['pn_overlay']['resident_cyclic']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Wrong implementation hash: PN overlay'):
            self.check(bad)

    def test_stable_source_contract_fixture(self):
        # Synthetic metadata branch, not a fabricated stable simulation.
        stable = copy.deepcopy(self.result)
        stable.pop('engine_path')
        stable.pop('engine_library_sha256')
        stable['runtime']['CNS'].pop('method')
        stable['runtime']['CNS'].pop('rhs_evaluations')
        module, path = verifier.implementations('stable')['CNS']
        stable['runtime']['installed_implementations']['CNS'] = {
            'module': module, 'source': str(path), 'sha256': self.lock[str(path)]}
        self.check(stable, engine='stable')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeContractFixtures)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'PASS' if outcome.wasSuccessful() else 'FAIL',
              'tests': outcome.testsRun, 'failures': len(outcome.failures), 'errors': len(outcome.errors),
              'supplement_sha256': verifier.sha(verifier.__file__), 'fixture_sha256': verifier.sha(__file__),
              'frozen_sources_sha256': verifier.FROZEN_SOURCES_SHA256,
              'recorded_fixture_result_sha256': verifier.sha(verifier.HERE / 'diagnostic_20ms_01/RESULT.json'),
              'scope': 'CPU only; recorded diagnostic plus deliberate metadata corruptions. No simulation or GPU call.'}
    output = Path(__file__).with_name('RUNTIME_CONTRACT_CPU.json')
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
