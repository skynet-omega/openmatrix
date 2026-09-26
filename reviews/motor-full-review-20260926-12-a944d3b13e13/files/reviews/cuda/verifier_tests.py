"""Small CPU-only rejection fixtures for the frozen review12 verifier.

Run with the same NumPy environment as the verifier. No CuPy, MuJoCo,
organism or engine imports are permitted or needed.
"""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import verify_runs as verifier


def traces(ms=100, origin=44_486_000_000):
    result = {}
    for key, shape in verifier.TRACE_SHAPES.items():
        kind = 'U6' if key == 'fase' else 'bool' if key == 'contact_active' else (
            'int64' if key in ('paso', 'CNS_time_ns', 'PN_time_ns', 'body_time_ns') else 'float64')
        result[key] = np.zeros((ms,) + shape, dtype=kind)
    result['fase'][:] = 'ensayo'
    result['paso'][:] = np.arange(1, ms + 1)
    for key in ('CNS_time_ns', 'PN_time_ns', 'body_time_ns'):
        result[key][:] = origin + np.arange(1, ms + 1) * verifier.NS_PER_MS
    return result


def wind_fixture(folder, ms=2000, origin=44_486_000_000):
    constants = verifier.protocol(verifier.read_json(HERE/'PLAN.json'))
    torque = constants['WIND_TORQUE_NATIVE']
    count = 800 if ms == 2000 else 0
    motor = dict(trial_steps=ms, body_calls=40*ms, wind_substeps=count,
                 expected_wind_substeps=800, nonzero_substeps=count,
                 wind_first_step=1001, wind_last_step=1020,
                 wind_torque_native=torque, wind_torque_Nm=torque*1e-7,
                 tau_s=constants['TAU_S'], threshold_rad_s=constants['THRESHOLD_RAD_S'],
                 output_rad_s=verifier.math.radians(5.),
                 mapping='current_pose_scratch_kinematics_comPos',
                 physical_restart_during_trial=False,
                 wind_generalized_peak_native=abs(torque) if count else 0.)
    (folder/'MOTOR.json').write_text(json.dumps(motor))
    trace = traces(ms, origin)
    trace['wind_torque_native'][1000:1020] = torque
    if count:
        calls = np.arange(40001, 40801, dtype=np.int64)
        generalized = np.zeros((800, 108), dtype=np.float64)
        generalized[:, 5] = torque
        data = dict(trial_step=np.repeat(np.arange(1001, 1021), 40), body_call=calls,
                    body_time_s=(origin+(calls-1)*verifier.BODY_NS)*1e-9,
                    world_torque=np.tile([0., 0., torque], (800, 1)),
                    generalized=generalized, qpos=np.zeros((800, 109), dtype=np.float64))
        np.savez_compressed(folder/'wind_substeps.npz', **data)
    return trace


class VerifierFixtures(unittest.TestCase):
    def test_complete_trace_and_truncation_rejection(self):
        good = traces()
        verifier.validate_traces(good, 100, 44_486_000_000, 'fixture')
        bad = dict(good, qpos=good['qpos'][:-1])
        with self.assertRaisesRegex(ValueError, 'trace length/layout qpos'):
            verifier.validate_traces(bad, 100, 44_486_000_000, 'fixture')

    def test_stale_trace_clock_rejection(self):
        bad = traces()
        bad['body_time_ns'][-1] -= verifier.NS_PER_MS
        with self.assertRaisesRegex(ValueError, 'stale/wrong clock body_time_ns'):
            verifier.validate_traces(bad, 100, 44_486_000_000, 'fixture')

    def test_pn_offsets_preserved_and_stale_local_clock_rejected(self):
        initial = {path: 864_100_000 if '/pn/' in path else 44_486_000_000
                   for path in verifier.PN_CLOCKS}
        final = {path: value+100*verifier.NS_PER_MS for path, value in initial.items()}
        verifier.validate_clock_progress(initial, final, 100*verifier.NS_PER_MS,
                                         verifier.PN_CLOCKS, 'fixture')
        final['/base/base/base/base/base/pn/pn/time_ns'] -= verifier.NS_PER_MS
        with self.assertRaisesRegex(ValueError, 'stale local state clock'):
            verifier.validate_clock_progress(initial, final, 100*verifier.NS_PER_MS,
                                             verifier.PN_CLOCKS, 'fixture')

    def test_wind_100ms_zero_and_2000ms_complete(self):
        for ms in (100, 2000):
            with self.subTest(ms=ms), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                trace = wind_fixture(folder, ms)
                audit, _ = verifier.validate_wind(folder, trace, ms, 44_486_000_000)
                self.assertEqual(audit['count'], 0 if ms == 100 else 800)

    def test_wrong_world_wind_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            trace = wind_fixture(folder)
            data = verifier.load_arrays(folder/'wind_substeps.npz')
            data['world_torque'][0, 2] *= -1
            np.savez_compressed(folder/'wind_substeps.npz', **data)
            with self.assertRaisesRegex(ValueError, 'Wrong literal world torque'):
                verifier.validate_wind(folder, trace, 2000, 44_486_000_000)

    def test_zero_mapped_wind_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            trace = wind_fixture(folder)
            data = verifier.load_arrays(folder/'wind_substeps.npz')
            data['generalized'][37] = 0.
            np.savez_compressed(folder/'wind_substeps.npz', **data)
            with self.assertRaisesRegex(ValueError, 'Wind mapped to zero'):
                verifier.validate_wind(folder, trace, 2000, 44_486_000_000)

    def test_missing_final_state_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for directory in ('final_state', 'state_100ms'):
                (folder/directory).mkdir()
                for stem in ('pn_state', 'published'):
                    for suffix in ('.json', '.npz'):
                        (folder/directory/(stem+suffix)).touch()
            verifier.require_final_files(folder, 100)
            (folder/'final_state/pn_state.npz').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing final_state/pn_state.npz'):
                verifier.require_final_files(folder, 100)

    def test_nonfinite_trace_rejection(self):
        bad = traces()
        bad['PN_gamma_nS'][37, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite array: fixture/PN_gamma_nS'):
            verifier.validate_traces(bad, 100, 44_486_000_000, 'fixture')

    def test_events_context_and_payload_exclusion(self):
        original = [dict(block=index, start_elapsed_ns=(index//2)*125000,
                         duration_ns=62500 if index % 2 == 0 else 125000, events=[])
                    for index in range(16)]
        event = dict(row=7, neuron_id=12, producer='gamma_cuda', time_s=1e-5, jump=.2, post_q=None)
        original[0]['events'] = [copy.deepcopy(event)]
        original[1]['events'] = [copy.deepcopy(event)]
        changed = copy.deepcopy(original)
        changed[0]['events'][0]['jump'] += .25
        changed[1]['events'][0]['row'] += 1
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder/'stable.json').write_text(json.dumps(original))
            (folder/'reviewed.json').write_text(json.dumps(changed))
            report = verifier.event_difference(folder/'stable.json', folder/'reviewed.json', 1)
        self.assertEqual(report['by_context']['predictor']['payload_compared_events'], 1)
        self.assertEqual(report['by_context']['predictor']['max_jump_difference'], .25)
        self.assertEqual(report['by_context']['committed']['payload_compared_events'], 0)
        self.assertEqual(report['by_context']['committed']['excluded_stable_events'], 1)
        self.assertEqual(report['by_context']['committed']['excluded_reviewed_events'], 1)

    def test_actual_diagnostic20_recorded_evidence(self):
        folder = HERE/'diagnostic_20ms_01'
        if not folder.exists():
            self.skipTest('Optional actual diagnostic artifact is absent')
        trace = verifier.load_arrays(folder/'traces.npz')
        neural = verifier.load_arrays(folder/'neural_states.npz')
        origin = int(neural['time_ns'][0])
        verifier.validate_traces(trace, 20, origin, 'actual20')
        verifier.validate_neural(neural, trace, 20, 'actual20')
        verifier.validate_initial(verifier.read_json(folder/'INITIAL.json'), verifier.PREPARED)
        verifier.require_final_files(folder, 20)
        initial = verifier.saved_tree(folder/'state_0ms/pn_state')
        final = verifier.saved_tree(folder/'final_state/pn_state')
        verifier.validate_clock_progress(initial, final, 20*verifier.NS_PER_MS,
                                         verifier.PN_CLOCKS, 'actual20')
        verifier.event_difference(folder/'EVENTS.json', folder/'EVENTS.json', 20)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(VerifierFixtures)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {'status': 'PASS' if outcome.wasSuccessful() else 'FAIL',
                'tests': outcome.testsRun, 'failures': len(outcome.failures), 'errors': len(outcome.errors),
                'skipped': len(outcome.skipped), 'verifier_sha256': verifier.sha(HERE/'verify_runs.py'),
                'test_sha256': verifier.sha(__file__), 'execution': 'CPU only; no simulations or GPU imports'}
    (Path(__file__).parent/'VERIFIER_TESTS_CPU.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps(evidence, indent=2))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
