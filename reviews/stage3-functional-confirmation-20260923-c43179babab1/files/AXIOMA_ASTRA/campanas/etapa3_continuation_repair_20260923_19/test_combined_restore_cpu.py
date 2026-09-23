"""Reuse the archived CPU failure-path suite against the actual campaign19 loader."""
from pathlib import Path
import importlib.util
import sys
import unittest

HERE = Path(__file__).resolve().parent
ORIGINAL_TEST = HERE.parents[1] / 'motor_nuevo/checkpoint_failure_fix_20260923/test_failure.py'
spec = importlib.util.spec_from_file_location('archived_failure_tests', ORIGINAL_TEST)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
prior_setup = module.FailureTests.setUpClass


@classmethod
def campaign_setup(cls):
    prior_setup()
    cls.fixed = module.load(HERE / 'restore_checkpoint.py', 'restore_campaign19')


module.FailureTests.setUpClass = campaign_setup

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(module.FailureTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
