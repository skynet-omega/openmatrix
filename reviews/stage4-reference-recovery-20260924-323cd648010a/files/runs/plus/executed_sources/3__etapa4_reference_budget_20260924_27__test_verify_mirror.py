"""Small falsification fixtures for the prospective mirrored-source decision."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


def module():
    path = Path(__file__).with_name("verify_mirror.py")
    spec = importlib.util.spec_from_file_location("stage4_verify_mirror_tested", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


V = module()


def metric(arm: str, *, input_contrast: float = 0.4,
           command_sign: float = 1.0, yaw_sign: float = 1.0):
    n = 400
    side = 1 if arm == "plus" else -1
    concentration = np.tile([0.55 + side * input_contrast / 2,
                             0.55 - side * input_contrast / 2], (n, 1))
    command = np.full(n, side * command_sign * np.deg2rad(0.05), dtype=float)
    yaw = np.linspace(0, side * yaw_sign * 0.02, n)
    return {"complete": True, "concentration": concentration,
            "command": command, "yaw": yaw,
            "signed_command_deg": float(np.rad2deg(command.sum() * .001))}


def runs(plus=None, minus=None, *, references=False):
    out = {("causal_cuda", "plus"): {"metrics": plus or metric("plus")},
           ("causal_cuda", "minus"): {"metrics": minus or metric("minus")}}
    if references:
        out.update({("reference_cuda", "plus"): {"metrics": metric("plus")},
                    ("reference_cuda", "minus"): {"metrics": metric("minus")}})
    return out


class DecisionTests(unittest.TestCase):
    def test_extra_unpaired_reader_row_rejected(self):
        trace={'fase':np.array(['preparacion']*40+['ensayo']*400),
               'DN_q_actual':np.zeros((441,4))}
        with self.assertRaisesRegex(ValueError,'Trace time length mismatch: DN_q_actual'):
            V.check_trace(trace,'plus')

    def test_wrong_reader_width_rejected(self):
        trace={'fase':np.array(['preparacion']*40+['ensayo']*400),
               'DN_q_actual':np.zeros((440,5)),
               'DN_q_usada':np.zeros((440,4)),
               'DN_baseline':np.zeros((440,4))}
        with self.assertRaisesRegex(ValueError,'Reader layout mismatch: DN_q_actual'):
            V.check_trace(trace,'plus')

    def test_geometry_rebuild(self):
        self.assertEqual(V.specs_from_geometry(), V.SPECS)
        self.assertEqual(V.source_check(Path(__file__).parent)["sources"], len(V.LOCK))

    def test_missing_pair_is_incomplete(self):
        self.assertEqual(V.decide({})["classification"], "INCOMPLETO")

    def test_native_effect_needs_references(self):
        self.assertEqual(V.decide(runs())["classification"], "PROMETEDOR_NO_CONFIRMADO")

    def test_reference_confirmed_only_scoped(self):
        decision = V.decide(runs(references=True))
        self.assertEqual(decision["rival"], "A")
        self.assertEqual(decision["classification"], "CONFIRMADO_LOCALMENTE_PENDIENTE_AUDITORIA")

    def test_consumed_input_without_command_is_b(self):
        decision = V.decide(runs(metric("plus", command_sign=0),
                                 metric("minus", command_sign=0)))
        self.assertEqual(decision["rival"], "B")
        self.assertTrue(decision["native_checks"]["input"])
        self.assertFalse(decision["native_checks"]["command_L1"])

    def test_body_only_failure_is_separately_reported(self):
        decision = V.decide(runs(metric("plus", yaw_sign=0),
                                 metric("minus", yaw_sign=0)))
        self.assertEqual(decision["rival"], "B")
        self.assertTrue(decision["native_checks"]["command_L1"])
        self.assertTrue(decision["native_checks"]["signed_command"])
        self.assertFalse(decision["native_checks"]["body_yaw"])

    def test_missing_lateral_separation_is_c(self):
        decision = V.decide(runs(metric("plus", input_contrast=0),
                                 metric("minus", input_contrast=0)))
        self.assertEqual(decision["rival"], "C")

    def test_wrong_turn_sign_rejected(self):
        decision = V.decide(runs(metric("plus", command_sign=-1, yaw_sign=-1),
                                 metric("minus", command_sign=-1, yaw_sign=-1)))
        self.assertEqual(decision["rival"], "B")

    def test_reference_parity_failure_blocks(self):
        pairs = runs(references=True)
        pairs[("reference_cuda", "plus")]["metrics"]["yaw"] += .1
        self.assertEqual(V.decide(pairs)["classification"], "BLOQUEADO")


if __name__ == "__main__":
    unittest.main()
